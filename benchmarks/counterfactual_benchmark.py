"""
去网关化的缓存收益反事实计算
────────────────────────────
在真实任务运行中本地快照每轮发送的消息，用 tiktoken 估算 token，
按标准缓存计费模型离线计算三种口径的输入成本：

  A. 无任何缓存        : 每轮全量输入 × 1x
  B. 仅分层缓存生效     : 稳定前缀首轮写入 1.25x，后续命中 0.1x，动态尾部 1x
     （假设网关不自动缓存，但 cache_control 断点被遵守）
  C. 网关实际计费       : 用网关返回的 prompt + 0.1*cached（含网关自动缓存）

B vs A 的差值 = 分层缓存机制本身的收益（与网关无关）。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import tiktoken

from agent.loop import AgentLoop
from config import AgentConfig
from benchmarks.real_task_benchmark import BIG_TASK, make_task, seed_workspace

MODULES = 8   # main() 可通过 --modules 覆盖
BIG = False   # --big 使用大代码量任务

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(obj) -> int:
    """估算任意消息结构的 token 数（消息体 JSON 化后计数 + 每条消息 4 token 开销）。"""
    if isinstance(obj, dict):
        return 4 + count_tokens(json.dumps(obj, ensure_ascii=False))
    if isinstance(obj, list):
        return sum(count_tokens(x) for x in obj)
    if isinstance(obj, str):
        return len(_enc.encode(obj))
    return len(_enc.encode(str(obj)))


class RoundSnapshot:
    def __init__(self, prefix_tokens: int, tail_tokens: int, usage: dict):
        self.prefix_tokens = prefix_tokens   # messages[:-2] 可缓存前缀
        self.tail_tokens = tail_tokens       # messages[-2:] 动态尾部
        self.usage = usage                   # 网关实际返回的 usage


def wrap_client_with_snapshot(client, snapshots: list[RoundSnapshot]) -> None:
    """猴补丁 chat_completion：发送前快照消息 token，发送后记录 usage。"""
    original = client.chat_completion

    def wrapped(*, model, messages, tools=None, max_tokens=4096):
        # 保活线程的 ping 请求（messages 只有 2 条）不计入任务成本
        is_task_call = len(messages) >= 3
        if is_task_call:
            prefix = count_tokens(messages[:-2])
            tail = count_tokens(messages[-2:])
        resp = original(model=model, messages=messages, tools=tools, max_tokens=max_tokens)
        if is_task_call:
            snapshots.append(RoundSnapshot(prefix, tail, resp.raw.get("usage", {})))
        return resp

    client.chat_completion = wrapped


def counterfactual_cost(snapshots: list[RoundSnapshot]) -> dict:
    """三种口径的输入成本（等效计费单位）。"""
    no_cache = 0.0       # A：全量 1x
    layered = 0.0        # B：分层缓存计费模型
    gateway = 0.0        # C：网关实际计费

    prev_prefix = 0
    for i, s in enumerate(snapshots):
        full = s.prefix_tokens + s.tail_tokens
        no_cache += full

        # B：命中 = 上轮前缀（本轮前缀与其重叠的最长稳定部分，用上轮前缀近似）
        #     新增前缀部分按写入 1.25x，命中部分 0.1x，尾部 1x
        hit = min(prev_prefix, s.prefix_tokens) if i > 0 else 0
        new_prefix = s.prefix_tokens - hit
        layered += hit * 0.1 + new_prefix * (1.25 if i == 0 or new_prefix > 0 else 1.0) + s.tail_tokens
        prev_prefix = s.prefix_tokens

        # C：网关实际计费（prompt 为未命中部分，cached 按 0.1x）
        cached = (s.usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        gateway += s.usage.get("prompt_tokens", 0) + cached * 0.1

    return {
        "rounds": len(snapshots),
        "no_cache_units": round(no_cache),
        "layered_cache_units": round(layered),
        "gateway_units": round(gateway),
        "layered_saving_pct": round((1 - layered / no_cache) * 100, 1) if no_cache else 0,
        "gateway_saving_pct": round((1 - gateway / no_cache) * 100, 1) if no_cache else 0,
    }


def run_once(cache_enabled: bool, tag: str) -> dict:
    config = AgentConfig()
    config.cache_enabled = cache_enabled
    config.max_rounds = 40
    config.workspace_dir = f".agent_workspace_cf_{tag}"
    seed_workspace(config.workspace_dir, MODULES, big=BIG)

    snapshots: list[RoundSnapshot] = []

    def on_event(ev: dict) -> None:
        if ev.get("type") == "round":
            print(f"  [{tag}] round {ev['num']}", flush=True)

    loop = AgentLoop(config=config, on_event=on_event)
    wrap_client_with_snapshot(loop.client, snapshots)

    t0 = time.time()
    result = loop.run(BIG_TASK if BIG else make_task(MODULES))
    costs = counterfactual_cost(snapshots)
    costs.update({
        "tag": tag,
        "cache_enabled": cache_enabled,
        "success": result.success,
        "agent_rounds": result.stats.rounds,
        "elapsed_s": round(time.time() - t0, 1),
    })
    return costs


def main() -> None:
    global MODULES, BIG
    args = sys.argv[1:]
    if "--big" in args:
        BIG = True
    if "--modules" in args:
        MODULES = int(args[args.index("--modules") + 1])
    print(f"任务: {'大代码量(600行单文件)' if BIG else f'{MODULES} 模块'}\n")
    print("▶ 运行：分层缓存开启（快照消息，离线反事实计算）")
    on = run_once(True, "on")
    print(f"  完成: {on}")

    print("\n▶ 运行：分层缓存关闭（用于对比网关自动缓存基线）")
    off = run_once(False, "off")
    print(f"  完成: {off}")

    print("\n" + "=" * 72)
    print("去网关化反事实结果（等效计费单位，越小越省）")
    print("-" * 72)
    print(f"{'口径':<34}{'缓存关':>16}{'缓存开':>16}")
    print("-" * 72)
    print(f"{'A 无任何缓存（基线）':<34}{off['no_cache_units']:>16,}{on['no_cache_units']:>16,}")
    print(f"{'B 仅分层缓存生效（机制本身）':<34}{'-':>16}{on['layered_cache_units']:>16,}")
    print(f"{'C 网关实际计费（含自动缓存）':<34}{off['gateway_units']:>16,}{on['gateway_units']:>16,}")
    print("-" * 72)
    print(f"分层缓存机制本身收益（B vs A）: {on['layered_saving_pct']}%")
    print(f"网关自动缓存收益（C(off) vs A）: {off['gateway_saving_pct']}%")
    print(f"网关+分层叠加（C(on) vs A）  : {on['gateway_saving_pct']}%")

    out = Path("benchmarks") / f"counterfactual_{int(time.time())}.json"
    out.write_text(json.dumps({"on": on, "off": off}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n原始数据已保存: {out}")


if __name__ == "__main__":
    main()
