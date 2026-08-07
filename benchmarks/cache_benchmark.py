"""
缓存分层 vs 无缓存：长会话输入 Token 成本对比基准
────────────────────────────────────────────────
方法：固定脚本回放（replay）20 轮对话，两种模式跑完全相同的请求序列，
唯一差异是是否注入 cache_control 断点。每轮用 max_tokens=1 压低输出成本，
只测量输入侧 usage。

  模式 A (off) : 原始 messages，无任何 cache_control
  模式 B (on)  : CacheManager.get_system_message + add_cache_breakpoint

判定依据：从响应 usage 中提取缓存字段（兼容 Anthropic / DeepSeek 两种命名）：
  cache_write = cache_creation_input_tokens | prompt_cache_miss_tokens 估算
  cache_read  = cache_read_input_tokens     | prompt_cache_hit_tokens

若代理不透传缓存字段（cache_read 恒为 0），说明缓存未生效，
此时对比 prompt_tokens 总量即可，结论即为"节省 0%"。

计费模型（Anthropic 官方价）：写入 1.25x，读取 0.1x，普通输入 1x。
若 usage 只给 prompt_tokens（OpenAI 风格，已扣减命中部分），
直接比较 prompt_tokens 总量。

用法：
  python -m benchmarks.cache_benchmark            # 快速模式（轮间不停顿）
  python -m benchmarks.cache_benchmark --slow     # 轮间 sleep，触发 TTL 过期+保活
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.api import ApiClient, ApiError
from agent.cache_manager import CacheManager
from agent.loop import _SYSTEM_PROMPT
from config import AgentConfig

ROUNDS = 20

# 模拟每轮的工具结果：固定内容，两种模式完全一致
_FAKE_TOOL_RESULT = (
    "执行输出：\n" + "\n".join(
        f"line {i}: result value = {i * 7 % 13}, status ok" for i in range(40)
    )
)


@dataclass
class RoundUsage:
    round: int
    prompt_tokens: int
    completion_tokens: int
    cache_read: int
    cache_write: int
    raw_usage: dict


@dataclass
class ModeResult:
    name: str
    rounds: list[RoundUsage] = field(default_factory=list)

    @property
    def total_prompt(self) -> int:
        return sum(r.prompt_tokens for r in self.rounds)

    @property
    def total_cache_read(self) -> int:
        return sum(r.cache_read for r in self.rounds)

    @property
    def total_cache_write(self) -> int:
        return sum(r.cache_write for r in self.rounds)

    @property
    def billed_units(self) -> float:
        """
        等效计费单位。两种口径：
        - 有缓存字段：未命中部分 1x + 写入 1.25x + 读取 0.1x
          （Anthropic 风格下 prompt_tokens 通常不含 cache_read，视代理而定，
            这里按 cache_read/cache_write 独立于 prompt_tokens 处理；
            若代理把 cache_read 计入 prompt_tokens，则 prompt_tokens 口径会更小，
            两种模式同口径对比，结论不受影响。）
        - 无缓存字段：prompt_tokens 1x
        """
        if self.total_cache_read == 0 and self.total_cache_write == 0:
            return float(self.total_prompt)
        return (
            self.total_prompt * 1.0
            + self.total_cache_write * 1.25
            + self.total_cache_read * 0.1
        )


def extract_cache_fields(usage: dict) -> tuple[int, int]:
    """从 usage 提取 (cache_read, cache_write)，兼容多种命名。"""
    read = usage.get("cache_read_input_tokens", 0)
    write = usage.get("cache_creation_input_tokens", 0)
    # DeepSeek 风格
    read = read or usage.get("prompt_cache_hit_tokens", 0)
    # OpenAI 风格 details
    details = usage.get("prompt_tokens_details") or {}
    read = read or details.get("cached_tokens", 0)
    return int(read), int(write)


def build_round_messages(messages: list[dict], round_num: int) -> None:
    """原地追加一轮固定的 assistant tool_call + tool result（两种模式一致）。"""
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": f"call_{round_num:03d}",
            "type": "function",
            "function": {
                "name": "run_command",
                "arguments": json.dumps({"command": f"echo step {round_num}"}),
            },
        }],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": f"call_{round_num:03d}",
        "content": _FAKE_TOOL_RESULT + f"\n(round={round_num})",
    })


def run_mode(
    name: str,
    use_cache: bool,
    client: ApiClient,
    config: AgentConfig,
    round_delay: float,
) -> ModeResult:
    cache_mgr = CacheManager(config)
    task = "基准任务：模拟 20 轮编码循环（固定回放内容）"

    if use_cache:
        system_msg = cache_mgr.get_system_message(
            stable_text=_SYSTEM_PROMPT,
            state_text=f"当前任务：{task}",
        )
    else:
        # 无缓存模式：纯字符串 system，不分层
        system_msg = {"role": "system", "content": _SYSTEM_PROMPT + f"\n\n当前任务：{task}"}

    messages: list[dict] = [system_msg, {"role": "user", "content": task}]

    if use_cache:
        cache_mgr.start_warmup(client=client, model=config.model, stable_system_msg=system_msg)

    result = ModeResult(name)
    try:
        for rnd in range(1, ROUNDS + 1):
            outgoing = cache_mgr.add_cache_breakpoint(messages) if use_cache else messages
            t0 = time.time()
            resp = client.chat_completion(
                model=config.model,
                messages=outgoing,
                max_tokens=1,   # 只测输入成本
            )
            elapsed = time.time() - t0
            read, write = extract_cache_fields(resp.raw.get("usage", {}))
            result.rounds.append(RoundUsage(
                round=rnd,
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                cache_read=read,
                cache_write=write,
                raw_usage=resp.raw.get("usage", {}),
            ))
            print(f"  [{name}] round {rnd:2d}: prompt={resp.usage.prompt_tokens:6d} "
                  f"cache_read={read:6d} cache_write={write:6d} ({elapsed:.1f}s)")

            build_round_messages(messages, rnd)
            if round_delay > 0 and rnd < ROUNDS:
                time.sleep(round_delay)
    finally:
        if use_cache:
            cache_mgr.stop_warmup()
    return result


def report(a: ModeResult, b: ModeResult) -> None:
    print("\n" + "=" * 64)
    print(f"{'指标':<24}{'无缓存 (A)':>16}{'分层缓存 (B)':>16}")
    print("-" * 64)
    print(f"{'总 prompt_tokens':<24}{a.total_prompt:>16,}{b.total_prompt:>16,}")
    print(f"{'总 cache_read':<24}{a.total_cache_read:>16,}{b.total_cache_read:>16,}")
    print(f"{'总 cache_write':<24}{a.total_cache_write:>16,}{b.total_cache_write:>16,}")
    print(f"{'等效计费单位':<24}{a.billed_units:>16,.0f}{b.billed_units:>16,.0f}")
    print("-" * 64)
    if a.billed_units > 0:
        saving = (1 - b.billed_units / a.billed_units) * 100
        print(f"输入成本变化：{saving:+.1f}%（正值=节省）")
    if b.total_cache_read == 0:
        print("\n⚠ 未观察到任何 cache_read：当前代理很可能不透传/不支持缓存断点，")
        print("  即缓存机制在你的链路上没有生效，真实节省为 0。")
        print("  首轮 raw usage 样例：")
        print("  ", json.dumps(b.rounds[0].raw_usage, ensure_ascii=False) if b.rounds else "  (无数据)")


def main() -> None:
    global ROUNDS
    # Windows GBK 控制台兼容
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--slow", action="store_true",
                        help="轮间 sleep 320s，测试 TTL 过期与保活续期（耗时约 1.8 小时）")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    args = parser.parse_args()
    ROUNDS = args.rounds

    config = AgentConfig()
    client = ApiClient(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        base_url=os.environ.get("ANTHROPIC_BASE_URL", "").strip(),
    )

    delay = 320.0 if args.slow else 0.0
    print(f"模型: {config.model}  轮数: {ROUNDS}  轮间延迟: {delay}s\n")

    print("▶ 模式 A：无缓存")
    a = run_mode("A:off", False, client, config, delay)
    print("\n▶ 模式 B：分层缓存 + 保活")
    b = run_mode("B:on", True, client, config, delay)

    report(a, b)

    out = Path("benchmarks") / f"cache_benchmark_{int(time.time())}.json"
    out.write_text(json.dumps({
        "model": config.model,
        "rounds": ROUNDS,
        "mode_off": [r.__dict__ for r in a.rounds],
        "mode_on": [r.__dict__ for r in b.rounds],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n原始数据已保存: {out}")


if __name__ == "__main__":
    main()
