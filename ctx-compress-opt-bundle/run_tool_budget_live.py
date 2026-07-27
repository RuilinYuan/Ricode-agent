#!/usr/bin/env python3
"""把 tool_budget 接进 hermes 实时跑（monkey-patch，零 hermes 代码改动）。

hermes 在工具结果产生后调用两个函数：
  - maybe_persist_tool_result  (agent/tool_executor.py:643 / 1181)  单条
  - enforce_turn_budget        (agent/tool_executor.py:679 / 1234)  每轮聚合
本脚本把这两个函数替换成转调我们的 tool_budget 模块，然后 run_oneshot 跑一个真实
任务，观察实时裁剪效果。

约束（必须遵守 hermes 契约）：
  - maybe_persist(content, tool_name, tool_use_id, env, config, threshold) -> str
  - enforce_turn_budget(tool_messages, env, config) -> list[dict]
      hermes 调用方【不用返回值、靠原地改 dict】，所以适配器必须原地突变。

用法（已激活 venv，且 .bashrc 里有 DASHSCOPE_API_KEY）：
    python ctx-compress-opt/run_tool_budget_live.py -d "live验证" --task "用bash执行: seq 1 6000"
    python ctx-compress-opt/run_tool_budget_live.py --result-size 20000 --task "..."
"""

import os
import sys
import json
import argparse
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)   # import tool_budget
sys.path.insert(0, _ROOT)   # import hermes 模块
os.chdir(_ROOT)             # hermes 期望 cwd = 仓库根

import tool_budget as tb  # noqa: E402

# ── 运行期全局（patch 闭包持有）──────────────────────────────────
_LIVE_CONFIG: tb.BudgetConfig = tb.BudgetConfig()
_LIVE_STATE: tb.ContentReplacementState = tb.create_state()
_STORAGE_DIR: str = ""
_TRIM_LOG: list[dict] = []   # 裁剪事件，跑完写报告


def _record(kind, tool_name, tool_call_id, orig_size, new_size, file_path=None):
    _TRIM_LOG.append({
        "kind": kind,
        "tool": tool_name,
        "tool_call_id": tool_call_id,
        "orig_chars": orig_size,
        "new_chars": new_size,
        "file": file_path,
    })


# ── 适配器：匹配 hermes 签名 ─────────────────────────────────────


def _patched_maybe_persist(content, tool_name, tool_use_id, env=None, config=None, threshold=None):
    """单条级。hermes 已在外层用 _is_multimodal_tool_result 守卫，非字符串双保险原样返回。"""
    if not isinstance(content, str):
        return content
    orig_size = len(content)
    msg = {"role": "tool", "tool_call_id": tool_use_id or "", "content": content, "name": tool_name}
    out = tb.maybe_persist_large_tool_result(_STORAGE_DIR, msg, tool_name, _LIVE_CONFIG, threshold)
    new_content = out.get("content", content)
    if new_content is not content and new_content != content:
        # 找出转存文件路径（从 <persisted-output> 块里提取）
        fp = None
        if isinstance(new_content, str) and "saved to:" in new_content:
            fp = new_content.split("saved to:", 1)[1].split("\n", 1)[0].strip()
        _record("L1-single", tool_name, tool_use_id, orig_size, len(new_content), fp)
    return new_content


def _patched_enforce_turn_budget(tool_messages, env=None, config=None):
    """每轮聚合级。必须原地改 dict（hermes 调用方不用返回值）。"""
    if not tool_messages:
        return tool_messages
    out, stats = tb.enforce_tool_result_budget(
        _STORAGE_DIR, tool_messages, _LIVE_STATE, _LIVE_CONFIG,
    )
    # 把结果内容原地写回 tool_messages 里的 dict（hermes 契约）
    for orig, new in zip(tool_messages, out):
        oc = orig.get("content")
        nc = new.get("content")
        if oc != nc:
            orig["content"] = nc
            cid = orig.get("tool_call_id", "")
            fp = None
            if isinstance(nc, str) and "saved to:" in nc:
                fp = nc.split("saved to:", 1)[1].split("\n", 1)[0].strip()
            _record("L2-turn", orig.get("name", ""), cid, len(oc) if isinstance(oc, str) else 0,
                    len(nc) if isinstance(nc, str) else 0, fp)
    if stats.newly_persisted or stats.reapplied:
        print(f"[tool_budget] L2: 新转存 {stats.newly_persisted} | 重放 {stats.reapplied} | "
              f"去掉 ~{tb._format_size(stats.shed_chars)}", file=sys.stderr)
    return tool_messages


def _install_patches():
    """同时 patch 模块属性 + tool_executor 已绑定的名字。"""
    import tools.tool_result_storage as trs
    import agent.tool_executor as te
    trs.maybe_persist_tool_result = _patched_maybe_persist
    te.maybe_persist_tool_result = _patched_maybe_persist
    trs.enforce_turn_budget = _patched_enforce_turn_budget
    te.enforce_turn_budget = _patched_enforce_turn_budget


# ── 报告 ──────────────────────────────────────────────────────────


def _build_report(desc, task, config, trim_log, storage_dir, ts_human) -> str:
    sep = "=" * 70
    lines = [
        sep,
        "hermes 实时工具结果裁剪报告 (tool_budget live)",
        sep,
        f"时间         : {ts_human}",
        f"说明(-d)     : {desc if desc else '(未提供)'}",
        f"任务         : {task}",
        f"单条阈值     : {config.default_result_size} 字符",
        f"每轮预算     : {config.turn_budget} 字符",
        f"预览大小     : {config.preview_size} 字符",
        f"永不转存工具 : {sorted(tb.PINNED_THRESHOLDS.keys())}",
        f"转存目录     : {storage_dir}",
        f"裁剪事件     : {len(trim_log)} 次",
        sep,
    ]
    for i, e in enumerate(trim_log):
        lines.append(
            f"  [{i}] {e['kind']} | {e['tool']}({e['tool_call_id']}) | "
            f"{e['orig_chars']} -> {e['new_chars']} 字符 | {e['file'] or '(未转存)'}"
        )
    lines += [sep, "", "注：L1-single=第一级单条转存；L2-turn=第二级每轮聚合转存。"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="把 tool_budget monkey-patch 进 hermes 实时跑。")
    parser.add_argument("-d", "--desc", default="", help="本次运行说明（支持中文）。")
    parser.add_argument("--task", default="用 bash 执行命令: for i in $(seq 1 5000); do echo \"line $i lorem ipsum dolor\"; done",
                        help="发给 hermes 的任务 prompt。")
    parser.add_argument("--model", default="", help="模型覆盖，不传用 config 默认。")
    parser.add_argument("--provider", default="", help="provider 覆盖，不传用 config 默认。")
    parser.add_argument("--toolsets", default="", help="toolsets 覆盖，逗号分隔。")
    parser.add_argument("--result-size", type=int, default=20000, help="单条阈值，默认 20000。")
    parser.add_argument("--turn-budget", type=int, default=60000, help="每轮预算，默认 60000。")
    parser.add_argument("--preview", type=int, default=tb.DEFAULT_PREVIEW_SIZE_CHARS, help="预览大小。")
    args = parser.parse_args()

    global _LIVE_CONFIG, _STORAGE_DIR
    _LIVE_CONFIG = tb.BudgetConfig(
        default_result_size=args.result_size,
        turn_budget=args.turn_budget,
        preview_size=args.preview,
    )
    now = datetime.now()
    ts_file = now.strftime("%Y%m%d_%H%M%S")
    ts_human = now.strftime("%Y-%m-%d %H:%M:%S")
    _STORAGE_DIR = os.path.join(_HERE, "runs", "tool-results-live", ts_file)
    os.makedirs(_STORAGE_DIR, exist_ok=True)

    print(f"=== tool_budget live | 单条 {args.result_size} | 每轮 {args.turn_budget} ===")
    print(f"=== 转存目录: {_STORAGE_DIR} ===")
    print(f"=== 任务: {args.task[:80]} ===\n")

    _install_patches()
    print("已安装 monkey-patch: maybe_persist_tool_result / enforce_turn_budget -> tool_budget\n")

    # 跑 hermes（run_oneshot 会重定向 stdout/stderr 到 devnull + 禁用 logging，
    # 所以裁剪事件只进 _TRIM_LOG；patched_enforce 的统计行写 stderr 也会被吞，
    # 跑完从 _TRIM_LOG 出报告）
    from hermes_cli.oneshot import run_oneshot
    rc = run_oneshot(
        args.task,
        model=args.model or None,
        provider=args.provider or None,
        toolsets=args.toolsets or None,
    )
    print(f"\n=== hermes run_oneshot 退出码: {rc} ===")
    print(f"=== 裁剪事件: {len(_TRIM_LOG)} 次 ===\n")
    for e in _TRIM_LOG:
        print(f"  {e['kind']} | {e['tool']}({e['tool_call_id']}) | "
              f"{e['orig_chars']} -> {e['new_chars']} | {e['file'] or ''}")

    # 报告
    out_dir = os.path.join(_HERE, "runs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"toolbudget_live_{ts_file}.txt")
    report = _build_report(args.desc, args.task, _LIVE_CONFIG, _TRIM_LOG, _STORAGE_DIR, ts_human)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n报告已保存: {out_path}")


if __name__ == "__main__":
    main()
