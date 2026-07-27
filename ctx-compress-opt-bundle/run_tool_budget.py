#!/usr/bin/env python3
"""工具结果预算剪裁 —— 独立驱动脚本。

跑 ctx-compress-opt/tool_budget.py 的两级预算，真实写盘，生成带时间戳的报告。
支持喂真实消息（--input xxx.json）或内置合成数据。

用法（在已激活 venv 的前提下）：
    python ctx-compress-opt/run_tool_budget.py                          # 合成数据，默认小阈值
    python ctx-compress-opt/run_tool_budget.py -d "单条阈值验证"
    python ctx-compress-opt/run_tool_budget.py --input qwen_replay_fixture/messages.json
    python ctx-compress-opt/run_tool_budget.py --result-size 20000 --turn-budget 60000

阈值默认故意调小（单条 2000 / 每轮 5000 / 预览 300），一跑就能看到裁剪效果。
"""

import os
import sys
import json
import argparse
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import tool_budget as tb  # noqa: E402


# =============================================================================
# 合成数据：造能同时触发两级预算的真实形状消息
# =============================================================================


def _big(marker: str, size: int) -> str:
    """造指定长度、带换行的文本，方便观察预览的换行截断效果。"""
    line = f"{marker} " + "x" * 78 + "\n"
    reps = size // len(line) + 1
    return (line * reps)[:size]


def _synthetic_messages() -> list[dict]:
    """构造覆盖各场景的消息：
      - c1: 3000 字符 > 单条阈值 2000 → 第一级单独触发
      - c2,c3,c4: 各 1800 字符，单条都没超，但合计 5400 > 每轮 5000 → 第二级触发
      - c5: read_file 结果 8000 字符，但永不转存（inf）
      - c6: 空结果 → 注入占位
    """
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": name}}
            for i, name in enumerate(
                ["bash", "grep", "grep", "grep", "read_file", "bash"], start=1
            )
        ],
    }
    tools = [
        {"role": "tool", "tool_call_id": "c1", "name": "bash", "content": _big("BASH", 3000)},
        {"role": "tool", "tool_call_id": "c2", "name": "grep", "content": _big("GREP-A", 1800)},
        {"role": "tool", "tool_call_id": "c3", "name": "grep", "content": _big("GREP-B", 1800)},
        {"role": "tool", "tool_call_id": "c4", "name": "grep", "content": _big("GREP-C", 1800)},
        {"role": "tool", "tool_call_id": "c5", "name": "read_file", "content": _big("READFILE", 8000)},
        {"role": "tool", "tool_call_id": "c6", "name": "bash", "content": "   "},
    ]
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Run some tools."},
        assistant,
        *tools,
    ]


def _load_messages(path: str) -> list[dict]:
    """从 json 文件加载消息。支持纯 list，或含 messages 字段的对象。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("messages", "history", "conversation"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"无法从 {path} 解析消息列表")


# =============================================================================
# 报告
# =============================================================================


def _preview_str(s, n=200) -> str:
    if s is None:
        return "(None)"
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    return (s[:n] + "…") if len(s) > n else s


def _format_messages(messages: list[dict]) -> str:
    lines = []
    for i, m in enumerate(messages):
        role = m.get("role")
        content = m.get("content")
        tag = " [PERSISTED]" if isinstance(content, str) and content.startswith(tb.PERSISTED_OUTPUT_TAG) else ""
        lines.append(f"--- [{i}] role={role}{tag} ---")
        if content is None:
            lines.append("(content=None)")
        elif isinstance(content, str):
            lines.append(content)
        else:
            lines.append(json.dumps(content, ensure_ascii=False, indent=2))
        if m.get("tool_calls"):
            names = [(c.get("function") or {}).get("name") for c in m["tool_calls"]]
            lines.append(f"[tool_calls: {names}]")
        lines.append("")
    return "\n".join(lines)


def _build_report(
    desc, config, input_src, before_chars, after_chars,
    before_msgs, after_msgs, stats, storage_dir, persisted_files, ts_human,
) -> str:
    sep = "=" * 70
    parts = [
        sep,
        "工具结果预算剪裁报告 (tool_budget.py)",
        sep,
        f"时间         : {ts_human}",
        f"说明(-d)     : {desc if desc else '(未提供)'}",
        f"数据来源     : {input_src}",
        f"单条阈值     : {config.default_result_size} 字符",
        f"每轮预算     : {config.turn_budget} 字符",
        f"预览大小     : {config.preview_size} 字符",
        f"永不转存工具 : {sorted(tb.PINNED_THRESHOLDS.keys())}",
        f"转存目录     : {storage_dir}",
        f"字符变化     : {before_chars} -> {after_chars}",
        f"消息条数     : {len(before_msgs)} -> {len(after_msgs)}",
        f"第二级统计   : 重放 {stats.reapplied} 条 | 新转存 {stats.newly_persisted} 条 | "
        f"去掉 ~{tb._format_size(stats.shed_chars)} | 超预算组 {stats.over_budget_groups}",
        f"转存文件     : {len(persisted_files)} 个",
        sep,
    ]
    for p in persisted_files:
        parts.append(f"  - {p}")
    parts += [
        sep,
        "",
        "#" * 70,
        "# 剪裁前 (BEFORE)",
        "#" * 70,
        "",
        _format_messages(before_msgs),
        "#" * 70,
        "# 剪裁后 (AFTER)",
        "#" * 70,
        "",
        _format_messages(after_msgs),
    ]
    return "\n".join(parts)


# =============================================================================
# main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行 tool_budget.py 的两级工具结果剪裁，真实写盘，存带时间戳的 txt 报告。"
    )
    parser.add_argument("-d", "--desc", default="", help="本次运行说明（支持中文），写入报告头部。")
    parser.add_argument("--input", default="", help="加载真实消息 json 文件；不传则用内置合成数据。")
    parser.add_argument("--result-size", type=int, default=tb.DEFAULT_RESULT_SIZE_CHARS,
                        help=f"单条结果转存阈值（字符），默认 {tb.DEFAULT_RESULT_SIZE_CHARS}。")
    parser.add_argument("--turn-budget", type=int, default=tb.DEFAULT_TURN_BUDGET_CHARS,
                        help=f"每轮聚合预算（字符），默认 {tb.DEFAULT_TURN_BUDGET_CHARS}。")
    parser.add_argument("--preview", type=int, default=tb.DEFAULT_PREVIEW_SIZE_CHARS,
                        help=f"预览大小（字符），默认 {tb.DEFAULT_PREVIEW_SIZE_CHARS}。")
    parser.add_argument("--storage-dir", default="",
                        help="转存目录，默认 ctx-compress-opt/runs/tool-results/<时间戳>/。")
    parser.add_argument("--cross-turn", action="store_true",
                        help="跨轮稳定性验证：同一批消息用同一 state 跑两遍，"
                             "验证第二遍纯重放、零新写盘、字节一致（保 prompt cache）。")
    args = parser.parse_args()

    config = tb.BudgetConfig(
        default_result_size=args.result_size,
        turn_budget=args.turn_budget,
        preview_size=args.preview,
    )

    # 加载消息
    if args.input:
        messages = _load_messages(args.input)
        input_src = args.input
    else:
        messages = _synthetic_messages()
        input_src = "内置合成数据"

    before_msgs = [dict(m) for m in messages]
    before_chars = sum(tb.content_size(m.get("content")) for m in messages)

    now = datetime.now()
    ts_file = now.strftime("%Y%m%d_%H%M%S")
    ts_human = now.strftime("%Y-%m-%d %H:%M:%S")

    storage_dir = args.storage_dir or os.path.join(_HERE, "runs", "tool-results", ts_file)

    print(f"\n=== 数据来源: {input_src} | 单条 {config.default_result_size} | "
          f"每轮 {config.turn_budget} | 预览 {config.preview_size} ===")
    print(f"=== 剪裁前: {before_chars} 字符, {len(before_msgs)} 条消息 ===\n")

    after_msgs, state, stats = tb.apply_budget(storage_dir, messages, config=config)

    after_chars = sum(tb.content_size(m.get("content")) for m in after_msgs)
    print(f"=== 剪裁后: {after_chars} 字符, {len(after_msgs)} 条消息 ===")
    print(f"=== 第二级: 重放 {stats.reapplied} | 新转存 {stats.newly_persisted} | "
          f"去掉 ~{tb._format_size(stats.shed_chars)} | 超预算组 {stats.over_budget_groups} ===\n")
    for i, m in enumerate(after_msgs):
        if m.get("role") == "tool":
            print(f"  [{i}] {m.get('tool_call_id')} -> {_preview_str(m.get('content'))}")

    # ── 跨轮稳定性验证：同一批原始消息 + 同一 state 再跑一遍 ──
    cross_turn_note = ""
    if args.cross_turn:
        # 第二遍用原始消息（模拟下一轮重新发送完整历史）+ 同一个 state
        after2, state2, stats2 = tb.apply_budget(storage_dir, messages, state=state, config=config)
        tool_contents_1 = [m.get("content") for m in after_msgs if m.get("role") == "tool"]
        tool_contents_2 = [m.get("content") for m in after2 if m.get("role") == "tool"]
        byte_identical = tool_contents_1 == tool_contents_2
        print("\n" + "=" * 70)
        print("跨轮稳定性验证（第二遍，同一 state）")
        print("=" * 70)
        print(f"  新转存   : {stats2.newly_persisted} (应为 0，纯重放)")
        print(f"  重放     : {stats2.reapplied} (应 > 0，命中缓存替换)")
        print(f"  字节一致 : {byte_identical} (前缀稳定 → 命中 prompt cache)")
        print(f"  转存目录文件数未变: {not stats2.newly_persisted}")
        print("=" * 70)
        cross_turn_note = (
            f" | 跨轮: 第二遍新转存 {stats2.newly_persisted}、重放 {stats2.reapplied}、"
            f"字节一致 {byte_identical}"
        )

    # 收集实际写出的转存文件
    persisted_files = []
    if os.path.isdir(storage_dir):
        persisted_files = sorted(
            os.path.join(storage_dir, fn)
            for fn in os.listdir(storage_dir)
            if fn.endswith(".txt")
        )

    # 保存报告
    out_dir = os.path.join(_HERE, "runs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"toolbudget_{ts_file}.txt")
    report = _build_report(
        desc=args.desc, config=config, input_src=input_src,
        before_chars=before_chars, after_chars=after_chars,
        before_msgs=before_msgs, after_msgs=after_msgs, stats=stats,
        storage_dir=storage_dir, persisted_files=persisted_files, ts_human=ts_human,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n结果已保存: {out_path}")
    if persisted_files:
        print(f"转存文件目录: {storage_dir}")


if __name__ == "__main__":
    main()
