#!/usr/bin/env python3
"""
比较两个 wire.json 的 token 估算。

估算方法：字符数 / 4（通用粗估，适合中英混合内容）。
分角色统计：system / user / assistant / tool，并细分 tool 结果里是否已被压缩。
"""

import json, sys, os, re
from pathlib import Path

CHARS_PER_TOKEN = 4  # 粗估系数

WIRE_DIR = Path(__file__).parent.parent / "hermes-chat-logs"
FILES = ["04-wire.json", "05-wire.json"]

COMPRESSED_PREFIXES = (
    "[TOOL_RESULT_TRUNCATED]",
    "[Old tool result content cleared",
)


def est(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def content_chars(content) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):          # 多模态
        return sum(len(p.get("text","")) for p in content if isinstance(p,dict))
    return 0


def is_compressed(content) -> bool:
    if not isinstance(content, str):
        return False
    return content.startswith(COMPRESSED_PREFIXES)


def analyze(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    msgs = data.get("messages", [])
    model = data.get("model", "?")
    captured_at = data.get("captured_at", "?")

    stats = {
        "model": model,
        "captured_at": captured_at,
        "total_msgs": len(msgs),
        "roles": {},          # role -> {count, chars, tokens}
        "tool_detail": {
            "total": 0,
            "compressed": 0,
            "compressed_chars": 0,
            "full_chars": 0,
        },
        "total_chars": 0,
        "total_tokens": 0,
    }

    for m in msgs:
        role = m.get("role", "unknown")
        content = m.get("content", "")

        # assistant 可能有 tool_calls（没有 content）
        extra_chars = 0
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                extra_chars += len(fn.get("name","")) + len(fn.get("arguments",""))

        chars = content_chars(content) + extra_chars

        r = stats["roles"].setdefault(role, {"count":0,"chars":0,"tokens":0})
        r["count"] += 1
        r["chars"] += chars
        r["tokens"] += est(content) if isinstance(content,str) else chars//CHARS_PER_TOKEN

        stats["total_chars"] += chars
        stats["total_tokens"] += chars // CHARS_PER_TOKEN

        # tool 细分
        if role == "tool":
            stats["tool_detail"]["total"] += 1
            c = m.get("content","")
            cc = content_chars(c)
            if is_compressed(c):
                stats["tool_detail"]["compressed"] += 1
                stats["tool_detail"]["compressed_chars"] += cc
            else:
                stats["tool_detail"]["full_chars"] += cc

    return stats


def fmt(n: int) -> str:
    return f"{n:,}"


def print_report(label: str, s: dict):
    print(f"\n{'='*56}")
    print(f"  {label}")
    print(f"  model={s['model']}  captured={s['captured_at']}")
    print(f"{'='*56}")
    print(f"  消息总数: {s['total_msgs']}")
    print(f"  总字符数: {fmt(s['total_chars'])}  估算 tokens: {fmt(s['total_tokens'])}")
    print()
    print(f"  {'角色':<12} {'消息数':>6} {'字符数':>10} {'估算tokens':>12}")
    print(f"  {'-'*44}")
    for role in ("system","user","assistant","tool"):
        r = s["roles"].get(role)
        if not r:
            continue
        print(f"  {role:<12} {r['count']:>6} {fmt(r['chars']):>10} {fmt(r['tokens']):>12}")
    print()
    td = s["tool_detail"]
    print(f"  tool 结果细分：共 {td['total']} 条")
    print(f"    已压缩（占位串）: {td['compressed']} 条  {fmt(td['compressed_chars'])} 字符")
    print(f"    未压缩（全文）  : {td['total']-td['compressed']} 条  {fmt(td['full_chars'])} 字符")


def main():
    results = {}
    for fn in FILES:
        p = WIRE_DIR / fn
        if not p.exists():
            print(f"文件不存在: {p}")
            sys.exit(1)
        results[fn] = analyze(p)
        print_report(fn, results[fn])

    # 差值对比
    a, b = [results[f] for f in FILES]
    print(f"\n{'='*56}")
    print(f"  对比：{FILES[1]}  vs  {FILES[0]}")
    print(f"{'='*56}")
    delta_chars  = b["total_chars"]  - a["total_chars"]
    delta_tokens = b["total_tokens"] - a["total_tokens"]
    print(f"  总字符差值  : {delta_chars:+,}")
    print(f"  估算token差 : {delta_tokens:+,}")
    print()
    for role in ("system","user","assistant","tool"):
        ra = a["roles"].get(role, {"chars":0,"tokens":0})
        rb = b["roles"].get(role, {"chars":0,"tokens":0})
        dc = rb["chars"] - ra["chars"]
        dt = rb["tokens"] - ra["tokens"]
        if dc:
            print(f"  {role:<12} chars {dc:+,}  tokens {dt:+,}")
    td_a, td_b = a["tool_detail"], b["tool_detail"]
    print()
    print(f"  tool 压缩数变化: {td_a['compressed']} → {td_b['compressed']}  "
          f"({td_b['compressed']-td_a['compressed']:+d})")
    print(f"  tool 全文chars : {fmt(td_a['full_chars'])} → {fmt(td_b['full_chars'])}  "
          f"({td_b['full_chars']-td_a['full_chars']:+,})")
    print()


if __name__ == "__main__":
    main()
