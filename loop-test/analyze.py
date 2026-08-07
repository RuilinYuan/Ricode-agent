"""
指标统计：从 results.jsonl 计算无效工具调用与减少率
无效判定：
  - 失败运行（success=False 或异常）：全部调用无效
  - 成功运行：指纹（工具名+结果前200字符 md5）重复的调用无效
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

RESULTS_FILE = Path(__file__).resolve().parent / "results.jsonl"


def ineffective_count(rec: dict) -> tuple[int, int]:
    """返回 (无效调用数, 总调用数)"""
    calls = rec.get("tool_calls", [])
    total = len(calls)
    if not rec.get("success"):
        return total, total  # 失败运行：所有调用都未促成完成
    seen: set[str] = set()
    bad = 0
    for c in calls:
        fp = hashlib.md5(
            f"{c['name']}:{c.get('output', '')[:200]}".encode()
        ).hexdigest()
        if fp in seen:
            bad += 1
        else:
            seen.add(fp)
    return bad, total


def main() -> None:
    records = [
        json.loads(line)
        for line in RESULTS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    agg: dict[str, dict[str, int]] = {}
    print(f"\n{'任务':<18}{'组别':<12}{'轮次':>4}{'调用':>6}{'无效':>6}  结果")
    print("-" * 64)
    for r in records:
        bad, total = ineffective_count(r)
        a = agg.setdefault(r["arm"], {"bad": 0, "total": 0})
        a["bad"] += bad
        a["total"] += total
        outcome = r.get("reason") or ("ok" if r.get("success") else "error")
        signals = ",".join(r.get("loop_signals", [])) or "-"
        print(f"{r['task']:<18}{r['arm']:<12}{r.get('rounds', 0):>4}"
              f"{total:>6}{bad:>6}  {outcome}  [signals: {signals}]")

    print("-" * 64)
    b, t = agg.get("baseline", {}), agg.get("treatment", {})
    print(f"\n总计  baseline : {b.get('bad', 0)}/{b.get('total', 0)} 次无效")
    print(f"总计  treatment: {t.get('bad', 0)}/{t.get('total', 0)} 次无效")
    if b.get("bad"):
        reduction = 1 - t.get("bad", 0) / b["bad"]
        print(f"\n>>> 无效工具调用减少率: {reduction:.1%}")


if __name__ == "__main__":
    main()
