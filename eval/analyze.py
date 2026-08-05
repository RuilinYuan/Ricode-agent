"""
结果汇总分析（三臂设计：off / on_w32000 / on_w200000，off 为共享对照）
────────────────────────────────────────────────────────────────────────
读取 eval/data/all_results.json，输出 eval/data/analysis.md：
  - 实验一（机制效果）：on_w32000 vs off —— 峰值上下文降低 %（验证 46%）、轮次倍数（验证 2x）
  - 实验二（真实配置）：on_w200000 vs off —— 200k 窗口下的自然触发率与效果
  - 任务解决率对比（压缩不应显著降低成功率）
  - 压缩触发分布、分仓库对照、逐任务明细
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

EVAL = Path(__file__).parent
DATA = EVAL / "data"
ARMS = ["off", "on_w32000", "on_w200000"]
mean, med = statistics.mean, statistics.median


def pct_drop(base: float, new: float) -> str:
    return "N/A" if base <= 0 else f"{(base - new) / base * 100:.1f}%"


def ratio(new: float, base: float) -> str:
    return "N/A" if base <= 0 else f"{new / base:.2f}x"


def arm_of(r: dict) -> str:
    return r.get("arm") or r.get("compression", "off")


def comparison_block(title: str, paired: dict, on_arm: str, repo_of: dict) -> list[str]:
    n = len(paired)
    peaks_off = [v["off"]["peak_context_tokens"] for v in paired.values()]
    peaks_on = [v[on_arm]["peak_context_tokens"] for v in paired.values()]
    rounds_off = [v["off"]["rounds"] for v in paired.values()]
    rounds_on = [v[on_arm]["rounds"] for v in paired.values()]
    res_off = sum(1 for v in paired.values() if v["off"].get("resolved"))
    res_on = sum(1 for v in paired.values() if v[on_arm].get("resolved"))
    comp = sum(v[on_arm].get("compressions", 0) for v in paired.values())
    triggered = sum(1 for v in paired.values() if v[on_arm].get("compressions", 0) > 0)

    lines = [
        f"## {title}\n",
        f"配对任务数：{n}；压缩触发过的任务：{triggered}（共 {comp} 次压缩）\n",
        "| 指标 | OFF | ON | 对比 |",
        "|---|---|---|---|",
        f"| 峰值上下文 token（均值） | {mean(peaks_off):,.0f} | {mean(peaks_on):,.0f} | 降低 {pct_drop(mean(peaks_off), mean(peaks_on))} |",
        f"| 峰值上下文 token（中位） | {med(peaks_off):,.0f} | {med(peaks_on):,.0f} | 降低 {pct_drop(med(peaks_off), med(peaks_on))} |",
        f"| 峰值上下文 token（最大） | {max(peaks_off):,} | {max(peaks_on):,} | 降低 {pct_drop(max(peaks_off), max(peaks_on))} |",
        f"| 运行轮次（均值） | {mean(rounds_off):.1f} | {mean(rounds_on):.1f} | {ratio(mean(rounds_on), mean(rounds_off))} |",
        f"| 运行轮次（中位） | {med(rounds_off):.1f} | {med(rounds_on):.1f} | {ratio(med(rounds_on), med(rounds_off))} |",
        f"| 任务解决数 | {res_off}/{n} | {res_on}/{n} | — |",
        "",
        "### 失败原因分布\n",
    ]
    for arm in ("off", on_arm):
        reasons = Counter(v[arm].get("reason", "") for v in paired.values())
        lines.append(f"- {arm}: {dict(reasons)}")
    lines.append("")

    # 压缩触发层级分布
    comp_events: Counter = Counter()
    for v in paired.values():
        ev_path = EVAL / "runs" / v[on_arm]["instance_id"] / on_arm / "events.jsonl"
        if ev_path.exists():
            for line in ev_path.read_text(encoding="utf-8").splitlines():
                ev = json.loads(line)
                if ev.get("type") == "compress":
                    comp_events[f"L{ev['level']}"] += 1
    lines.append(f"### 压缩触发层级分布\n\n{dict(comp_events)}\n")

    # 分仓库
    lines.append("### 分仓库对照\n")
    lines.append("| 仓库 | 任务数 | 峰值OFF均值 | 峰值ON均值 | 降低 | 轮次OFF | 轮次ON | 倍数 | 解决OFF | 解决ON |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    by_repo: dict[str, list] = defaultdict(list)
    for iid, v in paired.items():
        by_repo[repo_of.get(iid, "?")].append(v)
    for repo, vs in sorted(by_repo.items()):
        po = mean([v["off"]["peak_context_tokens"] for v in vs])
        pn = mean([v[on_arm]["peak_context_tokens"] for v in vs])
        ro = mean([v["off"]["rounds"] for v in vs])
        rn = mean([v[on_arm]["rounds"] for v in vs])
        so = sum(1 for v in vs if v["off"].get("resolved"))
        sn = sum(1 for v in vs if v[on_arm].get("resolved"))
        lines.append(f"| {repo} | {len(vs)} | {po:,.0f} | {pn:,.0f} | {pct_drop(po, pn)} "
                     f"| {ro:.1f} | {rn:.1f} | {ratio(rn, ro)} | {so} | {sn} |")
    lines.append("")
    return lines


def main() -> None:
    results = json.loads((DATA / "all_results.json").read_text(encoding="utf-8"))
    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_task[r["instance_id"]][arm_of(r)] = r

    repo_of = {}
    for line in (DATA / "valid_tasks.jsonl").read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        repo_of[t["instance_id"]] = t["repo"]

    lines = ["# 分层上下文压缩评测结果\n"]
    lines.append(f"任务总数：{len(by_task)}（SWE-bench Lite 子集）\n")

    for on_arm, title in [
        ("on_w32000", "实验一：32k 水位窗口（强制触发压缩，测机制效果）"),
        ("on_w200000", "实验二：200k 真实窗口（自然触发）"),
    ]:
        paired = {k: v for k, v in by_task.items()
                  if "off" in v and on_arm in v}
        if not paired:
            continue
        lines += comparison_block(title, paired, on_arm, repo_of)

        # 逐任务明细
        lines.append(f"### 逐任务明细（{on_arm}）\n")
        lines.append("| instance | repo | 峰值OFF | 峰值ON | 轮次OFF | 轮次ON | resolved OFF/ON | 压缩次数 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for iid, v in sorted(paired.items()):
            o, n2 = v["off"], v[on_arm]
            lines.append(
                f"| {iid} | {repo_of.get(iid,'?')} | {o['peak_context_tokens']:,} | {n2['peak_context_tokens']:,} "
                f"| {o['rounds']} | {n2['rounds']} | {'Y' if o.get('resolved') else 'n'}/{'Y' if n2.get('resolved') else 'n'} "
                f"| {n2.get('compressions', 0)} |")
        lines.append("")

    out = DATA / "analysis.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\n完整报告 → {out}")


if __name__ == "__main__":
    main()
