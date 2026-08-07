"""
回放验证：把 results.jsonl 中已记录的真实运行轨迹，
逐轮喂给真实的 LoopDetector，观察检测器在第几轮触发。

用途：
  1. 循环轨迹（如 missing_binary baseline）应尽早触发 —— 验证灵敏度
  2. 正常轨迹不应触发 —— 验证不误报

用法：
  python replay.py                    # 回放 results.jsonl 全部运行
  python replay.py results_run1.jsonl # 回放指定文件
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.loop_detector import LoopDetector, LoopSignal  # noqa: E402
from config import AgentConfig  # noqa: E402
from run_experiment import TASKS  # noqa: E402

_TASK_DESC = {t["id"]: t["desc"] for t in TASKS}

TEST_DIR = Path(__file__).resolve().parent


def replay(rec: dict) -> dict:
    """用真实 LoopDetector 回放一条运行记录，返回逐轮信号与首次熔断轮次。"""
    det = LoopDetector(AgentConfig())
    det.set_task(_TASK_DESC.get(rec["task"], ""))

    # 按轮次重组：reasoning 文本 + 该轮工具调用
    rounds: dict[int, dict] = {}
    for r in rec.get("reasoning", []):
        if isinstance(r, dict):
            rounds.setdefault(r["round"], {})["reasoning"] = r["text"]
        # 旧格式（纯字符串列表）无法精确对齐轮次，跳过该条
    for c in rec.get("tool_calls", []):
        rnd = c.get("round")
        if rnd is None:
            continue
        rounds.setdefault(rnd, {}).setdefault("calls", []).append(c)

    if not rounds or not any("calls" in v for v in rounds.values()):
        return {"error": "无轮次标注（旧格式数据），无法回放"}

    per_round = []
    first_trigger = None
    heavy_round = None
    for rnd in sorted(rounds):
        data = rounds[rnd]
        calls = data.get("calls", [])
        det.record(
            round_num=rnd,
            reasoning=data.get("reasoning", ""),
            tool_calls=[{"name": c["name"], "input": c.get("input", {})} for c in calls],
            tool_results=[{"content": c.get("output", ""),
                           "success": c.get("success", True)} for c in calls],
        )
        signal = det.check()
        per_round.append({"round": rnd, "score": det.last_score,
                          "signal": signal.value,
                          "count": det.consecutive_loop_count})
        if signal != LoopSignal.NONE and first_trigger is None:
            first_trigger = rnd
        if signal == LoopSignal.HEAVY and heavy_round is None:
            heavy_round = rnd
        # 模拟主循环：MEDIUM 后 reset
        if signal == LoopSignal.MEDIUM:
            det.reset()

    # 反事实：若在 heavy_round 熔断，节省多少后续调用
    saved = 0
    if heavy_round:
        saved = sum(1 for c in rec["tool_calls"] if c.get("round", 0) > heavy_round)

    return {
        "rounds": len(rounds),
        "first_trigger": first_trigger,
        "heavy_round": heavy_round,
        "calls_saved_if_fused": saved,
        "per_round": per_round,
    }


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else TEST_DIR / "results.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        res = replay(rec)
        print(f"\n=== {rec['task']} / {rec['arm']} ===")
        if "error" in res:
            print(" ", res["error"])
            continue
        print(f"  轮次={res['rounds']} 首次触发=第{res['first_trigger']}轮 "
              f"熔断=第{res['heavy_round']}轮 熔断可省调用={res['calls_saved_if_fused']}")
        for p in res["per_round"]:
            bar = "+" * p["score"]
            print(f"  round {p['round']:>2}: score={p['score']} {bar:<4} "
                  f"signal={p['signal']:<6} streak={p['count']}")


if __name__ == "__main__":
    main()
