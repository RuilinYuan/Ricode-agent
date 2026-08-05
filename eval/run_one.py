"""
单任务执行器（由 runner.py 以任务专属 venv 的 python 启动）
──────────────────────────────────────────────────────────
用法：
  <venv_python> eval/run_one.py --task <instance_id> --compression on|off \
      --run-dir eval/runs/<instance_id>/<compression>

行为：
  - cwd 切换到任务 workspace（agent 在仓库根目录工作）
  - PATH 前置 venv/Scripts，保证 `python -m pytest` 用对解释器
  - 以 on_event 回调把全部事件写入 events.jsonl
  - monkeypatch ApiClient.chat_completion，记录每次调用的上下文 token 估算
  - 结束后写 result.json（成功/轮次/token/压缩事件/峰值水位）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

EVAL = Path(__file__).parent
ROOT = EVAL.parent
sys.path.insert(0, str(ROOT))

from agent.context_compressor import _estimate_tokens  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--compression", choices=["on", "off"], required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-rounds", type=int, default=50)
    ap.add_argument("--window", type=int, default=0,
                    help="压缩水位基准窗口（0=用 config 默认 200k）")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── 任务定义 ──────────────────────────────────────────────────────
    task = None
    for line in (EVAL / "data" / "valid_tasks.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["instance_id"] == args.task:
            task = item
            break
    if task is None:
        print(f"任务 {args.task} 不在 valid_tasks.jsonl", file=sys.stderr)
        return 1

    ws = (EVAL / "workspaces" / args.task).resolve()
    os.chdir(ws)

    # PATH 前置 venv，保证 shell 里的 `python` / `pytest` 命中本任务环境
    venv_scripts = (EVAL / "venvs" / args.task / "Scripts").resolve()
    os.environ["PATH"] = str(venv_scripts) + os.pathsep + os.environ["PATH"]
    os.environ["COMPRESSION_ENABLED"] = "1" if args.compression == "on" else "0"
    os.environ["RAG_ENABLED"] = "0"  # 评测聚焦压缩机制，关闭 RAG 避免干扰

    from config import AgentConfig
    from agent.loop import AgentLoop
    from agent import api as api_mod

    config = AgentConfig(
        max_rounds=args.max_rounds,
        workspace_dir=str(ws),
    )
    if args.window > 0:
        config.context_window = args.window

    # ── 事件日志 ──────────────────────────────────────────────────────
    events_path = run_dir / "events.jsonl"
    events_f = events_path.open("w", encoding="utf-8")

    def log_event(ev: dict) -> None:
        ev = dict(ev)
        ev["t"] = round(time.time() - t0, 2)
        events_f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
        events_f.flush()

    # ── 上下文水位追踪（monkeypatch，不改 agent 源码）──────────────────
    water_marks: list[dict] = []
    orig_chat = api_mod.ApiClient.chat_completion

    def traced_chat(self, *, model, messages, **kwargs):
        marks = {"round_calls": len(water_marks) + 1,
                 "context_tokens": _estimate_tokens(messages),
                 "n_messages": len(messages)}
        water_marks.append(marks)
        return orig_chat(self, model=model, messages=messages, **kwargs)

    api_mod.ApiClient.chat_completion = traced_chat

    # ── 构造任务 prompt ───────────────────────────────────────────────
    prompt = (
        "你需要修复当前仓库中的一个 bug。仓库已检出到出错版本，工作目录即仓库根目录。\n\n"
        "【问题报告】\n"
        f"{task['problem_statement']}\n\n"
        "【要求】\n"
        "- 阅读相关源码，定位根因，直接修改仓库中的代码修复问题\n"
        "- 不要修改测试文件\n"
        "- 修复后用 python -m pytest 运行相关测试验证（仓库已有测试套件）\n"
        "- 确认修复后调用 task_complete，总结根因与修改内容"
    )

    t0 = time.time()
    agent = AgentLoop(config, on_event=log_event)
    try:
        result = agent.run(prompt)
    except Exception as e:  # 崩溃也要落盘，便于统计
        result = None
        log_event({"type": "crash", "error": f"{type(e).__name__}: {e}"})
    elapsed = time.time() - t0

    # ── 汇总 ──────────────────────────────────────────────────────────
    summary = {
        "instance_id": args.task,
        "compression": args.compression,
        "context_window": config.context_window,
        "elapsed_sec": round(elapsed, 1),
        "peak_context_tokens": max((m["context_tokens"] for m in water_marks), default=0),
        "final_context_tokens": water_marks[-1]["context_tokens"] if water_marks else 0,
        "api_calls": len(water_marks),
        "water_marks": water_marks,
    }
    if result is not None:
        summary.update({
            "agent_success": result.success,
            "reason": result.reason,
            "rounds": result.stats.rounds,
            "input_tokens": result.stats.input_tokens,
            "output_tokens": result.stats.output_tokens,
            "compressions": result.stats.compressions,
        })
    else:
        summary.update({"agent_success": False, "reason": "crash", "rounds": 0,
                        "input_tokens": 0, "output_tokens": 0, "compressions": 0})

    (run_dir / "result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "water_marks"},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
