"""
评测编排器
──────────
实验设计（共享对照组）：
  off       压缩关闭（窗口无关，只跑一次，作为两组实验的共同对照）
  on_w32000 压缩开启，水位窗口 32k（强制触发 L1–L4，测量机制效果）
  on_w200000 压缩开启，窗口 200k（真实配置下的自然触发率）

对 valid_tasks.jsonl 中每个任务 × 每个实验臂：
  1. git reset 工作区到 base_commit（保证起点一致）
  2. 用任务专属 venv 的 python 启动 run_one.py（子进程隔离）
  3. 结束后应用 test_patch，运行 F2P 测试判定 resolved
  4. 结果写入 eval/runs/<instance_id>/<arm>/{result,score}.json

用法：
  python runner.py                      # 全量 30 × 3 臂
  python runner.py --arms off on_w32000 # 只跑指定臂
  python runner.py --only <iid>         # 只跑某个任务
  python runner.py --jobs 4             # 并行度（默认 3）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EVAL = Path(__file__).parent
DATA = EVAL / "data"
RUNS = EVAL / "runs"

sys.path.insert(0, str(EVAL))
from setup_envs import git_apply, git_reset, run_f2p, venv_python  # noqa: E402

RUN_TIMEOUT = 3600  # 单个 run 上限 1 小时

# 实验臂 → (compression, window)
ARMS = {
    "off": ("off", 0),
    "on_w32000": ("on", 32_000),
    "on_w200000": ("on", 200_000),
}

# git reset 与工作区操作按任务串行（同一任务的不同臂可能并行跑）
_ws_locks: dict[str, threading.Lock] = {}
_ws_locks_guard = threading.Lock()


def _lock_for(iid: str) -> threading.Lock:
    with _ws_locks_guard:
        return _ws_locks.setdefault(iid, threading.Lock())


def score_run(task: dict) -> dict:
    """agent 结束后：应用 test_patch，跑 F2P 判定是否解决。
    调用方必须已持有该任务的工作区锁（run_one 全程持有）。"""
    iid = task["instance_id"]
    ws = EVAL / "workspaces" / iid
    venv = EVAL / "venvs" / iid
    if not git_apply(ws, task["test_patch"]):
        return {"resolved": False, "score_error": "test_patch 应用失败"}
    try:
        ok, out = run_f2p(venv, ws, task["repo"], task["fail_to_pass"], task["test_patch"])
    finally:
        git_reset(ws)  # 立即回滚，避免影响其他臂
    return {"resolved": ok, "f2p_output_tail": out[-300:]}


def run_one(task: dict, arm: str) -> dict:
    iid = task["instance_id"]
    compression, window = ARMS[arm]
    run_dir = RUNS / iid / arm
    result_path = run_dir / "result.json"
    score_path = run_dir / "score.json"

    if result_path.exists() and score_path.exists():
        print(f"  跳过（已完成）{iid} [{arm}]", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8")) | \
               json.loads(score_path.read_text(encoding="utf-8"))

    # 同一任务的所有臂共享一个工作区：整个 run（reset→agent→评分）持有任务锁，
    # 防止并行的另一条臂污染文件状态
    with _lock_for(iid):
        return _run_one_locked(task, arm, run_dir, result_path, score_path)


def _run_one_locked(task: dict, arm: str, run_dir: Path,
                    result_path: Path, score_path: Path) -> dict:
    iid = task["instance_id"]
    compression, window = ARMS[arm]
    ws = EVAL / "workspaces" / iid
    git_reset(ws)

    py = venv_python(EVAL / "venvs" / iid)
    cmd = [str(py), str(EVAL / "run_one.py"),
           "--task", iid, "--compression", compression,
           "--run-dir", str(run_dir),
           "--window", str(window)]
    print(f"  启动 {iid} [{arm}]", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=RUN_TIMEOUT, cwd=str(EVAL.parent))
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "stdout.log").write_text(proc.stdout[-20000:], encoding="utf-8")
        (run_dir / "stderr.log").write_text(proc.stderr[-20000:], encoding="utf-8")
    except subprocess.TimeoutExpired:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "stderr.log").write_text("RUN_TIMEOUT", encoding="utf-8")

    if not result_path.exists():
        result = {"instance_id": iid, "compression": compression, "arm": arm,
                  "context_window": window or 200_000,
                  "agent_success": False, "reason": "runner_crash",
                  "rounds": 0, "input_tokens": 0, "output_tokens": 0,
                  "compressions": 0, "peak_context_tokens": 0,
                  "final_context_tokens": 0, "api_calls": 0,
                  "elapsed_sec": round(time.time() - t0, 1)}
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    score = score_run(task)
    score_path.write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")
    merged = json.loads(result_path.read_text(encoding="utf-8")) | score
    merged["arm"] = arm
    print(f"  完成 {iid} [{arm}] resolved={score['resolved']} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只跑指定 instance_id")
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    choices=list(ARMS), help="只跑指定实验臂")
    ap.add_argument("--jobs", type=int, default=3, help="并行 run 数")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in (DATA / "valid_tasks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.only:
        tasks = [t for t in tasks if t["instance_id"] == args.only]

    # 臂优先排序：让并行的 worker 拿到不同任务（同任务各臂互斥，避免锁等待空转）
    work = [(t, arm) for arm in args.arms for t in tasks]
    print(f"共 {len(work)} 个 run（{len(tasks)} 任务 × {len(args.arms)} 臂），并行 {args.jobs}", flush=True)

    all_results: list[dict] = []
    results_path = DATA / "all_results.json"
    if results_path.exists():
        all_results = json.loads(results_path.read_text(encoding="utf-8"))

    lock = threading.Lock()

    def save() -> None:
        with lock:
            results_path.write_text(
                json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(run_one, t, arm): (t["instance_id"], arm) for t, arm in work}
        for fut in as_completed(futs):
            iid, arm = futs[fut]
            try:
                r = fut.result()
                with lock:
                    all_results = [x for x in all_results
                                   if not (x.get("instance_id") == iid and x.get("arm", x.get("compression")) == arm)]
                    all_results.append(r)
            except Exception as e:
                print(f"  [runner 异常] {iid} [{arm}]: {e}", flush=True)
            save()
    print(f"\n全部完成：{len(work)} 个 run", flush=True)


if __name__ == "__main__":
    main()
