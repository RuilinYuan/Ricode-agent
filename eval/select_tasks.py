"""
从 SWE-bench Lite 筛选候选任务
──────────────────────────────
筛选条件：
  - 纯 Python 仓库（Windows 可装，无需编译 C 扩展）
  - created_at 足够新，尽量兼容 Python 3.12
输出：eval/data/candidates.jsonl（按 repo 配额 + 时间新→旧排序）
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"

# 每个仓库的配额（凑满 30；按环境可行性加权）
QUOTA = {
    "django/django": 10,     # 长链路多文件改造，测 L3
    "sympy/sympy": 7,        # 调试-修复迭代，测 L2
    "pylint-dev/pylint": 6,  # 大输出工具轰炸（lint 输出），测 L1
    "pytest-dev/pytest": 3,  # 测试框架自身，输出大
    "sphinx-doc/sphinx": 2,  # 文档构建，输出大
    "pallets/flask": 2,      # 小型调试
}

# 各仓库兼容 Python 3.12 的大致起始年份（保守估计，实际由 setup_envs.py 验证）
MIN_YEAR = {
    "django/django": 2022,
    "sympy/sympy": 2022,
    "pylint-dev/pylint": 2022,
    "pytest-dev/pytest": 2022,
    "sphinx-doc/sphinx": 2022,
    "pallets/flask": 2023,
}


def main() -> None:
    df = pd.read_parquet(DATA / "swebench_lite.parquet")
    df["year"] = pd.to_datetime(df.created_at).dt.year

    selected: list[dict] = []
    for repo, quota in QUOTA.items():
        pool = df[(df.repo == repo) & (df.year >= MIN_YEAR[repo])]
        pool = pool.sort_values("created_at", ascending=False)
        # 候选数多取 3 倍，给环境验证留出淘汰余量
        for _, row in pool.head(quota * 3).iterrows():
            selected.append({
                "instance_id": row.instance_id,
                "repo": row.repo,
                "version": row.version,
                "base_commit": row.base_commit,
                "problem_statement": row.problem_statement,
                "patch": row.patch,
                "test_patch": row.test_patch,
                "fail_to_pass": json.loads(row.FAIL_TO_PASS),
                "pass_to_pass": json.loads(row.PASS_TO_PASS),
                "created_at": row.created_at,
                "quota_group": repo,
            })

    out = DATA / "candidates.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"候选 {len(selected)} 个 → {out}")
    from collections import Counter
    print(Counter(s["repo"] for s in selected))


if __name__ == "__main__":
    main()
