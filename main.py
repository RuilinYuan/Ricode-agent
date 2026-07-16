"""
自主编程 Agent — CLI 入口

用法：
  python main.py "用 Python 写一个 CSV 解析器，支持多种编码，带单元测试"
  python main.py --model claude-opus-4-8 --max-rounds 30 "任务描述"
  python main.py                          # 交互式输入任务
"""
from __future__ import annotations

import argparse
import sys

from agent.loop import AgentLoop
from config import AgentConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="自主编程 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("task", nargs="?", help="编程任务描述（省略则交互输入）")
    parser.add_argument("--model", default="claude-sonnet-5", help="模型 ID")
    parser.add_argument("--max-rounds", type=int, default=50, help="最大执行轮次")
    parser.add_argument("--workspace", default=".agent_workspace", help="工作目录")
    args = parser.parse_args()

    # 任务来源：命令行参数 or 交互输入
    task: str = args.task or ""
    if not task.strip():
        try:
            task = input("请输入编程任务：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已取消")
            return 1

    if not task:
        print("错误：任务描述不能为空", file=sys.stderr)
        return 1

    config = AgentConfig(
        model=args.model,
        max_rounds=args.max_rounds,
        workspace_dir=args.workspace,
    )

    agent = AgentLoop(config)

    try:
        result = agent.run(task)
    except KeyboardInterrupt:
        print("\n\n[中断] 用户终止")
        return 1

    # ── 输出结果 ──────────────────────────────────────────────────────────────
    line = "─" * 56
    print(f"\n{line}")

    if result.success:
        print("✅  任务完成")
        if result.summary:
            print(f"\n{result.summary}")
    else:
        print(f"❌  任务未完成  原因：{result.reason}")
        if result.summary:
            print(f"\n{result.summary}")

    # ── 统计 ──────────────────────────────────────────────────────────────────
    s = result.stats
    cache_ratio = (
        f"{s.cache_read_tokens / s.input_tokens * 100:.1f}%"
        if s.input_tokens > 0
        else "N/A"
    )
    print(f"\n{line}")
    print(
        f"  执行轮次：{s.rounds}  |  "
        f"输入 tokens：{s.input_tokens:,}  |  "
        f"输出 tokens：{s.output_tokens:,}"
    )
    print(
        f"  缓存命中：{s.cache_read_tokens:,} tokens（{cache_ratio}）  |  "
        f"压缩次数：{s.compressions}"
    )
    print(line)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
