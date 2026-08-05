# 分层上下文压缩评测（SWE-bench Lite 子集）

验证目标：分层压缩机制（L1–L4）在 30 个真实编程任务上
  1. 峰值上下文占用是否显著降低（参考声明：-46%）
  2. 单会话连续运行轮次是否提升（参考声明：2x）
  3. 任务解决率是否不受压缩影响（底线）

## 目录

```
eval/
├── select_tasks.py      # 从 SWE-bench Lite 筛候选（52 个，6 仓库配额）
├── setup_envs.py        # 下载源码 / 建 venv / 用 gold patch 双向验证环境
├── run_one.py           # 单任务执行器（子进程，任务专属 venv）
├── runner.py            # 编排 30 任务 × 2 配置，评分（F2P 测试）
├── analyze.py           # 汇总 A/B 对比，输出 analysis.md
├── data/                # 数据集、任务清单、报告
├── repos/               # 源码 zip 缓存
├── workspaces/          # 每任务一份仓库工作区（git 管理，可 reset）
├── venvs/               # 每任务一个独立 Python 环境
└── runs/                # 每 (任务, 配置) 的 events.jsonl / result.json / score.json
```

## 任务集

SWE-bench Lite（300 个真实 GitHub issue）中筛选 Windows + Python 3.12
可复现的 30 个，按仓库配额对应四类上下文压力：

| 仓库 | 数量 | 主要压力 | 对应压缩层 |
|---|---|---|---|
| django/django | 10 | 长链路多文件，历史轮次堆积 | L3 摘要 |
| sympy/sympy | 7 | 调试-修复迭代，reasoning 占比高 | L2 轨迹清理 |
| pylint-dev/pylint | 6 | lint/测试大输出 | L1 结果外置 |
| pytest-dev/pytest | 3 | 测试套件大输出 | L1 |
| sphinx-doc/sphinx | 2 | 文档构建大输出 | L1/L4 |
| pallets/flask | 2 | 小型调试 | L2 |

每个任务都经过双向环境验证：打 gold patch 后 F2P 全过、不打则有失败。

## 方法

- **三臂设计**（`COMPRESSION_ENABLED=1/0` 开关在 config.py，loop.py 中 gate）：
  - `off`：压缩关闭（与窗口无关，只跑一次，作为共享对照）
  - `on_w32000`：压缩开启，水位窗口缩至 32k —— 冒烟测试显示真实任务峰值约
    24k/200k（12%），不缩窗压缩几乎不触发；缩窗可强制 L1–L4 参与，测机制效果
  - `on_w200000`：压缩开启，真实 200k 窗口，观察自然触发率
- **上下文水位**：monkeypatch `ApiClient.chat_completion`，每次调用前用
  tiktoken 估算 messages token 数，记录峰值/最终值。
- **判定**：agent 结束后应用 `test_patch`，跑 SWE-bench 的 FAIL_TO_PASS
  测试，全过 = resolved。
- **隔离**：每任务独立工作区 + 独立 venv；agent 子进程以该 venv 的
  python 启动，PATH 前置 venv/Scripts。

## 运行

```bash
python select_tasks.py     # 一次性
python setup_envs.py       # 一次性（慢：下载+建 venv+双向验证，可断点续跑）
python runner.py --jobs 3  # 全量 30 任务 × 3 臂（数小时，API 成本）
python analyze.py          # 生成 data/analysis.md
```
