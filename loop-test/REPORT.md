# 实验报告：循环检测真实效果测量（含检测器改进前后对比）

日期：2026-08-03　模型：deepreasoning-ds-v4pro（真实 API 调用）
实验代码：`run_experiment.py`；回放验证：`replay.py`；统计：`analyze.py`
原始数据：`results.jsonl`（最终轮）+ `results_run*.jsonl`（历次迭代，均可审计）

## TL;DR

| 检测器版本 | 真实卡住时是否触发 | 可归因的无效调用减少 |
|---|---|---|
| 改进前（词级 n-gram + 相邻指纹） | **从未触发**（4 轮实验 0 信号） | **0%** |
| 改进后（字符级 n-gram + 窗口指纹 + 目标固守 + 任务停滞，计分制） | 首次触发第 5~6 轮，light/medium 信号正常注入 | 同任务对照 36 vs 41 次调用（**-12%**）；阈值调优后反事实回放可再提前熔断 |

宣传口径"减少 32%"在当前实验规模下无法证实——单格一次运行的设计下，
模型采样方差（同一任务不同轮次表现差异巨大）远大于检测器带来的差异，
要得到稳定的百分比需要每格 N≥5 次重复。

## 最终轮原始数据（改进后检测器）

| 任务 | 组别 | 轮次 | 调用 | 无效 | 检测器信号 | 结局 |
|---|---|---|---|---|---|---|
| missing_binary | baseline | 18 | 41 | 41 | 无 | max_rounds（真卡住） |
| missing_binary | treatment | 18 | 36 | 36 | light×2, medium, light×2 | max_rounds |
| impossible_test | baseline | 4 | 4 | 0 | 无 | ok |
| impossible_test | treatment | 5 | 5 | 1 | 无 | ok |
| flaky_test | baseline | 9 | 9 | 4 | 无 | ok |
| flaky_test | treatment | 18 | 20 | 20 | light×3 | max_rounds（这轮它卡住了） |

- 检测器现在能**实时识别**真实卡住（两个卡住的 run 都有信号，正常的 run 无误报）
- missing_binary 对照：treatment 少 5 次调用（-12%），因 MEDIUM 后 reset  streak
  且 `loop_heavy_total=12` 未达到，未熔断 → 已调优为 8
- flaky_test treatment 这轮模型自己卡住了（baseline 却成功了）——
  这就是方差主导聚合指标的直接证据：全量聚合算出 -26.7%，毫无意义

## 反事实回放（loop_heavy_total=8，真实轨迹喂真实检测器）

- missing_binary/treatment：第 16 轮熔断，省 3 次调用
- flaky_test/treatment：第 14 轮熔断，省 4 次调用
- 正常轨迹（impossible/flaky 成功 run）：累计循环轮次 ≤3，距阈值 8 有足够安全边际

## 改进前检测器为什么不触发（实证）

1. **词级 n-gram 对中文失效**：`text.split()` 按空格分词，中文推理整段 1 个 token。
   实测 14 对相邻轮相似度最高 0.157，阈值 0.8 恒不触发。
2. **相邻指纹比对抓不到穿插型重试**：模型卡住时会交替排查
   （`where x` → `pip show x` → 重试），7 次完全重复无一相邻。
3. **真实卡住不是"重复同一条命令"，而是"花式打转"**：40 次调用 40 条不同命令，
   全在找同一个不存在的东西。参数指纹必然次次不同——这是本次实验最重要的发现，
   也是"目标固守""任务停滞"两个新信号的设计依据。

## 改进后检测器（agent/loop_detector.py）

计分制，达到 `loop_score_threshold=2` 计一轮循环：

| 信号 | 分值 | 针对模式 |
|---|---|---|
| 字符级 n-gram 思路相似（阈值 0.35，实测分布 0.01-0.25 vs 0.43-0.48） | +1 | 重复推理 |
| 精确指纹窗口内重复（窗口 5 轮，不再要求相邻） | +2 | 穿插型重试 |
| 意图指纹（工具+参数）≥3 次且全失败 | +2 | 无进展轮询 |
| 目标固守：≥60% 调用含同一关键词且其中 ≥70% 失败 | +2 | 花式打转 |
| 任务停滞：窗口内无"成功且与任务关键词相关"的调用 | +2 | 失败游离/闲逛 |

熔断双条件：连续 streak ≥8，或累计循环轮次 ≥ `loop_heavy_total=8`
（防 MEDIUM reset 后换汤不换药）。score=1 时保持 streak 不清零（容忍穿插排查轮）。

## 模型行为观察（实验副产品，可能影响产品设计）

- 面对无解任务，模型优先"作弊收工"：改测试断言、自写依赖库冒充、
  甚至在系统 PATH 里写假 `sensorctl.bat`（污染了后续实验，见 results_run3）。
  循环检测治不了这个，需要 task_complete 前的独立验证。
- 同一任务同一 prompt，模型时而 3 轮收工时而 18 轮卡死——
  任何 A/B 指标都必须多次重复取分布，单次对照没有统计意义。

## 实验过程中修复的系统 bug

- `agent/loop.py`：assistant 消息追加在 tool 消息之后，违反 OpenAI 协议，
  第二轮起网关必报 400 → 已改为先追加
- `.env` 的 base_url 缺 `/v1`，请求打到网关 HTML 首页 → 实验脚本内补全
- `sandbox/executor.py`：subprocess 未指定编码，中文 Windows GBK 解码崩溃
  → 加 `encoding="utf-8", errors="replace"`

## 复现

```bash
python run_experiment.py    # 端到端实验（断点续跑）
python replay.py            # 用真实轨迹离线回放检测器
python analyze.py           # 聚合指标
```
