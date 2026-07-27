# ctx-compress-opt-bundle

上下文压缩优化子系统（从 hermes-agent 项目中抽取的独立可运行版本）。

本目录把 `hermes-agent/ctx-compress-opt/` 中构成「上下文压缩」实现逻辑的源码打包在一起，可在不依赖 hermes-agent 其余代码的情况下独立运行与阅读。

---

## 1. 这是什么

一个多阶段上下文压缩管线，用于在对话历史超过模型上下文窗口时，把旧轮次压回阈值内、尽量保留语义，使后续模型可以续接任务。

三个阶段（定义在 `compress.py` 的 `_compress_context()` 中）：

| 阶段 | 模块 | 做什么 | 是否调 LLM |
|------|------|--------|-----------|
| A | `tool_budget.py` | 超大工具结果整块转存磁盘，content 换成预览+文件路径 | 否 |
| B | `micro_compact.py` | 掐头去尾 / 清旧 tool result / 瘦身参数 / 删旧 reasoning / 占位兜底 | 否 |
| C | `compress.py` | 按对话轮次逐轮调 LLM 生成结构化摘要，仍超限则单轮兜底 | 是 |

> 当前 `compress.py` 里**阶段 B 被整段注释**（见 `_compress_context` 中 `阶段 B` 块），实际走 A → C。如需启用 B，按文件内注释取消注释即可。

---

## 2. 文件清单与角色

| 文件 | 角色 |
|------|------|
| `compress.py` | **核心**：压缩主流程（阶段 A/B/C + 按轮 LLM 摘要 + 单轮兜底）。含 token 估算、轮次切分、序列化、脱敏等全部 helper。 |
| `micro_compact.py` | **实现依赖**：阶段 B 本地微压缩（去重 / 工具结果信息化摘要 / 清 reasoning / JSON 感知缩减参数 / 占位兜底）。 |
| `tool_budget.py` | **实现依赖**：阶段 A 超大工具结果转存磁盘 + 预览替换（两级预算：单条阈值 + 批量合计）。 |
| `run_compress.py` | **入口驱动**：注入桩模块、解析凭证、构造真实 LLM provider，跑通压缩并把「压缩前 + 压缩后」落盘到 `runs/`。 |
| `run_tool_budget.py` | `tool_budget` 单独驱动。 |
| `run_tool_budget_live.py` | `tool_budget` 实数据驱动。 |
| `compare_wire.py` | 压缩前后对比工具。 |
| `micro_compact_time.py` | `micro_compact` 耗时基准。 |
| `mc_cache_test.py` | `micro_compact` 缓存测试。 |
| `.plan-micro-compact.md` | 阶段 B 设计文档。 |

---

## 3. 依赖关系（为什么只有这几个文件）

`compress.py` 顶部有三个 import：

```python
from configs.config import LOG_LEVEL, RUN_LOG_FILE, ROOT_PATH
from utils.log_utils import get_logger
from agent.redact import redact_sensitive_text   # 在 _summary_redact() 内部
```

这三个在 hermes-agent 项目里**并非真实可导入的包**（项目根有同名 `utils.py` 但不是包；`configs` / `agent.redact` 同理）。运行时通过两种方式解决，**因此无需打包**：

1. `configs.config` 与 `utils.log_utils`：由 `run_compress.py` 的 `_install_stub_modules()` 用 `types.ModuleType` 打桩注入（日志走标准 `logging`，ROOT_PATH 指向当前目录）。
2. `agent.redact`：`compress.py` 的 `_summary_redact()` 自带 try/except，导入失败则退化到轻量正则兜底（GitHub token / Bearer / sk- 前缀），最差原样返回，不中断摘要。

真正构成压缩实现逻辑的就是同目录 3 个文件：`compress.py`（主干）+ `micro_compact.py`（阶段 B）+ `tool_budget.py`（阶段 A）。

---

## 4. 运行方式

### 前置
- Python 3.10+
- `pip install tiktoken`（token 估算用；不可用时 compress.py 自动退化到字符估算）
- 一个 DashScope / 兼容 OpenAI 协议的 LLM key（阶段 C 摘要调用）

### 跑通压缩（用 run_compress.py 入口）

```powershell
cd C:\Users\Administrator\Desktop\ctx-compress-opt-bundle

# 凭证二选一：
$env:DASHSCOPE_API_KEY="sk-..."        # 方式1：环境变量
# 或：把凭证放进 ~/.hermes/auth.json 的 credential_pool.alibaba（方式2）

# 默认阈值触发按轮摘要
python run_compress.py

# 调大阈值，验证摘要质量（不立刻触发压缩）
python run_compress.py -d "调大阈值验证摘要质量"

# 指定模型 / 阈值
$env:COMPRESS_MODEL="qwen-plus"
$env:COMPRESS_MAX_TOKENS=2000
python run_compress.py
```

每次运行会在 `runs/` 下新建一个带时间戳的 txt，把「压缩前 + 压缩后」写在同一个文件里，`-d` 传入的说明写在文件头部。

> `run_compress.py` 是自包含的：它自己注入桩模块，所以放在桌面这个目录里、脱离 hermes-agent 也能跑。

### 直接 import compress.py
若想在自己的脚本里调用 `_compress_context()`，需自行注入 `configs.config` / `utils.log_utils` 两个桩（参考 `run_compress.py` 的 `_install_stub_modules()`），再 `import compress`。

---

## 5. 摘要提示词

阶段 C 的摘要提示词定义在 `compress.py` 的 `_summarize_round()` 中，输出按以下结构化字段：

```
用户请求
已执行操作
关键工具结果
结论与决策
相关文件或状态
未解决事项
```

要求要点：优先保留未完成请求、只写实际执行的操作、区分已验证 vs 未验证内容、凭证替换为 `[REDACTED]`、超预算时按 5 级优先级截断。摘要预算由 `_SUMMARY_MAX_TOKENS`（默认 1500）控制。

---

## 6. 与原项目的差异

本 bundle 从 `hermes-agent/ctx-compress-opt/` 抽取，**已剔除**：

- `runs/`、`tool-budget-cache/`、`compressed/`、`__pycache__/` —— 运行产物，每次跑会重新生成；
- `_probe_wsl_net.py`、`_probe_proxy.py` —— 与压缩无关的网络 / 代理探针。

其余源码与原项目一致，未做改动。
