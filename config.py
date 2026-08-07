import os
from dataclasses import dataclass
from pathlib import Path


# 自动加载 .env（所有入口统一生效：CLI / Streamlit / FastAPI）
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())


@dataclass
class AgentConfig:
    # ── 模型 ──────────────────────────────────────────────────
    model: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
    max_tokens: int = 8096
    max_rounds: int = 50

    # ── 上下文窗口（Claude 200k）────────────────────────────────
    context_window: int = 200_000

    # ── 分层压缩水位阈值（按成本从低到高累积执行）──────────────────
    context_l1_threshold: float = 0.60   # L1：删除 thinking / reasoning（近零损失，最先做）
    context_l2_threshold: float = 0.70   # L2：工具结果持久化（低损失，内容在磁盘可恢复）
    context_l3_threshold: float = 0.80   # L3：激进本地清理 5 步递进（中损失，不调 LLM）
    context_l4_threshold: float = 0.90   # L4：逐轮 LLM 摘要（最贵手段，最后上）

    # L1：单条 tool_result 超过此字符数才做持久化
    # 已弃用 — L2 改为"降序逐个外置直到水位回落"，不再需要此阈值
    l1_persist_threshold: int = 0
    # L1：同批工具结果合计 token 上限（二级预算）
    # 已弃用 — 同上
    l1_batch_budget: int = 0

    # L3：每轮摘要的最大输出 token 数
    l3_summary_max_tokens: int = 600

    # L4：兜底时保护最近 N 条工具结果不清理
    l4_keep_recent_tools: int = 3

    # 分层压缩总开关（评测 A/B 对比用，COMPRESSION_ENABLED=0 关闭）
    compression_enabled: bool = os.environ.get("COMPRESSION_ENABLED", "1") == "1"

    # ── 循环检测阈值 ──────────────────────────────────────────────
    loop_light_rounds: int = 3     # 告警 + 注入提示
    loop_medium_rounds: int = 5    # 强制换策略
    loop_heavy_rounds: int = 8     # 熔断
    loop_ngram_n: int = 5          # 字符级 n-gram（中文无空格，按词切分会失效）
    loop_ngram_threshold: float = 0.35  # 实测分布：正常 0.01-0.25 / 循环 0.43-0.48
    # 指纹滑动窗口：最近一轮调用在窗口内出现过即算重复（覆盖穿插排查型重试）
    loop_window: int = 5
    # 意图指纹（工具+参数，不看结果）窗口内最少重复次数，且结果全部失败才算无进展
    loop_intent_min_repeats: int = 3
    # 计分制触发线：思路重复+1 / 窗口内完全重复+2 / 意图重试全败+2 / 目标固守+2
    loop_score_threshold: int = 2
    # 累计循环轮次熔断兜底（MEDIUM 会 reset 连续计数，靠累计值防止换汤不换药）
    # 实测：正常轨迹累计 ≤3，真实卡住轨迹 ≥10，阈值取中间
    loop_heavy_total: int = 8

    # ── 缓存刷新 ──────────────────────────────────────────────────
    # 分层缓存总开关：关闭后不打 cache_control 断点、不启动保活
    cache_enabled: bool = os.environ.get("CACHE_ENABLED", "1") == "1"
    # Anthropic 缓存 TTL 5 分钟，提前 5 秒刷新
    cache_warmup_interval: int = 295

    # ── API 重试 ──────────────────────────────────────────────────
    max_retries: int = 3           # 遇 502/503/超时 的最大重试次数
    retry_base_delay: float = 1.0  # 指数退避基础延迟（秒）
    retry_max_delay: float = 60.0  # 单次延迟上限（秒）

    # ── 沙箱执行 ──────────────────────────────────────────────────
    exec_timeout: int = 30         # 单次执行超时（秒），同时控制 execute_python / run_command
    workspace_dir: str = ".agent_workspace"

    # ── 日志 ──────────────────────────────────────────────────────
    log_level: str = "INFO"        # DEBUG / INFO / WARNING / ERROR
    log_file: str = ""             # 留空 = 仅控制台；填路径则同时写文件

    # ── 沙箱后端：local（subprocess） / docker（容器隔离）─────────────
    sandbox_backend: str = os.environ.get("SANDBOX_BACKEND", "local")
    docker_image: str = "python:3.11-slim"
    docker_memory: str = "512m"    # 容器内存上限
    docker_cpus: float = 1.0       # 容器 CPU 核数上限
    docker_network: str = "none"   # none = 完全断网

    # ── RAG 代码检索 ────────────────────────────────────────────────
    rag_enabled: bool = os.environ.get("RAG_ENABLED", "1") == "1"
    rag_top_k: int = 5             # search_code 默认返回条数
    rag_chunk_lines: int = 60      # 代码分块行数
    rag_chunk_overlap: int = 10    # 分块重叠行数
    embedding_base_url: str = os.environ.get("EMBEDDING_BASE_URL", "")
    embedding_api_key: str = os.environ.get("EMBEDDING_API_KEY", "")
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    # ── 自动按窗口比例计算（可手动覆盖）───────────────────────────────

    def __post_init__(self) -> None:
        """dataclass 初始化后钩子，暂无自动计算项。"""
        pass
