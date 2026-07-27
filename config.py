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

    # ── 分层压缩水位阈值 ──────────────────────────────────────────
    context_l1_threshold: float = 0.60   # L1：持久化大 tool_result（两级预算）
    context_l2_threshold: float = 0.75   # L2：删除 thinking / reasoning（近零损失）
    context_l3_threshold: float = 0.85   # L3：逐轮 LLM 摘要（按需停手）
    context_l4_threshold: float = 0.95   # L4：单轮溢出兜底（5步递进）

    # L1：单条 tool_result 超过此字符数才做持久化
    l1_persist_threshold: int = 500
    # L1：同批工具结果合计 token 上限（二级预算）
    l1_batch_budget: int = 10_000

    # L3：每轮摘要的最大输出 token 数
    l3_summary_max_tokens: int = 600

    # L4：兜底时保护最近 N 条工具结果不清理
    l4_keep_recent_tools: int = 3

    # ── 循环检测阈值 ──────────────────────────────────────────────
    loop_light_rounds: int = 3     # 告警 + 注入提示
    loop_medium_rounds: int = 5    # 强制换策略
    loop_heavy_rounds: int = 8     # 熔断
    loop_ngram_n: int = 3          # trigram
    loop_ngram_threshold: float = 0.8

    # ── 缓存刷新 ──────────────────────────────────────────────────
    # Anthropic 缓存 TTL 5 分钟，提前 5 秒刷新
    cache_warmup_interval: int = 295

    # ── 沙箱执行 ──────────────────────────────────────────────────
    exec_timeout: int = 30         # 单次执行超时（秒）
    workspace_dir: str = ".agent_workspace"

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
