import os
from dataclasses import dataclass


@dataclass
class AgentConfig:
    # ── 模型 ──────────────────────────────────────────────────
    model: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
    max_tokens: int = 8096
    max_rounds: int = 50

    # ── 上下文窗口（Claude 200k）────────────────────────────────
    context_window: int = 200_000

    # ── 分层压缩水位阈值 ──────────────────────────────────────────
    context_l1_threshold: float = 0.60   # 持久化大 tool_result
    context_l2_threshold: float = 0.75   # 裁剪早期历史
    context_l3_threshold: float = 0.85   # 去除 thinking 块
    context_l4_threshold: float = 0.95   # 摘要替换旧历史

    # tool_result 超过此字符数才做 L1 持久化
    l1_persist_threshold: int = 500
    # L2 保留最近 N 条消息（user + assistant 各算 1 条）
    l2_keep_recent: int = 20

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
