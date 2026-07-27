"""按时间触发的微压缩（Time-Based Microcompact）—— hermes 包内版。

复现 Claude Code 的「按时间间隔触发的微压缩」：距上一次模型回复过久（默认
60 分钟）时，推断服务端提示缓存几乎肯定已过期 —— 这次请求反正要把整个前缀
重算一遍。既然要重算，就趁请求发出之前，先把又大又旧的工具结果内容原地清空
成一句占位串，让真正要重算的前缀更小。

它和「工具结果预算剪裁」（tool_budget.py）的分工：
  - 预算剪裁：按“大小”触发，转存到磁盘 + 换预览，并小心保住缓存。
  - 本机制：按“时间间隔”触发，前提就是缓存已凉，所以直接把旧内容原地清空成
    占位串，不转存、不留预览、也不在乎破坏缓存。

【触发时间 / 作用范围 / 默认状态】
  - 触发时间：把消息发给模型“之前”执行，命中即返回新消息列表。
  - 作用范围：所有“可压缩工具”产生的 role="tool" 消息内容，除了最近的 keep_recent
    个（默认 5），其余全部清空成占位串；最近的原样保留，给模型留近期工作上下文。
  - 默认开启（enabled=True），阈值 5 分钟 —— 对齐当前 provider 的提示缓存 TTL
    （5 分钟过期）。gap 超过 TTL 就认定缓存已凉，前缀反正要重算，趁机清掉旧结果。

【产品化说明】这是 ctx-compress-opt 里 micro_compact_time_openai.py 原型的包内版本，
消息格式与 tool_budget.py 一致 —— OpenAI Chat Completions 的 dict 形状：
  assistant: {"role":"assistant","tool_calls":[{"id":..,"function":{"name":..}}]}
  tool:      {"role":"tool","tool_call_id":..,"content":..,"name":..}

相对原型的适配差异：
  1. 用 dict 形状消息（对齐 tool_budget.py），不再用 dataclass ChatMessage。
  2. COMPACTABLE_TOOLS 换成 hermes 的真实工具名。
  3. 与 tool_budget.py 协同：认出被预算剪裁转存过的块（以 PERSISTED_OUTPUT_TAG
     开头），清理时保留其磁盘文件路径，模型日后仍能按路径把全文读回。
  4. content 若不是 str（多模态 content-part list / dict），一律原样放行，
     绝不清空 —— 保护 vision 工具结果结构不被破坏。
  5. 上一条 assistant 的时间戳 hermes 消息里通常没有，改由调用方显式传入
     last_activity_ms（如 agent 记录的上次 API 完成时间）；消息带 timestamp
     字段时也支持从中解析。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# 与预算剪裁共用同一个转存标签，保证两套机制对“已转存块”的判断完全一致。
from tools.tool_budget import PERSISTED_OUTPUT_TAG


# =============================================================================
# 一、占位串与可压缩工具集合
# =============================================================================

# 旧工具结果被清空后替换成的固定文本。不是摘要、不是落盘路径，就是一句提示。
TIME_BASED_MC_CLEARED_MESSAGE = "[Old tool result content cleared]"

# 两种清空占位串的公共前缀（带路径 / 不带路径都以它开头），用于判断“是否已清理过”。
_CLEARED_PREFIX = "[Old tool result content cleared"

# 对“已被预算剪裁转存过”的块，清理时保留文件路径的占位串前缀。
_CLEARED_WITH_PATH_PREFIX = "[Old tool result content cleared; full output saved to: "

# tool_budget.build_persisted_message 写完整结果路径时用的格式：
#   read_file(path="<path>", full_lines=true)
# 据此从预览块里把文件路径抠出来。
_PERSISTED_PATH_MARKER = 'read_file(path="'

# 只有这些工具产生的结果才会被压缩：结果可能很大、且清掉不影响后续正确性
# （文件内容、命令输出、搜索结果等，模型需要时可重新读 / 重新跑）。
# 工具名对齐 hermes 的注册名（见 toolsets.py 的 _HERMES_CORE_TOOLS）。
COMPACTABLE_TOOLS: set[str] = {
    "read_file",     # 读文件
    "terminal",      # 终端命令
    "process",       # 进程管理
    "search_files",  # 文件搜索（grep / glob 合一）
    "web_search",    # 联网搜索
    "web_extract",   # 网页抓取
    "write_file",    # 写文件
    "patch",         # 补丁 / 编辑
}


def _extract_persisted_path(content: str) -> Optional[str]:
    """从一个被预算剪裁转存过的块里抠出磁盘文件路径。

    tool_budget.build_persisted_message 会写一行：
        read_file(path="<path>", full_lines=true)
    定位 marker、取其后到下一个双引号之间的内容。抠不到返回 None。
    """
    idx = content.find(_PERSISTED_PATH_MARKER)
    if idx == -1:
        return None
    start = idx + len(_PERSISTED_PATH_MARKER)
    end = content.find('"', start)
    return content[start:end] if end != -1 else None


def _cleared_content_for(content: Optional[str]) -> str:
    """决定一个被清理的工具结果换成什么占位串。

      - 若它已是预算剪裁的转存预览（以 PERSISTED_OUTPUT_TAG 开头）→ 保留文件路径
        的占位串（只丢预览正文，路径留着，模型仍可再读回全文）。
      - 否则 → 普通占位串。
    """
    if isinstance(content, str) and content.startswith(PERSISTED_OUTPUT_TAG):
        path = _extract_persisted_path(content)
        if path:
            return f"{_CLEARED_WITH_PATH_PREFIX}{path}]"
    return TIME_BASED_MC_CLEARED_MESSAGE


# =============================================================================
# 二、配置
# =============================================================================


@dataclass(frozen=True)
class TimeBasedMCConfig:
    """按时间触发的微压缩配置。

      enabled                总开关。False 时整个机制什么都不做。默认开启。
      gap_threshold_minutes  触发阈值（分钟）。当 (现在 - 上次活动时间) 超过它，
                             就认为服务端缓存已过期。默认 5 分钟 —— 对齐当前
                             provider 的提示缓存 TTL（缓存 5 分钟过期）。
      keep_recent            保留最近多少个可压缩工具结果；更早的全部清空。
    """

    enabled: bool = True
    gap_threshold_minutes: float = 5
    keep_recent: int = 5


# 默认配置：默认关闭。
DEFAULT_TIME_BASED_MC_CONFIG = TimeBasedMCConfig()


# =============================================================================
# 三、统计
# =============================================================================


@dataclass
class MicrocompactStats:
    """时间微压缩执行统计，供报告使用。"""

    triggered: bool = False       # 本轮是否真的触发并清理
    gap_minutes: float = 0.0      # 距上次活动的间隔（分钟）
    cleared: int = 0              # 被清空的工具结果条数
    kept_recent: int = 0          # 保留的最近条数
    tokens_saved: int = 0         # 粗估省下的 token 数


# =============================================================================
# 四、token 估算辅助
# =============================================================================


def _rough_token_count(content: str, bytes_per_token: int = 4) -> int:
    """粗略 token 估算：字符数 / 4，四舍五入。"""
    return round(len(content) / bytes_per_token)


def _tool_result_tokens(content) -> int:
    """估算一条 tool 消息内容的 token 数；非字符串或空则为 0。"""
    if isinstance(content, str) and content:
        return _rough_token_count(content)
    return 0


# =============================================================================
# 五、收集可压缩工具结果 id
# =============================================================================


def collect_compactable_tool_ids(messages: list[dict]) -> list[str]:
    """按出现顺序遍历所有 assistant 消息的 tool_calls，收集那些“工具名在
    COMPACTABLE_TOOLS 里”的调用 id。返回顺序 == 工具被调用的先后顺序，
    后面据此判断“最近的 N 个”。
    """
    ids: list[str] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for call in m.get("tool_calls") or []:
            name = (call.get("function") or {}).get("name", "")
            cid = call.get("id")
            if cid and name in COMPACTABLE_TOOLS:
                ids.append(cid)
    return ids


# =============================================================================
# 六、时间戳解析 + 上次活动时间
# =============================================================================


def _parse_timestamp_ms(timestamp: Optional[str]) -> Optional[float]:
    """把 ISO 时间字符串解析成毫秒时间戳；解析不了返回 None。"""
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp).timestamp() * 1000
    except (ValueError, TypeError):
        return None


def _last_activity_ms(
    messages: list[dict], last_activity_ms: Optional[float]
) -> Optional[float]:
    """推断“上次模型活动时间”（毫秒）。

    优先用调用方显式传入的 last_activity_ms（hermes 消息通常不带时间戳，agent
    会记录上次 API 完成时间）；否则回退到最后一条带 timestamp 的 assistant 消息。
    两者都没有则返回 None（无法判断间隔 → 不触发）。
    """
    if last_activity_ms is not None:
        return last_activity_ms
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return _parse_timestamp_ms(m.get("timestamp"))
    return None


# =============================================================================
# 七、判断该不该触发
# =============================================================================


def evaluate_time_based_trigger(
    messages: list[dict],
    now_ms: float,
    config: TimeBasedMCConfig,
    last_activity_ms: Optional[float] = None,
) -> Optional[float]:
    """判断这次请求该不该触发。全满足才触发：

      1) 配置 enabled 为真
      2) 能确定“上次活动时间”
      3) 间隔 = (现在 - 上次活动) / 60000 分钟，有限且 >= 阈值

    触发返回 gap_minutes；否则返回 None。now_ms 显式传入以便可复现。
    """
    if not config.enabled:
        return None

    last_ms = _last_activity_ms(messages, last_activity_ms)
    if last_ms is None:
        return None

    gap_minutes = (now_ms - last_ms) / 60_000
    if not math.isfinite(gap_minutes) or gap_minutes < config.gap_threshold_minutes:
        return None
    return gap_minutes


# =============================================================================
# 八、主体：真正执行清理
# =============================================================================


def maybe_time_based_microcompact(
    messages: list[dict],
    now_ms: float,
    config: Optional[TimeBasedMCConfig] = None,
    last_activity_ms: Optional[float] = None,
) -> tuple[list[dict], MicrocompactStats]:
    """间隔超阈值时，把除最近 N 个之外的可压缩工具结果内容清空成占位串。

    返回 (处理后的消息列表, 统计)。未触发 / 无可清理 / 一个 token 都没省时，
    原样返回输入消息列表（stats.triggered=False）。

    直接改发出副本的消息内容 —— 缓存是冷的，没有热前缀需要保护。不就地改原对象，
    产出新 dict（canonical 历史不受影响，与 tool_budget.py 同一契约）。
    """
    config = config or DEFAULT_TIME_BASED_MC_CONFIG
    stats = MicrocompactStats()

    gap_minutes = evaluate_time_based_trigger(
        messages, now_ms, config, last_activity_ms
    )
    if gap_minutes is None:
        return messages, stats
    stats.gap_minutes = gap_minutes

    compactable_ids = collect_compactable_tool_ids(messages)

    # keep_recent 下限取 1：全清会让模型没有近期工作上下文，不合理；至少留最近 1 个。
    keep_recent = max(1, config.keep_recent)
    keep_set = set(compactable_ids[-keep_recent:])  # 末尾 keep_recent 个 = 最近的
    clear_set = {cid for cid in compactable_ids if cid not in keep_set}

    if not clear_set:
        return messages, stats

    tokens_saved = 0
    result: list[dict] = []
    for m in messages:
        content = m.get("content")
        # 已清过就别重复：用前缀判断，能同时覆盖普通占位串和“带路径”占位串，
        # 避免带路径的块在下一轮被再清一次（那会把路径也丢掉）。
        already_cleared = (
            isinstance(content, str) and content.startswith(_CLEARED_PREFIX)
        )
        # 多模态守卫：非字符串 content 一律原样放行，绝不清空。
        if (
            m.get("role") == "tool"
            and m.get("tool_call_id") in clear_set
            and isinstance(content, str)
            and not already_cleared
        ):
            tokens_saved += _tool_result_tokens(content)
            result.append({**m, "content": _cleared_content_for(content)})
        else:
            result.append(m)

    if tokens_saved == 0:
        return messages, stats

    stats.triggered = True
    stats.cleared = len(clear_set)
    stats.kept_recent = len(keep_set)
    stats.tokens_saved = tokens_saved
    return result, stats


# =============================================================================
# 九、便捷入口
# =============================================================================


def apply_time_based_microcompact(
    messages: list[dict],
    now_ms: float,
    config: Optional[TimeBasedMCConfig] = None,
    last_activity_ms: Optional[float] = None,
) -> tuple[list[dict], MicrocompactStats]:
    """时间微压缩入口。命中即返回清理后的消息列表 + 统计；没命中原样返回。

    典型用法（conversation_loop 发送前，作用于 api_messages 副本）：
        api_messages, stats = apply_time_based_microcompact(
            api_messages, now_ms=time.time() * 1000,
            config=TimeBasedMCConfig(enabled=True),
            last_activity_ms=agent._last_api_finished_ms,
        )
    """
    return maybe_time_based_microcompact(messages, now_ms, config, last_activity_ms)
