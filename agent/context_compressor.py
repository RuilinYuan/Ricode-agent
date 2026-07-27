"""
分层上下文压缩（改进版）
──────────────────────
使用 tiktoken 精确估算 Token 水位，按阈值逐级触发压缩，
信息损失从低到高：

  L1 (60%) 工具结果持久化：两级预算，整块存磁盘，context 保留预览引用
  L2 (75%) 删除 thinking / reasoning：近零损失，先于历史裁剪
  L3 (85%) 逐轮 LLM 摘要：按需停手，6 字段结构化，按轮次逐步压缩
  L4 (95%) 单轮溢出兜底：5 步递进，处理当前轮本身超限的极端情况
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


# ── Token 估算 ─────────────────────────────────────────────────────────────────

def _estimate_tokens(messages: list[dict]) -> int:
    """使用 tiktoken 精确估算消息列表的 token 数，失败时退化到字符估算。"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        return _estimate_tokens_fallback(messages)

    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(enc.encode(content))
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    total += len(enc.encode(block.get("text", "")))
                elif block.get("type") == "thinking":
                    total += len(enc.encode(block.get("thinking", "")))

        # DeepSeek / o1 系列的推理字段
        rc = msg.get("reasoning_content")
        if rc:
            total += len(enc.encode(str(rc)))

        # 工具调用结构
        if msg.get("tool_calls"):
            total += len(enc.encode(str(msg["tool_calls"])))

        total += 4  # 每条消息的角色/分隔符开销
    return total


def _estimate_tokens_fallback(messages: list[dict]) -> int:
    """tiktoken 不可用时的字符估算（2.5 字符 ≈ 1 token）。"""
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total_chars += len(block.get("text", ""))
                    total_chars += len(block.get("thinking", ""))
        rc = msg.get("reasoning_content")
        if rc:
            total_chars += len(str(rc))
        if msg.get("tool_calls"):
            total_chars += len(str(msg["tool_calls"]))
    return int(total_chars / 2.5)


# ── 数据类 ─────────────────────────────────────────────────────────────────────

@dataclass
class CompressionResult:
    messages: list[dict]
    level_applied: int    # 0=未压缩，1-4=对应级别
    tokens_saved: int     # 估算节省的 token 数


# ── 主压缩器 ───────────────────────────────────────────────────────────────────

class ContextCompressor:
    def __init__(self, config: Any, client: Any = None) -> None:
        self.config = config
        self.client = client  # ApiClient 实例，L3 摘要 / L4 兜底需要

        self._work_dir = Path(config.workspace_dir)
        self._work_dir.mkdir(exist_ok=True)
        self._l1_store: dict[str, str] = {}  # md5 key → 磁盘文件路径

    # ── 公开接口 ───────────────────────────────────────────────────────────────

    def check_and_compress(self, messages: list[dict]) -> CompressionResult:
        """
        内部用 tiktoken 估算水位，按需逐级压缩。
        返回（可能已修改的）消息列表与压缩元数据。
        """
        tokens = _estimate_tokens(messages)
        ratio = tokens / self.config.context_window

        if ratio >= self.config.context_l4_threshold:
            return self._compress_l4(messages, tokens)
        if ratio >= self.config.context_l3_threshold:
            return self._compress_l3(messages, tokens)
        if ratio >= self.config.context_l2_threshold:
            return self._compress_l2(messages)
        if ratio >= self.config.context_l1_threshold:
            return self._compress_l1(messages)
        return CompressionResult(messages, 0, 0)

    # ── L1：工具结果持久化（两级预算）──────────────────────────────────────────

    def _compress_l1(self, messages: list[dict]) -> CompressionResult:
        """
        两级预算：
          一级：单条 tool 消息超 l1_persist_threshold 字符 → 整块写磁盘，保留预览引用
          二级：同批工具结果合计 token 超 l1_batch_budget → 从大到小继续转存，直到回落
        """
        result = list(messages)
        tokens_saved = 0

        # ── 一级：单条超阈值 ───────────────────────────────────────────────────
        for i, msg in enumerate(result):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            if len(content) <= self.config.l1_persist_threshold:
                continue

            filepath = self._persist(content)
            preview = (
                content[:200]
                + f"\n...[内容已存至 {filepath.name}，共 {len(content)} 字符，"
                f"可用 read_file 工具读取]"
            )
            result[i] = {**msg, "content": preview}
            tokens_saved += (len(content) - len(preview)) // 4

        # ── 二级：批量合计超预算 ───────────────────────────────────────────────
        tool_indices = [i for i, m in enumerate(result) if m.get("role") == "tool"]
        batch_tokens = sum(_estimate_tokens([result[i]]) for i in tool_indices)

        if batch_tokens > self.config.l1_batch_budget:
            # 按内容长度降序，优先转存最大的
            sorted_indices = sorted(
                tool_indices,
                key=lambda i: len(result[i].get("content", "")),
                reverse=True,
            )
            for i in sorted_indices:
                if batch_tokens <= self.config.l1_batch_budget:
                    break
                content = result[i].get("content", "")
                if not isinstance(content, str) or "已存至" in content:
                    continue  # 已经是预览，跳过
                before = _estimate_tokens([result[i]])
                filepath = self._persist(content)
                preview = (
                    content[:200]
                    + f"\n...[批量预算超限，已存至 {filepath.name}]"
                )
                result[i] = {**result[i], "content": preview}
                after = _estimate_tokens([result[i]])
                saved = before - after
                batch_tokens -= saved
                tokens_saved += saved

        return CompressionResult(result, 1, tokens_saved)

    def _persist(self, content: str) -> Path:
        """将内容写入工作目录，返回文件路径。"""
        key = hashlib.md5(content.encode()).hexdigest()[:8]
        filepath = self._work_dir / f"tr_{key}.txt"
        filepath.write_text(content, encoding="utf-8")
        self._l1_store[key] = str(filepath)
        return filepath

    # ── L2：删除 thinking / reasoning（近零信息损失）──────────────────────────

    def _compress_l2(self, messages: list[dict]) -> CompressionResult:
        """
        删除所有 assistant 消息中的推理内容：
          - content 列表里 type=thinking 的块
          - reasoning_content 字段
        推理过程是模型的内部"草稿"，对后续轮次续接无参考价值，
        信息损失接近零，因此排在历史裁剪（L3）之前最先做。
        """
        result = []
        tokens_saved = 0

        for msg in messages:
            if msg.get("role") != "assistant":
                result.append(msg)
                continue

            new_msg = dict(msg)

            # 删除 content 里的 thinking 块
            content = msg.get("content")
            if isinstance(content, list):
                thinking_tokens = sum(
                    len(b.get("thinking", "")) // 4
                    for b in content if b.get("type") == "thinking"
                )
                new_msg["content"] = [
                    b for b in content if b.get("type") != "thinking"
                ]
                tokens_saved += thinking_tokens

            # 删除 reasoning_content 字段
            if "reasoning_content" in new_msg:
                rc = new_msg.pop("reasoning_content")
                if rc:
                    tokens_saved += len(str(rc)) // 4

            result.append(new_msg)

        return CompressionResult(result, 2, tokens_saved)

    # ── L3：逐轮 LLM 摘要 ─────────────────────────────────────────────────────

    def _compress_l3(self, messages: list[dict], current_tokens: int) -> CompressionResult:
        """
        将消息历史按对话轮次切分，从最早轮次开始逐轮调 LLM 生成摘要。
        每压完一轮重新估算 token，回落至阈值即停手，不做多余压缩。
        保护最后一轮（当前进行中的对话）不压缩。

        若只有一轮（典型的单任务 Agent），降级到 L4 兜底。
        """
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system  = [m for m in messages if m.get("role") != "system"]

        rounds = _split_into_rounds(non_system)

        if len(rounds) <= 1:
            # 只有一轮，无法按轮压缩，直接走 L4 兜底
            return self._compress_l4(messages, current_tokens)

        last_round   = rounds[-1]
        uncompressed = list(rounds[:-1])   # 可压缩的历史轮次（从老到新）
        compressed_rounds: list[list[dict]] = []
        tokens_saved = 0
        threshold = int(self.config.context_window * self.config.context_l3_threshold)

        while uncompressed:
            # 构建当前候选，检查是否已回落阈值
            candidate = (
                system_msgs
                + [m for cr in compressed_rounds for m in cr]
                + [m for rnd in uncompressed for m in rnd]
                + last_round
            )
            if _estimate_tokens(candidate) <= threshold:
                # 已回落，剩余轮次原样保留
                compressed_rounds.extend(uncompressed)
                break

            rnd = uncompressed.pop(0)

            if _is_complete_round(rnd) and self.client:
                summary = self._summarize_round(rnd)
                if summary:
                    user_msg = next(
                        (m for m in rnd if m.get("role") == "user"), None
                    )
                    if user_msg:
                        before = _estimate_tokens(rnd)
                        compressed_round = [
                            user_msg,
                            {"role": "assistant", "content": f"[历史摘要]\n{summary}"},
                        ]
                        tokens_saved += before - _estimate_tokens(compressed_round)
                        compressed_rounds.append(compressed_round)
                        continue

            # 无法压缩（不完整轮次 / 摘要失败 / 无客户端）→ 原样保留
            compressed_rounds.append(rnd)

        # 重建完整消息列表
        result = (
            system_msgs
            + [m for cr in compressed_rounds for m in cr]
            + last_round
        )
        return CompressionResult(result, 3, tokens_saved)

    def _summarize_round(self, round_msgs: list[dict]) -> Optional[str]:
        """对一个完整对话轮次调用 LLM 生成 6 字段结构化摘要。失败返回 None。"""
        serialized = _serialize_round(round_msgs)
        if not serialized.strip():
            return None

        prompt = (
            "你是上下文压缩助手。请将以下对话轮次压缩为结构化摘要，"
            "供后续 Agent 续接任务使用。\n\n"
            "---\n"
            f"{serialized}\n"
            "---\n\n"
            "请按以下格式输出（无内容的字段可省略）：\n"
            "用户请求：\n"
            "已执行操作：\n"
            "关键工具结果：\n"
            "结论与决策：\n"
            "相关文件或状态：\n"
            "未解决事项：\n\n"
            "要求：\n"
            "- 保留文件路径、报错信息、关键数值\n"
            "- 只写实际执行的操作，不写计划或建议\n"
            "- 不推断未经验证的结论\n"
            "- 只输出摘要正文，不要前言或解释"
        )

        try:
            resp = self.client.chat_completion(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.l3_summary_max_tokens,
            )
            summary = resp.content
            return summary.strip() if _is_summary_usable(summary) else None
        except Exception:
            return None

    # ── L4：单轮溢出兜底（5 步递进）───────────────────────────────────────────

    def _compress_l4(self, messages: list[dict], current_tokens: int) -> CompressionResult:
        """
        处理"当前轮次本身超限"的极端情况。
        在不能删除整轮的前提下，对消息内部按优先级逐步清理，
        每步后重估 token，回落至 L3 阈值即停手。

        Step 1: 旧工具结果 → 信息化一行摘要（最近 N 条不动）
        Step 2: 清除全部 thinking / reasoning_content
        Step 3: JSON 感知缩减工具调用参数
        Step 4: 头尾截断仍超大的工具结果（保留末尾报错/结论）
        Step 5: 原子删除最早的工具调用组（assistant+tool 整组删，保持配对）
        """
        result = copy.deepcopy(messages)
        limit   = int(self.config.context_window * self.config.context_l3_threshold)

        def fits() -> bool:
            return _estimate_tokens(result) <= limit

        # ── Step 1：旧工具结果 → 信息化一行摘要 ──────────────────────────────
        tool_ids   = _collect_tool_ids(result)
        keep_ids   = set(tool_ids[-self.config.l4_keep_recent_tools:])
        clear_ids  = set(tool_ids) - keep_ids

        for msg in result:
            if msg.get("role") != "tool":
                continue
            if msg.get("tool_call_id") not in clear_ids:
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or content == "[工具结果已清理]":
                continue
            # 保留工具名 + 前 100 字预览，形成一行信息化摘要
            tool_name = msg.get("name", "tool")
            preview   = content[:100].replace("\n", " ")
            msg["content"] = f"[{tool_name}] {preview}..."

        if fits():
            return CompressionResult(result, 4, current_tokens - _estimate_tokens(result))

        # ── Step 2：清除全部 thinking / reasoning_content ─────────────────────
        for msg in result:
            if "reasoning_content" in msg:
                msg["reasoning_content"] = ""
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = [b for b in content if b.get("type") != "thinking"]

        if fits():
            return CompressionResult(result, 4, current_tokens - _estimate_tokens(result))

        # ── Step 3：JSON 感知缩减工具调用参数 ────────────────────────────────
        for msg in result:
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            new_calls = []
            for call in msg["tool_calls"]:
                func = call.get("function", {})
                args = func.get("arguments", "")
                if len(args) > 500:
                    args = _shrink_json_args(args)
                    call = {**call, "function": {**func, "arguments": args}}
                new_calls.append(call)
            msg["tool_calls"] = new_calls

        if fits():
            return CompressionResult(result, 4, current_tokens - _estimate_tokens(result))

        # ── Step 4：头尾截断仍超大的工具结果 ─────────────────────────────────
        # 保留头部 + 尾部：报错信息和最终结论常在末尾，只截头部会丢关键信息
        max_chars = 1000
        for msg in result:
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > max_chars:
                msg["content"] = _head_tail_truncate(content, max_chars)

        if fits():
            return CompressionResult(result, 4, current_tokens - _estimate_tokens(result))

        # ── Step 5：原子删除最早的工具调用组 ─────────────────────────────────
        # assistant(tool_calls) + 对应 tool result 必须整组删，
        # 单独删任意一条都会破坏工具调用配对，导致 API 报错
        while not fits():
            groups = _find_tool_groups(result)
            if not groups:
                break
            del_set = set(groups[0])  # 删除最早的一组
            result  = [m for i, m in enumerate(result) if i not in del_set]

        tokens_saved = max(0, current_tokens - _estimate_tokens(result))
        return CompressionResult(result, 4, tokens_saved)


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def _split_into_rounds(messages: list[dict]) -> list[list[dict]]:
    """
    按 user 消息边界切分为对话轮次。
    每轮以 user 消息开头，包含其后所有 assistant / tool 消息。
    """
    rounds: list[list[dict]] = []
    current: list[dict] = []
    for msg in messages:
        if msg.get("role") == "user":
            if current:
                rounds.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        rounds.append(current)
    return rounds


def _is_complete_round(round_msgs: list[dict]) -> bool:
    """
    判断是否为可安全摘要的完整轮次：
      - 以 user 消息开头
      - 以无 tool_calls 的 assistant 消息结尾（有非空 content）
    """
    if not round_msgs:
        return False
    if round_msgs[0].get("role") != "user":
        return False
    last = round_msgs[-1]
    if last.get("role") != "assistant":
        return False
    content = last.get("content", "")
    if not isinstance(content, str) or not content.strip():
        return False
    if last.get("tool_calls"):
        return False
    return True


def _serialize_round(round_msgs: list[dict]) -> str:
    """
    把一个完整轮次序列化为带角色标签的文本，供摘要模型阅读。
    对过长内容做头尾截断（保留末尾报错/结论）。
    """
    parts: list[str] = []
    for msg in round_msgs:
        role    = msg.get("role", "")
        content = _extract_text(msg)
        content = _head_tail_truncate(content, max_chars=3000)

        if role == "user":
            parts.append(f"[USER]\n{content}")
        elif role == "assistant":
            if content.strip():
                parts.append(f"[ASSISTANT]\n{content}")
            for call in msg.get("tool_calls") or []:
                func = call.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", "")[:300]
                parts.append(f"[TOOL CALL] {name}({args})")
        elif role == "tool":
            name   = msg.get("name", "")
            header = f"[TOOL RESULT]{' ' + name if name else ''}"
            parts.append(f"{header}\n{content}")

    return "\n\n".join(parts)


def _is_summary_usable(summary: Any) -> bool:
    """判断摘要是否可用：非空、有实质内容、非明显拒答。"""
    if not isinstance(summary, str):
        return False
    s = summary.strip()
    if len(s) < 10:
        return False
    # 明显拒答且内容极短 → 不可用
    if len(s) < 200 and any(
        m in s.lower()
        for m in ("i cannot", "i can't", "sorry", "as an ai", "i am sorry")
    ):
        return False
    return True


def _head_tail_truncate(text: str, max_chars: int, tail_ratio: float = 0.3) -> str:
    """
    头尾截断：保留头部 + 尾部，中间截断。
    尾部比例 tail_ratio，因为报错信息和最终结论常在末尾，
    只保留头部会丢失最关键的信息。
    """
    if len(text) <= max_chars:
        return text
    tail_chars = int(max_chars * tail_ratio)
    head_chars = max_chars - tail_chars
    return text[:head_chars] + "\n...[已截断]...\n" + text[-tail_chars:]


def _collect_tool_ids(messages: list[dict]) -> list[str]:
    """按出现顺序收集所有工具调用的 tool_call_id（从 assistant 消息中提取）。"""
    ids: list[str] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            for call in msg.get("tool_calls") or []:
                cid = call.get("id", "")
                if cid:
                    ids.append(cid)
    return ids


def _find_tool_groups(messages: list[dict]) -> list[list[int]]:
    """
    识别可原子删除的工具调用组：每组 = [assistant_idx, tool_idx1, ...]。
    删除时必须整组删除，避免工具调用配对断裂导致 API 报错。
    """
    groups: list[list[int]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_ids = {tc.get("id") for tc in msg.get("tool_calls", [])}
            group    = [i]
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                if messages[j].get("tool_call_id") in tool_ids:
                    group.append(j)
                j += 1
            groups.append(group)
        i += 1
    return groups


def _shrink_json_args(args: str) -> str:
    """
    JSON 感知地截断过长的工具调用参数。
    先 parse → 递归截断过长的字符串叶子字段 → 重新序列化，
    保证参数仍是合法 JSON，不破坏工具调用结构。
    非合法 JSON 退化为头尾截断。
    """
    try:
        parsed = json.loads(args)
    except (ValueError, TypeError):
        return _head_tail_truncate(args, max_chars=500)

    def _shrink(obj: Any) -> Any:
        if isinstance(obj, str) and len(obj) > 300:
            return obj[:300] + "...[截断]"
        if isinstance(obj, dict):
            return {k: _shrink(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_shrink(v) for v in obj]
        return obj

    return json.dumps(_shrink(parsed), ensure_ascii=False)


def _extract_text(msg: dict) -> str:
    """从消息提取纯文本（兼容 str / list 格式的 content）。"""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            t = block.get("type", "")
            if t == "text":
                parts.append(block.get("text", ""))
            elif t == "thinking":
                parts.append(block.get("thinking", ""))
        return " ".join(parts)
    return ""
