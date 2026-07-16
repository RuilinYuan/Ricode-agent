"""
分层上下文压缩
──────────────
监控 API 返回的 usage.input_tokens / context_window 水位，
按阈值逐级触发压缩，信息损失从低到高：

  L1 (60%) 持久化大 tool_result 到磁盘，context 保留摘要引用
  L2 (75%) 裁剪早期历史，保留最近 N 条消息 + 一句摘要
  L3 (85%) 删除 thinking/reasoning 块，只保留结论
  L4 (95%) 对前半段历史调用 API 生成摘要，以摘要替换
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import anthropic


@dataclass
class CompressionResult:
    messages: list[dict]
    level_applied: int   # 0=未压缩，1-4=对应级别
    tokens_saved: int    # 粗略估算（字符数 / 4）


class ContextCompressor:
    def __init__(self, config: Any, client: Any = None) -> None:
        self.config = config
        self.client = client  # L4 摘要需要 API 调用，可选

        self._work_dir = Path(config.workspace_dir)
        self._work_dir.mkdir(exist_ok=True)
        self._l1_store: dict[str, str] = {}  # key → 文件路径

    # ── 公开接口 ───────────────────────────────────────────────────────────────

    def check_and_compress(
        self,
        messages: list[dict],
        usage: Any,  # anthropic.types.Usage
    ) -> CompressionResult:
        """
        读取 usage.input_tokens 计算水位，按需压缩。
        返回（可能已修改的）消息列表与压缩元数据。
        """
        ratio = usage.input_tokens / self.config.context_window

        if ratio >= self.config.context_l4_threshold:
            result = self._compress_l4(messages)
        elif ratio >= self.config.context_l3_threshold:
            result = self._compress_l3(messages)
        elif ratio >= self.config.context_l2_threshold:
            result = self._compress_l2(messages)
        elif ratio >= self.config.context_l1_threshold:
            result = self._compress_l1(messages)
        else:
            result = CompressionResult(messages, 0, 0)

        return result

    # ── L1：持久化大 tool_result ───────────────────────────────────────────────

    def _compress_l1(self, messages: list[dict]) -> CompressionResult:
        threshold = self.config.l1_persist_threshold
        compressed: list[dict] = []
        tokens_saved = 0

        for msg in messages:
            if msg["role"] != "user":
                compressed.append(msg)
                continue

            content = msg.get("content")
            if not isinstance(content, list):
                compressed.append(msg)
                continue

            new_content: list[dict] = []
            for block in content:
                if block.get("type") != "tool_result":
                    new_content.append(block)
                    continue

                text = _extract_block_text(block.get("content", ""))
                if len(text) <= threshold:
                    new_content.append(block)
                    continue

                # 持久化到磁盘
                key = hashlib.md5(text.encode()).hexdigest()[:8]
                filepath = self._work_dir / f"tr_{key}.txt"
                filepath.write_text(text, encoding="utf-8")
                self._l1_store[key] = str(filepath)

                # 保留摘要引用
                summary = (
                    text[:200]
                    + f"\n...[输出过长，已保存至 {filepath.name}，共 {len(text)} 字符]"
                )
                new_block = dict(block)
                new_block["content"] = [{"type": "text", "text": summary}]
                new_content.append(new_block)
                tokens_saved += (len(text) - len(summary)) // 4

            compressed.append({**msg, "content": new_content})

        return CompressionResult(compressed, 1, tokens_saved)

    # ── L2：裁剪早期历史 ───────────────────────────────────────────────────────

    def _compress_l2(self, messages: list[dict]) -> CompressionResult:
        keep = self.config.l2_keep_recent
        if len(messages) <= keep:
            return CompressionResult(messages, 2, 0)

        old_msgs = messages[:-keep]
        recent_msgs = messages[-keep:]

        # 从旧消息中提取关键推理句子作为摘要
        snippets: list[str] = []
        for msg in old_msgs:
            if msg["role"] == "assistant":
                text = _extract_msg_text(msg)
                if len(text) > 30:
                    snippets.append(text[:120].strip())
                    if len(snippets) >= 3:
                        break

        summary_body = "；".join(snippets) if snippets else "（早期步骤）"
        summary_text = (
            f"[历史摘要] 已压缩 {len(old_msgs)} 条早期消息。"
            f"关键进展：{summary_body}"
        )
        summary_msg = {"role": "user", "content": summary_text}

        tokens_saved = sum(len(_extract_msg_text(m)) for m in old_msgs) // 4
        return CompressionResult([summary_msg] + recent_msgs, 2, tokens_saved)

    # ── L3：删除 thinking 块 ───────────────────────────────────────────────────

    def _compress_l3(self, messages: list[dict]) -> CompressionResult:
        compressed: list[dict] = []
        tokens_saved = 0

        for msg in messages:
            if msg["role"] != "assistant":
                compressed.append(msg)
                continue

            content = msg.get("content")
            if not isinstance(content, list):
                compressed.append(msg)
                continue

            new_content = [
                block for block in content if block.get("type") != "thinking"
            ]
            removed = len(content) - len(new_content)
            if removed:
                tokens_saved += sum(
                    len(b.get("thinking", "")) // 4
                    for b in content
                    if b.get("type") == "thinking"
                )

            compressed.append({**msg, "content": new_content})

        return CompressionResult(compressed, 3, tokens_saved)

    # ── L4：API 摘要替换 ───────────────────────────────────────────────────────

    def _compress_l4(self, messages: list[dict]) -> CompressionResult:
        if len(messages) < 4:
            # 消息太少，降级到 L3
            return self._compress_l3(messages)

        split = len(messages) // 2
        old_half = messages[:split]
        new_half = messages[split:]

        # 构造摘要输入
        history_lines = [
            f"{m['role'].upper()}: {_extract_msg_text(m)[:200]}"
            for m in old_half
        ]
        history_text = "\n".join(history_lines)

        if self.client:
            try:
                resp = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=400,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "请用 3-5 句话总结以下编程 Agent 执行历史的关键进展，"
                                "保留重要的错误信息和已完成的步骤：\n\n"
                                + history_text
                            ),
                        }
                    ],
                )
                summary_text = resp.content[0].text
            except Exception as e:
                summary_text = f"（摘要生成失败：{e}）{history_text[:300]}"
        else:
            summary_text = f"[执行历史摘要] {history_text[:400]}..."

        summary_msg = {
            "role": "user",
            "content": f"[L4 历史摘要，已压缩 {len(old_half)} 条消息]\n{summary_text}",
        }
        tokens_saved = sum(len(_extract_msg_text(m)) for m in old_half) // 4
        return CompressionResult([summary_msg] + new_half, 4, tokens_saved)


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def _extract_block_text(content: Any) -> str:
    """从 tool_result content 字段提取纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content if b.get("type") == "text"
        )
    return str(content)


def _extract_msg_text(msg: dict) -> str:
    """从消息对象提取纯文本（兼容 str 和 list 格式的 content）。"""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("type", "")
                if t == "text":
                    parts.append(block.get("text", ""))
                elif t == "thinking":
                    parts.append(block.get("thinking", ""))
                elif t == "tool_result":
                    parts.append(_extract_block_text(block.get("content", "")))
        return " ".join(parts)
    return ""
