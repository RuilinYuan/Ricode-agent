"""
分层缓存布局与时间压缩机制
──────────────────────────
三层 prompt 结构：
  稳定层  system prompt 开头，标记 cache_control → Anthropic 缓存前缀
  状态层  当前任务目标，随任务更新，不缓存
  动态层  最新工具结果，每轮更新，不缓存

时间压缩机制：
  后台线程每 295s 发一次轻量请求，刷新 Anthropic 5 分钟缓存窗口，
  防止稳定前缀过期重算。
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import anthropic


class CacheManager:
    def __init__(self, config: Any) -> None:
        self.config = config
        self._stop_event = threading.Event()
        self._warmup_thread: threading.Thread | None = None

        # 刷新线程持有的引用（首次 API 调用后填充）
        self._client: Any = None
        self._model: str = ""
        self._warmup_system: list[dict] = []
        self._warmup_messages: list[dict] = []

    # ── 三层 system prompt 构建 ────────────────────────────────────────────────

    def build_system_prompt(self, stable_text: str, state_text: str) -> list[dict]:
        """
        返回 Anthropic messages.create(system=...) 所需的 list[dict]。

        稳定层尾部插入 cache_control，让其成为缓存前缀；
        状态层不缓存，随任务更新。
        """
        return [
            # 稳定层：极少变动，缓存命中率最高
            {
                "type": "text",
                "text": stable_text,
                "cache_control": {"type": "ephemeral"},
            },
            # 状态层：任务级更新，不插缓存断点
            {
                "type": "text",
                "text": state_text,
            },
        ]

    def add_cache_breakpoint(self, messages: list[dict]) -> list[dict]:
        """
        在消息列表的倒数第 2 条位置插入 cache_control 断点：
          messages[:-2]  历史层（稳定，命中缓存）
          messages[-2:]  动态尾部（每轮更新，不缓存）

        只操作 dict 浅拷贝，不修改原始 messages 列表。
        """
        if len(messages) < 3:
            return messages

        result = list(messages)
        breakpoint_idx = len(result) - 2  # 倒数第 2 条

        msg = dict(result[breakpoint_idx])
        content = msg.get("content")

        if isinstance(content, str):
            # 将纯字符串内容包装成带 cache_control 的 block
            msg["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif isinstance(content, list) and content:
            # 在最后一个 block 上插 cache_control
            new_content = list(content)
            last_block = dict(new_content[-1])
            last_block["cache_control"] = {"type": "ephemeral"}
            new_content[-1] = last_block
            msg["content"] = new_content

        result[breakpoint_idx] = msg
        return result

    # ── 缓存刷新（时间压缩机制）──────────────────────────────────────────────

    def start_warmup(
        self,
        client: Any,
        model: str,
        system: list[dict],
        stable_messages: list[dict],
    ) -> None:
        """
        启动后台刷新线程。
        stable_messages 只需传入历史层中不包含最新动态内容的部分（前 2 条即可）。
        """
        self._client = client
        self._model = model
        self._warmup_system = system
        # 只需要稳定前缀，不需要完整历史
        self._warmup_messages = stable_messages[:2] if stable_messages else [
            {"role": "user", "content": "ping"}
        ]

        if self._warmup_thread and self._warmup_thread.is_alive():
            return  # 已在运行

        self._stop_event.clear()
        self._warmup_thread = threading.Thread(
            target=self._warmup_worker,
            daemon=True,
            name="cache-warmup",
        )
        self._warmup_thread.start()

    def stop_warmup(self) -> None:
        """任务结束后停止刷新线程。"""
        self._stop_event.set()

    def _warmup_worker(self) -> None:
        """每 cache_warmup_interval 秒发一次 max_tokens=1 的请求刷新缓存。"""
        interval = self.config.cache_warmup_interval
        while not self._stop_event.wait(timeout=interval):
            try:
                self._client.messages.create(
                    model=self._model,
                    max_tokens=1,
                    system=self._warmup_system,
                    messages=self._warmup_messages,
                )
            except Exception:
                # 刷新失败不影响主流程，下次再试
                pass
