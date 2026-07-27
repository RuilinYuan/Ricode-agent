"""
分层缓存布局与时间压缩机制
──────────────────────────
三层 prompt 结构：
  稳定层  system prompt 主体，打 cache_control 断点，极少变动，缓存命中率最高
  状态层  当前任务描述，任务级更新，不缓存
  动态层  消息历史尾部，每轮更新，不缓存

时间压缩机制：
  后台线程每 295s 发一次 max_tokens=1 的轻量请求，
  主动刷新 Anthropic 5 分钟缓存窗口，防止稳定前缀过期冷启动重算。

注：cache_control 为 Anthropic 原生 API 特性，通过兼容 OpenAI 格式的
    中转代理调用时是否生效取决于代理实现；时间压缩机制对所有场景均无副作用。
"""
from __future__ import annotations

import threading
from typing import Any


class CacheManager:
    def __init__(self, config: Any) -> None:
        self.config = config
        self._stop_event = threading.Event()
        self._warmup_thread: threading.Thread | None = None

        # 首次 API 调用后填充
        self._client: Any = None
        self._model: str = ""
        self._stable_system_msg: dict = {}

    # ── 三层 system message 构建 ───────────────────────────────────────────────

    def get_system_message(self, stable_text: str, state_text: str = "") -> dict:
        """
        返回带 cache_control 的 system 消息（OpenAI messages 格式）。

        稳定层尾部插入 cache_control，让其成为缓存前缀；
        状态层（任务描述）追加在后，不缓存，随任务更新。

        content 使用 list 格式，支持 cache_control 字段透传给
        Anthropic 原生 API 或兼容代理。
        """
        content: list[dict] = [
            {
                "type": "text",
                "text": stable_text,
                "cache_control": {"type": "ephemeral"},  # 缓存断点
            }
        ]
        if state_text:
            content.append({"type": "text", "text": state_text})

        return {"role": "system", "content": content}

    def add_cache_breakpoint(self, messages: list[dict]) -> list[dict]:
        """
        在消息列表倒数第 2 条插入 cache_control 断点：
          messages[:-2]  历史层（稳定，命中缓存）
          messages[-2:]  动态尾部（每轮更新，不缓存）

        只操作浅拷贝，不修改原始列表。
        消息数不足 3 条时原样返回。
        """
        if len(messages) < 3:
            return messages

        result = list(messages)
        idx = len(result) - 2

        msg = dict(result[idx])
        content = msg.get("content")

        if isinstance(content, str):
            # 纯字符串包装为带 cache_control 的 block
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
            last = dict(new_content[-1])
            last["cache_control"] = {"type": "ephemeral"}
            new_content[-1] = last
            msg["content"] = new_content

        result[idx] = msg
        return result

    # ── 缓存保活（时间压缩机制）──────────────────────────────────────────────

    def start_warmup(self, client: Any, model: str, stable_system_msg: dict) -> None:
        """
        启动后台保活线程。
        每隔 cache_warmup_interval 秒发一次 max_tokens=1 的轻量请求，
        刷新稳定前缀的缓存 TTL，防止过期后冷启动重算。

        client        : ApiClient 实例
        model         : 当前使用的模型
        stable_system_msg : get_system_message() 返回的 system 消息
        """
        if self._warmup_thread and self._warmup_thread.is_alive():
            return  # 已在运行，无需重复启动

        self._client = client
        self._model = model
        self._stable_system_msg = stable_system_msg

        self._stop_event.clear()
        self._warmup_thread = threading.Thread(
            target=self._warmup_worker,
            daemon=True,
            name="cache-warmup",
        )
        self._warmup_thread.start()

    def stop_warmup(self) -> None:
        """任务结束后停止保活线程。"""
        self._stop_event.set()

    def _warmup_worker(self) -> None:
        """
        保活线程主体：每隔 cache_warmup_interval 秒发一次轻量请求。
        max_tokens=1 只为刷新缓存，不消耗实际输出配额。
        发送失败静默忽略，下次再试，不影响主流程。
        """
        interval = self.config.cache_warmup_interval

        while not self._stop_event.wait(timeout=interval):
            try:
                self._client.chat_completion(
                    model=self._model,
                    messages=[
                        self._stable_system_msg,
                        {"role": "user", "content": "ping"},
                    ],
                    max_tokens=1,
                )
            except Exception:
                # 保活失败不中断主流程，等下次再试
                pass
