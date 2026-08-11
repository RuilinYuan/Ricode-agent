"""
API 客户端封装
─────────────
使用 httpx 直接调用 OpenAI-compatible API，不依赖任何 SDK 版本。
支持指数退避重试（502/503/504/超时）。
支持 SSE 流式响应（stream=True），返回增量内容。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Optional

import httpx


@dataclass
class ApiUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ApiResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: ApiUsage = field(default_factory=ApiUsage)
    raw: dict = field(default_factory=dict)
    retry_count: int = 0  # 本次请求实际重试次数


# 流式增量回调类型：(delta_text, accumulated_text) → None
StreamCallback = Callable[[str, str], None]

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ApiClient:
    """OpenAI-compatible API 客户端，用 httpx 裸调，内置指数退避重试。"""

    def __init__(self, api_key: str, base_url: str,
                 max_retries: int = 3,
                 retry_base_delay: float = 1.0,
                 retry_max_delay: float = 60.0) -> None:
        self.api_key = api_key
        # 确保 base_url 以 /v1 结尾（裸域名时自动补上）
        base = base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        self.base_url = base
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self._client = httpx.Client(timeout=httpx.Timeout(600.0))

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        cache_edits: list[dict] | None = None,
        stream_callback: StreamCallback | None = None,
    ) -> ApiResponse:
        """
        非流式调用（向后兼容）或流式调用（传入 stream_callback 时）。

        stream_callback: 每收到一个 SSE chunk 就调用 (delta_text, accumulated_text)。
                         传入后自动启用 stream=True。
        """
        use_stream = stream_callback is not None

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": use_stream,
        }
        if tools:
            body["tools"] = tools
        if cache_edits:
            body["cache_edits"] = cache_edits

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        last_error: Exception | None = None
        retry_count = 0

        for attempt in range(self.max_retries + 1):
            try:
                if use_stream:
                    return self._stream_request(
                        body, headers, stream_callback, retry_count,
                    )

                resp = self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=body,
                )

                if resp.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                    retry_count += 1
                    delay = min(
                        self.retry_base_delay * (2 ** attempt),
                        self.retry_max_delay,
                    )
                    detail = resp.text[:200]
                    print(f"[api] {resp.status_code}，{delay:.1f}s 后重试 ({retry_count}/{self.max_retries}): {detail}")
                    time.sleep(delay)
                    continue

                if resp.status_code >= 400:
                    try:
                        detail = resp.json()
                    except Exception:
                        detail = resp.text[:500]
                    raise ApiError(resp.status_code, detail)

                data = resp.json()
                return self._parse_response(data, retry_count)

            except httpx.TimeoutException as e:
                retry_count += 1
                if attempt < self.max_retries:
                    delay = min(
                        self.retry_base_delay * (2 ** attempt),
                        self.retry_max_delay,
                    )
                    print(f"[api] 请求超时，{delay:.1f}s 后重试 ({retry_count}/{self.max_retries})")
                    time.sleep(delay)
                    continue
                last_error = e

            except httpx.RequestError as e:
                retry_count += 1
                if attempt < self.max_retries:
                    delay = min(
                        self.retry_base_delay * (2 ** attempt),
                        self.retry_max_delay,
                    )
                    print(f"[api] 网络错误: {e}，{delay:.1f}s 后重试 ({retry_count}/{self.max_retries})")
                    time.sleep(delay)
                    continue
                last_error = e

        raise ApiError(0, f"重试 {self.max_retries} 次后仍失败: {last_error}")

    # ── SSE 流式请求 ─────────────────────────────────────────────────────

    def _stream_request(
        self,
        body: dict,
        headers: dict,
        callback: StreamCallback,
        retry_count: int,
    ) -> ApiResponse:
        """
        发送 stream=True 请求，逐 chunk 解析 SSE，
        同时积累完整内容组装为 ApiResponse。
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_map: dict[int, dict] = {}  # index → {id, name, arguments_str}
        finish_reason = ""
        usage_raw: dict = {}

        with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=body,
            timeout=httpx.Timeout(600.0),
        ) as resp:
            if resp.status_code >= 400:
                detail = resp.read().decode("utf-8", errors="replace")[:500]
                raise ApiError(resp.status_code, detail)

            for line in resp.iter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.strip() == "data: [DONE]":
                    break
                if not line.startswith("data: "):
                    continue

                try:
                    chunk = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                # Extract usage from final chunk (some APIs include it)
                if "usage" in chunk:
                    usage_raw = chunk["usage"]

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr

                # ── 文本增量 ──────────────────────────────────────────
                delta_text = delta.get("content") or ""
                delta_reasoning = delta.get("reasoning_content") or ""

                if delta_text:
                    content_parts.append(delta_text)
                    callback(delta_text, "".join(content_parts))
                elif delta_reasoning:
                    reasoning_parts.append(delta_reasoning)
                    callback(delta_reasoning, "".join(reasoning_parts))

                # ── tool_calls 增量 ───────────────────────────────────
                for tc_delta in delta.get("tool_calls") or []:
                    idx = tc_delta.get("index", 0)
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": tc_delta.get("id", ""),
                            "name": "",
                            "arguments_str": "",
                        }
                    entry = tool_calls_map[idx]
                    if tc_delta.get("id"):
                        entry["id"] = tc_delta["id"]
                    func = tc_delta.get("function", {})
                    if func.get("name"):
                        entry["name"] = func["name"]
                    if func.get("arguments"):
                        entry["arguments_str"] += func["arguments"]

        # ── 组装最终响应 ───────────────────────────────────────────────
        # 优先使用 content，如果为空则使用 reasoning_content
        full_content = "".join(content_parts) or "".join(reasoning_parts)

        tool_calls = []
        for idx in sorted(tool_calls_map):
            entry = tool_calls_map[idx]
            if not entry["name"]:
                continue
            args = _safe_parse_json(entry["arguments_str"])
            tool_calls.append(ToolCall(
                id=entry["id"],
                name=entry["name"],
                arguments=args,
            ))

        details = usage_raw.get("prompt_tokens_details") or {}
        usage = ApiUsage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            cache_read_tokens=(
                usage_raw.get("cache_read_input_tokens", 0)
                or usage_raw.get("prompt_cache_hit_tokens", 0)
                or details.get("cached_tokens", 0)
            ),
            cache_write_tokens=usage_raw.get("cache_creation_input_tokens", 0),
        )

        return ApiResponse(
            content=full_content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            raw=usage_raw or {},  # raw 只保留 usage，完整 chunks 太多
            retry_count=retry_count,
        )

    # ── 非流式响应解析 ─────────────────────────────────────────────────

    def _parse_response(self, data: dict, retry_count: int) -> ApiResponse:
        """解析 API 响应为 ApiResponse。"""
        usage_raw = data.get("usage", {})
        details = usage_raw.get("prompt_tokens_details") or {}
        usage = ApiUsage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            cache_read_tokens=(
                usage_raw.get("cache_read_input_tokens", 0)
                or usage_raw.get("prompt_cache_hit_tokens", 0)
                or details.get("cached_tokens", 0)
            ),
            cache_write_tokens=usage_raw.get("cache_creation_input_tokens", 0),
        )

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content") or msg.get("reasoning_content") or ""

        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            func = tc.get("function", {})
            raw_args = func.get("arguments", "{}")
            args = _safe_parse_json(raw_args)
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=func.get("name", ""),
                arguments=args,
            ))

        return ApiResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", ""),
            usage=usage,
            raw=data,
            retry_count=retry_count,
        )


def _safe_parse_json(raw: str) -> dict:
    """容错 JSON 解析：修复模型输出中的常见格式问题。"""
    raw = raw.strip()
    while raw.startswith("{}"):
        raw = raw[2:].strip()
    while raw.endswith("{}"):
        raw = raw[:-2].strip()
    while ",," in raw:
        raw = raw.replace(",,", ",")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        start = raw.index("{")
        end = raw.rindex("}")
        return json.loads(raw[start:end + 1])
    except (ValueError, json.JSONDecodeError):
        return {}


class ApiError(Exception):
    def __init__(self, status_code: int, detail: Any) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API Error {status_code}: {detail}")
