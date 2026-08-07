"""
API 客户端封装
─────────────
使用 httpx 直接调用 OpenAI-compatible API，不依赖任何 SDK 版本。
支持指数退避重试（502/503/504/超时）。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

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
    ) -> ApiResponse:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            body["tools"] = tools

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        last_error: Exception | None = None
        retry_count = 0

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=body,
                )

                # 可重试的 HTTP 错误（502/503/504/429）
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

                # 不可重试的错误（4xx 非 429）
                if resp.status_code >= 400:
                    try:
                        detail = resp.json()
                    except Exception:
                        detail = resp.text[:500]
                    raise ApiError(resp.status_code, detail)

                # 成功
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
                # 网络层错误（连接重置等）也重试
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

        # 所有重试均失败
        raise ApiError(0, f"重试 {self.max_retries} 次后仍失败: {last_error}")

    def _parse_response(self, data: dict, retry_count: int) -> ApiResponse:
        """解析 API 响应为 ApiResponse。"""
        # 解析 usage（兼容多种缓存字段命名）
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

        # 解析 message
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        # DeepSeek 模型的 reasoning_content 在 content 为空时作为思考过程
        content = msg.get("content") or msg.get("reasoning_content") or ""

        # 解析 tool_calls
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
    # 修复前置空对象：{}{"key": ...} → {"key": ...}
    while raw.startswith("{}"):
        raw = raw[2:].strip()
    # 修复后置空对象：{"key": ...}{} → {"key": ...}
    while raw.endswith("{}"):
        raw = raw[:-2].strip()
    # 修复重复逗号：{,, → {
    while ",," in raw:
        raw = raw.replace(",,", ",")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 最后尝试：找第一个 { 和最后一个 }
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
