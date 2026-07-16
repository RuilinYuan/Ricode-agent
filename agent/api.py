"""
API 客户端封装
─────────────
使用 httpx 直接调用 OpenAI-compatible API，不依赖任何 SDK 版本。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class ApiUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


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


class ApiClient:
    """OpenAI-compatible API 客户端，用 httpx 裸调。"""

    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        # 确保 base_url 以 /v1 结尾（OpenAI SDK 也是这样处理）
        self.base_url = base_url.rstrip("/")
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

        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=body,
        )

        # 检查 HTTP 错误
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:500]
            raise ApiError(resp.status_code, detail)

        data = resp.json()

        # 解析 usage
        usage_raw = data.get("usage", {})
        usage = ApiUsage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
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
