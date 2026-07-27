"""
AgentLoop — 主执行循环
──────────────────────
支持 OpenAI-compatible API（通过 ANTHROPIC_BASE_URL 中转）
编排三个核心模块：
  ContextCompressor  Token 水位压缩
  CacheManager       缓存管理器（API 缓冲，非 Anthropic cache_control）
  LoopDetector       N-gram + 指纹 + 分级熔断

执行流程：
  用户任务 → 规划 → Execute-Observe-Fix 主循环 → 任务完成 / 熔断 / 超轮次
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent.api import ApiClient, ApiError
from agent.context_compressor import ContextCompressor
from agent.loop_detector import LoopDetector, LoopSignal
from agent.tools import TOOLS, execute_tool
from config import AgentConfig
from sandbox import create_executor

# ── 系统 Prompt ───────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
你是一个自主编程 Agent。根据用户需求，通过工具调用完成代码生成、执行验证、报错修复、测试通过的完整闭环。

# 工作原则
- 先写代码文件，再执行验证，不要只生成而不运行
- 遇到报错：读错误 → 分析根因 → 修改 → 再执行，不要猜测
- 有单元测试需求时，运行 run_tests 确认通过再结束
- 确认任务完成后调用 task_complete，传入总结

# 代码质量
- 函数职责单一，不超过 40 行
- 使用类型注解
- 重要逻辑写注释

# 禁止行为
- 不要重复相同的修复而不换思路
- 不要在未执行验证的情况下宣称完成
"""

# ── 工具格式转换 Anthropic → OpenAI ──────────────────────────────────────────

def _to_openai_tools(anthropic_tools: list[dict]) -> list[dict]:
    """将 Anthropic tool_use schema 转为 OpenAI function calling 格式。"""
    result = []
    for t in anthropic_tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return result

OPENAI_TOOLS = _to_openai_tools(TOOLS)

# ── 结果数据类 ────────────────────────────────────────────────────────────────

@dataclass
class RunStats:
    rounds: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    compressions: int = 0


@dataclass
class AgentResult:
    success: bool
    summary: str = ""
    reason: str = ""          # 失败原因：loop_detected / max_rounds / error
    stats: RunStats = field(default_factory=RunStats)


# ── 主循环 ────────────────────────────────────────────────────────────────────

class AgentLoop:
    def __init__(
        self,
        config: AgentConfig | None = None,
        on_event: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.on_event = on_event  # UI 回调；None 时退化为 print

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()

        # 清除可能干扰的 ANTHROPIC_AUTH_TOKEN
        if "ANTHROPIC_AUTH_TOKEN" in os.environ:
            os.environ.pop("ANTHROPIC_AUTH_TOKEN")

        self.client = ApiClient(api_key=api_key, base_url=base_url)
        self.executor = create_executor(self.config)
        self.compressor = ContextCompressor(self.config, client=self.client)
        self.loop_det = LoopDetector(self.config)

        # RAG 代码索引：配置 embedding 服务后对仓库建索引，供 search_code 工具使用
        if self.config.rag_enabled and self.config.embedding_base_url:
            from agent.rag import CodeIndex
            self.executor.code_index = CodeIndex(
                root=self.config.workspace_dir,
                base_url=self.config.embedding_base_url,
                api_key=self.config.embedding_api_key,
                model=self.config.embedding_model,
                chunk_lines=self.config.rag_chunk_lines,
                chunk_overlap=self.config.rag_chunk_overlap,
            )
            try:
                self.executor.code_index.build()
            except Exception as e:
                print(f"[rag] 索引构建失败，search_code 不可用：{e}")
                self.executor.code_index = None

        # 缓存管理器（Anthropic cache_control 不可用，保留框架供未来扩展）
        from agent.cache_manager import CacheManager
        self.cache_mgr = CacheManager(self.config)

    def _emit(self, event: dict) -> None:
        """发送事件：有回调则调用，否则 print 到终端。"""
        if self.on_event:
            self.on_event(event)
        else:
            _default_print(event)

    def run(self, task: str) -> AgentResult:
        stats = RunStats()

        # ── 分层 system message：稳定层打缓存断点，状态层（任务）不缓存 ──────
        system_msg = self.cache_mgr.get_system_message(
            stable_text=_SYSTEM_PROMPT,
            state_text=f"当前任务：{task}",
        )
        messages: list[dict] = [
            system_msg,
            {"role": "user", "content": task},
        ]

        # 启动缓存保活线程（防止稳定前缀 5 分钟过期）
        self.cache_mgr.start_warmup(
            client=self.client,
            model=self.config.model,
            stable_system_msg=system_msg,
        )

        self._emit({"type": "start", "task": task})

        for round_num in range(1, self.config.max_rounds + 1):
            stats.rounds = round_num
            self._emit({"type": "round", "num": round_num})

            # ── API 调用（发送前在历史尾部插入缓存断点）──────────────────────
            try:
                cached_messages = self.cache_mgr.add_cache_breakpoint(messages)
                response = self.client.chat_completion(
                    model=self.config.model,
                    messages=cached_messages,
                    tools=OPENAI_TOOLS,
                    max_tokens=self.config.max_tokens,
                )
            except ApiError as e:
                self._emit({"type": "error", "message": str(e)})
                return AgentResult(success=False, reason="api_error", stats=stats)

            # ── 更新 token 统计 ─────────────────────────────────────────────
            stats.input_tokens += response.usage.prompt_tokens
            stats.output_tokens += response.usage.completion_tokens

            # ── 解析响应 ────────────────────────────────────────────────────
            reasoning_text = response.content
            tool_calls = response.tool_calls

            # 调试：记录原始 API 响应
            self._emit({"type": "debug_raw", "raw": response.raw})

            if reasoning_text.strip():
                self._emit({"type": "reasoning", "text": reasoning_text})

            # 无工具调用 → 模型直接结束
            if not tool_calls and response.finish_reason == "stop":
                self._emit({"type": "end_turn"})
                return AgentResult(success=True, summary=reasoning_text, stats=stats)

            # ── 执行工具 ────────────────────────────────────────────────────
            task_done = False
            task_summary = ""
            tc_list_for_detector: list[dict] = []
            tr_list_for_detector: list[dict] = []

            for tc in tool_calls:
                tool_name: str = tc.name
                tool_input: dict = tc.arguments

                if tool_name == "task_complete":
                    task_done = True
                    task_summary = tool_input.get("summary", "任务完成")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "已记录完成。",
                    })
                    self._emit({"type": "task_complete", "summary": task_summary})
                    break

                self._emit({"type": "tool_call", "name": tool_name, "input": tool_input})

                exec_res = execute_tool(tool_name, tool_input, self.executor)

                output = str(exec_res)
                if len(output) > 3000:
                    output = output[:3000] + f"\n...[输出过长，已截断，共 {len(output)} 字符]"

                self._emit({
                    "type": "tool_result",
                    "name": tool_name,
                    "output": output,
                    "success": exec_res.success,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output,
                })

                tc_list_for_detector.append({"name": tool_name, "input": tool_input})
                tr_list_for_detector.append({"content": output})

            # 追加 assistant 消息到历史（API 需要的原始格式）
            assistant_msg: dict = {"role": "assistant", "content": reasoning_text}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_msg)

            # 任务完成
            if task_done:
                self.cache_mgr.stop_warmup()
                return AgentResult(success=True, summary=task_summary, stats=stats)

            # ── Token 水位检查与分层压缩 ────────────────────────────────────
            compress_result = self.compressor.check_and_compress(messages)
            if compress_result.level_applied > 0:
                messages = compress_result.messages
                stats.compressions += 1
                self._emit({
                    "type": "compress",
                    "level": compress_result.level_applied,
                    "tokens_saved": compress_result.tokens_saved,
                })

            # ── 循环检测 ────────────────────────────────────────────────────
            self.loop_det.record(
                round_num=round_num,
                reasoning=reasoning_text,
                tool_calls=tc_list_for_detector,
                tool_results=tr_list_for_detector,
            )
            signal = self.loop_det.check()

            if signal != LoopSignal.NONE:
                self._emit({
                    "type": "loop_signal",
                    "signal": signal.value,
                    "count": self.loop_det.consecutive_loop_count,
                })

            if signal == LoopSignal.HEAVY:
                self.cache_mgr.stop_warmup()
                result = AgentResult(
                    success=False,
                    reason="loop_detected",
                    summary="Agent 陷入无进展循环，已自动熔断。",
                    stats=stats,
                )
                self._emit({"type": "done", "result": result})
                return result

            if signal in (LoopSignal.MEDIUM, LoopSignal.LIGHT):
                guidance = self.loop_det.get_guidance_message()
                messages.append({"role": "user", "content": guidance})
                if signal == LoopSignal.MEDIUM:
                    self.loop_det.reset()

        # 超过最大轮次
        self.cache_mgr.stop_warmup()
        result = AgentResult(
            success=False,
            reason="max_rounds_exceeded",
            summary=f"已达最大轮次 {self.config.max_rounds}，任务未完成。",
            stats=stats,
        )
        self._emit({"type": "done", "result": result})
        return result


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def _msg_text(msg: dict) -> str:
    """从 OpenAI 格式消息提取纯文本（用于 token 估算）。"""
    content = msg.get("content") or ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _make_summary_msg(old_msgs: list[dict]) -> dict:
    """生成早期历史的简短摘要。"""
    snippets = []
    for m in old_msgs:
        if m["role"] == "assistant":
            t = (_msg_text(m) or "").strip()
            if t:
                snippets.append(t[:120])
                if len(snippets) >= 3:
                    break
    body = "；".join(snippets) if snippets else "（早期步骤）"
    return {
        "role": "user",
        "content": f"[历史摘要] 已压缩 {len(old_msgs)} 条早期消息。关键进展：{body}",
    }


# ── 默认终端打印（无 UI 时）─────────────────────────────────────────────────────

_TOOL_ICONS = {
    "write_file": "W", "read_file": "R", "execute_python": "P",
    "run_command": "C", "run_tests": "T", "task_complete": "DONE",
}

def _default_print(event: dict) -> None:
    t = event.get("type")
    if t == "start":
        line = "-" * 56
        print(f"\n{line}\n  Task: {event['task'][:50]}\n{line}")
    elif t == "round":
        print(f"  [{event['num']:02d}]", end=" ", flush=True)
    elif t == "tool_call":
        icon = _TOOL_ICONS.get(event["name"], event["name"])
        print(f"-> {icon}", end=" ", flush=True)
    elif t == "tool_result":
        status = "OK" if event.get("success") else "ERR"
        print(f"[{status}]", end=" ", flush=True)
    elif t == "compress":
        print(f"[L{event['level']} -{event['tokens_saved']}tk]", end=" ")
    elif t == "loop_signal":
        sig = event["signal"].upper()
        print(f"\n  [LOOP-{sig} x{event['count']}]", end=" ")
    elif t == "task_complete":
        print(f"\n  [DONE]")
    elif t == "end_turn":
        print("-> end")
    elif t == "error":
        print(f"\n  [ERROR] {event['message']}")
    elif t == "done":
        r = event["result"]
        s = r.stats
        line = "-" * 56
        status = "SUCCESS" if r.success else f"FAILED ({r.reason})"
        print(f"\n{line}")
        print(f"  {status} | {s.rounds} rounds")
        print(line)
