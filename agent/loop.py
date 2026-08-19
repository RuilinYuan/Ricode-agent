"""
AgentLoop — 主执行循环
──────────────────────
支持 OpenAI-compatible API（通过 ANTHROPIC_BASE_URL 中转）
编排三个核心模块：
  ContextCompressor  Token 水位压缩（级联 L1→L4）
  CacheManager       分层缓存 + 定时保活
  LoopDetector       N-gram + 指纹 + 分级熔断

执行流程：
  用户任务 → 规划 → Execute-Observe-Fix 主循环 → 任务完成 / 熔断 / 取消 / 超轮次
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent.api import ApiClient, ApiError
from agent.context_compressor import ContextCompressor
from agent.loop_detector import LoopDetector, LoopSignal
from agent.tools import TOOLS, execute_tool
from config import AgentConfig
from sandbox import create_executor

# ── 结构化日志 ─────────────────────────────────────────────────────────────────

logger = logging.getLogger("agent")


def _setup_logging(config: AgentConfig) -> None:
    """根据配置初始化日志：控制台 + 可选文件。"""
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.setLevel(level)

    # 控制台 handler
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # 文件 handler（可选）
        if config.log_file:
            fh = logging.FileHandler(config.log_file, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)


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

# ── 成本估算 ─────────────────────────────────────────────────────────────────

# 模型定价（每百万 token，USD）
_MODEL_PRICES: dict[str, tuple[float, float]] = {
    # (输入价格, 输出价格)  单位: USD / 1M tokens
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-4-8": (15.0, 75.0),
    "claude-haiku-4-5-20251001": (0.8, 4.0),
    "deepreasoning-ds-v4pro": (1.0, 3.0),
}

# 缓存读取价格系数（Anthropic: 0.1x）
_CACHE_READ_PRICE_FACTOR = 0.1


def _estimate_cost(model: str, input_tokens: int, output_tokens: int,
                   cache_read_tokens: int, cache_write_tokens: int) -> float:
    """根据模型和 token 使用量估算 USD 成本。"""
    prices = _MODEL_PRICES.get(model)
    if not prices:
        # 未知模型：取默认估计
        prices = (2.0, 8.0)
    input_price, output_price = prices

    # 未被缓存的输入（扣减命中部分）
    uncached_input = max(0, input_tokens - cache_read_tokens)
    input_cost = (uncached_input / 1_000_000) * input_price
    # 缓存读取（0.1x）
    cache_read_cost = (cache_read_tokens / 1_000_000) * input_price * _CACHE_READ_PRICE_FACTOR
    # 缓存写入（1.25x，Anthropic 官方定价）
    cache_write_cost = (cache_write_tokens / 1_000_000) * input_price * 1.25
    # 输出
    output_cost = (output_tokens / 1_000_000) * output_price

    return input_cost + cache_read_cost + cache_write_cost + output_cost


# ── 结果数据类 ────────────────────────────────────────────────────────────────

@dataclass
class RunStats:
    rounds: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    compressions: int = 0
    api_errors: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class AgentResult:
    success: bool
    summary: str = ""
    reason: str = ""          # 失败原因：loop_detected / max_rounds / cancelled / api_error / error
    stats: RunStats = field(default_factory=RunStats)


# ── 主循环 ────────────────────────────────────────────────────────────────────

class AgentLoop:
    def __init__(
        self,
        config: AgentConfig | None = None,
        on_event: Optional[Callable[[dict], None]] = None,
    ) -> None:
        # 配置是本实例的唯一运行策略来源：模型、重试、缓存和沙箱范围都由它决定。
        self.config = config or AgentConfig()
        # 所有运行状态统一经由该回调发送给 UI；CLI 模式下由 _emit 回退到终端打印。
        self.on_event = on_event  # UI 回调；None 时退化为 print

        # 结构化日志
        _setup_logging(self.config)

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()

        # 清除可能干扰的 ANTHROPIC_AUTH_TOKEN
        if "ANTHROPIC_AUTH_TOKEN" in os.environ:
            os.environ.pop("ANTHROPIC_AUTH_TOKEN")

        # 客户端只负责与模型服务通信；主循环本身不关心 HTTP、鉴权和重试细节。
        self.client = ApiClient(
            api_key=api_key,
            base_url=base_url,
            max_retries=self.config.max_retries,
            retry_base_delay=self.config.retry_base_delay,
            retry_max_delay=self.config.retry_max_delay,
        )
        # 工具通过执行器访问文件和命令，避免把执行权限直接暴露给模型。
        self.executor = create_executor(self.config)
        self.compressor = ContextCompressor(self.config, client=self.client)
        self.loop_det = LoopDetector(self.config)

        # 取消信号（线程安全）
        self._cancel_event = threading.Event()

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
                logger.warning("RAG 索引构建失败，search_code 不可用：%s", e)
                self.executor.code_index = None

        # 缓存管理器（Anthropic cache_control 不可用，保留框架供未来扩展）
        from agent.cache_manager import CacheManager
        self.cache_mgr = CacheManager(self.config)

    def cancel(self) -> None:
        """向正在执行的任务发送取消信号（线程安全）。"""
        self._cancel_event.set()
        logger.info("任务取消信号已发送")

    def _emit(self, event: dict) -> None:
        """发送事件：有回调则调用，否则 print 到终端。"""
        if self.on_event:
            self.on_event(event)
        else:
            _default_print(event)

    def run(self, task: str) -> AgentResult:
        # 一个 AgentLoop 可以复用执行多个任务，因此每次启动都必须复位一次性状态。
        self._cancel_event.clear()
        stats = RunStats()
        self.loop_det.set_task(task)  # 任务关键词供停滞检测使用

        # ── 分层 system message：稳定层打缓存断点，状态层（任务）不缓存 ──────
        if self.config.cache_enabled:
            system_msg = self.cache_mgr.get_system_message(
                stable_text=_SYSTEM_PROMPT,
                state_text=f"当前任务：{task}",
            )
            # 启动缓存保活线程（防止稳定前缀 5 分钟过期）
            self.cache_mgr.start_warmup(
                client=self.client,
                model=self.config.model,
                stable_system_msg=system_msg,
            )
        else:
            system_msg = {
                "role": "system",
                "content": _SYSTEM_PROMPT + f"\n\n当前任务：{task}",
            }
        # messages 是唯一的“工作记忆”：之后的 assistant/tool 消息都会持续追加到这里。
        messages: list[dict] = [
            system_msg,
            {"role": "user", "content": task},
        ]

        self._emit({"type": "start", "task": task})
        logger.info("任务开始: model=%s max_rounds=%d compression=%s cache=%s",
                     self.config.model, self.config.max_rounds,
                     self.config.compression_enabled, self.config.cache_enabled)

        for round_num in range(1, self.config.max_rounds + 1):
            # ── 取消检查 ───────────────────────────────────────────────────
            if self._cancel_event.is_set():
                self.cache_mgr.stop_warmup()
                logger.info("任务被用户取消 (round %d)", round_num)
                result = AgentResult(
                    success=False,
                    reason="cancelled",
                    summary="任务已被用户取消。",
                    stats=stats,
                )
                self._emit({"type": "done", "result": result})
                return result

            stats.rounds = round_num
            self._emit({"type": "round", "num": round_num})

            # ── API 调用（发送前在历史尾部插入缓存断点，流式接收）──────────────
            try:
                if self.config.cache_enabled:
                    # 不原地修改 messages，防止缓存协议字段进入本地“工作记忆”。
                    cached_messages = self.cache_mgr.add_cache_breakpoint(messages)
                else:
                    cached_messages = messages

                # 如果有 cache_edits，附加到请求中（仅 Claude）
                extra_params = {}
                if hasattr(self, '_pending_cache_edits') and self._pending_cache_edits:
                    extra_params["cache_edits"] = self._pending_cache_edits
                    self._pending_cache_edits = []  # 清空

                # 流式回调：逐 chunk 推送到 UI
                def _on_stream_chunk(delta: str, accumulated: str) -> None:
                    self._emit({
                        "type": "reasoning_delta",
                        "delta": delta,
                        "accumulated": accumulated,
                    })

                # 模型的下一步可以是自然语言回复，也可以是一个或多个工具调用。
                response = self.client.chat_completion(
                    model=self.config.model,
                    messages=cached_messages,
                    tools=OPENAI_TOOLS,
                    max_tokens=self.config.max_tokens,
                    stream_callback=_on_stream_chunk,
                    **extra_params
                )
                if response.retry_count > 0:
                    stats.api_errors += response.retry_count
                    logger.debug("API 调用经历了 %d 次重试", response.retry_count)
            except ApiError as e:
                stats.api_errors += 1
                logger.error("API Error: %s", e)
                self._emit({"type": "error", "message": str(e)})
                return AgentResult(success=False, reason="api_error", stats=stats)

            # ── 更新 token 统计 ─────────────────────────────────────────────
            stats.input_tokens += response.usage.prompt_tokens
            stats.output_tokens += response.usage.completion_tokens
            stats.cache_read_tokens += response.usage.cache_read_tokens
            stats.cache_write_tokens += response.usage.cache_write_tokens

            # ── 发送统计数据事件（供仪表盘使用）─────────────────────────────
            self._emit({
                "type": "stats_update",
                "round": round_num,
                "input_tokens": stats.input_tokens,
                "output_tokens": stats.output_tokens,
                "cache_read_tokens": stats.cache_read_tokens,
                "cache_write_tokens": stats.cache_write_tokens,
                "estimated_cost": round(_estimate_cost(
                    self.config.model,
                    stats.input_tokens, stats.output_tokens,
                    stats.cache_read_tokens, stats.cache_write_tokens,
                ), 4),
                "api_errors": stats.api_errors,
                "compressions": stats.compressions,
            })

            # ── 解析响应 ────────────────────────────────────────────────────
            reasoning_text = response.content
            tool_calls = response.tool_calls

            # 调试：记录原始 API 响应
            self._emit({"type": "debug_raw", "raw": response.raw})

            if reasoning_text.strip():
                self._emit({"type": "reasoning", "text": reasoning_text})

            # 无工具调用 → 模型直接结束
            if not tool_calls and response.finish_reason == "stop":
                # 此分支适合纯问答；工具型任务通常会以 task_complete 分支结束。
                stats.estimated_cost_usd = round(_estimate_cost(
                    self.config.model,
                    stats.input_tokens, stats.output_tokens,
                    stats.cache_read_tokens, stats.cache_write_tokens,
                ), 4)
                self._emit({"type": "end_turn"})
                logger.info("任务完成 (round %d, cost $%.4f)", round_num, stats.estimated_cost_usd)
                return AgentResult(success=True, summary=reasoning_text, stats=stats)

            # ── 执行工具 ────────────────────────────────────────────────────
            task_done = False
            task_summary = ""
            tc_list_for_detector: list[dict] = []
            tr_list_for_detector: list[dict] = []

            # 先追加 assistant 消息（OpenAI 协议要求 assistant.tool_calls
            # 必须位于对应 tool 消息之前，否则下游网关转换报错）
            if tool_calls:
                # 先保存“模型为什么/要求调用什么”，再保存工具结果；这是 OpenAI 消息协议的顺序要求。
                messages.append({
                    "role": "assistant",
                    "content": reasoning_text,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in tool_calls
                    ],
                })

            for tc in tool_calls:
                # ── 每条工具调用后也检查取消 ────────────────────────────────
                if self._cancel_event.is_set():
                    self.cache_mgr.stop_warmup()
                    logger.info("任务在工具执行中被取消")
                    result = AgentResult(
                        success=False, reason="cancelled",
                        summary="任务已被用户取消。", stats=stats,
                    )
                    self._emit({"type": "done", "result": result})
                    return result

                tool_name: str = tc.name
                tool_input: dict = tc.arguments

                if tool_name == "task_complete":
                    # task_complete 是控制信号，不交给沙箱执行；它只表示模型声明任务已完成。
                    task_done = True
                    task_summary = tool_input.get("summary", "任务完成")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "已记录完成。",
                    })
                    self._emit({"type": "task_complete", "summary": task_summary})
                    break

                self._emit({"type": "tool_call", "name": tool_name,
                            "tool_call_id": tc.id, "input": tool_input})
                logger.debug("工具调用: %s", tool_name)

                # 流式输出回调：命令每产生一行就推送到 UI
                _tc_id = tc.id

                def _on_tool_output(line: str, _cid=_tc_id, _name=tool_name) -> None:
                    self._emit({
                        "type": "tool_output_delta",
                        "tool_call_id": _cid,
                        "name": _name,
                        "line": line,
                    })

                # execute_tool 负责按名称分发；此处只负责编排并把实时输出转发给 UI。
                exec_res = execute_tool(
                    tool_name, tool_input, self.executor,
                    on_output=_on_tool_output,
                )

                output = str(exec_res)
                if len(output) > 3000:
                    # 工具日志可能极长；截断后再写回模型，避免一次构建日志耗尽上下文窗口。
                    output = output[:3000] + f"\n...[输出过长，已截断，共 {len(output)} 字符]"

                self._emit({
                    "type": "tool_result",
                    "name": tool_name,
                    "tool_call_id": _tc_id,
                    "output": output,
                    "success": exec_res.success,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output,
                })

                # 检测器保存的是轻量摘要，而不是整个 messages，便于跨轮比较重复行为。
                tc_list_for_detector.append({"name": tool_name, "input": tool_input})
                tr_list_for_detector.append({"content": output, "success": exec_res.success})

            # 任务完成
            if task_done:
                self.cache_mgr.stop_warmup()
                stats.estimated_cost_usd = round(_estimate_cost(
                    self.config.model,
                    stats.input_tokens, stats.output_tokens,
                    stats.cache_read_tokens, stats.cache_write_tokens,
                ), 4)
                logger.info("任务完成 (round %d, cost $%.4f)", round_num, stats.estimated_cost_usd)
                return AgentResult(success=True, summary=task_summary, stats=stats)

            # ── Token 水位检查与分层压缩 ────────────────────────────────────
            if self.config.compression_enabled:
                # 压缩发生在工具结果写回之后，确保刚完成的动作不会被过早丢弃。
                compress_result = self.compressor.check_and_compress(messages)
                if compress_result.level_applied > 0:
                    messages = compress_result.messages
                    stats.compressions += 1

                    # 如果有 cache_edits，保存到待发送队列（仅 Claude）
                    if compress_result.cache_edits:
                        if not hasattr(self, '_pending_cache_edits'):
                            self._pending_cache_edits = []
                        self._pending_cache_edits.extend(compress_result.cache_edits)
                        logger.debug("压缩 L%d 触发（Cache Edits），节省约 %d tokens",
                                    compress_result.level_applied, compress_result.tokens_saved)
                    else:
                        logger.debug("压缩 L%d 触发，节省约 %d tokens",
                                    compress_result.level_applied, compress_result.tokens_saved)

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
                logger.warning("循环信号: %s (连续 %d 轮)",
                              signal.value, self.loop_det.consecutive_loop_count)

            if signal == LoopSignal.HEAVY:
                self.cache_mgr.stop_warmup()
                stats.estimated_cost_usd = round(_estimate_cost(
                    self.config.model,
                    stats.input_tokens, stats.output_tokens,
                    stats.cache_read_tokens, stats.cache_write_tokens,
                ), 4)
                result = AgentResult(
                    success=False,
                    reason="loop_detected",
                    summary="Agent 陷入无进展循环，已自动熔断。",
                    stats=stats,
                )
                self._emit({"type": "done", "result": result})
                return result

            if signal in (LoopSignal.MEDIUM, LoopSignal.LIGHT):
                # 把纠偏指令伪装成新的用户消息，确保下一轮模型一定能在上下文中看到它。
                guidance = self.loop_det.get_guidance_message()
                messages.append({"role": "user", "content": guidance})
                if signal == LoopSignal.MEDIUM:
                    self.loop_det.reset()

        # 超过最大轮次
        self.cache_mgr.stop_warmup()
        stats.estimated_cost_usd = round(_estimate_cost(
            self.config.model,
            stats.input_tokens, stats.output_tokens,
            stats.cache_read_tokens, stats.cache_write_tokens,
        ), 4)
        result = AgentResult(
            success=False,
            reason="max_rounds_exceeded",
            summary=f"已达最大轮次 {self.config.max_rounds}，任务未完成。",
            stats=stats,
        )
        self._emit({"type": "done", "result": result})
        logger.warning("达到最大轮次限制: %d", self.config.max_rounds)
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
    # Windows 控制台 UTF-8 编码兼容
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
        print(f"  {status} | {s.rounds} rounds | cost ${s.estimated_cost_usd:.4f}")
        print(line)
