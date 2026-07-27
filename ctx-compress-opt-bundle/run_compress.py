#!/usr/bin/env python3
"""独立驱动脚本：为 compress.py 提供真实的 llm_provider.chat 实现并跑通压缩流程。

用法（在已激活 venv 的前提下）：
    python ctx-compress-opt/run_compress.py                       # 用默认阈值触发按轮摘要
    python ctx-compress-opt/run_compress.py -d "本次测试调大阈值验证摘要质量"
    COMPRESS_MAX_TOKENS=2000 python ctx-compress-opt/run_compress.py
    COMPRESS_MODEL=qwen-plus python ctx-compress-opt/run_compress.py

每运行一次会在 ctx-compress-opt/runs/ 下新建一个带时间戳的 txt，
把「压缩前 + 压缩后」写在同一个文件里；-d 传入的说明（支持中文）写在文件头部。

凭证解析顺序：
    1. 环境变量 DASHSCOPE_API_KEY
    2. ~/.hermes/auth.json 的 credential_pool.alibaba（自动提取，不打印明文）
"""

import os
import re
import sys
import json
import types
import asyncio
import logging
import argparse
from datetime import datetime


# ────────────────────────────────────────────────────────────────
# 1. 注入 compress.py 顶部依赖的两个模块（本项目里不存在），避免改动原文件
#    compress.py: from configs.config import LOG_LEVEL, RUN_LOG_FILE, ROOT_PATH
#                 from utils.log_utils import get_logger
# ────────────────────────────────────────────────────────────────
def _install_stub_modules() -> None:
    here = os.path.dirname(os.path.abspath(__file__))

    # configs.config 桩
    configs_pkg = types.ModuleType("configs")
    config_mod = types.ModuleType("configs.config")
    config_mod.LOG_LEVEL = os.environ.get("COMPRESS_LOG_LEVEL", "INFO")
    config_mod.RUN_LOG_FILE = "run_compress.log"
    config_mod.ROOT_PATH = here
    configs_pkg.config = config_mod
    sys.modules["configs"] = configs_pkg
    sys.modules["configs.config"] = config_mod

    # utils.log_utils 桩
    # 注意：项目根目录已有真实的 utils.py（含 atomic_replace，被 credential_pool 依赖），
    # 不能整体覆盖 utils，只补一个 utils.log_utils 子模块即可。
    log_mod = types.ModuleType("utils.log_utils")

    def get_logger(name, path=None, log_level="INFO"):
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
            logger.addHandler(handler)
        return logger

    log_mod.get_logger = get_logger
    sys.modules["utils.log_utils"] = log_mod


_install_stub_modules()
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)          # 让 import compress 生效
sys.path.insert(0, _ROOT)          # 让 import agent / utils 等项目模块生效

import compress  # noqa: E402  必须在桩注入之后再 import


# ────────────────────────────────────────────────────────────────
# 2. 真实的 LLM Provider —— 实现 compress.py 需要的 chat() 接口
#    要求：async，返回对象带 .content 属性
# ────────────────────────────────────────────────────────────────
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class _ChatResponse:
    """compress.py 里用的是 summary_response.content，这里对齐该形状。"""

    def __init__(self, content: str):
        self.content = content


class DashScopeProvider:
    def __init__(self, api_key: str, base_url: str = DASHSCOPE_BASE_URL):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        # 记录每一次「发给摘要模型」的调用：阶段 C 每压缩一轮就会调用一次 chat()，
        # 这里把真实发出的 prompt（_summarize_round 里序列化 + 结构化后的完整文本）
        # 以及模型返回的摘要一并留档，供 run 结束后写入 txt。
        self.summary_calls: list[dict] = []

    async def chat(self, messages, model, temperature=0.0, max_tokens=1000):
        # 提取本次真正发给模型的 user 文本（compress.py 只放一条 user 消息）
        sent_text = "\n\n".join(
            m.get("content", "") for m in messages
            if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)
        )
        resp = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content or ""
        self.summary_calls.append({
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt": sent_text,
            "response": content,
        })
        return _ChatResponse(content)


def _load_api_key(provider_id: str = "alibaba") -> str:
    """解析 DashScope key；绝不打印明文。

    顺序：
      1. 环境变量 DASHSCOPE_API_KEY（覆盖一切）
      2. 项目自带的 agent.credential_pool.load_pool（权威路径，
         与 hermes_cli.auth 里读凭证的方式一致）
    """
    key = os.environ.get("DASHSCOPE_API_KEY")
    if key:
        return key.strip()

    try:
        from agent.credential_pool import load_pool

        pool = load_pool(provider_id)
        if pool and pool.has_credentials():
            entry = pool.peek()
            if entry is not None:
                key = getattr(entry, "runtime_api_key", "") or getattr(entry, "access_token", "")
                key = str(key).strip()
                if key:
                    return key
    except Exception as exc:
        raise SystemExit(
            f"通过 agent.credential_pool 读取 provider={provider_id!r} 凭证失败: {exc}\n"
            f"请改用: export DASHSCOPE_API_KEY=你的key"
        )

    raise SystemExit(
        f"credential_pool 中 provider={provider_id!r} 没有可用凭证，"
        "请改用: export DASHSCOPE_API_KEY=你的key"
    )


# ────────────────────────────────────────────────────────────────
# 3. 构造多轮示例对话（4 轮），用小阈值强制触发「按轮 LLM 摘要」
# ────────────────────────────────────────────────────────────────
def _sample_messages() -> list[dict]:
    return [
        {"role": "system", "content": "You are a helpful research assistant."},
        # ── 轮 1 ──
        {"role": "user", "content": "Research the current state of solid-state batteries."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call-1", "type": "function", "function": {
                "name": "web_search",
                "arguments": '{"query": "solid-state batteries 2026 breakthroughs"}'}}],
         "reasoning_content": "User wants a research overview on solid-state batteries. Let me search."},
        {"role": "tool", "tool_call_id": "call-1", "name": "web_search",
         "content": "[{\"title\": \"What's next for EV batteries in 2026\", \"url\": \"https://example.com/a\"}, "
                    "{\"title\": \"US startup clears solid-state milestone\", \"url\": \"https://example.com/b\"}] "
                    + "filler " * 400},
        {"role": "assistant", "content": (
            "Solid-state batteries are moving from lab to production in 2026. ION Storage Systems "
            "qualified its Cornerstone Cell; Factorial hit 745 miles in a Mercedes test; Toyota targets "
            "2027-2028. Energy density targets 400-500 Wh/kg. Main challenges: scale manufacturing and cost.")},
        # ── 轮 2 ──
        {"role": "user", "content": "What's happening with the China housing market right now?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call-2", "type": "function", "function": {
                "name": "web_search",
                "arguments": '{"query": "China housing market 2026 real estate"}'}}],
         "reasoning_content": "Search for latest China property news."},
        {"role": "tool", "tool_call_id": "call-2", "name": "web_search",
         "content": "[{\"title\": \"China home prices seen falling faster before stabilising in 2027\"}] "
                    + "filler " * 400},
        {"role": "assistant", "content": (
            "China's property slump is in its fifth year. Home prices fell ~2.7% YoY in Dec 2025; ~80M unsold "
            "units; sector once ~25% of GDP. Beijing pivoting to a 'new model' focused on affordable housing. "
            "Prices projected to decline ~4% in 2026, possibly stabilising in 2027.")},
        # ── 轮 3 ──
        {"role": "user", "content": "Summarize AI chip export controls in 2026."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call-3", "type": "function", "function": {
                "name": "web_search",
                "arguments": '{"query": "AI chip export controls 2026"}'}}],
         "reasoning_content": "Search for export-control policy updates."},
        {"role": "tool", "tool_call_id": "call-3", "name": "web_search",
         "content": "[{\"title\": \"2026 update to semiconductor export rules\"}] " + "filler " * 400},
        {"role": "assistant", "content": (
            "In 2026 export controls tightened around advanced AI accelerators and HBM, with new licensing "
            "thresholds and expanded entity lists. Vendors adjusted product SKUs to stay under performance caps.")},
        # ── 轮 4（最后一轮，保留不压缩）──
        {"role": "user", "content": "Now compare all three topics in one paragraph."},
        {"role": "assistant", "content": (
            "All three show 2026 as an inflection year: batteries scaling toward production, Chinese property "
            "still contracting, and chip policy tightening — each reshaping its respective supply chain.")},
    ]


def _load_messages_from_file(path: str) -> tuple[list[dict], str | None]:
    """从 json 文件加载消息，供 --input 手动指定使用。

    兼容多种形状：
      1. hermes-chat-logs 里的 dict（含 messages 字段，如 01-local.json / 01-wire.json）；
         若带 model 字段，一并返回作为默认摘要模型。
      2. 纯 list（直接就是消息数组）；
      3. 其它含 messages / history / conversation 列表字段的 dict。
    返回 (messages, model_or_None)。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 形状 2：纯 list
    if isinstance(data, list):
        return data, None

    # 形状 1 / 3：dict，找出消息列表字段
    if isinstance(data, dict):
        model = data.get("model") if isinstance(data.get("model"), str) else None
        for key in ("messages", "history", "conversation"):
            val = data.get(key)
            if isinstance(val, list):
                return val, model

    raise SystemExit(
        f"无法从 {path} 解析消息列表：期望是消息数组，或含 "
        f"'messages'/'history'/'conversation' 字段的对象。"
    )


def _format_messages(messages: list[dict]) -> str:
    """把消息列表渲染成可读文本块（完整内容，不截断）。"""
    lines = []
    for i, m in enumerate(messages):
        role = m.get("role")
        content = m.get("content")
        tag = " [SUMMARY]" if isinstance(content, str) and "[Context Summary]" in content else ""
        lines.append(f"--- [{i}] role={role}{tag} ---")
        if content is None:
            lines.append("(content=None)")
        elif isinstance(content, str):
            lines.append(content)
        else:
            lines.append(json.dumps(content, ensure_ascii=False, indent=2))
        # 附带 tool_calls 摘要（若有），便于观察压缩前后结构
        if m.get("tool_calls"):
            names = [ (c.get("function") or {}).get("name") for c in m["tool_calls"] ]
            lines.append(f"[tool_calls: {names}]")
        lines.append("")
    return "\n".join(lines)


# ── 压缩「处理程度」分级 ──────────────────────────────────────────
# compress.py 的处理路径分多层，按以下优先级识别：
#
#   L0  未压缩（未超阈值，_compress_context 开头直接返回）
#   LA  阶段 A 达标（超大工具响应转存+预览，token 降回阈值内）
#   LB  阶段 B 达标（micro_compact 本地清理，token 降回阈值内）
#   LX  压过但仍超限（阶段 A/B 实际改动了内容、token 下降，但仍 > 阈值；
#       且没有走成功返回路径、也没进兜底——典型于阶段 C 按轮摘要被注释禁用时）
#   L1  轮次摘要压缩成功（按轮 LLM 摘要，循环中途达标）
#   L2  单轮兜底入口（只有单轮 / 无法按轮压缩）
#   L3  全摘要后仍超限（历史轮全 LLM 摘要后仍超限，进兜底）
#
# L2/L3 都会进入 _handle_single_round_overflow，后面追加实际执行到的子步骤：
_MICRO_STEPS = [
    ("Resolved by clearing old tool results",    "清除旧工具结果"),
    ("Resolved by clearing reasoning_content",   "清除思维链"),
    ("Resolved by clearing tool call arguments", "清除工具参数"),
    ("Resolved by hard-truncating tool results", "硬截断工具结果"),
    ("Deleted tool group",                       "删除工具调用组"),
    ("Cannot reduce further",                    "仍无法压缩(超限)"),
]


def _deepest_micro_step(joined: str) -> str | None:
    """返回兜底函数里命中的最深微压缩方法；都没命中返回 None（进入即达标）。"""
    reached = None
    for needle, label in _MICRO_STEPS:
        if needle in joined:
            reached = label
    return reached


def _detect_local_shrink(before_msgs: list[dict], after_msgs: list[dict]) -> bool:
    """结构上判断阶段 A/B 是否真的改动过内容（不依赖"达标返回"日志）。

    只要出现 tool_budget 的转存预览块、micro_compact 的一行摘要 / 去重 / 清空占位，
    或消息条数变少，就说明本地压缩确实生效了（哪怕最终没压到阈值内）。
    """
    # 阶段 A / B 会注入的标记串（前缀匹配）
    _MARKERS = (
        "[TOOL_RESULT_TRUNCATED]",                              # 阶段A 转存预览
        "[Duplicate tool output",                               # 阶段B 去重
        "[Old tool output",                                     # 阶段B 通用清空
        "[Old tool result content cleared",                    # 兼容旧版
    )
    # micro_compact 产出的一行工具摘要，形如 "[web_search] query=..."
    _summary_line = re.compile(r"^\[[a-z_][a-z._]+\] ")

    if len(after_msgs) < len(before_msgs):
        return True
    for m in after_msgs:
        c = m.get("content")
        if not isinstance(c, str):
            continue
        if c.startswith(_MARKERS) or _summary_line.match(c):
            return True
    return False


def _classify_level(
    logs: list[str],
    after_msgs: list[dict],
    before_tokens: int | None = None,
    after_tokens: int | None = None,
    max_tokens: int | None = None,
    before_msgs: list[dict] | None = None,
) -> str:
    """结合日志与结果结构，返回处理程度：顶层 level (+ 兜底内实际方法)。"""
    if after_msgs is None:
        return "ERROR: _compress_context returned None（函数内部出现未捕获异常或缺少 return）"
    joined = "\n".join(logs)

    # 各阶段达标的日志关键词
    stage_a_hit = "Context within limit after oversized-tool trim" in joined
    stage_b_hit = "Context within limit after micro_compact" in joined
    round_success = "Compressed context:" in joined
    entered_fallback = "applying microcompact" in joined
    has_summary = any(
        isinstance(m.get("content"), str) and "[Context Summary]" in m["content"]
        for m in after_msgs
    )

    # 是否仍超限：拿到 token 数才能判断（before_msgs 供结构兜底用）
    still_over = (
        after_tokens is not None and max_tokens is not None and after_tokens > max_tokens
    )

    # 阶段 A 达标（tool_budget 转存+预览就压回去了）
    if stage_a_hit and not stage_b_hit and not round_success and not entered_fallback:
        return "LA 阶段A达标（超大工具转存+预览，未进后续阶段）"

    # 阶段 B 达标（micro_compact 本地清理压回去了）
    if stage_b_hit and not round_success and not entered_fallback:
        prefix = "LA+LB" if stage_a_hit else "LB"
        return f"{prefix} 阶段B达标（micro_compact 本地清理，未调 LLM 摘要）"

    # 轮次摘要成功（或有摘要但未进兜底）
    if not entered_fallback:
        if round_success or has_summary:
            prefix = "LA+" if stage_a_hit else ""
            prefix += "LB+" if stage_b_hit else ""
            return f"{prefix}L1 轮次摘要压缩成功（循环中途达标，未进兜底）"

        # 没走成功返回路径，也没进兜底：可能是"压过但仍超限"，也可能是真未压缩。
        # 靠结果结构判断阶段 A/B 是否实际改动过内容，避免误标成 L0。
        shrank = _detect_local_shrink(before_msgs or [], after_msgs) if before_msgs is not None else False
        if still_over:
            if shrank:
                delta = (
                    f"{before_tokens}→{after_tokens}"
                    if before_tokens is not None else f"{after_tokens}"
                )
                return (
                    f"LX 压过但仍超限（阶段A/B已生效 {delta} tokens，仍 > {max_tokens}；"
                    f"阶段C按轮摘要被禁用，直接返回超限结果）"
                )
            return f"LX 仍超限但未见压缩（after={after_tokens} > {max_tokens}，阶段A/B未改动内容）"
        return "L0 未压缩（未超阈值）"

    # 进了兜底函数
    prefix = ""
    if stage_a_hit:
        prefix += "LA+"
    if stage_b_hit:
        prefix += "LB+"
    entry = "L3 全摘要后仍超限入口" if has_summary else "L2 单轮兜底入口（无法按轮压缩）"
    micro = _deepest_micro_step(joined)
    if micro is None:
        return f"{prefix}{entry} → 进入即达标(未触发子步骤)"
    return f"{prefix}{entry} → {micro}"


class _LogCapture(logging.Handler):
    """临时挂到 compress.run_logger 上，收集本次压缩产生的日志文本。"""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[str] = []

    def emit(self, record):
        self.records.append(record.getMessage())


def _format_summary_calls(calls: list[dict]) -> str:
    """把「发给摘要模型」的每一次调用（prompt 全文 + 返回摘要）渲染成文本。

    阶段 C 每压缩一轮调用一次；未触发阶段 C（如 L0/LA/LB）时列表为空。
    """
    if not calls:
        return "(本次运行未调用摘要模型：未进入阶段 C 按轮摘要，或全部轮次摘要被跳过。)"

    blocks = []
    for i, c in enumerate(calls, start=1):
        blocks.append(
            f"----- 摘要调用 #{i} "
            f"(model={c.get('model')}, temperature={c.get('temperature')}, "
            f"max_tokens={c.get('max_tokens')}) -----"
        )
        blocks.append(">>> 发给摘要模型的文本 (PROMPT):")
        blocks.append(c.get("prompt", ""))
        blocks.append("")
        blocks.append("<<< 摘要模型返回 (RESPONSE):")
        blocks.append(c.get("response", ""))
        blocks.append("")
    return "\n".join(blocks)


def _build_report(desc, model, max_tokens, before_tokens, after_tokens,
                  before_msgs, after_msgs, ts_human, level, summary_calls=None) -> str:
    """组织 before + after 到同一份报告文本；说明写在文件头部。"""
    sep = "=" * 70
    parts = [
        sep,
        "上下文压缩测试报告 (compress.py)",
        sep,
        f"时间       : {ts_human}",
        f"说明(-d)   : {desc if desc else '(未提供)'}",
        f"摘要模型   : {model}",
        f"压缩阈值   : max_tokens={max_tokens}",
        f"Token 变化 : {before_tokens} -> {after_tokens}",
        f"消息条数   : {len(before_msgs)} -> {len(after_msgs)}",
        f"处理程度   : {level}",
        f"摘要调用数 : {len(summary_calls) if summary_calls else 0}",
        sep,
        "",
        # 顺序：压缩前(BEFORE) → 发给摘要模型的文本 → 压缩后(AFTER)
        # 注：注释掉阶段 C 时不会有任何摘要调用，summary_calls 为空列表，
        #     _format_summary_calls 会输出一行说明而非报错，整体流程安全。
        "#" * 70,
        "# 压缩前 (BEFORE)",
        "#" * 70,
        "",
        _format_messages(before_msgs),
        "#" * 70,
        "# 发给摘要模型的文本化上下文 (SUMMARY MODEL INPUT/OUTPUT)",
        "#" * 70,
        "",
        _format_summary_calls(summary_calls or []),
        "#" * 70,
        "# 压缩后 (AFTER)",
        "#" * 70,
        "",
        _format_messages(after_msgs),
    ]
    return "\n".join(parts)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行 compress.py 的上下文压缩并把 before/after 结果存成带时间戳的 txt。"
    )
    parser.add_argument(
        "-d", "--desc", default="",
        help="本次运行的说明（支持中文），会写入结果 txt 的文件头部。",
    )
    parser.add_argument(
        "-m", "--max-tokens", type=int, default=None,
        help="压缩触发阈值（token）。不传则用环境变量 COMPRESS_MAX_TOKENS，默认 1200。",
    )
    parser.add_argument(
        "--model", default=None,
        help="摘要模型。不传则用环境变量 COMPRESS_MODEL，默认 qwen3.6-plus。",
    )
    parser.add_argument(
        "--input", default=None,
        help="手动指定输入 json 文件（如 hermes-chat-logs/01-local.json）；"
             "不传则用内置的多轮示例对话。",
    )
    args = parser.parse_args()

    # 加载消息：优先用 --input 指定的真实日志，否则回退到内置示例
    input_model = None
    if args.input:
        messages, input_model = _load_messages_from_file(args.input)
        print(f"=== 输入来源: {args.input}（{len(messages)} 条消息）===")
    else:
        messages = _sample_messages()

    # 摘要模型优先级：命令行参数 > 环境变量 > 输入文件内的 model > 内置默认
    max_tokens = args.max_tokens if args.max_tokens is not None else int(os.environ.get("COMPRESS_MAX_TOKENS", "1200"))
    model = args.model or os.environ.get("COMPRESS_MODEL") or input_model or "qwen3.6-plus"

    api_key = _load_api_key()
    provider = DashScopeProvider(api_key)

    before_msgs = [dict(m) for m in messages]  # 快照，避免后续引用被改动
    before = compress._estimate_tokens(messages)
    print(f"\n=== 压缩前: {before} tokens | 阈值 max_tokens={max_tokens} | 摘要模型={model} ===\n")

    # 挂上日志捕获，用于判定实际到达的「处理程度」
    capture = _LogCapture()
    compress.run_logger.addHandler(capture)
    try:
        result = await compress._compress_context(
            messages=messages,
            llm_provider=provider,
            llm_model=model,
            max_tokens=max_tokens,
        )
    finally:
        compress.run_logger.removeHandler(capture)

    after = compress._estimate_tokens(result)
    level = _classify_level(
        capture.records, result,
        before_tokens=before, after_tokens=after,
        max_tokens=max_tokens, before_msgs=before_msgs,
    )
    print(f"\n=== 压缩后: {after} tokens（{len(before_msgs)} 条 → {len(result)} 条消息）| 处理程度: {level} ===\n")
    for i, m in enumerate(result):
        role = m.get("role")
        content = m.get("content")
        preview = (content[:120] + "…") if isinstance(content, str) and len(content) > 120 else content
        tag = " [SUMMARY]" if isinstance(content, str) and "[Context Summary]" in content else ""
        print(f"[{i}] {role}{tag}: {preview!r}")

    # ── 保存结果：每次运行新建一个带时间戳的 txt，before+after 同文件 ──
    now = datetime.now()
    ts_file = now.strftime("%Y%m%d_%H%M%S")
    ts_human = now.strftime("%Y-%m-%d %H:%M:%S")
    out_dir = os.path.join(_HERE, "runs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"compress_{ts_file}.txt")

    report = _build_report(
        desc=args.desc, model=model, max_tokens=max_tokens,
        before_tokens=before, after_tokens=after,
        before_msgs=before_msgs, after_msgs=result, ts_human=ts_human,
        level=level, summary_calls=provider.summary_calls,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n结果已保存: {out_path}（含 {len(provider.summary_calls)} 次摘要模型调用文本）")

    # ── 原汁原味压缩结果（仅 --input 模式）：保存到 compressed/<源文件名>_<时间戳>.json ──
    # 文件名主干与 runs/ 下的时间戳完全一致，两份文件一眼可对应。
    if args.input:
        input_stem = os.path.splitext(os.path.basename(args.input))[0]
        compressed_dir = os.path.join(_HERE, "compressed")
        os.makedirs(compressed_dir, exist_ok=True)
        compressed_path = os.path.join(compressed_dir, f"{input_stem}_{ts_file}.json")
        with open(compressed_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"压缩后原始 JSON: {compressed_path}")


if __name__ == "__main__":
    asyncio.run(main())
