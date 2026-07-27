#!/usr/bin/env python3
# 指定使用 python3 解释器执行本脚本（用于直接 ./compress.py 运行的场景）

import os                # 操作系统相关工具，这里用于拼接日志文件路径
import copy              # 深拷贝工具，用于在压缩时避免原地修改输入消息
import json              # JSON 序列化，用于把工具调用信息拼进摘要提示词
import tiktoken          # OpenAI 的分词库，用于精确统计 token 数量
from configs.config import LOG_LEVEL, RUN_LOG_FILE, ROOT_PATH  # 从项目配置读取日志级别、日志文件名、根路径
from utils.log_utils import get_logger                          # 项目统一的日志器构造函数

# 构造运行日志器：名字为 run_logger，输出到 ROOT_PATH/RUN_LOG_FILE，级别由配置决定
run_logger = get_logger("run_logger", os.path.join(ROOT_PATH, RUN_LOG_FILE), log_level=LOG_LEVEL)


def _estimate_tokens(messages) -> int:
    """使用 tiktoken 精确统计消息历史的 token 数量

    使用 cl100k_base 编码器（兼容 GPT-4 / Claude / M2 等主流模型）
    """
    try:
        # 获取 cl100k_base 编码器（GPT-4 及多数现代模型使用的编码）
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # 兜底：若 tiktoken 初始化失败，退化到简单的字符估算
        return _estimate_tokens_fallback(messages)

    total_tokens = 0  # token 累加器

    # 逐条遍历消息，累计各字段占用的 token
    for msg in messages:
        # ── 统计 content 字段 ──────────────────────────────────
        if isinstance(msg.get("content"), str):
            # content 为纯字符串：直接编码后计数
            total_tokens += len(encoding.encode(msg["content"]))
        elif isinstance(msg.get("content"), list):
            # content 为多模态块列表（文本块 / 图片块等）
            for block in msg["content"]:
                if isinstance(block, dict):
                    # 将 dict 转字符串再计数（旧方案，已弃用）
                    # total_tokens += len(encoding.encode(str(block)))
                    if block.get("type") == "image_url":
                        # 图片内容按固定 token 估算，避免把图片 URL/base64 算爆
                        total_tokens += 2000
                    elif block.get("type") == "text":
                        # 文本块按实际文本编码计数
                        total_tokens += len(encoding.encode(block.get("text", "")))

        # ── 统计 reasoning_content（模型思维链）────────────────
        if "reasoning_content" in msg:
            total_tokens += len(encoding.encode(msg["reasoning_content"]))

        # ── 统计 tool_calls（工具调用结构）────────────────────
        if "tool_calls" in msg:
            # 直接把整个 tool_calls 结构转字符串编码计数（含函数名、参数等）
            total_tokens += len(encoding.encode(str(msg["tool_calls"])))

        # 每条消息的元数据开销（role/分隔符等），约 4 个 token
        total_tokens += 4
    return total_tokens


def _estimate_tokens_fallback(messages) -> int:
    """兜底的 token 估算方法（当 tiktoken 不可用时）"""
    total_chars = 0  # 字符累加器
    for msg in messages:
        # 累计 content 字段的字符数
        if isinstance(msg.get("content"), str):
            total_chars += len(msg["content"])
        elif isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict):
                    total_chars += len(str(block))

        # 累计思维链字符数
        if "reasoning_content" in msg:
            total_chars += len(msg["reasoning_content"])

        # 累计工具调用结构的字符数
        if "tool_calls" in msg:
            total_chars += len(str(msg["tool_calls"]))

    # 粗略换算：平均 2.5 个字符 ≈ 1 个 token
    return int(total_chars / 2.5)


def _generic_truncate(content: str, max_tokens: int) -> str:
    """
    通用截断策略。
    把过长的字符串截断到大约 max_tokens 对应的长度，并附上截断说明。
    """
    # 用 2.5 字符/token 的比例把 token 预算换算为字符预算
    max_chars = max_tokens * 2.5

    # 长度在预算内，无需截断，原样返回
    if len(content) <= max_chars:
        return content

    # 只保留预算的 80%，剩余 20% 留给截断提示，避免截断后仍然超预算
    keep_chars = int(max_chars * 0.8)

    # 拼接：保留的头部内容 + 截断说明（告知被删掉了多少字符）
    truncated = (
            content[:keep_chars] +
            f"\n\n... [TRUNCATED: {len(content) - keep_chars} characters] ..."
    )

    return truncated


def _truncate_oversized_tools(
        messages: list[dict],
        max_tool_tokens: int = 20000  # 单个工具响应允许的最大 token 数
) -> list[dict]:
    """
    截断超大的工具响应内容。
    遍历所有 tool 消息，只有超过阈值的才截断，其余原样保留。
    """
    truncated_messages = []  # 输出的新消息列表

    for msg in messages:
        if msg.get("role") == "tool":
            # 取出工具响应内容并估算其 token 数
            content = msg.get("content", "")
            estimated_tokens = _estimate_tokens([msg])

            if estimated_tokens > max_tool_tokens:
                # 超过阈值 → 执行截断
                truncated_content = _generic_truncate(content, max_tool_tokens)

                # 记录截断日志，便于排查
                run_logger.info(
                    f"Tool response too large ({estimated_tokens} tokens), "
                    f"truncating to {max_tool_tokens} tokens, "
                    f"return {len(truncated_content)} chars"
                )

                # 复制原消息并替换 content，保留其它字段（role/tool_call_id 等）
                truncated_msg = msg.copy()
                truncated_msg["content"] = truncated_content
                truncated_messages.append(truncated_msg)
            else:
                # 未超阈值：原样保留
                truncated_messages.append(msg)
        else:
            # 非工具消息：原样保留
            truncated_messages.append(msg)

    return truncated_messages


def _trim_oversized_tools_via_budget(messages: list[dict]) -> list[dict]:
    """用 tool_budget 的「转存磁盘 + 预览」策略剪裁超大工具结果。

    复用同目录下 ctx-compress-opt/tool_budget.py 的两级预算：
      - 第一级：单条工具结果超阈值 → 整块转存磁盘，content 换成预览 + 文件路径；
      - 第二级：同一批并行工具结果合计超预算 → 从大到小挑几条转存替换，直到降回预算内。
    完整内容写盘不丢失，发给模型的只保留一段预览，从而在不硬截断的前提下压缩上下文。

    为保持 `_compress_context()` 接口不变（不新增 storage_dir / state / config 参数），
    这里在函数内部自行推导转存目录（ROOT_PATH/tool-budget-cache）并使用一次性 state。
    任何异常都降级为原样返回，绝不因剪裁失败而中断整个压缩流程。
    """
    # 惰性导入：优先加载本目录（ctx-compress-opt）下的 tool_budget 副本。
    # 该模块只依赖标准库，放到函数内导入，避免文件缺失时拖垮整个模块加载。
    try:
        import sys
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        from tool_budget import apply_budget, create_state, BudgetConfig
    except Exception as e:
        run_logger.info(f"tool_budget unavailable, skip oversized-tool trim: {e}")
        return messages

    # 转存目录：放在项目根下的固定目录，跨调用可复用同一批文件（内容确定、写盘不覆盖）
    storage_dir = os.path.join(ROOT_PATH, "tool-budget-cache")

    try:
        trimmed, _state, stats = apply_budget(
            storage_dir,
            messages,
            state=create_state(),
            config=BudgetConfig(),
        )
    except Exception as e:
        # 剪裁失败：宁可超预算也不丢内容，原样返回交给后续按轮压缩处理
        run_logger.info(f"tool_budget apply failed, keep original messages: {e}")
        return messages

    if stats.newly_persisted or stats.reapplied:
        run_logger.info(
            f"Oversized tool trim: persisted {stats.newly_persisted}, "
            f"reapplied {stats.reapplied}, shed ~{stats.shed_chars} chars"
        )
    return trimmed


def _run_micro_compact(messages: list[dict], max_tokens: int) -> list[dict]:
    """用 micro_compact 的阈值触发渐进式清理压缩上下文（本地、无 LLM 调用）。

    复用同目录下 ctx-compress-opt/micro_compact.py：掐头去尾 → 清空旧 tool result
    → 瘦身旧 tool call 参数 → 删旧 reasoning → 占位符兜底，每步后重估、达标即返回。
    只作用于副本，不触碰系统消息 / 运行时标记 / 已压缩摘要区与当前 session。

    为保持 `_compress_context()` 接口不变，这里不新增参数；任何异常都降级为原样返回，
    绝不因清理失败而中断整个压缩流程。
    """
    try:
        import sys
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        from micro_compact import micro_compact
    except Exception as e:
        run_logger.info(f"micro_compact unavailable, skip: {e}")
        return messages

    try:
        return micro_compact(messages, max_tokens)
    except Exception as e:
        run_logger.info(f"micro_compact failed, keep original messages: {e}")
        return messages


def _handle_single_round_overflow_v1(
    messages: list[dict],
    max_tokens: int
) -> list[dict]:
    """
    处理单轮对话超限的情况。（旧版本 v1，按平均预算分配截断 tool）
    策略：
    - 保留所有 user / assistant 消息（内容不变）
    - 仅对 tool 消息进行激进截断
    - 严格保持原始对话顺序
    """
    run_logger.warning("Single round overflow, applying aggressive truncation")

    # ── 1. 统计所有 tool 消息的数量，用于平均分配 token ──────────────────
    tool_count = sum(1 for m in messages if m.get("role") == "tool")

    # ── 2. 先粗估非 tool 消息占用的 token，计算留给 tool 的预算 ──────────
    non_tool_msgs = [m for m in messages if m.get("role") != "tool"]  # 非 tool 消息集合
    non_tool_tokens = _estimate_tokens(non_tool_msgs)                 # 非 tool 消息 token 占用
    remaining_tokens = max_tokens - non_tool_tokens                   # 留给 tool 消息的预算

    if remaining_tokens < 500:
        # 非 tool 消息本身就快把预算吃满了，没法再给 tool 分配空间
        run_logger.error(
            f"Non-tool messages already consume {non_tool_tokens} tokens "
            f"(limit={max_tokens}), cannot allocate budget for tool messages"
        )
        # 实在放不下，只返回非 tool 消息，但顺序保持原样
        return non_tool_msgs

    # ── 3. 计算每条 tool 消息的 token 上限（激进截断） ───────────────────
    if tool_count > 0:
        tokens_per_tool = remaining_tokens // tool_count  # 平均分配预算给每条 tool
        tokens_per_tool = min(tokens_per_tool, 3000)      # 但单条最多 3000 tokens
    else:
        tokens_per_tool = 3000  # 没有 tool 消息时给个默认值（实际不会用到）

    run_logger.debug(
        f"tool_count={tool_count}, non_tool_tokens={non_tool_tokens}, "
        f"remaining={remaining_tokens}, tokens_per_tool={tokens_per_tool}"
    )

    # ── 4. 按原始顺序遍历，按角色决定是否截断 ────────────────────────────
    result: list[dict] = []
    for msg in messages:
        role = msg.get("role")

        if role == "tool":
            # 仅截断 tool 消息的 content
            content = msg.get("content", "")
            truncated_content = _generic_truncate(content, tokens_per_tool)
            truncated_msg = msg.copy()
            truncated_msg["content"] = truncated_content
            result.append(truncated_msg)
        else:
            # user / assistant / system 原样保留
            result.append(msg)

    # ── 5. 记录截断效果 ───────────────────────────────────────────────────
    original_tokens = _estimate_tokens(messages)  # 截断前 token
    final_tokens = _estimate_tokens(result)       # 截断后 token
    run_logger.info(
        f"Aggressive truncation: {original_tokens} → {final_tokens} tokens "
        f"(limit={max_tokens}, tool_msgs={tool_count}, "
        f"tokens_per_tool={tokens_per_tool})"
    )

    return result


def _handle_single_round_overflow_v2(
    messages: list[dict],
    max_tokens: int
) -> list[dict]:
    """
    处理单轮对话超限。（旧版本 v2，按优先级逐级截断）

    压缩策略（按优先级依次执行）：
    1. 删除空内容的 assistant/tool 消息
    2. 删除仅用于工具调用的 assistant 中间消息（带 tool_calls，且 content 很短）
    3. 截断 tool 消息 content
    4. 截断带 tool_calls 的 assistant 消息 content
    5. 截断普通 assistant 消息 content
    6. 优先删最早的非 system 消息

    目标：
    - 严格保证返回结果 token <= max_tokens
    - 尽量保留最终用户可见的对话语义
    - 尽量保持原始顺序
    """
    run_logger.warning("Single round overflow, applying aggressive truncation")

    # 内部小工具：判断给定消息列表是否已满足 token 限制
    def fits(msgs: list[dict]) -> bool:
        return _estimate_tokens(msgs) <= max_tokens

    # 内部小工具：浅拷贝每条消息（避免原地修改原始 messages）
    def clone_msgs(msgs: list[dict]) -> list[dict]:
        return [m.copy() for m in msgs]

    # 先复制，避免原地修改
    result = clone_msgs(messages)
    original_tokens = _estimate_tokens(result)  # 记录初始 token

    # 已经不超限，直接返回
    if original_tokens <= max_tokens:
        return result

    run_logger.warning(
        f"Overflow detected: original_tokens={original_tokens}, limit={max_tokens}"
    )

    # ------------------------------------------------------------
    # Step 1: 处理 tool_calls，将 execute_code 和 exec 的 arguments 置空
    # ------------------------------------------------------------
    compacted = []
    for msg in result:
        # 清空思维链（reasoning_content 通常很长且对最终结果影响小）
        if "reasoning_content" in msg:
            msg["reasoning_content"] = ""

        role = msg.get("role")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")

        # 针对带有工具调用的 assistant 消息，清空高开销工具的参数
        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            new_msg = msg.copy()

            new_tool_calls = []
            for call in tool_calls:
                new_call = call.copy()
                function = new_call.get("function")

                if isinstance(function, dict):
                    new_function = function.copy()
                    func_name = new_function.get("name")

                    # 把代码/命令类参数清空，仅保留结构（大幅减少 token）
                    if func_name == "execute_code":
                        new_function["arguments"] = '{"code": ""}'
                    elif func_name == "exec":
                        new_function["arguments"] = '{"command": ""}'

                    new_call["function"] = new_function

                new_tool_calls.append(new_call)

            new_msg["tool_calls"] = new_tool_calls

            compacted.append(new_msg)
            continue

        compacted.append(msg)

    result = compacted
    if fits(result):
        run_logger.info(
            f"Overflow resolved by removing empty assistant/tool messages: "
            f"{original_tokens} -> {_estimate_tokens(result)}"
        )
        return result

    # ------------------------------------------------------------
    # Step 3: 截断 tool 消息（每条压到约 1000 tokens）
    # ------------------------------------------------------------
    result = clone_msgs(result)
    for msg in result:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                msg["content"] = _generic_truncate(content, 1000)

    if fits(result):
        run_logger.info(
            f"Overflow resolved by truncating tool messages: "
            f"{original_tokens} -> {_estimate_tokens(result)}"
        )
        return result

    # ------------------------------------------------------------
    # Step 4: 截断带 tool_calls 的 assistant 消息（content 压到约 200 tokens）
    # ------------------------------------------------------------
    result = clone_msgs(result)
    for msg in result:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                msg["content"] = _generic_truncate(content, 200)

    if fits(result):
        run_logger.info(
            f"Overflow resolved by truncating assistant tool-call messages: "
            f"{original_tokens} -> {_estimate_tokens(result)}"
        )
        return result

    # ------------------------------------------------------------
    # Step 5: 截断普通 assistant 消息，优先截断较早的（压到约 300 tokens）
    # ------------------------------------------------------------
    result = clone_msgs(result)
    for i, msg in enumerate(result):
        if msg.get("role") == "assistant" and not msg.get("tool_calls"):
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 300:
                msg["content"] = _generic_truncate(content, 300)

    if fits(result):
        run_logger.info(
            f"Overflow resolved by truncating assistant final messages: "
            f"{original_tokens} -> {_estimate_tokens(result)}"
        )
        return result

    # ------------------------------------------------------------
    # Step 6: 极限兜底
    # 尽量只保留：
    # 如果还超限，只保留 system + 最近几条
    # 优先删最早的非 system 消息
    # ------------------------------------------------------------

    # 循环删除最早的非 system 消息，直到满足限制或只剩 2 条
    while _estimate_tokens(result) > max_tokens and len(result) > 2:
        # 优先删最早的非 system 消息
        removed = False
        for i, msg in enumerate(result):
            if msg.get("role") != "system":
                result.pop(i)
                removed = True
                break
        if not removed:
            # 只剩 system 消息，无法再删，跳出
            break

    run_logger.info(
        f"Semantic compression: final_tokens={_estimate_tokens(result)}, limit={max_tokens}"
    )
    return result


# 可被压缩（其结果/参数可清空或截断）的工具名单
COMPACTABLE_TOOLS = {
    "execute_code",
    "exec",
    "bash",
    "shell",
    "read_file",
    "write_file",
    "edit_file",
    "grep",
    "glob",
    "rag_retrieval",
    "file_parser",
    "precise_search",
    "fetch_web_page",
    "academic_search",
    "web_search",
    "web_fetch",
}

CLEARED_TOOL_RESULT = "[Old tool result content cleared]"  # 工具结果被清空后的占位文本
KEEP_RECENT_TOOL_RESULTS = 3                               # 至少保留最近 N 条工具结果不清空


def _collect_compactable_tool_ids(messages: list[dict]) -> list[str]:
    """按出现顺序收集所有可压缩工具的 tool_call ID。"""
    ids: list[str] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            # 遍历该 assistant 消息发起的每个工具调用
            for call in msg.get("tool_calls") or []:
                func_name = (call.get("function") or {}).get("name", "")  # 工具名
                call_id = call.get("id", "")                              # 工具调用 ID
                # 仅收集在白名单内且有 ID 的调用
                if func_name in COMPACTABLE_TOOLS and call_id:
                    ids.append(call_id)
    return ids


def _clear_tool_results(messages: list[dict], clear_set: set[str]) -> list[dict]:
    """将 clear_set 中 ID 对应的 tool 消息内容替换为占位符，同时保留消息结构。"""
    result = []
    for msg in messages:
        # 命中条件：是 tool 消息 + ID 在待清除集合 + 尚未被清除过
        if (
                msg.get("role") == "tool"
                and msg.get("tool_call_id") in clear_set
                and msg.get("content") != CLEARED_TOOL_RESULT
        ):
            # 用解包生成新 dict，替换 content 为占位符（不改动原消息）
            msg = {**msg, "content": CLEARED_TOOL_RESULT}
        result.append(msg)
    return result


def _clear_tool_call_arguments(messages: list[dict]) -> list[dict]:
    """
    清空可压缩工具的 arguments。
    execute_code → {"code": ""}，exec/bash/shell → {"command": ""}，其它 → {}。
    """
    # 不同工具对应的"空参数"模板
    EMPTY_ARGS: dict[str, str] = {
        "execute_code": '{"code": ""}',
        "exec": '{"command": ""}',
        "bash": '{"command": ""}',
        "shell": '{"command": ""}',
    }
    result = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            new_calls = []
            for call in msg["tool_calls"]:
                func = call.get("function") or {}
                name = func.get("name", "")
                if name in COMPACTABLE_TOOLS:
                    # 命中白名单：用对应空参数模板替换，找不到就用 "{}"
                    call = {
                        **call,
                        "function": {**func, "arguments": EMPTY_ARGS.get(name, "{}")},
                    }
                new_calls.append(call)
            msg = {**msg, "tool_calls": new_calls}
        result.append(msg)
    return result


def _find_deletable_tool_groups(messages: list[dict]) -> list[list[int]]:
    """
    识别所有可原子删除的"工具调用组"：每组 = [assistant_idx, tool_idx1, ...]。

    一组 = 一条带 tool_calls 的 assistant 消息 + 紧随其后属于它的 tool 消息。
    删除时必须整组删除；只删 assistant 或只删 tool 都会产生孤立的工具结果。
    """
    groups: list[list[int]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # 收集该 assistant 发起的所有工具调用 ID
            tool_ids = {tc.get("id") for tc in msg.get("tool_calls", [])}
            group = [i]      # 组内首个索引为 assistant 本身
            j = i + 1
            # 紧随其后的 tool 消息都属于这个 assistant
            while j < len(messages) and messages[j].get("role") == "tool":
                # 只把 ID 匹配的 tool 归入本组
                if messages[j].get("tool_call_id") in tool_ids:
                    group.append(j)
                j += 1
            groups.append(group)
        i += 1
    return groups


def _load_micro_compact_module():
    """惰性加载同目录 ctx-compress-opt/micro_compact.py（阶段 B）模块本身，
    以便复用它内部的工具结果信息化摘要 / 去重 / 参数截断 / reasoning 清理等 helper。

    加载失败返回 None，调用方降级为本地占位符清理策略，绝不中断压缩。
    """
    try:
        import sys
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        import micro_compact as mc
        return mc
    except Exception as e:
        run_logger.info(f"micro_compact helpers unavailable in single-round overflow: {e}")
        return None


def _handle_single_round_overflow(
        messages: list[dict],
        max_tokens: int,
) -> list[dict]:
    """
    处理单轮对话上下文超限。（复用阶段 B 的工具结果信息化摘要策略）

    压缩策略（按优先级，每步后重估 token，达标即返回）：
    1. [复用阶段 B] 去重 + 把较早的工具结果替换为「信息化摘要」——保留工具名称、
       关键参数、执行结果、报错和文件路径（如 "[terminal] ran `npm test` -> exit 0"），
       而非无信息量的占位符；最近若干条工具结果不动。
    2. 清除历史 reasoning_content（思维链，占用大且对结果影响小）。
    3. [复用阶段 B] JSON 感知地缩减过长的工具调用参数（parse→截断长字符串字段→
       重新序列化，保证参数仍是合法 JSON）。
    4. 截断仍然超大的工具结果（硬截断，保留残余内容而非删除）。
    5. [最后手段] 原子删除最早且完整的工具调用组（assistant tool_calls + 对应
       tool result 整组删除），避免破坏工具调用协议；尽量保留最近工具组。

    micro_compact（阶段 B）不可用时，降级为旧的占位符清理策略，绝不中断压缩。
    """
    run_logger.warning("Single round overflow, applying microcompact")

    # 内部小工具：判断是否已满足 token 限制
    def fits(msgs: list[dict]) -> bool:
        return _estimate_tokens(msgs) <= max_tokens

    # 深拷贝，彻底避免影响调用方的数据（含嵌套的 tool_calls）
    result = copy.deepcopy(messages)
    original_tokens = _estimate_tokens(result)

    # 未超限直接返回
    if original_tokens <= max_tokens:
        return result

    run_logger.warning(
        f"Overflow: {original_tokens} tokens > limit {max_tokens}"
    )

    mc = _load_micro_compact_module()

    if mc is not None:
        # ── 复用阶段 B：先算保护范围（保护头部 system/运行时标记/已压缩摘要区，
        #    以及末尾最近 KEEP_RECENT_MESSAGES 条消息），只在可清理区间内操作。──
        try:
            lo, tool_hi, reason_hi = mc._clearable_ranges(result, mc.KEEP_RECENT_MESSAGES)
            call_id_map = mc._build_call_id_map(result)

            # Step 1: 去重 + 信息化摘要（较早的工具结果 → 一行有信息量摘要）
            if tool_hi > lo:
                mc._deduplicate_tool_results(result, lo, tool_hi)
                mc._summarize_old_tool_results(result, lo, tool_hi, call_id_map)
                if fits(result):
                    run_logger.info(
                        f"Resolved by summarizing old tool results (Stage-B reuse): "
                        f"{original_tokens} -> {_estimate_tokens(result)}"
                    )
                    return result

            # Step 2: 清除历史 reasoning_content（仅历史轮，当前轮保留）
            if reason_hi > lo:
                mc._clear_reasoning(result, lo, reason_hi)
                if fits(result):
                    run_logger.info(
                        f"Resolved by clearing reasoning_content: "
                        f"{original_tokens} -> {_estimate_tokens(result)}"
                    )
                    return result

            # Step 3: JSON 感知地缩减过长的工具调用参数
            if tool_hi > lo:
                mc._truncate_tool_call_args(result, lo, tool_hi)
                if fits(result):
                    run_logger.info(
                        f"Resolved by truncating tool call arguments (JSON-aware): "
                        f"{original_tokens} -> {_estimate_tokens(result)}"
                    )
                    return result
        except Exception as e:
            # 复用阶段 B 出现意外：不中断，继续走下面的硬截断 / 原子删除兜底
            run_logger.info(
                f"Stage-B reuse failed, falling back to local truncation/deletion: {e}"
            )
    else:
        # ── 降级：micro_compact 不可用时，沿用旧的占位符清理策略 ──
        compactable_ids = _collect_compactable_tool_ids(result)
        keep_recent = max(1, KEEP_RECENT_TOOL_RESULTS)
        keep_set = set(compactable_ids[-keep_recent:])
        clear_set = set(compactable_ids) - keep_set
        if clear_set:
            result = _clear_tool_results(result, clear_set)
            if fits(result):
                run_logger.info(
                    f"Resolved by clearing old tool results "
                    f"(kept {len(keep_set)}, cleared {len(clear_set)}): "
                    f"{original_tokens} -> {_estimate_tokens(result)}"
                )
                return result

        for msg in result:
            if "reasoning_content" in msg:
                msg["reasoning_content"] = ""
        if fits(result):
            run_logger.info(
                f"Resolved by clearing reasoning_content: "
                f"{original_tokens} -> {_estimate_tokens(result)}"
            )
            return result

        result = _clear_tool_call_arguments(result)
        if fits(result):
            run_logger.info(
                f"Resolved by clearing tool call arguments: "
                f"{original_tokens} -> {_estimate_tokens(result)}"
            )
            return result

    # ------------------------------------------------------------------
    # Step 4: 硬截断仍然超大的工具结果（不区分工具类型，保留残余而非删除）。
    #         已被信息化摘要的一行内容长度远低于阈值，会被 len() 判断自然跳过。
    # ------------------------------------------------------------------
    tool_tokens = int(max_tokens / 5)  # 单条 tool 允许保留的 token 上限
    for msg in result:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if isinstance(content, str) and content != CLEARED_TOOL_RESULT and len(content) > tool_tokens:
                msg["content"] = _generic_truncate(content, tool_tokens)

    if fits(result):
        run_logger.info(
            f"Resolved by hard-truncating tool results: "
            f"{original_tokens} -> {_estimate_tokens(result)}"
        )
        return result

    # ------------------------------------------------------------------
    # Step 5: 最后手段——原子删除最早且完整的工具调用组
    #         （assistant tool_calls + 对应 tool result 整组删除），
    #         保持工具调用协议合法，最早的组先删、尽量保留最近工具组。
    # ------------------------------------------------------------------
    while not fits(result):
        groups = _find_deletable_tool_groups(result)
        if not groups:
            # 已无可删除的工具调用组，无法继续压缩
            run_logger.warning(
                f"Cannot reduce further, remaining: {_estimate_tokens(result)} tokens"
            )
            break
        # 删除最早的那组（索引最小）
        earliest_group = groups[0]
        indices_to_delete = set(earliest_group)
        # 通过枚举过滤掉待删索引，重建消息列表
        result = [msg for i, msg in enumerate(result) if i not in indices_to_delete]
        run_logger.info(
            f"Deleted tool group ({len(earliest_group)} messages), "
            f"remaining: {_estimate_tokens(result)} tokens"
        )

    run_logger.info(
        f"Final: {_estimate_tokens(result)} tokens (limit {max_tokens})"
    )
    return result


async def _compress_context(
    messages,
    llm_provider,
    llm_model,
    max_tokens
) -> list[dict]:
    """
    压缩上下文：
    1. 保留系统提示词
    2. 从前往后压缩（删除）完整轮次，直到总 token 数 < max_tokens
    3. 对超大工具响应进行智能截断
    """

    # 先估算当前总 token，仅用于日志与后续阶段的触发判断。
    # 阶段 A（超大工具响应转存）无条件执行：它是发送前的保护性处理，
    # 与当前是否超限无关——超限时转存可直接压回阈值，未超限时也能提前
    # 把单条巨型工具结果换成预览，避免后续轮次积累后才超限。
    current_tokens = _estimate_tokens(messages)
    if current_tokens > max_tokens:
        run_logger.info(f"Context too large ({current_tokens} tokens), limit={max_tokens}, trimming messages")
    else:
        run_logger.info(f"Context within limit ({current_tokens} tokens), running phase A anyway")

    # ##########################################################################
    # ## 阶段 A：超大工具响应剪裁（转存磁盘 + 保留预览）                       ##
    # ## ------------------------------------------------------------------- ##
    # ## 【单独测试提示】想跳过本阶段：把下面到「阶段 A 结束」之间整段注释掉。 ##
    # ##########################################################################
    # 无条件执行：完整内容写盘不丢失，发给模型的只留预览。
    # 未超限时可直接返回（阶段 B/C 不需要跑）；超限时继续走后续压缩。
    messages = _trim_oversized_tools_via_budget(messages)

    # 重新估算：若已回落到阈值内（或本来就未超限），直接返回，跳过后续所有压缩
    current_tokens = _estimate_tokens(messages)
    if current_tokens <= max_tokens:
        run_logger.info(
            f"Context within limit after oversized-tool trim ({current_tokens} tokens), no further trimming needed"
        )
        return messages
    # ## ===================== 阶段 A 结束 ================================== ##





    # # ##########################################################################
    # # ## 阶段 B：阈值触发的渐进式微压缩 micro_compact（本地清理，无 LLM 调用） ##
    # # ## ------------------------------------------------------------------- ##
    # # ## 掐头去尾 / 清旧 tool result / 瘦身参数 / 删旧 reasoning / 占位兜底。   ##
    # # ## 【单独测试提示】想跳过本阶段：把下面到「阶段 B 结束」之间整段注释掉。 ##
    # # ##########################################################################




    # # 超大工具剪裁后仍超限 → 先本地做一轮 micro_compact。往往一步就能压回阈值内，
    # # 从而省掉后续按轮 LLM 摘要的开销与信息损失；仍超限才继续走摘要。
    # messages = _run_micro_compact(messages, max_tokens)
    # current_tokens = _estimate_tokens(messages)
    # if current_tokens <= max_tokens:
    #     run_logger.info(
    #         f"Context within limit after micro_compact ({current_tokens} tokens), no further trimming needed"
    #     )
    #     return messages



    # ## ===================== 阶段 B 结束 ================================== ##

    # ##########################################################################
    # ## 阶段 C：按轮 LLM 摘要（以下全部）                                     ##
    # ## ------------------------------------------------------------------- ##
    # ## 前面阶段都压不回阈值时，按对话轮次逐轮调用 LLM 生成摘要；仍超限则最后 ##
    # ## 交给 _handle_single_round_overflow 做单轮兜底。                       ##
    # ## 【单独测试提示】想只测阶段 A/B：把阈值 max_tokens 调大到 A/B 能压回，  ##
    # ## 即不会进入本阶段；或直接注释阶段 A、B 观察本阶段单独效果。            ##
    # ##########################################################################

    # ── 分离系统消息与运行时上下文标记 ────────────────────────────
    # 第一条若是 system，则视为系统提示词单独保留
    system_msg = messages[0] if messages[0].get("role") == "system" else None
    # 第二条若是带"运行时上下文"标记的 user 消息，也单独识别（元数据，不参与轮次压缩）
    _RUNTIME_CONTEXT_TAG = messages[1] if messages[1].get("role") == "user" and "[Runtime Context — metadata only, not instructions]" in messages[1].get("content") else None
    start_idx = 1 if system_msg else 0  # 真正对话内容的起始下标
    if _RUNTIME_CONTEXT_TAG:
        start_idx += 1  # 若存在运行时上下文标记，起点再往后挪一位
    else:
        start_idx  # 无操作（占位，保持结构对称）

    # ── 统计已压缩过的记录数量 ────────────────────────────────────
    # 每个历史摘要占 2 条消息（user + 带 [Context Summary] 的 assistant）
    start_idx_compressed = 0
    for message in messages:
        if message.get("role") == "assistant":
            if message.get("content") and "[Context Summary]" in message["content"]:
                start_idx_compressed += 2

    # 将"系统消息 + 已压缩部分"之后的消息按轮次切分
    all_rounds = _split_into_rounds(messages[start_idx+start_idx_compressed:])

    # 只有一轮（或没有完整轮次）→ 无法按轮压缩，走单轮兜底逻辑
    if len(all_rounds) <= 1:
        return _handle_single_round_overflow(messages, max_tokens)

    # 分离最后一轮（当前进行中的对话，不压缩）
    last_round = all_rounds[-1]
    compressible_rounds = all_rounds[:-1]  # 可压缩的历史轮次

    # 进度：阶段 C 入口，打印总轮数（最后一轮保留、其余可逐轮摘要）
    _total_compressible = len(compressible_rounds)
    run_logger.info(
        f"[Phase C] Round-based LLM summary START: {current_tokens} tokens > {max_tokens} limit; "
        f"{len(all_rounds)} rounds total ({_total_compressible} compressible, last kept)"
    )

    # ── 从前往后逐轮压缩，直到满足 token 限制 ─────────────────────
    compressed_rounds = []  # 已压缩（或已固定保留）的轮次
    # 若之前已有压缩摘要（start_idx_compressed > 0），把这段历史摘要区间原样纳入结果头部。
    # 摘要区位于 messages[start_idx : start_idx + start_idx_compressed]。
    # 注意：守卫必须是 "> 0" 而非 ">= start_idx" —— 后者在"无 system 消息
    # (start_idx=0) 且无历史摘要 (start_idx_compressed=0)"时会因 0>=0 误触发，
    # 把第 0 条消息（当轮 user 问题）重复搬入结果，导致压缩后出现两条相同的 user。
    if start_idx_compressed > 0:
        compressed_rounds.append(messages[start_idx:start_idx + start_idx_compressed])

    uncompressed_rounds = compressible_rounds.copy()  # 尚未压缩的轮次队列

    while uncompressed_rounds:
        # 构建当前候选结果（系统消息 + 已压缩轮次 + 未压缩轮次 + 最后一轮）
        candidate_messages = []
        if system_msg:
            candidate_messages.append(system_msg)

        # 添加已压缩的轮次
        for round_msgs in compressed_rounds:
            candidate_messages.extend(round_msgs)

        # 添加未压缩的轮次
        for round_msgs in uncompressed_rounds:
            candidate_messages.extend(round_msgs)

        # 添加最后一轮次
        candidate_messages.extend(last_round)

        # 计算当前候选的 token 数
        candidate_tokens = _estimate_tokens(candidate_messages)

        # 如果已经满足限制，返回结果
        if candidate_tokens <= max_tokens:
            run_logger.info(
                f"Compressed context: {current_tokens} → {candidate_tokens} tokens "
                f"({len(compressed_rounds)} rounds compressed, {len(uncompressed_rounds)} rounds kept)"
            )
            return candidate_messages

        # 否则，压缩最早的一轮
        if not uncompressed_rounds:
            # 所有轮次都已压缩，但仍超限
            # 应该不会走到这个判断里
            run_logger.info(
                f"Cannot compress further: {candidate_tokens} tokens still exceeds limit"
            )
            return candidate_messages

        # 取出最早的一轮进行压缩
        first_round = uncompressed_rounds.pop(0)

        # 进度：当前压缩到第几轮（1-based），还剩多少轮、当前候选 token
        _round_no = _total_compressible - len(uncompressed_rounds)
        run_logger.info(
            f"[Phase C] Summarizing round {_round_no}/{_total_compressible} "
            f"({len(first_round)} msgs in round); candidate now {candidate_tokens} tokens, "
            f"{len(uncompressed_rounds)} rounds still queued"
        )

        # 压缩这一轮
        if _is_complete_round(first_round):
            # 完整轮次 → 调用 LLM 生成摘要
            summary = await _summarize_round(first_round, llm_provider, llm_model)

            if summary is None:
                # 摘要失败 / 结果为空 / 结果不可用：不使用兜底文本替换整轮，
                # 直接保留该轮原始消息（日志已在 _summarize_round 内记录）。
                # 该轮已从 uncompressed_rounds 弹出并放入 compressed_rounds，
                # 因此本次压缩循环不会再重复摘要同一轮。
                run_logger.info(
                    f"[Phase C] Round {_round_no}/{_total_compressible}: summary unusable, "
                    f"keeping original round"
                )
                compressed_rounds.append(first_round)
            else:
                run_logger.info(
                    f"[Phase C] Round {_round_no}/{_total_compressible}: summarized OK "
                    f"({len(summary)} chars)"
                )
                # 取出该轮的原始用户问题（保留问题，压缩回答）
                user_msg = next((m for m in first_round if m.get("role") == "user"), None)

                # 用"原始用户问题 + 摘要 assistant 消息"替换整轮
                compressed_round = [
                    user_msg,  # 保留原始用户问题
                    {
                        "role": "assistant",
                        "content": f"[Context Summary]\n{summary}",
                        "timestamp": first_round[-1].get("timestamp")
                    }
                ]
                compressed_rounds.append(compressed_round)
        else:
            # 不完整轮次保留原样（无法安全摘要）
            run_logger.info(
                f"[Phase C] Round {_round_no}/{_total_compressible}: incomplete round, "
                f"kept as-is (not summarized)"
            )
            compressed_rounds.append(first_round)

    # 理论上不会到这里，走到这里是因为对话还很长，可压缩的都压完之后仍然超长，这就要对当前轮次做处理了
    # 构建当前候选结果
    candidate_messages = []
    if system_msg:
        candidate_messages.append(system_msg)

    # 添加已压缩的轮次
    for round_msgs in compressed_rounds:
        candidate_messages.extend(round_msgs)

    # 添加未压缩的轮次
    for round_msgs in uncompressed_rounds:
        candidate_messages.extend(round_msgs)

    # 添加最后一轮次
    candidate_messages.extend(last_round)
    # 摘要都压完仍超限 → 交给单轮兜底逻辑做更激进的压缩
    return _handle_single_round_overflow(candidate_messages, max_tokens)

    # ── 测试用兜底 return（正常流程不会到这里）────────────────────
    # 当某些阶段被注释掉做单独测试时，函数执行到此返回原始消息而不是 None。
    # 这行在正常代码流中永远不会执行（上面所有路径都有 return）。
    return messages  # noqa: unreachable


def _split_into_rounds(messages: list[dict]) -> list[list[dict]]:
    """
    将消息列表分割为对话轮次。
    每轮从 user 消息开始，到下一个 user 消息之前结束。
    """
    rounds = []          # 所有轮次
    current_round = []   # 当前正在累积的轮次

    for msg in messages:
        if msg.get("role") == "user":
            # 遇到新的 user 消息，开始新轮次
            if current_round:
                rounds.append(current_round)  # 先把上一轮收尾
            current_round = [msg]             # 新轮次以该 user 消息开头
        else:
            # assistant 或 tool 消息，追加到当前轮次
            current_round.append(msg)

    # 添加最后一轮（可能不完整）
    if current_round:
        rounds.append(current_round)

    return rounds


def _is_complete_round(round_msgs: list[dict]) -> bool:
    """
    判断是否为完整轮次：
    - 必须以 user 消息开始
    - 必须以 assistant 消息结束（content 不为空）
    - 最后一条 assistant 不能包含 tool_calls
    """
    if not round_msgs:
        return False  # 空轮次

    # 检查开头：必须是 user
    if round_msgs[0].get("role") != "user":
        return False

    # 检查结尾：必须是 assistant
    last_msg = round_msgs[-1]
    if last_msg.get("role") != "assistant":
        return False

    # 结尾 assistant 的 content 必须为非空字符串
    content = last_msg.get("content")
    if not isinstance(content, str) or not content.strip():
        return False

    # 最后一条 assistant 不能还在发起工具调用（否则轮次尚未结束）
    if last_msg.get("tool_calls"):
        return False

    return True


# ── 阶段 C 摘要序列化相关常量 ──────────────────────────────────────
# 单条消息正文的字符预算：超过则做"头部+尾部"截断（报错/结论常在末尾）。
_SUMMARY_CONTENT_MAX = 6000    # 单条消息正文最大字符数
_SUMMARY_CONTENT_HEAD = 4000   # 截断时保留的头部字符数
_SUMMARY_CONTENT_TAIL = 1500   # 截断时保留的尾部字符数
# 工具调用参数（arguments）序列化时的字符预算与 JSON 内字段头部保留长度。
_SUMMARY_TOOL_ARGS_MAX = 1500  # 单个工具参数超过此长度才做 JSON 感知截断
_SUMMARY_ARGS_FIELD_HEAD = 300  # JSON 内过长字符串字段保留的头部长度
# 摘要输出的固定 token 预算（结构化格式比旧版更长，适当上调）。
_SUMMARY_MAX_TOKENS = 1500

# 摘要模型输出被判定为"不可用"的占位/前缀（命中则视为失败，保留原轮次）。
_UNUSABLE_SUMMARY_MARKERS = (
    "i cannot", "i can't", "i'm sorry", "i am sorry",
    "as an ai", "sorry, i",
)


def _summary_content_to_text(content) -> str:
    """把消息 content 规整为纯文本：字符串原样返回，多模态 list 拼接其中文本块。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image_url":
                    parts.append("[image]")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    if content is None:
        return ""
    return str(content)


def _summary_head_tail_truncate(
    text: str,
    max_chars: int = _SUMMARY_CONTENT_MAX,
    head_chars: int = _SUMMARY_CONTENT_HEAD,
    tail_chars: int = _SUMMARY_CONTENT_TAIL,
) -> str:
    """对过长文本做"头部 + 尾部"截断，保留结论/报错/测试汇总常在的末尾。

    参考 hermes context_compressor._serialize_for_summary 的思路：
    只保留头部会丢掉末尾的报错、测试结果和最终结论，因此中间截断、两头保留。
    """
    if not isinstance(text, str) or len(text) <= max_chars:
        return text
    return (
        text[:head_chars]
        + "\n...[truncated for summary]...\n"
        + text[-tail_chars:]
    )


def _summary_truncate_tool_args(args: str) -> str:
    """对过长的工具参数做 JSON 感知截断（移植自 hermes _truncate_tool_call_args_json 思路）。

    先 parse JSON → 递归截断过长的字符串叶子字段 → 重新序列化，
    从而避免直接切断 JSON 造成参数结构非法。非合法 JSON 则退化为头尾截断。
    """
    if not isinstance(args, str) or len(args) <= _SUMMARY_TOOL_ARGS_MAX:
        return args

    try:
        parsed = json.loads(args)
    except (ValueError, TypeError):
        # 非合法 JSON：不解析，退化为普通头尾截断（仅用于喂给摘要模型阅读）。
        return _summary_head_tail_truncate(
            args, _SUMMARY_TOOL_ARGS_MAX, _SUMMARY_ARGS_FIELD_HEAD, 200
        )

    def _shrink(obj):
        if isinstance(obj, str):
            if len(obj) > _SUMMARY_ARGS_FIELD_HEAD:
                return obj[:_SUMMARY_ARGS_FIELD_HEAD] + "...[truncated]"
            return obj
        if isinstance(obj, dict):
            return {k: _shrink(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_shrink(v) for v in obj]
        return obj

    return json.dumps(_shrink(parsed), ensure_ascii=False)


def _summary_redact(text: str) -> str:
    """发送摘要模型前的敏感信息脱敏。

    优先复用项目内的 redact 工具；不可用时退化为轻量正则兜底，最差也原样返回，
    绝不因脱敏失败而中断摘要流程。
    """
    if not isinstance(text, str) or not text:
        return text
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(text)
    except Exception:
        pass
    # 轻量兜底：常见 token 形态（GitHub token、Bearer、类 API key 长串）。
    try:
        import re
        text = re.sub(r"\bgh[pousr]_[A-Za-z0-9_]{8,}\b", "[REDACTED]", text)
        text = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}", "Bearer [REDACTED]", text)
        text = re.sub(r"\bsk-[A-Za-z0-9]{16,}\b", "[REDACTED]", text)
    except Exception:
        return text
    return text


def _serialize_round_for_summary(round_msgs) -> str:
    """按原始顺序把完整轮次序列化为带角色标签的文本，供摘要模型阅读。

    参考 hermes context_compressor._serialize_for_summary，但仅针对"单个轮次"：
      - user      → [USER]
      - assistant → [ASSISTANT]（正文）+ [TOOL CALL]（每个工具调用的名称与参数）
      - tool      → [TOOL RESULT]（保留工具返回内容）
    长内容做"头部+尾部"截断，工具参数做 JSON 感知截断，发送前统一脱敏。

    这样摘要模型能看到文件路径、报错、测试结果、关键数据和执行状态，
    而不再只有"用户问题 + 最终回答"。
    """
    parts: list[str] = []

    for msg in round_msgs:
        role = msg.get("role")

        if role == "user":
            text = _summary_redact(_summary_content_to_text(msg.get("content")))
            text = _summary_head_tail_truncate(text)
            if text.strip():
                parts.append(f"[USER]\n{text}")
            continue

        if role == "assistant":
            text = _summary_redact(_summary_content_to_text(msg.get("content")))
            text = _summary_head_tail_truncate(text)
            if text.strip():
                parts.append(f"[ASSISTANT]\n{text}")
            # 附加该 assistant 发起的工具调用（名称 + 截断后的参数）
            for call in msg.get("tool_calls") or []:
                func = call.get("function") or {}
                name = func.get("name", "unknown")
                args = func.get("arguments") or ""
                args = _summary_truncate_tool_args(args)
                args = _summary_redact(args)
                parts.append(f"[TOOL CALL]\n{name}({args})")
            continue

        if role == "tool":
            text = _summary_redact(_summary_content_to_text(msg.get("content")))
            text = _summary_head_tail_truncate(text)
            tool_name = msg.get("name", "")
            header = f"[TOOL RESULT]" + (f" {tool_name}" if tool_name else "")
            parts.append(f"{header}\n{text}")
            continue

        # 其它角色（少见）：原样带标签输出
        text = _summary_redact(_summary_content_to_text(msg.get("content")))
        text = _summary_head_tail_truncate(text)
        if text.strip():
            parts.append(f"[{str(role).upper()}]\n{text}")

    return "\n\n".join(parts)


def _is_summary_usable(summary) -> bool:
    """判断摘要模型返回结果是否可用：非空字符串、有实质内容、非明显拒答。"""
    if not isinstance(summary, str):
        return False
    stripped = summary.strip()
    if len(stripped) < 10:
        return False
    lowered = stripped.lower()
    # 命中拒答类前缀且整体很短 → 视为不可用
    if len(stripped) < 200 and any(m in lowered for m in _UNUSABLE_SUMMARY_MARKERS):
        return False
    return True


async def _summarize_round(round_msgs, llm_provider, llm_model):
    """为一个完整的对话轮次生成结构化摘要。

    返回：
      - 成功 → 摘要字符串；
      - 失败 / 结果为空 / 结果不可用 → None（由调用方保留原轮次，不做兜底替换）。

    与旧版的区别：不再只提取"用户问题 + 最终回答 + 工具名"，而是按原始顺序
    序列化完整轮次（user/assistant/tool call/tool result），使摘要模型能看到
    文件路径、报错、测试结果、关键数据与执行状态。
    """
    # ── 按原始顺序序列化完整轮次（含工具调用与工具结果）────────────
    serialized = _serialize_round_for_summary(round_msgs)
    if not serialized.strip():
        # 轮次内容为空（异常情况）→ 视为无法摘要，保留原轮次
        run_logger.info("Round has no serializable content, keeping original round")
        return None

    # ── 构建结构化摘要提示词 ──────────────────────────────────────
    summary_prompt = f"""你是上下文压缩助手。请将下面这一"对话轮次"压缩成可供后续模型直接续接任务的结构化摘要，目标约 {_SUMMARY_MAX_TOKENS} tokens 以内。
轮次原文按原始顺序提供，可能包含用户消息、助手消息、工具调用与工具结果：

---
{serialized}

---
请按以下格式输出，没有内容的字段可以省略：
用户请求
已执行操作
关键工具结果
结论与决策
相关文件或状态
未解决事项
摘要要求：
- 优先保留用户最近尚未回答或尚未完成的请求，确保后续模型可以直接继续处理。
- 只将实际执行且有明确结果的操作写入"已执行操作"；计划、建议和未执行方案不得写成已完成。
- 保留后续任务所需的关键对象、操作过程、输入输出、参数与条件、结果数据、异常信息和当前状态。
- 原文中出现的具体名称、数值、路径、命令或其他关键细节，应根据其对后续任务的重要性保留。
- 区分用户明确确认的决定、工具验证的事实和助手提出的建议；未经验证的内容必须标记为"未验证"。
- 如果早期方案、猜测或结论已被后续内容推翻，只保留最终有效结论；必要时简要注明原方案已放弃。
- 合并重复的工具调用、搜索和解释，删除无效尝试、寒暄以及与后续任务无关的信息。
- 不得根据上下文自行推断操作结果、当前状态、结论或用户决定。
- 不要自动生成用户没有提出的后续任务。
绝对不要保留 API key、访问令牌、密码、密钥、Cookie、连接字符串或其他凭证。如有出现，将具体值替换为 [REDACTED]。
当内容超过预算时，按以下优先级保留：
1. 用户最近尚未完成的请求；
2. 已确认的结论、决策和当前状态；
3. 关键结果、数据、异常信息和验证结果；
4. 后续续接所需的具体细节；
5. 一般背景和解释。
使用当前对话所使用的语言撰写。
只输出摘要正文，不要添加寒暄、前言、解释或额外前缀。"""

    try:
        # 调用 LLM 生成摘要（temperature=0 保证稳定，max_tokens 限制摘要长度）
        summary_response = await llm_provider.chat(
            messages=[{"role": "user", "content": summary_prompt}],
            model=llm_model,
            temperature=0.0,
            max_tokens=_SUMMARY_MAX_TOKENS,
        )
    except Exception as e:
        # 调用失败：记录日志并返回 None（调用方保留原轮次，不用兜底文本替换）
        run_logger.info(f"Failed to generate summary, keeping original round: {e}")
        return None

    summary = getattr(summary_response, "content", None)
    if not _is_summary_usable(summary):
        # 结果为空或明显不可用：返回 None，保留原轮次
        run_logger.info(
            "Summary result empty or unusable, keeping original round "
            f"(len={len(summary) if isinstance(summary, str) else 'n/a'})"
        )
        return None

    return summary.strip()


if __name__ == '__main__':
    # ── 本地自测：构造一段包含多轮对话的示例消息，验证压缩逻辑 ──────
    messages = [{"role": "user", "content": "Research the current state of solid-state batteries", "timestamp": "2026-03-16T11:32:12.508214"},
    {"role": "assistant", "content": "\n\nI'll research the current state of solid-state batteries for you. Let me search for the latest information.\n", "tool_calls": [{"id": "chatcmpl-tool-a285b81738931105", "type": "function", "function": {"name": "web_search", "arguments": "{\"query\": \"solid-state batteries 2026 latest developments breakthroughs\"}"}}], "reasoning_content": "The user wants me to research the current state of solid-state batteries. This is a research task that requires searching for recent information about solid-state battery technology, its development status, and recent breakthroughs.\n\nLet me search for current information about solid-state batteries.\n", "timestamp": "2026-03-16T11:32:12.508227"},
    {"role": "tool", "tool_call_id": "chatcmpl-tool-a285b81738931105", "name": "web_search", "content": "[{\"position\": 1, \"title\": \"What's next for EV batteries in 2026\", \"description\": \"Expect to see new chemistries hitting the roads, a shifting policy landscape, and a renewed focus on cost and performance.\", \"url\": \"https://www.technologyreview.com/2026/02/02/1132042/whats-next-for-ev-batteries-in-2026/\"}, {\"position\": 2, \"title\": \"Groundbreaking EV battery tech coming soon\", \"description\": \"Groundbreaking EV battery tech coming soon: The latest on BYD, Chery, Mercedes-Benz and more car brands' s\n... (truncated)", "timestamp": "2026-03-16T11:32:12.508235"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "chatcmpl-tool-bb63a6ea6f3fdb1f", "type": "function", "function": {"name": "fetch_web_page", "arguments": "{\"url\": \"https://www.technologyreview.com/2026/02/02/1132042/whats-next-for-ev-batteries-in-2026/\"}"}}, {"id": "chatcmpl-tool-bd7dfb2166f4dece", "type": "function", "function": {"name": "fetch_web_page", "arguments": "{\"url\": \"https://electrek.co/2026/03/10/a-us-startup-just-cleared-a-major-solid-state-battery-milestone-production-is-next/\"}"}}], "reasoning_content": "Good, I got some recent results. Let me fetch more details from a few of these sources to provide comprehensive information.\n", "timestamp": "2026-03-16T11:32:12.508238"},
    {"role": "tool", "tool_call_id": "chatcmpl-tool-bb63a6ea6f3fdb1f", "name": "fetch_web_page", "content": "{\"url\": \"https://www.technologyreview.com/2026/02/02/1132042/whats-next-for-ev-batteries-in-2026/\", \"title\": \"What's next for EV batteries in 2026\", \"markdown_content\": \"MIT Technology Review _'s What's Next series looks across industries, trends, and technologies to give you a first look at the future. You can read the rest of them_[_here_](https://www.technologyreview.com/tag/whats-next-in-tech/?)_._\\n\\nDemand for electric vehicles and the batteries that power them has never been hotter.\\n\\nIn\n... (truncated)", "timestamp": "2026-03-16T11:32:12.508244"},
    {"role": "tool", "tool_call_id": "chatcmpl-tool-bd7dfb2166f4dece", "name": "fetch_web_page", "content": "{\"url\": \"https://electrek.co/2026/03/10/a-us-startup-just-cleared-a-major-solid-state-battery-milestone-production-is-next/\", \"title\": \"A US startup just cleared a major solid-state battery milestone – production is next\", \"markdown_content\": \"A US startup just cleared a major solid-state battery milestone – production is next | Electrek\\n===============\\n[Skip to main content](https://electrek.co/2026/03/10/a-us-startup-just-cleared-a-major-solid-state-battery-milestone-production-is-next/#main\n... (truncated)", "timestamp": "2026-03-16T11:32:12.508248"},
    {"role": "assistant", "content": "## Current State of Solid-State Batteries (March 2026)\n\n### Key Advancements\n\n**Commercial Progress**\n- **ION Storage Systems** (Maryland, US) just achieved a major milestone – becoming the first US company to have a customer successfully qualify its solid-state \"Cornerstone Cell\" for performance. Production is expected to begin in 2026 at their Beltsville, Maryland facility.\n- **Factorial Energy** (US) provided cells for a Mercedes test vehicle that achieved **745 miles** on a single charge in real-world testing (September 2025). They plan to bring the tech to market by **2027**.\n- **Quantumscape** (US) is testing cells with automotive partners with commercial production planned for later this decade.\n\n**Major Automaker Timelines**\n- **Toyota** – Now targeting **2027-2028** for solid-state batteries in vehicles (after multiple delays from original 2020 goal).\n- **ProLogium** – Unveiled breakthrough \"superfluidized all-inorganic solid-state battery\" results at CES 2026.\n\n### Technology Overview\n\n**How they work:** Solid-state batteries replace the liquid electrolyte with a solid material, enabling higher energy density (targeting **400-500 Wh/kg** commercially, potentially 500-600 Wh/kg soon).\n\n**Benefits:**\n- Longer range for EVs\n- Faster charging\n- Improved safety (no liquid electrolyte = less flammable)\n- Longer lifespan\n\n**Challenges:**\n- Manufacturing at scale remains difficult\n- High production costs\n- Historical delays in commercialization\n\n### Market Landscape\n\n| Company | Status | Timeline |\n|---------|--------|----------|\n| ION Storage Systems (US) | Production qualified | 2026 |\n| Factorial Energy (US) | Testing with Mercedes | 2027 |\n| Quantumscape (US) | Partner testing | Late 2020s |\n| Toyota (Japan) | In development | 2027-2028 |\n| ProLogium (Taiwan) | CES 2026 breakthrough | TBD |\n\n### Transition: Semi-Solid-State\n\nMany Chinese manufacturers are pursuing **semi-solid-state** (gel electrolytes) as a stepping stone before full solid-state. This approach reduces liquid content while being easier to manufacture.\n\n### Bottom Line\n\nSolid-state batteries are transitioning from lab to production. While full-scale EV deployment is still likely **2027-2028 at earliest**, the technology is closer than ever to commercialization, with multiple companies achieving key milestones in 2025-2026.", "reasoning_content": "I now have good information about the current state of solid-state batteries. Let me compile a comprehensive summary for the user.\n", "timestamp": "2026-03-16T11:32:12.508251"},
    {"role": "user", "content": "What's happening with the China housing market right now?", "timestamp": "2026-03-16T11:47:57.995053"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "chatcmpl-tool-ba007475dc3c230a", "type": "function", "function": {"name": "web_search", "arguments": "{\"query\": \"China housing market 2026 latest news real estate crisis\"}"}}], "reasoning_content": "The user is asking about the current state of the China housing market. Let me search for the latest news and developments.\n", "timestamp": "2026-03-16T11:47:57.995068"},
    {"role": "tool", "tool_call_id": "chatcmpl-tool-ba007475dc3c230a", "name": "web_search", "content": "[{\"position\": 1, \"title\": \"China's home prices seen falling faster before stabilising in ...\", \"description\": \"Home prices projected to decline 4% in 2026, stabilise in 2027 · Sector faces challenges: demographic shifts, employment, affordability, unsold ...\", \"url\": \"https://www.reuters.com/world/asia-pacific/chinas-home-prices-seen-falling-faster-before-stabilising-2027-2026-03-13/\"}, {\"position\": 2, \"title\": \"China's property reset comes with a heavy price\", \"description\": \"It culminated in a\n... (truncated)", "timestamp": "2026-03-16T11:47:57.995078"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "chatcmpl-tool-962139376bfc3aa8", "type": "function", "function": {"name": "fetch_web_page", "arguments": "{\"url\": \"https://www.reuters.com/world/asia-pacific/chinas-home-prices-seen-falling-faster-before-stabilising-2027-2026-03-13/\"}"}}], "reasoning_content": "Let me get more details from some of these sources to provide a comprehensive answer.\n", "timestamp": "2026-03-16T11:47:57.995082"},
    {"role": "tool", "tool_call_id": "chatcmpl-tool-962139376bfc3aa8", "name": "fetch_web_page", "content": "{\"url\": \"https://www.reuters.com/world/asia-pacific/chinas-home-prices-seen-falling-faster-before-stabilising-2027-2026-03-13/\", \"title\": \"reuters.com\", \"markdown_content\": \"reuters.com\\n===============\"}", "timestamp": "2026-03-16T11:47:57.995087"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "chatcmpl-tool-a58bfa2c1d99268a", "type": "function", "function": {"name": "fetch_web_page", "arguments": "{\"url\": \"https://www.atlanticcouncil.org/blogs/econographics/chinas-property-slump-deepens-and-threatens-more-than-the-housing-sector/\"}"}}], "reasoning_content": "The fetch didn't get the full content. Let me try another source.\n\n", "timestamp": "2026-03-16T11:47:57.995090"},
    {"role": "tool", "tool_call_id": "chatcmpl-tool-a58bfa2c1d99268a", "name": "fetch_web_page", "content": "{\"url\": \"https://www.atlanticcouncil.org/blogs/econographics/chinas-property-slump-deepens-and-threatens-more-than-the-housing-sector/\", \"title\": \"China's property slump deepens—and threatens more than the housing sector\", \"markdown_content\": \"China's real estate slump is in its fifth year, with no end in sight. Key [indicators](https://rhg.com/research/chinas-economy-rightsizing-2025-looking-ahead-to-2026/)—sales, [prices](https://www.reuters.com/world/asia-pacific/china-home-prices-fall-37-thi\n... (truncated)", "timestamp": "2026-03-16T11:47:57.995097"},
    {"role": "assistant", "content": "## China Housing Market Update (March 2026)\n\n### The Current Situation\n\nChina's property slump is now in its **fifth year** with no end in sight. The market continues to deteriorate:\n\n**Key Metrics:**\n- **Home prices**: Fell **2.7% year-on-year** in December 2025 (70 major cities) — the steepest decline on record\n- **New home prices**: Dropped **0.04% month-on-month** in February 2026 — steepest decline since December 2022\n- **Primary property sales**: Expected to fall **10-14% in 2026** (S&P downgrade from earlier forecasts)\n- **Unsold/vacant homes**: Estimated **80 million** units clogging the market\n- **Wealth destruction**: ~85% of price gains since 2021 have evaporated\n\n### Economic Impact\n- Real estate once accounted for **~25% of GDP** and **15% of workforce**\n- The downturn has reduced annual GDP growth by ~**2 percentage points** in 2024-2025\n- Weak retail spending, low consumer confidence, and declining investment persist\n\n### Government Response\nBeijing has essentially **abandoned** the old \"high debt, high leverage\" model and is pivoting to a \"new model\" focused on:\n- Affordable housing\n- \"Basically stable prices\"\n- Reduced reliance on property as wealth storage\n\nHowever, measures so far have been largely ineffective.\n\n### Financial System Risks\n- **Zombie companies**: ~40% of bank loans to real estate go to firms that can't cover interest (up from 6% in 2018)\n- **Regional banks**: Growing concerns about smaller lenders with heavy exposure to local government financing vehicles\n- **Shadow banking**: Hidden stresses in wealth management products tied to developer loans\n\n### Outlook\n- Home prices projected to decline **~4% in 2026**, potentially stabilizing in **2027**\n- S&P warns the slump could be worse than expected\n- Analysts draw parallels to Japan's 1990s property crisis — a prolonged \"lost decade\" scenario", "reasoning_content": "This is a very comprehensive article. Let me summarize the key points for the user.\n", "timestamp": "2026-03-16T11:47:57.995100"}
    ]

    # 调用压缩函数并打印结果（注意：_compress_context 是 async，此处为示意）
    result = _compress_context(messages, max_tokens=130000)

    print(f"result:{result}")


    # 压缩后预期的结构示意：
    # [system_msg]
    # # 轮次 1 压缩
    # [user: "Research solid-state batteries"]
    # [assistant: "[Context Summary]\nUser asked about solid-state batteries..."]

    # # 轮次 2 压缩
    # [user: "What's happening with China housing?"]
    # [assistant: "[Context Summary]\nUser inquired about China's housing market..."]

    # # 轮次 3-5 完整保留
    # [user: "Tell me about AI regulations"]
    # [assistant: "Here's what I found..."]
    # [user: "Compare US and EU approaches"]
    # [assistant: ...]
    # [tool: ...]
    # [assistant: ...]
    # [user: "What about China?"]
    # [assistant: ...]
    # [tool: ...]  # 未完成的轮次
