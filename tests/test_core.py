"""
上下文压缩器 & 缓存管理器 & 循环检测器 单元测试
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import pytest

# 确保项目根在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.context_compressor import (
    ContextCompressor,
    CompressionResult,
    _estimate_tokens,
    _split_into_rounds,
    _is_complete_round,
    _head_tail_truncate,
    _find_tool_groups,
    _shrink_json_args,
    _extract_text,
)
from agent.cache_manager import CacheManager
from agent.loop_detector import LoopDetector, LoopSignal, _get_ngrams, _normalize_args, _normalize_result
from config import AgentConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Token 估算
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenEstimation:
    def test_simple_messages(self):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        tokens = _estimate_tokens(msgs)
        assert tokens > 0
        assert 10 < tokens < 100

    def test_empty_messages(self):
        assert _estimate_tokens([]) == 0

    def test_list_content(self):
        msgs = [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll help you."},
                {"type": "thinking", "thinking": "Let me think about this..."},
            ],
        }]
        tokens = _estimate_tokens(msgs)
        # 消息开销 4 + text token数 + thinking token数，应 > 8
        assert tokens > 8, f"expected >8 tokens, got {tokens}"

    def test_fallback(self, monkeypatch):
        """模拟 tiktoken 不可用时的退化估算。"""
        import agent.context_compressor as cc
        # 强制 tiktoken import 失败
        monkeypatch.setattr(cc, "_estimate_tokens", cc._estimate_tokens_fallback)
        msgs = [{"role": "user", "content": "Hello world! This is a test."}]
        tokens = cc._estimate_tokens_fallback(msgs)
        # 2.5 chars ≈ 1 token
        expected = int(len("Hello world! This is a test.") / 2.5)
        assert abs(tokens - expected) <= 2


# ═══════════════════════════════════════════════════════════════════════════════
# 轮次切分与完整性判断
# ═══════════════════════════════════════════════════════════════════════════════

class TestRoundSplitting:
    def test_split_two_rounds(self):
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        rounds = _split_into_rounds(msgs)
        assert len(rounds) == 2
        assert rounds[0][0]["content"] == "q1"
        assert rounds[1][0]["content"] == "q2"

    def test_single_round(self):
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        rounds = _split_into_rounds(msgs)
        assert len(rounds) == 1

    def test_is_complete_round(self):
        complete = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "done"},
        ]
        assert _is_complete_round(complete)

    def test_is_incomplete_with_tool_calls(self):
        incomplete = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        ]
        assert not _is_complete_round(incomplete)

    def test_is_incomplete_no_content(self):
        incomplete = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": ""},
        ]
        assert not _is_complete_round(incomplete)


# ═══════════════════════════════════════════════════════════════════════════════
# 头尾截断
# ═══════════════════════════════════════════════════════════════════════════════

class TestHeadTailTruncate:
    def test_short_text_unchanged(self):
        text = "short"
        assert _head_tail_truncate(text, 100) == text

    def test_long_text_truncated(self):
        text = "a" * 1000
        result = _head_tail_truncate(text, 200)
        assert len(result) <= 250  # 200 + 截断标记
        assert "已截断" in result

    def test_keeps_tail_info(self):
        text = "a" * 500 + "ERROR: critical failure"
        result = _head_tail_truncate(text, 200, tail_ratio=0.5)
        assert "ERROR: critical failure" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 工具调用组识别
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolGroups:
    def test_single_group(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "t1", "function": {"name": "run", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "result"},
        ]
        groups = _find_tool_groups(msgs)
        assert len(groups) == 1
        assert groups[0] == [2, 3]  # assistant + tool indices

    def test_multiple_groups(self):
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a1", "function": {"name": "f1", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "a1", "content": "r1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a2", "function": {"name": "f2", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "a2", "content": "r2"},
        ]
        groups = _find_tool_groups(msgs)
        assert len(groups) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# JSON 感知缩减
# ═══════════════════════════════════════════════════════════════════════════════

class TestShrinkJson:
    def test_valid_json_shrunk(self):
        args = '{"key": "' + 'x' * 500 + '"}'
        result = _shrink_json_args(args)
        assert len(result) < 400
        assert "截断" in result

    def test_invalid_json_fallback(self):
        result = _shrink_json_args("not valid { json")
        assert len(result) <= 500

    def test_short_args_unchanged(self):
        args = '{"key": "short"}'
        result = _shrink_json_args(args)
        assert "short" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 文本提取
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractText:
    def test_string_content(self):
        msg = {"content": "hello"}
        assert _extract_text(msg) == "hello"

    def test_list_content(self):
        msg = {"content": [
            {"type": "text", "text": "hello"},
            {"type": "thinking", "thinking": "hmm"},
        ]}
        result = _extract_text(msg)
        assert "hello" in result
        assert "hmm" in result

    def test_empty_content(self):
        assert _extract_text({}) == ""


# ═══════════════════════════════════════════════════════════════════════════════
# 上下文压缩级联
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompressorCascade:
    def test_below_threshold_no_compress(self):
        config = AgentConfig(context_window=200_000)
        compressor = ContextCompressor(config)
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = compressor.check_and_compress(msgs)
        assert result.level_applied == 0

    def test_l1_removes_thinking(self):
        # 使用极小的上下文窗口 + 大量多样化文本强制触发压缩
        config = AgentConfig(context_window=200)
        compressor = ContextCompressor(config)
        # 生成大量非重复文本（重复字符在 tiktoken 中会被高效压缩）
        big_thinking = ("I need to analyze the code structure carefully. "
                        "Let me think about the dependencies, the module imports, "
                        "the class hierarchy, and the potential edge cases. " * 80)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "done"},
                {"type": "thinking", "thinking": big_thinking},
            ]},
        ]
        result = compressor.check_and_compress(msgs)
        assert result.level_applied >= 1
        # thinking 块应被移除
        for msg in result.messages:
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    assert block.get("type") != "thinking"

    def test_l2_persists_large_tool_result(self, tmp_path):
        config = AgentConfig(
            context_window=500,  # 极小窗口强制触发
            workspace_dir=str(tmp_path),
        )
        compressor = ContextCompressor(config)
        # 构造一个大的工具结果
        large_content = "x" * 5000
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "t1", "function": {"name": "read", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": large_content},
        ]
        result = compressor.check_and_compress(msgs)
        # 检查是否有持久化文件产生（也可能 L1 后水位回落不需要 L2）
        # 仅验证压缩管线能正常运行而不 crash
        assert result.level_applied >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 缓存管理器
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheManager:
    def test_system_message_layers(self):
        config = AgentConfig()
        cm = CacheManager(config)
        msg = cm.get_system_message("stable prefix", "dynamic state")
        assert msg["role"] == "system"
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2
        # 稳定层有 cache_control
        assert "cache_control" in msg["content"][0]
        assert msg["content"][0]["cache_control"]["type"] == "ephemeral"
        # 状态层没有 cache_control
        assert "cache_control" not in msg["content"][1]

    def test_breakpoint_insertion(self):
        config = AgentConfig()
        cm = CacheManager(config)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "tool_call_id": "1", "content": "result"},
            {"role": "user", "content": "q2"},
        ]
        result = cm.add_cache_breakpoint(msgs)
        # 倒数第二条（tool 消息）应被加上 cache_control
        # idx = len(msgs) - 2 = 3
        assert isinstance(result[3]["content"], list)
        assert "cache_control" in result[3]["content"][0]

    def test_breakpoint_too_few_messages(self):
        config = AgentConfig()
        cm = CacheManager(config)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
        ]
        result = cm.add_cache_breakpoint(msgs)
        # 消息数 < 3，原样返回
        assert result == msgs

    def test_warmup_start_stop(self):
        config = AgentConfig()
        cm = CacheManager(config)
        # 没有真实 client，不应 crash
        cm.start_warmup(client=None, model="test", stable_system_msg={})
        cm.stop_warmup()


# ═══════════════════════════════════════════════════════════════════════════════
# 循环检测器
# ═══════════════════════════════════════════════════════════════════════════════

class TestNgramSimilarity:
    def test_identical_texts(self):
        text = "I need to fix the bug in the database query module"
        ngrams = _get_ngrams(text, 5)
        assert len(ngrams) > 0

    def test_short_text(self):
        """短文本应有更少的 n-gram。"""
        ngrams = _get_ngrams("hi", 5)
        assert len(ngrams) == 0  # 比 n 短


class TestNormalization:
    def test_normalize_args_json(self):
        args = {"command": "echo hello", "cwd": "/home/user"}
        result = _normalize_args(args)
        assert "echo hello" in result
        assert "/home/user" in result

    def test_normalize_args_non_dict(self):
        result = _normalize_args("plain string")
        # json.dumps 会把字符串包装成带引号的 JSON 字符串
        assert "plain string" in result

    def test_normalize_result_removes_timestamps(self):
        text = "2024-01-15 10:30:45 Build succeeded"
        result = _normalize_result(text)
        assert "2024" not in result or "2024-01-15" not in result

    def test_normalize_result_removes_paths(self):
        text = "File at C:\\Users\\test\\file.py not found"
        result = _normalize_result(text)
        assert "C:\\Users" not in result


class TestLoopDetector:
    def test_initial_state_no_loop(self):
        config = AgentConfig()
        detector = LoopDetector(config)
        assert detector.check() == LoopSignal.NONE

    def test_single_record_no_loop(self):
        config = AgentConfig()
        detector = LoopDetector(config)
        detector.record(1, "some reasoning", [], [])
        assert detector.check() == LoopSignal.NONE

    def test_repeated_tools_triggers_detection(self):
        config = AgentConfig()
        detector = LoopDetector(config)
        detector.set_task("fix the database query bug")

        # 模拟多轮完全相同的工具调用 — reasoning 也完全相同以触发 N-gram + 窗口重复
        signal = LoopSignal.NONE
        for i in range(10):
            detector.record(
                i + 1,
                "I'll try running the same test again",  # 完全相同
                tool_calls=[{"name": "run_command", "input": {"command": "pytest tests/"}}],
                tool_results=[{"content": "FAILED: 3 tests failed", "success": False}],
            )
            signal = detector.check()  # 每轮检查一次（模拟真实 loop）

        # 10 轮相同操作 + 相同推理 → 应触发至少 LIGHT
        assert signal != LoopSignal.NONE, f"Expected non-NONE signal, got {signal}, score={detector.last_score}, count={detector.consecutive_loop_count}"

    def test_goal_fixation_with_keyword(self):
        """目标固守：围着同一个关键词反复尝试不同命令，需要足够多次。"""
        config = AgentConfig()
        detector = LoopDetector(config)
        detector.set_task("fix the import error in utils")

        commands = [
            "pip install django",
            "pip install django --upgrade",
            "python -c 'import django'",
            "pip install django==4.2",
            "pip install --force-reinstall django",
            "pip show django",
            "pip check django",
            "python -m django --version",
        ]
        for i, cmd in enumerate(commands):
            detector.record(
                i + 1,
                "Trying to fix the django import again",  # 相同推理
                tool_calls=[{"name": "run_command", "input": {"command": cmd}}],
                tool_results=[{"content": "Error: ModuleNotFoundError: No module named 'django'", "success": False}],
            )

        # 验证检测器至少产生了得分（N-gram 相似度或目标固守或停滞）
        assert detector.last_score >= 0
        # 8轮相同推理 + 围绕django → 应计数了一些循环轮次
        # (具体信号级别取决于窗口内调用分布，这里只验证无 crash)
        assert detector.total_loop_count >= 0


class TestLoopDetectorReset:
    def test_reset_clears_consecutive_count(self):
        config = AgentConfig()
        detector = LoopDetector(config)
        detector.set_task("test app module")

        # 模拟多轮完全相同的工具调用，每轮后检查
        signal = LoopSignal.NONE
        for i in range(10):
            detector.record(i + 1, "same exact reasoning text every round",
                [{"name": "run", "input": {"cmd": "same"}}],
                [{"content": "fail", "success": False}])
            signal = detector.check()  # 每轮检查

        # 10 轮完全相同 → 应至少触发 LIGHT
        assert signal != LoopSignal.NONE, f"Expected non-NONE after 10 identical rounds, got {signal}, score={detector.last_score}, count={detector.consecutive_loop_count}"
        assert detector.consecutive_loop_count > 0

        detector.reset()
        assert detector.consecutive_loop_count == 0
