"""
循环检测与分级熔断
──────────────────
检测两个维度：
  1. 推理内容 N-gram 相似度 — 模型是否在重复相同思路
  2. 工具调用指纹轨迹     — 工具是否在重复相同操作

维护 consecutive_loop_count，按连续触发轮次分级响应：
  LIGHT  (≥3 轮) → 注入换思路提示
  MEDIUM (≥5 轮) → 强制策略切换
  HEAVY  (≥8 轮) → 熔断，主循环终止
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LoopSignal(Enum):
    NONE = "none"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


@dataclass
class StepRecord:
    round_num: int
    reasoning: str
    # 每次工具调用的 md5 指纹：hash(tool_name + args + result_preview)
    tool_fingerprints: list[str] = field(default_factory=list)


class LoopDetector:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.history: list[StepRecord] = []
        self.consecutive_loop_count: int = 0

    # ── 公开接口 ───────────────────────────────────────────────────────────────

    def record(
        self,
        round_num: int,
        reasoning: str,
        tool_calls: list[dict],
        tool_results: list[dict],
    ) -> None:
        """记录一轮执行的推理文本和工具调用情况。"""
        fingerprints = []
        for tc, tr in zip(tool_calls, tool_results):
            # 提取 result 文本用于指纹（取前 200 字符，避免随机噪声）
            result_text = _extract_result_text(tr)[:200]
            raw = f"{tc.get('name', '')}:{str(tc.get('input', ''))}:{result_text}"
            fingerprints.append(hashlib.md5(raw.encode()).hexdigest())

        self.history.append(StepRecord(round_num, reasoning, fingerprints))

    def check(self) -> LoopSignal:
        """检查当前是否处于循环状态，返回分级信号。"""
        if len(self.history) < 2:
            return LoopSignal.NONE

        is_looping = self._check_ngram_similarity() or self._check_fingerprint_patterns()

        if is_looping:
            self.consecutive_loop_count += 1
        else:
            self.consecutive_loop_count = 0

        cfg = self.config
        if self.consecutive_loop_count >= cfg.loop_heavy_rounds:
            return LoopSignal.HEAVY
        if self.consecutive_loop_count >= cfg.loop_medium_rounds:
            return LoopSignal.MEDIUM
        if self.consecutive_loop_count >= cfg.loop_light_rounds:
            return LoopSignal.LIGHT
        return LoopSignal.NONE

    def get_guidance_message(self) -> str:
        """根据当前循环程度返回注入 prompt 的提示文本。"""
        count = self.consecutive_loop_count
        if count >= self.config.loop_medium_rounds:
            return (
                "⚠️ 你已连续多轮采用相同或高度相似的修复方式，均未解决问题。"
                "请彻底放弃当前思路，考虑以下替代方向：\n"
                "1. 简化问题，先实现最小可行版本\n"
                "2. 换一种完全不同的实现方式\n"
                "3. 如果依赖包有问题，考虑替换依赖\n"
                "不要再重复之前的修改。"
            )
        return (
            "注意：你在近几轮中采用了相似的修复方式。"
            "请换一种不同的思路来解决问题。"
        )

    def reset(self) -> None:
        """策略切换后重置计数器（避免误触发 HEAVY）。"""
        self.consecutive_loop_count = 0

    # ── 私有检测方法 ───────────────────────────────────────────────────────────

    def _check_ngram_similarity(self) -> bool:
        """比较最近两轮推理文本的 N-gram Jaccard 相似度。"""
        if len(self.history) < 2:
            return False

        last_text = self.history[-1].reasoning
        prev_text = self.history[-2].reasoning

        # 文本太短时不做判断（避免误判空响应）
        if len(last_text) < 30 or len(prev_text) < 30:
            return False

        n = self.config.loop_ngram_n
        last_ngrams = _get_ngrams(last_text, n)
        prev_ngrams = _get_ngrams(prev_text, n)

        if not last_ngrams or not prev_ngrams:
            return False

        intersection = sum((last_ngrams & prev_ngrams).values())
        union = sum((last_ngrams | prev_ngrams).values())
        similarity = intersection / union if union > 0 else 0

        return similarity >= self.config.loop_ngram_threshold

    def _check_fingerprint_patterns(self) -> bool:
        """检测工具调用轨迹中的无进展模式。"""
        if len(self.history) < 2:
            return False

        last_fps = frozenset(self.history[-1].tool_fingerprints)
        prev_fps = frozenset(self.history[-2].tool_fingerprints)

        # 模式一：完全重复（相同工具 + 相同参数 + 相同结果）
        if last_fps and last_fps == prev_fps:
            return True

        # 模式二：A-B-A-B 交替（最近 4 轮）
        if len(self.history) >= 4:
            r1 = frozenset(self.history[-4].tool_fingerprints)
            r2 = frozenset(self.history[-3].tool_fingerprints)
            r3 = frozenset(self.history[-2].tool_fingerprints)
            r4 = frozenset(self.history[-1].tool_fingerprints)
            if r1 == r3 and r2 == r4 and r1 != r2 and r1 and r2:
                return True

        return False


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def _get_ngrams(text: str, n: int) -> Counter:
    words = text.split()
    return Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def _extract_result_text(tool_result: dict) -> str:
    content = tool_result.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content if block.get("type") == "text"
        )
    return str(content)
