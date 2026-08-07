"""
循环检测与分级熔断
──────────────────
检测五个维度（计分制，达到 loop_score_threshold 才算一轮循环）：
  1. 推理内容字符级 N-gram 相似度   +1  — 模型是否在重复相同思路
  2. 精确指纹窗口内重复             +2  — 相同工具+参数+结果在窗口内再现
  3. 意图指纹重试且全部失败         +2  — 相同工具+参数反复尝试、从未成功
  4. 目标固守                       +2  — 窗口内绝大多数调用围着同一个关键词打转
                                         且基本全部失败（每次命令不同、目标不变）
  5. 任务停滞                       +2  — 窗口内多轮没有任何"与任务相关且成功"
                                         的调用（应对失败游离：失败的、空成功的、
                                         以及偏离任务关键词的闲逛调用）

指纹设计：
  精确指纹  md5(工具名 + 归一化参数 + 归一化结果前200字符)
  意图指纹  md5(工具名 + 归一化参数)          — 不看结果，抓"同参数重试"
  归一化    去时间戳 / 去绝对路径 / 压缩空白，降低噪声导致的漏报

维护 consecutive_loop_count，按连续触发轮次分级响应：
  LIGHT  (≥3 轮) → 注入换思路提示
  MEDIUM (≥5 轮) → 强制策略切换
  HEAVY  (≥8 轮) → 熔断，主循环终止
"""
from __future__ import annotations

import hashlib
import json
import re
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
    # 精确指纹：hash(工具名 + 参数 + 归一化结果前200字符)
    tool_fingerprints: list[str] = field(default_factory=list)
    # 意图指纹：hash(工具名 + 参数)，与成功标志一一对应
    intent_fingerprints: list[str] = field(default_factory=list)
    intent_success: list[bool] = field(default_factory=list)
    # 每次调用参数中提取的关键词集合（目标固守检测用）
    arg_tokens: list[set[str]] = field(default_factory=list)
    # 每次调用 参数∪结果 的关键词集合（任务停滞检测用）
    all_tokens: list[set[str]] = field(default_factory=list)


class LoopDetector:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.history: list[StepRecord] = []
        self.consecutive_loop_count: int = 0
        self.total_loop_count: int = 0  # 累计循环轮次（reset 不清零，兜底熔断用）
        self.last_score: int = 0  # 最近一轮的循环得分（便于观测/调试）
        self._task_tokens: set[str] = set()  # 任务关键词（停滞检测用）

    def set_task(self, task: str) -> None:
        """主循环启动时注入任务描述，提取关键词供任务停滞检测使用。"""
        self._task_tokens = _extract_tokens(task)

    # ── 公开接口 ───────────────────────────────────────────────────────────────

    def record(
        self,
        round_num: int,
        reasoning: str,
        tool_calls: list[dict],
        tool_results: list[dict],
    ) -> None:
        """记录一轮执行的推理文本和工具调用情况。"""
        exact_fps, intent_fps, successes, tokens, all_tokens = [], [], [], [], []
        for tc, tr in zip(tool_calls, tool_results):
            name = tc.get("name", "")
            args = _normalize_args(tc.get("input", {}))
            result_text = _normalize_result(_extract_result_text(tr))[:200]
            exact_fps.append(_md5(f"{name}:{args}:{result_text}"))
            intent_fps.append(_md5(f"{name}:{args}"))
            successes.append(bool(tr.get("success", True)))
            arg_toks = _extract_tokens(args)
            tokens.append(arg_toks)
            all_tokens.append(arg_toks | _extract_tokens(result_text[:500]))

        self.history.append(
            StepRecord(round_num, reasoning, exact_fps, intent_fps,
                       successes, tokens, all_tokens)
        )

    def check(self) -> LoopSignal:
        """检查当前是否处于循环状态，返回分级信号。"""
        if len(self.history) < 2:
            return LoopSignal.NONE

        # 计分制：多信号互相印证，避免单点误报导致误熔断
        score = 0
        if self._check_ngram_similarity():
            score += 1
        if self._check_window_repeat():
            score += 2
        if self._check_intent_retry():
            score += 2
        if self._check_goal_fixation():
            score += 2
        if self._check_stagnation():
            score += 2
        self.last_score = score

        if score >= self.config.loop_score_threshold:
            self.consecutive_loop_count += 1
            self.total_loop_count += 1
        elif score == 0:
            self.consecutive_loop_count = 0
        # score 为 1（仅思路相似、无重复调用证据）：保持计数不变。
        # 卡顿时常穿插排查轮，直接清零会让 streak 永远到不了阈值。

        cfg = self.config
        # 熔断双条件：连续循环到 HEAVY，或累计循环轮次过多
        # （MEDIUM 后 streak 被 reset，模型若换汤不换药，靠累计值兜底）
        if (self.consecutive_loop_count >= cfg.loop_heavy_rounds
                or self.total_loop_count >= cfg.loop_heavy_total):
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
        """比较最近两轮推理文本的字符级 N-gram Jaccard 相似度。"""
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

    def _check_window_repeat(self) -> bool:
        """最近一轮的精确指纹，在滑动窗口内的任意一轮出现过即算重复。

        覆盖"重试 ← 排查 ← 重试"的穿插型无进展模式（相邻比对会漏）。
        """
        last_fps = set(self.history[-1].tool_fingerprints)
        if not last_fps:
            return False

        window = self.history[-(self.config.loop_window + 1) : -1]
        for rec in window:
            if last_fps & set(rec.tool_fingerprints):
                return True
        return False

    def _check_intent_retry(self) -> bool:
        """同一意图指纹（工具+参数）在窗口内反复出现且从未成功 → 无进展轮询。"""
        window = self.history[-self.config.loop_window :]
        min_repeats = self.config.loop_intent_min_repeats

        attempts: dict[str, list[bool]] = {}
        for rec in window:
            for fp, ok in zip(rec.intent_fingerprints, rec.intent_success):
                attempts.setdefault(fp, []).append(ok)

        # 只看最近一轮仍在尝试的意图：历史上试过但已放弃的不算循环
        current = set(self.history[-1].intent_fingerprints)
        for fp in current:
            results = attempts.get(fp, [])
            if len(results) >= min_repeats and not any(results):
                return True
        return False

    def _check_goal_fixation(self) -> bool:
        """目标固守：窗口内绝大多数调用包含同一关键词，且这些调用基本全部失败。

        真实卡顿的形态往往不是"重复同一条命令"，而是"围着同一个
        不存在的依赖/文件/服务换着花样找"——参数指纹每次都不同，
        但语义目标没变。用参数关键词的覆盖率 + 失败率来刻画。
        """
        window = self.history[-self.config.loop_window :]
        # (tokens, success) 拍平窗口内所有调用
        calls = [
            (toks, ok)
            for rec in window
            for toks, ok in zip(rec.arg_tokens, rec.intent_success)
            if toks
        ]
        if len(calls) < 6:
            return False

        # 找覆盖率最高的关键词
        cover: Counter = Counter()
        for toks, _ in calls:
            for tok in toks:
                cover[tok] += 1
        if not cover:
            return False
        top_tok, top_n = cover.most_common(1)[0]
        if top_n / len(calls) < 0.6:
            return False

        # 含该关键词的调用的失败率
        topical = [ok for toks, ok in calls if top_tok in toks]
        fail_ratio = 1 - sum(topical) / len(topical)
        return fail_ratio >= 0.7

    def _check_stagnation(self) -> bool:
        """任务停滞：最近若干轮中，没有任何"成功且与任务相关"的调用。

        真实卡住的形态常常是"失败游离"：调用要么失败，要么空成功
        （echo PATH / pwd / ls），要么干脆偏离任务闲逛。单看失败率
        或单看参数重复都抓不住，用"与任务关键词无交集的成功"来排除。
        """
        if not self._task_tokens:
            return False
        window = self.history[-self.config.loop_window :]
        if len(window) < 4:
            return False

        stagnant = 0
        for rec in window:
            progress = any(
                ok and (toks & self._task_tokens)
                for toks, ok in zip(rec.all_tokens, rec.intent_success)
            )
            if not progress and (rec.all_tokens or rec.reasoning):
                stagnant += 1
        return stagnant >= len(window) - 1


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def _get_ngrams(text: str, n: int) -> Counter:
    """字符级 n-gram。中文无空格，按词切分会让整段只剩 1 个 token 而失效；
    字符滑窗对中英文都适用（英文忽略空白后同样可比）。"""
    chars = [c for c in text.lower() if not c.isspace()]
    return Counter(tuple(chars[i : i + n]) for i in range(len(chars) - n + 1))


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


_TOKEN_RE = re.compile(r"[a-zA-Z][\w.\-]{3,}")
# 通用技术词不提供区分度，过滤
_TOKEN_STOP = {
    "python", "pip", "install", "read", "write", "file", "echo", "true",
    "false", "none", "null", "nul", "http", "https", "com", "www", "json",
    "command", "path", "code", "content", "list", "find", "grep", "name",
    "type", "head", "dev", "usr", "bin", "local", "dir", "where", "which",
}


def _extract_tokens(args_text: str) -> set[str]:
    """从序列化后的参数中提取关键词（用于目标固守检测）。"""
    return {t for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(args_text))
            if t not in _TOKEN_STOP}


def _normalize_args(args: Any) -> str:
    """参数序列化：key 排序，避免 dict 顺序导致同参不同纹。"""
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(args)


_TS_RE = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?")
_PATH_RE = re.compile(r"([A-Za-z]:)?[\\/][\w.\-/\\]+")


def _normalize_result(text: str) -> str:
    """结果归一化：去时间戳、去绝对路径、压缩空白。

    降低"同一件事因噪声算出不同指纹"的漏报。
    """
    text = _TS_RE.sub("", text)
    text = _PATH_RE.sub("", text)
    return " ".join(text.split())


def _extract_result_text(tool_result: dict) -> str:
    content = tool_result.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content if block.get("type") == "text"
        )
    return str(content)
