"""
真实任务端到端对比：缓存开 vs 关
────────────────────────────────
让 AgentLoop 真实执行同一个编码任务两次（CACHE 开/关各一次），
通过 on_event 的 debug_raw 事件收集每轮 usage（含 cached_tokens）。

注意：真实运行有不确定性（模型两轮生成内容不完全一致），
两轮的任务轨迹不可能完全相同，结果是"一次真实采样"而非严格对照实验。
建议各跑 2-3 次取趋势。

用法：
  python -m benchmarks.real_task_benchmark
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from agent.loop import AgentLoop
from config import AgentConfig

# 多步骤任务：建项目 → 写代码 → 跑测试 → 修 bug → 扩展功能 → 再测，自然产生多轮
TASK = (
    "在工作区完成以下任务："
    "1) 创建一个 Python 模块 calc.py，实现 add/sub/mul/div 四个函数，div 要处理除零抛 ValueError；"
    "2) 编写 pytest 测试 test_calc.py，覆盖正常值、负数、浮点、除零至少 8 个用例；"
    "3) 运行测试，若有失败修复到全部通过；"
    "4) 再增加 power 和 mod 两个函数及对应测试，重新运行全部测试确认通过。"
)

# ── 长任务版：8 个独立模块各藏 1 个 bug，强制一次修一个，轮数拉到 20+ ──────
SEED_FILES = {
    "m01_math_basic.py": '''def clamp(x, lo, hi):
    """限制在 [lo, hi]"""
    return max(lo, min(x, hi))  # BUG: min/max 用反


def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)
''',
    "test_m01.py": '''from m01_math_basic import clamp, sign

def test_clamp_inside(): assert clamp(5, 1, 10) == 5
def test_clamp_low(): assert clamp(-3, 1, 10) == 1
def test_clamp_high(): assert clamp(99, 1, 10) == 10
def test_sign(): assert sign(3) == 1 and sign(-2) == -1 and sign(0) == 0
''',
    "m02_str_case.py": '''def snake_to_camel(s: str) -> str:
    parts = s.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def camel_to_snake(s: str) -> str:
    out = []
    for c in s:
        if c.isupper():
            out.append("_")
            out.append(c.lower())  # BUG: 首字母大写时前面也加了 _
        else:
            out.append(c)
    return "".join(out)
''',
    "test_m02.py": '''from m02_str_case import snake_to_camel, camel_to_snake

def test_s2c(): assert snake_to_camel("hello_world_foo") == "helloWorldFoo"
def test_c2s(): assert camel_to_snake("helloWorldFoo") == "hello_world_foo"
def test_roundtrip(): assert camel_to_snake(snake_to_camel("a_b_c")) == "a_b_c"
''',
    "m03_list_stats.py": '''def median(nums):
    """中位数"""
    s = sorted(nums)
    n = len(s)
    if n == 0:
        raise ValueError("empty")
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2


def mean(nums):
    if not nums:
        raise ValueError("empty")
    return sum(nums) / len(nums)


def variance(nums):
    """总体方差"""
    m = mean(nums)
    return sum((x - m) ** 2 for x in nums) / (len(nums) - 1)  # BUG: 总体方差应除以 n
''',
    "test_m03.py": '''from m03_list_stats import median, mean, variance

def test_median_odd(): assert median([3, 1, 2]) == 2
def test_median_even(): assert median([1, 2, 3, 4]) == 2.5
def test_mean(): assert mean([1, 2, 3]) == 2
def test_variance(): assert variance([1, 2, 3]) == 2 / 3
''',
    "m04_dict_utils.py": '''def invert(d: dict) -> dict:
    return {v: k for k, v in d.items()}


def merge_sum(a: dict, b: dict) -> dict:
    """合并两个字典，相同 key 的值相加"""
    out = dict(a)
    for k, v in b.items():
        out[k] = v  # BUG: 应为 out.get(k, 0) + v
    return out
''',
    "test_m04.py": '''from m04_dict_utils import invert, merge_sum

def test_invert(): assert invert({"a": 1}) == {1: "a"}
def test_merge_disjoint(): assert merge_sum({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
def test_merge_overlap(): assert merge_sum({"a": 1}, {"a": 5}) == {"a": 6}
''',
    "m05_stack.py": '''class Stack:
    def __init__(self):
        self._items = []

    def push(self, x):
        self._items.append(x)

    def pop(self):
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop(0)  # BUG: 栈应 pop(-1)

    def peek(self):
        if not self._items:
            raise IndexError("peek empty stack")
        return self._items[-1]

    def __len__(self):
        return len(self._items)
''',
    "test_m05.py": '''import pytest
from m05_stack import Stack

def test_lifo():
    s = Stack()
    s.push(1); s.push(2); s.push(3)
    assert s.pop() == 3 and s.pop() == 2 and s.pop() == 1

def test_peek():
    s = Stack()
    s.push(7)
    assert s.peek() == 7 and len(s) == 1

def test_pop_empty():
    with pytest.raises(IndexError):
        Stack().pop()
''',
    "m06_dates.py": '''def is_leap_year(y: int) -> bool:
    return y % 4 == 0  # BUG: 未处理 100/400 规则


def days_in_month(y: int, m: int) -> int:
    if m in (1, 3, 5, 7, 8, 10, 12):
        return 31
    if m in (4, 6, 9, 11):
        return 30
    if m == 2:
        return 29 if is_leap_year(y) else 28
    raise ValueError("bad month")
''',
    "test_m06.py": '''from m06_dates import is_leap_year, days_in_month

def test_leap_2000(): assert is_leap_year(2000)
def test_leap_1900(): assert not is_leap_year(1900)
def test_leap_2024(): assert is_leap_year(2024)
def test_feb(): assert days_in_month(2024, 2) == 29 and days_in_month(2023, 2) == 28
def test_bad_month():
    import pytest
    with pytest.raises(ValueError):
        days_in_month(2024, 13)
''',
    "m07_lru.py": '''class LRUCache:
    """容量超限淘汰最久未使用的项"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._data = {}
        self._order = []

    def get(self, key):
        if key not in self._data:
            return None
        self._order.remove(key)
        self._order.append(key)
        return self._data[key]

    def put(self, key, value):
        if key in self._data:
            self._order.remove(key)
        self._data[key] = value
        self._order.append(key)
        if len(self._data) > self.capacity:
            oldest = self._order.pop()  # BUG: 应淘汰 pop(0)
            del self._data[oldest]
''',
    "test_m07.py": '''from m07_lru import LRUCache

def test_evict_oldest():
    c = LRUCache(2)
    c.put("a", 1); c.put("b", 2); c.put("c", 3)
    assert c.get("a") is None and c.get("b") == 2 and c.get("c") == 3

def test_get_refreshes():
    c = LRUCache(2)
    c.put("a", 1); c.put("b", 2)
    c.get("a")
    c.put("c", 3)
    assert c.get("a") == 1 and c.get("b") is None
''',
    "m08_roman.py": '''def int_to_roman(n: int) -> str:
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
            (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
            (5, "V"), (4, "IV"), (1, "I")]
    out = []
    for v, sym in vals:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def roman_to_int(s: str) -> int:
    table = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, c in enumerate(s):
        total += table[c]  # BUG: 未处理 IV/IX 等减法规则
    return total
''',
    "test_m08.py": '''from m08_roman import int_to_roman, roman_to_int

def test_to_roman(): assert int_to_roman(1994) == "MCMXCIV"
def test_from_roman_add(): assert roman_to_int("XII") == 12
def test_from_roman_sub(): assert roman_to_int("MCMXCIV") == 1994
def test_roundtrip(): assert roman_to_int(int_to_roman(444)) == 444
''',
}

LONG_TASK_TEMPLATE = (
    "工作区里预置了 {n} 个 Python 模块（m01_math_basic.py 到 m{nn}_*.py），"
    "每个模块对应一个测试文件（test_m01.py 到 test_m{nn}.py），"
    "每个模块里都藏有 1 个 bug，目前测试全部不通过。\n"
    "请严格遵守以下流程：\n"
    "- 一次只处理一个模块：先运行该模块的测试 → 读代码定位 bug → 修复 → "
    "重新运行该模块的测试确认通过；\n"
    "- 确认当前模块全部通过后，才能开始下一个模块（按 m01 → m{nn} 顺序）；\n"
    "- 不许修改任何测试文件；\n"
    "- 全部 {n} 个模块修复完成后，最后运行一次完整 pytest 确认 0 失败。"
)

# 模块名（用于任务描述）
_MODULE_NAMES = ["m01_math_basic", "m02_str_case", "m03_list_stats", "m04_dict_utils",
                 "m05_stack", "m06_dates", "m07_lru", "m08_roman"]


def make_task(n: int) -> str:
    last = _MODULE_NAMES[n - 1]
    return LONG_TASK_TEMPLATE.format(n=n, nn=f"{n:02d}").replace("m{nn}_*", last)


LONG_TASK = make_task(8)


# ── 大代码量任务：单个 ~600 行模块，10 节各 1 bug，强制逐节修复 ──────────────
def _pad_doc(name: str, desc: str, n_lines: int = 18) -> str:
    """生成填充文档注释，把每个函数撑到 ~55 行，模拟真实大文件。"""
    lines = [f'        {desc}', '', '        详细说明：']
    for i in range(n_lines):
        lines.append(f'        - 实现要点 {i + 1}: {name} 在处理边界输入时遵循统一的校验与规范化流程。')
    lines.append('')
    return "\n".join(lines)


def _big_func(idx: int, name: str, desc: str, body: str) -> str:
    header = f'def {name}(text: str) -> str:\n    """\n{_pad_doc(name, desc)}\n    """\n'
    comment = f"    # 第 {idx} 节：{desc}\n    # 输入假设：text 为任意字符串；输出见上方规范。\n"
    return header + comment + body + "\n\n"


BIG_MODULE = '"""文本处理工具库（共 10 节，每节一个函数，其中各藏 1 个 bug）"""\n\n\n' + "".join([
    _big_func(1, "sec01_title_case", "标题化：每个单词首字母大写，其余小写",
              '    return " ".join(w.capitalize() if w else w for w in text.split(" "))'),
    _big_func(2, "sec02_word_count", "统计单词数（按空白切分，空串为 0）",
              '    words = text.split()\n    return str(len(words) + 1)  # BUG: 多加 1'),
    _big_func(3, "sec03_strip_accents", "去除常见重音符号（é→e, à→a 等）",
              '    table = str.maketrans("éèêëàâäùûüôöîïç", "eeeeaaauuuooiic")\n    return text.translate(table)'),
    _big_func(4, "sec04_collapse_spaces", "把连续空白折叠为单个空格并去除首尾空白",
              '    return " ".join(text.split()) if text else text'),
    _big_func(5, "sec05_first_sentence", "取第一个句子（以 . ! ? 结尾，含标点）",
              '    import re\n    m = re.search(r"[.!?]", text)\n    return text[: m.start()] if m else text  # BUG: 丢了结尾标点'),
    _big_func(6, "sec06_wrap_width", "按宽度折行（简单按空格断行，词不拆）",
              '    words = text.split()\n    lines, cur = [], ""\n    for w in words:\n'
              '        if len(cur) + len(w) + 1 > 20:\n            lines.append(cur)\n            cur = w\n'
              '        else:\n            cur = (cur + " " + w).strip()\n'
              '    if cur:\n        lines.append(cur)\n    return "\\n".join(lines)'),
    _big_func(7, "sec07_count_char", "统计指定字符出现次数（默认区分大小写）",
              '    ch = "e"\n    return str(text.count(ch.lower() + ch.upper()))  # BUG: count 不接受多字符统计'),
    _big_func(8, "sec08_reverse_lines", "按行反转文本（行序颠倒，行内容不变）",
              '    lines = text.split("\\n")\n    return "\\n".join(reversed(lines))'),
    _big_func(9, "sec09_extract_numbers", "提取文本中的所有整数，用逗号连接返回",
              '    import re\n    nums = re.findall(r"\\d+", text)\n    return ",".join(nums) if nums else ""'),
    _big_func(10, "sec10_swap_case", "大小写互换",
              '    return text.swapcase()'),
])

BIG_TEST = '''import pytest
from text_proc import (
    sec01_title_case, sec02_word_count, sec03_strip_accents, sec04_collapse_spaces,
    sec05_first_sentence, sec06_wrap_width, sec07_count_char, sec08_reverse_lines,
    sec09_extract_numbers, sec10_swap_case,
)


# 第 1 节
def test_sec01_basic(): assert sec01_title_case("hello world") == "Hello World"
def test_sec01_mixed(): assert sec01_title_case("hELLO wORLD") == "Hello World"

# 第 2 节
def test_sec02_basic(): assert sec02_word_count("one two three") == "3"
def test_sec02_empty(): assert sec02_word_count("") == "0"

# 第 3 节
def test_sec03(): assert sec03_strip_accents("café crème") == "cafe creme"

# 第 4 节
def test_sec04(): assert sec04_collapse_spaces("  a   b\\t c ") == "a b c"

# 第 5 节
def test_sec05(): assert sec05_first_sentence("Hello world. Bye now.") == "Hello world."
def test_sec05_no_punct(): assert sec05_first_sentence("no end") == "no end"

# 第 6 节
def test_sec06():
    out = sec06_wrap_width("aa bb cc dd ee ff gg hh ii jj kk")
    assert all(len(line) <= 20 for line in out.split("\\n"))

# 第 7 节
def test_sec07(): assert sec07_count_char("EeE e") == "4"

# 第 8 节
def test_sec08(): assert sec08_reverse_lines("a\\nb\\nc") == "c\\nb\\na"

# 第 9 节
def test_sec09(): assert sec09_extract_numbers("a1 b22 c333") == "1,22,333"
def test_sec09_none(): assert sec09_extract_numbers("none") == ""

# 第 10 节
def test_sec10(): assert sec10_swap_case("AbC") == "aBc"
'''

BIG_SEED_FILES = {
    "text_proc.py": BIG_MODULE,
    "test_text_proc.py": BIG_TEST,
}

BIG_TASK = (
    "工作区里有一个约 300 行的文本处理库 text_proc.py，共 10 节（sec01–sec10），"
    "以及配套测试 test_text_proc.py。其中第 2、5、7 节藏有 bug，相关测试会失败。\n"
    "请严格遵守以下流程：\n"
    "- 按节顺序逐节检查：每处理一节，先完整阅读 text_proc.py 中该节的实现与文档，"
    "再运行该节对应的测试，若失败则修复并重跑确认；\n"
    "- 由于文件较大，每次阅读请用 read_file 读取完整文件以便理解上下文；\n"
    "- 不许修改测试文件；\n"
    "- 全部 10 节检查完成后，最后运行一次完整 pytest 确认 0 失败。"
)


def seed_workspace(workspace_dir: str, n_modules: int | None = None,
                   big: bool = False) -> None:
    root = Path(workspace_dir)
    # 清空旧产物，保证两次运行初始状态一致
    if root.exists():
        import shutil
        shutil.rmtree(root)
    root.mkdir(parents=True)
    files = BIG_SEED_FILES if big else SEED_FILES
    for name, content in files.items():
        if not big and n_modules is not None:
            # 只保留前 n_modules 个模块及其测试
            import re
            m = re.search(r"m(\d\d)", name)
            if m and int(m.group(1)) > n_modules:
                continue
        (root / name).write_text(content, encoding="utf-8")


def run_once(cache_enabled: bool, tag: str) -> dict:
    config = AgentConfig()
    config.cache_enabled = cache_enabled
    config.max_rounds = 40
    # 每个 run 用独立工作区，避免互相看到对方产物
    config.workspace_dir = f".agent_workspace_{tag}"
    seed_workspace(config.workspace_dir)

    rounds_usage: list[dict] = []
    events_log: list[str] = []

    def on_event(ev: dict) -> None:
        t = ev.get("type")
        if t == "debug_raw":
            usage = (ev.get("raw") or {}).get("usage") or {}
            if usage:
                rounds_usage.append(usage)
        elif t in ("round", "tool_call", "task_complete", "compress", "loop_signal"):
            events_log.append(f"{t}: {ev.get('num') or ev.get('name') or ev.get('summary') or ev.get('signal') or ''}")
            if t == "round":
                print(f"  [{tag}] round {ev['num']}", flush=True)

    loop = AgentLoop(config=config, on_event=on_event)
    t0 = time.time()
    result = loop.run(LONG_TASK)
    elapsed = time.time() - t0

    total_prompt = sum(u.get("prompt_tokens", 0) for u in rounds_usage)
    total_cached = sum((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                       for u in rounds_usage)
    return {
        "tag": tag,
        "cache_enabled": cache_enabled,
        "success": result.success,
        "reason": result.reason,
        "summary": result.summary[:200],
        "rounds": result.stats.rounds,
        "elapsed_s": round(elapsed, 1),
        "input_tokens_reported": result.stats.input_tokens,
        "output_tokens": result.stats.output_tokens,
        "total_prompt_tokens": total_prompt,
        "total_cached_tokens": total_cached,
        "per_round_usage": rounds_usage,
        "events": events_log,
    }


def report(a: dict, b: dict) -> None:
    print("\n" + "=" * 64)
    print(f"{'指标':<26}{'缓存关 (A)':>16}{'缓存开 (B)':>16}")
    print("-" * 64)
    print(f"{'任务成功':<26}{str(a['success']):>16}{str(b['success']):>16}")
    print(f"{'轮数':<26}{a['rounds']:>16}{b['rounds']:>16}")
    print(f"{'耗时(s)':<26}{a['elapsed_s']:>16}{b['elapsed_s']:>16}")
    print(f"{'未命中输入 tokens':<26}{a['total_prompt_tokens']:>16,}{b['total_prompt_tokens']:>16,}")
    print(f"{'命中缓存 tokens':<26}{a['total_cached_tokens']:>16,}{b['total_cached_tokens']:>16,}")
    ea = a["total_prompt_tokens"] + a["total_cached_tokens"] * 0.1
    eb = b["total_prompt_tokens"] + b["total_cached_tokens"] * 0.1
    print(f"{'等效输入计费单位(0.1x)':<26}{ea:>16,.0f}{eb:>16,.0f}")
    print("-" * 64)
    if ea > 0:
        print(f"输入成本变化：{(1 - eb / ea) * 100:+.1f}%（正值=节省）")
    print("\n⚠ 注意：真实任务两次运行的轨迹不同（模型生成有随机性），")
    print("  轮数/上下文大小不完全一致，以上是一次采样的参考值。")


def main() -> None:
    print("▶ 第一次：缓存关闭")
    a = run_once(False, "off")
    print(f"  完成：success={a['success']} rounds={a['rounds']} "
          f"prompt={a['total_prompt_tokens']:,} cached={a['total_cached_tokens']:,}")

    print("\n▶ 第二次：缓存开启")
    b = run_once(True, "on")
    print(f"  完成：success={b['success']} rounds={b['rounds']} "
          f"prompt={b['total_prompt_tokens']:,} cached={b['total_cached_tokens']:,}")

    report(a, b)

    out = Path("benchmarks") / f"real_task_benchmark_{int(time.time())}.json"
    out.write_text(json.dumps({"off": a, "on": b}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n原始数据已保存: {out}")


if __name__ == "__main__":
    main()
