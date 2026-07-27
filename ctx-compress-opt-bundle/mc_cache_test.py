#!/usr/bin/env python3
"""
微压缩前缀稳定性测试：对比两次 follow-up 的 token 花费。

实验设计
--------
  Turn 0（基础轮）：让 agent 读一个固定文件，获取基准工具结果。
  Turn A（< 5min）：立即追问，不触发 microcompact。
                    预期：高缓存命中，cache_read 大。
  Turn B（模拟 > 5min）：设置 _force_time_based_mc 再追问。
                    预期（旧行为）：microcompact 清工具结果，下一轮全文恢复，前缀变动。
                    预期（新行为）：清理后冻结，turn B 及之后前缀稳定，cache_read 仍高。

工具：read_file，固定路径，确定参数，确定结果。
API 日志：直接从 agent.conversation_loop 日志的格式化行解析。
"""

import sys, os, re, logging, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 提前加载 ~/.hermes/.env，让 API key 进入 os.environ
try:
    _env_path = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(_env_path):
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())
except Exception:
    pass

# ── 日志配置（关掉 DEBUG 噪音，只看 INFO 里的 API call 行）─────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s %(message)s",
)
# 把 agent 日志输出到这里以便解析
import io
_log_capture = io.StringIO()
_handler = logging.StreamHandler(_log_capture)
_handler.setLevel(logging.INFO)
logging.getLogger("agent.conversation_loop").addHandler(_handler)
logging.getLogger("agent.conversation_loop").setLevel(logging.INFO)

from run_agent import AIAgent
from hermes_cli.config import get_hermes_home
from hermes_logging import setup_logging
setup_logging()

# ── 被测文件（固定，确定结果）─────────────────────────────────────────────
TARGET_FILE = "/home/administrator/projects/hermes-agent/ctx-compress-opt/runs/compress_20260708_101722.txt"
PROMPT_READ  = f'请用 read_file 工具读取文件 {TARGET_FILE}，把前三行内容原文告诉我。'
PROMPT_FOLLOWUP = '刚才读取的那个文件，第一行是什么内容？'

# ── 解析日志里的 API call 行 ────────────────────────────────────────────────
_API_RE = re.compile(
    r"API call #(\d+).*?in=(\d+) out=(\d+) total=(\d+) latency=([\d.]+)s"
    r"(?:.*?cache=(\d+)/(\d+))?"
)

def _parse_last_api_call(log_text: str) -> dict:
    """从日志文本里取最后一次 API call 的 token 统计。"""
    rows = _API_RE.findall(log_text)
    if not rows:
        return {}
    n, inp, out, total, lat, cache_r, cache_total = rows[-1]
    d = dict(call=int(n), inp=int(inp), out=int(out),
             total=int(total), latency=float(lat))
    if cache_r:
        d["cache_read"] = int(cache_r)
        d["cache_pct"] = round(100 * int(cache_r) / int(cache_total))
    else:
        d["cache_read"] = 0
        d["cache_pct"] = 0
    return d

def _reset_log():
    _log_capture.truncate(0)
    _log_capture.seek(0)

def _last_stats():
    return _parse_last_api_call(_log_capture.getvalue())

# ── 创建 agent ─────────────────────────────────────────────────────────────
def make_agent():
    import yaml
    cfg_path = get_hermes_home() / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = cfg.get("model", {}).get("default", "qwen3.7-plus")
    provider = cfg.get("model", {}).get("provider", "alibaba")
    base_url = cfg.get("model", {}).get("base_url", "")
    api_key_env = "DASHSCOPE_API_KEY"
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        # 从 auth.json 读
        import json as _json
        auth = get_hermes_home() / "auth.json"
        if auth.exists():
            data = _json.loads(auth.read_text())
            api_key = data.get("api_key") or data.get("key") or ""
    agent = AIAgent(
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        max_iterations=10,
        quiet_mode=True,
    )
    return agent

# ── 主流程 ─────────────────────────────────────────────────────────────────
def run():
    print("=" * 60)
    print("微压缩缓存前缀稳定性测试")
    print("=" * 60)

    agent = make_agent()
    history = []

    # ── Turn 0：基础轮，让 agent 读文件 ────────────────────────────────────
    print("\n[Turn 0] 基础轮：读取固定文件...")
    _reset_log()
    t0 = time.time()
    res = agent.run(PROMPT_READ, conversation_history=history)
    t0_elapsed = time.time() - t0
    history = res["messages"]
    s0 = _last_stats()
    print(f"  in={s0.get('inp')}  out={s0.get('out')}  "
          f"cache_read={s0.get('cache_read')}({s0.get('cache_pct')}%)  "
          f"latency={s0.get('latency')}s")
    print(f"  agent 回复: {res.get('final_response','')[:120]}")

    # ── Turn A：立即追问（< 5min，microcompact 不触发）─────────────────────
    print("\n[Turn A] 立即追问（gap ≈ 0s，无 microcompact）...")
    _reset_log()
    tA = time.time()
    resA = agent.run(PROMPT_FOLLOWUP, conversation_history=history)
    tA_elapsed = time.time() - tA
    historyA = resA["messages"]
    sA = _last_stats()
    print(f"  in={sA.get('inp')}  out={sA.get('out')}  "
          f"cache_read={sA.get('cache_read')}({sA.get('cache_pct')}%)  "
          f"latency={sA.get('latency')}s")
    print(f"  agent 回复: {resA.get('final_response','')[:120]}")
    print(f"  mc_state replacements: {len(getattr(agent, '_microcompact_state', None) and agent._microcompact_state.replacements or {})}")

    # ── 模拟 > 5min gap：设置 _force_time_based_mc ─────────────────────────
    print("\n[模拟 5min+ gap] 设置 _force_time_based_mc = True ...")
    agent._force_time_based_mc = True
    # 同时把 _last_api_finished_ms 推到 10 分钟前，让 evaluate_time_based_trigger 也能从消息里感知
    agent._last_api_finished_ms = (time.time() - 700) * 1000   # 700s 前

    # ── Turn B：模拟冷缓存追问 ─────────────────────────────────────────────
    print("\n[Turn B] 模拟 5min+ 后追问（microcompact 应触发）...")
    _reset_log()
    # 从 Turn 0 的 history 开始（而不是 historyA），模拟"同一个长对话但刚回来"
    resB = agent.run(PROMPT_FOLLOWUP, conversation_history=history)
    sB = _last_stats()
    print(f"  in={sB.get('inp')}  out={sB.get('out')}  "
          f"cache_read={sB.get('cache_read')}({sB.get('cache_pct')}%)  "
          f"latency={sB.get('latency')}s")
    print(f"  agent 回复: {resB.get('final_response','')[:120]}")
    mc_state = getattr(agent, '_microcompact_state', None)
    print(f"  mc_state replacements: {len(mc_state.replacements) if mc_state else 0}")

    # ── Turn C：Turn B 之后立即再追问（验证冻结重放，前缀是否稳定）────────
    print("\n[Turn C] Turn B 后立即再追问（验证冻结重放）...")
    _reset_log()
    historyB = resB["messages"]
    resC = agent.run(PROMPT_FOLLOWUP, conversation_history=historyB)
    sC = _last_stats()
    print(f"  in={sC.get('inp')}  out={sC.get('out')}  "
          f"cache_read={sC.get('cache_read')}({sC.get('cache_pct')}%)  "
          f"latency={sC.get('latency')}s")
    print(f"  agent 回复: {resC.get('final_response','')[:120]}")

    # ── 汇总报告 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("汇总报告")
    print("=" * 60)
    print(f"{'场景':<30} {'in tokens':>10} {'cache_read':>12} {'cache%':>8} {'latency':>10}")
    print("-" * 75)
    rows = [
        ("Turn 0  基础轮(读文件)",       s0),
        ("Turn A  立即追问(<5min)",       sA),
        ("Turn B  模拟5min+后追问",       sB),
        ("Turn C  Turn B后立即追问",      sC),
    ]
    for label, s in rows:
        print(f"  {label:<28} {s.get('inp','-'):>10} {s.get('cache_read','-'):>12} "
              f"{s.get('cache_pct','-'):>7}% {s.get('latency','-'):>9}s")
    print()
    print("关键对比：")
    if sA.get('cache_pct') is not None and sB.get('cache_pct') is not None:
        print(f"  Turn A cache% = {sA['cache_pct']}%  ← 立即追问，缓存热")
        print(f"  Turn B cache% = {sB['cache_pct']}%  ← 触发 microcompact 后")
        print(f"  Turn C cache% = {sC['cache_pct']}%  ← 冻结重放后（新行为：应接近 A 或更高）")
        if sC.get('inp') and sB.get('inp'):
            delta = sC['inp'] - sB['inp']
            print(f"  Turn C vs B in tokens 差值: {delta:+d}（正=更大，负=更小）")
    print()
    mc_state = getattr(agent, '_microcompact_state', None)
    if mc_state:
        print(f"  最终 mc_state.replacements 条数: {len(mc_state.replacements)}")
        for cid, v in mc_state.replacements.items():
            print(f"    {cid[:20]}… → {v[:60]!r}")

if __name__ == "__main__":
    run()
