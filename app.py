"""
自主编程 Agent — Streamlit 界面

启动：
  streamlit run app.py
"""
from __future__ import annotations

import json as _json_mod
import os
import queue
import threading
import time
from pathlib import Path

import streamlit as st

# 自动加载 .env 文件
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ[k.strip()] = v.strip()

from agent.loop import AgentLoop, AgentResult
from config import AgentConfig

st.set_page_config(page_title="自主编程 Agent", page_icon="🤖", layout="wide")


def _init_state() -> None:
    for k, v in {
        "running": False, "events": [], "result": None,
        "event_queue": queue.Queue(), "thread": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def _run_agent(task: str, config: AgentConfig, q: queue.Queue) -> None:
    import traceback
    def on_event(e: dict) -> None: q.put(e)
    agent = AgentLoop(config=config, on_event=on_event)
    try:
        agent.run(task)
    except Exception as e:
        tb = traceback.format_exc()
        q.put({"type": "error", "message": str(e), "traceback": tb})
        q.put({"type": "done", "result": AgentResult(success=False, reason=str(e))})


# ── 侧边栏：配置 + 任务输入 ────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ 配置")

    api_key = st.text_input(
        "API Key", value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password", help="或写入 .env",
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    base_url = st.text_input(
        "中转 Base URL（可选）",
        value=os.environ.get("ANTHROPIC_BASE_URL", ""),
        placeholder="https://your-proxy.com/v1",
    )
    if base_url:
        os.environ["ANTHROPIC_BASE_URL"] = base_url
    else:
        os.environ.pop("ANTHROPIC_BASE_URL", None)

    default_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
    model_options = ["claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5-20251001"]
    if default_model not in model_options:
        model_options.insert(0, default_model)
    model = st.selectbox("模型", model_options, index=model_options.index(default_model))

    max_rounds = st.slider("最大执行轮次", 10, 100, 50)
    workspace = st.text_input("工作目录", value=".agent_workspace")

    st.divider()
    st.caption("核心模块：🗜️ 上下文压缩 | 🔍 循环检测")


# ── 主区域 ─────────────────────────────────────────────────────────────────────

st.title("🤖 自主编程 Agent")

# 任务输入栏（主区域顶部）
task_col, btn_col1, btn_col2 = st.columns([4, 1, 1])
with task_col:
    task_input = st.text_input(
        "任务描述",
        placeholder="用 Python 写一个 CSV 解析器，支持 UTF-8/GBK 编码自动检测，带完整单元测试",
        disabled=st.session_state.running,
        label_visibility="collapsed",
    )
with btn_col1:
    start_clicked = st.button(
        "🚀 执行", disabled=st.session_state.running or not task_input.strip(),
        use_container_width=True, type="primary",
    )
with btn_col2:
    clear_clicked = st.button(
        "🗑️ 清空", disabled=st.session_state.running, use_container_width=True,
    )

if clear_clicked:
    st.session_state.events = []
    st.session_state.result = None
    st.rerun()

if start_clicked and task_input.strip():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("请先在侧边栏填写 API Key")
    else:
        config = AgentConfig(model=model, max_rounds=max_rounds, workspace_dir=workspace)
        st.session_state.events = []
        st.session_state.result = None
        st.session_state.running = True
        q: queue.Queue = queue.Queue()
        st.session_state.event_queue = q
        threading.Thread(target=_run_agent, args=(task_input.strip(), config, q), daemon=True).start()
        st.rerun()

tab_log, tab_files, tab_debug = st.tabs(["📊 执行日志", "📁 生成文件", "🔍 调试"])

# ── Tab 1: 执行日志 ──────────────────────────────────────────────────────────

with tab_log:
    if st.session_state.running:
        q = st.session_state.event_queue
        for _ in range(30):
            try:
                event = q.get_nowait()
                st.session_state.events.append(event)
                if event.get("type") == "done":
                    st.session_state.running = False
                    st.session_state.result = event.get("result")
                    break
            except queue.Empty:
                break

    if not st.session_state.events:
        st.info("输入任务后点击侧边栏的 🚀 开始")

    for event in st.session_state.events:
        t = event.get("type")

        if t == "round":
            st.divider()
            st.caption(f"第 {event['num']} 轮")

        elif t == "reasoning":
            text = event.get("text", "").strip()
            if text:
                with st.expander("💭 思考过程", expanded=False):
                    st.text(text[:2000])

        elif t == "tool_call":
            name = event.get("name", "")
            inp = event.get("input", {})
            emoji = {"write_file": "📝", "read_file": "📖", "execute_python": "▶️",
                     "run_command": "💻", "run_tests": "🧪", "task_complete": "✅"}.get(name, "🔧")
            detail = ""
            if name == "write_file":
                detail = f" → `{inp.get('path', '?')}` ({len(inp.get('content', '').splitlines())} 行)"
            elif name == "run_command":
                detail = f" → `{inp.get('command', '')[:80]}`"
            elif name == "run_tests":
                detail = f" → `{inp.get('path', '.')}`"
            elif name == "execute_python":
                code = inp.get("code", "").strip()
                detail = f" → `{code[:60]}...`" if len(code) > 60 else ""
            st.markdown(f"{emoji} **{name}**{detail}")

        elif t == "tool_result":
            ok = event.get("success", True)
            status = "✅" if ok else "❌"
            output = (event.get("output") or "").strip()
            if output:
                with st.expander(f"{status} 输出", expanded=not ok):
                    st.code(output[:2000], language="text")
            else:
                st.caption(f"{status} 无输出")

        elif t == "task_complete":
            st.success(f"✅ {event.get('summary', '任务完成')}")

        elif t == "compress":
            st.info(f"🗜️ 上下文压缩 L{event['level']} — 节省约 {event['tokens_saved']:,} tokens")

        elif t == "loop_signal":
            sig = event["signal"]
            icons = {"light": "🟡", "medium": "🟠", "heavy": "🔴"}
            labels = {"light": "告警", "medium": "纠偏", "heavy": "熔断"}
            st.warning(f"{icons.get(sig, '')} 循环检测 [{labels.get(sig, sig)}] — 连续 {event['count']} 轮无进展")

        elif t == "error":
            st.error(f"**{event['message']}**")
            tb = event.get("traceback", "")
            if tb:
                with st.expander("堆栈"):
                    st.code(tb[-2000:])

        elif t == "done":
            r = event.get("result")
            if r and r.success:
                st.balloons()
            if r:
                s = r.stats
                st.success(f"完成 | {s.rounds} 轮 | 输入 {s.input_tokens:,} tokens | 输出 {s.output_tokens:,} tokens")

    if st.session_state.running:
        st.spinner("Agent 执行中...")
        time.sleep(0.4)
        st.rerun()

# ── Tab 2: 生成文件 ──────────────────────────────────────────────────────────

with tab_files:
    files = []
    if Path(workspace).exists():
        files = [f for f in sorted(Path(workspace).rglob("*"))
                 if f.is_file() and not f.name.startswith("tr_")]

    if files:
        selected = st.selectbox(
            "选择文件", [str(f.relative_to(Path(workspace))) for f in files],
            label_visibility="collapsed",
        )
        if selected:
            fp = Path(workspace) / selected
            try:
                content = fp.read_text(encoding="utf-8")
                st.code(content, language=fp.suffix.lstrip(".") or "text", line_numbers=True)
            except Exception as e:
                st.error(str(e))
    else:
        st.info("暂无生成的文件")

# ── Tab 3: 调试 ──────────────────────────────────────────────────────────────

with tab_debug:
    raw_events = [e for e in st.session_state.events if e.get("type") == "debug_raw"]
    if raw_events:
        e = raw_events[-1]
        st.json(e.get("raw", {}))
    else:
        st.info("执行一次任务后这里会显示最后一次 API 响应")
