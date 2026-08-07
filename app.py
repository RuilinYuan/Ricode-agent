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
        "agent": None, "dashboard": {
            "round": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read": 0, "cost": 0.0, "api_errors": 0, "compressions": 0,
        },
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def _run_agent(task: str, config: AgentConfig, q: queue.Queue) -> None:
    import traceback
    def on_event(e: dict) -> None: q.put(e)
    agent = AgentLoop(config=config, on_event=on_event)
    # 保存引用供取消使用
    # (通过 st.session_state 跨线程传递)
    st.session_state.agent = agent
    try:
        agent.run(task)
    except Exception as e:
        tb = traceback.format_exc()
        q.put({"type": "error", "message": str(e), "traceback": tb})
        q.put({"type": "done", "result": AgentResult(success=False, reason=str(e))})


# ═══════════════════════════════════════════════════════════════════════════════
# 侧边栏：配置 + 实时仪表盘
# ═══════════════════════════════════════════════════════════════════════════════

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

    # 高级选项（折叠）
    with st.expander("🔧 高级选项"):
        compression_enabled = st.checkbox("上下文压缩", value=True, help="关闭后不触发分层压缩")
        cache_enabled = st.checkbox("Prompt Cache", value=True, help="关闭后不注入 cache_control 断点")
        exec_timeout = st.number_input("命令超时(秒)", value=30, min_value=5, max_value=300)
        max_retries = st.number_input("API 重试次数", value=3, min_value=0, max_value=10)

    st.divider()

    # ── 实时仪表盘 ──────────────────────────────────────────────────────────

    st.subheader("📊 实时仪表盘")

    dash = st.session_state.dashboard

    # 状态指示灯
    if st.session_state.running:
        st.markdown("🟢 **执行中**")
    elif st.session_state.result is not None:
        r = st.session_state.result
        if r.success:
            st.markdown("✅ **已完成**")
        else:
            st.markdown(f"❌ **{r.reason}**")
    else:
        st.markdown("⚪ **待命**")

    # Token 统计
    col1, col2 = st.columns(2)
    with col1:
        st.metric("轮次", dash.get("round", 0))
        st.metric("输入", f"{dash.get('input_tokens', 0):,}")
    with col2:
        st.metric("压缩", dash.get("compressions", 0))
        st.metric("输出", f"{dash.get('output_tokens', 0):,}")

    # 缓存
    cache_hit = dash.get("cache_read", 0)
    if cache_hit > 0:
        st.metric("缓存命中", f"{cache_hit:,} tokens")

    # 成本
    cost = dash.get("cost", 0.0)
    st.metric("💰 预估费用", f"${cost:.4f}")

    # API 错误
    api_errs = dash.get("api_errors", 0)
    if api_errs > 0:
        st.warning(f"⚠️ API 错误/重试: {api_errs} 次")

    st.divider()
    st.caption("核心模块：🗜️ 上下文压缩 | 🔍 循环检测 | 🔄 自动重试")


# ═══════════════════════════════════════════════════════════════════════════════
# 主区域
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🤖 自主编程 Agent")

# 任务输入栏（主区域顶部）
task_col, btn_col1, btn_col2, btn_col3 = st.columns([4, 1, 1, 1])
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
    stop_clicked = st.button(
        "⏹️ 停止", disabled=not st.session_state.running,
        use_container_width=True,
    )
with btn_col3:
    clear_clicked = st.button(
        "🗑️ 清空", disabled=st.session_state.running, use_container_width=True,
    )

if clear_clicked:
    st.session_state.events = []
    st.session_state.result = None
    st.session_state.dashboard = {
        "round": 0, "input_tokens": 0, "output_tokens": 0,
        "cache_read": 0, "cost": 0.0, "api_errors": 0, "compressions": 0,
    }
    st.rerun()

if stop_clicked and st.session_state.running:
    agent = st.session_state.get("agent")
    if agent:
        agent.cancel()
        st.toast("⏹️ 正在停止...", icon="⏹️")

if start_clicked and task_input.strip():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("请先在侧边栏填写 API Key")
    else:
        config = AgentConfig(
            model=model, max_rounds=max_rounds, workspace_dir=workspace,
            compression_enabled=compression_enabled,
            cache_enabled=cache_enabled,
            exec_timeout=exec_timeout,
            max_retries=max_retries,
        )
        st.session_state.events = []
        st.session_state.result = None
        st.session_state.running = True
        st.session_state.dashboard = {
            "round": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read": 0, "cost": 0.0, "api_errors": 0, "compressions": 0,
        }
        q: queue.Queue = queue.Queue()
        st.session_state.event_queue = q
        threading.Thread(target=_run_agent, args=(task_input.strip(), config, q), daemon=True).start()
        st.rerun()

tab_log, tab_files, tab_debug = st.tabs(["📊 执行日志", "📁 生成文件", "🔍 调试"])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1: 执行日志
# ═══════════════════════════════════════════════════════════════════════════════

with tab_log:
    if st.session_state.running:
        q = st.session_state.event_queue
        for _ in range(50):
            try:
                event = q.get_nowait()
                st.session_state.events.append(event)

                # 更新仪表盘数据
                if event.get("type") == "stats_update":
                    st.session_state.dashboard.update({
                        "round": event.get("round", 0),
                        "input_tokens": event.get("input_tokens", 0),
                        "output_tokens": event.get("output_tokens", 0),
                        "cache_read": event.get("cache_read_tokens", 0),
                        "cost": event.get("estimated_cost", 0.0),
                        "api_errors": event.get("api_errors", 0),
                        "compressions": event.get("compressions", 0),
                    })

                if event.get("type") == "done":
                    st.session_state.running = False
                    st.session_state.result = event.get("result")
                    break
            except queue.Empty:
                break

    if not st.session_state.events:
        st.info("输入任务后点击 🚀 开始")

    for event in st.session_state.events:
        t = event.get("type")

        if t == "stats_update":
            continue  # 不在日志流中显示

        elif t == "round":
            st.divider()
            st.caption(f"第 {event['num']} 轮")

        elif t == "reasoning":
            text = event.get("text", "").strip()
            if text:
                with st.expander("💭 思考过程", expanded=False):
                    st.text(text[:3000])
                    if len(text) > 3000:
                        st.caption(f"（共 {len(text)} 字符，已截断）")

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
            elif name == "search_code":
                detail = f" → `{inp.get('query', '')[:60]}`"
            st.markdown(f"{emoji} **{name}**{detail}")

        elif t == "tool_result":
            ok = event.get("success", True)
            status = "✅" if ok else "❌"
            output = (event.get("output") or "").strip()
            if output:
                with st.expander(f"{status} 输出", expanded=not ok):
                    st.code(output[:3000], language="text")
                    if len(output) > 3000:
                        st.caption(f"（共 {len(output)} 字符，已截断）")
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
                    st.code(tb[-3000:])

        elif t == "done":
            r = event.get("result")
            if r and r.success:
                st.balloons()
            if r:
                s = r.stats
                cache_info = ""
                if s.cache_read_tokens > 0:
                    cache_info = f" | 缓存命中 {s.cache_read_tokens:,}"
                st.success(
                    f"完成 | {s.rounds} 轮 "
                    f"| 输入 {s.input_tokens:,} tokens "
                    f"| 输出 {s.output_tokens:,} tokens"
                    f"{cache_info}"
                    f"\n\n💰 预估费用: **${s.estimated_cost_usd:.4f}**"
                )
                if s.api_errors > 0:
                    st.caption(f"期间经历了 {s.api_errors} 次 API 重试")

    if st.session_state.running:
        with st.spinner("Agent 执行中..."):
            time.sleep(0.4)
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2: 生成文件
# ═══════════════════════════════════════════════════════════════════════════════

with tab_files:
    # 执行期间也刷新文件列表
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
                lang = fp.suffix.lstrip(".") or "text"
                # 语法高亮映射
                lang_map = {"py": "python", "js": "javascript", "ts": "typescript",
                           "md": "markdown", "json": "json", "yaml": "yaml",
                           "yml": "yaml", "html": "html", "css": "css"}
                st.code(content, language=lang_map.get(lang, lang), line_numbers=True)
            except UnicodeDecodeError:
                st.warning("⚠️ 二进制文件，无法预览")
            except Exception as e:
                st.error(str(e))
    else:
        st.info("暂无生成的文件")

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3: 调试
# ═══════════════════════════════════════════════════════════════════════════════

with tab_debug:
    # API 调用历史
    raw_events = [e for e in st.session_state.events if e.get("type") == "debug_raw"]

    if raw_events:
        st.subheader(f"API 调用历史（{len(raw_events)} 次）")

        # 汇总统计
        total_prompt = 0
        total_completion = 0
        total_cache = 0
        for e in raw_events:
            usage = (e.get("raw") or {}).get("usage") or {}
            total_prompt += usage.get("prompt_tokens", 0)
            total_completion += usage.get("completion_tokens", 0)
            details = usage.get("prompt_tokens_details") or {}
            total_cache += details.get("cached_tokens", 0)

        c1, c2, c3 = st.columns(3)
        c1.metric("总输入", f"{total_prompt:,}")
        c2.metric("总输出", f"{total_completion:,}")
        c3.metric("缓存命中", f"{total_cache:,}")

        # 每次调用的详情
        if st.checkbox("展开每次调用详情"):
            for i, e in enumerate(raw_events):
                usage = (e.get("raw") or {}).get("usage") or {}
                details = usage.get("prompt_tokens_details") or {}
                with st.expander(f"#{i+1}: prompt={usage.get('prompt_tokens','?'):,} "
                               f"completion={usage.get('completion_tokens','?'):,} "
                               f"cache={details.get('cached_tokens',0):,}"):
                    st.json(e.get("raw", {}))
    else:
        st.info("执行一次任务后这里会显示 API 调用详情")
