/**
 * Ricode Agent — VS Code 扩展
 * ──────────────────────────────────────
 * 打开一个 Webview 面板：输入任务 → POST /run（SSE 流）→ 实时渲染 Agent 事件。
 * 事件协议与 Python 端 AgentLoop.on_event 一致。
 * 激活时自动启动 Python 后端（uvicorn），关闭 VS Code 时自动停止。
 */
import * as vscode from "vscode";
import * as path from "path";
import { spawn, ChildProcess } from "child_process";

let backend: ChildProcess | undefined;

export function activate(context: vscode.ExtensionContext) {
  const cfg = vscode.workspace.getConfiguration("codingAgent");
  if (cfg.get<boolean>("autoStartBackend", true)) {
    void ensureBackend(context, cfg);
  }
  const cmd = vscode.commands.registerCommand("codingAgent.openPanel", () => {
    AgentPanel.show(context);
  });
  context.subscriptions.push(cmd);
}

export function deactivate() {
  // 关闭 VS Code 时停止插件拉起的后端
  backend?.kill();
  backend = undefined;
}

/** 健康检查：后端已在跑则直接复用，否则自动拉起 uvicorn */
async function ensureBackend(
  context: vscode.ExtensionContext,
  cfg: vscode.WorkspaceConfiguration
) {
  const serverUrl = cfg.get<string>("serverUrl", "http://127.0.0.1:8765");
  const port = new URL(serverUrl).port || "8765";

  if (await isHealthy(serverUrl)) {
    return; // 已有后端在运行（比如手动起的 uvicorn），不重复启动
  }

  const projectRoot =
    cfg.get<string>("projectRoot", "") ||
    path.resolve(context.extensionPath, ".."); // 插件位于项目根目录的 vscode-extension/ 下

  vscode.window.setStatusBarMessage("$(sync~spin) Ricode Agent 后端启动中…", 10000);
  backend = spawn(
    "python",
    ["-m", "uvicorn", "server:app", "--port", port],
    { cwd: projectRoot, shell: true, stdio: "ignore" }
  );
  backend.on("error", (e) => {
    vscode.window.showErrorMessage(`Ricode Agent 后端启动失败：${e.message}`);
  });

  // 轮询等待就绪（最多 30 秒）
  for (let i = 0; i < 30; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    if (await isHealthy(serverUrl)) {
      vscode.window.setStatusBarMessage("$(check) Ricode Agent 后端已就绪", 5000);
      return;
    }
  }
  vscode.window.showWarningMessage(
    "Ricode Agent 后端 30 秒内未就绪，请检查 Python 环境与 .env 配置"
  );
}

async function isHealthy(serverUrl: string): Promise<boolean> {
  try {
    const resp = await fetch(`${serverUrl}/health`, {
      signal: AbortSignal.timeout(1500),
    });
    return resp.ok;
  } catch {
    return false;
  }
}

interface AgentEvent {
  type: string;
  [key: string]: unknown;
}

class AgentPanel {
  private static current: AgentPanel | undefined;
  private readonly panel: vscode.WebviewPanel;
  private abort: AbortController | undefined;

  static show(context: vscode.ExtensionContext) {
    if (AgentPanel.current) {
      AgentPanel.current.panel.reveal();
      return;
    }
    const panel = vscode.window.createWebviewPanel(
      "codingAgent",
      "Ricode Agent",
      vscode.ViewColumn.Beside,
      { enableScripts: true, retainContextWhenHidden: true }
    );
    // 面板标签图标（SVG，非文本 emoji）
    panel.iconPath = vscode.Uri.joinPath(
      context.extensionUri, "media", "ricode.svg"
    );
    AgentPanel.current = new AgentPanel(panel, context);
  }

  private constructor(
    panel: vscode.WebviewPanel,
    private readonly context: vscode.ExtensionContext
  ) {
    this.panel = panel;
    this.panel.webview.html = renderHtml();
    this.panel.onDidDispose(() => {
      this.abort?.abort();
      AgentPanel.current = undefined;
    });
    this.panel.webview.onDidReceiveMessage((msg) => {
      if (msg.type === "run") {
        void this.runTask(msg.task);
      } else if (msg.type === "stop") {
        this.abort?.abort();
        this.post({ type: "stopped" });
      }
    });
  }

  private post(msg: unknown) {
    void this.panel.webview.postMessage(msg);
  }

  /** 调用 FastAPI 后端，按 SSE 逐事件转发给 Webview */
  private async runTask(task: string) {
    const cfg = vscode.workspace.getConfiguration("codingAgent");
    const serverUrl = cfg.get<string>("serverUrl", "http://127.0.0.1:8765");
    const sandbox = cfg.get<string>("sandbox", "docker");
    const workspace = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;

    this.abort = new AbortController();
    try {
      const resp = await fetch(`${serverUrl}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task, sandbox, workspace }),
        signal: this.abort.signal,
      });
      if (!resp.ok || !resp.body) {
        this.post({ type: "error", message: `HTTP ${resp.status}` });
        return;
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // SSE 帧以空行分隔
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";
        for (const frame of frames) {
          const line = frame.trim();
          if (line.startsWith("data:")) {
            const event = JSON.parse(line.slice(5).trim()) as AgentEvent;
            this.post({ type: "event", event });
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        this.post({ type: "error", message: String(e) });
      }
    }
  }
}

function renderHtml(): string {
  return /* html */ `<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body {
    font-family: var(--vscode-font-family);
    color: var(--vscode-foreground);
    display: flex; flex-direction: column;
    overflow: hidden;
  }

  /* ── 消息区 ─────────────────────────────── */
  #chat { flex: 1; overflow-y: auto; padding: 16px 14px 8px; }
  #chat::-webkit-scrollbar { width: 8px; }
  #chat::-webkit-scrollbar-thumb { background: var(--vscode-scrollbarSlider-background); border-radius: 4px; }

  .welcome {
    height: 100%; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 10px;
    color: var(--vscode-descriptionForeground); text-align: center;
  }
  .welcome .brand { font-size: 17px; font-weight: 600; color: var(--vscode-foreground); letter-spacing: .5px; }
  .welcome .hint { font-size: 12px; opacity: .8; line-height: 1.8; }

  .msg-user {
    margin: 8px 0 12px auto; max-width: 88%;
    background: var(--vscode-button-background);
    color: var(--vscode-button-foreground);
    padding: 9px 13px; border-radius: 14px 14px 4px 14px;
    font-size: 13px; line-height: 1.5; white-space: pre-wrap;
    width: fit-content;
  }
  .round-divider {
    display: flex; align-items: center; gap: 8px;
    margin: 14px 0 8px;
    color: var(--vscode-descriptionForeground); font-size: 11px;
  }
  .round-divider::before, .round-divider::after {
    content: ""; flex: 1; height: 1px;
    background: var(--vscode-panel-border);
  }
  .reasoning {
    font-size: 13px; line-height: 1.65; margin: 6px 0;
    white-space: pre-wrap; word-break: break-word;
  }
  .card {
    margin: 6px 0; border: 1px solid var(--vscode-panel-border);
    border-radius: 8px; overflow: hidden; font-size: 12px;
  }
  .card-head {
    display: flex; align-items: center; gap: 7px;
    padding: 7px 10px; cursor: pointer; user-select: none;
    background: var(--vscode-editor-background);
  }
  .card-head:hover { background: var(--vscode-list-hoverBackground); }
  .card-head .arrow { transition: transform .15s; font-size: 10px; opacity: .6; }
  .card.open .card-head .arrow { transform: rotate(90deg); }
  .card-head .name { font-weight: 600; }
  .card-head .arg {
    color: var(--vscode-descriptionForeground);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1;
  }
  .badge { margin-left: auto; font-size: 11px; }
  .badge.ok  { color: var(--vscode-charts-green); }
  .badge.err { color: var(--vscode-charts-red); }
  .badge.run { color: var(--vscode-charts-blue); }
  .card-body {
    display: none; padding: 9px 11px;
    border-top: 1px solid var(--vscode-panel-border);
    font-family: var(--vscode-editor-font-family);
    white-space: pre-wrap; word-break: break-word;
    max-height: 300px; overflow-y: auto;
  }
  .card.open .card-body { display: block; }

  .notice { font-size: 11px; margin: 8px 0; display: flex; gap: 6px; align-items: center; }
  .notice.compress { color: var(--vscode-charts-yellow); }
  .notice.loop    { color: var(--vscode-charts-red); }
  .notice.done    { color: var(--vscode-charts-green); font-size: 13px; font-weight: 600; margin: 12px 0 4px; }
  .notice.stats   { color: var(--vscode-descriptionForeground); }
  .notice.error   { color: var(--vscode-charts-red); }

  /* ── 输入区（固定底部）───────────────────── */
  #composer {
    border-top: 1px solid var(--vscode-panel-border);
    padding: 10px 12px 12px;
    background: var(--vscode-sideBar-background, var(--vscode-editor-background));
  }
  #input-wrap {
    display: flex; align-items: flex-end; gap: 8px;
    border: 1px solid var(--vscode-input-border, var(--vscode-panel-border));
    border-radius: 10px; padding: 8px 8px 8px 12px;
    background: var(--vscode-input-background);
  }
  #input-wrap:focus-within { border-color: var(--vscode-focusBorder); }
  #task {
    flex: 1; border: none; outline: none; resize: none;
    background: transparent; color: var(--vscode-input-foreground);
    font-family: inherit; font-size: 13px; line-height: 1.5;
    max-height: 120px;
  }
  #task::placeholder { color: var(--vscode-input-placeholderForeground); }
  .btn {
    border: none; border-radius: 7px; padding: 6px 12px;
    font-size: 12px; cursor: pointer; flex-shrink: 0;
  }
  #send { background: var(--vscode-button-background); color: var(--vscode-button-foreground); }
  #send:hover { background: var(--vscode-button-hoverBackground); }
  #send:disabled { opacity: .45; cursor: default; }
  #stop {
    background: transparent; color: var(--vscode-charts-red);
    border: 1px solid var(--vscode-charts-red); display: none;
  }
  #status-line {
    font-size: 11px; color: var(--vscode-descriptionForeground);
    margin-top: 7px; min-height: 14px; display: flex; align-items: center; gap: 5px;
  }
  .spinner { display: inline-block; animation: spin 1s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
  <div id="chat">
    <div class="welcome" id="welcome">
      <svg class="logo" width="64" height="64" viewBox="0 0 128 128" fill="none">
        <defs>
          <linearGradient id="g" x1="0" y1="0" x2="128" y2="128" gradientUnits="userSpaceOnUse">
            <stop stop-color="#4F8CFF"/><stop offset="1" stop-color="#9B5CFF"/>
          </linearGradient>
        </defs>
        <rect x="8" y="8" width="112" height="112" rx="26" fill="url(#g)"/>
        <path d="M44 48 L28 64 L44 80" stroke="white" stroke-width="9" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
        <path d="M84 48 L100 64 L84 80" stroke="white" stroke-width="9" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
        <path d="M68 40 L56 68 H64 L60 88 L76 60 H66 L68 40 Z" fill="white"/>
      </svg>
      <div class="brand">Ricode Agent</div>
      <div class="hint">描述你的编程任务，Agent 将自主完成<br>代码生成 · 执行验证 · 报错修复 · 测试通过</div>
    </div>
  </div>

  <div id="composer">
    <div id="input-wrap">
      <textarea id="task" rows="1" placeholder="描述编程任务，Enter 发送，Shift+Enter 换行"></textarea>
      <button class="btn" id="stop">■ 停止</button>
      <button class="btn" id="send">发送 ➤</button>
    </div>
    <div id="status-line"></div>
  </div>

<script>
  const vscode = acquireVsCodeApi();
  const chat = document.getElementById("chat");
  const taskEl = document.getElementById("task");
  const sendBtn = document.getElementById("send");
  const stopBtn = document.getElementById("stop");
  const statusLine = document.getElementById("status-line");
  const TOOL_ICON = {
    write_file: "📝", read_file: "📖", execute_python: "🐍",
    run_command: "⌨️", run_tests: "🧪", search_code: "🔍",
  };
  let running = false;
  const pendingCards = {};   // tool name -> card，结果回来后填状态

  const scrollBottom = () => { chat.scrollTop = chat.scrollHeight; };
  const el = (cls, text) => {
    const d = document.createElement("div");
    d.className = cls; if (text != null) d.textContent = text;
    chat.appendChild(d); scrollBottom(); return d;
  };

  function setRunning(v) {
    running = v;
    sendBtn.disabled = v;
    stopBtn.style.display = v ? "block" : "none";
    statusLine.innerHTML = v ? '<span class="spinner">◌</span> Agent 运行中…' : "";
  }

  function submit() {
    const task = taskEl.value.trim();
    if (!task || running) return;
    const w = document.getElementById("welcome");
    if (w) w.remove();
    el("msg-user", task);
    taskEl.value = ""; taskEl.style.height = "auto";
    setRunning(true);
    vscode.postMessage({ type: "run", task });
  }
  sendBtn.onclick = submit;
  stopBtn.onclick = () => vscode.postMessage({ type: "stop" });
  taskEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
  });
  taskEl.addEventListener("input", () => {
    taskEl.style.height = "auto";
    taskEl.style.height = Math.min(taskEl.scrollHeight, 120) + "px";
  });

  function toolCard(name, input) {
    const card = document.createElement("div");
    card.className = "card";
    const arg = JSON.stringify(input || {});
    card.innerHTML =
      '<div class="card-head">' +
        '<span class="arrow">▶</span>' +
        '<span>' + (TOOL_ICON[name] || "🔧") + '</span>' +
        '<span class="name">' + name + '</span>' +
        '<span class="arg">' + (arg.length > 60 ? arg.slice(0, 60) + "…" : arg) + "</span>" +
        '<span class="badge run">运行中</span>' +
      "</div>" +
      '<div class="card-body"></div>';
    card.querySelector(".card-head").onclick = () => card.classList.toggle("open");
    chat.appendChild(card); scrollBottom();
    pendingCards[name] = card;
    return card;
  }

  function fillResult(name, output, success) {
    const card = pendingCards[name];
    if (!card) return;
    const badge = card.querySelector(".badge");
    badge.textContent = success ? "✓ 完成" : "✗ 失败";
    badge.className = "badge " + (success ? "ok" : "err");
    card.querySelector(".card-body").textContent =
      (output || "(无输出)").slice(0, 4000);
    if (!success) card.classList.add("open");   // 失败自动展开
    scrollBottom();
  }

  window.addEventListener("message", (e) => {
    const msg = e.data;
    if (msg.type === "error") { el("notice error", "⚠ " + msg.message); setRunning(false); return; }
    if (msg.type === "stopped") { el("notice", "— 已手动停止 —"); setRunning(false); return; }
    if (msg.type !== "event") return;
    const ev = msg.event;
    switch (ev.type) {
      case "round":
        el("round-divider", "Round " + ev.num); break;
      case "reasoning":
        el("reasoning", ev.text); break;
      case "tool_call":
        toolCard(ev.name, ev.input); break;
      case "tool_result":
        fillResult(ev.name, ev.output, ev.success); break;
      case "compress":
        el("notice compress", "🗜 上下文压缩 L" + ev.level + "，节省 " + ev.tokens_saved + " tokens"); break;
      case "loop_signal":
        el("notice loop", "🔁 检测到重复循环（" + ev.signal + "），已注入纠偏提示"); break;
      case "task_complete":
        el("notice done", "✔ " + ev.summary); break;
      case "done": {
        const r = ev.result;
        el("notice stats",
          (r.success ? "任务成功" : "任务未完成（" + (r.reason || "") + "）") +
          " · " + r.rounds + " 轮 · tokens " + r.input_tokens + "↑ / " + r.output_tokens + "↓");
        setRunning(false); break;
      }
    }
  });
</script>
</body>
</html>`;
}
