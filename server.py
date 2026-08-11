"""
FastAPI 后端 — 供 VS Code 插件 / 第三方客户端调用
──────────────────────────────────────────────────
启动：
  uvicorn server:app --port 8765

接口：
  GET  /health        健康检查
  POST /run           提交任务，SSE 流式返回 Agent 事件
                      body: {"task": "...", "model": "...", "max_rounds": 50, "sandbox": "docker"}
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.loop import AgentLoop
from config import AgentConfig

app = FastAPI(title="Autonomous Coding Agent")


class RunRequest(BaseModel):
    task: str
    model: str | None = None
    max_rounds: int = 50
    sandbox: str | None = None      # "local" | "docker"
    workspace: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/run")
def run(req: RunRequest) -> StreamingResponse:
    """提交编程任务，以 SSE 流式推送 Agent 事件（与 Streamlit UI 同一事件协议）。"""
    q: queue.Queue = queue.Queue()

    def on_event(event: dict) -> None:
        q.put(event)

    def worker() -> None:
        try:
            config = AgentConfig(max_rounds=req.max_rounds)
            if req.model:
                config.model = req.model
            if req.sandbox:
                config.sandbox_backend = req.sandbox
            if req.workspace:
                config.workspace_dir = req.workspace
            result = AgentLoop(config=config, on_event=on_event).run(req.task)
            q.put({"type": "done", "result": {
                "success": result.success,
                "summary": result.summary,
                "reason": result.reason,
                "rounds": result.stats.rounds,
                "input_tokens": result.stats.input_tokens,
                "output_tokens": result.stats.output_tokens,
            }})
        except Exception as e:
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put(None)  # 结束哨兵

    threading.Thread(target=worker, daemon=True).start()

    async def event_stream():
        """异步生成器：从 worker 线程的 queue 取事件，逐条推送。

        用 asyncio.to_thread 包装阻塞的 queue.get，避免阻塞事件循环，
        让 uvicorn 能在每个 yield 后立即 flush（否则同步生成器会被缓冲）。
        """
        loop = asyncio.get_event_loop()
        while True:
            event = await loop.run_in_executor(None, q.get)
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用 nginx/代理缓冲
            "Connection": "keep-alive",
        },
    )
