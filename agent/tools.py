"""
工具集
──────
TOOLS          Anthropic tool_use 格式的工具定义列表
execute_tool   统一分发：根据工具名调用 executor 对应方法
"""
from __future__ import annotations

from pathlib import Path
from sandbox.executor import BaseExecutor, ExecutionResult


# ── Anthropic tool_use schema ─────────────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "write_file",
        "description": "将内容写入指定路径的文件。路径相对于工作目录。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径，如 main.py 或 src/utils.py"},
                "content": {"type": "string", "description": "写入的完整文件内容"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "read_file",
        "description": "读取指定路径文件的内容。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要读取的文件路径"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "execute_python",
        "description": "执行一段 Python 代码，返回 stdout 和 stderr。适合验证代码逻辑、运行脚本。",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要执行的 Python 代码"},
            },
            "required": ["code"],
        },
    },
    {
        "name": "run_command",
        "description": "执行 shell 命令，返回输出。适合 pip install、文件操作等。",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的 shell 命令"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "run_tests",
        "description": "对指定路径运行 pytest 测试，返回通过/失败详情。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "测试文件或目录路径，默认为 '.' 运行所有测试",
                    "default": ".",
                },
            },
            "required": [],
        },
    },
    {
        "name": "search_code",
        "description": "在整个工作目录中做语义检索，返回与查询最相关的代码片段（含路径和行号）。适合大仓库中定位相关实现，避免逐文件通读。需要先由系统建立索引。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "自然语言或代码描述，如 'CSV 编码检测逻辑'"},
                "top_k": {"type": "integer", "description": "返回片段数量，默认 5", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "task_complete",
        "description": "任务已完成时调用此工具，传入最终总结。",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "任务完成的简短总结，说明生成了哪些文件、测试结果如何",
                },
            },
            "required": ["summary"],
        },
    },
]


# ── 工具分发 ───────────────────────────────────────────────────────────────────

def execute_tool(
    tool_name: str,
    tool_input: dict,
    executor: BaseExecutor,
) -> ExecutionResult:
    """
    根据工具名将调用路由到对应的 executor 方法。
    task_complete 由主循环在调用本函数之前拦截，不会到达这里。
    """
    if tool_name == "write_file":
        return _write_file(tool_input, executor)

    if tool_name == "read_file":
        return _read_file(tool_input, executor)

    if tool_name == "execute_python":
        return executor.run_python(
            tool_input["code"],
            timeout=getattr(executor, "exec_timeout", 30),
        )

    if tool_name == "run_command":
        return executor.run_command(tool_input["command"])

    if tool_name == "run_tests":
        test_path = tool_input.get("path", ".")
        return executor.run_command(f"python -m pytest {test_path} -v --tb=short 2>&1")

    if tool_name == "search_code":
        return _search_code(tool_input, executor)

    # 未知工具
    from sandbox.executor import ExecutionResult as ER
    return ER(stdout="", stderr=f"未知工具：{tool_name}", exit_code=1)


def _write_file(tool_input: dict, executor: BaseExecutor) -> ExecutionResult:
    path_str = tool_input.get("path") or tool_input.get("file") or tool_input.get("filename") or ""
    if not path_str:
        return ExecutionResult(stdout="", stderr=f"write_file 缺少 path 参数，收到：{list(tool_input.keys())}", exit_code=1)
    content = tool_input.get("content") or tool_input.get("code") or ""
    path = Path(executor.work_dir) / path_str
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return ExecutionResult(
        stdout=f"已写入 {path_str}（{len(content)} 字符）",
        stderr="",
        exit_code=0,
    )


def _search_code(tool_input: dict, executor: BaseExecutor) -> ExecutionResult:
    """RAG 语义检索。索引由 AgentLoop 启动时挂在 executor.code_index 上。"""
    index = getattr(executor, "code_index", None)
    if index is None:
        return ExecutionResult(
            stdout="", exit_code=1,
            stderr="代码索引未启用（需配置 EMBEDDING_BASE_URL / EMBEDDING_API_KEY 且 rag_enabled=True）",
        )
    query = tool_input.get("query", "").strip()
    if not query:
        return ExecutionResult(stdout="", stderr="search_code 缺少 query 参数", exit_code=1)
    hits = index.search(query, top_k=int(tool_input.get("top_k", 5)))
    if not hits:
        return ExecutionResult(stdout="（无匹配结果）", stderr="", exit_code=0)
    parts = [
        f"### {h['path']}:{h['lines']} (score={h['score']})\n{h['text']}"
        for h in hits
    ]
    return ExecutionResult(stdout="\n\n".join(parts), stderr="", exit_code=0)


def _read_file(tool_input: dict, executor: BaseExecutor) -> ExecutionResult:
    path_str = tool_input.get("path") or tool_input.get("file") or ""
    if not path_str:
        return ExecutionResult(stdout="", stderr=f"read_file 缺少 path 参数，收到：{list(tool_input.keys())}", exit_code=1)
    path = Path(executor.work_dir) / path_str
    if not path.exists():
        return ExecutionResult(stdout="", stderr=f"文件不存在：{path_str}", exit_code=1)
    content = path.read_text(encoding="utf-8")
    return ExecutionResult(stdout=content, stderr="", exit_code=0)
