"""
沙箱执行器
──────────
BaseExecutor   抽象接口，后期换 Docker 只需实现此类
LocalExecutor  subprocess 本地执行（当前实现）
ExecutionResult 执行结果数据类
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# 输出流回调：每收到一行输出就调用 (line_text)
OutputCallback = Optional[Callable[[str], None]]


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    def __str__(self) -> str:
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[stderr]\n{self.stderr}")
        if not parts:
            return "(无输出)"
        return "\n".join(parts)


class BaseExecutor(ABC):
    """可插拔执行器接口。替换为 Docker 只需实现这三个方法。"""

    @abstractmethod
    def run_python(self, code: str, timeout: int = 30,
                   on_output: OutputCallback = None) -> ExecutionResult:
        """执行 Python 代码片段，返回 stdout/stderr/exit_code。
        传入 on_output 时逐行流式回调。"""

    @abstractmethod
    def run_command(self, cmd: str, timeout: int = 30,
                    on_output: OutputCallback = None) -> ExecutionResult:
        """执行 shell 命令。传入 on_output 时逐行流式回调。"""

    @abstractmethod
    def install_package(self, package: str) -> ExecutionResult:
        """pip install <package>。"""


def _run_streaming(
    args: list[str] | str,
    *,
    cwd: str,
    timeout: int,
    shell: bool,
    on_output: OutputCallback,
) -> ExecutionResult:
    """
    以流式方式运行子进程：合并 stdout/stderr，逐行读取，
    每行通过 on_output 回调实时推送，同时累积完整输出。
    支持超时（超时后杀进程并返回已收集的部分输出）。
    """
    # 注入 PYTHONUNBUFFERED=1 避免子进程的 Python 块缓冲
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        args,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # stderr 合并到 stdout，保证行序正确
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,                  # 行缓冲
        cwd=cwd,
        env=env,
    )

    lines: list[str] = []
    start = time.time()
    timed_out = False

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            if on_output:
                on_output(line.rstrip("\n"))
            # 超时检查（读循环内）
            if time.time() - start > timeout:
                proc.kill()
                timed_out = True
                break
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    finally:
        if proc.stdout:
            proc.stdout.close()

    output = "".join(lines)
    if timed_out:
        msg = f"\n[命令超时（{timeout}s），已终止]"
        if on_output:
            on_output(msg.strip())
        return ExecutionResult(stdout=output, stderr=msg.strip(), exit_code=1)

    return ExecutionResult(stdout=output, stderr="", exit_code=proc.returncode or 0)


class LocalExecutor(BaseExecutor):
    """
    使用 subprocess 在本机执行代码。
    代码写入临时文件后执行，避免 shell 注入。
    """

    def __init__(self, work_dir: str = ".agent_workspace", exec_timeout: int = 30) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        self.exec_timeout = exec_timeout

    def run_python(self, code: str, timeout: int = 30,
                   on_output: OutputCallback = None) -> ExecutionResult:
        # 写临时 .py 文件，确保多行代码、中文路径均正常
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            dir=self.work_dir,
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            # -u 强制无缓冲输出，保证流式实时性
            return _run_streaming(
                [sys.executable, "-u", tmp_path],
                cwd=str(self.work_dir),
                timeout=timeout,
                shell=False,
                on_output=on_output,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def run_command(self, cmd: str, timeout: int = 30,
                    on_output: OutputCallback = None) -> ExecutionResult:
        return _run_streaming(
            cmd,
            cwd=str(self.work_dir),
            timeout=timeout,
            shell=True,
            on_output=on_output,
        )

    def install_package(self, package: str) -> ExecutionResult:
        return self.run_command(
            f"{sys.executable} -m pip install {package} -q",
            timeout=120,
        )
