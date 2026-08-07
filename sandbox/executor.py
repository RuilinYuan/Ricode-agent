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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


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
    def run_python(self, code: str, timeout: int = 30) -> ExecutionResult:
        """执行 Python 代码片段，返回 stdout/stderr/exit_code。"""

    @abstractmethod
    def run_command(self, cmd: str, timeout: int = 30) -> ExecutionResult:
        """执行 shell 命令。"""

    @abstractmethod
    def install_package(self, package: str) -> ExecutionResult:
        """pip install <package>。"""


class LocalExecutor(BaseExecutor):
    """
    使用 subprocess 在本机执行代码。
    代码写入临时文件后执行，避免 shell 注入。
    """

    def __init__(self, work_dir: str = ".agent_workspace", exec_timeout: int = 30) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        self.exec_timeout = exec_timeout

    def run_python(self, code: str, timeout: int = 30) -> ExecutionResult:
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
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=str(self.work_dir),
            )
            return ExecutionResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr=f"执行超时（{timeout}s）",
                exit_code=1,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def run_command(self, cmd: str, timeout: int = 30) -> ExecutionResult:
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=str(self.work_dir),
            )
            return ExecutionResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr=f"命令超时（{timeout}s）",
                exit_code=1,
            )

    def install_package(self, package: str) -> ExecutionResult:
        return self.run_command(
            f"{sys.executable} -m pip install {package} -q",
            timeout=120,
        )
