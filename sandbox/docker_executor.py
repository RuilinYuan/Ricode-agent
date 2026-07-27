"""
Docker 沙箱执行器
─────────────────
在一次性 Docker 容器中执行代码，实现：
  - 网络隔离   --network none（可配置）
  - 资源限制   --memory / --cpus / --pids-limit
  - 文件系统   --read-only + tmpfs /tmp，仅挂载工作目录
  - 自动清理   --rm，容器退出即销毁

依赖宿主机的 docker CLI，无需 docker SDK。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from sandbox.executor import BaseExecutor, ExecutionResult


class DockerExecutor(BaseExecutor):
    """基于 Docker 的沙箱执行器（实现 BaseExecutor 三方法）。"""

    def __init__(
        self,
        work_dir: str = ".agent_workspace",
        image: str = "python:3.11-slim",
        memory: str = "512m",
        cpus: float = 1.0,
        network: str = "none",
        pids_limit: int = 128,
    ) -> None:
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(exist_ok=True)
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.network = network
        self.pids_limit = pids_limit

        if shutil.which("docker") is None:
            raise RuntimeError("未找到 docker CLI，请先安装 Docker 或改用 LocalExecutor")

    # ── 容器启动参数 ─────────────────────────────────────────────────
    def _base_cmd(self, timeout: int) -> list[str]:
        return [
            "docker", "run", "--rm",
            "--network", self.network,                # 网络隔离
            "--memory", self.memory,                  # 内存上限
            "--cpus", str(self.cpus),                 # CPU 上限
            "--pids-limit", str(self.pids_limit),     # 防 fork 炸弹
            "--read-only",                            # 根文件系统只读
            "--tmpfs", "/tmp:rw,size=64m",            # 可写临时目录（限额）
            "-v", f"{self.work_dir}:/workspace",      # 仅挂载工作目录
            "-w", "/workspace",
            "--stop-timeout", "2",
            self.image,
        ]

    def _run(self, argv: list[str], timeout: int) -> ExecutionResult:
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout + 10,
            )
            return ExecutionResult(
                stdout=proc.stdout, stderr=proc.stderr, exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="", stderr=f"执行超时（{timeout}s，容器已销毁）", exit_code=1,
            )

    # ── BaseExecutor 接口 ─────────────────────────────────────────────
    def run_python(self, code: str, timeout: int = 30) -> ExecutionResult:
        # 代码通过 stdin 传入容器，避免写临时文件
        argv = self._base_cmd(timeout) + ["python", "-"]
        try:
            proc = subprocess.run(
                argv, input=code, capture_output=True, text=True, timeout=timeout + 10,
            )
            return ExecutionResult(
                stdout=proc.stdout, stderr=proc.stderr, exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="", stderr=f"执行超时（{timeout}s，容器已销毁）", exit_code=1,
            )

    def run_command(self, cmd: str, timeout: int = 30) -> ExecutionResult:
        return self._run(self._base_cmd(timeout) + ["sh", "-c", cmd], timeout)

    def install_package(self, package: str) -> ExecutionResult:
        # 安装包需要临时放开网络
        argv = self._base_cmd(120)
        argv[argv.index("--network") + 1] = "bridge"
        return self._run(
            argv + ["sh", "-c", f"pip install {package} -q && echo INSTALLED"],
            timeout=120,
        )
