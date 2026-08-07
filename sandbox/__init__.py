"""
沙箱执行器工厂
──────────────
create_executor(config) 根据配置返回 LocalExecutor 或 DockerExecutor。
"""
from __future__ import annotations

from sandbox.executor import BaseExecutor, LocalExecutor


def create_executor(config) -> BaseExecutor:
    """按 config.sandbox_backend 选择执行器后端，docker 不可用时回退 local。"""
    backend = getattr(config, "sandbox_backend", "local")

    if backend == "docker":
        try:
            from sandbox.docker_executor import DockerExecutor
            return DockerExecutor(
                work_dir=config.workspace_dir,
                image=getattr(config, "docker_image", "python:3.11-slim"),
                memory=getattr(config, "docker_memory", "512m"),
                cpus=getattr(config, "docker_cpus", 1.0),
                network=getattr(config, "docker_network", "none"),
            )
        except RuntimeError as e:
            print(f"[sandbox] Docker 不可用，回退本地执行：{e}")

    return LocalExecutor(config.workspace_dir, exec_timeout=getattr(config, "exec_timeout", 30))
