@echo off
REM 启动 Coding Agent 后端（VS Code 插件依赖此服务）
REM --reload：Python 代码改动后自动重载，无需手动重启
cd /d %~dp0
python -m uvicorn server:app --port 8765 --reload
pause
