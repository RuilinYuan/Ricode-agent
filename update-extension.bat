@echo off
REM 重新打包并安装 VS Code 插件（改了 extension.ts 后运行一次，然后重启 VS Code）
cd /d %~dp0\vscode-extension
call npm.cmd run compile || goto :err
call npx vsce package --allow-missing-repository || goto :err
call code --install-extension autonomous-coding-agent-0.1.0.vsix --force || goto :err
echo.
echo [OK] 插件已更新，请重启 VS Code 生效
pause
exit /b 0
:err
echo [FAIL] 更新失败，请检查上方报错
pause
exit /b 1
