@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0\.."
set "AUK_LOCAL_HOME=%CD%"
set "PYTHONPATH=%CD%\src"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "PATH=%CD%\runtime;%CD%\runtime\Scripts;%CD%\runtime\ffmpeg;%SystemRoot%\system32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SystemRoot%\System32\WindowsPowerShell\v1.0"
if not exist "runtime\python.exe" (
  echo.
  echo [错误] 缺少整合包运行环境：runtime\python.exe
  set "AUK_EXIT_CODE=1"
  goto :keep_open
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\Test-AuK-Running.ps1" -RequireUi
if not errorlevel 1 (
  echo.
  echo [AuK] 检测到本整合包的服务已经运行，正在打开浏览器。
  start "" "http://127.0.0.1:7860"
  set "AUK_EXIT_CODE=0"
  goto :keep_open
)
echo.
echo [AuK] 正在启动本地工作台：http://127.0.0.1:7860
echo [AuK] 请保持此窗口开启。要停止服务，请在此窗口按 Ctrl+C。
echo.
start "AuK browser helper" /b powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "scripts\Open-AuK-WhenReady.ps1"
"runtime\python.exe" -m auk_local.cli --root "%CD%" serve --host 127.0.0.1 --port 7860
set "AUK_EXIT_CODE=%ERRORLEVEL%"
echo.
if "%AUK_EXIT_CODE%"=="42" (
  echo [AuK] 更新包已经校验完成，正在交给独立更新器安装并重启。
  exit /b 42
)
if "%AUK_EXIT_CODE%"=="0" goto :stopped_cleanly
echo [错误] AuK 服务异常退出，退出码：%AUK_EXIT_CODE%
echo [提示] 可双击“环境诊断.cmd”检查运行环境和模型文件。
goto :keep_open

:stopped_cleanly
echo [AuK] 服务已经停止。

:keep_open
echo.
echo 按任意键关闭此窗口...
pause >nul
exit /b %AUK_EXIT_CODE%
