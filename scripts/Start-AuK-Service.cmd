@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0\.."
set "AUK_LOCAL_HOME=%CD%"
set "PYTHONPATH=%CD%\src"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
powershell.exe -NoProfile -Command "try { $r=Invoke-RestMethod -Uri 'http://127.0.0.1:7860/api/v1/health' -TimeoutSec 2; if ($r.status -eq 'ok') { exit 0 } } catch {}; exit 1"
if not errorlevel 1 (
  echo AuK 本机服务已在 http://127.0.0.1:7860 运行。
  exit /b 0
)
"runtime\python.exe" -m auk_local.cli --root "%CD%" serve --host 127.0.0.1 --port 7860
if errorlevel 1 pause
