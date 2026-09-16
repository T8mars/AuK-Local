@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0\.."
set "AUK_LOCAL_HOME=%CD%"
set "PYTHONPATH=%CD%\src"
"runtime\python.exe" -m auk_local.cli --root "%CD%" diagnose %*
pause
