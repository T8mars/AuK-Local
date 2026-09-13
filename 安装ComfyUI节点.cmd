@echo off
setlocal
chcp 65001 >nul
if "%~1"=="" (
  set /p "COMFY_ROOT=请输入 ComfyUI 根目录（其中应包含 custom_nodes）: "
) else (
  set "COMFY_ROOT=%~1"
)
if not defined COMFY_ROOT exit /b 2
call "%~dp0scripts\Install-ComfyUI-Node.cmd" "%COMFY_ROOT%"
