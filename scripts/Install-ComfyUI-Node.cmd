@echo off
setlocal
if "%~1"=="" (
  echo Usage: Install-ComfyUI-Node.cmd "D:\path\to\ComfyUI"
  pause
  exit /b 2
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-ComfyUI-Node.ps1" -ComfyUIRoot "%~1"
if errorlevel 1 pause
