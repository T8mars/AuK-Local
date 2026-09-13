@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0\.."
set "AUK_LOCAL_HOME=%CD%"
set "PYTHONPATH=%CD%\src"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "PATH=%CD%\runtime\ffmpeg;%PATH%"
if not exist "runtime\python.exe" (
  echo ERROR: Missing runtime\python.exe
  pause
  exit /b 1
)
powershell.exe -NoProfile -Command "$token='data\session-token'; if (-not (Test-Path -LiteralPath $token)) { exit 1 }; $sha=[Security.Cryptography.SHA256]::Create(); try { $expected=([BitConverter]::ToString($sha.ComputeHash([IO.File]::ReadAllBytes($token)))).Replace('-','').ToLowerInvariant() } finally { $sha.Dispose() }; try { $r=Invoke-RestMethod -Uri 'http://127.0.0.1:7860/api/v1/health' -TimeoutSec 2; if ($r.status -eq 'ok' -and $r.ui_enabled -eq $true -and ([string]$r.protocol_version).Split('.')[0] -eq '1' -and $r.instance_id -eq $expected) { exit 0 } } catch {}; exit 1"
if not errorlevel 1 (
  start "" "http://127.0.0.1:7860"
  exit /b 0
)
start "AuK browser helper" /b powershell.exe -NoProfile -WindowStyle Hidden -Command "$deadline=(Get-Date).AddMinutes(5); while ((Get-Date) -lt $deadline) { try { $r=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:7860/api/v1/health' -TimeoutSec 2; if ($r.StatusCode -eq 200) { Start-Process 'http://127.0.0.1:7860'; break } } catch {}; Start-Sleep -Seconds 1 }"
"runtime\python.exe" -m auk_local.cli --root "%CD%" serve --host 127.0.0.1 --port 7860
if errorlevel 1 pause
