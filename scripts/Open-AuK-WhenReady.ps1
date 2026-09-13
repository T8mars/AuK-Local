param([int]$TimeoutSeconds = 300)
$ErrorActionPreference = 'Stop'
$packageRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$tokenPath = Join-Path $packageRoot 'data\session-token'
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    try {
        $token = [IO.File]::ReadAllText($tokenPath).Trim()
        $sha = [Security.Cryptography.SHA256]::Create()
        try {
            $expected = ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($token)))).Replace('-', '').ToLowerInvariant()
        } finally {
            $sha.Dispose()
        }
        $r = Invoke-RestMethod -Uri 'http://127.0.0.1:7860/api/v1/health' -TimeoutSec 2
        if ($r.status -in @('ok', 'degraded') -and $r.ui_enabled -eq $true -and ([string]$r.protocol_version).Split('.')[0] -eq '1' -and $r.instance_id -eq $expected) {
            Start-Process 'http://127.0.0.1:7860'
            return
        }
    } catch {
        # Startup may still be creating the token or binding the port.
    }
    Start-Sleep -Milliseconds 500
}
