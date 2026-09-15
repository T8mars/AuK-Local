param([switch]$RequireUi)

$ErrorActionPreference = 'Stop'
$packageRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$tokenPath = Join-Path $packageRoot 'data\session-token'
try {
    if (-not (Test-Path -LiteralPath $tokenPath -PathType Leaf)) {
        exit 1
    }
    $token = [IO.File]::ReadAllText($tokenPath).Trim()
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $expected = ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($token)))).Replace('-', '').ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
    $response = Invoke-RestMethod -Uri 'http://127.0.0.1:7860/api/v1/health' -TimeoutSec 2
    $uiMatches = -not $RequireUi -or $response.ui_enabled -eq $true
    if (
        $response.status -in @('ok', 'degraded') -and
        $uiMatches -and
        ([string]$response.protocol_version).Split('.')[0] -eq '1' -and
        $response.instance_id -eq $expected
    ) {
        exit 0
    }
} catch {
    # A missing service, incomplete startup, or foreign process is not this package.
}
exit 1
