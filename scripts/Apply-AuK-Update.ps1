param(
    [Parameter(Mandatory = $true)][string]$PackageRoot,
    [int]$ParentProcessId = 0,
    [switch]$NoRestart,
    [switch]$FailAfterFirstCopy
)

$ErrorActionPreference = 'Stop'
$Root = [IO.Path]::GetFullPath($PackageRoot)
$UpdatesRoot = [IO.Path]::GetFullPath((Join-Path $Root 'data\updates'))
$PendingPath = Join-Path $UpdatesRoot 'pending-update.json'
$LogPath = Join-Path $UpdatesRoot 'update.log'

function Write-UpdateLog([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
    Write-Host $Message
}

function Get-SHA256([string]$Path) {
    $algorithm = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try {
        return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-', '').ToLowerInvariant()
    } finally {
        $stream.Dispose()
        $algorithm.Dispose()
    }
}

function Resolve-Under([string]$Base, [string]$Relative) {
    if ([string]::IsNullOrWhiteSpace($Relative) -or $Relative.Contains('..') -or $Relative.Contains(':')) {
        throw "Invalid relative update path: $Relative"
    }
    $normalized = $Relative.Replace('/', '\')
    $candidate = [IO.Path]::GetFullPath((Join-Path $Base $normalized))
    $prefix = $Base.TrimEnd('\') + '\'
    if (-not $candidate.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Update path escapes its allowed root: $Relative"
    }
    return $candidate
}

function Test-AllowedPath([string]$Relative) {
    $clean = $Relative.Replace('\', '/')
    $parts = $clean.Split('/')
    $protected = @('models', 'runtime', 'ckpts', 'data', 'outputs', 'logs', '.git', '.secrets')
    if ($protected -contains $parts[0]) { return $false }
    if ($parts.Count -eq 1) {
        return @(
            'AuK-Local.exe', 'BUILD-INFO.json', 'LICENSE', 'README.md', 'README_zh.md',
            'requirements-windows-cu128.lock', 'pyproject.toml', '使用说明.md',
            '启动AuK.cmd', '启动AuK服务.cmd', '环境诊断.cmd'
        ) -contains $clean
    }
    if (@('src', 'scripts', 'docs', 'assets') -notcontains $parts[0]) { return $false }
    if ($parts[0] -eq 'assets' -and $clean -ne 'assets/AuK-Local.ico') { return $false }
    return $true
}

try {
    New-Item -ItemType Directory -Force -Path $UpdatesRoot | Out-Null
    if ($ParentProcessId -gt 0) {
        try { Wait-Process -Id $ParentProcessId -Timeout 45 -ErrorAction Stop } catch {
            if (Get-Process -Id $ParentProcessId -ErrorAction SilentlyContinue) {
                throw "Launcher did not exit before update timeout."
            }
        }
    }
    if (-not (Test-Path -LiteralPath $PendingPath -PathType Leaf)) {
        throw 'pending-update.json is missing.'
    }
    $Pending = Get-Content -LiteralPath $PendingPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([int]$Pending.schema_version -ne 1 -or -not $Pending.files -or -not $Pending.to_version) {
        throw 'Pending update metadata is invalid.'
    }
    $Staging = Resolve-Under $Root ([string]$Pending.staging_directory)
    $StagingPrefix = [IO.Path]::GetFullPath((Join-Path $UpdatesRoot 'staging')).TrimEnd('\') + '\'
    if (-not $Staging.StartsWith($StagingPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Staging directory is outside data\updates\staging.'
    }
    if (-not (Test-Path -LiteralPath $Staging -PathType Container)) {
        throw 'Staged update directory is missing.'
    }

    $seen = @{}
    foreach ($File in $Pending.files) {
        $relative = [string]$File.path
        if (-not (Test-AllowedPath $relative)) { throw "Protected or unknown path in update: $relative" }
        $key = $relative.ToLowerInvariant()
        if ($seen.ContainsKey($key)) { throw "Duplicate path in update: $relative" }
        $seen[$key] = $true
        $source = Resolve-Under $Staging $relative
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Staged file is missing: $relative" }
        $actualSize = (Get-Item -LiteralPath $source).Length
        $actualHash = Get-SHA256 $source
        if ($actualSize -ne [long]$File.size -or $actualHash -ne [string]$File.sha256) {
            throw "Staged file failed verification: $relative"
        }
    }

    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $Backup = Join-Path $UpdatesRoot "backups\$stamp-$($Pending.from_version)"
    New-Item -ItemType Directory -Force -Path $Backup | Out-Null
    $created = New-Object System.Collections.Generic.List[string]
    $copied = New-Object System.Collections.Generic.List[string]
    $restartAttempted = $false
    $newProcess = $null
    try {
        foreach ($File in $Pending.files) {
            $relative = [string]$File.path
            $source = Resolve-Under $Staging $relative
            $target = Resolve-Under $Root $relative
            if (Test-Path -LiteralPath $target -PathType Leaf) {
                $backupFile = Resolve-Under $Backup $relative
                New-Item -ItemType Directory -Force -Path (Split-Path -Parent $backupFile) | Out-Null
                Copy-Item -LiteralPath $target -Destination $backupFile -Force
            } else {
                $created.Add($relative)
            }
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
            $temporary = "$target.auk-update.tmp"
            Copy-Item -LiteralPath $source -Destination $temporary -Force
            Move-Item -LiteralPath $temporary -Destination $target -Force
            $copied.Add($relative)
            if ($FailAfterFirstCopy -and $copied.Count -eq 1) { throw 'Forced update failure for rollback test.' }
        }
        foreach ($File in $Pending.files) {
            $target = Resolve-Under $Root ([string]$File.path)
            $hash = Get-SHA256 $target
            if ($hash -ne [string]$File.sha256) { throw "Installed file failed verification: $($File.path)" }
        }

        if (-not $NoRestart) {
            $Launcher = Join-Path $Root 'AuK-Local.exe'
            if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) { throw 'Updated launcher is missing.' }
            $restartAttempted = $true
            $newProcess = Start-Process -FilePath $Launcher -WorkingDirectory $Root -PassThru
            $healthy = $false
            $deadline = [DateTime]::UtcNow.AddSeconds(75)
            while ([DateTime]::UtcNow -lt $deadline) {
                Start-Sleep -Milliseconds 500
                if ($newProcess.HasExited) { break }
                try {
                    $client = New-Object Net.WebClient
                    try {
                        $health = $client.DownloadString('http://127.0.0.1:7860/api/v1/health') | ConvertFrom-Json
                    } finally { $client.Dispose() }
                    if ([string]$health.version -eq [string]$Pending.to_version -and [string]$health.status -in @('ok', 'degraded')) {
                        $healthy = $true
                        break
                    }
                } catch { }
            }
            if (-not $healthy) { throw 'Updated AuK Local did not pass its startup health check within 75 seconds.' }
        }
        Remove-Item -LiteralPath $PendingPath -Force
        Write-UpdateLog "Updated AuK Local $($Pending.from_version) -> $($Pending.to_version); $($copied.Count) files and restart verified."
    } catch {
        if ($restartAttempted -and $newProcess -and -not $newProcess.HasExited) {
            & taskkill.exe /PID $newProcess.Id /T /F 2>$null | Out-Null
            Start-Sleep -Milliseconds 500
        }
        foreach ($relative in $copied) {
            $target = Resolve-Under $Root $relative
            $backupFile = Resolve-Under $Backup $relative
            if (Test-Path -LiteralPath $backupFile -PathType Leaf) {
                Copy-Item -LiteralPath $backupFile -Destination $target -Force
            } elseif ($created -contains $relative) {
                Remove-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue
            }
        }
        Write-UpdateLog "Update failed and rollback ran: $($_.Exception.Message)"
        if ($restartAttempted -and -not $NoRestart) {
            $FallbackLauncher = Join-Path $Root 'AuK-Local.exe'
            if (Test-Path -LiteralPath $FallbackLauncher -PathType Leaf) {
                Start-Process -FilePath $FallbackLauncher -WorkingDirectory $Root
            }
        }
        throw
    }
    exit 0
} catch {
    try { Write-UpdateLog "Update process failed: $($_.Exception.Message)" } catch { }
    Write-Error $_
    exit 1
}
