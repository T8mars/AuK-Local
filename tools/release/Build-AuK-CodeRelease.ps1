param(
    [string]$PackageRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path,
    [string]$PrivateKeyPath = (Join-Path $env:LOCALAPPDATA 'T8star-Aix\AuK-Local\release-signing-private.xml')
)

$ErrorActionPreference = 'Stop'
$Root = [IO.Path]::GetFullPath($PackageRoot)
$VersionFile = Join-Path $Root 'src\auk_local\version.py'
$VersionText = Get-Content -LiteralPath $VersionFile -Raw -Encoding UTF8
if ($VersionText -notmatch 'VERSION\s*=\s*"([0-9]+(?:\.[0-9]+){1,3})"') {
    throw 'Cannot read AuK Local version.'
}
$Version = $Matches[1]
$Dist = Join-Path $Root 'dist'
New-Item -ItemType Directory -Force -Path $Dist | Out-Null
$ArchiveName = "AuK-Local-v$Version-code-only.zip"
$ArchivePath = Join-Path $Dist $ArchiveName
$ManifestPath = Join-Path $Dist 'auk-local-release.json'
$ChecksumsPath = Join-Path $Dist 'SHA256SUMS.txt'

$rootNames = @(
    'AuK-Local.exe', 'BUILD-INFO.json', 'LICENSE', 'README.md', 'README_zh.md',
    'requirements-windows-cu128.lock', 'pyproject.toml', '使用说明.md',
    '启动AuK.cmd', '启动AuK服务.cmd', '环境诊断.cmd'
)
$files = New-Object System.Collections.Generic.List[IO.FileInfo]
foreach ($name in $rootNames) {
    $path = Join-Path $Root $name
    if (Test-Path -LiteralPath $path -PathType Leaf) { $files.Add((Get-Item -LiteralPath $path)) }
}
foreach ($directory in @('src', 'scripts', 'docs')) {
    Get-ChildItem -LiteralPath (Join-Path $Root $directory) -Recurse -File | Where-Object {
        $_.Extension -notin @('.pyc', '.pyo') -and
            $_.FullName -notmatch '[\\/]__pycache__[\\/]' -and
            $_.FullName -notmatch '[\\/][^\\/]+\.egg-info[\\/]'
    } | ForEach-Object { $files.Add($_) }
}
$icon = Get-Item -LiteralPath (Join-Path $Root 'assets\AuK-Local.ico')
$files.Add($icon)
$files = @($files | Sort-Object { $_.FullName.Substring($Root.Length + 1).Replace('\', '/') } -Unique)

$required = @(
    'AuK-Local.exe', 'src/auk_local/version.py', 'src/auk_local/update-public-key.xml',
    'scripts/Apply-AuK-Update.ps1'
)
$relativeNames = @($files | ForEach-Object { $_.FullName.Substring($Root.Length + 1).Replace('\', '/') })
foreach ($name in $required) {
    if ($relativeNames -notcontains $name) { throw "Release is missing required file: $name" }
}
foreach ($forbidden in @('models/', 'runtime/', 'ckpts/', 'data/', 'outputs/', 'logs/', '.secrets/')) {
    if ($relativeNames | Where-Object { $_.StartsWith($forbidden, [StringComparison]::OrdinalIgnoreCase) }) {
        throw "Release includes forbidden content: $forbidden"
    }
}

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
Remove-Item -LiteralPath $ArchivePath -Force -ErrorAction SilentlyContinue
$stream = [IO.File]::Open($ArchivePath, [IO.FileMode]::CreateNew)
try {
    $archive = New-Object IO.Compression.ZipArchive($stream, [IO.Compression.ZipArchiveMode]::Create, $false)
    try {
        foreach ($file in $files) {
            $relative = $file.FullName.Substring($Root.Length + 1).Replace('\', '/')
            [IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive, $file.FullName, $relative, [IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    } finally { $archive.Dispose() }
} finally { $stream.Dispose() }

$fileEntries = @($files | ForEach-Object {
    [ordered]@{
        path = $_.FullName.Substring($Root.Length + 1).Replace('\', '/')
        size = $_.Length
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
})
$packageHash = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
$payload = [ordered]@{
    schema_version = 1
    repository = 'T8mars/AuK-Local'
    channel = 'stable'
    version = $Version
    published_at = [DateTimeOffset]::UtcNow.ToString('o')
    notes = 'Makes waveform scissors, explicit trimming, clearing, and replacing audio update the actual submitted clip; prevents over-limit automatic duration estimates from breaking Gradio components.'
    compatibility = [ordered]@{ service_protocol = '1.0' }
    package = [ordered]@{
        name = $ArchiveName
        size = (Get-Item -LiteralPath $ArchivePath).Length
        sha256 = $packageHash
    }
    files = $fileEntries
}
$payloadJson = $payload | ConvertTo-Json -Depth 8 -Compress
$payloadBytes = [Text.Encoding]::UTF8.GetBytes($payloadJson)
if (-not (Test-Path -LiteralPath $PrivateKeyPath -PathType Leaf)) {
    throw 'Update signing private key is missing; refusing to create an unsigned manifest.'
}
$rsa = New-Object Security.Cryptography.RSACryptoServiceProvider
try {
    $rsa.FromXmlString((Get-Content -LiteralPath $PrivateKeyPath -Raw -Encoding UTF8))
    $signature = $rsa.SignData($payloadBytes, [Security.Cryptography.CryptoConfig]::MapNameToOID('SHA256'))
} finally { $rsa.Dispose() }
$envelope = [ordered]@{
    schema_version = 1
    algorithm = 'RSA-SHA256'
    signed_payload = [Convert]::ToBase64String($payloadBytes)
    signature = [Convert]::ToBase64String($signature)
}
[IO.File]::WriteAllText(
    $ManifestPath,
    ($envelope | ConvertTo-Json -Depth 4),
    [Text.UTF8Encoding]::new($false)
)
$verificationPayload = Join-Path $Dist '.auk-release-payload.tmp'
try {
    [IO.File]::WriteAllBytes($verificationPayload, $payloadBytes)
    & (Join-Path $Root 'AuK-Local.exe') --verify-update-signature $verificationPayload ([Convert]::ToBase64String($signature))
    if ($LASTEXITCODE -ne 0) { throw 'The built manifest signature was rejected by AuK-Local.exe.' }
} finally {
    Remove-Item -LiteralPath $verificationPayload -Force -ErrorAction SilentlyContinue
}
$manifestHash = (Get-FileHash -LiteralPath $ManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText(
    $ChecksumsPath,
    "$packageHash  $ArchiveName`n$manifestHash  auk-local-release.json`n",
    [Text.UTF8Encoding]::new($false)
)
Write-Output "Built $ArchivePath"
Write-Output "Built $ManifestPath"
Write-Output "Files: $($files.Count); archive bytes: $((Get-Item -LiteralPath $ArchivePath).Length)"
