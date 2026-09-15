param([string]$PackageRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path)

$ErrorActionPreference = 'Stop'
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path -LiteralPath $compiler)) {
    $compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe'
}
if (-not (Test-Path -LiteralPath $compiler)) {
    throw 'Microsoft C# compiler was not found.'
}

$source = Join-Path $PSScriptRoot 'AuKLauncher.cs'
$icon = Join-Path $PackageRoot 'assets\AuK-Local.ico'
$output = Join-Path $PackageRoot 'AuK-Local.exe'
if (-not (Test-Path -LiteralPath $icon)) {
    throw "Launcher icon was not found: $icon"
}

& $compiler /nologo /target:exe /platform:anycpu /optimize+ "/win32icon:$icon" "/out:$output" $source
if ($LASTEXITCODE -ne 0) {
    throw "Launcher compilation failed with exit code $LASTEXITCODE"
}
Write-Output "Built $output"
