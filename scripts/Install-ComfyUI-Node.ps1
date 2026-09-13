param([Parameter(Mandatory=$true)][string]$ComfyUIRoot)
$ErrorActionPreference = "Stop"
$packageRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$source = Join-Path $packageRoot "comfyui\ComfyUI-AuK-Local"
$customNodes = Join-Path $ComfyUIRoot "custom_nodes"
if (-not (Test-Path -LiteralPath $customNodes -PathType Container)) {
  throw "ComfyUI custom_nodes directory not found: $customNodes"
}
$target = Join-Path $customNodes "ComfyUI-AuK-Local"
New-Item -ItemType Directory -Path $target -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $source "__init__.py") -Destination $target -Force
Copy-Item -LiteralPath (Join-Path $source "nodes.py") -Destination $target -Force
Copy-Item -LiteralPath (Join-Path $source "README.md") -Destination $target -Force
$exampleWorkflows = Join-Path $target "example_workflows"
New-Item -ItemType Directory -Path $exampleWorkflows -Force | Out-Null
Get-ChildItem -LiteralPath (Join-Path $packageRoot "comfyui\workflows") -Filter "AuK-*.json" |
  Copy-Item -Destination $exampleWorkflows -Force
$tokenSource = Join-Path $packageRoot "data\session-token"
if (-not (Test-Path -LiteralPath $tokenSource -PathType Leaf)) {
  throw "AuK Local token not found. Start AuK once before installing the ComfyUI node."
}
Copy-Item -LiteralPath $tokenSource -Destination (Join-Path $target "session-token") -Force
$config = @{ token_file = "session-token" } | ConvertTo-Json
[System.IO.File]::WriteAllText((Join-Path $target "auk-local-config.json"), $config, [System.Text.UTF8Encoding]::new($false))
Write-Host "Installed AuK Local nodes to $target"
