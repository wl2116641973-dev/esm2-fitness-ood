# Run a local project command with caches and temporary files on D:.
param(
    [Parameter(Mandatory, Position=0)][string]$Command,
    [Parameter(Position=1, ValueFromRemainingArguments)][string[]]$CommandArgs
)

$ErrorActionPreference = 'Stop'
$cache = 'D:\Antigravity\cache'
$env:UV_CACHE_DIR = Join-Path $cache 'uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $cache 'uv-python'
$env:PIP_CACHE_DIR = Join-Path $cache 'pip'
$env:HF_HOME = Join-Path $cache 'huggingface'
$env:MPLCONFIGDIR = Join-Path $cache 'matplotlib'
$env:TEMP = Join-Path $cache 'tmp'
$env:TMP = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = '1'
[void][IO.Directory]::CreateDirectory($env:TEMP)

Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $global:LASTEXITCODE = 0
    & $Command @CommandArgs
    $result = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $result
