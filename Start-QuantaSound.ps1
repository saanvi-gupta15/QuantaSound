param([int]$Port = 7860)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonCandidates = @(
    (Join-Path $PSScriptRoot '.venv\Scripts\python.exe'),
    (Join-Path $PSScriptRoot '..\..\work\.venv\Scripts\python.exe')
)
$quantaPython = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $quantaPython) {
    $quantaPython = (Get-Command python -ErrorAction Stop).Source
}
& $quantaPython -c 'import gradio, torch, transformers, torchlibrosa'
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Install requirements-ai.txt in this Python environment before starting.'
    Write-Host "Python selected: $quantaPython"
    exit 1
}
Write-Host "Open http://127.0.0.1:$Port after startup. Stop with Ctrl+C."
& $quantaPython -m demo4.app --port $Port
