param(
    [Parameter(Mandatory = $true)]
    [string]$Directory,

    [Parameter(Mandatory = $true)]
    [string[]]$Query,

    [double]$MinPercent = 75,

    [int]$BatchSize = 64,

    [string]$Model = "ViT-B/32"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot

$Directory = (
    Resolve-Path $Directory
).Path

$OutputDirectory = Join-Path `
    $ProjectRoot `
    "output"

$ModelsDirectory = Join-Path `
    $ProjectRoot `
    "models"

New-Item `
    -ItemType Directory `
    -Force `
    $OutputDirectory |
    Out-Null

New-Item `
    -ItemType Directory `
    -Force `
    $ModelsDirectory |
    Out-Null

$Timestamp = Get-Date -Format `
    "yyyyMMdd_HHmmss"

$ReportName = `
    "report_$Timestamp.csv"

$QueryArguments = @()

foreach ($q in $Query) {
    $QueryArguments += "--query"
    $QueryArguments += $q
}

$DockerArguments = @(
    "run",
    "--rm",
    "--gpus",
    "all",

    "--mount",
    "type=bind,source=$Directory,target=/evidence,readonly",

    "--mount",
    "type=bind,source=$OutputDirectory,target=/output",

    "--mount",
    "type=bind,source=$ModelsDirectory,target=/root/.cache/clip",

    "forensic-media-search:dev",

    "--directory",
    "/evidence",

    "--display-root",
    $Directory,

    "--min-percent",
    $MinPercent,

    "--batch-size",
    $BatchSize,

    "--model",
    $Model,

    "--output",
    "/output/$ReportName"
)

$DockerArguments += $QueryArguments

& docker @DockerArguments

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Report generated:"
Write-Host (
    Join-Path `
        $OutputDirectory `
        $ReportName
)