param(
    [Parameter(Mandatory = $true)]
    [string]$Directory,

    [Parameter(Mandatory = $true)]
    [string[]]$Query,

    [int]$TopK = 5000,
    [int]$BatchSize = 64,
    [string]$SigLIPModel = "google/siglip2-base-patch16-224",
    [string]$CLIPModel = "ViT-B/32",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [Nullable[int]]$MaxImages = $null
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$Directory = (Resolve-Path $Directory).Path
$OutputDirectory = Join-Path $ProjectRoot "output"
$ModelsDirectory = Join-Path $ProjectRoot "models"
$HuggingFaceDirectory = Join-Path $ModelsDirectory "huggingface"

New-Item -ItemType Directory -Force $OutputDirectory, $ModelsDirectory, $HuggingFaceDirectory |
    Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ReportName = "report_$Timestamp.csv"
$DockerArguments = @(
    "run", "--rm", "--gpus", "all",
    "--mount", "type=bind,source=$Directory,target=/evidence,readonly",
    "--mount", "type=bind,source=$OutputDirectory,target=/output",
    "--mount", "type=bind,source=$ModelsDirectory,target=/root/.cache/clip",
    "--mount", "type=bind,source=$HuggingFaceDirectory,target=/root/.cache/huggingface",
    "forensic-media-search:dev",
    "--directory", "/evidence",
    "--display-root", $Directory,
    "--top-k", $TopK,
    "--batch-size", $BatchSize,
    "--siglip-model", $SigLIPModel,
    "--clip-model", $CLIPModel,
    "--device", $Device,
    "--output", "/output/$ReportName"
)

if ($null -ne $MaxImages) {
    $DockerArguments += @("--max-images", $MaxImages)
}
foreach ($item in $Query) {
    $DockerArguments += @("--query", $item)
}

& docker @DockerArguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Report generated:"
Write-Host (Join-Path $OutputDirectory $ReportName)
