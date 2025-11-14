# Upload dataset to Azure Machine Learning
# This script uploads a Lerobot dataset to Azure ML and registers it as a managed data asset

param(
    [Parameter(Mandatory=$true)]
    [string]$DatasetName,
    
    [Parameter(Mandatory=$true)]
    [string]$DatasetPath,
    
    [Parameter(Mandatory=$false)]
    [string]$DatasetVersion = "1"
)

# Check if required tools are available
Write-Host "Checking prerequisites..." -ForegroundColor Green

# Check if jq is available
try {
    $jqVersion = jq --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "jq not found"
    }
    Write-Host "✓ jq is available: $jqVersion" -ForegroundColor Green
} catch {
    Write-Error "jq is not installed or not in PATH. Please install jq from https://jqlang.org/"
    exit 1
}

# Check if Azure CLI is available
try {
    $az_version = az version 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "az not found"
    }
    Write-Host "✓ Azure CLI is available" -ForegroundColor Green
} catch {
    Write-Error "Azure CLI is not installed or not in PATH. Please install from https://learn.microsoft.com/cli/azure/install-azure-cli"
    exit 1
}

# Check if Azure ML extension is available
try {
    $mlExtension = az extension list --query "[?name=='ml'].version" -o tsv 2>$null
    if ([string]::IsNullOrEmpty($mlExtension)) {
        throw "ml extension not found"
    }
    Write-Host "✓ Azure ML extension is available: $mlExtension" -ForegroundColor Green
} catch {
    Write-Error "Azure ML extension is not installed. Please install with: az extension add -n ml"
    exit 1
}

# Check if user is logged in to Azure
try {
    $account = az account show --query "user.name" -o tsv 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "not logged in"
    }
    Write-Host "✓ Logged in to Azure as: $account" -ForegroundColor Green
} catch {
    Write-Error "Not logged in to Azure. Please run: az login"
    exit 1
}

# Validate that the dataset path exists
if (-not (Test-Path $DatasetPath)) {
    Write-Error "Dataset path '$DatasetPath' does not exist."
    exit 1
}

# Validate that required metadata files exist
$infoJsonPath = Join-Path $DatasetPath "meta\info.json"
$tasksJsonlPath = Join-Path $DatasetPath "meta\tasks.jsonl"

if (-not (Test-Path $infoJsonPath)) {
    Write-Error "Required metadata file not found: $infoJsonPath"
    exit 1
}

if (-not (Test-Path $tasksJsonlPath)) {
    Write-Error "Required metadata file not found: $tasksJsonlPath"
    exit 1
}

# Validate and determine dataset version
Write-Host "Validating dataset version..." -ForegroundColor Green

# Check if dataset with this name already exists and get existing versions
Write-Host "Checking for existing versions of dataset '$DatasetName'..." -ForegroundColor Yellow

try {
    $existingVersions = az ml data list --name $DatasetName --query "[].version" -o tsv 2>$null
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrEmpty($existingVersions)) {
        # Dataset exists, get the latest version
        $versions = $existingVersions -split "`n" | Where-Object { $_ -ne "" } | ForEach-Object { 
            if ($_ -match '^\d+$') { 
                [int]$_ 
            } else { 
                # Handle non-numeric versions by trying to extract numeric part
                if ($_ -match '(\d+)') {
                    [int]$Matches[1]
                } else {
                    0
                }
            }
        } | Sort-Object -Descending
        
        $latestVersion = $versions[0]
        $nextVersion = $latestVersion + 1
        
        Write-Host "Found existing versions. Latest version: $latestVersion" -ForegroundColor Yellow
        Write-Host "Auto-incrementing to version: $nextVersion" -ForegroundColor Green
        $DatasetVersion = $nextVersion.ToString()
    } else {
        Write-Host "No existing versions found. Using provided version: $DatasetVersion" -ForegroundColor Green
    }
} catch {
    Write-Host "Could not check existing versions (dataset may not exist). Using provided version: $DatasetVersion" -ForegroundColor Yellow
}

Write-Host "Final dataset version will be: $DatasetVersion" -ForegroundColor Green

Write-Host "Extracting metadata from dataset..." -ForegroundColor Green

# Extract metadata from dataset files
$datasetCodebaseVersion = jq -r '.codebase_version' "$infoJsonPath"
$datasetObservations = jq -r '.features | keys | join(",")' "$infoJsonPath"
$datasetFps = jq -r '.fps' "$infoJsonPath"
$datasetEpisodeCount = jq -r '.total_episodes' "$infoJsonPath"
$datasetFirstTask = jq -r 'select(.task_index == 0) | .task' "$tasksJsonlPath"
$datasetDescription = "Lerobot dataset with task $datasetFirstTask"

Write-Host "Dataset metadata:" -ForegroundColor Yellow
Write-Host "  Name: $DatasetName"
Write-Host "  Version: $DatasetVersion"
Write-Host "  Path: $DatasetPath"
Write-Host "  Codebase Version: $datasetCodebaseVersion"
Write-Host "  Observations: $datasetObservations"
Write-Host "  FPS: $datasetFps"
Write-Host "  Episode Count: $datasetEpisodeCount"
Write-Host "  First Task: $datasetFirstTask"
Write-Host "  Description: $datasetDescription"

Write-Host "Creating dataset in Azure ML..." -ForegroundColor Green

# Create the dataset in Azure ML
# Convert Windows path to Unix-style path for Azure ML compatibility
$normalizedPath = $DatasetPath.Replace('\', '/')
if ($normalizedPath.EndsWith('/')) {
    $normalizedPath = $normalizedPath.TrimEnd('/')
}

$createCommand = @(
    "az", "ml", "data", "create",
    "--name", $DatasetName,
    "--path", $normalizedPath,
    "--description", $datasetDescription,
    "--version", $DatasetVersion,
    "--set", "tags.observations=$datasetObservations",
    "--set", "tags.episodes=$datasetEpisodeCount", 
    "--set", "tags.fps=$datasetFps",
    "--set", "tags.task=$datasetFirstTask",
    "--set", "tags.codebase_version=$datasetCodebaseVersion",
    "--type", "uri_folder"
)

& $createCommand[0] $createCommand[1..($createCommand.Length-1)]

if ($LASTEXITCODE -eq 0) {
    Write-Host "Dataset '$DatasetName' version '$DatasetVersion' created successfully!" -ForegroundColor Green
} else {
    Write-Error "Failed to create dataset. Exit code: $LASTEXITCODE"
    exit $LASTEXITCODE
}