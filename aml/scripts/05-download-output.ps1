# Download Azure ML Job Outputs using AzCopy
# Simple script to download checkpoint from job

param(
    [Parameter(Mandatory=$true)]
    [string]$JobName,
    [string]$OutputName = "checkpoint", 
    [string]$LocalPath = "./outputs",
    [string]$SubPath = ""
)

$ErrorActionPreference = "Stop"

Write-Host "=== Azure ML AzCopy Downloader ===" -ForegroundColor Cyan
Write-Host "Job: $JobName" -ForegroundColor Yellow
Write-Host "Output: $OutputName" -ForegroundColor Yellow
if ($SubPath) {
    Write-Host "SubPath: $SubPath" -ForegroundColor Yellow
}
Write-Host ""

try {
    # Step 1: Check prerequisites
    Write-Host "Step 1: Checking prerequisites..." -ForegroundColor Green
    
    if (!(Get-Command "az" -ErrorAction SilentlyContinue)) {
        throw "Azure CLI not found. Install from: https://learn.microsoft.com/cli/azure/install-azure-cli"
    }
    
    if (!(Get-Command "azcopy" -ErrorAction SilentlyContinue)) {
        throw "AzCopy not found. Install from: https://learn.microsoft.com/azure/storage/common/storage-use-azcopy-v10"
    }
    
    Write-Host "✓ Azure CLI and AzCopy found" -ForegroundColor Green

    # Step 2: Verify Azure login
    Write-Host "Step 2: Verifying Azure login..." -ForegroundColor Green
    
    $account = az account show 2>$null | ConvertFrom-Json
    if ($null -eq $account) {
        throw "Not logged into Azure. Run: az login"
    }
    
    Write-Host "✓ Logged in as: $($account.user.name)" -ForegroundColor Green

    # Step 3: Get job details
    Write-Host "Step 3: Getting job information..." -ForegroundColor Green
    
    $jobInfo = az ml job show --name $JobName | ConvertFrom-Json
    if ($null -eq $jobInfo) {
        throw "Job '$JobName' not found"
    }
    
    # For Azure ML jobs, we need to construct the path from the job name
    # The checkpoint output doesn't have a direct path, so we use the workspaceblobstore datastore
    # and construct the path based on the job name
    
    Write-Host "✓ Job found with status: $($jobInfo.status)" -ForegroundColor Green

    # Step 4: Parse storage details
    Write-Host "Step 4: Getting storage details..." -ForegroundColor Green
    
    # Try to find the datastore from job outputs first
    $datastoreName = $null
    $blobPath = $null
    
    if ($jobInfo.outputs -and $jobInfo.outputs.$OutputName) {
        # Extract datastore info from the output
        $outputInfo = $jobInfo.outputs.$OutputName
        if ($outputInfo.path) {
            # Parse the path format: azureml://datastores/datastore_name/paths/... or azureml://datastores/datastore_name/ExperimentRun/dcid.job_name
            if ($outputInfo.path -match "azureml://datastores/([^/]+)/(.+)") {
                $datastoreName = $matches[1]
                $pathPart = $matches[2]
                
                # Handle different path formats
                if ($pathPart -match "^paths/(.+)") {
                    # Standard path format
                    $blobPath = $matches[1].TrimEnd('/')
                } else {
                    # Use the path part as-is
                    $blobPath = $pathPart.TrimEnd('/')
                }
                
                Write-Host "✓ Found output path in job: $($outputInfo.path)" -ForegroundColor Green
            }
        } else {
            # Handle custom_model outputs without explicit paths
            if ($outputInfo.type -eq "custom_model") {
                Write-Host "✓ Found custom_model output without path, discovering datastore..." -ForegroundColor Green
                
                # Get the workspace default datastore
                $workspace = az ml workspace show | ConvertFrom-Json
                $defaultDatastoreName = $workspace.default_datastore
                
                if ($defaultDatastoreName) {
                    $datastoreName = $defaultDatastoreName
                    Write-Host "✓ Using workspace default datastore: $datastoreName" -ForegroundColor Green
                } else {
                    # Fallback to workspaceblobstore if no default is set
                    $datastoreName = "workspaceblobstore"
                    Write-Host "✓ Using fallback datastore: $datastoreName" -ForegroundColor Yellow
                }
                
                $blobPath = "azureml/$JobName/$OutputName"
            }
        }
    }
    
    # Try to find datastore from other outputs if not found
    if (-not $datastoreName -and $jobInfo.outputs) {
        foreach ($output in $jobInfo.outputs.PSObject.Properties) {
            if ($output.Value.path -and $output.Value.path -match "azureml://datastores/([^/]+)/") {
                $datastoreName = $matches[1]
                Write-Host "✓ Found datastore from '$($output.Name)' output: $datastoreName" -ForegroundColor Green
                break
            }
        }
    }
    
    # Fallback: Find the default datastore if not found in outputs
    if (-not $datastoreName) {
        Write-Host "Output path not found in job, discovering default datastore..." -ForegroundColor Yellow
        
        # Get the workspace default datastore
        $workspace = az ml workspace show | ConvertFrom-Json
        $defaultDatastoreName = $workspace.default_datastore
        
        if ($defaultDatastoreName) {
            $datastoreName = $defaultDatastoreName
            Write-Host "✓ Using workspace default datastore: $datastoreName" -ForegroundColor Green
        } else {
            # Fallback to workspaceblobstore if no default is set
            $datastoreName = "workspaceblobstore"
            Write-Host "✓ Using fallback datastore: $datastoreName" -ForegroundColor Yellow
        }
        
        $blobPath = "azureml/$JobName/$OutputName"
    }
    
    # For checkpoint outputs without explicit path, construct the standard path
    if (-not $blobPath -or $OutputName -eq "checkpoint") {
        $blobPath = "azureml/$JobName/$OutputName"
    }
    
    # Add sub-path if specified
    if ($SubPath) {
        $blobPath += '/' + $SubPath.TrimStart('/').Replace('\', '/')
    }
    
    # Ensure blob path ends with / for directory download
    if (-not $blobPath.EndsWith('/')) {
        $blobPath += '/'
    }
    
    $workspace = az ml workspace show | ConvertFrom-Json
    $storageAccount = ($workspace.storage_account -split '/')[-1]
    
    # Get the correct blob container name from the datastore
    $datastoreInfo = az ml datastore show --name $datastoreName | ConvertFrom-Json
    $containerName = $datastoreInfo.container_name
    
    Write-Host "✓ Storage Account: $storageAccount" -ForegroundColor Green
    Write-Host "✓ Container: $containerName" -ForegroundColor Green
    Write-Host "✓ Blob Path: $blobPath" -ForegroundColor Green

    # Step 5: Download with AzCopy
    Write-Host "Step 5: Downloading with AzCopy..." -ForegroundColor Green
    
    # Set up authentication
    $env:AZCOPY_AUTO_LOGIN_TYPE = "AZCLI"
    
    # Create download directory
    $downloadPath = Join-Path $LocalPath $JobName "output=$OutputName"
    if ($SubPath) {
        $downloadPath = Join-Path $downloadPath $SubPath
    }
    if (!(Test-Path $downloadPath)) {
        New-Item -ItemType Directory -Path $downloadPath -Force | Out-Null
    }
    
    # Build source URL
    $sourceUrl = "https://$storageAccount.blob.core.windows.net/$containerName/$blobPath/*"
    
    Write-Host "Source: $sourceUrl" -ForegroundColor Gray
    Write-Host "Destination: $downloadPath" -ForegroundColor Gray
    
    # Execute AzCopy
    azcopy copy $sourceUrl $downloadPath --recursive --overwrite=true
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Download completed successfully!" -ForegroundColor Green
        
        # Show downloaded files
        $files = Get-ChildItem -Path $downloadPath -Recurse
        Write-Host "`nDownloaded $($files.Count) files to: $downloadPath" -ForegroundColor Green
        
        $files | Select-Object -First 10 | ForEach-Object {
            Write-Host "  $($_.Name)" -ForegroundColor White
        }
        
        if ($files.Count -gt 10) {
            Write-Host "  ... and $($files.Count - 10) more files" -ForegroundColor Gray
        }
    } else {
        throw "AzCopy failed with exit code $LASTEXITCODE"
    }

} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "`nTroubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Make sure you're logged in: az login" -ForegroundColor White
    Write-Host "2. Install AzCopy: https://learn.microsoft.com/azure/storage/common/storage-use-azcopy-v10" -ForegroundColor White
    Write-Host "3. Verify job exists: az ml job show --name $JobName" -ForegroundColor White
    exit 1
} finally {
    # Clean up
    if ($env:AZCOPY_AUTO_LOGIN_TYPE) {
        Remove-Item env:AZCOPY_AUTO_LOGIN_TYPE -ErrorAction SilentlyContinue
    }
}

Write-Host "`nDone! Files downloaded to: $downloadPath" -ForegroundColor Cyan