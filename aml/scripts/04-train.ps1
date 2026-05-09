#!/usr/bin/env pwsh

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArguments
)

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

# Parse extra arguments to detect policy type
$policyType = "act"
$filteredArguments = @()

for ($i = 0; $i -lt $ExtraArguments.Count; $i++) {
    $arg = $ExtraArguments[$i]
    if ($arg -eq "--set" -and $i + 1 -lt $ExtraArguments.Count) {
        $setArg = $ExtraArguments[$i + 1]
        if ($setArg -match '^inputs\.policy_type=(.+)$') {
            $policyType = $matches[1]
            Write-Host "Detected policy type: $policyType" -ForegroundColor Yellow
            # Skip both --set and the policy_type argument
            $i++
            continue
        }
    }
    $filteredArguments += $arg
}

# Determine which YAML file to use
$yamlFile = "./aml/jobs/train.yaml"
if ($policyType) {
    $policySpecificFile = "./aml/jobs/train-$policyType.yaml"
    if (Test-Path $policySpecificFile) {
        $yamlFile = $policySpecificFile
        Write-Host "Using policy-specific configuration: $yamlFile" -ForegroundColor Green
    } else {
        Write-Host "Policy-specific file $policySpecificFile not found, using generic train.yaml" -ForegroundColor Yellow
        # Add back the policy type argument since we're using the generic file
        $filteredArguments += "--set"
        $filteredArguments += "inputs.policy_type=$policyType"
    }
}

# Build the command with the determined YAML file
$modelDatastore = if ($env:MODEL_DATA_STORE) { $env:MODEL_DATA_STORE } else { "workspaceblobstore" }
$year = Get-Date -Format "yyyy"
$month = Get-Date -Format "MM"
$outputPath = "azureml://datastores/$modelDatastore/paths/checkpoints/$policyType/$year/$month/`${{name}}"
$command = @("ml", "job", "create", "-f", $yamlFile, "--set", "outputs.checkpoint.path=$outputPath")

# Add any remaining extra arguments
if ($filteredArguments) {
    $command += $filteredArguments
}

# Display the command being executed
Write-Host "az $($command -join ' ')"

# Execute the command
& az @command