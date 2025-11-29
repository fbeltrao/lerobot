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

# Build the command with the fixed -f parameter and any additional arguments
$command = @("ml", "job", "create", "-f", "./aml/jobs/train.yaml")

# Add any extra arguments passed to the script
if ($ExtraArguments) {
    $command += $ExtraArguments
}

# Display the command being executed
Write-Host "az $($command -join ' ')"

# Execute the command
& az @command