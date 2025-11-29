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

az ml environment create --name lerobot --build-context . --dockerfile-path ./aml/docker/Dockerfile.aml --tags "git_hash=$(git rev-parse HEAD)" --description "Lerobot environment"