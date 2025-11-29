<#
.SYNOPSIS
    Creates or updates a Hugging Face token secret in Azure Key Vault.

.DESCRIPTION
    This script securely stores a Hugging Face token in the Azure Key Vault associated with your Azure ML workspace.
    The token is passed as a SecureString parameter to prevent it from being logged or displayed in plain text.

.PARAMETER HfToken
    The Hugging Face token to store in the Key Vault. This is a SecureString parameter for security.

.PARAMETER SecretName
    The name of the secret in Key Vault. Defaults to "hf-token" if not specified.

.EXAMPLE
    # Interactive prompt for token (recommended)
    .\03-create-hf-token-secret.ps1 -HfToken (Read-Host "Enter HF Token" -AsSecureString)

.EXAMPLE
    # Convert plain text to SecureString
    $secureToken = ConvertTo-SecureString "hf_your_token_here" -AsPlainText -Force
    .\03-create-hf-token-secret.ps1 -HfToken $secureToken

.EXAMPLE
    # With custom secret name
    .\03-create-hf-token-secret.ps1 -HfToken (Read-Host "Enter HF Token" -AsSecureString) -SecretName "my-hf-token"
#>

param(
    [Parameter(Mandatory=$true, HelpMessage="The Hugging Face token to store in the Key Vault")]
    [Alias("Token")]
    [Security.SecureString]$HfToken,
    
    [Parameter(Mandatory=$false, HelpMessage="The name of the secret in Key Vault (default: hf-token)")]
    [string]$SecretName = "hf-token"
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


# Convert SecureString to plain text for validation and API calls
$HfTokenPlain = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($HfToken))

# Validate the HF token format (basic validation)
if ([string]::IsNullOrWhiteSpace($HfTokenPlain)) {
    Write-Error "Hugging Face token cannot be empty"
    exit 1
}

if ($HfTokenPlain.Length -lt 10) {
    Write-Warning "The provided token seems too short. Please ensure it's a valid Hugging Face token."
}

Write-Host "Using secret name: $SecretName" -ForegroundColor Cyan

# Get the Key Vault name from the Azure ML workspace
Write-Host "Getting Key Vault information from Azure ML workspace..." -ForegroundColor Yellow
try {
    $KEYVAULT_NAME = ($(az ml workspace show --query "key_vault" -o tsv) -split '/')[-1]
    if ([string]::IsNullOrWhiteSpace($KEYVAULT_NAME)) {
        throw "Could not retrieve Key Vault name"
    }
    Write-Host "✓ Found Key Vault: $KEYVAULT_NAME" -ForegroundColor Green
} catch {
    Write-Error "Failed to get Key Vault name from Azure ML workspace. Error: $_"
    exit 1
}

# Check if the secret already exists
Write-Host "Checking if secret '$SecretName' already exists in Key Vault..." -ForegroundColor Yellow
$existingSecret = az keyvault secret show --vault-name $KEYVAULT_NAME --name $SecretName 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Secret '$SecretName' already exists. It will be updated." -ForegroundColor Yellow
} else {
    Write-Host "Secret '$SecretName' does not exist. It will be created." -ForegroundColor Cyan
}

# Create or update the secret
Write-Host "Setting Hugging Face token in Key Vault secret '$SecretName'..." -ForegroundColor Yellow
try {
    $result = az keyvault secret set --vault-name $KEYVAULT_NAME --name $SecretName --value $HfTokenPlain --output json
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to set secret in Key Vault"
    }
    
    $secretInfo = $result | ConvertFrom-Json
    Write-Host "✓ Successfully created/updated secret '$SecretName' in Key Vault '$KEYVAULT_NAME'" -ForegroundColor Green
    Write-Host "  Secret ID: $($secretInfo.id)" -ForegroundColor Gray
    Write-Host "  Version: $($secretInfo.attributes.version)" -ForegroundColor Gray
    Write-Host "  Updated: $($secretInfo.attributes.updated)" -ForegroundColor Gray
} catch {
    Write-Error "Failed to create/update the secret in Key Vault. Error: $_"
    Write-Error "Please ensure you have the necessary permissions to write secrets to the Key Vault."
    exit 1
} finally {
    # Clear the plain text token from memory
    if ($HfTokenPlain) {
        $HfTokenPlain = $null
        [System.GC]::Collect()
    }
}

Write-Host ""
Write-Host "✓ Hugging Face token has been successfully stored in Azure Key Vault!" -ForegroundColor Green
Write-Host "You can now reference this secret in your Azure ML jobs using:" -ForegroundColor Cyan
Write-Host "  Secret name: $SecretName" -ForegroundColor White
Write-Host "  Key Vault: $KEYVAULT_NAME" -ForegroundColor White

