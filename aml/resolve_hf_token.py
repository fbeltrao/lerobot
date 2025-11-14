# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "azure-identity==1.24.0",
#   "azure-keyvault-secrets==4.10.0",
# ]
# ///
import os
import sys
import argparse
from pathlib import Path
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient

def main():
    parser = argparse.ArgumentParser(description='Retrieve HF token from Azure Key Vault')
    parser.add_argument('--keyvault-name', 
                       default=os.getenv('KEYVAULT_NAME', ''),
                       help='Azure Key Vault name (default: from KEYVAULT_NAME env var)')
    parser.add_argument('--secret-name', 
                       default=os.getenv('HF_TOKEN_SECRET_NAME', 'hf-token'),
                       help='Name of the secret in Key Vault (default: from HF_TOKEN_SECRET_NAME env var or hf-token)')

    args = parser.parse_args()
    
    # Check if HF_TOKEN is already available as environment variable
    if os.getenv('HF_TOKEN'):
        hf_token = os.getenv('HF_TOKEN')
        print("Using HF_TOKEN from environment variable", file=sys.stderr)
        return

    keyvault_name = (args.keyvault_name or "").strip()
    if not keyvault_name:
        print("No key vault name provided, skipping HF token retrieval", file=sys.stderr)
        return

    print(f"HF_TOKEN not found in environment, retrieving from Key Vault '{keyvault_name}'", file=sys.stderr)

    # See: https://github.com/Azure/azure-sdk-for-python/issues/32921
    client_id = os.environ.get("AZURE_CLIENT_ID", None)
    if client_id is None:
        os.environ["AZURE_CLIENT_ID"] = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
        print(f"AZURE_CLIENT_ID not found, using DEFAULT_IDENTITY_CLIENT_ID", file=sys.stderr)

    # Try ManagedIdentityCredential first for Azure ML environments
    if os.getenv('AZUREML_RUN_ID'):
        try:
            credential = ManagedIdentityCredential()
            # Test the credential by attempting to get a token
            credential.get_token("https://vault.azure.net/.default")
            print("Using ManagedIdentityCredential for authentication", file=sys.stderr)
        except Exception as e:
            print(f"ManagedIdentityCredential failed: {e}, falling back to DefaultAzureCredential", file=sys.stderr)
            credential = DefaultAzureCredential()
    else:
        credential = DefaultAzureCredential()

    secret_client = SecretClient(vault_url=f"https://{keyvault_name}.vault.azure.net/", credential=credential)
    hf_token = secret_client.get_secret(args.secret_name).value
    
    # Determine the token file path according to HF documentation
    # Priority: HF_TOKEN_PATH -> HF_HOME/token -> ~/.cache/huggingface/token
    if os.getenv('HF_TOKEN_PATH'):
        token_path = Path(os.getenv('HF_TOKEN_PATH'))
    else:
        hf_home = os.getenv('HF_HOME')
        if hf_home:
            token_path = Path(hf_home) / 'token'
        else:
            # Default location: ~/.cache/huggingface/token
            cache_home = os.getenv('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
            token_path = Path(cache_home) / 'huggingface' / 'token'
    
    # Create the directory if it doesn't exist
    token_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write token to the standard HF location
    with open(token_path, 'w') as f:
        f.write(hf_token)

    # Print the token file path for reference
    print(f"HF token written to: {token_path}", file=sys.stderr)    

if __name__ == "__main__":
    main()