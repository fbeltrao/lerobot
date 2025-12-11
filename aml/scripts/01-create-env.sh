#!/bin/bash

# Check if Azure CLI is available
if ! command -v az &> /dev/null; then
    echo "Error: Azure CLI is not installed or not in PATH. Please install from https://learn.microsoft.com/cli/azure/install-azure-cli" >&2
    exit 1
fi

if ! az version &> /dev/null; then
    echo "Error: Azure CLI is not working properly" >&2
    exit 1
fi

echo "✓ Azure CLI is available"

az ml environment create --name lerobot --build-context . --dockerfile-path ./aml/docker/Dockerfile.aml --tags "git_hash=$(git rev-parse HEAD)" --description "Lerobot environment"
az ml environment create --name lerobot-cuda-dev --build-context . --dockerfile-path ./aml/docker/Dockerfile.aml.cuda-dev --tags "git_hash=$(git rev-parse HEAD)" --description "Lerobot environment CUDA development"