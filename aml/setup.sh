#!/bin/bash

# Azure Machine Learning Setup Script for LeRobot
# This script helps set up the Azure ML environment for LeRobot training

echo "🤖 LeRobot Azure ML Setup"
echo "========================="

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI is not installed. Please install it first:"
    echo "   https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if logged in to Azure
if ! az account show &> /dev/null; then
    echo "🔐 Please log in to Azure:"
    az login
fi

# Install Azure ML Python dependencies
echo "📦 Installing Azure ML dependencies..."
pip install -r aml/requirements.txt

# Create config.json if it doesn't exist
if [ ! -f "aml/config.json" ]; then
    echo "⚙️  Creating Azure ML configuration..."
    cp aml/config.json.template aml/config.json
    
    echo "📝 Please edit aml/config.json with your Azure ML workspace details:"
    echo "   - subscription_id"
    echo "   - resource_group" 
    echo "   - workspace_name"
    echo ""
    echo "You can find these details in the Azure portal under your ML workspace."
    
    if command -v code &> /dev/null; then
        echo "Opening config.json in VS Code..."
        code aml/config.json
    fi
else
    echo "✅ Configuration file already exists: aml/config.json"
fi

echo ""
echo "🚀 Next steps:"
echo "1. Edit aml/config.json with your workspace details"
echo "2. Run: python aml/create_environment.py"
echo "3. Run: python aml/submit_job.py"
echo ""
echo "For more details, see aml/README.md"
