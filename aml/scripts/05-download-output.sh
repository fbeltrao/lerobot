#!/bin/bash
# Download Azure ML Job Outputs using AzCopy
# Simple script to download checkpoint from job

set -e

# Default values
OUTPUT_NAME="checkpoint"
LOCAL_PATH="./outputs"
SUB_PATH=""

# Parse command line arguments
show_help() {
    echo "Usage: $0 -j JOB_NAME [-o OUTPUT_NAME] [-l LOCAL_PATH] [-s SUB_PATH]"
    echo ""
    echo "Options:"
    echo "  -j, --job-name     Job name (required)"
    echo "  -o, --output       Output name (default: checkpoint)"
    echo "  -l, --local-path   Local download path (default: ./outputs)"
    echo "  -s, --sub-path     Sub path within the output"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -j my-job-name -o checkpoint -l ./downloads"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        -l|--local-path)
            LOCAL_PATH="$2"
            shift 2
            ;;
        -s|--sub-path)
            SUB_PATH="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if job name is provided
if [[ -z "$JOB_NAME" ]]; then
    echo "Error: Job name is required"
    show_help
    exit 1
fi

echo -e "\033[96m=== Azure ML AzCopy Downloader ===\033[0m"
echo -e "\033[93mJob: $JOB_NAME\033[0m"
echo -e "\033[93mOutput: $OUTPUT_NAME\033[0m"
if [[ -n "$SUB_PATH" ]]; then
    echo -e "\033[93mSubPath: $SUB_PATH\033[0m"
fi
echo ""

cleanup() {
    if [[ -n "$AZCOPY_AUTO_LOGIN_TYPE" ]]; then
        unset AZCOPY_AUTO_LOGIN_TYPE
    fi
}

trap cleanup EXIT

# Step 1: Check prerequisites
echo -e "\033[92mStep 1: Checking prerequisites...\033[0m"

if ! command -v az &> /dev/null; then
    echo "Error: Azure CLI not found. Install from: https://learn.microsoft.com/cli/azure/install-azure-cli"
    exit 1
fi

if ! command -v azcopy &> /dev/null; then
    echo "Error: AzCopy not found. Install from: https://learn.microsoft.com/azure/storage/common/storage-use-azcopy-v10"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "Error: jq not found. Install with: sudo apt-get install jq (Ubuntu/Debian) or brew install jq (macOS)"
    exit 1
fi

echo -e "\033[92m✓ Azure CLI, AzCopy, and jq found\033[0m"

# Step 2: Verify Azure login
echo -e "\033[92mStep 2: Verifying Azure login...\033[0m"

if ! account_info=$(az account show 2>/dev/null); then
    echo "Error: Not logged into Azure. Run: az login"
    exit 1
fi

user_name=$(echo "$account_info" | jq -r '.user.name')
echo -e "\033[92m✓ Logged in as: $user_name\033[0m"

# Step 3: Get job details
echo -e "\033[92mStep 3: Getting job information...\033[0m"

if ! job_info=$(az ml job show --name "$JOB_NAME" 2>/dev/null); then
    echo "Error: Job '$JOB_NAME' not found"
    exit 1
fi

job_status=$(echo "$job_info" | jq -r '.status')
echo -e "\033[92m✓ Job found with status: $job_status\033[0m"

# Step 4: Parse storage details
echo -e "\033[92mStep 4: Getting storage details...\033[0m"

datastore_name=""
blob_path=""

# Try to find the datastore from job outputs first
output_path=$(echo "$job_info" | jq -r --arg output "$OUTPUT_NAME" '.outputs[$output].path // empty')

if [[ -n "$output_path" && "$output_path" != "null" ]]; then
    # Parse the path format: azureml://datastores/datastore_name/paths/... or azureml://datastores/datastore_name/ExperimentRun/dcid.job_name
    if [[ "$output_path" =~ azureml://datastores/([^/]+)/(.+) ]]; then
        datastore_name="${BASH_REMATCH[1]}"
        path_part="${BASH_REMATCH[2]}"
        
        # Handle different path formats
        if [[ "$path_part" =~ ^paths/(.+) ]]; then
            # Standard path format
            blob_path="${BASH_REMATCH[1]%/}"
        else
            # Use the path part as-is
            blob_path="${path_part%/}"
        fi
        
        echo -e "\033[92m✓ Found output path in job: $output_path\033[0m"
    fi
else
    # Handle custom_model outputs without explicit paths
    output_type=$(echo "$job_info" | jq -r --arg output "$OUTPUT_NAME" '.outputs[$output].type // empty')
    if [[ "$output_type" == "custom_model" ]]; then
        echo -e "\033[92m✓ Found custom_model output without path, discovering datastore...\033[0m"
        
        # Get the workspace default datastore
        workspace_info=$(az ml workspace show)
        default_datastore=$(echo "$workspace_info" | jq -r '.default_datastore // empty')
        
        if [[ -n "$default_datastore" && "$default_datastore" != "null" ]]; then
            datastore_name="$default_datastore"
            echo -e "\033[92m✓ Using workspace default datastore: $datastore_name\033[0m"
        else
            # Fallback to workspaceblobstore if no default is set
            datastore_name="workspaceblobstore"
            echo -e "\033[93m✓ Using fallback datastore: $datastore_name\033[0m"
        fi
        
        blob_path="azureml/$JOB_NAME/$OUTPUT_NAME"
    fi
fi

# Try to find datastore from other outputs if not found
if [[ -z "$datastore_name" ]]; then
    # Get all output paths and find a datastore
    output_paths=$(echo "$job_info" | jq -r '.outputs // {} | to_entries[] | select(.value.path) | .value.path')
    for path in $output_paths; do
        if [[ "$path" =~ azureml://datastores/([^/]+)/ ]]; then
            datastore_name="${BASH_REMATCH[1]}"
            echo -e "\033[92m✓ Found datastore from output: $datastore_name\033[0m"
            break
        fi
    done
fi

# Fallback: Find the default datastore if not found in outputs
if [[ -z "$datastore_name" ]]; then
    echo -e "\033[93mOutput path not found in job, discovering default datastore...\033[0m"
    
    # Get the workspace default datastore
    workspace_info=$(az ml workspace show)
    default_datastore=$(echo "$workspace_info" | jq -r '.default_datastore // empty')
    
    if [[ -n "$default_datastore" && "$default_datastore" != "null" ]]; then
        datastore_name="$default_datastore"
        echo -e "\033[92m✓ Using workspace default datastore: $datastore_name\033[0m"
    else
        # Fallback to workspaceblobstore if no default is set
        datastore_name="workspaceblobstore"
        echo -e "\033[93m✓ Using fallback datastore: $datastore_name\033[0m"
    fi
    
    blob_path="azureml/$JOB_NAME/$OUTPUT_NAME"
fi

# For checkpoint outputs without explicit path, construct the standard path
if [[ -z "$blob_path" || "$OUTPUT_NAME" == "checkpoint" ]]; then
    blob_path="azureml/$JOB_NAME/$OUTPUT_NAME"
fi

# Add sub-path if specified
if [[ -n "$SUB_PATH" ]]; then
    SUB_PATH="${SUB_PATH#/}"  # Remove leading slash
    SUB_PATH="${SUB_PATH//\\//}"  # Replace backslashes with forward slashes
    blob_path="$blob_path/$SUB_PATH"
fi

# Ensure blob path ends with / for directory download
if [[ "$blob_path" != */ ]]; then
    blob_path="$blob_path/"
fi

workspace_info=$(az ml workspace show)
storage_account=$(echo "$workspace_info" | jq -r '.storage_account' | sed 's|.*/||')

# Get the correct blob container name from the datastore
datastore_info=$(az ml datastore show --name "$datastore_name")
container_name=$(echo "$datastore_info" | jq -r '.container_name')

echo -e "\033[92m✓ Storage Account: $storage_account\033[0m"
echo -e "\033[92m✓ Container: $container_name\033[0m"
echo -e "\033[92m✓ Blob Path: $blob_path\033[0m"

# Step 5: Download with AzCopy
echo -e "\033[92mStep 5: Downloading with AzCopy...\033[0m"

# Set up authentication
echo "Authenticating AzCopy with Azure CLI..."
export AZCOPY_AUTO_LOGIN_TYPE="AZCLI"
if ! azcopy login --login-type azcli; then
    echo "Error: Failed to authenticate AzCopy with Azure CLI"
    exit 1
fi
echo -e "\033[92m✓ AzCopy authenticated successfully\033[0m"

# Create download directory
download_path="$LOCAL_PATH/$JOB_NAME/output=$OUTPUT_NAME"
if [[ -n "$SUB_PATH" ]]; then
    download_path="$download_path/$SUB_PATH"
fi
mkdir -p "$download_path"

# Build source URL
source_url="https://$storage_account.blob.core.windows.net/$container_name/$blob_path*"

echo -e "\033[90mSource: $source_url\033[0m"
echo -e "\033[90mDestination: $download_path\033[0m"

# Execute AzCopy
echo "[Downloading...] azcopy copy \"$source_url\" \"$download_path\" --recursive --overwrite=true"
if azcopy copy "$source_url" "$download_path" --recursive --overwrite=true; then
    echo -e "\033[92m✓ Download completed successfully!\033[0m"
    
    # Show downloaded files
    if [[ -d "$download_path" ]]; then
        file_count=$(find "$download_path" -type f | wc -l)
        echo ""
        echo -e "\033[92mDownloaded $file_count files to: $download_path\033[0m"
        
        # Show first 10 files
        find "$download_path" -type f -printf "%f\n" | head -10 | while read -r filename; do
            echo -e "  \033[97m$filename\033[0m"
        done
        
        if [[ $file_count -gt 10 ]]; then
            remaining=$((file_count - 10))
            echo -e "  \033[90m... and $remaining more files\033[0m"
        fi
    fi
else
    echo "Error: AzCopy failed with exit code $?"
    echo ""
    echo -e "\033[93mTroubleshooting:\033[0m"
    echo -e "\033[97m1. Make sure you're logged in: az login\033[0m"
    echo -e "\033[97m2. Install AzCopy: https://learn.microsoft.com/azure/storage/common/storage-use-azcopy-v10\033[0m"
    echo -e "\033[97m3. Verify job exists: az ml job show --name $JOB_NAME\033[0m"
    exit 1
fi

echo ""
echo -e "\033[96mDone! Files downloaded to: $download_path\033[0m"