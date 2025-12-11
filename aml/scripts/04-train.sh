#!/bin/bash


# Get extra arguments passed to the script
extra_arguments=("$@")

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

# Parse extra arguments to detect policy type
policy_type="act"
filtered_arguments=()

i=0
while [ $i -lt ${#extra_arguments[@]} ]; do
    arg="${extra_arguments[$i]}"
    if [[ "$arg" == "--set" && $((i + 1)) -lt ${#extra_arguments[@]} ]]; then
        set_arg="${extra_arguments[$((i + 1))]}"
        if [[ "$set_arg" =~ ^inputs\.policy_type=(.+)$ ]]; then
            policy_type="${BASH_REMATCH[1]}"
            echo "Detected policy type: $policy_type"
            # Skip both --set and the policy_type argument
            ((i++))
            ((i++))
            continue
        fi
    fi
    filtered_arguments+=("$arg")
    ((i++))
done

# Determine which YAML file to use
yaml_file="./aml/jobs/train.yaml"
if [[ -n "$policy_type" ]]; then
    policy_specific_file="./aml/jobs/train-$policy_type.yaml"
    if [[ -f "$policy_specific_file" ]]; then
        yaml_file="$policy_specific_file"
        echo "Using policy-specific configuration: $yaml_file"
    else
        echo "Policy-specific file $policy_specific_file not found, using generic train.yaml"
        # Add back the policy type argument since we're using the generic file
        filtered_arguments+=("--set" "inputs.policy_type=$policy_type")
    fi
fi

# Build the command with the determined YAML file
MODEL_DATA_STORE=${MODEL_DATA_STORE:-"workspaceblobstore"}
year=$(date +%Y)
month=$(date +%m)
output_path="azureml://datastores/$MODEL_DATA_STORE/paths/checkpoints/$policy_type/$year/$month/\${{name}}"
command=("az" "ml" "job" "create" "-f" "$yaml_file" "--set" "outputs.checkpoint.path=$output_path")

# Add any remaining extra arguments
if [[ ${#filtered_arguments[@]} -gt 0 ]]; then
    command+=("${filtered_arguments[@]}")
fi

# Display the command being executed
echo "${command[*]}"

# Execute the command
"${command[@]}"