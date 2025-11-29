# Training Lerobot in Azure Machine Learning

Training Lerobot policies in Azure ML can be an alternative if you face a few of the following challenges:

**Working with sensitive or proprietary robotics data**
You have datasets that contain proprietary information or need to comply with strict security requirements. Your robotics data can't leave your organization's network boundaries, and you need enterprise-grade security controls while still being able to train effectively.

**Limited by local hardware constraints**
Your local machine doesn't have enough GPU power for complex model training, or you're stuck waiting hours (or days) for training jobs to complete. You need access to high-end hardware like H100s or A100s but don't want to make massive upfront investments in equipment that might become outdated or sit idle.

**Managing training infrastructure is becoming a headache**
You're spending too much time setting up environments, tracking experiments, managing dependencies, or coordinating with team members. You want to focus on developing better policies rather than dealing with infrastructure, resource management, and collaboration tools.

## Pre-requisites

- [Azure cli](https://learn.microsoft.com/cli/azure/install-azure-cli?view=azure-cli-latest) with [machine learning extension](https://learn.microsoft.com/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public).
- Login with Azure cli (`az login`)
- Setup Azure cli to use our current workspace as default: `az configure --defaults workspace=<azure-machine-learning-workspace-name> group=<resource-group-where-machine-learning-workspace-is-located>`. If you don't want to setup defaults, add parameters `-g <resource-group> and -w <workspace-name>` to all usages of Azure command line.
- Install [jq](https://jqlang.org/) to allow JSON parsing in command line.

## Step 1 - Create environment

Azure ML needs a consistent, reproducible environment to run your training jobs. This step creates a containerized environment (like a virtual machine image) that includes all the dependencies, libraries, and configurations needed to run Lerobot. Think of it as packaging your entire development environment into a reusable container that can be deployed anywhere in Azure.

1. Run command to create environment

```
01-create-env.ps1

```

## Step 2 - Upload dataset

Azure ML needs access to your training data, but it runs in the cloud where your local files aren't accessible. This step uploads your dataset to Azure's cloud storage and registers it as a managed data asset. This allows Azure ML to efficiently distribute your data to compute resources and enables data versioning, lineage tracking, and team collaboration. The metadata tags help you organize and discover datasets later.

```bash
./aml/02-create-dataset.ps1 -DatasetName <dataset-name> -DatasetPath <local-path>
```


## Step 3 - Run fine-tuning job

Yay, we can finally train our model! By creating a job, we are instructing Azure ML to provision the requested compute resources (GPUs), load your environment and dataset, then execute the training script. Azure ML handles compute management, experiment tracking and logging.

```
az ml job create -f ./aml/train.yaml

# to set the dataset
az ml job create -f ./aml/train.yaml --set "inputs.dataset.path=azureml:<data-asset-name>:<version or @latest>"

# to set the compute name
az ml job create -f ./aml/train.yaml --set "compute=<compute-name>"
```

## Step 4 - Download checkpoint

After training completes, your trained model (checkpoint files) exists in Azure's cloud storage. To use the model locally for inference, evaluation, or further development, you need to download these checkpoint files back to your local machine. Azure ML automatically manages these outputs and provides easy commands to retrieve them.

Once an training job is finished you can download the checkpoint files locally.
1. Identify the job name
2. Download the output locally `az ml job download --name <job-name> --download-path ./outputs --output-name checkpoint`

## Advanced

Depending on the police type you might require to download base weights from protected hugging face repositories (i.e, pi0fast).
To access them you will need to authenticate the job passing your HF_TOKEN. We don't want to pass as plain value to the job.
A solution is to store the secret in key vault and read it from the job.

### Storing the token in Key Vault

Either create manually or with Azure Portal a secret called hf_token in the Key Vault associated with your Azure Machine Learning Workspace.

To find out the Key Vault run command:

```Powershell
# Powershell
./aml/03-create-hf-token-secret.ps1 -HfToken (Read-Host "Enter HF Token" -AsSecureString)
```

```bash
# Bash
```

```powershell
$KEYVAULT_NAME=($(az ml workspace show --query "key_vault" -o tsv) -split '/')[-1]
az ml job create -f ./aml/train-with-hf-token.yaml --set "inputs.keyvault=$KEYVAULT_NAME" --set "inputs.policy_type=pi0fast"
```