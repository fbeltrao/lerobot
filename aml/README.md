# Azure Machine Learning Setup for LeRobot

This folder contains the necessary files to run LeRobot training on Azure Machine Learning.

## Files Overview

- `environment.yml` - Conda environment specification for the custom environment
- `Dockerfile` - Docker image definition for the custom environment
- `create_environment.py` - Script to create and register the custom environment in Azure ML
- `train_script.py` - Training script adapted for Azure ML
- `submit_job.py` - Script to submit training jobs to Azure ML
- `requirements.txt` - Azure ML Python dependencies
- `config.json.template` - Template for Azure ML workspace configuration

## Setup Instructions

### 1. Prerequisites

- Azure subscription with Azure Machine Learning workspace
- Azure CLI installed and configured
- Python environment with Azure ML SDK v2

### 2. Install Dependencies

```bash
pip install -r aml/requirements.txt
```

### 3. Configure Azure ML Workspace

1. Copy the configuration template:
   ```bash
   cp aml/config.json.template aml/config.json
   ```

2. Edit `aml/config.json` with your Azure ML workspace details:
   ```json
   {
       "subscription_id": "your-subscription-id",
       "resource_group": "your-resource-group",
       "workspace_name": "your-workspace-name"
   }
   ```

3. Authenticate with Azure:
   ```bash
   az login
   ```

### 4. Create Custom Environment

Run the environment creation script:

```bash
cd /path/to/lerobot
python aml/create_environment.py
```

This will:
- Build a Docker image with conda and Python 3.10
- Create the `lerobot` conda environment
- Install LeRobot in editable mode
- Register the environment in Azure ML

### 5. Submit Training Jobs

Use the job submission script:

```bash
python aml/submit_job.py
```

Or customize the training parameters by modifying the script or creating your own submission script.

## Environment Details

The custom environment includes:
- **Base Image**: Azure ML CUDA-enabled Ubuntu 22.04
- **Python Version**: 3.10
- **Conda Environment**: `lerobot`
- **LeRobot Installation**: Editable install (`pip install -e .`)

### System Dependencies

The environment includes all necessary system dependencies:
- Build tools (cmake, build-essential)
- Graphics libraries (libgl1-mesa-glx, libegl1-mesa)
- Media processing (ffmpeg)
- Other LeRobot dependencies

## Training Configuration

The training script (`train_script.py`) supports:
- Dataset specification
- Policy type selection
- Configurable epochs and batch size
- Output path for model artifacts
- Azure ML input/output handling

### Example Usage

```python
from azure.ai.ml import MLClient
from aml.submit_job import submit_training_job

ml_client = MLClient.from_config()

job = submit_training_job(
    ml_client=ml_client,
    dataset_name="lerobot/aloha_sim_insertion_human",
    policy_type="act",
    epochs=50,
    batch_size=32,
    compute_target="gpu-cluster"
)
```

## Monitoring

- Jobs can be monitored through Azure ML Studio
- Logs and metrics are automatically captured
- Model artifacts are saved to the specified output path

## Troubleshooting

### Common Issues

1. **Authentication Error**: Ensure you're logged in with `az login`
2. **Environment Build Failure**: Check Dockerfile and ensure all dependencies are available
3. **Compute Target Not Found**: Verify your compute cluster name in Azure ML workspace

### Environment Rebuilding

If you need to update the environment:

1. Modify the `Dockerfile` or `environment.yml`
2. Update the version in `create_environment.py`
3. Re-run the environment creation script

## Notes

- The environment uses GPU-enabled base images for optimal training performance
- All LeRobot dependencies are included in the editable installation
- The setup follows Azure ML best practices for custom environments
