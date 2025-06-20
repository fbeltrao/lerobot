# Azure Machine Learning Setup for LeRobot

1. Set environment variables
```plain
export DATASET=so101_multi_task_v2
export HF_USER=$(huggingface-cli whoami)
export STEPS=50
export POLICY_TYPE=pi0fast
export HF_TOKEN=
```

2. Run
```
az ml job create --file aml/job_pi0.yaml --resource-group robotics-ch-north-secure --workspace-name robotics-ch-north-secure --name "${POLICY_TYPE}_${DATASET}_${STEPS}steps"
```

