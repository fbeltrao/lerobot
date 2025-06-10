"""
Script to submit LeRobot training jobs to Azure Machine Learning
"""

import os
from pathlib import Path
from typing import Any

from azure.ai.ml import MLClient, Output, command
from azure.ai.ml.entities import BuildContext, Environment
from azure.identity import DefaultAzureCredential


def create_lerobot_environment(ml_client: MLClient, environment_name: str = "lerobot-env"):
    """
    Create and register the LeRobot custom environment in Azure ML

    Args:
        ml_client: Azure ML client
        environment_name: Name for the environment
    """

    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    # Create environment with build context
    lerobot_env = Environment(
        name=environment_name,
        description="Custom environment for LeRobot training with conda and Python 3.10",
        build=BuildContext(
            path=repo_root,  # Build context is the repository root
            dockerfile_path="docker/lerobot-gpu/Dockerfile",  # Dockerfile path relative to build context
        ),
        version="1.0",
    )

    # Register the environment
    print(f"Creating environment: {environment_name}")
    env = ml_client.environments.create_or_update(lerobot_env)
    print(f"Environment created: {env.name}:{env.version}")

    return env


class TrainingConfig:
    def __init__(
        self,
        compute_target: str,
        dataset_name: str,
        policy_type: str,
        arguments: dict[str, Any],
    ):
        self.compute_target = compute_target
        self.dataset_name = dataset_name
        self.policy_type = policy_type
        self.arguments = arguments

    @staticmethod
    def from_args():
        """
        Parses command line arguments to create a TrainingConfig instance.
        """
        import argparse

        parser = argparse.ArgumentParser(description="LeRobot training configuration")
        parser.add_argument(
            "--dataset.repo_id", type=str, required=True, help="Name of the dataset to train on"
        )
        parser.add_argument("--policy.type", type=str, required=True, help="Type of policy to train")
        parser.add_argument("--compute_target", type=str, required=True, help="Cluster to run the job on")

        # Parse known and unknown arguments to capture all arguments
        args, unknown_args = parser.parse_known_args()
        all_args = vars(args)

        # Parse unknown arguments and add them to all_args
        for i in range(0, len(unknown_args), 2):
            if i + 1 < len(unknown_args):
                key = unknown_args[i].lstrip("-")
                value = unknown_args[i + 1]
                all_args[key] = value

        # Remove compute_target from arguments as it's handled separately
        compute_target = all_args.pop("compute_target", None)

        return TrainingConfig(
            compute_target,
            getattr(args, "dataset.repo_id"),
            getattr(args, "policy.type"),
            all_args,
        )


def args_to_command_line(args: dict[str, Any]) -> str:
    """
    Convert a dictionary of arguments to a command line string.

    Args:
        args: Dictionary of arguments

    Returns:
        Command line string
    """

    def render_key(key: str) -> str:
        """Render the key for command line arguments"""
        if len(key) == 1:
            return f"-{key}"

        return f"--{key}"

    def render_value(value: Any) -> str:
        """Render the value for command line arguments"""
        if value is None:
            return ""

        return " " + str(value)

    return " ".join(f"{render_key(key)}{render_value(value)}" for key, value in args.items())


def submit_training_job(
    ml_client: MLClient,
    experiment_name: str = "lerobot-training",
):
    """
    Submit a training job to Azure ML

    Args:
        ml_client: Azure ML client
        experiment_name: Name of the experiment
        compute_target: Name of the compute target
    """

    # Create the command job
    config = TrainingConfig.from_args()
    python_command = f"python lerobot/scripts/train.py --output_dir ${{{{outputs.model_output}}}} {args_to_command_line(config.arguments)}"
    job = command(
        experiment_name=experiment_name,
        display_name=f"lerobot-{config.dataset_name.replace('/', '-')}",
        description=f"Training {config.policy_type} policy on {config.dataset_name} dataset",
        # Compute configuration
        compute=config.compute_target,
        environment=create_lerobot_environment(ml_client),
        # Job configuration
        command=python_command,
        # Outputs
        outputs={"model_output": Output(type="uri_folder", mode="rw_mount")},
        # Resource configuration
        instance_count=1,
        # Tags
        tags={
            "framework": "lerobot",
            "policy": config.policy_type,
            "dataset": config.dataset_name,
        },
    )

    # Submit the job
    print(f"Submitting training job for {config.policy_type} on {config.dataset_name}...")
    submitted_job = ml_client.jobs.create_or_update(job)

    print("Job submitted successfully!")
    print(f"Job name: {submitted_job.name}")
    print(f"Job URL: {submitted_job.studio_url}")

    return submitted_job


def main():
    """Main function to submit a training job"""

    # python aml/submit_job.py --compute_target NC24adsA100v4-singlenode-cluster --policy.type pi0fast --dataset.repo_id fbeltrao/so101_unplug_cable_4 --steps 10

    # Initialize ML Client
    ml_client = MLClient.from_config(
        credential=DefaultAzureCredential(),
        path=Path(__file__).parent / "config.json",
    )

    # Submit a sample training job
    job = submit_training_job(
        ml_client=ml_client,
    )

    print(f"Training job submitted: {job.name}")
    print(f"Monitor progress at: {job.studio_url}")


if __name__ == "__main__":
    main()
