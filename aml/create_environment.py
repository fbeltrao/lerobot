"""
Script to create and register a custom environment in Azure Machine Learning
"""

import os

from azure.ai.ml import MLClient
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


def main():
    """Main function to create the environment"""
    # Initialize ML Client with default credentials
    # Make sure you're logged in with `az login` or have proper authentication set up
    ml_client = MLClient.from_config()  # Reads from config.json in current directory

    # Create the environment
    env = create_lerobot_environment(ml_client)
    print(f"Successfully created environment: {env.name}:{env.version}")


if __name__ == "__main__":
    main()
