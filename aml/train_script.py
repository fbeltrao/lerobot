"""
Sample training script for Azure Machine Learning
This script demonstrates how to run LeRobot training in Azure ML
"""

import argparse
import os
import sys
from pathlib import Path

# Add the lerobot package to the path
sys.path.insert(0, "/workspace/lerobot")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="LeRobot training in Azure ML")

    # Azure ML specific arguments
    parser.add_argument("--data_path", type=str, help="Path to input data")
    parser.add_argument("--output_path", type=str, help="Path to output directory")

    # LeRobot training arguments
    parser.add_argument(
        "--config", type=str, default="lerobot/configs/policy/act.yaml", help="Path to training config file"
    )
    parser.add_argument("--policy", type=str, default="act", help="Policy type to train")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name to train on")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")

    return parser.parse_args()


def setup_environment():
    """Setup the training environment"""
    print("Setting up LeRobot training environment...")

    # Set environment variables
    os.environ["MUJOCO_GL"] = "egl"

    # Print system information
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    # Check if lerobot is importable
    try:
        import lerobot

        print(f"LeRobot version: {lerobot.__version__}")
    except ImportError as e:
        print(f"Error importing lerobot: {e}")
        sys.exit(1)


def main():
    """Main training function"""
    args = parse_args()

    print("Starting LeRobot training in Azure ML...")
    setup_environment()

    # Import lerobot modules
    from lerobot.scripts.train import train_policy
    from lerobot.common.utils.utils import init_hydra_config

    # Setup training configuration
    config_overrides = [
        f"dataset_repo_id={args.dataset}",
        f"training.batch_size={args.batch_size}",
        f"training.num_epochs={args.epochs}",
    ]

    if args.output_path:
        config_overrides.append(f"hydra.run.dir={args.output_path}")

    # Initialize Hydra configuration
    cfg = init_hydra_config(config_path="../lerobot/configs", config_name="train", overrides=config_overrides)

    # Start training
    print(f"Training policy: {args.policy}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    # Run training
    train_policy(cfg)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
