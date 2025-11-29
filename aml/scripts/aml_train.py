"""
AML Training Wrapper with MLflow logging

This script serves as a wrapper around lerobot-train that replaces WandBLogger
with MLflowLogger for logging in Azure ML environments. All arguments are passed
through to the original training script.
"""

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import torch
from termcolor import colored

from lerobot.configs.train import TrainPipelineConfig


class MLflowLogger:
    """A drop-in replacement for WandBLogger that uses MLflow instead of wandb."""

    def __init__(self, cfg: TrainPipelineConfig):
        # Set up MLflow
        self.cfg = cfg
        self._setup_mlflow(cfg)

    def _setup_mlflow(self, cfg: TrainPipelineConfig):
        """Initialize MLflow run."""

        if not os.getenv("MLFLOW_EXPERIMENT_NAME") and cfg.wandb.project:
            mlflow.set_experiment(cfg.wandb.project)

        # Start MLflow run
        mlflow.start_run()

        # Log configuration as parameters
        mlflow.log_params(self._flatten_config(cfg.to_dict()))

        # Log tags

        tags = {"seed": cfg.seed}
        if cfg.policy:
            tags["policy"] = cfg.policy.type
        if cfg.dataset:
            tags["dataset"] = cfg.dataset.repo_id
        if cfg.env:
            tags["env"] = cfg.env.type
        mlflow.set_tags(tags)

    def _flatten_config(self, config_dict: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested configuration dictionary for MLflow params."""
        flat_dict = {}
        for key, value in config_dict.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat_dict.update(self._flatten_config(value, new_key))
            elif isinstance(value, (list, tuple)):
                flat_dict[new_key] = str(value)
            elif value is not None:
                flat_dict[new_key] = value
        return flat_dict

    def log_policy(self, checkpoint_dir: Path):
        """Log policy checkpoint to MLflow."""
        if self.cfg.wandb.disable_artifact:
            return

        try:
            # Log the entire checkpoint directory
            mlflow.log_artifacts(str(checkpoint_dir), artifact_path="checkpoints")

            # Also log the model specifically if it exists
            model_path = checkpoint_dir / "pretrained_model" / "model.safetensors"
            if model_path.exists():
                mlflow.log_artifact(str(model_path), artifact_path="models")
        except Exception as e:
            logging.warning(f"Failed to log policy checkpoint to MLflow: {e}")

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ):
        """Log dictionary of metrics to MLflow."""
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        # Filter out non-numeric values and convert tensors
        metrics_to_log = {}
        for key, value in d.items():
            if isinstance(value, (int, float)):
                metrics_to_log[f"{mode}/{key}"] = value
            elif isinstance(value, torch.Tensor):
                # Handle PyTorch tensors by converting to scalar
                try:
                    if value.numel() == 1:
                        metrics_to_log[f"{mode}/{key}"] = value.item()
                    else:
                        logging.warning(
                            f'MLflow logging of key "{key}" was ignored as tensor has {value.numel()} elements (expected 1).'
                        )
                except (RuntimeError, ValueError) as e:
                    logging.warning(
                        f'MLflow logging of key "{key}" was ignored due to tensor conversion error: {e}'
                    )
            elif isinstance(value, str):
                # Log string values as tags instead of metrics
                mlflow.set_tag(f"{mode}/{key}", value)
            else:
                logging.warning(
                    f'MLflow logging of key "{key}" was ignored as its type "{type(value)}" is not supported.'
                )

        if metrics_to_log:
            mlflow.log_metrics(metrics_to_log, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        """Log video to MLflow as artifact."""
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        if os.path.exists(video_path):
            artifact_path = f"{mode}_videos/step_{step}"
            mlflow.log_artifact(video_path, artifact_path)
            logging.info(f"Logged video to MLflow: {artifact_path}")


def setup_mlflow_logger():
    """Replace WandBLogger with MLflowLogger."""
    # Import here to avoid circular imports
    import lerobot.rl.wandb_utils

    # Replace the WandBLogger class with MLflowLogger
    lerobot.rl.wandb_utils.WandBLogger = MLflowLogger

    logging.info("WandBLogger replaced with MLflowLogger for MLflow logging")


def main():
    """Main entry point that sets up MLflow logging and calls lerobot-train."""

    setup_mlflow_logger()

    # Import and run the original training script
    try:
        from lerobot.scripts.lerobot_train import main as lerobot_train_main

        # Call the original training function
        lerobot_train_main()

    finally:
        # Terminate MLflow run if one exists
        try:
            if mlflow.active_run() is not None:
                mlflow.end_run()
        except Exception as e:
            logging.warning(f"Failed to terminate MLflow run: {e}")


if __name__ == "__main__":
    main()
