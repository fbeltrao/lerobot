#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import re
from glob import glob
from pathlib import Path
from types import TracebackType
from typing import Optional, Protocol, Type

import mlflow
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored

from lerobot.common.constants import PRETRAINED_MODEL_DIR
from lerobot.common.utils.logging_utils import MetricsTracker
from lerobot.configs.train import TrainPipelineConfig


def cfg_to_group(cfg: TrainPipelineConfig, return_list: bool = False) -> list[str] | str:
    """Return a group name for logging. Optionally returns group name as list."""
    lst = [
        f"policy:{cfg.policy.type}",
        f"dataset:{cfg.dataset.repo_id}",
        f"seed:{cfg.seed}",
    ]
    if cfg.env is not None:
        lst.append(f"env:{cfg.env.type}")
    return lst if return_list else "-".join(lst)


def get_wandb_run_id_from_filesystem(log_dir: Path) -> str:
    # Get the WandB run ID.
    paths = glob(str(log_dir / "wandb/latest-run/run-*"))
    if len(paths) != 1:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    match = re.search(r"run-([^\.]+).wandb", paths[0].split("/")[-1])
    if match is None:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    wandb_run_id = match.groups(0)[0]
    return wandb_run_id


def get_safe_wandb_artifact_name(name: str):
    """WandB artifacts don't accept ":" or "/" in their name."""
    return name.replace(":", "_").replace("/", "_")


class ExperimentLogger(Protocol):
    """A protocol for logging experiments."""

    def log_policy(self, checkpoint_dir: Path):
        """Checkpoints the policy to the logger."""
        ...

    def log_dict(self, d: dict, step: int, mode: str = "train"):
        """Logs a dictionary of values."""
        ...

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        """Logs a video."""
        ...

    def log_metrics(self, metrics: MetricsTracker):
        """Logs a MetricsTracker object."""
        ...

    def __enter__(self) -> "ExperimentLogger": ...

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> None: ...


class MLFlowLogger(ExperimentLogger):
    """A helper class to log object using MLFlow."""

    def __init__(self, cfg: TrainPipelineConfig):
        import mlflow

        self.cfg = cfg
        run_tags = {
            "policy": cfg.policy.type,
            "dataset": cfg.dataset.repo_id,
            "seed": str(cfg.seed) if cfg.seed else "not_set",
            "env": cfg.env.type if cfg.env else "none",
        }

        self._run = mlflow.start_run(
            run_name=cfg.job_name,
            # experiment_id=cfg.wandb.project or cfg.wandb.entity or "train_eval",
            tags=run_tags,
            nested=True,  # Allows nested runs
        )

        mlflow.log_params(cfg.to_dict())

    def log_policy(self, checkpoint_dir: Path):
        """Checkpoints the policy to MLFlow."""
        mlflow.log_artifact(local_path=checkpoint_dir / PRETRAINED_MODEL_DIR / SAFETENSORS_SINGLE_FILE)

    def log_dict(self, d: dict, step: int, mode: str = "train"):
        """Logs a dictionary of values to MLFlow."""
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        mlflow.log_dict(dict)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        """Logs a video to MLFlow."""

        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        mlflow.log_artifact(video_path)

    def log_metrics(self, metrics: MetricsTracker):
        """Logs a MetricsTracker object to MLFlow."""

        # Log each metric in the MetricsTracker.
        for key, value in metrics.to_dict().items():
            mlflow.log_metric(key, value, step=metrics.steps)

    def __enter__(self) -> "MLFlowLogger":
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> None:
        """Exit the context manager and end the MLFlow run."""
        if self._run:
            self._run.__exit__(__exc_type, __exc_value, __traceback)


class WandBLogger(ExperimentLogger):
    """A helper class to log object using wandb."""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.wandb
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        self._group = cfg_to_group(cfg)

        # Set up WandB.
        os.environ["WANDB_SILENT"] = "True"
        import wandb

        wandb_run_id = (
            cfg.wandb.run_id
            if cfg.wandb.run_id
            else get_wandb_run_id_from_filesystem(self.log_dir)
            if cfg.resume
            else None
        )
        self._run = wandb.init(
            id=wandb_run_id,
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=self.job_name,
            notes=self.cfg.notes,
            tags=cfg_to_group(cfg, return_list=True),
            dir=self.log_dir,
            config=cfg.to_dict(),
            # TODO(rcadene): try set to True
            save_code=False,
            # TODO(rcadene): split train and eval, and run async eval with job_type="eval"
            job_type="train_eval",
            resume="must" if cfg.resume else None,
            mode=self.cfg.mode if self.cfg.mode in ["online", "offline", "disabled"] else "online",
        )
        print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        logging.info(f"Track this run --> {colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}")
        self._wandb = wandb

    def log_policy(self, checkpoint_dir: Path):
        """Checkpoints the policy to wandb."""
        if self.cfg.disable_artifact:
            return

        step_id = checkpoint_dir.name
        artifact_name = f"{self._group}-{step_id}"
        artifact_name = get_safe_wandb_artifact_name(artifact_name)
        artifact = self._wandb.Artifact(artifact_name, type="model")
        artifact.add_file(checkpoint_dir / PRETRAINED_MODEL_DIR / SAFETENSORS_SINGLE_FILE)
        self._wandb.log_artifact(artifact)

    def log_dict(self, d: dict, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        for k, v in d.items():
            if not isinstance(v, (int, float, str)):
                logging.warning(
                    f'WandB logging of key "{k}" was ignored as its type is not handled by this wrapper.'
                )
                continue
            self._wandb.log({f"{mode}/{k}": v}, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        wandb_video = self._wandb.Video(video_path, fps=self.env_fps, format="mp4")
        self._wandb.log({f"{mode}/video": wandb_video}, step=step)

    def log_metrics(self, metrics: MetricsTracker):
        log_dict = metrics.to_dict()
        self.log_dict(log_dict, step=metrics.steps)

    def __enter__(self) -> "WandBLogger":
        return self

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> None:
        # Finish the WandB run.
        if self._run:
            self._run.__exit__(__exc_type, __exc_value, __traceback)
