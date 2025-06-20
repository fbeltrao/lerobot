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

"""This script demonstrates how to slice a dataset and calculate the loss on a subset of the data.

This technique can be useful for debugging and testing purposes, as well as identifying whether a policy
is learning effectively.

Furthermore, relying on validation loss to evaluate performance is generally not considered a good practice,
especially in the context of imitation learning. The most reliable approach is to evaluate the policy directly
on the target environment, whether that be in simulation or the real world.
"""

import math

import torch

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig


def main():
    device = torch.device("cuda")

    # Set up the dataset.
    pretrained_policy_path = "fbeltrao/pi0fast_so101_multi_task_v2"
    policy = PI0FASTPolicy.from_pretrained(pretrained_policy_path, revision="v10000steps")
    policy.eval()
    policy.to(device)

    # delta_timestamps = {
    #     # Load the previous image and state at -0.1 seconds before current frame,
    #     # then load current image and state corresponding to 0.0 second.
    #     "observation.image": [-0.1, 0.0],
    #     "observation.state": [-0.1, 0.0],
    #     # Load the previous action (-0.1), the next action to be executed (0.0),
    #     # and 14 future actions with a 0.1 seconds spacing. All these actions will be
    #     # used to calculate the loss.
    #     "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    # }

    # Load the last 10% of episodes of the dataset as a validation set.
    # - Load dataset metadata
    dataset_name_or_path = "fbeltrao/so101_multi_task_v2"
    dataset_metadata = LeRobotDatasetMetadata(dataset_name_or_path)
    # - Calculate train and val episodes
    total_episodes = dataset_metadata.total_episodes
    episodes = list(range(dataset_metadata.total_episodes))
    num_train_episodes = math.floor(total_episodes * 90 / 100)
    train_episodes = episodes[:num_train_episodes]
    val_episodes = episodes[num_train_episodes:]
    print(f"Number of episodes in full dataset: {total_episodes}")
    # print(f"Number of episodes in training dataset (90% subset): {len(train_episodes)}")
    # print(f"Number of episodes in validation dataset (10% subset): {len(val_episodes)}")
    # - Load train and val datasets
    # train_dataset = LeRobotDataset(
    #     dataset_name_or_path, episodes=train_episodes, delta_timestamps=delta_timestamps
    # )
    # val_dataset = LeRobotDataset(dataset_name_or_path, episodes=val_episodes, delta_timestamps=delta_timestamps)
    val_dataset = make_dataset(
        TrainPipelineConfig(
            dataset=DatasetConfig(repo_id=dataset_name_or_path),
            policy=policy.config,
        )
    )
    # print(f"Number of frames in training dataset (90% subset): {len(train_dataset)}")
    # print(f"Number of frames in validation dataset (10% subset): {len(val_dataset)}")

    # Create dataloader for evaluation.
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=2,
        batch_size=2,
        shuffle=False,
        pin_memory=device != torch.device("cpu"),
        drop_last=False,
    )



    # Run validation loop.
    loss_cumsum = 0
    n_examples_evaluated = 0
    for batch in val_dataloader:
        batch_items = batch.items()
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_items}
        loss, _ = policy.forward(batch)

        loss_cumsum += loss.item()
        n_examples_evaluated += batch["index"].shape[0]

        print(f"Current average loss on validation set after {n_examples_evaluated} examples: {loss_cumsum / n_examples_evaluated:.4f}")

    # Calculate the average loss over the validation set.
    average_loss = loss_cumsum / n_examples_evaluated

    print(f"Final average loss on validation set: {average_loss:.4f}")


if __name__ == "__main__":
    main()
