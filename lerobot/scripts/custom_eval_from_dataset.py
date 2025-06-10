import datetime
import json
import random
from pathlib import Path
from typing import Iterable, TypedDict

import numpy as np
import pandas as pd
import torch
from PIL import Image

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.control_utils import predict_action
from lerobot.common.utils.utils import get_safe_torch_device


def save_image(image_path: Path, image_tensor: torch.tensor):
    """Saves a tensor image to a file."""
    # Convert from float32 (0-1) back to uint8 (0-255)
    # Tensor shape is (3, height, width), need to convert to (height, width, 3)
    image_numpy = (image_tensor * 255).clamp(0, 255).byte()
    image_numpy = image_numpy.permute(1, 2, 0).cpu().numpy()

    # Create PIL Image and save
    image = Image.fromarray(image_numpy, mode="RGB")
    image.save(image_path)


def extract_observations(
    dataset_name: str,
    output_path: Path,
    number_of_frames: int = 20,
    frames_to_extract: list[int] | None = None,
):
    dataset = LeRobotDataset(dataset_name)
    frames_to_extract = frames_to_extract or [
        random.randint(0, dataset.num_frames - 1) for _ in range(number_of_frames)
    ]

    output_path.mkdir(parents=True, exist_ok=True)

    for frame in frames_to_extract:
        print(f"Extracting frame {frame} from dataset {dataset_name}")
        raw_observation = dataset[frame]
        index = raw_observation["index"].numpy().tolist()
        observation = {
            "index": index,
            "episode_index": raw_observation["episode_index"].numpy().tolist(),
            "task": raw_observation["task"],
            "observation.state": raw_observation["observation.state"].numpy().tolist(),
            "action": raw_observation["action"].numpy().tolist(),
            "images": {},
        }
        for k, v in raw_observation.items():
            if "image" in k:
                image_path = output_path / f"{index}_{k}.png"
                save_image(image_path, v)
                observation["images"][k] = str(image_path)

        observation_file = output_path / f"{index}_observation.json"
        with open(observation_file, "w") as f:
            # Save the observation as a JSON file
            json.dump(
                observation,
                f,
                indent=4,
            )

        print(f"Observation at frame {frame}: {observation}")


class PredictionRecord(TypedDict):
    """A record of a prediction made by the policy."""

    episode_index: int
    index: int
    action: list[float]
    expected_action: list[float]
    mse_loss: float
    prediction_duration_ms: float


def evaluate_observations_on_policy(observations: Iterable[dict], policy: PreTrainedPolicy):
    """Evaluate a list of observations on a given policy."""

    prediction_records: list[PredictionRecord] = []

    torch_device = get_safe_torch_device(policy.config.device)
    for observation in observations:
        obs = observation.copy()
        task = obs.pop("task", None)
        obs.pop("action", None)
        obs.pop("index", None)
        obs.pop("episode_index", None)
        obs.pop("images", None)

        # clear cached actions
        if "_action_queue" in policy.__dict__:
            policy._action_queue.clear()

        prediction_start = datetime.datetime.now()
        action = predict_action(
            obs,
            policy,
            torch_device,
            policy.config.use_amp,
            task=task,
        )
        prediction_duration = datetime.datetime.now() - prediction_start
        expected: list[float] = observation["action"]
        predicted: list[float] = action.cpu().numpy().tolist()

        # Calculate MSE loss
        mse_loss = torch.nn.functional.mse_loss(
            torch.tensor(predicted, dtype=torch.float32), torch.tensor(expected, dtype=torch.float32)
        ).item()

        prediction_records.append(
            {
                "episode_index": observation["episode_index"],
                "index": observation["index"],
                "action": predicted,
                "expected_action": expected,
                "mse_loss": mse_loss,
                "prediction_duration_ms": prediction_duration.total_seconds()
                * 1000,  # Convert to milliseconds
            }
        )

    # Save the prediction records to a CSV file
    df_results = pd.DataFrame(prediction_records)
    output_folder = Path("outputs") / "model_evaluation" / "pi0fast"
    output_folder.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(
        output_folder / f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_predictions.csv",
        index=False,
    )

    print(df_results)


def load_observation_image(image_path: str) -> torch.Tensor:
    pil_img = Image.open(image_path).convert("RGB")
    return torch.from_numpy(np.array(pil_img))


def load_observations(
    output_path: Path,
    frames: list[int] | None = None,
) -> Iterable[dict]:
    """Load observations from the output path."""

    for observation_file in output_path.glob("*.json"):
        if frames is not None:
            # If specific frames are requested, check if the file matches
            index = int(observation_file.stem.split("_", maxsplit=1)[0])
            if index not in frames:
                continue

        with open(observation_file, "r") as f:
            observation = json.load(f)
            # Convert state to tensor
            observation["observation.state"] = torch.tensor(observation["observation.state"]).type(
                torch.float32
            )
            # Convert images paths to tensors
            for k, v in observation["images"].items():
                observation[k] = load_observation_image(v)

            yield observation


def main():
    device = torch.device("cuda")

    # Load the last 10% of episodes of the dataset as a validation set.
    # - Load dataset metadata
    dataset_name = "fbeltrao/so101_unplug_cable_4"
    output_path = Path("outputs") / "extracted_observations" / dataset_name

    frames: list[int] | None = None
    # frames = [8435]
    extract_observations(dataset_name, output_path=output_path, frames_to_extract=frames)

    if not output_path.exists():
        extract_observations(dataset_name, output_path=output_path, number_of_frames=20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_name_or_path = "fbeltrao/so101_unplug_cable_4"
    revision = "steps_10_000"

    policy = PI0FASTPolicy.from_pretrained(pretrained_name_or_path, revision=revision)
    policy.eval()
    policy.to(device)

    evaluate_observations_on_policy(
        load_observations(
            output_path,
            frames,
        ),
        policy,
    )


if __name__ == "__main__":
    main()
