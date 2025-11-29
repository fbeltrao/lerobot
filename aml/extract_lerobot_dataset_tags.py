# /// script
# dependencies = [
#   "lerobot==0.4.2",
# ]
# ///


from collections.abc import Iterable
import json
from pathlib import Path
from typing import TypedDict

import argparse
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import load_episodes
from lerobot.datasets.backward_compatibility import BackwardCompatibilityError

REPO_ID = "foo/bar"

class LeRobotDatasetTags(TypedDict):
    codebase_version: str
    features: list[str]
    fps: int
    total_episodes: int
    total_frames: int
    total_tasks: int
    robot_type: str
    tasks: list[str] | None
    camera_keys: list[str] | None

def extract_from_legacy_dataset(root: Path) -> LeRobotDatasetTags:

    meta_path = root / "meta"
    info_json_file = meta_path / "info.json"
    if not info_json_file.exists():
        raise FileNotFoundError(f"info.json not found in {root}")
    
    tasks_file = meta_path / "tasks.jsonl"
    if not tasks_file.exists():
        raise FileNotFoundError(f"tasks.jsonl not found in {root}")
    
    with open(info_json_file, "r") as f:
        info = json.load(f)
    with open(tasks_file, "r") as f:
        tasks = [json.loads(line) for line in f]
    return {
        "codebase_version": info["codebase_version"],
        "features": list_to_tag_value(info["features"].keys()),
        "fps": info["fps"],
        "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"],
        "total_tasks": len(tasks),
        "tasks": list_to_tag_value(item["task"] for item in tasks),
        "robot_type": info["robot_type"],
        "camera_keys": list_to_tag_value(k for k, v in info["features"].items() if v.get("dtype") == "video"),
    }

def list_to_tag_value(value: Iterable[str]) -> str:
    return ",".join(value)

def extract_from_dataset_version_3_1(root: Path) -> LeRobotDatasetTags:
    dataset_metadata = LeRobotDatasetMetadata(REPO_ID, root)
    return {
        "codebase_version": str(dataset_metadata._version),
        "features": list_to_tag_value(dataset_metadata.features.keys()),
        "fps": dataset_metadata.fps,
        "total_episodes": dataset_metadata.total_episodes,
        "total_frames": dataset_metadata.total_frames,
        "total_tasks": dataset_metadata.total_tasks,
        "tasks": list_to_tag_value(set(task for episode_tasks in dataset_metadata.episodes["tasks"] for task in episode_tasks)),
        "robot_type": dataset_metadata.robot_type,
        "camera_keys": list_to_tag_value(dataset_metadata.camera_keys),
    }

if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description="Extracts datasets tags from a local lerobot dataset."
    )

    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="The root directory of the dataset to convert.",
    )
    
    args = parser.parse_args()
    root = Path(args.root)

    try:
        dataset_tags = extract_from_dataset_version_3_1(root)
    except BackwardCompatibilityError as e:
        dataset_tags = extract_from_legacy_dataset(root)

    print(json.dumps(dataset_tags))