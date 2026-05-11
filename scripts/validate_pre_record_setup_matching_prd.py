#!/usr/bin/env python3
"""Validate local fixtures for the pre-record setup matching PRD.

This does not validate the future application behavior. It validates that the
local datasets used for PRD/product validation are present and shaped like the
PoC expects, so a reviewer can start from one reproducible command.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_FAILED_DATASET = Path(
    "/home/frbeltra/.cache/huggingface/lerobot/fbeltrao/"
    "rollout_1_so101_green_lego_in_metal_holder_20260511_114146"
)
DEFAULT_MOCK_ROBOT_DATASET = Path(
    "/home/frbeltra/.cache/huggingface/lerobot/fbeltrao/so101_green_lego_in_metal_holder"
)
DEFAULT_CAMERA_KEYS = ("observation.images.side", "observation.images.wrist")


@dataclass(frozen=True)
class DatasetValidation:
    label: str
    path: Path
    ok: bool
    errors: list[str]
    warnings: list[str]
    summary: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _relative_files(root: Path, pattern: str) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.glob(pattern) if path.is_file())


def validate_dataset(label: str, path: Path, camera_keys: tuple[str, ...]) -> DatasetValidation:
    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {"path": str(path)}

    if not path.exists():
        return DatasetValidation(
            label, path, False, [f"dataset path does not exist: {path}"], warnings, summary
        )

    required_paths = [
        path / "meta" / "info.json",
        path / "meta" / "stats.json",
        path / "meta" / "tasks.parquet",
    ]
    missing_required = [
        str(required_path.relative_to(path)) for required_path in required_paths if not required_path.exists()
    ]
    if missing_required:
        errors.append(f"missing required metadata files: {', '.join(missing_required)}")

    info: dict[str, Any] = {}
    info_path = path / "meta" / "info.json"
    if info_path.exists():
        try:
            info = _read_json(info_path)
        except json.JSONDecodeError as exc:
            errors.append(f"meta/info.json is not valid JSON: {exc}")

    features = info.get("features", {}) if isinstance(info.get("features"), dict) else {}
    missing_features = [camera_key for camera_key in camera_keys if camera_key not in features]
    if missing_features:
        errors.append(f"missing expected camera features: {', '.join(missing_features)}")

    missing_video_dirs = [
        camera_key for camera_key in camera_keys if not (path / "videos" / camera_key).exists()
    ]
    if missing_video_dirs:
        errors.append(f"missing expected camera video directories: {', '.join(missing_video_dirs)}")

    data_files = _relative_files(path, "data/**/*.parquet")
    episode_files = _relative_files(path, "meta/episodes/**/*.parquet")
    video_files = _relative_files(path, "videos/**/*.mp4")
    if not data_files:
        errors.append("no data parquet files found under data/")
    if not episode_files:
        errors.append("no episode metadata parquet files found under meta/episodes/")
    if not video_files:
        errors.append("no video files found under videos/")

    total_episodes = info.get("total_episodes")
    total_frames = info.get("total_frames")
    fps = info.get("fps")
    robot_type = info.get("robot_type")
    if not isinstance(total_episodes, int) or total_episodes <= 0:
        errors.append(f"total_episodes must be a positive integer, got {total_episodes!r}")
    if not isinstance(total_frames, int) or total_frames <= 0:
        errors.append(f"total_frames must be a positive integer, got {total_frames!r}")
    if not isinstance(fps, int | float) or fps <= 0:
        errors.append(f"fps must be a positive number, got {fps!r}")
    if robot_type != "so_follower":
        warnings.append(f"expected robot_type 'so_follower' for this validation fixture, got {robot_type!r}")

    summary.update(
        {
            "codebase_version": info.get("codebase_version"),
            "robot_type": robot_type,
            "fps": fps,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": info.get("total_tasks"),
            "splits": info.get("splits"),
            "camera_keys": [camera_key for camera_key in camera_keys if camera_key in features],
            "data_files": len(data_files),
            "episode_metadata_files": len(episode_files),
            "video_files": len(video_files),
        }
    )
    return DatasetValidation(label, path, not errors, errors, warnings, summary)


def validate_pair(
    failed_dataset: Path, mock_robot_dataset: Path, camera_keys: tuple[str, ...]
) -> tuple[list[DatasetValidation], list[str]]:
    validations = [
        validate_dataset("failed_episodes_dataset", failed_dataset, camera_keys),
        validate_dataset("mock_robot_dataset", mock_robot_dataset, camera_keys),
    ]
    cross_dataset_warnings: list[str] = []

    failed_summary = validations[0].summary
    mock_summary = validations[1].summary
    for key in ("fps", "robot_type"):
        failed_value = failed_summary.get(key)
        mock_value = mock_summary.get(key)
        if failed_value is not None and mock_value is not None and failed_value != mock_value:
            cross_dataset_warnings.append(f"{key} differs: failed={failed_value!r}, mock={mock_value!r}")

    failed_cameras = set(failed_summary.get("camera_keys", []))
    mock_cameras = set(mock_summary.get("camera_keys", []))
    if failed_cameras and mock_cameras and failed_cameras != mock_cameras:
        cross_dataset_warnings.append(
            f"camera keys differ: failed={sorted(failed_cameras)}, mock={sorted(mock_cameras)}"
        )

    return validations, cross_dataset_warnings


def build_validation_context(
    validations: list[DatasetValidation], camera_keys: tuple[str, ...], cross_dataset_warnings: list[str]
) -> dict[str, Any]:
    failed_summary = validations[0].summary
    mock_summary = validations[1].summary
    return {
        "ok": all(validation.ok for validation in validations),
        "default_reference_episode": 0,
        "default_reference_frame": 0,
        "default_replay_episode": 0,
        "default_camera_key": camera_keys[0] if camera_keys else None,
        "expected_camera_keys": list(camera_keys),
        "failed_episodes_dataset": failed_summary,
        "mock_robot_dataset": mock_summary,
        "warnings": [warning for validation in validations for warning in validation.warnings]
        + cross_dataset_warnings,
        "errors": [error for validation in validations for error in validation.errors],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-dataset", type=Path, default=DEFAULT_FAILED_DATASET)
    parser.add_argument("--mock-robot-dataset", type=Path, default=DEFAULT_MOCK_ROBOT_DATASET)
    parser.add_argument(
        "--camera-key",
        action="append",
        dest="camera_keys",
        help="Expected camera key. Repeat to validate multiple cameras.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable validation context.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    camera_keys = tuple(args.camera_keys or DEFAULT_CAMERA_KEYS)
    validations, cross_dataset_warnings = validate_pair(
        args.failed_dataset, args.mock_robot_dataset, camera_keys
    )
    context = build_validation_context(validations, camera_keys, cross_dataset_warnings)

    if args.json:
        print(json.dumps(context, indent=2, sort_keys=True))
    else:
        print("Pre-record setup matching PRD validation fixtures")
        print(f"status: {'ok' if context['ok'] else 'failed'}")
        print(f"default camera: {context['default_camera_key']}")
        print("default episodes: reference=0, replay=0, frame=0")
        for validation in validations:
            summary = validation.summary
            print(
                f"- {validation.label}: episodes={summary.get('total_episodes')}, "
                f"frames={summary.get('total_frames')}, fps={summary.get('fps')}, "
                f"cameras={', '.join(summary.get('camera_keys', []))}"
            )
        for warning in context["warnings"]:
            print(f"warning: {warning}")
        for error in context["errors"]:
            print(f"error: {error}")

    return 0 if context["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
