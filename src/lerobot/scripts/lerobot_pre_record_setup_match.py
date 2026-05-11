#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.setup_matching import ArtifactStore, BoundingBox, SetupMatchScenario, run_setup_match_scenario

DEFAULT_FAILED_DATASET = Path(
    "/home/frbeltra/.cache/huggingface/lerobot/fbeltrao/"
    "rollout_1_so101_green_lego_in_metal_holder_20260511_114146"
)
DEFAULT_MOCK_ROBOT_DATASET = Path(
    "/home/frbeltra/.cache/huggingface/lerobot/fbeltrao/so101_green_lego_in_metal_holder"
)
DEFAULT_CAMERA_KEY = "observation.images.side"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a pre-record setup matching PoC scenario.")
    parser.add_argument("--reference-dataset", type=Path, default=DEFAULT_FAILED_DATASET)
    parser.add_argument("--replay-dataset", type=Path, default=DEFAULT_MOCK_ROBOT_DATASET)
    parser.add_argument("--camera-key", default=DEFAULT_CAMERA_KEY)
    parser.add_argument("--reference-episode", type=int, default=0)
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--replay-episode", type=int, default=0)
    parser.add_argument("--replay-start-frame", type=int, default=0)
    parser.add_argument("--stream-frames", type=int, default=1)
    parser.add_argument("--artifact-root", type=Path, default=Path("outputs/pre_record_setup_matching"))
    parser.add_argument(
        "--object-roi",
        action="append",
        default=[],
        metavar="LABEL:X_MIN,Y_MIN,X_MAX,Y_MAX",
        help="Normalized ROI for an object. Repeat for multiple objects.",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Run without writing profile/reference/result artifacts."
    )
    parser.add_argument("--json", action="store_true", help="Print the full match result as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenario = SetupMatchScenario(
        reference_dataset_path=args.reference_dataset,
        replay_dataset_path=args.replay_dataset,
        camera_key=args.camera_key,
        reference_episode_index=args.reference_episode,
        reference_frame_index=args.reference_frame,
        replay_episode_index=args.replay_episode,
        replay_start_frame_index=args.replay_start_frame,
        stream_frames=args.stream_frames,
        object_rois=_parse_object_rois(args.object_roi),
    )
    artifact_store = None if args.no_save else ArtifactStore(args.artifact_root)
    match_result = run_setup_match_scenario(scenario, artifact_store=artifact_store)
    result = match_result.to_dict()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("Pre-record setup matching")
        print(f"status: {'ready' if match_result.ready else 'not ready'}")
        print(f"overall_score: {match_result.overall_score:.3f}")
        print(f"camera: {match_result.camera_key}")
        print(
            "lineage: "
            f"reference_episode={match_result.reference_episode_index}, "
            f"reference_frame={match_result.reference_frame_index}, "
            f"replay_episode={match_result.replay_episode_index}"
        )
        for object_score in match_result.object_scores:
            print(
                f"- {object_score.label}: score={object_score.score:.3f}, "
                f"present={object_score.present}, guidance={object_score.guidance}"
            )
        for warning in match_result.warnings:
            print(f"warning: {warning}")
        if not args.no_save:
            print(f"artifacts: {args.artifact_root}")
    return 0 if match_result.ready else 2


def _parse_object_rois(values: list[str]) -> dict[str, BoundingBox] | None:
    if not values:
        return None
    object_rois: dict[str, BoundingBox] = {}
    for value in values:
        try:
            label, raw_bbox = value.split(":", 1)
            bbox_values = [float(item) for item in raw_bbox.split(",")]
        except ValueError as exc:
            raise ValueError(f"Invalid --object-roi value {value!r}") from exc
        if len(bbox_values) != 4:
            raise ValueError(f"Invalid --object-roi value {value!r}: expected four bbox values")
        object_rois[label] = BoundingBox(*bbox_values)
    return object_rois


if __name__ == "__main__":
    raise SystemExit(main())
