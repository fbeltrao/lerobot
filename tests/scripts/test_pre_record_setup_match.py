#!/usr/bin/env python

from __future__ import annotations

import cv2
import numpy as np

from lerobot.setup_matching import ArtifactStore, BoundingBox, SetupMatchScenario, run_setup_match_scenario
from lerobot.setup_matching.artifacts import EpisodeReviewStore
from lerobot.setup_matching.detectors import TemplateMatchingDetector, create_object_template
from lerobot.setup_matching.frame_sources import StaticImageFrameSource
from lerobot.setup_matching.models import ReferenceState, SetupProfile
from lerobot.setup_matching.scoring import score_setup
from lerobot.setup_matching.web_app import (
    _read_urdf_joint_metadata,
    _refine_bbox_with_color_blob,
    _scenario_from_payload,
)


def test_template_detector_scores_exact_match_ready() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[30:45, 40:58] = (0, 255, 0)
    bbox = BoundingBox(0.40, 0.375, 0.58, 0.5625)
    profile = SetupProfile(
        "profile", [create_object_template(frame, "green_block", "cam", bbox)]
    )
    reference_state = ReferenceState(
        "reference", "profile", "dataset", 0, 0, "cam", {"green_block": bbox.centroid}
    )

    detections = TemplateMatchingDetector(confidence_threshold=0.1).detect(frame, profile, "cam")
    ready, overall_score, object_scores, warnings = score_setup(profile, reference_state, detections)

    assert ready
    assert overall_score == 1.0
    assert object_scores[0].guidance == "green_block: hold position"
    assert warnings == []


def test_scoring_missing_required_object_is_not_ready() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    bbox = BoundingBox(0.20, 0.20, 0.40, 0.40)
    profile = SetupProfile("profile", [create_object_template(frame, "block", "cam", bbox)])
    reference_state = ReferenceState("reference", "profile", "dataset", 0, 0, "cam", {"block": bbox.centroid})

    ready, overall_score, object_scores, warnings = score_setup(profile, reference_state, [])

    assert not ready
    assert overall_score == 0.0
    assert object_scores[0].guidance == "block: not detected"
    assert warnings == ["required object missing: block"]


def test_episode_review_store_roundtrips_labels_and_notes(tmp_path) -> None:
    store = EpisodeReviewStore(tmp_path / "reviews.json")

    store.set_review("dataset", 3, "failed", failure_category="missed_pick", note="Object slipped")

    assert store.get_review("dataset", 3) == {
        "status": "failed",
        "failure_category": "missed_pick",
        "note": "Object slipped",
    }
    assert store.get_review("dataset", 4) == {"status": "unreviewed"}


def test_static_image_source_returns_copy() -> None:
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    source = StaticImageFrameSource(frame)

    read_frame = source.read_frame(12)
    read_frame[0, 0] = (255, 255, 255)

    assert np.array_equal(frame, np.zeros((4, 5, 3), dtype=np.uint8))


def test_setup_match_scenario_saves_artifacts_with_synthetic_video_dataset(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    video_dir = dataset / "videos" / "observation.images.side" / "chunk-000"
    video_dir.mkdir(parents=True)
    (dataset / "meta").mkdir()
    (dataset / "meta" / "info.json").write_text(
        """
{
  "features": {"observation.images.side": {"dtype": "video", "shape": [64, 64, 3]}},
  "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
}
""".strip(),
        encoding="utf-8",
    )
    video_path = video_dir / "file-000.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 64))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[20:34, 25:40] = (0, 255, 0)
    writer.write(frame)
    writer.write(frame)
    writer.release()
    artifact_store = ArtifactStore(tmp_path / "artifacts")

    result = run_setup_match_scenario(
        SetupMatchScenario(
            reference_dataset_path=dataset,
            replay_dataset_path=dataset,
            camera_key="observation.images.side",
            object_rois={"green_block": BoundingBox(0.39, 0.31, 0.63, 0.53)},
            stream_frames=2,
        ),
        artifact_store=artifact_store,
    )

    assert result.ready
    assert result.overall_score >= 0.95
    assert len(result.frame_results) == 2
    assert (tmp_path / "artifacts" / "profiles" / "default-setup-profile.json").exists()
    assert list((tmp_path / "artifacts" / "reference_states").glob("*.json"))
    assert list((tmp_path / "artifacts" / "match_results").glob("*.json"))


def test_web_payload_builds_scenario_with_multiple_objects() -> None:
    scenario = _scenario_from_payload(
        {
            "reference_dataset_path": "failed-dataset",
            "replay_dataset_path": "mock-dataset",
            "camera_key": "observation.images.side",
            "reference_episode_index": 2,
            "reference_frame_index": 4,
            "replay_episode_index": 6,
            "stream_frames": 3,
            "objects": [
                {"label": "green_block", "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}},
                {"label": "holder", "bbox": {"x_min": 0.5, "y_min": 0.2, "x_max": 0.7, "y_max": 0.5}},
            ],
        }
    )

    assert scenario.reference_dataset_path.name == "failed-dataset"
    assert scenario.replay_dataset_path.name == "mock-dataset"
    assert scenario.camera_key == "observation.images.side"
    assert scenario.reference_episode_index == 2
    assert scenario.reference_frame_index == 4
    assert scenario.replay_episode_index == 6
    assert scenario.stream_frames == 3
    assert scenario.object_rois == {
        "green_block": BoundingBox(0.1, 0.2, 0.3, 0.4),
        "holder": BoundingBox(0.5, 0.2, 0.7, 0.5),
    }


def test_web_roi_refinement_tightens_saturated_object_crop() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[42:58, 45:63] = (0, 255, 0)
    rough_bbox = BoundingBox(0.35, 0.30, 0.75, 0.70)

    refined_bbox = _refine_bbox_with_color_blob(frame, rough_bbox)

    assert refined_bbox.x_min > rough_bbox.x_min
    assert refined_bbox.y_min > rough_bbox.y_min
    assert refined_bbox.x_max < rough_bbox.x_max
    assert refined_bbox.y_max < rough_bbox.y_max
    assert refined_bbox.x_min <= 0.45
    assert refined_bbox.x_max >= 0.63


def test_web_reads_urdf_joint_limits(tmp_path) -> None:
        urdf_path = tmp_path / "so101.urdf"
        urdf_path.write_text(
                """
<robot name="so101">
    <joint name="shoulder_pan" type="revolute">
        <limit lower="-1.57" upper="1.57" />
    </joint>
    <joint name="fixed_mount" type="fixed" />
    <joint name="gripper" type="prismatic">
        <limit lower="0" upper="1" />
    </joint>
</robot>
""".strip(),
                encoding="utf-8",
        )

        joints = _read_urdf_joint_metadata(urdf_path)

        assert joints == {
                "shoulder_pan": {"type": "revolute", "lower": -1.57, "upper": 1.57},
                "gripper": {"type": "prismatic", "lower": 0.0, "upper": 1.0},
        }
