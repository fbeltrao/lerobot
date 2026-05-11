#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .artifacts import ArtifactStore
from .detectors import TemplateMatchingDetector, create_object_template
from .frame_sources import EpisodeReplayFrameSource
from .models import BoundingBox, MatchResult, ReferenceState, SetupProfile
from .scoring import ScoreConfig, score_setup


@dataclass(frozen=True)
class SetupMatchScenario:
    reference_dataset_path: Path
    replay_dataset_path: Path
    camera_key: str
    reference_episode_index: int = 0
    reference_frame_index: int = 0
    replay_episode_index: int = 0
    replay_start_frame_index: int = 0
    object_rois: dict[str, BoundingBox] | None = None
    profile_id: str = "default-setup-profile"
    stream_frames: int = 1


def run_setup_match_scenario(
    scenario: SetupMatchScenario,
    artifact_store: ArtifactStore | None = None,
    detector: TemplateMatchingDetector | None = None,
    score_config: ScoreConfig | None = None,
) -> MatchResult:
    detector = detector or TemplateMatchingDetector()
    object_rois = scenario.object_rois or {"reference_crop": BoundingBox(0.35, 0.35, 0.65, 0.65)}

    reference_source = EpisodeReplayFrameSource(
        dataset_path=scenario.reference_dataset_path,
        camera_key=scenario.camera_key,
        episode_index=scenario.reference_episode_index,
    )
    replay_source = EpisodeReplayFrameSource(
        dataset_path=scenario.replay_dataset_path,
        camera_key=scenario.camera_key,
        episode_index=scenario.replay_episode_index,
    )
    reference_frame = reference_source.read_frame(scenario.reference_frame_index)
    profile = SetupProfile(
        profile_id=scenario.profile_id,
        objects=[
            create_object_template(reference_frame, label, scenario.camera_key, bbox)
            for label, bbox in object_rois.items()
        ],
    )
    reference_state = ReferenceState(
        reference_id=f"reference-{uuid4().hex[:12]}",
        profile_id=profile.profile_id,
        dataset_path=str(scenario.reference_dataset_path),
        episode_index=scenario.reference_episode_index,
        frame_index=scenario.reference_frame_index,
        camera_key=scenario.camera_key,
        target_centroids={label: bbox.centroid for label, bbox in object_rois.items()},
    )

    frame_results: list[dict] = []
    final_ready = False
    final_score = 0.0
    final_object_scores = []
    final_warnings: list[str] = []
    stream_frames = max(1, scenario.stream_frames)
    for stream_frame_index in range(stream_frames):
        replay_frame_index = scenario.replay_start_frame_index + stream_frame_index
        replay_frame = replay_source.read_frame(replay_frame_index)
        detections = detector.detect(replay_frame, profile, scenario.camera_key)
        ready, overall_score, object_scores, warnings = score_setup(
            profile, reference_state, detections, score_config
        )
        warnings = [*warnings, *_camera_drift_warnings(reference_frame, replay_frame)]
        frame_results.append(
            {
                "frame_index": replay_frame_index,
                "ready": ready,
                "overall_score": overall_score,
                "detections": [detection.to_dict() for detection in detections],
                "object_scores": [object_score.to_dict() for object_score in object_scores],
                "warnings": warnings,
            }
        )
        final_ready = ready
        final_score = overall_score
        final_object_scores = object_scores
        final_warnings = warnings

    match_result = MatchResult(
        match_id=f"match-{uuid4().hex[:12]}",
        profile_id=profile.profile_id,
        reference_id=reference_state.reference_id,
        reference_dataset_path=str(scenario.reference_dataset_path),
        reference_episode_index=scenario.reference_episode_index,
        reference_frame_index=scenario.reference_frame_index,
        replay_dataset_path=str(scenario.replay_dataset_path),
        replay_episode_index=scenario.replay_episode_index,
        camera_key=scenario.camera_key,
        ready=final_ready,
        overall_score=final_score,
        object_scores=final_object_scores,
        warnings=final_warnings,
        frame_results=frame_results,
    )
    if artifact_store is not None:
        artifact_store.save_profile(profile)
        artifact_store.save_reference_state(reference_state)
        artifact_store.save_match_result(match_result)
    return match_result


def _camera_drift_warnings(reference_frame: np.ndarray, replay_frame: np.ndarray) -> list[str]:
    reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    replay_gray = cv2.cvtColor(replay_frame, cv2.COLOR_BGR2GRAY)
    if reference_gray.shape != replay_gray.shape:
        return ["camera view differs: frame shapes do not match"]
    mean_abs_diff = float(np.mean(cv2.absdiff(reference_gray, replay_gray))) / 255.0
    if mean_abs_diff > 0.20:
        return [f"camera view drift warning: mean image difference {mean_abs_diff:.3f}"]
    return []
