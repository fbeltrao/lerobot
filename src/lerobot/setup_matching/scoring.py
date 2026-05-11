#!/usr/bin/env python

from __future__ import annotations

import math
from dataclasses import dataclass

from .guidance import movement_guidance
from .models import Detection, ObjectScore, ReferenceState, SetupProfile


@dataclass(frozen=True)
class ScoreConfig:
    centroid_tolerance: float = 0.08
    ready_threshold: float = 0.85
    low_confidence_threshold: float = 0.65


def score_setup(
    profile: SetupProfile,
    reference_state: ReferenceState,
    detections: list[Detection],
    config: ScoreConfig | None = None,
) -> tuple[bool, float, list[ObjectScore], list[str]]:
    config = config or ScoreConfig()
    detections_by_label = {detection.label: detection for detection in detections}
    object_scores: list[ObjectScore] = []
    warnings: list[str] = []
    weighted_score = 0.0
    total_weight = 0.0

    for object_template in profile.objects:
        if object_template.camera_key != reference_state.camera_key:
            continue
        target_centroid = reference_state.target_centroids[object_template.label]
        detection = detections_by_label.get(object_template.label)
        total_weight += object_template.weight
        if detection is None:
            object_score = ObjectScore(
                label=object_template.label,
                required=object_template.required,
                present=False,
                score=0.0,
                centroid_error=None,
                guidance=f"{object_template.label}: not detected",
            )
            object_scores.append(object_score)
            if object_template.required:
                warnings.append(f"required object missing: {object_template.label}")
            continue

        centroid_error = math.dist(target_centroid, detection.centroid)
        score = max(0.0, 1.0 - centroid_error / config.centroid_tolerance)
        if detection.confidence < config.low_confidence_threshold:
            warnings.append(
                f"low detector confidence for {object_template.label}: {detection.confidence:.3f}"
            )
        weighted_score += score * object_template.weight
        object_scores.append(
            ObjectScore(
                label=object_template.label,
                required=object_template.required,
                present=True,
                score=score,
                centroid_error=centroid_error,
                guidance=movement_guidance(object_template.label, target_centroid, detection.centroid),
                confidence=detection.confidence,
            )
        )

    overall_score = weighted_score / total_weight if total_weight else 0.0
    required_scores = [object_score for object_score in object_scores if object_score.required]
    ready = bool(required_scores) and all(
        object_score.present and object_score.score >= config.ready_threshold
        for object_score in required_scores
    )
    return ready, overall_score, object_scores, warnings
