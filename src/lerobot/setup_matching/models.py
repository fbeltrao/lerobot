#!/usr/bin/env python

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

ReviewStatus = Literal["failed", "good", "skipped", "unreviewed"]


@dataclass(frozen=True)
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(f"BoundingBox values must be normalized to [0, 1], got {values}")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError(f"BoundingBox minimums must be smaller than maximums, got {values}")

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def to_pixel_slice(self, height: int, width: int) -> tuple[slice, slice]:
        row_start = max(0, min(height - 1, round(self.y_min * height)))
        row_stop = max(row_start + 1, min(height, round(self.y_max * height)))
        col_start = max(0, min(width - 1, round(self.x_min * width)))
        col_stop = max(col_start + 1, min(width, round(self.x_max * width)))
        return slice(row_start, row_stop), slice(col_start, col_stop)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoundingBox:
        return cls(
            x_min=float(data["x_min"]),
            y_min=float(data["y_min"]),
            x_max=float(data["x_max"]),
            y_max=float(data["y_max"]),
        )


@dataclass(frozen=True)
class ObjectTemplate:
    label: str
    camera_key: str
    reference_bbox: BoundingBox
    template_bgr: np.ndarray
    required: bool = True
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "camera_key": self.camera_key,
            "reference_bbox": asdict(self.reference_bbox),
            "template_bgr_base64": base64.b64encode(self.template_bgr.tobytes()).decode("ascii"),
            "template_shape": list(self.template_bgr.shape),
            "template_dtype": str(self.template_bgr.dtype),
            "required": self.required,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectTemplate:
        shape = tuple(int(value) for value in data["template_shape"])
        dtype = np.dtype(data["template_dtype"])
        template = np.frombuffer(base64.b64decode(data["template_bgr_base64"]), dtype=dtype).reshape(shape)
        return cls(
            label=str(data["label"]),
            camera_key=str(data["camera_key"]),
            reference_bbox=BoundingBox.from_dict(data["reference_bbox"]),
            template_bgr=template.copy(),
            required=bool(data.get("required", True)),
            weight=float(data.get("weight", 1.0)),
        )


@dataclass(frozen=True)
class SetupProfile:
    profile_id: str
    objects: list[ObjectTemplate]
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "profile_id": self.profile_id,
            "objects": [object_template.to_dict() for object_template in self.objects],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SetupProfile:
        return cls(
            profile_id=str(data["profile_id"]),
            objects=[ObjectTemplate.from_dict(item) for item in data.get("objects", [])],
            version=int(data.get("version", 1)),
        )


@dataclass(frozen=True)
class ReferenceState:
    reference_id: str
    profile_id: str
    dataset_path: str
    episode_index: int
    frame_index: int
    camera_key: str
    target_centroids: dict[str, tuple[float, float]]
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "reference_id": self.reference_id,
            "profile_id": self.profile_id,
            "dataset_path": self.dataset_path,
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "camera_key": self.camera_key,
            "target_centroids": {label: list(centroid) for label, centroid in self.target_centroids.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReferenceState:
        return cls(
            reference_id=str(data["reference_id"]),
            profile_id=str(data["profile_id"]),
            dataset_path=str(data["dataset_path"]),
            episode_index=int(data["episode_index"]),
            frame_index=int(data["frame_index"]),
            camera_key=str(data["camera_key"]),
            target_centroids={
                str(label): (float(value[0]), float(value[1]))
                for label, value in data.get("target_centroids", {}).items()
            },
            version=int(data.get("version", 1)),
        )


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: BoundingBox

    @property
    def centroid(self) -> tuple[float, float]:
        return self.bbox.centroid

    @property
    def area(self) -> float:
        return self.bbox.area

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": asdict(self.bbox),
            "centroid": list(self.centroid),
            "area": self.area,
        }


@dataclass(frozen=True)
class ObjectScore:
    label: str
    required: bool
    present: bool
    score: float
    centroid_error: float | None
    guidance: str
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MatchResult:
    match_id: str
    profile_id: str
    reference_id: str
    reference_dataset_path: str
    reference_episode_index: int
    reference_frame_index: int
    replay_dataset_path: str
    replay_episode_index: int
    camera_key: str
    ready: bool
    overall_score: float
    object_scores: list[ObjectScore]
    warnings: list[str] = field(default_factory=list)
    frame_results: list[dict[str, Any]] = field(default_factory=list)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "match_id": self.match_id,
            "profile_id": self.profile_id,
            "reference_id": self.reference_id,
            "reference_dataset_path": self.reference_dataset_path,
            "reference_episode_index": self.reference_episode_index,
            "reference_frame_index": self.reference_frame_index,
            "replay_dataset_path": self.replay_dataset_path,
            "replay_episode_index": self.replay_episode_index,
            "camera_key": self.camera_key,
            "ready": self.ready,
            "overall_score": self.overall_score,
            "object_scores": [object_score.to_dict() for object_score in self.object_scores],
            "warnings": list(self.warnings),
            "frame_results": list(self.frame_results),
        }
