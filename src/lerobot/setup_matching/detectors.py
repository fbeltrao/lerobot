#!/usr/bin/env python

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .models import BoundingBox, Detection, ObjectTemplate, SetupProfile


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray, profile: SetupProfile, camera_key: str) -> list[Detection]:
        """Return object detections for a frame and camera."""


@dataclass(frozen=True)
class TemplateMatchingDetector:
    confidence_threshold: float = 0.5

    def detect(self, frame_bgr: np.ndarray, profile: SetupProfile, camera_key: str) -> list[Detection]:
        detections: list[Detection] = []
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_height, frame_width = frame_bgr.shape[:2]

        for object_template in profile.objects:
            if object_template.camera_key != camera_key:
                continue
            detection = self._match_template(frame_gray, frame_height, frame_width, object_template)
            if detection is not None and detection.confidence >= self.confidence_threshold:
                detections.append(detection)
        return detections

    def _match_template(
        self,
        frame_gray: np.ndarray,
        frame_height: int,
        frame_width: int,
        object_template: ObjectTemplate,
    ) -> Detection | None:
        template_gray = cv2.cvtColor(object_template.template_bgr, cv2.COLOR_BGR2GRAY)
        template_height, template_width = template_gray.shape[:2]
        if template_height > frame_height or template_width > frame_width:
            return None

        if float(np.std(template_gray)) < 1.0:
            match_result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_SQDIFF_NORMED)
            min_value, _, min_location, _ = cv2.minMaxLoc(match_result)
            confidence = max(0.0, 1.0 - float(min_value))
            left, top = min_location
        else:
            match_result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_value, _, max_location = cv2.minMaxLoc(match_result)
            confidence = max(0.0, min(1.0, float(max_value)))
            left, top = max_location

        right = left + template_width
        bottom = top + template_height
        if not math.isfinite(confidence):
            confidence = 0.0
        return Detection(
            label=object_template.label,
            confidence=confidence,
            bbox=BoundingBox(
                x_min=left / frame_width,
                y_min=top / frame_height,
                x_max=right / frame_width,
                y_max=bottom / frame_height,
            ),
        )


def create_object_template(
    frame_bgr: np.ndarray,
    label: str,
    camera_key: str,
    bbox: BoundingBox,
    required: bool = True,
    weight: float = 1.0,
) -> ObjectTemplate:
    rows, cols = bbox.to_pixel_slice(*frame_bgr.shape[:2])
    template = frame_bgr[rows, cols].copy()
    return ObjectTemplate(
        label=label,
        camera_key=camera_key,
        reference_bbox=bbox,
        template_bgr=template,
        required=required,
        weight=weight,
    )
