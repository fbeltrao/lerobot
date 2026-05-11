#!/usr/bin/env python

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class FrameSource(Protocol):
    def read_frame(self, frame_index: int) -> np.ndarray:
        """Return a BGR frame for a zero-based frame index within this source."""


@dataclass(frozen=True)
class StaticImageFrameSource:
    image_bgr: np.ndarray

    def read_frame(self, frame_index: int) -> np.ndarray:
        return self.image_bgr.copy()


@dataclass(frozen=True)
class EpisodeReplayFrameSource:
    dataset_path: Path
    camera_key: str
    episode_index: int = 0
    video_file_index: int | None = None
    chunk_index: int = 0
    loop: bool = True

    def __post_init__(self) -> None:
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")

    def read_frame(self, frame_index: int) -> np.ndarray:
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        info = self._load_info()
        video_path = self.video_path
        if self._uses_av1_codec(info):
            return self._read_frame_with_ffmpeg(video_path, frame_index, info)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return self._read_frame_with_ffmpeg(video_path, frame_index, info)

        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                return self._read_frame_with_ffmpeg(video_path, frame_index, info)
            selected_frame = frame_index % frame_count if self.loop else frame_index
            if selected_frame >= frame_count:
                raise IndexError(
                    f"frame_index {frame_index} out of range for {video_path} ({frame_count} frames)"
                )
            capture.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                return self._read_frame_with_ffmpeg(video_path, selected_frame, info)
            return frame_bgr
        finally:
            capture.release()

    @property
    def video_path(self) -> Path:
        info = self._load_info()
        features = info.get("features", {})
        if self.camera_key not in features:
            raise KeyError(f"Camera key {self.camera_key!r} not found in dataset features")
        video_pattern = info.get("video_path")
        if not isinstance(video_pattern, str):
            raise ValueError(f"Dataset {self.dataset_path} does not define a video_path pattern")

        file_index = self.episode_index if self.video_file_index is None else self.video_file_index
        candidate = self.dataset_path / video_pattern.format(
            video_key=self.camera_key,
            chunk_index=self.chunk_index,
            file_index=file_index,
        )
        if candidate.exists():
            return candidate

        fallback = self.dataset_path / video_pattern.format(
            video_key=self.camera_key,
            chunk_index=self.chunk_index,
            file_index=0,
        )
        if fallback.exists():
            return fallback
        raise FileNotFoundError(
            f"No video file found for camera {self.camera_key!r} under {self.dataset_path}"
        )

    def _load_info(self) -> dict:
        info_path = self.dataset_path / "meta" / "info.json"
        with info_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _read_frame_with_ffmpeg(
        self, video_path: Path, frame_index: int, info: dict
    ) -> np.ndarray:
        feature = info.get("features", {}).get(self.camera_key, {})
        shape = feature.get("shape")
        if not isinstance(shape, list) or len(shape) < 2:
            raise ValueError(f"Could not infer frame shape for camera {self.camera_key!r}")
        height = int(shape[0])
        width = int(shape[1])
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        completed = subprocess.run(command, check=False, capture_output=True)
        expected_bytes = height * width * 3
        if completed.returncode != 0 or len(completed.stdout) != expected_bytes:
            message = completed.stderr.decode("utf-8", errors="replace").strip()
            raise OSError(f"ffmpeg could not read frame {frame_index} from {video_path}: {message}")
        return np.frombuffer(completed.stdout, dtype=np.uint8).reshape((height, width, 3)).copy()

    def _uses_av1_codec(self, info: dict) -> bool:
        feature = info.get("features", {}).get(self.camera_key, {})
        video_info = feature.get("info", {})
        return video_info.get("video.codec") == "av1"
