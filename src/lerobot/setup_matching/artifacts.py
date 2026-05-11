#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import MatchResult, ReferenceState, ReviewStatus, SetupProfile


class ArtifactStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def save_profile(self, profile: SetupProfile) -> Path:
        return self._write_json("profiles", profile.profile_id, profile.to_dict())

    def save_reference_state(self, reference_state: ReferenceState) -> Path:
        return self._write_json("reference_states", reference_state.reference_id, reference_state.to_dict())

    def save_match_result(self, match_result: MatchResult) -> Path:
        return self._write_json("match_results", match_result.match_id, match_result.to_dict())

    def _write_json(self, directory_name: str, artifact_id: str, data: dict[str, Any]) -> Path:
        directory = self.root / directory_name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{artifact_id}.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=True)
            file.write("\n")
        return path


class EpisodeReviewStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def set_review(
        self,
        dataset_path: Path | str,
        episode_index: int,
        status: ReviewStatus,
        failure_category: str | None = None,
        note: str | None = None,
    ) -> None:
        reviews = self._read()
        dataset_key = str(dataset_path)
        reviews.setdefault(dataset_key, {})[str(episode_index)] = {
            "status": status,
            "failure_category": failure_category,
            "note": note,
        }
        self._write(reviews)

    def get_review(self, dataset_path: Path | str, episode_index: int) -> dict[str, Any]:
        reviews = self._read()
        return reviews.get(str(dataset_path), {}).get(str(episode_index), {"status": "unreviewed"})

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _write(self, reviews: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(reviews, file, indent=2, sort_keys=True)
            file.write("\n")
