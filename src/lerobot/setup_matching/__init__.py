"""Pre-record setup matching primitives."""

from .artifacts import ArtifactStore, EpisodeReviewStore
from .detectors import TemplateMatchingDetector
from .frame_sources import EpisodeReplayFrameSource, StaticImageFrameSource
from .models import (
    BoundingBox,
    Detection,
    MatchResult,
    ObjectTemplate,
    ReferenceState,
    SetupProfile,
)
from .scenario import SetupMatchScenario, run_setup_match_scenario
from .scoring import ScoreConfig, score_setup

__all__ = [
    "ArtifactStore",
    "BoundingBox",
    "Detection",
    "EpisodeReplayFrameSource",
    "EpisodeReviewStore",
    "MatchResult",
    "ObjectTemplate",
    "ReferenceState",
    "ScoreConfig",
    "SetupMatchScenario",
    "SetupProfile",
    "StaticImageFrameSource",
    "TemplateMatchingDetector",
    "run_setup_match_scenario",
    "score_setup",
]
