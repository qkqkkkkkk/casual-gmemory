"""Standalone, non-invasive r/d/u prior module for G-Memory candidates."""

from .actions import (
    ActionCanonicalizer,
    ActionDistancePolicy,
    CanonicalAction,
    DistanceResult,
)
from .adapters import GMemoryAdapter
from .features import GMemoryEmbeddingAdapter, HashEmbedder
from .interventions import (
    classify_utility,
    collect_dependency_pair,
    collect_team_rollout_pair,
    verify_source_replay,
)
from .module import ControllerConfig, PriorModule
from .predictors import DependencyPredictor, TeamUtilityPredictor
from .reliability import ReliabilityPrior
from .types import (
    CandidateRecord,
    DecisionContext,
    DependencyInterventionResult,
    PriorScore,
    ReplayArtifact,
    ReplayExecution,
    ReplayResult,
    RetrievalRecord,
    TeamRolloutResult,
)

__all__ = [
    "ActionCanonicalizer",
    "ActionDistancePolicy",
    "CanonicalAction",
    "CandidateRecord",
    "ControllerConfig",
    "DecisionContext",
    "DependencyInterventionResult",
    "DependencyPredictor",
    "DistanceResult",
    "GMemoryAdapter",
    "GMemoryEmbeddingAdapter",
    "HashEmbedder",
    "PriorModule",
    "PriorScore",
    "ReplayArtifact",
    "ReplayExecution",
    "ReplayResult",
    "RetrievalRecord",
    "ReliabilityPrior",
    "TeamUtilityPredictor",
    "TeamRolloutResult",
    "classify_utility",
    "collect_dependency_pair",
    "collect_team_rollout_pair",
    "verify_source_replay",
]
