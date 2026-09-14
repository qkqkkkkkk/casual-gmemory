"""Event-level causal memory audit and reliance control for G-Memory."""

from .audit import CounterfactualAudit
from .behavior import ExactMatchDistance, TokenJaccardDistance
from .checkpoint import SnapshotCheckpointBackend
from .controller import RelianceController, controlled_receiver_context
from .diagnostics import (
    NoiseFloorReport,
    PretrainingGateReport,
    PivotalityBin,
    PivotalityObservation,
    SignConsistency,
    UtilityRetest,
    evaluate_pretraining_gate,
    observation_count_distribution,
    oracle_noise_floor,
    stratify_pivotality,
)
from .estimator import (
    AmortizedUtilityEstimator,
    PotentialOutcomeExample,
    example_from_audit,
)
from .features import EventFeatureBuilder, GMemoryEmbeddingAdapter, HashEmbedder
from .gmemory_adapter import (
    AdaptedGMemoryRetrieval,
    GMemoryPromptInputs,
    GMemoryRetrievalAdapter,
)
from .interventions import (
    InterventionBuilder,
    NoPlaceboAvailable,
    choose_length_matched_placebo,
)
from .logging import JsonlLogger
from .method import CausalMemoryControlMethod
from .oracle import (
    OracleControllability,
    OracleEvaluation,
    OracleExample,
    PolicyEvaluation,
)
from .protocols import (
    BehaviorDistance,
    BranchRunner,
    CallableBranchRunner,
    CheckpointBackend,
)
from .types import (
    ArmAuditResult,
    AuditCheckpoint,
    BranchOutcome,
    BranchRequest,
    CounterfactualAuditResult,
    Estimate,
    InterventionArm,
    MemoryCandidate,
    MemoryUseEvent,
    PairedRun,
    PotentialOutcomePrediction,
    RecipientContext,
    RelianceAction,
    RelianceDecision,
    RetrievalMetadata,
)

__all__ = [
    "AdaptedGMemoryRetrieval",
    "AmortizedUtilityEstimator",
    "ArmAuditResult",
    "AuditCheckpoint",
    "BehaviorDistance",
    "BranchOutcome",
    "BranchRequest",
    "BranchRunner",
    "CallableBranchRunner",
    "CausalMemoryControlMethod",
    "CheckpointBackend",
    "CounterfactualAudit",
    "CounterfactualAuditResult",
    "Estimate",
    "EventFeatureBuilder",
    "ExactMatchDistance",
    "GMemoryEmbeddingAdapter",
    "GMemoryPromptInputs",
    "GMemoryRetrievalAdapter",
    "HashEmbedder",
    "InterventionArm",
    "InterventionBuilder",
    "JsonlLogger",
    "MemoryCandidate",
    "MemoryUseEvent",
    "NoPlaceboAvailable",
    "NoiseFloorReport",
    "OracleControllability",
    "OracleEvaluation",
    "OracleExample",
    "PairedRun",
    "PivotalityBin",
    "PivotalityObservation",
    "PolicyEvaluation",
    "PotentialOutcomeExample",
    "PotentialOutcomePrediction",
    "PretrainingGateReport",
    "RecipientContext",
    "RelianceAction",
    "RelianceController",
    "RelianceDecision",
    "RetrievalMetadata",
    "SignConsistency",
    "SnapshotCheckpointBackend",
    "TokenJaccardDistance",
    "UtilityRetest",
    "choose_length_matched_placebo",
    "controlled_receiver_context",
    "evaluate_pretraining_gate",
    "example_from_audit",
    "observation_count_distribution",
    "oracle_noise_floor",
    "stratify_pivotality",
]
