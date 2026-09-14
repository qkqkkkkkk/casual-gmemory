"""Typed contracts for event-level causal memory control.

The unit of analysis is a *use event*: one candidate memory being considered
for one receiving agent at one frozen task state.  No score in this module is
treated as a permanent property of the memory itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Optional


class InterventionArm(str, Enum):
    USE = "use"
    DROP = "drop"
    PLACEBO = "placebo"
    GLOBAL_DROP = "global_drop"


class RelianceAction(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    VERIFY = "verify"


@dataclass(frozen=True)
class MemoryCandidate:
    memory_id: str
    memory_type: str
    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.memory_id:
            raise ValueError("memory_id must not be empty")
        if not self.memory_type:
            raise ValueError("memory_type must not be empty")


@dataclass(frozen=True)
class RetrievalMetadata:
    candidate_id: str
    source: str = "unknown"
    rank: Optional[int] = None
    similarity: Optional[float] = None
    distance: Optional[float] = None
    hop: Optional[int] = None
    path_weight: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecipientContext:
    """The candidate IDs visible to one agent at the injection boundary."""

    agent_id: str
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("agent_id must not be empty")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids must not contain duplicates")


@dataclass(frozen=True)
class MemoryUseEvent:
    """x = (query, state, receiver, memory, candidate set)."""

    event_id: str
    query: str
    task_state: str
    receiver_agent_id: str
    receiver_role: str
    memory: MemoryCandidate
    candidate_set: tuple[MemoryCandidate, ...]
    recipient_contexts: tuple[RecipientContext, ...] = ()
    retrieval: Optional[RetrievalMetadata] = None
    task_metadata: Mapping[str, Any] = field(default_factory=dict)
    reliability_prior: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        candidate_ids = tuple(candidate.memory_id for candidate in self.candidate_set)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate_set must not contain duplicate memory IDs")
        if self.memory.memory_id not in candidate_ids:
            raise ValueError("the target memory must be present in candidate_set")

        contexts = self.recipient_contexts
        if not contexts:
            contexts = (RecipientContext(self.receiver_agent_id, candidate_ids),)
            object.__setattr__(self, "recipient_contexts", contexts)
        if len({context.agent_id for context in contexts}) != len(contexts):
            raise ValueError("recipient_contexts must contain unique agent IDs")
        receiver = next(
            (context for context in contexts if context.agent_id == self.receiver_agent_id),
            None,
        )
        if receiver is None:
            raise ValueError("recipient_contexts must include the receiver")
        if self.memory.memory_id not in receiver.candidate_ids:
            raise ValueError("the target memory must be visible to the receiver")
        known_ids = set(candidate_ids)
        unknown = {
            memory_id
            for context in contexts
            for memory_id in context.candidate_ids
            if memory_id not in known_ids
        }
        if unknown:
            raise ValueError(f"recipient_contexts contain unknown IDs: {sorted(unknown)}")
        if self.retrieval and self.retrieval.candidate_id != self.memory.memory_id:
            raise ValueError("retrieval metadata belongs to a different candidate")
        if self.reliability_prior is not None and not 0.0 <= self.reliability_prior <= 1.0:
            raise ValueError("reliability_prior must be in [0, 1]")

    @property
    def observation_count(self) -> int:
        """|O(m)|: how many agents receive this memory in this event snapshot."""

        return sum(
            self.memory.memory_id in context.candidate_ids
            for context in self.recipient_contexts
        )

    def context_for(self, agent_id: str) -> RecipientContext:
        for context in self.recipient_contexts:
            if context.agent_id == agent_id:
                return context
        raise KeyError(agent_id)


@dataclass(frozen=True)
class AuditCheckpoint:
    """A host-owned state captured after retrieval and before prompt injection."""

    checkpoint_id: str
    event_id: str
    upstream_state: Any
    sampling_config: Mapping[str, Any]
    upstream_digest: str = ""

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id must not be empty")
        if not self.event_id:
            raise ValueError("event_id must not be empty")


@dataclass(frozen=True)
class BranchRequest:
    checkpoint: AuditCheckpoint
    event: MemoryUseEvent
    arm: InterventionArm
    seed: int
    repeat_index: int
    recipient_contexts: tuple[RecipientContext, ...]
    placebo_candidate: Optional[MemoryCandidate] = None

    def __post_init__(self) -> None:
        if self.checkpoint.event_id != self.event.event_id:
            raise ValueError("checkpoint and event IDs do not match")
        if self.arm == InterventionArm.PLACEBO and self.placebo_candidate is None:
            raise ValueError("PLACEBO requires a placebo_candidate")

    def context_for(self, agent_id: str) -> RecipientContext:
        for context in self.recipient_contexts:
            if context.agent_id == agent_id:
                return context
        raise KeyError(agent_id)


@dataclass(frozen=True)
class BranchOutcome:
    response: str
    team_reward: float
    local_metric: Optional[float] = None
    behavior_payload: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.team_reward)):
            raise ValueError("team_reward must be finite")
        if self.local_metric is not None and not math.isfinite(float(self.local_metric)):
            raise ValueError("local_metric must be finite when provided")

    @property
    def behavior(self) -> Any:
        return self.response if self.behavior_payload is None else self.behavior_payload


@dataclass(frozen=True)
class Estimate:
    mean: float
    stddev: float
    stderr: float
    ci_low: float
    ci_high: float
    count: int


@dataclass(frozen=True)
class PairedRun:
    repeat_index: int
    seed: int
    use: BranchOutcome
    control: BranchOutcome
    behavior_distance: float
    team_utility: float
    local_utility: Optional[float]


@dataclass(frozen=True)
class ArmAuditResult:
    arm: InterventionArm
    pairs: tuple[PairedRun, ...]
    behavior_effect: Estimate
    team_utility: Estimate
    local_utility: Optional[Estimate]


@dataclass(frozen=True)
class CounterfactualAuditResult:
    event: MemoryUseEvent
    checkpoint_id: str
    arms: tuple[ArmAuditResult, ...]

    def for_arm(self, arm: InterventionArm) -> ArmAuditResult:
        for result in self.arms:
            if result.arm == arm:
                return result
        raise KeyError(arm.value)

    @property
    def primary(self) -> ArmAuditResult:
        return self.for_arm(InterventionArm.DROP)


@dataclass(frozen=True)
class PotentialOutcomePrediction:
    q_use: float
    q_drop: float
    utility: float
    utility_class: str
    uncertainty: float
    source: str
    calibrated: bool
    training_samples: int


@dataclass(frozen=True)
class RelianceDecision:
    action: RelianceAction
    lower_bound: float
    upper_bound: float
    threshold: float
    reason: str
    prediction: PotentialOutcomePrediction

