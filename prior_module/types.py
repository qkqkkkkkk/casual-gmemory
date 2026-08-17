"""Public data contracts for the standalone prior module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class CandidateRecord:
    """A normalized memory candidate independent of its original storage."""

    memory_id: str
    memory_type: str
    polarity: str
    content: str
    structured_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.memory_type not in {"trajectory", "insight"}:
            raise ValueError("memory_type must be 'trajectory' or 'insight'.")
        if not self.memory_id:
            raise ValueError("memory_id must not be empty.")


@dataclass(frozen=True)
class RetrievalRecord:
    """Provenance emitted by the retrieval layer for one candidate."""

    candidate_id: str
    query_task: str
    retrieval_source: str
    rank: Optional[int] = None
    semantic_similarity: Optional[float] = None
    distance: Optional[float] = None
    actual_hop: Optional[int] = None
    path_weight: Optional[float] = None
    related_task_hit_count: Optional[int] = None


@dataclass(frozen=True)
class DecisionContext:
    """The state visible when one Agent decides whether to use a memory."""

    task_main: str
    task_description: str
    agent_id: str
    agent_profile: str = ""
    system_instruction: str = ""
    latest_observation: str = ""
    recent_task_trajectory: str = ""
    step_index: int = 0
    task_config: Mapping[str, Any] = field(default_factory=dict)
    graph_features: Mapping[str, float] = field(default_factory=dict)
    run_id: Optional[str] = None

    @property
    def task_text(self) -> str:
        return "\n".join(part for part in (self.task_main, self.task_description) if part)

    @property
    def agent_text(self) -> str:
        return "\n".join(
            part for part in (self.agent_profile, self.system_instruction) if part
        )

    @property
    def state_text(self) -> str:
        parts = [part for part in (self.latest_observation,) if part]
        if self.recent_task_trajectory:
            parts.append("Recent execution:\n" + self.recent_task_trajectory)
        return "\n".join(parts)


@dataclass(frozen=True)
class ReliabilityScore:
    mean: float
    variance: float
    evidence_count: float
    source: str
    calibrated: bool
    group_key: str


@dataclass(frozen=True)
class DependencyScore:
    expected_distance: float
    distance_distribution: Mapping[float, float]
    p_any_change: float
    p_material_change: float
    uncertainty: float
    source: str
    calibrated: bool


@dataclass(frozen=True)
class TeamUtilityScore:
    negative: float
    neutral: float
    positive: float
    uncertainty: float
    source: str
    calibrated: bool


@dataclass(frozen=True)
class PriorScore:
    """The three priors and an auditable controller recommendation."""

    memory_id: str
    memory_type: str
    polarity: str
    reliability: ReliabilityScore
    dependency: DependencyScore
    team_utility: TeamUtilityScore
    recommended_action: str
    reason_codes: tuple[str, ...]
    features: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayArtifact:
    """Raw source-task data required to validate a trajectory by replay."""

    memory_id: str
    source_task_config: Mapping[str, Any]
    raw_actions: tuple[str, ...]
    source_label: bool
    source_final_reward: Optional[float] = None
    seed: Optional[int] = None
    environment_version: Optional[str] = None
    mas_version: Optional[str] = None
    model_id: Optional[str] = None
    prompt_version: Optional[str] = None


@dataclass(frozen=True)
class ReplayExecution:
    goal_reached: bool
    final_reward: Optional[float]
    invalid_action_count: int = 0
    error: Optional[str] = None


@dataclass(frozen=True)
class ReplayResult:
    memory_id: str
    passed: bool
    execution: ReplayExecution
    success_threshold: float


@dataclass(frozen=True)
class DependencyInterventionResult:
    """One treatment/control action pair and its semantic distance label."""

    decision_id: str
    memory_id: str
    action_with: str
    action_control: str
    distance: float
    distance_reason: str
    seed: Optional[int] = None


@dataclass(frozen=True)
class TeamRolloutResult:
    """A paired team rollout used to supervise the utility head."""

    decision_id: str
    memory_id: str
    reward_with: float
    reward_control: float
    delta_reward: float
    utility_class: str
    threshold: float
    seed: Optional[int] = None
