"""Interfaces for replay, action-pair, and team-rollout label collection."""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Protocol
import uuid

from .actions import ActionCanonicalizer, ActionDistancePolicy
from .types import (
    DependencyInterventionResult,
    ReplayArtifact,
    ReplayExecution,
    ReplayResult,
    TeamRolloutResult,
)


class ReplayExecutor(Protocol):
    """Reset source_task_config and execute artifact.raw_actions exactly once."""

    def __call__(self, artifact: ReplayArtifact) -> ReplayExecution:
        """Return environment-level replay evidence."""


def verify_source_replay(
    artifact: ReplayArtifact,
    executor: ReplayExecutor,
    *,
    success_threshold: float = 0.0,
) -> ReplayResult:
    """Verify an original positive trajectory without touching G-Memory."""

    execution = executor(artifact)
    reward_ok = (
        execution.final_reward is not None
        and execution.final_reward >= success_threshold
    )
    passed = bool(
        artifact.source_label
        and execution.goal_reached
        and execution.invalid_action_count == 0
        and reward_ok
        and execution.error is None
    )
    return ReplayResult(
        memory_id=artifact.memory_id,
        passed=passed,
        execution=execution,
        success_threshold=success_threshold,
    )


def collect_dependency_pair(
    memory_id: str,
    action_with: str,
    action_control: str,
    *,
    canonicalizer: ActionCanonicalizer,
    distance_policy: ActionDistancePolicy,
    action_context: Optional[Mapping[str, object]] = None,
    decision_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> DependencyInterventionResult:
    """Label one fixed-context treatment/control forward pair."""

    left = canonicalizer.canonicalize(action_with)
    right = canonicalizer.canonicalize(action_control)
    result = distance_policy.distance(left, right, action_context)
    return DependencyInterventionResult(
        decision_id=decision_id or str(uuid.uuid4()),
        memory_id=memory_id,
        action_with=action_with,
        action_control=action_control,
        distance=result.distance,
        distance_reason=result.reason,
        seed=seed,
    )


def classify_utility(delta_reward: float, threshold: float) -> str:
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")
    if delta_reward < -threshold:
        return "negative"
    if delta_reward > threshold:
        return "positive"
    return "neutral"


def collect_team_rollout_pair(
    memory_id: str,
    reward_with: float,
    reward_control: float,
    *,
    threshold: float,
    decision_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> TeamRolloutResult:
    """Label a paired team rollout after the caller has run both branches."""

    delta = float(reward_with) - float(reward_control)
    return TeamRolloutResult(
        decision_id=decision_id or str(uuid.uuid4()),
        memory_id=memory_id,
        reward_with=float(reward_with),
        reward_control=float(reward_control),
        delta_reward=delta,
        utility_class=classify_utility(delta, threshold),
        threshold=threshold,
        seed=seed,
    )
