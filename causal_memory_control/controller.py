"""Uncertainty-aware ACCEPT/REJECT/VERIFY reliance controller."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .types import (
    MemoryUseEvent,
    PotentialOutcomePrediction,
    RecipientContext,
    RelianceAction,
    RelianceDecision,
)


class RelianceController:
    def __init__(self, *, delta: float = 0.0, kappa: float = 1.96):
        if delta < 0:
            raise ValueError("delta must be non-negative")
        if kappa < 0:
            raise ValueError("kappa must be non-negative")
        self.delta = delta
        self.kappa = kappa

    def decide(self, prediction: PotentialOutcomePrediction) -> RelianceDecision:
        if prediction.uncertainty < 0 or not math.isfinite(prediction.uncertainty):
            raise ValueError("prediction uncertainty must be finite and non-negative")
        lower = prediction.utility - self.kappa * prediction.uncertainty
        upper = prediction.utility + self.kappa * prediction.uncertainty
        if lower > self.delta:
            action = RelianceAction.ACCEPT
            reason = "utility_lower_bound_above_threshold"
        elif upper < -self.delta:
            action = RelianceAction.REJECT
            reason = "utility_upper_bound_below_negative_threshold"
        else:
            action = RelianceAction.VERIFY
            reason = "utility_interval_crosses_decision_region"
        return RelianceDecision(
            action=action,
            lower_bound=lower,
            upper_bound=upper,
            threshold=self.delta,
            reason=reason,
            prediction=prediction,
        )


def controlled_receiver_context(
    event: MemoryUseEvent,
    decisions: Mapping[str, RelianceDecision],
    *,
    abstain_on_verify: bool = True,
) -> RecipientContext:
    """Apply decisions to only the event's receiver candidate set.

    VERIFY is an abstention in V1.  Set ``abstain_on_verify=False`` only when a
    host has a real verification path and intentionally keeps the candidate
    visible while that path runs.
    """

    source = event.context_for(event.receiver_agent_id)
    kept = []
    for candidate_id in source.candidate_ids:
        decision = decisions.get(candidate_id)
        if decision is None:
            kept.append(candidate_id)
        elif decision.action == RelianceAction.ACCEPT:
            kept.append(candidate_id)
        elif decision.action == RelianceAction.VERIFY and not abstain_on_verify:
            kept.append(candidate_id)
    return RecipientContext(event.receiver_agent_id, tuple(kept))

