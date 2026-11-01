"""Opt-in Semantic ST deployment policies; saved calibration stays unchanged."""
from __future__ import annotations

from dataclasses import replace
import json
import math

from .runtime_gate import GMemoryExposureGate
from .types import ExposureAction


RULES = ("calibrated", "positive_unlimited")
POSITIVE_POLICY = {
    "rule": "positive_unlimited", "keep_condition": "predicted_U > 0",
    "drop_condition": "predicted_U <= 0", "kappa": 0.0, "delta": 0.0,
    "max_drops": None, "semantic_dedup": False, "retraining": False,
}


def positive_decisions(decisions):
    """Use predictions only, never the task's gold label or observed outcome."""
    result = []
    seen = set()
    for decision in decisions:
        memory_id = decision["memory_id"]
        if memory_id in seen:
            raise ValueError("duplicate Semantic ST candidate ID")
        seen.add(memory_id)
        prediction = decision["prediction"]
        value = prediction["utility"]
        if (prediction.get("calibrated") is not True or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value)
                or prediction.get("source") != "bootstrap_semantic_st_direct_u"):
            raise ValueError("a finite, trained Semantic ST prediction is required for every candidate")
        result.append({**decision, "action": "KEEP" if value > 0 else "DROP",
            "upper_utility_bound": value,
            "reason": "st_positive_utility" if value > 0 else "st_nonpositive_utility",
            "policy_scores": {**decision.get("policy_scores", {}), **POSITIVE_POLICY}})
    return result


def validate_st_checkpoint(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if (payload.get("schema") != "gmemory-team-exposure-gate-v1"
            or payload.get("estimator", {}).get("schema") != "st-direct-utility-estimator-v1"):
        raise ValueError(f"an existing Semantic ST direct-U checkpoint is required: {path}")
    return payload


class PositiveSTGate(GMemoryExposureGate):
    """Same frozen features and fitted ST model, optional positive-only rule."""

    def __init__(self, **kwargs):
        kwargs.update(kappa=0.0, delta=0.0, max_drops=None)
        super().__init__(**kwargs)

    @classmethod
    def from_checkpoint(cls, path, **kwargs):
        validate_st_checkpoint(path)
        return super().from_checkpoint(path, **kwargs)

    def _decide(self, event):
        decision = super()._decide(event)
        if self.mode != "learned" or decision.reason in (
                "candidate_kind_out_of_scope", "candidate_rank_out_of_scope"):
            return decision
        selected = positive_decisions([decision.to_dict()])[0]
        return replace(decision, action=ExposureAction[selected["action"]],
            upper_utility_bound=selected["upper_utility_bound"], reason=selected["reason"],
            policy_scores=selected["policy_scores"])
