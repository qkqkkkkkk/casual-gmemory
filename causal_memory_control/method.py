"""Small facade joining features, paired outcomes, and reliance decisions."""

from __future__ import annotations

from typing import Sequence

from .controller import RelianceController
from .diagnostics import PretrainingGateReport
from .estimator import (
    AmortizedUtilityEstimator,
    PotentialOutcomeExample,
    example_from_audit,
)
from .features import EventFeatureBuilder
from .types import (
    CounterfactualAuditResult,
    MemoryUseEvent,
    PotentialOutcomePrediction,
    RelianceDecision,
)


class CausalMemoryControlMethod:
    """End-to-end V1 predictor facade with an explicit pre-training gate."""

    def __init__(
        self,
        *,
        feature_builder: EventFeatureBuilder | None = None,
        estimator: AmortizedUtilityEstimator | None = None,
        controller: RelianceController | None = None,
        require_pretraining_gate: bool = True,
    ):
        self.feature_builder = feature_builder or EventFeatureBuilder()
        self.estimator = estimator or AmortizedUtilityEstimator()
        self.controller = controller or RelianceController()
        self.require_pretraining_gate = require_pretraining_gate
        self.pretraining_gate: PretrainingGateReport | None = None

    def approve_pretraining(self, report: PretrainingGateReport) -> None:
        self.pretraining_gate = report

    def fit(self, audits: Sequence[CounterfactualAuditResult]) -> bool:
        if self.require_pretraining_gate and (
            self.pretraining_gate is None or not self.pretraining_gate.can_train
        ):
            reasons = (
                self.pretraining_gate.reasons
                if self.pretraining_gate is not None
                else ("pretraining_gate_not_evaluated",)
            )
            raise RuntimeError("predictor training is blocked: " + ", ".join(reasons))
        examples: list[PotentialOutcomeExample] = [
            example_from_audit(audit, self.feature_builder.build(audit.event))
            for audit in audits
        ]
        return self.estimator.fit(examples)

    def predict(self, event: MemoryUseEvent) -> PotentialOutcomePrediction:
        return self.estimator.predict(self.feature_builder.build(event))

    def decide(self, event: MemoryUseEvent) -> RelianceDecision:
        return self.controller.decide(self.predict(event))

