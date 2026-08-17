"""The public prior-module orchestration API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .decision_log import DecisionLogger
from .features import FeatureBuilder, TextEmbedder
from .predictors import DependencyPredictor, TeamUtilityPredictor
from .reliability import ReliabilityPrior
from .types import (
    CandidateRecord,
    DecisionContext,
    PriorScore,
    ReplayResult,
    RetrievalRecord,
)


@dataclass(frozen=True)
class ControllerConfig:
    use_positive_threshold: float = 0.60
    ignore_negative_threshold: float = 0.60
    reliability_floor: float = 0.40
    high_uncertainty_threshold: float = 0.65


class PriorModule:
    """A standalone r/d/u prior module with no write path into G-Memory."""

    def __init__(
        self,
        *,
        embedder: Optional[TextEmbedder] = None,
        reliability: Optional[ReliabilityPrior] = None,
        dependency: Optional[DependencyPredictor] = None,
        team_utility: Optional[TeamUtilityPredictor] = None,
        controller: ControllerConfig = ControllerConfig(),
        decision_log_path: Optional[str] = None,
    ):
        self.features = FeatureBuilder(embedder)
        self.reliability = reliability or ReliabilityPrior()
        self.dependency = dependency or DependencyPredictor()
        self.team_utility = team_utility or TeamUtilityPredictor()
        self.controller = controller
        self.logger = DecisionLogger(decision_log_path) if decision_log_path else None

    def build_base_features(
        self,
        context: DecisionContext,
        candidate: CandidateRecord,
        retrieval: Optional[RetrievalRecord],
        candidates: Sequence[CandidateRecord],
    ) -> dict[str, float]:
        return self.features.build(context, candidate, retrieval, candidates)

    def score(
        self,
        context: DecisionContext,
        candidates: Sequence[CandidateRecord],
        retrieval_by_id: Optional[Mapping[str, RetrievalRecord]] = None,
    ) -> list[PriorScore]:
        """Score all retrieved candidates without altering their source memory."""

        retrieval_by_id = retrieval_by_id or {}
        scored: list[PriorScore] = []
        for candidate in candidates:
            retrieval = retrieval_by_id.get(candidate.memory_id)
            base_features = self.build_base_features(
                context, candidate, retrieval, candidates
            )
            reliability = self.reliability.score(candidate)
            dependency = self.dependency.predict(base_features)
            utility_features = dict(base_features)
            utility_features.update(
                {
                    "r0_mean": reliability.mean,
                    "r0_variance": reliability.variance,
                    "r0_evidence_count": reliability.evidence_count,
                    "d0_expected_distance": dependency.expected_distance,
                    "d0_p_any_change": dependency.p_any_change,
                    "d0_p_material_change": dependency.p_material_change,
                    "d0_uncertainty": dependency.uncertainty,
                }
            )
            team_utility = self.team_utility.predict(utility_features)
            recommendation, reason_codes = self._recommend(
                candidate, reliability.mean, dependency.uncertainty, team_utility
            )
            score = PriorScore(
                memory_id=candidate.memory_id,
                memory_type=candidate.memory_type,
                polarity=candidate.polarity,
                reliability=reliability,
                dependency=dependency,
                team_utility=team_utility,
                recommended_action=recommendation,
                reason_codes=tuple(reason_codes),
                features=utility_features,
            )
            scored.append(score)
            self._log_score(context, candidate, retrieval, score, base_features)
        return scored

    def fit_dependency(
        self,
        feature_rows: Sequence[Mapping[str, float]],
        distances: Sequence[float],
    ) -> bool:
        """Train d from paired canonical-action intervention labels."""

        return self.dependency.fit(feature_rows, distances)

    def fit_team_utility(
        self,
        feature_rows: Sequence[Mapping[str, float]],
        utility_classes: Sequence[str],
    ) -> bool:
        """Train u from paired team-rollout class labels."""

        return self.team_utility.fit(feature_rows, utility_classes)

    def record_replay(
        self,
        replay: ReplayResult,
        *,
        group_key: Optional[str] = None,
    ) -> None:
        """Update r evidence only after an external replay has been run."""

        self.reliability.update_replay(
            replay.memory_id, replay.passed, group_key=group_key
        )
        if self.logger is not None:
            self.logger.write("source_replay", {"replay": replay})

    def _recommend(
        self,
        candidate: CandidateRecord,
        reliability_mean: float,
        dependency_uncertainty: float,
        team_utility,
    ) -> tuple[str, list[str]]:
        reasons: list[str] = []
        if candidate.polarity == "failure_warning":
            return "WARN", ["FAILURE_WARNING_NOT_IMITATION"]
        if team_utility.source == "cold_start_uniform":
            reasons.append("NO_U_ROLLOUT_LABELS")
        if dependency_uncertainty >= self.controller.high_uncertainty_threshold:
            reasons.append("HIGH_D_UNCERTAINTY")
        if reliability_mean < self.controller.reliability_floor:
            reasons.append("LOW_R_RELIABILITY")

        if (
            team_utility.source != "cold_start_uniform"
            and team_utility.positive >= self.controller.use_positive_threshold
            and reliability_mean >= self.controller.reliability_floor
        ):
            return "USE", reasons or ["HIGH_EXPECTED_TEAM_UTILITY"]
        if (
            team_utility.source != "cold_start_uniform"
            and team_utility.negative >= self.controller.ignore_negative_threshold
        ):
            return "IGNORE", reasons or ["HIGH_NEGATIVE_TEAM_UTILITY"]
        return "VERIFY", reasons or ["INSUFFICIENT_EVIDENCE"]

    def _log_score(
        self,
        context: DecisionContext,
        candidate: CandidateRecord,
        retrieval: Optional[RetrievalRecord],
        score: PriorScore,
        base_features: Mapping[str, float],
    ) -> None:
        if self.logger is None:
            return
        self.logger.write(
            "prior_decision",
            {
                "run_id": context.run_id,
                "context": context,
                "candidate_id": candidate.memory_id,
                "candidate_type": candidate.memory_type,
                "candidate_polarity": candidate.polarity,
                "retrieval": retrieval,
                "base_features": base_features,
                "prior_score": score,
            },
        )
