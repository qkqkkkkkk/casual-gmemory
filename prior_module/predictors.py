"""Supervised d and u heads with transparent cold-start behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .actions import ActionDistancePolicy
from .models import SoftmaxClassifier, normalized_entropy
from .types import DependencyScore, TeamUtilityScore


DISTANCE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
UTILITY_CLASSES = ("negative", "neutral", "positive")


class DependencyPredictor:
    """Predict the distribution of paired canonical-action distances."""

    def __init__(
        self,
        *,
        material_threshold: float = 0.5,
        min_samples: int = 8,
    ):
        self.material_threshold = material_threshold
        self.model = SoftmaxClassifier(
            labels=DISTANCE_LEVELS,
            min_samples=min_samples,
        )

    def fit(
        self, feature_rows: Sequence[Mapping[str, float]], distances: Sequence[float]
    ) -> bool:
        labels = [self._nearest_level(distance) for distance in distances]
        return self.model.fit(feature_rows, labels)

    def predict(self, features: Mapping[str, float]) -> DependencyScore:
        probabilities = self.model.predict_proba(features)
        distribution = {
            float(level): float(probabilities[level]) for level in DISTANCE_LEVELS
        }
        expected = sum(level * probability for level, probability in distribution.items())
        p_any = 1.0 - distribution[0.0]
        p_material = sum(
            probability
            for level, probability in distribution.items()
            if level >= self.material_threshold
        )
        return DependencyScore(
            expected_distance=expected,
            distance_distribution=distribution,
            p_any_change=p_any,
            p_material_change=p_material,
            uncertainty=normalized_entropy(list(distribution.values())),
            source="supervised_ordinal" if self.model.fitted else "cold_start_uniform",
            calibrated=False,
        )

    @staticmethod
    def _nearest_level(distance: float) -> float:
        if not 0.0 <= distance <= 1.0:
            raise ValueError("Dependency distance must lie in [0, 1].")
        return min(DISTANCE_LEVELS, key=lambda level: abs(level - distance))


class TeamUtilityPredictor:
    """Predict a negative/neutral/positive marginal team utility class."""

    def __init__(self, *, min_samples: int = 8):
        self.model = SoftmaxClassifier(labels=UTILITY_CLASSES, min_samples=min_samples)

    def fit(
        self, feature_rows: Sequence[Mapping[str, float]], labels: Sequence[str]
    ) -> bool:
        return self.model.fit(feature_rows, labels)

    def predict(self, features: Mapping[str, float]) -> TeamUtilityScore:
        probabilities = self.model.predict_proba(features)
        values = [float(probabilities[label]) for label in UTILITY_CLASSES]
        return TeamUtilityScore(
            negative=values[0],
            neutral=values[1],
            positive=values[2],
            uncertainty=normalized_entropy(values),
            source="rollout_supervised"
            if self.model.fitted
            else "cold_start_uniform",
            calibrated=False,
        )
