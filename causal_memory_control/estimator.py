"""Amortized two-potential-outcome estimator with bootstrap uncertainty."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Mapping, Sequence

from .types import CounterfactualAuditResult, PotentialOutcomePrediction


@dataclass(frozen=True)
class PotentialOutcomeExample:
    features: Mapping[str, float]
    q_use: float
    q_drop: float
    event_id: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.q_use)) or not math.isfinite(float(self.q_drop)):
            raise ValueError("potential outcomes must be finite")


def example_from_audit(
    audit: CounterfactualAuditResult,
    features: Mapping[str, float],
) -> PotentialOutcomeExample:
    """Collapse repeated receiver-level USE/DROP audits into one training row."""

    primary = audit.primary
    return PotentialOutcomeExample(
        features=dict(features),
        q_use=sum(pair.use.team_reward for pair in primary.pairs) / len(primary.pairs),
        q_drop=sum(pair.control.team_reward for pair in primary.pairs)
        / len(primary.pairs),
        event_id=audit.event.event_id,
    )


@dataclass
class _LinearRegressor:
    learning_rate: float
    epochs: int
    l2: float
    seed: int
    feature_names: tuple[str, ...] = ()
    means: list[float] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0

    def fit(
        self, feature_rows: Sequence[Mapping[str, float]], targets: Sequence[float]
    ) -> None:
        self.feature_names = tuple(sorted({key for row in feature_rows for key in row}))
        matrix = [
            [float(row.get(name, 0.0)) for name in self.feature_names]
            for row in feature_rows
        ]
        self.means, self.scales = _fit_standardizer(matrix)
        matrix = [_standardize(row, self.means, self.scales) for row in matrix]
        self.weights = [0.0] * len(self.feature_names)
        self.bias = sum(targets) / len(targets)
        order = list(range(len(matrix)))
        randomizer = random.Random(self.seed)
        for epoch in range(self.epochs):
            randomizer.shuffle(order)
            rate = self.learning_rate / math.sqrt(1.0 + epoch * 0.03)
            for index in order:
                vector = matrix[index]
                prediction = self.bias + sum(
                    weight * value for weight, value in zip(self.weights, vector)
                )
                error = max(-10.0, min(10.0, prediction - targets[index]))
                self.bias -= rate * error
                for feature_index, value in enumerate(vector):
                    gradient = error * value + self.l2 * self.weights[feature_index]
                    self.weights[feature_index] -= rate * gradient

    def predict(self, features: Mapping[str, float]) -> float:
        vector = _standardize(
            [float(features.get(name, 0.0)) for name in self.feature_names],
            self.means,
            self.scales,
        )
        return self.bias + sum(
            weight * value for weight, value in zip(self.weights, vector)
        )


class AmortizedUtilityEstimator:
    """A dependency-free T-learner for Q_use and Q_drop.

    Each bootstrap member fits two potential-outcome regressors.  Their utility
    spread plus held-in residual floors gives a conservative uncertainty used
    by the controller.  A stronger regressor can replace this class without
    changing its public prediction contract.
    """

    def __init__(
        self,
        *,
        ensemble_size: int = 12,
        min_samples: int = 8,
        learning_rate: float = 0.025,
        epochs: int = 220,
        l2: float = 1e-4,
        seed: int = 23,
        utility_threshold: float = 0.0,
        include_residual_noise: bool = True,
    ):
        if ensemble_size < 2:
            raise ValueError("ensemble_size must be at least two")
        if min_samples < 2:
            raise ValueError("min_samples must be at least two")
        if utility_threshold < 0:
            raise ValueError("utility_threshold must be non-negative")
        self.ensemble_size = ensemble_size
        self.min_samples = min_samples
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.utility_threshold = utility_threshold
        self.include_residual_noise = include_residual_noise
        self._models: list[tuple[_LinearRegressor, _LinearRegressor]] = []
        self._residual_use = 0.0
        self._residual_drop = 0.0
        self.training_samples = 0
        self.fitted = False

    def fit(self, examples: Sequence[PotentialOutcomeExample]) -> bool:
        self.training_samples = len(examples)
        self._models = []
        if len(examples) < self.min_samples:
            self.fitted = False
            return False
        rows = [example.features for example in examples]
        use_targets = [float(example.q_use) for example in examples]
        drop_targets = [float(example.q_drop) for example in examples]
        randomizer = random.Random(self.seed)
        for member in range(self.ensemble_size):
            indices = [randomizer.randrange(len(examples)) for _ in examples]
            sampled_rows = [rows[index] for index in indices]
            use = _LinearRegressor(
                self.learning_rate, self.epochs, self.l2, self.seed + 2 * member
            )
            drop = _LinearRegressor(
                self.learning_rate, self.epochs, self.l2, self.seed + 2 * member + 1
            )
            use.fit(sampled_rows, [use_targets[index] for index in indices])
            drop.fit(sampled_rows, [drop_targets[index] for index in indices])
            self._models.append((use, drop))
        self.fitted = True

        use_predictions = [self._mean_prediction(row, outcome=0) for row in rows]
        drop_predictions = [self._mean_prediction(row, outcome=1) for row in rows]
        self._residual_use = _rmse(use_targets, use_predictions)
        self._residual_drop = _rmse(drop_targets, drop_predictions)
        return True

    def predict(self, features: Mapping[str, float]) -> PotentialOutcomePrediction:
        if not self.fitted:
            return PotentialOutcomePrediction(
                q_use=0.0,
                q_drop=0.0,
                utility=0.0,
                utility_class="neutral",
                uncertainty=1.0,
                source="cold_start",
                calibrated=False,
                training_samples=self.training_samples,
            )
        use_values = [use.predict(features) for use, _ in self._models]
        drop_values = [drop.predict(features) for _, drop in self._models]
        utility_values = [
            use_value - drop_value
            for use_value, drop_value in zip(use_values, drop_values)
        ]
        q_use = sum(use_values) / len(use_values)
        q_drop = sum(drop_values) / len(drop_values)
        utility = q_use - q_drop
        ensemble_variance = _sample_variance(utility_values)
        residual_variance = 0.0
        if self.include_residual_noise:
            residual_variance = self._residual_use**2 + self._residual_drop**2
        uncertainty = math.sqrt(max(0.0, ensemble_variance + residual_variance))
        if utility > self.utility_threshold:
            utility_class = "positive"
        elif utility < -self.utility_threshold:
            utility_class = "negative"
        else:
            utility_class = "neutral"
        return PotentialOutcomePrediction(
            q_use=q_use,
            q_drop=q_drop,
            utility=utility,
            utility_class=utility_class,
            uncertainty=uncertainty,
            source="bootstrap_t_learner",
            calibrated=True,
            training_samples=self.training_samples,
        )

    def _mean_prediction(self, features: Mapping[str, float], outcome: int) -> float:
        values = [models[outcome].predict(features) for models in self._models]
        return sum(values) / len(values)


def _fit_standardizer(
    matrix: Sequence[Sequence[float]],
) -> tuple[list[float], list[float]]:
    width = len(matrix[0]) if matrix else 0
    count = len(matrix)
    means = [sum(row[index] for row in matrix) / count for index in range(width)]
    scales = []
    for index, mean in enumerate(means):
        variance = sum((row[index] - mean) ** 2 for row in matrix) / count
        scales.append(math.sqrt(variance) or 1.0)
    return means, scales


def _standardize(
    vector: Sequence[float], means: Sequence[float], scales: Sequence[float]
) -> list[float]:
    return [
        (value - mean) / scale
        for value, mean, scale in zip(vector, means, scales)
    ]


def _rmse(expected: Sequence[float], observed: Sequence[float]) -> float:
    return math.sqrt(
        sum((left - right) ** 2 for left, right in zip(expected, observed))
        / len(expected)
    )


def _sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)

