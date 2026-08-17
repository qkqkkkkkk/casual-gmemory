"""Small dependency-free calibrated-model baselines for d and u."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Hashable, Mapping, Sequence


def normalized_entropy(probabilities: Sequence[float]) -> float:
    valid = [max(0.0, value) for value in probabilities]
    total = sum(valid)
    if total == 0 or len(valid) <= 1:
        return 0.0
    entropy = -sum(
        (value / total) * math.log(value / total)
        for value in valid
        if value > 0
    )
    return entropy / math.log(len(valid))


@dataclass
class SoftmaxClassifier:
    """A small linear multiclass baseline trained with deterministic SGD.

    It deliberately has no third-party ML dependency. For larger experiments,
    this class can be replaced by LightGBM while preserving the predictor API.
    """

    labels: tuple[Hashable, ...]
    learning_rate: float = 0.08
    epochs: int = 120
    l2: float = 1e-4
    min_samples: int = 8
    seed: int = 17
    feature_names: tuple[str, ...] = ()
    means: list[float] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)
    weights: list[list[float]] = field(default_factory=list)
    biases: list[float] = field(default_factory=list)
    fitted: bool = False
    training_samples: int = 0

    def fit(
        self, feature_rows: Sequence[Mapping[str, float]], labels: Sequence[Hashable]
    ) -> bool:
        if len(feature_rows) != len(labels):
            raise ValueError("feature_rows and labels must have the same length.")
        if len(feature_rows) < self.min_samples:
            self.fitted = False
            self.training_samples = len(feature_rows)
            return False
        if any(label not in self.labels for label in labels):
            raise ValueError("Observed label is absent from classifier labels.")

        self.feature_names = tuple(sorted({key for row in feature_rows for key in row}))
        matrix = [
            [float(row.get(feature, 0.0)) for feature in self.feature_names]
            for row in feature_rows
        ]
        self.means, self.scales = _fit_standardizer(matrix)
        matrix = [_standardize(row, self.means, self.scales) for row in matrix]
        class_count = len(self.labels)
        feature_count = len(self.feature_names)
        self.weights = [[0.0] * feature_count for _ in range(class_count)]
        self.biases = [0.0] * class_count
        label_to_index = {label: index for index, label in enumerate(self.labels)}
        order = list(range(len(matrix)))
        randomizer = random.Random(self.seed)

        for epoch in range(self.epochs):
            randomizer.shuffle(order)
            rate = self.learning_rate / math.sqrt(1.0 + epoch * 0.03)
            for sample_index in order:
                vector = matrix[sample_index]
                target = label_to_index[labels[sample_index]]
                probabilities = self._probabilities_for_vector(vector)
                for class_index in range(class_count):
                    error = probabilities[class_index] - float(class_index == target)
                    self.biases[class_index] -= rate * error
                    weights = self.weights[class_index]
                    for feature_index, value in enumerate(vector):
                        gradient = error * value + self.l2 * weights[feature_index]
                        weights[feature_index] -= rate * gradient

        self.fitted = True
        self.training_samples = len(feature_rows)
        return True

    def predict_proba(self, features: Mapping[str, float]) -> dict[Hashable, float]:
        if not self.fitted:
            probability = 1.0 / len(self.labels)
            return {label: probability for label in self.labels}
        raw = [float(features.get(name, 0.0)) for name in self.feature_names]
        vector = _standardize(raw, self.means, self.scales)
        probabilities = self._probabilities_for_vector(vector)
        return dict(zip(self.labels, probabilities))

    def _probabilities_for_vector(self, vector: Sequence[float]) -> list[float]:
        scores = [
            bias + sum(weight * value for weight, value in zip(weights, vector))
            for weights, bias in zip(self.weights, self.biases)
        ]
        maximum = max(scores)
        exponentials = [math.exp(score - maximum) for score in scores]
        total = sum(exponentials)
        return [value / total for value in exponentials]


def _fit_standardizer(matrix: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    count = len(matrix)
    width = len(matrix[0])
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
