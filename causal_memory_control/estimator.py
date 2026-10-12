"""Amortized two-potential-outcome estimator with bootstrap uncertainty."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
from typing import Mapping, Sequence

from .types import CounterfactualAuditResult, PotentialOutcomePrediction


@dataclass(frozen=True)
class PotentialOutcomeExample:
    features: Mapping[str, float]
    q_use: float
    q_drop: float
    event_id: str = ""
    group_id: str = ""

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
        self, feature_rows: Sequence[Mapping[str, float]], targets: Sequence[float],
        *, solver: str = "sgd",
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
        if solver == "ridge":
            import numpy as np

            values = np.asarray(matrix, dtype=float)
            centered = np.asarray(targets, dtype=float) - self.bias
            gram = values.T @ values / len(targets)
            gram += max(self.l2, 1e-8) * np.eye(values.shape[1])
            self.weights = np.linalg.solve(
                gram, values.T @ centered / len(targets)
            ).tolist()
            return
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

    def fit_pairwise(
        self,
        feature_rows: Sequence[Mapping[str, float]],
        targets: Sequence[float],
        group_ids: Sequence[str],
        *,
        weight: float,
        margin: float,
        min_gap: float,
        epochs: int,
    ) -> int:
        """Fine-tune with a within-task hinge-ranking objective.

        This is supervised contrastive ranking over counterfactual utilities:
        for two candidates retrieved for the same task, the candidate with the
        larger observed USE-minus-DROP effect should receive the larger score.
        Each optimization step includes the squared utility error as well as
        the ranking term, so ranking cannot drift without a regression cost.
        """

        if weight <= 0.0 or epochs <= 0:
            return 0
        vectors = [
            _standardize(
                [float(row.get(name, 0.0)) for name in self.feature_names],
                self.means,
                self.scales,
            )
            for row in feature_rows
        ]
        grouped: dict[str, list[int]] = {}
        for index, group_id in enumerate(group_ids):
            grouped.setdefault(str(group_id), []).append(index)
        pairs: list[tuple[int, int]] = []
        for indices in grouped.values():
            for left_position, left in enumerate(indices):
                for right in indices[left_position + 1 :]:
                    gap = float(targets[left]) - float(targets[right])
                    if gap == 0.0 or abs(gap) < min_gap:
                        continue
                    pairs.append((left, right) if gap > 0 else (right, left))
        if not pairs:
            return 0
        import numpy as np

        matrix = np.asarray(vectors, dtype=float)
        expected = np.asarray(targets, dtype=float)
        differences = matrix[[a for a, _ in pairs]] - matrix[[b for _, b in pairs]]
        coefficients = np.asarray(self.weights, dtype=float)
        # A dimension-aware step keeps rare standardized coordinates stable.
        step = min(self.learning_rate, 0.5 / max(1.0, float(np.square(matrix).sum(axis=1).mean())))
        for epoch in range(epochs):
            errors = self.bias + matrix @ coefficients - expected
            active = (differences @ coefficients < margin).astype(float)
            gradient = matrix.T @ errors / len(expected) + self.l2 * coefficients
            gradient -= weight * (differences.T @ active) / len(pairs)
            rate = step / math.sqrt(1.0 + epoch * 0.1)
            coefficients -= rate * gradient
            self.bias -= rate * float(errors.mean())
        self.weights = coefficients.tolist()
        return len(pairs)

    def to_dict(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2": self.l2,
            "seed": self.seed,
            "feature_names": list(self.feature_names),
            "means": self.means,
            "scales": self.scales,
            "weights": self.weights,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "_LinearRegressor":
        model = cls(
            learning_rate=float(payload["learning_rate"]),
            epochs=int(payload["epochs"]),
            l2=float(payload["l2"]),
            seed=int(payload["seed"]),
        )
        model.feature_names = tuple(str(name) for name in payload["feature_names"])
        model.means = [float(value) for value in payload["means"]]
        model.scales = [float(value) for value in payload["scales"]]
        model.weights = [float(value) for value in payload["weights"]]
        model.bias = float(payload["bias"])
        width = len(model.feature_names)
        if not (len(model.means) == len(model.scales) == len(model.weights) == width):
            raise ValueError("invalid regressor checkpoint dimensions")
        return model


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
        residual_noise_scale: float | None = None,
        direct_utility_weight: float = 0.0,
        contrastive_weight: float = 0.0,
        contrastive_margin: float = 0.05,
        contrastive_min_gap: float = 0.05,
        contrastive_epochs: int = 20,
        solver: str = "sgd",
        bootstrap_unit: str = "event",
    ):
        if ensemble_size < 2:
            raise ValueError("ensemble_size must be at least two")
        if min_samples < 2:
            raise ValueError("min_samples must be at least two")
        if utility_threshold < 0:
            raise ValueError("utility_threshold must be non-negative")
        if residual_noise_scale is not None and residual_noise_scale < 0:
            raise ValueError("residual_noise_scale must be non-negative")
        if not 0.0 <= direct_utility_weight <= 1.0:
            raise ValueError("direct_utility_weight must be in [0, 1]")
        if min(contrastive_weight, contrastive_margin, contrastive_min_gap) < 0:
            raise ValueError("contrastive settings must be non-negative")
        if contrastive_epochs < 0:
            raise ValueError("contrastive_epochs must be non-negative")
        if solver not in ("sgd", "ridge") or bootstrap_unit not in ("event", "task"):
            raise ValueError("invalid estimator solver or bootstrap unit")
        self.ensemble_size = ensemble_size
        self.min_samples = min_samples
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.utility_threshold = utility_threshold
        self.residual_noise_scale = (
            float(residual_noise_scale)
            if residual_noise_scale is not None
            else (1.0 if include_residual_noise else 0.0)
        )
        # Retain the old public field and checkpoint key for compatibility.
        self.include_residual_noise = self.residual_noise_scale > 0.0
        self.direct_utility_weight = float(direct_utility_weight)
        self.contrastive_weight = float(contrastive_weight)
        self.contrastive_margin = float(contrastive_margin)
        self.contrastive_min_gap = float(contrastive_min_gap)
        self.contrastive_epochs = int(contrastive_epochs)
        self.solver = solver
        self.bootstrap_unit = bootstrap_unit
        self._models: list[
            tuple[_LinearRegressor, _LinearRegressor, _LinearRegressor | None]
        ] = []
        self.contrastive_pairs = 0
        self._residual_use = 0.0
        self._residual_drop = 0.0
        self._residual_utility = 0.0
        self.training_samples = 0
        self.fitted = False

    def fit(self, examples: Sequence[PotentialOutcomeExample], *, progress: bool = False) -> bool:
        self.training_samples = len(examples)
        self._models = []
        self.contrastive_pairs = 0
        if len(examples) < self.min_samples:
            self.fitted = False
            return False
        rows = [example.features for example in examples]
        use_targets = [float(example.q_use) for example in examples]
        drop_targets = [float(example.q_drop) for example in examples]
        utility_targets = [use - drop for use, drop in zip(use_targets, drop_targets)]
        group_ids = [example.group_id or example.event_id for example in examples]
        groups: dict[str, list[int]] = {}
        for index, group in enumerate(group_ids):
            groups.setdefault(group or str(index), []).append(index)
        group_values = list(groups.values())
        randomizer = random.Random(self.seed)
        members = range(self.ensemble_size)
        if progress:
            from tqdm import tqdm

            members = tqdm(members, desc="Fit utility ensemble", unit="member", dynamic_ncols=True)
        for member in members:
            if self.bootstrap_unit == "task":
                draws = [randomizer.choice(group_values) for _ in group_values]
                indices = [index for draw in draws for index in draw]
                sampled_groups = [str(draw_id) for draw_id, draw in enumerate(draws) for _ in draw]
            else:
                indices = [randomizer.randrange(len(examples)) for _ in examples]
                sampled_groups = [group_ids[index] for index in indices]
            sampled_rows = [rows[index] for index in indices]
            use = _LinearRegressor(
                self.learning_rate, self.epochs, self.l2, self.seed + 2 * member
            )
            drop = _LinearRegressor(
                self.learning_rate, self.epochs, self.l2, self.seed + 2 * member + 1
            )
            use.fit(sampled_rows, [use_targets[index] for index in indices], solver=self.solver)
            drop.fit(sampled_rows, [drop_targets[index] for index in indices], solver=self.solver)
            direct: _LinearRegressor | None = None
            if self.direct_utility_weight > 0.0:
                direct = _LinearRegressor(
                    self.learning_rate,
                    self.epochs,
                    self.l2,
                    self.seed + 2 * self.ensemble_size + member,
                )
                sampled_targets = [utility_targets[index] for index in indices]
                direct.fit(sampled_rows, sampled_targets, solver=self.solver)
                self.contrastive_pairs += direct.fit_pairwise(
                    sampled_rows,
                    sampled_targets,
                    sampled_groups,
                    weight=self.contrastive_weight,
                    margin=self.contrastive_margin,
                    min_gap=self.contrastive_min_gap,
                    epochs=self.contrastive_epochs,
                )
            self._models.append((use, drop, direct))
        self.fitted = True

        use_predictions = [self._mean_prediction(row, outcome=0) for row in rows]
        drop_predictions = [self._mean_prediction(row, outcome=1) for row in rows]
        self._residual_use = _rmse(use_targets, use_predictions)
        self._residual_drop = _rmse(drop_targets, drop_predictions)
        if self.direct_utility_weight:
            self._residual_utility = _rmse(
                utility_targets, [self.predict(row).utility for row in rows]
            )
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
        use_values = [use.predict(features) for use, _, _ in self._models]
        drop_values = [drop.predict(features) for _, drop, _ in self._models]
        utility_values = [
            (1.0 - self.direct_utility_weight) * (use_value - drop_value)
            + self.direct_utility_weight * (
                direct.predict(features) if direct is not None else use_value - drop_value
            )
            for use_value, drop_value, (_, _, direct) in zip(
                use_values, drop_values, self._models
            )
        ]
        raw_q_use = sum(use_values) / len(use_values)
        raw_q_drop = sum(drop_values) / len(drop_values)
        utility = sum(utility_values) / len(utility_values)
        center = (raw_q_use + raw_q_drop) / 2.0
        q_use = center + utility / 2.0
        q_drop = center - utility / 2.0
        ensemble_variance = _sample_variance(utility_values)
        residual_variance = 0.0
        if self.residual_noise_scale > 0.0:
            noise = (self._residual_utility**2 if self.direct_utility_weight
                     else self._residual_use**2 + self._residual_drop**2)
            residual_variance = self.residual_noise_scale**2 * noise
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
            source=(
                ("bootstrap_hybrid_contrastive" if self.contrastive_weight else "bootstrap_hybrid")
                if self.direct_utility_weight > 0.0
                else "bootstrap_t_learner"
            ),
            calibrated=True,
            training_samples=self.training_samples,
        )

    def _mean_prediction(self, features: Mapping[str, float], outcome: int) -> float:
        values = [models[outcome].predict(features) for models in self._models]
        return sum(values) / len(values)

    def to_dict(self) -> dict:
        if not self.fitted:
            raise RuntimeError("cannot serialize an unfitted estimator")
        return {
            "schema": "amortized-utility-estimator-v2",
            "config": {
                "ensemble_size": self.ensemble_size,
                "min_samples": self.min_samples,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "l2": self.l2,
                "seed": self.seed,
                "utility_threshold": self.utility_threshold,
                "include_residual_noise": self.include_residual_noise,
                "residual_noise_scale": self.residual_noise_scale,
                "direct_utility_weight": self.direct_utility_weight,
                "contrastive_weight": self.contrastive_weight,
                "contrastive_margin": self.contrastive_margin,
                "contrastive_min_gap": self.contrastive_min_gap,
                "contrastive_epochs": self.contrastive_epochs,
                "solver": self.solver,
                "bootstrap_unit": self.bootstrap_unit,
            },
            "training_samples": self.training_samples,
            "residual_use": self._residual_use,
            "residual_drop": self._residual_drop,
            "residual_utility": self._residual_utility,
            "models": [
                {
                    "use": use.to_dict(),
                    "drop": drop.to_dict(),
                    "direct": direct.to_dict() if direct is not None else None,
                }
                for use, drop, direct in self._models
            ],
            "contrastive_pairs": self.contrastive_pairs,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "AmortizedUtilityEstimator":
        schema = payload.get("schema")
        if schema not in {
            "amortized-utility-estimator-v1",
            "amortized-utility-estimator-v2",
        }:
            raise ValueError("unsupported estimator checkpoint schema")
        config = dict(payload["config"])
        if schema == "amortized-utility-estimator-v1":
            config.update(
                direct_utility_weight=0.0,
                contrastive_weight=0.0,
            )
        estimator = cls(**config)
        estimator._models = [
            (
                _LinearRegressor.from_dict(item["use"]),
                _LinearRegressor.from_dict(item["drop"]),
                (
                    _LinearRegressor.from_dict(item["direct"])
                    if item.get("direct") is not None
                    else None
                ),
            )
            for item in payload["models"]
        ]
        if len(estimator._models) != estimator.ensemble_size:
            raise ValueError("checkpoint ensemble size does not match its config")
        estimator.training_samples = int(payload["training_samples"])
        estimator._residual_use = float(payload["residual_use"])
        estimator._residual_drop = float(payload["residual_drop"])
        estimator._residual_utility = float(payload.get("residual_utility", 0.0))
        estimator.contrastive_pairs = int(payload.get("contrastive_pairs", 0))
        estimator.fitted = True
        return estimator

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AmortizedUtilityEstimator":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


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
