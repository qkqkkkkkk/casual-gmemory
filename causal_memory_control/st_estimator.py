"""ST-loss adaptation of Influence Guided Context Selection, eqs. 6--7.

Frozen existing semantic features -> learned representation -> one direct U head.
Not the paper's BERT/list-attention architecture or generator-loss CI labels.
No E2E training or context recombination with unmeasured causal labels.
"""

from __future__ import annotations

from collections import Counter
import math
import random
from typing import Mapping, Sequence

from .estimator import PotentialOutcomeExample
from .types import PotentialOutcomePrediction


SCHEMA = "st-direct-utility-estimator-v1"

# Shared by the trainer and ST-only experiment runner. Prefixing these options
# keeps every existing estimator's CLI defaults unchanged.
ST_OPTIONS = {
    "hidden_sizes": (str, "128,64"), "epochs": (int, 10),
    "learning_rate": (float, 0.001), "batch_tasks": (int, 16),
    "device": (str, "cpu"), "beta": (float, 0.1),
    "temperature": (float, 1.0), "positive_epsilon": (float, 0.05),
    "negative_epsilon": (float, 0.25), "rarity_alpha": (float, 10.0),
    "rarity_bin_width": (float, 0.25), "max_rarity_weight": (float, 20.0),
    "hard_threshold": (float, 0.1),
}


def add_st_arguments(parser):
    for name, (kind, default) in ST_OPTIONS.items():
        parser.add_argument("--st-" + name.replace("_", "-"), type=kind, default=default)


def st_settings(args):
    settings = {key: getattr(args, "st_" + key) for key in ST_OPTIONS}
    try:
        settings["hidden_sizes"] = [int(v) for v in settings["hidden_sizes"].split(",")]
    except ValueError as exc:
        raise ValueError("--st-hidden-sizes must contain positive integers") from exc
    return settings


def rarity_weights(utilities, *, alpha=10.0, bin_width=0.25, max_weight=20.0):
    """Fit-only list rarity distribution; normalize clipped inverse frequencies.

    Paper does not specify frequency discretization. Equal-width bins and the
    cap are explicit adaptations for our bounded reward-difference labels.
    """
    rates = []
    for values in utilities:
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        rates.append(mean + alpha * std)
    bins = [math.floor(value / bin_width) for value in rates]
    counts = Counter(bins)
    raw = [min(max_weight, len(bins) / counts[b]) for b in bins]
    normalizer = sum(raw) / len(raw)
    return rates, [weight / normalizer for weight in raw], dict(sorted(counts.items()))


def contrastive_loss(representations, utilities, *, temperature=1.0,
                     positive_epsilon=0.05, negative_epsilon=0.25):
    """Same-list, no-self InfoNCE: average eq. 7 over all valid positives.

    Each positive has its own denominator (that positive + all negatives).
    Anchors lacking either a real positive or a negative contribute nothing.
    Dot products are intentionally not normalized, matching the paper.
    """
    import torch

    gaps = (utilities[:, None] - utilities[None, :]).abs()
    off_diagonal = ~torch.eye(len(utilities), dtype=torch.bool, device=utilities.device)
    positives = (gaps < positive_epsilon) & off_diagonal
    negatives = (gaps > negative_epsilon) & off_diagonal
    similarities = representations @ representations.T / temperature
    terms, pairs = [], 0
    for anchor in range(len(utilities)):
        positive = similarities[anchor][positives[anchor]]
        negative = similarities[anchor][negatives[anchor]]
        if positive.numel() and negative.numel():
            terms.append((torch.logaddexp(positive, torch.logsumexp(negative, dim=0)) - positive).mean())
            pairs += positive.numel()
    if not terms:
        return representations.sum() * 0.0, 0, 0
    return torch.stack(terms).mean(), len(terms), pairs


class STUtilityEstimator:
    def __init__(self, *, ensemble_size=12, min_samples=8, hidden_sizes=(128, 64),
                 epochs=10, learning_rate=0.001, l2=0.0001, seed=23, batch_tasks=16,
                 beta=0.1, temperature=1.0, positive_epsilon=0.05,
                 negative_epsilon=0.25, rarity_alpha=10.0, rarity_bin_width=0.25,
                 max_rarity_weight=20.0, hard_threshold=0.1,
                 residual_noise_scale=0.0, device="cpu"):
        if (ensemble_size < 2 or min_samples < 2 or epochs < 1 or batch_tasks < 1
                or not hidden_sizes or any(int(v) != v or v < 1 for v in hidden_sizes)):
            raise ValueError("ST requires ensemble>=2, min_samples>=2 and positive widths/epochs/batch_tasks")
        positive = (learning_rate, temperature, positive_epsilon, rarity_bin_width, max_rarity_weight)
        nonnegative = (l2, beta, rarity_alpha, residual_noise_scale)
        if (not all(math.isfinite(v) and v > 0 for v in positive)
                or not all(math.isfinite(v) and v >= 0 for v in nonnegative)
                or not math.isfinite(hard_threshold)
                or not positive_epsilon < negative_epsilon <= 2
                or max_rarity_weight < 1):
            raise ValueError("invalid ST loss/optimization settings (require 0 < epsilon+ < epsilon- <= 2)")
        self.config = dict(ensemble_size=ensemble_size, min_samples=min_samples,
            hidden_sizes=list(hidden_sizes), epochs=epochs, learning_rate=learning_rate,
            l2=l2, seed=seed, batch_tasks=batch_tasks, beta=beta, temperature=temperature,
            positive_epsilon=positive_epsilon, negative_epsilon=negative_epsilon,
            rarity_alpha=rarity_alpha, rarity_bin_width=rarity_bin_width,
            max_rarity_weight=max_rarity_weight, hard_threshold=hard_threshold,
            residual_noise_scale=residual_noise_scale)
        self.device = device
        self.feature_names, self.means, self.scales, self._models = [], [], [], []
        self.fitted = False
        self.training_samples, self.contrastive_pairs, self.residual_utility = 0, 0, 0.0
        self.training_diagnostics, self.loss_history = {}, []

    def _network(self):
        import torch.nn as nn

        layers, width = [], len(self.feature_names)
        for hidden in self.config["hidden_sizes"]:
            layers.extend((nn.Linear(width, hidden), nn.ReLU()))
            width = hidden
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers).to(self.device)

    def _matrix(self, rows):
        import numpy as np
        from .st_data import feature_matrix

        values = feature_matrix(rows, self.feature_names)
        if not np.isfinite(values).all():
            raise ValueError("ST features must be finite")
        return (values - self.means) / self.scales

    def fit(self, examples: Sequence[PotentialOutcomeExample], *, progress=False):
        import numpy as np
        import torch
        from tqdm import tqdm

        self.training_samples, self._models, self.fitted = len(examples), [], False
        self.contrastive_pairs, self.residual_utility = 0, 0.0
        self.loss_history, self.training_diagnostics = [], {}
        if len(examples) < self.config["min_samples"]:
            return False
        if any(not (0 <= e.q_use <= 1 and 0 <= e.q_drop <= 1) for e in examples):
            raise ValueError("ST outcomes must be finite values in [0,1]")
        if any(not e.group_id for e in examples):
            raise ValueError("ST requires task group_id for every example; never mix candidates across tasks")
        self.feature_names = sorted({name for e in examples for name in e.features})
        if not self.feature_names:
            raise ValueError("ST requires features")
        from .st_data import feature_matrix

        values = feature_matrix([e.features for e in examples], self.feature_names)
        if not np.isfinite(values).all():
            raise ValueError("ST features must be finite")
        self.means, self.scales = values.mean(0), values.std(0)
        self.scales = np.where(self.scales < 1e-6, 1.0, self.scales)
        x = torch.as_tensor((values - self.means) / self.scales, dtype=torch.float32, device=self.device)
        del values
        y = torch.tensor([e.q_use - e.q_drop for e in examples], dtype=torch.float32, device=self.device)
        grouped = {}
        for index, example in enumerate(examples):
            grouped.setdefault(example.group_id, []).append(index)
        groups = list(grouped.values())
        utilities = [[examples[i].q_use - examples[i].q_drop for i in group] for group in groups]
        rates, weights, counts = rarity_weights(utilities, alpha=self.config["rarity_alpha"],
            bin_width=self.config["rarity_bin_width"], max_weight=self.config["max_rarity_weight"])
        hard = [rate > self.config["hard_threshold"] for rate in rates]
        valid_lists = anchors = 0
        for indices, is_hard in zip(groups, hard):
            if is_hard:
                _, count, pairs = contrastive_loss(torch.zeros((len(indices), 1), device=self.device), y[indices],
                    temperature=self.config["temperature"], positive_epsilon=self.config["positive_epsilon"],
                    negative_epsilon=self.config["negative_epsilon"])
                valid_lists += bool(count)
                anchors += count
                self.contrastive_pairs += pairs
        self.training_diagnostics = {
            "fit_task_groups": len(groups), "frequency_scope": "fit_tasks_only",
            "rarity_bin_counts": {str(k): v for k, v in counts.items()},
            "rarity_min": min(rates), "rarity_max": max(rates),
            "weight_min": min(weights), "weight_max": max(weights),
            "hard_task_groups": sum(hard), "contrastive_task_groups": valid_lists,
            "contrastive_anchors": anchors, "contrastive_positive_pairs": self.contrastive_pairs,
            "utility_negative": sum(v < 0 for row in utilities for v in row),
            "utility_zero": sum(v == 0 for row in utilities for v in row),
            "utility_positive": sum(v > 0 for row in utilities for v in row),
            "contrastive_warning": None if valid_lists else "No valid same-task positive/negative anchors; contrastive term is zero, not fabricated",
            "architecture": "frozen_semantic_features_mlp_single_direct_u_head",
            "q_use_q_drop_fields": "derived 0.5 +/- U/2; not separately predicted outcomes",
        }
        if progress and not valid_lists:
            print("WARNING: ST has no valid contrastive task groups; inspect training diagnostics.", flush=True)
        randomizer = random.Random(self.config["seed"])
        devices = [] if self.device == "cpu" else [torch.device(self.device)]
        with torch.random.fork_rng(devices=devices):
            for member in tqdm(range(self.config["ensemble_size"]), desc="Fit ST direct-U", unit="member", disable=not progress):
                torch.manual_seed(self.config["seed"] + member)
                model = self._network()
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["l2"])
                draws = [randomizer.randrange(len(groups)) for _ in groups]
                epochs = tqdm(range(self.config["epochs"]), desc=f"ST member {member+1}", unit="epoch", leave=False, disable=not progress)
                for epoch in epochs:
                    randomizer.shuffle(draws)
                    sums, batches = [0.0, 0.0, 0.0], 0
                    model.train()
                    for start in range(0, len(draws), self.config["batch_tasks"]):
                        batch = draws[start:start + self.config["batch_tasks"]]
                        indices = [i for g in batch for i in groups[g]]
                        representation = model[:-1](x[indices])
                        predicted = torch.tanh(model[-1](representation).squeeze(-1))
                        regressions, contrasts, offset = [], [], 0
                        for g in batch:
                            end = offset + len(groups[g])
                            target = y[groups[g]]
                            regressions.append(weights[g] * (predicted[offset:end] - target).square().mean())
                            if hard[g] and self.config["beta"]:
                                term, count, _ = contrastive_loss(representation[offset:end], target,
                                    temperature=self.config["temperature"], positive_epsilon=self.config["positive_epsilon"],
                                    negative_epsilon=self.config["negative_epsilon"])
                                if count:
                                    contrasts.append(term)
                            offset = end
                        mse = torch.stack(regressions).mean()
                        cts = torch.stack(contrasts).mean() if contrasts else representation.sum() * 0
                        loss = mse + self.config["beta"] * cts
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"non-finite ST loss: member={member}, epoch={epoch}")
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                        for index, value in enumerate((loss, mse, cts)):
                            sums[index] += value.detach().item()
                        batches += 1
                    self.loss_history.append(dict(member=member, epoch=epoch + 1,
                        **dict(zip(("loss", "weighted_mse", "infonce"), (v / batches for v in sums)))))
                    epochs.set_postfix(loss=f"{sums[0]/batches:.5f}")
                self._models.append(model.eval())
        self.fitted = True
        error = 0.0
        for start in range(0, len(examples), 256):
            batch = examples[start:start+256]
            error += sum((p.utility - e.q_use + e.q_drop) ** 2
                for p, e in zip(self.predict_many([e.features for e in batch]), batch))
        self.residual_utility = math.sqrt(error / len(examples))
        return True

    def predict_many(self, rows: Sequence[Mapping[str, float]]):
        import torch

        if not rows:
            return []
        if not self.fitted:
            return [PotentialOutcomePrediction(0, 0, 0, "neutral", 1, "cold_start", False, self.training_samples) for _ in rows]
        x = torch.as_tensor(self._matrix(rows), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            values = torch.stack([torch.tanh(m(x).squeeze(-1)) for m in self._models])
            means, variances = values.mean(0).cpu().tolist(), values.var(0, unbiased=True).cpu().tolist()
        return [PotentialOutcomePrediction(0.5 + u/2, 0.5 - u/2, u,
            "positive" if u > 0 else "negative" if u < 0 else "neutral",
            math.sqrt(max(0, variance) + (self.config["residual_noise_scale"] * self.residual_utility) ** 2),
            "bootstrap_semantic_st_direct_u", True, self.training_samples) for u, variance in zip(means, variances)]

    def predict(self, features):
        return self.predict_many([features])[0]

    def to_dict(self):
        if not self.fitted:
            raise RuntimeError("cannot save unfitted ST estimator")
        return dict(schema=SCHEMA, config=self.config, feature_names=self.feature_names,
            means=self.means.tolist(), scales=self.scales.tolist(), training_samples=self.training_samples,
            contrastive_pairs=self.contrastive_pairs, residual_utility=self.residual_utility,
            training_diagnostics=self.training_diagnostics, loss_history=self.loss_history,
            models=[{k: v.detach().cpu().tolist() for k, v in m.state_dict().items()} for m in self._models])

    @classmethod
    def from_dict(cls, payload, *, device="cpu"):
        import numpy as np
        import torch

        if payload.get("schema") != SCHEMA:
            raise ValueError("unsupported ST checkpoint")
        model = cls(**payload["config"], device=device)
        model.feature_names = list(payload["feature_names"])
        model.means, model.scales = np.asarray(payload["means"], dtype="float32"), np.asarray(payload["scales"], dtype="float32")
        if (not model.feature_names or len(set(model.feature_names)) != len(model.feature_names)
                or model.means.shape != (len(model.feature_names),) or model.scales.shape != model.means.shape
                or not np.isfinite(model.means).all() or not np.isfinite(model.scales).all() or not (model.scales > 0).all()):
            raise ValueError("invalid ST feature standardizer")
        with torch.random.fork_rng(devices=[] if device == "cpu" else [torch.device(device)]):
            for raw in payload["models"]:
                network = model._network()
                state = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in raw.items()}
                if not all(torch.isfinite(v).all() for v in state.values()):
                    raise ValueError("non-finite ST weights")
                network.load_state_dict(state)
                model._models.append(network.eval())
        if len(model._models) != model.config["ensemble_size"]:
            raise ValueError("ST ensemble count mismatch")
        model.training_samples = int(payload["training_samples"])
        model.contrastive_pairs = int(payload["contrastive_pairs"])
        model.residual_utility = float(payload["residual_utility"])
        if model.training_samples < model.config["min_samples"] or not math.isfinite(model.residual_utility) or model.residual_utility < 0:
            raise ValueError("invalid ST training statistics")
        model.training_diagnostics, model.loss_history = payload["training_diagnostics"], payload["loss_history"]
        model.fitted = True
        return model
