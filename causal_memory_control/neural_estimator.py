"""Small nonlinear, three-head utility estimator with task bootstrap."""

from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

from .estimator import PotentialOutcomeExample
from .types import PotentialOutcomePrediction


class MLPUtilityEstimator:
    def __init__(self, *, ensemble_size=8, min_samples=8, hidden_sizes=(128, 64),
                 epochs=80, learning_rate=0.001, l2=0.0001, seed=23,
                 direct_utility_weight=0.5, contrastive_weight=0.1,
                 contrastive_margin=0.05, contrastive_min_gap=0.05,
                 residual_noise_scale=0.0, batch_tasks=16, device="cpu"):
        if ensemble_size < 2 or min_samples < 2 or epochs < 1 or batch_tasks < 1:
            raise ValueError("MLP requires ensemble>=2, min_samples>=2, positive epochs/batch_tasks")
        if not hidden_sizes or any(int(width) < 1 for width in hidden_sizes):
            raise ValueError("MLP hidden widths must be positive")
        if not 0 <= direct_utility_weight <= 1:
            raise ValueError("direct utility blend weight must be in [0,1]")
        if not all(math.isfinite(float(v)) and v >= 0 for v in (
            learning_rate, l2, contrastive_weight, contrastive_margin, contrastive_min_gap, residual_noise_scale
        )) or learning_rate == 0:
            raise ValueError("invalid MLP optimization settings")
        self.config = dict(ensemble_size=ensemble_size, min_samples=min_samples,
                           hidden_sizes=list(hidden_sizes), epochs=epochs, learning_rate=learning_rate,
                           l2=l2, seed=seed, direct_utility_weight=direct_utility_weight,
                           contrastive_weight=contrastive_weight, contrastive_margin=contrastive_margin,
                           contrastive_min_gap=contrastive_min_gap, residual_noise_scale=residual_noise_scale,
                           batch_tasks=batch_tasks)
        self.device, self.min_samples = device, min_samples
        self.direct_utility_weight, self.contrastive_weight = direct_utility_weight, contrastive_weight
        self.residual_noise_scale = residual_noise_scale
        self._models = []
        self.feature_names, self.means, self.scales = [], [], []
        self.training_samples, self.contrastive_pairs, self.residual_utility = 0, 0, 0.0
        self.fitted = False

    def _network(self):
        import torch.nn as nn

        layers = []
        width = len(self.feature_names)
        for hidden in self.config["hidden_sizes"]:
            layers.extend((nn.Linear(width, hidden), nn.ReLU()))
            width = hidden
        layers.append(nn.Linear(width, 3))
        return nn.Sequential(*layers).to(self.device)

    def _matrix(self, rows):
        import numpy as np

        values = np.asarray([[float(row.get(name, 0)) for name in self.feature_names] for row in rows], dtype="float32")
        if not np.isfinite(values).all():
            raise ValueError("MLP features must be finite")
        return (values - self.means) / self.scales

    @staticmethod
    def _outcomes(raw):
        import torch

        return torch.sigmoid(raw[:, :2]), torch.tanh(raw[:, 2])

    def fit(self, examples: Sequence[PotentialOutcomeExample], *, progress=False):
        import numpy as np
        import torch
        from tqdm import tqdm

        self.training_samples, self._models, self.fitted = len(examples), [], False
        self.contrastive_pairs, self.residual_utility = 0, 0.0
        if len(examples) < self.min_samples:
            return False
        if any(not (0 <= e.q_use <= 1 and 0 <= e.q_drop <= 1) for e in examples):
            raise ValueError("bounded MLP heads require outcomes in [0,1]")
        self.feature_names = sorted({name for e in examples for name in e.features})
        if not self.feature_names:
            raise ValueError("MLP needs at least one feature")
        values = np.asarray([[e.features.get(name, 0) for name in self.feature_names] for e in examples], dtype="float32")
        if not np.isfinite(values).all():
            raise ValueError("MLP features must be finite")
        self.means, self.scales = values.mean(axis=0), values.std(axis=0)
        self.scales = np.where(self.scales < 1e-6, 1.0, self.scales)
        x = torch.as_tensor(self._matrix([e.features for e in examples]), dtype=torch.float32, device=self.device)
        y = torch.tensor([[e.q_use, e.q_drop] for e in examples], dtype=torch.float32, device=self.device)
        utility = y[:, 0] - y[:, 1]
        grouped = {}
        for index, example in enumerate(examples):
            grouped.setdefault(example.group_id or example.event_id or str(index), []).append(index)
        groups = list(grouped.values())
        pairs_by_group = []
        for indices in groups:
            pairs = []
            for position, a in enumerate(indices):
                for b in indices[position + 1:]:
                    gap = examples[a].q_use - examples[a].q_drop - examples[b].q_use + examples[b].q_drop
                    if gap and abs(gap) >= self.config["contrastive_min_gap"]:
                        pairs.append((a, b) if gap > 0 else (b, a))
            pairs_by_group.append(pairs)
        randomizer = random.Random(self.config["seed"])
        with torch.random.fork_rng(devices=[] if self.device == "cpu" else [torch.device(self.device)]):
            for member in tqdm(range(self.config["ensemble_size"]), desc="Fit semantic MLP", unit="member", disable=not progress):
                torch.manual_seed(self.config["seed"] + member)
                model = self._network()
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["l2"])
                draws = [randomizer.randrange(len(groups)) for _ in groups]
                self.contrastive_pairs += sum(len(pairs_by_group[g]) for g in draws)
                epochs = tqdm(range(self.config["epochs"]), desc=f"MLP member {member+1}", unit="epoch", leave=False, disable=not progress)
                for _ in epochs:
                    randomizer.shuffle(draws)
                    model.train()
                    for start in range(0, len(draws), self.config["batch_tasks"]):
                        batch = draws[start:start + self.config["batch_tasks"]]
                        indices = [i for g in batch for i in groups[g]]
                        outcome_pred, direct_pred = self._outcomes(model(x[indices]))
                        loss = (outcome_pred - y[indices]).square().mean() + (direct_pred - utility[indices]).square().mean()
                        pair_positions, offset = [], 0
                        for g in batch:
                            lookup = {original: offset+j for j, original in enumerate(groups[g])}
                            pair_positions.extend((lookup[a], lookup[b]) for a, b in pairs_by_group[g])
                            offset += len(groups[g])
                        if pair_positions and self.contrastive_weight:
                            a, b = zip(*pair_positions)
                            loss = loss + self.contrastive_weight * torch.relu(
                                self.config["contrastive_margin"] - direct_pred[list(a)] + direct_pred[list(b)]).mean()
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"non-finite MLP loss in member {member}")
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                self._models.append(model.eval())
        self.fitted = True
        squared_error = 0.0
        for start in range(0,len(examples),256):
            batch = examples[start:start+256]
            predictions = self.predict_many([e.features for e in batch])
            squared_error += sum((p.utility-e.q_use+e.q_drop)**2 for p,e in zip(predictions,batch))
        self.residual_utility = math.sqrt(squared_error/len(examples))
        return True

    def predict_many(self, rows: Sequence[Mapping[str, float]]) -> list[PotentialOutcomePrediction]:
        import torch

        if not rows:
            return []
        if not self.fitted:
            return [PotentialOutcomePrediction(0, 0, 0, "neutral", 1, "cold_start", False, self.training_samples) for _ in rows]
        x = torch.as_tensor(self._matrix(rows), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            outcomes, utilities = [], []
            for model in self._models:
                q, direct = self._outcomes(model(x))
                outcomes.append(q)
                utilities.append((1-self.direct_utility_weight)*(q[:,0]-q[:,1])+self.direct_utility_weight*direct)
            mean_q = torch.stack(outcomes).mean(0).cpu().tolist()
            values = torch.stack(utilities)
            means = values.mean(0).cpu().tolist()
            variances = values.var(0, unbiased=True).cpu().tolist()
        result = []
        for q, u, variance in zip(mean_q, means, variances):
            center = sum(q)/2
            sigma = math.sqrt(max(0, variance)+(self.residual_noise_scale*self.residual_utility)**2)
            result.append(PotentialOutcomePrediction(center+u/2, center-u/2, u,
                "positive" if u>0 else "negative" if u<0 else "neutral", sigma,
                "bootstrap_semantic_mlp", True, self.training_samples))
        return result

    def predict(self, features):
        return self.predict_many([features])[0]

    def to_dict(self):
        if not self.fitted:
            raise RuntimeError("cannot save unfitted MLP")
        return {"schema": "mlp-utility-estimator-v1", "config": self.config,
                "feature_names": self.feature_names, "means": self.means.tolist(), "scales": self.scales.tolist(),
                "training_samples": self.training_samples, "contrastive_pairs": self.contrastive_pairs,
                "residual_utility": self.residual_utility,
                "models": [{name: value.detach().cpu().tolist() for name, value in m.state_dict().items()} for m in self._models]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, device="cpu"):
        import numpy as np
        import torch

        if payload.get("schema") != "mlp-utility-estimator-v1":
            raise ValueError("unsupported MLP checkpoint")
        model = cls(**payload["config"], device=device)
        model.feature_names = list(payload["feature_names"])
        model.means, model.scales = np.asarray(payload["means"]), np.asarray(payload["scales"])
        if not (len(model.feature_names)==len(model.means)==len(model.scales)) or not np.isfinite(model.means).all() or not (np.isfinite(model.scales).all() and (model.scales>0).all()):
            raise ValueError("invalid MLP feature standardizer")
        # Loading learned weights must not perturb the downstream MAS sampling.
        with torch.random.fork_rng(devices=[] if device == "cpu" else [torch.device(device)]):
            for raw in payload["models"]:
                network = model._network()
                state = {k:torch.tensor(v, dtype=torch.float32, device=device) for k,v in raw.items()}
                if not all(torch.isfinite(v).all() for v in state.values()):
                    raise ValueError("non-finite MLP weights")
                network.load_state_dict(state)
                model._models.append(network.eval())
        if len(model._models)!=model.config["ensemble_size"]:
            raise ValueError("MLP ensemble count mismatch")
        model.training_samples = int(payload["training_samples"])
        model.contrastive_pairs = int(payload.get("contrastive_pairs",0))
        model.residual_utility = float(payload.get("residual_utility",0))
        model.fitted = True
        return model
