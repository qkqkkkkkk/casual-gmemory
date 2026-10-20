"""Trainable BERT -> list self-attention -> one unbounded CI regression head.

Independent of st_estimator.py: no changes to the old ST architecture, losses,
checkpoints, threshold calibration or runtime dispatch are needed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .csm_st_data import task_text


@dataclass(frozen=True)
class CSMConfig:
    encoder: str = "bert-base-uncased"
    revision: str | None = None
    global_layers: int = 3
    heads: int = 8
    mlp_hidden: int = 256
    dropout: float = 0.1
    max_pair_tokens: int = 512
    pair_microbatch: int = 8
    gradient_checkpointing: bool = True

    def __post_init__(self):
        if min(self.global_layers, self.heads, self.mlp_hidden, self.pair_microbatch) < 1:
            raise ValueError("CSM dimensions/layers must be positive")
        if self.max_pair_tokens < 8 or not 0 <= self.dropout < 1:
            raise ValueError("invalid local encoder window/dropout")


class LocalGlobalCSM(nn.Module):
    def __init__(self, config: CSMConfig, encoder=None, *, local_files_only=False):
        super().__init__()
        if encoder is None:
            from transformers import AutoModel

            encoder = AutoModel.from_pretrained(config.encoder, revision=config.revision,
                trust_remote_code=False, local_files_only=local_files_only)
        self.config, self.encoder = config, encoder
        width = encoder.config.hidden_size
        if width % config.heads:
            raise ValueError("local hidden dimension must be divisible by global heads")
        if config.max_pair_tokens > encoder.config.max_position_embeddings:
            raise ValueError("local token window exceeds BERT positions")
        if config.gradient_checkpointing and hasattr(encoder, "gradient_checkpointing_enable"):
            encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        layer = nn.TransformerEncoderLayer(width, config.heads, dim_feedforward=4 * width,
            dropout=config.dropout, activation="gelu", batch_first=True)
        self.global_layer = nn.TransformerEncoder(layer, config.global_layers, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.Linear(width, config.mlp_hidden), nn.ReLU(),
                                    nn.Linear(config.mlp_hidden, 1))

    def forward(self, tokens, counts):
        total = sum(counts)
        if not counts or min(counts) < 1 or tokens["input_ids"].shape[0] != total:
            raise ValueError("empty or misaligned CSM lists")
        local = []
        for start in range(0, total, self.config.pair_microbatch):
            batch = {k: v[start:start + self.config.pair_microbatch] for k, v in tokens.items()}
            hidden = self.encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            local.append((hidden * mask).sum(1) / mask.sum(1).clamp_min(1))
        vectors = torch.cat(local)
        width = max(counts)
        pieces, masks, start = [], [], 0
        for count in counts:
            pieces.append(torch.nn.functional.pad(vectors[start:start + count], (0, 0, 0, width - count)))
            masks.append(torch.arange(width, device=vectors.device) < count)
            start += count
        valid = torch.stack(masks)
        global_vectors = self.global_layer(torch.stack(pieces), src_key_padding_mask=~valid)
        # No tanh: CE-derived contextual influence is not confined to [-1,1].
        scores = self.output(global_vectors).squeeze(-1).masked_fill(~valid, 0)
        return scores, global_vectors, valid


def tokenize_lists(tokenizer, rows, config, device):
    queries, memories, counts = [], [], []
    for row in rows:
        text = task_text(row["query"], row["state"])
        counts.append(len(row["candidates"]))
        queries.extend([text] * counts[-1])
        memories.extend(c["text"] for c in row["candidates"])
    if not counts or min(counts) < 1:
        raise ValueError("CSM cannot encode empty lists")
    tokens = tokenizer(queries, memories, padding=True, truncation="longest_first",
                       max_length=config.max_pair_tokens, return_tensors="pt")
    return {k: v.to(device) for k, v in tokens.items()}, counts


def contrastive_loss(representations, ci, *, temperature=1.0, positive_epsilon=0.1,
                     negative_epsilon=0.3):
    """Eq.7, same-list/global representations; exclude self and invalid pairs."""
    if temperature <= 0 or not 0 < positive_epsilon < negative_epsilon:
        raise ValueError("invalid InfoNCE temperature/CI thresholds")
    labels = torch.as_tensor(ci, device=representations.device, dtype=torch.float32)
    difference = (labels[:, None] - labels[None, :]).abs()
    nonself = ~torch.eye(len(ci), device=labels.device, dtype=torch.bool)
    positive = (difference < positive_epsilon) & nonself
    negative = (difference > negative_epsilon) & nonself
    logits = (representations.float() @ representations.float().T) / temperature
    losses = []
    for anchor in range(len(ci)):
        if not negative[anchor].any():
            continue
        neg = torch.logsumexp(logits[anchor, negative[anchor]], dim=0)
        for p in positive[anchor].nonzero().flatten():
            score = logits[anchor, p]
            losses.append(torch.logaddexp(score, neg) - score)
    return (torch.stack(losses).mean(), len(losses)) if losses else (representations.sum() * 0, 0)


class CSMPredictor:
    def __init__(self, network, tokenizer, device="cpu"):
        self.network, self.tokenizer, self.device = network.to(device).eval(), tokenizer, device

    def predict(self, query, state, candidates):
        if not candidates:
            return []
        tokens, counts = tokenize_lists(self.tokenizer,
            [{"query": query, "state": state, "candidates": candidates}], self.network.config, self.device)
        with torch.inference_mode():
            scores, _, _ = self.network(tokens, counts)
        values = scores[0, :counts[0]].float().cpu().tolist()
        if not all(math.isfinite(v) for v in values):
            raise ValueError("nonfinite predicted CI; refusing to apply a gate")
        return values

    @classmethod
    def load(cls, checkpoint, device="cpu"):
        import json
        from pathlib import Path
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from .compare_gate_runs import file_sha256

        checkpoint = Path(checkpoint)
        payload = json.loads(checkpoint.read_text())
        spec = payload["csm_st"]
        if spec.get("schema") != "csm-st-local-global-v1":
            raise ValueError("unsupported CSM checkpoint")
        if "weights.pt" not in spec["artifact_sha256"]:
            raise ValueError("CSM checkpoint lacks model weight hash")
        artifact = checkpoint.parent / spec["artifact_dir"]
        if artifact.resolve().parent != checkpoint.parent.resolve():
            raise ValueError("CSM model must be in its checkpoint's artifact directory")
        for name, expected in spec["artifact_sha256"].items():
            path = artifact / name
            if artifact.resolve() not in path.resolve().parents or file_sha256(path) != expected:
                raise ValueError(f"CSM artifact hash mismatch: {name}")
        config = CSMConfig(**spec["config"])
        # Loading a new estimator must not shift the host's paired-control RNG.
        with torch.random.fork_rng(devices=[]):
            encoder = AutoModel.from_config(AutoConfig.from_pretrained(artifact / "encoder", local_files_only=True),
                                            trust_remote_code=False)
            network = LocalGlobalCSM(config, encoder)
            network.load_state_dict(torch.load(artifact / "weights.pt", map_location="cpu", weights_only=True), strict=True)
        tokenizer = AutoTokenizer.from_pretrained(artifact / "tokenizer", local_files_only=True, trust_remote_code=False)
        return cls(network, tokenizer, device)
