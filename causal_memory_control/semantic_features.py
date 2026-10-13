"""Frozen sentence embeddings shared by estimator fitting and deployment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Sequence

from .features import TeamExposureFeatureBuilder
from .types import TeamMemoryExposureEvent


class SentenceTransformerEmbedder:
    """Lazy frozen encoder with a persistent, model-specific vector cache.

    Long states are split into tokenizer-sized windows and averaged, rather
    than silently truncating the latter HotpotQA context paragraphs.
    """

    def __init__(self, model: str, *, revision: str | None = None,
                 cache_path: str | Path | None = None, device: str = "cpu",
                 chunk_tokens: int = 192, batch_size: int = 32,
                 expected_dimensions: int | None = None):
        if chunk_tokens < 8 or batch_size < 1:
            raise ValueError("semantic chunk_tokens >= 8 and batch_size >= 1 required")
        self.model_name, self.revision, self.device = model, revision, device
        self.chunk_tokens, self.batch_size = chunk_tokens, batch_size
        self.expected_dimensions = expected_dimensions
        self._model = None
        self._vectors: dict[str, list[float]] = {}
        self._db = None
        if cache_path is not None:
            path = Path(cache_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(path), timeout=60)
            self._db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)")
            self._db.commit()

    def config(self) -> dict[str, Any]:
        return {"type": "sentence-transformer-chunk-mean-v1", "model": self.model_name,
                "revision": self.revision, "chunk_tokens": self.chunk_tokens,
                "dimensions": self.expected_dimensions or (len(next(iter(self._vectors.values()))) if self._vectors else None)}

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch

            with torch.random.fork_rng(devices=[] if self.device == "cpu" else [torch.device(self.device)]):
                self._model = SentenceTransformer(self.model_name, revision=self.revision,
                                                  device=self.device, trust_remote_code=False)
            self._model.eval()
        return self._model

    def _key(self, text: str) -> str:
        identity = {key:value for key,value in self.config().items() if key!="dimensions"}
        return hashlib.sha256(json.dumps({"encoder": identity, "text": text},
                                         sort_keys=True).encode()).hexdigest()

    def embed(self, text: str) -> Sequence[float]:
        self.precompute([text])
        return self._vectors[self._key(text)]

    def precompute(self, texts: Sequence[str]) -> None:
        pending: dict[str, str] = {}
        for text in texts:
            key = self._key(text)
            if key in self._vectors or key in pending:
                continue
            cached = self._db.execute("SELECT vector FROM embeddings WHERE key=?", (key,)).fetchone() if self._db else None
            if cached:
                vector = json.loads(cached[0])
                self._validate_vector(vector)
                self._vectors[key] = vector
            else:
                pending[key] = text
        if not pending:
            return
        model = self._load()
        tokenizer = model.tokenizer
        width = min(self.chunk_tokens, int(model.max_seq_length) - tokenizer.num_special_tokens_to_add(pair=False))
        if width < 1:
            raise ValueError("semantic encoder has no usable context window")
        for key, text in pending.items():
            ids = tokenizer.encode(text, add_special_tokens=False)
            chunks = [tokenizer.decode(ids[start:start + width], skip_special_tokens=True)
                      for start in range(0, len(ids), width)] or [""]
            vectors = model.encode(chunks, batch_size=self.batch_size,
                                   normalize_embeddings=True, show_progress_bar=False,
                                   convert_to_numpy=True)
            average = vectors.mean(axis=0).tolist()
            norm = math.sqrt(sum(value * value for value in average))
            vector = [float(value / norm) for value in average] if norm else average
            self._validate_vector(vector)
            self._vectors[key] = vector
            if self._db:
                self._db.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (key, json.dumps(vector)))
        if self._db:
            self._db.commit()

    def _validate_vector(self, vector):
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("semantic encoder produced invalid vector")
        dimensions = self.expected_dimensions or (len(next(iter(self._vectors.values()))) if self._vectors else len(vector))
        if len(vector)!=dimensions:
            raise ValueError("semantic encoder dimension differs from checkpoint or cache")

    def close(self):
        if self._db:
            self._db.close()
            self._db = None


class SemanticTeamFeatureBuilder(TeamExposureFeatureBuilder):
    """Full sentence coordinates and interactions, plus v2 provenance fields."""

    version = 3

    def __init__(self, embedder):
        super().__init__(embedder, version=2)
        self.version = 3

    def build(self, event: TeamMemoryExposureEvent) -> dict[str, float]:
        if hasattr(self.embedder, "precompute"):
            self.embedder.precompute([event.query, event.task_state, event.memory.content,
                                     *(peer.content for peer in event.candidate_set)])
        # v2's hash-coordinate folding must not be applied to semantic vectors.
        features = super().build(event)
        from .features import _add_insight_support_features

        _add_insight_support_features(features, event.memory.metadata)
        query, state, memory = map(self.embedder.embed, (event.query, event.task_state, event.memory.content))
        if len({len(query), len(state), len(memory)}) != 1:
            raise ValueError("semantic vector dimensions differ")
        peers = [self.embedder.embed(peer.content) for peer in event.candidate_set
                 if peer.memory_id != event.memory.memory_id]
        for index, (q, s, m) in enumerate(zip(query, state, memory)):
            for name, value in {
                "query": q, "state": s, "memory": m,
                "query_memory_product": q*m, "state_memory_product": s*m,
                "query_memory_difference": abs(q-m), "state_memory_difference": abs(s-m),
                "peer_mean": sum(peer[index] for peer in peers) / len(peers) if peers else 0.0,
            }.items():
                features[f"semantic_{name}::{index}"] = float(value)
        return features
