"""Features for amortized event-level potential-outcome prediction."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
import re
from typing import Any, Mapping, Optional, Protocol, Sequence

from .types import MemoryCandidate, MemoryUseEvent


class TextEmbedder(Protocol):
    def embed(self, text: str) -> Sequence[float]:
        ...


class HashEmbedder:
    """Stable no-dependency fallback; production can reuse G-Memory embeddings."""

    def __init__(self, dimensions: int = 256):
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token, count in Counter(re.findall(r"\w+", (text or "").lower())).items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            vector[value % self.dimensions] += (1.0 if value & 256 else -1.0) * count
        return _normalize(vector)


class GMemoryEmbeddingAdapter:
    def __init__(self, embedding_function: Any):
        self.embedding_function = embedding_function

    def embed(self, text: str) -> Sequence[float]:
        if hasattr(self.embedding_function, "embed_query"):
            return self.embedding_function.embed_query(text)
        if callable(self.embedding_function):
            return self.embedding_function(text)
        raise TypeError("embedding function must be callable or expose embed_query")


class EventFeatureBuilder:
    """Build x-features while keeping treatment z out of the representation."""

    def __init__(self, embedder: Optional[TextEmbedder] = None):
        self.embedder = embedder or HashEmbedder()

    def build(self, event: MemoryUseEvent) -> dict[str, float]:
        task = self.embedder.embed(event.query)
        state = self.embedder.embed(event.task_state)
        role = self.embedder.embed(event.receiver_role)
        memory = self.embedder.embed(event.memory.content)
        features: dict[str, float] = {
            "sim_task_memory": _cosine(task, memory),
            "sim_state_memory": _cosine(state, memory),
            "sim_role_memory": _cosine(role, memory),
            "candidate_count": float(len(event.candidate_set)),
            "recipient_count": float(len(event.recipient_contexts)),
            "memory_observation_count": float(event.observation_count),
            "memory_content_length": float(len(event.memory.content)),
            "memory_token_count": float(len(re.findall(r"\w+", event.memory.content))),
            "candidate_redundancy_max": self._redundancy(event, memory),
            f"memory_type::{_safe(event.memory.memory_type)}": 1.0,
            f"receiver_role::{_safe(event.receiver_role)}": 1.0,
        }
        _add_optional(features, "reliability_prior", event.reliability_prior)
        retrieval = event.retrieval
        if retrieval is None:
            for name in ("rank", "similarity", "distance", "hop", "path_weight"):
                _add_optional(features, f"retrieval_{name}", None)
            features["retrieval_source::missing"] = 1.0
        else:
            _add_optional(features, "retrieval_rank", retrieval.rank)
            _add_optional(features, "retrieval_similarity", retrieval.similarity)
            _add_optional(features, "retrieval_distance", retrieval.distance)
            _add_optional(features, "retrieval_hop", retrieval.hop)
            _add_optional(features, "retrieval_path_weight", retrieval.path_weight)
            features[f"retrieval_source::{_safe(retrieval.source)}"] = 1.0
            _add_numeric_mapping(features, "retrieval_meta", retrieval.metadata)

        _add_numeric_mapping(features, "memory_meta", event.memory.metadata)
        _add_numeric_mapping(features, "task_meta", event.task_metadata)
        for name in (
            "source_task_type",
            "source_model_id",
            "source_prompt_version",
            "memory_schema_version",
        ):
            value = event.memory.metadata.get(name)
            if value is not None:
                features[f"{name}::{_safe(value)}"] = 1.0
        for name in ("task_type", "game_name", "difficulty", "graph_type"):
            value = event.task_metadata.get(name)
            if value is not None:
                features[f"current_{name}::{_safe(value)}"] = 1.0
        return features

    def _redundancy(
        self, event: MemoryUseEvent, target_vector: Sequence[float]
    ) -> float:
        similarities = [
            _cosine(target_vector, self.embedder.embed(candidate.content))
            for candidate in event.candidate_set
            if candidate.memory_id != event.memory.memory_id
        ]
        return max(similarities, default=0.0)


def _add_numeric_mapping(
    features: dict[str, float], prefix: str, values: Mapping[str, Any]
) -> None:
    for key, value in values.items():
        if isinstance(value, bool):
            features[f"{prefix}::{_safe(key)}"] = float(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            features[f"{prefix}::{_safe(key)}"] = float(value)


def _add_optional(features: dict[str, float], name: str, value: Any) -> None:
    if isinstance(value, bool):
        numeric: Optional[float] = float(value)
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        numeric = float(value)
    else:
        numeric = None
    features[name] = numeric if numeric is not None else 0.0
    features[f"{name}_missing"] = float(numeric is None)


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else list(vector)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _safe(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value).strip().lower()) or "unknown"

