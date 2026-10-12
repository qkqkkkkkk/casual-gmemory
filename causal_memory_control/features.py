"""Features for amortized event-level potential-outcome prediction."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
import re
from typing import Any, Mapping, Optional, Protocol, Sequence

from .types import MemoryCandidate, MemoryUseEvent, TeamMemoryExposureEvent


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


class TeamExposureFeatureBuilder:
    """Role-free features for expected team-level exposure utility."""

    def __init__(self, embedder: Optional[TextEmbedder] = None, *, version: int = 1):
        if version not in (1, 2):
            raise ValueError("unsupported team feature version")
        self.embedder = embedder or HashEmbedder()
        self.version = version

    def build(self, event: TeamMemoryExposureEvent) -> dict[str, float]:
        task = self.embedder.embed(event.query)
        state = self.embedder.embed(event.task_state)
        memory = self.embedder.embed(event.memory.content)
        peer_similarities = [
            _cosine(memory, self.embedder.embed(candidate.content))
            for candidate in event.candidate_set
            if candidate.memory_id != event.memory.memory_id
        ]
        features: dict[str, float] = {
            "sim_task_memory": _cosine(task, memory),
            "sim_state_memory": _cosine(state, memory),
            "sim_task_state": _cosine(task, state),
            "candidate_count": float(len(event.candidate_set)),
            "exposure_count": float(event.exposure_count),
            "memory_content_length": float(len(event.memory.content)),
            "memory_token_count": float(len(re.findall(r"\w+", event.memory.content))),
            "candidate_redundancy_max": max(peer_similarities, default=0.0),
            f"memory_type::{_safe(event.memory.memory_type)}": 1.0,
        }
        if self.version == 2:
            _add_interaction_features(
                features,
                query=event.query,
                state_text=event.task_state,
                memory_text=event.memory.content,
                task_vector=task,
                state_vector=state,
                memory_vector=memory,
            )
            _add_insight_support_features(features, event.memory.metadata)
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
        for name in ("task_type", "game_name", "difficulty", "graph_type", "mas_type"):
            value = event.task_metadata.get(name)
            if value is not None:
                features[f"current_{name}::{_safe(value)}"] = 1.0
        return features


def task_context_metadata(task: Mapping[str, Any]) -> dict[str, Any]:
    """Share pre-outcome task metadata between diagnostic and runtime events."""
    excluded = {
        "task_main", "task_description", "claim", "claim_id", "id", "task_id",
        "task", "answer", "label", "gold_label", "gold_answer", "few_shots",
        "supporting_facts", "gold_evidence_page_sets",
    }
    return {
        key: value for key, value in task.items()
        if key not in excluded and isinstance(value, (str, int, float, bool))
    }


def _add_numeric_mapping(
    features: dict[str, float], prefix: str, values: Mapping[str, Any]
) -> None:
    for key, value in values.items():
        if isinstance(value, bool):
            features[f"{prefix}::{_safe(key)}"] = float(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            features[f"{prefix}::{_safe(key)}"] = float(value)


def _add_interaction_features(
    features: dict[str, float],
    *,
    query: str,
    state_text: str,
    memory_text: str,
    task_vector: Sequence[float],
    state_vector: Sequence[float],
    memory_vector: Sequence[float],
    dimensions: int = 64,
) -> None:
    """Expose richer query-memory interactions to the lightweight regressor.

    Cosine similarity compresses an entire semantic relation to one scalar.
    Elementwise products preserve which embedding dimensions agree, while
    lexical overlap gives a transparent signal for names and answer entities.
    """

    # Fold every coordinate; never silently discard the final 3/4 of a hash
    # embedding. These are lexical hash interactions, not pretrained semantics.
    def fold(vector: Sequence[float]) -> list[float]:
        values = [0.0] * dimensions
        for index, value in enumerate(vector):
            values[index % dimensions] += float(value)
        return _normalize(values)

    task_vector, state_vector, memory_vector = map(
        fold, (task_vector, state_vector, memory_vector)
    )
    width = dimensions
    for index in range(width):
        features[f"memory_text::{index}"] = memory_vector[index]
        features[f"task_memory_product::{index}"] = float(
            task_vector[index] * memory_vector[index]
        )
        features[f"state_memory_product::{index}"] = float(
            state_vector[index] * memory_vector[index]
        )
    task_tokens = _tokens(query)
    state_tokens = _tokens(state_text)
    memory_tokens = _tokens(memory_text)
    features["lexical_task_memory_jaccard"] = _jaccard(task_tokens, memory_tokens)
    features["lexical_state_memory_jaccard"] = _jaccard(state_tokens, memory_tokens)
    features["lexical_task_memory_recall"] = (
        len(task_tokens & memory_tokens) / len(task_tokens) if task_tokens else 0.0
    )


def _add_insight_support_features(
    features: dict[str, float], metadata: Mapping[str, Any]
) -> None:
    positive = metadata.get("positive_task_count")
    negative = metadata.get("negative_task_count")
    for key in ("legacy_score", "positive_task_count", "negative_task_count"):
        _add_optional(features, f"insight_{key}", metadata.get(key))
    if not all(isinstance(value, (int, float)) and math.isfinite(value)
               for value in (positive, negative)):
        features["insight_maintenance_support_missing"] = 1.0
        return
    positive_value = max(float(positive), 0.0)
    negative_value = max(float(negative), 0.0)
    total = positive_value + negative_value
    # This is deliberately named a maintenance signal, not reliability: the
    # source lists are GMemory rule-maintenance associations, not Bernoulli
    # correctness observations.
    features["insight_maintenance_support_missing"] = 0.0
    features["insight_maintenance_support_total"] = total
    features["insight_maintenance_positive_fraction"] = (
        positive_value / total if total else 0.5
    )


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", str(text).casefold()))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


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
