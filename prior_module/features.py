"""Feature construction for prior heads using text, vectors, and metadata."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
import re
from typing import Any, Mapping, Optional, Protocol, Sequence

from .types import CandidateRecord, DecisionContext, RetrievalRecord


class TextEmbedder(Protocol):
    def embed(self, text: str) -> Sequence[float]:
        """Return one stable vector for text."""


class HashEmbedder:
    """A deterministic dependency-free embedder for cold start and tests.

    Production should pass an adapter around the same frozen embedding model
    used by G-Memory retrieval. This fallback intentionally optimizes neither
    recall nor semantic quality; it only keeps the module runnable in isolation.
    """

    def __init__(self, dimensions: int = 256):
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8.")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[\w]+", (text or "").lower())
        for token, count in Counter(tokens).items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            sign = 1.0 if (value >> 8) & 1 else -1.0
            vector[index] += sign * count
        return _normalize(vector)


class GMemoryEmbeddingAdapter:
    """Adapt an existing G-Memory embedding function without changing it."""

    def __init__(self, embedding_func: Any):
        self.embedding_func = embedding_func

    def embed(self, text: str) -> Sequence[float]:
        if hasattr(self.embedding_func, "embed_query"):
            return self.embedding_func.embed_query(text)
        if callable(self.embedding_func):
            return self.embedding_func(text)
        raise TypeError("embedding_func must be callable or expose embed_query().")


class FeatureBuilder:
    """Build numeric features and explicit missingness indicators."""

    def __init__(self, embedder: Optional[TextEmbedder] = None):
        self.embedder = embedder or HashEmbedder()

    def build(
        self,
        context: DecisionContext,
        candidate: CandidateRecord,
        retrieval: Optional[RetrievalRecord],
        candidate_set: Sequence[CandidateRecord],
    ) -> dict[str, float]:
        task_vector = self.embedder.embed(context.task_text)
        state_vector = self.embedder.embed(context.state_text)
        role_vector = self.embedder.embed(context.agent_text)
        memory_vector = self.embedder.embed(candidate.content)
        metadata = candidate.structured_metadata

        features: dict[str, float] = {
            "sim_task_memory": _cosine(task_vector, memory_vector),
            "sim_state_memory": _cosine(state_vector, memory_vector),
            "sim_role_memory": _cosine(role_vector, memory_vector),
            "step_index": float(context.step_index),
            "candidate_text_length": float(len(candidate.content)),
            "candidate_token_count": float(len(re.findall(r"\w+", candidate.content))),
            "candidate_redundancy_max": self._redundancy(
                candidate, candidate_set, memory_vector
            ),
            "is_trajectory": float(candidate.memory_type == "trajectory"),
            "is_insight": float(candidate.memory_type == "insight"),
            "is_success_memory": float(candidate.polarity == "success"),
            "is_failure_warning": float(candidate.polarity == "failure_warning"),
            "is_rule_memory": float(candidate.polarity == "rule"),
        }
        self._add_retrieval(features, retrieval)
        self._add_metadata(features, metadata)
        self._add_context(features, context)
        return features

    def _add_retrieval(
        self, features: dict[str, float], retrieval: Optional[RetrievalRecord]
    ) -> None:
        if retrieval is None:
            for name in (
                "rank",
                "semantic_similarity",
                "distance",
                "actual_hop",
                "path_weight",
                "related_task_hit_count",
            ):
                self._add_optional(features, f"retrieval_{name}", None)
            features["retrieval_source::missing"] = 1.0
            return

        self._add_optional(features, "retrieval_rank", retrieval.rank)
        self._add_optional(
            features, "retrieval_semantic_similarity", retrieval.semantic_similarity
        )
        self._add_optional(features, "retrieval_distance", retrieval.distance)
        self._add_optional(features, "retrieval_actual_hop", retrieval.actual_hop)
        self._add_optional(features, "retrieval_path_weight", retrieval.path_weight)
        self._add_optional(
            features,
            "retrieval_related_task_hit_count",
            retrieval.related_task_hit_count,
        )
        features[
            "retrieval_source::" + _safe_category(retrieval.retrieval_source)
        ] = 1.0

    def _add_metadata(
        self, features: dict[str, float], metadata: Mapping[str, Any]
    ) -> None:
        source_label = metadata.get("source_label")
        self._add_optional(
            features,
            "source_label",
            1.0 if source_label is True else 0.0 if source_label is False else None,
        )
        for name in (
            "state_count",
            "source_reward_sum",
            "source_reward_last",
            "source_agent_count",
            "source_graph_in_degree_mean",
            "source_graph_out_degree_mean",
            "legacy_score",
            "positive_task_count",
            "negative_task_count",
        ):
            self._add_optional(features, name, metadata.get(name))

        features["has_key_steps"] = float(bool(metadata.get("key_steps")))
        features["key_steps_length"] = float(len(str(metadata.get("key_steps") or "")))
        features["has_fail_reason"] = float(bool(metadata.get("fail_reason")))
        features["has_raw_replay_artifact"] = float(
            bool(metadata.get("has_raw_replay_artifact"))
        )
        positive = _as_float(metadata.get("positive_task_count"))
        negative = _as_float(metadata.get("negative_task_count"))
        if positive is None or negative is None:
            features["insight_support_mean"] = 0.5
            features["insight_support_missing"] = 1.0
        else:
            features["insight_support_mean"] = (positive + 1.0) / (
                positive + negative + 2.0
            )
            features["insight_support_missing"] = 0.0

        for name in (
            "source_environment_family",
            "source_task_type",
            "source_model_id",
            "source_prompt_version",
            "memory_schema_version",
        ):
            value = metadata.get(name)
            if value is not None:
                features[f"{name}::{_safe_category(value)}"] = 1.0

    def _add_context(
        self, features: dict[str, float], context: DecisionContext
    ) -> None:
        features[f"agent_profile::{_safe_category(context.agent_profile)}"] = 1.0
        for name in ("task_type", "game_name", "difficulty"):
            value = context.task_config.get(name)
            if value is None:
                features[f"context_{name}_missing"] = 1.0
            else:
                features[f"context_{name}::{_safe_category(value)}"] = 1.0
        for name, value in context.graph_features.items():
            self._add_optional(features, f"current_graph_{name}", value)

    @staticmethod
    def _add_optional(
        features: dict[str, float], name: str, value: Any
    ) -> None:
        numeric = _as_float(value)
        if numeric is None:
            features[name] = 0.0
            features[f"{name}_missing"] = 1.0
        else:
            features[name] = numeric
            features[f"{name}_missing"] = 0.0

    def _redundancy(
        self,
        candidate: CandidateRecord,
        candidate_set: Sequence[CandidateRecord],
        candidate_vector: Sequence[float],
    ) -> float:
        similarities = [
            _cosine(candidate_vector, self.embedder.embed(other.content))
            for other in candidate_set
            if other.memory_id != candidate.memory_id
        ]
        return max(similarities, default=0.0)


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return list(vector)
    return [value / norm for value in vector]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same dimension.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _safe_category(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value).strip().lower()) or "unknown"
