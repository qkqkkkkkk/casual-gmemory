"""Adapters that read G-Memory objects without importing or modifying G-Memory."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping, Optional

from .types import CandidateRecord, RetrievalRecord


class GMemoryAdapter:
    """Duck-typed adapter for MASMessage, StateChain, and insight dictionaries."""

    def adapt_trajectory(
        self,
        message: Any,
        *,
        memory_id: Optional[str] = None,
        polarity: Optional[str] = None,
        source_metadata: Optional[Mapping[str, Any]] = None,
    ) -> CandidateRecord:
        task_main = str(getattr(message, "task_main", "") or "")
        task_description = str(getattr(message, "task_description", "") or "")
        trajectory = str(getattr(message, "task_trajectory", "") or "")
        source_label = getattr(message, "label", None)
        extra = self._extra_fields(message)
        chain = list(getattr(message, "chain_of_states", ()) or ())

        if polarity is None:
            polarity = "success" if source_label is True else "failure_warning"
        key_steps = extra.get("key_steps")
        content_parts = [part for part in (f"Source task: {task_description}" if task_description else "",) if part]
        if key_steps:
            content_parts.append(f"Key steps: {key_steps}")
        if trajectory:
            content_parts.append(f"Trajectory: {trajectory}")

        state_stats = self._state_stats(chain)
        metadata = {
            "source_task_main": task_main,
            "source_task_description": task_description,
            "source_label": source_label,
            "trajectory_text": trajectory,
            "clean_traj": extra.get("clean_traj"),
            "key_steps": key_steps,
            "fail_reason": extra.get("fail_reason"),
            "memory_schema_version": extra.get("memory_schema_version", "gmemory-v1"),
            **state_stats,
            **dict(source_metadata or {}),
        }
        memory_id = memory_id or extra.get("memory_id") or self._stable_id(
            "trajectory", task_main, task_description, trajectory, source_label
        )
        return CandidateRecord(
            memory_id=str(memory_id),
            memory_type="trajectory",
            polarity=polarity,
            content="\n".join(content_parts),
            structured_metadata=metadata,
        )

    def adapt_insight(
        self,
        insight: str | Mapping[str, Any],
        *,
        memory_id: Optional[str] = None,
        source_metadata: Optional[Mapping[str, Any]] = None,
    ) -> CandidateRecord:
        data: Mapping[str, Any] = {"rule": insight} if isinstance(insight, str) else insight
        rule = str(data.get("rule", "") or "")
        positive = tuple(data.get("positive_correlation_tasks", ()) or ())
        negative = tuple(data.get("negative_correlation_tasks", ()) or ())
        metadata = {
            "rule": rule,
            "legacy_score": data.get("score"),
            "positive_task_count": len(positive),
            "negative_task_count": len(negative),
            "positive_task_ids": positive,
            "negative_task_ids": negative,
            "memory_schema_version": data.get("memory_schema_version", "gmemory-v1"),
            **dict(source_metadata or {}),
        }
        memory_id = memory_id or data.get("memory_id") or self._stable_id(
            "insight", rule
        )
        return CandidateRecord(
            memory_id=str(memory_id),
            memory_type="insight",
            polarity="rule",
            content=f"Rule: {rule}",
            structured_metadata=metadata,
        )

    @staticmethod
    def retrieval_record(
        candidate_id: str,
        query_task: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> RetrievalRecord:
        metadata = metadata or {}
        return RetrievalRecord(
            candidate_id=candidate_id,
            query_task=query_task,
            retrieval_source=str(metadata.get("retrieval_source", "unknown")),
            rank=metadata.get("rank"),
            semantic_similarity=metadata.get("semantic_similarity"),
            distance=metadata.get("distance"),
            actual_hop=metadata.get("actual_hop"),
            path_weight=metadata.get("path_weight"),
            related_task_hit_count=metadata.get("related_task_hit_count"),
        )

    @staticmethod
    def _extra_fields(message: Any) -> Mapping[str, Any]:
        extra = getattr(message, "extra_fields", None)
        if isinstance(extra, Mapping):
            return extra
        getter = getattr(message, "get_extra_field", None)
        if getter is None:
            return {}
        return {
            key: getter(key)
            for key in ("clean_traj", "key_steps", "fail_reason", "memory_id")
            if getter(key) is not None
        }

    @staticmethod
    def _state_stats(chain: Iterable[Any]) -> dict[str, Any]:
        rewards: list[float] = []
        agent_names: set[str] = set()
        in_degrees: list[float] = []
        out_degrees: list[float] = []
        count = 0
        for state in chain:
            count += 1
            graph_data = getattr(state, "graph", {})
            reward = graph_data.get("reward") if isinstance(graph_data, Mapping) else None
            if isinstance(reward, (int, float)):
                rewards.append(float(reward))
            nodes = getattr(state, "nodes", None)
            if nodes is None:
                continue
            try:
                node_items = list(nodes(data=True))
            except TypeError:
                node_items = []
            for node_id, attrs in node_items:
                agent_name = attrs.get("agent_name") if isinstance(attrs, Mapping) else None
                if agent_name:
                    agent_names.add(str(agent_name))
                try:
                    in_degrees.append(float(state.in_degree(node_id)))
                    out_degrees.append(float(state.out_degree(node_id)))
                except (AttributeError, TypeError):
                    continue
        return {
            "state_count": count,
            "source_reward_sum": sum(rewards) if rewards else None,
            "source_reward_last": rewards[-1] if rewards else None,
            "has_source_reward": bool(rewards),
            "source_agent_names": tuple(sorted(agent_names)),
            "source_agent_count": len(agent_names),
            "source_graph_in_degree_mean": _mean_or_zero(in_degrees),
            "source_graph_out_degree_mean": _mean_or_zero(out_degrees),
            "has_raw_replay_artifact": False,
        }

    @staticmethod
    def _stable_id(*parts: Any) -> str:
        digest = hashlib.sha256(
            "\x1f".join(str(part) for part in parts).encode("utf-8")
        ).hexdigest()[:24]
        return f"pm-{digest}"


def _mean_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
