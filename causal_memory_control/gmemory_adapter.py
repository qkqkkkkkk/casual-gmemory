"""Duck-typed bridge from current G-Memory retrievals to use events."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Callable, Mapping, Optional, Sequence

from .types import (
    BranchRequest,
    MemoryCandidate,
    MemoryUseEvent,
    RecipientContext,
    RetrievalMetadata,
)


@dataclass(frozen=True)
class AdaptedGMemoryRetrieval:
    candidates: tuple[MemoryCandidate, ...]
    raw_by_id: Mapping[str, Any] = field(repr=False)
    successful_ids: tuple[str, ...] = ()
    insight_ids: tuple[str, ...] = ()
    failed_count: int = 0

    def candidate(self, memory_id: str) -> MemoryCandidate:
        for candidate in self.candidates:
            if candidate.memory_id == memory_id:
                return candidate
        raise KeyError(memory_id)


@dataclass(frozen=True)
class GMemoryPromptInputs:
    memory_few_shots: tuple[str, ...]
    insights: tuple[str, ...]


class GMemoryRetrievalAdapter:
    """Normalize ``GMemory.retrieve_memory()`` without importing G-Memory."""

    def adapt(self, result: tuple[Sequence[Any], Sequence[Any], Sequence[Any]]) -> AdaptedGMemoryRetrieval:
        successful, failed, insights = result
        candidates = []
        raw_by_id: dict[str, Any] = {}
        successful_ids = []
        insight_ids = []
        used_ids: set[str] = set()

        for index, message in enumerate(successful):
            candidate = self._trajectory(message, index)
            candidate = self._unique(candidate, used_ids)
            candidates.append(candidate)
            successful_ids.append(candidate.memory_id)
            raw_by_id[candidate.memory_id] = message
        for index, insight in enumerate(insights):
            candidate = self._insight(insight, index)
            candidate = self._unique(candidate, used_ids)
            candidates.append(candidate)
            insight_ids.append(candidate.memory_id)
            raw_by_id[candidate.memory_id] = insight
        return AdaptedGMemoryRetrieval(
            candidates=tuple(candidates),
            raw_by_id=raw_by_id,
            successful_ids=tuple(successful_ids),
            insight_ids=tuple(insight_ids),
            failed_count=len(failed),
        )

    def build_event(
        self,
        retrieval: AdaptedGMemoryRetrieval,
        *,
        target_id: str,
        query: str,
        task_state: str,
        receiver_agent_id: str,
        receiver_role: str,
        recipient_agent_ids: Sequence[str] = (),
        candidate_ids_by_recipient: Optional[Mapping[str, Sequence[str]]] = None,
        retrieval_metadata: Optional[RetrievalMetadata] = None,
        task_metadata: Optional[Mapping[str, Any]] = None,
        reliability_prior: Optional[float] = None,
        event_id: Optional[str] = None,
    ) -> MemoryUseEvent:
        target = retrieval.candidate(target_id)
        all_ids = tuple(candidate.memory_id for candidate in retrieval.candidates)
        if candidate_ids_by_recipient is None:
            recipients = tuple(recipient_agent_ids) or (receiver_agent_id,)
            if receiver_agent_id not in recipients:
                recipients = recipients + (receiver_agent_id,)
            contexts = tuple(RecipientContext(agent_id, all_ids) for agent_id in recipients)
        else:
            context_map = {
                agent_id: tuple(candidate_ids)
                for agent_id, candidate_ids in candidate_ids_by_recipient.items()
            }
            if receiver_agent_id not in context_map:
                raise ValueError("candidate_ids_by_recipient must include the receiver")
            contexts = tuple(
                RecipientContext(agent_id, candidate_ids)
                for agent_id, candidate_ids in context_map.items()
            )
        if event_id is None:
            event_id = "cmc-" + _digest(
                query, task_state, receiver_agent_id, target_id
            )[:24]
        return MemoryUseEvent(
            event_id=event_id,
            query=query,
            task_state=task_state,
            receiver_agent_id=receiver_agent_id,
            receiver_role=receiver_role,
            memory=target,
            candidate_set=retrieval.candidates,
            recipient_contexts=contexts,
            retrieval=retrieval_metadata,
            task_metadata=dict(task_metadata or {}),
            reliability_prior=reliability_prior,
        )

    def render_prompt_inputs(
        self,
        request: BranchRequest,
        retrieval: AdaptedGMemoryRetrieval,
        *,
        agent_id: str,
        trajectory_formatter: Optional[Callable[[Any], str]] = None,
    ) -> GMemoryPromptInputs:
        """Render exactly the candidates visible to one branch recipient."""

        context = request.context_for(agent_id)
        candidates = {candidate.memory_id: candidate for candidate in request.event.candidate_set}
        if request.placebo_candidate is not None:
            candidates[request.placebo_candidate.memory_id] = request.placebo_candidate
        shots = []
        insights = []
        for candidate_id in context.candidate_ids:
            candidate = candidates.get(candidate_id)
            if candidate is None:
                raise KeyError(f"candidate {candidate_id!r} is unavailable")
            raw = retrieval.raw_by_id.get(candidate_id)
            if candidate.memory_type == "trajectory":
                shots.append(
                    trajectory_formatter(raw)
                    if trajectory_formatter is not None and raw is not None
                    else candidate.content
                )
            elif candidate.memory_type == "insight":
                insights.append(_insight_text(raw) if raw is not None else candidate.content)
        return GMemoryPromptInputs(tuple(shots), tuple(insights))

    @staticmethod
    def _trajectory(message: Any, index: int) -> MemoryCandidate:
        task_main = str(getattr(message, "task_main", "") or "")
        task_description = str(getattr(message, "task_description", "") or "")
        trajectory = str(getattr(message, "task_trajectory", "") or "")
        label = getattr(message, "label", None)
        extra = getattr(message, "extra_fields", None)
        extra = extra if isinstance(extra, Mapping) else {}
        key_steps = extra.get("key_steps")
        content = "\n".join(
            part
            for part in (
                f"Source task: {task_description}" if task_description else "",
                f"Key steps: {key_steps}" if key_steps else "",
                f"Trajectory: {trajectory}" if trajectory else "",
            )
            if part
        )
        raw_id = extra.get("memory_id")
        memory_id = str(raw_id) if raw_id else "trajectory-" + _digest(
            task_main, task_description, trajectory, label
        )[:24]
        return MemoryCandidate(
            memory_id=memory_id,
            memory_type="trajectory",
            content=content,
            metadata={
                "retrieval_index": index,
                "source_label": label,
                "key_steps": key_steps,
                "fail_reason": extra.get("fail_reason"),
                "memory_schema_version": extra.get("memory_schema_version", "gmemory-v1"),
            },
        )

    @staticmethod
    def _insight(insight: Any, index: int) -> MemoryCandidate:
        text = _insight_text(insight)
        data = insight if isinstance(insight, Mapping) else {}
        raw_id = data.get("memory_id")
        memory_id = str(raw_id) if raw_id else "insight-" + _digest(text)[:24]
        return MemoryCandidate(
            memory_id=memory_id,
            memory_type="insight",
            content=text,
            metadata={
                "retrieval_index": index,
                "legacy_score": data.get("score"),
                "positive_task_count": len(data.get("positive_correlation_tasks", ()) or ()),
                "negative_task_count": len(data.get("negative_correlation_tasks", ()) or ()),
                "memory_schema_version": data.get("memory_schema_version", "gmemory-v1"),
            },
        )

    @staticmethod
    def _unique(candidate: MemoryCandidate, used_ids: set[str]) -> MemoryCandidate:
        memory_id = candidate.memory_id
        suffix = 1
        while memory_id in used_ids:
            suffix += 1
            memory_id = f"{candidate.memory_id}#{suffix}"
        used_ids.add(memory_id)
        if memory_id == candidate.memory_id:
            return candidate
        return MemoryCandidate(
            memory_id=memory_id,
            memory_type=candidate.memory_type,
            content=candidate.content,
            metadata=candidate.metadata,
        )


def _insight_text(insight: Any) -> str:
    if isinstance(insight, Mapping):
        return str(insight.get("rule", "") or "")
    return str(insight or "")


def _digest(*parts: Any) -> str:
    return hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()

