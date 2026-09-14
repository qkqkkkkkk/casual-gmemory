"""Pure receiver-level interventions over a frozen retrieval snapshot."""

from __future__ import annotations

from typing import Callable, Iterable, Optional

from .types import (
    AuditCheckpoint,
    BranchRequest,
    InterventionArm,
    MemoryCandidate,
    MemoryUseEvent,
    RecipientContext,
)


class NoPlaceboAvailable(ValueError):
    pass


def choose_length_matched_placebo(
    target: MemoryCandidate,
    pool: Iterable[MemoryCandidate],
    *,
    excluded_ids: Iterable[str] = (),
    is_irrelevant: Optional[Callable[[MemoryCandidate], bool]] = None,
) -> MemoryCandidate:
    """Choose a deterministic same-type, nearest-length irrelevant candidate.

    Semantic irrelevance is experiment-specific.  Callers should normally pass
    ``is_irrelevant`` (or pre-filter the pool); metadata ``irrelevant=True`` is
    honored when no predicate is supplied.
    """

    excluded = set(excluded_ids)
    candidates = []
    for candidate in pool:
        if candidate.memory_id == target.memory_id or candidate.memory_id in excluded:
            continue
        if candidate.memory_type != target.memory_type:
            continue
        irrelevant = (
            is_irrelevant(candidate)
            if is_irrelevant is not None
            else bool(candidate.metadata.get("irrelevant", False))
        )
        if irrelevant:
            candidates.append(candidate)
    if not candidates:
        raise NoPlaceboAvailable(
            f"no irrelevant {target.memory_type!r} placebo is available"
        )
    return min(
        candidates,
        key=lambda candidate: (
            abs(len(candidate.content) - len(target.content)),
            candidate.memory_id,
        ),
    )


class InterventionBuilder:
    """Create USE/DROP/PLACEBO/GLOBAL_DROP requests without mutating memory."""

    def __init__(
        self,
        placebo_pool: Iterable[MemoryCandidate] = (),
        *,
        is_irrelevant: Optional[Callable[[MemoryCandidate], bool]] = None,
    ):
        self.placebo_pool = tuple(placebo_pool)
        self.is_irrelevant = is_irrelevant

    def build(
        self,
        checkpoint: AuditCheckpoint,
        event: MemoryUseEvent,
        arm: InterventionArm,
        *,
        seed: int,
        repeat_index: int,
    ) -> BranchRequest:
        target_id = event.memory.memory_id
        contexts = event.recipient_contexts
        placebo = None

        if arm == InterventionArm.DROP:
            contexts = _remove_from(contexts, target_id, event.receiver_agent_id)
        elif arm == InterventionArm.GLOBAL_DROP:
            contexts = _remove_from(contexts, target_id, agent_id=None)
        elif arm == InterventionArm.PLACEBO:
            placebo = choose_length_matched_placebo(
                event.memory,
                self.placebo_pool,
                excluded_ids=(candidate.memory_id for candidate in event.candidate_set),
                is_irrelevant=self.is_irrelevant,
            )
            contexts = _replace_for_receiver(
                contexts,
                target_id,
                placebo.memory_id,
                event.receiver_agent_id,
            )
        elif arm != InterventionArm.USE:
            raise ValueError(f"unsupported intervention arm: {arm}")

        return BranchRequest(
            checkpoint=checkpoint,
            event=event,
            arm=arm,
            seed=seed,
            repeat_index=repeat_index,
            recipient_contexts=contexts,
            placebo_candidate=placebo,
        )


def _remove_from(
    contexts: tuple[RecipientContext, ...],
    target_id: str,
    agent_id: Optional[str],
) -> tuple[RecipientContext, ...]:
    return tuple(
        RecipientContext(
            context.agent_id,
            tuple(
                memory_id
                for memory_id in context.candidate_ids
                if memory_id != target_id
                or (agent_id is not None and context.agent_id != agent_id)
            ),
        )
        for context in contexts
    )


def _replace_for_receiver(
    contexts: tuple[RecipientContext, ...],
    target_id: str,
    replacement_id: str,
    receiver_id: str,
) -> tuple[RecipientContext, ...]:
    replaced = False
    result = []
    for context in contexts:
        ids = context.candidate_ids
        if context.agent_id == receiver_id:
            if target_id not in ids:
                raise ValueError("target is absent from the receiver context")
            ids = tuple(
                replacement_id if memory_id == target_id else memory_id
                for memory_id in ids
            )
            replaced = True
        result.append(RecipientContext(context.agent_id, ids))
    if not replaced:
        raise ValueError("receiver is absent from recipient_contexts")
    return tuple(result)

