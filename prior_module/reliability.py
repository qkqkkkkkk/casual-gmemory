"""Replay-backed empirical-Bayes reliability prior."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .types import CandidateRecord, ReliabilityScore


@dataclass
class _Evidence:
    passed: int = 0
    failed: int = 0

    def update(self, success: bool) -> None:
        if success:
            self.passed += 1
        else:
            self.failed += 1


class ReliabilityPrior:
    """Estimate trajectory reproducibility from replay evidence.

    The model is intentionally an empirical-Bayes baseline. It uses a weak
    source-label prior until replay labels arrive, then shrinks a memory's
    evidence toward its source group. The output is intrinsic record
    reliability, not transferability to the current task.
    """

    def __init__(
        self,
        *,
        success_alpha: float = 2.0,
        success_beta: float = 1.0,
        unknown_alpha: float = 1.0,
        unknown_beta: float = 1.0,
        global_strength: float = 3.0,
        group_weight: float = 0.25,
    ):
        self.success_alpha = success_alpha
        self.success_beta = success_beta
        self.unknown_alpha = unknown_alpha
        self.unknown_beta = unknown_beta
        self.global_strength = global_strength
        self.group_weight = group_weight
        self._global = _Evidence()
        self._by_group: dict[str, _Evidence] = defaultdict(_Evidence)
        self._by_memory: dict[str, _Evidence] = defaultdict(_Evidence)
        self._memory_groups: dict[str, str] = {}

    def register_candidate(self, candidate: CandidateRecord) -> str:
        group_key = self.group_key(candidate)
        previous = self._memory_groups.get(candidate.memory_id)
        if previous is not None and previous != group_key:
            raise ValueError(
                f"memory_id {candidate.memory_id!r} was registered under a different group."
            )
        self._memory_groups[candidate.memory_id] = group_key
        return group_key

    def group_key(self, candidate: CandidateRecord) -> str:
        metadata = candidate.structured_metadata
        values = (
            candidate.memory_type,
            metadata.get("source_environment_family")
            or metadata.get("environment_family")
            or "unknown_environment",
            metadata.get("source_task_type")
            or metadata.get("task_type")
            or "unknown_task_type",
            metadata.get("source_mas_version")
            or metadata.get("mas_version")
            or "unknown_mas",
            metadata.get("source_model_id") or metadata.get("model_id") or "unknown_model",
            metadata.get("source_prompt_version")
            or metadata.get("prompt_version")
            or "unknown_prompt",
            metadata.get("memory_schema_version", "unknown_schema"),
        )
        return "|".join(str(value) for value in values)

    def update_replay(
        self,
        memory_id: str,
        passed: bool,
        *,
        group_key: Optional[str] = None,
    ) -> None:
        resolved_group = group_key or self._memory_groups.get(memory_id)
        if resolved_group is None:
            raise KeyError(
                "Unknown memory_id. Register the candidate or provide group_key first."
            )
        self._memory_groups[memory_id] = resolved_group
        self._global.update(passed)
        self._by_group[resolved_group].update(passed)
        self._by_memory[memory_id].update(passed)

    def score(self, candidate: CandidateRecord) -> ReliabilityScore:
        group_key = self.register_candidate(candidate)
        memory = self._by_memory[candidate.memory_id]
        group = self._by_group[group_key]

        base_alpha, base_beta = self._base_prior(candidate)
        if self._global.passed + self._global.failed:
            global_mean = (base_alpha + self._global.passed) / (
                base_alpha + base_beta + self._global.passed + self._global.failed
            )
            base_alpha = self.global_strength * global_mean
            base_beta = self.global_strength * (1.0 - global_mean)

        # Remove this memory's own observations before adding group evidence.
        # Otherwise its replay result would be counted twice.
        group_passed = max(0, group.passed - memory.passed)
        group_failed = max(0, group.failed - memory.failed)
        alpha = base_alpha + self.group_weight * group_passed + memory.passed
        beta = base_beta + self.group_weight * group_failed + memory.failed
        total = alpha + beta
        mean = alpha / total
        variance = alpha * beta / (total * total * (total + 1.0))
        source = "group_beta" if (group.passed + group.failed) else "weak_beta"
        return ReliabilityScore(
            mean=mean,
            variance=variance,
            evidence_count=total,
            source=source,
            calibrated=False,
            group_key=group_key,
        )

    def export_evidence(self) -> dict[str, Any]:
        return {
            "global": vars(self._global),
            "groups": {key: vars(value) for key, value in self._by_group.items()},
            "memories": {key: vars(value) for key, value in self._by_memory.items()},
            "memory_groups": dict(self._memory_groups),
        }

    def _base_prior(self, candidate: CandidateRecord) -> tuple[float, float]:
        source_label = candidate.structured_metadata.get("source_label")
        if candidate.polarity == "success" or source_label is True:
            return self.success_alpha, self.success_beta
        return self.unknown_alpha, self.unknown_beta
