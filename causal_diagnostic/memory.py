"""Memory helpers that keep both causal branches on one frozen snapshot."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

from mas.memory.common import MASMessage
from mas.memory.mas_memory.GMemory import GMemory


@dataclass(frozen=True)
class FrozenRetrieval:
    """The exact retrieval result shared by the with/no-memory branches."""

    successful: tuple[MASMessage, ...]
    failed: tuple[MASMessage, ...]
    insights: tuple[str, ...]

    @classmethod
    def from_result(cls, result: tuple[list, list, list]) -> "FrozenRetrieval":
        successful, failed, insights = result
        return cls(
            successful=tuple(copy.deepcopy(successful)),
            failed=tuple(copy.deepcopy(failed)),
            insights=tuple(copy.deepcopy(insights)),
        )

    def without(self, candidate_kind: str, candidate_index: int) -> "FrozenRetrieval":
        if candidate_kind == "trajectory":
            if not 0 <= candidate_index < len(self.successful):
                raise IndexError(
                    f"trajectory candidate {candidate_index} is unavailable; "
                    f"retrieved {len(self.successful)} successful trajectories"
                )
            return FrozenRetrieval(
                successful=tuple(
                    value for index, value in enumerate(self.successful)
                    if index != candidate_index
                ),
                failed=self.failed,
                insights=self.insights,
            )
        if candidate_kind == "insight":
            if not 0 <= candidate_index < len(self.insights):
                raise IndexError(
                    f"insight candidate {candidate_index} is unavailable; "
                    f"retrieved {len(self.insights)} insights"
                )
            return FrozenRetrieval(
                successful=self.successful,
                failed=self.failed,
                insights=tuple(
                    value for index, value in enumerate(self.insights)
                    if index != candidate_index
                ),
            )
        raise ValueError(f"Unsupported candidate kind: {candidate_kind}")


class FrozenReadOnlyGMemory(GMemory):
    """G-Memory whose retrieval is fixed and whose writes are disabled."""

    def __init__(self, *, frozen_retrieval: FrozenRetrieval, **kwargs):
        self._frozen_retrieval = frozen_retrieval
        super().__init__(**kwargs)

    def retrieve_memory(self, **_kwargs) -> tuple[list, list, list]:
        return (
            copy.deepcopy(list(self._frozen_retrieval.successful)),
            copy.deepcopy(list(self._frozen_retrieval.failed)),
            copy.deepcopy(list(self._frozen_retrieval.insights)),
        )
