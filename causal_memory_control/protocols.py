"""Host protocols kept deliberately independent of the G-Memory runtime."""

from __future__ import annotations

from typing import Any, Protocol

from .types import AuditCheckpoint, BranchOutcome, BranchRequest, MemoryUseEvent


class CheckpointBackend(Protocol):
    def capture(self, event: MemoryUseEvent) -> AuditCheckpoint:
        """Freeze state after retrieval and before receiver prompt injection."""


class BranchRunner(Protocol):
    def run(self, request: BranchRequest) -> BranchOutcome:
        """Restore the checkpoint, inject the requested arm, and replay suffix."""


class BehaviorDistance(Protocol):
    def __call__(self, left: Any, right: Any) -> float:
        """Return a non-negative receiver behavior distance."""


class CallableBranchRunner:
    """Adapt a plain callable to :class:`BranchRunner`."""

    def __init__(self, function: Any):
        if not callable(function):
            raise TypeError("function must be callable")
        self.function = function

    def run(self, request: BranchRequest) -> BranchOutcome:
        outcome = self.function(request)
        if not isinstance(outcome, BranchOutcome):
            raise TypeError("branch callable must return BranchOutcome")
        return outcome

