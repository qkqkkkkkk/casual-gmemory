"""Convenience checkpoint adapter for hosts that can snapshot in memory."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Callable, Mapping, Optional

from .types import AuditCheckpoint, MemoryUseEvent


class SnapshotCheckpointBackend:
    """Capture opaque host state; the branch runner remains responsible for restore."""

    def __init__(
        self,
        snapshot: Callable[[MemoryUseEvent], Any],
        *,
        sampling_config: Optional[Mapping[str, Any]] = None,
    ):
        if not callable(snapshot):
            raise TypeError("snapshot must be callable")
        self.snapshot = snapshot
        self.sampling_config = dict(sampling_config or {})
        self._counter = 0

    def capture(self, event: MemoryUseEvent) -> AuditCheckpoint:
        self._counter += 1
        state = copy.deepcopy(self.snapshot(event))
        digest = hashlib.sha256(repr(state).encode("utf-8")).hexdigest()
        return AuditCheckpoint(
            checkpoint_id=f"{event.event_id}-checkpoint-{self._counter}",
            event_id=event.event_id,
            upstream_state=state,
            sampling_config=self.sampling_config,
            upstream_digest=digest,
        )

