"""Append-only JSONL logging for reproducible prior decisions."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any, Mapping


class DecisionLogger:
    """A thread-safe append-only log owned by the prior module."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, event_type: str, payload: Mapping[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **to_jsonable(dict(payload)),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
