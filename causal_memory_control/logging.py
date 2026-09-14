"""Append-only JSONL logging for events, audits, predictions, and decisions."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def write(self, event_type: str, payload: Any) -> None:
        if not event_type:
            raise ValueError("event_type must not be empty")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"event_type": event_type, "payload": _jsonable(payload)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)

