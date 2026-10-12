"""Read append-only experiment logs and recover an interrupted final write."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any


def read_jsonl(path: Path, *, repair_tail: bool = False) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    rows = []
    offset = 0
    for index, line in enumerate(lines):
        if not line.strip():
            offset += len(line)
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("log row must be an object")
        except (ValueError, UnicodeDecodeError) as exc:
            if not (repair_tail and index == len(lines) - 1 and not line.endswith(b"\n")):
                raise ValueError(f"invalid experiment log at {path}:{index + 1}: {exc}") from exc
            backup = path.with_name(f"{path.name}.partial-tail-{time.time_ns()}")
            backup.write_bytes(line)
            with path.open("r+b") as handle:
                handle.truncate(offset)
            print(f"Recovered incomplete final log row in {path}; saved fragment to {backup}")
            return rows
        rows.append(row)
        offset += len(line)
    # A valid last row without its newline also needs a delimiter before the
    # next append, otherwise two valid JSON objects would become one bad line.
    if repair_tail and data and not data.endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return rows
