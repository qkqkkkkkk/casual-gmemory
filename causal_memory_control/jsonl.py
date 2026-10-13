"""Read append-only experiment logs and recover an interrupted final write."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
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


def retry_jev_error_tasks(path: Path) -> None:
    """Keep completed successful tasks and back up rows needing Jev retry."""
    rows = read_jsonl(path)
    retained = [row for row in rows if not row.get("outcome", {}).get("jev_errors")]
    if len(retained) == len(rows):
        return
    backup = path.with_name(f"{path.name}.before-jev-retry-{time.time_ns()}")
    shutil.copy2(path, backup)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retained),
                         encoding="utf-8")
    temporary.replace(path)
    print(f"Retrying {len(rows) - len(retained)} Jev-error tasks; previous rows saved in {backup}")
