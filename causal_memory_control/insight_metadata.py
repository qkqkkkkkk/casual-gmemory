"""Recover GMemory insight provenance that is lost by ``retrieve_memory``.

GMemory persists insight dictionaries but currently returns only their rule
strings.  The gate uses this small catalog adapter to restore the persisted
maintenance signals without changing retrieval order or prompt contents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


INSIGHT_METADATA_FIELDS = (
    "legacy_score",
    "positive_task_count",
    "negative_task_count",
)


def load_insight_catalog(memory_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Return metadata keyed by the exact insight rule text.

    Missing catalogs are valid for snapshots with no insights.  Malformed
    catalogs fail closed because silently dropping provenance would recreate
    the train/runtime mismatch this adapter is intended to prevent.
    """

    path = Path(memory_dir) / "insights.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"GMemory insight catalog must be a list: {path}")
    catalog: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise ValueError(f"invalid insight catalog row {index}: {path}")
        rule = str(raw.get("rule", "") or "")
        if not rule:
            continue
        positive = tuple(raw.get("positive_correlation_tasks", ()) or ())
        negative = tuple(raw.get("negative_correlation_tasks", ()) or ())
        catalog[rule] = {
            "legacy_score": raw.get("score"),
            "positive_task_count": len(positive),
            "negative_task_count": len(negative),
            "positive_task_ids": positive,
            "negative_task_ids": negative,
        }
    return catalog


def enrich_insight_payload(
    payload: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Add catalog fields to one persisted diagnostic candidate payload."""

    result = dict(payload)
    text = str(result.get("text", result.get("rule", "")) or "")
    metadata = catalog.get(text)
    if metadata:
        result.update(metadata)
    return result
