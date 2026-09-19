"""Deterministic, leakage-safe data handling for binary FEVER."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence


BINARY_LABELS = ("SUPPORTS", "REFUTES")


class FeverDataError(ValueError):
    """Raised when a FEVER source or registered split is invalid."""


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_binary_fever(path: Path) -> list[dict[str, Any]]:
    """Load unique SUPPORTS/REFUTES examples, preserving source fields."""
    if not path.is_file():
        raise FeverDataError(f"FEVER source does not exist: {path}")
    examples: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FeverDataError(f"invalid JSON at line {line_number}") from exc
        label = str(row.get("label", "")).upper()
        if label not in BINARY_LABELS:
            continue
        try:
            claim_id = int(row["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FeverDataError(
                f"binary example at line {line_number} has no integer id"
            ) from exc
        claim = str(row.get("claim", "")).strip()
        if not claim:
            raise FeverDataError(f"binary example {claim_id} has an empty claim")
        if claim_id in seen:
            raise FeverDataError(f"duplicate binary FEVER id: {claim_id}")
        seen.add(claim_id)
        normalized = dict(row)
        normalized.update(id=claim_id, claim=claim, label=label)
        examples.append(normalized)
    if not examples:
        raise FeverDataError(f"no binary FEVER examples found in {path}")
    return examples


def _interleave(by_label: Mapping[str, Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    width = max((len(by_label[label]) for label in BINARY_LABELS), default=0)
    for index in range(width):
        for label in BINARY_LABELS:
            values = by_label[label]
            if index < len(values):
                result.append(dict(values[index]))
    return result


def stratified_split(
    examples: Sequence[Mapping[str, Any]],
    *,
    support_per_label: int,
    evaluation_per_label: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select deterministic, balanced, disjoint support/evaluation sets."""
    if support_per_label < 1:
        raise FeverDataError("support_per_label must be at least 1")
    if evaluation_per_label < 1:
        raise FeverDataError("evaluation_per_label must be at least 1")
    groups: dict[str, list[dict[str, Any]]] = {
        label: [] for label in BINARY_LABELS
    }
    for raw in examples:
        label = str(raw.get("label", "")).upper()
        if label in groups:
            groups[label].append(dict(raw))

    support: dict[str, list[dict[str, Any]]] = {}
    evaluation: dict[str, list[dict[str, Any]]] = {}
    required = support_per_label + evaluation_per_label
    for label_index, label in enumerate(BINARY_LABELS):
        values = sorted(groups[label], key=lambda row: int(row["id"]))
        if len(values) < required:
            raise FeverDataError(
                f"{label} has {len(values)} examples; {required} are required"
            )
        # Label-specific streams keep the split stable if label iteration changes.
        random.Random(int(seed) + 1_000_003 * label_index).shuffle(values)
        support[label] = values[:support_per_label]
        evaluation[label] = values[
            support_per_label : support_per_label + evaluation_per_label
        ]

    support_rows = _interleave(support)
    evaluation_rows = _interleave(evaluation)
    support_ids = {int(row["id"]) for row in support_rows}
    evaluation_ids = {int(row["id"]) for row in evaluation_rows}
    if support_ids & evaluation_ids:
        raise FeverDataError("support/evaluation split leakage")
    return support_rows, evaluation_rows


def examples_by_id(
    examples: Iterable[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for raw in examples:
        claim_id = int(raw["id"])
        if claim_id in result:
            raise FeverDataError(f"duplicate FEVER id: {claim_id}")
        result[claim_id] = dict(raw)
    return result


def resolve_registered_examples(
    examples: Sequence[Mapping[str, Any]], ids: Sequence[int]
) -> list[dict[str, Any]]:
    index = examples_by_id(examples)
    missing = [int(value) for value in ids if int(value) not in index]
    if missing:
        raise FeverDataError(f"registered FEVER ids are missing: {missing[:10]}")
    if len(set(int(value) for value in ids)) != len(ids):
        raise FeverDataError("registered FEVER ids contain duplicates")
    return [dict(index[int(value)]) for value in ids]

