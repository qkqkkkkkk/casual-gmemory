"""Deterministic, leakage-safe loading for standard HotpotQA JSON/JSONL."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence


class HotpotDataError(ValueError):
    pass


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise HotpotDataError(f"HotpotQA source does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HotpotDataError(
                    f"invalid HotpotQA JSONL at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise HotpotDataError(f"line {line_number} is not an object")
            rows.append(row)
        return rows
    if isinstance(value, dict) and isinstance(value.get("data"), list):
        value = value["data"]
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise HotpotDataError("HotpotQA source must be a JSON list or JSONL objects")
    return list(value)


def load_hotpotqa(path: Path) -> list[dict[str, Any]]:
    """Load context-supplied HotpotQA examples with stable integer IDs."""
    examples = []
    seen: set[str] = set()
    for source_index, raw in enumerate(_raw_rows(path)):
        hotpot_id = str(raw.get("_id", raw.get("id", ""))).strip()
        question = str(raw.get("question", "")).strip()
        answer = str(raw.get("answer", "")).strip()
        context = raw.get("context")
        supporting = raw.get("supporting_facts")
        if not hotpot_id or not question or not answer:
            raise HotpotDataError(
                f"HotpotQA row {source_index} lacks _id/question/answer"
            )
        if hotpot_id in seen:
            raise HotpotDataError(f"duplicate HotpotQA id: {hotpot_id}")
        if not isinstance(context, list) or not context:
            raise HotpotDataError(f"HotpotQA {hotpot_id} has no supplied context")
        normalized_context: list[list[Any]] = []
        context_sizes: dict[str, int] = {}
        for paragraph in context:
            if not isinstance(paragraph, (list, tuple)) or len(paragraph) != 2:
                raise HotpotDataError(f"HotpotQA {hotpot_id} has invalid context")
            title = str(paragraph[0]).strip()
            sentences = [str(value).strip() for value in paragraph[1]]
            if not title or not sentences:
                raise HotpotDataError(f"HotpotQA {hotpot_id} has empty context")
            normalized_context.append([title, sentences])
            context_sizes[title] = len(sentences)
        if not isinstance(supporting, list) or not supporting:
            raise HotpotDataError(
                f"HotpotQA {hotpot_id} has no supporting_facts annotations"
            )
        normalized_supporting = []
        for fact in supporting:
            if not isinstance(fact, (list, tuple)) or len(fact) != 2:
                raise HotpotDataError(
                    f"HotpotQA {hotpot_id} has invalid supporting fact"
                )
            title, sentence_index = str(fact[0]).strip(), int(fact[1])
            if title not in context_sizes or not 0 <= sentence_index < context_sizes[title]:
                raise HotpotDataError(
                    f"HotpotQA {hotpot_id} supporting fact is outside its context"
                )
            normalized_supporting.append([title, sentence_index])
        seen.add(hotpot_id)
        row = dict(raw)
        row.update(
            id=source_index,
            hotpot_id=hotpot_id,
            question=question,
            answer=answer,
            context=normalized_context,
            supporting_facts=normalized_supporting,
            type=str(raw.get("type", "unknown")),
            level=str(raw.get("level", "unknown")),
        )
        examples.append(row)
    if not examples:
        raise HotpotDataError(f"no HotpotQA examples found in {path}")
    return examples


def deterministic_split(
    examples: Sequence[Mapping[str, Any]],
    *,
    support_count: int,
    evaluation_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if support_count < 1 or evaluation_count < 1:
        raise HotpotDataError("support/evaluation counts must be at least 1")
    if support_count + evaluation_count > len(examples):
        raise HotpotDataError(
            f"requested {support_count + evaluation_count} examples but source has "
            f"only {len(examples)}"
        )
    values = sorted((dict(row) for row in examples), key=lambda row: row["hotpot_id"])
    random.Random(int(seed)).shuffle(values)
    support = values[:support_count]
    evaluation = values[support_count : support_count + evaluation_count]
    if {row["hotpot_id"] for row in support} & {
        row["hotpot_id"] for row in evaluation
    }:
        raise HotpotDataError("support/evaluation split leakage")
    return support, evaluation


def examples_by_id(
    examples: Iterable[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    result = {}
    for raw in examples:
        task_id = int(raw["id"])
        if task_id in result:
            raise HotpotDataError(f"duplicate normalized HotpotQA id: {task_id}")
        result[task_id] = dict(raw)
    return result


def resolve_registered_examples(
    examples: Sequence[Mapping[str, Any]], ids: Sequence[int]
) -> list[dict[str, Any]]:
    index = examples_by_id(examples)
    normalized = [int(value) for value in ids]
    if len(set(normalized)) != len(normalized):
        raise HotpotDataError("registered HotpotQA ids contain duplicates")
    missing = [value for value in normalized if value not in index]
    if missing:
        raise HotpotDataError(f"registered HotpotQA ids are missing: {missing[:10]}")
    return [dict(index[value]) for value in normalized]
