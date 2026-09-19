"""Objective FEVER evidence-page metric for the RQ4 local outcome."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping, Sequence


def gold_evidence_page_sets(example: Mapping[str, Any]) -> list[list[str]]:
    """Return alternative gold page sets from the original FEVER annotation."""
    result: list[list[str]] = []
    for raw_set in example.get("evidence", []) or []:
        pages: list[str] = []
        for item in raw_set or []:
            if isinstance(item, (list, tuple)) and len(item) >= 3 and item[2]:
                page = str(item[2])
                if page not in pages:
                    pages.append(page)
        if pages and pages not in result:
            result.append(pages)
    return result


def _normalize_page(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("-LRB-", "(").replace("-RRB-", ")")
    text = text.replace("_", " ").casefold().strip()
    return " ".join(re.findall(r"[\w]+", text, flags=re.UNICODE))


def parse_evidence_pages(output: str) -> list[str]:
    matches = re.findall(r"Evidence\s*\[([^\]]*)\]", str(output), re.IGNORECASE)
    if not matches:
        return []
    pages = []
    for value in matches[-1].split("|"):
        page = value.strip()
        if page and page.upper() not in {"NONE", "N/A", "UNKNOWN"}:
            pages.append(page)
    return list(dict.fromkeys(pages))


def score_evidence_pages(
    output: str, gold_sets: Sequence[Sequence[str]]
) -> dict[str, Any]:
    predicted_raw = parse_evidence_pages(output)
    predicted = {_normalize_page(value) for value in predicted_raw if _normalize_page(value)}
    normalized_gold_sets = [
        {_normalize_page(value) for value in values if _normalize_page(value)}
        for values in gold_sets
    ]
    normalized_gold_sets = [values for values in normalized_gold_sets if values]
    best = (0.0, 0.0, 0.0)
    for gold in normalized_gold_sets:
        overlap = len(predicted & gold)
        precision = overlap / len(predicted) if predicted else 0.0
        recall = overlap / len(gold)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        if (f1, recall, precision) > (best[2], best[1], best[0]):
            best = (precision, recall, f1)
    return {
        "metric": "fever_gold_evidence_page_f1",
        "precision": best[0],
        "recall": best[1],
        "f1": best[2],
        "predicted_pages": predicted_raw,
        "gold_page_sets": [list(values) for values in gold_sets],
    }

