"""Official-style HotpotQA answer metrics and supporting-fact F1."""

from __future__ import annotations

from collections import Counter
import re
import string
from typing import Any, Sequence


def normalize_answer(value: str) -> str:
    text = str(value).lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def score_answer(prediction: str, gold: str) -> dict[str, Any]:
    predicted = normalize_answer(prediction)
    expected = normalize_answer(gold)
    exact_match = float(predicted == expected)
    if predicted in {"yes", "no", "noanswer"} or expected in {
        "yes",
        "no",
        "noanswer",
    }:
        f1 = exact_match
        precision = exact_match
        recall = exact_match
    else:
        predicted_tokens = predicted.split()
        expected_tokens = expected.split()
        overlap = sum((Counter(predicted_tokens) & Counter(expected_tokens)).values())
        precision = overlap / len(predicted_tokens) if predicted_tokens else 0.0
        recall = overlap / len(expected_tokens) if expected_tokens else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return {
        "metric": "hotpotqa_answer",
        "em": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "prediction": str(prediction),
        "gold": str(gold),
    }


def _normalize_title(value: str) -> str:
    return " ".join(str(value).replace("_", " ").casefold().split())


def parse_supporting_facts(output: str) -> list[tuple[str, int]]:
    matches = re.findall(r"Evidence\s*\[([^\]]*)\]", str(output), re.IGNORECASE)
    if not matches:
        return []
    result = []
    for raw in matches[-1].split("|"):
        value = raw.strip()
        match = re.match(r"^(.*?)\s*(?:#|::)\s*(\d+)\s*$", value)
        if not match:
            continue
        fact = (match.group(1).strip(), int(match.group(2)))
        if fact not in result:
            result.append(fact)
    return result


def score_supporting_facts(
    output: str, gold_facts: Sequence[Sequence[Any]]
) -> dict[str, Any]:
    predicted_raw = parse_supporting_facts(output)
    predicted = {(_normalize_title(title), int(index)) for title, index in predicted_raw}
    gold = {(_normalize_title(str(title)), int(index)) for title, index in gold_facts}
    overlap = len(predicted & gold)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(gold) if gold else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "metric": "hotpotqa_supporting_fact_f1",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_facts": [[title, index] for title, index in predicted_raw],
        "gold_facts": [[str(title), int(index)] for title, index in gold_facts],
    }
