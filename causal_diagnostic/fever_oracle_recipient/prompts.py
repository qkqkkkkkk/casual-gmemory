"""Closed-book binary FEVER prompts used by every experimental branch."""

from __future__ import annotations

from typing import Any, Mapping

from .local_metric import gold_evidence_page_sets


FEVER_SYSTEM_PROMPT = """You are solving an offline closed-book FEVER classification task.
Classify each claim using only your internal knowledge and the context supplied in the prompt.
There are exactly two labels: SUPPORTS and REFUTES. Do not use or request web search.
First identify the Wikipedia page titles that would contain the decisive evidence.
Return exactly two lines in this format:
Evidence[Page title 1 | Page title 2]
Finish[SUPPORTS]
or:
Evidence[Page title 1 | Page title 2]
Finish[REFUTES]
Use the canonical page title when possible. Do not output both labels."""


FEVER_FEW_SHOTS = (
    "Claim: Every square has four sides.\nEvidence[Square]\nFinish[SUPPORTS]",
    "Claim: A standard week contains nine days.\nEvidence[Week]\nFinish[REFUTES]",
)


def prepare_example(example: Mapping[str, Any]) -> dict[str, Any]:
    """Create the task fields expected by native MacNet without leaking gold."""
    claim = str(example["claim"]).strip()
    task_main = f"FEVER claim: {claim}"
    task_description = (
        "Offline binary FEVER classification.\n"
        f"Claim: {claim}\n"
        "Choose exactly one label and finish with Finish[SUPPORTS] or "
        "Finish[REFUTES]. On the preceding line, provide the decisive page "
        "titles as Evidence[Page title 1 | Page title 2]."
    )
    return {
        "id": int(example["id"]),
        "claim": claim,
        "label": str(example["label"]).upper(),
        "gold_evidence_page_sets": gold_evidence_page_sets(example),
        "task_main": task_main,
        "task_description": task_description,
        "few_shots": list(FEVER_FEW_SHOTS),
    }
