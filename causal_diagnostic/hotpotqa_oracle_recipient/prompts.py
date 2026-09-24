"""Context-supplied HotpotQA prompts for deterministic offline evaluation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


HOTPOTQA_SYSTEM_PROMPT = """You are solving an offline multi-hop HotpotQA question.
Use only the supplied context and memory examples; do not browse or request web search.
Identify the minimal supporting facts, then give a short final answer.
Return exactly two lines:
Evidence[Wikipedia title#sentence_index | Wikipedia title#sentence_index]
Finish[short answer]
Sentence indices are the zero-based indices printed in the supplied context.
Do not include explanations outside these two lines."""


HOTPOTQA_FEW_SHOTS = (
    "Question: What color is the daytime sky on a clear day?\n"
    "Evidence[Sky#0]\nFinish[blue]",
)


def format_context(context: Sequence[Sequence[Any]]) -> str:
    blocks = []
    for title, sentences in context:
        rendered = "\n".join(
            f"[{index}] {str(sentence).strip()}"
            for index, sentence in enumerate(sentences)
        )
        blocks.append(f"Title: {str(title).strip()}\n{rendered}")
    return "\n\n".join(blocks)


def prepare_example(example: Mapping[str, Any]) -> dict[str, Any]:
    question = str(example["question"]).strip()
    answer = str(example["answer"]).strip()
    context = [[str(title), list(sentences)] for title, sentences in example["context"]]
    supporting = [
        [str(title), int(sentence_index)]
        for title, sentence_index in example["supporting_facts"]
    ]
    task_main = f"HotpotQA question: {question}"
    task_description = (
        "Offline context-supplied multi-hop question answering.\n"
        f"Question: {question}\n\nContext:\n{format_context(context)}\n\n"
        "Return Evidence[Title#sentence_index | ...] followed by Finish[short answer]."
    )
    return {
        "id": int(example["id"]),
        "hotpot_id": str(example["hotpot_id"]),
        "question": question,
        "answer": answer,
        # The shared causal runner uses `label` as the generic hidden target.
        "label": answer,
        "claim": question,
        # The shared runner passes this field to the task-specific local scorer.
        "gold_evidence_page_sets": supporting,
        "supporting_facts": supporting,
        "context": context,
        "task_main": task_main,
        "task_description": task_description,
        "few_shots": list(HOTPOTQA_FEW_SHOTS),
    }
