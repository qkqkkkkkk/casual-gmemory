from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import warnings

from causal_diagnostic.hotpotqa_oracle_recipient.data import (
    deterministic_split,
    load_hotpotqa,
)
from causal_diagnostic.hotpotqa_oracle_recipient.metrics import (
    score_answer,
    score_supporting_facts,
)
from causal_diagnostic.hotpotqa_oracle_recipient.offline_env import (
    OfflineHotpotQAEnv,
)
from causal_diagnostic.hotpotqa_oracle_recipient.prompts import prepare_example


def _example(index: int) -> dict:
    return {
        "_id": f"hp-{index}",
        "question": f"Question {index}?",
        "answer": "Eiffel Tower",
        "type": "bridge",
        "level": "easy",
        "supporting_facts": [["Alpha", 0], ["Beta", 1]],
        "context": [
            ["Alpha", ["First supporting sentence."]],
            ["Beta", ["Distractor.", "Second supporting sentence."]],
        ],
    }


class HotpotDataMetricEnvTests(unittest.TestCase):
    def test_standard_json_load_split_and_prompt_do_not_leak_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hotpot.json"
            path.write_text(
                json.dumps([_example(index) for index in range(8)]),
                encoding="utf-8",
            )
            rows = load_hotpotqa(path)
        support, evaluation = deterministic_split(
            rows, support_count=3, evaluation_count=4, seed=42
        )
        self.assertFalse(
            {row["hotpot_id"] for row in support}
            & {row["hotpot_id"] for row in evaluation}
        )
        task = prepare_example(evaluation[0])
        self.assertIn("Title: Alpha", task["task_description"])
        self.assertNotIn("Eiffel Tower", task["task_description"])

    def test_duplicate_and_equivalent_titles_use_union_of_sentence_indices(self) -> None:
        raw = _example(1)
        raw["context"] = [
            ["New_York", ["Sentence zero.", "Sentence one."]],
            ["new york", ["A duplicate-title paragraph."]],
        ]
        raw["supporting_facts"] = [[" NEW   YORK ", 1]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hotpot.json"
            path.write_text(json.dumps([raw]), encoding="utf-8")
            rows = load_hotpotqa(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["supporting_facts"], [["NEW   YORK", 1]])

    def test_invalid_support_is_warned_and_excluded_deterministically(self) -> None:
        invalid = _example(0)
        invalid["supporting_facts"] = [["Alpha", 99]]
        values = [invalid] + [_example(index) for index in range(1, 6)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hotpot.json"
            path.write_text(json.dumps(values), encoding="utf-8")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                rows = load_hotpotqa(path)
        self.assertEqual([row["id"] for row in rows], [1, 2, 3, 4, 5])
        self.assertNotIn("hp-0", {row["hotpot_id"] for row in rows})
        self.assertTrue(any("excluded 1 HotpotQA examples" in str(w.message) for w in caught))
        first = deterministic_split(rows, support_count=2, evaluation_count=2, seed=42)
        second = deterministic_split(rows, support_count=2, evaluation_count=2, seed=42)
        self.assertEqual(
            [[row["hotpot_id"] for row in split] for split in first],
            [[row["hotpot_id"] for row in split] for split in second],
        )

    def test_answer_and_supporting_fact_metrics(self) -> None:
        answer = score_answer("The Eiffel Tower", "Eiffel Tower")
        self.assertEqual(answer["em"], 1.0)
        evidence = score_supporting_facts(
            "Evidence[Alpha#0 | Beta#0]\nFinish[Eiffel Tower]",
            [["Alpha", 0], ["Beta", 1]],
        )
        self.assertEqual(evidence["precision"], 0.5)
        self.assertEqual(evidence["recall"], 0.5)
        self.assertEqual(evidence["f1"], 0.5)

    def test_environment_returns_f1_reward_and_em_success(self) -> None:
        raw = _example(1)
        raw.update(id=1, hotpot_id="hp-1")
        task = prepare_example(raw)
        env = OfflineHotpotQAEnv()
        env.set_env(task)
        _observation, reward, terminal = env.step(
            "Evidence[Alpha#0 | Beta#1]\nFinish[Tower]"
        )
        final_reward, success, _feedback = env.feedback()
        self.assertTrue(terminal)
        self.assertEqual(reward, final_reward)
        self.assertGreater(reward, 0.0)
        self.assertFalse(success)


if __name__ == "__main__":
    unittest.main()
