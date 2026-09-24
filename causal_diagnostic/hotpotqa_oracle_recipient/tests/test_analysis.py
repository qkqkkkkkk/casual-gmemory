from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from causal_diagnostic.hotpotqa_oracle_recipient.analysis import write_analysis


RECIPIENTS = ("solver_0", "solver_2")


def _rows() -> list[dict]:
    result = []
    scores = {
        "use_all": 0.8,
        "global_drop": 0.4,
        "only_solver_0": 0.7,
        "only_solver_2": 0.2,
    }
    evidence = {
        "use_all": {"solver_0": 0.8, "solver_2": 0.5},
        "global_drop": {"solver_0": 0.2, "solver_2": 0.2},
        "only_solver_0": {"solver_0": 0.9, "solver_2": 0.2},
        "only_solver_2": {"solver_0": 0.2, "solver_2": 0.8},
    }
    for repeat in range(4):
        for condition, score in scores.items():
            result.append(
                {
                    "design_hash": "design",
                    "run_hash": "run",
                    "event_id": "hotpotqa-1-candidate",
                    "task_id": 1,
                    "candidate": {"candidate_id": "candidate"},
                    "repeat_index": repeat,
                    "sample_seed": repeat,
                    "condition": condition,
                    "outcome": {
                        "team_score": score,
                        "success": float(score == 1.0),
                        "reward": score,
                        "steps": 1,
                        "score_definition": (
                            "official_hotpotqa_normalized_answer_token_f1"
                        ),
                        "decision_evidence_f1": 0.5,
                        "local_metrics": {
                            recipient: {
                                "metric": "hotpotqa_supporting_fact_f1",
                                "f1": value,
                            }
                            for recipient, value in evidence[condition].items()
                        },
                    },
                }
            )
    return result


class HotpotAnalysisTests(unittest.TestCase):
    def test_hotpot_report_and_tables_are_task_specific(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "design_hash": "design",
                "run_hash": "run",
                "design": {
                    "benchmark": "HotpotQA_distractor_offline",
                    "recipients": list(RECIPIENTS),
                },
                "run": {"sample_seed_base": 0},
            }
            (root / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (root / "branches.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in _rows()),
                encoding="utf-8",
            )
            output = write_analysis(root, bootstrap_samples=200)
            payload = json.loads(output.read_text(encoding="utf-8"))
            report = (root / "run_report.md").read_text(encoding="utf-8")

        self.assertEqual(payload["benchmark"], "HotpotQA_distractor_offline")
        self.assertEqual(
            payload["current"]["rq4"]["local_metric"],
            "HotpotQA supporting-fact F1",
        )
        self.assertIn("Answer F1", report)
        self.assertIn("supporting-fact", report)
        self.assertNotIn("FEVER", report)
        self.assertNotIn("PDDL", report)


if __name__ == "__main__":
    unittest.main()
