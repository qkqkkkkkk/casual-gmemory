from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from causal_diagnostic.oracle_recipient.analysis import (
    analyze_run,
    build_events,
    compare_runs,
    write_analysis,
)


RECIPIENTS = ("solver_0", "solver_1", "solver_2")


def _rows() -> list[dict]:
    values = {
        "use_all": 1.0,
        "global_drop": 0.0,
        "drop_solver_0": 0.0,
        "drop_solver_1": 1.0,
        "drop_solver_2": 1.0,
    }
    return [
        {
            "design_hash": "design",
            "run_hash": "run",
            "event_id": "fever-123-trajectory-abc",
            "task_id": 123,
            "candidate": {"candidate_id": "trajectory-abc"},
            "repeat_index": 0,
            "sample_seed": 0,
            "condition": condition,
            "outcome": {
                "team_score": value,
                "success": value,
                "reward": value,
                "steps": 1,
            },
        }
        for condition, value in values.items()
    ]


def _isolated_rows() -> list[dict]:
    result = []
    values = {
        "use_all": 1.0,
        "global_drop": 1.0,
        "only_solver_0": 0.0,
        "only_solver_2": 1.0,
    }
    local = {
        "use_all": {"solver_0": 1.0, "solver_2": 1.0},
        "global_drop": {"solver_0": 0.0, "solver_2": 0.0},
        "only_solver_0": {"solver_0": 1.0, "solver_2": 0.0},
        "only_solver_2": {"solver_0": 0.0, "solver_2": 0.0},
    }
    for repeat in range(6):
        for condition, value in values.items():
            result.append(
                {
                    "design_hash": "design",
                    "run_hash": "run",
                    "event_id": "fever-321-trajectory-isolated",
                    "task_id": 321,
                    "candidate": {"candidate_id": "trajectory-isolated"},
                    "repeat_index": repeat,
                    "sample_seed": repeat,
                    "condition": condition,
                    "outcome": {
                        "team_score": value,
                        "success": value,
                        "reward": value,
                        "steps": 1,
                        "local_metrics": {
                            recipient: {"f1": score}
                            for recipient, score in local[condition].items()
                        },
                    },
                }
            )
    return result


class FeverAnalysisTests(unittest.TestCase):
    def test_registered_fever_event_id_is_preserved(self):
        events = build_events(_rows(), RECIPIENTS)
        self.assertEqual(events[0]["event_id"], "fever-123-trajectory-abc")

    def test_report_uses_fever_accuracy_not_pddl_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "design_hash": "design",
                "run_hash": "run",
                "design": {
                    "benchmark": "FEVER_binary_offline",
                    "recipients": list(RECIPIENTS),
                },
                "run": {"sample_seed_base": 0},
            }
            (root / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (root / "branches.jsonl").write_text(
                "\n".join(json.dumps(row) for row in _rows()) + "\n",
                encoding="utf-8",
            )
            write_analysis(root, bootstrap_samples=200)
            report = (root / "run_report.md").read_text(encoding="utf-8")
            self.assertIn("exact FEVER binary-label accuracy", report)
            self.assertIn("offline closed-book", report)
            self.assertIn("Team Accuracy", report)
            self.assertNotIn("raw PDDL success", report)
            self.assertNotIn("Success-rate sensitivity", report)

    def test_isolated_rq3_and_objective_rq4_share_rollouts(self):
        analysis = analyze_run(
            _isolated_rows(),
            ("solver_0", "solver_2"),
            bootstrap_samples=200,
            seed=7,
        )
        rq3 = analysis["rq3"]["primary_team_score"]
        self.assertEqual(rq3["recipient_intervention"], "isolated_exposure")
        self.assertEqual(
            rq3["per_recipient"]["solver_0"]["mean_utility"]["estimate"],
            -1.0,
        )
        self.assertEqual(rq3["split_half_status"], "ok")
        rq4 = analysis["rq4"]
        self.assertEqual(rq4["status"], "ok")
        self.assertEqual(rq4["local_team_mismatch_rate"]["estimate"], 0.5)
        self.assertEqual(
            rq4["reproducible_local_team_mismatch_rate"]["estimate"], 0.5
        )
        comparison = compare_runs(
            analysis,
            analysis,
            bootstrap_samples=200,
            confidence_level=0.95,
            seed=11,
        )
        self.assertEqual(comparison["rq4"]["status"], "ok")
        self.assertEqual(
            comparison["rq4"][
                "cross_run_split_half_reproducible_mismatch_rate"
            ]["estimate"],
            0.5,
        )

    def test_rq4_audit_matrix_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "design_hash": "design",
                "run_hash": "run",
                "design": {
                    "benchmark": "FEVER_binary_offline",
                    "recipients": ["solver_0", "solver_2"],
                },
                "run": {"sample_seed_base": 0},
            }
            (root / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (root / "branches.jsonl").write_text(
                "\n".join(json.dumps(row) for row in _isolated_rows()) + "\n",
                encoding="utf-8",
            )
            write_analysis(root, bootstrap_samples=200)
            lines = (root / "rq4_local_team_matrix.csv").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(lines), 3)
            self.assertIn("reproducible_mismatch", lines[0])

    def test_optional_label_probability_and_margin_utilities(self):
        rows = _isolated_rows()
        probabilities = {
            "use_all": 0.6,
            "global_drop": 0.4,
            "only_solver_0": 0.7,
            "only_solver_2": 0.3,
        }
        for row in rows:
            probability = probabilities[row["condition"]]
            row["outcome"]["team_probability_score"] = probability
            row["outcome"]["team_margin_score"] = probability - 0.5
        analysis = analyze_run(
            rows,
            ("solver_0", "solver_2"),
            bootstrap_samples=200,
            seed=13,
        )
        probability = analysis["rq3"]["label_probability_sensitivity"]
        self.assertAlmostEqual(
            probability["per_recipient"]["solver_0"]["mean_utility"]["estimate"],
            0.3,
        )
        self.assertAlmostEqual(
            probability["per_recipient"]["solver_2"]["mean_utility"]["estimate"],
            -0.1,
        )
        self.assertIn("label_margin_sensitivity", analysis["rq2"])


if __name__ == "__main__":
    unittest.main()
