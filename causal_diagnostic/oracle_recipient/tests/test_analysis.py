from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from causal_diagnostic.oracle_recipient.analysis import (
    AnalysisError,
    analyze_run,
    compare_runs,
    write_analysis,
)


RECIPIENTS = ("solver_0", "solver_1", "solver_2")


def rows() -> list[dict]:
    result = []
    event_values = {
        10: {
            "use_all": 0.5,
            "global_drop": 0.75,
            "drop_solver_0": 0.0,
            "drop_solver_1": 1.0,
            "drop_solver_2": 0.5,
        },
        11: {
            "use_all": 1.0,
            "global_drop": 0.5,
            "drop_solver_0": 0.5,
            "drop_solver_1": 0.5,
            "drop_solver_2": 0.5,
        },
    }
    for task_id, conditions in event_values.items():
        for repeat_index in range(2):
            for condition, value in conditions.items():
                result.append(
                    {
                        "design_hash": "design",
                        "run_hash": "run",
                        "task_id": task_id,
                        "candidate": {"candidate_id": f"candidate-{task_id}"},
                        "repeat_index": repeat_index,
                        "sample_seed": repeat_index,
                        "condition": condition,
                        "outcome": {
                            "team_score": value,
                            "success": value,
                            "reward": value,
                            "steps": 2.0,
                        },
                    }
                )
    return result


class AnalysisTests(unittest.TestCase):
    def test_joint_oracle_and_recipient_metrics(self):
        analysis = analyze_run(
            rows(), RECIPIENTS, bootstrap_samples=200, seed=7
        )
        self.assertEqual(analysis["event_count"], 2)
        oracle = analysis["rq2"]["primary_team_score"]
        self.assertEqual(oracle["always_use_mean"], 0.75)
        self.assertEqual(oracle["oracle_selective_mean"], 0.875)
        self.assertEqual(
            oracle["oracle_gain_vs_always_use"]["estimate"], 0.125
        )
        recipient = analysis["rq3"]["primary_team_score"]
        self.assertEqual(recipient["direct_sign_flip_rate"]["estimate"], 0.5)
        self.assertEqual(recipient["mean_utility_range"]["estimate"], 0.5)

    def test_independent_retest_reproduces_oracle_and_flip(self):
        previous = analyze_run(rows(), RECIPIENTS, bootstrap_samples=200, seed=7)
        current = analyze_run(rows(), RECIPIENTS, bootstrap_samples=200, seed=8)
        retest = compare_runs(
            current,
            previous,
            bootstrap_samples=200,
            confidence_level=0.95,
            seed=9,
        )
        self.assertEqual(retest["shared_events"], 2)
        self.assertEqual(
            retest["rq2"]["global_utility_sign_consistency"]["estimate"],
            1.0,
        )
        self.assertEqual(
            retest["rq3"]["reproducible_directional_flip_rate"]["estimate"],
            0.5,
        )

    def test_missing_recipient_condition_is_rejected(self):
        incomplete = [
            row for row in rows() if row["condition"] != "drop_solver_2"
        ]
        with self.assertRaisesRegex(AnalysisError, "missing"):
            analyze_run(incomplete, RECIPIENTS, bootstrap_samples=200)

    def test_write_analysis_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "design_hash": "design",
                "run_hash": "run",
                "design": {"recipients": list(RECIPIENTS)},
            }
            (root / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (root / "branches.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows()) + "\n",
                encoding="utf-8",
            )
            output = write_analysis(root, bootstrap_samples=200)
            self.assertTrue(output.is_file())
            self.assertTrue((root / "recipient_matrix.csv").is_file())
            self.assertTrue((root / "rq2_team_score_policy_table.csv").is_file())
            self.assertTrue((root / "rq2_success_policy_table.csv").is_file())
            report = (root / "run_report.md").read_text(encoding="utf-8")
            self.assertIn("RQ2: Global selective-memory value", report)
            self.assertIn("Random (budget matched)", report)
            self.assertIn("Delta vs Always Use", report.replace("Δ", "Delta"))
            self.assertIn("RQ3: Recipient heterogeneity", report)

    def test_retest_writes_combined_cross_run_policy_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous_root = root / "seed0"
            current_root = root / "seed1000"
            previous_root.mkdir()
            current_root.mkdir()
            for results_root, sample_seed_base in (
                (previous_root, 0),
                (current_root, 1000),
            ):
                manifest = {
                    "design_hash": "design",
                    "run_hash": "run",
                    "design": {"recipients": list(RECIPIENTS)},
                    "run": {"sample_seed_base": sample_seed_base},
                }
                (results_root / "run_manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                (results_root / "branches.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows()) + "\n",
                    encoding="utf-8",
                )

            output = write_analysis(
                current_root,
                retest_results=previous_root,
                bootstrap_samples=200,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            table = payload["rq2_policy_tables"]["team_score"]
            self.assertEqual(table["run_labels"], ["Seed 0", "Seed 1000"])
            by_policy = {row["policy"]: row for row in table["rows"]}
            self.assertIn("cross_run_selective_drop", by_policy)
            self.assertEqual(
                by_policy["cross_run_selective_drop"]["mean_score"],
                by_policy["oracle_selective_drop"]["mean_score"],
            )
            report = (current_root / "run_report.md").read_text(encoding="utf-8")
            self.assertIn("Seed 0 Team Score", report)
            self.assertIn("Seed 1000 Team Score", report)
            self.assertIn("Cross-run Selective Drop", report)


if __name__ == "__main__":
    unittest.main()
