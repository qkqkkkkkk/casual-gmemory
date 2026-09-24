from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from causal_diagnostic.fever_oracle_recipient.decision_evidence_analysis import (
    analyze,
)


RECIPIENTS = ("solver_0", "solver_2")


def _write_run(path: Path, sample_seed_base: int) -> None:
    path.mkdir(parents=True)
    manifest = {
        "design_hash": "same-design",
        "run_hash": f"run-{sample_seed_base}",
        "design": {"recipients": list(RECIPIENTS)},
        "run": {"sample_seed_base": sample_seed_base},
    }
    (path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    raw = {
        "use_all": "Evidence[Gold Page]\nFinish[SUPPORTS]",
        "global_drop": "Evidence[Wrong Page]\nFinish[SUPPORTS]",
        "only_solver_0": "Evidence[Gold Page]\nFinish[SUPPORTS]",
        "only_solver_2": "Evidence[Wrong Page]\nFinish[SUPPORTS]",
    }
    rows = []
    for repeat in range(4):
        for condition, output in raw.items():
            rows.append(
                {
                    "design_hash": "same-design",
                    "run_hash": f"run-{sample_seed_base}",
                    "event_id": "fever-1-candidate",
                    "task_id": 1,
                    "task": {"gold_evidence_page_sets": [["Gold_Page"]]},
                    "candidate": {"candidate_id": "candidate"},
                    "repeat_index": repeat,
                    "sample_seed": sample_seed_base + repeat,
                    "condition": condition,
                    "outcome": {
                        "team_score": 1.0,
                        "success": 1.0,
                        "reward": 1.0,
                        "steps": 1,
                        "local_metrics": {
                            "solver_0": {"f1": float(condition == "only_solver_0")},
                            "solver_2": {"f1": 0.0},
                        },
                        "trace": [{"decision": {"raw_output": output}}],
                    },
                }
            )
    (path / "branches.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class DecisionEvidenceAnalysisTests(unittest.TestCase):
    def test_scores_existing_decision_outputs_without_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, output = root / "seed0", root / "seed1000", root / "out"
            _write_run(first, 0)
            _write_run(second, 1000)
            payload = analyze(
                first,
                retest_results=second,
                output_dir=output,
                bootstrap_samples=200,
                seed=3,
            )

            self.assertEqual(payload["runs"][0]["event_recipient_pairs"], 2)
            self.assertEqual(payload["runs"][0]["positive_pairs"], 1)
            self.assertEqual(payload["runs"][0]["neutral_pairs"], 1)
            self.assertEqual(payload["runs"][0]["mean_utility"]["estimate"], 0.5)
            self.assertEqual(
                payload["independent_retest"]["strict_sign_consistency"], 1.0
            )
            self.assertTrue((output / "decision_evidence_matrix.csv").is_file())
            self.assertTrue((output / "decision_evidence_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
