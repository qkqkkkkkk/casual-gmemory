from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from causal_diagnostic.fever_oracle_recipient.run_experiment import (
    _candidate_design,
    _candidate_specs,
    _prepare_output,
)


class RunExperimentCandidateScopeTests(unittest.TestCase):
    def test_all_candidate_specs_cover_every_configured_rank(self) -> None:
        args = SimpleNamespace(
            all_candidates=True,
            candidate_kind="trajectory",
            candidate_index=0,
            successful_topk=3,
            insights_topk=3,
        )

        self.assertEqual(
            _candidate_specs(args),
            (
                ("trajectory", 0),
                ("trajectory", 1),
                ("trajectory", 2),
                ("insight", 0),
                ("insight", 1),
                ("insight", 2),
            ),
        )
        self.assertEqual(
            _candidate_design(args)["candidate_scope"], "all_retrieved"
        )

    def test_resume_loads_multiple_candidates_for_the_same_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            args = SimpleNamespace(output_dir=output, resume=False)
            rows_path, existing = _prepare_output(
                args,
                {"design": "all-candidates"},
                {"run": "seed-0"},
            )
            self.assertFalse(existing)
            rows = [
                {
                    "task_id": 7,
                    "candidate": {
                        "candidate_id": candidate_id,
                        "kind": kind,
                        "index": index,
                    },
                    "repeat_index": 0,
                    "condition": "use_all",
                }
                for candidate_id, kind, index in (
                    ("trajectory-a-rank1", "trajectory", 0),
                    ("insight-b-rank1", "insight", 0),
                )
            ]
            rows_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            args.resume = True
            _, resumed = _prepare_output(
                args,
                {"design": "all-candidates"},
                {"run": "seed-0"},
            )
            self.assertEqual(len(resumed), 2)


if __name__ == "__main__":
    unittest.main()
