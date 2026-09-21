from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from causal_diagnostic.fever_oracle_recipient.run_all import (
    StageFailure,
    _base_runner_command,
    _candidate_design,
    _existing_cache_seed,
    _needs_run_resume,
    _needs_snapshot_resume,
    _run_command,
    _run_complete,
    _snapshot_complete,
    parse_args,
)


class RunAllTests(unittest.TestCase):
    def test_all_candidate_mode_is_forwarded_with_complete_design(self):
        args = parse_args(
            (
                "--all-candidates",
                "--gate-training-only",
                "--successful-topk",
                "2",
                "--insights-topk",
                "3",
            )
        )
        design = _candidate_design(args)
        command = _base_runner_command(args, 1, Path("output"))

        self.assertEqual(design["candidate_scope"], "all_retrieved")
        self.assertEqual(
            design["candidate_specs"],
            [
                {"kind": "trajectory", "index": 0},
                {"kind": "trajectory", "index": 1},
                {"kind": "insight", "index": 0},
                {"kind": "insight", "index": 1},
                {"kind": "insight", "index": 2},
            ],
        )
        self.assertIn("--all-candidates", command)
        self.assertIn("--gate-training-only", command)

    def test_completion_and_resume_are_derived_from_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory = root / "memory"
            output = root / "output"
            memory.mkdir()
            output.mkdir()
            (memory / "support_runs.jsonl").touch()
            (memory / "causal_snapshot_progress.json").write_text(
                json.dumps({"status": "running"}), encoding="utf-8"
            )
            (output / "run_manifest.json").write_text(
                json.dumps({"run": {"cache_seed_from": "/tmp/cache.sqlite"}}),
                encoding="utf-8",
            )
            (output / "branches.jsonl").touch()
            self.assertTrue(_needs_snapshot_resume(memory))
            self.assertFalse(_snapshot_complete(memory))
            self.assertTrue(_needs_run_resume(output))
            self.assertFalse(_run_complete(output))
            self.assertEqual(_existing_cache_seed(output), "/tmp/cache.sqlite")

            (memory / "causal_snapshot_manifest.json").touch()
            (memory / "causal_snapshot_progress.json").write_text(
                json.dumps({"status": "completed"}), encoding="utf-8"
            )
            (output / "oracle_recipient_analysis.json").touch()
            (output / "collection_progress.json").write_text(
                json.dumps({"status": "completed"}), encoding="utf-8"
            )
            self.assertTrue(_snapshot_complete(memory))
            self.assertTrue(_run_complete(output))

    def test_subprocess_output_is_logged_and_failure_is_detailed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            success_log = root / "success.log"
            _run_command(
                "success",
                (sys.executable, "-c", "print('stage-ok')"),
                success_log,
            )
            self.assertIn("stage-ok", success_log.read_text(encoding="utf-8"))

            failure_log = root / "failure.log"
            with self.assertRaises(StageFailure) as context:
                _run_command(
                    "failure",
                    (
                        sys.executable,
                        "-c",
                        "import sys; print('specific-error'); sys.exit(7)",
                    ),
                    failure_log,
                )
            self.assertEqual(context.exception.returncode, 7)
            self.assertIn("specific-error", context.exception.tail)
            self.assertEqual(context.exception.log_path, failure_log)


if __name__ == "__main__":
    unittest.main()
