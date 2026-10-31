"""The explicit v10 upgrade keeps measurements and configuration identities."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from causal_memory_control import continue_interactive_empty_memory as upgrade
from causal_memory_control import interactive_empty_memory as empty
from causal_memory_control import interactive_worker as worker
from causal_memory_control import run_interactive_gate_benchmark as cli
from causal_memory_control.interactive_data import digest, files_identity, save, sha


REPO = Path(__file__).resolve().parents[2]


def fixture(directory):
    root = Path(directory).resolve() / "v10"
    config = cli.configuration(cli.parse_args(["--datasets", "pddl", "--pddl-folds", "5",
        "--include-local-global-team-u", "--exclude-ce", "--output-dir", str(root), "--plan-only"]))
    config["experiments"]["alfworld"] = {"tasks": [], "split": {}, "protocol": "test-no-run"}
    config["code_sha256"].pop("causal_memory_control/interactive_empty_memory.py")
    for path, (before, _) in upgrade.UPGRADE.items():
        config["code_sha256"][path] = before
    config["model_identity"] = {"test": "unchanged"}
    save(root / "benchmark_manifest.json", config)
    for _, child in cli.dataset_jobs(config, "pddl"):
        save(Path(child["output_dir"]) / "benchmark_manifest.json", child)
    save(root / "pipeline_progress.json", {"stage": "collect", "dataset": "pddl/fold_02", "status": "failed"})
    save(root / "pddl_folds/fold_00/pddl/diagnostic/actions/parts/keep.json", {"cached": "old episode"})
    save(root / "alfworld/progress.json", {"stage": "collect", "completed": 43})
    (root / "matrix_results.md").write_text("existing combined report")
    return root, config


class ContinueEmptyMemoryTests(unittest.TestCase):
    def test_reviewed_target_hashes_match_the_actual_upgrade(self):
        for path, (_, after) in upgrade.UPGRADE.items():
            self.assertEqual(sha(REPO / path), after, path)

    def test_plan_has_no_writes_and_old_workers_require_explicit_registration(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root, config = fixture(directory)
            before = files_identity(root)
            plan = upgrade.continue_run(root, datasets=("pddl",), plan_only=True)
            self.assertEqual(plan["writes"], 0)
            self.assertEqual(len(plan["jobs"]), 5)
            self.assertEqual(files_identity(root), before)
            with self.assertRaisesRegex(ValueError, "back up and register"):
                empty.verify_registered_code(config)

    def test_backup_original_bytes_registration_and_dataset_scoped_resume(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root, config = fixture(directory)
            before = files_identity(root)
            calls = []

            def run(command, log, **kwargs):
                self.assertIn("pddl", command)
                self.assertNotIn("alfworld", command)
                job = json.loads(Path(command[command.index("--config") + 1]).read_text())
                empty.verify_registered_code(job)
                calls.append(job["cv_context"]["fold"])

            with mock.patch.object(upgrade, "run_logged", side_effect=run):
                state = upgrade.continue_run(root, datasets=("pddl",), stage="collect")
                self.assertEqual(calls, list(range(5)))
                self.assertEqual(files_identity(Path(state["backup_dir"])), before)
                for path, expected in before.items():
                    self.assertEqual(sha(root / path), expected, path)
                self.assertEqual(json.loads((root / "benchmark_manifest.json").read_text()), config)
                empty.verify_registered_code(config)
                backup = state["backup_dir"]
                again = upgrade.continue_run(root, datasets=("pddl",), stage="collect")
                self.assertEqual(again["backup_dir"], backup)
            # Registration changes neither seed nor action-cache key identity.
            self.assertEqual(digest(json.loads((root / "benchmark_manifest.json").read_text())), digest(config))

    def test_unknown_source_changed_data_and_active_lock_are_rejected_before_backup(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root, config = fixture(directory)
            before = files_identity(root)
            with upgrade.experiment_lock(root), mock.patch.object(upgrade, "backup_results") as backup:
                with self.assertRaisesRegex(ValueError, "still running/locked"):
                    upgrade.continue_run(root)
                backup.assert_not_called()
            bad = deepcopy(config)
            key = next(iter(upgrade.UPGRADE))
            bad["code_sha256"][key] = "unreviewed implementation"
            save(root / "benchmark_manifest.json", bad)
            with mock.patch.object(upgrade, "backup_results") as backup:
                with self.assertRaisesRegex(ValueError, "unsupported source/target"):
                    upgrade.continue_run(root)
                backup.assert_not_called()
            save(root / "benchmark_manifest.json", config)
            with mock.patch.object(upgrade, "backup_results", side_effect=OSError("disk full")), \
                 mock.patch.object(upgrade, "run_logged") as run:
                with self.assertRaisesRegex(OSError, "disk full"):
                    upgrade.continue_run(root)
                run.assert_not_called()
            self.assertEqual(files_identity(root), before)
            self.assertFalse((root / "empty_memory_upgrade").exists())


if __name__ == "__main__":
    unittest.main()
