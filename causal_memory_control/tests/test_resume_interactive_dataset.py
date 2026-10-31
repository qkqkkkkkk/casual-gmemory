"""Dataset-only resume must preserve registered identities and PDDL progress."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from unittest import mock

from causal_memory_control import interactive_worker as worker
from causal_memory_control import resume_interactive_dataset as resume
from causal_memory_control.interactive_data import digest, files_identity, save, sha
from causal_memory_control.interactive_protocol import learned_methods, policies, stages


REPO = Path(__file__).resolve().parents[2]


def fixture(directory):
    root = Path(directory).resolve() / "v10"
    data = {"protocol": "test-offline",
        "tasks": [{"id": str(i), "files": {}, "group": f"scene:{i}", "task_type": "put"}
                  for i in range(6)],
        "split": {"support": ["0", "1"], "fit": ["2"], "dev": ["3"], "test": ["4", "5"]},
        "primary_test_ids": ["4"], "test_cohorts": {"valid_unseen": ["4"], "train_scene_holdout": ["5"]}}
    config = {"repo": str(REPO), "output_dir": str(root), "exclude_ce": True,
        "include_local_global_team_u": True, "evaluation_seed": 2000, "random_seeds": [11, 23, 47],
        "diagnostic_seed": 0, "repeats": 2,
        "code_sha256": {p: sha(REPO / p) for p in ("causal_memory_control/interactive_runtime.py",
            "causal_memory_control/run_interactive_gate_benchmark.py")},
        "experiments": {"pddl": {"cv_splits": [{"test": []}]}, "alfworld": data}}
    save(root / "benchmark_manifest.json", config)
    save(root / "pipeline_progress.json", {"stage": "collect", "dataset": "pddl/fold_02", "status": "failed"})
    save(root / "pddl_folds/fold_02/pddl/progress.json", {"completed_tasks": 0, "total_tasks": 40})
    save(root / "pddl_folds/fold_00/pddl/diagnostic/tasks/done.json", {"finished": True})
    (root / "matrix_results.md").write_text("keep the existing combined report")
    with sqlite3.connect(root / "responses.sqlite") as db:
        db.execute("CREATE TABLE response (text TEXT)")
        db.execute("INSERT INTO response VALUES ('cached reply')")
    return root, config


class Runner:
    calls = []
    stop_on = None

    def __init__(self, config, root, memory_dir, **kwargs):
        self.memory = Path(memory_dir)
        self.memory.mkdir(parents=True, exist_ok=True)

    def run(self, task, seed):
        if task["id"] == self.stop_on:
            raise RuntimeError("incomplete support")
        self.calls.append(task["id"])
        (self.memory / (task["id"] + ".txt")).write_text("committed memory")
        return {"task_id": task["id"], "seed": seed, "outcome": {"success": 1}}

    def close(self):
        pass


class DatasetResumeTests(unittest.TestCase):
    def test_plan_validates_registered_code_but_does_not_write_or_call_service(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root, config = fixture(directory)
            before = files_identity(root)
            with mock.patch.object(worker, "verify_inputs", side_effect=AssertionError("service")):
                plan = resume.resume(root, plan_only=True)
            self.assertEqual(plan["stages"], list(stages(config)))
            self.assertEqual(plan["preserved_datasets"], ["pddl"])
            self.assertEqual(plan["config_digest"], digest(config))
            self.assertEqual(files_identity(root), before)
            self.assertFalse(Path(str(root) + ".lock").exists())

    def test_backup_and_all_stages_preserve_pddl_and_resume_only_unfinished_support(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root, config = fixture(directory)
            Runner.calls, Runner.stop_on = [], "1"
            with mock.patch("causal_memory_control.interactive_runtime.NativeRunner", Runner):
                with self.assertRaisesRegex(RuntimeError, "incomplete support"):
                    worker.support(config, "alfworld")
            Runner.stop_on = None
            before = files_identity(root)
            completed_episode = next((root / "alfworld/support/attempts").glob("*/episode.json"))
            episode_bytes = completed_episode.read_bytes()
            calls = []

            def run(command, log, *, cwd):
                self.assertEqual(command[command.index("--dataset") + 1], "alfworld")
                self.assertEqual(Path(command[command.index("--config") + 1]), root / "benchmark_manifest.json")
                current = command[command.index("--stage") + 1]
                calls.append(current)
                if current == "support":
                    worker.support(config, "alfworld")

            with mock.patch.object(worker, "verify_inputs"), mock.patch.object(resume, "run_logged", side_effect=run), \
                 mock.patch.object(resume, "dataset_report") as report, \
                 mock.patch("causal_memory_control.interactive_runtime.NativeRunner", Runner):
                Runner.calls = []
                state = resume.resume(root)
                self.assertEqual(Runner.calls, ["1"])
                self.assertEqual(calls, list(stages(config))[:-1])
                self.assertEqual(report.call_args.args, (config, "alfworld"))
                self.assertEqual(report.call_args.args[0]["experiments"]["pddl"], config["experiments"]["pddl"])
                backup = Path(state["backup_dir"])
                self.assertEqual(files_identity(backup), before)
                Runner.calls = []
                again = resume.resume(root, stage="support")
                self.assertEqual(again["backup_dir"], str(backup))
                self.assertEqual(Runner.calls, [])
            self.assertEqual(completed_episode.read_bytes(), episode_bytes)
            self.assertEqual(state["status"], "completed")
            for path, identity in before.items():
                if not path.startswith("alfworld/"):
                    self.assertEqual(sha(root / path), identity, path)

    def test_active_lock_and_copy_failure_never_start_a_worker(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root, _ = fixture(directory)
            before = files_identity(root)
            with mock.patch.object(resume, "run_logged") as run:
                with resume.experiment_lock(root), self.assertRaisesRegex(ValueError, "still running/locked"):
                    resume.resume(root)
                with mock.patch.object(resume.shutil, "copytree", side_effect=OSError("disk full")), \
                     self.assertRaisesRegex(OSError, "disk full"):
                    resume.resume(root)
                run.assert_not_called()
            self.assertEqual(files_identity(root), before)
            self.assertFalse(list(root.parent.glob("*.copying-*")))

    def test_code_change_or_existing_backup_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root, config = fixture(directory)
            with self.assertRaisesRegex(ValueError, "new independent"):
                resume.resume(root, backup_dir=root)
            bad = deepcopy(config)
            bad["code_sha256"]["causal_memory_control/interactive_runtime.py"] = "changed"
            save(root / "benchmark_manifest.json", bad)
            with self.assertRaisesRegex(ValueError, "implementation changed"):
                resume.resume(root)
            self.assertFalse((root / "dataset_resume").exists())

    def test_insufficient_space_preserves_source_and_never_starts_execution(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root, _ = fixture(directory)
            before = files_identity(root)
            usage = shutil.disk_usage(root)
            with mock.patch.object(resume.shutil, "disk_usage", return_value=usage._replace(free=0)), \
                 mock.patch.object(resume, "run_logged") as run, \
                 self.assertRaisesRegex(ValueError, "insufficient backup disk space"):
                resume.resume(root)
            run.assert_not_called()
            self.assertEqual(files_identity(root), before)

    def test_worker_failure_retains_backup_and_original_pipeline_progress(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root, _ = fixture(directory)
            before = files_identity(root)
            with mock.patch.object(worker, "verify_inputs"), \
                 mock.patch.object(resume, "run_logged", side_effect=subprocess.CalledProcessError(1, ["worker"])), \
                 self.assertRaises(subprocess.CalledProcessError):
                resume.resume(root, stage="collect")
            state = json.loads((root / "dataset_resume/alfworld/progress.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["stage"], "collect")
            self.assertEqual(files_identity(Path(state["backup_dir"])), before)
            self.assertEqual(files_identity(root)["pipeline_progress.json"], before["pipeline_progress.json"])

    def test_report_uses_full_digest_preserves_other_outputs_and_separates_cohorts(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root, config = fixture(directory)
            save(root / "alfworld/snapshot.json", {"test": True})
            for method in learned_methods(config):
                save(root / "alfworld" / method / "checkpoint.json", {"test": method})
            rows = []
            for task_id in config["experiments"]["alfworld"]["split"]["test"]:
                success = float(task_id == "5")
                baseline = {"task_id": task_id, "seed": 2000, "initial_fingerprint": task_id,
                    "retrieval_fingerprint": "retrieval-" + task_id, "candidates": [], "action_key": task_id,
                    "outcome": {"success": success, "progress": None}}
                for method in [m for m, _ in policies(config)] + [f"random_drop_seed{s}" for s in config["random_seeds"]]:
                    rows.append(worker.policy_record(config, method, baseline, baseline, []))
            path = root / "alfworld/evaluation/policies.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            before = files_identity(root)
            with mock.patch.object(worker, "snapshot"):
                payload = resume.dataset_report(config, "alfworld")
            self.assertEqual(payload["config_digest"], digest(config))
            stats = payload["policies"]["always_use"]
            self.assertEqual((stats["n"], stats["success"]), (1, 0))
            self.assertEqual(stats["cohort_results"]["train_scene_holdout"]["success"], 1)
            self.assertEqual(len(payload["policies"]), 11)
            self.assertTrue((root / "alfworld/dataset_report/comparison.csv").exists())
            for relative, identity in before.items():
                self.assertEqual(sha(root / relative), identity, relative)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
            with mock.patch.object(worker, "snapshot"), self.assertRaisesRegex(ValueError, "incomplete"):
                resume.dataset_report(config, "alfworld")
            path.write_text("".join(json.dumps({**row, "config_digest": "filtered-config-digest"}) + "\n" for row in rows))
            with mock.patch.object(worker, "snapshot"), self.assertRaisesRegex(ValueError, "unregistered"):
                resume.dataset_report(config, "alfworld")


if __name__ == "__main__":
    unittest.main()
