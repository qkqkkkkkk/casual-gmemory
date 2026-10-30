"""Preserve committed prefixes and provenance when recovering the v9 format abort."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from causal_memory_control import migrate_interactive_format_support as migration
from causal_memory_control import interactive_worker as worker
from causal_memory_control.interactive_data import digest, files_identity, save, sha
from causal_memory_control.run_interactive_gate_benchmark import dataset_jobs


REPO = Path(__file__).resolve().parents[2]


class Runner:
    calls = []
    stop_on = None

    def __init__(self, config, root, memory_dir, **kwargs):
        self.memory = Path(memory_dir)
        self.memory.mkdir(parents=True, exist_ok=True)

    def run(self, task, seed):
        if task["id"] == self.stop_on:
            raise RuntimeError("format abort before commit")
        self.calls.append(task["id"])
        (self.memory / (task["id"] + ".txt")).write_text("unchanged support memory")
        return {"task_id": task["id"], "seed": seed, "outcome": {"success": 1}, "trajectory": []}

    def close(self):
        pass


class FormatMigrationTests(unittest.TestCase):
    def fixture(self, directory):
        source, destination = Path(directory).resolve() / "v9", Path(directory).resolve() / "v10"
        tasks = [{"id": str(i), "files": {"tasks/envs/base_env.py": sha(REPO / "tasks/envs/base_env.py")}}
                 for i in range(6)]
        config = {"repo": str(REPO), "output_dir": str(source), "exclude_ce": True,
            "include_local_global_team_u": True, "code_sha256": {migration.RUNTIME: migration.OLD_RUNTIME_SHA},
            "experiments": {"pddl": {"tasks": tasks[:4], "cv_splits": [
                {"support": ["0", "1"], "fit": [], "dev": [], "test": []},
                {"support": ["2", "3"], "fit": [], "dev": [], "test": []}]},
                "alfworld": {"tasks": tasks[4:], "split": {"support": ["4", "5"], "fit": [], "dev": [], "test": []}}}}
        save(source / "benchmark_manifest.json", config)
        Runner.calls = []
        for name in config["experiments"]:
            for label, job in dataset_jobs(config, name):
                save(Path(job["output_dir"]) / "benchmark_manifest.json", job)
                Runner.stop_on = "5" if name == "alfworld" else None
                with mock.patch("causal_memory_control.interactive_runtime.NativeRunner", Runner):
                    if name == "alfworld":
                        with self.assertRaisesRegex(RuntimeError, "before commit"):
                            worker.support(job, name)
                    else:
                        worker.support(job, name)
                cache = Path(job["output_dir"]) / name / "support/responses.sqlite"
                with sqlite3.connect(cache) as db:
                    db.execute("CREATE TABLE responses (key TEXT PRIMARY KEY, response TEXT NOT NULL)")
                    db.execute("INSERT INTO responses VALUES (?, ?)", ("same-request-key", "```plaintext"))
        Runner.stop_on = None
        return source, destination, config

    def test_copy_keeps_source_bytes_caches_snapshots_and_resumes_only_unfinished_task(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source, destination, old = self.fixture(directory)
            original_files = files_identity(source)
            plan = migration.migrate(source, destination)
            self.assertFalse(destination.exists())
            self.assertEqual(sum(plan["committed_counts"].values()), 5)
            result = migration.migrate(source, destination, apply=True)
            self.assertEqual(result["committed_counts"], plan["committed_counts"])
            self.assertEqual(files_identity(source), original_files)
            new = json.loads((destination / "benchmark_manifest.json").read_text())
            self.assertEqual(new["code_sha256"][migration.RUNTIME], migration.NEW_RUNTIME_SHA)
            allowed = deepcopy(old)
            allowed["output_dir"] = str(destination)
            allowed["code_sha256"][migration.RUNTIME] = migration.NEW_RUNTIME_SHA
            self.assertEqual(new, allowed)
            self.assertEqual(json.loads((destination / "support_migration/source_manifest.json").read_text()), old)
            Runner.calls = []
            for name in new["experiments"]:
                for (label, previous), (_, job) in zip(dataset_jobs(old, name), dataset_jobs(new, name)):
                    root = Path(job["output_dir"]) / name
                    self.assertEqual(json.loads((Path(job["output_dir"]) / "benchmark_manifest.json").read_text()), job)
                    for marker in (root / "support/committed").glob("*.json"):
                        commit = json.loads(marker.read_text())
                        self.assertEqual(commit["config_digest"], digest(job))
                        episode = Path(commit["directory"]).parent / "episode.json"
                        self.assertEqual((root / episode).read_bytes(),
                                         (Path(previous["output_dir"]) / name / episode).read_bytes())
                    with sqlite3.connect(root / "support/responses.sqlite") as db:
                        self.assertEqual(db.execute("SELECT * FROM responses").fetchall(),
                                         [("same-request-key", "```plaintext")])
                    with mock.patch("causal_memory_control.interactive_runtime.NativeRunner", Runner):
                        worker.support(job, name)
                    self.assertTrue(worker.snapshot(job, name).is_dir())
            self.assertEqual(Runner.calls, ["5"])
            self.assertEqual(files_identity(source), original_files)
            # The failed/uncommitted attempt remains only in the source.
            self.assertEqual(len(list((destination / "alfworld/support/attempts").iterdir())), 2)

    def test_unknown_source_or_other_implementation_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source, destination, config = self.fixture(directory)
            path = source / "benchmark_manifest.json"
            bad = deepcopy(config)
            bad["code_sha256"][migration.RUNTIME] = "not-the-v9-runtime"
            save(path, bad)
            with self.assertRaisesRegex(ValueError, "known v9"):
                migration.migrate(source, destination, apply=True)
            bad = deepcopy(config)
            bad["code_sha256"]["tasks/envs/base_env.py"] = "changed"
            save(path, bad)
            with self.assertRaisesRegex(ValueError, "unexpected implementation"):
                migration.migrate(source, destination, apply=True)
            self.assertFalse(destination.exists())

    def test_tampered_episode_and_later_stage_artifacts_are_rejected(self):
        for cause in ("episode", "memory", "later-stage"):
            with self.subTest(cause=cause), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                source, destination, config = self.fixture(directory)
                root = source / "alfworld"
                marker = json.loads((root / "support/committed/0000.json").read_text())
                if cause == "episode":
                    episode = root / Path(marker["directory"]).parent / "episode.json"
                    episode.write_text(episode.read_text() + " ")
                elif cause == "memory":
                    (root / marker["directory"] / "g-memory/4.txt").write_text("changed memory")
                else:
                    save(root / "diagnostic/tasks/measured.json", {})
                with self.assertRaisesRegex(ValueError, "identity changed|later-stage"):
                    migration.migrate(source, destination, apply=True)
                self.assertFalse(destination.exists())

    def test_active_run_existing_destination_and_copy_failure_leave_source_intact(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source, destination, _ = self.fixture(directory)
            original_files = files_identity(source)
            with migration.experiment_lock(source), self.assertRaisesRegex(ValueError, "still running/locked"):
                migration.migrate(source, destination, apply=True)
            with mock.patch.object(migration.shutil, "copytree", side_effect=OSError("disk full")), \
                 self.assertRaisesRegex(OSError, "disk full"):
                migration.migrate(source, destination, apply=True)
            self.assertFalse(destination.exists())
            self.assertFalse(list(destination.parent.glob(destination.name + ".migrating-*")))
            destination.mkdir()
            with self.assertRaisesRegex(ValueError, "nonexistent destination"):
                migration.migrate(source, destination, apply=True)
            self.assertEqual(files_identity(source), original_files)

    def test_runtime_hash_pin_matches_the_actual_fix(self):
        self.assertEqual(sha(REPO / migration.RUNTIME), migration.NEW_RUNTIME_SHA)

    def test_real_cli_configuration_accepts_the_migrated_manifests(self):
        from causal_memory_control import run_interactive_gate_benchmark as cli

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source, destination = Path(directory).resolve() / "v9", Path(directory).resolve() / "v10"
            args = ["--datasets", "pddl", "--pddl-folds", "5", "--exclude-ce",
                    "--include-local-global-team-u", "--stage", "support"]
            previous = cli.configuration(cli.parse_args([*args, "--output-dir", str(source), "--plan-only"]))
            previous["code_sha256"][migration.RUNTIME] = migration.OLD_RUNTIME_SHA
            previous["model_identity"] = {"mock": "same-model"}
            save(source / "benchmark_manifest.json", previous)
            migration.migrate(source, destination, apply=True)
            with mock.patch("causal_memory_control.interactive_runtime.model_identity", return_value=previous["model_identity"]), \
                 mock.patch.object(cli, "run_logged") as run:
                cli.main([*args, "--output-dir", str(destination)])
                self.assertEqual(run.call_count, 5)


if __name__ == "__main__":
    unittest.main()
