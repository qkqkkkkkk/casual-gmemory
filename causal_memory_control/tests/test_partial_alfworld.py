"""Partial diagnostics are an explicit independent experiment, never truncation in place."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from causal_memory_control import run_partial_alfworld as partial
from causal_memory_control import interactive_worker as worker
from causal_memory_control.interactive_data import digest, files_identity, save
from causal_memory_control.interactive_runtime import ActionStore, baseline_for
from causal_memory_control.interactive_team_u import measured_lists
from causal_memory_control.resume_interactive_dataset import experiment_lock
from causal_memory_control.tests.test_interactive_empty_memory import empty_runner
from causal_memory_control.tests.test_interactive_team_u import fixture, collect


def source_fixture(directory, *, empty=False):
    source, config, runner, calls = fixture(Path(directory) / "source", "alfworld")
    config.update(exclude_ce=True, code_sha256={}, ensemble_size=2,
                  model="mock", endpoint="mock://offline", model_identity={})
    data = config["experiments"]["alfworld"]
    data.update(primary_test_ids=["4"], test_cohorts={"valid_unseen": ["4"], "train_scene_holdout": ["5"]})
    for task in data["tasks"]:
        task["files"] = {}
    memory = source / "alfworld/support/frozen/g-memory"
    memory.mkdir(parents=True)
    save(memory / "insights.json", [])
    with sqlite3.connect(memory / "chroma.sqlite3") as connection:
        connection.execute("create table example(value text)")
        connection.execute("insert into example values ('frozen memory')")
    save(source / "benchmark_manifest.json", config)
    save(source / "alfworld/snapshot.json", {"schema": "interactive-frozen-support-v1",
        "config_digest": digest(config), "support_ids": ["0"], "memory_dir": "support/frozen/g-memory",
        "files_sha256": files_identity(memory), "support_episode_sha256": {}})
    if empty:
        runner = empty_runner(calls)
    collect(config, "alfworld", runner)
    # Completion order must not be treated as all fit followed by all dev.
    missing_path = source / "alfworld/diagnostic/tasks" / (digest("2") + ".json")
    missing = partial.read(missing_path)
    missing_path.unlink()
    with sqlite3.connect(source / "alfworld/responses.sqlite") as connection:
        connection.execute("create table example(value text)")
        connection.execute("insert into example values ('cached response')")
    return source, config, runner, calls, missing


class PartialALFWorldTests(unittest.TestCase):
    def test_cache_backup_includes_committed_wal(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "source.sqlite", Path(directory) / "copy.sqlite"
            with sqlite3.connect(source) as writer:
                writer.execute("pragma journal_mode=wal")
                writer.execute("create table example(value text)")
                writer.execute("insert into example values ('committed in WAL')")
                writer.commit()
                self.assertTrue(Path(str(source) + "-wal").is_file())
                partial.copy_sqlite(source, destination)
                with sqlite3.connect(destination) as reader:
                    self.assertEqual(reader.execute("select value from example").fetchall(), [("committed in WAL",)])

    def test_prepare_preserves_source_and_exact_outcomes_split_memory_and_cohorts(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source, original, _, _, _ = source_fixture(directory)
            save(source / "alfworld/semantic_st/checkpoint.json", {"must_not_be_reused": True})
            before = files_identity(source)
            output = Path(directory) / "partial"
            with mock.patch("causal_memory_control.interactive_runtime.NativeRunner", side_effect=AssertionError("no execution")), \
                 mock.patch("causal_memory_control.interactive_runtime.model_identity", side_effect=AssertionError("no service")):
                partial.run(source, output, stage="prepare", expected_completed=2, train_device="cpu")
            self.assertEqual(before, files_identity(source))
            config = partial.read(output / "benchmark_manifest.json")
            data = config["experiments"]["alfworld"]
            self.assertEqual(data["split"], {"support": ["0"], "fit": ["1"], "dev": ["3"], "test": ["4", "5"]})
            for key in ("primary_test_ids", "test_cohorts"):
                self.assertEqual(data[key], original["experiments"]["alfworld"][key])
            self.assertFalse((output / "alfworld/semantic_st/checkpoint.json").exists())
            rows, _ = measured_lists(config, "alfworld")
            self.assertEqual([r["task_id"] for r in rows], ["1", "3"])
            self.assertEqual([r["team_u"] for r in rows], [[-.5, .5, 0], [-.5, .5, 0]])
            old_identity = partial.read(source / "alfworld/diagnostic/actions/identity.json")
            new_identity = partial.read(output / "alfworld/diagnostic/actions/identity.json")
            self.assertNotEqual(old_identity, new_identity)
            for path in (output / "alfworld/diagnostic/actions/parts").glob("*.json"):
                new = partial.read(path)
                old = partial.read(source / "alfworld/diagnostic/actions/parts" / path.name)
                self.assertEqual({k: v for k, v in new.items() if k not in ("input_identity", "record_sha256")},
                                 {k: v for k, v in old.items() if k not in ("input_identity", "record_sha256")})
            store = ActionStore(output / "alfworld/diagnostic/actions", new_identity)
            forbidden = mock.Mock()
            forbidden.run.side_effect = AssertionError("baseline must be reused")
            self.assertEqual(baseline_for(forbidden, store, {"id": "1"}, 0)["task_id"], "1")
            with sqlite3.connect(output / "alfworld/frozen_support/g-memory/chroma.sqlite3") as connection:
                self.assertEqual(connection.execute("select value from example").fetchall(), [("frozen memory",)])
            with sqlite3.connect(output / "alfworld/responses.sqlite") as connection:
                self.assertEqual(connection.execute("select value from example").fetchall(), [("cached response",)])

    def test_resume_keeps_frozen_subset_even_when_source_collection_grows(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source, _, _, _, missing = source_fixture(directory)
            output = Path(directory) / "partial"
            partial.run(source, output, stage="prepare", expected_completed=2)
            before = files_identity(output)
            save(source / "alfworld/diagnostic/tasks" / (digest("2") + ".json"), missing)
            partial.run(source, output, stage="prepare", expected_completed=2)
            self.assertEqual(before, files_identity(output))
            with self.assertRaisesRegex(ValueError, "fixed task count"):
                partial.run(source, output, stage="prepare", expected_completed=3)
            config, _ = partial.prepared(output, source)
            self.assertNotIn("2", config["experiments"]["alfworld"]["split"]["fit"])

    def test_plan_does_not_write_or_call_service_or_load_models(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source, _, _, _, _ = source_fixture(directory)
            output = Path(directory) / "partial"
            before = files_identity(Path(directory))
            with mock.patch("causal_memory_control.interactive_runtime.model_identity", side_effect=AssertionError("service")), \
                 mock.patch("causal_memory_control.interactive_runtime.NativeRunner", side_effect=AssertionError("model")):
                plan = partial.run(source, output, stage="plan", expected_completed=2)
            self.assertEqual(plan["completed_tasks"], 2)
            self.assertEqual(plan["split_counts"], {"support": 1, "fit": 1, "dev": 1, "test": 2})
            self.assertEqual(before, files_identity(Path(directory)))
            self.assertFalse(output.exists())

    def test_invalid_source_is_rejected_without_publishing_or_changing_source(self):
        for failure in ("missing_branch", "branch_hash", "aggregate", "candidate", "source_identity"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                source, _, _, _, _ = source_fixture(directory)
                path = source / "alfworld/diagnostic/tasks" / (digest("1") + ".json")
                raw = partial.read(path)
                if failure in ("missing_branch", "branch_hash"):
                    branch = source / "alfworld/diagnostic/actions/parts" / (raw["branches"][0]["drop_key"] + ".json")
                    if failure == "missing_branch":
                        branch.unlink()
                    else:
                        record = partial.read(branch)
                        record["outcome"]["success"] = .25
                        save(branch, record)
                elif failure == "source_identity":
                    path = source / "alfworld/diagnostic/actions/identity.json"
                    save(path, {"changed": True})
                else:
                    if failure == "aggregate":
                        raw["q_use"] = .25
                    else:
                        raw["candidates"][0]["content"] = "tampered"
                    save(path, raw)
                before = files_identity(source)
                output = Path(directory) / "partial"
                with self.assertRaises(ValueError):
                    partial.run(source, output, stage="prepare", expected_completed=2)
                self.assertFalse(output.exists())
                self.assertFalse(list(output.parent.glob(output.name + ".preparing-*")))
                self.assertEqual(before, files_identity(source))

    def test_source_lock_and_expected_count_and_empty_dev_are_respected(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source, _, _, _, _ = source_fixture(directory)
            output = Path(directory) / "partial"
            with experiment_lock(source), self.assertRaisesRegex(ValueError, "running/locked"):
                partial.run(source, output, stage="prepare")
            with self.assertRaisesRegex(ValueError, "expected 3 completed"):
                partial.run(source, output, stage="prepare", expected_completed=3)
            with self.assertRaisesRegex(ValueError, "independent"):
                partial.run(source, source / "partial", stage="prepare")
            (source / "alfworld/diagnostic/tasks" / (digest("3") + ".json")).unlink()
            with self.assertRaisesRegex(ValueError, "no original dev"):
                partial.run(source, output, stage="prepare")
            self.assertFalse(output.exists())

    def test_resume_detects_changes_to_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source, _, _, _, _ = source_fixture(directory)
            output = Path(directory) / "partial"
            partial.run(source, output, stage="prepare", expected_completed=2)
            path = output / "alfworld/diagnostic/tasks" / (digest("1") + ".json")
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "frozen partial diagnostic inputs changed"):
                partial.run(source, output, stage="prepare", expected_completed=2)

    def test_empty_subset_runs_all_existing_stages_reports_full_test_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source, _, runner, calls, _ = source_fixture(directory, empty=True)
            before = files_identity(source)
            output = Path(directory) / "partial"
            stages = []

            def execute(command, path, *, cwd):
                args = command[command.index("--config"):]
                stages.append(args[args.index("--stage") + 1])
                worker.main(args)

            with mock.patch.object(partial, "run_logged", side_effect=execute), \
                 mock.patch("causal_memory_control.interactive_runtime.NativeRunner", runner), \
                 mock.patch("causal_memory_control.interactive_runtime.model_identity", return_value={}), \
                 mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", side_effect=AssertionError("no encoder")):
                partial.run(source, output, stage="all", expected_completed=2, train_device="cpu")
                count = len(calls)
                partial.run(source, output, stage="all", expected_completed=2, train_device="cpu")
            self.assertEqual(len(calls), count)
            self.assertEqual(stages, list(partial.STAGES[:-1]) * 2)
            self.assertEqual(before, files_identity(source))
            report = partial.read(output / "results.json")
            self.assertEqual(report["partial_diagnostic"]["completed_tasks"], 2)
            self.assertEqual(report["partial_diagnostic"]["registered_diagnostic_tasks"], 3)
            self.assertEqual(report["policies"]["always_use"]["n"], 1)  # primary, separate holdout also evaluated
            self.assertEqual(report["policies"]["always_use"]["cohort_results"]["train_scene_holdout"]["n"], 1)
            self.assertEqual(report["test_cohorts"], {"valid_unseen": ["4"], "train_scene_holdout": ["5"]})
            self.assertIn("部分诊断训练结果", (output / "results.md").read_text())
            self.assertFalse(list(output.rglob("checkpoint.json")))  # explicit no-supervision statuses instead


if __name__ == "__main__":
    unittest.main()
