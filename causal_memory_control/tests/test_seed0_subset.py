from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import importlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from causal_memory_control import run_seed0_gate_matrix as seed0, train_gate
from causal_memory_control import fever_experiment, hotpotqa_experiment
from causal_memory_control.tests.test_seed0_gate_matrix import prepare_source
from causal_memory_control.tests.test_st_gate import _STMemory, _FeverSTMacNet
from causal_memory_control.tests.test_semantic_jev import TinyEncoder
from causal_memory_control.tests.test_full_gate_benchmark import cache
from causal_memory_control.tests.test_collection_runtime import Backend
from causal_memory_control.tests import test_fever_experiment as fever_tests
from causal_memory_control.tests import test_hotpotqa_gate as hotpot_tests
from mas.memory.common import MASMessage


def options(root, dataset):
    return ["--source-root", str(root / "source"), "--output-dir", str(root / "quick"),
        "--datasets", dataset, "--diagnostic-tasks", "10", "--collection-repeats", "2",
        "--collect-missing", "--include-always-global-drop"]


class CollectionMemory:
    def __init__(self, **_):
        self.memory_size = 2

    def retrieve_memory(self, **_):
        return ([MASMessage(task_main="example memory", task_description="example state",
            task_trajectory="Finish[New York City]", label=True)], [], ["Read evidence."])


class Seed0SubsetTests(unittest.TestCase):
    def test_plan_keeps_original_test_partition_and_cli_validation(self):
        for dataset in ("fever", "hotpotqa"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                root = Path(directory)
                _, original = prepare_source(root, dataset)
                plan = seed0.main([*options(root, dataset), "--plan-only"])[dataset]
                self.assertEqual(plan["gate_training_questions"], 10)
                self.assertEqual(plan["original_diagnostic_partition_questions"], 16)
                self.assertEqual(plan["unused_diagnostic_questions"], 6)
                self.assertEqual(plan["diagnostic_branches_per_seed"], 80)
                self.assertEqual(plan["maximum_executed_diagnostic_schedules"], 60)
                self.assertEqual(plan["sample_seeds"], [0, 1])
                self.assertEqual(plan["final_test_offset"], original.training_claims)
                self.assertEqual(plan["final_test_questions"], original.final_claims)
                self.assertIn(str(root / "quick" / "artifacts" / dataset), plan["training_sources"][0])
                self.assertFalse((root / "quick").exists())
                for bad in (["--collect-missing"], ["--collection-repeats", "2"], ["--diagnostic-tasks", "4"],
                            ["--diagnostic-tasks", "10", "--collection-repeats", "0"]):
                    with self.assertRaises(SystemExit):
                        seed0.parse_args(["--source-root", str(root / "source"), *bad])
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    seed0.main([*options(root, dataset), "--diagnostic-tasks", "17", "--plan-only"])

    def test_collects_only_subset_resumes_and_trains_five_methods_without_touching_source(self):
        for dataset in ("fever", "hotpotqa"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                adapter, original = prepare_source(root, dataset)
                source_cache = seed0.source_path(original).parent / "llm_cache.sqlite"
                cache(source_cache, {"original_cached_request": "original_cached_answer"})
                before = {p: p.read_bytes() for p in (root / "source").rglob("*") if p.is_file()}
                collector = importlib.import_module(f"causal_diagnostic.{dataset}_oracle_recipient.run_experiment")
                runner = fever_experiment if dataset == "fever" else hotpotqa_experiment
                snapshot = seed0.base._load_json(original.memory_dir / "causal_snapshot_manifest.json")
                stack.enter_context(mock.patch.object(collector, "_load_snapshot", return_value=snapshot))
                stack.enter_context(mock.patch.object(collector, "_directory_hash", return_value="frozen-content"))
                stack.enter_context(mock.patch.object(collector, "GMemory", CollectionMemory))
                stack.enter_context(mock.patch.object(collector, "EmbeddingFunc", return_value=None))
                backend = Backend(dataset)
                stack.enter_context(mock.patch("causal_diagnostic.oracle_recipient.seeded_client.OpenAI", return_value=backend))
                if dataset == "fever":
                    for patch in fever_tests.FeverRunnerTests._patches():
                        stack.enter_context(patch)
                    stack.enter_context(mock.patch.object(runner, "MacNet", _FeverSTMacNet))
                else:
                    hotpot_tests.HotpotRunnerTests.patches(stack)
                stack.enter_context(mock.patch.object(runner, "GMemory", _STMemory))
                stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", return_value=TinyEncoder()))
                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(redirect_stderr(io.StringIO()))
                commands, branch_calls = [], [0]
                branch = collector._run_branch

                def interrupt(*args, **kwargs):
                    branch_calls[0] += 1
                    if branch_calls[0] == 7:
                        raise RuntimeError("simulated subset collection interruption")
                    return branch(*args, **kwargs)

                def execute(label, command, log, **_):
                    commands.append((label, command))
                    if label.endswith(":collect_seed0"):
                        collector.main(command[4:])
                    elif label.endswith(":train"):
                        train_gate.main(command[4:])
                    elif label.endswith(":baseline") or label.endswith(":evaluate"):
                        self.assertEqual(command[command.index("--evaluation-offset") + 1], "16")
                        self.assertEqual(command[command.index("--claims") + 1], "4")
                        runner.main(command[4:])
                        if label.endswith(":baseline"):
                            cache(Path(command[command.index("--output-dir") + 1]) / "llm_cache.sqlite")
                    else:
                        self.fail(f"unexpected snapshot/retest/other method: {label}")

                stack.enter_context(mock.patch.object(seed0.base, "_run_command", side_effect=execute))
                argv = options(root, dataset)
                with mock.patch.object(collector, "_run_branch", side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, "subset collection interruption"):
                        seed0.main(argv)
                branch_file = root / "quick" / "artifacts" / dataset / "diagnostic" / "seed0" / "branches.jsonl"
                prefix = branch_file.read_bytes()
                state = seed0.base._load_json(root / "quick" / "seed0_progress.json")
                self.assertEqual(state["error"]["stage"], dataset + ":collect_seed0")
                commands.clear()
                result_path = seed0.main(argv)
                collect_command = commands[0][1]
                self.assertTrue(commands[0][0].endswith(":collect_seed0"))
                self.assertIn("--resume", collect_command)
                self.assertEqual(collect_command[collect_command.index("--cache-seed-from") + 1], str(source_cache))
                self.assertTrue(branch_file.read_bytes().startswith(prefix))
                rows = [json.loads(line) for line in branch_file.read_text().splitlines()]
                subset_ids = set(snapshot["evaluation_ids"][:10])
                final_ids = set(snapshot["evaluation_ids"][16:])
                self.assertEqual(len(rows), 80)
                self.assertEqual({r["task_id"] for r in rows}, subset_ids)
                self.assertEqual({r["sample_seed"] for r in rows}, {0, 1})
                with sqlite3.connect(branch_file.parent / "llm_cache.sqlite") as db:
                    self.assertEqual(db.execute("SELECT response FROM responses WHERE key=?", ("original_cached_request",)).fetchone()[0], "original_cached_answer")
                splits = []
                for arm in seed0.matrix.DEFAULT_ARMS:
                    checkpoint = seed0.base._load_json(root / "quick" / dataset / "checkpoints" / (arm + ".json"))
                    meta = checkpoint["training_metadata"]
                    self.assertEqual(set(meta["training_task_ids"]), subset_ids)
                    fit = {int(t) for t in meta["fit_task_ids"]}
                    dev = {int(t) for t in meta["policy_calibration"]["calibration_task_ids"]}
                    self.assertEqual((len(fit), len(dev)), (8, 2))
                    self.assertFalse(fit & dev or (fit | dev) & final_ids)
                    splits.append((fit, dev))
                    gate_rows = [json.loads(line) for line in (root / "quick" / dataset / arm / "memory_gate.jsonl").read_text().splitlines()]
                    self.assertEqual({r["task_id"] for r in gate_rows}, final_ids)
                self.assertEqual(splits[0], splits[1])
                self.assertEqual(splits[1], splits[2])
                report = json.loads(result_path.read_text())
                self.assertEqual(len(report["results"][dataset]), 5)
                self.assertTrue(all(r["paired_control_valid"] for r in report["details"][dataset].values()))
                for p, contents in before.items():
                    self.assertEqual(p.read_bytes(), contents, str(p))
                commands.clear()
                seed0.main(argv)
                self.assertFalse(commands)
                # Changed sample sizes/repeats cannot resume trained artifacts.
                for changed in (["--diagnostic-tasks", "9"], ["--collection-repeats", "1"]):
                    with self.assertRaisesRegex(ValueError, "plan changed"):
                        seed0.main([*argv, *changed])

    def test_subset_never_collects_implicitly_or_seeds_an_incompatible_cache(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            _, original = prepare_source(root, "hotpotqa")
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            argv = options(root, "hotpotqa")
            argv.remove("--collect-missing")
            with mock.patch.object(seed0.base, "_run_command") as run:
                with self.assertRaises(FileNotFoundError):
                    seed0.main(argv)
                run.assert_not_called()
            cache(seed0.source_path(original).parent / "llm_cache.sqlite")
            path = seed0.source_path(original).parent / "run_manifest.json"
            manifest = json.loads(path.read_text())
            manifest["run"]["endpoint"] = "http://another-model-service/v1"
            path.write_text(json.dumps(manifest))
            with mock.patch.object(seed0.base, "_run_command") as run:
                with self.assertRaisesRegex(RuntimeError, "source cache run"):
                    seed0.main([*argv, "--collect-missing"])
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
