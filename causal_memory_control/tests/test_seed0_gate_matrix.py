from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from causal_memory_control import run_full_gate_benchmark as full
from causal_memory_control import run_seed0_gate_matrix as seed0
from causal_memory_control import fever_experiment, hotpotqa_experiment, train_gate
from causal_memory_control.dataset_adapters import load_dataset_adapter
from causal_memory_control.tests.test_dataset_matrix import fever_source
from causal_memory_control.tests.test_semantic_jev import TinyEncoder
from causal_memory_control.tests.test_st_gate import create_diagnostics, _STMemory, _FeverSTMacNet
from causal_memory_control.tests.test_full_gate_benchmark import cache
from causal_memory_control.tests import test_fever_experiment as fever_tests
from causal_memory_control.tests import test_hotpotqa_gate as hotpot_tests
from causal_diagnostic.hotpotqa_oracle_recipient.data import load_hotpotqa


def prepare_source(root, dataset):
    if dataset == "fever":
        data, examples = fever_source(root, count=22)
    else:
        data = root / "hotpot.json"
        data.write_text(json.dumps([{"_id": f"h{i}", "question": f"Where is site {i}?", "answer": "New York City",
            "context": [["Place", ["The site is in New York City."]]], "supporting_facts": [["Place", 0]]} for i in range(22)]))
        examples = load_hotpotqa(data)
    suite = full.parse_args(["--datasets", dataset, f"--{dataset}-data", str(data),
        "--output-dir", str(root / "source"), "--support-count", "2", "--smoke-claims", "1",
        "--node-num", "2", "--successful-topk", "1", "--insights-topk", "1",
        "--model", "fake-model", "--embedding-model", "fake-embedding",
        "--semantic-model", "test-encoder", "--semantic-revision", "fixed-test",
        "--mlp-hidden-sizes", "8,4", "--mlp-epochs", "2", "--contrastive-epochs", "1", "--bootstrap-samples", "100"])
    adapter, args, _ = full.prepare_dataset(suite, dataset)
    full._save(root / "source" / "benchmark_manifest.json", {"schema": full.SCHEMA,
        "experiments": {dataset: adapter.configuration(args)}, "semantic": {dataset: {k: getattr(args, k)
        for k in ("semantic_model", "semantic_revision", "semantic_cache", "semantic_chunk_tokens",
                  "mlp_hidden_sizes", "mlp_epochs", "mlp_learning_rate", "mlp_batch_tasks", "mlp_device")}}})
    create_diagnostics(adapter, args, examples[2:])
    snapshot = seed0.base._load_json(args.memory_dir / "causal_snapshot_manifest.json")
    path = seed0.source_path(args)
    design = {**adapter.checkpoint_context(args, snapshot),
        "candidate_scope": "all_retrieved", "collection_scope": "gate_training",
        "candidate_specs": [{"kind": "trajectory", "index": 0}, {"kind": "insight", "index": 0}],
        "evaluation_ids": snapshot["evaluation_ids"][:args.training_claims],
        **{k: getattr(args, k) for k in ("repeats", "temperature", "selection_seed")}}
    run = {**design, "endpoint": args.endpoint, "sample_seed_base": args.sample_seed_base}
    manifest = {"design": design, "run": run, "design_hash": seed0.stable_hash(design), "run_hash": seed0.stable_hash(run)}
    full._save(path.parent / "run_manifest.json", manifest)
    full._save(path.parent / "collection_progress.json", {"status": "completed"})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row.update(run_hash=manifest["run_hash"], design_hash=manifest["design_hash"])
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return adapter, args


class Seed0MatrixTests(unittest.TestCase):
    def test_global_drop_extends_completed_runs_without_retraining_and_resumes_per_task(self):
        for dataset, reuse in (("fever", True), ("hotpotqa", False)):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                adapter, args = prepare_source(root, dataset)
                runner = fever_experiment if dataset == "fever" else hotpotqa_experiment
                if dataset == "fever":
                    for patch in fever_tests.FeverRunnerTests._patches():
                        stack.enter_context(patch)
                    parent = _FeverSTMacNet
                else:
                    hotpot_tests.HotpotRunnerTests.patches(stack)
                    parent = hotpot_tests._HotpotMacNet
                stack.enter_context(mock.patch.object(runner, "GMemory", _STMemory))
                stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", return_value=TinyEncoder()))
                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(redirect_stderr(io.StringIO()))
                scheduled = []
                failures = [False]

                class ControlMacNet(parent):
                    def schedule(self, task):
                        if self.memory.gate.mode == "planned_drop":
                            scheduled.append(task["id"])
                            if failures[0] and len(scheduled) == 2:
                                failures[0] = False
                                raise RuntimeError("simulated no-memory model timeout")
                        return super().schedule(task)

                stack.enter_context(mock.patch.object(runner, "MacNet", ControlMacNet))
                if reuse:
                    native = args.output_dir / "native_gmemory"
                    runner.main(adapter.evaluation_command(args, mode="always_keep", output=native, resume=False)[4:])
                    cache(native / "llm_cache.sqlite")
                originals = {p: p.read_bytes() for p in (root / "source").rglob("*") if p.is_file()}
                commands = []

                def execute(label, command, log, **_):
                    commands.append((label, command))
                    if label.endswith(":train"):
                        train_gate.main(command[4:])
                    elif label.endswith(":baseline") or label.endswith(":evaluate"):
                        runner.main(command[4:])
                        if label.endswith(":baseline"):
                            cache(Path(command[command.index("--output-dir") + 1]) / "llm_cache.sqlite")
                    else:
                        self.fail(f"unexpected collector or method: {label}")

                stack.enter_context(mock.patch.object(seed0.base, "_run_command", side_effect=execute))
                argv = ["--source-root", str(root / "source"), "--output-dir", str(root / "new"), "--datasets", dataset]
                seed0.main(argv)
                unchanged = {p: p.read_bytes() for arm in ("native_gmemory", *seed0.matrix.DEFAULT_ARMS)
                    for p in (root / "new" / dataset / arm).rglob("*") if p.is_file()}
                unchanged.update({p: p.read_bytes() for p in (root / "new" / dataset / "checkpoints").glob("*.json")})
                manifest = (root / "new" / "seed0_manifest.json").read_bytes()
                commands.clear()
                extended = [*argv, "--include-always-global-drop"]
                plan = seed0.main([*extended, "--plan-only"])[dataset]
                self.assertEqual(plan["controls"], ["always_global_drop"])
                self.assertEqual(plan["maximum_test_team_inferences"], 5 * args.final_claims)
                self.assertFalse((root / "new" / dataset / "plans").exists())
                failures[0] = True
                with self.assertRaisesRegex(RuntimeError, "no-memory model timeout"):
                    seed0.main(extended)
                self.assertEqual([label for label, _ in commands], [dataset + ":always_global_drop:evaluate"])
                output = root / "new" / dataset / "always_global_drop"
                prefix = (output / "memory_gate.jsonl").read_bytes()
                self.assertEqual(len(prefix.splitlines()), 1)
                progress = seed0.base._load_json(root / "new" / "seed0_progress.json")
                self.assertEqual(progress["error"]["stage"], dataset + ":always_global_drop:evaluate")
                self.assertIn("no-memory model timeout", progress["error"]["traceback"])
                commands.clear()
                result_path = seed0.main(extended)
                self.assertEqual([label for label, _ in commands], [dataset + ":always_global_drop:evaluate"])
                command = commands[0][1]
                self.assertIn("--resume", command)
                self.assertEqual(command[command.index("--max-drops") + 1], "-1")
                self.assertNotIn("--checkpoint", command)
                self.assertTrue((output / "memory_gate.jsonl").read_bytes().startswith(prefix))
                rows = [json.loads(line) for line in (output / "memory_gate.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), args.final_claims)
                self.assertEqual(len(scheduled), args.final_claims + 1)
                for row in rows:
                    self.assertEqual(row["mode"], "planned_drop")
                    self.assertEqual(row["dropped_count"], row["retrieved_count"])
                    self.assertEqual(row["kept_count"], 0)
                    self.assertEqual({d["memory_type"] for d in row["decisions"]}, {"trajectory", "insight"})
                    self.assertTrue(all(d["action"] == "drop" for d in row["decisions"]))
                results = json.loads(result_path.read_text())
                self.assertEqual([row["method"] for row in results["results"][dataset]],
                    ["native_gmemory", *seed0.matrix.DEFAULT_ARMS, "always_global_drop"])
                self.assertTrue(results["details"][dataset]["always_global_drop"]["paired_control_valid"])
                self.assertEqual(results["results"][dataset][-1]["keep_rate"], 0)
                self.assertIn("always_global_drop", (output.parent / "matrix_results.csv").read_text())
                self.assertEqual((root / "new" / "seed0_manifest.json").read_bytes(), manifest)
                for p, data in {**originals, **unchanged}.items():
                    self.assertEqual(p.read_bytes(), data, str(p))
                commands.clear()
                seed0.main(extended)
                self.assertFalse(commands)
                # Never hide retrieval drift by merely recomputing the report.
                plan_path = output.parent / "plans" / "always_global_drop.json"
                payload = json.loads(plan_path.read_text())
                first = next(iter(payload["tasks"].values()))
                first["drop_memory_ids"] = first["drop_memory_ids"][:1]
                plan_path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "plan changed"):
                    seed0.main(extended)

    def test_plan_and_strict_coverage_reject_partial_duplicates_and_changed_seed(self):
        for dataset in ("fever", "hotpotqa"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                root = Path(directory)
                adapter, args = prepare_source(root, dataset)
                argv = ["--source-root", str(root / "source"), "--output-dir", str(root / "new"), "--datasets", dataset]
                plan = seed0.main([*argv, "--plan-only"])[dataset]
                self.assertEqual(plan["new_diagnostic_branches"], 0)
                self.assertEqual(plan["sample_seeds"], [0, 1])
                self.assertEqual(len(plan["training_sources"]), 1)
                self.assertFalse((root / "new").exists())
                source = seed0.source_path(args)
                rows = source.read_text().splitlines()
                report = seed0.validate_seed0(adapter, args)
                self.assertEqual(report["tasks"], 16)
                self.assertEqual(report["branches"], 128)
                source.write_text("\n".join(rows[:-1]) + "\n")
                with self.assertRaisesRegex(ValueError, "incomplete seed0"):
                    seed0.validate_seed0(adapter, args)
                source.write_text("\n".join([*rows, rows[0]]) + "\n")
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    seed0.validate_seed0(adapter, args)
                bad = json.loads(rows[0])
                bad["sample_seed"] = 999
                source.write_text(json.dumps(bad) + "\n" + "\n".join(rows[1:]) + "\n")
                with self.assertRaisesRegex(ValueError, "seed"):
                    seed0.validate_seed0(adapter, args)

    def test_both_datasets_train_all_three_arms_resume_and_keep_source_read_only(self):
        for dataset, reuse in (("fever", True), ("hotpotqa", False)):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                adapter, args = prepare_source(root, dataset)
                runner = fever_experiment if dataset == "fever" else hotpotqa_experiment
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
                if reuse:
                    native = args.output_dir / "native_gmemory"
                    command = adapter.evaluation_command(args, mode="always_keep", output=native, resume=False)
                    runner.main(command[4:])
                    cache(native / "llm_cache.sqlite")
                originals = {str(p): p.read_bytes() for p in (root / "source").rglob("*") if p.is_file()}
                commands, failures = [], [True]

                def execute(label, command, log, **_):
                    commands.append(label)
                    if label.endswith(":train"):
                        self.assertNotIn("seed1000_retest", " ".join(command))
                        input_at = command.index("--input") + 1
                        self.assertEqual(command[input_at + 1], "--metric")
                        train_gate.main(command[4:])
                    elif label.endswith(":baseline") or label.endswith(":evaluate"):
                        if label.endswith("semantic_linear:evaluate") and failures and failures.pop():
                            raise RuntimeError("simulated evaluation interruption")
                        runner.main(command[4:])
                        out = Path(command[command.index("--output-dir") + 1])
                        if label.endswith(":baseline"):
                            cache(out / "llm_cache.sqlite")
                    else:
                        self.fail(f"unexpected collector or other method: {label}")

                stack.enter_context(mock.patch.object(seed0.base, "_run_command", side_effect=execute))
                argv = ["--source-root", str(root / "source"), "--output-dir", str(root / "new"), "--datasets", dataset]
                with self.assertRaisesRegex(RuntimeError, "simulated evaluation"):
                    seed0.main(argv)
                failure = json.loads((root / "new" / "seed0_progress.json").read_text())
                self.assertEqual(failure["error"]["stage"], dataset + ":semantic_linear:evaluate")
                commands.clear()
                report_path = seed0.main(argv)
                self.assertEqual(commands, [dataset + ":semantic_linear:evaluate", dataset + ":semantic_mlp:train", dataset + ":semantic_mlp:evaluate"])
                result = json.loads(report_path.read_text())
                self.assertEqual(len(result["results"][dataset]), 4)
                self.assertTrue(all(r["paired_control_valid"] for r in result["details"][dataset].values()))
                commands.clear()
                seed0.main(argv)
                self.assertFalse(commands)
                for path, data in originals.items():
                    self.assertEqual(Path(path).read_bytes(), data, path)
                split = []
                for arm in seed0.matrix.DEFAULT_ARMS:
                    payload = seed0.base._load_json(root / "new" / dataset / "checkpoints" / (arm + ".json"))
                    meta = payload["training_metadata"]
                    self.assertEqual(meta["training_sources"], [str(seed0.source_path(args))])
                    self.assertFalse(set(meta["fit_task_ids"]) & set(meta["policy_calibration"]["calibration_task_ids"]))
                    split.append(meta["fit_task_ids"])
                self.assertEqual(split[0], split[1])
                self.assertEqual(split[1], split[2])
                # A changed source cannot silently reuse trained checkpoints.
                source = seed0.source_path(args)
                source.write_text(source.read_text() + "\n")
                with self.assertRaisesRegex(ValueError, "plan changed"):
                    seed0.main(argv)


if __name__ == "__main__":
    unittest.main()
