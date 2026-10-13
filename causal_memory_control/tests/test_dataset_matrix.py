from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

from causal_diagnostic.fever_oracle_recipient.data import file_md5
from causal_diagnostic.fever_oracle_recipient.provenance import SNAPSHOT_SCHEMA, stable_hash
from causal_memory_control import fever_experiment, run_gate_matrix, train_gate
from causal_memory_control import run_hotpotqa_gate_matrix as matrix
from causal_memory_control.compare_gate_runs import main as compare_main
from causal_memory_control.dataset_adapters import (
    DatasetAdapter, FeverDatasetAdapter, HotpotQADatasetAdapter, load_dataset_adapter,
)
from causal_memory_control.jev_gate import main as jev_main
from causal_memory_control.jsonl import read_jsonl
from causal_memory_control.tests import test_fever_experiment as fever_test
from causal_memory_control.tests.test_semantic_jev import TinyEncoder


def fever_source(root, count=14):
    path = root / "fever.jsonl"
    rows = [{"id": i, "claim": f"A claim about site {i}.",
             "label": "SUPPORTS" if i % 2 else "REFUTES"} for i in range(1, count + 1)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path, rows


class DatasetSelectionTests(unittest.TestCase):
    def test_switch_selects_dataset_defaults_and_training_metric(self):
        hotpot, fever = HotpotQADatasetAdapter(), FeverDatasetAdapter()
        _, hotpot_args = matrix.parse_args([], adapter=hotpot)
        options, fever_args = matrix.parse_args([], adapter=fever)
        self.assertNotEqual(hotpot_args.data, fever_args.data)
        self.assertNotEqual(hotpot_args.memory_dir, fever_args.memory_dir)
        self.assertNotEqual(hotpot_args.diagnostic_results, fever_args.diagnostic_results)
        self.assertNotEqual(hotpot_args.output_dir, fever_args.output_dir)
        for adapter, args, metric in ((hotpot, hotpot_args, "team_score"), (fever, fever_args, "success")):
            child = matrix._arm_args(args, "semantic_mlp")
            command = adapter.train_command(child)
            self.assertEqual(command[command.index("--metric") + 1], metric)
            command = matrix._jev_train_command(options, child, "jev_direct", adapter=adapter)
            self.assertEqual(command[command.index("--metric") + 1], metric)

    def test_fever_all_data_keeps_split_disjoint_and_balanced(self):
        with tempfile.TemporaryDirectory() as directory:
            data, _ = fever_source(Path(directory), count=22)
            adapter = FeverDatasetAdapter()
            _, args = matrix.parse_args(["--data", str(data), "--support-count", "2", "--use-all-data"], adapter=adapter)
            adapter.resolve_plan(args)
            adapter.validate_args(args)
            self.assertEqual((args.support_count, args.training_claims, args.final_claims), (2, 16, 4))
            self.assertEqual(args.support_count + args.training_claims + args.final_claims, 22)
            self.assertEqual(args.evaluation_per_label, 10)

    def test_fever_rejects_unbalanced_all_data_and_odd_support_count(self):
        with tempfile.TemporaryDirectory() as directory:
            data, _ = fever_source(Path(directory), count=21)
            adapter = FeverDatasetAdapter()
            for extra, message in ((["--use-all-data", "--support-count", "2"], "equal"),
                                   (["--support-count", "3"], "even")):
                _, args = matrix.parse_args(["--data", str(data), *extra], adapter=adapter)
                with self.assertRaisesRegex(SystemExit, message):
                    adapter.resolve_plan(args)

    def test_plan_only_does_not_write_experiment_outputs(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as output:
            root = Path(directory)
            data, _ = fever_source(root)
            result = run_gate_matrix.main(["--dataset", "fever", "--data", str(data),
                "--support-count", "2", "--evaluation-count", "12", "--smoke-claims", "1",
                "--matrix-dir", str(root / "matrix"), "--plan-only"])
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["dataset"], "fever")
            self.assertEqual(plan["experiment_plan"]["final_test_questions"], 3)
            self.assertEqual(plan["arms"], list(matrix.DEFAULT_ARMS))
            self.assertEqual(plan["max_jev_judgments"], 0)
            self.assertEqual(plan["final_team_inferences"], 12)
            self.assertFalse(result.parent.exists())

    def test_external_adapter_factory_and_identity_are_checked(self):
        class ExternalHotpot(HotpotQADatasetAdapter):
            name = "example"
            matrix_schema = "example-matrix-v1"

        module = ModuleType("test_dataset_adapter")
        module.create = ExternalHotpot
        with mock.patch.dict(sys.modules, {module.__name__: module}):
            adapter = load_dataset_adapter("example", "test_dataset_adapter:create")
            self.assertIsInstance(adapter, DatasetAdapter)
            with self.assertRaisesRegex(SystemExit, "differs"):
                load_dataset_adapter("fever", "test_dataset_adapter:create")
            module.create = lambda: object()
            with self.assertRaisesRegex(SystemExit, "DatasetAdapter"):
                load_dataset_adapter("example", "test_dataset_adapter:create")
        with self.assertRaisesRegex(SystemExit, "unknown dataset"):
            load_dataset_adapter("missing")

    def test_dataset_switch_cannot_resume_another_datasets_matrix(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            data, _ = fever_source(root)
            output = root / "matrix"
            output.mkdir()
            (output / "matrix_manifest.json").write_text(json.dumps({"configuration_hash": "hotpot-run"}))
            with self.assertRaisesRegex(SystemExit, "arguments changed"):
                run_gate_matrix.main(["--dataset", "fever", "--data", str(data),
                    "--support-count", "2", "--evaluation-count", "12", "--smoke-claims", "1",
                    "--matrix-dir", str(output)])


class _MatrixFeverMemory(fever_test._FakeMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_size = 2


class _MatrixFeverMacNet(fever_test._FakeMacNet):
    def schedule(self, task):
        type(self).schedule_count += 1
        self.env.reset()
        self.memory.gate.filter_retrieval(([fever_test._FakeMessage()], [], ["Read evidence before choosing a label."]),
            query_task=task["task_main"], task_state=task["task_description"])
        self.env.step(f"Finish[{task['label']}]")
        reward, done, _ = self.env.feedback()
        return reward, done


class FeverMatrixIntegrationTests(unittest.TestCase):
    def test_five_arms_train_evaluate_report_and_resume_with_binary_metrics(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            data, examples = fever_source(root)
            memory, diagnostic = root / "g-memory", root / "diagnostic"
            argv = ["--dataset", "fever", "--data", str(data), "--memory-dir", str(memory),
                "--diagnostic-results", str(diagnostic), "--matrix-dir", str(root / "matrix"),
                "--support-count", "2", "--evaluation-count", "12", "--smoke-claims", "1",
                "--model", "fake-model", "--embedding-model", "fake-embedding", "--node-num", "2",
                "--successful-topk", "1", "--insights-topk", "1", "--endpoint", "http://unused.invalid/v1",
                "--semantic-model", "test-encoder", "--semantic-revision", "fixed-test",
                "--mlp-epochs", "2", "--mlp-hidden-sizes", "8,4", "--bootstrap-samples", "100",
                "--arms", ",".join(matrix.ARMS)]
            adapter = FeverDatasetAdapter()
            _, args = matrix.parse_args(argv[2:], adapter=adapter)
            adapter.resolve_plan(args)

            def collect():
                memory.mkdir()
                design = {"snapshot_schema": SNAPSHOT_SCHEMA, "benchmark": adapter.benchmark,
                    "source_md5": file_md5(data), "support_ids": [1, 2], "evaluation_ids": list(range(3, 15)),
                    "support_per_label": 1, "evaluation_per_label": 6, "split_seed": 42, "build_seed": 42,
                    **{key: getattr(args, key) for key in ("model", "graph_type", "node_num", "embedding_model",
                        "successful_topk", "failed_topk", "insights_topk", "threshold")}}
                snapshot = {**design, "design": design, "design_hash": stable_hash(design),
                            "memory_records": 2, "successful_total": 2, "failed_total": 0}
                (memory / "causal_snapshot_manifest.json").write_text(json.dumps(snapshot))
                context = {**adapter.checkpoint_context(args, snapshot), "memory_dir": str(args.memory_dir),
                    "candidate_scope": "all_retrieved", "candidate_specs": [{"kind": "trajectory", "index": 0}, {"kind": "insight", "index": 0}],
                    "collection_scope": "gate_training", "repeats": 2, "temperature": args.temperature,
                    "selection_seed": 42, "recipients": ["solver_0", "solver_1"],
                    "evaluation_ids": snapshot["evaluation_ids"][:args.training_claims]}
                message = fever_test._FakeMessage()
                candidates = [{"candidate_id": "trajectory-rank1", "kind": "trajectory", "index": 0,
                    "task_main": message.task_main, "task_description": message.task_description,
                    "trajectory": message.task_trajectory, "label": True},
                    {"candidate_id": "insight-rank1", "kind": "insight", "index": 0,
                     "text": "Read evidence before choosing a label."}]
                for seed, suffix in ((0, ""), (1000, "_retest")):
                    path = diagnostic / f"seed{seed}{suffix}"
                    path.mkdir(parents=True)
                    (path / "run_manifest.json").write_text(json.dumps({"design": context,
                        "run": {"endpoint": args.endpoint, "sample_seed_base": seed}}))
                    rows = []
                    for raw in examples[2:2 + args.training_claims]:
                        for candidate in candidates:
                            for repeat in range(2):
                                for condition in ("use_all", "global_drop"):
                                    outcome = bool(repeat) if condition == "use_all" else candidate["kind"] == "insight"
                                    rows.append({"task_id": raw["id"], "task": fever_experiment.prepare_example(raw),
                                        "candidate": candidate, "candidate_set": candidates, "condition": condition,
                                        "run_hash": str(seed), "sample_seed": seed + repeat, "repeat_index": repeat,
                                        "exposure_count": 3, "retrieval_sizes": {"successful_trajectories": 1, "insights": 1},
                                        "outcome": {"success": outcome}})
                    (path / "branches.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                (diagnostic / "experiment_progress.json").write_text(json.dumps({"status": "completed"}))

            commands = []

            def execute(stage, command, log_path, **_):
                commands.append(stage)
                if stage == "diagnostic":
                    collect()
                elif command[3] == "causal_memory_control.train_gate":
                    train_gate.main(command[4:])
                elif command[3] == "causal_memory_control.jev_gate":
                    jev_main(command[4:])
                elif command[3] == "causal_memory_control.fever_experiment":
                    log = fever_experiment.main(command[4:])
                    if not (log.parent / "llm_cache.sqlite").exists():
                        (log.parent / "llm_cache.sqlite").write_bytes(b"fake-cache")
                else:
                    compare_main(command[4:])

            for patch in fever_test.FeverRunnerTests._patches():
                stack.enter_context(patch)
            stack.enter_context(mock.patch.object(fever_experiment, "GMemory", _MatrixFeverMemory))
            stack.enter_context(mock.patch.object(fever_experiment, "MacNet", _MatrixFeverMacNet))
            stack.enter_context(mock.patch.object(matrix.base, "_run_command", side_effect=execute))
            stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", return_value=TinyEncoder()))
            stack.enter_context(mock.patch("causal_memory_control.jev_api.judge", side_effect=lambda request: {
                "probabilities": {name: (0.1 if name == "retain_helpful" and json.loads(request["state"])["target_memory"]["kind"] == "insight"
                    else 0.9 if name == "retain_helpful" else 0.1 if name in ("conflict", "redundancy") else 0.9)
                    for name in request["questions"]}}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            result = run_gate_matrix.main(argv)
            payload = json.loads(result.read_text())
            self.assertEqual(payload["dataset"], "fever")
            self.assertEqual(len(commands), 17)
            self.assertEqual(len(payload["results"]), 6)
            self.assertIn("accuracy", payload["results"][0])
            self.assertNotIn("answer_f1", payload["results"][0])
            self.assertIn("Accuracy", (result.parent / "matrix_results.md").read_text())
            for arm in matrix.ARMS:
                comparison = json.loads((result.parent / arm / "comparison.json").read_text())
                self.assertEqual(comparison["paired_tasks"], args.final_claims)
                self.assertTrue(comparison["paired_control_valid"])
                self.assertTrue(comparison["policy_valid"])
                checkpoint = json.loads((result.parent / "checkpoints" / f"{arm}.json").read_text())
                self.assertEqual(checkpoint["training_metadata"]["metric"], "success")
            run_gate_matrix.main(argv)
            self.assertEqual(len(commands), 17)
            # Reuse the completed Jev arm to exercise real FEVER selective retry.
            output = result.parent / "jev_direct"
            rows = read_jsonl(output / "memory_gate.jsonl")
            rows[0]["outcome"]["jev_errors"] = 1
            (output / "memory_gate.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            summary = json.loads((output / "summary.json").read_text())
            summary["jev_errors"] = 1
            (output / "summary.json").write_text(json.dumps(summary))
            scheduled = _MatrixFeverMacNet.schedule_count
            run_gate_matrix.main([*argv, "--retry-jev-errors"])
            self.assertEqual(_MatrixFeverMacNet.schedule_count, scheduled + 1)
            self.assertEqual(commands[-2:], ["jev_direct:evaluate", "jev_direct:compare"])
            self.assertTrue(list(output.glob("memory_gate.jsonl.before-jev-retry-*")))
            self.assertTrue(all(not row["outcome"]["jev_errors"] for row in read_jsonl(output / "memory_gate.jsonl")))


if __name__ == "__main__":
    unittest.main()
