from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import numpy as np

from causal_memory_control import train_gate
from causal_memory_control import run_st_gate_benchmark as st
from causal_memory_control import fever_experiment, hotpotqa_experiment
from causal_memory_control.dataset_adapters import FeverDatasetAdapter, HotpotQADatasetAdapter
from causal_memory_control.estimator import PotentialOutcomeExample
from causal_memory_control.runtime_gate import GMemoryExposureGate
from causal_memory_control.st_estimator import STUtilityEstimator, contrastive_loss, rarity_weights
from causal_memory_control.tests.test_semantic_jev import TinyEncoder, branch_fixture
from causal_memory_control.semantic_features import SemanticTeamFeatureBuilder
from causal_memory_control.st_data import CompactFeatures, feature_matrix, load_st_examples
from causal_memory_control.tests.test_dataset_matrix import fever_source
from causal_memory_control.tests import test_fever_experiment as fever_tests
from causal_memory_control.tests import test_hotpotqa_gate as hotpot_tests
from causal_memory_control.tests.test_full_gate_benchmark import cache
from causal_diagnostic.fever_oracle_recipient.provenance import stable_hash
from causal_diagnostic.hotpotqa_oracle_recipient.data import load_hotpotqa, file_md5


class STLossTests(unittest.TestCase):
    def test_equation7_has_real_same_list_positives_negatives_and_no_self(self):
        representations = torch.tensor([[1., 0.], [1., 0.], [0., 1.]], requires_grad=True)
        target = torch.tensor([0., 0., 1.])
        loss, anchors, pairs = contrastive_loss(representations, target)
        self.assertEqual((anchors, pairs), (2, 2))
        self.assertAlmostEqual(loss.item(), math.log1p(math.exp(-1)), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(representations.grad).all())
        no_pairs, anchors, pairs = contrastive_loss(representations, torch.tensor([0., 0.5, 1.]))
        self.assertEqual((anchors, pairs, no_pairs.item()), (0, 0, 0))
        # No negatives: all-zero influence is not useful contrastive supervision.
        self.assertEqual(contrastive_loss(representations, torch.zeros(3))[1], 0)

    def test_list_rarity_inverse_frequency_normalization(self):
        rates, weights, counts = rarity_weights([[0., 0.]] * 9 + [[-1., 1.]])
        self.assertEqual(rates[-1], 10)
        self.assertGreater(weights[-1], weights[0])
        self.assertAlmostEqual(sum(weights) / len(weights), 1)
        self.assertEqual(sum(counts.values()), 10)

    def test_direct_u_fit_roundtrip_and_rng_preservation(self):
        examples = [PotentialOutcomeExample({"x": x}, 0.9 if x > 0 else 0.1,
            0.1 if x > 0 else 0.9, event_id=f"{group}-{i}", group_id=str(group))
            for group in range(6) for i, x in enumerate((-1., -1., 1.))]
        estimator = STUtilityEstimator(ensemble_size=2, hidden_sizes=(8, 4),
            epochs=60, learning_rate=0.02, batch_tasks=6)
        rng = torch.random.get_rng_state().clone()
        self.assertTrue(estimator.fit(examples))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertGreater(estimator.predict({"x": 1}).utility, 0.5)
        self.assertLess(estimator.predict({"x": -1}).utility, -0.5)
        self.assertGreater(estimator.contrastive_pairs, 0)
        self.assertEqual(estimator.training_diagnostics["fit_task_groups"], 6)
        self.assertTrue(all(math.isfinite(row["loss"]) for row in estimator.loss_history))
        restored = STUtilityEstimator.from_dict(json.loads(json.dumps(estimator.to_dict())))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertAlmostEqual(restored.predict({"x": 1}).utility, estimator.predict({"x": 1}).utility, places=6)
        self.assertEqual(restored._models[0][-1].out_features, 1)
        prediction = restored.predict({"x": 1})
        self.assertAlmostEqual(prediction.q_use - prediction.q_drop, prediction.utility)

    def test_hard_selection_no_cross_task_pairs_and_invalid_settings(self):
        examples = [PotentialOutcomeExample({"x": float(g)}, 0.5, 0.5,
            event_id=f"{g}-{i}", group_id=str(g)) for g in range(5) for i in range(3)]
        estimator = STUtilityEstimator(ensemble_size=2, hidden_sizes=(4,), epochs=1)
        estimator.fit(examples)
        self.assertEqual(estimator.training_diagnostics["hard_task_groups"], 0)
        self.assertEqual(estimator.contrastive_pairs, 0)
        self.assertTrue(all(row["infonce"] == 0 for row in estimator.loss_history))
        for kwargs in ({"temperature": 0}, {"negative_epsilon": 0.01}, {"beta": float("nan")}, {"epochs": 0}):
            with self.assertRaises(ValueError):
                STUtilityEstimator(**kwargs)
        broken = estimator.to_dict()
        broken["scales"][0] = 0
        with self.assertRaisesRegex(ValueError, "standardizer"):
            STUtilityEstimator.from_dict(broken)
        with self.assertRaisesRegex(ValueError, "group_id"):
            estimator.fit([PotentialOutcomeExample({"x": 1}, 1, 0)] * 8)

    def test_cli_fit_dev_partition_and_real_runtime_dispatch(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            source, memory = branch_fixture(root)
            checkpoint = root / "st.json"
            stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", return_value=TinyEncoder()))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            train_gate.main(["--input", str(source), "--output", str(checkpoint), "--memory-dir", str(memory),
                "--metric", "team_score", "--estimator", "st", "--feature-version", "3", "--ensemble-size", "2",
                "--st-hidden-sizes", "8,4", "--st-epochs", "2", "--calibration-fraction", "0.2",
                "--semantic-model", "test-encoder", "--semantic-revision", "fixed-test", "--residual-noise-scale", "0"])
            payload = json.loads(checkpoint.read_text())
            metadata = payload["training_metadata"]
            self.assertEqual(metadata["estimator"]["type"], "st_single_direct_u")
            self.assertEqual(payload["estimator"]["training_diagnostics"]["fit_task_groups"], 8)
            self.assertFalse(set(metadata["fit_task_ids"]) & set(metadata["policy_calibration"]["calibration_task_ids"]))
            gate = GMemoryExposureGate.from_checkpoint(checkpoint, mode="learned", memory_dir=memory)
            gate.begin_task(30, {"task_main": "Question?", "task_description": "Supplied context"})
            gate.filter_retrieval(([], [], ["harmful memory", "helpful memory"]), query_task="Question?", task_state="Supplied context")
            self.assertEqual(len(gate._task_decisions), 2)
            self.assertTrue(all(d.prediction.source == "bootstrap_semantic_st_direct_u" for d in gate._task_decisions))
            gate.close()


def options(root, fever):
    return ["--datasets", "fever", "--fever-data", str(fever),
        "--source-root", str(root / "source"), "--output-dir", str(root / "st"),
        "--support-count", "2", "--smoke-claims", "1", "--ensemble-size", "2",
        "--st-epochs", "2", "--st-hidden-sizes", "8,4", "--bootstrap-samples", "100",
        "--semantic-model", "test-encoder", "--semantic-revision", "fixed-test",
        "--node-num", "2", "--model", "fake-model", "--embedding-model", "fake-embedding",
        "--successful-topk", "1", "--insights-topk", "1"]


class STRunnerTests(unittest.TestCase):
    def test_full_plan_only_no_writes_and_missing_sources_fail_without_old_methods(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            fever, _ = fever_source(root, count=22)
            argv = options(root, fever)
            plan = st.main([*argv, "--plan-only"])
            self.assertEqual(plan["fever"]["gate_training_questions"] + plan["fever"]["final_test_questions"], 20)
            self.assertEqual(plan["fever"]["existing_gates_run"], [])
            self.assertFalse((root / "source").exists())
            self.assertFalse((root / "st").exists())
            with mock.patch.object(st.base, "_run_command") as run:
                with self.assertRaisesRegex(FileNotFoundError, "diagnostics"):
                    st.main(argv)
                run.assert_not_called()
            failure = json.loads((root / "st" / "st_progress.json").read_text())
            self.assertEqual(failure["error"]["stage"], "fever:diagnostic")

    def test_recorded_endpoint_and_all_data_split_are_imported_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fever, _ = fever_source(root, count=22)
            suite = st.parse_args(options(root, fever))
            adapter, args = st.prepare_dataset(suite, "fever")
            config = st.full._jsonable(adapter.configuration(args))
            # Import a source-full-run-style configuration, before ST redirects its checkpoint/cache.
            config["endpoint"] = "http://127.0.0.1:11434/v1"
            st.full._save(root / "source" / "benchmark_manifest.json", {
                "schema": st.full.SCHEMA, "experiments": {"fever": config},
                "semantic": {"fever": {"semantic_model": "test-encoder", "semantic_revision": "fixed-test"}}})
            suite.shared_argv = []
            _, imported = st.prepare_dataset(suite, "fever")
            self.assertEqual(imported.endpoint, "http://127.0.0.1:11434/v1")
            self.assertEqual(imported.final_claims, 4)
            suite.shared_argv = ["--endpoint", "http://different"]
            with self.assertRaisesRegex(ValueError, "omit shared overrides"):
                st.prepare_dataset(suite, "fever")

    def test_only_st_is_run_and_resume_rejects_input_or_hyperparameter_changes(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            data, examples = fever_source(root, count=22)
            argv = options(root, data)
            suite = st.parse_args(argv)
            adapter, args = st.prepare_dataset(suite, "fever")
            create_diagnostics(adapter, args, examples[2:])
            for patch in fever_tests.FeverRunnerTests._patches():
                stack.enter_context(patch)
            stack.enter_context(mock.patch.object(fever_experiment, "GMemory", _STMemory))
            stack.enter_context(mock.patch.object(fever_experiment, "MacNet", _FeverSTMacNet))
            stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", return_value=TinyEncoder()))
            stack.enter_context(mock.patch.object(FeverDatasetAdapter, "diagnostic_complete", return_value=True))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            native = args.output_dir / "native_gmemory"
            baseline_command = adapter.evaluation_command(args, mode="always_keep", output=native, resume=False)
            fever_experiment.main(baseline_command[4:])
            cache(native / "llm_cache.sqlite")
            original_bytes = (native / "memory_gate.jsonl").read_bytes()
            commands = []

            def execute(label, command, path, **_):
                commands.append(label)
                if command[3] == "causal_memory_control.train_gate":
                    self.assertIn("st", command)
                    train_gate.main(command[4:])
                elif command[3] == "causal_memory_control.fever_experiment":
                    fever_experiment.main(command[4:])
                else:
                    self.fail(f"unexpected old-method invocation: {command}")

            stack.enter_context(mock.patch.object(st.base, "_run_command", side_effect=execute))
            result = st.main(argv)
            self.assertEqual(commands, ["fever:train_st", "fever:evaluate_st"])
            self.assertEqual((native / "memory_gate.jsonl").read_bytes(), original_bytes)
            report = json.loads(result.read_text())
            self.assertTrue(report["details"]["fever"]["semantic_st"]["result"]["paired_control_valid"])
            self.assertEqual(len(report["rows"]), 8)
            commands.clear()
            st.main(argv)
            self.assertFalse(commands)
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                st.main([*argv, "--st-beta", "0.5"])
            with (native / "llm_cache.sqlite").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "plan changed"):
                st.main(argv)

    def test_hotpot_only_prepares_missing_baseline_and_diagnostics_then_resumes_st(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            data = root / "hotpot.json"
            data.write_text(json.dumps([{"_id": f"h{i}", "question": f"Where is site {i}?", "answer": "New York City",
                "context": [["Place", ["The site is in New York City."]]],
                "supporting_facts": [["Place", 0]]} for i in range(22)]))
            argv = options(root, data)
            argv[argv.index("--datasets") + 1] = "hotpotqa"
            argv += ["--hotpotqa-data", str(data), "--prepare-missing"]
            suite = st.parse_args(argv)
            adapter, args = st.prepare_dataset(suite, "hotpotqa")
            examples = load_hotpotqa(data)
            hotpot_tests.HotpotRunnerTests.patches(stack)
            stack.enter_context(mock.patch.object(hotpotqa_experiment, "GMemory", _STMemory))
            stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", return_value=TinyEncoder()))
            stack.enter_context(mock.patch.object(HotpotQADatasetAdapter, "diagnostic_complete",
                side_effect=lambda _args: (args.memory_dir / "causal_snapshot_manifest.json").exists()))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            commands, failures = [], [True]

            def execute(label, command, path, **_):
                commands.append(label)
                if label.endswith(":diagnostic"):
                    create_diagnostics(adapter, args, examples[2:])
                elif label.endswith(":train_st"):
                    train_gate.main(command[4:])
                elif label.endswith(":baseline"):
                    hotpotqa_experiment.main(command[4:])
                    cache(args.output_dir / "native_gmemory" / "llm_cache.sqlite")
                elif label.endswith(":evaluate_st"):
                    if failures and failures.pop():
                        raise RuntimeError("simulated interruption")
                    hotpotqa_experiment.main(command[4:])
                else:
                    self.fail(f"unexpected old-method stage: {label}")

            stack.enter_context(mock.patch.object(st.base, "_run_command", side_effect=execute))
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                st.main(argv)
            commands.clear()
            report_path = st.main(argv)
            self.assertEqual(commands, ["hotpotqa:evaluate_st"])
            report = json.loads(report_path.read_text())
            result = report["details"]["hotpotqa"]["semantic_st"]["result"]
            self.assertTrue(result["paired_control_valid"])
            self.assertIn("gate_answer_em", result)
            self.assertEqual(result["paired_tasks"], 4)


class STStreamingTests(unittest.TestCase):
    def test_compact_features_align_missing_and_unknown_columns(self):
        row = CompactFeatures({"x": 1, "extra": 99}, {})
        np.testing.assert_array_equal(feature_matrix([row, {"y": 2}], ["x", "y"]), [[1, 0], [0, 2]])

    def test_streamed_labels_features_match_legacy_loader_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            source, memory = branch_fixture(root)
            kwargs = dict(metric="team_score", task_ids=None, hash_dimensions=256, feature_version=3,
                memory_dir=memory, feature_builder=SemanticTeamFeatureBuilder(TinyEncoder()), event_sink=[])
            compact = load_st_examples([source], **kwargs)
            normal = train_gate.load_examples([source], **{**kwargs, "event_sink": []})
            self.assertEqual([(e.group_id, e.q_use, e.q_drop) for e in compact], [(e.group_id, e.q_use, e.q_drop) for e in normal])
            for a, b in zip(compact, normal):
                names = sorted(set(a.features) | set(b.features))
                np.testing.assert_allclose(feature_matrix([a.features], names), feature_matrix([b.features], names))
            with self.assertRaisesRegex(ValueError, "duplicate counterfactual"):
                load_st_examples([source, source], **kwargs)
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            manifest_path = source.parent / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["design"]["repeats"] = 2
            manifest_path.write_text(json.dumps(manifest))
            # Matching but incomplete USE/DROP cells must not look like a complete run.
            with self.assertRaisesRegex(ValueError, "missing declared"):
                load_st_examples([source], **kwargs)
            manifest["design"].pop("repeats")
            manifest_path.write_text(json.dumps(manifest))
            # Remove BOTH branches of one candidate: do not silently train an incomplete list.
            source.write_text("".join(json.dumps(r) + "\n" for r in rows if not (r["task_id"] == 0 and r["candidate"]["candidate_id"] == "helpful")))
            with self.assertRaisesRegex(ValueError, "every retrieved candidate"):
                load_st_examples([source], **kwargs)


class _FeverSTMacNet(fever_tests._FakeMacNet):
    def schedule(self, task):
        self.env.reset()
        self.memory.gate.filter_retrieval(([fever_tests._FakeMessage()], [], ["Read evidence."]),
            query_task=task["task_main"], task_state=task["task_description"])
        self.env.step(f"Finish[{task['label']}]")
        reward, done, _ = self.env.feedback()
        return reward, done


class _STMemory(fever_tests._FakeMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_size = 2


def create_diagnostics(adapter, args, evaluation):
    """Synthetic task-disjoint measured branches, never real LLM calls."""
    from causal_diagnostic.fever_oracle_recipient.provenance import SNAPSHOT_SCHEMA as FEVER_SCHEMA
    from causal_diagnostic.hotpotqa_oracle_recipient.provenance import SNAPSHOT_SCHEMA as HOTPOT_SCHEMA

    design = {"snapshot_schema": FEVER_SCHEMA if adapter.name == "fever" else HOTPOT_SCHEMA,
        "benchmark": adapter.benchmark, "source_md5": file_md5(args.data),
        "support_ids": [1, 2] if adapter.name == "fever" else [0, 1],
        "evaluation_ids": [e["id"] for e in evaluation],
        **{k: getattr(args, k) for k in ("model", "graph_type", "node_num", "embedding_model",
            "successful_topk", "failed_topk", "insights_topk", "threshold")}}
    snapshot = {**design, "design": design, "design_hash": stable_hash(design),
                "memory_records": 2, "successful_total": 2, "failed_total": 0}
    st.full._save(args.memory_dir / "causal_snapshot_manifest.json", snapshot)
    context = adapter.checkpoint_context(args, snapshot)
    message = fever_tests._FakeMessage()
    candidates = [{"candidate_id": "trajectory-rank1", "kind": "trajectory", "index": 0,
        "task_main": message.task_main, "task_description": message.task_description,
        "trajectory": message.task_trajectory, "label": True},
        {"candidate_id": "insight-rank1", "kind": "insight", "index": 0, "text": "Read evidence."}]
    for seed, suffix in ((0, ""), (1000, "_retest")):
        path = args.diagnostic_results / f"seed{seed}{suffix}"
        st.full._save(path / "run_manifest.json", {"design": context})
        rows = []
        for raw in evaluation[:args.training_claims]:
            task = (fever_experiment.prepare_example(raw) if adapter.name == "fever"
                    else hotpotqa_experiment.prepare_example(raw))
            for candidate in candidates:
                for repeat in range(2):
                    for condition in ("use_all", "global_drop"):
                        outcome = float(repeat) if condition == "use_all" else float(candidate["kind"] == "insight")
                        rows.append({"task_id": raw["id"], "task": task, "candidate": candidate,
                            "candidate_set": candidates, "condition": condition, "run_hash": str(seed),
                            "sample_seed": seed + repeat, "repeat_index": repeat, "exposure_count": 3,
                            "retrieval_sizes": {"successful_trajectories": 1, "insights": 1},
                            "outcome": {"success": outcome, "team_score": outcome}})
        (path / "branches.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
