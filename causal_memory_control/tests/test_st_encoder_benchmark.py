"""Frozen encoder ablation: reuse measured labels, isolate outputs, audit pairs."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from causal_memory_control import fever_experiment, hotpotqa_experiment, train_gate
from causal_memory_control import run_st_encoder_benchmark as cli
from causal_memory_control import st_encoder_evaluate as wrapper
from causal_memory_control.estimator import PotentialOutcomeExample
from causal_memory_control.interactive_data import digest, save, sha
from causal_memory_control.interactive_runtime import ActionStore, event_from_row
from causal_memory_control.resume_interactive_dataset import experiment_lock
from causal_memory_control.runtime_gate import GMemoryExposureGate, save_gate_checkpoint
from causal_memory_control.semantic_features import SemanticTeamFeatureBuilder
from causal_memory_control.st_estimator import STUtilityEstimator
from causal_memory_control.tests import test_fever_experiment as fever_tests
from causal_memory_control.tests import test_hotpotqa_gate as hotpot_tests
from causal_memory_control.tests.test_full_gate_benchmark import cache
from causal_memory_control.tests.test_seed0_gate_matrix import prepare_source
from causal_memory_control.tests.test_semantic_jev import TinyEncoder, branch_fixture
from causal_memory_control.tests.test_st_gate import _FeverSTMacNet, _STMemory
from causal_memory_control.tests.test_st_policy_reevaluation import (
    Estimator, Features, episode, pddl_fixture, source_bytes,
)


class NewEncoder:
    """Different dimension and coordinates; no network, GPU or model weights."""
    def __init__(self, model, *, revision=None, chunk_tokens=192, **kwargs):
        self.model_name, self.revision, self.chunk_tokens = model, revision, chunk_tokens

    def embed(self, text):
        a, b, c = TinyEncoder().embed(text)
        return [c, a, b, .2]

    def config(self):
        return {"type": "sentence-transformer-chunk-mean-v1", "model": self.model_name,
            "revision": self.revision, "chunk_tokens": self.chunk_tokens, "dimensions": 4}

    def close(self):
        pass


def encoders(model, **kwargs):
    return TinyEncoder() if model == "test-encoder" else NewEncoder(model, **kwargs)


def arguments(root, datasets="fever"):
    model = root / "new-model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text('{"model_type":"test"}')
    return ["--datasets", datasets, "--output-dir", str(root / "new"), "--semantic-model", str(model),
        "--qa-root", str(root / "old"), "--interactive-root", str(root / "interactive"),
        "--rule-reference-root", str(root / "rule-reference")]


def old_qa_head(root):
    source, memory = branch_fixture(root)
    checkpoint = root / "old.json"
    train_gate.main(["--input", str(source), "--output", str(checkpoint), "--memory-dir", str(memory),
        "--estimator", "st", "--metric", "team_score", "--feature-version", "3", "--ensemble-size", "2",
        "--st-hidden-sizes", "8,4", "--st-epochs", "2", "--calibration-fraction", ".2",
        "--semantic-model", "test-encoder", "--semantic-revision", "fixed-test", "--residual-noise-scale", "0"])
    return {"name": "fever", "label": "fever", "checkpoint": checkpoint, "sources": [source],
        "memory": memory, "payload": cli.reuse.read(checkpoint)}


class EncoderFitTests(unittest.TestCase):
    def test_actual_head_refit_preserves_split_labels_loss_and_policy_and_pins_model(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", side_effect=encoders))
            args = cli.parse_args(arguments(root))
            job = old_qa_head(root)
            old = deepcopy(job["payload"])
            before = {p: p.read_bytes() for p in (job["checkpoint"], *job["sources"])}
            checkpoint = cli.fit_head(args, job)
            fitted = cli.reuse.read(checkpoint)
            self.assertEqual(fitted["feature_builder"]["encoder"]["dimensions"], 4)
            self.assertEqual(old["feature_builder"]["encoder"]["dimensions"], 3)
            self.assertNotEqual(fitted["estimator"]["feature_names"], old["estimator"]["feature_names"])
            self.assertEqual(fitted["estimator"]["config"], old["estimator"]["config"])
            self.assertEqual(cli.partitions(fitted), cli.partitions(old))
            metadata = fitted["training_metadata"]
            for key in ("recommended_kappa", "recommended_max_drops", "delta", "calibration_task_ids"):
                self.assertEqual(metadata["policy_calibration"][key], old["training_metadata"]["policy_calibration"][key])
            self.assertFalse(metadata["policy_calibration"]["enabled"])
            self.assertNotIn("selected", metadata["policy_calibration"])
            self.assertEqual(metadata["estimator"]["training_diagnostics"], fitted["estimator"]["training_diagnostics"])
            self.assertEqual(metadata["calibration_events"], 0)
            self.assertTrue(fitted["estimator"]["loss_history"])
            with mock.patch.object(STUtilityEstimator, "fit", side_effect=AssertionError("unwanted refit")):
                self.assertEqual(cli.fit_head(args, job), checkpoint)
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            (Path(args.semantic_model) / "config.json").write_text('{"model_type":"changed"}')
            with self.assertRaisesRegex(ValueError, "registered input/configuration changed"):
                cli.require_head(args, job)

    def test_old_head_cannot_be_silently_loaded_as_new_encoder_and_partition_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", side_effect=encoders))
            args = cli.parse_args(arguments(root))
            job = old_qa_head(root)
            save(args.output_dir / "fever/checkpoint.json", job["payload"])
            with self.assertRaisesRegex(ValueError, "new ST checkpoint changed"):
                cli.require_head(args, job)
            bad = deepcopy(job["payload"])
            bad["training_metadata"]["policy_calibration"]["calibration_task_ids"].append(
                bad["training_metadata"]["fit_task_ids"][0])
            with self.assertRaisesRegex(ValueError, "disjoint"):
                cli.partitions(bad)

    def test_execution_opt_in_and_no_alfworld_mode(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            argv = arguments(Path(directory))
            self.assertEqual(cli.parse_args(argv).stage, "plan")
            with self.assertRaises(SystemExit):
                cli.parse_args([*argv, "--stage", "all"])
            with self.assertRaises(SystemExit):
                cli.parse_args([*argv, "--datasets", "alfworld"])
            self.assertEqual(cli.parse_args([*argv, "--stage", "all", "--refit-head"]).stage, "all")

    def test_pddl_lock_conflict_does_not_interrupt_running_experiment(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            argv = arguments(root, "pddl")
            with experiment_lock(root / "interactive"), mock.patch.object(cli, "prepare") as prepare:
                with self.assertRaisesRegex(ValueError, "still running/locked"):
                    cli.main([*argv, "--stage", "fit", "--refit-head"])
                prepare.assert_not_called()


class EncoderEvaluationTests(unittest.TestCase):
    def test_qa_actual_evaluator_and_resume_preserve_original_files_even_with_interactive_lock(self):
        for dataset in ("fever", "hotpotqa"):
            with self.subTest(dataset=dataset):
                self.qa_case(dataset)

    def qa_case(self, dataset):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            adapter, old_args = prepare_source(root, dataset)
            runner = fever_experiment if dataset == "fever" else hotpotqa_experiment
            if dataset == "fever":
                for patch in fever_tests.FeverRunnerTests._patches():
                    stack.enter_context(patch)
            else:
                hotpot_tests.HotpotRunnerTests.patches(stack)
            stack.enter_context(mock.patch.object(runner, "GMemory", _STMemory))
            stack.enter_context(mock.patch.object(runner, "MacNet", _FeverSTMacNet if dataset == "fever" else hotpot_tests._HotpotMacNet))
            stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", side_effect=encoders))
            checkpoint = root / "old" / dataset / "checkpoint.json"
            sources = sorted(old_args.diagnostic_results.glob("*/branches.jsonl"))
            train_gate.main(["--input", *map(str, sources), "--output", str(checkpoint),
                "--memory-dir", str(old_args.memory_dir), "--estimator", "st", "--metric", adapter.training_metric,
                "--feature-version", "3", "--ensemble-size", "2", "--st-hidden-sizes", "8,4", "--st-epochs", "2",
                "--calibration-fraction", ".2", "--semantic-model", "test-encoder", "--semantic-revision", "fixed-test"])
            old_args.checkpoint = checkpoint
            old_args.output_dir = root / "old" / dataset
            old_args.gate_evaluation_seed = 2000
            policy = cli.reuse.read(checkpoint)["training_metadata"]["policy_calibration"]
            old_args.kappa, old_args.delta, old_args.max_drops = policy["recommended_kappa"], policy["delta"], policy["recommended_max_drops"]
            baseline, old = old_args.output_dir / "native_gmemory", old_args.output_dir / "semantic_st"
            for mode, target in (("always_keep", baseline), ("learned", old)):
                runner.main(adapter.evaluation_command(old_args, mode=mode, output=target, resume=False)[4:])
                if not (target / "llm_cache.sqlite").exists():
                    cache(target / "llm_cache.sqlite")
            retest = root / "rule-reference" / dataset / "semantic_st"
            wrapper.main(["--dataset", dataset, "--reference-log", str(old / "memory_gate.jsonl"),
                "--st-deletion-rule", "positive_unlimited",
                *adapter.evaluation_command(old_args, mode="learned", output=retest, resume=False)[4:]])
            prepared = {dataset: {"adapter": adapter, "args": old_args, "old": old, "baseline": baseline,
                "source_memory": old_args.memory_dir, "cache": old / "llm_cache.sqlite"}}
            stack.enter_context(mock.patch.object(cli.reuse, "prepare", return_value=(prepared, {"source_sha256": {}})))
            argv = arguments(root, dataset)
            before = {**source_bytes(root / "old"), **source_bytes(root / "source"), **source_bytes(old_args.memory_dir)}
            plan = cli.main(argv)
            self.assertFalse((root / "new").exists())
            self.assertEqual(plan["new_diagnostic_tasks"], 0)
            self.assertEqual(plan["rules"][dataset], "calibrated" if dataset == "fever" else "positive_unlimited")
            commands = []

            def execute(command, *args, **kwargs):
                commands.append(command)
                self.assertEqual(command[3], "causal_memory_control.st_encoder_evaluate")
                return wrapper.main(command[4:])

            stack.enter_context(mock.patch("causal_memory_control.interactive_progress.run_logged", side_effect=execute))
            with experiment_lock(root / "interactive"):
                cli.main([*argv, "--stage", "all", "--refit-head"])
            self.assertEqual(len(commands), 1)
            result = cli.reuse.read(root / "new/summary.json")["datasets"][dataset]
            self.assertEqual(result["n"], old_args.final_claims)
            self.assertEqual(result["baseline"], result["new_encoder"])
            self.assertIsNotNone(result["old_encoder_same_rule"])
            paired = cli.reuse.read(root / "new" / dataset / "semantic_st/encoder_comparison.json")
            self.assertTrue(paired["paired_control_valid"])
            with mock.patch.object(STUtilityEstimator, "fit", side_effect=AssertionError("unwanted refit")):
                cli.main([*argv, "--stage", "all", "--refit-head"])
            self.assertEqual(len(commands), 1)
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            self.assertTrue((root / "new/results.md").is_file())
            with self.assertRaisesRegex(ValueError, "registered input/configuration changed"):
                cli.main([*argv, "--stage", "evaluate", "--st-deletion-rule",
                    "positive_unlimited" if dataset == "fever" else "calibrated"])

    def test_wrapper_accepts_changed_predictions_but_rejects_changed_candidate_before_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "old.jsonl"
            original = GMemoryExposureGate(mode="learned", estimator=Estimator(), feature_builder=Features())
            original.begin_task("a", {})
            original.filter_retrieval(([], [], ["negative-a"]), query_task="goal")
            row = {"task_id": "a", "retrieval_fingerprint": original._retrieval_fingerprint,
                "decisions": [d.to_dict() for d in original._task_decisions]}
            row["decisions"][0]["prediction"]["utility"] = .8  # Old head is allowed to disagree.
            reference.write_text(json.dumps(row) + "\n")
            runner = SimpleNamespace(GMemoryExposureGate=GMemoryExposureGate)
            called = []

            def execute(argv):
                gate = runner.GMemoryExposureGate(mode="learned", estimator=Estimator(), feature_builder=Features())
                gate.begin_task("a", {})
                result = gate.filter_retrieval(([], [], ["negative-a"]), query_task="goal")
                called.append(result)

            runner.main = execute
            argv = ["--dataset", "hotpotqa", "--reference-log", str(reference)]
            with mock.patch.object(wrapper.importlib, "import_module", return_value=runner):
                wrapper.main(argv)
                self.assertEqual(called, [([], [], [])])
                row["decisions"][0]["memory_content"] = "changed text"
                reference.write_text(json.dumps(row) + "\n")
                with self.assertRaisesRegex(ValueError, "retrieval changed"):
                    wrapper.main(argv)
            self.assertIs(runner.GMemoryExposureGate, GMemoryExposureGate)

    def test_paired_encoder_comparison_handles_new_keep_all_and_checks_equal_drop_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root / "old.jsonl", root / "new.jsonl"
            row = {"task_id": "a", "run_metadata": {"model": "fixed"}, "retrieval_fingerprint": "fixed",
                "reward": 0., "done": False, "decisions": [{"memory_id": "m", "action": "DROP"}],
                "dropped_count": 1, "outcome": {"prediction": "wrong"}}
            old.write_text(json.dumps(row) + "\n")
            row.update(reward=1., done=True, dropped_count=0, decisions=[{"memory_id": "m", "action": "KEEP"}],
                outcome={"prediction": "correct"})
            new.write_text(json.dumps(row) + "\n")
            self.assertTrue(cli.compare_encoders(old, new, seed=42)["paired_control_valid"])
            row.update(dropped_count=1, decisions=[{"memory_id": "m", "action": "DROP"}])
            new.write_text(json.dumps(row) + "\n")
            self.assertFalse(cli.compare_encoders(old, new, seed=42)["paired_control_valid"])

    def test_pddl_real_new_head_reuses_measured_actions_and_empty_fold_never_loads_encoder(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", side_effect=encoders))
            source_root = pddl_fixture(directory, missing_multi=True)
            source = source_root / "pddl"
            config = cli.reuse.read(source_root / "benchmark_manifest.json")
            config["repeats"] = 2
            candidates = [{"memory_id": "m0", "memory_type": "insight", "content": "harmful memory", "metadata": {"retrieval_index": 0}},
                {"memory_id": "m1", "memory_type": "insight", "content": "helpful memory", "metadata": {"retrieval_index": 1}}]
            paths, examples = [], []
            builder = SemanticTeamFeatureBuilder(TinyEncoder())
            ids = [f"train{i}" for i in range(6)]
            for task_id in ids:
                row = {**episode(task_id, candidates), "config_digest": digest(config), "repeats": 2,
                    "q_use": .5, "q_drop": [.9, .1]}
                path = source / "diagnostic/tasks" / (digest(task_id) + ".json")
                save(path, row)
                paths.append(path)
                for index in range(2):
                    event = event_from_row(row, index)
                    examples.append(PotentialOutcomeExample(builder.build(event), .5, row["q_drop"][index], event.event_id, task_id))
            estimator = STUtilityEstimator(ensemble_size=2, hidden_sizes=(8, 4), epochs=2)
            estimator.fit([e for e in examples if e.group_id in ids[:4]])
            checkpoint = source / "semantic_st/checkpoint.json"
            save_gate_checkpoint(checkpoint, estimator, feature_version=3, feature_config={"encoder": TinyEncoder().config()},
                training_metadata={"training_task_ids": ids, "fit_task_ids": ids[:4], "dev_task_ids": ids[4:],
                    "policy_calibration": {"recommended_kappa": 1., "recommended_max_drops": 1, "delta": 0.,
                        "calibration_task_ids": ids[4:]}, "candidate_kinds": ["insight"]})
            identity = cli.reuse.read(source / "evaluation/actions/identity.json")
            store = ActionStore(source / "evaluation/actions", identity)
            reference_path = root / "rule-reference/pddl/fold_00/actions"
            reference = ActionStore(reference_path, {"schema": cli.reuse.SCHEMA,
                "rule": "positive_unlimited", "original_action_identity": identity})
            work = []
            for task_id in ("a", "b", "c"):
                baseline = cli.reuse.read(store.root / "baselines" / (digest([task_id, 2000]) + ".json"))
                memories = [c["memory_id"] for c in baseline["candidates"]]
                for size in range(1, len(memories) + 1):
                    for drops in itertools.combinations(memories, size):
                        (reference if size >= 2 else store).put(episode(task_id, baseline["candidates"], drops))
                old = cli.reuse.read(source / "evaluation/policy_parts" / (digest(task_id) + ".json"))[0]
                work.append({"task_id": task_id, "baseline": baseline, "old": old})
            fold = {"source": source, "config": config, "memory": source / "support/g-memory", "work": work}
            job = {"name": "pddl", "label": "pddl/fold_00", "checkpoint": checkpoint, "sources": paths,
                "memory": fold["memory"], "payload": cli.reuse.read(checkpoint), "item": fold}
            args = cli.parse_args(arguments(root, "pddl"))
            before = source_bytes(source_root)
            cli.fit_head(args, job)
            with mock.patch("causal_memory_control.interactive_runtime.NativeRunner", side_effect=AssertionError("unwanted LLM")):
                result = cli.run_pddl(args, {"reference_stores": {job["label"]: reference_path}}, [job])
                self.assertEqual(result["n"], 3)
                self.assertAlmostEqual(result["old_encoder_same_rule"], 2 / 3)
                self.assertEqual(result["untrained_empty_only_tasks"], 0)
                empty = {**job, "label": "pddl/fold_02", "checkpoint": None, "sources": [],
                    "item": {**fold, "work": [work[1]]}}
                with mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", side_effect=AssertionError("empty fold encoder")):
                    cli.fit_head(args, empty)
                    result = cli.run_pddl(args, {}, [empty])
                self.assertEqual(result["untrained_empty_only_tasks"], 1)
                self.assertEqual(result["new_encoder"], 0.)
            self.assertEqual(before, source_bytes(source_root))


if __name__ == "__main__":
    unittest.main()
