"""ALFWorld encoder ablation: preserve old eight policies and measured labels."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
import io
import tempfile
import unittest
from unittest import mock

from causal_memory_control import run_alfworld_encoder_comparison as suite
from causal_memory_control import run_partial_alfworld as partial
from causal_memory_control import run_st_encoder_benchmark as encoder
from causal_memory_control import interactive_worker as worker
from causal_memory_control.estimator import PotentialOutcomeExample
from causal_memory_control.interactive_data import digest, files_identity, save, sha
from causal_memory_control.interactive_empty_memory import coverage
from causal_memory_control.interactive_protocol import learned_methods
from causal_memory_control.interactive_runtime import event_from_row
from causal_memory_control.interactive_training import diagnostic_lists
from causal_memory_control.resume_interactive_dataset import experiment_lock
from causal_memory_control.runtime_gate import GMemoryExposureGate, save_gate_checkpoint
from causal_memory_control.semantic_features import SemanticTeamFeatureBuilder
from causal_memory_control.st_estimator import STUtilityEstimator
from causal_memory_control.tests.test_partial_alfworld import source_fixture
from causal_memory_control.tests.test_semantic_jev import TinyEncoder
from causal_memory_control.tests.test_st_encoder_benchmark import encoders


def arguments(root):
    model = root / "new-model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text('{"model_type":"test"}')
    return ["--source-dir", str(root / "source"), "--base-dir", str(root / "base"),
        "--output-dir", str(root / "comparison"), "--expected-completed", "2",
        "--semantic-model", str(model), "--train-device", "cpu"]


def completed_base(root):
    source, _, runner, calls, _ = source_fixture(root)
    base = root / "base"
    partial.run(source, base, stage="prepare", expected_completed=2)
    config = partial.read(base / "benchmark_manifest.json")
    data = config["experiments"]["alfworld"]
    rows = diagnostic_lists(base / "alfworld", data)
    builder = SemanticTeamFeatureBuilder(TinyEncoder())
    examples = [PotentialOutcomeExample(builder.build(event_from_row(row, i)), row["q_use"], row["q_drop"][i],
        digest([row["task_id"], i]), row["task_id"]) for row in rows for i in range(len(row["candidates"]))]
    estimator = STUtilityEstimator(ensemble_size=2, min_samples=2, hidden_sizes=(8, 4), epochs=2,
                                    residual_noise_scale=0.0)
    if not estimator.fit([e for e in examples if e.group_id in data["split"]["fit"]]):
        raise AssertionError("tiny test head did not fit")
    metadata = {"config_digest": digest(config), "metric": "success", "training_task_ids": ["1", "3"],
        "fit_task_ids": ["1"], "dev_task_ids": ["3"], "test_task_ids": ["4", "5"],
        "candidate_coverage": coverage(rows, data), "candidate_kinds": ["insight"],
        "diagnostic_sha256": {r["task_id"]: sha(base / "alfworld/diagnostic/tasks" / (digest(r["task_id"]) + ".json")) for r in rows},
        "policy_calibration": {"recommended_kappa": 0., "recommended_max_drops": 1, "delta": 0.,
                               "calibration_task_ids": ["3"]}}
    checkpoint = base / "alfworld/semantic_st/checkpoint.json"
    save_gate_checkpoint(checkpoint, estimator, feature_version=3,
        feature_config={"encoder": TinyEncoder().config()}, training_metadata=metadata)
    # Only ST is the model under test; other gate checkpoints are inert fixture identities.
    for method in set(learned_methods(config)) - {"semantic_st"}:
        save(base / "alfworld" / method / "checkpoint.json", {"fixture": method})
    gates = {method: GMemoryExposureGate.from_checkpoint(checkpoint, mode="learned", kappa=0, delta=0, max_drops=1)
             for method in learned_methods(config)}
    with mock.patch("causal_memory_control.interactive_training.learned_gates", return_value=gates), \
         mock.patch("causal_memory_control.interactive_runtime.NativeRunner", runner):
        worker.evaluate(config, "alfworld")
    _, plan = partial.prepared(base, source, expected_completed=2)
    partial.partial_report(config, plan)
    return source, base, config, calls


class ALFWorldEncoderComparisonTests(unittest.TestCase):
    def test_plan_is_read_only_and_explicitly_preserves_old_st_and_test_split(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            source_fixture(root)
            argv = arguments(root)
            before = files_identity(root)
            with mock.patch("causal_memory_control.interactive_runtime.NativeRunner", side_effect=AssertionError("model")):
                plan = suite.main(argv)
            self.assertEqual(plan["completed_tasks"], 2)
            self.assertEqual(plan["split_counts"]["test"], 2)
            self.assertEqual(plan["writes_or_model_or_service_calls"], 0)
            self.assertEqual(files_identity(root), before)
            self.assertFalse((root / "comparison").exists())

    def test_real_new_st_head_and_nine_policy_report_reuse_measured_actions_and_resume(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", side_effect=encoders))
            source, base, config, _ = completed_base(root)
            argv = arguments(root)
            before_source, before_base = files_identity(source), files_identity(base)
            old = partial.read(base / "alfworld/semantic_st/checkpoint.json")
            with mock.patch.object(partial, "run") as base_run, \
                 mock.patch("causal_memory_control.interactive_runtime.NativeRunner", side_effect=AssertionError("all one-drop outcomes already measured")):
                suite.main([*argv, "--stage", "all"])
            base_run.assert_called_once()
            output = root / "comparison"
            new = partial.read(output / "encoder/alfworld/checkpoint.json")
            self.assertEqual(old["feature_builder"]["encoder"]["dimensions"], 3)
            self.assertEqual(new["feature_builder"]["encoder"]["dimensions"], 4)
            self.assertEqual(encoder.partitions(old), encoder.partitions(new))
            self.assertEqual(old["estimator"]["config"], new["estimator"]["config"])
            for key in ("recommended_kappa", "recommended_max_drops", "delta"):
                self.assertEqual(old["training_metadata"]["policy_calibration"][key],
                                 new["training_metadata"]["policy_calibration"][key])
            result = partial.read(output / "results.json")
            self.assertEqual(len(result["policies"]), 12)  # original eight + three random seeds + new ST
            for name in ("semantic_st", suite.NEW_POLICY):
                self.assertEqual(result["policies"][name]["n"], 1)
                self.assertEqual(result["policies"][name]["cohort_results"]["train_scene_holdout"]["n"], 1)
            self.assertEqual(result["partial_diagnostic"]["completed_tasks"], 2)
            self.assertIn("semantic_st_new_encoder", (output / "results.md").read_text())
            self.assertEqual(files_identity(source), before_source)
            self.assertEqual(files_identity(base), before_base)
            self.assertEqual(partial.read(output / "encoder/summary.json")["datasets"]["alfworld"]["evaluated_test_tasks"], 2)
            with mock.patch.object(STUtilityEstimator, "fit", side_effect=AssertionError("unwanted refit")), \
                 mock.patch("causal_memory_control.interactive_runtime.NativeRunner", side_effect=AssertionError("unwanted episode")):
                suite.main([*argv, "--stage", "evaluate"])
            self.assertEqual(files_identity(source), before_source)
            self.assertEqual(files_identity(base), before_base)
            with experiment_lock(base), self.assertRaisesRegex(ValueError, "running/locked"):
                suite.main([*argv, "--stage", "report"])
            # A forged task outcome cannot enter the combined comparison.
            log = output / "encoder/alfworld/policies.jsonl"
            values = encoder.reuse.rows(log)
            values[0]["outcome"] = {**values[0]["outcome"], "success": 1 - values[0]["outcome"]["success"]}
            log.write_text("".join(__import__("json").dumps(row) + "\n" for row in values))
            with self.assertRaisesRegex(ValueError, "matched measured action"):
                suite.main([*argv, "--stage", "report"])

    def test_alfworld_encoder_requires_source_lock(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            source_fixture(root)
            argv = arguments(root)
            args = suite.parse_args(argv)
            args.base_dir.mkdir()
            with experiment_lock(args.base_dir), mock.patch.object(encoder, "prepare") as prepare:
                with self.assertRaisesRegex(ValueError, "running/locked"):
                    encoder.main(suite.encoder_arguments(args, "fit"))
                prepare.assert_not_called()

    def test_encoder_refuses_incomplete_original_test_results_before_fitting(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", side_effect=encoders))
            _, base, _, _ = completed_base(root)
            args = suite.parse_args(arguments(root))
            (base / "alfworld/evaluation/policy_parts" / (digest("5") + ".json")).unlink()
            with mock.patch.object(encoder, "fit_head", side_effect=AssertionError("no fitting on incomplete base")):
                with self.assertRaises(FileNotFoundError):
                    encoder.main(suite.encoder_arguments(args, "fit"))
            self.assertFalse((root / "comparison/encoder/manifest.json").exists())

    def test_empty_memory_preserves_all_tasks_without_creating_a_new_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            source, _, runner, _, _ = source_fixture(root, empty=True)
            argv = arguments(root)
            def run_worker(command, path, *, cwd):
                return worker.main(command[command.index("--config"):])
            stack.enter_context(mock.patch.object(partial, "run_logged", side_effect=run_worker))
            stack.enter_context(mock.patch("causal_memory_control.interactive_runtime.NativeRunner", runner))
            stack.enter_context(mock.patch("causal_memory_control.interactive_runtime.model_identity", return_value={}))
            stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", side_effect=AssertionError("empty encoder")))
            suite.main([*argv, "--stage", "all"])
            result = partial.read(root / "comparison/results.json")
            self.assertEqual(result["policies"][suite.NEW_POLICY]["n"], 1)
            self.assertFalse((root / "comparison/encoder/alfworld/checkpoint.json").exists())
            self.assertTrue((root / "comparison/encoder/alfworld/training_status.json").exists())


if __name__ == "__main__":
    unittest.main()
