"""Fit/dev-only MPNet adaptation, isolated test evaluation and safe resume."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from causal_memory_control import run_st_mpnet_tuning as cli
from causal_memory_control import run_st_encoder_benchmark as encoder_cli
from causal_memory_control import st_encoder_evaluate as wrapper
from causal_memory_control import train_gate, fever_experiment, hotpotqa_experiment
from causal_memory_control.estimator import PotentialOutcomeExample
from causal_memory_control.interactive_data import save, sha
from causal_memory_control.st_estimator import STUtilityEstimator
from causal_memory_control.tests.test_st_encoder_benchmark import encoders
from causal_memory_control.tests.test_seed0_gate_matrix import prepare_source
from causal_memory_control.tests.test_st_policy_reevaluation import source_bytes
from causal_memory_control.tests.test_full_gate_benchmark import cache
from causal_memory_control.tests.test_st_gate import _FeverSTMacNet, _STMemory
from causal_memory_control.tests import test_fever_experiment as fever_tests
from causal_memory_control.tests import test_hotpotqa_gate as hotpot_tests


def fixture(root, name, *, training_task_limit=None):
    """Real measured QA branches/checkpoint, without any completed test log."""
    adapter, args = prepare_source(root, name)
    old = root / "old" / name / "semantic_st"
    checkpoint = root / "old" / name / "checkpoint.json"
    sources = sorted(args.diagnostic_results.glob("*/branches.jsonl"))
    snapshot = cli.reuse.read(args.memory_dir / "causal_snapshot_manifest.json")
    task_subset = (["--task-ids", ",".join(map(str, snapshot["evaluation_ids"][:training_task_limit]))]
                   if training_task_limit is not None else [])
    train_gate.main(["--input", *map(str, sources), "--output", str(checkpoint),
        "--memory-dir", str(args.memory_dir), "--estimator", "st", "--metric", adapter.training_metric,
        "--feature-version", "3", "--ensemble-size", "2", "--st-hidden-sizes", "8,4", "--st-epochs", "2",
        "--calibration-fraction", ".2", "--semantic-model", "test-encoder", "--semantic-revision", "fixed-test",
        *task_subset])
    payload = cli.reuse.read(checkpoint)
    policy = payload["training_metadata"]["policy_calibration"]
    args.checkpoint = checkpoint
    args.output_dir = root / "old" / name
    args.gate_evaluation_seed = 2000
    args.kappa, args.delta, args.max_drops = policy["recommended_kappa"], policy["delta"], policy["recommended_max_drops"]
    save(old / "run_manifest.json", {"run": {"checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "evaluation_ids": snapshot["evaluation_ids"][args.training_claims:args.evaluation_count],
        "kappa": args.kappa, "delta": args.delta, "max_drops": args.max_drops}})
    save(root / "old/reevaluation_manifest.json", {"trained_root": str(root / "trained")})
    save(root / "trained/csm_st_manifest.json", {"source_root": str(root / "source")})
    return adapter, args


def arguments(root, datasets="fever"):
    model = root / "mpnet"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text('{"model_type":"test"}')
    trials = root / "trials.json"
    save(trials, [{"id": "original", "head": {}},
                 {"id": "lr_beta", "head": {"learning_rate": .0003, "beta": 0.}},
                 {"id": "chunk", "head": {"epochs": 1}, "chunk_tokens": 128}])
    return ["--datasets", datasets, "--qa-root", str(root / "old"), "--semantic-model", str(model),
        "--output-dir", str(root / "tuned"), "--mpnet-reference-root", str(root / "mpnet-old"),
        "--rule-reference-root", str(root / "rules"), "--trials-json", str(trials)]


class TuningTests(unittest.TestCase):
    def setup_stack(self, stack):
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))
        stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", side_effect=encoders))

    def test_plan_without_test_logs_or_models_and_tune_actual_heads_resume_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            self.setup_stack(stack)
            fixture(root, "fever")
            argv = arguments(root)
            before = {**source_bytes(root / "old"), **source_bytes(root / "source")}
            with mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", side_effect=AssertionError("plan loaded a model")):
                plan = cli.main(argv)
            self.assertFalse((root / "tuned").exists())
            self.assertFalse(plan["minilm_refitted"])
            self.assertEqual(plan["maximum_head_fits"], 3)
            # No test logs/caches exist: tune must still work, without a runner.
            self.assertFalse((root / "old/fever/semantic_st/memory_gate.jsonl").exists())
            calls = []
            original_fit = STUtilityEstimator.fit

            def fit(estimator, examples, **kwargs):
                calls.append({str(e.group_id) for e in examples})
                return original_fit(estimator, examples, **kwargs)

            with mock.patch.object(STUtilityEstimator, "fit", fit), mock.patch.object(cli.reuse, "prepare_qa", side_effect=AssertionError("test results read during tuning")):
                selected = cli.main([*argv, "--stage", "tune", "--refit-head"])
            self.assertEqual(len(calls), 3)
            jobs, _ = cli.prepare(cli.parse_args(argv))
            self.assertTrue(all(ids == jobs["fever"]["fit"] for ids in calls))
            checkpoints = root / "tuned/fever/trials"
            original = cli.reuse.read(checkpoints / "original/checkpoint.json")
            other = cli.reuse.read(checkpoints / "lr_beta/checkpoint.json")
            chunk = cli.reuse.read(checkpoints / "chunk/checkpoint.json")
            self.assertEqual(original["estimator"]["config"], jobs["fever"]["payload"]["estimator"]["config"])
            self.assertEqual(other["estimator"]["config"]["learning_rate"], .0003)
            self.assertEqual(other["estimator"]["config"]["beta"], 0.)
            self.assertEqual(chunk["feature_builder"]["encoder"]["chunk_tokens"], 128)
            self.assertEqual(chunk["estimator"]["config"]["epochs"], 1)
            for key in cli.POLICY_KEYS:
                self.assertEqual(other["training_metadata"]["policy_calibration"][key],
                                 jobs["fever"]["payload"]["training_metadata"]["policy_calibration"][key])
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            with mock.patch.object(STUtilityEstimator, "fit", side_effect=AssertionError("unwanted refit")):
                self.assertEqual(cli.main([*argv, "--stage", "tune", "--refit-head"]), selected)
            (root / "mpnet/config.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "registered input/configuration changed"):
                cli.main([*argv, "--stage", "tune", "--refit-head"])
            (root / "mpnet/config.json").write_text('{"model_type":"test"}')
            (checkpoints / "original/checkpoint.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                cli.main([*argv, "--stage", "tune", "--refit-head"])

    def test_task_balanced_sign_metrics_and_equal_dataset_shared_selection(self):
        examples = [PotentialOutcomeExample({"p": 1.}, 1., 0., group_id="a")]
        examples += [PotentialOutcomeExample({"p": 0.}, 1., 0., group_id="b")] * 10
        examples += [PotentialOutcomeExample({"p": 0.}, 0., 0., group_id="c")] * 100
        class Predictor:
            def predict_many(self, rows):
                return [SimpleNamespace(utility=row["p"]) for row in rows]
        metrics = cli.utility_metrics(Predictor(), examples)
        self.assertAlmostEqual(metrics["classes"]["positive"]["task_balanced_mse"], .5)
        self.assertAlmostEqual(metrics["class_balanced_mse"], .25)
        manifest = {"datasets": ["fever", "hotpotqa"], "trials": cli.DEFAULT_TRIALS[:2]}
        summaries = {"fever": {"original": {"dev_metrics": {"class_balanced_mse": 1.}},
                               "lr_3e4": {"dev_metrics": {"class_balanced_mse": .8}}},
                     "hotpotqa": {"original": {"dev_metrics": {"class_balanced_mse": .1}},
                                  "lr_3e4": {"dev_metrics": {"class_balanced_mse": .14}}}}
        choice = cli.select_shared(manifest, summaries)
        self.assertEqual(choice["selected_trial"], "original")  # pooled MSE would pick lr_3e4
        summaries["hotpotqa"]["lr_3e4"]["dev_metrics"]["class_balanced_mse"] = .11
        self.assertEqual(cli.select_shared(manifest, summaries)["selected_trial"], "lr_3e4")
        summaries["fever"]["lr_3e4"]["dev_metrics"]["class_balanced_mse"] = 1.
        summaries["hotpotqa"]["lr_3e4"]["dev_metrics"]["class_balanced_mse"] = .1
        self.assertEqual(cli.select_shared(manifest, summaries)["selected_trial"], "original")

    def test_both_datasets_fit_separate_heads_but_select_one_shared_profile(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            self.setup_stack(stack)
            fixture(root, "fever")
            source = root / "source/benchmark_manifest.json"
            first = cli.reuse.read(source)
            _, hotpot_args = fixture(root, "hotpotqa", training_task_limit=10)
            second = cli.reuse.read(source)
            for key in ("experiments", "semantic"):
                first[key].update(second[key])
            save(source, first)
            argv = arguments(root, "fever,hotpotqa")
            jobs, plan = cli.prepare(cli.parse_args(argv))
            old_hotpot_tasks = jobs["hotpotqa"]["fit"] | jobs["hotpotqa"]["dev"]
            self.assertEqual(len(old_hotpot_tasks), 10)
            self.assertLess(len(old_hotpot_tasks), hotpot_args.training_claims)
            counts = plan["sources"]["hotpotqa"]["partition_counts"]
            self.assertEqual(counts["unused_registered_training_tasks"], hotpot_args.training_claims - 10)
            self.assertEqual(counts["checkpoint_training_tasks"], 10)
            result = cli.main([*argv, "--stage", "tune", "--refit-head"])
            self.assertEqual(set(result["selected_checkpoints_sha256"]), {"fever", "hotpotqa"})
            self.assertTrue(all(set(row["per_dataset_relative_mse"]) == {"fever", "hotpotqa"}
                                for row in result["leaderboard"]))
            for name in ("fever", "hotpotqa"):
                paths = list((root / "tuned" / name / "trials").glob("*/trial.json"))
                self.assertEqual(len(paths), 3)
            for path in (root / "tuned/hotpotqa/trials").glob("*/checkpoint.json"):
                payload = cli.reuse.read(path)
                self.assertEqual(cli.partitions(payload), (jobs["hotpotqa"]["fit"], jobs["hotpotqa"]["dev"]))
            with self.assertRaisesRegex(ValueError, "registered input/configuration changed"):
                cli.main([*argv, "--datasets", "fever", "--stage", "tune", "--refit-head"])

    def test_reject_invalid_profiles_leakage_and_same_output(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            self.setup_stack(stack)
            fixture(root, "fever")
            argv = arguments(root)
            with self.assertRaisesRegex(ValueError, "separate, non-nested"):
                cli.parse_args([*argv, "--output-dir", str(root / "old/child")])
            with self.assertRaises(SystemExit):
                cli.parse_args([*argv, "--datasets", "alfworld"])
            with self.assertRaises(SystemExit):
                cli.parse_args([*argv, "--stage", "tune"])
            args = cli.parse_args(argv)
            save(args.trials_json, [{"id": "original", "head": {}}, {"id": "../escape", "head": {}}])
            with self.assertRaisesRegex(ValueError, "invalid/duplicate"):
                cli.trial_specs(args)
            save(args.trials_json, [{"id": "original", "head": {}}, {"id": "bad", "head": {"beta": -1}}])
            with self.assertRaisesRegex(ValueError, "invalid trial parameter"):
                cli.trial_specs(args)
            save(args.trials_json, [{"id": "original", "head": {}}])
            manifest = root / "old/fever/semantic_st/run_manifest.json"
            value = cli.reuse.read(manifest)
            value["run"]["evaluation_ids"] = [999]
            save(manifest, value)
            with self.assertRaisesRegex(ValueError, "partition mismatch or leakage"):
                cli.main(argv)

    def test_reject_checkpoint_tasks_outside_training_pool_and_support_test_overlap(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            self.setup_stack(stack)
            _, source_args = fixture(root, "hotpotqa", training_task_limit=10)
            argv = arguments(root, "hotpotqa")
            args = cli.parse_args(argv)
            jobs, _ = cli.prepare(args)
            used = sorted(jobs["hotpotqa"]["fit"] | jobs["hotpotqa"]["dev"])[0]
            path = jobs["hotpotqa"]["memory"] / "causal_snapshot_manifest.json"
            original = cli.reuse.read(path)
            for reason in ("checkpoint_tasks_outside_training_pool", "checkpoint_tasks_in_test_or_support",
                           "registered_training_overlap", "support_test_overlap"):
                with self.subTest(reason=reason):
                    snapshot = deepcopy(original)
                    if reason == "checkpoint_tasks_outside_training_pool":
                        snapshot["evaluation_ids"] = [999 if str(v) == used else v
                                                      for v in snapshot["evaluation_ids"]]
                    elif reason == "checkpoint_tasks_in_test_or_support":
                        snapshot["support_ids"].append(int(used))
                    elif reason == "registered_training_overlap":
                        unused = next(v for v in original["evaluation_ids"][:source_args.training_claims]
                                      if str(v) not in (jobs["hotpotqa"]["fit"] | jobs["hotpotqa"]["dev"]))
                        snapshot["support_ids"].append(unused)
                    else:
                        snapshot["support_ids"].append(snapshot["evaluation_ids"][-1])
                    save(path, snapshot)
                    with self.assertRaisesRegex(ValueError, reason):
                        cli.prepare(args)
            save(path, original)
            self.assertFalse((root / "tuned").exists())

    def test_output_lock_and_incomplete_trial_resume(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            self.setup_stack(stack)
            fixture(root, "fever")
            argv = arguments(root)
            with cli.output_lock(root / "tuned"):
                with self.assertRaisesRegex(ValueError, "already running/locked"):
                    cli.main([*argv, "--stage", "tune", "--refit-head"])
            original_fit = STUtilityEstimator.fit
            count = [0]
            def fail_after_first(estimator, examples, **kwargs):
                count[0] += 1
                if count[0] == 2:
                    raise RuntimeError("interrupted head")
                return original_fit(estimator, examples, **kwargs)
            with mock.patch.object(STUtilityEstimator, "fit", fail_after_first):
                with self.assertRaisesRegex(RuntimeError, "interrupted head"):
                    cli.main([*argv, "--stage", "tune", "--refit-head"])
            self.assertTrue((root / "tuned/fever/trials/original/trial.json").is_file())
            self.assertFalse((root / "tuned/selection.json").exists())
            calls = []
            def record(estimator, examples, **kwargs):
                calls.append(1)
                return original_fit(estimator, examples, **kwargs)
            with mock.patch.object(STUtilityEstimator, "fit", record):
                cli.main([*argv, "--stage", "tune", "--refit-head"])
            self.assertEqual(len(calls), 2)

    def test_actual_selected_only_qa_evaluation_and_original_results_unchanged(self):
        for name in ("fever", "hotpotqa"):
            with self.subTest(dataset=name), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                self.setup_stack(stack)
                adapter, old_args = fixture(root, name)
                runner = fever_experiment if name == "fever" else hotpotqa_experiment
                if name == "fever":
                    for patch in fever_tests.FeverRunnerTests._patches():
                        stack.enter_context(patch)
                else:
                    hotpot_tests.HotpotRunnerTests.patches(stack)
                stack.enter_context(mock.patch.object(runner, "GMemory", _STMemory))
                stack.enter_context(mock.patch.object(runner, "MacNet", _FeverSTMacNet if name == "fever" else hotpot_tests._HotpotMacNet))
                # Replace the planning-only manifest with real complete original evaluations.
                (root / "old" / name / "semantic_st/run_manifest.json").unlink()
                for mode, target in (("always_keep", "native_gmemory"), ("learned", "semantic_st")):
                    output = old_args.output_dir / target
                    runner.main(adapter.evaluation_command(old_args, mode=mode, output=output, resume=False)[4:])
                    if not (output / "llm_cache.sqlite").exists():
                        cache(output / "llm_cache.sqlite")
                retest = root / "rules" / name / "semantic_st"
                wrapper.main(["--dataset", name, "--reference-log", str(old_args.output_dir / "semantic_st/memory_gate.jsonl"),
                    "--st-deletion-rule", "positive_unlimited",
                    *adapter.evaluation_command(old_args, mode="learned", output=retest, resume=False)[4:]])
                argv = arguments(root, name)
                # Also create the untouched original MPNet comparison using its existing entry point.
                encoder_args = encoder_cli.parse_args(["--datasets", name, "--qa-root", str(root / "old"),
                    "--output-dir", str(root / "mpnet-old"), "--semantic-model", str(root / "mpnet"),
                    "--rule-reference-root", str(root / "rules")])
                job = {"name": name, "label": name, "payload": cli.reuse.read(old_args.checkpoint),
                    "checkpoint": old_args.checkpoint, "sources": sorted(old_args.diagnostic_results.glob("*/branches.jsonl")),
                    "memory": old_args.memory_dir}
                encoder_cli.fit_head(encoder_args, job)
                item = {"args": old_args, "adapter": adapter, "source_memory": old_args.memory_dir,
                    "old": old_args.output_dir / "semantic_st", "baseline": old_args.output_dir / "native_gmemory",
                    "cache": old_args.output_dir / "semantic_st/llm_cache.sqlite", "rule_reference": retest / "memory_gate.jsonl"}
                calls = []
                def execute(command, *a, **kw):
                    calls.append(command)
                    return wrapper.main(command[4:])
                stack.enter_context(mock.patch("causal_memory_control.interactive_progress.run_logged", side_effect=execute))
                encoder_cli.run_qa(encoder_args, name, item, job)
                calls.clear()
                before = {**source_bytes(root / "source"), **source_bytes(root / "old"),
                          **source_bytes(root / "mpnet-old"), **source_bytes(root / "rules")}
                selection = cli.main([*argv, "--stage", "tune", "--refit-head"])
                self.assertFalse(calls)
                selected_path = root / "tuned/selection.json"
                selected_bytes = selected_path.read_bytes()
                changed = deepcopy(selection)
                changed["selected_trial"] = "not_selected"
                save(selected_path, changed)
                with self.assertRaisesRegex(ValueError, "registered input/configuration changed"):
                    cli.main([*argv, "--stage", "evaluate"])
                selected_path.write_bytes(selected_bytes)
                reference_checkpoint = root / "mpnet-old" / name / "checkpoint.json"
                reference_bytes = reference_checkpoint.read_bytes()
                changed = cli.reuse.read(reference_checkpoint)
                changed["estimator"]["config"]["beta"] += .1
                save(reference_checkpoint, changed)
                with self.assertRaisesRegex(ValueError, "original MPNet reference uses different"):
                    cli.main([*argv, "--stage", "evaluate"])
                self.assertFalse(calls)
                reference_checkpoint.write_bytes(reference_bytes)
                with mock.patch.object(STUtilityEstimator, "fit", side_effect=AssertionError("evaluation trained a head")):
                    result = cli.main([*argv, "--stage", "evaluate"])
                    cli.main([*argv, "--stage", "evaluate"])
                self.assertEqual(len(calls), 1)
                evaluated = Path(calls[0][calls[0].index("--checkpoint") + 1])
                self.assertEqual(evaluated.parent.name, selection["selected_trial"])
                self.assertEqual(result[name]["n"], old_args.final_claims)
                self.assertIsNotNone(result[name]["original_mpnet_same_rule"])
                self.assertIsNotNone(result[name]["original_minilm_same_rule"])
                self.assertEqual(before, {p: p.read_bytes() for p in before})
                self.assertTrue((root / "tuned/results.md").is_file())
                self.assertTrue((root / "tuned/dev_results.md").is_file())


if __name__ == "__main__":
    unittest.main()
