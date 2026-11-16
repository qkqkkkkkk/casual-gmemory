"""Dev-only decision selection, runtime equivalence, isolated QA evaluation."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from causal_memory_control import run_st_decision_calibration as cli
from causal_memory_control import st_decision_calibration as decision
from causal_memory_control import run_st_mpnet_tuning as tuning
from causal_memory_control import run_st_encoder_benchmark as encoder_cli
from causal_memory_control import st_encoder_evaluate, st_single_drop_evaluate
from causal_memory_control import fever_experiment, hotpotqa_experiment
from causal_memory_control.csm_st_worker import dedup_identity
from causal_memory_control.interactive_data import save
from causal_memory_control.runtime_gate import GMemoryExposureGate
from causal_memory_control.st_estimator import STUtilityEstimator
from causal_memory_control.tests.test_st_mpnet_tuning import fixture, arguments
from causal_memory_control.tests.test_st_encoder_benchmark import encoders
from causal_memory_control.tests.test_st_policy_reevaluation import source_bytes
from causal_memory_control.tests.test_full_gate_benchmark import cache
from causal_memory_control.tests.test_st_gate import _FeverSTMacNet, _STMemory
from causal_memory_control.tests import test_fever_experiment as fever_tests
from causal_memory_control.tests import test_hotpotqa_gate as hotpot_tests


def row(task="1", memory="a", *, predicted=-.1, sigma=0., gain=.2, order=0, scope=True):
    return {"task_id": task, "memory_id": memory, "memory_type": "insight", "rank": order + 1,
        "order": order, "predicted_u": predicted, "uncertainty": sigma, "q_use": .5,
        "q_drop": .5 + gain, "calibrated": True, "in_scope": scope}


class DecisionTests(unittest.TestCase):
    def test_selection_uses_task_gain_and_single_drop_not_candidate_sums(self):
        records = [row("1", "harm", predicted=-.03, sigma=.04, gain=-.3),
                   row("2", "help", predicted=-.2, gain=.4),
                   row("2", "tempting", predicted=-.1, gain=-.4, order=1)]
        chosen = decision.calibrate(records, {"1", "2", "3"})
        policy = chosen["selected"]["policy"]
        score = decision.assess(records, {"1", "2", "3"}, policy)
        self.assertAlmostEqual(score["mean_observed_gain"], .4 / 3)
        self.assertEqual(score["drop_tasks"], 1)
        self.assertEqual(score["improved_tasks"], 1)
        self.assertEqual(policy["max_drops"], 1)
        self.assertEqual(score["task_decisions"][1]["drop_ids"], ["help"])
        self.assertEqual(len(chosen["grid"]), 26)
        self.assertAlmostEqual(score["mean_regret_to_observed_single_drop_oracle"], 0.)

    def test_keep_all_strict_zero_scopes_and_stable_retrieval_order(self):
        records = [row(gain=-.2), row(memory="zero", predicted=0, order=1, gain=.3),
                   row(memory="excluded", predicted=-1., order=2, gain=.5, scope=False)]
        self.assertEqual(decision.calibrate(records, {"1"})["selected"]["policy"], decision.KEEP_ALL)
        neutral = [row(gain=0.)]
        self.assertEqual(decision.calibrate(neutral, {"1"})["selected"]["policy"], decision.KEEP_ALL)
        tied = [row(memory="second", gain=-.3, order=1), row(memory="first", order=0)]
        chosen = decision.choose(tied, {"kappa": 0, "delta": 0, "max_drops": 1})
        self.assertEqual([r["memory_id"] for r in chosen], ["first"])
        diag = decision.diagnose(records, {"1"}, {"rule": "positive_unlimited", "kappa": 0, "delta": 0, "max_drops": None}, decision.KEEP_ALL)
        self.assertEqual(diag["predicted_exact_zero"], 1)
        self.assertFalse(diag["policies"]["original_rule"]["joint_effect_identified"])
        self.assertIsNone(diag["policies"]["original_rule"]["mean_observed_gain"])

    def test_invalid_coverage_and_nonfinite_prediction_fail(self):
        for records in ([row(), row()], [row(task="test")], [row(predicted=float("nan"))], [row(sigma=-1)]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                decision.calibrate(records, {"1"})

    def test_offline_choice_matches_real_runtime_scope_threshold_and_budget(self):
        from causal_memory_control.tests.test_semantic_jev import TinyEncoder
        from causal_memory_control.semantic_features import SemanticTeamFeatureBuilder
        from causal_memory_control.types import PotentialOutcomePrediction

        class Predictor:
            fitted = True

            def predict(self, features):
                u = -.2 if features.get("semantic_memory::2") else -.05
                return PotentialOutcomePrediction(.5 + u / 2, .5 - u / 2, u, "negative", .02, "test", True, 20)

        for policy in (decision.KEEP_ALL, {"kappa": 1., "delta": .01, "max_drops": 1}):
            gate = GMemoryExposureGate(mode="learned", estimator=Predictor(),
                feature_builder=SemanticTeamFeatureBuilder(TinyEncoder()), **{k: policy[k] for k in ("kappa", "delta", "max_drops")})
            gate.begin_task(1, {})
            gate.filter_retrieval(([], [], ["helpful", "harmful"]), query_task="task", task_state="state")
            rows = [row(memory=d.memory_id, predicted=d.prediction.utility, sigma=d.prediction.uncertainty, order=i)
                    for i, d in enumerate(gate._task_decisions)]
            actual = [d.memory_id for d in gate._task_decisions if d.action.value == "drop"]
            self.assertEqual(actual, [r["memory_id"] for r in decision.choose(rows, policy)])


def test_identity(encoder, spec):
    result = dedup_identity(encoder, spec)
    if spec["model"] == "test-encoder":
        result["resolved_commit"] = "fixed-test"
    return result


class ExperimentTests(unittest.TestCase):
    def setup_stack(self, stack):
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))
        stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", side_effect=encoders))
        stack.enter_context(mock.patch("causal_memory_control.csm_st_worker.dedup_identity", side_effect=test_identity))

    def setup_models(self, root, name):
        adapter, old_args = fixture(root, name)
        argv = arguments(root, name)
        tuning.main([*argv, "--stage", "tune", "--refit-head"])
        encoder_args = encoder_cli.parse_args(["--datasets", name, "--qa-root", str(root / "old"),
            "--output-dir", str(root / "mpnet-old"), "--semantic-model", str(root / "mpnet"),
            "--rule-reference-root", str(root / "rules")])
        job = {"name": name, "label": name, "payload": cli.reuse.read(old_args.checkpoint),
            "checkpoint": old_args.checkpoint, "sources": sorted(old_args.diagnostic_results.glob("*/branches.jsonl")),
            "memory": old_args.memory_dir}
        encoder_cli.fit_head(encoder_args, job)
        return adapter, old_args, argv, encoder_args, job

    def argv(self, root, name):
        return ["--datasets", name, "--tuning-root", str(root / "tuned"), "--output-dir", str(root / "decisions")]

    def test_read_only_plan_dev_only_calibration_no_fitting_resume_and_tamper(self):
        for name in ("fever", "hotpotqa"):
            with self.subTest(dataset=name), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                self.setup_stack(stack)
                self.setup_models(root, name)
                before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
                argv = self.argv(root, name)
                with mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", side_effect=AssertionError("plan loaded encoder")):
                    plan = cli.main(argv)
                self.assertFalse((root / "decisions").exists())
                self.assertFalse(plan["new_training"])
                self.assertEqual(set(plan["models"][name]), set(cli.VARIANTS))
                self.assertFalse((root / "old" / name / "semantic_st/memory_gate.jsonl").exists())
                with mock.patch.object(STUtilityEstimator, "fit", side_effect=AssertionError("unwanted fitting")), mock.patch.object(cli.reuse, "prepare_qa", side_effect=AssertionError("held-out results read")):
                    selected = cli.main([*argv, "--stage", "calibrate"])
                    with mock.patch.object(cli, "dev_predictions", side_effect=AssertionError("resume recomputed predictions")):
                        self.assertEqual(cli.main([*argv, "--stage", "calibrate"]), selected)
                dev = set(plan["partitions"][name]["dev_ids"])
                for variant in cli.VARIANTS:
                    output = root / "decisions" / name / variant
                    records = cli.reuse.read(output / "dev_predictions.json")
                    self.assertEqual({r["task_id"] for r in records}, dev)
                    self.assertIn(cli.reuse.read(output / "policy.json")["policy"]["max_drops"], (0, 1))
                    self.assertEqual(len(cli.reuse.read(output / "calibration.json")["grid"]), 26)
                self.assertTrue((root / "decisions/dev_diagnostics.md").is_file())
                self.assertEqual(before, {p: p.read_bytes() for p in before})
                path = root / "decisions" / name / "original_mpnet/dev_predictions.json"
                path.write_text("[]")
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    cli.main([*argv, "--stage", "calibrate"])

    def test_preflight_rejects_nested_output_split_changes_and_weight_drift(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            self.setup_stack(stack)
            self.setup_models(root, "fever")
            argv = self.argv(root, "fever")
            with self.assertRaisesRegex(ValueError, "non-nested"):
                cli.main([*argv, "--output-dir", str(root / "tuned/new")])
            path = root / "mpnet-old/fever/checkpoint.json"
            before = path.read_bytes()
            changed = cli.reuse.read(path)
            changed["training_metadata"]["fit_task_ids"] = ["test-task"]
            save(path, changed)
            with self.assertRaisesRegex(ValueError, "partition|split"):
                cli.main(argv)
            path.write_bytes(before)
            (root / "mpnet/config.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "weights differ"):
                cli.main([*argv, "--stage", "calibrate"])

    def test_full_qa_single_drop_evaluation_audits_retrieval_predictions_and_resumes(self):
        for name in ("fever", "hotpotqa"):
            with self.subTest(dataset=name), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                self.setup_stack(stack)
                adapter, old_args, tuning_argv, encoder_args, job = self.setup_models(root, name)
                runner = fever_experiment if name == "fever" else hotpotqa_experiment
                if name == "fever":
                    for patch in fever_tests.FeverRunnerTests._patches():
                        stack.enter_context(patch)
                else:
                    hotpot_tests.HotpotRunnerTests.patches(stack)
                stack.enter_context(mock.patch.object(runner, "GMemory", _STMemory))
                stack.enter_context(mock.patch.object(runner, "MacNet", _FeverSTMacNet if name == "fever" else hotpot_tests._HotpotMacNet))
                (root / "old" / name / "semantic_st/run_manifest.json").unlink()
                for mode, label in (("always_keep", "native_gmemory"), ("learned", "semantic_st")):
                    target = old_args.output_dir / label
                    runner.main(adapter.evaluation_command(old_args, mode=mode, output=target, resume=False)[4:])
                    if not (target / "llm_cache.sqlite").exists():
                        cache(target / "llm_cache.sqlite")
                retest = root / "rules" / name / "semantic_st"
                st_encoder_evaluate.main(["--dataset", name, "--reference-log", str(old_args.output_dir / "semantic_st/memory_gate.jsonl"),
                    "--st-deletion-rule", "positive_unlimited",
                    *adapter.evaluation_command(old_args, mode="learned", output=retest, resume=False)[4:]])
                if not (retest / "llm_cache.sqlite").exists():
                    cache(retest / "llm_cache.sqlite")

                calls = []

                def execute(command, *a, **kw):
                    calls.append(command)
                    module = st_single_drop_evaluate if command[3].endswith("st_single_drop_evaluate") else st_encoder_evaluate
                    result = module.main(command[4:])
                    target = Path(command[command.index("--output-dir") + 1])
                    if not (target / "llm_cache.sqlite").exists():
                        cache(target / "llm_cache.sqlite")
                    return result

                stack.enter_context(mock.patch("causal_memory_control.interactive_progress.run_logged", side_effect=execute))
                item = {"args": old_args, "adapter": adapter, "source_memory": old_args.memory_dir,
                    "old": old_args.output_dir / "semantic_st", "baseline": old_args.output_dir / "native_gmemory",
                    "cache": old_args.output_dir / "semantic_st/llm_cache.sqlite", "rule_reference": retest / "memory_gate.jsonl"}
                encoder_cli.run_qa(encoder_args, name, item, job)
                # Finish the already selected tuned checkpoint's original rule.
                # Real run manifests now replace the fixture's planning-only one.
                # The original tuning pin was made before those manifests existed;
                # re-register the identical checkpoint's now-complete run inputs.
                tuning_manifest = root / "tuned/manifest.json"
                current_jobs, current_manifest = tuning.prepare(tuning.parse_args(tuning_argv))
                self.assertEqual(current_jobs[name]["payload"], job["payload"])
                save(tuning_manifest, current_manifest)
                selected = cli.reuse.read(root / "tuned/selection.json")
                from causal_memory_control.interactive_data import digest
                selected["manifest_sha256"] = digest(current_manifest)
                save(root / "tuned/selection.json", selected)
                tuning.evaluate(tuning.parse_args(tuning_argv), current_jobs, current_manifest, selected)
                calls.clear()
                before = {**source_bytes(root / "old"), **source_bytes(root / "source"),
                    **source_bytes(root / "tuned"), **source_bytes(root / "mpnet-old"), **source_bytes(root / "rules")}
                argv = self.argv(root, name)
                with mock.patch.object(STUtilityEstimator, "fit", side_effect=AssertionError("calibration/evaluation retrained")):
                    cli.main([*argv, "--stage", "calibrate"])
                    result = cli.main([*argv, "--stage", "evaluate"])
                    cli.main([*argv, "--stage", "evaluate"])
                self.assertEqual(len(calls), 3)
                self.assertEqual(set(result[name]), set(cli.VARIANTS))
                for variant in cli.VARIANTS:
                    output = root / "decisions" / name / variant / "semantic_st"
                    audit = cli.reuse.read(output / "original_rule_comparison.json")
                    self.assertTrue(audit["paired_control_valid"])
                    self.assertEqual(audit["same_selection_mismatches"], [])
                    rows = cli.reuse.rows(output / "memory_gate.jsonl")
                    self.assertEqual(len(rows), old_args.final_claims)
                    self.assertTrue(all(r["dropped_count"] <= 1 for r in rows))
                self.assertEqual(before, {p: p.read_bytes() for p in before})
                self.assertTrue((root / "decisions/results.md").is_file())
                command = calls[0]
                bad = command.copy()
                bad[bad.index("--max-drops") + 1] = "-1"
                with self.assertRaisesRegex(ValueError, "runtime arguments"):
                    st_single_drop_evaluate.main(bad[4:])
                # Changes in original retrieval/predictions cannot be hidden by
                # merely reusing previously completed new results.
                ref = Path(command[command.index("--reference-log") + 1])
                ref.write_text(ref.read_text().replace('"utility":', '"utility_changed":', 1))
                with self.assertRaisesRegex(ValueError, "reference changed|registered input/configuration changed"):
                    cli.main([*argv, "--stage", "evaluate"])


if __name__ == "__main__":
    unittest.main()
