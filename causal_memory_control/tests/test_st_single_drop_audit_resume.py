"""Numeric drift must preserve retrieval, deletion choices and frozen progress."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from causal_memory_control import resume_st_single_drop_evaluation as resume
from causal_memory_control import run_st_decision_calibration as original
from causal_memory_control import run_st_mpnet_tuning as tuning
from causal_memory_control import run_st_encoder_benchmark as encoder_cli
from causal_memory_control import st_encoder_evaluate
from causal_memory_control import st_single_drop_audit_evaluate as audit
from causal_memory_control import fever_experiment, hotpotqa_experiment
from causal_memory_control.interactive_data import digest, save
from causal_memory_control.st_estimator import STUtilityEstimator
from causal_memory_control.tests import test_st_decision_calibration as fixtures_module
from causal_memory_control.tests.test_st_gate import _STMemory, _FeverSTMacNet
from causal_memory_control.tests.test_full_gate_benchmark import cache
from causal_memory_control.tests.test_st_policy_reevaluation import source_bytes
from causal_memory_control.tests import test_fever_experiment as fever_tests
from causal_memory_control.tests import test_hotpotqa_gate as hotpot_tests


def decision(memory="a", utility=-.2, uncertainty=.01, action="DROP"):
    return {"memory_id": memory, "memory_type": "insight", "memory_content": memory,
        "memory_metadata": {}, "retrieval_rank": 1, "action": action, "reason": "confidently_harmful",
        "prediction": {"q_use": .5 + utility / 2, "q_drop": .5 - utility / 2,
            "utility": utility, "utility_class": "negative" if utility < 0 else "positive" if utility > 0 else "neutral",
            "uncertainty": uncertainty, "source": "bootstrap_semantic_st_direct_u", "calibrated": True, "training_samples": 10}}


POLICY = {"rule": "single_drop_calibrated", "kappa": 1., "delta": .1, "max_drops": 1}


class NumericAuditTests(unittest.TestCase):
    def assess(self, actual, expected, policy=POLICY, fingerprint="same"):
        return audit.audit_decisions(actual, {"retrieval_fingerprint": "same", "decisions": expected}, policy,
                                     fingerprint=fingerprint, task_id="150648")

    def test_tiny_float_error_accepted_with_evidence_but_large_error_rejected(self):
        old = decision()
        current = decision(utility=-.2 + 5e-8, uncertainty=.01 + 1e-8)
        result = self.assess([current], [old])
        self.assertEqual(result["status"], "tolerated_numeric_difference")
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["reference_prediction_drop_ids_under_new_policy"], ["a"])
        self.assertTrue(result["prediction_differences"])
        result = self.assess([decision(utility=-.19)], [old])
        self.assertEqual(result["status"], "failed")
        self.assertIn("prediction_changed_candidate_0", result["errors"])

    def test_retrieval_order_text_metadata_and_fingerprint_remain_exact(self):
        old = [decision(), decision("b", action="KEEP")]
        for field, value in (("memory_content", "different"), ("memory_metadata", {"new": 1}),
                             ("retrieval_rank", 2), ("memory_id", "other")):
            current = deepcopy(old)
            current[0][field] = value
            self.assertEqual(self.assess(current, old)["status"], "failed", field)
        self.assertEqual(self.assess(old[::-1], old)["status"], "failed")
        self.assertEqual(self.assess(old[:1], old)["status"], "failed")
        self.assertIn("retrieval_fingerprint_changed", self.assess(old, old, fingerprint="other")["errors"])

    def test_small_error_must_not_change_threshold_or_ranking_choice(self):
        old = decision(utility=-.11)
        current = decision(utility=-.11000001, action="DROP")
        # Reference upper = -.1, which is kept by the strict inequality.
        self.assertIn("numerical_difference_changes_deletion_choice", self.assess([current], [old])["errors"])
        old = [decision("a", utility=-.2), decision("b", utility=-.2, action="KEEP")]
        current = [decision("a", utility=-.2, action="KEEP"), decision("b", utility=-.20000001)]
        self.assertIn("numerical_difference_changes_deletion_choice", self.assess(current, old)["errors"])

    def test_keep_all_and_near_zero_derived_label_allow_only_same_decision(self):
        policy = {**POLICY, "max_drops": 0}
        self.assertEqual(self.assess([decision(utility=-.20000001, action="KEEP")],
                                    [decision()], policy)["status"], "tolerated_numeric_difference")
        old = decision(utility=0, action="KEEP")
        current = decision(utility=1e-8, action="KEEP")
        self.assertEqual(self.assess([current], [old])["status"], "tolerated_numeric_difference")
        current["prediction"]["source"] = "different_model"
        self.assertEqual(self.assess([current], [old])["status"], "failed")
        current = decision()
        current["prediction"]["utility"] = float("nan")
        self.assertEqual(self.assess([current], [decision()])["status"], "failed")

    def test_dispatch_is_scoped_and_restores_after_failure(self):
        from causal_memory_control import interactive_progress

        previous = interactive_progress.run_logged
        with mock.patch.object(interactive_progress, "run_logged", side_effect=RuntimeError("worker stopped")) as worker:
            with self.assertRaisesRegex(RuntimeError, "worker stopped"), resume.audited_workers():
                interactive_progress.run_logged(["python", "-u", "-m", "causal_memory_control.st_single_drop_evaluate"], "log", cwd=".")
            self.assertIs(interactive_progress.run_logged, worker)
            self.assertEqual(worker.call_args.args[0][3], "causal_memory_control.st_single_drop_audit_evaluate")
            with self.assertRaisesRegex(ValueError, "unrelated"), resume.audited_workers():
                interactive_progress.run_logged(["python", "-u", "-m", "other"], "log", cwd=".")
        self.assertIs(interactive_progress.run_logged, previous)


class ResumeIntegrationTests(unittest.TestCase):
    def test_resume_float_failure_partial_task_and_old_results_without_recalibration(self):
        for name in ("fever", "hotpotqa"):
            with self.subTest(dataset=name), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                fixtures = fixtures_module.ExperimentTests()
                fixtures.setup_stack(stack)
                adapter, old_args, tuning_argv, encoder_args, job = fixtures.setup_models(root, name)
                runner = fever_experiment if name == "fever" else hotpotqa_experiment
                if name == "fever":
                    for patch in fever_tests.FeverRunnerTests._patches():
                        stack.enter_context(patch)
                else:
                    hotpot_tests.HotpotRunnerTests.patches(stack)
                parent = _FeverSTMacNet if name == "fever" else hotpot_tests._HotpotMacNet
                stack.enter_context(mock.patch.object(runner, "GMemory", _STMemory))
                stack.enter_context(mock.patch.object(runner, "MacNet", parent))
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

                def execute(command, *args, **kwargs):
                    calls.append(command)
                    module = importlib.import_module(command[3])
                    result = module.main(command[4:])
                    target = Path(command[command.index("--output-dir") + 1])
                    if not (target / "llm_cache.sqlite").exists():
                        cache(target / "llm_cache.sqlite")
                    return result

                stack.enter_context(mock.patch("causal_memory_control.interactive_progress.run_logged", side_effect=execute))
                encoder_cli.run_qa(encoder_args, name, {"args": old_args, "adapter": adapter, "source_memory": old_args.memory_dir,
                    "old": old_args.output_dir / "semantic_st", "baseline": old_args.output_dir / "native_gmemory",
                    "cache": old_args.output_dir / "semantic_st/llm_cache.sqlite", "rule_reference": retest / "memory_gate.jsonl"}, job)
                jobs, manifest = tuning.prepare(tuning.parse_args(tuning_argv))
                save(root / "tuned/manifest.json", manifest)
                choice = original.reuse.read(root / "tuned/selection.json")
                choice["manifest_sha256"] = digest(manifest)
                save(root / "tuned/selection.json", choice)
                tuning.evaluate(tuning.parse_args(tuning_argv), jobs, manifest, choice)
                argv = fixtures.argv(root, name)
                original.main([*argv, "--stage", "calibrate"])
                frozen = {**source_bytes(root / "old"), **source_bytes(root / "source"),
                    **source_bytes(root / "tuned"), **source_bytes(root / "mpnet-old"), **source_bytes(root / "rules")}
                for path in (root / "decisions").rglob("*.json"):
                    frozen[path] = path.read_bytes()
                prediction = STUtilityEstimator.predict

                def drift(estimator, features):
                    before = prediction(estimator, features)
                    u = before.utility + 5e-8
                    return replace(before, utility=u, q_use=.5 + u / 2, q_drop=.5 - u / 2)

                with mock.patch.object(STUtilityEstimator, "predict", drift), mock.patch.object(STUtilityEstimator, "fit", side_effect=AssertionError("refit")), mock.patch.object(original, "dev_predictions", side_effect=AssertionError("recomputed dev")):
                    # Reproduce the exact-dictionary failure with unchanged
                    # retrieval, checkpoints and policy, without a real LLM.
                    with self.assertRaisesRegex(ValueError, "retrieval/predictions changed"):
                        original.main([*argv, "--stage", "evaluate"])
                    failure = [True]
                    scheduled = []

                    class Interrupted(parent):
                        def schedule(self, task):
                            scheduled.append(task["id"])
                            if failure[0] and len(scheduled) == 2:
                                failure[0] = False
                                raise RuntimeError("simulated interruption")
                            return super().schedule(task)

                    with mock.patch.object(runner, "MacNet", Interrupted):
                        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                            resume.main([*argv, "--stage", "evaluate"])
                        log = root / "decisions" / name / "original_minilm/semantic_st/memory_gate.jsonl"
                        prefix = log.read_bytes()
                        self.assertEqual(len(prefix.splitlines()), 1)
                        result = resume.main([*argv, "--stage", "evaluate"])
                        self.assertTrue(log.read_bytes().startswith(prefix))
                        self.assertEqual(len(scheduled), old_args.final_claims * 3 + 1)
                    count = len(calls)
                    resume.main([*argv, "--stage", "evaluate"])
                    self.assertEqual(len(calls), count)
                self.assertEqual(set(result[name]), set(original.VARIANTS))
                self.assertEqual(frozen, {p: p.read_bytes() for p in frozen})
                binding = original.reuse.read(root / "decisions/numeric_audit_resume.json")
                self.assertFalse(binding["dev_recomputed"])
                reports = [original.reuse.read(p) for p in (root / "decisions").glob(f"{name}/*/semantic_st/single_drop_audit/*.json")]
                self.assertTrue(reports)
                self.assertTrue(all(r["status"] == "tolerated_numeric_difference" for r in reports))
                for variant in original.VARIANTS:
                    rows = original.reuse.rows(root / "decisions" / name / variant / "semantic_st/memory_gate.jsonl")
                    self.assertEqual(len(rows), old_args.final_claims)
                    self.assertTrue(all(r["dropped_count"] <= 1 for r in rows))


if __name__ == "__main__":
    unittest.main()
