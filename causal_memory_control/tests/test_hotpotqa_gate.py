from __future__ import annotations

import json
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import io
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from causal_diagnostic.hotpotqa_oracle_recipient.data import file_md5
from causal_diagnostic.hotpotqa_oracle_recipient.provenance import SNAPSHOT_SCHEMA, stable_hash
from causal_memory_control import hotpotqa_experiment as runner
from causal_memory_control import run_hotpotqa_gate_comparison as pipeline
from causal_memory_control import train_gate
from causal_memory_control.compare_gate_runs import main as compare_main
from causal_memory_control.estimator import AmortizedUtilityEstimator, PotentialOutcomeExample
from causal_memory_control.features import HashEmbedder, TeamExposureFeatureBuilder
from causal_memory_control.hotpotqa_experiment import _summary, validate_checkpoint_for_hotpotqa
from causal_memory_control.insight_metadata import load_insight_catalog
from causal_memory_control.jsonl import read_jsonl
from causal_memory_control.runtime_gate import CHECKPOINT_SCHEMA, GMemoryExposureGate
from causal_memory_control.tests.test_fever_experiment import (
    _FakeClient, _FakeMacNet, _FakeMemory, _FakeMessage,
)


class HotpotGateUtilityTests(unittest.TestCase):
    def test_summary_reports_f1_and_exact_match_separately(self) -> None:
        rows = [
            {"mode": "learned", "reward": 0.5, "done": False,
             "keep_rate": 0.75, "retrieved_count": 4, "dropped_count": 1,
             "outcome": {"answer_em": 0.0, "llm_calls": 2, "cache_hits": 1}},
            {"mode": "learned", "reward": 1.0, "done": True,
             "keep_rate": 1.0, "retrieved_count": 2, "dropped_count": 0,
             "outcome": {"answer_em": 1.0, "llm_calls": 1, "cache_hits": 3}},
        ]
        summary = _summary(rows)
        self.assertEqual(summary["mode"], "learned")
        self.assertAlmostEqual(summary["mean_answer_f1"], 0.75)
        self.assertAlmostEqual(summary["answer_exact_match_rate"], 0.5)
        self.assertEqual(summary["dropped_candidates"], 1)
        self.assertEqual(summary["llm_calls"], 3)

    def test_checkpoint_validation_rejects_train_test_overlap(self) -> None:
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "training_metadata": {
                "training_task_ids": [10, 11],
                "candidate_kinds": ["trajectory"],
                "training_contexts": [{"benchmark": "HotpotQA_distractor_offline"}],
            },
        }
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_checkpoint_for_hotpotqa(
                payload,
                evaluation_ids=[11, 12],
                expected_context={"benchmark": "HotpotQA_distractor_offline"},
                candidate_kinds=("trajectory",),
            )


class JsonlRecoveryTests(unittest.TestCase):
    def test_repair_only_discards_an_incomplete_final_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_bytes(b'{"task_id": 1}\n{"task_id":')
            rows = read_jsonl(path, repair_tail=True)
            self.assertEqual(rows, [{"task_id": 1}])
            self.assertTrue(list(path.parent.glob("rows.jsonl.partial-tail-*")))
            self.assertEqual(path.read_text(encoding="utf-8"), '{"task_id": 1}\n')

    def test_middle_corruption_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"task_id": 1}\nnot-json\n{"task_id": 3}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rows.jsonl:2"):
                read_jsonl(path, repair_tail=True)


class _HotpotMacNet(_FakeMacNet):
    fail_id = None

    def schedule(self, task):
        if task["id"] == type(self).fail_id:
            raise RuntimeError("simulated model timeout")
        type(self).schedule_count += 1
        nodes = (*self._agent_nodes.values(), self._decision_node)
        assert all(not node.history for node in nodes), "previous task state leaked"
        self.env.reset()
        self.memory.gate.filter_retrieval(
            ([_FakeMessage()], [], ["Read both supporting paragraphs."]),
            query_task=task["task_main"], task_state=task["task_description"],
        )
        # Partial credit is distinct from a completed episode and exact match.
        self.env.step("Finish[New York]")
        reward, done, _ = self.env.feedback()
        for node in nodes:
            node.history.append(task["id"])
        return reward, done


class HotpotRunnerTests(unittest.TestCase):
    @staticmethod
    def fixture(root):
        data = root / "hotpot.json"
        data.write_text(json.dumps([
            {"_id": f"h{i}", "question": f"Where is site {i}?", "answer": answer,
             "context": [["Place", ["The site is in New York City."]]],
             "supporting_facts": [["Place", 0]]}
            for i, answer in enumerate(("support", "training", "New York City", "New York"))
        ]), encoding="utf-8")
        memory = root / "g-memory"
        memory.mkdir()
        design = {
            "snapshot_schema": SNAPSHOT_SCHEMA, "benchmark": "HotpotQA_distractor_offline",
            "source_md5": file_md5(data), "model": "fake-model", "graph_type": "Chain",
            "node_num": 2, "embedding_model": "fake-embedding",
            "support_ids": [0], "evaluation_ids": [1, 2, 3],
        }
        (memory / "causal_snapshot_manifest.json").write_text(json.dumps({
            **design, "design": design, "design_hash": stable_hash(design),
            "memory_records": 1, "successful_total": 1, "failed_total": 0,
        }), encoding="utf-8")
        return data, memory

    @staticmethod
    def argv(data, memory, output):
        return ["--data", str(data), "--memory-dir", str(memory), "--output-dir", str(output),
                "--claims", "2", "--evaluation-offset", "1", "--model", "fake-model",
                "--node-num", "2", "--embedding-model", "fake-embedding",
                "--endpoint", "http://unused.invalid/v1", "--seed", "2000",
                "--temperature", "0.7", "--candidate-kinds", "trajectory,insight", "--kappa", "0"]

    @staticmethod
    def patches(stack):
        for name, value in {
            "_load_runtime_components": lambda: None,
            "SeededCachedChat": _FakeClient, "GMemory": _FakeMemory, "MacNet": _HotpotMacNet,
            "EmbeddingFunc": lambda _name: object(), "ReasoningIO": lambda **_: object(),
        }.items():
            stack.enter_context(mock.patch.object(runner, name, value))

    def test_interrupted_run_resumes_and_reports_partial_f1(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            data, memory = self.fixture(root)
            output = root / "run"
            argv = self.argv(data, memory, output)
            self.patches(stack)
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            _HotpotMacNet.schedule_count, _HotpotMacNet.fail_id = 0, 3
            with self.assertRaisesRegex(RuntimeError, "simulated model timeout"):
                runner.main(argv)
            progress = json.loads((output / "progress.json").read_text())
            self.assertEqual(progress["current_question_id"], 3)
            self.assertEqual(progress["completed_questions"], 1)
            self.assertIn("simulated model timeout", progress["error"]["traceback"])
            log = output / "memory_gate.jsonl"
            with log.open("ab") as handle:
                handle.write(b'{"interrupted":')
            _HotpotMacNet.fail_id = None
            runner.main([*argv, "--resume"])
            rows = read_jsonl(log)
            self.assertEqual([row["task_id"] for row in rows], [2, 3])
            self.assertEqual(_HotpotMacNet.schedule_count, 2)
            self.assertAlmostEqual(rows[0]["reward"], 0.8)
            self.assertFalse(rows[0]["done"])
            self.assertTrue(rows[0]["outcome"]["episode_done"])
            self.assertTrue(rows[1]["done"])
            runner.main([*argv, "--resume"])
            self.assertEqual(_HotpotMacNet.schedule_count, 2)
            with self.assertRaisesRegex(SystemExit, "incompatible"):
                runner.main([*argv, "--temperature", "0.4", "--resume"])

    def test_completed_artifact_validation_checks_tasks_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            data, memory = self.fixture(root)
            output = root / "run"
            self.patches(stack)
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            _HotpotMacNet.fail_id = None
            runner.main(self.argv(data, memory, output))
            args = pipeline.parse_args([])
            args.data, args.memory_dir, args.output_dir = data, memory, root
            args.training_claims, args.evaluation_count, args.final_claims = 1, 3, 2
            args.model, args.embedding_model, args.node_num = "fake-model", "fake-embedding", 2
            args.endpoint = "http://unused.invalid/v1"
            self.assertTrue(pipeline._validated_evaluation_complete(args, output, mode="always_keep"))
            args.temperature = 0.4
            with self.assertRaisesRegex(RuntimeError, "temperature"):
                pipeline._validated_evaluation_complete(args, output, mode="always_keep")


class EstimatorImprovementTests(unittest.TestCase):
    def test_hybrid_ranking_fits_signs_and_serializes(self):
        examples = [
            PotentialOutcomeExample({"x": x}, q_use=0.5 + 0.2*x, q_drop=0.5 - 0.2*x,
                                    event_id=f"{i}-{x}", group_id=str(i))
            for i in range(12) for x in (-1.0, 1.0)
        ]
        estimator = AmortizedUtilityEstimator(
            ensemble_size=4, min_samples=8, solver="ridge", bootstrap_unit="task",
            direct_utility_weight=0.5, contrastive_weight=0.1, residual_noise_scale=0,
        )
        self.assertTrue(estimator.fit(examples))
        self.assertGreater(estimator.contrastive_pairs, 0)
        self.assertGreater(estimator.predict({"x": 1}).utility, 0.3)
        self.assertLess(estimator.predict({"x": -1}).utility, -0.3)
        restored = AmortizedUtilityEstimator.from_dict(estimator.to_dict())
        self.assertTrue(math.isfinite(restored.predict({"x": 0.5}).uncertainty))
        self.assertAlmostEqual(restored.predict({"x": 0.5}).utility,
                               estimator.predict({"x": 0.5}).utility)

    def test_calibration_never_crosses_task_groups(self):
        examples = [PotentialOutcomeExample({"x": float(j)}, 0.5, 0.5,
                                            event_id=f"{i}-{j}", group_id=str(i))
                    for i in range(10) for j in range(3)]
        fit, calibration = train_gate._split_calibration_examples(examples, fraction=0.2, seed=71)
        self.assertFalse({e.group_id for e in fit} & {e.group_id for e in calibration})
        self.assertEqual(len(calibration), 6)

    def test_calibration_uses_keep_all_when_every_selected_drop_harms(self):
        estimator = SimpleNamespace(predict=lambda _: SimpleNamespace(
            utility=-0.2, uncertainty=0.1, calibrated=True))
        harmful = [PotentialOutcomeExample({}, 1.0, 0.0, group_id=str(i)) for i in range(5)]
        result = train_gate._calibrate_policy(estimator, harmful, kappas=[0, 0.25, 1.96], max_drops=1, delta=0)
        self.assertEqual(result["selected"]["policy"], "always_keep")
        self.assertEqual(result["recommended_max_drops"], 0)
        beneficial = [PotentialOutcomeExample({}, 0.0, 1.0, group_id=str(i)) for i in range(5)]
        result = train_gate._calibrate_policy(estimator, beneficial, kappas=[0, 0.25], max_drops=1, delta=0)
        self.assertEqual(result["recommended_kappa"], 0.25)
        self.assertEqual(result["recommended_max_drops"], 1)

    def test_training_and_runtime_v2_features_agree_without_gold_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "insights.json").write_text(json.dumps([{
                "rule": "Read both paragraphs.", "score": 4,
                "positive_correlation_tasks": [0, 1, 2], "negative_correlation_tasks": [3],
            }]))
            gate = GMemoryExposureGate(mode="always_keep", memory_dir=root,
                                       feature_builder=TeamExposureFeatureBuilder(HashEmbedder(32), version=2))
            task = {"task_main": "HotpotQA question: Where?", "task_description": "Context: New York",
                    "answer": "hidden", "label": "hidden", "gold_answer": "hidden", "id": 9}
            gate.begin_task(9, task)
            gate.set_exposure_agent_ids(["a", "b", "decision"])
            adapted = gate.adapter.adapt(([], [], ["Read both paragraphs.", "Compare the dates."]))
            event = gate._build_event(adapted, adapted.candidates[0], index=0,
                                      query_task=task["task_main"], task_state=task["task_description"])
            payloads = []
            catalog = load_insight_catalog(root)
            for index, candidate in enumerate(adapted.candidates):
                payloads.append({"kind": "insight", "index": index, "candidate_id": candidate.memory_id,
                                 "text": candidate.content, **catalog.get(candidate.content, {})})
            row = {"task_id": 9, "candidate": payloads[0], "candidate_set": payloads,
                   "task": task, "retrieval_sizes": {"insights": 2}, "exposure_count": 3,
                   "run_hash": "run", "sample_seed": 0, "repeat_index": 0}
            examples = train_gate._examples_from_branches(
                [{**row, "condition": "use_all", "outcome": {"team_score": 0.8}},
                 {**row, "condition": "global_drop", "outcome": {"team_score": 0.6}}],
                metric="team_score", task_ids=None, hash_dimensions=32, feature_version=2,
            )
            runtime_features = gate.feature_builder.build(event)
            self.assertEqual(runtime_features, examples[0].features)
            self.assertEqual(runtime_features["insight_positive_task_count"], 3)
            self.assertNotIn("gold_answer", gate._event_task_metadata())
            task["answer"], task["label"] = "different gold", "different gold"
            gate.begin_task(9, task)
            changed = gate._build_event(adapted, adapted.candidates[0], index=0,
                                        query_task=task["task_main"], task_state=task["task_description"])
            self.assertEqual(runtime_features, gate.feature_builder.build(changed))

    def test_legacy_estimator_checkpoint_preserves_predictions(self):
        estimator = AmortizedUtilityEstimator(ensemble_size=2, min_samples=4, epochs=50)
        estimator.fit([PotentialOutcomeExample({"x": float(i)}, 0.8, 0.2) for i in range(8)])
        payload = estimator.to_dict()
        payload["schema"] = "amortized-utility-estimator-v1"
        for key in ("direct_utility_weight", "contrastive_weight", "contrastive_margin",
                    "contrastive_min_gap", "contrastive_epochs", "solver", "bootstrap_unit"):
            payload["config"].pop(key)
        for member in payload["models"]:
            member.pop("direct")
        restored = AmortizedUtilityEstimator.from_dict(payload)
        self.assertAlmostEqual(restored.predict({"x": 2.5}).utility,
                               estimator.predict({"x": 2.5}).utility)


class HotpotPipelineTests(unittest.TestCase):
    def test_plan_and_child_arguments_cover_all_candidates_and_held_out_test(self):
        args = pipeline.parse_args([])
        args.training_claims, args.final_claims, args.valid_source_questions = 400, 100, 7405
        plan = pipeline._plan(args)
        self.assertEqual(plan["diagnostic_branches_two_seeds"], 19200)
        diagnostic_args = pipeline.diagnostic.parse_args(pipeline._diagnostic_command(args)[4:])
        self.assertTrue(diagnostic_args.all_candidates)
        self.assertTrue(diagnostic_args.gate_training_only)
        self.assertEqual(diagnostic_args.claims, 400)
        child = runner.parse_args(pipeline._evaluation_command(
            args, mode="always_keep", output=Path("baseline"), resume=True)[4:])
        self.assertEqual((child.evaluation_offset, child.claims), (400, 100))
        self.assertTrue(child.resume)

    def test_invalid_existing_checkpoint_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            args = pipeline.parse_args([])
            args.checkpoint = Path(directory) / "gate.json"
            self.assertFalse(pipeline._checkpoint_complete(args))
            args.checkpoint.write_text("broken checkpoint")
            with self.assertRaisesRegex(RuntimeError, "unreadable"):
                pipeline._checkpoint_complete(args)

    def test_pipeline_records_child_failure_with_stage_and_log_tail(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            data, _ = HotpotRunnerTests.fixture(root)
            output = root / "output"
            stack.enter_context(mock.patch.object(pipeline, "load_hotpotqa", return_value=[{}] * 11))
            stack.enter_context(mock.patch.object(pipeline, "_status", return_value={stage: False for stage in pipeline.STAGES}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))

            def fail(stage, command, log_path, **_):
                raise pipeline.StageFailure(stage, 7, command, log_path, "specific model-server failure")

            stack.enter_context(mock.patch.object(pipeline, "_run_command", side_effect=fail))
            with self.assertRaises(pipeline.StageFailure):
                pipeline.main(["--data", str(data), "--support-count", "1", "--evaluation-count", "10",
                               "--memory-dir", str(root / "memory"), "--diagnostic-results", str(root / "diagnostic"),
                               "--checkpoint", str(root / "gate.json"), "--output-dir", str(output)])
            progress = json.loads((output / "pipeline_progress.json").read_text())
            self.assertEqual(progress["error"]["stage"], "diagnostic")
            self.assertEqual(progress["error"]["exit_code"], 7)
            self.assertEqual(progress["error"]["log_tail"], "specific model-server failure")

    def test_auto_policy_can_keep_everything(self):
        with tempfile.TemporaryDirectory() as directory:
            args = pipeline.parse_args([])
            args.training_claims, args.final_claims = 400, 100
            args.checkpoint = Path(directory) / "gate.json"
            args.checkpoint.write_text(json.dumps({"training_metadata": {"policy_calibration": {
                "recommended_kappa": 1.96, "recommended_max_drops": 0,
            }}}))
            child = runner.parse_args(pipeline._evaluation_command(
                args, mode="learned", output=Path("gate"), resume=False)[4:])
            self.assertEqual(child.max_drops, 0)

    def test_no_drop_difference_saves_report_then_rejects_comparison(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            baseline, learned, report = root / "base.jsonl", root / "gate.jsonl", root / "comparison.json"
            row = {"task_id": 1, "reward": 1.0, "done": True, "dropped_count": 0,
                   "run_metadata": {"benchmark": "HotpotQA_distractor_offline"},
                   "retrieval_fingerprint": "same", "outcome": {"prediction": "New York"}}
            baseline.write_text(json.dumps(row) + "\n")
            learned.write_text(json.dumps({**row, "reward": 0.5, "done": False}) + "\n")
            with self.assertRaisesRegex(SystemExit, "no-DROP"):
                compare_main(["--baseline", str(baseline), "--gate", str(learned),
                              "--bootstrap-samples", "10", "--output", str(report), "--require-no-drop-match"])
            payload = json.loads(report.read_text())
            self.assertFalse(payload["paired_control_valid"])
            self.assertEqual(payload["no_drop_mismatches"][0]["task_id"], "1")

    def test_changed_retrieval_invalidates_comparison_even_with_drop(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            baseline, learned, report = root / "base.jsonl", root / "gate.jsonl", root / "comparison.json"
            row = {"task_id": 1, "reward": 1.0, "done": True,
                   "retrieval_fingerprint": "original", "run_metadata": {}}
            baseline.write_text(json.dumps(row) + "\n")
            learned.write_text(json.dumps({**row, "dropped_count": 1, "retrieval_fingerprint": "changed"}) + "\n")
            with self.assertRaises(SystemExit):
                compare_main(["--baseline", str(baseline), "--gate", str(learned),
                              "--bootstrap-samples", "10", "--output", str(report), "--require-no-drop-match"])
            payload = json.loads(report.read_text())
            self.assertEqual(payload["no_drop_mismatches"], [])
            self.assertEqual(payload["retrieval_mismatches"], ["1"])
            self.assertFalse(payload["paired_control_valid"])


if __name__ == "__main__":
    unittest.main()
