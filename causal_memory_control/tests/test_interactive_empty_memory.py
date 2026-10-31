"""Empty lists execute normally and never become fabricated gate supervision."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from causal_memory_control import interactive_empty_memory as empty
from causal_memory_control import interactive_team_u as team
from causal_memory_control import interactive_training as training
from causal_memory_control import interactive_worker as worker
from causal_memory_control.interactive_data import digest, save
from causal_memory_control.interactive_report import report
from causal_memory_control.interactive_runtime import audit
from causal_memory_control.interactive_protocol import learned_methods, policies
from causal_memory_control.tests.test_interactive_team_u import fixture, collect, Encoder
from causal_memory_control.tests.test_interactive_benchmark import measured


def empty_runner(calls, empty_ids=None):
    class Runner:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, task, seed, drops=(), expected=None):
            calls.append((task["id"], seed, tuple(drops)))
            row = measured(task["id"], seed, drops)
            if empty_ids is None or task["id"] in empty_ids:
                row["candidates"] = []
                row["retrieval_fingerprint"] = digest([])
                row["outcome"].update(success=float(seed % 2), raw_reward=float(seed % 2),
                                      progress=float(seed % 2), won=bool(seed % 2), done=bool(seed % 2))
            if expected:
                audit(row, expected, drops)
            return row

        def close(self):
            pass
    return Runner


class EmptyMemoryTests(unittest.TestCase):
    def test_empty_fold_collect_label_skip_evaluate_report_and_resume(self):
        for dataset in ("pddl", "alfworld"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory, \
                 redirect_stdout(io.StringIO()), \
                 mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder",
                            side_effect=AssertionError("empty lists must not load an encoder")):
                root, config, _, calls = fixture(directory, dataset)
                config["exclude_ce"] = True
                runner = empty_runner(calls)
                collect(config, dataset, runner)
                self.assertEqual(len(calls), 6)  # three tasks, two measured KEEP repeats
                rows = training.diagnostic_lists(root / dataset, config["experiments"][dataset])
                self.assertTrue(all(r["q_drop"] == r["branches"] == [] for r in rows))
                self.assertTrue(all(r["q_use"] == .5 and r["use_samples"] == [0, 1] for r in rows))
                team.labels(config, dataset)
                training.reward_train(config, dataset)
                team.train(config, dataset)
                self.assertFalse(list((root / dataset).rglob("checkpoint.json")))
                self.assertEqual(training.learned_gates(config, dataset),
                                 {method: None for method in learned_methods(config)})
                with mock.patch("causal_memory_control.interactive_runtime.NativeRunner", runner), \
                     mock.patch.object(worker, "snapshot", return_value=root / "unused"):
                    worker.evaluate(config, dataset)
                    summary = report(config)[dataset]
                    self.assertEqual(len(calls), 8)  # one shared episode per test task
                    self.assertEqual(summary["checkpoints_sha256"], {})
                    self.assertEqual(set(summary["untrained_methods"]), set(learned_methods(config)))
                    for method, _ in policies(config):
                        self.assertEqual(summary["policies"][method]["n"], 2)
                        self.assertEqual(summary["policies"][method]["empty_candidate_tasks"], 2)
                        self.assertEqual(summary["policies"][method]["paired_delta"], 0)
                    records = [json.loads(line) for line in (root / dataset / "evaluation/policies.jsonl").read_text().splitlines()]
                    for task_id in ("4", "5"):
                        selected = [r for r in records if r["task_id"] == task_id]
                        self.assertEqual(len({tuple(r["action_keys"]) for r in selected if r["policy"] != "random_drop"}), 1)
                    collect(config, dataset, runner)
                    team.labels(config, dataset)
                    training.reward_train(config, dataset)
                    team.train(config, dataset)
                    worker.evaluate(config, dataset)
                    self.assertEqual(len(calls), 8)

    def test_mixed_lists_preserve_empty_tasks_and_only_label_real_candidates(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root, config, _, calls = fixture(directory)
            config["exclude_ce"] = True
            collect(config, "pddl", empty_runner(calls, {"2"}))
            rows = training.diagnostic_lists(root / "pddl", config["experiments"]["pddl"])
            counts = empty.coverage(rows, config["experiments"]["pddl"])
            self.assertEqual(counts["fit"]["empty_task_ids"], ["2"])
            self.assertEqual(counts["fit"]["supervised_task_ids"], ["1"])
            self.assertEqual(len(calls), 18)
            with mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", return_value=Encoder()):
                team.labels(config, "pddl")
            labels = [json.loads(line) for line in (root / "pddl/local_global_team_u/team_u_labels.jsonl").read_text().splitlines()]
            self.assertEqual(next(r for r in labels if r["task_id"] == "2")["team_u"], [])
            self.assertEqual(sum(len(r["team_u"]) for r in labels), 6)
            with self.assertRaisesRegex(ValueError, "requires every"):
                empty.save_skip(config, "pddl", rows, ("semantic_linear",))

    def test_untrained_nonempty_inference_and_fabricated_empty_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "require a trained gate"):
            training.select(None, {"candidates": [{"memory_id": "x"}]})
        self.assertEqual(training.select(None, {"candidates": []}), ([], []))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root, config, _, calls = fixture(directory)
            config["exclude_ce"] = True
            collect(config, "pddl", empty_runner(calls))
            training.reward_train(config, "pddl")
            target = root / "pddl/diagnostic/tasks" / (digest("1") + ".json")
            row = json.loads(target.read_text())
            row["q_drop"] = [0]
            save(target, row)
            with self.assertRaisesRegex(ValueError, "misaligned"):
                empty.validated_skip(config, "pddl", "semantic_linear")

    def test_mixed_reward_training_preserves_registered_split_and_filters_examples(self):
        from causal_memory_control import estimator, neural_estimator, st_estimator, runtime_gate, train_gate

        class FeatureEncoder(Encoder):
            def config(self):
                return {"type": "test", "model": "tiny-dedup"}

        class Estimator:
            calls = []
            def __init__(self, **kwargs):
                pass
            def fit(self, rows, **kwargs):
                self.calls.append([row.group_id for row in rows])
                return True

        def checkpoint(path, _estimator, **kwargs):
            save(path, {"training_metadata": kwargs["training_metadata"]})

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root, config, _, calls = fixture(directory)
            config.update(exclude_ce=True, ensemble_size=2)
            collect(config, "pddl", empty_runner(calls, {"2"}))
            with mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", return_value=FeatureEncoder()), \
                 mock.patch.object(estimator, "AmortizedUtilityEstimator", Estimator), \
                 mock.patch.object(neural_estimator, "MLPUtilityEstimator", Estimator), \
                 mock.patch.object(st_estimator, "STUtilityEstimator", Estimator), \
                 mock.patch.object(runtime_gate, "save_gate_checkpoint", side_effect=checkpoint), \
                 mock.patch.object(train_gate, "_calibrate_policy", return_value={}):
                training.reward_train(config, "pddl")
            self.assertEqual(Estimator.calls, [["1"] * 3] * 3)
            for method in ("semantic_linear", "semantic_mlp", "semantic_st"):
                metadata = json.loads((root / "pddl" / method / "checkpoint.json").read_text())["training_metadata"]
                self.assertEqual(metadata["fit_task_ids"], ["1", "2"])
                self.assertEqual(metadata["candidate_coverage"]["fit"]["empty_task_ids"], ["2"])

    def test_no_supervision_cannot_be_invented_by_a_bad_status_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root, config, _, calls = fixture(directory)
            config["exclude_ce"] = True
            collect(config, "pddl", empty_runner(calls))
            training.reward_train(config, "pddl")
            target = root / "pddl/semantic_linear/training_status.json"
            status = json.loads(target.read_text())
            save(target, {**status, "trained": True})
            with self.assertRaisesRegex(ValueError, "source or configuration changed"):
                empty.validated_skip(config, "pddl", "semantic_linear")
            save(target, status)
            save(target.with_name("checkpoint.json"), {"fake": True})
            with self.assertRaisesRegex(ValueError, "cannot coexist"):
                empty.validated_skip(config, "pddl", "semantic_linear")

    def test_mixed_team_u_training_only_receives_candidates_and_preserves_split_metadata(self):
        from causal_memory_control import csm_st_training

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root, config, _, calls = fixture(directory)
            config["exclude_ce"] = True
            collect(config, "pddl", empty_runner(calls, {"2"}))
            with mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", return_value=Encoder()):
                team.labels(config, "pddl")

            def fit(rows, target, **kwargs):
                self.assertEqual([r["task_id"] for r in rows], ["1", "3"])
                self.assertEqual(kwargs["fit_ids"], {"1"})
                self.assertEqual(kwargs["dev_ids"], {"3"})
                (target / "dev_predictions.jsonl").write_text(json.dumps({
                    "id": "3", "ci": [0, 0, 0], "predicted_ci": [0, 0, 0]}) + "\n")
                return {"csm_st": {}, "training_metadata": {"policy_calibration": {}},
                        "training_diagnostics": {"loss_history": [{"dev_ci_sign_accuracy": .5}]}}

            module = SimpleNamespace(CSMConfig=lambda **kwargs: kwargs)
            with mock.patch.dict(sys.modules, {"causal_memory_control.csm_st_model": module}), \
                 mock.patch.object(csm_st_training, "train", side_effect=fit):
                team.train(config, "pddl")
            payload = json.loads((root / "pddl/local_global_team_u/checkpoint.json").read_text())
            self.assertEqual(payload["training_metadata"]["fit_task_ids"], ["1", "2"])
            self.assertEqual(payload["training_metadata"]["candidate_coverage"]["fit"]["supervised_task_ids"], ["1"])
            self.assertIn("local_global_team_u", payload)
            self.assertNotIn("csm_st", payload)

    def test_five_fold_report_keeps_every_test_task_with_an_empty_fold(self):
        from causal_memory_control.run_interactive_gate_benchmark import dataset_jobs
        from causal_memory_control.csm_st_data import write_rows

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root, config, _, _ = fixture(directory)
            config["exclude_ce"] = True
            data = config["experiments"]["pddl"]
            data["tasks"] = [{**data["tasks"][0], "id": str(i), "group": f"problem:{i}"} for i in range(10)]
            data.pop("split")
            data["cv_splits"] = [{"support": [str((fold * 2 + 2) % 10)],
                "fit": [str((fold * 2 + 3) % 10)], "dev": [str((fold * 2 + 4) % 10)],
                "test": [str(fold * 2), str(fold * 2 + 1)]} for fold in range(5)]
            for _, child in dataset_jobs(config, "pddl"):
                child_root = Path(child["output_dir"])
                save(child_root / "benchmark_manifest.json", child)
                save(child_root / "pddl/snapshot.json", {"test": True})
                if child["cv_context"]["fold"] == 2:
                    calls = []
                    runner = empty_runner(calls)
                    collect(child, "pddl", runner)
                    training.reward_train(child, "pddl")
                    team.labels(child, "pddl")
                    team.train(child, "pddl")
                    with mock.patch("causal_memory_control.interactive_runtime.NativeRunner", runner), \
                         mock.patch.object(worker, "snapshot", return_value=root / "unused"):
                        worker.evaluate(child, "pddl")
                else:
                    for method in learned_methods(child):
                        save(child_root / "pddl" / method / "checkpoint.json", {"mock": method})
                    records = []
                    for task_id in child["experiments"]["pddl"]["split"]["test"]:
                        baseline = measured(task_id)
                        baseline["action_key"] = "measured-" + task_id
                        methods = [m for m, _ in policies(child)] + [f"random_drop_seed{s}" for s in child["random_seeds"]]
                        records.extend(worker.policy_record(child, m, baseline, baseline, []) for m in methods)
                    write_rows(child_root / "pddl/evaluation/policies.jsonl", records)
            summary = report(config)["pddl"]
            self.assertEqual(summary["policies"]["always_use"]["n"], 10)
            self.assertEqual(summary["policies"]["semantic_st"]["empty_candidate_tasks"], 2)
            self.assertEqual(summary["policies"]["semantic_st"]["untrained_empty_only_tasks"], 2)
            self.assertEqual(set(summary["untrained_methods"]), {"pddl/fold_02"})
            self.assertEqual(report(config)["pddl"]["task_ids"], summary["task_ids"])


if __name__ == "__main__":
    unittest.main()
