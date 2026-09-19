from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from causal_memory_control import (
    AmortizedUtilityEstimator,
    GMemoryExposureGate,
    PotentialOutcomeExample,
    save_gate_checkpoint,
)
from causal_memory_control.train_gate import load_examples, main as train_main
from causal_memory_control.compare_gate_runs import compare


class FakeMessage:
    def __init__(self, name: str):
        self.task_main = name
        self.task_description = f"description {name}"
        self.task_trajectory = f"trajectory {name}"
        self.label = True
        self.extra_fields = {"key_steps": f"steps {name}"}


def negative_estimator() -> AmortizedUtilityEstimator:
    examples = [
        PotentialOutcomeExample(
            {
                "sim_task_memory": 0.0,
                "candidate_count": 2.0,
            },
            q_use=0.0,
            q_drop=1.0,
            event_id=f"event-{index}",
        )
        for index in range(12)
    ]
    estimator = AmortizedUtilityEstimator(
        ensemble_size=4,
        min_samples=4,
        epochs=20,
        include_residual_noise=False,
    )
    assert estimator.fit(examples)
    return estimator


class EstimatorCheckpointTests(unittest.TestCase):
    def test_round_trip_preserves_prediction(self) -> None:
        estimator = negative_estimator()
        before = estimator.predict({"sim_task_memory": 0.0, "candidate_count": 2.0})
        restored = AmortizedUtilityEstimator.from_dict(estimator.to_dict())
        after = restored.predict({"sim_task_memory": 0.0, "candidate_count": 2.0})
        self.assertAlmostEqual(before.utility, after.utility)
        self.assertAlmostEqual(before.uncertainty, after.uncertainty)

    def test_training_cli_writes_loadable_gate_checkpoint(self) -> None:
        rows = [
            {
                "event_id": f"event-{index}",
                "task_id": index,
                "features": {"x": float(index)},
                "q_use": float(index % 2),
                "q_drop": float((index + 1) % 2),
            }
            for index in range(4)
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "examples.jsonl"
            checkpoint = Path(directory) / "gate.json"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            train_main(
                (
                    "--input",
                    str(source),
                    "--output",
                    str(checkpoint),
                    "--ensemble-size",
                    "2",
                    "--min-samples",
                    "2",
                    "--epochs",
                    "4",
                )
            )
            gate = GMemoryExposureGate.from_checkpoint(checkpoint, mode="learned")
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        self.assertTrue(gate.estimator.fitted)
        self.assertEqual(gate.estimator.training_samples, 4)
        self.assertEqual(
            payload["training_metadata"]["training_task_ids"], [0, 1, 2, 3]
        )


class RuntimeGateTests(unittest.TestCase):
    def test_always_keep_is_exact_identity(self) -> None:
        result = ([FakeMessage("one")], [], ["rule one"])
        gate = GMemoryExposureGate(mode="always_keep")
        filtered = gate.filter_retrieval(
            result, query_task="query", task_state="state"
        )
        self.assertEqual(filtered, result)

    def test_always_drop_removes_exposed_candidates_but_not_failed_pool(self) -> None:
        failed = FakeMessage("failed")
        gate = GMemoryExposureGate(mode="always_drop")
        filtered = gate.filter_retrieval(
            ([FakeMessage("one")], [failed], ["rule one"]),
            query_task="query",
            task_state="state",
        )
        self.assertEqual(filtered, ([], [failed], []))

    def test_untrained_retrieval_rank_defaults_to_keep(self) -> None:
        first, second = FakeMessage("one"), FakeMessage("two")
        gate = GMemoryExposureGate(
            mode="always_drop", candidate_kinds=("trajectory",), candidate_ranks=(1,)
        )
        filtered = gate.filter_retrieval(
            ([first, second], [], []), query_task="query", task_state="state"
        )
        self.assertEqual(filtered, ([second], [], []))
        self.assertEqual(
            gate._task_decisions[1].reason, "candidate_rank_out_of_scope"
        )

    def test_learned_gate_drops_only_one_candidate_by_default(self) -> None:
        gate = GMemoryExposureGate(
            mode="learned",
            estimator=negative_estimator(),
            kappa=0.0,
        )
        filtered = gate.filter_retrieval(
            ([FakeMessage("one")], [], ["rule one"]),
            query_task="query",
            task_state="state",
        )
        self.assertEqual(len(filtered[0]) + len(filtered[2]), 1)

    def test_task_log_contains_decisions_and_final_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.jsonl"
            gate = GMemoryExposureGate(mode="always_keep", log_path=path)
            gate.begin_task(7, {"task_main": "query"})
            gate.filter_retrieval(
                ([FakeMessage("one")], [], []),
                query_task="query",
                task_state="state",
            )
            row = gate.end_task(
                1.0, True, outcome_metadata={"prediction": "SUPPORTS"}
            )
            persisted = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(persisted, row)
            self.assertEqual(row["keep_rate"], 1.0)
            self.assertTrue(row["done"])
            self.assertEqual(row["outcome"]["prediction"], "SUPPORTS")

            resumed = GMemoryExposureGate(
                mode="always_keep", log_path=path, resume=True
            )
            self.assertTrue(resumed.task_is_complete(7))

    def test_existing_log_requires_explicit_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                GMemoryExposureGate(mode="always_keep", log_path=path)

    def test_full_gate_checkpoint_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            save_gate_checkpoint(path, negative_estimator(), hash_dimensions=32)
            gate = GMemoryExposureGate.from_checkpoint(
                path, mode="learned", kappa=0.0
            )
            self.assertEqual(gate.feature_builder.embedder.dimensions, 32)

    def test_checkpoint_limits_runtime_scope_to_trained_memory_kinds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            save_gate_checkpoint(
                path,
                negative_estimator(),
                training_metadata={"candidate_kinds": ["trajectory"]},
            )
            gate = GMemoryExposureGate.from_checkpoint(
                path,
                mode="learned",
                kappa=0.0,
                candidate_kinds=("trajectory", "insight"),
            )
            filtered = gate.filter_retrieval(
                ([], [], ["unseen insight"]),
                query_task="query",
                task_state="state",
            )
        self.assertEqual(filtered[2], ["unseen insight"])

    def test_checkpoint_limits_runtime_scope_to_trained_retrieval_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            save_gate_checkpoint(
                path,
                negative_estimator(),
                training_metadata={
                    "candidate_kinds": ["trajectory"],
                    "candidate_ranks": [1],
                },
            )
            gate = GMemoryExposureGate.from_checkpoint(
                path, mode="learned", kappa=0.0
            )
            gate.filter_retrieval(
                ([FakeMessage("one"), FakeMessage("two")], [], []),
                query_task="query",
                task_state="state",
            )
        self.assertEqual(
            gate._task_decisions[1].reason, "candidate_rank_out_of_scope"
        )


class BranchLoaderTests(unittest.TestCase):
    def test_matched_repeats_become_expected_outcomes(self) -> None:
        rows = []
        for repeat, (use, drop) in enumerate(((1.0, 0.0), (0.0, 0.0))):
            for condition, score in (("use_all", use), ("global_drop", drop)):
                rows.append(
                    {
                        "task_id": 3,
                        "task": {"task_main": "move the ball", "game_name": "gripper"},
                        "candidate": {
                            "candidate_id": "trajectory-a",
                            "kind": "trajectory",
                            "index": 0,
                            "task_main": "old task",
                            "trajectory": "old steps",
                        },
                        "repeat_index": repeat,
                        "condition": condition,
                        "retrieval_sizes": {
                            "successful_trajectories": 3,
                            "insights": 1,
                        },
                        "outcome": {"success": score},
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "seed0.jsonl"
            second = Path(directory) / "seed1000.jsonl"
            first.write_text(
                "".join(json.dumps(row) + "\n" for row in rows[:2]), encoding="utf-8"
            )
            second.write_text(
                "".join(json.dumps(row) + "\n" for row in rows[2:]), encoding="utf-8"
            )
            examples = load_examples(
                (first, second), metric="success", task_ids=None, hash_dimensions=32
            )
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].q_use, 0.5)
        self.assertEqual(examples[0].q_drop, 0.0)
        self.assertEqual(examples[0].features["candidate_count"], 4.0)
        self.assertEqual(examples[0].features["retrieval_rank"], 1.0)

    def test_unmatched_use_drop_repeats_are_rejected(self) -> None:
        rows = [
            {
                "task_id": 3,
                "task": {"task_main": "move the ball"},
                "candidate": {
                    "candidate_id": "trajectory-a",
                    "kind": "trajectory",
                    "index": 0,
                },
                "repeat_index": repeat,
                "sample_seed": repeat,
                "condition": condition,
                "outcome": {"success": 1.0},
            }
            for repeat, condition in ((0, "use_all"), (1, "global_drop"))
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "branches.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "not matched"):
                load_examples(
                    path, metric="success", task_ids=None, hash_dimensions=32
                )


class ComparisonTests(unittest.TestCase):
    def test_final_outcomes_are_compared_by_paired_task_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.jsonl"
            gate = Path(directory) / "gate.jsonl"
            baseline.write_text(
                "\n".join(
                    json.dumps({"task_id": task_id, "reward": 0, "done": False})
                    for task_id in (1, 2)
                )
                + "\n",
                encoding="utf-8",
            )
            gate.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "reward": task_id - 1,
                            "done": task_id == 2,
                            "keep_rate": 0.5,
                        }
                    )
                    for task_id in (1, 2)
                )
                + "\n",
                encoding="utf-8",
            )
            result = compare(baseline, gate, bootstrap_samples=100, seed=1)
        self.assertEqual(result["paired_tasks"], 2)
        self.assertEqual(result["completion_rate_delta"], 0.5)
        self.assertEqual(result["gate_mean_keep_rate"], 0.5)

    def test_mismatched_run_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.jsonl"
            gate = Path(directory) / "gate.jsonl"
            baseline.write_text(
                json.dumps(
                    {
                        "task_id": 1,
                        "reward": 0,
                        "done": False,
                        "run_metadata": {"model": "a"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            gate.write_text(
                json.dumps(
                    {
                        "task_id": 1,
                        "reward": 1,
                        "done": True,
                        "run_metadata": {"model": "b"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "run metadata differs"):
                compare(baseline, gate, bootstrap_samples=10)

    def test_incomplete_pairing_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.jsonl"
            gate = Path(directory) / "gate.jsonl"
            baseline.write_text(
                "".join(
                    json.dumps({"task_id": task_id, "reward": 1, "done": True})
                    + "\n"
                    for task_id in (1, 2)
                ),
                encoding="utf-8",
            )
            gate.write_text(
                json.dumps({"task_id": 1, "reward": 1, "done": True}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "different task IDs"):
                compare(baseline, gate, bootstrap_samples=10)


if __name__ == "__main__":
    unittest.main()
