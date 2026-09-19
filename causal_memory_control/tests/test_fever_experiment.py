from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from causal_diagnostic.fever_oracle_recipient.data import file_md5
from causal_diagnostic.fever_oracle_recipient.provenance import (
    SNAPSHOT_SCHEMA,
    stable_hash,
)
from causal_memory_control import fever_experiment
from causal_memory_control.estimator import (
    AmortizedUtilityEstimator,
    PotentialOutcomeExample,
)
from causal_memory_control.runtime_gate import CHECKPOINT_SCHEMA, save_gate_checkpoint


class CheckpointValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = {
            "benchmark": "FEVER_binary_offline",
            "source_md5": "abc",
            "model": "model-a",
        }
        self.payload = {
            "schema": CHECKPOINT_SCHEMA,
            "training_metadata": {
                "training_task_ids": [1, 2],
                "candidate_kinds": ["trajectory"],
                "candidate_ranks": [1],
                "training_contexts": [dict(self.context)],
            },
        }

    def test_disjoint_compatible_checkpoint_is_accepted(self) -> None:
        fever_experiment.validate_checkpoint_for_fever(
            self.payload,
            evaluation_ids=[3, 4],
            expected_context=self.context,
            candidate_kinds=("trajectory",),
        )

    def test_training_evaluation_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "leakage"):
            fever_experiment.validate_checkpoint_for_fever(
                self.payload,
                evaluation_ids=[2, 3],
                expected_context=self.context,
                candidate_kinds=("trajectory",),
            )

    def test_incompatible_training_context_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "model"):
            fever_experiment.validate_checkpoint_for_fever(
                self.payload,
                evaluation_ids=[3],
                expected_context={**self.context, "model": "model-b"},
                candidate_kinds=("trajectory",),
            )

    def test_physical_chroma_hash_drift_is_not_a_semantic_mismatch(self) -> None:
        payload = json.loads(json.dumps(self.payload))
        payload["training_metadata"]["training_contexts"][0][
            "memory_sha256"
        ] = "hash-before-opening-chroma"
        fever_experiment.validate_checkpoint_for_fever(
            payload,
            evaluation_ids=[3, 4],
            expected_context={
                **self.context,
                "memory_sha256": "hash-after-opening-chroma",
            },
            candidate_kinds=("trajectory",),
        )

    def test_missing_training_ids_is_rejected(self) -> None:
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "training_metadata": {"candidate_kinds": ["trajectory"]},
        }
        with self.assertRaisesRegex(ValueError, "training_task_ids"):
            fever_experiment.validate_checkpoint_for_fever(
                payload,
                evaluation_ids=[3],
                expected_context=self.context,
                candidate_kinds=("trajectory",),
            )


class _FakeMessage:
    def __init__(self) -> None:
        self.task_main = "support task"
        self.task_description = "support description"
        self.task_trajectory = "Finish[SUPPORTS]"
        self.label = True
        self.extra_fields = {}


class _FakeAgent:
    def add_task_instruction(self, _instruction: str) -> str:
        return "ok"


class _FakeNode:
    def __init__(self, node_id: str) -> None:
        self.id = node_id
        self.reasoning_config = SimpleNamespace(temperature=0.0)
        self._agent = _FakeAgent()


class _FakeClient:
    def __init__(self, *_args, **_kwargs) -> None:
        self.calls = 0
        self.cache_hits = 0

    def close(self) -> None:
        pass


class _FakeMemory:
    def __init__(self, *_args, **_kwargs) -> None:
        self.memory_size = 1
        self.gate = None

    def set_retrieval_gate(self, gate) -> None:
        self.gate = gate


class _FakeMacNet:
    schedule_count = 0
    last_instance = None

    def build_system(self, _reasoning, memory, env, config) -> None:
        type(self).last_instance = self
        self.memory = memory
        self.env = env
        self._agent_nodes = {
            index: _FakeNode(f"solver_{index}")
            for index in range(int(config["node_num"]))
        }
        self._decision_node = _FakeNode("dicision")
        self.agents_team = {
            node.id: node._agent for node in self._agent_nodes.values()
        }

    def schedule(self, task) -> tuple[float, bool]:
        type(self).schedule_count += 1
        self.env.reset()
        self.memory.gate.filter_retrieval(
            ([_FakeMessage()], [], []),
            query_task=task["task_main"],
            task_state=task["task_description"],
        )
        self.env.step(f"Finish[{task['label']}]")
        reward, done, _ = self.env.feedback()
        return reward, done


class FeverRunnerTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        data = root / "fever.jsonl"
        rows = (
            {"id": 1, "claim": "support", "label": "SUPPORTS"},
            {"id": 2, "claim": "evaluation one", "label": "SUPPORTS"},
            {"id": 3, "claim": "evaluation two", "label": "REFUTES"},
        )
        data.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        memory = root / "g-memory"
        memory.mkdir()
        design = {
            "snapshot_schema": SNAPSHOT_SCHEMA,
            "benchmark": "FEVER_binary_offline",
            "source_md5": file_md5(data),
            "model": "fake-model",
            "graph_type": "Chain",
            "node_num": 2,
            "embedding_model": "fake-embedding",
            "support_ids": [1],
            "evaluation_ids": [2, 3],
        }
        manifest = {
            **design,
            "design": design,
            "design_hash": stable_hash(design),
            "memory_records": 1,
            "successful_total": 1,
            "failed_total": 0,
        }
        (memory / "causal_snapshot_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return data, memory

    @staticmethod
    def _patches():
        return (
            mock.patch.object(fever_experiment, "SeededCachedChat", _FakeClient),
            mock.patch.object(fever_experiment, "GMemory", _FakeMemory),
            mock.patch.object(fever_experiment, "MacNet", _FakeMacNet),
            mock.patch.object(
                fever_experiment, "EmbeddingFunc", lambda _name: object()
            ),
            mock.patch.object(
                fever_experiment,
                "ReasoningIO",
                lambda **_kwargs: object(),
            ),
        )

    def test_offline_runner_logs_predictions_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, memory = self._fixture(root)
            output = root / "run"
            argv = (
                "--data",
                str(data),
                "--memory-dir",
                str(memory),
                "--output-dir",
                str(output),
                "--claims",
                "2",
                "--model",
                "fake-model",
                "--node-num",
                "2",
                "--embedding-model",
                "fake-embedding",
                "--endpoint",
                "http://unused.invalid/v1",
            )
            _FakeMacNet.schedule_count = 0
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                log_path = fever_experiment.main(argv)
            rows = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(_FakeMacNet.schedule_count, 2)
            self.assertEqual([row["task_id"] for row in rows], [2, 3])
            self.assertEqual(rows[0]["outcome"]["prediction"], "SUPPORTS")
            self.assertEqual(rows[1]["outcome"]["prediction"], "REFUTES")
            self.assertEqual(rows[0]["exposure_count"], 3)
            self.assertEqual(rows[0]["kept_count"], 1)
            configured_nodes = (
                *_FakeMacNet.last_instance._agent_nodes.values(),
                _FakeMacNet.last_instance._decision_node,
            )
            self.assertTrue(
                all(node.reasoning_config.stop_strs is None for node in configured_nodes)
            )

            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                resumed = fever_experiment.main((*argv, "--resume"))
            self.assertEqual(resumed, log_path)
            self.assertEqual(_FakeMacNet.schedule_count, 2)
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["completion_rate"], 1.0)

            # Opening persistent Chroma can rewrite SQLite bytes without
            # changing any logical record.  Such drift must not break resume.
            (memory / "chroma-byte-drift.sqlite3").write_bytes(b"physical drift")
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                resumed_after_drift = fever_experiment.main((*argv, "--resume"))
            self.assertEqual(resumed_after_drift, log_path)
            self.assertEqual(_FakeMacNet.schedule_count, 2)

    def test_learned_runner_loads_checkpoint_and_drops_in_scope_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, memory = self._fixture(root)
            estimator = AmortizedUtilityEstimator(
                ensemble_size=2,
                min_samples=2,
                epochs=4,
                include_residual_noise=False,
            )
            examples = [
                PotentialOutcomeExample(
                    features={"constant": 1.0},
                    q_use=0.0,
                    q_drop=1.0,
                    event_id=f"event-{index}",
                )
                for index in range(4)
            ]
            self.assertTrue(estimator.fit(examples))
            manifest = json.loads(
                (memory / "causal_snapshot_manifest.json").read_text(encoding="utf-8")
            )
            context = {
                "benchmark": "FEVER_binary_offline",
                "source_md5": file_md5(data),
                "snapshot_design_hash": manifest["design_hash"],
                "memory_sha256": fever_experiment._directory_hash(memory),
                "model": "fake-model",
                "graph_type": "Chain",
                "node_num": 2,
                "embedding_model": "fake-embedding",
                "successful_topk": 3,
                "failed_topk": 0,
                "insights_topk": 3,
                "threshold": 0.0,
            }
            checkpoint = root / "checkpoint.json"
            save_gate_checkpoint(
                checkpoint,
                estimator,
                training_metadata={
                    "training_task_ids": [1],
                    "candidate_kinds": ["trajectory"],
                    "candidate_ranks": [1],
                    "training_contexts": [context],
                },
            )
            # Reproduce the production failure: checkpoint provenance was
            # captured before a read-only Chroma open changed physical bytes.
            (memory / "post-training-chroma-drift.sqlite3").write_bytes(b"drift")
            output = root / "learned"
            argv = (
                "--data",
                str(data),
                "--memory-dir",
                str(memory),
                "--output-dir",
                str(output),
                "--claims",
                "2",
                "--model",
                "fake-model",
                "--node-num",
                "2",
                "--embedding-model",
                "fake-embedding",
                "--endpoint",
                "http://unused.invalid/v1",
                "--mode",
                "learned",
                "--checkpoint",
                str(checkpoint),
                "--kappa",
                "0",
            )
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                log_path = fever_experiment.main(argv)
            rows = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["dropped_count"] for row in rows], [1, 1])
            self.assertTrue(
                all(
                    row["decisions"][0]["reason"] == "confidently_harmful"
                    for row in rows
                )
            )


if __name__ == "__main__":
    unittest.main()
