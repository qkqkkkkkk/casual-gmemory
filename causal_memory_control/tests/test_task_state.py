from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from causal_memory_control.task_state import (
    TASK_STATE_POLICY, archive_legacy_evaluations,
    reset_task_execution_state, validate_task_state_policy,
)


class _PromptAgent:
    def __init__(self, name):
        self.name, self.profile = name, "solver"
        self.prompts = []

    def response(self, prompt, config):
        self.prompts.append(prompt)
        return "Finish[current answer]"


class TaskStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise the real graph and Node methods without importing the LLM
        # configuration or loading models. Only their external interfaces are fake.
        stubs = {}
        for name, fields in {
            "mas": {}, "mas.agents": {"Agent": _PromptAgent},
            "mas.llm": {"Message": SimpleNamespace},
            "mas.reasoning": {"ReasoningConfig": SimpleNamespace},
        }.items():
            module = ModuleType(name)
            for key, value in fields.items():
                setattr(module, key, value)
            stubs[name] = module
        source = Path(__file__).resolve().parents[2] / "tasks/mas_workflow/macnet"
        with mock.patch.dict(sys.modules, stubs):
            for filename in ("node", "graph"):
                name = f"task_state_test_{filename}"
                spec = importlib.util.spec_from_file_location(name, source / f"{filename}.py")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                setattr(cls, filename, module)

    def make_mas(self):
        nodes = {index: self.node.Node(_PromptAgent(f"solver_{index}")) for index in range(3)}
        mask = self.graph.gen_graph_mask_info("Chain", 3)
        for matrix, connect in (
            (mask.fixed_spatial_masks, self.node.Node.add_spatial_edge),
            (mask.fixed_temporal_masks, self.node.Node.add_temporal_edge),
        ):
            for source, row in enumerate(matrix):
                for target, connected in enumerate(row):
                    if connected:
                        connect(nodes[source], nodes[target])
        return SimpleNamespace(_agent_nodes=nodes, _decision_node=self.node.Node(_PromptAgent("decision")))

    @staticmethod
    def remember(node, answer):
        node._input, node._output = ["previous question"], [answer]
        node.update_memory()

    def next_prompt(self, mas):
        recipient = mas._agent_nodes[0]
        recipient.execute(SimpleNamespace(content="Identical current task and retrieved memories"), False)
        return recipient._agent.prompts[-1]

    def test_previous_drop_changes_next_prompt_only_without_task_reset(self):
        baseline, gate = self.make_mas(), self.make_mas()
        self.remember(baseline._agent_nodes[2], "Finish[previous KEEP result]")
        self.remember(gate._agent_nodes[2], "Finish[previous DROP result]")
        self.assertNotEqual(self.next_prompt(baseline), self.next_prompt(gate))
        reset_task_execution_state(baseline)
        reset_task_execution_state(gate)
        self.assertEqual(self.next_prompt(baseline), self.next_prompt(gate))

    def test_reset_clears_workers_and_decision_but_preserves_graph_and_same_task_history(self):
        mas = self.make_mas()
        for node in (*mas._agent_nodes.values(), mas._decision_node):
            self.remember(node, "previous task output")
        reset_task_execution_state(mas)
        for node in (*mas._agent_nodes.values(), mas._decision_node):
            self.assertEqual(node.memory, ([], []))
            self.assertEqual(node.current_output, ())
            self.assertEqual(node._input, [])
        self.assertEqual([n.id for n in mas._agent_nodes[0].temporal_predecessors], ["solver_2"])
        self.assertEqual([n.id for n in mas._agent_nodes[0].spatial_successors], ["solver_1"])
        self.remember(mas._agent_nodes[2], "same task previous round")
        self.assertIn("same task previous round", self.next_prompt(mas))

    def test_resumed_question_receives_same_prompt_as_uninterrupted_run(self):
        uninterrupted = self.make_mas()
        self.remember(uninterrupted._agent_nodes[2], "last completed question")
        reset_task_execution_state(uninterrupted)
        restarted = self.make_mas()
        reset_task_execution_state(restarted)
        self.assertEqual(self.next_prompt(uninterrupted), self.next_prompt(restarted))

    def test_resume_rejects_pre_isolation_results(self):
        with self.assertRaisesRegex(SystemExit, "without isolated task state"):
            validate_task_state_policy({"design": {}}, Path("legacy"))
        validate_task_state_policy({"design": {"task_state_policy": TASK_STATE_POLICY}}, Path("current"))

    def test_legacy_baseline_backup_preserves_artifacts_and_rebuilds_dependent_arms(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            for method, schema in (
                ("native_gmemory", "gmemory-team-exposure-hotpotqa-evaluation-v1"),
                ("semantic_linear", "gmemory-team-exposure-fever-evaluation-v1"),
                ("semantic_mlp", "gmemory-team-exposure-hotpotqa-evaluation-v2"),
            ):
                output = root / method
                output.mkdir()
                (output / "run_manifest.json").write_text(json.dumps({"design": {"runner_schema": schema}}))
                (output / "memory_gate.jsonl").write_bytes(b'{"task_id": 1}\n')
                (output / "llm_cache.sqlite").write_bytes(b"keep this cache")
            for filename in ("matrix_manifest.json", "matrix_results.json", "matrix_progress.json"):
                (root / filename).write_text("{}")
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            (checkpoints / "semantic_linear.json").write_bytes(b"trained weights")
            backup = archive_legacy_evaluations(root, ("native_gmemory", "semantic_linear", "semantic_mlp"))
            self.assertIsNotNone(backup)
            self.assertEqual((backup / "native_gmemory/llm_cache.sqlite").read_bytes(), b"keep this cache")
            self.assertEqual((backup / "semantic_linear/memory_gate.jsonl").read_bytes(), b'{"task_id": 1}\n')
            self.assertTrue((backup / "matrix_results.json").is_file())
            self.assertTrue((root / "matrix_manifest.json").is_file())
            self.assertEqual((checkpoints / "semantic_linear.json").read_bytes(), b"trained weights")
            self.assertTrue((backup / "semantic_mlp/memory_gate.jsonl").is_file())
            self.assertFalse((root / "native_gmemory").exists())
            self.assertFalse((root / "matrix_results.json").exists())
            self.assertIsNone(archive_legacy_evaluations(root, ("native_gmemory", "semantic_linear", "semantic_mlp")))
            self.assertEqual(len(list((root / "evaluation_backups").iterdir())), 1)

    def test_legacy_arm_repair_keeps_current_baseline_and_other_arms(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            for method, version in (("native_gmemory", 2), ("semantic_linear", 1), ("semantic_mlp", 2)):
                output = root / method
                output.mkdir()
                (output / "run_manifest.json").write_text(json.dumps({"design": {
                    "runner_schema": f"gmemory-team-exposure-hotpotqa-evaluation-v{version}",
                }}))
            backup = archive_legacy_evaluations(root, ("native_gmemory", "semantic_linear", "semantic_mlp"))
            self.assertTrue((backup / "semantic_linear/run_manifest.json").is_file())
            self.assertTrue((root / "native_gmemory/run_manifest.json").is_file())
            self.assertTrue((root / "semantic_mlp/run_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
