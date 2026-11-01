"""Matched branches keep the measured order without hiding real retrieval drift."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from causal_memory_control.gmemory_adapter import GMemoryRetrievalAdapter
from causal_memory_control.interactive_data import digest
from causal_memory_control.interactive_runtime import (
    ActionStore, NativeRunner, baseline_for, candidate_rows, execute, retrieval_check,
)
from causal_memory_control.tests.test_interactive_benchmark import measured


def trajectory(name, **extra):
    return SimpleNamespace(task_main=name, task_description="source " + name,
                           task_trajectory="go to " + name, label=True, extra_fields=extra)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.adapter = GMemoryRetrievalAdapter()
        self.original = ([trajectory("a"), trajectory("b")], [], ["rule a", "rule b"])
        self.reference = {"query": "goal", "state": "observation",
                          "candidates": candidate_rows(self.adapter.adapt(self.original))}

    def check(self, root, result, **kwargs):
        return retrieval_check(self.adapter, result, self.reference, query_task="goal",
            task_state=kwargs.get("state", "observation"), root=root, task_id="alfworld:task",
            seed=2000, drops=[self.reference["candidates"][0]["memory_id"]])

    def test_identical_retrieval_has_no_writes_and_preserves_native_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIs(self.check(directory, self.original), self.original)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_only_order_is_restored_and_audited_without_changing_contents_or_ranks(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            result = (list(reversed(self.original[0])), [], list(reversed(self.original[2])))
            before = candidate_rows(self.adapter.adapt(result))
            aligned = self.check(directory, result)
            self.assertEqual(candidate_rows(self.adapter.adapt(aligned)), self.reference["candidates"])
            self.assertIs(aligned[0][0], self.original[0][0])
            self.assertEqual(candidate_rows(self.adapter.adapt(result)), before)
            records = list((Path(directory) / "retrieval_checks").glob("*.json"))
            self.assertEqual(len(records), 1)
            record = json.loads(records[0].read_text())
            self.assertEqual(record["classification"], "order_aligned_to_baseline")
            self.assertNotEqual(record["observed_ids"], record["expected_ids"])

    def test_actual_membership_content_metadata_context_and_multiplicity_changes_still_fail(self):
        self.original[0][0].extra_fields["memory_id"] = "stable-a"
        self.reference["candidates"] = candidate_rows(self.adapter.adapt(self.original))
        cases = {
            "membership": ([self.original[0][0], trajectory("c")], [], self.original[2]),
            "missing": (self.original[0][:1], [], self.original[2]),
            "duplicate": ([self.original[0][0]] * 2, [], self.original[2]),
            "metadata": ([trajectory("a", memory_id="stable-a", fail_reason="changed"),
                          self.original[0][1]], [], self.original[2]),
            "insight_content": (self.original[0], [], ["rule a", "changed rule"]),
            "context": self.original,
        }
        # With explicit stable IDs a changed trajectory must still be rejected.
        changed = deepcopy(self.original)
        changed[0][0].task_trajectory += " changed"
        cases["trajectory_content"] = changed
        for name, result in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "original retrieval changed before exposure.*details="):
                    self.check(directory, result, state="different state" if name == "context" else "observation")
                records = list((Path(directory) / "retrieval_checks").glob("*.json"))
                self.assertEqual(len(records), 1)
                record = json.loads(records[0].read_text())
                self.assertEqual(record["classification"], "public_context_changed" if name == "context"
                                 else "candidate_data_changed")

    def test_duplicate_text_is_not_deduplicated_by_alignment(self):
        original = ([trajectory("a"), trajectory("b")], [], ["same", "same", "other"])
        self.reference["candidates"] = candidate_rows(self.adapter.adapt(original))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            aligned = self.check(directory, (list(reversed(original[0])), [], ["other", "same", "same"]))
            self.assertEqual(candidate_rows(self.adapter.adapt(aligned)), self.reference["candidates"])
            self.assertEqual(aligned[2], original[2])

    def test_cache_resume_registers_retrieval_without_running_an_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ActionStore(directory, {"phase": "diagnostic"})
            runner = NativeRunner.__new__(NativeRunner)
            runner._retrieval_references = {}
            runner.run = mock.Mock(return_value=measured("task", 0, []))
            first = baseline_for(runner, store, {"id": "task"}, 0)
            runner._retrieval_references.clear()
            again = baseline_for(runner, store, {"id": "task"}, 0)
            self.assertEqual(again, first)
            runner.run.assert_called_once()
            self.assertEqual(runner._retrieval_references["task"]["candidates"], first["candidates"])
            # The retrieval anchor is independent of the diagnostic sample seed.
            runner.remember_retrieval({**again, "seed": 1})
            changed = {**again, "candidates": []}
            with self.assertRaisesRegex(ValueError, "cached baseline retrieval changed"):
                runner.remember_retrieval(changed)
            self.assertEqual(store.get("task", 0, first, []), first)

    def test_runner_drop_and_new_repeat_use_cached_baseline_order_before_prompt_exposure(self):
        # Exercise NativeRunner.run and its actual nested FixedGate. Only the
        # external host/environment and GPU hooks are faked; no LLM is called.
        class Context:
            @staticmethod
            def to_dict(context):
                return {}

        class Episode:
            def __init__(self, *args):
                self.task_main, self.task_description = "goal", "observation"
                self.initial_fingerprint, self.resets, self.trace = "initial-task", [], []

            def result(self):
                return measured("task", 0, [])["outcome"]

            def close(self):
                pass

        exposures = []

        class Host:
            agents_team = ["solver0", "solver1", "solver2"]
            _decision_node = SimpleNamespace(id="decision")

            def add_observer(self, observer):
                pass

            def build_system(self, reasoner, memory, episode, config):
                self.memory = memory

            def schedule(self, config):
                self.memory.current_task_context = Context()
                result = self.memory._retrieval_gate.filter_retrieval(next(self.memory.retrievals),
                    query_task="goal", task_state="observation")
                exposures.append(([m.task_main for m in result[0]], result[2]))
                return 0.0, False

        memory = SimpleNamespace(retrievals=iter([
            self.original, (list(reversed(self.original[0])), [], list(reversed(self.original[2]))),
            (list(reversed(self.original[0])), [], list(reversed(self.original[2]))),
            ([trajectory("changed")], [], self.original[2])]))
        memory.set_retrieval_gate = lambda gate: setattr(memory, "_retrieval_gate", gate)
        modules = {"torch": SimpleNamespace(manual_seed=lambda seed: None,
                        cuda=SimpleNamespace(is_available=lambda: False)),
            "mas.reasoning": SimpleNamespace(ReasoningIO=lambda **kwargs: None),
            "tasks.mas_workflow.macnet.graph_mas": SimpleNamespace(MacNet=Host),
            "tasks.prompts": SimpleNamespace(get_dataset_system_prompt=lambda *args: "rules",
                                             get_task_few_shots=lambda *args: [])}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
             mock.patch.dict(sys.modules, modules), \
             mock.patch("causal_memory_control.interactive_envs.Episode", Episode), \
             mock.patch("causal_memory_control.interactive_runtime.configure_action_nodes"), \
             mock.patch("causal_memory_control.task_state.reset_task_execution_state"):
            runner = NativeRunner.__new__(NativeRunner)
            runner.config = {"repo": directory, "max_steps": 2, "few_shots": 0, "temperature": .7,
                             "successful_topk": 2, "insights_topk": 2}
            runner.root = runner.memory_dir = Path(directory)
            runner._retrieval_references = {}
            runner.memory = memory
            runner.client = SimpleNamespace(calls=0, cache_hits=0, branch_errors=0)
            task = {"id": "task", "dataset": "alfworld", "config": {}}
            store = ActionStore(Path(directory) / "actions", {"phase": "diagnostic"})
            first = baseline_for(runner, store, task, 0)
            runner._retrieval_references.clear()  # emulate a restart after the KEEP commit
            self.assertEqual(baseline_for(runner, store, task, 0), first)
            drop_id = first["candidates"][0]["memory_id"]
            dropped = execute(runner, store, task, 0, [drop_id], first)
            repeated = baseline_for(runner, store, task, 1)
            self.assertEqual(dropped["candidates"], first["candidates"])
            self.assertEqual(repeated["retrieval_fingerprint"], first["retrieval_fingerprint"])
            self.assertEqual(exposures, [(["a", "b"], ["rule a", "rule b"]),
                                        (["b"], ["rule a", "rule b"]),
                                        (["a", "b"], ["rule a", "rule b"])])
            self.assertEqual(store.get("task", 0, first, [drop_id]), dropped)
            before = list((store.root / "parts").glob("*.json"))
            with self.assertRaisesRegex(ValueError, "candidate_data_changed"):
                execute(runner, store, task, 0, [first["candidates"][1]["memory_id"]], first)
            self.assertEqual(list((store.root / "parts").glob("*.json")), before)
            self.assertIsNone(memory._retrieval_gate)


if __name__ == "__main__":
    unittest.main()
