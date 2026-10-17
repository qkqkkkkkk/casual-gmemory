from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from causal_diagnostic import collection_runtime as runtime
from causal_diagnostic.memory import FrozenBranchMemory, FrozenReadOnlyGMemory, FrozenRetrieval
from causal_diagnostic.fever_oracle_recipient import run_experiment as fever
from causal_diagnostic.hotpotqa_oracle_recipient import run_experiment as hotpot
from causal_memory_control.tests.test_dataset_matrix import fever_source
from mas.memory.common import MASMessage
from mas.memory.mas_memory.GMemory import GMemory


def frozen_result():
    return ([MASMessage(task_main=f"memory-{i}", task_description=f"example {i}",
        task_trajectory="Finish[SUPPORTS]", label=True) for i in range(3)], [],
        [f"Keep guidance rule {i}." for i in range(3)])


class Backend:
    def __init__(self, dataset):
        self.requests = []
        self.closed = 0
        self.dataset = dataset
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **payload):
        self.requests.append(payload)
        if payload.get("logprobs"):
            alternatives = [SimpleNamespace(token="A", logprob=-0.2), SimpleNamespace(token="B", logprob=-1.2)]
            return SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(
                content=[SimpleNamespace(top_logprobs=alternatives)]))])
        answer = "SUPPORTS" if self.dataset == "fever" else "New York City"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f"Evidence[Page#0]\nFinish[{answer}]"))])

    def close(self):
        self.closed += 1


class LiveMemory:
    loads = 0

    def __init__(self, **_):
        type(self).loads += 1
        self.memory_size = 50

    def retrieve_memory(self, **_):
        return copy.deepcopy(frozen_result())


def legacy_memory(*, frozen_retrieval, memory_records):
    # The old GMemory wrapper, with only disk opening mocked. Its inherited
    # task-context, summarize and read-only behavior are exercised unchanged.
    result = FrozenReadOnlyGMemory.__new__(FrozenReadOnlyGMemory)
    result._read_only = True
    result._retrieval_gate = None
    result._frozen_retrieval = frozen_retrieval
    return result


def collector_options(root, module):
    memory = root / "g-memory"
    memory.mkdir()
    (memory / "causal_snapshot_manifest.json").write_text(json.dumps({"memory_records": 50}))
    if module is fever:
        data, _ = fever_source(root, count=2)
        evaluation_ids = [1, 2]
    else:
        data = root / "hotpot.json"
        data.write_text(json.dumps([{"_id": f"h{i}", "question": f"Where is site {i}?", "answer": "New York City",
            "context": [["Place", ["The site is in New York City."]]], "supporting_facts": [["Place", 0]]} for i in range(2)]))
        evaluation_ids = [0, 1]
    snapshot = {"evaluation_ids": evaluation_ids, "support_ids": [], "memory_records": 50, "design_hash": "frozen-snapshot"}
    args = ["--test", str(data), "--memory-dir", str(memory), "--claims", "2", "--repeats", "2",
        "--all-candidates", "--gate-training-only", "--output-dir", str(root / "run"),
        "--endpoint", "http://unused.invalid/v1", "--model", "test-model"]
    if module is fever:
        args.append("--label-probabilities")
    return args, snapshot


class CollectionOptimizationTests(unittest.TestCase):
    def test_real_schedules_and_seeded_requests_match_with_fewer_schedules(self):
        for module, name in ((fever, "fever"), (hotpot, "hotpotqa")):
            with self.subTest(dataset=name), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                argv, snapshot = collector_options(root, module)
                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(redirect_stderr(io.StringIO()))
                backend = Backend(name)
                stack.enter_context(mock.patch("causal_diagnostic.oracle_recipient.seeded_client.OpenAI", return_value=backend))
                stack.enter_context(mock.patch.object(module, "_load_snapshot", return_value=snapshot))
                stack.enter_context(mock.patch.object(module, "_directory_hash", return_value="fixed"))
                stack.enter_context(mock.patch.object(module, "GMemory", LiveMemory))
                stack.enter_context(mock.patch.object(module, "EmbeddingFunc", return_value=None))
                stack.enter_context(mock.patch.object(GMemory, "memory_size", new_callable=mock.PropertyMock, return_value=50))
                branch = module._run_branch
                # Reference: all 24 schedules/task, the legacy GMemory wrapper,
                # actual RecipientMaskedMacNet and actual SeededCachedChat DB.
                with mock.patch.object(module, "SharedUseAll", return_value=None), mock.patch.object(
                        module, "FrozenBranchMemory", side_effect=legacy_memory), mock.patch.object(module, "_run_branch", wraps=branch) as call:
                    module.main(argv)
                    self.assertEqual(call.call_count, 48)
                reference = [json.loads(line) for line in (root / "run" / "branches.jsonl").read_text().splitlines()]
                requests = copy.deepcopy(backend.requests)
                backend.requests.clear()
                LiveMemory.loads = 0
                argv[argv.index("--output-dir") + 1] = str(root / "optimized")
                with mock.patch.object(module, "_run_branch", wraps=branch) as call:
                    module.main(argv)
                    self.assertEqual(call.call_count, 28)
                rows = [json.loads(line) for line in (root / "optimized" / "branches.jsonl").read_text().splitlines()]
                self.assertEqual(LiveMemory.loads, 1)
                self.assertEqual(requests, backend.requests)
                self.assertEqual(len(rows), 48)
                for old, new in zip(reference, rows):
                    for key in ("candidate", "candidate_set", "task", "repeat_index", "sample_seed", "condition", "outcome"):
                        self.assertEqual(old[key], new[key], key)
                stats = json.loads((root / "optimized" / "collection_diagnostics.json").read_text())
                self.assertEqual(stats["shared_use_all_branches_this_process"], 20)
                self.assertEqual(stats["executed_branches_this_process"], 28)

    def test_old_partial_log_resumes_without_rewriting_or_repeating_shared_use(self):
        for module, name in ((fever, "fever"), (hotpot, "hotpotqa")):
            with self.subTest(dataset=name), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                argv, snapshot = collector_options(root, module)
                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(redirect_stderr(io.StringIO()))
                backend = Backend(name)
                stack.enter_context(mock.patch("causal_diagnostic.oracle_recipient.seeded_client.OpenAI", return_value=backend))
                stack.enter_context(mock.patch.object(module, "_load_snapshot", return_value=snapshot))
                stack.enter_context(mock.patch.object(module, "_directory_hash", return_value="fixed"))
                stack.enter_context(mock.patch.object(module, "GMemory", LiveMemory))
                stack.enter_context(mock.patch.object(module, "EmbeddingFunc", return_value=None))
                stack.enter_context(mock.patch.object(GMemory, "memory_size", new_callable=mock.PropertyMock, return_value=50))
                branch, counter = module._run_branch, [0]

                def interrupt(*args, **kwargs):
                    counter[0] += 1
                    if counter[0] == 6:
                        raise KeyboardInterrupt()
                    return branch(*args, **kwargs)

                with mock.patch.object(module, "SharedUseAll", return_value=None), mock.patch.object(
                        module, "FrozenBranchMemory", side_effect=legacy_memory), mock.patch.object(module, "_run_branch", side_effect=interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        module.main(argv)
                path = root / "run" / "branches.jsonl"
                # Old versions have no execution metadata.
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                for row in rows:
                    row.pop("execution")
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                prefix = path.read_bytes()
                with mock.patch.object(module, "_run_branch", wraps=branch) as call:
                    module.main([*argv, "--resume"])
                    # Two DROP rows already persisted: 10 remaining DROP on
                    # task 1 plus 14 schedules on task 2; no new task-1 USE.
                    self.assertEqual(call.call_count, 24)
                self.assertTrue(path.read_bytes().startswith(prefix))
                self.assertEqual(len(path.read_text().splitlines()), 48)
                with mock.patch.object(module, "_run_branch") as call:
                    module.main([*argv, "--resume"])
                    call.assert_not_called()

    def test_frozen_branch_deep_copies_retrieval_and_isolates_task_history(self):
        frozen = FrozenRetrieval.from_result(frozen_result())
        a = FrozenBranchMemory(frozen_retrieval=frozen, memory_records=50)
        b = FrozenBranchMemory(frozen_retrieval=frozen, memory_records=50)
        a.init_task_context("a", "state-a")
        b.init_task_context("b", "state-b")
        a.retrieve_memory()[0][0].task_description = "mutated"
        a.save_task_context(label=True, feedback="feedback")
        self.assertEqual(b.summarize(), "state-b" + b.current_task_context.task_trajectory)
        self.assertNotEqual(b.retrieve_memory()[0][0].task_description, "mutated")
        self.assertEqual(a.memory_size, 50)
        self.assertFalse(hasattr(a, "main_memory"))


if __name__ == "__main__":
    unittest.main()
