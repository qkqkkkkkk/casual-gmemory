"""Keep real HTTP/SDK failures visible through legacy MacNet retry handling."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from causal_memory_control.interactive_data import pddl_inventory
from causal_memory_control.interactive_envs import quiet_native_imports
from causal_memory_control.interactive_runtime import NativeRunner, request_error_details


REPO = Path(__file__).resolve().parents[2]


class RequestErrorTests(unittest.TestCase):
    def test_error_chain_preserves_http_status_and_handles_cycles(self):
        original = RuntimeError("generation service failed")
        original.status_code = 503
        wrapper = RuntimeError("LLM request failed after four attempts")
        wrapper.__cause__ = original
        original.__context__ = wrapper
        details = request_error_details(wrapper)
        self.assertEqual(len(details), 2)
        self.assertEqual(details[1]["http_status"], 503)
        self.assertIn("generation service failed", details[1]["message"])

    def test_native_unassigned_action_does_not_replace_original_request_failure(self):
        self._run_failure(empty=False)

    def test_empty_model_response_is_reported_and_never_saved_as_an_outcome(self):
        self._run_failure(empty=True)

    def _run_failure(self, *, empty):
        quiet_native_imports()
        from mas.memory.mas_memory.memory_base import MASMemoryBase

        class Memory(MASMemoryBase):
            def retrieve_memory(self, **kwargs):
                return self.apply_retrieval_gate(([], [], ["sample memory"]),
                    query_task=kwargs["query_task"], task_state=self.current_task_context.task_description)

        class Chat:
            def __init__(self, *args, **kwargs):
                self.calls = self.cache_hits = 0

            def __call__(self, *args, **kwargs):
                self.calls += 1
                if empty:
                    return ""
                original = RuntimeError("HTTP 503: generation runner unavailable")
                original.status_code = 503
                raise RuntimeError("LLM request failed after four attempts") from original

            def close(self):
                pass

        rows, _ = pddl_inventory(REPO)
        selected = next(r for r in rows if r["id"] == "pddl:blockworld:problem1")
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
             redirect_stderr(io.StringIO()) as errors, \
             mock.patch("causal_diagnostic.oracle_recipient.seeded_client.SeededCachedChat", Chat), \
             mock.patch("mas.memory.mas_memory.GMemory.GMemory", Memory), \
             mock.patch("mas.utils.EmbeddingFunc", return_value=None):
            config = {"repo": str(REPO), "model": "mock", "endpoint": "http://127.0.0.1:1/v1", "max_steps": 2,
                "few_shots": 0, "successful_topk": 0, "insights_topk": 1, "temperature": .7,
                "embedding_model": "never-load"}
            runner = NativeRunner(config, directory, Path(directory) / "unused")
            try:
                with self.assertRaisesRegex(RuntimeError, "incomplete episode was not saved") as failure:
                    runner.run(selected, 2000)
                expected = "empty model answer" if empty else "HTTP 503: generation runner unavailable"
                self.assertIn(expected, str(failure.exception))
                self.assertNotIsInstance(failure.exception.__cause__, UnboundLocalError)
                self.assertEqual(runner.client.branch_errors, 3)
                events = [json.loads(line) for line in errors.getvalue().splitlines() if line.startswith("{")]
                self.assertEqual(len(events), 3)
                self.assertEqual(events[0]["event"], "interactive_llm_request_failed")
                if not empty:
                    self.assertEqual(events[0]["error_chain"][-1]["http_status"], 503)
                self.assertFalse((Path(directory) / "evaluation/policies.jsonl").exists())
            finally:
                runner.close()


if __name__ == "__main__":
    unittest.main()
