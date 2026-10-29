"""Exercise command-only prompts through native MacNet and the PDDL environment."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from causal_memory_control.interactive_data import pddl_inventory
from causal_memory_control.interactive_envs import Episode, quiet_native_imports
from causal_memory_control.interactive_runtime import (
    ACTION_OUTPUT_INSTRUCTION, ActionFormatError, NativeRunner, audit, validate_action_response,
)


REPO = Path(__file__).resolve().parents[2]


class ActionOutputTests(unittest.TestCase):
    def test_format_check_is_limited_to_action_requests_and_leaves_legality_to_environment(self):
        messages = [{"role": "system", "content": ACTION_OUTPUT_INSTRUCTION}]
        for answer in ("## Action Plan", "```", "pick ball1 rooma left\nmove rooma roomb"):
            with self.subTest(answer=answer), self.assertRaises(ActionFormatError):
                validate_action_response(messages, answer)
        # An impossible object is an ordinary measured invalid action, not an
        # infrastructure/format failure or an invitation to substitute a plan.
        validate_action_response(messages, "pick nonexistent roomb left")
        validate_action_response(messages, "Action: pick ball1 rooma left")
        validate_action_response([{"role": "system", "content": "Summarize the support memory"}],
                                 "## Action Plan\nA summary may use Markdown.")

    def test_all_four_nodes_receive_domain_rules_and_commands_replay_in_real_gripper(self):
        quiet_native_imports()
        from mas.memory.mas_memory.memory_base import MASMemoryBase

        class Memory(MASMemoryBase):
            def retrieve_memory(self, **kwargs):
                return self.apply_retrieval_gate(([], [], []),
                    query_task=kwargs["query_task"], task_state=self.current_task_context.task_description)

        # A fixed responder tests prompt propagation and actual action parsing;
        # this does not test Qwen's planning ability or use any expert backend.
        commands = ["pick ball1 rooma left", "move rooma roomb", "drop ball1 roomb left",
                    "pick ball2 roomb left", "pick ball3 roomb right", "move roomb rooma",
                    "drop ball2 rooma left", "drop ball3 rooma right", "move rooma roomb",
                    "pick ball4 roomb left", "move roomb rooma", "drop ball4 rooma left"]
        requests = []

        class Chat:
            def __init__(self, *args, **kwargs):
                self.calls = self.cache_hits = 0

            def __call__(self, messages, **kwargs):
                self.calls += 1
                requests.append((messages, kwargs))
                system = messages[0].content
                if "Your name is Robby." not in system or ACTION_OUTPUT_INSTRUCTION not in system:
                    return "## Action Plan"
                return commands[((self.calls - 1) // 4) % len(commands)]

            def close(self):
                pass

        tasks, _ = pddl_inventory(REPO)
        task = next(t for t in tasks if t["id"] == "pddl:gripper:prob14")
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
             mock.patch("causal_diagnostic.oracle_recipient.seeded_client.SeededCachedChat", Chat), \
             mock.patch("mas.memory.mas_memory.GMemory.GMemory", Memory), \
             mock.patch("mas.utils.EmbeddingFunc", return_value=None):
            config = self.config()
            runner = NativeRunner(config, directory, Path(directory) / "empty-memory")
            try:
                baseline = runner.run(task, 2000)
                replay = runner.run(task, 2000, expected=baseline)
                audit(replay, baseline, [])
            finally:
                runner.close()
        self.assertEqual(baseline["outcome"]["success"], 1.0)
        self.assertEqual(baseline["outcome"]["invalid_actions"], 0)
        self.assertEqual([t["action"] for t in baseline["trajectory"]], commands)
        self.assertEqual(len(requests), 2 * 4 * len(commands))
        for messages, options in requests:
            self.assertIn("Your name is Robby.", messages[0].content)
            self.assertIn(ACTION_OUTPUT_INSTRUCTION, messages[0].content)
            self.assertEqual(options["stop_strs"], ["\n"])
            self.assertEqual(options["temperature"], .7)

    def test_heading_only_response_stops_before_any_environment_step_or_memory_write(self):
        quiet_native_imports()
        from mas.memory.mas_memory.memory_base import MASMemoryBase

        saved = []

        class Memory(MASMemoryBase):
            def retrieve_memory(self, **kwargs):
                return self.apply_retrieval_gate(([], [], []),
                    query_task=kwargs["query_task"], task_state=self.current_task_context.task_description)

            def add_memory(self, row):
                saved.append(row)

        class Chat:
            def __init__(self, *args, **kwargs):
                self.calls = self.cache_hits = 0

            def __call__(self, *args, **kwargs):
                self.calls += 1
                return "## Action Plan"

            def close(self):
                pass

        tasks, _ = pddl_inventory(REPO)
        task = next(t for t in tasks if t["id"] == "pddl:gripper:prob14")
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
             redirect_stderr(io.StringIO()) as errors, \
             mock.patch("causal_diagnostic.oracle_recipient.seeded_client.SeededCachedChat", Chat), \
             mock.patch("mas.memory.mas_memory.GMemory.GMemory", Memory), \
             mock.patch("mas.utils.EmbeddingFunc", return_value=None), \
             mock.patch.object(Episode, "step", side_effect=AssertionError("must not execute a heading")) as step:
            runner = NativeRunner(self.config(), directory, Path(directory) / "empty-memory")
            try:
                with self.assertRaisesRegex(RuntimeError, "LLM action format failed") as raised:
                    runner.run(task, 2000)
                self.assertIsInstance(raised.exception.__cause__, ActionFormatError)
                self.assertIn("## Action Plan", str(raised.exception))
                step.assert_not_called()
                self.assertEqual(saved, [])
                events = [json.loads(line) for line in errors.getvalue().splitlines() if line.startswith("{")]
                self.assertEqual(len(events), 3)
                self.assertTrue(all(e["event"] == "interactive_action_format_failed" for e in events))
            finally:
                runner.close()

    @staticmethod
    def config():
        return {"repo": str(REPO), "model": "mock", "endpoint": "http://127.0.0.1:1/v1",
                "max_steps": 30, "few_shots": 0, "successful_topk": 0, "insights_topk": 0,
                "temperature": .7, "embedding_model": "never-load"}


if __name__ == "__main__":
    unittest.main()
