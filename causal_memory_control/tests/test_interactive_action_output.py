"""Exercise command-only prompts through native MacNet and the PDDL environment."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from causal_memory_control.interactive_data import pddl_inventory
from causal_memory_control.interactive_envs import quiet_native_imports
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

    def test_heading_only_response_counts_as_invalid_action_without_aborting_episode(self):
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
             mock.patch("mas.utils.EmbeddingFunc", return_value=None):
            config = {**self.config(), "max_steps": 2}
            runner = NativeRunner(config, directory, Path(directory) / "empty-memory")
            try:
                result = runner.run(task, 2000)
                self.assertEqual(result["outcome"]["steps"], 2)
                self.assertEqual(result["outcome"]["success"], 0)
                self.assertEqual(result["outcome"]["invalid_actions"], 2)
                self.assertEqual(result["outcome"]["termination_reason"], "step_limit")
                self.assertTrue(all(t["action"] == "## Action Plan" for t in result["trajectory"]))
                self.assertEqual(len(saved), 1)
                self.assertFalse(saved[0].label)
                self.assertEqual(runner.client.branch_errors, 0)
                self.assertEqual(runner.client.calls, 8)
                events = [json.loads(line) for line in errors.getvalue().splitlines() if line.startswith("{")]
                self.assertEqual(len(events), 8)
                self.assertTrue(all(e["event"] == "interactive_action_format_warning" for e in events))
            finally:
                runner.close()

    def test_alfworld_fence_consumes_one_step_and_next_action_can_succeed(self):
        quiet_native_imports()
        from mas.memory.mas_memory.memory_base import MASMemoryBase
        from types import SimpleNamespace

        class Memory(MASMemoryBase):
            def retrieve_memory(self, **kwargs):
                return self.apply_retrieval_gate(([], [], []),
                    query_task=kwargs["query_task"], task_state=self.current_task_context.task_description)

        class Chat:
            def __init__(self, *args, **kwargs):
                self.calls = self.cache_hits = 0

            def __call__(self, *args, **kwargs):
                self.calls += 1
                return "```plaintext" if self.calls % 8 == 4 else "go to countertop 1"

            def close(self):
                pass

        class TextEnvironment:
            def __init__(self):
                self.commands = []

            def reset(self):
                return SimpleNamespace(feedback="Your task is to: put bread on countertop.", won=False,
                                       admissible_commands=["go to countertop 1"])

            def step(self, action):
                self.commands.append(action)
                valid = action == "go to countertop 1"
                return SimpleNamespace(feedback="Done." if valid else "Nothing happens.", won=valid,
                                       admissible_commands=["go to countertop 1"]), int(valid), valid

            def close(self):
                pass

        game = "mock-game.z8"
        task = {"id": "alfworld:format-test", "dataset": "alfworld", "task_type": "put",
                "config": {"task_type": "put", "env_kwargs": {"gamefile": game}}, "files": {game: "test-hash"}}
        environment = TextEnvironment()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
             redirect_stderr(io.StringIO()), \
             mock.patch("textworld.start", return_value=environment), \
             mock.patch("causal_diagnostic.oracle_recipient.seeded_client.SeededCachedChat", Chat), \
             mock.patch("mas.memory.mas_memory.GMemory.GMemory", Memory), \
             mock.patch("mas.utils.EmbeddingFunc", return_value=None):
            runner = NativeRunner({**self.config(), "max_steps": 2}, directory, Path(directory) / "empty-memory")
            try:
                result = runner.run(task, 2000)
                replay = runner.run(task, 2000, expected=result)
                audit(replay, result, [])
                self.assertEqual(result["outcome"]["success"], 1)
                self.assertEqual(result["outcome"]["steps"], 2)
                self.assertEqual(result["outcome"]["invalid_actions"], 1)
                self.assertEqual(result["trajectory"][0]["action"], "```plaintext")
                self.assertFalse(result["trajectory"][0]["valid_action"])
                self.assertEqual(result["trajectory"][1]["action"], "go to countertop 1")
                self.assertEqual(runner.client.calls, 16)
                self.assertEqual(runner.client.branch_errors, 0)
                self.assertEqual(environment.commands, ["```plaintext", "go to countertop 1"] * 2)
            finally:
                runner.close()

    @staticmethod
    def config():
        return {"repo": str(REPO), "model": "mock", "endpoint": "http://127.0.0.1:1/v1",
                "max_steps": 30, "few_shots": 0, "successful_topk": 0, "insights_topk": 0,
                "temperature": .7, "embedding_model": "never-load"}


if __name__ == "__main__":
    unittest.main()
