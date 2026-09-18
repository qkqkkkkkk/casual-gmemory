from __future__ import annotations

import os
import unittest

os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:1/v1")
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

from causal_diagnostic.oracle_recipient.masked_macnet import (
    MemoryMaskPolicy,
    RecipientMaskedMacNet,
)


class FakeTrajectory:
    def __init__(self, name):
        self.task_description = name
        self.task_trajectory = f"trajectory-{name}"

    def get_extra_field(self, _key):
        return "steps"


class FakeReasoning:
    def __call__(self, _messages, _config):
        return "ACTION: test"


class FakeEnvironment:
    max_trials = 1

    def __init__(self):
        self.infos = {"steps": 0}

    def reset(self):
        self.infos = {"steps": 0}

    @staticmethod
    def process_action(action):
        return str(action).replace("ACTION:", "").strip()

    def step(self, _action):
        self.infos["steps"] = 1
        return "done", 1.0, True

    @staticmethod
    def feedback():
        return 1.0, True, "complete"


class FakeMemory:
    def __init__(self):
        self.counter = 0

    def init_task_context(self, _task_main, _task_description):
        return None

    @staticmethod
    def retrieve_memory(**_kwargs):
        return [FakeTrajectory("target"), FakeTrajectory("other")], [], ["rule"]

    @staticmethod
    def summarize(**_kwargs):
        return "frozen state"

    def add_agent_node(self, _message, upstream_agent_ids):
        self.counter += 1
        return f"node-{self.counter}"

    @staticmethod
    def move_memory_state(_action, _observation, **_kwargs):
        return None

    @staticmethod
    def save_task_context(label, feedback):
        return {"label": label, "feedback": feedback}

    @staticmethod
    def backward(_reward):
        return None


class MaskedScheduleTests(unittest.TestCase):
    def test_receiver_mask_keeps_decision_prompt_full(self):
        mas = RecipientMaskedMacNet()
        mas.build_system(
            FakeReasoning(),
            FakeMemory(),
            FakeEnvironment(),
            {
                "graph_type": "Chain",
                "node_num": 3,
                "use_critic": False,
                "successful_topk": 2,
                "failed_topk": 0,
                "insights_topk": 1,
                "threshold": 0.0,
                "use_projector": False,
            },
        )
        mas.set_memory_policy(
            MemoryMaskPolicy(
                condition="drop_solver_2",
                candidate_kind="trajectory",
                candidate_index=0,
                drop_recipients=frozenset(("solver_2",)),
            )
        )
        reward, done = mas.schedule(
            {
                "task_main": "goal",
                "task_description": "description",
                "few_shots": [],
            }
        )
        self.assertEqual(reward, 1.0)
        self.assertTrue(done)
        trace = mas.execution_trace[0]
        self.assertFalse(trace["workers"]["solver_2"]["candidate_exposed"])
        self.assertTrue(trace["decision"]["candidate_exposed"])
        self.assertNotEqual(
            trace["workers"]["solver_2"]["prompt_hash"],
            trace["decision"]["prompt_hash"],
        )
        self.assertEqual(
            trace["workers"]["solver_0"]["prompt_hash"],
            trace["decision"]["prompt_hash"],
        )


if __name__ == "__main__":
    unittest.main()

