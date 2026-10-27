from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import unittest

from causal_memory_control.interactive_barman import symbolic_plan
from causal_memory_control.interactive_data import pddl_inventory
from causal_memory_control.interactive_envs import Episode, make_expert


REPO = Path(__file__).resolve().parents[2]


class BarmanConstructiveTests(unittest.TestCase):
    def test_all_registered_barman_problems_replay_in_the_unchanged_environment(self):
        rows, _ = pddl_inventory(REPO)
        problems = [r for r in rows if r["domain"] == "barman"]
        self.assertEqual(len(problems), 20)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            for task in problems:
                with self.subTest(task=task["id"]):
                    episode = Episode(task, REPO, max_steps=200)
                    try:
                        commands = symbolic_plan(episode.native)
                        self.assertTrue(commands)
                        self.assertEqual(commands, symbolic_plan(episode.native))
                        for command in commands:
                            _, _, done = episode.step(command)
                            if done:
                                break
                        self.assertTrue(episode.won)
                        self.assertEqual(episode.invalid_actions, 0)
                        self.assertLessEqual(episode.steps, 200)
                    finally:
                        episode.close()

    def test_p09_is_verified_and_records_the_constructive_backend(self):
        rows, _ = pddl_inventory(REPO)
        task = next(r for r in rows if r["id"] == "pddl:barman:p09")
        timeout = int(os.environ.get("CMC_EXPERT_TEST_TIMEOUT", "300"))
        result = make_expert(task, REPO, timeout=timeout, pddl_expert="barman_constructive")
        self.assertEqual(result["source"], "barman_constructive_symbolic_v1")
        self.assertEqual(result["validation"]["success"], 1)
        self.assertEqual(result["validation"]["invalid_actions"], 0)
        self.assertEqual(result["target"], "\n".join(result["actions"]))
        phases = [p["expert_phase"] for p in result["expert_execution"]["phases"]]
        self.assertIn("barman_constructive_plan", phases)
        self.assertNotIn("planner_search", phases)


if __name__ == "__main__":
    unittest.main()
