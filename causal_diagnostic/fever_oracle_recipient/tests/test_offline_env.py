from __future__ import annotations

import unittest

from causal_diagnostic.fever_oracle_recipient.offline_env import OfflineFeverEnv


class OfflineFeverEnvTests(unittest.TestCase):
    def test_parser_normalizes_label_and_finish_syntax(self):
        self.assertEqual(
            OfflineFeverEnv.process_action("supports"), "Finish[SUPPORTS]"
        )
        self.assertEqual(
            OfflineFeverEnv.process_action("Answer: Finish[refutes]"),
            "Finish[REFUTES]",
        )
        self.assertNotEqual(
            OfflineFeverEnv.process_action("SUPPORTS or REFUTES"),
            "Finish[REFUTES]",
        )
        self.assertEqual(
            OfflineFeverEnv.process_action(
                "Evidence[Soul Food (film)]\nFinish[SUPPORTS]"
            ),
            "Evidence[Soul Food (film)]\nFinish[SUPPORTS]",
        )

    def test_exact_label_reward_and_one_step_completion(self):
        env = OfflineFeverEnv(max_trials=1)
        task_main, description = env.set_env(
            {"id": 7, "claim": "A claim", "label": "REFUTES"}
        )
        self.assertIn("A claim", task_main)
        self.assertIn("Finish[REFUTES]", description)
        env.reset()
        observation, reward, done = env.step("REFUTES")
        self.assertEqual(reward, 1.0)
        self.assertTrue(done)
        self.assertIn("CORRECT", observation)
        self.assertEqual(env.infos["steps"], 1)
        self.assertEqual(env.feedback()[:2], (1.0, True))

    def test_wrong_label_is_terminal_failure(self):
        env = OfflineFeverEnv(max_trials=1)
        env.set_env({"id": 8, "claim": "A claim", "label": "SUPPORTS"})
        _observation, reward, done = env.step("Finish[REFUTES]")
        self.assertEqual(reward, 0.0)
        self.assertTrue(done)
        self.assertEqual(env.feedback()[:2], (0.0, False))


if __name__ == "__main__":
    unittest.main()
