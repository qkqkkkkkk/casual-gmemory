from __future__ import annotations

import os
from pathlib import Path
import unittest

os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:1/v1")
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

from causal_diagnostic.oracle_recipient.masked_macnet import (
    MemoryMaskPolicy,
    RecipientMaskedMacNet,
)
from causal_diagnostic.oracle_recipient.seeded_client import SeededCachedChat


class MaskPolicyTests(unittest.TestCase):
    def test_receiver_drop_changes_only_that_worker(self):
        mas = RecipientMaskedMacNet()
        mas.memory_policy = MemoryMaskPolicy(
            condition="drop_solver_1",
            candidate_kind="trajectory",
            candidate_index=0,
            drop_recipients=frozenset(("solver_1",)),
        )
        trajectories = ["target", "other"]
        insights = ["rule"]
        kept, _, exposed = mas._memory_for(
            trajectories, insights, recipient="solver_0"
        )
        self.assertEqual(kept, trajectories)
        self.assertTrue(exposed)
        dropped, _, exposed = mas._memory_for(
            trajectories, insights, recipient="solver_1"
        )
        self.assertEqual(dropped, ["other"])
        self.assertFalse(exposed)
        decision, _, exposed = mas._memory_for(
            trajectories, insights, recipient=None, decision=True
        )
        self.assertEqual(decision, trajectories)
        self.assertTrue(exposed)

    def test_global_drop_masks_workers_and_decision(self):
        mas = RecipientMaskedMacNet()
        mas.memory_policy = MemoryMaskPolicy(
            condition="global_drop",
            candidate_kind="insight",
            candidate_index=0,
            drop_recipients=frozenset(("solver_0", "solver_1")),
            drop_from_decision=True,
        )
        _, worker_insights, exposed = mas._memory_for(
            ["trajectory"], ["target", "other"], recipient="solver_0"
        )
        self.assertEqual(worker_insights, ["other"])
        self.assertFalse(exposed)
        _, decision_insights, exposed = mas._memory_for(
            ["trajectory"], ["target", "other"], recipient=None, decision=True
        )
        self.assertEqual(decision_insights, ["other"])
        self.assertFalse(exposed)

    def test_prompt_seed_is_stable_and_experiment_specific(self):
        first = object.__new__(SeededCachedChat)
        first.model_name = "model"
        first.experiment_seed = 7
        second = object.__new__(SeededCachedChat)
        second.model_name = "model"
        second.experiment_seed = 8
        messages = [{"role": "user", "content": "same"}]
        self.assertEqual(first._request_seed(messages), first._request_seed(messages))
        self.assertNotEqual(first._request_seed(messages), second._request_seed(messages))


if __name__ == "__main__":
    unittest.main()

