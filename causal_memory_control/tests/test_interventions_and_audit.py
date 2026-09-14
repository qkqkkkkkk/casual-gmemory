from __future__ import annotations

import unittest

from causal_memory_control import (
    AuditCheckpoint,
    BranchOutcome,
    CounterfactualAudit,
    InterventionArm,
    InterventionBuilder,
    MemoryCandidate,
    MemoryUseEvent,
    RecipientContext,
)


def make_event() -> MemoryUseEvent:
    target = MemoryCandidate("m1", "trajectory", "use red key")
    other = MemoryCandidate("m2", "insight", "avoid locked blue door")
    return MemoryUseEvent(
        event_id="event-1",
        query="open the red door",
        task_state="two keys are visible",
        receiver_agent_id="agent-a",
        receiver_role="solver",
        memory=target,
        candidate_set=(target, other),
        recipient_contexts=(
            RecipientContext("agent-a", ("m1", "m2")),
            RecipientContext("agent-b", ("m1", "m2")),
        ),
    )


class InterventionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.event = make_event()
        self.checkpoint = AuditCheckpoint("cp-1", "event-1", {}, {"temperature": 0})
        self.placebo = MemoryCandidate(
            "p1",
            "trajectory",
            "use tan key",
            {"irrelevant": True},
        )
        self.builder = InterventionBuilder([self.placebo])

    def build(self, arm: InterventionArm):
        return self.builder.build(
            self.checkpoint, self.event, arm, seed=7, repeat_index=0
        )

    def test_receiver_drop_does_not_change_other_recipients(self) -> None:
        request = self.build(InterventionArm.DROP)
        self.assertEqual(request.context_for("agent-a").candidate_ids, ("m2",))
        self.assertEqual(request.context_for("agent-b").candidate_ids, ("m1", "m2"))

    def test_global_drop_changes_every_recipient(self) -> None:
        request = self.build(InterventionArm.GLOBAL_DROP)
        self.assertEqual(request.context_for("agent-a").candidate_ids, ("m2",))
        self.assertEqual(request.context_for("agent-b").candidate_ids, ("m2",))

    def test_placebo_is_type_and_position_matched(self) -> None:
        request = self.build(InterventionArm.PLACEBO)
        self.assertEqual(request.context_for("agent-a").candidate_ids, ("p1", "m2"))
        self.assertEqual(request.context_for("agent-b").candidate_ids, ("m1", "m2"))
        self.assertEqual(request.placebo_candidate, self.placebo)


class FakeRunner:
    def __init__(self):
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        receiver_has_target = "m1" in request.context_for("agent-a").candidate_ids
        # Seed noise is exactly matched and cancels in paired utility.
        noise = (request.seed % 3) * 0.01
        return BranchOutcome(
            response="target" if receiver_has_target else "control",
            team_reward=(1.0 if receiver_has_target else 0.25) + noise,
            local_metric=(0.8 if receiver_has_target else 0.3) + noise,
        )


class AuditTests(unittest.TestCase):
    def test_matched_seed_audit_shares_use_branch_across_controls(self) -> None:
        runner = FakeRunner()
        audit = CounterfactualAudit(
            runner,
            intervention_builder=InterventionBuilder(
                [MemoryCandidate("p1", "trajectory", "tan key", {"irrelevant": True})]
            ),
            repeats=3,
            base_seed=11,
        )
        result = audit.run(
            AuditCheckpoint("cp-1", "event-1", {}, {"temperature": 0}),
            make_event(),
            arms=(InterventionArm.DROP, InterventionArm.GLOBAL_DROP),
        )
        self.assertEqual(len(runner.requests), 9)  # 3 * (one USE + two controls)
        self.assertAlmostEqual(result.primary.team_utility.mean, 0.75)
        self.assertAlmostEqual(result.primary.local_utility.mean, 0.5)
        self.assertEqual(result.primary.behavior_effect.mean, 1.0)
        self.assertEqual([pair.seed for pair in result.primary.pairs], [11, 12, 13])


if __name__ == "__main__":
    unittest.main()

