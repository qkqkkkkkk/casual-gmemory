from __future__ import annotations

from dataclasses import dataclass, field
import unittest

from causal_memory_control import (
    AuditCheckpoint,
    GMemoryRetrievalAdapter,
    InterventionArm,
    InterventionBuilder,
    OracleControllability,
    OracleExample,
    PivotalityObservation,
    UtilityRetest,
    evaluate_pretraining_gate,
    observation_count_distribution,
    oracle_noise_floor,
    stratify_pivotality,
)


@dataclass
class FakeMessage:
    task_main: str
    task_description: str
    task_trajectory: str
    label: bool
    extra_fields: dict = field(default_factory=dict)


class AdapterTests(unittest.TestCase):
    def test_current_retrieval_tuple_is_adapted_without_gmemory_import(self) -> None:
        message = FakeMessage(
            "open door",
            "Open the red door",
            "take red key\nopen red door",
            True,
            {"key_steps": "take key; open door"},
        )
        adapter = GMemoryRetrievalAdapter()
        retrieval = adapter.adapt(([message], [], ["Check key color first."]))
        event = adapter.build_event(
            retrieval,
            target_id=retrieval.successful_ids[0],
            query="open door",
            task_state="red and blue keys are visible",
            receiver_agent_id="worker-a",
            receiver_role="solver",
            recipient_agent_ids=("worker-a", "worker-b", "worker-c"),
        )
        self.assertEqual(event.observation_count, 3)
        self.assertEqual(observation_count_distribution([event]), {3: 1})

        request = InterventionBuilder().build(
            AuditCheckpoint("cp", event.event_id, {}, {}),
            event,
            InterventionArm.DROP,
            seed=1,
            repeat_index=0,
        )
        prompt = adapter.render_prompt_inputs(
            request,
            retrieval,
            agent_id="worker-a",
            trajectory_formatter=lambda raw: f"FORMATTED:{raw.task_description}",
        )
        self.assertEqual(prompt.memory_few_shots, ())
        self.assertEqual(prompt.insights, ("Check key color first.",))
        other = adapter.render_prompt_inputs(
            request,
            retrieval,
            agent_id="worker-b",
            trajectory_formatter=lambda raw: f"FORMATTED:{raw.task_description}",
        )
        self.assertEqual(other.memory_few_shots, ("FORMATTED:Open the red door",))


class DiagnosticTests(unittest.TestCase):
    def test_mismatch_is_stratified_by_other_agent_margin(self) -> None:
        rows = (
            PivotalityObservation("tie-bad", (True, False), True),
            PivotalityObservation("tie-good", (False, True), False),
            PivotalityObservation("unanimous", (True, True), False),
        )
        bins = stratify_pivotality(rows)
        self.assertEqual(bins[0].count, 2)
        self.assertEqual(bins[0].mismatch_rate, 0.5)
        self.assertEqual(bins[-1].count, 1)
        self.assertEqual(bins[-1].mismatch_rate, 0.0)

    def test_retest_sign_consistency_exposes_noise_floor(self) -> None:
        report = oracle_noise_floor(
            (
                UtilityRetest("a", 1.0),
                UtilityRetest("a", 0.5),
                UtilityRetest("a", -0.2),
                UtilityRetest("b", -1.0),
                UtilityRetest("b", -0.4),
            )
        )
        event_a = next(row for row in report.per_event if row.event_id == "a")
        self.assertAlmostEqual(event_a.majority_fraction, 2 / 3)
        self.assertAlmostEqual(event_a.pairwise_agreement, 1 / 3)
        self.assertAlmostEqual(report.weighted_pairwise_agreement, 0.5)

        oracle = OracleControllability().evaluate(
            (
                OracleExample("a", 1.0, 0.0, {}),
                OracleExample("b", 0.0, 1.0, {}),
            ),
            score_names=(),
        )
        gate = evaluate_pretraining_gate(
            oracle, report, minimum_sign_consistency=0.7
        )
        self.assertTrue(gate.oracle_passed)
        self.assertFalse(gate.noise_floor_passed)
        self.assertFalse(gate.can_train)


if __name__ == "__main__":
    unittest.main()
    OracleControllability,
    OracleExample,
