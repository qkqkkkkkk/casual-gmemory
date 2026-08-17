from __future__ import annotations

import tempfile
import unittest
import importlib.util
from pathlib import Path

from prior_module import (
    CandidateRecord,
    DecisionContext,
    PriorModule,
    ReplayExecution,
    ReplayResult,
    RetrievalRecord,
)
from prior_module.interventions import (
    collect_dependency_pair,
    collect_team_rollout_pair,
)
from prior_module.actions import ActionCanonicalizer, ActionDistancePolicy
from prior_module.adapters import GMemoryAdapter


def make_context() -> DecisionContext:
    return DecisionContext(
        task_main="put a clean mug in cabinet 1",
        task_description="Find a mug, clean it, and put it in cabinet 1.",
        agent_id="decision-0",
        agent_profile="decision",
        system_instruction="Choose the next executable action.",
        latest_observation="A dirty mug is in the sink. Cabinet 1 is closed.",
        recent_task_trajectory="go to sink\nexamine mug",
        step_index=2,
        task_config={"task_type": "alfworld", "game_name": "pick_clean_then_place"},
    )


def make_candidate() -> CandidateRecord:
    return CandidateRecord(
        memory_id="traj-001",
        memory_type="trajectory",
        polarity="success",
        content=(
            "Source task: clean and store a mug\n"
            "Key steps: clean mug; open cabinet; put mug in cabinet."
        ),
        structured_metadata={
            "source_label": True,
            "state_count": 4,
            "source_reward_sum": 1.0,
            "source_reward_last": 1.0,
            "has_raw_replay_artifact": True,
            "source_environment_family": "alfworld",
            "source_task_type": "pick_clean_then_place",
            "source_model_id": "test-model",
            "source_prompt_version": "v1",
            "memory_schema_version": "gmemory-v1",
        },
    )


class PriorModuleTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("dotenv") is not None,
        "requires the optional G-Memory runtime dependencies",
    )
    def test_adapter_reads_current_mas_message_shape(self) -> None:
        from mas.memory.common import MASMessage

        message = MASMessage(
            task_main="clean mug",
            task_description="Clean the mug and store it.",
        )
        message.move_state("go to sink", "A dirty mug is in the sink.", reward=0.0)
        message.move_state("clean mug", "The mug is clean.", reward=1.0)
        message.label = True
        message.add_extra_field("key_steps", "go to sink; clean mug")
        adapted = GMemoryAdapter().adapt_trajectory(
            message,
            source_metadata={"source_environment_family": "alfworld"},
        )

        self.assertEqual(adapted.memory_type, "trajectory")
        self.assertEqual(adapted.polarity, "success")
        self.assertEqual(adapted.structured_metadata["state_count"], 2)
        self.assertEqual(adapted.structured_metadata["source_reward_last"], 1.0)
        self.assertEqual(
            adapted.structured_metadata["source_environment_family"], "alfworld"
        )

    def test_cold_start_score_and_replay_update(self) -> None:
        module = PriorModule()
        context = make_context()
        candidate = make_candidate()
        retrieval = RetrievalRecord(
            candidate_id=candidate.memory_id,
            query_task=context.task_text,
            retrieval_source="vector_fallback",
            rank=1,
            semantic_similarity=0.8,
        )
        first = module.score(context, [candidate], {candidate.memory_id: retrieval})[0]
        self.assertEqual(first.recommended_action, "VERIFY")
        self.assertAlmostEqual(first.dependency.expected_distance, 0.5)
        self.assertEqual(first.dependency.source, "cold_start_uniform")
        self.assertEqual(first.team_utility.source, "cold_start_uniform")

        replay = ReplayResult(
            memory_id=candidate.memory_id,
            passed=True,
            execution=ReplayExecution(True, 1.0, 0),
            success_threshold=0.0,
        )
        module.record_replay(replay)
        updated = module.score(context, [candidate], {candidate.memory_id: retrieval})[0]
        self.assertEqual(updated.reliability.source, "group_beta")
        self.assertGreater(updated.reliability.mean, first.reliability.mean)

    def test_supervised_heads_are_trainable(self) -> None:
        module = PriorModule()
        context = make_context()
        candidate = make_candidate()
        base = module.build_base_features(context, candidate, None, [candidate])
        rows = []
        distances = []
        utility_rows = []
        utility_labels = []
        for index in range(9):
            row = dict(base)
            row["step_index"] = float(index)
            rows.append(row)
            distances.append(0.75 if index % 2 else 0.25)
            utility_row = dict(row)
            utility_row["r0_mean"] = 0.7
            utility_row["r0_variance"] = 0.02
            utility_row["r0_evidence_count"] = 5.0
            utility_row["d0_expected_distance"] = distances[-1]
            utility_row["d0_p_any_change"] = 1.0
            utility_row["d0_p_material_change"] = float(distances[-1] >= 0.5)
            utility_row["d0_uncertainty"] = 0.3
            utility_rows.append(utility_row)
            utility_labels.append("positive" if index % 2 else "neutral")

        self.assertTrue(module.fit_dependency(rows, distances))
        self.assertTrue(module.fit_team_utility(utility_rows, utility_labels))
        score = module.score(context, [candidate])[0]
        self.assertEqual(score.dependency.source, "supervised_ordinal")
        self.assertEqual(score.team_utility.source, "rollout_supervised")

    def test_intervention_helpers_and_jsonl_log(self) -> None:
        pair = collect_dependency_pair(
            "traj-001",
            "heat apple 1 with microwave 1",
            "heat apple 2 with microwave 1",
            canonicalizer=ActionCanonicalizer(),
            distance_policy=ActionDistancePolicy(),
            decision_id="dep-1",
        )
        self.assertEqual(pair.distance, 0.5)
        rollout = collect_team_rollout_pair(
            "traj-001", 1.0, 0.0, threshold=0.5, decision_id="util-1"
        )
        self.assertEqual(rollout.utility_class, "positive")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior.jsonl"
            module = PriorModule(decision_log_path=str(path))
            module.score(make_context(), [make_candidate()])
            self.assertTrue(path.exists())
            self.assertIn('"event_type": "prior_decision"', path.read_text())
