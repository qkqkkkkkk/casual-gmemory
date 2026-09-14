from __future__ import annotations

import unittest

from causal_memory_control import (
    AmortizedUtilityEstimator,
    OracleControllability,
    OracleExample,
    PotentialOutcomeExample,
    PotentialOutcomePrediction,
    RelianceAction,
    RelianceController,
)


class EstimatorTests(unittest.TestCase):
    def test_two_potential_outcomes_are_learned(self) -> None:
        examples = []
        for index in range(40):
            x = -1.0 + 2.0 * index / 39
            examples.append(
                PotentialOutcomeExample(
                    {"x": x, "bias_feature": 1.0},
                    q_use=0.5 + 0.8 * x,
                    q_drop=0.5 - 0.8 * x,
                    event_id=f"e-{index}",
                )
            )
        estimator = AmortizedUtilityEstimator(
            ensemble_size=8,
            min_samples=8,
            epochs=180,
            include_residual_noise=False,
        )
        self.assertTrue(estimator.fit(examples))
        positive = estimator.predict({"x": 0.8, "bias_feature": 1.0})
        negative = estimator.predict({"x": -0.8, "bias_feature": 1.0})
        self.assertGreater(positive.utility, 0.7)
        self.assertLess(negative.utility, -0.7)
        self.assertAlmostEqual(positive.utility, positive.q_use - positive.q_drop)

    def test_cold_start_abstains(self) -> None:
        estimator = AmortizedUtilityEstimator()
        prediction = estimator.predict({"x": 1.0})
        decision = RelianceController(delta=0.1, kappa=1.0).decide(prediction)
        self.assertFalse(prediction.calibrated)
        self.assertEqual(decision.action, RelianceAction.VERIFY)


class ControllerTests(unittest.TestCase):
    @staticmethod
    def prediction(utility: float, uncertainty: float) -> PotentialOutcomePrediction:
        return PotentialOutcomePrediction(
            q_use=utility,
            q_drop=0.0,
            utility=utility,
            utility_class="positive" if utility > 0 else "negative",
            uncertainty=uncertainty,
            source="test",
            calibrated=True,
            training_samples=100,
        )

    def test_confidence_bounds_drive_three_actions(self) -> None:
        controller = RelianceController(delta=0.1, kappa=2.0)
        self.assertEqual(
            controller.decide(self.prediction(0.5, 0.1)).action,
            RelianceAction.ACCEPT,
        )
        self.assertEqual(
            controller.decide(self.prediction(-0.5, 0.1)).action,
            RelianceAction.REJECT,
        )
        self.assertEqual(
            controller.decide(self.prediction(0.2, 0.1)).action,
            RelianceAction.VERIFY,
        )


class OracleTests(unittest.TestCase):
    def test_oracle_and_heuristics_use_matched_budget(self) -> None:
        examples = (
            OracleExample("a", 1.0, 0.0, {"similarity": 0.1}),
            OracleExample("b", 0.0, 1.0, {"similarity": 0.9}),
            OracleExample("c", 0.8, 0.2, {"similarity": 0.2}),
            OracleExample("d", 0.1, 0.7, {"similarity": 0.8}),
        )
        result = OracleControllability(delta=0.0).evaluate(
            examples, score_names=("similarity",)
        )
        self.assertEqual(result.matched_budget, 2)
        self.assertEqual(
            result.policy("similarity_matched_budget").accepted_count, 2
        )
        self.assertGreater(
            result.policy("oracle").mean_reward,
            result.policy("always_use").mean_reward,
        )
        self.assertTrue(result.oracle_has_headroom())


if __name__ == "__main__":
    unittest.main()

