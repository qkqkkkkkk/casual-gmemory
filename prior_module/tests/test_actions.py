from __future__ import annotations

import unittest

from prior_module.actions import ActionCanonicalizer, ActionDistancePolicy


class ActionDistanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canonicalizer = ActionCanonicalizer()
        self.policy = ActionDistancePolicy()

    def test_aliases_have_zero_distance(self) -> None:
        left = self.canonicalizer.canonicalize("go to microwave 1")
        right = self.canonicalizer.canonicalize("navigate to the microwave 1")
        self.assertEqual(left.action_type, "go")
        self.assertEqual(right.action_type, "go")
        self.assertEqual(left.destination, right.destination)
        result = self.policy.distance(left, right)
        self.assertEqual(result.distance, 0.0)

    def test_target_change_is_material(self) -> None:
        left = self.canonicalizer.canonicalize("heat apple 1 with microwave 1")
        right = self.canonicalizer.canonicalize("heat apple 2 with microwave 1")
        result = self.policy.distance(left, right)
        self.assertEqual(result.distance, 0.5)
        self.assertEqual(result.reason, "different_target_or_destination")

    def test_context_can_upgrade_to_critical_distance(self) -> None:
        left = self.canonicalizer.canonicalize("heat apple 1 with microwave 1")
        right = self.canonicalizer.canonicalize("heat apple 2 with microwave 1")
        result = self.policy.distance(left, right, {"critical_objects": ["apple 1"]})
        self.assertEqual(result.distance, 1.0)

    def test_invalid_text_is_not_silently_zero(self) -> None:
        left = self.canonicalizer.canonicalize("write a poem")
        right = self.canonicalizer.canonicalize("sing a song")
        result = self.policy.distance(left, right)
        self.assertEqual(result.distance, 0.5)
        self.assertEqual(result.reason, "unparsed_action")
