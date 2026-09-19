from __future__ import annotations

from types import SimpleNamespace
import unittest

from causal_diagnostic.fever_oracle_recipient.sampling import (
    configure_fever_sampling,
)


def _node() -> SimpleNamespace:
    return SimpleNamespace(
        reasoning_config=SimpleNamespace(
            temperature=0.0,
            stop_strs=["\n"],
        )
    )


class FeverSamplingTests(unittest.TestCase):
    def test_multiline_answer_is_not_stopped_after_evidence_line(self):
        workers = {"solver_0": _node(), "solver_1": _node()}
        decision = _node()
        mas = SimpleNamespace(
            _agent_nodes=workers,
            _decision_node=decision,
        )

        configure_fever_sampling(mas, 0.7)

        for node in (*workers.values(), decision):
            self.assertEqual(node.reasoning_config.temperature, 0.7)
            self.assertIsNone(node.reasoning_config.stop_strs)


if __name__ == "__main__":
    unittest.main()
