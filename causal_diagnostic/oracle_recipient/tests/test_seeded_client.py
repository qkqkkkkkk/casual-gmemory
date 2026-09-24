from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from causal_diagnostic.oracle_recipient.seeded_client import SeededCachedChat


def _response(*items: tuple[str, float]) -> SimpleNamespace:
    alternatives = [
        SimpleNamespace(token=token, logprob=logprob) for token, logprob in items
    ]
    token = SimpleNamespace(top_logprobs=alternatives)
    return SimpleNamespace(
        choices=[SimpleNamespace(logprobs=SimpleNamespace(content=[token]))]
    )


class BinaryLabelProbabilityTests(unittest.TestCase):
    def test_extracts_and_normalizes_ab_logprobs_then_uses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = SeededCachedChat(
                "test-model",
                Path(directory) / "cache.sqlite",
                experiment_seed=7,
                api_base="http://127.0.0.1:1/v1",
            )
            fake = SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **_: _response((" A", -0.2), ("B", -1.2))
                    )
                )
            )
            client.client = fake
            messages = [{"role": "user", "content": "Return A or B"}]
            first = client.binary_label_probabilities(messages)
            second = client.binary_label_probabilities(messages)
            client.close()

        expected = math.exp(-0.2) / (math.exp(-0.2) + math.exp(-1.2))
        self.assertAlmostEqual(first["supports_probability"], expected)
        self.assertEqual(first, second)
        self.assertEqual(client.calls, 1)
        self.assertEqual(client.cache_hits, 1)

    def test_missing_competing_token_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "both A and B"):
            SeededCachedChat._binary_top_logprobs(
                _response(("A", -0.01), ("C", -4.0))
            )


if __name__ == "__main__":
    unittest.main()
