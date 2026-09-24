from __future__ import annotations

from pathlib import Path
import unittest

from causal_diagnostic.hotpotqa_oracle_recipient.run_all import (
    _base_runner_command,
    parse_args,
)


class HotpotRunAllTests(unittest.TestCase):
    def test_default_command_targets_only_hotpot_package(self) -> None:
        args = parse_args(
            (
                "--data",
                "data/hotpotqa/dev.json",
                "--support-count",
                "20",
                "--evaluation-count",
                "50",
            )
        )
        command = _base_runner_command(args, 4, Path("output"))
        rendered = " ".join(command)
        self.assertIn(
            "causal_diagnostic.hotpotqa_oracle_recipient.run_experiment",
            rendered,
        )
        self.assertNotIn("fever_oracle_recipient", rendered)
        self.assertNotIn("label-probabilities", rendered)


if __name__ == "__main__":
    unittest.main()
