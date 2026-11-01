"""Run a QA evaluator with an explicitly selected ST rule, without training."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from .st_policy import PositiveSTGate, RULES, positive_decisions


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", choices=("fever", "hotpotqa"), required=True)
    parser.add_argument("--st-deletion-rule", choices=RULES, default="calibrated")
    parser.add_argument("--st-reference-log", type=Path,
                        help="Audit original candidates and planned decisions before exposing memory")
    args, remaining = parser.parse_known_args(argv)
    runner = importlib.import_module(f"causal_memory_control.{args.dataset}_experiment")
    original = runner.GMemoryExposureGate
    selected = PositiveSTGate if args.st_deletion_rule == "positive_unlimited" else original
    if args.st_reference_log:
        values = [json.loads(line) for line in args.st_reference_log.read_text().splitlines() if line.strip()]
        reference = {str(row["task_id"]): row for row in values}
        if not reference or len(reference) != len(values):
            raise ValueError("empty or duplicate original ST evaluation task IDs")

        class AuditedGate(selected):
            def filter_retrieval(self, *call_args, **kwargs):
                result = super().filter_retrieval(*call_args, **kwargs)
                expected = reference[str(self._task_id)]
                decisions = (positive_decisions(expected["decisions"])
                             if args.st_deletion_rule == "positive_unlimited" else expected["decisions"])
                observed = [d.to_dict() for d in self._task_decisions]
                if (self._retrieval_fingerprint != expected["retrieval_fingerprint"]
                        or [(d["memory_id"], d["action"].upper()) for d in decisions] !=
                           [(d["memory_id"], d["action"].upper()) for d in observed]):
                    raise ValueError(f"ST runtime retrieval/decisions differ from the saved-score plan: {self._task_id}")
                return result
        selected = AuditedGate
    if args.st_deletion_rule == "positive_unlimited":
        # These are also written into the host run manifest. The class override
        # additionally handles U == 0, which the old strict negative rule keeps.
        remaining += ["--kappa", "0", "--delta", "0", "--max-drops", "-1"]
    try:
        runner.GMemoryExposureGate = selected
        return runner.main(remaining)
    finally:
        runner.GMemoryExposureGate = original


if __name__ == "__main__":
    main()
