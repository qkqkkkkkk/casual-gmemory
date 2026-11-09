"""Evaluate a new ST encoder/head, auditing retrieval against the old run."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from .reevaluate_st_policy import indexed, rows
from .st_policy import PositiveSTGate, RULES, resolve_rule


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", choices=("fever", "hotpotqa"), required=True)
    parser.add_argument("--reference-log", type=Path, required=True)
    parser.add_argument("--st-deletion-rule", choices=RULES, default="dataset_default")
    options, remaining = parser.parse_known_args(argv)
    reference = indexed(rows(options.reference_log))
    runner = importlib.import_module(f"causal_memory_control.{options.dataset}_experiment")
    original = runner.GMemoryExposureGate
    positive = resolve_rule(options.st_deletion_rule, options.dataset) == "positive_unlimited"
    selected = PositiveSTGate if positive else original

    class AuditedEncoderGate(selected):
        def filter_retrieval(self, *args, **kwargs):
            result = super().filter_retrieval(*args, **kwargs)
            expected = reference[str(self._task_id)]
            # Predictions/actions are allowed to change; the retrieved texts are not.
            actual = [d.to_dict() for d in self._task_decisions]
            fields = ("memory_id", "memory_type", "memory_content", "memory_metadata", "retrieval_rank")
            if (self._retrieval_fingerprint != expected["retrieval_fingerprint"]
                    or [[d.get(k) for k in fields] for d in actual] !=
                       [[d.get(k) for k in fields] for d in expected["decisions"]]):
                raise ValueError(f"encoder comparison retrieval changed: {self._task_id}")
            return result

    if positive:
        remaining += ["--kappa", "0", "--delta", "0", "--max-drops", "-1"]
    try:
        runner.GMemoryExposureGate = AuditedEncoderGate
        return runner.main(remaining)
    finally:
        runner.GMemoryExposureGate = original


if __name__ == "__main__":
    main()
