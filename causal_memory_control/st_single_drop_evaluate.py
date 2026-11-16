"""Run an existing QA checkpoint with its separately saved dev-selected policy."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from .interactive_data import sha
from .reevaluate_st_policy import indexed, read, rows
from .runtime_gate import GMemoryExposureGate


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", choices=("fever", "hotpotqa"), required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--reference-log", type=Path, required=True)
    options, remaining = parser.parse_known_args(argv)
    saved = read(options.policy)
    policy = saved["policy"]
    checkpoint = Path(remaining[remaining.index("--checkpoint") + 1])
    actual = {k: float(remaining[remaining.index("--" + k.replace("_", "-")) + 1]) for k in ("kappa", "delta", "max_drops")}
    if (saved["dataset"] != options.dataset or saved["selection_scope"] != "original_dev_only"
            or policy["rule"] != "single_drop_calibrated" or policy["max_drops"] not in (0, 1)
            or actual != {k: policy[k] for k in actual}
            or str(checkpoint.resolve()) != saved["checkpoint"] or sha(checkpoint) != saved["checkpoint_sha256"]):
        raise ValueError("runtime arguments do not match the frozen dev-selected single-drop policy")
    reference = indexed(rows(options.reference_log))
    runner = importlib.import_module(f"causal_memory_control.{options.dataset}_experiment")
    original = runner.GMemoryExposureGate

    class AuditedSingleGate(GMemoryExposureGate):
        def filter_retrieval(self, *args, **kwargs):
            result = super().filter_retrieval(*args, **kwargs)
            expected = reference[str(self._task_id)]
            actual = [d.to_dict() for d in self._task_decisions]
            fields = ("memory_id", "memory_type", "memory_content", "memory_metadata", "retrieval_rank", "prediction")
            if (self._retrieval_fingerprint != expected["retrieval_fingerprint"]
                    or [[d.get(k) for k in fields] for d in actual] !=
                       [[d.get(k) for k in fields] for d in expected["decisions"]]):
                raise ValueError(f"single-drop comparison retrieval/predictions changed: {self._task_id}")
            if sum(d["action"].upper() == "DROP" for d in actual) > policy["max_drops"]:
                raise ValueError("single-drop runtime budget exceeded")
            return result

    try:
        runner.GMemoryExposureGate = AuditedSingleGate
        return runner.main(remaining)
    finally:
        runner.GMemoryExposureGate = original


if __name__ == "__main__":
    main()
