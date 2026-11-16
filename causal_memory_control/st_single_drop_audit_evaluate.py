"""Numerically aware, decision-preserving audit for existing ST evaluations.

The legacy entry remains immutable because completed calibration runs pin its
source hash. This entry records differences before allowing memory exposure.
"""
from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path

from .interactive_data import digest, save, sha
from .reevaluate_st_policy import indexed, read, rows
from .runtime_gate import GMemoryExposureGate
from .st_decision_calibration import choose


SCHEMA = "semantic-st-single-drop-numeric-audit-v1"
ATOL = 1e-6
RTOL = 1e-6
RETRIEVAL_FIELDS = ("memory_id", "memory_type", "memory_content", "memory_metadata", "retrieval_rank")
NUMERIC_FIELDS = ("q_use", "q_drop", "utility", "uncertainty")
AUDIT_PROTOCOL = {"schema": SCHEMA, "atol": ATOL, "rtol": RTOL,
    "retrieval": "exact_fingerprint_candidate_order_text_metadata_rank",
    "predictions": "finite_numeric_tolerance; exact_source_calibration_sample_count",
    "deletion_choice": "must_equal_reference_predictions_under_the_same_new_policy",
    "substantive_mismatch": "save_details_and_fail_before_exposure"}


def prediction_differences(actual, expected):
    differences, invalid = [], []
    if set(actual) != set(expected) or not set(NUMERIC_FIELDS) <= set(actual):
        return [{"field": "prediction_keys", "actual": sorted(actual), "expected": sorted(expected)}], ["prediction_keys"]
    for field in sorted(actual):
        left, right = actual[field], expected[field]
        if field in NUMERIC_FIELDS:
            numeric = all(type(v) in (int, float) and math.isfinite(v) for v in (left, right))
            close = numeric and math.isclose(left, right, abs_tol=ATOL, rel_tol=RTOL)
            if left != right or not close:
                differences.append({"field": field,
                    "actual": left if numeric else repr(left), "expected": right if numeric else repr(right),
                    "absolute_difference": abs(left - right) if numeric else None, "within_tolerance": close})
            if not close:
                invalid.append(field)
        elif left != right:
            # A tiny numerical change across zero may change only this derived
            # label. The policy's actual deletion choice is audited separately.
            near_zero = (field == "utility_class" and all(type(p.get("utility")) in (int, float)
                and math.isfinite(p["utility"]) and abs(p["utility"]) <= ATOL for p in (actual, expected)))
            differences.append({"field": field, "actual": left, "expected": right, "within_tolerance": near_zero})
            if not near_zero:
                invalid.append(field)
    return differences, invalid


def audit_decisions(actual, expected, policy, *, fingerprint, task_id):
    report = {"schema": SCHEMA, "task_id": str(task_id), "protocol": AUDIT_PROTOCOL,
        "policy": policy, "actual_fingerprint": fingerprint,
        "expected_fingerprint": expected.get("retrieval_fingerprint"),
        "retrieval_differences": [], "prediction_differences": [], "errors": []}
    reference = expected.get("decisions", [])
    if fingerprint != expected.get("retrieval_fingerprint"):
        report["errors"].append("retrieval_fingerprint_changed")
    if len(actual) != len(reference):
        report["errors"].append("candidate_count_changed")
    predicted_records = []
    for index, (left, right) in enumerate(zip(actual, reference)):
        fields = [k for k in RETRIEVAL_FIELDS if left.get(k) != right.get(k)]
        if fields:
            report["retrieval_differences"].append({"index": index, "fields": fields,
                "actual": {k: left.get(k) for k in fields}, "expected": {k: right.get(k) for k in fields}})
        changes, invalid = prediction_differences(left.get("prediction", {}), right.get("prediction", {}))
        if changes:
            report["prediction_differences"].append({"index": index, "memory_id": left.get("memory_id"),
                "differences": changes, "invalid_fields": invalid})
        if invalid:
            report["errors"].append(f"prediction_changed_candidate_{index}")
        prediction = right.get("prediction", {})
        if not invalid:
            predicted_records.append({"memory_id": right["memory_id"], "order": index,
                "predicted_u": prediction["utility"], "uncertainty": prediction["uncertainty"],
                "calibrated": prediction.get("calibrated", False),
                "in_scope": left.get("reason") not in ("candidate_kind_out_of_scope", "candidate_rank_out_of_scope")})
    if report["retrieval_differences"]:
        report["errors"].append("retrieval_candidates_changed")
    dropped = sorted(r["memory_id"] for r in actual if r["action"].upper() == "DROP")
    report["actual_drop_ids"] = dropped
    if len(dropped) > policy["max_drops"]:
        report["errors"].append("single_drop_budget_exceeded")
    if not report["errors"]:
        expected_drops = sorted(r["memory_id"] for r in choose(predicted_records, policy))
        report["reference_prediction_drop_ids_under_new_policy"] = expected_drops
        if dropped != expected_drops:
            report["errors"].append("numerical_difference_changes_deletion_choice")
    report["status"] = "failed" if report["errors"] else "tolerated_numeric_difference" if report["prediction_differences"] else "exact_match"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", choices=("fever", "hotpotqa"), required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--reference-log", type=Path, required=True)
    options, remaining = parser.parse_known_args(argv)
    saved = read(options.policy)
    policy = saved["policy"]
    checkpoint = Path(remaining[remaining.index("--checkpoint") + 1])
    parameters = {k: float(remaining[remaining.index("--" + k.replace("_", "-")) + 1]) for k in ("kappa", "delta", "max_drops")}
    if (saved["dataset"] != options.dataset or saved["selection_scope"] != "original_dev_only"
            or policy["rule"] != "single_drop_calibrated" or policy["max_drops"] not in (0, 1)
            or parameters != {k: policy[k] for k in parameters}
            or str(checkpoint.resolve()) != saved["checkpoint"] or sha(checkpoint) != saved["checkpoint_sha256"]):
        raise ValueError("runtime arguments do not match the frozen dev-selected single-drop policy")
    reference = indexed(rows(options.reference_log))
    output = Path(remaining[remaining.index("--output-dir") + 1]) / "single_drop_audit"
    runner = importlib.import_module(f"causal_memory_control.{options.dataset}_experiment")
    original = runner.GMemoryExposureGate

    class AuditedSingleGate(GMemoryExposureGate):
        def filter_retrieval(self, *args, **kwargs):
            result = super().filter_retrieval(*args, **kwargs)
            task_id = str(self._task_id)
            if task_id not in reference:
                raise ValueError(f"single-drop reference is missing task {task_id}")
            report = audit_decisions([d.to_dict() for d in self._task_decisions], reference[task_id], policy,
                                     fingerprint=self._retrieval_fingerprint, task_id=task_id)
            if report["status"] != "exact_match":
                path = output / (digest(task_id) + ".json")
                save(path, report)
                if report["status"] == "failed":
                    raise ValueError(f"single-drop audit failed: task={task_id}; "
                                     f"errors={report['errors']}; details={path}")
            return result

    try:
        runner.GMemoryExposureGate = AuditedSingleGate
        return runner.main(remaining)
    finally:
        runner.GMemoryExposureGate = original


if __name__ == "__main__":
    main()
