#!/usr/bin/env python3
"""Paired final-outcome comparison for Always Keep and learned-gate runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


PHYSICAL_AUDIT_FIELDS = frozenset({"memory_sha256"})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require-no-drop-match", action="store_true")
    parser.add_argument("--require-policy-success",action="store_true")
    parser.add_argument("--replica-manifest", type=Path, default=None,
        help="Explicit verified Ollama replica evidence; allows only its recorded endpoint pair")
    return parser.parse_args(argv)


def _load(path: Path) -> dict[str, Mapping[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indexed = {}
    for row in rows:
        key = str(row["task_id"])
        if key in indexed:
            raise ValueError(f"duplicate task_id {key!r} in {path}")
        indexed[key] = row
    return indexed


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _semantic_run_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Drop byte-level Chroma audit fields while retaining experiment identity."""
    return {
        key: item
        for key, item in value.items()
        if key not in PHYSICAL_AUDIT_FIELDS
    }


def _paired_interval(
    differences: Sequence[float], *, samples: int, seed: int
) -> tuple[float, float]:
    if not differences:
        return 0.0, 0.0
    randomizer = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(
            _mean([randomizer.choice(differences) for _ in differences])
        )
    estimates.sort()
    return (
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    )


def compare(
    baseline_path: Path,
    gate_path: Path,
    *,
    bootstrap_samples: int = 10000,
    seed: int = 42,
    replica_manifest: Path | None = None,
) -> dict[str, Any]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    proof = None
    if replica_manifest is not None:
        from .evaluation_replicas import validate_manifest

        proof = validate_manifest(replica_manifest, baseline_path)
    baseline = _load(baseline_path)
    gate = _load(gate_path)
    baseline_ids, gate_ids = set(baseline), set(gate)
    if baseline_ids != gate_ids:
        raise ValueError(
            "paired logs contain different task IDs: "
            f"baseline_only={sorted(baseline_ids - gate_ids)}, "
            f"gate_only={sorted(gate_ids - baseline_ids)}"
        )
    task_ids = sorted(baseline_ids)
    if not task_ids:
        raise ValueError("the two logs have no paired task IDs")
    for task_id in task_ids:
        baseline_metadata = _semantic_run_metadata(
            baseline[task_id].get("run_metadata", {})
        )
        gate_metadata = _semantic_run_metadata(
            gate[task_id].get("run_metadata", {})
        )
        if proof is not None:
            from .evaluation_replicas import paired_metadata

            baseline_metadata, gate_metadata = paired_metadata(baseline_metadata, gate_metadata, proof)
        if baseline_metadata != gate_metadata:
            raise ValueError(
                f"run metadata differs for paired task {task_id}: "
                f"baseline={baseline_metadata}, gate={gate_metadata}"
            )
    base_rewards = [float(baseline[key]["reward"]) for key in task_ids]
    gate_rewards = [float(gate[key]["reward"]) for key in task_ids]
    base_success = [float(bool(baseline[key]["done"])) for key in task_ids]
    gate_success = [float(bool(gate[key]["done"])) for key in task_ids]
    reward_delta = [right - left for left, right in zip(base_rewards, gate_rewards)]
    success_delta = [right - left for left, right in zip(base_success, gate_success)]
    keep_rates = [float(gate[key].get("keep_rate", 1.0)) for key in task_ids]
    reward_ci = _paired_interval(reward_delta, samples=bootstrap_samples, seed=seed)
    success_ci = _paired_interval(
        success_delta, samples=bootstrap_samples, seed=seed + 1
    )
    no_drop_mismatches = []
    retrieval_mismatches = []
    for task_id in task_ids:
        b, g = baseline[task_id], gate[task_id]
        if ("retrieval_fingerprint" in b and "retrieval_fingerprint" in g
                and b["retrieval_fingerprint"] != g["retrieval_fingerprint"]):
            retrieval_mismatches.append(task_id)
        if int(g.get("dropped_count", 0)) != 0:
            continue
        fields = []
        for key in ("reward", "done", "retrieval_fingerprint"):
            if key in b and key in g and b[key] != g[key]:
                fields.append(key)
        b_out, g_out = b.get("outcome", {}), g.get("outcome", {})
        if "prediction" in b_out and "prediction" in g_out and b_out["prediction"] != g_out["prediction"]:
            fields.append("prediction")
        if fields:
            no_drop_mismatches.append({"task_id": task_id, "different_fields": fields})
    result = {
        "paired_tasks": len(task_ids),
        "baseline_mean_reward": _mean(base_rewards),
        "gate_mean_reward": _mean(gate_rewards),
        "reward_delta": _mean(reward_delta),
        "reward_delta_95ci": list(reward_ci),
        "baseline_completion_rate": _mean(base_success),
        "gate_completion_rate": _mean(gate_success),
        "completion_rate_delta": _mean(success_delta),
        "completion_rate_delta_95ci": list(success_ci),
        "gate_mean_keep_rate": _mean(keep_rates),
        "tasks_with_drop": sum(int(gate[key].get("dropped_count", 0)) > 0 for key in task_ids),
        "no_drop_mismatches": no_drop_mismatches,
        "retrieval_mismatches": retrieval_mismatches,
        "paired_control_valid": not no_drop_mismatches and not retrieval_mismatches,
        "baseline_log_sha256": file_sha256(baseline_path),
        "gate_log_sha256": file_sha256(gate_path),
        "drop_improved_tasks":sum(gate[key]["reward"]>baseline[key]["reward"] and gate[key].get("dropped_count",0)>0 for key in task_ids),
        "drop_harmed_tasks":sum(gate[key]["reward"]<baseline[key]["reward"] and gate[key].get("dropped_count",0)>0 for key in task_ids),
        **{metric:sum(float(gate[key].get("outcome",{}).get(metric,0)) for key in task_ids) for metric in (
            "jev_errors","jev_calls","jev_cache_hits","jev_latency_seconds","jev_input_tokens","jev_output_tokens","jev_reported_cost"
        )},
    }
    result["policy_valid"] = result["jev_errors"]==0
    if proof is not None:
        result["evaluation_replica_provenance"] = {
            "manifest": str(Path(replica_manifest).resolve()),
            "manifest_sha256": file_sha256(Path(replica_manifest)),
            "source_endpoint": proof["source_endpoint"], "target_endpoint": proof["target_endpoint"],
            "replicas": proof["replicas"], "limitation": proof["limitation"]}
    if baseline[task_ids[0]].get("run_metadata", {}).get("benchmark") == "HotpotQA_distractor_offline":
        result.update(
            baseline_answer_f1=result["baseline_mean_reward"],
            gate_answer_f1=result["gate_mean_reward"],
            answer_f1_delta=result["reward_delta"],
            answer_f1_delta_95ci=result["reward_delta_95ci"],
            baseline_answer_em=result["baseline_completion_rate"],
            gate_answer_em=result["gate_completion_rate"],
            answer_em_delta=result["completion_rate_delta"],
            answer_em_delta_95ci=result["completion_rate_delta_95ci"],
        )
    return result


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = compare(
        args.baseline,
        args.gate,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        replica_manifest=args.replica_manifest,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.require_no_drop_match and not result["paired_control_valid"]:
        raise SystemExit("no-DROP outcomes or original retrievals differ from baseline; inspect comparison.json")
    if args.require_policy_success and not result["policy_valid"]:
        raise SystemExit("Jev API errors occurred; decisions defaulted to KEEP. Fix service settings and retry error tasks.")
    return result


if __name__ == "__main__":
    main()
