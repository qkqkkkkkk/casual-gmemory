#!/usr/bin/env python3
"""Paired final-outcome comparison for Always Keep and learned-gate runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
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
) -> dict[str, Any]:
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
        baseline_metadata = baseline[task_id].get("run_metadata", {})
        gate_metadata = gate[task_id].get("run_metadata", {})
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
    return {
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
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = compare(
        args.baseline,
        args.gate,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return result


if __name__ == "__main__":
    main()
