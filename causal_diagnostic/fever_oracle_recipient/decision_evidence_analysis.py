"""Post-hoc continuous FEVER utility from final decision evidence-page F1.

This analysis requires no new rollout.  It scores the final decision node's
``Evidence[...]`` pages and defines recipient utility as
``F1(only_recipient) - F1(global_drop)`` on matched repeats.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from causal_diagnostic.fever_oracle_recipient.local_metric import (
    score_evidence_pages,
)
from causal_diagnostic.oracle_recipient.analysis import AnalysisError, build_events


SCHEMA = "fever-decision-evidence-utility-v1"


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise AnalysisError("mean requires at least one value")
    return sum(float(value) for value in values) / len(values)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _event_bootstrap(
    values_by_event: Sequence[Sequence[float]],
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    event_means = [_mean(values) for values in values_by_event]
    rng = random.Random(seed)
    draws = [
        _mean(
            [event_means[rng.randrange(len(event_means))] for _ in event_means]
        )
        for _ in range(samples)
    ]
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "estimate": _mean(event_means),
        "ci_low": _percentile(draws, alpha),
        "ci_high": _percentile(draws, 1.0 - alpha),
        "confidence_level": confidence_level,
        "bootstrap_samples": samples,
        "bootstrap_unit": "held_out_task_candidate_event",
        "event_count": len(event_means),
    }


def _sign(value: float, delta: float) -> int:
    return 1 if value > delta else -1 if value < -delta else 0


def _load_and_score(results_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = results_dir / "run_manifest.json"
    rows_path = results_dir / "branches.jsonl"
    if not manifest_path.is_file() or not rows_path.is_file():
        raise AnalysisError(
            f"{results_dir} must contain run_manifest.json and branches.jsonl"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for line_number, line in enumerate(
        rows_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        original = json.loads(line)
        row = copy.deepcopy(original)
        outcome = row.get("outcome", {})
        if "decision_evidence_f1" not in outcome:
            try:
                raw_output = outcome["trace"][-1]["decision"]["raw_output"]
                gold_sets = row["task"]["gold_evidence_page_sets"]
            except (KeyError, IndexError, TypeError) as exc:
                raise AnalysisError(
                    f"line {line_number} lacks final decision output or gold evidence"
                ) from exc
            metric = score_evidence_pages(raw_output, gold_sets)
            outcome["decision_evidence_metric"] = metric
            outcome["decision_evidence_f1"] = float(metric["f1"])
        rows.append(row)
    if not rows:
        raise AnalysisError(f"{rows_path} is empty")
    return manifest, rows


def _analyze_run(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    delta: float,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    recipients = tuple(str(value) for value in manifest["design"]["recipients"])
    events = build_events(rows, recipients)
    pair_rows = []
    by_event: list[list[float]] = []
    for event in events:
        event_values = []
        for recipient in recipients:
            control = event["outcomes"]["global_drop"]
            treated = event["outcomes"][f"only_{recipient}"]
            control_samples = control["samples"]["decision_evidence_f1"]
            treated_samples = treated["samples"]["decision_evidence_f1"]
            deltas = [
                float(right) - float(left)
                for left, right in zip(control_samples, treated_samples)
            ]
            utility = _mean(deltas)
            local_control = control.get("local_metric_samples", {}).get(recipient)
            local_treated = treated.get("local_metric_samples", {}).get(recipient)
            local_utility = None
            if local_control is not None and local_treated is not None:
                local_utility = _mean(
                    [
                        float(right) - float(left)
                        for left, right in zip(local_control, local_treated)
                    ]
                )
            half_signs: list[int] = []
            midpoint = len(deltas) // 2
            if len(deltas) >= 4 and midpoint > 0:
                half_signs = [
                    _sign(_mean(deltas[:midpoint]), delta),
                    _sign(_mean(deltas[midpoint:]), delta),
                ]
            pair_rows.append(
                {
                    "event_id": event["event_id"],
                    "task_id": event["task_id"],
                    "candidate_id": event["candidate_id"],
                    "recipient": recipient,
                    "utility": utility,
                    "sign": _sign(utility, delta),
                    "local_utility": local_utility,
                    "local_sign": (
                        _sign(local_utility, delta)
                        if local_utility is not None
                        else None
                    ),
                    "repeat_deltas": deltas,
                    "split_half_signs": half_signs,
                }
            )
            event_values.append(utility)
        by_event.append(event_values)

    signs = [int(row["sign"]) for row in pair_rows]
    local_available = [row for row in pair_rows if row["local_sign"] is not None]
    split_available = [row for row in pair_rows if len(row["split_half_signs"]) == 2]
    summary: dict[str, Any] = {
        "run_label": f"seed{manifest.get('run', {}).get('sample_seed_base', 'unknown')}",
        "event_count": len(events),
        "recipient_count": len(recipients),
        "event_recipient_pairs": len(pair_rows),
        "mean_utility": _event_bootstrap(
            by_event,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed,
        ),
        "positive_pairs": signs.count(1),
        "neutral_pairs": signs.count(0),
        "negative_pairs": signs.count(-1),
        "nonzero_pairs": signs.count(1) + signs.count(-1),
        "local_and_decision_nonzero_pairs": sum(
            int(row["sign"] != 0 and row["local_sign"] != 0)
            for row in local_available
        ),
        "local_decision_direct_opposites": sum(
            int(int(row["sign"]) * int(row["local_sign"]) == -1)
            for row in local_available
        ),
        "split_half_status": "ok" if len(split_available) == len(pair_rows) else "insufficient_repeats",
        "split_half_sign_agreement": (
            _mean(
                [
                    float(row["split_half_signs"][0] == row["split_half_signs"][1])
                    for row in split_available
                ]
            )
            if split_available
            else None
        ),
        "split_half_direct_reversal_rate": (
            _mean(
                [
                    float(row["split_half_signs"][0] * row["split_half_signs"][1] == -1)
                    for row in split_available
                ]
            )
            if split_available
            else None
        ),
        "pair_rows": pair_rows,
    }
    summary["per_recipient"] = {
        recipient: {
            "mean_utility": _event_bootstrap(
                [[float(row["utility"])] for row in pair_rows if row["recipient"] == recipient],
                samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=seed + index + 1,
            ),
            "positive": sum(
                row["recipient"] == recipient and row["sign"] == 1
                for row in pair_rows
            ),
            "neutral": sum(
                row["recipient"] == recipient and row["sign"] == 0
                for row in pair_rows
            ),
            "negative": sum(
                row["recipient"] == recipient and row["sign"] == -1
                for row in pair_rows
            ),
        }
        for index, recipient in enumerate(recipients)
    }
    return summary


def _compare_runs(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    def keyed(run: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
        return {
            (str(row["event_id"]), str(row["recipient"])): row
            for row in run["pair_rows"]
        }

    left_rows, right_rows = keyed(left), keyed(right)
    shared = sorted(set(left_rows) & set(right_rows))
    if not shared:
        raise AnalysisError("runs have no shared event-recipient pairs")
    sign_pairs = [
        (int(left_rows[key]["sign"]), int(right_rows[key]["sign"]))
        for key in shared
    ]
    return {
        "shared_event_recipient_pairs": len(shared),
        "strict_sign_consistency": _mean([float(a == b) for a, b in sign_pairs]),
        "direct_opposite_sign_rate": _mean([float(a * b == -1) for a, b in sign_pairs]),
        "nonzero_neutral_transition_rate": _mean(
            [float((a == 0) != (b == 0)) for a, b in sign_pairs]
        ),
        "stable_nonzero_rate": _mean(
            [float(a == b and a != 0) for a, b in sign_pairs]
        ),
    }


def _fmt_ci(value: Mapping[str, Any]) -> str:
    return (
        f"{float(value['estimate']):+.4f} "
        f"[{float(value['ci_low']):+.4f}, {float(value['ci_high']):+.4f}]"
    )


def _write_outputs(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "decision_evidence_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "run",
        "event_id",
        "task_id",
        "candidate_id",
        "recipient",
        "utility",
        "sign",
        "local_utility",
        "local_sign",
        "repeat_deltas",
        "split_half_signs",
    ]
    with (output_dir / "decision_evidence_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in payload["runs"]:
            for row in run["pair_rows"]:
                rendered = {"run": run["run_label"], **row}
                rendered["repeat_deltas"] = json.dumps(row["repeat_deltas"])
                rendered["split_half_signs"] = json.dumps(row["split_half_signs"])
                writer.writerow({field: rendered.get(field) for field in fields})

    lines = [
        "# FEVER decision-evidence continuous utility",
        "",
        "`U_decision-evidence = F1(final decision evidence | only recipient) - "
        "F1(final decision evidence | global drop)`.",
        "",
        "This is a zero-rollout evidence-page sensitivity analysis; it is not "
        "the model's gold-label probability.",
        "",
        "| Run | Events | Pairs | Mean utility [95% CI] | Positive | Neutral | Negative | Local & decision nonzero | Opposite | Split-half agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in payload["runs"]:
        split = run["split_half_sign_agreement"]
        lines.append(
            f"| {run['run_label']} | {run['event_count']} | "
            f"{run['event_recipient_pairs']} | {_fmt_ci(run['mean_utility'])} | "
            f"{run['positive_pairs']} | {run['neutral_pairs']} | "
            f"{run['negative_pairs']} | {run['local_and_decision_nonzero_pairs']} | "
            f"{run['local_decision_direct_opposites']} | "
            f"{split:.1%} |"
        )
    if payload.get("independent_retest") is not None:
        value = payload["independent_retest"]
        lines.extend(
            [
                "",
                "## Independent-seed reproducibility",
                "",
                f"- Strict sign consistency: {value['strict_sign_consistency']:.1%}",
                f"- Direct opposite-sign rate: {value['direct_opposite_sign_rate']:.1%}",
                "- Nonzero/neutral transition rate: "
                f"{value['nonzero_neutral_transition_rate']:.1%}",
                f"- Stable nonzero rate: {value['stable_nonzero_rate']:.1%}",
            ]
        )
    (output_dir / "decision_evidence_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def analyze(
    results: Path,
    *,
    retest_results: Path | None = None,
    output_dir: Path,
    delta: float = 0.0,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    if bootstrap_samples < 100:
        raise AnalysisError("bootstrap_samples must be at least 100")
    manifest, rows = _load_and_score(results)
    runs = [
        _analyze_run(
            manifest,
            rows,
            delta=delta,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed,
        )
    ]
    if retest_results is not None:
        retest_manifest, retest_rows = _load_and_score(retest_results)
        if retest_manifest.get("design_hash") != manifest.get("design_hash"):
            raise AnalysisError("retest design_hash differs from the first run")
        runs.append(
            _analyze_run(
                retest_manifest,
                retest_rows,
                delta=delta,
                bootstrap_samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=seed + 10_000,
            )
        )
    payload = {
        "analysis_schema": SCHEMA,
        "metric": "FEVER gold evidence-page F1 from final decision output",
        "utility_definition": "only_recipient_minus_global_drop",
        "delta": delta,
        "runs": runs,
        "independent_retest": _compare_runs(runs[0], runs[1]) if len(runs) == 2 else None,
    }
    _write_outputs(output_dir, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--retest-results", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    payload = analyze(
        args.results,
        retest_results=args.retest_results,
        output_dir=args.output_dir,
        delta=args.delta,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    output = args.output_dir / "decision_evidence_analysis.json"
    print(
        json.dumps(
            {
                "output": str(output),
                "runs": [
                    {
                        "run_label": run["run_label"],
                        "event_count": run["event_count"],
                        "event_recipient_pairs": run["event_recipient_pairs"],
                        "mean_utility": run["mean_utility"],
                        "positive_pairs": run["positive_pairs"],
                        "neutral_pairs": run["neutral_pairs"],
                        "negative_pairs": run["negative_pairs"],
                    }
                    for run in payload["runs"]
                ],
                "independent_retest": payload["independent_retest"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return output


if __name__ == "__main__":
    main()
