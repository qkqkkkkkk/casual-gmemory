"""Event-clustered RQ2 oracle and RQ3 recipient analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


ANALYSIS_SCHEMA = "gmemory-macnet-oracle-recipient-analysis-v2"


class AnalysisError(ValueError):
    pass


def _sign(value: float, epsilon: float) -> int:
    return 1 if value > epsilon else -1 if value < -epsilon else 0


def _label(value: int) -> str:
    return {1: "positive", 0: "neutral", -1: "negative"}[value]


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise AnalysisError("mean requires at least one value")
    return sum(float(value) for value in values) / len(values)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise AnalysisError("percentile requires at least one value")
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap(
    values: Sequence[float],
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    numeric = tuple(float(value) for value in values)
    if not numeric:
        raise AnalysisError("bootstrap requires at least one event")
    rng = random.Random(seed)
    draws = [
        _mean([numeric[rng.randrange(len(numeric))] for _ in numeric])
        for _ in range(samples)
    ]
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "estimate": _mean(numeric),
        "ci_low": _percentile(draws, alpha),
        "ci_high": _percentile(draws, 1.0 - alpha),
        "confidence_level": confidence_level,
        "bootstrap_samples": samples,
        "event_count": len(numeric),
    }


def _load_results(results_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = results_dir / "run_manifest.json"
    rows_path = results_dir / "branches.jsonl"
    if not manifest_path.is_file() or not rows_path.is_file():
        raise AnalysisError(
            f"{results_dir} must contain run_manifest.json and branches.jsonl"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise AnalysisError(f"{rows_path} is empty")
    for row in rows:
        if row.get("design_hash") != manifest.get("design_hash"):
            raise AnalysisError("branch design_hash differs from manifest")
        if row.get("run_hash") != manifest.get("run_hash"):
            raise AnalysisError("branch run_hash differs from manifest")
    return manifest, rows


def build_events(
    rows: Sequence[Mapping[str, Any]], recipients: Sequence[str]
) -> list[dict[str, Any]]:
    recipients = tuple(recipients)
    required = ("use_all", "global_drop", *(f"drop_{r}" for r in recipients))
    grouped: dict[tuple[int, str], dict[str, dict[int, Mapping[str, Any]]]] = {}
    for row in rows:
        key = (int(row["task_id"]), str(row["candidate"]["candidate_id"]))
        condition = str(row["condition"])
        repeat = int(row["repeat_index"])
        by_condition = grouped.setdefault(key, {})
        repeats = by_condition.setdefault(condition, {})
        if repeat in repeats:
            raise AnalysisError(f"duplicate branch row for {key, condition, repeat}")
        repeats[repeat] = row

    events = []
    for (task_id, candidate_id), conditions in sorted(grouped.items()):
        missing = set(required) - set(conditions)
        if missing:
            raise AnalysisError(
                f"event {(task_id, candidate_id)} is missing {sorted(missing)}"
            )
        repeat_sets = [set(conditions[name]) for name in required]
        if any(values != repeat_sets[0] for values in repeat_sets[1:]):
            raise AnalysisError(
                f"event {(task_id, candidate_id)} has unmatched repeats"
            )
        repeat_indices = sorted(repeat_sets[0])
        outcome_names = ("team_score", "success", "reward", "steps")
        outcomes: dict[str, Any] = {}
        for condition in required:
            samples = {
                name: [
                    float(conditions[condition][repeat]["outcome"][name])
                    for repeat in repeat_indices
                ]
                for name in outcome_names
            }
            outcomes[condition] = {
                name: _mean(values) for name, values in samples.items()
            }
            outcomes[condition]["samples"] = samples
        for repeat in repeat_indices:
            seeds = {
                int(conditions[condition][repeat]["sample_seed"])
                for condition in required
            }
            if len(seeds) != 1:
                raise AnalysisError(
                    f"event {(task_id, candidate_id)} repeat {repeat} has unmatched seeds"
                )
        first = conditions["use_all"][repeat_indices[0]]
        events.append(
            {
                "event_id": f"pddl-{task_id}-{candidate_id}",
                "task_id": task_id,
                "candidate_id": candidate_id,
                "candidate": dict(first["candidate"]),
                "repeat_count": len(repeat_indices),
                "outcomes": outcomes,
            }
        )
    if not events:
        raise AnalysisError("no complete events found")
    return events


def _oracle_summary(
    events: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    delta: float,
    estimate: Any,
) -> dict[str, Any]:
    always_use = []
    always_drop = []
    oracle = []
    decisions = []
    gains = []
    for event in events:
        q_use = float(event["outcomes"]["use_all"][metric])
        q_drop = float(event["outcomes"]["global_drop"][metric])
        use = q_use - q_drop > delta
        selected = q_use if use else q_drop
        always_use.append(q_use)
        always_drop.append(q_drop)
        oracle.append(selected)
        decisions.append(use)
        gains.append(selected - q_use)
    return {
        "metric": metric,
        "always_use_mean": _mean(always_use),
        "always_global_drop_mean": _mean(always_drop),
        "oracle_selective_mean": _mean(oracle),
        "oracle_memory_keep_rate": sum(decisions) / len(decisions),
        "oracle_gain_vs_always_use": estimate(gains),
        "oracle_decisions": decisions,
    }


def _recipient_summary(
    events: Sequence[Mapping[str, Any]],
    recipients: Sequence[str],
    *,
    metric: str,
    delta: float,
    estimate: Any,
) -> dict[str, Any]:
    decorated = []
    for event in events:
        q_use = float(event["outcomes"]["use_all"][metric])
        utilities = {
            recipient: q_use
            - float(event["outcomes"][f"drop_{recipient}"][metric])
            for recipient in recipients
        }
        signs = {recipient: _sign(value, delta) for recipient, value in utilities.items()}
        decorated.append(
            {
                "event_id": event["event_id"],
                "task_id": event["task_id"],
                "candidate_id": event["candidate_id"],
                "utilities": utilities,
                "signs": {recipient: _label(value) for recipient, value in signs.items()},
                "direct_sign_flip": 1 in signs.values() and -1 in signs.values(),
                "any_sign_heterogeneity": len(set(signs.values())) > 1,
                "utility_range": max(utilities.values()) - min(utilities.values()),
            }
        )
    per_recipient = {}
    for recipient in recipients:
        values = [float(event["utilities"][recipient]) for event in decorated]
        signs = [_sign(value, delta) for value in values]
        per_recipient[recipient] = {
            "mean_utility": estimate(values),
            "positive_events": signs.count(1),
            "neutral_events": signs.count(0),
            "negative_events": signs.count(-1),
        }
    pairwise = {}
    for left, right in combinations(recipients, 2):
        differences = [
            float(event["utilities"][left]) - float(event["utilities"][right])
            for event in decorated
        ]
        sign_pairs = [
            (
                _sign(float(event["utilities"][left]), delta),
                _sign(float(event["utilities"][right]), delta),
            )
            for event in decorated
        ]
        pairwise[f"{left}_vs_{right}"] = {
            "mean_difference": estimate(differences),
            "mean_absolute_difference": estimate([abs(value) for value in differences]),
            "sign_disagreement_rate": estimate(
                [float(a != b) for a, b in sign_pairs]
            ),
            "direct_opposite_sign_rate": estimate(
                [float(a * b == -1) for a, b in sign_pairs]
            ),
        }
    return {
        "metric": metric,
        "direct_sign_flip_rate": estimate(
            [float(event["direct_sign_flip"]) for event in decorated]
        ),
        "any_sign_heterogeneity_rate": estimate(
            [float(event["any_sign_heterogeneity"]) for event in decorated]
        ),
        "mean_utility_range": estimate(
            [float(event["utility_range"]) for event in decorated]
        ),
        "per_recipient": per_recipient,
        "pairwise": pairwise,
        "events": decorated,
    }


def analyze_run(
    rows: Sequence[Mapping[str, Any]],
    recipients: Sequence[str],
    *,
    delta: float = 0.0,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    if bootstrap_samples < 100:
        raise AnalysisError("bootstrap_samples must be at least 100")
    if not 0.0 < confidence_level < 1.0:
        raise AnalysisError("confidence_level must be in (0,1)")
    events = build_events(rows, recipients)
    next_seed = seed

    def estimate(values: Sequence[float]) -> dict[str, Any]:
        nonlocal next_seed
        next_seed += 1
        return _bootstrap(
            values,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=next_seed,
        )

    rq2 = {
        "primary_team_score": _oracle_summary(
            events, metric="team_score", delta=delta, estimate=estimate
        ),
        "success_sensitivity": _oracle_summary(
            events, metric="success", delta=delta, estimate=estimate
        ),
    }
    rq3 = {
        "primary_team_score": _recipient_summary(
            events, recipients, metric="team_score", delta=delta, estimate=estimate
        ),
        "success_sensitivity": _recipient_summary(
            events, recipients, metric="success", delta=delta, estimate=estimate
        ),
    }
    return {
        "analysis_schema": ANALYSIS_SCHEMA,
        "event_count": len(events),
        "recipient_count": len(recipients),
        "recipients": list(recipients),
        "delta": delta,
        "bootstrap_unit": "held_out_task_candidate_event",
        "rq2": rq2,
        "rq3": rq3,
        "events": events,
    }


def _event_map(analysis: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(event["event_id"]): event for event in analysis["events"]}


def compare_runs(
    current: Mapping[str, Any],
    previous: Mapping[str, Any],
    *,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    if current["recipients"] != previous["recipients"]:
        raise AnalysisError("retest recipients differ")
    recipients = tuple(current["recipients"])
    current_events = _event_map(current)
    previous_events = _event_map(previous)
    shared = sorted(set(current_events) & set(previous_events))
    if not shared:
        raise AnalysisError("retest has no shared events")
    next_seed = seed

    def estimate(values: Sequence[float]) -> dict[str, Any]:
        nonlocal next_seed
        next_seed += 1
        return _bootstrap(
            values,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=next_seed,
        )

    delta = float(current["delta"])
    global_sign_agreement = []
    global_direct_reversal = []
    previous_to_current_gain = []
    current_to_previous_gain = []
    decision_agreement = []
    event_sign_agreement = []
    recipient_sign_agreement = []
    stable_flip = []
    reproducible_direction = []
    current_rq3 = {
        event["event_id"]: event
        for event in current["rq3"]["primary_team_score"]["events"]
    }
    previous_rq3 = {
        event["event_id"]: event
        for event in previous["rq3"]["primary_team_score"]["events"]
    }

    for event_id in shared:
        curr = current_events[event_id]
        prev = previous_events[event_id]
        curr_use = float(curr["outcomes"]["use_all"]["team_score"])
        curr_drop = float(curr["outcomes"]["global_drop"]["team_score"])
        prev_use = float(prev["outcomes"]["use_all"]["team_score"])
        prev_drop = float(prev["outcomes"]["global_drop"]["team_score"])
        curr_decision = curr_use - curr_drop > delta
        prev_decision = prev_use - prev_drop > delta
        decision_agreement.append(float(curr_decision == prev_decision))
        curr_sign = _sign(curr_use - curr_drop, delta)
        prev_sign = _sign(prev_use - prev_drop, delta)
        global_sign_agreement.append(float(curr_sign == prev_sign))
        global_direct_reversal.append(float(curr_sign * prev_sign == -1))
        previous_to_current_gain.append(
            (curr_use if prev_decision else curr_drop) - curr_use
        )
        current_to_previous_gain.append(
            (prev_use if curr_decision else prev_drop) - prev_use
        )

        curr_recipient = current_rq3[event_id]
        prev_recipient = previous_rq3[event_id]
        pairs = [
            (
                {"positive": 1, "neutral": 0, "negative": -1}[
                    prev_recipient["signs"][recipient]
                ],
                {"positive": 1, "neutral": 0, "negative": -1}[
                    curr_recipient["signs"][recipient]
                ],
            )
            for recipient in recipients
        ]
        event_sign_agreement.append(float(all(a == b for a, b in pairs)))
        recipient_sign_agreement.append(sum(a == b for a, b in pairs) / len(pairs))
        left_flip = bool(prev_recipient["direct_sign_flip"])
        right_flip = bool(curr_recipient["direct_sign_flip"])
        stable_flip.append(float(left_flip and right_flip))
        previous_orientations = {
            (a, b)
            for a in recipients
            for b in recipients
            if prev_recipient["signs"][a] == "positive"
            and prev_recipient["signs"][b] == "negative"
        }
        current_orientations = {
            (a, b)
            for a in recipients
            for b in recipients
            if curr_recipient["signs"][a] == "positive"
            and curr_recipient["signs"][b] == "negative"
        }
        reproducible_direction.append(
            float(bool(previous_orientations & current_orientations))
        )

    return {
        "shared_events": len(shared),
        "current_coverage": len(shared) / len(current_events),
        "previous_coverage": len(shared) / len(previous_events),
        "rq2": {
            "global_utility_sign_consistency": estimate(global_sign_agreement),
            "global_direct_sign_reversal_rate": estimate(global_direct_reversal),
            "oracle_decision_agreement": estimate(decision_agreement),
            "previous_decisions_on_current_gain": estimate(
                previous_to_current_gain
            ),
            "current_decisions_on_previous_gain": estimate(
                current_to_previous_gain
            ),
            "bidirectional_cross_run_gain": estimate(
                [
                    (left + right) / 2.0
                    for left, right in zip(
                        previous_to_current_gain, current_to_previous_gain
                    )
                ]
            ),
        },
        "rq3": {
            "exact_sign_vector_agreement": estimate(event_sign_agreement),
            "recipient_sign_consistency": estimate(recipient_sign_agreement),
            "stable_direct_sign_flip_rate": estimate(stable_flip),
            "reproducible_directional_flip_rate": estimate(
                reproducible_direction
            ),
        },
    }


def _run_label(manifest: Mapping[str, Any], fallback: str) -> str:
    seed = manifest.get("run", {}).get("sample_seed_base")
    return fallback if seed is None else f"Seed {seed}"


def _rq2_policy_table(
    current: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    *,
    metric: str,
    current_label: str,
    previous_label: str | None,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Build a budget-matched RQ2 table without additional rollouts.

    With an independent retest, every statistic is computed on shared events.
    Each event contributes the mean of its two independent-run outcomes, so the
    bootstrap unit remains the held-out task-memory event.
    """

    current_events = _event_map(current)
    if previous is None:
        event_ids = sorted(current_events)
        run_specs = [("current", current_label, current_events)]
    else:
        previous_events = _event_map(previous)
        event_ids = sorted(set(current_events) & set(previous_events))
        if not event_ids:
            raise AnalysisError("policy table has no shared retest events")
        run_specs = [
            ("previous", previous_label or "Previous run", previous_events),
            ("current", current_label, current_events),
        ]

    delta = float(current["delta"])
    values: dict[str, dict[str, list[float]]] = {}
    decisions: dict[str, list[bool]] = {}
    keep_rates: dict[str, float] = {}
    for key, _label_value, events in run_specs:
        use = [
            float(events[event_id]["outcomes"]["use_all"][metric])
            for event_id in event_ids
        ]
        drop = [
            float(events[event_id]["outcomes"]["global_drop"][metric])
            for event_id in event_ids
        ]
        keep = [left - right > delta for left, right in zip(use, drop)]
        keep_rate = sum(keep) / len(keep)
        decisions[key] = keep
        keep_rates[key] = keep_rate
        values[key] = {
            "always_use": use,
            "always_global_drop": drop,
            # Exact expectation of a uniformly random policy with the same
            # per-run keep budget as that run's same-sample oracle.
            "random_budget_matched": [
                keep_rate * left + (1.0 - keep_rate) * right
                for left, right in zip(use, drop)
            ],
            "oracle_selective_drop": [
                left if keep_value else right
                for left, right, keep_value in zip(use, drop, keep)
            ],
        }

    if previous is not None:
        values["previous"]["cross_run_selective_drop"] = [
            left if keep else right
            for left, right, keep in zip(
                values["previous"]["always_use"],
                values["previous"]["always_global_drop"],
                decisions["current"],
            )
        ]
        values["current"]["cross_run_selective_drop"] = [
            left if keep else right
            for left, right, keep in zip(
                values["current"]["always_use"],
                values["current"]["always_global_drop"],
                decisions["previous"],
            )
        ]

    policy_order = [
        ("always_use", "Always Use"),
        ("always_global_drop", "Always Global Drop"),
        ("random_budget_matched", "Random (budget matched)"),
    ]
    if previous is not None:
        policy_order.append(
            ("cross_run_selective_drop", "Cross-run Selective Drop")
        )
    policy_order.append(("oracle_selective_drop", "Oracle Selective Drop"))

    baseline_by_event = [
        _mean([values[key]["always_use"][index] for key, _, _ in run_specs])
        for index in range(len(event_ids))
    ]
    oracle_by_event = [
        _mean(
            [
                values[key]["oracle_selective_drop"][index]
                for key, _, _ in run_specs
            ]
        )
        for index in range(len(event_ids))
    ]
    oracle_gain = _mean(
        [right - left for left, right in zip(baseline_by_event, oracle_by_event)]
    )
    mean_budget = _mean(list(keep_rates.values()))

    rows = []
    for policy, label in policy_order:
        policy_by_event = [
            _mean([values[key][policy][index] for key, _, _ in run_specs])
            for index in range(len(event_ids))
        ]
        gains = [
            score - baseline
            for score, baseline in zip(policy_by_event, baseline_by_event)
        ]
        gain = _bootstrap(
            gains,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + len(rows) + 1,
        )
        if policy == "always_use":
            keep_rate = 1.0
            gap_recovery: float | None = 0.0
        elif policy == "always_global_drop":
            keep_rate = 0.0
            # Gap recovery conventionally compares selective policies, not
            # the no-memory endpoint.
            gap_recovery = None
        else:
            keep_rate = mean_budget
            gap_recovery = (
                float(gain["estimate"]) / oracle_gain
                if oracle_gain > 0.0
                else None
            )
        display_label = (
            f"{label} @ {keep_rate:.1%}"
            if policy
            in {
                "random_budget_matched",
                "cross_run_selective_drop",
                "oracle_selective_drop",
            }
            else label
        )
        rows.append(
            {
                "policy": policy,
                "label": display_label,
                "memory_keep_rate": keep_rate,
                "run_means": {
                    run_label: _mean(values[key][policy])
                    for key, run_label, _ in run_specs
                },
                "mean_score": _mean(policy_by_event),
                "delta_vs_always_use": gain,
                "oracle_gap_recovery": gap_recovery,
            }
        )

    return {
        "metric": metric,
        "event_count": len(event_ids),
        "run_labels": [run_label for _, run_label, _ in run_specs],
        "bootstrap_unit": "shared_held_out_task_candidate_event",
        "random_policy": "exact expectation under a uniform fixed-budget selector",
        "cross_run_policy": (
            "each run is selected using the other run's oracle decisions"
            if previous is not None
            else None
        ),
        "rows": rows,
    }


def _fmt(metric: Mapping[str, Any], percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return (
        f"{float(metric['estimate']) * scale:.2f}{suffix} "
        f"[{float(metric['ci_low']) * scale:.2f}, "
        f"{float(metric['ci_high']) * scale:.2f}]"
    )


def _fmt_table_score(value: float, percent: bool) -> str:
    return f"{value * 100.0:.2f}%" if percent else f"{value:.4f}"


def _fmt_table_delta(metric: Mapping[str, Any], percent: bool) -> str:
    if percent:
        return (
            f"{float(metric['estimate']) * 100.0:+.2f} pp "
            f"[{float(metric['ci_low']) * 100.0:+.2f}, "
            f"{float(metric['ci_high']) * 100.0:+.2f}]"
        )
    return (
        f"{float(metric['estimate']):+.4f} "
        f"[{float(metric['ci_low']):+.4f}, "
        f"{float(metric['ci_high']):+.4f}]"
    )


def _report_policy_table(
    table: Mapping[str, Any], *, score_name: str, percent_score: bool
) -> list[str]:
    confidence_level = float(
        table["rows"][0]["delta_vs_always_use"]["confidence_level"]
    )
    headers = [
        "Policy",
        "Memory Keep Rate ↓",
        *(f"{label} {score_name} ↑" for label in table["run_labels"]),
    ]
    if len(table["run_labels"]) > 1:
        headers.append(f"Mean {score_name} ↑")
    headers.extend(
        (
            f"Δ vs Always Use [{confidence_level:.0%} CI] ↑",
            "Oracle Gap Recovery ↑",
        )
    )
    alignments = ["---", "---:", *(["---:"] * len(table["run_labels"]))]
    if len(table["run_labels"]) > 1:
        alignments.append("---:")
    alignments.extend(("---:", "---:"))
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(alignments) + "|",
    ]
    for row in table["rows"]:
        cells = [row["label"], f"{float(row['memory_keep_rate']):.1%}"]
        cells.extend(
            _fmt_table_score(float(row["run_means"][label]), percent_score)
            for label in table["run_labels"]
        )
        if len(table["run_labels"]) > 1:
            cells.append(_fmt_table_score(float(row["mean_score"]), percent_score))
        cells.append(
            "—"
            if row["policy"] == "always_use"
            else _fmt_table_delta(row["delta_vs_always_use"], percent_score)
        )
        recovery = row["oracle_gap_recovery"]
        cells.append("—" if recovery is None else f"{float(recovery):.1%}")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _report(payload: Mapping[str, Any]) -> str:
    current = payload["current"]
    recipient = current["rq3"]["primary_team_score"]
    retest = payload.get("independent_retest")
    lines = [
        "# Native GMemory + MacNet causal audit",
        "",
        f"- Held-out task-memory events: {current['event_count']}",
        f"- Recipients: {', '.join(current['recipients'])}",
        "- Primary outcome: `reward - cost_weight * steps / max_trials`",
        "- Secondary outcome: raw PDDL success",
        "",
        "## RQ2: Global selective-memory value",
        "",
    ]
    lines.extend(
        _report_policy_table(
            payload["rq2_policy_tables"]["team_score"],
            score_name="Team Score",
            percent_score=False,
        )
    )
    lines.extend(
        [
            "",
            "Team Score = task reward minus the configured step-cost penalty. "
            "CIs are event-paired percentile-bootstrap intervals.",
            "Random is the exact expected score of a uniform selector at the "
            "same keep budget as Oracle; it requires no extra rollout.",
            "Cross-run Selective Drop applies each run's decisions to the other "
            "run. Oracle Selective Drop is a same-sample upper bound, not a "
            "deployable policy.",
            "",
            "### Success-rate sensitivity",
            "",
        ]
    )
    lines.extend(
        _report_policy_table(
            payload["rq2_policy_tables"]["success"],
            score_name="Success Rate",
            percent_score=True,
        )
    )
    lines.extend(
        [
            "",
            "## RQ3: Recipient heterogeneity",
            "",
            "- Direct positive/negative recipient sign-flip rate: "
            + _fmt(recipient["direct_sign_flip_rate"], percent=True),
            "- Any recipient sign heterogeneity rate: "
            + _fmt(recipient["any_sign_heterogeneity_rate"], percent=True),
            "- Mean within-event utility range: "
            + _fmt(recipient["mean_utility_range"]),
            "",
            "| Recipient | Mean utility [CI] | Positive | Neutral | Negative |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in current["recipients"]:
        row = recipient["per_recipient"][name]
        lines.append(
            f"| {name} | {_fmt(row['mean_utility'])} | {row['positive_events']} | "
            f"{row['neutral_events']} | {row['negative_events']} |"
        )
    lines.extend(["", "## Independent retest", ""])
    if retest is None:
        lines.append(
            "Not available. A disjoint inference-seed run is required before "
            "treating oracle decisions or recipient flips as replicated."
        )
    else:
        lines.extend(
            [
                f"- Shared events: {retest['shared_events']}",
                "- RQ2 global utility sign consistency: "
                + _fmt(
                    retest["rq2"]["global_utility_sign_consistency"],
                    percent=True,
                ),
                "- RQ2 bidirectional cross-run gain: "
                + _fmt(retest["rq2"]["bidirectional_cross_run_gain"]),
                "- RQ3 recipient sign consistency: "
                + _fmt(
                    retest["rq3"]["recipient_sign_consistency"], percent=True
                ),
                "- RQ3 stable direct sign-flip rate: "
                + _fmt(
                    retest["rq3"]["stable_direct_sign_flip_rate"], percent=True
                ),
                "- RQ3 reproducible directional flip rate: "
                + _fmt(
                    retest["rq3"]["reproducible_directional_flip_rate"],
                    percent=True,
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def _write_matrix(path: Path, analysis: Mapping[str, Any]) -> None:
    recipients = tuple(analysis["recipients"])
    rq3_events = {
        event["event_id"]: event
        for event in analysis["rq3"]["primary_team_score"]["events"]
    }
    fields = ["event_id", "task_id", "candidate_id"]
    for recipient in recipients:
        fields.extend((f"{recipient}_utility", f"{recipient}_sign"))
    fields.extend(("direct_sign_flip", "any_sign_heterogeneity", "utility_range"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in analysis["events"]:
            row = {
                "event_id": event["event_id"],
                "task_id": event["task_id"],
                "candidate_id": event["candidate_id"],
            }
            diagnostic = rq3_events[event["event_id"]]
            for recipient in recipients:
                row[f"{recipient}_utility"] = diagnostic["utilities"][recipient]
                row[f"{recipient}_sign"] = diagnostic["signs"][recipient]
            row.update(
                direct_sign_flip=diagnostic["direct_sign_flip"],
                any_sign_heterogeneity=diagnostic["any_sign_heterogeneity"],
                utility_range=diagnostic["utility_range"],
            )
            writer.writerow(row)


def _write_policy_table(path: Path, table: Mapping[str, Any]) -> None:
    run_fields = [f"{label} score" for label in table["run_labels"]]
    fields = [
        "policy",
        "label",
        "memory_keep_rate",
        *run_fields,
        "mean_score",
        "delta_vs_always_use",
        "delta_ci_low",
        "delta_ci_high",
        "confidence_level",
        "oracle_gap_recovery",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in table["rows"]:
            gain = row["delta_vs_always_use"]
            output = {
                "policy": row["policy"],
                "label": row["label"],
                "memory_keep_rate": row["memory_keep_rate"],
                "mean_score": row["mean_score"],
                "delta_vs_always_use": gain["estimate"],
                "delta_ci_low": gain["ci_low"],
                "delta_ci_high": gain["ci_high"],
                "confidence_level": gain["confidence_level"],
                "oracle_gap_recovery": row["oracle_gap_recovery"],
            }
            for label, field in zip(table["run_labels"], run_fields):
                output[field] = row["run_means"][label]
            writer.writerow(output)


def write_analysis(
    results_dir: Path,
    *,
    retest_results: Path | None = None,
    delta: float = 0.0,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Path:
    manifest, rows = _load_results(results_dir)
    recipients = tuple(manifest["design"]["recipients"])
    current = analyze_run(
        rows,
        recipients,
        delta=delta,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    payload: dict[str, Any] = {
        "analysis_schema": ANALYSIS_SCHEMA,
        "design_hash": manifest["design_hash"],
        "current_results": str(results_dir),
        "current": current,
        "previous_results": None,
        "previous": None,
        "independent_retest": None,
    }
    previous: dict[str, Any] | None = None
    previous_manifest: dict[str, Any] | None = None
    if retest_results is not None:
        previous_manifest, previous_rows = _load_results(retest_results)
        if previous_manifest["design_hash"] != manifest["design_hash"]:
            raise AnalysisError("retest design_hash differs from current run")
        previous = analyze_run(
            previous_rows,
            recipients,
            delta=delta,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + 10_000,
        )
        payload.update(
            previous_results=str(retest_results),
            previous=previous,
            independent_retest=compare_runs(
                current,
                previous,
                bootstrap_samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=seed + 20_000,
            ),
        )
    current_label = _run_label(manifest, "Current run")
    previous_label = (
        _run_label(previous_manifest, "Previous run")
        if previous_manifest is not None
        else None
    )
    if previous_label == current_label:
        previous_label = f"{previous_label} (previous)"
        current_label = f"{current_label} (current)"
    payload["rq2_policy_tables"] = {
        "team_score": _rq2_policy_table(
            current,
            previous,
            metric="team_score",
            current_label=current_label,
            previous_label=previous_label,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + 30_000,
        ),
        "success": _rq2_policy_table(
            current,
            previous,
            metric="success",
            current_label=current_label,
            previous_label=previous_label,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + 40_000,
        ),
    }
    output = results_dir / "oracle_recipient_analysis.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_matrix(results_dir / "recipient_matrix.csv", current)
    _write_policy_table(
        results_dir / "rq2_team_score_policy_table.csv",
        payload["rq2_policy_tables"]["team_score"],
    )
    _write_policy_table(
        results_dir / "rq2_success_policy_table.csv",
        payload["rq2_policy_tables"]["success"],
    )
    (results_dir / "run_report.md").write_text(_report(payload), encoding="utf-8")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--retest-results", type=Path, default=None)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    output = write_analysis(
        args.results,
        retest_results=args.retest_results,
        delta=args.delta,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    print(output)
    return output


if __name__ == "__main__":
    main()
