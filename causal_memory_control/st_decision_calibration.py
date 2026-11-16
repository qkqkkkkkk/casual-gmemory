"""Offline single-deletion decisions from existing paired development labels.

U = Q_use - Q_drop; a deletion's measured gain is -U. Never add the
single-deletion gains to estimate an unmeasured multiple-deletion action.
"""
from __future__ import annotations

from collections import defaultdict
import math
import statistics


KAPPAS = (0., .5, 1., 1.96, 3.)
DELTAS = (0., .01, .025, .05, .1)
KEEP_ALL = {"rule": "single_drop_calibrated", "kappa": 0., "delta": 0., "max_drops": 0}
PROTOCOL = {
    "scope": "original_dev_only",
    "objective": "mean_measured_single_drop_reward_gain_including_keep_tasks",
    "kappas": list(KAPPAS), "deltas": list(DELTAS),
    "drop_condition": "calibrated and in_scope and predicted_U + kappa * uncertainty < -delta",
    "budget": "at_most_one; keep_all_is_an_explicit_candidate",
    "tie_break": "fewer_drops_then_larger_delta_then_larger_kappa",
    "selection_per": "dataset_and_existing_checkpoint; same_grid_and_objective_for_all",
    "test_outcomes_used": False, "retraining": False,
    "caveat": "dev gains are selection estimates, not held-out test evidence; oracle is label-noisy and optimistic",
}


def groups(records, task_ids):
    result = {str(t): [] for t in task_ids}
    seen = set()
    for row in records:
        task = str(row["task_id"])
        key = task, row["memory_id"]
        if task not in result or key in seen:
            raise ValueError("unexpected dev task or duplicate candidate")
        seen.add(key)
        for field in ("predicted_u", "uncertainty", "q_use", "q_drop"):
            if not math.isfinite(row[field]):
                raise ValueError("non-finite dev prediction/label")
        if row["uncertainty"] < 0 or not all(0 <= row[k] <= 1 for k in ("q_use", "q_drop")):
            raise ValueError("invalid dev prediction/label range")
        result[task].append(row)
    if not result:
        raise ValueError("empty development task partition")
    return result


def choose(candidates, policy, *, force_one=False):
    if policy["max_drops"] == 0:
        return []
    scored = []
    for row in candidates:
        if not row["in_scope"] or not row["calibrated"]:
            continue
        upper = row["predicted_u"] + policy["kappa"] * row["uncertainty"]
        eligible = upper <= 0 if policy.get("rule") == "positive_unlimited" else upper < -policy["delta"]
        if force_one or eligible:
            scored.append((upper, row["order"], row))
    ordered = [r for _, _, r in sorted(scored, key=lambda r: r[:2])]
    return ordered if policy["max_drops"] is None else ordered[:policy["max_drops"]]


def assess(records, task_ids, policy, *, force_one=False):
    grouped = groups(records, task_ids)
    effects, details = [], []
    improved = harmed = neutral = drops = changed = 0
    oracle, regrets = [], []
    for task, candidates in sorted(grouped.items()):
        selected = choose(candidates, policy, force_one=force_one)
        best = max([0., *(r["q_drop"] - r["q_use"] for r in candidates if r["in_scope"])])
        oracle.append(best)
        drops += len(selected)
        changed += bool(selected)
        # Existing labels identify only KEEP ALL and individual deletions.
        effect = None if len(selected) > 1 else (selected[0]["q_drop"] - selected[0]["q_use"] if selected else 0.)
        if effect is not None:
            effects.append(effect)
            regrets.append(best - effect)
            improved += effect > 0
            harmed += effect < 0
            neutral += bool(selected) and effect == 0
        details.append({"task_id": task, "drop_ids": [r["memory_id"] for r in selected],
            "observed_gain": effect, "best_observed_single_drop_gain": best,
            "selected": [{k: r[k] for k in ("memory_id", "memory_type", "rank", "predicted_u", "uncertainty", "q_use", "q_drop")} for r in selected]})
    measurable = len(effects) == len(grouped)
    return {"policy": dict(policy), "tasks": len(grouped), "drop_tasks": changed,
        "drop_task_rate": changed / len(grouped), "drop_candidates": drops,
        "joint_effect_identified": measurable, "unmeasured_multi_drop_tasks": len(grouped) - len(effects),
        "mean_observed_gain": statistics.mean(effects) if measurable else None,
        "improved_tasks": improved if measurable else None, "harmed_tasks": harmed if measurable else None,
        "neutral_drop_tasks": neutral if measurable else None,
        "beneficial_fraction_of_drops": improved / changed if measurable and changed else None,
        "harm_rate_among_changed": harmed / (improved + harmed) if measurable and improved + harmed else None,
        "best_observed_single_drop_mean_gain": statistics.mean(oracle),
        "mean_regret_to_observed_single_drop_oracle": statistics.mean(regrets) if measurable else None,
        "task_decisions": details}


def calibrate(records, task_ids):
    policies = [KEEP_ALL, *({"rule": "single_drop_calibrated", "kappa": k, "delta": d, "max_drops": 1}
                            for k in KAPPAS for d in DELTAS)]
    grid = [{k: v for k, v in assess(records, task_ids, policy).items() if k != "task_decisions"}
            for policy in policies]
    # A zero/negative gain must not displace KEEP ALL, including numerical ties.
    beneficial = [r for r in grid if r["mean_observed_gain"] > 1e-12]
    best = max(beneficial, key=lambda r: (round(r["mean_observed_gain"], 12), -r["drop_tasks"],
               r["policy"]["delta"], r["policy"]["kappa"])) if beneficial else grid[0]
    return {"protocol": PROTOCOL, "selected": best, "grid": grid}


def bucket_summary(records, field, boundaries):
    buckets = defaultdict(list)
    for row in records:
        if row["in_scope"] and row["calibrated"]:
            value = row[field]
            index = sum(value >= bound for bound in boundaries)
            buckets[index].append(row)
    result = []
    for index, batch in sorted(buckets.items()):
        observed = [r["q_use"] - r["q_drop"] for r in batch]
        result.append({"lower_inclusive": boundaries[index - 1] if index else None,
            "upper_exclusive": boundaries[index] if index < len(boundaries) else None,
            "candidates": len(batch), "tasks": len({r["task_id"] for r in batch}),
            "predicted_u_mean": statistics.mean(r["predicted_u"] for r in batch),
            "uncertainty_mean": statistics.mean(r["uncertainty"] for r in batch),
            "observed_u_mean": statistics.mean(observed),
            "observed_harmful_memory": sum(u < 0 for u in observed),
            "observed_neutral_memory": sum(u == 0 for u in observed),
            "observed_helpful_memory": sum(u > 0 for u in observed)})
    return result


def diagnose(records, task_ids, original_policy, selected):
    grouped = groups(records, task_ids)
    selected_records = [r for candidates in grouped.values() for r in choose(candidates, selected)]
    gain_records = [{**r, "conservative_predicted_drop_gain": -r["predicted_u"] - selected["kappa"] * r["uncertainty"]}
                    for r in selected_records]
    return {"label_definition": "U=Q_use-Q_drop; harmful memory U<0; deletion gain=-U",
        "dev_tasks": len(grouped), "dev_candidates": len(records),
        "label_counts": {name: sum(test(r["q_use"] - r["q_drop"]) for r in records)
                         for name, test in (("negative", lambda u: u < 0), ("zero", lambda u: u == 0), ("positive", lambda u: u > 0))},
        "predicted_exact_zero": sum(r["predicted_u"] == 0 for r in records),
        "predicted_near_zero_abs_le_001": sum(abs(r["predicted_u"]) <= .01 for r in records),
        "utility_buckets": bucket_summary(records, "predicted_u", (-.1, -.025, -.01, 0., .01, .025, .1)),
        "uncertainty_buckets": bucket_summary(records, "uncertainty", (.01, .025, .05, .1, .2)),
        "selected_drop_gain_buckets": bucket_summary(gain_records, "conservative_predicted_drop_gain", (0., .01, .025, .05, .1, .2)),
        "policies": {
            "original_rule": assess(records, task_ids, original_policy),
            "forced_lowest_predicted_u_one": assess(records, task_ids,
                {"rule": "single_drop_calibrated", "kappa": 0., "delta": 0., "max_drops": 1}, force_one=True),
            "negative_u_one": assess(records, task_ids,
                {"rule": "single_drop_calibrated", "kappa": 0., "delta": 0., "max_drops": 1}),
            "dev_calibrated_single": assess(records, task_ids, selected),
        },
        "warning": "Existing dev labels were also used for prior model/calibration selection. These are exploratory diagnostics; a measured oracle can overfit rollout noise. Multi-drop effects are not inferred."}
