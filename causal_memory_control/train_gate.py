#!/usr/bin/env python3
"""Train a role-free team-exposure gate from matched USE/DROP outcomes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .estimator import AmortizedUtilityEstimator, PotentialOutcomeExample
from .features import HashEmbedder, TeamExposureFeatureBuilder, task_context_metadata
from .insight_metadata import enrich_insight_payload, load_insight_catalog
from .runtime_gate import save_gate_checkpoint
from .compare_gate_runs import file_sha256
from .types import MemoryCandidate, RetrievalMetadata, TeamMemoryExposureEvent


def parse_task_ids(value: str) -> set[int]:
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            if end < start:
                raise argparse.ArgumentTypeError("task-id range must be ascending")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    if not selected:
        raise argparse.ArgumentTypeError("at least one task id is required")
    return selected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more branch JSONL files/directories, or one direct JSON/JSONL file",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", default="success")
    parser.add_argument(
        "--task-ids",
        type=parse_task_ids,
        default=None,
        help="Optional training-only task subset, for example 0-49,61",
    )
    parser.add_argument("--ensemble-size", type=int, default=12)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--hash-dimensions", type=int, default=256)
    parser.add_argument("--feature-version", type=int, choices=(1, 2), default=2)
    parser.add_argument("--solver", choices=("sgd", "ridge"), default="ridge")
    parser.add_argument("--bootstrap-unit", choices=("event", "task"), default="task")
    parser.add_argument(
        "--memory-dir", type=Path, default=None,
        help="Restore missing insight metadata in older branches from their registered snapshot",
    )
    parser.add_argument(
        "--direct-utility-weight",
        type=float,
        default=0.5,
        help=(
            "Blend weight for a direct USE-minus-DROP utility head; 0 keeps "
            "the legacy T-learner and 0.5 enables the hybrid estimator"
        ),
    )
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=0.1,
        help="Within-task pairwise ranking loss weight for the direct utility head",
    )
    parser.add_argument("--contrastive-margin", type=float, default=0.05)
    parser.add_argument("--contrastive-min-gap", type=float, default=0.05)
    parser.add_argument("--contrastive-epochs", type=int, default=20)
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of diagnostic task groups held out from estimator fitting "
            "to choose kappa; 0 disables policy calibration"
        ),
    )
    parser.add_argument(
        "--kappa-grid",
        default="0,0.1,0.25,0.5,1.0,1.96",
        help="Comma-separated non-negative kappa values for calibration",
    )
    parser.add_argument("--calibration-seed", type=int, default=71)
    parser.add_argument("--calibration-max-drops", type=int, default=1)
    parser.add_argument("--calibration-delta", type=float, default=0.0)
    parser.add_argument(
        "--exclude-residual-noise",
        action="store_true",
        help=(
            "Deprecated alias for --residual-noise-scale 0; do not add the "
            "training-residual floor to deployment uncertainty"
        ),
    )
    parser.add_argument(
        "--residual-noise-scale",
        type=float,
        default=None,
        help=(
            "Scale applied to the residual uncertainty floor; 0 uses only "
            "bootstrap-ensemble uncertainty and 1 preserves the legacy floor"
        ),
    )
    return parser.parse_args(argv)


def load_examples(
    source: Path | Sequence[Path],
    *,
    metric: str,
    task_ids: set[int] | None,
    hash_dimensions: int,
    feature_version: int = 1,
    memory_dir: Path | None = None,
) -> list[PotentialOutcomeExample]:
    sources = [source] if isinstance(source, Path) else list(source)
    rows: list[dict[str, Any]] = []
    for item in sources:
        path = item / "branches.jsonl" if item.is_dir() else item
        if not path.is_file():
            raise FileNotFoundError(path)
        source_rows = _read_rows(path)
        if memory_dir is not None:
            source_manifest = json.loads((path.parent / "run_manifest.json").read_text(encoding="utf-8"))
            snapshot = json.loads((memory_dir / "causal_snapshot_manifest.json").read_text(encoding="utf-8"))
            if source_manifest["design"]["snapshot_design_hash"] != snapshot["design_hash"]:
                raise ValueError("--memory-dir is not the diagnostic's registered snapshot")
            catalog = load_insight_catalog(memory_dir)
            for row in source_rows:
                candidate = row.get("candidate", {})
                if candidate.get("kind") == "insight":
                    row["candidate"] = enrich_insight_payload(candidate, catalog)
                if isinstance(row.get("candidate_set"), list):
                    row["candidate_set"] = [
                        enrich_insight_payload(peer, catalog) if peer.get("kind") == "insight" else peer
                        for peer in row["candidate_set"]
                    ]
        rows.extend(source_rows)
    if rows and "condition" in rows[0]:
        return _examples_from_branches(
            rows,
            metric=metric,
            task_ids=task_ids,
            hash_dimensions=hash_dimensions,
            feature_version=feature_version,
        )
    if len(sources) > 1:
        raise ValueError("multiple --input paths are supported only for branch rows")
    return _examples_from_direct_rows(rows, task_ids=task_ids)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        rows = payload.get("examples", payload.get("events"))
        if isinstance(rows, list):
            return rows
    raise ValueError("JSON input must be a row list or contain examples/events")


def _examples_from_direct_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    task_ids: set[int] | None,
) -> list[PotentialOutcomeExample]:
    examples = []
    for index, row in enumerate(rows):
        if task_ids is not None and int(row.get("task_id", -1)) not in task_ids:
            continue
        q_use = _mean_value(row, "q_use", "use_rewards")
        q_drop = _mean_value(row, "q_drop", "drop_rewards")
        if "features" not in row:
            raise ValueError(
                "direct rows must contain features plus q_use/q_drop or reward arrays"
            )
        examples.append(
            PotentialOutcomeExample(
                features={key: float(value) for key, value in row["features"].items()},
                q_use=q_use,
                q_drop=q_drop,
                event_id=str(row.get("event_id", f"direct-{index}")),
                group_id=str(row.get("group_id", row.get("task_id", ""))),
            )
        )
    return examples


def _examples_from_branches(
    rows: Iterable[Mapping[str, Any]],
    *,
    metric: str,
    task_ids: set[int] | None,
    hash_dimensions: int,
    feature_version: int = 1,
) -> list[PotentialOutcomeExample]:
    groups: dict[tuple[Any, str], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    observed_candidates: dict[Any, dict[str, MemoryCandidate]] = defaultdict(dict)
    for row in rows:
        task_id = row.get("task_id")
        if task_ids is not None and int(task_id) not in task_ids:
            continue
        candidate = row.get("candidate", {})
        candidate_id = str(candidate.get("candidate_id", ""))
        if not candidate_id:
            raise ValueError("branch row is missing candidate.candidate_id")
        groups[(task_id, candidate_id)][str(row.get("condition"))].append(row)
        observed_candidates[task_id][candidate_id] = _candidate(candidate)

    feature_builder = TeamExposureFeatureBuilder(HashEmbedder(hash_dimensions), version=feature_version)
    examples = []
    for (task_id, candidate_id), conditions in sorted(
        groups.items(), key=lambda item: (str(item[0][0]), item[0][1])
    ):
        if not conditions.get("use_all") or not conditions.get("global_drop"):
            continue
        use_rows, drop_rows = _matched_conditions(
            conditions["use_all"], conditions["global_drop"],
            event_key=(task_id, candidate_id),
        )
        first = use_rows[0]
        for row in (*use_rows, *drop_rows):
            for key in ("candidate", "candidate_set", "task", "retrieval_sizes", "exposure_count"):
                if row.get(key) != first.get(key):
                    raise ValueError(f"matched branch context differs for task {task_id}, field {key}")
        q_use = _condition_mean(use_rows, metric)
        q_drop = _condition_mean(drop_rows, metric)
        candidate = _candidate(first["candidate"])
        query = _query(first.get("task", {}))
        exposure_count = _exposure_count(first)
        event = TeamMemoryExposureEvent(
            event_id=str(first.get("event_id", f"team-{task_id}-{candidate_id}")),
            query=query,
            task_state=str(first.get("task", {}).get("task_description", query)),
            memory=candidate,
            candidate_set=_candidate_set(
                first,
                candidate,
                observed=tuple(observed_candidates[task_id].values()),
            ),
            exposure_agent_ids=tuple(
                f"host-recipient-{index}" for index in range(exposure_count)
            ),
            retrieval=RetrievalMetadata(
                candidate_id=candidate.memory_id,
                source="g-memory",
                rank=int(first["candidate"].get("index", 0)) + 1,
            ),
            task_metadata=_task_metadata(first.get("task", {})),
        )
        examples.append(
            PotentialOutcomeExample(
                features=feature_builder.build(event),
                q_use=q_use,
                q_drop=q_drop,
                event_id=event.event_id,
                group_id=str(task_id),
            )
        )
    return examples


def _candidate(payload: Mapping[str, Any]) -> MemoryCandidate:
    kind = str(payload.get("kind", "trajectory"))
    if kind == "insight":
        content = str(payload.get("text", ""))
    else:
        description = str(
            payload.get("task_description") or payload.get("task_main") or ""
        )
        key_steps = str(payload.get("key_steps") or "")
        trajectory = str(payload.get("trajectory") or "")
        content = "\n".join(
            value
            for value in (
                f"Source task: {description}" if description else "",
                f"Key steps: {key_steps}" if key_steps else "",
                f"Trajectory: {trajectory}" if trajectory else "",
            )
            if value
        )
    return MemoryCandidate(
        memory_id=str(payload["candidate_id"]),
        memory_type=kind,
        content=content,
        metadata={
            "retrieval_index": int(payload.get("index", 0)),
            "source_label": payload.get("label"),
            "key_steps": payload.get("key_steps"),
            "legacy_score": payload.get("legacy_score"),
            "positive_task_count": payload.get("positive_task_count"),
            "negative_task_count": payload.get("negative_task_count"),
            "memory_schema_version": payload.get(
                "memory_schema_version", "gmemory-v1"
            ),
        },
    )


def _query(task: Mapping[str, Any]) -> str:
    if task.get("task_main") is not None:
        return str(task["task_main"])
    if task.get("claim") is not None:
        return f"FEVER claim: {str(task['claim']).strip()}"
    return str(task.get("task", ""))


def _candidate_set(
    row: Mapping[str, Any],
    target: MemoryCandidate,
    *,
    observed: Sequence[MemoryCandidate] = (),
) -> tuple[MemoryCandidate, ...]:
    """Use logged peers when available and preserve the full retrieved size."""
    if isinstance(row.get("candidate_set"), list):
        peers = tuple(_candidate(payload) for payload in row["candidate_set"])
        if target.memory_id not in {peer.memory_id for peer in peers}:
            raise ValueError("logged candidate_set does not contain target")
        return peers
    sizes = row.get("retrieval_sizes", {})
    total = int(sizes.get("successful_trajectories", 0)) + int(
        sizes.get("insights", 0)
    )
    total = max(total, 1)
    unique_observed = {
        candidate.memory_id: candidate
        for candidate in observed
        if candidate.memory_id != target.memory_id
    }
    peers_with_payload = tuple(unique_observed.values())[: max(0, total - 1)]
    peers = tuple(
        MemoryCandidate(
            memory_id=f"unobserved-peer-{index}",
            memory_type="unobserved",
            content="",
        )
        for index in range(max(0, total - 1 - len(peers_with_payload)))
    )
    return (target, *peers_with_payload, *peers)


def _task_metadata(task: Mapping[str, Any]) -> dict[str, Any]:
    return task_context_metadata(task)


def _exposure_count(row: Mapping[str, Any]) -> int:
    trace = row.get("outcome", {}).get("trace", [])
    if trace:
        first = trace[0]
        count = len(first.get("workers", {}))
        if "decision" in first:
            count += 1
        return max(count, 1)
    return max(int(row.get("exposure_count", 1)), 1)


def _condition_mean(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    values = [float(row["outcome"][metric]) for row in rows]
    return sum(values) / len(values)


def _matched_conditions(
    use_rows: Sequence[Mapping[str, Any]],
    drop_rows: Sequence[Mapping[str, Any]],
    *,
    event_key: tuple[Any, str],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    def index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, Any, Any], Mapping[str, Any]]:
        result = {}
        for row in rows:
            key = (
                row.get("run_hash"),
                row.get("sample_seed"),
                row.get("repeat_index"),
            )
            if key in result:
                raise ValueError(f"duplicate counterfactual branch for {event_key}: {key}")
            result[key] = row
        return result

    use = index(use_rows)
    drop = index(drop_rows)
    if set(use) != set(drop):
        raise ValueError(
            f"USE/DROP branches are not matched for {event_key}: "
            f"use_only={sorted(map(str, set(use) - set(drop)))}, "
            f"drop_only={sorted(map(str, set(drop) - set(use)))}"
        )
    keys = sorted(use, key=str)
    return [use[key] for key in keys], [drop[key] for key in keys]


def _mean_value(row: Mapping[str, Any], scalar: str, samples: str) -> float:
    if scalar in row:
        return float(row[scalar])
    values = [float(value) for value in row[samples]]
    if not values:
        raise ValueError(f"{samples} must not be empty")
    return sum(values) / len(values)


def _split_calibration_examples(
    examples: Sequence[PotentialOutcomeExample],
    *,
    fraction: float,
    seed: int,
) -> tuple[list[PotentialOutcomeExample], list[PotentialOutcomeExample]]:
    """Split by task group so candidates from one task never cross the split."""

    if fraction <= 0.0:
        return list(examples), []
    groups = sorted({example.group_id or example.event_id for example in examples})
    if len(groups) < 5:
        raise SystemExit(
            "policy calibration requires at least five task groups; "
            "use --calibration-fraction 0 for a smoke test"
        )
    random.Random(seed).shuffle(groups)
    calibration_count = max(1, round(len(groups) * fraction))
    calibration_count = min(calibration_count, len(groups) - 1)
    calibration_groups = set(groups[:calibration_count])
    fit = [
        example
        for example in examples
        if (example.group_id or example.event_id) not in calibration_groups
    ]
    calibration = [
        example
        for example in examples
        if (example.group_id or example.event_id) in calibration_groups
    ]
    return fit, calibration


def _calibrate_policy(
    estimator: AmortizedUtilityEstimator,
    examples: Sequence[PotentialOutcomeExample],
    *,
    kappas: Sequence[float],
    max_drops: int,
    delta: float = 0.0,
) -> dict[str, Any]:
    """Choose a DROP confidence multiplier on held-out diagnostic tasks.

    The observed effect of dropping is ``q_drop - q_use``.  Candidate choices
    are made once per task, using the same lowest-upper-bound rule as runtime.
    Final benchmark tasks are never involved in this selection.
    """

    if not examples:
        return {
            "enabled": False,
            "reason": "no_calibration_split",
            "recommended_kappa": None,
            "candidates": [],
        }
    if max_drops != 1:
        raise ValueError("single-candidate counterfactuals can calibrate only max_drops=1")
    grouped: dict[str, list[PotentialOutcomeExample]] = defaultdict(list)
    for example in examples:
        grouped[example.group_id or example.event_id].append(example)
    predictions_by_group = {
        group_id: [(example, estimator.predict(example.features)) for example in group_examples]
        for group_id, group_examples in grouped.items()
    }
    candidates = [{
        "policy": "always_keep", "kappa": None,
        "calibration_tasks": len(grouped), "selected_drops": 0,
        "drop_task_rate": 0.0, "mean_observed_reward_delta": 0.0,
        "improved_drops": 0, "harmed_drops": 0,
        "harm_rate_among_changed": 0.0,
    }]
    for kappa in kappas:
        effects: list[float] = []
        selected_count = 0
        selected_tasks = 0
        harmed = improved = 0
        for scored_examples in predictions_by_group.values():
            scored = []
            for example, prediction in scored_examples:
                upper = prediction.utility + float(kappa) * prediction.uncertainty
                if prediction.calibrated and upper < -delta:
                    scored.append((upper, example))
            selected = [row[1] for row in sorted(scored, key=lambda row: row[0])[:max_drops]]
            group_effect = sum(example.q_drop - example.q_use for example in selected)
            effects.append(group_effect)
            selected_count += len(selected)
            selected_tasks += bool(selected)
            improved += sum(example.q_drop > example.q_use for example in selected)
            harmed += sum(example.q_drop < example.q_use for example in selected)
        mean_effect = sum(effects) / len(effects) if effects else 0.0
        candidates.append(
            {
                "kappa": float(kappa),
                "policy": "learned",
                "calibration_tasks": len(grouped),
                "selected_drops": selected_count,
                "drop_task_rate": selected_tasks / len(effects),
                "mean_observed_reward_delta": mean_effect,
                "improved_drops": improved,
                "harmed_drops": harmed,
                "harm_rate_among_changed": (
                    harmed / (improved + harmed) if improved + harmed else 0.0
                ),
            }
        )
    # Maximize held-out reward, then prefer the more conservative multiplier
    # on ties. This permits bolder dropping only when held-out interventions
    # support it.
    beneficial = [row for row in candidates if row["mean_observed_reward_delta"] > 0.0]
    selected = (max(beneficial, key=lambda row: (row["mean_observed_reward_delta"], row["kappa"]))
                if beneficial else candidates[0])
    return {
        "enabled": True,
        "selection_scope": "held_out_diagnostic_task_groups",
        "objective": "maximize_mean_observed_reward_delta_then_larger_kappa",
        "max_drops": max_drops,
        "delta": delta,
        "recommended_kappa": selected["kappa"] if selected["kappa"] is not None else max(kappas),
        "recommended_max_drops": 0 if selected["policy"] == "always_keep" else 1,
        "calibration_task_ids": sorted(grouped),
        "selected": selected,
        "candidates": candidates,
    }


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.residual_noise_scale is not None and args.residual_noise_scale < 0:
        raise SystemExit("--residual-noise-scale must be non-negative")
    if args.exclude_residual_noise and args.residual_noise_scale not in (None, 0.0):
        raise SystemExit(
            "--exclude-residual-noise conflicts with a nonzero "
            "--residual-noise-scale"
        )
    if not 0.0 <= args.direct_utility_weight <= 1.0:
        raise SystemExit("--direct-utility-weight must be in [0,1]")
    if min(
        args.contrastive_weight,
        args.contrastive_margin,
        args.contrastive_min_gap,
        args.contrastive_epochs,
    ) < 0:
        raise SystemExit("contrastive settings must be non-negative")
    if not 0.0 <= args.calibration_fraction < 1.0:
        raise SystemExit("--calibration-fraction must be in [0,1)")
    if args.calibration_max_drops != 1:
        raise SystemExit("single-candidate calibration requires --calibration-max-drops 1")
    if not math.isfinite(args.calibration_delta) or args.calibration_delta < 0:
        raise SystemExit("--calibration-delta must be finite and non-negative")
    try:
        kappa_grid = sorted(
            {float(value.strip()) for value in args.kappa_grid.split(",") if value.strip()}
        )
    except ValueError as exc:
        raise SystemExit("--kappa-grid must contain only numbers") from exc
    if not kappa_grid or not all(math.isfinite(k) and k >= 0 for k in kappa_grid):
        raise SystemExit("--kappa-grid must contain non-negative values")
    residual_noise_scale = (
        0.0
        if args.exclude_residual_noise
        else 1.0
        if args.residual_noise_scale is None
        else float(args.residual_noise_scale)
    )
    examples = load_examples(
        args.input,
        metric=args.metric,
        task_ids=args.task_ids,
        hash_dimensions=args.hash_dimensions,
        feature_version=args.feature_version,
        memory_dir=args.memory_dir,
    )
    fit_examples, calibration_examples = _split_calibration_examples(
        examples,
        fraction=args.calibration_fraction,
        seed=args.calibration_seed,
    )
    fit_scope = _candidate_kind_ranks(fit_examples)
    missing_scope = [f"{kind}:rank{rank}"
                     for kind, ranks in _candidate_kind_ranks(calibration_examples).items()
                     for rank in ranks if rank not in fit_scope.get(kind, [])]
    if missing_scope:
        raise SystemExit("calibration contains candidates absent from fitting tasks: "
                         + ", ".join(missing_scope))
    estimator = AmortizedUtilityEstimator(
        ensemble_size=args.ensemble_size,
        min_samples=args.min_samples,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2=args.l2,
        seed=args.seed,
        residual_noise_scale=residual_noise_scale,
        direct_utility_weight=args.direct_utility_weight,
        contrastive_weight=args.contrastive_weight,
        contrastive_margin=args.contrastive_margin,
        contrastive_min_gap=args.contrastive_min_gap,
        contrastive_epochs=args.contrastive_epochs,
        solver=args.solver,
        bootstrap_unit=args.bootstrap_unit,
    )
    if not estimator.fit(fit_examples, progress=True):
        raise SystemExit(
            f"only {len(fit_examples)} fitting events after calibration split; "
            f"min_samples={args.min_samples}"
        )
    policy_calibration = _calibrate_policy(
        estimator,
        calibration_examples,
        kappas=kappa_grid,
        max_drops=args.calibration_max_drops,
        delta=args.calibration_delta,
    )
    recorded_task_ids = (
        sorted(args.task_ids)
        if args.task_ids is not None
        else _training_task_ids(args.input)
    )
    save_gate_checkpoint(
        args.output,
        estimator,
        hash_dimensions=args.hash_dimensions,
        feature_version=args.feature_version,
        training_metadata={
            "metric": args.metric,
            "training_events": len(examples),
            "estimator_fit_events": len(fit_examples),
            "calibration_events": len(calibration_examples),
            "calibration_fraction": args.calibration_fraction,
            "calibration_seed": args.calibration_seed,
            "kappa_grid": kappa_grid,
            "training_sources": [str(path) for path in args.input],
            "training_source_sha256": {
                str(path): file_sha256(path / "branches.jsonl" if path.is_dir() else path)
                for path in args.input
            },
            "training_task_ids": recorded_task_ids,
            "candidate_kinds": _candidate_kinds(fit_examples),
            "candidate_ranks": _candidate_ranks(fit_examples),
            "candidate_kind_ranks": _candidate_kind_ranks(fit_examples),
            "fit_task_ids": sorted({e.group_id for e in fit_examples if e.group_id}),
            "residual_noise_scale": residual_noise_scale,
            "estimator": {
                "type": "hybrid_t_direct_utility",
                "direct_utility_weight": args.direct_utility_weight,
                "contrastive_weight": args.contrastive_weight,
                "contrastive_margin": args.contrastive_margin,
                "contrastive_min_gap": args.contrastive_min_gap,
                "contrastive_epochs": args.contrastive_epochs,
                "contrastive_pairs": estimator.contrastive_pairs,
            },
            "policy_calibration": policy_calibration,
            "training_contexts": _training_contexts(args.input),
        },
    )
    predictions = [estimator.predict(example.features) for example in fit_examples]
    utility_rmse = (
        sum(
            (
                prediction.utility - (example.q_use - example.q_drop)
            ) ** 2
            for prediction, example in zip(predictions, fit_examples)
        )
        / len(fit_examples)
    ) ** 0.5
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "training_events": len(examples),
                "estimator_fit_events": len(fit_examples),
                "calibration_events": len(calibration_examples),
                "metric": args.metric,
                "residual_noise_scale": residual_noise_scale,
                "direct_utility_weight": args.direct_utility_weight,
                "contrastive_weight": args.contrastive_weight,
                "contrastive_pairs": estimator.contrastive_pairs,
                "recommended_kappa": policy_calibration.get("recommended_kappa"),
                "held_in_utility_rmse": utility_rmse,
                "warning": "held-in fit is diagnostic only; evaluate on disjoint tasks",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return args.output


def _candidate_kinds(examples: Sequence[PotentialOutcomeExample]) -> list[str]:
    prefix = "memory_type::"
    return sorted(
        {
            feature_name.removeprefix(prefix)
            for example in examples
            for feature_name, value in example.features.items()
            if feature_name.startswith(prefix) and float(value) > 0
        }
    )


def _candidate_ranks(examples: Sequence[PotentialOutcomeExample]) -> list[int]:
    return sorted(
        {
            int(example.features["retrieval_rank"])
            for example in examples
            if float(example.features.get("retrieval_rank_missing", 0.0)) == 0.0
            and float(example.features.get("retrieval_rank", 0.0)) >= 1.0
        }
    )


def _candidate_kind_ranks(
    examples: Sequence[PotentialOutcomeExample],
) -> dict[str, list[int]]:
    prefix = "memory_type::"
    pairs: dict[str, set[int]] = defaultdict(set)
    for example in examples:
        if float(example.features.get("retrieval_rank_missing", 0.0)) != 0.0:
            continue
        rank = int(example.features.get("retrieval_rank", 0.0))
        if rank < 1:
            continue
        for feature_name, value in example.features.items():
            if feature_name.startswith(prefix) and float(value) > 0:
                pairs[feature_name.removeprefix(prefix)].add(rank)
    return {kind: sorted(ranks) for kind, ranks in sorted(pairs.items())}


def _training_task_ids(sources: Sequence[Path]) -> list[int]:
    task_ids: set[int] = set()
    for source in sources:
        path = source / "branches.jsonl" if source.is_dir() else source
        for row in _read_rows(path):
            if row.get("task_id") is not None:
                task_ids.add(int(row["task_id"]))
    return sorted(task_ids)


def _training_contexts(sources: Sequence[Path]) -> list[dict[str, Any]]:
    keys = (
        "benchmark",
        "source_md5",
        "snapshot_design_hash",
        "memory_sha256",
        "model",
        "graph_type",
        "node_num",
        "embedding_model",
        "successful_topk",
        "failed_topk",
        "insights_topk",
        "threshold",
    )
    contexts: dict[str, dict[str, Any]] = {}
    for source in sources:
        path = source / "branches.jsonl" if source.is_dir() else source
        manifest_path = path.parent / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        design = manifest.get("design", {})
        if not isinstance(design, Mapping):
            continue
        context = {key: design[key] for key in keys if key in design}
        canonical = json.dumps(context, ensure_ascii=False, sort_keys=True)
        contexts[canonical] = context
    return [contexts[key] for key in sorted(contexts)]


if __name__ == "__main__":
    main()
