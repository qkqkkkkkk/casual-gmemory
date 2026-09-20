#!/usr/bin/env python3
"""Train a role-free team-exposure gate from matched USE/DROP outcomes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .estimator import AmortizedUtilityEstimator, PotentialOutcomeExample
from .features import HashEmbedder, TeamExposureFeatureBuilder
from .runtime_gate import save_gate_checkpoint
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
    parser.add_argument(
        "--exclude-residual-noise",
        action="store_true",
        help="Ablation only: do not add training residuals to deployment uncertainty",
    )
    return parser.parse_args(argv)


def load_examples(
    source: Path | Sequence[Path],
    *,
    metric: str,
    task_ids: set[int] | None,
    hash_dimensions: int,
) -> list[PotentialOutcomeExample]:
    sources = [source] if isinstance(source, Path) else list(source)
    rows: list[dict[str, Any]] = []
    for item in sources:
        path = item / "branches.jsonl" if item.is_dir() else item
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(_read_rows(path))
    if rows and "condition" in rows[0]:
        return _examples_from_branches(
            rows,
            metric=metric,
            task_ids=task_ids,
            hash_dimensions=hash_dimensions,
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
            )
        )
    return examples


def _examples_from_branches(
    rows: Iterable[Mapping[str, Any]],
    *,
    metric: str,
    task_ids: set[int] | None,
    hash_dimensions: int,
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

    feature_builder = TeamExposureFeatureBuilder(HashEmbedder(hash_dimensions))
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
    return {
        key: value
        for key, value in task.items()
        if isinstance(value, (str, int, float, bool))
        and key
        not in {
            "task_main",
            "task_description",
            "claim",
            "claim_id",
            "id",
            "task_id",
            "task",
            "answer",
            "label",
            "gold_label",
        }
    }


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


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    examples = load_examples(
        args.input,
        metric=args.metric,
        task_ids=args.task_ids,
        hash_dimensions=args.hash_dimensions,
    )
    estimator = AmortizedUtilityEstimator(
        ensemble_size=args.ensemble_size,
        min_samples=args.min_samples,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2=args.l2,
        seed=args.seed,
        include_residual_noise=not args.exclude_residual_noise,
    )
    if not estimator.fit(examples):
        raise SystemExit(
            f"only {len(examples)} complete events; min_samples={args.min_samples}"
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
        training_metadata={
            "metric": args.metric,
            "training_events": len(examples),
            "training_sources": [str(path) for path in args.input],
            "training_task_ids": recorded_task_ids,
            "candidate_kinds": _candidate_kinds(examples),
            "candidate_ranks": _candidate_ranks(examples),
            "candidate_kind_ranks": _candidate_kind_ranks(examples),
            "training_contexts": _training_contexts(args.input),
        },
    )
    predictions = [estimator.predict(example.features) for example in examples]
    utility_rmse = (
        sum(
            (
                prediction.utility - (example.q_use - example.q_drop)
            ) ** 2
            for prediction, example in zip(predictions, examples)
        )
        / len(examples)
    ) ** 0.5
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "training_events": len(examples),
                "metric": args.metric,
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
