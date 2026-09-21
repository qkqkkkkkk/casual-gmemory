#!/usr/bin/env python3
"""Run and resume the complete native-GMemory versus learned-gate FEVER study."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

from tqdm import tqdm

from causal_diagnostic.fever_oracle_recipient.data import (
    BINARY_LABELS,
    load_binary_fever,
)


PIPELINE_SCHEMA = "gmemory-fever-gate-comparison-pipeline-v3"
STAGES = ("diagnostic", "train_gate", "native_gmemory", "learned_gate", "compare")


class StageFailure(RuntimeError):
    def __init__(
        self,
        stage: str,
        returncode: int,
        command: Sequence[str],
        log_path: Path,
        tail: str,
    ) -> None:
        self.stage = stage
        self.returncode = int(returncode)
        self.command = tuple(command)
        self.log_path = log_path
        self.tail = tail
        super().__init__(f"stage {stage} failed with exit code {returncode}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/fever/fever_dev.jsonl"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:11436/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=Path(
            "causal_diagnostic/memory_snapshots/"
            "fever_evidence_support50_all_binary_7b_v5/g-memory"
        ),
    )
    parser.add_argument(
        "--diagnostic-results",
        type=Path,
        default=Path(
            "causal_diagnostic/results/native_fever_all_binary_gate_7b_v5"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "causal_memory_control/checkpoints/fever_gate_all_binary_v5.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "causal_memory_control/results/fever_gate_all_binary_v5"
        ),
    )
    parser.add_argument("--support-per-label", type=int, default=25)
    parser.add_argument("--evaluation-per-label", type=int, default=50)
    parser.add_argument("--training-claims", type=int, default=40)
    parser.add_argument(
        "--use-all-binary-data",
        action="store_true",
        help=(
            "Use every SUPPORTS/REFUTES example exactly once across support, "
            "gate training, and final evaluation"
        ),
    )
    parser.add_argument(
        "--training-fraction",
        type=float,
        default=0.8,
        help=(
            "In all-data mode, fraction of post-support examples per label "
            "assigned to gate training; the remainder is final evaluation"
        ),
    )
    parser.add_argument("--smoke-claims", type=int, default=4)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--build-seed", type=int, default=42)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--sample-seed-base", type=int, default=0)
    parser.add_argument("--retest-seed-base", type=int, default=1000)
    parser.add_argument("--gate-evaluation-seed", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--node-num", type=int, default=3)
    parser.add_argument(
        "--graph-type",
        choices=("Chain", "FullConnected", "Debate", "Star"),
        default="Chain",
    )
    parser.add_argument("--isolated-recipients", default="solver_0,solver_2")
    parser.add_argument("--successful-topk", type=int, default=3)
    parser.add_argument("--failed-topk", type=int, default=0)
    parser.add_argument("--insights-topk", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--candidate-kind",
        choices=("trajectory", "insight"),
        default="trajectory",
    )
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument(
        "--candidate-scope",
        choices=("all_retrieved", "single"),
        default="all_retrieved",
        help=(
            "Train and evaluate every exposed retrieval by default; use single "
            "only for the legacy one-kind/one-rank design"
        ),
    )
    parser.add_argument(
        "--diagnostic-scope",
        choices=("gate_training", "full_causal"),
        default="gate_training",
        help=(
            "gate_training collects only use_all/global_drop; full_causal also "
            "runs recipient branches for RQ3/RQ4"
        ),
    )
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--max-drops", type=int, default=1)
    parser.add_argument(
        "--residual-noise-scale",
        type=float,
        default=0.0,
        help=(
            "Scale the held-in residual uncertainty floor; 0 uses ensemble-only "
            "uncertainty and 1 reproduces the conservative legacy behavior"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the resolved split and branch counts without writing or running",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.data.is_file():
        raise SystemExit(f"FEVER data does not exist: {args.data}")
    if args.support_per_label < 1 or args.evaluation_per_label < 1:
        raise SystemExit("support/evaluation counts must be positive")
    evaluation_total = 2 * args.evaluation_per_label
    if not 1 <= args.training_claims < evaluation_total:
        raise SystemExit(
            f"--training-claims must be in [1,{evaluation_total - 1}]"
        )
    if not 1 <= args.smoke_claims <= args.training_claims:
        raise SystemExit("--smoke-claims must be in [1, training-claims]")
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.diagnostic_scope == "full_causal" and (
        args.repeats < 4 or args.repeats % 2
    ):
        raise SystemExit(
            "full_causal diagnostics require an even --repeats >= 4"
        )
    if args.node_num < 2:
        raise SystemExit("--node-num must be at least 2")
    if min(args.successful_topk, args.failed_topk, args.insights_topk) < 0:
        raise SystemExit("retrieval top-k values must be non-negative")
    if args.candidate_index < 0:
        raise SystemExit("--candidate-index must be non-negative")
    if (
        args.candidate_scope == "all_retrieved"
        and args.successful_topk + args.insights_topk < 1
    ):
        raise SystemExit(
            "all_retrieved scope requires --successful-topk or --insights-topk > 0"
        )
    if not 0.0 < args.training_fraction < 1.0:
        raise SystemExit("--training-fraction must be in (0,1)")
    if (
        args.temperature < 0
        or args.kappa < 0
        or args.delta < 0
        or args.residual_noise_scale < 0
    ):
        raise SystemExit(
            "temperature, kappa, delta, and residual-noise-scale must be "
            "non-negative"
        )
    if args.max_drops < -1:
        raise SystemExit("--max-drops must be non-negative or -1")
    if args.bootstrap_samples < 100:
        raise SystemExit("--bootstrap-samples must be at least 100")


def _resolve_data_plan(args: argparse.Namespace) -> None:
    """Expand all-data mode into the existing balanced registered split."""
    args.binary_label_counts = None
    args.binary_examples_used = None
    args.binary_examples_unused = None
    if not args.use_all_binary_data:
        return

    examples = load_binary_fever(args.data)
    counts = Counter(str(row["label"]) for row in examples)
    missing = [label for label in BINARY_LABELS if counts[label] < 2]
    if missing:
        raise SystemExit(f"all-data mode has insufficient labels: {missing}")
    if len({counts[label] for label in BINARY_LABELS}) != 1:
        raise SystemExit(
            "--use-all-binary-data requires equal SUPPORTS/REFUTES counts so "
            "the registered split can remain balanced without discarding rows"
        )
    balanced_per_label = min(counts[label] for label in BINARY_LABELS)
    evaluation_per_label = balanced_per_label - int(args.support_per_label)
    if evaluation_per_label < 2:
        raise SystemExit(
            "all-data mode leaves fewer than two post-support examples per label"
        )
    training_per_label = math.floor(
        evaluation_per_label * float(args.training_fraction)
    )
    if not 1 <= training_per_label < evaluation_per_label:
        raise SystemExit(
            "--training-fraction leaves no training or final-evaluation examples"
        )

    args.evaluation_per_label = evaluation_per_label
    args.training_claims = 2 * training_per_label
    args.binary_label_counts = {
        label: int(counts[label]) for label in BINARY_LABELS
    }
    args.binary_examples_used = 2 * balanced_per_label
    args.binary_examples_unused = len(examples) - args.binary_examples_used


def _experiment_plan(args: argparse.Namespace) -> dict[str, Any]:
    candidate_count = len(_candidate_specs(args))
    recipient_count = len(
        {
            value.strip()
            for value in args.isolated_recipients.split(",")
            if value.strip()
        }
    )
    condition_count = (
        2 if args.diagnostic_scope == "gate_training" else 2 + recipient_count
    )
    branches_per_seed = (
        args.training_claims
        * candidate_count
        * args.repeats
        * condition_count
    )
    return {
        "binary_label_counts": getattr(args, "binary_label_counts", None),
        "binary_examples_used": getattr(args, "binary_examples_used", None),
        "binary_examples_unused": getattr(args, "binary_examples_unused", None),
        "support_claims": 2 * args.support_per_label,
        "gate_training_claims": args.training_claims,
        "gate_training_candidate_events": args.training_claims * candidate_count,
        "final_evaluation_claims": (
            2 * args.evaluation_per_label - args.training_claims
        ),
        "candidate_count_per_claim": candidate_count,
        "diagnostic_conditions": condition_count,
        "repeats_per_seed": args.repeats,
        "diagnostic_branches_per_seed": branches_per_seed,
        "diagnostic_branches_two_seeds": 2 * branches_per_seed,
    }


def _candidate_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.candidate_scope == "single":
        return [
            {
                "kind": str(args.candidate_kind),
                "index": int(args.candidate_index),
            }
        ]
    return [
        *(
            {"kind": "trajectory", "index": index}
            for index in range(int(args.successful_topk))
        ),
        *(
            {"kind": "insight", "index": index}
            for index in range(int(args.insights_topk))
        ),
    ]


def _diagnostic_candidate_design(args: argparse.Namespace) -> dict[str, Any]:
    if args.candidate_scope == "all_retrieved":
        return {
            "candidate_scope": "all_retrieved",
            "candidate_specs": _candidate_specs(args),
        }
    return {
        "candidate_kind": str(args.candidate_kind),
        "candidate_index": int(args.candidate_index),
    }


def _evaluation_candidate_kinds(args: argparse.Namespace) -> tuple[str, ...]:
    if args.candidate_scope == "single":
        return (str(args.candidate_kind),)
    return tuple(
        kind
        for kind, topk in (
            ("trajectory", args.successful_topk),
            ("insight", args.insights_topk),
        )
        if int(topk) > 0
    )


def _expected_candidate_kind_ranks(
    args: argparse.Namespace,
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for spec in _candidate_specs(args):
        result.setdefault(str(spec["kind"]), []).append(int(spec["index"]) + 1)
    return result


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _stable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_fields(
    artifact: str,
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    mismatches = [
        f"{key}: stored={actual.get(key)!r}, requested={value!r}"
        for key, value in expected.items()
        if actual.get(key) != value
    ]
    if mismatches:
        details = "\n  - ".join(mismatches)
        raise RuntimeError(
            f"completed {artifact} does not match this pipeline:\n  - {details}\n"
            "Use the original arguments or choose new diagnostic/checkpoint/output paths."
        )


def _tail(path: Path, lines: int = 100) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"unable to read log: {exc}"
    return "\n".join(content.replace("\r", "\n").splitlines()[-lines:])


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "pipeline_schema": PIPELINE_SCHEMA,
        "data": str(args.data),
        "endpoint": args.endpoint,
        "model": args.model,
        "memory_dir": str(args.memory_dir),
        "diagnostic_results": str(args.diagnostic_results),
        "checkpoint": str(args.checkpoint),
        "output_dir": str(args.output_dir),
        "support_per_label": args.support_per_label,
        "evaluation_per_label": args.evaluation_per_label,
        "training_claims": args.training_claims,
        "use_all_binary_data": args.use_all_binary_data,
        "training_fraction": args.training_fraction,
        "final_evaluation_claims": 2 * args.evaluation_per_label - args.training_claims,
        "smoke_claims": args.smoke_claims,
        "skip_smoke": args.skip_smoke,
        "repeats": args.repeats,
        "split_seed": args.split_seed,
        "build_seed": args.build_seed,
        "selection_seed": args.selection_seed,
        "sample_seed_base": args.sample_seed_base,
        "retest_seed_base": args.retest_seed_base,
        "gate_evaluation_seed": args.gate_evaluation_seed,
        "temperature": args.temperature,
        "node_num": args.node_num,
        "graph_type": args.graph_type,
        "isolated_recipients": args.isolated_recipients,
        "successful_topk": args.successful_topk,
        "failed_topk": args.failed_topk,
        "insights_topk": args.insights_topk,
        "threshold": args.threshold,
        "embedding_model": args.embedding_model,
        "candidate_scope": args.candidate_scope,
        "candidate_specs": _candidate_specs(args),
        "evaluation_candidate_kinds": list(_evaluation_candidate_kinds(args)),
        "diagnostic_scope": args.diagnostic_scope,
        "kappa": args.kappa,
        "delta": args.delta,
        "max_drops": args.max_drops,
        "residual_noise_scale": args.residual_noise_scale,
        "bootstrap_samples": args.bootstrap_samples,
        "experiment_plan": _experiment_plan(args),
    }


def _prepare_manifest(args: argparse.Namespace) -> dict[str, Any]:
    path = args.output_dir / "pipeline_manifest.json"
    configuration = _configuration(args)
    expected = {
        "pipeline_schema": PIPELINE_SCHEMA,
        "configuration_hash": _stable_hash(configuration),
        "configuration": configuration,
    }
    if path.is_file():
        actual = _load_json(path)
        if actual is None or actual.get("configuration_hash") != expected["configuration_hash"]:
            raise SystemExit(
                "existing FEVER comparison uses different arguments; rerun the original "
                "command or choose a new --output-dir"
            )
        return actual
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"non-empty output directory has no pipeline manifest: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(path, expected)
    return expected


def _validate_diagnostic_run(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    sample_seed_base: int,
) -> None:
    manifest = _load_json(output_dir / "run_manifest.json")
    if manifest is None:
        raise RuntimeError(f"completed diagnostic has no readable manifest: {output_dir}")
    design = manifest.get("design")
    run = manifest.get("run")
    if not isinstance(design, Mapping) or not isinstance(run, Mapping):
        raise RuntimeError(f"completed diagnostic has an invalid manifest: {output_dir}")
    snapshot = _load_json(args.memory_dir / "causal_snapshot_manifest.json")
    if snapshot is None:
        raise RuntimeError("completed diagnostic snapshot has no readable manifest")
    recipients = [
        value.strip()
        for value in args.isolated_recipients.split(",")
        if value.strip()
    ]
    _require_fields(
        str(output_dir),
        design,
        {
            "benchmark": "FEVER_binary_offline",
            "source_md5": _file_md5(args.data),
            "snapshot_design_hash": snapshot.get("design_hash"),
            "memory_dir": str(args.memory_dir),
            "model": args.model,
            **_diagnostic_candidate_design(args),
            "collection_scope": args.diagnostic_scope,
            "repeats": args.repeats,
            "temperature": args.temperature,
            "node_num": args.node_num,
            "graph_type": args.graph_type,
            "successful_topk": args.successful_topk,
            "failed_topk": args.failed_topk,
            "insights_topk": args.insights_topk,
            "threshold": args.threshold,
            "embedding_model": args.embedding_model,
            "selection_seed": args.selection_seed,
            "recipients": recipients,
        },
    )
    expected_ids = [
        int(value)
        for value in snapshot.get("evaluation_ids", [])[: args.training_claims]
    ]
    if [int(value) for value in design.get("evaluation_ids", [])] != expected_ids:
        raise RuntimeError(
            f"completed {output_dir} does not contain the registered first "
            f"{args.training_claims} training claims"
        )
    _require_fields(
        str(output_dir),
        run,
        {
            "endpoint": args.endpoint,
            "sample_seed_base": sample_seed_base,
        },
    )


def _validate_completed_diagnostic(args: argparse.Namespace) -> None:
    snapshot = _load_json(args.memory_dir / "causal_snapshot_manifest.json")
    if snapshot is None:
        raise RuntimeError("completed diagnostic snapshot has no readable manifest")
    _require_fields(
        "diagnostic snapshot",
        snapshot,
        {
            "source_md5": _file_md5(args.data),
            "support_per_label": args.support_per_label,
            "evaluation_per_label": args.evaluation_per_label,
            "split_seed": args.split_seed,
            "build_seed": args.build_seed,
            "model": args.model,
            "graph_type": args.graph_type,
            "node_num": args.node_num,
            "successful_topk": args.successful_topk,
            "failed_topk": args.failed_topk,
            "insights_topk": args.insights_topk,
            "threshold": args.threshold,
            "embedding_model": args.embedding_model,
        },
    )
    _validate_diagnostic_run(
        args,
        args.diagnostic_results / f"seed{args.sample_seed_base}",
        sample_seed_base=args.sample_seed_base,
    )
    _validate_diagnostic_run(
        args,
        args.diagnostic_results / f"seed{args.retest_seed_base}_retest",
        sample_seed_base=args.retest_seed_base,
    )


def _diagnostic_complete(args: argparse.Namespace) -> bool:
    progress = _load_json(args.diagnostic_results / "experiment_progress.json")
    seed0 = args.diagnostic_results / f"seed{args.sample_seed_base}" / "branches.jsonl"
    retest = (
        args.diagnostic_results
        / f"seed{args.retest_seed_base}_retest"
        / "branches.jsonl"
    )
    complete = bool(
        progress
        and progress.get("status") == "completed"
        and seed0.is_file()
        and retest.is_file()
    )
    if complete:
        _validate_completed_diagnostic(args)
    return complete


def _checkpoint_complete(args: argparse.Namespace) -> bool:
    payload = _load_json(args.checkpoint)
    if payload is None:
        return False
    metadata = payload.get("training_metadata", {})
    if not isinstance(metadata, Mapping):
        return False
    structurally_complete = bool(
        payload.get("schema") == "gmemory-team-exposure-gate-v1"
        and metadata.get("training_task_ids")
        and metadata.get("candidate_kinds")
        and metadata.get("candidate_ranks")
        and metadata.get("training_contexts")
        and (
            args.candidate_scope == "single"
            or metadata.get("candidate_kind_ranks")
        )
    )
    if not structurally_complete:
        return False
    estimator_payload = payload.get("estimator", {})
    if not isinstance(estimator_payload, Mapping):
        return False
    estimator_config = estimator_payload.get("config", {})
    if not isinstance(estimator_config, Mapping):
        return False
    stored_residual_scale = estimator_config.get("residual_noise_scale")
    if stored_residual_scale is None:
        stored_residual_scale = (
            1.0 if estimator_config.get("include_residual_noise", True) else 0.0
        )
    if not math.isclose(
        float(stored_residual_scale),
        float(args.residual_noise_scale),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "gate checkpoint residual_noise_scale="
            f"{stored_residual_scale!r}, but this pipeline requests "
            f"{args.residual_noise_scale!r}"
        )
    seed0 = args.diagnostic_results / f"seed{args.sample_seed_base}" / "branches.jsonl"
    retest = (
        args.diagnostic_results
        / f"seed{args.retest_seed_base}_retest"
        / "branches.jsonl"
    )
    _require_fields(
        "gate checkpoint",
        metadata,
        {
            "metric": "success",
            "training_sources": [str(seed0), str(retest)],
        },
    )
    seed_manifest = _load_json(seed0.parent / "run_manifest.json")
    if seed_manifest is None or not isinstance(seed_manifest.get("design"), Mapping):
        raise RuntimeError("cannot validate checkpoint: seed diagnostic manifest is missing")
    expected_ids = sorted(
        int(value) for value in seed_manifest["design"]["evaluation_ids"]
    )
    if sorted(int(value) for value in metadata["training_task_ids"]) != expected_ids:
        raise RuntimeError(
            "gate checkpoint training_task_ids do not match the diagnostic training claims"
        )
    trained_kinds = {str(value) for value in metadata["candidate_kinds"]}
    expected_kind_ranks = _expected_candidate_kind_ranks(args)
    raw_kind_ranks = metadata.get("candidate_kind_ranks")
    if isinstance(raw_kind_ranks, Mapping):
        try:
            trained_kind_ranks = {
                str(kind): {int(rank) for rank in ranks}
                for kind, ranks in raw_kind_ranks.items()
            }
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "gate checkpoint candidate_kind_ranks is invalid"
            ) from exc
    else:
        global_ranks = {int(value) for value in metadata["candidate_ranks"]}
        trained_kind_ranks = {
            kind: set(global_ranks) for kind in trained_kinds
        }
    missing_pairs = [
        f"{kind}[rank={rank}]"
        for kind, ranks in expected_kind_ranks.items()
        for rank in ranks
        if kind not in trained_kinds
        or rank not in trained_kind_ranks.get(kind, set())
    ]
    if missing_pairs:
        raise RuntimeError(
            "gate checkpoint has no matched counterfactual training coverage for: "
            + ", ".join(missing_pairs)
        )
    return True


def _evaluation_complete(path: Path) -> bool:
    progress = _load_json(path / "progress.json")
    return bool(
        progress
        and progress.get("status") == "completed"
        and (path / "memory_gate.jsonl").is_file()
        and (path / "summary.json").is_file()
        and (path / "run_manifest.json").is_file()
    )


def _evaluation_resume(path: Path) -> bool:
    if not path.exists():
        return False
    if not (path / "run_manifest.json").is_file():
        recoverable = {path / "run_manifest.json.tmp"}
        if path.is_dir() and set(path.iterdir()).issubset(recoverable):
            return False
        raise RuntimeError(
            f"cannot resume {path}: output exists but run_manifest.json is missing"
        )
    return not _evaluation_complete(path)


def _comparison_complete(path: Path) -> bool:
    payload = _load_json(path)
    return bool(payload and int(payload.get("paired_tasks", 0)) > 0)


def _child_progress(args: argparse.Namespace, stage: str) -> dict[str, Any] | None:
    paths = {
        "diagnostic": args.diagnostic_results / "experiment_progress.json",
        "native_gmemory": args.output_dir / "native_gmemory" / "progress.json",
        "learned_gate": args.output_dir / "learned_gate" / "progress.json",
    }
    path = paths.get(stage)
    if path is None:
        return None
    payload = _load_json(path)
    if payload is None:
        return None
    result: dict[str, Any] = {"path": str(path), "content": payload}
    if stage == "diagnostic":
        active = payload.get("current_stage")
        nested_paths = {
            "snapshot": args.memory_dir / "causal_snapshot_progress.json",
            "smoke": (
                args.diagnostic_results
                / f"smoke_seed{args.sample_seed_base}"
                / "collection_progress.json"
            ),
            "seed0": (
                args.diagnostic_results
                / f"seed{args.sample_seed_base}"
                / "collection_progress.json"
            ),
            "seed1000_retest": (
                args.diagnostic_results
                / f"seed{args.retest_seed_base}_retest"
                / "collection_progress.json"
            ),
        }
        nested_path = nested_paths.get(str(active))
        nested = _load_json(nested_path) if nested_path is not None else None
        if nested_path is not None and nested is not None:
            result["nested"] = {"path": str(nested_path), "content": nested}
    return result


def _run_command(
    stage: str,
    command: Sequence[str],
    log_path: Path,
    *,
    api_key: str | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = shlex.join(command)
    header = (
        f"\n{'=' * 20} {_timestamp()} {stage} {'=' * 20}\n"
        f"cwd: {Path.cwd()}\ncommand: {rendered}\n\n"
    ).encode("utf-8")
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    if api_key is not None:
        environment["OPENAI_API_KEY"] = api_key
    with log_path.open("ab") as log_handle:
        log_handle.write(header)
        log_handle.flush()
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
            bufsize=0,
        )
        assert process.stdout is not None
        try:
            try:
                while True:
                    chunk = os.read(process.stdout.fileno(), 4096)
                    if not chunk:
                        break
                    log_handle.write(chunk)
                    log_handle.flush()
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                returncode = process.wait()
            except KeyboardInterrupt:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        process.wait(timeout=15)
                raise
        finally:
            process.stdout.close()
    if returncode != 0:
        raise StageFailure(stage, returncode, command, log_path, _tail(log_path))


def _diagnostic_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "causal_diagnostic.fever_oracle_recipient.run_all",
        "--data",
        str(args.data),
        "--endpoint",
        args.endpoint,
        "--model",
        args.model,
        "--memory-dir",
        str(args.memory_dir),
        "--results-root",
        str(args.diagnostic_results),
        "--support-per-label",
        str(args.support_per_label),
        "--evaluation-per-label",
        str(args.evaluation_per_label),
        "--claims",
        str(args.training_claims),
        "--smoke-claims",
        str(args.smoke_claims),
        "--split-seed",
        str(args.split_seed),
        "--build-seed",
        str(args.build_seed),
        "--selection-seed",
        str(args.selection_seed),
        "--sample-seed-base",
        str(args.sample_seed_base),
        "--retest-seed-base",
        str(args.retest_seed_base),
        "--temperature",
        str(args.temperature),
        "--repeats",
        str(args.repeats),
        "--node-num",
        str(args.node_num),
        "--graph-type",
        args.graph_type,
        "--isolated-recipients",
        args.isolated_recipients,
        "--successful-topk",
        str(args.successful_topk),
        "--failed-topk",
        str(args.failed_topk),
        "--insights-topk",
        str(args.insights_topk),
        "--threshold",
        str(args.threshold),
        "--embedding-model",
        args.embedding_model,
        "--candidate-kind",
        args.candidate_kind,
        "--candidate-index",
        str(args.candidate_index),
        "--delta",
        str(args.delta),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
    ]
    if args.skip_smoke:
        command.append("--skip-smoke")
    if args.candidate_scope == "all_retrieved":
        command.append("--all-candidates")
    if args.diagnostic_scope == "gate_training":
        command.append("--gate-training-only")
    return command


def _train_command(args: argparse.Namespace) -> list[str]:
    seed0 = args.diagnostic_results / f"seed{args.sample_seed_base}" / "branches.jsonl"
    retest = (
        args.diagnostic_results
        / f"seed{args.retest_seed_base}_retest"
        / "branches.jsonl"
    )
    return [
        sys.executable,
        "-u",
        "-m",
        "causal_memory_control.train_gate",
        "--input",
        str(seed0),
        str(retest),
        "--metric",
        "success",
        "--output",
        str(args.checkpoint),
        "--residual-noise-scale",
        str(args.residual_noise_scale),
    ]


def _evaluation_command(
    args: argparse.Namespace,
    *,
    mode: str,
    output: Path,
    resume: bool,
) -> list[str]:
    final_claims = 2 * args.evaluation_per_label - args.training_claims
    command = [
        sys.executable,
        "-u",
        "-m",
        "causal_memory_control.fever_experiment",
        "--data",
        str(args.data),
        "--memory-dir",
        str(args.memory_dir),
        "--output-dir",
        str(output),
        "--endpoint",
        args.endpoint,
        "--model",
        args.model,
        "--evaluation-offset",
        str(args.training_claims),
        "--claims",
        str(final_claims),
        "--mode",
        mode,
        "--seed",
        str(args.gate_evaluation_seed),
        "--temperature",
        str(args.temperature),
        "--node-num",
        str(args.node_num),
        "--graph-type",
        args.graph_type,
        "--successful-topk",
        str(args.successful_topk),
        "--failed-topk",
        str(args.failed_topk),
        "--insights-topk",
        str(args.insights_topk),
        "--threshold",
        str(args.threshold),
        "--embedding-model",
        args.embedding_model,
        "--candidate-kinds",
        ",".join(_evaluation_candidate_kinds(args)),
        "--delta",
        str(args.delta),
        "--kappa",
        str(args.kappa),
        "--max-drops",
        str(args.max_drops),
    ]
    if mode == "learned":
        baseline_cache = args.output_dir / "native_gmemory" / "llm_cache.sqlite"
        command.extend(
            (
                "--checkpoint",
                str(args.checkpoint),
                "--cache-seed-from",
                str(baseline_cache),
            )
        )
    if resume:
        command.append("--resume")
    return command


def _compare_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "causal_memory_control.compare_gate_runs",
        "--baseline",
        str(args.output_dir / "native_gmemory" / "memory_gate.jsonl"),
        "--gate",
        str(args.output_dir / "learned_gate" / "memory_gate.jsonl"),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--seed",
        str(args.gate_evaluation_seed),
        "--output",
        str(args.output_dir / "comparison.json"),
    ]


def _status(args: argparse.Namespace) -> dict[str, bool]:
    return {
        "diagnostic": _diagnostic_complete(args),
        "train_gate": _checkpoint_complete(args),
        "native_gmemory": _evaluation_complete(args.output_dir / "native_gmemory"),
        "learned_gate": _evaluation_complete(args.output_dir / "learned_gate"),
        "compare": _comparison_complete(args.output_dir / "comparison.json"),
    }


def _mark_progress(
    path: Path,
    state: dict[str, Any],
    *,
    status: str,
    current_stage: str | None,
    stage_status: str | None = None,
    error: Mapping[str, Any] | None = None,
) -> None:
    if current_stage is not None and stage_status is not None:
        state.setdefault("stages", {}).setdefault(current_stage, {}).update(
            status=stage_status,
            updated_at=_timestamp(),
        )
    state.update(
        status=status,
        current_stage=current_stage,
        updated_at=_timestamp(),
        error=dict(error) if error is not None else None,
    )
    _write_json(path, state)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    args.data = args.data.resolve()
    _resolve_data_plan(args)
    args.memory_dir = args.memory_dir.resolve()
    args.diagnostic_results = args.diagnostic_results.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    _validate_args(args)
    print(
        json.dumps(
            {"experiment_plan": _experiment_plan(args)},
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.plan_only:
        return args.output_dir / "pipeline_manifest.json"
    _prepare_manifest(args)

    progress_path = args.output_dir / "pipeline_progress.json"
    state = _load_json(progress_path) or {
        "pipeline_schema": PIPELINE_SCHEMA,
        "created_at": _timestamp(),
        "stages": {stage: {"status": "pending"} for stage in STAGES},
    }
    logs = args.output_dir / "logs"
    baseline = args.output_dir / "native_gmemory"
    learned = args.output_dir / "learned_gate"
    comparison = args.output_dir / "comparison.json"
    current_stage: str | None = None
    current_command: list[str] = []
    current_log: Path | None = None
    progress: tqdm[Any] | None = None
    try:
        statuses = _status(args)
        for stage, complete in statuses.items():
            if complete:
                state.setdefault("stages", {}).setdefault(stage, {})[
                    "status"
                ] = "completed"
        _mark_progress(
            progress_path,
            state,
            status="running",
            current_stage=None,
        )
        progress = tqdm(
            total=len(STAGES),
            initial=sum(statuses.values()),
            desc="FEVER native-vs-gate pipeline",
            unit="stage",
            dynamic_ncols=True,
        )
        for stage in STAGES:
            if _status(args)[stage]:
                progress.set_postfix_str(f"{stage}: already complete", refresh=True)
                continue
            current_stage = stage
            if stage == "diagnostic":
                current_command = _diagnostic_command(args)
            elif stage == "train_gate":
                if not _diagnostic_complete(args):
                    raise RuntimeError("diagnostic branches are incomplete")
                current_command = _train_command(args)
            elif stage == "native_gmemory":
                current_command = _evaluation_command(
                    args,
                    mode="always_keep",
                    output=baseline,
                    resume=_evaluation_resume(baseline),
                )
            elif stage == "learned_gate":
                if not _checkpoint_complete(args):
                    raise RuntimeError("gate checkpoint is incomplete")
                if not (baseline / "llm_cache.sqlite").is_file():
                    raise RuntimeError("native baseline cache is missing")
                current_command = _evaluation_command(
                    args,
                    mode="learned",
                    output=learned,
                    resume=_evaluation_resume(learned),
                )
            else:
                if not _evaluation_complete(baseline) or not _evaluation_complete(learned):
                    raise RuntimeError("paired evaluation runs are incomplete")
                current_command = _compare_command(args)

            current_log = logs / f"{stage}.log"
            progress.set_postfix_str(stage, refresh=True)
            _mark_progress(
                progress_path,
                state,
                status="running",
                current_stage=stage,
                stage_status="running",
            )
            _run_command(
                stage,
                current_command,
                current_log,
                api_key=args.api_key,
            )
            if not _status(args)[stage]:
                raise RuntimeError(
                    f"stage {stage} exited successfully but its completion artifacts are missing"
                )
            _mark_progress(
                progress_path,
                state,
                status="running",
                current_stage=stage,
                stage_status="completed",
            )
            progress.update(1)
        progress.close()
    except BaseException as exc:
        if progress is not None:
            progress.close()
        failure_stage = current_stage or "initialization"
        if isinstance(exc, StageFailure):
            failure_stage = exc.stage
            details = {
                "type": type(exc).__name__,
                "message": str(exc),
                "exit_code": exc.returncode,
                "command": list(exc.command),
                "command_shell": shlex.join(exc.command),
                "log_path": str(exc.log_path),
                "log_tail": exc.tail,
            }
        else:
            details = {
                "type": type(exc).__name__,
                "message": str(exc),
                "command": current_command,
                "command_shell": shlex.join(current_command) if current_command else None,
                "log_path": str(current_log) if current_log else None,
                "log_tail": _tail(current_log) if current_log and current_log.exists() else None,
            }
        details.update(
            stage=failure_stage,
            child_progress=_child_progress(args, failure_stage),
            traceback=traceback.format_exc(),
        )
        _mark_progress(
            progress_path,
            state,
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            current_stage=failure_stage,
            stage_status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=details,
        )
        print(json.dumps(details, ensure_ascii=False, indent=2), file=sys.stderr)
        raise

    _mark_progress(
        progress_path,
        state,
        status="completed",
        current_stage=None,
    )
    report = _load_json(comparison)
    print(
        json.dumps(
            {
                "pipeline": str(args.output_dir),
                "native_summary": str(baseline / "summary.json"),
                "learned_summary": str(learned / "summary.json"),
                "comparison": str(comparison),
                "result": report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return comparison


if __name__ == "__main__":
    main()
