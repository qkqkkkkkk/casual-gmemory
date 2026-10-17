#!/usr/bin/env python3
"""Resumable native-GMemory versus improved-gate experiment on HotpotQA."""

from __future__ import annotations

import argparse
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

from causal_diagnostic.hotpotqa_oracle_recipient.data import load_hotpotqa
from causal_diagnostic.hotpotqa_oracle_recipient import run_all as diagnostic
from causal_diagnostic.hotpotqa_oracle_recipient.provenance import stable_hash
from causal_memory_control.jsonl import read_jsonl
from causal_memory_control import hotpotqa_experiment as evaluation
from causal_memory_control.compare_gate_runs import file_sha256
from causal_memory_control.task_state import TASK_STATE_POLICY, archive_legacy_evaluations


PIPELINE_SCHEMA = "gmemory-hotpotqa-gate-comparison-pipeline-v1"
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


def parse_args(
    argv: Sequence[str] | None = None, *, description: str | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/hotpotqa/raw/hotpot_dev_distractor_v1.json"),
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:11436/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=Path("causal_diagnostic/memory_snapshots/hotpotqa_gate_support50_7b_v2/g-memory"),
    )
    parser.add_argument(
        "--diagnostic-results",
        type=Path,
        default=Path("causal_diagnostic/results/hotpotqa_gate_training_7b_v2"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("causal_memory_control/checkpoints/hotpotqa_gate_v2.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("causal_memory_control/results/hotpotqa_gate_v2"),
    )
    parser.add_argument("--support-count", type=int, default=50)
    parser.add_argument(
        "--evaluation-count",
        type=int,
        default=500,
        help="Post-support questions registered for gate training plus final test",
    )
    parser.add_argument(
        "--training-fraction",
        type=float,
        default=0.8,
        help="Fraction of registered post-support questions used for gate diagnostics",
    )
    parser.add_argument(
        "--use-all-data",
        action="store_true",
        help="Use every valid source question once across support/training/final test",
    )
    parser.add_argument("--smoke-claims", type=int, default=2)
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
        "--graph-type", choices=("Chain", "FullConnected", "Debate", "Star"), default="Chain"
    )
    parser.add_argument("--successful-topk", type=int, default=3)
    parser.add_argument("--failed-topk", type=int, default=0)
    parser.add_argument("--insights-topk", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--estimator", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--feature-version", type=int, choices=(2,3), default=2)
    parser.add_argument("--semantic-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--semantic-revision", default=None)
    parser.add_argument("--semantic-cache", type=Path, default=None)
    parser.add_argument("--semantic-chunk-tokens", type=int, default=192)
    parser.add_argument("--mlp-hidden-sizes", default="128,64")
    parser.add_argument("--mlp-epochs", type=int, default=80)
    parser.add_argument("--mlp-learning-rate", type=float, default=0.001)
    parser.add_argument("--mlp-batch-tasks", type=int, default=16)
    parser.add_argument("--mlp-device", default="cpu")
    parser.add_argument("--baseline-cache-seed-from", type=Path, default=None)
    parser.add_argument("--kappa", default="auto", help="auto or a non-negative number")
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--max-drops", type=int, default=1)
    parser.add_argument("--residual-noise-scale", type=float, default=0.0)
    parser.add_argument("--direct-utility-weight", type=float, default=0.5)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-margin", type=float, default=0.05)
    parser.add_argument("--contrastive-min-gap", type=float, default=0.05)
    parser.add_argument("--contrastive-epochs", type=int, default=20)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--kappa-grid", default="0,0.1,0.25,0.5,1.0,1.96")
    parser.add_argument("--max-changed-harm-rate", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--plan-only", action="store_true")
    parser.set_defaults(**dict(defaults or {}))
    return parser.parse_args(argv)


def _resolve_plan(args: argparse.Namespace) -> None:
    examples = load_hotpotqa(args.data)
    args.valid_source_questions = len(examples)
    if args.use_all_data:
        args.evaluation_count = len(examples) - args.support_count
    args.training_claims = math.floor(args.evaluation_count * args.training_fraction)
    args.final_claims = args.evaluation_count - args.training_claims


def _validate_args(args: argparse.Namespace) -> None:
    if not args.data.is_file():
        raise SystemExit(f"HotpotQA data does not exist: {args.data}")
    if args.support_count < 1 or args.evaluation_count < 2:
        raise SystemExit("support count must be positive and evaluation count at least two")
    if args.support_count + args.evaluation_count > args.valid_source_questions:
        raise SystemExit(
            f"requested {args.support_count + args.evaluation_count} questions, "
            f"but loader found {args.valid_source_questions} valid questions"
        )
    if not 0.0 < args.training_fraction < 1.0 or min(args.training_claims, args.final_claims) < 1:
        raise SystemExit("--training-fraction must leave non-empty training and final sets")
    if not 1 <= args.smoke_claims <= args.training_claims:
        raise SystemExit("--smoke-claims must be in [1, training questions]")
    if args.repeats < 1 or args.node_num < 2:
        raise SystemExit("--repeats must be positive and --node-num at least two")
    if min(args.successful_topk, args.failed_topk, args.insights_topk) < 0:
        raise SystemExit("retrieval top-k values must be non-negative")
    if args.successful_topk + args.insights_topk < 1:
        raise SystemExit("at least one successful trajectory or insight must be retrieved")
    if min(
        args.temperature,
        args.delta,
        args.residual_noise_scale,
        args.contrastive_weight,
        args.contrastive_margin,
        args.contrastive_min_gap,
    ) < 0:
        raise SystemExit("non-negative experiment settings received a negative value")
    if not 0.0 <= args.direct_utility_weight <= 1.0:
        raise SystemExit("--direct-utility-weight must be in [0,1]")
    if not 0.0 < args.calibration_fraction < 1.0:
        raise SystemExit("--calibration-fraction must be in (0,1)")
    if args.max_drops != 1:
        raise SystemExit("single-candidate intervention training requires --max-drops 1")
    if args.training_claims < 5:
        raise SystemExit("at least five diagnostic tasks are needed for a calibration split")
    if args.contrastive_epochs < 0 or args.bootstrap_samples < 100:
        raise SystemExit("contrastive epochs must be non-negative; bootstrap samples must be >= 100")
    if args.failed_topk != 0:
        raise SystemExit("MacNet does not expose failed trajectories; use --failed-topk 0")
    if args.estimator == "mlp" and args.feature_version != 3:
        raise SystemExit("--estimator mlp requires --feature-version 3")
    try:
        widths = [int(value) for value in args.mlp_hidden_sizes.split(",")]
    except ValueError as exc:
        raise SystemExit("--mlp-hidden-sizes must be comma-separated positive integers") from exc
    if (not widths or min(widths) < 1 or args.mlp_epochs < 1
            or args.mlp_batch_tasks < 1 or args.semantic_chunk_tokens < 8
            or not math.isfinite(args.mlp_learning_rate) or args.mlp_learning_rate <= 0):
        raise SystemExit("invalid semantic encoder or MLP optimization settings")
    if not 0 <= args.max_changed_harm_rate <= 1:
        raise SystemExit("--max-changed-harm-rate must be in [0,1]")
    if set(range(args.sample_seed_base, args.sample_seed_base + args.repeats)) & set(
        range(args.retest_seed_base, args.retest_seed_base + args.repeats)
    ):
        raise SystemExit("diagnostic seed ranges must be disjoint")
    if args.kappa != "auto":
        try:
            if not math.isfinite(float(args.kappa)) or float(args.kappa) < 0:
                raise ValueError
        except ValueError as exc:
            raise SystemExit("--kappa must be auto or a non-negative number") from exc


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _stable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _tail(path: Path, lines: int = 100) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"unable to read log: {exc}"
    return "\n".join(text.replace("\r", "\n").splitlines()[-lines:])


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    candidate_count = args.successful_topk + args.insights_topk
    branches_per_seed = args.training_claims * candidate_count * args.repeats * 2
    return {
        "valid_source_questions": args.valid_source_questions,
        "support_questions": args.support_count,
        "gate_training_questions": args.training_claims,
        "policy_calibration_fraction_within_training": args.calibration_fraction,
        "final_test_questions": args.final_claims,
        "candidates_per_question": candidate_count,
        "diagnostic_branches_per_seed": branches_per_seed,
        "diagnostic_branches_two_seeds": 2 * branches_per_seed,
        "baseline_final_inferences": args.final_claims,
        "gate_final_inferences": args.final_claims,
    }


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    ignored = {"api_key"}
    # Preserve the identity of v2 runs started before optional semantic arms
    # existed. They must still resume after updating the code.
    if args.estimator == "linear" and args.feature_version == 2:
        ignored.update({"estimator", "feature_version", "semantic_model", "semantic_revision", "semantic_cache",
                        "semantic_chunk_tokens", "mlp_hidden_sizes", "mlp_epochs", "mlp_learning_rate",
                        "mlp_batch_tasks", "mlp_device"})
    if args.baseline_cache_seed_from is None:
        ignored.add("baseline_cache_seed_from")
    if args.max_changed_harm_rate == 1.0:
        ignored.add("max_changed_harm_rate")
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in ignored and key != "plan_only"
    } | {"pipeline_schema": PIPELINE_SCHEMA, "experiment_plan": _plan(args),
         "source_md5": diagnostic._file_md5(args.data)}


def _prepare_manifest(args: argparse.Namespace) -> None:
    path = args.output_dir / "pipeline_manifest.json"
    configuration = _configuration(args)
    expected_hash = _stable_hash(configuration)
    if path.is_file():
        existing = _load_json(path)
        if existing is None or existing.get("configuration_hash") != expected_hash:
            raise SystemExit(
                "existing HotpotQA comparison uses different arguments; "
                "use its original command or choose a new --output-dir"
            )
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"non-empty output directory has no pipeline manifest: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        path,
        {
            "pipeline_schema": PIPELINE_SCHEMA,
            "configuration_hash": expected_hash,
            "configuration": configuration,
        },
    )


def _diagnostic_complete(args: argparse.Namespace) -> bool:
    progress = _load_json(args.diagnostic_results / "experiment_progress.json")
    complete = bool(
        progress
        and progress.get("status") == "completed"
        and (args.diagnostic_results / f"seed{args.sample_seed_base}" / "branches.jsonl").is_file()
        and (args.diagnostic_results / f"seed{args.retest_seed_base}_retest" / "branches.jsonl").is_file()
        and (args.memory_dir / "causal_snapshot_manifest.json").is_file()
    )
    if complete:
        command = _diagnostic_command(args)
        child_args = diagnostic.parse_args(command[4:])
        diagnostic._validate_completed_snapshot(child_args)
        snapshot = _load_json(args.memory_dir / "causal_snapshot_manifest.json")
        expected_ids = snapshot["evaluation_ids"][:args.training_claims]
        for seed, suffix in ((args.sample_seed_base, ""), (args.retest_seed_base, "_retest")):
            directory = args.diagnostic_results / f"seed{seed}{suffix}"
            diagnostic._validate_completed_run(
                child_args, directory, claims=args.training_claims,
                repeats=args.repeats, sample_seed_base=seed,
            )
            manifest = _load_json(directory / "run_manifest.json")
            _require_fields("diagnostic", manifest["design"], {
                "evaluation_ids": expected_ids,
                "benchmark": "HotpotQA_distractor_offline",
                "source_md5": snapshot["source_md5"],
                "snapshot_design_hash": snapshot["design_hash"],
            })
            if stable_hash(manifest["design"]) != manifest.get("design_hash"):
                raise RuntimeError("invalid diagnostic design hash")
    return complete


def _require_fields(name: str, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    mismatches = {key: {"stored": actual.get(key), "requested": value}
                  for key, value in expected.items() if actual.get(key) != value}
    if mismatches:
        raise RuntimeError(f"{name} is incompatible: {mismatches}; choose new artifact paths")


def _checkpoint_complete(
    args: argparse.Namespace, *, benchmark: str = "HotpotQA_distractor_offline",
    metric: str = "team_score",
    sources: Sequence[Path] | None = None,
) -> bool:
    payload = _load_json(args.checkpoint)
    if not args.checkpoint.exists():
        return False
    if payload is None or payload.get("schema") != "gmemory-team-exposure-gate-v1":
        raise RuntimeError(f"unreadable or incompatible checkpoint: {args.checkpoint}; use a new path")
    metadata = payload.get("training_metadata", {})
    contexts = metadata.get("training_contexts", ()) if isinstance(metadata, Mapping) else ()
    complete = bool(
        isinstance(metadata, Mapping)
        and metadata.get("training_task_ids")
        and metadata.get("candidate_kind_ranks")
        and metadata.get("policy_calibration", {}).get("enabled")
        and contexts
        and all(context.get("benchmark") == benchmark for context in contexts)
    )
    if not complete:
        raise RuntimeError(f"incomplete or incompatible existing checkpoint: {args.checkpoint}; use a new path")
    _require_fields("checkpoint features", payload.get("feature_builder", {}), {
        "type": f"team-exposure-features-v{args.feature_version}", "hash_dimensions": 256,
    })
    if args.feature_version == 3:
        _require_fields("semantic encoder", payload["feature_builder"]["encoder"], {
            "type":"sentence-transformer-chunk-mean-v1", "model":args.semantic_model,
            "revision":args.semantic_revision,"chunk_tokens":args.semantic_chunk_tokens,
        })
    snapshot = _load_json(args.memory_dir / "causal_snapshot_manifest.json")
    if snapshot is None:
        raise RuntimeError("checkpoint exists without its snapshot")
    sources = [str(path) for path in sources] if sources is not None else [
        str(args.diagnostic_results / f"seed{seed}{suffix}" / "branches.jsonl")
        for seed, suffix in ((args.sample_seed_base, ""), (args.retest_seed_base, "_retest"))]
    _require_fields("checkpoint metadata", metadata, {
        "metric": metric, "training_sources": sources,
        "training_source_sha256": {source: file_sha256(Path(source)) for source in sources},
        "training_task_ids": sorted(snapshot["evaluation_ids"][:args.training_claims]),
        "calibration_fraction": args.calibration_fraction,
        "kappa_grid": sorted({float(v) for v in args.kappa_grid.split(",")}),
    })
    estimator_fields = {
        key: getattr(args, key) for key in (
            "direct_utility_weight", "contrastive_weight", "contrastive_margin",
            "contrastive_min_gap", "residual_noise_scale",
        )
    }
    if args.estimator == "linear":
        estimator_fields.update(solver="ridge",bootstrap_unit="task",contrastive_epochs=args.contrastive_epochs)
        if payload["estimator"].get("schema") != "amortized-utility-estimator-v2":
            raise RuntimeError("checkpoint estimator type differs from the requested linear arm")
    else:
        estimator_fields.update(hidden_sizes=[int(v) for v in args.mlp_hidden_sizes.split(",")],
                                epochs=args.mlp_epochs,learning_rate=args.mlp_learning_rate,
                                batch_tasks=args.mlp_batch_tasks)
        if payload["estimator"].get("schema") != "mlp-utility-estimator-v1":
            raise RuntimeError("checkpoint estimator type differs from the requested MLP arm")
    _require_fields("checkpoint estimator", payload["estimator"]["config"], estimator_fields)
    _require_fields("policy calibration", metadata["policy_calibration"], {
        "max_drops": 1, "delta": args.delta,
    })
    if metadata["policy_calibration"].get("max_changed_harm_rate",1.0)!=args.max_changed_harm_rate:
        raise RuntimeError("checkpoint calibrated with a different harm constraint")
    for context in contexts:
        _require_fields("checkpoint context", context, {
            "snapshot_design_hash": snapshot["design_hash"],
            "source_md5": snapshot["source_md5"],
            **{key: getattr(args, key) for key in (
                "model", "graph_type", "node_num", "embedding_model", "successful_topk",
                "failed_topk", "insights_topk", "threshold",
            )},
        })
    return True


def _evaluation_complete(path: Path) -> bool:
    progress = _load_json(path / "progress.json")
    return bool(
        progress
        and progress.get("status") == "completed"
        and (path / "memory_gate.jsonl").is_file()
        and (path / "summary.json").is_file()
    )


def _validated_evaluation_complete(
    args: argparse.Namespace, path: Path, *, mode: str, command: Sequence[str] | None = None,
    runner: Any = evaluation, benchmark: str = "HotpotQA_distractor_offline",
    completed_field: str = "completed_questions",
) -> bool:
    if not _evaluation_complete(path):
        return False
    manifest = _load_json(path / "run_manifest.json")
    snapshot = _load_json(args.memory_dir / "causal_snapshot_manifest.json")
    if not manifest or not snapshot:
        raise RuntimeError(f"completed evaluation has no valid manifest: {path}")
    design, run = manifest.get("design", {}), manifest.get("run", {})
    if (runner._stable_hash(design) != manifest.get("design_hash")
            or runner._stable_hash(run) != manifest.get("run_hash")):
        raise RuntimeError(f"invalid evaluation manifest hash: {path}")
    expected_ids = snapshot["evaluation_ids"][args.training_claims:args.evaluation_count]
    _require_fields("completed evaluation", design, {
        "runner_schema": runner.RUNNER_SCHEMA,
        "task_state_policy": TASK_STATE_POLICY,
        "benchmark": benchmark,
        "source_md5": snapshot["source_md5"],
        "snapshot_design_hash": snapshot["design_hash"],
        "evaluation_ids": expected_ids,
        "evaluation_offset": args.training_claims,
        "endpoint": args.endpoint, "seed": args.gate_evaluation_seed,
        **{key: getattr(args, key) for key in (
            "model", "temperature", "graph_type", "node_num", "embedding_model",
            "successful_topk", "failed_topk", "insights_topk", "threshold",
        )},
        "candidate_kinds": [kind for kind, count in (
            ("trajectory", args.successful_topk), ("insight", args.insights_topk)
        ) if count > 0],
    })
    command = command or _evaluation_command(args, mode=mode, output=path, resume=False,
                                             runner_module=runner.__name__)
    child = runner.parse_args(command[4:])
    child.checkpoint = child.checkpoint.resolve() if child.checkpoint else None
    seed_cache = child.cache_seed_from.resolve() if child.cache_seed_from else None
    _require_fields("completed evaluation run", run, {
        "mode": mode, "delta": child.delta, "kappa": child.kappa,
        "max_drops": child.max_drops,
        "checkpoint": str(child.checkpoint) if child.checkpoint else None,
        "checkpoint_sha256": runner._file_sha256(child.checkpoint) if child.checkpoint else None,
        "cache_seed_from": str(seed_cache) if seed_cache else None,
        "cache_seed_sha256": runner._file_sha256(seed_cache) if seed_cache else None,
    })
    rows = read_jsonl(path / "memory_gate.jsonl")
    ids = [int(row["task_id"]) for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise RuntimeError(f"completed evaluation has incomplete or duplicate task IDs: {path}")
    for row in rows:
        if row.get("mode") != mode or row.get("run_metadata") != design:
            raise RuntimeError(f"evaluation log disagrees with its manifest: {path}")
    _require_fields("evaluation summary", _load_json(path / "summary.json") or {}, {
        completed_field: len(expected_ids), "mode": mode,
    })
    return True


def _evaluation_resume(path: Path) -> bool:
    if not path.exists():
        return False
    if not (path / "run_manifest.json").is_file():
        if path.is_dir() and not any(path.iterdir()):
            return False
        raise RuntimeError(f"cannot resume {path}: run_manifest.json is missing")
    return not _evaluation_complete(path)


def _comparison_complete(path: Path) -> bool:
    result = _load_json(path)
    baseline = path.parent / "native_gmemory" / "memory_gate.jsonl"
    gate = path.parent / "learned_gate" / "memory_gate.jsonl"
    return bool(
        result and int(result.get("paired_tasks", 0)) > 0 and result.get("paired_control_valid")
        and baseline.is_file() and gate.is_file()
        and result.get("baseline_log_sha256") == file_sha256(baseline)
        and result.get("gate_log_sha256") == file_sha256(gate)
    )


def _status(args: argparse.Namespace) -> dict[str, bool]:
    return {
        "diagnostic": _diagnostic_complete(args),
        "train_gate": _checkpoint_complete(args),
        "native_gmemory": _validated_evaluation_complete(args, args.output_dir / "native_gmemory", mode="always_keep"),
        "learned_gate": _validated_evaluation_complete(args, args.output_dir / "learned_gate", mode="learned"),
        "compare": _comparison_complete(args.output_dir / "comparison.json"),
    }


def _effective_kappa(args: argparse.Namespace) -> float:
    if args.kappa != "auto":
        return float(args.kappa)
    payload = _load_json(args.checkpoint)
    try:
        value = payload["training_metadata"]["policy_calibration"]["recommended_kappa"]
        return float(value)
    except (TypeError, KeyError, ValueError) as exc:
        raise RuntimeError("checkpoint has no calibrated recommended_kappa") from exc


def _run_command(
    stage: str,
    command: Sequence[str],
    log_path: Path,
    *,
    api_key: str | None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    if api_key is not None:
        environment["OPENAI_API_KEY"] = api_key
    header = (
        f"\n{'=' * 20} {_timestamp()} {stage} {'=' * 20}\n"
        f"cwd: {Path.cwd()}\ncommand: {shlex.join(command)}\n\n"
    ).encode("utf-8")
    with log_path.open("ab") as handle:
        handle.write(header)
        handle.flush()
        process = subprocess.Popen(
            list(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=environment, bufsize=0,
        )
        assert process.stdout is not None
        try:
            while True:
                chunk = os.read(process.stdout.fileno(), 4096)
                if not chunk:
                    break
                handle.write(chunk)
                handle.flush()
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
        sys.executable, "-u", "-m", "causal_diagnostic.hotpotqa_oracle_recipient.run_all",
        "--data", str(args.data), "--endpoint", args.endpoint, "--model", args.model,
        "--memory-dir", str(args.memory_dir), "--results-root", str(args.diagnostic_results),
        "--support-count", str(args.support_count), "--evaluation-count", str(args.evaluation_count),
        "--claims", str(args.training_claims), "--smoke-claims", str(args.smoke_claims),
        "--split-seed", str(args.split_seed), "--build-seed", str(args.build_seed),
        "--selection-seed", str(args.selection_seed),
        "--sample-seed-base", str(args.sample_seed_base),
        "--retest-seed-base", str(args.retest_seed_base),
        "--temperature", str(args.temperature), "--repeats", str(args.repeats),
        "--isolated-recipients", f"solver_0,solver_{args.node_num - 1}",
        "--node-num", str(args.node_num), "--graph-type", args.graph_type,
        "--successful-topk", str(args.successful_topk),
        "--failed-topk", str(args.failed_topk),
        "--insights-topk", str(args.insights_topk),
        "--threshold", str(args.threshold), "--embedding-model", args.embedding_model,
        "--all-candidates", "--gate-training-only",
    ]
    if args.skip_smoke:
        command.append("--skip-smoke")
    return command


def _train_command(args: argparse.Namespace, *, metric: str = "team_score",
                   sources: Sequence[Path] | None = None) -> list[str]:
    seed0 = args.diagnostic_results / f"seed{args.sample_seed_base}" / "branches.jsonl"
    retest = args.diagnostic_results / f"seed{args.retest_seed_base}_retest" / "branches.jsonl"
    return [
        sys.executable, "-u", "-m", "causal_memory_control.train_gate",
        "--input", *(str(path) for path in (sources if sources is not None else (seed0, retest))), "--metric", metric,
        "--output", str(args.checkpoint),
        "--memory-dir", str(args.memory_dir),
        "--feature-version", str(args.feature_version), "--estimator", args.estimator,
        "--solver", "ridge", "--bootstrap-unit", "task",
        "--semantic-model",args.semantic_model,"--semantic-chunk-tokens",str(args.semantic_chunk_tokens),
        "--mlp-hidden-sizes",args.mlp_hidden_sizes,"--mlp-epochs",str(args.mlp_epochs),
        "--mlp-learning-rate",str(args.mlp_learning_rate),"--mlp-batch-tasks",str(args.mlp_batch_tasks),
        "--mlp-device",args.mlp_device,
        "--residual-noise-scale", str(args.residual_noise_scale),
        "--direct-utility-weight", str(args.direct_utility_weight),
        "--contrastive-weight", str(args.contrastive_weight),
        "--contrastive-margin", str(args.contrastive_margin),
        "--contrastive-min-gap", str(args.contrastive_min_gap),
        "--contrastive-epochs", str(args.contrastive_epochs),
        "--calibration-fraction", str(args.calibration_fraction),
        "--kappa-grid", args.kappa_grid,
        "--calibration-max-drops", str(args.max_drops),
        "--calibration-delta", str(args.delta),
        "--max-changed-harm-rate",str(args.max_changed_harm_rate),
    ] + (["--semantic-revision",args.semantic_revision] if args.semantic_revision else []) + (
        ["--semantic-cache",str(args.semantic_cache)] if args.semantic_cache else [])


def _evaluation_command(
    args: argparse.Namespace,
    *,
    mode: str,
    output: Path,
    resume: bool,
    runner_module: str = "causal_memory_control.hotpotqa_experiment",
) -> list[str]:
    command = [
        sys.executable, "-u", "-m", runner_module,
        "--data", str(args.data), "--memory-dir", str(args.memory_dir),
        "--output-dir", str(output), "--endpoint", args.endpoint, "--model", args.model,
        "--evaluation-offset", str(args.training_claims), "--claims", str(args.final_claims),
        "--mode", mode, "--seed", str(args.gate_evaluation_seed),
        "--temperature", str(args.temperature), "--node-num", str(args.node_num),
        "--graph-type", args.graph_type, "--successful-topk", str(args.successful_topk),
        "--failed-topk", str(args.failed_topk), "--insights-topk", str(args.insights_topk),
        "--threshold", str(args.threshold), "--embedding-model", args.embedding_model,
        "--candidate-kinds", ",".join(
            kind for kind, count in (("trajectory", args.successful_topk), ("insight", args.insights_topk)) if count > 0
        ),
        "--delta", str(args.delta), "--kappa", str(_effective_kappa(args) if mode == "learned" else 0.0),
        "--max-drops", str(
            _load_json(args.checkpoint)["training_metadata"]["policy_calibration"]["recommended_max_drops"]
            if mode == "learned" and args.kappa == "auto" else args.max_drops
        ),
    ]
    if mode == "learned":
        command.extend(
            (
                "--checkpoint", str(args.checkpoint),
                "--cache-seed-from", str(args.output_dir / "native_gmemory" / "llm_cache.sqlite"),
            )
        )
    elif args.baseline_cache_seed_from is not None:
        command.extend(("--cache-seed-from",str(args.baseline_cache_seed_from)))
    if args.semantic_cache is not None:
        command.extend(("--semantic-cache",str(args.semantic_cache)))
    if resume:
        command.append("--resume")
    return command


def _compare_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, "-u", "-m", "causal_memory_control.compare_gate_runs",
        "--baseline", str(args.output_dir / "native_gmemory" / "memory_gate.jsonl"),
        "--gate", str(args.output_dir / "learned_gate" / "memory_gate.jsonl"),
        "--bootstrap-samples", str(args.bootstrap_samples),
        "--seed", str(args.gate_evaluation_seed),
        "--output", str(args.output_dir / "comparison.json"),
        "--require-no-drop-match",
    ]


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
            status=stage_status, updated_at=_timestamp()
        )
    state.update(
        status=status, current_stage=current_stage,
        updated_at=_timestamp(), error=dict(error) if error else None,
    )
    _write_json(path, state)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    args.data = args.data.resolve()
    _resolve_plan(args)
    args.memory_dir = args.memory_dir.resolve()
    args.diagnostic_results = args.diagnostic_results.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    _validate_args(args)
    print(json.dumps({"experiment_plan": _plan(args)}, ensure_ascii=False, indent=2))
    if args.plan_only:
        return args.output_dir / "pipeline_manifest.json"
    _prepare_manifest(args)
    archive_legacy_evaluations(args.output_dir, ("native_gmemory", "learned_gate"))

    progress_path = args.output_dir / "pipeline_progress.json"
    state = _load_json(progress_path) or {
        "pipeline_schema": PIPELINE_SCHEMA,
        "created_at": _timestamp(),
        "stages": {stage: {"status": "pending"} for stage in STAGES},
    }
    logs = args.output_dir / "logs"
    baseline = args.output_dir / "native_gmemory"
    learned = args.output_dir / "learned_gate"
    current_stage: str | None = None
    current_command: list[str] = []
    current_log: Path | None = None
    progress = None
    try:
        statuses = _status(args)
        for stage, complete in statuses.items():
            if complete:
                state.setdefault("stages", {}).setdefault(stage, {})["status"] = "completed"
        _mark_progress(progress_path, state, status="running", current_stage=None)
        progress = tqdm(
            total=len(STAGES), initial=sum(statuses.values()),
            desc="HotpotQA native-vs-gate pipeline", unit="stage", dynamic_ncols=True,
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
                    args, mode="always_keep", output=baseline, resume=_evaluation_resume(baseline)
                )
            elif stage == "learned_gate":
                if not _checkpoint_complete(args):
                    raise RuntimeError("improved gate checkpoint is incomplete")
                if not (baseline / "llm_cache.sqlite").is_file():
                    raise RuntimeError("native baseline cache is missing")
                current_command = _evaluation_command(
                    args, mode="learned", output=learned, resume=_evaluation_resume(learned)
                )
            else:
                if not _evaluation_complete(baseline) or not _evaluation_complete(learned):
                    raise RuntimeError("paired evaluation runs are incomplete")
                current_command = _compare_command(args)
            current_log = logs / f"{stage}.log"
            progress.set_postfix_str(stage, refresh=True)
            _mark_progress(
                progress_path, state, status="running", current_stage=stage, stage_status="running"
            )
            _run_command(stage, current_command, current_log, api_key=args.api_key)
            if not _status(args)[stage]:
                raise RuntimeError(f"stage {stage} exited but completion artifacts are missing")
            _mark_progress(
                progress_path, state, status="running", current_stage=stage, stage_status="completed"
            )
            progress.update(1)
        progress.close()
    except BaseException as exc:
        if progress is not None:
            progress.close()
        stage = exc.stage if isinstance(exc, StageFailure) else current_stage or "initialization"
        details = {
            "type": type(exc).__name__,
            "message": str(exc),
            "stage": stage,
            "command": list(exc.command) if isinstance(exc, StageFailure) else current_command,
            "command_shell": shlex.join(exc.command) if isinstance(exc, StageFailure) else (
                shlex.join(current_command) if current_command else None
            ),
            "log_path": str(exc.log_path) if isinstance(exc, StageFailure) else (
                str(current_log) if current_log else None
            ),
            "log_tail": exc.tail if isinstance(exc, StageFailure) else (
                _tail(current_log) if current_log and current_log.exists() else None
            ),
            "traceback": traceback.format_exc(),
        }
        if isinstance(exc, StageFailure):
            details["exit_code"] = exc.returncode
        _mark_progress(
            progress_path, state,
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            current_stage=stage,
            stage_status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=details,
        )
        print(json.dumps(details, ensure_ascii=False, indent=2), file=sys.stderr)
        raise

    _mark_progress(progress_path, state, status="completed", current_stage=None)
    report = _load_json(args.output_dir / "comparison.json")
    print(
        json.dumps(
            {
                "pipeline": str(args.output_dir),
                "selected_kappa": _effective_kappa(args),
                "native_summary": str(baseline / "summary.json"),
                "learned_summary": str(learned / "summary.json"),
                "comparison": str(args.output_dir / "comparison.json"),
                "result": report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return args.output_dir / "comparison.json"


if __name__ == "__main__":
    main()
