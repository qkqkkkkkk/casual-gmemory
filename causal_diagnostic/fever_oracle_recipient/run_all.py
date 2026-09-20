#!/usr/bin/env python3
"""Run and resume the complete offline-FEVER RQ2/RQ3/RQ4 pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Sequence

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_SCHEMA = "native-gmemory-macnet-fever-pipeline-v2"


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
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=Path(
            "causal_diagnostic/memory_snapshots/"
            "fever_evidence_support50_7b_v3/g-memory"
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            "causal_diagnostic/results/"
            "native_fever_rq234_pilot_7b_v3"
        ),
    )
    parser.add_argument("--support-per-label", type=int, default=25)
    parser.add_argument("--evaluation-per-label", type=int, default=50)
    parser.add_argument("--claims", type=int, default=40)
    parser.add_argument("--smoke-claims", type=int, default=4)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--build-seed", type=int, default=42)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--sample-seed-base", type=int, default=0)
    parser.add_argument("--retest-seed-base", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--node-num", type=int, default=3)
    parser.add_argument(
        "--isolated-recipients", default="solver_0,solver_2"
    )
    parser.add_argument(
        "--graph-type",
        choices=("Chain", "FullConnected", "Debate", "Star"),
        default="Chain",
    )
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
        "--all-candidates",
        action="store_true",
        help=(
            "Collect matched counterfactuals for every retrieved successful "
            "trajectory and insight"
        ),
    )
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.claims < 1 or args.smoke_claims < 1:
        raise SystemExit("--claims and --smoke-claims must be at least 1")
    if args.smoke_claims > args.claims:
        raise SystemExit("--smoke-claims cannot exceed --claims")
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.repeats < 4 or args.repeats % 2 != 0:
        raise SystemExit(
            "RQ3/RQ4 split-half analysis requires an even --repeats >= 4 "
            "(recommended: 6)"
        )
    if args.node_num < 2:
        raise SystemExit("--node-num must be at least 2")
    if args.candidate_index < 0:
        raise SystemExit("--candidate-index must be non-negative")
    if min(args.successful_topk, args.failed_topk, args.insights_topk) < 0:
        raise SystemExit("retrieval top-k values must be non-negative")
    if args.all_candidates and args.successful_topk + args.insights_topk < 1:
        raise SystemExit(
            "--all-candidates requires --successful-topk or --insights-topk > 0"
        )
    if args.temperature < 0:
        raise SystemExit("--temperature must be non-negative")


def _candidate_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.all_candidates:
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


def _candidate_design(args: argparse.Namespace) -> dict[str, Any]:
    if args.all_candidates:
        return {
            "candidate_scope": "all_retrieved",
            "candidate_specs": _candidate_specs(args),
        }
    return {
        "candidate_kind": str(args.candidate_kind),
        "candidate_index": int(args.candidate_index),
    }


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_fields(
    artifact: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    mismatches = [
        f"{key}: stored={actual.get(key)!r}, requested={value!r}"
        for key, value in expected.items()
        if actual.get(key) != value
    ]
    if mismatches:
        details = "\n  - ".join(mismatches)
        raise SystemExit(
            f"completed {artifact} does not match this command:\n  - {details}\n"
            "Use the original arguments or choose a new output path."
        )


def _validate_completed_snapshot(args: argparse.Namespace) -> None:
    manifest = _load_json(args.memory_dir / "causal_snapshot_manifest.json")
    if manifest is None:
        raise SystemExit("completed snapshot has no readable manifest")
    _require_fields(
        "snapshot",
        manifest,
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


def _validate_completed_run(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    claims: int,
    repeats: int,
    sample_seed_base: int,
) -> None:
    manifest = _load_json(output_dir / "run_manifest.json")
    if manifest is None:
        raise SystemExit(f"completed run has no readable manifest: {output_dir}")
    design = manifest.get("design")
    run = manifest.get("run")
    if not isinstance(design, dict) or not isinstance(run, dict):
        raise SystemExit(f"completed run has an invalid manifest: {output_dir}")
    _require_fields(
        str(output_dir),
        design,
        {
            "memory_dir": str(args.memory_dir),
            "model": args.model,
            **_candidate_design(args),
            "repeats": repeats,
            "temperature": args.temperature,
            "node_num": args.node_num,
            "graph_type": args.graph_type,
            "successful_topk": args.successful_topk,
            "failed_topk": args.failed_topk,
            "insights_topk": args.insights_topk,
            "threshold": args.threshold,
            "embedding_model": args.embedding_model,
            "selection_seed": args.selection_seed,
            "recipients": [
                value.strip()
                for value in args.isolated_recipients.split(",")
                if value.strip()
            ],
        },
    )
    if len(design.get("evaluation_ids", [])) != claims:
        raise SystemExit(
            f"completed {output_dir} has {len(design.get('evaluation_ids', []))} "
            f"claims, but this command requests {claims}; use a new results root"
        )
    _require_fields(
        str(output_dir),
        run,
        {
            "endpoint": args.endpoint,
            "sample_seed_base": sample_seed_base,
        },
    )


def _snapshot_complete(memory_dir: Path) -> bool:
    progress = _load_json(memory_dir / "causal_snapshot_progress.json")
    return bool(
        progress
        and progress.get("status") == "completed"
        and (memory_dir / "causal_snapshot_manifest.json").is_file()
    )


def _run_complete(output_dir: Path) -> bool:
    progress = _load_json(output_dir / "collection_progress.json")
    return bool(
        (output_dir / "oracle_recipient_analysis.json").is_file()
        and (progress is None or progress.get("status") == "completed")
    )


def _needs_snapshot_resume(memory_dir: Path) -> bool:
    return bool(
        memory_dir.is_dir()
        and (memory_dir / "causal_snapshot_progress.json").is_file()
        and (memory_dir / "support_runs.jsonl").is_file()
    )


def _needs_run_resume(output_dir: Path) -> bool:
    return bool(
        output_dir.is_dir()
        and (output_dir / "run_manifest.json").is_file()
        and (output_dir / "branches.jsonl").is_file()
    )


def _existing_cache_seed(output_dir: Path) -> str | None:
    manifest = _load_json(output_dir / "run_manifest.json")
    if not manifest:
        return None
    value = manifest.get("run", {}).get("cache_seed_from")
    return str(value) if value else None


def _tail_text(path: Path, limit: int = 80) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"unable to read log: {exc}"
    normalized = text.replace("\r", "\n")
    return "\n".join(normalized.splitlines()[-limit:])


def _run_command(stage: str, command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = shlex.join(command)
    header = (
        f"\n{'=' * 20} {_timestamp()} {stage} {'=' * 20}\n"
        f"cwd: {PROJECT_ROOT}\ncommand: {rendered}\n\n"
    ).encode("utf-8")
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("ab") as log_handle:
        log_handle.write(header)
        log_handle.flush()
        process = subprocess.Popen(
            list(command),
            cwd=str(PROJECT_ROOT),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
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
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        process.wait(timeout=10)
                raise
        finally:
            process.stdout.close()
    if returncode != 0:
        raise StageFailure(
            stage,
            returncode,
            command,
            log_path,
            _tail_text(log_path),
        )


def _base_runner_command(
    args: argparse.Namespace, claims: int, output: Path
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "causal_diagnostic.fever_oracle_recipient.run_experiment",
        "--test",
        str(args.data),
        "--claims",
        str(claims),
        "--memory-dir",
        str(args.memory_dir),
        "--endpoint",
        args.endpoint,
        "--model",
        args.model,
        "--candidate-kind",
        args.candidate_kind,
        "--candidate-index",
        str(args.candidate_index),
        "--selection-seed",
        str(args.selection_seed),
        "--temperature",
        str(args.temperature),
        "--node-num",
        str(args.node_num),
        "--isolated-recipients",
        args.isolated_recipients,
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
        "--delta",
        str(args.delta),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--confidence-level",
        str(args.confidence_level),
        "--output-dir",
        str(output),
    ]
    if args.all_candidates:
        command.append("--all-candidates")
    return command


def _mark_stage(
    state_path: Path,
    state: dict[str, Any],
    stage: str,
    status: str,
    **fields: Any,
) -> None:
    stages = state.setdefault("stages", {})
    stage_state = stages.setdefault(stage, {})
    stage_state.update(status=status, updated_at=_timestamp(), **fields)
    state.update(status="running", current_stage=stage, updated_at=_timestamp())
    _write_json_atomic(state_path, state)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    _validate_args(args)
    args.data = args.data.resolve()
    args.memory_dir = args.memory_dir.resolve()
    args.results_root = args.results_root.resolve()
    args.results_root.mkdir(parents=True, exist_ok=True)
    logs_dir = args.results_root / "logs"
    state_path = args.results_root / "experiment_progress.json"
    state = _load_json(state_path) or {
        "orchestrator_schema": ORCHESTRATOR_SCHEMA,
        "created_at": _timestamp(),
        "stages": {},
    }
    state["configuration"] = {
        "data": str(args.data),
        "memory_dir": str(args.memory_dir),
        "results_root": str(args.results_root),
        "endpoint": args.endpoint,
        "model": args.model,
        "claims": args.claims,
        "temperature": args.temperature,
        "repeats": args.repeats,
        "isolated_recipients": args.isolated_recipients,
        "sample_seed_base": args.sample_seed_base,
        "retest_seed_base": args.retest_seed_base,
        **_candidate_design(args),
    }
    state.update(status="running", current_stage=None, updated_at=_timestamp())
    _write_json_atomic(state_path, state)

    smoke_dir = args.results_root / f"smoke_seed{args.sample_seed_base}"
    seed0_dir = args.results_root / f"seed{args.sample_seed_base}"
    retest_dir = args.results_root / f"seed{args.retest_seed_base}_retest"

    snapshot_done = _snapshot_complete(args.memory_dir)
    smoke_done = not args.skip_smoke and _run_complete(smoke_dir)
    seed0_done = _run_complete(seed0_dir)
    retest_done = _run_complete(retest_dir)
    if snapshot_done:
        _validate_completed_snapshot(args)
    if smoke_done:
        _validate_completed_run(
            args,
            smoke_dir,
            claims=args.smoke_claims,
            repeats=1,
            sample_seed_base=args.sample_seed_base,
        )
    if seed0_done:
        _validate_completed_run(
            args,
            seed0_dir,
            claims=args.claims,
            repeats=args.repeats,
            sample_seed_base=args.sample_seed_base,
        )
    if retest_done:
        _validate_completed_run(
            args,
            retest_dir,
            claims=args.claims,
            repeats=args.repeats,
            sample_seed_base=args.retest_seed_base,
        )

    stages: list[tuple[str, bool, list[str]]] = []
    snapshot_command = [
        sys.executable,
        "-m",
        "causal_diagnostic.fever_oracle_recipient.build_snapshot",
        "--data",
        str(args.data),
        "--support-per-label",
        str(args.support_per_label),
        "--evaluation-per-label",
        str(args.evaluation_per_label),
        "--split-seed",
        str(args.split_seed),
        "--build-seed",
        str(args.build_seed),
        "--endpoint",
        args.endpoint,
        "--model",
        args.model,
        "--graph-type",
        args.graph_type,
        "--node-num",
        str(args.node_num),
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
        "--output-memory-dir",
        str(args.memory_dir),
    ]
    if _needs_snapshot_resume(args.memory_dir) and not snapshot_done:
        snapshot_command.append("--resume")
    stages.append(("snapshot", snapshot_done, snapshot_command))

    if not args.skip_smoke:
        smoke_command = _base_runner_command(args, args.smoke_claims, smoke_dir)
        smoke_command.extend(("--repeats", "1", "--sample-seed-base", str(args.sample_seed_base)))
        if _needs_run_resume(smoke_dir) and not smoke_done:
            smoke_command.append("--resume")
        stages.append(("smoke", smoke_done, smoke_command))

    seed0_command = _base_runner_command(args, args.claims, seed0_dir)
    seed0_command.extend(("--repeats", str(args.repeats), "--sample-seed-base", str(args.sample_seed_base)))
    existing_seed = _existing_cache_seed(seed0_dir)
    smoke_cache = smoke_dir / "llm_cache.sqlite"
    cache_seed = existing_seed or (
        str(smoke_cache) if not args.skip_smoke and smoke_cache.is_file() else None
    )
    if cache_seed:
        seed0_command.extend(("--cache-seed-from", cache_seed))
    if _needs_run_resume(seed0_dir) and not seed0_done:
        seed0_command.append("--resume")
    stages.append(("seed0", seed0_done, seed0_command))

    retest_command = _base_runner_command(args, args.claims, retest_dir)
    retest_command.extend(
        (
            "--repeats",
            str(args.repeats),
            "--sample-seed-base",
            str(args.retest_seed_base),
            "--retest-results",
            str(seed0_dir),
        )
    )
    if _needs_run_resume(retest_dir) and not retest_done:
        retest_command.append("--resume")
    stages.append(("seed1000_retest", retest_done, retest_command))

    overall = tqdm(total=len(stages), desc="Complete FEVER experiment", unit="stage", dynamic_ncols=True)
    active_stage: str | None = None
    try:
        for stage, already_complete, command in stages:
            active_stage = stage
            log_path = logs_dir / f"{stage}.log"
            rendered = shlex.join(command)
            overall.set_postfix_str(stage, refresh=True)
            if already_complete:
                _mark_stage(
                    state_path,
                    state,
                    stage,
                    "skipped_completed",
                    command=rendered,
                    log=str(log_path),
                )
                overall.update(1)
                continue
            _mark_stage(
                state_path,
                state,
                stage,
                "running",
                command=rendered,
                log=str(log_path),
                started_at=_timestamp(),
                error=None,
            )
            _run_command(stage, command, log_path)
            _mark_stage(
                state_path,
                state,
                stage,
                "completed",
                command=rendered,
                log=str(log_path),
                completed_at=_timestamp(),
                error=None,
            )
            overall.update(1)
    except StageFailure as exc:
        _mark_stage(
            state_path,
            state,
            exc.stage,
            "failed",
            command=shlex.join(exc.command),
            log=str(exc.log_path),
            returncode=exc.returncode,
            error_tail=exc.tail,
        )
        state.update(status="failed", current_stage=exc.stage, updated_at=_timestamp())
        _write_json_atomic(state_path, state)
        overall.close()
        print(
            f"\nERROR: stage {exc.stage} failed (exit {exc.returncode}).\n"
            f"Command: {shlex.join(exc.command)}\n"
            f"Full log: {exc.log_path}\n"
            f"Progress state: {state_path}\n\n"
            f"Last output:\n{exc.tail}",
            file=sys.stderr,
        )
        raise SystemExit(exc.returncode) from exc
    except KeyboardInterrupt:
        if active_stage is not None:
            _mark_stage(
                state_path,
                state,
                active_stage,
                "interrupted",
                message="Interrupted by user or scheduler; rerun the same command to resume.",
            )
        state.update(status="interrupted", current_stage=active_stage, updated_at=_timestamp())
        _write_json_atomic(state_path, state)
        overall.close()
        print(
            "\nExperiment interrupted safely. Rerun the identical run_all command; "
            "completed work will be reused.",
            file=sys.stderr,
        )
        raise SystemExit(130)

    overall.close()
    state.update(status="completed", current_stage=None, updated_at=_timestamp())
    state["artifacts"] = {
        "snapshot": str(args.memory_dir / "causal_snapshot_manifest.json"),
        "seed0": str(seed0_dir),
        "retest": str(retest_dir),
        "report": str(retest_dir / "run_report.md"),
    }
    _write_json_atomic(state_path, state)
    print(
        json.dumps(
            {
                "status": "completed",
                "progress": str(state_path),
                **state["artifacts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return retest_dir / "run_report.md"


if __name__ == "__main__":
    main()
