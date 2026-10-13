#!/usr/bin/env python3
"""Evaluate native GMemory or a learned exposure gate on offline HotpotQA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback
from typing import Any, Mapping, Sequence

from tqdm import tqdm

from causal_diagnostic.hotpotqa_oracle_recipient.data import (
    file_md5,
    load_hotpotqa,
    resolve_registered_examples,
)
from causal_diagnostic.hotpotqa_oracle_recipient.offline_env import OfflineHotpotQAEnv
from causal_diagnostic.hotpotqa_oracle_recipient.prompts import (
    HOTPOTQA_SYSTEM_PROMPT,
    prepare_example,
)
from causal_diagnostic.hotpotqa_oracle_recipient.provenance import (
    SnapshotProvenanceError,
    reconcile_retest_design,
    validate_snapshot_manifest,
)
from causal_diagnostic.hotpotqa_oracle_recipient.sampling import (
    configure_hotpotqa_sampling,
)
from causal_memory_control.runtime_gate import CHECKPOINT_SCHEMA, GMemoryExposureGate
from causal_memory_control.jsonl import read_jsonl


RUNNER_SCHEMA = "gmemory-team-exposure-hotpotqa-evaluation-v1"
PHYSICAL_MEMORY_CONTEXT_FIELDS = frozenset({"memory_sha256"})

SeededCachedChat = None
GMemory = None
ReasoningIO = None
EmbeddingFunc = None
MacNet = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _load_runtime_components() -> None:
    global SeededCachedChat, GMemory, ReasoningIO, EmbeddingFunc, MacNet
    if SeededCachedChat is None:
        from causal_diagnostic.oracle_recipient.seeded_client import SeededCachedChat as chat

        SeededCachedChat = chat
    if GMemory is None:
        from mas.memory.mas_memory.GMemory import GMemory as memory

        GMemory = memory
    if ReasoningIO is None:
        from mas.reasoning import ReasoningIO as reasoning

        ReasoningIO = reasoning
    if EmbeddingFunc is None:
        from mas.utils import EmbeddingFunc as embedding

        EmbeddingFunc = embedding
    if MacNet is None:
        from tasks.mas_workflow.macnet.graph_mas import MacNet as workflow

        MacNet = workflow


def memory_config(memory_dir: Path) -> dict[str, Any]:
    return {"working_dir": str(memory_dir.resolve().parent), "hop": 1, "read_only": True}


def _candidate_kinds(value: str) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not result or set(result) - {"trajectory", "insight"}:
        raise argparse.ArgumentTypeError("candidate kinds must be trajectory, insight, or both")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", "--test", dest="data", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--claims", type=int, default=100)
    parser.add_argument("--evaluation-offset", type=int, default=0)
    parser.add_argument(
        "--mode", choices=("always_keep", "always_drop", "learned", "jev_direct", "jev_decomposed"), default="always_keep"
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--node-num", type=int, default=3)
    parser.add_argument(
        "--graph-type", choices=("Chain", "FullConnected", "Debate", "Star"), default="Chain"
    )
    parser.add_argument("--successful-topk", type=int, default=3)
    parser.add_argument("--failed-topk", type=int, default=0)
    parser.add_argument("--insights-topk", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--candidate-kinds", type=_candidate_kinds, default=("trajectory",))
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--max-drops", type=int, default=1)
    parser.add_argument("--cache-seed-from", type=Path, default=None)
    parser.add_argument("--semantic-cache", type=Path, default=None)
    parser.add_argument("--jev-cache", type=Path, default=None)
    parser.add_argument("--retry-jev-errors",action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.claims < 1 or args.evaluation_offset < 0:
        raise SystemExit("--claims must be positive and --evaluation-offset non-negative")
    if args.node_num < 2:
        raise SystemExit("--node-num must be at least 2")
    if min(args.temperature, args.delta, args.kappa) < 0:
        raise SystemExit("temperature, delta, and kappa must be non-negative")
    if min(args.successful_topk, args.failed_topk, args.insights_topk) < 0:
        raise SystemExit("retrieval top-k values must be non-negative")
    if args.max_drops < -1:
        raise SystemExit("--max-drops must be non-negative or -1")
    if args.mode in ("learned", "jev_direct", "jev_decomposed"):
        if args.checkpoint is None or not args.checkpoint.is_file():
            raise SystemExit("this gate mode requires an existing --checkpoint")
    elif args.checkpoint is not None:
        raise SystemExit("--checkpoint is only valid with --mode learned")
    if args.cache_seed_from is not None and not args.cache_seed_from.is_file():
        raise SystemExit(f"cache seed does not exist: {args.cache_seed_from}")
    if args.retry_jev_errors and (not args.resume or not args.mode.startswith("jev_")):
        raise SystemExit("--retry-jev-errors requires --resume and a Jev mode")


def _stable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        value
        for value in path.rglob("*")
        if value.is_file() and not value.name.endswith(("-wal", "-shm", ".lock"))
    )
    if not files:
        raise ValueError(f"memory directory is empty: {path}")
    for file_path in files:
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _load_snapshot(args: argparse.Namespace, source_md5: str) -> dict[str, Any]:
    path = args.memory_dir / "causal_snapshot_manifest.json"
    if not path.is_file():
        raise SystemExit(f"snapshot manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validation_kind = "trajectory" if args.successful_topk > 0 else "insight"
    try:
        return validate_snapshot_manifest(
            manifest,
            source_md5=source_md5,
            model=args.model,
            graph_type=args.graph_type,
            node_num=args.node_num,
            embedding_model=args.embedding_model,
            claims=args.evaluation_offset + args.claims,
            candidate_kind=validation_kind,
        )
    except SnapshotProvenanceError as exc:
        raise SystemExit(str(exc)) from exc


def validate_checkpoint_for_hotpotqa(
    payload: Mapping[str, Any],
    *,
    evaluation_ids: Sequence[int],
    expected_context: Mapping[str, Any],
    candidate_kinds: Sequence[str],
) -> None:
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported GMemory exposure-gate checkpoint")
    metadata = payload.get("training_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint has no training metadata")
    raw_training_ids = metadata.get("training_task_ids")
    if not isinstance(raw_training_ids, list) or not raw_training_ids:
        raise ValueError("HotpotQA evaluation requires recorded training_task_ids")
    overlap = {int(value) for value in raw_training_ids} & {int(value) for value in evaluation_ids}
    if overlap:
        raise ValueError(f"checkpoint training/evaluation leakage: {sorted(overlap)[:10]}")
    trained_kinds = set(metadata.get("candidate_kinds", ()))
    if not trained_kinds.intersection(candidate_kinds):
        raise ValueError("checkpoint has no trained candidate kind requested by this run")
    contexts = metadata.get("training_contexts", ())
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("HotpotQA evaluation requires recorded training_contexts")
    for index, context in enumerate(contexts):
        if not isinstance(context, Mapping):
            raise ValueError(f"checkpoint training context {index} is invalid")
        required = set(expected_context) - PHYSICAL_MEMORY_CONTEXT_FIELDS
        missing = required - set(context)
        if missing:
            raise ValueError(f"checkpoint training context {index} lacks {sorted(missing)}")
        for key, expected in expected_context.items():
            if key not in PHYSICAL_MEMORY_CONTEXT_FIELDS and context.get(key) != expected:
                raise ValueError(
                    f"checkpoint training context {key}={context.get(key)!r}, "
                    f"but HotpotQA evaluation requires {expected!r}"
                )


def _prepare_output(
    args: argparse.Namespace, design: Mapping[str, Any], run: Mapping[str, Any]
) -> None:
    manifest_path = args.output_dir / "run_manifest.json"
    manifest = {
        "runner_schema": RUNNER_SCHEMA,
        "design_hash": _stable_hash(design),
        "run_hash": _stable_hash(run),
        "design": dict(design),
        "run": dict(run),
    }
    if args.resume:
        if not manifest_path.is_file():
            raise SystemExit("--resume requires an existing run_manifest.json")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("run_hash") != manifest["run_hash"]:
            raise SystemExit("resume arguments do not match the existing HotpotQA run")
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite {args.output_dir}; use --resume or a new path")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, manifest)


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    denominator = max(count, 1)
    em_values = [float(row.get("outcome", {}).get("answer_em", row.get("done", False))) for row in rows]
    return {
        "completed_questions": count,
        "mode": rows[0].get("mode") if rows else None,
        "mean_answer_f1": sum(float(row["reward"]) for row in rows) / denominator,
        "answer_exact_match_rate": sum(em_values) / denominator,
        "mean_keep_rate": sum(float(row.get("keep_rate", 1.0)) for row in rows) / denominator,
        "retrieved_candidates": sum(int(row.get("retrieved_count", 0)) for row in rows),
        "dropped_candidates": sum(int(row.get("dropped_count", 0)) for row in rows),
        "llm_calls": sum(int(row.get("outcome", {}).get("llm_calls", 0)) for row in rows),
        "cache_hits": sum(int(row.get("outcome", {}).get("cache_hits", 0)) for row in rows),
        **{key:sum(float(row.get("outcome",{}).get(key,0)) for row in rows) for key in (
            "jev_calls","jev_cache_hits","jev_latency_seconds","jev_input_tokens","jev_output_tokens","jev_reported_cost","jev_errors"
        )},
    }


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    _validate_args(args)
    args.data = args.data.resolve()
    args.memory_dir = args.memory_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.memory_dir.is_dir():
        raise SystemExit(f"memory directory does not exist: {args.memory_dir}")

    examples = load_hotpotqa(args.data)
    source_md5 = file_md5(args.data)
    snapshot = _load_snapshot(args, source_md5)
    registered = resolve_registered_examples(examples, snapshot["evaluation_ids"])
    selected = registered[args.evaluation_offset : args.evaluation_offset + args.claims]
    if len(selected) != args.claims:
        raise SystemExit("requested HotpotQA evaluation slice is incomplete")
    selected_ids = [int(row["id"]) for row in selected]
    if set(selected_ids) & {int(value) for value in snapshot["support_ids"]}:
        raise SystemExit("selected evaluation questions overlap snapshot support questions")

    effective_endpoint = args.endpoint or os.environ.get("OPENAI_API_BASE") or "http://127.0.0.1:11434/v1"
    memory_hash = _directory_hash(args.memory_dir)
    design = {
        "runner_schema": RUNNER_SCHEMA,
        "benchmark": "HotpotQA_distractor_offline",
        "source_md5": source_md5,
        "snapshot_design_hash": snapshot["design_hash"],
        "memory_sha256": memory_hash,
        "evaluation_ids": selected_ids,
        "evaluation_offset": args.evaluation_offset,
        "model": args.model,
        "endpoint": effective_endpoint,
        "seed": args.seed,
        "temperature": args.temperature,
        "graph_type": args.graph_type,
        "node_num": args.node_num,
        "embedding_model": args.embedding_model,
        "successful_topk": args.successful_topk,
        "failed_topk": args.failed_topk,
        "insights_topk": args.insights_topk,
        "threshold": args.threshold,
        "candidate_kinds": list(args.candidate_kinds),
    }
    if args.resume and (args.output_dir / "run_manifest.json").is_file():
        previous = json.loads((args.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
        try:
            design = reconcile_retest_design(design, previous)
        except SnapshotProvenanceError as exc:
            raise SystemExit(f"resume design is incompatible: {exc}") from exc

    checkpoint_hash = None
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
        checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        expected_context = {
            key: design[key]
            for key in (
                "benchmark", "source_md5", "snapshot_design_hash", "memory_sha256",
                "model", "graph_type", "node_num", "embedding_model",
                "successful_topk", "failed_topk", "insights_topk", "threshold",
            )
        }
        try:
            validate_checkpoint_for_hotpotqa(
                checkpoint,
                evaluation_ids=selected_ids,
                expected_context=expected_context,
                candidate_kinds=args.candidate_kinds,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        checkpoint_hash = _file_sha256(args.checkpoint)

    cache_seed = args.cache_seed_from.resolve() if args.cache_seed_from else None
    run = {
        **design,
        "mode": args.mode,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_sha256": checkpoint_hash,
        "delta": args.delta,
        "kappa": args.kappa,
        "max_drops": args.max_drops,
        "cache_seed_from": str(cache_seed) if cache_seed else None,
        "cache_seed_sha256": _file_sha256(cache_seed) if cache_seed else None,
    }
    _prepare_output(args, design, run)

    log_path = args.output_dir / "memory_gate.jsonl"
    if args.resume:
        read_jsonl(log_path, repair_tail=True)
    if args.retry_jev_errors:
        from .jsonl import retry_jev_error_tasks

        retry_jev_error_tasks(log_path)
    cache_path = args.output_dir / "llm_cache.sqlite"
    if cache_seed is not None and not cache_path.exists():
        shutil.copy2(cache_seed, cache_path)
    progress_path = args.output_dir / "progress.json"
    _write_json(
        progress_path,
        {
            "runner_schema": RUNNER_SCHEMA,
            "status": "running",
            "planned_questions": len(selected),
            "completed_questions": len(_load_rows(log_path)),
            "current_question_id": None,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "error": None,
        },
    )
    gate_kwargs = {
        "mode": args.mode,
        "delta": args.delta,
        "kappa": args.kappa,
        "candidate_kinds": args.candidate_kinds,
        "max_drops": None if args.max_drops == -1 else args.max_drops,
        "task_metadata": design,
        "memory_dir": args.memory_dir,
        "log_path": log_path,
        "resume": args.resume,
    }
    gate = (
        GMemoryExposureGate.from_checkpoint(args.checkpoint, **gate_kwargs,
            semantic_cache_path=args.semantic_cache or args.output_dir / "semantic_cache.sqlite",
            jev_cache_path=args.jev_cache or args.output_dir / "jev_cache.sqlite")
        if args.mode in ("learned", "jev_direct", "jev_decomposed")
        else GMemoryExposureGate(**gate_kwargs)
    )
    unknown = gate.completed_task_ids - {str(value) for value in selected_ids}
    if unknown:
        raise SystemExit(f"gate log contains IDs outside this evaluation slice: {sorted(unknown)[:10]}")

    os.environ["OPENAI_API_BASE"] = effective_endpoint
    os.environ["OPENAI_API_KEY"] = args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    client = None
    current_id: int | None = None
    try:
        _load_runtime_components()
        seed_everything(args.seed)
        client = SeededCachedChat(
            args.model, cache_path, experiment_seed=args.seed,
            api_base=args.endpoint, api_key=args.api_key,
        )
        memory = GMemory(
            namespace=args.memory_dir.name,
            global_config=memory_config(args.memory_dir),
            llm_model=client,
            embedding_func=EmbeddingFunc(args.embedding_model),
        )
        initial_memory_size = memory.memory_size
        if initial_memory_size != int(snapshot["memory_records"]):
            raise RuntimeError("frozen GMemory size differs from the snapshot manifest")
        env = OfflineHotpotQAEnv(max_trials=1)
        mas = MacNet()
        mas.build_system(
            ReasoningIO(llm_model=client), memory, env,
            {
                "graph_type": args.graph_type,
                "node_num": args.node_num,
                "use_critic": False,
                "successful_topk": args.successful_topk,
                "failed_topk": args.failed_topk,
                "insights_topk": args.insights_topk,
                "threshold": args.threshold,
                "use_projector": False,
            },
        )
        configure_hotpotqa_sampling(mas, args.temperature)
        for agent in mas.agents_team.values():
            agent.add_task_instruction(HOTPOTQA_SYSTEM_PROMPT)
        mas._decision_node._agent.add_task_instruction(HOTPOTQA_SYSTEM_PROMPT)
        exposure_ids = [node.id for node in mas._agent_nodes.values()] + [mas._decision_node.id]
        gate.set_exposure_agent_ids(exposure_ids)
        memory.set_retrieval_gate(gate)

        pending = [row for row in selected if not gate.task_is_complete(int(row["id"]))]
        progress = tqdm(
            pending,
            total=len(selected),
            initial=len(selected) - len(pending),
            desc=f"HotpotQA gate {args.mode}",
            unit="question",
            dynamic_ncols=True,
        )
        for raw in progress:
            current_id = int(raw["id"])
            task = prepare_example(raw)
            env.set_env(task)
            seed_everything((args.seed + current_id) % (2**32))
            gate.begin_task(current_id, task)
            before = memory.memory_size
            calls_before, hits_before = client.calls, client.cache_hits
            _write_json(
                progress_path,
                {
                    "runner_schema": RUNNER_SCHEMA,
                    "status": "running",
                    "planned_questions": len(selected),
                    "completed_questions": len(gate.completed_task_ids),
                    "current_question_id": current_id,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "error": None,
                },
            )
            reward, schedule_done = mas.schedule(task)
            after = memory.memory_size
            if before != initial_memory_size or after != initial_memory_size:
                raise RuntimeError("read-only GMemory changed persistent memory size")
            metrics = dict(env.last_metrics or {})
            exact_match = bool(metrics.get("em", 0.0))
            gate.end_task(
                reward,
                exact_match,
                outcome_metadata={
                    "prediction": env.last_prediction,
                    "gold_answer": task["answer"],
                    "answer_f1": float(metrics.get("f1", reward)),
                    "answer_em": float(metrics.get("em", 0.0)),
                    "answer_metrics": metrics,
                    "episode_done": bool(env.done),
                    "schedule_done": bool(schedule_done),
                    "steps": int(env.infos["steps"]),
                    "llm_calls": client.calls - calls_before,
                    "cache_hits": client.cache_hits - hits_before,
                    "memory_records_before": before,
                    "memory_records_after": after,
                },
            )
            progress.set_postfix_str(
                f"id={current_id} F1={float(reward):.3f} EM={int(exact_match)}",
                refresh=True,
            )
            current_id = None
        progress.close()
    except BaseException as exc:
        _write_json(
            progress_path,
            {
                "runner_schema": RUNNER_SCHEMA,
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "planned_questions": len(selected),
                "completed_questions": len(gate.completed_task_ids),
                "current_question_id": current_id,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            },
        )
        raise
    finally:
        if client is not None:
            client.close()
        gate.close()

    rows = _load_rows(log_path)
    completed = {str(row["task_id"]) for row in rows}
    expected = {str(value) for value in selected_ids}
    if completed != expected:
        raise RuntimeError(
            f"HotpotQA gate run has wrong task set: missing={sorted(expected-completed)}, "
            f"unexpected={sorted(completed-expected)}"
        )
    result = {"runner_schema": RUNNER_SCHEMA, "mode": args.mode, "log": str(log_path), **_summary(rows)}
    _write_json(args.output_dir / "summary.json", result)
    _write_json(
        progress_path,
        {
            "runner_schema": RUNNER_SCHEMA,
            "status": "completed",
            "planned_questions": len(selected),
            "completed_questions": len(rows),
            "current_question_id": None,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "error": None,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return log_path


if __name__ == "__main__":
    main()
