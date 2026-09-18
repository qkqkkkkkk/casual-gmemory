#!/usr/bin/env python3
"""Run native GMemory + MacNet RQ2/RQ3 interventions on held-out PDDL tasks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _early_cli_value(flag: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


# Several upstream GMemory modules read these variables at import time. Make
# the dedicated runner usable with --endpoint and --help while still allowing
# explicit environment configuration to take precedence.
os.environ.setdefault(
    "OPENAI_API_BASE",
    _early_cli_value("--endpoint") or "http://127.0.0.1:11434/v1",
)
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_diagnostic.build_snapshot import parse_task_ids
from causal_diagnostic.memory import FrozenReadOnlyGMemory, FrozenRetrieval
from causal_diagnostic.run_intervention import (
    candidate_metadata,
    memory_config,
    prepare_task,
    seed_everything,
)
from causal_diagnostic.oracle_recipient.analysis import write_analysis
from causal_diagnostic.oracle_recipient.masked_macnet import (
    MemoryMaskPolicy,
    RecipientMaskedMacNet,
)
from causal_diagnostic.oracle_recipient.seeded_client import SeededCachedChat
from mas.memory.mas_memory.GMemory import GMemory
from mas.reasoning import ReasoningIO
from mas.utils import EmbeddingFunc
from tasks.envs import get_env
from tasks.prompts import get_dataset_system_prompt


RUNNER_SCHEMA = "gmemory-macnet-oracle-recipient-v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids", type=parse_task_ids, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--candidate-kind", choices=("trajectory", "insight"), default="trajectory"
    )
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--sample-seed-base", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-trials", type=int, default=30)
    parser.add_argument("--node-num", type=int, default=3)
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
    parser.add_argument("--cost-weight", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--retest-results", type=Path, default=None)
    parser.add_argument("--cache-seed-from", type=Path, default=None)
    parser.add_argument("--allow-unverified-memory-provenance", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.node_num < 2:
        raise SystemExit("--node-num must be at least 2")
    if args.candidate_index < 0:
        raise SystemExit("--candidate-index must be non-negative")
    if args.max_trials < 1:
        raise SystemExit("--max-trials must be at least 1")
    if args.temperature < 0:
        raise SystemExit("--temperature must be non-negative")
    if not 0.0 <= args.cost_weight < 1.0:
        raise SystemExit("--cost-weight must be in [0,1)")
    if args.bootstrap_samples < 100:
        raise SystemExit("--bootstrap-samples must be at least 100")
    if not 0.0 < args.confidence_level < 1.0:
        raise SystemExit("--confidence-level must be in (0,1)")
    if args.cache_seed_from is not None and not args.cache_seed_from.is_file():
        raise SystemExit(f"cache seed does not exist: {args.cache_seed_from}")
    if args.retest_results is not None:
        if args.retest_results.resolve() == args.output_dir.resolve():
            raise SystemExit("--retest-results and --output-dir must differ")


def _stable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        value
        for value in path.rglob("*")
        if value.is_file()
        and not value.name.endswith(("-wal", "-shm", ".lock"))
    )
    if not files:
        raise SystemExit(f"memory directory is empty: {path}")
    for file_path in files:
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _snapshot_provenance(
    memory_dir: Path,
    task_ids: Sequence[int],
    *,
    allow_unverified: bool,
) -> dict[str, Any]:
    manifest_path = memory_dir / "causal_snapshot_manifest.json"
    if not manifest_path.is_file():
        if not allow_unverified:
            raise SystemExit(
                f"{manifest_path} is missing; use a causal snapshot or explicitly "
                "pass --allow-unverified-memory-provenance"
            )
        return {"verified": False, "manifest": None, "support_task_ids": None}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    support_ids = {int(value) for value in manifest.get("task_ids", [])}
    overlap = support_ids & set(task_ids)
    if overlap:
        raise SystemExit(
            f"memory support/test leakage: task IDs appear in both sets: {sorted(overlap)}"
        )
    return {
        "verified": True,
        "manifest": str(manifest_path.resolve()),
        "support_task_ids": sorted(support_ids),
    }


def _prepare_output(
    args: argparse.Namespace,
    design: Mapping[str, Any],
    run: Mapping[str, Any],
) -> tuple[Path, dict[tuple[int, str, int, str], dict[str, Any]]]:
    output = args.output_dir
    manifest_path = output / "run_manifest.json"
    rows_path = output / "branches.jsonl"
    design_hash = _stable_hash(design)
    run_hash = _stable_hash(run)
    manifest = {
        "runner_schema": RUNNER_SCHEMA,
        "design_hash": design_hash,
        "run_hash": run_hash,
        "design": dict(design),
        "run": dict(run),
    }
    if args.resume:
        if not manifest_path.is_file() or not rows_path.is_file():
            raise SystemExit("--resume requires run_manifest.json and branches.jsonl")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("run_hash") != run_hash:
            raise SystemExit("resume arguments do not match the existing run")
    else:
        if manifest_path.exists() or rows_path.exists():
            raise SystemExit(
                f"refusing to overwrite {output}; use a new directory or --resume"
            )
        output.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows_path.touch(exist_ok=False)
    existing: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    for line_number, line in enumerate(
        rows_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (
            int(row["task_id"]),
            str(row["candidate"]["candidate_id"]),
            int(row["repeat_index"]),
            str(row["condition"]),
        )
        if key in existing:
            raise SystemExit(f"duplicate branch at line {line_number}: {key}")
        existing[key] = row
    return rows_path, existing


def _conditions(recipients: Sequence[str]) -> tuple[str, ...]:
    return ("use_all", "global_drop", *(f"drop_{name}" for name in recipients))


def _policy(condition: str, recipients: Sequence[str], args: argparse.Namespace) -> MemoryMaskPolicy:
    if condition == "use_all":
        dropped: frozenset[str] = frozenset()
        decision = False
    elif condition == "global_drop":
        dropped = frozenset(recipients)
        decision = True
    elif condition.startswith("drop_"):
        recipient = condition.removeprefix("drop_")
        if recipient not in recipients:
            raise ValueError(f"unknown recipient in condition: {condition}")
        dropped = frozenset((recipient,))
        decision = False
    else:
        raise ValueError(f"unknown condition: {condition}")
    return MemoryMaskPolicy(
        condition=condition,
        candidate_kind=args.candidate_kind,
        candidate_index=args.candidate_index,
        drop_recipients=dropped,
        drop_from_decision=decision,
    )


def _validate_exposure(
    trace: Sequence[Mapping[str, Any]],
    condition: str,
    recipients: Sequence[str],
) -> None:
    if not trace:
        raise RuntimeError("branch produced no MacNet steps")
    for step in trace:
        exposed = {
            name: bool(step["workers"][name]["candidate_exposed"])
            for name in recipients
        }
        decision_exposed = bool(step["decision"]["candidate_exposed"])
        if condition == "use_all":
            valid = all(exposed.values()) and decision_exposed
        elif condition == "global_drop":
            valid = not any(exposed.values()) and not decision_exposed
        else:
            dropped = condition.removeprefix("drop_")
            valid = (
                not exposed[dropped]
                and all(exposed[name] for name in recipients if name != dropped)
                and decision_exposed
            )
        if not valid:
            raise RuntimeError(
                f"candidate exposure invariant failed for {condition}: "
                f"workers={exposed}, decision={decision_exposed}"
            )


def _run_branch(
    args: argparse.Namespace,
    task_config: Mapping[str, Any],
    frozen: FrozenRetrieval,
    condition: str,
    recipients: Sequence[str],
    sample_seed: int,
    cache_path: Path,
) -> tuple[dict[str, Any], int, int]:
    seed_everything(sample_seed)
    branch_task = copy.deepcopy(dict(task_config))
    env = get_env("pddl", {}, args.max_trials)
    env.set_env(branch_task)
    client = SeededCachedChat(
        args.model,
        cache_path,
        experiment_seed=sample_seed,
        api_base=args.endpoint,
        api_key=args.api_key,
    )
    memory = FrozenReadOnlyGMemory(
        namespace=args.memory_dir.name,
        global_config=memory_config(args.memory_dir),
        llm_model=client,
        embedding_func=EmbeddingFunc(args.embedding_model),
        frozen_retrieval=frozen,
    )
    mas = RecipientMaskedMacNet()
    mas.build_system(
        ReasoningIO(llm_model=client),
        memory,
        env,
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
    actual_recipients = tuple(node.id for node in mas._agent_nodes.values())
    if tuple(recipients) != actual_recipients:
        raise RuntimeError(
            f"MacNet recipient mismatch: expected {recipients}, got {actual_recipients}"
        )
    mas.set_memory_policy(_policy(condition, recipients, args))
    mas.set_sampling_temperature(args.temperature)
    instruction = get_dataset_system_prompt("pddl", branch_task)
    for agent in mas.agents_team.values():
        agent.add_task_instruction(instruction)
    before = memory.memory_size
    calls_before, hits_before = client.calls, client.cache_hits
    reward, done = mas.schedule(branch_task)
    calls = client.calls - calls_before
    hits = client.cache_hits - hits_before
    after = memory.memory_size
    client.close()
    if after != before:
        raise RuntimeError("read-only GMemory changed persistent memory size")
    _validate_exposure(mas.execution_trace, condition, recipients)
    steps = int(env.infos.get("steps", len(mas.execution_trace)))
    outcome = {
        "success": bool(done),
        "reward": float(reward),
        "steps": steps,
        "team_score": float(reward)
        - args.cost_weight * steps / args.max_trials,
        "score_definition": "reward-cost_weight*steps/max_trials",
        "cost_weight": args.cost_weight,
        "actions": [
            step["decision"]["processed_action"] for step in mas.execution_trace
        ],
        "trace": mas.execution_trace,
        "memory_records_before": before,
        "memory_records_after": after,
    }
    return outcome, calls, hits


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    _validate_args(args)
    memory_dir = args.memory_dir.resolve()
    if not memory_dir.is_dir():
        raise SystemExit(f"memory directory does not exist: {memory_dir}")
    args.memory_dir = memory_dir
    provenance = _snapshot_provenance(
        memory_dir,
        args.task_ids,
        allow_unverified=args.allow_unverified_memory_provenance,
    )
    memory_hash = _directory_hash(memory_dir)
    recipients = tuple(f"solver_{index}" for index in range(args.node_num))
    design = {
        "runner_schema": RUNNER_SCHEMA,
        "host": "native_gmemory_macnet_pddl",
        "task_ids": args.task_ids,
        "memory_dir": str(memory_dir),
        "memory_sha256": memory_hash,
        "memory_provenance": provenance,
        "model": args.model,
        "candidate_kind": args.candidate_kind,
        "candidate_index": args.candidate_index,
        "repeats": args.repeats,
        "temperature": args.temperature,
        "max_trials": args.max_trials,
        "node_num": args.node_num,
        "recipients": recipients,
        "graph_type": args.graph_type,
        "successful_topk": args.successful_topk,
        "failed_topk": args.failed_topk,
        "insights_topk": args.insights_topk,
        "threshold": args.threshold,
        "embedding_model": args.embedding_model,
        "cost_weight": args.cost_weight,
        "selection_seed": args.selection_seed,
    }
    run = {
        **design,
        "endpoint": args.endpoint or os.environ.get("OPENAI_API_BASE"),
        "sample_seed_base": args.sample_seed_base,
        "cache_seed_from": (
            str(args.cache_seed_from.resolve()) if args.cache_seed_from else None
        ),
    }
    if args.retest_results is not None:
        previous_manifest_path = args.retest_results / "run_manifest.json"
        if not previous_manifest_path.is_file():
            raise SystemExit("--retest-results has no run_manifest.json")
        previous_manifest = json.loads(
            previous_manifest_path.read_text(encoding="utf-8")
        )
        if previous_manifest.get("design_hash") != _stable_hash(design):
            raise SystemExit("retest design_hash does not match current design")

    rows_path, existing = _prepare_output(args, design, run)
    cache_path = args.output_dir / "llm_cache.sqlite"
    if args.cache_seed_from is not None and not cache_path.exists():
        shutil.copy2(args.cache_seed_from, cache_path)
    design_hash, run_hash = _stable_hash(design), _stable_hash(run)
    exclusions: list[dict[str, Any]] = []
    total_calls = total_hits = 0

    for task_id in args.task_ids:
        task_config, _ = prepare_task(task_id, args.max_trials)
        retrieval_client = SeededCachedChat(
            args.model,
            cache_path,
            experiment_seed=args.selection_seed,
            api_base=args.endpoint,
            api_key=args.api_key,
        )
        live_memory = GMemory(
            namespace=args.memory_dir.name,
            global_config=memory_config(args.memory_dir),
            llm_model=retrieval_client,
            embedding_func=EmbeddingFunc(args.embedding_model),
        )
        try:
            frozen = FrozenRetrieval.from_result(
                live_memory.retrieve_memory(
                    query_task=task_config["task_main"],
                    successful_topk=args.successful_topk,
                    failed_topk=args.failed_topk,
                    insight_topk=args.insights_topk,
                    threshold=args.threshold,
                )
            )
            candidate = candidate_metadata(
                frozen, args.candidate_kind, args.candidate_index
            )
        except IndexError as exc:
            exclusions.append({"task_id": task_id, "reason": str(exc)})
            retrieval_client.close()
            continue
        total_calls += retrieval_client.calls
        total_hits += retrieval_client.cache_hits
        retrieval_client.close()

        for repeat_index in range(args.repeats):
            sample_seed = args.sample_seed_base + repeat_index
            for condition in _conditions(recipients):
                key = (
                    task_id,
                    str(candidate["candidate_id"]),
                    repeat_index,
                    condition,
                )
                if key in existing:
                    continue
                outcome, calls, hits = _run_branch(
                    args,
                    task_config,
                    frozen,
                    condition,
                    recipients,
                    sample_seed,
                    cache_path,
                )
                total_calls += calls
                total_hits += hits
                row = {
                    "runner_schema": RUNNER_SCHEMA,
                    "design_hash": design_hash,
                    "run_hash": run_hash,
                    "task_id": task_id,
                    "task": {
                        key: task_config[key]
                        for key in ("game_name", "problem_index", "task_main")
                    },
                    "candidate": candidate,
                    "repeat_index": repeat_index,
                    "sample_seed": sample_seed,
                    "condition": condition,
                    "outcome": outcome,
                    "llm_calls": calls,
                    "cache_hits": hits,
                }
                with rows_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                existing[key] = row

    diagnostics = {
        "selected_tasks": len(args.task_ids),
        "excluded_tasks": exclusions,
        "completed_branches": len(existing),
        "conditions_per_event": len(_conditions(recipients)),
        "llm_calls_this_process": total_calls,
        "cache_hits_this_process": total_hits,
    }
    (args.output_dir / "collection_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not existing:
        raise SystemExit("no complete branches were collected")
    analysis_path = write_analysis(
        args.output_dir,
        retest_results=args.retest_results,
        delta=args.delta,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.selection_seed,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "analysis": str(analysis_path),
                **diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return analysis_path


if __name__ == "__main__":
    main()
