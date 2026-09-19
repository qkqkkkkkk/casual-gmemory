#!/usr/bin/env python3
"""Run native GMemory + MacNet RQ2/RQ3/RQ4 interventions on held-out FEVER."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

from tqdm import tqdm


def _early_cli_value(flag: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault(
    "OPENAI_API_BASE",
    _early_cli_value("--endpoint") or "http://127.0.0.1:11434/v1",
)
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_diagnostic.fever_oracle_recipient.data import (
    file_md5,
    load_binary_fever,
    resolve_registered_examples,
)
from causal_diagnostic.fever_oracle_recipient.offline_env import OfflineFeverEnv
from causal_diagnostic.fever_oracle_recipient.local_metric import score_evidence_pages
from causal_diagnostic.fever_oracle_recipient.provenance import (
    SnapshotProvenanceError,
    validate_snapshot_manifest,
)
from causal_diagnostic.fever_oracle_recipient.prompts import (
    FEVER_SYSTEM_PROMPT,
    prepare_example,
)
from causal_diagnostic.fever_oracle_recipient.utils import (
    candidate_metadata,
    memory_config,
    seed_everything,
)
from causal_diagnostic.memory import FrozenReadOnlyGMemory, FrozenRetrieval
from causal_diagnostic.oracle_recipient.analysis import write_analysis
from causal_diagnostic.oracle_recipient.masked_macnet import (
    MemoryMaskPolicy,
    RecipientMaskedMacNet,
)
from causal_diagnostic.oracle_recipient.seeded_client import SeededCachedChat
from mas.memory.mas_memory.GMemory import GMemory
from mas.reasoning import ReasoningIO
from mas.utils import EmbeddingFunc


RUNNER_SCHEMA = "native-gmemory-macnet-fever-rq234-v2"
_ACTIVE_PROGRESS_PATH: Path | None = None
_ACTIVE_PROGRESS: dict[str, Any] = {}


def _update_progress(**fields: Any) -> None:
    """Atomically persist collection state after every resolved branch."""
    global _ACTIVE_PROGRESS
    if _ACTIVE_PROGRESS_PATH is None:
        return
    _ACTIVE_PROGRESS = {
        **_ACTIVE_PROGRESS,
        **fields,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    temporary = _ACTIVE_PROGRESS_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(_ACTIVE_PROGRESS, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(_ACTIVE_PROGRESS_PATH)


def _record_uncaught_error(exc: BaseException) -> None:
    if _ACTIVE_PROGRESS_PATH is None:
        return
    _update_progress(
        status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
        error={
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--claims", type=int, default=100)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument(
        "--candidate-kind", choices=("trajectory", "insight"), default="trajectory"
    )
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--sample-seed-base", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--node-num", type=int, default=3)
    parser.add_argument(
        "--isolated-recipients",
        default="solver_0,solver_2",
        help="Comma-separated MacNet workers used by the isolated-exposure probe",
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
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--retest-results", type=Path, default=None)
    parser.add_argument("--cache-seed-from", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.claims < 1:
        raise SystemExit("--claims must be at least 1")
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.candidate_index < 0:
        raise SystemExit("--candidate-index must be non-negative")
    if args.node_num < 2:
        raise SystemExit("--node-num must be at least 2")
    if args.temperature < 0:
        raise SystemExit("--temperature must be non-negative")
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


def _parse_recipient_names(value: str) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not names:
        raise SystemExit("--isolated-recipients must contain at least one worker")
    return names


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


def _load_snapshot(
    args: argparse.Namespace, source_md5: str
) -> dict[str, Any]:
    manifest_path = args.memory_dir / "causal_snapshot_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"snapshot manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        return validate_snapshot_manifest(
            manifest,
            source_md5=source_md5,
            model=args.model,
            graph_type=args.graph_type,
            node_num=args.node_num,
            embedding_model=args.embedding_model,
            claims=args.claims,
            candidate_kind=args.candidate_kind,
        )
    except SnapshotProvenanceError as exc:
        raise SystemExit(str(exc)) from exc


def _conditions(recipients: Sequence[str]) -> tuple[str, ...]:
    return ("use_all", "global_drop", *(f"only_{name}" for name in recipients))


def _policy(
    condition: str, recipients: Sequence[str], args: argparse.Namespace
) -> MemoryMaskPolicy:
    if condition == "use_all":
        dropped: frozenset[str] = frozenset()
        drop_decision = False
    elif condition == "global_drop":
        dropped = frozenset(recipients)
        drop_decision = True
    elif condition.startswith("only_"):
        recipient = condition.removeprefix("only_")
        if recipient not in recipients:
            raise ValueError(f"unknown recipient in condition: {condition}")
        dropped = frozenset(name for name in recipients if name != recipient)
        drop_decision = True
    else:
        raise ValueError(f"unknown condition: {condition}")
    return MemoryMaskPolicy(
        condition=condition,
        candidate_kind=args.candidate_kind,
        candidate_index=args.candidate_index,
        drop_recipients=dropped,
        drop_from_decision=drop_decision,
    )


def _validate_exposure(
    trace: Sequence[Mapping[str, Any]],
    condition: str,
    recipients: Sequence[str],
) -> None:
    if not trace:
        raise RuntimeError("branch produced no MacNet steps")
    for step in trace:
        workers = {
            name: bool(step["workers"][name]["candidate_exposed"])
            for name in recipients
        }
        decision = bool(step["decision"]["candidate_exposed"])
        if condition == "use_all":
            valid = all(workers.values()) and decision
        elif condition == "global_drop":
            valid = not any(workers.values()) and not decision
        else:
            recipient = condition.removeprefix("only_")
            valid = (
                workers[recipient]
                and all(not workers[name] for name in recipients if name != recipient)
                and not decision
            )
        if not valid:
            raise RuntimeError(
                f"candidate exposure invariant failed for {condition}: "
                f"workers={workers}, decision={decision}"
            )


def _prepare_output(
    args: argparse.Namespace,
    design: Mapping[str, Any],
    run: Mapping[str, Any],
) -> tuple[Path, dict[tuple[int, str, int, str], dict[str, Any]]]:
    output = args.output_dir
    manifest_path = output / "run_manifest.json"
    rows_path = output / "branches.jsonl"
    manifest = {
        "runner_schema": RUNNER_SCHEMA,
        "design_hash": _stable_hash(design),
        "run_hash": _stable_hash(run),
        "design": dict(design),
        "run": dict(run),
    }
    if args.resume:
        if not manifest_path.is_file() or not rows_path.is_file():
            raise SystemExit("--resume requires run_manifest.json and branches.jsonl")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("run_hash") != manifest["run_hash"]:
            raise SystemExit("resume arguments do not match the existing run")
    else:
        if output.exists():
            raise SystemExit(
                f"refusing to overwrite {output}; use a new directory or --resume"
            )
        output.mkdir(parents=True)
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


def _run_branch(
    args: argparse.Namespace,
    task: Mapping[str, Any],
    frozen: FrozenRetrieval,
    condition: str,
    recipients: Sequence[str],
    sample_seed: int,
    cache_path: Path,
) -> tuple[dict[str, Any], int, int]:
    seed_everything(sample_seed)
    branch_task = copy.deepcopy(dict(task))
    env = OfflineFeverEnv(max_trials=1)
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
    if actual_recipients != tuple(recipients):
        client.close()
        raise RuntimeError(
            f"MacNet recipient mismatch: expected {recipients}, got {actual_recipients}"
        )
    mas.set_memory_policy(_policy(condition, recipients, args))
    mas.set_sampling_temperature(args.temperature)
    for agent in mas.agents_team.values():
        agent.add_task_instruction(FEVER_SYSTEM_PROMPT)
    mas._decision_node._agent.add_task_instruction(FEVER_SYSTEM_PROMPT)
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
    local_metrics = {
        name: score_evidence_pages(
            mas.execution_trace[-1]["workers"][name]["raw_output"],
            branch_task.get("gold_evidence_page_sets", []),
        )
        for name in recipients
    }
    outcome = {
        "success": bool(done),
        "reward": float(reward),
        "steps": int(env.infos["steps"]),
        "team_score": float(reward),
        "score_definition": "exact_fever_binary_label_accuracy",
        "prediction": env.last_prediction,
        "gold_label": branch_task["label"],
        "local_metrics": local_metrics,
        "actions": [
            step["decision"]["processed_action"] for step in mas.execution_trace
        ],
        "trace": mas.execution_trace,
        "memory_records_before": before,
        "memory_records_after": after,
    }
    return outcome, calls, hits


def main(argv: Sequence[str] | None = None) -> Path:
    global _ACTIVE_PROGRESS_PATH, _ACTIVE_PROGRESS
    args = parse_args(argv)
    _validate_args(args)
    args.test = args.test.resolve()
    args.memory_dir = args.memory_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.memory_dir.is_dir():
        raise SystemExit(f"memory directory does not exist: {args.memory_dir}")
    examples = load_binary_fever(args.test)
    source_md5 = file_md5(args.test)
    snapshot = _load_snapshot(args, source_md5)
    evaluation = resolve_registered_examples(examples, snapshot["evaluation_ids"])
    selected = evaluation[: args.claims]
    selected_ids = [int(row["id"]) for row in selected]
    support_ids = {int(value) for value in snapshot["support_ids"]}
    if support_ids & set(selected_ids):
        raise SystemExit("selected evaluation claims overlap snapshot support claims")

    memory_hash = _directory_hash(args.memory_dir)
    all_recipients = tuple(f"solver_{index}" for index in range(args.node_num))
    isolated_recipients = _parse_recipient_names(args.isolated_recipients)
    unknown_recipients = set(isolated_recipients) - set(all_recipients)
    if unknown_recipients:
        raise SystemExit(
            "--isolated-recipients contains workers outside this MacNet: "
            f"{sorted(unknown_recipients)}"
        )
    design = {
        "runner_schema": RUNNER_SCHEMA,
        "benchmark": "FEVER_binary_offline",
        "source_md5": source_md5,
        "snapshot_design_hash": snapshot["design_hash"],
        "memory_dir": str(args.memory_dir),
        "memory_sha256": memory_hash,
        "evaluation_ids": selected_ids,
        "model": args.model,
        "candidate_kind": args.candidate_kind,
        "candidate_index": args.candidate_index,
        "repeats": args.repeats,
        "temperature": args.temperature,
        "node_num": args.node_num,
        "recipient_intervention": "isolated_exposure",
        "recipients": list(isolated_recipients),
        "all_recipients": list(all_recipients),
        "graph_type": args.graph_type,
        "successful_topk": args.successful_topk,
        "failed_topk": args.failed_topk,
        "insights_topk": args.insights_topk,
        "threshold": args.threshold,
        "embedding_model": args.embedding_model,
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
        args.retest_results = args.retest_results.resolve()
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
    total_calls = total_hits = 0
    exclusions: list[dict[str, Any]] = []
    conditions = _conditions(isolated_recipients)
    branches_per_claim = args.repeats * len(conditions)
    planned_branches = len(selected) * branches_per_claim
    resolved_branches = len(existing)
    _ACTIVE_PROGRESS_PATH = args.output_dir / "collection_progress.json"
    _ACTIVE_PROGRESS = {}
    _update_progress(
        runner_schema=RUNNER_SCHEMA,
        status="running",
        registered_claims=len(selected),
        repeats=args.repeats,
        conditions=list(conditions),
        planned_branches=planned_branches,
        resolved_branches=resolved_branches,
        persisted_branches=len(existing),
        excluded_claims=0,
        current=None,
        llm_calls_this_process=0,
        cache_hits_this_process=0,
        error=None,
    )
    progress = tqdm(
        total=planned_branches,
        initial=min(resolved_branches, planned_branches),
        desc=f"FEVER seed {args.sample_seed_base}",
        unit="branch",
        dynamic_ncols=True,
        mininterval=1.0,
    )

    for raw_example in selected:
        task = prepare_example(raw_example)
        claim_id = int(task["id"])
        completed_for_claim = {
            (key[2], key[3]) for key in existing if key[0] == claim_id
        }
        expected_for_claim = {
            (repeat_index, condition)
            for repeat_index in range(args.repeats)
            for condition in conditions
        }
        candidate_ids_for_claim = {
            key[1] for key in existing if key[0] == claim_id
        }
        if len(candidate_ids_for_claim) > 1:
            raise RuntimeError(
                f"claim {claim_id} has multiple persisted candidate ids: "
                f"{sorted(candidate_ids_for_claim)}"
            )
        if completed_for_claim == expected_for_claim:
            progress.set_postfix_str(
                f"claim={claim_id} already complete", refresh=True
            )
            continue
        progress.set_postfix_str(f"claim={claim_id} retrieval", refresh=True)
        _update_progress(
            current={"claim_id": claim_id, "phase": "retrieval"},
        )
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
        if live_memory.memory_size != int(snapshot["memory_records"]):
            retrieval_client.close()
            raise RuntimeError("frozen GMemory size differs from snapshot manifest")
        try:
            frozen = FrozenRetrieval.from_result(
                live_memory.retrieve_memory(
                    query_task=task["task_main"],
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
            exclusions.append({"claim_id": claim_id, "reason": str(exc)})
            retrieval_client.close()
            resolved_branches += branches_per_claim
            progress.update(branches_per_claim)
            progress.set_postfix_str(
                f"claim={claim_id} excluded: {exc}", refresh=True
            )
            _update_progress(
                resolved_branches=resolved_branches,
                persisted_branches=len(existing),
                excluded_claims=len(exclusions),
                current={
                    "claim_id": claim_id,
                    "phase": "excluded",
                    "reason": str(exc),
                },
            )
            continue
        total_calls += retrieval_client.calls
        total_hits += retrieval_client.cache_hits
        retrieval_client.close()
        retrieval_sizes = {
            "successful_trajectories": len(frozen.successful),
            "failed_trajectories": len(frozen.failed),
            "insights": len(frozen.insights),
        }

        for repeat_index in range(args.repeats):
            sample_seed = args.sample_seed_base + repeat_index
            for condition in conditions:
                key = (
                    claim_id,
                    str(candidate["candidate_id"]),
                    repeat_index,
                    condition,
                )
                if key in existing:
                    continue
                progress.set_postfix_str(
                    f"claim={claim_id} repeat={repeat_index} {condition}",
                    refresh=True,
                )
                _update_progress(
                    current={
                        "claim_id": claim_id,
                        "repeat_index": repeat_index,
                        "condition": condition,
                        "phase": "inference",
                    }
                )
                outcome, calls, hits = _run_branch(
                    args,
                    task,
                    frozen,
                    condition,
                    all_recipients,
                    sample_seed,
                    cache_path,
                )
                total_calls += calls
                total_hits += hits
                row = {
                    "runner_schema": RUNNER_SCHEMA,
                    "design_hash": design_hash,
                    "run_hash": run_hash,
                    "event_id": f"fever-{claim_id}-{candidate['candidate_id']}",
                    "task_id": claim_id,
                    "task": {
                        "claim_id": claim_id,
                        "claim": task["claim"],
                        "gold_label": task["label"],
                        # Persisted after inference for objective-metric audit.
                        # The prompt builders never expose this field.
                        "gold_evidence_page_sets": task[
                            "gold_evidence_page_sets"
                        ],
                    },
                    "candidate": candidate,
                    "retrieval_sizes": retrieval_sizes,
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
                resolved_branches += 1
                progress.update(1)
                _update_progress(
                    resolved_branches=resolved_branches,
                    persisted_branches=len(existing),
                    excluded_claims=len(exclusions),
                    current={
                        "claim_id": claim_id,
                        "repeat_index": repeat_index,
                        "condition": condition,
                        "phase": "persisted",
                    },
                    llm_calls_this_process=total_calls,
                    cache_hits_this_process=total_hits,
                )

    progress.close()

    diagnostics = {
        "registered_evaluation_claims": len(selected),
        "excluded_claims": exclusions,
        "completed_branches": len(existing),
        "conditions_per_event": len(conditions),
        "llm_calls_this_process": total_calls,
        "cache_hits_this_process": total_hits,
    }
    (args.output_dir / "collection_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not existing:
        raise SystemExit("no complete FEVER branches were collected")
    _update_progress(
        status="analyzing",
        resolved_branches=resolved_branches,
        persisted_branches=len(existing),
        excluded_claims=len(exclusions),
        current={"phase": "analysis"},
        llm_calls_this_process=total_calls,
        cache_hits_this_process=total_hits,
    )
    analysis_path = write_analysis(
        args.output_dir,
        retest_results=args.retest_results,
        delta=args.delta,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.selection_seed,
    )
    _update_progress(
        status="completed",
        resolved_branches=resolved_branches,
        persisted_branches=len(existing),
        excluded_claims=len(exclusions),
        current=None,
        analysis=str(analysis_path),
        llm_calls_this_process=total_calls,
        cache_hits_this_process=total_hits,
        error=None,
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
    try:
        main()
    except BaseException as error:
        _record_uncaught_error(error)
        raise
