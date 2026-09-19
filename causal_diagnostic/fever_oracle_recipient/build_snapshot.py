#!/usr/bin/env python3
"""Build one frozen native-GMemory snapshot from a registered FEVER split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from tqdm import tqdm


def _early_cli_value(flag: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_CLI_ENDPOINT = _early_cli_value("--endpoint")
if _CLI_ENDPOINT:
    os.environ["OPENAI_API_BASE"] = _CLI_ENDPOINT
else:
    os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:11434/v1")
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_diagnostic.fever_oracle_recipient.data import (
    BINARY_LABELS,
    file_md5,
    load_binary_fever,
    stratified_split,
)
from causal_diagnostic.fever_oracle_recipient.offline_env import OfflineFeverEnv
from causal_diagnostic.fever_oracle_recipient.provenance import SNAPSHOT_SCHEMA
from causal_diagnostic.fever_oracle_recipient.prompts import (
    FEVER_SYSTEM_PROMPT,
    prepare_example,
)
from causal_diagnostic.fever_oracle_recipient.sampling import (
    configure_fever_sampling,
)
from causal_diagnostic.fever_oracle_recipient.utils import seed_everything
from causal_diagnostic.oracle_recipient.seeded_client import SeededCachedChat
from mas.memory.mas_memory.GMemory import GMemory
from mas.reasoning import ReasoningIO
from mas.utils import EmbeddingFunc
from tasks.mas_workflow.macnet.graph_mas import MacNet


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--support-per-label", type=int, default=25)
    parser.add_argument("--evaluation-per-label", type=int, default=50)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--build-seed", type=int, default=42)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--graph-type", default="Chain", choices=(
        "Chain", "FullConnected", "Debate", "Star"
    ))
    parser.add_argument("--node-num", type=int, default=3)
    parser.add_argument("--max-trials", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--successful-topk", type=int, default=3)
    parser.add_argument("--failed-topk", type=int, default=0)
    parser.add_argument("--insights-topk", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--output-memory-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.support_per_label < 1 or args.evaluation_per_label < 1:
        raise SystemExit("support/evaluation counts must be at least 1 per label")
    if args.node_num < 2:
        raise SystemExit("--node-num must be at least 2")
    if args.max_trials != 1:
        raise SystemExit("offline FEVER requires --max-trials 1")
    if args.temperature < 0:
        raise SystemExit("--temperature must be non-negative")


def _stable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_progress(path: Path, **fields: Any) -> None:
    _write_json(
        path,
        {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields},
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _design(args: argparse.Namespace, source_md5: str, support: list[dict[str, Any]], evaluation: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "snapshot_schema": SNAPSHOT_SCHEMA,
        "benchmark": "FEVER_binary_offline",
        "source_md5": source_md5,
        "labels": list(BINARY_LABELS),
        "split_seed": args.split_seed,
        "support_per_label": args.support_per_label,
        "evaluation_per_label": args.evaluation_per_label,
        "support_ids": [int(row["id"]) for row in support],
        "evaluation_ids": [int(row["id"]) for row in evaluation],
        "model": args.model,
        "graph_type": args.graph_type,
        "node_num": args.node_num,
        "max_trials": args.max_trials,
        "temperature": args.temperature,
        "successful_topk": args.successful_topk,
        "failed_topk": args.failed_topk,
        "insights_topk": args.insights_topk,
        "threshold": args.threshold,
        "embedding_model": args.embedding_model,
        "build_seed": args.build_seed,
    }


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    _validate_args(args)
    source = args.data.resolve()
    examples = load_binary_fever(source)
    support, evaluation = stratified_split(
        examples,
        support_per_label=args.support_per_label,
        evaluation_per_label=args.evaluation_per_label,
        seed=args.split_seed,
    )
    design = _design(args, file_md5(source), support, evaluation)
    design_hash = _stable_hash(design)
    output = args.output_memory_dir.resolve()
    progress_path = output / "causal_snapshot_progress.json"
    runs_path = output / "support_runs.jsonl"
    manifest_path = output / "causal_snapshot_manifest.json"

    if args.resume:
        if not output.is_dir() or not progress_path.is_file() or not runs_path.is_file():
            raise SystemExit(
                "--resume requires an existing snapshot directory with progress and runs"
            )
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("design_hash") != design_hash:
            raise SystemExit("resume arguments/source do not match snapshot design")
        if progress.get("status") == "completed" and manifest_path.is_file():
            print(manifest_path)
            return manifest_path
    else:
        if output.exists():
            raise SystemExit(
                f"refusing to overwrite {output}; use a new directory or --resume"
            )
        output.mkdir(parents=True)
        runs_path.touch(exist_ok=False)
        _write_progress(
            progress_path,
            status="initialized",
            design_hash=design_hash,
            design=design,
            total_tasks=len(support),
            completed_tasks=0,
            current_claim_id=None,
            error=None,
        )

    records = _load_jsonl(runs_path)
    expected_prefix = [int(row["id"]) for row in support[: len(records)]]
    actual_prefix = [int(row["claim_id"]) for row in records]
    if actual_prefix != expected_prefix:
        raise SystemExit("support_runs.jsonl is not the registered support prefix")

    cache_path = (
        args.cache.resolve()
        if args.cache is not None
        else output.parent / f".{output.name}_build_cache.sqlite"
    )
    seed_everything(args.build_seed)
    client = SeededCachedChat(
        args.model,
        cache_path,
        experiment_seed=args.build_seed,
        api_base=args.endpoint,
        api_key=args.api_key,
    )
    memory = GMemory(
        namespace=output.name,
        global_config={
            "working_dir": str(output.parent),
            "hop": 1,
            "read_only": False,
        },
        llm_model=client,
        embedding_func=EmbeddingFunc(args.embedding_model),
    )
    if memory.memory_size != len(records):
        client.close()
        raise SystemExit(
            "snapshot resume invariant failed: GMemory memory_size "
            f"{memory.memory_size} != completed support runs {len(records)}; "
            "use a fresh output directory"
        )

    env = OfflineFeverEnv(max_trials=1)
    mas = MacNet()
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
    configure_fever_sampling(mas, args.temperature)
    for agent in mas.agents_team.values():
        agent.add_task_instruction(FEVER_SYSTEM_PROMPT)
    mas._decision_node._agent.add_task_instruction(FEVER_SYSTEM_PROMPT)

    progress_bar = tqdm(
        range(len(records), len(support)),
        desc="Building FEVER support memory",
        unit="claim",
        dynamic_ncols=True,
        mininterval=2.0,
    )
    current_claim_id: int | None = None
    try:
        for index in progress_bar:
            example = support[index]
            task = prepare_example(example)
            current_claim_id = int(task["id"])
            _write_progress(
                progress_path,
                status="running",
                design_hash=design_hash,
                design=design,
                total_tasks=len(support),
                completed_tasks=len(records),
                current_claim_id=current_claim_id,
                error=None,
            )
            env = OfflineFeverEnv(max_trials=1)
            env.set_env(task)
            mas.set_env(env)
            before = memory.memory_size
            calls_before, hits_before = client.calls, client.cache_hits
            reward, done = mas.schedule(task)
            after = memory.memory_size
            if after != before + 1:
                raise RuntimeError(
                    f"GMemory added {after - before} records for one support claim"
                )
            record = {
                "claim_id": current_claim_id,
                "claim": task["claim"],
                "gold_label": task["label"],
                "prediction": env.last_prediction,
                "reward": float(reward),
                "success": bool(done),
                "steps": int(env.infos["steps"]),
                "decision_raw_output": (
                    mas._decision_node.current_output[0]
                    if mas._decision_node.current_output
                    else None
                ),
                "memory_size_after": after,
                "llm_calls": client.calls - calls_before,
                "cache_hits": client.cache_hits - hits_before,
            }
            with runs_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            records.append(record)
            current_claim_id = None
            progress_bar.set_postfix_str(
                f"claim {task['id']}, success={bool(done)}", refresh=True
            )
            _write_progress(
                progress_path,
                status="running",
                design_hash=design_hash,
                design=design,
                total_tasks=len(support),
                completed_tasks=len(records),
                current_claim_id=None,
                last_result=record,
                error=None,
            )
    except BaseException as exc:
        _write_progress(
            progress_path,
            status="failed",
            design_hash=design_hash,
            design=design,
            total_tasks=len(support),
            completed_tasks=len(records),
            current_claim_id=current_claim_id,
            error=repr(exc),
        )
        client.close()
        raise

    successful = sum(bool(row["success"]) for row in records)
    failed = len(records) - successful
    manifest = {
        "snapshot_schema": SNAPSHOT_SCHEMA,
        "design_hash": design_hash,
        "design": design,
        "source": str(source),
        **design,
        "endpoint": args.endpoint or os.environ.get("OPENAI_API_BASE"),
        "cache": str(cache_path),
        "memory_records": memory.memory_size,
        "successful_total": successful,
        "failed_total": failed,
        "support_runs": records,
    }
    _write_json(manifest_path, manifest)
    _write_progress(
        progress_path,
        status="completed",
        design_hash=design_hash,
        design=design,
        total_tasks=len(support),
        completed_tasks=len(records),
        current_claim_id=None,
        error=None,
        manifest=str(manifest_path),
    )
    client.close()
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "memory_records": memory.memory_size,
                "successful_total": successful,
                "failed_total": failed,
                "support_ids": design["support_ids"],
                "evaluation_ids": design["evaluation_ids"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return manifest_path


if __name__ == "__main__":
    main()
