#!/usr/bin/env python3
"""Build a G-Memory PDDL snapshot from a disjoint support-task set."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

# The original runner uses this mirror because many cluster nodes cannot reach
# huggingface.co directly. Respect an explicit user-provided endpoint instead.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mas.llm import GPTChat
from mas.memory.mas_memory.GMemory import GMemory
from mas.reasoning import ReasoningIO
from mas.utils import EmbeddingFunc
from tasks.envs import get_env, get_task
from tasks.mas_workflow import get_mas
from tasks.prompts import get_dataset_system_prompt, get_task_few_shots


def parse_task_ids(value: str) -> list[int]:
    """Parse `0-9,12,15-17` into ordered, unique integer task ids."""
    ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"Invalid descending range: {part}")
            ids.extend(range(start, end + 1))
        else:
            ids.append(int(part))
    if not ids:
        raise argparse.ArgumentTypeError("At least one support task id is required")
    return list(dict.fromkeys(ids))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids", type=parse_task_ids, required=True)
    parser.add_argument("--output-memory-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, help="OpenAI-compatible model name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-trials", type=int, default=30)
    parser.add_argument("--node-num", type=int, default=3)
    parser.add_argument("--graph-type", choices=("Chain", "FullConnected", "Debate", "Star"), default="Chain")
    parser.add_argument("--successful-topk", type=int, default=3)
    parser.add_argument("--failed-topk", type=int, default=0)
    parser.add_argument("--insights-topk", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def prepare_task(task_id: int, max_trials: int) -> tuple[dict[str, Any], Any]:
    tasks = get_task("pddl")
    if not 0 <= task_id < len(tasks):
        raise ValueError(f"task-id must be between 0 and {len(tasks) - 1}")
    task_config = copy.deepcopy(tasks[task_id])
    env = get_env("pddl", {}, max_trials)
    task_main, task_description = env.set_env(task_config)
    task_config.update(
        task_main=task_main,
        task_description=task_description,
        few_shots=get_task_few_shots("pddl", task_config, few_shots_num=1),
    )
    return task_config, env


def main() -> Path:
    args = parse_args()
    output_memory_dir = args.output_memory_dir.resolve()
    if output_memory_dir.exists():
        raise FileExistsError(
            f"Refusing to reuse or overwrite an existing memory snapshot: {output_memory_dir}"
        )
    output_memory_dir.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    llm = GPTChat(model_name=args.model)
    embedding = EmbeddingFunc(args.embedding_model)
    memory = GMemory(
        namespace=output_memory_dir.name,
        global_config={"working_dir": str(output_memory_dir.parent), "hop": 1, "read_only": False},
        llm_model=llm,
        embedding_func=embedding,
    )

    first_task, env = prepare_task(args.task_ids[0], args.max_trials)
    mas = get_mas("macnet")
    mas.build_system(
        ReasoningIO(llm_model=llm),
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

    records: list[dict[str, Any]] = []
    for task_id in args.task_ids:
        task_config, task_env = prepare_task(task_id, args.max_trials)
        mas.set_env(task_env)
        instruction = get_dataset_system_prompt("pddl", task_config)
        for agent in mas.agents_team.values():
            agent.add_task_instruction(instruction)
        reward, done = mas.schedule(task_config)
        records.append(
            {
                "task_id": task_id,
                "game_name": task_config["game_name"],
                "problem_index": task_config["problem_index"],
                "reward": reward,
                "success": bool(done),
                "steps": int(task_env.infos.get("steps", 0)),
            }
        )

    manifest = {
        "experiment": "gmemory_pddl_support_memory_snapshot",
        "seed": args.seed,
        "model": args.model,
        "task_ids": args.task_ids,
        "graph_type": args.graph_type,
        "node_num": args.node_num,
        "records": records,
        "memory_records": memory.memory_size,
    }
    manifest_path = output_memory_dir / "causal_snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest_path)
    return manifest_path


if __name__ == "__main__":
    main()
