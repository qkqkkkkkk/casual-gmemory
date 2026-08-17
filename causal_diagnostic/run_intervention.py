#!/usr/bin/env python3
"""Run one frozen-memory intervention on one PDDL task with MacNet.

The script first retrieves candidates once from an existing G-Memory store.  It
then runs two fresh, identically seeded PDDL/MacNet episodes: the original
retrieval and the same retrieval with one trajectory or insight removed.
Neither branch writes to the supplied memory directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

# Keep the diagnostic runnable on the same restricted networks as tasks/run.py.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_diagnostic.memory import FrozenReadOnlyGMemory, FrozenRetrieval
from mas.llm import GPTChat
from mas.memory.common import MASMessage
from mas.memory.mas_memory.GMemory import GMemory
from mas.reasoning import ReasoningIO
from mas.utils import EmbeddingFunc
from tasks.envs import get_env, get_task
from tasks.mas_workflow import get_mas
from tasks.prompts import get_dataset_system_prompt, get_task_few_shots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True, help="Index in data/pddl/test.jsonl")
    parser.add_argument("--memory-dir", type=Path, required=True, help="Existing g-memory persistence directory")
    parser.add_argument("--model", required=True, help="OpenAI-compatible model name")
    parser.add_argument("--candidate-kind", choices=("trajectory", "insight"), default="trajectory")
    parser.add_argument("--candidate-index", type=int, default=0, help="Zero-based index in the frozen retrieval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-trials", type=int, default=30)
    parser.add_argument("--node-num", type=int, default=3)
    parser.add_argument("--graph-type", choices=("Chain", "FullConnected", "Debate", "Star"), default="Chain")
    parser.add_argument("--successful-topk", type=int, default=3)
    parser.add_argument("--failed-topk", type=int, default=0)
    parser.add_argument("--insights-topk", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-dir", type=Path, required=True)
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


def message_id(message: MASMessage) -> str:
    payload = json.dumps(MASMessage.to_dict(message), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def prepare_task(task_id: int, max_trials: int) -> tuple[dict[str, Any], Any]:
    tasks = get_task("pddl")
    if not 0 <= task_id < len(tasks):
        raise ValueError(f"task-id must be between 0 and {len(tasks) - 1}")

    task_config = copy.deepcopy(tasks[task_id])
    env_config: dict[str, Any] = {}
    env = get_env("pddl", env_config, max_trials)
    task_main, task_description = env.set_env(task_config)
    task_config.update(
        task_main=task_main,
        task_description=task_description,
        few_shots=get_task_few_shots("pddl", task_config, few_shots_num=1),
    )
    task_instruction = get_dataset_system_prompt("pddl", task_config)
    return task_config, (env, task_instruction)


def memory_config(memory_dir: Path) -> dict[str, Any]:
    memory_dir = memory_dir.resolve()
    if not memory_dir.exists():
        raise FileNotFoundError(f"Memory directory does not exist: {memory_dir}")
    if not memory_dir.is_dir():
        raise NotADirectoryError(memory_dir)
    # MASMemoryBase appends namespace to working_dir, so use the directory above
    # the existing g-memory folder and keep the namespace unchanged.
    return {"working_dir": str(memory_dir.parent), "hop": 1, "read_only": True}


def build_live_memory(args: argparse.Namespace, llm: GPTChat, embedder: EmbeddingFunc) -> GMemory:
    return GMemory(
        namespace=args.memory_dir.name,
        global_config=memory_config(args.memory_dir),
        llm_model=llm,
        embedding_func=embedder,
    )


def run_branch(
    args: argparse.Namespace,
    task_config: dict[str, Any],
    frozen_retrieval: FrozenRetrieval,
    condition: str,
) -> dict[str, Any]:
    seed_everything(args.seed)
    branch_task = copy.deepcopy(task_config)
    env_config: dict[str, Any] = {}
    env = get_env("pddl", env_config, args.max_trials)
    env.set_env(branch_task)

    llm = GPTChat(model_name=args.model)
    embedder = EmbeddingFunc(args.embedding_model)
    memory = FrozenReadOnlyGMemory(
        namespace=args.memory_dir.name,
        global_config=memory_config(args.memory_dir),
        llm_model=llm,
        embedding_func=embedder,
        frozen_retrieval=frozen_retrieval,
    )
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
    task_instruction = get_dataset_system_prompt("pddl", branch_task)
    for agent in mas.agents_team.values():
        agent.add_task_instruction(task_instruction)

    memory_records_before = memory.memory_size
    reward, done = mas.schedule(branch_task)
    memory_records_after = memory.memory_size
    state_chain = memory.current_task_context.chain_of_states
    actions = [state.graph.get("action") for state in state_chain]
    return {
        "condition": condition,
        "reward": reward,
        "success": bool(done),
        "steps": int(env.infos.get("steps", len(actions))),
        "actions": actions,
        "task_trajectory": memory.current_task_context.task_trajectory,
        "memory_records_before": memory_records_before,
        "memory_records_after": memory_records_after,
    }


def candidate_metadata(frozen: FrozenRetrieval, kind: str, index: int) -> dict[str, Any]:
    if kind == "trajectory":
        candidate = frozen.successful[index]
        return {
            "kind": kind,
            "index": index,
            "candidate_id": f"trajectory-{message_id(candidate)}",
            "task_main": candidate.task_main,
            "label": candidate.label,
            "trajectory": candidate.task_trajectory,
        }
    candidate = frozen.insights[index]
    return {
        "kind": kind,
        "index": index,
        "candidate_id": f"insight-{hashlib.sha256(candidate.encode('utf-8')).hexdigest()[:16]}",
        "text": candidate,
    }


def main() -> Path:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    task_config, _ = prepare_task(args.task_id, args.max_trials)
    llm = GPTChat(model_name=args.model)
    embedder = EmbeddingFunc(args.embedding_model)
    live_memory = build_live_memory(args, llm, embedder)
    baseline = FrozenRetrieval.from_result(
        live_memory.retrieve_memory(
            query_task=task_config["task_main"],
            successful_topk=args.successful_topk,
            failed_topk=args.failed_topk,
            insight_topk=args.insights_topk,
            threshold=args.threshold,
        )
    )
    candidate = candidate_metadata(baseline, args.candidate_kind, args.candidate_index)
    without_candidate = baseline.without(args.candidate_kind, args.candidate_index)

    with_result = run_branch(args, task_config, baseline, "with_memory")
    without_result = run_branch(args, task_config, without_candidate, "without_candidate")
    result = {
        "experiment": "gmemory_frozen_single_candidate_intervention",
        "task_id": args.task_id,
        "task": {key: task_config[key] for key in ("game_name", "problem_index", "task_main")},
        "seed": args.seed,
        "memory_dir": str(args.memory_dir.resolve()),
        "read_only_memory": True,
        "candidate": candidate,
        "retrieval_sizes": {
            "successful_trajectories": len(baseline.successful),
            "failed_trajectories": len(baseline.failed),
            "insights": len(baseline.insights),
        },
        "with_memory": with_result,
        "without_candidate": without_result,
        "utility": {
            "success_delta": int(with_result["success"]) - int(without_result["success"]),
            "reward_delta": with_result["reward"] - without_result["reward"],
            "step_delta": without_result["steps"] - with_result["steps"],
        },
    }
    output_path = args.output_dir / f"task_{args.task_id}_{candidate['candidate_id']}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)
    return output_path


if __name__ == "__main__":
    main()
