#!/usr/bin/env python3
"""Compare local worker and team causal utility in frozen G-Memory PDDL runs.

Each candidate memory is evaluated twice on an identical held-out task:
with the frozen retrieval and with that one candidate removed. The script
only wraps MacNet nodes to save traces; it does not change scheduling.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import MethodType
from typing import Any

from causal_diagnostic.memory import FrozenReadOnlyGMemory, FrozenRetrieval
from causal_diagnostic.run_intervention import (
    build_live_memory, memory_config, message_id, prepare_task, seed_everything,
)
from mas.llm import GPTChat
from mas.reasoning import ReasoningIO
from mas.utils import EmbeddingFunc
from tasks.envs import get_env
from tasks.mas_workflow import get_mas
from tasks.prompts import get_dataset_system_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidate-kind", choices=("trajectory", "insight"), default="trajectory")
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-trials", type=int, default=50)
    parser.add_argument("--node-num", type=int, default=2)
    parser.add_argument("--graph-type", choices=("Chain", "FullConnected", "Debate", "Star"), default="Chain")
    parser.add_argument("--successful-topk", type=int, default=3)
    parser.add_argument("--failed-topk", type=int, default=0)
    parser.add_argument("--insights-topk", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--cost-weight", type=float, default=0.25)
    parser.add_argument("--classification-margin", type=float, default=1e-9)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def candidate_metadata(frozen: FrozenRetrieval, kind: str, index: int) -> dict[str, Any]:
    if kind == "trajectory":
        candidate = frozen.successful[index]
        return {
            "kind": kind, "index": index,
            "candidate_id": f"trajectory-{message_id(candidate)}",
            "task_main": candidate.task_main, "label": candidate.label,
            "trajectory": candidate.task_trajectory,
        }
    candidate = frozen.insights[index]
    return {
        "kind": kind, "index": index,
        "candidate_id": f"insight-{hashlib.sha256(candidate.encode('utf-8')).hexdigest()[:16]}",
        "text": candidate,
    }


def normalize_action(env: Any, output: Any) -> str:
    return env.process_action(str(output or ""))


def action_is_valid(env: Any, output: Any) -> bool:
    """Parse against the current PDDL state without stepping the environment."""
    action = normalize_action(env, output)
    lowered = action.lower().strip()
    if lowered in {"check valid actions", "look around"}:
        return True
    if "think" in lowered:
        return False
    try:
        return env._text_to_action(action) is not None
    except Exception:
        return False


def install_trace(node: Any, env: Any, trace: list[dict[str, Any]], kind: str) -> None:
    """Wrap Node.execute only to record its output, then return it unchanged."""
    original_execute = node.execute

    def traced_execute(self: Any, user_message: Any, use_critic: bool) -> str:
        output = original_execute(user_message, use_critic)
        trace.append({
            "call_index": len(trace),
            "environment_step_before": int(env.infos.get("steps", 0)),
            "agent_id": self.id, "role": self.role, "kind": kind,
            "raw_output": output, "action": normalize_action(env, output),
            "valid": action_is_valid(env, output),
        })
        return output

    node.execute = MethodType(traced_execute, node)


def summarize_local(trace: list[dict[str, Any]]) -> dict[str, Any]:
    valid_count = sum(int(row["valid"]) for row in trace)
    action_count = len(trace)
    per_agent: dict[str, dict[str, int | float]] = {}
    for row in trace:
        stats = per_agent.setdefault(row["agent_id"], {"valid": 0, "total": 0})
        stats["valid"] += int(row["valid"])
        stats["total"] += 1
    for stats in per_agent.values():
        stats["valid_rate"] = stats["valid"] / stats["total"] if stats["total"] else 0.0
    return {
        "score_definition": "worker action validity rate",
        "score": valid_count / action_count if action_count else 0.0,
        "valid_count": valid_count, "action_count": action_count,
        "per_agent": per_agent, "trace": trace,
    }


def run_branch(args: argparse.Namespace, task_config: dict[str, Any], frozen: FrozenRetrieval, condition: str) -> dict[str, Any]:
    seed_everything(args.seed)
    branch_task = copy.deepcopy(task_config)
    env = get_env("pddl", {}, args.max_trials)
    env.set_env(branch_task)
    llm = GPTChat(model_name=args.model)
    memory = FrozenReadOnlyGMemory(
        namespace=args.memory_dir.name, global_config=memory_config(args.memory_dir),
        llm_model=llm, embedding_func=EmbeddingFunc(args.embedding_model),
        frozen_retrieval=frozen,
    )
    mas = get_mas("macnet")
    mas.build_system(ReasoningIO(llm_model=llm), memory, env, {
        "graph_type": args.graph_type, "node_num": args.node_num,
        "use_critic": False, "successful_topk": args.successful_topk,
        "failed_topk": args.failed_topk, "insights_topk": args.insights_topk,
        "threshold": args.threshold, "use_projector": False,
    })
    instruction = get_dataset_system_prompt("pddl", branch_task)
    for agent in mas.agents_team.values():
        agent.add_task_instruction(instruction)

    worker_trace: list[dict[str, Any]] = []
    decision_trace: list[dict[str, Any]] = []
    for node in mas._agent_nodes.values():
        install_trace(node, env, worker_trace, "worker")
    install_trace(mas._decision_node, env, decision_trace, "decision")
    reward, done = mas.schedule(branch_task)
    steps = int(env.infos.get("steps", 0))
    return {
        "condition": condition, "local": summarize_local(worker_trace),
        "decision_trace": decision_trace,
        "team": {
            "success": bool(done), "reward": reward, "steps": steps,
            "score_definition": "final_reward - cost_weight * steps / max_trials",
            "cost_weight": args.cost_weight,
            "score": float(reward) - args.cost_weight * steps / args.max_trials,
            "actions": [state.graph.get("action") for state in memory.current_task_context.chain_of_states],
        },
        "task_trajectory": memory.current_task_context.task_trajectory,
    }


def classify(local_utility: float, team_utility: float, margin: float) -> str:
    if local_utility > margin and team_utility < -margin:
        return "local_positive_team_negative"
    if local_utility < -margin and team_utility > margin:
        return "local_negative_team_positive"
    if abs(local_utility) <= margin and abs(team_utility) <= margin:
        return "both_neutral"
    return "aligned_or_unclassified"


def main() -> Path:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    task_config, _ = prepare_task(args.task_id, args.max_trials)
    llm = GPTChat(model_name=args.model)
    live_memory = build_live_memory(args, llm, EmbeddingFunc(args.embedding_model))
    baseline = FrozenRetrieval.from_result(live_memory.retrieve_memory(
        query_task=task_config["task_main"], successful_topk=args.successful_topk,
        failed_topk=args.failed_topk, insight_topk=args.insights_topk, threshold=args.threshold,
    ))
    candidate = candidate_metadata(baseline, args.candidate_kind, args.candidate_index)
    with_memory = run_branch(args, task_config, baseline, "with_memory")
    without_candidate = run_branch(args, task_config, baseline.without(args.candidate_kind, args.candidate_index), "without_candidate")
    local_utility = with_memory["local"]["score"] - without_candidate["local"]["score"]
    team_utility = with_memory["team"]["score"] - without_candidate["team"]["score"]
    result = {
        "experiment": "gmemory_local_team_frozen_intervention", "task_id": args.task_id,
        "task": {key: task_config[key] for key in ("game_name", "problem_index", "task_main")},
        "seed": args.seed, "memory_dir": str(args.memory_dir.resolve()), "read_only_memory": True,
        "candidate": candidate,
        "retrieval_sizes": {"successful_trajectories": len(baseline.successful), "failed_trajectories": len(baseline.failed), "insights": len(baseline.insights)},
        "with_memory": with_memory, "without_candidate": without_candidate,
        "utility": {
            "local_utility": local_utility, "team_utility_score": team_utility,
            "success_delta": int(with_memory["team"]["success"]) - int(without_candidate["team"]["success"]),
            "reward_delta": with_memory["team"]["reward"] - without_candidate["team"]["reward"],
            "step_delta": without_candidate["team"]["steps"] - with_memory["team"]["steps"],
            "classification": classify(local_utility, team_utility, args.classification_margin),
        },
    }
    output = args.output_dir / f"task_{args.task_id}_{candidate['candidate_id']}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    return output


if __name__ == "__main__":
    main()
