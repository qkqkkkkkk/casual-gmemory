"""Isolated benchmark stages. Invoked by run_interactive_gate_benchmark."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback
import uuid

from .interactive_data import digest, files_identity, save, sha
from .interactive_protocol import learned_methods, policies


def progress(root, stage, **fields):
    save(Path(root) / "progress.json", {"stage": stage, **fields})


def verify_inputs(config, name):
    from .interactive_runtime import model_identity

    data = config["experiments"][name]
    for task in data["tasks"]:
        for path, expected in task["files"].items():
            if sha(Path(config["repo"]) / path) != expected:
                raise ValueError(f"registered task input changed: {path}")
    for path, expected in config["code_sha256"].items():
        if sha(Path(config["repo"]) / path) != expected:
            raise ValueError(f"registered implementation changed: {path}")
    if model_identity(config["endpoint"], config["model"]) != config["model_identity"]:
        raise ValueError("Ollama model/template/parameters changed")


def snapshot(config, name):
    root = Path(config["output_dir"]) / name
    manifest = json.loads((root / "snapshot.json").read_text())
    path = root / manifest["memory_dir"]
    if manifest["config_digest"] != digest(config) or files_identity(path) != manifest["files_sha256"]:
        raise ValueError("frozen support snapshot changed")
    return path


def support(config, name):
    from .interactive_runtime import NativeRunner, clone_memory

    root = Path(config["output_dir"]) / name
    if (root / "snapshot.json").exists():
        snapshot(config, name)
        return
    data = config["experiments"][name]
    tasks = {r["id"]: r for r in data["tasks"]}
    previous = None
    for index, task_id in enumerate(data["split"]["support"]):
        marker = root / "support/committed" / f"{index:04d}.json"
        if marker.exists():
            committed = json.loads(marker.read_text())
            if committed["task_id"] != task_id or committed["config_digest"] != digest(config):
                raise ValueError("support commit identity changed")
            previous = root / committed["directory"]
            if files_identity(previous / "g-memory") != committed["files_sha256"]:
                raise ValueError("previous support transaction changed")
            continue
        # Each support episode commits its own full snapshot. Failed attempts
        # remain reviewable; resume starts from the preceding committed state.
        attempt = root / "support/attempts" / (f"{index:04d}-" + uuid.uuid4().hex[:10])
        attempt.mkdir(parents=True)
        if previous:
            clone_memory(previous / "g-memory", attempt / "g-memory")
        runner = NativeRunner(config, root / "support", attempt / "g-memory", writable=True)
        try:
            row = runner.run(tasks[task_id], 10000 + index)
            save(attempt / "episode.json", row)
        finally:
            runner.close()
        # Freeze a consistent copy instead of copying a live WAL on the next task.
        committed_dir = attempt / "committed"
        stable = committed_dir / "g-memory"
        clone_memory(attempt / "g-memory", stable)
        save(marker, {"task_id": task_id, "directory": str(committed_dir.relative_to(root)),
            "config_digest": digest(config), "files_sha256": files_identity(stable),
            "episode_sha256": sha(attempt / "episode.json")})
        previous = committed_dir
        progress(root, "support", completed=index+1, total=len(data["split"]["support"]))
    save(root / "snapshot.json", {"schema": "interactive-frozen-support-v1", "config_digest": digest(config),
        "support_ids": data["split"]["support"], "memory_dir": str((previous / "g-memory").relative_to(root)),
        "files_sha256": files_identity(previous / "g-memory"),
        "support_episode_sha256": {p.name: sha(p) for p in sorted((root / "support/committed").glob("*.json"))}})


def collect(config, name):
    from .interactive_runtime import ActionStore, NativeRunner, baseline_for, execute

    root = Path(config["output_dir"]) / name
    memory = snapshot(config, name)
    store = ActionStore(root / "diagnostic/actions", {"config_digest": digest(config),
                        "snapshot_sha256": sha(root / "snapshot.json"), "phase": "diagnostic"})
    runner = NativeRunner(config, root, memory)
    data = config["experiments"][name]
    by_id = {r["id"]: r for r in data["tasks"]}
    ids = sorted(data["split"]["fit"] + data["split"]["dev"])
    try:
        for task_number, task_id in enumerate(ids):
            path = root / "diagnostic/tasks" / (digest(task_id) + ".json")
            if path.exists():
                continue
            uses, drops, references = [], None, []
            first = None
            for repeat in range(config["repeats"]):
                seed = config["diagnostic_seed"] + repeat
                baseline = baseline_for(runner, store, by_id[task_id], seed)
                if not baseline["candidates"]:
                    raise ValueError(f"no diagnostic candidates: {task_id}; inspect support successes")
                if first is not None and any(first[k] != baseline[k] for k in
                        ("query", "state", "candidates", "retrieval_fingerprint", "initial_fingerprint")):
                    raise ValueError("diagnostic candidate lists changed across repeats")
                first = baseline
                uses.append(baseline["outcome"]["success"])
                if drops is None:
                    drops = [[] for _ in baseline["candidates"]]
                for index, candidate in enumerate(baseline["candidates"]):
                    progress(root, "collect", completed_tasks=task_number, total_tasks=len(ids),
                        current_task=task_id, repeat=repeat, single_drop_index=index)
                    branch = execute(runner, store, by_id[task_id], seed, [candidate["memory_id"]], baseline)
                    drops[index].append(branch["outcome"]["success"])
                    references.append({"repeat": repeat, "sample_seed": seed,
                        "candidate_id": candidate["memory_id"], "keep_key": baseline["action_key"],
                        "drop_key": branch["action_key"]})
            save(path, {k: first[k] for k in ("task_id", "query", "state", "task_config", "candidates",
                "exposure_count", "retrieval_fingerprint", "initial_fingerprint")} | {
                "repeats": config["repeats"], "q_use": statistics.mean(uses),
                "q_drop": [statistics.mean(v) for v in drops], "use_samples": uses, "drop_samples": drops,
                "branches": references, "config_digest": digest(config)})
            progress(root, "collect", completed_tasks=task_number+1, total_tasks=len(ids))
    finally:
        runner.close()
    snapshot(config, name)


def experts(config, name):
    root = Path(config["output_dir"]) / name
    data = config["experiments"][name]
    ids = sorted(data["split"]["fit"] + data["split"]["dev"])
    for index, task_id in enumerate(ids):
        path = root / "experts" / (digest(task_id) + ".json")
        if path.exists():
            result = json.loads(path.read_text())
            if result["task_id"] != task_id or result["validation"]["success"] != 1:
                raise ValueError("invalid completed expert episode")
            continue
        progress(root, "experts", current_task=task_id, completed=index, total=len(ids))
        # Planner/library calls run in a bounded child. On timeout or invalid
        # plan, stop with the missing task visible; never fabricate a target.
        run_expert_task(config, name, task_id)


def run_expert_task(config, name, task_id):
    from .interactive_envs import expert_process_environment

    command = [sys.executable, "-X", "faulthandler", "-m", "causal_memory_control.interactive_worker",
        "--config", str(Path(config["output_dir"]) / "benchmark_manifest.json"),
        "--dataset", name, "--stage", "expert_one", "--task-id", task_id]
    subprocess.run(command, check=True, timeout=config["expert_timeout"], env=expert_process_environment())


def expert_one(config, name, task_id):
    from .interactive_envs import _make_expert_once

    data = config["experiments"][name]
    if task_id not in data["split"]["fit"] + data["split"]["dev"]:
        raise ValueError("expert supervision is restricted to fit/dev")
    task = next(r for r in data["tasks"] if r["id"] == task_id)
    # run_expert_task already provides a fresh, bounded process for this task.
    result = _make_expert_once(task, config["repo"], max_steps=200,
        traceback_interval=min(60.0, max(0.1, config["expert_timeout"] * 0.8)))
    save(Path(config["output_dir"]) / name / "experts" / (digest(task_id) + ".json"), result)


def policy_record(config, policy, baseline, branch, drops, decisions=None):
    return {"policy": policy, "task_id": baseline["task_id"], "seed": baseline["seed"],
        "config_digest": digest(config), "initial_fingerprint": baseline["initial_fingerprint"],
        "retrieval_fingerprint": baseline["retrieval_fingerprint"],
        "retrieved_count": len(baseline["candidates"]), "dropped_count": len(drops), "drop_ids": sorted(drops),
        "decisions": decisions or [{"memory_id": c["memory_id"],
            "action": "DROP" if c["memory_id"] in drops else "KEEP"} for c in baseline["candidates"]],
        "outcome": branch["outcome"], "action_keys": [branch["action_key"]]}


def evaluate(config, name):
    import random
    from .interactive_runtime import ActionStore, NativeRunner, baseline_for, execute
    from .interactive_training import learned_gates, select
    from .csm_st_data import write_rows

    root = Path(config["output_dir"]) / name
    memory = snapshot(config, name)
    checkpoints = {m: sha(root / m / "checkpoint.json") for m in learned_methods(config)}
    store = ActionStore(root / "evaluation/actions", {"config_digest": digest(config),
        "snapshot_sha256": sha(root / "snapshot.json"), "checkpoints_sha256": checkpoints, "phase": "test"})
    gates = learned_gates(config, name)
    runner = NativeRunner(config, root, memory)
    data = config["experiments"][name]
    tasks = {r["id"]: r for r in data["tasks"]}
    completed = []
    try:
        for index, task_id in enumerate(data["split"]["test"]):
            part = root / "evaluation/policy_parts" / (digest(task_id) + ".json")
            if part.exists():
                records = json.loads(part.read_text())
                if any(r["config_digest"] != digest(config) for r in records):
                    raise ValueError("completed policy configuration changed")
                expected = {m for m, _ in policies(config)} | {f"random_drop_seed{s}" for s in config["random_seeds"]}
                if len(records) != len(expected) or {r["policy"] for r in records} != expected:
                    raise ValueError("completed policy file lacks registered strategy coverage")
                completed.extend(records)
                continue
            seed = config["evaluation_seed"]
            progress(root, "evaluate", completed=index, total=len(data["split"]["test"]), current_task=task_id)
            baseline = baseline_for(runner, store, tasks[task_id], seed)
            ids = [c["memory_id"] for c in baseline["candidates"]]
            actions = [(baseline, [])]
            for candidate in ids:
                drops = [candidate]
                branch = execute(runner, store, tasks[task_id], seed, drops, baseline)
                actions.append((branch, drops))
            drop_all = execute(runner, store, tasks[task_id], seed, ids, baseline)
            actions.append((drop_all, ids))
            records = [policy_record(config, "always_use", baseline, baseline, []),
                       policy_record(config, "global_drop", baseline, drop_all, ids)]
            random_rows = []
            for random_seed in config["random_seeds"]:
                rng = random.Random(int(digest([random_seed, task_id]), 16))
                drops = rng.sample(ids, min(1, len(ids)))
                branch = execute(runner, store, tasks[task_id], seed, drops, baseline)
                r = policy_record(config, f"random_drop_seed{random_seed}", baseline, branch, drops)
                records.append(r)
                random_rows.append(r)
            average = {**random_rows[0], "policy": "random_drop", "drop_ids": None, "decisions": None,
                "dropped_count": statistics.mean(r["dropped_count"] for r in random_rows),
                "action_keys": [r["action_keys"][0] for r in random_rows],
                "outcome": {**random_rows[0]["outcome"], **{k: statistics.mean(r["outcome"][k] for r in random_rows)
                    for k in ("success", "raw_reward", "steps", "invalid_actions")},
                    "progress": statistics.mean(r["outcome"]["progress"] for r in random_rows) if name == "pddl" else None,
                    "done": None, "won": None, "termination_reason": "aggregate_over_random_seeds"}}
            # With no candidates, average is exactly the shared baseline.
            if not ids:
                average["outcome"] = baseline["outcome"]
            records.append(average)
            for method, gate in gates.items():
                drops, decisions = select(gate, baseline)
                branch = execute(runner, store, tasks[task_id], seed, drops, baseline)
                records.append(policy_record(config, method, baseline, branch, drops, decisions))
            # Primary Success; deterministic conservative tie break by action order.
            winner, drops = max(actions, key=lambda pair: pair[0]["outcome"]["success"])
            oracle = policy_record(config, "oracle", baseline, winner, drops)
            oracle["oracle_actions"] = [{"drop_ids": d, "success": r["outcome"]["success"],
                "action_key": r["action_key"]} for r, d in actions]
            oracle["post_hoc"] = True
            oracle["primary_selection_metric"] = "success"
            records.append(oracle)
            save(part, records)
            completed.extend(records)
        write_rows(root / "evaluation/policies.jsonl", completed)
    finally:
        runner.close()
        for gate in gates.values():
            gate.close()
    snapshot(config, name)


def smoke(config, name):
    from .interactive_runtime import NativeRunner, audit
    from .interactive_envs import Episode

    root = Path(config["output_dir"]) / name / "smoke"
    data = config["experiments"][name]
    task = next(r for r in data["tasks"] if r["id"] == data["split"]["fit"][0])
    episode = Episode(task, config["repo"], config["max_steps"])
    try:
        episode.reset()
        episode.reset()
        save(root / "environment.json", {"task_id": task["id"], "resets": episode.resets})
    finally:
        episode.close()
    run_expert_task(config, name, task["id"])
    expert = json.loads((Path(config["output_dir"]) / name / "experts" / (digest(task["id"]) + ".json")).read_text())
    save(root / "expert.json", expert)
    runner = NativeRunner(config, root, root / "empty-memory")
    try:
        baseline = runner.run(task, config["evaluation_seed"])
        replay = runner.run(task, config["evaluation_seed"], expected=baseline)
        audit(replay, baseline, [])
        save(root / "baseline.json", baseline)
        save(root / "replay.json", replay)
        save(root / "summary.json", {"status": "passed", "task_id": task["id"],
            "no_drop_parity": True, "expert_success": expert["validation"]["success"],
            "note": "empty-memory smoke only; full-snapshot candidate coverage checked during collection"})
    finally:
        runner.close()


def main(argv=None):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("pddl", "alfworld"), required=True)
    parser.add_argument("--stage", choices=("support", "collect", "experts", "expert_one", "labels",
                                            "train_reward", "train_csm", "labels_team_u", "train_team_u",
                                            "evaluate", "smoke"), required=True)
    parser.add_argument("--task-id")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    root = Path(config["output_dir"]) / args.dataset
    try:
        if args.stage != "expert_one":
            verify_inputs(config, args.dataset)
        if args.stage in ("labels_team_u", "train_team_u"):
            if not config.get("include_local_global_team_u"):
                raise ValueError("Team U is not registered in this experiment")
            from .interactive_team_u import labels, train
            {"labels_team_u": labels, "train_team_u": train}[args.stage](config, args.dataset)
        elif args.stage in ("labels", "train_reward", "train_csm"):
            from .interactive_training import ce_labels, csm_train, reward_train
            {"labels": ce_labels, "train_reward": reward_train, "train_csm": csm_train}[args.stage](config, args.dataset)
        elif args.stage == "expert_one":
            expert_one(config, args.dataset, args.task_id)
        else:
            globals()[args.stage](config, args.dataset)
        if args.stage != "expert_one":
            progress(root, args.stage, status="completed")
    except Exception:
        if args.stage != "expert_one":
            save(root / "errors" / (args.stage + "-" + uuid.uuid4().hex + ".json"), {
                "stage": args.stage, "command": sys.argv, "traceback": traceback.format_exc()})
            progress(root, args.stage, status="failed", error_directory=str(root / "errors"))
        raise


if __name__ == "__main__":
    main()
