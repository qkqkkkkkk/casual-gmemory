"""Resume one registered dataset without changing the experiment manifest.

Kept outside interactive_*.py so adding this orchestration-only entry point does
not change the implementation fingerprint of existing v10 measurements.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import uuid

from .interactive_data import digest, save, sha
from .interactive_progress import STAGE_NAMES, run_logged, show_progress
from .interactive_protocol import learned_methods, policies, stages


@contextmanager
def experiment_lock(root):
    with Path(str(root) + ".lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"experiment is still running/locked: {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def registered_config(root, dataset):
    root = Path(root).resolve()
    config = json.loads((root / "benchmark_manifest.json").read_text())
    if Path(config["output_dir"]).resolve() != root:
        raise ValueError("manifest output directory does not match the existing result root")
    if Path(config["repo"]).resolve() != Path(__file__).resolve().parents[1]:
        raise ValueError("run this entry point in the registered repository")
    if dataset not in config["experiments"]:
        raise ValueError(f"dataset was not registered: {dataset}")
    if "cv_splits" in config["experiments"][dataset]:
        raise ValueError("this entry point requires a fixed-holdout dataset, not PDDL CV folds")
    if not config.get("exclude_ce") or not config.get("include_local_global_team_u"):
        raise ValueError("this entry point requires the registered eight-policy no-CE experiment")
    for path, expected in config["code_sha256"].items():
        if sha(Path(config["repo"]) / path) != expected:
            raise ValueError(f"registered implementation changed: {path}")
    for task in config["experiments"][dataset]["tasks"]:
        for path, expected in task["files"].items():
            if sha(Path(config["repo"]) / path) != expected:
                raise ValueError(f"registered task input changed: {path}")
    return config


def backup_results(root, dataset, config, destination=None):
    """Copy the entire stopped experiment once, including caches and progress."""
    record = root / "dataset_resume" / dataset / "backup.json"
    if destination is None and record.exists():
        previous = json.loads(record.read_text())
        path = Path(previous["backup_dir"])
        manifest = path / "benchmark_manifest.json"
        if (previous["config_digest"] == digest(config) and manifest.is_file()
                and sha(manifest) == sha(root / "benchmark_manifest.json")):
            print(f"[备份] 复用首次续跑前的完整备份：{path}", flush=True)
            return path
        raise ValueError("recorded backup is missing or changed; inspect it before resuming")
    destination = (Path(destination).resolve() if destination else
        root.with_name(root.name + ".backup-before-" + dataset + "-" +
                       datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]))
    if (destination.exists() or root == destination or root in destination.parents
            or destination in root.parents or not destination.parent.is_dir()):
        raise ValueError("backup must be a new independent directory with an existing parent")
    temporary = destination.with_name(destination.name + ".copying-" + uuid.uuid4().hex)
    files = [p for p in root.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    if shutil.disk_usage(destination.parent).free < total_bytes:
        raise ValueError(f"insufficient backup disk space; need at least {total_bytes} bytes; original results preserved")
    print(f"[备份] 完整复制现有结果、缓存和进度：{destination}", flush=True)
    copied, last_update = 0, time.monotonic()
    show_progress("备份现有结果", 0, len(files), f"{total_bytes / 1024**3:.2f} GiB")

    def copy_file(source, target):
        nonlocal copied, last_update
        result = shutil.copy2(source, target)
        copied += 1
        now = time.monotonic()
        if now - last_update >= 1 or copied == len(files):
            show_progress("备份现有结果", copied, len(files))
            last_update = now
        return result

    try:
        shutil.copytree(root, temporary, copy_function=copy_file)
        if sha(temporary / "benchmark_manifest.json") != sha(root / "benchmark_manifest.json"):
            raise ValueError("manifest changed during backup")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    save(record, {"backup_dir": str(destination), "config_digest": digest(config),
                  "manifest_sha256": sha(root / "benchmark_manifest.json")})
    print(f"[备份完成] {destination}", flush=True)
    return destination


def dataset_report(config, dataset):
    """Report one dataset using the original full configuration identity."""
    from .interactive_report import policy_statistics
    from .interactive_worker import snapshot

    root = Path(config["output_dir"]) / dataset
    snapshot(config, dataset)
    data = config["experiments"][dataset]
    tasks = {row["id"]: row for row in data["tasks"]}
    expected_ids = set(data["split"]["test"])
    registered = [method for method, _ in policies(config)]
    expected_methods = set(registered) | {f"random_drop_seed{s}" for s in config["random_seeds"]}
    groups = {method: {} for method in expected_methods}
    source = root / "evaluation/policies.jsonl"
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        method, task_id = row["policy"], row["task_id"]
        if (row["config_digest"] != digest(config) or bool(row.get("simulation", False))
                or method not in groups or task_id not in expected_ids
                or task_id in groups[method]):
            raise ValueError("unregistered, duplicate or simulated evaluation result")
        groups[method][task_id] = row
    if not expected_ids or any(set(rows) != expected_ids for rows in groups.values()):
        raise ValueError("incomplete strategy/task coverage; refusing partial averages")
    primary = data.get("primary_test_ids", sorted(expected_ids))
    cohorts = data.get("test_cohorts", {})
    if not primary or len(set(primary)) != len(primary) or not set(primary) <= expected_ids:
        raise ValueError("invalid primary evaluation cohort")
    flattened = [task_id for ids in cohorts.values() for task_id in ids]
    if cohorts and (set(flattened) != expected_ids or len(flattened) != len(expected_ids)):
        raise ValueError("test cohorts must exactly partition all evaluated tasks")
    baseline = groups["always_use"]
    results = {}
    for method, rows in groups.items():
        for task_id, row in rows.items():
            base = baseline[task_id]
            if (row["seed"] != base["seed"] or row["seed"] != config["evaluation_seed"]
                    or row["initial_fingerprint"] != base["initial_fingerprint"]
                    or row["retrieval_fingerprint"] != base["retrieval_fingerprint"]
                    or row["retrieved_count"] != base["retrieved_count"]
                    or (row["dropped_count"] == 0 and row["outcome"] != base["outcome"])):
                raise ValueError("report initial-state/seed/retrieval/no-DROP parity mismatch")
        results[method] = policy_statistics(sorted(primary), rows, baseline, tasks, dataset)
        if cohorts:
            results[method]["cohort_results"] = {cohort: policy_statistics(sorted(ids), rows, baseline, tasks, dataset)
                                                    for cohort, ids in cohorts.items()}
    for task_id in expected_ids:
        if groups["oracle"][task_id]["outcome"]["success"] < groups["global_drop"][task_id]["outcome"]["success"]:
            raise ValueError("Oracle must cover DROP ALL")
    results["random_drop"]["random_seed_success"] = {str(s): results[f"random_drop_seed{s}"]["success"]
                                                         for s in config["random_seeds"]}
    results["random_drop"]["random_seed_success_stdev"] = statistics.stdev(
        results[f"random_drop_seed{s}"]["success"] for s in config["random_seeds"])
    payload = {"schema": "interactive-dataset-only-report-v1", "dataset": dataset,
        "config_digest": digest(config), "manifest_sha256": sha(Path(config["output_dir"]) / "benchmark_manifest.json"),
        "evaluation_log_sha256": sha(source), "snapshot_sha256": sha(root / "snapshot.json"),
        "checkpoints_sha256": {m: sha(root / m / "checkpoint.json") for m in learned_methods(config)},
        "primary_test_ids": sorted(primary), "test_cohorts": cohorts,
        "bootstrap": "paired_cluster_percentile_2000_resamples_seed71_by_scene_or_problem",
        "interpretation": "dataset-only report; other registered datasets remain paused; Oracle is post hoc",
        "metric": "success_explicit_won", "ce_enabled": False, "policies": results}
    output = root / "dataset_report"
    save(output / "comparison.json", payload)
    text = f"# {dataset} 八策略结果（跳过 CE）\n\n主测评集：{len(primary)} 题；原实验配置摘要：`{digest(config)}`。\n\n"
    text += "| Policy | Cohort | N | Success % | Delta pp |\n| --- | --- | ---: | ---: | ---: |\n"
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy", "cohort", "n", "success_percent", "paired_delta_pp"])
        for method in registered:
            values = {"primary": results[method], **results[method].get("cohort_results", {})}
            for cohort, stats in values.items():
                writer.writerow([method, cohort, stats["n"], 100 * stats["success"], 100 * stats["paired_delta"]])
                text += f"| {method} | {cohort} | {stats['n']} | {100*stats['success']:.2f} | {100*stats['paired_delta']:.2f} |\n"
    (output / "comparison.md").write_text(text)
    print(f"[结果] {output / 'comparison.md'}", flush=True)
    return payload


def resume(root, dataset="alfworld", *, stage="all", backup_dir=None, plan_only=False):
    root = Path(root).resolve()
    if plan_only:
        config = registered_config(root, dataset)
        plan = {"dataset": dataset, "output_dir": str(root), "config_digest": digest(config),
                "stages": list(stages(config)) if stage == "all" else [stage],
                "preserved_datasets": sorted(set(config["experiments"]) - {dataset}),
                "backup_before_execution": True, "model_or_service_calls": 0}
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return plan
    with experiment_lock(root):
        config = registered_config(root, dataset)
        original_manifest = (root / "benchmark_manifest.json").read_bytes()
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        backup = backup_results(root, dataset, config, backup_dir)
        from .interactive_worker import verify_inputs
        verify_inputs(config, dataset)
        selected = stages(config) if stage == "all" else (stage,)
        state_path = root / "dataset_resume" / dataset / "progress.json"
        state = {"dataset": dataset, "config_digest": digest(config), "backup_dir": str(backup),
                 "stages": list(selected), "completed_stages": [], "status": "running"}
        print(f"[续跑] 仅执行 {dataset}；沿用现有 manifest、记忆库、缓存和提交进度。", flush=True)
        try:
            for index, current in enumerate(selected):
                state.update(stage=current, status="running")
                save(state_path, state)
                show_progress(f"{dataset} 总阶段", index, len(selected), STAGE_NAMES[current])
                if current == "report":
                    verify_inputs(config, dataset)
                    dataset_report(config, dataset)
                else:
                    command = [sys.executable, "-u", "-X", "faulthandler", "-m",
                        "causal_memory_control.interactive_worker", "--config", str(root / "benchmark_manifest.json"),
                        "--dataset", dataset, "--stage", current]
                    run_logged(command, root / dataset / "logs" / (current + ".log"), cwd=Path(config["repo"]))
                if (root / "benchmark_manifest.json").read_bytes() != original_manifest:
                    raise ValueError("registered manifest changed during dataset resume")
                state["completed_stages"].append(current)
                save(state_path, state)
            state["status"] = "completed"
            save(state_path, state)
            show_progress(f"{dataset} 总阶段", len(selected), len(selected), "本次命令已完成")
        except (Exception, KeyboardInterrupt, SystemExit) as exc:
            state.update(status="failed", error=str(exc) or type(exc).__name__)
            save(state_path, state)
            raise
        return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, required=True, help="Existing v10 result root")
    parser.add_argument("--dataset", choices=("alfworld",), default="alfworld")
    parser.add_argument("--stage", choices=("all", "support", "collect", "labels_team_u", "train_reward",
                                          "train_team_u", "evaluate", "report", "smoke"), default="all")
    parser.add_argument("--backup-dir", type=Path, help="Optional new independent backup destination")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    return resume(args.output_dir, args.dataset, stage=args.stage, backup_dir=args.backup_dir, plan_only=args.plan_only)


if __name__ == "__main__":
    main()
