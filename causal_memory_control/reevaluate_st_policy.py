"""Reevaluate only an existing Semantic ST model with an optional runtime rule.

No support collection, diagnostic collection, training or dev calibration.
Original result directories are read-only inputs. PDDL reuses measured actions;
QA reuses the previous ST prompt cache and the original held-out baseline.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import sqlite3
import statistics
import shutil
from types import SimpleNamespace

from .interactive_data import digest, locked, save, sha
from .interactive_progress import run_logged, show_progress
from .st_policy import POSITIVE_POLICY, RULES, positive_decisions, validate_st_checkpoint


SCHEMA = "semantic-st-policy-reevaluation-v1"
DEFAULT_RESULTS = Path("causal_memory_control/results")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def indexed(values):
    result = {str(row["task_id"]): row for row in values}
    if not result or len(result) != len(values):
        raise ValueError("missing or duplicate evaluated task IDs")
    return result


def selection(old, rule):
    decisions = positive_decisions(old["decisions"]) if rule == "positive_unlimited" else old["decisions"]
    return sorted(d["memory_id"] for d in decisions if d["action"].upper() == "DROP"), decisions


def pin(identity, path):
    path = Path(path).resolve()
    identity[str(path)] = sha(path)


def separate_output(output, source):
    output, source = Path(output).resolve(), Path(source).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError(f"use a separate, non-nested output directory, away from {source}")


def prepare_qa(suite, name, identity):
    from .source_run import import_source_dataset
    from .task_state import validate_task_state_policy

    previous = suite.qa_root / name
    old, baseline = previous / "semantic_st", previous / "native_gmemory"
    reevaluation = suite.qa_root / "reevaluation_manifest.json"
    trained = Path(read(reevaluation)["trained_root"]) / "csm_st_manifest.json"
    source = Path(read(trained)["source_root"]).resolve()
    adapter, args = import_source_dataset(SimpleNamespace(source_root=source, shared_argv=[]), name)
    old_manifest = read(old / "run_manifest.json")
    validate_task_state_policy(old_manifest, old)
    previous_run = old_manifest["run"]
    old_rows, baseline_rows = indexed(rows(old / "memory_gate.jsonl")), indexed(rows(baseline / "memory_gate.jsonl"))
    expected = {str(t) for t in previous_run["evaluation_ids"]}
    if set(old_rows) != expected or set(baseline_rows) != expected:
        raise ValueError(f"{name}: original ST/baseline evaluation is incomplete")
    snapshot_path = args.memory_dir / "causal_snapshot_manifest.json"
    snapshot = read(snapshot_path)
    ids = snapshot["evaluation_ids"][args.training_claims:args.evaluation_count]
    if [str(v) for v in ids] != [str(v) for v in previous_run["evaluation_ids"]]:
        raise ValueError(f"{name}: source held-out partition differs from the previous ST evaluation")
    args.checkpoint = Path(previous_run["checkpoint"]).resolve()
    validate_st_checkpoint(args.checkpoint)
    if sha(args.checkpoint) != previous_run["checkpoint_sha256"]:
        raise ValueError(f"{name}: previous ST checkpoint changed")
    # Preserve the actual retest endpoint/seed, which may differ from the older
    # full-benchmark configuration. No test-time hyperparameter search.
    for key in ("endpoint", "model", "temperature", "node_num", "graph_type", "embedding_model",
                "successful_topk", "failed_topk", "insights_topk", "threshold"):
        setattr(args, key, previous_run[key])
    args.gate_evaluation_seed = previous_run["seed"]
    args.output_dir = suite.output_dir / name
    args.semantic_cache = args.output_dir / "semantic.sqlite"
    if suite.st_deletion_rule == "positive_unlimited":
        args.kappa, args.delta, args.max_drops = 0.0, 0.0, -1
    else:
        args.kappa, args.delta, args.max_drops = (previous_run[k] for k in ("kappa", "delta", "max_drops"))
    adapter.validate_checkpoint(read(args.checkpoint), args, snapshot)
    cache = old / "llm_cache.sqlite"
    if not cache.is_file() or any(Path(str(cache) + suffix).exists() for suffix in ("-wal", "-journal")):
        raise ValueError(f"a completed, closed ST answer cache is required: {cache}")
    planned = {}
    for task_id, row in old_rows.items():
        base = baseline_rows[task_id]
        if row["retrieval_fingerprint"] != base["retrieval_fingerprint"]:
            raise ValueError(f"{name}: previous ST retrieval differs from baseline: {task_id}")
        drops, _ = selection(row, suite.st_deletion_rule)
        planned[task_id] = drops
    for path in (reevaluation, trained, source / "benchmark_manifest.json", args.checkpoint,
                 snapshot_path, old / "run_manifest.json", baseline / "run_manifest.json",
                 old / "memory_gate.jsonl", baseline / "memory_gate.jsonl", cache):
        pin(identity, path)
    for path in sorted(args.memory_dir.rglob("*")):
        if path.is_file() and not path.name.endswith(("-wal", "-shm")):
            pin(identity, path)
    for source_path in (suite.qa_root, source, args.checkpoint.parent, args.memory_dir):
        separate_output(suite.output_dir, source_path)
    plan = {"test_tasks": len(expected), "checkpoint": str(args.checkpoint),
        "source_root": str(source), "previous_evaluation": str(old), "prompt_cache": str(cache),
        "seed": args.gate_evaluation_seed, "endpoint": args.endpoint, "model": args.model,
        "changed_drop_tasks": sum(planned[t] != sorted(d["memory_id"] for d in old_rows[t]["decisions"]
            if d["action"].upper() == "DROP") for t in planned),
        "maximum_task_evaluations": len(expected), "new_llm_calls": "cache misses only",
        "planned_drop_ids": planned}
    return {"adapter": adapter, "args": args, "old": old, "baseline": baseline,
            "source_memory": args.memory_dir,
            "cache": cache, "plan": plan}


def prepare_pddl(suite, identity):
    from .interactive_runtime import ActionStore
    from .interactive_empty_memory import training_artifacts
    from .interactive_worker import snapshot
    from .run_interactive_gate_benchmark import dataset_jobs

    root = suite.interactive_root
    config = read(root / "benchmark_manifest.json")
    if Path(config["output_dir"]).resolve() != root:
        raise ValueError("PDDL root does not match its registered manifest")
    separate_output(suite.output_dir, root)
    pin(identity, root / "benchmark_manifest.json")
    jobs, seen = [], set()
    for label, expected in dataset_jobs(config, "pddl"):
        directory = Path(expected["output_dir"])
        path = directory / "benchmark_manifest.json"
        child = read(path)
        if child != expected:
            raise ValueError(f"PDDL fold configuration changed: {path}")
        memory = snapshot(child, "pddl")
        source = directory / "pddl"
        old_identity = read(source / "evaluation/actions/identity.json")
        checkpoints, skipped = training_artifacts(child, "pddl")
        expected_identity = {"config_digest": digest(child), "snapshot_sha256": sha(source / "snapshot.json"),
                             "checkpoints_sha256": checkpoints, "phase": "test"}
        if skipped:
            expected_identity["untrained_methods"] = skipped
        if old_identity != expected_identity:
            raise ValueError("PDDL measured actions no longer match their registered inputs")
        store = ActionStore(source / "evaluation/actions", old_identity)
        checkpoint = source / "semantic_st/checkpoint.json"
        if checkpoint.exists():
            validate_st_checkpoint(checkpoint)
            pin(identity, checkpoint)
        else:
            pin(identity, source / "semantic_st/training_status.json")
        for path in (path, source / "snapshot.json", source / "evaluation/actions/identity.json"):
            pin(identity, path)
        work = []
        for task_id in child["experiments"]["pddl"]["split"]["test"]:
            if task_id in seen:
                raise ValueError("PDDL task appears in more than one outer test fold")
            seen.add(task_id)
            part = source / "evaluation/policy_parts" / (digest(task_id) + ".json")
            pin(identity, part)
            records = read(part)
            by_policy = {r["policy"]: r for r in records}
            old = by_policy["semantic_st"]
            if old["config_digest"] != digest(child):
                raise ValueError("previous PDDL ST evaluation belongs to a different configuration")
            index = source / "evaluation/actions/baselines" / (digest([task_id, child["evaluation_seed"]]) + ".json")
            baseline = store.get(task_id, child["evaluation_seed"], read(index), [])
            if baseline is None:
                raise ValueError("missing original held-out baseline action")
            if (old["retrieval_fingerprint"] != baseline["retrieval_fingerprint"]
                    or old["initial_fingerprint"] != baseline["initial_fingerprint"]
                    or old["seed"] != baseline["seed"]
                    or old["retrieved_count"] != len(baseline["candidates"])
                    or [d["memory_id"] for d in old["decisions"]] !=
                       [c["memory_id"] for c in baseline["candidates"]]):
                raise ValueError("previous PDDL decisions do not match the measured candidate list")
            if not checkpoint.exists() and baseline["candidates"]:
                raise ValueError("an untrained fold cannot apply ST to nonempty test candidates")
            previous_branch = store.get(task_id, baseline["seed"], baseline, old["drop_ids"])
            if (previous_branch is None or old["outcome"] != previous_branch["outcome"]
                    or old["action_keys"] != [previous_branch["action_key"]]):
                raise ValueError("previous PDDL policy outcome differs from its measured action")
            drops, decisions = selection(old, suite.st_deletion_rule)
            measured = store.get(task_id, baseline["seed"], baseline, drops)
            for key in {baseline["action_key"], *(old["action_keys"]),
                        *([measured["action_key"]] if measured else [])}:
                pin(identity, store.root / "parts" / (key + ".json"))
            work.append({"task_id": task_id, "baseline": baseline, "old": old,
                "drops": drops, "decisions": decisions, "measured": measured})
        jobs.append({"label": label, "config": child, "source": source, "memory": memory, "work": work})
    plan = {"test_tasks": len(seen), "folds": len(jobs),
        "reused_measured_actions": sum(w["measured"] is not None for j in jobs for w in j["work"]),
        "maximum_new_episodes": sum(w["measured"] is None for j in jobs for w in j["work"]),
        "changed_drop_tasks": sum(w["drops"] != sorted(w["old"]["drop_ids"]) for j in jobs for w in j["work"])}
    return {"jobs": jobs, "plan": plan}


def prepare(suite):
    identity = {}
    for path in (Path(__file__), Path(__file__).with_name("st_policy.py"),
                 Path(__file__).with_name("st_policy_evaluate.py")):
        pin(identity, path)
    prepared = {}
    for name in suite.datasets:
        prepared[name] = prepare_pddl(suite, identity) if name == "pddl" else prepare_qa(suite, name, identity)
    policy = POSITIVE_POLICY if suite.st_deletion_rule == "positive_unlimited" else {
        "rule": "calibrated", "settings": "unchanged saved kappa/delta/budget"}
    manifest = {"schema": SCHEMA, "policy": policy,
        "datasets": {name: item["plan"] for name, item in prepared.items()},
        "source_sha256": identity, "new_training_runs": 0, "new_diagnostic_tasks": 0,
        "new_calibration_runs": 0, "original_results_overwritten": False}
    return prepared, manifest


def run_qa(suite, name, item):
    from .compare_gate_runs import compare
    from .jsonl import read_jsonl

    args, adapter = item["args"], item["adapter"]
    # Chroma can write bookkeeping even with logically read-only GMemory.
    # Keep those writes in a private copy, away from every previous result.
    args.memory_dir = private_memory(item["source_memory"], suite.output_dir / name / "frozen_memory/g-memory")
    output = suite.output_dir / name / "semantic_st"
    host = adapter.evaluation_command(args, mode="learned", output=output,
                                     resume=(output / "run_manifest.json").is_file())
    host[host.index("--cache-seed-from") + 1] = str(item["cache"])
    command = [*host[:3], "causal_memory_control.st_policy_evaluate", "--dataset", name,
               "--st-deletion-rule", suite.st_deletion_rule,
               "--st-reference-log", str(item["old"] / "memory_gate.jsonl"), *host[4:]]
    if not adapter.evaluation_complete(args, output, mode="learned", command=host):
        run_logged(command, suite.output_dir / "logs" / (name + ".log"), cwd=Path.cwd())
    if not adapter.evaluation_complete(args, output, mode="learned", command=host):
        raise ValueError(f"{name}: ST evaluation exited without complete results")
    actual = indexed(read_jsonl(output / "memory_gate.jsonl"))
    for task_id, drops in item["plan"]["planned_drop_ids"].items():
        observed = sorted(d["memory_id"] for d in actual[task_id]["decisions"] if d["action"].upper() == "DROP")
        if drops != observed:
            raise ValueError(f"{name}: runtime ST selection differs from the saved-score plan: {task_id}")
    comparison = compare(item["baseline"] / "memory_gate.jsonl", output / "memory_gate.jsonl",
                         bootstrap_samples=2000, seed=args.gate_evaluation_seed)
    if not comparison["paired_control_valid"] or not comparison["policy_valid"]:
        raise ValueError(f"{name}: paired baseline/retrieval audit failed")
    save(output / "comparison.json", comparison)
    old_comparison = compare(item["baseline"] / "memory_gate.jsonl", item["old"] / "memory_gate.jsonl",
                             bootstrap_samples=2000, seed=args.gate_evaluation_seed)
    return {"n": comparison["paired_tasks"], "baseline": comparison["baseline_mean_reward"],
        "old_st": old_comparison["gate_mean_reward"], "semantic_st": comparison["gate_mean_reward"],
        "metric": "Accuracy" if name == "fever" else "Answer F1", "comparison": comparison,
        "old_answer_em": old_comparison.get("gate_answer_em"), "answer_em": comparison.get("gate_answer_em")}


def copy_cache(source, target):
    """Copy a closed source cache through SQLite, never open it for writes."""
    if target.exists() or not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    # Publish atomically so an interrupted backup cannot become a usable cache.
    temp = target.with_suffix(".sqlite.tmp")
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as src, sqlite3.connect(temp) as dst:
        src.backup(dst)
    temp.replace(target)


def private_memory(source, target):
    from .interactive_runtime import clone_memory

    if target.exists():
        return target
    building = target.with_name("g-memory.building")
    if building.exists():
        shutil.rmtree(building)
    clone_memory(source, building)
    building.replace(target)
    return target


def run_pddl(suite, item):
    from .interactive_runtime import ActionStore, NativeRunner, execute
    from .interactive_worker import verify_inputs

    records = []
    for job in item["jobs"]:
        config, source = job["config"], job["source"]
        output = suite.output_dir / job["label"]
        store = ActionStore(output / "actions", {"schema": SCHEMA, "rule": suite.st_deletion_rule,
            "original_action_identity": read(source / "evaluation/actions/identity.json")})
        tasks = {t["id"]: t for t in config["experiments"]["pddl"]["tasks"]}
        runner = None
        try:
            for number, work in enumerate(job["work"]):
                baseline, drops = work["baseline"], work["drops"]
                part = output / "policy_parts" / (digest(work["task_id"]) + ".json")
                branch = work["measured"] or store.get(work["task_id"], baseline["seed"], baseline, drops)
                reused = branch is not None
                if branch is None:
                    if runner is None:
                        verify_inputs(config, "pddl")
                        copy_cache(source / "responses.sqlite", output / "responses.sqlite")
                        runner = NativeRunner(config, output, job["memory"])
                    branch = execute(runner, store, tasks[work["task_id"]], baseline["seed"], drops, baseline)
                record = {**work["old"], "policy": "semantic_st", "runtime_deletion_rule": suite.st_deletion_rule,
                    "drop_ids": drops, "dropped_count": len(drops), "decisions": work["decisions"],
                    "outcome": branch["outcome"], "action_keys": [branch["action_key"]],
                    "measurement_reused": reused, "source_fold": job["label"],
                    "baseline_outcome": baseline["outcome"], "old_st_outcome": work["old"]["outcome"]}
                save(part, record)
                records.append(record)
                show_progress("ST 规则重评", number + 1, len(job["work"]),
                              job["label"] + (" | 复用实测结果" if reused else " | 新删除组合"))
        finally:
            if runner is not None:
                runner.close()
    target = suite.output_dir / "pddl"
    target.mkdir(parents=True, exist_ok=True)
    (target / "policies.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    mean = lambda field, metric: statistics.mean(r[field][metric] for r in records)
    summary = {"n": len(records), "metric": "Success", "baseline": mean("baseline_outcome", "success"),
        "old_st": mean("old_st_outcome", "success"), "semantic_st": mean("outcome", "success"),
        "baseline_progress": mean("baseline_outcome", "progress"),
        "old_st_progress": mean("old_st_outcome", "progress"), "mean_progress": mean("outcome", "progress"),
        "mean_drop_count": statistics.mean(r["dropped_count"] for r in records),
        "empty_candidate_tasks": sum(r["retrieved_count"] == 0 for r in records),
        "untrained_empty_only_tasks": sum(r.get("gate_training_status") ==
            "skipped_no_candidate_supervision" for r in records),
        "improved_tasks": [r["task_id"] for r in records if r["outcome"]["success"] > r["baseline_outcome"]["success"]],
        "harmed_tasks": [r["task_id"] for r in records if r["outcome"]["success"] < r["baseline_outcome"]["success"]]}
    save(target / "summary.json", summary)
    return summary


def write_report(root, results):
    save(root / "summary.json", {"schema": SCHEMA, "datasets": results})
    table = []
    for name, result in results.items():
        metrics = [(result["metric"], "baseline", "old_st", "semantic_st")]
        if name == "hotpotqa":
            metrics.append(("Answer EM", None, "old_answer_em", "answer_em"))
        if name == "pddl":
            metrics.append(("Progress", "baseline_progress", "old_st_progress", "mean_progress"))
        for metric, baseline, old, new in metrics:
            table.append({"Dataset": name, "Metric": metric, "N": result["n"],
                "Always Use (%)": 100 * result[baseline] if baseline else None,
                "Original ST (%)": 100 * result[old], "Selected ST rule (%)": 100 * result[new]})
    with (root / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    columns = list(table[0])
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in table:
        lines.append("| " + " | ".join("—" if row[c] is None else f"{row[c]:.2f}"
                     if isinstance(row[c], float) else str(row[c]) for c in columns) + " |")
    (root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--datasets", default="fever,hotpotqa,pddl")
    parser.add_argument("--qa-root", type=Path,
        default=DEFAULT_RESULTS / "all_trained_gates_with_old_st_retest_v2")
    parser.add_argument("--interactive-root", type=Path,
        default=DEFAULT_RESULTS / "pddl_alfworld_gate_expanded_v10_formatwarn_no_ce")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--st-deletion-rule", choices=RULES, default="calibrated",
                        help="calibrated preserves the old rule; positive_unlimited keeps only U > 0")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    args.datasets = list(dict.fromkeys(v.strip() for v in args.datasets.split(",") if v.strip()))
    if not args.datasets or set(args.datasets) - {"fever", "hotpotqa", "pddl"}:
        parser.error("--datasets must select fever,hotpotqa,pddl; ALFWorld collection is never started")
    for name in ("qa_root", "interactive_root", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv=None):
    suite = parse_args(argv)
    prepared, manifest = prepare(suite)
    display = {key: value for key, value in manifest.items() if key != "source_sha256"}
    display["datasets"] = {name: {k: v for k, v in item.items() if k != "planned_drop_ids"}
                           for name, item in manifest["datasets"].items()}
    print(json.dumps({"plan_only": suite.plan_only, **display,
                      "pinned_source_files": len(manifest["source_sha256"])}, ensure_ascii=False, indent=2), flush=True)
    if suite.plan_only:
        return manifest
    if suite.output_dir.exists() and any(suite.output_dir.iterdir()) and not (suite.output_dir / "manifest.json").exists():
        raise ValueError("refusing a nonempty output directory without this reevaluation's manifest")
    from .resume_interactive_dataset import experiment_lock
    suite.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(experiment_lock(suite.output_dir))
        if "pddl" in suite.datasets:
            # The existing continuation command holds this same lock while
            # ALFWorld is collecting. Never signal an unrelated process.
            stack.enter_context(experiment_lock(suite.interactive_root))
        locked(suite.output_dir / "manifest.json", manifest)
        state, results = {"status": "running", "completed_datasets": []}, {}
        try:
            for name, item in prepared.items():
                state["current_dataset"] = name
                save(suite.output_dir / "progress.json", state)
                results[name] = run_pddl(suite, item) if name == "pddl" else run_qa(suite, name, item)
                state["completed_datasets"].append(name)
                write_report(suite.output_dir, results)
            state["status"] = "completed"
        except BaseException as exc:
            state.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                         error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            save(suite.output_dir / "progress.json", state)
    print(f"[完成] ST 规则重评结果：{suite.output_dir / 'results.md'}", flush=True)
    return suite.output_dir / "summary.json"


if __name__ == "__main__":
    main()
