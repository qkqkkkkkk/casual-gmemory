"""Train/evaluate three gates using only a completed first collection group.

Optionally append an Always Global Drop (all retrieved memory hidden) control,
including to a previously completed four-method run without retraining it.

Import the original full-run configuration and frozen memory. An explicit
--diagnostic-tasks subset can collect missing data with --collect-missing.
Never invokes seed1000, HotpotQA (unless selected), Jev or ST. Original
source artifacts are read-only; all new checkpoints/results use a separate root.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

from tqdm import tqdm

from . import run_full_gate_benchmark as full
from . import run_hotpotqa_gate_comparison as base
from . import run_hotpotqa_gate_matrix as matrix
from .compare_gate_runs import compare, file_sha256
from .planned_drop import build_plan
from .source_run import import_source_dataset


SCHEMA = "seed0-only-gate-matrix-v1"
GLOBAL_DROP = "always_global_drop"
NOTE = ("One completed collection group only; repeats inside that group remain distinct. "
        "Original frozen memory and held-out test boundary are preserved. "
        "Optional diagnostic subset changes fitting data only, not test tasks. "
        "Calibrate learned DROP on diagnostic dev tasks; max one DROP, KEEP ALL fallback allowed. "
        "Optional Always Global Drop hides ALL retrieved trajectories and insights from all agents; "
        "task context and fixed few-shots remain available. "
        "FEVER reports binary Accuracy, not evidence-aware FEVER Score; HotpotQA reports Answer F1/EM.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path,
        default=Path("causal_memory_control/results/full_gate_benchmark_v1"))
    parser.add_argument("--output-dir", type=Path,
        default=Path("causal_memory_control/results/seed0_gate_matrix_v1"))
    parser.add_argument("--datasets", default="fever")
    parser.add_argument("--diagnostic-tasks", type=int,
        help="Use the first N tasks of the original randomized diagnostic partition; keep the original test offset/count")
    parser.add_argument("--collection-repeats", type=int,
        help="Repeat count for a new diagnostic subset, e.g. 2; requires --diagnostic-tasks")
    parser.add_argument("--collect-missing", action="store_true",
        help="Explicitly collect/resume only the requested diagnostic subset; never rebuild memory or run seed1000")
    parser.add_argument("--collection-endpoints",
        help="Comma-separated Ollama replicas for disjoint task shards; first must match the recorded endpoint; use a new output directory")
    cache_options = parser.add_mutually_exclusive_group()
    cache_options.add_argument("--collection-cache-seed-from", type=Path,
        help="Optional stopped serial collector cache to reuse when switching to a new parallel output; its parent must contain the original run_manifest.json")
    cache_options.add_argument("--no-collection-cache-seed", action="store_true",
        help="Do not import legacy collector responses when their model weight version is uncertain")
    parser.add_argument("--include-always-global-drop", action="store_true",
        help="Append a no-retrieved-memory control; safely extends existing four-method output without retraining")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    args.datasets = list(dict.fromkeys(v.strip() for v in args.datasets.split(",") if v.strip()))
    if not args.datasets or set(args.datasets) - {"fever", "hotpotqa"}:
        parser.error("--datasets must select fever,hotpotqa")
    if args.diagnostic_tasks is not None and args.diagnostic_tasks < 5:
        parser.error("--diagnostic-tasks must be at least 5 for task-group fit/dev calibration")
    if args.collection_repeats is not None and args.collection_repeats < 1:
        parser.error("--collection-repeats must be positive")
    if (args.collect_missing or args.collection_repeats is not None or args.collection_cache_seed_from is not None
            or args.no_collection_cache_seed) and args.diagnostic_tasks is None:
        parser.error("--collect-missing/--collection-repeats requires an explicit --diagnostic-tasks subset")
    if args.collection_endpoints:
        from .parallel_collection import parse_endpoints
        args.collection_endpoints = parse_endpoints(args.collection_endpoints)
        if args.diagnostic_tasks is None or len(args.collection_endpoints) > args.diagnostic_tasks:
            parser.error("--collection-endpoints requires --diagnostic-tasks >= the number of replicas")
    args.source_root, args.output_dir = args.source_root.resolve(), args.output_dir.resolve()
    if args.collection_cache_seed_from is not None:
        args.collection_cache_seed_from = args.collection_cache_seed_from.resolve()
    if (args.source_root == args.output_dir or args.source_root in args.output_dir.parents
            or args.output_dir in args.source_root.parents):
        parser.error("source/output must be separate, non-nested directories")
    if not (args.source_root / "benchmark_manifest.json").is_file():
        parser.error("--source-root must contain the original full-run benchmark_manifest.json")
    # The shared importer reads recorded data/model/split paths, never CLI defaults.
    args.shared_argv = []
    return args


def source_path(args):
    return args.diagnostic_results / f"seed{args.sample_seed_base}" / "branches.jsonl"


def diagnostic_count(args):
    # args.training_claims remains the ORIGINAL final-test offset. Never replace
    # it with a smaller training sample count or diagnostic tasks leak into test.
    return getattr(args, "diagnostic_task_limit", args.training_claims)


def diagnostic_ids(args):
    snapshot = base._load_json(args.memory_dir / "causal_snapshot_manifest.json")
    if snapshot is None:
        raise FileNotFoundError(f"existing frozen snapshot is required: {args.memory_dir}; memory is never rebuilt here")
    offset = getattr(args, "diagnostic_task_offset", 0)
    return snapshot["evaluation_ids"][offset:offset + diagnostic_count(args)]


def collection_workers(args):
    endpoints = getattr(args, "collection_endpoints", ())
    if not endpoints:
        return [args]
    from .parallel_collection import shard_ranges
    workers = []
    for index, (offset, count) in enumerate(shard_ranges(diagnostic_count(args), len(endpoints))):
        child = copy.copy(args)
        child.collection_endpoints = ()
        child.endpoint = endpoints[index]
        child.diagnostic_task_limit, child.diagnostic_task_offset = count, offset
        child.diagnostic_results = args.diagnostic_results / "workers" / f"worker_{index:03d}"
        workers.append(child)
    return workers


def training_sources(args):
    return [source_path(child) for child in collection_workers(args)]


def collection_complete(args):
    return all((base._load_json(source.parent / "collection_progress.json") or {}).get("status") == "completed"
               for source in training_sources(args))


def collection_command(adapter, args, original, *, verified_replica_cache=False):
    """Collect one matched subset group using the existing optimized collector."""
    output = source_path(args).parent
    command = [sys.executable, "-u", "-m",
        f"causal_diagnostic.{adapter.name}_oracle_recipient.run_experiment",
        "--test", str(args.data), "--memory-dir", str(args.memory_dir),
        "--claims", str(diagnostic_count(args)), "--output-dir", str(output),
        "--all-candidates", "--gate-training-only",
        "--isolated-recipients", f"solver_0,solver_{args.node_num - 1}"]
    if getattr(args, "diagnostic_task_offset", 0):
        command += ["--evaluation-offset", str(args.diagnostic_task_offset)]
    for name in ("endpoint", "model", "repeats", "sample_seed_base", "selection_seed",
            "temperature", "node_num", "graph_type", "successful_topk", "failed_topk",
            "insights_topk", "threshold", "embedding_model", "bootstrap_samples", "delta"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    manifest_path = output / "run_manifest.json"
    seed_cache = None
    if manifest_path.is_file():
        # Keep a resumed collection's original cache provenance, even when no
        # original full-run cache existed on its first invocation.
        manifest = base._load_json(manifest_path)
        if not manifest:
            raise ValueError(f"unreadable subset collection manifest: {manifest_path}")
        seed_cache = manifest["run"].get("cache_seed_from")
        command.append("--resume")
    elif not getattr(args, "disable_collection_cache_seed", False):
        override_cache = getattr(args, "collection_cache_seed_from", None)
        candidate_cache = override_cache or source_path(original).parent / "llm_cache.sqlite"
        source = candidate_cache.parent / "branches.jsonl"
        if override_cache is not None and not candidate_cache.is_file():
            raise FileNotFoundError(f"requested collection cache is missing: {candidate_cache}")
        if candidate_cache.is_file():
            manifest = base._load_json(source.parent / "run_manifest.json")
            snapshot = base._load_json(args.memory_dir / "causal_snapshot_manifest.json")
            if not manifest or not snapshot:
                raise ValueError("cannot seed subset cache without the original collection/snapshot manifests")
            base._require_fields("source cache context", manifest["design"], adapter.checkpoint_context(args, snapshot))
            base._require_fields("source cache run", manifest["run"], {"endpoint": original.endpoint if verified_replica_cache else args.endpoint,
                "sample_seed_base": args.sample_seed_base})
            offset = getattr(args, "diagnostic_task_offset", 0)
            if manifest["design"].get("evaluation_ids", [])[offset:offset + diagnostic_count(args)] != diagnostic_ids(args):
                raise ValueError("source collection cache belongs to a different diagnostic partition")
            seed_cache = str(candidate_cache)
    if seed_cache:
        command += ["--cache-seed-from", str(seed_cache)]
    return command


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_seed0(adapter, args):
    """Streaming preflight: require all tasks/ranks/repeats and matched branches.

    Do not repair or mutate source logs, accept partial collections, fabricate a
    second collection group, or rely only on a completed progress flag.
    """
    source = source_path(args)
    progress = base._load_json(source.parent / "collection_progress.json")
    manifest = base._load_json(source.parent / "run_manifest.json")
    snapshot = base._load_json(args.memory_dir / "causal_snapshot_manifest.json")
    if not source.is_file() or not manifest or not snapshot or not progress or progress.get("status") != "completed":
        raise FileNotFoundError(f"first collection is not completed: {source.parent}; finish seed0 before this run")
    design, run = manifest["design"], manifest["run"]
    if (stable_hash(design) != manifest["design_hash"] or stable_hash(run) != manifest["run_hash"]):
        raise ValueError("seed0 manifest hash is invalid")
    ids = diagnostic_ids(args)
    specs = [(kind, index) for kind, count in (
        ("trajectory", args.successful_topk), ("insight", args.insights_topk)) for index in range(count)]
    base._require_fields("seed0 collection", design, {
        "benchmark": adapter.benchmark, "source_md5": snapshot["source_md5"],
        "snapshot_design_hash": snapshot["design_hash"], "evaluation_ids": ids,
        "candidate_scope": "all_retrieved", "collection_scope": "gate_training",
        "candidate_specs": [{"kind": k, "index": i} for k, i in specs],
        **{key: getattr(args, key) for key in ("repeats", "model", "temperature", "node_num",
            "graph_type", "successful_topk", "failed_topk", "insights_topk", "threshold", "embedding_model", "selection_seed")},
    })
    base._require_fields("seed0 run", run, {"sample_seed_base": args.sample_seed_base, "endpoint": args.endpoint})
    if design.get("evaluation_offset", 0) != getattr(args, "diagnostic_task_offset", 0):
        raise ValueError("seed0 task shard offset changed")
    expected = {(int(t), k, i, repeat, condition)
        for t in ids for k, i in specs for repeat in range(args.repeats)
        for condition in ("use_all", "global_drop")}
    observed, contexts, candidates = set(), {}, {}
    with source.open(encoding="utf-8") as handle:
        for number, line in enumerate(tqdm(handle, desc=f"Validate {adapter.display_name} seed0", unit="branch"), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                candidate = row["candidate"]
                key = (int(row["task_id"]), candidate["kind"], int(candidate["index"]),
                       int(row["repeat_index"]), row["condition"])
                if key not in expected or key in observed:
                    raise ValueError(f"duplicate/unexpected branch {key}")
                observed.add(key)
                if (row["run_hash"] != manifest["run_hash"] or row["design_hash"] != manifest["design_hash"]
                        or row["sample_seed"] != args.sample_seed_base + key[3]):
                    raise ValueError("branch run/seed does not match its manifest")
                reward = float(row["outcome"][adapter.training_metric])
                if not math.isfinite(reward) or not 0 <= reward <= 1:
                    raise ValueError("invalid bounded training reward")
                peers = row["candidate_set"]
                if {(p["kind"], p["index"]) for p in peers} != set(specs) or len(peers) != len(specs):
                    raise ValueError("incomplete retrieved candidate set")
                if candidate not in peers:
                    raise ValueError("branch candidate differs from its logged candidate set")
                context = stable_hash({k: row[k] for k in ("task", "candidate_set", "retrieval_sizes", "exposure_count")})
                if contexts.setdefault(key[0], context) != context:
                    raise ValueError("task or retrieval differs across matched branches")
                slot = key[:3]
                if candidates.setdefault(slot, candidate["candidate_id"]) != candidate["candidate_id"]:
                    raise ValueError("candidate identity differs across matched branches")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid seed0 data at {source}:{number}: {exc}") from exc
    if observed != expected:
        raise ValueError(f"incomplete seed0 data: {len(observed)}/{len(expected)} branches; missing {list(expected-observed)[:5]}")
    return {"source": str(source), "source_sha256": file_sha256(source),
        "manifest_sha256": file_sha256(source.parent / "run_manifest.json"),
        "snapshot_sha256": file_sha256(args.memory_dir / "causal_snapshot_manifest.json"),
        "tasks": len(ids), "branches": len(observed), "repeats": args.repeats,
        "sample_seeds": [args.sample_seed_base + i for i in range(args.repeats)]}


def validate_sources(adapter, args):
    if not getattr(args, "collection_endpoints", None):
        return validate_seed0(adapter, args)
    from .parallel_collection import validate_manifest, validate_union
    workers = collection_workers(args)
    reports = [validate_seed0(adapter, child) for child in workers]
    validate_union([diagnostic_ids(child) for child in workers], diagnostic_ids(args))
    replica_manifest = args.diagnostic_results / "parallel_collection_manifest.json"
    validate_manifest(adapter, args)
    return {"sources": reports, "tasks": sum(r["tasks"] for r in reports),
        "branches": sum(r["branches"] for r in reports),
        "replica_manifest_sha256": file_sha256(replica_manifest)}


def train_command(adapter, args):
    return [*base._train_command(args, metric=adapter.training_metric, sources=training_sources(args)), "--stream-branches"]


def checkpoint_ready(adapter, args):
    if not base._checkpoint_complete(args, benchmark=adapter.benchmark,
            metric=adapter.training_metric, sources=training_sources(args),
            training_task_ids=diagnostic_ids(args)):
        return False
    payload = base._load_json(args.checkpoint)
    metadata = payload["training_metadata"]
    base._require_fields("seed0 fit/dev split", metadata, {"calibration_seed": 71})
    fit = set(metadata["fit_task_ids"])
    dev = set(metadata["policy_calibration"]["calibration_task_ids"])
    if not fit or not dev or fit & dev or fit | dev != {str(t) for t in metadata["training_task_ids"]}:
        raise ValueError("invalid task-disjoint seed0 fit/dev split")
    adapter.validate_checkpoint(payload, args, base._load_json(args.memory_dir / "causal_snapshot_manifest.json"))
    return True


def evaluation_command(adapter, args, output, baseline):
    command = adapter.evaluation_command(args, mode="learned", output=output,
        resume=(output / "run_manifest.json").is_file())
    command[command.index("--cache-seed-from") + 1] = str(baseline / "llm_cache.sqlite")
    return command


def audit(adapter, args, output, baseline):
    result = compare(baseline / "memory_gate.jsonl", output / "memory_gate.jsonl",
        bootstrap_samples=args.bootstrap_samples, seed=args.gate_evaluation_seed)
    full._save(output / "comparison.json", result)
    if result["paired_tasks"] != args.final_claims or not result["paired_control_valid"] or not result["policy_valid"]:
        raise ValueError(f"invalid paired comparison; inspect {output / 'comparison.json'}")
    return result


def global_drop_plan(args):
    return args.output_dir / "plans" / f"{GLOBAL_DROP}.json"


def prepare_global_drop(args, baseline):
    if args.failed_topk != 0:
        raise ValueError("Always Global Drop no-memory control requires failed-topk=0; failed trajectories are not gate candidates")
    full.save_plan(global_drop_plan(args), build_plan(baseline / "memory_gate.jsonl", policy="global_all"))


def report(root, prepared, baselines, *, include_always_global_drop=False):
    tables, details, protocols = {}, {}, {}
    lines = ["# Seed0-only gate results", "", NOTE, ""]
    for name, (adapter, args) in prepared.items():
        protocols[name] = {"diagnostic_tasks": diagnostic_count(args), "repeats": args.repeats,
            "original_diagnostic_partition_tasks": args.training_claims,
            "final_test_offset": args.training_claims, "final_test_tasks": args.final_claims}
        if getattr(args, "collection_endpoints", None):
            protocols[name]["collection_replicas"] = list(args.collection_endpoints)
            protocols[name]["training_sources"] = [str(p) for p in training_sources(args)]
        baseline = base._load_json(baselines[name] / "summary.json")
        rows = [{"method": "native_gmemory", "tasks": args.final_claims,
            "keep_rate": 1.0, "tasks_with_drop": 0, "drop_improved_tasks": 0, "drop_harmed_tasks": 0}]
        for metric in adapter.metrics:
            rows[0].update({metric.name: baseline[metric.summary_field], metric.delta_field: 0.0, metric.interval_field: [0.0, 0.0]})
        arms = (*matrix.DEFAULT_ARMS, *((GLOBAL_DROP,) if include_always_global_drop else ()))
        for arm in arms:
            result = base._load_json(args.output_dir / arm / "comparison.json")
            row = {"method": arm, "tasks": result["paired_tasks"], "keep_rate": result["gate_mean_keep_rate"],
                **{k: result[k] for k in ("tasks_with_drop", "drop_improved_tasks", "drop_harmed_tasks")}}
            for metric in adapter.metrics:
                row.update({metric.name: result[metric.gate_field], metric.delta_field: result[metric.delta_field],
                            metric.interval_field: result[metric.interval_field]})
            rows.append(row)
            details.setdefault(name, {})[arm] = result
        tables[name] = rows
        fields = list(rows[0])
        with (args.output_dir / "matrix_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        full._save(args.output_dir / "matrix_results.json", {"dataset": name, "results": rows, "notes": NOTE,
            "data_protocol": protocols[name]})
        columns = ["method", *(m.label for m in adapter.metrics), "primary delta (pp)", "95% CI (pp)", "keep rate", "DROP tasks", "improved", "harmed"]
        table = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        primary = adapter.metrics[0]
        for row in rows:
            ci = row[primary.interval_field]
            cells = [row["method"], *(f"{100*row[m.name]:.2f}%" for m in adapter.metrics),
                f"{100*row[primary.delta_field]:+.2f}", f"[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]",
                f"{100*row['keep_rate']:.2f}%", *(str(row[k]) for k in ("tasks_with_drop", "drop_improved_tasks", "drop_harmed_tasks"))]
            table.append("| " + " | ".join(cells) + " |")
        (args.output_dir / "matrix_results.md").write_text("\n".join(table) + "\n", encoding="utf-8")
        lines += [f"## {adapter.display_name}", "",
            f"Diagnostic: {diagnostic_count(args)} tasks x {args.repeats} repeats; "
            f"final test: {args.final_claims} tasks (original offset {args.training_claims}).", "", *table, ""]
    full._save(root / "seed0_results.json", {"schema": SCHEMA, "notes": NOTE, "results": tables,
        "details": details, "data_protocol": protocols})
    (root / "seed0_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    suite = parse_args(argv)
    if suite.collection_endpoints and not suite.plan_only:
        from .parallel_collection import exclusive_run
        with exclusive_run(suite.output_dir):
            return _main(suite)
    return _main(suite)


def _main(suite):
    prepared, source_args = {}, {}
    for name in suite.datasets:
        adapter, original = import_source_dataset(suite, name)
        source_args[name] = copy.copy(original)
        args = copy.copy(original)
        args.output_dir = suite.output_dir / name
        args.checkpoint = args.output_dir / "unused.json"
        args.semantic_cache = args.output_dir / "semantic_cache.sqlite"
        args.kappa = "auto"
        if suite.diagnostic_tasks is not None:
            if suite.diagnostic_tasks > original.training_claims:
                raise ValueError(f"--diagnostic-tasks exceeds the original {name} diagnostic partition ({original.training_claims})")
            args.diagnostic_task_limit = suite.diagnostic_tasks
            args.diagnostic_results = suite.output_dir / "artifacts" / name / "diagnostic"
            if suite.collection_repeats is not None:
                args.repeats = suite.collection_repeats
        if suite.collection_endpoints:
            if suite.collection_endpoints[0].rstrip("/") != original.endpoint.rstrip("/"):
                raise ValueError("first --collection-endpoints entry must match the original endpoint; final evaluation stays there")
            args.collection_endpoints = suite.collection_endpoints
        if suite.collection_cache_seed_from is not None:
            args.collection_cache_seed_from = suite.collection_cache_seed_from
        if suite.no_collection_cache_seed:
            args.disable_collection_cache_seed = True
        if suite.include_always_global_drop and args.failed_topk != 0:
            raise ValueError("Always Global Drop requires failed-topk=0 to guarantee no retrieved memory remains")
        prepared[name] = adapter, args
    plans = {name: {**adapter.plan(args), "training_sources": [str(p) for p in training_sources(args)],
        "new_diagnostic_branches": 0, "collection_groups": 1,
        "sample_seeds": list(range(args.sample_seed_base, args.sample_seed_base + args.repeats)),
        "arms": list(matrix.DEFAULT_ARMS),
        "controls": [GLOBAL_DROP] if suite.include_always_global_drop else [],
        "maximum_test_team_inferences": (5 if suite.include_always_global_drop else 4) * args.final_claims}
        for name, (adapter, args) in prepared.items()}
    for plan in plans.values():
        plan.pop("diagnostic_branches_two_seeds", None)
    for name, (_, args) in prepared.items():
        if suite.diagnostic_tasks is not None:
            count = diagnostic_count(args)
            branches = count * (args.successful_topk + args.insights_topk) * args.repeats * 2
            plans[name].update(gate_training_questions=count,
                original_diagnostic_partition_questions=args.training_claims,
                unused_diagnostic_questions=args.training_claims - count,
                diagnostic_selection="prefix_of_original_randomized_diagnostic_partition_not_outcome_selected",
                diagnostic_branches_per_seed=branches,
                new_diagnostic_branches=branches if suite.collect_missing else 0,
                maximum_executed_diagnostic_schedules=count * args.repeats * (1 + args.successful_topk + args.insights_topk),
                collect_missing=suite.collect_missing,
                final_test_offset=args.training_claims, final_test_questions=args.final_claims)
        if suite.collection_endpoints:
            plans[name]["parallel_workers"] = [{"endpoint": child.endpoint,
                "offset": child.diagnostic_task_offset, "tasks": diagnostic_count(child),
                "source": str(source_path(child))} for child in collection_workers(args)]
    print(json.dumps({"schema": SCHEMA, "datasets": plans, "notes": NOTE}, ensure_ascii=False, indent=2))
    if suite.plan_only:
        return plans
    config = full._jsonable({"schema": SCHEMA, "source_root": suite.source_root, "datasets": suite.datasets,
        "experiments": {name: adapter.configuration(args) for name, (adapter, args) in prepared.items()},
        "training_protocol": "single-group-streamed-v1", "arms": matrix.DEFAULT_ARMS})
    # The optional control has a separate immutable intervention plan containing
    # its baseline hash, retrieval fingerprints and all DROP IDs. Keep the
    # existing training/evaluation identity unchanged so a completed four-arm
    # run can be extended without modifying or rerunning its fitted models.
    manifest = suite.output_dir / "seed0_manifest.json"
    if not manifest.exists() and suite.output_dir.exists() and any(p.name != ".seed0_parallel.lock" for p in suite.output_dir.iterdir()):
        raise ValueError("nonempty seed0 output has no manifest; use a fresh --output-dir")
    full.save_plan(manifest, config)
    full._save(suite.output_dir / "experiment_plan.json", plans)
    state = {"status": "running", "stages": {}, "current_stage": None, "error": None}
    current_stage, current_log, current_command = "initialization", None, []
    baselines = {}

    def stage(label, action, command=None):
        nonlocal current_stage, current_log, current_command
        current_stage, current_command = label, full._strip_key(command or [])
        current_log = suite.output_dir / "logs" / (label.replace(":", "_") + ".log")
        state.update(current_stage=label)
        state["stages"][label] = "running"
        full._save(suite.output_dir / "seed0_progress.json", state)
        action()
        state["stages"][label] = "completed"
        full._save(suite.output_dir / "seed0_progress.json", state)
        progress.update(1)

    def run_if_needed(ready, command, args):
        if not ready():
            base._run_command(current_stage, command, current_log, api_key=args.api_key)
            if not ready():
                raise RuntimeError(f"stage {current_stage} exited with incomplete artifacts")

    try:
        with tqdm(total=((14 if suite.include_always_global_drop else 11) + int(suite.collect_missing)) * len(prepared),
                  desc="Seed0-only gate matrix", unit="stage") as progress:
            for name, (adapter, args) in prepared.items():
                if suite.collect_missing:
                    # Also validates that a frozen snapshot exists before any
                    # child is launched. It is never silently built/replaced.
                    diagnostic_ids(args)
                    if suite.collection_endpoints:
                        from .parallel_collection import collect_parallel
                        stage(name + ":collect_seed0", lambda: collect_parallel(adapter, args, source_args[name]))
                    else:
                        command = collection_command(adapter, args, source_args[name])
                        stage(name + ":collect_seed0", lambda: run_if_needed(
                            lambda: collection_complete(args), command, args), command)
                stage(name + ":validate_seed0", lambda: full.save_plan(args.output_dir / "reused_inputs.json", validate_sources(adapter, args)))
                selection = args.output_dir / "baseline_selection.json"
                saved = base._load_json(selection)
                if saved:
                    baseline = Path(saved["path"])
                else:
                    original = source_args[name]
                    source_baseline = original.output_dir / "native_gmemory"
                    baseline = source_baseline if adapter.evaluation_complete(original, source_baseline, mode="always_keep") else args.output_dir / "native_gmemory"
                    full.save_plan(selection, {"path": str(baseline)})
                baselines[name] = baseline
                if baseline != args.output_dir / "native_gmemory":
                    original = source_args[name]
                    def reuse_baseline():
                        if not adapter.evaluation_complete(original, baseline, mode="always_keep"):
                            raise ValueError("previously selected source baseline is incomplete")
                        full.save_plan(args.output_dir / "reused_baseline.json", {
                            "path": str(baseline), "sha256": {p: file_sha256(baseline / p)
                            for p in ("memory_gate.jsonl", "run_manifest.json", "llm_cache.sqlite")}})
                    stage(name + ":baseline", reuse_baseline)
                else:
                    command = adapter.evaluation_command(args, mode="always_keep", output=baseline,
                        resume=(baseline / "run_manifest.json").is_file())
                    stage(name + ":baseline", lambda: run_if_needed(
                        lambda: adapter.evaluation_complete(args, baseline, mode="always_keep", command=command), command, args), command)
                for arm in matrix.DEFAULT_ARMS:
                    child = matrix._arm_args(args, arm)
                    output = args.output_dir / arm
                    command = train_command(adapter, child)
                    stage(name + ":" + arm + ":train", lambda: run_if_needed(
                        lambda: checkpoint_ready(adapter, child), command, child), command)
                    command = evaluation_command(adapter, child, output, baseline)
                    stage(name + ":" + arm + ":evaluate", lambda: run_if_needed(
                        lambda: adapter.evaluation_complete(child, output, mode="learned", command=command), command, child), command)
                    payload = base._load_json(child.checkpoint)
                    full._save(output / "training_summary.json", {
                        "estimator_config": payload["estimator"]["config"],
                        "training_metadata": payload["training_metadata"]})
                    stage(name + ":" + arm + ":compare", lambda: audit(adapter, child, output, baseline))
                if suite.include_always_global_drop:
                    stage(name + ":" + GLOBAL_DROP + ":plan", lambda: prepare_global_drop(args, baseline))
                    output, plan = args.output_dir / GLOBAL_DROP, global_drop_plan(args)
                    command = full.planned_command(adapter, args, output, plan, baseline / "llm_cache.sqlite")
                    stage(name + ":" + GLOBAL_DROP + ":evaluate", lambda: run_if_needed(
                        lambda: full.planned_ready(adapter, args, output, command, plan), command, args), command)
                    stage(name + ":" + GLOBAL_DROP + ":compare", lambda: audit(adapter, args, output, baseline))
                # A per-dataset table is available even before the next dataset runs.
                report(suite.output_dir, {k: prepared[k] for k in baselines}, baselines,
                    include_always_global_drop=suite.include_always_global_drop)
        state.update(status="completed", current_stage=None)
        full._save(suite.output_dir / "seed0_progress.json", state)
        print((suite.output_dir / "seed0_results.md").read_text(encoding="utf-8"))
        return suite.output_dir / "seed0_results.json"
    except BaseException as exc:
        trace = traceback.format_exc()
        if current_log:
            current_log.parent.mkdir(parents=True, exist_ok=True)
            with current_log.open("a", encoding="utf-8") as handle:
                handle.write("\n" + trace)
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error={
            "stage": current_stage, "type": type(exc).__name__, "message": str(exc),
            "exit_code": getattr(exc, "returncode", None), "command": current_command,
            "log_path": str(current_log), "traceback": trace,
            "log_tail": base._tail(current_log) if current_log and current_log.is_file() else None})
        full._save(suite.output_dir / "seed0_progress.json", state)
        raise


if __name__ == "__main__":
    main()
