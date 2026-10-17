"""Train/evaluate three gates using only a completed first collection group.

Import the original full-run configuration and frozen memory. Never invokes a
diagnostic collector, seed1000, HotpotQA (unless selected), Jev or ST. Original
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
import traceback

from tqdm import tqdm

from . import run_full_gate_benchmark as full
from . import run_hotpotqa_gate_comparison as base
from . import run_hotpotqa_gate_matrix as matrix
from .compare_gate_runs import compare, file_sha256
from .source_run import import_source_dataset


SCHEMA = "seed0-only-gate-matrix-v1"
NOTE = ("One completed collection group only; repeats inside that group remain distinct. "
        "Same task-disjoint fit/dev/test split and original frozen memory. "
        "Calibrate DROP on diagnostic dev tasks; max one DROP, KEEP ALL fallback allowed. "
        "FEVER reports binary Accuracy, not evidence-aware FEVER Score; HotpotQA reports Answer F1/EM.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path,
        default=Path("causal_memory_control/results/full_gate_benchmark_v1"))
    parser.add_argument("--output-dir", type=Path,
        default=Path("causal_memory_control/results/seed0_gate_matrix_v1"))
    parser.add_argument("--datasets", default="fever")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    args.datasets = list(dict.fromkeys(v.strip() for v in args.datasets.split(",") if v.strip()))
    if not args.datasets or set(args.datasets) - {"fever", "hotpotqa"}:
        parser.error("--datasets must select fever,hotpotqa")
    args.source_root, args.output_dir = args.source_root.resolve(), args.output_dir.resolve()
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
    ids = snapshot["evaluation_ids"][:args.training_claims]
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


def train_command(adapter, args):
    return [*base._train_command(args, metric=adapter.training_metric, sources=[source_path(args)]), "--stream-branches"]


def checkpoint_ready(adapter, args):
    if not base._checkpoint_complete(args, benchmark=adapter.benchmark,
            metric=adapter.training_metric, sources=[source_path(args)]):
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


def report(root, prepared, baselines):
    tables, details = {}, {}
    lines = ["# Seed0-only gate results", "", NOTE, ""]
    for name, (adapter, args) in prepared.items():
        baseline = base._load_json(baselines[name] / "summary.json")
        rows = [{"method": "native_gmemory", "tasks": args.final_claims,
            "keep_rate": 1.0, "tasks_with_drop": 0, "drop_improved_tasks": 0, "drop_harmed_tasks": 0}]
        for metric in adapter.metrics:
            rows[0].update({metric.name: baseline[metric.summary_field], metric.delta_field: 0.0, metric.interval_field: [0.0, 0.0]})
        for arm in matrix.DEFAULT_ARMS:
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
        full._save(args.output_dir / "matrix_results.json", {"dataset": name, "results": rows, "notes": NOTE})
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
        lines += [f"## {adapter.display_name}", "", *table, ""]
    full._save(root / "seed0_results.json", {"schema": SCHEMA, "notes": NOTE, "results": tables, "details": details})
    (root / "seed0_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    suite = parse_args(argv)
    prepared, source_args = {}, {}
    for name in suite.datasets:
        adapter, original = import_source_dataset(suite, name)
        source_args[name] = copy.copy(original)
        args = copy.copy(original)
        args.output_dir = suite.output_dir / name
        args.checkpoint = args.output_dir / "unused.json"
        args.semantic_cache = args.output_dir / "semantic_cache.sqlite"
        args.kappa = "auto"
        prepared[name] = adapter, args
    plans = {name: {**adapter.plan(args), "training_sources": [str(source_path(args))],
        "new_diagnostic_branches": 0, "collection_groups": 1,
        "sample_seeds": list(range(args.sample_seed_base, args.sample_seed_base + args.repeats)),
        "arms": list(matrix.DEFAULT_ARMS), "maximum_test_team_inferences": 4 * args.final_claims}
        for name, (adapter, args) in prepared.items()}
    for plan in plans.values():
        plan.pop("diagnostic_branches_two_seeds", None)
    print(json.dumps({"schema": SCHEMA, "datasets": plans, "notes": NOTE}, ensure_ascii=False, indent=2))
    if suite.plan_only:
        return plans
    config = full._jsonable({"schema": SCHEMA, "source_root": suite.source_root, "datasets": suite.datasets,
        "experiments": {name: adapter.configuration(args) for name, (adapter, args) in prepared.items()},
        "training_protocol": "single-group-streamed-v1", "arms": matrix.DEFAULT_ARMS})
    manifest = suite.output_dir / "seed0_manifest.json"
    if not manifest.exists() and suite.output_dir.exists() and any(suite.output_dir.iterdir()):
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
        with tqdm(total=11 * len(prepared), desc="Seed0-only gate matrix", unit="stage") as progress:
            for name, (adapter, args) in prepared.items():
                stage(name + ":validate_seed0", lambda: full.save_plan(args.output_dir / "reused_inputs.json", validate_seed0(adapter, args)))
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
                # A per-dataset table is available even before the next dataset runs.
                report(suite.output_dir, {k: prepared[k] for k in baselines}, baselines)
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
