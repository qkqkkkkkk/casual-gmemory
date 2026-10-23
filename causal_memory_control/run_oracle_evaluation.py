"""Oracle-only extension of a completed trained-gate reevaluation.

Reuse the latest held-out baseline, policy outcomes and answer caches. Collect
only missing single-memory DROP branches (and DROP ALL if absent). Never train
models, collect diagnostic labels, rebuild memory, or rerun learned gates.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from . import run_full_gate_benchmark as full
from . import run_trained_gate_evaluation as trained
from .compare_gate_runs import compare, file_sha256
from .csm_st_data import save_json
from .planned_drop import build_plan, indexed_rows
from .source_run import import_source_dataset


SCHEMA = "held-out-oracle-only-v1"
ORACLE = "oracle_selective_drop"
NOTE = ("Post-hoc maximum over KEEP ALL, every single-memory DROP and DROP ALL, using the latest "
        "held-out evaluation and its seed. Not a deployable policy or an upper bound on expected "
        "reward/all memory subsets. Same-action outcomes must agree. Other trained policies are "
        "reused for reporting; arbitrary multi-drop CSM actions do not expand the Oracle action space. "
        "FEVER: binary Accuracy, not evidence-aware FEVER Score. HotpotQA: Answer F1/EM. "
        "No training, diagnostic-label collection or memory rebuilding.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--evaluation-root", type=Path, required=True,
                        help="Completed run_trained_gate_evaluation result root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", default="fever,hotpotqa")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    args.datasets = list(dict.fromkeys(v.strip() for v in args.datasets.split(",") if v.strip()))
    if not args.datasets or set(args.datasets) - {"fever", "hotpotqa"}:
        parser.error("--datasets must select fever,hotpotqa")
    args.evaluation_root, args.output_dir = args.evaluation_root.resolve(), args.output_dir.resolve()
    return args


def source_command(adapter, args, root):
    """Reconstruct the stored policy for the existing strict evaluation audit."""
    run = trained.read(root / "run_manifest.json")["run"]
    child = copy.copy(args)
    child.checkpoint = Path(run["checkpoint"]) if run["checkpoint"] else None
    child.kappa, child.delta, child.max_drops = run["kappa"], run["delta"], run["max_drops"]
    child.baseline_cache_seed_from = Path(run["cache_seed_from"]) if run["cache_seed_from"] else None
    command = adapter.evaluation_command(child, mode=run["mode"], output=root, resume=False)
    if "--cache-seed-from" in command:
        command[command.index("--cache-seed-from") + 1] = run["cache_seed_from"]
    if run["mode"] == "planned_drop":
        command += ["--intervention-plan", run["intervention_plan"]]
    return command, run


def prepare(suite):
    manifest_path = suite.evaluation_root / "reevaluation_manifest.json"
    manifest = trained.read(manifest_path)
    if manifest.get("schema") != trained.SCHEMA:
        raise ValueError("--evaluation-root must contain a trained-gate reevaluation manifest")
    trained_root = Path(manifest["trained_root"]).resolve()
    csm_path = trained_root / "csm_st_manifest.json"
    source_root = Path(trained.read(csm_path)["source_root"]).resolve()
    for root in (suite.evaluation_root, trained_root, source_root):
        if suite.output_dir == root or root in suite.output_dir.parents or suite.output_dir in root.parents:
            raise ValueError(f"use separate non-nested output, away from {root}")
    paths = {manifest_path, csm_path, source_root / "benchmark_manifest.json"}
    prepared, plan = {}, {}
    for name in suite.datasets:
        recorded = manifest["datasets"][name]
        adapter, args = import_source_dataset(SimpleNamespace(source_root=source_root, shared_argv=[]), name)
        if recorded["test_offset"] != args.training_claims or recorded["test_tasks"] != args.final_claims:
            raise ValueError(f"{name}: held-out partition differs from original source")
        args.endpoint, args.gate_evaluation_seed = recorded["endpoint"], recorded["seed"]
        if recorded["model"] != args.model or args.failed_topk != 0:
            raise ValueError(f"{name}: original model/failed-topk protocol changed")
        args.output_dir = suite.output_dir / name
        args.checkpoint, args.kappa, args.delta, args.max_drops = None, 0.0, 0.0, -1
        args.baseline_cache_seed_from = None
        args.semantic_cache = args.output_dir / "semantic_cache.sqlite"
        snapshot_path = args.memory_dir / "causal_snapshot_manifest.json"
        paths.add(snapshot_path)
        methods = recorded["methods"]
        if "native_gmemory" not in methods:
            raise ValueError(f"{name}: completed latest baseline is required")
        baseline = suite.evaluation_root / name / "native_gmemory" / "memory_gate.jsonl"
        outcomes, sources, caches = {}, {}, []
        for method in methods:
            root = suite.evaluation_root / name / method
            if not (root / "run_manifest.json").is_file():
                raise FileNotFoundError(f"finish the latest reevaluation first: {root}")
            command, run = source_command(adapter, args, root)
            if not adapter.evaluation_complete(args, root, mode=run["mode"], command=command):
                raise ValueError(f"finish the latest reevaluation first: {root}")
            log = root / "memory_gate.jsonl"
            rows = indexed_rows(log)
            comparison = compare(baseline, log, bootstrap_samples=1, seed=args.gate_evaluation_seed)
            if not comparison["paired_control_valid"] or not comparison["policy_valid"]:
                raise ValueError(f"invalid paired source: {root}")
            allowed = all(len(drops := {d["memory_id"] for d in row["decisions"] if d["action"] == "drop"}) <= 1
                          or drops == {d["memory_id"] for d in row["decisions"]} for row in rows.values())
            if method == "always_global_drop" and not all(
                    all(d["action"] == "drop" for d in row["decisions"]) for row in rows.values()):
                raise ValueError("latest Global Drop must hide ALL retrieved candidates")
            if allowed:
                sources[method] = log
            caches.append(root / "llm_cache.sqlite")
            paths.update(root / p for p in ("run_manifest.json", "memory_gate.jsonl", "summary.json", "llm_cache.sqlite", "progress.json"))
            if method != "native_gmemory":
                # Recompute CIs for the reports; never trust stale table values.
                outcomes[method] = (comparison if suite.plan_only else compare(baseline, log,
                    bootstrap_samples=args.bootstrap_samples, seed=args.gate_evaluation_seed))
            else:
                outcomes[method] = trained.read(root / "summary.json")
        count = args.successful_topk + args.insights_topk
        if count < 1:
            raise ValueError("Oracle requires at least one retrieved candidate")
        prepared[name] = (adapter, args, sources, caches, outcomes)
        plan[name] = {"test_tasks": args.final_claims, "test_offset": args.training_claims,
            "endpoint": args.endpoint, "model": args.model, "seed": args.gate_evaluation_seed,
            "single_drop_branches_per_task": count, "new_global_drop_required": "always_global_drop" not in methods,
            "maximum_new_team_inferences": args.final_claims * (count + int("always_global_drop" not in methods)),
            "reused_methods": methods, "new_training_runs": 0, "new_diagnostic_branches": 0,
            "action_space": "KEEP ALL + each single-memory DROP + DROP ALL"}
    return prepared, plan, {str(p): file_sha256(p) for p in sorted(paths)}


def write_report(root, prepared, results):
    policies = [*trained.POLICIES, (ORACLE, "Oracle Selective Drop (post-hoc)")]
    rows = []
    for method, label in policies:
        row = {"Policy": label}
        for name, (adapter, _, _, _, _) in prepared.items():
            result = results.get(name, {}).get(method)
            for metric in adapter.metrics[:1 if name == "fever" else 2]:
                row[f"{adapter.display_name} {metric.label}"] = None if result is None else 100 * result[
                    metric.summary_field if method == "native_gmemory" else metric.gate_field]
        rows.append(row)
    save_json(root / "oracle_results.json", {"schema": SCHEMA, "notes": NOTE, "rows": rows, "details": results})
    columns = list(rows[0])
    with (root / "oracle_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(str(row[c]) if c == "Policy" else "—" if row[c] is None else
        f"{row[c]:.2f}%" for c in columns) + " |" for row in rows]
    (root / "oracle_results.md").write_text("\n".join(lines) + "\n\n" + NOTE + "\n", encoding="utf-8")


def main(argv=None):
    suite = parse_args(argv)
    prepared, plan, identity = prepare(suite)
    print(json.dumps({"plan_only": suite.plan_only, "datasets": plan, "notes": NOTE}, ensure_ascii=False, indent=2))
    if suite.plan_only:
        return plan
    config = {"schema": SCHEMA, "evaluation_root": str(suite.evaluation_root), "datasets": plan, "input_sha256": identity}
    manifest = suite.output_dir / "oracle_run_manifest.json"
    if not manifest.is_file() and suite.output_dir.exists() and any(suite.output_dir.iterdir()):
        raise ValueError("refusing nonempty Oracle output without oracle_run_manifest.json")
    full.save_plan(manifest, config)
    results = {name: copy.deepcopy(value[4]) for name, value in prepared.items()}
    state = {"status": "running", "current_stage": None, "stages": {}}
    progress = suite.output_dir / "oracle_progress.json"

    def stage(label, action):
        state["current_stage"] = label
        state["stages"][label] = "running"
        save_json(progress, state)
        value = action()
        state["stages"][label] = "completed"
        save_json(progress, state)
        return value

    try:
        write_report(suite.output_dir, prepared, results)
        for name, (adapter, args, original_sources, caches, _) in prepared.items():
            baseline = args.output_dir / "native_gmemory" / "memory_gate.jsonl"
            baseline.parent.mkdir(parents=True, exist_ok=True)
            original_baseline = original_sources["native_gmemory"]
            if baseline.exists():
                if file_sha256(baseline) != file_sha256(original_baseline):
                    raise ValueError("copied baseline changed")
            else:
                shutil.copy2(original_baseline, baseline)
            save_json(baseline.parent / "reused_from.json", {"path": str(original_baseline), "sha256": file_sha256(baseline)})
            sources = {**original_sources, "native_gmemory": baseline}
            cache = args.output_dir / "cache_seeds" / "latest_union.sqlite"
            stage(name + ":freeze_cache", lambda: full.merge_caches(caches, cache))
            jobs = [] if "always_global_drop" in sources else [("always_global_drop", "global_all", 0)]
            jobs += [(f"oracle_branches/drop_{i:03d}", "candidate_index", i)
                     for i in range(args.successful_topk + args.insights_topk)]
            for method, policy, index in jobs:
                output = args.output_dir / method
                path = args.output_dir / "plans" / (method.replace("/", "_") + ".json")
                full.save_plan(path, build_plan(baseline, policy=policy, index=index))
                command = full.planned_command(adapter, args, output, path, cache)

                def evaluate():
                    if not full.planned_ready(adapter, args, output, command, path):
                        full.base._run_command(name + ":" + method, command,
                            suite.output_dir / "logs" / f"{name}_{method.replace('/', '_')}.log", api_key=args.api_key)
                    if not full.planned_ready(adapter, args, output, command, path):
                        raise RuntimeError(f"incomplete Oracle branch: {output}")
                    return full.audited_comparison(args, output)

                comparison = stage(name + ":" + method, evaluate)
                if method == "always_global_drop":
                    results[name][method] = comparison
                sources[method] = output / "memory_gate.jsonl"
                cache = output / "llm_cache.sqlite"
            results[name][ORACLE] = stage(name + ":oracle", lambda: full.build_oracle(args, sources))
            # Record the restricted action space alongside the original selector metadata.
            save_json(args.output_dir / ORACLE / "action_space.json", {"notes": NOTE, "actions": plan[name]["action_space"]})
            write_report(suite.output_dir, prepared, results)
        for path, expected in identity.items():
            if file_sha256(Path(path)) != expected:
                raise ValueError(f"read-only source changed during Oracle evaluation: {path}")
        state.update(status="completed", current_stage=None)
        save_json(progress, state)
        print((suite.output_dir / "oracle_results.md").read_text())
        return suite.output_dir / "oracle_results.json"
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error={"type": type(exc).__name__, "message": str(exc)})
        save_json(progress, state)
        raise


if __name__ == "__main__":
    main()
