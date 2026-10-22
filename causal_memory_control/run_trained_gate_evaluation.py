"""Reevaluate existing gates on their original held-out tasks, without training.

The CSM report locates the old checkpoints. A fresh baseline supplies the only
answer cache shared by the new evaluations; no old answer cache is imported.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from . import run_csm_st_benchmark as csm
from . import run_full_gate_benchmark as full
from . import run_hotpotqa_gate_comparison as base
from .compare_gate_runs import compare, file_sha256
from .csm_st_data import digest, save_json
from .planned_drop import build_plan
from .source_run import import_source_dataset


SCHEMA = "trained-gates-evaluation-v1"
POLICIES = (
    ("native_gmemory", "Always Use (baseline)"),
    ("always_global_drop", "Always Global Drop"),
    ("linear", "Linear Gate"),
    ("semantic_linear", "Semantic Linear Gate"),
    ("semantic_mlp", "Semantic MLP"),
    ("semantic_st", "Old Semantic ST"),
    ("csm_st", "CSM-ST Local–Global (CE proxy)"),
)
NOTE = ("All available policies are reevaluated using fixed existing checkpoints on the original "
        "held-out tasks. No training, calibration or CE collection. The new baseline starts with an "
        "empty answer cache; gates share only this new baseline cache for paired prompt replay. "
        "Old calibrated budgets are preserved; CSM keeps CI>0 with unlimited drops. "
        "FEVER is binary Accuracy, not evidence-aware FEVER Score. HotpotQA is Answer F1/EM. "
        "CSM uses a single-generator CE proxy. Missing old ST is shown as unavailable.")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trained-root", "--csm-root", type=Path, required=True,
                        help="Completed CSM benchmark root containing csm_st_manifest.json and csm_st_results.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", default="fever,hotpotqa")
    parser.add_argument("--csm-device", default="cuda:0", help="CSM/BERT device; does not move the Ollama service")
    parser.add_argument("--seed", type=int, help="Optional new evaluation seed, applied equally to every policy")
    parser.add_argument("--fever-old-st-root", type=Path)
    parser.add_argument("--hotpotqa-old-st-root", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    args.datasets = list(dict.fromkeys(v.strip() for v in args.datasets.split(",") if v.strip()))
    if not args.datasets or set(args.datasets) - {"fever", "hotpotqa"}:
        parser.error("--datasets must select fever,hotpotqa")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be nonnegative")
    args.trained_root, args.output_dir = args.trained_root.resolve(), args.output_dir.resolve()
    return args


def prepare(suite):
    manifest_path = suite.trained_root / "csm_st_manifest.json"
    config, report = read(manifest_path), read(suite.trained_root / "csm_st_results.json")
    if config.get("schema") != csm.SCHEMA or report.get("schema") != csm.SCHEMA:
        raise ValueError("--trained-root must be a CSM benchmark, not an old ST/matrix root")
    source_root = Path(config["source_root"]).resolve()
    sources = {manifest_path, suite.trained_root / "csm_st_results.json",
               source_root / "benchmark_manifest.json"}
    roots = {suite.trained_root, source_root}
    prepared, plan = {}, {}
    for name in suite.datasets:
        recorded = config["experiments"][name]
        adapter, args = import_source_dataset(SimpleNamespace(source_root=source_root, shared_argv=[]), name)
        snapshot_path = args.memory_dir / "causal_snapshot_manifest.json"
        snapshot = read(snapshot_path)
        sources.add(snapshot_path)
        expected_ids = snapshot["evaluation_ids"][args.training_claims:args.evaluation_count]
        if ([str(v) for v in expected_ids] != [str(v) for v in recorded["test_ids"]]
                or args.training_claims != recorded["original_test_offset"]):
            raise ValueError(f"{name}: original held-out test partition changed")
        if args.failed_topk:
            raise ValueError("Global DROP ALL requires the original failed_topk=0 protocol")
        args.output_dir = suite.output_dir / name
        args.semantic_cache = args.output_dir / "semantic_cache.sqlite"
        args.baseline_cache_seed_from = None
        args.endpoint = config["settings"].get("evaluation_endpoint") or args.endpoint
        if suite.seed is not None:
            args.gate_evaluation_seed = suite.seed
        methods = {}
        details = report.get("details", {}).get(name, {})
        for method, _ in POLICIES:
            child = copy.copy(args)
            if method in ("native_gmemory", "always_global_drop"):
                child.checkpoint, child.kappa, child.max_drops = None, 0.0, -1
            else:
                old_st = getattr(suite, f"{name}_old_st_root") if method == "semantic_st" else None
                previous = (old_st.resolve() / name / "evaluation" if old_st else
                            Path(details[method]["path"]) if method in details else None)
                if method == "semantic_st" and previous is None:
                    continue
                if method == "csm_st":
                    child.checkpoint = suite.trained_root / name / "checkpoint.json"
                    child.kappa, child.delta, child.max_drops = 0.0, 0.0, -1
                else:
                    if previous is None:
                        raise FileNotFoundError(f"{name}/{method}: checkpoint location missing from CSM report")
                    roots.add(previous.parent.parent.resolve())
                    run_path = previous / "run_manifest.json"
                    if run_path.is_file():
                        sources.add(run_path)
                        run = read(run_path)["run"]
                        child.checkpoint = Path(run["checkpoint"]).resolve()
                        child.kappa, child.delta, child.max_drops = run["kappa"], run["delta"], run["max_drops"]
                    elif old_st:
                        child.checkpoint = old_st.resolve() / name / "checkpoint.json"
                        child.kappa = "auto"
                    else:
                        raise FileNotFoundError(f"original evaluated policy manifest required: {run_path}")
                payload = read(child.checkpoint)
                adapter.validate_checkpoint(payload, child, snapshot)
                sources.add(child.checkpoint)
                if method == "csm_st":
                    if payload["training_metadata"].get("config_digest") != digest(config):
                        raise ValueError(f"{name}: CSM checkpoint belongs to another training configuration")
                    spec = payload["csm_st"]
                    artifact = child.checkpoint.parent / spec["artifact_dir"]
                    if artifact.resolve().parent != child.checkpoint.parent.resolve():
                        raise ValueError("CSM model must be beside its checkpoint")
                    if "weights.pt" not in spec["artifact_sha256"]:
                        raise ValueError("CSM checkpoint lacks model weight hash")
                    for relative, expected in spec["artifact_sha256"].items():
                        path = (artifact / relative).resolve()
                        if artifact.resolve() not in path.parents or file_sha256(path) != expected:
                            raise ValueError(f"CSM artifact changed: {path}")
                        sources.add(path)
            methods[method] = child
        prepared[name] = (adapter, methods)
        plan[name] = {"test_tasks": len(expected_ids), "test_offset": args.training_claims,
            "endpoint": args.endpoint, "model": args.model, "seed": args.gate_evaluation_seed,
            "methods": list(methods), "unavailable": [m for m, _ in POLICIES if m not in methods],
            "checkpoints": {m: str(a.checkpoint) for m, a in methods.items() if a.checkpoint},
            "maximum_team_inferences": len(expected_ids) * len(methods),
            "new_training_runs": 0, "new_ce_forwards": 0, "old_answer_caches_imported": 0}
    # Never put a reevaluation inside any of its original result directories.
    for root in roots:
        if suite.output_dir == root or root in suite.output_dir.parents or suite.output_dir in root.parents:
            raise ValueError(f"use a separate, non-nested output directory, away from {root}")
    identity = {str(path.resolve()): file_sha256(path) for path in sorted(sources)}
    return prepared, plan, identity


def commands(suite, name, adapter, args, method):
    output = args.output_dir / method
    if method == "always_global_drop":
        plan = args.output_dir / "plans" / "always_global_drop.json"
        full.save_plan(plan, build_plan(args.output_dir / "native_gmemory" / "memory_gate.jsonl", policy="global_all"))
        host = full.planned_command(adapter, args, output, plan,
                                    args.output_dir / "native_gmemory" / "llm_cache.sqlite")
    else:
        host = adapter.evaluation_command(args, mode="always_keep" if method == "native_gmemory" else "learned",
            output=output, resume=(output / "run_manifest.json").is_file())
    command = host
    if method == "csm_st":
        command = [*host[:3], "causal_memory_control.csm_st_evaluate", "--dataset", name,
                   "--csm-device", suite.csm_device, *host[4:]]
    return host, command


def write_report(root, prepared, results):
    rows = []
    for method, label in POLICIES:
        row = {"Policy": label}
        for name, (adapter, _) in prepared.items():
            result = results.get(name, {}).get(method)
            for metric in adapter.metrics[:1 if name == "fever" else 2]:
                row[f"{adapter.display_name} {metric.label}"] = (None if result is None else
                    100 * result[metric.summary_field if method == "native_gmemory" else metric.gate_field])
        rows.append(row)
    save_json(root / "reevaluation_results.json", {"schema": SCHEMA, "notes": NOTE,
                                                  "rows": rows, "details": results})
    with (root / "reevaluation_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(str(row[c]) if c == "Policy" else
        "—" if row[c] is None else f"{row[c]:.2f}%" for c in columns) + " |" for row in rows]
    (root / "reevaluation_results.md").write_text("\n".join(lines) + "\n\n" + NOTE + "\n", encoding="utf-8")


def main(argv=None):
    suite = parse_args(argv)
    prepared, plan, identity = prepare(suite)
    print(json.dumps({"plan_only": suite.plan_only, "datasets": plan, "notes": NOTE}, ensure_ascii=False, indent=2))
    if suite.plan_only:
        return plan
    manifest = suite.output_dir / "reevaluation_manifest.json"
    config = {"schema": SCHEMA, "trained_root": str(suite.trained_root), "datasets": plan,
              "csm_device": suite.csm_device, "input_sha256": identity}
    if manifest.is_file():
        if read(manifest) != config:
            raise ValueError("reevaluation configuration/inputs changed; use original arguments or a new output directory")
    elif suite.output_dir.exists() and any(suite.output_dir.iterdir()):
        raise ValueError("refusing nonempty output without reevaluation_manifest.json")
    save_json(manifest, config)
    state, results = {"status": "running", "current_stage": None, "stages": {}}, {}
    progress = suite.output_dir / "reevaluation_progress.json"
    try:
        for name, (adapter, methods) in prepared.items():
            results[name] = {}
            for method, args in methods.items():
                label = f"{name}:{method}"
                state["current_stage"] = label
                state["stages"][label] = "running"
                save_json(progress, state)
                output = args.output_dir / method
                host, command = commands(suite, name, adapter, args, method)
                mode = "planned_drop" if method == "always_global_drop" else "always_keep" if method == "native_gmemory" else "learned"

                def complete():
                    if method == "always_global_drop":
                        return full.planned_ready(adapter, args, output, host,
                            args.output_dir / "plans" / "always_global_drop.json")
                    return adapter.evaluation_complete(args, output, mode=mode, command=host)

                reused = complete()
                if not reused:
                    base._run_command(label, command, suite.output_dir / "logs" / f"{name}_{method}.log", api_key=args.api_key)
                if not complete():
                    raise RuntimeError(f"{label}: evaluation exited without complete results")
                if method == "native_gmemory":
                    result = read(output / "summary.json")
                    if not (output / "llm_cache.sqlite").is_file():
                        raise FileNotFoundError("new baseline answer cache is required before gate evaluation")
                else:
                    result = compare(args.output_dir / "native_gmemory" / "memory_gate.jsonl",
                        output / "memory_gate.jsonl", bootstrap_samples=args.bootstrap_samples, seed=args.gate_evaluation_seed)
                    save_json(output / "comparison.json", result)
                    if (result["paired_tasks"] != args.final_claims or not result["paired_control_valid"]
                            or not result["policy_valid"]):
                        raise ValueError(f"{label}: invalid paired control; inspect {output / 'comparison.json'}")
                results[name][method] = result
                state["stages"][label] = "reused" if reused else "completed"
                save_json(progress, state)
                write_report(suite.output_dir, prepared, results)
        for path, expected in identity.items():
            if file_sha256(Path(path)) != expected:
                raise ValueError(f"training/source artifact changed during evaluation: {path}")
        state.update(status="completed", current_stage=None)
        save_json(progress, state)
        print((suite.output_dir / "reevaluation_results.md").read_text())
        return suite.output_dir / "reevaluation_results.json"
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error={"type": type(exc).__name__, "message": str(exc)})
        save_json(progress, state)
        raise


if __name__ == "__main__":
    main()
