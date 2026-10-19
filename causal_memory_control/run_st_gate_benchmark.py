"""Full-data FEVER/HotpotQA ST-only extension: reuse data, memory and baseline.

Never invokes another gate, the matrix runner or the seven-policy runner.
Missing diagnostics/baseline require explicit --prepare-missing authorization.
The --seed0-only extension reuses registered full/subset/sharded data; it never
collects data, changes the original test boundary or runs a missing baseline.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import traceback
from pathlib import Path

from tqdm import tqdm

from . import run_full_gate_benchmark as full
from . import run_hotpotqa_gate_comparison as base
from .compare_gate_runs import compare, file_sha256
from .source_run import import_source_dataset
from .st_estimator import SCHEMA as ESTIMATOR_SCHEMA, ST_OPTIONS, STUtilityEstimator, add_st_arguments, st_settings


SCHEMA = "full-data-st-only-benchmark-v1"
NOTE = ("ST-loss adaptation (paper eqs.6-7), not full CSM reproduction. One direct U head, "
        "existing semantic features and USE-minus-DROP reward labels; no E2E/recombination. "
        "FEVER: binary Accuracy, not evidence-aware FEVER Score. HotpotQA: Answer F1/EM. "
        "All valid input samples partition into disjoint support, diagnostic fit/dev and test. "
        "Registered diagnostic subsets/shards reuse only their sampled tasks; the original test boundary is unchanged. "
        "Learned gate budget <=1, dev may select KEEP ALL. Other results are reused, never rerun.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path,
        default=Path("causal_memory_control/results/full_gate_benchmark_v1"))
    parser.add_argument("--output-dir", type=Path,
        default=Path("causal_memory_control/results/st_gate_benchmark_v1"))
    parser.add_argument("--datasets", default="fever,hotpotqa")
    parser.add_argument("--seed0-only", action="store_true",
        help="Reuse only the completed first collection group; never collect data or run a missing baseline")
    parser.add_argument("--seed0-results-root", type=Path,
        help="Existing run_seed0_gate_matrix root: reuse its registered diagnostic subset/shards, baseline/cache and audited old scores")
    parser.add_argument("--evaluation-endpoint",
        help="Seed0-only: send new ST evaluations to a verified Ollama replica; preserve source data and baseline")
    parser.add_argument("--fever-data", type=Path, default=Path("data/fever/fever_dev.jsonl"))
    parser.add_argument("--hotpotqa-data", type=Path, default=Path("data/hotpotqa/raw/hotpot_dev_distractor_v1.json"))
    parser.add_argument("--prepare-missing", action="store_true",
        help="Only if missing: collect full diagnostics/frozen memory and run baseline; never run other gates")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--ensemble-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--l2", type=float, default=0.0001)
    add_st_arguments(parser)
    args, shared = parser.parse_known_args(argv)
    args.datasets = list(dict.fromkeys(v.strip() for v in args.datasets.split(",") if v.strip()))
    if not args.datasets or set(args.datasets) - {"fever", "hotpotqa"}:
        parser.error("--datasets must select fever,hotpotqa")
    if args.seed0_only and args.prepare_missing:
        parser.error("--seed0-only cannot use --prepare-missing; existing data and baseline are required")
    if args.seed0_results_root is not None and not args.seed0_only:
        parser.error("--seed0-results-root requires --seed0-only")
    if args.evaluation_endpoint is not None:
        from .evaluation_replicas import validate_endpoint

        if not args.seed0_only:
            parser.error("--evaluation-endpoint requires --seed0-only")
        args.evaluation_endpoint = validate_endpoint(args.evaluation_endpoint)
    forbidden = {"--arms", "--matrix-dir", "--memory-dir", "--diagnostic-results", "--data", "--checkpoint",
                 "--estimator", "--feature-version", "--use-all-data", "--evaluation-count"}
    if any(v.split("=", 1)[0] in forbidden for v in shared):
        parser.error("data/split/artifact paths are managed by this full-data runner")
    args.source_root, args.output_dir = args.source_root.resolve(), args.output_dir.resolve()
    if (args.output_dir == args.source_root or args.source_root in args.output_dir.parents
            or args.output_dir in args.source_root.parents):
        parser.error("use separate, non-nested source and ST output directories")
    if args.seed0_only and not (args.source_root / "benchmark_manifest.json").is_file():
        parser.error("--seed0-only requires the original full-run benchmark_manifest.json")
    if args.seed0_results_root is not None:
        args.seed0_results_root = args.seed0_results_root.resolve()
        if (args.output_dir == args.seed0_results_root or args.seed0_results_root in args.output_dir.parents
                or args.output_dir in args.seed0_results_root.parents):
            parser.error("ST output must be separate from the read-only seed0 results root")
    args.shared_argv = shared
    STUtilityEstimator(**st_settings(args), ensemble_size=args.ensemble_size, seed=args.seed, l2=args.l2)
    return args


def prepare_dataset(suite, name):
    """Import recorded full-run settings; never silently substitute CLI defaults."""
    adapter, args = import_source_dataset(suite, name)
    if suite.seed0_only:
        args.st_seed0_only = True
    if suite.seed0_results_root is not None:
        # Do not assume the baseline lives in the original full-run directory:
        # the seed0 matrix may have run it in its own independent output root.
        from .run_seed0_gate_matrix import SCHEMA as SEED0_SCHEMA

        manifest = base._load_json(suite.seed0_results_root / "seed0_manifest.json")
        if (not manifest or manifest.get("schema") != SEED0_SCHEMA
                or manifest.get("source_root") != str(suite.source_root)
                or name not in manifest.get("experiments", {})):
            raise ValueError("seed0 results must register this dataset and the same original --source-root")
        recorded = manifest["experiments"][name]
        configuration = adapter.configuration(args)
        keys = ("data", "memory_dir", "source_md5", "endpoint", "model",
            "training_claims", "final_claims", "evaluation_count", "sample_seed_base",
            "selection_seed", "temperature", "gate_evaluation_seed", "node_num", "graph_type",
            "successful_topk", "failed_topk", "insights_topk", "threshold", "embedding_model",
            "calibration_fraction", "kappa_grid", "delta")
        base._require_fields("seed0 comparison protocol", recorded, {key: configuration[key] for key in keys})
        import_seed0_diagnostics(suite, name, recorded, args)
        args.st_results_dir = suite.seed0_results_root / name
        selection = base._load_json(args.st_results_dir / "baseline_selection.json")
        if not selection or not selection.get("path"):
            raise FileNotFoundError(f"seed0 baseline_selection.json is required: {args.st_results_dir}")
        selected = Path(selection["path"]).resolve()
        if selected not in (args.output_dir / "native_gmemory", args.st_results_dir / "native_gmemory"):
            raise ValueError("seed0 baseline selection points outside its registered source/result directories")
        args.st_baseline_dir = selected
    if suite.evaluation_endpoint is not None and suite.evaluation_endpoint != args.endpoint:
        args.st_evaluation_endpoint = suite.evaluation_endpoint
        args.st_replica_manifest = suite.output_dir / name / "evaluation_replicas.json"
    args.checkpoint = suite.output_dir / name / "checkpoint.json"
    args.semantic_cache = suite.output_dir / name / "semantic_cache.sqlite"
    # This is a training-method comparison. Keep deployment policy calibrated,
    # even if the source was once run with a manually forced test-time kappa.
    args.kappa = "auto"
    return adapter, args


def import_seed0_diagnostics(suite, name, recorded, args):
    """Reuse only the completed collection registered by the old matrix.

    training_claims stays the original test offset, even when fewer diagnostic
    tasks were sampled. Never rediscover worker paths or rewrite their hashes.
    """
    subset = "diagnostic_task_limit" in recorded
    count = recorded.get("diagnostic_task_limit", args.training_claims)
    if type(count) is not int or not 5 <= count <= args.training_claims:
        raise ValueError("invalid registered seed0 diagnostic task count")
    if recorded.get("diagnostic_task_offset", 0) != 0:
        raise ValueError("seed0 diagnostic subset must start at the original diagnostic prefix")
    expected_root = (suite.seed0_results_root / "artifacts" / name / "diagnostic"
                     if subset else args.diagnostic_results)
    base._require_fields("seed0 diagnostic location", recorded,
        {"diagnostic_results": str(expected_root)})
    if subset:
        repeats = recorded.get("repeats")
        if type(repeats) is not int or repeats < 1:
            raise ValueError("invalid registered seed0 repeat count")
        args.diagnostic_task_limit = count
        args.diagnostic_results = expected_root
        args.repeats = repeats
    else:
        base._require_fields("seed0 comparison protocol", recorded, {"repeats": args.repeats})
    endpoints = recorded.get("collection_endpoints")
    if endpoints:
        from .parallel_collection import parse_endpoints

        if not subset or not isinstance(endpoints, list):
            raise ValueError("parallel seed0 collection requires a registered diagnostic subset")
        parsed = parse_endpoints(",".join(endpoints))
        if len(parsed) > count or parsed[0] != args.endpoint.rstrip("/"):
            raise ValueError("registered collection endpoints do not match the original source")
        args.collection_endpoints = parsed


def diagnostic_count(args):
    return getattr(args, "diagnostic_task_limit", args.training_claims)


def training_sources(args):
    if getattr(args, "st_seed0_only", False):
        from .run_seed0_gate_matrix import training_sources as seed0_sources

        return seed0_sources(args)
    groups = [(args.sample_seed_base, "")]
    groups.append((args.retest_seed_base, "_retest"))
    return [args.diagnostic_results / f"seed{seed}{suffix}" / "branches.jsonl"
            for seed, suffix in groups]


def baseline_dir(args):
    return getattr(args, "st_baseline_dir", args.output_dir / "native_gmemory")


def diagnostics_ready(adapter, args):
    if getattr(args, "st_seed0_only", False):
        from .run_seed0_gate_matrix import validate_sources

        validate_sources(adapter, args)
        return True
    return adapter.diagnostic_complete(args)


def evaluation_command(adapter, args, output):
    command = adapter.evaluation_command(evaluation_args(args), mode="learned", output=output,
        resume=(output / "run_manifest.json").is_file())
    command[command.index("--cache-seed-from") + 1] = str(baseline_dir(args) / "llm_cache.sqlite")
    return command


def evaluation_args(args):
    child = copy.copy(args)
    child.endpoint = getattr(args, "st_evaluation_endpoint", args.endpoint)
    return child


def verify_evaluation_replica(args, output):
    from .evaluation_replicas import build_manifest, validate_manifest

    path = args.st_replica_manifest
    if path.exists():
        validate_manifest(path, baseline_dir(args) / "memory_gate.jsonl", source_endpoint=args.endpoint,
            target_endpoint=args.st_evaluation_endpoint, model=args.model)
        # Completed evaluations can be audited/reported offline. An unfinished
        # run must recheck live replicas before making any new model calls.
        if base._evaluation_complete(output):
            return
    full.save_plan(path, build_manifest(baseline_dir(args), source_endpoint=args.endpoint,
        target_endpoint=args.st_evaluation_endpoint, model=args.model))
    validate_manifest(path, baseline_dir(args) / "memory_gate.jsonl", source_endpoint=args.endpoint,
        target_endpoint=args.st_evaluation_endpoint, model=args.model)


def train_command(suite, adapter, args):
    command = [sys.executable, "-u", "-m", "causal_memory_control.train_gate",
        "--input", *(str(p) for p in training_sources(args)), "--output", str(args.checkpoint),
        "--metric", adapter.training_metric, "--memory-dir", str(args.memory_dir),
        "--estimator", "st", "--feature-version", "3",
        "--ensemble-size", str(suite.ensemble_size), "--seed", str(suite.seed), "--l2", str(suite.l2),
        "--semantic-model", args.semantic_model, "--semantic-chunk-tokens", str(args.semantic_chunk_tokens),
        "--semantic-cache", str(args.semantic_cache), "--residual-noise-scale", "0",
        "--calibration-fraction", str(args.calibration_fraction), "--calibration-seed", "71",
        "--kappa-grid", args.kappa_grid, "--calibration-max-drops", "1",
        "--calibration-delta", str(args.delta), "--max-changed-harm-rate", str(args.max_changed_harm_rate)]
    if args.semantic_revision:
        command += ["--semantic-revision", args.semantic_revision]
    for key in ST_OPTIONS:
        command += ["--st-" + key.replace("_", "-"), str(getattr(suite, "st_" + key))]
    return command


def reuse_identity(args):
    paths = [*training_sources(args), args.memory_dir / "causal_snapshot_manifest.json",
             baseline_dir(args) / "memory_gate.jsonl",
             baseline_dir(args) / "run_manifest.json",
             baseline_dir(args) / "llm_cache.sqlite"]
    paths += [p.parent / "run_manifest.json" for p in training_sources(args)]
    if hasattr(args, "st_results_dir"):
        paths += [args.st_results_dir.parent / "seed0_manifest.json", args.st_results_dir / "baseline_selection.json"]
    if hasattr(args, "st_replica_manifest"):
        paths.append(args.st_replica_manifest)
    if getattr(args, "collection_endpoints", None):
        paths.append(args.diagnostic_results / "parallel_collection_manifest.json")
    return {str(p): file_sha256(p) for p in paths}


def checkpoint_ready(suite, adapter, args):
    if not args.checkpoint.is_file():
        return False
    payload = base._load_json(args.checkpoint)
    if not payload or payload.get("estimator", {}).get("schema") != ESTIMATOR_SCHEMA:
        raise ValueError(f"invalid ST checkpoint: {args.checkpoint}")
    expected = STUtilityEstimator(**st_settings(suite), ensemble_size=suite.ensemble_size,
        seed=suite.seed, l2=suite.l2).config
    base._require_fields("ST estimator", payload["estimator"]["config"], expected)
    metadata = payload["training_metadata"]
    sources = training_sources(args)
    snapshot = base._load_json(args.memory_dir / "causal_snapshot_manifest.json")
    base._require_fields("ST training", metadata, {
        "metric": adapter.training_metric,
        "training_source_sha256": {str(p): file_sha256(p) for p in sources},
        "training_task_ids": sorted(snapshot["evaluation_ids"][:diagnostic_count(args)]),
        "calibration_fraction": args.calibration_fraction, "calibration_seed": 71,
        "kappa_grid": sorted({float(v) for v in args.kappa_grid.split(",")}),
    })
    feature = payload["feature_builder"]
    base._require_fields("ST features", feature, {"type": "team-exposure-features-v3"})
    base._require_fields("ST encoder", feature["encoder"], {"model": args.semantic_model,
        "revision": args.semantic_revision, "chunk_tokens": args.semantic_chunk_tokens})
    base._require_fields("ST calibrated policy", metadata["policy_calibration"], {
        "enabled": True, "max_drops": 1, "delta": args.delta,
        "max_changed_harm_rate": args.max_changed_harm_rate})
    fit_ids = set(metadata["fit_task_ids"])
    dev_ids = set(metadata["policy_calibration"]["calibration_task_ids"])
    all_ids = {str(i) for i in metadata["training_task_ids"]}
    if not fit_ids or not dev_ids or fit_ids & dev_ids or fit_ids | dev_ids != all_ids:
        raise ValueError("invalid ST fit/dev task split")
    adapter.validate_checkpoint(payload, args, snapshot)
    # Also verify weight tensors/shape/count before skipping the fitting stage.
    STUtilityEstimator.from_dict(payload["estimator"])
    return True


def audited_comparison(args, output):
    result = compare(baseline_dir(args) / "memory_gate.jsonl", output / "memory_gate.jsonl",
        bootstrap_samples=args.bootstrap_samples, seed=args.gate_evaluation_seed,
        replica_manifest=getattr(args, "st_replica_manifest", None))
    full._save(output / "comparison.json", result)
    if result["paired_tasks"] != args.final_claims or not result["paired_control_valid"] or not result["policy_valid"]:
        raise ValueError(f"invalid paired control; inspect {output / 'comparison.json'}")
    return result


def write_reports(suite, prepared):
    policies = list(full.POLICIES)
    if suite.seed0_only:
        policies = [(key, label) for key, label in policies
            if key in {"native_gmemory", "linear", "semantic_linear", "semantic_mlp"}
            or (key == "always_global_drop" and any(
                (getattr(args, "st_results_dir", args.output_dir) / key / "comparison.json").is_file()
                for _, args in prepared.values()))]
    policies.append(("semantic_st", "Semantic ST (direct U + InfoNCE)"))
    rows, details = [], {}
    for policy, label in policies:
        row = {"Policy": label}
        for name, (adapter, args) in prepared.items():
            root = (suite.output_dir / name / "evaluation" if policy == "semantic_st" else
                    baseline_dir(args) if policy == "native_gmemory" else
                    getattr(args, "st_results_dir", args.output_dir) / policy)
            metric = adapter.metrics[0]
            result = base._load_json(root / ("summary.json" if policy == "native_gmemory" else "comparison.json"))
            if result and policy != "native_gmemory":
                baseline = baseline_dir(args) / "memory_gate.jsonl"
                gate = root / "memory_gate.jsonl"
                valid = (result.get("paired_control_valid") and result.get("policy_valid", True)
                    and result.get("paired_tasks") == args.final_claims and gate.is_file() and baseline.is_file()
                    and result.get("baseline_log_sha256") == file_sha256(baseline)
                    and result.get("gate_log_sha256") == file_sha256(gate))
                if not valid:
                    result = None
            score = None if result is None else result[metric.summary_field if policy == "native_gmemory" else metric.gate_field]
            row[adapter.display_name] = None if score is None else 100 * score
            if result is not None:
                details.setdefault(name, {})[policy] = {"reused": policy != "semantic_st", "path": str(root), "result": result}
        rows.append(row)
    protocol = {name: {"collection_groups": 1 if suite.seed0_only else 2,
        "training_sources": [str(p) for p in training_sources(args)],
        "diagnostic_tasks": diagnostic_count(args), "repeats_per_group": args.repeats,
        "original_diagnostic_partition_tasks": args.training_claims,
        "final_test_offset": args.training_claims, "final_test_tasks": args.final_claims,
        "baseline": str(baseline_dir(args)), "source_endpoint": args.endpoint,
        "evaluation_endpoint": evaluation_args(args).endpoint}
        for name, (_, args) in prepared.items()}
    full._save(suite.output_dir / "st_results.json", {"schema": SCHEMA, "notes": NOTE,
        "data_protocol": protocol, "rows": rows, "details": details})
    fields = list(rows[0])
    with (suite.output_dir / "st_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[k]) if k == "Policy" else
            "not run / invalid" if row[k] is None else f"{row[k]:.2f}%" for k in fields) + " |")
    (suite.output_dir / "st_results.md").write_text("\n".join(lines) + "\n\n" + NOTE + "\n", encoding="utf-8")
    for name, (adapter, _) in prepared.items():
        write_dataset_metrics(suite.output_dir / name, adapter, policies, details.get(name, {}))


def write_dataset_metrics(root, adapter, policies, details):
    """Render all task-specific metrics, not just the cross-dataset primary score."""
    rows = []
    for method, label in policies:
        if method not in details:
            continue
        entry = details[method]
        result = entry["result"]
        baseline = method == "native_gmemory"
        primary = adapter.metrics[0]
        row = {"method": method, "Policy": label,
            **{m.name: result[m.summary_field if baseline else m.gate_field] for m in adapter.metrics},
            "primary_delta": 0.0 if baseline else result[primary.delta_field],
            "primary_delta_95ci": [0.0, 0.0] if baseline else result[primary.interval_field],
            "keep_rate": 1.0 if baseline else result["gate_mean_keep_rate"],
            "tasks_with_drop": 0 if baseline else result["tasks_with_drop"],
            "improved": 0 if baseline else result["drop_improved_tasks"],
            "harmed": 0 if baseline else result["drop_harmed_tasks"], "reused": entry["reused"]}
        rows.append(row)
    full._save(root / "st_metrics.json", {"dataset": adapter.name, "results": rows, "notes": NOTE})
    if not rows:
        return
    with (root / "st_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fields = ["Policy", *(m.label for m in adapter.metrics), "primary delta (pp)", "95% CI (pp)",
              "keep rate", "DROP tasks", "improved", "harmed"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lo, hi = row["primary_delta_95ci"]
        cells = [row["Policy"], *(f"{100 * row[m.name]:.2f}%" for m in adapter.metrics),
            f"{100 * row['primary_delta']:+.2f}", f"[{100 * lo:+.2f}, {100 * hi:+.2f}]",
            f"{100 * row['keep_rate']:.2f}%",
            *(str(row[k]) for k in ("tasks_with_drop", "improved", "harmed"))]
        lines.append("| " + " | ".join(cells) + " |")
    (root / "st_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    suite = parse_args(argv)
    prepared = {name: prepare_dataset(suite, name) for name in suite.datasets}
    config = full._jsonable({"schema": SCHEMA, "source_root": suite.source_root,
        "datasets": suite.datasets, "estimator": STUtilityEstimator(**st_settings(suite),
            ensemble_size=suite.ensemble_size, seed=suite.seed, l2=suite.l2).config,
        "device": suite.st_device,
        "experiments": {name: adapter.configuration(args) for name, (adapter, args) in prepared.items()},
        "semantic": {name: {k: getattr(args, k) for k in ("semantic_model", "semantic_revision", "semantic_chunk_tokens")}
                     for name, (_, args) in prepared.items()}})
    if suite.seed0_only:
        config.update(collection_protocol="seed0-only-v1", seed0_results_root=str(suite.seed0_results_root)
            if suite.seed0_results_root is not None else None)
    plans = {name: {**adapter.plan(args), "new_test_method": "semantic_st", "existing_gates_run": [],
                   "training_sources": [str(p) for p in training_sources(args)],
                   "baseline": str(baseline_dir(args)),
                   "evaluation_endpoint": evaluation_args(args).endpoint,
                   "replica_manifest": str(args.st_replica_manifest) if hasattr(args, "st_replica_manifest") else None,
                   "maximum_new_test_team_inferences": args.final_claims}
             for name, (adapter, args) in prepared.items()}
    if suite.seed0_only:
        for name, (_, args) in prepared.items():
            plans[name].pop("diagnostic_branches_two_seeds", None)
            plans[name].update(collection_groups=1, new_diagnostic_branches=0,
                sample_seeds=[args.sample_seed_base + repeat for repeat in range(args.repeats)],
                reused_results_root=str(suite.seed0_results_root) if suite.seed0_results_root else None)
            count = diagnostic_count(args)
            plans[name].update(gate_training_questions=count,
                diagnostic_branches_per_seed=count * (args.successful_topk + args.insights_topk) * args.repeats * 2,
                original_diagnostic_partition_questions=args.training_claims,
                unused_diagnostic_questions=args.training_claims - count,
                final_test_offset=args.training_claims)
    print(json.dumps({"schema": SCHEMA, "plan_only": suite.plan_only, "prepare_missing": suite.prepare_missing,
                      "datasets": plans, "notes": NOTE}, ensure_ascii=False, indent=2))
    if suite.plan_only:
        return plans
    manifest = suite.output_dir / "st_manifest.json"
    old = base._load_json(manifest)
    if old is not None and old != config:
        raise ValueError("ST experiment configuration changed; use original arguments or a fresh --output-dir")
    if old is None and suite.output_dir.exists() and any(suite.output_dir.iterdir()):
        raise ValueError("nonempty ST output has no manifest")
    full._save(manifest, config)
    full._save(suite.output_dir / "experiment_plan.json", plans)
    state = {"status": "running", "stages": {}, "current_stage": None, "error": None}
    current_stage, current_log, current_command = "initialize", None, []

    def stage(label, ready, command=None, action=None):
        nonlocal current_stage, current_log, current_command
        current_stage, current_command = label, full._strip_key(command or [])
        current_log = suite.output_dir / "logs" / (label.replace(":", "_") + ".log")
        state.update(current_stage=label)
        state["stages"][label] = "running"
        full._save(suite.output_dir / "st_progress.json", state)
        complete = ready() if ready else False
        if not complete:
            if command:
                base._run_command(label, current_command, current_log, api_key=args.api_key)
            elif action:
                action()
            if ready is not None and not ready():
                raise RuntimeError(f"stage {label} returned with incomplete artifacts")
        state["stages"][label] = "reused" if complete else "completed"
        full._save(suite.output_dir / "st_progress.json", state)
        progress.update(1)

    try:
        with tqdm(total=5 * len(prepared) + 1 + sum(hasattr(args, "st_replica_manifest")
                  for _, args in prepared.values()), desc="Full-data ST only", unit="stage") as progress:
            for name, (adapter, args) in prepared.items():
                def require_existing(check, description):
                    if check():
                        return True
                    if not suite.prepare_missing:
                        if suite.seed0_only:
                            raise FileNotFoundError(f"{name}: {description} missing/incomplete. "
                                "Seed0-only never collects data or runs a baseline; check --source-root "
                                "and --seed0-results-root for the completed original artifacts.")
                        raise FileNotFoundError(f"{name}: {description} missing/incomplete. No other method was run. "
                            "Use the correct --source-root, or --prepare-missing to prepare only diagnostics/baseline.")
                    return False

                stage(name + ":diagnostic", lambda: require_existing(lambda: diagnostics_ready(adapter, args), "full diagnostics"),
                      command=None if suite.seed0_only else adapter.diagnostic_command(args))
                native = baseline_dir(args)
                stage(name + ":baseline", lambda: require_existing(lambda: adapter.evaluation_complete(args, native, mode="always_keep"), "baseline"),
                      command=None if suite.seed0_only else adapter.evaluation_command(args, mode="always_keep", output=native,
                          resume=(native / "run_manifest.json").is_file()))
                output = suite.output_dir / name / "evaluation"
                if hasattr(args, "st_replica_manifest"):
                    stage(name + ":verify_evaluation_replica", None,
                        action=lambda: verify_evaluation_replica(args, output))
                identity_path = suite.output_dir / name / "reused_inputs.json"
                full.save_plan(identity_path, reuse_identity(args))
                stage(name + ":train_st", lambda: checkpoint_ready(suite, adapter, args),
                      command=train_command(suite, adapter, args))
                checkpoint = base._load_json(args.checkpoint)
                full._save(suite.output_dir / name / "training_summary.json", {
                    "schema": ESTIMATOR_SCHEMA, "config": checkpoint["estimator"]["config"],
                    "training_diagnostics": checkpoint["estimator"]["training_diagnostics"],
                    "loss_history": checkpoint["estimator"]["loss_history"],
                    "policy_calibration": checkpoint["training_metadata"]["policy_calibration"],
                    "fit_task_ids": checkpoint["training_metadata"]["fit_task_ids"],
                    "training_task_ids": checkpoint["training_metadata"]["training_task_ids"]})
                del checkpoint
                command = evaluation_command(adapter, args, output)
                stage(name + ":evaluate_st", lambda: adapter.evaluation_complete(
                    evaluation_args(args), output, mode="learned", command=command), command=command)
                stage(name + ":compare", None, action=lambda: audited_comparison(args, output))
                write_reports(suite, prepared)
            stage("reports", None, action=lambda: write_reports(suite, prepared))
        state.update(status="completed", current_stage=None)
        full._save(suite.output_dir / "st_progress.json", state)
        print((suite.output_dir / "st_results.md").read_text(encoding="utf-8"))
        for name, (adapter, _) in prepared.items():
            print(f"\n{adapter.display_name}: all metrics\n")
            print((suite.output_dir / name / "st_metrics.md").read_text(encoding="utf-8"))
        return suite.output_dir / "st_results.json"
    except BaseException as exc:
        trace = traceback.format_exc()
        if current_log:
            current_log.parent.mkdir(parents=True, exist_ok=True)
            with current_log.open("a", encoding="utf-8") as handle:
                handle.write("\n" + trace)
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error={
            "stage": current_stage, "type": type(exc).__name__, "message": str(exc),
            "exit_code": getattr(exc, "returncode", None),
            "command": current_command, "log_path": str(current_log), "traceback": trace,
            "log_tail": base._tail(current_log) if current_log and current_log.is_file() else None})
        full._save(suite.output_dir / "st_progress.json", state)
        raise


if __name__ == "__main__":
    main()
