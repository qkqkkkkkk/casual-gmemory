"""New, independent CSM-ST method transfer to frozen GMemory FEVER/HotpotQA.

Never trains/runs old ST, Linear, MLP, Jev or a missing baseline. CE labels use
a frozen single-generator proxy; evaluation still uses the original MAS host.
This is NOT a reproduction of the paper's FlashRAG benchmark numbers.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

from . import run_st_gate_benchmark as reuse
from .compare_gate_runs import file_sha256
from .csm_st_data import digest, save_json


SCHEMA = "csm-st-gmemory-benchmark-v1"
NOTE = ("CSM-ST method transfer, not original FlashRAG score reproduction. Trainable BERT + "
        "3-layer/8-head list self-attention + direct CI head; CE-difference supervision, "
        "fit-only downsampling/recombination/reweighting/InfoNCE. CE uses a single frozen HF "
        "generator proxy, not the joint likelihood or reward of the multi-agent host. "
        "Keep all predicted CI>0 contexts; unlimited drops and semantic dedup, unlike old max-1 gates. "
        "Original frozen memory and held-out test boundary are unchanged. "
        "FEVER binary Accuracy (not evidence-aware FEVER Score); HotpotQA Answer F1/EM. "
        "Old methods/baseline are read-only and never rerun.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path, default=Path("causal_memory_control/results/full_gate_benchmark_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("causal_memory_control/results/csm_st_gmemory_v1"))
    parser.add_argument("--datasets", default="fever,hotpotqa")
    for name in ("fever", "hotpotqa"):
        parser.add_argument(f"--{name}-results-root", type=Path,
            help="Registered old seed0 matrix (including subset/shards), reused read-only")
        parser.add_argument(f"--{name}-old-st-root", type=Path, help="Optional old ST benchmark root, report only")
    parser.add_argument("--evaluation-endpoint", help="Optional verified Ollama replica for new evaluations only")
    parser.add_argument("--stage", choices=("all", "labels", "train", "evaluate", "report"), default="all")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--acknowledge-ce-proxy", action="store_true",
        help="Acknowledge single-generator CE/possible HF vs Ollama quantization mismatch, not team-reward CI")
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--generator-revision")
    parser.add_argument("--ce-device", default="cuda:0")
    parser.add_argument("--ce-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--ce-max-length", type=int, default=32768)
    parser.add_argument("--ce-reduction", choices=("mean", "sum"), default="mean")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--encoder-model", default="bert-base-uncased")
    parser.add_argument("--encoder-revision")
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--evaluation-device", default="cpu", help="CSM only; Ollama device is configured on its own service")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-tasks", type=int, default=16)
    parser.add_argument("--pair-microbatch", type=int, default=8)
    parser.add_argument("--max-pair-tokens", type=int, default=512)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0, help="InfoNCE tau, not generator sampling temperature")
    parser.add_argument("--positive-epsilon", type=float, default=0.1)
    parser.add_argument("--negative-epsilon", type=float, default=0.3)
    parser.add_argument("--rarity-alpha", type=float, default=10.0)
    parser.add_argument("--trivial-threshold", type=float, default=5.0)
    parser.add_argument("--hard-threshold", type=float, default=5.0)
    parser.add_argument("--rarity-bin-width", type=float, default=1.0)
    parser.add_argument("--trivial-keep-fraction", type=float, default=0.2)
    parser.add_argument("--intervention-gamma", type=float, default=0.1)
    parser.add_argument("--query-max-cosine", type=float, default=0.3)
    parser.add_argument("--interventions-per-hard", type=int, default=1)
    parser.add_argument("--dedup-threshold", type=float, default=0.98)
    args = parser.parse_args(argv)
    args.datasets = list(dict.fromkeys(v.strip() for v in args.datasets.split(",") if v.strip()))
    if not args.datasets or set(args.datasets) - {"fever", "hotpotqa"}:
        parser.error("--datasets must select fever,hotpotqa")
    import math

    numeric = (args.beta, args.temperature, args.encoder_learning_rate, args.head_learning_rate,
        args.weight_decay, args.rarity_alpha, args.trivial_threshold, args.hard_threshold,
        args.rarity_bin_width, args.trivial_keep_fraction, args.positive_epsilon, args.negative_epsilon,
        args.intervention_gamma, args.query_max_cosine, args.dedup_threshold)
    if not all(math.isfinite(v) for v in numeric):
        parser.error("all numeric settings must be finite")
    if (min(args.epochs, args.batch_tasks, args.pair_microbatch) < 1 or args.max_pair_tokens < 8
            or args.ce_max_length < 8 or args.interventions_per_hard < 0):
        parser.error("invalid training/context/intervention count")
    if (args.beta < 0 or args.temperature <= 0 or args.weight_decay < 0 or args.rarity_alpha < 0
            or min(args.encoder_learning_rate, args.head_learning_rate, args.rarity_bin_width) <= 0
            or not 0 < args.positive_epsilon < args.negative_epsilon
            or not 0 < args.trivial_keep_fraction <= 1 or not -1 <= args.query_max_cosine <= 1
            or not 0 < args.dedup_threshold <= 1 or args.intervention_gamma < 0
            or args.trivial_threshold > args.hard_threshold):
        parser.error("invalid CSM-ST loss/sampling settings")
    args.source_root, args.output_dir = args.source_root.resolve(), args.output_dir.resolve()
    roots = [args.source_root]
    for name in ("fever", "hotpotqa"):
        for suffix in ("results_root", "old_st_root"):
            value = getattr(args, f"{name}_{suffix}")
            if value is not None:
                value = value.resolve()
                setattr(args, f"{name}_{suffix}", value)
                roots.append(value)
    if any(args.output_dir == r or args.output_dir in r.parents or r in args.output_dir.parents for r in roots):
        parser.error("new output must be separate and non-nested with all original roots")
    if not (args.source_root / "benchmark_manifest.json").is_file():
        parser.error("--source-root requires the original benchmark_manifest.json")
    if args.stage in ("all", "labels") and not args.plan_only and not args.acknowledge_ce_proxy:
        parser.error("new CE labels require --acknowledge-ce-proxy; inspect --plan-only first")
    return args


def prepare(suite, name):
    # Read-only source/subset import shared with the old runner. Its training
    # entry, estimator, losses and runtime are never called or modified.
    source = SimpleNamespace(source_root=suite.source_root, output_dir=suite.output_dir,
        seed0_only=True, seed0_results_root=getattr(suite, f"{name}_results_root"),
        evaluation_endpoint=suite.evaluation_endpoint, shared_argv=[])
    adapter, args = reuse.prepare_dataset(source, name)
    if args.failed_topk != 0:
        raise ValueError("CSM transfer currently requires the registered failed_topk=0 protocol")
    args.csm_baseline_args = copy.copy(args)
    args.kappa, args.delta, args.max_drops = 0.0, 0.0, -1
    return adapter, args


def settings(suite):
    training_names = ("seed", "epochs", "batch_tasks", "encoder_learning_rate", "head_learning_rate",
        "weight_decay", "beta", "temperature", "positive_epsilon", "negative_epsilon", "rarity_alpha",
        "trivial_threshold", "hard_threshold", "rarity_bin_width", "trivial_keep_fraction", "local_files_only")
    return {**{key: getattr(suite, key) for key in ("generator_model", "generator_revision", "ce_device",
        "ce_dtype", "ce_max_length", "ce_reduction", "local_files_only", "intervention_gamma",
        "query_max_cosine", "interventions_per_hard", "evaluation_device", "evaluation_endpoint")},
        "model_config": {"encoder": suite.encoder_model, "revision": suite.encoder_revision,
            "global_layers": 3, "heads": 8, "mlp_hidden": 256, "dropout": 0.1,
            "max_pair_tokens": suite.max_pair_tokens, "pair_microbatch": suite.pair_microbatch,
            "gradient_checkpointing": True},
        "training": {**{key: getattr(suite, key) for key in training_names}, "device": suite.train_device}}


def source_identity(args):
    # A new replica proof is an OUTPUT, not a pre-existing source artifact.
    source = copy.copy(args)
    if hasattr(source, "st_replica_manifest"):
        del source.st_replica_manifest
    return reuse.reuse_identity(source)


def dataset_config(suite, adapter, args, *, with_hashes):
    snapshot = reuse.base._load_json(args.memory_dir / "causal_snapshot_manifest.json")
    if not snapshot:
        raise FileNotFoundError("registered frozen snapshot is missing; it will not be rebuilt")
    count = reuse.diagnostic_count(args)
    kinds = {kind: list(range(1, n + 1)) for kind, n in (
        ("trajectory", args.successful_topk), ("insight", args.insights_topk)) if n}
    return {"sources": [str(p) for p in reuse.training_sources(args)],
        "diagnostic_ids": [str(v) for v in snapshot["evaluation_ids"][:count]],
        "support_ids": [str(v) for v in snapshot["support_ids"]],
        "test_ids": [str(v) for v in snapshot["evaluation_ids"][args.training_claims:args.evaluation_count]],
        "original_test_offset": args.training_claims, "dev_fraction": args.calibration_fraction,
        "baseline": str(reuse.baseline_dir(args)), "source_model": args.model,
        "context": adapter.checkpoint_context(args, snapshot),
        "candidate_kinds": list(kinds), "candidate_ranks": sorted({v for ranks in kinds.values() for v in ranks}),
        "candidate_kind_ranks": kinds, "dedup": {"model": args.semantic_model,
            "revision": args.semantic_revision, "chunk_tokens": args.semantic_chunk_tokens,
            "threshold": suite.dedup_threshold},
        "input_sha256": source_identity(args) if with_hashes else None}


def labels_ready(root, config):
    manifest = reuse.base._load_json(root / "labels_manifest.json")
    if manifest is None:
        return False
    if manifest.get("config_digest") != digest(config):
        raise ValueError("CE labels belong to another configuration")
    for path, expected in manifest["artifact_sha256"].items():
        if file_sha256(Path(path)) != expected:
            raise ValueError(f"CE artifact changed: {path}")
    if file_sha256(root / "generator_identity.json") != manifest["generator_identity_sha256"]:
        raise ValueError("generator identity changed")
    return True


def checkpoint_ready(adapter, args, config):
    payload = reuse.base._load_json(args.checkpoint)
    if payload is None:
        return False
    if payload.get("csm_st", {}).get("schema") != "csm-st-local-global-v1":
        raise ValueError("not a new CSM-ST checkpoint; old ST is not interchangeable")
    if payload["training_metadata"].get("config_digest") != digest(config):
        raise ValueError("CSM checkpoint configuration changed")
    artifact = args.checkpoint.parent / "model"
    for path, expected in payload["csm_st"]["artifact_sha256"].items():
        target = artifact / path
        if artifact.resolve() not in target.resolve().parents or file_sha256(target) != expected:
            raise ValueError("CSM model artifact changed")
    adapter.validate_checkpoint(payload, args, reuse.base._load_json(args.memory_dir / "causal_snapshot_manifest.json"))
    return True


def evaluation_commands(suite, adapter, args, name):
    output = suite.output_dir / name / "evaluation"
    host = reuse.evaluation_command(adapter, args, output)
    command = [*host[:3], "causal_memory_control.csm_st_evaluate", "--dataset", name,
               "--csm-device", suite.evaluation_device, *host[4:]]
    return host, command


def reports(suite, prepared):
    policies = [("native_gmemory", "Always Use (baseline)"), ("always_global_drop", "Always Global Drop"),
        ("linear", "Linear Gate"), ("semantic_linear", "Semantic Linear Gate"),
        ("semantic_mlp", "Semantic MLP"), ("semantic_st", "Old Semantic ST (unchanged)"),
        ("csm_st", "CSM-ST Local–Global (CE proxy)")]
    table, details = [], {}
    for method, label in policies:
        row = {"Policy": label}
        for name, (adapter, args) in prepared.items():
            if method == "csm_st":
                root = suite.output_dir / name / "evaluation"
            elif method == "semantic_st":
                old = getattr(suite, f"{name}_old_st_root")
                root = old / name / "evaluation" if old else None
            else:
                root = (reuse.baseline_dir(args) if method == "native_gmemory" else
                    getattr(args, "st_results_dir", args.output_dir) / method)
            result = None if root is None else reuse.base._load_json(root / (
                "summary.json" if method == "native_gmemory" else "comparison.json"))
            if result and method != "native_gmemory":
                gate = root / "memory_gate.jsonl"
                valid = (result.get("paired_control_valid") and result.get("policy_valid", True)
                    and result.get("paired_tasks") == args.final_claims and gate.is_file()
                    and result.get("baseline_log_sha256") == file_sha256(reuse.baseline_dir(args) / "memory_gate.jsonl")
                    and result.get("gate_log_sha256") == file_sha256(gate))
                if not valid:
                    result = None
            for metric in adapter.metrics[:1 if name == "fever" else 2]:
                key = f"{adapter.display_name} {metric.label}"
                row[key] = None if result is None else 100 * result[
                    metric.summary_field if method == "native_gmemory" else metric.gate_field]
            if result is not None:
                details.setdefault(name, {})[method] = {"path": str(root), "reused": method != "csm_st", "result": result}
        table.append(row)
    save_json(suite.output_dir / "csm_st_results.json", {"schema": SCHEMA, "notes": NOTE,
        "rows": table, "details": details})
    columns = list(table[0])
    with (suite.output_dir / "csm_st_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(str(row[c]) if c == "Policy" else
        "—" if row[c] is None else f"{row[c]:.2f}%" for c in columns) + " |" for row in table]
    (suite.output_dir / "csm_st_results.md").write_text("\n".join(lines) + "\n\n" + NOTE + "\n", encoding="utf-8")
    return table


def main(argv=None):
    suite = parse_args(argv)
    prepared = {name: prepare(suite, name) for name in suite.datasets}
    config = {"schema": SCHEMA, "source_root": str(suite.source_root), "output_dir": str(suite.output_dir),
        "settings": settings(suite), "experiments": {name: dataset_config(suite, adapter, args, with_hashes=False)
            for name, (adapter, args) in prepared.items()}, "notes": NOTE}
    plan = {name: {"diagnostic_tasks": len(data["diagnostic_ids"]), "final_test_tasks": len(data["test_ids"]),
        "final_test_offset": data["original_test_offset"], "baseline": data["baseline"], "sources": data["sources"],
        "new_original_ce_forwards_before_dedup": len(data["diagnostic_ids"]) * (1 + sum(len(r) for r in data["candidate_kind_ranks"].values())),
        "new_augmented_ce_forwards": "data dependent; recomputed, never copied from original lists",
        "new_reward_rollouts": 0, "new_test_team_inferences_at_most": len(data["test_ids"]),
        "other_methods_run": [], "selection_rule": "keep_iff_ci_gt_zero; unlimited drops"}
        for name, data in config["experiments"].items()}
    print(json.dumps({"schema": SCHEMA, "plan_only": suite.plan_only, "stage": suite.stage,
        "datasets": plan, "settings": config["settings"], "notes": NOTE}, ensure_ascii=False, indent=2))
    if suite.plan_only:
        return plan
    # No source mutation and no expensive CE calls before strict artifact preflight.
    for name, (adapter, args) in prepared.items():
        reuse.diagnostics_ready(adapter, args)
        if not adapter.evaluation_complete(args.csm_baseline_args, reuse.baseline_dir(args), mode="always_keep"):
            raise FileNotFoundError(f"{name}: completed original baseline required; it is never rerun")
        config["experiments"][name] = dataset_config(suite, adapter, args, with_hashes=True)
    manifest = suite.output_dir / "csm_st_manifest.json"
    previous = reuse.base._load_json(manifest)
    if previous is not None and previous != config:
        raise ValueError("CSM configuration or source files changed; use original arguments/a fresh output directory")
    if previous is None and suite.output_dir.exists() and any(suite.output_dir.iterdir()):
        raise ValueError("refusing a nonempty output without its CSM manifest")
    save_json(manifest, config)
    save_json(suite.output_dir / "experiment_plan.json", plan)
    state = {"status": "running", "current_stage": None, "stages": {}}
    progress_path = suite.output_dir / "csm_st_progress.json"
    try:
        for name, (adapter, args) in prepared.items():
            root = suite.output_dir / name
            for stage in ("labels", "train", "evaluate"):
                if suite.stage not in ("all", stage):
                    continue
                label = f"{name}:{stage}"
                state["current_stage"] = label
                save_json(progress_path, state)
                if stage == "labels":
                    ready = labels_ready(root, config)
                    command = [sys.executable, "-u", "-m", "causal_memory_control.csm_st_worker",
                        "--config", str(manifest), "--dataset", name, "--stage", "labels"]
                elif stage == "train":
                    if not labels_ready(root, config):
                        raise FileNotFoundError("new CE labels required; old reward labels cannot substitute")
                    ready = checkpoint_ready(adapter, args, config)
                    command = [sys.executable, "-u", "-m", "causal_memory_control.csm_st_worker",
                        "--config", str(manifest), "--dataset", name, "--stage", "train"]
                else:
                    if not labels_ready(root, config) or not checkpoint_ready(adapter, args, config):
                        raise FileNotFoundError("new CSM-ST labels/checkpoint required")
                    output = root / "evaluation"
                    if hasattr(args, "st_replica_manifest"):
                        reuse.verify_evaluation_replica(args, output)
                    host, command = evaluation_commands(suite, adapter, args, name)
                    ready = adapter.evaluation_complete(reuse.evaluation_args(args), output, mode="learned", command=host)
                state["stages"][label] = "reused" if ready else "running"
                save_json(progress_path, state)
                if not ready:
                    reuse.base._run_command(label, command, suite.output_dir / "logs" / f"{name}_{stage}.log",
                                             api_key=args.api_key)
                if stage == "labels" and not labels_ready(root, config):
                    raise RuntimeError("label worker exited without complete labels")
                if stage == "train" and not checkpoint_ready(adapter, args, config):
                    raise RuntimeError("training worker exited without a complete checkpoint")
                if stage == "evaluate":
                    if not adapter.evaluation_complete(reuse.evaluation_args(args), root / "evaluation", mode="learned", command=host):
                        raise RuntimeError("CSM evaluation incomplete")
                    reuse.audited_comparison(args, root / "evaluation")
                state["stages"][label] = "reused" if ready else "completed"
                save_json(progress_path, state)
            if suite.stage == "report" and (root / "evaluation" / "memory_gate.jsonl").is_file():
                reuse.audited_comparison(args, root / "evaluation")
        reports(suite, prepared)
        # Detect source/cache changes rather than silently accepting mixed runs.
        for name, (_, args) in prepared.items():
            if source_identity(args) != config["experiments"][name]["input_sha256"]:
                raise ValueError("read-only source changed during CSM experiment")
        state.update(status="completed", current_stage=None)
        save_json(progress_path, state)
        print((suite.output_dir / "csm_st_results.md").read_text())
        return suite.output_dir / "csm_st_results.json"
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        save_json(progress_path, state)
        raise


if __name__ == "__main__":
    main()
