"""New FEVER/HotpotQA Local--Global + measured Team U experiment.

Read registered seed0 diagnostic branches and frozen memory, fit a new BERT /
list-attention / scalar utility model, then evaluate only that model. The old ST
and CE methods are independent. Full-list Team U scoring is followed by the CE
method's semantic deduplication. No CE forwards, Qwen training, diagnostic
collection or candidate recombination are performed.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
from pathlib import Path
import sys
import traceback

from . import run_csm_st_benchmark as csm
from .compare_gate_runs import file_sha256
from .csm_st_data import digest, save_json
from .local_global_team_u import SCHEMA, checkpoint_spec, deployment_dedup, validate_labels


NOTE = ("Local--Global architecture with measured paired Team U = mean(R_USE)-mean(R_SINGLE_DROP). "
        "Trainable BERT + 3-layer/8-head candidate self-attention + one unbounded utility head. "
        "Original candidate lists retained for measured targets and model scoring. "
        "Semantic deduplication follows scoring, using the CE method's frozen model/greedy rule; "
        "unlike CE, the predictor sees the original list, not an already-deduplicated list. "
        "No recombination or new label collection. "
        "Rarity-weighted MSE + same-list InfoNCE; reward-scale thresholds are explicit adaptations. "
        "Keep predicted U>0 nonduplicate memories; unlimited drops and no matched-keep-rate calibration. "
        "Single-drop labels do not identify joint multi-drop reward. "
        "FEVER binary Accuracy; HotpotQA Answer F1/EM. No generator fine-tuning or CE scoring.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path, default=Path("causal_memory_control/results/full_gate_benchmark_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("causal_memory_control/results/local_global_team_u_dedup_v1"))
    parser.add_argument("--datasets", default="fever,hotpotqa")
    for dataset in ("fever", "hotpotqa"):
        parser.add_argument(f"--{dataset}-results-root", type=Path,
                            help="Registered seed0 matrix/subset/shards and completed baseline")
    parser.add_argument("--evaluation-endpoint", help="Optional independently verified Ollama replica")
    parser.add_argument("--stage", choices=("all", "labels", "train", "evaluate", "report"), default="all")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--encoder-model", default="bert-base-uncased")
    parser.add_argument("--encoder-revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--evaluation-device", default="cpu")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-tasks", type=int, default=16)
    parser.add_argument("--pair-microbatch", type=int, default=8)
    parser.add_argument("--max-pair-tokens", type=int, default=512)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0, help="InfoNCE temperature")
    parser.add_argument("--positive-epsilon", type=float, default=0.05)
    parser.add_argument("--negative-epsilon", type=float, default=0.25)
    parser.add_argument("--rarity-alpha", type=float, default=10.0)
    parser.add_argument("--hard-threshold", type=float, default=0.1)
    parser.add_argument("--rarity-bin-width", type=float, default=0.25)
    parser.add_argument("--dedup-threshold", type=float, default=0.98,
                        help="CE-style semantic cosine threshold; applied after full-list Team U prediction")
    args = parser.parse_args(argv)
    args.datasets = list(dict.fromkeys(v.strip() for v in args.datasets.split(",") if v.strip()))
    if not args.datasets or set(args.datasets) - {"fever", "hotpotqa"}:
        parser.error("--datasets must select fever,hotpotqa")
    if min(args.epochs, args.batch_tasks, args.pair_microbatch) < 1 or args.max_pair_tokens < 8:
        parser.error("training counts must be positive and max-pair-tokens >=8")
    numbers = (args.encoder_learning_rate, args.head_learning_rate, args.weight_decay, args.beta,
        args.temperature, args.positive_epsilon, args.negative_epsilon, args.rarity_alpha,
        args.hard_threshold, args.rarity_bin_width, args.dedup_threshold)
    if (not all(math.isfinite(v) for v in numbers) or min(args.encoder_learning_rate,
            args.head_learning_rate, args.temperature, args.rarity_bin_width) <= 0
            or min(args.beta, args.weight_decay, args.rarity_alpha) < 0
            or not 0 < args.positive_epsilon < args.negative_epsilon <= 2
            or not 0 < args.dedup_threshold <= 1):
        parser.error("invalid Team U loss/optimization settings")
    args.source_root, args.output_dir = args.source_root.resolve(), args.output_dir.resolve()
    roots = [args.source_root]
    for dataset in ("fever", "hotpotqa"):
        value = getattr(args, f"{dataset}_results_root")
        if value is not None:
            value = value.resolve()
            setattr(args, f"{dataset}_results_root", value)
            roots.append(value)
    if any(args.output_dir == root or args.output_dir in root.parents or root in args.output_dir.parents for root in roots):
        parser.error("output must be separate and non-nested with every original result root")
    if not (args.source_root / "benchmark_manifest.json").is_file():
        parser.error("--source-root requires the recorded full benchmark manifest")
    return args


def settings(suite):
    names = ("seed", "epochs", "batch_tasks", "encoder_learning_rate", "head_learning_rate", "weight_decay",
             "beta", "temperature", "positive_epsilon", "negative_epsilon", "rarity_alpha", "hard_threshold",
             "rarity_bin_width", "local_files_only")
    return {"model_config": {"encoder": suite.encoder_model, "revision": suite.encoder_revision,
        "global_layers": 3, "heads": 8, "mlp_hidden": 256, "dropout": 0.1,
        "max_pair_tokens": suite.max_pair_tokens, "pair_microbatch": suite.pair_microbatch,
        "gradient_checkpointing": True}, "training": {**{name: getattr(suite, name) for name in names},
        "device": suite.train_device, "trivial_threshold": suite.hard_threshold, "trivial_keep_fraction": 1.0},
        "evaluation_device": suite.evaluation_device, "evaluation_endpoint": suite.evaluation_endpoint,
        "selection_rule": "keep_iff_team_u_gt_zero_and_not_duplicate",
        "deduplication": True, "dedup_placement": "post_score", "recombination": False}


def dataset_config(suite, adapter, args, hashes):
    # Source import/partition validation is shared; CE labels and settings are not.
    data = csm.dataset_config(suite, adapter, args, with_hashes=hashes)
    data["training_metric"] = adapter.training_metric
    return data


def checkpoint_ready(adapter, args, config):
    if not args.checkpoint.is_file():
        return False
    raw = json.loads(args.checkpoint.read_text())
    # A killed training adapter can leave the shared loop's intermediate file;
    # only this run's own intermediate checkpoint may trigger a fresh fit.
    if ("local_global_team_u" not in raw and "csm_st" in raw
            and raw.get("training_metadata", {}).get("config_digest") == digest(config)
            and raw["training_metadata"].get("estimator", {}).get("type") == "local_global_team_u"):
        return False
    payload, spec, _ = checkpoint_spec(args.checkpoint)
    if payload["training_metadata"].get("config_digest") != digest(config):
        raise ValueError("Team U checkpoint belongs to another configuration")
    provenance = json.loads((args.checkpoint.parent / "labels_manifest.json").read_text())
    if spec["dedup"] != deployment_dedup(provenance):
        raise ValueError("checkpoint dedup policy differs from its recorded training protocol")
    adapter.validate_checkpoint(payload, args, csm.reuse.base._load_json(args.memory_dir / "causal_snapshot_manifest.json"))
    return True


def evaluation_commands(suite, adapter, args, dataset):
    output = suite.output_dir / dataset / "evaluation"
    host = csm.reuse.evaluation_command(adapter, args, output)
    return host, [*host[:3], "causal_memory_control.local_global_team_u", "--dataset", dataset,
        "--stage", "evaluate", "--device", suite.evaluation_device, *host[4:]]


def report(suite, prepared):
    rows, details = [], {}
    for method, label in (("native_gmemory", "Always Use (baseline)"),
                           ("local_global_team_u", "Local–Global + Team U + semantic dedup")):
        row = {"Policy": label}
        for dataset, (adapter, args) in prepared.items():
            baseline = csm.reuse.baseline_dir(args)
            output = baseline if method == "native_gmemory" else suite.output_dir / dataset / "evaluation"
            result = csm.reuse.base._load_json(output / ("summary.json" if method == "native_gmemory" else "comparison.json"))
            if result and method != "native_gmemory":
                valid = (result.get("paired_control_valid") and result.get("policy_valid", True)
                    and result.get("paired_tasks") == args.final_claims
                    and result.get("baseline_log_sha256") == file_sha256(baseline / "memory_gate.jsonl")
                    and result.get("gate_log_sha256") == file_sha256(output / "memory_gate.jsonl"))
                if not valid:
                    raise ValueError("invalid Team U comparison; inspect its paired audit")
            for metric in adapter.metrics[:1 if dataset == "fever" else 2]:
                row[f"{adapter.display_name} {metric.label}"] = (None if result is None else
                    100 * result[metric.summary_field if method == "native_gmemory" else metric.gate_field])
            if result:
                details.setdefault(dataset, {})[method] = {"path": str(output), "result": result}
        rows.append(row)
    save_json(suite.output_dir / "team_u_results.json", {"schema": SCHEMA, "notes": NOTE, "rows": rows, "details": details})
    columns = list(rows[0])
    with (suite.output_dir / "team_u_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(str(row[c]) if c == "Policy" else "—" if row[c] is None else
                               f"{row[c]:.2f}%" for c in columns) + " |" for row in rows]
    (suite.output_dir / "team_u_results.md").write_text("\n".join(lines) + "\n\n" + NOTE + "\n", encoding="utf-8")


def main(argv=None):
    suite = parse_args(argv)
    prepared = {dataset: csm.prepare(suite, dataset) for dataset in suite.datasets}
    config = {"schema": SCHEMA, "source_root": str(suite.source_root), "output_dir": str(suite.output_dir),
        "settings": settings(suite), "notes": NOTE,
        "experiments": {name: dataset_config(suite, adapter, args, False) for name, (adapter, args) in prepared.items()}}
    plan = {name: {"diagnostic_tasks": len(data["diagnostic_ids"]), "final_test_tasks": len(data["test_ids"]),
        "final_test_offset": data["original_test_offset"], "sources": data["sources"], "baseline": data["baseline"],
        "label_metric": data["training_metric"], "new_label_team_rollouts": 0, "ce_forwards": 0,
        "new_test_team_inferences_at_most": len(data["test_ids"]), "other_methods_run": [],
        "selection_rule": "keep_iff_team_u_gt_zero_and_not_duplicate; unlimited drops",
        "deduplication": True, "dedup_placement": "post_score", "dedup": data["dedup"],
        "matched_keep_rate": False,
        "recombination": False} for name, data in config["experiments"].items()}
    print(json.dumps({"schema": SCHEMA, "plan_only": suite.plan_only, "datasets": plan,
                      "settings": config["settings"], "notes": NOTE}, ensure_ascii=False, indent=2), flush=True)
    if suite.plan_only:
        return plan
    for name, (adapter, args) in prepared.items():
        csm.reuse.diagnostics_ready(adapter, args)
        if not adapter.evaluation_complete(args.csm_baseline_args, csm.reuse.baseline_dir(args), mode="always_keep"):
            raise FileNotFoundError(f"{name}: completed original baseline is required")
        config["experiments"][name] = dataset_config(suite, adapter, args, True)
    config["code_sha256"] = {name: file_sha256(Path(__file__).with_name(name)) for name in
        (Path(__file__).name, "local_global_team_u.py", "csm_st_model.py", "csm_st_training.py", "csm_st_data.py",
         "csm_st_worker.py")}
    manifest = suite.output_dir / "team_u_manifest.json"
    if not manifest.exists() and suite.output_dir.exists() and any(suite.output_dir.iterdir()):
        raise ValueError("use a new Team U output directory")
    suite.output_dir.mkdir(parents=True, exist_ok=True)
    with (suite.output_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another Team U runner is using this output directory") from exc
        csm.reuse.full.save_plan(manifest, config)
        save_json(suite.output_dir / "experiment_plan.json", plan)
        state = {"status": "running", "current_stage": None, "stages": {}}
        progress = suite.output_dir / "team_u_progress.json"
        try:
            for dataset, (adapter, args) in prepared.items():
                root = suite.output_dir / dataset
                for stage in ("labels", "train", "evaluate"):
                    if suite.stage not in ("all", stage):
                        continue
                    label = f"{dataset}:{stage}"
                    state.update(current_stage=label)
                    if stage == "labels":
                        ready = validate_labels(root, config)
                    else:
                        if not validate_labels(root, config):
                            raise FileNotFoundError("extract original Team U labels first (--stage labels)")
                        ready = checkpoint_ready(adapter, args, config)
                        if stage == "evaluate":
                            if not ready:
                                raise FileNotFoundError("train the Local--Global Team U checkpoint first")
                            if hasattr(args, "st_replica_manifest"):
                                csm.reuse.verify_evaluation_replica(args, root / "evaluation")
                            host, command = evaluation_commands(suite, adapter, args, dataset)
                            ready = adapter.evaluation_complete(csm.reuse.evaluation_args(args),
                                root / "evaluation", mode="learned", command=host)
                    if stage != "evaluate":
                        command = [sys.executable, "-u", "-m", "causal_memory_control.local_global_team_u",
                                   "--config", str(manifest), "--dataset", dataset, "--stage", stage]
                    state["stages"][label] = "reused" if ready else "running"
                    save_json(progress, state)
                    if not ready:
                        csm.reuse.base._run_command(label, command,
                            suite.output_dir / "logs" / f"{dataset}_{stage}.log", api_key=args.api_key)
                    if stage == "labels" and not validate_labels(root, config):
                        raise RuntimeError("Team U label extraction is incomplete")
                    if stage == "train" and not checkpoint_ready(adapter, args, config):
                        raise RuntimeError("Team U training is incomplete")
                    if stage == "evaluate":
                        if not adapter.evaluation_complete(csm.reuse.evaluation_args(args), root / "evaluation",
                                                           mode="learned", command=host):
                            raise RuntimeError("Team U evaluation is incomplete")
                        csm.reuse.audited_comparison(args, root / "evaluation")
                    state["stages"][label] = "reused" if ready else "completed"
                    save_json(progress, state)
                if suite.stage == "report" and (root / "evaluation" / "memory_gate.jsonl").is_file():
                    if not checkpoint_ready(adapter, args, config):
                        raise ValueError("report requires a complete Team U checkpoint")
                    csm.reuse.audited_comparison(args, root / "evaluation")
                if csm.source_identity(args) != config["experiments"][dataset]["input_sha256"]:
                    raise ValueError("read-only source artifacts changed during the Team U experiment")
            report(suite, prepared)
            state.update(status="completed", current_stage=None)
            save_json(progress, state)
            print((suite.output_dir / "team_u_results.md").read_text())
            return suite.output_dir / "team_u_results.json"
        except BaseException as exc:
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
            save_json(progress, state)
            raise


if __name__ == "__main__":
    main()
