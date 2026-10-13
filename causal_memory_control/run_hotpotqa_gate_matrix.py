"""Shared gate matrix engine; the legacy entry point defaults to HotpotQA."""

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

from . import run_hotpotqa_gate_comparison as base
from . import hotpotqa_experiment as evaluation
from .compare_gate_runs import file_sha256
from .dataset_adapters import HotpotQADatasetAdapter


ARMS = ("linear", "semantic_linear", "semantic_mlp", "jev_direct", "jev_decomposed")
DEFAULT_ARMS = ("linear", "semantic_linear", "semantic_mlp")


def parse_args(argv=None, *, adapter=None):
    adapter = adapter or HotpotQADatasetAdapter()
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("--matrix-dir",type=Path,default=adapter.default_matrix_dir)
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                        help="Default: linear,semantic_linear,semantic_mlp. Jev arms require explicit selection.")
    parser.add_argument("--jev-model",default="jev-1.13.0")
    parser.add_argument("--jev-endpoint",default="https://api.typesafe.ai/v1/systemone")
    parser.add_argument("--jev-adapter",default="causal_memory_control.jev_api:judge")
    parser.add_argument("--jev-adapter-version",default="typesafe-systemone-v1")
    parser.add_argument("--jev-threshold-grid",default="0,0.1,0.25,0.5,0.75,0.9,1")
    parser.add_argument("--retry-jev-errors",action="store_true")
    parser.add_argument("--help", "-h", action="store_true")
    args,rest = parser.parse_known_args(argv)
    if args.help:
        parser.print_help()
        print("\nShared data, split, model, embedding, MLP and calibration options:")
        try:
            adapter.parse_args(["--help"])
        except SystemExit:
            pass
        raise SystemExit(0)
    args.arms = list(dict.fromkeys(value.strip() for value in args.arms.split(",") if value.strip()))
    if not args.arms or set(args.arms)-set(ARMS):
        raise SystemExit(f"--arms must select from {ARMS}")
    try:
        thresholds = [float(value) for value in args.jev_threshold_grid.split(",")]
    except ValueError as exc:
        raise SystemExit("--jev-threshold-grid requires comma-separated numbers") from exc
    if not thresholds or not all(math.isfinite(value) and 0 <= value <= 1 for value in thresholds):
        raise SystemExit("Jev thresholds must be finite values in [0,1]")
    shared = adapter.parse_args(rest)
    shared.output_dir = args.matrix_dir.resolve()
    shared.data,shared.memory_dir,shared.diagnostic_results = shared.data.resolve(),shared.memory_dir.resolve(),shared.diagnostic_results.resolve()
    shared.semantic_cache = shared.semantic_cache.resolve() if shared.semantic_cache else shared.output_dir/"semantic_cache.sqlite"
    shared.estimator,shared.feature_version = "linear",2
    args.matrix_dir = shared.output_dir
    return args,shared


def _arm_args(args, arm):
    child = copy.copy(args)
    child.checkpoint = args.output_dir/"checkpoints"/f"{arm}.json"
    child.estimator = "mlp" if arm=="semantic_mlp" else "linear"
    child.feature_version = 3 if arm.startswith("semantic_") else 2
    return child


def _jev_config(matrix):
    return {"adapter":matrix.jev_adapter,"model_id":matrix.jev_model,
            "adapter_version":matrix.jev_adapter_version,"endpoint":matrix.jev_endpoint}


def _jev_complete(matrix,args,arm, *, adapter=None):
    adapter = adapter or HotpotQADatasetAdapter()
    if not args.checkpoint.exists():
        return False
    payload = base._load_json(args.checkpoint)
    if not payload or payload.get("schema")!=evaluation.CHECKPOINT_SCHEMA or "jev_policy" not in payload:
        raise RuntimeError(f"invalid Jev checkpoint: {args.checkpoint}")
    base._require_fields("Jev policy",payload["jev_policy"],{
        "mode":arm,"provider":_jev_config(matrix),"prompt_version":"memory-retention-judgments-v1",
    })
    sources = [args.diagnostic_results/f"seed{seed}{suffix}"/"branches.jsonl"
               for seed,suffix in ((args.sample_seed_base,""),(args.retest_seed_base,"_retest"))]
    metadata = payload["training_metadata"]
    snapshot = base._load_json(args.memory_dir/"causal_snapshot_manifest.json")
    if not snapshot:
        raise RuntimeError("Jev checkpoint exists without its frozen snapshot")
    base._require_fields("Jev calibration",metadata,{
        "metric":adapter.training_metric,"calibration_fraction":args.calibration_fraction,"calibration_seed":71,
        "training_sources":[str(path) for path in sources],
        "training_task_ids":sorted(snapshot["evaluation_ids"][:args.training_claims]),
        "training_source_sha256":{str(path):file_sha256(path) for path in sources},
        "threshold_grid":sorted({float(v) for v in matrix.jev_threshold_grid.split(",")}),
    })
    policy = metadata["policy_calibration"]
    base._require_fields("Jev calibrated policy",policy,{
        "enabled":True,"max_drops":1,"max_changed_harm_rate":args.max_changed_harm_rate,
        "recommended_threshold":payload["jev_policy"].get("threshold"),
    })
    threshold = policy["recommended_threshold"]
    if (isinstance(threshold,bool) or not isinstance(threshold,(int,float))
            or not math.isfinite(threshold) or not 0 <= threshold <= 1
            or policy.get("recommended_max_drops") not in (0,1)):
        raise RuntimeError("invalid Jev calibrated threshold or DROP budget")
    calibration_ids = {str(value) for value in policy.get("calibration_task_ids",[])}
    if not calibration_ids or not calibration_ids <= {str(value) for value in metadata["training_task_ids"]}:
        raise RuntimeError("invalid Jev calibration task IDs")
    adapter.validate_checkpoint(payload, args, snapshot)
    return True


def _jev_train_command(matrix,args,arm, *, adapter=None):
    adapter = adapter or HotpotQADatasetAdapter()
    sources = [args.diagnostic_results/f"seed{seed}{suffix}"/"branches.jsonl"
               for seed,suffix in ((args.sample_seed_base,""),(args.retest_seed_base,"_retest"))]
    return [sys.executable,"-u","-m","causal_memory_control.jev_gate",
        "--input",*(str(p) for p in sources),"--output",str(args.checkpoint),
        "--memory-dir",str(args.memory_dir),"--mode",arm,
        "--metric",adapter.training_metric,
        "--adapter",matrix.jev_adapter,"--model-id",matrix.jev_model,
        "--endpoint",matrix.jev_endpoint,"--adapter-version",matrix.jev_adapter_version,
        "--cache",str(args.output_dir/"jev_cache.sqlite"),
        "--calibration-fraction",str(args.calibration_fraction),
        "--threshold-grid",matrix.jev_threshold_grid,
        "--max-changed-harm-rate",str(args.max_changed_harm_rate)]


def _evaluation_command(matrix,args,arm,output, *, adapter=None):
    adapter = adapter or HotpotQADatasetAdapter()
    resume = (output/"run_manifest.json").is_file()
    if arm.startswith("jev_"):
        command = adapter.evaluation_command(args,mode="always_keep",output=output,resume=resume)
        if "--cache-seed-from" in command:
            start = command.index("--cache-seed-from")
            del command[start:start+2]
        command[command.index("--mode")+1] = arm
        payload = base._load_json(args.checkpoint)
        command[command.index("--max-drops")+1] = str(payload["training_metadata"]["policy_calibration"]["recommended_max_drops"])
        command += ["--checkpoint",str(args.checkpoint),"--jev-cache",str(args.output_dir/"jev_cache.sqlite"),
                    "--cache-seed-from",str(args.output_dir/"native_gmemory"/"llm_cache.sqlite")]
        if matrix.retry_jev_errors and resume:
            command.append("--retry-jev-errors")
        return command
    return adapter.evaluation_command(args,mode="learned",output=output,resume=resume)


def _evaluation_complete(matrix,args,arm,output, *, adapter=None):
    adapter = adapter or HotpotQADatasetAdapter()
    if arm.startswith("jev_"):
        if not adapter.evaluation_complete(args,output,mode=arm,
                command=_evaluation_command(matrix,args,arm,output,adapter=adapter)):
            return False
        summary = base._load_json(output/"summary.json")
        if matrix.retry_jev_errors and summary.get("jev_errors",0):
            return False
        return True
    return adapter.evaluation_complete(args,output,mode="learned")


def _comparison_complete(args,output):
    payload = base._load_json(output/"comparison.json")
    baseline_log = args.output_dir/"native_gmemory"/"memory_gate.jsonl"
    gate_log = output/"memory_gate.jsonl"
    return bool(payload and payload.get("paired_control_valid") and payload.get("policy_valid",True)
        and baseline_log.is_file() and gate_log.is_file()
        and payload.get("baseline_log_sha256")==file_sha256(baseline_log)
        and payload.get("gate_log_sha256")==file_sha256(gate_log))


def _report(root,arms, *, adapter=None):
    adapter = adapter or HotpotQADatasetAdapter()
    rows = []
    baseline = base._load_json(root/"native_gmemory"/"summary.json")
    baseline_row = {"method":"native_gmemory",
                 "keep_rate":1.0,"tasks_with_drop":0,"drop_improved_tasks":0,"drop_harmed_tasks":0,
                 "jev_errors":0,"jev_calls":0,"jev_cache_hits":0,"jev_input_tokens":0,"jev_output_tokens":0,
                 "jev_latency_seconds":0.0}
    def delta_name(metric):
        return metric.name.removeprefix("answer_") + "_delta"

    for metric in adapter.metrics:
        baseline_row.update({metric.name: baseline[metric.summary_field],
                             delta_name(metric): 0.0, delta_name(metric)+"_95ci": [0,0]})
    rows.append(baseline_row)
    for arm in arms:
        result = base._load_json(root/arm/"comparison.json")
        row = {"method":arm,
                     "keep_rate":result["gate_mean_keep_rate"],
                     **{k:result[k] for k in ("tasks_with_drop","drop_improved_tasks","drop_harmed_tasks","jev_errors",
                         "jev_calls","jev_cache_hits","jev_input_tokens","jev_output_tokens","jev_latency_seconds")}}
        for metric in adapter.metrics:
            row.update({metric.name: result[metric.gate_field],
                        delta_name(metric): result[metric.delta_field],
                        delta_name(metric)+"_95ci": result[metric.interval_field]})
        rows.append(row)
    base._write_json(root/"matrix_results.json",{"dataset":adapter.name,
        "benchmark":adapter.benchmark,"training_metric":adapter.training_metric,
        "arms":arms,"results":rows})
    with (root/"matrix_results.csv").open("w",newline="",encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    primary = adapter.metrics[0]
    include_jev = any(arm.startswith("jev_") for arm in arms)
    columns = ["method", *(metric.label for metric in adapter.metrics), f"{primary.label} delta",
               "95% CI", "keep rate", "DROP tasks", "improved", "harmed"]
    if include_jev:
        columns.append("Jev errors")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        cells = [row["method"], *(f"{row[metric.name]:.4f}" for metric in adapter.metrics),
                 f"{row[delta_name(primary)]:+.4f}", str(row[delta_name(primary)+"_95ci"]),
                 f"{row['keep_rate']:.4f}", *(str(row[key]) for key in (
                     "tasks_with_drop", "drop_improved_tasks", "drop_harmed_tasks"))]
        if include_jev:
            cells.append(str(row["jev_errors"]))
        lines.append("| " + " | ".join(cells) + " |")
    (root/"matrix_results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return rows


def main(argv=None, *, adapter=None):
    adapter = adapter or HotpotQADatasetAdapter()
    matrix,args = parse_args(argv, adapter=adapter)
    adapter.resolve_plan(args)
    adapter.validate_args(args)
    configuration = {"schema":adapter.matrix_schema,"shared":adapter.configuration(args),
                     "matrix":{k:str(v) if isinstance(v,Path) else v for k,v in vars(matrix).items()
                               if k not in ("retry_jev_errors","help")}}
    if type(adapter) is not HotpotQADatasetAdapter:
        configuration["dataset_adapter"] = {"name":adapter.name,"version":adapter.version,
            "class":f"{type(adapter).__module__}:{type(adapter).__qualname__}",
            "benchmark":adapter.benchmark,"training_metric":adapter.training_metric}
    # Semantic fields are deliberately omitted from legacy v2 identities, but
    # are mandatory for this new multi-arm experiment.
    configuration["semantic"] = {k:str(getattr(args,k)) for k in (
        "semantic_model","semantic_revision","semantic_cache","semantic_chunk_tokens",
        "mlp_hidden_sizes","mlp_epochs","mlp_learning_rate","mlp_batch_tasks","mlp_device")}
    identity = hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest()
    plan = {"dataset":adapter.name,"experiment_plan":adapter.plan(args),"arms":matrix.arms,
            "final_team_inferences":args.final_claims*(1+len(matrix.arms)),
            "max_jev_judgments":sum(arm.startswith("jev_") for arm in matrix.arms)*
                (round(args.training_claims*args.calibration_fraction)+args.final_claims)*(args.successful_topk+args.insights_topk)}
    print(json.dumps(plan,ensure_ascii=False,indent=2))
    if args.plan_only:
        return args.output_dir/"matrix_results.json"
    root = args.output_dir
    manifest = root/"matrix_manifest.json"
    if manifest.exists():
        existing = base._load_json(manifest)
        if not existing or existing.get("configuration_hash")!=identity:
            raise SystemExit("matrix arguments changed; use the original command or a new --matrix-dir")
    else:
        if root.exists() and any(root.iterdir()):
            raise SystemExit("non-empty matrix directory has no manifest")
        base._write_json(manifest,{"configuration_hash":identity,"configuration":configuration})
    state = {"status":"running","stages":{},"error":None}
    current_stage,current_command,current_log = "initialization",[],None

    def run_stage(stage,command,complete):
        nonlocal current_stage,current_command,current_log
        current_stage,current_command,current_log = stage,command,root/"logs"/f"{stage.replace(':','_')}.log"
        if complete():
            state["stages"][stage] = "completed"
            return
        state["current_stage"],state["stages"][stage] = stage,"running"
        base._write_json(root/"matrix_progress.json",state)
        base._run_command(stage,command,current_log,api_key=args.api_key)
        if not complete():
            raise RuntimeError(f"stage {stage} exited but artifacts are incomplete")
        state["stages"][stage] = "completed"
        base._write_json(root/"matrix_progress.json",state)

    try:
        total = 2+3*len(matrix.arms)
        with tqdm(total=total,desc=f"{adapter.display_name} gate methods",unit="stage") as progress:
            run_stage("diagnostic",adapter.diagnostic_command(args),lambda:adapter.diagnostic_complete(args))
            progress.update(1)
            native = root/"native_gmemory"
            run_stage("native_gmemory",adapter.evaluation_command(args,mode="always_keep",output=native,
                resume=(native/"run_manifest.json").exists()),
                lambda:adapter.evaluation_complete(args,native,mode="always_keep"))
            progress.update(1)
            for arm in matrix.arms:
                child,output = _arm_args(args,arm),root/arm
                if arm.startswith("jev_"):
                    train_command = _jev_train_command(matrix,child,arm,adapter=adapter)
                    check = lambda:_jev_complete(matrix,child,arm,adapter=adapter)
                else:
                    train_command,check = adapter.train_command(child),lambda:adapter.checkpoint_complete(child)
                run_stage(f"{arm}:train",train_command,check)
                progress.update(1)
                run_stage(f"{arm}:evaluate",_evaluation_command(matrix,child,arm,output,adapter=adapter),
                          lambda:_evaluation_complete(matrix,child,arm,output,adapter=adapter))
                progress.update(1)
                command = [sys.executable,"-u","-m","causal_memory_control.compare_gate_runs",
                    "--baseline",str(native/"memory_gate.jsonl"),"--gate",str(output/"memory_gate.jsonl"),
                    "--bootstrap-samples",str(args.bootstrap_samples),"--seed",str(args.gate_evaluation_seed),
                    "--output",str(output/"comparison.json"),"--require-no-drop-match","--require-policy-success"]
                run_stage(f"{arm}:compare",command,lambda:_comparison_complete(args,output))
                progress.update(1)
        rows = _report(root,matrix.arms,adapter=adapter)
        state.update(status="completed",current_stage=None)
        base._write_json(root/"matrix_progress.json",state)
        print(json.dumps(rows,ensure_ascii=False,indent=2))
        return root/"matrix_results.json"
    except BaseException as exc:
        details = {"stage":current_stage,"type":type(exc).__name__,"message":str(exc),
                   "command":current_command,"log_path":str(current_log),
                   "log_tail":base._tail(current_log) if current_log and current_log.exists() else None,
                   "traceback":traceback.format_exc()}
        state.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",error=details)
        base._write_json(root/"matrix_progress.json",state)
        raise


if __name__ == "__main__":
    main()
