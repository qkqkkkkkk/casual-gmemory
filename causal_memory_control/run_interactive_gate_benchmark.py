"""PDDL/ALFWorld benchmark: eight policies plus optional Local–Global Team U."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

from .interactive_data import SCHEMA, dependencies, digest, inventory, locked, save, sha, workload
from .interactive_protocol import STAGES, policies, stages, training_options


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output-dir", type=Path, default=Path("causal_memory_control/results/pddl_alfworld_gate_v1"))
    p.add_argument("--datasets", default="pddl,alfworld")
    p.add_argument("--stage", choices=("all", "smoke", *STAGES, "labels_team_u", "train_team_u"), default="all")
    p.add_argument("--include-local-global-team-u", action="store_true",
                   help="Add the ninth policy: measured Team U, Local–Global model and post-score semantic dedup")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--endpoint", default="http://127.0.0.1:11434/v1")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--generator-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--generator-revision")
    p.add_argument("--encoder-model", default="bert-base-uncased")
    p.add_argument("--ce-device", default="cuda:0")
    p.add_argument("--train-device", default="cuda:0")
    p.add_argument("--evaluation-device", default="cpu")
    p.add_argument("--ce-max-length", type=int, default=32768)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--few-shots", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--successful-topk", type=int, default=3)
    p.add_argument("--insights-topk", type=int, default=3)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--split-seed", type=int, default=71)
    p.add_argument("--diagnostic-seed", type=int, default=0)
    p.add_argument("--evaluation-seed", type=int, default=2000)
    p.add_argument("--random-seeds", default="11,23,47")
    p.add_argument("--ensemble-size", type=int, default=12)
    p.add_argument("--alf-support", type=int, default=50)
    p.add_argument("--alf-fit", type=int, default=80)
    p.add_argument("--alf-dev", type=int, default=20)
    p.add_argument("--alf-heldout-test", type=int, default=0,
                   help="Extra test tasks from reserved train scenes, reported separately from 134 unseen tasks")
    p.add_argument("--pddl-folds", type=int, choices=(1, 5), default=1,
                   help="1 preserves the small pilot; 5 runs independent outer-fold experiments and reports OOF results")
    p.add_argument("--expert-timeout", type=int, default=120)
    a = p.parse_args(argv)
    a.datasets = list(dict.fromkeys(a.datasets.split(",")))
    try:
        a.random_seeds = [int(v) for v in a.random_seeds.split(",")]
    except ValueError:
        p.error("--random-seeds must be distinct integers")
    if not a.datasets or set(a.datasets) - {"pddl", "alfworld"}:
        p.error("only pddl,alfworld are supported; old datasets cannot be executed here")
    if a.stage in ("labels_team_u", "train_team_u") and not a.include_local_global_team_u:
        p.error("Team U stages require --include-local-global-team-u")
    if len(set(a.random_seeds)) != len(a.random_seeds) or len(a.random_seeds) < 2:
        p.error("declare at least two distinct random seeds")
    if (min(a.max_steps, a.repeats, a.alf_support, a.alf_fit, a.alf_dev, a.expert_timeout) < 1
            or a.alf_heldout_test < 0 or a.ensemble_size < 2 or a.few_shots < 0 or min(a.successful_topk, a.insights_topk) < 0
            or a.successful_topk + a.insights_topk < 1 or a.ce_max_length < 8
            or not math.isfinite(a.temperature) or not 0 <= a.temperature <= 2):
        p.error("invalid episode, sampling, retrieval or training settings")
    a.repo = a.repo.resolve()
    a.output_dir = a.output_dir.resolve()
    return a


def configuration(args):
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ("plan_only", "stage", "datasets", "alf_support", "alf_fit", "alf_dev")}
    config["schema"] = SCHEMA
    config["experiments"] = inventory(args.repo, args.datasets, args.split_seed,
                                     (args.alf_support, args.alf_fit, args.alf_dev), alf_heldout=args.alf_heldout_test,
                                     pddl_folds=args.pddl_folds)
    source_files = list((args.repo / "causal_memory_control").glob("interactive_*.py"))
    source_files += [Path(__file__).resolve()]
    source_files += [args.repo / p for p in (
        "causal_memory_control/runtime_gate.py", "causal_memory_control/estimator.py",
        "causal_memory_control/neural_estimator.py", "causal_memory_control/st_estimator.py",
        "causal_memory_control/semantic_features.py", "causal_memory_control/csm_st_model.py",
        "causal_memory_control/csm_st_training.py", "causal_memory_control/csm_st_data.py",
        "causal_memory_control/csm_st_runtime.py", "causal_memory_control/csm_st_scoring.py",
        "causal_memory_control/local_global_team_u.py", "causal_memory_control/csm_st_worker.py",
        "causal_memory_control/train_gate.py", "causal_memory_control/features.py",
        "causal_memory_control/gmemory_adapter.py", "causal_memory_control/task_state.py",
        "causal_diagnostic/oracle_recipient/seeded_client.py")]
    for directory in ("tasks/mas_workflow/macnet", "tasks/prompts", "mas"):
        source_files += list((args.repo / directory).rglob("*.py"))
    source_files += [args.repo / "tasks/envs/pddl_env/pddl_env.py"]
    source_files += list((args.repo / "tasks/envs/pddl_env/pddlgym").rglob("*.py"))
    config["code_sha256"] = {str(p.relative_to(args.repo)): sha(p) for p in sorted(set(source_files))}
    from importlib.metadata import PackageNotFoundError, version
    config["installed_versions"] = {}
    for package in ("numpy", "torch", "transformers", "sentence-transformers", "alfworld", "textworld",
                    "gym", "nltk", "chromadb", "openai", "fast-downward-textworld"):
        try:
            config["installed_versions"][package] = version(package)
        except PackageNotFoundError:
            config["installed_versions"][package] = None
    config["protocol"] = {
        "primary_metric": "success_explicit_won", "pddl_secondary": "historical_max_goal_progress",
        "retrieval_boundary": "one_native_retrieval_at_episode_start_successful_and_insights_failed0",
        "oracle_actions": "KEEP ALL + every SINGLE DROP + DROP ALL",
        "random_budget": "one SINGLE DROP; matched seed prompt cache; average predeclared random seeds",
        "csm_target": "replay_verified_successful_native_full_plan_from_public_initial_context",
        "csm_ce": "frozen_HF_mean_continuation_CE_including_EOS_no_truncation",
        "csm_action": "dedup_then_keep_iff_predicted_CI_gt_0_unlimited_drops",
        "pddl_action_parser": "native_domain_arguments_hyphen_preserving_regex_v1",
        "reward_gates": "existing_semantic_linear_three_head_MLP_and_direct_U_ST_beta001_seed23",
        "test_tuning": False, "jev": False, "large_model_downloads": False,
        "old_datasets": "FEVER/HotpotQA never executed by this entry point"}
    if args.include_local_global_team_u:
        config["protocol"]["local_global_team_u"] = {
            "label": "mean_success_KEEP_ALL-minus-mean_success_SINGLE_DROP; paired_original_lists",
            "architecture": "same_CSMConfig_BERT_3_layers_8_heads_unbounded_head",
            "action": "score_full_list_then_semantic_dedup_keep_iff_predicted_U_gt_0_unlimited_drops",
            "dedup_threshold": 0.98, "kappa": 0.0, "delta": 0.0, "keep_all_fallback": False,
            "training_candidate_deduplication": False, "cross_task_recombination": False,
            "new_label_rollouts": 0, "ce_forwards": 0,
            "training": training_options(config, team_u=True),
            "comparison": "matches_QA_Team_U; conditioning_and_reward_scale_adaptations_are_not_a_pure_CE_label_ablation"}
    return config


def plan(config, args):
    deps = dependencies()
    return {"schema": config["schema"], "output_dir": config["output_dir"], "stage": args.stage,
        "plan_only": True, "writes_or_model_loads_or_service_calls": 0,
        "workload": workload(config["experiments"], config["successful_topk"] + config["insights_topk"], config["repeats"],
                             include_team_u=config.get("include_local_global_team_u", False)),
        "smoke_additional_schedules": (2*sum(len(v.get("cv_splits", [None])) for v in config["experiments"].values())
                                       if args.stage == "smoke" else 0),
        "dependencies": deps, "missing_dependencies": [k for k, v in deps.items() if not v],
        "local_model_paths_exist": {key: Path(config[key]).is_dir() for key in
                                  ("generator_model", "encoder_model", "embedding_model")},
        "unverified_until_runtime": ["Ollama digest/template/parameters", "HF local weights/tokenizer identity",
            "all selected expert plans replay successfully", "all PDDL annotation goals match native registration",
            "support successes yield nonempty candidate lists", "full-snapshot retrieval/no-DROP parity"],
        "protocol": config["protocol"], "config_digest_without_service_identity": digest(config),
        "resume": "same configuration resumes task/action files; any split/model/seed/code/checkpoint change rejected",
        "stages": list(stages(config)), "policies": [m for m, _ in policies(config)],
        "cost_note": "No wall-time promise. Schedules are multi-agent multi-step, not API calls."}


def dataset_jobs(config, name):
    """Each CV fold owns its manifest, memory, prompt cache and checkpoints."""
    data = config["experiments"][name]
    if "cv_splits" not in data:
        return [(name, config)]
    jobs = []
    for fold, split in enumerate(data["cv_splits"]):
        child = deepcopy(config)
        child["output_dir"] = str(Path(config["output_dir"]) / "pddl_folds" / f"fold_{fold:02d}")
        child["experiments"] = {name: {k: v for k, v in data.items() if k != "cv_splits"}}
        child["experiments"][name]["split"] = split
        child["experiments"][name]["protocol"] = "registered60_outer_fold_success"
        child["cv_context"] = {"fold": fold, "outer_folds": len(data["cv_splits"]),
                               "parent_config_digest": digest(config), "test_outcomes_used_for_tuning": False}
        jobs.append((f"{name}/fold_{fold:02d}", child))
    return jobs


def main(argv=None):
    args = parse_args(argv)
    config = configuration(args)
    if args.plan_only:
        print(json.dumps(plan(config, args), ensure_ascii=False, indent=2))
        return
    # All model loading remains local. No model download is triggered by this
    # benchmark; the user prepares the weights on their server separately.
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from .interactive_runtime import model_identity

    config["model_identity"] = model_identity(config["endpoint"], config["model"])
    target = args.output_dir / "benchmark_manifest.json"
    if args.output_dir.exists() and not target.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output directory has unrelated files; select a new independent directory")
    locked(target, config)
    save(args.output_dir / "plan.json", plan(config, args))
    selected_stages = stages(config) if args.stage == "all" else (args.stage,)
    try:
        for stage in selected_stages:
            if stage == "report":
                from .interactive_report import report
                report(config)
                continue
            for name in config["experiments"]:
                for label, job in dataset_jobs(config, name):
                    job_root = Path(job["output_dir"])
                    job_manifest = job_root / "benchmark_manifest.json"
                    locked(job_manifest, job)
                    command = [sys.executable, "-u", "-m", "causal_memory_control.interactive_worker",
                        "--config", str(job_manifest), "--dataset", name, "--stage", stage]
                    save(args.output_dir / "pipeline_progress.json", {"stage": stage, "dataset": label,
                        "command": command, "status": "running"})
                    logs = job_root / name / "logs"
                    logs.mkdir(parents=True, exist_ok=True)
                    with (logs / (stage + ".log")).open("a") as stream:
                        subprocess.run(command, check=True, cwd=args.repo, stdout=stream, stderr=subprocess.STDOUT)
                    print(f"completed {label}/{stage}; logs: {logs / (stage + '.log')}", flush=True)
        save(args.output_dir / "pipeline_progress.json", {"status": "completed", "stages": list(selected_stages)})
    except Exception as exc:
        save(args.output_dir / "pipeline_progress.json", {"status": "failed", "stage": stage,
            "error": str(exc), "details": "dataset errors/*.json and logs/<stage>.log"})
        raise


if __name__ == "__main__":
    main()
