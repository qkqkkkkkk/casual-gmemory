"""Back up and explicitly continue the known v10 experiment with empty lists.

Original manifests and action identities remain unchanged. A separate,
hash-bound registration records the reviewed implementation upgrade. No source
hash is silently replaced and no support/diagnostic/evaluation result is erased.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .interactive_data import digest, locked, save, sha
from .interactive_progress import STAGE_NAMES, run_logged, show_progress
from .interactive_protocol import stages
from .resume_interactive_dataset import backup_results, experiment_lock
from .run_interactive_gate_benchmark import dataset_jobs


# Only these reviewed stage changes are compatible with v10 measurements.
# Native prompts, actions, memory writes, seeds and LLM calls are unchanged.
UPGRADE = {
    "causal_memory_control/interactive_worker.py": (
        "6481014752eeb934ee64981b978703e24a6e7a7a5f952b57a2855aa62d073574",
        "dac39b50d7714da7f35df7c2245135dd0f112ccb510538d873ee578113ce65c2"),
    "causal_memory_control/interactive_training.py": (
        "7afda4faa4f757f6d29949b0e2b282eb2dcc28731bc3cb8bf61877b971a58422",
        "396099eb9c0e2b997258a43d83f59b6d12d45449959c60b7251ca1d48ed044ea"),
    "causal_memory_control/interactive_team_u.py": (
        "728cf1649e583229d93d85c9e180a9c5f79be4c9a4fb230a90a47471cebf1ac3",
        "a7569ac3a9dc70adad76969f6dc8e6caa46741f0f7708935e3cdebbd70c47921"),
    "causal_memory_control/interactive_report.py": (
        "76f217fa8ce8a85f2927e18c94c9dc376bc076729cba5017a2dc10ef456c73f8",
        "6cbaa9eeda9f1ae0d48f554b75d55865482ed8e4419a47217dfee9786d4dc3ba"),
}
HELPERS = ("causal_memory_control/interactive_empty_memory.py",
           "causal_memory_control/continue_interactive_empty_memory.py",
           "causal_memory_control/resume_interactive_dataset.py")


def upgrade_registration(config):
    repo = Path(config["repo"])
    if (repo.resolve() != Path(__file__).resolve().parents[1]
            or not config.get("exclude_ce") or not config.get("include_local_global_team_u")):
        raise ValueError("upgrade requires the original eight-policy no-CE repository/configuration")
    for path, (before, after) in UPGRADE.items():
        if config["code_sha256"].get(path) != before or sha(repo / path) != after:
            raise ValueError(f"unsupported source/target implementation for empty-memory upgrade: {path}")
    for path, expected in config["code_sha256"].items():
        if path not in UPGRADE and sha(repo / path) != expected:
            raise ValueError(f"unrelated registered implementation changed: {path}")
    return {"schema": "interactive-empty-memory-upgrade-v1", "config_digest": digest(config),
        "original_code_sha256": config["code_sha256"],
        "effective_code_sha256": {path: sha(repo / path) for path in sorted(set(config["code_sha256"]) | set(HELPERS))},
        "compatibility": "native episodes and action identities unchanged; empty lists retained; no fabricated supervision",
        "original_manifest_immutable": True}


def preflight(root):
    config = json.loads((root / "benchmark_manifest.json").read_text())
    if Path(config["output_dir"]).resolve() != root:
        raise ValueError("registered output directory differs from requested root")
    registration = upgrade_registration(config)
    jobs = []
    for name, data in config["experiments"].items():
        for task in data["tasks"]:
            for path, expected in task["files"].items():
                if sha(Path(config["repo"]) / path) != expected:
                    raise ValueError(f"registered task input changed: {path}")
        for label, child in dataset_jobs(config, name):
            directory = Path(child["output_dir"])
            manifest = directory / "benchmark_manifest.json"
            if manifest.exists():
                if json.loads(manifest.read_text()) != child:
                    raise ValueError(f"registered fold manifest changed: {manifest}")
            elif directory.exists() and any(directory.iterdir()):
                raise ValueError(f"existing fold data lacks its original manifest: {directory}")
            record = upgrade_registration(child)
            prior = directory / "empty_memory_upgrade/registration.json"
            if prior.exists() and json.loads(prior.read_text()) != record:
                raise ValueError(f"existing upgrade registration differs: {prior}")
            jobs.append((name, label, child, record))
    return config, registration, jobs


def continue_run(root, *, datasets=("pddl", "alfworld"), stage="all", plan_only=False):
    root = Path(root).resolve()

    def prepare():
        config, registration, jobs = preflight(root)
        selected = tuple(dict.fromkeys(datasets))
        if not selected or set(selected) - set(config["experiments"]):
            raise ValueError("select registered datasets only")
        selected_stages = stages(config) if stage == "all" else (stage,)
        if any(s not in (*stages(config), "smoke") for s in selected_stages):
            raise ValueError("stage is unavailable in the no-CE experiment")
        return config, registration, jobs, selected, selected_stages

    if plan_only:
        config, _, jobs, selected, selected_stages = prepare()
        plan = {"output_dir": str(root), "datasets": selected, "stages": selected_stages,
            "jobs": [label for name, label, _, _ in jobs if name in selected],
            "config_digest": digest(config), "backup_before_upgrade": True,
            "original_manifests_preserved": True, "completed_action_cache_reused": True,
            "model_or_service_calls": 0, "writes": 0}
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return plan
    with experiment_lock(root):
        config, registration, jobs, selected, selected_stages = prepare()
        backup = backup_results(root, "empty-memory-upgrade", config)
        for _, _, child, record in jobs:
            directory = Path(child["output_dir"])
            locked(directory / "benchmark_manifest.json", child)
            locked(directory / "empty_memory_upgrade/registration.json", record)
        locked(root / "empty_memory_upgrade/registration.json", registration)
        state_path = root / "empty_memory_upgrade/progress.json"
        state = {"config_digest": digest(config), "datasets": selected, "backup_dir": str(backup),
                 "stages": selected_stages, "completed": [], "status": "running"}
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("[续跑] 空候选允许执行；保留原 manifest、support、已测分支和测试分母。", flush=True)
        try:
            for index, current in enumerate(selected_stages):
                state.update(stage=current, status="running")
                save(state_path, state)
                show_progress("空记忆兼容续跑", index, len(selected_stages), STAGE_NAMES[current])
                if current == "report":
                    from .interactive_report import report
                    report(config, datasets=selected)
                else:
                    for name, label, child, _ in jobs:
                        if name not in selected:
                            continue
                        directory = Path(child["output_dir"])
                        print(f"[执行] {label}/{current}（已完成任务与分支自动复用）", flush=True)
                        command = [sys.executable, "-u", "-X", "faulthandler", "-m",
                            "causal_memory_control.interactive_worker", "--config",
                            str(directory / "benchmark_manifest.json"), "--dataset", name, "--stage", current]
                        run_logged(command, directory / name / "logs" / (current + ".log"), cwd=Path(config["repo"]))
                state["completed"].append(current)
                save(state_path, state)
            state["status"] = "completed"
            save(state_path, state)
            show_progress("空记忆兼容续跑", len(selected_stages), len(selected_stages), "本次命令已完成")
        except BaseException as exc:
            state.update(status="failed", error=str(exc) or type(exc).__name__)
            save(state_path, state)
            raise
        return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", default="pddl,alfworld")
    parser.add_argument("--stage", choices=("all", "support", "collect", "labels_team_u", "train_reward",
                                          "train_team_u", "evaluate", "report", "smoke"), default="all")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    continue_run(args.output_dir, datasets=tuple(args.datasets.split(",")), stage=args.stage, plan_only=args.plan_only)


if __name__ == "__main__":
    main()
