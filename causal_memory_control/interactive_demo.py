"""Generate visibly synthetic nine-policy tables without data, models or APIs."""
from __future__ import annotations

import argparse
from pathlib import Path

from .csm_st_data import write_rows
from .interactive_data import DOMAINS, digest, pddl_cross_validation, save
from .interactive_protocol import learned_methods, policies
from .run_interactive_gate_benchmark import dataset_jobs


def _record(config, task, method, success, *, drops=1):
    outcome = {"success": float(success), "raw_reward": float(success), "progress": float(success),
        "won": bool(success), "done": bool(success), "steps": 1, "invalid_actions": 0,
        "termination_reason": "simulated"}
    return {"simulation": True, "policy": method, "task_id": task["id"], "seed": 2000,
        "config_digest": digest(config), "retrieval_fingerprint": "sim-retrieval:" + task["id"],
        "initial_fingerprint": "sim-initial:" + task["id"], "retrieved_count": 6,
        "dropped_count": drops, "drop_ids": [f"sim-memory-{i}" for i in range(drops)],
        "action_keys": ["sim-action:" + method + ":" + task["id"]], "outcome": outcome}


def generate(output, *, alf_unseen=134, alf_extra=500):
    from .interactive_report import report

    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("simulation requires a new empty output directory")
    pddl = []
    for domain, count in zip(DOMAINS, (20, 10, 20, 10)):
        for index in range(count):
            identity = f"sim:pddl:{domain}:{index:02d}"
            pddl.append({"id": identity, "group": identity, "dataset": "pddl", "domain": domain,
                         "difficulty": "easy" if index % 2 == 0 else "hard"})
    folds = pddl_cross_validation(pddl)
    alf, split, cohorts = [], {p: [] for p in ("support", "fit", "dev", "test")}, {}
    for part, count in (("support", 100), ("fit", 800), ("dev", 200),
                        ("valid_unseen", alf_unseen), ("train_scene_holdout", alf_extra)):
        selected = []
        for index in range(count):
            identity = f"sim:alfworld:{part}:{index:04d}"
            alf.append({"id": identity, "group": f"sim:scene:{part}:{index % 4}", "dataset": "alfworld",
                        "task_type": ("put", "examine", "clean", "heat", "cool", "puttwo")[index % 6]})
            selected.append(identity)
        if part in split:
            split[part] = selected
        else:
            cohorts[part] = selected
            split["test"] += selected
    config = {"schema": "interactive-simulation-v1", "simulation": True,
        "include_local_global_team_u": True, "output_dir": str(output), "evaluation_seed": 2000,
        "random_seeds": [11, 23, 47], "experiments": {
            "pddl": {"tasks": pddl, "split": folds[0], "cv_splits": folds, "protocol": "SIMULATED_five_fold"},
            "alfworld": {"tasks": alf, "split": split, "protocol": "SIMULATED_two_cohorts",
                         "primary_test_ids": cohorts["valid_unseen"], "test_cohorts": cohorts}}}
    save(output / "benchmark_manifest.json", config)
    # These arbitrary success counts illustrate formatting, not performance forecasts.
    counts = {
        "pddl": [30, 26, 30, 32, 33, 34, 35, 36, 42],
        "valid_unseen": [50, 42, 49, 52, 54, 56, 55, 53, 70],
        "train_scene_holdout": [220, 200, 218, 230, 240, 245, 250, 248, 300],
    }
    cohorts_by_task = {t: part for part, ids in cohorts.items() for t in ids}
    by_id = {t["id"]: t for t in pddl + alf}
    pddl_order = {t["id"]: i for i, t in enumerate(sorted(pddl, key=lambda t: t["id"]))}
    for name in config["experiments"]:
        for _, child in dataset_jobs(config, name):
            root = Path(child["output_dir"])
            save(root / "benchmark_manifest.json", child)
            save(root / name / "snapshot.json", {"simulation": True, "config_digest": digest(child)})
            for method in learned_methods(child):
                save(root / name / method / "checkpoint.json", {"simulation": True, "method": method})
            records = []
            for task_id in child["experiments"][name]["split"]["test"]:
                task = by_id[task_id]
                cohort = "pddl" if name == "pddl" else cohorts_by_task[task_id]
                index = pddl_order[task_id] if name == "pddl" else cohorts[cohort].index(task_id)
                total = 60 if name == "pddl" else len(cohorts[cohort])
                reference_total = {"pddl": 60, "valid_unseen": 134, "train_scene_holdout": 500}[cohort]
                for method_index, (method, _) in enumerate(policies(child)):
                    cutoff = round(counts[cohort][method_index] * total / reference_total)
                    if method == "random_drop":
                        seeds = []
                        for offset, seed in zip((-1, 0, 1), config["random_seeds"]):
                            value = int(index < max(0, min(total, cutoff + offset)))
                            seeds.append(value)
                            records.append(_record(child, task, f"random_drop_seed{seed}", value))
                        value = sum(seeds) / len(seeds)
                    else:
                        value = int(index < cutoff)
                    records.append(_record(child, task, method, value,
                        drops=0 if method == "always_use" else 6 if method == "global_drop" else 2 if method in (
                            "csm_st", "local_global_team_u") else 1))
            write_rows(root / name / "evaluation/policies.jsonl", records)
    summaries = report(config)
    text = ("**模拟结果：人工构造数据，仅演示表格格式；不是实测或效果预测。单位：%。**\n\n"
            f"| Policy | PDDL 五折 OOF（60） | ALFWorld unseen（{alf_unseen}） | ALFWorld 额外留出（{alf_extra}） |\n"
            "| --- | ---: | ---: | ---: |\n")
    for method, label in policies(config):
        p = summaries["pddl"]["policies"][method]
        a = summaries["alfworld"]["policies"][method]
        extra = a["cohort_results"]["train_scene_holdout"]
        text += f"| {label} | {100*p['success']:.2f} | {100*a['success']:.2f} | {100*extra['success']:.2f} |\n"
    (output / "example_results.md").write_text(text)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    generate(args.output_dir)
    print((args.output_dir / "example_results.md").read_text())


if __name__ == "__main__":
    main()
