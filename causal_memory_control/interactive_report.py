"""Strict paired reports from measured, provenance-bound interactive logs."""
from __future__ import annotations

from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import random
import statistics

from .interactive_data import digest, locked, save, sha
from .interactive_protocol import POLICIES, learned_methods, policies


def interval(values, *, groups=None, seed=71, samples=2000):
    """Percentile paired bootstrap, clustered by scene/problem when supplied."""
    if not values:
        raise ValueError("cannot bootstrap an empty task set")
    if groups is None:
        groups = list(range(len(values)))
    buckets = defaultdict(list)
    for value, group in zip(values, groups):
        buckets[group].append(value)
    keys = sorted(buckets, key=str)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        selected = [v for _ in keys for v in buckets[rng.choice(keys)]]
        draws.append(statistics.mean(selected))
    draws.sort()
    return [draws[int(.025 * (samples-1))], draws[int(.975 * (samples-1))]]


def policy_statistics(ids, values, baseline, tasks, name):
    ordered, base = [values[t] for t in ids], [baseline[t] for t in ids]
    groups = [tasks[t]["group"] for t in ids]
    rewards = [r["outcome"]["success"] for r in ordered]
    effects = [a["outcome"]["success"]-b["outcome"]["success"] for a, b in zip(ordered, base)]
    grouped = {}
    for field in (("domain", "difficulty") if name == "pddl" else ("task_type",)):
        bins = defaultdict(list)
        for row in ordered:
            bins[tasks[row["task_id"]][field]].append(row["outcome"]["success"])
        grouped[field] = {k: {"n": len(v), "success": statistics.mean(v)} for k, v in bins.items()}
    kept = sum(r["retrieved_count"]-r["dropped_count"] for r in ordered)
    total = sum(r["retrieved_count"] for r in ordered)
    return {"n": len(ids), "independent_problem_or_scene_clusters": len(set(groups)),
        "success": statistics.mean(rewards), "success_95ci": interval(rewards, groups=groups),
        "paired_delta": statistics.mean(effects), "paired_delta_95ci": interval(effects, groups=groups),
        "improved_tasks": [t for t, e in zip(ids, effects) if e > 0],
        "harmed_tasks": [t for t, e in zip(ids, effects) if e < 0],
        "mean_drop_count": statistics.mean(r["dropped_count"] for r in ordered),
        "keep_rate": kept/total if total else 1.0, "groups": grouped,
        "macro_success": {f: statistics.mean(v["success"] for v in bins.values()) for f, bins in grouped.items()},
        "mean_progress": statistics.mean(r["outcome"]["progress"] for r in ordered) if name == "pddl" else None,
        "paired_tasks": [{"task_id": t, "baseline": b["outcome"]["success"], "policy": a["outcome"]["success"],
            "delta": e, "action_keys": a["action_keys"]} for t, a, b, e in zip(ids, ordered, base, effects)]}


def aggregate_cv(config, name):
    """Validate fold provenance before creating a derived OOF policy log."""
    from .run_interactive_gate_benchmark import dataset_jobs
    from .csm_st_data import write_rows

    root = Path(config["output_dir"]) / name
    records, sources, checkpoints, fold_scores, tested = [], {}, {}, {}, set()
    for label, expected in dataset_jobs(config, name):
        child_root = Path(expected["output_dir"])
        child_path = child_root / "benchmark_manifest.json"
        child = json.loads(child_path.read_text())
        if child != expected:
            raise ValueError("CV fold manifest differs from the registered parent")
        summary = report(child)[name]
        split = child["experiments"][name]["split"]
        if tested & set(split["test"]):
            raise ValueError("CV problem evaluated in multiple outer folds")
        tested.update(split["test"])
        path = child_root / name / "evaluation/policies.jsonl"
        sources[label] = {"root": str(child_root), "manifest_sha256": sha(child_path),
                          "policy_log_sha256": sha(path), "snapshot_sha256": summary["snapshot_sha256"],
                          "split": split}
        checkpoints[label] = summary["checkpoints_sha256"]
        fold_scores[label] = {m: v["success"] for m, v in summary["policies"].items()}
        for raw in path.read_text().splitlines():
            if raw.strip():
                row = json.loads(raw)
                records.append({**row, "source_config_digest": row["config_digest"], "config_digest": digest(config),
                                "cv_fold": child["cv_context"]["fold"], "source_root": str(child_root / name)})
    if tested != {t["id"] for t in config["experiments"][name]["tasks"]}:
        raise ValueError("CV out-of-fold test does not cover every registered PDDL problem")
    source_manifest = {"schema": "interactive-five-fold-oof-sources-v1", "parent_config_digest": digest(config),
        "folds": sources, "checkpoints_sha256": checkpoints,
        "constraint": "independent memory/checkpoints/caches per fold; no outer-test tuning"}
    locked(root / "cv_sources.json", source_manifest)
    write_rows(root / "evaluation/policies.jsonl", records)
    return source_manifest, fold_scores


def report(config, qa_root=None):
    if config.get("simulation") and qa_root is not None:
        raise ValueError("simulation reports cannot mix in measured QA results")
    output = Path(config["output_dir"])
    registered_policies = policies(config)
    summaries = {}
    for name, data in config["experiments"].items():
        root = output / name
        cv_provenance, fold_scores = (aggregate_cv(config, name) if "cv_splits" in data else (None, None))
        path = root / "evaluation/policies.jsonl"
        if not path.exists():
            raise ValueError(f"evaluation incomplete: {path}")
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        by_method = defaultdict(dict)
        tasks = {r["id"]: r for r in data["tasks"]}
        expected = (set(sum((s["test"] for s in data["cv_splits"]), [])) if cv_provenance
                    else set(data["split"]["test"]))
        for row in rows:
            method, task_id = row["policy"], row["task_id"]
            if bool(row.get("simulation", False)) != bool(config.get("simulation", False)):
                raise ValueError("cannot mix simulated and measured policy logs")
            if task_id in by_method[method]:
                raise ValueError("duplicate task-policy result")
            if row["config_digest"] != digest(config) or task_id not in expected:
                raise ValueError("unregistered evaluation result")
            by_method[method][task_id] = row
        methods = {m for m, _ in registered_policies} | {f"random_drop_seed{s}" for s in config["random_seeds"]}
        if set(by_method) != methods or any(set(g) != expected for g in by_method.values()):
            raise ValueError("incomplete strategy/task coverage; refusing partial averages")
        baseline = by_method["always_use"]
        ids = sorted(expected)
        primary_ids = sorted(data.get("primary_test_ids", ids))
        cohorts = data.get("test_cohorts", {})
        if not primary_ids or not set(primary_ids) <= expected:
            raise ValueError("invalid primary evaluation cohort")
        if cohorts and (set(sum(cohorts.values(), [])) != expected
                or sum(map(len, cohorts.values())) != len(expected)):
            raise ValueError("test cohorts must exactly partition all evaluated tasks")
        result = {}
        for method, values in by_method.items():
            ordered = [values[t] for t in ids]
            base = [baseline[t] for t in ids]
            for a, b in zip(ordered, base):
                if a["retrieval_fingerprint"] != b["retrieval_fingerprint"]:
                    raise ValueError("report retrieval mismatch")
                if a["seed"] != b["seed"] or a["initial_fingerprint"] != b["initial_fingerprint"]:
                    raise ValueError("report initial-state/seed mismatch")
                if a["dropped_count"] == 0 and a["outcome"] != b["outcome"]:
                    raise ValueError("report no-DROP parity failure")
            result[method] = policy_statistics(primary_ids, values, baseline, tasks, name)
            if fold_scores:
                scores = {fold: values[method] for fold, values in fold_scores.items()}
                result[method]["outer_fold_success"] = scores
                result[method]["outer_fold_success_stdev"] = statistics.stdev(scores.values())
            if cohorts:
                result[method]["cohort_results"] = {cohort: policy_statistics(sorted(selected), values, baseline, tasks, name)
                    for cohort, selected in cohorts.items()}
        random_scores = [result[f"random_drop_seed{s}"]["success"] for s in config["random_seeds"]]
        result["random_drop"]["random_seed_success"] = dict(zip(map(str, config["random_seeds"]), random_scores))
        result["random_drop"]["random_seed_success_stdev"] = statistics.stdev(random_scores)
        for task_id in ids:
            oracle = by_method["oracle"][task_id]["outcome"]["success"]
            if oracle < by_method["global_drop"][task_id]["outcome"]["success"]:
                raise ValueError("Oracle candidate set should include DROP ALL")
        checkpoints = (cv_provenance["checkpoints_sha256"] if cv_provenance else
            {m: sha(root / m / "checkpoint.json") for m in learned_methods(config)})
        summaries[name] = {"schema": "interactive-comparison-v1", "metric": "success",
            "simulation": bool(config.get("simulation", False)),
            "aggregation": "task-weighted_micro_mean; group macro also reported",
            "bootstrap": "paired_cluster_percentile_2000_resamples_seed71_by_scene_or_problem",
            "small_test_warning": f"Primary test has {len(primary_ids)} tasks; inspect scene/problem cluster count",
            "protocol": data["protocol"], "task_ids": ids, "task_ids_sha256": digest(ids),
            "primary_test_ids": primary_ids, "test_cohorts": cohorts,
            "source_root": str(root), "evaluation_log_sha256": sha(path), "checkpoints_sha256": checkpoints,
            "snapshot_sha256": sha(root / ("cv_sources.json" if cv_provenance else "snapshot.json")),
            "seed": config["evaluation_seed"],
            "evaluation_design": "five_fold_out_of_fold" if cv_provenance else "fixed_holdout",
            "config_digest": digest(config), "policies": result,
            "budgets": {"reward_gates": "dev-selected 0 or 1 SINGLE DROP", "random": "one SINGLE DROP per seed",
                "csm": "unlimited; keep iff predicted CI>0 after dedup",
                "local_global_team_u": ("unlimited; full-list U scoring then dedup; keep iff U>0"
                                        if config.get("include_local_global_team_u") else "not enabled"),
                "oracle": "KEEP ALL + every SINGLE DROP + DROP ALL; excludes arbitrary CSM/Team U subsets"},
            "interpretation": "method-level comparison; proxy, architecture and action budgets differ; Oracle is post hoc"}
        save(root / "comparison.json", summaries[name])
        save(root / "summary.json", {k: v for k, v in summaries[name].items() if k != "policies"} | {
            "policies": {m: {k: v for k, v in stats.items() if k != "paired_tasks"} for m, stats in result.items()}})
    prior = {}
    if qa_root is not None:
        from .interactive_prior_results import import_qa
        prior = import_qa(qa_root)
    simulation = bool(config.get("simulation", False))
    save(output / "matrix_results.json", {"schema": "interactive-matrix-v1", "simulation": simulation, "datasets": summaries,
        "fever_hotpotqa": prior or "not executed or inferred from screenshot; no imported verified source provided"})
    extra_cohort = "train_scene_holdout" in summaries.get("alfworld", {}).get("test_cohorts", {})
    text = ("| Policy | FEVER binary Accuracy % | HotpotQA Answer F1 / EM % | PDDL Success % | ALFWorld unseen Success % |"
            + (" ALFWorld extra holdout Success % |" if extra_cohort else "")
            + "\n| --- | ---: | ---: | ---: | ---: |" + (" ---: |" if extra_cohort else "") + "\n")
    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(["policy", "dataset", "primary_metric", "primary_percent", "answer_em_percent",
                     "paired_delta_pp", "delta_95ci_pp", "n", "simulation"])
    for method, label in registered_policies:
        cells = []
        for name in ("fever", "hotpotqa"):
            old = prior.get(name, {}).get("policies", {}).get(method)
            cells.append("—" if old is None else f"{100*old['primary']:.2f}" +
                         (f" / {100*old['answer_em']:.2f}" if name == "hotpotqa" else ""))
            if old:
                writer.writerow([label, name, prior[name]["metric"], 100*old["primary"],
                    100*old["answer_em"] if old["answer_em"] is not None else "", 100*old["paired_delta"],
                    json.dumps([100*v for v in old["paired_delta_95ci"]]), len(old["task_ids"]), simulation])
        for name in ("pddl", "alfworld"):
            stats = summaries.get(name, {}).get("policies", {}).get(method)
            cells.append(f"{100*stats['success']:.2f}" if stats else "—")
            if stats:
                writer.writerow([label, name, "success", 100*stats["success"], "", 100*stats["paired_delta"],
                    json.dumps([100*v for v in stats["paired_delta_95ci"]]), stats["n"], simulation])
                for cohort, sub in stats.get("cohort_results", {}).items():
                    writer.writerow([label, name + ":" + cohort, "success", 100*sub["success"], "",
                        100*sub["paired_delta"], json.dumps([100*v for v in sub["paired_delta_95ci"]]), sub["n"], simulation])
        if extra_cohort:
            stats = summaries["alfworld"]["policies"][method]["cohort_results"]["train_scene_holdout"]
            cells.append(f"{100*stats['success']:.2f}")
        text += f"| {label} | {' | '.join(cells)} |\n"
    if "pddl" in summaries:
        pddl = summaries["pddl"]
        text += (f"\nPDDL 为 {len(pddl['primary_test_ids'])} 题的"
                 + ("五折 OOF 测评，每题仅测一次。" if pddl["evaluation_design"] == "five_fold_out_of_fold"
                    else "固定留出测评。"))
    text += ("\nSuccess 使用显式 won，progress 单独保存。"
             "Random 为预声明随机种子的均值；Oracle 只在已声明动作中事后选择。"
             "FEVER/HotpotQA 仅可从已有日志只读导入，缺失保留为空。未核验的截图分数不纳入本表。\n")
    if config.get("include_local_global_team_u"):
        text += ("\nTeam U 在原候选列表上训练和评分，随后语义去重，保留 U>0 的非重复项，删除不限量。"
                 "复用已测 KEEP/SINGLE DROP Success，不使用专家或 CE，不重组候选列表；损失采用 QA Team U 的 reward 尺度参数。"
                 "Oracle 的固定动作集不包含任意 learned 多条删除子集，因此不保证压过它们。\n")
    for name, summary in summaries.items():
        if summary["test_cohorts"]:
            text += f"\n{name} 主表仅使用预声明的 {len(summary['primary_test_ids'])} 题主测评集。额外测评分组：\n\n"
            text += "| Policy | Cohort | N | Success % |\n| --- | --- | ---: | ---: |\n"
            for method, label in registered_policies:
                for cohort, stats in summary["policies"][method]["cohort_results"].items():
                    text += f"| {label} | {cohort} | {stats['n']} | {100*stats['success']:.2f} |\n"
    if simulation:
        text = "**模拟输出：全部分数来自人工构造的合成日志，仅展示格式，不代表实测或预期效果。**\n\n" + text
    (output / "matrix_results.md").write_text(text)
    (output / "matrix_results.csv").write_text(table.getvalue())
    return summaries


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qa-results-root", type=Path)
    args = parser.parse_args()
    report(json.loads(args.manifest.read_text()), args.qa_results_root)
