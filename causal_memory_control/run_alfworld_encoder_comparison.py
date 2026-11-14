"""Run original eight ALFWorld policies plus an independent frozen-encoder ST.

Use completed diagnostics only, preserve the original experiment and old ST,
and compare encoders with the original calibrated ST deletion rule.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path

from . import run_partial_alfworld as partial
from . import run_st_encoder_benchmark as encoder
from .interactive_data import digest, locked, save, sha
from .interactive_protocol import policies
from .resume_interactive_dataset import experiment_lock


SCHEMA = "alfworld-original-and-new-st-encoder-v1"
NEW_POLICY = "semantic_st_new_encoder"


def encoder_arguments(args, stage):
    values = ["--datasets", "alfworld", "--interactive-root", str(args.base_dir),
        "--output-dir", str(args.output_dir / "encoder"), "--semantic-model", args.semantic_model,
        "--train-device", args.train_device, "--st-deletion-rule", "calibrated", "--stage", stage]
    if args.semantic_revision:
        values += ["--semantic-revision", args.semantic_revision]
    if stage in ("fit", "all"):
        values += ["--refit-head"]
    return values


def binding(args):
    return {"schema": SCHEMA, "source_dir": str(args.source_dir), "base_dir": str(args.base_dir),
        "output_dir": str(args.output_dir), "expected_completed": args.expected_completed,
        "source_manifest_sha256": sha(args.source_dir / "benchmark_manifest.json"),
        "semantic_model": args.semantic_model, "semantic_revision": args.semantic_revision,
        "train_device": args.train_device, "st_deletion_rule": "calibrated",
        "implementation_sha256": {name: sha(Path(__file__).with_name(name)) for name in (
            "run_alfworld_encoder_comparison.py", "run_partial_alfworld.py", "run_st_encoder_benchmark.py",
            "reevaluate_st_policy.py", "st_policy.py")},
        "new_support_or_diagnostic_collection": False, "old_st_preserved": True}


def combined_report(args):
    """Audit all test tasks and their measured actions before reporting nine rows."""
    from .interactive_report import policy_statistics
    from .interactive_runtime import ActionStore

    config, plan = partial.prepared(args.base_dir, args.source_dir,
        expected_completed=args.expected_completed, train_device=args.train_device)
    root = args.base_dir / "alfworld"
    baseline_report = partial.read(args.base_dir / "results.json")
    source_log = root / "evaluation/policies.jsonl"
    if (baseline_report["config_digest"] != digest(config)
            or baseline_report["evaluation_log_sha256"] != sha(source_log)):
        raise ValueError("original eight-policy report/configuration changed")
    new_root = args.output_dir / "encoder"
    manifest = partial.read(new_root / "manifest.json")
    if manifest["datasets"] != ["alfworld"] or manifest["rules"] != {"alfworld": "calibrated"}:
        raise ValueError("encoder comparison must preserve the original calibrated ALFWorld rule")
    for path, expected in manifest["source_sha256"].items():
        if sha(path) != expected:
            raise ValueError(f"encoder comparison source changed: {path}")
    output = new_root / "alfworld"
    checkpoint = output / "checkpoint.json"
    checkpoint_sha = sha(checkpoint) if checkpoint.exists() else None
    original_identity = partial.read(root / "evaluation/actions/identity.json")
    original = ActionStore(root / "evaluation/actions", original_identity)
    new_identity = {"schema": encoder.SCHEMA, "rule": "calibrated",
        "checkpoint_sha256": checkpoint_sha, "original_identity": original_identity}
    if partial.read(output / "actions/identity.json") != new_identity:
        raise ValueError("new ST measured action identity changed")
    new_store = ActionStore(output / "actions", new_identity)
    new_log = output / "policies.jsonl"
    rows = encoder.reuse.rows(new_log)
    selected = encoder.reuse.indexed(rows)
    data = config["experiments"]["alfworld"]
    ids = set(data["split"]["test"])
    if set(selected) != ids:
        raise ValueError("new ST must evaluate every original test task")
    baselines = {row["task_id"]: row for row in encoder.reuse.rows(source_log) if row["policy"] == "always_use"}
    if set(baselines) != ids:
        raise ValueError("original baseline task coverage changed")
    for task_id, row in selected.items():
        if row["rule"] != "calibrated" or row["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("new ST output has a different rule/checkpoint")
        reference = baselines[task_id]
        if any(row[k] != reference[k] for k in ("seed", "initial_fingerprint", "retrieval_fingerprint", "retrieved_count")):
            raise ValueError("new ST changed the original seed/state/retrieval")
        index = original.root / "baselines" / (digest([task_id, config["evaluation_seed"]]) + ".json")
        baseline = original.get(task_id, row["seed"], partial.read(index), [])
        if baseline is None:
            raise ValueError("missing original test baseline")
        measured = (original.get(task_id, row["seed"], baseline, row["drop_ids"])
                    or new_store.get(task_id, row["seed"], baseline, row["drop_ids"]))
        if (measured is None or row["outcome"] != measured["outcome"]
                or row["action_keys"] != [measured["action_key"]]
                or row["baseline_outcome"] != baseline["outcome"]
                or row["dropped_count"] != len(row["drop_ids"])
                or [d["memory_id"] for d in row["decisions"]] != [c["memory_id"] for c in baseline["candidates"]]
                or sorted(d["memory_id"] for d in row["decisions"] if d["action"].upper() == "DROP") != sorted(row["drop_ids"])):
            raise ValueError("new ST result differs from its matched measured action/decisions")
    tasks = {task["id"]: task for task in data["tasks"]}
    primary = data.get("primary_test_ids", data["split"]["test"])
    stats = policy_statistics(sorted(primary), selected, baselines, tasks, "alfworld")
    stats["cohort_results"] = {name: policy_statistics(sorted(values), selected, baselines, tasks, "alfworld")
                               for name, values in data.get("test_cohorts", {}).items()}
    payload = deepcopy(baseline_report)
    payload.update(schema=SCHEMA, original_results_dir=str(args.base_dir),
        original_report_sha256=sha(args.base_dir / "results.json"),
        encoder_manifest_sha256=sha(new_root / "manifest.json"), new_st_log_sha256=sha(new_log),
        new_encoder=manifest["encoder"], deletion_rule="unchanged_original_calibrated_ST_rule",
        comparison_scope="original eight policies plus a separately refitted ST head; no new diagnostic collection")
    payload["policies"][NEW_POLICY] = stats
    save(args.output_dir / "results.json", payload)
    names = [name for name, _ in policies(config)] + [NEW_POLICY]
    table = []
    for name in names:
        values = payload["policies"][name]
        cohorts = values.get("cohort_results") or {"primary": values}
        for cohort, result in cohorts.items():
            table.append({"Policy": name, "Cohort": cohort, "N": result["n"],
                "Success (%)": 100 * result["success"], "Delta vs Always Use (pp)": 100 * result["paired_delta"],
                "Mean drops": result["mean_drop_count"]})
    fields = list(table[0])
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    text = (f"# ALFWorld 九项对照：原八策略 + 新编码器 ST\n\n"
            f"完整诊断子集 {plan['completed_tasks']}/{plan['registered_diagnostic_tasks']}；"
            f"fit={plan['split_counts']['fit']}，dev={plan['split_counts']['dev']}；"
            f"全部原测试任务 {len(ids)} 题。部分训练协议与完整诊断训练分开报告。\n\n"
            f"新 ST 冻结编码器：`{args.semantic_model}`。原 `semantic_st` 保留；"
            "新头复用同一批标签、任务划分、损失和原 ST 删除参数，编码器不训练。\n\n"
            "Oracle 仍使用原实验预先规定的动作集合，不因新增 ST 改变其定义。\n\n")
    text += "| " + " | ".join(fields) + " |\n| " + " | ".join("---" for _ in fields) + " |\n"
    for row in table:
        text += "| " + " | ".join(f"{row[k]:.2f}" if isinstance(row[k], float) else str(row[k]) for k in fields) + " |\n"
    (args.output_dir / "results.md").write_text(text)
    print(f"[九项对照结果] {args.output_dir / 'results.md'}", flush=True)
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True, help="Independent original-eight-policy partial experiment")
    parser.add_argument("--output-dir", type=Path, required=True, help="Independent ninth-policy and combined report directory")
    parser.add_argument("--expected-completed", type=int, default=845)
    parser.add_argument("--semantic-model", required=True)
    parser.add_argument("--semantic-revision")
    parser.add_argument("--train-device", default="cpu")
    parser.add_argument("--stage", choices=("plan", "all", "base", "fit", "evaluate", "report"), default="plan")
    args = parser.parse_args(argv)
    for name in ("source_dir", "base_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    for source, output in ((args.source_dir, args.base_dir), (args.source_dir, args.output_dir),
                           (args.base_dir, args.output_dir)):
        partial.separate_output(source, output)
    if Path(args.semantic_model).is_dir():
        args.semantic_model = str(Path(args.semantic_model).resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "plan":
        if args.base_dir.exists():
            _, plan = partial.prepared(args.base_dir, args.source_dir,
                expected_completed=args.expected_completed, train_device=args.train_device)
        else:
            *_, plan = partial.selection(args.source_dir, args.expected_completed)
        result = {"schema": SCHEMA, "completed_tasks": plan["completed_tasks"],
            "split_counts": plan["split_counts"], "candidate_coverage": plan["candidate_coverage"],
            "completed_task_types": plan["completed_task_types"], "source_dir": str(args.source_dir),
            "base_dir": str(args.base_dir), "output_dir": str(args.output_dir),
            "new_encoder": args.semantic_model, "st_rule": "original calibrated rule for both encoders",
            "phases": ["original eight policies", "new ST head fit", "new ST evaluation", "nine-policy report"],
            "additional_support_or_diagnostic_episodes": 0, "writes_or_model_or_service_calls": 0}
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with experiment_lock(args.output_dir):
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not (args.output_dir / "manifest.json").exists():
            raise ValueError("refusing a nonempty output without a nine-policy comparison manifest")
        locked(args.output_dir / "manifest.json", binding(args))
        state = {"stage": args.stage, "status": "running"}
        try:
            if args.stage in ("all", "base"):
                state["phase"] = "original_eight"
                save(args.output_dir / "progress.json", state)
                partial.run(args.source_dir, args.base_dir, stage="all",
                    expected_completed=args.expected_completed, train_device=args.train_device)
            if args.stage in ("all", "fit", "evaluate"):
                state["phase"] = "new_st_encoder"
                save(args.output_dir / "progress.json", state)
                encoder.main(encoder_arguments(args, "all" if args.stage == "all" else args.stage))
            if args.stage in ("all", "evaluate", "report"):
                with experiment_lock(args.base_dir):
                    combined_report(args)
            state["status"] = "completed"
        except BaseException as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            save(args.output_dir / "progress.json", state)
    return args.output_dir


if __name__ == "__main__":
    main()
