"""Train/evaluate a separate ALFWorld run from completed diagnostic tasks.

The source experiment stays resumable. Original fit/dev membership, measured
KEEP/SINGLE DROP outcomes, support memory, test cohorts and policy settings are
preserved. Only the explicitly registered training subset and output change.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import uuid

from .interactive_data import digest, files_identity, save, sha, validate_split
from .interactive_empty_memory import coverage, verify_registered_code
from .interactive_progress import STAGE_NAMES, run_logged, show_progress
from .resume_interactive_dataset import dataset_report, experiment_lock, registered_config


SCHEMA = "alfworld-partial-diagnostic-v1"
NAME = "alfworld"
STAGES = ("labels_team_u", "train_reward", "train_team_u", "evaluate", "report")


def read(path):
    return json.loads(Path(path).read_text())


def separate_output(source, output):
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("partial output must be independent of the source experiment")


def selection(source, expected_completed=None):
    """Inspect only local artifacts; no service calls or model imports."""
    from .interactive_worker import snapshot

    config = registered_config(source, NAME)
    data = config["experiments"][NAME]
    validate_split(data["tasks"], data["split"])
    memory = snapshot(config, NAME)
    root = source / NAME
    registered = set(data["split"]["fit"] + data["split"]["dev"])
    expected_files = {digest(task_id) + ".json": task_id for task_id in registered}
    original_digest = digest(config)
    rows, hashes = [], {}
    for path in sorted((root / "diagnostic/tasks").glob("*.json")):
        row = read(path)
        if (path.name not in expected_files or row["task_id"] != expected_files[path.name]
                or row["config_digest"] != original_digest or row["repeats"] != config["repeats"]):
            raise ValueError(f"unregistered diagnostic summary: {path}")
        rows.append(row)
        hashes[str(path.relative_to(source))] = sha(path)
    if expected_completed is not None and len(rows) != expected_completed:
        raise ValueError(f"expected {expected_completed} completed tasks, found {len(rows)}")
    ids = {row["task_id"] for row in rows}
    split = deepcopy(data["split"])
    for part in ("fit", "dev"):
        split[part] = [task_id for task_id in split[part] if task_id in ids]
        if not split[part]:
            raise ValueError(f"completed tasks contain no original {part} tasks; collect more first")
    counts = coverage(rows, {"split": split})
    if (any(counts[p]["candidate_count"] for p in counts)
            and not all(counts[p]["candidate_count"] for p in counts)):
        raise ValueError("nonempty supervision is required in both original fit and dev")
    by_id = {row["id"]: row for row in data["tasks"]}
    plan = {"schema": SCHEMA, "source_dir": str(source), "completed_tasks": len(rows),
        "registered_diagnostic_tasks": len(registered),
        "split_counts": {part: len(values) for part, values in split.items()},
        "original_split_counts": {part: len(values) for part, values in data["split"].items()},
        "candidate_coverage": {p: {"supervised_tasks": len(counts[p]["supervised_task_ids"]),
            "empty_tasks": len(counts[p]["empty_task_ids"]), "candidates": counts[p]["candidate_count"]}
            for p in counts},
        "completed_task_types": {p: dict(Counter(by_id[t]["task_type"] for t in split[p]))
                                 for p in ("fit", "dev")},
        "primary_test_count": len(data.get("primary_test_ids", split["test"])),
        "test_cohort_counts": {p: len(values) for p, values in data.get("test_cohorts", {}).items()},
        "selection_rule": "completed atomic summaries; retain original fit/dev membership; no outcome selection",
        "stages": list(STAGES), "additional_support_or_diagnostic_episodes": 0}
    return config, memory, rows, hashes, split, plan


def copy_sqlite(source, destination):
    """Copy committed cache data, including WAL, without opening it writable."""
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as src:
        with sqlite3.connect(destination) as dest:
            src.backup(dest)


def prepare(source, output, *, expected_completed=None, train_device=None):
    """Caller holds source/output locks. Publish only a verified private copy."""
    from .interactive_runtime import ActionStore, clone_memory
    from .interactive_team_u import measured_lists
    from .interactive_worker import snapshot

    original, memory, rows, source_hashes, split, plan = selection(source, expected_completed)
    source_root = source / NAME
    original_identity = {"config_digest": digest(original),
        "snapshot_sha256": sha(source_root / "snapshot.json"), "phase": "diagnostic"}
    identity_path = source_root / "diagnostic/actions/identity.json"
    if read(identity_path) != original_identity:
        raise ValueError("source diagnostic action-store identity changed")
    original_store = ActionStore(source_root / "diagnostic/actions", original_identity)
    config = deepcopy(original)
    config["output_dir"] = str(output)
    data = deepcopy(original["experiments"][NAME])
    data["split"] = split
    selected = set(sum(split.values(), []))
    data["tasks"] = [task for task in data["tasks"] if task["id"] in selected]
    validate_split(data["tasks"], split)
    config["experiments"] = {NAME: data}
    if train_device is not None:
        config["train_device"] = train_device
    helper_paths = {"causal_memory_control/run_partial_alfworld.py",
                    "causal_memory_control/resume_interactive_dataset.py"}
    summary_hashes_digest = digest(source_hashes)
    registration = source / "empty_memory_upgrade/registration.json"
    if registration.exists():
        helper_paths.update(read(registration)["effective_code_sha256"])
        source_hashes[str(registration.relative_to(source))] = sha(registration)
    config["code_sha256"] = {p: sha(Path(config["repo"]) / p)
                            for p in sorted(set(original["code_sha256"]) | helper_paths)}
    config["partial_diagnostic"] = {"schema": SCHEMA, "source_dir": str(source),
        "source_config_digest": digest(original), "source_manifest_sha256": sha(source / "benchmark_manifest.json"),
        "source_snapshot_sha256": original_identity["snapshot_sha256"],
        "source_action_identity_sha256": sha(identity_path),
        "selected_summary_hashes_digest": summary_hashes_digest, "test_unchanged": True,
        "completed_tasks": len(rows), "registered_diagnostic_tasks": plan["registered_diagnostic_tasks"],
        "selection_rule": plan["selection_rule"]}
    config_digest = digest(config)
    temporary = output.with_name(output.name + ".preparing-" + uuid.uuid4().hex)
    temporary.mkdir()
    published = False
    try:
        root = temporary / NAME
        target_memory = root / "frozen_support/g-memory"
        print(f"[冻结子集] {len(rows)}/{plan['registered_diagnostic_tasks']}；"
              f"fit={len(split['fit'])}，dev={len(split['dev'])}，test={len(split['test'])}（测试集保持原划分）。", flush=True)
        clone_memory(memory, target_memory)
        old_snapshot = read(source_root / "snapshot.json")
        save(root / "snapshot.json", {**old_snapshot, "config_digest": config_digest,
            "memory_dir": "frozen_support/g-memory", "files_sha256": files_identity(target_memory),
            "source_snapshot_sha256": original_identity["snapshot_sha256"]})
        new_identity = {"config_digest": config_digest,
            "snapshot_sha256": sha(root / "snapshot.json"), "phase": "diagnostic"}
        new_store = ActionStore(root / "diagnostic/actions", new_identity)

        def capture(path):
            source_hashes[str(path.relative_to(source))] = sha(path)

        for number, raw in enumerate(rows, 1):
            for repeat in range(config["repeats"]):
                seed = config["diagnostic_seed"] + repeat
                key = original_store.key(raw["task_id"], seed, raw, [])
                path = original_store.root / "parts" / (key + ".json")
                initial = read(path)
                keep = original_store.get(raw["task_id"], seed, initial, [])
                if keep is None:
                    raise ValueError("source summary lacks a measured KEEP branch")
                capture(path)
                new_keep = new_store.put(keep)
                save(new_store.root / "baselines" / (digest([raw["task_id"], seed]) + ".json"),
                     new_keep)
                for candidate in raw["candidates"]:
                    drops = [candidate["memory_id"]]
                    dropped = original_store.get(raw["task_id"], seed, keep, drops)
                    if dropped is None:
                        raise ValueError("source summary lacks a measured SINGLE DROP branch")
                    key = original_store.key(raw["task_id"], seed, keep, drops)
                    capture(original_store.root / "parts" / (key + ".json"))
                    new_store.put(dropped)
            save(root / "diagnostic/tasks" / (digest(raw["task_id"]) + ".json"),
                 {**raw, "config_digest": config_digest})
            if number % 25 == 0 or number == len(rows):
                show_progress("复制并核验已完成诊断", number, len(rows))
        if (source_root / "responses.sqlite").is_file():
            copy_sqlite(source_root / "responses.sqlite", root / "responses.sqlite")
        for old_path, new_name in ((source / "benchmark_manifest.json", "source_manifest.json"),
                                   (source_root / "snapshot.json", "source_snapshot.json"),
                                   (identity_path, "source_action_identity.json")):
            capture(old_path)
            destination = temporary / "partial_diagnostic" / new_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_path, destination)
        if registration.exists():
            shutil.copy2(registration, temporary / "partial_diagnostic/source_upgrade_registration.json")
        # Detect a noncooperating writer before publishing the copied inputs.
        for relative, expected in source_hashes.items():
            if sha(source / relative) != expected:
                raise ValueError(f"source changed while freezing partial inputs: {relative}")
        snapshot(original, NAME)
        provenance = {**plan, "source_files_sha256": source_hashes,
            "included_task_ids": {p: split[p] for p in ("fit", "dev")},
            "missing_task_ids": {p: sorted(set(original["experiments"][NAME]["split"][p]) - set(split[p]))
                                 for p in ("fit", "dev")},
            "identity_migration": "validated source records; only config/store identities rebound; outcomes and action keys unchanged"}
        save(temporary / "partial_diagnostic/source_provenance.json", provenance)
        save(temporary / "benchmark_manifest.json", config)
        temporary.rename(output)
        published = True
        snapshot(config, NAME)
        _, measured_hashes = measured_lists(config, NAME)
        save(output / "partial_diagnostic/prepared.json", {"schema": SCHEMA, "config_digest": config_digest,
            "provenance_sha256": sha(output / "partial_diagnostic/source_provenance.json"),
            "source_copies_sha256": files_identity(output / "partial_diagnostic"),
            "measured_source_sha256": measured_hashes})
        return config, plan
    except BaseException:
        if published:
            output.rename(temporary)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def prepared(output, source, *, expected_completed=None, train_device=None):
    """Resume the frozen subset, even if source collection later progresses."""
    from .interactive_team_u import measured_lists
    from .interactive_worker import snapshot

    output, source = Path(output).resolve(), Path(source).resolve()
    config = read(output / "benchmark_manifest.json")
    info = config["partial_diagnostic"]
    if (info["schema"] != SCHEMA or info["source_dir"] != str(source)
            or Path(config["output_dir"]).resolve() != output
            or Path(config["repo"]).resolve() != Path(__file__).resolve().parents[1]):
        raise ValueError("partial experiment identity/path changed")
    if expected_completed is not None and info["completed_tasks"] != expected_completed:
        raise ValueError("existing partial experiment has a different fixed task count")
    if train_device is not None and config["train_device"] != train_device:
        raise ValueError("partial experiment train device changed; use a new output directory")
    marker = output / "partial_diagnostic/prepared.json"
    if not marker.is_file():
        raise ValueError("partial snapshot is incomplete; use a new output directory")
    record = read(marker)
    if record["config_digest"] != digest(config):
        raise ValueError("partial experiment configuration changed")
    verify_registered_code(config)
    provenance_path = output / "partial_diagnostic/source_provenance.json"
    if sha(provenance_path) != record["provenance_sha256"]:
        raise ValueError("partial source provenance changed")
    for relative, expected in record["source_copies_sha256"].items():
        if sha(output / "partial_diagnostic" / relative) != expected:
            raise ValueError("partial source metadata copy changed")
    snapshot(config, NAME)
    _, hashes = measured_lists(config, NAME)
    if hashes != record["measured_source_sha256"]:
        raise ValueError("frozen partial diagnostic inputs changed")
    return config, read(provenance_path)


def partial_report(config, plan):
    payload = dataset_report(config, NAME)
    payload["partial_diagnostic"] = {k: plan[k] for k in (
        "completed_tasks", "registered_diagnostic_tasks", "split_counts", "original_split_counts",
        "selection_rule", "completed_task_types")}
    payload["interpretation"] = "partial diagnostic training subset; original held-out test unchanged; not the full 1000-task training protocol"
    output = Path(config["output_dir"])
    save(output / "results.json", payload)
    text = (f"# ALFWorld 部分诊断数据实验\n\n使用 {plan['completed_tasks']}/"
            f"{plan['registered_diagnostic_tasks']} 个完整诊断任务；"
            f"fit={plan['split_counts']['fit']}，dev={plan['split_counts']['dev']}。\n\n"
            "按原 fit/dev 归属取已完成子集，未按结果重新筛选或划分；support、测试集和策略规则沿用源实验。"
            "这是部分诊断训练结果，不等同于完整 1000 任务训练。任务按顺序完成，需留意任务类型覆盖偏差。\n\n")
    text += (output / NAME / "dataset_report/comparison.md").read_text()
    (output / "results.md").write_text(text)
    print(f"[部分数据结果] {output / 'results.md'}", flush=True)


def run(source, output, *, stage="all", expected_completed=None, train_device=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    separate_output(source, output)
    if stage == "plan":
        if output.exists():
            _, plan = prepared(output, source, expected_completed=expected_completed, train_device=train_device)
        else:
            *_, plan = selection(source, expected_completed)
        print(json.dumps({k: v for k, v in plan.items() if k not in (
            "source_files_sha256", "included_task_ids", "missing_task_ids", "identity_migration")}
            | {"output_dir": str(output), "writes_or_model_or_service_calls": 0}, ensure_ascii=False, indent=2), flush=True)
        return plan
    output.parent.mkdir(parents=True, exist_ok=True)
    with experiment_lock(output):
        if output.exists():
            config, plan = prepared(output, source, expected_completed=expected_completed, train_device=train_device)
        else:
            with experiment_lock(source):
                config, plan = prepare(source, output, expected_completed=expected_completed, train_device=train_device)
        if stage == "prepare":
            print(f"[冻结完成] {output}；未调用模型、未开始训练。", flush=True)
            return plan
        selected = STAGES if stage == "all" else STAGES[:3] if stage == "train" else (stage,)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        state_path = output / "partial_diagnostic/progress.json"
        state = {"schema": SCHEMA, "config_digest": digest(config), "stages": list(selected),
                 "completed_stages": [], "status": "running"}
        try:
            for number, current in enumerate(selected):
                state.update(stage=current, status="running")
                save(state_path, state)
                show_progress("部分诊断数据实验", number, len(selected), STAGE_NAMES[current])
                if current == "report":
                    partial_report(config, plan)
                else:
                    command = [sys.executable, "-u", "-X", "faulthandler", "-m",
                        "causal_memory_control.interactive_worker", "--config",
                        str(output / "benchmark_manifest.json"), "--dataset", NAME, "--stage", current]
                    run_logged(command, output / NAME / "logs" / (current + ".log"), cwd=Path(config["repo"]))
                state["completed_stages"].append(current)
                save(state_path, state)
            state["status"] = "completed"
            save(state_path, state)
            show_progress("部分诊断数据实验", len(selected), len(selected), "本次命令已完成")
        except BaseException as exc:
            state.update(status="failed", error=str(exc) or type(exc).__name__)
            save(state_path, state)
            raise
        return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-completed", type=int, help="Fail rather than silently freeze a different task count")
    parser.add_argument("--train-device", help="Optional device for this independent run, e.g. cpu or cuda:0")
    parser.add_argument("--stage", choices=("plan", "prepare", "train", "all", *STAGES), default="all")
    args = parser.parse_args(argv)
    run(args.source_dir, args.output_dir, stage=args.stage,
        expected_completed=args.expected_completed, train_device=args.train_device)


if __name__ == "__main__":
    main()
