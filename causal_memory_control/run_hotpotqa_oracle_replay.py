"""Obtain HotpotQA Oracle from a NEW outcome-blind canonical-cache replay.

Replay KEEP ALL, DROP ALL and every SINGLE DROP, not the learned policies.
Old conflicting outcomes are historical references, never Oracle candidates.
The original baseline must reproduce with zero network calls. Subsequent jobs
inherit the preceding completed response cache and emit task/request ledgers.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import copy
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace

from causal_diagnostic.oracle_recipient import seeded_client
from . import audit_oracle_replay as audit
from . import run_oracle_evaluation as original
from . import run_full_gate_benchmark as full
from . import oracle_replay_worker as replay_worker
from . import hotpotqa_experiment as hotpotqa
from .compare_gate_runs import compare, file_sha256
from .csm_st_data import save_json
from .planned_drop import build_plan, indexed_rows
from .oracle_replay_worker import validate_ledger


SCHEMA = "hotpotqa-canonical-oracle-replay-v1"
NOTE = ("NEW canonical-cache action replay, not an unchanged continuation of historical gate scores. "
        "Cached response conflicts use a fixed source priority independent of rewards. "
        "Every Oracle action is executed again with cache reuse; historical policy outcomes are not "
        "Oracle candidates. KEEP ALL must reproduce the original baseline without network calls. "
        "Oracle is post-hoc over KEEP ALL + each SINGLE DROP + DROP ALL, not deployable, "
        "not all memory subsets or an expected-performance upper bound. HotpotQA Answer F1/EM; "
        "F1 selects actions, so only F1 is guaranteed to dominate each measured control. "
        "No training, CE collection, memory rebuilding or FEVER execution.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--cache-only", action="store_true",
                        help="Block on the first uncached request; resume without this flag to permit inference")
    parser.add_argument("--acknowledge-canonical-replay", action="store_true",
                        help="Accept a NEW fixed-priority cache replay, not unchanged historical policy scores")
    args = parser.parse_args(argv)
    if not args.plan_only and not args.acknowledge_canonical_replay:
        parser.error("execution requires --acknowledge-canonical-replay")
    args.evaluation_root, args.output_dir = args.evaluation_root.resolve(), args.output_dir.resolve()
    return args


def canonical_cache(caches, destination, cache_audit):
    """Explicitly resolve cache versions for a new replay, never old outcomes.

    All alternatives remain in immutable source caches and the audit report.
    The strict legacy merge_caches implementation is intentionally untouched.
    """
    descriptor = destination.with_suffix(".sources.json")
    identity = {"schema": SCHEMA, "source_priority": list(caches),
                "sources": {str(path): file_sha256(path) for path in caches.values()},
                "conflicts": cache_audit["conflicts"], "selection": "first_source_in_fixed_priority_not_reward"}
    if descriptor.exists() and original.trained.read(descriptor).get("identity") != identity:
        raise ValueError("canonical-cache inputs changed; use original inputs or a new output directory")
    if destination.exists():
        if not descriptor.is_file() or original.trained.read(descriptor).get("sha256") != file_sha256(destination):
            raise ValueError("canonical cache was changed or lacks provenance")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="canonical-build-", dir=destination.parent) as temporary:
        path = Path(temporary) / "canonical.sqlite"
        with closing(sqlite3.connect(path)) as target:
            for table, column in audit.TABLES:
                target.execute(f"CREATE TABLE {table}(key TEXT PRIMARY KEY, {column} TEXT NOT NULL)")
            for method, source_path in caches.items():
                with closing(sqlite3.connect(source_path.resolve().as_uri() + "?mode=ro", uri=True)) as source:
                    tables = {r[0] for r in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    for table, column in audit.TABLES:
                        if table in tables:
                            target.executemany(f"INSERT OR IGNORE INTO {table} VALUES (?,?)",
                                source.execute(f"SELECT key, {column} FROM {table} ORDER BY key"))
            target.commit()
            for conflict in cache_audit["conflicts"]:
                column = dict(audit.TABLES)[conflict["table"]]
                row = target.execute(f"SELECT {column} FROM {conflict['table']} WHERE key=?",
                                     (conflict["key"],)).fetchone()
                if row is None or hashlib.sha256(row[0].encode("utf-8")).hexdigest() != conflict["proposed_response_sha256"]:
                    raise ValueError("canonical response does not match declared source priority")
        for source_path in caches.values():
            if identity["sources"][str(source_path)] != file_sha256(source_path):
                raise ValueError("source cache changed during canonical-cache construction")
        save_json(descriptor, {"identity": identity, "sha256": file_sha256(path)})
        path.replace(destination)


def prepare(suite):
    adapted, plans, hashes = original.prepare(SimpleNamespace(evaluation_root=suite.evaluation_root,
        output_dir=suite.output_dir, datasets=["hotpotqa"], plan_only=True))
    adapter, args, sources, _, historical = adapted["hotpotqa"]
    report = audit.audit(suite.evaluation_root, "hotpotqa")
    if "always_global_drop" not in report["source_priority_proposal"]:
        raise ValueError("this repair entry requires the completed historical DROP ALL control")
    for path, expected in report["source_sha256"].items():
        if path in hashes and hashes[path] != expected:
            raise ValueError(f"source changed between validation and audit: {path}")
        hashes[path] = expected
    for path in (__file__, replay_worker.__file__, hotpotqa.__file__, original.__file__, full.__file__,
                 audit.__file__, seeded_client.__file__):
        hashes[str(Path(path).resolve())] = file_sha256(Path(path))
    # Only a complete test/KEEP/ALL/SINGLE grid is supported here, without
    # silently giving a missing candidate rank the meaning KEEP ALL.
    count = args.successful_topk + args.insights_topk
    baseline = indexed_rows(sources["native_gmemory"])
    if any(len(row["decisions"]) != count for row in baseline.values()):
        raise ValueError("candidate count varies; an explicit variable-action replay plan is required")
    caches = {method: suite.evaluation_root / "hotpotqa" / method / "llm_cache.sqlite"
              for method in report["source_priority_proposal"]}
    plan = {"schema": SCHEMA, "dataset": "hotpotqa", **plans["hotpotqa"],
        "reused_methods": [], "historical_methods_reference_only": list(caches),
        "drop_all_cache_replay_required": True,
        "source_priority": list(caches), "conflicting_cached_requests": report["summary"]["conflicting_cached_requests"],
        "historically_observed_oracle_actions": report["summary"]["observed_oracle_task_actions"],
        "historically_missing_oracle_actions": report["summary"]["missing_oracle_task_actions"],
        "maximum_team_schedules": len(baseline) * (count + 2),
        "maximum_new_team_inferences": len(baseline) * (count + 1),
        "existing_outcomes_reused_without_execution": 0,
        "baseline_network_calls_allowed": False, "other_network_calls_allowed": not suite.cache_only,
        "new_llm_calls": None, "notes": NOTE}
    return adapter, args, caches, report, historical, hashes, plan


def replay_commands(adapter, args, method, cache, plan_path=None):
    output = args.output_dir / method
    if plan_path is None:
        child = copy.copy(args)
        child.baseline_cache_seed_from = cache
        host = adapter.evaluation_command(child, mode="always_keep", output=output,
                                         resume=(output / "run_manifest.json").is_file())
    else:
        host = full.planned_command(adapter, args, output, plan_path, cache)
    command = [*host[:3], "causal_memory_control.oracle_replay_worker", *host[4:]]
    return host, command


def write_results(suite, args, results, historical):
    rows = []
    for method, label in (("native_gmemory", "Always Use (cache-only verified)"),
                          ("always_global_drop", "DROP ALL (canonical replay)"),
                          (original.ORACLE, "Oracle (canonical replay, post-hoc)")):
        result = results.get(method)
        rows.append({"Policy": label, "Answer F1 (%)": None if result is None else 100 * result[
            "mean_answer_f1" if method == "native_gmemory" else "gate_answer_f1"],
            "Answer EM (%)": None if result is None else 100 * result[
            "answer_exact_match_rate" if method == "native_gmemory" else "gate_answer_em"]})
    execution = {method: original.trained.read(args.output_dir / method / "summary.json")
                 for method in results if method != original.ORACLE}
    statistics = {"team_schedules": sum(value["completed_questions"] for value in execution.values()),
                  "new_llm_calls": sum(value["llm_calls"] for value in execution.values()),
                  "cache_hits": sum(value["cache_hits"] for value in execution.values()),
                  "count_scope": "legacy successful-call/hit counters on completed tasks; "
                                 "not total HTTP attempts, failed-task calls or a billing estimate"}
    save_json(suite.output_dir / "oracle_replay_results.json", {"schema": SCHEMA, "notes": NOTE,
        "execution_statistics": statistics,
        "rows": rows, "details": results,
        "historical_policy_results_reference_only_not_replayed": historical})
    with (suite.output_dir / "oracle_replay_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Policy | Answer F1 (%) | Answer EM (%) |", "| --- | --- | --- |"]
    lines += ["| " + " | ".join(str(value) if key == "Policy" else "—" if value is None else f"{value:.2f}"
                                for key, value in row.items()) + " |" for row in rows]
    (suite.output_dir / "oracle_replay_results.md").write_text("\n".join(lines) + "\n\n" + NOTE + "\n", encoding="utf-8")


def execute(suite, adapter, args, caches, report, historical, hashes, plan):
    # cache-only is permission to use the network, not a new sampling protocol;
    # a blocked cache-only probe may resume with permission, keeping all results.
    identity_plan = {k: v for k, v in plan.items() if k != "other_network_calls_allowed"}
    config = {"schema": SCHEMA, "evaluation_root": str(suite.evaluation_root),
              "plan": identity_plan, "input_sha256": hashes}
    manifest = suite.output_dir / "oracle_replay_manifest.json"
    if not manifest.exists() and any(p.name != ".replay.lock" for p in suite.output_dir.iterdir()):
        raise ValueError("use a NEW output directory, not the failed or completed old Oracle directory")
    full.save_plan(manifest, config)
    save_json(suite.output_dir / "source_audit.json", report)
    state, results = {"status": "running", "current_stage": None, "stages": {}}, {}

    def stage(label, callback):
        state["current_stage"] = label
        state["stages"][label] = "running"
        save_json(suite.output_dir / "oracle_replay_progress.json", state)
        value = callback()
        state["stages"][label] = "completed"
        save_json(suite.output_dir / "oracle_replay_progress.json", state)
        return value

    try:
        cache = args.output_dir / "cache_seeds" / "canonical.sqlite"
        stage("canonical_cache", lambda: canonical_cache(caches, cache, report["cache_audit"]))
        sources = {}
        count = args.successful_topk + args.insights_topk
        jobs = [("native_gmemory", None, 0), ("always_global_drop", "global_all", 0)] + [
            (f"oracle_branches/drop_{i:03d}", "candidate_index", i) for i in range(count)]
        for method, policy, index in jobs:
            output, path = args.output_dir / method, None
            if policy is not None:
                path = args.output_dir / "plans" / (method.replace("/", "_") + ".json")
                full.save_plan(path, build_plan(args.output_dir / "native_gmemory" / "memory_gate.jsonl",
                                                 policy=policy, index=index))
            host, command = replay_commands(adapter, args, method, cache, path)
            if method == "native_gmemory" or suite.cache_only:
                command.append("--cache-only")

            def evaluate():
                def complete():
                    return (adapter.evaluation_complete(args, output, mode="always_keep", command=host)
                            if path is None else full.planned_ready(adapter, args, output, host, path))

                if not complete():
                    full.base._run_command("hotpotqa:" + method, command,
                        suite.output_dir / "logs" / (method.replace("/", "_") + ".log"), api_key=args.api_key)
                if not complete():
                    raise RuntimeError(f"incomplete canonical replay: {output}")
                validate_ledger(output)
                if method == "native_gmemory":
                    parity = compare(suite.evaluation_root / "hotpotqa" / method / "memory_gate.jsonl",
                                     output / "memory_gate.jsonl", bootstrap_samples=1, seed=args.gate_evaluation_seed)
                    if not parity["paired_control_valid"]:
                        raise ValueError("cache-only baseline did not reproduce the old baseline")
                    return original.trained.read(output / "summary.json")
                return full.audited_comparison(args, output)

            results[method] = stage(method, evaluate)
            sources[method] = output / "memory_gate.jsonl"
            cache = output / "llm_cache.sqlite"
            write_results(suite, args, results, historical)
        results[original.ORACLE] = stage("oracle", lambda: full.build_oracle(args, sources))
        save_json(args.output_dir / original.ORACLE / "action_space.json", {"schema": SCHEMA,
            "actions": plan["action_space"], "notes": NOTE,
            "historical_policy_outcomes_used_for_selection": False})
        for path, expected in hashes.items():
            if file_sha256(Path(path)) != expected:
                raise ValueError(f"read-only source changed during replay: {path}")
        write_results(suite, args, results, historical)
        state.update(status="completed", current_stage=None)
        save_json(suite.output_dir / "oracle_replay_progress.json", state)
        print((suite.output_dir / "oracle_replay_results.md").read_text())
        return suite.output_dir / "oracle_replay_results.json"
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error={"type": type(exc).__name__, "message": str(exc)})
        save_json(suite.output_dir / "oracle_replay_progress.json", state)
        raise


def main(argv=None):
    suite = parse_args(argv)
    prepared = prepare(suite)
    plan = prepared[-1]
    print(json.dumps({"plan_only": suite.plan_only, "plan": plan}, ensure_ascii=False, indent=2), flush=True)
    if suite.plan_only:
        return plan
    manifest = suite.output_dir / "oracle_replay_manifest.json"
    if not manifest.exists() and suite.output_dir.exists() and any(
            p.name != ".replay.lock" for p in suite.output_dir.iterdir()):
        raise ValueError("use a NEW output directory, not the failed or completed old Oracle directory")
    lock_path = suite.output_dir / ".replay.lock"
    if lock_path.is_symlink():
        raise ValueError("replay lock must not be a symbolic link")
    suite.output_dir.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another process is already writing this replay output") from exc
        return execute(suite, *prepared)


if __name__ == "__main__":
    main()
