"""Read-only inventory for repairing a completed evaluation's Oracle inputs.

No model imports/calls, cache merging, response selection or experiment writes.
An optional new JSON report records evidence, not an executable replay plan.
Legacy caches do not map requests to tasks: six differing outcomes do NOT prove
that only six tasks used noncanonical responses.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3


SCHEMA = "oracle-replay-audit-v1"
SOURCE_PRIORITY = (
    "native_gmemory", "always_global_drop", "linear", "semantic_linear",
    "semantic_mlp", "semantic_st", "csm_st",
)
TABLES = (("responses", "response"), ("binary_scores", "payload_json"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Observation:
    task_id: str
    fingerprint: str
    candidates: tuple[str, ...]
    drops: tuple[str, ...]
    outcome: tuple[float, bool, str | None]


def read_observations(path):
    rows, metadata = {}, None
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row["task_id"])
            if task_id in rows:
                raise ValueError(f"duplicate task {task_id}: {path}")
            current = {k: v for k, v in row["run_metadata"].items() if k != "memory_sha256"}
            if metadata is None:
                metadata = current
            if current != metadata:
                raise ValueError(f"mixed run metadata: {path}")
            decisions = row["decisions"]
            candidates = tuple(sorted(d["memory_id"] for d in decisions))
            if len(set(candidates)) != len(candidates):
                raise ValueError(f"duplicate candidate IDs: {path}, task {task_id}")
            if any(d["action"] not in {"keep", "drop"} for d in decisions):
                raise ValueError(f"unknown action: {path}, task {task_id}")
            drops = tuple(sorted(d["memory_id"] for d in decisions if d["action"] == "drop"))
            reward, done = float(row["reward"]), row["done"]
            prediction = row.get("outcome", {}).get("prediction")
            if (not math.isfinite(reward) or not isinstance(done, bool)
                    or prediction is not None and not isinstance(prediction, str)):
                raise ValueError(f"invalid outcome: {path}, task {task_id}")
            rows[task_id] = Observation(task_id, row["retrieval_fingerprint"], candidates,
                                        drops, (reward, done, prediction))
    if not rows:
        raise ValueError(f"empty evaluation: {path}")
    return rows, metadata


def scan_caches(caches):
    """Hash all values, retaining no answer text and never creating a database."""
    variants, counts = {}, {}
    for method, path in caches.items():
        counts[method] = {}
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "responses" not in tables:
                raise ValueError(f"missing responses table: {path}")
            for table, column in TABLES:
                if table not in tables:
                    continue
                count = 0
                for key, value in db.execute(f"SELECT key, {column} FROM {table}"):
                    if not isinstance(key, str) or not isinstance(value, str):
                        raise ValueError(f"non-text cache entry: {path}, {table}")
                    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
                    versions = variants.setdefault((table, key), {})
                    versions.setdefault(digest, []).append(method)
                    count += 1
                counts[method][table] = count
    conflicts = []
    for (table, key), versions in sorted(variants.items()):
        if len(versions) < 2:
            continue
        first_digest = next(iter(versions))
        conflicts.append({"table": table, "key": key,
            "proposed_canonical_source": versions[first_digest][0],
            "proposed_response_sha256": first_digest,
            "variants": [{"response_sha256": digest, "methods": owners}
                         for digest, owners in versions.items()]})
    return {"unique_cached_requests": len(variants), "source_rows": counts,
            "conflicting_cached_requests": len(conflicts), "conflicts": conflicts}


def audit(evaluation_root, dataset):
    root = Path(evaluation_root).resolve()
    identities = {}

    def source(path):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        identities[str(path)] = sha256(path)
        return path

    manifest = read(source(root / "reevaluation_manifest.json"))
    if manifest.get("schema") != "trained-gates-evaluation-v1":
        raise ValueError("expected a completed run_trained_gate_evaluation root")
    config = manifest["datasets"][dataset]
    registered = config["methods"]
    if (len(set(registered)) != len(registered) or "native_gmemory" not in registered
            or set(registered) - set(SOURCE_PRIORITY)):
        raise ValueError("invalid/unsupported registered policy list")
    methods = [m for m in SOURCE_PRIORITY if m in registered]
    observations, caches, reference_metadata = {}, {}, None
    for method in methods:
        policy = root / dataset / method
        progress = read(source(policy / "progress.json"))
        if progress.get("status") != "completed":
            raise ValueError(f"source evaluation is not completed: {policy}")
        run = read(source(policy / "run_manifest.json"))["run"]
        rows, metadata = read_observations(source(policy / "memory_gate.jsonl"))
        if len(rows) != config["test_tasks"]:
            raise ValueError(f"task count differs from registered test set: {policy}")
        for key in ("model", "endpoint", "seed"):
            if run.get(key) != config[key] or metadata.get(key) != config[key]:
                raise ValueError(f"source {key} differs: {policy}")
        if any(run.get(key) != value for key, value in metadata.items()):
            raise ValueError(f"run manifest and row metadata disagree: {policy}")
        if method == "native_gmemory":
            reference_metadata = metadata
            if any(row.drops for row in rows.values()):
                raise ValueError("baseline is not KEEP ALL")
        else:
            baseline = observations["native_gmemory"]
            if set(rows) != set(baseline) or metadata != reference_metadata:
                raise ValueError(f"task IDs or run metadata differ: {policy}")
            for task_id, row in rows.items():
                original = baseline[task_id]
                if (row.fingerprint != original.fingerprint or row.candidates != original.candidates):
                    raise ValueError(f"retrieval differs: {policy}, task {task_id}")
                if method == "always_global_drop" and row.drops != row.candidates:
                    raise ValueError("Global Drop does not hide all candidates")
                if not row.drops and row.outcome != original.outcome:
                    raise ValueError(f"no-DROP outcome differs: {policy}, task {task_id}")
        observations[method] = rows
        caches[method] = source(policy / "llm_cache.sqlite")

    cache_report = scan_caches(caches)
    groups = defaultdict(list)
    for method, rows in observations.items():
        for task_id, row in rows.items():
            groups[(task_id, row.drops)].append((method, row.outcome))
    conflicts = []
    for (task_id, drops), values in sorted(groups.items()):
        if len({outcome for _, outcome in values}) < 2:
            continue
        conflicts.append({"task_id": task_id, "drop_memory_ids": list(drops),
            "score_differs": len({outcome[:2] for _, outcome in values}) > 1,
            "observations": [{"method": method, "reward": value[0], "done": value[1],
                              "prediction": value[2]} for method, value in values]})

    required = {}
    for task_id, row in observations["native_gmemory"].items():
        required[(task_id, ())] = "keep_all"
        required[(task_id, row.candidates)] = "drop_all"
        for candidate in row.candidates:
            required.setdefault((task_id, (candidate,)), "single_drop")
    missing = [{"task_id": task_id, "drop_memory_ids": list(drops), "kind": kind}
               for (task_id, drops), kind in sorted(required.items()) if (task_id, drops) not in groups]
    # Changed files invalidate both the hashes and conclusions of this audit.
    for path, expected in identities.items():
        if sha256(path) != expected:
            raise ValueError(f"source changed during audit; stop writers and retry: {path}")
    affected = sorted({c["task_id"] for c in conflicts})
    summary = {
        "test_tasks": config["test_tasks"], "policy_task_records": sum(map(len, observations.values())),
        "distinct_observed_task_actions": len(groups),
        "repeated_task_action_groups": sum(len(values) > 1 for values in groups.values()),
        "inconsistent_outcome_action_groups": len(conflicts),
        "score_inconsistent_action_groups": sum(c["score_differs"] for c in conflicts),
        "known_inconsistent_tasks": len(affected),
        "unique_cached_requests": cache_report["unique_cached_requests"],
        "conflicting_cached_requests": cache_report["conflicting_cached_requests"],
        "required_oracle_task_actions": len(required),
        "observed_oracle_task_actions": len(required) - len(missing),
        "missing_oracle_task_actions": len(missing),
        "missing_by_kind": dict(Counter(row["kind"] for row in missing)),
        "oracle_action_groups_with_conflicting_observations": sum(
            (c["task_id"], tuple(c["drop_memory_ids"])) in required for c in conflicts),
    }
    return {"schema": SCHEMA, "evaluation_root": str(root), "dataset": dataset,
        "protocol": {**{k: config[k] for k in ("model", "endpoint", "seed", "test_offset")},
                     "temperature": reference_metadata.get("temperature")},
        "source_priority_proposal": methods, "source_priority_applied": False,
        "summary": summary, "cache_audit": cache_report,
        "same_action_conflicts": conflicts, "known_inconsistent_task_ids": affected,
        "missing_oracle_actions": missing, "source_sha256": identities,
        "replay_scope": {"exact": False, "reason": "legacy caches have no per-task request ledger",
            "known_conflicts_are_not_an_exact_replay_list": True,
            "cache_only_task_traces_upper_bound": summary["policy_task_records"],
            "required_follow_up": "trace old executions with their own caches, network disabled; "
                "map consumed response hashes to task/action, then replay affected actions under one canonical cache",
            "new_llm_calls": None},
        "notes": "Read-only preflight, not an execution plan or a full source/snapshot/checkpoint audit. "
            "Missing actions count team schedules, not LLM calls. Existing observed actions are not "
            "automatically reusable under a canonical cache. Cache source priority is an outcome-blind "
            "proposal only; no conflict is resolved, no response is discarded and no score is changed. "
            "Keep FEVER and all original artifacts unchanged. No training, collection or memory rebuilding."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("fever", "hotpotqa"), default="hotpotqa")
    parser.add_argument("--output-report", type=Path,
                        help="Optional NEW JSON report, outside original evaluation; never overwrite")
    args = parser.parse_args(argv)
    root = args.evaluation_root.resolve()
    if args.output_report is not None:
        report = args.output_report.resolve()
        if (report == root or root in report.parents or args.output_report.is_symlink()
                or report.exists()):
            parser.error("--output-report must be a new file outside the source evaluation")
    result = audit(root, args.dataset)
    if args.output_report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"dataset": args.dataset, "summary": result["summary"],
        "known_inconsistent_task_ids": result["known_inconsistent_task_ids"],
        "exact_replay_scope": False,
        "report": str(report) if args.output_report is not None else None,
        "notes": result["notes"]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
