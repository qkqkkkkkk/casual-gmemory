"""Bounded-raw-memory loading for full ST diagnostics.

Spool compact branch records to temporary SQLite, group one task at a time, and
retain float32 feature arrays instead of millions of repeated Python dict keys.
Original diagnostics are read-only. Contexts and rollout matching stay strict.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm


class CompactFeatures(Mapping):
    def __init__(self, features, layouts):
        names = tuple(sorted(features))
        self.names, self.index = layouts.setdefault(names, (names, {k: i for i, k in enumerate(names)}))
        self.values = np.asarray([features[k] for k in names], dtype="float32")

    def __getitem__(self, key):
        return float(self.values[self.index[key]])

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)


def feature_matrix(rows, names):
    result = np.zeros((len(rows), len(names)), dtype="float32")
    lookup = {name: i for i, name in enumerate(names)}
    columns = {}
    for i, row in enumerate(rows):
        if isinstance(row, CompactFeatures):
            if row.names not in columns:
                known = [(lookup[k], j) for j, k in enumerate(row.names) if k in lookup]
                columns[row.names] = ([a for a, _ in known], [b for _, b in known])
            target, source = columns[row.names]
            result[i, target] = row.values[source]
        else:
            result[i] = [row.get(k, 0) for k in names]
    return result


def load_st_examples(sources, *, metric, task_ids, hash_dimensions, feature_version,
                     memory_dir, feature_builder, event_sink, progress_label="ST"):
    from . import train_gate as training
    from .insight_metadata import enrich_insight_payload, load_insight_catalog

    catalog = load_insight_catalog(memory_dir) if memory_dir else {}
    snapshot = json.loads((memory_dir / "causal_snapshot_manifest.json").read_text()) if memory_dir else None
    examples, layouts = [], {}
    # Scratch files are automatically removed, including on exceptions; no user
    # result, diagnostic file or SQLite cache is modified/deleted.
    with tempfile.TemporaryDirectory(prefix="gmemory-st-branches-") as scratch:
        with sqlite3.connect(str(Path(scratch) / "branches.sqlite")) as db:
            db.execute("CREATE TABLE contexts(task TEXT, candidate TEXT, payload TEXT, digest TEXT, PRIMARY KEY(task,candidate))")
            db.execute("CREATE TABLE branches(task TEXT, candidate TEXT, condition TEXT, rollout TEXT, reward REAL, PRIMARY KEY(task,candidate,condition,rollout))")
            for source in sources:
                path = source / "branches.jsonl" if source.is_dir() else source
                if path.suffix != ".jsonl":
                    raise ValueError("full ST requires original branch JSONL files")
                manifest_path = path.parent / "run_manifest.json"
                manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
                if snapshot is not None:
                    if manifest["design"]["snapshot_design_hash"] != snapshot["design_hash"]:
                        raise ValueError("--memory-dir is not the diagnostic's registered snapshot")
                repeats = manifest.get("design", {}).get("repeats")
                declared = manifest.get("design", {}).get("evaluation_ids")
                declared = {str(t) for t in declared if task_ids is None or int(t) in task_ids} if declared is not None else None
                observed, expected_candidates, observed_candidates = {}, {}, {}
                with path.open(encoding="utf-8") as handle:
                    for number, line in enumerate(tqdm(handle, desc=f"Read {progress_label} {path.parent.name}", unit="branch"), 1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except ValueError as exc:
                            raise ValueError(f"invalid diagnostic JSON at {path}:{number}") from exc
                        if "condition" not in row:
                            raise ValueError("ST requires measured branch rows, not direct precomputed examples")
                        if task_ids is not None and int(row["task_id"]) not in task_ids:
                            continue
                        if row["condition"] not in ("use_all", "global_drop"):
                            continue
                        context = {k: row.get(k) for k in ("candidate", "candidate_set", "task", "retrieval_sizes")}
                        if not isinstance(context["candidate_set"], list) or not context["candidate_set"]:
                            raise ValueError("ST needs the complete logged candidate_set; recollect missing contexts instead of inventing peers")
                        context["exposure_count"] = training._exposure_count(row)
                        if catalog:
                            context["candidate"] = enrich_insight_payload(context["candidate"], catalog) if context["candidate"]["kind"] == "insight" else context["candidate"]
                            context["candidate_set"] = [enrich_insight_payload(p, catalog) if p["kind"] == "insight" else p for p in context["candidate_set"]]
                        task, candidate = str(row["task_id"]), str(context["candidate"]["candidate_id"])
                        expected_candidates.setdefault(task, {str(c["candidate_id"]) for c in context["candidate_set"]})
                        observed_candidates.setdefault(task, set()).add(candidate)
                        observed.setdefault((task, candidate), set()).add((row["condition"], row.get("repeat_index")))
                        if repeats is not None:
                            repeat = row.get("repeat_index")
                            seed_base = manifest.get("run", {}).get("sample_seed_base")
                            if not isinstance(repeat, int) or not 0 <= repeat < repeats or (
                                    seed_base is not None and row.get("sample_seed") != seed_base + repeat):
                                raise ValueError(f"invalid declared ST repeat/seed at {path}:{number}")
                        payload = json.dumps(context, sort_keys=True, ensure_ascii=False)
                        digest = hashlib.sha256(payload.encode()).hexdigest()
                        existing = db.execute("SELECT digest FROM contexts WHERE task=? AND candidate=?", (task, candidate)).fetchone()
                        if existing is not None and existing[0] != digest:
                            raise ValueError(f"matched branch context differs for task {task}, candidate {candidate}")
                        if existing is None:
                            db.execute("INSERT INTO contexts VALUES (?,?,?,?)", (task, candidate, payload, digest))
                        rollout = json.dumps([row.get("run_hash"), row.get("sample_seed"), row.get("repeat_index")])
                        reward = float(row["outcome"][metric])
                        if not np.isfinite(reward) or not 0 <= reward <= 1:
                            raise ValueError(f"invalid bounded ST reward at {path}:{number}")
                        try:
                            db.execute("INSERT INTO branches VALUES (?,?,?,?,?)", (task, candidate, row["condition"], rollout, reward))
                        except sqlite3.IntegrityError as exc:
                            raise ValueError(f"duplicate counterfactual branch for task {task}, candidate {candidate}, rollout {rollout}") from exc
                if declared is not None and set(expected_candidates) != declared:
                    raise ValueError(f"ST source does not cover declared diagnostic task IDs: {path}")
                for task, candidates in expected_candidates.items():
                    if observed_candidates[task] != candidates:
                        raise ValueError(f"ST needs counterfactual coverage for every retrieved candidate in task {task}: {path}")
                if repeats is not None and any(len(cells) != 2 * repeats for cells in observed.values()):
                    raise ValueError(f"missing declared USE/DROP repeats in ST source: {path}")
                db.commit()
            tasks = [r[0] for r in db.execute("SELECT DISTINCT task FROM contexts ORDER BY task")]
            for task in tqdm(tasks, desc=f"Encode {progress_label} task lists", unit="task"):
                contexts = {r[0]: json.loads(r[1]) for r in db.execute("SELECT candidate,payload FROM contexts WHERE task=?", (task,))}
                first_context = next(iter(contexts.values()))
                for context in contexts.values():
                    if any(context.get(k) != first_context.get(k) for k in ("candidate_set", "task", "retrieval_sizes", "exposure_count")):
                        raise ValueError(f"ST list contexts differ across candidates for task {task}")
                expected = {str(c["candidate_id"]) for c in first_context["candidate_set"]}
                if set(contexts) != expected:
                    raise ValueError(f"ST needs counterfactual coverage for every retrieved candidate in task {task}")
                rows, conditions = [], {}
                for candidate, condition, rollout, reward in db.execute("SELECT candidate,condition,rollout,reward FROM branches WHERE task=?", (task,)):
                    context = contexts[candidate]
                    run, seed, repeat = json.loads(rollout)
                    conditions.setdefault(candidate, set()).add(condition)
                    rows.append({**context, "task_id": int(task), "condition": condition,
                        "run_hash": run, "sample_seed": seed, "repeat_index": repeat, "outcome": {metric: reward}})
                if any(conditions.get(c) != {"use_all", "global_drop"} for c in expected):
                    raise ValueError(f"missing USE or DROP branches for ST task {task}")
                events = []
                batch = training._examples_from_branches(rows, metric=metric, task_ids=None,
                    hash_dimensions=hash_dimensions, feature_version=feature_version,
                    feature_builder=feature_builder, event_sink=events, progress=False)
                for (event, _), example in zip(events, batch):
                    compact = replace(example, features=CompactFeatures(example.features, layouts))
                    examples.append(compact)
                    fingerprint = hashlib.sha256((event.memory.memory_type + "\n" + event.memory.content).encode()).hexdigest()
                    event_sink.append((SimpleNamespace(content_fingerprint=fingerprint), compact))
    return examples
