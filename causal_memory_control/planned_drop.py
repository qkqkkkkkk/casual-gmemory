"""Outcome-free, replayable interventions for control and oracle branch runs.

Plans contain candidate IDs and retrieval fingerprints, never gold answers or
rewards. Oracle selection happens only after all predeclared branches finish.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

from .jsonl import read_jsonl
from .runtime_gate import ExposureDecision, GMemoryExposureGate
from .types import ExposureAction, PotentialOutcomePrediction


PLAN_SCHEMA = "gmemory-fixed-intervention-plan-v1"


def indexed_rows(path):
    rows = read_jsonl(Path(path))
    indexed = {str(row["task_id"]): row for row in rows}
    if len(indexed) != len(rows) or not indexed:
        raise ValueError(f"empty or duplicate task IDs in {path}")
    return indexed


def build_plan(baseline_path, *, policy, seed=42, index=0, reference_path=None):
    """Choose interventions from pre-exposure candidates, not task outcomes."""
    baseline_path = Path(baseline_path)
    baseline = indexed_rows(baseline_path)
    reference = indexed_rows(reference_path) if reference_path else None
    if reference is not None and set(reference) != set(baseline):
        raise ValueError("random budget reference has different task IDs")
    if policy not in {"global_top1", "global_all", "random_one", "candidate_index"}:
        raise ValueError(f"unknown intervention policy {policy}")
    if index < 0:
        raise ValueError("candidate index must be non-negative")
    tasks = {}
    for task_id, row in baseline.items():
        decisions = row["decisions"]
        ids = [d["memory_id"] for d in decisions]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate candidate IDs for task {task_id}")
        if policy == "global_all":
            dropped = ids
        elif policy == "global_top1":
            dropped = [d["memory_id"] for d in decisions
                       if d["memory_type"] == "trajectory" and d["retrieval_rank"] == 1]
            if not dropped:
                raise ValueError(f"global_top1 requires a trajectory rank 1: task {task_id}")
        elif policy == "candidate_index":
            dropped = ids[index:index + 1]
        else:
            budget = 1
            if reference is not None:
                ref = reference[task_id]
                if ref["retrieval_fingerprint"] != row["retrieval_fingerprint"]:
                    raise ValueError(f"random reference retrieval mismatch: {task_id}")
                budget = int(ref["dropped_count"])
                if budget not in (0, 1):
                    raise ValueError("random matching supports only a zero/one budget")
            task_seed = int(hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest(), 16)
            dropped = random.Random(task_seed).sample(ids, min(budget, len(ids)))
        tasks[task_id] = {"retrieval_fingerprint": row["retrieval_fingerprint"],
                          "candidate_ids": ids, "drop_memory_ids": dropped}
    return {
        "schema": PLAN_SCHEMA, "policy": policy, "seed": seed, "candidate_index": index,
        "baseline_log_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "reference_log_sha256": hashlib.sha256(Path(reference_path).read_bytes()).hexdigest() if reference_path else None,
        "selection_uses_outcomes": False, "tasks": tasks,
    }


def load_plan(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != PLAN_SCHEMA or payload.get("selection_uses_outcomes") is not False:
        raise ValueError("invalid or outcome-selected intervention plan")
    if not isinstance(payload.get("tasks"), dict) or not payload["tasks"]:
        raise ValueError("intervention plan has no tasks")
    for task_id, task in payload["tasks"].items():
        ids, drops = task["candidate_ids"], task["drop_memory_ids"]
        if (len(ids) != len(set(ids)) or len(drops) != len(set(drops))
                or not set(drops) <= set(ids) or not task.get("retrieval_fingerprint")):
            raise ValueError(f"invalid intervention candidates for {task_id}")
        if payload["policy"] != "global_all" and len(drops) > 1:
            raise ValueError("single-candidate policy has multiple DROP IDs")
    return payload


class PlannedDropGate(GMemoryExposureGate):
    """Execute a fixed, audited plan at the native retrieval boundary."""

    MODES = GMemoryExposureGate.MODES | {"planned_drop"}

    def __init__(self, plan_path, **kwargs):
        kwargs.pop("mode", None)
        self.plan = load_plan(plan_path)
        super().__init__(mode="planned_drop", **kwargs)

    def begin_task(self, task_id, task_config):
        if str(task_id) not in self.plan["tasks"]:
            raise ValueError(f"task {task_id} is missing from the intervention plan")
        super().begin_task(task_id, task_config)

    def _decide(self, event):
        dropped = event.memory.memory_id in self.plan["tasks"][str(self._task_id)]["drop_memory_ids"]
        prediction = PotentialOutcomePrediction(0, 0, 0, "neutral", 0,
                                                 "fixed_intervention_not_causal_prediction", False, 0)
        return ExposureDecision(event.event_id, event.memory.memory_id, event.memory.memory_type,
                                ExposureAction.DROP if dropped else ExposureAction.KEEP,
                                0.0, "planned_drop" if dropped else "planned_keep", prediction)

    def filter_retrieval(self, *args, **kwargs):
        result = super().filter_retrieval(*args, **kwargs)
        expected = self.plan["tasks"][str(self._task_id)]
        if self._retrieval_fingerprint != expected["retrieval_fingerprint"]:
            raise ValueError(f"planned intervention retrieval changed for task {self._task_id}")
        if [d.memory_id for d in self._task_decisions] != expected["candidate_ids"]:
            raise ValueError(f"planned intervention candidate order changed for task {self._task_id}")
        actual = [d.memory_id for d in self._task_decisions if d.action == ExposureAction.DROP]
        if set(actual) != set(expected["drop_memory_ids"]):
            raise ValueError(f"planned DROP was not executed for task {self._task_id}")
        self._task_decisions = [replace(d, policy_scores={"plan_policy": self.plan["policy"]})
                                for d in self._task_decisions]
        return result


def validate_plan_args(args):
    path = getattr(args, "intervention_plan", None)
    if args.mode == "planned_drop":
        if path is None or not path.is_file():
            raise SystemExit("--mode planned_drop requires an existing --intervention-plan")
        load_plan(path)
    elif path is not None:
        raise SystemExit("--intervention-plan is only valid with --mode planned_drop")


def plan_run_metadata(args, selected_ids):
    if args.mode != "planned_drop":
        return {}
    path = args.intervention_plan.resolve()
    if set(load_plan(path)["tasks"]) != {str(value) for value in selected_ids}:
        raise SystemExit("intervention plan does not match the selected evaluation task IDs")
    return {"intervention_plan": str(path),
            "intervention_plan_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
