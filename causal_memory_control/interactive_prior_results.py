"""Read-only import of completed QA logs for the four-dataset report."""
from __future__ import annotations

import json
from pathlib import Path

from .interactive_data import digest, sha

METHODS = {"always_use": "native_gmemory", "global_drop": "always_global_drop",
           "random_drop": "random_one", "semantic_linear": "semantic_linear",
           "semantic_mlp": "semantic_mlp", "semantic_st": "semantic_st", "csm_st": "csm_st",
           "oracle": "oracle_selective_drop"}


def import_qa(root):
    from .compare_gate_runs import _load, compare

    root = Path(root).resolve()
    original = root
    oracle_run = root / "oracle_run_manifest.json"
    if oracle_run.exists():
        original = Path(json.loads(oracle_run.read_text())["evaluation_root"])
    registration_path = original / "reevaluation_manifest.json"
    registration = json.loads(registration_path.read_text())
    if registration.get("schema") != "trained-gates-evaluation-v1":
        raise ValueError("QA import requires completed trained-gate evaluation or its Oracle extension")
    result = {}
    for name in ("fever", "hotpotqa"):
        if name not in registration["datasets"]:
            continue
        baseline = original / name / "native_gmemory/memory_gate.jsonl"
        baseline_manifest = original / name / "native_gmemory/run_manifest.json"
        design = json.loads(baseline_manifest.read_text())["design"]
        if design.get("task_state_policy") != "isolated-task-state-v1":
            raise ValueError("QA baseline predates task isolation")
        registered = set(map(str, design["evaluation_ids"]))
        base = _load(baseline)
        if set(base) != registered or len(base) != registration["datasets"][name]["test_tasks"]:
            raise ValueError("QA baseline does not cover its registered held-out IDs")
        methods = {}
        for policy, method in METHODS.items():
            path = (root if policy == "oracle" else original) / name / method / "memory_gate.jsonl"
            if not path.exists():
                continue
            if policy != "oracle":
                method_manifest = path.parent / "run_manifest.json"
                method_design = json.loads(method_manifest.read_text())["design"]
                if (method_design.get("task_state_policy") != "isolated-task-state-v1"
                        or set(map(str, method_design["evaluation_ids"])) != registered):
                    raise ValueError("QA policy uses a different task isolation or partition")
            comparison = compare(baseline, path, bootstrap_samples=2000, seed=71)
            if not comparison["paired_control_valid"] or not comparison["policy_valid"]:
                raise ValueError(f"QA paired audit failed: {path}")
            rows = _load(path)
            if not all(r.get("retrieval_fingerprint") for r in rows.values()):
                raise ValueError("QA log lacks original retrieval fingerprints")
            if policy == "global_drop" and any(r["dropped_count"] != r["retrieved_count"] for r in rows.values()):
                raise ValueError("QA Global Drop did not hide all candidates")
            if policy == "oracle":
                manifest_path = path.parent / "oracle_manifest.json"
                oracle_manifest = json.loads(manifest_path.read_text())
                space = json.loads((path.parent / "action_space.json").read_text())
                if "DROP ALL" not in json.dumps(space) or "always_global_drop" not in oracle_manifest["sources"]:
                    raise ValueError("QA Oracle does not declare DROP ALL")
                drop = original / name / "always_global_drop/memory_gate.jsonl"
                if oracle_manifest["sources"]["always_global_drop"] != sha(drop):
                    raise ValueError("QA Oracle used a different Global Drop source")
                drop_rows = _load(drop)
                if any(rows[t]["reward"] < drop_rows[t]["reward"] for t in rows):
                    raise ValueError("QA Oracle primary reward is below its DROP ALL candidate")
            methods[policy] = {"primary": comparison["gate_mean_reward"],
                "answer_em": comparison["gate_completion_rate"] if name == "hotpotqa" else None,
                "paired_delta": comparison["reward_delta"], "paired_delta_95ci": comparison["reward_delta_95ci"],
                "source_root": str(path.parent), "log_sha256": sha(path), "baseline_log_sha256": sha(baseline),
                "task_ids": sorted(rows), "task_ids_sha256": digest(sorted(rows)),
                "run_metadata": next(iter(rows.values()))["run_metadata"],
                "checkpoint_metadata": next(iter(rows.values())).get("checkpoint_metadata", {}),
                "action_budget": {"max_drops": next(iter(rows.values())).get("max_drops"),
                                  "mode": next(iter(rows.values())).get("mode")}}
        result[name] = {"metric": "binary_accuracy" if name == "fever" else "answer_f1_and_em",
            "registration_sha256": sha(registration_path), "baseline_manifest_sha256": sha(baseline_manifest),
            "policies": methods, "source_root": str(root)}
    return result
