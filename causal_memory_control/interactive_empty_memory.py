"""Explicit no-supervision outcomes for interactive memory selection.

An empty candidate set is a valid observation, not a zero utility label. A
training skip is a separate artifact and must never impersonate a checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

from .interactive_data import digest, locked, sha
from .interactive_protocol import learned_methods


SCHEMA = "interactive-no-candidate-supervision-v1"


def verify_registered_code(config):
    """Keep the original manifest immutable; accept only an explicit upgrade."""
    repo = Path(config["repo"])
    changed = {path for path, expected in config["code_sha256"].items() if sha(repo / path) != expected}
    if not changed:
        return
    registration = Path(config["output_dir"]) / "empty_memory_upgrade/registration.json"
    if not registration.exists():
        raise ValueError("registered implementation changed; use continue_interactive_empty_memory "
                         "to back up and register the empty-memory upgrade")
    from .continue_interactive_empty_memory import upgrade_registration
    if json.loads(registration.read_text()) != upgrade_registration(config):
        raise ValueError("empty-memory upgrade registration or implementation changed")


def upgrade_provenance(config):
    path = Path(config["output_dir"]) / "empty_memory_upgrade/registration.json"
    if not path.exists():
        return None
    verify_registered_code(config)
    return {"registration_sha256": sha(path), "original_config_digest": digest(config),
            "note": "original measurement identities retained; stage implementation upgrade explicitly registered"}


def coverage(rows, data):
    result = {}
    for part in ("fit", "dev"):
        ids = set(data["split"][part])
        selected = [row for row in rows if row["task_id"] in ids]
        if len(selected) != len(ids) or {row["task_id"] for row in selected} != ids:
            raise ValueError(f"incomplete diagnostic {part} task coverage")
        result[part] = {
            "registered_task_ids": sorted(ids),
            "supervised_task_ids": sorted(row["task_id"] for row in selected if row["candidates"]),
            "empty_task_ids": sorted(row["task_id"] for row in selected if not row["candidates"]),
            "candidate_count": sum(len(row["candidates"]) for row in selected),
        }
    return result


def skip_payload(config, name, rows, method):
    from .interactive_training import diagnostic_lists
    from .interactive_team_u import measured_lists

    root = Path(config["output_dir"]) / name
    data = config["experiments"][name]
    # Reload all registered rows when validating a skip, rather than trusting a
    # hand-authored status file or inferring emptiness from support success.
    rows = diagnostic_lists(root, data) if rows is None else rows
    counts = coverage(rows, data)
    if any(counts[part]["candidate_count"] for part in counts):
        raise ValueError("no-supervision status requires every fit/dev list to be empty")
    _, source_hashes = measured_lists(config, name)
    return {"schema": SCHEMA, "config_digest": digest(config), "dataset": name,
        "method": method, "status": "skipped_no_candidate_supervision", "trained": False,
        "checkpoint_created": False, "coverage": counts, "measured_source_sha256": source_hashes,
        "diagnostic_sha256": {row["task_id"]: sha(root / "diagnostic/tasks" /
            (digest(row["task_id"]) + ".json")) for row in rows},
        "deployment": "empty_lists_only; nonempty_test_candidates_require_a_trained_gate"}


def save_skip(config, name, rows, methods):
    root = Path(config["output_dir"]) / name
    for method in methods:
        if (root / method / "checkpoint.json").exists():
            raise ValueError("no-supervision status conflicts with an existing checkpoint")
        locked(root / method / "training_status.json", skip_payload(config, name, rows, method))
        print(f"[训练] {name}/{method}：候选监督为 0，记录未训练；不创建 checkpoint。", flush=True)


def validated_skip(config, name, method):
    root = Path(config["output_dir"]) / name / method
    path = root / "training_status.json"
    if not path.exists():
        return None
    if (root / "checkpoint.json").exists():
        raise ValueError("checkpoint and no-supervision status cannot coexist")
    value = json.loads(path.read_text())
    if value != skip_payload(config, name, None, method):
        raise ValueError("no-supervision source or configuration changed")
    return value


def training_artifacts(config, name):
    root = Path(config["output_dir"]) / name
    checkpoints, skipped = {}, {}
    for method in learned_methods(config):
        status = validated_skip(config, name, method)
        if status is None:
            checkpoints[method] = sha(root / method / "checkpoint.json")
        else:
            skipped[method] = {"status": status["status"], "trained": False,
                "training_status_sha256": sha(root / method / "training_status.json")}
    return checkpoints, skipped
