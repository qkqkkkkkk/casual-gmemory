"""Fail-closed validation for frozen FEVER snapshot manifests."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


SNAPSHOT_SCHEMA = "native-gmemory-macnet-fever-evidence-snapshot-v3"


class SnapshotProvenanceError(ValueError):
    pass


def stable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def reconcile_retest_design(
    current_design: Mapping[str, Any],
    previous_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Align a retest with seed 0 while ignoring Chroma byte-level drift.

    Opening a persistent Chroma collection can change SQLite/WAL bytes even
    when GMemory is read-only and its semantic records are unchanged.  Every
    experimental field must still match; only the physical directory hash may
    be inherited from the registered seed-0 design.
    """
    previous_design = previous_manifest.get("design")
    if not isinstance(previous_design, dict):
        raise SnapshotProvenanceError(
            "retest seed-0 manifest has no design payload"
        )
    previous_hash = previous_manifest.get("design_hash")
    if stable_hash(previous_design) != previous_hash:
        raise SnapshotProvenanceError(
            "retest seed-0 manifest has an invalid design hash"
        )

    ignored = {"memory_sha256"}
    current_semantic = {
        key: value for key, value in current_design.items() if key not in ignored
    }
    previous_semantic = {
        key: value for key, value in previous_design.items() if key not in ignored
    }
    if current_semantic != previous_semantic:
        keys = sorted(set(current_semantic) | set(previous_semantic))
        mismatches = [
            key
            for key in keys
            if current_semantic.get(key) != previous_semantic.get(key)
        ]
        raise SnapshotProvenanceError(
            "retest semantic design differs from seed 0 in: "
            + ", ".join(mismatches)
        )

    aligned = dict(current_design)
    if "memory_sha256" in previous_design:
        aligned["memory_sha256"] = previous_design["memory_sha256"]
    if stable_hash(aligned) != previous_hash:
        raise SnapshotProvenanceError(
            "retest design could not be aligned with seed 0"
        )
    return aligned


def validate_snapshot_manifest(
    manifest: Mapping[str, Any],
    *,
    source_md5: str,
    model: str,
    graph_type: str,
    node_num: int,
    embedding_model: str,
    claims: int,
    candidate_kind: str,
) -> dict[str, Any]:
    if manifest.get("snapshot_schema") != SNAPSHOT_SCHEMA:
        raise SnapshotProvenanceError(
            "memory directory is not a registered FEVER snapshot"
        )
    design = manifest.get("design")
    if not isinstance(design, dict):
        raise SnapshotProvenanceError(
            "snapshot manifest has no immutable design payload"
        )
    if stable_hash(design) != manifest.get("design_hash"):
        raise SnapshotProvenanceError("snapshot manifest design hash is invalid")
    for key, value in design.items():
        if manifest.get(key) != value:
            raise SnapshotProvenanceError(
                f"snapshot manifest duplicates disagree for {key}"
            )

    checks = {
        "source_md5": source_md5,
        "model": model,
        "graph_type": graph_type,
        "node_num": int(node_num),
        "embedding_model": embedding_model,
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise SnapshotProvenanceError(
                f"snapshot {key}={manifest.get(key)!r}, "
                f"but runner requested {expected!r}"
            )

    support_ids = [int(value) for value in manifest.get("support_ids", [])]
    evaluation_ids = [int(value) for value in manifest.get("evaluation_ids", [])]
    if not support_ids or not evaluation_ids:
        raise SnapshotProvenanceError(
            "snapshot manifest has no registered support/evaluation split"
        )
    if len(set(support_ids)) != len(support_ids):
        raise SnapshotProvenanceError("snapshot support ids contain duplicates")
    if len(set(evaluation_ids)) != len(evaluation_ids):
        raise SnapshotProvenanceError("snapshot evaluation ids contain duplicates")
    if set(support_ids) & set(evaluation_ids):
        raise SnapshotProvenanceError(
            "snapshot manifest contains support/evaluation leakage"
        )
    if int(manifest.get("memory_records", -1)) != len(support_ids):
        raise SnapshotProvenanceError(
            "snapshot memory record count does not match support split"
        )
    if candidate_kind == "trajectory" and int(
        manifest.get("successful_total", 0)
    ) < 1:
        raise SnapshotProvenanceError(
            "snapshot has no successful trajectory; trajectory intervention is undefined"
        )
    if int(claims) > len(evaluation_ids):
        raise SnapshotProvenanceError(
            f"--claims {claims} exceeds {len(evaluation_ids)} registered evaluation claims"
        )
    return dict(manifest)
