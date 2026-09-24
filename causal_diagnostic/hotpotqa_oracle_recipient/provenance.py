"""Fail-closed validation for frozen HotpotQA snapshot manifests."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


SNAPSHOT_SCHEMA = "native-gmemory-macnet-hotpotqa-snapshot-v1"


class SnapshotProvenanceError(ValueError):
    pass


def stable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def reconcile_retest_design(
    current_design: Mapping[str, Any], previous_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    previous = previous_manifest.get("design")
    if not isinstance(previous, dict):
        raise SnapshotProvenanceError("retest manifest has no design payload")
    previous_hash = previous_manifest.get("design_hash")
    if stable_hash(previous) != previous_hash:
        raise SnapshotProvenanceError("retest manifest has an invalid design hash")
    ignored = {"memory_sha256"}
    current_semantic = {k: v for k, v in current_design.items() if k not in ignored}
    previous_semantic = {k: v for k, v in previous.items() if k not in ignored}
    if current_semantic != previous_semantic:
        keys = sorted(set(current_semantic) | set(previous_semantic))
        mismatches = [
            key
            for key in keys
            if current_semantic.get(key) != previous_semantic.get(key)
        ]
        raise SnapshotProvenanceError(
            "retest semantic design differs from seed 0 in: " + ", ".join(mismatches)
        )
    aligned = dict(current_design)
    if "memory_sha256" in previous:
        aligned["memory_sha256"] = previous["memory_sha256"]
    if stable_hash(aligned) != previous_hash:
        raise SnapshotProvenanceError("retest design could not be aligned")
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
            "memory directory is not a registered HotpotQA snapshot"
        )
    design = manifest.get("design")
    if not isinstance(design, dict) or stable_hash(design) != manifest.get(
        "design_hash"
    ):
        raise SnapshotProvenanceError("snapshot manifest design hash is invalid")
    for key, value in design.items():
        if manifest.get(key) != value:
            raise SnapshotProvenanceError(
                f"snapshot manifest duplicates disagree for {key}"
            )
    for key, expected in {
        "source_md5": source_md5,
        "model": model,
        "graph_type": graph_type,
        "node_num": int(node_num),
        "embedding_model": embedding_model,
    }.items():
        if manifest.get(key) != expected:
            raise SnapshotProvenanceError(
                f"snapshot {key}={manifest.get(key)!r}, runner requested {expected!r}"
            )
    support_ids = [int(value) for value in manifest.get("support_ids", [])]
    evaluation_ids = [int(value) for value in manifest.get("evaluation_ids", [])]
    if not support_ids or not evaluation_ids:
        raise SnapshotProvenanceError("snapshot has no registered split")
    if len(set(support_ids)) != len(support_ids) or len(set(evaluation_ids)) != len(
        evaluation_ids
    ):
        raise SnapshotProvenanceError("snapshot split contains duplicates")
    if set(support_ids) & set(evaluation_ids):
        raise SnapshotProvenanceError("snapshot contains support/evaluation leakage")
    if int(manifest.get("memory_records", -1)) != len(support_ids):
        raise SnapshotProvenanceError("memory count does not match support split")
    if candidate_kind == "trajectory" and int(manifest.get("successful_total", 0)) < 1:
        raise SnapshotProvenanceError(
            "snapshot has no exact-match successful trajectory"
        )
    if int(claims) > len(evaluation_ids):
        raise SnapshotProvenanceError(
            f"--claims {claims} exceeds {len(evaluation_ids)} registered examples"
        )
    return dict(manifest)
