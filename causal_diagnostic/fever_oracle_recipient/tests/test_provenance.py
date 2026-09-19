from __future__ import annotations

import unittest

from causal_diagnostic.fever_oracle_recipient.provenance import (
    SNAPSHOT_SCHEMA,
    SnapshotProvenanceError,
    reconcile_retest_design,
    stable_hash,
    validate_snapshot_manifest,
)


def _manifest() -> dict:
    design = {
        "snapshot_schema": SNAPSHOT_SCHEMA,
        "source_md5": "abc",
        "model": "qwen2.5:7b",
        "graph_type": "Chain",
        "node_num": 3,
        "embedding_model": "embedder",
        "support_ids": [1, 2],
        "evaluation_ids": [3, 4],
    }
    return {
        **design,
        "design": design,
        "design_hash": stable_hash(design),
        "memory_records": 2,
        "successful_total": 1,
    }


def _validate(manifest: dict) -> dict:
    return validate_snapshot_manifest(
        manifest,
        source_md5="abc",
        model="qwen2.5:7b",
        graph_type="Chain",
        node_num=3,
        embedding_model="embedder",
        claims=2,
        candidate_kind="trajectory",
    )


class SnapshotProvenanceTests(unittest.TestCase):
    def test_valid_registered_snapshot_passes(self):
        self.assertEqual(_validate(_manifest())["successful_total"], 1)

    def test_tampered_registered_split_is_rejected(self):
        manifest = _manifest()
        manifest["evaluation_ids"] = [2, 4]
        with self.assertRaisesRegex(SnapshotProvenanceError, "disagree"):
            _validate(manifest)

    def test_zero_successful_trajectory_is_rejected(self):
        manifest = _manifest()
        manifest["successful_total"] = 0
        with self.assertRaisesRegex(SnapshotProvenanceError, "no successful"):
            _validate(manifest)

    def test_retest_allows_only_physical_chroma_hash_drift(self):
        previous_design = {
            "runner_schema": "runner-v3",
            "snapshot_design_hash": "snapshot-design",
            "memory_dir": "/frozen/memory",
            "memory_sha256": "seed0-physical-hash",
            "temperature": 0.7,
            "evaluation_ids": [1, 2],
        }
        previous_manifest = {
            "design": previous_design,
            "design_hash": stable_hash(previous_design),
        }
        current = {**previous_design, "memory_sha256": "retest-physical-hash"}

        aligned = reconcile_retest_design(current, previous_manifest)

        self.assertEqual(aligned, previous_design)

    def test_retest_rejects_semantic_design_drift(self):
        previous_design = {
            "memory_sha256": "old",
            "temperature": 0.7,
            "evaluation_ids": [1, 2],
        }
        previous_manifest = {
            "design": previous_design,
            "design_hash": stable_hash(previous_design),
        }
        current = {**previous_design, "memory_sha256": "new", "temperature": 0.0}

        with self.assertRaisesRegex(
            SnapshotProvenanceError, "temperature"
        ):
            reconcile_retest_design(current, previous_manifest)


if __name__ == "__main__":
    unittest.main()
