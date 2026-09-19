from __future__ import annotations

import unittest

from causal_diagnostic.fever_oracle_recipient.provenance import (
    SNAPSHOT_SCHEMA,
    SnapshotProvenanceError,
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


if __name__ == "__main__":
    unittest.main()

