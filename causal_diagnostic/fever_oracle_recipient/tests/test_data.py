from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from causal_diagnostic.fever_oracle_recipient.data import (
    FeverDataError,
    load_binary_fever,
    stratified_split,
)


class FeverDataTests(unittest.TestCase):
    def _source(self, root: Path) -> Path:
        rows = []
        for index in range(8):
            rows.append(
                {"id": index, "label": "SUPPORTS", "claim": f"support {index}"}
            )
            rows.append(
                {"id": 100 + index, "label": "REFUTES", "claim": f"refute {index}"}
            )
        rows.append({"id": 999, "label": "NOT ENOUGH INFO", "claim": "skip"})
        path = root / "fever.jsonl"
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        return path

    def test_split_is_balanced_deterministic_interleaved_and_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            examples = load_binary_fever(self._source(Path(directory)))
            first = stratified_split(
                examples, support_per_label=2, evaluation_per_label=3, seed=42
            )
            second = stratified_split(
                examples, support_per_label=2, evaluation_per_label=3, seed=42
            )
            self.assertEqual(first, second)
            support, evaluation = first
            self.assertEqual([row["label"] for row in support[:2]], ["SUPPORTS", "REFUTES"])
            self.assertEqual([row["label"] for row in evaluation[:2]], ["SUPPORTS", "REFUTES"])
            self.assertFalse(
                {row["id"] for row in support}
                & {row["id"] for row in evaluation}
            )

    def test_duplicate_binary_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.jsonl"
            row = {"id": 1, "label": "SUPPORTS", "claim": "claim"}
            path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FeverDataError, "duplicate"):
                load_binary_fever(path)


if __name__ == "__main__":
    unittest.main()

