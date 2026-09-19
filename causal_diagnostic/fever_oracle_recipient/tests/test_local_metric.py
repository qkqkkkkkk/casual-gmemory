from __future__ import annotations

import unittest

from causal_diagnostic.fever_oracle_recipient.local_metric import (
    gold_evidence_page_sets,
    parse_evidence_pages,
    score_evidence_pages,
)


class LocalMetricTests(unittest.TestCase):
    def test_gold_alternatives_and_page_normalization(self):
        example = {
            "evidence": [
                [[1, 2, "Soul_Food_-LRB-film-RRB-", 0]],
                [[3, 4, "Alternative_Page", 1]],
            ]
        }
        gold = gold_evidence_page_sets(example)
        self.assertEqual(len(gold), 2)
        score = score_evidence_pages(
            "Evidence[Soul Food (film)]\nFinish[SUPPORTS]", gold
        )
        self.assertEqual(score["precision"], 1.0)
        self.assertEqual(score["recall"], 1.0)
        self.assertEqual(score["f1"], 1.0)

    def test_pipe_separated_evidence_is_parsed(self):
        self.assertEqual(
            parse_evidence_pages("Evidence[Page A | Page B]\nFinish[REFUTES]"),
            ["Page A", "Page B"],
        )


if __name__ == "__main__":
    unittest.main()
