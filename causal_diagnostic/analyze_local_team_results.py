#!/usr/bin/env python3
"""Aggregate local-vs-team intervention JSON files."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def get(item: dict[str, Any], *path: str) -> Any:
    for key in path:
        item = item[key]
    return item


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for path in sorted(args.results_root.rglob("task_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping {path}: {exc}")
            continue
        if data.get("experiment") != "gmemory_local_team_frozen_intervention":
            continue
        records.append({
            "file": str(path), "task_id": data["task_id"], "seed": data["seed"],
            "candidate_id": get(data, "candidate", "candidate_id"),
            "candidate_index": get(data, "candidate", "index"),
            "local_with": get(data, "with_memory", "local", "score"),
            "local_without": get(data, "without_candidate", "local", "score"),
            "local_utility": get(data, "utility", "local_utility"),
            "team_with": get(data, "with_memory", "team", "score"),
            "team_without": get(data, "without_candidate", "team", "score"),
            "team_utility": get(data, "utility", "team_utility_score"),
            "success_delta": get(data, "utility", "success_delta"),
            "reward_delta": get(data, "utility", "reward_delta"),
            "step_delta": get(data, "utility", "step_delta"),
            "classification": get(data, "utility", "classification"),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "local_team_interventions.csv"
    fields = list(records[0]) if records else ["file"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    counts = Counter(record["classification"] for record in records)
    mismatch_count = counts["local_positive_team_negative"] + counts["local_negative_team_positive"]
    summary = {
        "n_interventions": len(records), "classification_counts": dict(counts),
        "mismatch_count": mismatch_count,
        "mismatch_rate": mismatch_count / len(records) if records else 0.0,
        "csv": str(csv_path),
    }
    (args.output_dir / "local_team_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
