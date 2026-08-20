#!/usr/bin/env python3
"""Visualize local-versus-team causal-utility intervention results.

Reads the JSON records emitted by ``run_local_team_intervention.py``. It does
not run a model or modify any result file.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


COLORS = {
    "local_positive_team_negative": "#c93d3d",
    "local_negative_team_positive": "#287c96",
    "both_neutral": "#8b8b8b",
    "aligned_or_unclassified": "#4e9a5a",
}

DISPLAY_NAMES = {
    "local_positive_team_negative": "Local+ / Team-",
    "local_negative_team_positive": "Local- / Team+",
    "both_neutral": "Both neutral",
    "aligned_or_unclassified": "Aligned / other",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-all", action="store_true", help="Label every point instead of mismatch cases only.")
    return parser.parse_args()


def load_records(results_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(results_root.rglob("task_*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping {path}: {exc}")
            continue
        if item.get("experiment") != "gmemory_local_team_frozen_intervention":
            continue
        utility = item["utility"]
        records.append(
            {
                "task_id": item["task_id"],
                "seed": item["seed"],
                "candidate_index": item["candidate"]["index"],
                "local_utility": utility["local_utility"],
                "team_utility": utility["team_utility_score"],
                "classification": utility["classification"],
            }
        )
    return records


def point_label(record: dict[str, Any]) -> str:
    return f"T{record['task_id']}, C{record['candidate_index']}, S{record['seed']}"


def main() -> None:
    args = parse_args()
    records = load_records(args.results_root)
    if not records:
        raise SystemExit(f"No local-team intervention JSON files found below: {args.results_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12})
    figure, (scatter_ax, count_ax) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.35, 1]})

    for record in records:
        classification = record["classification"]
        color = COLORS.get(classification, "#555555")
        scatter_ax.scatter(record["local_utility"], record["team_utility"], s=78, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        mismatch = classification in {"local_positive_team_negative", "local_negative_team_positive"}
        if args.label_all or mismatch:
            scatter_ax.annotate(point_label(record), (record["local_utility"], record["team_utility"]), xytext=(5, 5), textcoords="offset points", fontsize=8)

    scatter_ax.axhline(0, color="#555555", linewidth=0.9, zorder=1)
    scatter_ax.axvline(0, color="#555555", linewidth=0.9, zorder=1)
    scatter_ax.grid(alpha=0.2, linewidth=0.6)
    scatter_ax.set_xlabel("Local causal utility: action-validity rate delta")
    scatter_ax.set_ylabel("Team causal utility: reward-cost score delta")
    scatter_ax.set_title("Local and Team Causal Utility")
    scatter_ax.text(0.98, 0.03, "Local+ / Team-", transform=scatter_ax.transAxes, ha="right", va="bottom", color=COLORS["local_positive_team_negative"], fontsize=9)
    scatter_ax.text(0.02, 0.97, "Local- / Team+", transform=scatter_ax.transAxes, ha="left", va="top", color=COLORS["local_negative_team_positive"], fontsize=9)
    legend_keys = [key for key in DISPLAY_NAMES if any(row["classification"] == key for row in records)]
    scatter_ax.legend([Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLORS[key], markeredgecolor="white", markersize=8) for key in legend_keys], [DISPLAY_NAMES[key] for key in legend_keys], loc="upper right", frameon=True)

    counts = Counter(record["classification"] for record in records)
    categories = [key for key in DISPLAY_NAMES if counts[key]]
    labels = [DISPLAY_NAMES[key] for key in categories]
    values = [counts[key] for key in categories]
    bars = count_ax.barh(labels, values, color=[COLORS[key] for key in categories], height=0.6)
    count_ax.set_xlim(0, max(values) + 0.8)
    count_ax.set_xlabel("Number of interventions")
    count_ax.set_title(f"Outcome Categories (n={len(records)})")
    count_ax.grid(axis="x", alpha=0.2, linewidth=0.6)
    for bar, value in zip(bars, values):
        count_ax.text(value + 0.06, bar.get_y() + bar.get_height() / 2, str(value), va="center")

    figure.tight_layout()
    png_path = args.output_dir / "local_team_utility.png"
    svg_path = args.output_dir / "local_team_utility.svg"
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
