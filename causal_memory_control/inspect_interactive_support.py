"""Read committed support actions without opening databases or loading models.

Invoke with python -m causal_memory_control.inspect_interactive_support. This
inspection-only file is outside the registered interactive_*.py source set.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def episodes(root):
    root = Path(root).resolve()
    markers = sorted((root / "support/committed").glob("*.json"))
    if not markers:
        raise ValueError(f"no committed support markers: {root / 'support/committed'}")
    rows = []
    for marker in markers:
        commit = json.loads(marker.read_text())
        directory = Path(commit["directory"])
        if directory.is_absolute() or ".." in directory.parts:
            raise ValueError(f"invalid committed directory: {marker}")
        path = (root / directory).parent / "episode.json"
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"episode path escapes the dataset root: {marker}")
        contents = path.read_bytes()
        if hashlib.sha256(contents).hexdigest() != commit["episode_sha256"]:
            raise ValueError(f"episode hash differs from its commit: {path}")
        row = json.loads(contents)
        if row["task_id"] != commit["task_id"]:
            raise ValueError(f"episode task differs from its commit: {path}")
        rows.append((path, row))
    return rows


def inspect(root, *, task_ids=(), steps=8, summary_only=False, transcript_tail=0):
    if steps < 0 or transcript_tail < 0:
        raise ValueError("steps and transcript-tail must be nonnegative; steps=0 shows all steps")
    rows = episodes(root)
    requested = set(task_ids)
    missing = requested - {row["task_id"] for _, row in rows}
    if missing:
        raise ValueError(f"no committed support episode for: {sorted(missing)}")
    outcomes = [row["outcome"] for _, row in rows]
    total_steps = sum(out["steps"] for out in outcomes)
    invalid = sum(out["invalid_actions"] for out in outcomes)
    print(f"committed_support={len(rows)} successful_support={sum(out['success'] > 0 for out in outcomes)}")
    print(f"steps={total_steps} invalid_actions={invalid} invalid_rate={invalid/total_steps:.1%}"
          if total_steps else "steps=0 invalid_actions=0 invalid_rate=undefined")
    for _, row in rows:
        print(row["task_id"], json.dumps(row["outcome"], ensure_ascii=False))
    if summary_only:
        return
    for path, row in rows:
        if requested and row["task_id"] not in requested:
            continue
        print(f"\n=== {row['task_id']} ===\nepisode={path}")
        print("initial_state=", row.get("state", "(not recorded)"))
        trace = row.get("trajectory", [])
        repeated = Counter(str(step.get("action", "")) for step in trace if step.get("valid_action") is False)
        print("most_common_invalid_actions=", json.dumps(repeated.most_common(5), ensure_ascii=False))
        print("recorded_environment_steps:")
        for step in (trace[:steps] if steps else trace):
            print(json.dumps(step, ensure_ascii=False))
        if steps and len(trace) > steps:
            print(f"({len(trace)-steps} more steps; use --steps 0 to show all)")
        if transcript_tail:
            print("observer_transcript_tail (recorded observer text; not a raw API response dump):")
            for message in row.get("agent_transcript", [])[-transcript_tail:]:
                print(message if isinstance(message, str) else json.dumps(message, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True, help="Dataset directory containing support/committed")
    parser.add_argument("--task-id", action="append", default=[], help="Exact support task ID; may repeat")
    parser.add_argument("--steps", type=int, default=8, help="Steps per selected task; 0 means all")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--transcript-tail", type=int, default=0)
    args = parser.parse_args(argv)
    inspect(args.root, task_ids=args.task_id, steps=args.steps, summary_only=args.summary_only,
            transcript_tail=args.transcript_tail)


if __name__ == "__main__":
    main()
