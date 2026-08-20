# Frozen-Memory Causal Diagnostic

This directory runs one PDDL/MacNet episode twice on the same G-Memory
snapshot. The `with_memory` branch receives the frozen retrieval unchanged.
The `without_candidate` branch receives the same retrieval except for one
trajectory or insight. Both branches use the same seed and a fixed MacNet
graph, and neither branch writes to the supplied memory store.

First create a new snapshot from support tasks that do not overlap with the
test task. For example, task IDs `0-9` are support tasks and ID `10` is a
test task:

```bash
python causal_diagnostic/build_snapshot.py \
  --task-ids 0-9 \
  --output-memory-dir causal_diagnostic/memory_snapshots/pddl_support_0_9/g-memory \
  --model qwen2.5:14b \
  --graph-type Chain
```

`build_snapshot.py` refuses to overwrite an existing directory. Do not use
the test task's ID in `--task-ids`.

Run from the repository root after loading `.env`:

```bash
export $(grep -v '^#' .env | xargs)
python causal_diagnostic/run_intervention.py \
  --task-id 10 \
  --memory-dir causal_diagnostic/memory_snapshots/pddl_support_0_9/g-memory \
  --model qwen2.5:14b \
  --candidate-kind trajectory \
  --candidate-index 0 \
  --graph-type Chain \
  --output-dir causal_diagnostic/results/smoke
```

The output JSON contains `success`, `reward`, `steps`, action traces, the
retrieved candidate identifier, and deltas. Here `step_delta` is
`steps_without_candidate - steps_with_memory`, so a positive value means the
candidate saved steps while preserving the same task setup.

Use the actual persisted G-Memory directory for `--memory-dir`: it is the
inner `g-memory` folder created by the original runner, such as
`.db/<model>/pddl/macnet/g-memory/g-memory`. The script will fail early if the
directory does not exist. Do not point it at a new, empty directory when
testing whether a stored memory is useful.

## Local-vs-team causal utility

`run_local_team_intervention.py` keeps the same frozen-memory intervention,
but records every MacNet worker output. Since workers propose actions while the
decision node executes the final action, the local score is a transparent
proxy: the worker PDDL-action validity rate. The team score is
`reward - cost_weight * steps / max_trials`; the raw success, reward, steps,
and complete worker/decision traces are retained in each JSON result.

```bash
python causal_diagnostic/run_local_team_intervention.py \
  --task-id 35 \
  --memory-dir causal_diagnostic/memory_snapshots/pddl_gripper_support_14b_clean/g-memory \
  --model qwen2.5:14b \
  --candidate-kind trajectory \
  --candidate-index 0 \
  --graph-type Chain \
  --node-num 2 \
  --max-trials 50 \
  --cost-weight 0.25 \
  --output-dir causal_diagnostic/results/local_team/task35_candidate0
```

`utility.classification` flags `local_positive_team_negative` and
`local_negative_team_positive` mismatches. Aggregate completed runs with:

```bash
python causal_diagnostic/analyze_local_team_results.py \
  --results-root causal_diagnostic/results/local_team \
  --output-dir causal_diagnostic/results/local_team_summary
```

Generate a report-ready PNG and SVG, with `U_local` vs. `U_team` quadrants
and outcome-category counts:

```bash
python causal_diagnostic/visualize_local_team_results.py \
  --results-root causal_diagnostic/results/local_team \
  --output-dir causal_diagnostic/results/local_team_summary
```
