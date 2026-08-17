# Frozen-Memory Causal Diagnostic

This directory runs one PDDL/MacNet episode twice on the same G-Memory
snapshot. The `with_memory` branch receives the frozen retrieval unchanged.
The `without_candidate` branch receives the same retrieval except for one
trajectory or insight. Both branches use the same seed and a fixed MacNet
graph, and neither branch writes to the supplied memory store.

Run from the repository root after loading `.env`:

```bash
export $(grep -v '^#' .env | xargs)
python causal_diagnostic/run_intervention.py \
  --task-id 0 \
  --memory-dir .db/qwen2.5-latest/pddl/macnet/g-memory/g-memory \
  --model qwen2.5:latest \
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
