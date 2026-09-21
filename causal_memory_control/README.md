# Team-Level Causal Memory Gate

This package provides a deployable plug-in experiment for GMemory. It leaves
memory construction, graph storage, native retrieval, MAS topology, and the
write policy unchanged. The only intervention point is after
`GMemory.retrieve_memory()` has selected candidates and before those candidates
are rendered into downstream agent prompts.

The deployment unit is a host-defined team exposure event:

```text
x = (task, current state, retrieved memory, exposure set, other candidates)
```

There is deliberately no receiver role in the learned representation. For each
event, the estimator learns two expected final team outcomes from repeated,
matched counterfactual runs:

```text
Q_use(x)  = E[G_team | do(KEEP), x]
Q_drop(x) = E[G_team | do(DROP), x]
U(x)      = Q_use(x) - Q_drop(x)
```

The runtime controller is one-sided and conservative:

```text
DROP  when U_hat + kappa * sigma < -delta
KEEP  otherwise
```

Thus cold starts and uncertain predictions preserve native GMemory behavior.
By default at most one candidate is dropped from a retrieved set because the
training intervention estimates a one-candidate drop while holding the other
candidates fixed.

## Full-pipeline integration

`GMemory` exposes a small post-retrieval hook through `MASMemoryBase`.
`tasks/run.py` installs `GMemoryExposureGate` on that hook, so the same code is
used by AutoGen, DyLAN, and MacNet. A gate run requires an exact memory
directory and automatically makes it read-only. Each task writes a JSONL row
containing all candidate decisions, keep rate, final reward, and completion
status.

Runtime arms:

- `always_keep`: exact native exposure; this is the main baseline.
- `always_drop`: removes all in-scope retrieved trajectories and insights.
- `learned`: applies the uncertainty-aware rule above.

The older receiver-level audit types remain available for analysis and
backward compatibility, but they are not inputs to this deployment gate.

## 1. Train a checkpoint

The trainer directly accepts `branches.jsonl` from the existing native oracle
runners. It groups matched `use_all` and `global_drop` repeats across all input
files and averages the selected final team metric. Training and evaluation
tasks must be disjoint.

```bash
python -m causal_memory_control.train_gate \
  --input causal_diagnostic/results/TRAIN_SEED0/branches.jsonl \
          causal_diagnostic/results/TRAIN_SEED1000/branches.jsonl \
  --task-ids 0-49 \
  --metric success \
  --output causal_memory_control/checkpoints/pddl_gate.json
```

Use `--metric team_score` or `--metric reward` for a continuous return if that
field exists in the branch outcomes. The reported held-in RMSE is only a sanity
check, not evidence of generalization. The checkpoint also records which
memory types and retrieval ranks appeared in training; unseen types/ranks are
always kept at deployment. When `--task-ids` is omitted, the trainer records
all task IDs present in the input files so a FEVER evaluation can still enforce
claim-level train/test separation.

Direct JSON/JSONL examples are also accepted:

```json
{
  "event_id": "event-1",
  "task_id": 1,
  "features": {"sim_task_memory": 0.7, "exposure_count": 4},
  "use_rewards": [1, 0, 1, 1],
  "drop_rewards": [1, 1, 1, 1]
}
```

`q_use` and `q_drop` scalars may replace the reward arrays.

## 2. Run the Always Keep baseline

Use a frozen snapshot, held-out tasks, and a fresh output directory:

```bash
python tasks/run.py \
  --task pddl \
  --mas_type macnet \
  --mas_memory g-memory \
  --model qwen2.5:14b \
  --memory_dir /ABSOLUTE/PATH/TO/FROZEN/g-memory \
  --memory_gate always_keep \
  --task_ids 50-84 \
  --successful_topk 1 \
  --failed_topk 0 \
  --insights_topk 3 \
  --run_dir causal_memory_control/results/pddl_always_keep
```

The primary output is `memory_gate.jsonl` in `--run_dir`.
If a long run is interrupted after completed task rows were flushed, rerun the
same command with `--gate_resume`; those task IDs are skipped. Without that
flag, an existing gate log is rejected to prevent accidental duplicate rows.

## 3. Run the learned gate

Keep every host, retrieval, task, and seed setting identical to the baseline:

```bash
python tasks/run.py \
  --task pddl \
  --mas_type macnet \
  --mas_memory g-memory \
  --model qwen2.5:14b \
  --memory_dir /ABSOLUTE/PATH/TO/FROZEN/g-memory \
  --memory_gate learned \
  --gate_checkpoint causal_memory_control/checkpoints/pddl_gate.json \
  --gate_kappa 1.96 \
  --gate_delta 0 \
  --gate_max_drops 1 \
  --task_ids 50-84 \
  --successful_topk 1 \
  --failed_topk 0 \
  --insights_topk 3 \
  --run_dir causal_memory_control/results/pddl_learned
```

Append `--task_limit 1` for a smoke test. For the Always-Drop diagnostic,
replace `learned` with `always_drop` and omit the checkpoint.

## 4. Compare final completion

```bash
python -m causal_memory_control.compare_gate_runs \
  --baseline causal_memory_control/results/pddl_always_keep/memory_gate.jsonl \
  --gate causal_memory_control/results/pddl_learned/memory_gate.jsonl \
  --output causal_memory_control/results/pddl_comparison.json
```

The report contains paired mean reward and completion-rate deltas, paired
bootstrap 95% intervals, and the learned gate's keep rate.

## Offline FEVER end-to-end experiment

Use the dedicated FEVER runner instead of `tasks/run.py --task fever`. This
path is closed-book and network-free at the environment level, uses the
registered support/evaluation split in the frozen snapshot manifest, and
scores an exact `SUPPORTS`/`REFUTES` final label. It still uses native GMemory
retrieval and native MacNet execution.

### Recommended: run the complete comparison with one command

The comparison orchestrator builds the frozen snapshot, collects two
counterfactual training repeat sets, trains the gate, runs native GMemory and
the learned gate on the same held-out claims, and writes the paired report:

```bash
python -u -m causal_memory_control.run_fever_gate_comparison \
  --data data/fever/fever_dev.jsonl \
  --endpoint http://127.0.0.1:11436/v1 \
  --model qwen2.5:7b \
  --use-all-binary-data
```

The bundled FEVER dev file has 3,333 `SUPPORTS`, 3,333 `REFUTES`, and
3,333 `NOT ENOUGH INFO` rows. This experiment is binary, so all-data mode uses
all 6,666 binary rows exactly once: 50 support-memory claims, 5,292 gate
training claims, and 1,324 final-test claims with the default
`--training-fraction 0.8`. The three sets are registered, disjoint, and checked
for checkpoint/test leakage. `NOT ENOUGH INFO` remains outside the binary task.
Inspect the resolved split and cost without creating files or calling a model:

```bash
python -m causal_memory_control.run_fever_gate_comparison \
  --data data/fever/fever_dev.jsonl \
  --use-all-binary-data \
  --plan-only
```

The default comparison now uses `--candidate-scope all_retrieved`. For each
claim it collects matched USE/DROP outcomes separately for successful
trajectory ranks 1–3 and insight ranks 1–3. The checkpoint records coverage
by both candidate kind and rank, and the learned run predicts utility for every
returned successful trajectory and insight. Failed trajectories are not gate
candidates because MacNet does not expose that return value to its prompt.

All candidates receive a utility prediction before filtering. The deployment
default remains `--max-drops 1`: if several candidates are confidently
harmful, only the one with the lowest upper utility bound is removed. This
matches the training intervention, which drops one candidate while keeping
its peers fixed. Use `--max-drops -1` only as an explicit multi-drop ablation.

For scalable gate training, the comparison defaults to
`--diagnostic-scope gate_training --repeats 2`. It collects only the matched
`use_all/global_drop` branches consumed by `train_gate`; the recipient-specific
RQ3/RQ4 branches are omitted. Use `--diagnostic-scope full_causal --repeats 6`
only when those causal reports are also required. Gate-only branch rows also
omit full execution traces and evidence-page audit fields to limit disk use.

The v5 deployment defaults address the overly conservative v4 uncertainty:

- `--residual-noise-scale 0` uses bootstrap-ensemble uncertainty without
  treating the held-in outcome residual as an irreducible per-candidate floor;
- `--kappa 0.5` retains an uncertainty penalty but is less conservative than
  the old `1.96` setting;
- both values are recorded in manifests/checkpoints and may be overridden for
  ablations. A scale of `1` plus `kappa 1.96` reproduces the old behavior.

Here `native_gmemory` is the no-gate baseline. Internally it uses
`always_keep`, which is an identity intervention: every item returned by
native GMemory retrieval is exposed unchanged. The hook only records a
standardized log so that the two arms can be compared claim by claim.

The terminal displays an overall five-stage progress bar plus nested support
claim, diagnostic branch, and final evaluation claim progress bars. The main
artifacts are:

```text
causal_memory_control/results/fever_gate_all_binary_v5/
├── pipeline_manifest.json
├── pipeline_progress.json
├── logs/
│   ├── diagnostic.log
│   ├── train_gate.log
│   ├── native_gmemory.log
│   ├── learned_gate.log
│   └── compare.log
├── native_gmemory/
│   ├── progress.json
│   ├── memory_gate.jsonl
│   └── summary.json
├── learned_gate/
│   ├── progress.json
│   ├── memory_gate.jsonl
│   └── summary.json
└── comparison.json
```

After an interruption, rerun the exact same command. Completed stages are
skipped; an incomplete diagnostic branch collection or final claim run resumes
from its append-only records. The baseline LLM cache is also reused for
identical learned-arm prompts. Changing arguments under the same output path is
rejected to prevent accidental mixing of experiments.

On failure, `pipeline_progress.json` records the failed stage, exception type,
message, exit code, exact command, child progress, the last 100 log lines, and
the traceback. Full subprocess output remains in `logs/<stage>.log`. The
claim-level `progress.json` files record the current claim and their own
traceback.

For a server run, keep it alive in `tmux` and save the console stream as well:

```bash
tmux new -s fever-gate
cd /ABSOLUTE/PATH/TO/gmemory
conda activate GMemory
python -u -m causal_memory_control.run_fever_gate_comparison \
  --data data/fever/fever_dev.jsonl \
  --endpoint http://127.0.0.1:11436/v1 \
  --model qwen2.5:7b \
  --use-all-binary-data \
  2>&1 | tee fever_gate_all_binary_v5_console.log
```

Detach with `Ctrl-b d`; reconnect with `tmux attach -t fever-gate`. If the
server requires a key, export `OPENAI_API_KEY` before running. Do not place the
key directly in a shell history command; the orchestrator passes it to child
processes through the environment and never includes it in logged commands.
Create the server environment first with `conda create -n GMemory python=3.12`
and `pip install -r requirements.txt` if it does not already exist. Do not copy
the macOS `gmemory_env` virtual environment to a Linux server.

In all-binary mode, each diagnostic seed plans
`5292 claims × 6 candidates × 2 repeats × 2 conditions = 127008` branches;
the two training seeds total 254,016 branches. This remains a large server
experiment. Missing retrieval ranks are recorded as exclusions, completed
branches are append-only, and rerunning the identical command resumes rather
than restarting. The two diagnostic collections use seeds 0 and 1000; the
final comparison uses seed 2000.

### Manual stage-by-stage commands

First collect the counterfactual training branches described in
`causal_diagnostic/fever_oracle_recipient/README.md`. In all-binary mode, train
on the two independent repeat sets:

```bash
python -m causal_memory_control.train_gate \
  --input causal_diagnostic/results/native_fever_all_binary_gate_7b_v5/seed0/branches.jsonl \
          causal_diagnostic/results/native_fever_all_binary_gate_7b_v5/seed1000_retest/branches.jsonl \
  --metric success \
  --residual-noise-scale 0 \
  --output causal_memory_control/checkpoints/fever_gate_all_binary_v5.json
```

The all-binary snapshot registers 6,616 post-support claims. The diagnostic
uses positions 0–5,291 for training, so use the disjoint positions 5,292–6,615
for final evaluation. Run Always Keep first:

```bash
python -m causal_memory_control.fever_experiment \
  --data data/fever/fever_dev.jsonl \
  --memory-dir causal_diagnostic/memory_snapshots/fever_evidence_support50_all_binary_7b_v5/g-memory \
  --endpoint http://127.0.0.1:11436/v1 \
  --model qwen2.5:7b \
  --graph-type Chain \
  --node-num 3 \
  --temperature 0.7 \
  --seed 2000 \
  --evaluation-offset 5292 \
  --claims 1324 \
  --mode always_keep \
  --candidate-kinds trajectory,insight \
  --output-dir causal_memory_control/results/fever_all_binary_v5_always_keep
```

Then run the learned policy with identical host, retrieval, task, and seed
settings. Seeding its cache from the baseline makes every identical prompt use
the same saved response; prompts changed by a DROP still invoke the model.

```bash
python -m causal_memory_control.fever_experiment \
  --data data/fever/fever_dev.jsonl \
  --memory-dir causal_diagnostic/memory_snapshots/fever_evidence_support50_all_binary_7b_v5/g-memory \
  --endpoint http://127.0.0.1:11436/v1 \
  --model qwen2.5:7b \
  --graph-type Chain \
  --node-num 3 \
  --temperature 0.7 \
  --seed 2000 \
  --evaluation-offset 5292 \
  --claims 1324 \
  --mode learned \
  --candidate-kinds trajectory,insight \
  --checkpoint causal_memory_control/checkpoints/fever_gate_all_binary_v5.json \
  --kappa 0.5 \
  --delta 0 \
  --max-drops 1 \
  --cache-seed-from causal_memory_control/results/fever_all_binary_v5_always_keep/llm_cache.sqlite \
  --output-dir causal_memory_control/results/fever_all_binary_v5_learned
```

Each output directory contains `run_manifest.json`, resumable `progress.json`,
`memory_gate.jsonl`, `llm_cache.sqlite`, and `summary.json`. Add `--resume` to
the identical command after interruption. The learned runner refuses a
checkpoint if its recorded training claim IDs overlap the selected evaluation
claims, if it lacks the source-run provenance manifest, or if recorded
snapshot/model/retrieval settings disagree.

Compare exact-label accuracy and keep rate with the same paired reporter:

```bash
python -m causal_memory_control.compare_gate_runs \
  --baseline causal_memory_control/results/fever_all_binary_v5_always_keep/memory_gate.jsonl \
  --gate causal_memory_control/results/fever_all_binary_v5_learned/memory_gate.jsonl \
  --output causal_memory_control/results/fever_all_binary_v5_comparison.json
```

## Tests

```bash
python -m unittest discover -s causal_memory_control/tests -v
```

The tests do not open or mutate a real GMemory database.
