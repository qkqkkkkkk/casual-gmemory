# Offline FEVER native-GMemory causal diagnostic

This directory replaces the infeasible PDDL data collection path for RQ2/RQ3/RQ4.
It does not delete or reinterpret the PDDL result: the Qwen2.5-7B PDDL support
snapshot completed technically, but produced 0/20 successful trajectories, so
the successful-trajectory intervention was undefined.

The replacement experiment keeps the two components under study native to the
GMemory repository:

- persistent memory: `GMemory`;
- multi-agent architecture: three-worker `MacNet`, `Chain` graph;
- task: offline, closed-book binary FEVER (`SUPPORTS`/`REFUTES`);
- support and evaluation claims: deterministic balanced split, frozen in the
  snapshot manifest;
- branches per held-out claim: `use_all`, `global_drop`, `only_solver_0`, and
  `only_solver_2`; the Chain endpoints are the two structurally most different
  recipients;
- objective RQ4 local outcome: gold FEVER evidence-page F1 parsed from each
  worker's `Evidence[...]` output;
- no online Wikipedia and no new retrieval inside matched branches.

Do not reuse the earlier `fever_evidence_support50_7b_v2` snapshot or its LLM
cache. Native MacNet's newline stop truncated the required second-line
`Finish[...]` label in that version. The v3 snapshot disables that stop for
this two-line FEVER protocol and uses a separate directory and cache.

For the full-data gate experiment, prefer
`python -m causal_memory_control.run_fever_gate_comparison
--use-all-binary-data`. It automatically registers the full disjoint split and
passes `--all-candidates --gate-training-only` here. Gate-training-only mode
collects just `use_all/global_drop`, accepts fewer repeats, writes compact
branch rows, and produces `gate_training_collection.json`; it intentionally
does not produce the recipient/RQ3/RQ4 report described below.

## Recommended: run everything with one resumable command

From the repository root, this one command builds the frozen snapshot, runs a
four-claim smoke check, runs the 40-event/6-repeat seed-0 pilot, and then runs the independent
seed-1000 retest and combined analysis:

```bash
python -u -m causal_diagnostic.fever_oracle_recipient.run_all \
  --endpoint http://127.0.0.1:11436/v1 \
  --all-candidates \
  --results-root causal_diagnostic/results/native_fever_all_candidates_7b_v4
```

`--all-candidates` collects a separate matched intervention event for every
returned successful trajectory and insight rank. With the default retrieval
settings this covers trajectory ranks 1–3 and insight ranks 1–3. Use a new
results root as shown above; the older v3 result contains only the selected
Top-1 trajectory event and cannot train the expanded gate.

The command has three live progress levels:

- overall stage progress (`snapshot`, `smoke`, `seed0`, `seed1000_retest`);
- support-memory progress by claim;
- intervention progress by persisted branch. The all-candidate pilot has four
  conditions per event and six paired repeats: at most 5760 persisted branches
  per seed (`40 × 6 candidates × 6 repeats × 4 conditions`). A configured rank
  that is absent from a claim's actual retrieval is recorded as an exclusion.

If the process, SSH session, model server, or scheduler interrupts the run,
rerun the **identical command**. It automatically:

- skips stages whose final analysis already exists;
- invokes snapshot `--resume` for an incomplete memory build;
- invokes experiment `--resume` for incomplete seed runs;
- reads the append-only `branches.jsonl`, so completed branches are never run
  again;
- refuses to resume when the model, split, memory, or other design arguments
  differ from the original manifest.

Progress and errors are persisted under:

```text
causal_diagnostic/results/native_fever_all_candidates_7b_v4/
├── experiment_progress.json
├── logs/
│   ├── snapshot.log
│   ├── smoke.log
│   ├── seed0.log
│   └── seed1000_retest.log
├── smoke_seed0/collection_progress.json
├── seed0/collection_progress.json
└── seed1000_retest/collection_progress.json
```

On failure the terminal prints the failed stage, exit code, exact command,
full log path, and the last 80 log lines. The JSON progress file also records
the error. The per-run progress file records the exact claim, repeat, and arm
being evaluated when the error occurred.

For an unreliable SSH connection, start the same command inside `tmux`. The
resume mechanism still protects completed work if the process itself stops.

To omit the preliminary four-claim smoke run, add `--skip-smoke`. To put data,
memory, or results elsewhere, use `--data`, `--memory-dir`, and
`--results-root`; reruns must use the same values.

## 1. Build one frozen support memory

Run from the GMemory repository root. The default split uses 25 claims per
label for support and 50 per label for evaluation (50 + 100 claims total).

```bash
python -m causal_diagnostic.fever_oracle_recipient.build_snapshot \
  --data data/fever/fever_dev.jsonl \
  --support-per-label 25 \
  --evaluation-per-label 50 \
  --split-seed 42 \
  --endpoint http://127.0.0.1:11436/v1 \
  --model qwen2.5:7b \
  --graph-type Chain \
  --node-num 3 \
  --output-memory-dir causal_diagnostic/memory_snapshots/fever_evidence_support50_7b_v3/g-memory
```

If interrupted, rerun the identical command with `--resume`. Resume is fail
closed: the number of persisted GMemory records must exactly equal the number
of completed `support_runs.jsonl` rows. Do not rebuild merely to obtain a more
favorable success count.

Before evaluation, inspect:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("causal_diagnostic/memory_snapshots/fever_evidence_support50_7b_v3/g-memory/causal_snapshot_manifest.json")
m = json.loads(p.read_text())
print({k: m[k] for k in ("memory_records", "successful_total", "failed_total")})
print("support/eval overlap:", len(set(m["support_ids"]) & set(m["evaluation_ids"])))
PY
```

`successful_total` must be greater than zero and overlap must be zero.

## 2. Four-claim smoke test

The registered evaluation order alternates the two labels, so four claims test
both classes.

```bash
python -m causal_diagnostic.fever_oracle_recipient.run_experiment \
  --test data/fever/fever_dev.jsonl \
  --claims 4 \
  --memory-dir causal_diagnostic/memory_snapshots/fever_evidence_support50_7b_v3/g-memory \
  --endpoint http://127.0.0.1:11436/v1 \
  --model qwen2.5:7b \
  --graph-type Chain \
  --node-num 3 \
  --isolated-recipients solver_0,solver_2 \
  --temperature 0.7 \
  --sample-seed-base 0 \
  --all-candidates \
  --output-dir causal_diagnostic/results/native_fever_all_candidates_7b_v4/smoke_seed0
```

Check `collection_diagnostics.json`; `excluded_claims` should normally be
empty. The smoke output is only a pipeline check and must not be merged into
the main estimate.

## 3. Registered 40-event RQ2/RQ3/RQ4 pilot and independent retest

Seed 0:

```bash
python -m causal_diagnostic.fever_oracle_recipient.run_experiment \
  --test data/fever/fever_dev.jsonl \
  --claims 40 \
  --memory-dir causal_diagnostic/memory_snapshots/fever_evidence_support50_7b_v3/g-memory \
  --endpoint http://127.0.0.1:11436/v1 \
  --model qwen2.5:7b \
  --graph-type Chain \
  --node-num 3 \
  --isolated-recipients solver_0,solver_2 \
  --temperature 0.7 \
  --repeats 6 \
  --sample-seed-base 0 \
  --all-candidates \
  --output-dir causal_diagnostic/results/native_fever_all_candidates_7b_v4/seed0
```

Independent seed 1000, using the exact same frozen snapshot:

```bash
python -m causal_diagnostic.fever_oracle_recipient.run_experiment \
  --test data/fever/fever_dev.jsonl \
  --claims 40 \
  --memory-dir causal_diagnostic/memory_snapshots/fever_evidence_support50_7b_v3/g-memory \
  --endpoint http://127.0.0.1:11436/v1 \
  --model qwen2.5:7b \
  --graph-type Chain \
  --node-num 3 \
  --isolated-recipients solver_0,solver_2 \
  --temperature 0.7 \
  --repeats 6 \
  --sample-seed-base 1000 \
  --all-candidates \
  --retest-results causal_diagnostic/results/native_fever_all_candidates_7b_v4/seed0 \
  --output-dir causal_diagnostic/results/native_fever_all_candidates_7b_v4/seed1000_retest
```

Add `--resume` to the same command after an interruption. Never seed the
seed-1000 retest from seed 0's LLM cache; that would make it a replay rather
than an independent inference run.

The main artifacts are:

- `run_report.md`: RQ2 policy table and RQ3 recipient summary;
- `rq2_team_score_policy_table.csv`: accuracy and paired 95% bootstrap CIs;
- `recipient_matrix.csv`: one row per claim-memory event and per-recipient
  utility/sign;
- `rq4_local_team_matrix.csv`: evidence-page-F1 utility, team utility,
  split-half patterns, and reproducible local/team mismatches;
- `oracle_recipient_analysis.json`: complete machine-readable analysis;
- `branches.jsonl`: predictions plus complete worker/decision exposure traces.

The diagnostic defaults (`temperature=0.7`, six paired repeats, 40 events)
follow the RQ3/RQ4 pilot design. RQ2 uses the same `use_all/global_drop`
rollouts as its MacNet anchor; RQ3 and RQ4 share the isolated-exposure
rollouts, so RQ4 adds scoring but no extra model calls.

## Continuous outcomes

Existing v3 results can be rescored for final-decision evidence-page F1 with
no model calls:

```bash
python -m causal_diagnostic.fever_oracle_recipient.decision_evidence_analysis \
  --results causal_diagnostic/results/native_fever_rq234_pilot_7b_v3/seed0 \
  --retest-results causal_diagnostic/results/native_fever_rq234_pilot_7b_v3/seed1000_retest \
  --output-dir causal_diagnostic/results/native_fever_rq234_pilot_7b_v3/decision_evidence_review
```

This reports `U_decision-evidence = F1(only_i) - F1(global_drop)`. It is an
evidence-output sensitivity measure, not a label probability.

True label-probability sensitivity requires a new scoring call because legacy
caches contain only generated text. First probe the endpoint:

```bash
python -m causal_diagnostic.fever_oracle_recipient.probe_logprobs \
  --endpoint http://127.0.0.1:11436/v1 \
  --model qwen2.5:7b
```

Only if the probe prints `"status": "supported"`, add
`--label-probabilities --label-top-logprobs 5` to both seed-0 and seed-1000
commands and use new output directories. The runner records the binary-
normalized gold-label probability and gold-label log odds for every branch;
it fails closed if either A or B is absent from `top_logprobs`.
