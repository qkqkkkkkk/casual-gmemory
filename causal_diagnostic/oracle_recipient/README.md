# Native GMemory + MacNet Oracle/Recipient Audit

This package runs RQ2 and RQ3 inside the official GMemory host rather than a
GMemory-style reimplementation. It reuses:

- the persisted hierarchical `GMemory` store and native retrieval;
- the original PDDL environment;
- the original three-worker MacNet graph and decision node;
- the original prompts, agent roles, state updates, and final feedback.

The official source files are not modified. `RecipientMaskedMacNet` copies the
native scheduling semantics and changes only which frozen candidate appears in
each worker's prompt.

## Causal conditions

For each held-out `(task, retrieved candidate)` event, the runner collects:

```text
USE_ALL
GLOBAL_DROP
DROP_solver_0
DROP_solver_1
DROP_solver_2
```

The decision node receives full memory under receiver-only DROP, so
`DROP_solver_i` changes only worker `i`'s direct memory exposure. Under
`GLOBAL_DROP`, the candidate is removed from all workers and the decision
node. Every step records and validates this exposure invariant.

The estimands are:

```text
U_global(m) = Y(USE_ALL) - Y(GLOBAL_DROP)
U_i(m)      = Y(USE_ALL) - Y(DROP_solver_i)
```

The primary team score is:

```text
reward - cost_weight * steps / max_trials
```

Raw PDDL success, reward, steps, actions, prompts hashes, worker outputs, and
decision outputs are retained. Raw success is reported as a sensitivity
outcome.

## Safety and reproducibility

- The memory snapshot is read-only in every branch.
- A snapshot manifest is required by default, and support/test task overlap is
  rejected.
- Retrieval is frozen once per task before branching.
- Graph, environment, retrieval, and sampling seed are matched across arms.
- The request seed is a stable function of experiment seed and prompt.
- Identical requests are persisted in `llm_cache.sqlite`.
- Every completed branch is appended immediately; `--resume` is supported.
- A second run with a disjoint inference-seed range provides the independent
  retest.

## Tests

From the GMemory repository root:

```bash
python -m unittest discover \
  -s causal_diagnostic/oracle_recipient/tests -v
```

## 1. Build a leakage-free support snapshot

Skip this step if an existing causal snapshot has a manifest and uses the same
model/configuration. The example reserves tasks `0-9` for memory construction
and tasks `10-59` for evaluation.

```bash
export OPENAI_API_BASE=http://127.0.0.1:11436/v1
export OPENAI_API_KEY=EMPTY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python causal_diagnostic/build_snapshot.py \
  --task-ids 0-9 \
  --output-memory-dir causal_diagnostic/memory_snapshots/pddl_support_0_9_qwen3b/g-memory \
  --model qwen2.5:3b \
  --graph-type Chain \
  --node-num 3 \
  --successful-topk 3 \
  --failed-topk 0 \
  --insights-topk 3
```

## 2. Smoke run

This uses the original GMemory/MacNet temperature of zero. One repeat is
sufficient for the deterministic primary design; uncertainty is clustered
over held-out tasks rather than pretending repeated deterministic calls are
independent observations.

```bash
python -m causal_diagnostic.oracle_recipient.run_experiment \
  --task-ids 10 \
  --memory-dir causal_diagnostic/memory_snapshots/pddl_support_0_9_qwen3b/g-memory \
  --endpoint http://127.0.0.1:11436/v1 \
  --model qwen2.5:3b \
  --candidate-kind trajectory \
  --candidate-index 0 \
  --graph-type Chain \
  --node-num 3 \
  --temperature 0 \
  --repeats 1 \
  --sample-seed-base 0 \
  --output-dir causal_diagnostic/results/native_oracle_recipient_smoke_seed0
```

Inspect:

```bash
cat causal_diagnostic/results/native_oracle_recipient_smoke_seed0/run_report.md
jq . causal_diagnostic/results/native_oracle_recipient_smoke_seed0/collection_diagnostics.json
```

The smoke run should contain five completed branches for one eligible event.

## 3. First full run

Start with tasks `10-29` (20 held-out tasks). Expand to `10-59` only after the
smoke and 20-task pilot are healthy.

```bash
python -m causal_diagnostic.oracle_recipient.run_experiment \
  --task-ids 10-29 \
  --memory-dir causal_diagnostic/memory_snapshots/pddl_support_0_9_qwen3b/g-memory \
  --endpoint http://127.0.0.1:11436/v1 \
  --model qwen2.5:3b \
  --candidate-kind trajectory \
  --candidate-index 0 \
  --graph-type Chain \
  --node-num 3 \
  --successful-topk 3 \
  --failed-topk 0 \
  --insights-topk 3 \
  --temperature 0 \
  --repeats 1 \
  --sample-seed-base 0 \
  --bootstrap-samples 10000 \
  --output-dir causal_diagnostic/results/native_oracle_recipient_20_seed0
```

If interrupted, rerun the exact command with `--resume`.

## 4. Independent retest

Keep all design arguments unchanged. Only change the inference-seed range,
output directory, and add `--retest-results`:

```bash
python -m causal_diagnostic.oracle_recipient.run_experiment \
  --task-ids 10-29 \
  --memory-dir causal_diagnostic/memory_snapshots/pddl_support_0_9_qwen3b/g-memory \
  --endpoint http://127.0.0.1:11436/v1 \
  --model qwen2.5:3b \
  --candidate-kind trajectory \
  --candidate-index 0 \
  --graph-type Chain \
  --node-num 3 \
  --successful-topk 3 \
  --failed-topk 0 \
  --insights-topk 3 \
  --temperature 0 \
  --repeats 1 \
  --sample-seed-base 1000 \
  --bootstrap-samples 10000 \
  --retest-results causal_diagnostic/results/native_oracle_recipient_20_seed0 \
  --output-dir causal_diagnostic/results/native_oracle_recipient_20_seed1000_retest
```

Final outputs:

```text
run_report.md
oracle_recipient_analysis.json
rq2_team_score_policy_table.csv
rq2_success_policy_table.csv
recipient_matrix.csv
branches.jsonl
collection_diagnostics.json
run_manifest.json
llm_cache.sqlite
```

## 5. Optional stochastic robustness run

Do not mix this with the temperature-zero primary design. Use a new output
directory and matching independent retest:

```text
--temperature 0.7 --repeats 3
```

This estimates expected utility under stochastic decoding. The primary native
GMemory result should remain temperature zero because that matches the
original MacNet configuration.

## Interpretation

RQ2 is supported when the oracle gain over Always Use is positive with a
positive event-bootstrap lower bound. A deployable rule is not established
unless the bidirectional cross-run gain is also positive and stable.

The RQ2 report uses a paper-style, budget-matched policy table with these rows:

    Always Use
    Always Global Drop
    Random (budget matched)
    Cross-run Selective Drop       # available after the independent retest
    Oracle Selective Drop          # same-sample upper bound

`Memory Keep Rate` is the fraction of evaluated task-candidate events in which
the frozen candidate is kept globally; it is not the fraction of the entire
GMemory database retained. Random is the exact expected outcome of a uniform
selector with the same per-run keep budget as Oracle, so it needs no extra
model calls. Cross-run Selective Drop applies each run's decisions to the other
run and is the transfer check; it must not be described as a trained predictor.
The reported 95% intervals resample held-out task-candidate events and preserve
the pairing between policies and independent runs.

RQ3's one-run direct sign-flip rate is descriptive. The strongest evidence is
the independent `reproducible_directional_flip_rate`: the same recipient is
positive and another recipient is negative for the same task-memory event in
both runs.
