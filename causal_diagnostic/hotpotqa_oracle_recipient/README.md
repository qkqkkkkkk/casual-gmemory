# HotpotQA native-GMemory RQ2/RQ3/RQ4 audit

This package is task-isolated from `fever_oracle_recipient`. It has its own
loader, prompts, offline environment, metrics, snapshot schema, runner,
orchestrator, tests, and report wording. It shares only task-agnostic frozen
GMemory, MacNet recipient masking, seeded caching, and bootstrap machinery.

## Experimental unit and outcomes

Use the standard HotpotQA distractor/full-context development JSON containing
`_id`, `question`, `answer`, `context`, and `supporting_facts`. Every causal
branch receives exactly the same supplied context; only exposure to one
retrieved GMemory candidate changes.

- Primary team outcome: official normalized answer token-F1.
- Sensitivity team outcome: normalized answer exact match.
- Objective local outcome: supporting-fact F1 over `(title, sentence index)`.
- RQ2 global effect: `use_all - global_drop`.
- RQ3 recipient effect: `only_recipient - global_drop`.
- RQ4 gap: recipient supporting-fact-F1 utility versus team answer-F1 utility.

Because answer F1 is already continuous, the binary A/B logprob probe used by
FEVER is neither needed nor valid for free-form HotpotQA answers.

## Data

Place the standard file at:

```text
data/hotpotqa/raw/hotpot_dev_distractor_v1.json
```

The loader also accepts JSONL. It fails closed when IDs, supplied context, or
supporting-fact annotations are missing or inconsistent.

## One-command experiment

From the repository root, with the model served on port 11436:

```bash
python -m causal_diagnostic.run_causal_audit hotpotqa \
  --data data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
  --endpoint http://127.0.0.1:11436/v1 \
  --model qwen2.5:7b \
  --memory-dir causal_diagnostic/memory_snapshots/hotpotqa_support50_7b_v1/g-memory \
  --results-root causal_diagnostic/results/native_hotpotqa_rq234_pilot_7b_v1 \
  --support-count 50 \
  --evaluation-count 100 \
  --claims 40 \
  --smoke-claims 4 \
  --repeats 6 \
  --sample-seed-base 0 \
  --retest-seed-base 1000 \
  --temperature 0.7 \
  --node-num 3 \
  --isolated-recipients solver_0,solver_2 \
  --graph-type Chain \
  --candidate-kind trajectory \
  --candidate-index 0 \
  --successful-topk 3 \
  --failed-topk 0 \
  --insights-topk 3 \
  --threshold 0.0 \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

The dispatcher keeps the benchmark packages isolated. The corresponding FEVER
entry point is `python -m causal_diagnostic.run_causal_audit fever ...`; its
existing arguments and experimental design are forwarded unchanged.

The orchestrator performs four resumable stages:

1. build one immutable 50-question GMemory snapshot;
2. run a four-question smoke test;
3. run seed 0 on 40 registered held-out questions;
4. run the independent seed-1000 retest and combined analysis.

Rerun the identical command after interruption. Completed stages and branches
are skipped; an incompatible manifest fails rather than silently mixing data.
Full subprocess output is retained under `RESULTS_ROOT/logs/`, and
`experiment_progress.json` identifies the failing stage and error tail.

## Main result files

The final retest directory contains:

- `run_report.md`: paper-readable RQ2/RQ3/RQ4 tables and conclusions;
- `rq2_team_score_policy_table.csv`: answer-F1 policy comparison with 95% CIs;
- `rq2_success_policy_table.csv`: answer-EM sensitivity table;
- `rq2_decision_evidence_f1_policy_table.csv`: final supporting-fact table;
- `recipient_matrix.csv`: per-question, per-recipient answer-F1 utilities;
- `rq4_local_team_matrix.csv`: supporting-fact versus answer utility patterns;
- `oracle_recipient_analysis.json`: complete machine-readable analysis;
- `branches.jsonl`: auditable prompts/outputs, scores, exposure traces and seeds.

Do not mix these outputs with FEVER directories. Snapshot and runner schemas
are benchmark-specific and cross-benchmark resume/retest attempts are rejected.
