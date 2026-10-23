# Oracle on the latest held-out reevaluation

`run_oracle_evaluation` extends a **completed** `run_trained_gate_evaluation`
result without rebuilding memory, fitting models, collecting training labels,
or reevaluating any learned gate. All registered methods of the selected
datasets must have complete audited evaluations first. The latest manifest
supplies the actual evaluation seed, endpoint, model and original test offset;
the source loaders recheck the data and frozen snapshot. A sampled diagnostic
subset (e.g. HotpotQA 1,000 questions) never moves the final-test boundary.

The action space is predeclared: KEEP ALL, each single-memory DROP, and DROP
ALL. Candidates retain their original IDs and retrieval fingerprints. The
entry merges the latest immutable response caches and rejects conflicting
responses for the same seeded prompt. Each single-DROP branch then reuses the
previous branch's response cache, so identical actions/prompts share answers.
This is a paired experiment, not independent stochastic resampling of KEEP.

Oracle chooses the observed best reward per test question, breaking ties by
EM/done, then preference for baseline. It is **post-hoc and not deployable**,
and is not an upper bound on expected reward or all possible memory subsets.
With six candidates, it searches eight actions, not all 64 subsets. An
arbitrary multi-DROP CSM action is not added to this action space. Existing
learned policy logs are checked against it only when their actions for all
questions lie in the allowed set; their scores are still reported in the
table. Same-action disagreements stop the experiment. Do not tune a deployable
gate using these held-out Oracle selections.

## Run

Upload `run_oracle_evaluation.py` alongside the existing runtime. Optional
verification/docs: `tests/test_oracle_evaluation.py`,
`tests/test_trained_gate_evaluation.py`, and this document.

```bash
cd /mnt/mydata/hjq/casual-gmemory
conda activate gmemory

oracle_run=(
  python -u -m causal_memory_control.run_oracle_evaluation
  --evaluation-root causal_memory_control/results/all_trained_gates_with_old_st_retest_v2
  --output-dir causal_memory_control/results/latest_heldout_oracle_v1
  --datasets fever,hotpotqa
)

"${oracle_run[@]}" --plan-only
"${oracle_run[@]}"
```

Use the actual latest completed reevaluation root. `--plan-only` is read-only,
loads no neural weights, and makes no model calls. Missing or incomplete source
results stop before new calls. The final run reuses baseline and existing DROP
ALL; if DROP ALL is absent, it collects that control once. Neither HF CE Qwen
nor the BERT utility estimator is needed. Python may use a GPU for the frozen
retrieval encoder; Ollama generation stays on the recorded endpoint's service
and is not moved by `CUDA_VISIBLE_DEVICES`.

With the known six-candidate partition, the maximum **new team runs** are
FEVER 1,324×6=7,944 and HotpotQA 1,471×6=8,826 if DROP ALL already exists. One
team run can contain several LLM calls; cache replay can reduce new calls.
This is not the old diagnostic claim×candidate×repeat×USE/DROP collection.

Interrupt and rerun the exact same command to resume. Configuration/source
changes require a new output directory. Original results/caches remain
read-only. Output includes:

```text
latest_heldout_oracle_v1/
  oracle_run_manifest.json
  oracle_progress.json
  oracle_results.md / .csv / .json
  logs/{dataset}_oracle_branches_drop_000.log ...
  {dataset}/
    native_gmemory/memory_gate.jsonl   # copy, source hash in reused_from.json
    cache_seeds/latest_union.sqlite
    plans/oracle_branches_drop_000.json ...
    oracle_branches/drop_000/ ...
      memory_gate.jsonl / summary.json / comparison.json
      run_manifest.json / progress.json / llm_cache.sqlite
    oracle_selective_drop/
      memory_gate.jsonl               # selected action and all observed choices
      oracle_manifest.json / action_space.json
      summary.json / comparison.json
```

The table retains old method scores and adds Oracle. FEVER is binary Accuracy;
HotpotQA columns are Answer F1 and EM. Confidence intervals describe the
observed per-task post-hoc maximum, not out-of-sample deployable performance.
