# Independent CSM-ST transfer to GMemory

This is a new experiment. `st_estimator.py`, `st_data.py`, `train_gate.py`,
`run_st_gate_benchmark.py`, existing checkpoints and host evaluation runners are
not edited. Old methods are never retrained or reevaluated by the new entry.

## What is reproduced, and what is transferred

The implementation follows the components of Deng et al., *Influence Guided
Context Selection for Effective Retrieval-Augmented Generation*, §§4.1–4.2,
Appendices A/B/C.3: trainable BERT query–context encoder, masked mean pooling,
three 8-head list-attention layers, a two-layer MLP and one **unbounded** CI head;
gold-answer cross-entropy leave-one-out labels; trivial-list downsampling;
fit-only cross-instance interventions (eqs.5/10); inverse rarity-frequency MSE
and same-list global-representation InfoNCE (eqs.6/7); keep every predicted CI>0.
No CSM-E2E training or generator fine-tuning is required.

This is **not** the original FlashRAG benchmark reproduction. GMemory's frozen
candidate memories, supplied task context, support/fit/dev/test split and MAS
inference host are retained. The CE teacher is a **single frozen HF generator
proxy**, with a new recorded task+memory chat template, not the joint likelihood
of the agents' stochastic trajectory. HF Qwen and Ollama's quantized Qwen need
not have identical probabilities. Execution therefore requires
`--acknowledge-ce-proxy`. Never describe these labels as measured MAS reward
effects or these scores as reproducing the original paper's table.

For one answer, `CI_i = CE_without_i - CE_all`. With multiple valid answers,
use the minimum CE per context set (paper eq.1). Only continuation tokens,
including EOS when available, are scored; prompt tokens are not targets.
The default reduction is mean over target tokens (`--ce-reduction mean`). Sum
is available but changes label scale; use a fresh output directory if changing
it. The scorer performs no answer sampling and **refuses context overflow**.

The old reward branches supply task text, candidate text and gold answers,
not the new CI values. USE ALL is scored **once per list**, then each
leave-one-out set once, with no repeat multiplier. Recombined lists always get
fresh CE labels; old labels are never copied. Exact/semantic duplicates are
removed consistently before labeling and deployment. Augmentations identical
to an existing candidate set for the same task are skipped.

Paper-stated defaults: beta=0.1, tau=1, epochs=10, batch=16, seed=2024,
rarity alpha=10 and trivial/hard thresholds=5 (Appendix A). Other details are
explicit engineering choices, **not claimed official-code defaults**:
AdamW BERT LR=2e-5/head LR=1e-4, weight decay=.01, histogram bin width=1,
trivial keep fraction=.2, positive/negative eps=.1/.3, intervention gamma=.1,
donor query cosine<.3 (up to 256 seeded donor trials), one donor per hard list,
semantic dedup cosine>=.98, attention residual/FFN blocks, MLP hidden=256,
longest-first pair truncation to BERT's 512-token window. These settings and
the actual model configuration/commit are recorded. The official repository
was not consulted for unspecified details. Do not call this a bit-exact clone.

If there are no hard lists or valid positive/negative pairs, training explicitly
warns that InfoNCE is inactive. Read `labels_manifest.json` CI distributions and
`training_summary.json` before tuning fit/dev thresholds. Do not tune on test.

## Install and inspect

In the server repository and the existing `gmemory` environment:

```bash
cd /mnt/mydata/hjq/casual-gmemory
conda activate gmemory
python -m pip install -r causal_memory_control/requirements_csm_st.txt
```

Use a **Hugging Face model directory/model ID**, not an Ollama `models/blobs`
directory, for `--generator-model`. The Qwen generator stays frozen. BERT is
trained. CE scoring and training run in separate processes, releasing GPU
allocations between stages. `CUDA_VISIBLE_DEVICES` controls these Python
processes, not an already-running Ollama server. Pick an available GPU first.
For RTX 5090, keep a PyTorch/CUDA build that actually supports Blackwell;
the minimum package version alone does not establish GPU compatibility.

Read-only plan using the known original result directories:

```bash
python -m causal_memory_control.run_csm_st_benchmark \
  --source-root causal_memory_control/results/full_gate_benchmark_v1 \
  --fever-results-root causal_memory_control/results/fever_seed0_gate_matrix_v1 \
  --hotpotqa-results-root causal_memory_control/results/hotpotqa_seed0_train1000_r2_dual_v1 \
  --output-dir causal_memory_control/results/csm_st_gmemory_v1 \
  --datasets fever,hotpotqa \
  --plan-only
```

Each `*-results-root` must contain its registered `seed0_manifest.json` and
completed baseline. It imports the existing diagnostic subset/shards. For the
known HotpotQA run this means **1000 diagnostic tasks**, not 5883, while keeping
the original test offset **5883** and test count **1471**. This entry does not
silently expand/recollect the diagnostic subset. Omit a results-root only when
the baseline and completed seed0 diagnostics actually live in the original
full-run directory. A missing baseline is an error, never a request to rerun it.

## Run the new method only

Once the plan paths are confirmed, use the same arguments without `--plan-only`:

```bash
CUDA_VISIBLE_DEVICES=2 \
python -u -m causal_memory_control.run_csm_st_benchmark \
  --source-root causal_memory_control/results/full_gate_benchmark_v1 \
  --fever-results-root causal_memory_control/results/fever_seed0_gate_matrix_v1 \
  --hotpotqa-results-root causal_memory_control/results/hotpotqa_seed0_train1000_r2_dual_v1 \
  --output-dir causal_memory_control/results/csm_st_gmemory_v1 \
  --datasets fever,hotpotqa \
  --generator-model Qwen/Qwen2.5-7B-Instruct \
  --encoder-model bert-base-uncased \
  --ce-device cuda:0 --train-device cuda:0 --evaluation-device cuda:0 \
  --acknowledge-ce-proxy
```

Model IDs may download weights on first actual execution, never during
`--plan-only`. Prefer immutable HF commit revisions or local model directories
and `--local-files-only` on offline servers. Local CE model files are hashed;
resolved remote generator identities are checked on partial-label resume.
This example doesn't start/stop Ollama or reserve GPU 2; check availability.
The new estimator includes BERT, so GPU evaluation is useful here (unlike the
old tiny ST MLP). Use `--evaluation-device cpu` if no evaluation GPU is available.

To run only one dataset, use `--datasets fever` or `--datasets hotpotqa` and a
**separate output root** for that configuration. To use a replica endpoint,
add `--evaluation-endpoint http://127.0.0.1:11440/v1`; existing replica
verification must pass. This is unrelated to the HF CE scoring device.

Stages are `--stage labels`, `train`, `evaluate`, `report`, or `all` (default).
Repeat all configuration arguments when resuming or selecting a stage.
`labels` makes no new MAS answer generations. `evaluate` requires new labels
and a new CSM checkpoint; old ST checkpoints are rejected. If training is
interrupted, its stage restarts deterministically from the base encoder;
there is no optimizer-step resume. CE resumes per list and per cached forward;
final evaluation resumes per completed task. A changed source or configuration
requires a new result directory. Model training never uses final-test labels.

## Results and logs

The new output contains:

- `csm_st_manifest.json`, `experiment_plan.json`, `csm_st_progress.json`;
- `logs/{dataset}_{labels,train,evaluate}.log`;
- `{dataset}/lists.jsonl`, `split.json`, `ce_labels.jsonl`, `ce_cache.sqlite`,
  `generator_identity.json`, `ce_progress.json`, `labels_manifest.json`;
- `{dataset}/augmentation_plan.jsonl`, `augmented_labels.jsonl` and per-list
  part files (checkpoint copies; potentially large);
- `{dataset}/model/`, `checkpoint.json`, `loss_history.json`,
  `training_summary.json`, `dev_predictions.jsonl`;
- `{dataset}/evaluation/{memory_gate.jsonl,run_manifest.json,summary.json,
  comparison.json,llm_cache.sqlite,progress.json}`;
- `csm_st_results.md`, `.csv`, `.json`, with FEVER Accuracy and HotpotQA F1/EM.

The comparison JSON retains paired bootstrap CIs, improved/harmed tasks, drop
counts and strict original-retrieval/no-DROP consistency checks. Candidate
logs retain predicted CI and why each context was kept/dropped/deduplicated.
`q_use/q_drop` are explicitly marked placeholders in this single-head policy;
they are not learned potential outcomes, nor are CI scores confidence bounds.

Old baseline/Linear/Semantic Linear/MLP/Global Drop rows are included only when
their existing comparisons and log hashes are valid against this baseline;
otherwise the table shows `—`. To include old ST, add its independent root via
`--fever-old-st-root PATH` / `--hotpotqa-old-st-root PATH`; these flags only
affect reporting, never run old ST. A multi-drop/semantic-dedup CSM policy and
old max-one-drop gates are **method-level comparisons**, not architecture-only
ablations. Different beta/selection/truncation/CE-scale ablations need separate
directories and dev-based choices before final testing.

## Reevaluate all trained policies without retraining

`run_csm_st_benchmark --stage evaluate` reuses completed evaluations and only
handles CSM. To make a new evaluation of **all trained policies**, upload the
independent `run_trained_gate_evaluation.py` entry and use a separate directory:

```bash
CUDA_VISIBLE_DEVICES=2 \
python -u -m causal_memory_control.run_trained_gate_evaluation \
  --trained-root causal_memory_control/results/csm_st_gmemory_v2 \
  --output-dir causal_memory_control/results/all_trained_gates_retest_v1 \
  --datasets fever,hotpotqa \
  --csm-device cuda:0 \
  --plan-only
```

Use the actual completed CSM root if its name differs. Remove `--plan-only`
after checking the checkpoint paths, endpoint, test counts and unavailable
methods. No CE generator weights are loaded and no fitting/calibration runs.
The endpoint, temperature, frozen memory and held-out task boundary come from
the recorded experiment. Existing policy thresholds/budgets remain unchanged.
The default evaluation seed is the recorded original seed; `--seed N` sets a
new **evaluation** seed for every policy, without changing the train/dev/test
split or the fitted models. Use a separate output directory for each seed.

The new Always Use baseline starts with an empty answer cache. Every gate is
evaluated again, importing only this new baseline cache so identical prompts
share responses and no-DROP parity remains testable. Old policy evaluations
and their response caches are not replayed. Repeating a completed command
reuses this **new** run; interrupted evaluations resume per completed task.
CUDA visibility applies to Python/CSM, not an already-running Ollama service.

Old ST is included if the input CSM report registers it. If it is absent, it
is shown as `—`; optionally specify `--fever-old-st-root PATH` and/or
`--hotpotqa-old-st-root PATH` to load existing old-ST checkpoints. The entry
never trains a missing model. All three old gates and the new CSM checkpoint
must exist before any new model calls start. Global Drop is evaluated on both
datasets even if an old Global Drop row was missing.

For checkpoints saved directly by a dev ablation, use
`--fever-old-st-checkpoint FILE` / `--hotpotqa-old-st-checkpoint FILE` instead.
Each dataset accepts either a root or a checkpoint, not both. Direct checkpoint
selection reads its saved dev-calibrated kappa, delta and zero/one DROP budget;
it does not copy/move weights, recalibrate, or infer settings from the folder
name. The plan prints its actual beta, training seed and deployment policy.
For the registered beta=.01/seed=23 runs:

```bash
CUDA_VISIBLE_DEVICES=2 \
python -u -m causal_memory_control.run_trained_gate_evaluation \
  --trained-root causal_memory_control/results/csm_st_gmemory_v2 \
  --output-dir causal_memory_control/results/all_trained_gates_with_old_st_retest_v2 \
  --datasets fever,hotpotqa \
  --csm-device cuda:0 \
  --fever-old-st-checkpoint causal_memory_control/results/fever_seed0_st_beta_dev_v1/beta_0.01/checkpoint.json \
  --hotpotqa-old-st-checkpoint causal_memory_control/results/hotpotqa_seed0_st_beta001_seed23_gpu2_v1/hotpotqa/checkpoint.json \
  --plan-only
```

Remove `--plan-only` to evaluate all seven policies. A different explicit
checkpoint changes the evaluation identity and requires a separate output.

The new output contains `reevaluation_results.md`, `.csv`, `.json`,
`reevaluation_manifest.json`, `reevaluation_progress.json`,
`logs/{dataset}_{policy}.log`, and
`{dataset}/{policy}/{memory_gate.jsonl,summary.json,run_manifest.json,llm_cache.sqlite}`.
Every non-baseline policy also gets `comparison.json` with deltas, paired
confidence intervals, drop counts and retrieval/no-DROP checks. Final test
scores are for reporting; choose hyperparameters on diagnostic dev data.

Do not upload `model/`, `*.sqlite*` or entire CE part directories just to review
results: they include model weights/caches and can be large. Start with the
table, manifests, training summary, loss history, dev predictions, final
summary/comparison and runtime candidate logs. No experiments have been run by
adding this code; offline tests use randomly initialized tiny BERTs/mocks only.
