#!/usr/bin/env bash
# Eight policies: exclude CE, retain Local–Global Team U and native progress.
set -euo pipefail
if (( $# > 1 )); then
  echo "Usage: bash causal_memory_control/run_interactive_expanded_v8.sh [plan|all|smoke|support|collect|labels_team_u|train_reward|train_team_u|evaluate|report]" >&2
  exit 2
fi
cmc_stage="${1:-all}"
case "$cmc_stage" in
  plan|all|smoke|support|collect|labels_team_u|train_reward|train_team_u|evaluate|report) ;;
  *) echo "Stage is unavailable in this CE-excluded run: $cmc_stage" >&2; exit 2 ;;
esac
cmc_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$cmc_repo"
cmc_output="${CMC_INTERACTIVE_OUTPUT:-$cmc_repo/causal_memory_control/results/pddl_alfworld_gate_expanded_v8_no_ce}"
cmc_gpu="${CMC_INTERACTIVE_GPU:-0}"
cmc_python="${CMC_INTERACTIVE_PYTHON:-python}"
cmc_arguments=(
  "$cmc_python" -u -X faulthandler -m causal_memory_control.run_interactive_gate_benchmark
  --repo "$cmc_repo" --output-dir "$cmc_output"
  --datasets pddl,alfworld --pddl-folds 5
  --alf-support 100 --alf-fit 800 --alf-dev 200 --alf-heldout-test 500
  --include-local-global-team-u --exclude-ce
  --endpoint "${CMC_INTERACTIVE_ENDPOINT:-http://127.0.0.1:11442/v1}" --model qwen2.5:7b
  --encoder-model bert-base-uncased --embedding-model sentence-transformers/all-MiniLM-L6-v2
  --train-device cuda:0 --evaluation-device cpu
  --max-steps 30 --few-shots 1 --temperature 0.7
  --successful-topk 3 --insights-topk 3 --repeats 2 --split-seed 71
  --diagnostic-seed 0 --evaluation-seed 2000 --random-seeds 11,23,47 --ensemble-size 12
)
if [[ "$cmc_stage" == plan ]]; then
  exec "${cmc_arguments[@]}" --stage all --plan-only
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "Missing flock; this launcher requires Linux util-linux flock." >&2
  exit 2
fi
mkdir -p -- "$(dirname -- "$cmc_output")"
echo "[启动] 八策略，跳过 CE；阶段=$cmc_stage；GPU=$cmc_gpu；输出=$cmc_output"
echo "[终端] 自动显示采样、建库、诊断、Team U 标签、训练、测评和汇总进度。"
exec env CUDA_VISIBLE_DEVICES="$cmc_gpu" flock -n -E 75 -F "$cmc_output.lock" \
  "${cmc_arguments[@]}" --stage "$cmc_stage"
