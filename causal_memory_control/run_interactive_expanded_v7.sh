#!/usr/bin/env bash
# Nine-policy expanded run. Native progress prints in this same terminal.
set -euo pipefail
if (( $# > 1 )); then
  echo "Usage: bash causal_memory_control/run_interactive_expanded_v7.sh [plan|all|smoke|experts|support|collect|labels|labels_team_u|train_reward|train_csm|train_team_u|evaluate|report]" >&2
  exit 2
fi
cmc_stage="${1:-all}"
case "$cmc_stage" in
  plan|all|smoke|experts|support|collect|labels|labels_team_u|train_reward|train_csm|train_team_u|evaluate|report) ;;
  *) echo "Unknown stage: $cmc_stage" >&2; exit 2 ;;
esac
cmc_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$cmc_repo"
cmc_output="${CMC_INTERACTIVE_OUTPUT:-$cmc_repo/causal_memory_control/results/pddl_alfworld_gate_expanded_v7_native_progress}"
cmc_gpu="${CMC_INTERACTIVE_GPU:-0}"
cmc_python="${CMC_INTERACTIVE_PYTHON:-python}"
cmc_arguments=(
  "$cmc_python" -u -X faulthandler -m causal_memory_control.run_interactive_gate_benchmark
  --repo "$cmc_repo" --output-dir "$cmc_output"
  --datasets pddl,alfworld --pddl-folds 5
  --alf-support 100 --alf-fit 800 --alf-dev 200 --alf-heldout-test 500
  --include-local-global-team-u --expert-timeout 600
  --pddl-expert barman_tyreworld_constructive
  --endpoint "${CMC_INTERACTIVE_ENDPOINT:-http://127.0.0.1:11442/v1}" --model qwen2.5:7b
  --generator-model "${CMC_INTERACTIVE_GENERATOR:-Qwen/Qwen2.5-7B-Instruct}"
  --generator-revision "${CMC_INTERACTIVE_GENERATOR_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}"
  --encoder-model bert-base-uncased --embedding-model sentence-transformers/all-MiniLM-L6-v2
  --ce-device cuda:0 --train-device cuda:0 --evaluation-device cpu
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
echo "[启动] 阶段=$cmc_stage；GPU=$cmc_gpu；输出=$cmc_output"
echo "[终端] 将自动显示数据采样、专家、建库、标签、训练和测评进度；Ctrl+C 中断本次实验。"
exec env CUDA_VISIBLE_DEVICES="$cmc_gpu" flock -n -E 75 -F "$cmc_output.lock" \
  "${cmc_arguments[@]}" --stage "$cmc_stage"
