# 扩大实验 v2：PDDL 五折与 ALFWorld 更多独立任务

用户于 2026-10-23 确认：PDDL 保留原 60 题做五折；ALFWorld 使用 support 100 / fit 800 / dev 200，额外预留 500 题测评，并独立保留 134 题 unseen 测评。旧 ST、QA 代码、旧 checkpoint 和结果不变。小规模 v1 的运行说明保留在 `PDDL_ALFWORLD_BENCHMARK.md`；此文是扩大版的命令。

## 数据与报告口径

| 数据集 | support | fit | dev | 测评 |
| --- | ---: | ---: | ---: | --- |
| PDDL 每个外层折 | 8 | 32 | 8 | 12 |
| PDDL 五折合计 | 40 次任务使用 | 160 次任务使用 | 40 次任务使用 | 60 个不同问题，各测一次 |
| ALFWorld | 100 | 800 | 200 | 500 个额外场景留出任务 + 原 134 个 unseen 任务 |

PDDL 按 domain 分层划出五个互斥的外层 test，每折 test=12。该折剩余 48 题内，按 domain 各取 2 support、2 dev，其余 fit。每折独立建立冻结记忆库，独立训练和选择模型、独立保存回答缓存；不把其他折模型或记忆库用于当前折。原 60 题依然只有 60 个独立问题，五折不会产生 160 个新问题，也不会让一个模型同时看到其他折的训练数据。模型超参数预先固定，各折只用自己的 dev 选 kappa/删除预算或 CSM epoch。

最终 PDDL 主表为 60 题的 out-of-fold（OOF）Success：每个问题只使用它所在外层 test 折的预测。保存各折 Success、折间标准差、每折 split/模型/快照/日志哈希。配对 bootstrap 按问题计算，描述这些 OOF 结果的不确定性；它不覆盖重新训练种子的全部方差，也不应把训练集重叠的五个折当成五次完全独立实验。只有 32 个 fit 问题的单折 BERT 训练仍可能不稳定，这是遵守“原 60 题、不新增生成题”选择后的数据限制。

ALFWorld 仍从符合六类任务且标记为可解的 3,553 个 train 游戏中抽样。扩大模式先按 FloorPlan 分配约 10% / 50% / 20% / 20% 的场景给 support / fit / dev / 额外 test，再在各区内按 task type 轮流抽取 100 / 800 / 200 / 500 个任务。任何场景不能跨这些分区，也不能与 134 个 unseen 游戏的场景重叠。数量不足会明确失败，不放宽隔离约束。默认 seed=71 的实际本地数据已验证能容纳这个规模。

134 个 valid_unseen 继续作为 ALFWorld 主表指标；额外 500 题以 `train_scene_holdout` 独立输出。两组不能合成“634 题官方 unseen 成绩”。JSON/CSV/MD 均包含独立分组的 Success、配对变化、CI、改善/伤害任务、保留率与独立场景数。专家 CE target 只为该实验 fit/dev 创建，不读取额外 test 的专家动作。

fit 从 80 增至 800，使 ALFWorld 的独立训练任务数增加十倍；每题多个候选、repeats 仍属于同一个任务。更多数据可以帮助估计稳定性，但不保证超过 Always Use：memory 的真实效用和标签噪声也会影响结果。v2 的 support 也增加了，v1/v2 的性能差不能全部归因为 fit 数量变化。

## 预计规模

默认 K≤6、诊断 repeats=2，所有策略与同动作共享实际 outcome。

| 项目 | PDDL 五折合计 | ALFWorld | 合计 |
| --- | ---: | ---: | ---: |
| support 团队运行 | 40 | 100 | 140 |
| 诊断团队运行上限 | 2,800 | 14,000 | 16,800 |
| 测试团队运行上限 | 540 | 5,706 | 6,246 |
| 专家验证任务使用次数 | 200 | 1,000 | 1,200 |
| 原候选列表 CE forward 上限 | 1,400 | 7,000 | 8,400 |
| 重组列表 CE forward 上限 | 2,240 | 11,200 | 13,440 |

这些是多 agent、多步的团队运行数，不是 API 请求数或工时。额外 smoke 为五个 PDDL 折及 ALFWorld 各 2 次运行，共 12 次团队运行。未启动任何完整扩大实验。

扩大验证时还修复了新适配器的 PDDL 动作解析：NLTK 不可用时，旧 fallback 会把 fill-shot、jack-up 的连字符拆开。新 episode 以原域谓词和对象解析参数并保留连字符，旧环境文件不改。tyreworld 又存在域常量与问题对象重复声明（jack/pump/wrench），严格规划器会拒绝；新适配器仅对交给规划器的文本去除同类型重复对象声明，保留所有事实和对象身份，不改原文件，并登记变换与文本摘要。生成计划仍须在未改动的原生环境回放成功。

四个 PDDL 域的代表问题已完成实际符号计划回放（blockworld、gripper，以及修复后的 barman p01 / tyreworld p01），记录了动作合法性；这不代替其余 fit/dev 专家计划的逐任务验证。

## 上传与服务器命令

上传 v2 包（原 v1 上传包保留）。包含原先 10 个新增文件及本说明，共 11 个文件；不包含 data、旧结果、SQLite、模型或无关未跟踪文件。服务器已有的完整 GMemory 仓库和数据继续使用。新代码的 hash 与 v1 不同，扩大版必须使用新的输出目录；不要将它当成旧运行的同配置恢复。

```bash
cd /mnt/mydata/hjq/casual-gmemory
conda activate gmemory
tar -xzf pddl_alfworld_gate_code_v2.tar.gz

python -m unittest causal_memory_control.tests.test_interactive_benchmark -q
nvidia-smi
ss -ltnp | grep -E ':(11434|11442)([[:space:]]|$)'
```

确认已有 HF Qwen 目录。下面的路径沿用此前建议，实际不存在时改为本机真实目录；BERT/MiniLM 可以使用已有离线缓存或本地目录。实际运行不自动下载模型。示例 endpoint 使用 11442；必须与实际 Ollama 服务一致，并在第一次运行前固定。

```bash
GENERATOR_DIR=/mnt/mydata/hjq/ollama/models/huggingface/Qwen2.5-7B-Instruct
test -f "$GENERATOR_DIR/config.json"

expanded=(
  python -u -m causal_memory_control.run_interactive_gate_benchmark
  --output-dir causal_memory_control/results/pddl_alfworld_gate_expanded_v2
  --datasets pddl,alfworld
  --pddl-folds 5
  --alf-support 100
  --alf-fit 800
  --alf-dev 200
  --alf-heldout-test 500
  --endpoint http://127.0.0.1:11442/v1
  --model qwen2.5:7b
  --generator-model "$GENERATOR_DIR"
  --encoder-model bert-base-uncased
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
  --ce-device cuda:0
  --train-device cuda:0
  --evaluation-device cpu
  --max-steps 30
  --repeats 2
)

CUDA_VISIBLE_DEVICES=5 "${expanded[@]}" --plan-only
```

计划只读，不连接服务或初始化模型，也不创建 CV 子目录。GPU 5 仅是此前记录的候选空闲卡；启动前以实际检查为准。Python 的 `CUDA_VISIBLE_DEVICES=5` 不会移动已运行的 Ollama。若确有空闲 GPU 5 和 11442，可在独立终端手动启动，不停止原服务：

```bash
CUDA_VISIBLE_DEVICES=5 \
OLLAMA_HOST=127.0.0.1:11442 \
OLLAMA_MODELS=/mnt/mydata/hjq/ollama/models \
OLLAMA_NUM_PARALLEL=1 \
ollama serve
```

先 smoke，再验证完整 fit/dev 专家，后续按阶段执行。各阶段串行，不能在同一根目录并发写入。CE 与 Ollama 各自加载一份 Qwen，labels 前检查显存。正式命令的规模明显大于 v1，应先核对 plan 的分支数：

```bash
CUDA_VISIBLE_DEVICES=5 "${expanded[@]}" --stage smoke
CUDA_VISIBLE_DEVICES=5 "${expanded[@]}" --stage experts
CUDA_VISIBLE_DEVICES=5 "${expanded[@]}" --stage support
CUDA_VISIBLE_DEVICES=5 "${expanded[@]}" --stage collect
CUDA_VISIBLE_DEVICES=5 "${expanded[@]}" --stage labels
CUDA_VISIBLE_DEVICES=5 "${expanded[@]}" --stage train_reward
CUDA_VISIBLE_DEVICES=5 "${expanded[@]}" --stage train_csm
CUDA_VISIBLE_DEVICES=5 "${expanded[@]}" --stage evaluate
CUDA_VISIBLE_DEVICES=5 "${expanded[@]}" --stage report
```

PDDL 折产物位于 `pddl_folds/fold_00` 至 `fold_04`，每个目录有独立 manifest、pddl/snapshot、训练、缓存及评测日志。根目录 `pddl/cv_sources.json` 登记每折来源；`pddl/evaluation/policies.jsonl` 是经过来源核验的 OOF 派生日志，原折日志不改。ALFWorld 仍位于根目录 `alfworld/`，其报告 `policies.<method>.cohort_results` 中包含两个独立测评组。

纯报告无需 Ollama，且可以只读导入已完成的 QA 日志：

```bash
python -m causal_memory_control.interactive_report \
  --manifest causal_memory_control/results/pddl_alfworld_gate_expanded_v2/benchmark_manifest.json \
  --qa-results-root causal_memory_control/results/latest_heldout_oracle_v1
```

旧 Oracle 未完成时，可将 qa-results-root 改为已完成的 `all_trained_gates_with_old_st_retest_v2`，缺失项留空。FEVER/HotpotQA 不重新训练或测评。

本地验证：完整 `test_interactive_benchmark` 共 22 项，覆盖五折隔离与 OOF 来源核验、扩大 ALFWorld 的场景划分、两种测评组不混合、默认 v1 行为、恢复和真实 PDDL 计划回放。扩大版 plan-only 已核验任务数与分支上限；未运行服务器训练、完整 CE 或正式测评，没有新增成绩。
