# 扩大实验 v2：PDDL 五折与 ALFWorld 更多独立任务

2026-10-26 新增的九策略版本见文末“v3：增加 Local–Global + Team U + semantic dedup”。原 v2 八策略命令和结果保留。

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

## v3：增加 Local–Global + Team U + semantic dedup

新增行复用 FEVER/HotpotQA 的 `local_global_team_u.py` 模型加载和决策类。其标签是同一原始候选列表、同初态与 sample seed 下的 `U_i = mean(Success_KEEP_ALL) - mean(Success_SINGLE_DROP_i)`。标签阶段逐条核验原始 action 文件哈希、分支身份、全部 repeats 和聚合均值，不读取专家动作或 CE，也不新增诊断团队运行。持久化字段为 `team_u`、`q_use`、`q_drop`；训练循环内部暂用通用的 `ci` 张量字段，导出的 checkpoint、dev 预测及 loss 字段均标识 Team U。

与 CE 行相同的条件包括：support/fit/dev/test、冻结 memory、检索候选与顺序、团队配置、生成模型/seed/温度/步数、BERT + 3 层 8 头列表 attention + 无界单头、优化器/学习率、seed 2024、10 epochs、batch 16、beta .1，以及 MiniLM 语义去重阈值 .98。两行的冻结去重模型身份必须一致。各折依然独立训练、独立 dev 选 epoch。

部署沿用 QA Team U 的具体规则：先对完整原列表评分，再按原检索顺序进行 greedy semantic dedup；只保留 `predicted_team_u > 0` 的非重复项。U=0 也删除，kappa=delta=0，删除数不限，没有 KEEP ALL fallback 或按保留率校准。日志保存每条预测、重复来源、删除原因和实际动作结果；同动作继续复用 ActionStore。

必须保留的协议差异：CE 先去重再标注/评分，fit 可以跨题重组并重新计算 CE；Team U 对应的是已测原列表，训练和评分保留此列表，禁用跨题重组和训练前去重。其损失使用 QA Team U 已有的 reward 尺度适配：positive/negative epsilon=.05/.25、hard threshold=.1、rarity bin width=.25、trivial keep fraction=1；CE 仍为 .1/.3、5、1、.2。两者相同的损失结构是 rarity-weighted MSE + 同列表 InfoNCE，但不能称为完全只换标签的单变量消融。若没有有效对，照实记录 InfoNCE 未激活。SINGLE DROP 的 U 不识别联合多条删除效用；Oracle 的固定动作集也不包含任意 learned 多条删除子集。

新增代码为 `interactive_protocol.py`、`interactive_team_u.py`、`interactive_demo.py`、`tests/test_interactive_team_u.py`；同步更新 `run_interactive_gate_benchmark.py`、`interactive_data.py`、`interactive_training.py`、`interactive_worker.py`、`interactive_report.py` 和本文。服务器还需要仓库中已有的 `local_global_team_u.py`。旧 QA、ST、CE 核心文件及已完成结果不改动。

默认依旧八行；加入 `--include-local-global-team-u` 后才执行九行。更新源码和策略集合需要新输出目录，旧 manifest 和输出保持原状；不通过修改旧 config_digest 来混入不同协议的分支。下列完整命令在服务器自行执行，当前未启动正式实验：

```bash
cd /mnt/mydata/hjq/casual-gmemory
conda activate gmemory

python -m unittest causal_memory_control.tests.test_interactive_benchmark \
  causal_memory_control.tests.test_interactive_team_u -q

expanded_team_u=(
  python -u -m causal_memory_control.run_interactive_gate_benchmark
  --output-dir causal_memory_control/results/pddl_alfworld_gate_expanded_v3_team_u
  --datasets pddl,alfworld --pddl-folds 5
  --alf-support 100 --alf-fit 800 --alf-dev 200 --alf-heldout-test 500
  --include-local-global-team-u
  --endpoint http://127.0.0.1:11442/v1 --model qwen2.5:7b
  --generator-model /mnt/mydata/hjq/ollama/models/huggingface/Qwen2.5-7B-Instruct
  --encoder-model bert-base-uncased
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
  --ce-device cuda:0 --train-device cuda:0 --evaluation-device cpu
  --max-steps 30 --repeats 2
)

CUDA_VISIBLE_DEVICES=5 "${expanded_team_u[@]}" --plan-only
CUDA_VISIBLE_DEVICES=5 "${expanded_team_u[@]}" --stage smoke
# 核对数据、模型路径、GPU/endpoint、smoke 和预算后，用户启动正式全流程：
CUDA_VISIBLE_DEVICES=5 "${expanded_team_u[@]}" --stage all
```

设备与端点须按实际服务指定。九策略 stages 为 experts → support → collect → labels → labels_team_u → train_reward → train_csm → train_team_u → evaluate → report。新增 Team U 标签不加载 Qwen；新增 BERT 训练和推理分别留在各自子进程。已有 v3 同配置采集完成后也可以单独运行 `--stage labels_team_u`、`--stage train_team_u`。

规模保持 support=140、诊断上限=16,800、专家任务使用=1,200、CE forward 上限不变。测试团队运行上限从 6,246 增至 6,940：PDDL 600、ALFWorld 6,340；新增至多 694，若 Team U 选中已有动作则更少。模型训练增加五个独立 PDDL 折及一个 ALFWorld 的 Team U 模型。主表共九行，PDDL 为 60 题 OOF，ALFWorld 为 134 unseen 和额外 500 独立列；逐任务、配对比较、CI、loss、进度和哈希继续保存。旧 QA 只读导入；没有被验证导入的 QA Team U 分数保持空白。

只生成模拟输出无需数据文件、环境、模型或服务：

```bash
python -m causal_memory_control.interactive_demo \
  --output-dir causal_memory_control/results/pddl_alfworld_team_u_simulation
```

该命令用带 `sim:` ID 的人工构造日志调用正式报告器，生成 `example_results.md` 和 `matrix_results.md/.csv/.json`。模拟配置、原始策略行、汇总 JSON 和 CSV 都带 simulation 标记，MD 顶部明确说明，不可与实测行混用；模拟器需要全新空目录。它仅展示九行表格及两组 ALFWorld 测评的输出格式，不是新方法性能预测。

本次本地验证共 32 项通过：原交互基准 24 项、新 Team U 8 项，包含真实小型 BERT 训练/保存/重载、配对分支篡改拒绝、去重模型身份一致、无限删除、动作复用、恢复覆盖、五折 OOF 和模拟/实测隔离。完整九策略只读 plan 已核对上述规模；未运行正式模型训练或服务器实验，也未重跑 FEVER/HotpotQA。

## 共享服务器上的专家超时

若单测提示 `expert subprocess timed out after 120s: pddl:barman:p01`，表示专家子进程的总墙钟预算用完；该预算包含冷启动依赖导入、环境初始化、PDDL 转换、原生规划搜索及回放，不能据此认定数据损坏或规划器一定无解。GPU 和 Ollama 端口不参与这个 CPU 专家检查。

修复后，专家子进程单独隐藏 GPU、将 BLAS/OpenMP 线程限制为 1，并强制 HF offline；父进程的训练设备/线程设置保持原值。`expert_phase` 日志区分 environment_init、pddl_translate、planner_load、planner_search、native_replay 和 validated；成功文件保留阶段耗时。慢任务每 60 秒输出 faulthandler 栈，超时异常保留子进程输出，拒绝生成不完整标签。单测默认每个真实专家任务 300 秒，可用 `CMC_EXPERT_TEST_TIMEOUT` 增加；正式实验仍由 `--expert-timeout` 单独控制，默认 120 秒。

本地同步本次修改的 `interactive_envs.py`、`interactive_worker.py`、`tests/test_interactive_benchmark.py` 和本文后，在服务器先单独验证 Barman，再执行完整测试：

```bash
CMC_EXPERT_TEST_TIMEOUT=600 python -X faulthandler -m unittest \
  causal_memory_control.tests.test_interactive_benchmark.PipelineTests.test_barman_expert_hyphen_actions_replay_without_nltk -v

CMC_EXPERT_TEST_TIMEOUT=600 python -X faulthandler -m unittest \
  causal_memory_control.tests.test_interactive_benchmark \
  causal_memory_control.tests.test_interactive_team_u -q
```

正式实验若采用每任务 600 秒，应在定义数组时显式加入 `--expert-timeout 600`，并先运行 `--stage experts` 验证全部 fit/dev 专家，成功后再开始 support 和诊断 LLM 采集。若已生成旧 manifest，本次源码/超时设置变更须选新输出目录（如 `pddl_alfworld_gate_expanded_v4_team_u`）；保留旧产物。增加等待时间不会放宽成功回放、非法动作数或 CE target 验证。若仍超时，查看异常中的最后一个 `expert_phase` 和 faulthandler 栈，再根据实际阶段排查。
