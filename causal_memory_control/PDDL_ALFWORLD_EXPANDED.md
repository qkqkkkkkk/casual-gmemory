# 扩大实验 v2：PDDL 五折与 ALFWorld 更多独立任务

当前用户选择跳过 CE，只运行其余八策略，见文末“v8：只跑其余八策略，跳过 CE”。终端直接输出与 tyreworld 超时修复见“v7：主程序直接输出进度与 tyreworld 专家”。先前修复和九策略命令保留。

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

## v6：barman 专家与实时进度条（2026-10-27）

服务器 v5 的实际异常已定位到 `pddl:barman:p09`：环境/依赖初始化约 142 秒，之后停在 `planner_search`，最终触发包含整个子进程生命周期的 600 秒硬超时。日志中的每次 `Timeout (0:01:00)` 是 faulthandler 的周期栈输出，最终的 `subprocess.TimeoutExpired` 才是任务失败。增加 GPU 显存无法加速这个 CPU 符号搜索。

新增可选参数 `--pddl-expert barman_constructive`。barman 按当前问题的配方和正向 contains 目标构造符号动作，用目标 shot 量取配料、倒入 shaker、清洗 shot、摇匀并倒回目标 shot。每次处理后恢复空净 shaker，已经满足的目标 shot 不再使用。构造过程不搜索备用 shot 的排列，也不改变原域、问题、目标或动作语义。生成的每一步仍通过原生动作解释器和环境执行；仅成功且非法动作数为零的计划才保存、进入 fit/dev 的 CE 标签。其他三个 PDDL 域仍用 Fast Downward；ALFWorld 仍用原手写文本专家。

默认 `--pddl-expert fast_downward` 保留旧入口。新记录的 `source=barman_constructive_symbolic_v1`、执行后端、动作、原生轨迹、成功验证和源码哈希区分两种监督来源；不把新计划称为 Fast Downward 输出，也不与旧 CE 标签混用。目标依旧是完整成功计划的冻结 Qwen CE proxy，不是最短计划、逐步观测 CE 或真实团队 reward。构造计划的长度可能与旧规划器不同，应在报告中保留此监督来源说明。

本地完整回归 36 项通过，包括全部 20 个注册 barman 问题的成功原生回放、零非法动作、p09 独立专家子进程以及错误状态保留任务/折/日志位置。未运行正式 Qwen 训练、完整团队评测或旧 QA 实验。服务器还需运行新后端的原生回放检查。

新增文件：`interactive_barman.py`、`watch_interactive_progress.py`、`run_interactive_expanded_v6.sh`、`tests/test_interactive_barman.py`。同时更新 `interactive_envs.py`、`interactive_worker.py`、`run_interactive_gate_benchmark.py`、`tests/test_interactive_benchmark.py` 和本文。原 FEVER/HotpotQA、旧 gate 和 CE 核心未修改。因为专家后端和登记源码改变，启动脚本使用新目录 `pddl_alfworld_gate_expanded_v6_barman`，保留 v5 数据与日志。

服务器同步代码后执行；先确认旧实验进程已退出，再更新源码：

```bash
cd /mnt/mydata/hjq/casual-gmemory
conda activate gmemory

CMC_EXPERT_TEST_TIMEOUT=600 python -m unittest \
  causal_memory_control.tests.test_interactive_barman -v

bash causal_memory_control/run_interactive_expanded_v6.sh plan
```

启动脚本完整定义九策略、PDDL 五折、ALFWorld 100/800/200+500+134、预算、种子与设备参数。默认 GPU 0、端点 11442；HF teacher 使用已找到的离线仓库缓存，并固定到 `a09a35458c702b33eeacc393d103063234e8bc28`，不重新下载。使用 Linux `flock`，锁文件在结果目录外；检测到同一目录已有运行时自动转为查看其进度，不再启动第二份实验。检查和实际加锁之间若遇到并发启动，实际加锁失败仍退出 75。脚本不依赖之前终端的数组或变量。可用 `CMC_INTERACTIVE_GPU`、`CMC_INTERACTIVE_ENDPOINT`、`CMC_INTERACTIVE_OUTPUT`、`CMC_INTERACTIVE_GENERATOR`、`CMC_INTERACTIVE_GENERATOR_REVISION`、`CMC_INTERACTIVE_PYTHON` 显式覆盖；实验配置改变时选新输出目录。

前台运行会在同一终端自动打印阶段、数据集/折、任务计数进度条、当前任务和新增阶段日志，无需再打开查看终端。此时 Ctrl+C 会中断前台实验：

```bash
bash causal_memory_control/run_interactive_expanded_v6.sh all
```

后台启动一次，启动日志放在结果目录外：

```bash
nohup bash causal_memory_control/run_interactive_expanded_v6.sh all \
  >> /mnt/mydata/hjq/pddl_alfworld_v6_barman.launch.log 2>&1 &
echo "实验 PID：$!"
```

若已在后台启动，在当前终端跟随现有实验输出即可，无需重启。下面的 `follow` 只读、不加载模型、不启动第二份实验；Ctrl+C 仅退出查看，后台实验继续运行：

```bash
bash causal_memory_control/run_interactive_expanded_v6.sh follow
```

查看器也可直接运行；不加 `--stream` 时采用刷新式显示：

```bash
cd /mnt/mydata/hjq/casual-gmemory
python causal_memory_control/watch_interactive_progress.py \
  --output-dir causal_memory_control/results/pddl_alfworld_gate_expanded_v6_barman \
  --stream
```

进度条是当前阶段的已保存任务/总任务数，不是总耗时比例。专家、建库、诊断和测试计数从登记 split 与已保存任务文件交叉计算，因此阶段失败后仍显示已保存工作；自动跟随各折和 ALFWorld，附带当前阶段日志。模型训练显示原有训练日志，刷新式查看也读取已记录 epoch/loss；没有可用计数时明确显示未知，不伪造百分比或 ETA。串流每秒检查变化，单个长任务每 30 秒打印一次当前进度。后台任务结束时串流自动退出；状态文件仍保留最后记录。

直接运行查看器会先移除脚本目录在 `sys.path` 中的入口，避免同目录 `types.py` / `logging.py` 遮蔽 Python 标准库。查看器和启动脚本都不在实验源码哈希集合中，可以在 v6 正在运行时单独更新这两项；更新后不会重启或改变当前实验。

只读查看旧 v5 的失败原因，包含已保存的完整 traceback：

```bash
python causal_memory_control/watch_interactive_progress.py \
  --output-dir causal_memory_control/results/pddl_alfworld_gate_expanded_v5_team_u_hfcache \
  --once
```

查看器能从旧子进程记录找回丢失的 dataset/fold。v6 父进程会直接在错误状态保留 dataset、命令和 log_path；子进程失败时保留最后的任务 ID、计数与 total。查看器文件不在 `interactive_*.py` 哈希集合中，单独添加/修改显示工具不改变实验身份。`--once` 读到 failed 时退出 1，表示已保存的实验失败，并非查看器重新启动了实验。

## v7：主程序直接输出进度与 tyreworld 专家（2026-10-27）

v6 的后续错误已从服务器日志确认：`pddl/fold_01` 的专家阶段在 `pddl:tyreworld:p02` 停止，已完成 35/40 题。该题的环境初始化约 4.66 秒，进入 `planner_search` 约 4.95 秒，最终耗尽 600 秒。源问题内部名为 tyreworld-9，包含 9 个轮毂及车轮，但目标只要求更换其中两个；通用搜索探索大量无关动作组合。这不是查看器的 types 导入错误，也不是 GPU 显存不足的证据。

增加可选后端 `--pddl-expert barman_tyreworld_constructive`：barman 沿用 v6 构造器，tyreworld 根据初态和目标只拆卸必要轮胎，获取工具、处理紧/松螺母、顶升、拆装、归还旧轮胎及充气。blockworld/gripper 仍用 Fast Downward。保留旧 fast_downward/barman_constructive 两种选项。新 tyreworld 记录为 `source=tyreworld_constructive_symbolic_v1`；每步仍须在原生环境执行，最终 Success=1、非法动作数=0 才保存为 fit/dev 的 CE target。不承诺最短计划，不改原域、问题或目标，不把构造结果当成 Fast Downward 输出。该监督来源及其计划长度必须在结果中说明。

主程序从启动起直接打印数据扫描/划分、源码和服务校验，以及十个实验阶段：专家验证 → 建库 → 诊断采集 → CE 标签 → Team U 标签 → Reward Gate 训练 → CE 训练 → Team U 训练 → 测评 → 汇总。每个数据集/折打印实际任务计数，测评附当前策略/分支，训练原有 epoch/batch 进度直接转发。百分比表示文件扫描数、任务数或阶段数，不是预计时间。子进程输出同时写入原阶段日志并转发至当前终端，包括错误与原有 tqdm 的回车刷新；无需启动查看器或打开其他终端。前台 Ctrl+C/SIGTERM 会清理本次 worker 及其专家子进程，不影响其他服务。

协议、九策略、划分、seed、动作预算和 GPU/模型默认值不变。由于专家来源和登记代码变化，v7 默认新目录为 `pddl_alfworld_gate_expanded_v7_native_progress`；不得修改旧 manifest 来恢复 v6。查看器的直接入口也已修复 types.py/logging.py 遮蔽标准库的问题，仍可用于只读检查旧结果。

### 本地打包与自行上传

`interactive_v7_upload_files.txt` 精确列出需要同步的 17 个文件，包含说明、测试、既有 barman 后端和独立进度工具；没有数据、模型、旧结果、SQLite 或 FEVER/HotpotQA 修改。本次 v7 不自动上传或启动服务器实验。

在本地 Mac 的项目根目录执行：

```bash
cd /Users/xiaobaobei/gmemory
COPYFILE_DISABLE=1 tar -czf /private/tmp/gmemory_interactive_v7_fix.tar.gz \
  -T causal_memory_control/interactive_v7_upload_files.txt
scp /private/tmp/gmemory_interactive_v7_fix.tar.gz \
  szu:/mnt/mydata/hjq/casual-gmemory/
```

`szu` 使用此前已验证的本地 SSH 别名。先确认下面的旧实验进程检查没有结果，再在服务器解压；不要在旧实验运行时更换源码：

```bash
cd /mnt/mydata/hjq/casual-gmemory
conda activate gmemory
pgrep -afu "$(id -u)" 'causal_memory_control[.](run_interactive_gate_benchmark|interactive_worker|interactive_envs)'

# 上面无匹配进程后再执行：
mkdir -p /mnt/mydata/hjq/cmc_deploy_backups
tar -czf "/mnt/mydata/hjq/cmc_deploy_backups/before_v7_$(date +%Y%m%d_%H%M%S).tar.gz" \
  causal_memory_control/interactive_data.py \
  causal_memory_control/interactive_envs.py \
  causal_memory_control/interactive_worker.py \
  causal_memory_control/interactive_training.py \
  causal_memory_control/interactive_team_u.py \
  causal_memory_control/run_interactive_gate_benchmark.py
tar -xzf gmemory_interactive_v7_fix.tar.gz
```

### 清理两次失败实验（可选，删除后不能从这些结果恢复）

以下只清理已定位失败的 v5/v6 结果、其启动日志和锁文件。若任何此类实验仍有主/子进程，命令会跳过删除；有匹配进程时先按 PID 检查归属，不能直接删除仍被写入的目录。不要停止 Ollama。FEVER/HotpotQA、其他 results 和 data 均不在删除清单中。

```bash
cd /mnt/mydata/hjq/casual-gmemory
if pgrep -afu "$(id -u)" 'causal_memory_control[.](run_interactive_gate_benchmark|interactive_worker|interactive_envs)'; then
  echo "仍有实验进程：未删除。先确认这些进程属于哪次实验。"
else
  cmc_check_status=$?
  if [ "$cmc_check_status" -eq 1 ]; then
    rm -rf -- \
      causal_memory_control/results/pddl_alfworld_gate_expanded_v5_team_u_hfcache \
      causal_memory_control/results/pddl_alfworld_gate_expanded_v6_barman
    rm -f -- \
      /mnt/mydata/hjq/pddl_alfworld_v5_team_u_hfcache.launch.log \
      /mnt/mydata/hjq/pddl_alfworld_v6_barman.launch.log \
      /mnt/mydata/hjq/pddl_alfworld_v5_team_u_hfcache.lock \
      causal_memory_control/results/pddl_alfworld_gate_expanded_v5_team_u_hfcache.lock \
      causal_memory_control/results/pddl_alfworld_gate_expanded_v6_barman.lock
    echo "已删除这两次失败实验及其启动日志。"
  else
    echo "进程检查失败：未删除。"
  fi
fi
```

### 验证和同一终端运行

```bash
cd /mnt/mydata/hjq/casual-gmemory
conda activate gmemory

CMC_EXPERT_TEST_TIMEOUT=600 python -X faulthandler -m unittest \
  causal_memory_control.tests.test_interactive_barman \
  causal_memory_control.tests.test_interactive_tyreworld \
  causal_memory_control.tests.test_interactive_progress -q

bash causal_memory_control/run_interactive_expanded_v7.sh plan
bash causal_memory_control/run_interactive_expanded_v7.sh smoke
```

检查模型/数据、服务、资源及上述输出后，正式运行一次：

```bash
set -o pipefail
bash causal_memory_control/run_interactive_expanded_v7.sh all 2>&1 \
  | tee /mnt/mydata/hjq/pddl_alfworld_v7_native_progress.launch.log
```

不用 nohup 或 `&`，进度立即显示在当前终端。终端关闭可能中断前台任务；需要断线后保留可先进入自己的 tmux 会话后运行同一条命令。同配置再次执行恢复 v7 已保存任务；不从 v5/v6 导入不同后端的 CE 计划或旧标签。更改 GPU 可在命令前设置 `CMC_INTERACTIVE_GPU`，服务仍须对应指定 endpoint。

本地最终回归 43 项通过（88.263 秒）：覆盖旧交互基准与 Team U、全部 20 个 barman 和 10 个 tyreworld 原生回放、p09/p02 独立专家子进程、终端输出在子进程退出前到达、原始日志完整保留、失败码透传、终止时清理专家后代、直接查看器入口，以及采样进度不改变任务划分。完整 v7 plan-only 已验证九策略、五折和 ALFWorld 两组测评规模，无写入/模型加载/服务调用。未启动 v7 的正式训练或测评，未重跑 FEVER/HotpotQA；这些测试不是实际策略成绩。

## v8：只跑其余八策略，跳过 CE

用户于 2026-10-27 选择先不跑 CE 方法。本轮使用 `--include-local-global-team-u --exclude-ce`，保留：Always Use、Always Global Drop、Random Drop、Semantic Linear、Semantic MLP、Semantic ST、Local–Global + Team U + semantic dedup、Oracle，共八行。旧 CE 代码和默认九策略能力保留；未传 `--exclude-ce` 的原入口行为保留。

本轮阶段为：数据采样/划分与校验 → 建立 support 记忆库 → 诊断采集 → Team U 标签核验 → Reward Gate 训练 → Team U 训练 → 测评 → 汇总。正式七阶段中的 `experts`、`labels`、`train_csm` 完全不执行，smoke 同样不生成专家计划。手动要求这些禁用阶段会直接报配置错误。Team U 仍只用同题配对的真实团队 Success 差，保留原来的完整候选列表评分、后置语义去重和 `U>0` 保留规则。

本轮不需要 HF Qwen CE 权重，也不加载 CE scorer；Ollama 的 qwen2.5:7b 仍用于交互生成，BERT 与 MiniLM 仍需已备好的本地缓存。八策略的模型/报告检查不要求 csm_st checkpoint。报告只输出八行，并登记 `ce_enabled=false`。PDDL 五折及 ALFWorld 100/800/200+500+134、split seed、团队配置、随机种子和其他预算不变；ALFWorld 134 unseen 与额外 500 继续分开报告。

上限：support 团队运行 140、诊断团队运行 16,800、测试团队运行 6,246（PDDL 540、ALFWorld 5,706），专家任务和 CE forward 都为零。Team U 与已有动作继续复用缓存；这不是 API 请求数量或工时估计。新目录 `pddl_alfworld_gate_expanded_v8_no_ce` 区分此前失败的 v5/v6 和九策略 v7，不编辑旧 manifest、不继承旧 CE 标签。

本地手动上传包现列出 27 个确定文件（包含后续请求错误诊断、动作输出回归与 support 迁移），清单为 `interactive_v8_upload_files.txt`。保留 v7 构造器用于以后选择恢复 CE，但 v8 不调用。无数据、模型、结果、FEVER/HotpotQA 改动；不自动上传。

```bash
# 本地 Mac 执行：
cd /Users/xiaobaobei/gmemory
COPYFILE_DISABLE=1 tar -czf /private/tmp/gmemory_interactive_v8_no_ce.tar.gz \
  -T causal_memory_control/interactive_v8_upload_files.txt
scp /private/tmp/gmemory_interactive_v8_no_ce.tar.gz szu:/mnt/mydata/hjq/casual-gmemory/
```

在服务器确认旧主/子进程均无匹配后备份代码并解压；上文的“清理两次失败实验”命令仍可使用，只删除 v5/v6，不删除 v8：

```bash
cd /mnt/mydata/hjq/casual-gmemory
conda activate gmemory
pgrep -afu "$(id -u)" 'causal_memory_control[.](run_interactive_gate_benchmark|interactive_worker|interactive_envs)'

# 无匹配进程后执行：
mkdir -p /mnt/mydata/hjq/cmc_deploy_backups
tar -czf "/mnt/mydata/hjq/cmc_deploy_backups/before_v8_$(date +%Y%m%d_%H%M%S).tar.gz" \
  causal_memory_control/*.py
tar -xzf gmemory_interactive_v8_no_ce.tar.gz

python -X faulthandler -m unittest \
  causal_memory_control.tests.test_interactive_no_ce \
  causal_memory_control.tests.test_interactive_progress -q

bash causal_memory_control/run_interactive_expanded_v8.sh plan
bash causal_memory_control/run_interactive_expanded_v8.sh smoke
```

只读计划应显示 `ce_enabled=false`、8 个 policies、7 个 stages、`expert_episodes=0`、两类 CE forward 上限=0，并保留 local_global_team_u。smoke 验证原环境和 no-DROP parity，`expert_check=skipped_ce_excluded` 是预期状态。

正式在当前终端运行一次，自动显示数据采样、阶段、任务、策略和训练进度并保存日志：

```bash
set -o pipefail
bash causal_memory_control/run_interactive_expanded_v8.sh all 2>&1 \
  | tee /mnt/mydata/hjq/pddl_alfworld_v8_no_ce.launch.log
```

不要改回 v7 启动脚本，也不用 nohup 或 `&`；终端直接显示进度。需要断线保留时先进入 tmux 再运行。设备默认 GPU 0、Ollama 11442；`CMC_INTERACTIVE_GPU` / `CMC_INTERACTIVE_ENDPOINT` 可按已核实的实际服务设置。正式结果仍需运行后生成，本次没有新增策略实测成绩。

v8 最终本地回归 47 项通过（88.377 秒），包括八策略配置/报告/恢复不要求 CE checkpoint、诊断 Team U 使用原分支、smoke 不调用专家、显式 CE 阶段拒绝，以及旧九策略入口继续工作。完整 v8 只读计划确认八行、七阶段、专家与 CE forward 为零、测试 schedule 上限合计 6,246、输出目录未创建。未运行服务器正式实验，也未执行 FEVER/HotpotQA 的训练或测评。

### v8 的真实生成请求失败排查

服务器单测 9 项通过、plan 通过后，真实 smoke 在 `pddl/fold_00` 的第一次 solver 调用失败。日志顺序为 `LLM request failed after four attempts`（原生 solver 又重试三次），然后原生 MacNet 使用尚未赋值的 `action`，触发 `UnboundLocalError`。该 smoke 没有通过，不能据单测和模型身份校验成功就启动 all。Ollama `/api/tags` / `/api/show` 读取成功，只说明元信息可读；不保证 `/v1/chat/completions` 能实际生成。

现有日志没有打印底层 SDK/HTTP 错误，因此还不能判断连接、超时、API 参数或 Ollama runner/GPU 是否出了问题。Gym/FINCH/Chroma telemetry 的警告之后程序继续执行，它们没有直接触发这次终止。plan 的 `local_model_paths_exist=false` 是把模型仓库 ID 当成本地路径检查的结果，不足以判定 HF 缓存缺失；BERT 训练缓存仍需在实际训练前核实。

先在服务器运行一个最小生成请求并保留 HTTP 状态码和响应体：

```bash
curl --noproxy '*' -sS -i --connect-timeout 5 --max-time 90 \
  http://127.0.0.1:11442/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5:7b","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":8,"temperature":0.7,"seed":2000,"n":1,"stream":false}'
```

若 curl 能生成、实验仍失败，再通过同一 OpenAI SDK 检查异常，以便区分 HTTPX 的代理/连接设置与服务生成问题：

```bash
python - <<'PY'
from openai import OpenAI
with OpenAI(base_url="http://127.0.0.1:11442/v1", api_key="EMPTY",
            timeout=180, max_retries=0) as client:
    result = client.chat.completions.create(
        model="qwen2.5:7b", messages=[{"role":"user", "content":"Reply OK"}],
        temperature=0.7, max_tokens=16, stop=["\n"], n=1, seed=2000)
    print(repr(result.choices[0].message.content))
PY
```

诊断修复只改 `interactive_runtime.py`：StrictChat 保留异常原因链、异常类型和 HTTP 状态；NativeRunner 将原生错误重新关联到请求原因，拒绝保存不完整 episode。失败事件直接打印并进入阶段日志。成功请求、采样参数、缓存键和配对检查不变；没有修改共享的 seeded_client.py 或原生 MacNet，也没有给 action 填假值或把接口失败算成任务失败成绩。本地 13 项针对性检查通过，包含真实原生 MacNet 配合假 HTTP 503/空响应时保留根因、正常无 DROP 回放、八策略和终端日志；没有实际远程模型调用。

用户手动通过 Git 同步本次诊断修改时，只提交这四个文件：

```bash
git add causal_memory_control/interactive_runtime.py \
  causal_memory_control/tests/test_interactive_request_errors.py \
  causal_memory_control/interactive_v8_upload_files.txt \
  causal_memory_control/PDDL_ALFWORLD_EXPANDED.md
git commit -m "Preserve original LLM request errors in interactive runs"
git push origin main
```

服务生成问题修复后，再运行 smoke。若同时应用了这次适配器源码修改，因为实现哈希已改变，要保留失败的 v8 目录并使用新的输出目录：

```bash
export CMC_INTERACTIVE_OUTPUT=/mnt/mydata/hjq/casual-gmemory/causal_memory_control/results/pddl_alfworld_gate_expanded_v8_no_ce_requestfix
bash causal_memory_control/run_interactive_expanded_v8.sh smoke
```

只有五折 PDDL 和 ALFWorld smoke 全部通过后，再用同一个 `CMC_INTERACTIVE_OUTPUT` 执行 all。如果只修复外部 Ollama 服务、源码和原配置完全未变，可以恢复原 v8 目录，无需仅因服务曾报错而删除实验数据。

### 动作输出修复：decision 节点遗漏任务指令

本次服务器 fold_00 的 8 个 support 任务均为 `success=0`、`step_limit=30`，共 240 步中的 168 步无效，最终 insights 数为零。当前协议只暴露成功轨迹和 insights，所以 `collect` 的 `no diagnostic candidates` 是结果；数据库文件存在不代表有可检索的成功轨迹。`gripper:prob14` 的前 8 步均为 `## Action Plan`，证明提交给环境的是标题而非动作。

原生 MacNet 只把三个 solver 放入 `agents_team`，最终提交环境动作的 `_decision_node` 不在其中。适配器原先只遍历 `agents_team` 添加领域指令，导致 decision 缺少动作语法规则。节点还配置了 `stop=["\n"]`，标题后的换行会结束生成，原生 `process_action` 也只保留第一行，因此不能指望从标题后提取尚未生成的动作。

修复位于 `interactive_runtime.py`：对三个 solver 和 decision 统一追加原有任务规则及“只输出下一步的一条命令、不加标题或说明”的要求；保持采样温度、步数、换行停止、检索、标签和八策略设置。v9 首版把纯标题、代码围栏或多行回复视为 `interactive_action_format_failed` 并拒绝提交 episode；后续 ALFWorld 在第 54 个 support 遇到代码围栏，这条过严的处理已由下一节的 warning 修复替代。合法性仍由环境判断，不用规划器替换动作，不过滤普通非法动作，也不把部分目标进度当成成功。共享 MacNet、FEVER/HotpotQA 和 seeded client 不修改。本修复不保证真实 Qwen 每题成功。

本地测试使用固定回复驱动真实 Gripper `prob14`：全部四个节点收到任务/输出规则，12 步完成目标、无效动作为零，并通过无 DROP 回放。v9 首版另测试了纯标题回复中止；下一节已改为验证格式错误正常消耗步数并保留失败结果。这里没有调用服务器上的 Qwen。

以下命令每一条占一整行，逐条复制执行，不使用 heredoc 或反斜杠续行。本地只暂存这五个相关文件（同时包含先前尚未提交的请求错误诊断）：

```bash
git add causal_memory_control/interactive_runtime.py causal_memory_control/tests/test_interactive_request_errors.py causal_memory_control/tests/test_interactive_action_output.py causal_memory_control/interactive_v8_upload_files.txt causal_memory_control/PDDL_ALFWORLD_EXPANDED.md
git commit -m "Fix interactive action instructions for the decision node"
git push origin main
```

服务器先确认没有旧实验运行，再查看本地源码改动并拉取：

```bash
cd /mnt/mydata/hjq/casual-gmemory
pgrep -af 'causal_memory_control\.(run_interactive_gate_benchmark|interactive_worker|interactive_envs)'
git status --short
git pull --ff-only origin main
python -X faulthandler -m unittest causal_memory_control.tests.test_interactive_action_output causal_memory_control.tests.test_interactive_request_errors causal_memory_control.tests.test_interactive_no_ce causal_memory_control.tests.test_interactive_progress -q
```

`pgrep` 无匹配的 PID 行表示这三个实验入口没有在运行；若仍有实际 PID，则先处理本次旧实验，勿并发写结果。若 `git pull` 因服务器源码改动拒绝合并，先保留并核对这些改动，不运行 `git reset --hard`。

提示词和实现身份改变，必须新建输出目录，保留原 v8 库用于排错。沿用已验证的 `CMC_INTERACTIVE_ENDPOINT` 和 `CMC_INTERACTIVE_GPU`；未设置时启动器分别使用 11442 与 GPU 0。先 smoke，再只验证第一折的 8 个 support 任务，避免尚未检查库可用性就运行全部 support：

```bash
export CMC_INTERACTIVE_OUTPUT=/mnt/mydata/hjq/casual-gmemory/causal_memory_control/results/pddl_alfworld_gate_expanded_v9_actionfix_no_ce
bash causal_memory_control/run_interactive_expanded_v8.sh smoke
CUDA_VISIBLE_DEVICES="${CMC_INTERACTIVE_GPU:-0}" flock -n -E 75 -F "$CMC_INTERACTIVE_OUTPUT.lock" python -u -X faulthandler -m causal_memory_control.interactive_worker --config "$CMC_INTERACTIVE_OUTPUT/pddl_folds/fold_00/benchmark_manifest.json" --dataset pddl --stage support
python -c 'import json,os; from pathlib import Path; r=Path(os.environ["CMC_INTERACTIVE_OUTPUT"])/"pddl_folds/fold_00/pddl"; rows=[json.loads((r/json.loads(p.read_text())["directory"]).parent.joinpath("episode.json").read_text()) for p in sorted((r/"support/committed").glob("*.json"))]; print("committed=",len(rows),"successes=",sum(x["outcome"]["success"] for x in rows)); [print(x["task_id"],x["outcome"]) for x in rows]'
```

第一折 support 有成功轨迹后，可用同一 manifest 验证 collect 的实际候选。以下命令会启动该折的正式诊断采集，已有分支可恢复，不是无成本检查：

```bash
CUDA_VISIBLE_DEVICES="${CMC_INTERACTIVE_GPU:-0}" flock -n -E 75 -F "$CMC_INTERACTIVE_OUTPUT.lock" python -u -X faulthandler -m causal_memory_control.interactive_worker --config "$CMC_INTERACTIVE_OUTPUT/pddl_folds/fold_00/benchmark_manifest.json" --dataset pddl --stage collect
```

第一折诊断成功后继续 all，将复用新目录中已提交的第一折 support 和诊断结果，继续其他折及 ALFWorld。旧 v8 数据不迁入新协议：

```bash
set -o pipefail
bash causal_memory_control/run_interactive_expanded_v8.sh all 2>&1 | tee /mnt/mydata/hjq/pddl_alfworld_v9_actionfix_no_ce.launch.log
```

若新 support 仍全部失败或格式错误，保留输出并检查新轨迹，不直接放宽 success、不移除空候选校验、不增加步数掩盖格式问题。smoke 成功仍不代表 support 库可用。

### v10：模型格式错误交给环境反馈，并保留 v9 的已提交 support

服务器 v9 在 ALFWorld support 的 `53/100` 后停止，任务为 `pick_cool_then_place_in_recep-Bread-None-CounterTop-15/trial_T20190909_085557_322273`。真正异常是 decision 生成了代码围栏开头 ` ```plaintext `，随后本地格式校验抛出 `ActionFormatError`；不是本次请求的连接或超时错误。换行停止令这次回复只包含围栏，无法从未生成的后续行提取命令。缓存保存了这条回复，同样配置直接重启仍可能在同一处停止。

`interactive_runtime.py` 现在对这类模型输出打印 `interactive_action_format_warning`，原样返回给原生 MacNet。solver 输出进入原有协作；decision 输出经原有 `process_action` 进入环境，不合法时计入无效动作和步数，再获得环境反馈。持续无效可达到步数上限，作为真实失败任务保存；不额外重试直到成功、不跳过任务、不添加替代动作。提示词、温度、种子、停止参数、记忆检索和成功指标不变。连接异常、HTTP 错误和空回复仍中止并保留原因链。

为复用 v9 已提交结果，新增独立命令 `migrate_interactive_format_support.py`。它仅接受已知 v9 runtime 的 SHA256，校验除本次 runtime 修复以外的全部已登记源码、任务文件、折 manifest、连续提交标记、episode 和冻结记忆哈希；拒绝已经存在诊断/训练/测评产物的源数据集。v9 任何节点的格式错误都会使整个 episode 无法提交，所以已提交的前缀不含被新处理改变的路径，可复用。新目录记录旧 manifest/提交标记、迁移原因和新旧配置摘要；episode 与记忆内容保持相同，模型请求缓存使用 SQLite backup 复制。只重绑定新目录与已知 runtime 代码身份，旧目录不修改，也不绕过普通入口的配置校验。

迁移需要源和目标实验锁均可获取，目标必须是不存在的独立目录。没有 `--apply` 时仅校验，可能创建锁文件但不创建目标结果目录、不调用模型。复制先写入临时目录，全部校验后再重命名为目标；复制失败会清理本次临时目录。未提交的失败 attempt 留在原目录，恢复时从最后一次成功提交的记忆状态重新执行该任务。对服务器日志中的进度，预期可复用 5 折共 40 个 PDDL support 和 53 个 ALFWorld support；实际数量以迁移输出为准。

本地只提交本次六个文件，再手动 push：

```bash
git add causal_memory_control/interactive_runtime.py causal_memory_control/migrate_interactive_format_support.py causal_memory_control/tests/test_interactive_action_output.py causal_memory_control/tests/test_interactive_format_migration.py causal_memory_control/interactive_v8_upload_files.txt causal_memory_control/PDDL_ALFWORLD_EXPANDED.md
git commit -m "Keep malformed interactive actions within episode budgets and migrate support"
git push origin main
```

服务器逐条执行。先确认旧实验入口没有实际 PID，再拉取代码；如果服务器有未提交源码改动，先保留核对，不强制覆盖：

```bash
cd /mnt/mydata/hjq/casual-gmemory
pgrep -af 'causal_memory_control\.(run_interactive_gate_benchmark|interactive_worker|interactive_envs)'
git pull --ff-only origin main
python -X faulthandler -m unittest causal_memory_control.tests.test_interactive_action_output causal_memory_control.tests.test_interactive_request_errors causal_memory_control.tests.test_interactive_format_migration causal_memory_control.tests.test_interactive_no_ce causal_memory_control.tests.test_interactive_progress -q
export CMC_INTERACTIVE_OUTPUT=/mnt/mydata/hjq/casual-gmemory/causal_memory_control/results/pddl_alfworld_gate_expanded_v10_formatwarn_no_ce
python -m causal_memory_control.migrate_interactive_format_support --source /mnt/mydata/hjq/casual-gmemory/causal_memory_control/results/pddl_alfworld_gate_expanded_v9_actionfix_no_ce --output-dir "$CMC_INTERACTIVE_OUTPUT" --apply
bash causal_memory_control/run_interactive_expanded_v8.sh smoke
```

迁移及 smoke 成功后，在同一个终端用相同 endpoint/GPU 继续 all。会跳过已提交的 PDDL/ALFWorld support，从 ALFWorld 未完成任务继续；阶段日志仍直接显示在当前终端：

```bash
set -o pipefail
bash causal_memory_control/run_interactive_expanded_v8.sh all 2>&1 | tee -a /mnt/mydata/hjq/pddl_alfworld_v10_formatwarn_no_ce.launch.log
```

无需删除 v9、手工编辑 manifest、清空模型缓存或改随机种子。若迁移校验拒绝，保留错误以便核对，不绕过哈希检查。新流程仍可能有模型任务失败；格式 warning 本身不再导致整个实验退出。

本地针对性回归共 25 项通过：动作/请求错误 7 项、迁移 6 项、八策略/终端进度及原生回放/事务恢复/采集测评 12 项。迁移测试包含源文件内容不变、模型回复缓存保留、仅补未完成任务、已有冻结快照可恢复、拒绝未知源码或篡改产物、运行锁、复制失败清理，以及迁移后的配置能被正式 CLI 接受。ALFWorld 格式回归使用假 TextWorld 反馈验证原生 MacNet/适配器行为；未调用服务器 Qwen 或运行远程实验。

### 暂停 PDDL，保留 v10 全部进度并只续跑 ALFWorld

若某个 PDDL 折的 support 缺少成功记忆，诊断采集会报 `no diagnostic candidates`。这不要求删除结果或重新执行已完成的 ALFWorld support。新增 `resume_interactive_dataset.py` 读取现有根目录的原始 manifest，执行其中已注册的 ALFWorld 阶段；不重建 split、不缩减 manifest 的 experiments、不重绑定任何已提交结果摘要。直接给原主入口改 `--datasets alfworld` 会改变配置身份，不能用于当前混合目录的续跑。

新文件不属于被登记的 `interactive_*.py` 或原主入口；本次仅增加独立调度与报告工具，原采样、support、诊断、训练、测评源码保持不变。因此不用再次迁移 v10，也不会让已有测量的源码指纹失配。每个 worker 仍核对原 manifest 的源码、任务文件和服务模型身份。

真正执行前获取与原启动器相同的根目录 sidecar 锁，先完整复制当前 v10 目录到同级 `.backup-before-alfworld-日期时间-随机串` 目录，保留 PDDL 各折、ALFWorld、所有缓存、日志、manifest 和进度文件。复制有文件计数进度条，磁盘空间不足或复制失败时不启动实验，源结果保持原样。后续同命令重启默认复用首次备份；可用 `--backup-dir` 显式创建额外备份，目标必须是不存在的独立路径。备份含 SQLite/WAL 原文件，只有在停止旧实验并取得排他锁后复制；不应直接在备份路径运行原配置，因为原 manifest 登记的是 v10 路径。

ALFWorld 按 `support → collect → labels_team_u → train_reward → train_team_u → evaluate → report` 继续，保持八策略且跳过 CE。support 已完整冻结时直接复用，部分提交则从未完成任务继续；诊断/测评复用原有提交和动作缓存，未提交任务从原任务边界补跑。PDDL 各折及根 `pipeline_progress.json` 原样保留，ALFWorld 单独调度进度保存在 `dataset_resume/alfworld/progress.json`，逐阶段日志继续追加到 `alfworld/logs/` 并实时输出。

原联合 report 会要求 PDDL 测评完整，因此单独生成 `alfworld/dataset_report/comparison.json`、`.csv`、`.md`，保留根联合结果表。此报告使用原混合配置摘要，校验策略/任务完整覆盖、配对身份和 no-DROP 一致性，复用原 scene-cluster bootstrap 统计函数。134 个 valid_unseen 为主表，500 个 train_scene_holdout 分开报告，不合并为官方 unseen 成绩。

本地仅上传本次四个文件，命令逐条执行：

```bash
git add causal_memory_control/resume_interactive_dataset.py causal_memory_control/tests/test_resume_interactive_dataset.py causal_memory_control/interactive_v8_upload_files.txt causal_memory_control/PDDL_ALFWORLD_EXPANDED.md
git commit -m "Resume ALFWorld alone with full experiment backup"
git push origin main
```

服务器先确认旧实验入口没有实际 PID；不要并发操作结果目录，再拉取。源码若有未提交修改，先核对保留，不强制覆盖：

```bash
cd /mnt/mydata/hjq/casual-gmemory
pgrep -af 'causal_memory_control\.(run_interactive_gate_benchmark|interactive_worker|interactive_envs|resume_interactive_dataset)'
git pull --ff-only origin main
python -m unittest causal_memory_control.tests.test_resume_interactive_dataset -q
```

可先只读检查执行计划，不备份、不写结果、不调用模型/服务：

```bash
python -m causal_memory_control.resume_interactive_dataset --output-dir /mnt/mydata/hjq/casual-gmemory/causal_memory_control/results/pddl_alfworld_gate_expanded_v10_formatwarn_no_ce --dataset alfworld --plan-only
```

正式只续跑 ALFWorld（仍在同一个 v10 根目录，自动备份，不用 smoke 或重新采样）：

```bash
CUDA_VISIBLE_DEVICES="${CMC_INTERACTIVE_GPU:-0}" python -u -X faulthandler -m causal_memory_control.resume_interactive_dataset --output-dir /mnt/mydata/hjq/casual-gmemory/causal_memory_control/results/pddl_alfworld_gate_expanded_v10_formatwarn_no_ce --dataset alfworld --stage all
```

终端出现 `[备份完成]` 后才继续校验服务及执行阶段。以后中断仍用同一命令续跑，不重复迁移、不删除 PDDL、也不回到会先跑 PDDL 的原 `all` 启动器。当前只暂停 PDDL，空候选原因仍留待单独排查；ALFWorld 自身若也有空候选或请求错误仍会如实停止，不跳过失败任务。

### fold_02 全失败：动作检查、原生冷启动与空记忆边界

服务器检查显示 fold_02 的 8 个 support 均 success=0、30 步耗尽，累计 240 步有 199 步无效（82.9%），insight_count=0。这确认该折没有成功 support 来源及已保存 insights，但还不能确定为何动作失败。特别是高无效比例可能来自标题/代码围栏、对象或参数顺序、动作前置条件不满足、重复动作或观测处理等；必须对照实际轨迹，不能直接当作纯粹的规划难度或步数不够。有效动作也不保证目标推进，合法性与规划质量需分开看。

核对本地原项目 upstream/main（d9b5aef）的 tasks/run.py、MacNet 与 GMemory：常规入口创建/加载持久库，连续跑任务，每题结束追加成功或失败交互；空成功列表/空 insights 直接作为空列表进入提示词，仍可凭任务说明、静态 few-shot 和环境反馈作答，没有单独保证成功轨迹的 init 命令。init_task_context 只建立当前题的上下文，不会创建成功经验。当前扩大实验 support 已承担前置建库作用，PDDL 各域规则及默认 1 条静态 few-shot 也已提供。

原实现的 insights 默认每累计 5 条任务记录触发提炼；源码 _finetune_insights 的生成来源是成功任务分块和成功/失败对比。全失败时，这两种来源都为空，不能指望失败轨迹自动形成新规则。失败交互本身可以入库，但当前 MacNet 取 successful_trajectories, _, insights，忽略返回的失败轨迹，常规入口默认 failed_topk=0；单独提高 failed_topk 并不能让 MacNet 自动使用失败经验。

与原在线连续积累不同，当前公平配对实验只在 support 写库，随后冻结，每个 PDDL 折独立；fit/dev/test 的试跑不会补充记忆。这一隔离是为了共享同一记忆、避免标签/测试泄漏，但小 support 加低成功率会导致不可训练的折。no diagnostic candidates 是新增诊断环节对缺少标注对象的限制，不是原生 GMemory 遇空记忆就无法运行。

空记忆任务可以采用原生无检索分支；若把这种处理纳入正式比较，应保留测试分母，各策略的记忆选择在该任务上等价，不伪造成功率。没有候选不等于某条记忆的 U=0；fit/dev 没有任何候选则没有逐条监督，不能声称训练好了 gate。仅删除 collect 的检查仍会被后续训练的候选/覆盖检查拒绝，需要设计并记录无监督折的处理，而不能静默跳过。

可选的预热应另立 PDDL 协议：在当前折已指定 support 上预先固定多个 rollout 种子/预算，或导入该 support 的可验证专家轨迹并明确标记专家初始化；均不使用 fit/dev/test 目标轨迹、不拼接其他折记忆，也不事后反复抽 seed 直到成功。扩充 support 需要重新登记分区；这些都改变记忆分布，应使用独立目录，不能追写已冻结 v10。当前只加入只读检查工具，未改动任何采样、空候选逻辑、训练或 ALFWorld 运行代码。

工具先校验 commit 对应的 episode 哈希，只读已提交的 JSON，不打开 SQLite/Chroma、不反序列化 pickle、不加载环境/模型。默认每题展示前 8 步、初态和最常见无效动作；--steps 0 展示全部步数，--task-id 可多次指定。trajectory.action 是送入环境的处理后动作，不等同于模型完整原回复。--transcript-tail 展示已记录 observer 末尾文本，也不冒充原始 API 回复。

可直接使用已有代码的单行命令查看两个 Gripper 任务的完整动作，无需更新服务器，不会影响正在运行的 ALFWorld。在服务器项目根目录执行：

~~~bash
python -c 'import json; from pathlib import Path; r=Path("causal_memory_control/results/pddl_alfworld_gate_expanded_v10_formatwarn_no_ce/pddl_folds/fold_02/pddl"); rows=[json.loads((r/json.loads(p.read_text())["directory"]).parent.joinpath("episode.json").read_text()) for p in sorted((r/"support/committed").glob("*.json"))]; [(print("TASK=",x["task_id"],"INITIAL=",x.get("state")),[print(json.dumps(s,ensure_ascii=False)) for s in x["trajectory"]]) for x in rows if x["task_id"].startswith("pddl:gripper:")]'
~~~

下次停止实验后同步工具（无需为此中断已运行的 ALFWorld），本地仅提交这些文件再 push，服务器 pull：

~~~bash
git add causal_memory_control/inspect_interactive_support.py causal_memory_control/tests/test_inspect_interactive_support.py causal_memory_control/interactive_v8_upload_files.txt causal_memory_control/PDDL_ALFWORLD_EXPANDED.md
git commit -m "Add read-only support action inspection and document cold-start behavior"
git push origin main
~~~

服务器模块命令（项目根目录、gmemory 环境）：

~~~bash
python -m causal_memory_control.inspect_interactive_support --root causal_memory_control/results/pddl_alfworld_gate_expanded_v10_formatwarn_no_ce/pddl_folds/fold_02/pddl --task-id pddl:gripper:prob03 --task-id pddl:gripper:prob20 --steps 0
~~~

### 空记忆兼容修复与原目录续跑（2026-10-31）

本节替代此前“空候选会停止”“仅加入检查工具”的描述。collect 现在允许空候选：每个预声明种子的无记忆 KEEP 基线照常执行和保存，无 SINGLE DROP 分支，q_drop/drop_samples/branches 都为空，不把“无候选”伪造成 U=0。support 后仍然冻结，fit/dev/test 不追加记忆，不更改种子、动作解析、LLM 提示词、任务划分或 CE 排除设置。

混合列表保留所有诊断任务，仅真实候选产生训练样本，checkpoint 额外记录 candidate_coverage，区分注册任务、实际监督任务和空任务。若整折 fit/dev 都没有候选，三个 Reward Gate 和 Team U 写 training_status.json，标记 skipped_no_candidate_supervision、trained=false，不创建 checkpoint，也不加载训练编码器。测试任务仍实际执行，同任务同种子的八策略复用无记忆结果，计入原测试分母。报表记录 empty_candidate_tasks、empty_candidate_rate、untrained_empty_only_tasks 和 untrained_methods；空记忆上的相同分数不能作为 gate 训练效果的证据。

边界仍如实报错：如果 fit/dev 只有一侧存在监督、Reward Gate 真实样本少于原最小训练量，或未训练 gate 遇到非空测试候选，程序停止并说明原因；不借用其他折、测试集或伪造模型。完整五折 OOF 汇总仍要求每个注册问题恰好测一次。

旧 v10 manifest 包含源码哈希，因此上传后不要直接用旧 shell all 命令覆盖校验。新增 continue_interactive_empty_memory 仅接受已核对的 v10 四个阶段文件升级，其他已登记源码和数据仍须一致。它先取得同一实验锁，完整备份结果目录，再在根目录和各折写独立 empty_memory_upgrade/registration.json；保留原 manifest/config digest、support、已测分支、checkpoint 和模型回复缓存。报告另存升级登记的哈希；未完成任务继续跑，已提交任务和已测分支复用。备份失败或实验仍在运行时不会启动 worker；不删除旧目录。

以下命令由用户手动执行。正在跑的 ALFWorld 可先继续；准备更新服务器代码时，先等其结束，或在其前台终端 Ctrl+C 并等待退出，再 pull，避免运行过程中混用源码。

本地提交本次文件（保留已有其他改动；不要 git add .）：

~~~bash
git add causal_memory_control/interactive_empty_memory.py causal_memory_control/continue_interactive_empty_memory.py causal_memory_control/interactive_worker.py causal_memory_control/interactive_training.py causal_memory_control/interactive_team_u.py causal_memory_control/interactive_report.py causal_memory_control/resume_interactive_dataset.py causal_memory_control/tests/test_interactive_empty_memory.py causal_memory_control/tests/test_continue_interactive_empty_memory.py causal_memory_control/PDDL_ALFWORLD_EXPANDED.md causal_memory_control/interactive_v8_upload_files.txt causal_memory_control/inspect_interactive_support.py causal_memory_control/tests/test_inspect_interactive_support.py
~~~

~~~bash
git commit -m "Allow empty memory throughout interactive benchmark and preserve v10 progress"
~~~

~~~bash
git push origin main
~~~

服务器项目根目录（gmemory 环境），旧实验退出后逐条执行：

~~~bash
git pull --ff-only origin main
~~~

~~~bash
python -m unittest causal_memory_control.tests.test_interactive_empty_memory causal_memory_control.tests.test_continue_interactive_empty_memory causal_memory_control.tests.test_interactive_no_ce causal_memory_control.tests.test_resume_interactive_dataset causal_memory_control.tests.test_interactive_progress -q
~~~

先只读核对升级计划（不备份、不写结果、不调用模型或服务）：

~~~bash
python -m causal_memory_control.continue_interactive_empty_memory --output-dir /mnt/mydata/hjq/casual-gmemory/causal_memory_control/results/pddl_alfworld_gate_expanded_v10_formatwarn_no_ce --datasets pddl --plan-only
~~~

只续跑 PDDL，其余 ALFWorld 结果保留（GPU 0 可按你的分配修改）：

~~~bash
CUDA_VISIBLE_DEVICES=0 python -u -X faulthandler -m causal_memory_control.continue_interactive_empty_memory --output-dir /mnt/mydata/hjq/casual-gmemory/causal_memory_control/results/pddl_alfworld_gate_expanded_v10_formatwarn_no_ce --datasets pddl --stage all
~~~

如果仍先跑 ALFWorld，把 --datasets pddl 改为 --datasets alfworld；两个都接着跑用 --datasets pddl,alfworld。所有情况都沿用相同 v10 根目录，不用新建 v11，不删失败折，不重复 smoke。中断后原命令可再次执行，完整备份只做一次。仅选择一个数据集时，表格输出到 empty_memory_upgrade/report_pddl 或 report_alfworld，不覆盖根目录的合并表；全部选择时才生成完整合并表。

终端会显示“空记忆……保存无记忆基线”“候选监督为 0，记录未训练；不创建 checkpoint”等明确状态，阶段进度条继续显示。此修复处理空候选流程，不会自动修正 Gripper 轨迹中机器人房间、持球状态、think 输出及动作前置条件推理错误。
