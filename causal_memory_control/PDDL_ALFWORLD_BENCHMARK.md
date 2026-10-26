# PDDL / ALFWorld 八策略实验

本文保留小规模 v1 协议。用户随后选择的 PDDL 五折与 ALFWorld 扩大版见 `PDDL_ALFWORLD_EXPANDED.md`，新版应使用独立的 v2 输出目录。

本入口承接 `PDDL_ALFWORLD_HANDOFF.md`，2026-10-23 已由用户确认固定留出规模和成功计划 CE proxy。旧环境、旧 ST、CSM、FEVER/HotpotQA 入口、checkpoint 和结果均未修改。新增入口只接受 `pddl,alfworld`，不会运行旧数据集。

## 已核查的数据与指标

| 数据集 | support | fit | dev | test | 分组与范围 |
| --- | ---: | ---: | ---: | ---: | --- |
| PDDL | 8 | 28 | 12 | 12 | 当前注册的 60 个问题内，按 domain 分层、problem 内容分组固定留出 |
| ALFWorld | 50 | 80 | 20 | 134 | support/fit/dev 来自 train，按 FloorPlan 场景隔离；test 为原 134 个 valid_unseen 游戏 |

split seed=71。PDDL 每个 domain support=2；20 题域各 fit/dev/test=10/4/4，10 题域各 4/2/2。稳定 ID 包含 domain + problem 文件名，分组内容去掉 PDDL problem 名字后计算哈希。ALFWorld 先将训练场景分配到 support/fit/dev，再在每区内按 task type 轮流抽样；未抽中的训练游戏保持未使用。测试场景不能出现在任何训练分区。各候选、repeat、反事实分支随整个任务划分。

本地核查发现旧 PDDL loader 按 barman 开始附 difficulty，但标注文件从 gripper 开始。新 inventory 按 `additional_info.subtask` 对齐域，域内沿用原生 lexicographic 文件顺序（如 problem1、problem10、problem2），执行时再逐目标比对标注。本地全部 60 题均已通过目标、原生索引和重复 reset 核查。PDDL 结果只能称为“12 题固定留出 Success”，不能称为完整官方测试成绩。

ALFWorld 本地 train 有 4,652 个 `game.tw-pddl`；按本入口六类 task type 与 `solvable=true` 筛选后，可用训练任务为 3,553 个（1,096 个未标为可解、3 个类型不支持）。valid_seen 为 187 个文件 / 140 个有效游戏；valid_unseen 为 195 / 134，原 134 个测试路径全部存在。另有未使用的 valid_train 200 个游戏。四个目录按文件筛选合计 4,027 个有效游戏条目；本次只使用 train 的 150 个加 valid_unseen 的 134 个，共 284 个任务。

`traj_data.json` 的 Thor 动作并未直接冒充文本专家动作。新环境直接打开自带 domain/grammar 的 TextWorld 游戏，使用原生 AlfredDemangler，关闭 domain randomization；无需不存在的外部 logic、detectors 或视觉权重。本地已回放成功的手写专家样例包括 19 步训练游戏，以及本次选中 fit 中的 12 步游戏。这不是全部 100 个监督任务已验证的声明。

两个新列均以显式 `won` 定义 Success，标签在 0..1；`done`、失败终止、step limit 分开保存。PDDL 另存原生历史最大 goal progress，避免把它当二分类准确率；单步无效动作的 -1 不进入 outcome 训练标签。主表为按任务数加权微平均，同时保存 domain、difficulty 或 task type 的分组及宏平均。

## 策略与监督

八行依次为 Always Use、DROP ALL、Random Drop、Semantic Linear、三头 Semantic MLP、旧单头 Semantic ST、CSM-ST Local–Global、Oracle。

- 原生 retrieval 每个 episode 只发生一次，默认 3 条成功轨迹 + 3 条 insight，failed=0；过滤作用于整个团队。没有检索 memory 时，测试控制组退化为 KEEP；诊断没有候选则停止，不能虚构训练样本。
- Semantic Linear/MLP/ST 复用原类，在新任务上训练独立 checkpoint。监督为同题同候选集合下 `U=mean(Success_KEEP)-mean(Success_SINGLE_DROP)`。fit/dev 固定，不重新随机划分。ensemble=12，训练 seed=23；ST beta=.01、10 epochs，MLP 三头 128→64、80 epochs。dev 从 kappa=0/.5/1/1.96 和 KEEP ALL 中选择，最多删一条。
- Random 每题 SINGLE DROP 一条，预声明种子 11/23/47，生成 sample seed 保持 2000；表中报告三个随机策略成绩的平均，另存各 seed 成绩与标准差。它不匹配 CSM 的无限删除预算。
- CSM 使用可训练 BERT、3 层/8 头列表 self-attention 和无界单 CI 头；旧 CSM 文件不变。默认 seed=2024、beta=.1、10 epochs、batch=16；语义去重、平凡列表下采样、仅 fit 内重组及重新 CE 标注沿用既有实现。dev 只选 epoch，不搜索测试阈值。无有效 hard-list 对时会报告 InfoNCE 未激活。
- Oracle 固定只搜索 KEEP ALL、各 SINGLE DROP、DROP ALL，以 Success 选择，平分按 KEEP、候选顺序、DROP ALL 顺序保守取首。它不包含任意 CSM 多条删除子集，不是全部子集或期望性能的上界。

CSM 的监督定义已由用户确认：从公开初始任务和初始观察，teacher-force 一整段成功原生动作计划。PDDL 用本地 Fast Downward 符号规划，转为原生动作文本；ALFWorld 用已有 handcoded 文本专家。每个计划先完整回放，必须成功且无非法动作。专家只允许 fit/dev；test 不生成专家计划。完整运行首先执行便宜的专家验证阶段，失败会在任何 support/诊断 LLM 采集前停止。

```text
CI_i = CE(成功动作计划 | 公开初态, memories_without_i)
       - CE(成功动作计划 | 公开初态, all_memories)
```

每列表只算一次 CE_ALL，逐条 SINGLE DROP，不乘诊断 repeat。仅 continuation tokens 算 CE，包含 EOS，mean reduction，无静默截断；一题一个验证成功计划，不扩展别名或穷举多解。专家观察/隐藏事实和动作序列不进入 gate 输入；模型运行时只看 query、公开初始 observation 和候选文本。重组列表沿用 anchor 的验证计划并重算 CE。

这是单冻结 HF Qwen 的完整计划 CE proxy。它不同于逐步观察条件下的动作 CE、多 agent 联合轨迹似然和实际团队 Success；HF 与 Ollama 量化概率也可能不同。完整计划长度可能超过团队 30 步预算，仍须标注为专家监督 proxy。CE 超窗会明确失败；BERT 的 512-token pair 截断则是已有架构限制，会记录在训练日志。若 pilot 显示步数预算造成几乎全零 Success，应在正式采集前用新输出目录预声明新的 `--max-steps`，不能看 test 后修改。

## 规模、恢复和日志

默认两次诊断 repeat（sample seed=0/1）、最终 seed=2000、temperature=.7、MacNet Chain 的 3 solver + decision、无 critic、原生 1 few-shot、每 episode 最多 30 步。few-shot 与任务指令保留原生内容，所有策略相同；gate 不控制它们。

| 项目 | PDDL | ALFWorld | 合计 |
| --- | ---: | ---: | ---: |
| support 团队运行 | 8 | 50 | 58 |
| 诊断逻辑 USE/DROP 行数上限 | 960 | 2,400 | 3,360 |
| 共享 KEEP 后诊断 schedule 上限 | 560 | 1,400 | 1,960 |
| 测试共享动作 schedule 上限 | 108 | 1,206 | 1,314 |
| 专家监督任务 | 40 | 100 | 140 |
| 原列表 CE forward 上限 | 280 | 700 | 980 |
| fit 重组 CE forward 上限 | 392 | 1,120 | 1,512 |

测试每题最多 KEEP+6 SINGLE+ALL+一个新 CSM 子集。其余 gate、Random 和 Oracle 引用同动作 outcome。实际候选不足 6 时更少。每个 schedule 包含多步、多 agent 和多次 LLM 请求，以上不是 API 次数或工时保证。smoke 额外为每数据集 2 次团队运行（baseline 和 no-DROP replay），使用一个 fit 游戏与空 memory；正式冻结库上的候选覆盖还会在 collection 检查。

所有新结果写入独立根目录。support 按 episode 事务提交完整快照，中断从上一份成功提交恢复，失败尝试留存。评测器复制一致的 SQLite/向量快照到私有运行目录后只读加载，不让 Chroma 的数据库记账改变冻结原件。

分支 outcome 的唯一键包括任务、生成 seed、初态、原始 retrieval fingerprint、删除集合；JSON 原子落盘。SQLite 回答缓存由实际 prompt、模型、参数和 seed 共同索引。绑定并检查 Ollama digest、模板/参数摘要、代码及数据哈希和包版本。遇到不同配置、变更快照或冲突 outcome 会停止。一个输出根目录只运行一个写入进程。

新 episode 清空节点 buffers、memory task context、insights cache 并重置 RNG/环境；题内交互历史保留。schedule 内第二次 reset 必须与前一次初态一致。所有反事实必须匹配原始检索列表。no-DROP 必须复用同一 outcome；smoke 另外实际重放并比对动作、观察、Success/done。失败不会从分母中静默剔除。

主要文件（路径相对 output 根目录）：

```text
benchmark_manifest.json        完整 split、具体任务 ID、数据/代码/版本/模型 identity
plan.json                     规模、协议与缺失输入
pipeline_progress.json
{pddl,alfworld}/
  snapshot.json               冻结库路径、support ID、文件哈希
  support/committed/*.json    可恢复的 support 事务
  diagnostic/actions/parts/  原始 KEEP/SINGLE DROP 完整轨迹与调用统计
  diagnostic/tasks/*.json    repeats、q_use/q_drop、分支引用、原始候选
  experts/*.json             成功计划、来源、验证轨迹、初态 fingerprint
  {semantic_linear,semantic_mlp,semantic_st}/checkpoint.json
  {semantic_linear,semantic_mlp,semantic_st}/training_summary.json
  {semantic_linear,semantic_mlp,semantic_st}/loss_history.json
  csm_st/                    CE 缓存/逐列表进度、labels manifest、BERT checkpoint、loss history
  evaluation/actions/parts/  唯一实测动作的环境和 agent 全轨迹
  evaluation/policy_parts/   可恢复逐任务策略决策（Oracle 含候选动作成绩）
  evaluation/policies.jsonl
  comparison.json            配对变化、95% CI、改善/伤害 ID、分组、保留率、来源哈希
  responses.sqlite
  progress.json
  errors/*.json              阶段、命令、traceback
  logs/{stage}.log
matrix_results.{md,csv,json}  四数据集八行表，缺失保持为空
```

线性/MLP 旧 trainer 没有逐 epoch loss 输出，新的 loss_history 文件明确记为不可用；ST 和 CSM 保存实际 loss history。不能将没有记录的曲线补造出来。CI 为 2,000 次固定种子的配对 cluster percentile bootstrap：PDDL 按 problem，ALFWorld 按 FloorPlan；12 题 PDDL 的区间和泛化证据都有限。

## 上传和服务器命令

只需上传以下新增源码和文档，保留目录结构；不上传本地 data、旧 results、figures 或模型：

```text
causal_memory_control/run_interactive_gate_benchmark.py
causal_memory_control/interactive_data.py
causal_memory_control/interactive_envs.py
causal_memory_control/interactive_runtime.py
causal_memory_control/interactive_training.py
causal_memory_control/interactive_worker.py
causal_memory_control/interactive_report.py
causal_memory_control/interactive_prior_results.py
causal_memory_control/tests/test_interactive_benchmark.py
causal_memory_control/PDDL_ALFWORLD_BENCHMARK.md
```

服务器地址没有可靠记录，因此不提供猜测的 scp 主机。先在服务器已有仓库执行只读检查（服务器无 rg，使用 grep）：

```bash
cd /mnt/mydata/hjq/casual-gmemory
conda activate gmemory

nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv
ss -ltnp | grep -E ':(11434|11436|11438|11440|11442)([[:space:]]|$)'
python -m pip check
python -m causal_memory_control.run_interactive_gate_benchmark --plan-only
```

plan-only 不访问服务、不初始化模型、不写目录、不下载。模型参数是名字时，`local_model_paths_exist=false` 仅表示不是本地目录，不代表 HF 离线缓存一定缺失。实际运行强制 HF offline；缺权重就明确失败。已有 GMemory 依赖见根 requirements.txt，CSM 附加依赖见 requirements_csm_st.txt；规划器包为 TextWorld 配套 `fast-downward-textworld`，只在确认缺失时处理依赖，不重复覆盖整个 GPU 环境。

以下使用曾建议的 HF Qwen 路径，须先确认实际存在。BERT/MiniLM 可使用已有离线缓存的名字，也可替换为本机实际目录。参数必须在第一次运行前确定，后续同输出目录保持相同：

```bash
GENERATOR_DIR=/mnt/mydata/hjq/ollama/models/huggingface/Qwen2.5-7B-Instruct
test -f "$GENERATOR_DIR/config.json"
find "$GENERATOR_DIR" -maxdepth 1 -type f -name '*.safetensors' -print

benchmark=(
  python -u -m causal_memory_control.run_interactive_gate_benchmark
  --output-dir causal_memory_control/results/pddl_alfworld_gate_v1
  --datasets pddl,alfworld
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

CUDA_VISIBLE_DEVICES=5 "${benchmark[@]}" --plan-only
```

`CUDA_VISIBLE_DEVICES=5` 后 Python 设备是 `cuda:0`。它不移动任何已运行的 Ollama。只有确认 GPU 5 和 11442 均可用时，用户才在独立终端手动启动专属服务；不终止或接管旧服务：

```bash
CUDA_VISIBLE_DEVICES=5 \
OLLAMA_HOST=127.0.0.1:11442 \
OLLAMA_MODELS=/mnt/mydata/hjq/ollama/models \
OLLAMA_NUM_PARALLEL=1 \
ollama serve
```

已有合适服务时，将 benchmark endpoint 改为其真实地址即可，不需要另起服务。smoke 是新实验的小规模实际调用，不是只读操作：

```bash
CUDA_VISIBLE_DEVICES=5 "${benchmark[@]}" --stage smoke
cat causal_memory_control/results/pddl_alfworld_gate_v1/pddl/smoke/summary.json
cat causal_memory_control/results/pddl_alfworld_gate_v1/alfworld/smoke/summary.json
```

两者通过后，正式命令按阶段执行。专家全体验证先于 LLM 数据采集；每条命令仅恢复/执行相应阶段：

```bash
CUDA_VISIBLE_DEVICES=5 "${benchmark[@]}" --stage experts
CUDA_VISIBLE_DEVICES=5 "${benchmark[@]}" --stage support
CUDA_VISIBLE_DEVICES=5 "${benchmark[@]}" --stage collect
CUDA_VISIBLE_DEVICES=5 "${benchmark[@]}" --stage labels
CUDA_VISIBLE_DEVICES=5 "${benchmark[@]}" --stage train_reward
CUDA_VISIBLE_DEVICES=5 "${benchmark[@]}" --stage train_csm
CUDA_VISIBLE_DEVICES=5 "${benchmark[@]}" --stage evaluate
CUDA_VISIBLE_DEVICES=5 "${benchmark[@]}" --stage report
```

也支持用户自行选择 `--stage all`；本次没有执行。只有一张空闲卡时，CE 和 Ollama 各自的 7B 副本会竞争显存；各阶段不并行，CE/训练子进程结束后释放自身显存。仍须在 labels 前核查可用显存，不能依据 GPU 可见性推定显存足够，也不自动停止旧进程。

只追加报告、只读复用已有 QA 结果：

```bash
python -m causal_memory_control.interactive_report \
  --manifest causal_memory_control/results/pddl_alfworld_gate_v1/benchmark_manifest.json \
  --qa-results-root causal_memory_control/results/latest_heldout_oracle_v1
```

若最新 Oracle 尚未完成，可把 qa-results-root 指向 `all_trained_gates_with_old_st_retest_v2`，Oracle/Random 缺失保留为空。导入重新从日志计算、校验任务集合、task isolation、原始检索和 no-DROP parity，记录源日志/checkpoint metadata/seed/哈希；不会调用 QA 模型。Oracle 导入要求 action_space 明确包含 DROP ALL，并核对与来源 Global Drop 的关系。截图成绩不作为输入。

## 验证与当前边界

本地已通过：新增 14 项测试（包括真实原生 MacNet 配假 LLM、support 事务恢复、真实 PDDL 规划回放、CE 符号、划分、防泄漏、缓存/动作冲突、八策略编排与只读 QA 导入），旧入口 33 项回归测试；全部 60 个 PDDL 问题 reset/目标核查；两个 ALFWorld 专家样例。测试和这些 CPU 环境检查没有调用付费 API。

```bash
gmemory_env/bin/python -m unittest causal_memory_control.tests.test_interactive_benchmark -v
gmemory_env/bin/python -m unittest \
  causal_memory_control.tests.test_oracle_evaluation \
  causal_memory_control.tests.test_trained_gate_evaluation \
  causal_memory_control.tests.test_full_gate_benchmark \
  causal_memory_control.tests.test_dataset_matrix \
  causal_memory_control.tests.test_task_state \
  causal_memory_control.tests.test_collection_runtime -q
```

服务器安装、所有专家计划、实际 HF CE 和 GPU 训练、冻结库上的完整采集/评测仍须按上面阶段执行；本地没有新成绩可填写。审阅打包优先包含 manifests、progress/errors、MD/CSV/JSON 表、comparison、training_summary/loss_history 和必要策略 JSONL；不要默认打包模型权重、SQLite、runtime_memory、support attempts 或全部环境资源。
