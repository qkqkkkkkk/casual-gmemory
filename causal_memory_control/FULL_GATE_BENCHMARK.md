# 全量数据、双数据集、七策略对比

入口：`python -m causal_memory_control.run_full_gate_benchmark`。
这是新的实验编排，不修改现有 Linear、Semantic Linear、Semantic MLP 的模型、
特征、loss、默认训练参数或原 GMemory/MAS 流程。Jev 不启用。

## 七行表格的含义

| Policy | 行为 |
| --- | --- |
| Always Use (baseline) | 保留本题全部原始检索 memory |
| Always Global Drop | 默认清空本题全部检索 memory，所有 Agent 均不接收；不删除数据库记录 |
| Random Drop | 默认每题从全部候选中均匀随机删除一条，对全部 Agent 隐藏；任务级固定随机种子 |
| Linear Gate | 现有词汇/结构特征 + 线性效用估计器 + 开发集策略 |
| Semantic Linear Gate | 现有语义特征 + 线性效用估计器 + 开发集策略 |
| Semantic MLP | 现有语义特征 + 三头 MLP + 开发集策略 |
| Oracle Selective Drop | 在相同测试样本上，事后选择实测分数最高的动作；不可部署 |

Oracle 预先运行 KEEP、每个候选的单条删除、Global Drop；再取每题实际 reward 最大值。
三种 learned gate 和 Random 的实测动作也纳入一致性核对。相同动作必须有相同结果，
否则停止报错，不把随机重采样的好运当成 Oracle 收益。相同 reward 时优先 EM/done，
再优先 baseline。因此它是**有限次观察结果的事后包络**，不是期望效用的严格上界。

默认 Global Drop 清空全部，因此 Oracle 的动作空间包括 DROP ALL；它可以删除多条，
**不是与最多删一条的 learned gate 等预算比较**。报告中明确注明。
`--global-drop top1` 可改为每题固定删除 trajectory rank 1，global 指对所有 Agent 隐藏。

Random 默认不是匹配某个 gate 的删除率。若要比较同题同预算的候选选择能力，可加：
`--random-drop-reference semantic_mlp`（也支持 `linear`、`semantic_linear`）。
此时逐题匹配指定 gate 的零/一预算，再随机选择候选；若参考 gate 全 KEEP，Random
也全 KEEP。这个匹配版本不能检验 gate 的“哪些题该删”能力。选择不同定义需新目录。

## “全量”的范围及指标

使用当前输入文件 loader 接受的全部有效样本，互斥划分为：

1. 50 题构建冻结 memory；
2. 剩余 80% 做反事实诊断，其中 20% 留作策略校准；
3. 剩余 20% 做最终测试。

同一道题及其所有候选/采样分支不会跨训练、校准、测试。
这不是每道题同时参与训练和测试，也不是对官方全部 train/dev/test 文件的自动合并。
脚本新建独立产物，不混用以前 100 题 pilot 的汇总或 checkpoint；新划分可能与旧
pilot 的任务交叉，不能把这种扩展实验称为从未用于探索的盲测。

表格列定义固定：

- FEVER：SUPPORTS/REFUTES 二分类 Accuracy × 100%，排除 NEI，非官方完整 FEVER Score。
- HotpotQA：离线 distractor、上下文已提供的 Answer F1 × 100%；EM 在详细 JSON 中。

不增加在线 Wikipedia 检索，不改变原有任务定义。FEVER 全量模式要求当前文件的
SUPPORTS/REFUTES 数量相等；否则报错，不静默丢掉多数类。`--failed-topk` 必须为 0。

按此前文件：

| 数据集 | 有效题数 | Support | 诊断（拟合+校准） | 最终测试 | 两组采样的诊断团队分支 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FEVER | 6,666 | 50 | 5,292 | 1,324 | 254,016 |
| HotpotQA | 7,404 | 50 | 5,883 | 1,471 | 282,384 |

两者合计约 **536,400 个诊断团队分支**，一个团队分支可能包含多次 LLM 调用。
Oracle 另需每道测试题至多 6 个单候选分支；总量远大于 pilot。缓存会复用相同
prompt，但这不是运行时长承诺。以服务器 `--plan-only` 的真实文件统计为准。

## 服务器运行

先更新包含新脚本的代码，使用原环境和实际模型端点：

```bash
cd /mnt/mydata/hjq/casual-gmemory
conda activate gmemory
export OPENAI_API_KEY=ollama

python -u -m causal_memory_control.run_full_gate_benchmark \
  --fever-data data/fever/fever_dev.jsonl \
  --hotpotqa-data data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
  --endpoint http://127.0.0.1:11434/v1 \
  --model qwen2.5:7b \
  --output-dir causal_memory_control/results/full_gate_benchmark_v1 \
  --plan-only
```

计划检查不写结果、不创建 memory、不拟合模型、不调用 LLM。核对规模后，删除最后
的 `--plan-only` 再执行同一命令即可正式运行。默认先 FEVER，再 HotpotQA，自动跑完
七策略。`--datasets fever` 可只运行 FEVER；改变 datasets 或其他设置需新输出目录。

原共享参数也可传入，例如 `--training-fraction 0.8`、`--repeats 2`、
`--mlp-hidden-sizes 128,64`、`--mlp-device cpu`、`--kappa auto`。全量模式自动启用；
不可用 `--evaluation-count` 将它变成子样本试验。数据集路径用上述两个专属参数；
memory/诊断/checkpoint 路径和三种 gate arms 由脚本管理，避免串数据集。
API key 推荐只放环境变量；显式 `--api-key` 也会从子命令和配置记录中移除，通过环境传递。

## 产物

```text
full_gate_benchmark_v1/
├── benchmark_manifest.json       # 数据来源与全量实验身份；改变参数拒绝混跑
├── experiment_plan.json
├── benchmark_progress.json       # 总阶段、具体错误、traceback、命令及日志路径
├── policy_results.md             # 七行、FEVER/HotpotQA 两列的实测表格
├── policy_results.csv            # Excel 可读，百分制数值
├── policy_results.json           # 含 CI、keep rate、DROP 改善/伤害及 HotpotQA EM
├── policy_results.html           # 与示例布局相似的可浏览表格
├── policy_results.svg            # 矢量表格，可放论文/幻灯片
├── logs/                         # 每个数据集/策略/Oracle 分支的控制台输出与异常
├── artifacts/<dataset>/
│   ├── g-memory/                 # 冻结 snapshot + source/split provenance
│   └── diagnostic/               # 两组采集、branches.jsonl、原始 LLM cache 等
├── fever/                        # HotpotQA 使用相同结构
│   ├── matrix_manifest.json
│   ├── matrix_progress.json
│   ├── matrix_results.*
│   ├── checkpoints/{linear,semantic_linear,semantic_mlp}.json
│   ├── native_gmemory/
│   ├── linear/
│   ├── semantic_linear/
│   ├── semantic_mlp/
│   ├── always_global_drop/
│   ├── random_drop/
│   ├── plans/                    # 只含检索候选/预定删除 ID，不含 gold/reward
│   ├── oracle_branches/drop_000/ # 其余候选 drop_001 ...
│   ├── oracle_selective_drop/
│   │   ├── memory_gate.jsonl     # 每题选哪个分支及各分支分数、删除内容
│   │   ├── oracle_manifest.json  # 明确 uses_test_outcomes=true、deployable=false
│   │   ├── comparison.json
│   │   └── summary.json
│   └── cache_seeds/              # 固定合并缓存，校验相同 seeded prompt 的响应一致
└── hotpotqa/
```

普通策略和每个候选分支均保留 `run_manifest.json`、`progress.json`、`summary.json`、
`memory_gate.jsonl`、`llm_cache.sqlite`；比较结果含逐题配对检查、差值和配对 bootstrap CI。
`memory_gate.jsonl` 保存任务、memory 内容、预测、删除原因和最终结果。原始 LLM 响应
保留在 SQLite cache。主表不会用示例数值填充；未完成数据集显示 pending，不伪装成 0。

## 进度、恢复和错误

中断后使用**同一目录、同一完整正式命令**重跑，不加特殊恢复参数：

- 已完成阶段校验身份和日志后跳过；中断的诊断/评测按题/分支恢复。
- 若 MLP 拟合阶段本身中断，该拟合阶段重新训练；不会重采已完成反事实。
- 干预计划固定并记录 hash；检索候选改变、计划改变或输入日志改变会明确报错。
- 三个 gate 的缓存先合并，再串行复用到控制组和 Oracle 候选；相同 seeded prompt
  返回不同文本时停止，不挑选更幸运的一次。
- 总异常在 `benchmark_progress.json` 和 `logs/<stage>.log`；细分题目异常见对应
  `progress.json`，诊断和三 gate 的内部阶段见 `matrix_progress.json`。

不要删除原 cache、修改 manifest 绕过校验，或使用同目录切换数据/模型/策略定义。
修改实验设计请选新的 `--output-dir`。

## 本地验证范围

```bash
gmemory_env/bin/python -m unittest discover -s causal_memory_control/tests -t .
```

新增测试覆盖干预计划无答案输入、Random 可复现和预算匹配、全部删除、检索漂移拒绝、
FEVER/HotpotQA evaluator 接入及恢复、缓存冲突、Oracle 完整分支覆盖与事后上界、
表格百分制输出，以及 mock 双数据集全流程的中断恢复。真实全量 LLM 实验仍需在服务器运行。
