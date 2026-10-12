# 改进 gate 与 HotpotQA 实验

该版本完整运行：构建冻结 GMemory → 采集反事实分支 → 训练 estimator → 校准
DROP 策略 → 原生 GMemory baseline → gate → 配对对比。服务器执行入口是
`python -m causal_memory_control.run_hotpotqa_gate_comparison`。

## gate 到底学习什么

在检索完成、memory 进入团队 prompt 之前，针对每个候选构造
`x = (q, s_t, m, E_t, C_t)`。

| 符号 | 当前实现 |
| --- | --- |
| `q` | 当前问题；FEVER 是待判断 claim，HotpotQA 是 question |
| `s_t` | 检索之后、团队求解之前的 `task_description`；HotpotQA 包含问题和提供的上下文。这版是一次检索处的状态，不是逐轮动态推理状态 |
| `m` | 正在判断的 trajectory 或 insight 的内容、检索位置和可用来源字段 |
| `E_t` | 会收到 memory 的团队成员；特征使用暴露人数，不使用角色特定 utility |
| `C_t` | 本次检索到的完整候选集合；用于候选数量、候选间冗余等特征 |

对于同一问题、同一个候选、同一完整候选集合和同一采样 seed，执行两种条件：

| 条件 | 操作 | 最终监督信号 |
| --- | --- | --- |
| `use_all` | 所有 memory 都保留给团队 | 最终团队任务 reward |
| `global_drop` | 只删掉当前这条 memory，其他 memory 保持；所有团队成员都看不到被删候选 | 最终团队任务 reward |

把匹配重复结果取平均，得到 `r_use` 和 `r_drop`，训练标签为
`y = r_use - r_drop`。正值表示这条 memory 有帮助，负值表示删除可能更好。
HotpotQA 使用标准归一化答案 token-F1 作为 reward；EM 单独报告。答案和
gold supporting facts 只参与结果评分，不进入 gate 特征或团队输入 prompt。

旧模型是 bootstrap T-learner：分别回归 `Q_use(x)` 和 `Q_drop(x)`，相减得到
utility。该版保留这两个 head，增加直接回归 `y` 的 utility head：

```text
L_Q = mean((Q_use-r_use)^2 + (Q_drop-r_drop)^2)
L_U = mean((U_direct-y)^2)
L_pair = mean(max(0, margin - sign(y_i-y_j)*(U_direct_i-U_direct_j)))
U_hat = (1-alpha)*(Q_use-Q_drop) + alpha*U_direct
```

默认 `alpha=0.5`、`contrastive-weight=0.1`。配对仅在同一问题的候选之间建立，
跳过相同 utility 或绝对标签差小于 `0.05` 的对。utility head 先做 ridge 回归，
随后每一步同时保留绝对 utility 回归、L2 和辅助排序目标，避免仅排序而丢失
零点含义。它是有监督的候选排序，不是语义 embedding 的 InfoNCE 预训练。
这里的损失没有通过人为把标签改成负值来强迫 DROP。

每个 bootstrap 成员抽样完整的问题组，保留同一问题的全部候选，不再把
同一问题下的六条 memory 当作六个独立问题。默认 12 个成员，成员间 utility
标准差作为 `sigma`。不确定性是模型分歧指标，不保证具有某个置信覆盖率。

## insight 信号与更大胆的 DROP

此前 GMemory 只返回 insight 字符串，gate 看不到 `insights.json` 中的 score、
正/负关联任务计数。v2 在诊断采集和运行时都按原始 rule 文本恢复这些字段，
并记录缺失标志、关联数量和正关联比例。新日志直接保存被判断 memory 的
文本、rank 和元数据，便于追查误删。

这些是 GMemory 的规则维护信号，**不是 factual correctness 概率，也不是
当前问题的边际 utility 标签**。可靠的通用规则仍可能对某个问题无帮助，
甚至引导错误；因此把这些字段作为弱上下文特征学习，不强行把正关联规则
全部 KEEP。修复信息丢失和训练/部署不一致，不能保证消除所有 harmful DROP。

v2 还加入 query-memory 和 state-memory 的词项交互、词项重叠及 memory
内容坐标。estimator 当前使用稳定词项 hash 向量，不是预训练语义 encoder；
`--embedding-model` 控制的是原生 GMemory 的检索 embedding。

部署仍使用 `U_hat + kappa*sigma < -delta`。HotpotQA 默认
`residual-noise-scale=0`，移除原来的独立 outcome 残差下限，保留 bootstrap
分歧。设非零时，新 hybrid 模型使用 utility 残差，不直接叠加两个 Q 残差。

| 选择 | 行为 |
| --- | --- |
| `--kappa auto`（默认） | 在独立校准问题上比较 `0,0.1,0.25,0.5,1,1.96` 和全部保留，按观察到的平均 DROP 收益选取；收益相同优先较大 kappa |
| `--kappa 0.1` | 明确测试更宽松的门槛，需使用新的结果目录；不要根据最终测试结果选最优门槛后当作独立测试结论 |
| `--kappa 0` | utility 点估计为负就允许 DROP，是最大胆的现有单条删除策略 |
| `--max-drops 1` | 每个候选都预测，但最多删掉一个上界最低且过门槛的候选 |

若没有任何候选策略在校准集上带来严格正收益，auto 会选择全部保留
（`recommended_max_drops=0`）。单候选 DROP 训练不能验证多条同时删除的
交互效应，所以该完整管线要求 `max-drops=1`，没有直接取消删除数量限制。
校准收益仍可能受小样本噪声影响，必须看最终独立测试的配对置信区间。

## 数据划分与开销

默认 pilot 使用：

| 用途 | 问题数 | 是否用于训练 estimator |
| --- | ---: | --- |
| 构建冻结 memory | 50 | 否，仅构建 support memory |
| gate 反事实采集 | 400 | 内部再按问题划分，约 320 拟合、80 校准 |
| 最终 baseline/gate 测试 | 100 | 否，不参与拟合或 kappa 选择 |

同一个问题的全部候选只能在一个子集中。保存所有划分 ID，并检查 checkpoint
的采集来源、snapshot、模型、检索配置和训练/测试不重叠。

使用 `--use-all-data` 会把有效样本全部分配给 support、gate 数据和最终测试，
不会让同一批问题同时用于训练和测试。有效数由 loader 计算；缺少可评分字段
或不含上下文的数据不适用。本实验使用提供上下文的 HotpotQA distractor
JSON/JSONL；若使用官方 dev 做内部划分，应报告内部 held-out 结果，不能称为
官方未见 test 集结果。

默认检索 Top-3 trajectory + Top-3 insight、两组 seed、每组 repeats=2，
每个训练问题最多记录 `6 × 2条件 × 2重复 × 2seed = 48` 条团队分支。
pilot 的 400 个问题最多 19,200 条分支；这是团队运行次数，不是 LLM 调用次数。
相同 prompt 缓存会减少调用，缺失 rank 会记录 exclusion。扩大到全部数据之前，
先用 `--plan-only` 查看规模和完成 pilot。

## 服务器命令

在仓库根目录执行，使用之前已能跑 FEVER 的环境。把 `--data` 改为服务器
实际文件路径，endpoint 改为实际运行的 Ollama 端口。

```bash
conda activate gmemory
cd /mnt/mydata/hjq/casual-gmemory
export OPENAI_API_KEY=ollama
export HF_ENDPOINT=https://hf-mirror.com
python -c 'import langchain_chroma, numpy, tqdm, openai; print("dependencies ready")'
curl --fail http://127.0.0.1:11436/v1/models
```

先查看计划，不启动 LLM 调用：

```bash
python -m causal_memory_control.run_hotpotqa_gate_comparison \
  --data data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
  --endpoint http://127.0.0.1:11436/v1 \
  --model qwen2.5:7b \
  --plan-only
```

执行 pilot，自动完成所有阶段：

```bash
python -u -m causal_memory_control.run_hotpotqa_gate_comparison \
  --data data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
  --endpoint http://127.0.0.1:11436/v1 \
  --model qwen2.5:7b
```

默认目录均为新的 HotpotQA v2 目录，不复用 FEVER checkpoint。若中断，
**重新执行完全相同的命令**，无需额外 resume 参数。完成阶段会验证并跳过；
诊断按已完成分支、测试按已完成问题继续。当前未完成分支/问题重跑，但其
已提交的 LLM 响应仍可从缓存复用。一次 estimator 拟合若中断会重做拟合，
无需重采反事实数据。

若末行 JSONL 写了一半，会把片段备份为 `.partial-tail-*` 后恢复。中间行
损坏会明确报路径和行号，不自动丢弃。进度条涵盖阶段、反事实分支、ensemble
拟合及最终测试；错误保存到 `pipeline_progress.json`、阶段 `.log` 和子任务
`progress.json`，包含出错阶段、问题 ID（有时为子日志字段）、异常与 traceback。

所有有效数据实验使用独立目录。先在下面命令末尾加 `--plan-only` 查看规模：

```bash
python -u -m causal_memory_control.run_hotpotqa_gate_comparison \
  --data data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
  --endpoint http://127.0.0.1:11436/v1 \
  --model qwen2.5:7b \
  --use-all-data \
  --memory-dir causal_diagnostic/memory_snapshots/hotpotqa_gate_all_v2/g-memory \
  --diagnostic-results causal_diagnostic/results/hotpotqa_gate_all_v2 \
  --checkpoint causal_memory_control/checkpoints/hotpotqa_gate_all_v2.json \
  --output-dir causal_memory_control/results/hotpotqa_gate_all_v2
```

改变数据划分、模型、retrieval 或训练参数，必须使用匹配的原始采集配置或新
目录，不能把新配置混进旧结果。代码更新前已中断的旧格式诊断最好另开新目录；
新旧分支上下文不一致会明确拒绝训练。

## 对比学习消融与已有 FEVER 数据

pilot 完成后，保持采集配置和数据划分不变，用同一份完成的诊断训练数据，
禁用 pairwise 排序并重新拟合、校准、评测：

```bash
python -u -m causal_memory_control.run_hotpotqa_gate_comparison \
  --data data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
  --endpoint http://127.0.0.1:11436/v1 \
  --model qwen2.5:7b \
  --contrastive-weight 0 \
  --checkpoint causal_memory_control/checkpoints/hotpotqa_gate_v2_no_pairwise.json \
  --output-dir causal_memory_control/results/hotpotqa_gate_v2_no_pairwise
```

这样可以比较“有排序/无排序”；两臂各自的 auto kappa 可能不同，属于完整策略
对比。如果只考察训练损失，预先固定相同 kappa。不同消融输出各自包含相同的
原生 baseline；若要跨运行逐条比较，还应确认 baseline 响应一致。

已有 FEVER 的完整反事实数据也能训练新 estimator，无须重新下载数据：

```bash
python -m causal_memory_control.train_gate \
  --input causal_diagnostic/results/native_fever_all_binary_gate_7b_v5/seed0/branches.jsonl \
          causal_diagnostic/results/native_fever_all_binary_gate_7b_v5/seed1000_retest/branches.jsonl \
  --memory-dir causal_diagnostic/memory_snapshots/fever_evidence_support50_all_binary_7b_v5/g-memory \
  --metric success \
  --feature-version 2 --solver ridge --bootstrap-unit task \
  --direct-utility-weight 0.5 --contrastive-weight 0.1 \
  --residual-noise-scale 0 --calibration-fraction 0.2 \
  --output causal_memory_control/checkpoints/fever_gate_improved_v2.json
```

输入路径按服务器实际目录调整。旧数据缺少精确 `candidate_set` 时，trainer
使用同问题下已观察到的其他候选恢复集合；部分候选未采集的旧数据不能完全
恢复 peer 内容。部署用新 checkpoint、新输出目录、同一冻结 snapshot，读取
checkpoint 的 `training_metadata.policy_calibration` 推荐 kappa；如果推荐
`max_drops=0`，评测也要设置为 0。旧 FEVER 一键管线不会自动读取推荐策略。
复现旧训练配置显式设置 `--feature-version 1 --solver sgd --bootstrap-unit event
--direct-utility-weight 0 --contrastive-weight 0`，并使用原先 residual scale。

## 结果有效性与传回文件

gate cache 初始化自同次 baseline 的 SQLite。相同原始候选和相同 prompt 应
复用同样输出。比较时同时检查原始检索 fingerprint，并要求无 DROP 的问题
具有相同答案与得分。若出现此前 FEVER 的 no-DROP 5 害/4 益现象，结果写出
后阶段报错，`paired_control_valid=false`；先查配对问题，不将差异计入 gate
增益。比较报告绑定两份源日志摘要，修改日志后不会误复用旧 comparison。

| 文件 | 用途 |
| --- | --- |
| `causal_memory_control/results/hotpotqa_gate_v2/comparison.json` | F1/EM 两臂均值、配对增益、95% bootstrap 区间、DROP 问题数和一致性检查 |
| 两臂 `summary.json`、`memory_gate.jsonl`、`run_manifest.json` | 汇总、每题结果、候选决策/文本/元数据和实验配置 |
| `causal_memory_control/checkpoints/hotpotqa_gate_v2.json` | estimator、特征版本、训练/校准划分和 kappa 策略选择 |
| 管线 `pipeline_manifest.json`、`pipeline_progress.json` 和 `logs/` | 运行配置、阶段状态与错误 |
| 诊断两个 seed 的 `run_manifest.json`、`summary.json`、`branches.jsonl`（如需深查训练） | 训练标签、候选作用和重复之间噪声 |

上述结果和 checkpoint 可传回分析。无需传虚拟环境、模型权重或 SQLite cache；
缓存若要用于服务器继续运行请保留。代码已经过离线测试；实际 F1/EM 是否改善
以服务器完成的独立测试为准。

## Git 上传范围

把本次以下源码和文档一起提交，然后服务器 `git pull` 更新整个提交。不要
提交数据、memory 数据库、结果和环境目录。

```bash
git add \
  causal_memory_control/README.md \
  causal_memory_control/HOTPOTQA_GATE_V2.md \
  causal_memory_control/hotpotqa_experiment.py \
  causal_memory_control/run_hotpotqa_gate_comparison.py \
  causal_memory_control/insight_metadata.py \
  causal_memory_control/jsonl.py \
  causal_memory_control/estimator.py \
  causal_memory_control/features.py \
  causal_memory_control/runtime_gate.py \
  causal_memory_control/train_gate.py \
  causal_memory_control/gmemory_adapter.py \
  causal_memory_control/compare_gate_runs.py \
  causal_memory_control/fever_experiment.py \
  causal_memory_control/tests/test_hotpotqa_gate.py \
  causal_diagnostic/fever_oracle_recipient/run_experiment.py \
  causal_diagnostic/fever_oracle_recipient/utils.py \
  causal_diagnostic/hotpotqa_oracle_recipient/run_experiment.py \
  causal_diagnostic/hotpotqa_oracle_recipient/utils.py
git diff --cached --stat
git commit -m "Improve memory utility gate and add resumable HotpotQA comparison"
git push
```

这些命令由你在本地执行；尚未自动提交或 push。服务器使用 `git pull`，不需
按文件逐一拉取；如果服务器存在本地改动，先保留这些改动再处理合并。
