# ST-only：全量 FEVER / HotpotQA 新方法实验

入口：`python -m causal_memory_control.run_st_gate_benchmark`。
只新增 ST estimator 的训练和最终测试；不会启动旧 Linear、Semantic Linear、三头 MLP、
Jev、Random、Global Drop 或 Oracle 实验。旧结果若存在且配对审计有效，仅在表中复用。

## 借用了论文的哪些部分

依据 Deng et al., *Influence Guided Context Selection for Effective
Retrieval-Augmented Generation*, arXiv:2509.21359v2，§4.2、公式 6–7。
ST 是 Supervised Training，不是 straight-through，也不包含 §4.3 的 E2E。

本实现是 **ST loss 的适配实验，不是论文完整 CSM 复现**：

| 项目 | 新 ST 版本 |
| --- | --- |
| 输入 | 现有冻结 MiniLM：task/query、合法任务 state、目标 memory、其他候选及结构特征 |
| 网络 | 128 → 64 ReLU 表示层 + **一个** tanh 直接 U 输出头 |
| 标签 | 同题、同候选集合下，配对采样的 `U = mean(R_USE) - mean(R_DROP)` |
| FEVER / HotpotQA reward | 二分类正确性 / Answer F1；不是论文的生成器 cross-entropy CI |
| 回归 | 按整道题的 rarity 经验频率进行加权 MSE |
| 对比 | **同一道题内**，隐藏表示的点积 InfoNCE；不是旧标量输出的 hinge ranking |
| 推理 | 保留现有开发集校准的单条删除 gate；不采用论文“只保留正 CI”的全部筛选规则 |

公式（`g` 是最后一层共享隐藏表示）：

\[
g_m=h_\theta(z_m),\qquad \hat U_m=\tanh(w^Tg_m+b).
\]

\[
r_i=\mathrm{mean}(U_i)+10\,\mathrm{std}(U_i),\qquad
L_{MSE}=\mathrm{mean}_{i}\left[w_i\,\mathrm{mean}_{m\in C_i}(\hat U_m-U_m)^2\right].
\]

用拟合任务的 `r_i` 分箱估计频率 `p_i`；`w_i` 是 `min(20, 1/p_i)` 再归一化到均值 1。
分箱、权重上限及 reward 尺度上的 hard/正负阈值是本实验的显式适配，非论文给定值。
频率、标准化统计量、hard 任务选择、正负对只使用 **fit** 部分的标签。

对 `r_i > hard_threshold` 的任务，排除 anchor 自身：

- 正对：`abs(U_anchor - U_positive) < positive_epsilon`。
- 负对：`abs(U_anchor - U_negative) > negative_epsilon`。
- 每个正对的分母为该正对及该 anchor 的所有负对；对所有有效正对/anchor 求平均。
- 缺正对或缺负对则跳过；不跨题造对，不虚构标签。

\[
L_{ST}=L_{MSE}+0.1L_{InfoNCE},\qquad \tau=1.
\]

不做论文的 BERT 微调、全局 self-attention、语义去重或跨题 context 重组。
重组会改变原候选集合和真实效用，必须重新采集标签，不能直接挪用旧标签。
现有冻结语义特征包含其他候选的均值，但不等价于论文的 list-attention。

使用 12 个 task-bootstrap 成员，默认每个训练 10 epochs、batch 16 tasks、AdamW lr=0.001。
成员平均值是预测 U；成员标准差是 gate 的不确定性启发值，**不是具有覆盖保证的置信区间**。
runtime 兼容字段 `q_use/q_drop = 0.5 ± U/2` 仅是派生占位，不是两个训练出来的 outcome 头。
所有训练/加载操作隔离 Torch RNG，避免改变下游 MAS 采样序列。

## “全量”及删除预算

复用输入文件全部有效样本的互斥划分；不是把测试题混入训练。
若采用旧全量脚本的默认划分：

| 数据集 | 有效样本 | 冻结 memory support | 诊断 fit+dev | 最终测试 |
| --- | ---: | ---: | ---: | ---: |
| FEVER binary | 6,666 | 50 | 5,292 | 1,324 |
| HotpotQA distractor | 7,404 | 50 | 5,883 | 1,471 |

诊断里再按 task 分 80% fit / 20% dev。同题的所有候选始终在一起。
沿用两组采集（seed0、seed1000_retest），而不是重新采集两套 ST 数据。
trajectory 和 insight 的所有已检索 rank 都训练、预测，**不是只允许删 insight**。

在独立 dev 中从 `0,0.1,0.25,0.5,1,1.96` 选择 κ：
候选需满足 `U + κσ < -delta`，每题最多删除一条上界最低的 memory。
若任何 κ 都没有正的 dev 平均收益，选择 KEEP ALL（预算为 0）。
最终测试不会用来调阈值，也不为“产生 DROP”强制改变规则。

## 服务器：已有全量诊断和 baseline

本次需同步的代码/文档（不包含数据、结果、其他未跟踪改动）：

```bash
git add \
  causal_memory_control/st_estimator.py \
  causal_memory_control/st_data.py \
  causal_memory_control/run_st_gate_benchmark.py \
  causal_memory_control/train_gate.py \
  causal_memory_control/runtime_gate.py \
  causal_memory_control/tests/test_st_gate.py \
  causal_memory_control/ST_GATE_BENCHMARK.md \
  causal_memory_control/README.md
git commit -m "Add ST-only contrastive utility gate benchmark"
git push origin main
```

服务器在原仓库执行 `git pull --ff-only origin main`，再执行下方实验命令。
如果服务器有未提交改动，先检查并保留它们，不要用强制 reset 覆盖。

确保 `--source-root` 指向上一次 **全量** `run_full_gate_benchmark` 结果根目录。
不要求旧七种方法都完成；只要求冻结 memory、两组完整诊断、原 baseline 存在。
100 题 pilot 不可伪装为全量输入。

```bash
cd /mnt/mydata/hjq/casual-gmemory
conda activate gmemory
export OPENAI_API_KEY=ollama

python -u -m causal_memory_control.run_st_gate_benchmark \
  --source-root causal_memory_control/results/full_gate_benchmark_v1 \
  --output-dir causal_memory_control/results/st_gate_benchmark_v1 \
  --plan-only
```

核对数据来源和题数后，正式运行同一命令，**去掉 `--plan-only`**。
source 存在 `benchmark_manifest.json` 时，会精确导入 endpoint/model/split/检索参数；
不用再写 `--endpoint`，防止把原来的 11434 改成 11436。
原数据路径和来源目录需要保留；不要移动后手改 SHA 或 manifest。

## 还没有全量数据产物：只补齐必要前置步骤

没有完整全量诊断时，默认会明确报错，不自动产生昂贵调用。
确认需要收集后，使用**新的共享数据目录**并显式加 `--prepare-missing`：

```bash
python -u -m causal_memory_control.run_st_gate_benchmark \
  --source-root causal_memory_control/results/st_shared_data_v1 \
  --output-dir causal_memory_control/results/st_gate_benchmark_v1 \
  --fever-data data/fever/fever_dev.jsonl \
  --hotpotqa-data data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
  --endpoint http://127.0.0.1:11434/v1 \
  --model qwen2.5:7b \
  --prepare-missing \
  --plan-only
```

同样核对后去掉 `--plan-only`。这一路径只补齐冻结 memory、诊断、baseline，然后
训练/测试 ST，**不会跑其他 gate**。默认两个数据集新诊断约 536,400 个团队分支，
成本远大于只复用产物；不是轻量连通性测试。模型和原有依赖沿用 `gmemory` 环境。
可用 `--datasets fever` 或 `--datasets hotpotqa` 单独运行。切换参数/数据集/超参需新
ST 输出目录；不与旧目录混跑。

## 日志、表格和恢复

新结果全部保存在独立 ST 目录，旧方法结果文件不会被改写：

```text
st_gate_benchmark_v1/
  st_manifest.json / experiment_plan.json
  st_progress.json                    # 当前阶段、错误、traceback、命令、log tail
  st_results.md / .csv / .json         # 新 ST + 可验证复用的旧表格行；缺结果明确标记
  logs/<dataset>_train_st.log          # 原始训练控制台、进度条
  logs/<dataset>_evaluate_st.log       # 测试进度与错误
  <dataset>/
    reused_inputs.json                # 两组分支、snapshot、baseline/cache 指纹
    checkpoint.json                  # 模型、标准化、训练上下文及 split
    training_summary.json            # rarity、U 分布、有效对比对、逐成员/epoch loss、dev 策略
    semantic_cache.sqlite
    evaluation/
      run_manifest.json / progress.json / summary.json
      memory_gate.jsonl              # 每题所有候选正文、预测 U/σ、动作、最终输出和奖励
      llm_cache.sqlite
      comparison.json               # 配对增益、CI、改善/伤害和 no-DROP 审计
```

完全相同命令可恢复：已完成的诊断、baseline、ST checkpoint、测试题会复用；测试使用
原 baseline cache，无 DROP 任务必须保持输出一致，否则拒绝把差异当成 gate 收益。
若在 estimator 拟合中断，尚未完成的整次拟合会重做（不重新调用 LLM 收集诊断）；
不声称支持 optimizer/minibatch 级恢复。临时 SQLite 自动清理；输入日志始终只读。

若 `contrastive_task_groups=0`，表示在当前阈值下没有真实正负对，InfoNCE 实际为零；
日志会告警。若 dev 选择 KEEP ALL，表示没有测得可靠收益，不代表脚本没预测 U。

回传分析通常只需 `st_results.*`、`st_progress.json`、两数据集的 `training_summary.json`、
`evaluation/{comparison,summary,run_manifest}.json` 和 `memory_gate.jsonl`；排除 SQLite
及大型 checkpoint。未在本机执行真实全量 LLM 实验，服务器实测以最终日志为准。
