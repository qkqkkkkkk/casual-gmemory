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

## 已完成 FEVER seed0：只追加 ST 训练和测试

如果已经完成全量 FEVER seed0 采集以及 `run_seed0_gate_matrix` 测评，
无需等待或补采 seed1000。新选项 `--seed0-only` 只读取第一组已有分支；
`--seed0-results-root` 指向上次 **seed0 测评根目录**，从其
`fever/baseline_selection.json` 找到真正使用过的 baseline 和回答缓存。
这两个目录用途不同，不能把原采集目录替换成测评目录。

| 本次步骤 | 数据及行为 |
| --- | --- |
| 冻结记忆库 | 只读复用原 FEVER snapshot，不重建 |
| 训练标签 | 原 5,292 诊断任务 × 6 候选 × 2 repeats；127,008 条配对分支日志 |
| estimator fit / 开发集 | 同题所有候选一起划分，沿用旧方法的 task split（calibration seed 71） |
| 新训练 | 仅单输出头 Semantic ST，weighted MSE + 同题表示 InfoNCE |
| 新评测 | 原独立 1,324 题，仅 ST gate；沿用原 endpoint/model/seed/top-k |
| 旧方法 | baseline、Linear、Semantic Linear、三头 MLP 仅复用已通过审计的成绩 |
| 删除策略 | 开发集选 κ，每题最多删 1 条；没有正开发集收益则 KEEP ALL |

上表题数对应默认全量 FEVER 配置，以实际 manifest 为准。seed0 组的两个
repeat 对应 sample seed 0、1，不是只有一次回答，也不伪装为两组独立采集。
本选项**不会执行任何采集，也不会补跑缺失的 baseline 或其他旧方法**；
seed0 缺分支、baseline/cache 缺失或配置不一致会直接报错。不能与
`--prepare-missing` 同用。未提供 `--seed0-results-root` 时，仅尝试复用
原 `--source-root/fever/native_gmemory`，不会自动寻找其他目录。

服务器上先检查执行计划（不调用 LLM、不写新结果）：

```bash
cd /mnt/mydata/hjq/casual-gmemory
conda activate gmemory
export OPENAI_API_KEY=ollama

python -u -m causal_memory_control.run_st_gate_benchmark \
  --source-root causal_memory_control/results/full_gate_benchmark_v1 \
  --seed0-results-root causal_memory_control/results/fever_seed0_gate_matrix_v1 \
  --output-dir causal_memory_control/results/fever_seed0_st_gate_v1 \
  --datasets fever \
  --seed0-only \
  --plan-only
```

若上次 seed0 测评用了不同的目录名，请只替换 `--seed0-results-root`。
计划应显示 `collection_groups=1`、`new_diagnostic_branches=0`、
`sample_seeds=[0,1]`，且只有一个 seed0 `branches.jsonl` 输入。
确认后去掉 `--plan-only` 正式运行；相同命令可断点恢复。
不会改写源目录或旧 seed0 结果；新的 checkpoint、训练 summary、逐题 gate
日志、comparison、阶段进度和异常都保存到 `fever_seed0_st_gate_v1`。

主表 `st_results.md/csv/json` 比较 baseline、Linear、Semantic Linear、
Semantic MLP、Semantic ST；如旧目录已有 Always Global Drop，也只读加入该行。
旧方法缺少可验证的成绩时显示 `not run / invalid`，绝不自动重跑。
`st_results.json` 的 `details` 包含 ST 的配对收益、95% CI、DROP 改善/伤害数，
`data_protocol` 记录单组输入和最终测试边界。`fever/training_summary.json`
记录每轮 loss、rarity 分布、真实有效对比对数与开发集策略选择。

这次相对当前已同步代码，只需上传以下改动：

```bash
git add \
  causal_memory_control/run_st_gate_benchmark.py \
  causal_memory_control/evaluation_replicas.py \
  causal_memory_control/compare_gate_runs.py \
  causal_memory_control/tests/test_st_seed0.py \
  causal_memory_control/ST_GATE_BENCHMARK.md
git diff --cached --stat
git commit -m "Reuse completed seed0 data and evaluations for ST utility gate"
git push origin main
```

服务器执行 `git pull --ff-only origin main` 后运行上面的命令。
这是论文 ST 损失的适配，不是完整 CSM 复现，也不是只改 loss 的严格消融：
与旧三头 MLP 相比还改变了输出头与训练配置。不能保证一定提升。
FEVER 约 91.89% 的诊断 U 为零时，对比项尤其依赖少量同题内具有不同 U 的候选；
需查看 `contrastive_task_groups`/`contrastive_positive_pairs`，若为零，实际只训练加权回归。
最终测试已被多轮查看，本轮应视为探索性比较；不要根据这次测试表现继续反复调参，
正式结论宜在新的保留测试集上验证。

### 可选：ST 训练和新答题服务使用 GPU 0

Python 的 `CUDA_VISIBLE_DEVICES` 只约束 Python 本身，不能移动一个已经运行的
Ollama 服务。要与 GPU 2 的 HotpotQA 服务并行，需单独在 GPU 0 启动 Ollama，
并使用 `--evaluation-endpoint`。不要重启系统 11434 或已有 11438 服务。

在服务器新终端确认 `ss -ltnp | grep -E ':11440([[:space:]]|$)'` 没有监听输出后：

```bash
CUDA_VISIBLE_DEVICES=GPU-3529cecd-95a8-0ba6-626f-582dc307e183 \
OLLAMA_HOST=127.0.0.1:11440 \
OLLAMA_MODELS=/mnt/mydata/hjq/ollama/models \
OLLAMA_NUM_PARALLEL=1 \
/usr/local/bin/ollama serve
```

该终端保持运行；模型目录直接包含 `blobs/`、`manifests/`，可以与另一个只读推理
实例共用，但不要在实验期间 pull/rm/create 覆盖模型。GPU 0 并非独占，先确认其
显存与负载允许。另一个终端用 `OLLAMA_HOST=127.0.0.1:11440 ollama list` 检查模型。

先核对原、新服务的上下文设置；有已加载模型时可查看：

```bash
curl --fail --silent --show-error http://127.0.0.1:11434/api/ps
curl --fail --silent --show-error http://127.0.0.1:11440/api/ps
```

`context_length`（服务版本支持时）和生成设置应一致；`/api/show` 不能证明服务环境
变量完全相同。不要为了启动新服务而修改原 manifest。新服务若尚未加载模型，可先
运行一次 `OLLAMA_HOST=127.0.0.1:11440 ollama run qwen2.5:7b '仅回复 OK'`，再检查
`nvidia-smi -i 0`。本脚本不会替用户启停 Ollama 或选择其 GPU。

正式运行（建议先追加 `--plan-only` 看计划；plan-only 不访问网络）：

```bash
cd /mnt/mydata/hjq/casual-gmemory
conda activate gmemory
export OPENAI_API_KEY=ollama

CUDA_VISIBLE_DEVICES=GPU-3529cecd-95a8-0ba6-626f-582dc307e183 \
python -u -m causal_memory_control.run_st_gate_benchmark \
  --source-root causal_memory_control/results/full_gate_benchmark_v1 \
  --seed0-results-root causal_memory_control/results/fever_seed0_gate_matrix_v1 \
  --output-dir causal_memory_control/results/fever_seed0_st_gpu0_api11440_v1 \
  --datasets fever \
  --seed0-only \
  --st-device cuda:0 \
  --evaluation-endpoint http://127.0.0.1:11440/v1
```

保持新目录，不能把已有 endpoint/设备配置的目录就地改成另一套配置。ST 参数拟合
使用 GPU 0；当前语义特征编码、checkpoint 加载后的在线 ST gate U 预测默认仍在 CPU。
答题请求发往 11440，由该服务的 GPU 0 runner 执行。检索、MAS、训练标签、开发集规则
及原 baseline cache 不变；不会重跑其他方法或重新采集 seed0。

实际运行会在训练前只读探测原、新 endpoint 的 Ollama `/api/tags`、`/api/show` 和
`/api/version`，核对模型 digest、模板/参数与版本。任一不匹配或服务不可达会停止，
不绕过检查、不虚构概率/收益。依据保存到 `fever/evaluation_replicas.json`，绑定原
baseline 日志、manifest、cache 的 SHA256。source 数据始终只读。

原 baseline 日志继续记录 11434，新 ST 日志如实记录 11440。
配对审计只在显式验证依据下允许这一对 endpoint 不同，其他 metadata、检索指纹、
无 DROP 输出检查保持严格；未提供验证依据时 `compare_gate_runs` 仍会拒绝端口差异。
`comparison.json` 保存验证文件指纹和局限性，不能将 metadata 匹配理解成所有随机
输出或 GPU 数值计算逐位一致。旧 baseline 没有历史模型 digest，现在的探测也不能
追溯证明其生成时权重未变；应确保原 tag 自 baseline 生成后没有被更新。

未完成评测的恢复会重新核对在线副本；已完成评测可以在服务离线时重新审计/汇总。
相同命令恢复时不能改变任一副本的模型权重、模板或版本。

## 服务器：已有全量诊断和 baseline

本次需同步的代码/文档（不包含数据、结果、其他未跟踪改动）：

```bash
git add \
  causal_memory_control/st_estimator.py \
  causal_memory_control/st_data.py \
  causal_memory_control/run_st_gate_benchmark.py \
  causal_memory_control/evaluation_replicas.py \
  causal_memory_control/compare_gate_runs.py \
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
