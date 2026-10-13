# 语义 MLP 与独立 Jev 基线

统一入口：`python -m causal_memory_control.run_gate_matrix --dataset hotpotqa`，
改用 `--dataset fever` 可切换到 FEVER。数据集切换、目录和扩展方式见
[DATASET_GATE_MATRIX.md](DATASET_GATE_MATRIX.md)。
旧入口 `python -m causal_memory_control.run_hotpotqa_gate_matrix` 仍然可用。
同一次实验共用冻结 GMemory、反事实数据、按问题划分的拟合/开发/最终测试集，
以及一份原生 GMemory baseline 和它的 LLM 缓存。它替换检索后、memory 暴露给
团队前的 gate，不改变原生检索、MacNet 协作或 USE/DROP 采集条件。

## 对比方法

| 输出目录 | 特征与模型 | 监督/校准 | 用途 |
| --- | --- | --- | --- |
| `native_gmemory` | 原生 GMemory，全 KEEP | 无 | 公共 baseline |
| `linear` | v2 词项 hash、交互、来源字段 + 线性三头 ensemble | USE/DROP outcome；开发集选 kappa | 保留已有轻量方法 |
| `semantic_linear` | v3 冻结语义 embedding、交互、来源字段 + 线性三头 ensemble | 同上 | 隔离语义特征的贡献 |
| `semantic_mlp` | 同一 v3 特征 + 小型三头 MLP ensemble | 同上 | 进一步隔离非线性模型的贡献 |
| `jev_direct` | Jev 的 `retain_helpful` 概率 | 不训练 Jev；开发集选择 retention threshold | 独立语义判断基线 |
| `jev_decomposed` | Jev 的事实相关性、通用指导、冲突、冗余概率 | 固定组合公式；开发集选择 retention threshold | 检验区分事实记忆和推理规范是否有帮助 |

默认跑 `linear`、`semantic_linear`、`semantic_mlp` 三种 Gate，加上原生 baseline
共四组。Jev 对比暂时关闭，当前默认命令不需要 Jev Key；它的代码和显式
`--arms` 选项保留供以后使用。Jev 特征也没有融合进 MLP。

## 语义 estimator 的训练

编码器默认 `sentence-transformers/all-MiniLM-L6-v2`，保持冻结。对问题 `q`、
当前提供的上下文 `s_t`、待判断 memory `m` 和其他候选分别编码。长上下文按
token 窗口分块，汇总各块的归一化向量后再归一化，避免直接截断末尾段落。
该汇总会丢失部分跨段关系，不等同于一个专门训练的多跳推理 encoder。

模型输入包含完整语义向量、query-memory/state-memory 的逐维乘积和绝对差、
其他候选的均值向量，以及 v2 的检索位置、候选统计和 insight 维护来源字段。
原生 `--embedding-model` 仍只控制 GMemory 检索；`--semantic-model` 控制 gate。

同一问题、候选集合、目标 memory 和 sampling seed 的匹配分支提供监督：

```text
r_use  = 平均最终团队 reward（全部 memory 保留）
r_drop = 平均最终团队 reward（只删除目标 memory）
y      = r_use - r_drop
```

MLP 默认两层 `128 → 64`、ReLU，共享表示输出两个 sigmoid outcome head 和
一个 tanh utility head。训练目标为 outcome MSE、utility MSE、同一问题内的
utility 排序 hinge loss；AdamW、梯度裁剪、默认 80 epochs。
默认 12 个 bootstrap 成员按完整问题组抽样，同一问题的候选一起进入 minibatch。
标准化参数只由拟合集估计。编码器没有被这个 loss 微调。

```text
U_hat = (1-alpha)*(Q_use-Q_drop) + alpha*U_direct       alpha 默认 0.5
DROP 当 U_hat + kappa*sigma < -delta
```

`sigma` 是 ensemble 分歧，不是具有已验证覆盖率的置信区间。默认去掉独立
outcome residual floor（`--residual-noise-scale 0`）；`--kappa auto` 使用开发集
选择，包含 always KEEP。日志的 Q 字段沿用 hybrid 接口，由 outcome 中心与混合
utility 重构，不应当解释为经过概率校准的成功率。

每条检索候选都会判断；checkpoint 未覆盖的 kind/rank 默认 KEEP。
最多删除一条符合门槛的 memory，因为训练标签只验证单条删除。
如果开发集上没有观察到严格正收益，就选择 `recommended_max_drops=0`。

## 可选 Jev 协议与 DROP 规则（默认关闭）

HTTP adapter 按用户提供的 TypeSafe 协议实现：

| 配置 | 当前值 |
| --- | --- |
| 请求 | `POST https://api.typesafe.ai/v1/systemone` |
| 认证 | 环境变量 `TYPESAFE_API_KEY`，Bearer header |
| 默认模型 | 固定 `jev-1.13.0`，可用 `--jev-model` 指定 |
| body | `model`、结构化 `state`、`questions` |
| 二元回答 | `answers.<question_id>.noul`，校验有限值且在 `[0,1]` |
| prompt 版本 | `memory-retention-judgments-v1` |

Direct 使用 `retain_helpful`；Decomposed 请求 `claim_specific_relevance`、
`general_guidance`、`conflict`、`redundancy`，预先固定组合：

```text
score = max(claim_specific_relevance, general_guidance)
        * (1-conflict) * (1-0.5*redundancy)
```

这个组合是启发式 score，不是概率，也不是真实 counterfactual utility。
`score < threshold` 才有资格 DROP，最多删除 score 最低的一条。
阈值只根据独立开发问题的真实 USE/DROP outcome 选择；Jev 模型权重不训练，
最终测试 reward 不用于选阈值。

`state` 只包含当前问题、提供的上下文、目标和其他 memory 的文本及检索位置、
暴露人数、历史 memory 的合法来源字段。当前题 gold label/answer、gold supporting
facts、下游 agent 输出和真实 USE/DROP reward 均不发送给 Jev。
旧 memory 的 `source_label` 是历史 trajectory 信息，不是当前题的正确答案。

通用 insight 即使缺少当前问题的事实信息，也可通过 `general_guidance` 保留价值。
这缓解“事实不相关就删除规则”的问题；不能保证消除所有误删。GMemory insight
score 与正/负任务关联是维护信号，不是当前问题的成功概率。

请求按模型、adapter 版本、prompt、完整 state 和 questions 存入 SQLite 缓存。
模型版本、prompt 版本、耗时和服务返回的 token usage 被记录；API Key 不进入
checkpoint、缓存或实验日志。费用只记录服务实际返回的 `cost`，缺失时零不表示
免费；当前没有自行推算账单。

429、5xx、网络错误最多重试三次；401 等非暂时错误直接记录。
在线失败时 KEEP，并记录 `jev_error_default_keep` 和真实错误类型。
任何 Jev 失败会使对应正式对比无效，矩阵停止在 compare 阶段，避免把失败降级
产生的全 KEEP 当作成功的 Jev 结果。

## 在服务器运行

使用已有 `gmemory` 环境，以下从仓库根目录执行。数据文件和 Ollama 端口改为
服务器实际值。默认只运行三组本地 estimator 和原生 baseline。

```bash
conda activate gmemory
cd /mnt/mydata/hjq/casual-gmemory
export OPENAI_API_KEY=ollama
export HF_ENDPOINT=https://hf-mirror.com
python -c 'import torch, sentence_transformers, langchain_chroma, tqdm; print("dependencies ready")'
```

如果缺少 torch，在这个环境中安装与服务器硬件匹配的 PyTorch。CPU 训练也可跑，
默认 `--mlp-device cpu`；GPU 训练可指定 `--mlp-device cuda:0`，推理默认 CPU。
为重复实验固定语义模型的 Hugging Face commit：`--semantic-revision <commit>`，
或 `--semantic-model /path/to/immutable/model`。不要运行途中改变本地模型目录。
默认未指定 revision 时，缓存按模型名称区分，不能保证跨日期下载相同权重。

先只查看规模，不调用模型：

```bash
python -m causal_memory_control.run_hotpotqa_gate_matrix \
  --data data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
  --endpoint http://127.0.0.1:11436/v1 --model qwen2.5:7b \
  --plan-only
```

正式执行同样命令，去掉 `--plan-only`：

```bash
python -u -m causal_memory_control.run_hotpotqa_gate_matrix \
  --data data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
  --endpoint http://127.0.0.1:11436/v1 --model qwen2.5:7b
```

默认 pilot：50 个 memory support 问题、400 个诊断问题（约 320 拟合、80 开发）、
100 个最终测试问题。诊断最多 `400 × 48 = 19,200` 条团队分支；LLM 调用可能
多于团队分支数。默认 Jev 请求数为零。

也可以显式固定三组本地 estimator，并指定单独目录：

```bash
python -u -m causal_memory_control.run_hotpotqa_gate_matrix \
  --data data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
  --endpoint http://127.0.0.1:11436/v1 --model qwen2.5:7b \
  --arms linear,semantic_linear,semantic_mlp \
  --matrix-dir causal_memory_control/results/hotpotqa_semantic_only_v3
```

旧矩阵已登记五种 Gate 时，关闭 Jev 后也要换新的 `--matrix-dir`，可保持
原 memory/diagnostic 路径复用采集结果。以后恢复 Jev 时显式添加
`--arms linear,semantic_linear,semantic_mlp,jev_direct,jev_decomposed`，
设置环境变量 `TYPESAFE_API_KEY`，用新 `--matrix-dir`，保持相同 memory 和 diagnostic 路径，并用
`--baseline-cache-seed-from causal_memory_control/results/hotpotqa_semantic_only_v3/native_gmemory/llm_cache.sqlite`
复用之前的原生 baseline 响应。矩阵配置不允许直接改变既有目录的 arms。

可选 Jev 按给定协议通过 mock 验证；真实服务连通性和固定模型版本的权限
需要实际运行时验证。启用时可在服务器 Bash 终端输入 Key，不粘贴到聊天或 git：

```bash
read -rsp 'TypeSafe API key: ' TYPESAFE_API_KEY
export TYPESAFE_API_KEY
printf '\n'
```

默认 memory/diagnostic 路径沿用 HotpotQA v2。已有且配置一致的采集结果会验证
后复用，无须重采；模型、数据划分、retrieval、snapshot 或采样设置不同则拒绝混用。
矩阵内部 checkpoint 位于 `<matrix-dir>/checkpoints/<arm>.json`；这里使用
`--matrix-dir` 指定结果目录，不使用旧单方法入口的 `--output-dir/--checkpoint`。

`--use-all-data` 可以把全部有效样本分到 support、诊断、最终测试。需另设对应
的 `--memory-dir`、`--diagnostic-results`、`--matrix-dir`，先加 `--plan-only`
估算规模。所有方法遵循同一划分；不能把全部问题同时当训练集和测试集。

## 恢复、误删控制与结果

中断后重新执行原命令：已完成阶段自动验证并跳过；采集按分支、评测按问题
继续；成功的 embedding/Jev/LLM 请求缓存会复用。MLP 若在拟合途中中断，当前
拟合阶段重新训练，反事实采集不会重跑。JSONL 不完整末行先备份再修复，中间
损坏会明确报文件和行号。

如果 Jev 请求曾出错，修复 Key/权限/网络后，用原命令加 `--retry-jev-errors`。
只重跑含 Jev 错误的测试问题，旧行备份为 `.before-jev-retry-*`；该问题成功的
Jev 请求仍读缓存。开发集调用失败时 checkpoint 尚未创建，重跑原命令继续
使用成功的 Jev 缓存完成校准。

`--max-changed-harm-rate 0.25` 可要求开发集中，在 outcome 有变化的 DROP 中，
伤害占比不超过 25%；默认 1.0 不施加这个约束。它只是经验筛选，不是统计风险
保证；更严格可能回到全部 KEEP。改变该参数需新矩阵目录。

正式对比必须满足相同原始 retrieval fingerprint，且没有 DROP 的问题必须得到
与 baseline 相同的 prediction/reward/done。检查失败会写出具体任务 ID 并停止，
不能把其他随机差异归因于 gate。

| 文件 | 内容 |
| --- | --- |
| `matrix_results.md/csv/json` | 所有方法的 F1、EM、F1 增益和配对 CI、保留率、改善/伤害问题数 |
| `matrix_manifest.json` | 锁定实验配置和身份 |
| `matrix_progress.json`、`logs/*.log` | 当前阶段、具体异常、命令、traceback、日志末尾 |
| `<arm>/comparison.json` | 与公共 baseline 的完整配对比较、有效性检查、Jev usage |
| `<arm>/memory_gate.jsonl` | 每道题的最终结果、被删 memory 内容和预测/概率 |
| `<arm>/summary.json`、`progress.json` | 该方法整体指标与进度 |
| `checkpoints/*.json` | 模型、拟合/开发 ID、开发策略和采集来源 |
| `semantic_cache.sqlite`、各方法的 `llm_cache.sqlite`、可选 `jev_cache.sqlite` | 运行恢复缓存；默认不创建 Jev 缓存 |

分析时传回前三类配置/报告、各方法的 JSONL 与 summary、checkpoints 即可。
SQLite 缓存为继续运行保留在服务器，通常无需传回。

这是问题不重叠、共享冻结 memory 的部署实验。checkpoint 的
`candidate_content_overlap` 报告拟合/开发之间共享 memory 文本的情况；它没有
实现 candidate-disjoint 划分，不能据此声称泛化到未见 memory/insight。
此前 FEVER 最终集已用于方法设计，新结果应标为复查/诊断；HotpotQA 的新 held-out
问题可用于本轮验证。F1 提升只有配对区间等证据支持时才可声称稳定收益。

## FEVER 复用

FEVER 已接入同样的矩阵，默认也只跑本地四组对比，统一入口会自动使用 `--metric success`，切换
FEVER 的原生环境、数据划分、checkpoint 校验和准确率报告：

```bash
python -u -m causal_memory_control.run_gate_matrix \
  --dataset fever --data data/fever/fever_dev.jsonl \
  --endpoint http://127.0.0.1:11436/v1 --model qwen2.5:7b
```

也可以手动用已有匹配分支训练：

```bash
python -m causal_memory_control.train_gate \
  --input /path/to/train/seed0/branches.jsonl /path/to/train/seed1000_retest/branches.jsonl \
  --memory-dir /path/to/fever/g-memory --metric success \
  --feature-version 3 --estimator mlp --residual-noise-scale 0 \
  --calibration-fraction 0.2 --semantic-cache /path/to/fever-semantic.sqlite \
  --output causal_memory_control/checkpoints/fever_semantic_mlp.json

python -m causal_memory_control.jev_gate \
  --input /path/to/train/seed0/branches.jsonl /path/to/train/seed1000_retest/branches.jsonl \
  --memory-dir /path/to/fever/g-memory --metric success \
  --mode jev_direct --cache /path/to/fever-jev.sqlite \
  --output causal_memory_control/checkpoints/fever_jev_direct.json
```

FEVER 评测时，沿用原有完全相同的 held-out slice、snapshot、采样设置和 baseline
cache；`--mode learned` 加 MLP checkpoint，或 `--mode jev_direct/jev_decomposed`
加对应 Jev checkpoint。显式使用 checkpoint 开发集建议的 `kappa`、`threshold`
（Jev threshold 自动读 checkpoint）和 `recommended_max_drops`。
手动 FEVER 调用的错误恢复按原 `--resume` 继续；选择性重试加
`--resume --retry-jev-errors`。统一矩阵只需在原命令后加 `--retry-jev-errors`，
它自动为未完成的评测添加 resume。

## 上传代码

本轮需提交以下文件，不要提交数据、缓存、模型、实验结果或 API Key：

```bash
git add causal_memory_control/README.md \
  causal_memory_control/DATASET_GATE_MATRIX.md \
  causal_memory_control/HOTPOTQA_SEMANTIC_JEV.md \
  causal_memory_control/dataset_adapters.py \
  causal_memory_control/run_gate_matrix.py \
  causal_memory_control/semantic_features.py \
  causal_memory_control/neural_estimator.py \
  causal_memory_control/jev_api.py \
  causal_memory_control/jev_gate.py \
  causal_memory_control/run_hotpotqa_gate_matrix.py \
  causal_memory_control/train_gate.py \
  causal_memory_control/runtime_gate.py \
  causal_memory_control/hotpotqa_experiment.py \
  causal_memory_control/fever_experiment.py \
  causal_memory_control/jsonl.py \
  causal_memory_control/run_hotpotqa_gate_comparison.py \
  causal_memory_control/compare_gate_runs.py \
  causal_memory_control/tests/test_semantic_jev.py \
  causal_memory_control/tests/test_dataset_matrix.py
git commit -m "Unify FEVER and HotpotQA semantic and Jev gate comparisons"
git push
```

服务器拉取整个提交：`git pull --ff-only`，然后执行上面的服务器命令。
