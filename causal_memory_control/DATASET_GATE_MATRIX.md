# 同一个入口切换数据集

统一入口：`python -m causal_memory_control.run_gate_matrix`。
通过 `--dataset hotpotqa` 或 `--dataset fever` 切换数据加载、冻结 memory 的
构建和校验、反事实采集、评测环境以及报告指标。训练、开发集策略选择、公共
baseline、进度条、缓存和断点恢复共用同一套流程。

| 参数 | 数据与任务 | estimator 监督 | 最终报告 |
| --- | --- | --- | --- |
| `--dataset hotpotqa` | HotpotQA distractor JSON；沿用离线多跳 QA 环境 | USE/DROP 的 answer F1 差值（`team_score`） | answer F1、EM、配对增益及 95% CI |
| `--dataset fever` | FEVER JSONL；沿用离线 SUPPORTS/REFUTES 二分类环境 | USE/DROP 的正确率差值（`success`） | accuracy、completion rate、配对增益及 95% CI |

FEVER 的 NOT ENOUGH INFO 样本不进入当前实验；本次封装没有增加三分类或在线
Wikipedia 证据检索。两种数据集都复用各自原有的 GMemory/MacNet 执行方式。

默认比较原生 GMemory 与 `linear`、`semantic_linear`、`semantic_mlp` 三种 Gate，
共四行结果。Jev 对比暂时关闭，默认不调用 Jev，也不需要 `TYPESAFE_API_KEY`。
保留 `jev_direct`、`jev_decomposed` 供以后显式开启。语义 MLP 和可选 Jev 的
模型、loss、请求协议及阈值含义见 [HOTPOTQA_SEMANTIC_JEV.md](HOTPOTQA_SEMANTIC_JEV.md)。
两种 memory（trajectory、insight）的所有检索位置均采集和判别；最多删除一条，
是否删除由各方法的预测和开发集策略决定。

## 服务器运行

从仓库根目录运行，先激活现有环境。模型地址和数据路径改成服务器实际值。

```bash
conda activate gmemory
cd /mnt/mydata/hjq/casual-gmemory
export OPENAI_API_KEY=ollama
python -c 'import torch, sentence_transformers, langchain_chroma, tqdm; print("dependencies ready")'
```

下面的默认命令只运行本地 estimator。也可显式写
`--arms linear,semantic_linear,semantic_mlp` 固定当前对比组。

先看计划：下面的命令不会构建 memory、写实验结果或调用 LLM/Jev。

```bash
python -m causal_memory_control.run_gate_matrix \
  --dataset hotpotqa \
  --data data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
  --endpoint http://127.0.0.1:11436/v1 --model qwen2.5:7b \
  --plan-only
```

HotpotQA 正式实验：

```bash
python -u -m causal_memory_control.run_gate_matrix \
  --dataset hotpotqa \
  --data data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
  --endpoint http://127.0.0.1:11436/v1 --model qwen2.5:7b
```

FEVER 使用同一入口，只改数据集名称和数据文件：

```bash
python -u -m causal_memory_control.run_gate_matrix \
  --dataset fever \
  --data data/fever/fever_dev.jsonl \
  --endpoint http://127.0.0.1:11436/v1 --model qwen2.5:7b
```

```bash
python -m causal_memory_control.run_gate_matrix --list-datasets
python -m causal_memory_control.run_gate_matrix --dataset fever --help
```

旧的 `run_hotpotqa_gate_matrix` 命令仍然可用，与 `--dataset hotpotqa` 使用相同
配置和实验身份，已有一致配置的 HotpotQA 矩阵可以继续运行。

## 样本规模与目录隔离

默认 `--support-count 50 --evaluation-count 500 --training-fraction 0.8`：
50 个样本用于 memory 构建、400 个用于 Gate 诊断、100 个用于最终测试；诊断
样本内部再按问题划出 20% 开发集。support、诊断、最终测试题目互不重叠。

`--evaluation-count` 包含诊断和最终测试，不能解释为最终测试数量。
计划会报告采集分支数和最终推理规模；默认最多 19,200 条诊断团队分支，
团队分支可能包含多次 LLM 调用。

| 数据集 | 默认 frozen memory | 默认反事实数据 | 默认比较结果 |
| --- | --- | --- | --- |
| HotpotQA | `causal_diagnostic/memory_snapshots/hotpotqa_gate_support50_7b_v2/g-memory` | `causal_diagnostic/results/hotpotqa_gate_training_7b_v2` | `causal_memory_control/results/hotpotqa_gate_semantic_jev_v3` |
| FEVER | `causal_diagnostic/memory_snapshots/fever_gate_matrix_support50_7b_v1/g-memory` | `causal_diagnostic/results/fever_gate_matrix_training_7b_v1` | `causal_memory_control/results/fever_gate_semantic_jev_v1` |

FEVER 新目录避免与此前 binary v5 实验混用。已有 HotpotQA memory/诊断数据
配置相同会校验后复用。切换数据集不能在同一个结果目录里续跑；配置、数据源、
snapshot、checkpoint 和候选覆盖范围均检查一致性。

自定义规模可加 `--support-count`、`--evaluation-count`、`--training-fraction`。
FEVER 的 support/evaluation 总数必须为偶数，每种标签各取一半，且每种标签
都要有足够样本。`--use-all-data` 会将有效样本分到三个互不重叠的集合；
FEVER 的全部有效样本必须具有相等的 SUPPORTS/REFUTES 数量，否则明确报错，
不会悄悄丢掉多数类。

改变规模、模型、方法组或策略设置时，指定新的 `--memory-dir`、
`--diagnostic-results`、`--matrix-dir`，先用 `--plan-only` 检查规模。若仅增加
Gate 方法，可在新矩阵中复用配置一致的 memory/diagnostic，并用
`--baseline-cache-seed-from <旧矩阵>/native_gmemory/llm_cache.sqlite` 复用 baseline。
矩阵的 checkpoint 自动存入 `<matrix-dir>/checkpoints/<arm>.json`。

如果旧矩阵已经登记了包含 Jev 的方法组，这次关闭 Jev 后需选择新的
`--matrix-dir`，例如 `causal_memory_control/results/hotpotqa_semantic_only_v3`。
保持原来的 memory/diagnostic 路径可复用采集结果，并按上面的
`--baseline-cache-seed-from` 复用 baseline 缓存；原矩阵产物保留。
以后恢复 Jev，需要显式加
`--arms linear,semantic_linear,semantic_mlp,jev_direct,jev_decomposed`，
设置 `TYPESAFE_API_KEY` 并选择新的矩阵目录。

## 进度、恢复与结果

中断后重新执行原命令即可：完成阶段校验后跳过；反事实采集按分支恢复，
评测按问题恢复，成功请求使用缓存。MLP 拟合中断只会重新执行该拟合阶段。
进度条显示阶段和子任务进度；详细异常在 `matrix_progress.json` 与 `logs/*.log`。

显式启用 Jev 时，HotpotQA 和 FEVER 均支持 Jev 错误任务重试：修好网络、Key 或模型权限后，
原命令追加 `--retry-jev-errors`。只移出报错的评测任务并重跑，旧行备份到
`memory_gate.jsonl.before-jev-retry-*`；成功任务不重跑。开发集 Jev 报错时，
重跑原命令会使用成功请求缓存继续校准。

| 文件 | 用途 |
| --- | --- |
| `matrix_results.md/csv/json` | 所选方法（默认四组）的指标、相对 baseline 增益、配对 CI、保留率及 DROP 改善/伤害数 |
| `matrix_manifest.json` | 数据集与实验配置身份 |
| `matrix_progress.json`、`logs/*.log` | 进度、阶段、命令、具体错误、traceback 与日志末尾 |
| `<arm>/comparison.json` | 逐任务配对比较及一致性校验 |
| `<arm>/memory_gate.jsonl` | 题目、memory 决策/内容、最终结果及 Jev 使用情况 |
| `<arm>/summary.json`、`checkpoints/*.json` | 汇总指标、训练来源及开发集策略 |

所有方法复用同一 baseline。无 DROP 的问题必须与 baseline 的预测、reward、
done 一致，原始 retrieval fingerprint 也必须一致；不一致时停止并给出任务
ID。Jev 请求失败在线默认 KEEP，正式比较会报告无效，修复重试后才能完成矩阵。

## 扩展其他数据集

`dataset_adapters.py` 定义 `DatasetAdapter` 和 `MatrixMetric`，内置注册表提供
FEVER 与 HotpotQA。第三个数据集可写一个返回 `DatasetAdapter` 实例的 factory：

```bash
python -u -m causal_memory_control.run_gate_matrix \
  --dataset my_dataset \
  --dataset-adapter my_package.my_dataset:create_adapter \
  --data /path/to/my_dataset.json \
  --matrix-dir causal_memory_control/results/my_dataset_gate_v1
```

`my_package` 必须已安装或在 Python 路径中；factory 是不带参数的函数，返回
对象的 `name` 必须等于 `--dataset`。这只是扩展入口示例，仓库尚未内置
`my_dataset`，不能只更换 `--data` 就使用任意文件。

| adapter 负责 | 需要实现的行为 |
| --- | --- |
| 数据与划分 | loader、合法样本、稳定 task ID、互不重叠的 support/诊断/test 集 |
| 原生环境 | task/state 渲染、任务工具、正确的 reward/done 定义 |
| 反事实采集 | 冻结 memory，匹配的 USE/DROP 分支、候选覆盖与 provenance manifest |
| checkpoint 校验 | 当前 dataset/版本、source/snapshot、训练 ID 与最终测试不重叠 |
| 命令与完成检查 | 对应 diagnostic/evaluation runner 的 CLI 和有效结果判定 |
| 指标映射 | 将 summary 与公共配对比较的字段映射到 `MatrixMetric` |

公共 trainer 读取匹配分支，公共 comparer 支持 reward/done 以及当前 QA 的
answer F1/EM 字段。新任务可把所需标量得分定义为 reward，复用增益及 CI；若
需要额外专属指标，还需扩展 comparer，不能只增加表格列。adapter 的 `version`
用于锁定任务语义，改变 loader/env/scorer 时应更新版本并使用新产物目录。

## 上传代码

包含本轮统一封装和此前语义 MLP/Jev 的改动。仅提交这些代码、测试和说明：

```bash
git add causal_memory_control/README.md \
  causal_memory_control/DATASET_GATE_MATRIX.md \
  causal_memory_control/HOTPOTQA_SEMANTIC_JEV.md \
  causal_memory_control/dataset_adapters.py \
  causal_memory_control/run_gate_matrix.py \
  causal_memory_control/run_hotpotqa_gate_matrix.py \
  causal_memory_control/run_hotpotqa_gate_comparison.py \
  causal_memory_control/semantic_features.py \
  causal_memory_control/neural_estimator.py \
  causal_memory_control/jev_api.py \
  causal_memory_control/jev_gate.py \
  causal_memory_control/train_gate.py \
  causal_memory_control/runtime_gate.py \
  causal_memory_control/hotpotqa_experiment.py \
  causal_memory_control/fever_experiment.py \
  causal_memory_control/jsonl.py \
  causal_memory_control/compare_gate_runs.py \
  causal_memory_control/tests/test_semantic_jev.py \
  causal_memory_control/tests/test_dataset_matrix.py
git commit -m "Unify FEVER and HotpotQA semantic and Jev gate comparisons"
git push
```

服务器从仓库根目录拉取：

```bash
git pull --ff-only
conda activate gmemory
```

数据、模型权重、API Key、SQLite 缓存和实验结果不在这次 git 提交中。
