# Semantic ST：MPNet 独立适配调参

入口：`python -m causal_memory_control.run_st_mpnet_tuning`。

这里只适配**冻结 MPNet 的 ST 小预测头和输入分块**。MiniLM 不参与重新训练或调参，旧 MiniLM 和原始 MPNet 实验都是只读对照。默认新结果目录是 `causal_memory_control/results/st_mpnet_tuning_qa_v1`，不嵌套在旧目录中。本功能只新增入口，不修改旧训练、编码、评测模块或 ALFWorld 的代码。

## 实验边界

- 仅支持 FEVER、HotpotQA。默认两者一起运行，各训练自己的预测头，从两者开发集共同选择一个共享的参数配置编号。
- 复用原 checkpoint 登记的原始 USE / SINGLE DROP 诊断标签、冻结记忆、任务级 fit/dev 划分。没有重新采集；没有 CE 计算；没有训练编码器。
- 旧 checkpoint 的 fit/dev 可以只覆盖预留训练池的一个子集；调参严格复用这个子集，不补采、不扩大训练范围、不重新划分。计划中的 `partition_counts` 分别报告训练池、旧头实际使用、未使用、support 和 test 的任务数。仍拒绝池外任务、fit/dev 重叠、任何 support/test 泄漏及原评测任务不一致；出错时会列出冲突数量和前 10 个任务 ID。
- 输入分块仍是原实现的“逐块编码、平均、归一化”，改变的是 tokenizer token 窗口长度。它不会改变 GMemory 的检索编码器、检索列表或生成模型上下文窗口。
- FEVER 沿用原保守删除规则和已保存的 kappa / delta / 删除预算；HotpotQA 沿用 `positive_unlimited`，保留预测 U > 0 的条目。调参不搜索删除规则，也不重新校准这些数值。
- `tune` 只训练小头和计算 dev 的效用预测误差，不调用 Ollama、不生成测试答案、不读取测试结果日志。`evaluate` 是显式的另一个阶段，只评测共同选中的配置，不逐一测试所有配置。
- 配置中未覆盖的参数，包括 ensemble 数量、种子、任务批大小、InfoNCE 温度和正负样本阈值，均继承各数据集的原 ST checkpoint。不同数据集的头独立拟合，共享的是候选表和被选中的配置编号。

已有测试结果此前已经被查看，当前数据集删除规则也参考过此前结果，因此后续测试对比应报告为**探索性适配实验**。本入口保证这一次程序选参使用开发集，不能把已经看过的旧测试集重新称作完全未接触的新测试集。

## 默认候选表

默认 10 个配置，每个数据集最多拟合 10 个 bootstrap 集成预测头。`original` 是冻结 MPNet 配合原有小头超参数的复现锚点，不会替换已保存的原始 MPNet checkpoint。

| 配置 | 相对原 ST 只改变的项目 |
| --- | --- |
| original | 无；使用 MPNet，其他参数和原 ST 一致 |
| lr_3e4 | learning_rate = 0.0003 |
| l2_1e3 | l2 = 0.001 |
| beta_0 | InfoNCE 权重 beta = 0 |
| beta_003 | beta = 0.003 |
| beta_03 | beta = 0.03 |
| wide | hidden_sizes = [256, 128] |
| epochs_20 | epochs = 20 |
| chunk_128 | 分块长度 = 128 tokens |
| chunk_320 | 分块长度 = 320 tokens |

这是有限的单因素对照表，**不是全部参数的笛卡尔积，也不会自动把多个单项最优值拼起来**。beta=0 是明确关闭 InfoNCE 的消融对照，并不代表完整的 U + InfoNCE 方法。不同分块的特征在不同缓存目录中保存；同一分块的特征在多个小头之间复用。显存/内存占用按一个数据集、一个分块处理，不同时加载所有配置。

可以在运行前通过 `--trials-json path/to/trials.json` 指定 JSON 数组，元素格式如 `{"id":"lr_l2","head":{"learning_rate":0.0003,"l2":0.001},"chunk_tokens":192}`。第一项必须为 `{"id":"original","head":{}}`。`head` 仅接受 `learning_rate`、`l2`、`beta`、`hidden_sizes`、`epochs`。配置一旦登记到结果目录不能更改；新候选表需另开输出目录。不要在查看测试成绩后反复修改表格并称其为独立测试选参。

## 如何共同选参

所有配置都在原 fit 任务上训练，在原 dev 任务上计算 `U = Q_use - Q_drop` 的误差。不给 FEVER 的大量零效用候选以样本量优势：

1. 按真实效用将 dev 候选分为负、精确零、正三组。
2. 每组先在每个任务内平均平方误差，再在任务间平均；最后对该数据集实际出现的类别等权平均，得到 class-balanced MSE。
3. 将每个配置的该项误差除以本数据集 `original` 的对应误差，分母下限为 1e-8。
4. FEVER、HotpotQA 的上述相对误差等权平均，取最小值。同分按预先登记的候选顺序，`original` 在第一位。

开发集结果中的 0.9 代表这项误差相对原配置约降低 10%，**不是 90% 任务成功率**。同时报告整体 utility RMSE、逐类别候选数/任务数/误差和非零效用符号准确率。这个目标是适配预测头的代理指标，不保证最终 Accuracy / F1 提升，不能将单条效用误差直接当成同时删除多条记忆的真实收益。

## Git 同步

需要新增的文件只有：

- `causal_memory_control/run_st_mpnet_tuning.py`
- `causal_memory_control/ST_MPNET_TUNING.md`
- `causal_memory_control/tests/test_st_mpnet_tuning.py`

依赖此前已有的 `run_st_encoder_benchmark.py`、`st_encoder_evaluate.py` 等配套模块，服务器应已同步原 MPNet 实验版本。无需上传结果、模型或修改旧源码。下面所有 shell 命令均为单行。

本地终端确认差异后提交这三个文件（不要使用 `git add .`）：

```bash
cd /Users/xiaobaobei/gmemory
```

```bash
git add causal_memory_control/run_st_mpnet_tuning.py causal_memory_control/ST_MPNET_TUNING.md causal_memory_control/tests/test_st_mpnet_tuning.py
```

```bash
git commit -m "Add isolated MPNet ST dev tuning for QA"
```

```bash
git push origin main
```

服务器可以仅取出这三个新增文件，避免合并覆盖旧实验的本地改动。若服务器上已经存在这三个同名文件，先查看/保留自己的修改再执行 `restore`：

```bash
cd /mnt/mydata/hjq/casual-gmemory
```

```bash
git fetch origin main
```

```bash
git restore --source=origin/main -- causal_memory_control/run_st_mpnet_tuning.py causal_memory_control/ST_MPNET_TUNING.md causal_memory_control/tests/test_st_mpnet_tuning.py
```

```bash
conda activate gmemory
```

## 服务器执行

现有 MPNet 模型目录：`/mnt/mydata/hjq/models/all-mpnet-base-v2`。实验强制离线加载，不自动下载模型。

先检查计划：只读源数据与清单，不加载编码器、不调用模型服务、不创建结果目录。

```bash
python -u -m causal_memory_control.run_st_mpnet_tuning --datasets fever,hotpotqa --semantic-model /mnt/mydata/hjq/models/all-mpnet-base-v2 --output-dir causal_memory_control/results/st_mpnet_tuning_qa_v1 --stage plan
```

只进行开发集调参。编码和小头训练默认在 CPU，限制线程；不会启动或停止 ALFWorld：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -X faulthandler -m causal_memory_control.run_st_mpnet_tuning --datasets fever,hotpotqa --semantic-model /mnt/mydata/hjq/models/all-mpnet-base-v2 --output-dir causal_memory_control/results/st_mpnet_tuning_qa_v1 --train-device cpu --stage tune --refit-head
```

查看开发集排名和共同选中的配置：

```bash
cat causal_memory_control/results/st_mpnet_tuning_qa_v1/dev_results.md
```

调参结束且生成服务资源可用后，只测试选中的配置：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -X faulthandler -m causal_memory_control.run_st_mpnet_tuning --datasets fever,hotpotqa --semantic-model /mnt/mydata/hjq/models/all-mpnet-base-v2 --output-dir causal_memory_control/results/st_mpnet_tuning_qa_v1 --train-device cpu --stage evaluate
```

```bash
cat causal_memory_control/results/st_mpnet_tuning_qa_v1/results.md
```

最终表格列出 Always Use、原 MiniLM（同删除规则）、原 MPNet（同删除规则）、调参后 MPNet 的 Accuracy 或 F1 / EM。可选同规则对照缺失时显示 `—`，不会把另一种删除规则的分数混入编码器对比。

也可打印现有开发集和最终结果，不启动训练/评测：

```bash
python -u -m causal_memory_control.run_st_mpnet_tuning --datasets fever,hotpotqa --semantic-model /mnt/mydata/hjq/models/all-mpnet-base-v2 --output-dir causal_memory_control/results/st_mpnet_tuning_qa_v1 --stage report
```

默认原目录可通过同名参数改写，但同一输出目录的每次运行必须保持参数不变：

| 参数 | 默认目录 | 用途 |
| --- | --- | --- |
| --qa-root | results/all_trained_gates_with_old_st_retest_v2 | 原 MiniLM checkpoint、划分及 Always Use；结果路径前缀均为 causal_memory_control/ |
| --mpnet-reference-root | results/st_mpnet_qa_v1 | 最终评测时可选读取的原始 MPNet 同规则对照 |
| --rule-reference-root | results/st_positive_unlimited_retest_v1 | 最终评测时可选读取的 MiniLM 新规则对照 |

原始 MPNet 对照必须与本次使用相同真实编码器权重、原头超参数和删除参数，否则拒绝混作同条件对照。最终评测会核对每题检索候选及生成控制条件，优先从旧日志配套的缓存复制可复用答案；不同删除集合产生的新提示仍可能调用原 Ollama endpoint。隐藏 Python 的 GPU 不会隔离 Ollama 服务。`tune` 不争用 Ollama，但仍使用 CPU 和内存；`evaluate` 与 ALFWorld 共用服务时仍会竞争资源。

## 产物与中断续跑

- `manifest.json`：固定候选表、源 checkpoint/诊断/源码散列、开发集选择规则。
- `fever|hotpotqa/features/chunk_*/`：独立向量缓存及真实模型身份；本地模型文件散列或 Hub commit 在读缓存前核对。
- `fever|hotpotqa/trials/<id>/`：独立 checkpoint、训练诊断、损失曲线数据、dev 指标和完成记录。
- `selection.json`、`dev_results.md`：仅由 dev 指标生成的共享配置选择与完整排名。
- `evaluation_manifest.json`：最终评测的源结果和选择记录散列。
- `fever|hotpotqa/semantic_st/`：只有所选配置的测试日志、配对差值和 bootstrap 区间。
- `summary.json`、`results.md`：最终任务分数。

中断后重发同一命令。完整小头验证后跳过，不完整的小头从头拟合；其余已完成配置保留。重新加载分支时可复用向量缓存。显式评测阶段不训练任何头，且只允许完整调参后已登记的共享配置。原测试流程按已提交任务续跑。同一新输出目录有排他锁；不要对相同输出目录同时发两个命令。

更改候选表、源码、标签、参数或模型文件会拒绝混写，应开一个新输出目录。默认两个数据集等权共同选参，不要在原输出目录把 `--datasets` 改成单一数据集；单数据集探索需要另外的目录，不能称为跨数据集共同选择。

本地无网络验证：

```bash
gmemory_env/bin/python -X faulthandler -m unittest causal_memory_control.tests.test_st_mpnet_tuning causal_memory_control.tests.test_st_encoder_benchmark causal_memory_control.tests.test_st_policy_reevaluation causal_memory_control.tests.test_st_gate -q
```
