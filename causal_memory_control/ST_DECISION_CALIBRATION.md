# 现有 Semantic ST 的开发集诊断与单条删除校准

这是一个独立的决策规则对照：复用原 MiniLM、原 MPNet、开发集调参选中的 MPNet 三个已训练 checkpoint，读取原来的开发集配对标签。**不重新采集，不训练预测头，也不更新编码器。** 原实验的权重、规则和结果均保留。

新增的五个文件：

- `causal_memory_control/run_st_decision_calibration.py`：独立入口。
- `causal_memory_control/st_decision_calibration.py`：离线选择、校准和诊断。
- `causal_memory_control/st_single_drop_evaluate.py`：单条删除评测与候选/预测一致性检查。
- `causal_memory_control/tests/test_st_decision_calibration.py`：回归测试。
- 本说明文件。

## 新对照具体改变什么

原有的 FEVER calibrated 规则和 HotpotQA positive_unlimited 规则保持原样。新增分支对每个数据集、每个已有 checkpoint，使用同一套候选规则与选择目标：

1. 读取该 checkpoint 原来的 `dev_task_ids` 或 `calibration_task_ids`，核对三个 checkpoint 的 fit/dev 划分、诊断标签和记忆快照一致，并排除 support/test 泄漏。
2. 使用冻结编码器和已有预测头计算每条候选的预测效用 `U_hat` 和不确定性 `sigma`。
3. 在预先固定的网格中选择规则：`kappa ∈ {0, 0.5, 1, 1.96, 3}`，`delta ∈ {0, .01, .025, .05, .1}`，另加 KEEP ALL，共 26 个候选。
4. 只有 `U_hat + kappa * sigma < -delta` 才能删除。在满足条件的候选中，按上式的值从小到大排序，**每题最多删除一条**；相同分值按原检索顺序处理。预测恰为零不会在 `kappa=delta=0` 时被删除。
5. 根据已有配对标签计算所选单次删除的收益 `Q_drop - Q_use = -U`，不删除的题收益记零，以全部开发集任务的平均收益选择规则。不是按候选数加权，不再以 MSE 选择删除阈值。
6. 没有正收益的候选规则时选择 KEEP ALL。收益相同时优先更少删除，再优先更大的 `delta`、`kappa`。选择结果写入独立 `policy.json`，不改 checkpoint。

FEVER 的收益是准确率变化，HotpotQA 的收益是 Answer F1 变化；HotpotQA 的 EM 只报告，不用来选择规则。每个数据集/已有 checkpoint 可选出不同参数，但网格、目标与回退逻辑相同。没有跨编码器选择“测试冠军”。

HotpotQA 原来的 positive_unlimited 可能同时删除多条。单条干预标签不能推导这些联合干预的效果，所以开发集报告中原规则涉及多删时，收益标为 `unmeasured`；不会把多条单删收益相加。测试阶段才运行实际的新提示并比较最终分数。

## Git 同步

在本地电脑执行，每条命令都是单行。只提交本次五个新增文件：

```bash
cd /Users/xiaobaobei/gmemory
git add causal_memory_control/run_st_decision_calibration.py causal_memory_control/st_decision_calibration.py causal_memory_control/st_single_drop_evaluate.py causal_memory_control/tests/test_st_decision_calibration.py causal_memory_control/ST_DECISION_CALIBRATION.md
git commit --only -m "Add dev-only ST decision calibration comparison" -- causal_memory_control/run_st_decision_calibration.py causal_memory_control/st_decision_calibration.py causal_memory_control/st_single_drop_evaluate.py causal_memory_control/tests/test_st_decision_calibration.py causal_memory_control/ST_DECISION_CALIBRATION.md
git push origin main
```

在服务器执行。按文件同步可避开之前六个 ST 文件的本地修改冲突，也不会改动正在运行的 ALFWorld 使用的旧模块：

```bash
cd /mnt/mydata/hjq/casual-gmemory
git fetch origin main
git restore --source=origin/main -- causal_memory_control/run_st_decision_calibration.py causal_memory_control/st_decision_calibration.py causal_memory_control/st_single_drop_evaluate.py causal_memory_control/tests/test_st_decision_calibration.py causal_memory_control/ST_DECISION_CALIBRATION.md
conda activate gmemory
```

`git restore --source=origin/main` 只更新指定文件，不会把服务器分支整体切换到远端版本。再次同步这五个文件会覆盖它们的本地修改，请保留自己的修改后再同步。

## 完整运行顺序

默认从已完成的调参目录导入原数据目录、模型目录、已选中的 `wide` 等配置和三个 checkpoint；不用重新指定训练参数或模型地址。新输出使用独立目录 `st_decision_calibration_qa_v1`。

先预览计划，不创建结果目录、不加载模型、不调用 Ollama：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m causal_memory_control.run_st_decision_calibration --datasets fever,hotpotqa --tuning-root causal_memory_control/results/st_mpnet_tuning_qa_gpu3_v1 --output-dir causal_memory_control/results/st_decision_calibration_qa_v1 --stage plan
```

再运行离线开发集诊断和校准。为诊断分支临时 SQLite 指定数据盘目录，避免原先 `/tmp` 配额不足；此处只创建新临时目录，不清理任何旧文件：

```bash
mkdir -p /mnt/mydata/hjq/tmp/st_decision_calibration
TMPDIR=/mnt/mydata/hjq/tmp/st_decision_calibration CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m causal_memory_control.run_st_decision_calibration --datasets fever,hotpotqa --tuning-root causal_memory_control/results/st_mpnet_tuning_qa_gpu3_v1 --output-dir causal_memory_control/results/st_decision_calibration_qa_v1 --stage calibrate
```

编码器和小型预测头在 CPU 上推理，不占用 Ollama，也不使用 GPU。脚本强制 Hugging Face 离线模式：原 MiniLM 和 MPNet 均须已在本机缓存或原 checkpoint 记载的本地路径中。它会复制权重身份匹配的旧 MPNet 向量缓存到新目录；缺失向量重新编码，但不训练或采集轨迹。原 MiniLM 若没有历史权重哈希，脚本记录当前加载的固定提交，测试时再核对同一 checkpoint 的预测与原日志一致。

查看开发集结果：

```bash
cat causal_memory_control/results/st_decision_calibration_qa_v1/dev_diagnostics.md
python -m json.tool causal_memory_control/results/st_decision_calibration_qa_v1/selection.json
python -m json.tool causal_memory_control/results/st_decision_calibration_qa_v1/fever/tuned_mpnet/diagnostics.json
python -m json.tool causal_memory_control/results/st_decision_calibration_qa_v1/hotpotqa/tuned_mpnet/diagnostics.json
```

`diagnostics.json` 包含逐题选择记录，内容较长。只打印标签分布、分桶和各规则的汇总，可用下面的单行命令；把 `tuned_mpnet` 替换为 `original_minilm`、`original_mpnet` 即可检查另外两组：

```bash
python -c 'import json; from pathlib import Path; r=Path("causal_memory_control/results/st_decision_calibration_qa_v1"); [(print("==",d,"tuned_mpnet =="), print(json.dumps({k:v for k,v in json.loads((r/d/"tuned_mpnet/diagnostics.json").read_text()).items() if k!="policies"},ensure_ascii=False,indent=2)), [print(n,json.dumps({k:v for k,v in s.items() if k!="task_decisions"},ensure_ascii=False)) for n,s in json.loads((r/d/"tuned_mpnet/diagnostics.json").read_text())["policies"].items()]) for d in ("fever","hotpotqa")]'
```

开发集检查完成后，显式启动测试评测：

```bash
TMPDIR=/mnt/mydata/hjq/tmp/st_decision_calibration CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m causal_memory_control.run_st_decision_calibration --datasets fever,hotpotqa --tuning-root causal_memory_control/results/st_mpnet_tuning_qa_gpu3_v1 --output-dir causal_memory_control/results/st_decision_calibration_qa_v1 --stage evaluate
```

测试会依次评测两种数据集的三个已有 checkpoint，共六组新的规则对照。复用原 Always Use、原规则日志、原响应缓存；新提示没有缓存时会调用**原实验记录的 Ollama endpoint**。不会自动另开服务或停止 ALFWorld。离线校准可与 ALFWorld 同时运行；若测试共用同一 Ollama，仍会竞争服务资源，建议安排服务空闲时运行测试。

测试源目录要求三个旧规则的正式评测完整且响应缓存已关闭。运行前会核对参考 checkpoint；执行时核对检索文本、元数据、顺序及同一 checkpoint 的预测，检查删除预算；评测后做配对结果与相同删除集合的一致性检查。如果失败，不会用新日志覆盖旧实验。

看最终测试结果：

```bash
cat causal_memory_control/results/st_decision_calibration_qa_v1/results.md
python -m json.tool causal_memory_control/results/st_decision_calibration_qa_v1/summary.json
python -m json.tool causal_memory_control/results/st_decision_calibration_qa_v1/fever/tuned_mpnet/semantic_st/original_rule_comparison.json
python -m json.tool causal_memory_control/results/st_decision_calibration_qa_v1/hotpotqa/tuned_mpnet/semantic_st/original_rule_comparison.json
```

或者一次打印已生成的开发集与测试表格：

```bash
python -u -m causal_memory_control.run_st_decision_calibration --datasets fever,hotpotqa --tuning-root causal_memory_control/results/st_mpnet_tuning_qa_gpu3_v1 --output-dir causal_memory_control/results/st_decision_calibration_qa_v1 --stage report
```

## 结果怎么看

`dev_diagnostics.md` 对每个模型列出四项：

| 规则 | 用途 |
| --- | --- |
| `original_rule` | 原部署规则；多删的联合收益不推算 |
| `forced_lowest_predicted_u_one` | 不管正负都选预测效用最低的一条，用于诊断排序能力，不部署 |
| `negative_u_one` | 仅当预测效用严格小于零才删，最多一条，不加不确定性或安全边界 |
| `dev_calibrated_single` | 新增的开发集校准结果，最多一条，也可能 KEEP ALL |

重点查看：

- `mean_observed_gain`：所选单条删除相对保留全部的平均实测收益，包含没删的题。表格乘 100 后单位是百分点。
- `improved_tasks / harmed_tasks / neutral_drop_tasks`：删对、删坏和实测不变的题数。HotpotQA 还要看收益幅度，不能只数改善/损害题数。
- `best_observed_single_drop_mean_gain`：每题利用已测标签选最好的一条或不删的上界。它是事后且受采样噪声影响的参考值，不是可直接部署的策略。
- `mean_regret_to_observed_single_drop_oracle`：与该上界的差距。大差距而上界明显为正，说明识别/排序仍有改进空间。
- `utility_buckets`：预测效用分桶中，实际有害/中性/有用记忆各有多少。负效用桶实际有害比例高，才支持“预测负值可以指导删除”。
- `uncertainty_buckets`：模型分歧与实测标签的关系；不确定性不是严格统计置信保证。
- `selected_drop_gain_buckets`：新策略实际选中的删除，按保守预测收益 `-U_hat-kappa*sigma` 分桶。查看高分桶是否更常改善。
- `policy.json` 中 `max_drops=0` 表示开发集选择全部保留，不是脚本没有运行。

几种后续判断：

1. 单删上界本来就很小：当前开发集上的可利用空间有限，不能据此断言换编码器就会改善。
2. 单删上界较大，但最低预测 U 的选择经常损害：优先查效用监督噪声、排序/特征/预测头；单纯换更大的编码器未必有效。
3. 无约束的单删经常损害，而校准后删得更少且开发集收益变正：说明决策阈值可能是一个问题，再看测试能否维持这种变化。
4. 开发集校准有收益、测试没有：可能是开发集选择波动或泛化不足，需要进一步验证，不能继续根据同一测试集反复挑规则。

测试表格比较 Always Use、同一 checkpoint 的原规则、新增单条规则；`original_rule_comparison.json` 给出配对差值及 bootstrap 95% 区间。区间跨零时，不宣称已证实提高。开发集的网格最优收益也不能当作无偏泛化收益。

这些测试集之前已经查看过，本实验属于后续探索性对照；正式确认需要新的预留数据或预先固定协议的重复实验。

## 续跑与隔离

- `calibrate`、`evaluate` 中断后重用各自的同一条命令：已保存且哈希一致的开发集预测不重算，已完成测试任务通过原 QA runner 的 resume 逻辑跳过。
- 保持 `--datasets`、输入、网格、源码及输出目录一致；改协议时换新输出目录，不删除旧目录。
- 不要让两个终端同时写同一新输出目录；入口有非阻塞输出锁。
- 不涉及 PDDL 或 ALFWorld 的采集、训练、评测入口，不改变它们的进度文件。
- 源 checkpoint、标签、快照、校准结果和响应缓存均记录哈希；变更会阻止混用结果。

本地测试命令：

```bash
gmemory_env/bin/python -m unittest causal_memory_control.tests.test_st_decision_calibration -v
```
