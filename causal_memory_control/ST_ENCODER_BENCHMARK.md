# Semantic ST：冻结语义编码器对照实验

新增入口 `run_st_encoder_benchmark` 复用已经保存的 FEVER、HotpotQA 和 PDDL 诊断分支，比较原 MiniLM 与另一个冻结 SentenceTransformer 编码器。默认候选是 `sentence-transformers/all-mpnet-base-v2`；也可以用 `--semantic-model` 指定完整本地模型目录或其他兼容编码器。

MPNet 是适合这些英文任务的较大语义编码器候选，不保证最终任务得分高于 MiniLM。原 MiniLM 通常输出 384 维，MPNet 输出 768 维；编码更慢、占用内存更大。ST 直接使用向量坐标和交互特征，更换编码器后必须训练新的小型预测头。即使换成同维度模型，也不能直接沿用旧头的权重。

## 控制变量和复用范围

| 项目 | 本次行为 |
| --- | --- |
| support、冻结 GMemory、诊断轨迹和 USE / SINGLE DROP 标签 | 复用原文件，不重新采集 |
| GMemory 检索编码器、检索数量、候选文本和顺序 | 保留，运行时核对原候选 |
| Semantic ST 文本编码器 | 改为指定的新冻结编码器，不训练编码器 |
| ST 预测头 | 按原任务级 fit/dev 划分重新训练，原 checkpoint 保留 |
| 集成数量、隐藏层、轮数、随机种子、MSE / InfoNCE 参数 | 从原 checkpoint 逐项继承 |
| 开发集策略搜索 | 不重新搜索，沿用原来的数值参数 |
| 删除规则 | 默认 FEVER 保守；HotpotQA / PDDL 删除所有预测 U ≤ 0 的项 |
| baseline 和其他方法 | 使用原结果 |
| 生成答案 / 执行环境 | 相同提示或删除集合尽量复用已测结果；新提示 / 新组合需要重新推理 |
| PDDL 无监督的空记忆折 | 继续标记未训练，直接评测空候选，不伪造 checkpoint |

因此这项实验需要**重新训练 ST 小预测头、重新评测 ST**，无需重跑数据采集或其他方法的训练。编码维数改变也会改变预测头第一层的输入宽度和参数量；它不是严格等参数量的比较。新结果比较同一删除规则下的两个编码器，避免同时改变规则。现有按数据集选定的规则是在看过此前结果后选定的，报告时应说明这一背景。

## 与正在运行的 ALFWorld 的关系

只上传本次新增的两个 Python 文件，不覆盖 `interactive_*.py`、`runtime_gate.py`、`semantic_features.py` 或旧训练模块，因而不会修改 ALFWorld 登记的源码。新输出使用独立目录，旧结果无需移动、删除或备份。

- FEVER / HotpotQA 可以单独运行。`fit` 阶段只读取旧标签、编码文本和训练小头，不调用 Ollama。下面命令限制 CPU 线程并隐藏当前 Python 进程的 GPU，适合先运行这一阶段。
- `evaluate` 阶段仍使用旧评测登记的生成模型和 endpoint。如果与 ALFWorld 共用 Ollama / GPU，会竞争服务容量，可能变慢或超时。隐藏 Python 的 GPU 不会隐藏独立 Ollama 服务的 GPU；要避免这种竞争，等 ALFWorld 结束后再评测，或安排独立服务及正式的配对复现验证。
- PDDL 与 ALFWorld 位于同一个原结果根目录。本入口对 PDDL 保留原目录锁；ALFWorld 续跑占锁时会拒绝启动 PDDL，不会终止 ALFWorld。先运行 QA，PDDL 待该锁释放后运行。

## 上传与准备模型

在本地电脑终端运行，每条命令均为一整行：

```bash
cd /Users/xiaobaobei/gmemory
```

```bash
scp causal_memory_control/run_st_encoder_benchmark.py causal_memory_control/st_encoder_evaluate.py hjq@czhpc:/mnt/mydata/hjq/casual-gmemory/causal_memory_control/
```

入口依赖之前已部署的 `reevaluate_st_policy.py`、`st_policy.py` 以及原实验训练、评测模块。不要为这项实验整体覆盖服务器仓库，以免影响 ALFWorld。

以下命令在服务器的 `gmemory` 环境运行：

```bash
cd /mnt/mydata/hjq/casual-gmemory
```

```bash
conda activate gmemory
```

服务器尚无 MPNet 时，先下载到独立目录。下载需要服务器能访问 Hugging Face；有完整离线模型时直接使用其目录即可：

```bash
python -c 'from huggingface_hub import snapshot_download; print(snapshot_download(repo_id="sentence-transformers/all-mpnet-base-v2",local_dir="/mnt/mydata/hjq/models/all-mpnet-base-v2",ignore_patterns=["onnx/*","openvino/*","*.h5","*.ot","pytorch_model.bin"]))'
```

实验阶段强制离线加载模型，不隐式下载权重。它记录实际 Hub commit 或本地模型文件散列，在读取向量缓存前核对权重身份。中断后不要更改同一目录的模型文件。

## 先运行 FEVER 和 HotpotQA

先检查计划。此命令只读原文件，不加载编码器或调用生成服务，也不创建结果目录：

```bash
python -u -m causal_memory_control.run_st_encoder_benchmark --datasets fever,hotpotqa --semantic-model /mnt/mydata/hjq/models/all-mpnet-base-v2 --output-dir causal_memory_control/results/st_mpnet_qa_v1 --stage plan
```

只训练两个新 ST 预测头，可先与 ALFWorld 采集同时进行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -X faulthandler -m causal_memory_control.run_st_encoder_benchmark --datasets fever,hotpotqa --semantic-model /mnt/mydata/hjq/models/all-mpnet-base-v2 --output-dir causal_memory_control/results/st_mpnet_qa_v1 --train-device cpu --stage fit --refit-head
```

生成服务可用后，使用已训练的新头做评测：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -X faulthandler -m causal_memory_control.run_st_encoder_benchmark --datasets fever,hotpotqa --semantic-model /mnt/mydata/hjq/models/all-mpnet-base-v2 --output-dir causal_memory_control/results/st_mpnet_qa_v1 --train-device cpu --stage evaluate
```

若资源已安排好，可以将训练命令中的 `--stage fit` 改为 `--stage all`，一次完成训练和评测。默认 `--stage plan`；执行训练时需要 `--refit-head`。这是训练新头的显式开关，不涉及重新采集。

默认原目录：

| 参数 | 默认目录 |
| --- | --- |
| `--qa-root` | `causal_memory_control/results/all_trained_gates_with_old_st_retest_v2` |
| `--interactive-root` | `causal_memory_control/results/pddl_alfworld_gate_expanded_v10_formatwarn_no_ce` |
| `--rule-reference-root` | `causal_memory_control/results/st_positive_unlimited_retest_v1` |

最后一个目录是可选的，用于读取原 MiniLM 在新删除规则下的已测成绩和缓存。若缺失且旧实验未测过同一删除集合，对照列显示 `—`，不会把不同规则的结果混作编码器对照。通过对应参数指定其他已有目录。

## ALFWorld 结束后运行 PDDL

使用另一独立输出目录，不在 QA 结果目录中改变数据集列表：

```bash
python -u -m causal_memory_control.run_st_encoder_benchmark --datasets pddl --semantic-model /mnt/mydata/hjq/models/all-mpnet-base-v2 --output-dir causal_memory_control/results/st_mpnet_pddl_v1 --stage plan
```

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -X faulthandler -m causal_memory_control.run_st_encoder_benchmark --datasets pddl --semantic-model /mnt/mydata/hjq/models/all-mpnet-base-v2 --output-dir causal_memory_control/results/st_mpnet_pddl_v1 --train-device cpu --stage all --refit-head
```

PDDL 沿用原实验的 Success 效用标签，额外报告最终 Success 和 Progress。新 ST 删除集合在原 action store 中已有测量时直接复用，没有时才执行新环境分支。新分支只保存到新目录。

## 结果与续跑

```bash
cat causal_memory_control/results/st_mpnet_qa_v1/results.md
```

```bash
cat causal_memory_control/results/st_mpnet_pddl_v1/results.md
```

`results.md` / `results.csv` 列出原 Always Use、原编码器在**同规则**下的结果、新编码器结果。`summary.json` 保存完整汇总；QA 的 `semantic_st/encoder_comparison.json` 保存配对差值和 bootstrap 区间，核对相同删除集合的生成结果；`checkpoint.json`、`training_summary.json` 保存新头、fit/dev 分布和真实训练损失；`manifest.json` 和 `encoder_identity.json` 保存输入与权重身份。

中断后使用同一条命令、同一模型和同一输出目录续跑。已有完整 checkpoint 会验证后复用；训练尚未保存 checkpoint 时会重新训练该数据集的小头。QA 按原评测流程跳过已提交任务，PDDL 按已保存 action 分支续跑。配置、源码、标签或模型文件变化时会拒绝混写，须使用新的输出目录。

本地验证（使用小型确定性编码器和模拟环境，不下载或调用模型）：

```bash
gmemory_env/bin/python -X faulthandler -m unittest causal_memory_control.tests.test_st_encoder_benchmark causal_memory_control.tests.test_st_policy_reevaluation causal_memory_control.tests.test_st_gate -q
```
