# ST 单条删除评测的数值审计续跑

`single-drop comparison retrieval/predictions changed: 150648` 表示：当前任务的原始检索或 ST 预测，与旧实验参考记录不完全相同。它发生在记忆交给智能体之前，本身不是 Ollama 端口冲突、超时或 ALFWorld 资源冲突的证据。

旧入口对整个预测字典进行精确相等比较。相同权重的浮点推理也可能出现极小差异，因此这个比较过于严格。但旧报错没有显示差异字段，不能仅凭它确定服务器上是浮点差异还是检索、模型或输入确实改变。

新增的 `resume_st_single_drop_evaluation` 使用独立 worker 继续原评测：

- 检索指纹、候选数量与顺序、ID、文本、元数据、排名必须完全一致。
- `q_use`、`q_drop`、`utility`、`uncertainty` 必须有限，并满足固定的 `atol=1e-6, rtol=1e-6`。来源、校准状态、训练样本量等离散字段仍精确比较。
- 由零附近的效用推导的 `utility_class` 可随微小数值变化，但实际删除集合必须与“用参考预测执行同一份新策略”的结果完全一致。跨越阈值或改变最差候选排名仍会停止。
- 所有非精确匹配均保存逐字段报告。真实检索变化、明显预测变化或删除选择变化均在记忆暴露前停止。

原代码、manifest、开发集预测、校准策略、checkpoint 和已完成的测试记录不改写。新入口额外保存 `numeric_audit_resume.json`，记录本次数值审计协议、实现哈希以及原输入绑定。它不采集或训练，也不重新选择策略；未完成的测试题可能调用原 Ollama 服务。

## 上传文件

仅需新增四个文件，不需要覆盖旧入口或删除实验目录：

```text
causal_memory_control/resume_st_single_drop_evaluation.py
causal_memory_control/st_single_drop_audit_evaluate.py
causal_memory_control/tests/test_st_single_drop_audit_resume.py
causal_memory_control/ST_SINGLE_DROP_AUDIT_RESUME.md
```

## 在服务器续跑

在仓库根目录、原 `gmemory` 环境执行。输出目录及其他参数须与中断前相同，入口替换为新增模块。

```bash
mkdir -p /mnt/mydata/hjq/tmp/st_decision_calibration
TMPDIR=/mnt/mydata/hjq/tmp/st_decision_calibration CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m causal_memory_control.resume_st_single_drop_evaluation --datasets fever,hotpotqa --tuning-root causal_memory_control/results/st_mpnet_tuning_qa_gpu3_v1 --output-dir causal_memory_control/results/st_decision_calibration_qa_v1 --stage evaluate
```

如原命令还有其他路径参数，须一并保留。不要重新执行 `calibrate` 或删除 manifest 来绕过检查。

如仍有真实差异，报错会给出 `details=...` 路径。可直接查看该 JSON；任务 150648 的报告也可这样查找：

```bash
python -c 'import json; from pathlib import Path; p=Path("causal_memory_control/results/st_decision_calibration_qa_v1/fever/original_minilm/semantic_st/single_drop_audit"); [print(str(f),json.dumps(r,ensure_ascii=False,indent=2)) for f in p.glob("*.json") if str((r:=json.loads(f.read_text()))["task_id"])=="150648"]'
```

`status=tolerated_numeric_difference` 表示候选一致、差异在容差内且删除集合相同；`status=failed` 会显示具体检索字段或预测字段。若检索变化，应检查冻结库、候选顺序、编码器及 tokenizer 版本和输入特征，不能扩大容差绕过。

ALFWorld 无须因这条错误而中断。若两个实验实际访问同一个 Ollama 服务，会共享推理资源，可能增加排队时间；等待 ALFWorld 结束本身不能修复检索或预测的审计不一致。不要同时启动两个写入同一个 ST 评测输出目录的进程。

## 本地验证

```bash
gmemory_env/bin/python -m unittest causal_memory_control.tests.test_st_single_drop_audit_resume causal_memory_control.tests.test_st_decision_calibration causal_memory_control.tests.test_st_mpnet_tuning causal_memory_control.tests.test_st_encoder_benchmark causal_memory_control.tests.test_st_policy_reevaluation -v
```

测试使用本地小型数据和模拟智能体，不连接 Ollama 或下载编码器。覆盖旧精确比较失败的复现、数值容差、严格检索检查、阈值与排名变化拒绝、FEVER/HotpotQA 中断续跑，以及原校准结果和参考文件保持不变。
