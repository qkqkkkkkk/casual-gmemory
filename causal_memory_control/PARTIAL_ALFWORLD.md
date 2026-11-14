# 使用已完成的 ALFWorld 诊断任务训练和测评

当原诊断采集停止在 845/1000 时，可以建立独立的部分数据实验。
入口是 `python -m causal_memory_control.run_partial_alfworld`。
不修改原 manifest、不删除原目录，不调用 support 或 collect。
这不是完整 1000 任务训练协议的结果。

## 具体使用哪些内容

- 仅选择原 `alfworld/diagnostic/tasks/*.json` 中已完整保存的任务。
- 按原划分与完成集合取交集；fit 仍用于拟合，dev 仍用于校准与选取 epoch。
  845 个任务不能解释成“800 fit + 45 dev”，实际数量由预检输出。
- 核验每个任务所有重复的 KEEP ALL / SINGLE DROP 实测分支、配对种子、候选和汇总标签。
  缺分支、损坏记录、混入其他配置时停止，不把未完成任务当成零标签。
- 私有复制冻结 support 记忆和模型响应缓存，使用 SQLite backup 包含已提交 WAL 内容。
- 在新目录重新拟合三种 semantic gate 与 Local–Global Team U；不复制旧训练头。
  训练数据里的空候选保留为空观察，不能制造监督标签。
- 所有策略规则沿用原交互实验，包括原 Semantic ST 校准规则；不切换编码器或删除规则。
- support、全部原测试任务、primary_test_ids、test_cohorts 均保持不变。
  本次原配置中需评测 634 题，其中主测试集 valid_unseen 为 134 题，另有 500 题场景留出集。
- 新 manifest 与来源记录明确标记部分数据协议。原有 action key / outcome 不变，
  副本的 config/store identity 显式迁移并记录来源，不冒充原完整实验。
- 选中集合在首次准备时冻结。原目录以后补齐更多任务，新目录仍使用首次固定的子集。

## 上传与执行

在本地终端上传新入口（不需要上传权重或结果目录）：

```bash
scp /Users/xiaobaobei/gmemory/causal_memory_control/run_partial_alfworld.py hjq@czhpc:/mnt/mydata/hjq/casual-gmemory/causal_memory_control/
```

以下各命令都是单行，在服务器已激活的 gmemory 环境中逐条执行。
源实验在初次冻结期间应已停止。入口会检查实验锁，复制输入时也会检测来源文件变化。

```bash
cd /mnt/mydata/hjq/casual-gmemory
python -u -m causal_memory_control.run_partial_alfworld --source-dir causal_memory_control/results/pddl_alfworld_gate_expanded_v10_formatwarn_no_ce --output-dir causal_memory_control/results/alfworld_partial845_v1 --expected-completed 845 --train-device cpu --stage plan
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -X faulthandler -m causal_memory_control.run_partial_alfworld --source-dir causal_memory_control/results/pddl_alfworld_gate_expanded_v10_formatwarn_no_ce --output-dir causal_memory_control/results/alfworld_partial845_v1 --expected-completed 845 --train-device cpu --stage all
```

`plan` 不写结果，也不加载模型或访问 Ollama；实际运行会再次核验全部分支。
`all` 依次执行 Team U 标签核验、reward gates 训练、Team U 训练、测试、汇总。
保留原 Ollama 服务：现有 worker 在训练前也检查其身份，测试阶段会实际请求模型。
训练使用本地缓存中的原 MiniLM/BERT。这里选择 CPU 训练以减少与 Ollama 竞争 GPU，
但仍需要足够系统内存；CPU 选择并不能消除之前 SIGKILL 的所有可能原因。
若要用 GPU，首次创建新目录时可改为 `--train-device cuda:0`，以后沿用同一设备配置。

也可把最后的 `--stage all` 换成 `--stage train`，先只运行标签与训练；
随后分别运行 `--stage evaluate` 和 `--stage report`。其余参数保持相同。
`--stage prepare` 仅核验并复制输入，不加载模型。

中断后原样重跑 `--stage all`：复用该新目录已保存的 checkpoint 与完整测试任务/动作分支。
未完成训练方法可能需要重新拟合，不承诺按 minibatch 恢复。
不会回原目录补采剩余诊断任务，也不会重新选择子集。

## 查看结果

```bash
cat causal_memory_control/results/alfworld_partial845_v1/results.md
```

完整机器可读报告为 `results.json`，策略分测试集的 CSV 为
`alfworld/dataset_report/comparison.csv`。训练/评测日志位于 `alfworld/logs/`，
完整源配置、选中的 fit/dev ID、排除 ID、来源哈希及任务类型覆盖位于
`partial_diagnostic/`。

部分数据按原采集顺序完成，可能尚未覆盖某些任务类型。应该以预检中的
`completed_task_types` 核查这种偏差，不能把这次结果直接称作完整训练结果，
也不能仅凭任务数认定它与随机抽样 845 个诊断任务等价。
