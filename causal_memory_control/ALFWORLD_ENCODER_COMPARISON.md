# ALFWorld：原八策略 + 新编码器 Semantic ST

此入口使用已经完整保存的 845 个诊断任务，先完成原八策略，再增加独立的 MPNet Semantic ST，汇总九项对比。不会继续采集剩余 155 个任务，也不会更改原始采集目录。部分诊断训练结果不能当成完整 1000 任务训练结果。

## 比较内容

原八项为 `always_use`、`global_drop`、`random_drop`、`semantic_linear`、`semantic_mlp`、`semantic_st`、`local_global_team_u`、`oracle`。新增第九项是 `semantic_st_new_encoder`；没有为交互环境增加 CE proxy。

旧 `semantic_st` 保留 MiniLM 编码器。新 ST 改用冻结的 `all-mpnet-base-v2`，复用相同标签和原 fit/dev 归属，只重新训练小型 ST 集成预测头。沿用原头的隐藏层、集成数、轮数、损失、随机种子及开发集选定的数值删除参数，不重新搜索删除规则。新旧两版在 ALFWorld 都用原 `calibrated` 保守规则，不默认切换到无限删除。向量维度改变会改变第一层参数量，所以这不是等参数量比较。

测试集保持原 634 题，分别报告 `valid_unseen`（134，主结果）和 `train_scene_holdout`（500）。原 Oracle 仍使用原先规定的动作集合，不因新增 ST 扩大其动作菜单。原八项中已完成的完整 checkpoint 和任务结果会验证后复用；尚未写出 checkpoint 的训练会从该方法重新训练，不提供按 minibatch 断点恢复。

## 目录

| 用途 | 路径 |
| --- | --- |
| 原始、可以以后继续收集到 1000 的实验 | `causal_memory_control/results/pddl_alfworld_gate_expanded_v10_formatwarn_no_ce` |
| 原八策略的独立 845 任务实验 | `causal_memory_control/results/alfworld_partial845_v1` |
| 新编码器 ST 和九项汇总 | `causal_memory_control/results/alfworld_partial845_mpnet_v1` |
| 服务器上的完整本地 MPNet | `/mnt/mydata/hjq/models/all-mpnet-base-v2` |

三个实验目录必须互不包含。已创建的 845 子集不会随后自动扩大，原采集以后增长不改变它。若从未冻结子集且原采集数已不等于 845，命令会停止，避免悄悄改动样本数。应明确选择新的任务数和新输出目录。

## 通过 Git 同步

下面命令均为单独一行。先在本地 `/Users/xiaobaobei/gmemory` 提交这次改动，再推送；这里不使用 `git add .`，避免加入实验结果或其他工作。

```bash
cd /Users/xiaobaobei/gmemory
```

```bash
git add -- causal_memory_control/reevaluate_st_policy.py causal_memory_control/st_policy.py causal_memory_control/st_policy_evaluate.py causal_memory_control/ST_ENCODER_BENCHMARK.md causal_memory_control/run_st_encoder_benchmark.py causal_memory_control/st_encoder_evaluate.py causal_memory_control/run_alfworld_encoder_comparison.py causal_memory_control/ALFWORLD_ENCODER_COMPARISON.md causal_memory_control/tests/test_alfworld_encoder_comparison.py causal_memory_control/tests/test_st_encoder_benchmark.py causal_memory_control/tests/test_st_policy_reevaluation.py
```

```bash
git diff --cached --stat
```

```bash
git commit -m "Add ALFWorld MPNet ST comparison while preserving original policies"
```

```bash
git push origin main
```

然后在服务器中执行。先确认原采集和其他实验训练进程已停止；以下步骤不替你终止任何进程。

```bash
cd /mnt/mydata/hjq/casual-gmemory
```

```bash
conda activate gmemory
```

下面将此前冲突的六个文件、可能单独下载过的部分诊断入口及本次配套文件暂存为 Git stash。仅处理列出的代码和文档文件，不包含 `results/`、模型目录或其他未列文件。只传入实际存在的文件，避免旧版本缺少新文件时产生 pathspec 错误。

```bash
python -c 'import subprocess; from pathlib import Path; root=Path("causal_memory_control"); names="reevaluate_st_policy.py st_policy.py st_policy_evaluate.py ST_ENCODER_BENCHMARK.md run_st_encoder_benchmark.py st_encoder_evaluate.py run_partial_alfworld.py PARTIAL_ALFWORLD.md tests/test_partial_alfworld.py run_alfworld_encoder_comparison.py ALFWORLD_ENCODER_COMPARISON.md tests/test_alfworld_encoder_comparison.py tests/test_st_encoder_benchmark.py tests/test_st_policy_reevaluation.py".split(); paths=[str(root/n) for n in names if (root/n).is_file()]; subprocess.run(["git","stash","push","-u","-m","server-before-alfworld-mpnet","--",*paths],check=True) if paths else print("No matching files to stash")'
```

```bash
git pull --ff-only origin main
```

保留 stash 作为旧代码备份，不要立即 `git stash pop`，否则会把旧版本重新混回新代码。若拉取仍报告其他冲突，先核对那些具体文件，不能用 `git reset --hard` 或 `git clean` 清空实验目录。

```bash
python -m causal_memory_control.run_alfworld_encoder_comparison --help
```

## 查看计划并运行

先检查只读计划，不加载 MPNet，不调用 Ollama，不创建输出目录：

```bash
python -u -m causal_memory_control.run_alfworld_encoder_comparison --source-dir causal_memory_control/results/pddl_alfworld_gate_expanded_v10_formatwarn_no_ce --base-dir causal_memory_control/results/alfworld_partial845_v1 --output-dir causal_memory_control/results/alfworld_partial845_mpnet_v1 --expected-completed 845 --semantic-model /mnt/mydata/hjq/models/all-mpnet-base-v2 --train-device cpu --stage plan
```

完成原八策略、新 ST 头训练、新 ST 测评和九项汇总：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -X faulthandler -m causal_memory_control.run_alfworld_encoder_comparison --source-dir causal_memory_control/results/pddl_alfworld_gate_expanded_v10_formatwarn_no_ce --base-dir causal_memory_control/results/alfworld_partial845_v1 --output-dir causal_memory_control/results/alfworld_partial845_mpnet_v1 --expected-completed 845 --semantic-model /mnt/mydata/hjq/models/all-mpnet-base-v2 --train-device cpu --stage all
```

`cpu` 指训练预测头的设备。原八策略训练和 ST 文本编码还会加载已登记的本地语义模型，线程限制不会限制 Ollama 的 GPU 使用。生成服务仍使用原登记的模型和 endpoint，请保持原 Ollama 服务可用。没有完整离线 MPNet 权重时会停止，不隐式联网下载。

没有重新采集诊断数据不等于完全不调用生成模型：原八策略未测完的测试任务仍需执行环境；新 ST 优先复用原测评中的相同删除集合，没有测过的组合才在新目录执行。原保守规则通常最多删一条时，原单删动作的覆盖可以减少新 ST 的生成调用。

中断后原样重跑同一条 `--stage all` 命令。原采集目录保持不变，以后可使用原采集命令继续到 1000；不要同时对同一实验目录运行两个写入进程。

## 查看结果

```bash
cat causal_memory_control/results/alfworld_partial845_mpnet_v1/results.md
```

`results.csv`、`results.json` 包含原八策略和第九项结果；JSON 保留完整逐组统计和部分诊断说明。新头及其日志在 `encoder/alfworld/`，原 MiniLM 的 checkpoint 仍在 `alfworld_partial845_v1/alfworld/semantic_st/`。尚未完成九项汇总时查看根目录的 `progress.json` 及原八策略目录的 `alfworld/logs/`。

## 本地验证

测试使用小型确定性编码器和模拟环境，不下载模型、不调用真实服务。验证新头拟合、旧结果保留、相同动作结果复用、断点续跑、空记忆、来源锁及结果防混写。

```bash
gmemory_env/bin/python -m unittest causal_memory_control.tests.test_alfworld_encoder_comparison causal_memory_control.tests.test_st_encoder_benchmark causal_memory_control.tests.test_st_policy_reevaluation causal_memory_control.tests.test_st_gate causal_memory_control.tests.test_partial_alfworld -q
```
