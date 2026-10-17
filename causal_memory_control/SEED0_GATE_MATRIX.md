# 复用 seed0 的训练评测及反事实采集提速

## 先跑当前 FEVER seed0 的四方法对比

入口是 `causal_memory_control.run_seed0_gate_matrix`，默认只选 FEVER。
它读取已有全量实验的配置、冻结 memory 和完成的第一组分支数据，
不启动诊断采集、不启动 seed1000、不运行 Jev/ST/Random/Oracle。

不是直接把 seed0 的训练题成绩当测试成绩：先用 seed0 的 USE/DROP 标签训练
Linear、Semantic Linear、原三头 Semantic MLP，然后在原独立测试集上实际回答并评估。
任务划分、模型、endpoint、temperature、top-k 均从原 manifest 导入。
诊断任务再按 task 划分 fit/dev，三种模型使用相同划分；最终测试题不参与训练或选 kappa。

默认全量 FEVER 配置对应 50 support、5,292 诊断 fit+dev、1,324 最终测试任务。
以原 manifest 为准。seed0 的 `repeats=2` 仍保留 sample seeds 0、1，
只是省去 seed1000_retest 的 1000、1001；并非把同一道题的重复结果当独立训练任务。

在服务器原终端等待 seed0 的 `collection_progress.json` 变成 `completed`，
再按一次 Ctrl+C 停止原总流程，并等待子进程退出。seed0 达到 100% 后还可能正在写汇总，
仅进度条到 100% 不等于该组已完成。停止后再更新代码。
不要把原总流程放在后台继续采集，同时启动这个入口；不要删原结果或缓存。

```bash
cd /mnt/mydata/hjq/casual-gmemory
conda activate gmemory
git pull --ff-only origin main

python -u -m causal_memory_control.run_seed0_gate_matrix \
  --source-root causal_memory_control/results/full_gate_benchmark_v1 \
  --output-dir causal_memory_control/results/fever_seed0_gate_matrix_v1 \
  --datasets fever \
  --plan-only
```

确认路径后正式运行：

```bash
python -u -m causal_memory_control.run_seed0_gate_matrix \
  --source-root causal_memory_control/results/full_gate_benchmark_v1 \
  --output-dir causal_memory_control/results/fever_seed0_gate_matrix_v1 \
  --datasets fever
```

源路径若不是 `full_gate_benchmark_v1`，替换为当前原实验根目录；
必须包含原 `benchmark_manifest.json`，并保持 manifest 记录的原绝对路径可用。
不要为了修改 endpoint 而手改 manifest；入口会沿用原 endpoint。

运行前逐行验证 seed0 的所有 task/kind/rank/repeat/USE/DROP，以及原配置、种子和候选集合。
不完整的、重复的、上下文不匹配的分支会报出文件行号，不自动修补源数据。
训练通过临时 SQLite 按任务读取大日志，特征以 float32 压缩存储；标签和既有损失不变。
临时文件只属于本次训练，会在退出时清理。

baseline：原最终测试集 baseline 若已完成且配置验证通过，直接只读复用。
否则在新目录补跑最终测试集 baseline，再作为 gate 的回答缓存来源。
不会用 seed0 训练题的 USE ALL 当最终 baseline。

新结果目录与源目录独立。默认每题最多删一条；开发集无正收益时可选择 KEEP ALL。
FEVER 是去掉 NEI 的 binary Accuracy，不是官方 evidence-aware FEVER Score。

## 结果、日志与中断恢复

```text
fever_seed0_gate_matrix_v1/
  seed0_manifest.json
  experiment_plan.json
  seed0_progress.json
  seed0_results.md                     # 四方法成绩表
  seed0_results.json                   # 含配对比较详情
  logs/fever_*.log                     # 各模型训练/评测输出及 traceback
  fever/
    reused_inputs.json                 # seed0、manifest、snapshot 校验值
    baseline_selection.json
    reused_baseline.json               # 只在复用原 baseline 时出现
    native_gmemory/                    # 只在需要补跑 baseline 时出现
    semantic_cache.sqlite
    checkpoints/{linear,semantic_linear,semantic_mlp}.json
    matrix_results.{md,csv,json}
    linear/
    semantic_linear/
    semantic_mlp/
      training_summary.json            # fit/dev IDs、损失配置、校准详情
      progress.json
      run_manifest.json
      memory_gate.jsonl                # 逐题决策、memory 内容、U/sigma、最终结果
      summary.json
      comparison.json                 # 差值、CI、误删/改善、无 DROP 配对审计
      llm_cache.sqlite
```

中断后运行完全相同的正式命令即可。已保存且验证通过的 checkpoint 会跳过训练，
最终评测按任务恢复。模型拟合中断而 checkpoint 尚未保存时，仅重启该模型拟合，
不重新采集 USE/DROP。错误原因、阶段、退出码、日志路径与 traceback 保存在
`seed0_progress.json`。无 DROP 结果或原检索与 baseline 不一致时，配对审计会失败。

两种数据集均已完成第一组数据时可选 `--datasets fever,hotpotqa`；
若已经用一个数据集启动结果目录，增加数据集请选新输出目录。

## 已优化的 FEVER / HotpotQA 诊断采集

优化在两个原 `run_experiment` 入口生效，原命令及旧断点格式继续使用。
不改变原生检索排序、prompt、请求种子、候选或 USE/DROP 标签定义。

| 操作 | 原执行方式 | 新执行方式 |
| --- | --- | --- |
| 原生检索 memory / graph 加载 | 每题重新打开 | 每个采集进程打开一次，只读复用 |
| 分支内 memory | 每个分支打开 Chroma/graph/insights | 独立内存任务状态，读取冻结检索结果；不打开持久化 backend |
| 同题同 sample seed 的 USE ALL | 每个候选均执行完整 MAS 调度 | 仅 gate-training 模式共享同一个真实 USE ALL 结果 |
| 不同 repeat / sample seed | 分别采样 | 仍分别采样，不合并 |
| DROP | 每条候选各运行一次 | 仍对每条候选分别运行 |
| 断点完成查询 | 反复扫描全部历史分支 | 启动时建立索引，随后增量更新 |
| HTTP / SQLite | 每分支关闭 SQLite | 正常、报错、中断均关闭该分支 HTTP / SQLite |

每题 6 候选、2 repeats、2 conditions：逻辑分支仍为 **24 条**，
实际 MAS 调度由 **24 次变为 14 次**，减少 41.7%。每题约 25 次持久化初始化变为进程级一次。
原先回答缓存已复用重复 USE ALL 的 API 响应，因此不能把 41.7% 当作新的 API 或总耗时节省。
实际墙钟提速需服务器测量；未做真实服务吞吐基准。

每条新分支的 `execution` 标记是 `executed` 或 `shared_use_all`，
共享记录仍包含本候选的 task/candidate/condition/repeat/seed/outcome，
并记录来源 event/candidate，不把共享视为 LLM cache hit。
部分旧日志恢复时，只在任务、候选集合、run hash、sample seed 等均匹配时共享，
同题同 seed 的旧 USE ALL 结果若冲突则中止，避免掩盖数据问题。
完整 causal recipient probes 保持原调度，不共享 USE ALL。

`collection_diagnostics.json` 新增本进程实际执行/共享计数、检索/分支/共享耗时、
持久化 memory 加载次数及初始化/原生检索耗时；`collection_progress.json`
逐分支持续更新执行/共享与耗时计数。重启后这些统计从零重新计，不是整个实验累计量。

## 上传代码

以下显式文件清单包含当前训练入口依赖的前一版 ST 文件；并不启用 ST 实验。
不包含数据集、memory snapshot、结果大文件或其他研究目录的未跟踪改动。
在本地仓库执行；先检查 staged diff，再自行 commit/push：

```bash
git add \
  causal_diagnostic/collection_runtime.py \
  causal_diagnostic/memory.py \
  causal_diagnostic/fever_oracle_recipient/run_experiment.py \
  causal_diagnostic/hotpotqa_oracle_recipient/run_experiment.py \
  causal_diagnostic/oracle_recipient/seeded_client.py \
  causal_diagnostic/oracle_recipient/tests/test_seeded_client.py \
  causal_memory_control/source_run.py \
  causal_memory_control/run_seed0_gate_matrix.py \
  causal_memory_control/run_hotpotqa_gate_comparison.py \
  causal_memory_control/train_gate.py \
  causal_memory_control/runtime_gate.py \
  causal_memory_control/st_data.py \
  causal_memory_control/st_estimator.py \
  causal_memory_control/run_st_gate_benchmark.py \
  causal_memory_control/tests/test_collection_runtime.py \
  causal_memory_control/tests/test_seed0_gate_matrix.py \
  causal_memory_control/tests/test_st_gate.py \
  causal_memory_control/SEED0_GATE_MATRIX.md \
  causal_memory_control/ST_GATE_BENCHMARK.md \
  causal_memory_control/README.md

git diff --cached --stat
git commit -m "Add seed0-only gate evaluation and optimize counterfactual collection"
git push origin main
```

## 本地验证

使用真实 MAS scheduler 和 SQLite 回答缓存、模拟 HTTP 和原生检索 backend，验证两个数据集
优化前后任务/候选/seed/outcome 及发出的请求（含 prompt）相同，实际调度由 24 降为 14，
旧版部分日志可续跑且已有行保留。seed0-only 的两个数据集集成测试验证三种模型训练、
baseline 复用或补跑、配对审计、中断恢复、源文件未被改写，以及不完整分支被拒绝。
这些是正确性测试，不能代替服务器上的真实实验成绩与提速测量。
