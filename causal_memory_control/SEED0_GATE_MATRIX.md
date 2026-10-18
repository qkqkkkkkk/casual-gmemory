# 复用 seed0 的训练评测及反事实采集提速

## 先跑当前 FEVER seed0 的四方法对比

入口是 `causal_memory_control.run_seed0_gate_matrix`，默认只选 FEVER。
它读取已有全量实验的配置、冻结 memory 和完成的第一组分支数据，
默认不启动诊断采集，不启动 seed1000，不运行 Jev/ST/Random/Oracle。
只有显式设置 `--diagnostic-tasks N --collect-missing`，才采集新的训练诊断子集，见下文快速模式。

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

## 可选第五项：Always Global Drop（无检索记忆对照）

在原正式命令后添加 `--include-always-global-drop`，即可比较 baseline、Linear、
Semantic Linear、Semantic MLP、Always Global Drop 五种方法。
该选项适用于 FEVER 和 HotpotQA；不采集 seed1000，不需要额外的反事实训练数据。

Always Global Drop 的含义是：每题把本次检索出的 **全部 successful trajectory 和 insight**
从所有下游 Agent（含最终决策节点）的可见上下文中移除，不只是删除 top1。
它不受 learned gate 的单条删除预算或开发集 KEEP ALL 决策限制，不需要 utility estimator。
**题目自身、HotpotQA 提供的 context 和固定 few-shot 提示保持不变**；这是无检索记忆对照，
不是删除答题证据或所有提示信息。为确保没有未受控的 failed trajectory 残留，
此对照要求原配置 `failed_topk=0`（当前全量协议的默认值）。

HotpotQA 完成 seed0 采集后运行：

```bash
python -u -m causal_memory_control.run_seed0_gate_matrix \
  --source-root causal_memory_control/results/full_gate_benchmark_v1 \
  --output-dir causal_memory_control/results/hotpotqa_seed0_gate_matrix_v1 \
  --datasets hotpotqa \
  --include-always-global-drop
```

已有 FEVER 四方法成绩时，在原结果目录追加该参数补跑即可：

```bash
python -u -m causal_memory_control.run_seed0_gate_matrix \
  --source-root causal_memory_control/results/full_gate_benchmark_v1 \
  --output-dir causal_memory_control/results/fever_seed0_gate_matrix_v1 \
  --datasets fever \
  --include-always-global-drop
```

已完成且验证通过的 baseline、checkpoint 和三种 Gate 评测会复用；
仅新增 Always Global Drop 的最终任务推理与配对审计，不重采 seed0、不重训模型。
已有方法的配对统计/表格可能重新生成，但不会重新调用模型推理。
不能把 seed0 采集时“删除单条候选”的 `global_drop` 分支当作这个“删除全部”的新对照。

为兼容原四方法目录，`seed0_manifest.json` 中的训练/评测配置身份保持不变。
新增对照使用独立的、不可变的 `DATASET/plans/always_global_drop.json`，
记录原 baseline 日志哈希、每题检索指纹和全部待删除候选 ID，**不使用正确答案或 reward 选择删除项**。
推理时逐题核对实际检索、执行的删除集合及 plan 哈希，发现变化会中止。
新对照复用实际选中的 baseline cache，保持相同最终测试集、model、endpoint、seed 和温度。

控制结果位于 `DATASET/always_global_drop/`，包含逐题 `memory_gate.jsonl`、
`run_manifest.json`、`progress.json`、`summary.json`、`comparison.json` 与 `llm_cache.sqlite`。
该行会加入原来的 `seed0_results.md/json` 及 `DATASET/matrix_results.md/csv/json`，
展示成绩、相对 baseline 的差值/95% CI、keep rate、DROP 任务数和改善/损害任务数。
阶段日志仍在根目录 `logs/DATASET_always_global_drop_*.log`。

中断后重跑含 `--include-always-global-drop` 的同一命令；已保存任务不重跑。
若之后不加此参数，只运行/汇总原四种方法，新对照文件不会删除。
若只上传这次新增第五项的改动，需要以下三个文件：

```bash
git add \
  causal_memory_control/run_seed0_gate_matrix.py \
  causal_memory_control/tests/test_seed0_gate_matrix.py \
  causal_memory_control/SEED0_GATE_MATRIX.md
git diff --cached --stat
git commit -m "Add optional Always Global Drop control to seed0 matrix"
git push origin main
```

## 快速模式：1,000 题 × 2 repeats，最终测试仍保持原全量

针对 HotpotQA 反事实采集耗时较长，可以只采集原诊断区的 1,000 道任务。
**这不是全量训练，而是训练子集 + 原独立完整测试集。** 原来的测试集起点不变，
不会把剩余的未采样诊断任务加入测试，也不改变冻结 support。

| 配置 | 默认原全量计划 | 本快速模式 |
| --- | ---: | ---: |
| 冻结 support | 50 | 50（复用，不重建） |
| 诊断 fit+dev 任务 | 5,883 | 1,000 |
| 默认 fit / dev | 约 80% / 20% | 800 / 200 |
| repeats | 2 | 2，sample seeds 0、1 |
| 每题候选 | 3 trajectory + 3 insight | 相同，全部采集 |
| 逻辑分支 | 141,192 | 24,000 |
| 共享 USE ALL 后实际调度 | 最多 82,362 | 最多 14,000 |
| 最终测试任务 | 1,471 | **仍为 1,471** |
| 最终测试 offset | 5,883 | **仍为 5,883，而不是 1,000** |

以原 manifest 的真实数量为准。子集使用原随机划分后诊断列表的前 N 个 ID，
不按 reward、错误任务、效用或实验结果挑选。同一题的所有候选和 repeats 放在同一 fit/dev 组，
三种 estimator 使用相同的诊断数据与分组。

当前若还在跑全量采集，先在原终端按一次 Ctrl+C，并等子进程退出后再更新代码。
不要删除或手改原 `full_gate_benchmark_v1` 中的 manifest、分支或缓存。

服务器运行（先加 `--plan-only` 可查看数量，不会启动采集/训练）：

```bash
python -u -m causal_memory_control.run_seed0_gate_matrix \
  --source-root causal_memory_control/results/full_gate_benchmark_v1 \
  --output-dir causal_memory_control/results/hotpotqa_seed0_train1000_r2_v1 \
  --datasets hotpotqa \
  --diagnostic-tasks 1000 \
  --collection-repeats 2 \
  --collect-missing \
  --include-always-global-drop
```

这一条命令按顺序执行：仅采集/恢复这 1,000 题 seed0 → 完整性验证 → baseline →
Linear / Semantic Linear / Semantic MLP 训练及评测 → Always Global Drop → 五项成绩表。
不执行 seed1000，不动 FEVER。

必须使用**新的快速实验输出目录**，不能把原四/五方法全量训练结果改作子集训练结果。
`--diagnostic-tasks` 不得大于原诊断区数量；`--collection-repeats` 必须为正整数。
默认仍然只读已有诊断数据，只有显式 `--collect-missing` 才允许采集缺失子集。
此入口要求已有冻结 snapshot；缺失时失败并说明原因，不自动重建。

子集分支保存到：

```text
hotpotqa_seed0_train1000_r2_v1/
  artifacts/hotpotqa/diagnostic/seed0/
    branches.jsonl
    run_manifest.json
    collection_progress.json
    collection_diagnostics.json
    gate_training_collection.json
    llm_cache.sqlite
  logs/hotpotqa_collect_seed0.log
  hotpotqa/{native_gmemory,linear,semantic_linear,semantic_mlp,always_global_drop}/
  seed0_progress.json
  seed0_results.{md,json}
```

若原 HotpotQA seed0 有缓存，先验证原 snapshot、数据、模型、endpoint 和采样 seed 等上下文，
再作为新子集的 cache seed 复制过去；只复用请求键精确匹配的响应。
**不把原部分分支直接改名成新子集分支，不伪造完成状态。** 新子集会重新落盘正确的分支记录，
已有回答命中缓存时不再请求模型。原文件保留，原全量流程日后仍可按原命令续跑。

中断后重跑同一条快速模式命令，会恢复子集采集及后续逐题评测。
已完成且验证通过的采集、checkpoint、推理不会重跑。改变子集大小/repeats 后请换新输出目录。
按此前进度粗略线性估算，1,000×2 的采集约 20～25 小时，但不是保证；
后续训练和五项最终评测时间另计。

上传本次快速模式及前一版 Global Drop 改动：

```bash
git add \
  causal_memory_control/run_seed0_gate_matrix.py \
  causal_memory_control/run_hotpotqa_gate_comparison.py \
  causal_memory_control/tests/test_seed0_gate_matrix.py \
  causal_memory_control/tests/test_seed0_subset.py \
  causal_memory_control/SEED0_GATE_MATRIX.md
git diff --cached --stat
git commit -m "Support diagnostic subsets with unchanged held-out gate evaluation"
git push origin main
```

## 双卡并行：两个 Ollama 副本，各采集 500 题 × 2 repeats

这是**两个独立模型副本并行处理不同任务**，不是把一个模型拆到两张卡上。
仅采集并行；后续仍在原主 endpoint 上完成原测试集的五项评测。
先确认第二张 GPU 是你获准使用的资源，停止当前采集控制器后再换新输出目录。

当前服务器诊断：PID 2240063 属于系统用户 `ollama`，父 PID 2442；
它在 GPU 1 上有主要推理负载，在 GPU 0 上也分配了少量显存。
PID 1838208 是用户 `hjq` 自己监听 11436 的服务。
**不要 kill 系统 PID 2442/2240063，也不要为了本实验重启/改动系统 Ollama。**
下面另建 11438 服务，保留系统服务和已有 11436 服务。

先查询原实验记录的 HotpotQA endpoint：

```bash
python -c 'import json; from pathlib import Path; p=Path("causal_memory_control/results/full_gate_benchmark_v1/benchmark_manifest.json"); print(json.loads(p.read_text())["experiments"]["hotpotqa"]["endpoint"])'
```

若打印 `http://127.0.0.1:11434/v1`，下方命令可直接使用。
若是 11436，则把后面 `--collection-endpoints` 的**第一项**替换为原 11436 endpoint，
不要修改原 manifest。第一项必须与原 endpoint 一致，最终评测不会改端口。

在另一个终端或 tmux 窗口，确认 11438 未被占用：

```bash
ss -ltnp | grep -E ':11438([[:space:]]|$)'
```

没有输出才使用该端口。确认 GPU 2 可用后启动独立副本（GPU UUID 以当前 `nvidia-smi` 为准）：

```bash
CUDA_VISIBLE_DEVICES=GPU-6b7b35b1-9e70-ae60-f975-4681a64a2bac \
OLLAMA_HOST=127.0.0.1:11438 \
OLLAMA_MODELS=/mnt/mydata/hjq/ollama_gpu2_models \
OLLAMA_NUM_PARALLEL=1 \
ollama serve
```

保持服务终端运行。该命令只设置**新启动服务**的 GPU，不会迁移已经运行的系统模型。
上例显式使用该用户自己的独立模型目录；如果要复用已有模型，可改为你自己的、
已安装对应权重的用户模型目录。不要让新服务写入、清理系统共享模型目录。

在实验终端检查新服务的模型：

```bash
OLLAMA_HOST=127.0.0.1:11438 ollama list
```

必须装有与主服务**同一权重版本**的 `qwen2.5:7b`。如果没有，可在这个新服务上安装：

```bash
OLLAMA_HOST=127.0.0.1:11438 ollama pull qwen2.5:7b
```

相同模型名不保证相同权重，尤其是旧模型 tag 已更新时。
代码在启动采集前调用两个服务的 `/api/tags`、`/api/show`、`/api/version`，
校验模型 digest、模板/模型配置指纹和 Ollama 版本一致；不匹配或服务不可达会报错，不混用。
除此以外，两个服务的 context length 等服务级设置也应一致；可查看主服务的
`/api/ps` 中当前模型的 context_length（若该版本支持），再为新服务配置对应的
`OLLAMA_CONTEXT_LENGTH`。检查不等于 GPU 绑定验证，也不保证跨硬件 bitwise 一致。

确认服务就绪后，一条命令并行采集，再自动训练/评测：

```bash
python -u -m causal_memory_control.run_seed0_gate_matrix \
  --source-root causal_memory_control/results/full_gate_benchmark_v1 \
  --output-dir causal_memory_control/results/hotpotqa_seed0_train1000_r2_dual_v1 \
  --datasets hotpotqa \
  --diagnostic-tasks 1000 \
  --collection-repeats 2 \
  --collect-missing \
  --collection-endpoints http://127.0.0.1:11434/v1,http://127.0.0.1:11438/v1 \
  --include-always-global-drop
```

可先在末尾加 `--plan-only` 看分片计划：worker_000 处理原诊断列表索引 0～499，
worker_001 处理 500～999；每题仍完整执行 6 候选 × 2 repeats × USE/DROP。
两边合计 **1,000 题、24,000 条逻辑分支**，不是两边各 1,000 题。
最终测试 offset/count 保持原值。相同题的 USE/DROP 和两个 repeats 都留在同一个 worker，
避免跨模型进程拼接配对标签。

若要复用刚才**单进程 1,000 题快速模式**已经产生的回答缓存，可在上述正式命令末尾额外添加：

```bash
--collection-cache-seed-from causal_memory_control/results/hotpotqa_seed0_train1000_r2_v1/artifacts/hotpotqa/diagnostic/seed0/llm_cache.sqlite
```

只在该文件确实存在且原进程已停止时添加；文件父目录必须有匹配的 run_manifest.json。
未指定时仍尝试复用原 full-run seed0 cache。两种情况都只复制匹配响应，不改写原分支/缓存。
旧单进程 manifest 通常未记录权重 digest，因此历史缓存的权重版本仍需你确认未变；
当前副本一致性检查不能倒推出旧响应生成时的权重版本。若模型 tag 曾更新，
请在新输出目录命令上加 `--no-collection-cache-seed`，禁止导入旧响应，
也不要绕过模型副本检查。此选项与 `--collection-cache-seed-from` 互斥。

双卡必须使用新的输出目录；不要直接在已有单卡结果目录加 endpoints 参数。
每个 worker 的 manifest、原始 run_hash、分支日志、SQLite 都独立保留。
合并采用**验证后联合读取多个训练源**：完整性、重复任务、漏任务、候选/repeat/seed、
snapshot/模型等都检查通过后才训练；不会把不同 endpoint 的行伪装成单 endpoint 的新日志。
每个原始源仍可追溯，最终 checkpoint 记录两个 source 文件及其 SHA256。

```text
hotpotqa_seed0_train1000_r2_dual_v1/
  .seed0_parallel.lock
  artifacts/hotpotqa/diagnostic/
    parallel_collection_manifest.json       # 模型副本一致性、分片 ID、来源
    parallel_collection_progress.json       # 两个 PID/进度、退出码、错误日志尾部
    logs/worker_000.log
    logs/worker_001.log
    workers/worker_000/seed0/{branches.jsonl,run_manifest.json,llm_cache.sqlite,...}
    workers/worker_001/seed0/{branches.jsonl,run_manifest.json,llm_cache.sqlite,...}
  hotpotqa/{linear,semantic_linear,semantic_mlp,always_global_drop}/
  seed0_results.{md,json}
```

控制台显示合计进度条，逐 worker 的详细日志保存在上述独立文件中。
一个 worker 失败时，控制器会停止**本次启动的另一个采集 worker**，保留两边断点，
不会停止任何 Ollama 或其他用户的进程。重跑同一命令，已完成 worker 跳过，未完成的继续；
模型版本或分片配置改变则拒绝继续。输出目录有独占锁，避免两个控制器同时写入。
训练和最终评测也保持断点续跑。采集的理想耗时可能接近单卡的一半，但共享服务负载、
GPU 性能、上下文长度、CPU 检索等都会影响提速，不保证整体实验减半。

上传本次双卡相关文件：

```bash
git add \
  causal_diagnostic/fever_oracle_recipient/run_experiment.py \
  causal_diagnostic/hotpotqa_oracle_recipient/run_experiment.py \
  causal_memory_control/run_seed0_gate_matrix.py \
  causal_memory_control/parallel_collection.py \
  causal_memory_control/tests/parallel_worker_fixture.py \
  causal_memory_control/tests/test_parallel_collection.py \
  causal_memory_control/SEED0_GATE_MATRIX.md
git diff --cached --stat
git commit -m "Add resumable task-sharded collection across Ollama replicas"
git push origin main
```

服务器先停止原采集控制器，等进程退出后 `git pull --ff-only origin main`。
上述部署步骤不会由本地修改代码自动执行；需要在服务器手动启动新服务和实验。

## 结果、日志与中断恢复

```text
fever_seed0_gate_matrix_v1/
  seed0_manifest.json
  experiment_plan.json
  seed0_progress.json
  seed0_results.md                     # 四方法/可选五方法成绩表
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
    plans/always_global_drop.json       # 仅启用新对照时出现
    always_global_drop/                 # 仅启用新对照时出现；同样保留逐题日志/缓存/比较
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
