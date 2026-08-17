# G-Memory 先验模块设计

## 1. 目标、范围与结论

本文为 G-Memory 增加一个 **prior module（先验模块）** 的详细设计。它不替换现有的 Interaction Graph、Query Graph、Insight Graph 或 Top-K 检索，而是在每一个“某个 Agent 当前是否应使用某条已检索记忆”的决策点，为候选记忆输出三个不同含义的、带不确定度的预测：

```text
输入：
  当前 task + 当前 Agent + 当前执行状态 + 候选记忆 + 候选集 + 检索来源信息

输出：
  r0 = 这条记忆作为历史记录本身是否可信、可复现
  d0 = 把它展示给当前 Agent 对下一步环境行为造成的预期距离
  u0 = 把它加入团队上下文是否会带来边际团队效用
```

这里的 `0` 表示“在执行昂贵验证以前的先验预测”，不是最终真值。三个量不能合并为一个分数，也不能把任意一个解释为另外两个：

| 量 | 严格定义 | 决策问题 | 与当前任务相关性 |
| --- | --- | --- | --- |
| `r0` | 记忆内容及其来源轨迹是否完整、忠实、可复现 | 这是一条值得相信的历史记录吗？ | 不直接包含 |
| `d0` | 在固定当前上下文下，暴露该记忆造成的下一规范化环境动作预期距离 | Agent 会因它在多大程度上改变行为？ | 包含 |
| `u0` | 在固定当前任务、队伍、随机性下，展示该记忆相对于不展示/安慰剂展示的最终团队奖励差 | 团队用了它会更好吗？ | 包含 |

一个记忆可以 `r0` 高但 `u0` 接近零，例如“打开微波炉前先放入物体”是真规则，但对当前清洗任务没有边际帮助；也可以 `d0` 高但 `u0` 为负，例如 Agent 被一条相似但失败的经验强烈带偏。可靠、会改变行为、有团队价值是三个不同事实。

### 1.1 本设计的边界

- **保留 G-Memory 原有检索。** `GMemory.retrieve_memory()` 仍决定召回哪些成功轨迹、失败轨迹和 insight；prior module 只对召回结果做结构化适配、打分、筛选和日志记录。
- **不把检索相似度当作可靠性。** 相似度只回答“文本像不像”；它不回答记录是否真、是否会影响行为、是否有正面效用。
- **不在冷启动时伪造真值。** 未采集干预标签时只能有透明声明的弱先验或启发式分数，不能称为已校准的可靠性/效用预测。
- **优先从 task-level trajectory 开始。** Insight 的 `score` 和正负关联任务目前是规则维护信号，不是被验证的事实正确率；它们可以作为弱特征，不能直接作为 `r` 的监督标签。

### 1.2 推荐的总体结构

```text
G-Memory 原检索
   |  successful MASMessage / failed MASMessage / insight rule
   v
CandidateAdapter
   |  CandidateRecord + RetrievalRecord
   v
FeatureBuilder
   |  文本、同一嵌入空间的向量特征、结构化字段、缺失掩码
   v
PriorModule
   |-- ReliabilityPrior      -> r0, Beta 不确定度
   |-- DependencyPredictor   -> d0, 预期环境动作距离
   `-- TeamUtilityPredictor  -> P(u< -delta), P(|u|<=delta), P(u>delta)
   v
Memory Controller
   |  USE / VERIFY / IGNORE；把原因与上下文写入决策日志
   v
后续验证与训练数据
   |-- source replay         -> r 标签
   |-- paired action test    -> d 标签
   `-- fork/team rollout    -> u 标签
```

`r0` 可以在候选记忆进入当前任务时缓存，因为它主要是记忆本身的属性；`d0` 和 `u0` 需要随“接收 Agent + 当前步骤状态”重算。当前代码在任务开始时检索一次，这可以保留，但不应因此把 `d0/u0` 固定到整条任务上。

## 2. 当前 G-Memory 中已存在的数据

以下字段来自当前代码，不是假设的理想 schema。路径以仓库根目录为起点。

### 2.1 task-level trajectory：`MASMessage`

定义在 `mas/memory/common.py`：

```python
@dataclass
class MASMessage:
    task_main: str
    task_description: Optional[str]
    task_trajectory: Optional[str]
    label: Optional[bool]
    chain_of_states: StateChain
    extra_fields: dict[str, Any]
```

字段和可用含义如下。

| 字段 | 当前存储内容 | prior module 的用途 | 不能据此断言的内容 |
| --- | --- | --- | --- |
| `task_main` | 简短任务主目标 | 检索查询、任务文本、任务族分组 | 具体当前状态 |
| `task_description` | 任务的详细自然语言描述 | `task_text`、任务语义相似度、任务类型特征 | 记忆是否正确 |
| `task_trajectory` | `action\nobservation` 的可读串 | 候选内容、轨迹长度、步骤抽取 | 原始完整轨迹是否仍可重放 |
| `label` | 完成该源任务后的布尔结果 | 结果极性、弱冷启动信号、分层字段 | 一条经验是否可安全迁移；也不是因果效用 |
| `chain_of_states` | 一组按时间顺序的有向消息图 | 步数、上游 Agent、动作/观察/奖励摘要 | 当前接收 Agent 一定会照做 |
| `extra_fields` | 扩展字段字典 | 读取 `clean_traj`、`key_steps`、`fail_reason` | 这些 LLM 摘要一定忠实 |

`StateChain` 的每个状态是一个 NetworkX `DiGraph`。状态图级别字段通常包括：

```python
state.graph["action"]       # 此步环境动作字符串
state.graph["observation"]  # 此步环境观察字符串
state.graph["reward"]       # 调用 move_state 时传入才存在
```

图的消息节点来自 `AgentMessage`，包含：

```python
agent_name
system_instruction
user_instruction
message
```

节点之间的 `edge_type='spatial'` 边表达当前步骤的上游消息依赖。节点前驱/后继数、消息生产者角色、轨迹中角色切换次数等，都可以成为“来源和协作上下文”特征；这是从图结构**可派生**的信息，意思是无需在数据库新增原始字段，读取图后即可计算。

`GMemory._extract_mas_message()` 还写入：

| `extra_fields` 键 | 内容 | 用途 |
| --- | --- | --- |
| `clean_traj` | 清洗过数字的轨迹文本 | 轻量文本特征、失败分析输入 |
| `key_steps` | LLM 从轨迹提炼的关键步骤 | 精简候选展示文本、步骤长度特征 |
| `fail_reason` | 对失败轨迹的 LLM 归因 | 失败警告特征；不是环境证明的失败原因 |

**当前限制。** `_extract_mas_message()` 会在写库前移除 `reward < 0` 的状态。成功轨迹的 `task_trajectory` 也会被重建为剩余步骤。因此已经持久化的记录不保证带有完整原始动作序列，不能直接用于严格 replay，也不能据此可靠计算负奖励比例。后文要求额外保存原始链。

### 2.2 Insight：`insights_memory` 中的规则字典

Insight Manager 保存的每条规则形如：

```json
{
  "rule": "...自然语言规则...",
  "score": 8,
  "positive_correlation_tasks": ["task A", "task B"],
  "negative_correlation_tasks": ["task C"]
}
```

| 字段 | 当前含义 | 可以怎么用 | 不应怎么用 |
| --- | --- | --- | --- |
| `rule` | 被检索与注入 prompt 的规则文本 | 候选正文、规则长度、文本嵌入 | 自动当作可执行真理 |
| `score` | `backward()` 与 ADD/AGREE/EDIT/REMOVE 规则维护后的启发式分数 | 低权重的历史支持度特征 | 当成可靠性概率或团队效用标签 |
| `positive_correlation_tasks` | 规则维护中被认为与规则相关的一组任务名 | 支持任务数、与查询任务的重合 | 验证成功次数 |
| `negative_correlation_tasks` | 规则被移除/反对时关联的一组任务名 | 反对任务数、弱风险特征 | 验证失败次数 |

特别要避免下面这个错误：

```text
Beta(1 + len(positive_correlation_tasks),
     1 + len(negative_correlation_tasks))
```

它在数学上能算，但统计语义不成立。当前正/负集合是 LLM 在 insight 合并、同意、编辑、移除时更新的“规则维护关联”，不是独立的“该规则在一次检验中正确/错误”的伯努利观测。它最多可作为一个**封顶、低权重**的 cold-start support feature。

### 2.3 Interaction Graph、Query Graph 与任务上下文

| 来源 | 可取得字段或可派生量 | 当前代码中的实际情况 |
| --- | --- | --- |
| Interaction Graph / `StateChain` | 当前/历史步骤的空间前驱、后继、消息生产者、角色拓扑、动作和观察 | 已随 `MASMessage.state_chain` 保存 |
| Query Graph / `TaskLayer.graph` | 相似任务节点以及 k-hop 邻居 | `retrieve_related_task(query_task, node_num, hop)` 返回任务名；当前没有把具体 hop、路径或边权随检索结果返回 |
| 向量检索 | 最终 Top-K、成功/失败记忆分组、相似度阈值 | `_retrieve_memory_raw()` 内部算 cosine 并排序，但 `retrieve_memory()` 最终只返回对象，不返回具体 similarity/rank/distance |
| 当前任务 | `task_config`、`task_main`、`task_description`、`few_shots` | 在 workflow `schedule(task_config)` 中可得 |
| 当前 Agent | `agent.name`、`agent.profile`、`agent.system_instruction` | 已存在于 MAS workflow；不同 workflow 的字段名字要由适配器统一 |
| 在线状态 | 最近 action、observation、近期轨迹、当前图节点 | 运行期已有；训练必须在决策时保存快照 |

所谓“hop、边权、前驱/后继是可派生的”是指：它们不一定现在被写入 `MASMessage` 或返回值，但可以从现有图结构、检索过程或修改后的检索元数据中计算出来。例如 Query Graph 的实际 hop 可用 `nx.shortest_path_length(graph, source, target)` 计算；但前提是把 `target` 对应的任务节点一起保留。

## 3. 统一输入：不改变检索，只补一层适配器

当前 `retrieve_memory()` 返回三组异构对象：成功 `MASMessage`、失败 `MASMessage`、`str` 类型 insight。prior module 不应让三个 head 分别了解这些内部细节。建议在检索返回后立即适配为两个记录。

```python
@dataclass
class CandidateRecord:
    memory_id: str                 # 稳定 ID；当前需补充，不能用列表下标
    memory_type: str               # "trajectory" 或 "insight"
    polarity: str                  # "success", "failure_warning", "rule", "unknown"
    content: str                   # 供 Agent 与编码器阅读的候选正文
    structured_metadata: dict      # 下节列出的原始/派生字段


@dataclass
class RetrievalRecord:
    candidate_id: str
    query_task: str
    retrieval_source: str          # "query_graph" / "vector_fallback" / "insight"
    rank: int | None
    semantic_similarity: float | None
    distance: float | None
    actual_hop: int | None
    path_weight: float | None
    related_task_hit_count: int | None
```

### 3.1 候选正文怎样拼接

不需要把所有字段直接塞进一个 prompt，也不应只保留 `task_main`。

```text
trajectory candidate.content =
  "Source task: " + task_description
  + "\nKey steps: " + extra_fields["key_steps"]
  + "\nTrajectory: " + task_trajectory

insight candidate.content =
  "Rule: " + rule
```

`clean_traj` 可用于训练特征，通常不必与 `task_trajectory` 同时给 Agent，避免重复和上下文膨胀。`fail_reason` 仅给失败警告候选添加明确标签，例如 `Historical failure warning: ...`；不能把失败轨迹伪装为应模仿的正向经验。

### 3.2 适配器要保留的结构化字段

```python
# trajectory 候选
{
  "source_task_main": msg.task_main,
  "source_task_description": msg.task_description,
  "source_label": msg.label,
  "trajectory_text": msg.task_trajectory,
  "clean_traj": msg.get_extra_field("clean_traj"),
  "key_steps": msg.get_extra_field("key_steps"),
  "fail_reason": msg.get_extra_field("fail_reason"),
  "state_count": len(msg.chain_of_states),
  "source_reward_sum": sum(s.graph.get("reward", 0.0) for s in msg.chain_of_states),
  "source_reward_last": last_reward_or_missing,
  "source_agent_names": derived_agent_names,
  "source_graph_in_degree_mean": derived_value,
  "source_graph_out_degree_mean": derived_value,
  "has_raw_replay_artifact": False,   # 新日志存在时为 True
  "memory_schema_version": "gmemory-v1"
}

# insight 候选
{
  "rule": insight["rule"],
  "legacy_score": insight["score"],
  "positive_task_count": len(insight["positive_correlation_tasks"]),
  "negative_task_count": len(insight["negative_correlation_tasks"]),
  "positive_task_ids": insight["positive_correlation_tasks"],
  "negative_task_ids": insight["negative_correlation_tasks"],
  "memory_schema_version": "gmemory-v1"
}
```

`source_reward_sum` 和 `source_reward_last` 要同时输出 `is_missing` 掩码。代码不保证每个状态有 reward，而且已持久化链可能被稀疏化；缺失不是零奖励。类似地，Query Graph 当前没有边权，应设置 `path_weight=None` 和 `path_weight_missing=1`，不可臆造一个数。

### 3.3 必须补的稳定 ID 和检索元数据

现有代码将 Chroma 文档转回 `MASMessage` 时没有暴露 document ID、距离与 rank，insight 也被压成规则字符串。prior module 至少要补：

1. `memory_id`：Chroma 文档 ID，或写库时生成的 UUID；insight 用稳定 UUID，不能只用 rule 文本，因为 EDIT 会改变文本。
2. `retrieval_source`：候选来自 Query Graph 邻居、向量回填还是 Insight 检索。
3. `rank` 和 `semantic_similarity`：在 `_retrieve_memory_raw()` 已经算了 cosine；将其随候选返回即可，不必改变 Top-K 的逻辑。
4. Query Graph 的 `actual_hop`：返回任务节点时一起记录，或由图再计算。
5. 洞见完整字典：保留 `{rule, score, positive_correlation_tasks, negative_correlation_tasks}`，不只返回字符串。

这是一层**统一数据格式**，不是改造 G-Memory 的检索方法。

## 4. 从文本、向量、字符串到特征

prior module 的一个样本不是“记忆”单独存在，而是一个决策上下文：

```text
x = (task T, receiving agent A, state S, candidate memory M,
     retrieved candidate set C, retrieval record Q)
```

### 4.1 四段在线文本

以下是在线临时构造的文本，不建议作为 `MASMessage` 的长期字段。

```python
task_text = (
    task_config["task_main"]
    + "\n"
    + task_config["task_description"]
)

agent_text = agent.profile + "\n" + agent.system_instruction

state_text = (
    latest_observation
    + "\nRecent execution:\n"
    + recent_task_trajectory   # 只取最近 N 步，防止无界增长
)

memory_text = candidate.content
```

例如，在 ALFWorld 中：

```text
task_text:
  put a clean mug in the cabinet
  Your task is to find a mug, clean it if necessary, and place it in cabinet 1.

agent_text:
  decision
  You choose the next executable action from the environment observation.

state_text:
  You are in the kitchen. A dirty mug is in the sink. Cabinet 1 is closed.
  Recent execution:
  go to sink
  examine mug

memory_text:
  Source task: clean and store a cup
  Key steps: find the mug; clean it in the sink; open the target cabinet; put mug in cabinet.
  Trajectory: ...
```

`state_text` 不是必须新设为 G-Memory 的长期字段。在线推断时直接拼接即可；但为了之后训练 `d/u`，每一次实际决策时必须把当时的 `state_text` 或其版本化、可重构快照写进决策日志。否则事后已无法知道模型当时看到了什么。

### 4.2 向量特征与“已有 Top-K 相似度”的关系

G-Memory 本身已经通过 embedding 和 cosine/select top-k 召回候选，因此**不需要另起一套检索算法或重新检索**。不过先验头需要数值特征，而现有公开返回值不带检索相似度；而且 `d/u` 与“当前状态、当前角色”的关系并不等于“任务标题与历史任务”的相似度。

推荐先冻结 G-Memory 当前使用的 embedding 模型，在同一向量空间计算：

```text
sim_task_memory  = cosine(E(task_text),  E(memory_text))
sim_state_memory = cosine(E(state_text), E(memory_text))
sim_role_memory  = cosine(E(agent_text), E(memory_text))
```

其中 `sim_task_memory` 可直接沿用检索中已得到的 cosine，前提是同时保存候选正文和数值；如果目前未返回该数值，再用**相同 embedding 模型**计算一次即可。`sim_state_memory` 与 `sim_role_memory` 没有被原有 task-level Top-K 覆盖，仍需新算。这不是重复检索，而是为打分构造两个额外特征。

低成本的初版只需上述 3 个余弦值、rank 与文本长度。不要一开始微调 embedding；否则先验模型变化、检索变化和数据分布变化会混在一起，难以解释。

### 4.3 结构化特征清单

所有数值特征至少配一个缺失掩码，例如 `has_reward_last`、`has_actual_hop`。分类文本（模型版本、任务族、Agent profile）可用 one-hot、目标编码或小型 embedding。

| 特征族 | 具体特征 | 来源与处理 | 首要服务对象 |
| --- | --- | --- | --- |
| 文本相似 | `sim_task_memory`, `sim_state_memory`, `sim_role_memory` | 四段文本编码后的 cosine | `d0`, `u0` |
| 检索 | `rank`, `semantic_similarity`, `distance`, `retrieval_source`, `actual_hop`, `path_weight` | `RetrievalRecord` | `d0`, `u0` |
| 轨迹结果 | `source_label`, `state_count`, reward 摘要、动作/观察长度 | `MASMessage`/`StateChain`；稀疏化后要标记 | `r0`, `u0` |
| 轨迹质量 | `has_key_steps`, `key_steps_length`, `has_fail_reason`, `has_raw_replay_artifact` | `extra_fields` 与新 artifact | `r0` |
| 来源链 | 生产 Agent 名称/角色、不同 Agent 数、均值入度/出度、链长度 | `chain_of_states` 图遍历 | `r0`, `d0` |
| Insight 支持 | `legacy_score`, 正/负关联任务数、封顶 support 均值与方差 | insight 字典；低权重使用 | insight 的 `r0` 弱特征，`u0` |
| 当前 Agent | `profile`, 系统提示版本、当前节点入/出度、是否终端节点 | workflow/当前 Interaction Graph | `d0`, `u0` |
| 当前任务 | `task_type`, `game_name`, `difficulty`, step index | `task_config`、运行期状态 | 三个头的分层与校准 |
| 候选集关系 | 候选数、最大相似度、与已注入项的冗余、冲突分数 | 当前召回集合 | `u0` |

`conflict_max` 不在当前字段中。只有在后续对候选对运行 NLI/规则冲突判别器后才存在；它应该先标记为“新增派生特征”，不能写作已可用字段。`redundancy_max` 可以从候选文本嵌入直接计算，不需新增原始采集。

### 4.4 数据构造伪代码

```python
def build_features(context, candidate, retrieval, candidate_set):
    texts = make_texts(context, candidate)
    vectors = embed_same_model(texts)  # 初版冻结，与 G-Memory 同一模型

    return {
        "sim_task_memory": cosine(vectors.task, vectors.memory),
        "sim_state_memory": cosine(vectors.state, vectors.memory),
        "sim_role_memory": cosine(vectors.agent, vectors.memory),
        "source_label": encode_label(candidate.structured_metadata),
        "state_count": candidate.structured_metadata.get("state_count"),
        "retrieval_rank": retrieval.rank,
        "retrieval_hop": retrieval.actual_hop,
        "agent_profile": context.agent_profile,
        "task_type": context.task_config.get("task_type"),
        "candidate_redundancy_max": max_similarity(candidate, candidate_set),
        "missing": make_missingness_masks(...),
    }
```

## 5. `r0`：记忆可靠性的先验

### 5.1 先固定可检验的定义

对成功轨迹，建议将 `r` 定义为：

> 在记录时的原始环境、任务配置、Agent/MAS 版本和执行约束下，该 task-level memory 所声称的正向经验是否是一段完整、忠实且可以安全重放的记录。

这使 `r` 的真值可被验证：重置源任务环境，按保存的原始动作序列重放，检查是否达到源任务目标、动作是否有效、最终奖励是否满足原记录的成功约束。它不等于“当前任务能否迁移”，后者必须交给 `d/u`。

对失败轨迹，必须显式选一种语义，不能沿用“安全模仿的正向经验”：

1. **推荐初版：** `r` 只适用于成功经验；失败候选是 `failure_warning`，不参与“应模仿”的 `r` 排序。
2. **后续扩展：** 单独定义 `r_warning` 为“该失败轨迹及 `fail_reason` 是否准确地描述应避免的失败模式”。它需要另一套反事实/重放标签，不能将 `label=False` 简单当作 `r=0`。

因此 `label=True` 只是“源回合最终成功”的已知结果，不能直接证明记忆文本完整、摘要忠实、动作可重放或对其他任务安全。

### 5.2 `r` 的验证标签：source replay

所谓 **replay 数据** 是指：为已经存进记忆库的一条历史轨迹，再在相同源任务上执行一次保存的原始动作序列，并保存验证结果。它不是让 LLM 复述原任务，也不是把当前任务重新运行一次。

```text
原任务执行完成
  -> 在写入 G-Memory 前保存 raw replay artifact
  -> 之后重置为同一个 task_config / 同一随机种子（若环境支持）
  -> 逐步执行 raw_actions
  -> 保存目标是否完成、各步是否有效、最终 reward、异常原因
  -> 得到 y_r
```

建议先使用二值标签：

```text
y_r = 1  当且仅当：
  source_label == True
  AND replay_goal_reached == True
  AND invalid_action_count == 0
  AND replay_final_reward >= success_threshold

y_r = 0  否则
```

这一定义是保守的。若环境有合理的随机性，可改为每条重放 `K` 次，令 `y_r` 为通过比例，而不是硬二值。保存的最小 artifact 是：

```json
{
  "memory_id": "traj-...",
  "source_task_config": {"game_name": "...", "problem_index": 12},
  "environment_version": "...",
  "mas_version": "...",
  "model_id": "...",
  "prompt_version": "...",
  "seed": 123,
  "raw_actions": ["go to sink", "clean mug", "..."],
  "raw_observations": ["..."],
  "raw_rewards": [0, 0, 1],
  "source_final_reward": 1.0
}
```

ALFWorld 至少要保存 `task_config['env_kwargs']['gamefile']`；PDDL 环境至少要保存 `game_name`、problem/index、环境版本与 seed。必须在 `_extract_mas_message()` 稀疏化之前保存，因为该函数会删除负奖励状态。artifact 可以写入独立 JSONL/SQLite 表，不建议塞进 Chroma metadata。

### 5.3 Cold start：经验贝叶斯 Beta，而不是假定所有记忆同样可靠

在未有足够 memory-specific replay 的时候，按来源相似的分组估计初始可靠性：

```text
group_key = (
  memory_type,
  environment_family,
  task_type_or_difficulty,
  MAS_architecture_version,
  model_id,
  prompt_version,
  memory_extraction_version
)
```

对一个组 `g`，收集已 replay 的通过/失败次数 `S_g, F_g`：

```text
alpha_g = alpha_base + S_g
beta_g  = beta_base  + F_g

r0_mean(g) = alpha_g / (alpha_g + beta_g)
r0_var(g)  = alpha_g * beta_g /
             ((alpha_g + beta_g)^2 * (alpha_g + beta_g + 1))
```

`alpha_base, beta_base` 是明确写在实验配置中的弱主观假设，例如成功轨迹先验 `Beta(2, 1)`，失败/未知来源 `Beta(1, 2)`。它们不是“证明过的真实概率”；它们只在没有 replay 证据时提供温和方向，且等价样本量很小。若不想表达方向性，可用 `Beta(1, 1)`。

对某条 `m` 的最终初始值，采用分层收缩，而不是直接使用一个小组的极端比例：

```text
global prior:     Beta(alpha_0, beta_0)
environment prior: Beta(alpha_env, beta_env)
group prior:       Beta(alpha_g, beta_g)
memory evidence:   S_m, F_m

r0(m) = posterior_mean after hierarchical / empirical-Bayes shrinkage
```

简化可实现版本：用 global 的 `Beta(alpha_0, beta_0)` 作为 group 的 base；group replay 结果更新 group Beta；memory 自己 replay 的结果再更新 memory Beta。每个输出都报告总等价样本数 `n_eff = alpha + beta` 或方差，不允许控制器把一个 `Beta(2,1)` 当作和 `Beta(200,50)` 一样确定。

Insight 初期可构造一个封顶 support proxy：

```text
p = len(positive_correlation_tasks)
n = len(negative_correlation_tasks)
w = min(p + n, 3) / (p + n + 1)       # 最大等价证据量低
support_mean = (p + 1) / (p + n + 2)
```

它只能是 `r_weak_support` 特征或人工审阅排序依据，不能写为经过 replay 验证的 `r0`。后续若要有 insight 的可靠性，需要验证其规则在对应环境/任务中的可执行性。

### 5.4 何时把 `r0` 做成学习头

有 `y_r` 后可训练一个校准模型：输入来源、摘要质量、轨迹拓扑、文本嵌入和 artifact 完整性；输出 replay pass 概率。建议顺序是：

1. 先以 Beta/经验贝叶斯作为基线；
2. 收集 replay 标签并按任务族划分验证集；
3. 用逻辑回归或 LightGBM 预测 `P(y_r=1 | x_r)`；
4. 用 isotonic/temperature calibration 校准；
5. 将模型概率与分层 Beta 后验融合，或至少同时报告二者的证据来源。

不要为了“统一网络”让未标注的 `r` head 从 `label`、insight score 直接学习。那会学到源任务成功率或数据收集偏差，而不是记录可靠性。

## 6. `d0`：行为依赖的先验

### 6.1 定义与标签

`d` 不表示“记忆是好是坏”，而表示该记忆对当前 Agent **实际可执行环境行为的干预强度**。不应把模型输出字符串直接比较；应先将最终送往环境的动作解析为统一对象：

```python
CanonicalAction(
    tool_name,          # 环境/tool API 名称；没有 tool 时为 None
    action_type,        # go / open / take / heat / put / ...
    target_object,      # 环境中的规范对象 ID，例如 apple 1
    destination,        # 规范位置或容器 ID，例如 microwave 1
    arguments,          # 其余按 key 排序、类型规范化的参数
    parse_status        # valid / invalid / ambiguous
)
```

优先复用环境执行器已有的动作 parser、对象 ID 和别名表，而不是让另一个 LLM 自由地解释动作文本。这样：

```text
go to microwave
navigate to the microwave
```

会解析为同一个 `CanonicalAction`；而：

```text
heat apple 1 with microwave 1
heat apple 2 with microwave 1
```

会因 `target_object` 不同被识别为不同环境行为。

定义环境动作距离 `D_env(a, b) in [0, 1]`。初版可使用下面的、按环境预先配置的有序等级；等级本身可以在后续数据中校准，但不能临时按结果改写。

| `D_env` | 判据 | 示例 |
| --- | --- | --- |
| `0.0` | 规范化后的工具、动作类型、目标、位置、参数全部相同 | 两种“去微波炉”的同义表述 |
| `0.25` | 动作类型相同，只有被环境标为非关键的参数不同 | 两个等价观察选项或不影响状态的参数 |
| `0.50` | 目标对象、目的地或子目标不同，但动作仍可低成本撤回 | 对不同可替代物品执行同类动作 |
| `0.75` | 工具/动作类型不同，或进入不同可逆计划分支 | `open cabinet` 对比 `go to sink` |
| `1.0` | 终止、提交、删除、消耗唯一资源、不可逆加热/移动，或违反任务关键约束 | 执行终止任务与继续搜索；破坏唯一目标物 |

“参数是否关键”和“动作是否不可逆”是**环境语义配置**，不能只由文本相似度决定。`heat apple 1` 与 `heat apple 2` 至少应为 `0.5`；当 apple 1 是唯一任务目标，或加热不可逆时，应升为 `1.0`。`parse_status=invalid/ambiguous` 也必须作为单独特征与日志字段；不要静默地把未解析动作当作字符串或距离零。

模型的自由解释、理由文本和未执行计划不属于 `D_env`。同一规范化动作的说明文字不同，环境行为距离仍应为 `0.0`，而不是 `0.2`。如果研究也关心多 Agent 沟通是否被记忆改变，可额外记录 `D_message` 或 `D_plan`；只有计划文本会被下游 Agent 自动采纳并执行时，才把它纳入另一个“协作影响”量，不能污染 `d` 的环境动作定义。

在确定性执行下，行为依赖标签是单个距离。在随机策略下，主定义是两个动作分布之间的预期成对距离：

```text
d(T, A, S, M) =
  E[D_env(A_with, A_control) | T, A, S]

A_with    ~ pi(. | T, A, S, show M)
A_control ~ pi(. | T, A, S, show placebo)
```

所以 `d0` 是落在 `[0, 1]` 的“预期行为距离”先验，不再只是粗粒度二分类概率。为方便控制器做阈值决策，还应同时派生两个辅助量：

```text
p_any_change      = P(D_env > 0)
p_material_change = P(D_env >= tau_material)  # 例如 tau_material = 0.5
```

获得 `d` 标签的成对行动干预：

```text
固定 task T、Agent A、决策状态 S、模型参数和随机种子

分支 M：把真实候选 M 注入本步 prompt -> action_with
分支 P：移除 M，或用长度/格式匹配的安慰剂 -> action_control

y_d = D_env(CanonicalAction(action_with),
            CanonicalAction(action_control))
```

安慰剂不应是任意“别的有用记忆”，而是尽量控制上下文长度、标题和格式、只剥离候选的可行动内容的文本。例如：`Historical note unavailable for this task.` 加上同长度无关字符会破坏语言流畅性，通常不如一条结构相同的通用中性提示。

对于确定性解码（`temperature=0`），每个决策点的一对前向即可得到一个有序标签。对于随机解码，要在相同上下文下重复 `K` 次，估计两组动作分布；可用所有跨分支动作对的平均距离，或在共同随机种子下配对后取平均。后者估计的是“相同采样扰动下记忆造成的变化”，要在日志中写清采用了哪一种估计量。

### 6.2 模型输入与训练

`d` head 的核心输入是当前状态而不是源轨迹的最终 label：

```text
x_d = [sim_task_memory, sim_state_memory, sim_role_memory,
       current step index, current agent role, current graph topology,
       memory type/polarity, retrieval rank, source quality features,
       candidate redundancy, missingness masks]
```

由于标签有自然顺序，初版建议用有序分类，而不是直接回归一个看似精确的连续数：类别为 `{0.0, 0.25, 0.50, 0.75, 1.0}`，模型输出每类概率，再取其期望作为 `d0`。逻辑回归的 ordinal 版本或 LightGBM 多分类都可用；后者适合候选量和任务类型增多后的非线性特征。输出同时保留距离分布与阈值概率：

```json
{
  "d0_expected_distance": 0.52,
  "d0_distance_distribution": {
    "0.0": 0.14,
    "0.25": 0.16,
    "0.50": 0.31,
    "0.75": 0.25,
    "1.0": 0.14
  },
  "p_any_change": 0.86,
  "p_material_change": 0.70,
  "d0_uncertainty": 0.19,
  "label_definition": "paired_canonical_environment_action_distance",
  "calibrated": false
}
```

没有成对前向数据时，不存在可从当前 G-Memory 字段中训练出的有效 `d` 真先验网络。冷启动可输出 `d0_expected_distance=0.5`、高不确定度和均匀等级分布，或只输出用于主动采样的 heuristic score，例如 `0.6 * sim_state_memory + 0.4 * sim_role_memory`，但必须标为 `heuristic`。

### 6.3 行为依赖能否用互信息

可以，且在随机动作或需要比较“改变程度”时比单一的动作变化标签更细。但它应是后续升级，不是替代干预设计。动作分布首先应定义在 `CanonicalAction` 上，而不是未经处理的自由文本上。

在固定上下文 `C=(T,A,S)` 下重复采样，估计：

```text
p_with(a)    = P(A_canonical=a | C, M=1)
p_placebo(a) = P(A_canonical=a | C, M=0)
```

局部行为差异可用 Jensen-Shannon divergence：

```text
dependency_strength(C, M) = JS(p_with || p_placebo)
```

跨多个上下文时，若将 `M` 看作随机处理变量，条件互信息为：

```text
I(A_canonical; M | C)
```

它量化在控制 task、Agent、状态之后，记忆处理变量对规范化动作分布提供多少信息。实际估计仍要求随机/平衡地分配“展示真实记忆”和“展示安慰剂”，并在每个上下文有重复样本；仅凭历史日志里“有无记忆”不能消除选择偏差。研究初期建议仍以有序动作距离标签训练/评估，待每个上下文有足够重复样本再报告 JS/条件互信息。

## 7. `u0`：团队边际效用的先验

### 7.1 严格定义

此处 `u` 是**团队效用**，不是单个 Agent 的局部效用：

```text
u(T, team, S, M) =
  E[R_team | show M, T, team, S]
  - E[R_team | placebo/no M, T, team, S]
```

`R_team` 应预先固定：可以是最终任务成功奖励，或 `final_reward - lambda * token_cost - mu * environment_steps`。所有实验必须使用同一效用定义。当前的 `MASMessage.label` 只能提供某一次任务的成败，不是该候选记忆的边际因果效用；同样，`GMemory.backward(final_done)` 会把同一个最终结果更新到整批 cached insight，不能当作单条记忆的 `u` 标签。

### 7.2 标签：fork/team rollout

**fork rollout** 指在完全相同的任务初始条件或同一决策状态上，分叉出两个控制变量唯一不同的团队执行分支：

```text
Branch M: 当前 Agent/团队上下文包含候选 M
Branch P: 相同上下文，但移除 M 或放入匹配安慰剂

y_u = R_team(Branch M) - R_team(Branch P)
```

如果环境支持保存/恢复状态，在当前步骤 `S` clone 环境与 Agent 对话状态后 fork，成本最低。若不支持中途 fork，则从相同 `task_config` 和 seed 重置两次，分别完整执行两个团队分支；这仍是有效的初始实现，只是更贵。尽量使用相同随机种子或 common random numbers，并随机交换两个分支的执行顺序，避免时间、服务波动和环境随机性造成系统偏差。

由于 rollout 标签噪声大，先按阈值三分类更稳健：

```text
y_u = negative  if delta_reward < -delta
      neutral   if |delta_reward| <= delta
      positive  if delta_reward > +delta
```

`delta` 必须按环境奖励尺度确定并固定，例如二元成功奖励可以先设为 `0.5`；若纳入成本则先归一化。不要在训练集上临时调 `delta` 以追求好看的结果。

### 7.3 `u` 的特征与模型

```text
x_u = [
  d0 或 d 特征,
  r0_mean / r0_uncertainty,
  sim_task_memory, sim_state_memory, sim_role_memory,
  source label/reward/quality,
  current Agent role and graph position,
  task type/difficulty,
  retrieval rank/hop,
  candidate-set redundancy/conflict,
  step index and remaining budget
]
```

将 `r0/d0` 作为 `u` 特征或门控变量是合理的，但不要硬编码 `u = r * d`：一个可靠、能改变动作的记忆仍可能带来负效用。初版用三分类 LightGBM/逻辑回归，输出校准概率而不是声称精确的 reward 回归：

```json
{
  "u0_probability": {
    "negative": 0.16,
    "neutral": 0.31,
    "positive": 0.53
  },
  "u0_uncertainty": 0.22,
  "label_definition": "paired_team_reward_delta",
  "calibrated": false
}
```

仅有 `label`、reward、`fail_reason` 的冷启动阶段，`u0` 可以产生主动采样优先级，但没有充分根据宣称其为团队效用先验。例如高 `sim_state_memory`、高 `d` 启发式、低冗余和成功来源可以优先被 rollout 标注；这是一种“值得验证”的排序，而不是正效用证据。

## 8. 三个头如何组合，而不是混淆

### 8.1 是否用一个先验网络同时输出 `r/d/u`

最终可以采用共享特征编码器加三个 head：

```text
text embeddings + structured features
             |
        shared encoder
         /     |      \
  r-head    d-head    u-head
```

但三头**不应要求同一时期、同一种标签、同一损失函数**：

| head | 最可信标签 | 成本 | 建议初始实现 |
| --- | --- | --- | --- |
| `r` | source replay pass/fail | 低到中 | Beta/经验贝叶斯，之后校准分类器 |
| `d` | paired canonical-action intervention | 低 | 成对前向后训练有序分类器 |
| `u` | fork/team rollout reward delta | 高 | 分层抽样、三分类器 |

因此推荐的工程结构是“一个 `PriorModule` API，内部三个独立估计器”。开始时它们甚至不是神经网络；在标签量、任务覆盖和校准足够后，再共享 encoder。强行把未标注的三个量放在一个网络里，只会让便宜但不等价的 proxy（如 `label`、rank）泄漏到 `u/r`。

### 8.2 控制器的使用规则

控制器需要关注方向、风险和验证成本：

```text
IGNORE：P(u>delta) 低，或候选与已选项高度冗余
VERIFY：r0/d0/u0 不确定度高且潜在信息价值高
USE：P(u>delta) 高，校准可信，且没有 r 的质量门控风险
WARN：failure_warning 候选以“避免此模式”格式展示，不作为模仿步骤
```

`r0` 很低不必自动丢弃所有 memory：可能应转为 `VERIFY`，或仅允许在不执行动作的建议模式中使用。`d0` 很低而 `u0` 高通常是模型或标签不一致的诊断信号，因为团队效用通常需要至少一个关键 Agent 的行为发生某种变化；不应依靠公式硬修正，而应检查 rollout 设计。

## 9. 训练数据与日志：历史干预数据从哪里来

历史干预数据不是当前 G-Memory 自动已有的字段，而是从运行系统时按实验设计主动采集：

| 数据集 | 采集时间 | 控制变量 | 标签 |
| --- | --- | --- | --- |
| `replay_results` | memory 写入后/离线批处理 | 同一源任务与原动作序列 | `y_r`、奖励、无效动作、失败原因 |
| `dependency_pairs` | 候选展示给 Agent 前或离线重放决策点 | 真记忆 vs 安慰剂 | `y_d=D_env(a_with,a_control)`、两种规范化动作、采样 seed |
| `team_rollout_pairs` | 高价值候选抽样时 | 真记忆 vs 安慰剂/移除 | `y_u`、两分支 reward、成本、完成状态 |

建议将它们写入 `prior_data/` 的独立、版本化 JSONL 或 SQLite 表。示例决策记录：

```json
{
  "decision_id": "dec-...",
  "timestamp": "2026-08-05T10:00:00+08:00",
  "task_run_id": "run-...",
  "task_config_snapshot": {"task_main": "...", "game_name": "..."},
  "agent_id": "decision-0",
  "agent_profile": "decision",
  "step_index": 3,
  "state_text": "...",
  "candidate_id": "traj-...",
  "candidate_ids_presented": ["traj-...", "insight-..."],
  "retrieval": {"rank": 1, "semantic_similarity": 0.82, "actual_hop": 1},
  "feature_schema_version": "prior-v1",
  "prior_output": {"r0_mean": 0.67, "d0_expected_distance": 0.50, "u0": null},
  "controller_action": "VERIFY"
}
```

这类记录是之后能否训练网络的前提。只保留“任务最终成功”会造成严重的归因问题：同一次成功可能来自多个 memory、Agent 自己的推理、环境容易度或随机性，不能告诉我们某条候选产生了什么效果。

### 9.1 防止选择偏差

如果控制器只把“看起来好”的候选送去 rollout，训练集会永远缺少困难/负面候选，`u` 模型会过度乐观。采样策略应混合：

```text
50%  不确定度最高的候选（探索）
25%  预期正效用最高的候选（利用）
15%  高相似度但历史失败/低质量的风险区域
10%  分层随机基线（覆盖任务族、角色、rank、记忆类型）
```

比例可随预算调整，关键是保留随机探索与任务族分层。训练/测试必须按任务实例或任务族划分，不能让同一源任务的近重复轨迹同时出现在训练和测试中。

## 10. 冷启动到可学习先验的实施阶段

### Phase 0：无新标签的透明弱先验

| 输出 | 可用做法 | 解释 |
| --- | --- | --- |
| `r0` | 成功轨迹 `Beta(2,1)`、未知 `Beta(1,1)`；insight 仅输出 capped support | 主观弱假设，未校准 |
| `d0` | 预期距离 `0.5`、均匀等级分布与高不确定度，或仅用相似度做 `sampling_score` | 当前字段没有行为因果标签 |
| `u0` | 三类均匀概率，或仅输出 rollout 抽样优先级 | 当前字段没有候选的因果团队效用标签 |

不建议把 `label=True` 直接设为 `r0=1`，或把 insight 正负关联次数直接变为强 Beta 后验。前者忽略摘要/重放完整性，后者把 LLM 维护信号误当独立实验观测。

### Phase 1：先补数据契约与日志

1. 在写入记忆前记录 `memory_id`、原始动作链、环境可重置配置、版本、seed。
2. 修改检索适配层，保留候选 ID、rank、cosine、Query Graph 来源和实际 hop。
3. 每次候选展示/过滤前记录 `DecisionContext` 快照、候选集和 prior 输出。
4. 记录模型参数（model ID、temperature、prompt 版本），否则后续干预不能复现。

### Phase 2：低成本标签与两个基线

1. 对所有成功 trajectory 批量做 source replay，获得 `y_r`。
2. 对一部分候选决策点做 paired action test，获得 `y_d`。
3. 分别训练/校准 `r`、`d` 的轻量模型；保留 Beta 与“均匀距离等级、期望 `d0=0.5`”作为基线。

### Phase 3：昂贵的团队效用标签

1. 根据第 9.1 节的混合策略挑选候选。
2. 运行 paired team rollouts，得到 `delta_reward` 和三分类 `y_u`。
3. 训练 `u` 三分类器，按任务族、角色、记忆类型评估校准和收益。

### Phase 4：共享编码器（可选）

只有在三个任务都有足够的、覆盖多环境的标签时，才尝试共享 encoder + 三头联合训练。即使如此，仍保留各 head 的独立校准层与不确定度报告。

## 11. 评估与 Go/No-Go 测量

### 11.1 先做 source replay 异质性测量

这一步回答“是否值得专门建模 `r`”，而不是先假定答案为是。

1. 从每个 `environment_family x task_type/difficulty x MAS/model/prompt version` 分层抽样成功轨迹；每层至少有足以报告置信区间的数量。
2. 按第 5.2 节重放每条轨迹，统计 `replay_pass_rate`、无效动作率、最终奖励偏差。
3. 给总体和每个分层计算 Wilson/Beta 后验区间；比较不同模型、任务族、抽取版本的差异。
4. 单独检查“原始 replay 成功但 `key_steps` 重演失败”的比例，以区分原始轨迹可靠与摘要可靠。
5. 在未参与分组/模型设计的 task family 上验证。

建议的判定：若总体 replay 失败率小于约 2%，各关键分组置信区间高度重叠，且摘要重演也稳定，那么 `r` 的可预测变化很小，单独复杂建模价值低，可以只采用常数质量门控。反之，若 producer、任务族、memory extraction version 之间有实质差异，则分层 `r` prior 有明确价值。

这里的 `2%` 只是可预注册的工程阈值示例，不是理论常数；应按错误记忆一次被采纳的实际成本调整。

### 11.2 每个头的指标

| 头 | 判别能力 | 概率质量 | 决策价值 |
| --- | --- | --- | --- |
| `r` | AUROC / PR-AUC（replay pass） | Brier、ECE、Beta 区间覆盖 | 低 `r` 记录是否更常 replay 失败 |
| `d` | ordinal macro-F1、MAE、Spearman（动作距离） | 等级概率 Brier、ECE | 高 `d` 的候选是否有更大实测动作距离/JS 行为差异 |
| `u` | macro-F1 / one-vs-rest AUROC | 三类 Brier、ECE | 控制器相对原 Top-K 的团队奖励、成功率、token/步数成本 |

除平均指标外，必须报告按任务族、Agent role、记忆类型、retrieval rank 的分组表现。否则模型可能只在高频易任务上看起来有效。

## 12. 推荐 API 与最终输出

```python
class PriorModule:
    def score(
        self,
        context: DecisionContext,
        candidates: list[CandidateRecord],
        retrieval: dict[str, RetrievalRecord],
    ) -> list[PriorScore]:
        ...
```

一个 trajectory 候选的输出示例：

```json
{
  "memory_id": "traj-0031",
  "memory_type": "trajectory",
  "polarity": "success",
  "reliability": {
    "mean": 0.72,
    "variance": 0.018,
    "evidence_count": 12,
    "source": "group_beta",
    "calibrated": false
  },
  "dependency": {
    "expected_distance": 0.50,
    "distance_distribution": {
      "0.0": 0.20,
      "0.25": 0.20,
      "0.50": 0.20,
      "0.75": 0.20,
      "1.0": 0.20
    },
    "p_any_change": 0.80,
    "p_material_change": 0.60,
    "uncertainty": 0.50,
    "source": "cold_start_default",
    "calibrated": false
  },
  "team_utility": {
    "negative": 0.33,
    "neutral": 0.34,
    "positive": 0.33,
    "uncertainty": 1.0,
    "source": "cold_start_uniform",
    "calibrated": false
  },
  "recommended_action": "VERIFY",
  "reason_codes": ["NO_D_INTERVENTION_LABELS", "NO_U_ROLLOUT_LABELS"]
}
```

字段解释：

| 字段 | 含义 |
| --- | --- |
| `mean` | 当前 `r0` 后验/模型预测的中心值，不是“保证正确” |
| `variance` / `uncertainty` | 对该预测有多不确定；不同类型不应强行比较数值尺度 |
| `evidence_count` | Beta 的等价样本量或实际 replay 次数，必须与概率一同看 |
| `source` | `group_beta`、`replay_calibrated_model`、`cold_start_default` 等可追溯来源 |
| `calibrated` | 是否在独立验证集上做过概率校准 |
| `expected_distance` | `d0`：真实记忆相对安慰剂造成的预期规范化环境动作距离 |
| `distance_distribution` | `D_env` 五个有序等级的预测概率；其加权期望等于 `expected_distance` |
| `p_any_change` / `p_material_change` | 任意动作变化、以及超过关键阈值的实质动作变化概率 |
| `negative/neutral/positive` | `u0` 三种团队边际效用类别的概率，不是 Agent 自己的得失 |
| `reason_codes` | 控制器选择 USE/VERIFY/IGNORE 的可审计原因 |

## 13. 当前代码的最小改动清单

本设计文档不直接修改代码。实际落地时优先做以下小改动，避免重构 G-Memory：

1. 为 trajectory 和 insight 建立稳定 `memory_id`，并在写库时持久化。
2. 在 `GMemory._extract_mas_message()` 前，将原始 `chain_of_states`、动作序列、`task_config`、seed/版本写到 replay artifact。
3. 给 `_retrieve_memory_raw()`/`retrieve_memory()` 增加可选的 metadata 返回值，至少带 rank、cosine、来源、Query Graph hop；保持默认返回兼容现有 workflow。
4. insight 检索可选返回完整字典，保留当前 `list[str]` 兼容路径。
5. 增加独立 `prior_data` 日志写入器和 `CandidateAdapter`，不要把大 JSON 或每次 DecisionContext 写入 Chroma metadata。
6. 在 workflow 的“构造 Agent prompt 之前”调用 prior module；真实 USE/VERIFY/IGNORE 以及最终呈现的候选 ID 都写入日志。

## 14. 最终设计立场

从现有 G-Memory 出发，最可信的起点不是训练一个声称同时理解可靠性、行为依赖和团队效用的大网络，而是：

```text
现有字段 -> 统一候选/检索格式 -> 透明弱先验与日志
source replay -> r 的可验证先验
paired action intervention -> d 的监督先验
fork/team rollout -> u 的团队效用先验
```

最终确实可以得到一个输入文本、向量和结构化字段，同时输出 `r0/d0/u0` 的 prior module；但它的可信度来自后续三种不同成本、不同语义的反事实/验证数据，而不是来自“把 Top-K 相似度、成功标签和 insight 分数喂进同一个网络”。在目前的数据条件下，先保留 G-Memory 的检索优势、补齐决策与干预数据契约，是风险最低且最能支撑研究结论的路径。
