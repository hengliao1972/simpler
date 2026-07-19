# FDWIC 泳道排他分区与闭合可行性分析

本文分析当前 FDWIC 泳道中各阶段为什么不能直接相加，并给出一套具有明确
业务语义的排他分区。目标是让同一父区间下的互斥子区间尽可能严格闭合，
同时保留 Atomic、lap marker 等非加和诊断信息。

本文前半部保留原始可行性分析；第 6 节持续记录 standalone 中已落地的
代码、观测口径和分级验证结果。前文定量结果来自真实 A5 level-4
诊断样本：

```text
outputs/TestPagedAttentionUnroll_Case1_20260718_161520/
    l2_swimlane_records.json
```

该样本为 schema v3、96 个物理子核、1 GHz cycle domain，共 973,430 条
记录，`dropped_records=0`。样本中的 Atomic 和 ClockBaseline 来自带逐
atomic 观察的诊断分支；数值只描述该次采集，不应外推成所有分支或无插桩
性能基线。

阅读时请区分时间快照：第 1～5 节中的“当前”指 2026-07-18 的历史
schema-v3 样本与当时源码；当前工作树的 schema-v4 实施和实测结果以第 6
节为准。

## 1. 目标、范围与口径边界

### 1.1 目标

本分析回答三个问题：

1. 当前泳道阶段为什么不能直接相加为总耗时；
2. 哪些区间已经有明确业务语义，哪些目前只能标为 residual；
3. 是否能以较小改动形成可验证、可闭合的分区。

本文讨论的是观测口径优化，不是 FDWIC 调度器性能优化。任何后续实现仍需
用相同观察模式做性能 A/B，不能把泳道插桩样本直接当作无诊断性能基线。

### 1.2 必须区分的三个“总耗时”

#### 单核 Submit replay 包络

```text
T_core_submit_envelope(i)
    = last_submit_end(i) - first_submit_start(i)
```

它可以在单核上精确分解为 Submit 并集与相邻 Submit 之间的 gap：

```text
T_core_submit_envelope(i)
    = union(Submit(i, *)) + InterSubmitResidual(i)
```

这是最容易形成严格加和关系的现有口径。

#### 单核 completion window

推荐将完整业务窗口定义为启动屏障结束后，到本核 final drain 完成为止：

```text
T_core_completion(i)
    = OrchestrationReplay(i) + FinalDrain(i)
```

如果还需要描述启动偏斜和 DFX 落盘，应在其外层另建
`WorkerObservedWindow`，而不是混入业务阶段。

#### 全局 makespan

```text
T_global_submit
    = max_i(last_submit_end(i)) - min_i(first_submit_start(i))
```

这是跨核墙钟包络。它的起点和终点可能属于不同物理核，不能用某个单核的
阶段和，也不能用 96 核 core-work 之和解释。

该样本的全局首末 Submit 包络为 6,088.433 us：起点来自 core 15，终点
来自 core 74。若要对它做可加业务归因，需要依赖图和可跨核的关键路径模型；
当前样本目录没有 `deps.json`，因此本文不声称已经得到全局关键路径分解。

### 1.3 当前分析器的口径

[分析脚本](../../../scripts/analyze_fdwic_swimlane_critical_path.py)
当前使用所有事件的最早 start 和最晚 end 形成 `global_span`，并按 phase
直接累计每条记录的 duration。

这会同时遇到两个问题：

- 分母是墙钟包络，分子可能是多核 aggregate core-work；
- `Submit`、Kernel、Atomic 和 lap marker 存在父子或重叠关系。

因此 `phase_rows()` 的结果适合观察各类事件规模，不是可加的时间预算。
分析器已有 `SubmitChildren + SubmitExclusive = Submit` 的区间并集逻辑，
但还没有构造 Submit 之外的顶层排他分区。

脚本中的 `child_intervals_by_core_task` 实际使用十列 ABI 的第 0、2 列，
即 `core_id` 和 `lane`，不是 `task_id`。当前行为等价于按同一 scalar lane
做时间包含判断，这对排他墙钟是合理的：Submit 开头可能执行前序 task 的
Kernel，不能因 task id 不同而漏减。但变量名容易误导，后续实现时应明确其
lane-temporal 语义。

## 2. 建议的业务层级

建议把时间事件分成“父区间、互斥子区间、Overlay”三种角色：

```text
WorkerObservedWindow（单核观察窗口）
├─ StartupBarrier
├─ WorkerCompletionWindow（业务完成窗口）
│  ├─ OrchestrationReplay
│  │  ├─ OrchestrationSetup
│  │  ├─ SubmitRuntimeEnvelope × N
│  │  │  ├─ EfDrain
│  │  │  │  ├─ PreviousTaskKernel
│  │  │  │  └─ DrainControl
│  │  │  ├─ Materialize
│  │  │  ├─ PrepareMap
│  │  │  ├─ Claim
│  │  │  ├─ Fanin
│  │  │  ├─ Register
│  │  │  └─ SubmitTail
│  │  │     ├─ WinnerBuild
│  │  │     ├─ LoserReplay
│  │  │     └─ AllocComplete
│  │  ├─ BetweenSubmitResidual × (N - 1)
│  │  └─ OrchestrationTail
│  └─ FinalDrain
└─ DFXFinalize

Overlay：Atomic、ClockBaseline、Commit、旧 Replay/Build/Alloc lap marker
```

这里严格沿用 1.2 节的口径：`WorkerCompletionWindow` 只包含
`OrchestrationReplay + FinalDrain`；`StartupBarrier` 与 `DFXFinalize` 属于更外层
的 `WorkerObservedWindow`，不能混进业务完成窗口。

同一层的 sibling 必须互斥；父区间不能与子区间再次相加。Overlay 可以画在
相同 lane 上，但不进入 stacked sum。

### 2.1 顶层分区

#### `StartupBarrier`

业务边界是 worker 初始化完成后，到所有 worker 到达启动屏障。

它包含：

- `started_count` 到达计数；
- 等待其他 worker；
- fatal 检查和 watchdog。

代码位于
[core_main.h](../../../src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/core_main.h)。
如果主指标仍是“全局首个 Submit 到末个 Submit”，该阶段应排除；如果查看
完整 worker 生命周期，则应单列。

#### `OrchestrationReplay`

它是每个 worker 重放整个 PA orchestration 的父区间，对应
`dist_submit_replay_orch(runtime)` 的完整调用。

业务内容包括：

- CCEC orchestration 参数 invalidate 和本地复制；
- PA tensor、shape、view 和参数对象构造；
- 全部 Submit；
- Submit 之间的 scalar 控制逻辑；
- orchestration 返回前的尾部逻辑。

当前代码只在调用前做 `TRACE_LAP_RESET`，没有记录该父区间。后续若补充
边界，应覆盖 `dist_submit_replay_orch()` 的调用前后，而不是用首末 Submit
近似整个 orchestration。

#### `OrchestrationSetup`

它是 `OrchestrationReplay.start` 到首个 Submit API 进入之间的排他部分。

PA Case1 中主要包括：

- 读取 batch、head、block 等 tensor metadata；
- 构造外部 Tensor wrapper；
- 创建循环不变量 `TensorCreateInfo`；
- 进入 batch/q-loop 前的控制逻辑。

当前 raw 没有 orchestration 父边界，所以它与 StartupBarrier 混合在
`PreFirstSubmitResidual` 中，尚不能可靠分离。

#### `BetweenSubmitResidual`

当前可证明的定义是：

> 同一物理 scalar lane 上，上一个 `Submit.end` 到下一个
> `Submit.start` 之间的全部时间。

PA Case1 中可能包括：

- `TensorCreateInfo`、tensor view；
- `L0TaskArgs::reset()` 和各类 `add_*()`；
- 循环、索引、条件和 `std::min` 等 scalar 控制；
- `TaskOutputTensors::get_ref()`；
- `rt_submit_*` 包装层的 fatal 查询和 `MixedKernels` 构造；
- 当前 Submit span 未覆盖的 prologue、epilogue 和返回开销；
- Submit 结束记录的 DFX 写入开销。

所以在没有更细边界前，应使用中性的 `BetweenSubmitResidual` 或
`OrchestrationExclusive`，不能直接命名为“参数构造耗时”。

PA orchestration 源码已有 `param_extract`、`create_info`、`tensor_view`、
`param_setup`、`scope_and_loop` 等聚合分类，可参考
[paged_attention_orch.cpp][pa-orch]。
但该实现受 `ENABLE_PROFILING` 控制并读取 AICPU `cntvct_el0`，不能原样
搬到 CCEC/AICore replay 路径。

#### `OrchestrationTail`

它是最后一个 Submit 返回后，到 `aicpu_orchestration_entry()` 返回之间的
排他时间，包括循环退出、scope 退出和函数返回逻辑。

当前 raw 没有 orchestration end 边界，因此它与 FinalDrain、
ClockBaseline 混合在 `PostLastSubmitResidual` 中。

#### `FinalDrain`

它对应 `dist_submit_drain_to_completion()` 的完整调用。结束条件是：

- 所有 worker 已完成 orchestration replay；
- 本核 private ring 为空；
- 没有待处理的 BlockWon lane。

业务内容包括：

- 继续导入和执行 ready task；
- 等待 fanin、`replay_done` 和其他 worker；
- 发布 completion flag 和 vend；
- 推进 frontier；
- 清理 joint task 状态。

第一版只需要一个完整 `FinalDrain` 父 span，再用已有 Kernel 区间得到：

```text
FinalDrain = FinalDrainKernel + FinalDrainResidual
```

其中 `FinalDrainResidual` 仍是控制、完成发布和等待的混合时间。没有更细的
互斥边界前，不能直接命名为“atomic wait”。

#### `DFXFinalize`

该分区包括 ClockBaseline、trace buffer flush 和 worker finish publication。
它是观察系统自身的尾部，不属于 PA 调度业务时间。

如果总口径是完整带插桩 worker wall，它必须单列；如果总口径是业务
completion window，应显式排除。

### 2.2 `SubmitRuntimeEnvelope` 的准确语义

当前 `Submit` 不是完整 `rt_submit_*` API 时间。它的起点位于
`ActiveMask`、joint 判断和 `dist_submit_begin()` 之后，终点是结束时间戳
采样时刻，早于 trace record 真正写入和函数返回。

因此建议把当前事件称为 `SubmitRuntimeEnvelope`：

> 一次 orchestration submit 调用中，DistSubmitCtx 初始化完成后，runtime
> frontend、机会式 progress 和可能的前序 Kernel 执行所形成的包络。

它不仅是“把当前 task 放入队列”的时间。尤其是 Submit 开头的 `EfDrain`
可能执行之前 task 的 Kernel。

### 2.3 Submit 子分区

| 分区 | 当前 task 归属 | 业务语义 |
| --- | --- | --- |
| `EfDrain` | 否/混合 | Submit 开头机会式推进已有任务，可能执行前序 Kernel |
| `Materialize` | 是 | 计算输出布局、处理 heap ring、构造输出 Tensor、推进 `heap_next` |
| `PrepareMap` | 是 | 按 task id 和窗口 `H` 清退 TensorMap 历史项 |
| `Claim` | 是 | 判断 lane 资格并通过 cursor atomic 竞争唯一 winner |
| `Fanin` | 是，winner-only | 从 owner 和 TensorMap 查找、去重依赖生产者 |
| `Register` | 是 | 将 INOUT/OUTPUT_EXISTING 注册到 TensorMap |
| `WinnerBuild` | 是，winner-only | 等待容量、申请 slot、复制 payload、发布 joint deposit |
| `LoserReplay` | 否/混合 | loser 退出前再次机会式处理 BlockWon |
| `AllocComplete` | 是，alloc winner | 检查 heap 并发布 allocation task 完成状态 |

`EfDrain` 应继续分为：

```text
EfDrain = PreviousTaskKernel + DrainControl
```

Kernel 在时间层级上属于包含它的 EfDrain 或 FinalDrain，但在任务归因上必须
保留 Kernel 自己的 task id，不能归给外层当前 Submit。

### 2.4 当前 lap marker 的语义问题

当前 `Build`、`Replay`、`Alloc` 使用 `TRACE_LAP`：

- `Build` marker 写在 `dist_submit_build_winner_task()` 之前；
- `Replay` marker 写在 loser 的 `drain_block_won()` 之前；
- `Alloc` marker 写在 `dist_submit_complete_alloc()` 之后，但其起点仍是较早的
  lap reset。

因此它们是累计边界，不是对应动作的独占耗时，会覆盖 Materialize、Claim、
Register 等已有阶段。它们必须作为历史 Overlay，不能进入加和分区。

若后续实施，应让 `Build`、`Replay`、`Alloc` 分别围绕实际动作形成显式
start/end span；这是 `SubmitResidual` 获得清晰业务语义的前提。

### 2.5 非加和 Overlay

以下事件应继续展示，但不得进入互斥时间预算：

| Overlay | 原因 |
| --- | --- |
| `Atomic` direct | 嵌套在 Claim、EfDrain、completion 等业务阶段内 |
| `Atomic` PollBatch | 是等待 episode 包络，可能包含交错的 direct atomic |
| `ClockBaseline` | DFX 计时与返回依赖校准，不是业务阶段 |
| `Commit` | 零时长 instant |
| 旧 `Build/Replay/Alloc` | lap 累计区间，与显式阶段重叠 |

Atomic 可用于调用次数、站点分布和局部等待诊断，但不能把其 duration 再加到
父阶段上。

## 3. 样本区间复算

### 3.1 顶层闭合

真实 A5 level-4 样本中，事件包络最长的 core 5 为 6,213.433 us。
按当前可观测边界可严格分成：

| 顶层分区 | 时间 |
| --- | ---: |
| `PreFirstSubmitResidual` | 97.498 us |
| `SubmitRuntimeEnvelope` 并集 | 4,730.084 us |
| `BetweenSubmitResidual` | 698.937 us |
| `PostLastSubmitResidual` | 686.914 us |
| 合计 | **6,213.433 us** |

这里的闭合是数学事实，但前后 residual 尚没有纯业务语义：

- `PreFirstSubmitResidual` 混合 StartupBarrier 和 OrchestrationSetup；
- `PostLastSubmitResidual` 混合 OrchestrationTail、FinalDrain 和
  ClockBaseline。

这正是补 `OrchestrationReplay` 与 `FinalDrain` 外层边界的主要理由。

### 3.2 Inter-submit 规模

按每个物理核的首末 Submit 包络统计：

| 角色 | `InterSubmitResidual` 中位数 | 占 Submit 包络中位比例 |
| --- | ---: | ---: |
| AIC，32 核 | 720.179 us | 13.18% |
| AIV，64 核 | 907.114 us | 15.09% |

因此 inter-submit 不是可以忽略的计时缝隙。优先补其排他口径，比继续增加
Submit 内部重叠事件更能提高总时间覆盖率。

### 3.3 Submit 内部排他分解

对 core 5 的 4,730.084 us Submit 并集，暂时可以形成：

| Submit 子分区 | 时间 |
| --- | ---: |
| `PreviousTaskKernel in EfDrain` | 550.853 us |
| `EfDrainControl` | 722.489 us |
| `Materialize` | 1,010.177 us |
| `PrepareMap` | 206.058 us |
| `Claim` | 656.820 us |
| `Fanin` | 20.026 us |
| `Register` | 274.385 us |
| `SubmitResidual` | 1,289.276 us |
| 合计 | **4,730.084 us** |

`SubmitResidual` 当前混合：

- 实际 WinnerBuild、LoserReplay、AllocComplete；
- 子阶段之间未打点的条件和调用逻辑；
- 子阶段结束记录写入等 DFX 开销；
- 不在现有显式业务 span 中的 Atomic 时间。

所以该数值能用于判断“还有多大未归因空间”，不能直接命名为某个业务热点。

### 3.4 全核 phase sum 的解释边界

样本中全核 `Submit` duration 总和约为 477.585 ms，而全局事件包络只有约
6.215 ms。前者是 96 核并行执行形成的 aggregate core-work，后者是墙钟。

因此分析器中的 all-core `span%` 可以超过 100%，这不是硬件时间异常，而是
维度不一致。报告应分别展示：

- global makespan；
- AIC/AIV 每核排他分区的 median、p95、max；
- aggregate core-work，并明确单位语义。

三者不能放在同一加和柱中。

## 4. 可行性分级与建议边界

### 4.1 仅修改分析器：高可行性

不增加设备插桩即可完成：

- 按 core/lane 对 Submit 区间求并集；
- 从首末 Submit 包络中取补集，生成 `BetweenSubmitResidual`；
- 把 lap marker、Atomic、ClockBaseline、Commit 标为 Overlay；
- 将 `SubmitExclusive` 改称 `SubmitResidual`；
- 对每个父区间执行整数 cycle 闭合断言；
- 分开报告 per-core、aggregate core-work 和 global makespan。

这一层可以精确闭合已有边界，但不能把 residual 自动归因成具体业务动作。

### 4.2 增加最小外层边界：中高可行性

最有价值的新增边界只有：

- `OrchestrationReplay`；
- `FinalDrain`；
- 可选的 `StartupBarrier` 和 `DFXFinalize`。

它们每核每轮只需少量记录，远小于逐 Submit 或逐 Atomic 事件规模。若新增
phase id 或改变既有 phase 语义，应同步更新 schema version、host 导出、
converter、测试和文档，禁止让新旧 raw 静默混用。

### 4.3 修正 Build/Replay/Alloc：中等可行性

将三个 lap marker 替换为真实动作 span，不必增加同数量级的记录，但会改变
阶段语义和指令布局。应先完成 schema/version 迁移，再对 A5Sim 和真实 A5 做
相同观察模式的验证。

### 4.4 细拆 orchestration residual：条件可行

PA 源码已有一套聚合分类设计，可以复用其业务分类，不应再发明另一套名字。
但其现有计时实现面向 AICPU，若要用于 CCEC/AICore，需要使用 FDWIC 已有
cycle 读取路径，并评估每次 lap 对 I-cache、worker 到达和竞争时序的扰动。

只有在补完顶层区间后，`OrchestrationExclusive` 仍是显著热点时，才值得增加
这一级细分。

## 5. 验收标准

后续若实施，至少满足以下条件。

### 5.1 语义验收

- 每种事件明确标为 parent、exclusive child 或 Overlay；
- 同一父区间下的 child 互斥；
- Kernel 保留自身 task id，不因嵌套在另一个 Submit 而改归属；
- residual 使用中性命名，不把补集直接解释成参数、atomic 或 I-cache 时间；
- global makespan 与 per-core/core-work 指标分开。

### 5.2 数学验收

所有计算使用原始整数 cycle：

```text
sum(exclusive_children(parent)) == duration(parent)
```

如果存在截断或边界裁剪，误差上限必须显式定义，不能依赖浮点 us 舍入掩盖
差值。还应验证：

- exclusive duration 不为负；
- Submit 区间在同一 core/lane 上不重叠；
- Overlay 不进入加和；
- 所有未归因时间只出现一次，并统一落入 residual。

### 5.3 数据完整性验收

- `dropped_records == 0`；
- PA Case1 每核恰有 1280 个 Submit，task id 为 0 到 1279；
- 96 核角色和物理 id 完整；
- schema version 与 phase 语义匹配；
- raw 到报告的记录数和关键汇总保持一致。

### 5.4 性能与扰动验收

- 新旧方案必须使用相同硬件、负载、构建布局和观察级别比较；
- 分别报告带插桩的时间闭合质量与关闭诊断后的业务性能；
- 不把 level-4 Atomic 样本与 phase-only 或 no-swimlane 样本直接相减；
- 若新增插桩显著改变 Submit span、fanin 重试或 RingBp，应先处理观察扰动，
  不能仅以“分区已经闭合”作为验收通过。

## 6. standalone 实施进展（2026-07-19）

### 6.1 已落地的最小 schema-v4

当前实施严格限定在 `tests/atomic_probe/pa_scheduler` 独立目录，
没有依赖 simpler runtime 源码或外部分析脚本。

- 每个 worker 新增一条 `OrchestrationReplay` 和一条 `FinalDrain`
  父 span，两者在原始 cycle 上首尾相接；
- 旧 `Build/Replay/Alloc` lap 不再产生，改用围住实际动作的
  `WinnerBuild/AllocComplete`；standalone loser 没有可单独计时的真实动作，
  因此 producer 不生成 `LoserReplay` 记录，converter 也拒绝旧记录；
- `EfDrain` 从历史 lap 改为明确 start/end span；
- raw `trace_schema_version` 升为 4，phase-only level-1 与 atomic level-4
  都携带 `fdwic_summary`，分析器可以独立验证 records 和
  `dropped_records=0`；
- `SubmitResidual` 和 `BetweenSubmitResidual` 由 converter/analyzer 根据已有
  span 的补集离线生成，不增加设备记录、字段或 `SYS_CNT` 读取；
- v4 merged 的 duration 事件只保留 Chrome Trace 必需的六个字段，业务语义
  编码在稳定名称中，不再逐事件复制 `args` 和 `cat`。raw 仍是权威取数件，
  merged 只负责可视化，排他统计只进入小型 analysis 报告；
- schema-v4 禁止未使用的历史 `DrainWon`；每个 Kernel 必须唯一包含于
  `EfDrain`、`WinnerBuild`、`AllocComplete` 或 `FinalDrain`，越界、多重归属和
  孤儿 Kernel 都使该轮分析失败。
- 除允许执行前序 task 的 Kernel 外，每个 exclusive child 的 task id 必须与
  包含它的 Submit 一致。

设备侧每 worker 固定新增 `OrchestrationReplay/FinalDrain` 两条父记录。
默认 b256 全局只有 1,024 条 `WinnerBuild` 和 256 条
`AllocComplete`；121,600 个 loser 不生成 tail。相对“每个 loser 一条
`LoserReplay`”的过程态，phase 记录固定减少 121,600；总 raw 事件差
还会受 Atomic/PollBatch 运行时次数影响。residual 只在离线产物中出现，
不消耗 trace buffer。

这里的 Kernel 门禁只证明它在时间上唯一落入上述四类支持的父区间；
不再额外建立 `WinnerBuild ↔ Kernel ↔ Commit` 的
`(core, task, function)` 身份链。完整 PA 协议正确性仍由 raw 导出前的 host
`Validate()` 断言负责；排他分析器保持聚焦时间容器、整数闭合和丢记录门禁，
不为了重复 host 已有的协议校验而继续膨胀记录或报告。

### 6.2 排他分析器与闭合门禁

`swimlane_exclusive_analyzer.py` 直接复用本目录 converter 的 raw/schema
校验，用原始整数 cycle 完成：

```text
Submit = EfDrain + Materialize + PrepareMap + Claim + Fanin + Register
       + WinnerBuild + AllocComplete + SubmitResidual

SubmitEnvelope = SubmitUnion + BetweenSubmitResidual
EfDrain = contained Kernel union + EfDrainControl

OrchestrationReplay = Setup + Submit union + BetweenSubmitResidual + Tail
FinalDrain = contained Kernel union + FinalDrainResidual
WorkerCompletion = OrchestrationReplay + FinalDrain
```

分析器不用 `max(0, residual)` 吞掉越界；子区间重叠、Kernel 穿越父边界、
孤儿 Kernel、父 span 数量错误、逐核 task id 不连续、角色拓扑不全或 dropped
非零都直接失败。报告分开输出 global Submit makespan、aggregate core-work 和 AIC/AIV
每核分布，不再用 96 核累加值除以全局墙钟包络。

`run.sh swimlane` 已串起三份产物，每个 writer 都在单件完成后再原子
发布自己的文件；这是 raw → merged → exclusive report 的逐件流水线，
不是三件整体的目录级事务：

```text
l2_swimlane_records.json
merged_swimlane.json
swimlane_exclusive_analysis.json
```

### 6.3 已完成的分级验证

- Python 完整回归：PMU HTML、PMU sidecar、converter 和 exclusive analyzer
  共 100 项全部通过；`git diff --check` 通过；
- 历史 schema-v3 真机百万行样本：967,307 条事件通过，
  Submit、EfDrain 和 Submit envelope 均精确闭合，全局 Submit makespan
  为 5,774,295 cycle；
- 删除 loser 标记后的 CCEC A5 b256 规模门禁：845,813 条 raw 事件、零丢记录，
  全局 Submit makespan 为 5,326.055 us，六组闭合全部通过。raw 为
  56,212,672 bytes；旧 merged 为 248,767,986 bytes；同一 raw 经当前瘦身
  converter 得到 138,349,686 bytes，减少 44.4%，且不改变设备采集。
  该轮早于最终相邻边界复用，只作规模/容量证据；
- 当前边界迭代只跑 CCEC A5 b1。最新一轮为 4,118 条 raw 事件、零丢记录，
  全局 Submit makespan 为 89.313 us，六组闭合全部通过；当前 converter 直接
  生成的 merged 为 428,455 bytes。

当前可直接使用的产物：

```text
outputs/pa_scheduler_swimlane_20260719_103435_542368/ccec/l2_swimlane_records.json
outputs/pa_scheduler_swimlane_20260719_103435_542368/ccec/merged_swimlane_thin.json
outputs/pa_scheduler_swimlane_20260719_103435_542368/ccec/swimlane_exclusive_analysis.json
outputs/pa_scheduler_swimlane_20260719_110756_584549/ccec/
```

早期 CPU/CCEC/AscendC b1 及 CCEC b256 schema-v4 过程态样本曾用于建立
父区间和闭合门禁，但 raw 仍含 `LoserReplay`；当前 converter 会直接
拒绝它们。上述无 loser 的规模门禁和最新 b1 证明当前 standalone
的 schema-v4 数据完整性、语义和数学闭合成立；
它仍不是完整 PA Case1 依赖图的性能结论。插桩扰动还需用相同 level-4
观察口径解释，不用单轮差值宣称性能改善。含
`LoserReplay` 的历史过程态曾在相同 level-4 配置连续跑三轮 b256：

```text
5785.939 / 5503.109 / 5381.844 us，中位数 5503.109 us
```

过程态的完整 raw + merged + exclusive 导出轮为 5680.749 us；历史
schema-v3 同负载 level-4 导出轮为 5774.295 us，二者单轮差为
93.546 us（v4 约低 1.62%）。这个差值显著小于 v4 自身三轮的
404.095 us 极差，所以只能作为历史观察扰动证据，不能外推为
当前最终边界版本的性能改善或不回退结论。

### 6.4 residual 的当前收敛状态

设备端不为 residual 增加记录。converter 与 analyzer 分别从同一
raw 独立求补集：前者在 Perfetto 中以 `submit_residual` 表示 Submit
内部 child-to-child gap，以等长名称 `submit_tail_gap` 表示最后一个
child 到 `SubmitEnd` 的尾段，并绘制 `between_submit_residual`；后者不读
merged，分别汇总 `submit_internal_residual`、`submit_tail_residual` 和
`between_submit_residual`，直接按“前一边界 → 后一边界”校验闭合，避免
给每个 gap 增加属性。

边界迭代复用已有相邻 phase 的 end 作为后继 begin：`EfDrain → Materialize
→ PrepareMap`，以及按分支连接 `Claim/Fanin/Register/WinnerBuild/
AllocComplete`。这不会新增时间戳，并使最新 b1 的全部 child-to-child gap
归零。该轮 Submit aggregate core-work 为 2,241,892 cycle，其中
`submit_internal_residual=0`；最后一个真实 child 到 `SubmitEnd` 的
`submit_tail_residual` 为 184,788 cycle，占 8.2425%：

```text
Register → SubmitEnd       155,827 cycle / 380 次 loser
Claim → SubmitEnd           27,305 cycle / 95 次 Alloc loser
WinnerBuild → SubmitEnd      1,430 cycle / 4 次 winner
AllocComplete → SubmitEnd      226 cycle / 1 次 Alloc winner
```

这些尾段主要包含最后一条 phase 记录发布、分支收束、`submits++` 和读取
`SubmitEnd`。当前不把它伪装成 loser 业务阶段。同轮 Submit 间 residual 为
155,679 cycle，占逐核首末 Submit 包络的 6.493%；它明确覆盖上一个 Submit
尾记录发布、`AcceptTaskOutputs()`、下一个 `Build*Args()` 以及调用衔接。

后续边界迭代默认只跑 A5 b1；b256 仅在阶段性规模/容量收口或明确要求时运行，
避免用大样本反复验证纯结构改动。

2026-07-19 按明确要求完成一次当前生产者的 CCEC b256 规模复核：

```text
outputs/pa_scheduler_swimlane_20260719_114815_617346/ccec/
```

该轮使用 `real-compute/6,28,4,1`，全局 Submit 为 5,360.061 us；raw
839,526 条、`dropped=0`，merged 1,085,191 条事件。raw/merged 分别为
55,791,947/88,775,668 bytes，所有语义断言和六类整数闭合通过。
Submit aggregate core-work 为 433,383,588 cycle：内部 residual 仍为 0，
尾部 residual 为 41,008,786 cycle（9.4625%）；逐核首末 Submit 包络为
500,448,909 cycle，Submit 间 residual 为 67,065,321 cycle（13.4010%）。
Perfetto 中恰有 122,880 条 `submit_tail_gap` 和 122,784 条
`between_submit_residual`，分别对应 `96*1280` 个 Submit 尾段和
`96*(1280-1)` 个相邻 Submit 间段；重分类未增加设备记录或逐事件字段。

### 6.5 第一项真实消减：winner 冷路布局

对基线优化后 IR 的核对表明，Alloc loser 在 Claim 记录发布后要跨过约
2,200 个 IR 编号的 winner 重块，其他四类 loser 在 Register 记录发布后要
跨过约 3,300 个 IR 编号的 winner 重块，才能进入公共 Submit 尾部。两个分支
原先都没有概率信息，而每个 task 实际只有 1/96 worker 为 winner。B256 基线
中 121,600 次 loser 占尾部 cycle 的 99.62%，因此先做只改变基本块布局的
单变量实验：在 Alloc 和非 Alloc 的两个重型 winner 分支上使用
`__builtin_expect(winner, 0)`。该改动不移动时间边界，不增删记录，也不改变
Claim、完成发布、失败返回或 winner 计算语义。

CCEC 产物的 `.text` 由 AIC/AIV 的 `0x5d708/0x5ec38` 增至
`0x5eb18/0x5fc38`，分别增加 5,136/4,096 bytes。这个体积代价不能忽略；
后续若继续叠加布局改动，必须同时检查代码尺寸和 Submit PMU 的 I-cache 指标。

B1 采用“基线重编译三轮 → 候选重编译三轮”的复测，所有轮次均保持 96 核、
每核 5 Submit、`dropped=0`、phase/atomic 记录闭合和两类整数闭合：

| 口径 | 基线三轮 | 候选三轮 | 中位数变化 |
| --- | --- | --- | --- |
| `submit_tail_residual` | 193,292 / 184,214 / 191,029 cycles | 124,758 / 121,064 / 117,931 cycles | 191,029 → 121,064，-36.63% |
| `submit_envelope` | 2,426,417 / 2,289,891 / 2,540,203 cycles | 2,273,874 / 2,251,771 / 2,322,538 cycles | 2,426,417 → 2,273,874，-6.29% |

尾部变化集中在高频 loser：Register→SubmitEnd 的单次均值由约
405～422 cycles 降至 251～268 cycles，Claim→SubmitEnd 由约 304～361
cycles 降至 173～286 cycles。低频 winner 的 WinnerBuild/AllocComplete 尾部
变长，但 B1 每轮仅有 4/1 次，未抵消 loser 收益。

阶段性 B256 规模门禁使用：

```text
outputs/pa_scheduler_swimlane_20260719_123520_660296/ccec/
```

与 6.4 节基线相比：

| 指标 | 基线 | winner 冷路布局 | 变化 |
| --- | ---: | ---: | ---: |
| 全局首末 Submit | 5,360.061 us | 5,278.401 us | -81.660 us，-1.52% |
| Submit 尾部 residual | 41,008,786 | 27,155,661 cycles | -33.78% |
| Submit union | 433,383,588 | 414,313,448 cycles | -4.40% |
| Submit 间 residual | 67,065,321 | 69,022,235 cycles | +2.92% |
| Submit envelope | 500,448,909 | 483,335,683 cycles | -3.42% |

因此尾部下降没有被等量搬到 Submit 间：即使计入间隙，逐核完整 Submit 区间
仍净减 17,113,226 cycles。候选 raw 为 839,465 条、`dropped=0`，所有语义
断言和整数闭合通过；raw/merged 分别为 55,787,614/88,758,522 bytes，未因
该优化增加观察数据量。

关闭泳道记录后又做了候选—基线—候选的 ABA 验证，每组 5 轮：候选两组
中位数分别为 3.665017/3.715385 ms，夹在中间的基线为 3.988115 ms，候选
分别快 8.10%/6.84%。这说明收益并非只存在于 trace 发布路径。墙钟仍受
frontier helping、kernel 长尾和 winner 分布影响，因此后续迭代继续以原始
逐核 cycle 闭合作为主证据，以关闭泳道的 Submit span 作为实际性能门禁。

### 6.6 同步真实 PA：恢复 atomic 观测接入后的布局回退

standalone 的两处低概率 winner 分支严格映射到真实
`submit_runtime.h` 的 kernel BuildWinner 和 AllocComplete 分支。winner-only
Fanin 分支保持不动；真实 kernel loser 仍执行既有 Replay lap 和
`drain_block_won()`。补丁只把这两处改为
`__builtin_expect(is_winner, 0)`，没有增删 trace/atomic 调用，也没有移动
Submit 或子阶段边界。

正式 A/B 使用真实 A5 Case1、256 batch、level-1 phase 泳道和当前 CANN 9.1。
首先重建当前源码的 host/AICPU/AICore runtime；早先 7 月 17 日的通用 runtime
缓存无法解释当前新增 phase 枚举，曾产生一轮只有 `Kernel/Alloc` 名称的无效 raw，
该轮已排除。有效样本均满足 96 核、每核 1,280 Submit、全局 122,880 Submit，
固定 phase 数量与 task id 连续性全部通过。正式轮次对称关闭 case 结束后的 merged
转换，只省去离线 221 MB 可视化文件，不改变设备采集；基线和候选各另留一轮完整
merged warmup。

| 构建 | warmup | 正式三轮首末 Submit | 中位数 |
| --- | ---: | --- | ---: |
| 当前 atomic 观测版基线 | 5.621733 ms | 5.774073 / 5.596633 / 5.631038 ms | 5.631038 ms |
| 两处 winner 冷路布局 | 5.171619 ms | 5.165473 / 5.198404 / 5.192087 ms | 5.192087 ms |

当前源码内的相对下降为 0.438951 ms，即 7.80%。逐核 Submit span 中位数也由
5.530236 ms 降为 5.094550 ms，说明变化不是单个全局长尾造成。三个指标分别取
三轮中位数时，Submit 累计由 439,990,736 降为 403,166,351 cycles，Submit 间
累计由 81,293,671 降为 76,481,692 cycles，逐核首末 Submit 区间累计由
520,995,184 降为 479,761,953 cycles；每一轮内部都满足
`Submit + between = envelope` 整数闭合，不能把三个独立中位数再次相加。

收益主要落在 AIV loser：AIC/AIV loser 单次均值的三轮中位数分别由
2,864.829/3,558.067 降为 2,839.587/3,261.205 cycles。按 task 类型，QK/SF/PV/UP
loser 分别下降约 289/306/376/194 cycles，Alloc loser 则增加约 128 cycles；
高频路径的净收益覆盖了低频路径回退。

这项结果必须放回历史基线解释。7 月 17 日 pre-atomic 三轮中位数为
5.115620 ms；当前候选 5.192087 ms 仍慢约 1.49%，因此本节是**恢复观测接入后的
布局回退**，不是在旧 5.1 ms 上新增 7.8% 收益。编译产物提供了对应证据：

| 产物 | pre-atomic 缓存 | 当前基线 | winner 冷路布局 |
| --- | ---: | ---: | ---: |
| AIC `dist_engine .text` | `0x13bb8` | `0x54ce0` | `0x54d90` |
| AIV `dist_engine .text` | `0x13c10` | `0x56e80` | `0x572f8` |
| 最终 PA ELF `.text` | `0x3e518` | `0xb6b50` | `0xb7050` |
| AIC/AIV `dist_submit_impl` | 42,136 / 42,152 B | 100,724 / 102,588 B | 100,860 / 103,676 B |

level 1 的 raw 没有 Atomic 或 ClockBaseline 记录，说明本轮约 0.44 ms 回退不是
atomic 落盘或逐 atomic 时间戳的直接成本。但同一 ELF 必须允许运行时切到 level 4，
完整 atomic 慢路仍被编译并大量内联；level 1 只走
`g_fdwic_swimlane_level < 4` 快速返回。两条概率提示在不删除观察能力的情况下恢复
大部分性能，结合 AIV loser 的变化，当前证据最支持“诊断代码膨胀后热控制流布局/
取指恶化”为主要原因。尚余约 1.5% 历史差值，继续按单变量方式优化 level-4 冷路；
专门采 I-cache miss 的诊断构建则应整体编译去除泳道与 atomic，避免观察代码污染
待测 ELF。

权威样本为：

```text
基线：outputs/TestPagedAttentionUnroll_Case1_20260719_130443/
      outputs/TestPagedAttentionUnroll_Case1_20260719_130608/
      outputs/TestPagedAttentionUnroll_Case1_20260719_130728/
候选：outputs/TestPagedAttentionUnroll_Case1_20260719_131319/
      outputs/TestPagedAttentionUnroll_Case1_20260719_131435/
      outputs/TestPagedAttentionUnroll_Case1_20260719_131551/
完整候选泳道：outputs/TestPagedAttentionUnroll_Case1_20260719_131116/merged_swimlane.json
```

### 6.7 外提 level-4 atomic 冷代码，消除 level-1 取指回退

对象审计确认，atomic 接入后不只是五类 wrapper 被内联：每个普通 phase/lap
边界还展开了 `fdwic_atomic_poll_boundary_at()` 的十类 PollBatch 遍历与记录写入。
level 1 虽然在运行时快速返回，这些冷代码仍被复制进 Submit、Alloc、Materialize、
Heap wait、drain 等函数，造成上一节记录的数倍 `.text` 膨胀。

按单变量顺序做了两步外提，均不增加设备记录或字段：

1. `fdwic_swimlane_detail_record_atomic()` 设为设备端 `noinline`。Atomic 的
   `begin -> 指令 -> end` 仍留在原 wrapper，只有 `end` 之后的 GM 记录发布共享；
   level 1 不调用它。AIC/AIV `.text` 由 347,536/357,112 B 降为
   319,384/328,656 B，`dist_submit_impl` 由 100,860/103,676 B 降为
   93,144/95,756 B。真实 A5 三轮为 5.461806/5.077269/5.096506 ms，
   中位数 5.096506 ms，较 6.6 节候选下降 1.84%。
2. `fdwic_atomic_poll_boundary_at()` 保留原内联快速门，只把已确认
   `level >= 4 && active_mask != 0` 后的十类遍历外提到
   `fdwic_atomic_poll_boundary_slow()`。因此 level 1 仍只执行原有条件判断，
   不新增 call/ret；level 4 的 `end_cycle` 仍在调用前取得，PollBatch 时间边界
   不包含记录发布。AIC/AIV `.text` 进一步降为 66,768/67,120 B，
   `dist_submit_impl` 降为 18,812/18,872 B，实际 PA ELF `.text` 为
   178,768 B。

第二步的 level-1 暖测为 4.816453 ms；正式三轮为：

| 样本 | 首末 Submit | Submit 数 | 结果 |
| --- | ---: | ---: | --- |
| `135241` | 4.821897 ms | 122,880 | PASS |
| `135340` | 4.890447 ms | 122,880 | PASS |
| `135435` | 4.752956 ms | 122,880 | PASS |

三轮中位数 4.821897 ms，较两处 winner 冷路的 5.192087 ms 再下降 7.13%，
较历史 pre-atomic 5.115620 ms 低 5.74%。每轮均为 96 核，固定 phase 数量、
逐核 Submit 数和 task id 连续性通过；没有 Atomic/ClockBaseline level-4 记录混入
level-1 结果。该收益来自减少冷诊断代码复制后形成的目标代码与布局变化，不能在
没有 PMU 取证时进一步写成某个确定的 I-cache miss 数值。

保留完整 atomic 能力的 level-4 真机门禁位于：

```text
outputs/TestPagedAttentionUnroll_Case1_20260719_135629/
```

该轮 971,044 条记录、107,608 条 Atomic、115,309 次 atomic 调用、8,056 次
批处理轮询和 355 条 PollBatch，严格满足
`107608 = 115309 - 8056 + 355`；96 核各有两条 ClockBaseline，
`dropped_records=0`。生产 converter 对全部 site/op、`result_used`、
`return_ready`、PollBatch flags 和 metadata/raw 计数完成逐条校验并通过。

结论是：此前主要回退属于可消除的编译期复制与热路布局成本，不是开启普通泳道
后必须支付的 atomic 落盘成本。专门分析 I-cache miss 时仍应使用独立
`submit-pmu` 构建，把普通泳道和 atomic 泳道整体编译去除；这样测到的是调度源码
本身，而不是本节虽已降温、但仍存在于 ELF 的诊断代码。

## 结论

历史 schema-v3 泳道不闭合的根因不是缺少一次 duration 求和，而是
Submit 父包络、Kernel 嵌套、Atomic/Clock/lap Overlay 和缺失的 orchestration/final
drain 外层边界被混在同一加和口径中。

第 6 节记录的 schema-v4 已完成当时建议的第一轮落地：分析器以原始
整数 cycle 产生严格排他的 per-core 分区，设备侧新增
`OrchestrationReplay/FinalDrain` 父 span，并用真实的
`WinnerBuild/AllocComplete` 替换历史 lap；没有真实工作的 loser 不产生伪阶段。
未归因时间仍只落入中性 residual，不被擅自命名为 atomic、I-cache 或参数构造
开销。当前方向是继续减少 residual，而不是通过增加字段、记录或虚假业务名称
把它藏起来。

[pa-orch]: ../../st/a5/tensormap_and_ringbuffer/paged_attention_unroll/kernels/orchestration/paged_attention_orch.cpp
