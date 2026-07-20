# FDWIC 泳道排他分区与闭合可行性分析

本文分析当前 FDWIC 泳道中各阶段为什么不能直接相加，并给出一套具有明确
业务语义的排他分区。目标是让同一父区间下的互斥子区间尽可能严格闭合，
同时保留 Atomic、lap marker 等非加和诊断信息。

本文前半部保留原始可行性分析；第 6 节持续记录 standalone 中已落地的
代码、观测口径和分级验证结果。只关心当前实现时，应直接阅读 6.8～6.11 节；
前文定量结果来自真实 A5 level-4
诊断样本：

```text
outputs/TestPagedAttentionUnroll_Case1_20260718_161520/
    l2_swimlane_records.json
```

该样本为 schema v3、96 个物理子核、按 1 ns/tick 解释的 `SYS_CNT`
时间戳域，共 973,430 条记录，`dropped_records=0`。样本中的 Atomic 和
ClockBaseline 来自带逐
atomic 观察的诊断分支；数值只描述该次采集，不应外推成所有分支或无插桩
性能基线。

阅读时请区分时间快照：第 1～5 节中的“当前”指 2026-07-18 的历史
schema-v3 样本与当时源码；当前工作树的 schema-v4 实施和实测结果以第 6
节为准。

计时域必须区分：本机同一轮 CANN 9.1 profiler 的 DeviceInfo 同时报告
`hwts_frequency=1000 MHz`、`aic_frequency=1650 MHz` 和
`aiv_frequency=1650 MHz`。因此泳道 `start/end` 的 `SYS_CNT` 按
1 ns/tick 解释；这不表示 AIC/AIV 执行频率是 1 GHz，也不外推为
其他芯片型号的通用规格。分析 JSON 中 `*_cycles` 是
沿用的字段名，物理单位实际是整数 SYS_CNT tick；跨核求和应称为
core-ns。PMU total/scalar 等事件才是执行 cycle，本机同窗比值约为
1.65 cycles/ns，换算时间必须除以对应的 cycles/ns，不能把两个
计时域混用。

上述设备频率证据位于
`outputs/pmu_validation/icache_single_128x5_20260718_090151_3235468/`
`PROF_000001_20260718090151604_03235516CDDPPJLE/device_0/info.json.0`。真实
本机 CANN 9.1 的 `info_conf_reader.py` 也明确使用 HWTS frequency 换算
`sys_cnt/delta_syscnt`，DAV3510 的 `GetSystemCycleImpl()` 则直接读
`SYS_CNT`。真实 PA orchestration 中 AICPU 使用的 50 MHz `cntvct_el0`
是第三个计时域，不属于本文的 AIC/AIV `SYS_CNT` 泳道时间戳。

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

## 2. 历史 schema-v3 阶段建议

当时建议把时间事件分成“父区间、互斥子区间、Overlay”三种角色：

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
Submit = EfDrain + Claim + Materialize + PrepareMap + Fanin + Register
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

当前 compete-first eager 实现把 Claim 移到同步 callback 构参之前。
callback 构参没有新增 raw 字段或设备记录，其耗时由现有
`Claim.end -> Materialize.begin` 内部 residual 表达；这与
submit-PMU 的 Claim/Materialize 边界保持同一业务顺序，同时不扩大 profiling raw。

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

### 6.4 compete-first 前历史快照：residual 的当时收敛状态

> 本节至 6.11 的数值、布局图与“当前源码”表述，都是
> compete-first eager 移植前的历史快照。它们保留作为边界收敛与
> 旧样本的取证记录，不得用来解释移植后的阶段顺序或绝对耗时。
> 移植后的权威布局与实测数据见 6.12。

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

### 6.8 schema-v4 权威完整性样本的布局

本节冻结 schema-v4 的布局、边界归属与整数闭合，作为后续瓶颈
取证的共同参照。性能优化在这些语义解释清楚之前暂停。权威
完整性输入为：

```text
outputs/pa_scheduler_swimlane_20260719_123520_660296/ccec/
    l2_swimlane_records.json
    merged_swimlane.json
    swimlane_exclusive_analysis.json
```

该轮为 CCEC A5、`real-compute/6,28,4,1`、256 batch：

- schema v4，32 个 AIC 与 64 个 AIV；
- 每核 1,280 个 Submit，共 122,880 个；
- raw 共 839,465 条记录，`dropped_records=0`；
- 本机 profiler 元数据报告 HWTS/SYS_CNT 时间戳域为 1000 MHz，
  因此本文将泳道 `start/end` 的单段差值按 1 ns/tick 解释；
  这不是 AIC/AIV 的执行频率。PMU cycle 域经本机同窗比值约为
  1.65 GHz（ALL/AIC/AIV 分别为 1.649844/1.650062/1.649731 cycles/ns）；
- 全局首个 Submit 开始到最后一个 Submit 结束为 5.278401 ms；
- 下文的 aggregate core-work 是 96 核各自时间的总和，不是 5.278401 ms
  墙钟的切片。

该 raw 产生于 12:35，早于 12:55 提交的 standalone winner 冷分支
布局提示。后者不改变 span 边界、事件数和闭合公式，因此本节仍是
语义与完整性的权威样本；但本节的绝对时间不冒充当前源码性能。
当前工作树重编后的两轮 B256 分布见 6.10.6。

该样本的实际布局如下，当前源码仍使用同一套边界。
`WorkerCompletion`、`OrchestrationSetup`、
`SubmitTransition`、`SubmitFinalize` 和父区间内的 Kernel 补集均由已有边界
离线推导，不增加设备字段、记录或 `SYS_CNT` 读取：

```text
StartupBarrier                                      窗口前；无完整父边界，未量化

WorkerCompletion                                   离线推导
├─ OrchestrationReplay                             每核 1 条 raw 父区间
│  ├─ OrchestrationSetup                           初始化/首批 Alloc 构参 → 首个 Submit.start
│  ├─ Submit × 1280/core
│  │  ├─ EfDrain
│  │  │  ├─ Kernel                                 嵌套，不重复相加
│  │  │  └─ DrainReady 的 scalar 控制/完成发布
│  │  ├─ Materialize
│  │  ├─ PrepareMap
│  │  ├─ Alloc 路径：Register → Claim → AllocComplete（条件执行）
│  │  ├─ 非 Alloc：Claim → Fanin（条件执行）→ Register
│  │  │                         → WinnerBuild（条件执行）
│  │  └─ SubmitFinalize                            最后 child.end → Submit.end
│  ├─ SubmitTransition × 1279/core                 相邻 Submit.end → start
│  └─ OrchestrationTail                            最后 Submit.end → 父区间结束
└─ FinalDrain                                      每核 1 条 raw 父区间
   ├─ Kernel                                       嵌套，不重复相加
   └─ FinalDrainControlWait                        Kernel 之外的控制/同步/清空

DFXFinalize                                        窗口后；无完整父边界，未量化
Atomic / RingBp / ClockBaseline / Commit           Overlay，不进入加和
```

当前可视化和分析 JSON 尚保留原字段名：Perfetto 中的 `submit_tail_gap` 对应
本文 `SubmitFinalize`，`between_submit_residual` 对应
`SubmitTransition.<前一类→后一类>`；分析 JSON 中分别为
`submit_tail_residual` 和 `between_submit_residual`。本文给出的是由现有边界
可证明的显示语义，不代表新增了一类设备记录。

顶层 aggregate core-work 严格闭合为：

| 当前区域 | aggregate core-ns | 占 WorkerCompletion | 次数 | 每次/每核平均 |
| --- | ---: | ---: | ---: | ---: |
| OrchestrationSetup | 133,971 | 0.0264% | 96 | 1.396 us/核 |
| Submit 内部并集 | 414,313,448 | 81.5424% | 122,880 | 3.372 us/次 |
| SubmitTransition | 69,022,235 | 13.5845% | 122,784 | 562.1 ns/次 |
| OrchestrationTail | 53,303 | 0.0105% | 96 | 555.2 ns/核 |
| FinalDrain | 24,572,973 | 4.8363% | 96 | 255.968 us/核 |
| **WorkerCompletion** | **508,095,930** | **100%** | 96 | 5.293 ms/核 |

其中：

```text
OrchestrationReplay
  = 133,971 + 414,313,448 + 69,022,235 + 53,303
  = 483,522,957 SYS_CNT ticks

WorkerCompletion
  = 483,522,957 + 24,572,973
  = 508,095,930 SYS_CNT ticks
```

所有等式都在原始整数 SYS_CNT tick 上精确成立。分析器还验证了：同核 Submit 不重叠、
子阶段不重叠、task id 连续、父区间每核恰好一条、相邻父区间首尾相接，以及
Kernel 具有唯一受支持的时间父区间。

事件数也与 Case1 的结构精确一致：

| 事件 | 实际次数 | 结构预期 |
| --- | ---: | ---: |
| Submit / EfDrain / Materialize / PrepareMap / Claim / Register | 各 122,880 | `96 × 256 × 5` |
| Fanin / WinnerBuild | 各 1,024 | `256 × 4` |
| AllocComplete | 256 | `256 × 1` |
| Kernel / Commit | 各 1,024 | `256 × 4` |
| OrchestrationReplay / FinalDrain | 各 96 | 每核各 1 条 |
| ClockBaseline | 192 | 每核 2 条 |
| RingBp | 0 | 本轮未发生 ring/heap 背压 |

1,024 个 Kernel 中，1,015 个唯一落在 EfDrain，9 个唯一落在 FinalDrain，
没有孤儿、跨界或多重归属。Kernel 通常不在 `WinnerBuild` 内执行：
`WinnerBuild` 只构造私有 slot，实际 engine 工作等依赖 ready 后由
`DrainReady()` 执行。

Atomic 记录的身份同样闭合，但它是 Overlay：97,449 条记录代表 99,302 次
源码调用，其中 2,093 次轮询调用被压成 240 条 PollBatch，满足：

```text
97,449 = 99,302 - 2,093 + 240
```

Atomic 记录 duration 的 aggregate 为 59,250,773 SYS_CNT ticks；direct 与 PollBatch
可能互相嵌套，因此该值既不能与排他阶段相加，也不能当成 Atomic 的区间并集。

### 6.9 每个 span 的边界、业务语义与源码

#### 6.9.1 相邻边界复用的真实含义

当前为减少 `SYS_CNT` 读取和 raw 体积，让后一个阶段直接复用前一个阶段的
`end`。`TraceTimestamp()` 先采样，前一阶段的 `WriteTrace()` 随后才向 GM
写 64-byte record。因此 `submit_internal_residual=0` 证明的是时间轴连续，
不代表各 span 已经剥离观测代码，也不代表阶段间没有 scalar 胶水。

当前记录成本的实际归属为：

| 报告中的 span | 除同名业务动作外，还实际覆盖 |
| --- | --- |
| EfDrain | 直接从 Submit.start 开始；不吸收上一条 child record |
| Materialize | EfDrain record 发布和进入 `MaterializeTask()` 前的衔接 |
| PrepareMap | Materialize record 发布 |
| Alloc Register / 非 Alloc Claim | PrepareMap record 发布 |
| Alloc Claim | Alloc Register record 发布 |
| 非 Alloc Fanin 或 Register | Claim record 发布 |
| 非 Alloc 且执行 Fanin 后的 Register | Fanin record 发布 |
| WinnerBuild | Register record 发布 |
| AllocComplete | Claim record 发布 |
| SubmitFinalize | 最后一条 child record、分支汇合、`++submits` 和 SubmitEnd 取时 |
| SubmitTransition | 前一条 Submit record、返回控制、下一任务构参和下一次 `BeginSubmit()` |
| OrchestrationTail | 最后一条 Submit record、最后一次返回和循环退出 |

Atomic span 自身都在对应 Atomic record 写入前结束，但 record 成本的父阶段
归属不能一概而论：direct Atomic 的 record 在外围 phase.end 之前发布，通常仍
算入当前排他 span；PollBatch 若由 phase.end 的 `TraceTimestamp()` 关闭，其
record 在已采样的 end 后写入，通常由后一个 span 吸收；若在 Kernel.begin 被
关闭，record 会进入随后 Kernel 完整区间；若由 `AtomicPollRegionEnd()` 在
普通控制流中关闭，则仍留在当时的外围父区间。权威 B256 的 9 个 FinalDrain
Kernel 中有 17 条这类 begin-boundary PollBatch，EfDrain 的 1,015 个 Kernel
未发现该情况。由此，泳道 span 表示“以业务动作命名的完整观察窗口”，不是
去掉 DFX 后的纯函数微基准。后续若判断具体函数体是否是瓶颈，必须与去掉泳道的
`submit-pmu` 诊断构建交叉验证，不能直接按同名 span 机械相减。

关键实现位于：

- `TraceTimestamp()`：`common/pa_scheduler_core.h:58-72`；
- `WriteTrace()`：`common/pa_trace.h:521-567`；
- 完整 Submit：`common/pa_scheduler_core.h:541-737`；
- orchestration 与 FinalDrain：`common/pa_scheduler_core.h:880-997`。

#### 6.9.2 Submit 内部 span

以 `SubmitUnion=414,313,448 core-ns` 为 100%，当前完整分布为：

| Span | aggregate core-ns | Submit 占比 | 次数 | 单次平均 |
| --- | ---: | ---: | ---: | ---: |
| Materialize | 136,717,945 | 32.9987% | 122,880 | 1,112.6 ns |
| Claim | 79,359,417 | 19.1544% | 122,880 | 645.8 ns |
| EfDrain | 75,838,419 | 18.3046% | 122,880 | 617.2 ns |
| Register | 49,140,768 | 11.8608% | 122,880 | 399.9 ns |
| PrepareMap | 37,104,949 | 8.9558% | 122,880 | 302.0 ns |
| SubmitFinalize | 27,155,661 | 6.5544% | 122,880 | 221.0 ns |
| WinnerBuild | 6,403,823 | 1.5456% | 1,024 | 6,253.7 ns |
| AllocComplete | 1,422,201 | 0.3433% | 256 | 5,555.5 ns |
| Fanin | 1,170,265 | 0.2825% | 1,024 | 1,142.8 ns |
| **合计** | **414,313,448** | **100%** | — | — |

这些 span 的主要业务语义和实现入口是：

- **EfDrain**：`DrainReady()` 扫描本核私有 slot；对 fanin flag 做 ready
  判断，执行已就绪的前序 Kernel，随后发布 vend/flag、推进 frontier、写
  Commit 并释放 slot。源码为 `common/pa_scheduler_core.h:243-280`。
- **Materialize**：`MaterializeTask()` 扫描 tag，形成 output/register mask，
  计算输出布局与 heap ring 连续区间，构造 GM TensorDesc，最后推进
  `heap_next`。源码为 `common/pa_frontend.h:986-1051`。
- **PrepareMap**：`AdvanceTensorMap()` 按 `task_id-H` 推进存活下界，把过期
  producer 从 bucket 链摘除并放回 free list。源码为
  `common/pa_frontend.h:778-801`。
- **Claim**：先做 active-mask、role 与 lane 路由，再按 task id 选择四分片
  cursor 并执行 `atomicMax` 竞争，把返回结果写入 SubmitContext/统计。源码为
  `common/pa_scheduler_core.h:435-497`。
- **Fanin**：只在需要构建 slot 的路径上调用 `CollectFanin()`；扫描 owner 和
  TensorMap，选择最新重叠 producer 并去重。源码为
  `common/pa_frontend.h:873-927`。
- **Register**：`RegisterOutputs()` 依据 register mask，把已有 Inout/
  OutputExisting 作为当前 task 的新 hazard 版本插入本核 TensorMap。源码为
  `common/pa_frontend.h:929-951`。
- **WinnerBuild**：`WaitForSlot()`、`HeapGuard()`、私有 slot 分配，以及
  descriptor/scalar/fanin payload 复制；这里只发布待执行 slot，不执行当前
  Kernel。源码为 `common/pa_scheduler_core.h:499-538`。
- **AllocComplete**：Alloc 路径执行 `HeapGuard()` 后，以 vend、flag、
  frontier 的顺序发布完成。源码为 `common/pa_scheduler_core.h:640-653` 和
  `common/pa_scheduler_core.h:197-212`。
- **SubmitFinalize**：这是已有边界的离线语义名称，不是新设备 phase。它覆盖
  最后 child.end 到 Submit.end；分析器字段仍叫 `submit_tail_residual`，避免
  把尚未细拆的指令冒充成一个独立业务 API。

`SubmitFinalize` 可继续按最后一个业务动作解释，而不是只显示一个无意义的
residual：

| 最后动作 → SubmitEnd | 次数 | aggregate core-ns | 单次平均 | 占 SubmitFinalize |
| --- | ---: | ---: | ---: | ---: |
| Register → SubmitEnd | 97,280 | 21,318,053 | 219.1 ns | 78.50% |
| Claim → SubmitEnd | 24,320 | 5,442,753 | 223.8 ns | 20.04% |
| WinnerBuild → SubmitEnd | 1,024 | 316,153 | 308.7 ns | 1.16% |
| AllocComplete → SubmitEnd | 256 | 78,702 | 307.4 ns | 0.29% |

这里没有缺失的 standalone loser 计算。高频两类尾段主要是最后 record 发布、
公共统计、分支汇合和返回前时间戳；低频两类的单次尾段更长，但当前
边界还不足以把差值归因到某一条冷路指令。

#### 6.9.3 SubmitTransition 的五种业务含义

`between_submit_residual` 在数学上仍是相邻 Submit 的补集；在布局解释中应按
task 顺序显示为 `SubmitTransition.<前一类→后一类>`。这些名称完全可以由
现有 task id 和边界离线推导，不需要新增 raw 字段。

| 过渡 | 次数 | aggregate core-ns | 占全部过渡 | 单次平均 |
| --- | ---: | ---: | ---: | ---: |
| UP → 下一批 Alloc | 24,480 | 33,515,127 | 48.56% | 1,369.1 ns |
| QK → SF | 24,576 | 9,768,832 | 14.15% | 397.5 ns |
| Alloc → QK | 24,576 | 9,576,128 | 13.87% | 389.7 ns |
| SF → PV | 24,576 | 8,649,604 | 12.53% | 352.0 ns |
| PV → UP | 24,576 | 7,512,544 | 10.88% | 305.7 ns |

每段都包含前一 Submit record 发布、前一调用返回、本地统计和下一次
`BeginSubmit()`；其间真正的 orchestration 工作分别是：

- **Alloc → QK**：接收三个 Alloc 输出，计算 block group，创建动态 QK
  CreateInfo，reset TaskArgs，加入 query/key/table、QK 输出和两个 scalar；
- **QK → SF**：接收 QK score，创建 SF CreateInfo，reset，加入 score、三个
  output 和三个 scalar；
- **SF → PV**：接收 SF 的 probability/max/sum，reset，加入三个 input、一个
  output 和两个 scalar；
- **PV → UP**：接收 PV output，reset，加入三个 input、四个 Inout 和两个
  scalar；
- **UP → Alloc**：推进 batch 循环，读取下一 batch 的 context length，计算
  block 数，构造 query/output view，重新构造 TaskArgs 并初始化 tag/
  dump-selection，再加入三个 Alloc output。

对应源码为 `common/pa_scheduler_core.h:897-953` 和
`common/pa_frontend.h:477-705`。UP→Alloc 少 96 次，是因为最后一批 UP 后
不再进入下一批；其余四类均为 `256 × 96` 次。

#### 6.9.4 EfDrain、FinalDrain 与 Kernel

EfDrain 的排他拆分为：

| 子区域 | aggregate core-ns | 占 EfDrain | 说明 |
| --- | ---: | ---: | --- |
| KernelUnion | 32,039,768 | 42.2474% | 1,015 个观测到的 Kernel 调用完整区间并集 |
| DrainReady scalar 控制 | 43,798,651 | 57.7526% | slot 扫描、fanin、完成发布、frontier、清槽和观测 |
| **EfDrain** | **75,838,419** | **100%** | 精确闭合 |

全部 1,024 个 Kernel 的分布为：

| Kernel kind | 次数 | aggregate core-ns | 单次平均 | engine role |
| --- | ---: | ---: | ---: | --- |
| QK | 256 | 10,613,638 | 41.460 us | AIC |
| SF | 256 | 13,744,506 | 53.689 us | AIV |
| PV | 256 | 7,245,005 | 28.301 us | AIC |
| UP | 256 | 692,060 | 2.703 us | AIV |

其中 1,015 个 Kernel 在 Submit 期间由 EfDrain 执行，剩余 9 个在 FinalDrain
执行。EfDrain 所属的当前 Submit task kind 与其中 Kernel 的 task kind 不必相同，
所以不能按外层 Submit 名称把 Kernel 错归给当前任务。`Kernel` 边界包围的是
`ExecuteKernel()` 调用，包含 engine launch/completion wait wrapper；若 Kernel
begin 恰好关闭 PollBatch，还会吸收该 batch record 发布，因此不是纯计算单元
指令的净时间。

FinalDrain 精确拆分为：

| 子区域 | aggregate core-ns | 占 FinalDrain |
| --- | ---: | ---: |
| KernelUnion（9 个） | 255,441 | 1.0395% |
| FinalDrainControlWait | 24,317,532 | 98.9605% |
| **FinalDrain** | **24,572,973** | **100%** |

`FinalDrainControlWait` 是已有父区间减去 Kernel 并集后的源码语义名称，包含
replay_done 发布/轮询、fanin 检查、`SpinHint()`、完成发布、frontier、清槽和
观测成本。该区域内 Atomic 去重并集为 23,401,154 core-ns，占 FinalDrain
95.23%，主要来自 PollBatch 轮询窗口。这支持“最终阶段主要在同步与等待”，
但 PollBatch 覆盖完整轮询 episode，不能把 95.23% 解释成 isolated atomic
指令延迟。

#### 6.9.5 AIC/AIV 的完整区间分布

以下只保留每核平均这一种统计量，以便与 aggregate 整数总量对应：

| 区域 | AIC 每核平均 | AIV 每核平均 | 当前可直接读出的差异 |
| --- | ---: | ---: | --- |
| WorkerCompletion | 5.293429 ms | 5.292285 ms | 最终完成时间基本一致 |
| OrchestrationReplay | 4.990676 ms | 5.059708 ms | AIV replay 较长 |
| 首末 Submit 完整区间 | 4.988870 ms | 5.057685 ms | 与 replay 差异方向一致 |
| SubmitUnion | 4.305289 ms | 4.321003 ms | 两类核接近 |
| SubmitTransition | 683.581 us | 736.682 us | AIV 主要多在 Submit 间 |
| FinalDrain | 302.753 us | 232.576 us | AIC 较长，抵消 replay 差异 |

这组数据说明 AIV 的额外 replay 时间主要落在 SubmitTransition，而 AIC 在
FinalDrain 停留更久，最终使两类核的 WorkerCompletion 接近。它只说明负载
分布，不证明 AIV transition 或 AIC final drain 的某条指令就是根因。

#### 6.9.6 OrchestrationSetup、OrchestrationTail 与窗口外 DFX

- **OrchestrationSetup** 从 `orchestration_begin` 到首个 Submit.start，覆盖
  `InitPaOrchestration()`、首批 `BuildAllocArgs()`、相应统计，以及首个
  `BeginSubmit()`。它是可复算的初始化区间，不与 StartupBarrier 混合。
- **OrchestrationTail** 从最后一个 Submit.end 到 `orchestration_end`，覆盖
  最后一条 Submit record 发布、`SubmitTask()` 返回、循环退出和结束时间戳；
  swimlane 构建中的 `PmuWindowStop()` 是空实现。它不包含 FinalDrain。
- **StartupBarrier** 位于 `orchestration_begin` 之前；**DFXFinalize** 位于
  `final_drain_end` 之后。`OrchestrationReplay` 与 `FinalDrain` 两条父记录也
  在 `final_drain_end` 之后才写出，所以记录发布本身不落入
  `WorkerCompletion` 的两个父区间。当前 raw 没有这两段的完整父边界，
  因此只能确认它们位于已量化窗口两侧，不能计算完整耗时。

对应实现为 `common/pa_scheduler_core.h:880-1010`。这三个边界把初始化、业务
完成和诊断落盘分开，避免再用首末 Submit 近似完整 worker 生命周期。

### 6.10 分布合理性与当前瓶颈证据

本节只记录数据和源码共同支持的结论；尚未由微观数据证明的部分明确保留为
待验证问题，不据此直接修改代码。

#### 6.10.1 task kind 分布是否符合业务工作量

下表是排他 span 的单次平均 ns（即 SYS_CNT tick）。EfDrain 执行的是前序 ready task，按当前
Submit kind 聚合会产生错误归因，因此不放进本表：

| 阶段 | Alloc | QK | SF | PV | UP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Materialize | 1,566.6 | 830.9 | 1,580.2 | 983.2 | 602.2 |
| PrepareMap | 275.9 | 280.8 | 212.9 | 252.2 | 488.0 |
| Claim | 924.2 | 453.5 | 734.8 | 439.3 | 677.4 |
| Register | 65.8 | 239.0 | 397.4 | 300.4 | 997.1 |
| Fanin（条件执行） | — | 773.1 | 553.8 | 619.1 | 2,625.3 |
| WinnerBuild（条件执行） | — | 6,088.0 | 6,043.1 | 5,777.0 | 7,106.9 |

当前能由实现直接解释的趋势有：

- Alloc/SF 各物化三个新输出，QK/PV 各一个，UP 没有新 Output；
  Materialize 的高低与 descriptor 数量基本一致。该 span 同时包含前一条
  EfDrain record，因此还不能把全部差值归给 `MaterializeTask()`。
- Claim 的 atomic 参与核数按业务为 Alloc 96、QK/PV 32、SF/UP 64；因此
  Alloc 最高、SF/UP 居中、QK/PV 较低，与 `Claim()` 的 active-role 路由相符。
- 只有 UP 的 Register 真正插入四个 Inout，其他类型的 register mask 为空；
  UP 显著较高符合 TensorMap 插入工作量。
- 本轮 `H=64`；稳态 `AdvanceTensorMap()` 每次清理 `task_id-H-1`，即
  `task_id-65`。65 恰好是五任务周期的整数倍，所以 UP 阶段会清理较早 UP
  插入的四个条目；PrepareMap 的 UP 均值最高与这一实现相符。
- Fanin 的固定依赖数为 QK 0、SF/PV 各 1、UP 3；UP 还扫描最多的 tensor，
  其 Fanin 和 WinnerBuild 最高符合查找、去重和 payload 拷贝量。

这些相关性支持当前 span 路由暂未见明显错位，但“相关”仍不等于
已经量出某条 scalar 指令、I-cache miss 或 GM 访问的净成本。

#### 6.10.2 aggregate core-work 分布与调查优先级

下列排序反映 96 核累计的 scalar/core-work 观察区域，不是全局
5.278401 ms 的关键路径分解。总量较小的 winner-only 阶段仍可能串住
后继依赖；是否优先处理还需要关键路径和依赖到达证据。

1. **Materialize 是当前最大排他区域（Submit 的 33.00%）**。源码上它同时
   做 tag 扫描、输出布局、heap ring 处理和 128-byte TensorDesc 初始化；
   task-kind 趋势支持输出数量是重要变量。尚需用独立 `submit-pmu materialize`
   区分同名调用主体与前一条 record 发布、取指和 GM 访问。
2. **Claim 占 Submit 的 19.15%**。73,728 条 ClaimMax `return_ready` Atomic
   合计 25,878,336 core-ns，全部落在 Claim 内，相当于 Claim 观察区间总量的
   32.61%。这证明返回型 Atomic 是 Claim 的重要组成，但不能把该比例直接当成
   “删除 Atomic 可获得的收益”，因为竞争协议、记录扰动和其他 scalar 工作仍
   叠加在父区间中。
3. **EfDrain 占 Submit 的 18.30%**。现有边界已经把其中 42.25% 精确归为
   KernelUnion，57.75% 归为 DrainReady scalar 控制。下一步分析应只针对后者，
   再按无 slot、fanin 未就绪、完成发布三类路径取证，不能把 engine 计算当作
   scalar 调度瓶颈。
4. **SubmitTransition 占 WorkerCompletion 的 13.58%**。UP→Alloc 一项占
   全部 transition 的 48.56%，其源码确实包含换 batch、context GM load、
   view 与完整 TaskArgs 构造。泳道目前只能确定热点落在这段完整代码范围，
   尚未区分 Submit record、GM load、清对象和 add 参数各自成本。
5. **Register 占 Submit 的 11.86%**。UP 的真实四次 TensorMap 插入有明确
   工作量证据；QK/SF/PV/Alloc 的非零基础成本则混有前一 record 发布、空
   register 调用和统计，应结合 `submit-pmu register` 判断可优化比例。
6. **PrepareMap 占 Submit 的 8.96%**。UP 清理旧 UP 条目是已知重路径；其余
   时间需要按 map entry 数量和清理/无清理调用分层，当前没有 Atomic，可优先
   查 scalar load/store 与 I-cache，而不是猜测同步成本。
7. **SubmitFinalize 占 Submit 的 6.55%**。98.55% 的总量来自两类高频公共
   返回路径，主要是最后 record 发布和 epilogue。它不是遗漏的业务阶段；后续
   只应把它作为泳道观测成本/控制流布局门禁，不通过移动边界伪装成消减。
8. **FinalDrain 占 WorkerCompletion 的 4.84%**。其中 98.96% 是非 Kernel
   控制/等待，Atomic/PollBatch 去重并集又覆盖 95.23%。这已经足以把调查重点
   放到 replay_done/fanin 到达偏斜和轮询 episode，而不是最后 9 个 Kernel；
   但不能把 PollBatch 时间除以 atomic calls 当作单次 Atomic 延迟。
9. **Fanin、WinnerBuild、AllocComplete 总量较小**，分别只占 Submit 的
   0.28%、1.55%、0.34%。其单次成本不低、调用次数少；这些数字只能
   说明它们不是 aggregate core-work 主体，不能单凭总量决定是否优先
   改动协议。

#### 6.10.3 Atomic 的位置只作 Overlay 归因

97,449 条 Atomic record 全部可以找到业务位置，没有落在未解释的空白：

| 位置 | records | aggregate core-ns | 解释边界 |
| --- | ---: | ---: | --- |
| Claim | 73,728 | 25,878,336 | 直接 ClaimMax，`return_ready` |
| FinalDrain | 281 | 24,061,516 | direct 与 PollBatch 混合，不能直接相加解释 |
| EfDrain | 20,563 | 5,036,049 | fanin/frontier/completion 等 |
| AllocComplete | 1,790 | 441,083 | 完成发布与 frontier |
| WinnerBuild | 768 | 224,899 | `HeapGuard` 首圈 fast path 的 `FatalPoll`；本轮无 frontier/vend load |
| StartupBarrier | 319 | 3,608,890 | WorkerCompletion 之前的启动同步；父区间未完整量化 |

这张表只回答“Atomic 发生在哪个业务区域”。Atomic duration 已包含在父 span
中，禁止把表中数值再次加到 Submit/FinalDrain；PollBatch 还会与 direct 记录
重叠，表内 aggregate 也不等于位置的 Atomic 区间并集。

#### 6.10.4 当前数据尚不能支持的结论

- 不能把 `Materialize=33%` 解释成 `MaterializeTask()` 纯函数体占 33%；
- 不能把 Atomic aggregate 与父阶段相减，或把 PollBatch 当单条 Atomic 延迟；
- 不能由泳道直接量出 I-cache miss、GM stall、scalar busy 各自损失的时间；
- 不能把 span 边界移动后的重新归类称为性能提升；
- 不能用单轮 B256 排名直接决定代码改法；
- 不能把 standalone 的 FinalDrain、loser 或 Kernel 模拟直接等同于真实 PA。

standalone 与真实 PA 的当前差异必须单列，不能把本节绝对占比直接迁移：

- 真实 Submit 的 EfDrain 还先执行 `drain_block_won()`，kernel loser 也会执行
  一次该逻辑，FinalDrain 还检查 pending won；standalone 不实现 joint/
  BlockWon，loser 没有这部分业务动作。
- standalone `TraceRecord` 为 64 B，并让相邻 child 复用前一 end；真实
  `FdwicSwimlaneRecord` 为 32 B，Materialize/PrepareMap/Claim/Fanin/Register
  等显式 child 当前用独立 `TRACE_SPAN_BEGIN/END` 取时，EfDrain 仍是 lap。
  因此 phase record 和间隙成本在两条路径中的归属不同，不能照搬 standalone
  的相邻 span 数值解释真实 PA residual。
- standalone 复用一个 `TaskArgs` 并直接调用 `SubmitTask()`；真实 orchestration
  还有各 scope 的 L0TaskArgs、`TaskOutputTensors::get_ref()`、`rt_submit_*`
  wrapper、MixedKernels 和 fatal 路径，SubmitTransition 的绝对成本不完全对等。
- standalone `real-compute` 只提供固定 128×128 synthetic engine 工作量，
  不是实际 QK/SF/PV/UP kernel；当前真实路径也还没有 standalone 这两条
  `OrchestrationReplay/FinalDrain` raw 父记录。

Case1 当前 task 均为单 lane，依赖拓扑仍可用于调度观察，但以上 scalar 控制、
DFX 和计算体差异必须在迁移到真实 PA 时重新取证。真实入口见
`src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/submit_runtime.h:254-380`
及 `examples/a5/fully_distributed_within_core/paged_attention_unroll/kernels/orchestration/paged_attention_orch.cpp:156-278`。

#### 6.10.5 复用已有 submit-pmu 的微观证据

已有五份 B256 `submit-pmu` 结果可以直接复用，不需要为本轮布局审计重新上板：

```text
outputs/submit_pmu_b256_for_swimlane_20260719_114815_617346/
    none/
    claim/
    efdrain/
    materialize/
    register/
```

这组数据与 11:48 的 schema-v4 泳道基线配套，也早于 12:35 的
完整性样本。两者 span 语义相同，但 ELF 布局不同，因此本节只复用 PMU
对阶段性质的取证，不把它与 12:35 泳道 tick 逐项相减，也不把旧
PMU 数值冒充成当前源码的精确性能。调度落点也已变化：12:35
泳道的 Kernel 在
EfDrain/RingBp/FinalDrain 分别为 `1015/0/9`，而 none/claim/efdrain/
materialize/register 五个旧 PMU ELF 分别为 `817/188/19`、`935/68/21`、
`870/140/14`、`897/105/22`、`917/76/31`。因此两组比例只能定性
并列，不能当作同一执行分布。

`none` 变体只开一次完整 Submit PMU gate，不插入局部 read-clear。其每核平均为：

| 角色 | PMU total（校准） | scalar busy（校准） | scalar/total | I-cache request | I-cache miss | miss/request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AIC | 5,249.211 us | 4,302.579 us | 81.9662% | 408,521.2 | 39,264.8 | 9.6114% |
| AIV | 5,080.418 us | 4,198.206 us | 82.6350% | 445,515.7 | 54,738.3 | 12.2865% |

这说明该诊断 ELF 的完整 Submit 中约 82% PMU cycle 被 `scalar_instr_busy`
覆盖；其余约 18% 不能直接命名为 scalar 空闲，仍混有 engine 等待、取指/
访存 stall 和其他未被 CNT2 覆盖的周期。AIV 每核比 AIC 多约 37.0k request、
15.5k miss，且 miss rate 高 2.675 个百分点，I-cache 的角色差异确实值得继续
定位。

按已有 cold/warm 标尺 90 ns 做最简单的串行等效换算，AIC/AIV 分别约为
3.534/4.926 ms 每核。它只回答“若所有 miss 延迟完全串行会有多大”，已经接近
甚至覆盖整个 Submit，因此不能当成实际损失时间；miss 延迟会与流水、其他请求、
Atomic 等待和 engine 工作重叠。要得到可优化的净损失，仍需同一源码的布局 A/B
或更直接的 stall 证据。

四个单阶段 ELF 的局部结果如下。时间占比以各自 ELF 的首末 Submit 完整区间为
分母；request/miss 占比以各自 ELF 的完整 Submit primary 为分母。上下界差异
很小，表中保留到能表达结论的精度：

| 局部阶段 | SYS_CNT/调用 | 阶段时间占比 | request 占比 | miss 占比 | 局部 observed miss/request |
| --- | ---: | ---: | ---: | ---: | ---: |
| Claim | 349.8 ns | 11.88% | 6.674% | 7.990%～7.995% | 12.172% |
| EfDrain | 941.7 ns | 26.03% | 8.265%～8.273% | 11.391% | 14.006% |
| Materialize | 859.6 ns | 23.82% | 38.771% | 36.857% | 9.773% |
| Register | 136.9 ns | 4.20% | 15.700%～15.702% | 9.121%～9.124% | 4.756% |

每个局部调用仍新增 begin/end 两次 SYS_CNT 和 running read-clear；阶段时间不含
两侧 `ld_dev`，但包含时间戳边界及紧邻 context/stat 归档的观察扰动。它比普通
泳道更接近同名调用主体，但不是无扰动函数净耗时。表中局部
`miss/request` 是 shadow 直接观测的下界比率，不是加上 residual 后的上界。

这些独立 PMU 结果把 6.10.2 的判断进一步收紧为：

- **Materialize 是已测阶段中最明显的取指热点**：时间占 23.82%，却覆盖
  38.77% request 和 36.86% miss；AIC/AIV 的局部 miss 分别为
  11,550.0/19,233.8 次每核，占各自完整 Submit miss 的 67.45%/32.44%。
- **EfDrain 的主要问题不只是取指**：时间占 26.03%，request/miss 只占
  8.27%/11.39%；旧 PMU ELF 与当前泳道的 Kernel 落点不同，因此只能
  定性地与当前 42.25% EfDrain KernelUnion 并列。这些证据更支持 engine
  工作、一次 fanin 依赖检查/Atomic load 和完成控制共同构成，而不是
  单纯 I-cache 主导；standalone 这里不会对 fanin 轮询等待。
- **Claim 的时间占比高于取指活动占比**：时间占 11.88%，request 只占
  6.67%；这与 ClaimMax 返回等待占父阶段相当比例的 Atomic 证据一致。
- **Register 的取指请求占比不低，但 miss 密度较低**：request 占 15.70%，miss 只占
  9.12%，局部 miss/request 为 4.76%。AIC 每核局部仅约 49.6 miss，而 AIV
  约 4,847.3；但两类核都会完整回放 UP 并插入同样四个 TensorMap 条目，不能把
  角色差值归因成插入次数不同。现有数据还不能区分角色 ELF 布局、前后控制流
  或其他硬件行为，需保持同源码后继续取证。

四个 phase ELF 的完整 Submit 时间本身并不相同，说明编译期局部观测会改变
代码布局；这些局部值不能跨 ELF 相加，也不能与 `none` 相减得到“净阶段成本”。
但上述时间/request/miss 的同 ELF 内比例、角色方向和与源码工作量的一致性，
足以用于确定下一轮微观调查顺序。

#### 6.10.6 当前源码重编后的两轮 B256 与末端路径

为避免用 12:35 的旧 ELF 替当前源码下性能结论，本机使用用户
`/home/q00473782/.venv`、当前 CANN 9.1 和当前工作树重编 CCEC，然后用
同一份 mixed ELF 连续采集两轮：

本节的“当前”特指 **standalone CCEC、level-4 atomic 合并泳道**。它不等于
真实 PA Case1 的 level-1 正式基线；后者三轮中位数为 4.821897 ms。当前两轮
应与同口径 standalone 历史值 5.278401 ms 比较，不能据此宣称真实 PA 从
4.8 ms 回退到 5.2 ms。

```text
outputs/pa_scheduler_swimlane_20260719_162653_1031549/ccec/
outputs/pa_scheduler_swimlane_20260719_162930_1038954/ccec/
```

两轮均通过全部语义断言、父子关系与整数闭合，`dropped_records=0`：

| 指标 | 当前轮 1 | 当前轮 2 |
| --- | ---: | ---: |
| raw records | 839,933 | 839,093 |
| 跨核首末 Submit 完整区间 | 5.313009 ms | 5.245894 ms |
| SubmitUnion aggregate core-work | 414.827631 ms | 412.993668 ms |
| SubmitTransition aggregate core-work | 69.600295 ms | 69.193462 ms |
| FinalDrain aggregate core-work | 27.072400 ms | 22.717534 ms |
| Kernel 落点 EfDrain/RingBp/FinalDrain | 1016/0/8 | 1015/0/9 |
| 最后 Submit 所在核 | core89/AIV | core65/AIV |
| 最长 SF Kernel 完整区间 | 255.201 us | 207.998 us |

两轮的非 Atomic 记录数都固定为 742,016；raw 总数的 840 条差异全部来自
运行时轮询次数变化带来的 Atomic 记录数差异。这进一步说明 span 布局和固定
工作量事件数稳定，不能把 raw 总数的小幅波动误解成边界遗漏或重复记录。

高频 scalar 区域的 aggregate core-work 在两轮间很稳定：Materialize、
PrepareMap、Claim、Register、SubmitFinalize 和 Fanin 的差值分别为
`-0.007%`、`+0.149%`、`-0.246%`、`-0.836%`、`-0.139%` 和
`-0.301%`。EfDrain 相差 1.565%，但它包含下述不稳定 Kernel 长尾；
FinalDrain 相差 16.09%，反映的主要是跨核到达偏斜。因此当前能证明的
是“高频前端工作量总体稳定”，不是“全局末端也由这些 aggregate
排名决定”。

按两轮 aggregate core-work 的平均占比，当前大致布局为：

| WorkerCompletion 分区 | 两轮平均占比 | 主要语义 |
| --- | ---: | --- |
| Submit 内部并集 | 81.42% | 每次 Submit 内已有排他子阶段之和 |
| SubmitTransition | 13.65% | 前一 Submit 发布/返回与下一任务构参 |
| FinalDrain | 4.89% | replay_done/fanin 同步、剩余 slot 清空；不在首末 Submit 墙钟内 |
| Setup + Tail | 0.04% | 首批构参与最后一次返回 |

Submit 内部再按两轮平均分布为：

| Submit 内 span | 两轮平均占比 | 当前两轮典型单次 | 分布解释 |
| --- | ---: | ---: | --- |
| Materialize | 32.95% | 约 1.0 us | 随输出 descriptor 数量分层 |
| Claim | 19.13% | 约 0.72 us | attempted 与未参与核分层；包含 ClaimMax 返回等待 |
| EfDrain | 18.33% | empty fast path 约 0.18 us | 稀疏的 ready/Kernel 路径抬高总量，不能只看中位数 |
| Register | 11.76% | 约 0.30 us | UP 的四次 TensorMap 插入形成重路径 |
| PrepareMap | 9.05% | 约 0.26 us | UP 清理旧条目形成重路径 |
| SubmitFinalize | 6.59% | 约 0.17 us | 最后一条 record、公共收尾与 SubmitEnd 取时 |
| WinnerBuild | 1.54% | 约 5.9 us | 每 task 仅一个非 Alloc winner |
| AllocComplete | 0.36% | 约 4.8～5.2 us | 每 batch 仅一个 Alloc winner |
| Fanin | 0.28% | 约 0.71 us | 条件执行；UP 因三路依赖约 2.67 us |

这里的“典型单次”只用于描述常见路径，不参与 aggregate 闭合。特别是 EfDrain
把大量 empty fast path 与少量 Kernel/完成发布混在同一分布中；两轮中其
KernelUnion 平均占 EfDrain 42.40%，其余 scalar 控制占 57.60%。五类
SubmitTransition 中，UP→Alloc 的典型值约 1.38 us，其他四类约
0.28～0.37 us，符合换 batch 和重建 Alloc 参数的额外业务量。

Kernel 自身按 task kind 的两轮合并中位数为 QK 41.095 us、SF 53.269 us、
PV 27.638 us、UP 2.586 us；这说明默认 synthetic engine 工作量整体稳定。
当前墙钟末端异常来自 SF 的 207.998/255.201 us 完整区间长尾，而不是 SF
常态负载整体增大。

跨核首末 Submit 完整区间也不是一条已证明因果关系的 96 核关键路径。
它在两轮中分别闭合为：

```text
5.313009 ms = core89 的首次 Submit 起点偏移 0.009434 ms
            + core89 自身 Submit 完整区间 5.303575 ms

5.245894 ms = core65 的首次 Submit 起点偏移 0.000365 ms
            + core65 自身 Submit 完整区间 5.245529 ms
```

两轮末端都被同一个业务 task 的长尾串住，但物理核和具体
EfDrain 落点不同：

- 轮 1 的 core89 构建 task1277/SF，其 255.201 us Kernel 在 task1279
  的 EfDrain 执行；对应 EfDrain/Submit 为 262.196/270.986 us；
- 轮 2 的 core65 构建 task1277/SF，其 207.998 us Kernel 在 task1278
  的 EfDrain 执行；对应 EfDrain/Submit 为 221.231/223.673 us。

这证明当前两轮的最后 Submit 主要被 `ExecuteKernel()` 完整区间拖长，
不是该 Submit 内的 Claim Atomic。Kernel 区间包含 engine launch/completion-wait
wrapper，不能进一步说成纯 TADD 指令执行净耗时。12:35 样本中的
core89/task1277～1279 多个普通 scalar span 同时变长没有在当前两轮
原样复现，因此不把旧单轮现象归因为 Atomic、I-cache 或 GM。

两轮仍显示明确的两端效应：在 Materialize/PrepareMap/Claim/Register
这四个通常为亚微秒到约 1～2 us 的前端 span 中，大于 3 us 的
样本几乎只出现在前 20 个 task 或最后 10 个 task；两轮只各有 1 条
AIC 稳态例外。这支持将冷启动和末端干扰与稳态单次分布分开，
但 3 us 只是用于查询尾部样本的阈值，不是性能规格。

FinalDrain 在两轮中都是同步补偿区域：其 start 跨核偏斜为
312.070/276.616 us，start 与 duration 的相关系数为
`-0.9892/-0.9925`，而 end 只分散 33.208/25.218 us。它不在用户关注的
首末 Submit 窗口内；较大的 aggregate 主要表示早到核等待晚到核，不能
当成 96 个可独立消除的本地算法瓶颈。

### 6.11 后续取证顺序：先确认瓶颈，再讨论优化

下一阶段不先写优化补丁，也不继续扩张设备记录。先按以下顺序把现有 span
能够回答和不能回答的问题分开；每一步仍以当前完整布局和整数闭合作为
正确性门禁：

1. **固定权威基线**：后续结构迭代用 b1，阶段性结论才跑 b256；保留
   `dropped=0`、事件数、父子关系、Kernel 唯一归属和六类整数闭合。
2. **分开总体工作量与墙钟末端**：Materialize、Claim、EfDrain、
   SubmitTransition、Register、PrepareMap 的 aggregate core-work 用于描述 96 核
   总体工作量；最后 Submit 所在核及其串行 span 才用于解释本轮墙钟终点。
   两者排名不能相互替代。
3. **先解释现有 Kernel 长尾**：当前两轮都出现 task1277/SF 的
   `ExecuteKernel()` 完整区间长尾，但物理核和承接它的 EfDrain 会漂移。先沿
   现有 synthetic engine、完成 flag 和 `DrainReady()` 调用链核对
   `ExecuteKernel()` 区间内的 engine launch/completion wait，并单独核对区间
   结束后仍属于 EfDrain/FinalDrain control 的完成发布与 Commit；在没有进一步
   证据前，不把它命名为纯 TADD、scalar、Atomic 或 I-cache 瓶颈。
4. **微观计数按问题复用**：只有需要区分 scalar busy 与 I-cache 时，才复用
   独立 `submit-pmu none`；Materialize、Claim、EfDrain、Register 的局部 PMU
   只作为已有定性证据。不同 ELF 的阶段值不相减，也不为继续细分而默认新增
   设备字段或扩大 raw。
5. **形成单变量候选清单**：每个候选必须明确预期影响哪个现有 span，并
   检查成本是否转移到相邻 span，同时检查 `.text`、I-cache PMU、泳道 core-work 和
   关闭诊断后的实际 Submit 时间；完成这些取证后才开始性能修改。

还需特别区分 `submit-pmu` 的两个时间口径：primary PMU counter 覆盖
`PmuWindowStart()` 到 `PmuWindowStop()`，而当前阶段时间占比的分母是每核首个
Submit begin 到末个 Submit end。前者比后者多 Setup/最后返回等小段；报告必须
明确分母，不能把二者静默当成同一完整区间。

### 6.12 compete-first eager 移植后的当前权威布局

2026-07-20 已将受控 B 版的稳定收益形状移植回 standalone 主路，
但没有移植 C 版 lazy 跳过构参。当前 96 个 worker 仍全部同步构造
完整 `TaskArgs`；改变的是 Claim 与构参的先后关系，以及 CCEC
caller/runtime-state/noinline-finish 的编译布局。当前业务时间线为：

```text
OrchestrationSetup
  InitPaOrchestration / BeginPaBatchForCallback / context read / BeginCallbackSubmit

Submit
  EfDrain
  Claim
  Claim.end -> Materialize.begin residual
    Claim record 发布 + 同步 eager callback 构造完整 TaskArgs + 调用衔接
  Materialize
  PrepareMap
  winner non-Alloc: Fanin
  Register
  winner: WinnerBuild 或 AllocComplete
  loser/winner 公共收尾 -> Submit.end

SubmitTransition
  前一 Submit record 发布/返回 + AcceptTaskOutputs + block-group/batch 准备
  + 下一次 BeginCallbackSubmit
```

Materialize 之后的相邻 child 继续复用已有 end 边界，只有
`Claim -> Materialize` 因真实 callback 构参而保留非零内部 residual。
这个 residual 不新增 raw 字段、记录或 `SYS_CNT` 读取，由 converter/
analyzer 使用现有 Claim.end 和 Materialize.begin 离线复算。旧 6.4～6.11
中“构参全部属于 SubmitTransition”、Alloc `Register -> Claim` 和
“Submit 内无 child-to-child gap”都只属于移植前历史样本。

当前 CCEC A5 b1 权威产物为：

```text
outputs/pa_scheduler_swimlane_20260720_092021_1729726/ccec/
  l2_swimlane_records.json
  merged_swimlane.json
  swimlane_exclusive_analysis.json
```

该轮使用 `real-compute/6,28,4,1`，共 480 个 Submit、4,146 条 raw
记录，`dropped_records=0`；atomic 闭合为
`865 = 1475 - 862 + 252`，全部语义、角色、split state、Kernel 唯一归属与
整数分区门禁通过。Submit aggregate core-work 为 2,345,639 SYS_CNT ticks：

| 区域 | aggregate ticks | SubmitUnion 占比 |
| --- | ---: | ---: |
| Claim | 778,137 | 33.1738% |
| Materialize | 446,670 | 19.0426% |
| Claim→Materialize 构参 residual | 354,842 | 15.1277% |
| Register | 228,377 | 9.7362% |
| EfDrain | 222,825 | 9.4995% |
| PrepareMap | 145,613 | 6.2078% |
| Submit tail residual | 115,514 | 4.9246% |
| WinnerBuild / AllocComplete / Fanin | 53,661 | 2.2877% |

上表是带泳道/atomic 观察的 b1 aggregate core-work，用于验证新边界
分类，不用来宣称关闭观察后的局部净耗时。关闭泳道的 b256
独立进程样本为 `3706.483 / 3778.698 / 3735.032 / 3741.987 /
3595.101 us`，中位数 3,735.032 us。移植前同口径五轮中位数为
3,889.180 us，本轮中位数减少 154.148 us，约 3.9635%；候选五轮
全部快于旧基线的最快一轮 3,825.697 us。该结果证明收益已在
standalone 主路复现；它不直接承诺 simpler 真实 PA 也有相同比例。

## 结论

历史 schema-v3 泳道不闭合的根因不是缺少一次 duration 求和，而是
Submit 父区间、Kernel 嵌套、Atomic/Clock/lap Overlay 和缺失的 orchestration/final
drain 外层边界被混在同一加和口径中。

当前 schema-v4 的父子关系、事件数量、Kernel 归属和整数闭合已在
compete-first eager 移植后的 CCEC A5 b1 重新验证。Submit 内部当前只有
`Claim.end -> Materialize.begin` 保留有意义的 child-to-child residual，
它对应 Claim record 发布、同步 eager callback 构参和调用衔接。
`SubmitFinalize` 仍是最后 record 与公共收尾；`SubmitTransition` 改为记录发布、
返回、输出接收和下一次 Submit 上下文初始化，不再包含完整参数构造。
这些 residual 仍保留数学补集属性，不能仅靠改名伪装成性能消减。

当前 b1 带观察的 aggregate 分布中，Claim、Materialize、同步构参
residual、Register 和 EfDrain 是主要区域；泳道仍只能给出完整观察窗口，
不能独立分出纯业务、Atomic、I-cache、GM stall 与 DFX 记录成本。
关闭泳道的五轮 b256 已证明 standalone 主路收益可复现；下一步是按
同一 compete-first eager 语义移植到 simpler 真实 PA，并在真实路径重新完成
正确性、泳道闭合和性能 A/B，不直接套用 standalone 收益比例。

[pa-orch]: ../../st/a5/tensormap_and_ringbuffer/paged_attention_unroll/kernels/orchestration/paged_attention_orch.cpp
