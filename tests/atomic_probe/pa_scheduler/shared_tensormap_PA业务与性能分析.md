# Shared TensorMap PA 业务流程与最佳泳道性能分析

## 1. 文档目的

本文把当前 standalone shared TensorMap 版 PA 调度器的业务合同、实际执行路径和 A5 性能证据放在同一份文档中，供后续独立分析优化方案时使用。

本文只回答以下问题：

1. 96 个 AIC/AIV worker 如何回放 PA task、竞争 owner、发布 shared TensorMap 元数据并执行 kernel；
2. winner、true loser、not-attempted、Register 串行链、Fanin、Build 和 Drain 的真实关系；
3. 当前最佳完整泳道中，时间具体消耗在哪里；
4. 哪些是业务计算，哪些是 atomic/等待，哪些只是观察代码或统计口径；
5. 后续方案必须保持哪些正确性约束，以及最值得继续推演的问题是什么。

本文不是优化收益报告，也不把历史试验中的最好数字拼成一条不存在的执行路径。除明确标为“无泳道 perf-clock 对照”的数据外，所有性能结论都来自同一个 best 泳道样本。

## 2. 唯一分析样本与可信边界

### 2.1 样本

唯一性能分析对象：

- [shared_b256_best_1659167us_swimlane.json](test_record/2026-8-1/shared_b256_best_1659167us_swimlane.json)
- 对应原始运行目录：`outputs/pa_scheduler_shared_swimlane_20260801_092241_2274323/ccec/`
- TensorMap 模式：`shared`
- batch 数：256
- worker：32 个 AIC、64 个 AIV，共 96 个
- task：每 batch 5 个，共 1,280 个 task id
- 每个 worker 都按同一顺序回放 1,280 次 Submit，共 122,880 次 Submit
- final barrier：`two-16`
- winner workload：`real-compute`
- QK/SF/PV/UP 的模拟次数：`6/28/4/1`
- 模拟单元：完整的 `128 × 128` engine pipeline iteration
- QK、PV 使用 Cube；SF、UP 使用 Vector；Alloc 不发射 engine kernel

这里的 `real-compute` 表示 CCEC 确实发射 Cube/Vector 指令并等待真实 engine pipeline，不是 Scalar NOP；但它仍是 standalone 校准负载，不是生产 PA 数值 kernel。四类 workload 使用预先初始化的固定 128 × 128 workspace，主要用于模拟执行时长和 AIC/AIV pipeline 行为，并不真正消费本次 Materialize 得到的全部 PA buffer。调度依赖、descriptor 发布和 fanin 协议是真实执行的，kernel 数据流则是受控近似。不能从本样本推导生产 PA kernel 的访存/cache 性能。

原始泳道包含 771,161 个可视事件。设备侧共写入 397,304 条普通记录、134,840 条 atomic 记录、1,550 条聚合 poll 记录和 4,288 条 DCCI 记录，`dropped_records=0`。独占区间分析器对以下关系均验证为 PASS：

- 每核 task id 连续且数量相等；
- 每核 Submit 不重叠；
- Submit、EfDrain、Materialize、Register 和 worker lifecycle 的父子区间精确闭合；
- winner、true loser、not-attempted 数量闭合；
- 每个 kernel 只归属一个实际 Drain；
- Register 和 Materialize 的明细区间与父区间身份、边界和分区一致。

因此，这份数据适合回答“当前一次执行中时间落在哪里”。但它仍然只有一个 best 样本，不能单独回答波动分布或候选优化的稳定收益。

### 2.2 时钟口径

泳道时间来自 A5 `SYS_CNT`：本数据元信息记录 `clock_freq_hz=1,000,000,000`，即一个 tick 为 1 ns。这里的 1 GHz 是系统计数器口径，不是 Scalar PMU 主频。

此前校准得到的约 1.65 GHz 是 PMU cycle 与 SYS_CNT 的换算关系。本文没有把泳道 tick 当作 1.65 GHz cycle，也没有用 PMU cycle 反算这些泳道区间。

### 2.3 四种不能混用的时间

| 名称 | 含义 | 能否直接相加/比较 |
| --- | --- | --- |
| Submit 墙钟 | 全局最早 Submit 起点到最晚 Submit 终点 | 用于描述本次完整 Submit makespan |
| 每核 Submit envelope | 单核第一个 Submit 起点到最后一个 Submit 终点 | 包含该核 Submit 间空白 |
| 核累计时间（core-time） | 96 个核上某区间时长之和 | 只用于工作量归因，不能当墙钟 |
| 单事件耗时 | 某次 Claim、Register、Kernel 等的时长 | 可做分布；不能把并发事件之和当端到端时间 |

泳道中的 `kernel` 轨道是 Scalar 事件的显示别名，用来把 engine 工作放到稳定的 kernel lane；它不是第二次执行，也不能与 Scalar 父区间重复相加。

## 3. 当前业务模型

### 3.1 PA task 图

每个 batch 的固定 task 顺序和依赖如下：

```text
Alloc ───────────────────────────────┐
                                     │
QK ──> SF ──> PV ───────────────────┼──> UP
       └─────────────────────────────┘
```

具体语义：

| Task | 核型 | 新输出数 | 直接 task producer | 主要工作 |
| --- | ---: | ---: | --- | --- |
| Alloc | AIC/AIV 都可 Claim | 3 | 0 | 为三个 accumulator 物化 descriptor，随后立即完成 |
| QK | AIC | 1 | 0 | query/key/table 参数，执行 Cube matmul |
| SF | AIV | 3 | 1：QK | 读取 score，生成 probability/max/sum，执行 Vector add |
| PV | AIC | 1 | 1：SF | 读取 probability/value，执行 Cube matmul |
| UP | AIV | 0 | 3：Alloc、SF、PV | 读取 max/sum/PV output，并更新三个 Alloc INOUT，执行 Vector mul |

UP 虽然包含多个 SharedOutputRef 输入和三个 INOUT，但按 producer task id 去重后只有 3 条 fanin。每 batch 的唯一 fanin 边数为 `0 + 1 + 1 + 3 = 5`，B256 共 1,280 条。

参数构造可从 [pa_scheduler_core.h](same_core/common/pa_scheduler_core.h#L3494) 的 `BuildCallbackSubmitArgs()` 对照。真实模拟负载尺寸和默认次数定义在 [winner_workload.h](same_core/common/winner_workload.h#L22)。

### 3.2 96 核回放，不是一个共享执行队列

32 个 AIC 和 64 个 AIV 都回放同一份 1,280-task 计划。当前 shared 模式的核心关系是：

```text
所有 worker 回放同一 task N
          │
          ├─ 不具备该 task 核型资格：not-attempted
          │
          └─ 具备资格：参加 Claim Tournament
                    │
                    ├─ true loser：立即结束本次 Submit
                    │
                    └─ 唯一 winner
                         ├─ 本核构参、Materialize、Register、Fanin
                         ├─ 本核把完整执行包写入自己的 private LocalSlot
                         └─ 后续仍由本物理核的 Drain 执行该 slot
```

也就是说：

- QK/PV 若由某个 AIC 构建，最终仍由这个 AIC 执行；
- SF/UP 若由某个 AIV 构建，最终仍由这个 AIV 执行；
- 当前没有跨核共享 ready queue，也没有 task stealing；
- “Submit 与执行解耦”只表示二者在时间上分离，不表示构建核与执行核分离。

private slot 及其 fanin 快照定义在 [pa_model.h](same_core/common/pa_model.h#L1668)，winner 写入本核 slot 的入口见 [pa_scheduler_core.h](same_core/common/pa_scheduler_core.h#L1078)。

### 3.3 Claim 候选人口与两级 Tournament

当前没有通过缩小候选人口换性能：

| Task | 合法候选 | 每 task 的 local group 数 | 每组最大同地址竞争宽度 | root 最大竞争宽度 |
| --- | ---: | ---: | ---: | ---: |
| Alloc | 96 | 8 | 12 | 8 |
| QK/PV | 32 AIC | 6 | 6（部分组为 5） | 6 |
| SF/UP | 64 AIV | 8 | 8 | 8 |

每个候选只执行一次 local CAS。local winner 才继续执行一次 root CAS，root 唯一 winner 成为 task owner。实现见 [pa_scheduler_core.h](same_core/common/pa_scheduler_core.h#L903)，人口和 group 常量见 [pa_model.h](same_core/common/pa_model.h#L209)。

B256 的精确数量是：

| 分类 | 计算 | 次数 |
| --- | --- | ---: |
| Submit | `256 × 5 × 96` | 122,880 |
| attempted Claim | `256 × (96 + 32 + 64 + 32 + 64)` | 73,728 |
| winner | 每 task 恰好一个 | 1,280 |
| true loser | `73,728 - 1,280` | 72,448 |
| not-attempted | `122,880 - 73,728` | 49,152 |
| local CAS | 每 attempted 一次 | 73,728 |
| root CAS | `256 × (8 + 6 + 8 + 6 + 8)` | 9,216 |

Claim 只决定“谁负责 task N”，不承担 shared TensorMap 的 task-id 插入顺序。严格顺序由后面的 `deps_prepared` commit chain 单独保证。这两个协议不能合并理解。

### 3.4 Shared TensorMap 的实际快速路径

当前 PA 的普通新输出并不进入按地址区间查询的 ordinary TensorMap ring，而是使用 task-indexed 表：

```text
shared_outputs[producer_task_id].tensors[output_slot]
```

descriptor、published 原子和 last-writer 原子分开占用 cache line，结构见 [pa_model.h](same_core/common/pa_model.h#L1485)。因此本样本中 1,280 个 Register 的 `ordinary_tensormap_entries` 全部为 0，不表示“没有发布输出”：

- Alloc/QK/SF/PV 的 2,048 个新输出 descriptor 在 Materialize 发布到 `shared_outputs`；
- 每个 task 都必须推进 `deps_prepared`；
- UP 的三个 INOUT symbol 通过 writer history 和 group latest 维护前任 writer；
- Fanin 仍读取 published/last-writer/history，Build 仍把 descriptor 复制进本核 slot。

这一点非常重要：本样本能代表 PA SharedOutputRef 快速路径和严格插入链，但不能代表 ordinary region TensorMap payload 很多的任意算子。任何后续通用优化都不能把 `ordinary_count == 0` 当成系统恒真条件。

## 4. 一次 Submit 的精确流程

```text
Submit(task N)
│
├─ 1. EfDrain
│    扫描本核已有 private slots；fanin ready 时执行旧 task kernel 并发布完成
│
├─ 2. Claim
│    not-attempted ────────────────> Submit 结束
│    local/root CAS loser ─────────> Submit 结束
│    root winner
│
├─ 3. ArgBuild / winner bridge
│    winner 才同步执行 callback 参数 thunk，构造 TaskArgs/SubmitContext
│
├─ 4. Materialize（可并行）
│    ├─ 从 shared heap 分配实际 output buffer
│    ├─ 构造 TensorDesc 和 writer delta
│    ├─ 发布 task-indexed SharedOutput descriptor
│    └─ UP 提前准备不可变 writer-history payload
│
├─ 5. Register（唯一全局 task-id 串行区）
│    ├─ 等待 task[N-1].deps_prepared == N-1
│    ├─ 发布 N 的 ordinary/symbol writer metadata
│    └─ CAS 发布 task[N].deps_prepared = N
│
├─ 6. Fanin（已经离开串行区）
│    查询 writer，过滤 producer >= N，保存唯一 producer task ids
│
├─ 7. Build / Complete
│    Alloc：直接 CompleteTask
│    其他：构造本核 private LocalSlot，保存 tensors/scalars/fanin
│
└─ Submit 结束

后续某次 Submit 的 EfDrain，或所有 replay 完成后的 FinalDrain
└─ 检查 slot fanin flag
   ├─ 未 ready：slot 保留
   └─ ready：本核执行 engine workload -> vend -> flag -> 释放 slot
```

### 4.1 EfDrain

每次 Submit 入口先尝试推进本核旧 slot，入口见 [pa_scheduler_core.h](same_core/common/pa_scheduler_core.h#L617) 和 [pa_scheduler_core.h](same_core/common/pa_scheduler_core.h#L685)。当前 opportunistic polling 会在连续无进展时降低探测频率，但 WaitForSlot 和 FinalDrain 仍强制推进。

因此泳道中的一条 loser Submit 可能很长：长时间不一定属于 loser Claim，而可能是它在 Claim 之前替旧 winner 执行了 kernel。分析 loser 必须从 Submit 中扣除 `efdrain.kernel`，不能直接把整条 loser Submit 当 loser 控制开销。

### 4.2 Claim 后才构参

shared 模式下 loser 不执行完整 callback 参数构造。winner 在 Claim 后执行参数 thunk，随后进入 `FinishSharedWinnerSubmitBody()`。业务入口可对照 [pa_shared_submit_path.h](same_core/common/pa_shared_submit_path.h#L583)。

泳道没有为这段新增一个大记录字段；它主要落在 `Claim -> Materialize` residual 中。这样控制了记录体积，但 residual 不能被误解成一种业务操作。

### 4.3 Materialize 不占用严格插入链

winner 在 Materialize 中完成 descriptor、最小 writer delta、SharedOutput 发布和 UP history payload 准备，代码见 [pa_shared_submit_path.h](same_core/common/pa_shared_submit_path.h#L620)。这些工作可以由不同 task owner 并发执行。

只有 writer history 的 payload 在这里写出；它的 last-writer CAS 要等取得 Register 顺序后才发布，因此 reader 不会提前看到半成品 history。

### 4.4 Register 只串行元数据 commit

当前唯一全局串行区是：

```text
wait deps_prepared[N-1]
-> publish metadata(N)
-> CAS deps_prepared[N] = N
```

代码边界见 [pa_shared_submit_path.h](same_core/common/pa_shared_submit_path.h#L729)。空 writer 集合也必须推进 task N 的完成字，否则 N+1 无法安全前进。

Fanin lookup、Build 和 kernel 都不在这条全局串行链内。N 发布 `deps_prepared` 后，N+1 owner 可以进入 Register；N 自己再并行做 Fanin/Build。实现中的明确边界见 [pa_shared_submit_path.h](same_core/common/pa_shared_submit_path.h#L936)。

### 4.5 Fanin 与 Build

Fanin 查询只接受严格早于当前 task 的 producer，并把结果复制到 SubmitContext/LocalSlot。Build 之后不再依赖临时 TaskArgs 指针。

Build 为本核 private slot 建立完整执行包。slot 只有两个，满时会通过 WaitForSlot 先执行 ready 任务。当前 best 样本没有 RingBackpressure kernel，说明两槽容量没有形成可见的单独背压阶段。

### 4.6 Kernel、完成发布与 FinalDrain

当前 CCEC 的 `ExecuteKernel()` 同步调用真实 Cube/Vector 模拟体，见 [ccec_ops.h](same_core/ccec/ccec_ops.h#L293)。发射 engine 指令后仍会在最终 FIX/MTE3 完成点等待；这段时间会落在 kernel 区间，Scalar 目前不能同时继续另一份调度 continuation。

所有 worker 回放完 1,280 个 Submit 后，先进入层次 final barrier，再持续 Drain 自己的 slot，直到“所有 worker 不再生产新 slot”和“本核 slot 为空”同时成立。对应状态机见 [pa_scheduler_core.h](same_core/common/pa_scheduler_core.h#L4955)。

## 5. Best 样本的全局性能

### 5.1 墙钟和生命周期

| 指标 | 本次结果 | 说明 |
| --- | ---: | --- |
| 完整 Submit makespan | **1,659.167 µs** | 最早 Submit 到最晚 Submit |
| 最早 Submit | 30.532 µs | core 82，AIV0，task 0 |
| 最晚 Submit 结束 | 1,689.699 µs | core 8，AIC，task 1279 |
| Orchestration 全局区间 | 1,661.315 µs | 含 Submit 间空白 |
| Worker lifecycle 全局区间 | 1,738.577 µs | Orchestration 加 FinalDrain 尾部 |
| 最晚 worker 完成 | 1,767.166 µs | core 47，AIV1 |

本次 Submit 的关键尾部由 AIC core 8 决定。不能据此断言所有运行都由 AIC 决定，但它与 AIC 承担 QK/PV Cube 工作、只有 32 个 AIC 的负载结构一致。

### 5.2 每核时间分布

下表都是“每核中位数”，用于描述典型 AIC/AIV 的负载差异：

| 每核指标 | AIC | AIV | 主要解释 |
| --- | ---: | ---: | --- |
| Submit envelope | 1,612.685 µs | 1,585.374 µs | 首尾 Submit 及中间空白 |
| Submit union | 1,427.968 µs | 1,413.957 µs | 本核所有 Submit 区间之和 |
| Submit 间空白 | 185.186 µs | 172.347 µs | callback/状态推进及记录边界 |
| EfDrain kernel union | 550.356 µs | 223.742 µs | AIC 数更少且承担 QK/PV |
| Claim | 260.434 µs | 264.246 µs | 两类核的竞争总负担接近 |
| Materialize | 123.087 µs | 129.207 µs | winner 数和 output 形状共同决定 |
| Register | 30.268 µs | 314.068 µs | AIV owner 更常提前撞到有序前沿 |
| Fanin | 58.753 µs | 47.277 µs | task 类型不同 |
| WinnerBuild | 68.178 µs | 81.334 µs | UP 的参数多，主要落在 AIV |
| FinalDrain | 70.792 µs | 102.242 µs | replay 后剩余 slot 和 barrier 尾部 |

最明显的角色差异有两个：

1. AIC 的 engine 工作更重，容易决定 Submit 尾部；
2. AIV 的 Register wait 显著更大。AIV 经常赢得 Alloc/SF/UP，在前序 AIC task 还未提交 metadata 时就到达 Register，因此把跨核到达差异显式表现为等待。

第二点是相关性解释，不是“把 AIV Register wait 删除就能等量缩短 283 µs 墙钟”的结论。多个核的等待高度并发，只有把等待变成有用工作或改变关键路径，才会形成端到端收益。

本次 owner 分布还显示：256 个 Alloc 全部由 AIV 赢得，其中 AIV0 为 118 个、AIV1 为 138 个；64 个 AIV 中有 62 个赢过 Alloc，活跃核各赢 1–9 个，中位数 4 个。这是本次自然到达顺序的观测结果，不是协议规定，也不能据此把 Alloc 候选永久裁成 AIV。把五类 task 合看，每核 winner 数为 9–17 个，中位数 13 个；每核实际 kernel 数为 6–17 个，中位数 9 个。当前不存在“只有少数固定核承担全部 winner”的明显证据。

## 6. 纯 Submit 控制开销分布

### 6.1 统计分母

96 核的 Submit union 累计为 136,026.458 µs，其中实际 engine kernel union 为 31,248.148 µs。扣除 kernel 后，本文用于归因的 Submit Scalar/控制 core-time 为：

```text
136,026.458 - 31,248.148 = 104,778.310 µs
```

这里仍包含 atomic return-ready 等待，因为它真实占住当前 Scalar 控制流；后文会把 atomic 与非 atomic 单独拆开。这个分母不是“Scalar ALU busy”，也不是端到端墙钟。

### 6.2 独占阶段分布

| 独占阶段 | 96 核累计 | 占 Submit Scalar/控制 | 主要内容 |
| --- | ---: | ---: | --- |
| Claim | 25,379.051 µs | **24.222%** | 候选判断、local/root CAS、结果处理 |
| EfDrain control | 21,287.721 µs | **20.317%** | 扣除 kernel 后的 slot 扫描、fanin poll、完成发布 |
| Register | 21,203.695 µs | **20.237%** | 前序等待、writer metadata、完成字 handoff |
| Materialize | 12,309.132 µs | **11.748%** | heap/descriptor/writer delta/SharedOutput 发布 |
| Submit residual | 11,988.212 µs | **11.442%** | ArgBuild、阶段桥接、记录边界及短尾 |
| WinnerBuild | 7,286.698 µs | **6.954%** | LocalSlot payload、tensor/scalar/fanin 复制 |
| Fanin | 4,901.697 µs | **4.678%** | writer 查询、历史回退、producer 去重 |
| AllocComplete | 422.104 µs | **0.403%** | Alloc 无 kernel，直接完成 |
| **合计** | **104,778.310 µs** | **100.000%** | 不含 Submit 内 engine kernel |

这张表给出三个当前最大的累计区域：Claim、EfDrain control、Register。三者性质不同：Claim 主要是并发 atomic，Register 主要是顺序等待，EfDrain control 是依赖检查与 completion 推进，不能用同一种手段优化。

### 6.3 Submit 外仍有多少 Scalar 时间

Orchestration 内 96 核累计时间为 153,130.420 µs：

- Submit envelope 累计 153,001.449 µs；
- Submit union 累计 136,026.458 µs；
- Submit 间空白累计 16,974.991 µs；
- setup/tail 只有约 129 µs。

扣除 Submit 内 kernel 后，Orchestration 的 Scalar/控制 core-time 为 121,882.272 µs，其中 Submit 间空白占 13.927%，Submit 内 Scalar/控制占 85.967%。因此只看 Submit 是合理的主线，但 Submit 之间约 17 ms 的核累计空白也不是零。

FinalDrain 另有 8,907.407 µs core-time，其中 kernel 1,062.080 µs、control/residual 7,845.327 µs。完整 worker lifecycle 扣除全部 kernel 后的 Scalar/控制 core-time 为 129,727.599 µs，FinalDrain control 占 6.048%。

## 7. Winner、loser 与 not-attempted

### 7.1 为什么不能用整条 loser Submit 衡量 loser 本身

| Submit actor | 数量 | 含 kernel 的平均 Submit | 扣 kernel 后平均控制时间 | 说明 |
| --- | ---: | ---: | ---: | --- |
| winner | 1,280 | 38.299 µs | 38.299 µs | 当前 winning Submit 内没有执行自己刚 Build 的 kernel |
| true loser | 72,448 | 0.728 µs | 0.593 µs | 373 条 actor 在 Claim 前替旧 slot 执行了 374 个 kernel |
| not-attempted | 49,152 | 0.697 µs | 0.260 µs | 600 条 actor 在 Claim 前执行了 601 个旧 kernel |

winner 的 kernel 在后续 Submit 或 FinalDrain 落地。因此：

- 长 loser/not-attempted bar 多数反映 EfDrain 替旧 task 做事；
- `claim.lost` 才是 Claim loser 的直接边界；
- “winner Submit 没有 kernel”不表示 winner 没执行，只表示 Build/Execute 时间解耦。

### 7.2 Claim 细分

| Claim 类型 | 数量 | 累计 | mean | median | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| won | 1,280 | 820.571 µs | 0.641 µs | 0.645 µs | 0.765 µs | — |
| true lost | 72,448 | 24,439.297 µs | 0.337 µs | 0.320 µs | 0.597 µs | 1.997 µs |
| not-attempted | 49,152 | 119.183 µs | 0.002 µs | — | — | — |

Claim 的 25,379.051 µs 中：

- local/root CAS 共 21,055.756 µs，占 Claim **82.965%**；
- 非 atomic 部分共 4,323.295 µs，占 Claim 17.035%；
- Claim atomic 单独占全部 Submit Scalar/控制分母的 **20.096%**。

atomic 明细：

| atomic | 次数 | 累计 | mean | median | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| local CAS | 73,728 | 18,685.860 µs | 0.253 µs | 0.259 µs | 0.316 µs | 0.874 µs |
| root CAS | 9,216 | 2,369.896 µs | 0.257 µs | 0.261 µs | 0.335 µs | 1.481 µs |

72,448 个 true loser 中：

- 64,512 个在 local 层失败，只做 1 次返回型 CAS；
- 7,936 个赢 local、输 root，做 2 次返回型 CAS；
- 扣除这些 atomic 后，true-loser 软件壳累计约 4,021.984 µs，平均仅 **55.5 ns**，中位数约 50 ns，p95 约 83 ns。

结论很明确：当前 loser 的主要矛盾不是 C++ 分支、统计加法或返回路径，而是必要的返回型 owner 仲裁。继续消减非 atomic loser 壳的绝对上限已经很小；若要大幅下降，只能在不改变候选人口和 exactly-one-owner 语义的前提下减少返回型 atomic 数量或同地址竞争宽度。

### 7.3 EfDrain 中的 fanin poll

除了 Claim，另一个数量较大的返回型读取来自 slot ready 检查。本样本有 29,545 次直接 `fanin_flag_load`，事件累计 7,581.371 µs，平均约 0.257 µs。它们已经包含在 EfDrain/FinalDrain 父区间中，不能再次加到 104,778.310 µs 分母。

FinalDrain 还把紧密循环中的读取聚合成 PollBatch：46 条 fanin batch 代表 1,521 次逻辑 load；96 条 replay-done batch 代表 5,609 次逻辑 load。PollBatch 的时长包围整段重复循环，可能包含 SpinHint 和其他状态检查，只能用于判断轮询强度，不能解释成单条 atomic 延迟。

与 Claim 不同，fanin flag 通常由少量实际 consumer 在 producer 完成前后读取，没有出现 96 核同时争同一 Claim 地址的同等级异常。因此它是 EfDrain control 的重要组成，但当前证据不支持把它排在 Claim Tournament 之前。

## 8. Winner 路径按 task kind 分析

### 8.1 Winning Submit

| Task | Winning Submit mean | Materialize mean | Register mean | Fanin mean | Build/Complete mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Alloc | 40.818 µs | 13.416 µs | 23.636 µs | — | 1.649 µs |
| QK | 18.586 µs | 7.627 µs | 1.890 µs | 3.525 µs | 3.547 µs |
| SF | 47.553 µs | 13.716 µs | 23.544 µs | 2.669 µs | 5.381 µs |
| PV | 20.420 µs | 7.734 µs | 1.771 µs | 3.881 µs | 4.911 µs |
| UP | **64.117 µs** | 5.591 µs | **31.986 µs** | **9.072 µs** | **14.625 µs** |

各列没有强制精确相加为 Winning Submit，因为 Claim、ArgBuild 和很短的阶段桥接也在父区间内。

UP 是 winner 路径最重的 task：它没有新输出，Materialize 反而最短；真正重的是等待前序 metadata、查询多个 SharedOutputRef/INOUT writer，以及构造包含更多 tensor/scalar 的 slot。

### 8.2 Register：97% 是等待，不是写 TensorMap

| Register 子区间 | 96 核累计 | 占 Register |
| --- | ---: | ---: |
| 等待前序 `deps_prepared` | **20,603.595 µs** | **97.170%** |
| 发布 writer metadata | 168.892 µs | 0.797% |
| 发布本 task 插入完成字 | 431.208 µs | 2.034% |
| **Register 总计** | **21,203.695 µs** | **100.000%** |

Register 的 1,279 个前序等待记录累计执行 67,503 次返回型 load。task 0 没有前序，其余 task 每个 winner 各有一条聚合 wait 记录。

不同 task 的平均等待：

| Task | predecessor wait mean |
| --- | ---: |
| Alloc | 23.205 µs |
| QK | 1.454 µs |
| SF | 23.141 µs |
| PV | 1.276 µs |
| UP | 31.407 µs |

因此，单纯优化 `publish_writer_metadata` 的理论累计上限只有约 0.169 ms core-time；即使连完成 CAS 一起免费，也只能消掉 Register 的 2.83%。Register 的主要问题是 owner 到达严格 task-id commit 前沿后的空等。

但这 20.604 ms 是多个核并发等待的总和，不等于墙钟损失。真正有价值的方案需要在保持严格顺序的同时，让等待 owner 去做不依赖该 commit 的工作，而不是把等待从 AIV 迁移到另一个核或阶段。

### 8.3 Materialize

Materialize 共 12,309.132 µs，平均每 winner 9.617 µs：

| 子区间 | 累计 | 占 Materialize |
| --- | ---: | ---: |
| output 发布之前：物化、writer delta 等 | 5,890.937 µs | 47.858% |
| `publish_task_outputs` | 6,123.897 µs | 49.751% |
| 发布后的短尾 | 294.298 µs | 2.391% |

`publish_task_outputs` 内部：

- descriptor copy：2,004.416 µs，占 32.731%；
- descriptor DCCI flush：565.888 µs，占 9.241%；
- reserve/published atomic、固定屏障和 helper residual：3,553.593 µs，占 58.028%。

本轮有 1,024 个产生新输出的 task，共 2,048 个 output：

- descriptor DCCI：1,024 次、4,096 条 cache line、562.097 µs；
- output writer reserve FetchMax：2,048 次、459.912 µs；
- output published Exchange：2,048 次、451.155 µs；
- 256 个 UP writer-history DCCI：256 条 line、104.629 µs。

Materialize 有继续压缩空间，但它不是当前最大的独占区域；同时 descriptor 发布和 DCCI 是跨核 reader 正确读取 payload 的必要边界，不能把删除 DCCI 当普通代码精简。

### 8.4 Fanin 与 WinnerBuild

Fanin 共 4,901.697 µs，覆盖 1,024 个非 Alloc winner，平均 4.787 µs。主要共享访问数量为：

| 操作 | 次数 | 累计 |
| --- | ---: | ---: |
| output `published` 返回型 load | 2,048 | 523.201 µs |
| output `last_writer` 返回型 load | 2,048 | 528.101 µs |
| shared map head load | 1,280 | 322.241 µs |
| shared map tail load | 1,280 | 302.063 µs |
| descriptor invalidate | 2,048 次 / 4,096 lines | 72.347 µs |
| writer-history invalidate | 768 lines | 48.538 µs |

WinnerBuild 共 7,286.698 µs，平均每 kernel winner 7.116 µs。UP 的 Fanin 9.072 µs、Build 14.625 µs 都显著高于其他 task，说明“多个参数引用同一 producer”虽然已经在 fanin task id 上去重，descriptor 发布检查、解析和 slot payload copy 仍按参数发生。

这里值得研究 task-level 或 producer-level 的批量证明/复制方式，但必须保留：

- output 可能部分发布；
- INOUT 必须回退到 `< current task id` 的 writer；
- 同一 producer 的不同 output slot 仍有不同 descriptor；
- 方案必须能推广到不满足 PA 固定 3-producer 形状的算子。

## 9. Kernel 落点与 Scalar 占用

### 9.1 Kernel 时间

| Task | 数量 | core-time 累计 | mean | median | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QK / Cube | 256 | 10,729.009 µs | 41.910 µs | 41.020 µs | 45.608 µs | 50.001 µs |
| SF / Vector | 256 | 13,761.616 µs | 53.756 µs | 53.454 µs | 57.381 µs | 61.306 µs |
| PV / Cube | 256 | 7,160.755 µs | 27.972 µs | 27.388 µs | 30.784 µs | 33.835 µs |
| UP / Vector | 256 | 658.848 µs | 2.574 µs | 2.537 µs | 2.993 µs | 3.416 µs |
| **合计** | **1,024** | **32,310.228 µs** | — | — | — | — |

Kernel 落点：

- 975 个（95.215%）在后续 Submit 的 EfDrain 执行，core-time 31,248.148 µs；
- 49 个（4.785%）在 FinalDrain 执行，core-time 1,062.080 µs；
- FinalDrain 剩余 13 个 PV、12 个 SF、24 个 UP，没有 QK；
- 没有 kernel 落在 RingBackpressure。

### 9.2 当前 Scalar 与 engine 没有真正 overlap

当前同一个物理 winner 核负责 Build 和 Execute。CCEC workload 在最终 FIX/MTE3 完成前同步等待，因此 kernel 区间既表示 engine 在工作，也表示本核 Scalar 控制流不能继续另一份 Submit。

独立机制探针已经证明：

- 使用 `get_buf/rls_buf + try_wait(buffer_id, BUFFER_ID)` 可以动态查询最终 engine 尾部；
- 当前 AIV 模拟负载可稳定隐藏约 3.5–3.6 µs 尾部；
- AIC 探针在 replay 足够长时可隐藏约 8.3–8.4 µs；
- 两份 128B continuation frame 的完整基础区间约 0.78 µs；
- 这些只是独立探针结果，尚未接入本 PA scheduler。

详细证据在 [perf_opt_record.md](../perf_opt_record.md) 的 §15.20–§15.22。正式接入必须保证：engine task 未完成时 slot 仍占用，task completion 不得发布，恢复后仍执行原最终 wait，再按 `vend -> flag` 顺序完成。

## 10. Residual 与观察代码影响

### 10.1 已知 residual

Submit residual 共 11,988.212 µs，主要可继续分为：

| residual | 累计 | 数量 | 单次平均 | 主要含义 |
| --- | ---: | ---: | ---: | --- |
| Claim -> Materialize | 1,836.627 µs | 1,280 winner | 1.435 µs | ArgBuild、ticket/finish bridge |
| Claim -> SubmitEnd | 10,019.471 µs | 121,600 loser/not-attempted | 0.082 µs | Finish/trace/返回短尾 |
| WinnerBuild -> SubmitEnd | 100.056 µs | 1,024 | 0.098 µs | winner 收口 |
| AllocComplete -> SubmitEnd | 32.058 µs | 256 | 0.125 µs | Alloc 收口 |

Submit 间空白共 16,974.991 µs，按 task transition 为：

| transition | 累计 | 单次平均 |
| --- | ---: | ---: |
| UP -> 下一 batch Alloc | 4,826.403 µs | 0.197 µs |
| Alloc -> QK | 4,062.148 µs | 0.165 µs |
| PV -> UP | 3,219.041 µs | 0.131 µs |
| SF -> PV | 2,649.893 µs | 0.108 µs |
| QK -> SF | 2,217.506 µs | 0.090 µs |

这些区间包含真实 callback/状态推进，也包含 trace 时间戳和 raw record 写入移动后的边界效应。它们适合指出“仍有未归因区域”，不适合未经 trace-free 对照就逐行微调。

### 10.2 泳道不是权威端到端基线

当前无泳道、无 atomic trace 的 perf-clock 对照中，最终 `atomicMax(INT64_MIN)` 恒等读取版本两轮各 12 次的均值为 1,466.810 µs 和 1,467.009 µs，合并约 1,466.910 µs。完整记录见 [perf_opt_record.md](../perf_opt_record.md) §15.12。

本 best full-swimlane 是 1,659.167 µs，数值上比上述 perf-clock 均值长约 192.3 µs（约 13.1%）。这个差只能说明完整观察版本仍有明显扰动，不能把 192.3 µs 全部归因成“写 JSON/泳道记录的固定成本”，原因包括：

- 两者是不同 ELF；
- full-swimlane 增加时间戳、atomic/DCCI raw 和分支；
- 代码体积、I-cache/D-cache 排布和竞争交错都会变化；
- 一个是单次 best，一个是多轮均值。

因此证据链应继续分工：

- perf-clock 决定候选是否真的提升端到端；
- full-swimlane 解释收益/回退可能落在哪个业务区域；
- submit-pmu 用于辅助判断 Scalar busy、I-cache request/miss 的同向变化；
- 三种 ELF 的绝对区间不能互相扣减后冒充局部收益。

### 10.3 DCCI 记录中的观察者流量

元信息汇总的 DCCI 总数为 4,480 calls、108,964 lines。其中泳道结束后导出 trace buffer 本身就有 96 份记录、288 次 DCCI、99,460 条 line，约 2,329.348 µs core-time。这部分在被测 worker lifecycle 之后，不应算入 Submit 业务阶段。

Submit 业务内主要 DCCI 是 output descriptor、UP history、Fanin history 和 Build descriptor 的发布/失效。分析 D-cache 时必须把 trace export 与业务 DCCI 分开，否则会错误认为 PA 热路径每轮清理了十万条业务 cache line。

## 11. 当前瓶颈判断

### 11.1 第一类：Claim 返回型 atomic 竞争

证据最明确：Claim atomic 占 Claim 的 82.965%，true-loser 非 atomic 壳平均只有 55.5 ns。当前两级 Tournament 已把 root 同地址宽度从 96/32/64 降到 8/6/8，但仍需 73,728 次 local CAS 和 9,216 次 root CAS。

如果目标是继续降低 loser，新的 owner election 至少要同时满足：

1. Alloc/QK-PV/SF-UP 的合法候选仍为 96/32/64；
2. 每 task 恰好一个 owner，不能只有 at-most-one；
3. loser 不等待 owner 完成；
4. owner election 与严格 TensorMap commit 顺序解耦；
5. 在 A5 同地址原子近似随并发宽度增长的约束下，返回型 atomic 总量或热点宽度确实下降；
6. 协议和状态增量的成本要小于节省的 atomic。

不能用固定减少候选核的方案作为主线，因为这会降低空闲核接管能力并改变调度语义。

### 11.2 第二类：Register 有序前沿的空等

Register 累计 21.204 ms，其中 20.604 ms 是前序等待。metadata 真正写入只有 0.169 ms。

值得研究的不是“如何让 CAS 快几十 ns”，而是：owner 在已经完成 Materialize、尚未轮到 N commit 时，能否保存最小 continuation，去做对当前 commit 无依赖的工作，再恢复 N。候选工作可能包括 loser replay 或推进本核旧 slot，但必须解决：

- `TaskArgs/SubmitContext/writer_delta/ticket` 保存位置和生命周期；
- 本核再次 win 时的有限 pending-owner 容量；
- N+1 绝不能绕过 N 发布 metadata；
- 已发布 SharedOutput、未发布 metadata 的失败回滚；
- 不能因“减少等待 core-time”而增加关键路径或破坏 same-core slot 所有权。

这本质上是 continuation/协程状态机，而不是简单地把 poll 挪位置。

### 11.3 第三类：engine 尾部与 Scalar 调度串行

1,024 个 kernel 都由 winner 本核同步执行。独立探针已经证明最终 engine 尾部存在可覆盖窗口，且 `try_wait(BUFFER_ID)` 可用。

最小集成方案应先限制每核只有一个 in-flight engine task：launch 后保留 slot 和 continuation，Scalar 继续 replay；在后续安全点 try_wait，ready 后恢复最终 wait、CompleteTask 和 slot release。该方向可能同时减少 EfDrain kernel 对 Submit 的阻塞和 FinalDrain spill，但需要用正式 perf-clock 证明收益。

### 11.4 第四类：UP 的 Fanin/Build

UP winning Submit 平均 64.117 µs，其中 Register wait 31.407 µs、Fanin 9.072 µs、Build 14.625 µs。排除 Register 等待后，UP 仍是参数处理最重的 task。

可重点检查：

- 同 producer 的多个 output slot 是否能共享一次 task-level published 证明；
- descriptor cache invalidate 是否能按连续 line 合并；
- Fanin producer 去重后，Build 是否仍重复解析相同 producer 元数据；
- LocalSlot 的 tensor/scalar/args 布局是否造成不必要的整块 copy。

任何优化应基于参数/producer 关系泛化，不能硬编码 UP、三个 accumulator 或固定 PA task id。

### 11.5 第五类：Submit 间和 residual 的冷代码/记录扰动

Submit residual 11.988 ms、Submit 间空白 16.975 ms 的核累计并不小，但单次只有约 0.08–1.44 µs，且容易被 trace 写入、I-cache 排布和边界取时污染。

这一类只有在 trace-free compile variant 中能稳定复现、且 perf-clock 同向时才值得保留。不能仅依据 full-swimlane residual 直接内联、预取或改业务结构。

## 12. 不可破坏的业务与测量约束

后续方案必须保持：

1. shared TensorMap metadata 按 task id 严格提交：N 的完成发布早于 N+1 的 metadata side effect；
2. Claim owner election 与插入顺序是两个协议；优化其中之一不能默认另一个自动成立；
3. Build task N 只能使用 producer `< N`，遇到 future writer 必须沿不可变 history 回退；
4. 每个 task 恰好一个 winner，loser 不执行 Materialize/Register/Fanin/Build；
5. 当前合法候选人口保持 Alloc 96、QK/PV 32、SF/UP 64；
6. fresh output descriptor payload 完整可见后才能发布对应 atomic 状态；
7. INOUT writer history payload 完整发布后，才能更新 last-writer/group latest；
8. kernel 完成顺序保持 `engine 完成 -> vend -> flag`，不能在 engine pending 时提前解除 fanin；
9. slot、TaskArgs、SubmitContext、heap reservation 和 engine token 的生命周期必须一起闭合；
10. 当前同核 Build/Execute 若被改变，必须显式设计跨核执行队列和 descriptor/slot 所有权，不能默认可偷取；
11. ordinary TensorMap 未来可能非空，不能把本 PA 的 `ordinary_count=0` 固化为通用调度器合同；
12. full-swimlane、perf-clock、submit-pmu 的绝对时间不得互相相减冒充优化收益；
13. core-time 只能用于归因，不能当墙钟；inclusive atomic/poll/DCCI 明细不能与父区间重复相加；
14. 不通过增加全局 DSB 来获得好看的 atomic 完成边界；观察能力不能改写热路径原语语义。

## 13. 建议交给后续分析者的具体问题

### 13.1 Owner election

在保持 96/32/64 候选和 exactly-one-owner 的前提下，是否存在比当前“每候选 1 local CAS、每组 winner 1 root CAS”更少返回型原子的通用协议？需要给出最坏交错证明，特别说明快 loser 越过旧 task 时不会造成某 task 无 winner。

### 13.2 Ordered commit continuation

能否把“Materialize 完成、等待 deps_prepared[N-1]”建模为有限 continuation，在不允许 metadata 越序的情况下，让该 Scalar 核继续做旧 slot Drain 或 loser replay？需要给出 pending context 上限、恢复顺序、失败回滚和 slot/heap 生命周期。

### 13.3 Engine continuation

如何把已验证的 `try_wait(BUFFER_ID)` 最小机制接入现有两槽 private ring，使一个核在 engine 尾部 pending 时继续 Submit，又不提前发布 completion？是否应先只允许一个 engine in-flight，并在第二个 winner 或 slot 满时强制收口最老 continuation？

### 13.4 SharedOutputRef 批量证明

对于一个 task 多个参数引用同一 producer 的情况，能否用一次 producer-level publication/version 证明替代多次返回型 load，同时仍支持不同 output slot、部分发布和 INOUT future-writer 回退？

### 13.5 关键路径而非核累计

Register wait 和 Claim atomic 的 core-time 都很大，但哪些等待真实落在 1.659 ms 关键路径上？能否在不增加高频 trace 字段的前提下，通过离线依赖图从现有 `(core, task, phase)` 和 task DAG 重建一条 critical path，区分“并发等待总量”与“端到端可节省量”？

## 14. 当前结论

当前 shared TensorMap PA 已经实现了以下关键架构分离：

- 所有 worker 回放同一 task 流，但每 task 只有一个 Claim owner；
- Materialize 可以并行；
- 只有 metadata commit 按 task id 严格串行；
- Fanin、Build 和 kernel 均已移出全局插入链；
- winner 在本核 Build，后续仍由本核 Drain 执行；
- PA fresh output 走 task-indexed SharedOutputRef 快速路径，UP INOUT 走 writer history。

best full-swimlane 的完整 Submit 为 **1,659.167 µs**。扣除 Submit 内 engine kernel 后的 104.778 ms Scalar/控制 core-time 中，Claim 24.222%、Register 20.237%、EfDrain control 20.317%，是当前三大区域。

其中最硬的两条证据是：

1. Claim 的 82.965% 是返回型 atomic；true-loser 非 atomic 壳平均只有 55.5 ns；
2. Register 的 97.170% 是等待前序 task metadata commit，真正 writer metadata 只占 0.797%。

因此，后续若想获得显著收益，重点不应是继续微调几十 ns 的 loser 分支或 Register 写入体，而应分别解决：

- 更少返回型原子、但仍严格 exactly-one 的 owner election；
- 把有序 commit 等待转成可恢复的有用工作；
- 利用已验证的 engine completion 查询，把 Scalar 调度与 engine 尾部 overlap；
- 降低 UP 多引用 Fanin/Build 的通用数据搬运和发布检查成本。

这些方向都必须最终回到无观察代码的 perf-clock A/B 判断是否保留；泳道负责说明机理，不负责单独宣判端到端收益。
