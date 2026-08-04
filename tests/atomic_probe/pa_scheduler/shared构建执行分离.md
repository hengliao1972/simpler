# Shared TensorMap 构建与执行分离设计记录

## 0. 当前状态

| 项目 | 状态 |
| ---- | ---- |
| 目标 | 让 task 的构建 owner 与 kernel 执行 owner 可以是不同物理核 |
| 当前代码 | 96 Scalar 通过中央 ticket 恰好一次 Build；Build owner 发布 task-indexed shared payload，K2 排除 Build owner 后的 1 或 2 个 eligible executor 竞争执行 |
| 本文性质 | 持续更新的架构与内存模型设计记录 |
| 正式实现 | S0–S6.53 已形成中央 ticket + 严格 TensorMap 插入 + K2 异核 Execute；公共计划显式发布 execution route/count，A5 单 lane placement 已独立于 PA adapter；执行扫描得到的完整 control 快照直接参与 Claim CAS；execution drain 采用 16 组单向 arrival，并在同一次 FetchAdd 中汇合 owner-local 完成数，固定 root 不再逐 task 原子扫描；正式单轮 TensorMap 实例明确禁止回收，lookup 不再读取恒为 0 的 bucket head；descriptor 引用、unique-ticket 单 CAS、发布 Exchange 非等待和 winner fatal 重复读取候选均已撤销 |
| CPU 正确性用例 | S1–S4 K2、S5a 对侧角色 Build 与 S5b 全 96 Scalar Build 门槛已完成 |
| A5 跨核发布探针 | S2 已完成，100 轮共 3200 case 通过 |
| A5 PA 功能/性能 | S6.43 起，唯一裁决口径为最早 startup 起点到最后 FinalDrain 结束；旧 Submit-only 与 first-Submit-to-FinalDrain 数据只保留为历史证据，不与新口径直接相减。S6.43 平面基线 6 次中位为 `1.465 ms`；功能与终态全部 PASS |
| S4 动态 Execute election | K2 首版已通过 CPU B1/B256 和 A5 B1/B256；B256 中两候选都有实际胜出，非法 owner 为 0 |
| S5 Build 拓扑 | S5a 已通过 CPU/CCEC/A5；S5b 五类 task 全 96/G8 已通过 CPU/CCEC/A5 B1/B256，物理 Claim CAS 精确闭合 |
| 当前调度缺口 | 双 token Claim-first 已解决旧的 FinalDrain backlog；S6.29 复用 Claim 快照，S6.32–S6.36 收敛 drain，S6.38 删除 root 的 2560 次逐 task 终态读取，S6.42 再删除 1280 次无回收 lookup head 原子读取。当前 startup-to-FinalDrain 的 trace-free 中位约 `1.465 ms`，下一步优先验证 startup 分组 Atomic，不能再用单次诊断 core-time 代替冻结 A/B |
| 明确非目标 | 不引入 `try_wait`、engine continuation 或“kernel 运行期间同一 Scalar 继续调度” |

本文先定义需要证明的内存合同，不预设最终一定采用中央队列、per-core 队列或 task-indexed cell。任何候选实现都必须先通过本文列出的跨核发布、唯一执行和生命周期门槛，再讨论性能；只有引入 cell 复用时才需要回收门槛。

相关现状与证据：

- [shared_tensormap_PA业务与性能分析.md](shared_tensormap_PA业务与性能分析.md)
- [ATOMIC_USAGE_GUIDE.md](../ATOMIC_USAGE_GUIDE.md)
- [cache_preload_usage_guide.md](../cache_preload_usage_guide.md)

## 1. 要解决的问题

当前 shared TensorMap PA 的一个 Claim winner 同时承担：

```text
Materialize
-> Register
-> Fanin
-> Build private LocalSlot
-> 后续在本核 EfDrain/FinalDrain 执行 kernel
-> CompleteTask
```

Build 和 Execute 虽然在时间上分离，但仍绑定同一个物理 worker。这样做的优点是简单：`LocalSlot` 从创建到释放都只有本核访问，跨核可见性只需要处理 TensorMap、SharedOutput 和 task completion。

新的目标是：

```text
构建 owner：Materialize -> Register -> Fanin -> Build portable task
                                           |
                                           v
                                      跨核发布执行包
                                           |
执行 worker：领取兼容 task -> 绑定本核上下文 -> 等 fanin -> kernel -> completion
```

长期预期收益是让 AIC 主要负责 QK/PV 的 Cube 发射与完成，让其他有空闲 Scalar 能在 Cube 执行期间继续构建后续 task。SF/UP 同理可以由空闲 Scalar 构建，再交给 AIV 执行。但是，“Build owner 扩大到所有 Scalar”与“已构建执行包可以跨核消费”是两个独立改动；第一版先保持现有 Build 候选核拓扑，只证明后者。

这个目标新增了两类共享对象：

1. **已构建 task payload**：TensorDesc、scalar、fanin、function 等执行所需信息；
2. **执行所有权状态**：谁可以读取 payload、谁取得唯一执行权、何时完成、何时允许复用。

因此，原来只围绕 TensorMap 建立的内存合同已经不够；执行包本身必须成为一个完整、可证明的跨核发布协议。

## 2. 三种所有权必须分开

后续讨论统一使用三种 owner，不再笼统称为 winner：

| 角色 | 职责 | 是否要求特定核型 |
| ---- | ---- | ---------------- |
| Build owner | Materialize、Register、Fanin、构造 portable task payload | 理论上只要求 Scalar 能执行对应代码 |
| Execute owner | 唯一领取已构建 task，绑定执行核上下文并发射 kernel | QK/PV 必须 AIC，SF/UP 必须 AIV |
| Completion owner | engine 完成后发布 vend/flag，结束执行槽生命周期 | 原则上与 Execute owner 相同 |

当前代码把三者隐式合并为 Claim winner。分离之后，必须分别证明：

- 每个 task 恰好一个 Build owner；
- 每个非 Alloc task 恰好一个 Execute owner；
- 只有 Execute/Completion owner 能发布 task 完成；
- Build 完成不等于 kernel 完成；
- TensorMap 插入完成不等于执行包已经发布；
- 执行包发布也不等于 fanin 已 ready。

这几种状态不能复用同一个布尔量表达。

本文所说的 Completion owner 是 Execute owner 在完成阶段承担的责任，不是首版再新增第三次 owner 仲裁。

### 2.1 先给出内存模型结论

`same_core` 只表示“Build 和 Execute 由同一个 worker 完成”，不表示它的整套内存模型都是本核私有的。现有 shared TensorMap 路径已经有三类真实跨核发布：

1. Materialize 把 fresh output `TensorDesc` 发布到 task-indexed `shared_outputs`；
2. Register 通过 `task[N-1].deps_prepared -> metadata(N) -> task[N].deps_prepared` 严格保序插入 TensorMap；
3. Execute/Complete 通过 task `vend/flag` 向其他核发布 kernel 完成，fanin reader 跨核读取。

`cross_core` 不应重做这三套协议。它真正新增的是第四类对象：**已构建执行包的跨核发布、唯一领取和生命周期**。因此实现边界应是“保留现有 TensorMap/SharedOutput/completion 合同，扩展 dispatch payload 合同”，而不是重写整个 shared 调度器。

### 2.2 `same_core` 与 `cross_core` 必须相同的合同

| 对象/阶段 | 现有 `same_core` 合同 | `cross_core` 决策 |
| --------- | --------------------- | ----------------- |
| Build owner | 现有 per-task Claim 保证每个 task 唯一 Build owner | 首版原样复用，不同时改 Claim 协议 |
| output 内存 | shared heap 为 task 保留 GM 区域；`TensorDesc` 指向该共享内存 | executor 直接写该 GM 地址，不再发明一套 output “转移所有权”协议 |
| SharedOutput 发布 | descriptor 普通写 -> `FlushRegion()` -> atomic `published` | 原样保留；`published` 仍只表示 descriptor 可读，不表示 kernel 完成 |
| TensorMap 有序插入 | 等 `deps_prepared[N-1]`，发布 N 的 metadata/history，再 CAS 发布 `deps_prepared[N]` | 原样保留；跨核 Execute 不参与该串行链 |
| fanin lookup | Build 阶段只接受 `producer < N`，再把 fanin 保存到执行包 | 过滤边界和 TensorMap 语义不变 |
| fanin ready | consumer 以 task completion `flag` 的返回型 atomic load 判断 producer 完成 | 原样保留，只是读取它的 worker 可能换了 |
| kernel 完成 | engine 真正完成后先发布 vend，再发布 completion flag | 顺序和语义不变，由 Execute owner 完成 |
| 核型约束 | QK/PV 由 AIC 执行，SF/UP 由 AIV 执行 | 不变；首版只映射到另一个兼容核 |
| cacheline 原则 | atomic-only line 不做 payload DCCI；ordinary payload 依靠显式 DCCI 发布 | 不变，并延伸到新 execution payload |
| fatal/收口 | fatal 是 terminal atomic，replay 停产后才允许 final drain 收口 | 语义不变，但 final drain 的完成条件需扩展 |

这里特别需要澄清一点：现有 `TaskCell::flag/vend/deps_prepared` 三个 atomic 字共用一条 64B cacheline，这条 line 在热路径中始终 atomic-only，不执行 DCCI。因此通用原则是“**atomic control 与 ordinary+DCCI payload 不得共线**”，不是“每个 atomic 必须独占一行”。新 execution state 首版先独占一行是为了缩小正确性证明面，不宣布它是最终布局。

### 2.3 两种方式真正不同的点

| 问题 | `same_core` | `cross_core` 新合同 |
| ---- | ----------- | ------------------- |
| dispatch payload 位置 | `WorkerState::LocalSlot` 属于 Build owner 私有 | 先发布到 64B 隔离的 shared execution cell |
| payload publication | `built` 可早于 payload 填充，因为只有本核读 | payload 全部写完并 `FlushRegion()` 后才 atomic 发布 `BUILT` |
| payload acquire | 无跨核 acquire | Execute owner 先取得所有权，再 invalidate payload，最后 ordinary load |
| `args/context` | Build 时直接指向本核 slot/context | portable payload 不带 builder 私有指针；executor 在本地重建 |
| fanin 执行期状态 | `SlotReady()` 可原地压缩本核 private fanin | shared payload 在 `BUILT` 后不可变；当前每核固定两个紧凑 execution token，压缩/游标只放对应 token |
| completion vend | shared Materialize 把 `reservation.aggregate_vend` 写入 Build owner 的 `worker.heap_next`，同核 `CompleteTask()` 随后消费 | Build 时冻结 task 的 completion-vend 快照并显式随 payload 交接；不得读 executor 的 `heap_next` |
| Execute exactly-once | 隐含由同一 Claim winner 保证 | 先用单一指定 executor + atomic phase transition 验证，再单独加多候选仲裁 |
| cell 生命周期 | 本核 `occupied_count` 与 private ring 回收 | 首版 task-indexed 且单轮不复用；后续环形复用才引入 generation/reclaim |
| FinalDrain | 所有 replay actor 停产 + 本核 private slot 为空 | 还要证明所有 kernel cell 已 DONE、每核两个 execution token 均为 IDLE、engine 无 in-flight |
| Build 候选核 | 当前与 task 核型拓扑绑定 | 首版不改；扩大到所有 Scalar 是后续独立性能阶段 |

completion vend 是 heap 进度快照，不是 output 数据地址或内存所有权 token。真正的 output 地址已经在 `TensorDesc` 中，executor 对该 GM 区域执行 kernel。

### 2.4 差异点与当前执行计划的映射

| 新增证明责任 | 已准备的执行阶段 | 首个通过标准 |
| ------------ | ---------------- | ------------ |
| control/payload cacheline 隔离、portable ABI | S0 静态布局与 CPU 结构检查 | 对齐/大小/offset 断言闭合，payload 不含 builder-private pointer |
| 状态机 exactly-once | S1 CPU 定向并发交错 | 不完整 payload 不可读，单候选与多候选都无 duplicate/missing |
| 本地 staging、单向 GM pack、一次批量 DCCI | S0 固定唯一 pack/publish helper，S1 记录 helper 调用，S3a 单独量交接成本 | 每 task 一次 forward pack、一次 FlushRegion、一次 winner InvalidateRegion，binding 后零 shared-payload 回读 |
| A5 ordinary payload 发布/取得 | S2 独立 CCEC 跨核探针 | 不依赖 kernel-end 自动 DCCI，延迟注入和多 cacheline 均读到精确值 |
| DCache preload 可选性能 hint | S2 先在关闭时闭合正确性，再于 S2/S3a/S3b 做编译变体 A/B | on/off 合同完全相同，只保留正确性不变且性能稳定改善的位置 |
| 使用 shared payload 本身的代价 | S3a task-indexed cell，仍映射给 Build owner | 正确性不变，单独量出 publication/copy 税 |
| 真正跨核的 args/context/vend/fanin 交接 | S3b 为每个 task 预路由两个兼容候选；每核在实际 Submit 已知 `(task_id, Kind)` 时只登记属于自己的候选任务，再由 Build owner 唯一确定异核 executor | B1/B256 的 task/descriptor/fanin/vend/completion 精确校验全部一致；热发现路径每 task 最多两个 control observer |
| 多 executor 唯一领取 | S4 动态 election | 先保证恰好一个 executor，再比较 atomic 成本 |
| executor 本地容量与全局 backlog 解耦 | S0–S5 先用单 token 闭合协议；S6 当前固定扩为两个 token | 有空 token 才发射 CAS；抢到后立即检查依赖，任一 owner-local ready task 优先执行 |
| Build owner 扩大到其他 Scalar | S5 独立改候选核拓扑 | 不借助跨核发布的正确性掩盖 Build 角色变化 |
| 复用、队列和回收 | S6 性能/容量优化 | task-indexed 单轮先闭合；需复用时才引入 generation/ABA 证明 |

这个顺序刻意不在首版同时改 Build Claim、执行队列和 cell 复用。
本轮性能优化也不引入 engine coroutine，避免把跨核发布、owner 仲裁、
回收与另一套调度机制混在一起。

## 3. 当前 private `LocalSlot` 为什么不能直接共享

### 3.1 `built` 的写入顺序只适用于本核

当前 [pa_frontend.h](same_core/common/pa_frontend.h#L1829) 中 `BuildSlotPayload()` 的顺序是：

```text
occupied = true
task_id/kind/function = ...
built = 1
PopulateSlotPayload(...)
```

`built=1` 先于 TensorDesc、scalar、args、context 和 fanin 的完整填充。因为 slot 只由本 worker 后续读取，这个顺序只是复刻 private ring 状态机，不是跨核 publication。

一旦执行核可以不同，`built` 必须改成最后发布的独立 atomic 状态：在 payload 全部写完、逐行 DCCI clean-out、DSB 完成后才能从 `BUILDING` 变为 `BUILT`。

### 3.2 `occupied/built` 不是 atomic，而且与 payload 共线

当前 [LocalSlot](same_core/common/pa_model.h#L1668) 的前两个字节是普通 `bool occupied/built`。它们既没有 atomic exactly-once 语义，也与 header/payload 共用 cacheline。

A5 已经实测：一条 line 上若同时存在 ordinary dirty 快照和其他核更新的 atomic word，后续 DCCI 可能把 atomic 新值冲回旧值。因此共享执行槽的状态控制字必须放在独立 atomic-only cacheline，payload DCCI 绝不能覆盖它。

### 3.3 布局不满足跨槽 cacheline 隔离

当前 `LocalSlot`：

- 大小 `4,824 B`；
- 对齐仅 `8 B`；
- 大小不是 64B 的整数倍；
- `WorkerState::slots` 是紧邻数组。

因此相邻 slot 可能共享边界 cacheline。private 模式下只有本核访问，问题不显式；共享后，一个 builder 对 slot N 做 DCCI 可能触及 slot N+1 的数据。新的共享执行槽必须从 64B 地址开始，payload 长度向 64B 对齐，任何两个可由不同核同时拥有的 slot 不得共线。

### 3.4 `args` 含 slot 自引用地址

当前 Build 会写入：

```text
args[i] = &slot.tensors[i]
args[local_context_index] = &slot.local_context
args[global_context_index] = &slot.global_context
```

代码见 [pa_frontend.h](same_core/common/pa_frontend.h#L1752)。因此有两种完全不同的消费方式：

1. **执行核直接使用同一份共享 slot**：这些绝对地址仍有效，但 slot 在 kernel 完成前不能回收，且 executor 不得修改共享 payload；
2. **执行核复制 active prefix 到自己的一个紧凑 execution token**：所有 self-pointer 必须重建，不能原样复制 `args[]`；当前每核固定两个 token，但不为此恢复 4-slot private ring。

第一版协议必须明确选一种，不能既复制 payload 又沿用 builder 写出的 self-pointer。

### 3.5 LocalContext 和执行核身份目前在 Build 时确定

当前 Build 写入：

- `local_context`；
- `global_context/sub_block_id`；
- `is_multicore`；
- `won_block/won_slot`。

其中部分字段与最终执行核或 BlockWon 资格相关。Build/Execute 分离后，应将执行包拆为：

```text
Portable Build payload：task/function/tensors/scalars/fanin/输出与完成信息
Executor binding：block/lane/sub_block/engine token/local context/workspace
```

不能让 builder 的 block/lane 身份被错误传给 executor。

### 3.6 当前 fanin 会在 slot 中原地修改

[SlotReady](same_core/common/pa_scheduler_core.h#L564) 在 shared 模式下会把已经 ready 的 fanin 前缀从 private slot 中移除，并修改 `fanin[]/fanin_count`。

共享执行 payload 应在 `BUILT` 后保持不可变，否则 executor 的普通写会形成 dirty cacheline，给后续 DCCI 和复用带来 stale writeback 风险。当前不恢复每核 4 个完整 `LocalSlot`，而是采用固定容量为 2 的紧凑领取状态：

- `SharedExecCell[]` 是全局 backlog，未领取任务始终保留在 `BUILT`；
- 每个 executor 固定两个紧凑 execution token；至少一个 token 为 `IDLE` 时才允许参与 `BUILT -> CLAIMED`；
- Claim winner 领取后把有效 fanin 和执行必要字段一次复制到所选 token，并立即完成第一次 fanin 检查；
- 第一个 token 未 ready 时允许领取第二个 task；两个 token 都占用时不得发射第三次 Claim；
- 任一已领取 token ready 时，Execute 优先于新 Claim 和新 Build。

不应直接在共享 payload 上复用现有原地压缩逻辑。

### 3.7 `CompleteTask` 隐含使用同核保存的 `worker.heap_next`

当前 [CompleteTask](same_core/common/pa_scheduler_core.h#L542) 把 `worker.heap_next` 写入 task vend。shared Materialize 成功预留 output heap 后，会把 `reservation.aggregate_vend` 写入 Build owner 的 `worker.heap_next`；Build 和 Execute 同核时，后续 `CompleteTask()` 消费的正是这份快照。

分离之后，executor 的 `worker.heap_next` 是另一个 worker 的状态，不得冒充该 task 在 Materialize 时冻结的 vend。执行包必须显式携带 completion-vend 快照，或者 shared heap 协议另行定义 task 级 vend。该 vend 只是 heap 进度快照，不是 output 地址或内存所有权 token。

### 3.8 当前 `StoreBarrier()` 不是新 payload 的发布协议

CCEC 的 [StoreBarrier](same_core/ccec/ccec_ops.h#L237) 为刻意的 no-op；现有 SharedOutput/TensorMap payload 依靠显式 `FlushRegion()` 完成 DCCI+DSB。

新执行包必须调用自己的显式 payload flush。不能在普通写后只调用 `StoreBarrier()`，也不能假设紧随其后的 atomic 自动让相邻普通 payload 对远端可见。

## 4. 已经由 A5 探针建立的内存规则

以下不是候选优化，而是当前 A5/CANN 9.1 已有证据支持的工程边界。完整依据见 [ATOMIC_USAGE_GUIDE.md](../ATOMIC_USAGE_GUIDE.md)。

### 4.1 以 64B cacheline 为最小协议单位

- A5 Scalar DCache 不能按 CPU coherent cache 理解；
- 不能只证明一个 8B word 没有竞争，必须说明它所在整条 64B line 的所有普通写、atomic 和 DCCI；
- atomic 控制 line 与普通 payload line 必须分离；
- 多个 atomic 可以在充分证明后共线，但整条 line 必须始终 atomic-only；第一版优先一条热点 atomic 独占一行。
- 即使两个 control 都是 atomic-only，若一个被持续 poll、另一个仍有并发 RMW，也不应共享 cache line。execution drain 最终只保留 16 条分组 `arrival`，每条独占 64B；反向 `release` 已由单向 root 收口合同消除。

### 4.2 ordinary payload 的写者发布顺序

```text
取得 payload 独占所有权
-> 必要时清理本核旧 cache entry
-> ordinary store 完整 payload
-> compiler barrier
-> 对所有 payload line 执行 SINGLE CACHELINE_OUT DCCI
-> DSB
-> atomic publish BUILT（reuse 版同时携带 generation）
```

连续多条 DCCI 可以批量发射，最后在 atomic publish 前统一一次 DSB；前提是中间没有动作依赖其中某条 line 已完成。

### 4.3 consumer 的取得与读取顺序

```text
atomic CAS 取得唯一 Execute owner
-> 消费 CAS 返回值确认成功
-> 调用 InvalidateRegion(payload)
   内部：逐行 DCCI -> DSB -> compiler barrier
-> ordinary load payload
```

上述是待 A5 探针证明的最小候选序列，不是已经宣布为硬件定理的 acquire。当前 CCEC `InvalidateRegion()` 已经内含逐行 DCCI、尾部 DSB 和 compiler barrier，因此文档不再默认在它之前叠加一个 DSB。S2 必须同时对比：

- 最小序列：消费 CAS 返回值 -> `InvalidateRegion()` -> load；
- 保守序列：消费 CAS 返回值 -> 前置 DSB -> `InvalidateRegion()` -> load。

只有最小序列在受控延迟和多 cacheline fresh-cell 探针中不能闭合时，才保留前置 DSB。未来引入 reuse 时还要重新验证该序列。这样既不脑补 atomic 自带 CPU 式 acquire，也不把未证实必要的屏障放进热路径。

当前 DCCI 没有已经验证的按地址 invalidate-only 模式，它可能 clean dirty line。因此该 consumer 在取得所有权前绝不能持有这批 payload line 的 dirty 普通副本。最简单的合同是：

- `BUILT` 后 payload 全生命周期不可变；
- 候选 executor 在 CAS 成功前只访问 atomic control，不预读 payload；
- 只有 CAS 成功的 executor 执行 DCCI 和 ordinary load；
- executor 若需要修改 fanin/context，先复制到 owner-private token。

### 4.4 禁止的写法

- 用 ordinary store、`st_dev`、DCCI 或 DSB 代替跨核 `state/ready/owner/done` atomic；
- 在包含 atomic state 的 line 上写 ordinary header 或执行 payload DCCI；
- 在 payload publish 前只做 DSB、不做 DCCI clean-out；
- consumer 看到 `BUILT` 后不刷新自身 stale cache 就读取 payload；
- 多个 executor 同时 DCCI/read 一个仍可能被 builder 修改的 payload；
- 使用 `ENTIRE_DATA_CACHE` 简化共享 slot 发布；
- 把 `dc_preload` 当成可见性、顺序或所有权原语。

## 5. 候选共享执行槽内存模型

本节是设计候选，不是已经存在的接口。

### 5.1 首版与复用版分开定义

首版使用 `SharedExecCell[task_id]`，每个 task 本轮只用一次，host 初始化后 device 不回收。因此它不需要 generation、`FREE(g+1)` 或 ABA 处理，最小状态机是：

```text
EMPTY
  -> BUILDING(build_owner)
  -> BUILT
  -> CLAIMED(execute_owner)
  -> DONE(execute_owner)
```

现有 Claim 已经保证唯一 Build owner；`EMPTY -> BUILDING` 首版仍用返回型 CAS 检查，作为重复 builder 的显式正确性门。`BUILT -> CLAIMED` 即使在固定单 executor 阶段也保留 atomic phase transition，但该阶段没有仲裁竞争；它只验证发布与所有权转换。

执行扫描读取到完整 packed control 后，若该快照已经是合法 `BUILT`，Claim
必须直接把**同一个快照**作为 `BUILT -> CLAIMED` CAS 的 expected 值，不得在
CAS 前再次返回型读取同一 control。CAS 才是唯一的执行所有权线性化点：快照
若已过期，CAS 返回最新值并按 `Lost/NotBuilt` 进入既有重新观察路径；因此复用
快照不会用旧状态取得所有权。只有没有上游扫描快照的独立 helper 调用，才允许
自行读取一次 control 后进入同一 CAS 实现。

每个 executor 当前固定保留两个本地 execution token。每个 token 的正常主干均为 `IDLE -> BINDING -> WAITING_FANIN -> ENGINE_INFLIGHT -> COMPLETING -> VEND_PUBLISHED -> COMPLETION_PUBLISHED -> IDLE`，错误收敛到 `FAULTED`。这些不是新的共享 atomic phase；跨核只需要清晰的 `BUILT`、`CLAIMED` 和 `DONE` 边界。每个 token 只保存 task id、有效 payload binding、fanin 游标和 completion 所需的紧凑状态，不是 4,824B `LocalSlot` 的另一份拷贝。

executor 在读取 ordinary payload 之前，就必须知道自己是否兼容、cell 是否对应请求 task，以及需要 invalidate 多少条 line。因此首版 packed atomic state 除 phase/owner 外，还必须随 `BUILT` 一次性发布：

- `engine_class`：AIC/AIV/无 kernel，供候选 executor 在不读 payload 的前提下过滤；
- `payload_lines`：本 task 已发布的 payload cacheline 数，供 winner 计算 `InvalidateRegion()` 范围；
- `task_id`：在任何 payload DCCI/read 前拒绝错误 cell 索引，并在 DONE CAS 中绑定同一 task。

`payload_lines` 必须在 DCCI 前先检查 `1 <= payload_lines <= kMaxExecPayloadLines`；越界时直接 terminal fatal，不能用损坏的控制字越界刷新 GM。task id 同时由 `SharedExecCell[task_id]` 索引、packed state 和 payload header 三处交叉验证。具体 bit 宽由 S0 静态断言锁定。

只有后续将 cell 改为有界 ring 并真正复用时，才扩展为：

```text
FREE(g) -> BUILDING(g) -> BUILT(g) -> CLAIMED(g) -> DONE(g) -> FREE(g+1)
```

该阶段再证明 generation、慢 reader 和 reclaim，不把它们提前塞进 task-indexed 内存模型门槛。

### 5.2 建议的物理分区

```text
cacheline 0：phase/owner/engine_class/payload_lines/task_id（单个 packed atomic-only state，不 DCCI）
cacheline 1..N：immutable portable payload（ordinary + DCCI）
```

约束：

- control 与 payload 地址范围完全不重叠；
- payload 起点和长度都按 64B 对齐；
- 相邻 slot 不共享 cacheline；
- task-indexed 首版没有 queue head/tail；后续引入队列时，它们与 slot state 分别评估热点；
- host 初始化不能留下与 device atomic 共线的 ordinary dirty 路径。

### 5.3 Portable payload 最小字段

| 字段 | 原因 |
| ---- | ---- |
| task id | consumer 检查 state 与 payload 对应同一 task；reuse 版再加 generation |
| function id/engine kind | 路由 AIC/AIV executor |
| active tensor/scalar/fanin count | 约束有效前缀，拒绝越界 |
| payload bytes | acquire 后与 atomic `payload_lines` 交叉校验紧凑序列化边界 |
| active TensorDesc 或稳定引用 | kernel 参数 |
| scalar 值 | kernel 参数 |
| fanin task ids | executor 判断依赖完成 |
| completion vend/reservation 快照 | 不能使用 executor 的 heap cursor |
| 通用 multicore 元数据 | 后续支持 joint task，不能永久按 PA 单 lane 特化 |

以下信息原则上不直接放入 portable payload：

- builder 栈地址；
- builder `TaskPayload *`；
- builder private `LocalSlot *`；
- builder 的 block/lane/sub-block；
- builder workspace 地址；
- 指向上述对象的预构造 `args[]` 指针。

### 5.4 首版 payload 逻辑布局

下图是 S0 应固化的逻辑布局，不是尚未求证就写死的 C++ bit 宽或最终 ABI：

```text
SharedExecCell[task_id]
├─ control cacheline（atomic-only）
│  └─ packed state
│     phase | owner | engine_class | payload_lines
└─ payload（64B aligned，ordinary-only）
   ├─ fixed header
   │  task_id | function id/address | counts | payload_bytes
   │  completion_vend | portable multicore metadata
   ├─ TensorDesc[active tensor_count]
   ├─ scalar[active scalar_count]
   ├─ fanin[active fanin_count]
   └─ tail padding to 64B
```

布局合同：

- control 中只有一个被 atomic 读改的 packed state，不得放 ordinary debug 字段；
- task id 一方面由 cell 下标确定，另一方面放在 payload header 中交叉校验；
- `engine_class/payload_lines` 随 atomic `BUILT` 发布，executor 不需偷读尚未 invalidate 的 header；
- `payload_lines` 从 payload 首地址起算，同时决定 builder `FlushRegion()` 和 executor `InvalidateRegion()` 的精确相同范围；
- acquire 后再以 header 的 `payload_bytes/counts` 重新计算有效边界，必须满足 `expected_bytes <= payload_lines * 64`；
- `TensorDesc` 依现有 ABI 为 128B，S0 要用 `static_assert` 继续锁定；descriptor 区从 64B 边界开始，便于两条 line 为一组复制；
- scalar/fanin 可以紧凑排列，因为它们都是同一 owner 的 ordinary payload；但尾部必须向上取整到 64B；
- 未使用尾部不必清零，但不得被执行、checksum 或 host 校验当成有效数据；
- cell stride 必须是 64B 整数倍，一个 cell 的最后一条 payload line 不得与下一 cell 共线。

当前实现不直接把 4,824B `LocalSlot` 当 shared payload。它只序列化 active prefix，然后由 executor 绑定到自己的一个紧凑 execution token/context；每核固定两个此类 token。

### 5.5 默认性能合同：本地准备，GM 只单向经过一次

same-core 优化已经形成一条必须延续到 cross-core 的经验：**控制计算尽量只使用寄存器和栈上小型 POD；只有必须持久或跨核的结果才落 GM，并且每条 destination line 单向写一次、最后统一 DCCI 发布**。

“owner-private”只表示没有其他核并发访问，不表示它不在 GM。当前 `WorkerState::LocalSlot` 虽是本 worker 私有，物理上仍是 GM 对象，同样可能付出 DCache miss。因此 owner-private GM 只用于必须跨 Submit/等待点存活的 token 状态，不应冒充临时 staging。

现有 `same_core` 中的 [SharedTaskWriterDelta](same_core/common/pa_shared_submit_path.h#L19) 就是可复用的结构模式：先在调用栈上的 trivial object 中准备 ordinary entry、bucket/局部序号和 symbol key，取得有序插入资格后才消费这份不可变计划，不在热串行段重复扫描 GM/args。cross-core execution payload 应复制这个模式，而不是在 Materialize/Register/Fanin 每一阶段都去修改 shared cell。

开发时的默认数据流必须是：

```text
Build owner:
  Claim / Materialize / Register / Fanin
  -> 小型 header、count、offset、fanin、completion-vend
     保持在寄存器/调用栈 staging
  -> 所有数量和边界确定后，执行一次 forward pack
  -> 每个有效 shared payload byte 只写一次
  -> 对整个 [payload, payload + payload_lines*64) 调用一次 FlushRegion
     内部逐行 DCCI(CACHELINE_OUT)，最后只用一次 DSB 收口
  -> atomic BUILT

Execute owner:
  先确认本核两个 execution token 中至少一个为 IDLE
  -> 只 poll/CAS atomic control
  -> 取得 CLAIMED 后对完整 payload 调用一次 InvalidateRegion
  -> 执行一次 forward read/copy 到 executor-private binding
  -> 从此只操作 private fanin/context/args
  -> 等待、kernel 和 completion 不再回读 shared payload
  -> 发布 DONE 后才把对应 execution token 恢复为 IDLE
```

这里的“一次 DCCI”指一次**批量发布 helper**：多 line payload 在硬件上仍需要每条 line 一条 DCCI，但中间不夹杂业务读写，末尾只一次 DSB。禁止把 `FlushRegion()` 放入 tensor/scalar/fanin 的字段循环，也禁止“先写 header -> flush -> 再补 count -> 再 flush”。

这条原则不等于把完整 payload 复制到大栈对象：

- header、count、offset、fanin 去重结果等小数据先在本地准备；
- 128B `TensorDesc` 等大前缀可以在最终 pack 时从已验证的源地址直接流式复制到最终 GM 位置，不再经过另一份 4KB 级栈镜像；
- S0 必须记录 staging 结构大小并检查编译后栈框/溢出；“写成栈变量”不能用来掩盖过大本地对象造成的反向回退。

executor-private binding 也必须区分两类：立即 ready 且在当前调用内完成的参数尽量保留在寄存器/栈；需跨 fanin 等待点存活的两个 execution token 才写 owner-private GM。每个 token 只写紧凑必要字段，不复制整个 4,824B `LocalSlot`，也不扩展为无界 pending 列表。

对应的专项开发门槛：

| 门槛 | 检查方式 | 通过标准 |
| ---- | -------- | -------- |
| shared payload 唯一写入口 | 代码审查 + 集中 `PackExecPayload()` helper | `BUILT` 前只有该 helper 写 payload，之后无 ordinary writer |
| 批量发布 | CPU test Ops 记录 `FlushRegion(address, bytes)` | 每个 kernel task 恰好一次，范围精确等于 `payload_lines*64` |
| consumer 一次刷新 | CPU test Ops 记录 `InvalidateRegion(address, bytes)` | 只有 Execute owner 调用一次，loser 为零次，范围与 publisher 一致 |
| 不回读 shared payload | 在 test build 用可控 read wrapper/阶段计数，perf build 完全消除 | private binding 完成后，fanin poll/kernel/completion 的 shared-payload read 计数为零 |
| 无分段 DCCI | 源码搜索 + CCEC 生成物审查 | pack 循环内无 DCCI/DSB，只在统一 publish helper 内出现 |
| staging 不反向膨胀 | `sizeof/static_assert` + CCEC 栈框/汇编审计 | 无完整 `LocalSlot` 栈镜像，无未解释的 GM spill，实际本地开销有记录 |
| GM 端到端代价 | S3a same-owner shared payload 对照 | pack + flush + invalidate + copy 单独可见，不用端到端波动掩盖新 GM 开销 |

已有探针显示 cold GM/DCache 访问处于百纳秒量级是合理的性能警戒，但具体延迟会受并发、预取、地址和流水重叠影响。因此本节把“减少 GM 触碰次数”定为结构门槛，但不用 `GM access 次数 × 100 ns` 冒充可兑现的 Submit 墙钟收益。

### 5.6 两种 payload 消费策略

| 策略 | 优点 | 主要问题 |
| ---- | ---- | -------- |
| executor 直接从 shared payload dispatch | 少一次 4KB 级复制；self-pointer 可指向稳定 shared 地址 | payload 必须保持到 kernel 完成；mutable context 必须外置；执行侧 DCCI 覆盖较多 line |
| executor 复制有效前缀到两个紧凑 token | 执行期状态和 fanin 可本地修改；第一项未 ready 时可领取第二项 | 必须重建 args 指针并增加一次有效前缀复制；两个 token 都占用时不能再领取 |

当前建议“共享 portable payload + 每 executor 两个紧凑 binding”：descriptor/scalar/fanin 只复制 active prefix，`args/local_context/global_context` 由 executor 重建。两个 token 是固定有界 admission，不恢复 same-core 的 4-slot ring，也不建立无界任务列表。

### 5.7 `dc_preload` 的预埋点与边界

本节只在 5.5 的“减少 GM 触碰次数”结构门槛已满足后才启用。`dc_preload` 不得用来给反复读写 shared payload、分段 DCCI 或过大 GM token 擦屁股。

现有 [cache_preload_usage_guide.md](../cache_preload_usage_guide.md) 已证明：A5 上 `dc_preload` 是可能被硬件当成 NOP 的性能 hint，不是可见性、顺序或所有权原语；128B/384B publish 和 128B consume 定向模型存在稳定改善，但不能直接外推为新 execution payload 的收益。

因此首版只**预留编译期可消除的 hook**，默认关闭，不把 preload 写入协议 ABI：

```text
Build owner:
  EMPTY -> BUILDING 成功
  -> optional preload 本 cell 后续要写的 payload destination line
  -> 利用既有寄存器计算/字段准备提供 lead
  -> ordinary pack/copy
  -> FlushRegion
  -> BUILT

Execute owner:
  CAS BUILT -> CLAIMED 成功
  -> InvalidateRegion(payload) 完成
  -> optional preload 后续要读的 shared payload source line
  -> 选择一个空闲 execution token、初始化 executor-local context 等独立工作
  -> ordinary copy/read shared payload
```

可预埋的候选 hook：

| 位置 | 优先级 | 前置条件 |
| ---- | ------ | -------- |
| builder 写 shared payload destination | 高 | 已取得 `BUILDING`，cell 无其他 writer，预取后有足够独立工作 |
| executor 读 shared payload source | 高 | 必须在 `InvalidateRegion()` 之后；否则 hint 可能随后被 DCCI 丢弃 |
| executor 写 private binding destination | 中 | destination 属于本 executor，但 slot 可能本来已 hot，需单独 A/B |
| packed atomic control line | 禁止默认接入 | preload 不参与 atomic 新鲜度/顺序，还可能污染热点竞争 |

实现约束：

- hook 的 CPU 实现是 no-op，CCEC 实现复用已查证的 `PreloadDataCache()`/`dc_preload`；
- on/off 变体必须保留完全相同的 CAS、DCCI、DSB、payload 布局和正确性校验；
- 不等待 preload 状态，不使用 preload 结果决定是否读写；
- 小 payload 立即读写时 hint 成本可能大于收益，不硬编码距离/密度阈值；S2/S3a 先扫描，S3b 再用真实跨核调度做配对 A/B；
- 若做滚动预取，只对后续仍在 payload 有效范围内的未来 line 发 hint，不能越过 `payload_lines` 触及相邻 cell。

## 6. 完整发布与执行协议

### 6.1 Build owner

```text
1. 复用现有 Claim 结果，确认本核是唯一 Build owner
2. 对 task-indexed cell 执行返回型 CAS：EMPTY -> BUILDING
3. Materialize，并按现有协议发布 fresh SharedOutput descriptor
4. 严格按 task id Register metadata，发布 deps_prepared[N]
5. Fanin lookup，过滤 producer >= current task
6. 冻结 completion-vend，在寄存器/本地状态中计算并校验 payload_lines
7. 可选对后续要写的 payload destination 发起 dc_preload；不参与正确性
8. 构造 portable payload；不写 builder-local pointer
9. Flush payload 全部有效 cacheline；FlushRegion 内部 DSB 收口
10. atomic BUILDING -> BUILT，同时发布 engine_class/payload_lines
11. Build owner 不再读写该 payload
```

task-indexed 首版的 cell 容量与 task 数一一对应，不存在设备侧等空槽。后续改成有界 ring 时，才必须在进入 Register 有序段之前预留容量；不能占住 `deps_prepared` 链等 executor 释放 slot。

### 6.2 Execute candidate/owner 与调度优先级

`SharedExecCell[]` 已经是 GM 中单轮不复用的全局 Built queue。当前选定的
调度模型不是 producer-driven Ready queue，也不是 ready-before-Claim；
executor 先竞争一个已经 `BUILT` 的 task，取得执行所有权并完成 payload
acquire 后，才检查该 task 的 fanin。每个核当前允许同时持有两个已领取 task，
槽数以后再参数化，首版固定为 2。

```text
1. 已领取的 Build ticket 必须完整发布到 BUILT；处理中途不切换角色
2. 回到调度决策点后，先逐一检查本核两个已占用 execution token
3. 任一 token 的 fanin ready：立即执行；不得先领取新 task 或 Build ticket
4. 已占用 token 均未 ready、且仍有空 token：扫描本核 K2-compatible cell，
   并保留本次返回的完整 control 快照
5. 快照为 BUILT 时，以该快照作为 expected 执行 CAS BUILT -> CLAIMED(self)；
   CAS 前不重复 load，CAS loser 不占 token，并重新观察
6. CAS winner 立即 Invalidate/复制完整 payload，重建对应 token 的 binding
7. 抢到后立刻检查 fanin；不能把首次 ready 检查推迟到下个 Submit
8. 第一个 token 未 ready 时，允许再抢一个 BUILT task 到第二个 token
9. 第二个 task 抢到后同样立即检查；随后再次复核两个已占用 token
10. 只要有一个 token ready，就优先同步执行 kernel 并发布 vend/flag/DONE
11. 两个 token 都占用且都未 ready，或 Built queue 暂无可抢任务，才领取 Build ticket
12. Build 完整发布后回到步骤 2；Build 全部停产后继续轮询/执行直到全部 DONE
```

这里的“Execute 优先”存在一个必要的非抢占边界：Scalar 一旦已经取得 Build
ticket，就必须完成该 task 的 Materialize、严格有序 Register、Fanin/Build 和
`BUILT` 发布。否则持有 `deps_prepared[N]` 下一棒的 worker 若中途执行较长
kernel，会把 TensorMap 串行插入链一起拖住。优先级只在**没有未完成 Build
ticket 的调度决策点**生效。

两个 token 都是 executor-private 的已领取上下文；对应 shared cell 已经进入
`CLAIMED`，不得被其他核重复领取，也不得因暂时未 ready 悄悄退回 `BUILT`。
每个 token 各自保存 task id、完整 active payload binding 和 fanin ready
prefix，避免轮询时反复读取已经确认完成的依赖。

K2 的 primary/secondary 都保持合法执行资格。首选核宽限若继续保留，只能
影响谁取得空 token，不能让已经 ready 的 owner-local token 排在 Build 之后。
CAS loser 必须继续扫描队列或处理本核已有 token，不能把一次竞争失败解释成
“本核没有 Execute 工作”。

槽数从 1 增到 2 会扩大 owner-private GM token 容量，但不会改变 shared
payload 的发布协议，也不需要真正 Ready queue、反向 fanout 或 Claim 前的
额外 payload DCCI。它的主要风险是两个槽同时被未 ready task 占满；首版必须
用定向交错测试和 A5 占用分布证明 PA 不会因此丢失可执行工作或永久停顿。

### 6.3 Completion 与生命周期

当前完成顺序必须继续保持：

```text
engine final wait
-> output 已完成
-> 以交接的 completion-vend 发布 task vend
-> task completion flag
-> shared execution cell DONE
```

在 task-indexed 首版中，`DONE` 是本轮终态，device 不执行 `DONE -> FREE`。host 在 kernel 结束后校验每个 cell 的唯一 owner、completion 和终态，从而先把“跨核交接”与“设备侧回收”彻底拆开。

FinalDrain 中，每核证明 scanner 封口且两个 token 均已复位后，按
`block_id % 16` 对所属 `exec_drain.arrivals[group]` 执行一次 FetchAdd。
当前 32 block、每 block 1 AIC + 2 AIV 的拓扑使每组精确包含 6 个 worker。
同一个 64-bit arrival word 的低 8 位累加 worker 到达数，高位累加该 worker
三个 placement 计数之和；即每核只发射一次
`FetchAdd(1 + (completed << 8))`。低位最大为 6，不会向完成数字段进位。

这份合计足以在 device 内证明本轮全部 kernel 已完成，前提来自现有合同而非
统计猜测：中央 Build ticket 给出完整 task 集；`BUILT -> CLAIMED` CAS 使每个
kernel 至多产生一个 Execute owner；placement 只在 completion flag 和 cell
`DONE` 均发布成功后递增；每核又必须等 scanner 封口、两个 token 清空且 engine
无 in-flight 才能到达。因此 16 组全部到齐且完成数合计等于计划显式
发布的 `executable_task_count`，即可推出所有可执行 task 恰好完成，不能由
重复完成补齐缺口。公共收口不允许用“每 batch 一个 Alloc”之类算子形状反推
该数量；host 算子计划器还必须独立重建预期值并与发布值交叉校验。

非 root 发布后即可结束；固定 root 只读取 16 个分组 word，确认每组低位为 6、
高位合计为计划 kernel 数后结束整个 device kernel。它不再串行读取 1280 个
cell control 和 1280 个 completion flag。host 在 kernel 返回后仍逐 task
校验 owner、cell、flag 和 payload，保留精确故障定位；这不是设备收口的替代。
反向 release 继续不需要，但证明不再依赖一层独立 replay barrier：每个
worker 只有在中央 ticket 返回越界后才能进入 FinalDrain；属于本核的候选若
仍是 `EMPTY/BUILDING`，scanner 会保留在该 task，不能提前发布 drain arrival；
arrival 又要求候选扫描结束、两个 token 全字段复位。因而 96 个 worker 全部
到达同时证明所有 builder 已退出、所有可执行 cell 已完成，root 再核对显式
`executable_task_count` 后成为全局收口点。缺失 cell 或缺失到达仍由既有
2 秒 device watchdog 发布 fatal，不能退化成永久自旋。

有界复用版才需要特别求证：新增加的 `DONE(g) -> FREE(g+1)` 与既有 completion atomic 之间采用什么硬件顺序。不能仅因源码中两个 atomic 前后相邻，就默认所有远端核观察顺序一致。

该复用版的回收条件至少包括：

- Execute owner 已完成最后一次 ordinary payload 读取；
- executor-private execution token 不再引用 shared slot；
- task completion 已经发布；
- 没有观察线程在本轮仍可能读取 payload；
- reclaimer 通过 generation-aware CAS 完成 `DONE(g) -> FREE(g+1)`。

复用前，下一 builder 已经取得整槽独占所有权。若不能证明每个 payload byte 都会覆盖，应先按所有权协议 DCCI 旧 entry，再写新 generation。

## 7. TensorMap 顺序与执行顺序的关系

构建/执行分离不能改变以下边界：

```text
task N Materialize + publish fresh SharedOutput descriptor
-> wait deps_prepared[N-1]
-> publish metadata(N)
-> publish deps_prepared[N]
-> Fanin/Build
-> publish execution payload(N)
```

fresh output cell 由 task 唯一 Build owner 独占，它的 descriptor 发布不需占住 TensorMap 有序链；本 task 最终的 `deps_prepared[N]` 仍在 metadata 发布后封口。严格顺序只约束 TensorMap metadata side effect，不约束 Build 包发布或 kernel 执行顺序。执行顺序由 fanin 决定：

- 无依赖 task 可以乱序并发执行；
- 有依赖 task 必须等 producer completion flag；
- 一个 task 已经 `BUILT` 不表示其 fanin ready；
- executor 不能因为队列头未 ready 阻止后续独立 task 执行。

当前正式 PA 是单轮、不复用的 TensorMap 实例：host 将
`reclaim_upto` 初始化为 `-1`，device 不推进它，host 终态也要求它
仍为 `-1`。因此每个 bucket 的 `head` 在整个 kernel 内恒为 0，
正式 Fanin lookup 可在编译期选用无回收实例，不再对 `head`
执行返回型原子读取。`tail` 仍由前序插入 owner 推进，lookup
必须保留其原子读取。无回收实例若遇到 `tail < 0`、
`tail - head > capacity` 或 slot seq 双检失败，必须直接报协议错误，
不能伪装成可回收场景。通用可回收实例仍保留 `head/tail`
双读、混合快照重读和 ABA 判定；未来一旦正式路径引入复用，
必须撤销该编译期选项，不得仅修改运行时 `reclaim_upto`。

因此，一个简单按 task id 排序的单 FIFO 可能产生 head-of-line blocking。它可以作为第一版正确性模型，但不能未经数据就当最终高性能结构。

### 7.1 当前调度优化的理论边界

在明确不引入 `try_wait`、不让同一个 Scalar 在 kernel 运行期间继续调度的
前提下，理想完成时间应尽量逼近下面几项的最大值，而不是把它们串行相加：

```text
max(
  TensorMap 严格插入关键链，
  全部 Build Scalar 工作 / 96，
  全部 AIC kernel 工作 / 32，
  全部 AIV kernel 工作 / 64，
  task DAG 关键路径
)
```

因此当前候选采用有界 admission 的 work-conserving list scheduling：每个
兼容 Scalar 最多从 Built queue 领取两个 task。每次取得所有权后立即检查
依赖；owner-local token 中只要有 ready task 就优先 Execute。两个已领取 task
都未 ready、或队列暂时没有可领取 task 时，该 Scalar 才领取一个 Build
ticket。不同 Scalar 上的 Build 和 Execute 可以重叠；同一 Scalar 上已开始的
Build 与同步 kernel 都不做抢占。

首版明确不做 producer-driven Ready queue。`SharedExecCell[task_id]` 固定槽
继续承载 Built queue 与 payload；K2 scanner 竞争 `BUILT -> CLAIMED` 即完成
领取。基础数据用于判断两个槽是否足够，以及第二槽带来的绕过收益，而不是
重新讨论是否应在 Claim 前构造 ready summary。

### 7.2 改调度器前必须补齐的基础数据

现有 `2026-08-03` B256 full-swimlane 已提供第一条结构证据：1024 个 kernel
中只有 158 个落在 EfDrain、52 个落在 OrchestrationTail，**814 个落在
FinalDrain**。它证明当前“每次取 Build ticket 前只给 Execute 一次推进机会”
确实留下了大量执行 backlog；但它还不能决定应使用哪种发现结构。

下一步先在独立诊断构建中累计每核计数，不给每次扫描增加 raw 事件：

| 基础数据 | 要回答的问题 |
| -------- | ------------ |
| 每核第一个/第二个 token 的 Claim 成功数 | 第二槽是否真实承担了绕过工作，而不是只扩大状态 |
| Claim 后首次检查即 ready / not-ready 的数量 | Built task 中实际有多少需要等待 |
| 第一槽未 ready、第二槽 ready 并先执行的次数 | 两槽模型消除 head-of-line blocking 的直接收益 |
| 两槽同时未 ready 的次数、持续轮数和最大时长 | 固定容量 2 是否仍经常把本核 admission 填满 |
| 每次调度决策中 token-ready poll、queue scan、CAS 数 | Execute 优先的控制开销和重复观察量 |
| 两个 K2 候选的 CAS win/lost 与首选宽限次数 | 第二候选是否造成无效原子竞争 |
| 有 pending token 时继续 Build 的次数及插入等待分布 | 等依赖期间做 Build 是否拖慢或帮助 TensorMap baton |
| replay 结束时 Built/Claimed/DONE 数及 kernel 区域分布 | 工作是否只是从 Submit 搬到 FinalDrain |

诊断计数只用于结构归因。权威性能由无泳道、无诊断计数的端到端构建裁决，
边界固定为全局最早的首个 Submit 起点到全局最晚的 FinalDrain 结束。旧
Submit-only 数字只保留为历史诊断证据，不再决定候选保留或撤销。

## 8. 队列/任务发现机制的候选

### 8.1 Task-indexed execution cell

每个 task id 独占一个 execution cell，本轮不复用。

从调度视角看，这组 cell 本身就是一份**单轮、固定容量、全局可见的
GM task list**：Build owner 把 portable payload 写入对应 task 的固定
cell，批量 `FlushRegion()` 后以 atomic `BUILT` 发布；Execute winner
观察到发布状态后，先对 payload 执行 `InvalidateRegion()`，再加载并执行。
这条跨核发布/取得合同已经由现有 standalone 探针闭合，并不存在额外的
cache coherence 假设。

优点：

- 没有 wrap/ABA；
- 容量在 launch 前按 `task_count` 固定，本轮不需要反压、回收或复用；
- task/state/payload 一一对应，最容易建立内存模型；
- 失败定位和 host 精确检查简单。

该阶段的 cell 只从 `EMPTY` 走到 `DONE`，不因为未来可能改 ring 就预先加 generation 和 device reclaimer。

缺点：

- 最大 payload 若沿用 4,824B，4,352 个 task 需要约 21 MiB；
- executor 如何高效发现 BUILT task 仍需 cursor/bitset；
- 大量固定空白字段会增加 DCCI 和容量。

建议第一版正确性原型采用 task-indexed cell，但只发布紧凑有效前缀，不把它直接宣布为最终结构。

### 8.2 单轮全局 Built list 与真正 Ready queue 的区别

这里不能把 `BUILT` 和 `dependency ready` 混成同一个概念：

```text
Build owner 完成 payload 发布
        -> task 进入全局 Built list

所有 fanin producer 完成
        -> task 才真正可以执行
```

PA 中 SF/PV/UP 的 Build 可能早于其 QK/SF/PV producer kernel 完成。因此，
“Build 完即发布到 GM，winner DCCI 后加载执行”在**内存模型上没有问题**，
当前已经选择 **Built queue + Claim 后检查**：task 一旦 `BUILT` 就可以由
兼容 executor 竞争；CAS winner 取得 payload 后立刻检查 fanin。若未 ready，
task 保持 `CLAIMED` 并占用该核一个 owner-local token；第一项阻塞时允许再抢
第二项。正常路径不构造真正 Ready queue，也不增加反向 fanout、剩余依赖计数
或 producer-driven 入队协议。

“抢到后立刻检查”是调度合同，不只是实现偏好。若把首次检查延迟到下一次
Submit，第二槽可能在本可立即执行的任务已经 ready 时仍被错误闲置；若在检查
第一槽前直接连续抢满两个，又会不必要地扩大 pending 集合。因此顺序固定为：

```text
Claim slot 0 -> acquire -> check slot 0
    ready     -> execute slot 0
    not ready -> Claim slot 1 -> acquire -> check slot 1
                    -> 再次优先检查/执行两个 owner-local token
```

若 Built list 仍采用 `SharedExecCell[task_id]` 固定槽，就不需要 append tail；
immutable plan 已经能推导 task kind 和 K2 候选，executor 只需发现哪些固定
cell 已经 `BUILT`。这里所说的“队列抢占”由 task control 的
`BUILT -> CLAIMED` CAS 完成，不额外建立 append tail，也不增加第二套队列
发布状态。

### 8.3 可复用中央 MPMC ring

优点是结构紧凑、天然提供已构建 task 流。问题是 producer/consumer head、tail 和 slot seq 会成为新的 atomic 热点；A5 同地址并发 atomic 代价可能抵消构建/执行 overlap。

如果采用，必须有 per-slot generation/seq；仅靠全局 head/tail 不能防止 consumer 读取尚未发布或已经复用的 payload。

这些容量、ABA 和回收问题只属于未来的**复用 ring**，不能反向否定
8.1/8.2 的单轮固定槽方案。

### 8.4 Per-builder queue + work stealing

每个 builder 对自己的 SPSC/MPSC 队列发布，executor 扫描或窃取兼容 task。

它可以分散 atomic 热点，但会增加：

- 队列选择和负载均衡；
- task age/fairness；
- 多队列 fanin-not-ready 跳过；
- FinalDrain 判断“全局不再生产且全部队列为空”的难度。

该方案应在 task-indexed 内存模型已经闭合后再评估，不作为第一步。

### 8.5 Ready bitmask/summary

按组发布 AIC/AIV built/ready bitmask 可以减少 executor 全表扫描，但 bitmask 更新本身是共享 atomic。需要区分：

- `built`：payload 可读；
- `dependency ready`：可立即执行；
- `claimed`：已有 executor；
- `done`：可以回收。

不能用一个 bit 同时表达四种状态。

该方案当前不采用。首版明确允许每核两个 token 持有未 ready task，以第二槽
绕过第一槽等待；只有 A5 数据证明“两槽同时长期未 ready”仍是主要瓶颈时，
才重新评估 ready summary。不能在没有这条反证时先增加新的共享 atomic 协议。

## 9. 容量、背压和活性

### 9.1 当前 private slot 容量不能直接沿用

当前每 worker 有 4 个物理 slot，其中 2 个给 BlockWon 预留，普通 kernel 只能使用 2 个。而且该容量检查发生在 Claim winner 已经产生之后：满时是本核 Drain/等待，不是放弃已经赢得的 task。这是 same-core “owner 必须保存到执行完”的局部背压，不是 cross-core 应继承的任务池形态。

task-indexed 首版为每个 task 预留一个 GM cell，它们共同承担全局 backlog；
builder 发布后不需要占有本核执行 slot。当前每个 executor 固定两个紧凑
execution token；它们只是“本核已经领取的执行责任”容量，不是全局 task
存储容量。槽数以后再参数化，首版按 2 验证。

后续把 task-indexed cell 改为有界 ring 时，才需要按 builder 生产率、AIC/AIV kernel 时长和 fanin 等待确定全局容量；不能简单沿用 `96 × 2` 或每核 4-slot 布局。

### 9.2 Builder 不能占住有序 Register 等空槽

危险交错：

```text
task N owner 取得 Register turn
-> 发现执行队列满
-> 等待 executor 释放 slot
-> task N+1.. 永远不能推进 metadata
```

正确做法是让容量等待发生在 Register 外，或为已经取得 turn 的 owner 预留保证可用的发布槽。二者需要通过容量证明，不能依赖“通常会很快释放”。

本节只约束后续有界复用结构；S3 task-indexed cell 不会进入该等待路径。

### 9.3 全局 backlog 与本核 execution token

首版不实现无界 executor-private pending list，只定义两个有界 token：

```text
GM task list:
  SharedExecCell[task 0]  EMPTY/BUILDING/BUILT/CLAIMED/DONE + portable payload
  SharedExecCell[task 1]  EMPTY/BUILDING/BUILT/CLAIMED/DONE + portable payload
  ...

Executor local admission:
  token[0..1]: IDLE -> Binding -> WAITING_FANIN -> ENGINE_INFLIGHT
                                -> COMPLETING -> IDLE

任一 token ready : 优先 Execute，不领取新 task/Build ticket
存在 IDLE token  : 允许竞争一个兼容 BUILT，winner 立即 acquire/check
两个 token 均占用且未 ready : 不再 Claim，可以完成一个 Build ticket
```

- 未领取 task 一直位于 GM `SharedExecCell[task_id]`，状态为 `BUILT`；
- executor 有 IDLE token 时才允许对兼容 `BUILT` 发射 Claim CAS；
- CAS 失败不占 token，也不读取完整 payload；
- CAS 成功后立即 acquire/bind 到具体 token，并马上执行第一次 fanin 检查；
- 未 ready task 保持 `CLAIMED`，由该 token 保存 ready prefix；不回退状态；
- 第一 token 未 ready 且第二 token 空闲时，允许继续领取一个后续 task；
- 第二 token 取得后也必须立即检查；任一 ready 时先执行 ready token；
- CAS 成功后该核对 task 负责到 `DONE`，fatal 路径继续 fail-closed；
- 两个 token 都未 ready 时仍允许本核完成 Build 工作，但每次回到调度点必须
  先复核两个 token，不能连续 Build 而饿死已经 ready 的已领取 task。

两槽模型只能绕过一个未 ready task，不自动获得任意深度的 work stealing。
其活性门槛必须覆盖：第一槽阻塞而第二槽 ready、两个槽均阻塞但 Build 仍推进、
较早 producer 后发布、K2 两候选负载不均，以及生产关闭后全部 token 收口。
若真实 PA 经常出现两个槽同时长期阻塞，再用数据决定把容量参数增大或引入
ready 摘要，不能在首版同时改两套机制。

### 9.4 Alloc 的特殊边界

Alloc 没有 engine kernel，当前 Register 后立即 `CompleteTask()`。第一版应继续由 Build/Completion owner 本地完成，不进入 AIC/AIV execution queue。不能为了统一状态机给 Alloc 增加一次没有业务意义的执行仲裁。

## 10. 泛化问题

PA 只是第一个算子，设计不得固化以下现状：

- ordinary TensorMap entry 永远为 0；
- 每 batch 固定 Alloc/QK/SF/PV/UP；
- UP 固定三个 INOUT；
- task 永远单 lane；
- `function_id` 足以替代真实 `function_address`；
- executor 永远只需 AIC 或 AIV 单核；
- output 都能由 task-indexed SharedOutputRef 表达；
- 最大 fanin、tensor、scalar 的有效前缀与 PA 相同。

当前边界固定为三层：

| 层次 | 允许知道的信息 | 不允许知道的信息 |
| ---- | ---------------- | -------------------- |
| 公共调度协议 | task-id、是否需要 engine、engine class、Build/Execute owner、fanin、completion | `TaskKind`、batch 形状、UP/INOUT 数量 |
| A5 后端策略 | 32 AIC + 64 AIV 物理拓扑、K2 候选、16 组完成归约 | Alloc/QK/SF/PV/UP 的业务含义 |
| 算子适配层 | 随机访问构参、function 路由、payload shape、输出与 INOUT 规则 | 不得让公共 scanner 反向解析这些信息 |

当前 `exec_route` 是公共执行器唯一消费的任务路由，显式编码“是否执行 +
engine class”。PA 的 `batch/encoded_meta` 只由 Build adapter 解码，并在发布
execution cell 前交叉验证两份路由一致。终态完成数同样来自计划显式发布的
`executable_task_count`，不再由 PA 任务拓扑推导。

代码边界也按这三层收敛：`shared_exec_protocol.h` 保存算子无关的 execution
状态、payload 与 route；`a5_exec_policy.h` 只实现当前 A5 单 lane 的
32/64 worker、K2 候选和 owner eligibility；`pa_exec_adapter.h` 只解释 PA
function 与参数。A5 placement 不再读取 `kMaxTasks` 或 PA task kind，计划
容量由 dispatch 解码层独立拒绝。以后接入 Joint、固定 block affinity 或
multicore task 时，应替换/扩展后端 placement，而不是向 PA adapter 增加分支。

首版可以在 standalone PA 上缩小验证范围，但协议字段和失败检查必须为通用算子留出明确扩展点。K2、双 token 和 no-reclaim 都是必须显式选择的调度/生命周期能力，不是 PA 身份带来的默认事实。PA adapter 特例不能进入通用 shared execution runtime。

迁移时不能整份复制 `pa_scheduler_core.h`或 `pa_model.h`。前者同时
包含 PA replay/adapter 与通用 scanner 原型，后者仍有 batch/meta、
SharedOutputRef、UP writer history 和旧 Claim Tournament 状态。正式公共化
应抽取以下四个接口：

1. plan builder：发布 task-id、execution route 与 opaque operator metadata；
2. payload builder：按 task-id 构建 portable tensor/scalar/fanin payload；
3. backend placement/dispatch：在 A5 上选 owner 并发射相应 engine；
4. completion sink：解释 vend/flag 或算子自己的 completion unit。

当前通用原型的明确上限是：稠密 task-id、可随机访问 Build、单
AIC/AIV lane、一 task 一 completion，payload 不超过 32 tensor/16 scalar/
16 fanin。超出这些上限的算子是尚未实现的能力扩展，不能使用
PA task kind 特判冒充支持。

## 11. 失败、取消和终止

首版 task-indexed cell 不复用，因此失败时不尝试在 device 上“撤销后继续跑”。任一部分发布都进入全局 terminal fatal，半构建 cell 保留现场，host 做精确状态校验。至少需要定义以下路径：

| 失败点 | 必须发生的动作 |
| ------ | -------------- |
| cell 进入 BUILDING 后、payload publish 前 | 发布 fatal，cell 保持非 BUILT；其他核不能读 payload |
| TensorMap 已 Register、Build 失败 | metadata 不可回滚；全局 fatal，禁止发布可执行 task |
| payload 已 BUILT、executor 校验失败 | 发布 fatal，禁止执行；首版不复用该 cell |
| executor CLAIMED 后 kernel 失败 | 不得发布正常 completion；FinalDrain 必须能终止 |
| completion 已发布、DONE 发布失败 | 不得重复执行；保留 owner/completion 现场并终止 |

不能用 host 超时作为唯一活性机制。设备侧必须有 terminal fatal，使所有 builder/executor 停止生产并收敛退出。

后续引入 ring 复用时，再单独定义 `CANCELLED`、`DONE -> FREE`、generation 和回收失败；不将这些状态借尸到首版单轮协议。

## 12. 正确性不变量

第一版实现提交前，至少逐项证明：

1. 每个 task 恰好一个 Build owner；
2. 每个 kernel task 恰好一个 Execute owner；
3. 任一 executor 观察到 `BUILT` 时，本 task payload 已完整写回 GM；
4. executor 在读取 payload 前完成自己的 cache 刷新；
5. `BUILT` 后 payload 不再被 builder 或其他候选普通写；
6. payload line 与所有 atomic control line 完全分离；
7. executor 在 DCCI 前已从 atomic state 验证 `engine_class/payload_lines` 上界；
8. builder flush 与 executor invalidate 使用相同的 `payload_lines * 64`，不触及 control 或相邻 cell；
9. acquire 后由 header counts/payload_bytes 重算的范围不超过 atomic 发布范围；
10. executor 读取的 state、task id、kind 和 count 一致；
11. 所有 self-pointer 在最终消费地址空间中正确；
12. fanin 只包含 `< current task` 的 producer；
13. fanin 未 ready 时不执行 kernel；
14. engine 未完成时不发布 task flag；
15. completion vend 来自 task 的构建/分配上下文，不来自错误 executor heap cursor；
16. task completion 只发布一次；
17. task-indexed 首版在整轮 kernel 内不复用 cell；
18. 每个 executor 最多持有两个已领取 task；两个 token 都非 `IDLE` 时不得发射第三次 Claim CAS；
19. Claim winner 必须完成 payload acquire/bind 并立即做第一次 fanin 检查，不能把首次检查推迟到下一轮调度；
20. 第一个 token 未 ready 时允许领取第二项；任一 owner-local token ready 时，Execute 必须优先于新 Claim 和新 Build；
21. 未被空闲 executor 领取的 task 保持 `BUILT`，不因某个忙核而移入无界私有队列；
22. 正常路径先发布 completion 和 cell `DONE`，然后才把对应 execution token 恢复为 `IDLE`；
23. FinalDrain 同时证明 builder 停止生产、所有可执行 cell 全部 DONE、每核两个 execution token 均为 `IDLE`、engine 无 in-flight；每个 worker 只向所属 arrival group 到达一次，并在同一次 FetchAdd 中携带本核已成功发布 DONE 的完成数；非 root 到达后可以结束，固定 root 只有看到 16 组各 6 个到达且完成数合计等于计划发布的 `executable_task_count` 后才允许最后结束；
24. 后续 reuse 版额外证明慢 reader 不会把旧 generation 当新任务；
25. AIC/AIV function 路由严格匹配；
26. fatal 路径不会执行半构建或已取消 payload；
27. ordinary TensorMap 和 INOUT writer history 语义不因执行 owner 改变；
28. `dc_preload` 关闭/开启变体通过同一 oracle，hint 被硬件忽略也不改变正确性；
29. CPU 模型通过不代替 A5 DCCI/atomic 同构门槛。

## 13. 建议的验证顺序

### 贯穿 S0–S6 的观测门槛（不编号）

三条证据链互不混算，不等到 S6 才补观测：

- 无泳道端到端构建：决定候选保留/撤销；现有命令名仍为
  `perf-clock`，但唯一结果是首个 Submit 到 FinalDrain 结束；
- swimlane：检查 Build publish、Exec claim/bind、token
  `WAITING_FANIN`、kernel 和 completion 落点；
- submit-PMU：辅助解释 Scalar/I-cache 变化。

每个功能阶段先过正确性门槛，再用同阶段的端到端构建/泳道
做异常级抽查；只有对应构建的绝对数可在同一证据链内比较，
不把不同 ELF 的绝对时间直接相减。贯穿过程必须逐步补齐以下对照：

1. 现有 private `LocalSlot` 同核执行基线；
2. shared payload 但仍同核执行：单独量发布/取得/重绑税；
3. shared payload、固定跨核映射：量跨核交接税；
4. 受控动态 executor：量 election 成本和负载分布；
5. 扩大 Build owner 候选核：量真正构建负载转移收益；
6. 全 96 Scalar 自由 Build 竞争 + K2 异核 Execute：量当前目标架构的最终端到端收益。

### S0：冻结 ABI 与合同，不接 PA 业务

- 在 `cross_core/` 内定义独立的 task-indexed `SharedExecCell`，`same_core/` 只作对照；
- 首版 packed state 包含 `phase + owner + engine_class + payload_lines + task_id`，phase 只有 `EMPTY/BUILDING/BUILT/CLAIMED/DONE`，不包含 generation/reclaim；
- 定义每 executor 固定两个紧凑 execution token，以及 `IDLE/BINDING/WAITING_FANIN/ENGINE_INFLIGHT/COMPLETING/VEND_PUBLISHED/COMPLETION_PUBLISHED/FAULTED` 本地状态；容量固定为 2，不定义无界 pending 数组；
- 静态断言 control/payload/cell 64B 隔离；
- 列出 portable payload 的有效字段，锁定 `TensorDesc` 和 compact offset，禁止 builder-private pointer；
- 将所有 shared payload ordinary write 收口到唯一 `PackExecPayload()`，将 DCCI/DSB 收口到唯一 publish helper；
- 记录所有 staging POD 大小并审计 CCEC 栈框/溢出，不创建完整 `LocalSlot` 栈镜像；
- 明确每个状态转换的 atomic 以及哪些返回值参与正确性判断；
- 把 publish/acquire 都实现为单一 helper，明确 DCCI 地址范围和 DSB 收口点；
- 预留默认关闭、CPU no-op 的 builder-destination 和 executor-source DCache preload hook，hook 不改 ABI。

### S1：CPU 定向并发门槛

用小型 synthetic payload 覆盖：

- builder 在 payload 一半处暂停，executor 不得读取；
- flush 完成但 BUILT 尚未发布，executor 不得读取；
- 固定单 executor 的 `BUILT -> CLAIMED` 只成功一次；
- 多 executor 同时 CAS 时恰好一个成功；
- 两个 execution token 均非 IDLE 时不得发射第三次 Claim CAS；仍有一个 IDLE token 时允许领取第二项；
- 第一 token 未 ready、第二 token ready 时必须先执行第二项；任一时刻每 executor 最多两个 token 处于占用状态；
- 每次 Claim winner 完成 payload acquire 后必须立即执行首次 fanin 检查，不能延迟到下个 Submit；
- test Ops 观察每个 kernel task 恰好一次完整 payload `FlushRegion`、一次 winner `InvalidateRegion`，所有 loser 为零次；
- private binding 建立后继续推进 fanin/completion，test-only shared-payload read 计数必须保持零；
- executor claim 后延迟，cell 不会被任何 builder 改写；
- 损坏的 `payload_lines/counts/payload_bytes` 在任何越界 DCCI 或 copy 前触发 fatal；
- fanin 未 ready、随后 ready，kernel 不早执行；
- completion 使用 payload 中的 vend，故意设置不同 executor heap cursor 也不影响结果；
- fatal 在 `BUILDING/BUILT/CLAIMED` 各状态都能使所有角色收敛。

CPU 只验证状态机和交错，不用于证明 A5 cache 可见性。generation/reuse 不属于该阶段。

### S2：A5 CCEC 最小跨核 payload 探针

不 include PA scheduler，固定 2–4 个核，分别覆盖：

- AIV0 builder -> AIV1 executor；
- AIV builder -> AIC executor；
- AIC builder -> AIV executor；
- 同 role 不同 block；
- portable payload 1、2、8、68 条 cacheline；
- publisher/consumer 在每个协议边界主动延迟；
- atomic line 邻接 guard、payload 每字段精确值、最终 GM 快照；
- 对比“CAS 返回依赖 -> `InvalidateRegion()`”与“CAS 后额外前置 DSB -> `InvalidateRegion()`”，只保留实测必要的序列。

必须显式关闭/记录编译器自动 Scalar DCCI 和 kernel-end DCCI，避免自动行为掩盖协议缺口。先在 preload hook 关闭时闭合上述全部门槛，再使用相同布局和 oracle 开启 builder/executor hook 做 A/B；reuse generation 属于后续 ring 门槛，不阻塞首个 fresh-cell 结论。

### S3a：Shared task cell，仍由 Build owner 执行

在 standalone 中：

- 保持现有 Build Claim 候选核拓扑；
- Build owner 发布 shared portable payload，再以 executor 身份取得它；
- 只在本核存在 IDLE execution token 时取得，复制 active prefix 并重建 args/context，使用 payload vend 完成；
- 此时 Build/Execute 仍是同一物理核，分开量 forward pack、批量 flush、整体 invalidate 和一次 private copy，不用一个混合端到端值掩盖重复 GM 触碰。

### S3b：固定不同核执行

- 为每个 kernel task 按 task id 和 engine 预先计算 `primary/secondary`
  两个兼容 observer；AIC 候选相邻，AIV secondary 保持 lane 并移动到下一
  物理 block；
- 若 Build owner 不是 primary，则由 primary 执行；若 Build owner 恰好是
  primary，则由 secondary 执行。该规则保持现有 Build Claim 候选拓扑，
  同时确定性保证 `build_owner != execute_owner`；
- QK/PV 只映射 AIC，SF/UP 只映射 AIV，Alloc 仍由 Build owner 本地 completion；
- 每个 worker 在实际 Submit 已知 `(task_id, Kind)` 时独立计算同一对候选；当且仅当自己是 primary/secondary 时，才把该 task 登记到 owner-local 紧凑位图/队列；
- 非候选核不重建历史 PA batch plan，不保存 `batch/offset` 游标，也不读 shared cell control。候选队列是 Scalar owner-local 状态，不需要 DCCI；
- 两个候选才能观察 control，而最终唯一指定 executor 才发射 Claim CAS。这里的两个 observer 不是两个执行竞争者；
- 每个 executor 按 task id 递增处理自己的固定映射序列；第一 token 已领取且未 ready 时允许继续领取下一项到第二 token，两个 token 都占用后不再领取；
- 候选看到 `EMPTY` 时保留队头；看到已带 `build_owner` 的 `BUILDING/BUILT/CLAIMED/DONE` 后，非 target 候选才能丢弃本地记录；只有全局停产后的残留 `EMPTY/BUILDING` 才是 terminal 缺口；
- 先跑 B1 正确性，再跑 B256；host 精确检查 build owner 与 execute owner 不同、task/descriptor/fanin/vend/completion 全部一致；
- FinalDrain 使用“所有 builder 停产 + 每核 scanner/token/engine 排空 + 16 组完成数精确等于计划 kernel 数”收口；唯一 Claim CAS 给出至多一次，完成数只在 DONE 成功后递增，因此该汇总仍证明所有 kernel cell DONE，不再由 root 逐项读取。

S3a 和 S3b 把“shared payload 发布税”与“跨核取得税”分开，同时不混入动态 election。

### S4：加入 exactly-once execution election

本节已完成 K2 实现以及 CPU/CCEC/A5 功能门槛。它仍是
受控双候选，不是通用动态任务池。

首版采用 K2，只改 S3b 中“两个 observer 里固定一个 executor”
的最后一步：

- 复用现有 `primary/secondary` 两个 observer 与 owner-local
  候选位图，不扩散为全核扫描；
- 从 K2 中排除 control 里的 `build_owner`。Build owner 不在 K2
  时得到 2 个 eligible executor，Build owner 在 K2 时得到
  1 个 eligible executor；
- eligible 且至少有一个 token 为 `IDLE` 的 observer 都可以对 `BUILT`
  发射 `BUILT -> CLAIMED(self)` CAS；唯一 winner 读 payload，loser
  不做 payload DCCI；
- observer 对 control 的一次返回型读取同时承担资格判断与 CAS expected；CAS
  本身验证快照是否仍有效，失败后重新观察，不在二者之间插入重复 atomic load；
- CAS 返回 `ExecClaimResult::Lost`，或竞争窗口中返回
  `NotBuilt`，都保留候选位并重新观察 control；后续只在确认
  另一合法候选已进入 `CLAIMED/DONE` 后退役本 task。这些结果
  **不是 fatal**，不得被转换成 execution fatal 或全局 fatal。

候选位的状态合同固定为：

| cell/token 状态 | 动作 |
| --------------- | ---- |
| `EMPTY` | 生产未闭合时保留位图，不 CAS、不读 payload |
| `BUILDING` | 若本 observer 就是 `build_owner`，立即退役其本地候选位；其他 eligible observer 保留位图等待 `BUILT` |
| `BUILT` + 存在空 token | eligible 核发射 CAS；winner 绑定到一个确定空槽并立即检查 fanin，`Lost/NotBuilt` 保留位图并重新观察 |
| 一个 token 忙、另一个空闲 | 先检查已绑定 token；若未 ready，允许扫描并领取第二项 |
| 两个 token 均忙 | 只推进两个已绑定 token，不观察、不 CAS 第三项；新 task 候选位保留 |
| `CLAIMED` / `DONE` | 确认 `execute_owner` 是排除 Build owner 后的 K2 合法候选，再退役本地位；不报 fatal，全局终态仍等待 `DONE` |

TensorMap 严格有序插入、`deps_prepared` 发布和 fanin 冻结都在
Build owner 发布 `BUILT` 之前完成。S4 不修改这条串行链；
`EMPTY/BUILDING` observer 也不得提前读 payload。

`FinalDrain` 继续使用已有全局合同：所有 builder 停产、所有
kernel cell 为 `DONE`、每核两个 token 均为 `IDLE`、engine 无 in-flight。
设备侧通过 16 组 packed arrival 的完成数汇总证明该终态，host 返回后仍逐
cell 校验；root 不再在性能边界内做全表 Atomic scan。
生产闭合后的 `EMPTY/BUILDING` 是发布缺口；`BUILT` 必须继续
election；`CLAIMED` 只能等待 `DONE`，不能因 loser 本地位图已清理
就提前结束。
终态 validator 接受 K2 中任一非 Build owner 的合法 Execute owner，
不再要求唯一预定 owner；payload、completion、token 与 FinalDrain
仍使用原有完整断言。

K3 留作后续对照：预路由 3 个兼容 observer，排除 Build owner
后再固定 2 个 eligible 竞争者，能保证始终有两个竞争者；
代价是每 task 的 control observer 从 2 增到 3，观察量增加
50%。所以 K3 不与 K2 首次实现混做，当前尚未实现。
K2/K3 都不代表通用动态任务池，当前也不宣称它们有任何性能收益。

CPU 已有的动态证据为：

- 完整 CPU build 和全部公共/隔离门槛 PASS；
- CPU B1 semantic/postprocess PASS：5 个 task、4 个 kernel；
- CPU B256 semantic/postprocess PASS：1280 个 task、1024 个 kernel，动态
  合法 owner、payload、completion terminal、token 和 drain 等断言
  全部通过。

CPU 耗时不用于推导 A5 性能，上述结果也不代替 A5 atomic/DCCI
可见性与动态竞争门槛。A5 证据已进一步闭合：B1 full-swimlane
Submit 为 `280.683 us`；B256 full-swimlane Submit 为 `27.476 ms`，
1280 个 task、1024 个 kernel 和全部 terminal/drain 断言 PASS，trace drop
为 0。Build owner 不在 K2 的 956 个 task 中，primary/secondary 实际
分别胜出 319/637 次；Build owner 命中 K2 时则均由另一候选执行。

B1 偶发长尾不得归因于 S4：fixed-owner 父版同样在同机跑出
`36.922 s`。已保存的 S4 `805.959 ms` 泳道又将第一个异常定位在
task 3 `Materialize` 的第一个 heap atomic 之前，之后才被 TensorMap
严格顺序链和 FinalDrain 无退避轮询放大。该公平性/退避问题留作
独立性能课题，不改变 S4 已闭合的状态机结论。

### S5：独立扩大 Build owner 候选核

S5 拆成两个正交阶段，先证明可移植性，再扩大竞争人口：

1. **S5a 跨角色 Build 门槛**：Alloc 保持 96/G8；QK/PV 改由 64 个
   AIV/G8 Build、仍由 AIC K2 Execute；SF/UP 改由 32 个 AIC/G6 Build、
   仍由 AIV K2 Execute。B256 的 local/root/总物理 Claim CAS 仍精确为
   `73,728 / 9,216 / 82,944`，因此本阶段不会把 payload 可移植性与新增
   atomic 竞争混算。Build owner 位于 engine 对侧，也不会占用 K2 中的
   任一席位，两个 execute candidate 都保留竞争资格。
2. **S5b 任意 Scalar Build**：把 Alloc/QK/SF/PV/UP 的 Build 候选统一扩到
   96/G8。B256 的 local/root/总物理 CAS 精确变为
   `122,880 / 10,240 / 133,120`，相对 S5a 总物理 CAS 增加约 60.5%；
   该代价必须单独和更好的到达式负载均衡比较。

两阶段共同保持：Materialize/Register/Fanin/portable Build 不携带执行
核私有状态；只有 TensorMap metadata 通过 `deps_prepared` 按 task-id
严格发布，Build payload 和 kernel 执行不进入该串行链；host 必须独立
复算 Build 角色、K2 和终态，不能调用设备 helper 形成同错 oracle。

S5a 已通过完整 CPU build、CPU B1/B256 real-compute、CCEC 双入口编译和
A5 B1/B256。逐 kernel 证明 QK/PV 的 Build owner 位于 AIV、SF/UP 的
Build owner 位于 AIC，Execute owner 仍匹配目标 engine、属于 host 独立
复算的 K2 且不同于 Build owner。fanin payload 发布量也随 Build 角色
反转，而 ready load 仍落在 Execute 角色。A5 B256 普通/完整泳道 Submit
分别为 `27.143 ms / 27.301 ms`，完整泳道中 primary/secondary 分别执行
`363/661` 个 kernel，非法 owner 为 0、trace drop 为 0。该单轮数值只用于
排除倍数级结构性异常，不宣称相对 S4 有稳定性能收益。

S5b 已闭合 CPU/CCEC/A5：五类 task 的 96 个 Scalar 都实际进入
Build Claim，Build owner 可以是 AIC 或 AIV；Execute owner 仍必须匹配
engine、位于 host 独立复算的 K2，且不得与 Build owner 相同。
CPU B1/B256 real-compute 及全部独立门槛 PASS；B256 逻辑 Claim
精确为 `122,880`，隔离 Tournament 门槛精确闭合
`122,880 / 10,240 / 133,120`。CPU B256 全 atomic trace 会因
FinalDrain 轮询记录耗尽通用 trace 容量，因此不用它代替 A5 的
物理 CAS 与 DCCI 动态证据。A5 B256 普通/完整泳道 Submit 分别为
`27.347 ms / 28.250 ms`；完整泳道内 local/root/总物理 Claim CAS
为 `122,880 / 10,240 / 133,120`，K2 primary/secondary 分别执行
`375/649` 个 kernel，非法 owner 为 0。相对 S5a 单样本未观察到性能收益；
差异只用于说明没有倍数级异常，不宣称稳定回退。

### S6：性能评估与容量/复用优化

- 量化终点为 B256、real-compute、96 Scalar、严格插入链与 K2 异核执行
  保持不变时，`submit_to_final_drain <= 1.0 ms`；不得用缩减候选人口或
  kernel/Scalar overlap 换取该数字；
- 使用贯穿 S0–S5 累积的三条证据链，对 publication、
  handoff、election 和 Build 负载转移做同口径收益审计；
- S6.1 已用 owner-local token/candidate 门控消除无本核执行工作时的
  122880 次完整 EfDrain 入口，把十轮 perf-clock 中位从 27.496 ms
  降至 9.078 ms；Claim CAS、严格插入和执行语义保持不变；
- 当前 S6 调度候选把 owner-local token 固定从 1 扩为 2：每次 Claim winner
  必须立即检查依赖；第一项未 ready 时才领取第二项，任一 ready token 始终
  排在新 Claim 和 Build ticket 之前；
- 先决定 task-indexed 方案是否已经满足性能和容量，不为了
  预设最终架构就提前引入队列；
- 只有 task-indexed 内存模型、受控动态 election 和端到端
  收益都有证据后，才评估紧凑 ring/per-builder queue；
- 需要复用时才引入 generation、ABA、容量背压和 device reclaim
  门槛，并将每个新合同作为独立正确性阶段验证。

### Simpler 迁移门槛（不编号）

迁移不是 S4–S6 中的一个功能阶段。standalone 的对应协议通过
CPU/CCEC 正确性、A5 动态和贯穿观测门槛后，再单独评估是否
迁移到 Simpler 真实路径。迁移时必须逐项对照生产 TensorDesc/
function ABI、heap/completion 合同、构建宏边界和现有 PA 正确性，
不因 standalone 已通过就直接默认真实路径等价。

## 14. 性能上最危险的成本

### 14.1 直接发布完整 `LocalSlot`

`LocalSlot` 为 4,824B，向上覆盖约 76 条 cacheline。若每个 1,024 个 kernel task 都由 builder flush、executor invalidate 一次，仅 payload DCCI 就接近：

```text
1,024 × 76 × 2 = 155,648 cacheline operations
```

这还没有计算复制、队列 atomic 和 I-cache 膨胀，极可能吞掉跨核构建/执行分离的潜在收益。因此不能为了尽快跨核，直接把现有最大结构逐 task 全量发布。

### 14.2 紧凑 payload 与重复 lookup 的权衡

可比较：

- 发布 active TensorDesc/scalar/fanin 的连续紧凑 payload；
- 只发布 SharedOutputRef，executor 再读取 shared_outputs；
- 按 producer 批量发布 descriptor block；
- shared payload 只保存 offsets，executor 重建 args。

紧凑 payload 增加 pack/unpack，引用模式增加 atomic/DCCI/lookup。需要用真实 task 形状分别量发布 line 数、executor load、Build 时间和端到端，不能只看结构体大小。

### 14.3 新 execution election 不能重演 Claim 热点

当前主要瓶颈已经是多核同地址返回型 atomic。若每个 built task 再让 32/64 个 executor 竞争一个中央状态，会新增一套 Claim 级热点。

第一版固定映射用于闭合内存模型；动态版本不能默认让 32/64 核 CAS 同一地址，
也不能建立无界 pending 列表。当前只允许 K2 候选和每核两个 token；若两槽
同时等待仍是瓶颈，再依据 A5 数据比较扩大有界容量与 ready summary。

### 14.4 观测代码不能进入协议 cacheline

泳道记录、计数器和 debug checksum 不得放进 execution slot control/payload line。观察动作应在取得时间端点后写每核独占 trace buffer，不能为了显示状态而给每次 poll 新增 raw record。

## 15. 当前推荐的第一版原型

当前 standalone 首版按以下顺序推进：

1. 使用 task-indexed、单轮不复用的 `SharedExecCell[task_id]`，它就是 GM 中的全局待执行 task list/backlog；
2. packed state 首版携带 `phase/owner/engine_class/payload_lines/task_id`，phase 为 `EMPTY/BUILDING/BUILT/CLAIMED/DONE`，不加 generation、queue 和 device reclaim；
3. 每 cell 一个独占 atomic state line，payload 独立 64B 对齐；
4. payload 只保存有效 tensor/scalar/fanin、task id 和 Materialize 后冻结的 completion vend，不保存 self-pointer；
5. 小型 staging 只用寄存器/栈/owner-private 状态；数量全部确定后才一次 forward pack 到 GM，不创建完整 payload 栈镜像；
6. 全部 96 Scalar 继续通过中央 ticket 恰好一次 Build；Build owner 对整个 payload 仅调用一次 `FlushRegion()`，随后才发布 BUILT/engine_class/payload_lines；
7. 每个 executor 当前固定两个紧凑 execution token；两个槽均占用时不发射第三次 Claim CAS，不建立无界 pending list；
8. immutable plan 继续提供 task kind 和 K2 候选，本核按单调候选序列观察固定 task-indexed Built queue；
9. 存在空 token 时直接竞争 `BUILT -> CLAIMED`；CAS loser 不读 payload，winner 绑定到确定空槽；
10. winner 校验 payload_lines、对完整 payload 调用一次 `InvalidateRegion()`、一次 forward copy active prefix 到该 token，之后不再读 shared payload；
11. payload acquire 完成后立刻检查该 token 的 fanin；ready 就立即执行，未 ready 且第二槽空闲才继续领取下一项；
12. 每次调度点先检查两个已占用 token；任一 ready 都先于新 Claim 和 Build ticket，两个都未 ready 时才完成一个 Build task；
13. completion 使用 token payload 中的 vend；发布 DONE 后才释放对应 token，FinalDrain 要求两槽均 IDLE；
14. DCache preload 默认关闭；只有双 token 正确性和基线性能闭合后才做独立 A/B；
15. 槽数以后再参数化；当前先固定 2，只有两槽同时等待的 A5 数据支持时才比较其他容量或 ready summary。

这一版不是最终高性能形态。它的价值是把三个未知量拆开：

```text
跨核 payload 内存模型
≠ 动态执行仲裁
≠ Build owner 候选拓扑
```

只有前一项闭合后，后两项的性能结果才有解释价值。当前三项均已
分阶段闭合；S6 只在这一架构内优化，不再引入第四套 engine/Scalar
协程机制。

上面的 1--15 是 S0--S5 首版落地顺序，不再代表 S6 当前调度策略。其中
“每核只有一个 token，未 ready 后不能领取第二项”已经被最新泳道中的
FinalDrain backlog 反证为性能瓶颈。S6 下一候选只替换本核 admission 与调度
优先级：保持 task-indexed Built queue、portable payload、K2 exactly-once、
同步 kernel 和 completion/FinalDrain 内存合同不变，改为 6.2/7.1/9.3 定义的
Claim 后立即检查、每核两个 token、owner-local ready 优先模型。

## 16. 尚未决定的问题

1. S5b 已选择所有 96 个 Scalar 作为 Build 候选；尚需由 A5 证据决定负载均衡收益能否覆盖额外 60.5% 物理 Claim CAS。
2. 最终版 executor 是直接 dispatch shared payload，还是复制 active prefix 到紧凑 token？首版继续复制，并把 token 数从 1 固定扩为 2；以后才参数化。
3. portable payload 的最小通用 ABI 是什么，怎样兼容真实 function address 和 joint task？
4. 更通用的 heap 协议中 completion vend 应如何表达？首版已决定在 Materialize reservation 成功后冻结。
5. 两个 token 同时未 ready 在 AIC/AIV 上的发生率和持续时间是否足以要求继续扩容？
6. 若容量 2 仍不够，应先增加可配置 token 数，还是再引入 completion-driven ready summary？
7. task-indexed cell 的内存开销是否可接受到哪个阶段？
8. 动态 executor election 如何不新增 Claim 等级的同地址 atomic 热点？
9. AIC 与 AIV 是否使用完全独立的执行发现结构？
10. 动态队列版 FinalDrain 如何证明 builder 停产、queue 为空、每核全部 token 均为 IDLE 和 engine in-flight 为零？task-indexed 版先用全 cell 终态计数收口。
11. BlockWon/multicore task 的 portable context 与执行 owner 如何表达？
12. shared execution payload 是否能复用生产 RingSlot ABI，还是需要独立中间 ABI？

## 17. 更新记录

### 2026-08-02：完成 S4 K2 动态 Execute election 的 CPU 门槛

- S4 首版决定复用 S3b 的两个 observer 和 owner-local 位图，
  排除 Build owner 后由剩余 1 或 2 个 eligible 核竞争 `BUILT` cell；
- 明确正常 CAS `Lost` 不是 fatal，并固定 `EMPTY/BUILDING`、
  token 忙、`CLAIMED/DONE` 与 `FinalDrain` 的候选位生命周期；
- TensorMap 有序插入仍在 `BUILT` 之前完成，不被 execution
  election 改写；
- K3 可保证排除 Build owner 后始终有两个竞争者，但 control
  observer 由 2 增至 3，观察量增加 50%，留作后续对照。
- K2 代码已实现，确定性 CPU 用例覆盖 Build owner 跳过、
  primary/secondary 分别获胜、CAS `Lost/NotBuilt` 重新观察、
  忙 token 不接触新 cell、合法动态终态和 FinalDrain；
- 完整 CPU build 全门槛 PASS；CPU B1 为 5 task/4 kernel，B256 为
  1280 task/1024 kernel，两者 semantic/postprocess 均 PASS。

A5 尚未运行；不宣称通用动态池，也不用 CPU 耗时推导 A5 性能收益。

### 2026-08-02：S3b B256 full-swimlane 与 host 终态口径闭合

- 固定两候选异核执行的 B256 full-swimlane 已完整通过：1280 个 Build、
  1024 个异核 kernel、四类 kernel 各 256，fanin/payload/vend/route、96 核
  execution drain 和 fatal 均闭合；Submit 为 **27127.645 us**，trace 无丢失。
- A5 的 owner-local execution token 不做 kernel-end DCCI，host D2H 对 token
  本体只能作为诊断快照。权威终态保持为设备侧逐核 token/scanner 自检、
  全局 execution drain 和 bypass 发布的 `final_occupied`；不向设备热路径
  增加只为 host 观察服务的 DCCI/DSB。
- `Validate()` 现在要求 runner 显式声明 raw token 观察权威性：CPU coherent
  路径继续严格断言，CCEC 只呈现 `RESET/NON_FINAL`。完整证据和泳道路径记录在
  [PA调度器分离版实现过程](cross_core/PA调度器分离版实现过程.md)。

### 2026-08-02：S3a 泳道暴露全核扫描热点，修正 S3b 发现合同

- S3a B256 full-swimlane 的 122880 个 EfDrain 中位数为
  **14.962 us**、p95 为 **20.549 us**、最大为 **217.371 us**；同口径
  same-core 最优版本分别只有 0.025 us、0.791 us 和 70.860 us。
- S3a EfDrain 占 Submit 聚合核时间的 **89.683%**，其中 control 占
  EfDrain 的 **99.912%**，kernel 只占 0.088%。这不是 kernel 等待或偶然
  尾部，而是发现协议本身主导了执行时间。
- 根因是旧 S3a scanner 让 96 个 worker 对每个 closed task 都调用一次
  `Ops::Load(exec_cell.control)`；A5 CCEC 将该 int64 load 实现为返回型恒等
  `atomicMax(INT64_MIN)`，于是每 task 形成同地址的最多 96 核竞争。
- 因此撤销“所有同 role executor 逐 task 读取 control，再从 build_owner
  判断归属”的 S3b 计划。新合同改为 task-id 预路由的 primary/secondary：
  非候选零 control load，最多两个候选观察，且只有由 build_owner 唯一确定的
  target 发射 Claim CAS。这样不减少现有 Build Claim 候选，也不允许同核执行。
- S3a 新 execution control/DCCI 当前尚未形成独立 raw 事件，导致长
  EfDrain 在泳道中表现为空白 control 区。S3b 取证必须补齐这些操作的明确
  atomic/DCCI 事件，不能继续只依赖父区间推断。

### 2026-08-02：撤销“全核重建 plan 后再跳过”的 S3b 过程态

- 首个两候选实现仍让 96 个 worker 各自维护
  `task/batch/offset` 三元组，先重建当前 PA batch plan，再决定是否读
  cell。它虽然去掉了非候选的 GM atomic load，却没有真正去掉 96 份任务
  发现状态。
- A5 B1 实测出现了结构性反证：一轮语义全通过但 Submit
  达 **849245.117 us**；紧接着一轮由 worker 63 在 task 2 发布
  `invalid-built-control`，而 task 2 的合法 AIV observer 应为 34/36，cell 尚为
  `EMPTY`。首错发生在 cell load 之前，证明本地 plan 游标/解析已不可作为
  可靠的候选发现根据。
- 两候选映射规则本身仍保留；撤销的是“每核事后重建 plan”的实现。
  新合同直接复用当次 Submit 已经持有的 `(task_id, Kind)`，只在两个候选
  核的 owner-local 紧凑位图/队列中登记。这一改动先经 CPU 状态机闭合，
  再重新进入 A5 B1；现有异常二进制不进入 B256。

### 2026-08-02：S0/S1 落地并进入 S2 动态门槛

- `cross_core` 当时形成独立 portable payload、task-indexed cell、每 executor 单 token 和 global fatal 首版；单 token 已在 2026-08-03 的 S6 设计中被两个 token 取代；
- packed state 增加 task id；删除无独立生产语义的 `output_task_id`，header 高 32 位改为强制零保留位；
- CPU 确定性交错、100 次重复、ASan/UBSan、TSan 已通过，详细证据见 [PA调度器分离版实现过程](cross_core/PA调度器分离版实现过程.md)；
- AIC/AIV CCEC 编译及 Claim CAS 返回依赖、DCCI/DSB 顺序的自动 IR 门槛已通过；A5 最小探针完成 100 × 32 case，全量覆盖四种跨核方向、四种 payload 大小和两种 acquire 路径，详细证据见实现过程文档；
- fresh task-indexed cell 上 minimal 返回依赖路径已经闭合，默认不增加前置 DSB；该结论不外推到未来 generation/ring reuse，也不代替完整 PA 验证。

### 2026-08-01：复核 `same_core`/`cross_core` 内存合同

- 纠正“`same_core` 没有跨核内存模型”的隐含误解：SharedOutput、TensorMap 有序插入和 task completion 本来就是跨核合同；
- 将 `cross_core` 的新证明面收敛为 execution payload 发布/取得、portable binding、completion-vend 交接、唯一 Execute owner 和新 FinalDrain 生命周期；
- 纠正 atomic 布局表述：硬约束是 atomic-only control 不与 ordinary+DCCI payload 共线，不是所有 atomic 永久一字一行；
- 依据现有 CCEC helper 修正 consumer 顺序：`InvalidateRegion()` 已内含尾部 DSB 和 compiler barrier，前置 DSB 改为 S2 对照候选，不预设必须；
- 将 task-indexed 首版与 ring 复用版拆开：首版无 generation/reclaim/ABA，避免不必要的状态和 atomic；
- 补齐 execution payload 逻辑布局，将 `engine_class/payload_lines` 纳入 atomic `BUILT` 发布，解决 executor 不能在 invalidate 前读 header 的前置依赖；
- 把 same-core 已有的“owner-local staging -> 单向 GM pack -> 末尾一次批量 DCCI”提升为 cross-core 默认性能合同，并增加 flush/invalidate 次数、binding 后零回读、栈框/溢出等专项门槛；
- 当时去除“cross-core 继续每核维护多个 private pending slot”的默认：`SharedExecCell[]` 承担全局 backlog，每 executor 首版只有一个紧凑 execution token，忙时不 Claim；该历史合同已被 2026-08-03 的两槽 Claim-first 模型取代；
- 将动态任务池与受控 exactly-once election 分开：前者还必须闭合 ready-only 发现或等价活性证明，防止全部 token 被未 ready task 占满；
- 预留 builder destination、executor source 和 executor-private destination 三类编译期 DCache preload hook，明确它们默认关闭、不改 ABI、不代替 DCCI/DSB/atomic；
- 将验证拆为 same-owner shared publication、fixed different-core handoff、dynamic Execute election 和 Build 候选核扩大四个正交阶段。

### 2026-08-01：建立问题边界

- 明确 Build owner、Execute owner、Completion owner 三种角色；
- 确认当前 `LocalSlot` 的提前 `built`、普通状态共线、非 64B 步长、自引用 args、mutable fanin 和 executor heap cursor 都不能直接跨核复用；
- 以现有 A5 DCCI/atomic 探针为依据，给出 payload publish/acquire/reclaim 基础合同；
- 建议先做 task-indexed、固定跨核映射，分离验证内存模型与动态执行仲裁；
- 当前没有实现和运行结果，本文所有新结构仍为待验证候选。

### 2026-08-02：K2 正常快路径采用主候选优先、备选有限兜底

K2 的架构含义保持为“两个核都具备合法执行资格”，不是把 task 静态分配给
一个固定 executor。为减少两个候选同时争抢同一个 `BUILT -> CLAIMED` CAS，
正常快路径增加稳定首选：优先选择与 Build owner 不同的 primary；如果它与
Build owner 重合，则首选 secondary。另一个候选第一次看到 `BUILT` 时让出
一个本地 progress 机会，随后仍可接管；FinalDrain 不执行该让出。

这个选择保留了跨核执行和候选故障/繁忙时的有限前进性，同时没有增加共享
状态、发布步骤或 cache-coherence 假设。本地宽限计数复用既有紧凑统计空间，
候选 cursor 前进即清零。CPU 交错、CCEC 和 A5 B256 均已闭合；十轮
perf-clock 中位从 `3.678558 ms` 降至 `3.637995 ms`，七对交错 A/B 中 6 对
胜出，且 FinalDrain 未增加。该机制因此作为当前 K2 执行仲裁基线保留。

最终性能验收目标已经调整为 B256 perf-clock 稳定不高于 `1.0 ms`。主候选
优先只贡献约 1% 的已验证收益，后续仍需从 Claim、发布/取得以及 execution
progress 的结构性原子和数据访问中继续消减，不能把候选优先视为目标闭合。

### 2026-08-03：解除任意 Build owner 对本核完整 replay 历史的依赖

shared `PaOutputHandle` 是确定性的 `(producer_task_id, output_slot)`，不是
winner 私有 TensorDesc 指针。基于这一合同，已增加随机访问构参 helper：从
权威 task 身份重建 batch/group 动态状态，并只恢复当前 callback 实际消费的
前驱 handle。混合 G0/G1/G2/G4 的 41 个 task 与原顺序 replay 逐字段一致，
所有 active producer 都严格小于当前 task id。

这只证明“单个任意 Scalar 无须保存前序 `AcceptTaskOutputs()` 状态也能正确
构参”，尚未把生产改为中央 task queue。完整发放协议还必须同时提供：

1. 全局连续、不可变且由 host 独立校验的 batch/task plan；
2. exactly-once Build ticket，保持 96 个 Scalar 都能竞争新工作；
3. 替代每核 Close 前缀与 candidate 位图的 execution 可见性合同；
4. builder 停产、全部 task 已 Build、K2 token 全空和 kernel 全完成的终态闭合。

因此该门槛不会被解释成已有性能收益，也不会直接删除当前 replay。下一阶段先
在 CPU 协议用例闭合 task ticket、严格 TensorMap commit chain 和全局发布前沿，
再决定是否接入 CCEC。

### 2026-08-03：中央 ticket 候选先完成 CPU 协议证明

新增的独立门槛把候选协议拆成四条明确边界：

```text
immutable host flat plan
        │
        ├─ global monotonic ticket ──> exactly-one Build owner/task
        │
        ├─ random-access args ───────> producer 只允许落在 [0, N)
        │
        ├─ deps_prepared[N-1] ──────> 严格串行 metadata commit
        │                              发布 N 后立即释放插入 baton
        │
        └─ all workers retired ─────> production_closed
```

CPU B256 用 96 个线程连续跑十轮，证明 1280 个 task 恰好一次发放、插入严格
保序、Build 可以在插入完成后乱序推进、最终停产只在全部 worker 退出后闭合。
测试还强制 task 0 先交出插入 baton、后完成 Build，并要求至少 16 个后继先
完成，避免只凭偶然线程时序得到“可并发”的假阳性。

该候选把现有 B256 G8 Claim 的 133120 次物理 CAS，变为 1376 次 ticket
`FetchAdd`。这只是静态调用量和 CPU 正确性证据；A5 对同一地址的返回型 atomic
可能近似串行，尚不能判断其真实耗时。下一步先以独立 A5 微基准比较两种地址
拓扑，再选择中央或分片发放。CCEC Submit、K2 candidate 可见性与 FinalDrain
均未修改，因此此处没有生产性能结论。

### 2026-08-03：A5 原子拓扑允许中央 ticket 进入生产候选

独立 mixed CCEC 探针已按真实 32 AIC + 64 AIV、B256/1280 task 对比当前
G8 per-task CAS 与中央 ticket。11 轮中位 device span 为
`556.900 us` 对 `253.851 us`；empty 为 `16.535 us`，三种模式全部语义
检查通过。中央路径虽然把物理调用从 133120 降到 1376，但仍消耗约
0.25 ms，不能视作免费全局 cursor。

因此下一步可以进入生产 CCEC 小步接入，但验收口径仍是完整 B1/B256 终态与
perf-clock A/B，不从微基准外推收益。接入还必须显式替换“每核 replay 才能
形成连续 Close 前缀和 owner-local K2 candidate 位”的旧合同；不能只换 Claim
函数便宣称完成动态发放。

### 2026-08-03：K2 发现不再依赖每核完整 Close 历史

中央 ticket 不能继续沿用 `exec_candidate_bitmap`：该位图由每个 worker 在
顺序 Close 自己的全量 replay 时建立，而新的 Build owner 只会看见自己取得的
task。替代合同已经先在现有 CPU 发放门槛中闭合：

```text
immutable flat plan + owner-local candidate cursor
        │
        ├─ Alloc / wrong-engine task ──> 直接越过
        ├─ relevant EMPTY/BUILDING ────> production open 时保留队头
        └─ relevant BUILT ─────────────> 固定 K2 唯一 Execute
                                         Build owner 不执行本 task
```

十轮 B256 中 1024 个 kernel task 全部 exactly-once 执行，执行者均为合法 K2
候选且不同于 Build owner；不需要每核 replay 1280 次 Submit，也没有新增共享
candidate 位图写。下一版 device 计划只携带构参和判断 engine 所需的紧凑身份，
不携带 TensorDesc 或 worker 私有地址。生产关闭仍以“全部 ticket owner 已完成
Build”作为 EMPTY 可判缺口的唯一边界。

### 2026-08-03：紧凑 device plan 的内存布局

standalone state 尾部新增 `SharedBuildDispatchState`，不移动此前已验证的共享
对象。布局固定为：

```text
cache line 0 : next_task atomic（device 可写）
cache line 1 : task_count / batch_count / executable_task_count（launch 前写定）
remaining    : 4 B × task_count 的只读 identity
```

每条 identity 保存 PA Build adapter 所需的 `batch/encoded_meta`，以及
公共执行器所需的 `exec_route`；task id 是下标，记录仍为 4 B。
`batch_start` 从 task id 与 kind/group 恢复。计划不包含任何 descriptor、args、
payload 或 owner 地址。host 从独立 task planner 填充，Build adapter 解码 PA identity，
execution scanner 只解码 `exec_route`。编译上限下 sidecar 仍为 17,536 B，
B256 实际有效计划仅 5 KiB。

混合 G0/G1/G2/G4 的 host 发布和 device 解码门槛已通过，CPU 全套与 CCEC
perf-clock 构建也已通过。Submit 尚未切换到该计划，A5 尚未运行。

### 2026-08-03：执行发现切换到计划，发放尚未切换

K2 scanner 现在以 immutable plan 的 `exec_route` 为任务资格依据：
metadata-only task 与错 engine task 直接越过，相关 `EMPTY/BUILDING`
保留队头，`BUILT` 后继续使用原 K2 仲裁。scanner 不解码 PA 的
`batch/TaskKind/group`。旧 `exec_candidate_bitmap` 不再决定 scanner 是否读取 cell；它只因旧
96 份 replay 尚未删除而暂时保留登记，并在游标推进时清理。

CPU 全套、CCEC 构建、A5 B1/B256 均通过。五对同窗 B256 中位从旧路径
`3.659993 ms` 变为 `3.666555 ms`，差 `+0.179%`，处于运行波动量级。由此可以
进入 central ticket 接入；下一步必须同时删除 replay Close/位图登记，不能把
本阶段过渡形态当成最终实现。

### 2026-08-03：确定两槽 Claim-first Execute 调度模型

最新 B256 full-swimlane 显示 1024 个 kernel 中只有 158 个位于 EfDrain、
52 个位于 OrchestrationTail，814 个滞留 FinalDrain。该证据足以否定“每次
领取 Build ticket 前只推进一个单 token”是最终模型。

设计合同先修正为：单轮 `SharedExecCell[task_id]` 本身就是固定容量的全局
Built list，不存在复用 ring 的容量反压、ABA 和回收问题；Build payload
`FlushRegion -> BUILT`、Execute winner `InvalidateRegion -> load` 的跨核内存
合同也已经闭合。当前不构造真正 Ready queue，也不在 Claim 前预读 fanin。
每核固定两个 execution token：抢到 `BUILT` 后立即 acquire 并检查依赖；第一
项未 ready 时允许再抢第二项，任一 owner-local task ready 都必须优先执行。
两个 token 均未 ready 时，本核仍可完成一个 Build ticket，返回调度点后先
复核已领取 task。

该模型已经接入 standalone 正式路径。CPU 定向交错证明了“第一槽 blocked、
第二槽 ready”、两槽 pending 时停止第三次 Claim 但仍允许完成一个 Build ticket，
以及两个 token 的 FinalDrain/错误终态收口；完整 CPU、CCEC 和 A5 B1/B256
均通过。状态只增加 `96 × 4928 B = 473088 B` 的第二 token，没有新增逐次 raw
记录、共享 ready 状态或跨核普通发布。

无泳道性能构建只保留一个权威区间：

```text
global min(first Submit.begin) -> global max(FinalDrain.end)
```

设备端复用 `finish_cycle` 保存本核 FinalDrain 排空后的终点；性能构建中的
`submit_end` 固定为 0，不再读取或输出 Submit-only 边界。每核只读取首个
Submit 起点和 FinalDrain 终点两个时钟，不新增结果字段、泳道事件或逐 task
记录。

严格同底座的三对交错 A5 B256 过渡性对照如下。表中的 Submit 行来自收敛
口径前的诊断版本，只用于解释工作从 FinalDrain 前移，不再是当前性能输出：

| 指标 | 单 token | 双 token | 变化 |
|---|---:|---:|---:|
| Submit 中位 | 1.871997 ms | 2.455617 ms | +31.18% |
| 完整周期中位 | 9.345107 ms | 5.880792 ms | -37.07% |
| EfDrain kernel 中位 | 208 | 432 | +224 |
| FinalDrain kernel 中位 | 816 | 592 | -224 |
| fanin load 中位 | 16472 | 13361 | -18.89% |

因此双 token 的保留依据不是原 Submit 指标变快，而是 224 个 kernel 从尾部
前移、完整周期下降约 3.464 ms。若只看 Submit，会把预期中的工作前移误判为
回退；若只看 FinalDrain，则又会遗漏前移所付出的控制和 kernel 时间。

### 2026-08-04：收敛有序链与 terminal 观察合同

重新按共享副作用逐项核对后，当前 cross-core 设计只要求 **Register 的
TensorMap metadata 插入**沿 task id 严格保序：task N 完成 metadata/history
写入并发布 `deps_prepared[N]` 后，N+1 才能取得插入资格。该有序链不延伸到
Fanin、portable payload Build 或 kernel Execute。

因此 WinnerBuild 只保留单 task 内部的发布顺序：

```text
EMPTY -> BUILDING CAS
-> pack portable payload
-> payload DCCI clean-out + DSB
-> BUILDING -> BUILT CAS
```

已经取得 Build ticket、已经 Claim 的 execution token 和已经开始的 completion
均视为合法工作单元。并发出现其他 task 的 terminal 错误时，这些工作单元允许
完成自身发布，不能为了“更早停止”在其每个内部边界重复读取同一全局 fatal
cache line。否则既不加强 TensorMap 顺序，也会把 96 核对同一地址的返回型
Atomic 竞争带进正常热路径。

错误控制分为两个职责，不再互相充当第二条停止线：

- `exec_fatal` 只由实际发现 execution 协议错误的核首写精确原因，供 host
  定位；成功路径不读取它；
- scheduler global fatal 是唯一跨 worker 停产条件，由实际错误发布者同步
  设置；其他 worker 在领取下一 Build ticket 前的调度边界观察它；
- 已通过调度边界并成功领取 ticket 的 winner 不在
  `FinishSharedWinnerSubmitBody()` 入口重复读取 global fatal，而是完成该
  合法工作单元；该读取不是 TensorMap 严格插入或 payload 发布的
  正确性边界；
- 本核直接发现 payload/control/DCCI/completion 错误时仍立即发布两类信息并
  返回，不等待下一调度边界；
- FinalDrain 必须最终观察 global fatal、把本核尚未复位的 token 收敛为
  `FAULTED`，并继续参与退出屏障，不能把 host 超时当设备终止协议。

S6.26 已按该合同通过 CPU、CCEC 和 A5 B256，并把完整周期十轮中位降至
`2.823 ms`。S6.27 进一步把 FinalDrain 的 global-fatal 读取从“每次
progress”改为“第 0 轮立即检查、之后每 256 轮按 owner-local 计数检查”。
CPU、CCEC、A5 B256 full-swimlane 和 trace-free 十轮已全部通过；完整周期中位
进一步降至 `2.519 ms`，因此该低频最终观察已纳入当前设计合同。

逐阶段代码、泳道与性能数字继续记录在
[PA调度器分离版实现过程](cross_core/PA调度器分离版实现过程.md)，本文只保存
采用后的架构合同和仍待验证的设计边界。
