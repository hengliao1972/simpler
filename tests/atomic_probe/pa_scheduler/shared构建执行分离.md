# Shared TensorMap 构建与执行分离设计记录

## 0. 当前状态

| 项目 | 状态 |
| ---- | ---- |
| 目标 | 让 task 的构建 owner 与 kernel 执行 owner 可以是不同物理核 |
| 当前代码 | Claim winner 在本核构建 private `LocalSlot`，随后仍由本核 Drain 执行 |
| 本文性质 | 持续更新的架构与内存模型设计记录 |
| 正式实现 | S0–S3a 已完成；S3b 固定两候选异核执行已接入，B256 待闭合 |
| CPU 正确性用例 | S1–S3b 协议、PA payload、两候选发现和 drain 门槛已完成 |
| A5 跨核发布探针 | S2 已完成，100 轮共 3200 case 通过 |
| A5 PA 功能/性能 | S3b B1 固定异核执行 4 轮全部通过；暂不作性能结论 |

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
| fanin 执行期状态 | `SlotReady()` 可原地压缩本核 private fanin | shared payload 在 `BUILT` 后不可变；首版每核只有一个紧凑 execution token，压缩/游标只放该 token |
| completion vend | shared Materialize 把 `reservation.aggregate_vend` 写入 Build owner 的 `worker.heap_next`，同核 `CompleteTask()` 随后消费 | Build 时冻结 task 的 completion-vend 快照并显式随 payload 交接；不得读 executor 的 `heap_next` |
| Execute exactly-once | 隐含由同一 Claim winner 保证 | 先用单一指定 executor + atomic phase transition 验证，再单独加多候选仲裁 |
| cell 生命周期 | 本核 `occupied_count` 与 private ring 回收 | 首版 task-indexed 且单轮不复用；后续环形复用才引入 generation/reclaim |
| FinalDrain | 所有 replay actor 停产 + 本核 private slot 为空 | 还要证明所有 kernel cell 已 DONE、每核 execution token 为 IDLE、engine 无 in-flight |
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
| executor 本地容量与全局 backlog 解耦 | S0 定义单 execution token，S1/S3 验证忙核不 Claim | 没有空闲 token 时不发射 CAS，task 保持 `BUILT` 并可由其他兼容核领取 |
| Build owner 扩大到其他 Scalar | S5 独立改候选核拓扑 | 不借助跨核发布的正确性掩盖 Build 角色变化 |
| engine 执行与 Scalar 继续调度 | S6 continuation/`try_wait` | 只在 S0–S5 闭合后接入，不与跨核 payload 首次验证混做 |
| 复用、队列和回收 | S7 之后的容量优化 | task-indexed 单轮先闭合；需复用时才引入 generation/ABA 证明 |

这个顺序刻意不在首版同时改 Build Claim、执行队列、cell 复用和 engine coroutine。否则即使出错，也无法确定是发布合同、owner 仲裁还是回收引起的。

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
2. **执行核复制 active prefix 到自己唯一的紧凑 execution token**：所有 self-pointer 必须重建，不能原样复制 `args[]`；不为此恢复 4-slot private ring。

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

共享执行 payload 应在 `BUILT` 后保持不可变，否则 executor 的普通写会形成 dirty cacheline，给后续 DCCI 和复用带来 stale writeback 风险。首版不再建立“每核多个 private pending slot”，而是：

- `SharedExecCell[]` 是全局 backlog，未领取任务始终保留在 `BUILT`；
- 每个 executor 只有一个紧凑 execution token，只在 token 为 `IDLE` 时参与 `BUILT -> CLAIMED`；
- 领取后把有效 fanin 和执行必要字段一次复制到该 token，在本地压缩或只记 ready-prefix 游标；
- token 未回到 `IDLE` 前，本 executor 不再领取第二个 task。

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
- executor 若需要修改 fanin/context，先复制到 private continuation。

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

每个 executor 首版只保留一个本地 execution token。正常主干为 `IDLE -> BINDING -> WAITING_FANIN -> ENGINE_INFLIGHT -> COMPLETING -> VEND_PUBLISHED -> COMPLETION_PUBLISHED -> IDLE`，错误收敛到 `FAULTED`。这些不是新的共享 atomic phase；跨核只需要清晰的 `BUILT`、`CLAIMED` 和 `DONE` 边界。该 token 只保存 task id、有效 payload binding、fanin 游标和 completion 所需的紧凑状态，不是 4,824B `LocalSlot` 的另一份拷贝。

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

首版不直接把 4,824B `LocalSlot` 当 shared payload。它只序列化 active prefix，然后由 executor 绑定到自己唯一的紧凑 execution token/context。

### 5.5 默认性能合同：本地准备，GM 只单向经过一次

same-core 优化已经形成一条必须延续到 cross-core 的经验：**控制计算尽量只使用寄存器和栈上小型 POD；只有必须持久或跨核的结果才落 GM，并且每条 destination line 单向写一次、最后统一 DCCI 发布**。

“owner-private”只表示没有其他核并发访问，不表示它不在 GM。当前 `WorkerState::LocalSlot` 虽是本 worker 私有，物理上仍是 GM 对象，同样可能付出 DCache miss。因此 owner-private GM 只用于必须跨 Submit/等待点存活的 continuation，不应冒充临时 staging。

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
  先确认本核 execution token == IDLE
  -> 只 poll/CAS atomic control
  -> 取得 CLAIMED 后对完整 payload 调用一次 InvalidateRegion
  -> 执行一次 forward read/copy 到 executor-private binding
  -> 从此只操作 private fanin/context/args
  -> 等待、kernel 和 completion 不再回读 shared payload
  -> 发布 DONE 后才把 execution token 恢复为 IDLE
```

这里的“一次 DCCI”指一次**批量发布 helper**：多 line payload 在硬件上仍需要每条 line 一条 DCCI，但中间不夹杂业务读写，末尾只一次 DSB。禁止把 `FlushRegion()` 放入 tensor/scalar/fanin 的字段循环，也禁止“先写 header -> flush -> 再补 count -> 再 flush”。

这条原则不等于把完整 payload 复制到大栈对象：

- header、count、offset、fanin 去重结果等小数据先在本地准备；
- 128B `TensorDesc` 等大前缀可以在最终 pack 时从已验证的源地址直接流式复制到最终 GM 位置，不再经过另一份 4KB 级栈镜像；
- S0 必须记录 staging 结构大小并检查编译后栈框/溢出；“写成栈变量”不能用来掩盖过大本地对象造成的反向回退。

executor-private binding 也必须区分两类：立即 ready 且在当前调用内完成的参数尽量保留在寄存器/栈；需跨 fanin 等待点存活的唯一 execution token 才写 owner-private GM。后者只写紧凑必要字段，不复制整个 4,824B `LocalSlot`，也不扩展为本核多项 pending 列表。

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
| executor 复制有效前缀到唯一紧凑 token | 执行期状态和 fanin 可本地修改；不继承每核 4 个完整 slot | 必须重建 args 指针；增加一次有效前缀复制；token 占用时不能再领取 |

第一版更建议“共享 portable payload + 每 executor 一个紧凑 binding”：descriptor/scalar/fanin 只复制 active prefix，`args/local_context/global_context` 由 executor 重建。这样保留现有 `DrainReady()` 的局部执行逻辑，但不把 same-core 的 4-slot ring 带入 cross-core。

### 5.7 `dc_preload` 的预埋点与边界

本节只在 5.5 的“减少 GM 触碰次数”结构门槛已满足后才启用。`dc_preload` 不得用来给反复读写 shared payload、分段 DCCI 或过大 GM continuation 擦屁股。

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
  -> 预留唯一 execution token、初始化 executor-local context 等独立工作
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

### 6.2 Execute candidate/owner

```text
1. 先检查本核 execution token == IDLE；非 IDLE 时不扫描、不发射 Claim CAS
2. 只读取 atomic state，筛选 engine kind 兼容的 BUILT
3. S3 的唯一指定 executor 执行 CAS BUILT -> CLAIMED(self)
4. S4 才允许多候选竞争；只有 CAS winner 继续，loser 不接触 payload DCCI
5. 消费 CAS 返回值，先校验 atomic state 中的 engine_class/payload_lines 上界
6. 将本核唯一 execution token 绑定到该 task，直到 DONE/fatal 前不再领取其他 task
7. 调用 InvalidateRegion(payload)
8. 可选对后续要读的 payload source 发起 dc_preload；不参与正确性
9. 检查 task/count/function/engine kind 和由 counts 重算的 payload 边界；reuse 版再检查 generation
10. 一次复制有效字段并重建紧凑 executor-private args/context
11. 若 fanin 未 ready，token 转为 WAITING_FANIN；本核可继续不需要第二 execution token 的 Scalar 工作，但不再 Claim task
12. fanin ready 后发射 kernel，token 转为 ENGINE_INFLIGHT
13. 等待 engine 真正完成
14. 使用 payload 中的 task id/completion-vend 发布原有 vend/flag
15. atomic 发布 DONE，然后才将本核 token 恢复为 IDLE
```

`SharedExecCell[]` 本身是 GM 中的全局待执行 task list，不是每核 private ring 的拷贝。没有本地 token 容量时，task 保持 `BUILT`，当前核不得先 Claim 后堆入多项 pending 列表。

S3 固定映射阶段必须让每个 executor 按 task id 递增处理自己的映射序列，不允许跳过早期任务去占住唯一 token。这份序列不得通过“96 核各自重建整份 PA plan”事后推导：每核在自己的 Submit 路径中本来就同时拥有 `task_id` 和模板 `Kind`，应在本次 Submit 成功闭合时把候选身份登记到 Scalar 本地紧凑位图/任务号队列。非候选任务只体现为本地空位，不读 GM control，也不保存 batch/offset 重建游标。

S4 动态任务池不得直接把“随机领取未 ready task”当最终策略；在引入广泛动态竞争前，必须先二选一地闭合活性：

- 提供独立、可验证的 ready summary，使 executor 只对 dependency-ready task 发射 Claim；或
- 给出不会让全部 token 被未 ready task 占满的容量和推进证明。

首选是 ready summary；没有这份证明前，S4 只做受控任务的 exactly-once election，不宣称已形成通用动态调度器。

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

因此，一个简单按 task id 排序的单 FIFO 可能产生 head-of-line blocking。它可以作为第一版正确性模型，但不能未经数据就当最终高性能结构。

## 8. 队列/任务发现机制的候选

### 8.1 Task-indexed execution cell

每个 task id 独占一个 execution cell，本轮不复用。

优点：

- 没有 wrap/ABA；
- task/state/payload 一一对应，最容易建立内存模型；
- 失败定位和 host 精确检查简单。

该阶段的 cell 只从 `EMPTY` 走到 `DONE`，不因为未来可能改 ring 就预先加 generation 和 device reclaimer。

缺点：

- 最大 payload 若沿用 4,824B，4,352 个 task 需要约 21 MiB；
- executor 如何高效发现 BUILT task 仍需 cursor/bitset；
- 大量固定空白字段会增加 DCCI 和容量。

建议第一版正确性原型采用 task-indexed cell，但只发布紧凑有效前缀，不把它直接宣布为最终结构。

### 8.2 中央 MPMC ring

优点是结构紧凑、天然提供已构建 task 流。问题是 producer/consumer head、tail 和 slot seq 会成为新的 atomic 热点；A5 同地址并发 atomic 代价可能抵消构建/执行 overlap。

如果采用，必须有 per-slot generation/seq；仅靠全局 head/tail 不能防止 consumer 读取尚未发布或已经复用的 payload。

### 8.3 Per-builder queue + work stealing

每个 builder 对自己的 SPSC/MPSC 队列发布，executor 扫描或窃取兼容 task。

它可以分散 atomic 热点，但会增加：

- 队列选择和负载均衡；
- task age/fairness；
- 多队列 fanin-not-ready 跳过；
- FinalDrain 判断“全局不再生产且全部队列为空”的难度。

该方案应在 task-indexed 内存模型已经闭合后再评估，不作为第一步。

### 8.4 Ready bitmask/summary

按组发布 AIC/AIV built/ready bitmask 可以减少 executor 全表扫描，但 bitmask 更新本身是共享 atomic。需要区分：

- `built`：payload 可读；
- `dependency ready`：可立即执行；
- `claimed`：已有 executor；
- `done`：可以回收。

不能用一个 bit 同时表达四种状态。

这不只是扫描优化，也是“每 executor 一个 token”进入通用动态任务池前的首选活性条件：未 ready task 留在全局 cell 中，不提前占住某个 executor。ready summary 如何从 fanin completion 中低成本产生属于 S4 的独立设计门槛，不在 S0–S3 臆造新 atomic 协议。

## 9. 容量、背压和活性

### 9.1 当前 private slot 容量不能直接沿用

当前每 worker 有 4 个物理 slot，其中 2 个给 BlockWon 预留，普通 kernel 只能使用 2 个。而且该容量检查发生在 Claim winner 已经产生之后：满时是本核 Drain/等待，不是放弃已经赢得的 task。这是 same-core “owner 必须保存到执行完”的局部背压，不是 cross-core 应继承的任务池形态。

task-indexed 首版为每个 task 预留一个 GM cell，它们共同承担全局 backlog；builder 发布后不需要占有本核执行 slot。每个 executor 只有一个紧凑 execution token，该 token 是“已领取执行责任”的本地容量，不是 task 存储容量。

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

首版明确不实现 executor-private pending task list：

```text
GM task list:
  SharedExecCell[task 0]  EMPTY/BUILDING/BUILT/CLAIMED/DONE + portable payload
  SharedExecCell[task 1]  EMPTY/BUILDING/BUILT/CLAIMED/DONE + portable payload
  ...

Executor local admission:
  one execution token: IDLE -> WAITING_FANIN -> ENGINE_INFLIGHT
                              -> COMPLETING -> IDLE

token == IDLE  : 允许对一个兼容 BUILT task 发射 Claim CAS
token != IDLE  : 不 Claim；其他 BUILT task 继续留在 GM task list
```

- 未领取 task 一直位于 GM `SharedExecCell[task_id]`，状态为 `BUILT`；
- executor 的唯一 token 为 `IDLE` 时才允许对一个兼容 task 发射 Claim CAS；
- CAS 失败不占 token，也不读 payload；
- CAS 成功后该核对 task 负责到 `DONE`，不得再领取第二个 task；
- token 可在 `WAITING_FANIN` 时让 Scalar 处理不需要第二 token 的工作，但不得把“继续 Claim”伪装成 overlap。

这一版主动放弃“一核囤积多个未 ready task”带来的绕过能力，换取最小所有权和内存模型。固定映射通过每 executor 的 task-id 顺序保证推进：所有 fanin producer 都严格小于 consumer task id，同一 executor 不跳过早期映射 task，因此跨 executor 等待链上的 task id 只能严格递减，不能形成环。这份证明依赖“兼容 executor 最终被调度、engine 最终完成”的基本活性前提。通用动态版则必须先有 ready-only 发现机制或独立的活性证明。

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

首版可以在 standalone PA 上缩小验证范围，但协议字段和失败检查必须为通用算子留出明确扩展点。PA adapter 特例不能进入通用 shared execution runtime。

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
18. execution token 非 IDLE 时本 executor 不发射新的 Claim CAS，任一时刻最多持有一个已领取 task；
19. 未被空闲 executor 领取的 task 保持 `BUILT`，不因某个忙核而移入私有队列；
20. 正常路径先发布 completion 和 cell `DONE`，然后才把本核 execution token 恢复为 IDLE；
21. FinalDrain 同时证明 builder 停止生产、所有 kernel cell 全部 DONE、每核 execution token 为 IDLE、engine 无 in-flight；
22. 后续 reuse 版额外证明慢 reader 不会把旧 generation 当新任务；
23. AIC/AIV function 路由严格匹配；
24. fatal 路径不会执行半构建或已取消 payload；
25. ordinary TensorMap 和 INOUT writer history 语义不因执行 owner 改变；
26. `dc_preload` 关闭/开启变体通过同一 oracle，hint 被硬件忽略也不改变正确性；
27. CPU 模型通过不代替 A5 DCCI/atomic 同构门槛。

## 13. 建议的验证顺序

### S0：冻结 ABI 与合同，不接 PA 业务

- 在 `cross_core/` 内定义独立的 task-indexed `SharedExecCell`，`same_core/` 只作对照；
- 首版 packed state 包含 `phase + owner + engine_class + payload_lines + task_id`，phase 只有 `EMPTY/BUILDING/BUILT/CLAIMED/DONE`，不包含 generation/reclaim；
- 定义每 executor 唯一的紧凑 execution token，以及 `IDLE/BINDING/WAITING_FANIN/ENGINE_INFLIGHT/COMPLETING/VEND_PUBLISHED/COMPLETION_PUBLISHED/FAULTED` 本地状态，不定义 private pending 数组；
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
- execution token 非 IDLE 的 executor 不得发射 Claim CAS，待领取 cell 仍保持 `BUILT` 并能被另一兼容空闲 executor 领取；
- 任一时刻每 executor 最多一个 token 处于 `WAITING_FANIN/ENGINE_INFLIGHT/COMPLETING`，不存在第二个 private pending task；
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
- 只在本核唯一 execution token 为 IDLE 时取得，复制 active prefix 并重建 args/context，使用 payload vend 完成；
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
- 每个 executor 按 task id 递增处理自己的固定映射序列，token 忙时不跳过早期 task 领取更晚 task；
- 候选看到 `EMPTY` 时保留队头；看到已带 `build_owner` 的 `BUILDING/BUILT/CLAIMED/DONE` 后，非 target 候选才能丢弃本地记录；只有全局停产后的残留 `EMPTY/BUILDING` 才是 terminal 缺口；
- 先跑 B1 正确性，再跑 B256；host 精确检查 build owner 与 execute owner 不同、task/descriptor/fanin/vend/completion 全部一致；
- FinalDrain 使用“所有 builder 停产 + 所有 kernel cell DONE + 每核 token IDLE + engine 无 in-flight”收口。

S3a 和 S3b 把“shared payload 发布税”与“跨核取得税”分开，同时不混入动态 election。

### S4：加入 exactly-once execution election

- 多个兼容 executor 竞争同一 task；
- loser 不读 payload；
- 统计 duplicate/missing execution；
- 忙 executor 不发射 CAS，未领取 task 始终保持 `BUILT`；
- 受控测试覆盖 executor 延迟、乱序和单 token `WAITING_FANIN`；
- 在扩展为通用动态任务池前，必须闭合 ready-only 发现或等价活性证明，防止所有 token 被未 ready task 占满；
- 先在 task-indexed cell 上比较候选集和 atomic 数量，不在同一步引入中央 MPMC ring。

### S5：独立扩大 Build owner 候选核

- 在 S3/S4 已证明的 execution handoff 上，再比较现有核型绑定 Build 与“任意兼容 Scalar Build”；
- 先证明 Materialize/Register/Fanin/portable Build 不携带执行核私有状态；
- 只有 TensorMap metadata 保持 task-id 严格串行，Build payload 和 kernel 执行不得因此串行；
- 用独立 commit 记录该拓扑变化，不让它污染 S3 内存模型结论。

### S6：再接 engine/Scalar overlap

- executor 发射 engine 后继续使用同一个 execution token 保存 continuation；
- `try_wait(BUFFER_ID)` 只决定何时恢复；
- 最终仍执行原 wait，再发布 completion；
- 同核 in-flight 从 1 开始，不另建多项 pending 列表，不直接扩展任意深度。

### S7：性能证据与后续容量优化

固定三条互不混算的证据链：

- perf-clock：决定候选保留/撤销；
- swimlane：看 Build publish、Exec claim/bind、token WAITING_FANIN、kernel 和 completion 落点；
- submit-pmu：解释 Scalar/I-cache 变化。

必须增加的对照：

1. 现有 private `LocalSlot` 同核执行基线；
2. shared payload 但仍同核执行：单独量发布/取得/重绑税；
3. shared payload、固定跨核映射：量跨核交接税；
4. 动态 executor：量负载均衡收益与 election 成本；
5. 扩大 Build owner 候选核：量真正构建负载转移收益；
6. full overlap：量最终端到端收益。

只有 task-indexed 内存模型、动态 election 和端到端收益都有证据后，才评估紧凑 ring/per-builder queue。到那一步再引入 generation、ABA、容量背压和 device reclaim 门槛。

## 14. 性能上最危险的成本

### 14.1 直接发布完整 `LocalSlot`

`LocalSlot` 为 4,824B，向上覆盖约 76 条 cacheline。若每个 1,024 个 kernel task 都由 builder flush、executor invalidate 一次，仅 payload DCCI 就接近：

```text
1,024 × 76 × 2 = 155,648 cacheline operations
```

这还没有计算复制、队列 atomic 和 I-cache 膨胀，极可能吞掉 overlap 收益。因此不能为了尽快跨核，直接把现有最大结构逐 task 全量发布。

### 14.2 紧凑 payload 与重复 lookup 的权衡

可比较：

- 发布 active TensorDesc/scalar/fanin 的连续紧凑 payload；
- 只发布 SharedOutputRef，executor 再读取 shared_outputs；
- 按 producer 批量发布 descriptor block；
- shared payload 只保存 offsets，executor 重建 args。

紧凑 payload 增加 pack/unpack，引用模式增加 atomic/DCCI/lookup。需要用真实 task 形状分别量发布 line 数、executor load、Build 时间和端到端，不能只看结构体大小。

### 14.3 新 execution election 不能重演 Claim 热点

当前主要瓶颈已经是多核同地址返回型 atomic。若每个 built task 再让 32/64 个 executor 竞争一个中央状态，会新增一套 Claim 级热点。

第一版固定映射用于闭合内存模型；动态版本应优先评估 ready summary、分组、bitmask 或分散任务发现结构，而不是默认全核 CAS 同一地址，也不是让每核提前囤积多个未 ready task。

### 14.4 观测代码不能进入协议 cacheline

泳道记录、计数器和 debug checksum 不得放进 execution slot control/payload line。观察动作应在取得时间端点后写每核独占 trace buffer，不能为了显示状态而给每次 poll 新增 raw record。

## 15. 当前推荐的第一版原型

当前 standalone 首版按以下顺序推进：

1. 使用 task-indexed、单轮不复用的 `SharedExecCell[task_id]`，它就是 GM 中的全局待执行 task list/backlog；
2. packed state 首版携带 `phase/owner/engine_class/payload_lines/task_id`，phase 为 `EMPTY/BUILDING/BUILT/CLAIMED/DONE`，不加 generation、queue 和 device reclaim；
3. 每 cell 一个独占 atomic state line，payload 独立 64B 对齐；
4. payload 只保存有效 tensor/scalar/fanin、task id 和 Materialize 后冻结的 completion vend，不保存 self-pointer；
5. 小型 staging 只用寄存器/栈/owner-private 状态；数量全部确定后才一次 forward pack 到 GM，不创建完整 payload 栈镜像；
6. 保持现有 Build Claim 候选核拓扑；Build owner 对整个 payload 仅调用一次 `FlushRegion()`，随后才发布 BUILT/engine_class/payload_lines；
7. 每个 executor 首版只定义一个紧凑 execution token；token 非 IDLE 时不发射 Claim CAS，不建立 private pending task list；
8. 先让 Build owner 自己以 executor 身份取得 shared payload，量出 publication/rebind 代价；
9. 再固定映射到另一个兼容核；每核在 Submit 成功闭合时用当下已知的 `(task_id, Kind)` 登记 owner-local 候选位，executor 按 task id 递增处理本地序列，不重建历史 PA plan，不做动态竞争；
10. executor 消费 CAS 返回值、校验 payload_lines、对整个 payload 调用一次 `InvalidateRegion()`、一次 forward copy active prefix 到唯一 token binding，之后不再读 shared payload；
11. 在唯一 token 内复用现有 `SlotReady -> ExecuteKernel` 的局部主体，completion 改为使用 task payload 中的 vend；发布 DONE 后才释放 token；
12. A5 独立探针和 B1/B256 正确性闭合后，再引入受控的动态 execution election；
13. 通用动态任务池必须在 ready-only 发现或等价活性证明闭合后才启用，不允许所有 token 被未 ready task 占满；
14. S2 正确性先在 DCache preload 关闭时闭合；之后才用预留 hook 为 S2/S3a/S3b 生成独立 A/B 变体；
15. 动态 election 闭合后，再独立扩大 Build owner 候选核，最后才接 engine/Scalar overlap 和有界 ring。

这一版不是最终高性能形态。它的价值是把三个未知量拆开：

```text
跨核 payload 内存模型
≠ 动态执行仲裁
≠ Scalar/engine coroutine
```

只有第一项闭合后，后两项的性能结果才有解释价值。而“Build owner 扩大到所有 Scalar”又是第四个正交变量，不与前三项首次实现混做。

## 16. 尚未决定的问题

1. S5 扩大 Build owner 时，候选应为所有 Scalar、只用 AIV，还是按当前 task 核型？S0–S4 已决定保持现有拓扑。
2. 最终版 executor 是直接 dispatch shared payload，还是复制 active prefix 到唯一紧凑 token？首版已决定后者，不复活 4-slot ring。
3. portable payload 的最小通用 ABI 是什么，怎样兼容真实 function address 和 joint task？
4. 更通用的 heap 协议中 completion vend 应如何表达？首版已决定在 Materialize reservation 成功后冻结。
5. S4 通用动态版的 ready-only 发现由 completion 驱动的 ready summary、定向 scout，还是其他结构实现？S3 固定映射已决定为单 token + task-id 顺序。
6. ready-only 发现如何避免一个未 ready task 导致 head-of-line blocking，同时不为每个 loser 引入 payload DCCI/读取？
7. task-indexed cell 的内存开销是否可接受到哪个阶段？
8. 动态 executor election 如何不新增 Claim 等级的同地址 atomic 热点？
9. AIC 与 AIV 是否使用完全独立的执行发现结构？
10. 动态队列版 FinalDrain 如何证明 builder 停产、queue 为空、每核 token IDLE 和 engine in-flight 为零？task-indexed 版先用全 cell 终态计数收口。
11. BlockWon/multicore task 的 portable context 与执行 owner 如何表达？
12. shared execution payload 是否能复用生产 RingSlot ABI，还是需要独立中间 ABI？

## 17. 更新记录

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

- `cross_core` 已形成独立 portable payload、task-indexed cell、每 executor 单 token 和 global fatal 首版；
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
- 去除“cross-core 继续每核维护多个 private pending slot”的默认：`SharedExecCell[]` 承担全局 backlog，每 executor 首版只有一个紧凑 execution token，忙时不 Claim；
- 将动态任务池与受控 exactly-once election 分开：前者还必须闭合 ready-only 发现或等价活性证明，防止全部 token 被未 ready task 占满；
- 预留 builder destination、executor source 和 executor-private destination 三类编译期 DCache preload hook，明确它们默认关闭、不改 ABI、不代替 DCCI/DSB/atomic；
- 将验证拆为 same-owner shared publication、fixed different-core handoff、dynamic Execute election、Build 候选核扩大、engine continuation 五个正交阶段。

### 2026-08-01：建立问题边界

- 明确 Build owner、Execute owner、Completion owner 三种角色；
- 确认当前 `LocalSlot` 的提前 `built`、普通状态共线、非 64B 步长、自引用 args、mutable fanin 和 executor heap cursor 都不能直接跨核复用；
- 以现有 A5 DCCI/atomic 探针为依据，给出 payload publish/acquire/reclaim 基础合同；
- 建议先做 task-indexed、固定跨核映射，分离验证内存模型与动态执行仲裁；
- 当前没有实现和运行结果，本文所有新结构仍为待验证候选。
