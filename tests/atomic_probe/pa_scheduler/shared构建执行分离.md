# Shared TensorMap 构建与执行分离设计记录

## 0. 当前状态

| 项目 | 状态 |
| --- | --- |
| 目标 | 让 task 的构建 owner 与 kernel 执行 owner 可以是不同物理核 |
| 当前代码 | Claim winner 在本核构建 private `LocalSlot`，随后仍由本核 Drain 执行 |
| 本文性质 | 持续更新的架构与内存模型设计记录 |
| 正式实现 | **尚未开始** |
| CPU 正确性用例 | **尚未开始** |
| A5 跨核发布探针 | **尚未开始** |
| A5 PA 功能/性能 | **尚未运行** |

本文先定义需要证明的内存合同，不预设最终一定采用中央队列、per-core 队列或 task-indexed cell。任何候选实现都必须先通过本文列出的跨核发布、唯一执行和回收门槛，再讨论性能。

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

预期收益是让 AIC 主要负责 QK/PV 的 Cube 发射与完成，让其他有空闲 Scalar 能在 Cube 执行期间继续构建后续 task。SF/UP 同理可以由空闲 Scalar 构建，再交给 AIV 执行。

这个目标新增了两类共享对象：

1. **已构建 task payload**：TensorDesc、scalar、fanin、function 等执行所需信息；
2. **执行所有权状态**：谁可以读取 payload、谁取得唯一执行权、何时完成、何时允许复用。

因此，原来只围绕 TensorMap 建立的内存合同已经不够；执行包本身必须成为一个完整、可证明的跨核发布协议。

## 2. 三种所有权必须分开

后续讨论统一使用三种 owner，不再笼统称为 winner：

| 角色 | 职责 | 是否要求特定核型 |
| --- | --- | --- |
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
2. **执行核复制到自己的 private slot**：所有 self-pointer 必须重建，不能原样复制 `args[]`。

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

共享执行 payload 应在 `BUILT` 后保持不可变，否则 executor 的普通写会形成 dirty cacheline，给后续 DCCI 和复用带来 stale writeback 风险。可选处理只有：

- executor 领取后把 fanin 复制到自己的 private continuation，再在本地压缩；或
- 不修改共享 fanin，每次只记录 executor-private 的 ready-prefix 游标。

不应直接在共享 payload 上复用现有原地压缩逻辑。

### 3.7 `CompleteTask` 隐含使用执行核的 `worker.heap_next`

当前 [CompleteTask](same_core/common/pa_scheduler_core.h#L542) 把 `worker.heap_next` 写入 task vend。Build 和 Execute 同核时，这个值隐式正确。

分离之后，executor 的 `heap_next` 不等于 builder Materialize 时的 reservation/vend 快照。执行包必须显式携带正确的 completion vend，或者 shared heap 协议另行定义 task 级 vend；不能继续读取 executor 的本地 heap cursor。

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
-> atomic publish BUILT(generation)
```

连续多条 DCCI 可以批量发射，最后在 atomic publish 前统一一次 DSB；前提是中间没有动作依赖其中某条 line 已完成。

### 4.3 consumer 的取得与读取顺序

```text
atomic CAS 取得唯一 Execute owner
-> 消费 CAS 返回值确认成功
-> DSB
-> 对 payload line 执行 DCCI
-> DSB
-> ordinary load payload
```

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

### 5.1 一个 packed atomic 状态机

第一版优先用一个独占 64B line 的 64-bit packed state 表达 generation、phase 和 owner：

```text
FREE(g)
  -> BUILDING(g, build_owner)
  -> BUILT(g)
  -> CLAIMED(g, execute_owner)
  -> RUNNING/PENDING(g, execute_owner)
  -> DONE(g, execute_owner)
  -> FREE(g+1)
```

所有跨核转换使用 CAS/Exchange，不能普通写。generation 防止一个慢 executor 把上轮 slot 的 `BUILT/DONE` 当成本轮状态。

第一版可以保留更多显式 phase，等门槛闭合后再证明哪些状态可以合并。不能为了减少一个 atomic，先把 `BUILT`、`CLAIMED` 和 `DONE` 混成含义模糊的 ready。

### 5.2 建议的物理分区

```text
cacheline 0：state/generation/owner（atomic-only，不 DCCI）
cacheline 1：可选队列链接或单独控制字（atomic-only）
cacheline 2..N：immutable portable payload（ordinary + DCCI）
```

约束：

- control 与 payload 地址范围完全不重叠；
- payload 起点和长度都按 64B 对齐；
- 相邻 slot 不共享 cacheline；
- queue head、queue tail、slot state 分别评估热点，第一版不要挤在同一行；
- host 初始化不能留下与 device atomic 共线的 ordinary dirty 路径。

### 5.3 Portable payload 最小字段

| 字段 | 原因 |
| --- | --- |
| generation、task id | consumer 检查没有读错复用代次 |
| function id/engine kind | 路由 AIC/AIV executor |
| active tensor/scalar/fanin count | 约束有效前缀，拒绝越界 |
| active TensorDesc 或稳定引用 | kernel 参数 |
| scalar 值 | kernel 参数 |
| fanin task ids | executor 判断依赖完成 |
| output/completion task id | 发布正确 task flag |
| completion vend/reservation 快照 | 不能使用 executor 的 heap cursor |
| 通用 multicore 元数据 | 后续支持 joint task，不能永久按 PA 单 lane 特化 |

以下信息原则上不直接放入 portable payload：

- builder 栈地址；
- builder `TaskPayload *`；
- builder private `LocalSlot *`；
- builder 的 block/lane/sub-block；
- builder workspace 地址；
- 指向上述对象的预构造 `args[]` 指针。

### 5.4 两种 payload 消费策略

| 策略 | 优点 | 主要问题 |
| --- | --- | --- |
| executor 直接从 shared payload dispatch | 少一次 4KB 级复制；self-pointer 可指向稳定 shared 地址 | payload 必须保持到 kernel 完成；mutable context 必须外置；执行侧 DCCI 覆盖较多 line |
| executor 复制有效前缀到 private slot | 执行期状态和 fanin 可本地修改；复用现有 Drain 结构较容易 | 必须重建 args 指针；增加复制；shared slot 仍要等复制完成才能复用 |

第一版更建议“共享 portable payload + executor-private binding”：descriptor/scalar/fanin 可从共享包读入 private slot，`args/local_context/global_context` 全部由 executor 重建。这样最接近现有 `DrainReady()`，同时避免执行期把共享 payload 写脏。

## 6. 完整发布与执行协议

### 6.1 Build owner

```text
1. 通过 CAS 取得 FREE(g) -> BUILDING(g)
2. 确认上一 generation 已无 reader/executor
3. 在不占住 TensorMap Register 串行链的位置等待/取得执行槽容量
4. Materialize
5. 严格按 task id Register metadata
6. Fanin lookup，过滤 producer >= current task
7. 构造 portable payload；不写 builder-local pointer
8. Flush payload 全部有效 cacheline；DSB
9. atomic BUILDING(g) -> BUILT(g)
10. Build owner 不再读写该 payload
```

第 3 步很重要：不能先进入 Register 串行区，再因为执行队列满而长期等待 slot，否则会把原本很短的 metadata commit 链变成执行背压链。可以在 Materialize 前预留 slot，也可以在 Register 后、Build 前预留，但等待必须发生在严格插入通道之外，且中间上下文生命周期要明确。

### 6.2 Execute candidate/owner

```text
1. 只读取 atomic state，筛选 engine kind 兼容的 BUILT(g)
2. CAS BUILT(g) -> CLAIMED(g, self)
3. 只有 CAS winner 继续；loser 不接触 payload DCCI
4. DSB -> payload DCCI -> DSB
5. 检查 generation/task/count/function/engine kind
6. 复制有效字段并重建 executor-private args/context
7. 若 fanin 未 ready，把任务保存在本核 private pending continuation
8. fanin ready 后发射 kernel
9. 等待 engine 真正完成
10. 使用 payload 中的 task id/vend 发布 completion
11. atomic 发布 DONE(g)
```

Execute owner 取得 task 后不应因为一个 fanin 未 ready 就永久阻塞整个 executor。它可以保留有限 pending continuation，继续领取或推进其他 ready task；但这会引入容量、年龄和恢复顺序，需要单独门槛。

### 6.3 Completion 与回收

当前完成顺序必须继续保持：

```text
engine final wait
-> output 已完成
-> task vend
-> task completion flag
-> shared execution slot DONE
-> 允许回收
```

需要特别求证：新增加的 `DONE -> FREE` 与既有 completion atomic 之间采用什么硬件顺序。不能仅因源码中两个 atomic 前后相邻，就默认所有远端核观察顺序一致。

回收条件至少包括：

- Execute owner 已完成最后一次 ordinary payload 读取；
- executor-private copy/continuation 不再引用 shared slot；
- task completion 已经发布；
- 没有观察线程在本轮仍可能读取 payload；
- reclaimer 通过 generation-aware CAS 完成 `DONE(g) -> FREE(g+1)`。

复用前，下一 builder 已经取得整槽独占所有权。若不能证明每个 payload byte 都会覆盖，应先按所有权协议 DCCI 旧 entry，再写新 generation。

## 7. TensorMap 顺序与执行顺序的关系

构建/执行分离不能改变以下边界：

```text
task N Materialize
-> wait deps_prepared[N-1]
-> publish metadata(N)
-> publish deps_prepared[N]
-> Fanin/Build
-> publish execution payload(N)
```

严格顺序只约束 TensorMap metadata side effect。执行顺序由 fanin 决定：

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

## 9. 容量、背压和活性

### 9.1 当前 private slot 容量不能直接沿用

当前每 worker 有 4 个物理 slot，其中 2 个给 BlockWon 预留，普通 kernel 只能使用 2 个。共享执行队列的容量需要按 builder 生产率、AIC/AIV kernel 时长和 fanin 等待共同确定，不能简单用 `96 × 2` 或继续每核固定 2 个。

### 9.2 Builder 不能占住有序 Register 等空槽

危险交错：

```text
task N owner 取得 Register turn
-> 发现执行队列满
-> 等待 executor 释放 slot
-> task N+1.. 永远不能推进 metadata
```

正确做法是让容量等待发生在 Register 外，或为已经取得 turn 的 owner 预留保证可用的发布槽。二者需要通过容量证明，不能依赖“通常会很快释放”。

### 9.3 Executor 领取未 ready task

若 executor 领取后原地 spin，会浪费专门保留的 AIC/AIV；若放回共享队列，又需要第二次所有权转移和 ABA 处理。第一版可以：

- 领取后放入 executor-private 有界 pending 列表；
- 优先执行 ready task；
- pending 满时停止领取并推进最老 fanin；
- FinalDrain 时强制收口全部 pending。

这与已有 Scalar continuation 机制相似，但 task payload 生命周期更长。

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

至少需要定义以下路径：

| 失败点 | 必须发生的动作 |
| --- | --- |
| slot reserve 后、payload publish 前 | 回到 FREE 或进入 terminal CANCELLED；其他核不能读取 payload |
| TensorMap 已 Register、Build 失败 | metadata 不可回滚；全局 fatal，禁止发布可执行 task |
| payload 已 BUILT、executor 校验失败 | 标记 fatal/CANCELLED，禁止执行和复用 |
| executor CLAIMED 后 kernel 失败 | 不得发布正常 completion；FinalDrain 必须能终止 |
| completion 已发布、DONE 发布失败 | 不得重复执行；reclaim 进入可诊断终态 |

不能用 host 超时作为唯一活性机制。设备侧必须有 terminal fatal，使所有 builder/executor 停止生产并收敛退出。

## 12. 正确性不变量

第一版实现提交前，至少逐项证明：

1. 每个 task 恰好一个 Build owner；
2. 每个 kernel task 恰好一个 Execute owner；
3. 任一 executor 观察到 `BUILT(g)` 时，本 generation payload 已完整写回 GM；
4. executor 在读取 payload 前完成自己的 cache 刷新；
5. `BUILT` 后 payload 不再被 builder 或其他候选普通写；
6. payload line 与所有 atomic control line 完全分离；
7. executor 读取的 generation、task id、kind 和 count 一致；
8. 所有 self-pointer 在最终消费地址空间中正确；
9. fanin 只包含 `< current task` 的 producer；
10. fanin 未 ready 时不执行 kernel；
11. engine 未完成时不发布 task flag；
12. completion vend 来自 task 的构建/分配上下文，不来自错误 executor heap cursor；
13. task completion 只发布一次；
14. slot 在 executor 最后一次读取前不复用；
15. reuse 后慢 reader 不会把旧 generation 当新任务；
16. FinalDrain 同时证明 builder 停止生产、所有队列为空、所有 executor/pending 为空；
17. AIC/AIV function 路由严格匹配；
18. fatal 路径不会执行半构建或已取消 payload；
19. ordinary TensorMap 和 INOUT writer history 语义不因执行 owner 改变；
20. CPU 模型通过不代替 A5 DCCI/atomic 同构门槛。

## 13. 建议的验证顺序

### S0：冻结合同，不接 PA

- 定义 packed state、payload line、generation 和 ownership；
- 静态断言 control/payload/slot 64B 隔离；
- 明确每个状态转换使用的 atomic 及返回值是否参与判断；
- 明确 DCCI 地址范围和唯一 DSB 收口点。

### S1：CPU 定向并发门槛

用很小的 synthetic payload 覆盖：

- builder 在 payload 一半处暂停，executor 不得读取；
- flush 完成但 BUILT 尚未发布，executor 不得读取；
- 多 executor 同时 CAS，恰好一个成功；
- executor claim 后延迟，slot 不得复用；
- generation wrap/reuse 前后慢 reader 不误认；
- fanin 未 ready、随后 ready；
- fatal/cancel 每个状态均能退出。

CPU 只验证状态机和交错，不用于证明 A5 cache 可见性。

### S2：A5 CCEC 最小跨核 payload 探针

不 include PA scheduler，固定 2–4 个核，分别覆盖：

- AIV0 builder -> AIV1 executor；
- AIV builder -> AIC executor；
- AIC builder -> AIV executor；
- 同 role 不同 block；
- payload 1、2、8、76 条 cacheline；
- fresh slot 与 reuse generation；
- publisher/consumer 在每个协议边界主动延迟；
- atomic line 邻接 guard、payload 每字段精确值、最终 GM 快照。

必须显式关闭/记录编译器自动 Scalar DCCI 和 kernel-end DCCI，避免自动行为掩盖协议缺口。

### S3：Shared task cell，但执行仍固定映射

在 standalone 中：

- Build owner 发布 shared portable payload；
- executor 先用 `(task_id % compatible_workers)` 固定映射，不做动态竞争；
- 验证跨核 Build/Execute、fanin、completion 和回收；
- 先跑 B1 正确性，再跑 B256；
- 此阶段不宣称负载均衡或性能收益。

固定映射可以把“payload 内存模型”与“动态 execution election”分开验证。

### S4：加入 exactly-once execution election

- 多个兼容 executor 竞争同一 task；
- loser 不读 payload；
- 统计 duplicate/missing execution；
- 覆盖 executor 延迟、乱序和 fanin pending；
- 比较 task-indexed、中央队列和分片队列的 atomic 数量。

### S5：再接 engine/Scalar overlap

- executor 发射 engine 后保存 continuation；
- `try_wait(BUFFER_ID)` 只决定何时恢复；
- 最终仍执行原 wait，再发布 completion；
- 同核 in-flight 从 1 开始，不直接扩展任意深度。

### S6：性能证据

固定三条互不混算的证据链：

- perf-clock：决定候选保留/撤销；
- swimlane：看 Build publish、Exec claim/bind、pending、kernel 和 completion 落点；
- submit-pmu：解释 Scalar/I-cache 变化。

必须增加的对照：

1. 现有 private LocalSlot 同核执行基线；
2. shared payload 但仍同核执行：单独量发布税；
3. shared payload、固定跨核映射：量跨核取得税；
4. 动态 executor：量负载均衡收益与 election 成本；
5. full overlap：量最终端到端收益。

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

第一版固定映射用于闭合内存模型；动态版本应优先评估分组、bitmask、per-executor queue 或一次领取一批 task，而不是默认全核 CAS 同一地址。

### 14.4 观测代码不能进入协议 cacheline

泳道记录、计数器和 debug checksum 不得放进 execution slot control/payload line。观察动作应在取得时间端点后写每核独占 trace buffer，不能为了显示状态而给每次 poll 新增 raw record。

## 15. 当前推荐的第一版原型

在尚未做任何实现前，当前最稳妥的推进顺序是：

1. 使用 task-indexed、单轮不复用的 `SharedExecCell[task_id]`；
2. 每 cell 一个独占 atomic state line，payload 独立 64B 对齐；
3. payload 只保存有效 tensor/scalar/fanin 和 completion vend，不保存 self-pointer；
4. Build owner 完整 flush 后发布 BUILT；
5. executor 先固定映射到另一个兼容核，不做动态竞争；
6. executor CAS 取得所有权、DCCI payload、复制到 private slot并重建 args；
7. 继续复用现有 `SlotReady -> ExecuteKernel -> CompleteTask` 主体，但 completion vend 改从 task payload 取得；
8. A5 同构探针和 B1/B256 正确性闭合后，再引入动态执行队列；
9. 确认共享 payload 发布成本仍有收益空间后，才做紧凑化和 engine continuation。

这一版不是最终高性能形态。它的价值是把三个未知量拆开：

```text
跨核 payload 内存模型
≠ 动态执行仲裁
≠ Scalar/engine coroutine
```

只有第一项闭合后，后两项的性能结果才有解释价值。

## 16. 尚未决定的问题

1. Build owner 候选应为所有 Scalar、只用 AIV，还是按当前 task 核型？
2. executor 直接 dispatch shared payload，还是先复制到 private slot？
3. portable payload 的最小通用 ABI 是什么，怎样兼容真实 function address 和 joint task？
4. completion vend 应在 Materialize、Build 还是 execution bind 时冻结？
5. fanin 未 ready 的 task 由 executor 持有，还是在 ready 后才进入执行队列？
6. 如何避免一个未 ready FIFO head 阻塞后续独立 task？
7. task-indexed cell 的内存开销是否可接受到哪个阶段？
8. 动态 executor election 如何不新增 Claim 等级的同地址 atomic 热点？
9. AIC 与 AIV 是否使用完全独立的执行发现结构？
10. FinalDrain 如何证明 builder 停产、queue 为空、pending 为空和 engine in-flight 为零？
11. BlockWon/multicore task 的 portable context 与执行 owner 如何表达？
12. shared execution payload 是否能复用生产 RingSlot ABI，还是需要独立中间 ABI？

## 17. 更新记录

### 2026-08-01：建立问题边界

- 明确 Build owner、Execute owner、Completion owner 三种角色；
- 确认当前 `LocalSlot` 的提前 `built`、普通状态共线、非 64B 步长、自引用 args、mutable fanin 和 executor heap cursor 都不能直接跨核复用；
- 以现有 A5 DCCI/atomic 探针为依据，给出 payload publish/acquire/reclaim 基础合同；
- 建议先做 task-indexed、固定跨核映射，分离验证内存模型与动态执行仲裁；
- 当前没有实现和运行结果，本文所有新结构仍为待验证候选。
