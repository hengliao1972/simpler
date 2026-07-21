# TensorMap 与全分布式调度：private/shared 实现导读

本文面向第一次接触本仓调度运行时的开发者，目标是回答以下问题：

1. TensorMap 到底保存什么，为什么调度器需要它？
2. 当前仓库的 private TensorMap 如何参与一次任务的提交和执行？
3. 一个完整算子在 AIC、AIV 和 AICPU 之间如何流转？
4. 对比 checkout 中的 shared TensorMap 尝试做了什么，为什么还不能认为它已经
   在真硬件上完成？
5. 两份实现距离 [`atomic_minibench.md`](../../atomic_minibench.md) 定义的总体目标
   还有哪些差距？

## 0. 文档范围与结论

本文核对的代码基线如下：

| 对象 | 基线 | 定位 |
|---|---:|---|
| 当前仓库 | `57841544` | split 架构，只有 private TensorMap 运行时 |
| shared 对比 checkout | `68649810` | monolithic 架构，shared TensorMap 功能尝试 |
| 总体目标 | `atomic_minibench.md` | MB-1 到 MB-9 的完整并发与一致性契约 |

shared 对比 checkout 在本文中指工作区相邻的
`../../../../glm/simpler-fully_distributed`。它只用于分析，不是当前仓库的一部分。

先给出结论：

- **TensorMap 是依赖索引，不是张量数据容器。**它把一个 GM tensor 的地址区间映射
  到最近写这个区间的 task id，提交新任务时据此生成 fan-in 依赖。
- **当前仓库是 private 模式。**每个 AIC/AIV 核都重放同一份 orchestration，并维护
  自己的 TensorMap 副本。只有赢得任务 claim 的核使用查询结果构建任务，但所有核都
  必须执行相同的 insert/retire，才能保持副本一致。
- **AICPU 不参与每个 task 的依赖查询。**它主要负责初始化、唤醒 AICore、等待结束和
  清理。任务提交、claim、TensorMap 更新、等待依赖及 kernel 执行都发生在 AIC/AIV
  的重放过程中。
- **shared checkout 给出了可参考的算法骨架。**它有全局唯一 map、严格按 task id 的
  append sequencer、`seq` ABA 防护、`core_progress[]`、回收、run-ahead、DEPSIG 和
  TMOPS 等设计。
- **shared checkout 不能直接视为完成品。**它在 a5sim 和 host mirror/probe 上有进展，
  但真硬件上的“普通 payload 写入 + 原子发布字段”仍有 cache 可见性问题，集成运行时
  也没有完成可信的 private/shared 等价性上板证明。
- **总体目标不等于“把 map 改成 shared”。**MB-1 到 MB-9 还覆盖 claim、完成标志、
  frontier、`block.won`、确定性 heap、进度反压以及 dcci/coherent seam。shared map
  建立在这些基础协议之上。

本文使用以下状态词，避免把不同强度的证据混在一起：

| 状态 | 含义 |
|---|---|
| 已实现 | 对应逻辑确实位于集成 runtime 源码中 |
| host 通过 | 只证明 host 侧算法模型，不证明 A5 cache/atomic 行为 |
| sim 通过 | 证明模拟环境行为，不自动证明真硬件 cache 可见性 |
| 上板数值通过 | 最终 tensor 数值正确，但不一定证明模式开关或内部协议生效 |
| 严格契约通过 | 数值、DEPSIG、TMOPS、模式确认和压力条件均符合目标 |
| 未证明 | 代码或测试存在，但现有证据还不足以得出目标已完成 |

## 1. 从问题开始：调度器为什么需要 TensorMap

### 1.1 Orchestration 描述的是任务图

一个算子通常不会只发出一个 kernel。以矩阵计算为例，orchestration 可能依次提交：

```text
task 0: GEMM(A0, B0) -> P0
task 1: ADD(C, P0)   -> C
task 2: GEMM(A1, B1) -> P1
task 3: ADD(C, P1)   -> C
```

这里的“依次提交”只确定 task id，不表示必须严格串行执行。真正的约束来自数据依赖：

```text
task 0 ------> task 1 ------> task 3
                                 ^
task 2 --------------------------|
```

- task 1 读取 P0，因此要等待 task 0。
- task 3 读取并改写 C，因此要等待上一次改写 C 的 task 1。
- task 3 读取 P1，因此还要等待 task 2。
- task 0 和 task 2 不共享中间结果，可以并行执行。

调度器需要在提交 task 1 和 task 3 时，自动找出这些 producer task id。TensorMap 就是
完成这项查询的数据结构。

### 1.2 TensorMap 保存的是“最近 producer”

可以先把 TensorMap 想成下面这张逻辑表：

| GM buffer | 字节区间 | 最近 producer |
|---|---:|---:|
| `P_buffer` | `[0, size(P0))` | task 0 |
| `C_buffer` | `[group_begin, group_end)` | task 1 |
| `P_buffer` | `[size(P0), size(P0)+size(P1))` | task 2 |

提交一个读取 tensor `T` 的任务时，调度器查询：

```text
lookup(T.buffer, T.byte_range) -> 最近一个重叠写入 T 的 task id
```

返回的 task id 会进入新任务的 `fanin[]`。执行端只有观察到这些 producer 的完成标志
后，才会调用 kernel。

实际实现不是一张线性表。当前 private map 使用 hash bucket、entry 链和按 task 分组的
回收链；shared 尝试使用 ring-per-bucket。两者服务的是同一个逻辑契约。

### 1.3 “tensor 相同”按地址区间判断

当前 private 实现在
[`tensor_map.h`](../src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/tensor_map.h)
中把 Tensor 转为：

```text
(buffer 基地址, 起始字节偏移 lo, 结束字节偏移 hi)
```

两个记录满足以下条件时被认为重叠：

```text
buffer 地址相同
且 query.lo < entry.hi
且 entry.lo < query.hi
```

连续 tensor 的范围由 shape 乘积计算。非连续 tensor 使用缓存的 extent，因而更接近
一个包围区间，可能产生保守的额外依赖，但不能漏掉真正的地址重叠。

这也解释了为什么仅比较 C++ 对象地址或 tensor 名称是不够的：不同 `Tensor::view`
对象可能指向同一 buffer 的重叠区域，它们在调度意义上仍然相关。

### 1.4 TensorMap 不是这些东西

TensorMap 很容易和其他运行时结构混淆。它不是：

- **不是 tensor 数据本身。**数据仍位于 GM/HBM 或相应 cache 层次。
- **不是任务队列。**待执行任务位于每核 `RingSlot` 或 `block.won` 投递结构中。
- **不是完成标志。**producer 是否结束由 `DistGlobal::tasks[id].flag` 表示。
- **不是 claim 锁。**哪个核负责一个 task 由全局 sharded cursor 决定。
- **不是 cache 一致性协议。**即使 fan-in 完全正确，如果 producer 的 scalar store 只留在
  L1，consumer 仍可能读到旧数据。
- **不是完整读写冲突数据库。**当前实现主要追踪 producer。是否查询 map、是否注册为
  producer，取决于 tensor 参数 tag。

一句话概括各结构的职责：

```text
TensorMap: 谁最后写过这片数据？
claim cursor: 这个 task 由哪个核负责？
RingSlot/block.won: task 在哪里排队和执行？
task flag: producer 是否已经执行完成？
frontier: 从 task 0 开始，连续完成到了哪里？
dcci/atomic seam: 一个核写的内容，另一个核是否真的看得见？
```

## 2. TensorMap 上有哪些操作

### 2.1 reset

每个 worker 开始重放前初始化自己的 private map：

```text
bucket heads = empty
task retirement heads = empty
free list = empty
alive_floor = 0
```

当前 map 位于 `DistGlobal::cores[core_id].map` 所在的 GM 大对象中，但它在并发语义上是
private 的：只有对应 worker 读写该副本，不需要给 map 本身加跨核锁。

### 2.2 lookup

`lookup(T)` 扫描 T 所在 hash bucket，寻找：

1. buffer 基地址相同；
2. byte range 重叠；
3. producer 未早于 `alive_floor`；
4. producer id 最大。

返回最大 producer id 很重要。假设 C 先后被 task 1、task 3 和 task 5 改写，task 6
读取 C 时只需要直接依赖 task 5；task 5 自身已通过 fan-in 串起更早的修改。

当前提交逻辑只对 `INPUT` 和 `INOUT` 做 TensorMap lookup。`OUTPUT` 是新分配结果，不
查询旧 producer。其他 tag 的确切语义应以提交代码为准，不能仅根据名称推断。

### 2.3 insert

提交任务 N 后，下列写方向会注册为新的 producer：

- `OUTPUT`
- `INOUT`
- `OUTPUT_EXISTING`

每次注册都新增一条 `(address range -> N)` entry，而不是覆盖旧 entry。保留窗口内的
历史记录能让重叠 view 和回收逻辑保持简单。

private 模式的关键点是：**所有 worker 都 insert，不只是 claim winner。**否则各核 map
会在第一个输掉 claim 的 task 后产生差异，后续 winner 落到另一个核时就可能计算出不同
依赖图。

### 2.4 retire

map 不能无限增长。提交 task N 前，当前实现计算：

```text
new_floor = N - H
```

并回收 producer id 小于该窗口下界的 entry。entry 同时挂在：

- hash bucket 链，用于 lookup；
- `task_heads[producer % kTaskWindow]` 链，用于按 producer 快速回收。

`H` 因而不仅是性能参数，也是正确性契约：运行时必须保证不会再需要窗口外的 producer。
如果 task 可以无限 run ahead，而旧 producer 仍可能被未来 consumer 引用，单纯按
`N-H` 回收就会漏依赖。

### 2.5 owner task id 是另一条依赖来源

新分配 output 会携带 `owner_task_id`。例如：

```cpp
TaskOutputTensors outs = rt_submit_aic_task(...);
params.add_input(outs.get_ref(0));
```

`get_ref(0)` 中的 tensor 可以直接告诉 consumer：“我是 task N 创建的”。当前
`dist_submit_collect_fanin()` 会先收集这个 owner id，再对 `INPUT/INOUT` 做 map lookup，
最后去重。

这有两个后果：

1. 对新临时 tensor，显式 owner id 和 TensorMap 可能同时找到同一个 producer，去重后
   只保留一条边。
2. 只测试 `get_ref()` 链路不能充分验证 TensorMap。要验证 map，测试还需要使用没有
   owner id 的 external buffer/view，让依赖只能靠地址区间查询建立。

## 3. 当前 fully distributed runtime 的整体结构

### 3.1 三类执行角色

| 角色 | 主要职责 |
|---|---|
| AICPU | 创建全局状态、准备参数、启动 worker、等待完成、回收资源 |
| AIC | 重放 orchestration、参与 claim、执行 cube/AIC kernel |
| AIV | 重放 orchestration、参与 claim、执行 vector/AIV kernel |

当前 fully distributed 路径不是“AICPU 每提交一个任务，再通知 AIC/AIV”。更接近：

```text
AICPU: setup -> wake all workers ---------------------> wait -> teardown
                   |                                     ^
AIC/AIV core 0:    replay orchestration + execute tasks -|
AIC/AIV core 1:    replay orchestration + execute tasks -|
...
```

因此，在 orchestration 执行期间 AICPU 和 AIV 没有逐 task 交互。AIC/AIV 之间则通过
`DistGlobal` 中的 cursor、flags、frontier、`block.won` 等结构持续交互。

源码里仍能看到名为 `aicpu_orchestration_entry` 的入口。这个名字有历史原因；在当前
distributed AICore 路径中，AIC/AIV 也会调用它来重放 orchestration，不能仅凭函数名
判断它运行在 AICPU 上。

### 3.2 为什么所有核都重放同一份 orchestration

每个核都执行同样的 C++ 控制流，并用自己的 `local_index++` 生成 task id。若输入和控制
流确定，则所有核看到的逻辑提交序列一致：

```text
core 0: task 0, task 1, task 2, task 3, ...
core 1: task 0, task 1, task 2, task 3, ...
core 2: task 0, task 1, task 2, task 3, ...
```

每个 task 只允许一个合适的核赢得 claim 并真正拥有执行责任。其余核虽然不执行该
task，仍要完成确定性的元数据工作，例如：

- 生成相同 task id；
- 为 `OUTPUT` 计算相同 heap 地址；
- 对 private TensorMap 执行相同 retire/insert；
- 继续重放到下一个提交点。

private 模式的正确性建立在“各核重放完全一致”上。任何依赖本核时间、claim 结果或
未同步输入的控制分支，都可能破坏副本确定性。

### 3.3 三层状态归属

当前核心定义位于
[`state.h`](../src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/common/state.h)。

| 内存/归属域 | 主要结构 | 访问方式 |
|---|---|---|
| 全局共享 | claim cursors、task flags、frontier、heap、fatal | 多核访问，必须满足原子与可见性契约 |
| block 共享 | `BlockWon`/`WonSlot` | anchor 向同一 block 的 follower 投递子任务 |
| 每核私有 | `DistCore::map`、`RingSlot[4]`、`local_index`、`heap_next` | 单 owner 读写，不需要跨核锁 |

“每核私有”是所有权概念，不一定表示对象物理位于芯片私有存储。`DistCore` 数组仍嵌在
全局状态中，只是协议规定每个元素由唯一 worker 维护。

### 3.4 当前重要结构

`DistCore` 中主要有：

- worker 的 role、block 和 lane 信息；
- `local_index`，即本核下一次提交的 task id；
- `heap_next`，确定性 output heap 游标；
- 一个 `DistTensorMap map`；
- 四个 private `RingSlot`；
- task payload 环。

`DistGlobal` 中主要有：

- cube/vector/alloc 三组 sharded claim cursors；
- 每 task 一条 64 字节 `DistTaskCell`，包含完成 flag 和 vector-end 信息；
- contiguous completion `frontier`；
- GM output heap；
- worker/block 布局；
- 每 block 的 `BlockWon`；
- replay/start/fatal 等生命周期状态；
- 所有 `DistCore`。

当前 `DistGlobal` 中**没有**：

- `SharedTensorMap shared_map`；
- `tm_insert_next`；
- `core_progress[]`；
- shared map reclaim/runahead 状态；
- `PTO_DIST_TENSORMAP_MODE` 的运行时分支；
- 集成运行时的 DEPSIG/TMOPS 输出。

这就是“当前仓库是 private 实现”的直接源码依据。

### 3.5 TensorMap 之外，执行中还会动态生成什么

TensorMap 只是调度元数据中的一部分。一次算子从开始重放到所有 kernel 结束，还会生成
或持续推进 task 身份、参数快照、output 地址、claim 结果、依赖边、执行 packet、完成
状态和 heap 水位等数据。

这里的“动态生成”不一定表示每次都调用 `malloc`。当前 runtime 预先保留了
`DistGlobal`、`DistCore`、payload ring、private slot 和 task cell 等容器；执行过程中
主要是在这些有界容器中填入新内容，再在满足生命周期条件后复用 slot。

#### 3.5.1 动态数据总览

| 动态数据 | 归属 | 何时产生或更新 | 主要消费者 |
|---|---|---|---|
| worker 拓扑和启动状态 | 全局 | 本轮 runtime 注册和 worker 启动时 | 所有 worker、AICPU |
| task id | 每核 | 每次 submit 的 `local_index++` | claim、map、slot、完成表 |
| `DistSubmitCtx` | 当前调用栈 | 每次 submit 入口 | 本次提交管线 |
| `DistTaskPayload` | 每核 payload ring | 参数物化时 | winner slot、`block.won` |
| output layout 和 Tensor 元数据 | 每核临时值与 payload | 处理 `OUTPUT` 时 | orchestration、kernel、map |
| heap 虚拟水位 | 每核 | 每次物化新 output 后 | 后续 output 分配、容量反压 |
| claim cursor 和 winner 结果 | 全局 + 本次 ctx | claim 时 | winner/loser 分支 |
| `fanin[]` 依赖边 | winner 的 ctx/slot | winner 查询 owner id 和 map 时 | ready 检查、DEPSIG |
| `RingSlot` 执行 packet | winner/follower 每核 | build 阶段 | 对应 AIC/AIV kernel |
| `WonSlot`/`BuiltSubtask` | block 共享 | 多核 task 由 anchor 发布时 | follower lanes |
| output tensor 数据 | GM heap 或 external buffer | kernel 执行时 | 后继 task、最终用户 |
| `vend`、completion flag | 每 task 全局 cell | task 完成时 | heap 反压、fanin waiter |
| contiguous frontier | 全局 | 完成 flag 出现后协作推进 | 回收、容量判断、结束条件 |
| ring 占用和生命周期计数 | 每核/全局 | 提交、drain、退出时 | 反压和结束判断 |
| trace/dep edge | 可选诊断状态 | tracing 开启时 | swimlane 和依赖分析 |

把它们放到一条数据流上看：

```text
本轮 runtime 配置和 worker topology
  -> orchestration 的 L0TaskArgs
  -> task id + DistSubmitCtx
  -> DistTaskPayload 参数快照
  -> DistOutputLayout + output Tensor 地址/owner id
  -> claim winner + kernel id/lane 信息
  -> owner id/TensorMap lookup 得到 fanin[]
  -> RingSlot，或 WonSlot + 多个 RingSlot
  -> kernel 写实际 tensor 数据
  -> vend + completion flag
  -> frontier、heap 回收判断和 slot 复用
```

TensorMap 位于中间两步：它使用已经物化好的 Tensor 地址查询 fan-in，再登记本 task 将要
写入的地址。它既不保存前面的完整参数，也不保存后面的 kernel output 和完成状态。

#### 3.5.2 每轮运行生成的拓扑和生命周期状态

AICPU 调用 `dist_engine_register()` 时，会为本轮运行写入或重置：

- `heap_base/heap_size`：本轮 GM output heap 的地址和容量；
- `H`：依赖及回收窗口，默认值可由配置覆盖；
- `num_workers/num_blocks`：本轮参与的 worker 和物理 block 数；
- `layout[worker]`：每个 worker 对应的 block 和 AIC/AIV lane；
- `orch_args`：本轮 orchestration 的输入参数入口；
- 三组 claim cursor：重置为 `-1`；
- task cell、frontier、fatal：重置为未完成状态；
- `started_count/replay_done`：重置为 0；`replay_done` 仅保留原有 ABI 位置；
- 固定 G=16 final 树：重置 leaf/root arrival 和 release，并根据本轮
  `layout[]` 写入每个活跃叶组的 worker 数与活跃组数。

这些数据不是每个 task 都重新生成，但它们是每次算子运行的动态上下文。例如同一个程序
用不同 worker 数运行时，`layout[]` 和 `num_blocks` 会随本轮资源重新推导。

worker 进入 `dist_core_main()` 后，先原子增加 `started_count`。所有 worker 到齐后才开始
重放，避免一部分核已经提交很远、另一部分核尚未启动。每个 worker 重放完 orchestration
后原子增加 `final_barrier.leaf_arrivals[block_id % 16]`。每组的静态 AIC 代表在本组
到齐后向 root 转发一次；root 收到全部活跃组后发布全局 release，各组代表
再发布本组 release。等待过程中 worker 仍继续 drain 已经排队的 slot；只有观察到
本组的全局 release，且本核没有待执行 slot/`block.won` 时，才退出。

因此：

```text
started_count 表示“多少 worker 已进入本轮”
final leaf/root/release 表示“所有 worker 是否已生成完全部逻辑 task”
task flags    表示“具体 kernel task 是否执行完成”
```

三者不能互相替代。

#### 3.5.3 task id 与提交期临时上下文

每个 submit 首先生成本核的 task id，再构造一个栈上的 `DistSubmitCtx`。它暂存：

- `self` 和 task id；
- 本 task 的 payload slot；
- tensor 数、output 总字节数和返回值 `TaskOutputTensors`；
- winner 的 `fanin[]`；
- claim 得到的 `won`、`kernel_id`；
- 多核任务的 `joint_block/joint_slot/joint_count`。

`DistSubmitCtx` 只活在这次 submit 调用中。真正需要等到以后执行的数据，必须在函数返回前
复制到 `DistTaskPayload`、`RingSlot` 或 `WonSlot`，不能让执行端引用这个栈对象。

task id 也有两层含义：

- 每核的 `local_index` 是生成器状态，各核各自保存；
- 数值相同的 N 表示同一个逻辑 task，并作为 map producer、flag 下标和 slot 标识。

所以“task id 是每核生成的”和“task N 全局唯一”并不矛盾：前者描述生成方式，后者描述
确定性重放后的逻辑身份。

#### 3.5.4 参数快照 `DistTaskPayload`

orchestration 中的 `L0TaskArgs` 是提交侧描述。runtime 会把本 task 的内容物化到每核
`DistTaskPayload`：

- tensor tag；
- 完整 Tensor 元数据；
- scalar 参数；
- tensor/scalar 数量。

payload 按 `task_id & kTaskPayloadMask` 放入每核有界 ring。private 模式下，同一个逻辑
task 会在每个 worker 上形成一份内容应当相同的 payload 快照。它们不是 kernel 计算结果，
而是后续构建执行 packet 的原材料。

Tensor 元数据至少包含 buffer 地址、shape、stride/extent、dtype、offset 和 owner task id
等。kernel 最终通过这些元数据找到真正的数据，而不是把整块 tensor 数据复制进 payload。

#### 3.5.5 output layout、地址和 `TaskOutputTensors`

遇到 `OUTPUT` 时，runtime 会临时构造 `DistOutputLayout`，计算：

```text
每个 output 的 buffer_size_bytes
每个 output 在本 task packed 区域中的 offset
本 task 所有新 output 的 total_output_size
```

随后使用每核单调增长的 `heap_next` 计算虚拟地址，再对 `heap_size` 取模得到 GM heap 中的
物理位置。必要时会跳到 ring 的下一圈，避免一个 task 的 packed outputs 跨越物理末尾。

由此动态生成两类对象：

1. payload 中可供 kernel 使用的 `Tensor` 元数据；
2. 返回给 orchestration 的 `TaskOutputTensors` 引用。

新 Tensor 会写入 `owner_task_id = N`。后续 `get_ref()` 因而能携带显式 producer。所有核
都独立进行同样计算，所以它们的 `heap_next` 和 output 地址必须一致；真正的 GM output
buffer 仍只有同一片全局物理存储。

`INOUT` 和 `OUTPUT_EXISTING` 使用已有 buffer，不为数据另分配新 heap 区域。它们仍会被
登记为本 task 的写入，因此会产生新的 TensorMap producer 记录。

#### 3.5.6 claim 状态和 winner 元数据

task N 的候选 worker 会更新对应的 sharded cursor。cursor 是跨 task 持续推进的全局状态，
本次 claim 还会在 `DistSubmitCtx` 中生成：

- 本核是否 `won`；
- 选中的 `kernel_id`；
- 是否为多核 joint task；
- joint task 的 anchor block、lane 数和共享 slot。

loser 不会生成可执行主 slot，但它之前已经生成 task id、参数快照和 output 地址，之后还会
更新 private TensorMap。winner 身份只控制“谁负责执行”，不能改变逻辑 task 描述。

#### 3.5.7 `fanin[]`：从 TensorMap 结果变成显式依赖边

TensorMap lookup 返回的是一个 producer id。winner 会把所有输入的 owner id 和 lookup
结果收集、过滤和去重，生成本 task 的 `fanin[]`：

```text
TensorMap entry:  某地址区间 -> producer N
fanin edge:       consumer M -> 等待 producer N
```

map 是跨多个 task 存活的索引；`fanin[]` 是某一个 consumer 的依赖快照。即使后续 map
又插入了同地址的新 producer，已经构建好的 task M 仍按自己 slot 中保存的 fan-in 等待，
不会重新查询 map。

#### 3.5.8 `RingSlot`：真正可执行的 packet

winner 把 payload 和 fan-in 复制到一个 private `RingSlot`。除了 task/kernel 标识，slot
还动态生成：

- tensor/scalar 的执行期副本；
- `args[]`，即传给 kernel ABI 的参数数组；
- `LocalContext` 和 `GlobalContext`；
- AIV1 等 lane 使用的 `sub_block_id`；
- 多核 task 对应的 `won_block/won_slot`；
- `occupied/built` 生命周期状态。

`args[]` 中的 tensor 参数指向 slot 自己的 Tensor 元数据，scalar 直接写入参数值，固定
位置还会写入 local/global context 地址。kernel 调用消费的是这个稳定 packet，而不是
orchestration 的临时 `L0TaskArgs`。

每核只有四个 private slot。slot 满时，worker 会先 drain `block.won` 和 ready slot，形成
自然反压。kernel 完成后清除 `built/occupied`，该物理 slot 才能给后续 task 复用。

#### 3.5.9 多核 task 的 `WonSlot` 和子任务

一个逻辑 task 需要多个 lane 时，anchor 还会动态构建 block 共享的 `WonSlot`：

- 逻辑 task id；
- 参与 lane 数 `remaining`；
- 每个 follower 的 `BuiltSubtask`；
- 每 lane 的 `drained` 状态；
- 整个 deposit 的 `state` 和 block 的 `any_pub`。

每个 `BuiltSubtask` 又包含 kernel id/address、tensor/scalar、fanin 和 sub-block id。follower
观察到发布状态后，把属于自己的 `BuiltSubtask` 再复制到自己的 private `RingSlot`。

所有 lane 执行同一个逻辑 task 的不同 subtask。每完成一个 lane，就原子递减
`remaining`；最后一个 lane 才清空 `WonSlot` 并发布整个逻辑 task 的 completion flag。

#### 3.5.10 真正的 tensor 数据、完成状态和水位

前面大部分都是元数据。kernel 被调用后才会动态产生或修改真正的 tensor 数据：

- `OUTPUT` 写 GM heap 中的新 buffer；
- `INOUT` 原地修改 external 或先前生成的 buffer；
- `INPUT` 只读取，不登记当前 task 为 producer。

kernel 返回后，当前代码更新每 task 的 `DistTaskCell`：

- `vend`：记录完成该 task 时执行核的当前 `heap_next` 水位；
- `flag`：发布 task 已完成。

需要注意，源码是在 task **完成时**写入当前 `heap_next`，不是在 task 提交时保存一份
“该 task 自己 output 的精确结束地址”。后续 heap 容量判断会读取窗口边界 task 的
`vend`。理解或修改回收协议时应依据这一实际写入时机，不能仅凭字段名推断。

任意 worker 随后可扫描连续 ready 的 flags，推进 `frontier`。所以：

```text
flag[N] = 1     说明单个 task N 已完成
frontier = F    说明 [0, F] 整段 task 已连续完成
heap_next       说明某个 worker 已物化到哪个虚拟 heap 水位
vend[N]         是 task N 完成时发布的 heap 水位
```

这些状态共同支持 fan-in 等待、窗口回收和 heap ring 反压，不能只留下 completion flag。

#### 3.5.11 哪些数据有多份，哪些全局只有一份

private 模式下最容易混淆的是“同一个 task 被所有核重放”和“同一个 task 只执行一次”。

| 数据 | 同一逻辑 task 的份数 | 原因 |
|---|---:|---|
| task id 数值 | 每核各生成一次，但数值相同 | 确定性重放 |
| `DistTaskPayload` | 每核一份 | 每核独立走 submit 管线 |
| output Tensor 元数据 | 每核一份，地址应相同 | 确定性 heap 物化 |
| private TensorMap entry | 每核一份 | 保持副本供未来 winner 查询 |
| claim cursor 记录 | 全局一份 | 选出唯一 winner |
| 主 `RingSlot` | 单核 task 通常只有 winner 一份 | 只执行一次 |
| joint follower slot | 每个参与 lane 一份 | 每 lane 执行自己的 subtask |
| 实际 GM output buffer | 全局同一片地址 | 所有元数据指向同一数据 |
| task completion cell | 每逻辑 task 全局一条 | 所有 consumer 等待同一结果 |
| frontier | 全局一个 | 表示全局连续完成前沿 |

这个区分也说明 shared TensorMap 优化的范围：它主要消除“每核一份 map entry”，不会把
payload、执行 slot、真实 output、completion flag 和 frontier 一并替代掉。

## 4. 当前一次 submit 的完整路径

当前 kernel 提交入口在
[`submit_direct.h`](../src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/submit_direct.h)，
通用步骤在
[`submit_core.h`](../src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/submit_core.h)。

### 4.1 第一步：生成 task id

每个核执行：

```text
N = self.local_index++
```

task id 不由 claim winner 分配。它来自确定性重放，所以各核必须在第 N 个提交点得到同样
的 N。

### 4.2 第二步：先帮助推进已经存在的任务

提交新 task 前会：

1. drain 本 block 已发布的 `block.won` 子任务；
2. drain 本核 private ring 中依赖已经满足的 slot。

这是协作式推进。worker 不是只顾提交自己的下一项工作，它也利用每个 submit 点执行
已经 ready 的任务，避免 ring 堵塞。

### 4.3 第三步：物化参数和 output 地址

`INPUT/INOUT` tensor 被复制到本核 task payload。`OUTPUT` 没有现成地址，需要按照 shape
和 dtype 计算大小，再从 GM heap ring 中物化地址。

每核有自己的 `heap_next`，但所有核执行相同的对齐和布局算法，因此同一个 task 的 output
必须得到相同物理地址。这是 MB-6 所要求的“确定性分配”，不是多个核通过一个原子 allocator
抢地址。

新 output 同时记录 `owner_task_id = N`，供后续显式依赖使用。

### 4.4 第四步：推进 private map 的回收窗口

在查询当前 task 前，各核都调用：

```text
dist_tensor_map_advance_retire(self.map, N, H)
```

这会把窗口下界推进到 `N-H`，释放过期 producer entry。

### 4.5 第五步：claim 唯一 owner

所有符合 task 类型的 worker 竞争全局 cursor。cursor 按 `N % 4` 分片，目的是减少所有
task 都争用同一 cacheline 的热点。

概念上，claim 要保证：

```text
同一个 task 恰有一个 winner
不能跳过 task
winner 的核类型能执行该 kernel
```

单 AIC task 可由一个 AIC winner 执行，单 AIV task 可由一个 AIV winner 执行。多核
MIX/双 AIV task 还需要选择 anchor，并通过 `block.won` 组织 follower 子任务。

### 4.6 第六步：只有 winner 查询 fan-in

当前源码在 `is_winner` 分支中执行 `dist_submit_collect_fanin()`。也就是说：

- winner 根据 owner id 和自己的 private map 构建 `fanin[]`；
- loser 不做这次 lookup；
- private 副本一致性仍然必要，因为下一个 task 的 winner 可能是另一个核。

收集结果会去重，并受 `kMaxFanin` 上限约束。

### 4.7 第七步：所有核注册 producer

不论 claim 输赢，所有核都对 `OUTPUT/INOUT/OUTPUT_EXISTING` 执行 private map insert。

顺序必须是“先 lookup 当前输入，再 insert 当前输出”。如果先把 N 写入 map，N 查询自己
的 `INOUT` 时可能得到自依赖。

当前实际顺序是：

```text
claim -> winner lookup -> all-core insert -> winner build
```

claim 放在 lookup 前不会改变依赖图，因为 loser 不需要 fan-in；lookup 和 insert 的相对
顺序才是 TensorMap 语义的关键。

### 4.8 第八步：winner 构建并发布任务

单核任务进入 winner 的 private `RingSlot`。slot 保存：

- task id 和 kernel 地址；
- tensor/scalar 参数；
- fan-in producer 列表；
- local/global context；
- 多核任务需要的 `won_block/won_slot` 信息。

多核任务由 anchor 构建多个 lane 的 `BuiltSubtask`，再通过 `WonSlot.state` 等发布状态让
followers 抽取。最后一个完成的 subtask 负责把整个逻辑 task 标记为完成。

### 4.9 第九步：等待依赖、执行 kernel、发布完成

drain 逻辑检查 slot 的每个 fan-in：

```text
for producer in fanin:
    wait until tasks[producer].flag is ready
```

全部 ready 后才调用 kernel。kernel 返回后：

1. 单核 task 直接发布自己的完成 flag；
2. 多核 task 递减 remaining，最后一个 subtask 发布逻辑 task 的 flag；
3. 任意 worker 可协作推进 contiguous frontier。

TensorMap 只负责生成 producer id。真正的 happens-before 链应当是：

```text
producer 写 output 数据
  -> 数据对 GM 可见
  -> 发布 task completion flag
  -> consumer 观察 flag
  -> consumer 使输入 cache 状态正确
  -> consumer kernel 读取数据
```

其中任一步缺失，都可能出现“依赖图正确，但数值错误或死等”。

## 5. 完整算子对照：BGEMM 的四个 task

本节使用当前仓库的
[`bgemm_orch.cpp`](../examples/a5/fully_distributed_within_core/benchmark_bgemm/kernels/orchestration/bgemm_orch.cpp)
作为完整例子。

### 5.1 Orchestration 做了什么

对每个 group 和每个 `k_idx`：

1. 从 external A/B 创建当前 K 分片的 `A_view` 和 `B_view`；
2. 提交 AIC GEMM，创建临时 output；
3. 把 group 对应的 `C_view` 作为 `INOUT`；
4. 把 GEMM output 作为 ADD 的另一个 input；
5. 提交 AIV ADD，把本轮部分和累积进 C。

简化代码为：

```cpp
params_gemm.add_input(A_view);
params_gemm.add_input(B_view);
params_gemm.add_output(group_ci);
TaskOutputTensors gemm_outs = rt_submit_aic_task(FUNC_GEMM_TILE, params_gemm);

params_add.add_inout(C_view);
params_add.add_input(gemm_outs.get_ref(0));
rt_submit_aiv_task(FUNC_TILE_ADD, params_add);
```

### 5.2 假设一个 group、`grid_k = 2`

逻辑提交序列如下：

| task | 核类型 | 输入 | 写入 |
|---:|---|---|---|
| 0 | AIC | A0, B0 | 新临时 P0 |
| 1 | AIV | C, P0 | 原地更新 C |
| 2 | AIC | A1, B1 | 新临时 P1 |
| 3 | AIV | C, P1 | 原地更新 C |

下面逐个看 TensorMap。

### 5.3 提交 task 0：`GEMM(A0, B0) -> P0`

查询阶段：

- A0/B0 是 external input；假设此前没有 runtime task 写过对应区间，lookup 返回 -1。
- P0 是 `OUTPUT`，不查询旧 producer。

注册阶段：

```text
insert(P0 range -> task 0)
```

map 的逻辑内容变为：

| 区间 | producer |
|---|---:|
| P0 | 0 |

task 0 没有内部 fan-in，可以在 AIC winner 的 slot ready 后执行。

### 5.4 提交 task 1：`ADD(C, P0) -> C`

查询 C：

- C 是 `INOUT`，需要 lookup。
- 这是第一次写 C，假设 C 来自 external 初始值，map 中没有 runtime producer。
- 因此 C 暂不产生内部依赖。

查询 P0：

- `gemm_outs.get_ref(0)` 携带 `owner_task_id = 0`；
- map lookup 也能找到 P0 -> 0；
- fan-in 去重后得到 `[0]`。

注册当前写入：

```text
insert(C range -> task 1)
```

此时 map 为：

| 区间 | producer |
|---|---:|
| P0 | 0 |
| C | 1 |

task 1 必须等待 task 0 完成，然后由 AIV 执行。

### 5.5 提交 task 2：`GEMM(A1, B1) -> P1`

A1/B1 没有内部 producer，task 2 没有 fan-in。注册：

```text
insert(P1 range -> task 2)
```

map 变为：

| 区间 | producer |
|---|---:|
| P0 | 0 |
| C | 1 |
| P1 | 2 |

虽然 task 2 的提交发生在 task 1 后面，它不依赖 task 1，所以 task 0 和 task 2 两个
GEMM 可以并行或交错执行。

### 5.6 提交 task 3：`ADD(C, P1) -> C`

查询 C：

```text
lookup(C) -> task 1
```

查询 P1：

```text
owner_task_id -> task 2
lookup(P1)    -> task 2
去重后保留 task 2
```

所以 task 3 的 fan-in 为 `[1, 2]`。随后注册：

```text
insert(C range -> task 3)
```

lookup 总是选择最大重叠 producer，因此后续读取 C 会得到 task 3，而不是更旧的 task 1。

最终依赖图为：

```text
task 0 GEMM --P0--> task 1 ADD --C--+
                                      +--> task 3 ADD
task 2 GEMM ------------------P1-----+
```

### 5.7 从提交到执行的可能时间线

task id 顺序和执行顺序不是一回事。一种合法时间线是：

```text
时间 ----->

AIC x: claim task 0 | execute GEMM0 | publish flag[0]
AIC y: claim task 2 | execute GEMM1 | publish flag[2]
AIV z: claim task 1 | wait flag[0] | execute ADD0 | publish flag[1]
AIV w: claim task 3 | wait flag[1,2] | execute ADD1 | publish flag[3]
```

如果 GEMM1 很快，flag[2] 可以早于 flag[1] ready。frontier 仍只能从连续完成区间推进：

```text
flag[0] = 1, flag[2] = 1, flag[1] = 0  => frontier 不能越过 1
flag[1] = 1                            => frontier 可连续推进到 3
```

### 5.8 同一个例子在 private 模式中的额外工作

假设有 C 个重放 worker。对四个 task：

- task 0 的 P0 entry 被 C 个 private map 各插入一次；
- task 1 的 C entry 被 C 个 private map 各插入一次；
- task 2 的 P1 entry被 C 个 private map 各插入一次；
- task 3 的 C entry 被 C 个 private map 各插入一次；
- 每个 task 仍只有一个 claim winner 真正构建主任务。

如果逻辑上共有 D 次 producer 注册，private 模式总 insert 工作约为 `C * D`。

private 模式的优势是 map lookup/insert 不发生跨核竞争；代价是内存和重复计算随 worker
数量增长，并且强依赖确定性重放。

### 5.9 同一个例子在理想 shared 模式中应如何工作

理想 shared 模式只有一个全局 map，逻辑 append 顺序必须是：

```text
append task 0: P0 -> 0
append task 1: C  -> 1
append task 2: P1 -> 2
append task 3: C  -> 3
```

无论哪个核先跑到提交点，每个 task 的 producer 记录全局只追加一次，总 insert 工作约为
`D`。但这引入了新的要求：

- append 必须按 task id 严格定序；
- reader 不能看到“未来 task”的 entry；
- ring slot 复用时必须防 ABA；
- 最慢 worker 仍可能查询的 entry 不能提前回收；
- 普通 payload 和 publish flag 必须在 A5 真硬件上正确可见。

shared 模式减少了复制，却把问题从“确定性副本”转化成“并发 publication 协议”。

### 5.10 BGEMM 中其他动态数据如何变化

前面只跟踪了 TensorMap。把同一个 `grid_k = 2` 例子放回完整 runtime 后，每个 task 还会
产生以下动态数据：

| task | output/heap | winner 与 slot | fan-in | 完成时更新 |
|---:|---|---|---|---|
| 0 GEMM | 物化 P0，`owner=0` | AIC winner 的 slot | 空 | P0 数据、`vend[0]`、`flag[0]` |
| 1 ADD | 不新分配，原地 C | AIV winner 的 slot | `[0]` | C 数据、`vend[1]`、`flag[1]` |
| 2 GEMM | 物化 P1，`owner=2` | AIC winner 的 slot | 空 | P1 数据、`vend[2]`、`flag[2]` |
| 3 ADD | 不新分配，原地 C | AIV winner 的 slot | `[1,2]` | C 数据、`vend[3]`、`flag[3]` |

以 task 1 为例，完整的数据演化是：

```text
所有 worker:
  local_index 生成 task id 1
  -> 把 C_view、P0 和 config 复制到各自 payload[1]
  -> output_bytes = 0，heap_next 不因 task 1 增长
  -> 参与 vector cursor claim

AIV winner:
  owner id/TensorMap 查询得到 producer 0
  -> 生成 fanin[0] = 0
  -> 构建 RingSlot(task=1, kernel=ADD, args, contexts, fanin)

所有 worker:
  在各自 private map 中 insert(C -> 1)

执行 AIV:
  等待 flag[0]
  -> ADD kernel 原地修改 GM 中的 C
  -> 发布 vend[1] 和 flag[1]
  -> 尝试推进 frontier
  -> 清空并复用 RingSlot
```

这里 `config` 也是 payload/slot 中动态复制的 INPUT 元数据，但它来自 external buffer，
没有 runtime producer 时不会产生内部 fan-in。P0 则同时带 owner id 和 map 记录，最后只
形成一条去重后的依赖边。

task 0 和 task 2 还会推进每个 worker 的 `heap_next`。虽然只有 AIC winner 真正写 P0/P1，
所有 worker 都会生成相同的 P0/P1 Tensor 地址和 owner id，保证后续不论哪个 AIV 赢得
ADD claim，都能构建一致的任务描述。

随着执行完成，可能出现：

```text
flag[0] = 1, flag[2] = 1, flag[1] = 0
```

此时 P0/P1 数据和两个 task cell 已经动态更新，但 frontier 仍不能越过 0。等 task 1
完成后，frontier 才能连续越过 1 和 2。这个例子说明 TensorMap、实际数据、单 task flag
和全局 frontier 分别描述不同维度，缺少任何一个都不能完整表示算子执行状态。

## 6. shared 对比 checkout 的实现思路

shared 尝试集中在：

```text
../../../../glm/simpler-fully_distributed/
  src/common/runtime/fully_distributed_within_core/dist_engine.cpp
```

它与当前仓库的 split 目录结构不同，因此适合提取协议和测试思路，不适合整文件覆盖。

### 6.1 ring-per-bucket

shared 尝试把 TensorMap 组织成多个 bucket，每个 bucket 是有界 ring。每个 slot 大体包含：

```text
buffer 地址
lo/hi 字节范围
producer task id
seq 发布序号
```

head/tail 单调增长，物理 slot 通过 mask 复用。相比当前 private linked hash map，这种布局
更容易定义有界容量、顺序追加和 ABA 检查，但需要处理并发 publication 和回收。

该 checkout 也让 private/shared 两种模式使用相近的 ring 语义，以减少两套算法的行为
差异。这是值得参考的方向，但不等于当前 private map 必须先整体改写。

### 6.2 `tm_insert_next`：严格按 task id 追加

每个重放 worker 在 task N 都会进入 shared append 流程，但全局只允许一个 appender：

```text
等待 tm_insert_next == N
CAS(N -> BUSY)
赢得 CAS 的核追加 task N 的所有 write entries
release publish tm_insert_next = N + 1
其他核等待 N 的 append 已发布
```

为什么不能“谁先到谁 append”？考虑 core A 已跑到 task 11，core B 还在 task 10。若先追加
task 11 的 C->11，再让 task 10 lookup C，task 10 就可能错误地依赖未来 task 11，甚至形成
环。

严格序列化 append 让 shared map 的逻辑历史与 private 确定性重放一致。

### 6.3 shared lookup 的时间过滤

shared map 可能已经被更快的 core 追加到 task N 之后，因此 task N 查询时不能简单选全表
最大 producer。shared 尝试只接受：

```text
floor <= producer < N
```

- `producer < N` 排除未来 entry 和当前 task 自身；
- `producer >= floor` 排除已退休窗口。

lookup 先 acquire 一次 tail 快照，再以 relaxed 方式扫描候选 slot，并检查 slot 的 `seq`。
这是目标文档所说的“单 acquire lookup + seq ABA 护栏”的来源。

### 6.4 `seq` 为什么用于防 ABA

ring 的物理 slot 会重复使用。假设 reader 记住了物理 slot 7：

```text
第一次: slot 7 表示逻辑位置 7
回收后: slot 7 表示逻辑位置 519
```

仅凭物理索引无法区分这两个时代。`seq` 记录逻辑位置/版本，reader 在读取 payload 前后
核对它，才能确认没有把新旧内容拼在一起。

### 6.5 `core_progress[]`、回收和 run-ahead

shared map 的回收不能只看最快 core 的 task id。只要最慢 core 还可能查询旧 entry，复用
该 slot 就有 use-after-recycle 风险。

shared 尝试让每个 worker 发布 `core_progress[core_id]`，再使用最小进度计算安全回收下界。
同时提供两种反压：

- 全局 task run-ahead：快核不能领先最慢核无限远；
- shared TensorMap run-ahead：append 不能把有界 ring 撑爆或覆盖慢 reader 所需内容。

这是 MB-7 与 MB-5 的交叉部分。shared map 不是只添加一个全局数组；进度、容量和回收
协议必须一起成立。

### 6.6 DEPSIG 与 TMOPS

shared 尝试包含两类重要诊断：

- **DEPSIG**：对 `(consumer task, producer task)` 依赖边生成稳定签名；
- **TMOPS**：统计 map insert/append 等操作次数。

它们分别回答：

```text
private 和 shared 得到的是不是同一张依赖图？
shared 是否真的把 C * D 次复制降成约 D 次追加？
```

只比较最终数值不能完整回答这两个问题。某些任务序列即使漏边，也会因为执行时间碰巧
串行而得到正确数值；环境变量即使被 runtime 忽略，普通 golden 也可能照样通过。

## 7. shared 尝试为什么还不能直接搬过来

### 7.1 模拟器内存模型不等于 A5 cache 行为

host 和 a5sim 更容易呈现类似统一一致内存的效果。真硬件上，控制字段和 payload 可能
经过不同 cache/atomic 路径：

- scalar 普通 store 可能只更新本核 L1 data cache；
- vector TSTORE 已观察到走 L2 write-through，不需要同样的 flush；
- 原子操作可直接作用到共享层，但不会自动替同一协议中的其他普通字段完成写回；
- 不恰当的整 cacheline flush 还可能把本核的陈旧邻居写回，覆盖别的核的更新。

因此“C++ release store 了 `seq`”不能自动推出“先前普通写入的 payload 已经到 HBM”。
语言级 memory order 和设备 cache publication 是两个都要满足的层次。

### 7.2 当前最关键的 compound publication 风险

`SharedRingSlot` 的 payload 字段使用普通 store，而 `seq` 使用 coherent/atomic 路径发布。
已知调试记录指出两类风险：

1. producer 写完 payload 后直接发布 `seq`，payload 可能仍停留在本核 L1；
2. `seq` 操作如果对同 cacheline 做 invalidate，甚至可能丢弃尚未写回的 dirty payload；
3. consumer 原子地观察到新 `seq`，也不表示其本地 payload cacheline 已失效；
4. consumer 可能把新 `seq` 与旧 payload 组合起来。

同类问题也存在于 `block.won`：anchor 普通写 `BuiltSubtask`，再发布 `state`；follower
观察到 state 后读取 payload。控制字段正确不代表整份子任务内容已经可见。

### 7.3 一个可验证的 publication 顺序

具体 primitive 仍应由 A5 probe 结果决定，但协议至少需要表达以下 happens-before：

producer：

```text
1. 写完整 payload
2. 把 payload 所在 cacheline 写回到共享可见层
3. 最后以原子/release 方式发布 seq 或 state
```

consumer：

```text
1. 以原子/acquire 方式观察 seq 或 state
2. invalidate payload 所在 cacheline
3. 读取 payload
4. 必要时重读 seq，确认读取期间 slot 没有被复用
```

布局上还应尽量把 publish control 与普通 payload 分离到明确的 cacheline，避免 flush 或
invalidate 一个字段时破坏同一 line 上的其他 owner 数据。

这里不能简单得出“所有 scalar 读写都到处加 dcci”。本项目已有上板结论：

| 写路径 | 到 HBM 的观察 | dcci 要求 |
|---|---|---|
| scalar store，例如 `out[i] = value` | 不自动到 HBM | writer 需要 `dcci CACHELINE_OUT` |
| vector TSTORE | L2 write-through | 不需要额外 dcci flush |

应当在明确的 publication seam 处理正确范围和方向，而不是在 runtime 任意位置 flush 整个
data cache。

### 7.4 shared checkout 的集成证据边界

对比 checkout 的 [`tests/ONBOARD_ISSUES.md`](../../../../glm/simpler-fully_distributed/tests/ONBOARD_ISSUES.md)
记录了以下状态：

- a5sim 的 vector/mix 路径和若干 host atomic probe 有通过记录；
- 真硬件 `vector_example` 曾以 `507018` 超时；
- crumb 定位到 winner 在 kernel `fn(s.args)` 调用前后，停在调用处未返回；
- plain `BuiltSubtask`/`SharedRingSlot` payload 的 publication 仍被明确列为风险。

这些问题不一定与当前 split 仓的每个失败一一相同，但调试方法值得复用：

1. 先用 crumb 把 hang 缩小到 claim、drain、kernel call 或 completion；
2. 用 skip-kernel/最小 kernel 区分调度死锁和 kernel 参数问题；
3. 用独立 atomic/cacheline probe 验证 primitive，不靠集成测试猜硬件语义；
4. 对 control 与 payload 分别检查 publish/observe；
5. 最后回到集成 private/shared DEPSIG 对照。

因此，共享 checkout 能提供设计和实验素材，但不能把对应提交整体 cherry-pick 后就宣称
shared TensorMap 已完成。

## 8. private 与 shared 的直接对比

| 维度 | 当前 private | shared 目标/尝试 |
|---|---|---|
| map 数量 | 每 worker 一份 | 全局一份 |
| map owner | 单核 owner，无 map 锁 | 多核访问，单序列 appender + 并发 reader |
| insert 次数 | 约 `C * D` | 约 `D` |
| lookup | 本地副本，无跨核竞争 | acquire snapshot + slot/version 检查 |
| 顺序来源 | 所有核确定性重放 | `tm_insert_next` 严格定序 |
| 未来 entry | 本核尚未 insert，天然不可见 | 必须用 `producer < N` 过滤 |
| 回收依据 | 本核 N 与 H | 全核最小 progress、H、ring 容量 |
| ABA 风险 | linked entry 回收，单 owner | ring slot 跨核复用，必须检查 seq |
| 主要优势 | 协议简单、lookup 无共享争用 | 降低内存和重复 insert |
| 主要风险 | 副本漂移、内存/计算放大 | publication、争用、回收、慢核阻塞 |

private 模式不是“临时错误实现”。它是合理的参考语义和回归基线。shared 模式应证明与
private 模式生成同一依赖图，而不是在没有诊断的情况下直接替换 private。

## 9. 对照 atomic minibench 的总体差距

[`atomic_minibench.md`](../../atomic_minibench.md) 定义的是完整并发契约。下表中的“当前”
指 `57841544` 的集成 runtime，不把 standalone mirror UT 当成 runtime 已实现。

| MB | 目标 | 当前 private 仓 | shared 对比 checkout | 还需证明/实现 |
|---:|---|---|---|---|
| 1 | sharded claim 恰一 winner、无跳号 | 有 sharded cursor；host UT 覆盖模型 | 有相关 atomic/probe | 集成压力下的唯一性、无跳号及诊断 |
| 2 | 64B 完成 flag，防邻居 clobber | 有 64B `DistTaskCell`；严格用例缺 DEPSIG | 尝试 atomicMax/coherent seam | 真硬件发布/观察、邻居压力、签名输出 |
| 3 | 多核协作推进 contiguous frontier | runtime 有 frontier；host UT 通过 | 有对应模型/probe | 集成乱序完成与压力证明 |
| 4 | `block.won` 多 lane 协作 | 已实现；严格复验有数值 mismatch | sim 有进展；payload 发布有风险 | 修复复合发布并上板压力通过 |
| 5 | shared map 定序、seq、回收 | **不存在 shared runtime 模式** | 算法基本齐全，真硬件未证明 | 移植到 split 架构并证明等价 |
| 6 | 确定性 GM heap、回收反压 | 每核确定性 `heap_next` 已实现 | 有相应尝试 | wrap/runahead/长序列压力及诊断 |
| 7 | `core_progress[]` 与 run-ahead | 当前 runtime **不存在** | 已实现尝试，主要是 sim 证据 | split 集成、慢核场景和真硬件证明 |
| 8 | `Coherent<T>`/dcci seam | 没有统一 seam；严格数据链失败 | 有 probe；复合 payload 未闭环 | 明确各字段的发布/观察协议 |
| 9 | private map 每核副本确定性 | private insert/retire 路径存在；host mirror UT 通过 | sim 中有对照思路 | 集成 DEPSIG/状态对照和长窗口压力 |

### 9.1 当前严格测试揭示了什么

当前测试在 `57841544` 中增加了
[`_runtime_contract.py`](st/a5/fully_distributed_within_core/atomic_minibench/_runtime_contract.py)，
开始强制检查 DEPSIG、TMOPS 和模式是否真正生效。这改变了“测试通过”的含义。

已执行的严格上板复验呈现为：

| 用例 | 数值/执行现象 | 严格契约结果 |
|---|---|---|
| MB-2 Flags512 | 基础 task/数值可完成 | runtime 未输出要求的 DEPSIG，失败 |
| MB-4 Mix3 | `Vfinal` 数值 mismatch，最大差约 `0.08154` | 失败 |
| MB-5 PrivateReference | private 数值 task 可完成 | 无 DEPSIG；shared 对照不能继续，失败 |
| MB-6 Normal | 基础数值可完成 | 无 DEPSIG，失败 |
| MB-7 Default | 基础数值可完成 | 无 DEPSIG，失败 |
| MB-8 Rounds50 | 实际序列从 `4001` 起，预期从 `4901` 起，最大差 `900` | 失败 |

这与
[`ATOMIC_MINIBENCH_ONBOARD_LOG.md`](ATOMIC_MINIBENCH_ONBOARD_LOG.md)
顶部 `e63f072d` 的历史“全通过”记录并不矛盾：当时主要检查数值，且测试没有强制证明
runtime 识别了 shared mode、DEPSIG 和 TMOPS。那份结果可以作为历史基础运行记录，不能
作为当前完整契约已经通过的证据。

### 9.2 为什么新增严格断言是必要防护

例如测试设置：

```text
PTO_DIST_TENSORMAP_MODE=shared
PTO_DIST_DEPSIG=1
PTO_DIST_OVERHEAD=1
```

如果 runtime 根本没有读取这些变量，普通数值 workload 仍可能使用 private 模式并输出
正确结果。没有 marker/DEPSIG/TMOPS 断言时，测试会把“环境变量被忽略”误判为“shared
模式通过”。

因此 shared 验收至少要同时满足：

1. runtime 明确打印/导出实际模式；
2. private/shared 最终数值都匹配 golden；
3. private/shared DEPSIG 逐位一致；
4. private TMOPS 约为 `C * D`，shared 约为 `D`；
5. sim 与真硬件都覆盖目标要求的场景；
6. 慢核、ring wrap、slot reuse 和 cacheline 邻居压力不失败。

## 10. 建议的收敛顺序

### 阶段 1：先建立可信观测

在改变 map 算法前，先让当前 split runtime 输出：

- 实际 TensorMap mode；
- DEPSIG；
- TMOPS；
- 必要的 claim/flag/frontier/`block.won` 诊断。

原因很直接：没有这些观测，后续无法判断 shared 分支是否真的执行，也无法区分“结果碰巧
正确”和“依赖图完全等价”。

### 阶段 2：先修 shared map 依赖的基础 seam

优先闭环 MB-2、MB-4 和 MB-8：

- 完成 flag 的原子发布和观察；
- 64 字节隔离与邻居 clobber 压力；
- `block.won` 的 payload-before-state publication；
- scalar store 的精确 dcci flush；
- consumer 读取前的精确 invalidate；
- 禁止会回写陈旧控制状态的 entire-cache flush。

shared TensorMap 的 slot publication 与 `block.won` 属于同一类“复合对象发布”问题。
先把更小的 MB-4 协议做对，能降低直接调试 shared map 的变量数量。

### 阶段 3：在当前 split 架构中加入 shared 状态

以当前仓库作为集成基线，逐项引入，而不是复制 monolithic 文件：

1. 在 common state 中加入有界 `SharedTensorMap`；
2. 加入 `tm_insert_next` 和 mode 配置；
3. 保留当前 private 路径作为 reference；
4. 把 lookup/insert 抽到明确的 mode seam；
5. 实现按 task id 的 exactly-once append；
6. 对 payload、seq、head/tail 明确 cacheline 布局和 publication primitive。

这样可以复用当前已经拆分好的 AICore/AICPU 生命周期与 submit 管线，避免把 shared checkout
中与本任务无关的历史调试改动一起带入。

### 阶段 4：加入 progress、回收和反压

shared map 正常跑几个 task 还不够。随后需要：

- 每核发布 `core_progress[]`；
- 使用 min progress 计算安全 reclaim floor；
- 限制普通 task run-ahead；
- 限制 TensorMap append run-ahead；
- 在 ring 容量不足时可靠等待，而不是覆盖 live slot；
- 对 stalled/slow core 运行长序列压力测试。

### 阶段 5：分层验证

建议按以下顺序推进：

| 层次 | 主要回答的问题 |
|---|---|
| host algorithm UT | hash/ring、顺序、回收、ABA 的纯算法是否正确？ |
| A5 atomic/cache probe | atomic、flush、invalidate、cacheline 布局的硬件事实是什么？ |
| a5sim ST | 集成控制流、mode 分支和基础数值是否正确？ |
| A5 onboard ST | 真硬件 publication、并发和 kernel 数据链是否正确？ |
| private/shared differential | 两种模式是否生成完全相同的依赖图与数值？ |

每层只能证明它负责的部分。host mirror 的 MB-5 通过，不能替代集成 runtime 的 shared
路径；上板数值通过，也不能替代 DEPSIG/TMOPS。

## 11. 阅读源码的推荐顺序

第一次阅读时，建议不要从 shared 的三千多行 monolithic 文件开始。按以下顺序更容易
建立完整模型：

1. [`bgemm_orch.cpp`](../examples/a5/fully_distributed_within_core/benchmark_bgemm/kernels/orchestration/bgemm_orch.cpp)
   看一个算子如何表达 INPUT、OUTPUT 和 INOUT。
2. [`submit_direct.h`](../src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/submit_direct.h)
   看一次 submit 的固定步骤。
3. [`submit_core.h`](../src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/submit_core.h)
   看物化、fanin、producer 注册、slot 和完成路径。
4. [`tensor_map.h`](../src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/tensor_map.h)
   看 private lookup/insert/retire。
5. [`state.h`](../src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/common/state.h)
   把每核和全局状态对应起来。
6. [`core_main.h`](../src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/core_main.h)
   与
   [`aicpu_executor.cpp`](../src/a5/runtime/fully_distributed_within_core/aicpu/aicpu_executor.cpp)
   看 AICPU/AICore 生命周期边界。
7. [`atomic_minibench.md`](../../atomic_minibench.md) 按 MB-1 到 MB-9 检查每个并发契约。
8. 最后阅读 shared checkout 的 `dist_engine.cpp` 和 `tests/ONBOARD_ISSUES.md`，对照本文的
   sequencer、seq、progress 和 publication 问题。

## 12. 常见问题

### 12.1 当前 TensorMap 是 AICPU 和 AIV 各持有一份吗？

不是按“AICPU 一份、AIV 一份”划分。当前 map 属于每个 distributed worker 的
`DistCore`，AIC 和 AIV worker 各有自己的 private 副本。AICPU 负责 setup/wait，不在
每个 task 的 orchestration 中维护一份对等 TensorMap。

### 12.2 当前 AICPU 与 AIV 在任务执行过程中完全不交互吗？

逐 task 调度基本不交互，但不能说整个生命周期完全无交互。AICPU 会初始化全局状态、
发布启动条件、等待 worker 结束并处理结果；AIC/AIV 在这段期间自行重放和协作调度。

### 12.3 为什么 loser 也要更新 private TensorMap？

因为 task N 的 winner 和 task N+1 的 winner 可能不是同一个核。若 loser 跳过 N 的
insert，它下次成为 winner 时会用缺失历史的 map 计算 fan-in。

### 12.4 shared 模式为什么还要求所有核到达 append 流程？

所有核仍要保持相同 task 序列，并且任意到达的核都可能负责 task N 的 exactly-once
append。sequencer 决定谁写一次，其他核确认 append 已发布后才能继续，不能各写一份。

### 12.5 有 task completion flag 后，为什么还需要 TensorMap？

flag 只回答“task 17 完成了吗”，并不知道当前 consumer 应该等待 task 17 还是 task 23。
TensorMap 先找出 producer id，flag 再等待对应 producer。

### 12.6 有 TensorMap 后，为什么仍可能数值错误？

TensorMap 只能建立逻辑边。producer 的输出必须先对 GM 可见，再发布完成；consumer 观察
完成后还要以正确方式读取。A5 scalar L1 数据未 flush 就是典型反例。

### 12.7 能否把 shared checkout 的提交全部搬到当前仓？

不建议整体搬。两边架构已经不同，shared checkout 还混有 onboard 调试修改和未闭环的
publication 风险。应复用明确的协议和小型 probe，把功能逐项落到当前 split ownership
边界中。

### 12.8 private 最终是否一定要删除？

不应该在 shared 完成前删除。private 是行为 reference、差分测试基线和故障回退路径。
只有 shared 在数值、DEPSIG、TMOPS、压力和真硬件上都满足目标后，才有条件讨论默认模式。

## 13. 从测试描述到结果回传的完整业务流程

前文从 `dist_engine` 内部解释了 task。对调度开发者而言，还需要知道一个算子如何从
Python/测试描述进入芯片，以及执行完的结果如何回到调用者。本节把仓库上下游连起来。

### 13.1 先确认 runtime 边界

当前 `src/a5/runtime/` 实际并存三种实现：

| runtime | 图/依赖主要在哪里生成 | 谁做逐 task 调度 |
|---|---|---|
| `host_build_graph` | Host 显式建图 | AICPU/执行框架消费 host 图 |
| `tensormap_and_ringbuffer` | AICPU orchestration + TensorMap | AICPU scheduler |
| `fully_distributed_within_core` | 每个 AIC/AIV SPMD 重放 | AIC/AIV 自己 claim/执行 |

本文只解释第三种。它复用了第二种的 Tensor、Arg、Callable 和部分 host/platform 基础设施，
所以文件名、注释或 API 里仍可能出现 PTO2、TRB、AICPU orchestration 等历史术语。复用
类型不等于复用执行流程，必须以当前 runtime 的实际调用链判断 owner。

仓内旧的 `src/a5/docs/runtimes.md` 当前只列出前两种，不能作为本 runtime 是否存在或如何
执行的完整目录。`src/a5/runtime/fully_distributed_within_core/` 和本文记录的实际源码链路
是当前判断依据。

### 13.2 再看全链路

```text
SceneTestCase.CALLABLE / CASES
  -> 编译 orchestration、incore 和 runtime 三类产物
  -> 组装并注册 ChipCallable
  -> Worker/ChipWorker 初始化 device 和执行器
  -> 每轮生成 host tensor/scalar 与 golden
  -> host runtime 做 H2D、构建 Runtime 和预制 arena
  -> DeviceRunner 启动 AICore 与 AICPU
  -> AICPU/AICore 完成一次性 handshake
  -> 所有 AIC/AIV 进入 dist_core_main
  -> SPMD 重放 orchestration
  -> submit/claim/TensorMap/slot/kernel/completion
  -> drain、worker done、AICPU shutdown
  -> host stream 同步
  -> D2H 回传 OUT/INOUT
  -> Python 与 golden 比较
```

可以把它分成四个层次：

| 层次 | 主要问题 | 典型代码 |
|---|---|---|
| 业务描述 | 算子有哪些输入、kernel 和任务关系？ | example/test、orchestration |
| 构建与注册 | 源码如何变成 callable 和 AICore 镜像？ | `scene_test.py`、builder |
| 设备生命周期 | 内存、stream、AICPU/AICore 如何启动？ | host/platform、executor |
| 分布式调度 | task 如何认领、等待和执行？ | `dist_engine` |

修改调度代码时，先确定问题属于哪一层。比如“TensorMap 漏边”和“顶层 OUT 被错误标成
IN，导致没有 D2H”都表现为 golden mismatch，但根因完全不同。

### 13.3 SceneTest 描述业务和验收

以
[`test_benchmark_bgemm.py`](../examples/a5/fully_distributed_within_core/benchmark_bgemm/test_benchmark_bgemm.py)
为例，`SceneTestCase` 主要声明：

- `CALLABLE.orchestration`：编排源码、入口符号和 chip 级参数方向；
- `CALLABLE.incores`：每个 `func_id` 对应的 kernel 源码、核类型和签名；
- `CASES`：平台、`block_dim`、AICPU 线程数和业务参数；
- `generate_args()`：生成本轮 host tensor/scalar；
- `compute_golden()`：在 host 上生成期望结果。

[`scene_test.py`](../simpler_setup/scene_test.py) 会把 test args 转成
`ChipStorageTaskArgs`，并根据 orchestration 的 chip 级签名选出最终需要比较的 OUT/INOUT。
golden 在执行前基于输入 clone 计算，因此测试数据、设备实跑结果和期望值相互独立。

这里的 `level=2` 表示一次本地 chip callable 执行。仓库还有 L3 及更高层 Worker/DAG
流程，但它们不是本文 A5 fully distributed 的芯片内调度主线。开发本 runtime 时，应先
用 L2 用例隔离芯片内问题，再考虑上层多 device 编排。

### 13.4 编译出 `ChipCallable`

`_compile_chip_callable_from_spec()` 会：

1. 编译 orchestration 源码；
2. 按 AIC/AIV 类型分别编译每个 incore kernel；
3. 用 `func_id` 把 child `CoreCallable` 放入 `ChipCallable`；
4. 保存 chip 入口签名、orchestration 符号名和 child kernel 信息；
5. 为 A5 fully distributed 构建本 callable 专用的 AICore 镜像。

同一 pytest session 会缓存编译后的 callable，避免每个 case 重复编译。缓存 key 包含测试
类、平台和 runtime；AICore override 还会对 orchestration/incore 源文件内容做 fingerprint。

### 13.5 Worker 初始化

[`Worker._init_level2()`](../python/simpler/worker.py) 使用
[`RuntimeBuilder`](../simpler_setup/runtime_builder.py) 找到对应平台和 runtime 的：

- host runtime；
- AICPU executor；
- AICore executor；
- onboard dispatcher 或 sim context；
- 当前 callable 的 AICore override。

随后 `ChipWorker.init()`：

1. 预加载统一日志库；
2. sim 平台预加载 sim context；
3. 加载 host runtime；
4. onboard 初始化 device、stream、dispatcher 和 AICPU runtime SO；
5. 缓存 AICPU/AICore 执行器，供多轮 run 复用。

Worker 与 device/runtime 在初始化后绑定，不能在同一个对象上随意切换到另一张卡或另一种
runtime。callable 则可以通过 handle 注册、复用和注销。

### 13.6 Callable prepare 与 run 是两阶段

`worker.register(callable)` 最终进入 `prepare_callable()`。prepare 阶段只处理与本轮参数
无关的内容：

- 上传或映射 child kernel；
- 建立 `func_id -> CoreCallable` 地址表；
- 缓存 callable 签名；
- 保存 orchestration 相关元数据；
- 对相同二进制做 hash 去重。

每次 `worker.run(handle, args, config)` 才处理本轮变化的数据：

- tensor/scalar 实参；
- `block_dim` 和 AICPU thread 数；
- GM heap 大小；
- profiling/trace 配置；
- 每轮 Runtime、DistGlobal 和完成状态。

这个分层解释了为什么同一 callable 连续跑多轮时，不需要反复上传 kernel binary，但必须
重新 H2D 输入、重置 task flags 和重新绑定本轮 runtime arena。

### 13.7 Host 绑定本轮 tensor 和 arena

[`bind_callable_to_runtime_impl()`](../src/a5/runtime/fully_distributed_within_core/host/runtime_maker.cpp)
执行本轮 host 侧绑定：

1. 遍历 chip 级 tensor 参数；
2. 为普通 host tensor 分配 device buffer；
3. 按参数方向执行 H2D 或 device memset；
4. 用 device 地址重写 Tensor 描述符；
5. 保存 D2H ledger；
6. 准备 pooled GM output heap；
7. 在 host mirror 上预构建 `PTO2Runtime + DistGlobal` arena；
8. 把整个 arena 一次 H2D 到 device；
9. 把本轮 device args 和 arena 地址写入 host `Runtime`。

此时还没有任何 orchestration task 被提交。host 只完成数据和控制结构的准备。

### 13.8 DeviceRunner 组织一次芯片运行

onboard 的
[`DeviceRunner::run()`](../src/a5/platform/onboard/host/device_runner.cpp)
会：

1. 校验或自动解析 `block_dim`；
2. 计算 worker 数并初始化 handshake 数组；
3. 准备 AICore register 地址和 AICPU affinity；
4. 初始化可选 DFX buffer；
5. 把 host `Runtime` 再 H2D 成 device `Runtime`；
6. 准备 `KernelArgs`；
7. 先启动 AICore kernel；
8. 再启动 AICPU run kernel；
9. 等待 AICPU/AICore 两条 stream 完成；
10. 收集 profiling、handshake 和 device wall time。

A5 一个 `block_dim` 对应一组 `1 AIC + 2 AIV`。因此当前代码中：

```text
num_workers = block_dim * 3
num_AIC     = block_dim
num_AIV     = block_dim * 2
```

`block_dim=24` 对应 24 个 AIC worker 和 48 个 AIV worker，共 72 个 SPMD
参与者。它不是 orchestration 生成的 task 数。

### 13.9 为什么先启动 AICore，再启动 AICPU

这不是正确性要求，因为 handshake 允许任一侧先到；它是 A5 当前实现的重要启动优化。
首次加载 AICore image 可能有明显延迟。若先让 AICPU 占住设备并开始 spin，再触发首次
AICore load，曾观测到启动大幅变慢并接近 op timeout。

因此 host 先提交 AICore kernel，让 worker 等待 `aicpu_ready`，再启动 AICPU。修改 launch
顺序时不能只看逻辑等价，还要评估首次加载和 timeout 行为。

### 13.10 AICPU 做一次性控制面工作

[`aicpu_executor.cpp`](../src/a5/runtime/fully_distributed_within_core/aicpu/aicpu_executor.cpp)
中的 AICPU 流程为：

1. 与所有 AICore 完成物理 core id/register handshake；
2. 从本轮 `Runtime` 复制 orchestration entry args；
3. 定位预构建的 `PTO2Runtime`；
4. 调用 `dist_engine_register()` 重置 cursor、flag、frontier 和拓扑；
5. flush 全局状态；
6. 把每个 worker 的 `aicpu_ready` 设为 `DIST_RUN`；
7. 等待所有 worker 完成；
8. 触发 teardown/EXIT。

配置中的多个 AICPU thread 沿用了公共 executor/affinity 框架。当前 fully distributed
路径只有一个 thread 执行上述 setup/wait；其余被称为 scheduler 的 thread 只是等待
`runtime_done_`，**不做逐 task 调度**。

### 13.11 AICore 进入分布式主循环

每个 AIC/AIV 在
[`aicore_execute()`](../src/a5/runtime/fully_distributed_within_core/aicore/aicore_executor.cpp)
完成 handshake 后等待 `AICPU_READY_DIST_RUN`，然后只调用一次：

```text
dist_core_main(runtime, core_idx, core_type)
```

`dist_core_main()`：

1. 通过 `Runtime::dist.shared_addr` attach 到 `DistGlobal`；
2. 从 `layout[core_idx]` 得到 block/lane；
3. reset 本核 `DistCore`；
4. 参与 `started_count` 启动屏障；
5. 重放完整 orchestration；
6. 到达固定 G=16 final 树，并等待全局 release；
7. drain 本核和 `block.won` 的剩余任务；
8. 发布 worker done；
9. 返回 AICore executor，等待 AICPU EXIT。

这一步之后才进入前文介绍的 submit/claim/TensorMap/slot 流程。

### 13.12 执行完成和结果回传

所有 worker 完成后：

1. AICPU 结束等待并向每个 AICore 发送 EXIT；
2. AICore 在 register 协议上确认 EXITED；
3. host 的 AICPU/AICore stream sync 返回；
4. `validate_runtime_impl()` 按 ledger 把 OUT/INOUT D2H；
5. 每轮普通 top-level tensor device allocation 被释放；
6. callable binary 保留到 unregister/finalize，pooled arena 保留到 Worker finalize；
7. Python 比较实际输出和 golden。

所以 golden mismatch 已经位于整个链路最后。它可能来自构建、H2D、调度、kernel、cache
publication 或 D2H，不能默认归咎于 TensorMap。

## 14. 构建产物、Callable 与 `func_id`

### 14.1 Runtime 本身有三个独立程序

[`build_config.py`](../src/a5/runtime/fully_distributed_within_core/build_config.py)
分别构建：

| 目标 | 运行位置 | 本 runtime 的职责 |
|---|---|---|
| host | Host CPU | device 数据、arena、stream 和 callable 生命周期 |
| aicpu | AICPU | handshake、状态发布、等待、teardown |
| aicore | AIC/AIV | orchestration 重放、调度和 kernel 执行 |

三者独立编译，靠共享 struct ABI、device pointer 和 handshake 协议连接。修改一个跨边界
struct 时，不能只重编其中一个目标。

### 14.2 `ChipCallable` 是什么

[`callable.h`](../src/common/task_interface/callable.h) 中：

- `CoreCallable` 保存一个 leaf kernel 的签名、二进制和 resolved address；
- `ChipCallable` 保存 chip 级入口签名、orchestration artifact，以及多个 child
  `CoreCallable`；
- 每个 child 通过 `func_id` 标识。

orchestration 提交的不是 C++ 函数指针，而是稳定 `func_id`：

```cpp
rt_submit_aic_task(FUNC_GEMM_TILE, args);
```

构建系统和运行时必须对 `FUNC_GEMM_TILE` 的数值达成一致。以下错误都可能让 kernel 调错：

- `CALLABLE.incores[].func_id` 与 orchestration 宏不一致；
- 同一 callable 内重复 func id；
- kernel 核类型与提交 API 不匹配；
- AICore override 没有包含最新 kernel source；
- 复用了错误平台/runtime 的缓存产物。

### 14.3 A5 onboard 的 linked dispatch

对 A5 onboard，`scene_test.py` 会生成一个 wrapper，把各 AIC/AIV kernel source 以重命名
入口包含进专用 AICore image，并定义：

```text
pto_call_linked_kernel_aic(func_id, args)
pto_call_linked_kernel_aiv(func_id, args)
```

orchestration source 也链接进同一个 image。CCEC 路径的 slot 中
`function_bin_addr` 可以为 0，执行时根据当前核类型和 `func_id` 调 linked wrapper。

因此真硬件调试时，真正执行的是 per-callable AICore override，不是单独 child binary
地址跳转。只检查 `Runtime::func_id_to_addr_` 不能证明 onboard 调到了哪个链接符号。

### 14.4 a5sim 的函数地址 dispatch

a5sim 仍把 orchestration 链接进 AICore sim image，但 child kernel 通过 sim loader 解析为
host function pointer。`resolve_kernel_addr()` 从 `CoreCallable::resolved_addr()` 取地址，
`execute_slot()` 再调用该函数。

所以 sim 和 onboard 共享 task 语义，却不共享最终 kernel dispatch 机制：

| 平台 | orchestration | incore dispatch |
|---|---|---|
| a5sim | 链接进 AICore sim image | `resolved_addr()` function pointer |
| a5 | 链接进 AICore image | linked wrapper 按 `func_id` 分派 |

这也是“sim 能调用，onboard 卡在 kernel entry”时必须检查链接/ABI，而不只检查调度图的
原因。

### 14.5 Standalone orchestration SO 的历史兼容层

公共 `ChipCallable` 和 DeviceRunner 仍有 standalone orchestration SO、symbol name 和
callable cache 的兼容字段。当前 distributed 执行路径不让 AICPU 调用该 SO；AICore
直接调用链接进 image 的 `aicpu_orchestration_entry`。

因此源码中出现 `dev_orch_so`、`func_name` 或旧注释，不表示当前 orchestration 仍运行在
AICPU。判断执行位置要沿 `aicore_execute -> dist_core_main -> direct_replay_orch` 看实际调用。

### 14.6 改什么需要重建什么

| 修改内容 | 至少需要更新的产物 |
|---|---|
| `dist_engine`/runtime C++ | 重新构建 runtime host/AICPU/AICore |
| orchestration source | 重新编译 callable，并重建 AICore override |
| incore source | 重新编译 child，并重建 onboard linked wrapper/image |
| shared ABI header | 所有引用该 header 的独立程序 |
| Python test only | 通常无需重编 baseline runtime |

仓库安装不会在每次 Python import 时自动保证 C++ binary 与 source 同步。出现“代码看起来
已经改了，行为还是旧的”时，第一项检查应是实际加载的 `build/lib` 和 AICore override
是否由当前 source 生成。

## 15. 内存、地址和数据搬运

### 15.1 不要把 `Runtime`、`PTO2Runtime` 和 `DistGlobal` 混为一个对象

当前链路中有三个相关但职责不同的对象：

| 对象 | 主要内容 | 主要使用者 |
|---|---|---|
| `Runtime` | handshake、worker 数、entry args、func 表 | host、AICPU、AICore entry |
| `PTO2Runtime` | GM heap 和 DistGlobal 指针的轻量 header | AICPU setup |
| `DistGlobal` | cursor、flags、frontier、blocks、cores | 分布式调度热路径 |

指针关系可简化为：

```text
KernelArgs.runtime_args ----------> device Runtime
device Runtime.dist.shared_addr --> device DistGlobal
device Runtime.prebuilt arena ----> device PTO2Runtime
device PTO2Runtime.dist_global ---> 同一个 device DistGlobal
device PTO2Runtime.gm_heap -------> pooled GM intermediate heap
```

host 先在 mirror arena 中写好 `PTO2Runtime + DistGlobal` 的初始镜像，再一次 H2D。AICPU
setup 根据 `Runtime` 中记录的 arena 地址找到它，之后 AICore 通过 `shared_addr` attach。

### 15.2 五类主要 device 内存

| 内存 | 内容 | 生命周期 |
|---|---|---|
| top-level tensor buffers | 用户输入、最终输出 | 每轮 bind 到 validate |
| callable/kernel buffers | child callable、兼容 orch artifact | callable/Worker 生命周期 |
| pooled GM heap | orchestration 内 `OUTPUT` 中间结果 | Worker pool，内容按运行重用 |
| pooled runtime arena | `PTO2Runtime + DistGlobal` | Worker pool，每轮覆盖初始镜像 |
| device `Runtime/KernelArgs` | 本轮启动与 handshake | 每轮 run |

另有 profiling、register address 和 device-wall 等辅助 buffer，它们不属于 TensorMap 或
算子数据本身。

### 15.3 Chip 级签名控制 H2D/D2H

`CALLABLE.orchestration.signature` 的方向用于 host/device 数据搬运：

| `ArgDirection` | run 前 | run 后 |
|---|---|---|
| `IN` | H2D | 不 D2H |
| `OUT` | device memset 为 0 | D2H |
| `INOUT` | H2D | D2H |

这层签名错误会直接破坏数据：

- 把真正输入写成 OUT，会用 0 覆盖 device 初始内容；
- 把真正输出写成 IN，执行后不会复制回 host；
- 漏掉 INOUT，会丢失初值或最终值。

`child_memory` tensor 是特殊的已分配 device memory，host bind 直接透传地址，不再为它做
普通 H2D/D2H ledger。

### 15.4 Top-level OUT 与中间 `OUTPUT` 不是一回事

chip 入口的 OUT 是调用者已经提供的 host tensor，host runtime 为它分配 device buffer，
并在 run 前清零。

orchestration 内 `L0TaskArgs.add_output(TensorCreateInfo)` 创建的是 GM heap 中的中间 buffer。
当前 materialize 只计算地址和 Tensor 元数据，**不会自动清零整块中间 output**。kernel
必须完整写入它后 consumer 才能读取；若算法依赖初值，应显式生成该初值，不能依赖 pooled
heap 的旧内容。

中间 output 也不会自动 D2H。只有最终写入 chip 入口 OUT/INOUT 的数据会由 host ledger
回传。

### 15.5 地址有“虚拟水位”和“物理 ring 位置”

`heap_next` 单调增长，用于保持确定性和计算 live span；实际 GM 地址使用：

```text
physical = virtual_heap_offset % heap_size
```

因此两个不同 task 在足够长的运行中可能复用同一个物理 heap 区间。正确性依赖 frontier、
H、vend 和容量反压保证旧 consumer 已经不再需要该区间。

shared TensorMap 开发不能只验证 map slot 回收，还要验证 map entry 的生命周期与 GM data
区间复用保持一致。

### 15.6 Cache 是地址语义的一部分

“两个 Tensor 指向同一个 GM 地址”只表示逻辑地址相同，不表示两个核此刻看到相同字节。
真硬件还需要考虑：

- producer 的写停在哪级 cache；
- publish 前是否写回；
- consumer 观察 flag 后是否 invalidate；
- control 和 payload 是否共享 cacheline；
- flush 是否会把陈旧邻居写回。

所以内存生命周期图必须同时画 data ownership 和 publication edge。单纯把 pointer 放入
shared struct 不是跨核发布。

## 16. Orchestration 参数和任务类型

### 16.1 两层“参数方向”服务不同目的

新手最容易混淆这两层：

| 层次 | API | 用途 |
|---|---|---|
| chip 入口 | `ArgDirection::IN/OUT/INOUT` | host H2D/D2H 和测试输出选择 |
| task 内部 | `add_input/output/inout/...` | 分配、TensorMap 和 fan-in 推导 |

chip 入口把 C 标成 OUT，不会自动让 orchestration 中每个 kernel 都把 C 当 output。每次
L0 submit 仍要正确写 `add_inout(C_view)` 或 `add_output(C_view)`。

### 16.2 当前 L0 tag 的精确行为

以当前 `dist_submit_collect_fanin()` 和 `dist_submit_register_outputs()` 为准：

| tag | owner id fan-in | map lookup | 新分配 | 注册 producer |
|---|---|---|---|---|
| `INPUT` | 是 | 是 | 否 | 否 |
| `OUTPUT` | 否 | 否 | 是 | 是 |
| `INOUT` | 是 | 是 | 否 | 是 |
| `OUTPUT_EXISTING` | 是 | 否 | 否 | 是 |
| `NO_DEP` | 是 | 否 | 否 | 否 |

`OUTPUT_EXISTING` 是 write-only existing tensor。当前不查询 overlap map，只依赖 tensor 自带
creator。如果它指向没有 owner id 的 external buffer，runtime 不会自动等待该 buffer 的
上一个 map producer。需要根据业务真实的 read-before-write 语义选择 INOUT，而不是因为
“都是写输出”随意互换。

### 16.3 Tensor view 只改元数据

`Tensor::view/transpose/permute/slice/reshape` 不复制数据。它们主要改变：

- `start_offset`；
- shape 和 stride；
- contiguous/extent 缓存；
- 可选 `manual_dep`。

view 通常保留 parent 的 buffer 和 owner task id。因此 runtime 可以同时通过 creator 和
地址 overlap 找依赖。多个 view 是否冲突由实际 byte range 决定，不由 C++ 变量名决定。

### 16.4 参数构建的基本约束

- tensor 参数必须先于 scalar 添加；
- `L0TaskArgs` 保存的是 Tensor/TensorCreateInfo 指针，源对象至少活到 submit 完成物化；
- scalar 在 slot 中按 64 位 payload 保存，kernel 两侧必须按同一位级语义解释；
- tensor 参数顺序必须与 kernel ABI 一致；
- `TaskOutputTensors.get_ref()` 只能对具名 lvalue 调用，避免立刻悬空；
- fan-in 上限、tensor/scalar 上限不是动态 vector，超限必须显式处理。

### 16.5 普通 kernel task

普通 `rt_submit_aic_task/rt_submit_aiv_task` 走前文九步 submit：物化、claim、fanin、map
注册、slot、执行和 completion。它是 BGEMM 中 GEMM/ADD 的路径。

### 16.6 `alloc_tensors` 是一个没有 kernel 的逻辑 task

`alloc_tensors(args)` 仍然消耗 task id，并：

1. 在所有核上确定性物化 output 地址；
2. 在 private map 中注册 output producer；
3. 用独立 `alloc_cursor` 选一个 winner；
4. winner 做 heap 容量检查；
5. winner 直接发布该 alloc task 完成，不构建 kernel slot。

后续 kernel 使用 allocation 返回的 Tensor 时，会把 alloc task 的 owner id 纳入 fan-in。
这让“Tensor 描述符已经创建”也进入统一 task 时序。

[`submit_dependency_smoke`](../examples/a5/fully_distributed_within_core/submit_dependency_smoke)
中的 `kernels/orchestration/submit_dependency_orch.cpp` mode 7/8 展示了 allocation、
descriptor 传递和后续 INOUT 使用。

### 16.7 MIX task 是一个逻辑 task、多个 lane

`MixedKernels` 最多声明 AIC、AIV0、AIV1 三个 subtask。当前 direct runtime：

- 选择一个物理 block；
- 第一个 active lane 作为 anchor；
- anchor 和 followers 各得到自己的 `RingSlot`；
- 所有 lane 共享同一个逻辑 task id 和 fan-in；
- `remaining` 归零后只发布一个 completion flag。

共享同一份参数不等于 runtime 自动隔离输出。各 lane kernel 必须按约定只写自己的结果
区域，否则会形成真实 data race。`mix_coown` 中 AIC、AIV0、AIV1 各自写 Cmm、V0、V1，
就是业务层对共享参数的分工。

### 16.8 Orchestration 内读取 tensor 数据

orchestration 有 `get_tensor_data/set_tensor_data` API，可用于根据 tensor 内容决定循环或
索引。paged attention 会读取 external `context_lens`。

这类读取比读 shape/scalar 风险更高：所有 worker 必须读到同样值，否则 orchestration
分支和 task id 序列会分叉。当前 onboard 的依赖等待/cache 语义还不完整，详见第 19 节。
开发新的数据依赖控制流前，应先证明读取对象在所有核上已发布且不可变。

## 17. Handshake、完成层次和退出

### 17.1 启动 handshake 不是逐 task dispatch

AICPU 和每个 AICore 的 `Handshake` 是 64 字节对齐结构。启动过程大致为：

```text
AICPU: aicpu_ready = HANDSHAKE
AICore: 上报 physical_core_id，aicore_regs_ready = 1
AICPU: 初始化该 core registers，aicpu_regs_ready = 1
AICore: 上报 core_type，aicore_done = core_id + 1
AICPU: 初始化 DistGlobal，aicpu_ready = DIST_RUN
AICore: 进入 dist_core_main
```

旧 runtime 的 `Handshake.task` 字段仍保留兼容 payload 指针，但 distributed phase 不用它
逐 task 投递。运行期 task 流量走 GM 中的 cursor/map/slot/flag。

### 17.2 仓内有多层“完成”

| 完成状态 | 回答的问题 |
|---|---|
| `tasks[N].flag` | 逻辑 task N 的所有必要 kernel 是否完成？ |
| `frontier` | 从 0 开始连续完成到哪个 task？ |
| `WonSlot.remaining` | 某 MIX task 还有几个 lane 未完成？ |
| `DistCore.occupied_count` | 本核还有几个 private slot？ |
| final leaf/root/release | 所有 worker 是否已生成完全部 task？ |
| `Runtime::dist.done_count`/COND | 有多少 worker 已退出 dist engine？ |
| `runtime_done_` | AICPU setup thread 是否结束等待？ |
| host stream sync | 整个 device operation 是否已结束？ |

例如观察到 final leaf release 只表示所有核都走完 orchestration 源码，不能立刻
退出；private ring 中可能仍有等待 producer 的任务。每核还要 drain 到 ring 空且
没有待收取的 `block.won`。

### 17.3 Task completion 的发布顺序

单核 task：

```text
kernel return
  -> 当前 runtime 的 store barrier
  -> kernel 写路径必须另外保证数据已经 publication
  -> vend
  -> task flag
  -> frontier advance
  -> slot free
```

MIX task：

```text
每 lane kernel return
  -> remaining--
  -> 最后一个 lane 清 WonSlot
  -> vend + single task flag
  -> frontier advance
```

如果把 flag 提前到 output 数据可见之前，fanin waiter 会合法地开始执行，却读取旧数据。
这就是 completion protocol 不能只验证 flag 原子性的原因。

### 17.4 Worker 和 AICore kernel 的退出

worker 从 `dist_core_main` 返回时，在 onboard 通过 COND register 发布 FIN。AICPU 等到所有
worker FIN 后设置本轮 `runtime_done_`。指定 AICPU thread 再对各 core 执行 deinit/EXIT；
AICore executor 在 `DATA_MAIN_BASE` 看到 EXIT 后写 EXITED 并返回顶层 kernel。

只有此后 host stream sync 才完整结束。若 task 都完成但 EXIT handshake 有问题，表现会是
“结果可能已经写好，host 仍 timeout”。

### 17.5 Fatal 和 timeout 属于不同层次

- `DistGlobal::fatal`：分布式 engine 内部错误/容量失败；
- AICPU executor rc：初始化、handshake 或 shutdown 失败；
- stream timeout：host 观察到整个 operation 未结束；
- device poisoned：某些 A5 op timeout 后同一 context 的后续调用也会失败。

host `DeviceRunner` 对 launch/sync 错误有 bounded recovery 和 finalize force-reset 逻辑。它是
设备生命周期保护，不应被当成调度算法的重试机制。一个可复现的调度 hang 仍需定位是哪层
完成状态没有推进。

## 18. a5sim 与 A5 真硬件的边界

| 维度 | a5sim | A5 onboard |
|---|---|---|
| 执行载体 | host pthread | 真 AICPU/AIC/AIV |
| 地址空间 | 同一进程地址空间 | 独立 device GM/cache/寄存器 |
| atomic | CPU atomic | A5 device atomic primitive |
| dcci | 基本为空操作 | 决定 cache publication/observation |
| orchestration | 链接进 sim AICore SO | 链接进 CCEC AICore image |
| incore 调用 | host function pointer | linked wrapper + `func_id` |
| worker 完成 | host counter + sim register | COND register |
| watchdog/trace | 有 host-clock 辅助 | 当前 direct trace 能力有限 |

### 18.1 sim 能证明什么

- 所有 worker 是否重放同一 task 序列；
- claim、slot、fanin 和算法控制流是否基本成立；
- host function pointer/ABI 在 sim 组合下是否可调用；
- 纯 CPU memory model 下是否死锁；
- host mirror 的数值结果。

### 18.2 sim 不能证明什么

- scalar store 是否写回 HBM；
- cacheline flush 是否 clobber 邻居；
- atomic 观察与普通 payload 的 publication 是否成立；
- CCEC linked wrapper 和 kernel symbol 是否正确；
- AICPU/AICore register handshake 时序；
- 真硬件上的并发速度差是否触发 ring/reclaim 边界。

所以正确的验证顺序不是“sim 过了就结束”，而是 sim 缩小逻辑问题，probe 固化硬件事实，
最后 onboard 集成和压力证明完整协议。

## 19. 当前 direct runtime 尚未接入的 API

本节非常重要。以下类型/API 来自公共 PTO2 表面或旧 runtime，头文件可编译不代表当前
A5 direct engine 已实现其业务语义。

### 19.1 当前实现边界表

| API/字段 | 当前 direct 行为 | 开发时结论 |
|---|---|---|
| `PTO2_SCOPE`/guard | begin/end 是空函数 | 不存在 scope 栈式回收 |
| explicit dependencies | Arg 可保存，submit 不读取 | `add_dep/set_dependencies` 不生效 |
| `launch_spec` | submit 不读取 | 不支持按 task 扩展多 block/sync start |
| `rt_submit_dummy_task` | 返回空结果 | 不生成真实 task/barrier |
| `rt_orchestration_done` | 空函数 | 结束依赖外层 replay 流程 |
| orchestration config | 当前路径不调用 | `expected_arg_count` 未校验 |
| public log/report | AICore glue 基本为空 | 不可依赖 orchestration LOG 定位 |
| `rt_is_fatal()` onboard | 当前返回 false | 内部 fatal 与公共查询未打通 |
| Arg `has_error` | direct submit 不检查 | 参数构建错误未统一上报 |
| `PTO2Runtime::mode` | host 固定 EXECUTE | simulate/graph-only 不在该路径 |

### 19.2 Scope 注释与当前行为不同

一些继承示例把 `PTO2_SCOPE` 描述为中间 tensor 的 arena 生命周期，并声称 scope end 会
释放 output。当前 [`api_glue.h`](../src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/api_glue.h)
中的 scope begin/end 是 no-op；回收主要依靠 heap ring、frontier 和 H。

`TaskOutputTensors` 当前指向每核 `DistTaskPayload` ring 中的 Tensor 描述符。实际 descriptor
可能在同核再提交 `kTaskPayloadSlots` 个 task 后被覆盖，而不是由 scope end 精确回收。
保守用法是得到 output 后尽快构建 consumer，不把 `get_ref()` 保存到长生命周期容器。

在真正实现 scope 前，不能把现有 `PTO2_SCOPE_GUARD` 当作内存正确性证明。

### 19.3 显式依赖目前不会进入 `fanin[]`

`L0TaskArgsWithDeps` 和 `set_dependencies()` 可以把 dep id 保存到 Arg，但当前
`dist_submit_collect_fanin()` 只读取 owner id 和 TensorMap，没有遍历
`explicit_dep_count()`。

因此从其他 runtime 复制使用显式 barrier/dummy task 的 orchestration 到本路径，会出现
“代码编译通过、依赖实际缺失”。若后续实现它，必须把 explicit deps 纳入 DEPSIG、去重、
fanin 容量和 private/shared 等价性测试。

### 19.4 `launch_spec` 目前不会扩展任务

公共类型支持 `core_num` 和 `require_sync_start`，但当前 direct submit 未读取。当前
`MixedKernels` 只描述一个物理 block 内 AIC/AIV0/AIV1 的 active lanes，不等于公共 runtime
中的多 block SPMD task。

因此调度开发中要区分：

```text
block_dim: 本轮启动多少物理 worker block
MixedKernels: 一个逻辑 task 使用该 block 内哪些 lane
launch_spec.core_num: 公共 API 字段，当前 direct 路径未实现
```

### 19.5 Scalar tensor access 的 sim/onboard 行为不同

sim 的 `get_tensor_data/set_tensor_data` 会先用 private map 找 producer，并协作 drain 到其
完成。CCEC onboard 的 `wait_tensor_data_access_ready()` 当前是 no-op，随后直接做 scalar
load/store；scalar store 也没有在该 helper 中自动 dcci flush。

因此：

- 读取 immutable top-level input 可以作为当前特定用法，但仍要保证各核读到相同值；
- 读取前序 runtime task 动态生成的数据不能假设会自动等待；
- scalar 写后若其他核/host 需要观察，writer 必须按已验证协议 flush；
- 该 API 不能作为 shared TensorMap publication primitive。

### 19.6 `manual_dep` 尚无一致跨平台语义

当前 sim fan-in 路径在 `manual_dep=true` 时跳过 map lookup；CCEC 分支仍直接 lookup。
output registration 也没有形成完整一致的 manual-dep 策略。

因此不要用 `manual_dep` 规避当前 map 问题，除非先补齐 sim/onboard 一致实现和测试。否则
同一 orchestration 可能在两个平台生成不同依赖图，直接违反 SPMD/differential 目标。

### 19.7 Orchestration 日志与 fatal 不能当作完整诊断面

当前 AICore `LOG_*` 和 `rt_report_fatal` glue 大多为空，CCEC 的公共 `rt_is_fatal()` 也不
反映内部 `DistGlobal::fatal`。调试时应观察实际全局状态、task flags、slot 和硬件 marker，
不能因为 orchestration 没打印错误就判断它没失败。

## 20. 做并发调度开发必须守住的不变量

### 20.1 SPMD 控制流确定性

所有 worker 必须生成相同的：

- submit 次数和顺序；
- task id；
- tensor/scalar 参数数量和顺序；
- output 大小、对齐和 heap 地址；
- TensorMap insert/retire 序列；
- shared 模式下的逻辑 append 内容。

危险控制流包括：

- 根据本核 role、claim 输赢或 wall clock 决定是否 submit；
- 读取未同步的动态 GM 数据后分支；
- 使用随机数、未初始化值或核局部状态控制循环次数；
- winner 才执行会影响后续 orchestration 的元数据操作；
- 一个核提前 fatal/return，其他核继续进入 shared barrier。

### 20.2 先定义 owner，再选择 atomic

每新增一个字段，应先写清：

```text
谁创建？
谁能写？
能写几次？
谁读取？
何时允许复用？
```

然后再选 plain store、single-owner、CAS、fetch-add 或 sequencer。没有 ownership 定义时，
“加一个 atomic”通常只保护一个字段，无法保护整个复合状态。

### 20.3 控制发布必须覆盖 payload

所有 `payload + state/seq/flag` 协议都要证明：

```text
payload write
  happens-before publish
  happens-before observe
  happens-before payload read
```

并核对 cacheline 布局、flush/invalidate 范围和 slot reuse。`block.won`、shared map、task
completion 和 AICPU handoff 都属于这一类。

### 20.4 不要混用完成层次

- 等某个 producer：看 task flag；
- 回收连续历史：看 frontier；
- 判断所有核生成完图：看 final 树的本组 release；
- 判断本核可退出：还要 ring 空、无 pending won；
- 判断 host 可回收本轮：等 stream 完成。

拿 final release 代替 task completion，或拿 frontier 代替最慢 core progress，都会在并发
速度变化后出错。

### 20.5 所有有界结构都需要反压或失败语义

当前关键上限包括：

| 结构 | 当前上限/窗口 |
|---|---|
| task id/cell | `kFlagCap = 65536` |
| private slots | 每核 4，预留部分给 won |
| fan-in | `kMaxFanin = 16` |
| task payload ring | 2048 slots |
| private map entries | 16384 |
| map retirement task ring | 1024 |
| default H | 64 |
| worker | 最多 108 |

新增 shared ring 后还会增加每 bucket capacity 和 run-ahead 上限。每个上限都必须选择：

- wait 并协作推进；
- 安全回收；
- 或设置 fatal 并让所有核一致退出。

静默截断 fan-in、覆盖 live payload 或 insert 失败后继续执行，都会把容量问题伪装成随机
数值错误。

### 20.6 ABI 是跨三个程序的并发契约

`Runtime`、`KernelArgs`、`Handshake`、`Tensor`、`DistGlobal` 和 slot 结构跨 host/AICPU/
AICore 使用。修改字段时要检查：

- size/alignment/offset；
- host mirror 与 device pointer 修补；
- AIC 与 AIV 两次编译的布局；
- sim 与 CCEC 条件编译；
- cacheline ownership；
- 所有独立 binary 是否重建。

一个布局错位会表现为任意地址、错误 core type 或 kernel args 损坏，症状常常远离修改点。

### 20.7 Kernel 数据写入是调度协议的一部分

completion flag 的正确性依赖 kernel 写入已经可见。开发调度器时不能把 kernel 当作完全
黑盒：至少要知道它使用 scalar store、vector TSTORE、异步 engine 还是其他写路径。

当前已验证：scalar store 需要 writer dcci，vector TSTORE 不需要同样 flush。新增通用
post-kernel flush 前必须证明不会写回陈旧控制 cacheline，也不能对所有数据路径一刀切。

### 20.8 Private 是 shared 的差分参考

实现 shared TensorMap 时，private 路径应继续可运行，并要求：

- 同输入、同 task 序列；
- 同 fan-in edge 集合；
- 同最终数值；
- 不同且符合预期的 TMOPS；
- 同样的 heap 地址和 task completion 语义。

任何“shared 数值能跑，但 private/shared DEPSIG 不同”的状态都不能进入后续性能比较。

## 21. 调试地图和修改入口

### 21.1 按阶段定位失败

| 阶段 | 常见现象 | 首先检查 |
|---|---|---|
| 编译 | CCEC/链接符号失败 | AICore override、wrapper、func id |
| callable prepare | handle/地址错误 | `ChipCallable` child 和缓存 |
| host bind | 首元素已错、输出不回传 | chip 签名、H2D/D2H ledger |
| AICore launch | 207001/启动超时 | image、block_dim、launch 顺序 |
| handshake | worker 未进入 replay | ready/regs/core type/COND |
| replay | 核间 task 数不同、全局卡死 | 数据依赖分支、local_index |
| claim | task 无 owner 或重复执行 | cursor shard、核类型、原子旧值 |
| fan-in | consumer 永远 pending | map/owner id、producer flag |
| `block.won` | MIX follower 不执行 | state/payload/drained/remaining |
| kernel call | 进入 slot 后 hang/mismatch | func id、args ABI、输入可见性 |
| completion | kernel 返回但后继不跑 | data flush、flag、frontier |
| drain/exit | task 完成但 host timeout | final leaf/root/release、ring、COND/EXIT |
| validate | device 对、host 错 | D2H 和 top-level direction |

### 21.2 不同改动从哪里下手

| 目标 | 主要文件 |
|---|---|
| 改 orchestration 业务图 | example/test 的 `kernels/orchestration` |
| 改 task 参数/tag | `pto_types.h`、`pto_orchestration_api.h` |
| 改 submit/claim/slot | `submit_direct.h`、`submit_core.h` |
| 改 private TensorMap | `tensor_map.h`、`state.h` |
| 加 shared TensorMap | `state.h` + 新 mode seam + submit 路径 |
| 改 cache/atomic primitive | `primitive.h`、`atomic.h` + probe |
| 改 AICore 生命周期 | `aicore_executor.cpp`、`core_main.h` |
| 改 AICPU setup/wait | `aicpu_executor.cpp`、`control_plane.h` |
| 改 host 内存/arena | `runtime_maker.cpp`、`pto_runtime2_init.cpp` |
| 改 launch/stream | platform `device_runner.cpp` |
| 改 callable/link | `scene_test.py`、builder、callable headers |
| 改跨程序 ABI | `runtime.h`、`kernel_args.h`、common types |

### 21.3 推荐的最小定位顺序

遇到一个新的并发失败时：

1. 确认当前 source、runtime binary 和 AICore override 对应同一版本；
2. 确认测试真的进入目标 runtime/mode；
3. 确认 host 输入和 chip 级签名；
4. 确认所有 worker 通过启动屏障并生成相同 task 总数；
5. 找出第一个没有完成的 task N；
6. 确认 N 是否被唯一 claim；
7. 确认 N 的 slot、kernel id 和 fan-in；
8. 逐个确认 producer flag 和 producer output 数据；
9. 若是 MIX，再检查 won payload/remaining；
10. 最后检查 worker drain、COND 和 D2H。

这个顺序从“第一处状态分叉”定位，而不是从最终 golden 倒猜所有可能原因。

### 21.4 测试证据如何组合

- host UT：证明数据结构算法；
- a5sim：证明集成控制流和确定性重放；
- atomic/cache probe：证明单个硬件 primitive；
- onboard minibench：证明复合 publication 协议；
- 完整算子：证明业务图和数值；
- DEPSIG/TMOPS：证明模式与依赖图；
- 长序列/慢核压力：证明容量、回收和 run-ahead。

上板命令必须通过本机的 `task-submit` 设备锁执行。未隔离的并发硬件运行会把其他进程的
资源竞争混入时序结果，不适合作为 race 复现率证据。

## 22. 最终判断

当前仓库已经具备 fully distributed runtime 的 private TensorMap 主干：所有 AIC/AIV
确定性重放、唯一 claim、winner fan-in lookup、全核 producer 注册、private slot 执行、
全局 completion/frontier，以及确定性 output 地址生成。

但它距离总体目标仍有三个层面的缺口：

1. **可观测性缺口**：集成 runtime 尚无目标要求的 mode marker、DEPSIG 和 TMOPS；
2. **基础一致性缺口**：最新严格上板中 MB-4 和 MB-8 仍有真实数值问题；
3. **shared 功能缺口**：当前 split runtime 没有 shared map、sequencer、seq、progress、
   reclaim 和 run-ahead。

shared 对比 checkout 已探索第三层的大部分算法构件，但没有解决到可以直接合入并宣称
上板完成的程度，尤其是普通 payload 与原子发布字段之间的硬件可见性。

因此合理路线是：以当前 private split runtime 为 reference，先让诊断和 MB-2/4/8
可信，再逐项移植 shared 协议，最后用 private/shared 的数值、DEPSIG 和 TMOPS 三重
差分完成验收。这样才能证明优化的是 TensorMap 存储与更新方式，而没有悄悄改变依赖图。
