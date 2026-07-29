# FDWIC Shared Map 候选方案：有序单写、多读流水

> 类型与状态：交互式维护的单一候选方案规格，不是当前实现说明或
> Shared Map 通用协议。本文只把已经对齐的方案内规则写成确定项；
> 尚待实现验证收敛的内容列在第 6.7 节，不视为已完成实现。

本文只记录“有序单写、多读流水”这一具体候选方案。相关内容按以下
文档边界归档：

- 当前实现事实和 region-intent 时序放在
  [`shared_map_current_execution_model.md`][current-model]；
- INOUT 发布问题和多个候选方案的比较放在
  [`shared_tensormap_inout_publication_protocol.md`][protocol-comparison]；
- 本文只描述有序单写、多读流水方案的前提、性质和实现约束。

后文出现的 `H(N) -> I(N) -> L(N) -> B(N)`、`producer_task_id < N`、
后续 insert 越过当前 lookup 等结论，都只在本文定义的方案前提下成立。
它们不能直接用于描述当前 Shared Map，也不能直接套用到 per-core
overlay、版本化句柄或其他发布方案。

## 1. 候选方案前提

本文方案建立在以下前提上：

1. 每个 task 都执行一次有序 heap commit 和 insert commit；对应阶段没有
   实际工作时也执行 empty commit；
2. `next_heap_task_id` 和 `next_insert_task_id` 分别串行化 H、I，二者是
   相互独立的原子有序游标；
3. `I(N)` 位于本 task 的 `H(N)` 之后；对 task N 执行 lookup 前，所有
   `I(K), K < N` 都已经完成；
4. 原子变量 `next_insert_task_id` 是 TensorMap 唯一的有序写锁；
   lookup 不获取或修改该锁，但必须位于一次有效的 acquire-observe 之后；
5. `next_insert_task_id` 同时提供 writer 互斥、task 顺序和 insert
   commit/entry 可见性发布，不再设置独立的 free/busy 锁；
6. winner N 在同一条控制流中连续执行 heap、insert、lookup 和 build；
7. lookup 按 consumer task id 查询历史视图，而不是读取物理最新
   writer。

这些前提不是当前实现事实；改变任一前提时，必须重新检查本文的全部
推导结论。

## 2. 方案内术语和阶段

对 task N 定义：

- `calc_output_layout(N)`：winner-local 准备步骤，在 H 前根据 OUTPUT
  `TensorCreateInfo` 计算 output index、size、alignment/padding 和 task
  总 logical heap 需求；它不访问任何 H/I 共享状态；
- `H(N)`：使用预先算好的 output layout，从现有的单调递增
  `shared_heap_vend` 预留逻辑地址区间，并初始化 shared logical Tensor
  descriptor；H 不写物理 output buffer，没有新建 OUTPUT 时执行 empty
  commit；
- ref 准备：H commit 后、I 锁外解析所有 `FdwicOutputRef` 及其多维 view
  metadata，再从完整 logical args 统计准确 `writer_count`；
- `I(N)`：对普通 kernel task N 的全部 writer 调用 TensorMap insert，并
  完成有序 task commit；没有 writer 时执行 empty commit；
- `L(N)`：task N 查询 input/inout 前驱；
- `B(N)`：按既有 heap 容量/回收策略等待 task N 的预留区间可用；普通
  kernel task 把 shared logical Tensor copy 到 slot 并设置物理地址，再
  完成 task build；`alloc_tensors` 执行不创建 kernel slot 的轻量 build；
- `E(N)`：task N fanin ready 后执行。

本文只使用 TensorMap entry 这一名称，不为它引入其他别名。entry 保存在
TensorMap 自己拥有的共享 GM entry pool 中；它不引用临时 args/private
slot，也不存放在 `SharedOutputCell` 或 logical/physical heap 中。

执行阶段沿用以下名称：

- heap 准备态：Claim winner 后执行 `H(N)`；
- 插入态：Claim winner 后执行 `I(N)`；
- 构建态：同一个 winner 继续执行 `L(N)` 和 `B(N)`；
- 运行态：检查 fanin，并在 ready 后执行 `E(N)`。

这些阶段名只区分同步范围，不表示独立工作项、队列或所有权移交。
在正常的无故障执行假设下，同一个 winner 连续负责：

```text
普通 kernel task:
  calc_output_layout(N) -> H(N) -> resolve refs/count writers
                        -> I(N) -> L(N) -> B(N)

alloc_tensors:
  calc_output_layout(N) -> H(N) -> I(N) empty commit -> B_alloc(N)
```

loser 不执行这些阶段，也不构造完整 args；它只返回 symbolic output refs
并立即继续 replay。

对普通 kernel task 和 `alloc_tensors`，wrong-role 核不属于该 task 的
候选核，因此不参与 Claim，也不执行当前 task 的 H/I/L/B。它可以在跳过
当前 task 前 drain 本核已有的 runnable RingSlot 或已经赢得的旧工作，
以帮助系统推进；随后只根据 `task_id` 和 output 序号生成 symbolic
output refs，并继续 replay。只有 role 匹配的候选核参与 Claim，唯一
winner 负责推进上述阶段。MIX follower 不是这种普通 wrong-role 路径，
而是按下述现有 joint/WonSlot 机制接收并执行同 block subtask。

MIX 的共同所有权仍采用固定 block 配对：只有 AIC 候选核参与 MIX 的
Claim；唯一 AIC winner 同时选定该 MIX 的获胜 block，该 block 内 active
mask 对应的 AIV 核与 winner 共同执行其余 subtasks。这些 AIV 核没有参与
Claim，也不是 Claim loser 或普通 wrong-role；它们的 subtask 所有权由
同 block AIC winner 派生。subtask 继续通过现有 joint/WonSlot 机制从
AIC winner 交给这些核，不为本方案重新设计一套 follower handoff。

MIX 的 H/I/L 是 task-level 工作，只由 AIC winner 各执行一次。H 一次
初始化该 task 的全部 shared logical outputs；I 一次插入该 task 的全部
TensorMap entries，并只推进一次 `next_insert_task_id`；L 一次收集整个
MIX task 的 fanin。同 block AIV 核不重复执行 H/I/L，也不推进对应阶段
游标；其 subtasks 后续复用 AIC winner 生成的 logical Tensors。
MIX 不设置独立的 heap gate：AIC winner 直接复用普通 winner 的 B，在
task-level heap 容量门通过后完成物理映射、填充 WonSlot，并构建自己的
privateSlot；AIV 不重复执行 heap 容量检查。

现有 joint/WonSlot 生命周期保持不变：AIC build 时先填充但不立即发布
WonSlot；AIC privateSlot 达到 ready 时发布 WonSlot 并执行 AIC subtask；
对应 AIV 通过 `drain_block_won()` 把自己的 subtask 搬入 privateSlot，
再按既有 privateSlot ready/execute 流程执行。原始 fanin 只保存在 AIC
privateSlot 并由 AIC 检查；AIV subtask 的 `fanin_count = 0`，AIC
ready 后对 WonSlot 的发布为 AIV 提供传递的依赖准入。该机制只需要适配 B
产出的新 logical-to-physical Tensor copy，不改变所有权、slot 状态机或
完成计数。

MIX 的 WonSlot/privateSlot 推进仍采用协作式 polling，不增加中断或后台
调度线程。每个核在以下三类检查点依次调用
`drain_block_won_if_enabled()` 和现有 `drain_phase_b()`：

1. 每个 replay task 的 presubmit/Claim 前；shared 路径可先用
   `dist_submit_has_drain_work()` 跳过确定无工作的调用；
2. H/I、privateSlot、WonSlot 和 heap 容量等所有阻塞等待循环的每轮；
3. replay 结束后的 drain-to-completion 循环，直到本核 privateSlot、
   待接收 WonSlot 和 joint launch 计数全部收敛。

AIC winner build 完成后不额外立即执行一次 drain；其 task 在该核到达
下一个上述检查点时进入 ready 检查。AIC 的执行 drain 发布已 ready 的
WonSlot；对应 AIV 在自己的下一个检查点先把 WonSlot subtask 搬入
privateSlot，再由同一检查点紧随其后的执行 drain 检查该 slot。

现有函数名 `drain_phase_b()` 沿用旧阶段命名，其实际行为是扫描并执行
fanin-ready privateSlots，语义上属于本文的 E 进展，不是本文负责 heap
反压、物理映射和 build 的 `B(N)`。旧 region-intent、per-output
published wait 和 heap-shard wait 删除后，其专属 drain 调用点也随之
删除；新 H/I/B 等待循环必须保留上述协作推进。

## 3. 本方案内已确认的并发规则

### 3.1 H、I 使用两个独立的有序游标

两个阶段分别按 task 顺序提交：

```text
H(N):
  wait next_heap_task_id == N
  -> 分配 task N 的单调逻辑 heap 区间并初始化 shared logical Tensors
  -> publish next_heap_task_id = N+1

I(N):
  本核已经完成 H(N)
  -> wait next_insert_task_id == N
  -> 插入 task N 的全部 TensorMap entries；无 entry 时不修改 TensorMap
  -> publish next_insert_task_id = N+1
```

`next_heap_task_id` 只约束 H 之间的顺序，`next_insert_task_id` 只约束
I 之间的顺序。就跨 task 的准入条件而言，`H(N)` 只等待 `H(N-1)`，
`I(N)` 只等待 `I(N-1)`；`H(N) -> I(N)` 是同一个 winner 内部的
task 内依赖。

`alloc_tensors` 即使含 OUTPUT，`I(N)` 也始终是 empty commit：它只原子
推进 `next_insert_task_id`，不分配 entry、不更新 bucket head。

Claim 已经保证每个 task 只有一个 winner。当 `next_heap_task_id == N`
时，只有 winner N 能修改有序 heap 状态；当
`next_insert_task_id == N` 时，只有 winner N 能修改 TensorMap。两个
阶段游标因而分别兼任对应阶段的 task 顺序、互斥、commit/可见性发布和
下一次权限交接，不再需要二值 free/busy 锁。

`next_heap_task_id` 是阶段发布游标，不等同于以 byte 为单位的 heap
分配位置。现有 `shared_heap_vend` 是 H 唯一的、只增不减的逻辑 heap
分配状态；现有 `heap_base` 仍只是固定的物理 GM heap 基址。候选方案删除
`shared_heap_cursor[]`、heap shard 选择和 shard 内 wrap 逻辑。

worker 启动前，control plane 初始化
`next_heap_task_id = 0`、`next_insert_task_id = 0`、
`shared_heap_vend = 0`、`entry_high_water = 0`，并把全部 bucket head
初始化为 `-1`。per-task vend 初始化为 0。上述状态只在首次 worker
启动前，或确认不存在任何 H/I/L/B/E 访问的全局静默边界整体 reset；
运行中的单个 task 不得局部重置阶段游标或 allocation cursor。

由于 I 在全局范围内只有一个 writer，TensorMap 内部不再使用
writer-writer 原子竞争协议：entry allocation cursor 普通递增；删除
atomic fetch-add、`-2` bucket lock、CAS/RMW 重试和全局 insert lock。
`bucket.head` 仍使用硬件 atomic 读写，但它只作为单字的内存级读取和
发布原语，不再表示 bucket 被锁定，也不存在抢占或重试。entry 内容的
可见性由 entry flush、atomic head 发布、task-level insert cursor
发布以及 reader invalidate 共同建立。

### 3.2 TensorMap insert 直接使用 args

本方案中普通 kernel task 的 winner N 目标流程为：

```text
Claim winner N
  -> calc_output_layout(N)
  -> 等待 next_heap_task_id == N
  -> 根据预计算 layout 从 shared_heap_vend 预留 task_base/task_vend
  -> 初始化 shared logical Tensor 的 storage identity 和完整几何
  -> 发布 next_heap_task_id = N+1，提交 H(N)
  -> invalidate 并解析全部 FdwicOutputRef 及其多维 view metadata
  -> 遍历完整 logical args，统计 I(N) 实际需要插入的 writer_count
  -> 等待 next_insert_task_id == N
  -> 独占 task N 的 TensorMap 写权限
  -> 按 writer_count 预检并预留连续 entry slots
  -> OUTPUT: TensorMap::insert(shared logical Tensor, N)
  -> INOUT/OUTPUT_EXISTING: TensorMap::insert(Tensor, N)
  -> 发布 next_insert_task_id = N+1，提交 I(N)
  -> lookup task N 的前驱
  -> 等待 task N 的 heap 预留区间已经可以安全复用
  -> copy logical Tensor，设置 physical buffer address
  -> build task N
```

args 为 `OUTPUT` 提供 `TensorCreateInfo`；H 使用预计算 layout 和
create info 初始化 shared logical Tensor，并补充单调逻辑地址。直接
Tensor 参数已经提供 descriptor；shared OUTPUT 的
`INPUT/INOUT/OUTPUT_EXISTING` 参数则在 H commit 后从
`SharedOutputCell::tensors` 解析为 winner-local logical Tensor。

准确 `writer_count` 必须在这些 ref 解析后统计，因为解析前无法读取源
Tensor 的 `manual_dep`。统计仍在 I 锁外，并通过 winner-local 状态传给
I；不需要 shared 字段或单独发布。I 不构造 normalized delta，也不要求
调用方构造额外中间对象。
`TensorMap::insert` 直接读取 Tensor。首版仍从 Tensor 计算并保存临时
byte range `[lo, hi)`；真实高维求交需要的完整几何字段和算法留待后续
启用。

`manual_dep` 只关闭自动 TensorMap 路径。H 仍为其新 OUTPUT 完成 logical
heap reservation 和 descriptor 初始化；H 后的准确 `writer_count` 统计
排除所有 `manual_dep` Tensor，I 不为其插入 entry。L 仍收集显式依赖和
有效的 owner/creator 依赖，只跳过该 Tensor 的 TensorMap lookup。

TensorMap slot 分配、entry 初始化、bucket/history link 更新和 entry
publication 全部位于 I。B 只能修改 copy 到 private slot/payload 的
执行 Tensor，不能修改 H 已发布并可能正被 lookup 读取的 shared logical
Tensor。任何 output buffer 写入（包括 initial-value fill）必须位于 B
的 heap 容量门之后。

`alloc_tensors` 使用同一 H 顺序初始化 logical Tensors，但
`writer_count = 0`，其 I 只执行 empty commit：

```text
alloc winner N
  -> H(N)
  -> 等待 next_insert_task_id == N
  -> I(N) empty commit：atomic publish next_insert_task_id = N+1
  -> B_alloc(N)：heap 反压、物理映射、initial fill
  -> publish task completion flag
```

`B_alloc` 是轻量 build，不执行 TensorMap lookup，不创建 kernel
RingSlot，也没有 E 阶段。它必须位于 I empty commit 之后，使 alloc 的
heap 等待不会占住 insert 前缀、阻塞后续 task 的 I。
alloc OUTPUT 是新 storage，其 creator/completion 依赖由
`SharedOutputCell::tensors` 和 owner task id 表达；后续对该 storage 的
INOUT/OUTPUT_EXISTING 才向 TensorMap 追加 entry。

候选方案保留 `SharedOutputCell::tensors`，供
`FdwicOutputRef{producer_task_id, output_slot}` 直接取得 logical Tensor；
该表不通过 TensorMap 间接定位。SharedOutputCell 只保留 tensors，删除
per-output `published` 和 `last_writer`。H 完整初始化 task N 的全部
tensors 后，由 `next_heap_task_id = N+1` 一次性完成 task-level 发布。
I 和后续 lookup 再以这些 Tensor 的 `logical_addr` 访问 writer history。

`FdwicOutputRef` 必须完整表达 1 至 `MAX_TENSOR_DIMS` 维的延迟 view
metadata，不能保留当前只支持一维的 `view_shape0/view_offset0` 特例：

```cpp
struct FdwicOutputRef {
    int32_t producer_task_id;
    int16_t output_slot;
    uint8_t flags;
    uint8_t view_ndims;
    uint32_t view_shapes[MAX_TENSOR_DIMS];
    uint32_t view_offsets[MAX_TENSOR_DIMS];
};
static_assert(sizeof(FdwicOutputRef) == 48);
```

无 view 时 view flag 不设置，两个数组不参与语义。第一次 `view()` 保存
全部维度的 shape/offset；嵌套 view 要求维数一致，并逐维检查
`new_offset[d] + new_shape[d] <= old_shape[d]`，然后执行：

```text
view_offsets[d] += new_offset[d]
view_shapes[d] = new_shape[d]
```

解析时要求 `1 <= view_ndims <= MAX_TENSOR_DIMS` 且
`view_ndims == source.ndims`，再使用完整数组调用多维 `Tensor::view`。
实现删除全部 `ndims == 1` 断言和 `shape0/offset0` 分支，并增加二维、
五维和嵌套 view 覆盖。这里的“延迟”只表示 view metadata 先随
`FdwicOutputRef` 保存，之后再应用到 H 已发布的 logical Tensor；不引入
另一种 Tensor 或 view 类型。

所有 ref 都在 H commit 后、I 前解析为 winner-local logical Tensor，
因此 unresolved ref 不再进入 B、RingSlot 或 WonSlot。本方案删除
`RingSlot` 和 `BuiltSubtask` 中的 `shared_ref_mask/shared_refs[]`，同时
删除 `dist_resolve_slot_shared_refs()`、执行前 shared-ref wait/resolve
以及对应 copy/flush/invalidate 分支。MIX AIC winner 在 B 中把已经解析并
完成物理映射的 Tensor 写入 WonSlot，AIV 只复制 Tensor。48B
`FdwicOutputRef` 只存在于 replay/args 阶段，不占用常驻 privateSlot 或
WonSlot payload。

Tensor 使用第二条 cache line 中 `strides[]` 后现有的 36B reserved 区域
保存一个稳定 logical address。该字段使用公共 8B `MaskPointer` 值类型，
由它统一封装地址低 3 位的附加信息：

```text
offset 92:  uint8  alignment_pad[4]
offset 96:  MaskPointer logical_addr
offset 104: uint8  reserved[24]
```

`MaskPointer::ptr()` 清除低 3 位并返回纯地址部分，
`MaskPointer::uint64()` 返回包含所有 tag 的完整 64-bit 值；
`active<Bit>()` 检查指定低位。Tensor logical address 首版定义：

```cpp
enum class TensorLogicalAddrInfo : uint8_t {
    IsTensorGM = 0,
};
```

这里的 `IsTensorGM` 特指由本 runtime GM ring heap 管理、需要在 B 中
映射的 Tensor，不泛指所有位于 GM 的 external Tensor。其余两个 tag bit
首版保留。

external Tensor 保留已有的物理 `buffer.addr`，并以 tag 全零的
`buffer.addr` 初始化 `logical_addr`；由于 `MaskPointer` 占用低 3 位，
external buffer base 必须至少 8B 对齐，并在创建入口检查。B 遇到
`IsTensorGM == false` 时不修改该 Tensor copy 的 `buffer.addr`，不需要
通过 `MaskPointer::ptr()` 恢复 external 地址。

新建 OUTPUT 使用 H 从 `shared_heap_vend` 分配的、至少 8B 对齐的单调
logical byte address，并设置 `IsTensorGM`。H 发布的 logical Tensor 将
`buffer.addr` 明确置为 0，只保留 `buffer.size`、完整几何和
`logical_addr`；B 之前不得进行数据访问。H 从进入阶段时的
`shared_heap_vend` 计算 task_base，把各 OUTPUT 放入该 task 的逻辑区间，
并得到包含 alignment 和 wrap padding 的 task_vend。
`shared_heap_vend` 和 `DistTaskCell::vend` 始终保存不含 tag 的原始 byte
cursor。

H 为 task N 新建的每个 internal OUTPUT 设置
`owner_task_id = N`。`FdwicOutputRef` 解析、Tensor copy 和 view 原样保留
该 owner；它们不能把 owner 改为当前 consumer。external Tensor 继续由
创建入口设置 invalid owner。L 使用该既有 owner/creator 关系形成依赖，
它与 TensorMap 的 writer history 是两条独立来源，最终统一去重。

TensorMap 的 hash 和 identity 比较使用 `logical_addr.uint64()`，使
internal/external namespace tag 参与 identity。所有 copy 和 view 类操作
必须原样传播完整 `MaskPointer`；view 只修改 offset、shape 和 stride。

本方案要求 `heap_size` 是 2 的幂，因此 B 使用
`reinterpret_cast<uint64_t>(logical_addr.ptr()) & (heap_size - 1)` 得到
internal Tensor 的物理 offset，并只在 private Tensor copy 中写入
`buffer.addr = heap_base + physical_offset`。H 必须把 ring 尾部 padding
计入 `shared_heap_vend`，保证每个要求连续存储的 OUTPUT 都满足
`physical_offset + buffer_size <= heap_size`，不能跨物理 heap 尾部。

只要 `I(N)` 是一次按 task 顺序完成的提交，它完成时就意味着所有
`I(K), K < N` 已经完成。由于 `L(N)` 紧随其后，winner N 不需要再等待
一个独立的全局发布前缀。

### 3.3 每个 task 都推进两个阶段游标

每个 task 的 winner 都必须完成一次 H commit 和 I commit：

```text
H(N): 有新建 OUTPUT 则预留逻辑 heap 并发布 logical Tensors，否则 empty commit
      -> next_heap_task_id = N+1
I(N): 普通 kernel task 有需跟踪的 writer 则发布全部 entries，否则 empty commit
      alloc_tensors 始终执行 empty commit
      -> next_insert_task_id = N+1
```

否则 N+1 无法判断 N 已经经过相应阶段，流水会停止。“是否有实际阶段
工作”和“是否推进阶段游标”是两个不同问题。

H 的 empty commit 不更新 `shared_heap_vend`，也不 flush
`SharedOutputCell::tensors`，但仍把当前 vend 写入
`task_cell(N).vend`，并原子推进 `next_heap_task_id`。

### 3.4 `I(N)` 是整个 task 的提交点

一个 task 有多个 output/inout 时，winner N 在同一次写权限持有期内
发布全部 TensorMap entries：

```text
等待 next_insert_task_id == N
  -> 预留连续 entry slots
  -> 初始化 task N 的全部 entries 和 bucket links
  -> 批量 flush 连续 entry 区间 + dsb
  -> atomic publish 各受影响 bucket 的新 head
  -> store_barrier()
  -> commit I(N)
  -> atomic publish next_insert_task_id = N+1
```

不得在 task N 的部分 entries 发布后提前推进
`next_insert_task_id`，也不得在同一个 task 的多个 entries 之间
把写权限交给其他 task。

所有 TensorMap entries 必须先完成全部字段初始化，再执行一次覆盖该连续
slot 区间的批量 flush。只有 flush 和 dsb 完成后，才允许发布任何受影响
bucket 的新 head。entry 一旦发布就保持不可变，任何 reader 都不能观察到
半初始化 entry。

N 的全部 entry 写入和 entry publication 都发生在
`next_insert_task_id = N+1` 之前。该原子变量的发布是 task-level
commit：后续 task 只有 acquire-observe 到大于 N 的值后，才允许读取并
使用 task N 的 entries。

task N 的多个 bucket heads 不要求同时变得可见，可以在写临界区内逐个
发布；但此时它们可达的 task N entries 已经全部完成批量 flush。较早
lookup 可能通过部分已更新的 heads 观察到 task N 或 N+1 的完整 entries，
但会根据 producer task id 将它们排除，因此不会使用尚未完成 task-level
commit 的 writer。

没有 TensorMap entry 的 empty commit（包括所有 `alloc_tensors`）和包含
多个 entries 的 commit 使用相同的 task 级边界。区别只在于提交前是否
需要修改 TensorMap 内容。

### 3.5 后续 insert 可以越过当前 lookup

两个独立游标形成 task 间流水。`H(N-1)` 提交后，winner N 可以执行
`H(N)`，同时 winner N-1 执行 `I(N-1)`：

```text
Core A, winner N-1:
  H(N-1) ---- I(N-1) ---- L(N-1) ---- B(N-1)

Core B, winner N:
              H(N) ------ I(N) ------ L(N) ------ B(N)
```

其必要偏序只有：

```text
H(N-1) -> H(N)
I(N-1) -> I(N)
H(N)   -> I(N)
```

因此 `H(N)` 可与 `I(N-1)` 并发；`I(N+1)` 也可与 `L(N)`/`B(N)`
并发。协议不要求 `L(N) -> I(N+1)`，这区别于把所有阶段放进 global
exact-turn 临界区的方案。

loser 即使立即继续 replay 并赢得 N+1，也必须在 H(N+1) 前等待
`next_heap_task_id == N+1`，并在 I(N+1) 前等待
`next_insert_task_id == N+1`。因此 `L(N+1)` 不可能越过 I(N)，无需再让
task N 的 loser 等待 winner 发布 writer intent。

候选方案据此删除 region-intent 分支、`deps_prepared`/`PreparedDeps`、
对应 token bits 和 shared-ref `last_writer` exchange。所有 task 统一为
Claim 后 winner-only 构造完整 args；loser 只可操作 symbolic ref/view，
不得在 H/I 提交前读取 output Tensor metadata 或物理数据。H、I 和 heap
容量等待循环必须持续 drain 本核已有执行任务，保证阻塞期间仍可推进。

heap 容量和回收沿用既有策略：H 把本 task 的 task_vend 原子写入现有
`DistTaskCell::vend`；B 使用该 task-local vend 检查回收前沿，容量不足
时等待并协作推进执行。不能直接使用已经被未来 H 推进的全局
`shared_heap_vend`，否则未来 reservation 可能反过来阻塞本可执行的较早
task。该 B 反压会逐核停止 replay，因此不再规定独立的 H 相对 I/exec
最大超前量。

每个 task 都必须写自己的 `DistTaskCell::vend`。没有新 OUTPUT 的 task
继承当前 `shared_heap_vend`，保证 frontier 跨过连续无输出 task 时，回收
水位仍连续。shared completion 只发布完成 flag，不再用可能已被后续 H
推进的 `DistCore::heap_next` 覆盖该 per-task vend；`heap_next` 在
non-shared 路径中的既有用途不受影响。

B 对含新 OUTPUT 的 task N 使用以下反压条件：

```text
task_vend = atomic_load(task_cell(N).vend)
loop:
  F = atomic_load(frontier)
  R = F - H
  reclaimed_vend = (R < 0) ? 0 : atomic_load(task_cell(R).vend)
  if task_vend - reclaimed_vend <= heap_size:
    允许物理映射并继续 build

  // 只在首次判断容量不足后按需推进完成前缀
  advance_frontier_until(N - 1, bounded_steps)
  F = atomic_load(frontier)
  R = F - H
  reclaimed_vend = (R < 0) ? 0 : atomic_load(task_cell(R).vend)
  if task_vend - reclaimed_vend <= heap_size:
    允许物理映射并继续 build
  if F >= N - 1:
    设置 fatal：全部较早 task 已完成时该 task 仍无法放入 heap

  drain 本核已有执行任务并继续等待
```

没有新 OUTPUT 的 task 跳过该容量检查。公式中的 `task_vend` 固定属于
task N，不能在循环中改读全局 `shared_heap_vend`。所有共享标量读写仍
通过公共 atomic 封装。普通 task completion 和 MIX 最后一个 subtask
completion 只发布 task flag，不在完成热路径调用 `advance_frontier()`；
frontier 仅在 B/B_alloc 实际发现容量不足后按上述方式懒推进。

`B_alloc(N)` 使用同一反压公式。容量可用后，它把 logical address 映射到
物理 heap，并对临时 Tensor 副本执行所需的 initial-value fill；不能修改
H 已发布的 `SharedOutputCell::tensors`。即使没有 initial fill，只要 alloc
包含新 OUTPUT，也必须先通过容量门再发布 completion flag。等待期间持续
drain 本核已有任务。

当前 shared 流程在物理 heap reserve 时执行复用等待。候选方案把 logical
address 与 physical ring address 分离后，H、I、L 都不映射或写物理
buffer，TensorMap 也只使用 logical address。因此物理 heap 的 reclaim、
容量检查和等待可以从 insert 前整体后移到 B；B 通过容量门后才允许映射
物理地址、执行 initial-value fill 或发布可执行 slot。

### 3.6 Lookup 使用 task-id 历史视图并返回多个 producer

`L(N)` 的物理执行时间可以晚于 `I(N)`、`I(N+1)`，甚至更多未来 task
的 insert，但它的逻辑查询边界固定为 task N 之前：

```text
producers(N, X) =
  unique { W | W < N and W writes an overlapping part of X }
```

一个查询区域可能同时覆盖多个 task 写过的子区域，因此 lookup 允许返回
多个 producer。每次 insert 必须追加 TensorMap entry，或者以其他方式保留
等价的 writer 历史；后续 insert 不能覆盖并丢失仍可能形成依赖的较早
writer。

首版不消除被后续 writer 完全覆盖的旧 entry。即使后续 INOUT 覆盖旧
entry，也不能 unlink、复用或标记旧 entry 失效；lookup 仍扫描所有满足
task-id cutoff 和相交条件的 entries，最后只按 `producer_task_id` 去重。
因此首版允许保留可由传递依赖蕴含的冗余 fanin。现有 TensorMap 的
`OverlapStatus::COVERED -> remove_entry()` 路径不能直接带入本共享方案；
空间覆盖消除只能作为后续独立优化。

本方案直接移除单值 `last_writer` 字段及其 exchange 路径。所有
predecessor 都通过带 producer task id 的历史 entries 查询，不保留
`last_writer` 缓存或双路径 fallback。底层实现如果需要保存版本链头
指针，它只是定位历史 entries 的结构索引，应使用不同名称。

也就是说，`L(N)` 必须：

- 只考虑 `producer_task_id < N` 的 writer；
- 排除 task N 自己在 `I(N)` 中发布的 writer；
- 排除 task N+1 及更晚 task 已经发布的 writer；
- 允许扫描到未来 task 的部分完整 entries，但不能使用它们；
- 对所有 tensor 参数、显式依赖、owner 依赖和 map 命中得到的 producer
  task id 做全局去重，再占用 fanin 槽位；
- 对 `manual_dep` Tensor 保留显式依赖和 owner/creator 依赖，但跳过
  TensorMap lookup；
- 在所有 `< N` task 的 insert 已提交后才开始。

当前阶段沿用固定 `kMaxFanin`：去重后的 producer 总数达到上限后，后续
唯一 producer 暂时丢弃。该行为可能丢失真实依赖边，属于已知正确性限制；
实现中的丢弃分支必须保留明确 TODO，后续改为可表达全部依赖的方案。
先去重再检查容量可以最小化依赖边，并尽量避免重复 producer 挤占槽位。

因此，lookup 读取的是以 N 为 cutoff 的多 producer 逻辑历史视图，而不是
查询时 TensorMap 中物理上最新的单一 writer。

### 3.7 Fatal 的热路径策略

shared submit 在每次 submit 入口只通过公共 atomic 封装读取一次全局
fatal。若入口已经观察到 fatal，本核不再 Claim，也不进入 H/I/L/B，
立即返回 orchestration/replay。

正常 submit 的直线控制流不在 H/I/L/B 各步骤重复查询 fatal，避免所有核
对同一 atomic cacheline 形成热竞争。本核在 layout/capacity/task-id
检查或其他中间步骤发现不可恢复错误时，只通过公共封装设置 fatal、记录
错误，并立即退出当前 submit；设置后不得继续执行本 submit 的正常发布或
构建副作用。

已经进入无界 cursor/heap/slot/WonSlot 或 tail-drain 等等待循环的核使用
低频 fatal 检查。等待循环每轮仍只检查其本来的 progress word，并维护
winner-local stall counter；只有 progress word 未变化且本轮协作 drain
也没有产生进展，累计达到 `kFatalPollInterval` 后，才通过公共 atomic
封装读取一次 fatal。progress word 变化或 drain 有进展时立即清零 stall
counter。

等待慢路径观察到 fatal 后，立即退出等待并逐层终止当前 submit/drain，
不得伪造未完成的 H/I commit 或 task completion。设置 fatal 的核不需要
poison 阶段 cursor、frontier 或 slot。`kFatalPollInterval` 的首版取值是
实现调优参数，必须用正常无错误 workload 验证其不会把 fatal cacheline
重新变成竞争热点，同时用 fatal injection 验证所有无界等待都能有界
退出。

因此正常无等待路径只有 submit 入口的一次 fatal load；正常但发生短暂
等待且持续有进展的路径也不产生额外 fatal load。禁止在每轮 spin 中查询
fatal。

## 4. 由本方案前提推导的实现约束

以下约束由第 1 节的候选方案前提和第 3 节的并发规则共同导出。
如果改用其他发布方案，这些约束可能不再成立。

### 4.1 TensorMap 必须支持历史查询

第 3.6 节已经确认 TensorMap 必须保留 writer 历史，并且不再使用单值
`last_writer` 表达依赖关系。

实现通过 append-only、不可变的 TensorMap entries 和 bucket chain 保留
历史信息。

TensorMap entry 的哈希身份来自 Tensor 的 `logical_addr.uint64()`；view
和同一 allocation 的后续 INOUT 因而进入同一个 bucket，internal/external
tag 也参与 identity。当前 shared ref 的 `last_writer` 字段及 exchange
路径必须删除，使 shared ref 也只从该历史索引获得 predecessor。

首版直接沿用并修改现有单 cacheline `SharedRegionEntry`：

```text
MaskPointer logical_addr // 取代 buffer_addr 的物理地址语义
lo, hi
producer_task_id
next_in_bucket        // entry slot index
padding
```

`logical_addr` 取代当前的物理 `buffer_addr`，`[lo, hi)` 是相对该 logical
storage identity 的 byte range。lookup 首版仍按
`logical_addr.uint64()` 相等且
`query.lo < entry.hi && entry.lo < query.hi` 判断相交。

真实高维求交所需的 `start_offset`、`version`、`ndims`、`dtype`、
`is_contiguous`、`shapes[]`、`extent_elem_cache` 和 `strides[]` 暂不作为
实际成员参与首版逻辑；实现中在 `SharedRegionEntry` 定义旁以注释和明确
TODO 保留，后续直接补充并启用到同一个 `SharedRegionEntry`，不再引入
另一种 TensorMap entry 类型。现有 128B `PTO2TensorMapEntry` 只作为字段
组织和求交算法的参考；append-only 方案不需要它的 `prev_in_bucket`、
`next_in_task`、`prev_in_task` 等回收链字段。首版和扩展后的布局都必须用
对应的 `sizeof`、alignment 和 offset `static_assert` 固定 ABI。

`SharedOutputCell::published` 也从候选结构删除。reader 不再等待
per-output descriptor flag；它在协议规定的位置观察 task-level H/I
cursor 后，invalidate 并读取 immutable `SharedOutputCell::tensors`。

### 4.2 H/I 发布点和 A5 可见性实现

按照第 3.4 节的 task 级提交规则，`I(N)` 完成必须表示 task N 的全部
TensorMap entries 已经可供 reader 查询。没有 entry 时，`I(N)` 完成
表示 empty commit 已经推进 `next_insert_task_id`。

已确认的抽象 happens-before 关系为：

```text
完成 task N 的逻辑 heap reservation 和 shared logical Tensors
  -> publish next_heap_task_id = N+1
  -> task N+1 获得 H 权限

初始化 task N 的全部 TensorMap entries
  -> 批量 flush entries
  -> atomic publish 受影响的 bucket heads
  -> publish next_insert_task_id = N+1
  -> 后续 task observe next_insert_task_id > N
  -> 读取 task N 的 entries
```

H 使用现有 `shared_heap_vend` 的具体顺序为：

```text
在 H 外完成 calc_output_layout(N)
等待 atomic_load(next_heap_task_id) == N
old_vend = atomic_load(shared_heap_vend)
task_vend = old_vend
若有新 OUTPUT：
  使用预计算 layout 得到 task_base/task_vend，包括 ring-tail padding
  初始化 task N 的全部 SharedOutputCell::tensors
  批量 flush Tensor descriptors + dsb
  atomic_exchange(shared_heap_vend, task_vend)
atomic_exchange(task_cell(N).vend, task_vend)
store_barrier()
atomic_exchange(next_heap_task_id, N+1)
```

没有新 OUTPUT 时跳过 `shared_heap_vend` 更新和 Tensor descriptor flush，
但仍发布继承的 per-task vend，并执行最后的 `next_heap_task_id` empty
commit。`shared_heap_cursor[]` 不再参与地址分配或反压，并从候选结构、
初始化和 onboard invalidate 路径中删除。

AICore 原子封装会忽略 `__ATOMIC_RELEASE/ACQUIRE` 参数，因此这些名称
不能单独证明 onboard 可见性。H、I 两个发布边界都必须建立“writer
flush+dsb 后发布 cursor，reader 观察 cursor 后 invalidate+dsb 再读取”
的顺序，并通过生成代码审计和 A5 onboard litmus 验证。

`bucket.head` 已确定保留 atomic 读写。I 的 task 级批量发布顺序为：

```text
为 task N 预留连续 entry slots
读取各 bucket 的旧 head，并初始化全部 entries/next_in_bucket
flush 整个连续 entry 区间 + dsb
对每个受影响 bucket 执行 atomic_exchange(head, new_head)
store_barrier()
atomic_exchange(next_insert_task_id, N+1)
```

同一 task 的 entries 连续分配，因此只需一次 region flush 和一次 dsb，
不再逐 entry flush，也不在 heads 或 cursor 发布后重复 flush。多个 entries
落入同一 bucket 时，先在未发布的 entries 间构造本 task 的局部链，再为
该 bucket 发布一次最终 new head。

reader 使用 atomic load 读取最新的 `bucket.head`，再 invalidate 并读取
不可变 entry 链。A5 上现有封装分别以 `atomicAdd(addr, 0)` 和
`atomicExch` 实现这两个操作。这样无需为一个 head 标量额外执行整条
cacheline 的 writeback/invalidate，但不能替代 entry 本体的批量
flush/invalidate。

所有 atomic 操作都必须调用项目公共 atomic 封装，不能在 TensorMap、
阶段游标或其他业务代码中直接调用 `atomicAdd`、`atomicExch` 等上板硬件
原语。sim 和 onboard 必须使用同一组调用入口，由封装内部的 target
分支分别映射到 host atomic 与 A5 硬件原语。底层平台实现细节只允许
出现在 atomic 封装内部。`bucket.head` 具体使用现有的 `atomic_load`
和 `atomic_exchange`。

这里保留 atomic 是标量发布与一致读取的实现选择，不是恢复 bucket 锁：
协议中没有 `-2` 锁定值，没有 writer-writer 竞争，也没有 CAS/RMW
重试。多个 head 更新完成后通过公共 `store_barrier()` 封装再发布
`next_insert_task_id`；该封装的 A5 映射是否充分，以及 reader 观察
cursor 后的精确顺序，仍需由 onboard litmus 确认。

### 4.3 TensorMap entry 生命周期与分配

首版 TensorMap 在一次 runtime/replay 生命周期内采用 append-only
entries：

- TensorMap 容量以 entry 数量计，不以 task 数量计；使用独立的
  编译期常量 `kSharedTensorMapCap = 1 << 16`，不与 `kFlagCap` 绑定；
- `entry_high_water` 只由当前 I writer 普通读写，使用
  H 后、I 前准备的 `writer_count` 一次性预检并预留连续 slots，不需要
  atomic；
- entry 发布后保持不可变，运行中不回收、不复用；
- 只有在确认不存在任何 I/L/B/E 访问 TensorMap 的全局静默边界，才允许
  reset 整个 map 和 allocation cursor；
- 容量不足时设置 fatal，不能等待形成反压，因为本协议中没有运行时回收
  动作可以解除等待。

容量检查必须发生在 task N 写入任何 entry 或更新任何 bucket head
之前。容量足够时，I 才把 `entry_high_water` 一次推进
`writer_count`；容量不足时，在发布任何 task N writer 前设置 fatal。
这样不会留下只插入了部分 writer 的 task。统计发生在 H commit 后的
winner-local ref/args 准备中，但 entry reservation 仍属于获得 TensorMap
独占写权限后的 I。

不能仅因某个 producer 已执行完成就立即复用其 entry。`I(N+1)` 可以与
`L(N)` 并发，reader 可能仍在遍历旧 entry；没有 reader epoch 或等价
安全回收协议时，复用会产生 reader/reclaimer 竞态。动态回收属于后续
独立方案，不进入首版实现。

普通 kernel task 中按 insert 规则跟踪的多个 OUTPUT、INOUT 或
OUTPUT_EXISTING 分别占用各自的 TensorMap entry；`alloc_tensors` 的
OUTPUT 不占用 entry。首版 `SharedRegionEntry` 保持 64B，因此
`kSharedTensorMapCap` 对应 4 MiB entry pool。即使它与当前 `kFlagCap`
数值相同，也必须定义成独立常量，不能写成 `kFlagCap` 的别名；容量单位
始终是 entry，而不是 task。以后启用真实高维 entry 布局时，再根据实际
entry 大小和 GM 总预算重新核算该常量，不用
`task_count * MAX_TENSOR_ARGS` 预设最坏情况。

首版同样保持现有有界 task-id 生命周期：`task_id >= kFlagCap` 时直接
设置 fatal，不在一次 runtime/replay 内对 task id 做 modulo wrap。task
cell、`SharedOutputCell` 和 TensorMap entry 中的 `producer_task_id`
因而都可以继续使用无 generation 的 task id。以后若支持 wrap，必须同时
为这些索引和引用定义 generation/epoch 以及静默复用边界，不能只对
task id 取模。

## 5. 实现顺序

协议语义已经完成对齐。实现前先做只读代码差异审计，再按以下依赖顺序
修改：

1. 公共基础：`MaskPointer`、Tensor ABI 和多维 `FdwicOutputRef`；
2. H 流水：`next_heap_task_id`、`shared_heap_vend`、per-task vend，并删除
   heap shard/cursor；
3. I/L：单写 append-only TensorMap、task-id 历史 lookup、producer
   去重和 `kMaxFanin` 已知限制 TODO；
4. B：logical-to-physical 映射、heap backpressure、惰性 frontier 和
   `B_alloc`；
5. 执行集成：删除延迟 ref/`last_writer` 路径，适配 MIX WonSlot 和 AIV
   `fanin_count = 0`；
6. sim、并发压力、A5 可见性和性能验证。

每一阶段开始编码前，都要先列出对应文件、类型、函数、需删除路径和新增
测试；不能只按名称机械替换。真实高维求交首版继续使用 `[lo, hi)`，在
`SharedRegionEntry` 旁保留字段和算法 TODO，不阻塞前五阶段。

## 6. 验证方案

### 6.1 验证总原则

- 编译通过只说明编译门禁通过，不能作为功能正确性结论；
- 每个实现阶段根据该阶段实际代码影响、修改或新增的用例以及第 6.6 节
  协议矩阵确定验证范围，不能用固定 smoke 代替受影响协议的专用用例；
- 每完成一个实现阶段或一次语义修复后，必须重新编译 native binding、
  A5Sim/A5 runtime 和本阶段动态生成的 orchestration/incore kernels，
  再运行本阶段全部必需用例；
- 禁止使用上一阶段、另一 commit、另一 map mode、另一编译器或另一
  PTO ISA revision 生成的 `.o`、`.so`、runtime 或 kernel 作为本阶段
  证据；
- 验证前必须确认实际源码、native binding、runtime、动态 kernel 和
  PTO ISA checkout 属于本轮记录的同一 source state、map mode 与
  revision；
- 需要排除缓存时必须隔离 `build/cache`、`build/lib`、native binding
  和相关目标文件，确保对应 `.o`、`.so`、runtime 与 kernel 重新生成；
- 不得修改项目配置、golden、timeout、依赖规则或用例参数来掩盖失败；
  如确需改变配置，必须先把它作为独立验证变量记录并重新定义通过条件；
- 功能、并发压力、可见性 litmus 和性能/profiling 是不同证据，不能
  互相替代；
- 所有源码检查、旧产物隔离、编译和 A5Sim 在普通 shell 中执行；所有
  实际访问 A5 设备的操作只通过 `task-submit`；
- A5 onboard 当前仍必须通过 `task-submit`，A5Sim 不使用
  `task-submit`；当前环境没有 `npu-smi` 时，仅按已有用户授权跳过
  arch precheck，不能跳过 onboard 用例；
- 任一必需验证失败时停止当前阶段，保留 source state、构建现场、完整
  命令和日志，先确定根因，不能用后续改动或偶然通过覆盖失败。

### 6.2 验证环境

验证统一从项目根目录使用本地 `build_runtimes.sh` 和 `run_tests.sh`。
它们是验证工具，不属于产品实现。两个脚本必须使用完全相同的环境参数：

- 项目根目录 `.venv` 已存在，并包含 pytest、nanobind 和用例依赖；
- 外部 shell 设置 `ASCEND_HOME_PATH`，随后 source
  `$ASCEND_HOME_PATH/set_env.sh`，不另外维护一套 CANN 路径；
- `PTO_ISA_ROOT` 指向项目 `build/` 之外的独立、干净 checkout；
- `PTO_ISA_COMMIT` 与 CI revision 一致；当前默认 pin 为
  `ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8`；
- `gcc` 和 `g++` 都必须是真正的 major version 15；
- `PYTHONPATH` 把项目根目录和 `python/` 放在已有值之前；
- shared 模式设置
  `CXXFLAGS=-DPTO_FDWIC_SHARED_MAP=1`，需要 private 对照时设置为 0。

首次开始端到端验证且项目根目录尚不存在这两个脚本时，按本节模板各创建
一次，然后执行：

```bash
chmod 700 build_runtimes.sh run_tests.sh
```

后续源码修改、commit 切换或 shared/private 切换只重新运行脚本，不因此
删除或重新生成脚本。只有本节脚本规范本身改变时才同步更新它们。创建时
必须把两个模板中的 `/path/to/pto-isa` 替换为当前环境真实的、位于项目
`build/` 外部的干净 PTO ISA checkout；两个脚本中的
`PTO_ISA_ROOT`/`PTO_ISA_COMMIT` 必须完全相同。

#### 6.2.1 生成 `build_runtimes.sh`

项目根目录的 `build_runtimes.sh` 使用以下完整内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <expected-simpler-commit> <shared|private>" >&2
    exit 2
fi

expected_simpler_commit="$1"
map_mode="$2"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv_path="$repo_root/.venv"

case "$map_mode" in
    shared) shared_map="1" ;;
    private) shared_map="0" ;;
    *)
        echo "Invalid map mode: $map_mode (expected shared or private)" >&2
        exit 2
        ;;
esac

cd "$repo_root"
test "$(git rev-parse --show-toplevel)" = "$repo_root"
source "$venv_path/bin/activate"

: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH must be set}"
source "$ASCEND_HOME_PATH/set_env.sh"

export PTO_ISA_ROOT="/path/to/pto-isa"
export PTO_ISA_COMMIT="ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8"
export CXXFLAGS="-DPTO_FDWIC_SHARED_MAP=$shared_map"
export PYTHONPATH="$repo_root/python:$repo_root${PYTHONPATH:+:$PYTHONPATH}"

test "$(git rev-parse HEAD)" = \
    "$(git rev-parse "$expected_simpler_commit^{commit}")"
test "$(gcc -dumpversion | cut -d. -f1)" = "15"
test "$(g++ -dumpversion | cut -d. -f1)" = "15"
test -d "$PTO_ISA_ROOT/.git"
test "$(git -C "$PTO_ISA_ROOT" rev-parse HEAD)" = "$PTO_ISA_COMMIT"
test -z "$(git -C "$PTO_ISA_ROOT" status --short)"

stamp="$(date +%Y%m%d-%H%M%S)"
build_backup="$repo_root/../simpler-build-backup-$stamp"
binding_backup="$repo_root/../simpler-binding-backup-$stamp"

if [ -e "$repo_root/build" ]; then
    test ! -e "$build_backup"
    mv "$repo_root/build" "$build_backup"
fi
mkdir -p "$repo_root/build"

if compgen -G "$repo_root/python/_task_interface*.so" >/dev/null; then
    test ! -e "$binding_backup"
    mkdir -p "$binding_backup"
    mv "$repo_root"/python/_task_interface*.so "$binding_backup"/
fi

test -z "$(
    find "$repo_root/build" \
        -type f \
        \( -name '*.o' -o -name '*.so' -o -name CMakeCache.txt \) \
        -print -quit
)"

strip_shim="$(mktemp -d /tmp/simpler-aarch64-strip.XXXXXX)"
trap 'rm -rf "$strip_shim"' EXIT
ln -s \
    "$ASCEND_HOME_PATH/tools/hcc/aarch64-target-linux-gnu/bin/strip" \
    "$strip_shim/strip"
export PATH="$strip_shim:$PATH"

nanobind_dir="$(python -m nanobind --cmake_dir)"
cmake \
    -S "$repo_root" \
    -B "$repo_root/build/python-binding" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$(command -v python)" \
    -Dnanobind_DIR="$nanobind_dir"

cmake \
    --build "$repo_root/build/python-binding" \
    --target _task_interface \
    -j"$(nproc)"

test -n "$(
    find "$repo_root/python" \
        -maxdepth 1 \
        -type f \
        -name '_task_interface*.so' \
        -print -quit
)"

python "$repo_root/simpler_setup/build_runtimes.py" \
    --lib-dir "$repo_root/build/lib" \
    --cache-dir "$repo_root/build/cache" \
    --platforms a5sim a5 \
    --pto-isa-commit "$PTO_ISA_COMMIT"
```

该脚本只负责环境检查、旧产物隔离和干净编译，不运行测试，也不占用
`task-submit`。同一 commit、map mode、编译器和 PTO ISA revision 下，
一次构建可供多个用例复用。

#### 6.2.2 生成 `run_tests.sh`

项目根目录的 `run_tests.sh` 使用以下完整内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <expected-simpler-commit> <shared|private> <a5|a5sim> <repeat-count> <pytest-target> [<pytest-target> ...]" >&2
    exit 2
fi

expected_simpler_commit="$1"
map_mode="$2"
platform="$3"
repeat_count="$4"
shift 4
pytest_targets=("$@")
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv_path="$repo_root/.venv"

case "$map_mode" in
    shared) shared_map="1" ;;
    private) shared_map="0" ;;
    *)
        echo "Invalid map mode: $map_mode (expected shared or private)" >&2
        exit 2
        ;;
esac

if ! [[ "$repeat_count" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid repeat count: $repeat_count (expected a positive integer)" >&2
    exit 2
fi

cd "$repo_root"
test "$(git rev-parse --show-toplevel)" = "$repo_root"
source "$venv_path/bin/activate"

: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH must be set}"
source "$ASCEND_HOME_PATH/set_env.sh"

export PTO_ISA_ROOT="/path/to/pto-isa"
export PTO_ISA_COMMIT="ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8"
export CXXFLAGS="-DPTO_FDWIC_SHARED_MAP=$shared_map"
export PYTHONPATH="$repo_root/python:$repo_root${PYTHONPATH:+:$PYTHONPATH}"

test "$(git rev-parse HEAD)" = \
    "$(git rev-parse "$expected_simpler_commit^{commit}")"
test "$(gcc -dumpversion | cut -d. -f1)" = "15"
test "$(g++ -dumpversion | cut -d. -f1)" = "15"
test -d "$PTO_ISA_ROOT/.git"
test "$(git -C "$PTO_ISA_ROOT" rev-parse HEAD)" = "$PTO_ISA_COMMIT"
test -z "$(git -C "$PTO_ISA_ROOT" status --short)"

common_pytest_args=(
    "${pytest_targets[@]}"
    --platform "$platform"
    -p no:xdist
    -v
    --require-pto-isa
    --pto-isa-commit "$PTO_ISA_COMMIT"
)

case "$platform" in
    a5sim)
        for ((iteration = 1; iteration <= repeat_count; iteration++)); do
            echo "[run_tests] a5sim iteration $iteration/$repeat_count"
            python -m pytest "${common_pytest_args[@]}"
        done
        ;;
    a5)
        onboard_command=(
            env
            "PTO_ISA_ROOT=$PTO_ISA_ROOT"
            "PTO_ISA_COMMIT=$PTO_ISA_COMMIT"
            "CXXFLAGS=$CXXFLAGS"
            "PYTHONPATH=$PYTHONPATH"
            "$venv_path/bin/python"
            -m pytest
            "${common_pytest_args[@]}"
            --device __TASK_DEVICE__
        )
        printf -v onboard_command_string '%q ' "${onboard_command[@]}"
        onboard_command_string="${onboard_command_string//__TASK_DEVICE__/'$TASK_DEVICE'}"
        printf -v shell_command 'bash -ic %q' "$onboard_command_string"
        for ((iteration = 1; iteration <= repeat_count; iteration++)); do
            echo "[run_tests] a5 iteration $iteration/$repeat_count"
            task-submit \
                --timeout 90 \
                --max-time 90 \
                --device auto \
                --device-num 1 \
                --run "$shell_command"
        done
        ;;
    *)
        echo "Invalid platform: $platform (expected a5 or a5sim)" >&2
        exit 2
        ;;
esac
```

该脚本接收一个或多个 pytest targets。A5Sim 直接执行 pytest；A5 在
普通 shell 完成环境和 revision 检查，只把最终 pytest 命令交给
`task-submit`，避免环境准备占用设备锁。

#### 6.2.3 脚本生成与维护规则

- 两个脚本只在首次需要端到端验证且文件不存在时按模板生成；不能从旧
  build、临时 runner 或 shell history 拼装；
- 若文件已经存在，先核对它与当前模板的接口和环境约束，不因普通源码
  修改、commit 切换或 map mode 切换而重建；
- 模板规范改变时，必须同时更新两个脚本，尤其不能只更新其中一个
  `PTO_ISA_ROOT` 或 `PTO_ISA_COMMIT`；
- `PTO_ISA_ROOT` 不能位于会被构建脚本隔离的项目 `build/` 下；
- `.venv` 必须在脚本生成前准备完成，脚本不负责安装或升级 Python
  dependencies；
- 外部 shell 应先设置 `ASCEND_HOME_PATH`。`set_env.sh` 把它从
  `latest` 规范化为具体 CANN 版本目录是正常行为，脚本不能要求 source
  后仍保留 `latest` 字面值；
- CANN 的 `PATH`、`LD_LIBRARY_PATH`、`CMAKE_PREFIX_PATH` 和
  `ASCEND_OPP_PATH` 等由 `set_env.sh` 提供，不在脚本中维护第二套手写
  路径；
- 脚本把项目根目录和 `python/` 前置到 `PYTHONPATH`，但保留已有
  `PYTHONPATH` 在其后；
- `gcc` 和 `g++` 必须分别检查，不能只设置 `CXX` 或只验证 `g++`；
- 两个脚本权限固定为仅 owner 可执行的 `700`，作为本地验证工具，不随
  产品实现提交；
- build 脚本把旧 build 和 native binding 移到项目同级、带时间戳的
  backup 目录，不静默覆盖或删除；需要清理备份时另行确认目标；
- A5 arch precheck 在 `run_tests.sh ... a5 ...` 前完成；没有 `npu-smi`
  时只有已有明确授权才能跳过 precheck，不能据此跳过 onboard 测试。

运行前必须检查：

```text
simpler HEAD == 本轮声明的 expected commit
PTO ISA HEAD == PTO_ISA_COMMIT
PTO ISA worktree clean
gcc major == 15
g++ major == 15
构建模式 == 本轮 runner 的 shared/private 参数
```

默认执行 A5 arch precheck。环境没有 `npu-smi` 时，只有在已有明确授权
下才能跳过 precheck；该豁免不能跳过 onboard 用例。

### 6.3 干净构建

以下任一条件变化后，必须重新运行：

```text
./build_runtimes.sh <expected-simpler-commit> <shared|private>
```

- 修改 runtime、platform、orchestration 或 incore kernel；
- 切换 shared/private；
- 切换 commit 或 branch；
- 修改编译器、编译参数或 PTO ISA revision。

构建脚本必须先隔离或清理旧 `build/`、native binding 和相关
`.o`/`.so`/CMake cache，再重新构建 `_task_interface` 以及 A5Sim/A5
runtimes。禁止用前一阶段或另一宏模式的产物代替重新构建。构建通过只
表示编译门禁通过，不能作为功能正确性结论。

同一 simpler commit、map mode、GCC/G++、编译参数和 PTO ISA revision
均未变化时，一次 runtime 构建可以供多个 pytest targets 和多轮
A5Sim/A5 调用复用。任一项变化后都必须重新构建。

pytest 会动态编译 orchestration/incore kernels，因此
`run_tests.sh` 的 map mode 必须和最近一次 `build_runtimes.sh` 完全一致。
若先验证 shared、再验证 private、最后回到 shared 调试，三个切换点都要
各自重新构建，不能把最后一次 private 产物当作 shared 集成状态。

禁止用 `python -m pip install '.[test]'` 代替本节构建流程：该命令可能
探测并构建无关平台，也可能把 A5 专用宏传播到错误目标。Python 依赖在
预先准备的 `.venv` 中管理，native binding 和 runtimes 只由
`build_runtimes.sh` 生成。

### 6.4 用例执行入口

runner 接口为：

```text
./run_tests.sh \
  <expected-simpler-commit> \
  <shared|private> \
  <a5sim|a5> \
  <repeat-count> \
  <pytest-target> [<pytest-target> ...]
```

`repeat-count` 必须是正整数。runner 在启动 pytest 前重复检查 simpler
commit、编译器、PTO ISA 和 map mode，并统一传入：

```text
--platform <a5sim|a5>
-p no:xdist
-v
--require-pto-isa
--pto-isa-commit <PTO_ISA_COMMIT>
```

A5Sim 每轮直接启动一次 pytest。所有 A5 设备操作必须通过
`task-submit`；普通 shell 只完成环境检查，然后把最终 pytest 命令交给
设备任务，并通过 `$TASK_DEVICE` 传递 `--device`。每一轮 A5 重复必须
重新提交一次独立 `task-submit`，不能用一个设备任务包住多轮 pytest。
普通功能用例使用 90 秒 `timeout/max-time`；stress 或性能任务只有在
记录轮数、预计时长和理由后才能提高。

多个 pytest targets 可以在一次 `run_tests.sh` 调用中传入，并共享同一
次 runtime 构建。需要 pytest option 时直接追加在 targets 后，例如用例
明确要求纳入 manual cases 时追加 `--manual include`。是否追加某个
option 必须由被选用例本身决定，不能把一个用例的参数套到全部用例。

功能验证若定义了 golden，必须运行真实 kernel 和真实 golden。
`--use-example-exec-time` 只允许用于明确的 sim 性能分析，
`--skip-golden` 不能用于形成正确性结论。性能任务应与功能任务分开
调用，避免性能配置改变功能门禁语义。

标准调用形式为：

```bash
COMMIT=$(git rev-parse HEAD)

./build_runtimes.sh "$COMMIT" shared
./run_tests.sh "$COMMIT" shared a5sim 1 <本阶段 targets...>
./run_tests.sh "$COMMIT" shared a5 1 <本阶段 targets...>
```

是否需要 private 对照、具体 pytest targets、平台和重复轮数由本方案的
验证矩阵决定。

每个实现阶段的实际执行顺序为：

1. 固定并记录 simpler commit、map mode、GCC/G++ 和 PTO ISA revision；
2. 运行一次对应 mode 的干净 `build_runtimes.sh`；
3. 按本阶段验证矩阵选择 targets，先执行所需 A5Sim 功能/压力用例；
4. 完成 A5 precheck 后，每轮通过独立 `task-submit` 执行所需 onboard
   用例；
5. 功能通过后再单独运行本阶段要求的性能/profiling；
6. 记录每个 target 的完整命令、轮数、耗时、pass/fail 和未覆盖风险。

任何必需用例失败时停止当前阶段，保留 commit、构建产物和日志进行分析；
不能继续用后续改动或后续轮次的通过结果覆盖该失败。

### 6.5 验证结论与记录

- 编译通过不能代替功能验证；
- golden 功能用例必须运行真实 kernel 和真实 golden，不能用
  `--use-example-exec-time` 或 `--skip-golden` 得出功能结论；
- 性能/profiling 与功能 golden 分开运行和报告，不能互相替代；
- 失败时保留当前 commit、构建模式、日志和可重复条件，不用后续结果
  覆盖；
- 每份结果记录 simpler commit、GCC/G++、PTO ISA revision、map mode、
  平台、完整命令、repeat、pass/fail、耗时和未覆盖风险。

### 6.6 本候选方案特有的验证矩阵

本方案从已确认协议反推以下验证范围。代码差异审计阶段要把每一行落实为
“可复用的现有用例”或“需要新增的最小用例”，并明确 A5Sim/A5 平台、
运行轮数和通过条件。

| 范围 | 必须验证的协议性质 |
| --- | --- |
| Tensor/ABI | `sizeof`/offset/alignment；internal/external logical tag；所有 constructor、copy、view 和 fast path 完整传播 `MaskPointer`；external 8B 对齐失败 |
| H | 有/无/多个 OUTPUT 都推进 cursor；无 OUTPUT 继承 vend；descriptor 先完整发布再推进 cursor；H(N) 可与 I(N-1) 并发 |
| Ref/view | 二维、五维、嵌套 view；H 后 I 前全部解析；`manual_dep` 在解析后准确排除；无 unresolved ref 进入 slot/WonSlot |
| I | empty commit；单/多 entry；同 bucket 局部链只发布一次 head；容量 fatal 前不留下部分 task entries；业务代码只调用公共 atomic 封装 |
| L | 忽略 self/future writer；返回多个历史 producer；显式、owner 和 map producer 全局去重；`manual_dep` 跳过 map；`kMaxFanin` overflow 分支保留明确 TODO |
| B/B_alloc | ring-tail padding；只使用 task-local vend；未来 H 不反向阻塞较早 B；容量不足才惰性推进 frontier；无法容纳单 task 时 fatal；initial fill 位于容量门后 |
| MIX | 只有 AIC 执行 task-level H/I/L/B；AIV `fanin_count = 0`；WonSlot 只在 AIC 完整 fanin ready 后发布；三个协作 drain 检查点均能推进 |
| Fatal | 正常 submit 只有入口一次 atomic load；错误核设置后立即退出；cursor/heap/slot/WonSlot/tail-drain 无进展时低频检查；未提交 H/I 的 fatal 不造成永久等待或伪提交 |
| 可见性 | writer flush+dsb 后发布 cursor、reader observe 后 invalidate+dsb；调度扰动下不能读取半初始化 Tensor/entry；A5 生成代码审计和 onboard litmus |

并发压力用例需要主动扰动 H/I/L/B 的相对进度，至少覆盖：

- `H(N)` 与 `I(N-1)` 重叠；
- `I(N+1)` 已发布而 `L(N)` 仍在扫描；
- 多个未来 task 的 bucket heads 已可见，但 lookup cutoff 仍为 N；
- 连续 empty H/I commits；
- heap 容量长期不足时，各等待循环依靠 drain 推进并最终解除反压；
- MIX 的 AIC/AIV 到达不同 drain 检查点；
- 分别在 calc/H/I/B 和 tail-drain 注入 fatal，并让其他核预先进入对应
  等待；记录退出迭代数、fatal load 次数以及是否出现伪 commit、
  completion、timeout 或 deadlock。

### 6.7 尚待实现验证收敛的事项

以下不再是协议选择，但必须在对应实现落地时补齐证据：

1. 公共 barrier/cache 封装的精确 A5 映射及 H/I onboard litmus；
2. sim 和 onboard 压力轮数、随机种子、超时与可重复失败信息；
3. 以当前 Shared Map 为 baseline 的吞吐、Claim/H/I/L/B 分段耗时和
   TensorMap/heap 等待指标；
4. 真实高维 `SharedRegionEntry` 启用后的求交正确性与容量预算。

[current-model]: shared_tensormap_record.md
[protocol-comparison]: tensormap_inout_issue.md

## 7. 实现进展台账

文档确认并开始编码后，在本节尾部按时间顺序追加所有实现进展。每个实现
commit 对应一条独立记录；不能只在最终完成时补写汇总，也不能用后续
结果覆盖早期失败或错误假设。

### 7.1 记录规则

- 提交时间使用带时区的 ISO 8601 格式，统一记录 Asia/Shanghai 对应的
  `+08:00` 时间；
- 记录最终 commit hash 和 subject，并标明所属实现阶段、状态
  （进行中/完成/回退）及其依赖的前序 commit；
- “前因”说明当前代码限制、触发问题或上一阶段为什么不足；“后果”说明
  本提交改变了哪些协议性质、数据布局、控制流和兼容边界；
- 变更内容至少精确到主要文件、类型和函数；删除旧路径时同时记录删除
  原因和替代路径；
- 每条设计结论要能回指本文对应章节，若实现中发现协议需修改，先更新
  协议并记录原因，不能静默偏离；
- 记录完整验证环境：source state、map mode、GCC/G++、PTO ISA、
  A5Sim/A5、设备任务信息、关键编译参数和是否干净重建；
- 记录实际执行的完整 build/test 命令、pytest targets、参数、随机种子、
  repeat、样本数、warmup、timeout 和日志/产物位置；
- 功能结果不能只写“pass”：至少记录 pass/fail/skip 数、总耗时、每轮
  结果；失败时记录错误类型、首次失败轮次、可复现频率和关键差异；
- 并发/压力结果至少记录 task 数、block/core 配置、调度扰动方式、轮数、
  成功率、fatal/timeout/deadlock 数和最长完成时间；
- 性能结果必须提供同环境 baseline 与候选实现的原始数据或汇总数据，
  包括样本数、单位、min/median/p95/max、吞吐或延迟变化百分比；若有
  Claim/H/I/L/B、TensorMap 或 heap wait 分段数据，一并记录；
- A5 可见性验证记录 `task-submit` 任务标识、litmus 迭代数、观察到的
  outcome 计数和 forbidden outcome 是否为零；
- 未运行的必需验证必须显式写“未运行”、原因、风险和补测条件，不能
  留空或写成通过；
- 台账采用 append-only 语义。事实订正、回退、rebase 或 squash 不静默
  改写历史条目，而是追加订正及 old-hash -> new-hash 映射；
- 进展台账不能替代详细 commit message。非平凡实现 commit 的 message
  同样必须包含背景/根因或设计理由、关键修改和实际验证。

Git commit 无法在自身内容中保存自己的最终 hash。工作流采用：

1. 实现并完成本 commit 计划内验证；
2. 在同一变更中先追加完整台账条目，hash 暂写 `待回填`；
3. 创建实现 commit 后取得最终 hash；
4. 在下一次台账更新中回填该 hash；最后一个实现 commit 由最终的
   documentation-only 台账收尾提交回填。

纯粹用于回填 hash 或整理台账的 documentation-only commit 不算新的实现
进展，避免产生无限自引用；但其操作时间和用途必须写在所回填条目的
“台账更新”字段中。

### 7.2 单条记录模板

```text
### <序号> <实现阶段/主题>

- Commit：<hash 或待回填>（<subject>）
- 提交时间：<YYYY-MM-DDTHH:MM:SS+08:00>
- 状态：<进行中/完成/回退>
- 前序依赖：<commit/hash/协议章节>
- 前因与目标：
  - <当前限制、问题或设计原因>
- 实现内容：
  - <文件::类型/函数：行为变化>
  - <删除路径及替代方案>
- 协议影响：
  - <对应本文章节、保持或改变的 invariant>
- 验证环境：
  - source/map mode/compiler/PTO ISA/platform/device/build state
- 验证命令：
  - `<完整命令>`
- 功能结果：
  - cases=<n>, pass=<n>, fail=<n>, skip=<n>, repeat=<n>, time=<...>
- 压力/可见性结果：
  - tasks/blocks/cores/seeds/iterations/outcomes/timeouts/fatals/time
- 性能结果：
  - baseline/candidate/samples/unit/min/median/p95/max/delta
- 未覆盖风险与下一步：
  - <未运行项、原因、风险、补测条件>
- 台账更新：
  - <回填 hash 的 documentation-only commit 时间或订正映射>
```

### 7.3 当前状态

尚未开始候选方案实现；当前仅完成协议对齐和实现/验证计划归档。
