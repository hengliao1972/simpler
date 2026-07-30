# Shared TensorMap PA 快路：为什么当前用例不需要 ordinary TensorMap 插入

## 1. 结论与范围

本文讨论的是 standalone shared TensorMap 的 PA Case1 路径，源码基线为
`fd463c4c`。

准确结论是：

> 当前 PA Case1 不需要向按地址区间组织的 ordinary TensorMap
> 插入任何 region entry；fresh output、显式依赖和 INOUT 版本关系已经由
> `SharedOutputRef`、`shared_outputs[]`、writer history 和 task completion
> 分别表达。

这不等于“整个 shared TensorMap sidecar 都没有用”，也不等于 Register
阶段可以直接删除。当前仍然需要：

- task-indexed output descriptor 发布；
- `published` 原子完成字；
- INOUT symbol 的 `last_writer` 与不可变 writer history；
- 按 task id 推进的 `deps_prepared` 完成链；
- 真正 kernel 执行前的 task completion fanin。

本文将按地址区间查找、插入的 bucket/ring 部分简称为
**ordinary TensorMap**，将 `(producer_task_id, output_slot)` 直接寻址路径
简称为 **shared-output 快路**。

## 2. 三类对象不能混为一谈

| 对象 | 身份 | 当前 PA 中的存放位置 | 是否进入 ordinary TensorMap |
| ---- | ---- | -------------------- | --------------------------- |
| fresh output descriptor | `(producer_task_id, output_slot)` | `shared_outputs[task].tensors[slot]` | 否 |
| fresh output 发布状态 | 同一个 task/slot | `shared_outputs[task].published[slot]` | 否 |
| fresh output 最新 writer | 同一个 task/slot | `shared_outputs[task].last_writer[slot]` | 否 |
| INOUT writer 前驱链 | `(writer_task, symbol_key)` | `writer_history[writer_task]` | 否 |
| 普通非 `manual_dep` Tensor region | buffer、offset、shape、owner | ordinary bucket/ring | 是 |
| task 是否执行完成 | task id | task completion flag | 否 |

`SharedOutputRef` 的 ABI 定义在
`common/pa_frontend.h`。它保存稳定的
`producer_task_id` 和 `output_slot`，当前 PA 只接受 view/flags 全零的
plain ref。`SharedTaskOutputs` 只保存当前 task id 和 output count，并在
需要时恢复这些稳定引用，不携带 winner 私有的 `TensorDesc *`。

`SharedOutputCell` 定义在 `common/pa_model.h`。每个 task 独占一格，
`published[]`、`last_writer[]` 和 descriptor payload 分开布置，避免三类
访问共享同一条 cache line。

## 3. 为什么 fresh output 不需要 ordinary TensorMap

ordinary TensorMap 解决的是另一类问题：调用方只有一段 tensor 地址和
region 信息，需要按地址范围寻找它在窗口内的最近 producer/writer，并处理
别名、覆盖和回收。

当前 PA fresh output 的信息更强：

1. producer task id 在 orchestration 构参时已经确定；
2. output slot 在 task 类型中已经确定；
3. 每个 task 恰好只有一个 Claim winner；
4. winner 独占写入 `shared_outputs[task_id]`；
5. consumer 持有精确的 `(producer, slot)`，无需哈希、扫描或比较地址区间；
6. consumer 的执行依赖最终仍由 producer task completion flag 保证。

因此再把同一个 fresh output 以地址 region 形式插入 ordinary TensorMap，
不会补充新的依赖信息，只会重复增加：

- bucket hash 和同 bucket 顺序计算；
- ring/head/tail/seq 访问；
- append 前检查；
- region payload 写入和 DCCI；
- 后续 ordinary lookup；
- 容量、回收和 ABA 管理。

这就是当前 PA 能走 shared-output 快路的根本原因：不是“TensorMap
查找做得更快”，而是业务在构参阶段已经提供了比地址查找更精确的
task/slot 身份，ordinary 查插本身变成了冗余表达。

## 4. fresh output 的完整发布与消费链

### 4.1 producer 发布

唯一 Claim winner 在 Materialize 中完成真实内存分配和 descriptor
构造，随后调用 `PublishSharedTaskOutputs()`：

```text
唯一 task winner
  -> Materialize 得到真实 TensorDesc
  -> 复制到 shared_outputs[task].tensors[slot]
  -> descriptor DCCI clean-out
  -> StoreBarrier
  -> published[slot] Atomic Exchange(task_id)
```

这里不需要额外锁：

- cell 由 task id 唯一定位；
- 一个 task 只有一个合法 winner；
- 只有 winner 写 descriptor payload；
- 其他核只在 `published == producer_task_id` 后读取。

但“不需要锁”不等于“不需要发布协议”。普通 payload 写、DCCI、
StoreBarrier 和原子完成字缺一不可。否则 consumer 可能先看到就绪状态，
再读到尚未写回的 descriptor。

### 4.2 consumer 收集依赖

`CollectSharedFanin()` 遇到 `SharedOutputRef` 时直接定位对应 output cell：

```text
SharedOutputRef(producer, slot)
  -> 检查 producer < consumer
  -> 检查 published[slot]
  -> 读取 last_writer / 必要时沿 writer history 回退
  -> 把得到的 writer task 加入 fanin
```

当前 ordered Submit 在 consumer 完成本 task 的 metadata handoff 后执行
lookup。前序 `deps_prepared` 链已经证明更早 task 完成了 output/metadata
发布，所以正常快路只做一次精确发布检查，不重新开启无界轮询。

BuildWinner 把 descriptor 复制进本核执行 slot 前，还会对共享 descriptor
执行 invalidate。完整可见性链是：

```text
producer 写 descriptor
  -> producer DCCI clean-out
  -> StoreBarrier
  -> published Atomic
  -> consumer 检查 published
  -> consumer DCCI invalidate
  -> descriptor 复制到本核执行 slot
```

随后 kernel 是否可以执行仍由 slot 中保存的 fanin task completion flag
决定。`published` 只表示 descriptor 可读，不能冒充 task 已执行完成。

## 5. QK、SF、PV 有 output，为什么 Register 仍没有 ordinary 插入

PA 每组 task 的 fresh output 数量由 `FrontendTaskOutputCount()` 固定：

| task | fresh output 数 |
| ---- | --------------: |
| Alloc | 3 |
| QK | 1 |
| SF | 3 |
| PV | 1 |
| UP | 0 |

Alloc/QK/SF/PV 的 output 都存在，只是它们已经在 Materialize 中发布到
task-indexed `shared_outputs[]`，而不是在 Register 中追加 ordinary
region entry。

因此泳道中看不到 QK/SF/PV 的 ordinary TensorMap DCCI 是合理现象：
descriptor DCCI 位于 Materialize，不在 Register。不能据此判断这些 task
“没有发布 output”。

## 6. UP 的 INOUT 为什么也不进入 ordinary TensorMap

UP 的三个 accumulator 参数是 `SharedOutputRef + INOUT`。它们需要表达的
不是“这段地址属于哪个 region”，而是“同一个 fresh symbol 的 writer
版本如何从前一组推进到本 task”。

当前使用 symbol writer history：

```text
读取三个 symbol 的 previous writer
  -> 写本 task 独占的 writer_history payload
  -> history DCCI clean-out
  -> StoreBarrier
  -> 三次 last_writer CAS
```

`symbol_key` 由 `(producer_task_id, output_slot)` 无碰撞编码。慢 reader
如果看到指向未来 task 的 `last_writer`，可以沿不可变 history 回退到
严格小于自身 task id 的版本。

这条版本链已经精确表达了 PA INOUT 关系，因此不需要再把三个 accumulator
按地址 region 重复插入 ordinary TensorMap。

UP 还带有一个 `manual_dep` output view。`manual_dep` 明确表示该依赖由
调用方管理，不参加 TensorMap 自动 hazard；它也不能被当作 ordinary
writer 偷偷登记。

## 7. Register 仍然在做什么

“ordinary entry 为 0”不能解释成“Register 为空”。当前 winner Register
仍承担：

1. 等待 task `N-1` 的 `deps_prepared`，保证 metadata 按 task id 发布；
2. 对 UP 发布 writer-history 和三个 `last_writer`；
3. 用本 task 的 completion CAS 推进 `deps_prepared[N]`；
4. 放行 `N+1` owner 进入它的 metadata 发布段。

完成 handoff 后，当前 owner 的 fanin lookup、Build 和 kernel slot
管理在有序通道外继续执行。因此串行的是 metadata 插入/发布顺序，
不是 task Build 和 task 执行。

对 Alloc/QK/SF/PV，writer delta 的 `ordinary_count=0` 且
`symbol_count=0`，Register 主要剩前驱等待和 completion handoff。
对 UP，`ordinary_count` 仍为 0，但 `symbol_count=3`，所以仍有真实的
writer metadata 工作。

`register.publish_writer_metadata[ordinary_tensormap_entries=0]` 只说明
ordinary entry 数量为零，不能说明 symbol metadata 数量也为零。

## 8. 源码如何锁死这条快路的边界

当前实现没有在运行时“有 ordinary entry 就悄悄退回慢路”，而是明确
fail-closed：

- `PrepareSharedTaskWriterDelta()`：
  - `SharedOutputRef` 进入 `symbol_keys[]`；
  - 非 `manual_dep` 的 `GmTensor/LocalTensor` 才进入
    `ordinary_entries[]`。
- `ValidatePreparedPaWriterShape()`：
  - 当前 PA 专用形状要求 `ordinary_count == 0`；
  - 非 UP 要求 `symbol_count == 0`；
  - UP 要求精确三个 symbol key。
- `ValidateEmptySharedRegistration()`：
  - `SharedOutputRef` 和 `manual_dep` 可以越过 ordinary 登记；
  - 任意非 `manual_dep` 普通 writer 都直接失败。

这三个门槛很重要。它们证明当前快路只服务已知 PA 形状，不会把尚未覆盖
的普通 region 依赖静默漏掉。

## 9. B256 已验证的数量闭合

G1、B256 一共有 1,280 个 task。已有 A5 全量泳道和 host oracle 对当前
业务形状给出以下闭合结果：

| 项目 | 数量 | 含义 |
| ---- | ---: | ---- |
| task | 1,280 | 256 × `Alloc/QK/SF/PV/UP` |
| fresh output descriptor | 2,048 | 每 batch `3+1+3+1+0=8` |
| descriptor flush task | 1,024 | Alloc/QK/SF/PV 各 256 次 |
| shared symbol INPUT load | 1,280 | 每组 5 条显式 input 读取 |
| dependency edge | 1,280 | 每组 5 条最终 fanin |
| logical INOUT symbol commit | 768 | 256 个 UP × 3 |
| UP writer-history flush | 256 | 每个 UP 一次整批 history DCCI |
| per-task insert completion | 1,280 | 每 task 推进一次 `deps_prepared` |
| ordinary TensorMap entry | 0 | 没有非 `manual_dep` ordinary writer |
| ordinary append flush | 0 | 没有 ordinary payload 需要发布 |
| region logical/physical append | 0 | ordinary ring 保持初始状态 |

这些数据同时说明两件事：

- output、fanin 和 INOUT 发布都真实存在；
- ordinary TensorMap 的查插确实没有参与当前 PA 数据流。

## 10. 什么时候必须重新启用 ordinary TensorMap

以下任一条件出现，都不能继续套用当前 PA 快路：

1. 参数是非 `manual_dep` 的 `GmTensor/LocalTensor`；
2. consumer 只有地址/region，无法得到稳定的 `(producer, slot)`；
3. 不同 tensor descriptor 可能别名到同一物理 region；
4. subview/offset/shape 语义不能由当前 plain `SharedOutputRef` 表达；
5. 需要按地址寻找窗口内最近 reader/writer；
6. writer 关系不是当前 symbol history 可以完整表达的单链；
7. task id 或 output cell 在同一轮发生复用，却没有 generation/ABA
   证明。

此时 ordinary TensorMap 不是“可选的慢实现”，而是正确性所需的数据
结构。必须恢复 region lookup/append、DCCI、容量和回收协议，并补充相应
门槛测试。

## 11. 后续优化时应保持的边界

可以继续消减的内容：

- PA 已知形状下的通用 ordinary 分支、空 delta 扫描和无效统计；
- `ordinary_count=0` 时无业务意义的地址准备；
- symbol-only Register 中可由 task 类型直接推导的重复解码；
- 不改变发布顺序的纯本地控制流。

不能仅凭 ordinary entry 为 0 就删除的内容：

- `shared_outputs[]` descriptor payload；
- descriptor clean-out 和 consumer invalidate；
- `published` 原子完成字；
- UP writer history、history DCCI 和 `last_writer` CAS；
- `deps_prepared` 的 per-task metadata 完成链；
- kernel 执行前的 task completion fanin。

性能分析中也应保持三类口径分离：

```text
ordinary TensorMap：地址 region 的 lookup/append
shared-output 快路：descriptor 与 symbol writer 的发布/读取
task scheduler：Claim、deps_prepared、fanin completion 与执行 slot
```

只有第一类在当前 PA Case1 中为零。后两类仍是 shared 方案的真实成本和
后续优化对象。
