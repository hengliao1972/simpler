# SharedOutputRef 依赖查询方案分析

## 1. 目的与结论

本文供方案评审，分析 shared TensorMap 模式下，`SharedOutputRef`
为什么使用独立的 symbol writer，以及是否应改为与 private 一致：

```text
SharedOutputRef
  -> 解析 TensorDesc
  -> 按物理地址区间查询 ordinary TensorMap
  -> 得到 fanin producer
```

分析基于提交 `c3aaf99f` 的 PA standalone 实现，范围限定为
`tests/atomic_probe/pa_scheduler`。本文不代表已决定修改实现，也不宣称
任一方案已经取得性能收益。

结论如下：

1. shared 模式仍然构造正常的 `fanin`。symbol 是查找 writer 的 key，
   不是 `fanin` 本身。
2. `SharedOutputRef` 需要保留，用于跨 worker 稳定定位 fresh output
   descriptor；但 writer 依赖不一定必须通过 symbol writer 查询。
3. `SharedOutputRef` 可以先解析为 `TensorDesc`，再走 ordinary region
   TensorMap。只要 descriptor 和较早 writer 已有序发布，正确性可以成立。
4. 当前 symbol writer 是 fresh output 的专用索引：以 `last_writer`、
   writer history 和有序 CAS，换取正常 INPUT 的 O(1) writer 查询，并减少
   region entry。
5. region 方案可统一 private/shared 语义并改善物理 alias 表达，但会增加
   descriptor 解析、region 扫描、payload DCCI、seq/tail 原子及容量压力。
6. 两种方案谁更快不能由结构直接推断，需要在相同依赖图和构建配置下 A/B。

## 2. 当前 private/shared 依赖模型

### 2.1 三类状态不是同一件事

当前 `SharedTensorMapSidecar` 同时容纳以下状态：

| 状态 | key | 作用 |
| --- | --- | --- |
| shared output table | `(origin_task, slot)` | 保存 fresh output `TensorDesc` |
| output publication | `(origin_task, slot)` | 表示 descriptor 已完整可读 |
| symbol writer | `(origin_task, slot)` | 保存逻辑 tensor 的 latest writer |
| writer history | `(writer_task, symbol)` | 将 latest 回退到过去版本 |
| ordinary region map | `(buffer, lo, hi)` | 查找重叠物理区间的 producer |

`SharedOutputRef` 只包含 `(producer_task_id, output_slot)`，没有地址、offset、
shape 或 buffer size，因此不能不经解析直接调用 region lookup。

相关定义为 `common/pa_frontend.h:53-74` 中的引用类型，以及
`common/pa_model.h:983-1068` 中的 region、output、history 和 sidecar。

### 2.2 private：按物理区间维护 writer 历史

private 输出句柄是本 worker 可直接解引用的 `TensorDesc *`：

```cpp
using PaOutputHandle = PA_GM TensorDesc *;
```

每个 worker 按本核 Submit 时间线更新自己的 TensorMap：

```text
task N 读取/修改 X
  -> 吸收 X.owner_task_id
  -> 按 X 的物理区间查询本核 TensorMap
  -> 得到最近重叠 writer，加入 fanin

task N 修改 X
  -> 插入 (X region, producer=N)
```

private 同样维护 writer 历史，只是历史已存在于每核 ordinary TensorMap，
不需要单独的 symbol writer。task N 查询时，本核尚未登记 task > N 的
writer，因此不需要从 future latest 向后回退。

相关实现为 `common/pa_frontend.h:450` 的 private handle，以及
`common/pa_frontend.h:1227-1382` 的 lookup、fanin 和 register。

### 2.3 shared：按引用类型选择 writer 索引

shared 输出句柄为：

```cpp
using PaOutputHandle = FdwicOutputRef;
```

当前 fanin 分流如下：

```text
SharedOutputRef
  -> 检查 descriptor 已发布
  -> ResolveSharedSymbolWriterBefore()
  -> 得到 max(symbol writer < current_task)
  -> 加入 fanin

GmTensor / LocalTensor
  -> 吸收 descriptor.owner_task_id
  -> SharedLookupTensor()
  -> 得到 max(overlap producer < current_task)
  -> 加入 fanin
```

当前 ordered Submit 先发布本 task writer metadata，再收集本 task fanin：

```text
Materialize/publish fresh outputs
  -> 准备 writer delta
  -> 等待 task[N-1].deps_prepared
  -> 发布 task N 的 symbol/ordinary writer metadata
  -> 发布 task[N].deps_prepared
  -> 收集 task N fanin
  -> Build
```

因此本 task 是 INOUT writer 时，`last_writer` 已经等于本 task，必须沿
history 回退到严格小于 N 的前驱。symbol 路径并非所有引用都只有一次 load。

writer prepare/commit 和 fanin 顺序见
`common/pa_shared_submit_path.h:43-254,571-745`。symbol 回退、两种 fanin
分支和 prepared commit 见
`common/pa_scheduler_core.h:1199-1260,1367-1516,1989-2079`。

## 3. 统一走 region TensorMap 的可行方案

候选流程如下：

```text
SharedOutputRef S
  -> 确认 shared_outputs[S].published
  -> invalidate/copy shared_outputs[S].tensors
  -> 得到 resolved TensorDesc
  -> 将 resolved.owner_task_id 加入 fanin
  -> SharedLookupTensor(resolved, current_task)
  -> 将最近重叠 writer 加入 fanin
  -> 若为 Inout/OutputExisting：
       准备 (resolved region, producer=current_task)
  -> 在 ordered commit 中发布 region entry
```

这与 private 的业务语义一致：

```text
owner_task_id 表示初始 producer；
region map 表示后续 Inout/OutputExisting writer。
```

采用该方案仍需保留：

- `SharedOutputRef`：跨 worker 传递稳定 handle；
- `SharedOutputCell::published`：descriptor 发布闸门；
- `SharedOutputCell::tensors`：共享 descriptor。

若 region map 成为唯一 writer 权威，理论上可退出主路径：

- `SharedOutputCell::last_writer`；
- `writer_history`；
- `ResolveSharedSymbolWriterBefore()`；
- `CommitPreparedSymbolSharedWriterIntentSet()`。

正式删除前仍需审计其他 runtime、测试和 ABI 消费者。

### 3.1 必须补齐的协议

1. **串行区外解析 descriptor。** 当前 prepare 可提前生成 symbol key，但不能
   直接生成 region。若把 descriptor 等待和 DCCI 放进 ordered Register，
   会扩大所有 task 共享的串行区。
2. **descriptor 只解析一次。** Build 当前还会 invalidate/copy descriptor。
   候选实现应复用 fanin 阶段结果，避免两次读取；同时检查本地状态增长和
   CCEC spill。
3. **较早 writer 完整发布。** task N 的 region entry 必须先于 N+1 lookup
   可见。当前 `deps_prepared` 链可以继续承担：

   ```text
   payload DCCI -> seq/tail -> task[N].deps_prepared
   ```

   仅过滤 `producer < current_task` 无法补回尚未发布的较早 writer。
4. **证明容量。** 当前 ordinary ring 固定 `reclaim_upto=-1`。增加 symbol
   writer entry 后，必须检查 B256/B512 的最坏 bucket occupancy，不能只看
   平均 entry 数。
5. **只保留一个 writer 权威。** 若 symbol 和 region 同时维护，必须定义
   哪个是权威及不一致时如何恢复；否则不建议长期保留双索引。

## 4. 方案权衡与当前工作量

### 4.1 语义和成本对照

| 维度 | symbol writer | ordinary region map |
| --- | --- | --- |
| key | 逻辑 `(origin, slot)` | 物理 `(buffer, lo, hi)` |
| future writer | history 回退 | producer `< N` 过滤 |
| 地址 alias | 不自动发现 | 可发现重叠区间 |
| 不重叠 view | 同 slot 保守串行 | 可按 byte range 区分 |
| reader 成本 | 快路 O(1) | hash + bucket 扫描 |
| writer 成本 | history flush + CAS | payload DCCI + seq/tail |
| 容量压力 | task-indexed history | bucket ring 与回收 |

symbol writer 当前每个 writer task 先连续写一份 history，执行一次
`FlushRegion(history)`，再对每个 symbol 执行 `last_writer` CAS。

ordinary writer 当前每条 entry 需要容量/seq 检查、payload invalidate、
64 B payload flush、seq 发布和 tail 更新。对应实现见
`common/pa_scheduler_core.h:2044-2078` 和
`common/pa_shared_tensormap.h:716-763`。

region 按物理范围表达 alias/view 更自然，但当前 lookup 只返回一个最大
producer。若 reader 同时覆盖两个没有传递依赖的 writer 区间，是否需要两个
fanin，仍需业务规则和定向测试确认。

### 4.2 当前 PA Case1 的操作量

当前 host oracle 固定检查：

```text
ordinary region inserts       = 0
ordinary map lookups          = 5 * group_count
symbol INPUT loads            = 5 * group_count
symbol INOUT writer commits   = 3 * group_count
fanin edges                   = 5 * group_count
```

来源为 `common/host_support.h:3917-3991`。

这说明当前 workload 同时存在 ordinary lookup 和 symbol lookup，只是 fresh
output 的后续 writer 不进入 region map。

统一 region 后，候选操作模型为：

- 每个 symbol INPUT 增加 descriptor-based region lookup；
- 每个 symbol INOUT 增加 predecessor region lookup；
- 每个 symbol INOUT 增加一条 region writer entry；
- symbol history flush、last_writer CAS 和 history 回退退出主路径。

这是源码计数推导，不是实测 DCCI 或性能结果。

## 5. 评审问题与验收口径

### 5.1 需要先确认的业务问题

1. 一个 `SharedOutputRef` 是否始终代表同一 backing region？
2. 不同 symbol 或普通 `GmTensor` 是否可能 alias 同一物理区域？
3. reader 横跨多个独立 writer 区间时，需要一个还是多个 fanin？
4. shared-ref view 是近期目标，还是继续只支持 plain ref？
5. 目标是缩短 ordered Register，还是降低完整 Submit/makespan？
6. 是否允许为了统一语义接入 ordinary ring reclaim？

### 5.2 正确性验收

候选实现至少应满足：

- 计算结果与当前 shared/private golden 一致；
- `dependency_signature` 和预期 fanin 拓扑一致；
- `fatal == 0`；
- reader 不依赖 self/future task；
- 较早 writer 未发布时，后续 reader 不能将其误判为外部输入；
- chained INOUT 的每代 writer 不丢失；
- B1/B256/B512 不发生未解释的 map capacity failure；
- CPU、CCEC 和 A5 使用相同逻辑依赖口径。

定向测试至少覆盖：

- origin -> INOUT -> INPUT；
- 多次 INOUT writer；
- future writer 先物理到达、较早 reader 后查询；
- 普通 tensor 与 `SharedOutputRef` 指向同一 backing region；
- 两个不重叠 view 和一个横跨两区的 reader。

### 5.3 性能验收

性能比较必须使用同一提交、task 数、编译配置和设备隔离，至少记录：

- 完整 Submit/makespan；
- ordered Register 的执行与等待时间；
- fanin lookup 和 descriptor resolve 时间；
- symbol history flush/CAS/回退次数；
- region lookup 扫描 entry 数；
- region payload DCCI、seq/tail 原子次数；
- map high-water 和最坏 bucket occupancy；
- trace-free 主性能与泳道诊断结果。

泳道图用于解释差异，不替代 trace-free 性能结论。

### 5.4 待选方案

**方案 A：保留 symbol writer。** 适合 whole-slot 语义足够、O(1) 查询收益
明确，且不希望 ordinary ring 承载所有 symbol writer 的情况。后续重点是
缩短 symbol ordered commit。

**方案 B：统一走 ordinary region TensorMap。** 适合希望统一
private/shared writer 语义、需要 alias/view 物理区间能力，并能在串行区外
解析和复用 descriptor、证明 map 容量的情况。

**方案 C：symbol 仅作 region cache。** 双索引一致性和故障处理最复杂，只应
在 A/B 实测都不能满足目标时考虑，不应作为默认折中。

评审未明确 writer 权威、alias/view 语义和性能目标前，不建议直接删除 symbol
writer，也不建议让 `SharedOutputRef` 同时更新两套 writer 状态。
