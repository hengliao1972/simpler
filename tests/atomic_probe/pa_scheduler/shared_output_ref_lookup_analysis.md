# SharedOutputRef 依赖查询方案分析

## 1. 问题、结论与边界

本文回答：

1. private 为什么用 `TensorDesc + TensorMap`，当前 shared 为什么对
   `SharedOutputRef` 使用 `last_writer + writer_history`？
2. `SharedOutputRef` 能否先解析成 `TensorDesc`，再像普通
   `GmTensor/LocalTensor` 一样查询 shared ordinary TensorMap？

结论：

- shared 仍然生成正常 `fanin`。symbol 是 writer 查询 key，不是 fanin。
- `SharedOutputRef` 用于跨 worker 定位 descriptor；writer 使用 symbol
  还是 region 索引是另一项设计，没有必然绑定。
- 解析 `SharedOutputRef` 后统一走 region TensorMap，在协议上可行。
- “使用 region map”和“复刻 private fanin”不是同一个决定。private 把
  owner 与 region lookup 结果都加入 fanin；当前 shared symbol 只加入
  reader 之前的 latest writer。
- 传递依赖闭合时，两种 fanin 可能得到相同计算结果，但边数量、等待对象和
  故障敏感性不同，必须先确定业务口径。

核对基线为远端 `fdwic-swimlane-deps@bbd18779`；实现代码来自其父提交
`c3aaf99f`。工作区其他未提交改动不属于本文依据。

### 1.1 当前状态分别承载什么

| 状态 | key | 回答的问题 |
| --- | --- | --- |
| shared output table | `(origin,slot)` | tensor 在哪里、什么形状 |
| `published` | `(origin,slot)` | descriptor 是否完整可读 |
| symbol writer | `(origin,slot)` | 谁最后修改了这个逻辑 tensor |
| ordinary TensorMap | `(buffer,lo,hi)` | 谁写过重叠物理区域 |

`SharedOutputRef` 只有 `(producer_task_id, output_slot)`，没有地址、offset、
shape 和 size。它不能直接传给 `SharedLookupTensor()`；走 region 前必须先
取得共享 `TensorDesc`。

## 2. 当前代码的三条真实路径

### 2.1 private：owner 与 region writer 都加入 fanin

private 输出句柄是本 worker 可解引用的 `TensorDesc *`。对 Input/Inout，
`CollectFanin()` 执行：

```text
AddFanin(tensor.owner_task_id)
AddFanin(LookupTensor(private_map, tensor))
```

`AddFanin()` 只去除相同 task id，不会因为 map writer 更新而删除旧 owner：

```text
private fanin = unique(owner, latest_overlap_writer)
```

Inout/OutputExisting 随后由 `RegisterOutputs()` 把当前 task 的 region
登记进本 worker 的 map。

### 2.2 shared symbol：只加入 latest symbol writer

`SharedOutputRef` 进入 `CollectSharedFanin()` 的 symbol 分支：

```text
检查 published
  -> 读取 last_writer
  -> latest >= reader 时沿 writer_history 回退
  -> 得到 max(writer < reader)
  -> 只把该 writer 加入 fanin
```

该分支不另外加入 descriptor 的 `owner_task_id`。Inout/OutputExisting 在
ordered Register 中执行：

```text
history[current] = previous_writer
  -> flush history
  -> CAS last_writer: previous -> current
```

### 2.3 shared ordinary：仍按物理 region 查询

普通 `GmTensor/LocalTensor` 当前执行：

```text
AddFanin(owner_task_id)
AddFanin(SharedLookupTensor(shared_map, tensor))
```

writer 在 ordered Register 中追加：

```text
(buffer_addr, lo, hi, producer=current_task)
```

因此当前 shared 是两套 writer 索引：

```text
SharedOutputRef       -> symbol writer
GmTensor/LocalTensor  -> ordinary region TensorMap
```

### 2.4 tag 行为对比

以下是当前代码事实：

| tag | private Gm/Local | shared symbol | shared ordinary |
| --- | --- | --- | --- |
| Input | owner + lookup | symbol latest | owner + lookup |
| Inout | owner + lookup；register | previous symbol；commit | owner + lookup；append |
| OutputExisting | owner；register | previous symbol；commit | owner + lookup；append |
| Output | Materialize owner | publish descriptor/symbol | 不做 reader lookup |

所以“改得与 private 一样”还必须明确：

- `OutputExisting` 是否读取旧 region writer；
- resolved descriptor 的 `manual_dep` 是否关闭自动依赖；
- fanin 是 `owner + latest`，还是只保留 latest。

代码入口：

- private：`pa_frontend.h::CollectFanin/RegisterOutputs`
- symbol：`pa_scheduler_core.h::ResolveSharedSymbolWriterBefore`
- shared collect：`pa_scheduler_core.h::CollectSharedFanin`
- ordered commit：
  `pa_scheduler_core.h::CommitPreparedSymbolSharedWriterIntentSet`

## 3. 用具体 task 对比

### 3.1 示例一：Output 经过一次 Inout

设 tensor `X`：

```text
symbol     = (10,0)
descriptor = { buffer=A, range=[0,4096), owner_task_id=10 }
```

逻辑任务：

| task | 操作 |
| --- | --- |
| T10 | `Output X` |
| T12 | `Inout X` |
| T20 | `Input X` |

#### T12 的行为

| 实现 | 查询过程 | T12 fanin | writer 发布 |
| --- | --- | --- | --- |
| private | owner=10；map=NONE | `[10]` | private map 追加 producer 12 |
| shared symbol | commit 后从 history 回退 | `[10]` | history `12->10`，latest=12 |
| shared region | owner=10；map 中 self 被 `<12` 过滤 | `[10]` | shared map 追加 producer 12 |

当前 ordered shared 在 fanin 前发布本 task writer。symbol 因而从
`last_writer=12` 回退到 10；region 则通过 `producer < current_task`
过滤自己的 entry。

#### T20 的行为

| 实现 | owner | latest | T20 fanin |
| --- | ---: | ---: | --- |
| private | 10 | 12 | `[10,12]` |
| 当前 shared symbol | 不单独加入 | 12 | `[12]` |
| shared region/private 口径 | 10 | 12 | `[10,12]` |
| shared region/latest-only | fallback 用 | 12 | `[12]` |

private 的 `[10,12]` 是当前源码的真实结果，不是笔误。因为 T12 已依赖 T10，
等待 T10 通常是传递冗余，但 private 仍显式保留这条边。

当前 shared 使用：

```text
T20 waits T12
T12 waits T10
```

它依赖 writer 链完整，以传递关系覆盖 T10。

这个例子证明有两个独立选择：

1. writer 历史存入 symbol 还是 region；
2. fanin 采用 private 的 `unique(owner,latest)`，还是 latest-only。

### 3.2 示例二：future writer 先物理发布

逻辑顺序：

```text
T10 Output X -> T12 Inout X -> T20 Input X -> T25 Inout X
```

metadata commit 与较早 task 的 fanin/Build 可以流水，因此可能出现：

```text
T12 metadata published
T25 metadata published
T20 才开始 lookup
```

T20 必须得到 12，不能依赖未来 T25：

| 实现 | 如何排除 T25 |
| --- | --- |
| private | 本核处理 T20 时尚未在 private map 插入 T25 |
| shared symbol | `last_writer=25`，沿 `25->12` history 回退 |
| shared region | 扫描 12/25，但只接受 `producer < 20` |

future 过滤两种 shared 结构都能实现。

### 3.3 反例：较早 writer 尚未发布

```text
T12 逻辑上会 Inout X，但 metadata 尚未发布
T20 已经 lookup X
```

symbol 和 region 都只能看到 10，无法区分：

```text
T12 不存在
T12 存在但还没发布
```

所以 task-id 过滤只能排除 future，不能补回 missing past。当前
`task[N-1].deps_prepared` 链必须保证：

```text
T20 lookup 前，T0..T19 的 writer metadata 全部完成发布
```

无论选 symbol 还是 region，这条合同都不能删除。

### 3.4 alias 与 view 对比

| 场景 | symbol writer | region map | 待确认业务规则 |
| --- | --- | --- | --- |
| 不同 symbol 指向同一地址 | 两条链，不能互相发现 | 可按重叠发现 | 是否禁止跨 symbol alias |
| 同 symbol 的不重叠 view | whole-slot 保守串行 | 可区分 byte range | 是否需要 view 并行 |
| reader 横跨两个独立 writer | 若同链则传递等待 | 当前只返回最大 producer | 是否必须返回多个 fanin |

第三种场景示例：

```text
T12 writes X[0,4096)
T14 writes X[4096,8192)      // 与 T12 没有传递依赖
T20 reads  X[0,8192)
```

当前 `SharedLookupTensor()` 只返回最大 producer 14。若业务允许这种图，仅等
T14 不能证明 T12 已完成。采用 region 不自动等于 view 语义完整。

当前生产路径只接受 plain `SharedOutputRef`，view ABI 尚未接入；上述内容是
选型边界，不是当前 PA Case1 已验证的收益。

## 4. DCCI 与操作成本对比

### 4.1 当前 ordinary lookup

控制字不是普通 GM load：

```text
head/tail/seq -> Ops::Load -> atomicAdd(address, 0)
```

payload 读取：

```text
atomic load seq
  -> DCCI invalidate payload
  -> copy buffer/lo/hi/producer
  -> atomic load seq again
```

writer：

```text
write payload
  -> DCCI CACHELINE_OUT + DSB
  -> publish seq
  -> publish tail
```

因此每扫描一个有效 slot，reader 都可能付出一次 payload DCCI。

### 4.2 当前 symbol lookup

- `published`、`last_writer` 使用 `atomicAdd(0)`；
- `last_writer < reader` 时不读取 history；
- latest 指向 self/future 时，先 invalidate history 再回退；
- Build 复制 shared descriptor 前另行 invalidate descriptor；
- writer 先 flush descriptor/history，再发布原子控制字。

### 4.3 SharedOutputRef 改走 region 后

不能直接把共享 descriptor 引用传给 lookup。正确顺序至少是：

```text
atomic observe published
  -> DCCI invalidate shared TensorDesc
  -> copy descriptor to winner-local snapshot
  -> build region query
  -> ordinary map lookup
```

当前 Build 已经 invalidate/copy descriptor。若 fanin 再做一次，会重复 DCCI；
候选实现应考虑解析一次并由 Build 复用，同时检查 local state 和 CCEC spill。

### 4.4 总体对比

| 维度 | private | 当前 shared symbol | 候选 shared region |
| --- | --- | --- | --- |
| handle | 本核 `TensorDesc*` | `SharedOutputRef` | `SharedOutputRef` |
| writer key | physical region | logical symbol | physical region |
| 示例 T20 fanin | `[10,12]` | `[12]` | 取决于 fanin 口径 |
| future 过滤 | 本核时间线 | history 回退 | `producer < N` |
| alias | 可发现 | 不跨 symbol | 可发现 |
| reader DCCI | 无跨核 payload | history/descriptor | descriptor + map slot |
| writer 成本 | 本核普通写 | history flush + CAS | entry flush + seq/tail |
| 串行区 | 无共享 map | symbol commit | region append |
| 容量 | 每核 map | task-indexed history | shared bucket ring |

### 4.5 当前 PA Case1 的源码计数

```text
ordinary region inserts       = 0
ordinary map lookups          = 5 * group_count
symbol INPUT loads            = 5 * group_count
symbol INOUT writer commits   = 3 * group_count
fanin edges                   = 5 * group_count
```

全部改为 region 后，初步模型为：

- 每个 group 的 5 次 symbol INPUT 增加 descriptor resolve + region lookup；
- 每个 group 的 3 次 symbol writer 增加 predecessor lookup + region append；
- 移除 history flush、`last_writer` CAS 和 history 回退；
- private fanin 口径可能增加 edge 数与 dependency signature；
- shared ring 从零插入变为每个 group 至少 3 条，需重做最坏 bucket
  容量证明。

这是源码操作量推导，不是性能结论。symbol 通常是一次 history flush 加多个
CAS；region 是每 entry DCCI、seq/tail 原子和后续扫描，必须在 A5 上实测。

## 5. 评审选项与验收

### 5.1 方案

| 方案 | writer 权威 | fanin 口径 | 主要特点 |
| --- | --- | --- | --- |
| A 保持当前 | symbol | latest-only | O(1) 快路，whole-slot，保留 history |
| B region/private | region | owner + latest | 与 private 一致，可发现 alias，边更多 |
| C region/latest | region | latest，否则 owner | 去掉 history，但不声称复刻 private |
| D 双索引 | 两者 | 需仲裁 | 双写与故障恢复复杂，不建议默认采用 |

### 5.2 必须先回答

1. fanin 必须与 private 列表完全一致，还是传递依赖等价即可？
2. 不同 `SharedOutputRef` 是否可能 alias 同一 backing region？
3. `OutputExisting` 是否读取旧 writer？
4. resolved descriptor 的 `manual_dep` 如何解释？
5. 一个读取区间是否可能需要多个独立 producer？
6. 目标是缩短 ordered Register，还是完整 Submit/makespan？
7. ordinary ring 不回收时，B256/B512 的最坏 bucket 是否足够？

### 5.3 正确性与性能验收

正确性至少覆盖：

- `Output -> Inout -> Input` 的逐 task 精确 fanin；
- 多次 Inout、future writer、missing past；
- 跨 symbol alias、不重叠 view、横跨多 writer reader；
- B1/B256/B512 的结果、`fatal`、依赖签名和 map 容量。

性能在相同提交、task 数、编译配置和设备隔离下记录：

- trace-free 完整 Submit/makespan；
- ordered Register 执行与等待；
- descriptor resolve、fanin lookup 和 Build copy；
- history flush/CAS/invalidate；
- region DCCI、扫描 slot、seq/tail 原子及最坏 bucket occupancy；
- level-4 泳道用于解释，不替代 trace-free 主性能。

在 fanin、alias/view 和 `OutputExisting/manual_dep` 口径确认前，不应机械地
把 symbol helper 替换为 `SharedLookupTensor()`，也不应让两套索引同时成为
writer 权威。
