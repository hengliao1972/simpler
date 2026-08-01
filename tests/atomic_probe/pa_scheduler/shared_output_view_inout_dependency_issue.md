# PA SharedOutputRef view-INOUT 依赖缺口分析

> 日期：2026-07-31
> 范围：atomic-probe shared PA Submit，并对照 A5 FDWIC ordinary
> TensorMap。
> 状态：代码分析结论；不表示 view symbol 已经实现。

下文 PA 头文件默认位于
`tests/atomic_probe/pa_scheduler/same_core/common/`，A5 runtime 头文件默认位于
`src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/`。

## 1. 问题和直接结论

讨论下面三个逻辑 task：

| Task | 参数语义 |
| ---- | -------- |
| `N` | `OUTPUT` 生成完整 tensor `X` |
| `N+1` | 对 `X.view()` 的子区间做 `INOUT` |
| `N+2` | 将不带 view 的原始完整 `X` 作为 `INPUT` |

例如：

```text
N:   writes  X[0, 4096)
N+1: updates X[1024, 2048)
N+2: reads   X[0, 4096)
```

带 view 的是 `N+1`，不是 `N+2`。由于 `N+2` 的读取覆盖 `N+1`
的写入范围，正确依赖必须包含：

```text
N -> N+1 -> N+2
```

只有 `N -> N+2` 不够，否则 `N+1` 和 `N+2` 可在 `N` 完成后并行，
`N+2` 可能读到更新前的数据。

直接结论：

| 路径 | 结果 | 原因 |
| ---- | ---- | ---- |
| A5 ordinary TensorMap | 单-view 场景正确 | 按物理区间登记和查询 |
| 当前 PA `SharedOutputRef` | 功能未支持 | `N+1` 的 view ref 被拒绝 |
| 强行混用 symbol 与 region | 会漏依赖 | writer 和 reader 查询不同索引 |

当前代码是 **fail-closed**：它不会静默漏掉依赖后执行，而是在
`N+1` 的 Materialize 阶段失败。但它也没有实现：

```text
lookup(full X, reader=N+2) -> writer N+1
```

问题本质是：`X.view()` writer 与 plain `X` reader 是否进入同一个、
能够识别 alias 的 writer 索引。

## 2. ordinary TensorMap 为什么可以建立依赖

`Tensor::view()` 通过 `init_from_line1()` 继承 buffer 和 owner，只调整
offset、shape 等 view metadata（`src/common/task_interface/tensor.h`
第 379-402 行）。所以 view 仍携带：

```text
owner=N, buffer=X.buffer, range=[view_lo, view_hi)
```

production FDWIC 的处理顺序如下。

1. 非 `manual_dep` 的 `INOUT` / `OUTPUT_EXISTING` 被加入
   `register_mask`（`submit_core.h:423-439`）。
2. `N+1` 收集 fanin 时，先加入 descriptor owner `N`，再用 view
   的物理区间查询 TensorMap（`submit_core.h:568-624`）。
3. Register 将 view writer 登记为
   `(X.buffer, view_lo, view_hi, producer=N+1)`
   （`submit_core.h:685-703`）。
4. `N+2` 虽然不带 view，但会用完整 `X` 的 descriptor 查询
   `(X.buffer, full_lo, full_hi)`。

region lookup 选择满足以下条件的最大 producer：

```text
producer < reader_task
candidate.range overlaps query.range
```

实现位于 `shared_tensor_map.h:94-123`。因为 `N+1` 的 view 区间与
完整 `X` 重叠，所以单-view 场景中 `N+2` 查询得到 `N+1`。
这个结论不要求 `N+2` 也带 view。

## 3. 当前 PA 路径具体断在哪里

### 3.1 `INOUT` 不等于 `ordinary_count > 0`

`MaterializeTask()` 会把全部 `INOUT` / `OUTPUT_EXISTING` 参数加入
`register_mask`（`pa_frontend.h:1501-1504`），但
`PrepareSharedTaskWriterDelta()` 还会按引用种类分流：

```text
SharedOutputRef                  -> symbol_count++
GM/Local 且 manual_dep == false -> ordinary_count++
GM/Local 且 manual_dep == true  -> 两者都不增加
```

代码为 `pa_shared_submit_path.h:43-179`。因此：

> `ordinary_count > 0` 表示存在需要 region TensorMap 登记的
> GM/Local writer，不是“存在 INOUT 参数”的同义词。

正式 PA shape 反而要求 `delta.ordinary_count == 0`；UP 必须正好有
三个 symbol writer，其他 PA task 必须没有 writer
（`pa_shared_submit_path.h:206-238`）。

所以正式 PA task 若真的产生 `ordinary_count > 0`，当前逻辑不是继续
执行 ordinary TensorMap insert，而是 shape validation 失败并 fatal。

### 3.2 PA UP 如何保证后继读到更新值

PA UP 的参数形状是：

```text
3 个 fresh INPUT
3 个 plain SharedOutputRef accumulator INOUT
1 个外部 LocalTensor output_view INOUT
```

三个 accumulator 是 `accumulated_max`、`accumulated_sum` 和
`accumulated_output`（`pa_scheduler_core.h:3376-3394`）。
`output_view` 被显式设置为 `manual_dep=true`
（`pa_frontend.h:835-844`、`pa_scheduler_core.h:3395-3399`）。

所以 UP 的 writer delta 实际为：

```text
ordinary_count = 0
symbol_count   = 3
```

UP 不插入 ordinary region entry。它为三个 accumulator 发布三条
`writer_history` record，并把这组三个同步更新的 group latest
（Alloc cell 的 `last_writer[0]`）推进到当前 UP。查询专路把 accumulator
slot 0-2 映射到该 group latest，但仍用各自 symbol key 回退 history
（`pa_scheduler_core.h:1282-1305`）。

因此，后继 task 即使仍持有最初 Alloc 产生的 plain ref，也能以稳定的
`(Alloc task, output slot)` identity 解析到最新 UP writer。

这就是后继读取 `li_update` 一类累计结果的依赖来源：

```text
plain SharedOutputRef INOUT
  -> writer_history / last_writer
  -> 后继 plain ref 得到最新 UP
```

若 `last_writer` 已被 future writer 覆盖，reader 沿 immutable history
回退到 `max(writer < reader_task)`。key 和回退逻辑位于
`pa_scheduler_core.h:1227-1237,1260-1381`。

该证明有严格前提：writer 和 reader 必须使用同一个 symbol key。

### 3.3 泳道中的 wait 和 insert/lookup 并行

泳道事件：

```text
register.wait_predecessor_tensormap_insert#N
```

是转换器用 Register 起点到 metadata 起点合成的名称
（`swimlane_converter.py:1659-1666`）。设备侧对应：

```text
WaitForSharedTaskInsertTurn(...)
pa_shared_submit_path.h:729-750
```

它等待 `task(N-1).deps_prepared`，取得全局有序 metadata publish turn。
这是通用事件名，不代表当前 task 一定插入 ordinary TensorMap entry。

对 PA UP，这个区间实际发布三个 symbol 的 history/group
`last_writer`，再 handoff 本 task 的 insert turn：

```text
symbol metadata: pa_shared_submit_path.h:693-767
turn handoff:    pa_shared_submit_path.h:779-803
fanin lookup:    pa_shared_submit_path.h:936-955
```

handoff 后，`insert(N)` 与 `lookup/build(N-1)` 可以并行。同-key reader
即使读到 future `last_writer`，也能沿 history 回退；但 history 不能
跨越 symbol 与 region 两套索引去发现 alias。

### 3.4 本文反例的准确失败点

`FdwicOutputRef` 虽预留一维 view ABI，但注释明确说明当前只支持 plain
ref（`pa_frontend.h:62-74`）。

若 `N+1` 以 view `SharedOutputRef` 做 INOUT，`MaterializeTask()` 要求
`IsPlainSharedOutputRef()`，在 `pa_frontend.h:1519-1529` 返回失败；
`pa_shared_submit_path.h:627-641` 随后设置 fatal。

即使只绕过 Materialize，后续仍有两层 plain-only 限制：

- `SharedSymbolHistoryKey()`：`pa_scheduler_core.h:1227-1237`；
- `CollectSharedFanin()`：`pa_scheduler_core.h:1476-1484`。

若进一步强行把 `N+1` 的 view 放入 ordinary map，却让 `N+2` 的 plain
parent ref 继续走 symbol lookup，状态会分裂为：

```text
symbol history:  latest(parent X) = N
ordinary region: X[view_lo, view_hi) -> N+1
N+2 lookup:      只查 symbol history -> N
```

结果就是缺少 `N+1 -> N+2`。问题不是 insert/lookup 并行本身，而是
view writer 与 parent reader 被放进了两个互不连通的 writer 索引。

## 4. 修正方向和正确性门槛

方案 A 是将任意 view 写保守投影到 root parent symbol：

```text
N+1 view INOUT -> latest(parent X) = N+1
N+2 plain X    -> symbol lookup 得到 N+1
```

它改动较小并保持 O(1) latest，但会把子区间写当成完整 tensor 写，
使不重叠 view 也串行化；还要定义 nested view 如何还原稳定的 root
`(producer, slot)`。

方案 B 是让 region 成为统一 alias 权威：

1. 将 shared output/view 解析成稳定 descriptor；
2. `N+1` 将 view range 登记到 region map；
3. `N+2` 即使使用 plain parent ref，也查询完整 descriptor 的 region；
4. 合并 owner、symbol 和 region fanin。

该方案可保留不重叠 view 的并行性，但不能让 symbol 和 region 各维护
一半状态。更一般地，若 `A`、`B` 并行写 `X` 的两个不重叠区域，
随后 `C` 读取完整 `X`，则 `C` 必须同时依赖 `A` 和 `B`。当前 region
lookup 只返回最大重叠 producer；完整 view 语义最终需要 producer set。

后续若宣称支持本文场景，最小定向测试必须断言：

1. `N+1` 的 view INOUT 成功 Materialize，且 fanin 包含 `N`；
2. `N+1` writer metadata 在 turn handoff 前完成发布；
3. plain `X` 的 `fanin(N+2)` 包含 `N+1`；
4. `N+1.flag == 0` 时 `N+2` 不可执行；
5. future writer 已发布时，仍选择正确的 `< N+2` writer；
6. 不得用 `manual_dep` 或显式依赖绕过自动 hazard tracking。

当前代码在第 1 项 fail-closed。这是“view symbol 未支持”，不能把拒绝
非法输入的测试等同于依赖语义已经实现。

## 5. 最终判断

当前 PA UP 对三个完整 accumulator 的处理是正确的，因为它们始终是同一
组 plain symbol，UP 和后继 reader 共享一条 `last_writer/history` 链。

但这不证明通用 view alias 正确。对本文场景：

```text
N:   OUTPUT X
N+1: INOUT X.view()
N+2: INPUT  X
```

最终判断是：

```text
ordinary TensorMap 单-view路径：可以建立 N+1 -> N+2
当前 PA SharedOutputRef 路径：在 N+1 Materialize 时拒绝
绕过校验但拆分 symbol/region：会漏掉 N+1 -> N+2
```

在实现 parent-symbol 保守投影，或让所有 alias reader 统一查询 region
之前，不能认为当前 `last_writer + writer_history` 已替代支持 view
alias 的通用 TensorMap。
