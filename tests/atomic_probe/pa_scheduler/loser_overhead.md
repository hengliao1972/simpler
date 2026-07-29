# Shared claim loser Submit 开销分析

本文回答：shared tensormap runtime 的 **claim loser** 在泳道里为何
**Submit 父区间经常 >1µs**，并结合样本

`Downloads/per_task_deps_prepared_register_writer_partition_r5ij_b256_merged_swimlane.json`

说明：**主因是固定公共外壳三段叠加**，以及其中有多少来自 **泳道图生成（插桩）本身**。

相关代码：`common/pa_scheduler_core.h` 的 `SubmitCallbackTask` /
`FinishSharedLoserSubmit` / `CloseSharedCallbackSubmit`。

---

## 1. 结论（先看这个）

1. Loser **已经不做** Materialize / Register / Fanin / TensorMap / 重构参；轻路径成立。
2. 但泳道契约仍要求每个逻辑 Submit 都有 **`EfDrain` + `Claim`**；另有
   `PrepareSharedTaskOutputs`（**只声明 `(task_id,slot)` 符号，不物化 descriptor、
   不分配 heap**）与收尾。
3. 在上述 R5ij b256 shared 泳道中，`claim.lost` 的 Submit 父区间
   **median ≈ 1.33 µs**，约 **89% >1µs**；去掉 EfDrain 内偶发 Kernel 后仍约 **71% >1µs**。
4. **多数时候没有任何单项单独 >1µs**，而是：

   ```text
   空 EfDrain（~0.2 µs 中位）
 + Claim（含/不含 ClaimMax，~0.7 µs 中位）
 + submit_tail_gap / residual（~0.4 µs 中位）
 ≈ 1.3 µs
   ```

5. **很大一部分测量值来自泳道插桩**（多次 `SYS_CNT` / `Ops::Now()`、
   `WriteTrace`、ClaimMax 的 atomic bracket）。业务上真正不可避免的主要是
   **一次 `atomicMax`（attempted loser）+ 极薄的符号/收尾**；不能把泳道上的
   1.3 µs 直接当成无观察热路径的净成本。
6. `PrepareSharedTaskOutputs` 与 `WriteTrace(Submit)` **语义独立**：前者是编排 ABI
   （详见 §2.1），后者是泳道/计数闭合。

---

## 2. Loser 实际执行路径

```text
BeginCallbackSubmit                 ← 在 Submit.start 之外
── Submit 父区间 ──
  EfDrain = DrainReady()            ← 每个逻辑 task 必做
  Claim   = role 路由 + 可能 atomicMax
  PrepareSharedTaskOutputs()        ← 声明 (task_id, slot) 符号
  loser 跳过 BuildCallbackSubmitArgs
  FinishSharedLoserSubmit()
    → CloseSharedCallbackSubmit()   ← ++submits + 写 Submit 记录
── Submit.end ──
```

Winner 才进入 Materialize / Register / Build。Loser 零 TensorMap 访问——
这与定向测试（如 loser 零 map access、guard-page）一致。

### 2.1 `PrepareSharedTaskOutputs()`：做什么、不做什么

它和后面的 `CloseSharedCallbackSubmit()` → `TraceTimestamp` / `WriteTrace(Submit)`
**只是顺序相邻，语义独立**。前者不是为了配合泳道取时/落盘。

#### 做什么

只在本核 `context.shared_result`（`SharedTaskOutputs`）里登记稳定符号句柄：

```text
(producer_task_id = 当前 task_id, output_slot = 0 .. count-1)
```

对应 `pa_frontend.h`：按 `FrontendTaskOutputCount(kind)` 循环 `AddOutputRef`。
源码注释写明：loser 也必须把同一组 `(producer, slot)` 交给本核后续
orchestration；**仅声明稳定符号**。

#### 为什么 loser 也要做

Shared 下不能把本核私有的 `TensorDesc*` 传给后续编排。Submit 的“返回值”是
`SharedOutputRef`，identity 只由 `(task_id, slot)` 决定，**与谁 claim 赢无关**。

每个 worker 都按同一顺序回放全部 task。本核这次 claim 输了，后面仍要继续构参，
例如：

```text
Alloc 输了 → 仍要用 Alloc 的 OutputRef(0/1/2) 挂后续 UP 输入
QK 输了   → 仍要用 QK 的 OutputRef 挂 SF 的 Input
```

若跳过 `PrepareSharedTaskOutputs`，本核 orchestration 没有这组符号，后续
`AppendSharedOutputRef(...)` 会断。Winner 负责真正 publish descriptor；loser
只需同一套**逻辑句柄**把图接下去。

`SubmitCallbackTask` 中的注释与此一致：fresh Output 返回值是 task/slot 符号，
不依赖哪个 worker 获胜；在 finish 前为所有 replay actor 建立同一句柄集。

#### 不做什么（常见误解）

| 误解 | 事实 |
| --- | --- |
| 生成 output tensor descriptor | **否**。不写 `TensorDesc`，不读 shared cell |
| 通过 heap ring 分配输出缓冲 | **否**。不碰 `heap_base` / `heap_size` / vend |
| 发布 `published` / 更新 `last_writer` | **否**。那是 winner 侧 `PublishSharedTaskOutputs` 等 |
| 为了 `WriteTrace(Submit)` 才调用 | **否**。与泳道闭合无关，是编排 ABI |

真正做 descriptor + heap 的是 **winner** 路径，例如：

- `MaterializeTask(...)`（按 heap 分配物理输出）；
- `PublishSharedTaskOutputs(...)`（把 descriptor 写入
  `SharedOutputCell.tensors[]` 并发布）。

Loser 只有符号；要用地址/shape 时，后续 winner 再等 `published` 并从 shared
cell 读取 descriptor。

#### 和 `CloseSharedCallbackSubmit` 的分工

| 调用 | 目的 |
| --- | --- |
| `PrepareSharedTaskOutputs` | 填 `shared_result`，给本核后续编排用 |
| `CloseSharedCallbackSubmit` → `TraceTimestamp` + `WriteTrace(Submit)` | 计 Submit 次数、闭合泳道父区间 |

`PrepareSharedTaskOutputs` 本身只是几次 `AddOutputRef`，成本很小；tail 里
~0.4 µs 主要来自写 Claim 记录、取 `Submit.end` 与薄校验，而不是“物化输出”。

---

## 3. 泳道样本测量（R5ij b256）

| 项 | 值 |
| --- | ---: |
| 模式 | shared / real-compute / schema-v5 |
| 时间单位 | µs（QK≈44 µs 可交叉验证） |
| Submit 总数 | 122,880（96×1280） |
| `claim.won` / `lost` / `not_attempted` | 1,280 / 72,448 / 49,152 |
| 完整 Submit 墙钟（首末） | ≈ 3.406 ms |

### 3.1 Submit 父区间按 claim 结果

| 口径 | n | median | mean | p95 | >1µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `claim.lost` | 72,448 | **1.327** | 1.785 | 3.256 | **88.9%** |
| `not_attempted` | 49,152 | 0.909 | 1.684 | 3.042 | 45.4% |
| 上述且 EfDrain **无 Kernel** | 120,592 | **1.295** | 1.452 | 3.061 | **71.1%** |
| EfDrain **含 Kernel** 的 loser | 1,008 | **44.0** | 36.7 | 59.5 | （drain 执行） |

仅 **0.8%** loser Submit 被前序 winner 的 Kernel 嵌进 EfDrain；去掉后仍有约七成 >1µs。
因此日常看到的 “loser >1µs” **主要是空外壳**，不是偶发算力污染。

空 loser 时长直方图（无 Kernel）：

| 区间 (µs) | 占比 |
| --- | ---: |
| [0, 1) | ~29% |
| **[1.0, 1.5)** | **~35%**（主体） |
| [1.5, 3) | ~31% |
| ≥3 | ~5.6% |

### 3.2 空 loser 且 Submit>1µs 的三段构成（n=85,727）

| 段 | median | mean | 约占 Submit |
| --- | ---: | ---: | ---: |
| Claim | **0.74** | 0.78 | **~45%** |
| EfDrain（空） | 0.21 | 0.52 | ~30% |
| residual（≈`submit_tail_gap`） | **0.43** | 0.44 | ~25% |

“谁单独就能 >1µs”：

| 模式 | 占比 |
| --- | ---: |
| **三项都 ≤1µs，但加总 >1µs** | **~65%** |
| 仅 EfDrain >1 | ~18% |
| 仅 Claim >1 | ~14% |
| residual 单独 >1 | ≈0 |

按“哪一段最大”：Claim 主导 ~73%，EfDrain ~20%，residual ~7%。

`claim.lost` 中 `claim_max` 嵌在 Claim 内，约占 Claim 的一半
（ClaimMax median ≈0.30 µs / Claim ≈0.68 µs）。`not_attempted` **没有**
ClaimMax，Submit median 仍约 0.91 µs——说明 **>1µs 不是单靠 FetchMax**。

---

## 4. 为什么说「主因是三段叠加」

### 4.1 字面含义

对空 loser，典型中位数大约是：

```text
0.20 (EfDrain) + 0.68 (Claim) + 0.43 (tail) ≈ 1.31 µs
```

没有哪一段“坏到 1µs 以上才拖垮整体”；是 **三条固定必经路径的耗时相加**
越过 1µs 线。65% 的 >1µs 样本属于这种「分项都不大、总和过线」的模式。

### 4.2 结构原因：loser 仍走完整 Submit 外壳

为了 SPMD 回放与泳道闭合，每个 actor 对每个逻辑 task 仍调用同一套
`SubmitCallbackTask` 前缀。参考路径更接近 `rt_submit_loser` 只回符号；
当前实现为了：

- 每核 `submits == 5*batches`；
- 每个 Submit 都有 `EfDrain`/`Claim` child（`SHARED_REQUIRED_ON_EVERY_SUBMIT`）；
- loser 也要 `PrepareSharedTaskOutputs` 以便后续 orchestration 持有同一套符号；

而把 **EfDrain + Claim + 收尾** 留在了 loser 热路径上。这三段与 TensorMap
无关，但是 **每次逻辑 task 都付一次**。

### 4.3 计时边界如何把成本拆进这三段（关键）

源码顺序（泳道构建，`Profile=true`）大致是：

```text
submit_begin / efdrain_begin = TraceTimestamp()     // 或与 task0 共用
DrainReady()                                        // 空则几乎立刻返回
efdrain_end = TraceTimestamp()
WriteTrace(EfDrain)                                 // ← 落在 Claim 时间窗内！
claim_begin = efdrain_end
Claim()  → 可能 TraceAtomicFetchMax（内部再 Now×2 + 写 Atomic 记录）
claim_end = TraceTimestamp()
WriteTrace(Claim)                                   // ← 落在 submit_tail_gap！
PrepareSharedTaskOutputs()                          // ← 只填符号，非 descriptor/heap；非泳道专用
CloseSharedCallbackSubmit():
  submit_end = TraceTimestamp()
  WriteTrace(Submit)
```

因此泳道上看到的三段 **并不等于** 三段纯业务函数的净耗时：

| 泳道 span | 实际装了什么 |
| --- | --- |
| **EfDrain** | `DrainReady` 本体 + **结束处一次 `Now()`**（`WriteTrace(EfDrain)` 还不在里面） |
| **Claim** | `WriteTrace(EfDrain)` + role 路由 +（lost 时）ClaimMax bracket + Claim 收尾 + **结束处一次 `Now()`** |
| **submit_tail_gap / residual** | `WriteTrace(Claim)` + `PrepareSharedTaskOutputs`（薄：仅符号）+ loser 校验/收尾 + **`Now()` 取 Submit.end** |

这解释了：

- 为什么空 EfDrain 中位还有 ~0.2 µs（主要是边界取时，而不是扫 slot）；
- 为什么 Claim 明显大于内嵌的 ClaimMax（另一半是 **上一段的 WriteTrace + 外壳**）；
- 为什么 residual 稳定在 ~0.4 µs 且与 `submit_tail_gap` 几乎同分布（主要是 **本段 WriteTrace + 取时 + 薄收尾**）。

**叠加过 1µs 的直接算术原因**：每段都带着「边界时钟 / 落盘」的固定税，三段各收一次税，总和自然落在 1–1.5 µs 的桶里。

---

## 5. 是否因为泳道图生成本身的开销？

### 5.1 短答

**是，占大头；但不等于“全部都是假的”。**

- **泳道插桩**（多次 `Ops::Now()` / PollBatch 边界、`WriteTrace`、ClaimMax 的
  begin/end bracket）把空 loser 的测量值系统性抬到 ~1.3 µs。
- **真实业务仍保留**：attempted loser 的一次 **`atomicMax`（ClaimMax median ~0.3 µs）**、
  空 `DrainReady` 的函数调用、符号声明与计数闭合。这些在无泳道构建里也会存在，
  但通常远小于泳道上看到的整段 Submit 父区间。

### 5.2 插桩如何“制造”三段税

一次空 `claim.lost` Submit 在泳道构建下至少涉及：

| 动作 | 次数（量级） | 计入哪段 |
| --- | ---: | --- |
| `TraceTimestamp` / `Now`（EfDrain 结束） | 1 | EfDrain |
| `WriteTrace(EfDrain)` | 1 | **Claim** |
| ClaimMax：`Now` 起、`NowAfterAtomic` 止 + 写 Atomic 记录 | 2+写 | Claim（且显示为 `claim_max` child） |
| `TraceTimestamp`（Claim 结束） | 1 | Claim |
| `WriteTrace(Claim)` | 1 | **tail** |
| `TraceTimestamp`（Submit 结束） | 1 | tail |
| `WriteTrace(Submit)` | 1 | Submit.end 之后（常进下一 gap / OrchestrationTail） |

`not_attempted` 没有 ClaimMax bracket，Submit 更短（median 0.91 µs），但仍有
EfDrain/Claim/Submit 的边界取时与两次阶段 `WriteTrace`——与“插桩税为主、
atomic 为辅”一致。

### 5.3 与无观察构建的旁证

- 文档约定：开启泳道后 **不应**把 bracket / Submit span 与未插桩基线直接相减当绝对占比
  （插桩改变布局、到达顺序与竞争）。
- 历史 **S4.12a** 裁掉 loser 的空 finish 外壳后，路径与观察都正确，但
  **perf-clock 墙钟中性并已撤销**——说明当时墙钟并不卡在“再少一层可裁的 finish 壳”，
  也侧面说明 **泳道上显眼的 loser 外壳 ≠ 无观察主瓶颈的同等放大**。
- `PA_BUILD_TRACE_FREE` / perf-clock / submit-pmu 路径会去掉额外 `SYS_CNT` 与
  span 写入；要用那些构建重新量空 loser，才能得到接近热路径的净成本。

### 5.4 如何区分「真业务」和「泳道税」（建议口径）

| 问题 | 建议看什么 |
| --- | --- |
| 泳道上为何 >1µs | 本文：三段叠加 + 插桩边界错位计入 |
| 无观察时 loser 还要多久 | 同业务 **perf-clock / TRACE_FREE** A/B，不要用本 merged JSON 的 1.3 µs 当净成本 |
| atomic 本身 | 看 `claim_max` child（本样本 lost median ~0.30 µs），仍含 bracket 取时 |
| 是否被 kernel 污染 | 看 EfDrain 窗口内是否有 `QK/SF/PV/UP`（本样本仅 0.8%） |

粗分本样本空 `claim.lost`（中位量级）：

```text
ClaimMax 硬件+bracket     ~0.30 µs   ← 部分真、部分观察
Claim 内其余（含写 EfDrain 记录等）~0.38 µs   ← 多为泳道落盘/外壳
空 EfDrain span           ~0.20 µs   ← 多为结束取时
tail / residual           ~0.43 µs   ← 多为写 Claim 记录 + 取 Submit.end + 薄收尾
                                      （含 PrepareSharedTaskOutputs 符号声明，非 heap 物化）
────────────────────────────────────
Submit 父区间             ~1.3 µs
```

因此：**“三项叠过 1µs”的现象，在这份泳道图上 largely 是泳道生成方式导致的计量形态**；
若关掉插桩，数字会明显下降，但 **ClaimMax + 仍存在的 Submit 外壳**不会降到零。

---

## 6. 和「理想 loser 轻路径」的差距

| | 参考意图 | 当前 standalone shared |
| --- | --- | --- |
| Claim 后 | `rt_submit_loser` 只回符号 | 仍跑完整 Submit 父区间 |
| EfDrain | 可对空/wrong-role 更早跳过（实验项） | 每 task 必进 |
| 观察 | 无 | 每段 Now + WriteTrace，且 WriteTrace 计入下一段 |

优化若以「泳道上 loser <1µs」为目标，会主要在打插桩税，墙钟未必动。
若以墙钟为目标，应：

1. 用 exclusive JSON **剔除 EfDrain 含 Kernel 的 loser**；
2. 在 **TRACE_FREE / perf-clock** 上重测空 loser；
3. 再考虑 wrong-role / 空 slot 更早跳过 EfDrain（类 S4.12b），而不是继续抠
   `FinishSharedLoserSubmit` 里已很薄的符号返回。

---

## 7. 样本与入口索引

| 项 | 位置 |
| --- | --- |
| 本文依据泳道 | `per_task_deps_prepared_register_writer_partition_r5ij_b256_merged_swimlane.json` |
| Submit / Claim / loser 收尾 | `common/pa_scheduler_core.h` |
| `PrepareSharedTaskOutputs` / `SharedTaskOutputs` | `common/pa_frontend.h`（§2.1） |
| Winner 物化 / 发布 descriptor | `MaterializeTask`、`PublishSharedTaskOutputs`（`pa_shared_submit_path.h` 等） |
| TraceTimestamp / ClaimMax bracket | `common/pa_scheduler_core.h`, `common/pa_trace.h` |
| §12 vs 实现差异 | `shared_tensormap_imp_analysis.md` |
| 泳道残余语义 | `swimlane_opt_anal.md`（`submit_tail_gap` 等） |
