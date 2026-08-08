# Shared TensorMap：cross-core 与 same-core 协议及性能差异分析

## 1. 分析目的

本文比较 standalone Shared TensorMap PA 调度器的两种实现：

- `same_core`：同一个 worker 负责 task 的 Build、保存本地执行包，并在后续 `EfDrain/FinalDrain` 中执行；
- `cross_core_ordinary`：Build owner 与 Execute owner 分离，Build owner 把执行包发布到 task-indexed GM cell，另一 Scalar 取得并执行。

分析不以泳道图代替架构推导。本文采用三层证据：

1. **协议不变量**：先判断两种实现必须保持什么、各自新增了什么同步边；
2. **源码事实**：核对 TensorMap、SharedOutput、执行包和完成发布的真实顺序；
3. **B256 泳道数据**：验证调用数量、DCCI 范围和工作落点是否符合理论预期。

因此，文中的“为什么会变快/变慢”首先来自协议和工作量模型；泳道只负责确认这些工作确实发生，以及给出本次样本中的量级。

## 2. 样本、口径与结论摘要

### 2.1 对比样本

| 实现 | 样本 | 关键属性 |
| --- | --- | --- |
| same-core | [`test_record/2026-8-1/shared_b256_best_1659167us_swimlane.json`](test_record/2026-8-1/shared_b256_best_1659167us_swimlane.json) | B256，96 Scalar，Shared TensorMap，全员 replay，`two-16` 尾同步 |
| cross-core | [`test_record/2026-8-4/cross_b256_no_publish_reload_1p442ms.json`](test_record/2026-8-4/cross_b256_no_publish_reload_1p442ms.json) | B256，96 Scalar，Shared TensorMap，`central_ticket`，K2 异核执行，单向 execution drain |

两份样本均为：

- 256 batches、每批 `Alloc/QK/SF/PV/UP` 共 5 个 task，总计 1,280 个 task；
- 1,024 个真实 A5 engine kernel，其中 QK/PV 为 Cube，SF/UP 为 Vector；
- 32 AIC + 64 AIV；
- trace schema v5，Atomic 与 DCCI 记录均无丢失；
- 泳道时间来自 1 GHz `SYS_CNT`，不是约 1.65 GHz 的 Scalar PMU cycle。

### 2.2 先给结论

1. **两种方式的 Shared TensorMap 业务语义没有变化。**两者都只允许 `task N` 在 `deps_prepared[N-1]` 完成后提交 N 的 writer metadata，再以 `deps_prepared[N]` CAS 把插入资格交给 N+1。Build/Execute 分离没有放宽这条全局严格有序链。

2. **cross-core 的最大理论收益不是“异核执行”本身，而是把全员 replay + Claim tournament 改成每 task 唯一 Build ticket。**same-core 的 B256 有 122,880 个 Submit actor、82,944 次 Claim CAS；cross-core 只有 1,280 个有效 Build actor和 96 个越界 ticket，Build 发放共 1,376 次 FetchAdd。

3. **cross-core 为 Build/Execute 分离支付了新的跨核内存税。**1,024 个 kernel task 各增加一次执行包 clean-out 和一次 executor invalidate，共增加 2,048 次业务 DCCI、23,552 条 cache line；此外还增加 execution cell 的 reserve、BUILT、Claim、DONE 等 Atomic 状态转换。

4. **TensorMap 插入操作数量不变，但等待变重。**两边都发布 1,280 次 `deps_prepared[N]`；本次 cross-core 的前序轮询却从 67,503 次增到 76,012 次，Register core-time 从 21.204 ms 增到 29.730 ms。原因不是插入协议改变，而是中央 ticket 只保证 task id 的领取顺序，不保证分布到不同核后的 Materialize/到达完成顺序。

5. **cross-core 把更多 kernel 推迟到 replay 尾部和 FinalDrain。**same-core 有 975/1,024 个 kernel 在 Submit 的 EfDrain 中执行，FinalDrain 仅 49 个；cross-core 分别为 EfDrain 565、OrchestrationTail 90、FinalDrain 369。构建工作更集中、更快结束，但 executor 的扫描、fanin ready 和执行进度没有完全追平生产前沿。

6. **本次完整泳道中 cross-core 的 startup→最后 FinalDrain 为 1.442 ms，same-core 为 1.767 ms，表面缩短 18.4%。**这个方向与工作量模型一致，但不能把 18.4% 全部归因于生产协议：same-core 记录了 122,880 个 Submit actor，cross-core 只记录 1,280 个，观察扰动本身也显著减少。是否保留优化仍应由同口径、无泳道的端到端 A/B 决定。

---

# 第一大章：两种方式相同的部分

## 3. 相同的 PA 任务图与 engine 工作量

两种方式执行同一份 `Alloc -> QK -> SF/PV -> UP` 任务计划：

```text
每 batch：
Alloc（只做调度和内存准备）
  └─ 4 个 group × (QK -> SF -> PV -> UP)

B256：
256 × 5 = 1,280 task
256 × 4 = 1,024 engine kernel
```

Build owner 或 Execute owner 换核，不允许改变：

- task id 和 function id；
- QK/PV 只能由 AIC 发射，SF/UP 只能由 AIV 发射；
- TensorDesc、scalar、fanin 和 completion vend 的业务值；
- 生产者完成后消费者才能执行的 DAG 依赖。

泳道核对结果也与此一致：两份样本均完成 1,280 个 task 和 1,024 个 kernel。same-core 的全部 kernel core-time 为 32.310 ms；cross-core 在 Submit、OrchestrationTail 和 FinalDrain 中的 kernel union 合计约 33.422 ms。单次诊断 ELF 的指令布局和记录扰动会造成小幅差异，但没有证据表明 cross-core 靠减少真实 kernel 工作取得收益。

## 4. 相同的 SharedOutput 发布合同

两种方式都在 Materialize 中为 fresh output 完成以下发布：

```text
预留 shared heap
  -> 写 shared_outputs[task].tensors[slot] 的 TensorDesc
  -> 对 descriptor cache line 执行 DCCI clean-out + DSB
  -> atomic Exchange 发布 output.published
```

对应的 B256 数量完全相同：

| 操作 | same-core | cross-core |
| --- | ---: | ---: |
| 产生 fresh output 的 task | 1,024 | 1,024 |
| 发布的 TensorDesc | 2,048 | 2,048 |
| descriptor DCCI | 1,024 次 / 4,096 lines | 1,024 次 / 4,096 lines |
| output writer reserve FetchMax | 2,048 | 2,048 |
| output published Exchange | 2,048 | 2,048 |

这说明“构建与执行分离”没有重新分配 output 实际内存，也没有发明第二套 output 所有权。executor 仍然使用 TensorDesc 指向的 shared heap 地址。

## 5. 相同的 TensorMap 严格插入链

这是两种实现最重要的共同不变量。

### 5.1 共同的顺序

[`same_core/common/pa_shared_submit_path.h`](same_core/common/pa_shared_submit_path.h) 与 [`cross_core_ordinary/common/pa_shared_submit_path.h`](cross_core_ordinary/common/pa_shared_submit_path.h) 的核心 Register 逻辑保持同构：

```text
Materialize：并行准备 output descriptor 和 writer delta

Register(task N)：
  N == 0，直接取得插入资格
  N > 0，atomic poll deps_prepared[N-1]
      -> 发布 ordinary TensorMap / symbol writer metadata(N)
      -> CAS deps_prepared[N]：-1 -> N
      -> task N+1 才能进入自己的 metadata commit
```

因此：

- **只有 TensorMap metadata side effect 严格按 task id 串行；**
- Materialize 可以乱序并行；
- Fanin、执行包发布和 kernel 执行不要求按 task id 串行，只受真实 fanin 约束；
- `deps_prepared[N]` 表示 N 的 metadata 已完整发布，不表示 N 的 kernel 已完成。

### 5.2 本 PA 快路径下实际插入了什么

本 PA 的普通输入主要使用 `SharedOutputRef`，所以两份样本所有 1,280 个 Register 的 `ordinary_tensormap_entries` 都是 0。这不等于 Register 没有业务作用：

- 1,280 个 task 都必须推进严格插入完成字；
- 256 个 UP 需要提交 accumulator symbol writer；
- 3 个 INOUT symbol 的 history payload 仍分别保留，共 768 个引用；
- 当前 group commit 把同一 UP 的三个 latest-writer 发布合为一次 CAS，因此两边都是 256 次 writer commit；
- 每个 reader 仍按 `< current task id` 过滤 writer，避免看到未来 task 的覆盖值。

### 5.3 数量不变量

| TensorMap/metadata 操作 | same-core | cross-core | 是否改变 |
| --- | ---: | ---: | --- |
| Register parent | 1,280 | 1,280 | 否 |
| 等待前序记录 | 1,279 | 1,279 | 否，task 0 无前序 |
| 插入完成 CAS | 1,280 | 1,280 | 否 |
| ordinary entry | 0 | 0 | 否，本 PA 快路径决定 |
| UP group writer commit CAS | 256 | 256 | 否 |
| writer history DCCI | 256 calls / 256 lines | 256 calls / 256 lines | 否 |
| TensorMap tail load | 1,280 | 1,280 | 否 |

所以，cross-core 没有通过弱化 TensorMap 顺序换性能。变化的是 owner 到达有序链的时刻和等待强度，后文单独分析。

## 6. 相同的 fanin 与 completion 语义

Build 阶段在两种方式中都必须：

1. 只接受 `producer_task_id < current_task_id`；
2. 根据 latest writer/history 解析 INOUT 前任；
3. 把最终 fanin task id 固化到执行包；
4. executor 在所有 fanin completion flag ready 后才能发射 kernel；
5. kernel 真正结束后先发布 completion vend，再发布 completion flag。

same-core 的 1,280 个 task 各发布一次普通 completion vend/flag。cross-core 把它按任务类型拆成：

- 256 个 Alloc 继续走普通 completion vend/flag；
- 1,024 个 kernel task 由 Execute owner 走 `shared_exec_completion_vend/flag`。

两类合计仍然各为 1,280 次。换言之，completion 的发布责任从 Build owner 转给了 Execute owner，但“每 task 一次 vend + 一次 flag”的业务数量和先后关系没有变化。

## 7. 相同的 A5 cacheline 原则

A5 Scalar 之间没有硬件 cache coherence，两种方式都遵守同一条内存规则：

- Atomic control line 只做 Atomic，不用 ordinary dirty store，也不对它做 payload DCCI；
- ordinary payload 由唯一 writer 写完后，逐 cache line clean-out，并以尾部 DSB 收口；
- 跨核 reader 在取得所有权后先 invalidate，再 ordinary load；
- 相邻、可由不同核并发拥有的对象不能共享 payload cache line。

cross-core 只是把这个既有规则扩展到新的 execution payload。它没有改变 SharedOutput、TensorMap history 和 task completion 原有的发布合同。

---

# 第二大章：两种方式不同的部分

## 8. Build 所有权：全员 replay Claim 与中央 ticket

### 8.1 same-core 的工作量模型

same-core 中 96 个 worker 都回放全部 1,280 个逻辑 task，因此产生：

```text
Submit actor = 96 × 1,280 = 122,880
```

并非每个 actor 都做 Claim。按当前 AIC/AIV 候选拓扑：

```text
Alloc： 256 × 96 = 24,576 次 local CAS
QK/PV：512 × 32 = 16,384 次 local CAS
SF/UP：512 × 64 = 32,768 次 local CAS
合计 local CAS      = 73,728

各 local group winner 再参加 root CAS：9,216 次
Claim CAS 总计：73,728 + 9,216 = 82,944
```

最终只有 1,280 个 winner，其余为：

- 72,448 个 true loser；
- 49,152 个 not-attempted actor。

这套结构的优点是 task 到达自然分散，winner 直接拥有本核 Build/Execute 生命周期；代价是大量重复 replay、候选判断和返回型 Atomic 仲裁。

### 8.2 cross-core 的工作量模型

cross-core 使用一个单调 `build_dispatch.next_task`：

```text
ticket = atomic FetchAdd(next_task, 1)
ticket < 1,280：本核是该 task 唯一 Build owner
ticket >= 1,280：本 worker 结束生产
```

因此 B256 的精确调用数为：

```text
1,280 个有效 ticket + 96 个 worker 的终止 ticket = 1,376
```

不存在 same-core 意义上的 true loser，也不再生成 122,880 个 Submit actor；每个 task 只有一个 owner actor。中央 ticket 保证“每个 task 恰好 Build 一次”，但不承担 TensorMap 插入顺序，后者仍由 `deps_prepared` 链完成。

### 8.3 理论收益与边界

cross-core 在 Build 前端直接消除了：

- 82,944 次 Claim tournament CAS；
- loser/not-attempted 的控制壳；
- 非 owner 对 PA callback、Submit 边界和观察记录的重复处理；
- winner 由 Claim 结果反向绑定到执行核的限制。

它新增 1,376 次 ticket FetchAdd。单从返回型仲裁工作量看：

```text
82,944 -> 1,376，减少 81,568 次，约 98.34%
```

这就是 cross-core 最确定的理论收益来源。它与泳道中 Claim core-time 从 25.379 ms 降到 0.614 ms（下降 97.58%）方向一致。

### 8.4 Build 角色拓扑也随之解耦

same-core 的 Build owner 随后必须在本核执行，所以 Claim 候选天然受 engine 角色约束：QK/PV 只由 32 个 AIC 竞争，SF/UP 只由 64 个 AIV 竞争，Alloc 才允许 96 核参与。

cross-core 的中央 ticket 可以由任意 Scalar 领取。Build owner 只负责构造 portable payload；真正 Execute owner 再由与 engine 匹配的 K2 候选产生，并显式排除 Build owner。这个变化有两面：

- 好处是 96 个 Scalar 可以共同承担参数构造、Materialize、Register、Fanin 和 payload pack，Build 负载不再被 32/64 的执行角色边界限制；
- 代价是即使 Build owner 恰好具备目标 engine，也不能复用本核 private slot 直接执行，仍必须完成统一的 shared payload 发布和异核取得协议。

因此，中央 ticket 同时改变了“仲裁次数”和“Build 负载放置”。前者可由 Atomic 数量直接证明；后者对关键路径的收益取决于 AIC/AIV 当次负载，必须通过端到端 A/B 判断，不能只按 96 核平均分摊做静态估算。

## 9. Execute 所有权：本核 LocalSlot 与跨核 SharedExecCell

### 9.1 same-core

same-core winner 完成 Materialize/Register/Fanin 后，把执行所需的 tensor、scalar、fanin 和上下文写入本 worker 的 private slot。后续仍由同一物理 Scalar 在 EfDrain/FinalDrain 中读取并执行。

由于没有跨核 reader：

- slot 不需要发布 `EMPTY -> BUILDING -> BUILT -> CLAIMED -> DONE` 状态机；
- Build payload 不需要为另一个 Scalar 做 clean-out；
- 执行前不需要为跨核可见性 invalidate；
- completion vend 可以沿用 Build owner 保存的本核 heap 快照。

### 9.2 cross-core

cross-core 在 [`cross_core_ordinary/common/shared_exec_protocol.h`](cross_core_ordinary/common/shared_exec_protocol.h) 中为每个 task 建立 `SharedExecCell`：

```text
cacheline 0：packed Atomic control
cacheline 1..N：immutable portable payload
```

Build owner 的发布顺序是：

```text
CAS EMPTY -> BUILDING
  -> 在 GM cell 中 pack portable payload
  -> payload clean-out + DSB
  -> CAS BUILDING -> BUILT
```

Execute candidate 的取得顺序是：

```text
atomic load control
  -> CAS BUILT -> CLAIMED，确定唯一 Execute owner
  -> invalidate payload + DSB
  -> 校验 header/layout，重建本执行核 dispatch binding
  -> 等 fanin ready
  -> 发射并等待 kernel
  -> 发布 completion vend/flag
  -> CAS CLAIMED -> DONE
```

当前每个 worker 有两个紧凑 token，只保存可变 phase、fanin 前缀和本核 dispatch binding；immutable payload 仍直接指向 task-indexed `SharedExecCell`，不再复制到第二份 GM token。

### 9.3 新增 Atomic

本次 cross-core B256 的 execution 协议新增：

| 操作 | 次数 | 作用 |
| --- | ---: | --- |
| `shared_exec_build_reserve` CAS | 1,024 | 验证每个 kernel cell 只有一个 builder |
| `shared_exec_built_publish` CAS | 1,024 | payload DCCI 完成后发布可读状态 |
| `shared_exec_cell_state_load` | 2,723 | K2 candidate 观察 cell 状态 |
| `shared_exec_claim` CAS | 1,025 | 1,024 个 winner，仅出现 1 次竞争 loser |
| `shared_exec_done_publish` CAS | 1,024 | completion 发布后封口执行生命周期 |
| `shared_exec_drain_arrive` FetchAdd | 96 | 每 worker 向所属完成组单向到达 |

这些操作是 Build/Execute 分离的新增成本。它们没有抵消 Claim tournament 的数量收益：仅表中前五项的 execution control 就有 6,820 次，其中 2,723 次是状态观察；但它们解释了为什么 cross-core 不会“只剩 1,376 次 Atomic”。

> 注：上表中前五项的精确合计为 6,820 次，连同 96 次 drain arrival 为 6,916 次；另有 1,024+1,024 次 cross-core 专用 completion vend/flag，但这两类取代了 same-core kernel task 的普通 completion 发布，不能重复算作业务总量增加。

## 10. 跨核执行包带来的 DCCI 增量

### 10.1 先排除观察者导出流量

泳道结束后，observer 会把 raw buffer clean-out 给 host。它发生在业务生命周期之后，且记录规模随事件数量变化：

- same-core observer export：288 calls / 99,460 lines；
- cross-core observer export：288 calls / 73,758 lines。

因此，metadata 中的总 DCCI lines 从 108,964 降到 106,814，并不能说明 cross-core 业务 DCCI 更少。必须先扣掉 observer export。

### 10.2 生命周期内 DCCI 对比

下表扣除了生命周期结束后的 observer export，但仍包含每核一次 startup config invalidate。为行文简洁，后文把这个口径简称为“业务 DCCI”；startup 在两边均为 96 calls / 288 lines，不影响执行包增量结论。

| 指标 | same-core | cross-core | 变化 |
| --- | ---: | ---: | ---: |
| 业务 DCCI calls | 4,192 | 6,240 | +2,048，+48.86% |
| 业务 DCCI lines | 9,504 | 33,056 | +23,552，+247.82% |
| 业务 DCCI 记录跨度 core-time | 790.255 µs | 2,282.233 µs | +1,491.978 µs |

新增量恰好由执行包交接组成：

| cross-core 独有操作 | calls | lines | 本次记录跨度 |
| --- | ---: | ---: | ---: |
| builder `shared_exec_payload_flush` | 1,024 | 11,776 | 1,452.690 µs |
| executor `shared_exec_payload_invalidate` | 1,024 | 11,776 | 73.957 µs |
| **合计** | **2,048** | **23,552** | **1,526.647 µs** |

payload 的形状是：

- 768 个 QK/SF/PV task 各 10 条 cache line；
- 256 个 UP task 各 16 条 cache line；
- 平均每 kernel task 11.5 条 cache line。

builder flush 比 executor invalidate 的记录跨度大很多，原因不能只解释成 DCCI 指令本身：builder 区间还受 dirty line 写回、总线排队和尾部 DSB 完成时间影响；executor 面对的是已经发布的 immutable payload。两者仍是同一个跨核可见性合同的生产者和消费者两端，不能删除任一端。

### 10.3 共同 DCCI 保持稳定

除执行包之外，两种方式的主要业务 DCCI 数量保持一致：

| 共同操作 | same-core | cross-core |
| --- | ---: | ---: |
| SharedOutput descriptor flush | 1,024 calls / 4,096 lines | 1,024 / 4,096 |
| Build source descriptor invalidate | 2,048 / 4,096 | 2,048 / 4,096 |
| INOUT history invalidate | 768 / 768 | 768 / 768 |
| UP writer history flush | 256 / 256 | 256 / 256 |
| startup config invalidate | 96 / 288 | 96 / 288 |

site 名称中 `shared_winner_build_descriptor_invalidate` 与 `shared_exec_build_source_descriptor_invalidate` 不同，但数量和语义相同：都是 Build 阶段取得 SharedOutput TensorDesc，而不是 execution payload 的跨核交接。

## 11. Atomic 总量与分布变化

### 11.1 总量

| 指标 | same-core | cross-core | 变化 |
| --- | ---: | ---: | ---: |
| 逻辑 Atomic 调用 | 208,093 | 116,801 | -91,292，-43.87% |
| 直接 Atomic 调用 | 133,290 | 28,355 | -104,935，-78.73% |
| PollBatch 代表的逻辑 load | 74,803 | 88,446 | +13,643，+18.24% |
| Atomic raw records | 134,840 | 29,994 | -104,846，-77.76% |

这个结构比“Atomic 总数下降”更有解释力：

- 大量一次性 Claim CAS 被删除，所以直接调用大幅下降；
- TensorMap 插入、fanin ready 和 FinalDrain 仍会等待，所以批量轮询反而增加；
- cross-core 的主要矛盾从“同 task 多核 Claim 竞争”转成“有序插入等待 + producer/consumer 进度差”。

### 11.2 关键 Atomic 对照

| 类别 | same-core | cross-core | 解释 |
| --- | ---: | ---: | --- |
| Build owner 仲裁 | local/root CAS 82,944 | ticket FetchAdd 1,376 | 最大数量消减 |
| strict insert completion | 1,280 CAS | 1,280 CAS | 协议不变 |
| strict insert predecessor load | 67,503 | 76,012 | 乱序到达使轮询增加 |
| heap cursor/reserve/vend | 4,352 | 4,352 | Materialize 合同不变 |
| output reserve/published | 4,096 | 4,096 | SharedOutput 合同不变 |
| INOUT last-writer load/commit | 2,304 | 2,304 | writer 语义不变 |
| TensorMap tail load | 1,280 | 1,280 | lookup 边界不变 |
| fanin completion flag load | 31,066 | 15,875 | 本次到达/执行相位变化，不是固定协议常数 |
| execution state/load/claim/done | 0 | 6,820 | cross-core 新增 |

当前 cross-core 还比 same-core 少两类读取：

- 1,280 次无回收模式下恒为 0 的 TensorMap head load；
- 2,048 次已被严格 completion chain 覆盖的 output `published` load。

这两项是当前 cross-core 实现中已验证的通用优化，不是“异核执行天然不需要”的结果。若把它们迁回满足同样证明前提的 same-core，理论上也可以获得同类收益，不能把全部差值都归因于架构分离。

## 12. TensorMap 插入协议不变，但 Register 为什么更慢

### 12.1 实测变化

| Register 指标 | same-core | cross-core | 变化 |
| --- | ---: | ---: | ---: |
| parent core-time | 21.204 ms | 29.730 ms | +40.21% |
| predecessor wait core-time | 20.604 ms | 29.087 ms | +41.17% |
| predecessor logical loads | 67,503 | 76,012 | +12.61% |
| metadata publish core-time | 0.169 ms | 0.075 ms | 本体仍很小 |
| insert completion core-time | 0.431 ms | 0.569 ms | 数量仍为 1,280 |

### 12.2 理论原因

中央 ticket 的线性化顺序只是：

```text
某核先拿到 task N，另一核随后拿到 task N+1
```

它不保证：

```text
task N 的 owner 一定先完成参数构造和 Materialize，
也不保证它先到达 Register。
```

不同核的角色、当前 token、GM miss、Materialize output 数量和 DCCI 时间都不同。于是可能出现：

```text
core A 领取 N，但 Materialize 较慢
core B 领取 N+1，很快到达 Register
core B 必须持续 poll deps_prepared[N]
```

same-core 的 winner 也会乱序到达 Register，但它的 owner 由每 task Claim 的自然到达竞争产生，较快到达该 task 的核更容易赢。central ticket 则在 Materialize 之前分配唯一 owner，领取快不等于完成快，因而更容易把到达差异显式变成有序链等待。

这说明 Register 回退不是 TensorMap 写操作变多，也不能通过删除严格顺序修复。可优化的对象是：

- 减少 owner 在取得 ticket 后、进入有序 commit 前的非必要工作；
- 让可并行准备停留在栈/本地 staging，串行区只消费预计算 delta；
- 改善 Build owner 的负载分布或在不破坏“已领取 task 必须闭合”的前提下减少到达乱序；
- 不能把 `deps_prepared` 链替换成不保证 task-id 顺序的 owner election。

## 13. Kernel 调度落点与 FinalDrain 差异

### 13.1 实测落点

| kernel 所在区域 | same-core | cross-core |
| --- | ---: | ---: |
| Submit 内 EfDrain | 975（95.21%） | 565（55.18%） |
| OrchestrationTail | 0 | 90（8.79%） |
| FinalDrain | 49（4.79%） | 369（36.04%） |
| 合计 | 1,024 | 1,024 |

### 13.2 理论原因

same-core winner 已经拥有完整 local slot，后续每次 Submit 的 EfDrain 只需检查本核 slot fanin；owner、payload 地址和执行角色都已确定。

cross-core 则多出一条 producer/consumer 流水：

```text
Build ticket owner
  -> 完成严格 TensorMap Register
  -> 发布 BUILT payload
  -> K2 candidate 扫描到该 task
  -> CAS 取得 Execute owner
  -> invalidate/rebind
  -> fanin ready
  -> kernel
```

即使 Build 前端更快，Execute candidate 仍可能因为以下原因落后：

- 只扫描本核 K2 candidate 序列，而不是直接消费本核刚 Build 的 task；
- 每核两个 token 可能被尚未 ready 的 task 占用；
- candidate 首选/备选和 Build owner 排除会产生短暂 defer；
- payload acquire、dispatch rebuild 和 fanin 检查增加了每个执行 task 的前置工作；
- Build 与 Execute 分离后，生产完成不再等价于本核执行队列已排空。

因此 cross-core 的 Submit 区间更短，并不代表全部工作已经结束。必须使用 startup→最后 FinalDrain 的完整周期，而不能只拿 1.006 ms Submit 宣称端到端性能。

## 14. 尾同步协议不同

same-core 使用 `two-16` replay-done 屏障：所有 replay actor 停产、本核 private slot 排空后，再通过分组到达和 release 收口。本次记录中有 129 次 replay-done increment 和 5,609 次逻辑 poll。

cross-core 的 FinalDrain 要证明更强的终态：

- Build ticket 已耗尽，不再生产新 cell；
- 本核 candidate scanner 已封口；
- 两个 execution token 都恢复 IDLE；
- engine 无 in-flight；
- 本核成功发布 DONE 的数量已确定。

满足这些条件后，每个 worker 只对所属 16 组之一执行一次 arrival FetchAdd，并把到达数和完成数一起汇总。固定 root 读取各组结果，确认 96 个 worker 到齐且 DONE 总数等于计划中的 1,024；不再做反向 release fanout，也不再逐 task 原子扫描终态。

本次 cross-core 对应 96 次 arrival FetchAdd 和 60 次 root 逻辑 poll。这个收口优化依赖 cross-core 已经建立的 cell/DONE/token 不变量，是当前实现差异；它不是 TensorMap 插入协议的一部分。

## 15. 分阶段性能差异

下表是 96 核累计 core-time，不是墙钟；多个核的等待高度并发，不能把某行差值直接当作端到端收益。

| Submit 独占区域 | same-core | cross-core | 变化 | 理论解释 |
| --- | ---: | ---: | ---: | --- |
| Claim | 25.379 ms | 0.614 ms | -97.58% | tournament 被中央 ticket 取代 |
| EfDrain control | 21.288 ms | 13.016 ms | -38.86% | 不再为 122,880 个 Submit actor 都推进本地 slot；但新增 token/scanner |
| Register | 21.204 ms | 29.730 ms | +40.21% | 严格链不变，ticket owner 到达更乱序 |
| Materialize | 12.309 ms | 12.013 ms | -2.41% | SharedOutput/heap 工作基本不变 |
| Submit residual | 11.988 ms | 2.406 ms | -79.93% | 全员 callback/边界/loser 工作被大量删除 |
| WinnerBuild | 7.287 ms | 8.865 ms | +21.66% | LocalSlot build 变成 shared payload pack + publish |
| Fanin | 4.902 ms | 3.862 ms | -21.22% | 当前版去掉重复 published/head load；单样本到达相位也不同 |
| AllocComplete | 0.422 ms | 0.378 ms | -10.55% | 语义基本相同，属于样本差异 |
| **Submit Scalar/control 合计** | **104.778 ms** | **70.884 ms** | **-32.35%** | 前端冗余消减大于跨核交接新增工作 |

cross-core 的主要“赢”和“输”非常清楚：

```text
赢：Claim、重复 Submit/replay、同核 slot 推进冗余
输：Register 等待、shared execution payload pack/DCCI/acquire、scanner/token 状态机
```

## 16. 墙钟结果应如何解释

### 16.1 同一泳道口径下的观测值

| 指标 | same-core | cross-core |
| --- | ---: | ---: |
| Submit actor 数 | 122,880 | 1,280 |
| Submit 全局区间 | 1.659 ms | 1.006 ms |
| startup→最后 FinalDrain | 1.767 ms | 1.442 ms |
| 完整周期表面差值 | — | -0.325 ms，-18.39% |

理论上，cross-core 删除的大量 Claim/replay 工作足以支持“方向应当改善”的判断；但 18.39% 不是干净的生产收益，原因有三点：

1. 两份泳道来自不同日期、不同代码版本，不是冻结 ELF 的交替 A/B；
2. same-core 的 trace raw records 为 397,304，cross-core 为 51,306；观察写入本身减少 87.09%；
3. same-core 有 134,840 条 Atomic record、122,880 个 Submit parent，cross-core 分别只有 29,994 和 1,280，代码体积、I-cache、D-cache 与总线交错都不同。

所以，这两份泳道可以证明**结构和性能机理**，也可以说明 cross-core 已经没有离谱回退；但不能单独给出“生产性能精确提升 18.39%”的结论。

### 16.2 当前 cross-core 的无观察数据

最新 cross-core 去掉 2,048 次重复 output `published` load 后，24 对 trace-free B256 的结果为：

```text
冻结基线 median：1.419225 ms
当前版本 median：1.411355 ms
配对收益中位数：+0.556%
```

该数据只裁决 cross-core 内部的 S6.61 优化。历史 same-core perf-clock 的起止边界和当前 startup→FinalDrain 口径并非同一冻结实验，因此本文不把两个 trace-free 绝对值直接相减。

## 17. 理论性能模型

令：

- `W = 96`：Scalar 数；
- `N = 1,280`：task 数；
- `E = 1,024`：kernel task 数；
- `C_same = 82,944`：same-core Claim CAS 数；
- `L_exec = 23,552`：cross-core execution payload 的发布+取得 cache lines。

可以把两种方式的主要调度工作抽象为：

```text
T_same ≈ W×N 的 replay/边界工作
       + C_same 的返回型 Claim 竞争
       + TensorMap 严格插入链
       + E 个本核 slot Build/执行
       + fanin/completion/尾同步

T_cross ≈ (N+W) 个 Build ticket
        + N 个唯一 owner 的参数构造/Submit
        + TensorMap 严格插入链
        + E 个 shared payload pack/publish/acquire
        + E 个 Execute owner 状态机
        + fanin/completion/execution drain
```

两者共同的 TensorMap 严格插入链不能在差分中消掉“等待时间”，因为 owner 到达分布不同；但其业务操作数和顺序不变量可以消掉。cross-core 只有在下式成立时才会整体获益：

```text
被删除的全员 replay + Claim 竞争
  > 新增的 execution payload DCCI + Execute 状态机
    + 因 owner 乱序增加的 Register 等待
    + execution backlog 的尾部成本
```

当前数据满足这个方向：Submit Scalar/control core-time 下降 32.35%，完整泳道也没有端到端回退。但 cross-core 的进一步上限已经不再取决于 Claim tournament，而主要取决于右侧三个新增项。

## 18. 后续优化优先级与不可越过的边界

### 18.1 优先级一：降低执行包的跨核搬运税

当前每 kernel task 平均发布并取得 11.5 条 cache line，builder flush 是新增 DCCI 的最大块。可以继续审视：

- portable payload 是否仍携带 executor 可从 immutable plan 重建的字段；
- UP 的 16-line active prefix 是否可在保持通用 ABI 的前提下压缩；
- 是否存在 Build 后重复读取 source descriptor 或重复 layout 校验；
- 所有字段是否先在栈上准备，再单向写 GM，避免同一 line 多次变脏。

不能删除 payload clean-out、BUILT 发布、winner invalidate，不能让 Atomic control 与 ordinary payload 共线。

### 18.2 优先级二：减少严格插入链前的到达乱序和空等

Register 约 98% 仍是等待。目标不是放松 TensorMap 顺序，而是让 owner 更接近“准备好就提交”：

- 串行区外完成 bucket/key/delta 等预计算；
- 串行区只做必要 writer commit 和 completion CAS；
- 避免 ticket owner 在持有前序 baton 后被非必要执行工作阻塞；
- 用端到端数据判断负载分配是否让慢角色长期卡住插入前沿。

### 18.3 优先级三：让 Execute 消费追上 Build 生产

当前 36.04% kernel 落在 FinalDrain，说明生产和消费仍不平衡。应继续核对：

- K2 scanner 是否做了重复 GM state load；
- 两 token 是否经常被未 ready task 同时占满；
- preferred/fallback defer 是否形成无效空转；
- fanin ready 后是否能立即优先执行；
- OrchestrationTail/FinalDrain 的状态检查是否有重复 Atomic 或 GM 访问。

这里优化的是执行发现和本核 admission，不是让 kernel 按 task id 串行，也不是把 TensorMap 插入链扩展到执行阶段。

### 18.4 明确不能用的“优化”

- 取消 `deps_prepared[N-1] -> metadata(N) -> deps_prepared[N]`；
- 用 Build ticket 的领取顺序冒充 TensorMap 完成顺序；
- 让 executor 在 `BUILT` 前读取 ordinary payload；
- 只用 Atomic 状态发布而省略 payload DCCI，或只做 DSB 不做 DCCI；
- 在 payload line 中混放 Atomic control；
- 为减少 FinalDrain 数字而把未完成 kernel 排除在性能终点外；
- 只依据泳道 core-time 保存优化，而不做无观察 startup→FinalDrain A/B。

## 19. 最终判断

cross-core 与 same-core 的核心差异不是“TensorMap 是否 shared”，两者本来都使用同一份 Shared TensorMap；真正差异是：

```text
same-core：Build 所有权、执行包所有权、Execute 所有权绑定在同一 worker

cross-core：
  中央 ticket 决定唯一 Build owner
  deps_prepared 链独立决定 TensorMap 提交顺序
  SharedExecCell 发布可跨核执行包
  K2 CAS 决定唯一 Execute owner
  execution drain 证明全部 Execute 生命周期闭合
```

这种解耦在理论上以“删除全员 replay 和高竞争 Claim”为收益，以“新增执行包 DCCI/Atomic 和 producer-consumer backlog”为代价。当前 B256 数据说明前者暂时大于后者；同时也明确暴露出 Register 等待、execution payload 交接和 FinalDrain backlog 是 cross-core 后续真正需要优化的三个区域。

最重要的是：**TensorMap 严格顺序插入从未被删除，也不应成为 Build/Execute 分离的牺牲品。**后续所有性能候选都应在这条不变量、完整 fanin 过滤、跨核 payload 发布和 startup→FinalDrain 完整周期四个条件同时成立时再保留。
