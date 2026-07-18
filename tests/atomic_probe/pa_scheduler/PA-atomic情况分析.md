# A5 FDWIC Paged Attention Submit 原子操作与优化记录

## 1. 范围与当前结论

本文记录 `TestPagedAttentionUnroll::Case1` 在真实 A5 上的 FDWIC AICore
Submit 路径，供后续继续优化。快照日期为 2026-07-17，当前代码提交为
`e3b748b43c7f226c025c4dcdfc2eb2805cec7f21`。

范围限定为：

- 用例：`examples/a5/fully_distributed_within_core/paged_attention_unroll/`；
- Case：`Case1`；
- runtime：`fully_distributed_within_core`；
- 平台：真实 A5，A5Sim 只用于功能回归；
- 性能口径：所有 worker 中最早的 `Submit.start` 到最晚的 `Submit.end`；
- 原子操作清单：该用例实际经过的 FDWIC Submit、执行完成和外围生命周期路径。

当前结论：

- 96 个 worker（32 AIC + 64 AIV）分别回放 1280 次 Submit，共有
  122880 个完整 Submit 事件；
- 本轮已经消除了纯单 lane 图中 146944 次无效的 BlockWon
  `atomic_load(any_pub)`，在 A5 上实际对应 `atomicAdd(addr, 0)`；
- 仍在 Submit 热路径中执行且数量最大的原子操作是 Claim：固定
  73728 次 `atomicMax`；
- 三轮最终版本的首末 Submit 中位数为 5.115620 ms，相比
  5.642245 ms 基线下降 0.526625 ms，即 9.33%；最好单轮为
  5.096685 ms；
- 优化没有改变通用 atomic 语义，也没有把任务推迟到最终 drain 来制造
  表面收益；下一优先级应是 Claim 竞争，其次是 completion/frontier 和
  fanin ready 轮询。

环境安装、编译和基线复现过程见
[A5 FDWIC Paged Attention 安装与复现指南](../A5_FDWIC_PAGED_ATTENTION_REPRO.md)。

## 2. Case1 工作量与 atomic 语义

Case1 的关键参数是 `batch=256`、`num_heads=16`、`block_size=128`、
`context_len=8192`。当前 orchestration 中 `N_UNROLL=64`，因此每个 batch
只有一个 block group，且 `q_loop=1`。每个 worker 回放的 task id 分布如下：

| task id | 每 batch 操作 | 执行 lane | fanin 数 | 新分配输出 |
| ------- | ------------- | --------- | -------: | ---------- |
| `5b+0` | Alloc | 任一 worker 竞争 winner | 0 | 有 |
| `5b+1` | QK | AIC | 0 | 有 |
| `5b+2` | SF | AIV | 1 | 有 |
| `5b+3` | PV | AIC | 1 | 有 |
| `5b+4` | UP | AIV | 3 | 无，仅 INOUT |

其中 `b=0..255`，所以每核恰好回放 1280 个 task；全局实际执行
256 个 Alloc 和 1024 个 kernel task。

A5 CCEC 下的封装位于
`src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/common/atomic.h`：

| C++ 封装 | A5 指令封装 | 说明 |
| -------- | ----------- | ---- |
| `atomic_load(x)` | `atomicAdd(&x, 0)` | 不是普通 load，而是 read-modify-write |
| `atomic_exchange(x, v)` | `atomicExch(&x, v)` | 发布或重置共享状态 |
| `atomic_fetch_add(x, v)` | `atomicAdd(&x, v)` | 计数器递增 |
| `atomic_fetch_sub(x, v)` | `atomicSub(&x, v)` | joint task 完成计数 |
| `atomic_fetch_max(x, v)` | `atomicMax(&x, v)` | Claim 和 frontier 前推 |

因此，即使源码写的是 `atomic_load`，大量 worker 读取同一个地址时仍会形成
RMW 竞争。本文中的次数分为三类：

- “固定”表示可以由 Case1 拓扑和源码精确推出；
- “下界”表示每次正确执行至少发生这些操作，竞争和未就绪轮询会增加次数；
- “条件”表示 trace 只能确认分支是否等待，不能直接得到 atomic 指令计数。

## 3. 当前原子操作分布

### 3.1 Claim cursor：当前最大固定项

调用链为
`submit_runtime.h::dist_submit_claim_*()` ->
`submit_helpers.h::claim()` -> `atomic_fetch_max(cursor, task_id)`。
Claim span 会在所有 worker 上记录，但只有符合 lane 的 worker 执行
`atomicMax`；Alloc 则由全部 96 个 worker 竞争。

| 操作 | task 数 | 每 task 竞争 worker | 固定 atomicMax 次数 |
| ---- | ------: | ------------------: | ------------------: |
| Alloc | 256 | 96 | 24576 |
| QK | 256 | 32 AIC | 8192 |
| SF | 256 | 64 AIV | 16384 |
| PV | 256 | 32 AIC | 8192 |
| UP | 256 | 64 AIV | 16384 |
| 合计 | 1280 | - | **73728** |

每核累计 Claim 中位数中，AIC 从基线 0.517 ms 上升到最终三轮中位数
0.608 ms，AIV 从 0.515 ms 上升到 0.613 ms。跳过 BlockWon 轮询后
worker 更同步，cursor 瞬时竞争反而变强。总时间仍然下降，但 Claim 已成为
最明确的剩余原子热点。

### 3.2 Heap 容量保护：winner action tail

位置为 `submit_runtime.h::dist_submit_wait_heap_capacity()`。只有 winner 且
`output_bytes>0` 时进入检查；本 Case 每 batch 的 Alloc、QK、SF、PV 满足，
UP 不满足，所以共有 1024 次有效调用。

最好一轮没有 `RingBp` 事件，表示每次调用都在第一次循环检查后返回：

- `fatal_set()`：固定 1024 次 `atomic_load(g_dist.fatal)`；
- frontier：固定 1024 次 `atomic_load(g_dist.frontier)`；
- `load_task_vend(F-H)`：当索引非负时执行 `atomic_load(cell.vend)`，本轮
  次数介于 0 和 1024，现有 trace 不能给出精确值；
- 如果未来出现 heap backpressure，上述三项会在循环中重复，耗时会归入
  `RingBp`。

无等待时，这些 atomic 位于 winner 的 Build/Alloc action 中。对 kernel
Submit 来说，泳道上表现为 `Build.end` 到 `Submit.end` 的尾段。

### 3.3 fanin ready 与 task completion

依赖轮询位于 `submit_core.h::drain_phase_b()`：每个 fanin 通过
`task_flag_ready()` 执行一次 `atomic_load(cell.flag)`。QK、SF、PV、UP 的
fanin 分别为 0、1、1、3，因此任务真正开始执行前至少有：

~~~text
256 * (0 + 1 + 1 + 3) = 1280 次 atomic_load(flag)
~~~

依赖未就绪时会提前退出并在下一次 drain 重试，所以 1280 只是成功检查的
下界，不包含失败轮询。它们发生在执行 kernel 的 drain 位置：通常是下一个
Submit 的 `EfDrain`，有背压时可能在 `RingBp`，剩余任务则在最终 drain。

每个 task 完成一次，路径为
`execute_slot()/dist_submit_complete_alloc()` ->
`complete_executed_task()` -> `advance_frontier()`：

| 共享状态 | 操作 | 次数性质 | Case1 次数 |
| -------- | ---- | -------- | ---------: |
| `task.vend` | `atomic_exchange` | 每 task 固定一次 | 1280 |
| `task.flag` 发布 | `atomic_exchange` | 每 task 固定一次 | 1280 |
| `frontier` 初始读取 | `atomic_load` | 每 task 固定一次 | 1280 |
| 后续 `task.flag` ready | `atomic_load` | 下界 | >=1280 |
| `frontier` 前推 | `atomic_fetch_max` | 下界，竞争时可重复 | >=1280 |

最终最好一轮的 1024 个 kernel 中，1011 个在某个 Submit 的 `EfDrain`
中执行，0 个在 `RingBp` 中执行，13 个在本核最后一个 Submit 之后的最终
drain 中执行。completion atomic 紧随 kernel，分布也基本相同；Alloc 的
completion 则在 Alloc winner 的 Submit 内完成。基线对应分布是
979 个 `EfDrain`、29 个 `RingBp`、16 个最终 drain。

### 3.4 BlockWon：代码仍保留，当前 Case 动态次数为零

BlockWon 用于同一物理 block 内至少两个 lane 联合执行一个 task。相关位置：

- `submit_helpers.h::alloc_won_slot()`：`atomicMax(state)`；
- `submit_helpers.h::populate_won_slot_from_submit()`：`atomicExch(drained)`；
- `submit_core.h::publish_won_slot()`、`clear_won_slot_state()`：
  `atomicExch(state)`；
- `submit_core.h::claim_won_lane()`：`atomicExch(drained)`；
- `submit_core.h::decrement_won_remaining_is_last()`：`atomicSub(remaining)`；
- `submit_runtime.h::publish_joint_deposits()`：`atomicExch(any_pub)`；
- `submit_core.h::drain_block_won()`、`has_pending_won()`：读取
  `any_pub/state/drained`，在 A5 上均为 `atomicAdd(addr, 0)`。

PA Case1 的 QK/PV 为单 AIC lane，SF/UP 为单 AIV lane，没有 joint task，
trace 中也没有 `DrainWon`。当前由 worker-local 的
`g_fdwic_joint_submit_seen` 关闭这些轮询，所以以上 BlockWon atomic 在该 Case 的 AICore
Submit/执行路径中动态次数为 **0**；AICPU 初始化仍会重置相关状态，joint 用例也仍走完整原协议。

### 3.5 Submit 窗口外和非本路径 atomic

这些操作不属于用户关注的首末 Submit 耗时，但做端到端优化时不能忽略：

- worker 启动屏障，`core_main.h`：96 次
  `atomic_fetch_add(started_count)`，以及等待期间重复读取
  `started_count/fatal`；
- worker 最终 drain，`submit_runtime.h`：96 次
  `atomic_fetch_add(replay_done)`，以及等待期间重复读取 `replay_done`；
- AICPU 初始化，`aicpu/control_plane.h`：12 个分片 cursor、frontier、
  fatal、replay_done、started_count、32 个 BlockWon 的 `any_pub` 和
  128 个 won slot state 均以 atomic exchange 重置；65536 个 task cell
  各重置 flag/vend，共 131072 次 atomic exchange；
- AICPU 外层生命周期，`aicpu/aicpu_executor.cpp`：线程状态使用
  `std::atomic`，并轮询 `runtime->dist.done_count`。真实 A5 的 AICore 通过
  COND 寄存器发布完成，不走 A5Sim 中的 `done_count` atomic add；
- `debug_dump.h` 中的 atomic load 只在诊断 dump 时运行；centralized
  scheduler 的 completion mailbox 和 SDMA completion atomic 不属于本
  FDWIC PA 执行路径，未计入本文。

异常路径中的 `set_fatal()` 使用 `atomic_exchange(g_dist.fatal, 1)`；上述
成功运行没有触发。由于 fanin/frontier/屏障轮询次数依赖时序，不能把静态
下界当作整次运行的 atomic 总数。若需要精确动态计数，应使用 PMU 或独立
诊断构建；直接在热路径增加共享计数器会反过来改变竞争形态。

## 4. 当前优化逻辑与效果

### 4.1 首个 joint task 前跳过 BlockWon 轮询

涉及文件：

- `dist_engine/common/worker_state.h`：增加 worker-local、单调的
  `g_fdwic_joint_submit_seen`；
- `dist_engine/aicore/core_main.h`：每次运行开始时重置为 `false`；
- `dist_engine/aicore/submit_runtime.h`：当当前 active core mask 的
  popcount 大于等于 2 时，在该 Submit 的首次 drain 前置为 `true`；
- `dist_engine/aicore/submit_core.h`：flag 为 `false` 时，
  `drain_block_won()` 和 `has_pending_won()` 直接返回。

flag 一旦变为 `true` 就不再清零。不能只根据“当前 task 不是 joint”跳过，
因为较快 worker 可能已经发布后续 joint slot，而较慢 worker 仍在处理前一
task。当前方案利用所有 worker 回放相同 task stream 的条件：较慢 worker
到达自己的首个 joint task 时，会在首次 drain 之前打开轮询；此后保留原
协议，所以不会漏掉已发布或稍后发布的 won slot。

对纯单 lane 的 PA Case1，flag 始终为 `false`。因此删除的是：

- 每次 Submit 开头、归入 `EfDrain` 的 AIV `any_pub` load：
  `64 * 1280 = 81920` 次；
- kernel loser 在 `Replay.end` 到 `Submit.end` 尾段的 load：
  QK/PV 的 `512 * 64 = 32768` 次，加 SF/UP 的
  `512 * 63 = 32256` 次，共 65024 次；
- Submit 内合计 146944 次无效 `atomicAdd(addr, 0)`。

所以本次 atomic 优化直接缩短的是 `EfDrain` 和 loser 的
`Replay.end -> Submit.end` 尾段，不是 Claim。三轮 joint-skip 版本的全局
中位数为 5.171330 ms，相对基线下降 0.470915 ms（8.35%）。AIV 每核累计
`EfDrain` 中位数由约 0.765 ms 降至约 0.409 ms；Replay 后尾段也明显
缩短。同时 Claim 竞争上升，抵消了一部分收益。

### 4.2 复用参数 tag 扫描结果

该项不是 atomic 优化，目标是继续减少每核重复执行的 Submit 前端工作：

- `calculate_output_layout()` 第一次扫描 tensor tag 时同时生成
  `output_mask` 和 `register_mask`；
- materialize 用 `output_mask` 只访问 OUTPUT 参数；
- register 用保存在 `DistSubmitCtx` 中的 `register_mask` 只访问 INOUT 和
  OUTPUT_EXISTING 参数；
- 不改变参数 tag、tensor map 行为或对外接口。

只加入 register mask 的三轮中位数为 5.186679 ms，相比 joint-skip 的
5.171330 ms 没有稳定的全局收益；加入 output mask 后，最终三轮中位数为
5.115620 ms。最终版本的每核 `Materialize+Register` 累计中位数约为
AIC 1.416 ms、AIV 1.426 ms，基线分别约为 1.453 ms、1.451 ms。
设备未锁独占且每组只有三轮，因此 mask 的独立收益只能视为方向性证据，
不能按两组中位数差值作严格因果拆分。

### 4.3 性能结果和证据文件

| 版本 | 三轮首末 Submit（ms） | 中位数（ms） |
| ---- | --------------------- | -----------: |
| atomic baseline | 5.642245（稳定样本） | 5.642245 |
| joint polling skip | 5.171330 / 5.239468 / 5.167902 | 5.171330 |
| + register mask | 5.201349 / 5.128588 / 5.186679 | 5.186679 |
| + output/register masks | 5.115620 / 5.145057 / 5.096685 | **5.115620** |

最终中位数比基线快 9.33%，最好单轮比基线快 9.67%。最后一个 task 的
96-worker Submit 起点波宽从基线 463.913 us 降至最终三轮中位数
71.919 us，表明 worker 长尾明显收敛。1024 个 kernel span 的累计时长
从基线 32.518 ms 到最终三轮中位数 32.607 ms，未缩短，证明收益来自
Submit 调度而非 kernel 计算变快。

本地证据文件：

- 基线：`outputs/TestPagedAttentionUnroll_Case1_20260717_023809/merged_swimlane_atomic_load.json`；
- `ld_dev()+nop(100)`：`outputs/TestPagedAttentionUnroll_Case1_20260717_035341/merged_swimlane_nop100.json`；
- `ld_dev()+nop(10)`：`outputs/TestPagedAttentionUnroll_Case1_20260717_035954/merged_swimlane_nop10.json`；
- 最终最好一轮：`outputs/TestPagedAttentionUnroll_Case1_20260717_055638/merged_swimlane_best_joint_poll_skip_arg_masks_5.096685ms.json`；
- 对应原始记录均为相同目录下的 `l2_swimlane_records.json`。

`ld_dev()+nop(100)` 和 `ld_dev()+nop(10)` 的完整 Submit 样本分别为
5.343592 ms 和 5.401034 ms，但它们仅用于定位 atomic load 成本，已回退，
不能作为安全方案。普通 device load 加固定 NOP 不具备 atomic RMW 的同步
语义，也不能保证 cache 可见性或顺序；延迟长短不能修复协议正确性。

## 5. 后续优化顺序与验证要求

本节是首轮分析时给出的候选方向。实际推进采用“单变量、低风险到高风险”的
阶段门禁，并持续记录在第 7 节；若两处顺序不同，以第 7 节已经完成验证的结论为准。

建议按以下顺序继续，且每次只改一个变量：

1. **Claim cursor 竞争。** 73728 次 `atomicMax` 是最大固定项。可以研究
   确定性 owner、分层/分片 Claim 或减少无胜算 worker 参与，但必须保持
   AIC/AIV 负载均衡、ring slot 容量和 joint placement 语义。当前四分片是按
   task id 分片，不能假设再加 shard 一定能降低同一 task 的竞争。
2. **completion/frontier。** 研究减少重复 `atomicMax(frontier)` 和连续 flag
   扫描，例如单 advancer、批量前推或分层 frontier；最终判断必须保持“只有
   连续完成的 task 才能释放 heap”的约束。
3. **fanin ready 轮询。** 统计失败 load 次数，再判断是否能利用已知拓扑、
   本地完成缓存或更少的 drain 扫描。不能跳过 acquire-ready 条件后直接执行。
4. **heap guard。** 当前无 RingBp，优先级较低。可以缓存 frontier/vend 做
   advisory fast path，但真正覆盖 ring 前仍需可靠的共享状态确认。
5. **非 atomic 前端。** 继续观察 tensor map、PrepareMap、参数复制及每核约
   0.8 ms 的 inter-submit gap；这些不能由替换 atomic load 解决。

每个候选改动至少验证：

- PA Case1：A5Sim 正确性、真实 A5 正确性和 10 轮以上首末 Submit A/B；
- joint mixed（AIC+AIV）和 dual-AIV 的重复 task/slot 复用；
- heap/ring 压力、依赖未就绪、不同 kernel 时长和 worker 到达顺序；
- 96 核均有 1280 个 Submit，task id 完整为 0..1279；
- `DrainWon`、`RingBp`、kernel placement 和最终 drain 数量没有异常漂移；
- 同时报告中位数、离散度、最好/最差值，并记录设备是否独占。

泳道统计时，`Build` 和 `Replay` 是 lap marker，会覆盖前面的阶段，不能与
`Materialize/PrepareMap/Claim/Fanin/Register` 相加。可加总的是显式互斥
span，或单独定义 `Submit.end - Build/Replay/Alloc.end` 为 action tail。
后续文档和脚本都应沿用这一口径。

## 6. 独立 PA 调度复现的对应关系

`pa_scheduler/` 下的 CCEC、AscendC 和 CPU 用例以本文的真实 PA 路径为模型，
目标是脱离 simpler 的编译和链接依赖后，仍能单独研究 Submit 调度性能。这里的
“脱离 simpler”不表示删减 PA 调度逻辑。当前独立模型保留：

- Case1 的 256 batch、Alloc/QK/SF/PV/UP 五 task 拓扑和 AIC/AIV active mask；
- 96 worker 全量回放、四分片 Claim cursor 和固定 73728 次 Claim atomicMax；
- TaskArgs/Tensor/TaskPayload/DistSubmitCtx 的关键 ABI 和真实 tag 扫描；
- materialize、TensorMap retire/lookup/insert、register、fanin、slot payload；
- EfDrain、Replay、WaitForSlot、HeapGuard、flag/vend/frontier 和最终 drain；
- 单 lane Case1 的 BlockWon 动态次数为零，以及真实泳道记录格式。

只有 QK/SF/PV/UP 的计算体由可控 NOP 模拟；NOP 默认值按本文最好真实泳道的
44.170/53.729/27.626/1.565 us 校准。独立用例不会在 Claim、Register、
PrepareMap 或等待路径中增加 NOP 来硬凑 5 ms。

当前严格校验覆盖 73,728 次 Claim、每 task 唯一 winner、1,024 个 kernel、
TensorMap/heap 最终状态、fanin、flag、vend、frontier、cursor、ring placement
和每 worker 的前端操作次数。2026-07-17 三个 CCEC 独立进程首轮为
4.846431/4.798260/4.830184 ms，中位数 4.830184 ms；AscendC 独立进程首轮为
4.917014 ms，均为 PASS；真实最好泳道为 5.096685 ms。
差异主要来自真实 orchestration 与 `dist_submit_impl` 跨翻译单元，而 standalone
共享实现会和固定任务图一起被编译器优化。为制造编译边界而做的强制 noinline、
拆设备目标和全局 memory clobber 实验曾分别触发状态破坏、device exception 或
明显 RingBp，均已回退。

四阶段诊断通过 `--profile-phases` 输出：Claim 和 EfDrain 每 worker 调用
1280 次；WaitForSlot 由 1024 个 kernel winner 调用；HeapGuard 由每 batch 的
Alloc/QK/SF/PV winner 调用，共 1024 次。当前代表性 CCEC 轮次的累计中位数为：

| role | Claim | EfDrain | WaitForSlot | HeapGuard |
| ---- | ----: | ------: | ----------: | --------: |
| AIC | 470.503 us | 711.005 us | 234.063 us（全 AIC 43 次等待） | 21.349 us（约 1 次 heap 等待） |
| AIV | 533.755 us | 401.962 us | 0.067 us（无等待） | 3.723 us（约 1 次 heap 等待） |

完整构建、参数、内存占用、冷热运行口径和脱仓复制方法见同目录
[PA 调度器独立复现与泳道使用指南](PA调度器独立复现与泳道使用指南.md)。后续调度优化应先在该
独立用例做协议回归和阶段定位，再回到真实 PA Case1 做最终性能确认。

## 7. Atomic 消减阶段日志

从 2026-07-17 起，每个候选必须按以下节奏更新本节：先写静态证明和直接消减
口径，再记录 standalone 三后端结果、压力/反例结果和性能 A/B，最后写是否允许
进入真实 PA。失败、超时和回退同样保留，不能只记录正向数据。一次只验证一个
变量，后续阶段不得把前一阶段的间接调度变化冒充为本阶段的直接 atomic 收益。

| 阶段 | 单变量 | 当前状态 |
| ---- | ------ | -------- |
| H1 | HeapGuard 首圈 fast path | 正确性与真实性能完成，保留并本地提交 |
| F1 | fanin 依赖检查顺序 | H1 提交后开始 |
| 后续 | ready cache、退避、frontier、Claim | 尚未开始，必须逐项验证 |

### 7.1 阶段 H1：HeapGuard 首圈 fast path

#### 7.1.1 修改范围与正确性证明

本阶段只在 standalone 的 `HeapGuard()` 中增加以下判断；原 slow path 未改：

```cpp
while (!IsFatal<Ops>(state)) {
    if (worker.heap_next <= ring) {
        return true;
    }
    // 原 frontier/vend 容量检查。
}
```

判断放在 `IsFatal()` 之后，因此仍保留每次 HeapGuard 的 fatal 原子检查。profile
版本在 fast path 返回前仍累计 HeapGuard 时间，没有改变统计调用次数。

该判断依赖以下由现有代码确认的不变量：

1. 每个 worker 的 `heap_next` 从 0 开始，materialize 只按单调逻辑地址推进；
2. 只有生成物理输出地址时才对 ring 取模；
3. 每个输出按 1 KiB 对齐，单 task 输出超过 ring 会直接失败；
4. 当前 task 若跨越 ring 尾部，会先把 `task_base` 推到下一圈；
5. HeapGuard 在当前 task 完成 materialize 后执行。

所以在 `uint64_t` 未溢出的前提下，`heap_next <= ring` 表示所有已分配逻辑区间
仍位于 `[0, ring)`，取模后一一对应，尚不可能覆盖旧输出。`heap_next == ring`
也安全，因为已分配区间右端为开区间；第一个真正进入第二圈的非零输出会使
`heap_next > ring`。不能用 `output_bytes <= ring` 替代该条件，后者无法证明
历史分配没有 wrap。

本阶段不顺手修改 slow path 中 `heap_next - vend` 的无符号下溢边界，也不修改
frontier、fanin、Claim 或 completion 协议，避免把两个正确性问题混在一起。

#### 7.1.2 直接 atomic 消减口径

默认 256 batch 的 standalone 共调用 1024 次 HeapGuard。默认 256 MiB ring 下，
每个 worker 的最终状态为：

```text
heap_next = 206,569,472 B
ring      = 268,435,456 B
剩余       =  61,865,984 B
```

1024 次调用全部处于第一圈，直接消减如下：

| 操作 | 修改前 | 修改后 | 直接消减 |
| ---- | -----: | -----: | -------: |
| fatal atomic load | 1024 | 1024 | 0 |
| frontier atomic load | 1024 | 0 | 固定 1024 |
| vend atomic load | `V` | 0 | 动态 `V` |

只有 `retire = frontier - H >= 0` 时原逻辑才读取 vend。按 1280 task、`H=64`
和输出 task 分布，`0 <= V <= 972`。所以默认 workload 直接删除的是
`1024 + V` 次 HeapGuard 原子读取，不是所有观测指标的变化量。

#### 7.1.3 默认配置正确性门禁

恢复默认 256 MiB 配置前，本阶段已经完成一次 CCEC、AscendC、CPU 的全量构建，
以及 256 batch、零 NOP、开启四阶段统计、关闭泳道大缓冲区的完整运行：

```bash
./run.sh run all --device 0 --batches 256 --runs 1 \
  --nop-count 0 --profile-phases --no-swimlane
```

三个后端均为 `semantic_status=PASS`、`postprocess_status=PASS`。严格校验包括
73,728 次 Claim、1280 个唯一 winner、1024 个 kernel、1024 次 HeapGuard、
fanin、flag、vend、frontier、cursor、TensorMap/heap 最终状态和每 worker
前端操作次数。该结果只证明 standalone 协议，没有替代真实输出 golden。

#### 7.1.4 16 MiB wrap 压力与 CPU 对照

为覆盖首圈之外的原 slow path，曾临时把 `kHeapBytes` 从 256 MiB 改为 16 MiB
并全量重编。256 batch 的静态状态为：

```text
ring                  = 16,777,216 B
最终逻辑 heap_next    = 209,385,472 B
跨 ring 尾部跳转      = 12 次
首圈 fast path        = 82 次 HeapGuard
原 slow path          = 942 次 HeapGuard
```

CCEC 和 AscendC 的 256 batch 完整语义与阶段统计均 PASS。设备结果确实进入慢路径：
CCEC 的 HeapGuard wait event 为 `629 + 3 = 632`，AscendC 为
`542 + 6 = 548`，不是仍停留在首圈的伪压力测试。

CPU 结果需要单独解释：fast 版的 32、64、128 batch 均快速 PASS；256 batch
在开启阶段统计时持续运行超过 8 分钟未结束，关闭阶段统计后也未在观察窗口结束。
为排除 fast path 回归，保持 16 MiB 和 256 batch 不变，只临时撤掉 fast path、
重编 CPU 原版，并用统一的 120 秒门限运行；原版同样超时。由此只能得出：

- 该 CPU synthetic-heap 模型在 16 MiB、多圈、256 batch 下存在原有的长程活性
  或 host 线程调度问题；
- 当前证据不能把该超时归因于 H1，也不能把 CPU 256 batch 记为 PASS；
- 真实 tiny-ring 数据 golden 仍是生产迁移前不可省略的门禁。

对照完成后已恢复 `kHeapBytes = 256 MiB`，临时原版构建不保留为源码改动。

恢复后再次执行 `build all`，并用与 7.1.3 相同的 256 batch 命令完成终验；
CCEC、AscendC、CPU 再次全部 PASS。该轮 HeapGuard wait event 均为 0，符合默认
配置全程不 wrap 的静态结论。CCEC/AscendC/CPU 的 Submit span 分别为
3499.426 us、3523.687 us、229235.747 us；CPU 时间仅反映 host pthread 调度，
不参与 A5 性能比较。

#### 7.1.5 CCEC A5 standalone 十样本 A/B

baseline 和 H1 均使用独立进程首轮，各保留 10 个具有完整 PASS 结果的有效样本；
单位为微秒：

```text
baseline = [
  4451.574, 4847.797, 5089.239, 5403.477, 5227.148,
  5217.446, 4929.629, 4084.562, 5268.516, 5246.626
]

H1 = [
  3759.041, 4142.396, 4582.740, 4134.989, 5670.639,
  4348.529, 4503.739, 5507.320, 4286.019, 4088.046
]
```

| 指标 | baseline（us） | H1（us） | 相对变化 |
| ---- | ------------: | -------: | -------: |
| 样本数 | 10 | 10 | - |
| 中位数 | 5153.3425 | 4317.2740 | -16.2238% |
| 均值 | 4976.6014 | 4502.3458 | -9.5297% |
| p90 | 5268.516 | 5507.320 | +4.5327% |
| 最小值 | 4084.562 | 3759.041 | - |
| 最大值 | 5403.477 | 5670.639 | - |

九组可按采集顺序配对的样本中 H1 有 7 组更快，配对变化中位数为 `-14.55%`，
两组反向样本为 `+8.48%` 和 `+34.83%`。动态指标也随调度时序变化：fanin
load 中位数从 91,254 降至 66,318.5（-27.33%），但 p90 上升 11.47%；
RingBp placement 中位数从 118.5 降至 101（-14.77%），p90 下降 3.76%。

另有两次没有生成完整结果的 baseline 异常：首次外层 launch 约四分钟无结果后
人工中断；后续 baseline `r9` 在 60 秒门限超时。二者均未计入耗时分布，也不能
记为 PASS 或 FAIL；异常后设备 smoke 恢复正常。正式 A/B 必须让两边使用相同
timeout，并把 timeout 率单独报告，避免只统计成功样本造成幸存者偏差。

fanin 和 RingBp 代码并未在 H1 修改，其变化幅度又远大于固定 1024 次 frontier
读取，因此只能解释为 worker 到达、依赖完成和 drain 相对时序改变后的间接效应。
当前中心趋势显示方向性收益，但 H1 的 p90 变差且离散较大，不能声称尾延迟稳定
改善，更不能把 16.22% 全部归因于 `1024 + V` 次直接 atomic 消减。

#### 7.1.6 阶段结论与下一门禁

H1 当前状态是“standalone 默认配置通过，允许进入真实 PA 正确性验证”，不是
“真实 PA 已优化完成”。继续推进前按以下顺序执行：

1. 默认 256 MiB 恢复后重建三后端并再次执行完整回归，确认临时压力常量无残留；
2. 对真实 `dist_submit_wait_heap_capacity()` 做一次单点、等价修改；
3. 运行 fully-distributed PA Case1 的 A5Sim golden；
4. 运行已有 68 KiB `AllocFillRunAhead`、`AllocHeapBackPressure` 和 MB6 heap
   reclaim 用例，验证真实输出没有 premature reuse；
5. A5 正确性通过后，再做双方相同门限、独立进程、交错顺序的至少 10 轮 A/B；
6. H1 通过真实门禁后，下一项低风险候选是只调整 fanin 检查顺序；ready cache、
   backoff、frontier 合并和 Claim 参与者缩减必须分别作为后续单变量阶段。

standalone 使用 synthetic heap 且计算 kernel 由 NOP 模拟，它能验证控制协议、
分支和 atomic 计数，但不能用 synthetic 地址证明真实 tensor 内容未被提前覆盖。

#### 7.1.7 真实 fully-distributed 迁移与正确性结果

standalone 门禁通过后，只在真实实现的
`dist_submit_wait_heap_capacity()` 中加入同形判断：

```cpp
while (!fatal_set()) {
    if (ctx.self->heap_next <= ring) return true;
    // 原 slow path 不变。
}
```

生产实现没有 standalone 的 HeapGuard phase profile，因此没有搬入额外统计字段。
判断仍位于 `TRACE_SPAN_BEGIN` 之后和 fatal 检查之内：原本不等待的成功路径就不会
生成 RingBp 事件，新 fast path 的 trace 语义不变。真实函数共被 1280 个 winner
调用，UP 因 `output_bytes == 0` 在进入 atomic-bearing loop 前返回；实际消减口径
仍是 Alloc/QK/SF/PV 的 1024 次 guard。

测试统一使用用户环境 `/home/q00473782/.venv`、CANN 9.1、本地 GCC 15，并按
仓库 CI 固定 PTO-ISA 到：

```text
ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8
```

未传 SHA 时测试会把 managed clone 更新到当时的 `origin/main ea90d400`，该组合的
CPU stub/`pto_instr.hpp` 出现大量重复定义，kernel 未能编译；这不是 H1 结果，
也没有通过修改 PTO-ISA 头文件规避。非交互 shell 还必须显式加入用户本地
`g++-15` 路径，不能假设会自动 source `.bashrc`。

真实数据门禁结果如下：

| 平台 | 用例 | 覆盖点 | 结果 |
| ---- | ---- | ------ | ---- |
| A5Sim | `AllocFillRunAhead67`，68 KiB | 首圈边界 | PASS |
| A5Sim | `AllocFillRunAhead128`，68 KiB | 跨圈 slow path | PASS |
| A5Sim | `AllocHeapBackPressure`，68 KiB | 等待、回收、真实数据 golden | PASS |
| A5Sim | PA Case1，真实 kernel | 256 batch 数值 golden | PASS |
| A5 | `AllocFillRunAhead67`，68 KiB | 首圈边界 | PASS |
| A5 | `AllocFillRunAhead128`，68 KiB | 跨圈 slow path | PASS |
| A5 | `AllocHeapBackPressure`，68 KiB | 等待、回收、真实数据 golden | PASS |
| A5 | PA Case1，真实 kernel | 256 batch 数值 golden | PASS |

PA A5Sim 的真实 kernel orchestration 约为 38.824 s。另一次带
`--use-example-exec-time` 的 A5Sim 调度运行也 PASS，但该选项会跳过数值 golden，
所以只作为控制流证据，不计入上表的正确性依据。

MB6 需要保留两个基线问题，不能写成 PASS：

1. `Normal` 已完成底层数值运行，但 `DistRuntimeContractMixin` 随后因没有捕获到
   `[dist] DEPSIG` 而失败；当前 `src/`、`simpler_setup/` 和 runtime 构建源码中
   查不到 `PTO_DIST_DEPSIG/DEPSIG` 实现，因此属于测试契约与当前 runtime 不匹配，
   不能靠加 shell 环境变量伪造 oracle。
2. `Heavy` 的 8 MiB/H=64 A5Sim 压力在运行中触发 native abort。保持全部参数
   不变、只撤销生产 H1 后，基线同样以相同调用路径 abort；所以该失败不是 H1
   引入，但也不能作为 H1 的通过项。`FullCore36` 在 Heavy 基线已经失败后未继续跑。

综合现有证据，H1 已通过目标 PA 与真实 68 KiB 回卷/回收的 A5Sim+A5 golden，
可以进入真实 PA 性能 A/B。MB6 的两项既有问题作为未关闭风险保留，不能被其他
PASS 掩盖，也不在 H1 中顺手修改测试框架或 heap slow path。

#### 7.1.8 真实 A5 PA Case1 十对性能 A/B

真实性能使用提交前基线 `290dbda0` 的 detached 临时 worktree 和带 H1 的主
worktree。两边均预先完成一次不计入统计的 warm run，随后按
`baseline -> H1` 顺序采集 10 对独立 pytest 进程。统一条件为：

- A5 device 0，真实 PA kernel，256 batch；
- 用户 `/home/q00473782/.venv`、CANN 9.1、GCC 15；
- PTO-ISA 固定为 `ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8`；
- 开启 L2 swimlane，不使用 `--use-example-exec-time` 或 `--skip-golden`；
- 每个进程统一 180 s timeout；双方均 10/10 PASS、0 timeout；
- 为避免自动生成数百 MiB merged JSON 影响采集周转，双方仅在测试进程内对称地
  关闭 post-case converter；设备执行和原始 `l2_swimlane_records.json` 不变；
- 每个原始文件均有 122,880 个 Submit 事件，指标直接取最早 start 到最晚 end。

预热样本为 baseline 5.134950 ms、H1 5.116055 ms，只验证两边构建和设备状态，
不进入下面统计。10 对正式样本为：

| 对次 | baseline（ms） | H1（ms） | 配对变化 |
| ---: | ------------: | -------: | -------: |
| 1 | 5.149955 | 5.301485 | +2.942% |
| 2 | 5.331087 | 5.110060 | -4.146% |
| 3 | 5.137861 | 5.098696 | -0.762% |
| 4 | 5.150200 | 5.119599 | -0.594% |
| 5 | 5.129350 | 5.112860 | -0.321% |
| 6 | 5.146475 | 5.129671 | -0.327% |
| 7 | 5.111929 | 5.105638 | -0.123% |
| 8 | 5.274224 | 5.229773 | -0.843% |
| 9 | 5.114477 | 5.137014 | +0.441% |
| 10 | 5.125086 | 5.125040 | -0.001% |

| 统计 | baseline（ms） | H1（ms） | 相对变化 |
| ---- | ------------: | -------: | -------: |
| 中位数 | 5.142168 | 5.122320 | -0.386% |
| 均值 | 5.167064 | 5.146984 | -0.389% |
| nearest-rank p90 | 5.274224 | 5.229773 | -0.843% |
| 最小值 | 5.111929 | 5.098696 | - |
| 最大值 | 5.331087 | 5.301485 | - |
| 样本标准差 | 0.073960 | 0.065764 | - |

H1 在 10 对中 8 胜 2 负；配对变化中位数为 -0.324%，均值为 -0.373%。第 1、2
对分别有 +2.942% 和 -4.146% 的反向大波动，且采集顺序固定为 baseline 在前，
因此这 10 对只能支持“小幅方向性收益”，不能声称统计上已经稳定到每轮必胜。
与 standalone 的 -16.22% 中位数不同，真实 PA 的约 0.3%～0.4% 中心改善更符合
固定删除少量 HeapGuard atomic 的规模；standalone 的巨大间接调度变化不应外推。

最佳 H1 样本为第 3 对的 5.098696 ms，单独生成的泳道文件为：

```text
outputs/TestPagedAttentionUnroll_Case1_20260717_173313/
merged_swimlane_heapguard_first_lap_fastpath_5.098696ms.json
```

原始 A/B 文件分别位于临时 baseline worktree 和主 worktree 的对应 timestamp
目录。自动 converter 被关闭的正式样本仍保留 raw JSON；上述最佳样本在统计完成后
单独补做了 converter，未重新执行 device workload。

#### 7.1.9 H1 最终阶段决定

H1 满足保留条件：正确性证明成立，standalone 三后端默认配置 PASS，真实
A5Sim/A5 的 68 KiB 首圈/跨圈/回收 golden 与 PA Case1 golden 全部 PASS，真实
A5 十对中心趋势和 p90 均未回退，并且直接 atomic 数量确定下降。因此将 standalone
与生产同形改动、测试证据文档作为一个本地 commit 保存，不 push。

真实性能收益较小且仍有单轮反向，不把 H1 宣传成大幅优化。下一阶段 F1 只研究
fanin 依赖检查顺序；不得同时加入 ready cache 或退避，避免失去因果归属。
