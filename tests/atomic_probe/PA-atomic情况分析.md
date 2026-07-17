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
[A5 FDWIC Paged Attention 安装与复现指南](A5_FDWIC_PAGED_ATTENTION_REPRO.md)。

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
