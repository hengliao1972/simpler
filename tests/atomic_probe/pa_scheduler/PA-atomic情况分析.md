# A5 FDWIC Paged Attention Submit 原子操作与优化记录

## 1. 范围与当前结论

本文记录 `TestPagedAttentionUnroll::Case1` 在真实 A5 上的 FDWIC AICore
Submit 路径，供后续继续优化。快照日期更新至 2026-07-18；当前保留的生产
优化基线为 `2c3dd1e2`，F1 负结果记录提交为 `c93c3666`。文档中
standalone 观察链路的最新状态更新至 2026-07-19。

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
- H1 又消除了默认 256 MiB heap 第一圈的 1024 次 frontier atomic load，
  十对真实 A5 的配对变化中位数为 -0.324%，最好单轮为 5.098696 ms；
- 优化没有改变通用 atomic 语义，也没有把任务推迟到最终 drain 来制造
  表面收益。F1 的 fanin 顺序重排已经证明性能回退并撤销；下一步先精确区分
  fanin 成功/失败 load 与 frontier 重复前推，再进行单变量消减。
- standalone 观察产物现已固定为两类：`swimlane` 使用 schema-v4 合并
  排他业务阶段与 atomic（direct Atomic 加 PollBatch）泳道；atomic flags、
  weighted call 和 PollBatch ABI 沿用已验证的 schema-v3 语义。`submit-pmu`
  独立重编译完整 Submit PMU，现行
  白名单为 `none|claim|efdrain|materialize|register`。两者不在同一进程采集；
  `none` 提供完整 Submit 的严格闭合计数，局部 phase 只提供 running
  read-clear 的下界/保守上界。

环境安装、编译和基线复现过程见
[A5 FDWIC Paged Attention 安装与复现指南](../a5_fdwic_atomic_swimlane_repo.md)。
standalone I-cache 的当前构建、采集和解读见
[`../icache_miss_usage_guide.md`](../icache_miss_usage_guide.md)。

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

H1 后最好一轮没有 `RingBp` 事件，且默认 256 MiB heap 的 1024 次有效调用都
命中第一圈 fast path：

- `fatal_set()`：固定 1024 次 `atomic_load(g_dist.fatal)`；
- frontier：第一圈为 0 次；只有逻辑 heap 超过一整圈后才恢复原
  `atomic_load(g_dist.frontier)`；
- `load_task_vend(F-H)`：第一圈为 0 次；跨圈后仍按原 frontier/vend 容量
  规则执行；
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
| 后续 `task.flag` ready | `atomic_load` | 若前推次数为 A，则准确为 A+1280 | >=2560 |
| `frontier` 前推 | `atomic_fetch_max` | 记为 A，竞争时可重复 | >=1280 |

原因是每次 `advance_frontier()` 都以一次未就绪 flag load 结束；每次成功
`atomic_fetch_max(frontier)` 前又恰有一次 ready flag load。因此完整成功运行中，
该处 flag load 不是笼统的“至少 1280”，而是准确的 `A + 1280`。

若把 fanin ready load 记为 `G`，H1 后完整 Submit + completion 路径的 atomic
总数为：

~~~text
73728 Claim + 1024 HeapGuard fatal + G fanin
+ 1280 vend exchange + 1280 flag exchange + 1280 frontier initial load
+ A frontier atomicMax + (A + 1280) frontier flag load
= 79872 + G + 2A
~~~

由于 `G>=1280`、`A>=1280`，硬下界为 83712 次。`G` 和 `A` 受真实调度时序
影响，不能用静态下界代替动态样本。

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
下界当作整次运行的 atomic 总数。若需要精确动态计数，应使用 worker-local 软件
计数或独立诊断构建；当前 PMU 事件不直接给出 atomic 条数。直接在热路径增加共享
计数器会反过来改变竞争形态。

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

真实 PA 与历史 standalone schema-v3 泳道中的 `Build` 和 `Replay` 是 lap
marker，会覆盖前面的阶段，不能与
`Materialize/PrepareMap/Claim/Fanin/Register` 相加。当前 standalone
schema-v4 已改为显式互斥的 `WinnerBuild/LoserReplay/AllocComplete` 尾 span；
后续文档和脚本必须先按 capture schema 选择口径，不能把历史 lap 与 v4 尾 span
混为一类。

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

Case1 的 standalone 依赖图由 tensor owner 与每 worker TensorMap 的 lookup 共同
收集，不是按 task 名字直接跳过依赖：Alloc 和 QK 的 fanin 均为 0；SF 依赖 QK；
PV 依赖 SF；UP 的多个输入/原地输出按 producer 去重后依赖 Alloc、SF、PV。
所以每 batch 的 fanin 数是 `0+0+1+1+3=5`，默认 256 batch 的严格终态断言
要求 `fanin_edges=1280`。EfDrain 只有在这些 producer completion flag 全部
ready 后才执行 winner 负载。

该图只对等 Case1 的调度依赖。真计算 workspace 用统一的受控 A/B 输入分别验证
QK/PV matmul、SF add 和 UP mul，并按 worker-kind 写独占输出 tile；这些数值输出
没有按 QK→SF→PV→UP 串接成 PA 的真实 tensor 数据流。因此它能验证 fanin、完成
发布、角色路由和各类引擎算术，不能验证后继 task 消费前驱真实数值的语义。
当前也只覆盖 Case1 的单 block group、`q_loop=1` 和全单-lane 图；通用多 group、
多 q-loop、跨迭代更新及 joint/mixed task 的依赖数量与竞争形态均未模拟。

当前三后端无参数默认使用 `real-compute`：CCEC/AscendC 的 QK/PV 运行真实 Cube
matmul，SF/UP 运行真实 Vector add/mul，每轮都含 GM load、计算、GM store 和完成
等待；CPU 使用相同 workspace、角色路由和数学运算做协议回归，不代表 A5 引擎
时间或 PMU。可控 `scalar-nop` 只作为显式兼容/校准模式保留，其历史默认值按
本文最好真实泳道的 44.170/53.729/27.626/1.565 us 校准。两种模式都不会在
Claim、Register、PrepareMap 或等待路径中硬补 5 ms。

当前真计算默认次数为 QK/SF/PV/UP=`6,28,4,1`。依据是其每 task 完整
load/compute/store span 已接近真实 PA 的 44.170/53.729/27.626/1.565 us；它
优先保持 per-task core work 口径，不通过增加无关 repeat 把 standalone 总时间
硬凑到 5.1 ms。standalone 未覆盖的真实控制流、代码布局和资源竞争仍会形成
端到端差值，不能把这部分差值反推成缺少的 kernel repeat。

当前严格校验覆盖 73,728 次 Claim、每 task 唯一 winner、1,024 个 kernel、
TensorMap/heap 最终状态、fanin、flag、vend、frontier、cursor、ring placement
和每 worker 的前端操作次数。2026-07-17 三个 CCEC 独立进程首轮为
4.846431/4.798260/4.830184 ms，中位数 4.830184 ms；AscendC 独立进程首轮为
4.917014 ms，均为 PASS；真实最好泳道为 5.096685 ms。
差异主要来自真实 orchestration 与 `dist_submit_impl` 跨翻译单元，而 standalone
共享实现会和固定任务图一起被编译器优化。为制造编译边界而做的强制 noinline、
拆设备目标和全局 memory clobber 实验曾分别触发状态破坏、device exception 或
明显 RingBp，均已回退。

下列四阶段是普通 runtime `--profile-phases` 的历史诊断口径，不是第 7.5 节的
`submit-pmu` 局部白名单；调度器中的 WaitForSlot 协议和普通泳道仍保留。
Claim 和 EfDrain 每 worker 调用
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
| F1 | fanin 依赖检查顺序 | standalone 性能未改善，已回退 |
| 后续 | ready cache、退避、frontier、Claim | 按风险从低到高逐项验证 |

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

### 7.2 阶段 F1：fanin 依赖检查顺序

#### 7.2.1 候选改动与静态分析

F1 只在 standalone `CollectFanin()` 完成原有去重和 16 项截断后，将有效
fanin 前缀按 producer task id 降序排列。排序放在截断之后，所以不改变
原实现选中的依赖集合、`kMaxFanin=16` 语义或 slot ABI。fanin ready 是 AND
条件，仅改变检查顺序不改变“所有 producer 都完成后才可执行”的结果。

PA 中只有 UP 包含 3 个 fanin，原顺序是 `[SF, PV, Alloc]`，降序后为
`[PV, SF, Alloc]`。PV 依赖 SF，因此在其他调度状态不变时：

- SF、PV 均未完成：两种顺序都在第 1 次 ready load 后返回；
- SF 已完成、PV 未完成：原顺序读 2 次，新顺序读 1 次；
- PV 已完成：由依赖关系可知 SF 已完成，两种顺序都成功读完 3 项。

所以在固定调度状态下，UP 每次失败检查的 ready atomic load 不会增加，
且在 SF 已完成而 PV 尚未完成的窗口可减少 1 次。QK 的 fanin 为 0，
SF/PV 各为 1，本改动对它们不产生重排。不过，排序本身会改变标量指令和
worker 到达时序；真实动态调度不能由上述固定状态推导出必然性能收益。

#### 7.2.2 standalone 正确性与完整运行

F1 修改后重建 CCEC、AscendC 和 CPU 三后端，并分别运行 smoke、256 batch 零
NOP、256 batch 当时默认 NOP。全部结果的语义断言和后处理都为 PASS；完整用例
仍为 73,728 次 Claim、1,280 个唯一 winner、1,024 个 kernel，placement 总数为
1,024。关键单轮结果如下：

| 场景 | CCEC `fanin / Submit us` | AscendC | CPU |
| ---- | -----------------------: | ------: | --: |
| smoke | `6 / 34.912` | `5 / 40.142` | `6 / 156383.064` |
| 256 batch、零 NOP | `42546 / 3415.839` | `25515 / 3498.038` | `1983 / 163840.941` |
| 256 batch、当时默认 NOP | `40958 / 4019.835` | `47692 / 4283.306` | `1051412 / 185456.323` |

CPU 只用于语义对照，不作为 A5 性能依据。单轮 fanin load 对 worker 调度非常
敏感，不能用上表三个数值直接归因，因此又执行了独立进程的交错 A/B。

#### 7.2.3 CCEC 十对交错 A/B

基线固定在 H1 提交 `2c3dd1e2`，使用 detached worktree
`/tmp/simpler-f1-baseline`。两端都用 256 batch、当时默认 NOP、关闭泳道和 phase profile；
每个样本是独立进程首轮，统一 60 s timeout。前 5 对为 baseline -> F1，
后 5 对为 F1 -> baseline。双方均 10/10 PASS、0 timeout、Claim=73,728、CAS retry=0。

| 指标 | baseline | F1 | 相对变化 |
| ---- | -------: | -: | -------: |
| Submit 中位数（us） | 4290.401 | 4623.944 | **+7.774%** |
| Submit 均值（us） | 4498.804 | 4684.302 | **+4.123%** |
| Submit nearest-rank p90（us） | 5303.373 | 5219.956 | -1.573% |
| Submit 样本标准差（us） | 944.376 | 475.794 | - |
| fanin load 中位数 | 67914 | 65433.5 | -3.652% |
| fanin load 均值 | 82844.5 | 72219.2 | -12.826% |
| fanin load nearest-rank p90 | 118955 | 94088 | -20.905% |

F1 的 Submit 为 4 胜 6 负；配对相对变化中位数为 `+6.439%`，均值为
`+8.410%`。交换顺序后，后 5 对的配对变化中位数仍为 `+3.423%`，没有把
中心趋势变成收益。fanin load 的配对变化中位数为 `-13.582%`，但均值为
`+0.665%`，单对范围从 `-58.082%` 到 `+89.165%`，说明其仍强烈受动态调度影响。

F1 的 p90 略好主要由 baseline 的 5.303 ms 和 6.640 ms 慢样本抬高；在中位数、
均值和配对中心全部回退时，不能单独用这个 p90 宣称收益。原始日志保留在：

```text
/tmp/f1_standalone_ab_20260718_010300/
```

#### 7.2.4 阶段决定

F1 在固定状态下的 atomic load 不增证明成立，三后端语义也全部通过；但
standalone 的 Submit 中位数、均值和配对中心都没有改善。因此不将这个启发式
重排迁移到真实 `dist_submit_collect_fanin()`，也不进行真实 PA A/B。standalone
候选代码已撤回，本节保留负结果，防止后续重复同一实验。

### 7.3 阶段 O1：建立 CCEC 每核 scalar/PIPE_UTIL PMU 观察链路

#### 7.3.1 直接 PMU owner 是唯一正式取数链路

整任务级 raw counter 无法由 kernel 内局部 gate 缩成 Submit 子窗口，因此不能作为
Claim、EfDrain、Materialize、Register 的局部取数依据。当前正式链路不消费
这类整任务汇总：CCEC host 使用本目录自带的 Main AICPU Path-A owner 保存、配置、
读回并最终恢复 PMU 状态；kernel 在同一 runtime TU 内完成门控、读取和发布。

owner 的 Path-A loader、dispatcher 和 `simpler_aicpu_exec` 均随 standalone CCEC
构建，不依赖父目录探针。host 复用 A5 runtime 的
`halResMap(PROCESS_CP1, RES_AICORE)` 布局，将 36 个 AICore 展开为 108 个物理子核
MMIO base；kernel 用真实 `get_coreid()` 索引。观察链路逐核核对并读取
`CNT0..CNT8`，最后一次性把 `CNT_TOTAL` 和九个 programmable counter 写入每 worker
独占结果区。

这里没有臆测事件编号。正式依据来自仓内 A5 platform 实现：

- `src/a5/platform/include/common/platform_config.h` 给出 `CNT0..CNT8` 的 MMIO
  offset `0x4210..0x4250`、selector offset `0x2500..0x2520`，以及
  `CNT_TOTAL` 的低/高 32 bit offset `0x4260/0x4264`；
- `src/a5/platform/include/common/pmu_profiling.h` 的
  `PMU_EVENTS_A5_PIPE_UTIL` 给出以下 event id 与正式名称；
- standalone `ccec/kernel.cpp` 在冻结 gate 后逐项读取 selector，只有九项都与
  下表相等、物理子核映射有效且 total 非零时，host 才把该 worker 标为 trusted。

| counter | event id | 仓内正式名称 | 本文简写 |
| ------- | -------: | ------------ | -------- |
| CNT0 | `0x501` | `pmu_idc_aic_vec_busy_o` | vector busy |
| CNT1 | `0x301` | `cube_instr_busy` | cube busy |
| CNT2 | `0x001` | `scalar_instr_busy` | scalar busy |
| CNT3 | `0x701` | `mte1_instr_busy` | MTE1 busy |
| CNT4 | `0x202` | `mte2_instr_busy` | MTE2 busy |
| CNT5 | `0x203` | `mte3_instr_busy` | MTE3 busy |
| CNT6 | `0x034` | `icache_req` | I-cache request |
| CNT7 | `0x035` | `icache_miss` | I-cache miss |
| CNT8 | `0x714` | `pmu_fix_instr_busy` | fix busy |

`CNT0..CNT8` 是 32 bit programmable counter，`CNT_TOTAL` 是由低/高两项组成的
64 bit raw counter。正式门禁要求本轮最大 programmable counter 小于
`UINT32_MAX/4`（25% 高水位），并报告剩余 headroom。这是缩短窗口后采用的保守
风险阈值，只能降低未察觉回卷的风险；硬件没有随样本提供 wrap 次数，最终值即使
远低于门槛，也不能证明此前没有恰好回卷一圈或多圈。

这套 PMU 事件不直接给出 atomic 操作条数。atomic 条数继续由源码不变量和
worker-local 软件计数精确核对；PMU 用于观察这些 atomic 及周边 scalar 指令造成
的周期、I-cache request/miss 和竞争时序变化，两种证据不能相互冒充。

#### 7.3.2 A5 动态验证结果

2026-07-18 已用 standalone 自包含 owner 在真实 A5 上完成三种校准模式和一个
`submit-all` 小样本。每种校准模式本轮各取一个样本，均为 96/96 trusted、物理核 id
唯一、32 AIC + 64 AIV、32 个完整 `1 AIC + 2 AIV` triplet，且 owner Restore PASS：

| 窗口 | 96 核 PMU raw total 中位数 | 本轮可支持的结论 |
| ---- | -------------------------: | ---------------- |
| empty | 约 214 | 空 gate 路径能够闭环 |
| scalar 100,000 | 约 56,568 | scalar NOP 对 total/事件产生明确正向响应 |
| scalar-double 2×100,000 | 约 112,994 | 双段结果接近单段两倍，暂停后可继续累计 |

这三个数是单次上板验收样本，不是多轮 A/A 稳定性统计，也不用于宣称某个 PMU raw
count 等于同数值的硬件 cycle 或纳秒。

同一版本还完成 `batches=1,nop-count=0,submit-all` 单次闭环：96 个 worker 都有
实际 start/stop，九项 selector、owner bitmap membership、worker slot、物理 role、
triplet 和 Restore 均通过。all/AIC/AIV 的 total 中位数约为
36,066.5/28,708/39,745；96 核 scalar busy 求和约 2,661,612，I-cache
request/miss 求和为 210,399/30,283，按总和计算的 miss rate 约 14.3931%；host
`submit_span_us` 约 47.770。该样本只证明观察闭环可运行，不能替代 256 batch
约 5 ms 基线，也不能作为性能优化收益。

同一源码和当时默认 PA NOP 随后完成了 3 个独立进程的
`batches=256,submit-all,PMU-only`。三轮均为 96/96 trusted，owner/slot/role/
triplet/start/stop 门禁全部 PASS，Restore PASS，且原始 96 核记录重算
ALL/AIC/AIV 的 `sum/mean/median/p95/max` 与 JSON summary 完全一致：

| 独立进程 | Submit span | ALL/AIC/AIV total 中位数 | scalar busy 总和 | I-cache request/miss 总和 | miss rate |
| -------- | ----------: | ---------------------------: | -------------------: | -----------------------------: | --------: |
| run1 | 3,688.236 us | 5,812,276 / 5,546,285 / 5,835,567 | 448,463,772 | 69,812,583 / 5,854,421 | 8.3859% |
| run2 | 4,089.057 us | 6,624,292 / 6,365,728 / 6,640,326 | 523,142,035 | 69,451,706 / 5,847,256 | 8.4192% |
| run3 | 4,673.237 us | 7,251,930.5 / 7,230,620 / 7,272,650 | 594,774,017 | 70,065,443 / 5,830,645 | 8.3217% |

三轮 Submit span 中位数为 4,089.057 us，仍处于 standalone 的毫秒级调度区间，
但不能与无 PMU 三轮的 4,904.346 us 做非配对单样本减法：`PIPE_ALL`
门控会改变多核到达与争用时序。本三轮内，I-cache request/miss 总和及
miss rate 比 total/scalar busy 更稳定；AIC/AIV 分组 miss rate 的三轮中位数
分别约为 3.1138% 和 12.0556%。这只是当前同配置三轮的可重复现象，
尚不能说明 miss 的来源，也不能直接换算成精确 stall 时间；可按 7.3.3 的
90 ns/miss 标尺做一阶等效数量级估算。三轮最大 programmable
counter 分别为 5,559,659、6,021,532 和 7,082,609，均低于 25% 风险门槛；
这仍不构成“已证明未回卷”。本机未入库证据文件位于：

~~~text
outputs/scalar_observation_final_20260718/pmu_submit_all_ccec_b256_v2/pmu_submit_all.json
outputs/scalar_observation_final_20260718/pmu_submit_all_ccec_b256_v2_run2/pmu_submit_all.json
outputs/scalar_observation_final_20260718/pmu_submit_all_ccec_b256_v2_run3/pmu_submit_all.json
~~~

#### 7.3.3 单次 CNT7 I-cache miss 的时间标尺

此前 100000 NOP 的 aggregate residual 只能证明 I-cache 事件会响应，不能把一段
热循环里的多次 request/miss 直接除成“单次 miss 延迟”。为给 scalar 性能分析
建立与“单次 atomic 约 160 ns”同样直观的数量级，CCEC 新增
`--pmu-window icache-single` 配对校准：目标函数只有 8 B 并按 128 B I-cache line
对齐；每个 cold trial 先在窗口外执行 64 KiB 指令 capacity sweep，warm trial 则在
PMU read-clear 前额外调用一次相同目标。1 GHz sys counter 只包住最终目标调用，
两条路径的 PMU gate、harness 和目标符号完全相同。

64 trials/core × 10 轮以及 128 trials/core × 5 轮均满足：

~~~text
cold CNT7 miss == trials
warm CNT7 miss == 0
miss_delta == 96 * trials
calibrated_cores == 96/96
~~~

按每轮 `sum(cold_ticks-warm_ticks) / sum(cold_miss-warm_miss)` 计算，
完整分布为：

| 规模 | ALL median（range） | AIC median（range） | AIV median（range） |
| ---- | ------------------------: | ------------------------: | ------------------------: |
| 64 trials/core × 10 | 86.596（86.532～86.792）ns/miss | 85.913（85.848～86.202）ns/miss | 86.938（86.861～87.086）ns/miss |
| 128 trials/core × 5 | 89.629（89.615～89.648）ns/miss | 92.100（91.984～92.267）ns/miss | 88.410（88.310～88.440）ns/miss |

总体系数观测范围为 86.532～89.648 ns/miss；本表 64 和 128 trials 的中位数
分别为 86.596 和 89.629 ns/miss。AIC/AIV 差值的方向在两组规模间改变，不建立
跨时段的角色精确常数。几 ns 的变化属于并发环境下的有效 miss penalty 波动，
不应保留成伪精确常数；后续统一取 90 只用于量级估算：

~~~text
T_icache_est_ns = CNT7_miss_total * 90
T_icache_est_us = CNT7_miss_total * 0.09
~~~

即 1,000 个 miss 约解释 90 us，10,000 个约解释 0.9 ms；单次 I-cache miss 的
量级约为当前 160 ns atomic 标尺的 56%。如果结果已经按角色拆开，也可以使用当次
探针打印的 AIC/AIV 系数分别计算后相加；只有总 miss 时统一乘 90 ns。

compulsory、capacity、conflict miss 都会进入 `CNT7` 总数，PMU 本身不提供原因
分类，所以上式三类全算。本探针刻意制造的是 capacity eviction，测得的是 96 核
并发、cold 相对 warm 的一阶等效时间。真实 scalar 路径可能存在预取、多个 miss
重叠、不同下级命中位置或排队，因此乘积用于直观归因和数量级判断，不能宣称为
逐次精确、完全可加的 stall 时间。

本机未入库的原始日志为：

~~~text
tests/atomic_probe/pa_scheduler/outputs/pmu_validation/
  icache_single_64x10_20260718_085929_3232836_console.log
  icache_single_128x5_20260718_090151_3235468_console.log
~~~

#### 7.3.4 阶段决定与后续使用边界

直接 owner、每核 selector/MMIO、实际 gate 状态、96 核拓扑和 Restore 已达到继续
建设观察链路的正确性门槛。正式 Submit PMU 只保留一个窗口：`submit-all`。它从
每 worker 的 orchestration 初始化前开始，在该 worker 最后一次 Submit 返回后停止；
不再通过热路径内反复 stop/start 尝试扣除模拟计算体。

`metrics_prof_start/stop()` 自带 `PIPE_ALL` 屏障，门控会收口流水并可能改变多核
到达时序。A/B 两侧必须保持相同 gate 次数、winner mode/count 和代码布局，端到端
收益仍由无 PMU、无泳道的独立进程交错 A/B 判断。`scalar-nop` 的 QK/SF/PV/UP
循环由 scalar 执行并计入 scalar busy；CCEC `real-compute` 则真实激活
Cube/Vector/MTE/FIX。两种模式不能混比，且两者 sidecar 都不能直接替代真实 PA。

### 7.4 阶段 D1：精确分类 fanin 与 frontier 动态原子次数

#### 7.4.1 计数实现与闭合关系

D1 已在 standalone 公共调度器中增加 worker-local 软件计数，不新增共享 atomic：

- fanin 每次 flag load 只递增一个分类：`ready` 或 `not_ready`；
- 每次 completion 递增一个 frontier initial load；
- ready flag 与紧随其后的 FetchMax 共用一个一一对应计数 `A`；
- 扫描遇到 not-ready flag 退出时递增 terminal load。

计数先保存在本核 `LocalStats`，kernel 结束时才发布到独占 `WorkerResult`。
`WorkerResult` 在 D1 从 704 B 扩展到 768 B；合入 atomic 泳道计数和 I-cache
cold/warm 配对字段后，standalone 诊断 sidecar 按完整 cache line 扩展到 832 B。
原 PMU 字段 offset、生产 DistGlobal/DistCore offset 和 `LocalSlot` ABI 都不变。
三后端完整重建后，smoke 和 256 batch 当时默认 NOP 均通过全部语义断言。

对任一 worker，若其完成数为 `Cw`、fanin 边数为 `Ew`、失败 load 为 `Fw`，则
逐核检查：

~~~text
frontier_initial_w == Cw
frontier_terminal_w == Cw
fanin_ready_w >= Ew
fanin_ready_w - Ew <= 2 * Fw
~~~

最后一个上界来自 PA 最大 fanin 为 3：一次失败检查最多先重读两个 ready 前缀，
然后在一个 not-ready 依赖上返回。全局 `T=1280` 时还必须满足：

~~~text
frontier_initial = frontier_terminal = T
frontier_ready = frontier_FetchMax = A >= T
frontier_flag_loads = A + T
Submit+completion ops = 79872 + G + 2A
~~~

其中 `G=fanin_ready+fanin_not_ready`。CCEC/AscendC 的 `A` 对应真实 A5
atomicMax；CPU 的 FetchMax 是 load/CAS 实现，`A` 只能解释为逻辑调用数。

#### 7.4.2 CCEC 十轮动态基线

在 256 batch、当时默认 NOP、关闭泳道和 phase profile 的同一进程十轮基线中，全部
语义断言和上述计数恒等式均 PASS：

| 指标 | 中位数 | 均值 | nearest-rank p90 | 范围 |
| ---- | -----: | ---: | ----------------: | ---: |
| Submit（us） | 4776.940 | 4802.821 | 5335.931 | 3805.757～5535.473 |
| fanin 总 load `G` | 93201.5 | 99442.6 | 145045 | 56620～151260 |
| fanin ready | 7323.5 | 7418.0 | 9370 | 4191～9933 |
| fanin not-ready | 86675.5 | 92024.6 | 136347 | 49038～141327 |
| frontier FetchMax `A` | 15365 | 14957.8 | 18957 | 5587～20026 |
| Submit+completion ops | 203803.5 | 209230.2 | 264969 | 164508～269046 |

日志位于：

~~~text
tests/atomic_probe/pa_scheduler/outputs/atomic_diagnostics/
  ccec_baseline_10_20260718_023551.log
~~~

该十轮不是独立进程交错 A/B，因此只用于确认动态规模，不用于宣布性能收益。
分类代码新增约 `2T+A` 次 worker 私有 scalar 增量，且结果 sidecar 扩大了一条
cache line；本节绝对 Submit 不能与 D1 之前的二进制直接归因比较。

#### 7.4.3 对下一候选的约束

当前动态最大项不是 ready 前缀，而是 not-ready 重试：其中位数约 86676 次；
frontier helping 的额外 FetchMax 中位数为 `A-T=14085` 次。ready-prefix cache 在
固定轮询序列下最多删除 `fanin_ready-E=6043.5` 次中位重复 load，约占 fanin 总量
6.5%、Submit+completion ops 3.0%，预期只能是小幅候选。

仍先做该候选，因为它不改变依赖集合、flag 发布或跨核共享状态，正确性风险最低。
但 probe 变短后可能在同一等待期间发起更多 not-ready 重试，所以整轮 `G` 不保证
静态单调下降。保留门槛是：优化后 `fanin_ready == fanin_edges == 1280`，CCEC
独立进程交错十对中 fanin 总 load 配对中心下降、Submit 中心不回退；否则记录负
结果并撤回，不迁移真实 FDWIC。standalone 第一版只验证当前单-lane PA Case1，
不能拿它的 PASS 代替 joint/BlockWon 覆盖。

### 7.5 阶段 O2：atomic schema-v3 泳道与 Submit scalar PMU

本节保留 atomic ABI 在 schema-v3 中建立和验收的完整过程。当前 raw 已
升为 schema-v4，atomic site/op/flags、PollBatch 与 weighted summary 语义不变；
atomic ABI 没有重定义。phase schema 则新增排他父区间，以真实 Submit 尾 span
替换旧 lap，将 EfDrain 改为显式 span，并禁止未使用的 `DrainWon`。

#### 7.5.1 观察目标与证据拆分

O2 先完成观察链路，不在同一阶段继续消减 atomic。它回答两个不同问题：

1. atomic 泳道回答“每个 site/op 执行了多少次逻辑调用”：direct Atomic 逐条给出
   source-issue 或本核 return-ready bracket；显式等待区内六类 observation load
   用带精确 `call_count` 的 PollBatch 给出逻辑轮询 episode；
2. PMU sidecar 回答“指定 Submit 窗口内，每个物理子核累计了多少
   scalar/vector/cube/MTE/fix busy、I-cache request/miss 和 raw total”。

两类数据默认分开采集。direct trace 会增加两次 `get_sys_cnt()`、分支和一条
64 B 私有 record 写入，PollBatch 还会增加等待区内的私有累计与边界落盘；二者都会
改变代码布局、I-cache、worker 到达顺序、共享地址竞争和轮询次数。PMU-only 不写
完整泳道记录，观察机制与扰动来源不同；由于
`submit-all` gate 自带 `PIPE_ALL`，当前没有同配置 A/B 证明其总扰动一定更小。
它用于取 AIC/AIV 平均和 PMU event A/B。正式 PMU JSON 强制
`--no-swimlane`，不与 atomic trace 或 phase profile 合并；
两套结果只按同一源码版本和各自明确的 scope 交叉解释，不能声称逐 tick 精确对齐。

#### 7.5.2 十五个 site、四种 op 与六类 PollBatch

`AtomicSite` 是 raw/merged trace 的稳定编号，当前覆盖 standalone PA 公共调度器中
所有显式共享 atomic 源码调用点。`AtomicOp` 只有 `Load/Exchange/FetchAdd/FetchMax`
四种；A5 CCEC 的 `Load` 仍是 `atomicAdd(address, 0)`，不是普通 GM load。

| id | AtomicSite | AtomicOp | 调度语义 |
| --: | ---------- | -------- | -------- |
| 0 | `StartupIncrement` | `FetchAdd` | worker 启动计数发布 |
| 1 | `StartupPoll` | `Load` | worker 启动屏障轮询 |
| 2 | `FatalPoll` | `Load` | 成功/异常路径 fatal 检查 |
| 3 | `FatalSet` | `Exchange` | 首次异常发布 |
| 4 | `ClaimMax` | `FetchMax` | 分片 cursor Claim |
| 5 | `FaninFlagLoad` | `Load` | 依赖 ready 检查 |
| 6 | `CompletionVendExchange` | `Exchange` | task vend 发布 |
| 7 | `CompletionFlagExchange` | `Exchange` | task flag 发布 |
| 8 | `FrontierInitialLoad` | `Load` | completion 开始时读取 frontier |
| 9 | `FrontierFlagLoad` | `Load` | 连续完成区间扫描 |
| 10 | `FrontierMax` | `FetchMax` | frontier 帮助前推 |
| 11 | `HeapFrontierLoad` | `Load` | HeapGuard slow path 读取 frontier |
| 12 | `HeapVendLoad` | `Load` | HeapGuard slow path 读取 retire vend |
| 13 | `ReplayDoneIncrement` | `FetchAdd` | worker 回放结束计数发布 |
| 14 | `ReplayDonePoll` | `Load` | 最终 drain 的 replay_done 轮询 |

schema-v3 要求 `metadata.trace_schema_version=3` 且 `l2_swimlane_level=4`，并将
**逻辑调用数**与**物理记录数**分开。direct Atomic 仍是一条源码调用
对应一条同时包含 `start_cycle/end_cycle` 的记录；只有下列六类 observation load
在匹配的显式等待区内才允许聚合为 PollBatch：

| `site_id` | site | op |
| ---: | --- | --- |
| 1 | `startup_poll` | `Load` |
| 2 | `fatal_poll` | `Load` |
| 5 | `fanin_flag_load` | `Load` |
| 11 | `heap_frontier_load` | `Load` |
| 12 | `heap_vend_load` | `Load` |
| 14 | `replay_done_poll` | `Load` |

standalone 没有真实 PA 追加的 BlockWon site，也没有允许聚合的幂等失败 exchange；
不能照搬真实 PA 的“九类 observation load 加一类 exchange”allowlist。同一 site 在
显式等待区外的一次性或 opportunistic 读取仍逐条记录。

raw 的 `auxiliary` 保存 site id。direct Atomic 的 `flags` 约定为：低 4 bit 是 op id，
bit 4 表示返回旧值是否参与后续判断，bit 5 仅对 Load 表示观察值是否为零，bit 6
表示是否有“返回值本核可消费”依赖证据，bit 7 必须为 0；bits 8..31 仅对 direct
FetchMax 保存软件 retry 数，不能按调用次数解析。能归属任务时 `task_id` 写真实
task id，生命周期 atomic 写 `-1`。

PollBatch 的 `flags` 则必须同时满足：低 4 bit 为 `Load(0)`、bit 4 为 1、bit 5/6
为 0、bit 7 为 1，bits 8..31 保存 `1..0xFFFFFF` 的精确无符号 24 bit
`call_count`；`task_id=-1`、`function_id=-1`。达到 `0xFFFFFF` 时先落盘，再从
1 开启下一条 batch，不能饱和后丢计数。

converter 把 direct Atomic、PollBatch 和 ClockBaseline 放在对应 AIC/AIV 的原
scalar lane，atomic 不再伪装成与 scalar 并行的独立执行单元。direct 名称显式带边界：
`atomic.return_ready.<site>.<op>#<task_id>` 或
`atomic.source_issue.<site>.<op>#<task_id>`；category 也分别为
`atomic.return_ready`/`atomic.source_issue`，无需点开 span 即可过滤区分。direct
args 中保留 `call_count=1`、整数 `cycles`、site/op、
`result_used`、`return_ready_observed`、`completion_boundary`、Load 的
`value_zero` 和 FetchMax 的 `retries`。Perfetto 的浮点微秒显示不用于
替代 raw 整数 tick。

PollBatch 名称为 `atomic.poll_batch.<site>.load×<call_count>`，category 为
`atomic.poll_batch`；args 还明确给出 `poll_window_cycles`、
`batch_semantics=observation_load_calls`、
`duration_semantics=logical_poll_episode_envelope_not_single_atomic_latency` 和
`may_contain_interleaved_direct_atomics=true`。一个等待区可以同时累计多个 site，
所以不同 PollBatch 窗口可以重叠，窗口内也可能交错 direct Atomic。
host 文字分析也保持两套口径：`[TRACE_ATOMIC]` 只统计 direct bracket，
`[TRACE_ATOMIC_POLL]` 只统计 episode 数、精确逻辑调用数和等待包络分布。

#### 7.5.3 按调用点语义区分的两种结束边界

不能按 `Exchange/FetchAdd/FetchMax` 指令名称一概选边界，必须看该源码
调用点是否真正消费 atomic 返回的旧值。当前有两种口径：

1. `source_issue_bracket`：

~~~text
begin = get_sys_cnt()
old   = atomic(...)
end   = get_sys_cnt()
~~~

它用于返回旧值本来就不使用的发布型调用。`end` 与 `old` 无数据
依赖，只表示源码发射包围区间；不表示返回值就绪、atomic retire 或
其他核可见。

2. `return_value_ready`：

~~~text
begin = get_sys_cnt()
old   = atomic(...)
asm volatile("MOV old, old; MOV end, SYS_CNT"
             : "+l"(old), "=&l"(end))
~~~

该边界只用在协议本来就要判断 `old` 的 `Load/FetchMax` 调用。CCEC AIC
和 AIV 后端已确认生成紧邻的 `ATOM -> dependent MOV -> MOV SYS_CNT`；
`=&l` 防止时间戳输出与 atomic 返回寄存器重叠。它能证明 `old` 已可被
本核 scalar 消费，仍不证明跨核全局可见。默认不加 DSB/ISB/额外 GM
地址依赖；这些操作要么本平台后端不支持，要么会明显改写被测路径。

standalone 的五个 `Exchange/FetchAdd` 调用点都不消费返回旧值，但共享
新值仍由协议的后续 load 消费：

| 发布调用点 | 旧值 | 新值消费者 | b1 热路实际情况 |
| ------------ | ---- | ------------ | ---------------- |
| `StartupIncrement` | 丢弃 | `StartupPoll` | 96 次发布，所有 worker 参与轮询 |
| `ReplayDoneIncrement` | 丢弃 | `ReplayDonePoll` | 96 次发布，所有 worker 参与轮询 |
| `CompletionFlagExchange` | 丢弃 | `FaninFlagLoad`/`FrontierFlagLoad` | 每 task 发布一次 |
| `CompletionVendExchange` | 丢弃 | wrap 后的 `HeapVendLoad` | b1 首圈 fast path 不读 vend |
| `FatalSet` | 丢弃 | `FatalPoll` | 成功 b1 不执行 `FatalSet` |

所以强制这五处消费 `old` 会把原 no-return/发布路径改成等待返回型
观测变体：Startup/Replay 会改变屏障到达和轮询次数，vend 会推迟 flag
发布，flag 会推迟 frontier helping。如果之后需要这种返回路径对照，必须
做独立、单 site mask 的 A/B 诊断，不能把 B 的数据称为原 PA 热路。更贴近
PA 的端到端 GAP 是“发布发射 -> 自然消费者首次观察到新值”；可以
复用现有 load 记录做离线派生，但其包含互连可见性、消费者调度与轮询
间隔，且在对外报告跨核时间差前还需单独验证各核 `SYS_CNT` 的对齐性。

两种 direct 边界的 `end` 都在本条 64 B trace record 写入之前取得，所以本条
duration 不直接包含自己的 record 写入；但这次写入、附加指令和代码布局会影响
后续 atomic 到达、竞争与 I-cache，整轮仍是插桩运行。direct record 写入不会隐式
关闭活跃 PollBatch，否则等待区内自然交错的发布/推进 atomic 会把一个逻辑等待
episode 人为切碎。

PollBatch 使用另一种时间语义：`duration`/`poll_window_cycles` 是从该 site 在显式
等待区内首次累计调用到边界关闭的逻辑等待 episode 包络。它不是独占 scalar 时间，
不是 `call_count` 次 atomic 延迟之和，也不是其中某次 load 的 return-ready 延迟；
不同 site 的窗口可以重叠，窗口内还可能包含 direct Atomic，不能把 PollBatch 混入
direct 单次延迟的 median/p95。

以下是历史 schema-v3 边界实现复用真实 PA 规则时的约束；其中 lap helper 仍保留
历史兼容能力，但当前 standalone schema-v4 producer 已不再调用
`ResetTraceLap/WriteTraceLap`：

1. 显式等待区退出时关闭与该 region 匹配的 PollBatch；
2. `TraceTimestamp` 先采 cycle，再以该 cycle 关闭全部活跃 batch，然后写 phase
   begin/end；
3. `ResetTraceLap` 在推进 lap 起点前关闭，`WriteTraceLap` 在写 lap 前用同一
   `end_cycle` 关闭；
4. Kernel begin/end 都经过 `TraceTimestamp`，所以 PollBatch 不能跨入或跨出 Kernel；
5. `FlushTraceCore` 的最终关闭只作防御性兜底，不能代替上述语义边界。

开启 atomic trace 时，每个 worker 在最终 drain 之后额外记录两条
`ClockBaseline`：一条是连续两次 `get_sys_cnt()`，另一条是纯寄存器依赖
hook 后读 `SYS_CNT`。它们只给出同一二进制、同一物理核上两类边界的
计时分辨率和固定底噪分布，使用边界是：

- AIC 与 AIV 分开报告 median/p95/max；
- 不把某个 role 的中位数逐条从 atomic duration 中相减；
- 不因某条 atomic 接近 ClockBaseline 就声称该 atomic 没有竞争或没有等待；
- 不用 ClockBaseline 推导 atomic retire、cache 一致性或跨核可见性时刻。

#### 7.5.4 计数闭环与容量门禁

schema-v3 结果只有同时满足逻辑调用与物理记录闭环才可进入正式分析。设
`direct_atomic_records` 是 bit 7 为 0 的物理 Atomic 条数，则逐核和全局都必须满足：

~~~text
logical_atomic_calls = direct_atomic_records + Σ(PollBatch.call_count)
physical_atomic_records = direct_atomic_records + poll_batch_records
physical_atomic_records
    = logical_atomic_calls - batched_poll_calls + poll_batch_records
~~~

这里 `batched_poll_calls` 是所有 PollBatch `call_count` 之和，不是 batch 条数。
producer 的逐核 state、host 扫描 raw 行、导出的 metadata 和 converter 重算必须同时
闭合以下七项：

~~~text
records
atomic_records
clock_baseline_records
atomic_calls
batched_poll_calls
poll_batch_records
dropped_records
~~~

此外还必须满足：

1. 每条 PollBatch 都属于六类 allowlist，flags 合法且 `call_count>0`；
2. 每个 worker 恰有 2 条 ClockBaseline，因此 96 核全局固定为 192 条；
3. 原 phase、动态 wait、物理 Atomic 和 ClockBaseline 总数与 trace header 精确相等；
4. 每核和全局 `dropped_records==0`，raw 到 merged 后物理 Atomic 条数不变；
5. 原有 Claim、winner、kernel、fanin、frontier、cursor、heap 和每 worker
   1280 Submit 语义断言仍全部 PASS。

trace 容量不足时必须明确报 overflow 并判该轮观察无效，不能只分析前缀，也不能
根据静态公式补齐被丢弃的 duration。CPU pthread 启动轮询可能远多于 A5，CPU 的
容量结果不能替代真实 A5 CCEC 的闭环；CPU/AscendC 主要用于公共 schema、编译和
语义回归。

#### 7.5.5 Submit PMU 窗口与平均口径

CCEC 只保留一个正式逐 worker Submit 窗口，CPU/AscendC 对应 hook 是空实现，不伪造
PMU 数据：`submit-all` 在本 worker 通过启动屏障后，于 orchestration 初始化前
直接调用 `PmuWindowStart`，在该 worker 最后一次 Submit 返回后关闭。窗口包含
参数构造、全部 Submit 调度，以及本 worker 在窗口内执行的 winner 计算体：当前
无参数默认的 `real-compute` 是真实 Cube/Vector 与对应搬运流水；显式
`scalar-nop` 校准样本的 NOP 则在 scalar 上执行并计入窗口。

它是“每核从 orchestration 到最后一次 Submit 返回”的累计窗口，不是全局最早
`Submit.start` 到最晚 `Submit.end` 的墙钟窗口，也不是逐次 Submit 窗口。
启动屏障、`replay_done` 发布/轮询和最后一次 Submit 后的 final drain 不在该 PMU
窗口内；因此不能把十五个 atomic site 的全生命周期 trace 总和与 Submit PMU
直接一一对齐。

`metrics_prof_start/stop()` 都含 `PIPE_ALL` 屏障，门控边界会收口流水并可能改变
worker 到达与争用时序。这个开销属于观察配置本身；只能比较 gate、负载模式和构建
布局完全相同的样本，不能把 PMU 样本的 Submit span 当成无观察性能基线。

PMU JSON sidecar 按 worker 保留 raw 记录，并分别汇总 32 AIC、64 AIV 和全部 96 核。
各字段严格按下面口径解释：

- `total_cycles`：gate 活跃期间的每核 64 bit PMU raw total；本机受控同窗校准为
  `1,817,457 PMU cycles / 1,101,593 ns = 1.649844 cycles/ns`，AIC/AIV
  分别约为 `1.650062/1.649731 cycles/ns`。因此可按
  `time_us = cycles / cycles_per_ns / 1000` 换算每核 cycle-equivalent；96 核
  求和仍是 core-work，不是 Submit 墙钟时间。墙钟只看 host 记录的
  `submit_span_us`；
- `scalar_busy`：`CNT2 scalar_instr_busy` 的事件累计，不等于窗口内全部时间，
  也不等于“纯调度耗时”。atomic 由 scalar 发射，其发射、返回值依赖或资源
  阻塞可能在该事件中有所体现，但不能据此认定某条 atomic 的全部执行完成延迟
  都等量算入 `scalar_busy`；
- `icache_requests/misses`：`CNT6/CNT7` 的事件累计。整体 miss rate 必须用
  `sum(misses)/sum(requests)`，AIC/AIV 分开计算，不能平均 96 个逐核百分比。
  request/miss 是“次数”事件，不是 busy 或 stall 时长；I-cache miss 引起的取指空泡
  可能体现为 scalar 无法发射而不增加 `scalar_busy`，因此不存在“每个 miss 自动加到
  scalar busy”的恒等式；
- `vector/cube/MTE1/MTE2/MTE3/fix busy`：同一 PipeUtilization 配置的辅助证据，
  用来检查窗口内实际活跃单元，不能从事件名称反推出一条指令造成的精确 stall；
- `window_started/window_stopped`：来自每个 worker 实际执行 gate 的状态位，不用模式
  配置推导“应该执行过”；所有 worker 还必须通过九项 selector、唯一物理核 id、
  owner bitmap membership、worker slot/物理 role/triplet、miss 不大于 request 和
  counter 风险门槛等门禁。

raw JSON 与自包含 HTML 都必须同时保留完整 Submit 的 `submit_span_us`、
`total_cycles` 和 `scalar_busy`，并按 ALL/AIC/AIV 显示每核分布；不得只展示
I-cache request/miss。HTML 在保留 raw cycle 的同时按上述角色频率显示等效 µs；
顶部“完整 Submit（最早开始 → 最晚结束）”单列 96 核整体耗时，ALL/AIC/AIV
角色卡片则只用逐核 mean 表示典型值，并补充 min/max 显示核间范围，二者不是
同一个统计量；total 与 scalar 的极值也不保证来自同一物理核。
SYS_CNT 的 1 ns tick 和 `90 ns/miss` 一阶标尺不受这一 PMU 频率换算影响。当前局部
phase 边界仅中途 read-clear shadow request/miss，
没有局部 `total_cycles` 或 `scalar_busy`。文档和 HTML 中的 total/scalar 因此都是
该 phase ELF 的完整 Submit 窗口，不能当成 Materialize、Register 等局部阶段的独占时间。
90 ns/miss 仍只是受控 cold/warm 探针的一阶 core-equivalent 标尺，既不加入
`scalar_busy`，也不从 `total_cycles` 中直接扣除。

##### 7.5.5.1 自包含 Main AICPU Path-A owner 闭环

owner 已收入 `pa_scheduler/ccec`，所有构建和运行文件均位于 standalone 目录内。
构建产出 Path-A dispatcher 与 owner AArch64 SO；host 通过已验证的 bootstrap、mode-0
注册和 `simpler_aicpu_exec` 执行 Configure/Restore，不需要另一个进程提供 selector。
当前 owner 没有跨进程互斥锁；同一设备上不得并发运行另一个 PMU owner 会话或
`msprof` PMU 会话，否则 selector、启动前保存态和 Restore 都可能被相互覆盖。
本次上板闭环包括：

- 对同一 stream 调用 `aclrtGetStreamResLimit`，A5 实报 AIC 32、AIV 64，活跃
  subcore 总数 96，不再把 108 个物理 MMIO slot 容量当成活跃数；
- AICPU 扫描 108 项，每项执行 save/configure/readback；成功 96 项置 bitmap，
  失败 12 项当场恢复并跳过，当前 bitmap 为
  `000003ff:fff3ff7f:f7cffcff:fffdf7ff`；
- 第一个跳过项是 index 11，`CTRL0` 读回 `0xffffffff`，期望 `0x7`；这是
  不可配置 slot 的显式证据，不再泛化成 loader 失败；
- restore 只按 bitmap 从 107 到 0 逆序还原，standalone 本体的 owner 闭环以
  96/96 物理子核、完整 triplet 和 Restore PASS 为准；
- 另有一组独立基础微基准位于 `tests/atomic_probe/ccec`：dependent-atomic 最小样本
  和 WARM/COLD I-cache 对照使用成功 bitmap 中的物理核 18，三个独立会话共
  33 对均为 warm miss 0、cold miss 68，且每个会话都为
  `pmu_restore_and_cleanup=PASS`。这只是旁路的 CNT6/CNT7 方向性证据，不代替
  `pa_scheduler` 的 96 核 owner 验收或 7.3.3 的 `icache-single` 标定。

standalone 本体还进一步通过 96/96 owner membership、精确 worker slot、物理 role、
32 个完整 triplet、每核 window started/stopped 和 Restore 门禁。`batches=1,nop=0`
的 `submit-all` 单次样本见 7.3.2；它证明窗口可用，不代表 256 batch 的正式归因。

正式 PMU JSON 必须以 `--no-swimlane --runs 1 --pmu-window submit-all` 独立采集，
不能同时启用 atomic trace、phase profile、泳道分析或泳道 JSON。host 拒绝覆盖已有
JSON/同名 `.tmp`，并只在协议、PMU/owner 门禁、Restore 和 runtime 清理全部成功后
发布文件。`--no-swimlane` 会关闭 phase/atomic record 写，但普通调度路径中的阶段
`SYS_CNT` 调用点仍存在；sidecar 会明确记录“有 timestamp call、无 record write、
无 atomic trace、无 profile accumulation”，不能把该模式描述成编译期零插桩。

#### 7.5.6 当前实现状态、历史证据与正式重采矩阵

当前 schema-v4 源码完整沿用 schema-v3 建立的六类 PollBatch、精确
`call_count`、七项 summary 闭环、两条 ClockBaseline 和 scalar lane converter。
以下先保留 2026-07-18 schema-v3 边界修复版 CCEC b1/b256 的历史证据：

| 样本 | winner 负载 | 总 `records` | 逻辑 `atomic_calls` | direct | 物理 Atomic | `batched_poll_calls` | PollBatch | ClockBaseline | dropped | 首末 Submit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| b1 | `scalar-nop=0` | 4,414 | 1,031 | 613 | 850 | 418 | 237 | 192 | 0 | 54.056 us |
| b256 | `real-compute/6,28,4,1` | 967,307 | 105,580 | 103,618 | 103,883 | 1,962 | 265 | 192 | 0 | 5,774.295 us |

两轮的 producer、raw 行与 converter 七项 summary 均闭合，96 核逐核异常数为 0：

~~~text
b1:   613 + 418 = 1,031;       613 + 237 = 850 = 1,031 - 418 + 237
b256: 103,618 + 1,962 = 105,580; 103,618 + 265 = 103,883 = 105,580 - 1,962 + 265
~~~

b1/b256 每核记录峰值分别为 57/10,252，均低于 65,536 条固定容量。PollBatch
实际分布在 `startup_poll/fatal_poll/fanin_flag_load/replay_done_poll` 四类 site：
b1 的物理 episode/逻辑调用依次为 `96/143、42/47、2/6、97/222`，b256 为
`96/143、42/47、16/467、111/1305`；HeapGuard 两类 allowlist 本轮为 0，不是漏插桩。

同核区间复核中，b1 的 237 个 PollBatch 与 4 个 Kernel、b256 的 265 个 PollBatch
与 1,024 个 Kernel 均为严格 overlap 0；分别有 3 和 31 处仅在 `end==begin` 端点
相接，符合“边界先以同一 cycle 关闭 PollBatch，再进入/退出 Kernel”的设计。b256
真实 Cube/Vector 计算、调度终态和全部语义断言均 PASS。产物为：

~~~text
outputs/pa_scheduler_swimlane_20260718_182649_4060527/ccec/l2_swimlane_records.json
outputs/pa_scheduler_swimlane_20260718_182649_4060527/ccec/merged_swimlane.json
outputs/pa_scheduler_swimlane_20260718_182725_4061524/ccec/l2_swimlane_records.json
outputs/pa_scheduler_swimlane_20260718_182725_4061524/ccec/merged_swimlane.json
~~~

b1 是零 winner 负载的快速验收，b256 是开启 atomic 泳道的诊断运行；5.774295 ms
只证明观察构建保持目标量级，不是关闭 trace 的正式性能基线，也不能与下列历史样本
做单轮减法归因观察开销。

schema-v4 于 2026-07-19 用同一 atomic ABI 完成 CCEC b1、AscendC b1 与
CCEC b256 真机验收。v4 新增 `OrchestrationReplay/FinalDrain` 父 span，并用
`WinnerBuild/LoserReplay/AllocComplete` 替代旧 lap；Atomic 仍由原 direct 与
PollBatch 规则产生。三轮均为 96 核、192 条 ClockBaseline、
`dropped=0`，Atomic weighted summary 和 raw 行逐项闭合：

| 样本 | 总 `records` | 逻辑 `atomic_calls` | 物理 Atomic | `batched_poll_calls` | PollBatch | 首末 Submit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CCEC b1 | 4,602 | 1,403 | 846 | 790 | 233 | 90.741 us |
| AscendC b1 | 4,597 | 1,338 | 841 | 725 | 228 | 86.037 us |
| CCEC b256 | 964,724 | 102,324 | 101,108 | 1,471 | 255 | 5,680.749 us |

CCEC b256 还验证了每核 1,280 个 Submit、1,024 个 Kernel 全部落入
EfDrain 或 FinalDrain：前者 1,011 个、后者 13 个，孤儿和越界均为 0；
Submit/EfDrain/Orchestration/FinalDrain/WorkerCompletion 排他闭合。最终分析器
还逐条验证 exclusive child 的 task 身份，以及 Alloc loser=`Claim.end`、
非 Alloc loser=`Register.end` 的零时长锚点。完整产物为：

~~~text
outputs/pa_scheduler_swimlane_20260719_055449_334552/ccec/
outputs/pa_scheduler_swimlane_20260719_060409_345364/ascendc/
outputs/pa_scheduler_swimlane_20260719_060634_347455/ccec/
~~~

最终源码和相同 v4 level-4 配置连续三轮 CCEC b256 为
5.785939/5.503109/5.381844 ms，中位数 5.503109 ms，极差
0.404095 ms；完整导出轮比上文 v3 导出轮低 1.62%，仍小于当前自身
轮间波动。因此当前只记录为“未观察到明显回退”，不把这个单轮差值
当成优化收益。

以下是 2026-07-18 的**历史 schema-v2、逐调用、边界修复前**证据，保留用于追溯，
不作为当前 schema-v4 的物理记录规模、逻辑调用数或边界闭合验收。历史 b1
atomic-trace-only 上板中，全部协议断言 PASS，raw 共 4959 条、
`expected=4959`、`dropped=0`，其中逐条 Atomic 1395 条、ClockBaseline 192 条。
Claim 当时已按 schema-v2 闭合为 `won=5/lost=283/not_attempted=192`。当时 converter
回归最终为 5/5 PASS，Atomic direct bracket 为
`return_ready=1193/source_issue=202`。历史未入库产物为：

~~~text
outputs/scalar_observation_final_20260718/atomic_inlineasm_ccec_b1/raw.json
outputs/scalar_observation_final_20260718/atomic_inlineasm_ccec_b1/merged_swimlane.json
~~~

该历史 b1 只用于说明旧边界、旧 schema 和泳道布局，不用它的插桩后 Submit span
替代 256 batch 约 5 ms 的无诊断基线。同一历史二进制的 256 batch 随后也完成一轮：
全部断言 PASS，raw/merged 均为 963368 个 span，`expected=963368`、
`dropped=0`，Atomic 99944 条、ClockBaseline 192 条；Claim 为
`won=1280/lost=72448/not_attempted=49152`，恰好闭环 `96*1280`。合并泳道
不存在旧 `AIC/AIV·atomic` 线程，atomic 全部位于对应 scalar lane。本轮
边界计数为 `return_ready=97192/source_issue=2752`，与动态 site 计数之和一致；
插桩 Submit span 为 5.209261 ms，只证明观察构建未把数量级打坏；正式
性能仍以关闭 trace/PMU 的独立进程 A/B 为准。本轮未入库产物为：

~~~text
outputs/scalar_observation_final_20260718/atomic_inlineasm_ccec_b256/raw.json
outputs/scalar_observation_final_20260718/atomic_inlineasm_ccec_b256/merged_swimlane.json
~~~

同一历史源码随后关闭 swimlane/atomic-trace/PMU 做了三个独立进程，Submit 为
3.729925/4.904346/5.563417 ms，中位数 4.904346 ms，三轮语义全部 PASS。
这组数据说明该历史 standalone 版本复现了约 5 ms 量级，也显示了多核轮询、
frontier helping 和 winner 分布会带来明显轮间波动；一次 5.209261 ms 插桩轮
不能与三轮中任一轮做单样本减法后归因 trace 成本。

正式重采状态如下：

| 结果 | 当前状态 |
| ---- | -------- |
| schema-v3 六类 PollBatch、flags、logical/physical 公式与拓扑的 converter 静态回归 | **19/19 PASS；不替代真机验收** |
| schema-v3 b1 的七项 summary、逐核/全局公式与 dropped | **96/96 闭合，dropped=0** |
| schema-v3 256 batch 的 direct/PollBatch raw→merged 与容量 | **已闭合，单核峰值 10,252/65,536** |
| schema-v4 排他父区间、真实 Submit 尾动作与 atomic ABI | **CCEC/AscendC b1 均通过；旧 lap=0** |
| schema-v4 CCEC b256 raw→merged→exclusive report | **964,724 条，dropped=0，六组整数 cycle 闭合** |
| 15-site schema 与六类 PollBatch allowlist | **历史 schema-v3 b1/b256 两轮 flags 全合法；实际四类有事件、HeapGuard 两类为 0** |
| 192 条 ClockBaseline 与 Kernel 边界 | **历史 schema-v3 b1/b256 两轮均闭合；严格 overlap=0** |
| `submit-all` 的 owner/gate/CNT0..8/total 闭环 | **b1、零 NOP 单次上板已通过** |
| 256 batch `submit-all` 的 I-cache 与 pipe 分组分布 | **3 个 PMU-only 独立进程已采并完成 raw→summary 重算** |
| real-compute 引擎 PMU 与 placement | **b8 count1/count2 精确倍增并逐 worker 闭合** |

此前开发过程中的探索性输出不在这里引用为正式结论。最终按同一源码依次执行：

1. 三后端全量重建及 no-trace/no-PMU 语义回归；
2. 历史 schema-v3 atomic-trace-only 已按 b1、b256 完成；当前 schema-v4
   又按 CCEC b1、AscendC b1、CCEC b256 完成六类 allowlist、flags、七项
   summary、ClockBaseline、Kernel 包含、raw→merged 与排他报告闭合；
   后续改动仍按同一门禁复测；
3. CCEC PMU-only 的 `submit-all` 已完成 3 个独立进程 A/A；后续每轮仍使用
   唯一 JSON 路径，并检查 96 核、owner bitmap/triplet、start/stop、25% 风险门槛和 Restore；
4. real-compute b8 count1/count2 已完成 Cube/Vector busy 精确倍增与
   `EfDrain` placement 闭合；
5. 比较任何 atomic 优化前后时，双方使用相同观察模式、相同 winner mode/count、
   相同 gate/trace 配置和
   独立进程交错 A/B；正式端到端 Submit 仍以无诊断构建复测。

#### 7.5.7 绝不能从 O2 声称的结论

即使上述重采全部通过，也不能声称：

- 单条 source-issue 或 return-ready bracket 就是硬件 atomic retire 延迟、跨核
  可见延迟或一致性完成时间；
- 所有 atomic duration 相加就是 Submit 中“atomic 占用时间”，或删除这些 atomic
  一定能等量缩短墙钟时间；不同核 direct bracket 会重叠，PollBatch 之间也可重叠，
  poll 数还会随插桩改变；
- PollBatch duration 是单次 load 延迟、`call_count` 次延迟之和或独占 scalar 时间；
  它只表示逻辑等待 episode 包络；
- `ClockBaseline` 可以逐事件相减并得到无偏的 atomic 硬件净耗时；
- trace 开启后的 Submit span 可以直接与无 trace 基线比较并宣布性能收益；
- `scalar_busy/total_cycles` 是墙钟时间，或 I-cache miss 数可以直接换算成精确
  stall ns；7.3.3 的 90 ns/miss 只是受控 cold/warm 探针得到的一阶等效估算；
- standalone 的 `scalar-nop` 已经真实模拟了 vector/cube task 期间 scalar 的等待状态；
  NOP 本身在 scalar 上执行并被 `submit-all` 统计。CCEC `real-compute` 虽已真实激活
  Cube/Vector/MTE/FIX，也仍不能由此推导完整真实 PA 的 scalar 等待与资源竞争；
- 最大 32-bit counter 低于 25% 风险门槛就证明本轮没有回卷；该门槛只降低风险，
  最终寄存器值不能直接检出恰好一圈或多圈的 wrap；
- standalone 的 PMU/atomic 分布可以直接替代真实 FDWIC PA。迁移真实用例时仍需
  复用已验证的最小观察代码，并重新完成计数闭环、正确性和无诊断性能 A/B。

#### 7.5.8 历史 schema-v2：256 batch loser ClaimMax 的定量归因

本节只保留 2026-07-18 边界修复前 `atomic_inlineasm_ccec_b256` 的历史定量结果。
ClaimMax 在 schema-v3 中仍是 direct FetchMax，不会被 PollBatch 聚合；但下列数值来自
旧二进制，只能作为后续重测的优先级假设，不能充当当前版本验收数据。该历史样本中，
Claim 三态为 1280 个 winner、
72448 个 attempted loser 和 49152 个 role-filtered not-attempted。每个 attempted Claim
恰好包含一条 `claim_max.fetch_max`，与 73728 条 ClaimMax 精确闭环。

| 口径 | 次数 | median | p95 | max | 多核累计 core-work |
| ---- | ---: | ---: | ---: | ---: | ---: |
| loser ClaimMax | 72448 | 280 ns | 637 ns | 4791 ns | 24.133954 ms |
| winner ClaimMax | 1280 | 268 ns | 508 ns | 1277 ns | 0.376479 ms |
| AIC loser ClaimMax | 23808 | 269 ns | 523 ns | 3434 ns | 7.058334 ms |
| AIV loser ClaimMax | 48640 | 290 ns | 677 ns | 4791 ns | 17.075620 ms |

loser ClaimMax 占全部 99944 条 atomic bracket core-work 的 58.6%；去掉 Submit
窗口外的 startup/replay-done 生命周期 atomic 后，它占 Submit 内 atomic bracket
core-work 的 77.7%。因此它是“atomic 内部”的第一大项，但优势主要来自
72448 次动态调用，不是 loser 单次比 winner 慢一个数量级。

不能由此把整个 Replay 或整个 Submit 都归因给 ClaimMax。`Replay` 是 lap
外层区间：从 materialize 前开始，包含 Materialize、PrepareMap、Claim、Register，
到 loser 分支结束，这些嵌套 span 不能相加。72448 个 attempted-loser Replay 的
外层 core-work 为 192.977600 ms，其中 loser ClaimMax 直接 bracket 占 12.5%；
49152 个完全没有 FetchMax 的 not-attempted Replay 仍有 93.430073 ms core-work。

按每核累计看，loser ClaimMax 中位数为 0.258278 ms，占每核 Claim 中位
累计的 45.7%、Replay 外层的 8.45%、Submit envelope 的 5.0%。本轮最晚结束的
AIV core93 上，对应数字为 0.261731/5.191702 ms，也约 5.0%。这只是直接
可见的本核返回等待占比；真正改动 Claim 协议后还会改变 winner 到达、
frontier/fanin 时序和竞争形态，不能简单从 5.209261 ms 中减去 0.258 ms。

历史阶段性结论是：ClaimMax loser 是后续 atomic 消减的第一优先级，但尚未证明
它是整个调度的主瓶颈。256 batch PMU-only 三轮 A/A 已补齐 Submit 全窗的
I-cache 和 pipe 分组基线；它显示 AIV miss rate 在本三轮持续高于 AIC，但现有
counter 仍不能把 miss 定位到 materialize/register 或某条 atomic。开始改 Claim 后，
应使用同配置的 PMU-only 交错 A/B 观察 request/miss 是否同步变化，端到端收益
仍由无诊断的独立进程 A/B 确认。

#### 7.5.9 CCEC winner 负载从 scalar NOP 迁移到真实 Cube/Vector

为避免 winner 执行期的 scalar NOP 污染 `scalar_busy`，2026-07-18 先在
standalone CCEC 增加显式 `--winner-workload real-compute`，没有迁移真实 PA。
在提交 `e66001ff` 对应的这个历史阶段，无参数默认当时仍为 `scalar-nop`，用于
保证旧三后端基线不静默变化；当时选择真计算而未指定次数时，QK/SF/PV/UP
使用 `6,28,4,1`。三后端闭环后当前默认已经切换为 `real-compute`，但本节后续
数据仍按当时的显式模式和参数解读，不能追改成新默认口径。

实现与门禁如下：

1. QK/PV 只在 AIC 执行完整 `128x128 float` Cube matmul，包含
   MTE2/MTE1、M、FIX、GM store 和最终完成等待；
2. SF/UP 只在 AIV 执行完整 Vector add/mul，包含 MTE2、V、MTE3、GM store
   和最终完成等待；每次 repeat 完成写回后才复用 tile；
3. 两个 GM 输入 tile 为所有 worker 只读共享，每 worker、每角色使用独占输出
   tile。host 在计时外初始化/传输，计时后逐 tile 验证 768/5/6 与 inactive
   sentinel，共 12,713,984 bytes；
4. 最终 device ELF 严格限制为两个 mixed kernel GLOBAL 入口；冷路径 dispatcher
   与 Cube/add/mul 三个执行 helper 必须是非空 LOCAL 函数。这个门禁来自一次已
   复现故障：错误暴露的 GLOBAL helper
   被 runtime 当入口启动，导致 scalar 模式也进入 Cube 路径；修正后同一 ELF 的
   scalar b1 与 real b1 均通过；
5. repeats 上限为 128，避免 b256 极端 winner 分布让 32-bit PMU 计数接近回卷
   区域。真计算 count 与 NOP count 互斥，0 次被拒绝。

标定不是以凑齐 5.1 ms 为目标。三个独立 b256 进程的 QK/SF/PV 中位数约为
41.336/54.039/27.971 us，最接近真实泳道目标 44.170/53.729/27.626 us；
UP 一次完整流水约 2.5 us，已经是当前正整数下限。三轮 Submit 为
3.808/3.555/3.706 ms，中位数 3.706 ms；最终重建后的单轮为 3.683649 ms。
Cube/Vector 分布在物理子核并行执行，与 scalar NOP 串行占用 scalar 的到达时序
本来不同，不能通过增加无关 repeat 把总时间硬拉回 5.1 ms。

最终常量 tile 只证明某个 active worker-kind 至少完成一次，因为同一 tile 会被
后续 repeat 覆盖；全部 repeat 的证据来自下面的受控 PMU 倍增，而不是数值结果本身。

`submit-all` PMU 做了独立倍增取证。b8、四类 count=1 时，窗口内 placement 为
29 个 EfDrain 和 3 个 FinalDrain；恰有 14 个 AIC worker 的 Cube 非零且逐核
`cube_busy=8281`，15 个 AIV worker 的 Vector 非零且逐核
`vector_busy=936/937`，与 29 个 EfDrain 精确闭合。独立 count=2 样本为
28 个 EfDrain、4 个 FinalDrain，14 个 AIC 与 14 个 AIV 非零计数档位分别为
16562 和 1872/1874，精确两倍。获胜 worker 会随调度变化，不能把两轮按同一核
强配对。FinalDrain 位于每核 PMU stop 之后，故 count1/count2 的 3/4 个
FinalDrain kernel 不应出现在 Submit sidecar；这不是漏计。

泳道 raw/merged 的 `trace_schema_version` 仍为 2；本轮只扩展可选 metadata，
没有把它误升为 PMU JSON 的 schema v3。metadata 同时保存 mode、四类 count、迭代单位与
QK/PV=Cube、SF/UP=Vector 映射，并增加可见的全局 capture metadata 事件。
最终 b1 real-compute 泳道有 4964 条 raw data event；converter 产出 4965 条
data event（增加一条 capture instant），再加 256 条 process/thread metadata，
最终 `traceEvents` 为 5221 条，`dropped=0`。转换器含旧 schema 兼容在内为 5/5 PASS。

当时的阶段边界：提交 `e66001ff` 只证明 CCEC standalone 的真实引擎负载、
数值、角色、PMU 与泳道闭环；当时 AscendC 真实计算和 CPU 对等算术尚未完成，
因此本阶段没有宣称三后端 winner 负载已经对等，也没有迁移真实 PA。
后续完成情况分别记录在 7.5.10 和 7.5.11。

#### 7.5.10 AscendC winner 负载迁移到真实 Cube/Vector

提交 `9aeda0dd` 在不改动 PA 调度模型的前提下，把 CCEC 已验证的
winner workload 布局、参数解析和 host 数值校验抽到三后端共用头文件，然后
在 AscendC mixed kernel 内分角色接入真实计算：

1. AIC 上的 QK/PV 执行 `128x128 float` matmul，路径为 GM ND 到
   A1/B1 的 NZ 布局、A1/B1 到 A2/B2、`Mmad`、FIX 回写 GM；
   `MTE2_MTE1`、`MTE1_M`、`M_FIX` 和最后的 `FIX_S` 保证 Kernel span
   包住本轮回写完成边界；
2. AIV 上的 SF/UP 执行 GM 到 UB、`Add`/`Mul`、UB 到 GM，并用
   `MTE2_V`、`V_MTE3`、`MTE3_S` 等待完整的 load/Vector/store 流水；
3. 12,713,984-byte workspace、输入 2/3、输出 768/5/6、每 worker-kind
   独占 tile 和 inactive sentinel 与 CCEC 共用同一个口径。H2D 初始化在
   launch 计时前，D2H 与数值校验在计时后，不把 host 搬运写入 Submit span。

首个 AIC 版本的数值闭环暴露了真实错误：A1 和 B1 虽然是不同逻辑位置，
但映射到同一块物理 L1；两个 64 KiB tile 都从地址 0 开始时，后搬入的 B
覆盖 A，因而把本应为 `2 * 3 * 128 = 768` 的结果错算成
`3 * 3 * 128 = 1152`。将 B1 起始地址错开 64 KiB 后，整个输出 tile 恢复为
768。右矩阵 B 在 GM 中是普通 KxN row-major，所以 L1 zN 到 L0B nZ 时设置
分形 transpose，而不是盲目复用 A 的非转置参数。当前常量 B=3 不能单独
证明转置差异，因此本阶段先以 CANN 布局语义修正；随后再用 7.5.12 的非均匀
输入逐元素上板闭合，没有把常量输出冒充布局证据。

修正后的分层证据为：

| 场景 | 参数 | Submit span | 数值/调度结果 |
| ---- | ---- | ----------: | ---------------- |
| b1 | count=`1,1,1,1` | 59.280 us | PASS，4 active tile + 188 sentinel tile |
| b8 | count=`1,1,1,1` | 166.426 us | PASS，32 active tile + 160 sentinel tile |
| b256 样本 1 | count=`6,28,4,1` | 3810.471 us | PASS，191 active tile + 1 sentinel tile |
| b256 样本 2 | count=`6,28,4,1` | 4828.567 us | PASS，192 active tile |
| b256 样本 3 | count=`6,28,4,1` | 3777.371 us | PASS，192 active tile |

三个独立 b256 进程的 Submit span 中位数为 3810.471 us。相同三个样本中，
QK/SF/PV/UP 的每 task 平均 Kernel span 分别约为 41.1–41.4 / 47.3–49.7 /
27.5–28.2 / 2.5–2.7 us。这些 span 含上述引擎完成等待，但 AscendC 路径
没有 CCEC Main AICPU PMU owner，因此不伪造 Cube/Vector busy counter，也不用
这三轮总时间反推 scalar PMU。

b1 count=1 还完成了带 atomic 的泳道闭环：4652 条 raw data event 与静态
期望精确一致，`dropped=0`，atomic 源码边界调用为 1088。converter 增加一条
capture instant 后有 4653 条 data event，再加 256 条 process/thread metadata，
最终 `traceEvents=4909`。raw 和 merged 都记录 `real-compute`、`1,1,1,1`
及 QK/PV=Cube、SF/UP=Vector 映射，数值校验也同轮 PASS。

本阶段证明 AscendC standalone 已脱离 scalar NOP 执行真实 A5 Cube/Vector
流水，且调度语义、输出和泳道闭环一致；它不是真实 PA 迁移，也不替代
CCEC 的 PMU 取证。

#### 7.5.11 CPU winner 负载补齐对等算术

提交 `1d3a374a` 让 CPU 后端使用同一组 workspace、角色路由、repeat
参数和输出校验：QK/PV 以三重浮点循环执行完整 `128x128` 矩阵乘，
SF/UP 执行逐元素 add/mul；每个 repeat 后保留 compiler memory boundary，避免
因为下一轮覆盖同一 tile 而被 O3 删除。每轮重新初始化 2/3 输入和输出
sentinel，所以 `runs>1` 不会沿用上一轮 winner 结果。

分层回归中，scalar-NOP b1、real-compute b1 count=`1,1,1,1` 均 PASS；
后者 Submit span 为 39.821 ms，并校验 4 active tile + 188 sentinel tile。b8
count=`2,3,2,1` 连续两轮也均 PASS，两轮分别为 80.310/54.185 ms，
均校验 25 active tile + 167 sentinel tile。同时保留 CLI 互斥和边界门禁。在该
历史阶段，real-compute 与 NOP override 不能混用，count 需要与 real-compute
模式一起使用，取值限制为 1..128，CCEC 专属 PMU 参数仍会被非 CCEC 后端拒绝。
当前无参数默认已是 real-compute，因此当前 CLI 允许 count/pattern 直接覆盖默认
真计算；显式 NOP override 会选择 scalar-nop，显式 real-compute 与 NOP override
仍然互斥。

CPU 阶段的“对等”只包括 PA 任务调度、AIC/AIV 角色选择、workspace 编址、
repeat 次数和 768/5/6 数学结果。CPU pthread 调度、普通浮点循环和 x86 atomic
都不是 A5 Cube/Vector/scalar 流水；上述毫秒数据只用于证明用例实际执行完成，
不用作 A5 timing 或 PMU 性能结论。

至此，三后端的 winner workload 参数、workspace、角色路由、数学输出和
泳道 metadata 口径已对齐；CCEC/AscendC 执行真实 A5 引擎负载，CPU 仅做
调度与算术回归。这仍然是 standalone 验证，不能替代真实 PA 的计数闭环、
正确性和无诊断性能 A/B。

#### 7.5.12 用非均匀输入闭合 Cube/Vector 数据布局

常量 A=2、B=3 能验证真实指令、角色和 GM 写回，却会同时掩盖 B 转置、
ND/NZ stride 与分形重排错误。为把该盲区变成可重复工具，三后端公共 CLI 增加
`--real-compute-pattern constant|layout-diagnostic`：默认仍为 `constant`，不改变
性能基线；诊断模式只改变计时窗外的 host 输入生成和输出期望，不进入
`SchedulerState`，也不在 CCEC/AscendC device 热路径增加分支。

诊断输入定义为：

```text
A[r,c] = (r == c) ? (r + 1) : 0
B[r,c] = 1 + ((131*r + 17*c + 7*r*c) mod 251)
```

QK/PV 的期望为 `(r+1)*B[r,c]`，SF 为 `A[r,c]+B[r,c]`，UP 为
`A[r,c]*B[r,c]`。B 是非对称稠密矩阵，A 是带权对角矩阵，因此 B 转置、A 行映射、
stride 或输出重排都会在确定元素上产生不同整数；最大结果不超过 32128，FP32
可做逐元素精确比较而无需容差。

严格按 CCEC → AscendC → CPU 顺序，以 b1、count=`1,1,1,1` 上板/运行：

| 后端 | Submit span | QK/SF/PV/UP Kernel span | 输出门禁 |
| ---- | ----------: | ----------------------- | -------- |
| CCEC | 37.682 us | 9.172 / 3.819 / 7.415 / 2.512 us | 4 active + 188 sentinel，PASS |
| AscendC | 55.041 us | 8.478 / 4.211 / 20.801 / 3.008 us | 4 active + 188 sentinel，PASS |
| CPU | 51.958 ms | 3.457 ms / 26.118 us / 3.390 ms / 17.349 us | 4 active + 188 sentinel，PASS |

CCEC 先证明公共期望与 PTO `A*B` 语义正确；随后 AscendC 使用同一输入通过，直接
闭合了 A1/B1 错位、普通 KxN B 的 zN→nZ 分形 transpose、ND/NZ stride 和
FIXPIPE NZ→ND 输出。CPU 只证明 host 公式与调度路由，不把毫秒值外推到 A5。
三后端又以 `pattern=constant` 做 b1 count1 回归，并以 `smoke all` 回归原
scalar-NOP，均 PASS，说明诊断模式没有静默改变当时的性能默认或旧控制路径。

AscendC 同模式泳道位于
`outputs/pa_scheduler_swimlane_20260718_125904_3613100/ascendc/`，raw 为
4584 条 data event、`dropped=0`，merged 增加一条 capture instant；raw/merged
的 `metadata.winner_workload.input_pattern` 都是 `layout-diagnostic`。CCEC 的
`submit-all` PMU sidecar 同样记录该字段，且 `accepted=true`、
`semantic_passed=true`。converter 对新字段保留旧 schema-v2 兼容，并新增非法
pattern 拒绝回归，当前两种 unittest 入口均为 6/6 PASS。

该诊断证明一次完整 engine pipeline 的数学和布局，不单独证明同一 task 的 N 次
repeat 都执行；repeat 完整性仍使用 CCEC count1→count2 engine PMU 精确倍增，
不把最终覆盖同一 tile 的结果夸大成次数证明。

#### 7.5.13 默认切换为真负载及同口径性能验收

完成 CCEC→AscendC→CPU 分阶段闭环后，共享 `WinnerWorkloadOptions`
的无参数默认从 `scalar-nop` 切换为
`real-compute/constant/6,28,4,1`。`smoke` 仍显式固定 b1/r1/scalar-nop=0；
旧命令未指定 mode 但显式给出 `--nop-count*` 时自动选择 scalar-nop。
显式 real-compute 与 NOP override、显式 scalar-nop 与 real count/pattern 仍互斥。

性能验收先纠正了观察口径：早先 3.7～4.4 ms 是 `--no-swimlane`，
真实 PA 5.1 ms 是标准 L2 泳道；泳道记录不仅增加指令，还会改变 worker
到达、fanin 失败重试和 RingBp，所以不能把两者直接相减成“缺失的调度时间”。
当前保留的 CCEC b256 真负载历史 phase-only 泳道、不开逐 atomic 的 5 个独立进程为：

```text
5002.413 / 4875.193 / 4968.894 / 4992.477 / 4876.282 us
```

中位数 4,968.894 us。真实 PA 最终三轮中位数 5,115.620 us，同口径
差 146.726 us，约 2.87%。五轮 standalone 的 QK/SF/PV/UP 每 task 均值
中位数为 41.461/54.007/28.053/2.649 us，总 core work 已贴近真实 PA，
不通过增加 repeat 继续硬凑总时间。

保留的一轮历史 phase-only raw 为
`outputs/performance_gap_20260718/standalone_ccec_real_b256_raw.json`；863,237 条
记录全部有效，与真实 PA 863,232 条的基本阶段数完全相同，只额外有 5 条
RingBp。这说明 standalone 已达到“独立复现约 5 ms 调度”的目标；仍然保留
本文第 6 节的边界：它不依赖 simpler 生产代码，也不复刻真实 PA 数值数据流和
通用多 group/joint 拓扑。当前正式 `swimlane` action 已固定合并普通阶段和
逐 atomic 记录，因此该 863,237 条历史文件只用于同口径性能参照，不能作为当前
合并泳道的记录数量或容量证据。

#### 7.5.14 默认真负载的 Submit I-cache 基线

本节只回答一个目标：完整 Submit 控制流窗口内，平均单核、尤其 AIV 上发生多少
I-cache miss，以及其中多少性能损失可以被当前证据支持。正式主指标按每轮角色
`sum / cores` 计算后再取五轮中位数；逐核百分比不做算术平均。

默认负载切换完成后，以当前同一 CCEC ELF 显式指定
`real-compute/constant/6,28,4,1`，关闭泳道和逐 atomic，采集 5 个独立
`submit-all` PMU-only 进程。每轮均为 schema v3、96/96 trusted、32 AIC +
64 AIV、32 个完整 triplet；协议、真计算数值、Submit 内 placement/engine、
start/stop、counter 门槛和 Restore 全部 PASS。新增的
`pmu_sidecar_analyzer.py` 从 96 条 raw 独立重算 ALL/AIC/AIV 全字段，与 host
summary 五轮完全一致：

| 轮次 | Submit span | request 总和 | miss 总和 | 总 miss rate | AIC / AIV miss rate |
| ---- | ----------: | -----------: | --------: | -------------: | --------------------: |
| 1 | 4,027.976 us | 45,857,671 | 5,908,019 | 12.8834% | 8.5382% / 15.0323% |
| 2 | 3,719.597 us | 45,498,660 | 5,896,760 | 12.9603% | 8.5318% / 15.1902% |
| 3 | 3,732.543 us | 45,520,846 | 5,909,025 | 12.9809% | 8.5716% / 15.2014% |
| 4 | 3,621.080 us | 45,603,314 | 5,897,686 | 12.9326% | 8.4889% / 15.1748% |
| 5 | 3,581.265 us | 45,551,060 | 5,894,904 | 12.9413% | 8.5307% / 15.1632% |

五轮中位数为 Submit 3,719.597 us、request 45,551,060、miss 5,897,686，
总/AIC/AIV miss rate 为 12.9413%/8.5318%/15.1748%。request 全范围跨度约
0.79%，miss 约 0.24%；相比 Submit span 和 scalar busy，I-cache 总量在这
五轮更稳定，但这不取消正式 A/B 对多个独立进程和交错顺序的要求。

直接面向目标的逐核结果为：ALL 平均 61,434.229 miss/核，AIC 平均
40,626.219 miss/核，AIV 平均 **71,865.516 miss/核**；AIV 的五轮范围为
71,768.250～72,068.734 miss/核，逐核分布 p95 的五轮中位数为 73,035。
AIV 平均 request 为 473,306/核，按组内 `sum(miss)/sum(request)` 得到上述
15.1748%。由于 AIC/AIV 是明显双峰，ALL 的逐核 median 不用于替代分角色均值。

同一 ELF 又显式运行 3 个历史标定强度的 scalar-NOP 样本
`129600,157900,79950,2400`。raw 重算中位数为 request 70,236,792、miss
5,942,635；与真计算相比，request 多约 35.1%，绝对 miss 只多约 0.76%。AIC/AIV
每核 miss 中位数分别约 40.2K/72.8K，真计算为 40.6K/71.9K。这个对照只支持：
当前约 5.9M 的 CNT7 总量对 winner 负载模式不敏感，更可能主要来自两种模式共用的
调度控制流和代码布局；不能仅看真计算 miss rate 较高，就错误归因为真引擎造成更多
miss。两组负载和调度到达不同，本轮也不是交错配对性能 A/B，故不使用其 Submit
差值宣称性能收益。

同一 ELF 的 3 个 `empty` 进程只在 RunScheduler 后打开空 gate。每轮 96 核
owner/selector/start/stop/Restore 均闭环，total 每核中位数为 173～179；全核
request/miss 中位数仅 958/444，分别约为正式 Submit 中位数的 0.0021%/0.0075%。
empty 的 46% 左右 miss rate 来自极小分母，没有性能含义，也不从 Submit 中机械
相减。

按 90 ns/miss 的受控 cold/warm 标尺，AIC/AIV 每核的串行等效量约为
3.66/6.47 ms；AIV 结果甚至超过本轮约 3.72 ms 的 PMU-only Submit span，也超过
熟悉的约 5 ms 泳道基线。这正面证明 miss 会在不同核间并行，在单核内也可能重叠，
并且受控 cold miss 与真实热路径 miss 不能假定为同一个可加常数。因此 6.47 ms
**不是实际损失**，当前 A5 PIPE_UTIL 事件表也没有已经核实的 I-cache stall-cycle
counter。实际暴露损失必须由同语义代码布局 A/B 同时给出 `ΔAIV miss/核` 和关闭
PMU/泳道后的 `ΔSubmit span`；在该 A/B 完成前，本节把实际损失明确记为“尚未测得”，
不从总 miss 机械换算。

这里的 3.719597 ms 是本组 `--no-swimlane`、PMU gate 打开的五轮中位数；约
5 ms 是标准泳道配置的熟悉基线，两者不是同一观察配置，不能直接相减。PMU
`submit-all` 是每 worker 从 orchestration 初始化前到本核最后一次 Submit 返回；
`submit_span_us` 则是 96 核最早的首个 `Submit.begin` 到最晚的末个
`Submit.end`。二者都覆盖完整 Submit 控制流，但边界不完全相同。

本机未入库的原始 sidecar 位于：

~~~text
outputs/pmu_submit_all_real_compute_b256_20260718T140539Z/run1.json ... run5.json
outputs/pmu_empty_real_compute_b256_20260718T140908Z/run1.json ... run3.json
outputs/pmu_submit_all_scalar_nop_b256_same_elf_20260718T141455Z/run1.json ... run3.json
~~~

#### 7.5.15 观察构建收敛为 `swimlane` 与 `submit-pmu`

2026-07-18 在旧统一 ELF 完成 atomic 泳道、PMU owner 和 I-cache 取数
可行性取证后，standalone 的最终观察产物收敛为两类：

| 构建 | 正式内容 | 隔离要求 |
| --- | --- | --- |
| `swimlane` | 普通阶段和逐 atomic 记录合并在对应 AIC/AIV scalar lane | 不配置 PMU，不导出 PMU JSON |
| `submit-pmu` | 每核完整 Submit PMU，可带一个编译期局部 phase | 编译掉泳道、逐 atomic、ClockBaseline、runtime phase-profile 和旧 I-cache 冲刷体 |

旧 O1/O2 文字中的 `empty/scalar/scalar-double/icache-single`、CNT8 fix-busy、
schema-v3 以及“`--no-swimlane` 仍保留 phase timestamp”都是观察链路建设过程。
它们保留为历史证据，不再定义当前命令和当前 ELF 口径。

当前 `submit-pmu` 仅白名单支持：

| phase | id | 现行边界 | 成功流 calls |
| --- | ---: | --- | ---: |
| `none` | 0 | 不执行局部 counter 读取，只取完整 Submit | 0 |
| `claim` | 1 | 每次 `Claim()` 前后 | 每核 `5*batches` |
| `efdrain` | 2 | 每次 Submit 开头唯一的 `DrainReady(...EfDrain...)` 前后 | 每核 `5*batches` |
| `materialize` | 4 | 每次 `MaterializeTask()` 前后，包含成功与失败返回边界 | 每核 `5*batches` |
| `register` | 5 | Alloc 和非 Alloc 两个互斥 `RegisterOutputs()` call-site 前后 | 每核 `5*batches` |

`efdrain` 不把复用 `DrainReady()` 的 RingBackpressure/FinalDrain 混入。
`register` 在 Alloc 路径传入 `include_existing=false`，在非 Alloc 路径传入
`true`；两个 call-site 互斥且所有 worker 每个 task 只进入一次，所以与
Claim/EfDrain/Materialize 一样可以用固定 `5*batches` 闭合。这个窗口匹配
现有 Register 泳道 span，但不意味五类 task 每次都实际执行 TensorMap insert；
短路径中局部 PMU 边界本身的扰动占比会更明显。

对应命令和产物为：

~~~bash
./run.sh build-submit-pmu ccec none
./run.sh build-submit-pmu ccec claim
./run.sh build-submit-pmu ccec efdrain
./run.sh build-submit-pmu ccec materialize
./run.sh build-submit-pmu ccec register

./run.sh submit-pmu ccec none \
  --device 0 --batches 256 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1 \
  --pmu-json ./outputs/<unique-none>/submit_icache_raw.json

./run.sh submit-pmu ccec claim \
  --device 0 --batches 256 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1 \
  --pmu-json ./outputs/<unique-claim>/submit_icache_raw.json

./run.sh submit-pmu ccec efdrain \
  --device 0 --batches 256 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1 \
  --pmu-json ./outputs/<unique-efdrain>/submit_icache_raw.json

./run.sh submit-pmu ccec materialize \
  --device 0 --batches 256 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1 \
  --pmu-json ./outputs/<unique-materialize>/submit_icache_raw.json

./run.sh submit-pmu ccec register \
  --device 0 --batches 256 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1 \
  --pmu-json ./outputs/<unique-register>/submit_icache_raw.json
~~~

成功采集后同目录自动生成 `submit_icache_report.html`。raw 继续作为权威证据，
HTML 只提供 ALL/AIC/AIV 汇总、逐核分布和 running phase lower/upper 的离线
可视化；顶部单列完整 Submit（最早开始 → 最晚结束）的整体耗时，PMU
total/scalar 角色卡片只显示逐核 min/mean/max。HTML 不改变原始统计和可信门禁。

~~~text
build/ccec/submit-pmu/none/
build/ccec/submit-pmu/claim/
build/ccec/submit-pmu/efdrain/
build/ccec/submit-pmu/materialize/
build/ccec/submit-pmu/register/
~~~

每个 phase 目录中的 host、mixed kernel、owner 与 dispatcher 是同一构建
集，不能跨 phase 复用。`submit-pmu` action 固定单轮、关闭泳道且只打开
完整 Submit 窗口；不接受 atomic trace、phase profile、泳道分析或泳道 JSON。

完整 Submit 的权威 I-cache 计数为不在局部边界读取的
`CNT6=0x34 primary request` 和 `CNT7=0x35 primary miss`。局部归因需要
一对可中途 read-to-clear 的重复计数槽。首版将 miss 放在
`CNT9=0x35`，但 A5 b1 上板中 CNT9 始终为 0，已反证该槽可用性。
因此正式配置调整为：

| 槽位 | 配置 | 用途 |
| --- | --- | --- |
| CNT5 | `0x35` | shadow miss；诊断 ELF 不再保留 MTE3 busy |
| CNT6 | `0x34` | primary whole request |
| CNT7 | `0x35` | primary whole miss |
| CNT8 | `0x34` | shadow request |
| CNT9 | `0x0` | 未使用 |

这个取舍只影响 `submit-pmu` 诊断 ELF，不影响标准 `swimlane`。
任一现行局部 phase 在 begin/end 读 CNT8/CNT5，stop 后再读 tail；所有片段的软件
累加构成 shadow whole。两种构建的接受条件不同：

~~~text
none（没有运行中 read-clear）：
  shadow request == primary request
  shadow miss    == primary miss

局部 phase（运行中反复 read-clear）：
  shadow request <= primary request
  shadow miss    <= primary miss

  request loss = primary request - shadow request
  miss loss    = primary miss - shadow miss

  phase request ∈ [observed request, observed request + request loss]
  phase miss    ∈ [observed miss,    observed miss    + miss loss]
~~~

上下界先按每核 raw 计算，再分别按 AIC/AIV 聚合，不能拿聚合后的 median
相减拼区间。CNT8/CNT5 是两条顺序 `ld_dev`，不是原子配对快照，所以
`phase miss <= phase request` 不是硬门禁；二者分别不得超过对应 shadow，
上界分别不得超过对应 primary。

除此之外，还必须闭合 build variant/phase id、每核 begin/end/calls、完整
Submit 的 `miss <= request`、96 个唯一物理子核、owner bitmap/role/triplet、
真计算输出、Submit placement/engine、counter 风险门槛和 Restore。`none`
的 phase calls/begin/end/request/miss 必须全为 0，并要求 96/96 shadow 精确
等于 primary；其余现行局部 phase 要求 96/96 shadow 不大于 primary，
exact 核数只作
诊断，不再伪装成逐事件精确切片。

局部 begin/end 读本身会增加 scalar 取指和改变多核时序，所以
所有局部 phase 都是带观察边界扰动的归因 ELF。不同 phase 的局部
request/miss 不可相加，也不能用 `phase - none` 宣称得到零扰动净值。
每个 ELF 的完整 Submit 始终以它自己的 CNT6/CNT7 primary whole 为准。
当协议、数值输出和 placement/engine 门禁全部通过时，运行中 shadow 的
单向负差只能描述为局部 PMU 分段误差，不能描述成 standalone 调度异常。

submit-pmu schema-v5 JSON 保留 96 条 raw，并分别给出 ALL/AIC/AIV 的 authoritative
whole、shadow loss 以及 phase lower/upper。raw 中显式保存
`shadow_request_loss`、`shadow_miss_loss`、
`phase_icache_requests_upper_bound` 和 `phase_icache_misses_upper_bound`。
完整 Submit 的 miss rate 只按 `Σmiss/Σrequest` 计算，不平均逐核百分比；
局部 lower miss/lower request 之比只叫 observed read-clear ratio，不是实际
miss rate 的数学下界。已有隔离微基准的约
90 ns/miss 只用于 `Σmiss * 0.09 us` 的 core-work 数量级感性估算；
它不是可相加的 Submit stall 常数。真正暴露的墙钟收益必须通过同语义
代码的交错 A/B，同时观察 `Δmiss/core` 和无 PMU/泳道的
`ΔSubmit span`。完整操作和排错见
[`../icache_miss_usage_guide.md`](../icache_miss_usage_guide.md)。

#### 7.5.16 完整 Submit 精确闭合与局部 read-clear 边界取证

2026-07-18 的四个独立 b256 `submit-pmu none` 进程均使用默认真负载
`real-compute/6,28,4,1`，且 96/96 核 shadow 与 primary 逐值相等：

| 轮次 | Submit span/us | request 总和 | miss 总和 | AIC miss/core | AIV miss/core |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3825.420 | 40,020,837 | 4,748,592 | 38,702.656 | 54,845.422 |
| 2 | 3600.091 | 40,035,347 | 4,726,896 | 38,466.438 | 54,624.531 |
| 3 | 3610.648 | 39,977,052 | 4,728,552 | 38,481.000 | 54,643.125 |
| 4 | 3731.247 | 39,787,176 | 4,715,096 | 38,583.875 | 54,381.438 |

四轮中位数为：Submit span 3670.948 us、AIC miss/core 38,532.438、
AIV miss/core 54,633.828。`none` 的严格精确仅指同 selector、同 gate 的完整
Submit counter 逐核闭合；它不等于真实 PA 的绝对 profile，也不把 PMU
进程的 Submit span 当成无诊断墙钟基线。

补充 interval-schema 的单轮 b256 A5 复核：

| 构建 | Submit span/us | exact/bounded 核 | shadow request loss | shadow miss loss | 语义与真计算 |
| --- | ---: | ---: | ---: | ---: | --- |
| `none` | 4584.835 | 96/96，96/96 | 0 | 0 | 全部 PASS |
| `claim` | 4382.161 | 40/96，96/96 | 253 | 580 | 全部 PASS |

`claim` 共执行 122,880 次 phase 调用；loss 随运行中边界读取次数放大，但
untouched primary、PA 任务拓扑、atomic 协议、winner、输出和 engine 观察
仍全部通过。这组证据把问题定位在运行中 read-to-clear 的局部分段能力，
而不是 standalone scheduler。

四轮 `none` 的约 90 ns/miss 感性标尺对应 AIC 约 3.468 ms/core-equivalent、
AIV 约 4.917 ms/core-equivalent。它们不能相加，也不能叫 Submit 墙钟损失；
实际暴露损失仍为 `UNMEASURED`，必须另做同语义优化前后交错 A/B。

本机原始证据位于：

~~~text
outputs/submit_pmu_none_validation_20260718_b256_real/run1.json ... run4.json
outputs/submit_pmu_final_gate_20260718/none_b256/run1.json
outputs/submit_pmu_final_gate_20260718/claim_b256/run1.json
~~~

#### 7.5.17 EfDrain 独立 phase 的 A5 闭环

EfDrain 只在 `SubmitTask` 开头的专属 call-site 划界；`DrainReady()` 仍由
RingBackpressure、HeapGuard 和 FinalDrain 复用，函数体本身没有 phase 分支。
因此成功流每核固定 calls 为 `5 * batches`：

| 规模 | 每核 calls | AIC/AIV calls | global calls | exact/bounded | request loss | miss loss | Submit span |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| b1 | 5 | 160 / 320 | 480 | 87/96，96/96 | 9 | 0 | 62.402 us |
| b256 | 1,280 | 40,960 / 81,920 | 122,880 | 60/96，96/96 | 1,113 | 0 | 4,844.973 us |

b256 的 phase request 为 `3,258,436..3,259,549`，phase miss 为
`482,396..482,396`；调度语义、任务拓扑、真计算输出、placement/engine、
owner/Restore 和分析器 raw 复算全部 PASS。该窗口既可能走空 ring fast path，
也可能执行 ready slot 的真实 Cube/Vector workload、scalar wait、completion
发布和 slot 回收。不同 phase ELF 的 Submit span 与局部计数仍不可相减。

本机证据：

~~~text
outputs/submit_pmu_phases_20260718/efdrain_b1/run1.json
outputs/submit_pmu_phases_20260718/efdrain_b256/run1.json
~~~

#### 7.5.18 描述性 I-cache 产物与现行五阶段 b256 复核

2026-07-19 使用相同的 `real-compute/6,28,4,1` 负载，分别为 `none`、
`claim`、`efdrain`、`materialize` 和 `register` 启动一个独立 b256 进程。
所有采集均通过
语义、96 核拓扑、owner/Restore、raw 重算和 phase calls 门禁。产物统一命名为：

```text
submit_icache_raw.json       # 96 核权威原始件
submit_icache_report.html    # 自包含 HTML 加工件
```

完整 Submit 的 primary whole 统计如下。每一行来自不同 phase ELF；边界读取会
改变代码布局与运行时序，因此这些行用于检查各自采集是否合理，不能相减成 phase
净开销。

| phase | Submit/us | AIC request/core | AIC miss/core | AIC miss rate | AIV request/core | AIV miss/core | AIV miss rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | 4750.810 | 408317.344 | 38664.344 | 9.4692% | 422480.609 | 55098.625 | 13.0417% |
| `claim` | 5271.392 | 443636.531 | 25863.438 | 5.8299% | 445036.422 | 61462.859 | 13.8107% |
| `efdrain` | 4807.369 | 442476.094 | 15960.688 | 3.6071% | 442261.094 | 57427.188 | 12.9849% |
| `materialize` | 4195.642 | 434706.750 | 20468.656 | 4.7086% | 445711.047 | 56744.156 | 12.7312% |
| `register` | 4960.087 | 431990.906 | 26459.562 | 6.1250% | 438417.688 | 50043.422 | 11.4146% |

同一批 raw 中的完整 Submit PMU total/scalar busy 如下。上表 `Submit/us` 是
96 核完整 Submit 的整体耗时；下表 raw 列是每核 event 均值，二者不是同一
统计量。等效时间列按 ALL 的 `1.649844 cycles/ns` 换算，只表示每核
cycle-equivalent。`scalar/total` 仅是两个同窗口事件求和的描述性比值，
不得称为 scalar 利用率、局部 phase 耗时占比或 96 核墙钟。

| phase | ALL total/core raw | total 等效 us/core | ALL scalar/core raw | scalar 等效 us/core | scalar/total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `none` | 7,213,914.333 | 4,372.483 | 5,717,308.729 | 3,465.363 | 79.2539% |
| `claim` | 6,878,355.458 | 4,169.094 | 5,371,850.677 | 3,255.975 | 78.0979% |
| `efdrain` | 6,910,059.479 | 4,188.311 | 5,499,963.656 | 3,333.626 | 79.5936% |
| `materialize` | 6,577,155.385 | 3,986.532 | 5,119,585.177 | 3,103.072 | 77.8389% |
| `register` | 6,913,673.615 | 4,190.501 | 5,524,170.406 | 3,348.299 | 79.9021% |

局部 running read-clear 结果为：

| phase | AIC/AIV/global calls | phase request lower..upper | request/Submit primary | phase miss lower..upper | miss/Submit primary | exact/bounded 核 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `claim` | 40960 / 81920 / 122880 | 2114523..2114676 | 4.9545%..4.9549% | 322606..322696 | 6.7757%..6.7775% | 36/96，96/96 |
| `efdrain` | 40960 / 81920 / 122880 | 3293444..3294284 | 7.7559%..7.7578% | 481284..481284 | 11.4972%..11.4972% | 73/96，96/96 |
| `materialize` | 40960 / 81920 / 122880 | 15285878..15286038 | 36.0209%..36.0213% | 1317810..1317818 | 30.7424%..30.7426% | 74/96，96/96 |
| `register` | 40960 / 81920 / 122880 | 5486034..5488868 | 13.0986%..13.1054% | 394420..395074 | 9.7400%..9.7562% | 36/96，96/96 |

这里 `Claim()` 为每个 worker 的五类任务选择唯一 winner；EfDrain 是每次
Submit 开头对已 ready 私有 ring slot 的机会式回收；Materialize 构造每个 worker
的任务 tensor 结果和本地 heap 状态；Register 包围两个互斥输出登记 call-site。
四个局部 phase 都每核固定 `5*batches` 次。
表中的占比只在每一行自己的 phase ELF 内计算：局部 lower/upper 分别除以
该 ELF 的完整 Submit primary 总量；不能用另一份 `none` 作分母。

本机证据：

~~~text
outputs/submit_pmu_none_20260719_b256_final/{submit_icache_raw.json,submit_icache_report.html}
outputs/submit_pmu_claim_20260719_b256/{submit_icache_raw.json,submit_icache_report.html}
outputs/submit_pmu_efdrain_20260719_b256/{submit_icache_raw.json,submit_icache_report.html}
outputs/submit_pmu_materialize_20260719_b256/{submit_icache_raw.json,submit_icache_report.html}
outputs/submit_pmu_register_20260719_b256/{submit_icache_raw.json,submit_icache_report.html}
~~~

#### 7.5.19 Materialize/Register b1+b256 真机闭环

2026-07-19 对新增的 Materialize 和 Register 分别完成 b1 与 b256 真实 A5
独立进程采集。每个 worker 对 Alloc/QK/SF/PV/UP 五类 Submit 各执行一次
该边界，所以 b1 固定
5 calls/core，b256 固定 1,280 calls/core。AIC/AIV/global 闭合分别为：

| phase | 规模 | AIC/AIV/global calls | exact/bounded | request loss | miss loss | Submit span |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `materialize` | b1 | 160 / 320 / 480 | 94/96，96/96 | 2 | 0 | 64.391 us |
| `materialize` | b256 | 40,960 / 81,920 / 122,880 | 74/96，96/96 | 160 | 8 | 4,195.642 us |
| `register` | b1 | 160 / 320 / 480 | 93/96，96/96 | 2 | 1 | 134.418 us |
| `register` | b256 | 40,960 / 81,920 / 122,880 | 36/96，96/96 | 2,834 | 654 | 4,960.087 us |

局部 request/miss 及它们占**同一 phase ELF**完整 Submit primary 的比例为：

| phase | 规模 | phase request lower..upper | request/whole | phase miss lower..upper | miss/whole |
| --- | --- | ---: | ---: | ---: | ---: |
| `materialize` | b1 | 95,704..95,706 | 42.5425%..42.5434% | 4,920..4,920 | 28.8462%..28.8462% |
| `materialize` | b256 | 15,285,878..15,286,038 | 36.0209%..36.0213% | 1,317,810..1,317,818 | 30.7424%..30.7426% |
| `register` | b1 | 34,883..34,885 | 16.7122%..16.7131% | 1,910..1,911 | 11.0853%..11.0911% |
| `register` | b256 | 5,486,034..5,488,868 | 13.0986%..13.1054% | 394,420..395,074 | 9.7400%..9.7562% |

完整 Submit 的 PMU total/scalar busy 仍由 primary whole gate 取得，与上表的局部
request/miss 不是同一种切片口径：

| phase | 规模 | PMU total sum | scalar busy sum |
| --- | --- | ---: | ---: |
| `materialize` | b1 | 2,596,263 | 2,074,125 |
| `materialize` | b256 | 631,406,917 | 491,480,177 |
| `register` | b1 | 2,670,764 | 2,075,880 |
| `register` | b256 | 663,712,667 | 530,320,359 |

四轮的 `capture.accepted`、语义、真计算输出、placement/engine、96 核拓扑、
phase call shape、primary/shadow bounded、counter 风险门槛、owner Restore 和离线 raw
复算均为 **PASS**。Materialize 在上层先保存 `MaterializeTask()` 布尔返回值、
关闭 phase 后再处理 fatal，因此失败返回也不会留下未闭合边界。Register 的
Alloc/非 Alloc call-site 互斥，因此没有重复计数。这些 PASS 证明当前工具与
standalone 协议闭合，不把单轮 Submit span 当成无观察基线，也不把局部
request/miss 比例解释成局部耗时比例。

权威 raw 和对应 HTML 加工件位于：

~~~text
outputs/submit_pmu_materialize_20260719_b1/{submit_icache_raw.json,submit_icache_report.html}
outputs/submit_pmu_materialize_20260719_b256/{submit_icache_raw.json,submit_icache_report.html}
outputs/submit_pmu_register_20260719_b1/{submit_icache_raw.json,submit_icache_report.html}
outputs/submit_pmu_register_20260719_b256/{submit_icache_raw.json,submit_icache_report.html}
~~~

#### 7.5.20 schema-v5 局部阶段时间取证

2026-07-19 在不增加新 phase、不改业务调用点的前提下，
`submit-pmu` 从 schema-v4 升到 schema-v5，给既有
`claim|efdrain|materialize|register` 边界补充 SYS_CNT 时间累计。设备侧复用
`WorkerResult` offset 744 的 64-bit 诊断槽，结构仍为 832B，没有改变
worker stride 或相邻字段 offset。

时间边界严格定义为：

~~~text
begin: shadow request/miss read-clear -> phase_begin = SYS_CNT
end:   phase_end = SYS_CNT -> shadow request/miss read-clear

phase_elapsed += phase_end - phase_begin
~~~

因此阶段时间不包含两侧顺序 `ld_dev`，但包含每次调用两次
SYS_CNT 和相关边界 bookkeeping 对被观察 ELF 的扰动。没有加 DSB/ISB。
当前 CANN `llvm-objdump` 对 `elf64-hiipu` 只能列符号而不能解码指令，
所以这里只声明已由源码顺序、device status 和真机逐核时间门禁闭环，
不冒充已完成 ISA 反汇编证明。

raw 对 96 个 worker 新增 `submit_elapsed_ticks`、`phase_elapsed_ticks`和
`phase_time_valid`；host 和离线分析器都要求 running phase 满足
`0 < phase_elapsed_ticks <= submit_elapsed_ticks`，`none` 则必须为 0。当前 v5
required status mask 为 `0x7cf`；历史 v4 仍以 `0x3cf` 只读兼容，且明确
标记阶段时间不可用，不从 PMU total 或 request/miss 反推。

ALL/AIC/AIV 的时间占比统一按同一 SYS_CNT 口径计算：

~~~text
时间占比 = Σ本组 phase_elapsed_ticks / Σ本组 submit_elapsed_ticks
~~~

这是逐核累计 core-time 构成，不是局部阶段占全局约 5 ms 墙钟的时间片；
不能把分子改除 `submit_span_us`，也不能平均 96 个逐核百分比。
HTML 已把 ALL/AIC/AIV 的时间、request、miss 三类占比移到页面最前；
时间是直接观察单值，request/miss 继续显示 lower..upper。

同一 A5、`real-compute/6,28,4,1`、b256 下的首轮 v5 取证如下。五个
独立进程均通过 96/96 记录、phase shape、`phase<=submit`、owner Restore、
真计算输出与 placement/engine 门禁：

| phase | Submit span | ALL 时间/core | ALL 占比 | AIC 占比 | AIV 占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `none` | 3.711584 ms | 不适用 | 不适用 | 不适用 | 不适用 |
| `claim` | 4.401747 ms | 470.641 us | 12.0845% | 7.7096% | 14.2972% |
| `efdrain` | 3.592376 ms | 536.279 us | 15.5068% | 20.2974% | 13.1974% |
| `materialize` | 6.770266 ms | 1,026.859 us | 16.7186% | 15.4860% | 17.3556% |
| `register` | 4.086936 ms | 158.728 us | 4.3089% | 3.6256% | 4.6568% |

`ALL 时间/core` 等于本轮 `Σphase_elapsed_ticks / 96`，不是全局墙钟。
Materialize 诊断 ELF 的 Submit span 增至 6.77 ms，直接说明高频边界观察本身
会改变取指和多核争用。所以表中比例只解释同一行的诊断 ELF；四行
不能相加，也不能与 `none` 相减得到无扰动阶段净时间。

本机权威 raw 和页面最前已带时间占比的 HTML 位于：

~~~text
outputs/submit_pmu_phase_time_v5_20260719/none_b256/
outputs/submit_pmu_phase_time_v5_20260719/claim_b256/
outputs/submit_pmu_phase_time_v5_20260719/efdrain_b256/
outputs/submit_pmu_phase_time_v5_20260719/materialize_b256/
outputs/submit_pmu_phase_time_v5_20260719/register_b256/
~~~

#### 7.5.21 排他泳道边界收敛对 atomic 口径的影响

2026-07-19 的后续改动只收敛普通 phase 边界和离线 residual，
没有增删 Atomic site，也没有改变 direct/PollBatch、`return_ready`/
`source_issue` 或 logical/physical weighted 计数公式。

standalone loser 没有真实 Replay 计算，因此已删除每个 loser 一条的
`LoserReplay` 过程态 phase 记录。这不是 atomic 消减：相关 Claim
FetchMax、fanin/completion/frontier 等 atomic 仍按实际调用采集；只是不再
用一条零工作 phase 伪装 loser 尾动作。最后一个真实 child 到
`SubmitEnd` 的时间在 Perfetto 中由离线 `submit_tail_gap` 展示，在排他
报告中汇总为 `submit_tail_residual`；它不被命名为 loser 业务阶段。

schema-v4 merged 中 direct atomic 名仍编码
`boundary/site/op/task_id`，PollBatch 名仍编码 `site/op/call_count`；
为控制数百 MiB 产物，duration 事件不再逐条复制 raw 的 `args/cat`。
精确 flags、cycle、retry、value-zero 和 weighted 计数仍以同目录
`l2_swimlane_records.json` 为准，不因 merged 瘦身丢失。

当前最新 CCEC A5 b1 为：

~~~text
outputs/pa_scheduler_swimlane_20260719_110756_584549/ccec/
~~~

该轮 4,118 条 raw 事件、`dropped=0`，全部 atomic 闭合与六类排他
时间闭合均通过；当前 converter 生成的 merged 为 428,455 bytes。
后续 atomic/边界迭代默认只跑 A5 b1；b256 仅用于阶段性规模/容量
收口或明确要求的长负载结论。

按明确要求完成的当前生产者 CCEC b256 规模复核位于：

~~~text
outputs/pa_scheduler_swimlane_20260719_114815_617346/ccec/
~~~

该轮 839,526 条 raw、97,510 条物理 Atomic、`dropped=0`，全局 Submit
为 5,360.061 us；全部 atomic 与排他闭合通过。Submit 内部阶段间 residual
为 0，尾部 residual 为 41,008,786/433,383,588 cycle（9.4625%），Submit
间 residual 为 67,065,321/500,448,909 cycle（13.4010%）。这些数值是
相同观测口径下的归因基线，不把边界重分类宣称为 atomic 或调度性能收益。
