# ordinary Scalar 联合泳道（2026-08-11）

本目录只保留 B256、ordinary Scalar Build 的七份最终 merged Perfetto
JSON，不保留 B1、raw trace、exclusive analysis、运行/构建日志、manifest
或文件 SHA。

## 最终文件

| 文件 | Policy | Real-compute counts | AICPU producer | Build union | FinalDrain union | 关系 | 结果 |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `ordinary_scalar_plan_ahead_closed_b256_real_compute_1_1_1_1_joint_full_atomic_swimlane.json` | `plan-ahead-closed` | `1,1,1,1` | 2189.105 us | 1022.708 us | 1108.190 us | producer completes before Build | PASS |
| `ordinary_scalar_plan_ahead_closed_b256_real_compute_6_28_4_1_joint_full_atomic_swimlane.json` | `plan-ahead-closed` | `6,28,4,1` | 2177.247 us | 1037.148 us | 1784.552 us | producer completes before Build | PASS |
| `ordinary_scalar_plan_ahead_closed_b256_real_compute_6_28_4_1_aicpu_taskplan_detail_5915us_joint_full_atomic_swimlane.json` | `plan-ahead-closed` | `6,28,4,1` | 2580.959 us | 1030.728 us | 1779.081 us | producer completes before Build | PASS |
| `ordinary_scalar_plan_ahead_closed_b256_real_compute_6_28_4_1_aicpu_atomic_cache_detail_13194us_joint_full_atomic_swimlane.json` | `plan-ahead-closed` | `6,28,4,1` | 9742.788 us | 1026.419 us | 1774.242 us | producer completes before Build | PASS |
| `ordinary_scalar_plan_ahead_closed_b256_real_compute_6_28_4_1_aicpu_atomic_cache_optimized_7940us_joint_full_atomic_swimlane.json` | `plan-ahead-closed` | `6,28,4,1` | 4495.542 us | 1024.405 us | 1790.605 us | producer completes before Build | PASS |
| `ordinary_scalar_streaming_future_p128_b256_real_compute_1_1_1_1_joint_full_atomic_swimlane.json` | `streaming-future-p128` | `1,1,1,1` | 3682.694 us | 3492.238 us | 1124.546 us | proven overlap | PASS |
| `ordinary_scalar_streaming_future_p128_b256_real_compute_6_28_4_1_joint_full_atomic_swimlane.json` | `streaming-future-p128` | `6,28,4,1` | 3618.343 us | 3352.975 us | 1794.165 us | proven overlap | PASS |

`Build union` 与 `FinalDrain union` 是 96 条 worker interval 的全局并集，
不是跨核 envelope，也不是 96 核工作量求和。两份 streaming 图中，AICPU
producer 与 Build union 的 midpoint overlap 分别为 3288.886 us 和
3156.840 us；analyzer 的 `proven-overlap` 结论已计入相关性误差区间。

文件名带 `aicpu_taskplan_detail_5915us` 的图是修复 AICPU 明细后的
重新采集结果：除 `RuntimePlanProducer` 外，还包含 5 个 Owner 阶段、1280
个 TaskPlan task 区间，以及每个 task 的 begin、orchestration、
stage_payload、defer_publish、publish 五类子区间。`5915us` 是这次 full
trace 采集的 `pipeline_e2e_us=5915.159`，只用于区分文件，不代表
trace-free 性能。

文件名带 `aicpu_atomic_cache_detail_13194us` 的图继续细化同一条
AICPU producer：Owner/TaskPlan 父子区间下面新增 `aicpu.atomic`、
`aicpu.cache`、`aicpu.barrier`、`aicpu.gm` 和 `aicpu.scalar` 事件。
AICPU 没有执行 AICore 的 `dcci` 指令；图中按真实 ARM 指令明确写成
`dc_cvac`（clean）和 `dc_civac`（clean+invalidate），避免把两套 ISA
混称。连续初始化操作按区间合并但保留精确次数，例如 4352 个 plan-cell
control 的 `dc civac` 和 acquire load 各只占一个事件，参数仍保留
`calls=4352`、首末 cell `0..4351`。

本次 AICPU operation trace 的闭合统计如下。`records` 是泳道事件数，
`calls/lines` 才是合并前真实操作量：

| 操作 | Records | Calls | Cache lines |
| --- | ---: | ---: | ---: |
| atomic acquire load | 13,589 | 18,964 | 0 |
| `dc cvac` | 2,820 | 15,119 | 15,119 |
| `dc civac` | 4 | 4,381 | 4,381 |
| `dsb sy` | 3,082 | 3,082 | 0 |
| `isb` | 1,544 | 1,544 | 0 |
| ordinary GM store | 2,819 | 2,819 | 0 |
| payload validation | 1,280 | 1,280 | 0 |

总计 25,138 条 operation record，`dropped=0`。`13194us` 是 full trace
下的 `pipeline_e2e_us=13193.825`；逐操作 `clock_gettime` 与 trace buffer
写入会显著放大 producer 时间，因此这一张仍只用于结构观察，不得拿来做
trace-free 性能对比。

文件名带 `aicpu_atomic_cache_optimized_7940us` 的新图是上述收缩后的
完整重新采集，不是对 13194us 历史图的覆盖。它仍保留 1280 个
TaskPlan 及全部 AICPU atomic/cache/barrier/GM/scalar 明细。两张图的原始
operation 调用量对比为：

| 操作 | 优化前 Calls | 优化后 Calls | 变化 |
| --- | ---: | ---: | ---: |
| atomic acquire load | 18,964 | 15,382 | -3,582 |
| `dc cvac` | 15,119 | 14,864 | -255 |
| `dc civac` | 4,381 | 1,309 | -3,072 |
| `dsb sy` | 3,082 | 21 | -3,061 |
| `isb` | 1,544 | 18 | -1,526 |
| ordinary GM store | 2,819 | 2,564 | -255 |
| payload validation | 1,280 | 1,280 | 0 |

优化后 1,309 次 `dc civac` 由 1,280 条实际 cell control 和 29 条
Owner 输入维护组成。cell control 按 10 个 128-cell 连续前缀区间展示；
record 数会因 scope 分段增加，但 `calls` 才是真实指令数。
`task_publish` 下的 DSB/ISB record 和 calls 都为 0。这次 full trace
共有 19,294 条 operation record、`dropped=0`，
`pipeline_e2e=7940.024us`、`producer_exec=4495.542us`；它同样只是
结构取证，性能仍以下面的 trace-free 数据为准。

### 优化后的 trace-free 口径

`aicpu_atomic_cache_detail_13194us` 是暴露过度维护的优化前结构
证据，为避免篡改历史样本没有覆盖。当前 PlanAheadClosed 已把
生产期的 256 次 frontier 推进收口为 Close 一次，把逐 task
DSB/ISB 收口到 Close，并把 B256 cell-control 复用准备从全容量
4352 条收缩为实际 `[0,1280)` 前缀。实际 payload/control line 的
`dc cvac` 仍全部保留。

真实 A5 device 0、B256、`real-compute=6,28,4,1`、trace-free 20 轮
中位数为：`producer_exec=1281.816us`、`plan_time=1366.861us`、
`aicore_time=2483.902us`、`pipeline_e2e=3850.988us`，20/20 全部 PASS。
相对优化前的 `1944.186us/2031.848us/2487.716us/4529.819us`，
producer 减少 34.07%，pipeline 减少 14.99%，AICore wall 只变化 0.15%。
详细分阶段数据见 `AICPU_TaskPlan实现过程.md` 8.7 节。

## 固定采集口径

- A5 device 0，ordinary shared TensorMap，96 个 Scalar Build worker；
- B256，共 1280 个 Runtime Plan task；
- workload 为 `real-compute`、`constant`，full swimlane + full atomic，
  `runs=1`；
- 两个 policy 分别完整构建，均使用 Plan ABI v3 和 clock-correlation ABI
  v2；前四份历史图使用五产物 manifest v8，AICPU task 明细图使用
  manifest v9，两份 AICPU operation 明细图使用 manifest v10；
- 前四次历史 A5 运行严格串行；新增图也独立运行。每次运行前均通过对应
  policy 的 manifest 校验；
- 每次采集前执行 4 个相关性样本，采集后执行 4 个样本。因此七张图均为
  `calibrated-structural-capture`，`warmup=true`，
  `performance_representative=false`；
- merged JSON 的 primary view 为 `joint_aicpu_aicore_structure`，只发布
  AICPU 与 AICore 设备时钟域，不含 Host timing 字段。

full atomic 与采集前相关性握手都会扰动绝对时间。本表只用于联合结构、相对
顺序和 overlap 取证；trace-free pipeline 性能必须单独测量，不能从这些数字
推导。

## 相关性与闭合门槛

| Policy / counts | Offset interval (tick) | Alignment error | Samples |
| --- | --- | ---: | ---: |
| `plan-ahead-closed` / `1,1,1,1` | `[225805608343,225805610241]` | 0.949 us | 4+4 |
| `plan-ahead-closed` / `6,28,4,1` | `[225805608378,225805610202]` | 0.912 us | 4+4 |
| `plan-ahead-closed` / `6,28,4,1`（AICPU task 明细） | `[225805608456,225805610244]` | 0.894 us | 4+4 |
| `plan-ahead-closed` / `6,28,4,1`（AICPU atomic/cache 明细） | `[225805608305,225805610043]` | 0.869 us | 4+4 |
| `plan-ahead-closed` / `6,28,4,1`（优化后 AICPU atomic/cache 明细） | `[225805608436,225805610240]` | 0.902 us | 4+4 |
| `streaming-future-p128` / `1,1,1,1` | `[225805608335,225805610247]` | 0.956 us | 4+4 |
| `streaming-future-p128` / `6,28,4,1` | `[225805608447,225805610230]` | 0.892 us | 4+4 |

七次 alignment error 均小于 50 us fail-closed 门槛。每张图还同时满足：

- AICPU `RuntimePlanProducer` 的 raw 与映射区间，以及全部 FDWIC 事件，
  均落在 pre/post 样本定义的双时钟域因果窗口内；
- merged trace 恰有一个独立 AICPU process 和一条
  `RuntimePlanProducer` lane；新增明细图还具有独立的 Owner 与 TaskPlan
  lane；新增 operation 明细图还把低层操作嵌套在对应 Owner/TaskPlan
  父区间内；
- analyzer 为 `joint_profiling=true`，Build/FinalDrain union 各包含
  96 个 parent interval；
- `semantic_status=PASS`、`postprocess_status=PASS`、`dropped=0`；
- Runtime Plan 以 `frontier=closed=N=1280`、`build_next=N+96=1376`、
  `workers_done=96`、`release=N`、`fatal=0` 精确收口。

B1 只可作为功能 smoke，不进入性能、泳道或状态汇报口径。
