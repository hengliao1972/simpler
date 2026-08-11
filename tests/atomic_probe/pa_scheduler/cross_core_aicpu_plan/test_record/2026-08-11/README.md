# ordinary Scalar 联合泳道（2026-08-11）

本目录只保留 B256、ordinary Scalar Build 的四份最终 merged Perfetto
JSON，不保留 B1、raw trace、exclusive analysis、运行/构建日志、manifest
或文件 SHA。

## 最终文件

| 文件 | Policy | Real-compute counts | AICPU producer | Build union | FinalDrain union | 关系 | 结果 |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `ordinary_scalar_plan_ahead_closed_b256_real_compute_1_1_1_1_joint_full_atomic_swimlane.json` | `plan-ahead-closed` | `1,1,1,1` | 2189.105 us | 1022.708 us | 1108.190 us | producer completes before Build | PASS |
| `ordinary_scalar_plan_ahead_closed_b256_real_compute_6_28_4_1_joint_full_atomic_swimlane.json` | `plan-ahead-closed` | `6,28,4,1` | 2177.247 us | 1037.148 us | 1784.552 us | producer completes before Build | PASS |
| `ordinary_scalar_streaming_future_p128_b256_real_compute_1_1_1_1_joint_full_atomic_swimlane.json` | `streaming-future-p128` | `1,1,1,1` | 3682.694 us | 3492.238 us | 1124.546 us | proven overlap | PASS |
| `ordinary_scalar_streaming_future_p128_b256_real_compute_6_28_4_1_joint_full_atomic_swimlane.json` | `streaming-future-p128` | `6,28,4,1` | 3618.343 us | 3352.975 us | 1794.165 us | proven overlap | PASS |

`Build union` 与 `FinalDrain union` 是 96 条 worker interval 的全局并集，
不是跨核 envelope，也不是 96 核工作量求和。两份 streaming 图中，AICPU
producer 与 Build union 的 midpoint overlap 分别为 3288.886 us 和
3156.840 us；analyzer 的 `proven-overlap` 结论已计入相关性误差区间。

## 固定采集口径

- A5 device 0，ordinary shared TensorMap，96 个 Scalar Build worker；
- B256，共 1280 个 Runtime Plan task；
- workload 为 `real-compute`、`constant`，full swimlane + full atomic，
  `runs=1`；
- 两个 policy 分别完整构建，均使用 Plan ABI v3、五产物 manifest v8、
  clock-correlation ABI v2；
- 四次 A5 运行严格串行，每次运行前均通过对应 policy 的 manifest 校验；
- 每次采集前执行 4 个相关性样本，采集后执行 4 个样本。因此四张图均为
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
| `streaming-future-p128` / `1,1,1,1` | `[225805608335,225805610247]` | 0.956 us | 4+4 |
| `streaming-future-p128` / `6,28,4,1` | `[225805608447,225805610230]` | 0.892 us | 4+4 |

四次 alignment error 均小于 50 us fail-closed 门槛。每张图还同时满足：

- AICPU `RuntimePlanProducer` 的 raw 与映射区间，以及全部 FDWIC 事件，
  均落在 pre/post 样本定义的双时钟域因果窗口内；
- merged trace 恰有一个独立 AICPU process 和一条
  `RuntimePlanProducer` lane；
- analyzer 为 `joint_profiling=true`，Build/FinalDrain union 各包含
  96 个 parent interval；
- `semantic_status=PASS`、`postprocess_status=PASS`、`dropped=0`；
- Runtime Plan 以 `frontier=closed=N=1280`、`build_next=N+96=1376`、
  `workers_done=96`、`release=N`、`fatal=0` 精确收口。

B1 只可作为功能 smoke，不进入性能、泳道或状态汇报口径。
