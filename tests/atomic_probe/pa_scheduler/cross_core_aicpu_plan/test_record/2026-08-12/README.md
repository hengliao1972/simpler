# PlanAheadClosed task.publish 精简验证（2026-08-12）

本目录只提交可复现口径和数据摘要，不提交 full operation trace 的大体积
raw/merged JSON。结论只适用于当前 A5 main aicpu_scheduler 的
PlanAheadClosed AICPU→AICore 路径；StreamingFuture 和 Host/SDMA→AICPU
不在删除范围内。

## 改动后的协议

```text
AICPU producer
  ordinary store 完整 PlanCell payload
  -> release atomic store(Published control)

AICore Scalar consumer
  return-ready atomic observe control
  -> DSB -> DCCI payload lines -> DSB
  -> decode/validate payload
```

`task.publish` 不再执行 payload/control `dc cvac`、DSB 或 ISB。跨轮复用时，
PlanAheadClosed 由 AICPU 对即将使用的 128-cell 单调前缀 release-store
Empty，再 acquire-load 校验；它不再对可能 dirty 的旧 Published control
执行 `dc civac`。StreamingFuture 保留原协议。

## 结构取证

口径：A5 device 0，B256，1280 tasks，`real-compute=6,28,4,1`，full
swimlane + full AICPU operation trace。该构建含逐操作取时，不代表性能。

| 操作 | 旧版本 Calls | 当前 Calls | 变化 |
| ---- | -----------: | ---------: | ---: |
| atomic acquire load | 15,382 | 15,382 | 0 |
| `dc cvac` | 14,864 | 16 | -14,848 |
| `dc civac` | 1,309 | 29 | -1,280 |
| `dsb sy` | 21 | 11 | -10 |
| `isb` | 18 | 8 | -10 |
| ordinary GM store | 2,564 | 1,284 | -1,280 |
| release atomic store | 0 | 2,560 | +2,560 |
| payload validation | 1,280 | 1,280 | 0 |

2,560 次 release store 分别是 1,280 次前缀 Empty reset 和 1,280 次
Published control。当前 `task_publish` 下 clean/barrier 调用均为 0。共生成
16,714 条 operation record，`dropped=0`；结构采集的
`pipeline_e2e=7229.859us`、`producer_exec=3781.946us` 只用于确认操作归属。

## trace-free 性能

口径：同一 A5 device 0，B256，1280 tasks，`real-compute=6,28,4,1`，
perf-clock，同进程 20 轮中位数。改动前后都为 20/20
execution/semantic/postprocess PASS。

| 指标 | 删除前 | 删除后 | 差值 | 相对变化 |
| ---- | -----: | -----: | ---: | -------: |
| Plan wall | 1356.843us | 1149.669us | -207.174us | -15.27% |
| Producer exec | 1279.676us | 1062.002us | -217.674us | -17.01% |
| AICore wall | 2475.957us | 2477.879us | +1.922us | +0.08% |
| startup→FinalDrain | 2447.496us | 2450.675us | +3.179us | +0.13% |
| Pipeline E2E | 3834.191us | 3626.891us | -207.300us | -5.41% |

## 复现命令

```bash
source /home/q00473782/Ascend/cann-9.1.0-weekly-20260708/cann-9.1.0/set_env.sh

tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/ordinary/scalar_build/run.sh \
  perf-clock ccec --device 0 --batches 256 --runs 20 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1

tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/ordinary/scalar_build/run.sh \
  run ccec --device 0 --batches 1 --runs 5 \
  --winner-workload scalar-nop --nop-count 0

tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/ordinary/scalar_build/run.sh \
  build-swimlane ccec
tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/ordinary/scalar_build/run.sh \
  swimlane ccec --device 0 --batches 256 --runs 1 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1
```

## 闭合门槛

- B1 同一 GM allocation 连续 5 轮全部 PASS；
- PlanAheadClosed native smoke 支持不依赖 Host cell reset 的同址重用，
  StreamingFuture 对该用法继续 fail-closed；
- 双 policy AICPU SO Host smoke 和 AArch64 反汇编门禁 PASS；
- converter 单元测试 27/27 PASS；
- B256 trace-free 与 structural capture 均通过 Plan/Build/Execute/
  FinalDrain 及 postprocess 检查。
