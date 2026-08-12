# PlanAheadClosed 发布与校验热路精简（2026-08-12）

本目录保留三份 B256 联合泳道图：逐 task cache maintenance 已删除但仍有
重复校验的历史基线、诊断校验移出热路的中间版，以及进一步删除 cell
Empty 预清零后的最终版。三图都使用 A5 device 0、1280 tasks、
`real-compute=6,28,4,1`，且 AICPU/AICore 时钟相关、语义、postprocess 和
零 drop 门槛全部通过。full trace 只用于结构归因，性能结论以 trace-free
20 轮中位数为准。

## 最终正常路径合同

```text
AICPU stage
  校验 callback 输入可安全序列化
  -> single-Pack canonical payload
  -> 直接复用 Pack 已构造的 header 作为 staged metadata
  -> 不预写 Empty control，不读回自检

AICPU publish（PlanAheadClosed）
  校验 staged magic / payload_lines / task_id 边界 / 最终 PA metadata
  -> patch final flags
  -> release atomic store(Published control)

AICPU close
  一次性逐 cell 校验 [0, N) 是完整 Published 连续前缀
  -> 发布 frontier=N / closed=N

AICore Scalar consumer
  return-ready atomic observe control
  -> DSB -> DCCI payload lines -> DSB -> 再读 control
  -> 校验 task-id / ABI / count / layout / published-lines envelope
  -> PA decode 校验实际消费的 tag / ref / adapter metadata
```

正常路径不再做以下诊断性重复工作：

1. AICPU publish 重扫自己刚 Pack 的整份 GM payload；
2. PlanAheadClosed 在 stage 和 publish 分别重复读取
   `fatal/closed/frontier/current/predecessor`；该 policy 在 Close 前没有
   consumer，最终连续性已由 Close 一次性验证；
3. AICore consumer 扫描不会参与解码的 inactive tag、tensor slot padding
   与末尾 cache-line padding 是否全零；
4. 对每个 PlanCell 先 `release-store Empty(0)`、再 `acquire-load` 读回 0。
   PlanAheadClosed 在 Close 前没有 consumer，实际 cell 最终都会由本轮
   `release-store Published` 覆盖；未用后缀以本轮 `frontier=N` 隔离。

`PA_RUNTIME_PLAN_DEBUG_FULL_VALIDATION=1` 会重新启用 producer 状态复核、
发布前完整 wire 校验和 consumer canonical-padding 全扫描。Host 事后 oracle
无论该开关为何值都保留完整校验。StreamingFuture 会与 consumer 并发，
其 stage/publish 状态复核、cache maintenance 和 barrier 也全部保留。

## 结构对比

三份 full trace 的 AICPU operation 计数如下。Calls 是源码级调用量，
不是合并后的 Perfetto event 数。

| 项目 | 7218us 历史图 | 5728us 中间图 | 5069us 最终图 |
| ---- | ------------: | ------------: | ------------: |
| AICPU operation records | 16,714 | 2,636 | 2,615 |
| `task_publish` acquire loads | 6,399 | 0 | 0 |
| `task_stage` acquire loads | 7,551 | 1,152 | 0 |
| Empty reset release stores（全 scope） | 1,280 | 1,280 | 0 |
| Empty readback acquire loads（全 scope） | 1,280 | 1,280 | 0 |
| `task_publish.payload_validation` | 1,280 | 0 | 0 |
| `task_publish` release Published | 1,280 | 1,280 | 1,280 |
| `task_publish` final-flags GM store | 1,280 | 1,280 | 1,280 |
| Close `[0,N)` Published loads | 1,280 | 1,280 | 1,280 |
| operation trace dropped | 0 | 0 | 0 |

5728us→5069us 共删除 2,561 次 atomic 调用：1,280 次 Empty store、
1,280 次 Empty readback 和 1 次 `cell[N]` 后缀哨兵读取。合并记录数减少
21 条。最终 `task_stage/backend_bind` 下均不存在 cell-control atomic；
`task_publish` 只剩 final-flags store 和 Published release store。

同一结构 trace 中，AICPU task phase 的变化为：

| phase | 7218us 总计 / p50 | 5728us 总计 / p50 | 总计变化 |
| ----- | -----------------: | -----------------: | -------: |
| `stage_payload` | 1824.079us / 1.017us | 1260.378us / 0.601us | -563.701us |
| `publish` | 1458.735us / 1.093us | 330.491us / 0.229us | -1128.244us |

这些数字含逐操作时间戳开销，只能说明事件归属和相对结构，不能代替
trace-free 性能。

## 保存的泳道图

### 历史基线：仍有发布前 payload 全量校验

`ordinary_scalar_plan_ahead_closed_b256_real_compute_6_28_4_1_release_publish_7218us_joint_full_atomic_swimlane.json`

- `plan_time=3873.053us`、`producer_exec=3814.350us`、
  `aicore_time=3345.037us`、`pipeline_e2e=7218.152us`；
- 16,714 条 AICPU operation record，`dropped=0`；
- SHA256：`79ffbcc8fbc62233fc6336fe95b694b8dc0797493fc649fd00f79b4e425d4200`。

### 中间版：诊断校验移出热路，但仍预清 Empty

`ordinary_scalar_plan_ahead_closed_b256_real_compute_6_28_4_1_normal_validation_5728us_joint_full_atomic_swimlane.json`

- `plan_time=2400.465us`、`producer_exec=2065.715us`、
  `aicore_time=3327.542us`、`pipeline_e2e=5728.069us`；
- 2,636 条 AICPU operation record，`dropped=0`；
- `task_publish.payload_validation=0`，publish acquire load=0；
- SHA256：`320c695eba8c2e212908f8be12306610406002e85ed53cd285783926f3e400b1`。

### 最终版：直接 Published，删除 Empty 预清零

`ordinary_scalar_plan_ahead_closed_b256_real_compute_6_28_4_1_direct_publish_5069us_joint_full_atomic_swimlane.json`

- `plan_time=1795.456us`、`producer_exec=1737.904us`、
  `aicore_time=3274.400us`、`pipeline_e2e=5069.908us`；
- 2,615 条 AICPU operation record，`dropped=0`；
- `task_stage.cell_control=0`、`backend_bind.cell_control=0`，仅保留
  1,280 次 `task_publish.store_release(Published)`；
- SHA256：`01c94c1189cb3f36232f6c5f3c645a23a8557dcfdf8eaf24c01b7eba4b78ac1d`。

## trace-free 性能

对比起点是删除逐 task cache maintenance 后、但仍保留上述重复校验的版本。
两组均为同一 A5 device 0、B256、`real-compute=6,28,4,1`、perf-clock、
同进程 20 轮中位数，且 20/20 execution/semantic/postprocess PASS。

| 指标 | 重复校验版本 | 校验精简版 | 删除 Empty 预清零 | 最后一步变化 |
| ---- | -----------: | ---------: | ----------------: | -----------: |
| Plan wall | 1149.669us | 1059.846us | 876.248us | -183.598us（-17.32%） |
| Producer exec | 1062.002us | 989.205us | 793.615us | -195.590us（-19.77%） |
| AICore wall | 2477.879us | 2473.651us | 2481.426us | +7.775us（+0.31%） |
| startup→FinalDrain | 2450.675us | 2443.734us | 2444.319us | +0.585us（+0.02%） |
| Pipeline E2E | 3626.891us | 3528.444us | 3363.339us | -165.105us（-4.68%） |

单独步骤之间只有数微秒到数十微秒，容易受轮间波动影响，因此不拿不同
20 轮样本的中位数强行相减做逐项精确归因。可以确认的是：task 数和
`6,28,4,1` workload 未变化。相对重复校验版本，最终端到端累计减少
263.552us（7.27%）；其中删除 Empty 预清零这一步减少 165.105us。

## 扩大审计结论

下列检查继续留在正常路径，不属于可直接删除的重复诊断：

- stage 对 callback `TaskArgs`、descriptor、output ref 和 payload 上界的
  检查：否则 Pack 本身可能越界或固化悬空指针；
- AICore 的 envelope、二次 control 观察和 PA 实际字段解码：这是跨
  AICPU→AICore 边界安全消费 GM payload 的最低条件；
- final flags 的 PA kind/group/batch/engine 一致性检查；
- Close 对 `[0,N)` Published 连续前缀的唯一终态检查；
- Host D2H 后的完整 canonical wire、输出和 TensorMap oracle。

`ValidatePaPlanSource`、通用 Pack 与 PA decode 内仍有少量字段级交叉检查，
但它们分别保护输入指针、通用 wire 和实际消费语义，当前没有第二个
整 payload 重扫。若继续优化，应先单独计时再合并接口，不能直接整块关闭。

## 复现命令

```bash
source /home/q00473782/Ascend/cann-9.1.0-weekly-20260708/cann-9.1.0/set_env.sh

tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/ordinary/scalar_build/run.sh \
  build-perf-clock ccec
tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/ordinary/scalar_build/run.sh \
  perf-clock ccec --device 0 --batches 256 --runs 20 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1

tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/ordinary/scalar_build/run.sh \
  build ccec
tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/ordinary/scalar_build/run.sh \
  swimlane ccec --device 0 --batches 256 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1
```

## 闭合门槛

- Runtime Plan protocol/PA adapter CPU 门槛 PASS；
- PlanAheadClosed 与 StreamingFuture 的默认、trace-on、debug-full-validation
  Host smoke全部 PASS；新增同地址、不清 cell control 的 14-task→5-task
  long-to-short 门槛 PASS；
- 两种 policy 的 AArch64 SO、导出符号与 cache/release 反汇编门禁 PASS；
- B256 trace-free 20/20 PASS；
- 最终 joint full atomic/DCCI swimlane：AICPU operation `dropped=0`，
  AICore generic/atomic/DCCI record 全部闭合。
