# AICPU Plan producer/backend（ordinary + Scalar Build，S1）

本目录只验证一件事：真实 PA orchestration 能否在 AICPU 上以原始
`Begin → callback → Finish` 顺序生成一份通用、可随机读取的 Plan。它不执行
Materialize、TensorMap、Build 或 kernel，也不把 PA task 公式放到 Host。

关键合同：

- `Begin` 按真实 callback 到达顺序分配连续 `task_id`；
- Alloc 身份来自 Alloc API，QK/SF/PV/UP 来自 `MixedKernels` 函数注册表；
- 禁止用 `task_id % 5`、batch 公式或 Host 计划推导身份；
- `Finish` 立即复制真实 `L0TaskArgs` 的 ABI 字节，并通过
  `aicpu_plan_pa_adapter.h` 序列化；descriptor 指针不会跨过 Finish 生命周期；
- 一个 AICPU 私有 pending cell 用于等待下一次真实 Begin，从而确定
  `UP.has_following_group` 与 batch 尾；返回给 orchestration 的
  `SharedTaskOutputs` 仍然立即由 `(task_id, output_count)` 构造；
- payload 与 control 均使用 ordinary store + exact clean；AICore consumer
  负责 return-ready atomic observe 和 invalidate。

## 同一 Plan storage 跨 run 复用

正式 Host 会在每轮开始时用 `aclrtMemset` 清零同一块
`RuntimeTaskPlanCell[]` GM。Host DMA 不会 snoop AICPU cache；上一轮由
AICPU 发布并执行 `dc cvac + dsb sy + isb` 的 cell control 可能仍以
clean-valid 形式驻留，因此不能直接用普通 load 判断本轮是否 Empty。

本机没有已验证的地址型 pure-invalidate 指令：AICPU EL0 的 `dc ivac`
在当前运行环境会因 UCI 权限被静默处理为 NOP。这里复用仓库已经用于
Host-DMA → AICPU 的精确协议：

```text
对每个 cell control 的第一条 64B line 发出 dc civac
  -> 统一 dsb sy
  -> isb
  -> 才逐 cell LoadControl 并要求 Empty
```

`dc civac` 在这里安全的前提是明确的，而不是把它冒充 pure invalidate：
上一轮唯一 control 写路径已经完成 `ordinary store -> dc cvac -> dsb sy ->
isb`，此后 AICPU 不再 ordinary-store 该 line，所以复用入口只允许
clean-valid；本轮 AICore 尚未启动，也没有并发 writer。若未来新增任何
可能令 cell control 重新变 dirty 的路径，必须先修改该协议，不能继续沿用
此处的前提。

验证：

```bash
source .venv/bin/activate
tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/ordinary/scalar_build/aicpu/build_smoke.sh
```

门槛包括：

1. x86 Host 真实 `dlopen(RTLD_NOW | RTLD_LOCAL)` PA orchestration SO；
2. 在相同 control/cells 地址连续运行两轮真实
   `bind -> aicpu_orchestration_entry -> close`；每轮 G1+G2 两 batch都应
   生成 14 个 PlanCell；Host 门槛用于锁死清零、复用和重新发布顺序，AICPU
   cache 行为仍必须由 A5 同地址 `--runs 2` 验证；
3. 每个 cell 重新执行公共 payload 校验，核对 engine/function/output/ref；
4. `readelf` 核对入口与 backend 导出，并拒绝未闭合的 `dist_*`/bridge 符号；
5. 用 CANN HCC 生成 AArch64 SO，并重复 ELF 导出/未解析符号门槛；
6. 反汇编最终 AArch64 `aicpu_plan_adapter_initialize`，要求第一条 control
   acquire load 之前严格出现 `dc civac -> dsb sy -> isb`。

2026-08-10 的 A5 同地址复用门槛使用完整 CCEC swimlane 产物：

```bash
timeout --foreground 180s \
  tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/ordinary/scalar_build/run.sh \
  run ccec --device 0 --batches 1 --runs 2 \
  --winner-workload scalar-nop --nop-count 0
```

两轮在同一 Host 进程、同一份 128B 对齐 GM Plan allocation 上连续执行，
每轮 Host 都对同一 cells 地址执行 `aclrtMemset`。两轮均得到 5 个 task，且
精确满足 `frontier=closed=release=5`、`build_next=101`、
`workers_done=96`、`fatal=0`，完整 Plan/Build/Execute/FinalDrain 语义门槛
全部 PASS。该门槛只证明跨 run cache/复用正确性；首轮 Path-A 冷启动与
第二轮热启动的 wall time 不作为性能 A/B 结论。

perf-clock 也允许在继续严格关闭泳道、Atomic、PMU 和 phase-profile 的
前提下运行多轮。未指定 `--runs` 时使用 Host 默认值 5；显式 `--runs N`
用于在同一进程、同一 GM allocation 上区分首轮 Path-A 冷启动和后续热态。
当前 ABI2 产物的 A5 结果如下：

```bash
tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/ordinary/scalar_build/run.sh \
  build-perf-clock ccec
tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/ordinary/scalar_build/run.sh \
  perf-clock ccec --device 0 --batches 1 --runs 5 \
  --winner-workload scalar-nop --nop-count 0
tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/ordinary/scalar_build/run.sh \
  perf-clock ccec --device 0 --batches 256 --runs 5
```

| workload | 运行方式 | startup→FinalDrain | plan_time | producer_exec | aicore_time | pipeline_e2e | 结果 |
| ---- | ---- | ----: | ----: | ----: | ----: | ----: | ---- |
| B1 scalar-nop=0 | 同进程 run 1（冷启动） | 174.179us | 6011.888us | 148.512us | 610.054us | 6622.200us | PASS |
| B1 scalar-nop=0 | 同进程 5 轮的中位数 | 156.880us | 168.773us | 103.489us | 207.296us | 371.006us | 5/5 PASS |
| B256 real-compute 6,28,4,1 | 5 个独立进程首轮的中位数 | 2456.439us | 8546.751us | 2672.490us | 2924.193us | 11475.583us | 5/5 PASS |
| B256 real-compute 6,28,4,1 | 同进程 5 轮的中位数 | 2460.846us | 2690.859us | 2626.289us | 2491.516us | 5180.220us | 5/5 PASS |

同进程 B1/B256 的第一轮仍包含 Path-A 冷启动；表中的五轮中位数是 Host
对全部 5 轮计算的正式口径，而不是删除首轮后另造口径。B1 五轮均保持
`frontier=closed=release=5`、`build_next=101`、`workers_done=96`、
`fatal=0`；B256 五轮均保持
`frontier=closed=release=1280`、`build_next=1376`、
`workers_done=96`、`fatal=0`。这些数据同时说明：cache 复用正确性已经
闭合，但当前 B256 完整 pipeline 仍远高于 1ms 目标。

Host smoke 本身仍只负责 producer 协议；上述 A5 证据来自同一源码生成的
正式 CCEC producer + Scalar Build + Execute 完整路径。
