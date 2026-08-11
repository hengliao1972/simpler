# AICPU Plan publish 不保留 DSB 后删除 ISB

**Date**: 2026-08-11

**Verdict**: dropped

## Question

PlanAheadClosed 的 task publish 原先在 payload clean 后执行 DSB，在
control store/clean 后再执行 DSB+ISB。Arm 文档将 DSB 定义为等待
cache-maintenance 完成的屏障，ISB 主要用于指令流和 context
同步。因此一个自然候选是保留 `dc cvac + dsb sy`，删除
`PublishControl()` 末尾的 `isb`。

## What was tried

在
`tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/ordinary/scalar_build/aicpu/aicpu_plan_adapter_bridge.cpp`
中仅删除 `PublishControl()` 对 `InstructionBarrier()` 的调用，control
store、`dc cvac`和 `dsb sy` 全部保持。使用真实 A5 device 0、
B256、1280 tasks、`real-compute=6,28,4,1`、PlanAheadClosed、
trace-free，连续执行两组 10-run 测量。

对照是同一阶段已收缩生产期 frontier 发布、但仍保留 ISB 的
10-run 中位数：`producer_exec=1824.717us`、
`pipeline_e2e=4401.727us`。

## Result

| 版本 | Producer exec | Pipeline E2E | 语义 |
| ---- | ----: | ----: | ---- |
| 保留 ISB | 1824.717us | 4401.727us | 10/10 PASS |
| 删除 ISB，第 1 组 | 1997.112us | 4574.664us | 10/10 PASS |
| 删除 ISB，第 2 组 | 2008.363us | 4557.886us | 10/10 PASS |

两次复测功能都正确，但 producer 分别回退 172.395us 和
183.646us，pipeline 分别回退 172.937us 和 156.159us。

## Why not (now)

在当前 A5/AICPU 上，“语义上可能多余”没有转化为性能收益，两次
独立测量都显著回退。因此该改动已撤回，当前仍保留 ISB。
后续已采用的优化是另一条路线：PlanAheadClosed 不再每 task
调用整个 publish barrier 序列，而在 Close 处统一收口；但真正执行
`PublishControl()` 的少数最终 control 仍保留 DSB+ISB。

## When to reconsider

- AICPU 微架构或 CANN toolchain 变化后，同样的双轮 A/B 能够稳定证明
  producer 不回退；
- 先用硬件计数器或更细的 AICPU 取时证明当前回退是可消除的
  独立机制，而不是 ISB 对当前指令/cache 流的实际影响。

## References

- [Arm: Memory access ordering in the Arm Architecture](https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/memory-access-ordering-part-3---memory-access-ordering-in-the-arm-architecture)
- [Arm: DPDK optimization on Arm](https://developer.arm.com/community/arm-community-blogs/b/tools-software-ides-blog/posts/dpdk-optimization-on-arm?tempkey=ea4df3f9-10dc-4806-b443-149059ff949d%3Ftempkey%3Dea4df3f9-10dc-4806-b443-149059ff949d)
