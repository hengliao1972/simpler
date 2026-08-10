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

验证：

```bash
source .venv/bin/activate
tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/ordinary/scalar_build/aicpu/build_smoke.sh
```

门槛包括：

1. x86 Host 真实 `dlopen(RTLD_NOW | RTLD_LOCAL)` PA orchestration SO；
2. 真实调用 `aicpu_orchestration_entry`，G1+G2 两 batch 应生成 14 个 PlanCell；
3. 每个 cell 重新执行公共 payload 校验，核对 engine/function/output/ref；
4. `readelf` 核对入口与 backend 导出，并拒绝未闭合的 `dist_*`/bridge 符号；
5. 用 CANN HCC 生成 AArch64 SO，并重复 ELF 导出/未解析符号门槛。

该 smoke 尚未接入 standalone CCEC worker，也没有 A5 端到端性能结论。
