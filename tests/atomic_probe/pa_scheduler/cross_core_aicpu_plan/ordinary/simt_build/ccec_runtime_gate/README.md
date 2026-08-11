# ordinary SIMT 真实状态映射门槛

本目录只验证窄 SIMT Build leaf 能直接消费 ordinary Scalar 已有的生产状态，
尚未接入正式 `RunScheduler`、Host launch 或 manifest。

`simt_real_state_runtime.h` 只保存一个 `SchedulerState *`，不分配也不定义
第二份 GM state。它直接映射：

- `SharedTensorMapSidecar` 的真实 ordinary bucket/slot、`SharedOutputCell`、
  `SharedWriterHistoryCell` 和 8-shard heap control；
- `SharedClaimTournamentTask::root.insert_completion`；
- 真实 `TaskCell::vend/flag`；
- `SharedExecCell` 与 `SharedExecFatalControl`。

SIMT 四个 Build leader 使用 build owner `0..3`。`build_owner` 与
`execute_owner` 是独立字段，但真实 A5 execute/Host policy 要求前者
`<96`；Build 角色由独立 leader report 辨识，不扩展正式 policy。算子
业务校验通过 `RoutePolicy` 注入；公共 adapter 不解释 PA TaskKind，也不从
task id 推导业务身份。每张 ticket 在进入 Build 前必须先调用
`BindTask(task_id)`。

## TensorMap 与内存发布

ordinary 预检和读取直接调用真实 generic helper：

- preflight：`SharedCheckPreparedTaskAppend`，固定 no-reclaim；
- lookup：`SharedLookupRegion<..., NoReclaim=true>`，保留 `[N-H,N)` 下界、
  absolute seq 双检以及 control → DCCI → payload → control 的 reader 顺序。

写侧保持真实 ring 的 `head/tail/absolute-seq` 算法和返回型原子旧值校验，
但不能直接调用 Scalar 的 ordinary-store + clean-out 实现。SIMT payload 固定
逐 64-bit `asc_stcg`，随后 `asc_threadfence`，最后才发布 seq 和 tail；writer
不执行 `asc_dcci_single`。descriptor、writer history 和 exec payload 使用
同一 publication 方向。metadata-only completion 不走 stcg，而是对真实同一
64B `TaskCell` 依次执行 atomic vend、fence、atomic flag，并验证两个旧值。

## 门槛

```bash
ordinary/simt_build/cpu/build_real_state_runtime.sh
ordinary/simt_build/ccec_runtime_gate/build.sh
```

CPU 门槛以约 1 GiB 的真实 `SchedulerState` 虚拟映射验证 metadata Build、
ordinary append/lookup、8-shard heap 以及所有返回地址都落在真实 state 内；
包含 O2、ASan 和 UBSan。

CCEC 门槛用 `dav-c310-vec -O3 -Werror` 生成真实 machine object 和优化 IR，
验证完整 `BuildCanonicalPlanTask` 已实例化，并锁定 s32/s64/u64 atomic、
`asc_stcg`、DCCI、fence、clock 与 SIMT entry。它证明 scalar 数据类型和
generic helper 能以 `__simt_callee__` 身份完成 machine lowering；不等价于
A5 板上并发内存 litmus，也不代表正式双 TU 调度入口已经完成。
