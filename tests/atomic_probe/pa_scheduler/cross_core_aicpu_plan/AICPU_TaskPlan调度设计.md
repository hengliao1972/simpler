# AICPU TaskPlan 调度设计

本文定义 `cross_core_aicpu_plan` standalone 的候选架构合同。实际进展见
[《AICPU TaskPlan 实现过程》](AICPU_TaskPlan实现过程.md)。

## 1. 目标与当前边界

目标是让 AICPU 在 device 上执行真实 orchestration callback，生成全局唯一的
canonical TaskPlan；AICore 不再 96 路重放 callback，而是按 task id 领取 Build，
再由 AIC/AIV 领取已构建任务执行。

这条路径必须满足：

- Host 只上传 orchestration SO、输入和运行期容量，不枚举 task identity；
- AICPU 必须调用真实 orchestration SO，不能用 PA 公式重建任务表；
- Plan 只做 Describe，Materialize、TensorMap、Fanin 和 executable payload
  仍属于 Build；
- Build owner 与 Execute owner 解耦；
- shared TensorMap 仍严格按 task id 完成 `N-1 -> N` 插入；
- 公共协议不得依赖 PA 的 task 数、TaskKind 排列或 Host 预制答案。

当前 S0 公共协议已有 CPU、CCEC 和独立 A5 可见性探针证据，但还没有
真实 AICPU Plan backend、AICore Build 主路径或 PA 端到端结果。文中的
性能目标是后续验收方向，不是已实现结论。

### 1.1 目录与两个正交维度

```text
cross_core_aicpu_plan/
├── common/                     # Scalar/SIMT、ordinary/DAG 共用 Plan ABI
├── ordinary/
│   ├── scalar_build/           # 普通严格顺序 TensorMap + Scalar Build
│   └── simt_build/             # 普通严格顺序 TensorMap + SIMT Build
├── dag/
│   ├── scalar_build/           # DAG TensorMap + Scalar Build
│   └── simt_build/             # DAG TensorMap + SIMT Build
└── test_record/
```

`ordinary/DAG` 描述 TensorMap metadata 的提交协议；`Scalar/SIMT` 描述
谁消费同一份 canonical Plan 并执行 Materialize/Fanin/Build。两条轴必须
正交：换成 SIMT Build 不能顺手改变 TensorMap 语义，换成 DAG 也不能另造
Host Plan 或 PA task 公式。四种模式共用 AICPU producer、PlanCell wire、
错误合同和相同的 startup→FinalDrain 性能边界。

## 2. 首版架构与真实时序

### 2.1 首版固定为 Plan-ahead

首版不做 Plan/Build 流式重叠。AICPU 必须先完整执行一次 orchestration，
发布所有 PlanCell 并封口，再唤醒 AICore Build worker：

```text
Host
  上传 orchestration SO / input / PlanStorageRef
  启动 AICore，随后启动 AICPU

AICPU（唯一 Plan producer）
  dist_engine_register
  装载并校验真实 orchestration SO
  绑定本轮 PlanProducerContext
  执行 orchestration entry 一次
    -> callback N 完整序列化 PlanCell[N]
    -> clean 实际 payload cache lines
    -> 发布 PlanCell[N]
    -> 推进无缺口 planned_frontier
  发布 closed_task_count == planned_frontier
  唤醒 AICore

AICore Build（首版为 Scalar）
  从 build_next 领取 N < closed_task_count
  acquire / invalidate / validate PlanCell[N]
  Materialize
  等待 TensorMap completion[N-1]
  发布 task N writer metadata
  发布 TensorMap completion[N]
  查询 producer < N 的 fanin
  构建并发布 SharedExecCell[N]

AIC/AIV Execute
  领取兼容的 Built task
  执行 kernel
  发布 completion / DONE
  FinalDrain 收口
```

Plan-ahead 首版中，Build 看到的必须是已封口的稳定计划：

```text
planned_frontier == closed_task_count == final_task_count
```

`planned_frontier` 仍用于证明 PlanCell 按无缺口前缀发布，但首版 Build 不在
Plan 尚未封口时领票。流式 frontier、暂时追平和重访逻辑不进入首版。

### 2.2 AICPU 必须执行真实 orchestration

当前 FDWIC AICPU 只做 `dist_engine_register` 和 worker handoff，不执行
orchestration。新路径需要复用仓库已有的 AICPU SO loader，并在 FDWIC
AICPU DSO 中补齐 orchestration 所调用的 `dist_*` Plan backend。

Plan backend 保持现有 orchestration API 签名，但改变其 device 实现语义：

- Begin 按 callback 顺序为任务分配唯一 task id；
- Finish 将 MixedKernels、Tensor/TensorCreateInfo、scalar、显式依赖和
  symbolic output reference 序列化进 PlanCell；
- orchestration entry 正常返回后才封口 Plan；
- SO 加载、config、符号或 descriptor 校验失败时 fail-closed，不唤醒
  AICore 进入半发布 Plan。

Host 不得为这条路径设置 PlanCell 内容。

## 3. 公共 Plan ABI 与内存合同

S0 草案位于
[公共 Plan 协议头](common/aicpu_plan_protocol.h)。其当前结构边界如下：

- `RuntimeTaskPlanHeader` 是 64B，它只是 payload 的公共头；
- 完整无指针 payload 上限是 4416B，共 69 条 64B cache line；
- payload 容纳公共 header、最多 32 个 Tensor/TensorCreateInfo 或 output
  reference、16 个 scalar 和 16 条显式依赖；
- 每个 tensor 使用固定 16-word canonical slot：TensorDesc 使用全部
  16 word，CreateInfo 使用 8+8 zero，完整 16B OutputRef 使用 2+14 zero；
- `RuntimeTaskPlanCell` 当前 stride 是 4608B：128B atomic-only control、
  4416B payload 以及 64B 尾部对齐；
- PlanCell 数组按本轮容量在 GM 动态外置分配，不固定追加到 1GiB
  SchedulerState；
- `RuntimePlanStorageRef` 独占并对齐到 128B，只携带 GM 基址、容量、cell
  stride 和 ABI version；Host 不填 task identity。
- `RuntimePlanControl` 的 frontier、close、build-next、workers-done、release
  和 fatal 分别独占一个 128B 原子冲突单元。

因此，“64B TaskPlan 描述符”是错误说法。64B 只能指代 header，不能
代表支持通用 Build 的完整任务语义。

生产接线时不应再维护第二套并行的 payload 布局和 pack/validate
逻辑。应对齐或提取复用
`runtime/dist_engine/common/cross_core_simt_request_protocol.h` 中已有的
4416B request payload 与序列化机制。AICPU Plan 可保留已验证的 128B
atomic 隔离 stride，但 payload 容量、Tensor/scalar/explicit-dependency
表达和校验规则必须只有一个权威来源。现有 SIMT header 中的
`function_address` 不能被盲目复制进跨 AICPU/AICore ABI；公共 Plan 保存
logical function id，由 Build 解析本地可执行地址。

### 3.1 无指针边界

Plan 中不允许保存：

- AICPU callback 栈地址；
- `L0TaskArgs` 中指向临时 Tensor/TensorCreateInfo 的指针；
- `explicit_deps_`、`error_msg` 等运行期指针；
- AICPU 虚拟地址下的 kernel function pointer。

producer 必须在 callback 返回前复制被引用对象的值语义内容。kernel 优先保存
logical function id，Build 再在 AICore 本地 runtime 中解析可执行地址。

### 3.2 A5 发布与获取

A5 Scalar 之间不假定 cache coherence。每个 PlanCell 必须遵守：

```text
AICPU producer:
  ordinary store 完整 payload
  -> exact cache clean(payload_lines)
  -> store barrier
  -> publish cell control

AICore consumer:
  return-ready observe cell control
  -> exact DCCI invalidate(payload_lines) + DSB
  -> validate header/task_id/ABI/payload_lines
  -> ordinary load payload
```

control 与 payload 分离 cache line。atomic-only line 不存普通 payload，也不执行
可回写 stale dirty snapshot 的 payload clean-out。

## 4. Build、TensorMap 与 Execute 不变量

1. Plan 是全局唯一、task-id 稠密且无缺口的 immutable descriptor 序列。
2. Build task id 只来自 runtime `build_next`，任务语义只来自已获取的
   PlanCell；不调用 PA random-access 公式恢复 args。
3. 唯一 Plan producer 加中央 Build ticket 已能保证每个 task 只 Build 一次，
   因此该模式不需要 per-task Build Tournament。
4. Plan 顺序不等于 TensorMap 已发布。Materialize 可并行，但 task N
   只能在观察到 completion[N-1] 后发布 writer metadata；没有 writer 的
   task 也必须推进 completion[N]。
5. task N 的 TensorMap 查询只接受 `producer < N`，不能因 Plan 已预先存在
   而读取未来 writer。
6. Build 只发布 executable payload，Execute owner 独立领取兼容 engine 的 Built
   task；同一 Scalar 可同时成为 Build owner 和 Execute owner，但协议不绑定两者。
7. 任一容量、ABI、descriptor、TensorMap 或 publication 错误都必须发布
   fatal 并让所有 worker 收口，不能留在半发布状态死等。

### 4.1 尚未实现的公共边界

下列能力不在首版声明范围内：

- Plan/Build 流式重叠、frontier 暂时追平、Build 重访和反压；
- PlanCell 复用、generation、ABA 和 ring reclaim；
- 多 AICPU producer 的分区与有序合并；
- 完整 Joint/MixedKernels、multi-core launch 语义和所有 engine 组合；
- 依赖 kernel 运行结果才能继续生成后续 Plan 的 result-driven
  orchestration；
- Submit 返回时必须立即取得 materialized `TaskOutputTensors::get_ref()` 的
  同步输出 API。

PA 首例只能使用已有 symbolic `SharedTaskOutputs/FdwicOutputRef` deferred
路径。遇到上述不支持语义时必须 fail-closed，不能伪造 Tensor descriptor。

## 5. 实现顺序、验证与性能口径

实现顺序固定为 ordinary Scalar，ordinary SIMT，最后才进入 DAG：

| 阶段 | 目标 | 当前状态 |
| ---- | ---- | -------- |
| S0 | 固定 Plan ABI、AICPU/AICore 发布合同和正确性门槛 | 已闭合 |
| S1 | 接通真实 AICPU orchestration SO 和 Plan backend | 独立 producer 已闭合，A5 正式 launch 待接 |
| S2 | ordinary Scalar Build，CPU 后 A5 B1/B256 | 独立 CPU 状态机已闭合，完整生产路径待接 |
| S3 | 在同一 Plan ABI 上替换为 ordinary SIMT Build | 未实现 |
| S4 | 证明 ordinary 闭合后迁移 DAG Scalar Build | 未实现 |
| S5 | 在 DAG 上替换为 SIMT Build | 未实现 |
| S6 | 在正确性与观测闭合后做性能收敛 | 未开始 |

S0 至少需要以下证据：

- CPU 布局、payload 边界、无指针序列化、稠密 task id 和唯一 Build 领取；
- CCEC AIC/AIV 对 control dependency、atomic、DCCI 顺序的静态检查；
- A5 多 cache-line payload 的发布前不可见、发布后完整可见；
- SO entry/config 和 `dist_*` 符号实际闭合，不能只依赖 `RTLD_LAZY`；
- TensorMap 含零 writer task 的严格 completion 顺序；
- B1、B256、短尾、INOUT 和至少一个非 PA symbolic-output 边界例。

性能只在对应功能门槛闭合后记录：

- 权威结果使用无泳道构建，起点是本轮最早的 device worker startup，
  终点是最后一个 AICore 完成 FinalDrain；
- 窗口必须包含 AICPU SO 检查与调用、Plan 生成与发布、Build、
  TensorMap、Execute 和 FinalDrain；
- SO 首次加载与 callable cache hit 需分组记录，不能只报更快的一组；
- 泳道只用于定位 Plan/Build/Execute/Atomic/DCCI 分布，不与无泳道绝对值
  相减；
- 对照必须使用同一 PA 输入、真实计算负载和 startup→FinalDrain
  边界。

旧 Host 预制/PA 公式方案的约 0.82ms 不是等价基线。当前没有 A5 结果，
也不能声称已达到 B256 小于 1ms。
