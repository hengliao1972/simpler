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

当前 ordinary + Scalar 首版已在 CPU 和 A5 闭合：真实 AICPU
orchestration callback 生成 canonical Plan，96 个 Scalar 使用中央 ticket
完成 Build，ordinary TensorMap 仍严格按 task id 插入，随后 AIC/AIV
执行并由 FinalDrain 收口。当前 B1 warm pipeline 已低于 1ms，但 B256
仍为多毫秒，所以完整目标尚未达成。ordinary SIMT 已完成四 leader 协议、
通用 ordinary writer、窄完整 Build 的 CPU 与 CCEC machine-code 门槛，
正式 runtime/A5 尚未接通；两种 DAG 模式尚未实现。

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
  启动 AICPU，同步等待 Plan 完整封口
  再启动 AICore

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
  结束 AICPU 阶段

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

当前正式 Host 实现使 AICPU 与 AICore 使用同一 stream，并在
AICPU launch 后显式 synchronize，因此 Plan-ahead 是真实的两段串行边界，
不是“AICore 早启动但在 GM 中轮询 Close”。这个选择先简化正确性，
同时意味着当前 `pipeline_e2e ≈ plan_time + aicore_time`。

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

当前 A5 owner request 只携带 Plan storage 引用、容量、输入 Tensor
metadata、`context_lens` 和 scalar。task 数、task kind、task identity 和
dispatch plan 均不是 Host 输入。Host 可在 Plan 生成后使用独立 PA
oracle 校验 D2H 结果，但 oracle 不进入 device 调度数据流。

## 3. 公共 Plan ABI 与内存合同

S0 草案位于
[公共 Plan 协议头](common/aicpu_plan_protocol.h)。其当前结构边界如下：

- `RuntimeTaskPlanHeader` 是 64B，它只是 payload 的公共头；公共 wire
  ABI 当前为 v2，header 的 16bit `adapter_data` 由算子 adapter 定义，
  PA 用它显式携带真实 Alloc callback 的 `batch_start`；
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
逻辑。现有
`runtime/dist_engine/common/cross_core_simt_request_protocol.h` 是
“AICore replay producer -> VF consumer”的另一份 request wire：control
只有 64B、payload 使用变长 tensor slot、引用被压缩为 1 word，并携带
AICore `function_address`；它缺少 Plan v2 的 `adapter_data`。因此只能复用
其中已经验证的 atomic/DCCI、VF leader 和 publication 实现经验，不能把
canonical Plan 再复制成该 request，也不能把两种 cell 强转互换。

ordinary Scalar 与 ordinary SIMT 都必须直接消费本目录唯一权威的
`RuntimeTaskPlanCell` ABI v2。公共 Plan 只保存 logical function id；Build
在 AICore 本地解析可执行地址。SIMT 可达 helper 即使受
`__simt_callee__` 限制而需要单独实现解码代码，也必须逐字段服从同一
header/layout/canonical-zero/OutputRef 定义，并由 CPU、CCEC 与 A5 门槛
交叉锁定，不能形成第二套 wire ABI。

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

同一 Plan storage 跨 run 复用时还必须处理 Host DMA 与 AICPU cache
不一致：Host 清零同一 GM 后，AICPU 在读取 cell control 前先丢弃上一轮
clean-valid control 行，并在统一 `dsb sy + isb` 后再检查 Empty。当前实现
使用仓内已验证的 `dc civac`；它成立的前提是上一轮唯一 producer 写已经
`dc cvac + dsb` 变成 clean，之后没有 ordinary writer，且新一轮 AICore
尚未启动。不得把该协议扩展到可能含 dirty ordinary payload 或并发 atomic
writer 的 cache line。

## 4. Build、TensorMap 与 Execute 不变量

1. Plan 是全局唯一、task-id 稠密且无缺口的 immutable descriptor 序列。
2. Build task id 只来自 runtime `build_next`，任务语义只来自已获取的
   PlanCell；不调用 PA random-access 公式恢复 args。PA 的 `batch_start`
   同样来自 PlanCell 的显式 `adapter_data`，固定 offset 只能做一致性校验。
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

### 4.1 ordinary Scalar 首版的精确收口

Plan 完整 Close 后，96 个 Scalar 只缓存一次稳定的 task 数 `N`，
然后重复对同一 `build_next` 做返回型 FetchAdd。成功 ticket 为
`[0, N)`，每个 worker 最后再消费一个越界 ticket，所以终态必须是：

```text
build_next == N + 96
workers_done == 96
build_release == N
fatal == 0
```

中央 ticket 只解决“每个 PlanCell 只 Build 一次”，不放宽 TensorMap
顺序。task N 可以在串行链外完成 Plan acquire 与 Materialize，但 writer
metadata 发布仍必须等待 completion[N-1]，发布后再推进 completion[N]。
完成所有 Build 后才发布 release，当前 Execute 是 post-build 扫描，并未实现
Plan/Build/Execute 流式重叠。

### 4.2 CCEC direct-entry 的核型本地状态

ordinary Scalar 保留 direct PlannedBuild，不恢复旧 callback split。但 CCEC 下
`LocalStats` 会跨 `noinline` Build/Execute helper 传递引用，不能放在普通栈上。
因此 AIC 和 AIV 各使用一份 role-specific `[[block_local]] LocalStats`，
并在每轮入口整体复位。CPU 实现继续使用栈对象。这只是 CCEC
存储约束，不改变 non-split `WorkerResult` ABI、Plan ticket 协议或调度语义。

### 4.3 ordinary SIMT 首版的精确合同

ordinary SIMT 继续使用完全相同的 AICPU closed Plan 和 Plan-ahead
生命周期。首版只有一个物理 builder VF：block0/AIV0 调用
`async_invoke` 启动 128 个 SIMT thread，组成 4 个 warp；每个 warp 只有
lane0 是 active Build leader。VF 完成后，该 AIV0 Scalar 仍回到普通 AIV
Execute/FinalDrain，因此 Execute 拓扑保持 32 AIC + 64 AIV，而不是为了
Build 永久牺牲一个执行核。

首个功能版本让 4 个 leader 共用 `build_next.FetchAdd(1)`：

```text
AICPU: frontier == closed == N
  -> 4 个 VF leader 动态领取 [0, N)
  -> 每 leader 完成当前 task 后才领取下一 task
  -> 每 leader 首次取得越界 ticket 后报到一次
  -> build_next == N + 4
  -> build_workers_done == 4
  -> completion[N-1] == N-1（N>0）
  -> 唯一 last leader 发布 build_release == N
```

该选择不是把 Scalar 的 96 核竞争原样搬进 VF：竞争人口从 96 降到 4，
不再存在 per-task Tournament。首版保留动态 ticket 的原因是它不依赖
task 类型和单 task 成本，面对其他算子仍可自动做 leader 负载均衡，并且
直接复用已经闭合的 closed-Plan 领取/报到/release 协议。正常原子调用数
为 `N+4` 次 build-next FetchAdd、4 次 arrival FetchAdd 和 1 次 release
publish。builder 启动前还会一次性读取 closed/frontier/fatal 并缓存
N；每个 PlanCell 的前后两次 control 观察与严格 TensorMap
completion 另行计数，不能与 build-next 控制原子混算。
全局 `build_workers_done==4` 只能证明“报到了四次”，不能单独
证明“四个不同 leader 各一次”。首版不为此增加第二个 GM
atomic bitmask；该 exactly-once 身份由固定 lane0 拓扑、每 leader 持久局部
arrival 状态和一次性退出结构共同证明，CPU 与 CCEC 门槛必须锁定这三点。

静态 residue（leader `L` 处理 `L, L+4, ...`）可完全消除 build-next
原子，现有 standalone/生产 SIMT 已证明该分工形式可编译运行；但它会把
负载均衡假设交给 task 序列。它保留为功能闭合后的独立性能候选，必须在
同一 Plan、同一算子输入下同时验证尾部不均衡、严格插入等待和端到端
收益，不能在首版中既改变分工又改变 Build 实现而失去可归因性。

每个 SIMT leader 的 task 内部顺序固定为：

```text
atomic observe PlanCell control
  -> 精确 invalidate 已发布 payload lines
  -> 再读 control 并校验 ABI/task/layout/tag/ref/canonical-zero
  -> 串行链外 Materialize、heap reserve 和 output descriptor 发布
  -> 等待 completion[N-1]（N=0 跳过）
  -> 发布 ordinary/symbol writer metadata
  -> 零 writer task 也发布 completion[N]
  -> 只查询 producer < N 的 fanin
  -> 发布 SharedExecCell 或完成 metadata-only task
```

这里的 Register 不能直接复用当前 ordinary Scalar 的
`FinishSharedWinnerSubmitBody`。该函数为 PA 首例保留了一个已测快路径：
`ValidatePreparedPaWriterShape` 要求 `ordinary_count==0`，并在取得 turn 后
只提交 UP 的三个 symbol writer。它能覆盖当前 PA，但不是普通 TensorMap
的通用合同。SIMT 完整实现必须显式走：

```text
PrepareSharedTaskWriterDelta
  -> WaitForSharedTaskInsertTurn
  -> PublishSharedTaskWriterMetadata
       -> SharedCheckPreparedTaskAppend
       -> CommitPreparedSymbolSharedWriterIntentSet
       -> SharedAppendPreparedTask
  -> HandoffSharedTaskInsertTurn
```

因此 `ordinary_count>0` 是 SIMT 模式的强制功能门槛，不能只用 PA
Case1 的零 ordinary writer 结果宣称泛化完成。

首版在全部 4 个 leader 完成 Build 后才允许 Execute。这样
`SharedExecCell::Empty` 仍可在 release 后无歧义地解释为 metadata-only；
若将来做 Build/Execute overlap，必须先增加“尚未 Build”与
“metadata-only terminal”的显式区分，不能让 Execute 扫描把暂时 Empty
静默跳过。

正式 CCEC 使用双 TU，而不是把 Scalar 调度头整体重定义成
`__simt_callee__`：

```text
AIV0 mixed entry / VF TU
  -> 校验 closed Plan 与运行身份
  -> async_invoke 128-thread VF
  -> 4 个 lane0 leader 执行窄 canonical Build
  -> V/S join（失败时也必须 join）
  -> 调用 Scalar scheduler continuation TU

其他 95 个 Scalar
  -> 初始化自身 Execute token/stats
  -> 在 startup/release 边界等待 AIV0

AIV0 continuation
  -> 作为第 96 个 Scalar 到达 startup
  -> 全员观察 build_release
  -> 32 AIC + 64 AIV Execute/FinalDrain
```

VF 前的入口必须先取得 config/Plan 身份；VF fatal 后也必须进入 continuation，
否则其余 95 核会永久卡在 startup。perf-clock 起点与 Build trace 起点必须在
VF 之前记录，并经独立 GM report 交给 continuation；不能从 join 后开始计时
而隐藏 SIMT Build。现有 Exec/Host policy 要求 `build_owner < 96`，所以四个
leader 的 Build owner 固定为 `0..3`；Build owner 与 Execute owner 是两个
独立字段，不表示同一物理角色。统计/trace 使用独占 cacheline，join 后由
AIV0 invalidate 后归并，不能写入 worker 0..3 的 Scalar `WorkerResult`。

writer publication 的 A5 合同固定为：

```text
writer: asc_stcg(payload words) -> asc_threadfence -> atomic control publish
reader: return-ready atomic observe -> asc_dcci_single(each line)
        -> asc_threadfence -> ordinary payload read -> control recheck
```

writer 侧禁止用 `asc_dcci_single` 代替 clean-out。completion 交棒前的失败
先撤销本 task 独占的 fresh-output published/last-writer 控制字，然后发布
全局 fatal；heap cursor、writer history 或 ordinary ring 的已提交前缀不做
并发局部回滚，整轮不得继续 Execute，下一轮必须完整清零 sidecar。该
fail-stop 合同与正常热路径分开，不能为了错误恢复给每个 task 增加额外原子。

### 4.4 尚未实现的公共边界

下列能力不在首版声明范围内：

- Plan/Build 流式重叠、frontier 暂时追平、Build 重访和反压；
- generation、ABA、ring reclaim 和不经 Host 全量清零的 PlanCell 循环复用；
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
| S1 | 接通真实 AICPU orchestration SO 和 Plan backend | 已闭合，正式 A5 owner/dispatcher 已通过 |
| S2 | ordinary Scalar Build，CPU 后 A5 B1/B256 | 已闭合，Plan-only B1 与 full B1/B256 均通过 |
| S3 | 在同一 Plan ABI 上替换为 ordinary SIMT Build | 已闭合四 leader/通用 writer/窄完整 Build 的 CPU 与 CCEC machine-code 门槛；正式 runtime/A5 尚未实现 |
| S4 | 证明 ordinary 闭合后迁移 DAG Scalar Build | 未实现 |
| S5 | 在 DAG 上替换为 SIMT Build | 未实现 |
| S6 | 在正确性与观测闭合后做性能收敛 | ordinary Scalar 已有首组无泳道样本，尚未收敛 |

S0 至少需要以下证据：

- CPU 布局、payload 边界、无指针序列化、稠密 task id 和唯一 Build 领取；
- CCEC AIC/AIV 对 control dependency、atomic、DCCI 顺序的静态检查；
- A5 多 cache-line payload 的发布前不可见、发布后完整可见；
- SO entry/config 和 `dist_*` 符号实际闭合，不能只依赖 `RTLD_LAZY`；
- TensorMap 含零 writer task 的严格 completion 顺序；
- B1、B256、短尾、INOUT 和至少一个非 PA symbolic-output 边界例。

性能只在对应功能门槛闭合后记录：

- 权威结果使用无泳道构建，主 pipeline 起点是 AICPU owner
  launch 之前，终点是最后一个 AICore 完成 FinalDrain 并同步结束；
- 主 pipeline 必须包含 AICPU 对真实 orchestration 的调用、Plan 生成与
  发布、Build、TensorMap、Execute 和 FinalDrain，不包含后续 D2H 和 Host
  oracle；
- Host 侧 owner DSO 首次加载/handle 初始化当前在主 pipeline 之前，
  必须将该冷启动成本与已初始化 handle 的重复运行分组记录，不能
  声称当前 `pipeline_e2e` 已包含 DSO 首次加载；
- `plan_time` 必须保留为 Path-A Host wall 口径，其中包含 owner
  launch/sync 固定成本；`producer_exec` 必须独立保留为 AICPU 内部
  真实 callback + canonical Plan 打包/发布窗口，两者不得混称；
- `aicore_time` 是 AICore launch/sync Host wall 口径；
  `startup→FinalDrain` 是排除 Host launch/sync 后的 AICore device 内主体口径，
  两者也不得互相冒充；
- 泳道只用于定位 Plan/Build/Execute/Atomic/DCCI 分布，不与无泳道绝对值
  相减；
- 对照必须使用同一 PA 输入、真实计算负载和 startup→FinalDrain
  边界。

旧 Host 预制/PA 公式方案的约 0.82ms 不是等价基线。当前 ABI v2、真实
`6,28,4,1` 负载、同一进程 5 轮的 trace-free 中位数为：

| workload | plan_time | producer_exec | aicore_time | startup→FinalDrain | pipeline_e2e |
| ---- | ----: | ----: | ----: | ----: | ----: |
| B1 | 168.773us | 103.489us | 207.296us | 156.880us | 371.006us |
| B256 | 2690.859us | 2626.289us | 2491.516us | 2460.846us | 5180.220us |

B1 首轮仍有 6011.888us Plan/6622.200us pipeline 的 Path-A 冷启动，
但第 2--5 轮 pipeline 已落在 331--476us；不能把首次执行成本冒充每轮
callback 成本，也不能把 warm 中位数冒充冷启动。B256 的 warm producer
本身仍为 2626.289us，完整 pipeline 为 5180.220us，因此小于 1ms 目标
明确未达成。
下一步必须先实现 ordinary SIMT，在同一 Plan ABI、ordinary
TensorMap 语义和 pipeline 口径下与 Scalar 比较；之后才能依次进入
DAG Scalar/SIMT。
