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

ordinary + Scalar 与 ordinary + SIMT 的功能门槛都已闭合。当前公共
canonical Plan 是 ABI v3；ordinary Scalar 长期保留两个编译期 pipeline
policy：默认 `plan-ahead-closed` 是严格串行基线，
`streaming-future-pN` 是并发实验路径。两者只改变 producer admission、
Host launch 顺序和 consumer ticket 解析，继续共享 Plan、Build、ordinary
TensorMap、32 AIC + 64 AIV Execute/FinalDrain。formal SIMT 当前显式绑定
`streaming-future-pN`，不参与这次 Scalar 双 policy 的性能因果对照。

历史 S3 的 SIMT A5 B1 smoke 与 B256 perf-clock 功能均 PASS，但只有
ABI v2 首次单次功能样本，B256 pipeline 约 71.8ms，不是稳态性能基线。两种 DAG 模式
均未实现；当前只做 ordinary Scalar pipeline 性能工作，不进入 DAG。

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

## 2. 双 pipeline policy 与真实时序

ordinary Scalar 通过 `PA_RUNTIME_PLAN_PIPELINE_POLICY` 做编译期选择；policy
同时进入 Host、AICore、AICPU producer、Build identity、产物目录和
manifest，不允许运行时把不同身份的四件套拼接。两条策略刻意只保留三处
差异：

| 边界 | `plan-ahead-closed`（默认） | `streaming-future-pN`（实验） |
| ---- | ---- | ---- |
| producer admission | 生产期保持 `NotReady=-2`，完整 Plan 后才 `Open=-1 -> Closed=N`；Close 要求尚无 Build ticket/arrival | 连续前缀在真实 batch 边界达到 prefill N 后发布 `Open=-1`；Close 允许与 Build 并发 |
| Host launch | AICPU launch 后同步 plan stream，再 launch AICore | AICPU/AICore 使用两条 stream，consumer 紧随 producer launch，两条 stream 最终都同步闭合 |
| consumer ticket | attach Closed Plan、缓存最终 N；`ticket>=N` 直接结束 | Ready 后可持有 future ticket，轮询自己的 cell，再结合最终 Close 解析 ticket |

除此之外，两条路径共享唯一 AICPU producer、canonical Plan ABI v3、同一
decode/Materialize/Alloc、ordinary TensorMap、Fanin、exec-cell Build、
Execute 和 FinalDrain 实现。policy 不是复制业务主链的理由。

### 2.1 长期基线：plan-ahead-closed

默认路径先完整执行一次 orchestration、发布所有 PlanCell 并封口，再启动
AICore Scalar Build：

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

AICore Scalar Build
  96 个 worker attach immutable Closed Plan 并缓存 N
  从 build_next 领取 [0, N)
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

正式 Host 在 AICPU launch 后显式 synchronize plan stream，因此
Plan-ahead 是真实的两段串行边界，不是“AICore 早启动但在 GM 中轮询
Close”。它是需要长期保留的正确性和性能基线，不能在流式实验接通后
删除。该路径通常满足
`pipeline_e2e ≈ plan_time + aicore_time`。

Build 看到的必须是已封口的稳定计划：

```text
planned_frontier == closed_task_count == final_task_count
```

`planned_frontier` 仍用于证明 PlanCell 按无缺口前缀发布。producer 在
NotReady 下可以完成同一条 single-Pack/Published 路径，但 Close 前不存在
consumer；Close 还必须验证 `build_next==0 && build_workers_done==0`。
96 个 Scalar attach 一次后，领取热路只有中央 FetchAdd。

### 2.2 实验路径：streaming-future-pN

Streaming 复用同一份 ABI v3 control，状态定义为：

```text
-3 = ReadyFailed
-2 = NotReady
-1 = Open
N >= 0 = Closed，N 是最终 task_count
```

Host 先 launch AICPU producer，然后立即在独立 AICore stream 上启动
consumer。producer 完成复用地址的 discard/reset 后仍保持 NotReady，并在
该状态下顺序发布 PlanCell；连续 Published 前缀在真实 batch 边界达到
`PA_AICPU_PLAN_READY_PREFILL_TASKS`（默认 128）后才发布 Open。短 Plan
不会为了凑 prefill 卡住：最终 Close 先完成 Open 握手，再发布 Closed(N)。

96 个 Scalar 在 Ready 前只轮询 closed line；看到 Open/Closed 后才允许读取
其他 Plan control 并领取 ticket。每个 worker 只有在完成当前 ticket 后才领
下一张，future ticket 不退回：

```text
FetchAdd build_next -> ticket t
  t 的 cell 已 Published：acquire / decode / Build
  t 的 cell 仍 Empty：轮询该 cell，低频观察 Close/fatal
  观察 Closed(N) 且 t >= N：本 worker 到达 Build 尾部
  观察 Closed(N) 且 t < N：必须再次 acquire 同一 cell；仍 Empty 才 fatal
```

最后一条重读规则处理 Close control 与 cell control 独立可见的情况，不能把
合法发布误判为 MissingPlanCell。`ReadyFailed=-3` 只保证 closed line 可见，
consumer 不得先摸其他可能尚未 reset 的 control。正常成功时两种 policy
都满足 `build_next=N+96`、`workers_done=96`、`release=N`、`fatal=0`；
Streaming 还由最后 arrival 核验 `frontier=closed=N`。

当前 p128 是实验 control，不代表 future-ticket admission 已被证明最优。
后续可以限制 Open 阶段 builder 数 K、Close 后再让全部 96 Scalar 加入，
但 limited-K 尚未实现，不能写入当前功能声明。

### 2.3 AICPU 必须执行真实 orchestration

这条 standalone 路径复用仓库已有的 AICPU SO loader，并在 FDWIC AICPU
DSO 中提供 orchestration 所调用的 `dist_*` Plan backend；不能退回只做
`dist_engine_register`/worker handoff 的旧行为。

Plan backend 保持现有 orchestration API 签名，但改变其 device 实现语义：

- Begin 按 callback 顺序为任务分配唯一 task id；
- Finish 在 callback 引用仍有效时，将 MixedKernels、
  Tensor/TensorCreateInfo、scalar、显式依赖和 symbolic output reference
  single-Pack 到目标 GM PlanCell，control 仍保持 Empty；
- 下一次 Begin 或最终 Close 只补 final flags、校验同一份 GM wire，再按
  payload clean、barrier、Published control 发布，不保留完整 payload
  staging 副本；
- orchestration entry 正常返回后才封口 Plan；
- SO 加载、config、符号或 descriptor 校验失败时 fail-closed，不唤醒
  AICore 进入半发布 Plan。

Host 不得为这条路径设置 PlanCell 内容。

当前 A5 owner request 只携带 Plan storage 引用、容量、输入 Tensor
metadata、`context_lens` 和 scalar。task 数、task kind、task identity 和
dispatch plan 均不是 Host 输入。正式 SIMT Host 只在 AICore 完成
FinalDrain/sync 后回读 immutable Plan snapshot，对 writer/output 做事后
Plan-driven 投影校验；该 oracle 不进入 device 调度数据流或权威
计时窗。

### 2.4 构建与观测身份隔离

Scalar CPU/CCEC 构建目录必须包含 policy key：

```text
ordinary/scalar_build/build/<backend>/shared/plan-ahead-closed/<variant>/
ordinary/scalar_build/build/<backend>/shared/streaming-future-pN/<variant>/
```

CCEC `pa_scheduler_artifacts/v6` manifest 必须固定 Runtime Plan ABI、
`scheduler_input`、`pipeline`、`launch_order`、`producer_ready`、
`consumer_admission`、`prefill` 以及 Host/kernel/AICPU owner/dispatcher
四件套 SHA256。Build identity 也编码 streaming bit，运行入口必须拒绝
跨 policy、跨 prefill 或跨 ABI 混件。

泳道输出目录带相同 policy key；raw metadata 显式保存上述 policy、
`runtime_plan_build_backend/build_workers/execute_workers/
build_trace_coverage`、`producer_task_count/task_count/task_kinds` 和 Runtime
Plan terminal snapshot。device Closed(N) 是 task_count 权威来源，并与
producer result/frontier/release 交叉；Host plan 只作 oracle。Scalar raw
使用 `scalar-task-detail`、W=96；formal SIMT raw 使用
`simt-coarse-direct-state`，只以 terminal `N+4/4/N` 闭合 VF Build，不伪造
Scalar child/VF atomic 记录。converter 只对
ABI v2 的历史缺字段 raw 推断 legacy `plan-ahead-closed`，并标记
`inferred_legacy_policy=true`；ABI v3 缺少显式 policy 必须拒绝。这样历史
回放、串行长期基线和并发实验不会落到同一模糊身份中。

正式 profiling 只合并两个设备时钟域：AICPU producer 使用
`CLOCK_MONOTONIC_RAW`，AICore 使用 `SYS_CNT`。每次 capture 在主体前、后
分别执行 4 组四时间戳握手
`(AICPU send, AICore receive, AICore send, AICPU receive)`，由 8 组 offset
区间的交集完成映射；最终相关误差必须不超过 50us，且每组四个原始时间戳、
区间上下界、往返时间和轮次身份都必须原样保留。Host 只负责启动和同步，
不作为第三个时钟域，不生成 Host lane，raw 也不得写入 Host timestamp、
clock bracket 或 lane metadata。

## 3. 公共 Plan ABI 与内存合同

S0 草案位于
[公共 Plan 协议头](common/aicpu_plan_protocol.h)。其当前结构边界如下：

- `RuntimeTaskPlanHeader` 是 64B，它只是 payload 的公共头；公共 wire
  ABI 当前为 v3，header 的 16bit `adapter_data` 由算子 adapter 定义，
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
AICore `function_address`；它缺少 canonical Plan v3 的 `adapter_data`。
因此只能复用
其中已经验证的 atomic/DCCI、VF leader 和 publication 实现经验，不能把
canonical Plan 再复制成该 request，也不能把两种 cell 强转互换。

ordinary Scalar 与 ordinary SIMT 都必须直接消费本目录唯一权威的
`RuntimeTaskPlanCell` ABI v3。公共 Plan 只保存 logical function id；Build
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

### 4.1 ordinary Scalar 双 policy 的精确收口

`plan-ahead-closed` 中，96 个 Scalar 在 Plan 完整 Close 后各自 attach 一次
并缓存稳定的 task 数 N，然后重复对同一 `build_next` 做返回型 FetchAdd。
成功 ticket 为 `[0, N)`，每个 worker 最后再消费一张越界 ticket。

`streaming-future-pN` 中，96 个 Scalar 在 Ready 后同样只对
`build_next` 做一次 FetchAdd/每 ticket，但 ticket 可以先于 cell。worker
完成该 ticket 的 Build 或在最终 Close 中证明 `ticket>=N` 后才报到；最后
arrival 重新校验 `frontier=closed=N`。因此两种 policy 的共同终态都是：

```text
build_next == N + 96
workers_done == 96
build_release == N
fatal == 0
```

Streaming 观察 Close 后若手中 `ticket<N`，必须重新 acquire 同一 cell，
不能直接报告缺失。Plan-ahead 则已经在 attach 时一次性 acquire 并交叉
校验 immutable close/frontier/capacity，热循环不再读取 future cell、close
或 frontier。

中央 ticket 只解决“每个 PlanCell 只 Build 一次”，不放宽 TensorMap
顺序。task N 可以在串行链外完成 Plan acquire 与 Materialize，但 writer
metadata 发布仍必须等待 completion[N-1]，发布后再推进 completion[N]。
完成所有 Build 后才发布 release，当前 Execute 仍是 post-build 扫描。
Streaming 只重叠 Plan 与 Build，尚未实现 Build/Execute 重叠。

### 4.2 CCEC direct-entry 的核型本地状态

ordinary Scalar 保留 direct PlannedBuild，不恢复旧 callback split。但 CCEC 下
`LocalStats` 会跨 `noinline` Build/Execute helper 传递引用，不能放在普通栈上。
因此 AIC 和 AIV 各使用一份 role-specific `[[block_local]] LocalStats`，
并在每轮入口整体复位。CPU 实现继续使用栈对象。这只是 CCEC
存储约束，不改变 non-split `WorkerResult` ABI、Plan ticket 协议或调度语义。

### 4.3 ordinary SIMT 的精确合同

S3 的 ABI v2 首版使用 closed Plan/Plan-ahead；当前 ABI v3 formal SIMT
为避免维护尚未验证的第二套 Host/VF 时序，源级显式绑定
`streaming-future-pN`。它仍直接消费完全相同的 canonical PlanCell，只有
一个物理 builder VF：block0/AIV0 调用
`async_invoke` 启动 128 个 SIMT thread，组成 4 个 warp；每个 warp 只有
lane0 是 active Build leader。VF 完成后，该 AIV0 Scalar 仍回到普通 AIV
Execute/FinalDrain，因此 Execute 拓扑保持 32 AIC + 64 AIV，而不是为了
Build 永久牺牲一个执行核。

该架构已接入正式 CCEC mode，不再只是 compile probe。唯一
正式 AIV global entry 中，AIV0 在任何 Build GM side effect 之前做
backend/ABI/variant/workers/storage 只读 preflight，再启动 VF；其他 AIV
直接进入 Scalar continuation。success 和 fatal 都必须先做 V->S join，
随后 AIV0 作为 worker 32 进入第 96 个 Scalar continuation，避免其余
95 核卡在 startup/release。AIC 继续复用 Scalar AIC 正式入口。

当前 4 个 leader 共用 `build_next.FetchAdd(1)`：

```text
AICPU: NotReady -> prefill -> Open -> Closed(N)
  -> 4 个 VF leader 等待 Ready 后动态领取 ticket
  -> future ticket 按 cell Published/最终 Closed(N) 解析
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
直接复用已经闭合的 open/future-ticket 领取、报到和 release 协议。正常
原子调用数
为 `N+4` 次 build-next FetchAdd、4 次 arrival FetchAdd 和 1 次 release
publish。每个 leader 在 Ready 前只观察 closed line，最终从 Close 得到并
交叉校验 N；每个 PlanCell 的前后 control 观察与严格 TensorMap
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

Alloc 也是上述动态 ticket 流中的真实 Plan task。AICPU 在 Alloc
PlanCell 中只发布逻辑 Alloc identity（`EngineClass::MetadataOnly`）和
fresh-output CreateInfo，不在
Plan 阶段预分配 heap。真实 8-shard heap reserve、descriptor Materialize、
`SharedOutputCell` publication 和 metadata vend/flag 都在拿到 Alloc ticket 的
Build leader 上完成；Alloc 不产生后续 executable kernel。

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
  -> 校验 Build identity 与 Plan storage
  -> async_invoke 128-thread VF
  -> 4 个 lane0 leader 等待 Ready 并执行窄 canonical Build
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

shared writer-history key 是one-based 公共 ABI：

```text
key = producer_task_id * kSharedOutputMaxPerTask + output_slot + 1
```

0 永久保留为 invalid/unset，所以 producer 0/slot 0 的合法 key
是 1，不能使用零基编码把它与空 history 混淆。SIMT writer、
future-writer reader 回溯和 Host oracle 必须使用同一编解码。

Host 在 AICore FinalDrain/sync 结束后才 D2H 回读 immutable PlanCell
前缀，以 Plan tensor tag/CreateInfo/descriptor/OutputRef 投影 fresh output、
ordinary writer、symbol history 和 final writer，再对照真实
`SharedTensorMapSidecar`。该 writer/output oracle 不读 PA group/kind 公式，
快照 D2H 也不在 pipeline 计时窗内，更不是 Build/Execute 调度输入。

正式 AIV 最终 metadata 要求 8344B SIMT stack 和 8320B divergence stack。
SIMT Host 在 ACL 首次初始化时显式配置向 512B 步长取整后的
`simt_stack_size=8704` 和 `simt_divergence_stack_size=8704`；Scalar
backend 不使用该配置。8704B 分别覆盖 8344B/8320B 的最终
metadata 要求。这是正式 SIMT ELF 能在 A5 启动的 runtime
资源合同，不是 Build 算法参数。

### 4.4 尚未实现的公共边界

下列能力不在当前声明范围内：

- 限制 Open 阶段 builder 数 K、Close 后扩容到全部 worker 的 admission；
- 自适应 prefill、producer 反压控制，以及 Build/Execute 流式重叠；
- generation、ABA、ring reclaim 和不经 Host 全量清零的 PlanCell 循环复用；
- 多 AICPU producer 的分区与有序合并；
- 完整 Joint/MixedKernels、multi-core launch 语义和所有 engine 组合；
- 依赖 kernel 运行结果才能继续生成后续 Plan 的 result-driven
  orchestration；
- Submit 返回时必须立即取得 materialized `TaskOutputTensors::get_ref()` 的
  同步输出 API。
- DAG TensorMap 的 Scalar/SIMT Build。当前只闭合 ordinary，DAG 按用户
  要求继续冻结，不能在 pipeline 优化中顺带实现。

PA 首例只能使用已有 symbolic `SharedTaskOutputs/FdwicOutputRef` deferred
路径。遇到上述不支持语义时必须 fail-closed，不能伪造 Tensor descriptor。

## 5. 实现顺序、验证与性能口径

历史实现顺序先 ordinary Scalar、再 ordinary SIMT；原先排在最后的 DAG
阶段当前按要求冻结：

| 阶段 | 目标 | 当前状态 |
| ---- | ---- | -------- |
| S0 | 固定 Plan ABI、AICPU/AICore 发布合同和正确性门槛 | 已闭合 |
| S1 | 接通真实 AICPU orchestration SO 和 Plan backend | 已闭合，正式 A5 owner/dispatcher 已通过 |
| S2 | ordinary Scalar Build，CPU 后 A5 B1/B256 | 已闭合，Plan-only B1 与 full B1/B256 均通过 |
| S3 | 在同一 Plan ABI 上替换为 ordinary SIMT Build | 已正式接通；CPU/CCEC 门槛和 A5 B1/B256 perf-clock 功能全部 PASS |
| S4 | 证明 ordinary 闭合后迁移 DAG Scalar Build | 按当前要求冻结，未实现 |
| S5 | 在 DAG 上替换为 SIMT Build | 按当前要求冻结，未实现 |
| S6 | 在正确性与观测闭合后做性能收敛 | 已保留 Scalar 串行 policy 并接通 streaming 实验 policy；同提交 ABI v3 A5 A/B 待补齐 |

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
  必须将该首次加载成本与已初始化 handle 的重复运行分组记录，不能
  声称当前 `pipeline_e2e` 已包含 DSO 首次加载；
- `plan_time` 必须保留为 Path-A Host wall 口径，其中包含 owner
  launch/sync 固定成本；`producer_exec` 必须独立保留为 AICPU 内部
  真实 callback + canonical Plan 打包/发布窗口，两者不得混称；
- `aicore_time` 是 AICore launch/sync Host wall 口径；Plan-ahead 中它覆盖
  AICore 独立阶段，Streaming 中则从早期 launch 起包含 Ready/PlanCell/
  Close 等待，只能作为 upper bound，不能冒充纯 Build/Execute 时间；
  `startup→FinalDrain` 是排除 Host launch/sync 后的 AICore device 内主体口径，
  两者也不得互相冒充；
- 正式泳道必须同时包含 AICPU producer 与 AICore，不再接受 AICore-only
  结果；Host wall duration 只能作为 trace-free 运行的独立摘要，不能进入
  联合泳道的时钟映射、lane 或 raw metadata；
- 相关握手发生在主体前后，因此 structural capture 已发生设备运行时预热；
  必须标记 `clock_correlation_warmup_before_pipeline=true`、
  `timing_scope=calibrated-structural-capture` 和
  `performance_representative=false`。泳道只解释结构，绝对性能统一取
  trace-free 同进程 warm 中位数，两者不得相减或互相替代；
- `RuntimePlanBuild`、真实 Kernel Execute 与 FinalDrain 分别按对齐后事件
  区间的 union 计算，同一阶段内并行 worker 的重叠只能计一次。AICPU 与
  各阶段的 overlap、非重叠 tail 和 signed gap 也必须由真实区间交并得到；
  `streaming-future-pN` 只是策略身份，不能预设必然发生重叠或把等待时间
  冒充 Build/Kernel；
- B1 只作为最小功能 smoke，不记录或汇报性能，不保留性能泳道；小于 1ms
  目标、性能 A/B 与状态汇报统一只看 B256；
- 当前正式留存集合固定为 B256 下两个 policy 分别运行真实计算
  `1,1,1,1` 与 `6,28,4,1` 的四份 joint structural capture；具体数值和
  文件清单只在 capture 完成并通过校验后写入日期目录 README；
- 对照必须来自同一源码 commit、同一 Runtime Plan ABI 和同一 single-Pack
  producer，使用同一 PA 输入、真实计算负载、trace 开关和计时边界；每个
  policy 分别在同一 Host 进程内取得 warm 中位数。

旧 Host 预制/PA 公式方案的约 0.82ms 不是等价基线。历史 ABI v2、真实
`6,28,4,1` 负载、同一进程 5 轮的 trace-free 中位数为：

| workload | plan_time | producer_exec | aicore_time | startup→FinalDrain | pipeline_e2e |
| ---- | ----: | ----: | ----: | ----: | ----: |
| B256 | 2690.859us | 2626.289us | 2491.516us | 2460.846us | 5180.220us |

B256 的 warm producer 本身仍为 2626.289us，完整 pipeline 为 5180.220us，
因此小于 1ms 目标明确未达成。B1 结果只用于功能 smoke，不再作为性能证据。

在双 policy 固化前还做过一次 B256、`real-compute=6,28,4,1`、同进程
5 轮 warm 诊断：历史串行 control 的 pipeline 为 5187.986us，naive
StreamingFuture p128 为 6575.206us，即 **5.188ms -> 6.575ms**。对应
Plan wall 为 2667.425->4414.344us，producer exec 为
2590.039->4336.106us。Plan 增加 1746.919us，producer throughput 约下降
40.3%；以串行 AICore span 2520.457us 作诊断性 counterfactual，重叠只
隐藏约 359.595us，最终 pipeline 净回退 1387.220us。

这个负结果说明不能默认“Ready + 96 个 future-ticket poller”最合理，但它
跨 ABI/实现版本，不是最终因果 A/B。正式结论必须在同 commit、ABI v3、
single-Pack 下分别构建两个 policy 并取得 warm pair。后续候选是只允许
K 个 Scalar 在 Open 阶段 Build、Close 后再开放全部 96 个 worker；K 限流
目前尚未实现，不能把建议写成现状。

ordinary SIMT 正式 `perf-clock` 的首次单次功能证据为：

| workload | startup→FinalDrain | plan_time | producer_exec | aicore_time | pipeline_e2e | 结果 |
| ---- | ----: | ----: | ----: | ----: | ----: | ---- |
| B256 | 62359.163us | 8871.954us | 2959.921us | 62884.823us | 71756.980us | execution/semantic/postprocess PASS |

该 B256 数据不是多轮 warm 中位数。B1 只保留功能 smoke 结论，不进入
性能表或汇报。formal ordinary SIMT 的功能、语义和后处理已收口；B256 的
`startup→FinalDrain` 为 62.359ms，`pipeline_e2e` 为 71.757ms，
离 1ms 目标很远；在取得多轮热态证据前不宣称 SIMT 性能
达标或优于 Scalar。ordinary 功能范围已经停止扩张；当前只进行 Scalar
pipeline 性能实验，DAG Scalar/SIMT 均未实现。
