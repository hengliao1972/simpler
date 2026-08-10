# AICPU TaskPlan 实现过程

本文只记录 `cross_core_aicpu_plan` standalone 已发生的实现、校验和性能
证据。架构合同见
[《AICPU TaskPlan 调度设计》](AICPU_TaskPlan调度设计.md)。

## 1. 记录约定

- CPU、CCEC 静态产物和 A5 动态结果分开记录，不互相替代。
- 没有实际运行的项目统一标记为 `NOT RUN`，不从编译通过推导上板
  正确性。
- 性能结论必须附同 workload、同边界、构建类型、运行次数和结果
  路径。
- 没有闭合功能门槛前，不记录“小于 1ms”或“比某模式更快”等
  结论。
- Host 预制 task identity 和 device PA 固定公式都不是可接受的功能捷径。

## 2. 阶段总览

| 阶段 | 目标 | 状态 |
| ---- | ---- | ---- |
| S0 | Plan ABI、动态存储和 AICPU/AICore 发布合同 | 已闭合 |
| S1 | 真实 AICPU orchestration SO 与 Plan backend | 已闭合，已通过正式 A5 AICPU launch 验证 |
| S2 | ordinary Scalar Build 端到端 | 已闭合，CPU 与 A5 B1/B256 均已通过 |
| S3 | ordinary SIMT Build 端到端 | 未开始 |
| S4 | DAG Scalar Build | 未开始 |
| S5 | DAG SIMT Build | 未开始 |
| S6 | 观测闭合后的性能收敛 | 已有首组无泳道样本，尚未收敛 |

当前已经闭合 ordinary + Scalar 的第一条完整路径：真实
AICPU callback 生成 canonical Plan，Plan-ahead 完整封口后，96 个
Scalar 用中央 ticket 完成 Materialize、ordinary TensorMap 严格顺序
插入、Fanin 和 Build，最后由 AIC/AIV Execute 与 FinalDrain 收口。
这证明了功能路径；B1 warm pipeline 已低于 1ms，但 B256 仍明显超过
1ms，不能把子阶段时间当成已达成整体目标。

## 3. 2026-08-10：S0 协议门槛

### 3.1 已写入源码的协议

当前只有
[公共 Plan 协议头](common/aicpu_plan_protocol.h)和一份 atomic/DCCI 源码覆盖检查
草案。协议头已表达：

- `RuntimeTaskPlanHeader` 为 64B；公共 wire ABI 当前为 v2，header 的
  16bit `adapter_data` 由算子 adapter 定义，PA 用它显式携带真实
  Alloc callback 的 `batch_start`；
- 完整无指针 payload 最大为 4416B，共 69 条 cache line；
- `RuntimeTaskPlanCell` 当前为 4608B stride，包含独立 128B control、完整
  payload 和尾部对齐；
- payload 可表达 Tensor/TensorCreateInfo 值、symbolic output reference、
  scalar 和显式依赖；
- PlanCell 数组由 `RuntimePlanStorageRef` 引用运行期外置 GM，避免按最大
  task 容量固定膨胀 SchedulerState；
- producer 按 payload clean、barrier、control publish 的顺序发布；
- consumer 观察 control 后按发布的 payload line 数 invalidate 并校验。

六条 Plan 控制状态分别占用 128B：连续发布前沿、最终 task 数、Build
领取游标、Build worker 到达数、Build release 和 fatal。首版已经删除
`TemporarilyEmpty` 等流式 reservation 口径；Build 只能在 Plan 完整封口
后调用 `TakeClosedPlanBuildTicket()`。96 个 worker 各完成一次越界领取，
所以成功终态严格为 `build_next == N + 96`，不是 `N`。

每个 tensor 固定占 16 个 word，避免 Scalar 与 SIMT 分别维护变长 offset：

- TensorDesc 使用 16 word；
- TensorCreateInfo 使用前 8 word，后 8 word 必须为零；
- FdwicOutputRef 使用完整 16B 两个 word，后 14 word 必须为零。

OutputRef 消费端逐字段校验 `producer < consumer`、`slot < 8`、plain/view
标志、一维 shape/offset 和加法溢出。Header 也改为显式逐 word 编解码，
不再通过对象表示强转而依赖 padding 或 strict-aliasing 行为。

ABI v2 不允许 Scalar 从 `task_id` 猜测 PA batch。AICPU producer 在真实
Alloc callback 到达时记录当前 `batch_start`，后续同 batch callback 沿
continuation 携带该值。Scalar 从 Plan header 读取它；PA 固定 offset 只在
adapter 内交叉校验显式 provenance，不能生成 provenance。

另一个仍未闭合项是与生产
`runtime/dist_engine/common/cross_core_simt_request_protocol.h` 的复用。当前
standalone 头文件已固定 canonical 4416B payload，但尚未证明两者的
pack/validate 逻辑已经收敛到同一权威实现。

### 3.2 本轮文档收敛

本轮只修正设计口径：

- 删除“`RuntimeTaskDesc` 恰好 64B”的错误定性；
- 明确 64B 是 header，完整通用 payload 上限是 4416B；
- 明确 PlanCell 是运行期动态外置数组，Host 只提供容量和 GM 地址；
- 明确后续要复用或提取生产 4416B request payload 的 pack/validate
  机制，不保留两套漂移 ABI；
- 将首版时序收敛为 Plan-ahead，先封口完整 Plan，再启动 Build；
- 明确 Plan 必须来自 AICPU 执行真实 orchestration SO 及 `dist_*`
  Plan backend；
- 明确 TensorMap 依然使用独立 `N-1 -> N` completion chain；
- 将实现顺序固定为 ordinary Scalar、ordinary SIMT，最后再进入
  DAG Scalar 和 DAG SIMT。

### 3.3 当前验证状态

- 文档与本地头文件布局人工对照：已完成。
- 链接与 Markdown 换行检查：已通过。
- CPU 公共协议测试：PASS。覆盖 32 tensor、16 scalar、16 explicit dep、
  69-line 最大 payload、完整 OutputRef、非法 control/payload、B1/G0/G1/
  mixed/B256，以及 96 worker 的 `N + 96` Build ticket 闭合。
- CPU PA adapter 测试：PASS。使用真实 `TaskArgs/TensorDesc/
  TensorCreateInfo/FdwicOutputRef` 做 Publish→Acquire→Decode 逐字段对照；
  五类 tag、plain/一维 view、PTO2TaskId raw、canonical zero 和不携带源指针
  均闭合；非法 PA meta、kind/engine、future ref、slot、dtype/rank 均拒绝。
- 上述两组测试均通过 `-Wall -Wextra -Werror`；PA adapter 另经
  ASan+UBSan 复核通过。
- CPU 外置 storage 门槛：PASS。当前下游 task-indexed 消费容量为
  4352，cell stride 4608B，总空间 20054016B（约 19.125MiB）；Host
  helper 仅写 StorageRef，20MiB canary 证明没有写入任何 cell identity
  或 payload。错位、零/越界容量、stride/ABI/reserved 和地址溢出均拒绝。
- ordinary/Scalar 旧 shared CPU 全量门槛在追加 Plan 尾状态后仍通过，
  包括 atomic/DCCI source coverage、strict insert、ring、多核 replay、
  Execute cell/token/drain 等现有测试。
- CCEC 公共协议模板检查：PASS。`dav-c310-vec` 已实例化
  `MakeRuntimePlanView()` 与 Pack 路径，地址空间修饰未丢失。
- A5 Plan 发布/获取独立探针：PASS，见
  [protocol_probe](common/protocol_probe/README.md)。G0、G1、mixed 和
  B256 均使用真实 AICPU ordinary store/clean/DSB 与 AIV return-ready
  atomic observe/invalidate/DSB；同地址 partial-clean+poison 负例在
  task 4、word 544 被拒绝，随后同地址完整发布恢复通过。
- A5 B256 探针最新一次取证：1280 cells、37888 条 payload cache line；
  close-only producer 3.071ms，AIV 全量逐 word oracle wall 19.847ms；
  per-item-frontier producer 3.382ms，AIV wall 16.764ms。该数据只衡量
  重型可见性 oracle，不是 PA 调度性能。
- CCEC Host 正式外置 storage 生命周期：PASS。`SchedulerState` 和
  `RuntimeTaskPlanCell[4352]` 都采用“原始分配 + 127B slack + 128B 内部
  对齐”，每轮执行
  `InitializeState -> 清零 cells -> ConfigureRuntimePlanStorage -> H2D`。
  Host 没有写入任何 task identity/payload；正式 A5 路径已经由 AICPU
  owner 向该 storage 发布 Plan，再由 AICore consumer 获取。
- A5 Plan-only B1：PASS。真实 AICPU callback 生成 5 个 PlanCell，
  `frontier=closed=5`；D2H 后的 Host oracle 只用于事后逐字段
  校验，不是 producer 或调度输入。
- A5 full B1/B256：PASS。B1 终态为
  `frontier=closed=release=5`、`build_next=101`、`workers_done=96`、
  `fatal=0`；B256 终态为 `frontier=closed=release=1280`、
  `build_next=1376`、`workers_done=96`、`fatal=0`。Plan 获取、strict
  ordinary TensorMap completion、exec payload、AIC/AIV 路由、kernel
  completion 和 FinalDrain 均闭合。
- 同一 Plan storage 跨轮复用：PASS。上一轮 AICPU 发布 control 后执行
  `dc cvac + dsb`，因此下一轮只可能留下 clean-valid 旧行；Host 对同一
  GM 清零后，AICPU 先对每个 cell 的独占 control 行执行 `dc civac`，
  统一 `dsb sy + isb` 后才检查 Empty。此时 AICore 尚未启动，不存在
  并发 writer。Host 同址两轮 smoke、AArch64 反汇编顺序门槛和 A5
  同进程多轮均已通过。
- 无泳道 pipeline 性能：已取样，见 5.4 节；B1 warm 已低于 1ms，
  B256 仍未达成目标。

### 3.4 S0 收口结论

S0 的公共 wire、CPU malformed gate、CCEC 模板实例化、A5 多 cache-line
发布/获取探针，以及正式 CCEC Host 外置 storage 生命周期均已闭合。容量
固定取 AICore consumer 的编译期上限，Host 只发布地址/容量，不生产业务
内容。Scalar 只能 invalidate StorageRef ordinary payload，Plan control 的
atomic-only cache line 不执行 DCCI。该合同已经被 S2 正式 A5 端到端
路径实际消费，不再只有独立探针证据。

## 4. 2026-08-10：S1 真实 orchestration Plan producer

### 4.1 已实现的 producer 边界

[ordinary/scalar_build/aicpu](ordinary/scalar_build/aicpu/README.md) 已新增
独立 AICPU Plan backend：

- 实际加载并调用仓库 PA `paged_attention_orch.cpp` 生成的 orchestration
  SO；
- `Begin` 按真实 callback 到达顺序分配连续 `task_id`；
- Alloc 身份来自 Alloc API，计算 task 的 engine/function 来自真实
  `MixedKernels`，没有 `task_id % 5`、Host PA 公式或 device PA 公式；
- `Finish` 在 callback 生命周期内立即把真实 1280B `L0TaskArgs` 转成无
  指针 canonical payload；
- batch provenance 来自真实 Alloc callback；G1+G2 smoke 已逐 task
  验证 `adapter_data` 分别为 0 和 5，并验证每个 batch 尾各有一条
  `LastSubmit`；
- 一个 AICPU 私有 pending cell 只用于等到下一次真实 Begin 后确定
  `UP.has_following_group` 与 batch 尾；
- producer 只发布 Plan，不做 Materialize、TensorMap、Build 或 kernel。

### 4.2 已完成验证

`aicpu/build_smoke.sh` 的 Host `dlopen` 门槛实际运行 G1+G2 两个 batch，
由 orchestration 生成 14 个 PlanCell，计数为 Alloc/AIC/AIV=`2/6/6`；每个
cell 都重新经过公共 control/payload、engine/function、flags、output/ref
校验。相同源码同时用 CANN 9.1 HCC 生成 AArch64 SO，`readelf` 证明入口
导出完整且没有未闭合的 `dist_*`/adapter bridge 符号。构建使用
`-Wall -Wextra -Werror`；仓库既有 CCEC-only 未用参数告警单独显式豁免。

正式 A5 路径现已使用独立 AICPU owner/dispatcher launch 这份
AArch64 producer。Host 向 owner request 只提供 Plan storage 引用、容量、
`context_lens`、通用 Tensor metadata 和 scalar，没有 task 数、task
kind、task identity 或 dispatch plan。AICPU 运行仓库内的真实 PA
orchestration callback，将得到的 canonical Plan 发布到同一份 GM
storage，完整 Close 后才启动 AICore。

Plan-only B1 已经证明该 owner 可在 A5 上生成 5 个稠密 PlanCell。
Host 在 producer 结束后独立重建的 PA oracle 只是校验件：它不被上传到
Plan storage，也不参与 AICPU/Scalar 调度。

## 5. 2026-08-10：S2 ordinary Scalar 完整路径

### 5.1 CPU 独立协议门槛

新增的 `build_plan_scheduler_gate.sh` 先按 standalone 的真实
`BeginPaBatchForCallback -> PreparePaBlockGroup -> callback` 顺序动态生成
Plan，再启动 96 个 Build worker。该门槛没有 Host task 表，也不从
`task_id` 反推 task kind；任务语义来自刚刚发生的 callback 和 PA adapter。

门槛已经闭合：

- 完整 Plan 关闭后，96 worker 通过同一个返回型 FetchAdd 领取 Build；
- 每 task exactly-once，中央 ticket 终值为 `N+96`；
- 所有 task（包括无 writer 的 task）都执行严格 `N-1 -> N` insertion
  completion；
- 96 次 arrival 与唯一 Build release 闭合；release 还必须同时校验
  `build_next == N+96`，不能只凭报到数放行；
- 最小 runtime exec cell 由 32 AIC/64 AIV 按 engine 消费，metadata-only
  不进入执行引擎。

覆盖 B1、G0、G1、G2、G4、mixed 和 B256。B256 实际为 1280 tasks、
1376 tickets、256 metadata-only、512 AIC、512 AIV，最终 1280 个 task
全部精确到达各自 terminal 状态。普通构建、ASan+UBSan 和连续三轮均
PASS。

这一门槛刻意只实现最小 exec cell 和 insertion completion，用来单独
校验协议边界。在此之上，正式 CPU main 已经改为消费 closed Plan：

- B1 与 B256 均通过完整 Materialize、strict ordinary TensorMap
  `N-1 -> N`、Fanin、exec-cell Build、Execute 与 FinalDrain；
- 96 个 Scalar 每个恰好完成一次越界领取，因此 B1/B256 分别为
  `build_next=5+96=101` 和 `build_next=1280+96=1376`；
- 旧 replay/Claim 路径在 PlannedBuild 模式不运行，Claim 计数、
  replay identity 和 replay_done 保持初始值；
- CPU 门槛同时覆盖 trace-free 和 full-swimlane + atomic 构建，观测构建
  不得改变 Plan/Build 终态。

CPU 线程 wall time 不是 A5 性能代理，只作正确性和 sanitizer 证据。

### 5.2 A5 正式 Plan→Build→Execute 闭合

A5 正式路径遵循两阶段 Plan-ahead：

1. Host 启动 AICPU owner 并同步等待完整 Plan Close；
2. 32 个 mixed block 启动 32 AIC + 64 AIV，96 个 Scalar 用中央
   `build_next` 领取；
3. Build 完成 Materialize、严格串行的 ordinary TensorMap 插入、
   Fanin 和 exec-cell 发布；
4. Build release 后，AIC/AIV 按 engine 扫描并执行已构建任务，
   FinalDrain 完成后 Host 才停止 pipeline 计时。

Plan-only B1、full B1 和 full B256 均已 PASS。B256 具体包含 1280
个 task，其中 metadata-only/AIC/AIV 为 256/512/512；每个 PlanCell
只 Build 一次，每个 kernel 只执行一次，并且 TensorMap 插入完成字严格
按 task id 推进。

### 5.3 WorkerResult 污染的根因与修复

首次 direct-entry full B1 的 Plan、Build、Execute 和 TensorMap 主状态均已
闭合，但 `WorkerResult` 中的 submit/Claim 等字段出现了类似时钟值的
污染。查证结果是：

- Host 清零、H2D/D2H 范围、`WorkerResult` ABI 和 32KiB 栈上限均没有
  出错；
- direct Plan 路径把 1152B `LocalStats` 放在普通栈上，又将引用
  跨 `noinline` Build/Execute helper 传递，触发了 CCEC 对大型栈对象
  跨非内联调用的后端限制；
- 旧 split 路径没有出错的关键是 AIC/AIV 分别使用了
  `[[block_local]]` 状态，不是 split callback 本身必须保留。

修复保留 direct PlannedBuild 和 896B non-split `WorkerResult` ABI，只将完整
`LocalStats` 改为 AIC/AIV 各自的 role-specific `[[block_local]]`，每轮入口
完整复位。最终 ELF 中两份状态分别位于 `0` 和 `0x480`，总计
2304B、64B 对齐且无重叠。修复后 full B1/B256 的 worker id、role、
时钟边界、零 Claim 计数和所有 Host 语义校验全部 PASS。该修复没有
恢复旧 replay/split 调度流程。

### 5.4 当前 A5 性能证据

下表是 ABI v2、跨轮 cache 修复后的同一 trace-free 产物，使用真实
`real-compute=6,28,4,1`，在同一 Host 进程和同一 Plan GM allocation 内
连续运行 5 轮得到的中位数。它不是把不同构建或不同轮次的局部最优值
拼在一起：

| workload | plan_time | producer_exec | aicore_time | startup→FinalDrain | pipeline_e2e | 结果 |
| ---- | ----: | ----: | ----: | ----: | ----: | ---- |
| B1 | 168.773us | 103.489us | 207.296us | 156.880us | 371.006us | 5/5 PASS |
| B256 | 2690.859us | 2626.289us | 2491.516us | 2460.846us | 5180.220us | 5/5 PASS |

五个时间口径必须分开解读：

- `plan_time` 是 Host wall time，从 Path-A AICPU owner launch 前到对应
  stream sync 结束；它包含 Host/Runtime Path-A launch/sync 的固定成本，
  不能称为“AICPU callback 执行时间”。
- `producer_exec` 由 AICPU owner 内部起止时钟得到，才是真实
  orchestration callback + canonical Plan 打包的 device 执行窗口；它还包含
  owner 内的输入解析、backend bind 和 Close，但不包含 Host Path-A launch/sync。
  例如 warm B1 的真实 producer 中位数为 103.489us。
- `aicore_time` 是 Host wall time，覆盖 AICore launch、Build、Execute、
  FinalDrain 和 stream sync，因此也含 launch/sync 固定成本。
- `startup→FinalDrain` 是 AICore device 内边界，从最早 Scalar startup 到
  最后 FinalDrain 结束；它是观察 Build/Execute 调度主体的更纯净口径，
  不含 Host launch/sync 固定成本。
- `pipeline_e2e` 从 AICPU launch 前到 AICore FinalDrain 后 stream sync 结束，
  不包含后续 D2H 和 Host oracle。当前 Plan-ahead 没有重叠 AICPU 与
  AICore，所以它基本等于 `plan_time + aicore_time`。

`plan_time` 和 `producer_exec` 同时保留正是为了区分 Path-A Host/Runtime
成本与真实 AICPU 业务执行；两者不得混称为“Plan 生成时间”。同进程
首轮仍存在 Path-A 首次执行冷启动，必须与 warm 中位数分开：首轮 B1
的 Plan/producer/AICore/pipeline 为
`6011.888/148.512/610.054/6622.200us`，首轮 B256 为
`8548.902/2686.402/2934.944/11484.044us`。Host 侧 owner DSO 的加载和
handle 初始化发生在 `pipeline_begin` 之前；这里观测到的是首次 Path-A
执行及 runtime warm-up，不把它武断归因成 callback 业务代码。

B1 第 2--5 轮 pipeline 为
`476.197/347.080/371.006/331.096us`，证明固定冷启动不是每轮成本；
B256 warm 仍约 5.1--5.2ms，说明其主要矛盾已经转成随 task 数增长的
Plan 打包和 AICore Build/Execute，而非单纯冷启动。

上表只能声明 ordinary Scalar 功能闭合后的当前成本。B1 warm pipeline
已经低于 1ms；B256 仍未达到目标：AICore device 内窗口中位数为
2460.846us，完整 pipeline 中位数为 5180.220us。不能只报某个子阶段，
也不能用 Plan-only、Host oracle 或 `producer_exec` 改写 pipeline 口径。

当前 ABI v2 的最终 B1 full-swimlane（真实负载）位于
`ordinary/outputs/pa_scheduler_aicpu_plan_scalar_ordinary_swimlane_20260810_214013_1990201/ccec/`。
它包含 5 个全局唯一 PlannedBuild、4 个 Kernel、0 个 legacy Claim，
physical generic records 为 2165、drop 为 0；converter 与 Plan 专用
exclusive analyzer 均 PASS。泳道 lifecycle 为 348.119us，只用于归因，
不能与上述 trace-free 绝对值相减。

## 6. 下一步

ordinary Scalar 的功能基线已经成立。下一步按既定顺序实现
`ordinary/simt_build`，要求继续消费同一份 canonical Plan，不改变
ordinary TensorMap 严格顺序插入语义。完成且比较 ordinary Scalar/SIMT
后，再依次进入 DAG Scalar 和 DAG SIMT。当前 DAG 目录只是目标布局，
尚未实现，不得把 ordinary 结果外推成 DAG 证据。
