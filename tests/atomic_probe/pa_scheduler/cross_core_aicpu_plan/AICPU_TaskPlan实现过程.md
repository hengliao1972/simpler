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
| S3 | ordinary SIMT Build 端到端 | 已正式接通；CPU/CCEC 门槛与 A5 B1/B256 perf-clock 功能全部 PASS |
| S4 | DAG Scalar Build | 按当前要求冻结，未开始 |
| S5 | DAG SIMT Build | 按当前要求冻结，未开始 |
| S6 | 观测闭合后的性能收敛 | 已保留 Scalar 串行基线并接通 ABI v3 双 pipeline policy；同提交 A5 因果 A/B 尚待补齐 |

S0--S3 在 ABI v2 上闭合了 ordinary Scalar/SIMT 的完整功能路径；这些
历史阶段和数值继续保留在第 3--7 节。S6 已把公共 Plan 协议升级为 ABI
v3，并在 ordinary Scalar 中长期保留两个编译期 policy：默认
`plan-ahead-closed` 是串行正确性/性能基线，`streaming-future-pN` 是允许
Plan/Build 重叠的实验路径。两者仍共享真实 AICPU callback、同一份
canonical Plan、ordinary TensorMap 严格插入、Build、96 Scalar
Execute/FinalDrain；差异只在 producer admission、Host launch 顺序和
consumer ticket 解析。当前不做 DAG，也不把跨 ABI 的旧数据冒充同提交
因果 A/B。

## 3. 2026-08-10：S0 协议门槛

### 3.1 已写入源码的协议

S0 当时只有
[公共 Plan 协议头](common/aicpu_plan_protocol.h)和一份 atomic/DCCI 源码覆盖检查
草案。协议头已表达：

- `RuntimeTaskPlanHeader` 为 64B；该阶段公共 wire ABI 为 v2，header 的
  16bit `adapter_data` 由算子 adapter 定义，PA 用它显式携带真实
  Alloc callback 的 `batch_start`；
- 完整无指针 payload 最大为 4416B，共 69 条 cache line；
- `RuntimeTaskPlanCell` 当前为 4608B stride，包含独立 128B control、完整
  payload 和尾部对齐；
- payload 可表达 Tensor/TensorCreateInfo 值、symbolic output reference、
  scalar 和显式依赖；
- PlanCell 数组由 `RuntimePlanStorageRef` 引用运行期外置 GM，避免按最大
  task 容量固定膨胀 SchedulerState；
- producer 按 payload clean、control publish 的逻辑顺序发布；
  StreamingFuture 逐 task 完成 barrier，PlanAheadClosed 在 Close 统一收口；
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

生产
`runtime/dist_engine/common/cross_core_simt_request_protocol.h` 不是待复用的
Plan wire，而是“AICore replay producer -> VF consumer”的另一套
request ABI。它的 control stride、tensor slot、OutputRef 和 header
字段都与 canonical Plan v2 不同。ordinary Scalar/SIMT 只能共享
`RuntimeTaskPlanCell` ABI v2；另一套 request 只能供 atomic/DCCI、
VF leader 和 publication 的实现方式参考，不做二次拷贝或强转。

### 3.2 本轮文档收敛

本轮只修正设计口径：

- 删除“`RuntimeTaskDesc` 恰好 64B”的错误定性；
- 明确 64B 是 header，完整通用 payload 上限是 4416B；
- 明确 PlanCell 是运行期动态外置数组，Host 只提供容量和 GM 地址；
- 明确 canonical Plan v2 是 Scalar/SIMT 唯一 wire ABI；生产
  SIMT request 布局不兼容，仅复用实现经验，不保留两套 Plan ABI；
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
- 无泳道 pipeline 性能：已取样，见 5.4 节；性能目标与 A/B 统一只看
  B256，目前仍未达成。B1 只作为功能 smoke。

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

### 5.4 ABI v2 的 A5 性能证据

下表是当时 ABI v2、跨轮 cache 修复后的同一 trace-free 产物，使用真实
`real-compute=6,28,4,1`，在同一 Host 进程和同一 Plan GM allocation 内
连续运行 5 轮得到的中位数。它不是把不同构建或不同轮次的局部最优值
拼在一起：

| workload | plan_time | producer_exec | aicore_time | startup→FinalDrain | pipeline_e2e | 结果 |
| ---- | ----: | ----: | ----: | ----: | ----: | ---- |
| B256 | 2690.859us | 2626.289us | 2491.516us | 2460.846us | 5180.220us | 5/5 PASS |

五个时间口径必须分开解读：

- `plan_time` 是 Host wall time，从 Path-A AICPU owner launch 前到对应
  stream sync 结束；它包含 Host/Runtime Path-A launch/sync 的固定成本，
  不能称为“AICPU callback 执行时间”。
- `producer_exec` 由 AICPU owner 内部起止时钟得到，才是真实
  orchestration callback + canonical Plan 打包的 device 执行窗口；它还包含
  owner 内的输入解析、backend bind 和 Close，但不包含 Host Path-A launch/sync。
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
首轮仍存在 Path-A 首次执行 warm-up 效应，必须与 warm 中位数分开：首轮 B256 为
`8548.902/2686.402/2934.944/11484.044us`。Host 侧 owner DSO 的加载和
handle 初始化发生在 `pipeline_begin` 之前；这里观测到的是首次 Path-A
执行及 runtime warm-up，不把它武断归因成 callback 业务代码。

B256 warm 仍约 5.1--5.2ms，说明其主要矛盾已经转成随 task 数增长的
Plan 打包和 AICore Build/Execute，而非单纯一次性初始化。

上表只能声明 ordinary Scalar 功能闭合后的当前成本。B1 以后只作最小
功能 smoke，不记录或汇报性能，也不保留性能泳道。B256 仍未达到目标：
AICore device 内窗口中位数为
2460.846us，完整 pipeline 中位数为 5180.220us。不能只报某个子阶段，
也不能用 Plan-only、Host oracle 或 `producer_exec` 改写 pipeline 口径。

## 6. ordinary 功能停止点

ordinary Scalar 和 ordinary SIMT 的功能基线都已成立。S3 完成时曾按
用户要求停止继续功能开发；之后只重新开启 S6 的 ordinary pipeline
性能优化，不扩大功能范围。DAG Scalar 和 DAG SIMT 继续冻结，DAG 目录
只是目标布局，没有实现与 A5 证据，不得把 ordinary 结果外推成 DAG
证据。SIMT 首次单次样本也不是可用于性能结论的稳态基线。

## 7. 2026-08-10—2026-08-11：S3 ordinary SIMT

### 7.1 已冻结的首版边界

S3 不复制旧 standalone 的 `FullPaTaskPlan`，也不把 canonical Plan 转成
生产 `SimtBuildRequestCell`。这两条旧路径分别含固定 PA 公式和不同的
wire layout，不能作为 AICPU Plan 的真实 consumer。

首版仍是 Plan-ahead：AICPU 先发布并关闭 ABI v2 Plan；block0/AIV0 再
启动一个 128-thread persistent VF。4 个 warp 仅各自 lane0 参与 Build，
通过同一 `build_next` 动态领取 task。每个 task 的语义必须来自 PlanCell
header/payload，代码禁止用 `task_id % 5`、Host task table 或 device 固定
PA 公式恢复 kind、engine、shape、fanin。

本阶段先选动态 ticket，而不是立即采用 standalone 的静态 residue，原因
是前者对任意 task 成本分布都成立，并能先隔离“Scalar Build 改为 SIMT
Build”这一项变化。静态 residue 会消除每 task FetchAdd，已作为后续独立
性能候选保留；只有在完整 ordinary Build 与 Execute 语义闭合后才做 A/B。

### 7.2 已完成的 CPU 协议门槛

新增：

- `ordinary/simt_build/common/simt_plan_build_protocol.h`；
- `ordinary/simt_build/test/test_simt_plan_build_protocol.cpp`；
- `ordinary/simt_build/cpu/build_protocol.sh`。

门槛直接复用公共 `RuntimeTaskPlanCell` ABI v2。builder 外层先
用 `AttachClosedPlan` 一次性校验 closed/frontier/fatal 并缓存 N；
4 个 warp leader 随后只用 `TakeAttachedBuildTicket` 动态消费
closed Plan，每张票只保留必要的 `build_next FetchAdd`。task 的
function id、adapter data、flags、engine、core/sync、tensor/scalar 和
explicit dependency 均从 Plan 逐字段验证；没有 PA task 周期公式。

普通与 ASan+UBSan 两种构建均通过以下 task 数：

| N | 预期 build-next 终值 | 结果 |
| ----: | ----: | ---- |
| 0 | 4 | PASS |
| 1 | 5 | PASS |
| 3 | 7 | PASS |
| 4 | 8 | PASS |
| 5 | 9 | PASS |
| 41 | 45 | PASS |
| 257 | 261 | PASS |
| 1280 | 1284 | PASS |

每组均验证：每 task exactly-once、cell N 从真实初值 N-1 以
CAS 推进到 N、4 个不同 leader 各参与一次且各 arrival 一次、唯一
last arrival、`build_release=N`。N=41 会刻意延迟 task0，证明后继
leader 即使已取到高 task 也不能越过 completion 链或提前 release。
所有等待均同时观察 fatal 并有有界 deadline，不用永久 spin 掩盖
失败。control 未知位与 payload 非法 engine 两类破坏均经过完整
leader 消费路径，必须发布 fatal 且保持 release pending。

这只是 CPU 内存模型下的协议门槛，不证明 `async_invoke`、VF
`__simt_callee__` 调用闭合、A5 SIMT DCCI 或完整 Materialize/TensorMap/
SharedExecCell。

### 7.3 已完成的 CCEC compile gate

新增 `ordinary/simt_build/ccec_probe/`，直接包含公共
`RuntimeTaskPlanCell` ABI v2 和四 leader 合同。真实 CCEC 产物已闭合：

- `static __simt_vf__ __aicore__ LAUNCH_BOUND(128)` 与 mixed AIV entry
  在同一 TU；
- `async_invoke` 后以 V→S flag/wait 收口；
- 仅 4 个 warp lane0 用返回型 atomic 观察 Plan control，按
  `published_lines` 逐行生成 DCCI，再次观察 control 并校验通用
  header；
- 最终 ELF 仅一个 GLOBAL AIV entry、一个 LOCAL SIMT entry，
  metadata 为 `SIMD_SIMT_MIX_VF=4`；
- CCEC 使用 `-Werror`，bitcode intrinsic、无 relocation/未定义
  GLOBAL 等静态门槛全部 PASS。

该探针刻意不领取 `build_next`，不执行 Materialize、ordinary
TensorMap 插入、Fanin 或 Execute。`asc_dcci_single +
asc_threadfence` 能被编译也不等于已证明 A5 内存模型。下一阶段
是复用真实 Build body 并闭合 B1 全链路；当前仍不得宣称
ordinary SIMT 已实现或已有性能收益。

### 7.4 通用 ordinary writer 门槛与 Build population 参数化

继续审计 Scalar PlannedBuild 时确认：现有
`FinishSharedWinnerSubmitBody` 会以 `ValidatePreparedPaWriterShape`
拒绝 `ordinary_count!=0`，正式 PA Register 也只提交 UP 的三个 symbol
writer。这是 PA 快路径，不是 ordinary TensorMap 的完整实现。SIMT 版
因此不照搬该 Finish，而把通用链固定为
`Wait turn → PublishSharedTaskWriterMetadata → Handoff completion`。

新增 `test/test_simt_ordinary_writer_gate.cpp`，直接实例化现有通用
`PublishSharedTaskWriterDelta` 与 `CollectSharedFanin`，没有复制一套
测试专用 map。4 个 leader 动态消费 17 个 task，并验证：

- `ordinary_count>0` 时真实发布 bucket slot payload、seq 与 tail；
- 零 writer task 不写 map，但仍从 N-1 严格交棒到 N；
- reader 2 刻意等 future writer 5 已发布后再查询同一区域，结果仍只能是
  `writer 1 < consumer 2`；
- 重复 completion 与损坏的前序 completion 均 fail-closed。

严格 `-Wall -Wextra -Werror`、ASan+UBSan 与 50 轮普通重复均 PASS。

同时把 Runtime Plan 的 Build population 从固定的 `kWorkers` 抽成
`kRuntimePlanBuildWorkers`：Scalar 默认值仍是 96，行为和 N+96 终态不变；
SIMT 构建会显式配置为 4，而 Execute/FinalDrain population 继续保持 96。
release 等待已从 Scalar arrival 中抽成独立 helper，后续 95 个非 builder
Scalar 可以只等待 SIMT 的 N+4/4-arrival 收口，不会误增
`build_workers_done`。

### 7.5 为什么不能直接把 Scalar Build 编译成 VF

本阶段先尝试了最小改造：把现有完整
`BuildRuntimePlanTask<SimtOps, false>` 以 `__simt_callee__` 身份重新实例化，
而不改业务流程。结果证明这条路不能作为正式实现：

1. 公共头内 callback lambda 硬编码为 Scalar `__aicore__` 身份，直接调用
   SIMT helper 会被 CCEC 前端拒绝；
2. 探针内临时提升 lambda 身份后，完整调用图可以生成优化 LLVM bitcode，
   但 `TaskArgs::TensorPointer` 的 local/GM runtime union 在 machine-code
   后端触发 `error pointer address space cast`；
3. 即使隔离 writer-delta，exec payload 的 local/GM descriptor union 仍会
   独立触发相同错误。固定成单一 GM 分支可越过部分错误，说明阻塞来自
   runtime 地址空间选择，而不是 PlanCell 或最外层 mixed VF 壳。

负向探针同时证明：canonical Plan acquire/decode、真实
`ordinary_count=1` metadata publication，以及 static VF + mixed AIV
`async_invoke/wait` 壳都能单独生成 object。问题的精确边界是“把带
Scalar pointer union 的完整调用图整体搬入 VF”，不能由此得出 SIMT
TensorMap 或 Plan 协议本身不可用。

### 7.6 窄 canonical Build 与双 TU 身份隔离

新增 `ordinary/simt_build/common/simt_plan_task_builder.h`。它不构造
`TaskArgs`，而是把 canonical Plan v2 解码到仅含值语义的
descriptor/create-info/reference/scalar scratch，并依次执行：

```text
Plan acquire + canonical validation
  -> fresh output Materialize/publish
  -> ordinary/symbol writer delta
  -> completion[N-1] wait
  -> generic ordinary/symbol metadata publication
  -> completion[N] handoff
  -> ordinary/symbol/explicit fanin
  -> metadata vend/flag 或 SharedExecCell BUILT
```

该实现没有 `task_id % 5`、`FullPaTaskPlan` 或 Host task table。PA 的
kind、engine 与 batch provenance 仍只来自 Plan v2 header；Build 主链本身
只依赖通用 tensor tag/reference/metadata/exec 合同。

CCEC 正向门槛采用两个 TU：AIV/VF TU 只以 SIMT 身份包含窄 Build，VF
join 后调用另一个以 Scalar 身份编译的 scheduler continuation。这样避免
include guard 把同一 helper 永久锁成错误 device identity。严格
`dav-c310-vec -O3 -Wall -Werror` 已生成并静态链接完整
`BuildCanonicalPlanTask` machine object；最终产物含一个约 67.9KiB 的
LOCAL VF、一个 GLOBAL mixed entry，Scalar continuation 已解析且没有
undefined GLOBAL。这个门槛证明完整 Build 模板已经进入真实机器码，不是
只 include 头或只生成 LLVM IR。

对应 CPU 门槛把 Runtime 直接映射到真实 `SchedulerState`、
`SharedTensorMapSidecar`、per-task insert completion、`TaskCell` 和
`SharedExecCell`，覆盖非周期 task 序列、fresh output、
`ordinary_count>0`、symbol history、future writer 过滤、strict
completion、三类 fanin、metadata/AIC/AIV route，以及四 leader 的
`N+4` 收口。普通 `-Werror` 与 ASan+UBSan 均 PASS。

### 7.7 A5 writer 发布与失败收口边界

窄 VF 不把 reader invalidate 冒充 writer clean-out。跨核 ordinary payload
固定逐 64-bit 使用 `asc_stcg` bypass store，随后 `asc_threadfence`，最后
才发布对应 atomic control；这覆盖 output descriptor、writer history、
ordinary region、metadata vend 和 exec payload。`asc_dcci_single` 只用于
reader 在观察到 control 后逐 cacheline invalidate。当前 CCEC source/IR
门槛已经锁定 `stcg -> fence -> atomic`，并拒绝 writer 侧 DCCI。

失败路径也不能伪装成可恢复事务。completion 交棒前若 output 预留、
descriptor、published 或后续 metadata 失败，必须撤销本 task 已预留/已发布
的 fresh output 控制字；heap FetchAdd 和可能已提交的 metadata 前缀不做
局部倒退，而是发布全局 fatal，禁止 Execute，并由下一轮 Host 完整重置
SchedulerState、sidecar 与 heap control。也就是说当前合同是“尽力撤销独占
output + 整轮 fail-stop”，不是在并发副作用后继续本轮或复用半发布状态。

上述仍然只是生产接线前门槛。compile harness 的紧密 atomic 数组、单 heap
cursor 和简化 metadata cell 不能作为 A5 正确性证据；正式 runtime 必须直接
映射现有 128B 隔离控制字、8-shard heap、真实 ordinary ring/history、
TaskCell vend/flag 与 SharedExecCell。现有 Exec policy 要求
`build_owner < 96`，所以四个 leader 的 Build owner 固定为 `0..3`；它与
独立的 Execute owner 字段不绑定。leader 身份与 Build 统计通过独占 GM
sidecar 在 VF join 后归并，不能覆盖后续 96 个 Scalar Execute worker 的
结果，也不能通过扩大 owner 取值范围绕过现有 Host/device 校验。

### 7.8 真实 SchedulerState 映射已经闭合

2026-08-11 新增
`ordinary/simt_build/common/simt_real_state_runtime.h`，把窄 canonical
Build 直接接到 ordinary Scalar 已有的真实状态；它只保存一个
`SchedulerState *`，不分配第二份 TensorMap、heap、TaskCell 或 ExecCell。
映射关系固定为：

- fresh output 直接使用 `SharedOutputCell` 的 `last_writer`、descriptor 和
  `published`；
- symbol history 使用真实 `SharedWriterHistoryCell`；
- ordinary writer 使用真实 bucket、slot、absolute sequence 和 tail；
- output heap 直接复用 8-shard `ReserveSharedOutputHeap`；
- 严格插入 completion 仍位于每 task tournament root 的第二条隔离线；
- metadata-only task 直接发布真实同一 `TaskCell` 的 `vend/flag`；
- executable task 直接发布正式 `SharedExecCell::BUILT`，并共用
  `SharedExecFatalControl`。

这层桥不解释 PA kind，也不从 task id 恢复 batch、engine 或函数。调用方
每次取得 ticket 后先 `BindTask(task_id)`；算子 adapter 只能核对 Plan v2
header 中的 `adapter_flags/adapter_data/function_id/engine`。四个 SIMT
leader 的 Build owner 使用 `0..3`，满足现有 device/Host
`build_owner < 96` 合同；物理 leader 身份另写 report，不借越界 owner
编码诊断信息。

ordinary 写侧没有调用 Scalar 的 ordinary-store + writer DCCI 实现，而是
保留相同的 head/tail/absolute-seq 算法，把 slot payload 改成四个
`stcg` word，随后 fence，再依次发布 seq 和 tail。reader 仍直接调用真实
`SharedLookupRegion<..., NoReclaim=true>`，保持 control、DCCI、payload、
control 双检以及 `[N-H,N)` writer 窗口。metadata completion 则使用两个
返回型 atomic 发布 vend 和 flag，不能改成 stcg。

审计期间发现并修正了一个 C++ 对象模型问题：布局相同并不允许把
`SharedWriterHistoryCell` 当作窄 history 类型，或把
`SimtWriterRegion[]` 当作 `SharedRegionValue[]`。当前 history accessor
返回真实类型，ordinary preflight/append 逐项构造真实 value；source gate
禁止这些 structured `reinterpret_cast`。保留的 raw `uint64_t *` 只用于
A5 stcg wire store，不是把一个已启动的结构对象冒充成另一个结构类型。

新增的 CPU real-state gate 使用真实约 1GiB `SchedulerState` 虚拟映射，
动态覆盖：fresh descriptor/published、symbol future-writer 回溯、同 bucket
两个 ordinary entry、8-shard heap、metadata completion、owner 0..3 的
ExecCell BUILT、scheduler/Plan 双 fatal、越界 accessor，以及空 heap base
在任何 FetchAdd 前拒绝。O2、ASan 和 UBSan 全部 PASS。

独立 CCEC gate 以 `dav-c310-vec -O3 -Werror` 生成完整 machine object 和
优化 IR，锁定真实状态地址、s32/s64/u64 atomic、stcg、fence 与 reader
DCCI。它同时复现并消除了旧的 `TensorDesc` local/GM 转换后端宽度问题：
SIMT 路径按同一溢出规则构造 value `SharedRegionValue`，直接进入真实
generic lookup，而不重新引入 Scalar pointer union。

失败语义仍是 terminal fail-stop，而不是事务回滚。completion 前会尽力
撤销本 task 独占的 output 控制字；heap cursor、symbol/ordinary 已发布前缀
或 completion 后失败不并发倒退。任一失败必须同时发布 scheduler 与 Plan
fatal，禁止 build release 和 Execute；下一轮由 Host 全量重置相关状态。

### 7.9 双 TU production-shape runtime gate

新增 `ordinary/simt_build/ccec_runtime/`，验证正式接线所需的双 TU 形态
能够完整生成和静态链接：

```text
mixed AIV0 entry（SIMT 身份）
  -> 在 VF 前记录起点并获取 closed Plan
  -> async_invoke 128-thread VF
  -> 4 个 warp lane0 Attach/Take/Bind/Build/Arrive/Release
  -> success 或 fatal 都执行 V->S join
  -> DCCI 获取 4 条隔离 leader report
  -> 调用另一 TU 的 Scalar continuation
```

四条 leader report 以及 launch/continuation report 各占 128B，writer 使用
`stcg -> fence -> magic`，AIV0 join 后才 invalidate/read。Scalar
continuation 会再次返回型读取 build release、Plan fatal 和 scheduler fatal，
不是空占位。当前最终 ELF 只有一个 2140B GLOBAL mixed entry、一个
93376B LOCAL VF 和一个 300B LOCAL Scalar continuation，metadata 为
`MIX_VF=4`，没有未定义 GLOBAL 或 relocation；动态 ticket、完整 Build、
四次唯一 arrival、release 以及 fatal/success continuation 都存在于优化 IR。

在该阶段结束时，这仍然只是 production-shape gate，不是正式
mode：它当时没有修改
`scalar_build` 的 Host、kernel、manifest、`run.sh` 或 Host oracle，也没有
在 A5 上验证 AICPU->SIMT、leader->AIV0、AIV0->Scalar 的可见性。下一阶段
必须把 backend hook 接入正式 RunScheduler，并把 Scalar Build 统计口径改成
SIMT 直接状态 oracle；不能通过伪造 96 个 Scalar 的 submits/wins 来满足旧
Host 校验。这些正式接线已在 7.10 完成。

### 7.10 正式 CCEC mode 与 A5 功能闭合

2026-08-11，`ordinary/simt_build` 已从 production-shape gate 接入
正式 CCEC Host/kernel/AICPU producer/manifest/`run.sh`。最终路径是：

```text
AICPU 唯一 producer
  -> 执行真实 orchestration callback
  -> 按无缺口 task id 发布 canonical Plan v2
  -> frontier == closed == N

AIV0 正式入口
  -> 只读核验 backend/ABI/variant/workers/Plan storage
  -> async_invoke 128-thread VF
  -> 4 个 warp lane0 leader 动态领取 N 个 Build task
  -> build_next == N + 4，arrivals == 4
  -> N>0 时确认 completion[N-1] == N-1
  -> 唯一 last leader 发布 build_release == N
  -> success/fatal 都完成 V->S join

96 Scalar continuation
  -> 32 AIC + 64 AIV 共同 Execute
  -> FinalDrain 收口
```

其他 95 个 Scalar 不冒充 Build worker，只 attach closed Plan 并等待
四 leader release；AIV0 在 VF join 后作为第 96 个 Scalar 进入同一
Execute/FinalDrain continuation。因此 Build population 是 4，Execute
population 始终是 96，不会为 SIMT Build 永久牺牲 AIV0。

Alloc task 保留在 canonical Plan 中：PlanCell 仅携带逻辑 Alloc
identity（`EngineClass::MetadataOnly`）和 fresh-output `TensorCreateInfo`，
AICPU 不分配 output heap。真实
8-shard heap reserve、descriptor 构造和 `SharedOutputCell` publication 都由拿到
Alloc ticket 的 SIMT Build leader 完成。Alloc 完成 metadata vend/flag 但不发布
executable kernel；因此“Alloc 在 Plan 中”不等于“Plan 已完成 heap
Materialize”。

正式 Host 在 AICore sync 和全部权威计时窗结束后，D2H 回读
AICPU 实际发布的 immutable PlanCell 前缀。writer/output oracle 从
Plan tensor tag、CreateInfo、inline descriptor 和 OutputRef 通用投影：

- fresh output 数量、大小、descriptor 和 8-shard heap 区间；
- reference `Inout/OutputExisting` 的 writer history 与最终
  `last_writer`；
- non-reference ordinary writer 的 bucket/head/tail/seq/payload；
- 每 task 严格 insert completion 和未用状态。

这部分校验不从 PA group/kind 公式猜 writer/output。shared writer-history
key 严格使用 one-based ABI：
`producer * kSharedOutputMaxPerTask + slot + 1`，其中 0 永久保留为
invalid/unset；这保证 producer 0/slot 0 的 key 为 1，SIMT writer、reader
和 Host projection 三方一致。Plan snapshot 是事后 oracle，不是 Host 调度
输入，也不进入 pipeline 计时。

正式 AIV ELF 最终 metadata 需要 8344B SIMT stack 和 8320B divergence
stack。Host 的 SIMT backend 不再使用默认 `aclInit(nullptr)`，而是在
ACL 首次初始化时传入按 512B 容量步长向上取整的
`simt_stack_size=8704` 和 `simt_divergence_stack_size=8704`，分别覆盖
8344B/8320B 的最终 metadata 要求。该修复
只在编译期 SIMT backend 生效，Scalar backend 仍保持原有 ACL 初始化。

首次正式 A5 `perf-clock` 功能运行的 run1 精确结果如下：

| workload | startup→FinalDrain | plan_time | producer_exec | aicore_time | pipeline_e2e | 结果 |
| ---- | ----: | ----: | ----: | ----: | ----: | ---- |
| B256 | 62359.163us | 8871.954us | 2959.921us | 62884.823us | 71756.980us | execution/semantic/postprocess PASS |

该 B256 数据只是 formal ordinary SIMT 首次单次运行的功能证据，不是
多轮 warm 中位数，不能当作稳态性能基线。B1 只作功能 smoke，不进入
性能表或汇报。B256 的 AICore device 内主体已达 62.359ms，完整
pipeline 为 71.757ms，离 1ms 目标很远。本阶段只宣称
B1/B256 的 execution、semantic 和 postprocess 全部闭合，不宣称
SIMT 性能达标或优于 Scalar。DAG 没有实现，且已按用户要求
在此停止继续功能开发；随后单独开启的 ordinary Scalar pipeline 性能工作
见第 8 节，不改变本节的 S3 功能结论。

## 8. 2026-08-11：S6 ordinary Scalar 双 pipeline policy

### 8.1 为什么长期保留两条路径

串行 Plan-ahead 不是准备在并发版本完成后删除的过渡代码。当前用
`PA_RUNTIME_PLAN_PIPELINE_POLICY` 编译期选择 ordinary Scalar 的阶段关系：

| 编译值 | 产物身份 | 定位 |
| ---- | ---- | ---- |
| `0`（默认） | `plan-ahead-closed` | 长期正确性基线，也是量化并发代价的稳定对照组 |
| `1` | `streaming-future-pN` | 实验路径；`N=PA_AICPU_PLAN_READY_PREFILL_TASKS`，默认 128 |

policy 被编入 Host、AICore kernel、AICPU owner/producer 与运行身份，不是
一次运行中的动态分支。默认值保持 0；串行构建拒绝显式 prefill，避免一个
看似无效的参数悄悄改变基线身份。当前普通 TensorMap 之外的 DAG 不在
这轮工作范围内。

### 8.2 三处差异和共享主链

两条 Scalar 路径只允许在以下三处不同：

| 边界 | `plan-ahead-closed` | `streaming-future-pN` |
| ---- | ---- | ---- |
| producer admission | producer 在生产期保持 `NotReady=-2`；完整 Plan 准备好后才做 `Open=-1 -> Closed=N`，Close 还要求 `build_next==workers_done==0` | 连续 Published 前缀在真实 batch 边界达到 prefill 后发布 `Open=-1`；Close 可与 Build 并发 |
| Host launch | AICPU launch、plan-stream sync、再 launch AICore | AICPU 与 AICore 使用两条 stream，consumer 在 producer 后立即 launch，最后两条 stream 都必须闭合 |
| consumer ticket | 96 Scalar 先 attach immutable Closed Plan 并缓存 N；热循环只有 `build_next.FetchAdd`，首张 `ticket>=N` 直接退出 | Ready 后 96 Scalar 可领取尚未发布的 future ticket，并轮询该 ticket 的 cell；观察 Close 后按最终 N 解析 |

以下内容没有分叉：真实 orchestration callback、唯一 AICPU producer、同一
ABI v3 `RuntimeTaskPlanCell`、Plan decode/validate、Materialize、Alloc 的
真实 heap reserve、ordinary TensorMap 严格 `N-1 -> N` 插入、Fanin、
exec-cell Build、engine-routed Execute 和 FinalDrain。两条路径也都禁止
Host task identity、`task_id % 5` 和 device PA 固定公式。

ABI v3 把 `closed_task_count` 明确定义为四态：`-3=ReadyFailed`、
`-2=NotReady`、`-1=Open`、非负值 `N=Closed`。`ReadyFailed=-3` 只保证
closed control line 已发布，consumer 不得先读取可能尚未完成 reset 的其他
control。Streaming 的 future ticket 若先观察到 Empty，随后观察到 Close 且
`ticket<N`，必须再次 acquire 同一个 cell；第二次仍 Empty 才能报告
MissingPlanCell，不能把 Close 与 cell publication 的独立可见性顺序误判成
丢任务。

AICPU producer 同时改为 single-Pack：`Finish` 在 callback 引用仍有效时
直接 Pack 到目标 GM cell，保持 control=Empty；下一次 `Begin` 或最终 Close
只补齐 final flags、校验同一份 GM wire，再按
`payload clean -> Published control` 的逻辑顺序发布。StreamingFuture
逐 task 完成 barrier，PlanAheadClosed 由 Close 统一收口。pending 状态只保留
一条隔离线内的 metadata，不再保存并二次复制完整 4416B payload。

正常成功终态对两条策略相同，其中 `W=96`：

```text
closed_task_count == N
build_next == N + W
build_workers_done == W
build_release == N
fatal == 0
```

最终两条 policy 都必须得到 `planned_frontier==N`。Plan-ahead 已在 attach
时验证 immutable Close/frontier；Streaming 的最后一个 arrival 必须在
producer 并发结束后重新核验最终 frontier。两条 policy 共享之后的 Build release、
Execute 和 FinalDrain 实现，避免为了做性能实验复制一套业务语义。

### 8.3 产物、manifest 与 raw 隔离

Scalar CPU/CCEC 产物按 policy 物理分目录：

```text
ordinary/scalar_build/build/<backend>/shared/plan-ahead-closed/<variant>/
ordinary/scalar_build/build/<backend>/shared/streaming-future-pN/<variant>/
```

Scalar CCEC manifest 升为 `pa_scheduler_artifacts/v10`，formal SIMT 为
`pa_scheduler_artifacts/v11`。除 Runtime Plan ABI v3、
容量、variant 和四件套 SHA256 外，还固定
`pipeline/launch_order/producer_ready/consumer_admission/prefill` 与
`scheduler_input`，并固定
`aicpu_task_trace_enabled=0|1`、`aicpu_task_trace_record_bytes=64`，
以及 operation trace 的开关、64B record、64 条固定余量和每 plan-cell
32 条容量。运行入口会同时检查 Scalar 36 行/SIMT 41 行 manifest、SHA、
源码新旧和编译期 Build
identity；因此不能把一个 policy 的 Host、kernel、AICPU SO 或
dispatcher 与另一个 policy 混用。

新采集的泳道输出目录名同样包含 `plan-ahead-closed` 或
`streaming-future-pN`。raw metadata 显式记录 Runtime Plan ABI、pipeline
及 `launch_order/producer_ready/consumer_admission/prefill`、
`runtime_plan_build_backend/build_workers/execute_workers/
build_trace_coverage`、`producer_task_count/task_count/task_kinds` 和
`runtime_plan_terminal`，converter 按相同合同校验。device Closed(N) 是
`task_count` 权威来源，并与 producer result、frontier 和 release 交叉；
Host 重建 task plan 仍只作 oracle。Scalar coverage 固定为
`scalar-task-detail`、W=96。历史 schema-v5 raw 若缺少 policy 字段，只允许在
`runtime_plan_abi<3` 时推断为 legacy `plan-ahead-closed`，并写出
`inferred_legacy_policy=true`；ABI v3 raw 缺少显式 policy 会直接拒绝。

S6 的正式泳道不再只看 AICore：每份结果必须同时包含 AICPU producer 与
AICore Build/Execute/FinalDrain。AICPU 使用 `CLOCK_MONOTONIC_RAW`，AICore
使用 `SYS_CNT`；主体前、后分别采 4 组四时间戳
`(AICPU send, AICore receive, AICore send, AICPU receive)`，以全部 8 组
offset 区间的交集完成映射，相关误差必须 `<=50us`。raw 必须逐组保留四个
原始时间戳、offset 上下界、round-trip、AICore service interval、轮次序号
与 nonce，不能只保留最终 offset。Host 只负责 launch/sync，不属于泳道
时钟域，不生成 Host lane，raw 也不保存 Host timestamp、clock bracket 或
lane metadata。

2026-08-11 补齐 AICPU 内部 TaskPlan 构建明细。Owner request ABI 从 v1
经 task trace v2 升到 operation trace v3；swimlane 构建由 Host 提供两块
64B 对齐 buffer。逐 task trace 在真实 Begin/Finish/publish callback 边界
记录 `CLOCK_MONOTONIC_RAW`；另一块记录 producer 实际执行的低层操作。
owner 闭合后统一 clean，Host 只在 stream sync 后回拷。raw 现在包含：

- `aicpu_scheduler_phases`：Owner 的 `OwnerSetup/BackendBind/
  Orchestration/BackendClose/OwnerFinalize` 五段精确分区；
- `aicpu_tasks`：逐 task 身份、output/payload 大小及
  `begin/orchestration/stage_payload/defer_publish/publish` 五段真实
  时间边界。`defer_publish` 明确表示 backend 必须等到下一
  callback（最后一个 task 则等 backend close）才能确定 flags；
- `aicpu_operations`：按 `owner_setup/backend_bind/task_stage/
  task_publish/frontier_advance/ready_publish/backend_close` 归属记录
  acquire load、`dc cvac`、`dc civac`、`dsb sy`、`isb`、关键 ordinary
  GM store 和最终 payload wire 校验。AICPU 不执行 AICore `dcci`，因此
  merged 名称坚持使用真实 ARM 指令 `dc_cvac/dc_civac`。

operation record 固定 64B，保存 task/scope/target、首末时间、calls、
cache lines、首末 cell index 和首末值。只合并语义完全相同且连续的操作：
例如优化前保留的 structural capture 中，bind 对 4352 个
cell-control 的 `dc civac` 和 acquire load 各形成一个
区间，但分别保留 `calls=4352` 与 index `0..4351`。这样不会退化成逐 cell
大文件，也不会因合并丢掉操作数量。Host 与 converter 都检查 count、record
size、drop、严格时间顺序、父区间包含和 target 语义，任一不闭合就拒绝
生成 merged。

converter 使用原有 4+4 四时闳相关样本映射每一个 AICPU
时间戳，不直接拼接两个时钟域，也不按 PA 周期公式伪造
task 时长。它还会 fail-closed 检查 Owner 五段无缝分割、
Closed(N) 的逐 task 全覆盖、callback 顺序、identity 和 69-line
payload 上限。最终 Perfetto 在独立 AICPU process 中保留 Owner 和
TaskPlan 两条 lane；task 外层区间与五个子区间形成可展开的
包含关系；低层 operation 继续嵌套在对应 Owner 或 TaskPlan lane 上。
新增逐 task、Owner 阶段和逐操作取时在 trace-free 构建中全部编译掉，
权威性能仍只取 trace-free warm 运行。

Host dlopen smoke 现在对两条 pipeline policy 分别运行 trace-off 和
trace-on 两种 native AICPU backend。trace-on 会真实执行 B256/1280-task
orchestration callback，逐条检查 task identity、engine/group/output/payload、
六个时间戳的严格顺序和前后 task 发布顺序，并检查七类 operation 均实际
出现、cache `calls=lines`。StreamingFuture 仍要求 bind 一次性准备全容量；
PlanAheadClosed 则检查按 128-cell 分块精确覆盖 `[0,N)`，且
`task_publish` 内的 DSB/ISB 计数必须为零。trace-off 反向检查
task/operation trace count 和 record bytes 全为零。因此明细不只是
converter 的合成 fixture，而是经过真实 callback 链生成和校验。

真实 A5 device 0 的 ordinary Scalar 闭合结果如下：

- B1 scalar-nop smoke 的 AICPU 5-task Plan、AICore 调度语义、atomic/
  DCCI 和零 drop 全部 PASS；
- B256、`real-compute=6,28,4,1` 的 capture 产生 1280 条 raw
  AICPU task、5 个 Owner phase，merged 中有 1 个
  `RuntimePlanProducer`、5 个 Owner 子区间、1280 个 task 外层区间和
  6400 个 task 子区间；五种子区间各 1280 个，没有空区间或
  错 lane；
- 该 structural capture 的 `pipeline_e2e=5915.159us`、
  `producer_exec=2580.959us`，仅用于结构取证，不作为无埋点
  性能数据；raw/merged 文件约为 4MiB/11MiB；
- trace-free B1 perf-clock 同样在 A5 通过，确认 owner phase 和
  task trace 未残留在权威性能构建中。

同日进一步完成 AICPU atomic/cache 明细实机闭合。B256、
`real-compute=6,28,4,1` 产生 25,138 条 operation record、零 drop：
18,964 次 acquire load、15,119 条 `dc cvac`、4,381 条 `dc civac`、
3,082 次 `dsb sy`、1,544 次 `isb`，以及 2,819 次关键 GM store 和
1,280 次 payload wire 校验。full trace 的 `pipeline_e2e=13193.825us`、
`producer_exec=9742.788us` 明显包含逐操作取时开销，只是结构证据，不能
替代 trace-free 性能。

formal SIMT 的两次 B256 额外运行中，AICPU 1280-task 明细已经
通过 Host 的逐条校验，但 merged 发布被既有 AICore 因果门槛
拒绝：最早 FDWIC 起点比 pre-correlation 早约
`4.23e9` tick，两次可重现。本次没有放宽该 fail-closed 规则来强行
生成 SIMT 图；这不影响 Scalar 实际图对 AICPU 明细的闭合。

### 8.4 AICPU+AICore joint structural capture 合同

pre-correlation 会在主体前实际启动 AICPU/AICore，因此联合泳道是校准后的
结构取证，不是未经预热的性能样本。每份结果必须显式标记：

```text
clock_correlation_warmup_before_pipeline = true
timing_scope = calibrated-structural-capture
performance_representative = false
```

绝对性能只取 trace-free、同进程 warm 多轮的中位数；joint capture 只回答
阶段位置、并行形状和重叠关系，不能与 trace-free 数值相减。分析器对
`RuntimePlanBuild`、真实 Kernel Execute 和 FinalDrain 分别做区间 union，
同一阶段内多个 worker 的重叠只计一次；再用映射后的 AICPU producer 区间
求各阶段 overlap、非重叠 tail 与 signed gap。Streaming 的结果必须按这些
真实交并关系分类，不得因 policy 名含 `streaming` 就预设已经形成有效重叠，
也不得把 Ready/PlanCell/Close 等待归入 Build 或 Kernel。

性能和正式泳道都只看 B256；B1 永久只作功能 smoke，不保留或汇报性能。
当前留存集合限定为两个 policy × 两种真实计算参数（`1,1,1,1` 和
`6,28,4,1`）共四份 joint structural capture。具体数值和文件路径只在四份
均完成语义、postprocess、零 drop、时钟相关和联合分析校验后写入日期目录
README。严格 A/B 仍要求同一源码 commit、ABI v3、single-Pack、device、
输入、trace 配置和边界；绝对值则由各 policy 的 trace-free warm pair 给出。

### 8.5 naive p128 的负结果

在双 policy 正式固化前做过一次 `real-compute=6,28,4,1`、B256、同进程
5 轮的 warm 诊断。串行 control 与“prefill 后立即放入 96 个
future-ticket consumer”的 naive p128 结果为：

| 路径 | plan time | producer exec | AICore Host span | pipeline E2E | 正确性 |
| ---- | ----: | ----: | ----: | ----: | ---- |
| 串行 Plan-ahead control | 2667.425us | 2590.039us | 2520.457us | 5187.986us | 5/5 PASS |
| naive StreamingFuture p128 | 4414.344us | 4336.106us | 6543.950us | 6575.206us | 5/5 PASS |

Streaming 的 `aicore_time` 从 AICore launch 起计，包含 Ready/PlanCell/Close
等待，只能当 upper bound，不能解释成纯 Build/Execute 用时。更有判别力
的是 producer 侧反压：Plan wall 增加 1746.919us，`producer_exec` 增加
1746.067us；按 1280 tasks 折算，producer throughput 约下降 40.3%。若用
串行 AICore span 作为诊断性 counterfactual，p128 只隐藏约 359.595us，
而 pipeline 实际回退 1387.220us，即 **5.188ms -> 6.575ms**。Plan 增量
减去隐藏量与该回退只差约 0.1us。

这组数足以否定“Ready + 96 future-ticket 自旋天然最优”的默认假设，但
还不是最终因果结论：串行 control 来自历史 ABI v2，实验路径来自 ABI v3/
single-Pack 开发状态。最终定量结论仍以 8.4 节要求的同 commit、ABI v3、
single-Pack warm pair 为准。

### 8.6 下一步只做 ordinary Scalar admission 优化

下一候选不是继续扩大 prefill 后的无界竞争人口，而是限制 Open 阶段可进入
Build 的 Scalar 数 `K`，Close 后再让全部 96 个 Scalar 加入并收尾。建议先
把 `K=8/16/32` 与当前 `K=96` control 分开测量，并把 K 编入构建身份、
manifest 和 raw；prefill 只作为另一条独立变量。这个 limited-K admission
目前**尚未实现**，不能在结果或目录名中宣称已有。

本轮只优化 ordinary Scalar 的 Plan/Build 阶段关系。DAG TensorMap、DAG
Scalar 和 DAG SIMT 都继续冻结，不在 limited-K 或 pipeline A/B 中顺带实现。

### 8.7 PlanAheadClosed 发布维护收缩

本节记录 2026-08-11 的第一阶段收缩及当时的协议结论；2026-08-12 的
同构门禁进一步证明 PlanAheadClosed 不需要逐 task clean，当前实现和最终
数据以 8.8 节为准。

operation 明细暴露出的 4,381 次 `dc civac`、3,082 次 `dsb sy`和
1,544 次 `isb` 中，有三类是 PlanAheadClosed 不需要的过度维护：

- consumer 在 Close 完成前根本不会启动，因此不需要每个
  task 都同步等待 `payload cvac -> DSB -> control cvac -> DSB -> ISB`；
- 生产期不需要每个 batch 重复发布 `planned_frontier`；
- B256 实际只有 1280 个 task，不需要在 bind 时为 4352-cell
  最大容量全量执行 `dc civac` 和 Empty 检查。

最终实现保留了每个实际 payload line 的 `dc cvac` 和每个实际
cell control 的 `dc cvac`；这两类是 AICPU ordinary store 对 AICore
consumer 可见的必需操作，没有删除。改动只是：

1. task publish 仍精确 clean payload/control，但不再做逐 task DSB/ISB；
   Close 发布最终 frontier 前的 DSB 一次等待全部先行 cache
   maintenance 完成；
2. `planned_frontier` 在生产期保持 0，Close 校验全部 Published
   cell 后一次推进到 N；
3. cell-control 复用准备按 128-cell 单调前缀分块执行，Close
   只要求覆盖 `[0,N)`。B256 的 `dc civac` 因此从 4352 次
   降为 1280 次，不再为 `cell[N]` 多准备一个 128-cell 块。

StreamingFuture 的全容量 bind 准备、逐 task payload/control 屏障和
生产期 frontier/Ready 发布全部保持原样，因为该 policy 的 consumer
会与 producer 并发读 future cell。

真实 A5 device 0，B256、1280 tasks、`real-compute=6,28,4,1`、
trace-free、同进程 warm 中位数如下。最终版为 20 轮，全部
execution/semantic/postprocess PASS；基线为优化前 10 轮。

| 版本 | Plan wall | Producer exec | AICore wall | startup→FinalDrain | Pipeline E2E |
| ---- | ----: | ----: | ----: | ----: | ----: |
| 优化前 | 2031.848us | 1944.186us | 2487.716us | 2449.952us | 4529.819us |
| 去掉生产期 frontier 重复发布 | 1914.995us | 1824.717us | — | — | 4401.727us |
| 再合并逐 task barrier | 1437.055us | 1343.615us | 2479.637us | 2449.607us | 3923.493us |
| 再收缩 cell-control 准备前缀（最终） | 1366.861us | 1281.816us | 2483.902us | 2451.542us | 3850.988us |

最终相对基线，Plan wall 减少 664.987us（32.73%），producer exec
减少 662.370us（34.07%），pipeline E2E 减少 678.831us（14.99%）。
AICore wall 只变化 0.15%，说明收益确实来自 AICPU Plan 维护收缩，
没有通过减少 task 数、计算负载或 AICore 工作量获得。

### 8.8 基于 AICPU/AICore 同构门禁删除 task.publish 维护

`tests/atomic_probe/aicpu_aicore_cache` 已把 Host/SDMA→AICPU 与
AICPU→AICore 分开验证。对当前 A5 main aicpu_scheduler，后一个方向的
Path-A 门禁支持以下最小发布合同：

```text
AICPU PlanAheadClosed producer
  ordinary store 完整 payload
  -> release atomic store(Published control)

AICore Scalar consumer
  return-ready atomic observe control
  -> DSB
  -> DCCI payload lines
  -> DSB
  -> decode/validate payload
```

因此 PlanAheadClosed 的 `task.publish` 删除 payload `dc cvac`、control
ordinary store + `dc cvac` 及显式 DSB/ISB，只保留 payload wire 校验和
release atomic control。StreamingFuture 的 consumer 会与 producer 并发，
仍保留旧的逐 task payload/control clean 与 barrier，不能跟随删除。

删除 control clean 后，上一轮 Published line 可能仍由 AICPU cache 持有；
若在 Host memset 后继续 `dc civac`，反而可能把旧 Published 写回覆盖 GM
中的 Empty。PlanAheadClosed 没有并发 consumer，所以复用协议同步改为：
AICPU 对即将使用的 128-cell 单调前缀 release-store Empty，再 acquire-load
核对，然后才 stage/publish。它主动重建 AICPU 对 cell control 的所有权，
不再消费 Host 对这些 cell 的清零。StreamingFuture 和 29 条真正的
Host/SDMA→AICPU Owner 输入维护仍保留 `dc civac`。

#### 8.8.1 结构计数

真实 A5 device 0，B256、1280 tasks、`real-compute=6,28,4,1`，full
operation trace 重新采集并通过 converter 全部门槛。下表以 8.7 节最终
结构图为旧版本；`Calls` 是合并前真实调用量，不是 Perfetto event 数。

| 操作 | 8.7 版本 Calls | 当前 Calls | 变化 |
| ---- | -----------: | ---------: | ---: |
| atomic acquire load | 15,382 | 15,382 | 0 |
| `dc cvac` | 14,864 | 16 | -14,848（-99.89%） |
| `dc civac` | 1,309 | 29 | -1,280（-97.78%） |
| `dsb sy` | 21 | 11 | -10 |
| `isb` | 18 | 8 | -10 |
| ordinary GM store | 2,564 | 1,284 | -1,280 |
| release atomic store | 0 | 2,560 | +2,560 |
| payload validation | 1,280 | 1,280 | 0 |

新增的 2,560 次 release store 由 1,280 次前缀 Empty reset 和 1,280 次
Published control 组成。`task_publish` 内精确得到 1,280 次 release store，
payload/control cache clean、DSB 和 ISB 均为 0；剩余 16 次 `cvac`、29 次
`civac` 和 Owner/Close barrier 不属于 task publish。完整 trace 共 16,714
条 operation record、`dropped=0`。full trace 的
`pipeline_e2e=7229.859us`、`producer_exec=3781.946us` 含逐操作取时，只作
结构证据，不能作为性能数据。

#### 8.8.2 trace-free 性能

优化前后都在同一分支工作区、同一 A5 device 0、B256、1280 tasks、
`real-compute=6,28,4,1`、perf-clock、同进程 20 轮下测量；两组均 20/20
execution/semantic/postprocess PASS。

| 指标 | 删除前中位数 | 删除后中位数 | 差值 | 相对变化 |
| ---- | -----: | -----: | ---: | -------: |
| Plan wall | 1356.843us | 1149.669us | -207.174us | -15.27% |
| Producer exec | 1279.676us | 1062.002us | -217.674us | -17.01% |
| AICore wall | 2475.957us | 2477.879us | +1.922us | +0.08% |
| startup→FinalDrain | 2447.496us | 2450.675us | +3.179us | +0.13% |
| Pipeline E2E | 3834.191us | 3626.891us | -207.300us | -5.41% |

AICore wall 与 startup→FinalDrain 基本不变，说明 task 数、workload 和
AICore 执行路径没有被偷减；约 207us 的端到端收益来自 AICPU producer
缩短，并直接落在串行 `AICPU Plan -> AICore` 的 Pipeline E2E 上。

#### 8.8.3 正确性与静态门禁

- A5 B1 scalar-nop=0 同一 Plan allocation 连续 5 轮全部 PASS，证明
  Host memset 后由 AICPU 覆盖 control 的跨轮复用可用；
- native Host smoke 同地址、不做 Host cell reset 的第二轮中，
  PlanAheadClosed 仍能覆盖旧 Published control 并精确 Close；
  StreamingFuture 仍必须拒绝该用法；
- AArch64 反汇编要求 PlanAheadClosed initialize 出现
  `stlr Empty -> ldar` 且无 cell `civac`，publish 出现 `stlr Published`
  且无 `cvac/civac/dmb/dsb/isb`；StreamingFuture 门禁保持原序列；
- CPU protocol、双 policy AICPU SO smoke、CCEC B1/B256 语义和泳道 converter
  均通过。详细命令与数据口径见
  `test_record/2026-08-12/README.md`。
