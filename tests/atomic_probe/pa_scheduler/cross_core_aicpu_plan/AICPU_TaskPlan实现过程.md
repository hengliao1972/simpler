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
| S1 | 真实 AICPU orchestration SO 与 Plan backend | 独立 producer 已闭合，A5 正式 launch 未接入 |
| S2 | ordinary Scalar Build 端到端 | 独立 CPU 状态机已闭合，完整生产路径未接入 |
| S3 | ordinary SIMT Build 端到端 | 未开始 |
| S4 | DAG Scalar Build | 未开始 |
| S5 | DAG SIMT Build | 未开始 |
| S6 | 观测闭合后的性能收敛 | 未开始 |

当前可以讨论 S0 协议、独立 A5 可见性探针，以及 S1 的真实 orchestration
Host `dlopen`/AArch64 构建证据。AICPU producer 尚未与 CCEC Scalar Build
组成 A5 端到端路径，因此 S2 及之后仍没有性能结论。

## 3. 2026-08-10：S0 协议门槛

### 3.1 已写入源码的协议

当前只有
[公共 Plan 协议头](common/aicpu_plan_protocol.h)和一份 atomic/DCCI 源码覆盖检查
草案。协议头已表达：

- `RuntimeTaskPlanHeader` 为 64B；
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
  Host 没有写入任何 task identity/payload；现有旧 replay kernel 的 A5 B1
  回归 PASS。该回归只证明新增分配和引用没有破坏旧入口，不证明新 Plan
  consumer 已运行。
- A5 PA B1/B256：`NOT RUN`。
- 无泳道 startup→FinalDrain 性能：`NOT RUN`。
- 小于 1ms 目标：无结论。

### 3.4 S0 收口结论

S0 的公共 wire、CPU malformed gate、CCEC 模板实例化、A5 多 cache-line
发布/获取探针，以及正式 CCEC Host 外置 storage 生命周期均已闭合。容量
固定取 AICore consumer 的编译期上限，Host 只发布地址/容量，不生产业务
内容。Scalar 只能 invalidate StorageRef ordinary payload，Plan control 的
atomic-only cache line 不执行 DCCI。

当前证据可以声明“公共 Plan 可见性协议在独立 A5 探针上成立”，不能声明
“AICPU Plan PA 已正确运行”或已有端到端性能收益。

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

这一步证明的是“真实 callback 可以生成 canonical Plan”，并不是 A5 AICPU
执行证据。下一阶段要把该 AArch64 producer 通过独立 AICPU loader 接到同一
GM Plan storage，再让 96 个 Scalar 使用中央 ticket Build。

## 5. 2026-08-10：S2 ordinary Scalar 独立 CPU 状态机

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

这一门槛刻意只实现最小 exec cell 和 insertion completion，不展开普通
TensorMap payload、完整 `FinishSharedWinnerSubmitBody`、真实 kernel 或
FinalDrain。因此它是 S2 的协议门槛，不是 ordinary Scalar 的端到端完成
证明，也不产生性能结论。
