# PA 调度器分离版实现过程

本文按阶段记录 `cross_core` standalone 的实际开发过程、验证证据和遗留问题。目标是先在独立代码中证明“构建与执行可跨 Scalar 核转移”，再决定是否迁移到 Simpler；本文不使用尚未运行的测试结果替代证据。

## 记录约定

- 每完成一个阶段，立即补充实现范围、验证命令、结果和未闭合问题。
- CPU、CCEC IR、A5 动态结果分别记录，不能互相替代。
- 性能结论必须同时给出对照版本、运行参数和结果文件；正确性阶段不提前宣称性能收益。
- 每个功能阶段闭合后都做一次轻量性能/泳道抽查：普通增量先记录后继续功能；出现倍数级回退、单一区域吞掉大部分时间或明显违背协议预期时，必须先定位和修正，不带着结构性异常进入下一阶段。
- 当前只实现 CPU 与 CCEC，暂不实现 AscendC。
- 当前目录是便于验证机制的 standalone；第一版不修改 Simpler 真实 PA 路径。
- 当前性能目标保留 TensorMap 严格串行插入和 96 Scalar 自由 Build 竞争；
  不引入 `try_wait`、engine continuation 或 kernel/调度 overlap。

## 阶段总览

| 阶段 | 目标 | 当前状态 |
| ---- | ---- | -------- |
| S0 | 固定跨核执行包 ABI、状态机和 cacheline 所有权 | standalone portable ABI 已闭合；真实 TensorDesc 对照留到 S3 |
| S1 | CPU 确定性交错测试闭合协议正确性 | 已完成 |
| S2 | CCEC 最小 A5 跨核发布/领取探针 | 已完成 |
| S3 | standalone PA 接入构建/执行分离 | 已完成：S3a 与固定两候选异核 S3b 均已闭合，S3b 通过 CPU、A5 B1/B256 和 B256 full-swimlane |
| S4 | 受控的动态 Execute election | K2 首版已通过完整 CPU、CCEC 和 A5 B1/B256 门槛 |
| S5 | 独立扩大 Build owner 候选核拓扑 | S5a 与 S5b 全 96 Scalar 均已通过 CPU/CCEC/A5 B1/B256 |
| S6 | 基于累积证据做性能评估与容量/复用优化 | 进行中 |
| 贯穿观测门槛（不编号） | 泳道、submit-PMU 与无泳道端到端三条互不混算的证据链 | 端到端与 full-swimlane 已可用；cross-core submit-PMU 尚未接入 |
| Simpler 迁移门槛（不编号） | 评估并迁移到 Simpler 真实路径 | 未开始 |

## 2026-08-02：S0 首版协议与 ABI

### 本阶段做了什么

新增独立公共协议头：

- `common/shared_exec_protocol.h`
- 全局 `SharedExecCell` 保存一个 task 的可执行包；每个执行参与者只保存一个私有 `ExecutionToken`。
- `SharedExecCell` 的 control 独占 64 B cacheline，payload 从下一条 cacheline 开始。
- payload 上限按当前 portable ABI 计算为 4352 B，即 68 条 cacheline：32 个 128 B TensorDesc、16 个 scalar、16 条 fanin，再加 64 B header。
- 构建侧执行 `EMPTY -> BUILDING -> BUILT`：先完整写 payload，再一次性 flush 有效区，最后发布 `BUILT`。
- 执行侧只允许 `BUILT -> CLAIMED` 的唯一成功者 invalidate/copy payload；loser 不读取 payload。
- 领取完成后，后续 fanin、engine 启动和 completion 只使用私有 token，不再回读共享 payload。
- completion 顺序固定为 vend 发布、completion flag 发布、`CLAIMED -> DONE`，最后才能复位 token。
- 全局 fatal 使用独立 64 B atomic-only line，首错 CAS 后永不清零；cell 保留 `BUILDING`、原始坏 `BUILT` 或 `CLAIMED` 现场，不新增 per-cell `FATAL` phase。
- packed cell state 同时携带 task id；payload word0 高 32 位保留并强制为 0，避免第二个含义不清的 task 身份。
- builder destination、executor shared source、executor private token 三个 preload hook 已固定位置与范围，当前 CPU/CCEC 都是无行为版本，不参与可见性合同。

### 本轮审计后已收敛的点

- control 的 64-bit packed state 增加 `task_id`，避免同 owner、同 engine、同 payload 大小的两个 cell 被错误交叉完成。
- token 增加 `BINDING`、`VEND_PUBLISHED`、`COMPLETION_PUBLISHED` 和 `FAULTED` 阶段，保留 completion 部分发布现场；terminal fatal 会阻止重入，已成功的 vend/flag 不会重复。
- engine 完成条件纳入 `TryMarkExecutionTokenCompleting()`，不能仅凭本地 phase 直接发布完成。
- Build-N 在序列化 fanin 时拒绝负数、自依赖和未来 task，约束 producer 必须位于 `[0, N)`。

### 尚未闭合

- “每个 executor 只有一个 token”目前由后续调度器布局保证；公共 helper 只能拒绝传入的 busy token，S3 需要用实际 executor 状态数组把这一约束锁死。
- portable TensorDesc 当前按 128 B 固定，迁移真实路径前必须与生产 TensorDesc 的 `sizeof/alignof` 做编译期对照。

## 2026-08-02：S1 CPU 门槛完成

### 已建立的测试能力

新增：

- `protocol_probe/test/test_shared_exec_protocol.cpp`
- `protocol_probe/cpu/build.sh`

现有测试覆盖 payload 布局边界、构建发布、领取绑定、fanin ready、engine 完成门控、vend/flag/DONE 顺序、构建中暂停、flush 后但 `BUILT` 前暂停、多 executor 竞争、busy token、重复 builder 和损坏 control/payload。

最新协议快照已经完成：

- GCC 15 普通构建与单次测试通过；同一二进制连续运行 100 次通过。
- GCC 15 ASan/UBSan 通过。
- GCC 15 TSan 通过；编译器明确提示 `atomic_thread_fence` 本身不受 TSan 建模，该提示已通过 `-Wno-error=tsan` 保留，不能把 TSan 当作 A5 cache 可见性证据。
- 操作账本证明 payload ordinary store → 单次完整 flush → `BUILT`，以及 Claim CAS → invalidate → shared load → private token store 的先后顺序。
- 16 个 executor 并发时恰好一个 Claim winner，逐 actor 证明每个 loser 都是零 DCCI、零 payload read。
- 两个同 shape cell 的 packed task id 阻止 token 交叉发布错误 `DONE`；completion 发生不可逆失败时发布 terminal fatal，已经成功的 vend/flag 不会重发。
- `BUILDING` pack 失败、非法 `BUILT` control、`CLAIMED` 后 payload 校验失败都发布独立 global fatal；fatal 后的新 Build/Claim 零 cell CAS、零 ordinary write、零 DCCI。
- waiting token 在 fatal 后不启动 engine；in-flight token 必须先等待真实 engine 完成，再进入 `FAULTED`，不发布 vend/flag/DONE。
- 负数、自依赖和未来 fanin 在 Build-N 阶段被拒绝；tensor/scalar/fanin count、payload bytes、word0 reserved bits 损坏均 fail-closed。

S1 的 CPU 结论只证明状态机、范围和受控交错，不能替代 S2 的 A5 DCCI/atomic 可见性验证。

## 2026-08-02：S2 CCEC 最小探针完成

### 已实现的探针结构

新增：

- `protocol_probe/ccec/probe_shared.h`
- `protocol_probe/ccec/kernel.cpp`
- `protocol_probe/ccec/host.cpp`
- `protocol_probe/ccec/cross_core_device_exports.map`

探针采用 1:1 mixed 构建、2 个 block，目标拓扑为 2 AIC + 2 AIV。覆盖 AIV→AIV、AIV→AIC、AIC→AIV、AIC→AIC 四种跨核方向，以及 1、2、8、68 条 cacheline 的 payload。每组同时比较：

- `return_dependency`：消费 CAS 返回值形成控制依赖后，直接做 payload acquire；
- `pre_dsb`：在相同依赖基础上额外加入前置 DSB，作为较重对照。

host 为每个 cell 填充独立 sentinel，并检查 active payload 完整、inactive payload 未被覆盖、control padding 未变化、四个参与者拓扑和最终 token/cell 状态。

### 当前已有的静态证据

自动化 CCEC IR 门槛当前同时检查 AIC/AIV：

- `return_dependency` 的唯一 Claim CAS 返回 SSA 必须参与比较，比较结果必须直接控制分支；顺序为 Claim CAS → DCCI → 唯一尾 DSB。
- `pre_dsb` 使用相同返回依赖，顺序为 Claim CAS → 第一条 DSB → DCCI → 尾 DSB。
- fatal 首错 CAS 的 expected 固定为 0；IR 检查明确将其与 expected 为动态 observed state 的 Claim CAS 区分，不能拿 fatal CAS 冒充 Claim acquire 证据。
- AIC/AIV object 和最终 mixed ELF 均无未定义 GLOBAL/runtime helper，也无 `__multi3`；最终 ELF 只有两个预期 device entry 和对应 metadata section。
- host 固定使用本用户 GCC 15 构建。

### A5 动态证据

在 CANN 9.1、device 0 上运行最新 mixed kernel：

```bash
ATOMIC_PROBE_DEVICE=0 \
  protocol_probe/build/ccec/cross_core_payload_probe_host \
  protocol_probe/build/ccec/cross_core_payload_probe_kernel.o 100
```

结果为 `100 runs × 32 cases/run = 3200 cases` 全部 PASS：

- 每轮参与者拓扑稳定为 2 AIC + 2 AIV，物理 core id 为 `0,18,54,72`；
- AIV→AIV、AIV→AIC、AIC→AIV、AIC→AIC 四种方向全部通过；
- 1、2、8、68 条 cacheline payload 全部通过；
- mid-pack、flush 后但 BUILT 前、Claim 前、Claim 后四类受控延迟均由 case 矩阵覆盖；
- `return_dependency` 与 `pre_dsb` 两种 acquire 都通过相同 sentinel、完整 payload、inactive tail、control padding、token/cell 终态 oracle；
- automatic scalar DCCI 与 kernel-end DCCI 均在编译时关闭，结果只依赖显式协议。

当前环境没有 `task-submit`，驱动安装也没有把 `npu-smi` 暴露给预检脚本；本轮依据已明确的 A5 环境和唯一 `/dev/davinci0` 直接运行，因此记录为未加设备锁。连续两次分别 20/20、100/100 PASS，未出现占用冲突。

### S2 结论边界

- 对本轮 fresh、task-indexed cell，消费返回型 Claim CAS 后直接执行精确范围 invalidate 已足够；额外前置 DSB 没有表现出正确性必要性，默认协议继续使用 minimal 路径。
- 该结论只覆盖首轮不复用 cell，不外推到未来 generation/ring reclaim。
- S2 证明跨核 payload 内存合同可行，但不代替完整 PA 的 task/fanin/vend/engine/FinalDrain 验证；下一步进入 S3a，再进入固定不同核的 S3b。

### S3 接入前的协议补强回归

为让 S3a/S3b 能从同一条 64-bit control 事后证明 owner 关系，packed
state 已同时保存 Build owner 与 Execute owner；`BUILT` 明确使用未绑定的
Execute owner，`CLAIMED/DONE` 同时保留两者。execution token 还增加了
fanin ready 前缀和独立 dispatch binding：已确认 ready 的依赖不重复读取，
executor 在取得 payload 后重建 tensor/scalar/context 参数，不能继承 builder
的自引用地址。

该布局变化不是只靠 CPU 推断。同步更新最小 CCEC 探针 host 的六参数终态
oracle 后，A5 device 0 重新运行 20 轮：`20/20` 轮、`640/640` case 全部
通过，core id 稳定为 `0,18,54,72`，四种 AIC/AIV 跨核方向、1/2/8/68
条 cacheline 以及 minimal/pre-DSB 组合均未回退。

## 2026-08-02：S3a 独立 PA fork 基线

### 为什么先建立这条基线

`cross_core/` 是从当前已经跑通的 shared-only same-core standalone 机械
复制出的独立源码闭包。它与 `same_core/` 对称，直接承载完整 PA；
只服务于 S0–S2 的小型协议探针则放在 `cross_core/protocol_probe/`。
后续只在 `cross_core/` 中替换非 Alloc 的执行包发布、领取、绑定和完成路径，
不在运行期或 include 路径上依赖 `same_core`。先运行原样
基线，是为了证明目录拆分没有改变 Materialize、Register、TensorMap、
SharedOutput、shared heap、Claim Tournament、PA task plan 和 host oracle；
这一步不属于构建执行分离的功能或性能结果。

### 已完成的基线门槛

- 使用 `CXX=/usr/bin/g++` 完整构建 shared perf-clock 及其全部 shared CPU
  隔离门槛，PollBatch、per-task insert completion、host task plan、普通
  region ring、稀疏 trace、compact trace、SharedOutput、writer intent、
  shared heap、Claim Tournament、Materialize 和 96-worker ordered Submit
  全部通过。
- 原样 fork 的 CPU B1 与 B256 均通过完整 host 语义检查，包括每 task 唯一
  winner、AIC/AIV 路由、1280 个 task 完成、1280 条 fanin 边、shared heap
  精确进度、writer history、descriptor 规范结果和 real-compute 输出。
- CPU 数值只用作正确性证据，不解释为 A5 性能。原样 B256 的 CPU
  perf-clock 约 14.65 s，主要受 96 个 pthread 与 CPU real-compute 影响，
  不进入后续 A5 性能对比。

### 工具链边界

用户目录的实验版 GCC 15.0.1 会生成 binutils 2.42 不识别的 `.base64`
伪指令；该失败发生在测试二进制汇编阶段，不是源码错误。same-core 既有指南
已经固定 CPU 回归使用 `/usr/bin/g++`，本轮沿用同一口径。CCEC 设备构建仍
使用 CANN 9.1 的 `ccec/ld.lld` 和用户目录 GCC 15 构建 host，二者不能混为
同一条工具链证据。

### 接入前审计发现的必须闭合项

1. 单条 64-bit execution control 必须同时保留 Build owner 与 Execute
   owner，否则终态只能看到后者，无法证明 S3a 同 owner 或 S3b 异 owner。
2. execution token 必须保存已确认 ready 的 fanin 前缀，不能在每次推进时
   从 edge 0 重复发射返回型 atomic load。
3. executor 需要独立的 dispatch binding，重新生成 50 个 args 指针、48 B
   LocalContext 与 4 B GlobalContext；不能复制 builder 的自引用指针。
4. token 忙时，本 worker 后续构建的 `BUILT` task 必须留在 task-indexed
   全局表，并由单调扫描游标重新发现；不能只在 Build 后尝试一次便遗失。
5. completion vend 必须在 Materialize 后冻结进 payload，执行完成时不得读取
   Execute owner 自己的 `worker.heap_next`。

这是当时进入动态验证前的硬门槛：上述项目闭合之前不运行 S3a A5 PA，
也不宣称 shared payload 已经接入完整调度器。后续闭合和实测证据记录如下。

## 2026-08-02：S3a 同 owner 构建/执行接线

### 当前实现边界

S3a 已把非 Alloc task 的 Build 结果序列化到 task-indexed
`SharedExecCell`，并由调度器扫描、领取、绑定和完成该执行包。这一阶段
刻意限定 `build_owner == execute_owner`：只允许构建该 task 的 worker
领取自己的 `BUILT` cell，用于先证明完整 PA payload、fanin、dispatch
重建与 completion 链路。它仍是 **build == execute** 的同 owner 实现，
并没有证明不同 Scalar 核可以接管执行。

S3b 尚未实现。后续 S3b 需要放开兼容 executor 对其他 build owner
的竞争，并用终态 `build_owner != execute_owner` 和跨核数据验证建立新证据；
当前结果不能写成 S3b 已完成。

### A5 Scalar 必须遵守的 cache 可见性合同

A5 Scalar 核之间没有可依赖的 cache coherence，因此 S3a 没有把
CPU 上的普通 load/store 可见性当作设备合同，而是固定了下面两条发布顺序：

1. 构建侧先用 ordinary store 写完有效 payload，再对完整有效 cacheline
   范围执行 DCCI `CACHELINE_OUT` clean-out 并以 DSB 收口，最后才用
   atomic CAS 把 cell 从 `BUILDING` 发布为 `BUILT`。`BUILT` 是 payload
   已对其他 Scalar 可见的唯一控制边界，不能提前。
2. 执行侧必须消费 Claim CAS 的返回值并由其建立分支依赖；只有
   Claim winner 才对已发布 payload 的精确范围做 invalidate，然后复制到
   本 executor 的私有 token。Claim loser 不 invalidate，也不读 payload。

`SharedExecCell.control`、global fatal 和 execution drain 控制字均独占
64 B atomic-only cacheline，payload 从下一条 cacheline 开始。这些原子控制行
不与 ordinary payload 共行，也不对它们执行会把 dirty snapshot 回写的
DCCI clean-out。

`ExecutionToken` 是 owner-local 普通 GM 状态，后续只由所属 Scalar 读写。
host H2D 写入的 `IDLE` 不能替代设备核本地初始化：同一物理 Scalar
仍可能保留前一 kernel 的 DCache 行。因此每轮 kernel 启动时，每个
worker 都由本核对自己的 token 执行 `ResetExecutionToken()`；跨核交接只经过
shared cell 的 atomic 和 DCCI 合同。

### CCEC 接线中闭合的 ABI 与 host 问题

- 修正 CCEC include order：先定义 `PA_DEVICE`、`PA_DEVICE_NOINLINE`、
  `PA_LOOP_NOUNROLL` 和 `PA_GM`，再纳入 PMU、winner workload 与 scheduler
  公共头。这保证公共闭包按 CCEC device 语义展开，不依赖头文件恰好
  以某个顺序被首次解析。
- 增加跨核执行扫描游标后，`CompeteFirstSplitRuntimeState` 的精确
  block-local ABI 由 1664 B 变为 **1728 B**。CCEC reserve size、runtime
  symbol、单 role `.bl.uninit` 和最终 AIC/AIV 双 role 布局都用同一
  1728 B 静态/产物检查锁定，不用多保留一条 cacheline 掩盖漂移。
- `SchedulerState` 尾部新增的 `exec_fatal + exec_drain + exec_cells +
  exec_tokens` 是一段连续 **19,691,648 B** execution sidecar。先前
  host 分段搬运只覆盖 shared TensorMap 和 Claim Tournament，漏掉了这个新尾段；
  现在每轮在 launch 前完整 H2D，kernel 同步完成后再完整 D2H，并用
  `offsetof(exec_fatal) + 19,691,648 == sizeof(SchedulerState)` 锁定无尾部缺口。
- host 从 execution payload 读取 `TensorDesc` 时，不再把 `uint64_t`
  word arena 直接 `reinterpret_cast` 成 `TensorDesc` 并解引用。现在先逐 word
  取得 volatile payload 的稳定快照，再用 `memcpy` 复制对象表示，避免
  strict-aliasing 未定义行为使优化后的 host oracle 前后读出不一致字段。
- PA adapter 现在在 Build 入口、Claim 后 dispatch 绑定入口和 kernel 发射前
  三处锁定精确 shape：QK=`4/2/0`、SF=`4/3/1`、PV=`4/2/1`、
  UP=`7/2/3`（tensor/scalar/fanin）。通用 payload 的“未超过容量”不能替代
  这份业务 ABI；畸形包必须在读取 fanin 或发射 kernel 前 fail-closed。
- host 终态 oracle 除了逐项检查计划内 Alloc/kernel cell，还会扫描
  `[task_count, kMaxTasks)`：control 必须保持 `EMPTY`，完整 payload 必须保持
  全零。这样能捕获向计划外 cell 的 ordinary write，而不把 host 检查放进
  Scalar 热路径。

### 当前验证证据

- S3a 接线后的 CPU B1 与 B256 此前已通过完整 host 正确性检查。
  CPU 结果只证明 task plan、payload、owner、fanin、dispatch、completion 与终态
  oracle 闭合，不代替 A5 cache/atomic 证据。
- A5 perf-clock B1 在修正上述 include/ABI/搬运与 cache 初始化问题后
  运行 10 轮，其中 **9 轮 PASS，1 轮 FAIL**。唯一失败轮在启动阶段已经
  进入调度器 global fatal，而 execution cell 全部仍为 `EMPTY`、所有 token
  仍为 `IDLE`；它没有进入 Build/Claim/payload 路径，因此不能归因于
  S3a 执行包协议。该结果也不能写成 B1 稳定 10/10 PASS，启动 fatal
  仍是独立的未闭合项。
- A5 perf-clock B256 的 **1280 个 task 全量 PASS**，完整 Submit 为
  **24986.974 us**。这一轮可以作为 S3a 在完整 PA 规模下未遗失
  `BUILT` task、未重复执行、fanin/completion/终态通过的正确性证据。
  24.987 ms 当前只是带有 S3a 机制的实测时间，没有同口径 A/B 能够将差值
  归因到单一机制，因此明确不作性能收益结论。
- A5 full-swimlane B256 也已闭合：1280 个 Submit、1024 个 kernel 全部
  通过，Submit 跨度为 **25086.894 us**，trace 无丢记录。该产物同时包含
  普通阶段、atomic 与 DCCI 事件，只用于检查 S3a 业务边界和发布动作是否
  齐全；它带有 737046 条合并事件和约 63 MiB 的加工文件，不能与
  perf-clock ELF 的绝对时间直接相减，也不作为 S3b 异核证据。

### S3a 阶段结论

S3a 已证明完整 PA task 可以被构造为共享执行包，再经由 atomic
发布、精确 DCCI acquire、owner-local token 和设备终态闭合。它当前仅完成
同 owner 的机制接线与正确性取证；B1 启动 fatal 仍需单独定位，性能优化与
S3b 真正的异 owner 领取都还没有开始。

## 2026-08-02：S3a EfDrain 异常与 S3b 发现协议修正

### 实测异常

S3a B256 full-swimlane 不是只有个别 14 us EfDrain：122880 个样本的
中位数为 **14.962 us**、p95 为 **20.549 us**、最大为
**217.371 us**，其中 92692 个样本不低于 14 us。对照同口径
same-core 最优泳道，中位数为 0.025 us、p95 为 0.791 us。

exclusive analyzer 进一步表明：EfDrain 占 Submit 聚合核时间的
89.683%，EfDrain 内 control 占 99.912%，kernel union 只占 0.088%。
因此不能把它解释成正常 kernel overlap 或观察噪声。

### 根因

S3a 的 `exec_scan_task` 让每个 worker 对每个 closed task 都读取一次
`exec_cells[task_id].control.state`。CCEC 的 int64 `Ops::Load()` 在 A5 上
是返回型恒等 `atomicMax(INT64_MIN)`；B256 正好形成
`96 workers × 1280 tasks = 122880` 次同地址分批竞争。Build owner 发布
同一 cell 的 CAS 也会与这些 reader 竞争，这解释了部分 WinnerBuild 和
EfDrain 的长尾。

新 execution control 的 atomic 与 payload DCCI 尚未单独进入 raw trace，
所以当前 merged 泳道把它们显示成没有子事件的 EfDrain control。现有父区间
足以定位总成本，但不足以做后续微观归因；S3b 取证需要补齐这些事件。

### 已修正的 S3b 方向

不再让 32/64 个同 role worker 读取每个 task control。每个 kernel task
根据 task id 和 engine 预路由 primary/secondary 两个 observer：

- Build owner 不是 primary 时，由 primary 执行；
- Build owner 等于 primary 时，由 secondary 执行；
- 每核在真实 Submit 已知 `(task_id, Kind)` 的调用点判断候选身份；非候选不登记、不读 shared control；
- 候选记录只进入 Scalar owner-local 紧凑位图/队列，不再用 `batch/offset` 重建历史 PA plan；
- 两个候选可以观察发布状态，但只有唯一实际 target 发射 Claim CAS。

该规则不减少现有 Build Claim 候选，确定性保证 Build/Execute 异核，并把
热发现路径每 task 的 control observer 上限从 96 降到 2。设备收口时的 terminal
validator 是另一条一次性证明路径，不应混入 EfDrain observer 计数。

### A5 反例与过程态撤销

首个 S3b 实现没有立即复用 Submit 当下的 `Kind`，而是给每个 worker
增加 `exec_scan_task/exec_scan_batch/exec_scan_batch_offset`，在 EfDrain 中先重建
batch plan。CPU 和 CCEC 编译均通过，但 A5 B1 连续运行已给出否定证据：

- 一轮语义全通过，Submit 却达 **849245.117 us**，不是可接受的普通波动；
- 下一轮在 task 2 发布 `invalid-built-control`，reporter 是 worker 63；
- task 2 为 SF，按同一映射独立计算的 AIV observer 是 34/36，worker 63 不是候选；
- 失败时 cell 仍是 `EMPTY`，且该错误发生在 scanner 读 cell 之前，直接指向本地 plan 游标/解析失配。

因此不对两个 16-bit 游标做局部修补，也不进入 B256。两候选映射保留，
全核事后重建 plan 的代码撤销；下一步先用 CPU 证明“Submit 成功闭合时登记
owner-local 候选位，EfDrain 只消费这份本地序列”，再重新构建 CCEC/A5。

## 2026-08-02：S3b Submit 当场登记的 CPU 门槛

### 实际落地的数据结构

每个 worker 不再保存 `task/batch/offset` 三元组，也不再重建 PA 计划。
当前 Submit 成功闭合时已经直接持有 `(task_id, Kind)`，因此只在本 worker
属于该 task 的 primary/secondary 时设置 owner-local candidate bit。

固定映射下，每核只可能命中两个 task-id 余数类：

- AIC worker `w` 命中 `task_id % 32 ∈ {w, w-1}`，最多 272 个潜在槽；
- AIV local worker `v` 命中 `task_id % 64 ∈ {v, v-2}`，最多 136 个潜在槽；
- 统一使用 9 个 `uint32_t`、共 288 bit，覆盖 AIC 最坏值并避免 16-bit
  设备访存假设。

位图和单调 slot 游标都属于 Scalar owner-local 状态，不是共享发布对象，
因此不需要 DCCI。新增字段复用了 `LocalStats` 的既有对齐空间：非 split
仍为 1152 B，split 仍为 1216 B，`CompeteFirstSplitRuntimeState` 仍为
1728 B；但 trace 在结构内的偏移发生变化，所以所有 CCEC TU 必须整套重编，
不能混用旧对象文件。

### 连续前缀合同

candidate scanner 只允许越过本 worker 已成功 Close 的连续 Submit 前缀。
生产 Close 现在强制：

```text
task_id == stats.result.submits
登记 candidate bit
stats.result.submits++
```

登记 helper 还会拒绝 `candidate_slot < exec_candidate_slot`。这两层门槛防止
乱序或重复 Close 把已经越过的旧 bit 重新置回，导致 scanner 永远不回头、
FinalDrain 永远无法排空。该合同依赖当前 PA task id 从 0 开始连续递增；
后续若改为稀疏 task id，必须显式改成独立连续前沿，不能继续借用 Submit 数。

### CPU 定向证据

严格告警构建下，execution scanner/drain 门槛已覆盖并通过：

- 96 worker × 全 `kMaxTasks` 的 potential slot 单调性和 task↔slot 往返；
- 真实连续 Submit Close 只为 kernel task 的 primary/secondary 登记，Alloc
  和其余 94 个 worker 不登记；
- 96 核面对同一个 `EMPTY` cell 时只有两个候选各发出一次测试 control load；
- `EMPTY` 保留队头，`BUILDING` 由非 target 越过而 target 等待，`BUILT`
  只由 target Claim；
- `CLAIMED/DONE` 的非 target 收敛、无效 control fail-closed、busy token
  恢复扫描、最后一个 task 与 96 核 execution drain closure；
- 跳号 Close、重复 Close、越过后重登和未越过时重复登记均被拒绝。

完整 `cross_core/run.sh build cpu` 也已通过，包括 PA adapter、ordered
Submit、TensorMap、SharedOutput、writer intent、heap、Claim Tournament
和 Materialize 等全部既有 CPU 门槛。CPU B1 的 host oracle 同时证明固定
异核 owner、四类 payload、fanin、vend、completion、token 复位和终态均通过。

### 当前仍未宣称的内容

本阶段只证明候选映射和 owner-local 发现状态机的 CPU 语义。global fatal
仍需在 Claim、kernel 发射与 completion 等不可逆边界收敛；新的 CCEC
整套编译和 A5 B1 多轮也尚未执行。因此此处不宣称 S3b 已完成，更不使用
旧的 849 ms 异常二进制进入 B256。

## 2026-08-02：S3b fatal 首错与不可逆边界收敛

### 为什么不能只在 Progress 入口检查一次

global scheduler fatal 与 execution fatal 位于不同的独占 atomic line。
如果 executor 只在 `ProgressCrossCoreExec()` 入口读取一次，另一 worker
仍可能在本次调用内部发布首错，随后本核继续 Claim、发射 kernel 或发布
completion。CPU 的顺序一致模型也能确定性构造这种交错，因此它不是只能
留给 A5 才讨论的弱序问题。

当前规则改为：

- 已经存在的 global fatal 只负责停止 Build/Exec，不再伪造
  `InvalidBuiltControl` execution reason；
- `BuildAndPublishExecPayload()` 返回 `FatalObserved`、`InvalidInput` 或
  `PublishConflict` 时保留 helper 已保存的精确 exec 首错，只把 terminal
  状态镜像到通用 fatal；只有本身不发布原因的 fresh-cell
  `CellUnavailable` 才补充 `ControlPublishConflict`；
- Progress 入口若发现任一 fatal，非 Idle owner-local token 转为
  `Faulted`；
- target 在 Claim 前、Claim/Bind 后、`WAITING_FANIN` 转 engine 后且真正
  发射 kernel 前、同步 kernel 返回后以及 completion 发布前重新检查停止
  条件；尚未发射的 token 直接转 `Faulted`，已经同步返回的 engine 不再
  发布正常 vend/flag/DONE。

这些复核不能宣称在两条独立 atomic line 之间建立“现实时间上的绝对同时
停止”；其合同是：本核一旦在对应不可逆边界观到 terminal fatal，就不再
执行下一项副作用。当前 engine helper 保持同步边界，本轮不扩展异步查询协议。

### CPU 确定性注入门槛

scanner 测试在生产 Ops 边界增加了只用于测试的注入点，并完成 20 轮重复：

1. 预置 scheduler global fatal 后调用 Build，exec fatal 保持零、cell 保持
   `EMPTY`，没有 control CAS 和 payload 统计，证明不伪造 execution 首错；
2. 真实 Claim/Bind 得到 `WAITING_FANIN` token，或用完整 helper 链得到
   `COMPLETING` token；下一次 Progress 观察 global fatal 后均转为
   `Faulted`，cell 保持 `CLAIMED`，vend/flag 不发布；
3. 在 target 首次读取 `BUILT` control 后注入 fatal，Claim CAS 为零；
4. 在最后一条 fanin ready load 返回后注入 fatal，Claim 恰好一次但 kernel
   调用为零；
5. 在同步 kernel 返回时注入 fatal，kernel 恰好执行一次，但 vend、flag 和
   cell `DONE` 均不发布；
6. 所有只注入 global fatal 的场景都保持 `exec_fatal == 0`。

完整 CPU 构建及 ordered Submit 等既有门槛再次全部通过。该阶段仍只提供
CPU 状态机证据；CCEC 编译和 A5 B1 重复稳定性属于下一门槛。

## 2026-08-02：S3b B1 异核执行功能闭合

### 为什么不能再用 2 秒判定有序插入失败

S3b 首轮 A5 取证曾出现 `first_not_ready=1/2`、kernel 未全部执行和
global fatal，但 execution fatal 始终为零。继续核对后确认：

- `first_not_ready` 是首个未发布 completion flag 的 task，不是 TensorMap
  插入失败原因；
- 旧 `[TENSORMAP] completed_tasks` 误打印了 host 计划 task 数，不是设备实际
  `deps_prepared` 连续完成前缀；
- TaskCell 的 `flag/vend/deps_prepared` 仍是纯 atomic cacheline，写方 CAS、
  读方 atomic Max，当前没有普通 dirty store 或 DCCI 覆盖这条 line 的证据；
- 当前最有力的首错候选是 `WaitForSharedTaskInsertTurn()` 复用的通用
  2 秒 watchdog：失败轮 execution fatal 为零，而未完成前缀正好从首个
  未就绪 task 开始。但当前没有保存单次 predecessor wait 的精确起止时间，
  因此不把这一归因写成已完全证明的硬件定理。

修正后不删除有界终止，而是把两种口径分开：

```text
启动屏障/旧隔离 helper: 2 s
cross-core TensorMap 有序插入等待: 60 s
```

60 秒只是 standalone 功能阶段的有界容忍值，不是对正常性能的许诺。
同时 host 改为按 `deps_prepared[N] == N` 计算真实连续完成前缀；失败时
最多打印 8 个未完整回放的 worker，用于区分回放截断与 split ticket 本身的错误。

### A5 B1 实测证据

CCEC perf-clock 整套重编后，B1 共运行 4 轮：

| 轮次 | Submit | 结果 |
| ---- | -----: | ---- |
| 1 | 885.188 ms | PASS |
| 2 | 2937.959 ms | PASS |
| 3 | 278.501 us | PASS |
| 4 | 228.422 us | PASS |

第 2 轮的整个 Submit 超过 2 秒仍最终完整通过，说明这个功能阶段不能用
一枚与观测开销、调度波动无关的 2 秒固定值代替协议正确性。这不能单独证明
该轮的某一次 predecessor wait 也超过 2 秒；本次改动的精确结论是：
先防止过早超时阻断 S3b 功能取证，后续再用独立时间证据定位巨幅波动。

4 轮都精确闭合：

- 5 个 task 只有 5 个 Build owner；
- QK/SF/PV/UP 各执行一次，QK/PV 只在 AIC，SF/UP 只在 AIV；
- host 独立复算的每 task `execute_owner` 都与 `build_owner` 不同；
- portable payload 的 descriptor/scalar/fanin/vend/route 逐项匹配；
- completion flag、vend、cell `DONE`、owner-local token reset 与 96 核 execution drain
  全部通过；
- global fatal 和 execution fatal 全部保持未发布。

这一阶段宣布 **S3b B1 固定两候选异核交接功能已闭合**。耗时从
228 us 到 2.938 s 的巨大波动仍需后续单独定位；当前不宣称性能收益。

## 2026-08-02：S3b B256 完整规模功能闭合

在与 B1 相同的 CCEC perf-clock ELF 上执行 B256，一轮完整通过：

| 指标 | 实测值 |
| ---- | -----: |
| Submit | 27.243 ms |
| 计划 task | 1280 |
| Build winner | 1280 |
| 异核 kernel | 1024 |
| QK / SF / PV / UP | 256 / 256 / 256 / 256 |
| fanin edge | 1280 |
| shared output | 2048 |
| TensorMap 插入完成前缀 | 1280 |

设备和 host 终态证据同时闭合：

- 96 个 worker 完整回放 1280 个 Submit，Claim 次数与既定拓扑精确相等；
- 每个 kernel cell 都保留互异的 `build_owner/execute_owner` 并到达 `DONE`；
- descriptor、scalar、fanin、completion vend 和核型 route 逐 task 匹配 host 独立计划；
- QK/PV 仅在 AIC，SF/UP 仅在 AIV，四类结果 tile 全部通过数值检查；
- 1280 个 completion flag/vend、全部 execution token reset、96 核 drain closure、
  global fatal 和 execution fatal 全部符合终态合同。

因此当前可以宣布 **S3b 固定两候选的非 Build 核执行功能已在 B1 和
B256 闭合**。本轮 27.243 ms 只是功能构建的实测值；还没有生成同代码的
B256 泳道，也没有对波动、非必要 atomic 或固定映射的性能作为保留判据。

## 2026-08-02：S3b B256 full-swimlane 与终态观测口径闭合

### 为什么 raw token 不能继续作为 A5 host 断言

首次运行 S3b B256 full-swimlane 时，1280 个 task、1024 个 kernel、96 核
execution drain、设备发布的 `final_occupied`、cell 终态和 fatal 均已通过，
唯一失败项是 host 直接 D2H 后检查 `exec_tokens[]` 本体未全部呈现为
`IDLE`。当时没有逐字段保存首个非 IDLE token，因此这里只能确定 raw
快照没有通过，不能进一步声称 host 具体看到了哪一个中间 phase。

该 raw 快照不具备 A5 语义权威性：

1. `ExecutionToken` 是每个 Scalar owner-local 的普通 GM 状态；CCEC 构建
   明确关闭 automatic scalar DCCI 和 kernel-end DCCI；
2. A5 Scalar 之间没有 cache coherence，kernel 返回后的 D2H 可能读到
   初始化值、自然回写的中间值或最终值；一次碰巧读到 `IDLE` 也不能反向
   证明这个观察通道始终可靠；
3. 每个 worker 只有在本核同时通过 `CrossCoreExecWorkerDrained()` 与
   `CrossCoreExecTokenFullyReset()` 后才能加入 `exec_drain`；最后到达者还会
   逐 task 验证终态并发布 drain release；
4. release 后每核再次检查 scanner、候选位图和 token 全字段，并把
   `final_occupied` 通过 bypass result 发布给 host。

因此修正只发生在 host oracle：`Validate()` 的两个调用点显式选择
`RawExecTokenSnapshotAuthority`。CPU 在线程 join 后采用
`Authoritative`，raw token 仍是严格断言；CCEC 采用 `DiagnosticOnly`，继续
打印 `RESET/NON_FINAL` 帮助诊断，但不再修改 `semantic_status`。设备侧 cell、
token 自检、execution drain、bypass result 和 fatal 门槛一个都没有删除，
也没有为了迎合 D2H 观察向设备热路径新增 DCCI、DSB 或 ordinary store。

### 回归与 B256 泳道证据

- 完整 CPU 公共构建和隔离测试通过；CPU B1 实跑继续显示
  `coherent executor token snapshot is fully reset PASS`，证明一致内存后端的
  严格门槛没有被放松；
- CCEC AIC/AIV、split caller/runtime/finish、mixed ELF、host 与 manifest
  整套重编通过；
- A5 B256 full-swimlane 的 Submit 为 **27127.645 us**，semantic 与
  postprocess 均为 PASS；本轮 raw D2H 恰好呈现 `RESET`，但仍只按诊断处理；
- 1280 个 Build winner、1024 个异核 kernel、1280 条 fanin edge 精确闭合，
  QK/SF/PV/UP 各 256；1016 个 kernel 落在 EfDrain、8 个落在 FinalDrain，
  orphan 为 0；
- 122880 个 Submit actor 精确拆成 1280 winner 与 121600 loser；raw trace
  无 drop，exclusive analyzer 的 Submit、EfDrain、orchestration、final drain、
  worker completion 和 winner/loser 分区全部 `exact=true`。

本轮产物不提交到 Git：

```text
tests/atomic_probe/pa_scheduler/outputs/
  pa_scheduler_cross_core_shared_swimlane_20260802_112542_3784828/ccec/
    l2_swimlane_records.json              23,784,330 B
    merged_swimlane.json                  65,524,051 B
    swimlane_exclusive_analysis.json         327,160 B
```

其中 `merged_swimlane.json` 可直接载入 Perfetto。该轮证明 S3b 的完整业务
边界和异核 kernel 均能被泳道导出；27.128 ms 仍是功能版本的观测构建耗时，
不宣称相对 same-core 有性能收益，也不把它与 perf-clock ELF 直接相减。

## 2026-08-02：S4 K2 动态 Execute election 实现与 CPU/A5 门槛

### 本节边界

本节记录 S4 K2 首版实现和 CPU/A5 取证。该版本只验证受控
候选中的 exactly-once election，不宣称已建成通用动态任务池。

### 首版 K2 候选集

- 实现复用 S3b 已有的 `primary/secondary` 两个 observer 和每核
  owner-local 候选位图，不新增全核扫描或中央队列。
- 观察到 control 中的 `build_owner` 后，将 Build owner 从这两核
  中排除：Build owner 不在 K2 时有 2 个 eligible executor，
  Build owner 恰好在 K2 时有 1 个 eligible executor。
- eligible 且 token 空闲的 observer 均有资格对同一 `BUILT` cell
  发射 `BUILT -> CLAIMED(self)` CAS。唯一 CAS winner 进入 payload
  invalidate/bind；`ExecClaimResult::Lost` 或竞争窗口中的
  `NotBuilt` 只会保留候选位并重新观察 control。当后续观察到
  另一合法候选发布的 `CLAIMED/DONE` 后，loser 才退役本 task；
  这些都不是 fatal，loser 也不触碰 payload DCCI。

### control、候选位与 FinalDrain 合同

| 观察状态 | S4 K2 实际动作 |
| -------- | -------------- |
| `EMPTY` | 生产未闭合时保留候选位，不发射 CAS，不读 payload |
| `BUILDING` | 若本 observer 就是 `build_owner`，立即退役其本地候选位；其他 eligible observer 保留候选位等待 `BUILT` |
| `BUILT` + token 空闲 | eligible observer 发射 Claim CAS；winner 绑定，`Lost/NotBuilt` 保留候选位并重新观察 |
| token 忙 | 只推进已绑定 token，不观察、不 CAS、不读取新 cell；新 task 的候选位保留 |
| `CLAIMED` / `DONE` | 校验 `execute_owner` 是排除 Build owner 后的 K2 合法候选；未中选 observer 退役本地位，不发布 fatal |

`FinalDrain` 仍以“所有 builder 停产、所有 kernel cell 到达
`DONE`、每核 token 为 `IDLE`、engine 无 in-flight”为权威收口。
生产闭合后残留的 `EMPTY/BUILDING` 是未发布缺口；`BUILT`
必须继续由空闲 eligible executor 竞争；`CLAIMED` 等待全局
cell 到达 `DONE`，不得因本地候选位已退役就提前释放 final drain。
终态校验不再假设唯一预定 executor；K2 中任一非 Build owner 的
合法候选获胜都是可接受终态，但 exactly-once、payload、completion
与 FinalDrain 断言仍全部保留。

### 与 TensorMap 和 K3 的边界

TensorMap/SharedOutput metadata 的严格有序插入仍由 Build owner 在发布
`BUILT` 之前完成；S4 只改变 `BUILT` 之后的 Execute owner 选择，
不改 `deps_prepared` 链、fanin 内容或 TensorMap 插入顺序。

K3 候选是后续对照：预路由 3 个兼容 observer，排除 Build owner
后再确定 2 个 eligible 竞争者，因而可以保证每个 task 始终有
2 个 Execute 竞争者。但它使 control observer 数由 2 增加到 3，
观察量增加 50%，所以不与 K2 首次实现混做。K3 尚未实现；
当前对 K2/K3 都不宣称性能收益。

### CPU 动态证据

- 完整 CPU build 与全部公共/隔离门槛均为 PASS，包括 K2
  eligibility、Build owner 跳过、primary/secondary 分别获胜、确定性
  CAS `Lost`、忙 token 不接触新 cell、合法动态终态与
  FinalDrain 收口。
- CPU B1 的 semantic/postprocess 均为 PASS：5 个 task、4 个 kernel
  全部闭合。
- CPU B256 的 semantic/postprocess 均为 PASS：1280 个 task、1024 个
  kernel 全部闭合；动态合法 owner、payload、completion terminal、
  token 与 drain 等断言全部通过。

CPU 结果证明协议状态机和 PA 功能规模已闭合，不能代替 A5
的 atomic/DCCI 动态证据，CPU 耗时也不用于推导 A5 性能。

### A5 动态证据

- CCEC AIC/AIV 通用协议实例化、入口编译和 artifact manifest 全部通过。
- B1 完整泳道 Submit 为 `280.683 us`，5 个 task、4 个 kernel、
  payload、terminal snapshot、execution drain 和实际计算结果全部 PASS，
  且无 trace drop。
- B256 普通运行 Submit 为 `27.420 ms`；随后独立完整泳道 Submit
  为 `27.476 ms`，1280 个 task、1024 个 kernel、1280 条 fanin edge、
  2048 个 shared output 全部闭合，trace drop 为 0。
- B256 泳道中 956 个 task 的 Build owner 不在 K2 内，实际由
  primary/secondary 分别赢得 319/637 次；Build owner 是 primary 的
  27 个 task 均由 secondary 执行，Build owner 是 secondary 的 41 个
  task 均由 primary 执行，非法 owner 为 0。这证明上板实际走了
  动态竞争，不是 host 只对唯一预定 owner 放行。

完整 B256 泳道在：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260802_125404_3864346/ccec/merged_swimlane.json`

干净 B1 泳道在：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260802_123052_3843470/ccec/merged_swimlane.json`

### 长尾的定位边界

B1 连续运行仍能观察到数十毫秒至数十秒的偶发长尾，但对照
已经排除“S4 CAS 造成状态环”：

- 父提交 `271ce596` 的 S3b fixed-owner perf-clock 同机 B1 五轮为
  `308.638 us / 281.619 us / 36.922 s / 192.829 ms / 298.487 us`，
  五轮语义均 PASS；因此几十秒长尾不是 K2 新增现象。
- S4 的 `805.959 ms` 长尾泳道中，第一个异常是 AIC core26 在
  task 3 `Materialize` 入口到第一个 heap atomic 之前停了约
  `805.817 ms`；该 heap atomic 本身只有 `0.281 us`。
- task 4 的 `register.wait_predecessor_tensormap_insert` 随后等待
  `805.799 ms`，是 TensorMap 严格顺序链传播上游停顿的结果。
  S4 election 只能在 task 3 `winner_build` 结束并发布 `BUILT` 后开始，
  不可能导致该停顿。
- 已到尾部的核在 FinalDrain 内无退避轮询，会将一次落后核
  放大为大量 atomic 流量；这是后续独立的等待公平性/退避问题，
  不在本 S4 功能提交中顺手改写。

长尾取证泳道在：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260802_124840_3856072/ccec/merged_swimlane.json`

与 S3b full-swimlane 的 `27.128 ms` 单样本相比，本轮 S4 为
`27.476 ms`，约 `+1.28%`。两者都是观测 ELF 的单轮样本，且上述长尾
已证实存在，因此只记录数值，不据此宣称稳定性能回退或收益。

## 2026-08-02：S5a 跨角色 Build 的 CPU 门槛

### 为什么先做角色反转

直接把所有 kernel 的 Build 候选扩成 96 核，会同时引入 portable Build
可移植性和约 60.5% 的额外物理 Claim CAS，若失败或回退将无法归因。
因此 S5a 先保持三档候选人口不变，只交换 kernel Build 角色：

| task | Build 候选 | Execute 候选 |
| ---- | ---------- | ------------ |
| Alloc | 96 Scalar / G8 | 无 kernel |
| QK、PV | 64 AIV / G8 | AIC K2 |
| SF、UP | 32 AIC / G6 | AIV K2 |

B256 的 local/root/总物理 Claim CAS 仍为
`73,728 / 9,216 / 82,944`。每个 kernel 的 Build owner 必然位于目标
engine 对侧，因此不会占用 K2 的任何一席；primary、secondary 都能参与
`BUILT -> CLAIMED(self)`，S4 的动态 exactly-once 状态机不需要改成唯一
指定 executor。

### 实现与独立校验

- `Claim()` 只在 shared 分支交换 QK/PV、SF/UP 的候选角色和
  Tournament 分组；private 分支保持原有 engine 同角色 cursor 语义。
- 当时新增 `PaCrossRoleBuildOwnerEligible()` 作为 S5a 阶段性拓扑策略；
  S5b 扩到全 96 Scalar 后，该 helper 已收敛为 `PaBuildOwnerEligible()`。portable
  `PaExecuteOwnerEligible()` 仍只要求 builder 有效、executor 匹配 engine、
  位于 K2 且不同于 builder，不把阶段性 Build 策略写死进通用 payload
  协议。
- Build 发布入口、执行扫描、active token 和 terminal cell 都显式校验
  对侧 Build 角色；payload、DCCI 发布/失效顺序、TensorMap
  `deps_prepared` 严格插入链没有改动。
- host 独立实现同一数学合同，不调用设备 helper：分别核对 AIC/AIV Claim
  尝试数、对侧 Build owner、目标 engine K2 Execute owner，以及 fanin
  payload 发布角色与 ready-load 执行角色，避免总量对称时掩盖角色接反。
- 稀疏泳道的 Claim attempted/winner 合法性同步采用新的 Build 角色；
  execution kernel 轨道仍按 Execute owner 的 AIC/AIV 角色呈现。

### CPU 证据

- 完整 `CXX=/usr/bin/g++ ./run.sh build cpu` 通过全部公共和隔离门槛，
  包括 Claim Tournament、稀疏/紧凑 trace、PA adapter、K2 scanner、
  96-worker ordered Submit 和 FinalDrain。
- Claim 门槛精确覆盖 Alloc 96/G8、QK/PV 64 AIV/G8、SF/UP 32 AIC/G6，
  保留 exact-one、replay 全输、loser 零 TensorMap 访问和严格插入链。
- CPU B1 real-compute 为 5 task/4 kernel，B256 real-compute 为
  1280 task/1024 kernel；两者 semantic/postprocess 全部 PASS。
- B1/B256 均逐 task 验证 Build owner 在目标 engine 对侧，Execute owner
  匹配 engine、属于 host 独立 K2 且不同于 Build owner；payload、vend、
  completion、token、fanin、shared heap、TensorMap 和计算结果全部闭合。

CPU 耗时只受 pthread 与 CPU real-compute 影响，不解释为 A5 性能，
也不能用 CPU cache coherence 替代 A5 Scalar 无 coherence 下的
DCCI/atomic 动态证据。

### CCEC 与 A5 动态证据

- CCEC 的 AIC/AIV 通用协议实例化、两类正式入口、compete-first
  caller/runtime/finish 角色符号、最终混合 ELF 和 artifact manifest
  全部通过；QK/PV 与 SF/UP 的 real-compute helper 仍只落在目标 engine，
  Build 角色反转没有改写 kernel 路由。
- A5 B1 real-compute 的 5 task/4 kernel、跨角色 Build、payload、fanin、
  terminal、drain、TensorMap、shared heap 和数值结果全部 PASS。本轮
  Submit 为 `1763.512 ms`，命中了 S4 已证明存在的偶发长尾；没有新增
  协议错误，因此按既定边界不阻断功能推进。
- A5 B256 普通运行 Submit 为 `27.143 ms`；完整泳道运行 Submit 为
  `27.301 ms`。两轮均完成 1280 task/1024 kernel，execution/semantic/
  postprocess 全部 PASS，完整泳道 trace drop 为 0。
- 完整泳道的 `winner_build` 轨道逐 task 直接闭合 Build 角色：QK/PV
  各 256 个均在 AIV Scalar，SF/UP 各 256 个均在 AIC Scalar。实际
  Execute election 中 primary/secondary 分别胜出 `363/661` 次，非法
  owner 为 0，证明两个 K2 候选均在 A5 上实际参与，而不是 host 只接受
  一个预定 executor。
- S5a full-swimlane `27.301 ms` 与 S4 单样本 `27.476 ms` 同量级，未出现
  倍数级结构性回退。两者均为观测 ELF 单轮样本，且已知长尾存在，不能
  据此宣称约 0.6% 的差异是稳定收益。

完整 B256 泳道在：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260802_133108_3915277/ccec/merged_swimlane.json`

至此 S5a 的 CPU/CCEC/A5 功能门槛闭合。在此基础上，下节 S5b
再将 kernel Build 候选扩大为 96/G8，并单独评估额外 Claim CAS
与到达式负载均衡的权衡。

## 2026-08-02：S5b 全 96 Scalar Build 的 CPU 门槛

### 本阶段改变了什么

- Alloc/QK/SF/PV/UP 的 Build Claim 统一扩为 96 个 Scalar、G8 两级
  Tournament；每个 local 节点精确对应 12 个候选核。
- kernel 类型只决定 function 和 Execute engine，不再决定 Build owner
  是 AIC 还是 AIV。`PaBuildOwnerEligible()` 只接受 `[0,96)` 的有效
  Scalar；Execute 仍需匹配 engine、位于 K2 且与 builder 不同核。
- TensorMap 严格插入顺序仍由 `deps_prepared` commit chain 保证；
  Claim 扩容没有把 metadata 插入或 kernel 执行塞入该串行链。
- private TensorMap 的 engine 同角色候选、cursor 和 `atomicMax` 路径保持不变。
- CCEC 两类 role 现在都可能赢得五类 task，因此预期每类入口
  的 split-finish relocation 从 3 条变为 5 条；该形状留待下一阶段
  CCEC 编译硬校验，本 CPU 提交不提前宣称通过。

### 计数合同

B256 共 1280 个 task，所以：

```text
local CAS = 1280 * 96 = 122880
root CAS  = 1280 * 8  = 10240
总物理 CAS = 133120
```

相对 S5a 的 `82,944` 次总物理 Claim CAS，S5b 增加约 60.5%。
这是到达式 Build 负载均衡的明确代价，A5 评估时必须单独呈现，
不能只看端到端单个数字。

### CPU 证据

- 完整 CPU build 及 Claim Tournament、portable adapter、K2 scanner、
  ordered Submit、稀疏/紧凑 trace 和 FinalDrain 门槛全部 PASS。
- 确定性用例覆盖 builder 位于 K2 primary、secondary、K2 外同角色
  与跨角色四种形状，并覆盖 BUILDING/BUILT、busy token、fatal、
  terminal 与 FinalDrain。
- CPU B1/B256 real-compute 的 semantic/postprocess 全部 PASS；B256 逻辑
  Claim 精确为 `122,880`。隔离 Tournament 门槛精确闭合
  `122,880 / 10,240 / 133,120`；B1 atomic 分析闭合 `480 / 40 / 520`。
- CPU B256 全 atomic trace 因 FinalDrain 轮询记录耗尽通用 trace 容量，
  无法进入新增的物理 CAS 后处理。这是观察容量限制，不是协议卡死；
  普通 B256 和隔离 Tournament 门槛已分别提供逻辑与物理计数证据。

CPU 耗时不用于推导 A5 性能，CPU cache coherence 也不能代替 A5
Scalar 无 coherence 下的 DCCI/atomic 动态证据。

### CCEC 与 A5 动态证据

- CCEC 通用协议实例、AIC/AIV 入口、两类 split runtime/finish、最终
  1:2 混合 ELF 与 artifact manifest 全部通过。两类 role caller 均由
  构建脚本硬校验 5 条 winner-finish relocation，与五类 task 都可能由
  AIC/AIV Build 的实例化形状一致。
- A5 B1 首轮业务、TensorMap、payload 和计算结果都通过，但 Submit
  命中 `8570.183 ms` 长尾，FinalDrain 轮询将记录容量耗尽并 drop 1 条，
  因此不作为正式证据。第二轮 Submit 为 `164.625 us`，5 task/4 kernel、
  DCCI/atomic、执行终态和真实计算全部 PASS，trace drop 为 0。
- A5 B256 普通运行 Submit 为 `27.347 ms`；完整泳道 Submit 为
  `28.250 ms`。两轮均闭合 1280 task/1024 kernel、122,880 次逻辑 Claim、
  1280 条 fanin edge、2048 个 shared output、严格 TensorMap 插入链与真实计算；
  fatal 为 0，完整泳道 trace drop 为 0。
- 完整泳道中 Claim local CAS 按角色精确为 AIC `40,960`、AIV
  `81,920`，合计 `122,880`；root CAS 为 `10,240`，总物理 CAS 为
  `133,120`。Alloc/QK/SF/PV/UP 的 Build winner 分布分别为 AIC/AIV
  `47/209、34/222、39/217、45/211、39/217`，合计 `204/1076`；这证明
  五类 task 的两种 Scalar 角色都在 A5 上实际完成过 Build。Alloc
  使用 `alloc_complete` 与 `materialize` 位置交叉核对，四类 kernel 使用
  `winner_build` 与 `materialize` 交叉核对，位置不一致数均为 0。
- 1024 个 kernel 由 K2 primary/secondary 分别执行 `375/649` 次，非法
  Execute owner 为 0。host 终态同时逐 task 校验 Build/Execute 不同核，
  泳道交叉核对的同核数也为 0，因此 Build 候选扩容没有破坏 S4
  exactly-once 执行合同。

S5b 普通运行与 S5a `27.143 ms` 单样本相比约 `+0.75%`；完整泳道与
S5a `27.301 ms` 单样本相比约 `+3.48%`。已知长尾与单轮波动使这些数字
不足以宣称稳定回退，但当前也没有证据说明全 96 Scalar 的到达式负载
均衡已覆盖 60.5% 额外物理 Claim CAS。S5b 作为“TensorMap 严格串行插入 +
96 Scalar 自由 Build 竞争 + K2 异核 Execute”的功能基线保留；S6 在此架构内
直接优化，不依赖 `try_wait` 或 kernel/调度 overlap 才判定性能去留。

完整 B256 泳道在：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260802_142824_3971096/ccec/merged_swimlane.json`

## 2026-08-02：S6.1 跳过无本核执行工作的 EfDrain 完整入口

### 问题不是候选扫描，而是无效入口的集中式 control 读取

S5b 的 B256 `perf-clock` 重新编译后做了 10 个独立进程样本：

```text
min/median/mean/max =
27.190804 / 27.496232 / 27.489246 / 27.747057 ms
```

对应 full-swimlane 的 Submit 为 `28.250448 ms`。排他分析中
`EfDrainControl` 聚合核时为 `2301.070442 ms`，占
`WorkerCompletion` 的 `81.992%`；而真正的 candidate potential slot
全局只有 `5120` 个，不能用“扫描 1280 个 cell”解释这项成本。源码审计
确认每个 actor 的每次 Submit 都无条件进入 `ProgressCrossCoreExec()`：
B256 共 `96 * 1280 = 122880` 次。即使本核 token 为 Idle、下一个 K2
候选还没有进入已 Close 前缀，入口仍先对 global fatal 和 exec fatal
执行两次集中式返回型原子读取，再复核 owner-local token 和候选游标。

### 门控合同

新增 `CrossCoreExecHasLocalProgressWork()`，只在以下任一条件成立时才让
opportunistic EfDrain 进入完整执行推进：

1. 本 executor 的 owner-local token 不是 Idle；
2. 本核紧凑候选游标指向的 task 已小于 `stats.result.submits`，即它已经
   落入本核成功 Close 的 Submit 前缀。

若两项都不成立，单调 cursor/registration 合同证明本核没有已经 Close
的 candidate 可读，也没有在途 token 要推进。当前调用点本来就忽略
`ProgressCrossCoreExec()` 的返回值；在 Idle/no-candidate 形状下，完整
入口的 terminal 读取不会阻止随后 Claim，也不会改变 token/cursor。
execution fatal 的首错仍由实际发布者镜像到 global fatal，所有 winner
不可逆边界和 FinalDrain 继续执行原有 authoritative 检查。因此门控删除
的是冗余观察，不是错误授权。非法参数一律返回“需要完整推进”，不能被
快路径吞掉；FinalDrain 也始终绕过该门控。

该判断只读取本 executor Scalar 独占的 token 与 LocalStats 游标，不把普通
load 当成跨核发布依据，也不依赖 A5 Scalar cache coherence。Claim
Tournament、TensorMap `deps_prepared` 严格插入链、96 Scalar Build 资格、
K2 Execute owner 和 payload DCCI 合同均未修改。

### 正确性与动态结果

- 新增 CPU 定向测试覆盖：前缀未 Close 时跳过、Alloc 非候选槽 Close 后
  必须进入 scanner 并推进、活跃 token 不得跳过、非法入口不得跳过。
- 完整 CPU build、执行协议/ordered Submit 门槛及 B1/B256 real-compute
  全部 PASS。
- CCEC AIC/AIV 通用实例、两类入口、split caller/runtime/finish、最终
  1:2 ELF、无 relocation 和 artifact manifest 全部 PASS。
- A5 B256 `perf-clock` 10 个独立进程样本全部 semantic/execution/
  postprocess PASS：

```text
min/median/mean/max =
8.761150 / 9.077919 / 9.102288 / 9.680661 ms
```

相对同一提交改动前的 10 轮中位数改善 `66.985%`。本轮波动范围约为
中位数的 `10.129%`，后续候选仍必须使用多轮 `perf-clock`，不能用单个
最好值作为去留依据。

新的 B256 full-swimlane Submit 为 `8.765063 ms`，全部业务、TensorMap、
payload、终态、真实计算和 trace closure PASS，drop 为 0。local/root
Claim CAS 仍精确为 `122880 / 10240`，总物理 CAS 仍是 `133120`；1024 个
kernel、1280 条 fanin edge、2048 个 shared output 和 1280 个插入完成字
均保持原合同。排他对比为：

| 指标 | S5b 原泳道 | S6.1 门控 | 变化 |
| ---- | ---------: | --------: | ---: |
| Submit 墙钟 | 28.250448 ms | 8.765063 ms | -68.974% |
| SubmitUnion 聚合核时 | 2572.730800 ms | 743.456767 ms | -71.102% |
| EfDrainControl 聚合核时 | 2301.070442 ms | 517.839958 ms | -77.500% |
| Claim 聚合核时 | 48.465912 ms | 44.030817 ms | -9.151% |
| FinalDrainResidual 聚合核时 | 212.765162 ms | 174.918873 ms | -17.788% |

新泳道在：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260802_153100_4051026/ccec/merged_swimlane.json`

用户随后把最终量化目标明确为：同一 B256 `perf-clock` 口径压到
`1.0 ms`。S6.1 只是把基线推进到约 `9.08 ms`；后续仍需重新拆解
剩余 EfDrain、payload handoff、DCCI/DSB、Claim 与 FinalDrain，不能把
本阶段的大幅收益误写成目标已经完成。

## 2026-08-02：S6.2 删除 scanner 入口重复 fatal 原子读取

### 删除范围与正确性边界

S6.1 之后，完整 `ProgressCrossCoreExec()` 仍在任何 scanner 工作之前
无条件读取 `state->fatal.value` 和 `exec_fatal.state`。这两个值在正常
运行中始终为零，却由所有 executor 汇聚读取；A5 的 `int32_t/int64_t`
共享读取分别是返回型恒等 RMW，不是普通 cache load。

本阶段删除的只有公共入口这两次读取，并保留以下不可逆边界：

- active token 在 kernel 发射前和 completion 发布前检查 global fatal，
  token helper 在推进状态前检查 exec fatal；
- Idle scanner 看到 `BUILT` 后，在 Claim CAS 前重新检查 global/exec
  fatal；
- Build publish、payload acquire、kernel、completion 和 FinalDrain 的
  既有错误检查均不变。

入口到这些边界之间只能读取 cell control，或推进 owner-local candidate
cursor。`EMPTY/BUILDING`、非候选槽和合法 CAS loser 都不会在这一段发布
跨核业务副作用。因此入口检查不是授权边界；删除后错误仍在第一个不可逆
动作前 fail-closed。为了让已经占有 token、但 fanin 尚未 ready 的 worker
及时响应 scheduler fatal，global fatal 检查被放到 active-token 专属入口，
而不是恢复到所有 Idle scanner 都经过的公共入口。

### 验证与性能

- 执行 scanner 的定向 CPU 用例全部 PASS，包括已有 global fatal 在
  `WAITING_FANIN/COMPLETING` 的收敛，以及 Claim/kernel/completion 三个
  精确注入窗口；
- 完整 CPU perf-clock build、自测和 B1/B256 real-compute 业务闭合；
- CCEC AIC/AIV 通用实例、两类入口、split caller/runtime/finish、最终
  1:2 ELF、无 relocation 和 manifest 全部通过；
- A5 B256 十个独立 perf-clock 进程全部 execution/semantic/postprocess
  PASS：

```text
min/median/mean/max =
6.924365 / 7.303955 / 7.259680 / 7.595064 ms
```

相对 S6.1 十轮中位 `9.077919 ms` 改善 `19.542%`。B256 full-swimlane
Submit 为 `6.894677 ms`，1280 task、1024 kernel、严格插入、K2 owner、
payload、DCCI/atomic 和 trace closure 全部 PASS，drop 为 0；其
`EfDrainControl` 聚合核时从 `517.839958 ms` 降到 `369.541582 ms`
（`-28.638%`）。Claim 聚合核时仍为 `42.015425 ms`，说明本阶段没有
缩减 96 Scalar Build 参与人口或 Tournament 物理工作。

新泳道在：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260802_161626_4109966/ccec/merged_swimlane.json`

当前距离 `1.0 ms` 仍有约 `6.30 ms`。剩余 full-swimlane 中
`EfDrainControl=369.542 ms` 聚合核时，loser 的该项 p50 仅
`0.203 us`、p95 却为 `23.310 us`；下一阶段应区分活跃 token 的 fanin
等待、候选 cell 尚未 BUILT、payload acquire/bind 与 completion，不能再把
全部长尾笼统归为 EfDrain。

## 2026-08-02：S6.3 未就绪 fanin 轮询避开集中式 fatal 原子读取

### 动态证据与改动边界

S6.2 泳道逐 Submit 展开后，长 EfDrain 不是随机空白：一个 executor
取得 task token 后，如果其首个 producer 尚未 ready，会在后续多个 Submit
入口反复推进同一个 `WAITING_FANIN` token。旧路径每次未就绪尝试至少包含：

1. active-token 入口的 global fatal 返回型读取；
2. `TryMarkExecutionTokenEngineInflight()` 入口的 exec fatal 返回型读取；
3. helper 返回未 ready 后由调用方再次读取 exec fatal。

fanin 未就绪时只读取 completion flag、更新 owner-local ready prefix 并
返回；它不发射 kernel、不发布 completion，也不推进任何共享业务状态。
因此本阶段先调用 `ExecutionTokenFaninReady()`：未就绪直接返回；全部 ready
后才进入保留 exec-fatal 门禁的状态转换 helper。helper 失败时根据
owner-local token phase 判断是否已进入 `Faulted`，不再为了区分“未 ready”
与“fatal”补读一次共享 fatal。

错误路径没有被删除：kernel 发射前、kernel 返回后、completion 发布前以及
Claim CAS 前仍保留原终止门禁；`production_closed=true` 的 FinalDrain
入口强制读取 global/exec fatal，并把无法再等到 fanin 的 active token 收敛
为 `Faulted`。机会式 EfDrain 遇到 global fatal 时可以暂时保持
`WAITING_FANIN`，但在此期间不能产生不可逆副作用，最终必须由 FinalDrain
闭合。

### 正确性与 A5 结果

- 定向 CPU 用例新增“机会式未 ready 不做副作用、FinalDrain 必须 fault”
  的两段门槛；Claim、kernel、completion 注错窗口和全部 scanner 用例 PASS；
- 完整 CPU perf-clock build 与全部 shared 协议、ordered Submit 用例 PASS；
- CCEC AIC/AIV 通用实例、双入口、split runtime/finish、最终 1:2 ELF、
  manifest 和无 relocation 门槛 PASS；
- A5 B256 十个独立 perf-clock 进程全部 execution/semantic/postprocess PASS：

```text
3.644191 ms（进入十轮前的独立首样本）

十轮：
4.093007, 3.593300, 3.651752, 3.570847, 3.583781,
3.904849, 3.635498, 3.814237, 3.701259, 3.529903 ms

min/median/mean/max =
3.529903 / 3.643625 / 3.707843 / 4.093007 ms
```

十轮中位相对 S6.2 的 `7.303955 ms` 改善 `50.113%`。同代码
full-swimlane Submit 为 `3.843832 ms`，1280 task、1024 kernel、严格
插入、96 Scalar Build 资格、K2 异核 Execute、payload、DCCI/atomic 和
trace closure 全部 PASS，drop 为 0。local/root Claim CAS 仍精确为
`122880/10240`，证明没有通过缩减竞争人口取得结果。

full-swimlane 的关键聚合核时变化为：

| 指标 | S6.2 | S6.3 | 变化 |
| ---- | ---: | ---: | ---: |
| Submit 墙钟 | 6.894677 ms | 3.843832 ms | -44.249% |
| SubmitUnion | 570.907881 ms | 302.512968 ms | -47.012% |
| EfDrainControl | 369.541582 ms | 149.858920 ms | -59.447% |
| Claim | 42.015425 ms | 42.439623 ms | +1.010% |

新泳道在：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260802_163545_4133201/ccec/merged_swimlane.json`

当前十轮中位距离 `1.0 ms` 仍差 `2.643625 ms`。新泳道中 Submit 内的
主要聚合核时依次为 EfDrainControl `149.859 ms`、Claim `42.440 ms`、
WinnerBuild `29.358 ms`、SubmitResidual `29.394 ms` 和
BetweenSubmitResidual `18.710 ms`。下一轮先解释 active token 仍有约
4 万次 fanin completion 读取以及每次未 ready progress 的非 atomic
固定成本，再考虑 payload/DCCI；FinalDrain 位于 perf-clock 窗口之外，
不得靠把 Submit 工作机械后移到 FinalDrain 冒充 1 ms 收益。

## 2026-08-02：S6.4 队首阻塞候选验证与撤回

本阶段在 `c3387747` 干净源码上重新取得十轮基线，中位/均值为
`3.678558/3.692169 ms`，并按 `perf-clock <= 1.0 ms` 作为后续统一门槛。

先验证了 scanner 的 `BUILT` control 快照复用。它保持 CAS expected 和
所有 fatal 边界，只删除 Claim helper 对同一 control 的第二次返回型读取；
CPU/CCEC/A5 均通过，但十轮中位仅改善 `0.312%`、均值回退 `0.260%`，已
判为波动内并撤回。

随后验证每 worker 两个 owner-local execution token。新增 CPU 交错覆盖：
第一个 token 持有未 ready SF 时，第二个 token 能领取并完成后续独立 UP；
依赖就绪后第一个 token 再完成，两个 cell 均 exactly-once 到达 `DONE`。
两个 token 都由同一 Scalar 普通读写；共享侧仍只有 task cell control CAS，
没有新增跨核普通发布或 cache-coherence 假设。SchedulerState 增量仅为
`4928 B × 96 = 473088 B`，host/device 初始化、终态检查、fatal 诊断和
FinalDrain 排空均曾扩展到两槽并通过完整 CPU、CCEC 与 A5 门槛。

动态结果却否定了该实现：fanin load 从约 `41K--43K` 降至约
`29K--31K`，但贪心双槽十轮中位/均值升至
`3.773950/3.779694 ms`。full-swimlane Submit 为 `3.873123 ms`，
`EfDrainControl=153.562096 ms`、`WinnerBuild=31.923958 ms` 聚合核时，
单核最大 winner 数达到 39。第二槽增加确定性 primary 优先后，五轮仍为
`3.746665/3.749903 ms`，没有消除回退。

因此单 token 的确存在未就绪队首阻塞，但“提前多领任务”会用 payload
搬运、逐槽检查和 K2 负载偏斜交换较少的 fanin poll，端到端不成立。所有
双 token 过程代码已完整撤回；当前源码恢复为单 token `c3387747`。后续若
继续处理该瓶颈，应采用不提前 Claim、不占 token 的 ready-only 候选重访，
并先闭合依赖可见性与活性证明。

## 2026-08-02：S6.5 ready-only 候选与 readiness owner 负向实验

### 第一版：保持 BUILT，双候选先看 fanin

为避免 S6.4 的 payload 提前搬运，本版在 `SharedExecControl` 原有 64B
atomic-only 行内增加不可变 fanin 摘要。摘要由 builder 在 payload flush 后、
`BUILT` 发布前用返回型 CAS 发布；整个 control 行仍禁止 ordinary store 与
DCCI。四条以内依赖可直接解出 producer，更多依赖显式回退原 payload 路径，
没有把 PA 的 QK/SF/PV/UP 依赖形状写入通用协议。

scanner 未看到全部依赖 ready 时不 Claim、不取得 token、不触碰 payload，
只保留一个 owner-local deferred candidate slot 后继续扫描。确定性 CPU 用例
逐项验证了 `BUILT` 保持不变、Claim CAS 为 0、payload invalidate/load 为 0，
以及依赖发布后 deferred task 与后续 task 均 exactly-once 完成。

该协议在 A5 上暴露出直接缺陷：两个 K2 候选都能看到 `BUILT`，因而同时轮询
同一 fanin。B256 单样本 `Submit=3.902965 ms`，`fanin_loads=82561`，其中
`ready/not-ready=13577/68984`；相较单 token 基线约 4.1 万至 4.3 万次读取
明显恶化。该版立即停止，没有用更多样本掩盖结构性问题。

### 第二版：先选唯一 readiness owner，延后 payload/token 绑定

第二版没有增加新的共享 phase 或第二次 CAS，而是把现有 `CLAIMED` 的含义
精确拆成两个本地步骤：

1. `BUILT -> CLAIMED` 仍是唯一一次 K2 所有权 CAS；
2. winner 若 fanin 未 ready，只保存 candidate slot 和已确认 ready prefix；
3. 此时共享 cell 已有唯一 execute owner，但 owner-local token 仍为 Idle，
   payload 未 invalidate、未复制；
4. 依赖全部完成后，owner 才绑定 payload/dispatch 并进入原同步执行链；
5. peer candidate 看到合法 `CLAIMED` 后清理自己的候选位。

CPU 门槛覆盖任意 Build owner、两候选先后顺序、CAS loss、global/exec fatal
注入、deferred resume、FinalDrain 和超出四条 fanin 的通用 fallback。完整
CPU、协议 probe、CCEC AIC/AIV 和 A5 B256 均通过；A5 的 ready 读取精确等于
业务依赖边 `1280`，说明 prefix 与唯一 owner 合同确实成立。

性能却稳定回退。同步基线十轮中位为 `3.678558 ms`；该候选独立首样本
`5.078318 ms`，后续五轮为：

```text
5.097780, 5.041501, 5.183033, 5.116527, 5.011070 ms
min/median/mean/max =
5.011070 / 5.097780 / 5.089182 / 5.183033 ms
```

中位回退 `38.580%`。唯一 owner 虽消除了双候选 fanin 读取，却会在延后槽
占用时阻止第二个有依赖 candidate 取得 owner，并增加摘要发布、Claim/Bind
分拆、恢复分支与 scanner 代码体。结果说明本架构下的 fanin 原子次数不是
可以脱离整体推进顺序单独优化的指标。

两版代码、ABI、host oracle 与测试改动均已完整撤回；没有形成生产提交。
S6.5 的结论是停止 ready-only/提前 ownership 路线，回到单 token 干净基线，
从实际 perf-clock 差额中寻找不引入新任务上下文的更小优化点。

## 2026-08-02：S6.6 fanin 轮询降频实验与计时边界复核

本轮验证了一个不增加任务上下文的小候选：active token 在机会式 EfDrain
读到未 ready fanin 后，跳过后续固定次数的 Submit 轮询；WaitForSlot 与
FinalDrain 不经过该门控。CPU 定向用例、完整 CPU、CCEC 和 A5 业务终态均
通过，说明协议活性没有丢失。

动态数据却证明它不应保留。skip=1 的十轮中位为 `3.643993 ms`，仅比
`3.678558 ms` 同步基线低 `0.940%`，同时 FinalDrain kernel 数明显增加。
skip=2 将 fanin load 从约 `42.5K` 降至约 `28.3K`；但两套独立 ELF 的
同时间段交错 A/B 中，基线三次为
`3.650139/3.647333/3.634740 ms`，skip=2 为
`3.763413/3.751776/3.915294 ms`，中位反而回退 `3.182%`。

更关键的是，代码边界复核确认 perf-clock 的 `submit_end` 在最后一次
`CloseSharedCallbackSubmit()` 内读取；replay-done barrier 和 FinalDrain
发生在该边界之后。skip=1/2 分别把 FinalDrain kernel 数提高到约
53--67 和 88--93，属于把执行推进移出计时窗，而不是消除调度工作。因此
即使某组 Submit 数字下降，也不具备保留资格。

所有降频过程代码和测试改动已经撤回。后续任何 execution-progress 候选都
必须同时核对机会式 placement 与 FinalDrain 增量；若目标仍定义为完整
Submit 调度周期，则最终还需要把权威 perf-clock 结束边界扩到 execution
drain 闭合，避免观察口径奖励工作后移。

## 2026-08-02：S6.7 K2 主候选优先，保留有限兜底

### 设计边界

K2 仍为每个 task 提供 primary/secondary 两个合法 Execute 候选，不把执行
owner 静态绑定到某一个核。正常路径优先选择与 Build owner 不同的 primary；
若 primary 恰好是 Build owner，则优先 secondary。非首选候选首次观察到
`BUILT` 时只在本核让出一次，第二次即可正常竞争；生产关闭后的 FinalDrain
不等待首选核。

该变化只改变同一 task 的两个合法候选何时发射 Claim CAS，不改变：

- TensorMap 严格有序插入和 `BUILT` 发布顺序；
- 96 个 Scalar 的 Build 资格；
- K2 的候选集合与跨核执行约束；
- execution control ABI、payload 发布/取得和 completion 协议；
- 单 execution token 与 FinalDrain 完整排空。

`LocalStats::max_occupied` 的取值上界是 `kPrivateSlots`，因此从 16 bit 收紧
为 8 bit，并在原 4-byte 紧凑块中复用 1 byte 保存本地 defer 次数，没有扩大
CCEC block-local 运行时。candidate cursor 一旦前进，defer 次数立即归零，
不会串到下一 task。

### 门槛与结果

CPU 新增确定性交错，证明：首选先到时立即执行；备选先到时第一次零 CAS、
首选随后执行；首选 token 忙时，备选在一次宽限后仍能接管。完整 CPU、CCEC
AIC/AIV 编译与 A5 B256 的 1280 task、1024 kernel、strict insert、K2 owner、
payload/fanin/vend/fatal/终态门槛均通过。

十轮 perf-clock 中位为 `3.637995 ms`，相对同步基线 `3.678558 ms` 改善
`1.103%`。七对独立 ELF 交错 A/B 中，候选 6 对更快；基线/候选中位分别为
`3.773684/3.631773 ms`，改善 `3.761%`。FinalDrain 数量为候选 `39--48`、
基线 `40--50`，fanin load 仍约 `42K`，没有把执行推进移出 Submit 窗口。

full-swimlane 证据位于：

```text
tests/atomic_probe/pa_scheduler/outputs/
pa_scheduler_cross_core_shared_swimlane_20260802_193244_139300/
ccec/merged_swimlane.json
```

其 Submit 为 `3.655982 ms`，全部检查通过且 trace drop 为 0。宽限为 2 的候选
未优于宽限为 1，已撤回。当前保留宽限 1，并继续以 B256 perf-clock 稳定
不高于 `1.0 ms` 为最终目标。

## 2026-08-02：S6.8 撤回次候选首次共享读取前移

本轮把 S6.7 次候选的一次本地让出移到首个 execution control load 之前，
理论上每个 task 最多少一次返回型共享读取。CCEC 构建与三个 A5 B256 样本
全部通过，Submit 为 `3.642054/3.668757/3.648742 ms`，中位
`3.648742 ms`，没有优于 S6.7 的 `3.637995 ms`。

该变化还会让依赖“一次 scanner 调用观察终态”的 9 个 CPU 场景多推进一次。
虽然协议最终仍能闭合，但现有数据不足以用更宽的时序状态空间交换最多
1280 次 control load。生产代码和临时测试断言已经撤回；下一阶段转向能改变
Claim 返回型原子竞争规模的结构性候选。

## 2026-08-02：S6.9 先验证 one-shot Claim 预过滤

真实 Claim 暂未改动。独立 CCEC AIV 微基准先复刻当前 local tournament
的关键拓扑：12 个候选竞争一条 per-task、初值为 `-1`、只发布一次 task id
且不复用的 atomic-only 状态。一个轮换首选候选直接 CAS，其他候选用
`ld_dev` 做一次保守预过滤，读到旧值时仍回退 CAS。

100 NOP 的版本把每轮 CAS 从 `12.0` 降至 `4.4`，loop/round 中位从
`2309.3` 降至 `1392.3` tick，分别减少约 `63.3%` 和 `39.7%`。首选候选
延后 1000 NOP 的对照中首选胜率为 0，但其他候选仍恰好选出一个 winner，
证明它不是静态 owner。所有变体 semantic failure 为 0。

该机制的正确性依据是 stale-low 只会增加 CAS，不会丢任务；因此只能用于
一次性 per-task node。下一阶段接入 local Claim 时保持 root CAS、96 核
资格和严格 TensorMap 插入不变，先看真实 B256 能否把微基准收益转化为
perf-clock 收益。

## 2026-08-02：S6.10 撤回 Claim 旁路预过滤

local Claim 实际接入了轮换首选候选与 `ld_dev` 预过滤，分别测量
0/50/100/500 NOP。所有版本都保留 96 核逻辑 Claim、G8 root、严格插入链
和完整业务终态；CPU 全套与 CCEC 编译均通过。干净 S6.7 在同一时段重编后
五轮中位为 `3.675272 ms`，四个候选中位依次为：

```text
nop0   3.687956 ms  (+0.345%)
nop50  3.680760 ms  (+0.149%)
nop100 3.671037 ms  (-0.115%)
nop500 3.671612 ms  (-0.100%)
```

名义最好改善只有 `0.115%`，不足以越过运行波动，也未优于 S6.7 历史十轮
中位 `3.637995 ms`。独立微基准的 CAS 降幅无法覆盖真实 Submit 新增的
旁路读取、控制流和错峰成本；500 NOP 还明显改变 fanin/执行推进节奏。

因此生产实现、host 统计范围和 CPU 断言全部恢复到每候选一次 local CAS，
并重新编译干净 perf-clock 产物确认。仅保留探针的 200/500 NOP 档位和本次
负向结论。下一步不再微调 prefilter 延后，而是寻找能直接减少协议级返回型
原子数量、且不在每次 Claim 前新增共享读取的结构性方案。

## 2026-08-03：S6.11 随机访问构参等价门槛

当前 96 个 worker 各自顺序 replay 1,280 个 task，除了参与 Claim，还用
`AcceptTaskOutputs()` 保存本核后继 callback 所需的输出 handle。源码审计确认
shared handle 只是确定性的 `(producer_task_id, output_slot)`，不携带 winner
私有地址；因此“任意 Scalar 能 Build”不必以“每核必须先回放全部前驱”为
唯一实现方式。

新增 `BindSharedPaTaskForRandomAccess()`，只从经过校验的
`(batch, batch_start, task_id, kind, group_index)` 重建当前 callback 会读取的
动态 batch/group 状态和前驱 output ref。它不触碰 TensorMap、Claim、执行
cell 或发布状态，并且只为 SF/PV/UP 写入严格早于当前 task 的 producer。

CPU 等价门槛逐 task 比较顺序 replay 与随机访问构参，覆盖混合
G0/G1/G2/G4、全部五类 task、动态 shape、query/output view、scalar 和
SharedOutputRef；共 41 个 task 全部逐字段一致。门槛还在每次随机构参前污染
上一 task 的动态状态，证明结果不依赖历史 `AcceptTaskOutputs()` 遗留值，并
验证所有 active shared producer 均满足 `producer < task_id`。
完整 CPU `perf-clock` 构建与全部门槛已通过；CCEC `perf-clock`
的 AIC/AIV 入口、split runtime/finish、mixed ELF、符号和重定位检查也已通过。

本阶段没有改生产调用点，也没有 A5 性能结论。它只关闭任务发放重构的第一项
前置证明。仍需独立解决：

- `batch_start` 必须来自全局连续且不可变的权威 plan；单 task helper 不扫描
  历史 batch，也不自行冒充全局 oracle；
- 当前 K2 scanner 依赖每 worker 的连续 Close 前缀和 owner-local candidate
  位图；减少全量 replay 前，必须用全局发布前沿或等价合同替换这两项依赖。

## 2026-08-03：S6.12 中央 Build ticket 的独立 CPU 协议门槛

本阶段仍不改 CCEC 生产 Submit。新增独立 96-thread CPU 门槛，验证一种可与
当前 96 份完整 replay 对照的候选协议：host 先产生不可变 flat plan，每个
Scalar 通过一个单调 `FetchAdd` 取得唯一 task id，再使用 S6.11 已验证的
随机构参入口准备该 task。

门槛刻意把 TensorMap 串行边界和 Build 完成拆开：task N 只等待
`deps_prepared[N-1]`，发布自己的 `deps_prepared[N]` 后立即交出插入 baton；
Fanin/Build 不属于该串行区。测试让 task 0 在发布完成字后继续延迟，直到至少
16 个后续 task 已完成 Build，稳定证明后继 Build 不会被前任 Build 串行化。
插入日志仍必须严格等于 `0..1279`。

十轮 B256 CPU 交错全部通过，逐轮同时满足：

- 1280 个 task 的 ticket、随机访问构参、owner、Build 均恰好一次；
- 96 个 worker 都取得过至少一个 task；
- task 0 延迟期间至少 16 个后继完成 Build；
- 每个 task 的插入完成字只由 `-1` 发布为自身 task id，顺序无跳跃；
- 全部 96 个 worker 退出生产后，最后一个退出者才发布
  `production_closed`；
- 错 task id、错 `batch_start` 和 `producer >= consumer` 的输入均被拒绝；
- flat plan 在并发执行前后指纹不变。

在当前 B256 G1 形状下，候选路径的 ticket `FetchAdd` 调用数为
`1280 + 96 = 1376`：每个有效 task 一次，另加每个 worker 一次越界取票后
退出。现有 G8 两级 Claim 的物理 CAS 为 `122880 + 10240 = 133120`，因此
这里只证明协议级原子调用数减少 `98.966%`。

CPU 不模拟 A5 同地址 atomic 的线性退化，不能据此宣称延迟或 Submit 收益。
在接入生产代码前必须先做 A5 微基准，对照当前 per-task G8 CAS 与中央
`FetchAdd` 热点；如果中央地址的串行化代价抵消调用数收益，则改为分片 ticket
或其他经过上板证据支持的发放结构。当前 K2 candidate 登记、执行发现和
FinalDrain 尚未迁移，本阶段不宣称已经替换 96 份 replay。

## 2026-08-03：S6.13 A5 mixed Build 发放原子拓扑取证

在接生产 Submit 前，新增独立 CCEC mixed 探针，直接复刻当前 B256 Claim
的物理形状：32 AIC + 64 AIV、全部 96 核参与、1280 个 task、每 task 八个
local node、512B node 间距，以及 `96 local CAS + 8 root CAS`。候选路径让
同一批 96 核从一个单调 `FetchAdd` cursor 领取 1280 个唯一 ticket，每核
取得首个越界值后退出。

同一 ELF 的 empty/G8/ticket 三模式轮换 11 轮。计时点位于 96 核启动会合
之后；host 同时按最早 begin 到最晚 end 计算 device span，并记录最慢单核
elapsed 和 begin spread。每轮均精确验证 task 总数、task-id sum/xor、全部
tournament node 终值、每核身份和最终 cursor。

```text
模式                    原子数    device span 中位    最慢单核 elapsed 中位
empty                        0          16.535 us                6.615 us
当前 per-task G8 CAS    133120         556.900 us              554.644 us
中央 FetchAdd ticket      1376         253.851 us              251.557 us
```

中央 ticket 在隔离原子流中缩短 `303.049 us`，相对 G8 的 raw span 降低
`54.417%`。empty 只有 16.535 us，说明约 0.303 ms 差值不是 mixed 启动
假象；同时 ticket 自身仍需约 0.25 ms，证明 A5 同地址返回型原子会形成明显
串行代价。

该证据支持继续做生产候选，但不等于完整 Submit 可直接减少 0.303 ms：真实
路径还会改变 replay、EfDrain、随机构参、TensorMap 插入等待和 K2 candidate
发布。下一阶段接入时必须保留严格插入链与最终闭合，并用独立 perf-clock
A/B 决定保留或撤销。

## 2026-08-03：S6.15 发布 device 可读的紧凑 task plan

在不改 Submit 控制流的前提下，先把 S6.11--S6.14 使用的权威 task identity
落入 standalone state 尾部。`SharedBuildDispatchState` 的第一条 cache line
只放未来的 `next_task` atomic；只读 header 从第二条 cache line 开始，随后
每个 task 只占 4 字节：`uint16 batch + uint8 encoded_meta + uint8 reserved`。
task id 由数组下标给出，`batch_start` 由既有 meta 中的 kind/group 和 task id
反推，因此不复制 TensorDesc、TaskArgs 或 worker 私有地址。

B256/G1 的 1280 个 task 只增加 5 KiB 计划数据；按编译上限 4352 task 计算，
整个 dispatch sidecar 为 17,536 B。它追加在 execution token 之后，不移动
production prefix、shared TensorMap、Claim Tournament、exec cell 或 token 的
既有 offset；host/device 尾部传输范围同步扩展。

host 从独立 `SharedHostTaskPlan` 一次发布计划；device 解码同时校验 task/batch
上限、reserved、meta 和全局末次标记。混合 G0/G1/G2/G4 自检逐 task 完成解码
与随机构参绑定，并拒绝非连续 host task id、越界 batch、非零 reserved 和丢失
末次标记。完整 CPU `perf-clock` 回归和 CCEC AIC/AIV/mixed ELF 构建均通过。

本阶段生产仍走原 96 份 replay，A5 为 **NOT RUN**；新增 plan 只是下一阶段
scanner 与 ticket 共用的唯一只读身份源，不能单独解释成性能收益。

## 2026-08-03：S6.16 K2 scanner 改用 immutable plan

在仍保留 96 份完整 replay 的过渡形态下，生产 scanner 已不再用
`exec_candidate_bitmap` 判断 task 是否存在。每个潜在 task 先解码 host 发布
的 4B identity：Alloc 和错 engine task 直接推进本地候选游标；属于本核 K2
的 task 在 `EMPTY/BUILDING` 时保留队头，`BUILT` 后继续走原主候选优先、备选
有限兜底和 exactly-once execution CAS。

旧 replay 目前仍在 Close 时登记位图，以便单独比较 scanner 改动；plan scanner
只在推进游标时清除过渡 bit，不读取它决定任务资格。CPU 状态机据此修正了旧
假设：权威计划已声明的相关 `EMPTY` 不能因“本地尚无 bit”而永久越过；独立
FinalDrain 测试也保持原 LocalStats cursor 生命周期，不再重建后让 execute
owner 重读自己的 `DONE` task。

完整 CPU 回归和 CCEC perf-clock 构建通过。A5 B1 为 `127.404 us`，全部语义
与终态检查通过。同一时段交错五对 B256，旧 scanner 与 plan scanner 分别为：

```text
旧位图：3.664350, 3.673144, 3.648508, 3.659993, 3.616991 ms
新计划：3.686876, 3.666555, 3.641949, 3.947700, 3.652301 ms
中位数：3.659993 ms -> 3.666555 ms（+0.179%）
```

新路径有一轮 3.948 ms 长尾，其余四轮为 3.642--3.687 ms；中位差处于当前
运行波动量级，没有证据表明结构性回退。该阶段保留，因为它关闭了 central
ticket 接入前的任务发现缺口；下一阶段才删除全员 replay 和过渡位图写。

## 2026-08-03：S6.14 去除全员 replay 后的 K2 发现门槛

生产接入前的源码审计发现，现有 `exec_candidate_bitmap` 不是共享任务队列：
每个 worker 顺序 Close 全部 Submit 时，只有真实 K2 候选在自己的本地位图
登记该 task。中央 ticket 让每个 task 只经过一个 Build owner 后，这一隐含
前提不再成立；若只替换 Claim，执行侧会漏掉没有在本核 Close 的 task。

因此扩展 S6.12 的同一个 96-thread CPU 门槛，不增加第二套发放模型。每个
worker 只持有单调 `candidate_slot`，按生产 `FixedPaExecuteCandidates()` 的
K2 映射枚举自己可能执行的 task，再用不可变 flat plan 判定 task kind 与
engine：Alloc 和错角色 task 直接越过；相关 task 在 Build 发布前保留队头，
发布后由两个候选竞争唯一 Execute owner。Build owner 若恰好属于 K2，只发布
执行包并越过，另一候选负责执行。

B256 连续十轮均满足：1280 个 task 恰好一次 Build，1024 个非 Alloc task
恰好一次执行；每个 Execute owner 都属于独立复算的 K2 集合且不等于 Build
owner；256 个 Alloc 不产生执行；task 0 延迟 Build 时后续 Build 仍继续推进；
全体 worker 退出生产后，最终扫描越过完整计划且没有等待未发布 task。

完整 `cross_core/cpu/build.sh perf-clock` 回归通过。该阶段仍是 CPU 状态机证明，
没有修改 CCEC Submit，也没有 A5 性能数据。生产接入必须让 device 获得同一份
只读紧凑 task identity，并让 scanner 以该 identity 替代 owner-local 位图；
在全体 Build owner 退出前，相关 `EMPTY/BUILDING` 仍只能解释为暂未发布，不能
提前当成缺口或永久越过。

## 2026-08-03：S6.17 中央 ticket 替换 96 份 Build replay

本阶段把 S6.11--S6.16 的前置证明接入 shared PA 正式路径。96 个 Scalar 不再
逐核回放 1280 个 task，也不再进入 per-task G8 Claim Tournament；每核从
`SharedBuildDispatchState::next_task` 执行一次返回型 `FetchAdd` 领取 task，
从 immutable plan 解码 task/batch/kind/group，再用已验证的随机访问入口构参。
每个 task 只有一个 Build owner，但 owner 不绑定核型，也不等于后续 K2
execute owner。

TensorMap 的正确性合同没有被 ticket 取代：Build owner 仍在 Register 中等待
`deps_prepared[N-1]`，只按 task-id 顺序发布 metadata 和本 task 完成字；完成
字发布后，Fanin/Build 与后继 task 的 metadata 插入重新并发。K2 scanner 在
每次取票前按 immutable plan 推进本核候选，最终阶段仍闭合全部 1024 个
非 Alloc task，因此“严格插入”和“Scalar 自由 Build/独立 Execute”继续解耦。

正常 B256 的发放原子预算从 G8 的 133120 次 CAS 变为：

```text
1280 个有效 task ticket + 96 个 worker 越界 ticket = 1376 FetchAdd
```

host 逐项验证全局 task `0..1279` 恰好一个 owner、每核恰一次越界 ticket、
旧 tournament/cursor 保持初值、每类 task 计数、task-id sum、split caller/
finish 稀疏所有权、严格插入链、K2 owner/payload/fanin/vend 和最终状态。
CPU 还覆盖错 plan、重复/缺失 owner、错误 endpoint 及错误 terminal cursor。

泳道 raw 继续使用既有 task-indexed endpoint 数组，不为中央发放增加每事件
字段。稀疏 owner 使观察导出必须清理全局 `task_count` 个 endpoint，而不能只
清理本核 `submits` 前缀；该 DCCI 位于 Submit/Kernel 计时之后。converter 和
analyzer 以 `submit_topology=central_ticket` 校验全局唯一 ownership。越界
ticket 前的 opportunistic drain 可能在末次有效 Submit 后执行 Kernel；分析器
用既有 `OrchestrationReplay` 与 Submit 边界把它严格归入
`OrchestrationTail`，并拆成 KernelUnion/ScalarControl，不增加 device raw。

A5 B256 full-swimlane 已完整通过，产物为：

```text
tests/atomic_probe/pa_scheduler/outputs/
pa_scheduler_cross_core_shared_swimlane_20260803_052743_821381/
ccec/merged_swimlane.json
```

该次 Submit 为 `2.280878 ms`，1280 task、1024 kernel、1376 ticket、严格
TensorMap 插入、K2 执行及 trace/atomic/DCCI 闭合均 PASS。

显式重编 perf-clock 后的五次 B256 为：

```text
2.311447, 2.511888, 2.093339, 2.641146, 2.362004 ms
median = 2.362004 ms
```

相对 S6.16 同阶段中位 `3.666555 ms`，下降 `1.304551 ms`，即
`35.580%`。收益来自删除 96 份 replay 和 133120 次两级 Claim CAS 的结构性
改动，不能只归因于单个 `FetchAdd`。中央 ticket 仍是一个约 0.25 ms 的 A5
同地址返回型原子热点；下一阶段应先根据新泳道重新排序非 ticket 开销，再决定
是分片发放还是继续消减随机访问构参/插入等待，不能在缺少协议证明时替换它。

## 2026-08-03：S6.18 为 WinnerBuild 建立稳定 descriptor 引用协议门槛

S6.17 的完整泳道中，1024 个非 Alloc task 的 `WinnerBuild` 聚合约
`72.954 ms`，单次常见为 `60--80 us`。源码审计确认当前 portable payload
不区分 descriptor 生命周期：即使 `SharedOutputCell` 或外部 GM 中的
`TensorDesc` 已经稳定发布，builder 仍复制完整 128B，executor 随后还要再次
搬入自己的 token。PA G1 每组共携带 19 个 tensor 参数，旧布局合计发布
`10+10+10+16=46` 条 cache line。

先验证了四个更小的候选，均未保留：目的 cache-line preload 的十轮中位相对
干净基线回退约 `3.10%`；source descriptor preload 的初次顺序样本看似改善，
但冻结 ELF 的六对交错 A/B 中位回退 `3.21%`，且 `WinnerBuild` 聚合反而增加
`1.39%`；把 failure-only helper 标成 cold/noinline 会把 AIC/AIV 既有 finish
调用点从 3 改为 4，结构审计直接拒绝。另一个“复用已验证 layout”候选经函数
机器字节逐字比较，AIC/AIV 都与基线完全一致，证明 CCEC 已做等价消除，也已
撤回。上述结果避免把设备漂移、源码简化或预取直觉冒充收益。

随后只增加通用协议能力，尚未让正式 PA 选用引用。`ExecPayloadSpec` 新增 active
tensor reference mask：bit=0 继续内联完整 128B descriptor；bit=1 只携带一个
8B、非空且对齐的稳定 GM 地址。变长布局按 mask 计算每个逻辑 tensor 的真实
word offset，因此全内联最大 payload 仍是 4352B/68 lines，`SharedExecCell` 和
`ExecutionToken` 容量 ABI 均未扩张。executor 取得 cell 后先 invalidate/copy
payload，再对每个引用 descriptor 独立 invalidate 128B；内联 descriptor 仍
重绑到 executor-private token，引用地址则直接进入 dispatch args。

CPU 协议门槛覆盖 3 个 tensor 中 2 引用、1 内联的混合形态：payload 从
496B/8 lines 降为 256B/4 lines，tensor/scalar/fanin 值逐项一致，两个引用地址
保持、内联地址重绑，claim 总计执行一次 payload invalidate 和两次 descriptor
invalidate；越界 mask 与空引用均 fail-closed。AIC/AIV 优化后 IR 进一步精确
验证 `CAS -> payload DCCI/DSB -> conditional reference DCCI/DSB`，pre-DSB
对照也保持额外屏障位于 payload acquire 之前。完整 CPU 回归和正式 CCEC
perf-clock 构建均通过。

本阶段正式 PA 仍使用 `tensor_reference_mask=0`，因此 A5 引用正确性和性能为
**NOT RUN**，也没有性能收益声明。下一阶段只能由 adapter 按地址空间与生命周期
选择引用：已经完整发布且在 kernel 完成前不可变的 GM descriptor 可引用；
builder 栈、block-local 或会被复用的 descriptor 必须继续内联。接线后需要以
CPU 故障门槛、A5 B1/B256、payload 行数和 perf-clock 交错 A/B 共同裁决。

## 2026-08-03：S6.19 正式 PA descriptor 引用候选回退并撤回

按 S6.18 的生命周期合同做了正式 PA 接线候选：fresh Output 和
`SharedOutputRef` 只引用 task-indexed `SharedOutputCell`，外部稳定 GM
descriptor 允许引用，builder 私有或 local descriptor 继续内联。host 不仅
比较 descriptor 内容，还把 device 地址映射回 D2H `SchedulerState`，逐 tensor
验证它精确指向预期 `(producer_task_id, output_slot)`，避免相同内容掩盖错误
地址。

隔离 adapter 测试得到如下真实布局：

| task | 旧 payload | 引用候选 | reference mask |
|---|---:|---:|---:|
| QK | 592 B / 10 lines | 472 B / 8 lines | `0x8` |
| SF | 604 B / 10 lines | 124 B / 2 lines | `0xf` |
| PV | 596 B / 10 lines | 356 B / 6 lines | `0x9` |
| UP | 988 B / 16 lines | 268 B / 5 lines | `0x3f` |
| 每组合计 | 46 lines | 21 lines | 13 个引用 tensor |

完整 CPU 回归通过；A5 B1 的 payload 精确地址、descriptor、fanin、K2 owner、
执行结果和终态检查全部通过，Submit 为 `262.405 us`。B256 单次也全部通过，
但 Submit 为 `2639.451 us`，已显示回退。随后冻结接线前/接线后两套 ELF，按
`B-C / C-B / B-C` 交错三对：

```text
接线前：2104.853, 2159.634, 2272.761 us
引用版：2773.649, 2433.551, 2614.464 us
中位数：2159.634 -> 2614.464 us
差值：+454.830 us（+21.060%）
```

三对均回退，不能用设备波动解释。相同样本中 `fanin_loads` 中位从 `15924`
升到 `17460`；这反映更慢的执行推进会让轮询进一步放大，不单独解释成引用
协议的直接成本。更直接的结构差异是：每组虽然少发布 25 条 payload cache
line，却把 13 个稳定 descriptor 的获取推迟给 executor；B256 一共新增/迁移
3328 个引用区域的 invalidate/读取，且执行侧地址更分散。

候选 full-swimlane 构建当时被“旧 AIC caller 必须有 5 个 finish 重定位”的
门槛拒绝，因此本节没有使用局部泳道数据。后续用精确父提交复核发现，基线与
候选的 AIC/AIV caller 都同样为 3 个重定位；这属于 S6.18 后代码合并形态变化，
不是 descriptor 引用候选造成的结构差异，不能当成第二条否决证据。正式 PA
adapter、host 校验和 PA 形态测试的候选改动仍依据稳定的 `+21.060%` perf-clock
回退全部撤回；S6.18 已提交的通用变长 payload 能力与隔离协议测试保留，但
生产继续使用全内联 mask=0。后续 WinnerBuild 优化不能再用“把 descriptor
读取整体后移到 executor”这一形态，应优先减少重复解析/发布操作，或先证明
可以批量获取引用区域而不增加逐 descriptor acquire 成本。

## 2026-08-03：S6.20 收敛 full-swimlane split-finish 结构门槛

S6.19 后用精确父提交和后继候选分别重编 full-swimlane，确认两套源码的
AIC/AIV caller 都恰好生成 3 个 role-compatible finish `.rela.text`
重定位；旧门槛仍把 swimlane AIC 固定为 5，因而会同时误拒绝干净基线和候选。
这不是 task kind 覆盖减少：中央 ticket 的五类 dispatch、1280 个 task 唯一
Build、1024 个 kernel 闭合继续由 CPU 动态测试验证，CCEC 结构检查也仍要求
caller/runtime/finish 的唯一强符号、角色隔离、外部 block-local state 引用和
最终 mixed ELF 无残留重定位。

本阶段只把 swimlane AIC 的精确期望从旧值 5 更新为当前真实值 3，与 AIV 和
perf-clock 保持一致；没有改成范围判断，也没有修改 device kernel。修正后，
精确父提交与后继候选的 AIC/AIV full-swimlane 构建均通过。该提交只恢复观察
链路，不包含 A5 性能收益。

## 2026-08-03：S6.21 撤回 unique-ticket 单 CAS 发布候选

中央 Build ticket 已证明每个 task 只有一个 builder，因此测试了去掉
`EMPTY -> BUILDING` CAS、只在 payload flush 后执行一次 `EMPTY -> BUILT`
CAS 的候选。隔离协议测试覆盖了 pack 中点和 flush 后两个暂停点：control
保持 `EMPTY` 时 executor 均返回 `NotBuilt`，不 invalidate、不读取半包；
优化后 AIC IR 也证明 payload DCCI/DSB 先于唯一发布 CAS。完整 CPU、CCEC、
A5 B1/B256 正确性均通过，AIC/AIV `PublishCrossCoreExecTask` 机器码各缩小
172B。

但性能证据不支持保留。冻结精确父提交和候选 ELF 后，六对交错 perf-clock
样本为：

```text
基线：2371.584, 2284.384, 2124.627, 2257.413, 2573.571, 2361.875 us
候选：2531.046, 2331.260, 2376.136, 2135.396, 2323.952, 2598.416 us
中位：2323.130 -> 2353.698 us，+30.569 us / +1.316%
逐对：4 对回退，2 对改善
```

为排除端到端波动掩盖局部收益，又在修正后的同一 full-swimlane 观察链上各跑
一次 B256。基线/候选的 `WinnerBuild` 聚合分别为
`71,461,026 / 73,034,423 cycles`，候选增加 `1,573,397 cycles / 2.202%`；
分析器的 global Submit makespan 也由 `1687.422 us` 增至 `1793.707 us`。
两边结构与分段闭合检查均通过，所以这不是观察结构不同造成的假差异。

该候选的正确性合同成立，但“少一次原子”在本机并没有转化为局部或端到端
收益；不能仅凭指令数量保留。生产调用、通用单 CAS 分支和对应过程测试已全部
撤回，继续使用通用的 `EMPTY -> BUILDING -> BUILT` 双 CAS 协议。本结果只
否决当前中央 ticket + PA payload 形态下的实现，不推导为其他 payload 或硬件
上的普遍结论。

## 2026-08-03：S6.22 Claim-first 每核双 token 与完整周期计时

### 调度合同

本阶段没有引入 ready-before-Claim 或真正 Ready queue，只把原单 token
admission 严格扩为两个 owner-local token：

1. 每次调度点先推进本核两个已领取 token；任一 ready task 先于新 Claim 和
   新 Build 执行；
2. 一个 token 占用且未 ready 时，允许对后续合法 `BUILT` task 发射 Claim，
   winner 绑定到第二槽并立即检查依赖；
3. 两槽都占用时不观察、不 CAS 第三个 task，但外层仍可领取并完整 Build 一个
   ticket；下一次调度点再次先检查两个 token；
4. Build ticket 一旦取得仍不可中途挂起；没有把 Build 改成协程；
5. FinalDrain 只有在 scanner 封口、两个 token 全字段复位、全部 execution cell
   到达终态并发布 drain release 后才允许退出。

两个 token 都只由 owner Scalar 普通读写；跨核可见性仍完全复用既有
`SharedExecCell` 的 payload DCCI/DSB、`BUILT -> CLAIMED` CAS 和 completion
发布。`SchedulerState` 只增加一个 `ExecutionToken`/worker：
`4928 B × 96 = 473088 B`；TaskCell、TensorMap、Build ticket 和 raw trace ABI
均未改变。

### 正确性门槛

CPU 定向用例先在旧单 token 代码上得到预期失败，再由新实现闭合：

- `busy-token-resumes-scanning`：slot0 持有 blocked task，slot1 领取并执行后续
  ready task；前者依赖发布后再 exactly-once 完成；
- `busy-candidate-second-token`：一个 occupied token 不再阻止第二次合法 Claim，
  `max_occupied` 精确达到 2；
- `two-blocked-tokens-stop-claim-permit-build`：两槽 blocked 时第三个 task 不发生
  control load/CAS，但同一 worker 仍能取得 Build ticket；
- `final-drain-closes-last-task`：两个 token、全部 cell 和全局 drain 一起闭合。

完整 CPU perf-clock build 的 shared plan、随机构参、动态 Build dispatch、
TensorMap ring、execution adapter、scan/drain 和 ordered Submit 全部通过；CCEC
AIC/AIV 通用实例、双入口、split runtime/finish、1:2 mixed ELF、manifest 和
无残留 relocation 门槛通过。A5 B1/B256 的 1280 task、1024 kernel、严格插入、
payload、K2 路由和终态检查也全部 PASS。

### 收敛为唯一端到端性能边界

旧无泳道构建只保存首个 Submit 起点和最后一个 Submit 终点。双 token 会主动
把 kernel 从 FinalDrain 前移到 Submit，单看该指标会把有效工作前移误判成调度
回退。新实现复用 `WorkerResult::finish_cycle`，在全部 execution drain 闭合后
通过单独 noinline helper 读取终点，唯一性能区间为：

```text
global min(first-submit-begin) -> global max(FinalDrain.end)
```

最终版本删除了末个 Submit 的性能取时，`submit_end` 在无泳道构建中固定为
0。每核只读取首个 Submit 和 FinalDrain 结束两个边界，没有新增状态字段、
泳道事件或逐 task 诊断。CPU runner 明确校验 `2 × 96 = 192` 次；host 只输出
`PERF-E2E` 和 `submit_to_final_drain_us`。

### A5 严格同底座对照

对照从父提交 `afad388d` 构建单 token ELF，只移植相同的完整周期时钟，并同步
主工作区本轮开始前已有的“删除成功 Build 重复 exec-fatal 读取”改动。候选为
双 token；两边均为 trace-free real-compute B256，按 `B-C / C-B / B-C`
交错运行：

```text
过渡性诊断中的单 token Submit：2022.789, 1871.997, 1861.860 us
过渡性诊断中的双 token Submit：2357.425, 2455.617, 2553.549 us
诊断中位：1871.997 -> 2455.617 us，+583.620 us / +31.176%

单 token 完整周期：9368.900, 9345.107, 9198.695 us
双 token 完整周期：5894.576, 5880.792, 5737.620 us
中位：9345.107 -> 5880.792 us，-3464.315 us / -37.071%
```

同一组三轮的分布闭合为：EfDrain kernel 中位 `208 -> 432`，FinalDrain
`816 -> 592`，恰有 224 个 kernel 从尾部前移；fanin load 中位
`16472 -> 13361`，下降 `18.887%`。所以双 token 的保留依据是完整周期显著
下降，不宣称原 Submit 窗改善。

随后还验证了“只在必要时回看较早 token”的最小轮询序列。它把 fanin load
中位从 `13392` 降到 `12224`，但三对交错中 Submit 中位
`2375.742 -> 2418.871 us`、完整周期中位
`5829.593 -> 5902.238 us`，分别回退 `1.815%/1.246%`；该过程改动已撤回。
这再次说明返回型读取数量只能作为解释量，不能替代完整周期性能裁决。

本阶段 full-swimlane 产物为：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260803_105154_1488931/ccec/merged_swimlane.json`

该次 host Submit 为 `2.714662 ms`；分析器全局 Submit 为 `1.952273 ms`，
EfDrain/OrchestrationTail/FinalDrain 分别容纳 `342/96/586` 个 kernel。与此前
单 token 最佳泳道的 `158/52/814` 相比，方向与 trace-free 完整周期一致。

### 唯一端到端口径复核

进一步删除末个 Submit 的性能取时后，CPU B1 明确闭合每核两次、全局
`2 × 96 = 192` 次边界读取；完整 CPU 回归和 CCEC AIC/AIV/mixed ELF
结构门槛均通过。A5 动态结果为：

- B1：`submit_to_final_drain_us=9287.240`，4 个 kernel 全部闭合；
- B256：`submit_to_final_drain_us=5969.437`，1280 个 task、1024 个 kernel
  和全部终态检查通过，EfDrain/FinalDrain 分别执行 `430/594` 个 kernel。

本机 shell 没有 `task-submit` 和 `npu-smi`，两轮均为 device 0 未加锁单样本。
B1 明显长尾和 B256 数字只证明新边界可用，不作为性能收益或稳定基线。

## 2026-08-03：S6.23 补齐跨核执行包 Atomic/DCCI 泳道

本阶段只完善观察边界，不改变 cross-core 调度、内存发布或执行语义。此前
`WinnerBuild` 与 `EfDrain` 已能显示业务父区间和 kernel，但跨核 execution
control 内部的大量 Atomic/DCCI 仍直接调用 `Ops`，导致长区间看起来像普通
Scalar 代码。现在通过同一个 `SharedExecTraceObserver` 把生产路径已有操作各
记录一次，覆盖以下协议边界：

- execution fatal 的读取与首错发布；
- execution cell state 读取、Build reserve、BUILT 发布和 Execute claim；
- completion vend/flag 发布、DONE 发布；
- FinalDrain 的 arrive、release 发布和 release poll；
- Build 源 descriptor invalidate、payload flush/invalidate、token 引用
  descriptor invalidate。

返回值未参与协议判断的 completion vend Exchange 保持
`source_issue`；其余实际消费返回值的调用保持 `return_ready`。observer 在
Atomic 结束计时后才写 trace record，且没有额外发射 Atomic 或 DCCI。测试专用
直连入口继续存在，但必须带精确豁免标记；源码审计同时要求 CCEC intrinsic
只能集中在 `ccec_ops.h`，防止后来新增一条绕过 observer 的设备原语。

`WinnerBuild` 中剩余最大的连续普通 Scalar 区间是
`PreloadBuildDestination + PackExecPayload`。正常成功 Build 的 reserve CAS 结束
到 payload-flush DCCI 开始恰好包围 payload 预取和打包。converter 据此离线
生成 `winner_build.pack_execution_payload`，错误路径缺少任一边界时不猜测。
该做法不依赖保留多余 fatal load，也不增加设备 raw 行；本次 B256 只在
merged 中增加 1024 条派生事件。

### A5 B256 实测闭合

经用户确认可直接使用 device 0 后，使用用户 `.venv`、本机 CANN 9.1 和
real-compute `6,28,4,1` 运行 B256。1280 个 task、1024 个 kernel、payload、
依赖、执行终态与全部 trace 计数检查 PASS，`dropped_records=0`。产物为：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260803_153817_181367/ccec/merged_swimlane.json`

旧泳道 `efdrain#558` 为 `409.005 us`，其中两个 QK kernel 合计
`83.238 us`，此前约 `325.4 us` 没有细化。新泳道中形态最接近的
`efdrain#518` 为 `412.563 us`，总长和 kernel 时间分别只差
`+0.87%/+0.75%`，可作为同型解释：

| 组成 | 时间 | 父区间占比 |
|---|---:|---:|
| Atomic | 304.410 us | 73.79% |
| 两个 QK kernel | 83.864 us | 20.33% |
| DCCI | 0.308 us | 0.07% |
| 其余普通 Scalar | 23.981 us | 5.81% |

其中仅 29 次 `shared_exec_fatal_load` 就占 `294.998 us`，单次范围
`8.511--12.487 us`。这证明旧 `409 us` 不是单次 task execute 变成数百微秒，
主要是 96 核反复读取同一 `exec_fatal` 地址时的返回型 Atomic 竞争。DCCI 在该
区间只有三次 payload invalidate，不是主因。

对全部含 kernel 的 281 个 EfDrain 做同一口径聚合，Atomic、kernel、DCCI 和
其余普通 Scalar 分别占 `68.89%`、`22.49%`、`0.03%`、`8.59%`，已解释
`91.41%`。1024 个 WinnerBuild 的总 core-time 为 `60,335.301 us`，其中
Atomic 占 `83.73%`、DCCI 占 `2.58%`、其余普通 Scalar 占 `13.69%`。
离线派生的 payload 打包区间共 1024 条，单次 `2.457--17.361 us`、均值
`5.005 us`；最长 `winner_build#1279` 中该区间为 `13.457 us`。长父区间中已
不再存在一整段数百微秒、却无法判断是 Atomic、DCCI、kernel 还是普通 Scalar
的空白。

### 验证结果

- standalone CPU 全量协议、adapter、scan/drain 和 ordered-submit：PASS；
- CCEC AIC/AIV、split runtime/finish、mixed ELF 与 manifest：PASS；
- converter：`63 passed`；
- same-core 与 cross-core Atomic/DCCI 源码审计：各 `5 passed`；
- A5 B256：功能、执行与观察计数全部 PASS。

本次 host Submit 为 `2.508790 ms`、完整生命周期为 `6.046633 ms`，但它是
full-swimlane 单次诊断数据，只用于解释区间，不与 trace-free 性能基线相减。

## 2026-08-04：S6.24 取消 WinnerBuild 的跨 task 伪保序

### 顺序合同复核

代码与业务依赖重新逐项核对后确认，shared TensorMap 只有 Register 元数据
插入必须保持 task-id 顺序。task N 发布插入完成字后，N+1 可以进入自己的
Register；N 的 Fanin、WinnerBuild 和 Execute 不再占用这条有序链。

WinnerBuild 只需要保持单 task 内部的 payload 发布顺序：

```text
EMPTY -> BUILDING CAS
-> Pack payload
-> payload DCCI clean-out + DSB
-> BUILDING -> BUILT CAS
```

因此，原本围绕同一个 Build 重复读取 scheduler fatal 和 execution fatal 的
操作不是保序协议，只是为了让并发错误更早终止。新的性能优先合同允许已经
开始的合法 Build 完成 BUILT；executor Claim 与 FinalDrain 仍能观察错误并
最终退出。

正常成功 WinnerBuild 的 terminal 读取由七次缩减为两次：入口一次 scheduler
fatal、helper 入口一次 execution fatal；reserve/Built 两次 CAS 和 payload
DCCI 不变。converter 的 `winner_build.pack_execution_payload` 改由
`build_reserve CAS.end -> payload_flush DCCI.start` 离线推导，不再依赖为了
观测而保留的 fatal load，也没有增加 raw 记录。

### 正确性和 A5 结果

CPU adapter 增加 flush 后并发 fatal 的定向交错，证明 Build 仍可发布 BUILT，
后续路径能观察并退出；完整 CPU、CCEC AIC/AIV、split runtime/finish、mixed
ELF 和源码覆盖门槛均通过。

对应 B256 full-swimlane：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260803_161231_209267/ccec/merged_swimlane.json`

相对 S6.23：

| 指标 | S6.23 | S6.24 | 变化 |
|---|---:|---:|---:|
| WinnerBuild 聚合 core-time | 60,335.301 us | 17,115.987 us | -71.63% |
| WinnerBuild Atomic | 50,518.268 us | 8,529.341 us | -83.12% |
| full-swimlane 生命周期 | 6.046633 ms | 5.123835 ms | -15.26% |

同一版本的 trace-free B256 共十个独立进程，完整边界均为首个 Submit 起点到
FinalDrain 结束，10/10 execution、semantic、postprocess PASS：

```text
min / median / max = 4.856928 / 4.992764 / 5.199626 ms
```

## 2026-08-04：S6.25 将 exec_fatal 收敛为只写首错原因

S6.24 的新泳道进一步显示，`shared_exec_fatal_load` 全局仍有 28,426 次，
聚合 `280,067.777 us`，其中 FinalDrain 有 20,001 次。该 control 保存
execution 协议的精确错误原因；生产错误路径在 helper 返回失败后还会同步
设置权威 scheduler fatal，因此它没有必要再充当第二条停止线。

本阶段保留所有首错发布和 host 原因校验，删除成功路径中以下原因读取：

- Build helper 入口；
- Claim 前、CAS 前和 Claim 后；
- fanin ready 前后与 engine-complete 后；
- vend、flag、DONE 发布前；
- scanner、FinalDrain 和 Claim 失败后的重复原因复核。

测试明确锁定新的职责边界：只写 `exec_fatal`、但没有像生产错误路径一样写
scheduler fatal 时，原因记录本身不阻止一个合法 Claim。FinalDrain 的退出仍
由 scheduler fatal 负责。完整 CPU 回归、CCEC 两种构建和 A5 B256 全部通过。

对应 full-swimlane：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260803_162909_227171/ccec/merged_swimlane.json`

动态结果：

- `shared_exec_fatal_load`：`28,426 -> 0`；
- full-swimlane Submit：`1.773787 ms`；
- full-swimlane FinalDrain：`2.843175 ms`；
- full-swimlane 完整生命周期：`4.329033 ms`；
- 1280 task、1024 kernel、payload、依赖、终态和 raw 计数全部 PASS。

trace-free 十个独立进程为：

```text
min / median / max = 3.985452 / 4.052038 / 4.219944 ms
S6.24 median -> S6.25 median = -0.940726 ms / -18.842%
```

## 2026-08-04：S6.26 将 global fatal 集中到 replay 调度边界

继续审视 scheduler fatal 后发现，成功路径仍有 7,190 次逐条返回型读取：
WinnerBuild、Claim 前后、kernel 前后、completion 前以及外层 Build 循环都在
重复读取同一 cache line。它们同样不是 TensorMap、Build 或 Execute 的顺序
边界。

本阶段采用“合法工作单元完整完成、下一调度边界停止”的合同：

1. 每次准备领取新 Build ticket 前保留一次权威 scheduler fatal 检查；
2. 已取得的合法 Build、Claim、kernel 和 completion 允许完成；
3. 本核实际发现 payload/control/发布错误时仍立即写精确原因、设置 scheduler
   fatal 并返回失败；
4. 其他核在下一次 ticket 调度边界或 FinalDrain 最终观察并退出。

CPU 定向测试不再维护“fatal 恰好插在 Claim/kernel/completion 中间时必须留下
半完成 task”的旧合同，改为证明：blocked token 在 FinalDrain 会转 Faulted；
已经开始的合法 task 即使与并发 fatal 相遇，也会完整发布 vend、flag 和 DONE，
不会遗留 CLAIMED 半状态。完整 CPU 回归通过。

源代码收敛改变了 CCEC 的等价尾合并形状。实际 object 证明 full-swimlane
AIC/AIV finish relocation 为 `3/4`，perf-clock 为 `3/3`；全部仍只指向本角色
唯一 finish/state，五类 task 和 1280 次唯一 Build 继续由动态门槛证明。构建
脚本据此更新精确冻结值，没有放宽为范围判断。

对应 full-swimlane：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260803_164608_247533/ccec/merged_swimlane.json`

动态结果：

- 逐条 `fatal_poll`：`7,190 -> 1,472`；
- full-swimlane Submit：`1.142582 ms`；
- full-swimlane FinalDrain：`1.971008 ms`；
- full-swimlane 完整生命周期：`2.990625 ms`；
- 所有业务与观察门槛 PASS。

trace-free 十个独立进程为：

```text
min / median / max = 2.689313 / 2.822871 / 2.945247 ms
S6.25 median -> S6.26 median = -1.229167 ms / -30.334%
S6.24 median -> S6.26 median = -2.169893 ms / -43.462%
```

该阶段原始泳道仍位于上述 `outputs` 路径；`test_record` 只保留后续最新最优
版本，不重复保存被取代的 `2.99 ms` 副本。

## 2026-08-04：S6.27 FinalDrain fatal 低频最终观察

S6.26 后，FinalDrain 仍在每次循环读取 scheduler fatal；全局 replay release
之后，`ProgressCrossCoreExec` 入口还会重复读取一次。当前候选删除 progress
入口读取，并在 FinalDrain 外层使用 owner-local 计数器：第 0 轮立即检查，
此后每 256 轮检查一次。命中后把本核两个 token 转为 Faulted，继续参与 replay
barrier，随后退出；本核直接发现协议错误仍立即发布 fatal。

### 正确性与结构门槛

CPU 定向用例改为直接验证 FinalDrain 观察点：scheduler fatal 不伪造
`exec_fatal`；blocked token 被收敛为 `FAULTED`；已经开始的合法 completion
仍完整发布。定向用例和完整 CPU 回归全部 PASS。

CCEC full-swimlane AIC/AIV 仍为 `3/4` 个同角色 finish relocation。trace-free
AIC 的等价尾部由 `3` 个合并为 `1` 个，AIV 保持 `3` 个；对象仍只导入本角色
唯一 finish/state，最终 mixed ELF 无残留 relocation。构建门槛据实冻结为
`1/3`，没有放宽成范围。

### A5 B256 full-swimlane

产物为：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260803_170039_262883/ccec/merged_swimlane.json`

动态结果：

- Submit：`1.081578 ms`；
- FinalDrain：`1.723799 ms`；
- 完整生命周期：`2.685995 ms`；
- EfDrain/FinalDrain kernel：`606/418`；
- FinalDrain 批量 `fatal_poll` 逻辑调用：`11,139 -> 125`；
- 全部 `fatal_poll` 逻辑调用：`12,611 -> 1,597`，下降 `87.336%`；
- 1280 task、1024 kernel、execution/semantic/postprocess 与 raw 计数全部 PASS。

该阶段当时归档的泳道副本为：

`tests/atomic_probe/pa_scheduler/test_record/2026-8-4/cross_b256_s627_2p686ms.json`

trace-free 十个独立进程全部 PASS：

```text
min / median / max = 2.420601 / 2.519461 / 2.691250 ms
mean = 2.519032 ms
S6.26 median -> S6.27 median = -0.303410 ms / -10.748%
```

该结果同时满足正确性与性能门槛，因此 S6.27 作为有效阶段保留。它只改变
terminal 的最终观察频率，不改变实际错误发布、本地错误立即退出、TensorMap
插入顺序、Build/Claim/completion 发布或 FinalDrain 全局收口条件。

## 2026-08-04：S6.28 删除 WinnerBody fatal guard（无收益，已撤回）

S6.27 泳道中 `shared_winner_fatal_guard_load` 共 1280 次，聚合约
`1.366 ms`。它位于 Build ticket 已经取得之后，不参与 TensorMap 插入、
payload 发布或 completion 的顺序判定；从 S6.26 的“合法工作单元完成、下一
调度边界停产”合同看，可以复用 ticket 领取前的 global-fatal 观察。

候选据此只删除 `FinishSharedWinnerSubmitBody()` 中这一条返回型读取，并增加
源码门槛防止它被重新放回 cross-core winner。完整 CPU、CCEC 和 A5 B256
full-swimlane 全部 PASS；泳道中该 site 确实由 `1280 -> 0`。候选单次完整
生命周期为 `2.693160 ms`，与 S6.27 的 `2.685995 ms` 基本持平。

首轮十个独立 trace-free 进程的候选中位为 `2.485622 ms`，低于先前 S6.27
十轮的 `2.519461 ms`，但候选均值为 `2.524567 ms`，反而略高于基线均值
`2.519032 ms`。由于结果受长尾影响，随后用提交 `7efdcc3e` 重新构建冻结基线，
按 B-C/C-B 交错运行各六次：

```text
S6.27 baseline : min / median / max / mean
                 2.422939 / 2.474204 / 2.495793 / 2.468115 ms
S6.28 candidate: min / median / max / mean
                 2.433765 / 2.479890 / 2.654143 / 2.502150 ms
candidate median 回退 0.005687 ms / 0.230%
```

因此，泳道聚合 core-time 不能直接换算为端到端收益：这些 guard 并未形成同样
大小的全局关键路径，删除后还会改变代码布局和并发相位。S6.28 的代码与新增
门槛已完整撤回，只保留本节取证记录；后续不再以“调用次数减少”代替冻结 A/B
性能裁决。

## 2026-08-04：S6.29 复用执行扫描快照完成 Claim CAS

S6.27 的最新泳道显示，`shared_exec_cell_state_load` 有 `5,054` 次逻辑调用，
聚合约 `1,351.052 us`。源码核对发现，K2 scanner 已经返回了一次完整 packed
control；当它判断 phase 为 `BUILT` 后，`ClaimAndBindExecPayload()` 又在同一
control 上执行一次返回型 load，随后才做 `BUILT -> CLAIMED` CAS。第二次 load
既不发布数据，也不是所有权线性化点。

本阶段新增接收 `observed_raw` 的 Claim 入口。scanner 将自己刚读取的完整
control 快照直接作为 CAS expected；CAS 仍是唯一执行所有权裁决点。若快照在
扫描与 CAS 之间过期，CAS 必然失败并返回新状态，原有 `Lost/NotBuilt` 路径会
保留候选并重新观察，因此不会凭旧快照错误取得执行权。原有无上游快照的公共
helper 继续保留，并且仍只在本地 token/owner 参数检查通过后读取一次 control，
没有改变定向测试依赖的“本地入口错误不产生 shared operation”合同。

先将 BUILT 正常路径的精确 control-load 期望从 `3` 改为 `2`，旧代码按预期
失败；实现后该门槛通过。完整 CPU 协议回归、CCEC full-swimlane/perf-clock
构建以及 A5 B256 full-swimlane 全部 PASS，1280 个 Build、1024 个 kernel、
TensorMap 严格插入、payload、fanin、completion 和终态均未改变。

本阶段 full-swimlane 为：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260803_180902_328262/ccec/merged_swimlane.json`

归档副本为：

`tests/atomic_probe/pa_scheduler/test_record/2026-8-4/cross_b256_s629_2p690ms.json`

动态结果：

- 完整生命周期：`2.689998 ms`；
- Submit：`1.141835 ms`；
- FinalDrain：`1.692159 ms`；
- `shared_exec_cell_state_load`：`5,054 -> 3,991`，减少 `1,063` 次，
  即 `21.033%`；
- `shared_exec_claim` 仍保留 `1,024` 次必要 CAS；
- 1280 task、1024 kernel 和全部后处理检查 PASS。

为避免把设备波动或泳道代码布局误判成收益，另用提交 `4431dbfd` 构建冻结
基线，与候选按 B-C/C-B 交错各运行六个独立 trace-free 进程：

```text
S6.27 frozen baseline: min / median / max / mean
                       2.424855 / 2.507471 / 2.615748 / 2.509293 ms
S6.29 candidate      : min / median / max / mean
                       2.345950 / 2.425596 / 2.540639 / 2.428607 ms
candidate median 改善 0.081875 ms / 3.265%
candidate mean   改善 0.080686 ms / 3.215%
```

该候选同时减少了已定位的非必要返回型 Atomic，并在冻结交错 A/B 中得到稳定
端到端收益，因此作为有效阶段保留。它不改变 Build/Execute 候选拓扑、
TensorMap 插入顺序、payload 发布、fanin 判断或 completion 顺序。

## 2026-08-04：S6.30 非零输出 heap vend 前置读取（无收益，已撤回）

S6.29 泳道中 `shared_heap_vend_load` 有 1280 次，其中 1024 个非零输出 task
随后还会执行 `shared_heap_vend_advance.fetch_add`，并消费 FetchAdd 返回的旧
vend 做对齐、容量和 `int64_t` 边界校验。候选据此尝试让非零输出直接消费
FetchAdd 返回值，仅为 256 个零输出 UP 保留 vend load。

该改动需要同时调整冷失败合同：合法配置已经由 host 在 worker 启动前按 task
plan 完成逐 shard 与总量准入；但若设备 control 已损坏，非零 reserve 只能在
RMW 返回后发现异常，因此不再满足“失败时没有任何共享写入”，而是保留 cursor
和 vend 的 terminal RMW 现场。CPU 定向测试先把正常非零 reservation 的原子
记录从 4 次收紧为 3 次，并精确验证四类损坏 vend 的失败现场。完整 CPU 协议
回归和 CCEC full-swimlane/perf-clock 构建均通过。

A5 B256 full-swimlane 为：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260803_182840_345595/ccec/merged_swimlane.json`

动态取证符合源码预期：

- `shared_heap_vend_load`：`1280 -> 256`，减少 `1024` 次，即 `80%`；
- `shared_heap_vend_advance`：仍为 `1024` 次；
- 完整生命周期单次 `2.621299 ms`；
- 1280 task、1024 kernel、heap、TensorMap、payload、fanin、completion 和
  终态全部 PASS。

候选十个独立 trace-free 进程全部 PASS：

```text
min / median / max / mean
2.334857 / 2.383275 / 2.644366 / 2.406163 ms
```

单组绝对数不足以裁决，随后以提交 `308f26c6` 构建冻结 S6.29 基线，按
B-C/C-B 交错各运行六个独立进程：

```text
S6.29 frozen baseline: min / median / max / mean
                       2.376348 / 2.386710 / 2.574386 / 2.416577 ms
S6.30 candidate      : min / median / max / mean
                       2.379444 / 2.409327 / 2.702979 / 2.469979 ms
candidate median 回退 0.022617 ms / 0.948%
candidate mean   回退 0.053402 ms / 2.210%
```

因此，“返回型 Atomic 次数减少”在这里没有转化为端到端收益，反而削弱了冷
失败的无写入合同。候选代码和测试修改已完整撤回，只保留本节反例；后续不再
以删除 heap vend 前置读取作为优化方向。

## 2026-08-04：S6.31 发布 Exchange 改为只记发射边界（无稳定收益，已撤回）

S6.29 泳道中 `shared_output_published_exchange` 有 2048 次。它在
descriptor 拷贝、DCCI 和 StoreBarrier 之后发布 `published=task_id`，
并消费 Exchange 返回的旧值检查是否为 `-1`。候选方案利用
`SharedExecCell EMPTY -> BUILDING` 已确立唯一 Build owner、每个 output
的 `last_writer` FetchMax 已完成单次预留这两个条件，尝试不再消费
published Exchange 的旧值；消费者仍通过 atomic poll 等待对应
`task_id`。

定向测试先冻结该 site 为 `source_issue`，旧实现按预期失败。候选
实现后，完整 CPU 协议回归、CCEC full-swimlane/perf-clock 构建和
A5 B256 full-swimlane 均 PASS，1280 task、1024 kernel、TensorMap 严格
插入、payload、fanin、completion 和终态均未变。候选 raw 为：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260803_184300_359040/ccec/l2_swimlane_records.json`

修正候选 converter 的站点语义后，2048 次 Exchange 全部显示为
`atomic.source_issue.shared_output_published_exchange.exchange`，不再误标为
`return_ready`。该次完整生命周期为 `2.641440 ms`，Submit 为
`1.078390 ms`，FinalDrain 为 `1.677379 ms`。

随后以提交 `2d0e6c95` 构建冻结 S6.29 基线，按 B-C/C-B 交错各
运行六个独立 trace-free 进程：

```text
S6.29 frozen baseline: min / median / max / mean
                       2.362611 / 2.396578 / 2.494395 / 2.408258 ms
S6.31 candidate      : min / median / max / mean
                       2.330193 / 2.405667 / 2.488242 / 2.404911 ms
candidate median 回退 0.009089 ms / 0.379%
candidate mean   改善 0.003347 ms / 0.139%
```

中位数和均值方向相反，且差异明显小于当前运行波动，无法证明
取消返回等待带来稳定端到端收益。同时，它会去掉生产者侧
对“在 last_writer 预留与 published Exchange 之间发生非法并发写”
的异常检测与回滚。因此候选代码、测试和 converter 口径已全部
撤回，保留原有 `return_ready` 协议，本节只记录反例证据。

## 2026-08-04：S6.32 拆分 execution drain 到达与释放 cache line

S6.29 的 execution drain 使用 96 核同地址 `arrived` FetchAdd 汇合每核
token 排空证据，已到达 worker 随后持续 atomic poll `release`。源码
注释声称 arrival/release 已分行，但真实 `SharedExecDrainControl` 只有
64B，`release` 位于偏移 8；两种访问实际争用同一 cache line。

本阶段先将编译期门槛收紧为 `sizeof == 128` 且 `release offset == 64`，
旧实现按预期以 `64 != 128` 和 `8 != 64` 失败。随后只在两个控制字
之间增加精确 padding，使 `arrived` 与 `release` 各自独占 64B atomic-only
行。cross-core execution tail 的精确搬运大小相应由 `20,182,272`
增加到 `20,182,336` 字节，仍是 `exec_fatal` 到 `SchedulerState` 尾部的
一段连续范围。

该改动不改变原子次数和协议：每核仍只 arrival 一次，最后到达者
仍逐 task 验证 Alloc `EMPTY` / kernel `DONE`，然后唯一发布 release；
其他核仍等待 release 后退出。完整 CPU 协议回归、CPU B256、CCEC
full-swimlane/perf-clock 构建以及 A5 B256 full-swimlane 全部 PASS。

A5 full-swimlane 为：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260803_190217_376268/ccec/merged_swimlane.json`

动态结果：

- 完整生命周期：`2.669770 ms`；
- Submit：`1.119853 ms`；
- FinalDrain：`1.662478 ms`；
- 96 次 arrival FetchAdd 泳道聚合：`731.706 -> 726.548 us`；
- release poll 逻辑调用：`8,127 -> 7,728`；
- 1280 task、1024 kernel、TensorMap、payload、fanin、completion 和终态全部 PASS。

两份泳道不是同一 ELF 的多轮性能样本，上述局部值只用于证明 atomic
数量与流程符合预期，不直接声称端到端收益。最终以提交
`4fdfe905` 构建冻结 S6.29 基线，按 B-C/C-B 交错各运行六个独立
trace-free 进程：

```text
S6.29 frozen baseline: min / median / max / mean
                       2.347100 / 2.433064 / 2.662387 / 2.453087 ms
S6.32 candidate      : min / median / max / mean
                       2.310743 / 2.354961 / 2.506264 / 2.378929 ms
candidate median 改善 0.078103 ms / 3.210%
candidate mean   改善 0.074158 ms / 3.023%
```

中位数和均值同向，且改善幅度远大于一条指令级删减，说明收益主要
来自不再让已到达核的 release poll 与未到达核的 arrival RMW 发生
同行干扰，而不是单条 arrival FetchAdd 本身大幅变快。该阶段不改
TensorMap 严格插入、Build/Execute owner、fanin、completion 或终态收口合同，
因此作为有效优化保留。

## 2026-08-04：S6.33 分片 execution drain arrival

S6.32 已把 arrival 与 release 拆到不同 cache line，但 96 个 worker 在
execution token 排空后仍对同一个 arrival 字执行返回型 FetchAdd。A5 对同地址
并发返回型 Atomic 的延迟会随竞争人口显著增长；上一阶段泳道中仅 96 次
arrival FetchAdd 的聚合核时仍为 `726.548 us`，说明“消除同行干扰”之后，
同地址 RMW 本身已成为下一处明确开销。

本阶段没有减少必须到达的 worker，也没有把 FinalDrain 终态检查交给 host。
实现将 execution drain 固定划为 16 个 arrival group：

```text
worker 完成本核 scanner/token 排空
-> group = block_id % 16
-> 对 arrivals[group] 做且只做一次 FetchAdd
-> 普通 worker 只轮询独立 release
-> 固定 root 轮询 16 个 group count
-> 每组精确等于 6
-> root 全量校验 Alloc EMPTY / kernel DONE
-> root 唯一 CAS 发布 release
```

当前 32 block、每 block `1 AIC + 2 AIV`，所以每个 group 正好覆盖两个
block、六个 Scalar。同地址返回型 FetchAdd 的最大并发人口由 96 降为 6；
代价是 root 增加 16 条分组计数的累计轮询。arrival、release 以及每个分组
计数仍各自独占 64B atomic-only cache line，所有 worker 全到达、所有 task
终态闭合和唯一 release 的正确性合同保持不变。

实现前先把布局门槛改为 `16 * 64B arrival + 64B release = 1088B`，并要求
release 偏移为 `1024B`；S6.32 旧实现按预期以 `128 != 1088`、`64 != 1024`
失败。实现后补齐了以下门槛：

- CPU worker 初始化严格复用生产拓扑的 block/lane/sub-block 映射；
- 定向用例证明 16 组各到达 6 次、root 在全组到达前不发布、全组到达后
  完成唯一 release，重复推进不会重复 arrival；
- host 从设备快照逐组核对 `arrival == 6`，不只检查 release；
- 新增 `shared_exec_drain_arrival_poll` AtomicSite 和 PollBatch 映射，避免把
  root 的 group load 混入所有 worker 的 release poll；
- execution state 精确搬运字节数由 `20,182,336` 更新为 `20,183,296`；
- full-swimlane AIC 对象经 `readelf` 求证为 4 个 split-finish relocation，
  且全部指向本角色唯一 finish 符号，构建门槛据此精确更新为 4，没有放宽
  成范围判断。

完整 CPU 协议回归、CPU B256、converter 63 项测试、CCEC
full-swimlane/perf-clock 构建以及 A5 B256 full-swimlane 全部 PASS。A5
full-swimlane 为：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260803_192526_397391/ccec/merged_swimlane.json`

归档副本为：

`test_record/2026-8-4/cross_b256_s633_2p725ms.json`

该次动态结果为：

- 完整生命周期：`2.725357 ms`；
- Submit：`1.134504 ms`；
- FinalDrain：`1.744753 ms`；
- arrival FetchAdd：96 次、聚合 `726.548 -> 65.585 us`，下降 `90.97%`；
- arrival FetchAdd 最大值：`18.208 -> 5.314 us`；
- root arrival poll：43 次逻辑 load；
- 1280 task、1024 kernel、TensorMap、payload、fanin、completion 和终态全部
  PASS。

单次 full-swimlane 受诊断布局与设备波动影响，不用于裁决。最终以提交
`74c97937` 构建冻结 S6.32 基线，与候选按 B-C/C-B 交错各运行六个独立
trace-free B256 进程：

```text
S6.32 frozen baseline: min / median / max / mean
                       2.342994 / 2.553943 / 2.752440 / 2.551345 ms
S6.33 candidate      : min / median / max / mean
                       2.388638 / 2.427673 / 2.513362 / 2.435302 ms
candidate median 改善 0.126270 ms / 4.944%
candidate mean   改善 0.116043 ms / 4.548%
```

中位数与均值同向，且候选六个样本的长尾明显低于基线。结合泳道中 arrival
RMW 聚合核时下降 `90.97%`，可以把端到端收益归因到竞争人口缩减，而不是
减少 worker、跳过终态检查或把工作移出计时边界。该阶段作为有效优化保留。

## 2026-08-04：S6.34 删除 per-task winner fatal guard（无稳定收益，已撤回）

S6.33 全泳道中 `SharedWinnerFatalGuardLoad` 恰好出现 1280 次，聚合核时
`1382.670 us`。每个 dispatched winner 在进入 Materialize 前读取一次全局
scheduler fatal；同一 dispatch 入口在领取 Build ticket 前已经通过
`IsFatal()` 读取该控制字，而 execution payload 路径也采用“调度边界统一
观察、已经取得的合法工作单元允许完成”的合同。因此本阶段验证：删除 winner
内的第二次读取，是否能在不改变正常业务语义的前提下获得端到端收益。

先在 host 原子闭合门槛中要求该 site 为 0，保留旧 device 代码运行 B1，结果
按预期为：业务 execution/semantic 全部 PASS，但后处理精确报告
`winner_fatal_guards=5/0` 并失败。候选随后删除第二次 load；下一 ticket 边界、
插入等待中的低频 fatal 检查和 FinalDrain 最终观察均保持不变。完整 CPU 协议
回归、converter 63 项测试、CCEC full-swimlane/perf-clock 构建和 A5 B256
full-swimlane 均 PASS，候选泳道中该 site 由 `1280 -> 0`，其余 1280 task、
1024 kernel、TensorMap、payload、fanin、completion 和终态均闭合。候选泳道为：

`outputs/pa_scheduler_cross_core_shared_swimlane_s634_20260803_194554_416031/ccec/merged_swimlane.json`

该次完整生命周期为 `2.662796 ms`，Submit 为 `1.100522 ms`，FinalDrain 为
`1.669444 ms`。单次诊断结果不用于裁决。首次以提交 `9dea7d83` 构建冻结
S6.33 基线，交错各跑六个 trace-free B256 进程时，候选中位改善 `1.307%`、
均值改善 `1.050%`；由于幅度接近设备波动，继续反向顺序补足到每版 12 个
独立进程。合并结果为：

```text
S6.33 frozen baseline: min / median / max / mean
                       2.374750 / 2.421990 / 2.458060 / 2.422280 ms
S6.34 candidate      : min / median / max / mean
                       2.360702 / 2.425896 / 2.591560 / 2.437115 ms
candidate median 回退 0.003906 ms / 0.161%
candidate mean   回退 0.014835 ms / 0.612%
```

扩样后中位数和均值同时转为回退，证明前六轮的约 1% 差异只是时间段波动。
该 load 虽然在源码上与 ticket 前检查重复，但删除它没有形成稳定性能收益，且会
扩大 fatal 已发布后继续 Materialize/Register 的窗口。因此代码、动态门槛和
构建产物改动均完整撤回，只保留本节反例，不把 Atomic 次数减少本身当作性能
优化成果。

## 2026-08-04：S6.35 分片 execution drain release

S6.33 已把 arrival RMW 拆成 16 组，但全部非 root worker 仍轮询同一条
release line。保留泳道中，`shared_exec_drain_release_poll` 有 8196 次逻辑
load，96 个等待 episode 的聚合跨度为 `129059.980 us`；root 的唯一 release
CAS 也因与 96 路读共享地址达到约 `16.244 us`。因此本阶段验证：保持同一份
全量终态校验，把 release 也按 arrival group 分片，是否能降低 A5 同地址并发。

实现前先把 `SharedExecDrainControl` 布局门槛由 1088B 收紧为 2048B，旧实现
按预期以 `1088 != 2048` 编译失败。候选协议为：

```text
每个 worker 排空本核 scanner/token
-> arrivals[block_id % 16] FetchAdd 一次
-> 非 root 只轮询 releases[block_id % 16]
-> root 轮询 16 个 arrival，要求每组精确为 6
-> root 全量校验 Alloc EMPTY / kernel DONE
-> root 向 16 条 release line 各发布一次 Exchange
-> 各组 worker 观察本组 release 后退出
```

root 身份固定，且本地 `released` 状态阻止重复推进再次发布，因此 16 次
Exchange 不消费旧值；消费者的 atomic poll 是发布完成可见性的权威边界。
这避免把 Exchange 强制改成返回型路径。host 最终逐组检查
`arrival == 6 && release == 1`，full-swimlane 还精确要求 release publish
事件为 16 次、操作为 source-issue Exchange。execution state 精确搬运大小
由 `20,183,296` 增加为 `20,184,256` 字节。

完整 CPU 协议回归、converter 63 项测试、CCEC full-swimlane/perf-clock
构建以及 A5 B256 full-swimlane 均 PASS。A5 full-swimlane 为：

`outputs/pa_scheduler_cross_core_shared_swimlane_s635_20260803_202002_449213/ccec/merged_swimlane.json`

归档副本为：

`test_record/2026-8-4/cross_b256_s635_2p437ms.json`

该次动态结果为：

- 完整生命周期：`2.436915 ms`；
- Submit：`1.099908 ms`；
- FinalDrain：`1.451253 ms`；
- release publish：16 次 source-issue Exchange，聚合 `12.465 us`；
- release poll：95 个 episode，聚合跨度 `99,839.692 us`，相对 S6.33 的
  `129,059.980 us` 下降 `22.64%`；
- release poll 逻辑 load 由约 8196 增至 41389，说明分片后单次轮询更快，
  在等待最慢 worker 的同一业务窗口内能推进更多次；该调用数不能单独当作
  回退；
- 1280 task、1024 kernel、TensorMap、payload、fanin、completion、逐组
  arrival/release 和全部终态 PASS。

最终以提交 `4efd9eb6` 构建冻结 S6.33 基线，与候选按 B-C/C-B 交错各运行
六个独立 trace-free B256 进程：

```text
S6.33 frozen baseline: min / median / max / mean
                       2.416024 / 2.453325 / 2.510049 / 2.456007 ms
S6.35 candidate      : min / median / max / mean
                       2.145582 / 2.168356 / 2.192012 / 2.167784 ms
candidate median 改善 0.284969 ms / 11.616%
candidate mean   改善 0.288223 ms / 11.735%
```

中位数、均值和全部六个候选样本都与基线清晰分离，改善远大于设备波动。
性能收益来自把 release 的同地址并发人口由 96 降为每地址 6；没有减少
worker、跳过终态校验、缩短计时边界或改变 TensorMap/Build/Execute 业务
语义。该阶段作为有效优化保留。

## 2026-08-04：S6.36 单向 execution drain 收口

S6.35 已把 release 轮询分到 16 条地址并获得明显收益，但协议仍要求 95 个
非 root worker 在本核完全排空后持续轮询，等待 root 完成全局校验再反向
放行。重新核对生命周期后，这层反向 release 并不是正确性必需条件：

1. replay barrier 已证明所有 Build ticket 耗尽，不会再出现新 BUILT；
2. 每核只有在 scanner 封口、两个 execution token 完全复位、engine 无
   in-flight 后才发布 arrival；
3. 非 root 此后没有新的本核工作，可以结束本核；
4. 固定 root 仍等待 16 组 arrival 各为 6，并全量校验 Alloc `EMPTY` 与
   kernel `DONE`；
5. 整个 device kernel 只有在 root 最后结束后才完成，因此全局终态证明
   没有转移给 host。

本阶段据此把 drain 改为单向协议。实现前先将布局门槛从 2048B 收紧为
`16 * 64B = 1024B`，S6.35 按预期以 `2048 != 1024` 编译失败。实现后删除
全部 release control、publish 和 poll；非 root 的本地 `closed` 在 arrival
成功后置位，root 的 `closed` 只在所有组到齐且全 task 校验成功后置位。
host 继续逐组检查 `arrival == 6`、逐核检查 `final_occupied == 0`，并在
full-swimlane 中精确要求 release publish 为 0。execution state 精确搬运
大小由 `20,184,256` 降为 `20,183,232` 字节。

单向分支改变了 CCEC 尾合并形状。`readelf` 逐条求证后，full-swimlane
AIC/AIV 的 finish relocation 精确为 `4/5`，perf-clock 精确为 `2/3`，
全部只指向各自角色唯一 finish 符号；构建门槛据实更新，没有改成范围。

完整 CPU 协议回归、converter 63 项测试、CCEC full-swimlane/perf-clock
构建以及 A5 B256 full-swimlane 均 PASS。A5 full-swimlane 为：

`outputs/pa_scheduler_cross_core_shared_swimlane_s636_20260803_203637_464219/ccec/merged_swimlane.json`

归档副本为：

`test_record/2026-8-4/cross_b256_s636_2p408ms.json`

该次动态结果为：

- 完整生命周期：`2.407780 ms`；
- Submit：`1.150132 ms`；
- FinalDrain：`1.404917 ms`；
- `shared_exec_drain_release_publish`：`16 -> 0`；
- `shared_exec_drain_release_poll`：全部消失；
- 16 组 arrival 各为 6，root 全量校验、1280 task、1024 kernel、TensorMap、
  payload、fanin、completion 和全部终态 PASS。

由于首轮 6+6 交错 A/B 只有约 1% 改善，继续按反向顺序扩展到每版 12 个
独立 trace-free B256 进程。最终以提交 `ad22756c` 构建的冻结 S6.35 基线
与候选结果为：

```text
S6.35 frozen baseline: min / median / max / mean
                       2.127455 / 2.162082 / 2.187897 / 2.157836 ms
S6.36 candidate      : min / median / max / mean
                       2.099396 / 2.139029 / 2.157356 / 2.135427 ms
candidate median 改善 0.023053 ms / 1.066%
candidate mean   改善 0.022409 ms / 1.038%
```

扩样后中位数和均值仍同向改善，没有出现 S6.34 的方向反转；候选最大值也低于
基线最大值。该阶段既删除了所有反向 release Atomic 和 1024B 无用控制状态，
又保留 device 内 root 的全局终态证明，因此作为有效优化保留。

## 2026-08-04：S6.37 取消 insert-turn handoff CAS 返回依赖（无收益，已撤回）

S6.36 泳道中，`SharedInsertTurnHandoff` 恰好出现 1280 次，聚合跨度约
`520.208 us`。现有实现由每个 task 唯一的中央 Build ticket owner 执行
`CAS(-1, task_id)`，并立即消费旧值以确认该 TaskCell 尚未发布。候选验证一个
更弱但仍可闭合的协议：保留条件 CAS，防止异常旧值被覆盖，但当前 owner 不再
等待和检查返回值；发布完成由 N+1 owner 对 `deps_prepared[N]` 的返回型 poll
观察，最后一个 task 则由 host 终态校验。

实现前先修改定向用例，要求当前 owner 只发射条件 CAS、异常值由下一有序 owner
拒绝；旧实现按预期仅该项失败。候选随后同步修改了 CPU/CCEC 公共路径、Atomic
site 的 `result_used/return_ready` 合同、泳道转换器和动态门槛。完整 CPU 协议
回归、converter 63 项测试、CCEC full-swimlane/perf-clock 构建以及 A5 B256
full-swimlane 均 PASS。候选泳道为：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260803_212408_507489/ccec/merged_swimlane.json`

该次动态结果为：

- 完整生命周期：`2.381547 ms`；
- Submit：`1.118154 ms`；
- 1280 条 handoff 记录全部为 `source_issue`，没有混入 `return_ready`；
- 1280 task、1024 kernel、TensorMap、payload、fanin、completion 和全部终态
  PASS。

单次诊断结果不用于裁决。最终以提交 `2fda4052` 构建冻结 S6.36 基线，与候选
按 B-C/C-B 交错各运行六个独立 trace-free B256 进程：

```text
S6.36 frozen baseline: min / median / max / mean
                       2.099381 / 2.126659 / 2.151769 / 2.125906 ms
S6.37 candidate      : min / median / max / mean
                       2.119215 / 2.127761 / 2.149570 / 2.130045 ms
candidate median 回退 0.001103 ms / 0.052%
candidate mean   回退 0.004140 ms / 0.195%
```

中位数与均值均未改善，候选 CCEC finish 函数反而发生代码膨胀：AIC
`35,252 -> 35,284 B`，AIV `36,532 -> 36,548 B`。这说明“不在 C++ 层消费
CAS 返回值”没有让当前 CCEC 生成更轻的热路径，也不能把泳道里的
`return_ready` 聚合跨度直接等价成可删除的端到端开销。候选还把同 task
异常从当前 owner 延迟到下一 owner/host 才报告，在没有收益时不值得放宽错误
闭合窗口。因此实现、测试和泳道 ABI 修改已全部撤回，只保留本节反例。

## 2026-08-04：S6.38 用分组完成数替代 root 逐 task 终态扫描

S6.36 已删除 execution drain 的反向 release，但固定 root 在 96 个 worker
全部到达后仍串行读取 1280 个 execution cell control 和 1280 个 completion
flag。保留泳道中，该收口路径有以下直接证据：

- root 的 cell-state 读取为 1324 次，聚合 `332.390 us`；
- fanin/completion flag 读取为 1359 次；
- FinalDrain 最大跨度为 `1315.149 us`；
- 其中绝大部分不是推进业务，而是在已经排空后重新逐项证明终态。

本阶段复用已有 16 个 drain arrival word，不增加 cache line，也不增加 Atomic
调用。每个 worker 在 scanner 封口、两个 token 完整复位且 engine 无
in-flight 后，汇总本核三个 placement 计数，并执行原有的唯一一次 FetchAdd：

```text
contribution = 1 + (local_completed << 8)
```

低 8 位累计到达 worker 数，每组固定为 6，不会向高位进位；高 56 位累计本组
已经完整发布 completion flag 与 cell `DONE` 的 kernel 数。root 只读取 16 个
分组 word，要求每组低位均为 6，且高位合计精确等于
`task_count - alloc_task_count`。

该汇总不是用统计量替代正确性：中央 Build ticket 给出完整 task 集；
`BUILT -> CLAIMED` CAS 保证每个 kernel 至多一个 Execute owner；placement
只在 vend、completion flag 和 cell `DONE` 都成功后递增；每个 worker 又只在
本地 scanner/token/engine 全部排空后到达。因此“计划完成总数全部出现”与
“每项至多完成一次”共同推出所有计划 kernel 恰好完成，重复完成不能掩盖
缺项。host 在 kernel 返回后仍逐 task 核验 owner、cell、flag 与 payload，
用于精确定位错误，但设备性能边界不再执行 2560 次逐 task 原子读取。

测试先于实现收紧：

- 原实现按预期未能让每组 arrival 携带 owner-local completion，定向用例失败；
- 新增“96 核全部到达但 B1 少一个 completion”反例，root 必须发布 fatal 并
  拒绝收口；
- 完整 CPU 协议回归与 converter 63 项回归全部 PASS；
- CCEC perf-clock 与 full-swimlane 构建 PASS。源码变化使 full-swimlane
  AIC/AIV finish relocation 精确为 `4/3`，perf-clock 保持 `2/3`；
  `readelf` 已确认全部 relocation 只指向各角色唯一 finish 符号，构建门槛
  据实更新，没有放宽为范围。

A5 B256 full-swimlane 为：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260803_215149_530746/ccec/merged_swimlane.json`

归档副本为：

`test_record/2026-8-4/cross_b256_s638_1p439ms.json`

该次动态结果为：

- 完整生命周期：`1.439066 ms`；
- Submit：`1.139314 ms`；
- FinalDrain：`0.442080 ms`；
- root cell-state 读取：`1324 -> 44`；
- root/fanin flag 相关逻辑读取：`1359 -> 78`；
- drain arrival root poll：`171 -> 70`；
- 1280 task、1024 kernel、TensorMap 严格插入、payload、fanin、completion、
  16 组到达与完成数、全部 host 终态均 PASS，泳道记录无丢失。

最终以提交 `0ad00700` 构建冻结 S6.36 基线，与候选按 B-C/C-B 交错各运行
六个独立 trace-free B256 进程：

```text
S6.36 frozen baseline: min / median / max / mean
                       2.084560 / 2.127268 / 2.159094 / 2.127638 ms
S6.38 candidate      : min / median / max / mean
                       1.404007 / 1.415611 / 1.433730 / 1.418414 ms
candidate median 改善 0.711657 ms / 33.454%
candidate mean   改善 0.709224 ms / 33.335%
```

六个候选样本与六个基线样本完全分离，中位数和均值同向改善。该阶段不改变
TensorMap 插入顺序、Build/Execute owner、任务依赖、完成发布顺序或计时边界；
只把已经存在的 owner-local 完成事实并入原有 arrival Atomic，删除 root 的
重复全表读取，因此作为有效优化保留。

## 2026-08-04：S6.39 两级终态 barrier 由 root 直接扇出 release（无稳定收益，已撤回）

S6.38 的两级 replay/final barrier 先由 16 个 leaf leader 向
`root_arrival` 发布到达；root 收齐后发布一条 `root_release`，再由
16 个 leaf leader 分别读取该返回型 Atomic，并转发本组
`leaf_release`。候选协议利用“root 已收齐所有 leaf arrival”这个现有证据，
删除 `root_release` 中间层，改为 root 直接发布 16 条 `leaf_release`。

定向 CPU 用例先收紧为“两级形态的 `root_release` 必须保持 0”，旧实现按预期
仅在两个终态 barrier 集成用例失败；候选实现后，完整 CPU 协议回归、
converter/analyzer 99 项回归、CCEC full-swimlane/perf-clock 构建以及
A5 B256 full-swimlane 均 PASS。诊断泳道为：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260803_221321_552522/ccec/merged_swimlane.json`

该次诊断结果为：

- 完整生命周期：`1.474642 ms`；
- Submit：`1.119619 ms`；
- final barrier：`0.382604 ms`；
- `replay_done` 轮询的逻辑调用数：`6345 -> 3449`；
- 1280 task、1024 kernel、TensorMap、payload、fanin、completion 与全部终态
  PASS，泳道记录无丢失。

单次诊断不用于裁决。最终以提交 `88830ca4` 构建冻结 S6.38 基线，
与候选按 B-C/C-B 交错各运行六个独立 trace-free B256 进程：

```text
S6.38 frozen baseline: min / median / max / mean
                       1.402857 / 1.431680 / 1.457888 / 1.431833 ms
S6.39 candidate      : min / median / max / mean
                       1.407996 / 1.427383 / 1.434691 / 1.423352 ms
candidate median 表面改善 0.004297 ms / 0.300%
candidate mean   表面改善 0.008481 ms / 0.592%
```

六对成对差值既有改善也有回退，中位数改善只有 0.3%，不足以证明稳定收益。
候选虽然减少了共享 `root_release` 的返回型读取，却把原本由 16 个 leaf leader
并行发布的 release 集中成 root 串行发布 16 次，净收益被抵消。因此实现和
构建门槛修改已全部撤回，仅保留本节反例；后续不再用“减少返回型读取数”
单一指标代替端到端裁决。

## 2026-08-04：S6.40 中央 Build ticket 按连续小段领取（破坏并行合同，已撤回）

S6.38 中央 Build 发放对 1280 个 task 各执行一次返回型 `FetchAdd(1)`，96 个
worker 还各执行一次越界领取后退场，共计 1376 次同地址 Atomic。候选尝试让
每个 worker 通过 `FetchAdd(4)` 一次预留 4 个连续 task，再用栈上局部游标逐个
消费；理论物理调用数可降为：

```text
ceil(1280 / 4) + 96 = 416
```

实现阶段补充了非整段尾部、越界退场、逐 task 恰好一次和 96 worker 并发领取
检查。独立 Build-dispatch 用例十轮全部 PASS，确认小段本身不会造成 task
遗漏或重复；其余 CPU 单元用例也通过。

但是，完整 ordered-submit 集成用例稳定拒绝了该协议：

- `release_before_build=FAIL`；
- `independent_kernel_overlap=FAIL`；
- 所有 task/kernel 最终仍能完成，fatal 也保持 0，因此失败不是普通计数错误，
  而是既定并行合同被破坏。

反例中，一个 worker 预留 task 4--7，并在 task 4 已完成 TensorMap 有序插入、
但尚未完成 Build 时被刻意暂停。由于 task 5--7 已被同一 worker 私有预留，其他
Scalar 无法接手；task 8 又必须等待 task 7 的 TensorMap 插入，结果原本只覆盖
单个 task 的暂停扩张成整个连续段的阻塞。相同原因也使本应与 task 4 Build
重叠的独立 kernel 无法及时执行。

这与 cross-core 的核心合同冲突：TensorMap metadata 必须按 task id 严格串行
插入，但插入 baton 交出后，Fanin/Build/Execute 不应继续受前一 task 的 owner
约束。小段领取虽然减少 Atomic 次数，却引入了额外的 owner-local 串行区，
因此不能作为通用优化保留。

本阶段在 CPU 集成门槛已经得到确定反例，未浪费时间执行 A5 性能测试。候选的
实现、host oracle 和临时测试适配均已完整撤回，只保留本节论证。后续若优化
中央 Build ticket，必须继续保持逐 task 可由任意可用 Scalar 独立取得，不能
用私有批量所有权换取 Atomic 数量下降。

## 2026-08-04：S6.41 省略 ordered Fanin 的 output-published 回读（无收益，已撤回）

S6.38 的正式 Build 在发布 task N 的 `deps_prepared[N]` 之后才执行 Fanin。
对任意合法引用 `P < N`，逐 task completion 链已经传递以下顺序：

```text
P descriptor 写入
-> descriptor DCCI clean-out + DSB
-> published[P,slot] 发布
-> P writer metadata 发布
-> deps_prepared[P] 发布
-> ...
-> N 取得并发布自己的 insert completion
-> N Fanin lookup
```

因此候选只在明确携带该前置条件的正式 ordered/latest-writer 实例中省略
`published[P,slot]` 回读；通用 helper 继续读取并拒绝未发布 producer，真正
决定依赖的 `last_writer` 返回型读取、history invalidate/解析、fanin 范围检查
和 execution completion flag 均保持不变。

实现前先增加定向门槛，旧实现因不存在可信实例而按预期编译失败；实现后证明：

- 通用实例仍恰好读取一次 published，并立即拒绝未发布 producer；
- 可信实例对 published 地址零次读取，但仍从 last_writer 得到精确 producer；
- 完整 CPU 协议回归、converter/analyzer 99 项回归以及 CCEC
  full-swimlane/perf-clock 构建全部 PASS；
- ordered-submit 的 `release_before_build` 和
  `independent_kernel_overlap` 两项并发门槛保持 PASS。

A5 B256 full-swimlane 位于：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260803_224932_583794/ccec/merged_swimlane.json`

该次动态结果为：

- 完整生命周期：`1.523871 ms`；
- Submit：`1.107103 ms`；
- `SharedFaninOutputPublishedLoad`：`2048 -> 0`；
- 保留的 `SharedFaninLastWriterLoad` 仍为 2048 次；
- 1280 task、1024 kernel、TensorMap、descriptor、history、fanin、payload、
  completion 和全部终态 PASS，泳道记录无丢失。

单次诊断不用于裁决。最终以提交 `88830ca4` 构建冻结 S6.38 基线，与候选按
B-C/C-B 交错各运行六个独立 trace-free B256 进程：

```text
S6.38 frozen baseline: min / median / max / mean
                       1.398461 / 1.423792 / 1.443002 / 1.421513 ms
S6.41 candidate      : min / median / max / mean
                       1.428045 / 1.438135 / 1.474755 / 1.442186 ms
candidate median 回退 0.014343 ms / 1.007%
candidate mean   回退 0.020673 ms / 1.454%
```

六对样本中五对回退。候选并非代码膨胀：perf-clock AIC/AIV finish `.text`
分别从 `42,704/43,992 B` 缩小为 `42,112/43,472 B`。现有证据只能确认删除
2048 次返回型读取改变了代码布局或并发相位，却没有带来端到端收益；不能在缺少
进一步证据时把回退武断归因于某一项。候选实现与测试适配已完整撤回，保留原有
publication 检查。本节再次说明：泳道中的 Atomic 聚合核时用于发现候选，不能
替代冻结无观察 A/B 的保留裁决。

## 2026-08-04：S6.42 用无回收合同删除 TensorMap head 原子读取

S6.38 的正式 PA 路径仍为每次 ordinary TensorMap lookup 读取
bucket `head` 和 `tail` 两个 control Atomic。但当前正式实例是
单轮不回收模式：

```text
host init: reclaim_upto = -1, bucket head = 0
device:    不调用 ordered reclaim，不推进 head
host end:  校验 reclaim_upto 仍为 -1
```

所以在这一个明确实例中，`head` 的动态读值恒为 0。候选给
`SharedLookupRegion/SharedLookupTensor/CollectSharedFanin` 增加默认为
false 的编译期无回收参数，只在正式 ordered PA 调用点显式选用。
这个实例：

- 以常量 0 作为 `head`，删除 `SharedMapLookupHeadLoad`；
- 保留会被前序插入更新的 `tail` 原子读取；
- 容量越界或 slot seq 双检失败时直接 fail-closed，不进入回收重读；
- 通用可回收实例保留原有 `head/tail` 混合快照、前缀回收和
  ABA 处理，没有用 PA 假设改写通用语义。

定向 CPU 用例证明无回收 lookup 返回精确 producer，对 bucket
head 零次 Load，对 tail 仍恰好一次 Load。完整 CPU 协议回归、
converter/analyzer 99 项回归、CCEC full-swimlane/perf-clock 构建和
A5 B256 full-swimlane 全部 PASS。诊断泳道为：

`outputs/pa_scheduler_cross_core_shared_swimlane_20260803_230133_596174/ccec/merged_swimlane.json`

该次诊断结果为：

- 完整生命周期：`1.452110 ms`；
- Submit：`1.113723 ms`；
- `SharedMapLookupHeadLoad`：`1280 -> 0`；
- `SharedMapLookupTailLoad`：保持 1280 次，聚合核时 `316.830 us`；
- 1280 task、1024 kernel、TensorMap 严格插入、payload、fanin、
  completion 和全部终态 PASS，泳道记录无丢失。

单次诊断不用于裁决。最终以提交 `88830ca4` 构建冻结 S6.38 基线，
与候选按 B-C/C-B 交错各运行 12 个独立 trace-free B256 进程：

```text
S6.38 frozen baseline: min / median / max / mean
                       1.405662 / 1.429825 / 1.470568 / 1.434233 ms
S6.42 candidate      : min / median / max / mean
                       1.390242 / 1.423283 / 1.461963 / 1.423091 ms
candidate median 改善 0.006543 ms / 0.458%
candidate mean   改善 0.011142 ms / 0.777%
```

12 对样本中候选有 10 对更快，中位数和均值同向改善。这是一项
小幅但可重复的收益；保留依据不是泳道聚合核时，而是正式无回收
不变量、完整正确性门槛和冻结无观察 A/B 三者同时成立。
未来若引入任何 device 侧回收或 ring 复用，调用点必须切回通用实例，
重新恢复 head 原子读取和 ABA 判定。

## 2026-08-04：S6.43 将无观察性能口径前移到 startup

启动屏障原子操作即将进入优化范围，但 S6.42 及之前的
perf-clock 第一个时钟读取在 startup 屏障结束后，它无法裁决
startup 候选的端到端收益。S6.43 因此只修正性能边界，不改
调度协议：

```text
旧：最早 first Submit begin -> 最后 FinalDrain end
新：最早 startup increment begin -> 最后 FinalDrain end
```

每核仍严格只读取两次专用性能时钟：startup increment 前一次、
FinalDrain 完全排空后一次。为了不增加 `WorkerResult` 字段，
perf-clock 构建复用 `submit_begin` 存放新起点；普通泳道与
submit-PMU 构建仍保持原 Submit 语义。host 输出同步改名为
`startup_to_final_drain_us`，避免用旧名称误导后续对比。

CPU B256 证明每核两次时钟读取、新端点顺序、完整业务协议与
终态全部 PASS；CCEC perf-clock 构建通过。A5 B256 新口径的
6 个独立进程为：

```text
S6.43 startup-to-FinalDrain: min / median / max / mean
                              1.434122 / 1.465285 / 1.485386 / 1.461014 ms
```

该数据是后续 startup 分组候选的新冻结基线。它比旧口径多包含
启动屏障，因此不能与 S6.42 的 `1.423 ms` 直接相减后宣称性能
回退，也不能再混用旧字段名做 A/B。

## 2026-08-04：S6.44 startup 固定 G=16 两级屏障（明显回退，已撤回）

S6.43 首次给出了 startup 起点到 FinalDrain 结束的无观察端到端口径。
本轮才在不混用 Submit-only 旧数据的前提下，测试将 96 核 flat startup
屏障替换为与 final `two-16` 同形、但独占状态的两级屏障。

候选的物理原子发布为：

```text
96 次 leaf arrival
+ 16 次 leaf leader -> root arrival
+ 1 次 root release
+ 16 次 leaf release
= 129 次 StartupIncrement
```

每个 leaf 由两个物理 block 的 AIC/AIV0/AIV1 组成，即 6 个 Scalar；
只有 16 个 AIC leader 竞争 root。这确实把 flat 同地址 96 核竞争
拆成了小组竞争和 leader 转发，但也新增了 33 次发布以及两级关键路径。

完整 CPU 协议用例全部 PASS。CCEC 后端因大函数布局变化，将
full-swimlane AIV 的等价 split-finish 尾部从 3 份拆成 4 份；修改前后
对象文件的 relocation 逐条对照证明，4 条均仅指向 AIV 本角色的
唯一 finish，没有跨角色调用。A5 B256 full-swimlane 进一步证明：

- 1280 task、1024 kernel、TensorMap 严格插入和所有终态全部 PASS；
- `StartupIncrement` 物理记录精确为 129；
- 本次诊断生命周期为 `1.556463 ms`，仅用于确认协议形状，不用于保留裁决。

最终以 S6.43 修改前产物为冻结基线，候选与基线按 B-C/C-B 顺序
交错各运行 12 个独立 A5 B256 无泳道进程：

```text
S6.43 flat startup   : min / median / max / mean
                       1.424651 / 1.446754 / 1.497455 / 1.452607 ms
S6.44 two-level-16   : min / median / max / mean
                       1.471507 / 1.496351 / 1.526530 / 1.495496 ms
candidate median 回退 0.049597 ms / 3.428%
candidate mean   回退 0.042889 ms / 2.953%
```

12 对样本中候选 0 对更快，结论不是普通波动。对当前 96 核 startup
来说，分组降低同地址并发的收益，不足以覆盖 leader 转发、多级
release 和新增原子发布的代价。候选的 device/host 状态、协议、构建门槛和
测试适配已全部撤回；正式路径继续使用 flat `started_count`。

## 2026-08-04：S6.45 用新端到端口径复审 root 直接扇出（仍不保留）

S6.39 在旧口径下曾有中位数 `0.300%`、均值 `0.592%` 的表面改善，
是全部回撤项中最接近可保留的候选之一。本轮在当前 S6.43/S6.42
底座上完整重建该协议：两级 final barrier 的 root 收齐 16 个 leaf
arrival 后，直接串行发布 16 条 leaf release，不再经过一条
`root_release` 和 16 个 leaf leader 的转发。三级 barrier 保持不变。

CPU 完整协议回归 PASS。CCEC 候选使 perf-clock AIC 的等价 split-finish
尾部从 2 份合并为 1 份，relocation 仍只指向 AIC 本角色唯一 finish。
最终按 B-C/C-B 顺序，对冻结当前基线与候选各运行 12 个独立 A5
B256 进程，口径统一为 startup 起点到 FinalDrain 结束：

```text
current baseline : min / median / max / mean
                   1.405892 / 1.450168 / 1.491611 / 1.453868 ms
direct fan-out   : min / median / max / mean
                   1.420853 / 1.456032 / 1.477817 / 1.455157 ms
candidate median 回退 0.005864 ms / 0.404%
candidate mean   回退 0.001289 ms / 0.089%
```

候选只有 5/12 对更快，中位数与均值均回退。这说明新口径并没有
改变 S6.39 的判断：root 减少了一层返回型读，却串行承担了 16 次
release 发布，两者互相抵消。候选代码、host 终态适配和精确机器码
门槛已再次全部撤回，继续保留现有 leader 并行转发。

## 2026-08-04：S6.46 用完整周期复审并恢复 winner 重复 fatal 读取消减

S6.34 曾删除 `FinishSharedWinnerSubmitBody()` 入口的
`SharedWinnerFatalGuardLoad`，但在当时底座上扩样后中位回退
`0.161%`，因此撤回。当前调度、fatal 收口、TensorMap 查询和
性能边界均已发生明确变化，本轮把它作为历史回撤项重新独立裁决，
不沿用旧数据直接翻案。

当前协议中，worker 在领取新 Build ticket 前已经读取权威
scheduler fatal。ticket 一旦成功领取，它就是必须闭合的合法工作单元；
本核在 Materialize、Register、Fanin 或 Build 中直接发现错误时仍立即
发布 fatal，其他核在下一个 ticket 调度边界或 FinalDrain 最终观察并
收口。因此 winner body 内对同一 global fatal cache line 的第二次
返回型读取不参与 TensorMap 严格插入、payload 发布或执行所有权
线性化，可以删除。代价是并发 fatal 发布后，已取得工作单元继续
闭合的窗口变大；这与当前已采用的“调度边界停产”合同一致，
而不是将错误吞掉或依赖 host 超时。

完整 CPU 协议回归和 CCEC perf-clock/full-swimlane 构建全部 PASS。
A5 B256 full-swimlane 结果为：

```text
outputs/pa_scheduler_cross_core_shared_swimlane_20260804_001737_669218/
ccec/merged_swimlane.json

startup -> FinalDrain      : 1.453484 ms
Submit                     : 1.056485 ms
SharedWinnerFatalGuardLoad : 1280 -> 0
```

同次上板中 1280 task、1024 kernel、TensorMap 严格插入、payload、
fanin、vend、completion、DCCI 闭合和所有终态全部 PASS，泳道记录无丢失。

最终以冻结当前基线与候选按 B-C/C-B 顺序交错各运行 12 个独立
A5 B256 trace-free 进程，口径统一为 startup 最早起点到 FinalDrain
最晚结束：

```text
current baseline : min / median / max / mean
                   1.435604 / 1.467659 / 1.486151 / 1.463602 ms
candidate        : min / median / max / mean
                   1.415616 / 1.436587 / 1.469359 / 1.438105 ms
candidate median 改善 0.031072 ms / 2.117%
candidate mean   改善 0.025497 ms / 1.742%
```

12/12 对样本均由候选更快，中位数和均值同向，且 A5 泳道确认
目标点位从 1280 次降为 0。这组新证据满足当前的正确性合同与
完整周期性能门槛，因此将该历史回撤项重新恢复。改善不按泳道
Atomic 聚合核时推算，也不声称能与历史 S6.34 的旧底座数据直接相减。

## 2026-08-04：S6.47 用完整周期复审 Fanin 发布位回读（仍不保留）

S6.41 利用严格插入 completion 链已传递 producer output 发布顺序的
事实，在可信 ordered/latest-writer 实例中省略
`SharedFaninOutputPublishedLoad`，曾从 2048 次降为 0。这一正确性论证
仍成立，且不会删除真正决定依赖的 last-writer/history/fanin 检查。
但它在 S6.41 旧底座上已回退，本轮必须用最新 S6.46 底座和完整
startup 到 FinalDrain 口径重新独立裁决。

候选的 CPU 完整协议回归和 CCEC perf-clock 构建 PASS。以提交
`253a2e8a` 冻结基线，按 B-C/C-B 交错各运行 12 个独立 A5 B256
trace-free 进程：

```text
S6.46 frozen baseline: min / median / max / mean
                       1.402488 / 1.434364 / 1.452503 / 1.430565 ms
S6.47 candidate      : min / median / max / mean
                       1.417571 / 1.436281 / 1.486135 / 1.438867 ms
candidate median 回退 0.001917 ms / 0.134%
candidate mean   回退 0.008302 ms / 0.580%
```

候选只有 6/12 对更快，中位数和均值都没有改善。这与 S6.41 的
旧证据同向：形式上减少 2048 次返回型读取，仍不等于全局关键路径
受益。候选代码已完整撤回，正式 ordered Fanin 继续显式确认 output
publication。该结果不否定 completion 链的内存顺序论证，只是否决它在
当前机器码和并发相位下的性能保留价值。

## 2026-08-04：S6.48 用完整周期复审 insert handoff 非返回候选（仍不保留）

S6.37 曾利用中央 Build ticket 的唯一 builder 不变量，保留
`deps_prepared[N]` 的条件 CAS，但不再消费旧值。这会让当前 owner
不能当场发现同 task 异常旧值；N+1 owner 仍会因未观察到 N 而
fail-closed，末 task 则由 host 终态拒绝，所以它是可收口但会延迟报错的
候选，不是等价的错误观察合同。

本轮在 S6.46 底座上重建无泳道候选，CCEC perf-clock 构建与源码
Atomic/DCCI 覆盖门槛 PASS。以提交 `253a2e8a` 冻结基线，按
B-C/C-B 交错各运行 12 个 A5 B256 trace-free 进程：

```text
S6.46 frozen baseline: min / median / max / mean
                       1.408208 / 1.436211 / 1.474566 / 1.440698 ms
S6.48 candidate      : min / median / max / mean
                       1.422702 / 1.439456 / 1.515063 / 1.446619 ms
candidate median 回退 0.003245 ms / 0.226%
candidate mean   回退 0.005921 ms / 0.411%
```

候选有 7/12 对更快，但中位数、均值和最大值均差于基线，没有形成
可保留的完整周期收益。在没有性能收益时，不值得用延迟异常检测
换取源码层面的“不使用返回值”。候选已完整撤回，当前 owner 仍使用
CAS 旧值当场验证 handoff 成功。

## 2026-08-04：S6.49 用完整周期复审 output published 非返回候选（仍不保留）

S6.31 曾利用唯一 Build owner 和 output writer 预留条件，将 descriptor
copy、DCCI 和 StoreBarrier 之后的 published Exchange 改为只发射、不消费
旧值。消费者的 poll 仍可以等到 `task_id`，但生产者不再能当场检测
last-writer 预留与 published 之间的非法并发写，因此这也是错误诊断合同
变弱的候选。

本轮在 S6.46 底座上重建无泳道候选，CCEC perf-clock 构建和
Atomic/DCCI 源码覆盖门槛 PASS。以提交 `253a2e8a` 冻结基线，按
B-C/C-B 交错各运行 12 个 A5 B256 trace-free 进程：

```text
S6.46 frozen baseline: min / median / max / mean
                       1.407474 / 1.443516 / 1.456336 / 1.439355 ms
S6.49 candidate      : min / median / max / mean
                       1.414010 / 1.447342 / 1.465090 / 1.445944 ms
candidate median 回退 0.003826 ms / 0.265%
candidate mean   回退 0.006589 ms / 0.458%
```

候选只有 4/12 对更快，中位数和均值同向回退。最新底座仍然没有为该
候选提供稳定收益，也就没有理由删除生产者侧的非法重复发布检测。
候选已完整撤回，`SharedOutputPublishedExchange` 继续使用返回值验证
旧状态为 `-1`。
