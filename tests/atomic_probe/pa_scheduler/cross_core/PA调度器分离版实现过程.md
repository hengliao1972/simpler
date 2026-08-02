# PA 调度器分离版实现过程

本文按阶段记录 `cross_core` standalone 的实际开发过程、验证证据和遗留问题。目标是先在独立代码中证明“构建与执行可跨 Scalar 核转移”，再决定是否迁移到 Simpler；本文不使用尚未运行的测试结果替代证据。

## 记录约定

- 每完成一个阶段，立即补充实现范围、验证命令、结果和未闭合问题。
- CPU、CCEC IR、A5 动态结果分别记录，不能互相替代。
- 性能结论必须同时给出对照版本、运行参数和结果文件；正确性阶段不提前宣称性能收益。
- 每个功能阶段闭合后都做一次轻量性能/泳道抽查：普通增量先记录后继续功能；出现倍数级回退、单一区域吞掉大部分时间或明显违背协议预期时，必须先定位和修正，不带着结构性异常进入下一阶段。
- 当前只实现 CPU 与 CCEC，暂不实现 AscendC。
- 当前目录是便于验证机制的 standalone；第一版不修改 Simpler 真实 PA 路径。

## 阶段总览

| 阶段 | 目标 | 当前状态 |
| ---- | ---- | -------- |
| S0 | 固定跨核执行包 ABI、状态机和 cacheline 所有权 | standalone portable ABI 已闭合；真实 TensorDesc 对照留到 S3 |
| S1 | CPU 确定性交错测试闭合协议正确性 | 已完成 |
| S2 | CCEC 最小 A5 跨核发布/领取探针 | 已完成 |
| S3 | standalone PA 接入构建/执行分离 | 进行中：S3a 已闭合；S3b 已完成 Submit 当场登记的 CPU 状态机门槛，尚需闭合 fatal 边界与 CCEC/A5 B1 |
| S4 | 泳道、PMU 与 perf-clock 三条证据链 | 未开始 |
| S5 | 根据证据优化非 atomic 路径 | 未开始 |
| S6 | 评估并迁移到 Simpler 真实路径 | 未开始 |

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
停止”；其合同是：本核一旦在对应不可逆边界观察到 terminal fatal，就不再
执行下一项副作用。未来 engine 改为异步 `try_wait` 后，`ENGINE_INFLIGHT`
必须先等硬件真正完成再转 `Faulted`，不能照搬当前同步 helper 的处理。

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
| ---- | ----: | ---- |
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
| ---- | ----: |
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
