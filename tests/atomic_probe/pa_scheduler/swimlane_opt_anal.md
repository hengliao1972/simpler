# PA Scheduler standalone 主执行流程与泳道全景

本文面向第一次接触 FDWIC PA 调度器的开发者，解释
`tests/atomic_probe/pa_scheduler` 这条 standalone 路径从 Host 启动到任务完成的
完整业务流程，以及每一层泳道区间究竟表示什么。

本文的主语始终是 **standalone PA Case1**。它复刻真实 PA 调度协议中的关键
结构，但不是 `simpler` 主场景本身。两者差异集中列在第 9 节；除该节外，不能把
“standalone 当前行为”自动外推成所有 `simpler` 场景的通用行为。

本文只使用当前源码能够证明的事实，不把历史样本数值作为业务定义，也不讨论
I-cache、PMU 或某次性能波动。性能数据会随编译布局和运行环境变化，而这里描述的
执行顺序、状态转移和闭合关系才是新人理解代码的稳定入口。

## 1. 一页看懂全流程

### 1.1 Host、worker 与后处理的总览

```text
Host
  │
  ├─ 解析参数、加载 mixed ELF
  ├─ 分配并初始化 SchedulerState / workload / trace
  ├─ H2D：共享状态前缀、standalone 控制区、可选 workload/trace header
  │
  ├─ launch 32 个 mixed block
  │      └─ 每个 block 启动 1 AIC + 2 AIV，共 96 个 worker
  │
  │   每个 worker
  │      ├─ 初始化自己的 WorkerState
  │      ├─ StartupBarrier：等待 96 个 worker 到齐
  │      ├─ OrchestrationReplay
  │      │    ├─ 初始化整轮 PA descriptor/template
  │      │    └─ 对每个 batch 回放 Alloc → QK → SF → PV → UP
  │      ├─ FinalDrain：等所有 worker 结束回放，并清空本核私有 slot
  │      └─ DFXFinalize：flush trace、汇总并发布 WorkerResult
  │
  ├─ stream synchronize
  ├─ D2H：共享完成状态、WorkerResult、trace header/有效 records
  ├─ 校验拓扑、计数、依赖、完成状态与 workload 输出
  └─ 校验通过后发布 raw JSON，再由脚本生成 Perfetto 和排他分析 JSON
```

这里最重要的三个认识是：

1. 96 个 worker 都按相同顺序回放全部 Submit，不是把五类 Submit 预先静态分给
   某几个核。
2. `Claim` 只为每个全局 task 选出一个 winner；loser 仍会完成参数构造、
   descriptor 物化和本核私有依赖状态维护。
3. kernel winner 在 Submit 内只把任务放进私有 slot。真正的 QK/SF/PV/UP
   计算通常由后续 drain 执行，因此“Submit 返回”不等于“kernel 已完成”。

### 1.2 一个 batch 的业务链

standalone 固定每个 batch 只有一个 q-loop、一个 block group，严格提交五个
task：

```text
BeginPaBatch
    │ 读取本 batch 的 context length，计算 block 数
    ▼
Submit Alloc ── Accept ── PreparePaBlockGroup
    │                          │
    │                          ▼
    │                     Submit QK ── Accept
    │                          │
    │                          ▼
    │                     Submit SF ── Accept
    │                          │
    │                          ▼
    │                     Submit PV ── Accept
    │                          │
    └──────────────────────────┴──► Submit UP
```

从数据依赖看，五个 task 形成下面的调度图：

```text
Alloc ───────────────────────────────┐
                                     │
QK ──► SF ──► PV ───────────────────┼──► UP
       └─────────────────────────────┘
```

显式 fanin 边分别是：

- QK：0 条，输入都是外部 tensor；
- SF：依赖 QK；
- PV：依赖 SF；
- UP：依赖 Alloc、SF 和 PV，共 3 条。

所以每个 batch 有 5 个 fanin edge。descriptor 会在 Submit 阶段先返回给
orchestration，真正的数据可用性由 producer task 的 completion flag 保证。
这使 orchestration 能继续提交后继任务，而不必在每个 Submit 后同步等待计算完成。

### 1.3 一次 Submit 的当前顺序

当前 standalone 使用 compete-first eager 顺序：先做与参数无关的进度推进和
Claim，再让所有 worker 同步构造完整参数，最后消费参数完成 Submit。

```text
Submit API call
  ├─ BeginCallbackSubmit       分配 task_id；位于记录的 Submit.start 之前
  ├─ recorded Submit parent
  │  ├─ EfDrain                尝试执行以前已 ready 的 slot
  │  ├─ Claim                  按 task role 竞争唯一 winner
  │  ├─ [ArgBuild residual]    所有 worker 同步构造完整 TaskArgs
  │  ├─ Materialize            生成输出 descriptor 和 register mask
  │  ├─ PrepareMap             退休超出 H 窗口的 TensorMap 条目
  │  ├─ Fanin                  仅非 Alloc winner 收集 producer task
  │  ├─ Register               更新本核的 hazard producer 视图
  │  ├─ WinnerBuild/AllocComplete（仅 winner）
  │  └─ [tail residual]        公共计数和 Submit.end 取时
  └─ 发布 Submit record 并返回 位于记录的 Submit.end 之后
```

方括号中的 `ArgBuild residual` 是业务解释，不是当前 schema-v4 的独立 raw
phase。泳道只知道它位于 `Claim.end` 与 `Materialize.begin` 之间；该空白还包含
Claim record 发布、Begin/Finish 衔接等成本，所以不能把整个区间都宣称为
“纯参数构造耗时”。

## 2. 先认识参与者与状态

### 2.1 物理拓扑和 worker 编号

CCEC Host launch 32 个 mixed block。每个 block 产生三条运行 lane：

| 运行 lane | worker 编号 | block/lane 映射 | 主要 task |
| --------- | ----------- | --------------- | --------- |
| AIC | `0..31` | `block=worker, lane=0` | QK、PV |
| AIV0/AIV1 | `32..95` | 两个 AIV 对应一个 block，lane 为 1/2 | SF、UP |

因此固定拓扑是 32 AIC + 64 AIV，共 96 个 worker。所有 worker 都有自己的
scalar 控制流、TensorMap、ring slot、heap cursor 和 task payload arena。

每个 task 的 Claim 参与范围不同：

| task | 真正执行 atomicMax 的 worker | 每个 task 的 winner 数 |
| ---- | ---------------------------: | ---------------------: |
| Alloc | 全部 96 个 worker | 1 |
| QK、PV | 32 个 AIC | 各 1 |
| SF、UP | 64 个 AIV | 各 1 |

不符合 role 的 worker 仍有一条 `Claim` span，但 `attempted=false`，不会执行
该 task 的 `atomicMax`。

### 2.2 四类状态及其业务职责

| 状态 | 所有权 | 保存内容 | 业务作用 |
| ---- | ------ | -------- | -------- |
| `SchedulerState` 共享前缀 | 96 核共享 | Claim cursor、task cell、frontier、屏障、fatal | 建立全局唯一 winner、完成可见性和生命周期同步 |
| `WorkerState` | 每个 worker 私有 | heap cursor、TensorMap、slot、payload | 让每核独立回放同一图，并暂存自己赢得的任务 |
| `TaskCell` | 每个全局 task 一份 | `vend`、`flag` | 发布 heap 水位和完成状态，供 fanin/回收读取 |
| `PaOrchestrationState`/`TaskArgs` | 当前 worker 回放栈 | PA descriptor、动态 shape、参数列表 | 把前一 Submit 的输出接到后一 Submit 的输入 |

共享与私有的边界非常重要：

- Claim cursor 和 completion flag 是两种模式都使用的跨核协议；frontier
  是 private ring 回收协议，shared Case1 的严格 no-wrap heap 不消费它；
- private TensorMap 和 slot 只属于本 worker，维护它们不需要跨核原子；
  shared TensorMap sidecar 是全局共享状态；
- 每个 worker 都维护同一 task stream 的私有 descriptor/producer 视图；
- kernel 的完成通过共享 `TaskCell::flag` 对所有 worker 可见。

### 2.3 `vend`、`flag` 和 `frontier` 分别表示什么

private 任务完成时按固定顺序发布：

1. 把 winner 当前的单调 `heap_next` 写入 task 的 `vend`；
2. 执行 store barrier；
3. 把 task 的 `flag` 发布为 ready；
4. 从当前 `frontier + 1` 开始扫描连续 ready 的 task，并用 `FetchMax`
   单调推进共享 frontier。

shared no-wrap Case1 只执行前三步。它用每 task flag 做 fanin，可用共享
shard cursor/vend 做容量终态校验，并且编译期禁止 heap 回绕，所以不需要在
每次完成后维护连续 frontier。`SchedulerState::frontier` 和相关 schema
仍保留给 private；未来 shared 支持回绕时必须恢复等价的回收代际协议。

三者不能混为一个概念：

- `flag(task)` 回答“这个 task 是否完成”；
- `vend(task)` 回答“完成该 task 时，该 worker 的逻辑 heap 走到了哪里”；
- `frontier` 回答“从 task 0 起连续完成到了哪里”，不能跨过中间未完成的空洞。

`HeapGuard` 使用 `frontier - H` 对应 task 的 `vend` 判断旧物理 heap 区间是否
已经允许复用。`H=64` 同时也是 TensorMap producer 的存活窗口，但 TensorMap
退休进度和共享 frontier 并不要求在同一时刻相等。

## 3. 五类 task 分别做什么

### 3.1 Alloc：建立本轮累计状态

Alloc 构造三个 `Output`：

- accumulated output tile；
- accumulated sum；
- accumulated max。

所有 worker 都会为这三个输出物化 descriptor。唯一 winner 不构建 kernel
slot，而是在 `AllocComplete` 中通过 `HeapGuard` 后立即发布 task 完成。
orchestration 随即用 `AcceptTaskOutputs` 保存三个 descriptor，供同 batch 的
UP 使用。

业务效果是“为本次 q/group 链建立累计结果的逻辑存储和 producer 身份”，不是
执行一次计算 kernel。

### 3.2 QK：产生 score descriptor 和 AIC 任务

QK 参数包括：

- 当前 batch/query 的 view；
- key cache 和 block table；
- 一个动态 shape 的 score `Output`；
- block 数和 block-table offset 两个 scalar。

32 个 AIC 竞争一个 winner。QK 输入都是外部 tensor，没有前序 task owner，
所以 winner 收集到 0 条 fanin。winner 在 `WinnerBuild` 中构造私有 slot，
score descriptor 则立即返回并被 SF 引用。

### 3.3 SF：等待 QK，产生 probability 和统计量

SF 参数包括 QK score，一个 probability `Output`、max/sum 两个 `Output`，
以及 scale、block 数和末 block 有效长度三个 scalar。

64 个 AIV 竞争一个 winner。score descriptor 的 owner 是 QK task，所以 SF
有 1 条 fanin。SF slot 可以先构建，但只有 QK 的 completion flag ready 后，
drain 才会执行它。

### 3.4 PV：等待 SF，产生新的 output tile

PV 参数包括 SF probability、value cache、block table，一个 tile `Output`，
以及 block 数和 block-table offset 两个 scalar。

32 个 AIC 竞争一个 winner。PV 对 SF 有 1 条 fanin。返回的 tile descriptor
会作为 UP 的输入之一。

### 3.5 UP：合并本组结果并登记写 hazard

UP 参数包括：

- SF 的 max、sum 和 PV 的 output，三项 `Input`；
- accumulated max/sum/output 和最终 output view，四项 `Inout`；
- first-group、last-group 两个边界 scalar。

64 个 AIV 竞争一个 winner。UP 的显式依赖去重后是 Alloc、SF、PV 三个 task。
它没有新的 `Output` descriptor，但四个 `Inout` 会形成 `register_mask`，并由
每个 worker 登记到自己的 TensorMap，表示“当前 UP 是这些 backing range 的
最新写者”。

### 3.6 默认 256 batch 时应看到的固定规模

设 batch 数为 `B`：

| 计数 | 公式 | `B=256` |
| ---- | ---: | ------: |
| 每 worker 的 Submit | `5B` | 1,280 |
| 96 核 Submit 总记录 | `96 × 5B` | 122,880 |
| 实际 Claim atomicMax | `288B` | 73,728 |
| 全局 winner task | `5B` | 1,280 |
| AllocComplete | `B` | 256 |
| WinnerBuild / Kernel | `4B` | 1,024 |
| fanin edge | `5B` | 1,280 |

这些是协议不变量，不是某次运行的经验均值。若这些计数不成立，应先判断为
回放、路由、丢记录或完成协议异常，而不是直接分析性能分布。

## 4. Worker 的完整生命周期

### 4.1 Host 准备和 launch

CCEC Host 的主要工作顺序是：

1. 在创建 ACL 资源前解析公共参数、winner workload 参数和构建变体约束；
2. 读取 mixed AICore ELF，并完成 ACL/stream/kernel 注册；
3. 分配 `SchedulerState`、真实计算 workspace 和可选 trace 区；
4. 每轮调用 `InitializeState`，配置 trace 和 winner workload；
5. H2D 只传共享状态前缀、standalone 控制区和必要输入，不搬运每个 worker
   的完整私有 arena；
6. 以 32 个 mixed block launch，并同步等待设备完成。

Host 的 launch-to-sync 墙钟包含设备 launch、worker 启动、全部回放、最终
drain、DFX 收尾和 stream 同步。它不包含 launch 前的 H2D，也不包含同步后的
D2H/JSON。因此它不能由 `OrchestrationReplay + FinalDrain` 直接闭合。

### 4.2 Worker 私有状态初始化

每个 worker 在进入 task 0 前会：

- 写入 role、core、block、lane、sub-block 身份；
- 把 `local_index` 和 `heap_next` 归零；
- 重置私有 TensorMap；
- 清空四个物理 slot；
- 附着当前 worker 的 trace/统计上下文。

四个物理 slot 中有两个保留给生产 BlockWon ABI，因此 standalone 单 lane
任务仍只有两个普通可用 slot。这个容量会触发真实的 ring backpressure，不能
因为 standalone 不执行 joint task 就把四个 slot 全部当成普通容量。

### 4.3 StartupBarrier：统一回放起点

每个 worker 对共享 `started_count` 做一次增量，然后轮询到配置的 worker 数。
等待期间同时检查 fatal，并由 watchdog 防止参与者缺失造成永久挂死。

该屏障的业务效果是压低各核进入 task 0 的启动偏斜。Claim 唯一性本身依赖
atomicMax，不依赖屏障来保证正确性。

`StartupBarrier` 位于 `OrchestrationReplay` 之前，当前没有完整父 span。它可以
作为 Atomic 发生位置观察，但不能被塞进 WorkerCompletion 的加和公式。

### 4.4 OrchestrationReplay：所有 worker 重放同一 PA 图

`OrchestrationReplay` 从 `InitPaOrchestration` 之前开始，到最后一个 Submit
返回并退出回放循环之后结束。它包含：

1. 建立整轮共用的外部 tensor descriptor 和固定 create-info；
2. 对每个 batch 读取 context length，并计算 block 数；
3. 依次执行五个 compete-first Submit；
4. 接收每个 Submit 返回的 output descriptor，更新后继 task 的输入；
5. 执行 Submit 间的循环、view、分组和返回控制逻辑。

所有 worker 都走这段代码，所以每核 `task_id` 都是连续的 `0..5B-1`，且
`task_id % 5` 固定映射为 Alloc/QK/SF/PV/UP。

### 4.5 FinalDrain：停止生产后清空在途任务

回放结束并不保证本核 slot 已空。每个 worker 先增加共享 `replay_done`，然后
循环执行 `DrainReady(FinalDrain)`，直到同时满足：

- `replay_done` 表明所有 worker 都已退出 orchestration replay；
- 当前 worker 的 `occupied_count == 0`。

第一个条件保证不会再产生新 slot，第二个条件保证本核已经执行完自己拥有的
旧 slot。循环若没有释放任何 slot就执行 spin hint，之后继续检查 fanin 和全局
完成状态。

standalone 没有 BlockWon，所以这里没有生产路径的 `has_pending_won()` 条件。
这是与 `simpler` 主运行时的重要差异之一。

### 4.6 DFXFinalize：发布证据，不再属于业务完成窗口

FinalDrain 结束后，worker 才执行：

- 延后写出 `OrchestrationReplay` 和 `FinalDrain` 两条父记录；
- atomic 观察构建中记录两条 `ClockBaseline`；
- flush 本核有效 trace 区；
- 汇总计数、最大 slot 占用、TensorMap 状态等；
- 把 `WorkerResult` 发布到独占的结果分区。

父记录故意在被测区间之后写出，避免它们自己的 GM 写入落入任一父区间。
`DFXFinalize` 当前没有完整父 span，因此 WorkerCompletion 闭合不等于整个 worker
函数从入口到返回的闭合。

## 5. 一次 Submit 的逐阶段业务语义

### 5.1 `BeginCallbackSubmit`：建立本次提交身份

每个 worker 都用自己的 `local_index++` 得到 task id，并把 context 绑定到：

- 当前 `WorkerState`；
- `task_id & payload_mask` 对应的私有 `TaskPayload`；
- 空的 result、fanin、register mask 和 kernel id。

这里不读取 `TaskArgs`。这是 Claim 能先于参数构造的前提，也是 Begin/Finish
之间只能同步衔接、不能嵌套另一次 Submit 的协议约束。

源码在 `BeginCallbackSubmit` 返回后才采集 `submit_begin`。所以这段 prologue
属于 OrchestrationReplay，却不属于 raw `Submit` 父 span：首个 task 落在
OrchestrationSetup，其余 task 落在 BetweenSubmitResidual。

### 5.2 `EfDrain`：用本次 Submit 的前沿推进旧任务

`EfDrain` 调用 `DrainReady` 扫描本 worker 的私有 slot。对于全部 fanin flag
已 ready 的 slot，它会：

1. 执行该 slot 对应的 winner workload；
2. 记录 `Kernel`；
3. 按 vend → flag → frontier 顺序完成任务；
4. 记录零时长 `Commit`；
5. 清空 slot 并降低 `occupied_count`。

因此 task `N` 的 `EfDrain` 可能执行 task `N-1` 或更早的 kernel。按外层
Submit 的 task kind 给其中 Kernel 归类会得到错误结论，必须使用 Kernel
自己的 `task_id/function_id`。

如果本核没有占用 slot，`EfDrain` 是快速空路径；如果有 ready slot，它可能
包含完整 engine workload 和完成发布。

### 5.3 `Claim`：决定谁负责本 task 的真实尾动作

Claim 先根据 task kind 生成 active role，再选择四个 cursor shard 之一：

- Alloc 使用 `alloc_cursor`；
- QK/PV 使用 `cube_cursor`；
- SF/UP 使用 `vector_cursor`。

符合 role 的 worker 对 cursor 执行 `atomicMax(task_id)`。返回旧值小于当前
task id 的唯一竞争者获胜；其他参与者是 attempted loser，不符合 role 的核是
not-attempted loser。

Claim 的输出包括：

- `won`：是否是唯一 winner；
- `attempted`：是否真正执行了 atomicMax；
- `function_id`：winner 应构建的 QK/SF/PV/UP 负载类型。

standalone 固定 task 都是单 lane；若 active mask 同时要求两条及以上 lane，
Claim 会显式拒绝，因为本用例没有实现 BlockWon/joint 协议。

### 5.4 eager 参数构造：所有 worker 构建相同参数

Claim 返回后，同步 callback 才构造 `TaskArgs`：

- Alloc 新建参数对象；
- QK/SF/PV/UP 复用并 reset 同一个参数对象；
- view、动态 create-info、tensor/scalar 参数都在 callback 中立即求值；
- callback 和内部 thunk 不被保存，也不会跨 Submit 生命周期执行。

winner 和 loser 都走完整构参路径。这保证所有 worker 的 descriptor、heap
cursor 和私有 dependency map 按相同 task stream 演进；compete-first 只改变
Claim 与构参的先后顺序，没有改成 winner-only lazy 构参。

CCEC standalone 的 Begin/Finish 之间传递固定 16-byte ticket，finish 通过同一
role-specific block-local state恢复 context，并校验 worker、task 顺序和 cookie。
CPU/AscendC 复用相同业务函数，但不需要这条 CCEC split ABI。

### 5.5 `Materialize`：把逻辑 Output 变成可引用 descriptor

`MaterializeTask` 完成四件事：

1. 扫描参数 tag，形成 `output_mask` 和 `register_mask`；
2. 计算每个新 Output 的大小及 1 KiB 对齐后的总空间；
3. 在单调逻辑 heap 上安排连续区间，跨物理 ring 尾时跳到下一圈；
4. 在本 worker 的 task payload 中初始化 GM `TensorDesc`，写入
   `owner_task_id`，并推进 `heap_next`。

其输出是后继 orchestration 立即可引用的 descriptor，以及本 task 的
`output_bytes`。它只分配和描述逻辑输出，不代表其中的数据已经由 kernel 写好。

所有 worker 都执行 Materialize，包括 loser。这样后续回放可以在任何 worker
上用同一个 task id 和 owner 拓扑继续构图。

### 5.6 `PrepareMap`：退休不再参与依赖查询的历史写者

`AdvanceTensorMap` 把本 worker 的 producer 存活下界推进到 `task_id - H`。
超出窗口的 producer entry 会沿 task 链批量从地址 bucket 摘除，再放回 free
list。

其业务效果是限制依赖表的生命周期和容量，同时保留窗口内“同一 backing
buffer、字节区间重叠、task id 最新”的 producer 查询语义。

TensorMap 是 worker 私有结构，所以这个阶段不做跨核同步。它也不等价于 heap
已经可复用；物理 heap 安全仍由共享 frontier/vend 的 `HeapGuard` 判断。

### 5.7 `Fanin`：winner 建立执行前置条件

只有非 Alloc winner 执行 `CollectFanin`。它对每个非纯 Output 参数：

1. 读取 descriptor 的显式 `owner_task_id`；
2. 对 Input/Inout 查询 TensorMap 中最新的重叠写者；
3. 对两种来源得到的 producer task id 去重。

最终 fanin 数组随 winner slot 保存。slot 执行前逐项读取共享 task flag，只要
有一项未 ready，就保留该 slot 等待下次 drain。

QK winner 即使最终 fanin 为 0，也会有一条真实 `Fanin` span；Alloc 不执行
该阶段；所有 loser 都不收集 fanin。

### 5.8 `Register`：更新后继任务看到的 hazard 版本

Register 使用 Materialize 生成的 `register_mask`，把 Inout 或
OutputExisting 登记为“由当前 task 写入”。

当前固定图中：

- private Alloc 以 `include_existing=false` 调用，实际不插入；
- private QK/SF/PV 只有新 Output，register mask 为空；
- private UP 有四个 Inout，因此每个 worker 都插入四个 producer entry；
- shared 只有 Claim winner 进入 Register，并只读验证 PA Case1 的
  ordinary region 为空；loser 在 Claim 后直接返回，没有 Register span。

所以“Register 不是 winner-only”只适用于 private 的每核 TensorMap。
shared 当前的 Register 是 winner-only 的空 region 协议验证。两种模式都
不能把 Register 解读为 kernel 已完成。

### 5.9 winner 和 loser 的尾部分支

非 Alloc winner 进入 `WinnerBuild`：

1. 若两个普通 slot 已满，`WaitForSlot` 循环 drain 直到有空间；
2. private 对有新 Output 的任务运行 `HeapGuard`，避免覆盖仍存活的
   ring 区间；shared 使用 Materialize 已完成的 no-wrap shared heap
   reservation，不执行 private HeapGuard；
3. 预留一个私有 slot；
4. 把 active descriptor、scalar、dispatch context 和 fanin 快照复制进 slot。

`WinnerBuild` 的产物是一个 `occupied && built` 的待执行 slot，不是已完成
kernel。

Alloc winner 进入 `AllocComplete`：它不创建 slot。private 在 HeapGuard
通过后发布 vend/flag/frontier；shared 使用已完成的 shared heap reserve，
建立任务完成态后发布 fresh-output descriptor。

private loser 没有额外业务动作，但仍走公共 Submit 收尾；不能为了让图
看起来完整而虚构 `Replay` 或 `LoserReplay`，剩余后缀归入
`SubmitResidual`。shared loser 在 Claim 后返回稳定 output symbol，不进入
generic/split finish，也没有 Submit 父区间；它只保留 EfDrain/Claim，
其余空白由 Orchestration residual 承接。

### 5.10 Submit 返回时已经保证了什么

成功路径的 Submit 返回只保证：

- private 本 worker、shared winner 已完成本 task 的 descriptor 和依赖
  前端；shared loser 只建立稳定 output symbol；
- 全局唯一 winner 已选出；
- Alloc winner 已完成，或 kernel winner 已把任务构造成待执行 slot；
- 后继 orchestration 拿到了 private descriptor 或 shared task/slot symbol。

它不保证：

- QK/SF/PV/UP kernel 已执行；
- 后继 task 的 fanin 已 ready；
- 所有 worker 已完成同一 task 的 Submit；
- heap 对更老逻辑区间已经可回收。

## 6. 延迟执行、背压与完成协议

### 6.1 `DrainReady` 有三个调用位置

同一套 drain 逻辑会在三个位置推进任务：

| 位置 | 触发原因 | 泳道父位置 |
| ---- | -------- | ---------- |
| `EfDrain` | 每次新 Submit 开头顺手推进旧任务 | 当前 Submit 的 `EfDrain` |
| Ring backpressure | slot 满或 heap 暂不可复用 | `WinnerBuild`/`AllocComplete` 内部 |
| `FinalDrain` | 所有 Submit 结束后清空尾部在途任务 | `FinalDrain` |

所以一条 Kernel 必须按实际时间包含关系归入上述唯一父位置，不能假设所有
Kernel 都在 EfDrain，也不能把 `Kernel` 再与已包含它的父 span相加。

### 6.2 slot 容量为什么会形成背压

每个 worker 只有两个普通可用 slot。如果 winner 连续产生任务，而旧 slot 因
fanin 未 ready 尚未释放，`WaitForSlot` 会主动调用 `DrainReady`。只有本核占用
数降到容量以下，新的 winner payload 才能入队。

等待不是纯自旋：只要有依赖已满足的旧 slot，本核会执行其 kernel并推进全局
完成状态。这保证 backpressure 路径本身也能帮助系统取得进展。

### 6.3 heap ring 为什么还需要另一层保护

slot 有空不代表输出 heap 可以安全覆盖。`heap_next` 是单调逻辑地址，真正落到
256 MiB 物理 ring 时才取模。第一圈内不会覆盖旧区间；发生回绕后，
`HeapGuard` 读取共享 frontier 和退休 task 的 vend，确保当前 live window 不
超过一圈。

若暂时不安全，`HeapGuard` 同样会 drain 本核 ready slot；若 frontier 已经追到
当前 task 前仍无法满足容量，则发布 fatal，而不是静默覆盖数据。

### 6.4 为什么 FinalDrain 是完整业务的一部分

最后一个 Submit 返回时，最后几条 winner task 可能仍在 slot 中。若只统计首个
Submit 到最后一个 Submit，就会漏掉这些 kernel、完成发布和跨核等待。

因此完整的 per-worker 业务口径必须是：

```text
WorkerCompletion = OrchestrationReplay + FinalDrain
```

这也是泳道 schema-v4 新增顶层父区间的根本原因。

## 7. 泳道分区与“总耗时”口径

### 7.1 三种事件角色

泳道事件必须先按角色分类，才能讨论加和：

1. **父区间**：例如 `Submit`、`OrchestrationReplay`、`FinalDrain`；
2. **排他子区间**：同一父区间内互不重叠，可与 residual 一起闭合父区间；
3. **嵌套或 Overlay**：例如 Kernel、Atomic、RingBp、Commit、ClockBaseline，
   用于定位，不得再次加到已经包含它的父区间。

“把所有 duration 再求和一次”会同时重复计算父子事件、多核并行时间和 Atomic
重叠观察窗口，因此没有明确的业务意义。

### 7.2 schema-v4 的严格层级

```text
WorkerCompletion                         离线派生的每核业务父口径
├─ OrchestrationReplay                   raw 父 span
│  ├─ OrchestrationSetup                 父起点到首个 Submit
│  ├─ SubmitUnion                        本核全部 Submit 的并集
│  │  └─ 每个 Submit
│  │     ├─ EfDrain
│  │     ├─ Claim
│  │     ├─ Materialize
│  │     ├─ PrepareMap
│  │     ├─ Fanin（条件存在）
│  │     ├─ Register
│  │     ├─ WinnerBuild/AllocComplete（winner 条件存在）
│  │     ├─ SubmitInternalResidual（merged: submit_residual）
│  │     └─ SubmitTailResidual（merged: submit_tail_gap）
│  ├─ BetweenSubmitResidual              相邻 Submit 之间的精确空白
│  └─ OrchestrationTail                  最后 Submit 到父终点
└─ FinalDrain                            raw 父 span
   ├─ KernelUnion
   └─ FinalDrainResidual

嵌套定位：Kernel、RingBp
非加和 Overlay：Atomic、ClockBaseline、Commit 等
```

其中 `EfDrain` 还能严格分成：

```text
EfDrain = EfDrain.KernelUnion + EfDrainControl
```

`WinnerBuild` 或 `AllocComplete` 内若发生背压 drain，也可能包含 Kernel。当前
排他报告会验证 Kernel 的唯一父位置，但不能把这些 Kernel 额外加到 Submit 上。

### 7.3 每核可以严格闭合的公式

所有计算先使用 raw 整数 tick，闭合后才按 metadata 中的频率换算时间：

```text
Submit
  = EfDrain + Claim + Materialize + PrepareMap
  + optional(Fanin) + Register
  + optional(WinnerBuild or AllocComplete)
  + SubmitInternalResidual + SubmitTailResidual

SubmitEnvelope
  = SubmitUnion + BetweenSubmitResidual

OrchestrationReplay
  = OrchestrationSetup + SubmitUnion
  + BetweenSubmitResidual + OrchestrationTail

FinalDrain
  = FinalDrain.KernelUnion + FinalDrainResidual

WorkerCompletion
  = OrchestrationReplay + FinalDrain
```

exclusive JSON 为兼容聚合口径，仍把两者之和记作 `submit_residual`：

```text
submit_residual
  = submit_internal_residual + submit_tail_residual
```

其中：

- Submit 前缀及相邻 child 之间的 internal residual；
- 最后一个 child 到 Submit.end 的 tail residual。

residual 是“当前边界尚未单列的真实时间”，不是一个可以随意命名的业务函数。

### 7.4 打点边界与 record 发布归属

raw span 记录的是两个时间戳的差，不等于同名 helper 的无扰动净耗时。当前实现
有意复用相邻边界以保证整数闭合，但 `WriteTrace` 本身发生在边界取时之后：

| 区间 | start 边界 | end 边界 | 紧邻的观测归属 |
| ---- | ---------- | -------- | -------------- |
| `Submit` | `BeginCallbackSubmit` 之后 | 公共计数之后、发布父 record 之前 | prologue 在父外；父 record 写入下一段 gap |
| `EfDrain` | 与 Submit.start 相同 | `DrainReady` 返回后 | 自己的 record 写入落入后续 Claim 观察区 |
| `Claim` | 复用 EfDrain.end | atomicMax/role 路由完成后 | Claim record 和 eager 构参位于随后 residual |
| `Materialize` | callback 完成、进入 Finish 后 | `MaterializeTask` 返回后 | 自己的 record 写入落入 PrepareMap 观察区 |
| `PrepareMap` | 复用 Materialize.end | `AdvanceTensorMap` 返回后 | record 写入落入 Fanin 或 Register 观察区 |
| `Fanin`/`Register` | 复用前一业务边界 | 对应 helper 返回后 | record 写入落入下一 child 或 tail residual |
| winner tail | 复用 Register.end | build/complete 返回后 | tail record 写入落入 Submit tail residual |

`OrchestrationReplay.end` 同时复用为 `FinalDrain.start`，两条父 record 都在
FinalDrain 结束后才发布，因此父记录写入不属于任一业务父区间。

这种布局的优点是父子区间可以无浮点误差地闭合；代价是阶段值应解释为“当前
源码边界下的观察区”，不能直接当成单个 helper 的纯函数耗时。若将来调整 mark，
必须同时验证业务边界、record 写入归属、闭合关系和插桩扰动，不能只移动标签。

### 7.5 三个 merged residual span 的具体业务内容

`between_submit_residual`、`submit_residual` 和 `submit_tail_gap` 都由
`swimlane_converter.py` 对 schema-v4 的现有父子区间取补集生成。设备端不写
这三个 phase，converter 也不增加新时间：三类 span 只是把原来未命名的区间按
位置显示出来。

converter 只为 `end > start` 的正区间生成 Perfetto X event；若两个边界相等，
该段数学贡献为 0，merged 中不会出现一条零时长 residual。

exclusive analyzer 使用的名称略有不同：

- merged `submit_residual` 对应 `submit_internal_residual`；
- merged `submit_tail_gap` 对应 `submit_tail_residual`；
- `between_submit_residual` 在两份产物中同名。

#### 7.5.1 `between_submit_residual`：两个已记录 Submit 父区间之间

精确边界是同一物理 scalar lane 上：

```text
previous Submit.end → next Submit.start
```

当前 standalone 成功路径中的公共动作顺序是：

```text
previous Submit.end
  ├─ 写 previous Submit 的 TraceRecord
  ├─ 检查它是否为本核最后一个 task，并从 Finish 返回
  ├─ 回到 batch orchestration，处理前一 task 的返回结果/循环控制
  ├─ 调用下一个 SubmitCallbackTask
  ├─ BeginCallbackSubmit：local_index++、绑定 payload、清 context
  └─ 读取 next Submit.start
```

`BeginCallbackSubmit` 明确在 `Submit.start` 之前，因此属于这个 gap，而不是下一
个 Submit。下一 Submit 的 EfDrain、Claim 和 eager 构参都在 `Submit.start`
之后，不属于 `between_submit_residual`。

五种 transition 的业务内容并不相同：

- **Alloc → QK**：保存 accumulated output/sum/max 三个返回 descriptor；执行
  `PreparePaBlockGroup(0)`，计算当前 group 的 block offset、block 数和末 block
  有效长度；然后进入 QK Submit prologue。
- **QK → SF**：保存 QK score descriptor，然后进入 SF Submit prologue。
  SF 的动态 probability create-info 和完整参数列表仍在下一 Submit 的
  Claim 后构造，不在这个 gap。
- **SF → PV**：保存 probability、max、sum 三个 descriptor，然后进入 PV
  Submit prologue。
- **PV → UP**：保存 PV output descriptor，然后进入 UP Submit prologue。
  first/last scalar 和 output view 也在 UP Submit 内的 eager callback 构造。
- **UP → 下一 batch 的 Alloc**：UP 没有新 Output 需要 Accept；代码退出本轮
  五 task 顺序，递增 batch，执行 `BeginPaBatchForCallback`。这里会从 GM
  `context_lens` 读取下一 batch 的真实 sequence length，计算 block 数，增加
  `context_reads` 统计，再进入 Alloc Submit prologue。

设 batch 数为 `B`，private 每核依次有 `B` 个 Alloc→QK、QK→SF、SF→PV
和 PV→UP 候选 gap，以及 `B-1` 个 UP→Alloc 候选 gap，总计 `5B-1`
个相邻 Submit 边界。shared 只记录本核 winner Submit，因此每核候选数为
`winner_submit_count-1`，中间的 loser EfDrain/Claim 会切割 residual，
不会被整段重复覆盖；没有 winner 的核不存在该类区间。严格 merged event
数仍以 `next.start > previous.end` 为准。父区间前后剩余时间分别属于
`OrchestrationSetup` 与 `OrchestrationTail`。

因此 `between_submit_residual` 是一个有明确边界、但内容随 transition 变化的
orchestration 区域。它不能整体命名成“参数构造”：当前主要参数构造已经移动到
下一 Submit 内的 compete-first eager callback。

#### 7.5.2 `submit_residual`：Submit 内显式 child 之间

converter 从每条已记录的 `Submit.start` 开始，按时间排序 EfDrain、
Claim、Materialize、PrepareMap、可选 Fanin、Register 和可选 winner tail。
private 记录所有逻辑 Submit，shared 只记录 winner Submit。每遇到
`previous_child.end < next_child.start`，就为这个空白生成一条
`submit_residual`。因此一个 Submit 理论上可以有多条同名 span，必须结合两侧
child 名称解释。

按当前打点布局，成功路径中主要的正区间是：

```text
Claim.end → Materialize.begin
```

它按源码顺序包含：

1. 发布当前 `Claim` 的 TraceRecord；
2. private 所有 worker 执行完整 `BuildCallbackSubmitArgs<Kind>`；
   shared 的 Alloc 所有 worker 只构三个轻量 Output 参数，QK/SF/PV/UP
   只有 winner 构造完整参数；
3. 检查 builder 是否有效，并累计 reset/view/create-info/参数个数统计；
4. private 所有 actor、shared winner 构造固定 16-byte
   `CallbackSubmitTicket`；
5. CCEC 路径跨入 noinline split-finish TU，恢复对应 role 的 block-local
   runtime state，校验 worker、task id、winner、cookie 和 ticket；
6. 进入 `FinishCallbackSubmitBody`，读取 `Materialize.begin`。

private 的第 2 步是 compete-first eager 主体，winner、attempted loser 和
not-attempted loser 都完整执行五类构参。shared 则只有 Alloc 保留全员
轻构参，其余四类只有 winner 执行：

- Alloc：构造新的 `TaskArgs`，加入三个 Output；
- QK：reset，构造 query view 和动态 score create-info，加入三个 Input、一个
  Output 和两个 scalar；
- SF：reset，构造动态 probability create-info，加入一个 Input、三个 Output
  和三个 scalar；
- PV：reset，加入三个 Input、一个 Output 和两个 scalar；
- UP：reset，构造 output view，加入三个 Input、四个 Inout 和两个 scalar。

CPU 不经过 CCEC 的 cross-TU ABI，但仍执行同一 builder、ticket 语义和
Finish 入口衔接。因此 private residual 可以描述为
“Claim record + 全员 eager ArgBuild + Begin/Finish bridge”；shared winner
residual 是“Claim record + winner ArgBuild + Begin/Finish bridge”。shared
loser 没有 Submit 父区间，不能套用该解释。

它不包含 Claim 的 role 路由/atomicMax，也不包含 `MaterializeTask`：前者已经在
`Claim` child 内结束，后者从 `Materialize.begin` 才开始。若未来出现其他
`A.end → B.start` 空白，converter 也会使用 `submit_residual` 名称，exclusive
JSON 中的 boundary 字段才是区分来源的依据。

#### 7.5.3 `submit_tail_gap`：最后一个 child 到 Submit.end

converter 对每个 Submit 只把最后一个显式 child 之后的后缀命名为
`submit_tail_gap`：

```text
last exclusive child.end → Submit.end
```

起点取决于 Claim 结果：

- 非 Alloc winner：从 `WinnerBuild.end` 开始；
- Alloc winner：从 `AllocComplete.end` 开始；
- private loser：没有 winner tail，从 `Register.end` 开始。

private 有上述三类路径；shared 只保留两类 winner Submit：

- **非 Alloc winner**：发布 `WinnerBuild` TraceRecord，增加本核 Submit 计数，
  读取 `Submit.end`；slot 等待、HeapGuard 和 payload build 已经位于
  `WinnerBuild` child 内。
- **Alloc winner**：发布 `AllocComplete` TraceRecord，增加 Submit 计数，读取
  `Submit.end`；HeapGuard、vend/flag 发布和 frontier 推进已经位于
  `AllocComplete` child 内。
- **private loser**：发布 `Register` TraceRecord，增加 Submit 计数，读取
  `Submit.end`。它在 Register 后没有 Replay/LoserReplay 或其他调度
  动作，所以这段不能解释为“loser replay”。

winner 路径的 Register record 写入发生在 `Register.end` 之后，但由于 winner
tail 复用 `Register.end` 作为 start，它在数值上属于 WinnerBuild/AllocComplete，
不属于 `submit_tail_gap`。private loser 没有该 tail child，所以同一笔
Register record 写入才落入它的 `submit_tail_gap`；shared loser 没有
Register、Submit 或 `submit_tail_gap`。

`WriteTrace(Submit)` 发生在 `Submit.end` 取时之后，也不属于
`submit_tail_gap`：非末次 Submit 时它落入随后的 `between_submit_residual`；
最后一个 Submit 时落入 `OrchestrationTail`。

所以 `submit_tail_gap` 的业务语义是“最后一个已记录 child 之后、Submit 父区间
结束前的公共观测与 epilogue”，不是独立调度阶段。它仍必须保留，才能使每个
Submit 在整数 tick 上严格闭合。

#### 7.5.4 三者在层级和加和中的关系

```text
OrchestrationReplay
  ├─ Submit（private 全量；shared winner-only）
  │    ├─ explicit exclusive children
  │    ├─ submit_residual       内部前缀/child 间空白，可有多段
  │    └─ submit_tail_gap       最后 child 后缀，至多一段
  ├─ shared loser EfDrain/Claim 已知 span
  └─ residual                   相邻 Submit 或父区间边缘的未知空白
```

排他闭合时只能这样加：

```text
Submit
  = explicit children
  + submit_internal_residual
  + submit_tail_residual

OrchestrationReplay
  = setup + SubmitUnion
  + between_submit_residual + tail
```

不能把 `between_submit_residual` 再加进某个 Submit，也不能把 merged 中的
`submit_residual` 和 exclusive JSON 聚合层的总 `submit_residual` 当成两个不同
区域重复相加。

### 7.6 必须区分的四个“总时间”

#### 单核 Submit envelope

```text
last_submit_end(core) - first_submit_start(core)
```

它只覆盖该核的首末 Submit，不含 OrchestrationSetup/Tail 和 FinalDrain。

#### 单核 WorkerCompletion

```text
final_drain_end(core) - orchestration_begin(core)
```

它是本文最完整、可严格闭合的设备业务口径，但仍不含 StartupBarrier 和
DFXFinalize。

#### 跨核 Submit makespan

```text
max(last_submit_end) - min(first_submit_start)
```

起点和终点可能来自不同核。它是墙钟包络，不等于任一核的阶段和，也不等于
96 核 duration 总和。

#### aggregate core-work

```text
sum(per_core_metric)
```

它回答“所有 scalar lane 累计投入了多少核时间”，适合比较阶段工作量分布，
不回答用户等待了多久。排他分析 JSON 明确把它与 global makespan 分开输出。

### 7.7 Atomic 为什么只能作为 Overlay

Atomic span 已经位于 Claim、EfDrain、FinalDrain、AllocComplete 等父区间内。
PollBatch 还可能用一条记录覆盖一个完整轮询 episode，而不是一条指令。

因此 Atomic 只能回答“原子访问发生在哪个业务区域、规模如何”，不能：

- 再加到父阶段上；
- 从父阶段机械相减得到纯 scalar 时间；
- 用 aggregate Atomic duration 除以调用数得到单条硬件延迟；
- 直接解释全局 makespan 的关键路径。

## 8. 从设备记录到可验收泳道

### 8.1 `run.sh swimlane` 的三份产物

```text
device trace
   │
   ▼
l2_swimlane_records.json          raw，业务证据和十列整数 ABI
   ├─ swimlane_converter.py
   │      └─ merged_swimlane.json Perfetto/Chrome Trace 可视化
   └─ swimlane_exclusive_analyzer.py
          └─ swimlane_exclusive_analysis.json 排他闭合与统计
```

`swimlane` action 固定执行一轮，并默认打开逐 Atomic 观察。它按 backend 建立
独立输出目录，先让 runner 完成语义校验和 raw 发布，再顺序运行 converter 和
exclusive analyzer；任一步失败都会返回非零。

排他分析器直接读取 raw，不以 merged 文件作为数值输入。merged 的职责是展示；
exclusive JSON 的职责是用整数区间验证加和关系。

### 8.2 raw 记录表达什么

raw 的 `fdwic_events` 每行固定十列：

```text
core, block, lane, task, function, phase, start, end, flags, auxiliary
```

时间戳单位不能按平台名称猜测，只能使用同一 raw metadata 中的
`clock_freq_hz`。当前 standalone schema-v4 支持 phase-only level 1 和带 Atomic
的 level 4；二者业务 phase 相同，观察扰动不同。

### 8.3 converter 做的事情

converter 会：

- 校验 schema、时钟、32 AIC + 64 AIV 拓扑和 producer summary；
- 把每个 mixed block 建成一个 Perfetto process；
- 为每条硬件 lane 拆出 runtime 和 kernel 子泳道；
- 根据 raw flags 标注 claim won/lost/not-attempted；
- 只用已有父子边界合成 `between_submit_residual`、`submit_residual` 和
  `submit_tail_gap`；
- 完整写入临时文件并同步后再原子替换 merged 输出。

合成 residual 不改变 raw ABI，也不凭空创造业务阶段。

### 8.4 exclusive analyzer 的 fail-closed 门禁

排他报告只有在以下条件全部满足时才生成：

- 96 个 core id、32/64 role map 和 block/lane 映射完整；
- `dropped_records == 0`；
- 每核 Submit task id 连续且 task stream 完全一致；
- 每个 Submit 恰有 EfDrain、Materialize、PrepareMap、Claim、Register；
- Fanin 和 winner tail 的条件数量与 Claim winner/Alloc 标记一致；
- 同一父区间的排他 child 不重叠，且 task id 与父 Submit 一致；
- 每核恰有一个 OrchestrationReplay 和一个 FinalDrain，二者边界相邻；
- 每条 Kernel 唯一归入 EfDrain、winner tail 或 FinalDrain；
- 所有闭合式在 raw 整数 tick 上精确相等。

Host 在 raw 发布前还会校验共享 flag/vend/frontier、Claim/winner 计数、每核
结果、trace header 和真实 workload 输出。也就是说，“能打开 JSON”不是验收
标准；producer、Host 和离线分析三层都通过才是一份可用证据。

### 8.5 最小复现入口

```bash
cd tests/atomic_probe/pa_scheduler
./run.sh build ccec
./run.sh swimlane ccec --batches 256
```

CCEC 才是本文 A5 mixed-core 路径的性能证据入口。CPU backend 用于协议和算术
语义检查，不是 A5 时序基线；不同 backend 的绝对 duration 不应互相代替。

## 9. 与 `simpler` 主场景的差异

下面比较 standalone 与当前
`examples/a5/fully_distributed_within_core/paged_attention_unroll` 加通用
dist runtime。这里区分“生产 runtime 具备的能力”和“PA Case1 本次实际会走的
分支”，避免把能力差异误写成每轮必然发生的动作。

| 维度 | standalone PA scheduler | `simpler` 主场景 |
| ---- | ----------------------- | ---------------- |
| 目标 | 独立复现 PA Case1 的 Submit/依赖/完成协议 | 通用 dist runtime 执行真实 example orchestration 和 kernel |
| Host/入口 | 专用 Host 准备 `SchedulerState` 并 launch mixed ELF | Runtime 准备 L2 参数，各 AICore 回放 orchestration entry |
| 图规模 | 每 batch 固定一组 `Alloc+QK+SF+PV+UP` | 可有多个 q-loop 和 block group；Case1 才退化为每 batch 5 task |
| 参数与数据 | PA 形状/descriptor 拓扑接近真实路径，地址与计算 workspace 为 standalone 输入 | descriptor 指向真实输入输出，kernel 结果构成真实数值数据流 |
| 当前 PA API | compete-first eager，所有 worker 构参 | 该 PA example 也使用 compete-first eager；旧 one-shot API 仍供其他 example 使用 |
| Begin/Finish ABI | 16-byte ticket，只携带单 lane winner 所需状态 | 32-byte ticket，还携带 joint、ready、kind 等生产状态 |
| Claim 能力 | 只允许单 lane，`active_count >= 2` 显式拒绝 | 通用 runtime 支持 joint task 和 BlockWon 发布/领取 |
| loser 尾动作 | kernel loser 无额外动作，只形成 residual | 调用 `drain_block_won()` 并记录 `LoserReplay`；Case1 可快速返回 |
| EfDrain | 只 drain 本核普通私有 slot | 先处理 BlockWon，再 drain phase-B 私有 slot |
| FinalDrain | 等 all-replayed 且本核 slot 为空 | 还必须确认没有 pending BlockWon lane |
| 数值 workload | 默认用独立 workspace 运行合成的 Cube/Vector 完整流水 | 执行 example 的真实 QK/SF/PV/UP kernel 与真实 tensor |
| trace record | standalone `TraceRecord` 为 64 B | `FdwicSwimlaneRecord` 为 32 B，且多 `LoserReplay` 等生产 phase |
| 拓扑假设 | analyzer 固定验证 32 AIC + 64 AIV | worker/block 数来自 Runtime 配置，通用代码不能假设永远为 96 |

### 9.1 图规模的具体差异

真实 unroll orchestration 的一轮 q-scope 是：

```text
Alloc × 1
for each block group:
    QK → SF → PV → UP
```

所以一般 task 数取决于 batch、`q_loop` 和 block group 数，并不总是 `5B`。
standalone 固定 `q_loop=1` 且每 batch 只有一个 group，才得到本文的五 task
周期。本文第 3.6 节的计数门禁只适用于这个固定输入边界。

### 9.2 “aicpu_orchestration_entry” 名称不能按字面误读

在当前 CCEC dist replay 中，每个有效 AICore worker 从 Runtime 复制
orchestration tensor/scalar 参数，再直接调用链接进设备镜像的
`aicpu_orchestration_entry`。这个符号名沿用 API 历史，不表示本文泳道中的
OrchestrationReplay 是 Host/AICPU 墙钟。

standalone 不经过这套通用 Runtime 参数装载，而是直接在
`RunSchedulerImpl` 中构造固定的 `PaOrchestrationState`。

### 9.3 数据依赖相似，但数值计算不能等同

standalone 对 descriptor owner、TensorMap overlap、fanin flag 和
vend/frontier 的调度依赖是实的；QK→SF→PV→UP 的顺序也是真实 PA Case1
拓扑。

但默认 `real-compute` 的 engine workload 使用独立 workspace，目的在于提供
稳定的 AIC/AIV 计算负载和完成等待，不会把 QK 的数值输出真正送入 SF，再送入
PV/UP。因此可以用它研究调度和取时布局，不能用它验证 PA 数值正确性或把其
kernel duration 直接当成主场景 kernel duration。

### 9.4 泳道数值不能跨两条路径直接横比

两条路径的 record 大小、BlockWon/loser 动作、Host 入口、计算体和编译布局都
不同。即使 phase 名相同，absolute duration 也不具备天然可比性。

可以复用的是：

- `OrchestrationReplay + FinalDrain` 的顶层业务口径；
- compete-first 的 EfDrain → Claim → eager ArgBuild → Finish 顺序；
- Materialize、PrepareMap、Fanin、Register 的业务定义；
- parent/child/residual/overlay 的排他建模方法。

必须重新验证的是：

- 实际 task 数和 dependency graph；
- joint/BlockWon 与 `LoserReplay` 的条件分支；
- 每条 Kernel 的唯一父位置；
- Host 端完整 makespan 的起止边界；
- 观察插桩对真实 ELF 布局和性能的影响。

## 10. 源码阅读地图

建议按以下顺序阅读，先建立业务图，再进入平台细节：

1. [common/pa_model.h](common/pa_model.h)

   查看拓扑常量、TaskKind、共享/私有状态、TracePhase 和 raw ABI。

2. [common/pa_frontend.h](common/pa_frontend.h)

   查看 descriptor、TensorMap、Materialize、Fanin、Register 和 slot payload。

3. [common/pa_scheduler_core.h](common/pa_scheduler_core.h)

   从 `RunSchedulerImpl` 进入，再读 `SubmitCallbackTask`、
   `FinishCallbackSubmitBody`、`DrainReady`、`CompleteTask`。

4. [ccec/callback_runtime_entry.cpp](ccec/callback_runtime_entry.cpp) 与
   [ccec/callback_finish.cpp](ccec/callback_finish.cpp)

   查看 CCEC worker 入口和 compete-first split finish 边界。

5. [ccec/host.cpp](ccec/host.cpp) 与 [run.sh](run.sh)

   查看 Host launch、D2H 校验、raw 发布和三段后处理流水线。

6. [swimlane_converter.py](swimlane_converter.py) 与
   [swimlane_exclusive_analyzer.py](swimlane_exclusive_analyzer.py)

   查看 Perfetto 映射、residual 合成和整数闭合门禁。

对照 `simpler` 主场景时，再阅读：

- [生产 Submit runtime](../../../src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/submit_runtime.h)
- [生产 core main](../../../src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/core_main.h)
- [compete-first API](../../../src/a5/runtime/fully_distributed_within_core/orchestration/pto_orchestration_api.h)
- [PA unroll orchestration](../../../examples/a5/fully_distributed_within_core/paged_attention_unroll/kernels/orchestration/paged_attention_orch.cpp)

## 结论

standalone PA scheduler 的主执行模型可以压缩为一句话：

> 96 个 worker 同步重放同一条 PA task stream；每个 Submit 先推进旧任务、竞争
> 唯一 winner，再由所有 worker 完成 eager 构参与私有依赖建模；winner 只把
> kernel 放入 slot，后续 drain 在 fanin ready 后执行并发布全局完成，最后由
> FinalDrain 清空尾部在途任务。

泳道的正确加和方式也可以压缩为一句话：

> 只在同一物理 lane、同一父区间内，对互斥 child 与 residual 做整数闭合；
> Kernel、Atomic 和多核 aggregate 不能再次加到墙钟父区间上。

掌握这两句话后，再去看某个 phase 的时间分布，才能区分“业务工作发生在哪里”、
“任务何时真正完成”和“观察器把时间记到了哪里”。
