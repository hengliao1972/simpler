# FDWIC Shared TensorMap 与 Swimlane Deps 分支迁移审查

本文记录从
`qingchuanyu/fdwic-swimlane-deps`
向
`chenpeng/fdwic-shared-tensormap`
迁移实现的代码审查结论。

本文先记录两个分支分别实现了什么、哪些内容需要迁移、哪些内容需要排除；
随着逐项对齐，继续记录已经确认的取舍、shared-aware 的适配要求、按影响面
排序的迁移阶段和每阶段验证门禁。

本文仍是迁移分析与执行前计划，不表示已经修改生产 runtime，也不把尚未
确认的建议自动视为已接受。

## 1. 审查快照

| 项目 | 固定值 |
| --- | --- |
| 审查日期 | 2026-07-25 |
| 目标分支 | `chenpeng/fdwic-shared-tensormap` |
| 目标提交 | `351ef62e` |
| 来源分支 | `qingchuanyu/fdwic-swimlane-deps` |
| 来源提交 | `1726a774` |
| 两分支 Git merge-base | `599703f5b153a3a0fd2a3516dff4efd49be3f00a` |
| 目标分支 private 基线 | `f5da1a2e` |
| 来源分支 private 基线 | `bd620630` |

`f5da1a2e` 和 `bd620630` 虽然是两个不同提交，但对应的整棵源码树完全一致。
因此本文使用下面两个区间区分双方真正独有的实现：

- `f5da1a2e..351ef62e`：Shared TensorMap 分支新增实现；
- `bd620630..1726a774`：Swimlane Deps 分支新增实现。

不能直接把两个分支 tip 做单向 diff 后按“增加/删除”理解。两个分支在相同
private runtime 基线上分别演进，直接 diff 会把 Shared TensorMap 的新增实现
显示成来源分支的删除。

直接 tip diff 共涉及约 629 个文件、`+140124/-8613` 行。其中
`tests/atomic_probe` 在来源分支中有 509 个文件，约 10.8 万新增行，是整体
diff 的主要组成部分。

本文结论基于固定 Git ref 的源码、提交历史和分支内实验记录；没有把本地
未提交文件作为分支事实，也没有重新执行 A5 性能测试。

## 2. 两分支的共同基础

在上述两个 private 基线之前，两边已经共同具备：

- A5 FDWIC AICPU orchestration；
- orchestration 与 AICore kernel 的联合执行；
- per-core runtime state 和 replay；
- 直接 CCEC submit；
- payload 和 output 生命周期处理；
- ring drain、fanin、task completion；
- private TensorMap、private heap 和 output layout；
- mixed AIC/AIV task 和 BlockWon 协议；
- 基础 tracing；
- private slot cache flush 优化。

这些内容不是本次迁移对象，不应重复搬运。

## 3. Shared TensorMap 分支实现内容

### 3.1 Shared TensorMap 数据结构

目标分支增加了三组 shared 状态：

- `SharedOutputCell`：按 `(producer_task_id, output_slot)` 保存 fresh output
  descriptor、发布标记和 `last_writer`；
- `SharedRegionMap`：记录普通 Tensor 地址区间及其 producer；
- shared heap：由 task winner 统一分配输出。

fresh output 通过 `FdwicOutputRef` 表示未物化的输出引用。普通 Tensor 和
INOUT 重叠关系仍通过 region map 解析。

主要实现文件：

- `runtime/dist_engine/common/state.h`
- `runtime/dist_engine/aicore/submit_core.h`
- `runtime/pto_types.h`

### 3.2 Winner-first submit 协议

目标分支把 shared submit 拆为：

```text
presubmit / claim
    -> winner: 构参、分配、注册依赖、发布输出、提交执行
    -> loser: 只返回 symbolic output reference
```

PA 的重参数构造被放入 `tok.won` 分支。loser 不再重复构造动态
`TensorCreateInfo`、完整 `L0TaskArgs`、输出 descriptor 或 shared map
记录。

### 3.3 Shared 引用解析和依赖

目标分支实现了：

- kernel 执行前 resolve `FdwicOutputRef`；
- producer 尚未发布时等待 `published`；
- descriptor invalidate 后复制到本地执行参数；
- 用 `last_writer` 维护 fresh output 的 INPUT/INOUT writer 链；
- 用 `SharedRegionMap` 维护普通 Tensor 的重叠依赖；
- region intent 在 replay 前发布 writer 信息；
- mixed follower 由 winner 解析 shared input 后再 launch。

### 3.4 缓存和并发顺序

目标分支补充了：

- shared descriptor 发布前的 DCCI flush；
- consumer 读取前的 invalidate；
- acquire/release publication；
- shared 状态 cache-line 隔离；
- winner readiness 和 follower launch gate；
- final drain 中的 joint-launch completion 检查。

### 3.5 Shared 热路径优化

目标分支还实现了：

- wrong-role worker skip；
- 按需执行 EfDrain；
- shared heap 分片；
- alloc contention 过滤；
- shared resolve 热路径 guard；
- 保留 trace flags 和更精确的 trace attribution。

### 3.6 Shared 测试和 trace

目标分支扩展了：

- `simple_orch_smoke`；
- `shared_symbol_smoke`；
- `submit_dependency_smoke`；
- shared region/ref chain、长 INOUT chain、delayed registration；
- mixed AIC/AIV 和 dual-AIV 场景；
- shared resolve trace。

当前 shared swimlane 已使用 phase ID 14～17：

- `Resolve = 14`
- `ResolveWait = 15`
- `ResolveInvalidate = 16`
- `ResolveCopy = 17`

## 4. Swimlane Deps 分支实现内容

### 4.1 Atomic probe

`tests/atomic_probe` 是一套独立实验资产，包含：

- CPU、CCEC 和 AscendC atomic CAS/exchange 等探针；
- `ld_dev`、`st_dev`、DCCI、bypass dcache；
- cache-line blast/clobber；
- I-cache 和 PMU 探针；
- PA standalone scheduler/model；
- atomic swimlane 分析工具；
- lazy-lambda A/B/C 实验及原始产物。

它不等价于生产 runtime DFX；多数 probe 可以独立构建和运行。

### 4.2 Private PA 参数准备优化

提交 `dbbf621a` 包含两类变化：

1. 复用一个 scope-local `L0TaskArgs`，覆盖 alloc 以及 QK、SF、PV、UP；
2. 把 `TensorRef` 默认构造改成不初始化未使用 slot。

分支记录的 A5 PA Case1 scalar-body 结果：

| 版本 | 时间 | 相对原始版本 |
| --- | ---: | ---: |
| 原始四个参数容器 | 6.170876 ms | 基线 |
| 一个容器复用四个 kernel | 5.720638 ms | -7.30% |
| 再省略未使用 slot 初始化 | 5.645826 ms | 累计 -8.51% |
| alloc 也复用同一容器 | 5.577570 ms | 累计 -9.61% |

这里的“省略未使用 slot 初始化”不是 lazy-lambda，但它与目标分支新增的
`FdwicOutputRef` union 成员存在构造语义冲突，不能原样应用。

### 4.3 BlockWon 首次 joint-submit gate

提交 `e3b748b4` 使用全局 latch，使没有出现过 joint task 的 private PA
不查询 BlockWon publication。

来源分支记录它消除了 146944 次无意义 atomic poll，PA 从 5.642245 ms
下降到 5.171330 ms。

目标分支已经有语义等价的 `g_fdwic_block_won_enabled`，并在
`drain_block_won_if_enabled()` 中使用。因此来源分支的主体优化已经存在。

仍有一个残余差异：目标分支的 `has_pending_won()` 没有检查已有 gate，
而 `dist_submit_has_drain_work()` 会调用它。

同一提交中的 register mask 单独没有稳定收益。目标分支的
`DistOutputLayout::output_indices` 也已经替代了 output rescan。

### 4.4 Private heap 首圈快速路径

提交 `04ec9b95` 在 private heap 的 fatal-checked 分配循环内增加 H1：

- 当 `heap_next <= ring` 时直接返回；
- 首圈不可能发生 wrap，因此不读取 frontier/vend atomic；
- PA 中减少 1024 次 frontier load；
- 10 次 A5 配对运行中 8 次获胜，中位数约 -0.324%；
- 已验证 wrap 和 backpressure。

目标代码尚未包含该优化，并且对应函数已经位于
`#if !PTO_FDWIC_SHARED_MAP` 下。

### 4.5 G16 两级 final barrier

提交 `c9ea57cb` 把平坦 `replay_done` 完成线改为：

```text
worker
    -> 16 个 leaf arrival
    -> root arrival
    -> root release
    -> 每个 leaf release
    -> worker exit
```

worker 在等待期间仍继续 drain。新状态追加在旧 ABI 后，原
`replay_done` 字段保留。

来源分支记录：

- G16 相比 G8，FinalDrain 减少 12.264 us，即 -3.425%；
- full completion 改善 -0.287%；
- Submit 基本不变；
- 已验证 A5Sim block dimension 1/36 和 A5 PA。

目标分支当前仍使用平坦 final barrier，但增加了
`joint_launches_drained_for_lane()`，迁移时必须保留 shared 退出条件。

### 4.6 PrepareMap 跳写候选

提交 `6b85d4fc` 在 `task_heads[slot]` 已经等于 `-1` 时跳过重复写入。

它有理论写次数、CPU differential test 和 A5Sim 验证，但提交记录明确没有
真实 A5 A/B 数据，代码量和控制分支反而增加。

### 4.7 Compete-first 和 lazy-lambda

提交 `2899cc35` 实现 compete-first eager submit：

- 先执行 EfDrain 和 claim；
- 再执行参数 callback；
- 但所有 worker 仍构造参数。

分支记录的实际改善约 -0.263%，区间重叠，主要价值是建立测量阶段边界。
目标分支的 winner-first shared API 已经让 loser 不构造参数，语义和收益路径
都更直接。

提交 `ba4334d1` 对比 lazy-lambda C 和 eager B：

- 中位数差异约 `+0.040%`；
- 22 次运行各自获胜 11 次；
- 没有可确认收益。

### 4.8 Atomic swimlane

提交 `cbaf7c60`、`dbb95bb5` 及相关提交实现：

- schema v3/v4；
- 每条 32-byte record；
- per-core trace state；
- 28 个 private atomic site、5 类 atomic op；
- 直接记录和 PollBatch 聚合；
- logical/physical atomic closure；
- `OrchestrationReplay`、`FinalDrain` 等 exclusive parent；
- `WinnerBuild`、`AllocComplete`、`LoserReplay`；
- host converter、analyzer 和 strict closure。

来源分支记录 PA 中有 115200 次 logical atomic call、110006 次 physical
atomic、340 个 PollBatch，且 dropped record 为 0。

### 4.9 PMU 和 perf-clock DFX

来源分支还增加：

- 每物理核 PMU owner；
- PMU save/config/restore；
- perf-clock；
- kernel aggregate；
- 多组 phase-specific profile；
- host report 和 provenance；
- ELF/profile gate。

这部分连同 atomic swimlane，约增加 1.3 万行 runtime、platform、host 和
工具代码。

### 4.10 兼容性修改

来源分支包含：

- 老 A5 driver 无 CPU topology 时的平坦 OCCUPY fallback；
- 用 `ACL_DEV_ATTR_AICPU_CORE_NUM` 校验 fallback 核数；
- 跨架构 host `strip` 兼容；
- onboard FDWIC scene 跳过 PTO tile `.text` extraction；
- BGEMM `pto::Stride` 限定和 orchestration weak symbol；
- minibench DCCI store 和 include/symbol 冲突修复。

其中一次提交曾加入 runtime-wide post-kernel output flush，随后已由
`b422c48f` 完整回退。最终结论是：标量 producer kernel 应自行完成发布，
不应让通用 runtime 无条件 flush 全部输出。

## 5. 逐项迁移判断

下面的编号用于后续逐项对齐。每项以其显式“状态”或“建议”为准。

### 5.1 建议接收

#### M01：整体接收 `tests/atomic_probe`

状态：**已确认整体接收（2026-07-25）**。

原因：

- 全部为来源分支新增内容；
- probe、原始数据和分析说明可以作为独立实验资产；
- 不要求把生产 runtime atomic/PMU DFX 一并迁入。

`lazy_lamda_sample` 也保留，但只作为“没有收益”的实验记录，不能据此启用
lazy-lambda 生产实现。

#### M02：PA 单 `L0TaskArgs` 复用

状态：**已确认不接收（2026-07-25）**。

原审查建议迁移的范围是：

- 一个 scope-local 容器复用 QK、SF、PV、UP；
- 在生命周期安全的前提下让 alloc 也复用该容器；
- 保留 shared 模式下只有 winner 填充重参数的行为。

现已确认不迁移这项 PA 参数容器复用。直接把 `TensorRef()` 改成
`= default` 本来就不属于 M02，仍单列为 C02，等待独立确认。

#### M03：Private heap H1 首圈快速路径

建议：**接收**。

原因：

- 修改局部；
- 只影响 private map；
- 有真实 A5 配对数据；
- wrap/backpressure 已验证；
- 不改变 shared heap。

#### M04：G16 两级 final barrier

建议：**适配后接收**。

当前目标分支在每个 worker 完成 orchestration replay 后执行：

```text
每个 worker 对同一个 replay_done cacheline 执行 FetchAdd
    -> 每个 worker 反复读取同一个 replay_done cacheline
    -> replay_done == num_workers 后，才允许退出 final drain
```

36 个 worker 会同时写、读同一条全局 cacheline。G16 不是修改 task replay、
TensorMap 或 kernel barrier，而是只替换“所有 replay worker 已经进入最终
排空阶段”的通知方式：

```text
worker
    -> 按 block_id % 16 到达本组 leaf_arrival
    -> 每组 AIC leader 等本组全部 worker 到达
    -> 16 个组最多各向 root_arrival 发布一次
    -> group 0 的 AIC leader 等所有 active group 到达
    -> 发布一次 root_release
    -> 每组 leader 发布本组 leaf_release
    -> 组内 worker 观察 leaf_release 后可退出
```

等待 barrier 的过程中，worker 仍执行 ring drain 和 BlockWon drain。因此它
只减少最终汇合 cacheline 的竞争，不提前停止排空，也不改变任务完成条件。

迁入目标分支需要修改四处：

1. 在 `DistGlobal` 尾部追加 16 组 leaf arrival/release 和一组 root
   arrival/release；保留旧 `replay_done` 字段及既有字段 offset；
2. AICPU register 时清零这些计数，并根据实际 `layout[]` 计算每个 leaf 的
   worker 数和 active group 数；
3. 把 `dist_submit_drain_to_completion()` 中对单一 `replay_done` 的
   FetchAdd/poll 换成 leaf/root/release 协议；
4. 更新 debug dump、布局断言和 final-barrier 定向测试。

来源补丁不能原样应用，目标分支必须保留以下差异：

- 调用 `drain_block_won_if_enabled()`，不退回来源分支的无条件
  `drain_block_won()`；
- shared 模式退出条件仍必须包含
  `joint_launches_drained_for_lane(self)`；
- 使用目标分支当前普通 atomic helper，不引入 C01 的 atomic trace wrapper；
- `FinalBarrierState` 追加在当前 `PreparedDeps` 和 `DistCore` 之后，避免改变
  shared ABI 中已有字段 offset；
- 同时验证 private/shared、block dimension 1/36，以及目标分支支持的最大
  block dimension。

修改后的退出条件等价于：

```text
本组已收到全局 release
&& 本核 ring 为空
&& 本 lane 没有待接收 BlockWon
&&（shared）本 lane 的 joint launch 已全部 drain
```

它不是把“全部任务完成”降级为“本组完成”；root release 仍严格表示所有
active group、也就是所有 replay worker 都已进入 final drain。

#### M05：补齐 `has_pending_won()` gate

建议：**接收缺失的小部分**。

目标分支已经有 `g_fdwic_block_won_enabled`，不需要引入来源分支的重复 latch。
只需让 `has_pending_won()` 在 gate 未开启时直接返回 false，避免按需 drain
查询继续执行 BlockWon atomic load。

来源分支的 8.35% 数据对应完整 gate，不应直接归因给这一处残余 guard；该
guard 仍需在目标分支重新测量。

#### M06：老 A5 topology fallback

建议：**接收**。

fallback 只能在 topology 不可用且 ACL AICPU core count 与 OCCUPY popcount
完全相等时启用，否则继续失败，风险有明确保护边界。

#### M07：必要的构建兼容修改

建议：**选择性接收**。

接收：

- 跨架构 host `strip` 兼容；
- onboard FDWIC scene 跳过不适用的 text extraction；
- BGEMM `pto::Stride` 和 weak symbol 修复；
- 确实被接收测试需要的 include/symbol 冲突修复。

不接收来源分支在 `runtime_builder.py` 中重复增加的 PTO include。目标分支
已经通过 `pto_isa_root` 和构建参数处理相同问题。

#### M08：`mix_coown` 测试意图

建议：**改写后接收**。

它独有地覆盖一个 task 同时包含 `1C+2V` 的 joint/co-owned 场景。目标分支
现有测试覆盖 1C+1V 和 dual-AIV，但缺少同一 task 的 1C+2V。

不建议原样复制 private `rt_submit_task` 用例；应改为目标分支显式
presubmit/winner/loser API，并增加 onboard 验证入口。

#### M09：旧 shared-mode 测试改用 `PTO_FDWIC_SHARED_MAP`

状态：**已确认适配后接收（2026-07-25）**。

来源测试不再通过运行时环境变量
`PTO_DIST_TENSORMAP_MODE=shared`
判断 shared 模式，统一改为目标分支的编译期
`PTO_FDWIC_SHARED_MAP`：

- C/C++ 测试代码使用 `#if PTO_FDWIC_SHARED_MAP` 区分协议和断言；
- 测试构建分别生成 `PTO_FDWIC_SHARED_MAP=0/1` 的 private/shared ELF；
- Python 驱动不凭旧环境变量猜测模式，而是依据构建 profile、manifest 或
  明确的 case 参数选择对应 ELF。

这项改造不能只替换开关名字。MB5/MB7 及相关 helper 中针对旧 shared ring
的 head/seq/window 等断言，必须改成当前 `SharedOutputCell`、
`SharedRegionMap`、symbolic output、publication 和 writer-chain 语义。

### 5.2 DFX 与需要重新设计的项目

#### C01：Atomic swimlane runtime DFX

状态：**已确认接收，并且必须同时支持 private/shared（2026-07-25）**。

迁移顺序：**放在全部非 DFX 修改完成并稳定后再迁移**。

不能直接移植的原因：

1. 目标分支 phase ID 14～17 已用于 shared resolve，来源 schema 与其冲突；
2. 来源分支的 28 个 site 只覆盖 private runtime；
3. shared 新增的下列原子状态尚未建模：
   - `shared_heap_cursor`
   - `shared_heap_vend`
   - output `published`
   - output `last_writer`
   - `joint_launch_expected/drained`
   - `deps_prepared`
   - region-map bucket、insert sentinel 和 high-water
4. 来源 phase 顺序基于 private/compete-first，目标是 shared winner-first；
5. instrumentation 曾引入 I-cache 回退，需要配套 cold/out-of-line 修改。

迁移时应定义新的 shared-aware schema，而不是直接复用来源分支 v3/v4。
详细设计见第 8 节。

#### C02：省略未使用 `TensorRef` slot 初始化

状态：**已确认不接收（2026-07-25）**。

来源实现使用 `TensorRef() = default`。目标分支的 union 新增
`FdwicOutputRef`，而 `FdwicOutputRef` 带默认成员初始化，原补丁不能安全地
机械应用。

该优化不是 lazy-lambda，但也不是功能依赖。来源分支的数据是在先完成 M02
单容器复用后，再加入 C02 得到的累计结果；没有 C02 相对原始多容器实现的
独立 A/B，也没有目标 shared 分支上的数据。M02 不接收后，C02 失去了已经
验证过的组合上下文，却仍需要为 shared union 重新设计 active-member 和
kind tag 的安全构造方式，当前收益证据不足以覆盖这项风险。

若未来 profile 单独证明 `L0TaskArgs`/`TensorRef` 默认构造是目标分支热点，
再把它作为新的独立优化重新评审，不沿用来源补丁的 `= default` 写法。

#### C03：PMU/perf-clock 完整工具链

状态：**已确认接收，并且必须同时支持 private/shared（2026-07-25）**。

迁移顺序：**放在 C01 之后，作为最后一组 runtime DFX 能力迁移**。

这套工具用于回答“PA Submit 时间为什么变化”，不提供新的调度、依赖或
atomic 功能。它包含两层观察：

| 观察层 | 记录内容 | 用途 |
| --- | --- | --- |
| `perf-clock` | 每核首个 Submit 开始、末个 Submit 结束的 SYS_CNT；可选聚合窗内 linked-kernel ticks/calls | 低扰动地取得每核 Submit 整窗和 kernel/residual 粗分解 |
| `submit-PMU` | A5 硬件 PMU 的 scalar/vector/cube busy、I-cache request/miss、总周期；可选一个局部 phase sidecar | 判断时间变化来自 scalar 控制、linked kernel、I-cache miss，还是某个 Submit 阶段 |

`submit-PMU` 为减少观测扰动，不在一次运行里同时打开全部阶段，而是针对
下面的阶段分别编译和采集 ELF：

- ArgBuild
- EmptyBracket
- Materialize
- Claim
- Register
- SubmitTransition
- EfDrainControl
- PrepareMap
- Fanin
- WinnerBuild
- AllocComplete
- LoserReplay

完整链路还包括：

1. AICPU PMU owner 根据物理 core/subcore 配置硬件 counter；
2. 保存原 PMU 配置，采集后执行 readback 和 restore；
3. AICore 在整窗或选中 phase 边界读 counter，并暂停 linked kernel 所在
   区间以区分 scalar control；
4. 每核发布固定结果和可选 phase sidecar；
5. host 校验物理核拓扑、counter selector、调用次数、owner 恢复和构建身份；
6. Python 工具汇总逐核分布、I-cache 指标、phase 占比并生成报告。

它与 C01 atomic swimlane 的区别是：

- C01 记录每个 atomic site 的逻辑/物理次数、轮询聚合和时序；
- C03 读取硬件性能计数器，观察整窗或某一阶段的忙周期和 I-cache 行为；
- 二者可以联合解释同一次回退；C03 的 return-ready 扣减会复用 C01 的
  atomic site 分类和 wrapper，因此实现顺序上 C01 先于 C03；
- 完整接收 `tests/atomic_probe` 也不要求把 C03 接入生产 runtime。

不能原样搬运的原因：

- 代码面大；
- profile 组合和环境 gate 多；
- 与来源分支 atomic schema、private phase 和 compete-first 历史强耦合；
- 目标分支 winner-first 后，ArgBuild、WinnerBuild、LoserReplay 等阶段边界需要
  重新定义；
- 来源 host/report 把采集拓扑写死为 32 AIC + 64 AIV，而目标 runtime
  支持最多 108 worker，必须从本次运行拓扑动态闭合；
- 它不属于 shared TensorMap 正确运行的前置条件，所以应在功能迁移全部稳定
  后再接入，避免 DFX 插桩掩盖功能回归。

详细 shared-aware 迁移设计见第 8 节。

### 5.3 建议排除

#### X01：Lazy-lambda 生产实现

状态：**已确认排除（2026-07-25）**。

原因：A/B/C 数据没有收益，且会改变 orchestration API 和构参生命周期。

#### X02：Compete-first eager submit

建议：**排除**。

原因：收益基本持平；目标分支的 shared presubmit/winner-only 已覆盖其主要
意图，并且 loser 路径更轻。

#### X03：来源分支 output/register mask

建议：**排除**。

原因：

- register mask 单独没有稳定收益；
- 目标分支已经用 `DistOutputLayout::output_indices` 避免 output 重扫；
- shared region/reference 注册语义与 private mask 不同。

#### X04：PrepareMap 已为 `-1` 时跳写

建议：**暂不接收**。

原因：没有真实 A5 A/B 证据，收益仅停留在理论写次数和模拟验证。

#### X05：Runtime-wide post-kernel output flush

建议：**排除**。

原因：来源分支最终已经回退。标量 producer 应自行发布，不能让通用 runtime
为所有 output 无条件执行 flush。

#### X06：只为 instrumentation 回归服务的冷路径修改

状态：**不作为独立性能优化接收；随 C01/C03 按需接收**。

包括 winner/alloc branch cold hint、atomic record/pollbatch out-of-line 等。
其中 atomic record/PollBatch out-of-line 是来源分支修复 DFX 插桩导致
I-cache 回退的必要组成，迁移 C01 时需要按目标 private/shared 热路径重新
验证后带入；不能脱离 C01 单独搬运，也不能机械复制 compete-first 布局的
branch hint。

#### X07：来源分支已否定或回退的实验

建议：**排除**。

包括：

- fanin reorder；
- BlockWon noinline slow path；
- cursor G8；
- 16-byte ticket；
- 其他最终未保留的负收益实验。

## 6. `tests/st` 额外用例判断

### 6.1 不应整体接收 atomic minibench

部分最终版 minibench oracle 要求日志或环境变量：

- `PTO_DIST_DEPSIG`
- `PTO_DIST_TENSORMAP_MODE`
- `PTO_DIST_OVERHEAD`
- `PTO_DIST_RUNAHEAD`

这些标记在来源分支最终 runtime 中已经没有对应生产实现，原样复制会产生
测试与 runtime 脱节。

### 6.2 旧 shared ring 测试改用当前编译期模式

MB5/MB7 依赖运行时选择：

```text
PTO_DIST_TENSORMAP_MODE=shared
```

其模型是旧 shared TensorMap ring。目标分支使用编译期
`PTO_FDWIC_SHARED_MAP`、`SharedOutputCell` 和 `SharedRegionMap`，协议不同。

已确认不再直接排除这些测试，而是按 M09 适配：

- 模式判断改用 `PTO_FDWIC_SHARED_MAP`；
- private/shared 分别构建和运行；
- 删除旧 runtime-selectable ring 假设；
- 把 `test_dist_atomic_mb5_shared_map` 及相关 helper 的真值模型改为当前
  shared output/region 协议。

### 6.3 其他不建议原样接收的测试

- `vector_example`：目标分支 dependency smoke 已覆盖更完整；
- MB1/MB3/MB9 synthetic UT：复制的是旧 cursor/ring 算法，不能代表当前
  production runtime；
- `test_fdwic_tensor_map_retire`：只服务于尚无真实收益证据的 PrepareMap
  候选；
- atomic PollBatch/converter/report 单测：只在接收 C01 时需要。

### 6.4 可保留的测试意图

- `mix_coown`：按 M08 改写；
- MB2/MB8 的直接数据一致性场景：若 `atomic_probe` 和目标 smoke 仍有覆盖
  缺口，可抽取测试意图，但不复制旧 runtime oracle。

## 7. 当前建议集合

当前已确认和仍待确认的集合为：

```text
已确认接收：
  M01 atomic_probe 整体
  M09 旧 shared-mode 测试改用 PTO_FDWIC_SHARED_MAP 并更新协议断言
  C01 atomic swimlane runtime DFX（private + shared）
  C03 PMU/perf-clock 完整工具链（private + shared）

已确认不接收：
  M02 PA 单参数容器复用
  C02 未使用 TensorRef slot 的安全惰性初始化
  X01 lazy-lambda

仍待逐项确认：
  M03 private heap H1
  M04 G16 final barrier
  M05 has_pending_won 现有 gate 补齐
  M06 legacy topology fallback
  M07 必要构建兼容修复
  M08 改写后的 mix_coown

排除：
  X02 compete-first
  X03 output/register mask
  X04 PrepareMap 跳写
  X05 runtime-wide output flush
  X07 已否定或回退的实验

绑定接收：
  X06 中确属 C01/C03 插桩正确性或 I-cache 修复所需的部分
```

实施约束已经确认：按影响面从小到大分阶段；每阶段通过第 10 节的完整门禁
后才进入下一阶段；atomic 和 PMU DFX 放在所有非 DFX 阶段之后。

## 8. C01/C03 的 shared-aware 迁移设计

### 8.1 总体边界

C01/C03 不以 cherry-pick 来源分支提交为目标，而是复用其中已经验证的
DFX 机制，并在目标分支当前协议上重新落点：

- 保留目标分支现有 shared winner-first、symbolic output、
  `SharedOutputCell`、`SharedRegionMap`、joint launch 和 resolve 语义；
- 不带入 compete-first、lazy-lambda、旧 shared ring 或 private
  96-worker 假设；
- 先冻结非 DFX runtime 的最终结构，再定义 atomic site 和 PMU phase；
- C01 先于 C03。C03 需要复用 C01 对 atomic `result-used`/
  `return-ready` 的分类，才能从 scalar submit 窗中正确扣除等待返回值的
  atomic dependency 时间；
- DFX 关闭时必须能从编译产物中完全裁掉记录路径，保留一份不含 DFX
  插桩的权威性能基准 ELF。

来源分支的 atomic schema v4 不能直接复用。目标分支 schema v1 的 phase
14～17 已经分别是 `Resolve`、`ResolveWait`、`ResolveInvalidate` 和
`ResolveCopy`，而来源分支把相同 ID 重新用于 `Atomic`、
`ClockBaseline`、`OrchestrationReplay` 和 `FinalDrain`。原样搬运会让旧
trace 被静默误解。

### 8.2 编译开关和 profile 正交化

map 模式只由编译期 `PTO_FDWIC_SHARED_MAP=0/1` 决定。DFX profile 与 map
模式正交组合，不再维护 private-only profile：

| profile | 主要开关 | 用途 |
| --- | --- | --- |
| `off` | 所有 C01/C03 开关关闭 | 功能和性能基准 |
| `swimlane` | `PTO_FDWIC_TRACE_ENABLED=1` | 原有 phase trace |
| `atomic` | trace + atomic level | C01 atomic site/等待聚合 |
| `perf-clock` | `PTO_FDWIC_PERF_CLOCK=1` | C03 每核 Submit 整窗 |
| `perf-clock-kernel` | 再开 kernel 聚合 | Submit 窗内 linked-kernel 粗分解 |
| `submit-pmu-none` | `PTO_FDWIC_SUBMIT_PMU=1` | C03 A5 PMU 整窗 |
| `submit-pmu-<phase>` | 再指定 phase ID | 每个 ELF 只采一个局部阶段 |

约束如下：

- `perf-clock`、`submit-PMU` 与普通/atomic trace 互斥；
- `perf-clock` 与 `submit-PMU` 互斥；
- atomic trace 只有在 trace 编译开关开启时才能启用；
- profile 不能改变 `PTO_FDWIC_SHARED_MAP`，同一 profile 分别构建
  private/shared ELF；
- build cache key、ELF marker、输出文件名和报告 provenance 都必须包含
  map 模式、profile、phase ID 和 schema version，发现运行时/ELF 模式
  不一致时直接失败；
- `PTO_FDWIC_TRACE_ENABLED=0` 的 ELF 中不得残留 atomic record、
  PollBatch 或 PMU phase 边界符号。

### 8.3 新 swimlane schema

建议新 schema 从 v5 起步，明确与目标 v1 和来源 v4 隔离：

1. 保留目标分支 phase 0～17 的编号和含义；
2. 从 18 开始追加 `Atomic`、`ClockBaseline`、
   `OrchestrationReplay`、`FinalDrain`、winner/loser 及 shared 专用
   phase；最终编号只在 M03～M08 取舍和非 DFX runtime 收敛后冻结；
3. 采用来源 v4 已验证的 32-byte record，把 core/lane/block 拓扑移到
   64-byte per-core state，以便继续保留每核 64K record 而不扩大设备
   分配；
4. header/core state 至少记录：
   - schema version、record bytes、records per core；
   - `PTO_FDWIC_SHARED_MAP`；
   - profile/trace level；
   - 本次实际 AIC/AIV/worker 拓扑；
   - event count、dropped count；
   - atomic logical calls、physical rows 和 PollBatch rows；
   - build identity 或可与 manifest 对应的 hash；
5. host 先验证 magic/version/record bytes/map mode/profile/topology，再读
   records；任何不一致都 fail closed；
6. offline converter 显式保留目标归档 v1、来源归档 v3/v4 和新 v5 的
   独立读取路径，不能猜测字段布局；host runtime 只接受本次 ELF 声明的
   schema。转换后的基础十列 JSON/CSV 形状保持稳定，避免下游工具全部
   重写；
7. exclusive analyzer 保留目标分支四个 Resolve phase，并为
   `OrchestrationReplay`、`FinalDrain` 建 parent interval。shared 子阶段
   只能在对应 parent 内出现，parent residual 由 analyzer 计算，不能用
   重叠区间重复计时。

### 8.4 C01 atomic site 迁移

#### 8.4.1 保留与扩展的 site

来源分支已有 28 个 private site，编号 0～27 应保留，便于复用现有
analyzer、UT 和历史数据：

```text
StartupIncrement, StartupPoll, FatalPoll, FatalSet,
ClaimMax, FaninFlagLoad,
CompletionVendExchange, CompletionFlagExchange,
FrontierInitialLoad, FrontierFlagLoad, FrontierMax,
HeapFrontierLoad, HeapVendLoad,
ReplayDoneIncrement, ReplayDonePoll,
WonSlotClaimMax, WonRemainingExchange,
WonLaneResetExchange, WonLaneDepositExchange,
WonStatePublishExchange, WonAnyPublishExchange,
WonAnyLoad, WonStateLoad, WonLaneClaimExchange,
WonLaneReleaseExchange, WonRemainingFetchSub,
WonStateClearExchange, WonDrainedLoad
```

shared 需要从 28 以后追加独立 site。下面是需要覆盖的语义集合，最终编号在
非 DFX runtime 冻结后生成，不能先按来源 private 枚举硬套：

| shared 位置 | 需要区分的 atomic 语义 |
| --- | --- |
| `SharedOutputCell::published` | 快速 probe load、等待 load、publish exchange |
| `SharedOutputCell::last_writer` | load、exchange、fetch-max |
| shared heap | cursor reserve fetch-add、wrap padding fetch-add、vend fetch-add |
| `SharedRegionMap` | bucket head load、sentinel wait load、claim exchange、publish exchange、high-water fetch-add |
| prepared deps | `deps_prepared` publish exchange、wait load |
| joint launch | expected fetch-add、drained fetch-add、expected load、drained load |
| final barrier | 以最终接受的 M04/现有协议分别标记 arrival、release 和 poll |

`insert_lock` 等当前没有生产调用者的 helper 不应伪造成“已覆盖的活跃
site”。迁移前先确认它们是删除、保留但静态未使用，还是确有调用，再决定
是否分配 site。

每个 site 的 schema 元数据必须显式给出：

- atomic op：Load、Exchange、FetchAdd、FetchMax 或 FetchSub；
- 返回值是否影响后续控制流；
- 是否允许进入 PollBatch；
- 适用模式：common、private-only 或 shared-only；
- 所属 wait region/phase。

#### 8.4.2 PollBatch 需要脱离 site ID

来源实现同时要求 site ID 小于 32，并直接用 `1U << site_id` 构造
PollBatch mask。shared site 加入后必然超过 32，这个限制必须拆开：

- record 的 `aux` 继续保存稳定的 16-bit site ID；
- 只有可批处理的等待 site 分配紧凑 `poll_batch_index`；
- batch mask 使用该紧凑 index，而不是 raw site ID；若可批处理 site
  超过 32，再把 mask 升为 64 bit 或拆成多个 word；
- schema/tool 同时发布 `site_id -> poll_batch_index` 映射；
- 所有状态迁移成功的 RMW 仍单独记录，只有幂等失败重试或纯观察 load
  可以批处理；
- phase/parent interval 结束、site 切换和真正状态迁移前必须关闭当前
  PollBatch，保证 logical call 数精确。

shared 中应重点建立的 wait region 包括 output publication、region bucket
sentinel、`deps_prepared`、joint drain、heap/frontier reuse、final barrier
和 BlockWon。报告必须同时闭合：

```text
logical atomic calls
  = direct physical rows
  + 所有 PollBatch 中的精确 poll_count
```

#### 8.4.3 shared 语义不能被 wrapper 改写

来源分支在 `api_glue.h` 中直接把部分普通 atomic helper 换成 trace
wrapper。目标 shared 模式的 `fatal_set()` 当前刻意返回 false；若机械换成
来源的 `FatalPoll` load，会在 shared 下新增真实 atomic 读取并改变控制流。

因此 wrapper 分层必须是：

- mode-specific helper 先保持原有语义；
- 只在该模式原本会执行 atomic 时记录对应 site；
- `set_fatal()` 的真实 exchange 可以记录，shared `fatal_set()==false`
  不伪造一次 load；
- DFX 关闭时 wrapper 内联回原 helper，不能增加额外 load、branch 或
  cache invalidate；
- AICPU 初始化/owner 的 host atomic 不进入 AICore swimlane site。

迁移时应增加静态检查：AICore 生产路径中的 atomic 调用必须来自已登记
wrapper 或明确 allowlist，避免 shared 新增 atomic 漏记。静态检查只负责
“是否分类”，动态 closure 负责“是否真正按协议执行”。

#### 8.4.4 I-cache 保护

来源 C01 曾因 inline record/PollBatch 扩大热路径而产生 I-cache 回退，随后
才把 cold record 路径 out-of-line，并补了少量 cold branch hint。这部分是
C01 完整性的一部分，但需要在目标协议上重做：

- record reserve、overflow、PollBatch flush 走 cold/out-of-line；
- private loser、shared loser、wrong-role、alloc non-candidate 分别检查
  code layout，不能复制 compete-first 的 hint；
- 比较 DFX-off、phase-only、atomic 三类 ELF 的 text size、关键符号和
  PA 配对性能；
- 若 DFX-off ELF 相对迁移前仍改变热路径，C01 阶段不得通过。

### 8.5 C03 perf-clock 与 submit-PMU 迁移

#### 8.5.1 先补齐对称的 Submit 生命周期

来源 private runtime 的 winner/loser 都有明确结束路径；目标 shared 的
loser wrapper 主要返回 symbolic refs，不再调用原 private loser replay。
如果直接套来源 hooks，shared loser、wrong-role worker 和 alloc
non-candidate 会启动窗口却不关闭，PMU 报告无法闭合。

在 DFX 编译开启时，需要给当前 API 增加对称、但 DFX-off 为零成本的观测
边界：

```text
presubmit / alloc begin
    -> wrong-role 或 non-candidate：对应 completion hook
    -> winner：winner completion hook
    -> loser：shared loser completion hook
```

每个 core 串行 replay，本地 profiling state 可以保存当前 task/window；
不应为了 DFX 扩大正常构建中的 `SubmitToken`。linked kernel 仍由统一
pause/resume 排除，避免把 vector/cube 执行混入 scalar control。

#### 8.5.2 shared phase 重新定义

来源 phase 中可以保留机制，但不能沿用 private PA 的固定次数公式：

- `ArgBuild` 在 shared 下只发生于 winner，应改名或定义为
  `WinnerArgBuild`，边界为 presubmit 返回到 winner submit 入口；
- `LoserReplay` 拆成 `PrivateLoserReplay` 和 `SharedLoserReturn`；
- wrong-role 早退单列低成本 phase，且必须关闭 Submit window；
- `PrepareMap` 在 shared 下应分为 `SharedDepsPrepare` 和
  `SharedDepsWait`；
- shared materialize 至少要能区分 heap reserve、output publish；
- region-intent 路径至少区分 lookup/register；
- 保留 `Resolve`、`ResolveWait`、`ResolveInvalidate`、
  `ResolveCopy`；
- joint/co-owned 路径需要 joint handoff/drain phase；
- `EmptyBracket` 只用于测量 PMU 边界自身开销，不代表业务 phase。

建议首批 shared phase 集合为：

```text
RoleFilter / Presubmit
Claim
WinnerArgBuild
Materialize
SharedHeapReserve
SharedOutputPublish
Fanin
SharedRegionLookup
SharedRegionRegister
SharedDepsPrepare
SharedDepsWait
Register
Resolve / ResolveWait / ResolveInvalidate / ResolveCopy
WinnerBuildControl
AllocCompleteControl
SharedLoserReturn
EfDrainControl
JointDrainControl
FinalDrainControl
SubmitTransition
```

具体 phase 可以先合并粗粒度边界，再由 C01 atomic 分析结果决定是否拆细；
不能一开始就在一个 ELF 同时打开全部 phase。每次 `submit-PMU` 构建只选择
一个 phase。

#### 8.5.3 拓扑和调用次数必须动态闭合

来源 `submit-PMU` 的 host/report 写死：

```text
num_cores = 96
aic_cores = 32
aiv_cores = 64
```

目标 runtime 的 `RUNTIME_MAX_WORKER` 是 108，并且 smoke 的 `MaxBd` 应使用
系统实际支持的最大 block 数。迁移后：

- AICPU owner 从本次 `layout[]`/runtime topology 得到实际参与的
  AIC/AIV/physical subcore 集合；
- header、bitmap、configured/restored count 和 triplet closure 都使用
  实际拓扑，不能把 96 写入 JSON 模板；
- owner 配置任一核失败时设置 AICPU abort 状态，禁止 worker 在半配置 PMU
  状态下继续运行；
- 采集结束必须逐核 readback、restore，并验证 active-after-restore 为 0；
- buffer 仍按最多 108 个 worker 预留，但报告只接受本次参与者；
- block dimension 1、36 和当前设备支持的 `MaxBd` 都需要定向验证。

PA “每 batch 四个 Kernel + 一个 Alloc”的公式只能留在 PA case/report
层。通用 runtime 依据 manifest/runtime counters 闭合以下动态不变量：

```text
每核 replayed submit 数
候选 / claim attempt 数
winner / loser / wrong-role 数
shared phase 调用数
output publication / region operation 数
全局每个 task/role 恰有一个 winner
所有开始的 Submit window 都已关闭
```

#### 8.5.4 buffer 与结果身份

`swimlane`、`perf-clock` 和 `submit-PMU` 可以复用 Runtime 到设备的 profiling
buffer plumbing，但三者的 header/version/record layout 必须独立，不能靠
同一指针猜测类型。设备 allocation 需保留原始指针和 64-byte 对齐后的
地址，释放原始 allocation。

每份结果至少携带或可追溯到：

- simpler commit；
- PTO ISA commit；
- `PTO_FDWIC_SHARED_MAP`；
- profile 和 phase ID；
- schema version 与完整 compile definitions；
- host/AICore ELF hash；
- workload/case、batch、block dimension 和实际 topology。

atomic 与 PMU 不在同一 ELF 同时采集。工具通过上述 build identity 和相同
workload signature 关联两次独立运行，不能把不匹配的捕获合并成一份报告。

### 8.6 主要代码影响面

| 层次 | C01 | C03 |
| --- | --- | --- |
| common ABI | `swimlane_types.h`、site/phase/schema | `perf_clock.h`、`submit_pmu_types.h` |
| AICore | atomic wrappers、PollBatch、phase 边界、cold record | window/phase hooks、counter read、kernel pause/resume |
| AICPU | buffer 注册/清零、拓扑 metadata | PMU owner 配置、readback、restore、abort |
| host runtime | trace level、buffer ownership、strict export | profile 选择、结果校验、动态 topology |
| build | trace-off/atomic profile | perf-clock/PMU profile 与 cache identity |
| tools/tests | schema、converter、exclusive analyzer、PollBatch UT | PMU report、provenance/closure UT、A5 定向验证 |

这也是 C01/C03 必须放在最后的具体原因：它们横跨 common ABI、AICore、
AICPU、host、构建和工具链，并且其观测边界依赖前面所有 runtime 取舍。

## 9. 按影响面从小到大的迁移阶段

以下是执行顺序框架，不把尚待确认的 M03～M08 自动纳入实现。某一项被
确认不接收时，跳过对应阶段；不能因此把后面的 DFX 阶段提前到仍未收敛的
功能阶段之前。

每个阶段只允许包含一类可解释的改动。每阶段结束均执行第 10 节的
shared/private 八组强制门禁和本阶段定向验证；任一失败立即停止，不在同一
工作树继续叠加下一阶段。

### 9.1 非 DFX 阶段

| 阶段 | 内容 | 影响面 | 本阶段额外验证 |
| --- | --- | --- | --- |
| N0 | 固定基线、编译器/PTO ISA、建立验证脚本和 DFX-off 性能基线 | 无产品改动 | 目标分支原始 tip 的八组门禁；记录 ELF/hash/PA 数据 |
| N1 | M07 中确认需要的 host strip、text extraction、BGEMM/include 构建兼容项 | 构建/host，无调度协议变化 | 干净构建、对应构建失败复现转通过、八组门禁 |
| N2 | M09 shared 测试改用 `PTO_FDWIC_SHARED_MAP`；M08 若确认则只加入改写后的 `mix_coown` 用例 | 测试层 | private/shared 分别编译；旧 ring oracle 已删除；1C+2V onboard |
| N3 | M05 `has_pending_won()` 已有 gate 的缺失 guard | 单个 AICore 查询点 | gate off 不产生 BlockWon load；gate on 功能不变 |
| N4 | M03 private heap H1 首圈快速路径 | private heap 局部热路径 | private 首圈/wrap/backpressure；shared 二进制与行为不受影响 |
| N5 | M06 legacy topology fallback | AICPU 注册与异常环境 | topology 正常/缺失、count 相等/不等四种组合；错误组合继续失败 |
| N6 | M04 G16 final barrier | 全 worker 最终排空协议 | private/shared，Bd1/Bd36/MaxBd，BlockWon/joint drain，超时和退出 closure |
| N7 | 非 DFX 收敛点 | 只做审计，不再增加功能 | 全量门禁、ABI/layout/static assert、DFX-off PA 配对基线 |

顺序依据：

- N1/N2 不改变 runtime 调度语义；
- N3 是单点 guard；
- N4 只影响 private heap；
- N5 触及 AICPU 注册但只在 topology 缺失时生效；
- N6 改变所有 worker 的最终同步协议，属于非 DFX 中影响最大的一项；
- N7 之后冻结 atomic site 和 PMU phase，避免 DFX 反复重编号或移动边界。

N2 中所有 shared 判断必须使用 `PTO_FDWIC_SHARED_MAP`。Python runner 可以
根据明确的 build manifest/case 参数选择 ELF，但不能恢复
`PTO_DIST_TENSORMAP_MODE` 运行时选择。

### 9.2 最后一组：DFX 阶段

只有 N0～N7 中所有实际被接受的阶段通过后，才开始下面的 DFX 尾部：

| 阶段 | 内容 | 影响面 | 本阶段额外验证 |
| --- | --- | --- | --- |
| D0 | M01 整体引入 `tests/atomic_probe`，校验 archive/manifest；不启用 runtime 插桩 | 测试、文档和历史资产 | 文件/manifest 完整性；适用的独立 probe；八组 runtime 门禁 |
| D1 | 新 v5 schema、32B record、per-core metadata、profile gate、host strict parser；所有新记录路径编译关闭 | common ABI、host、build | v1/v3/v4/v5 parser UT；mode/profile mismatch fail；DFX-off ELF 无新符号 |
| D2 | C01 来源 0～27 site、wrapper、PollBatch 和 cold/out-of-line 机制，先覆盖 private/common 路径 | AICore 热路径、host/tools | PollBatch 精确计数；private site closure；shared common 路径不回归 |
| D3 | C01 追加 shared site、shared wait region、mode-aware analyzer | shared AICore 热路径 | output/heap/region/deps/joint/final-barrier closure；无未分类 atomic |
| D4 | C03 `perf-clock` 和可选 kernel aggregate；补齐 shared winner/loser/wrong-role/non-candidate 对称 completion hook | Submit 生命周期、低扰动计时 | 两模式窗口全部关闭；kernel ticks/calls 与 Submit 窗闭合 |
| D5 | C03 `submit-PMU-none`：owner、整窗 counter、readback/restore、abort、动态 topology | AICPU + AICore + host | A5 Bd1/Bd36/MaxBd；owner 全配置/恢复；失败注入；无半配置运行 |
| D6 | C03 private/common 单 phase ELF 和报告 | 多 profile 构建、private phase | 每个 ELF 仅一个 phase；边界/call count/PMU counter closure |
| D7 | C03 shared phase 与 shared 动态公式 | shared phase 热路径、报告 | winner/loser/wrong-role、output/region/deps/resolve/joint 各路径闭合 |
| D8 | DFX 最终收敛 | 全工具链 | atomic 与 PMU 分次配对采集、provenance 对齐、observer-overhead 审计、最终八组门禁 |

D0 只迁移独立实验资产，不把历史二进制、原始数据或 lazy-lambda 样本当作
当前 runtime 功能测试。`ascendc/`、`ccec/`、`cpu/` 是硬件/编译器/环境
probe，按第 10.8 节执行，不在每个 DFX 阶段重复。

D1 的 schema 和 buffer plumbing 必须先证明在 DFX-off 下没有行为/布局
回归，D2 才能落 atomic call-site。D2/D3 必须先于 D5～D7，因为 PMU
scalar-window 扣减需要最终 atomic site 的 `result-used` 分类。

### 9.3 每阶段提交边界

每个阶段至少形成一个可独立回退、可独立验证的提交；若一个阶段包含
common ABI、device 和 host 三层，应继续拆为下列顺序，但只有整阶段全部
通过才算完成：

1. schema/types/tools UT；
2. host/build plumbing；
3. device runtime hooks；
4. case/报告适配；
5. 干净重编译和完整门禁。

禁止以“先把来源提交全部 cherry-pick，再集中修冲突”的方式实施。每次只
从来源提交提取当前阶段需要的语义；来源提交中夹带的 compete-first、
lazy-lambda、旧 96-worker 公式或旧 shared ring 必须在进入工作树前剔除。

## 10. 每阶段验证方法和范围

本节把
[`tests/atomic_probe/fdwic_shared_atomic_merge_plan.md`](../../tests/atomic_probe/fdwic_shared_atomic_merge_plan.md)
中的验证方法和范围复制并改写为本计划的“阶段”术语。后续执行以本节为
准，不需要依赖原计划的 cherry-pick 顺序。

### 10.1 验证总原则

- 编译通过不是功能验证结论。
- 每个阶段根据代码影响和新增/修改用例确定额外验证范围。
- 每完成一个阶段，或完成一次已确认的冲突/适配修改后，必须重新编译
  runtime 和 kernels，再运行本阶段门禁。
- 禁止使用上一阶段的 `.o`、`.so` 或 kernel 结果代替重新编译。
- 验证前确认源码、runtime 二进制和 PTO ISA revision 一致。
- 需要排除缓存时，隔离 `build/cache`、`build/lib` 及相关目标文件，确保
  `.o`、`.so` 和 kernel 重新生成。
- 不修改项目配置来绕过失败。
- onboard 验证必须通过 `task-submit` 运行。
- 性能配置、profiling 输出和功能 golden 不能互相替代。

### 10.2 Onboard 与 `task-submit` 强制规则

所有访问 A5 设备的操作都必须由 `task-submit` 触发；源码检查、清理、
编译和 A5Sim 在其外执行。

普通功能用例统一使用：

```text
--timeout 90
--max-time 90
--device auto
--device-num 1
```

超时视为异常。stress 或性能任务只有在记录轮数、预计时长和理由后才能
单独提高超时。

默认执行 A5 arch precheck。当前环境没有 `npu-smi` 时，只能按已有用户
授权跳过 precheck；该豁免不跳过 onboard 用例。A5 每一轮都必须单独调用
一次 `task-submit`，禁止用一次设备任务包住多轮 pytest 循环。A5Sim 不使用
`task-submit`。

### 10.3 根目录验证脚本和环境要求

在项目根目录建立两个本地验证脚本：

```text
build_runtimes.sh
run_tests.sh
```

它们是本地验证工具，不属于产品提交。创建后执行：

```bash
chmod 700 build_runtimes.sh run_tests.sh
```

脚本只在根目录不存在时创建一次；切换阶段、commit 或 private/shared 模式
时重新运行，不重复生成脚本。每个环境必须重新确认：

1. `PTO_ISA_ROOT` 指向项目 `build/` 之外的独立、干净 checkout；
2. `PTO_ISA_COMMIT` 默认固定为 CI 使用的
   `ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8`，CI pin 改变时两个脚本
   同步更新；
3. 根目录 `.venv` 存在，包含 pytest、nanobind 和用例依赖；
4. 外部 shell 已设置并加载 CANN：

   ```bash
   export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
   source "$ASCEND_HOME_PATH/set_env.sh"
   ```

   `set_env.sh` 把变量规范化为实际版本目录是正常行为；
5. 使用 `set_env.sh` 提供的 `PATH`、`LD_LIBRARY_PATH`、
   `CMAKE_PREFIX_PATH`、`ASCEND_OPP_PATH`，不另维护一套 CANN 路径；
6. 项目根目录和 `python/` 加入 `PYTHONPATH`，已有值保留在后；
7. `gcc` 和 `g++` 都必须是真正的 15，不能只设置 `CXX` 或只检查
   `g++`；
8. A5 onboard 必须经 `task-submit`；没有 `npu-smi` 只影响已授权的
   precheck 豁免，不影响 onboard 门禁。

### 10.4 `build_runtimes.sh`

脚本只负责环境检查、旧产物隔离和干净编译，不运行测试，也不占用
`task-submit`。同一 commit、map 模式、编译器、PTO ISA revision 和 DFX
profile 下可供多个用例复用；任一维度变化都必须重新构建。

将 `/path/to/pto-isa` 替换为当前环境的真实路径：

```bash
#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <expected-simpler-commit> <shared|private>" >&2
    exit 2
fi

expected_simpler_commit="$1"
map_mode="$2"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv_path="$repo_root/.venv"

case "$map_mode" in
    shared) shared_map="1" ;;
    private) shared_map="0" ;;
    *)
        echo "Invalid map mode: $map_mode (expected shared or private)" >&2
        exit 2
        ;;
esac

cd "$repo_root"
test "$(git rev-parse --show-toplevel)" = "$repo_root"
source "$venv_path/bin/activate"

: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH must be set}"
source "$ASCEND_HOME_PATH/set_env.sh"

export PTO_ISA_ROOT="/path/to/pto-isa"
export PTO_ISA_COMMIT="ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8"
export CXXFLAGS="-DPTO_FDWIC_SHARED_MAP=$shared_map"
export PYTHONPATH="$repo_root/python:$repo_root${PYTHONPATH:+:$PYTHONPATH}"

test "$(git rev-parse HEAD)" = \
    "$(git rev-parse "$expected_simpler_commit^{commit}")"
test "$(gcc -dumpversion | cut -d. -f1)" = "15"
test "$(g++ -dumpversion | cut -d. -f1)" = "15"
test -d "$PTO_ISA_ROOT/.git"
test "$(git -C "$PTO_ISA_ROOT" rev-parse HEAD)" = "$PTO_ISA_COMMIT"
test -z "$(git -C "$PTO_ISA_ROOT" status --short)"

stamp="$(date +%Y%m%d-%H%M%S)"
build_backup="$repo_root/../simpler-build-backup-$stamp"
binding_backup="$repo_root/../simpler-binding-backup-$stamp"

if [ -e "$repo_root/build" ]; then
    test ! -e "$build_backup"
    mv "$repo_root/build" "$build_backup"
fi
mkdir -p "$repo_root/build"

if compgen -G "$repo_root/python/_task_interface*.so" >/dev/null; then
    test ! -e "$binding_backup"
    mkdir -p "$binding_backup"
    mv "$repo_root"/python/_task_interface*.so "$binding_backup"/
fi

test -z "$(
    find "$repo_root/build" \
        -type f \
        \( -name '*.o' -o -name '*.so' -o -name CMakeCache.txt \) \
        -print -quit
)"

strip_shim="$(mktemp -d /tmp/simpler-aarch64-strip.XXXXXX)"
trap 'rm -rf "$strip_shim"' EXIT
ln -s \
    "$ASCEND_HOME_PATH/tools/hcc/aarch64-target-linux-gnu/bin/strip" \
    "$strip_shim/strip"
export PATH="$strip_shim:$PATH"

nanobind_dir="$(python -m nanobind --cmake_dir)"
cmake \
    -S "$repo_root" \
    -B "$repo_root/build/python-binding" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$(command -v python)" \
    -Dnanobind_DIR="$nanobind_dir"

cmake \
    --build "$repo_root/build/python-binding" \
    --target _task_interface \
    -j"$(nproc)"

test -n "$(
    find "$repo_root/python" \
        -maxdepth 1 \
        -type f \
        -name '_task_interface*.so' \
        -print -quit
)"

python "$repo_root/simpler_setup/build_runtimes.py" \
    --lib-dir "$repo_root/build/lib" \
    --cache-dir "$repo_root/build/cache" \
    --platforms a5sim a5 \
    --pto-isa-commit "$PTO_ISA_COMMIT"
```

DFX 阶段需要在此脚本增加一个显式 profile 参数，并把对应 compile
definitions 纳入 cache identity；不能继续只靠外部 `CXXFLAGS` 猜测 profile。
在 D1 落地前仍使用上面的两参数权威版本。

### 10.5 `run_tests.sh`

脚本接收一个或多个 pytest target。`a5sim` 直接执行 pytest；`a5` 在普通
shell 完成环境和 revision 检查，只把最终 pytest 命令交给
`task-submit`，避免环境准备占用设备锁。

`PTO_ISA_ROOT` 和 `PTO_ISA_COMMIT` 必须与 `build_runtimes.sh` 完全一致：

```bash
#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <expected-simpler-commit> <shared|private> <a5|a5sim> <repeat-count> <pytest-target> [<pytest-target> ...]" >&2
    exit 2
fi

expected_simpler_commit="$1"
map_mode="$2"
platform="$3"
repeat_count="$4"
shift 4
pytest_targets=("$@")
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv_path="$repo_root/.venv"

case "$map_mode" in
    shared) shared_map="1" ;;
    private) shared_map="0" ;;
    *)
        echo "Invalid map mode: $map_mode (expected shared or private)" >&2
        exit 2
        ;;
esac

if ! [[ "$repeat_count" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid repeat count: $repeat_count (expected a positive integer)" >&2
    exit 2
fi

cd "$repo_root"
test "$(git rev-parse --show-toplevel)" = "$repo_root"
source "$venv_path/bin/activate"

: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH must be set}"
source "$ASCEND_HOME_PATH/set_env.sh"

export PTO_ISA_ROOT="/path/to/pto-isa"
export PTO_ISA_COMMIT="ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8"
export CXXFLAGS="-DPTO_FDWIC_SHARED_MAP=$shared_map"
export PYTHONPATH="$repo_root/python:$repo_root${PYTHONPATH:+:$PYTHONPATH}"

test "$(git rev-parse HEAD)" = \
    "$(git rev-parse "$expected_simpler_commit^{commit}")"
test "$(gcc -dumpversion | cut -d. -f1)" = "15"
test "$(g++ -dumpversion | cut -d. -f1)" = "15"
test -d "$PTO_ISA_ROOT/.git"
test "$(git -C "$PTO_ISA_ROOT" rev-parse HEAD)" = "$PTO_ISA_COMMIT"
test -z "$(git -C "$PTO_ISA_ROOT" status --short)"

common_pytest_args=(
    "${pytest_targets[@]}"
    --platform "$platform"
    -p no:xdist
    -v
    --require-pto-isa
    --pto-isa-commit "$PTO_ISA_COMMIT"
)

case "$platform" in
    a5sim)
        for ((iteration = 1; iteration <= repeat_count; iteration++)); do
            echo "[run_tests] a5sim iteration $iteration/$repeat_count"
            python -m pytest "${common_pytest_args[@]}"
        done
        ;;
    a5)
        onboard_command=(
            env
            "PTO_ISA_ROOT=$PTO_ISA_ROOT"
            "PTO_ISA_COMMIT=$PTO_ISA_COMMIT"
            "CXXFLAGS=$CXXFLAGS"
            "PYTHONPATH=$PYTHONPATH"
            "$venv_path/bin/python"
            -m pytest
            "${common_pytest_args[@]}"
            --device __TASK_DEVICE__
        )
        printf -v onboard_command_string '%q ' "${onboard_command[@]}"
        onboard_command_string="${onboard_command_string//__TASK_DEVICE__/'$TASK_DEVICE'}"
        printf -v shell_command 'bash -ic %q' "$onboard_command_string"
        for ((iteration = 1; iteration <= repeat_count; iteration++)); do
            echo "[run_tests] a5 iteration $iteration/$repeat_count"
            task-submit \
                --timeout 90 \
                --max-time 90 \
                --device auto \
                --device-num 1 \
                --run "$shell_command"
        done
        ;;
    *)
        echo "Invalid platform: $platform (expected a5 or a5sim)" >&2
        exit 2
        ;;
esac
```

DFX profile 引入后，`run_tests.sh` 必须读取并核对 build manifest 中的
profile/map mode/ELF hash，不允许测试命令与已构建二进制不一致。

### 10.6 标准功能调用

下面以 shared、当前 HEAD 为例：

```bash
COMMIT=$(git rev-parse HEAD)

./build_runtimes.sh "$COMMIT" shared

./run_tests.sh "$COMMIT" shared a5sim 1 \
  examples/a5/fully_distributed_within_core/simple_orch_smoke \
  examples/a5/fully_distributed_within_core/shared_symbol_smoke \
  examples/a5/fully_distributed_within_core/submit_dependency_smoke \
  --manual include

./run_tests.sh "$COMMIT" shared a5 1 \
  examples/a5/fully_distributed_within_core/simple_orch_smoke \
  examples/a5/fully_distributed_within_core/shared_symbol_smoke \
  examples/a5/fully_distributed_within_core/submit_dependency_smoke \
  --manual include

./run_tests.sh "$COMMIT" shared a5sim 1 \
  examples/a5/fully_distributed_within_core/paged_attention_unroll/\
test_paged_attention_unroll.py

./run_tests.sh "$COMMIT" shared a5 1 \
  examples/a5/fully_distributed_within_core/paged_attention_unroll/\
test_paged_attention_unroll.py
```

private 阶段把以上五条命令中的 `shared` 全部替换为 `private`，并先做新的
干净构建。新增用例可以作为额外 pytest target 传入；多个 target 可共享
同一次匹配的 runtime 构建。

功能验证约束：

- smoke 固定为 `simple_orch_smoke`、`shared_symbol_smoke` 和
  `submit_dependency_smoke`，必须传 `--manual include`；
- PA 只验证 Case1；
- PA Case3 是已知非必过项，不纳入本次阶段门禁；
- PA A5Sim 必须运行真实 kernel 和真实 golden；
- PA 功能命令不得添加 `--use-example-exec-time` 或 `--skip-golden`；
- `--use-example-exec-time` 只允许用于 sim 性能分析，其结果不能作为功能
  正确性证据；
- `repeat-count` 必须是正整数，A5 每轮单独提交一个 `task-submit`。

### 10.7 重新编译规则和八组强制门禁

以下任一情况发生后必须重新运行匹配 map mode/profile 的干净构建：

- 进入一个新迁移阶段；
- 解决并落地一次冲突；
- 修改 runtime、platform、orchestration 或 incore kernel；
- 在 shared/private 宏之间切换；
- 切换分支或 commit；
- 修改编译器、编译参数、DFX profile 或 PTO ISA revision。

不得使用 `python -m pip install '.[test]'` 建立阶段构建：它会探测并构建
无关 a2a3sim，并可能把 A5 专用宏传播到错误平台。pytest 会动态编译
orchestration/incore kernel，因此 runner 的 map mode/profile 必须与本次
runtime 构建完全一致。

每个阶段必须全部通过以下八组门禁：

1. shared smoke 全量 A5Sim；
2. shared smoke 全量 A5 onboard；
3. shared PA Case1 A5Sim；
4. shared PA Case1 A5 onboard；
5. private smoke 全量 A5Sim；
6. private smoke 全量 A5 onboard；
7. private PA Case1 A5Sim；
8. private PA Case1 A5 onboard。

执行顺序：

1. 隔离旧产物；
2. `PTO_FDWIC_SHARED_MAP=1`，shared 干净构建；
3. shared smoke A5Sim、smoke A5、PA A5Sim、PA A5；
4. 再次隔离 shared 构建产物；
5. `PTO_FDWIC_SHARED_MAP=0`，private 干净构建；
6. private 的同四组门禁；
7. 记录八组结果和两次构建日志。

`MaxBd` 使用当前系统支持的最大 block 数，不写死 36。任何一组失败时停止
迁移，保留当前 commit 和构建现场分析；不得继续下一阶段，也不得用后续
阶段结果覆盖当前失败。

每阶段报告必须记录：

- simpler commit；
- 本阶段采用的来源 commit/代码片段；
- 是否发生冲突及采用的解法；
- GCC/G++ 版本；
- PTO ISA revision；
- map mode、DFX profile、phase ID、schema version、ELF hash；
- shared/private 是否分别完成全量干净重编译；
- 八组强制门禁的逐项 pass/fail；
- 本阶段新增或修改用例的逐项 pass/fail；
- DFX 阶段的 closure、dropped records、owner restore 和 observer
  overhead 结果。

### 10.8 新增用例和对照 worktree

每个阶段必须检查该阶段是否新增或修改测试：

```bash
git diff --name-status <stage-base> HEAD -- \
  examples/ tests/
```

所有新增/修改用例都是本阶段门禁的一部分。默认在 shared/private 两种干净
构建状态下分别执行；只有源码或设计明确声明仅适用于一种模式时才能跳过
另一种，并在报告中记录依据和风险。

先在当前集成分支直接验证。只有下列结果无法判定时才建立只读对照
worktree：

- 用例依赖来源 commit 当时的目录、脚本或环境，无法直接运行；
- 用例失败，但无法判断是 shared 适配问题还是来源 commit 自身限制；
- 用例所需硬件、工具链或外部条件当前不可用；
- 结果与来源文档矛盾，需要确认来源 commit 的真实基线。

普通编译失败、已经明确的功能失败或为了节省切换时间，不构成创建对照
worktree 的理由。对照规则：

1. 固定到本阶段采用的确切来源 commit，不能用来源分支最新 HEAD 代替；
2. 只读对照、编译和运行，不在其中修改或提交；
3. 使用独立 `.venv`、`build/cache`、`build/lib`，不复用集成目录产物；
4. 编译器、PTO ISA、平台、设备、参数和 pytest flags 与集成分支一致；
5. onboard 对照仍通过 A5 precheck 和 `task-submit`；
6. 记录来源/集成两个 commit 的结果后移除 worktree。

示例：

```bash
SOURCE_COMMIT=<current-source-commit>
CONTROL_DIR=../simpler-source-control-${SOURCE_COMMIT:0:12}

git worktree add --detach "${CONTROL_DIR}" "${SOURCE_COMMIT}"
cd "${CONTROL_DIR}"
python3 -m venv --system-site-packages .venv
```

完成对照后：

```bash
cd /path/to/simpler
git worktree remove "${CONTROL_DIR}"
```

若来源 commit 同样失败或无法运行，报告必须给出用例、来源 commit、双方
错误、编译器/PTO ISA/平台/命令、缺失条件和未覆盖风险。若来源通过而集成
失败，视为迁移回归，立即停止下一阶段。

### 10.9 `atomic_probe` 的验证范围

Atomic/PMU 阶段根据下列文档确定定向验证：

```text
tests/atomic_probe/pa_scheduler/PA-atomic情况分析.md
tests/atomic_probe/pa_scheduler/swimlane_opt_anal.md
tests/atomic_probe/icache_miss_usage_guide.md
tests/atomic_probe/perf_opt_record.md
tests/atomic_probe/a5_fdwic_atomic_swimlane_repo.md
```

`tests/atomic_probe/ascendc/`、`ccec/`、`cpu/` 及封装它们的
`test_atomic_probe.py` 只验证硬件、编译器或环境 atomic/PMU 能力，不是
runtime 迁移功能门禁：

- 不纳入每阶段 runtime 门禁；
- 不要求在 private/shared 下重复执行；
- 只在 D0 资产接收检查、硬件能力复核或环境归因时运行；
- 历史 raw data、预编译 artifact 和 lazy-lambda A/B/C 样本只校验
  manifest/可读性，不重解释为当前分支性能结果。

每个 DFX 阶段明确区分：

- atomic 语义和依赖正确性；
- PA scheduler 功能正确性；
- shared/private smoke 回归；
- A5Sim 功能；
- A5 onboard 功能；
- 仅在必要阶段执行的 performance/profiling。

### 10.10 C01 定向门禁

D1～D3 除八组功能门禁外至少验证：

1. schema/ABI：
   - v1、v3/v4 和 v5 parser 分流；
   - record/core-state size 与 64-byte partition alignment；
   - phase/site/op/result-used/poll-batchable 元数据穷举；
   - map mode/profile/version mismatch 必须失败；
2. PollBatch：
   - 单 site、多 site、成功 RMW、flush 边界、count overflow；
   - `logical = direct rows + batch poll_count`；
   - `dropped == 0` 才能作为完整 capture；
3. private：
   - 来源 0～27 site 的 op 和控制流保持一致；
   - startup、fatal、claim、fanin、completion、frontier、heap、
     final barrier、BlockWon 均能闭合；
4. shared：
   - output publication/last-writer；
   - shared heap reserve/wrap/vend；
   - region lookup/sentinel/claim/publish/high-water；
   - deps prepare/wait；
   - joint expected/drained；
   - 最终采用的 final barrier；
5. 静态分类：
   - AICore 生产路径无未登记 atomic；
   - AICPU init/owner 和明确 allowlist 不误计入；
6. observer effect：
   - DFX-off ELF 无 C01 record/PollBatch 符号；
   - phase-only 不产生 atomic rows；
   - atomic profile 产生完整 rows；
   - 比较 text size、关键函数 layout 和 PA 配对结果；
   - 若 DFX-off 性能/布局发生无法解释的变化，阶段失败。

A5Sim 负责功能和 schema closure；`return-ready`、真实 atomic contention、
I-cache 和物理时序结论必须来自 A5 onboard。

### 10.11 C03 定向门禁

D4～D7 除八组功能门禁外至少验证：

1. profile 互斥：
   - trace/atomic、perf-clock、submit-PMU 不可错误共存；
   - 每个 phase ELF 只包含一个 selected phase；
   - build manifest 与 ELF marker 一致；
2. perf-clock：
   - 每核首个 begin、末个 end 都存在且有序；
   - observed submit count 等于 expected；
   - linked-kernel calls/ticks 不超过整窗，residual 非负；
   - shared winner、loser、wrong-role、alloc non-candidate 均关闭窗口；
3. PMU owner：
   - 按实际 topology 配置，不写死 96；
   - Bd1、Bd36、MaxBd 的 physical IDs/role/triplet 闭合；
   - selector readback 正确；
   - configured/restored 数相等，active-after-restore 为 0；
   - 配置失败注入会触发 abort，worker 不开始半配置采集；
4. counter：
   - I-cache miss 不大于 request；
   - vector/cube/scalar/total 的关系满足所选窗口定义；
   - counter 未达到 wrap 风险阈值；
   - phase sidecar 可重建且位于 Submit 整窗内；
   - linked-kernel pause/resume 和 atomic return-ready 扣减全部闭合；
5. shared phase：
   - 逐一触发 RoleFilter、winner/loser、heap/output、region、deps、
     resolve、joint 和 final drain；
   - 每个开始边界都有结束边界；
   - 动态 winner/loser/role 数来自 runtime manifest/counters；
   - PA 特有“五次 submit/batch”公式只在 PA report 层验证；
6. 结果身份：
   - simpler/PTO ISA commit、map mode、profile、phase、schema、ELF hash、
     workload、block dimension、topology 完整；
   - atomic 与 PMU 捕获只有 identity/workload 匹配时才能关联；
7. observer effect：
   - `off`、`perf-clock`、`submit-pmu-none`、单 phase 分别配对采集；
   - 性能结论单独记录轮数、设备和统计方法；
   - 不以 profiling 数值代替功能 golden。

### 10.12 编译器和 PTO ISA 固定结论

A5Sim PA Case1 已有环境结论：

- GCC 13 在 shared/private 和多种优化配置下稳定出现约 `1.9` 的 golden
  mismatch；
- 真正 GCC/G++ 15 下，相同历史代码功能用例能够通过；
- 因此 GCC 13 的 PA A5Sim 失败作为独立环境问题处理，不归因于迁移，但也
  不能用它替代阶段门禁。

A5Sim 功能基线使用真正 GCC/G++ 15。PTO ISA 固定为：

```text
ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8
```

运行前确认 checkout HEAD 精确匹配且工作区干净。

### 10.13 最终收敛

全部已确认阶段完成后：

1. 检查最终提交顺序和每个提交的变更范围；
2. 确认未纳入临时调试、环境配置、缓存和本地验证脚本；
3. 完成 private/shared 全量 smoke；
4. 完成 PA Case1 A5Sim 与 A5 onboard；
5. 完成每个 DFX 阶段对应的 `tests/atomic_probe` 定向验证；
6. 单独报告性能环境、参数和结果，不把它当成功能结论；
7. 汇总未覆盖风险、已知限制及 CI/其他硬件待补验证。

### 10.14 恢复任务时的检查点

每次恢复迁移先执行：

```bash
git status --short
git branch --show-current
git log --oneline --decorate -n 20
```

然后确认：

1. 当前位于约定的 integration 分支；
2. 没有未完成的 cherry-pick/rebase/冲突；
3. 本文档、本地验证脚本和环境配置没有混入产品提交；
4. 已重新计算尚未完成的阶段和确切来源 commit；
5. shared/private/DFX profile 与现存构建产物一致；不一致则先干净重建；
6. 从下一个未完成阶段继续，遇到冲突或门禁失败立即停止并记录分析。
