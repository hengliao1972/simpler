# A5 PA Submit 性能优化全过程记录

## 1. 文档目的与状态约定

本文持续记录 A5 FDWIC Paged Attention `Case1` 的 Submit 调度性能优化过程。
**当前性能优化目标只包含真实 simpler PA 路径**；
`tests/atomic_probe/pa_scheduler` standalone 只保留为已经完成的历史方法验证、
负结果和模型边界证据，不再作为当前待办或下一阶段优化对象。目标是让后续真实
PA 优化能够从可复核的源码、提交、实测数据和产物继续推进，而不是仅保留最终
结论。

本文使用以下状态标签：

| 标签 | 含义 |
| --- | --- |
| **[已保留]** | 已进入当前代码路径，并完成与风险相称的正确性和性能验证 |
| **[已撤回]** | 做过实现或实验，但因语义不成立、性能回退或证据不足而不再保留 |
| **[观察工具]** | 用于建立测量能力，本身不是业务性能优化 |
| **[历史证据]** | 对当时源码和构建有效，不能自动代表当前 HEAD |
| **[受限]** | 已确认存在平台、模型或验证覆盖边界 |
| **[设计中]** | 只有经过源码核对的方案，尚无完成提交或性能结论 |

记录更新至 2026-07-21。当前分支为
`fdwic-swimlane-exclusive`，本阶段开始时 HEAD 为 `9f6140c1`，跟踪
`origin/fdwic-swimlane-deps`。后续每完成一个合理阶段，都应按第 12 节模板更新
本文并形成一条带详细中文说明的本地提交。

更细的专题资料分别见：

- [A5 FDWIC Paged Attention 安装与复现指南](a5_fdwic_atomic_swimlane_repo.md)；
- [PA 原子操作与优化记录](pa_scheduler/PA-atomic情况分析.md)；
- [PA 调度器独立复现与泳道使用指南](pa_scheduler/PA调度器独立复现与泳道使用指南.md)；
- [FDWIC 泳道排他分区与闭合分析](pa_scheduler/swimlane_opt_anal.md)；
- [I-cache Miss 采集与分析指南](icache_miss_usage_guide.md)。

## 2. 固定范围、环境与权威性能口径

### 2.1 本轮范围

- 真实用例：
  `examples/a5/fully_distributed_within_core/paged_attention_unroll/test_paged_attention_unroll.py`；
- Case：`Case1`；
- runtime：`fully_distributed_within_core`；
- 平台：A5Sim 用于功能和控制流回归，真实 A5 用于性能结论；
- A5 工作核：32 个 AIC、64 个 AIV，共 96 个 worker；
- 工作量：256 batch，每 batch 依次包含 Alloc、QK、SF、PV、UP 五个 task；
- 每核 Submit：`256 * 5 = 1280`，全局 Submit：`96 * 1280 = 122880`。

本文不把其他 runtime、其他 PA Case、A2/A3、整段 pytest wall time 或整个 device
任务耗时混入 Submit 性能结论。

### 2.2 已验证环境

| 项目 | 固定值 |
| --- | --- |
| 设备 | `/dev/davinci0`，Ascend950PR_958b |
| Driver | `7.0.t9.0.B798`，ascendhal `7.35.23` |
| CANN | 用户目录下 9.1.0 weekly 20260708 |
| CCEC | clang 15.0.5 |
| Python | `/home/q00473782/.venv`，Python 3.12.3 |
| PyTorch | 2.6.0+cpu |
| pytest | 7.4.4 |
| GCC 15 | `/home/q00473782/.local/gcc-15/root`，15.0.1 |
| PTO-ISA | `ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8` |

非交互 shell 不能假设自动读取 `.bashrc`。正式复测应显式 source 用户 CANN，
激活本用户 `.venv`，并显式设置用户 GCC 15 的 `PATH`、`LD_LIBRARY_PATH` 和
`CXX`。完整命令以安装复现指南为准。

### 2.3 权威性能定义

本文所说的“完整 Submit 时间”默认指：

```text
全部 worker 中最早的第一个 Submit 开始
    到
全部 worker 中最晚的最后一个 Submit 结束
```

它排除启动屏障和 FinalDrain，不等于 pytest wall time，也不等于整个 kernel
launch 的 device wall time。必须同时区分三类时间：

1. **跨核完整 Submit 时间**：上述约 5 ms 的全局墙钟范围，是候选是否保留的最终
   性能口径；
2. **逐核或全核累计工作量**：某个 span 在 96 核上的时长求和，用于描述工作分布，
   不是 96 核共同形成的墙钟；
3. **PMU total/core**：每个物理子核 PMU gate 内的周期数，是单核周期工作量，
   也不是跨核完整 Submit 时间。

泳道原始时间使用 1 ns/tick 的 `SYS_CNT`。本机 cold/warm 校准得到 PMU 频率约
1.65 cycles/ns：ALL/AIC/AIV 分别为 1.649844/1.650062/1.649731。PMU cycle
换算不能反过来改变 `SYS_CNT` 的 1 ns/tick 定义。

## 3. 起始基线

### 3.1 环境打通与 5.6 ms 基线

**[已保留] `657313c9` — `fix(a5): enable paged attention on legacy drivers`**

该提交完成 A5 平台 block 数解析和经过双重校验的 flat OCCUPY 回退，使当前旧
Driver 环境能够运行目标 PA。A5Sim 和 A5 Case1 均通过，真实 A5 level-1
泳道复现的首末 Submit 为 **5.642245 ms**；仓库更早的历史参考为
5.577570 ms。

基线产物：

```text
outputs/TestPagedAttentionUnroll_Case1_20260717_023809/
  merged_swimlane_atomic_load.json
  l2_swimlane_records.json
```

这是第一轮业务优化的比较起点，不代表后续加入观察代码后的 ELF。

## 4. 提交时间线与阶段结论

下表按当前分支中的逻辑推进顺序列出与本轮工作直接相关的提交。详细证据和适用
边界见后续各节；纯文档重命名没有单独列项。

| 日期 | 提交 | 类型 | 阶段摘要 |
| --- | --- | --- | --- |
| 2026-07-17 | `657313c9` | 正确性基础 | 打通旧 Driver 上的 A5 PA |
| 2026-07-17 | `ce89fae2` | 编译探针 | 建立嵌套 lambda 与跨 TU 基线 |
| 2026-07-17 | `e3b748b4` | 真实 PA 优化 | 跳过 BlockWon 无效轮询并复用参数 mask |
| 2026-07-17 | `a3f5ecc2` | 编译探针 | 隔离 inline/noinline Submit 边界 |
| 2026-07-17 | `290dbda0` | standalone | 建立 CCEC/AscendC/CPU 独立 PA 模型 |
| 2026-07-18 | `76df85ce` | 观察工具 | 隔离验证 atomic 与 I-cache PMU 归类 |
| 2026-07-18 | `04ec9b95` | 真实 PA/standalone 优化 | HeapGuard 首圈跳过冗余 atomic load |
| 2026-07-18 | `3d174a08` | 负结果 | 记录并撤回 fanin 顺序实验 |
| 2026-07-18 | `3d0aaea7` | 观察工具 | 建立 standalone 每核 scalar PMU |
| 2026-07-18 | `67407cc4` | 观察工具 | 细分 fanin/frontier 动态 atomic 次数 |
| 2026-07-18 | `8deefdef` | 观察工具 | 修正 PMU owner 活跃子核与恢复闭环 |
| 2026-07-18 | `13431a23` | 观察工具 | 建立 standalone 逐 atomic 泳道 |
| 2026-07-18 | `c93bd65d` | 观察工具 | 接通 standalone 自包含 PMU owner |
| 2026-07-18 | `640efe50` | 观察工具 | 建立单次 I-cache miss 标尺 |
| 2026-07-18 | `99971ac1` | 观察工具 | 建立 standalone Submit 全区间 PMU |
| 2026-07-18 | `c6daaeb7` | 文档证据 | 固化当时 atomic/PMU 观察口径 |
| 2026-07-18 | `5274945b` | 合并 | 吸纳 I-cache miss 标尺改动 |
| 2026-07-18 | `e66001ff` | standalone | CCEC 接入真实 Cube/Vector 负载 |
| 2026-07-18 | `9aeda0dd` | standalone | AscendC 接入真实 Cube/Vector 负载 |
| 2026-07-18 | `1d3a374a` | standalone | CPU 补齐对等算术负载 |
| 2026-07-18 | `0c9cebc3` | standalone | 增加非均匀布局诊断 |
| 2026-07-18 | `7bb118a8` | standalone | 默认切换为 `real-compute/6,28,4,1` |
| 2026-07-18 | `cbaf7c60` | 观察工具 | 真实 PA 接入 atomic/PollBatch 泳道 |
| 2026-07-18 | `44199a54` | 观察工具 | 固化 96 核 I-cache 分组分析 |
| 2026-07-18 | `187e54bc` | 观察工具 | 完善 standalone atomic 合并泳道 |
| 2026-07-18 | `5c2d39b8` | 观察工具 | 拆分 standalone `swimlane/submit-pmu` 构建 |
| 2026-07-18 | `8430f418` | 观察工具 | 增加历史 EfDrain 局部 PMU |
| 2026-07-19 | `7466e6f5` | 观察工具 | 完善局部 PMU、HTML 并退役 WaitForSlot phase |
| 2026-07-19 | `6caa269c` | 观察工具 | 收敛 standalone 排他 span 与分析器 |
| 2026-07-19 | `d2d8ce25` | standalone 优化 | 将低频 winner 调整为冷分支 |
| 2026-07-19 | `cafa9ca5` | 真实 PA 优化 | 恢复真实 PA loser 热路布局 |
| 2026-07-19 | `44367971` | 观察代码优化 | 外提 atomic 冷路径，消除取指布局回退 |
| 2026-07-19 | `14c2429f` | 文档证据 | 分离 atomic 与 I-cache 专题记录 |
| 2026-07-19 | `dbb95bb5` | 观察工具 | 将排他泳道迁入真实 FDWIC |
| 2026-07-19 | `911ecf9a` | 文档证据 | 固化排他 span 和性能分布分析 |
| 2026-07-20 | `ba4334d1` | 独立对照 | 交付 A/B/C compete-first/lazy 对照 |
| 2026-07-20 | `0d08c437` | standalone 优化 | 主路采用 compete-first eager |
| 2026-07-20 | `2899cc35` | 真实 PA 重构 | 接入 compete-first eager begin/finish |
| 2026-07-20 | `84e9d6d0` | 文档证据 | 建立真实 PA Submit 全过程记录 |
| 2026-07-20 | `8d5aa686` | 观察工具 | 建立真实 PA 低扰动 perf-clock 基线 |
| 2026-07-20 | `9f6140c1` | 观察工具 | 收口真实 PA 业务与 atomic 合并泳道 |
| 2026-07-21 | `faa370d9` | 观察工具 | 建立真实 PA `submit-pmu-none` 全窗证据链 |
| 2026-07-21 | `2a7dccee` | 观察工具 | 建立真实 PA `arg-build` 单阶段 PMU |
| 2026-07-21 | `81a1f382` | 观察工具 | 量化真实 PA running bracket 的空区间观察指纹 |
| 2026-07-21 | 本阶段提交 | 观察工具 | 建立真实 PA `materialize` 单阶段 PMU |

### 4.1 真实 PA 第一轮 atomic 与前端优化

#### 4.1.1 跳过单 lane 图无效 BlockWon 轮询并复用参数掩码

**[已保留] `e3b748b4` — `Update: 优化 A5 FDWIC Submit 热路径`**

该阶段包含两类改动：

1. 在本 worker 第一次见到 joint submit 之前，跳过无意义的 BlockWon 轮询；
2. 复用一次 tensor tag 扫描生成的 output/register mask，减少重复前端扫描。

PA Case1 全部 task 都是单 lane，因此第一项确定删除 Submit 内 **146944 次**
无效 `atomic_load(any_pub)`；A5 上该封装实际为 `atomicAdd(addr, 0)`，并非普通
load。删除位置主要落在每次 Submit 开头的 EfDrain 和高频 loser 的公共尾部，
没有删除 Claim。

实测结果：

| 版本 | 首末 Submit |
| --- | ---: |
| 初始基线 | 5.642245 ms |
| joint polling skip 三轮中位数 | 5.171330 ms |
| 加 register mask 三轮中位数 | 5.186679 ms |
| 加 output/register masks 三轮中位数 | **5.115620 ms** |

最终相对初始基线减少 **0.526625 ms，约 9.33%**；最好单轮为
**5.096685 ms**。kernel 累计时长没有随之缩短，证据支持收益来自调度前端，
而不是计算 kernel 变快。

最终最好产物：

```text
outputs/TestPagedAttentionUnroll_Case1_20260717_055638/
  merged_swimlane_best_joint_poll_skip_arg_masks_5.096685ms.json
```

mask 的独立收益只有三轮方向性证据，不能从组合版本中严格拆出因果比例；当前保留
的是经过整体正确性回归的组合改动。

#### 4.1.2 用普通 load 和 NOP 替换 atomic 的定位实验

**[已撤回] 未形成保留提交**

为确认 `atomic_load` 的成本量级，曾临时替换为 `ld_dev()+nop(100)` 和
`ld_dev()+nop(10)`：

| 诊断变体 | 首末 Submit | 产物 |
| --- | ---: | --- |
| `ld_dev()+nop(100)` | 5.343592 ms | `outputs/TestPagedAttentionUnroll_Case1_20260717_035341/merged_swimlane_nop100.json` |
| `ld_dev()+nop(10)` | 5.401034 ms | `outputs/TestPagedAttentionUnroll_Case1_20260717_035954/merged_swimlane_nop10.json` |

普通 device load 加固定 NOP 不具备 atomic RMW 的同步、可见性和顺序语义，
因此这些样本只用于定位成本，源码修改已经撤回，不能作为可用优化方案。

#### 4.1.3 HeapGuard 首圈 fast path

**[已保留] `04ec9b95` — `perf(a5): 跳过HeapGuard首圈冗余原子读取`**

历史文档也使用同内容提交号 `2c3dd1e2`；当前分支可达 hash 为
`04ec9b95`。默认 256 MiB heap 下，逻辑 heap 尚未走完第一圈时不可能覆盖旧输出，
因此在原 fatal 检查之内直接返回，不读取 frontier/vend。PA Case1 的 Alloc、QK、
SF、PV 共确定消减 **1024 次 frontier atomic load**；跨圈 slow path 保持原协议。

真实 A5 十对结果：

| 指标 | 基线 | H1 | 相对变化 |
| --- | ---: | ---: | ---: |
| 中位数 | 5.142168 ms | 5.122320 ms | -0.386% |
| 均值 | 5.167064 ms | 5.146984 ms | -0.389% |
| p90 | 5.274224 ms | 5.229773 ms | -0.843% |

十对中 8 胜 2 负，配对变化中位数为 **-0.324%**。因此只能称为“确定减少
atomic，真实 PA 中心趋势小幅正向”，不能把 standalone 的大幅波动外推到真实
PA。最好样本为 5.098696 ms：

```text
outputs/TestPagedAttentionUnroll_Case1_20260717_173313/
  merged_swimlane_heapguard_first_lap_fastpath_5.098696ms.json
```

#### 4.1.4 fanin 检查顺序实验

**[已撤回] `3d174a08` — `文档(a5): 记录fanin顺序实验与回退结论`**

F1 在 standalone 中把有效 fanin producer 按 task id 降序排列。静态分析证明固定
调度状态下不会增加 ready load，实测 fanin load 也下降；但十对交错 A/B 中：

```text
Submit 中位数：4290.401 -> 4623.944 us，+7.774%
Submit 配对变化中位数：+6.439%
```

fanin load 减少没有转化为 Submit 收益，说明指令布局和 worker 到达时序的间接
变化不能忽略。候选代码已撤回，未迁移到真实 FDWIC。

### 4.2 standalone 建立与动态 atomic 取证

#### 4.2.1 建立三后端独立 PA 模型

**[观察工具] `290dbda0` — `test(a5): 增加独立 PA 调度性能复现用例`**

在 `tests/atomic_probe/pa_scheduler` 下建立 CCEC、AscendC、CPU 三后端，保留
Case1 五 task 拓扑、96 worker 回放、四分片 Claim、TensorMap、fanin、私有 ring、
WaitForSlot、HeapGuard、completion flag/vend/frontier 和最终 drain。此时 winner
计算仍以可控 NOP 为主，目标首先是闭合调度协议与 atomic 次数。

`ce89fae2` 和 `a3f5ecc2` 还建立了嵌套 lambda、跨 TU caller context 和 inline/
noinline 边界探针。这些是后续拆分 callback/finish 时的编译行为依据，不是 PA
本身的性能收益。

#### 4.2.2 HeapGuard 对等修改与压力回归

**[已保留] `04ec9b95` 同时修改 standalone 与真实 PA**

standalone 默认 256 MiB 配置、16 MiB 多圈压力和恢复默认值后的三后端回归均经过
语义检查。16 MiB CCEC/AscendC b256 确实进入 slow path；CPU b256 在 fast 版和
临时撤销 fast path 的原版中都超过观察时限，不能记为 H1 PASS 或 FAIL，详见
第 9 节。

#### 4.2.3 fanin/frontier 软件计数

**[观察工具] `67407cc4` — `测试(a5): 细分PA依赖与frontier原子计数`**

计数只写 worker 私有 `LocalStats`，结束时一次发布，不增加共享 atomic。b256
十轮动态基线为：

| 指标 | 中位数 |
| --- | ---: |
| fanin 总 load | 93201.5 |
| fanin not-ready load | 86675.5 |
| frontier FetchMax | 15365 |
| Submit+completion atomic ops | 203803.5 |

该结果确认主要动态项是 not-ready 重试和 frontier helping，而不是 ready 前缀。
软件计数扩大了 sidecar 并增加私有 scalar 增量，因此这一版的绝对 Submit 时间不能
与无计数 ELF 直接归因比较。原始日志：

```text
tests/atomic_probe/pa_scheduler/outputs/atomic_diagnostics/
  ccec_baseline_10_20260718_023551.log
```

### 4.3 scalar、atomic 与 I-cache 观察工具

以下提交建立证据链，但不应被写成业务性能优化：

| 提交 | 状态 | 主要内容 |
| --- | --- | --- |
| `76df85ce` | [观察工具] | 用隔离 probe 验证 atomic 与 I-cache 等待周期的 PMU 归类 |
| `3d0aaea7` | [观察工具] | 在 standalone 建立每核 scalar PMU 读取链路 |
| `8deefdef` | [观察工具] | 修正 PMU owner 的活跃物理子核配置和恢复闭环 |
| `c93bd65d` | [观察工具] | 接入自包含 Main AICPU Path-A owner，保存、配置并恢复 32 AIC+64 AIV 状态 |
| `13431a23` | [观察工具] | 给 standalone 增加逐 atomic 泳道、调用点和边界语义 |
| `640efe50` | [观察工具] | 建立隔离 cold/warm 单次 I-cache miss 一阶标尺 |
| `99971ac1` | [观察工具] | 建立每核完整 Submit PMU gate 和 PMU-only JSON |
| `c6daaeb7` | [历史证据] | 固化当时 atomic 与 Submit PMU 的边界和使用限制 |
| `5274945b` | [观察工具] | 合并单次 I-cache miss 标尺相关改动 |
| `44199a54` | [观察工具] | 固化 96 核 raw 复算、AIC/AIV 分组和多轮分析口径 |
| `187e54bc` | [观察工具] | 合并普通阶段与 atomic 泳道，并用 PollBatch 压缩等待轮询 |
| `5c2d39b8` | [观察工具] | 将 `swimlane` 与 `submit-pmu` 拆成独立重编译产物 |
| `8430f418` | [观察工具] | 增加历史 EfDrain 局部 PMU 归因 |
| `7466e6f5` | [观察工具] | 增加 Materialize/Register、HTML 报告，退役 WaitForSlot 局部 PMU |

#### 4.3.1 atomic 时间边界

逐 atomic 观察不插入 DSB，也不强制原本不消费返回值的 Exchange/FetchAdd 变成
等待返回型操作：

- 真正消费返回值的 FetchMax、claim exchange 使用 `return_ready`；CCEC 通过
  返回值地址依赖后再读取 `SYS_CNT`，尽力使结束点晚于返回值可用；
- 不消费返回值的 Exchange/FetchAdd 使用 `source_issue`，只表示源码发射包围区间，
  不能解释成 atomic 已在全局完成；
- PollBatch 表示一个等待区间内多次逻辑轮询的整体时间和精确调用数，不能把其
  duration 除成单次 atomic latency。

#### 4.3.2 PMU owner 的响应校准

自包含 owner 的 empty、100000 scalar NOP、2×100000 scalar NOP 三组，96 核
PMU total 中位数约为 214、56568、112994 cycles。这只证明 gate 响应和工作量
近似倍增，不表示同数值的纳秒，也不是 PA Submit 基线。

#### 4.3.3 单次 I-cache miss 标尺

隔离微基准的结果为：

| 规模 | ALL 中位数 | 轮间范围 |
| --- | ---: | ---: |
| 64 trials/core × 10 轮 | 86.596 ns/miss | 86.532～86.792 |
| 128 trials/core × 5 轮 | 89.629 ns/miss | 89.615～89.648 |

统一使用 **90 ns/miss** 作为单核串行等效的一阶感性标尺。它不是实际 Submit
墙钟损失，不能把 `miss * 90 ns` 从约 5 ms 中直接减掉。

### 4.4 standalone winner 负载迁移到真实 Cube/Vector

| 提交 | 状态 | 主要内容 |
| --- | --- | --- |
| `e66001ff` | [已保留] | CCEC QK/PV 接入真实 Cube matmul，SF/UP 接入真实 Vector add/mul |
| `9aeda0dd` | [已保留] | AscendC 接入对等 Cube/Vector 负载并修正布局 |
| `1d3a374a` | [已保留] | CPU 增加对等算术和统一路由，用于功能对照 |
| `0c9cebc3` | [已保留] | 增加非均匀输入，验证转置、ND/NZ、stride 和输出布局 |
| `7bb118a8` | [已保留] | 三后端默认切到 `real-compute/6,28,4,1`，NOP 只保留显式兼容入口 |

最终同泳道口径下，standalone CCEC b256 五轮为：

```text
5.002413 / 4.875193 / 4.968894 / 4.992477 / 4.876282 ms
中位数 4.968894 ms
```

当时真实 PA 三轮中位数为 5.115620 ms，同口径差 146.726 us，约 2.87%。
这说明 standalone 达到“独立复现约 5 ms 调度”的目标，不证明两份实现的代码布局、
数据流和跨核时序完全一致。历史 raw：

```text
tests/atomic_probe/pa_scheduler/outputs/performance_gap_20260718/
  standalone_ccec_real_b256_raw.json
```

### 4.5 真实 PA atomic 泳道与 PollBatch

**[观察工具] `cbaf7c60` — `工具(a5): 接通真实PA atomic泳道与精确轮询聚合`**

该提交把 standalone 验证过的边界迁入真实 FDWIC，schema-v3 使用 32 B 紧凑
record、28 个稳定 site 和五类 atomic op，并只在允许的等待区聚合 PollBatch。
当时真实 A5 PA Case1 level-4 通过：

```text
115200 次逻辑 atomic
110006 条物理 Atomic
340 条 PollBatch
dropped_records = 0
```

逻辑调用、物理记录和 PollBatch 必须满足 producer 定义的闭合公式。level-4
结果用于观察 atomic 分布，不能替代关闭诊断后的性能基线。

### 4.6 schema-v4 排他 span 与 raw 规模控制

#### 4.6.1 standalone 排他 span

**[观察工具] `6caa269c` — `工具(a5): 收敛Submit观测边界与排他泳道分析`**

主要变化：

- 增加 OrchestrationReplay、FinalDrain 等父区间；
- 旧 Build/Replay/Alloc lap 改为真实 WinnerBuild/AllocComplete 尾动作；
- standalone loser 没有真实计算动作，不再为 121600 个 loser 生成伪
  `LoserReplay` record；
- Submit 内和 Submit 间未覆盖时间由 converter/analyzer 使用已有边界离线求差，
  不增加设备 record、字段或时间戳；
- merged 只保留 Perfetto 必需字段，raw 仍是权威数据；
- Kernel 必须唯一落入 EfDrain、WinnerBuild、AllocComplete 或 FinalDrain，越界、
  多重归属、孤儿 Kernel 或 dropped 非零都拒绝结果。

阶段性 b256：

```text
tests/atomic_probe/pa_scheduler/outputs/
  pa_scheduler_swimlane_20260719_114815_617346/ccec/
```

该轮首末 Submit 为 5.360061 ms，raw 839526 条、`dropped=0`；raw/merged
分别约 55.79/88.78 MB，全部父子关系和整数闭合通过。

#### 4.6.2 standalone winner 冷分支

**[已保留] `d2d8ce25` — `优化(a5): 将低频winner调整为冷分支`**

每个 task 只有一个 winner，绝大多数 worker 都走 loser。给 Alloc 和普通 Submit
的重型 winner 分支增加低概率布局提示，不移动边界、不改变协议。b256 同观察口径：

| 指标 | 基线 | 候选 | 变化 |
| --- | ---: | ---: | ---: |
| 全局首末 Submit | 5.360061 ms | 5.278401 ms | -1.52% |
| Submit 尾部未覆盖时间 | 41008786 cycles | 27155661 cycles | -33.78% |
| 完整逐核 Submit 区间累计 | 500448909 cycles | 483335683 cycles | -3.42% |

关闭泳道后又做候选—基线—候选 ABA，每组五轮：候选中位数分别为
3.665017/3.715385 ms，基线为 3.988115 ms，分别快 8.10%/6.84%。同时记录
了 `.text` 体积增长，避免只看速度不看取指代价。候选泳道：

```text
tests/atomic_probe/pa_scheduler/outputs/
  pa_scheduler_swimlane_20260719_123520_660296/ccec/
```

#### 4.6.3 真实 PA loser 热路布局恢复

**[已保留] `cafa9ca5` — `优化(a5): 恢复真实PA的loser热路布局`**

真实 PA 对等标记两处低频 winner 分支。当前 atomic 观察版基线三轮中位数从
5.631038 ms 降为 5.192087 ms，减少 0.438951 ms，约 7.80%。但历史
pre-atomic 中位数已经是 5.115620 ms，因此这一步的准确含义是：

> 恢复 atomic 观察代码接入后发生的热路布局回退，而不是在旧 5.1 ms 基线上
> 新增 7.8% 业务收益。

#### 4.6.4 外提 atomic 观察冷路径

**[已保留] `44367971` — `优化(a5): 外提atomic泳道冷路径消除取指回退`**

分两步推进：

1. 将 direct Atomic 的 record 发布外提为设备端共享 `noinline` 冷函数，保留
   `begin -> atomic -> end` 在原 wrapper；三轮中位数为 5.096506 ms；
2. 保留 PollBatch 的内联 level 快速门，只把 level-4 命中后的十类遍历和落盘
   外提到共享 slow 函数。

最终代码尺寸变化：

| 产物 | winner 冷路后 | atomic 冷路径外提后 |
| --- | ---: | ---: |
| AIC/AIV `dist_engine .text` | 347536 / 357112 B | 66768 / 67120 B |
| AIC/AIV `dist_submit_impl` | 100860 / 103676 B | 18812 / 18872 B |

真实 A5 level-1 正式三轮：

```text
4.821897 / 4.890447 / 4.752956 ms
中位数 4.821897 ms
```

相对 5.192087 ms 再下降 7.13%。该结论只能归为“消除未执行诊断代码的大量复制和
热路布局回退”；当时没有同时采集专用 I-cache PMU，不能继续写成确定的 miss
降幅。

level-4 能力复核：

```text
outputs/TestPagedAttentionUnroll_Case1_20260719_135629/

Atomic 物理记录闭合：107608 = 115309 - 8056 + 355
dropped_records = 0
```

`14c2429f` 随后把 I-cache、代码布局和 PMU 经验集中到 I-cache 指南，PA atomic
文档只保留 atomic 语义、次数和边界。

#### 4.6.5 排他泳道迁入真实 FDWIC

**[观察工具] `dbb95bb5` — `Support: 将 Submit 排他泳道分析移植到 FDWIC`**

真实路径获得与 standalone 同类的父区间、真实尾动作、离线未覆盖时间和整数闭合
分析，Atomic、Kernel 等 overlay 不参与排他加和。`911ecf9a` 进一步固定各 span 的
业务含义、AIC/AIV 分布和全核工作量与墙钟末端的区别。

### 4.7 compete-first eager

#### 4.7.1 A/B/C 三份独立 standalone 对照

**[历史证据] `ba4334d1` — `验证(pa): 交付 compete-first/lazy 三版独立对照`**

三版分别为：

- A：原始 Materialize-first、Submit 外 eager 构参；
- B：EfDrain、Claim 前移，全体 worker 在 Claim 后同步 eager 构造完整参数；
- C：与 B 相同控制流，仅让 loser 跳过 input/scalar thunk。

72 次独立 host 启动中 A/B/C 各 24 个样本，均通过语义与后处理门禁。去异常的
相邻配对结果：

| 比较 | 配对变化 | 结论 |
| --- | ---: | --- |
| B 相对 A | -206.270 us，-5.214% | 22/22 有效块更快；收益是 compete-first、split/outlining 与布局的组合 |
| C 相对 B | +1.503 us，+0.040% | 双方各 11/22；低于 5% 门槛，不支持 lazy 收益或回退 |

因此只推进 B 的 compete-first eager，不把 C 的 lazy 视为优化，也没有为 C 启动
I-cache PMU 对比。

#### 4.7.2 standalone 主路采用 compete-first eager

**[已保留] `0d08c437` — `优化(pa): standalone采用compete-first eager提交流程`**

当前时间线变为：

```text
EfDrain -> Claim -> 同步 eager callback 构参
        -> Materialize -> PrepareMap
        -> Fanin/Register -> WinnerBuild或AllocComplete -> Submit结束
```

Claim 与 Materialize 之间的构参时间使用现有边界离线求差，不新增 raw 字段。
关闭泳道的 b256 五轮中位数从 3889.180 us 降到 3735.032 us，减少
154.148 us，约 **3.9635%**。这证明收益在 standalone 主路复现，不承诺真实
PA 有同一比例。

compete-first 移植阶段的 standalone b1 历史布局证据：

```text
tests/atomic_probe/pa_scheduler/outputs/
  pa_scheduler_swimlane_20260720_092021_1729726/ccec/
    l2_swimlane_records.json
    merged_swimlane.json
    swimlane_exclusive_analysis.json
```

#### 4.7.3 真实 PA 接入 compete-first eager

**[已保留] `2899cc35` — `重构(pa): 真实路径接入compete-first eager提交流程`**

真实路径新增显式 begin/finish API 和 32 B 同步 ticket，同时保留旧 one-shot API
及原顺序，未把新语义强加给其他调用方。所有 worker 在 Claim 后仍完整构参，未采用
lazy 跳过。

真实 A5 level-1 三轮：

| 路径 | 三轮 | 中位数 |
| --- | --- | ---: |
| compete-first 最终版 | 4.843652 / 4.809211 / 4.805443 ms | 4.809211 ms |
| 原路径历史基线 | 4.821897 / 4.890447 / 4.752956 ms | 4.821897 ms |

中位数减少 12.686 us，约 0.263%，两组三轮波动区间重叠。因此当前结论是
**真实性能基本持平**，不是稳定的 0.263% 收益。保留该接口是因为阶段顺序和业务
边界更清晰，并为后续精确取证提供基础。

真实 level-4 权威件：

```text
outputs/TestPagedAttentionUnroll_Case1_20260720_104406/
  l2_swimlane_records.json
  merged_swimlane.json
```

该轮包含 122880 个 Submit、945653 条事件、`dropped_records=0`，首末 Submit
为 5.066862 ms；atomic 闭合为：

```text
106355 = 109392 - 3361 + 324
```

它证明 compete-first 后的阶段、atomic 和离线加工一致，不是关闭观察后的净性能
样本。

## 5. 当前保留优化汇总

| 优化 | 真实 PA 状态 | 当前可成立的效果结论 |
| --- | --- | --- |
| 首个 joint 前跳过 BlockWon 轮询 | [已保留] | PA 单 lane Case1 确定删除 146944 次无效 RMW；与参数 mask 的组合将 5.642245 ms 降至 5.115620 ms 中位数 |
| output/register mask 复用 | [已保留] | 组合结果正向；独立收益只具方向性，不作精确拆分 |
| HeapGuard 首圈 fast path | [已保留] | 确定删除 1024 次 frontier atomic load；真实十对配对中位 -0.324% |
| 低频 winner 冷分支 | [已保留] | 主要恢复 atomic 观察接入后的 loser 热路布局回退 |
| atomic record/PollBatch 冷路径外提 | [已保留] | 大幅缩小热函数和 `.text`，level-1 三轮中位恢复到 4.821897 ms，同时保持 level-4 闭合 |
| compete-first eager begin/finish | [已保留] | standalone -3.9635%；真实 PA 三轮只能判为基本持平 |

这些结果来自不同历史阶段，不能把表中百分比相加得到“总收益”。当前真实路径已经
同时包含这些改动，后续基线必须从当前 HEAD 重新建立。

## 6. 当前观察能力

### 6.1 真实 PA `swimlane`：业务 span 与 atomic 合并观察

**[观察工具，已具备]**

- 普通业务阶段和 Atomic/PollBatch 位于同一 AIC/AIV scalar lane；
- schema-v4 以父区间和互斥子区间闭合，Kernel、Atomic 是不可加和 overlay；
- residual/未覆盖时间由离线工具使用已有相邻边界计算，不新增设备 record；
- raw 是权威数据，merged 只负责 Perfetto 可视化，exclusive analysis 负责整数闭合；
- `dropped_records != 0`、父子越界、Kernel 孤儿或 atomic 公式不闭合时，整轮无效。

`swimlane` 用于回答“时间落在哪个业务区域、atomic 调用次数和边界是否改变”，
不作为无观察性能基线，也不直接给出 I-cache stall。

### 6.2 standalone `submit-pmu`：历史能力

**[观察工具，历史版本已具备；不是当前待办]**

现有历史版本能在独立 CCEC ELF 中编译掉泳道和逐 atomic 记录，采集每物理子核
PMU total、scalar busy、I-cache request/miss，并生成 96 核 raw 和自包含 HTML。
历史 `none/claim/efdrain/materialize/register` 数据只对当时边界和各自 ELF 有效。

历史 b256 `none` 一轮记录：

| 指标 | AIC 每核 | AIV 每核 |
| --- | ---: | ---: |
| request | 408317.344 | 422480.609 |
| miss | 38664.344 | 55098.625 |
| 加权 miss/request | 9.4692% | 13.0417% |
| PMU total 等效时间 | 4527.942 us | 4294.748 us |
| scalar busy 等效时间 | 3602.744 us | 3396.667 us |

历史产物：

```text
tests/atomic_probe/pa_scheduler/outputs/
  submit_pmu_none_20260719_b256_final/
    submit_icache_raw.json
    submit_icache_report.html
```

该目录可能不在当前机器保留，且从不随 Git 提交。上述数据只用于说明已经验证过的
PMU owner、逐核 raw、AIC/AIV 分组和 HTML 方法，不是当前重采 standalone 的要求，
也不能把旧绝对值当成真实 PA 数据。当前工作只在真实 PA 上按当前 compete-first
代码和最新真实 span 建立新证据。

### 6.3 A5 当前不可获得的 `scalar_wait_ib_time`

**[受限]**

在本机 CANN 9.1、A5/DAV3510 上分别尝试 `PipeUtilization`、
`PipeUtilization,MemoryDetail` 和 `Default` 三种正式 `msopprof` 入口，生成的
CSV 和 A5 正式事件表均没有 `scalar_wait_ib_time` 或
`scalar_wait_time`。当前不能套用 A2/A3 的事件号或字段含义。

对应历史目录：

```text
tests/atomic_probe/pa_scheduler/outputs/wait_ib_official_msopprof_20260719_b1_probe2/
tests/atomic_probe/pa_scheduler/outputs/wait_ib_official_msopprof_20260719_b1_probe3_memory_detail/
tests/atomic_probe/pa_scheduler/outputs/wait_ib_official_msopprof_20260719_b1_probe4_default/
```

因此 `PMU total - scalar_busy` 只能叫“非 Scalar-busy 残余”，不能命名为 scalar
空闲、wait vector 或 I-cache stall。

## 7. standalone 历史证据与真实 PA 的边界

**[历史证据，不是当前待办]**

standalone 已完成其方法验证职责：证明多后端调度模型、真 Cube/Vector 负载、
atomic 泳道、PMU owner、I-cache 标尺和排他区间工具可以工作。当前不再继续优化、
扩充或重采 standalone；下列对等关系只用于解释为什么历史经验可以作为真实 PA
实施时的参考，以及哪些结论绝不能外推。

### 7.1 已对等部分

- 256 batch、Alloc/QK/SF/PV/UP 五 task 拓扑；
- 32 AIC + 64 AIV、每核完整回放 1280 次 Submit；
- 四分片 Claim 和固定 73728 次 Claim atomicMax；
- TaskArgs、Tensor、TaskPayload、DistSubmitCtx 的关键布局和 tag 扫描；
- TensorMap materialize、retire、lookup、insert、register mask；
- fanin、winner/loser、私有 ring slot、WaitForSlot、HeapGuard；
- completion flag、vend、frontier 和最终 drain；
- QK/PV 真 Cube、SF/UP 真 Vector 的受控计算工作量；
- 普通阶段、Atomic/PollBatch、Kernel placement 和最终状态闭合。

依赖不是按 task 名硬编码跳过：SF 依赖 QK，PV 依赖 SF，UP 去重后依赖 Alloc、
SF、PV；每 batch fanin 边数为 5，b256 全局为 1280。

### 7.2 尚未对等部分

- 真计算 workspace 使用统一受控输入，数值没有按真实 QK→SF→PV→UP 数据流串接；
- 该历史模型只覆盖 Case1 单 block group、`q_loop=1` 和全单-lane 图；
- joint/mixed、多 group、多 q-loop 和跨迭代更新没有被完整模拟；
- synthetic heap、独立 ELF 布局和 host 启动状态与真实 simpler 不同；
- Kernel span 包含 engine launch/completion wait wrapper，不等于纯 Cube/Vector 指令时间；
- standalone 没有真实 PA 的 loser replay 业务动作，不应为追求图形对称而伪造该 span。

历史推进中，standalone 曾用于先验证接口、边界、计数、控制协议和候选方向，再迁
真实 PA 做同构正确性与性能 A/B；它的收益比例始终不能直接外推。当前这些方法能力
已经完成验证，后续真实 PA 三证据链不再把新增 standalone 实现或复测设为前置门禁。

## 8. 当前真实 PA 的三条互不混算证据链

### 8.1 `perf-clock`

**[观察工具，已实现]**

该阶段已经建立真实 PA 的权威低扰动性能基线。最终构建只额外定义：

```text
PTO_FDWIC_PERF_CLOCK=1
PTO_FDWIC_TRACE_ENABLED=0
```

实现口径为：

- **保留 `PTO2_PROFILING` 及其拥有的公开 Arg 布局/ABI**；
- 编译期去除 FDWIC 普通泳道、逐 atomic 观察和平台 PMU 路径；
- 每核只保留第一个 Submit 起点、最后一个 Submit 终点和 Submit 次数；
- 不为每个 Submit 写 record；
- 继续复用现有 host/device header 传输，但 device header 固定只有 **6976 B**，
  `records_per_core=0`，没有逐事件记录；
- host hook 随该构建重新编译并导出，不复用普通诊断构建的旧 host 产物；
- 候选保留或撤回最终由该构建决定。

第一版曾尝试设置 `PTO2_PROFILING=0`，在编译期触发 Arg 布局/cacheline
`static_assert`，没有进入设备执行。该尝试已经完整撤回；这证明
`PTO2_PROFILING` 不只是可随意关闭的观察开关，不能为了减少诊断代码破坏公开 ABI。
最终方案只关闭上述 FDWIC 观察路径，不能表述为“所有 PTO2 profiling 已移除”。

构建身份也已做双向 ELF 审计：perf ELF 含
`dist_perf_clock_expect_submits` 标记，并且不含 FDWIC swimlane、atomic 观察和平台
PMU 符号；普通 level-4 ELF 含正常泳道/atomic 符号，但不含 perf-clock 设备符号。
普通构建还做了前后布局复核：旧 AICore cache 身份 `6c55004bc91e15f0` 与新
AICore cache 身份 `110ff0c62a3adcf7` 的 `.text` 均为 `0x31a50` B，两个 ELF 中 `.text` 的 96 条
FUNC symbol 记录在地址、尺寸、绑定、可见性和名称上完全一致；`.text` 仅有 AIC/AIV
两份 `aicpu_orchestration_entry` 各一处单字节从 `0xa2` 变为 `0xa9`，对应
`PTO2_SCOPE` 源码行号从 162 移到 169，原因是前置新增 7 行。没有观察到普通构建
新增函数或代码尺寸膨胀；这里也不声称工具已经给出完整指令反汇编一致性。

在完成第 2.2 节环境准备后，最小复现命令为：

```bash
source /home/q00473782/.venv/bin/activate
python -m pytest examples/a5/fully_distributed_within_core/paged_attention_unroll/test_paged_attention_unroll.py \
  --platform a5 --case CaseB1 --manual include \
  --fdwic-profile perf-clock --rounds 1 -s -v

python -m pytest examples/a5/fully_distributed_within_core/paged_attention_unroll/test_paged_attention_unroll.py \
  --platform a5 --case Case1 \
  --fdwic-profile perf-clock --rounds 1 -s -v
```

该 profile 必须独占：不要同时传 `--enable-l2-swimlane`、`--enable-pmu`、
`--use-example-exec-time` 或其他诊断开关；命令行门禁会直接拒绝混用。每次只允许
`--rounds 1`，多轮基线必须由独立 pytest 进程取得。该 profile 会按源码指纹自动
重编 AICore override，但不会代替安装流程重编 host runtime；新环境首次复现前必须
先重建 `libhost_runtime.so`，并确认三个 `fdwic_perf_clock_host_*` hook 已导出。

最小 B1 两次有效 A5 闭合如下；两次都是 96 个物理子核、每核恰好 5 次 Submit：

| 产物时间戳 | 完整 Submit |
| --- | ---: |
| `20260720_172436` | 75.347 us |
| `20260720_172615` | 73.716 us |

Case1 golden 在 `20260720_172820` 通过，96 核均为 1280 Submit，原始
`SYS_CNT` 为 **4,489,247 ticks = 4489.247 us**。JSON 浮点输出采用默认 6 位有效
数字，因此 Case1 量级显示为 `4489.25`，而 B1 仍可显示 `73.716`；后续精确分析
应以 raw tick 除以 1000 为准。

Case1 golden 通过后，另起五个独立进程并使用 `--skip-golden` 得到本阶段干净基线；
**以下五次不包含上述 golden 样本**：

| 产物时间戳 | raw ticks | 完整 Submit |
| --- | ---: | ---: |
| `20260720_173140` | 4,495,677 | 4495.677 us |
| `20260720_173224` | 5,808,500 | 5808.500 us |
| `20260720_173307` | 5,342,774 | 5342.774 us |
| `20260720_173350` | 4,752,765 | 4752.765 us |
| `20260720_173433` | 4,823,114 | 4823.114 us |

五次均为 96 核、每核 1280 Submit；中位数 **4823.114 us**，最小值
**4495.677 us**，最大值 **5808.500 us**。该分布是后续候选做独立进程、交错
A/B 的起点，不能只挑 4489 us 的最好值作为稳定基线。

还完成两项边界门禁：

- Case2 负测试实际得到每核 576 Submit，而当前期望值为 320；host 按 fail-closed
  拒绝结果且不生成成功 summary。它只证明计数不符时不会产出伪成功结论，不能用于
  评价 Case2 性能；
- 同一真实源码的普通 level-4 B1（`20260720_173738`）得到 480 个 Submit、完整
  Submit **85.653 us**、`dropped=0`，排他闭合 `PASS`。它证明 perf-clock 的首尾
  边界和逐核计数与普通泳道来自同一执行语义；由于两者是不同 ELF，不能用
  `85.653 - 73.716` 计算观察开销。

本阶段的源码、构建身份、B1、Case1、负测试和普通泳道同源边界已经闭合，并已完成
独立源码审阅。后续进入真实 `swimlane` 构建复核与 `submit-pmu-none`，不在该工具
阶段顺带铺开 PMU 代码。

### 8.2 `swimlane`

**[观察工具，已实现]**

保留普通阶段和 atomic 合并泳道，用于回答：

- 收益或回退可能落在哪个业务 span；
- atomic 逻辑调用、物理记录和 PollBatch 是否变化；
- Kernel 落点、父子区间、逐核 task 连续性和记录容量是否正常。

它不决定候选的净性能，不与 perf-clock 的绝对时间相减。

本阶段没有调整设备端 span、atomic wrapper、raw ABI 或记录容量。审查发现原先
SceneTest 在 converter/analyzer 返回失败时只记录 warning，pytest 仍可能显示
PASS；而阶段序列、Kernel 唯一归属和六类整数闭合正是在该离线步骤中完成。现已将
真实 A5 FDWIC level-4 成功用例改为 fail-closed：raw 缺失、converter 失败、
`merged_swimlane.json` 或 `swimlane_exclusive_analysis.json` 缺失/为空都会使该用例
失败；若设备执行本身已经失败，则保留原始异常，离线转换不覆盖根因。

普通 trace-capable AICore ELF 也增加了正向身份门禁：必须含
`fdwic_atomic_poll_boundary_slow` 与 `fdwic_swimlane_detail_record_atomic` 两个已定义
观察慢体，并且不得含 `dist_perf_clock_expect_submits`。这没有新增一套等价的
swimlane profile：普通 level-0 与 level-4 仍共享同一个 trace-capable ELF，采集
模式由运行时 level 决定；门禁只防止误拿 perf-clock 或不完整产物。raw 中的
`trace_schema_version=4`、`l2_swimlane_level=4` 和 atomic 元数据继续证明运行模式。

当前源码先用真实 B1 验证结构门禁：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260720_234158/
```

该轮为 96 核、每核 5 个 Submit、4,559 条记录、`dropped=0`，所有父子关系、
Kernel 归属和整数闭合均 PASS。它的完整 Submit 为 302.072 us，明显受本轮冷启动/
轮询状态影响，只作为结构门禁，不替代 perf-clock 性能基线。

随后只运行一次当前 HEAD 的完整 Case1 并保留 golden 校验，权威第二证据链产物为：

```text
outputs/TestPagedAttentionUnroll_Case1_20260720_234305/
  l2_swimlane_records.json             75,397,613 B
  merged_swimlane.json                182,110,972 B
  swimlane_exclusive_analysis.json        121,264 B
```

该轮为 32 AIC + 64 AIV、每核 1,280 个 Submit、全局 122,880 个 Submit，
944,874 条 raw 记录且 `dropped=0`；全局首末 Submit 为 **5,095.821 us**。Submit、
Submit envelope、EfDrain、OrchestrationReplay、FinalDrain 和 WorkerCompletion
六类整数分区全部精确闭合。Atomic 物理记录、批处理轮询与逻辑调用满足：

```text
105577 - 330 + 3899 = 109146
```

其中 merged 中 `return_ready/source_issue/PollBatch` 分别为
102,495/2,752/330。Atomic 仍是不可加的 overlay；PollBatch 表示完整等待区及其
精确调用次数，不能用 duration 反推单次 atomic 延迟。

本轮设备端仍预留固定 **201,333,568 B** trace buffer。实际 raw 没有再增加字段，
但该固定容量和约 182 MB merged 进一步说明：swimlane 只在需要业务/atomic 定位时
采集，不能作为权威性能基线，也不应为普通 A/B 反复生成。生产 converter/analyzer
回归与新增 fail-closed/ELF 门禁共 86 项通过。

### 8.3 `submit-pmu-none` 与真实 span 单阶段 PMU

**[`submit-pmu-none`、`arg-build`、空 bracket 校准和 `materialize` 均已完成 A5 收口]**

本阶段没有修改或复测 standalone，而是在真实 PA 中建立独立诊断构建。该构建在
编译期去除普通泳道、atomic 观察和通用逐 task PMU ring，分为两种运行方式：

1. **`submit-pmu-none`**：每物理子核在完整 Submit 调度期只 start/stop 一次，
   不做中途 shadow read-clear；输出 96 核 PMU total、scalar busy、I-cache
   request/miss，并按 AIC/AIV 生成 raw 与 HTML；
2. **真实 span 单阶段 PMU**：一次 ELF 只选择一个当前真实泳道区域做局部观测，
   仍同时保留本 ELF 自己的完整 Submit primary，局部只与本 ELF、本轮、本角色的
   primary 和时间分母比较。首个 selector 为 Claim 完成到 Materialize 入口之间的
   `arg-build`。

#### 8.3.1 已闭合的 `submit-pmu-none`

入口为 `--fdwic-profile submit-pmu-none`，当前只接受真实 A5、FDWIC、level 2、
`rounds=1`。构建保留公开 `PTO2_PROFILING` Arg ABI，但固定
`PTO_FDWIC_TRACE_ENABLED=0`、`PTO_FDWIC_SUBMIT_PMU=1`；最终 ELF 必须包含
`dist_submit_pmu_expect_submits` 和 `fdwic_submit_pmu_read_counters`，并拒绝
perf-clock、swimlane/atomic、通用 PMU ring 和通用 PMU reg-base 符号。

每个物理子核在 attach 时先 stop/清计数；首个 Submit 读取 1 ns `SYS_CNT` 后开启
PMU，末个 Submit stop 后读取 CNT2/CNT6/CNT7，并用 CNT8/CNT5 做 request/miss
影子复核。FinalDrain 不进入有效窗口。AICPU owner 在发布 worker 运行状态前，逐核
保存并配置 PMU 控制寄存器和 selector；96 个 worker 完成后逆序恢复。正式 raw
要求 32 AIC、64 AIV、96 个唯一物理 ID、32 个完整 1:2 mixed triplet、配置/恢复
96/96、active-after-restore=0、每核 Submit 次数和窗口状态全部闭合。

实现过程中有两次由门禁揭示并修正的接口问题：

1. PA orchestration 原先只在 `PTO_FDWIC_PERF_CLOCK` 条件下声明预期 Submit 数，
   导致 submit-PMU 首轮实测为 count `5/0`、窗口未启动；修正为 perf-clock 与
   submit-PMU 复用同一真实挂点，不在 host 猜测次数。
2. host `Runtime::workers[].physical_core_id` 是 H2D 前的 host shadow，不会在 export
   前从设备 Runtime 整块回拷；真机核 1 已实证 device record 为 physical 1、host
   shadow 仍为 0。该无效比较已移除。逻辑核到物理核的可信关系由设备 AICPU owner
   校验、每核 record、唯一集合、角色和 triplet 三层闭合，不新增冗余映射字段。

B1 两次独立成功采集都通过 96 核、5 Submit/core、owner restore 和报告门禁：

| 产物 | 全局 Submit | AIC total/core mean | AIC scalar/core mean | AIV miss/core mean |
| --- | ---: | ---: | ---: | ---: |
| `..._002939` | 74.882 us | 28,778.9 cycles | 25,099.7 cycles | 129.56 |
| `..._003050` | 227.673 us | 36,999.4 cycles | 25,318.8 cycles | 129.84 |

第二轮的 AIC total 最大值从 122,226 增至 375,185 cycles，而 AIC scalar mean
仅增加约 0.87%；AIV request/miss 也基本不变。它说明“设备独占”并不等于每核
到达相位和非 scalar-busy 等待恒定，是后续波动专项归因的输入，当前不把两点样本
直接写成因果结论。

Case1/B256 正式收口结果为：

| 指标 | AIC | AIV |
| --- | ---: | ---: |
| core 数 | 32 | 64 |
| PMU total/core mean | 7,436,193 cycles | 7,410,246 cycles |
| scalar busy/core mean | 6,479,673 cycles | 6,761,739 cycles |
| scalar busy / total | 87.14% | 91.25% |
| I-cache request/core mean | 646,963.94 | 594,542.61 |
| I-cache miss/core mean | 1,196.88 | 16,943.17 |
| 聚合 miss/request | 0.1850% | 2.8498% |

全局首个 Submit 到最后一个 Submit 为 **5,075.360 us**，每核 1,280 次 Submit，
primary/shadow 96/96 相等；最大可编程计数 7,129,295，远小于
`0x3fffffff` 风险阈值。按 90 ns/miss 只能得到单核串行等效量级，不能把 AIV
约 1.525 ms/core 直接写成可消除的墙钟损失。raw 约 46 KB、HTML 约 77 KB，
没有复用约 200 MB 的泳道 buffer。

提交前还用当前工作树分别回归了 B1 perf-clock 与普通 level-4：perf-clock 为
96×5 Submit、254.084 us；level-4 为 4,549 条事件、89.071 us、`dropped=0`，
排他闭合 PASS。两者用于证明三种 ELF 的双向隔离和公共 Submit 挂点没有回退，
绝对时间仍不得跨 ELF 相减。

#### 8.3.2 首个真实 selector：`submit-pmu-arg-build`

入口为 `--fdwic-profile submit-pmu-arg-build`，编译期固定：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=1
PTO_FDWIC_TRACE_ENABLED=0
```

Kernel 与 Alloc 两条 compete-first 路径都使用同一组源码业务边界：起点在 Claim
完成后，终点在匹配 Finish 恢复并校验 ticket 后、Materialize 入口前。该区间覆盖
`dist_submit_make_ticket()`、Begin 返回、同步 eager callback 的 `build_args(args)`、
Finish 重入、`dist_submit_restore_from_ticket()` 和 ticket 校验；不包含 Claim 本体
和 Materialize 本体。它复用了泳道 `Claim.end -> Materialize.begin` 的业务语义，
但 PMU ELF 已编译掉泳道 record，二者不是同一 ELF，也不做逐 tick 对齐。

`submit-pmu-none` 保持原 128 B 前缀加 `96 × 64 B` 整窗记录，共 6,272 B；
`arg-build` 在其后追加 `96 × 64 B` phase sidecar，总计 12,416 B。每个 worker
仍独占 cacheline，只在窗口结束后发布一份汇总，不生成逐事件记录，也不复用约
200 MB 的泳道 buffer。CNT6/CNT7 是窗口中从不读取的整窗 primary；CNT8/CNT5
作为 running read-clear shadow，begin 样本只进入 shadow whole，end 样本同时
进入 shadow whole 和 phase observed，stop 后 tail 只进入 shadow whole。

这里特意不把 phase request/miss 称为业务事件数的“严格下界”。counter read 与
`SYS_CNT` 边界之间仍有少量观测 bookkeeping，它们的取指会进入 observed sample；
`primary - shadow` 只量化分段重建的 capture gap，不能抵消插桩自身的事件。
报告因此同时展示 `observed` 与 `observed + (primary - shadow)`，二者是当前插桩
ELF 的观测值和加全窗 capture gap 后的敏感性量尺，不是原业务区间的数学上下界。
phase 只提供 `SYS_CNT` 时间及 I-cache request/miss observed，不杜撰局部 PMU
total、scalar busy 或 I-cache stall 时间。

正式 raw 只有在逐核满足以下条件时发布：

- `phase_begin_reads == phase_end_reads == expected_submit_count`；B1 为 5，Case1
  为 1,280；
- phase status 为 `0x3f`，即 requested、边界平衡、调用 shape、数值顺序、时间
  落在本核 Submit 内和 tail read 六项全部成立；
- phase observed 分别不超过对应 shadow，shadow 不超过 primary；本阶段实测
  primary/shadow 精确相等，但通用契约仍只要求单向闭合；
- 可编程计数与每段最大读数都低于 `0x3fffffff` 风险阈值；
- phase 时间非零且不超过同核 `submit_elapsed_ticks`。

两轮独立 B1 结构样本分别位于：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260721_014154/
outputs/TestPagedAttentionUnroll_CaseB1_20260721_014301/
```

两轮均为 96 核、每核 5 次 begin/end、status `0x3f`、primary/shadow 96/96
精确相等；全局 Submit 分别为 247.205 us 和 80.702 us，phase core-time 份额分别
为 6.862% 和 7.432%。这两点证明结构闭合，也再次暴露独占设备仍有明显到达/等待
波动；不能拿两轮绝对时间直接形成性能结论。

Case1/B256 正式件为：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_014355/
  fdwic_submit_pmu_raw.json       72,811 B
  fdwic_submit_pmu_report.html    79,497 B
```

该轮全局 Submit 为 **4,964.039 us**，96 核每核 1,280 次，共 122,880 次
begin/end 全部闭合，primary/shadow 96/96 精确相等，最大 shadow request/miss
分段为 32,136/129。按同一 ELF、同一轮聚合，`arg-build` 的 core-time、request
observed、miss observed 份额分别为：

| 角色 | core-time | request observed | miss observed |
| --- | ---: | ---: | ---: |
| ALL | 5.557% | 20.716% | 21.334% |
| AIC | 4.297% | 20.099% | 10.640% |
| AIV | 6.183% | 21.031% | 21.417% |

这组数据说明 `arg-build` 在当前插桩 ELF 中取指观测份额高于时间份额，值得在完成
空 bracket 校准后继续判断观察 bookkeeping 占比；目前不能把约 21% 直接写成
原业务 I-cache 事件比例，更不能把 4.964 ms 与 none、perf-clock 或泳道绝对时间
相减。

最终还分别回归了三条互斥证据链：`submit-pmu-none` B1
`20260721_014602` 保持 96/96 primary=shadow 且不含任何 phase 字段；perf-clock
B1 `20260721_014715` 为 96×5 Submit、73.029 us；普通 level-4 B1
`20260721_014841` 为 4,560 条事件、90.275 us、`dropped=0`、排他闭合 PASS。
这些结果证明 phase sidecar、reader 和状态只进入选中的 `arg-build` ELF；三轮绝对
时间仍不互相相减。

#### 8.3.3 running bracket 空区间校准：`submit-pmu-empty-bracket`

**[观察工具，已完成两轮 B1 与两轮 Case1 实测闭合]**

`arg-build` 的 running read-clear 会在每次阶段 begin/end 各读一次 shadow counter。
在继续扩展业务 selector 前，本阶段先回答一个基础问题：同一套 begin/end 观察器
紧邻执行、其中不包任何业务体时，会在真实 A5 PA 中形成怎样的稳定时间和 I-cache
观察指纹。入口为 `--fdwic-profile submit-pmu-empty-bracket`，编译身份固定为：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=2
PTO_FDWIC_TRACE_ENABLED=0
```

host/runtime 契约使用独立 mode `3`、phase id `2`，phase 名称和边界分别为
`empty-bracket`、`claim_end_adjacent_empty_bracket`。真实 hook 没有另造调用拓扑：
Kernel 与 Alloc 的 compete-first Begin 都在现有 Claim 完成、`claim_end` 已取到的
同一源码调用点，紧邻执行一次原样 generic phase begin/end；每个 Submit 恰好一对，
B1 每核 5 对，Case1 每核 1,280 对。其他 profile 中该 wrapper 编译为空。

时间和 PMU 计数必须按两种不同边界解释：

- generic begin 在第一次 shadow read-clear 与 begin bookkeeping 之后读取内部
  `SYS_CNT`，generic end 在第二次 shadow read-clear 之前读取内部 `SYS_CNT`；若只看
  这两个内部 tick，空区间并没有覆盖完整的观察器调用成本；
- empty wrapper 因此保存旧累计值和 begin/end 次数，用外层两个 `SYS_CNT` 包住完整
  generic begin/end 对，校验次数各恰好增加 1、状态平衡且时间无溢出，再用外层
  delta 覆盖本次内部 elapsed 增量。raw 明示
  `time_semantics=outer_sys_cnt_around_adjacent_begin_end_pair`；
- request/miss 仍保持原 running read-clear 口径：begin 样本只进入 whole shadow，
  end 样本同时进入 whole shadow 与 phase observed。外层 `SYS_CNT` 不改变这组读清
  边界，raw 明示
  `counter_semantics=running_read_clear_empty_bracket_calibration`。

所以 elapsed 是“外层时间戳包住完整相邻 begin/end 对”的经验耗时，request/miss
是“两次 shadow read-clear 之间”的观察值；二者不是同一个精确指令边界。外层时间
也包含两个 `SYS_CNT` 自身的底噪，不能称为观察器数学最小成本。

该 profile 复用 `arg-build` 已有的 phase sidecar，没有增加设备 raw 字段、逐事件
record 或泳道 ring。`submit-pmu-none` 仍为 128 B header 加 `96 × 64 B` whole record，
共 6,272 B；`empty-bracket` 与其他单阶段 profile 同为再追加 `96 × 64 B` sidecar，
共 12,416 B。每个 worker 最终仍只发布一份 whole 汇总和一份 phase 汇总，因此本次
校准没有继续放大约 200 MB 的泳道数据。

正式发布继续 fail-closed：96 核必须满足 owner 配置/恢复、唯一物理核与 32 个 mixed
triplet 闭合；逐核 begin/end 次数都等于预期 Submit 数，phase status 为 `0x3f`，
边界平衡、shape、数值顺序、时间落在本核 Submit 内和 tail read 全部成立；phase
observed 不超过 shadow，shadow 不超过 primary，计数不越风险阈值。以下四轮均为
32 AIC + 64 AIV、status `0x3f`、primary/shadow 96/96 精确相等，因而 capture gap
为 0。

两轮 B1 先用于结构与冷启动观察：

| 产物 | 全局 Submit | ALL 每对 elapsed | ALL 每对 request | ALL 每对 miss |
| --- | ---: | ---: | ---: | ---: |
| `..._020932` | 257.430 us | 725.192 ns | 84.844 | 1.956 |
| `..._021100` | 303.032 us | 719.304 ns | 69.194 | 2.444 |

两轮完整 Case1 用于稳态经验尺度：

| 产物 | 全局 Submit | 角色 | 每对 elapsed | 每对 request | 每对 miss |
| --- | ---: | --- | ---: | ---: | ---: |
| `..._021158` | 4,972.718 us | ALL | 640.465 ns | 49.340 | 1.342 |
| 同上 | 同上 | AIC | 567.962 ns | 48.919 | 0.008 |
| 同上 | 同上 | AIV | 676.717 ns | 49.550 | 2.009 |
| `..._021311` | 4,866.126 us | ALL | 639.272 ns | 49.337 | 1.356 |
| 同上 | 同上 | AIC | 567.619 ns | 48.870 | 0.008 |
| 同上 | 同上 | AIV | 675.099 ns | 49.570 | 2.030 |

B1 只有每核 5 对，首次进入 reader、对应调用点和相关代码布局时的冷取指占比很高；
其 request 为 69.194～84.844/对，明显高于 Case1 稳定的约 49.34/对，elapsed 也从
Case1 的约 639～640 ns/对升至约 719～725 ns/对。因此 B1 继续只作为结构、次数和
冷启动门禁，不能替代 Case1 的稳态观察尺度。

Case1 两轮还复现了稳定的角色差异：AIC 约 568 ns、48.9 request、0.008 miss/对，
AIV 约 675～677 ns、49.6 request、2.01～2.03 miss/对。当前证据只证明同一观察
实现对 AIC/AIV 形成不同且可复验的自扰动指纹；它没有证明差异必然来自 reader
跨 I-cache line、某个固定冲突或 PMU 事件定义，后续若归因必须另做同构单变量证据。

为决定是否应在本阶段调整 reader，另对 empty 最终 ELF 做了只读核验。AIC/AIV 的
`fdwic_submit_pmu_phase_read_shadow_counters` 都只有一份、均为 92 B；两个 relocatable
object 中该函数机器码逐字节相同，最终 ELF 仅因 block-local relocation 出现一个立即数
字节差异。128 B line 下，AIC reader 起址行内偏移 76 B，AIV 为 120 B，两者都跨两行，
所以“跨行”不能单独解释只有 AIV 约 2 miss/对。本机 CANN 9.1 的 DAV3510 模型配置
显示 scalar I-cache 均为 4-way，但 AIC 为 32 KiB/64 sets、AIV 为 16 KiB/32 sets；
对应 combined `.text` 又分别为 68,024 B 和 82,000 B。这些证据支持容量、角色代码和
具体布局共同形成自扰动指纹，但聚合 PMU 仍不能定位到 reader 的某一条 cache line。
因此本阶段保留单份 noinline reader：不为追求较小数字而 inline 复制热路径，也不以
强制对齐改变整份诊断 ELF 的冲突集合。若以后局部信号确实被该量级淹没，应另做只改
reader 对齐的 empty A/B，而不是混入本次校准提交。

该校准不能直接从 `arg-build` 中扣除。首先，两者的时间边界不同：`arg-build`
elapsed 是两侧 observer 之间的内部业务区间，而 empty elapsed 用外层 tick 包住完整
begin/end 对；二者相减会把不同对象当成同一加法模型。其次，即使 request/miss 都
来自 running read-clear，两个 profile 仍是不同 ELF，代码布局、冷暖状态和前端竞争
都可能改变事件数。方向上，Case1 empty 约 49 request/对，相当于同一时期
`arg-build` AIC/AIV 约 118.31/121.36 request/对的四成；AIV empty 约 2.02
miss/对，相当于 `arg-build` 约 3.80 miss/对的一半。这只说明观察器污染不可忽略，
不是允许产出“扣除 empty 后的业务净 request/miss”。报告和后续结论都只保留原始
observed、capture gap 与这份经验尺度。

空 bracket 代码落定后又在同一工作树上串行回归了四种互斥构建：`arg-build` B1
`20260721_021930` 为 96×5 对、status `0x3f`、primary/shadow 96/96 精确相等，
并带有新增的 `time_semantics`，全局 Submit 为 249.064 us；`submit-pmu-none` B1
`20260721_022026` 不含 phase 字段、primary/shadow 96/96 精确相等，全局 Submit 为
265.977 us；perf-clock B1 `20260721_022115` 为 96×5 Submit、78.230 us；普通
level-4 B1 `20260721_022221` 为 4,546 条事件、88.595 us、`dropped=0`、排他闭合
PASS。四轮只证明新增 mode、元数据和公共 Claim.end 空 wrapper 没有破坏既有证据链，
B1 冷启动绝对时间仍不得跨 ELF 相减。

#### 8.3.4 第二个真实业务 selector：`submit-pmu-materialize`

**[观察工具，已完成两轮 B1 与两轮 Case1 实测闭合]**

本阶段不是从历史 `claim/efdrain/materialize/register` 名单中顺次取一个旧 phase，
而是重新查看当前 compete-first 真实 PA 的最新 Case1 排他结果
`outputs/TestPagedAttentionUnroll_Case1_20260720_234305/` 后再决定。该轮
`Materialize` 为 97,467,035 aggregate core-ticks，占 `SubmitUnion`
399,604,449 ticks 的 **24.391%**，是已具备明确源码起止边界的最大业务 span；同轮
`Claim`、`EfDrain`、`Register` 分别占 19.887%、15.853% 和 12.010%。已经完成的
`arg-build` 则覆盖 `Claim.end -> Materialize.begin`，二者首尾相接但不重叠。
因此这一轮选择来自最新真实布局和可复用边界，不是复活 standalone 或旧 schema
中的 selector 顺序。

入口为 `--fdwic-profile submit-pmu-materialize`，编译身份固定为：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=3
PTO_FDWIC_TRACE_ENABLED=0
```

host/runtime 使用独立 mode `4`、phase id `3`，phase 名称和边界分别为
`materialize`、`materialize_begin_to_materialize_end`；counter/time 口径分别为
`running_read_clear_observed_bracket` 和
`inner_sys_cnt_between_boundary_observers`。这仍是一轮只打开一个业务 phase 的诊断
ELF，不同时采 Claim、EfDrain 或 Register。

设备端没有另造一条近似路径，而是在四个真实入口精确打开同一个 phase：

1. 旧 Kernel `dist_submit_impl()` 在 EfDrain 完成、进入 Materialize 前 begin；
2. 旧 Alloc `dist_alloc_tensors()` 在相同业务边界 begin；
3. compete-first Kernel finish 在 ticket 恢复和校验成功、`materialize_begin`
   取时后 begin；
4. compete-first Alloc finish 在相同业务边界 begin。

四条入口统一调用 `dist_submit_materialize_and_prepare_map()`；唯一成功 end 位于该
helper 内部，在 `dist_submit_check_task_cap()` 和
`dist_submit_materialize_args()` 均成功返回后、泳道 `materialize_end` 取时前。
因此观测区间包含 task-cap 检查、tag/output/register-mask 扫描、heap ring 布局、
输出 Tensor 初始化及 `heap_next` 推进，但不包含后继 `PrepareMap`。submit-PMU 构建
已编译掉泳道 record，所以旧入口 begin 与 helper 之间的 trace 宏不会给本 ELF
增加一条实际记录。

失败路径刻意不伪造 end：task-cap 或参数、heap、输出物化任一检查失败时，helper
直接返回，遗留的 armed phase 会使 begin/end 不平衡、调用 shape 或最终 status
闭合失败；设备发布、host 校验和 HTML 加工据此 fail-closed，不能把一个被截断的
Materialize 当成有效短样本。正常 PA Case1 中 Materialize 每个 Submit 固定执行
一次，所以正式 shape 必须严格为 96 核、每核 1,280 次 begin 和 1,280 次 end，
全局各 122,880 次；B1 则固定为每核 5 次、全局各 480 次。

该 profile 复用已有 phase sidecar，没有扩展 raw ABI：`submit-pmu-none` 仍为
128 B header 加 `96 × 64 B` whole record，共 6,272 B；`materialize` 与
`arg-build/empty-bracket` 一样只再追加 `96 × 64 B` phase record，总计 12,416 B。
没有增加逐 Submit 记录、泳道字段或约 200 MB 的 trace ring。

两轮 B1 先验证真实挂点、固定 shape 和冷启动下的数值闭合：

| 产物 | 全局 Submit | ALL 每次 elapsed | ALL 每次 request | ALL 每次 miss | 同 ELF 时间/request/miss 占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `..._024533` | 82.091 us | 912.846 ns | 333.183 | 8.225 | 20.432% / 45.075% / 21.321% |
| `..._024641` | 81.741 us | 896.398 ns | 267.810 | 7.369 | 20.932% / 42.404% / 20.485% |

两轮都是 32 AIC + 64 AIV、每核 5 次 begin/end、phase status `0x3f`，96/96
primary/shadow 精确相等，最大单段 shadow request/miss 分别为 3,068/85 和
3,040/78。B1 仍只用于结构、次数、冷启动和快速门禁，其每次 request/miss 不外推
Case1 稳态。

两轮完整 Case1 产物为：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_024748/
outputs/TestPagedAttentionUnroll_Case1_20260721_024909/
```

两轮均为 96 核、每核 1,280 次 begin/end、phase status `0x3f`，owner 配置/恢复、
32 个 mixed triplet、固定 shape、数值顺序、phase 时间和风险阈值全部闭合；
primary/shadow 96/96 精确相等，capture gap 为 0。第一轮 raw/HTML 为
72,947/80,049 B，第二轮为 72,958/80,049 B。逐角色每次调用的原始 observed 为：

| 产物 | 全局 Submit | 角色 | 每次 elapsed | 每次 request | 每次 miss |
| --- | ---: | --- | ---: | ---: | ---: |
| `..._024748` | 4,922.142 us | ALL | 797.814 ns | 233.197 | 1.474 |
| 同上 | 同上 | AIC | 776.265 ns | 228.942 | 0.022 |
| 同上 | 同上 | AIV | 808.588 ns | 235.325 | 2.201 |
| `..._024909` | 4,851.282 us | ALL | 797.061 ns | 233.238 | 1.418 |
| 同上 | 同上 | AIC | 775.653 ns | 228.170 | 0.021 |
| 同上 | 同上 | AIV | 807.765 ns | 235.772 | 2.116 |

“阶段占比”只使用同一 ELF、同一轮、同一角色的数据：时间分子为 phase elapsed
core-time、分母为逐核首末 Submit elapsed core-time；request/miss 分子为 phase
observed、分母为本轮整窗 primary。两轮结果为：

| 产物 | 角色 | 时间占比 | request observed 占比 | miss observed 占比 |
| --- | --- | ---: | ---: | ---: |
| `..._024748` | ALL | 22.560% | 37.688% | 12.056% |
| 同上 | AIC | 22.540% | 36.532% | 10.487% |
| 同上 | AIV | 22.570% | 38.278% | 12.065% |
| `..._024909` | ALL | 22.105% | 37.741% | 11.681% |
| 同上 | AIC | 21.882% | 36.442% | 10.997% |
| 同上 | AIV | 22.214% | 38.404% | 11.685% |

两轮约 22.1%～22.6% 的同 ELF 时间份额和约 37.7% 的 request 份额可以说明
Materialize 是当前诊断布局中的重要取指区域；它们不能直接等同于关闭插桩后的净
业务成本。尤其 empty-bracket 的两轮 Case1 经验指纹约为 ALL 639～640 ns、
49.34 request、1.34～1.36 miss/对，而 materialize 约为 797 ns、233.2 request、
1.42～1.47 miss/次。empty elapsed 用外层 tick 包住完整 observer 对，materialize
elapsed 是两个 observer 内侧的业务时间；request/miss 即使都来自 running
read-clear，也属于不同 ELF、不同布局和不同缓存状态。故 empty 只能提示观察器的
自扰动量级不可忽略，绝不能从 materialize 中相减得到“净时间”或“净 miss”。
按角色看，AIV Materialize 为 2.20/2.12 miss/次，empty-bracket 为 2.01/2.03
miss/对，也仍处在同一量级；当前结果不能证明 Materialize 业务体带来了明确的 AIV
miss 增量。AIC Materialize 只有约 0.022/0.021 miss/次，同样只保留原始观测，不作
跨 ELF 扣减。

代码落定后串行回归了五类互斥 B1 构建：

| 构建 | 产物 | 结果 |
| --- | --- | --- |
| `submit-pmu-arg-build` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_025027/` | 96×5、status `0x3f`、primary/shadow 96/96 精确相等；257.392 us |
| `submit-pmu-empty-bracket` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_025125/` | 96×5、status `0x3f`、primary/shadow 96/96 精确相等；230.313 us |
| `submit-pmu-none` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_025214/` | 无 phase 字段、primary/shadow 96/96 精确相等；298.298 us |
| `perf-clock` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_025305/` | 96×5 Submit、ELF 身份和调用 shape 闭合；274.997 us |
| 普通 level-4 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_025353/` | 4,550 条事件、89.109 us、`dropped=0`、排他整数闭合 PASS |

这些回归只证明新增 mode、四个 begin 挂点和统一成功 end 没有破坏既有构建身份、
phase 契约或泳道加工；B1 的跨核到达/等待波动很大，五种 ELF 的绝对时间仍不得
互相相减。

最后将 AICPU header 校验从逐个枚举旧 phase mode 等价收敛为复用
`fdwic_submit_pmu_mode_has_phase()`，使 mode 判定与公共 phase/字节数契约只有一个
事实来源。该重构没有修改计数器配置、业务边界或设备 ABI；最终源码再次运行
`submit-pmu-materialize` B1：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260721_025856/
```

结果为全局 Submit **281.494 us**、96 核各 5 次，begin/end 全局 480/480、status
`0x3f`、primary/shadow 96/96 精确相等。它是最终源码状态的 materialize 回归；
前两轮 B1 仍保留为最初四挂点实现的独立结构样本。

#### 8.3.5 后续真实 selector 与三条证据链分工

单阶段 selector 必须直接跟随当前真实 PA compete-first 泳道和离线排他定义，候选
区域包括：

```text
efdrain              包含 opportunistic drain 中的 Kernel
efdrain-control      EfDrain 去除其中 Kernel 后的排他控制区域
claim
arg-build            Claim.end 到 Materialize.begin 的同步 eager 构参
materialize
prepare-map
fanin
register
winner-build
alloc-complete
loser-replay         真实 PA loser 的实际尾动作
submit-finalize      最后业务 child 到本次 Submit.end 的公共收尾
submit-transition    相邻 Submit 之间的输出接收、下一任务准备与调用衔接
kernel               真实计算区间 overlay
```

以上只是根据真实源码和最新泳道语义形成的 selector 设计清单，不表示全部区域已经
实现。`efdrain` 包含 `kernel`，而 `efdrain-control` 是去除 Kernel 后的排他控制，
三者不能相加。`kernel` 也是 overlay；`submit-finalize` 和
`submit-transition` 应复用现有真实边界，不为方便取数继续扩大泳道 raw。每个局部
selector 必须独立编译、独立运行、独立发布，不能在一个 ELF 中同时打开多段。

`arg-build`、running bracket 空区间校准和第二个业务 selector `materialize` 已经
实现；其余 selector 仍按最新真实 Case1 排他分布和可证明源码边界逐个推进，不按
旧名单批量复刻。后续每个业务 phase 都必须同时报告自己的原始 observed 与 empty
的经验尺度，但不得跨 ELF 扣减出伪精确净值。standalone 历史实现只提供 owner、
门禁和报告加工方法参考，不是该阶段的前置优化任务，也不能作为真实结果代替品。

三条证据链最终分工为：

| 构建 | 回答的问题 | 不能回答的问题 |
| --- | --- | --- |
| `perf-clock` | 候选是否真正缩短完整 Submit 墙钟 | 具体业务区域和 PMU 原因 |
| `swimlane` | 业务区域、atomic 次数、时序与闭合 | 无观察净性能、I-cache miss |
| `submit-pmu-none` / 单阶段 PMU | 完整窗口 AIC/AIV 每核 total/scalar/primary，以及一个真实 span 的同 ELF 时间和 request/miss observed | 跨 ELF 净阶段成本、局部 scalar busy、最终墙钟收益 |

还应单独比较一次 perf-clock 与诊断构建的完整时间，量出保留观察能力的剩余代价；
若差距已经很小，不继续为了观察代码做无目标的冷路径调整。

## 9. 已撤回、失败或不能外推的路线

### 9.1 已撤回的优化候选

| 路线 | 结果 | 决定 |
| --- | --- | --- |
| `ld_dev()+nop10/100` 替换 atomic | 性能用于定位，但同步语义不成立 | 撤回，不得作为优化 |
| fanin producer 降序 | fanin load 下降，Submit 中位反而 +7.774% | 撤回，不迁真实 PA |
| compete-first lazy C 版 | 相对 eager B 为 +0.040%，各 11/22 胜 | 不进入主路，不做 PMU |
| 16 B compete-first ticket | 相对同时段基线中位反向约 +0.434%，波动重叠 | 撤回 |
| pointer ticket | 首份真实 PA 样本 5.331474 ms，且引入生命周期约束 | 撤回 |
| standalone 伪 LoserReplay | loser 没有真实动作，却显著扩大 raw | 删除，改为离线未覆盖时间 |

仓库历史中没有可追溯的 `fanin-prefix` 提交或源码痕迹；该未提交过程态已经按要求
去除，本文不为它杜撰 hash、数据或收益。

### 9.2 正确性或平台验证的已知缺口

- standalone 16 MiB tiny-ring 的 CPU b256，H1 与临时恢复原实现都在观察窗口内
  未结束；只能认定模型存在既有长程活性或 host 调度问题，不能把它记为 H1 的
  PASS/FAIL；
- MB6 `Normal` 因当前 runtime 没有测试期望的 DEPSIG 而失败；
- MB6 `Heavy` 在 H1 和基线中走同一调用路径 abort，不能归因于 H1，也不能记 PASS；
- `FullCore36` 在 Heavy 基线失败后未继续运行；
- 上述缺口没有通过顺手修改测试契约或生产协议来掩盖。

perf-clock 的 Case2 负测试不属于上述缺口：其每核实际 576 次与期望 320 次不符，
host 已按设计拒绝并且没有输出成功 summary。这个结果只覆盖 fail-closed 门禁，
不能被改写成 Case2 正确性或性能 PASS。

### 9.3 被证明不适合的观察路线

- external task-based `msprof` 的 raw counter 不受 kernel 内 start/stop 缩窗控制，
  不能用于 Claim、EfDrain 等局部取数；
- A5 上把 I-cache miss selector `0x35` 复制到 CNT9 时计数恒为 0，因此现行
  shadow miss 使用 CNT5，并明确牺牲诊断 ELF 的 MTE3 busy；
- A5 CANN 9.1 没有 `scalar_wait_ib_time` 的正式事件或派生字段；
- 只在运行时把泳道 level 设为 0/1，不能移除 ELF 中的冷诊断代码和布局污染；
- 旧 runtime cache 不理解新增 phase 时曾只生成 Kernel/Alloc 名称，该轮已排除，
  不能作为有效基线；
- running phase 的 PMU read-clear 会改变布局和时序，不能将局部 ELF 与 `none`
  相减得到无扰动净时间。
- perf-clock 不能通过 `PTO2_PROFILING=0` 实现：该宏拥有公开 Arg 布局/ABI，首版
  尝试在 Arg cacheline `static_assert` 处失败、从未上板，并已撤回。最终实现保留
  该 ABI，只编译期移除 FDWIC 泳道/atomic 和平台 PMU 路径。

## 10. 本机证据产物索引

### 10.1 重要说明

`outputs/` 已被 Git 忽略。下列路径是本机采集证据索引，不属于源码提交，也不保证
fresh clone、其他 worktree 或后续清理后仍存在。文档中的数值必须与对应历史
commit 和采集配置一起理解，不能因为文件名存在就当作当前 HEAD 结果。

### 10.2 真实 PA

| 阶段 | 路径 | 主要用途 |
| --- | --- | --- |
| 初始 atomic-load 基线 | `outputs/TestPagedAttentionUnroll_Case1_20260717_023809/merged_swimlane_atomic_load.json` | 5.642245 ms 起点 |
| NOP100 诊断 | `outputs/TestPagedAttentionUnroll_Case1_20260717_035341/merged_swimlane_nop100.json` | atomic 成本定位，已撤回 |
| NOP10 诊断 | `outputs/TestPagedAttentionUnroll_Case1_20260717_035954/merged_swimlane_nop10.json` | atomic 成本定位，已撤回 |
| 第一轮最好结果 | `outputs/TestPagedAttentionUnroll_Case1_20260717_055638/merged_swimlane_best_joint_poll_skip_arg_masks_5.096685ms.json` | joint skip + masks |
| H1 最好结果 | `outputs/TestPagedAttentionUnroll_Case1_20260717_173313/merged_swimlane_heapguard_first_lap_fastpath_5.098696ms.json` | HeapGuard 首圈 fast path |
| winner 冷路完整泳道 | `outputs/TestPagedAttentionUnroll_Case1_20260719_131116/merged_swimlane.json` | 恢复 atomic 观察接入后的布局 |
| atomic 冷路径 level-4 | `outputs/TestPagedAttentionUnroll_Case1_20260719_135629/` | Atomic/PollBatch/ClockBaseline 闭合 |
| compete-first level-1 三轮 | `outputs/TestPagedAttentionUnroll_Case1_20260720_095456/`、`..._095724/`、`..._095929/` | 真实路径三轮 A/B |
| compete-first level-4 历史样本 | `outputs/TestPagedAttentionUnroll_Case1_20260720_104406/merged_swimlane.json` | 早期真实布局、阶段与 atomic 闭合，不代表当前 HEAD |
| compete-first A5Sim | `outputs/TestPagedAttentionUnroll_Case1_20260720_104649/` | 108 核模拟回归 |
| perf-clock B1 首轮 | `outputs/TestPagedAttentionUnroll_CaseB1_20260720_172436/fdwic_perf_clock_summary.json` | 96 核、5 Submit/core，75.347 us |
| perf-clock B1 复验 | `outputs/TestPagedAttentionUnroll_CaseB1_20260720_172615/fdwic_perf_clock_summary.json` | 96 核、5 Submit/core，73.716 us |
| perf-clock Case1 golden | `outputs/TestPagedAttentionUnroll_Case1_20260720_172820/fdwic_perf_clock_summary.json` | golden PASS；4,489,247 ticks，不计入五轮基线 |
| perf-clock Case1 五轮基线 | `outputs/TestPagedAttentionUnroll_Case1_20260720_173140/`、`..._173224/`、`..._173307/`、`..._173350/`、`..._173433/` | 中位 4823.114 us，范围 4495.677～5808.500 us |
| perf-clock Case2 负测试 | `outputs/TestPagedAttentionUnroll_Case2_20260720_173035/` | 576/core 与期望 320 不符；拒绝结果，无成功 summary |
| 同源普通 level-4 B1 | `outputs/TestPagedAttentionUnroll_CaseB1_20260720_173738/` | 480 Submit、85.653 us、dropped=0、排他 PASS；不与 perf-clock 相减 |
| 当前 HEAD level-4 B1 门禁 | `outputs/TestPagedAttentionUnroll_CaseB1_20260720_234158/` | 新 fail-closed/ELF 门禁上板 PASS；仅作结构证据 |
| 当前 HEAD level-4 Case1 | `outputs/TestPagedAttentionUnroll_Case1_20260720_234305/` | 96×1280 Submit、5095.821 us、944874 records、dropped=0、全部闭合 |
| submit-PMU B1 首轮 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_002939/` | 96×5 Submit、74.882 us、raw/HTML 闭合 |
| submit-PMU B1 复验 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_003050/` | 96×5 Submit、227.673 us；AIC total 波动线索 |
| submit-PMU Case1 | `outputs/TestPagedAttentionUnroll_Case1_20260721_003335/` | 96×1280 Submit、5075.360 us；真实 AIC/AIV PMU raw/HTML |
| submit-PMU 后 perf-clock B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_004056/` | 96×5 Submit、254.084 us；ELF 隔离 PASS |
| submit-PMU 后 level-4 B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_004154/` | 4549 records、89.071 us、dropped=0、排他闭合 PASS |
| arg-build PMU B1 首轮 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_014154/` | 96×5 bracket、247.205 us、observed/capture-gap 契约闭合 |
| arg-build PMU B1 复验 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_014301/` | 96×5 bracket、80.702 us、observed/capture-gap 契约闭合 |
| arg-build PMU Case1 | `outputs/TestPagedAttentionUnroll_Case1_20260721_014355/` | 96×1280 bracket、4964.039 us、raw/HTML 全部闭合 |
| arg-build 后 none B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_014602/` | 无 phase 字段、primary=shadow 96/96 |
| arg-build 后 perf-clock B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_014715/` | 96×5 Submit、73.029 us、ELF 隔离 PASS |
| arg-build 后 level-4 B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_014841/` | 4560 records、90.275 us、dropped=0、排他闭合 PASS |
| empty-bracket PMU B1 首轮 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_020932/` | 96×5 对、257.430 us；冷启动观察指纹闭合 |
| empty-bracket PMU B1 复验 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_021100/` | 96×5 对、303.032 us；冷启动观察指纹闭合 |
| empty-bracket PMU Case1 首轮 | `outputs/TestPagedAttentionUnroll_Case1_20260721_021158/` | 96×1280 对、4972.718 us；ALL 640.465 ns/对 |
| empty-bracket PMU Case1 复验 | `outputs/TestPagedAttentionUnroll_Case1_20260721_021311/` | 96×1280 对、4866.126 us；ALL 639.272 ns/对，稳态尺度复现 |
| empty 后 arg-build B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_021930/` | 96×5 对、249.064 us；新旧 phase 契约闭合 |
| empty 后 none B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_022026/` | 无 phase 字段、primary=shadow 96/96 |
| empty 后 perf-clock B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_022115/` | 96×5 Submit、78.230 us、ELF 隔离 PASS |
| empty 后 level-4 B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_022221/` | 4546 records、88.595 us、dropped=0、排他闭合 PASS |
| materialize PMU B1 首轮 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_024533/` | 96×5 bracket、82.091 us、固定 shape/observed 闭合 |
| materialize PMU B1 复验 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_024641/` | 96×5 bracket、81.741 us、固定 shape/observed 闭合 |
| materialize PMU Case1 首轮 | `outputs/TestPagedAttentionUnroll_Case1_20260721_024748/` | 96×1280 bracket、4922.142 us；ALL 797.814 ns/次 |
| materialize PMU Case1 复验 | `outputs/TestPagedAttentionUnroll_Case1_20260721_024909/` | 96×1280 bracket、4851.282 us；ALL 797.061 ns/次 |
| materialize 后 arg-build B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_025027/` | 96×5、status 0x3f、primary=shadow 96/96 |
| materialize 后 empty B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_025125/` | 96×5、status 0x3f、primary=shadow 96/96 |
| materialize 后 none B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_025214/` | 无 phase 字段、primary=shadow 96/96 |
| materialize 后 perf-clock B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_025305/` | 96×5 Submit、274.997 us、ELF 隔离 PASS |
| materialize 后 level-4 B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_025353/` | 4550 records、89.109 us、dropped=0、排他闭合 PASS |
| AICPU mode 重构后 materialize B1 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_025856/` | 最终源码 480/480、status 0x3f、primary=shadow 96/96 |

### 10.3 standalone

| 阶段 | 路径 | 主要用途 |
| --- | --- | --- |
| D1 dynamic atomic | `tests/atomic_probe/pa_scheduler/outputs/atomic_diagnostics/ccec_baseline_10_20260718_023551.log` | fanin/frontier 次数分布 |
| 历史 schema-v3 b256 | `tests/atomic_probe/pa_scheduler/outputs/pa_scheduler_swimlane_20260718_182725_4061524/ccec/` | atomic 合并泳道 5.774295 ms |
| 约 5 ms 对等样本 | `tests/atomic_probe/pa_scheduler/outputs/performance_gap_20260718/standalone_ccec_real_b256_raw.json` | 真计算同泳道比较 |
| schema-v4 b256 | `tests/atomic_probe/pa_scheduler/outputs/pa_scheduler_swimlane_20260719_114815_617346/ccec/` | 排他边界、raw 规模和整数闭合 |
| winner 冷分支 | `tests/atomic_probe/pa_scheduler/outputs/pa_scheduler_swimlane_20260719_123520_660296/ccec/` | 尾部下降和 b256 门禁 |
| compete-first b1 | `tests/atomic_probe/pa_scheduler/outputs/pa_scheduler_swimlane_20260720_092021_1729726/ccec/` | 当前 standalone 业务布局 |
| 历史 submit-pmu none | `tests/atomic_probe/pa_scheduler/outputs/submit_pmu_none_20260719_b256_final/` | 96 核 I-cache raw/HTML |
| 历史五 phase PMU | `tests/atomic_probe/pa_scheduler/outputs/submit_pmu_phase_time_v5_20260719/` | 旧边界同 ELF 比例，仅作历史 |
| 单 miss 标尺 | `tests/atomic_probe/pa_scheduler/outputs/pmu_validation/icache_single_64x10_20260718_085929_3232836_console.log`、`icache_single_128x5_20260718_090151_3235468_console.log` | 90 ns/miss 一阶标尺 |

## 11. 禁止混算和过度解释

1. **不同 ELF 不相减。** `perf-clock`、`swimlane`、不同 selector 的
   `submit-pmu` 都会改变代码布局；绝对时间不能机械相减。
2. **不同观察 level 不混用。** level-4 atomic 泳道、level-1 phase 泳道和
   `--no-swimlane` 不是同一性能口径。
3. **跨核墙钟不等于累计工作量。** 96 核 span 求和、每核 PMU mean 和最早到最晚
   Submit 都是不同量。
4. **PMU cycle 不等于 SYS_CNT tick。** 前者约 1.65 cycles/ns，后者 1 ns/tick。
5. **局部 phase 只能在本 ELF 内解释。** phase 时间、request、miss 只能除以同一
   进程、同一角色、同一 ELF 的完整区间；不同 phase 不相加，不与 `none` 相减。
6. **Atomic 边界按语义解释。** `return_ready` 不是全系统可见性屏障；
   `source_issue` 更不能解释为完成延迟。PollBatch 是等待区间，不是单次 atomic。
7. **90 ns/miss 不是墙钟损失。** `miss * 90 ns` 只给单核串行等效数量级；多核、
   预取、流水和等待会重叠。
8. **`total - scalar_busy` 不是 scalar 空闲。** 它还可能包含 I-cache refill、
   atomic 等待、Vector/Cube engine 等待和其他非 scalar-busy 周期。
9. **standalone 收益不外推真实 PA。** standalone compete-first 为 -3.9635%，
   真实 PA 只有波动重叠的 -0.263%，当前真实结论是基本持平。
10. **b1 与 b256 分工不同。** b1 用于结构、边界、调用次数和快速正确性门禁；
    b256 只在阶段性收口时用于规模和性能结论。
11. **不同 schema 的同名区域不直接相加。** schema-v2/v3 的 Build/Replay lap
    与 schema-v4 的排他尾动作定义不同。
12. **观察构建不能冒充净性能。** 带观察的 5.066862、5.278401、5.774295 ms
    都只能解释对应观察 ELF。
13. **中位数差不能脱离样本波动。** 三轮区间重叠时应写“基本持平”，不能只取
    小数点后的正向差宣布收益。
14. **固定次数与动态次数要分开。** Claim、H1 消减等可由拓扑推出；fanin retry、
    frontier helping 和 barrier poll 必须以动态记录为准。
15. **empty-bracket 不是可直接扣除的常数。** 它的 elapsed 使用外层 SYS_CNT，
    request/miss 使用 running read-clear 内边界，并且与业务 selector 属于不同 ELF；
    只能作为观察器自扰动的经验尺度，不能生成数学修正后的业务净值。

## 12. 后续阶段更新与提交模板

每个合理阶段只验证一个主要变量。完成该阶段的源码、正确性门禁、性能取数和本文
记录后，再形成一条详细中文提交；不得把多个无法拆因果的候选堆进同一个性能结论。

建议复制以下模板：

```markdown
### YYYY-MM-DD / 阶段代号：简短名称

状态：[设计中|观察工具|已保留|已撤回|受限]

目标：
- 要回答的唯一问题；
- 预期影响的现有 span 或计数。

源码与构建身份：
- 基线 commit：
- 候选 commit/工作树：
- 分支：
- 构建类型：perf-clock / swimlane / submit-pmu-<selector>；
- AIC/AIV ELF 或 manifest 身份：

单变量改动：
- 修改文件和代码语义；
- 明确未改变的协议、边界和 ABI。

环境与命令：
- CANN/PTO-ISA/GCC/Python；
- A5Sim/A5；
- batch、负载、观察 level、runs、timeout；
- A/B 顺序和预热规则。

正确性门禁：
- 语义/golden；
- 96 核、每核 Submit、task 连续性；
- winner、fanin、TensorMap、heap、frontier、placement；
- dropped、父子区间、atomic 或 PMU 闭合。

性能结果：
- 全部原始样本；
- 中心值、范围、timeout/失败数；
- 同构 A/B 变化；
- 不能解释的波动。

辅助观察：
- swimlane 变化；
- atomic 逻辑/物理次数；
- PMU total/scalar/request/miss；
- `.text` 和关键符号尺寸。

本机产物：
- raw：
- merged/HTML/analysis：
- 日志：

决定：
- 保留、撤回或继续取证；
- 结论适用范围；
- 未关闭风险。

提交：
- hash：
- 中文主题：
- 详细正文摘要：
```

### 12.1 当前下一阶段：只推进真实 simpler PA

当前只进入真实 PA 观察基础设施阶段，不继续修改 standalone，也不立即猜测新的
业务优化：

1. **[观察工具，已实现] 真实 PA `perf-clock`**：保留 PTO2 公开 Arg ABI，
   编译期去除 FDWIC 泳道、atomic 观察和平台 PMU；首尾 Submit、96 核逐核调用数、
   6976 B header、ELF 双向身份、B1、Case1 五轮基线、Case2 fail-closed 和普通
   level-4 同源边界均已验证，作为后续候选保留/撤销的权威低扰动 A/B 口径；
2. **[观察工具，已实现] 真实 PA `swimlane`**：普通业务 span 与 atomic 合并采集，
   已补最终 ELF 正向身份、schema-v4 fail-closed、父子/Kernel/整数闭合门禁，并由
   当前 HEAD 的 B1 与完整 Case1 上板验证；只用于定位业务区域和 atomic 变化，
   不与 perf-clock 绝对时间相减；
3. **[观察工具，已实现] 真实 PA `submit-pmu-none`**：编译期去除泳道、atomic 和
   通用逐 task PMU，每核完整 Submit 期只开关 PMU 一次；96 核 primary/shadow、
   owner Restore、AIC/AIV raw/HTML、两轮 B1 和一次 Case1 均已闭合；
4. **[观察工具，两个业务 selector 与空 bracket 均已实现] 真实 PA 单阶段 PMU**：
   `arg-build` 已完成两轮 B1 与一次 Case1；`empty-bracket` 复用同一 12,416 B ABI，
   已完成两轮 B1 与两轮 Case1，量出 running begin/end 的稳态自扰动指纹并明确
   外层 elapsed 与 read-clear request/miss 的不同边界；`materialize` 又按最新真实
   Case1 最大明确业务 span 建立 mode 4/phase 3，完成两轮 B1、两轮 Case1、五类
   互斥 B1 回归和最终 AICPU mode 重构回归，仍未扩展 ABI。其余 selector 继续分别
   编译、分别采集，不在同一提交中批量铺开，也不依赖新的 standalone 实现；
5. 结构和边界迭代先使用最小有效真实 PA 用例完成正确性门禁；只有构建身份、容量、
   业务边界和统计闭合后，才运行完整 Case1 b256 性能样本，避免反复生成数百 MiB
   profiling 文件。

下一步继续以已经闭合的 empty 经验尺度为解释约束，根据最新真实 Case1 排他占比和
可证明源码边界逐个选择其余业务 selector；不从 `arg-build/materialize` observed
中机械扣除 empty，也不按旧 selector 名单批量实现。真实 span 的边界、调用 shape
和完整 Case1 数据闭合后，再量化三种构建的观察代价并专项归因独占设备上的运行波动。
这些前置数据闭合前，不从单轮 I-cache 数直接提出生产优化，也不用 standalone 的
绝对数替代真实 PA 当前结果。
