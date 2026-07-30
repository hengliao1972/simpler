<!-- markdownlint-disable MD060 -->

# A5 PA Submit 性能优化全过程记录

## 1. 文档目的与状态约定

本文持续记录 A5 FDWIC Paged Attention `Case1` 的 Submit 调度性能优化过程。2026-07-21 阶段曾只推进真实 simpler PA；自 2026-07-30 起，当前任务重新以 `tests/atomic_probe/pa_scheduler` standalone shared TensorMap 为对象，在原始 `96/32/64` Claim 合同下冻结 atomic 协议，先消减并验证 loser/nonwinner 的非 atomic 控制开销，再决定是否迁移真实路径。目标是让后续优化能够从可复核的源码、提交、实测数据和产物继续推进，而不是仅保留最终结论。

本文使用以下状态标签：

| 标签 | 含义 |
| --- | --- |
| **[已保留]** | 已进入当前代码路径，并完成与风险相称的正确性和性能验证 |
| **[已撤回]** | 做过实现或实验，但因语义不成立、性能回退或证据不足而不再保留 |
| **[观察工具]** | 用于建立测量能力，本身不是业务性能优化 |
| **[历史证据]** | 对当时源码和构建有效，不能自动代表当前 HEAD |
| **[受限]** | 已确认存在平台、模型或验证覆盖边界 |
| **[设计中]** | 只有经过源码核对的方案，尚无完成提交或性能结论 |
| **[验证中]** | 候选已落盘并通过部分门禁，但尚未完成真实 A5 正确性或性能裁决 |

记录更新至 2026-07-30。当前分支为 `fdwic-swimlane-exclusive`，跟踪 `origin/fdwic-swimlane-deps`。后续每完成一个合理阶段，都应按第 12 节模板更新本文并形成一条带详细中文说明的本地提交。

更细的专题资料分别见：

- [A5 FDWIC Paged Attention 安装与复现指南](a5_fdwic_atomic_swimlane_repo.md)；
- [PA 原子操作与优化记录](pa_scheduler/PA-atomic情况分析.md)；
- [PA 调度器独立复现与泳道使用指南](pa_scheduler/PA调度器独立复现与泳道使用指南.md)；
- [FDWIC 泳道排他分区与闭合分析](pa_scheduler/swimlane_opt_anal.md)；
- [I-cache Miss 采集与分析指南](icache_miss_usage_guide.md)。

## 2. 固定范围、环境与权威性能口径

### 2.1 本轮范围

- 真实用例：`examples/a5/fully_distributed_within_core/paged_attention_unroll/test_paged_attention_unroll.py`；
- Case：`Case1`；
- runtime：`fully_distributed_within_core`；
- 平台：A5Sim 用于功能和控制流回归，真实 A5 用于性能结论；
- A5 工作核：32 个 AIC、64 个 AIV，共 96 个 worker；
- 工作量：256 batch，每 batch 依次包含 Alloc、QK、SF、PV、UP 五个 task；
- 每核 Submit：`256 * 5 = 1280`，全局 Submit：`96 * 1280 = 122880`。

本文不把其他 runtime、其他 PA Case、A2/A3、整段 pytest wall time 或整个 device 任务耗时混入 Submit 性能结论。

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

非交互 shell 不能假设自动读取 `.bashrc`。正式复测应显式 source 用户 CANN，激活本用户 `.venv`，并显式设置用户 GCC 15 的 `PATH`、`LD_LIBRARY_PATH` 和 `CXX`。完整命令以安装复现指南为准。

### 2.3 权威性能定义

本文所说的“完整 Submit 时间”默认指：

```text
全部 worker 中最早的第一个 Submit 开始
    到
全部 worker 中最晚的最后一个 Submit 结束
```

它排除启动屏障和 FinalDrain，不等于 pytest wall time，也不等于整个 kernel launch 的 device wall time。必须同时区分三类时间：

1. **跨核完整 Submit 时间**：上述约 5 ms 的全局墙钟范围，是候选是否保留的最终性能口径；
2. **逐核或全核累计工作量**：某个 span 在 96 核上的时长求和，用于描述工作分布，不是 96 核共同形成的墙钟；
3. **PMU total/core**：每个物理子核 PMU gate 内的周期数，是单核周期工作量，也不是跨核完整 Submit 时间。

泳道原始时间使用 1 ns/tick 的 `SYS_CNT`。本机 cold/warm 校准得到 PMU 频率约 1.65 cycles/ns：ALL/AIC/AIV 分别为 1.649844/1.650062/1.649731。PMU cycle 换算不能反过来改变 `SYS_CNT` 的 1 ns/tick 定义。

## 3. 起始基线

### 3.1 环境打通与 5.6 ms 基线

**[已保留] `657313c9` — `fix(a5): enable paged attention on legacy drivers`**

该提交完成 A5 平台 block 数解析和经过双重校验的 flat OCCUPY 回退，使当前旧 Driver 环境能够运行目标 PA。A5Sim 和 A5 Case1 均通过，真实 A5 level-1 泳道复现的首末 Submit 为 **5.642245 ms**；仓库更早的历史参考为 5.577570 ms。

基线产物：

```text
outputs/TestPagedAttentionUnroll_Case1_20260717_023809/
  merged_swimlane_atomic_load.json
  l2_swimlane_records.json
```

这是第一轮业务优化的比较起点，不代表后续加入观察代码后的 ELF。

## 4. 提交时间线与阶段结论

下表按当前分支中的逻辑推进顺序列出与本轮工作直接相关的提交。详细证据和适用边界见后续各节；纯文档重命名没有单独列项。

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
| 2026-07-21 | `81a1f382` | 观察工具 | 量化真实 PA running bracket 的空区间记录开销 |
| 2026-07-21 | `26cbece6` | 观察工具 | 建立真实 PA `materialize` 单阶段 PMU |
| 2026-07-21 | `d96f5a5a` | 观察工具 | 建立真实 PA `claim` 单阶段 PMU |
| 2026-07-21 | `2929b21e` | 观察工具 | 建立真实 PA `register` 单阶段 PMU |
| 2026-07-21 | `d1572c33` | 观察工具 | 建立真实 PA `submit-transition` 单阶段 PMU |
| 2026-07-21 | `53088c48` | 观察校准 | 交错量化三类真实 PA 观察构建的整体影响 |
| 2026-07-21 | `40bd6602` | 波动分析 | 收口独占设备上的 P 构建波动来源 |
| 2026-07-21 | `a58ee868` | PMU 分析 | 收口 N 构建的 Scalar/I-cache 波动载体 |
| 2026-07-21 | `a17c188a` | 观察工具 | 建立真实 PA Submit 窗内 Kernel 低容量聚合 |
| 2026-07-21 | `05397cd7` | 波动分析 | 收口 K 构建的 Kernel/residual 波动载体 |
| 2026-07-21 | `6acebc8f` | 环境取证 | 记录设备状态、低功耗接口及并发边界 |
| 2026-07-21 | `52dc04c7` | 阶段归因 | 排除 Materialize 为同 ELF 波动主载体 |
| 2026-07-21 | `7fa7399f` | 阶段归因 | 排除 Claim 为同 ELF 波动主载体 |
| 2026-07-21 | `afeb515a` | 阶段归因 | 排除 SubmitTransition 为同 ELF 波动主载体 |
| 2026-07-21 | `57aedfee` | 阶段归因 | 排除 ArgBuild 为同 ELF 波动主载体 |
| 2026-07-21 | `660bbff4` | 阶段归因 | 排除 Register 为同 ELF 波动主载体 |
| 2026-07-21 | `443a0bb3` | 观察工具 | 完善真实 PA I-cache 逐核时间加工口径 |
| 2026-07-21 | `15c54b33` | 观察工具 | 为真实 PA PMU 产物绑定构建 provenance |
| 2026-07-21 | `36547252` | 上板验证 | 闭合完整 Submit 与 Register 分段三件套 |
| 2026-07-21 | `21e0414c` | 观察工具 | 建立排除 linked Kernel 的 EfDrain-control 固定容量 PMU |
| 2026-07-21 | `77df3959` | 上板验证 | 闭合 EfDrain-control 的实际 K、N+K 与构建三件套 |
| 2026-07-21 | `159f3c4c` | 原因取数 | 形成 EfDrain-control 的首轮 Case1 AIC/AIV 稳态数据 |
| 2026-07-21 | `7131bdf5` | 负结果 | 记录并撤回 BlockWon 慢路冷外提 |
| 2026-07-21 | `0d6547b9` | 观察回归 | 补齐 I-cache 观察链与 schema-v4 atomic 组合门禁 |
| 2026-07-21 | `970e4fad` | 正确性门禁 | 直接覆盖真实 FDWIC TensorMap 清退语义 |
| 2026-07-21 | `2dc49a13` | 优化候选 | 跳过 PrepareMap 空 task-head 的冗余 GM 写回 |

### 4.1 真实 PA 第一轮 atomic 与前端优化

#### 4.1.1 跳过单 lane 图无效 BlockWon 轮询并复用参数掩码

**[已保留] `e3b748b4` — `Update: 优化 A5 FDWIC Submit 热路径`**

该阶段包含两类改动：

1. 在本 worker 第一次见到 joint submit 之前，跳过无意义的 BlockWon 轮询；
2. 复用一次 tensor tag 扫描生成的 output/register mask，减少重复前端扫描。

PA Case1 全部 task 都是单 lane，因此第一项确定删除 Submit 内 **146944 次** 无效 `atomic_load(any_pub)`；A5 上该封装实际为 `atomicAdd(addr, 0)`，并非普通 load。删除位置主要落在每次 Submit 开头的 EfDrain 和高频 loser 的公共尾部，没有删除 Claim。

实测结果：

| 版本 | 首末 Submit |
| --- | ---: |
| 初始基线 | 5.642245 ms |
| joint polling skip 三轮中位数 | 5.171330 ms |
| 加 register mask 三轮中位数 | 5.186679 ms |
| 加 output/register masks 三轮中位数 | **5.115620 ms** |

最终相对初始基线减少 **0.526625 ms，约 9.33%**；最好单轮为 **5.096685 ms**。kernel 累计时长没有随之缩短，证据支持收益来自调度前端，而不是计算 kernel 变快。

最终最好产物：

```text
outputs/TestPagedAttentionUnroll_Case1_20260717_055638/
  merged_swimlane_best_joint_poll_skip_arg_masks_5.096685ms.json
```

mask 的独立收益只有三轮方向性证据，不能从组合版本中严格拆出因果比例；当前保留的是经过整体正确性回归的组合改动。

#### 4.1.2 用普通 load 和 NOP 替换 atomic 的定位实验

**[已撤回] 未形成保留提交**

为确认 `atomic_load` 的成本量级，曾临时替换为 `ld_dev()+nop(100)` 和 `ld_dev()+nop(10)`：

| 诊断变体 | 首末 Submit | 产物 |
| --- | ---: | --- |
| `ld_dev()+nop(100)` | 5.343592 ms | `outputs/TestPagedAttentionUnroll_Case1_20260717_035341/merged_swimlane_nop100.json` |
| `ld_dev()+nop(10)` | 5.401034 ms | `outputs/TestPagedAttentionUnroll_Case1_20260717_035954/merged_swimlane_nop10.json` |

普通 device load 加固定 NOP 不具备 atomic RMW 的同步、可见性和顺序语义，因此这些样本只用于定位成本，源码修改已经撤回，不能作为可用优化方案。

#### 4.1.3 HeapGuard 首圈 fast path

**[已保留] `04ec9b95` — `perf(a5): 跳过HeapGuard首圈冗余原子读取`**

历史文档也使用同内容提交号 `2c3dd1e2`；当前分支可达 hash 为 `04ec9b95`。默认 256 MiB heap 下，逻辑 heap 尚未走完第一圈时不可能覆盖旧输出，因此在原 fatal 检查之内直接返回，不读取 frontier/vend。PA Case1 的 Alloc、QK、SF、PV 共确定消减 **1024 次 frontier atomic load**；跨圈 slow path 保持原协议。

真实 A5 十对结果：

| 指标 | 基线 | H1 | 相对变化 |
| --- | ---: | ---: | ---: |
| 中位数 | 5.142168 ms | 5.122320 ms | -0.386% |
| 均值 | 5.167064 ms | 5.146984 ms | -0.389% |
| p90 | 5.274224 ms | 5.229773 ms | -0.843% |

十对中 8 胜 2 负，配对变化中位数为 **-0.324%**。因此只能称为“确定减少 atomic，真实 PA 中心趋势小幅正向”，不能把 standalone 的大幅波动外推到真实 PA。最好样本为 5.098696 ms：

```text
outputs/TestPagedAttentionUnroll_Case1_20260717_173313/
  merged_swimlane_heapguard_first_lap_fastpath_5.098696ms.json
```

#### 4.1.4 fanin 检查顺序实验

**[已撤回] `3d174a08` — `文档(a5): 记录fanin顺序实验与回退结论`**

F1 在 standalone 中把有效 fanin producer 按 task id 降序排列。静态分析证明固定调度状态下不会增加 ready load，实测 fanin load 也下降；但十对交错 A/B 中：

```text
Submit 中位数：4290.401 -> 4623.944 us，+7.774%
Submit 配对变化中位数：+6.439%
```

fanin load 减少没有转化为 Submit 收益，说明指令布局和 worker 到达时序的间接变化不能忽略。候选代码已撤回，未迁移到真实 FDWIC。

### 4.2 standalone 建立与动态 atomic 取证

#### 4.2.1 建立三后端独立 PA 模型

**[观察工具] `290dbda0` — `test(a5): 增加独立 PA 调度性能复现用例`**

在 `tests/atomic_probe/pa_scheduler` 下建立 CCEC、AscendC、CPU 三后端，保留 Case1 五 task 拓扑、96 worker 回放、四分片 Claim、TensorMap、fanin、私有 ring、WaitForSlot、HeapGuard、completion flag/vend/frontier 和最终 drain。此时 winner 计算仍以可控 NOP 为主，目标首先是闭合调度协议与 atomic 次数。

`ce89fae2` 和 `a3f5ecc2` 还建立了嵌套 lambda、跨 TU caller context 和 inline/noinline 边界探针。这些是后续拆分 callback/finish 时的编译行为依据，不是 PA 本身的性能收益。

#### 4.2.2 HeapGuard 对等修改与压力回归

**[已保留] `04ec9b95` 同时修改 standalone 与真实 PA**

standalone 默认 256 MiB 配置、16 MiB 多圈压力和恢复默认值后的三后端回归均经过语义检查。16 MiB CCEC/AscendC b256 确实进入 slow path；CPU b256 在 fast 版和临时撤销 fast path 的原版中都超过观察时限，不能记为 H1 PASS 或 FAIL，详见第 9 节。

#### 4.2.3 fanin/frontier 软件计数

**[观察工具] `67407cc4` — `测试(a5): 细分PA依赖与frontier原子计数`**

计数只写 worker 私有 `LocalStats`，结束时一次发布，不增加共享 atomic。b256 十轮动态基线为：

| 指标 | 中位数 |
| --- | ---: |
| fanin 总 load | 93201.5 |
| fanin not-ready load | 86675.5 |
| frontier FetchMax | 15365 |
| Submit+completion atomic ops | 203803.5 |

该结果确认主要动态项是 not-ready 重试和 frontier helping，而不是 ready 前缀。软件计数扩大了 sidecar 并增加私有 scalar 增量，因此这一版的绝对 Submit 时间不能与无计数 ELF 直接归因比较。原始日志：

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

逐 atomic 观察不插入 DSB，也不强制原本不消费返回值的 Exchange/FetchAdd 变成等待返回型操作：

- 真正消费返回值的 FetchMax、claim exchange 使用 `return_ready`；CCEC 通过返回值地址依赖后再读取 `SYS_CNT`，尽力使结束点晚于返回值可用；
- 不消费返回值的 Exchange/FetchAdd 使用 `source_issue`，只表示源码发射包围区间，不能解释成 atomic 已在全局完成；
- PollBatch 表示一个等待区间内多次逻辑轮询的整体时间和精确调用数，不能把其 duration 除成单次 atomic latency。

#### 4.3.2 PMU owner 的响应校准

自包含 owner 的 empty、100000 scalar NOP、2×100000 scalar NOP 三组，96 核 PMU total 中位数约为 214、56568、112994 cycles。这只证明 gate 响应和工作量近似倍增，不表示同数值的纳秒，也不是 PA Submit 基线。

#### 4.3.3 单次 I-cache miss 标尺

隔离微基准的结果为：

| 规模 | ALL 中位数 | 轮间范围 |
| --- | ---: | ---: |
| 64 trials/core × 10 轮 | 86.596 ns/miss | 86.532～86.792 |
| 128 trials/core × 5 轮 | 89.629 ns/miss | 89.615～89.648 |

统一使用 **90 ns/miss** 作为单核串行等效的一阶感性标尺。它不是实际 Submit 墙钟损失，不能把 `miss * 90 ns` 从约 5 ms 中直接减掉。

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

当时真实 PA 三轮中位数为 5.115620 ms，同口径差 146.726 us，约 2.87%。这说明 standalone 达到“独立复现约 5 ms 调度”的目标，不证明两份实现的代码布局、数据流和跨核时序完全一致。历史 raw：

```text
tests/atomic_probe/pa_scheduler/outputs/performance_gap_20260718/
  standalone_ccec_real_b256_raw.json
```

### 4.5 真实 PA atomic 泳道与 PollBatch

**[观察工具] `cbaf7c60` — `工具(a5): 接通真实PA atomic泳道与精确轮询聚合`**

该提交把 standalone 验证过的边界迁入真实 FDWIC，schema-v3 使用 32 B 紧凑 record、28 个稳定 site 和五类 atomic op，并只在允许的等待区聚合 PollBatch。当时真实 A5 PA Case1 level-4 通过：

```text
115200 次逻辑 atomic
110006 条物理 Atomic
340 条 PollBatch
dropped_records = 0
```

逻辑调用、物理记录和 PollBatch 必须满足 producer 定义的闭合公式。level-4 结果用于观察 atomic 分布，不能替代关闭诊断后的性能基线。

### 4.6 schema-v4 排他 span 与 raw 规模控制

#### 4.6.1 standalone 排他 span

**[观察工具] `6caa269c` — `工具(a5): 收敛Submit观测边界与排他泳道分析`**

主要变化：

- 增加 OrchestrationReplay、FinalDrain 等父区间；
- 旧 Build/Replay/Alloc lap 改为真实 WinnerBuild/AllocComplete 尾动作；
- standalone loser 没有真实计算动作，不再为 121600 个 loser 生成伪 `LoserReplay` record；
- Submit 内和 Submit 间未覆盖时间由 converter/analyzer 使用已有边界离线求差，不增加设备 record、字段或时间戳；
- merged 只保留 Perfetto 必需字段，raw 仍是权威数据；
- Kernel 必须唯一落入 EfDrain、WinnerBuild、AllocComplete 或 FinalDrain，越界、多重归属、孤儿 Kernel 或 dropped 非零都拒绝结果。

阶段性 b256：

```text
tests/atomic_probe/pa_scheduler/outputs/
  pa_scheduler_swimlane_20260719_114815_617346/ccec/
```

该轮首末 Submit 为 5.360061 ms，raw 839526 条、`dropped=0`；raw/merged 分别约 55.79/88.78 MB，全部父子关系和整数闭合通过。

#### 4.6.2 standalone winner 冷分支

**[已保留] `d2d8ce25` — `优化(a5): 将低频winner调整为冷分支`**

每个 task 只有一个 winner，绝大多数 worker 都走 loser。给 Alloc 和普通 Submit 的重型 winner 分支增加低概率布局提示，不移动边界、不改变协议。b256 同观察口径：

| 指标 | 基线 | 候选 | 变化 |
| --- | ---: | ---: | ---: |
| 全局首末 Submit | 5.360061 ms | 5.278401 ms | -1.52% |
| Submit 尾部未覆盖时间 | 41008786 cycles | 27155661 cycles | -33.78% |
| 完整逐核 Submit 区间累计 | 500448909 cycles | 483335683 cycles | -3.42% |

关闭泳道后又做候选—基线—候选 ABA，每组五轮：候选中位数分别为 3.665017/3.715385 ms，基线为 3.988115 ms，分别快 8.10%/6.84%。同时记录了 `.text` 体积增长，避免只看速度不看取指代价。候选泳道：

```text
tests/atomic_probe/pa_scheduler/outputs/
  pa_scheduler_swimlane_20260719_123520_660296/ccec/
```

#### 4.6.3 真实 PA loser 热路布局恢复

**[已保留] `cafa9ca5` — `优化(a5): 恢复真实PA的loser热路布局`**

真实 PA 对等标记两处低频 winner 分支。当前 atomic 观察版基线三轮中位数从 5.631038 ms 降为 5.192087 ms，减少 0.438951 ms，约 7.80%。但历史 pre-atomic 中位数已经是 5.115620 ms，因此这一步的准确含义是：

> 恢复 atomic 观察代码接入后发生的热路布局回退，而不是在旧 5.1 ms 基线上新增 7.8% 业务收益。

#### 4.6.4 外提 atomic 观察冷路径

**[已保留] `44367971` — `优化(a5): 外提atomic泳道冷路径消除取指回退`**

分两步推进：

1. 将 direct Atomic 的 record 发布外提为设备端共享 `noinline` 冷函数，保留 `begin -> atomic -> end` 在原 wrapper；三轮中位数为 5.096506 ms；
2. 保留 PollBatch 的内联 level 快速门，只把 level-4 命中后的十类遍历和落盘外提到共享 slow 函数。

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

相对 5.192087 ms 再下降 7.13%。该结论只能归为“消除未执行诊断代码的大量复制和热路布局回退”；当时没有同时采集专用 I-cache PMU，不能继续写成确定的 miss 降幅。

level-4 能力复核：

```text
outputs/TestPagedAttentionUnroll_Case1_20260719_135629/

Atomic 物理记录闭合：107608 = 115309 - 8056 + 355
dropped_records = 0
```

`14c2429f` 随后把 I-cache、代码布局和 PMU 经验集中到 I-cache 指南，PA atomic 文档只保留 atomic 语义、次数和边界。

#### 4.6.5 排他泳道迁入真实 FDWIC

**[观察工具] `dbb95bb5` — `Support: 将 Submit 排他泳道分析移植到 FDWIC`**

真实路径获得与 standalone 同类的父区间、真实尾动作、离线未覆盖时间和整数闭合分析，Atomic、Kernel 等 overlay 不参与排他加和。`911ecf9a` 进一步固定各 span 的业务含义、AIC/AIV 分布和全核工作量与墙钟末端的区别。

### 4.7 compete-first eager

#### 4.7.1 A/B/C 三份独立 standalone 对照

**[历史证据] `ba4334d1` — `验证(pa): 交付 compete-first/lazy 三版独立对照`**

三版分别为：

- A：原始 Materialize-first、Submit 外 eager 构参；
- B：EfDrain、Claim 前移，全体 worker 在 Claim 后同步 eager 构造完整参数；
- C：与 B 相同控制流，仅让 loser 跳过 input/scalar thunk。

72 次独立 host 启动中 A/B/C 各 24 个样本，均通过语义与后处理门禁。去异常的相邻配对结果：

| 比较 | 配对变化 | 结论 |
| --- | ---: | --- |
| B 相对 A | -206.270 us，-5.214% | 22/22 有效块更快；收益是 compete-first、split/outlining 与布局的组合 |
| C 相对 B | +1.503 us，+0.040% | 双方各 11/22；低于 5% 门槛，不支持 lazy 收益或回退 |

因此只推进 B 的 compete-first eager，不把 C 的 lazy 视为优化，也没有为 C 启动 I-cache PMU 对比。

#### 4.7.2 standalone 主路采用 compete-first eager

**[已保留] `0d08c437` — `优化(pa): standalone采用compete-first eager提交流程`**

当前时间线变为：

```text
EfDrain -> Claim -> 同步 eager callback 构参
        -> Materialize -> PrepareMap
        -> Fanin/Register -> WinnerBuild或AllocComplete -> Submit结束
```

Claim 与 Materialize 之间的构参时间使用现有边界离线求差，不新增 raw 字段。关闭泳道的 b256 五轮中位数从 3889.180 us 降到 3735.032 us，减少 154.148 us，约 **3.9635%**。这证明收益在 standalone 主路复现，不承诺真实 PA 有同一比例。

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

真实路径新增显式 begin/finish API 和 32 B 同步 ticket，同时保留旧 one-shot API 及原顺序，未把新语义强加给其他调用方。所有 worker 在 Claim 后仍完整构参，未采用 lazy 跳过。

真实 A5 level-1 三轮：

| 路径 | 三轮 | 中位数 |
| --- | --- | ---: |
| compete-first 最终版 | 4.843652 / 4.809211 / 4.805443 ms | 4.809211 ms |
| 原路径历史基线 | 4.821897 / 4.890447 / 4.752956 ms | 4.821897 ms |

中位数减少 12.686 us，约 0.263%，两组三轮波动区间重叠。因此当前结论是 **真实性能基本持平**，不是稳定的 0.263% 收益。保留该接口是因为阶段顺序和业务边界更清晰，并为后续精确取证提供基础。

真实 level-4 权威件：

```text
outputs/TestPagedAttentionUnroll_Case1_20260720_104406/
  l2_swimlane_records.json
  merged_swimlane.json
```

该轮包含 122880 个 Submit、945653 条事件、`dropped_records=0`，首末 Submit 为 5.066862 ms；atomic 闭合为：

```text
106355 = 109392 - 3361 + 324
```

它证明 compete-first 后的阶段、atomic 和离线加工一致，不是关闭观察后的净性能样本。

## 5. 当前保留优化汇总

| 优化 | 真实 PA 状态 | 当前可成立的效果结论 |
| --- | --- | --- |
| 首个 joint 前跳过 BlockWon 轮询 | [已保留] | PA 单 lane Case1 确定删除 146944 次无效 RMW；与参数 mask 的组合将 5.642245 ms 降至 5.115620 ms 中位数 |
| output/register mask 复用 | [已保留] | 组合结果正向；独立收益只具方向性，不作精确拆分 |
| HeapGuard 首圈 fast path | [已保留] | 确定删除 1024 次 frontier atomic load；真实十对配对中位 -0.324% |
| 低频 winner 冷分支 | [已保留] | 主要恢复 atomic 观察接入后的 loser 热路布局回退 |
| atomic record/PollBatch 冷路径外提 | [已保留] | 大幅缩小热函数和 `.text`，level-1 三轮中位恢复到 4.821897 ms，同时保持 level-4 闭合 |
| compete-first eager begin/finish | [已保留] | standalone -3.9635%；真实 PA 三轮只能判为基本持平 |

这些结果来自不同历史阶段，不能把表中百分比相加得到“总收益”。当前真实路径已经同时包含这些改动，后续基线必须从当前 HEAD 重新建立。

## 6. 当前观察能力

### 6.1 真实 PA `swimlane`：业务 span 与 atomic 合并观察

**[观察工具，已具备]**

- 普通业务阶段和 Atomic/PollBatch 位于同一 AIC/AIV scalar lane；
- schema-v4 以父区间和互斥子区间闭合，Kernel、Atomic 是不可加和 overlay；
- residual/未覆盖时间由离线工具使用已有相邻边界计算，不新增设备 record；
- raw 是权威数据，merged 只负责 Perfetto 可视化，exclusive analysis 负责整数闭合；
- `dropped_records != 0`、父子越界、Kernel 孤儿或 atomic 公式不闭合时，整轮无效。

`swimlane` 用于回答“时间落在哪个业务区域、atomic 调用次数和边界是否改变”，不作为无观察性能基线，也不直接给出 I-cache stall。

### 6.2 standalone `submit-pmu`：历史能力

**[观察工具，历史版本已具备；不是当前待办]**

现有历史版本能在独立 CCEC ELF 中编译掉泳道和逐 atomic 记录，采集每物理子核 PMU total、scalar busy、I-cache request/miss，并生成 96 核 raw 和自包含 HTML。历史 `none/claim/efdrain/materialize/register` 数据只对当时边界和各自 ELF 有效。

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

该目录可能不在当前机器保留，且从不随 Git 提交。上述数据只用于说明已经验证过的 PMU owner、逐核 raw、AIC/AIV 分组和 HTML 方法，不是当前重采 standalone 的要求，也不能把旧绝对值当成真实 PA 数据。当前工作只在真实 PA 上按当前 compete-first 代码和最新真实 span 建立新证据。

### 6.3 A5 当前不可获得的 `scalar_wait_ib_time`

**[受限]**

在本机 CANN 9.1、A5/DAV3510 上分别尝试 `PipeUtilization`、`PipeUtilization,MemoryDetail` 和 `Default` 三种正式 `msopprof` 入口，生成的 CSV 和 A5 正式事件表均没有 `scalar_wait_ib_time` 或 `scalar_wait_time`。当前不能套用 A2/A3 的事件号或字段含义。

对应历史目录：

```text
tests/atomic_probe/pa_scheduler/outputs/wait_ib_official_msopprof_20260719_b1_probe2/
tests/atomic_probe/pa_scheduler/outputs/wait_ib_official_msopprof_20260719_b1_probe3_memory_detail/
tests/atomic_probe/pa_scheduler/outputs/wait_ib_official_msopprof_20260719_b1_probe4_default/
```

因此 `PMU total - scalar_busy` 只能叫“非 Scalar-busy 残余”，不能命名为 scalar 空闲、wait vector 或 I-cache stall。

## 7. standalone 历史证据与真实 PA 的边界

**[历史证据，不是当前待办]**

standalone 已完成其方法验证职责：证明多后端调度模型、真 Cube/Vector 负载、atomic 泳道、PMU owner、I-cache 标尺和排他区间工具可以工作。当前不再继续优化、扩充或重采 standalone；下列对等关系只用于解释为什么历史经验可以作为真实 PA 实施时的参考，以及哪些结论绝不能外推。

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

依赖不是按 task 名硬编码跳过：SF 依赖 QK，PV 依赖 SF，UP 去重后依赖 Alloc、SF、PV；每 batch fanin 边数为 5，b256 全局为 1280。

### 7.2 尚未对等部分

- 真计算 workspace 使用统一受控输入，数值没有按真实 QK→SF→PV→UP 数据流串接；
- 该历史模型只覆盖 Case1 单 block group、`q_loop=1` 和全单-lane 图；
- joint/mixed、多 group、多 q-loop 和跨迭代更新没有被完整模拟；
- synthetic heap、独立 ELF 布局和 host 启动状态与真实 simpler 不同；
- Kernel span 包含 engine launch/completion wait wrapper，不等于纯 Cube/Vector 指令时间；
- standalone 没有真实 PA 的 loser replay 业务动作，不应为追求图形对称而伪造该 span。

历史推进中，standalone 曾用于先验证接口、边界、计数、控制协议和候选方向，再迁真实 PA 做同构正确性与性能 A/B；它的收益比例始终不能直接外推。当前这些方法能力已经完成验证，后续真实 PA 三证据链不再把新增 standalone 实现或复测设为前置门禁。

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
- 继续复用现有 host/device header 传输，但 device header 固定只有 **6976 B**，`records_per_core=0`，没有逐事件记录；
- host hook 随该构建重新编译并导出，不复用普通诊断构建的旧 host 产物；
- 候选保留或撤回最终由该构建决定。

第一版曾尝试设置 `PTO2_PROFILING=0`，在编译期触发 Arg 布局/cacheline `static_assert`，没有进入设备执行。该尝试已经完整撤回；这证明 `PTO2_PROFILING` 不只是可随意关闭的观察开关，不能为了减少诊断代码破坏公开 ABI。最终方案只关闭上述 FDWIC 观察路径，不能表述为“所有 PTO2 profiling 已移除”。

构建身份也已做双向 ELF 审计：perf ELF 含 `dist_perf_clock_expect_submits` 标记，并且不含 FDWIC swimlane、atomic 观察和平台 PMU 符号；普通 level-4 ELF 含正常泳道/atomic 符号，但不含 perf-clock 设备符号。普通构建还做了前后布局复核：旧 AICore cache 身份 `6c55004bc91e15f0` 与新 AICore cache 身份 `110ff0c62a3adcf7` 的 `.text` 均为 `0x31a50` B，两个 ELF 中 `.text` 的 96 条 FUNC symbol 记录在地址、尺寸、绑定、可见性和名称上完全一致；`.text` 仅有 AIC/AIV 两份 `aicpu_orchestration_entry` 各一处单字节从 `0xa2` 变为 `0xa9`，对应 `PTO2_SCOPE` 源码行号从 162 移到 169，原因是前置新增 7 行。没有观察到普通构建新增函数或代码尺寸膨胀；这里也不声称工具已经给出完整指令反汇编一致性。

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

该 profile 必须独占：不要同时传 `--enable-l2-swimlane`、`--enable-pmu`、`--use-example-exec-time` 或其他诊断开关；命令行门禁会直接拒绝混用。每次只允许 `--rounds 1`，多轮基线必须由独立 pytest 进程取得。该 profile 会按源码指纹自动重编 AICore override，但不会代替安装流程重编 host runtime；新环境首次复现前必须先重建 `libhost_runtime.so`，并确认三个 `fdwic_perf_clock_host_*` hook 已导出。

最小 B1 两次有效 A5 闭合如下；两次都是 96 个物理子核、每核恰好 5 次 Submit：

| 产物时间戳 | 完整 Submit |
| --- | ---: |
| `20260720_172436` | 75.347 us |
| `20260720_172615` | 73.716 us |

Case1 golden 在 `20260720_172820` 通过，96 核均为 1280 Submit，原始 `SYS_CNT` 为 **4,489,247 ticks = 4489.247 us**。JSON 浮点输出采用默认 6 位有效数字，因此 Case1 量级显示为 `4489.25`，而 B1 仍可显示 `73.716`；后续精确分析应以 raw tick 除以 1000 为准。

Case1 golden 通过后，另起五个独立进程并使用 `--skip-golden` 得到本阶段干净基线；**以下五次不包含上述 golden 样本**：

| 产物时间戳 | raw ticks | 完整 Submit |
| --- | ---: | ---: |
| `20260720_173140` | 4,495,677 | 4495.677 us |
| `20260720_173224` | 5,808,500 | 5808.500 us |
| `20260720_173307` | 5,342,774 | 5342.774 us |
| `20260720_173350` | 4,752,765 | 4752.765 us |
| `20260720_173433` | 4,823,114 | 4823.114 us |

五次均为 96 核、每核 1280 Submit；中位数 **4823.114 us**，最小值 **4495.677 us**，最大值 **5808.500 us**。该分布是后续候选做独立进程、交错 A/B 的起点，不能只挑 4489 us 的最好值作为稳定基线。

还完成两项边界门禁：

- Case2 负测试实际得到每核 576 Submit，而当前期望值为 320；host 按 fail-closed 拒绝结果且不生成成功 summary。它只证明计数不符时不会产出伪成功结论，不能用于评价 Case2 性能；
- 同一真实源码的普通 level-4 B1（`20260720_173738`）得到 480 个 Submit、完整 Submit **85.653 us**、`dropped=0`，排他闭合 `PASS`。它证明 perf-clock 的首尾边界和逐核计数与普通泳道来自同一执行语义；由于两者是不同 ELF，不能用 `85.653 - 73.716` 计算观察开销。

本阶段的源码、构建身份、B1、Case1、负测试和普通泳道同源边界已经闭合，并已完成独立源码审阅。后续进入真实 `swimlane` 构建复核与 `submit-pmu-none`，不在该工具阶段顺带铺开 PMU 代码。

### 8.2 `swimlane`

**[观察工具，已实现]**

保留普通阶段和 atomic 合并泳道，用于回答：

- 收益或回退可能落在哪个业务 span；
- atomic 逻辑调用、物理记录和 PollBatch 是否变化；
- Kernel 落点、父子区间、逐核 task 连续性和记录容量是否正常。

它不决定候选的净性能，不与 perf-clock 的绝对时间相减。

本阶段没有调整设备端 span、atomic wrapper、raw ABI 或记录容量。审查发现原先 SceneTest 在 converter/analyzer 返回失败时只记录 warning，pytest 仍可能显示 PASS；而阶段序列、Kernel 唯一归属和六类整数闭合正是在该离线步骤中完成。现已将真实 A5 FDWIC level-4 成功用例改为 fail-closed：raw 缺失、converter 失败、`merged_swimlane.json` 或 `swimlane_exclusive_analysis.json` 缺失/为空都会使该用例失败；若设备执行本身已经失败，则保留原始异常，离线转换不覆盖根因。

普通 trace-capable AICore ELF 也增加了正向身份门禁：必须含 `fdwic_atomic_poll_boundary_slow` 与 `fdwic_swimlane_detail_record_atomic` 两个已定义观察慢体，并且不得含 `dist_perf_clock_expect_submits`。这没有新增一套等价的 swimlane profile：普通 level-0 与 level-4 仍共享同一个 trace-capable ELF，采集模式由运行时 level 决定；门禁只防止误拿 perf-clock 或不完整产物。raw 中的 `trace_schema_version=4`、`l2_swimlane_level=4` 和 atomic 元数据继续证明运行模式。

当前源码先用真实 B1 验证结构门禁：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260720_234158/
```

该轮为 96 核、每核 5 个 Submit、4,559 条记录、`dropped=0`，所有父子关系、Kernel 归属和整数闭合均 PASS。它的完整 Submit 为 302.072 us，明显受本轮冷启动/轮询状态影响，只作为结构门禁，不替代 perf-clock 性能基线。

随后只运行一次当前 HEAD 的完整 Case1 并保留 golden 校验，权威第二证据链产物为：

```text
outputs/TestPagedAttentionUnroll_Case1_20260720_234305/
  l2_swimlane_records.json             75,397,613 B
  merged_swimlane.json                182,110,972 B
  swimlane_exclusive_analysis.json        121,264 B
```

该轮为 32 AIC + 64 AIV、每核 1,280 个 Submit、全局 122,880 个 Submit，944,874 条 raw 记录且 `dropped=0`；全局首末 Submit 为 **5,095.821 us**。Submit、Submit envelope、EfDrain、OrchestrationReplay、FinalDrain 和 WorkerCompletion 六类整数分区全部精确闭合。Atomic 物理记录、批处理轮询与逻辑调用满足：

```text
105577 - 330 + 3899 = 109146
```

其中 merged 中 `return_ready/source_issue/PollBatch` 分别为 102,495/2,752/330。Atomic 仍是不可加的 overlay；PollBatch 表示完整等待区及其精确调用次数，不能用 duration 反推单次 atomic 延迟。

本轮设备端仍预留固定 **201,333,568 B** trace buffer。实际 raw 没有再增加字段，但该固定容量和约 182 MB merged 进一步说明：swimlane 只在需要业务/atomic 定位时采集，不能作为权威性能基线，也不应为普通 A/B 反复生成。生产 converter/analyzer 回归与新增 fail-closed/ELF 门禁共 86 项通过。

### 8.3 `submit-pmu-none` 与真实 span 单阶段 PMU

**[`submit-pmu-none`、`arg-build`、空 bracket 校准、`materialize`、`claim` 和 `register` 均已完成 A5 收口]**

本阶段没有修改或复测 standalone，而是在真实 PA 中建立独立诊断构建。该构建在编译期去除普通泳道、atomic 观察和通用逐 task PMU ring，分为两种运行方式：

1. **`submit-pmu-none`**：每物理子核在完整 Submit 调度期只 start/stop 一次，不做中途 shadow read-clear；输出 96 核 PMU total、scalar busy、I-cache request/miss，并按 AIC/AIV 生成 raw 与 HTML；
2. **真实 span 单阶段 PMU**：一次 ELF 只选择一个当前真实泳道区域做局部观测，仍同时保留本 ELF 自己的完整 Submit primary，局部只与本 ELF、本轮、本角色的 primary 和时间分母比较。首个 selector 为 Claim 完成到 Materialize 入口之间的 `arg-build`。

#### 8.3.1 已闭合的 `submit-pmu-none`

入口为 `--fdwic-profile submit-pmu-none`，当前只接受真实 A5、FDWIC、level 2、`rounds=1`。构建保留公开 `PTO2_PROFILING` Arg ABI，但固定 `PTO_FDWIC_TRACE_ENABLED=0`、`PTO_FDWIC_SUBMIT_PMU=1`；最终 ELF 必须包含 `dist_submit_pmu_expect_submits` 和 `fdwic_submit_pmu_read_counters`，并拒绝 perf-clock、swimlane/atomic、通用 PMU ring 和通用 PMU reg-base 符号。

每个物理子核在 attach 时先 stop/清计数；首个 Submit 读取 1 ns `SYS_CNT` 后开启 PMU，末个 Submit stop 后读取 CNT2/CNT6/CNT7，并用 CNT8/CNT5 做 request/miss 影子复核。FinalDrain 不进入有效窗口。AICPU owner 在发布 worker 运行状态前，逐核保存并配置 PMU 控制寄存器和 selector；96 个 worker 完成后逆序恢复。正式 raw 要求 32 AIC、64 AIV、96 个唯一物理 ID、32 个完整 1:2 mixed triplet、配置/恢复 96/96、active-after-restore=0、每核 Submit 次数和窗口状态全部闭合。

实现过程中有两次由门禁揭示并修正的接口问题：

1. PA orchestration 原先只在 `PTO_FDWIC_PERF_CLOCK` 条件下声明预期 Submit 数，导致 submit-PMU 首轮实测为 count `5/0`、窗口未启动；修正为 perf-clock 与 submit-PMU 复用同一真实挂点，不在 host 猜测次数。
2. host `Runtime::workers[].physical_core_id` 是 H2D 前的 host shadow，不会在 export 前从设备 Runtime 整块回拷；真机核 1 已实证 device record 为 physical 1、host shadow 仍为 0。该无效比较已移除。逻辑核到物理核的可信关系由设备 AICPU owner 校验、每核 record、唯一集合、角色和 triplet 三层闭合，不新增冗余映射字段。

B1 两次独立成功采集都通过 96 核、5 Submit/core、owner restore 和报告门禁：

| 产物 | 全局 Submit | AIC total/core mean | AIC scalar/core mean | AIV miss/core mean |
| --- | ---: | ---: | ---: | ---: |
| `..._002939` | 74.882 us | 28,778.9 cycles | 25,099.7 cycles | 129.56 |
| `..._003050` | 227.673 us | 36,999.4 cycles | 25,318.8 cycles | 129.84 |

第二轮的 AIC total 最大值从 122,226 增至 375,185 cycles，而 AIC scalar mean 仅增加约 0.87%；AIV request/miss 也基本不变。它说明“设备独占”并不等于每核到达相位和非 scalar-busy 等待恒定，是后续波动专项归因的输入，当前不把两点样本直接写成因果结论。

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

全局首个 Submit 到最后一个 Submit 为 **5,075.360 us**，每核 1,280 次 Submit，primary/shadow 96/96 相等；最大可编程计数 7,129,295，远小于 `0x3fffffff` 风险阈值。按 90 ns/miss 只能得到单核串行等效量级，不能把 AIV 约 1.525 ms/core 直接写成可消除的墙钟损失。raw 约 46 KB、HTML 约 77 KB，没有复用约 200 MB 的泳道 buffer。

提交前还用当前工作树分别回归了 B1 perf-clock 与普通 level-4：perf-clock 为 96×5 Submit、254.084 us；level-4 为 4,549 条事件、89.071 us、`dropped=0`，排他闭合 PASS。两者用于证明三种 ELF 的双向隔离和公共 Submit 挂点没有回退，绝对时间仍不得跨 ELF 相减。

#### 8.3.2 首个真实 selector：`submit-pmu-arg-build`

入口为 `--fdwic-profile submit-pmu-arg-build`，编译期固定：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=1
PTO_FDWIC_TRACE_ENABLED=0
```

Kernel 与 Alloc 两条 compete-first 路径都使用同一组源码业务边界：起点在 Claim 完成后，终点在匹配 Finish 恢复并校验 ticket 后、Materialize 入口前。该区间覆盖 `dist_submit_make_ticket()`、Begin 返回、同步 eager callback 的 `build_args(args)`、Finish 重入、`dist_submit_restore_from_ticket()` 和 ticket 校验；不包含 Claim 本体和 Materialize 本体。它复用了泳道 `Claim.end -> Materialize.begin` 的业务语义，但 PMU ELF 已编译掉泳道 record，二者不是同一 ELF，也不做逐 tick 对齐。

`submit-pmu-none` 保持原 128 B 前缀加 `96 × 64 B` 整窗记录，共 6,272 B；`arg-build` 在其后追加 `96 × 64 B` phase sidecar，总计 12,416 B。每个 worker 仍独占 cacheline，只在窗口结束后发布一份汇总，不生成逐事件记录，也不复用约 200 MB 的泳道 buffer。CNT6/CNT7 是窗口中从不读取的整窗 primary；CNT8/CNT5 作为 running read-clear shadow，begin 样本只进入 shadow whole，end 样本同时进入 shadow whole 和 phase observed，stop 后 tail 只进入 shadow whole。

这里特意不把 phase request/miss 称为业务事件数的“严格下界”。counter read 与 `SYS_CNT` 边界之间仍有少量观测 bookkeeping，它们的取指会进入 observed sample；`primary - shadow` 只量化分段重建的 capture gap，不能抵消插桩自身的事件。报告因此同时展示 `observed` 与 `observed + (primary - shadow)`，二者是当前插桩 ELF 的观测值和加全窗 capture gap 后的敏感性量尺，不是原业务区间的数学上下界。phase 只提供 `SYS_CNT` 时间及 I-cache request/miss observed，不杜撰局部 PMU total、scalar busy 或 I-cache stall 时间。

正式 raw 只有在逐核满足以下条件时发布：

- `phase_begin_reads == phase_end_reads == expected_submit_count`；B1 为 5，Case1 为 1,280；
- phase status 为 `0x3f`，即 requested、边界平衡、调用 shape、数值顺序、时间落在本核 Submit 内和 tail read 六项全部成立；
- phase observed 分别不超过对应 shadow，shadow 不超过 primary；本阶段实测 primary/shadow 精确相等，但通用契约仍只要求单向闭合；
- 可编程计数与每段最大读数都低于 `0x3fffffff` 风险阈值；
- phase 时间非零且不超过同核 `submit_elapsed_ticks`。

两轮独立 B1 结构样本分别位于：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260721_014154/
outputs/TestPagedAttentionUnroll_CaseB1_20260721_014301/
```

两轮均为 96 核、每核 5 次 begin/end、status `0x3f`、primary/shadow 96/96 精确相等；全局 Submit 分别为 247.205 us 和 80.702 us，phase core-time 份额分别为 6.862% 和 7.432%。这两点证明结构闭合，也再次暴露独占设备仍有明显到达/等待波动；不能拿两轮绝对时间直接形成性能结论。

Case1/B256 正式件为：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_014355/
  fdwic_submit_pmu_raw.json       72,811 B
  fdwic_submit_pmu_report.html    79,497 B
```

该轮全局 Submit 为 **4,964.039 us**，96 核每核 1,280 次，共 122,880 次 begin/end 全部闭合，primary/shadow 96/96 精确相等，最大 shadow request/miss 分段为 32,136/129。按同一 ELF、同一轮聚合，`arg-build` 的 core-time、request observed、miss observed 份额分别为：

| 角色 | core-time | request observed | miss observed |
| --- | ---: | ---: | ---: |
| ALL | 5.557% | 20.716% | 21.334% |
| AIC | 4.297% | 20.099% | 10.640% |
| AIV | 6.183% | 21.031% | 21.417% |

这组数据说明 `arg-build` 在当前插桩 ELF 中取指观测份额高于时间份额，值得在完成空 bracket 校准后继续判断观察 bookkeeping 占比；目前不能把约 21% 直接写成原业务 I-cache 事件比例，更不能把 4.964 ms 与 none、perf-clock 或泳道绝对时间相减。

最终还分别回归了三条互斥证据链：`submit-pmu-none` B1 `20260721_014602` 保持 96/96 primary=shadow 且不含任何 phase 字段；perf-clock B1 `20260721_014715` 为 96×5 Submit、73.029 us；普通 level-4 B1 `20260721_014841` 为 4,560 条事件、90.275 us、`dropped=0`、排他闭合 PASS。这些结果证明 phase sidecar、reader 和状态只进入选中的 `arg-build` ELF；三轮绝对时间仍不互相相减。

#### 8.3.3 running bracket 空区间校准：`submit-pmu-empty-bracket`

**[观察工具，已完成两轮 B1 与两轮 Case1 实测闭合]**

`arg-build` 的 running read-clear 会在每次阶段 begin/end 各读一次 shadow counter。在继续扩展业务 selector 前，本阶段先回答一个基础问题：同一套 begin/end 观察器紧邻执行、其中不包任何业务体时，会在真实 A5 PA 中形成怎样的稳定时间和 I-cache 记录开销量级。入口为 `--fdwic-profile submit-pmu-empty-bracket`，编译身份固定为：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=2
PTO_FDWIC_TRACE_ENABLED=0
```

host/runtime 契约使用独立 mode `3`、phase id `2`，phase 名称和边界分别为 `empty-bracket`、`claim_end_adjacent_empty_bracket`。真实 hook 没有另造调用拓扑：Kernel 与 Alloc 的 compete-first Begin 都在现有 Claim 完成、`claim_end` 已取到的同一源码调用点，紧邻执行一次原样 generic phase begin/end；每个 Submit 恰好一对，B1 每核 5 对，Case1 每核 1,280 对。其他 profile 中该 wrapper 编译为空。

时间和 PMU 计数必须按两种不同边界解释：

- generic begin 在第一次 shadow read-clear 与 begin bookkeeping 之后读取内部 `SYS_CNT`，generic end 在第二次 shadow read-clear 之前读取内部 `SYS_CNT`；若只看这两个内部 tick，空区间并没有覆盖完整的观察器调用成本；
- empty wrapper 因此保存旧累计值和 begin/end 次数，用外层两个 `SYS_CNT` 包住完整 generic begin/end 对，校验次数各恰好增加 1、状态平衡且时间无溢出，再用外层 delta 覆盖本次内部 elapsed 增量。raw 明示 `time_semantics=outer_sys_cnt_around_adjacent_begin_end_pair`；
- request/miss 仍保持原 running read-clear 口径：begin 样本只进入 whole shadow，end 样本同时进入 whole shadow 与 phase observed。外层 `SYS_CNT` 不改变这组读清边界，raw 明示 `counter_semantics=running_read_clear_empty_bracket_calibration`。

所以 elapsed 是“外层时间戳包住完整相邻 begin/end 对”的经验耗时，request/miss 是“两次 shadow read-clear 之间”的观察值；二者不是同一个精确指令边界。外层时间也包含两个 `SYS_CNT` 自身的底噪，不能称为观察器数学最小成本。

该 profile 复用 `arg-build` 已有的 phase sidecar，没有增加设备 raw 字段、逐事件 record 或泳道 ring。`submit-pmu-none` 仍为 128 B header 加 `96 × 64 B` whole record，共 6,272 B；`empty-bracket` 与其他单阶段 profile 同为再追加 `96 × 64 B` sidecar，共 12,416 B。每个 worker 最终仍只发布一份 whole 汇总和一份 phase 汇总，因此本次校准没有继续放大约 200 MB 的泳道数据。

正式发布继续 fail-closed：96 核必须满足 owner 配置/恢复、唯一物理核与 32 个 mixed triplet 闭合；逐核 begin/end 次数都等于预期 Submit 数，phase status 为 `0x3f`，边界平衡、shape、数值顺序、时间落在本核 Submit 内和 tail read 全部成立；phase observed 不超过 shadow，shadow 不超过 primary，计数不越风险阈值。以下四轮均为 32 AIC + 64 AIV、status `0x3f`、primary/shadow 96/96 精确相等，因而 capture gap 为 0。

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

B1 只有每核 5 对，首次进入 reader、对应调用点和相关代码布局时的冷取指占比很高；其 request 为 69.194～84.844/对，明显高于 Case1 稳定的约 49.34/对，elapsed 也从 Case1 的约 639～640 ns/对升至约 719～725 ns/对。因此 B1 继续只作为结构、次数和冷启动门禁，不能替代 Case1 的稳态观察尺度。

Case1 两轮还复现了稳定的角色差异：AIC 约 568 ns、48.9 request、0.008 miss/对，AIV 约 675～677 ns、49.6 request、2.01～2.03 miss/对。当前证据只证明同一观察实现对 AIC/AIV 形成不同且可复验的记录开销；它没有证明差异必然来自 reader 跨 I-cache line、某个固定冲突或 PMU 事件定义，后续若归因必须另做同构单变量证据。

为决定是否应在本阶段调整 reader，另对 empty 最终 ELF 做了只读核验。AIC/AIV 的 `fdwic_submit_pmu_phase_read_shadow_counters` 都只有一份、均为 92 B；两个 relocatable object 中该函数机器码逐字节相同，最终 ELF 仅因 block-local relocation 出现一个立即数字节差异。128 B line 下，AIC reader 起址行内偏移 76 B，AIV 为 120 B，两者都跨两行，所以“跨行”不能单独解释只有 AIV 约 2 miss/对。本机 CANN 9.1 的 DAV3510 模型配置显示 scalar I-cache 均为 4-way，但 AIC 为 32 KiB/64 sets、AIV 为 16 KiB/32 sets；对应 combined `.text` 又分别为 68,024 B 和 82,000 B。这些证据支持容量、角色代码和具体布局共同形成不同的记录开销，但聚合 PMU 仍不能定位到 reader 的某一条 cache line。因此本阶段保留单份 noinline reader：不为追求较小数字而 inline 复制热路径，也不以强制对齐改变整份诊断 ELF 的冲突集合。若以后局部信号确实被该量级淹没，应另做只改 reader 对齐的 empty A/B，而不是混入本次校准提交。

该校准不能直接从 `arg-build` 中扣除。首先，两者的时间边界不同：`arg-build` elapsed 是两侧 observer 之间的内部业务区间，而 empty elapsed 用外层 tick 包住完整 begin/end 对；二者相减会把不同对象当成同一加法模型。其次，即使 request/miss 都来自 running read-clear，两个 profile 仍是不同 ELF，代码布局、冷暖状态和前端竞争都可能改变事件数。方向上，Case1 empty 约 49 request/对，相当于同一时期 `arg-build` AIC/AIV 约 118.31/121.36 request/对的四成；AIV empty 约 2.02 miss/对，相当于 `arg-build` 约 3.80 miss/对的一半。这只说明观察器污染不可忽略，不是允许产出“扣除 empty 后的业务净 request/miss”。报告和后续结论都只保留原始 observed、capture gap 与这份经验尺度。

空 bracket 代码落定后又在同一工作树上串行回归了四种互斥构建：`arg-build` B1 `20260721_021930` 为 96×5 对、status `0x3f`、primary/shadow 96/96 精确相等，并带有新增的 `time_semantics`，全局 Submit 为 249.064 us；`submit-pmu-none` B1 `20260721_022026` 不含 phase 字段、primary/shadow 96/96 精确相等，全局 Submit 为 265.977 us；perf-clock B1 `20260721_022115` 为 96×5 Submit、78.230 us；普通 level-4 B1 `20260721_022221` 为 4,546 条事件、88.595 us、`dropped=0`、排他闭合 PASS。四轮只证明新增 mode、元数据和公共 Claim.end 空 wrapper 没有破坏既有证据链，B1 冷启动绝对时间仍不得跨 ELF 相减。

#### 8.3.4 第二个真实业务 selector：`submit-pmu-materialize`

**[观察工具，已完成两轮 B1 与两轮 Case1 实测闭合]**

本阶段不是从历史 `claim/efdrain/materialize/register` 名单中顺次取一个旧 phase，而是重新查看当前 compete-first 真实 PA 的最新 Case1 排他结果 `outputs/TestPagedAttentionUnroll_Case1_20260720_234305/` 后再决定。该轮 `Materialize` 为 97,467,035 aggregate core-ticks，占 `SubmitUnion` 399,604,449 ticks 的 **24.391%**，是已具备明确源码起止边界的最大业务 span；同轮 `Claim`、`EfDrain`、`Register` 分别占 19.887%、15.853% 和 12.010%。已经完成的 `arg-build` 则覆盖 `Claim.end -> Materialize.begin`，二者首尾相接但不重叠。因此这一轮选择来自最新真实布局和可复用边界，不是复活 standalone 或旧 schema 中的 selector 顺序。

入口为 `--fdwic-profile submit-pmu-materialize`，编译身份固定为：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=3
PTO_FDWIC_TRACE_ENABLED=0
```

host/runtime 使用独立 mode `4`、phase id `3`，phase 名称和边界分别为 `materialize`、`materialize_begin_to_materialize_end`；counter/time 口径分别为 `running_read_clear_observed_bracket` 和 `inner_sys_cnt_between_boundary_observers`。这仍是一轮只打开一个业务 phase 的诊断 ELF，不同时采 Claim、EfDrain 或 Register。

设备端没有另造一条近似路径，而是在四个真实入口精确打开同一个 phase：

1. 旧 Kernel `dist_submit_impl()` 在 EfDrain 完成、进入 Materialize 前 begin；
2. 旧 Alloc `dist_alloc_tensors()` 在相同业务边界 begin；
3. compete-first Kernel finish 在 ticket 恢复和校验成功、`materialize_begin` 取时后 begin；
4. compete-first Alloc finish 在相同业务边界 begin。

四条入口统一调用 `dist_submit_materialize_and_prepare_map()`；唯一成功 end 位于该 helper 内部，在 `dist_submit_check_task_cap()` 和 `dist_submit_materialize_args()` 均成功返回后、泳道 `materialize_end` 取时前。因此观测区间包含 task-cap 检查、tag/output/register-mask 扫描、heap ring 布局、输出 Tensor 初始化及 `heap_next` 推进，但不包含后继 `PrepareMap`。submit-PMU 构建已编译掉泳道 record，所以旧入口 begin 与 helper 之间的 trace 宏不会给本 ELF 增加一条实际记录。

失败路径刻意不伪造 end：task-cap 或参数、heap、输出物化任一检查失败时，helper 直接返回，遗留的 armed phase 会使 begin/end 不平衡、调用 shape 或最终 status 闭合失败；设备发布、host 校验和 HTML 加工据此 fail-closed，不能把一个被截断的 Materialize 当成有效短样本。正常 PA Case1 中 Materialize 每个 Submit 固定执行一次，所以正式 shape 必须严格为 96 核、每核 1,280 次 begin 和 1,280 次 end，全局各 122,880 次；B1 则固定为每核 5 次、全局各 480 次。

该 profile 复用已有 phase sidecar，没有扩展 raw ABI：`submit-pmu-none` 仍为 128 B header 加 `96 × 64 B` whole record，共 6,272 B；`materialize` 与 `arg-build/empty-bracket` 一样只再追加 `96 × 64 B` phase record，总计 12,416 B。没有增加逐 Submit 记录、泳道字段或约 200 MB 的 trace ring。

两轮 B1 先验证真实挂点、固定 shape 和冷启动下的数值闭合：

| 产物 | 全局 Submit | ALL 每次 elapsed | ALL 每次 request | ALL 每次 miss | 同 ELF 时间/request/miss 占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `..._024533` | 82.091 us | 912.846 ns | 333.183 | 8.225 | 20.432% / 45.075% / 21.321% |
| `..._024641` | 81.741 us | 896.398 ns | 267.810 | 7.369 | 20.932% / 42.404% / 20.485% |

两轮都是 32 AIC + 64 AIV、每核 5 次 begin/end、phase status `0x3f`，96/96 primary/shadow 精确相等，最大单段 shadow request/miss 分别为 3,068/85 和 3,040/78。B1 仍只用于结构、次数、冷启动和快速门禁，其每次 request/miss 不外推 Case1 稳态。

两轮完整 Case1 产物为：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_024748/
outputs/TestPagedAttentionUnroll_Case1_20260721_024909/
```

两轮均为 96 核、每核 1,280 次 begin/end、phase status `0x3f`，owner 配置/恢复、32 个 mixed triplet、固定 shape、数值顺序、phase 时间和风险阈值全部闭合；primary/shadow 96/96 精确相等，capture gap 为 0。第一轮 raw/HTML 为 72,947/80,049 B，第二轮为 72,958/80,049 B。逐角色每次调用的原始 observed 为：

| 产物 | 全局 Submit | 角色 | 每次 elapsed | 每次 request | 每次 miss |
| --- | ---: | --- | ---: | ---: | ---: |
| `..._024748` | 4,922.142 us | ALL | 797.814 ns | 233.197 | 1.474 |
| 同上 | 同上 | AIC | 776.265 ns | 228.942 | 0.022 |
| 同上 | 同上 | AIV | 808.588 ns | 235.325 | 2.201 |
| `..._024909` | 4,851.282 us | ALL | 797.061 ns | 233.238 | 1.418 |
| 同上 | 同上 | AIC | 775.653 ns | 228.170 | 0.021 |
| 同上 | 同上 | AIV | 807.765 ns | 235.772 | 2.116 |

“阶段占比”只使用同一 ELF、同一轮、同一角色的数据：时间分子为 phase elapsed core-time、分母为逐核首末 Submit elapsed core-time；request/miss 分子为 phase observed、分母为本轮整窗 primary。两轮结果为：

| 产物 | 角色 | 时间占比 | request observed 占比 | miss observed 占比 |
| --- | --- | ---: | ---: | ---: |
| `..._024748` | ALL | 22.560% | 37.688% | 12.056% |
| 同上 | AIC | 22.540% | 36.532% | 10.487% |
| 同上 | AIV | 22.570% | 38.278% | 12.065% |
| `..._024909` | ALL | 22.105% | 37.741% | 11.681% |
| 同上 | AIC | 21.882% | 36.442% | 10.997% |
| 同上 | AIV | 22.214% | 38.404% | 11.685% |

两轮约 22.1%～22.6% 的同 ELF 时间份额和约 37.7% 的 request 份额可以说明 Materialize 是当前诊断布局中的重要取指区域；它们不能直接等同于关闭插桩后的净业务成本。尤其 empty-bracket 两轮 Case1 测得的记录开销量级约为 ALL 639～640 ns、49.34 request、1.34～1.36 miss/对，而 materialize 约为 797 ns、233.2 request、1.42～1.47 miss/次。empty elapsed 用外层 tick 包住完整 observer 对，materialize elapsed 是两个 observer 内侧的业务时间；request/miss 即使都来自 running read-clear，也属于不同 ELF、不同布局和不同缓存状态。故 empty 只能提示观察器的自扰动量级不可忽略，绝不能从 materialize 中相减得到“净时间”或“净 miss”。按角色看，AIV Materialize 为 2.20/2.12 miss/次，empty-bracket 为 2.01/2.03 miss/对，也仍处在同一量级；当前结果不能证明 Materialize 业务体带来了明确的 AIV miss 增量。AIC Materialize 只有约 0.022/0.021 miss/次，同样只保留原始观测，不作跨 ELF 扣减。

代码落定后串行回归了五类互斥 B1 构建：

| 构建 | 产物 | 结果 |
| --- | --- | --- |
| `submit-pmu-arg-build` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_025027/` | 96×5、status `0x3f`、primary/shadow 96/96 精确相等；257.392 us |
| `submit-pmu-empty-bracket` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_025125/` | 96×5、status `0x3f`、primary/shadow 96/96 精确相等；230.313 us |
| `submit-pmu-none` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_025214/` | 无 phase 字段、primary/shadow 96/96 精确相等；298.298 us |
| `perf-clock` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_025305/` | 96×5 Submit、ELF 身份和调用 shape 闭合；274.997 us |
| 普通 level-4 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_025353/` | 4,550 条事件、89.109 us、`dropped=0`、排他整数闭合 PASS |

这些回归只证明新增 mode、四个 begin 挂点和统一成功 end 没有破坏既有构建身份、phase 契约或泳道加工；B1 的跨核到达/等待波动很大，五种 ELF 的绝对时间仍不得互相相减。

最后将 AICPU header 校验从逐个枚举旧 phase mode 等价收敛为复用 `fdwic_submit_pmu_mode_has_phase()`，使 mode 判定与公共 phase/字节数契约只有一个事实来源。该重构没有修改计数器配置、业务边界或设备 ABI；最终源码再次运行 `submit-pmu-materialize` B1：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260721_025856/
```

结果为全局 Submit **281.494 us**、96 核各 5 次，begin/end 全局 480/480、status `0x3f`、primary/shadow 96/96 精确相等。它是最终源码状态的 materialize 回归；前两轮 B1 仍保留为最初四挂点实现的独立结构样本。

#### 8.3.5 第三个真实业务 selector：`submit-pmu-claim`

**[观察工具，已完成两轮 B1 与两轮 Case1 实测闭合]**

本阶段继续按当前真实 Case1 排他布局选择观察对象，而不是复刻历史 phase 名单。`outputs/TestPagedAttentionUnroll_Case1_20260720_234305/` 中，`Claim` 为 79,470,788 aggregate core-ticks，占 `SubmitUnion` 399,604,449 ticks 的 **19.887%**；在 `Materialize` 完成取数后，它是剩余具备固定调用 shape 和明确源码边界的最大业务 span。同轮 122,880 条 Claim 与 122,880 条 Submit 一一对应，适合继续复用现有每 Submit 一对 begin/end 的强闭合契约。

入口为 `--fdwic-profile submit-pmu-claim`，编译身份固定为：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=4
PTO_FDWIC_TRACE_ENABLED=0
```

host/runtime 使用独立 mode `5`、phase id `4`，名称和边界为 `claim`、`claim_begin_to_claim_end`；counter/time 口径继续使用 `running_read_clear_observed_bracket` 和 `inner_sys_cnt_between_boundary_observers`。四条真实 API 路径分别在已有泳道 Claim 业务边界内打开和关闭同一个编译期 phase：

1. 旧 Kernel `dist_submit_impl()`：`prepare_map_end` 后 begin，包围 `dist_submit_claim(Kernel)` 和 `claim_flags` 构造，在 `claim_end` 取时前 end；
2. 旧 Alloc `dist_alloc_tensors()`：`register_end` 后 begin，以相同方式包围 Alloc Claim，在 `claim_end` 前 end；
3. compete-first Kernel begin：`efdrain_end` 后 begin，先执行属于现有 Claim span 的 `dist_submit_check_task_cap()`，再执行短路后的 Kernel Claim 和 flags，最后 end；
4. compete-first Alloc begin：同样从 `efdrain_end` 开始，包含 Alloc task-cap、Claim 和 flags，最后 end。

compete-first 原代码虽在声明 `claim_begin` 前计算 `ready`，但泳道的 Claim 起点本来就是更早取得的 `efdrain_end`，所以 task-cap 已经计入 Claim span。本阶段只把该计算移到 phase begin 之后，使 PMU 与现有业务边界一致，没有改变 task-cap 与 Claim 的执行顺序。四条路径的 end 都放在 `claim_flags` 形成后、泳道 `claim_end` 取时前；submit-PMU ELF 中 trace record 已编译去除，不会把 Claim/前序 record 发布混入局部计数。

四条边界之间都没有会绕过 phase end 的直接返回。compete-first 的 task-cap 失败会让 `ready=false` 并短路真实 Claim，但仍正常构造 flags 和关闭 phase；后续 ticket、完整 Submit 和 golden 门禁负责拒绝无效执行。旧 API 若在 Materialize/PrepareMap 阶段已经失败，则不会进入 Claim，同时也无法闭合预期 Submit 窗口。Claim helper 内部对非目标角色或无效输入返回 false，同样会回到外层统一 end。因此 phase `0x3f` 证明边界、次数、数值顺序和时间闭合，不单独证明每次都发射了 Claim atomic、也不证明 winner 协议正确；后两者仍由业务 golden 和独立泳道/atomic 证据链负责。

正常 PA 无论本核角色是否参与该 task 的 atomic 竞争，每个 Submit 都进入一次外层 Claim 边界，所以固定 shape 不依赖 winner 或 `claim_attempted`：B1 必须为 96 核每核 5 次 begin/end、全局各 480 次；Case1 必须为每核 1,280 次、全局各 122,880 次。该规则继续直接复用 `expected_submit_count`，没有为 Claim 增加动态次数字段。

设备 ABI 也没有扩容：`submit-pmu-none` 仍为 128 B header 加 `96 × 64 B` whole record，共 6,272 B；Claim 复用同一 `96 × 64 B` phase sidecar，总计 12,416 B。没有逐 Claim record、task kind、winner 或 atomic 新字段。最终 Claim CCEC image 的 `.text` 为 154,192 B，其中 `dist_engine_aic.o/.text` 与 `dist_engine_aiv.o/.text` 分别为 54,960/55,096 B；ELF 正向包含 `dist_submit_pmu_expect_submits`、`fdwic_submit_pmu_read_counters` 和每角色一份 `fdwic_submit_pmu_phase_read_shadow_counters`，并拒绝 perf-clock、普通泳道/atomic 慢体和通用 PMU ring 符号。该尺寸只描述当前 Claim 诊断 ELF，不能与其他 profile 尺寸或时间机械相减。

两轮 B1 先验证四条边界、固定 shape 和数值门禁：

| 产物 | 全局 Submit | ALL 每次 elapsed | ALL 每次 request | ALL 每次 miss | 同 ELF 时间/request/miss 占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `..._031756` | 82.413 us | 1,070.163 ns | 118.140 | 5.958 | 23.737% / 16.891% / 14.595% |
| `..._031954` | 264.184 us | 1,065.442 ns | 96.167 | 5.660 | 22.086% / 15.785% / 16.041% |

两轮均为 32 AIC + 64 AIV、480/480 begin/end、phase status `0x3f`、96/96 primary/shadow 精确相等，最大单段 shadow request/miss 分别为 3,250/114 和 3,206/106。局部 per-call 时间相近，而跨核全局 Submit 分别为 82.413 us 和 264.184 us；这再次说明 B1 只用于结构、次数、冷启动和构建隔离，绝不能把其绝对时间或不同 ELF 的先后当作性能结论。

两轮完整 Case1 产物为：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_032101/
outputs/TestPagedAttentionUnroll_Case1_20260721_032244/
```

两轮均为 96 核、每核 1,280 次 begin/end、phase status `0x3f`，owner 配置/恢复、32 个 mixed triplet、固定 shape、数值顺序、phase 时间与风险阈值全部闭合；primary/shadow 96/96 精确相等，capture gap 为 0。第一轮 raw/HTML 为 72,857/80,025 B，第二轮为 72,854/80,025 B。逐角色每次调用的原始 observed 为：

| 产物 | 全局 Submit | 角色 | 每次 elapsed | 每次 request | 每次 miss |
| --- | ---: | --- | ---: | ---: | ---: |
| `..._032101` | 4,994.863 us | ALL | 641.816 ns | 80.219 | 1.957 |
| 同上 | 同上 | AIC | 311.706 ns | 79.437 | 0.025 |
| 同上 | 同上 | AIV | 806.871 ns | 80.609 | 2.922 |
| `..._032244` | 4,704.936 us | ALL | 646.708 ns | 80.202 | 1.951 |
| 同上 | 同上 | AIC | 313.109 ns | 79.465 | 0.026 |
| 同上 | 同上 | AIV | 813.507 ns | 80.571 | 2.914 |

阶段占比仍只在同一 ELF、同一轮、同一角色内计算：phase elapsed core-time 除以逐核首末 Submit elapsed core-time，phase request/miss observed 分别除以本轮整窗 primary。两轮结果为：

| 产物 | 角色 | 时间占比 | request observed 占比 | miss observed 占比 |
| --- | --- | ---: | ---: | ---: |
| `..._032101` | ALL | 17.942% | 14.188% | 16.870% |
| 同上 | AIC | 8.862% | 13.636% | 4.324% |
| 同上 | AIV | 22.368% | 14.476% | 17.086% |
| `..._032244` | ALL | 18.493% | 14.182% | 16.917% |
| 同上 | AIC | 9.213% | 13.627% | 4.436% |
| 同上 | AIV | 22.940% | 14.473% | 17.133% |

两轮逐角色 per-call 和同 ELF 占比方向稳定：AIV Claim 时间约为 AIC 的 2.6 倍，但当前证据不能把全部差异归给 atomic、I-cache 或某一条角色分支。empty-bracket Case1 测得的 ALL 记录开销量级约为 639～640 ns、49.34 request、1.34～1.36 miss/对；Claim 则约为 642～647 ns、80.2 request、1.95～1.96 miss/次。empty elapsed 是外层 tick 包住完整 observer 对，Claim elapsed 是两侧 observer 内部区间；两者又来自不同 ELF、布局和缓存状态，所以数值接近不代表 Claim 业务耗时接近零，request/miss 之差也不能当作“净 Claim 事件”。empty 仍只能作为观察器自扰动的经验尺度，不能相减。

Claim 内已有一条独立的 atomic 证据。最新 `_234305` 泳道中，73,728 条 `ClaimMax` FetchMax `return_ready` bracket 全部嵌套在 122,880 条 Claim 内，aggregate 为 40,222,098 core-ticks，占该泳道 ELF Claim 总量的 50.612%。其返回值被 `N > old` 真实消费，因此该边界表示返回依赖就绪，不是 source-issue；但它仍不是全系统可见性屏障。Claim submit-PMU ELF 编译掉的是 atomic 观察与落盘代码，不是实际 FetchMax，因此 phase elapsed 已经包含真实 atomic 路径及其等待，不能再把 40,222,098 ticks 加到 phase 时间，也不能跨 ELF 扣除它来制造“非 atomic Claim”。这份 overlay 只能解释为何 Claim 值得继续观察，不能单独解释 AIC/AIV 差异或直接推出可获得的优化收益。

Claim 代码落定后串行回归了六类互斥 B1 构建：

| 构建 | 产物 | 结果 |
| --- | --- | --- |
| `submit-pmu-materialize` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_032445/` | 96×5、status `0x3f`、primary/shadow 96/96 精确相等；235.643 us |
| `submit-pmu-arg-build` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_032536/` | 96×5、status `0x3f`、primary/shadow 96/96 精确相等；215.137 us |
| `submit-pmu-empty-bracket` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_032624/` | 96×5、status `0x3f`、primary/shadow 96/96 精确相等；276.578 us |
| `submit-pmu-none` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_032712/` | 无 phase 字段、primary/shadow 96/96 精确相等；76.509 us |
| `perf-clock` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_032802/` | 96×5 Submit、ELF 身份和调用 shape 闭合；74.512 us |
| 普通 level-4 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_032914/` | 4,552 条事件、87.664 us、`dropped=0`、排他整数闭合 PASS |

六轮只证明 Claim 四边界没有破坏已有 phase、none、perf-clock 与合并泳道构建；B1 的 74～277 us 绝对时间明显波动，且冷启动与跨核到达均未被单独控制，所以只作结构回归，不对差值归因，也不参与候选保留/撤销或观察代价计算。

#### 8.3.6 第四个真实业务 selector：`submit-pmu-register`

**[观察工具，已完成两轮 B1、两轮 Case1 与互斥构建回归]**

最新权威 Case1 泳道 `outputs/TestPagedAttentionUnroll_Case1_20260720_234305/` 中，Register 为 47,991,560 aggregate core-ticks，占 SubmitUnion 的 12.010%，固定 122,880 次。它是 Claim 之后仍超过约 10%、shape 固定且能映射到连续真实调用体的下一区域，因而进入单阶段 PMU；不是按 standalone 的旧 phase 清单机械补齐。

编译期固定：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=5
PTO_FDWIC_TRACE_ENABLED=0
```

公共协议 mode 为 `6`，phase id 为 `5`，raw name/boundary 为 `register/register_outputs_call_entry_to_return`，counter/time semantics 仍为 `running_read_clear_observed_bracket` 和 `inner_sys_cnt_between_boundary_observers`。设备继续复用 12,416 B phase ABI，没有增加逐调用、task-kind 或 insert 数字段。

当前最终 AICore 缓存为 `aicore-extra/32c26e06ad76d186`：

| 文件 | SHA256 | `.text` |
| --- | --- | ---: |
| `aicore_kernel.o` | `8264a6afd39815c825e0630dcbb69d9a3492d2e26989a61136f06d5e371fb750` | 150,608 B |
| `aicore_aic_combined.o` | `fd542b9ffdb33ea480ac91527c3e361cf44ae27604d2fd9ec90fc66e1ef53306` | 68,352 B |
| `aicore_aiv_combined.o` | `9a1eeca6fc06f370c02ff3acc8d1546820a910e7272b866f3b743b91be6ce9bb` | 82,256 B |

AIC/AIV `.text` 之和与最终 ELF 的 150,608 B 精确相等。最终 ELF 含 `dist_submit_pmu_expect_submits`、整窗 counter reader 和 AIC/AIV phase reader；perf-clock、普通泳道、逐 atomic 与通用逐 task PMU 符号均不存在。现有 raw 未内嵌 ELF SHA，上表只记录最终缓存身份，不把前几轮产物包装成逐字节 ELF 存档。

真实边界没有复制一条新的 Register 近似实现，而是紧贴现有三个 `dist_submit_register_outputs()` 调用点：

1. 统一 `dist_submit_finish_kernel_tail()`，覆盖旧 Kernel 与 compete-first Kernel Finish，位于可选 Fanin 后，传 `include_existing=true`；
2. 旧 `dist_alloc_tensors()`，传 `include_existing=false`；
3. compete-first `dist_alloc_compete_first_finish()`，同样传 `include_existing=false`。

begin 在调用入口，end 在返回后、`TRACE_TIMESTAMP(register_end)` 前；因此刻意排除前一条 record 发布、Register 结束时间戳和 caller 衔接。它是普通泳道 Register 的核心调用体，不是 timestamp-to-timestamp span 的逐 tick 复制。三条路径在正常成功 Submit 中均恰好一次；固定 shape 和 phase status 会 fail-closed 拒绝提前返回造成的缺失边界。

该调用体存在重要业务混合：Kernel 的 `include_existing=true` 会按 `ctx.register_mask` 扫描并插入 existing tensor，Alloc 的 `false` 路径在 helper 入口直接返回；`register_mask=0` 的 Kernel 也可能接近空调用。当前 PA 每 batch 为 1 Alloc + 4 Kernel，但为控制观察扰动没有增加 task-kind 或逐 insert raw 字段。因此结果只能解释为 RegisterOutputs 调用体聚合，不能命名为单次 TensorMap insert 净成本。

两轮 B1 先闭合模式、三挂点、固定次数与数据门禁：

| 产物 | 全局 Submit | ALL elapsed/call | ALL request/call | ALL miss/call | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| `..._034517` | 80.904 us | 231.281 ns | 140.208 | 4.446 | 96×5、480/480、`0x3f`、primary=shadow |
| `..._034904` | 267.167 us | 236.838 ns | 123.869 | 3.956 | 96×5、480/480、`0x3f`、primary=shadow |

B1 全局时间再次明显变化，而局部 per-call 量级相近；本阶段没有分别控制冷暖态、跨核到达或非 scalar-busy 等待，因此不把差值归给其中任何一项。这两轮只作结构证据，不用于 Register 稳态判断。

两轮完整 Case1 为：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_035025/
outputs/TestPagedAttentionUnroll_Case1_20260721_035136/
```

两轮都通过 96 核、每核 1,280 次、122,880/122,880 begin/end、phase status `0x3f`、owner Restore、32 个 mixed triplet、primary/shadow 精确相等、数值顺序和风险阈值门禁。逐角色原始 observed 为：

| 产物 | 全局 Submit | 角色 | elapsed/call | request/call | miss/call |
| --- | ---: | --- | ---: | ---: | ---: |
| `..._035025` | 4,688.752 us | ALL | 187.688 ns | 86.699 | 1.352 |
| 同上 | 同上 | AIC | 187.833 ns | 88.301 | 0.045 |
| 同上 | 同上 | AIV | 187.615 ns | 85.898 | 2.005 |
| `..._035136` | 5,136.513 us | ALL | 187.879 ns | 86.517 | 1.392 |
| 同上 | 同上 | AIC | 188.787 ns | 88.166 | 0.050 |
| 同上 | 同上 | AIV | 187.425 ns | 85.692 | 2.063 |

阶段与完整 Submit 的比率仍只在同一 raw、同一角色内计算：

| 产物 | 角色 | 时间占比 | request observed 占比 | miss observed 占比 |
| --- | --- | ---: | ---: | ---: |
| `..._035025` | ALL | 5.249% | 14.889% | 7.132% |
| 同上 | AIC | 5.390% | 14.902% | 9.492% |
| 同上 | AIV | 5.181% | 14.882% | 7.112% |
| `..._035136` | ALL | 5.055% | 14.884% | 7.335% |
| 同上 | AIC | 5.120% | 14.844% | 10.510% |
| 同上 | AIV | 5.023% | 14.905% | 7.308% |

两轮约 188 ns/call、86.6 request/call 的方向稳定，AIV miss 约 2.0/call、AIC 约 0.05/call。但这不支持把普通泳道 Register 的 12.010% 全部归给该调用体：泳道边界更宽且属于另一 ELF。empty-bracket 的约 639～640 ns/对也只能说明观察器自扰动不可忽略；它的外层时间口径与 Register 的内层时间不同，更不能跨 ELF 相减。当前 phase 没有局部 scalar busy，`miss × 90 ns` 也只作单核串行量级感知。

代码落定后串行回归了七类互斥 B1 构建：

| 构建 | 产物 | 结果 |
| --- | --- | --- |
| `submit-pmu-claim` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_035425/` | 96×5、480/480、`0x3f`、primary=shadow；254.094 us |
| `submit-pmu-materialize` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_035505/` | 96×5、480/480、`0x3f`、primary=shadow；84.021 us |
| `submit-pmu-arg-build` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_035545/` | 96×5、480/480、`0x3f`、primary=shadow；73.728 us |
| `submit-pmu-empty-bracket` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_035625/` | 96×5、480/480、`0x3f`、primary=shadow；77.925 us |
| `submit-pmu-none` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_035703/` | 无 phase 字段、primary=shadow；259.429 us |
| `perf-clock` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_035744/` | 96×5 Submit、ELF 身份/shape 闭合；74.282 us |
| 普通 level-4 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_035843/` | 4,550 条事件、292.159 us、`dropped=0`、排他整数闭合 PASS |

七轮只证明新增 mode、三个挂点和报告链没有破坏旧 phase、整窗、权威基线与合并泳道构建。B1 绝对时间不参与跨 ELF 比较。Python 的 profile/report/cache/ELF 门禁共 121 项通过。

#### 8.3.7 第五个真实 selector：`submit-pmu-submit-transition`

**[观察工具，已完成两轮 B1、两轮 Case1 与互斥构建回归]**

最新权威 Case1 泳道 `outputs/TestPagedAttentionUnroll_Case1_20260720_234305/` 中，`BetweenSubmitResidual` 为 72,362,428 aggregate core-ticks，占 SubmitEnvelope 的 **15.332%**。每核 1,280 个 Submit 形成固定 1,279 个相邻间隙，全局共 122,784 个；该 shape 与源码边界都可证明，因此它成为 Register 后唯一继续实现的 selector。该阶段不是新的业务分类，只聚合观察“上一次 Submit 结束到下一次 `dist_submit_begin()` 完成”之间的返回、编排衔接和下一任务准备。

编译期固定：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=6
PTO_FDWIC_TRACE_ENABLED=0
```

公共协议 mode 为 `7`，phase id 为 `6`，raw name/boundary 为 `submit-transition/previous_submit_end_to_next_submit_begin`。设备没有新增业务挂点，而是在已经统一存在的 `fdwic_submit_pmu_submit_begin/end()` 内建立状态机：非末次 Submit 的 end 打开区间，下一次非首 Submit 在 `dist_submit_begin()` 后进入 begin hook 时关闭；末次 end 只停止完整 PMU 窗口，不制造没有后继的区间。由此 B1 必须为每核 `5 - 1 = 4` 次，Case1 必须为每核 `1280 - 1 = 1279` 次。

`fdwic_submit_pmu_expected_phase_calls()` 集中表达该差异：None 为 0，Transition 为 `N - 1`，其他 phase 仍为 N。设备 ShapeValid、C++ host 校验/raw 导出和 Python report 各自使用或复算该规则；`N <= 1` 的 Transition capture 会被拒绝。两个 64 B 逐核结构和 12,416 B 总设备 ABI 均未增加字段，也没有记录间隙类型或逐间隙事件。因此 raw 是三类相邻任务组合的加权聚合，不能从中单独还原 kernel→kernel、kernel→alloc 或 alloc→kernel。

最终 AICore 缓存为 `aicore-extra/d17b87d79477c0e1`：

| 文件 | SHA256 | ELF `.text` section |
| --- | --- | ---: |
| `aicore_kernel.o` | `dff1ba7ebc40be24afd48ee478e69b82c7877338beda66db1c915203f27e7099` | 153,424 B |
| `aicore_aic_combined.o` | `84e63975e5593f2e644c4582cfe0954c34916b503f08a125bcb8e76050c09bae` | 69,608 B |
| `aicore_aiv_combined.o` | `5d284cb7db1cf8d4fed7306c88e430f924246a68b738536f217fba47a87d8bdd` | 83,792 B |

缓存的 CCEC 命令行包含 phase 6 和 trace-off 定义；对象含整窗 reader、phase shadow reader 与 `dist_submit_pmu_expect_submits`，不含普通泳道、atomic 观察或 perf-clock 符号。`aicore_kernel.o` 的 GNU `size` text 类合计为 153,864 B，其中还包含 440 B 只读数据；上表明确记录 ELF `.text` section，避免把两种口径混写。

两轮 B1 先验证 N-1 状态机和三层门禁：

| 产物 | 全局 Submit | ALL elapsed/间隙 | ALL request/间隙 | ALL miss/间隙 | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| `..._042627` | 265.223 us | 778.982 ns | 162.406 | 10.604 | 96×4、384/384、`0x3f`、primary=shadow |
| `..._042750` | 78.873 us | 802.044 ns | 155.164 | 9.078 | 96×4、384/384、`0x3f`、primary=shadow |

B1 全局时间仍有明显冷启动/到达波动，只作结构证据。两轮完整 Case1 为：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_042914/
outputs/TestPagedAttentionUnroll_Case1_20260721_043036/
```

两轮都通过 96 核、每核 1,280 Submit/1,279 间隙、122,784/122,784 begin/end、全窗 status `0x7ff`、phase status `0x3f`、owner Restore、primary/shadow 精确相等、数值顺序和风险阈值门禁。逐角色原始 observed 为：

| 产物 | 全局 Submit | 角色 | elapsed/间隙 | request/间隙 | miss/间隙 | 同 ELF 时间占比 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `..._042914` | 4,708.545 us | ALL | 354.560 ns | 134.101 | 4.815 | 10.046% |
| 同上 | 同上 | AIC | 303.374 ns | 133.426 | 0.503 | 8.881% |
| 同上 | 同上 | AIV | 380.153 ns | 134.438 | 6.972 | 10.601% |
| `..._043036` | 4,649.434 us | ALL | 350.516 ns | 134.145 | 4.459 | 9.933% |
| 同上 | 同上 | AIC | 303.185 ns | 134.200 | 0.494 | 8.843% |
| 同上 | 同上 | AIV | 374.181 ns | 134.118 | 6.441 | 10.456% |

两轮时间与 request 方向稳定，且 AIV 间隙时间、miss 都高于 AIC；这只能描述当前 Transition 诊断 ELF 内的聚合现象。它与泳道 `BetweenSubmitResidual` 使用同源源码边界，但泳道还发布 record，PMU 构建则在边界 observer 内部取时且改变代码布局，所以 350～355 ns/间隙不能与泳道约 589 ns/间隙逐 tick 对齐。empty-bracket 的约 640 ns/对是另一 ELF 测得的记录开销量级，也不能扣除。当前阶段不声称得到 “无观察净间隙成本”，更不把 `miss × 90 ns` 当作墙钟损失。

代码落定后串行回归四类互斥 B1 构建：

| 构建 | 产物 | 结果 |
| --- | --- | --- |
| `submit-pmu-register` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_043202/` | 96×5、480/480、`0x3f`、primary=shadow；80.010 us |
| `submit-pmu-none` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_043241/` | 无 phase 字段、primary=shadow；250.164 us |
| `perf-clock` | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_043322/` | 96×5 Submit、ELF 身份/shape 闭合；312.747 us |
| 普通 level-4 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_043553/` | 4,548 条 FDWIC record、887 条 atomic record、`dropped=0`、排他整数闭合 PASS；90.406 us |

Register 代表原有 N 次 phase，none、perf-clock 和普通泳道分别覆盖另外三条互斥构建；配合 Python profile/report/cache/ELF 门禁单测，证明共享 N-1 helper 没有把旧 phase 改成错误 shape，也没有污染其他构建。B1 绝对时间仍不参与跨 ELF 比较。

#### 8.3.8 selector 停止线与三条证据链分工

`arg-build`、`materialize`、`claim`、`register` 和 `submit-transition` 已经实现；前四项覆盖 Submit 内明确业务区域，Transition 单独覆盖相邻 Submit 间隙，empty-bracket 只提供 running 观察器经验尺度，不计入业务覆盖。完成 Transition 后重新按占比、调用 shape、边界连续性和约 640 ns/对的 empty 经验尺度筛选，没有理由继续把所有短 span 批量做成 profile：

- EfDrain 总量为 63,350,314 core-ticks，但其中 32,102,416 ticks 是 Kernel overlay；剩余 control 约 254 ns/Submit，又混有 atomic，当前局部 PMU 指标不能把三者可靠拆开；
- PrepareMap 为 22,673,322 ticks、约 184.5 ns/次，SubmitFinalize 为 19,944,392 ticks、约 162.3 ns/次且有四类尾边界；二者都短于 empty 经验尺度；
- Fanin、WinnerBuild、AllocComplete 和真实 loser 尾动作占比更小或 shape 非统一，继续增加 profile 更可能放大观察器影响，而不是提高归因能力。

`submit-transition` 已按 `expected_submits - 1` 的独立 shape 契约闭合，不再是待决候选。至此停止扩张 phase，转入三类构建观察代价、独占设备波动和 perf-clock 候选优化。后续单阶段 PMU 仍只独立编译、独立运行、独立发布，只报告本 raw 内的原始 observed 和占比，不与 empty 或任何其他 selector 相减。standalone 历史实现只提供 owner、门禁和报告加工方法参考，不是当前真实 PA 的前置任务，也不能替代真实结果。

三条证据链最终分工为：

| 构建 | 回答的问题 | 不能回答的问题 |
| --- | --- | --- |
| `perf-clock` | 候选是否真正缩短完整 Submit 墙钟 | 具体业务区域和 PMU 原因 |
| `swimlane` | 业务区域、atomic 次数、时序与闭合 | 无观察净性能、I-cache miss |
| `submit-pmu-none` / 单阶段 PMU | 完整窗口 AIC/AIV 每核 total/scalar/primary，以及一个真实 span 的同 ELF 时间和 request/miss observed | 跨 ELF 净阶段成本、局部 scalar busy、最终墙钟收益 |

第 8.4 节按这一分工交错比较 perf-clock 与两类诊断构建的完整时间；只有构建级偏移超过同构建波动时，才允许讨论保留观察能力的整体影响。

### 8.4 三类观察构建的交错量化

**[观察校准，已完成；当前波动下构建级差异不可可靠分辨]**

#### 8.4.1 固定对象与比较规则

本阶段冻结真实 PA 源码于 `d1572c33c942018005fa4c1f631569915f2a7e26`，只比较以下三种独立构建的完整 Submit 时间：

| 代号 | 构建 | 唯一用途 |
| --- | --- | --- |
| P | `--fdwic-profile perf-clock` | 后续业务候选的权威低扰动时间基线 |
| S | `--enable-l2-swimlane 4` | 普通业务 span 与 atomic 合并泳道 |
| N | `--fdwic-profile submit-pmu-none` | 完整 Submit 的 AIC/AIV PMU 与 I-cache |

三者均为独立 pytest 进程，固定 Case1、`--rounds 1 --skip-golden`。P 从 `fdwic_perf_clock_summary.json` 取整数 `global_submit_span_ticks`；S 从严格排他分析取 `global_submit_makespan.duration_cycles`；N 先经生产报告器完整校验，再取 `window.global_submit_span_ticks`。三个字段都表示跨核最早首 Submit 到最晚末 Submit，按 1 ns/tick 解释；1.65 GHz 只用于 PMU cycle，不参与这次全局时间换算。

三份最终 AICore 身份为：

| 构建 | cache key | `aicore_kernel.o` SHA256 | 文件大小 | ELF `.text` |
| --- | --- | --- | ---: | ---: |
| P | `138ce601ea506665` | `1fc81e8f3169c980de3d9103e0741d5ac8ecc92a59da83c0147d197658f3f0f3` | 2,468,576 B | 132,688 B |
| S | `faf05ee682e6a57d` | `16a2f2c4688f668c41d07a8e20262e482649c28e0daf7d5c0873cd622647e586` | 2,693,320 B | 203,344 B |
| N | `aa43623282e2a7db` | `53934476e4a12f72f74da5e3035c9f066924ffbf36f1515e0baffdadad6bc3b9` | 2,513,848 B | 138,064 B |

P 定义为 `PTO_FDWIC_PERF_CLOCK=1;PTO_FDWIC_TRACE_ENABLED=0`，N 定义为 `PTO_FDWIC_SUBMIT_PMU=1;PTO_FDWIC_TRACE_ENABLED=0`，S 不带私有 profile 定义。三者共用的 `libhost_runtime.so` SHA256 为 `5c2b6de4426e29a7aba6c65cbf83e9a3bebcc9fb19dd6bf126fa66f0aac7bd53`。测量期间 HEAD 和所有构建输入源码未变化；05:28 出现的另一个会话 standalone 说明文档工作树改动不参与编译，也不纳入本阶段提交。

这次比较只允许回答“整个观察构建相对 P 的时间分布是否发生可辨认偏移”。即使偏移稳定，也不能把 S-P 称为某条 record/atomic 的纯成本，不能把 N-P 称为 PMU start/stop 的纯成本，更不能用于 Claim、I-cache 或其他阶段的跨 ELF 扣减。

#### 8.4.2 B1 预检与扩样门禁

正式 Case1 前按 P→N→S 各运行一次 B1：

| 构建 | 产物 | 完整 Submit | 结构结果 |
| --- | --- | ---: | --- |
| P | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_044523/` | 77.478 us | 96 核、每核 5 Submit、身份闭合 |
| N | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_044616/` | 77.464 us | 96 核、primary=shadow、owner Restore、validation PASS |
| S | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_044712/` | 92.890 us | 4,548 条 FDWIC record、`dropped=0`、排他整数闭合 PASS |

B1 仍只作结构预检，不进入时间统计。正式首轮使用六个排列块：

```text
P→S→N, S→N→P, N→P→S, P→N→S, N→S→P, S→P→N
```

每种构建在第一、第二、第三位置各出现两次。预先约定的接受门禁为：六个配对差中至少 5/6 与中位数同方向，且 `abs(median delta) > 2 × MAD(delta)`；否则反序补一套完整六排列。首轮 P/S/N 的 MAD 分别为 602.632/12.021/153.896 us，S-P 与 N-P 都只有 3/6 同方向，因此按规则补跑：

```text
S→P→N, N→S→P, P→N→S, N→P→S, S→N→P, P→S→N
```

最终共 36 个独立 Case1 进程，每种构建 12 个样本。全部样本通过各自门禁：P 为 96 核且每核 1,280 Submit；S 均为 schema-v4、96×1,280、`dropped=0`、排他分区和整数闭合 PASS；N 均通过 profile、owner、96×1,280、primary/shadow、计数阈值与 producer/consumer 重算。没有静默丢弃或替换任何样本。

#### 8.4.3 12 个交错块的原始结果

以下所有数值均为完整 Submit 的 us；delta 只表示同一时间块中两个完整构建结果之差：

| 块 | 顺序 | P | S | N | S-P | N-P |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | P→S→N | 5703.409 | 5116.147 | 5137.214 | -587.262 | -566.195 |
| 2 | S→N→P | 5811.557 | 5113.538 | 5619.488 | -698.019 | -192.069 |
| 3 | N→P→S | 5386.290 | 5134.027 | 5015.682 | -252.263 | -370.608 |
| 4 | P→N→S | 4498.145 | 5103.117 | 5079.540 | +604.972 | +581.395 |
| 5 | N→S→P | 4465.933 | 5087.037 | 5740.450 | +621.104 | +1274.517 |
| 6 | S→P→N | 4784.124 | 5127.158 | 4893.280 | +343.034 | +109.156 |
| 7 | S→P→N | 5429.436 | 5092.613 | 5821.280 | -336.823 | +391.844 |
| 8 | N→S→P | 4470.463 | 5068.575 | 4467.564 | +598.112 | -2.899 |
| 9 | P→N→S | 6713.689 | 5068.551 | 5077.279 | -1645.138 | -1636.410 |
| 10 | N→P→S | 5486.364 | 5062.530 | 4651.644 | -423.834 | -834.720 |
| 11 | S→N→P | 5543.774 | 5069.357 | 4602.872 | -474.417 | -940.902 |
| 12 | P→S→N | 4563.104 | 5109.824 | 4727.538 | +546.720 | +164.434 |

按构建自身汇总：

| 构建 | n | 最小值 | 最大值 | 中位数 | MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| P | 12 | 4465.933 | 6713.689 | 5407.863 | 513.717 |
| S | 12 | 5062.530 | 5134.027 | 5097.865 | 23.395 |
| N | 12 | 4467.564 | 5821.280 | 5046.480 | 356.890 |

配对统计中的“中位相对差”先按每块计算 `100 × (X/P - 1)`，再对 12 个相对差取中位数；不是用表中两个独立中位数相除：

| 比较 | 范围 | 中位差 | MAD | 同方向 | 中位相对差 | 门禁 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S-P | -1645.138～+621.104 us | -294.543 us | 520.527 us | 7/12 | -5.444% | 不通过 |
| N-P | -1636.410～+1274.517 us | -97.484 us | 479.019 us | 7/12 | -1.685% | 不通过 |
| S-N | -728.667～+601.011 us | +70.961 us | 325.625 us | 7/12 | +1.412% | 仅辅助，同样不可分辨 |

扩样后的标准是至少 10/12 同方向且差值绝对值大于 `2×MAD`。三组都只有 7/12，S-P 的 294.543 us 也小于 1041.054 us，N-P 的 97.484 us 小于 958.038 us。因此不能声称 S 或 N 相对 P 更快/更慢，也不能从三组中挑选最接近预期的样本给出 “观察代码成本”。

#### 8.4.4 当前能够成立的结论

P 的首六/后六中位数为 5085.207/5457.900 us，N 为 5108.377/4689.591 us，方向相反；S 为 5114.842/5068.966 us，变化较小。按执行位置分组时，P 的第一/第二/第三位中位数为 5133.257/5407.863/5007.118 us，N 为 4833.663/5078.409/5015.247 us，S 为 5103.075/5098.431/5085.834 us。位置本身也没有给 P/N 提供统一单调解释。

唯一稳定事实是：当前 S 构建自身的 12 轮分布明显窄于 P/N。但 S 使用更大的 203,344 B `.text`、高频 record 和不同同步/轮询时序，这个现象可能是观察代码改变了系统的到达分布，也可能是 P/N 暴露了某个偶发慢尾；现有跨 ELF 数据无法区分原因，更不能反向证明“泳道没有代价”。

因此本阶段以**负结果**收口：在当前独占设备和运行方式下，三构建完整 Submit 的交错差值被 P/N 的同构建波动淹没，无法可靠量出观察构建整体影响。下一阶段不再增加跨 ELF 样本，而是固定同一 P ELF 连续运行足量样本，分别检查完整墙钟、逐核 body、首核到达和最慢核尾部，先定位波动来自哪里；在此之前不进入业务优化 A/B。

### 8.5 同一 `perf-clock` ELF 的独占设备波动定位

**[波动定位，已完成；主统计载体为多数核 Submit body 共同伸缩，统计残差 X 表现为迁移慢尾]**

#### 8.5.1 冻结对象与采样协议

第 8.4 节已经证明跨 ELF 差值会被同构建波动淹没。本阶段不再比较 P/S/N，而是只用同一份真实 PA `perf-clock` ELF 连续运行，回答波动首先落在以下哪种时间成分：

1. 96 核首个 Submit 的到达偏斜；
2. 多数核从首个到末个 Submit 的完整 body 共同伸缩；
3. 某个慢核在多数核 body 之外额外形成的尾部。

冻结身份保持为：

| 对象 | 固定值 |
| --- | --- |
| AICore cache | `138ce601ea506665` |
| `aicore_kernel.o` SHA256 | `1fc81e8f3169c980de3d9103e0741d5ac8ecc92a59da83c0147d197658f3f0f3` |
| `aicore_kernel.o` 大小 / `.text` | 2,468,576 B / 132,688 B |
| AICore 定义 | `PTO_FDWIC_PERF_CLOCK=1;PTO_FDWIC_TRACE_ENABLED=0` |
| `libhost_runtime.so` SHA256 | `5c2b6de4426e29a7aba6c65cbf83e9a3bebcc9fb19dd6bf126fa66f0aac7bd53` |

运行期间没有修改源码、触发重编译或穿插其他 profile。开始前先运行一次 B1，只作 96 核、调用 shape 和构建身份预检；再运行一次 Case1 预热并明确排除出统计；正式样本使用 20 个连续但彼此独立的 pytest 进程，每个进程固定：

```text
--case Case1 --fdwic-profile perf-clock --rounds 1 --skip-golden
```

没有使用单进程 `--rounds 20`，没有人工 sleep，也没有删除或替换快慢极值：

| 用途 | 产物 | 结果 |
| --- | --- | --- |
| B1 预检 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_054534/` | 96×5 Submit，74.552 us，结构和身份闭合 |
| Case1 预热 | `outputs/TestPagedAttentionUnroll_Case1_20260721_054947/` | pytest PASS；不进入统计 |
| 正式 20 轮 | `outputs/TestPagedAttentionUnroll_Case1_20260721_055106/` 至 `..._060447/` | 20/20 pytest PASS |

B1 的 74.552 us 几乎完全由 AIC core 23 的 74.484 us body 长尾决定；该核只比全局最早核晚到 0.068 us，因此 B1 也支持“长体不是起跑偏斜”的先验。但 B1 只有每核 5 次 Submit，不能与 Case1 时间混算，也不进入正式分布。

#### 8.5.2 raw 闭合与统计定义

20 份 `fdwic_perf_clock_summary.json` 全部满足：

- schema 为 `fdwic-perf-clock-v1`，参考时钟为 1 GHz；
- 96 个唯一 core，32 AIC + 64 AIV，core/block/lane 映射闭合；
- 每核 1,280 次 Submit，单轮共 122,880 次，20 轮合计 2,457,600 次；
- 每核 `elapsed = last_submit_end - first_submit_start`；
- 全局 `span = max(last_submit_end) - min(first_submit_start)`。

JSON 中 `global_submit_span_us` 和 group mean 是展示用舍入值；所有计算与门禁都使用整数 tick 原值，不对舍入后的浮点字段二次运算。这里的 1 GHz 是 sys-counter 参考时钟，不是约 1.65 GHz 的 PMU/core cycle 频率。raw 本身不内嵌 ELF SHA；通过采样前后的外部构建身份关联，另行确认 AICore ELF 和 host runtime SHA 完全一致。

对一轮中的核 `c` 定义：

```text
s[c] = first_submit_start
e[c] = last_submit_end
b[c] = e[c] - s[c]

S = min(s[c])
E = max(e[c])
G = E - S
```

对该轮最晚结束核 `z`，进一步作精确分解：

```text
A = s[z] - S                # 最晚结束核相对全局首核的晚到时间
M = median(b[c])            # 96 核完整 Submit body 的中位数
X = b[z] - M                # 最晚结束核相对中位数的额外慢尾
G = A + M + X               # 由整数 tick 派生并在每轮精确闭合
```

预先固定的“主要统计载体”门禁为：`abs(Spearman(G, component)) >= 0.7`，且该 component 的 `P90-P10` 不小于 `G` 的 `P90-P10` 的 50%。这只能判断波动落在哪个可观察时间成分，不能把统计关联直接命名为 atomic、I-cache、flag 等待或频率变化。

#### 8.5.3 20 轮原始分解

下表单位均为 us；`末核` 同时给出 core 类型、block 和 lane：

| 轮次 | 输出时间戳 | G | M | X | A | 末核 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `055106` | 5282.858 | 4639.647 | 634.045 | 9.166 | c28/AIC/b28/l0 |
| 2 | `055149` | 6066.913 | 5311.507 | 746.303 | 9.103 | c22/AIC/b22/l0 |
| 3 | `055232` | 4634.349 | 4397.435 | 228.024 | 8.891 | c15/AIC/b15/l0 |
| 4 | `055315` | 4898.490 | 4362.309 | 531.446 | 4.736 | c17/AIC/b17/l0 |
| 5 | `055357` | 5320.520 | 5049.764 | 268.651 | 2.105 | c16/AIC/b16/l0 |
| 6 | `055439` | 4633.401 | 4391.994 | 228.202 | 13.205 | c17/AIC/b17/l0 |
| 7 | `055522` | 4473.506 | 4377.564 | 94.329 | 1.614 | c1/AIC/b1/l0 |
| 8 | `055604` | 4509.646 | 4380.363 | 117.660 | 11.623 | c4/AIC/b4/l0 |
| 9 | `055647` | 5896.752 | 5595.158 | 299.450 | 2.144 | c77/AIV/b22/l2 |
| 10 | `055731` | 4897.879 | 4695.680 | 192.089 | 10.110 | c35/AIV/b1/l2 |
| 11 | `055813` | 5478.058 | 5172.396 | 302.890 | 2.771 | c25/AIC/b25/l0 |
| 12 | `055857` | 4653.463 | 4490.646 | 162.746 | 0.070 | c9/AIC/b9/l0 |
| 13 | `055941` | 4471.733 | 4375.400 | 88.891 | 7.442 | c77/AIV/b22/l2 |
| 14 | `060024` | 4557.543 | 4360.690 | 184.311 | 12.541 | c90/AIV/b29/l1 |
| 15 | `060110` | 5037.755 | 4675.526 | 360.102 | 2.127 | c30/AIC/b30/l0 |
| 16 | `060153` | 4764.823 | 4404.648 | 348.097 | 12.078 | c53/AIV/b10/l2 |
| 17 | `060238` | 5128.149 | 4862.399 | 265.724 | 0.026 | c40/AIV/b4/l1 |
| 18 | `060321` | 6169.966 | 5575.921 | 593.804 | 0.241 | c20/AIC/b20/l0 |
| 19 | `060404` | 4577.567 | 4381.504 | 188.969 | 7.094 | c70/AIV/b19/l1 |
| 20 | `060447` | 5860.558 | 5033.376 | 826.275 | 0.907 | c27/AIC/b27/l0 |

#### 8.5.4 分布、相关性与门禁结果

主要分量的完整分布如下。MAD 为未缩放的中位绝对偏差，P10/P90 使用线性插值：

| 指标 | 最小值 | 中位数 | 最大值 | MAD | P10 | P90 | P90-P10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G | 4471.733 | 4898.184 | 6169.966 | 362.658 | 4506.032 | 5913.768 | 1407.736 |
| 首核到达跨度 | 11.504 | 13.136 | 13.979 | 0.542 | 11.788 | 13.470 | 1.681 |
| M | 4360.690 | 4565.147 | 5595.158 | 188.665 | 4374.091 | 5337.948 | 963.858 |
| X | 88.891 | 267.188 | 826.275 | 87.895 | 115.327 | 645.271 | 529.944 |
| A | 0.026 | 5.915 | 13.205 | 4.002 | 0.224 | 12.124 | 11.900 |

与 G 的关联和主要载体门禁为：

| 序列 | Pearson | Spearman | 序列跨度 / G 跨度 | 判定 |
| --- | ---: | ---: | ---: | --- |
| M | +0.943 | +0.895 | 68.5% | PASS |
| X | +0.771 | +0.850 | 37.6% | FAIL：相关但跨度不足，只列为次要慢尾 |
| A | -0.436 | -0.395 | 0.8% | FAIL |
| 首核到达跨度 | +0.116 | +0.120 | 0.1% | FAIL |
| AIC body 中位数 | +0.943 | +0.896 | 89.8% | 辅助序列满足同阈值；不是 A/M/X 分解项 |
| AIV body 中位数 | +0.934 | +0.890 | 63.1% | 辅助序列满足同阈值；不是 A/M/X 分解项 |

以下交叉证据排除了“只有中位数偶然同向”的解释：

- 96 个核的 body 与 G 跨轮 Spearman 全部为正，95/96 不小于 0.7，中位数为 +0.862；相邻轮方向一致率逐核为 78.9%～94.7%，96/96 核均不低于 70%；
- body 最小值与 G 的 Spearman 为 +0.869，跨度达到 G 的 69.5%，说明连快核也随整轮共同伸缩；
- AIC 与 AIV body 中位数跨轮 Spearman 为 +0.973，不支持某类核心稳定单边拉长；AIC 中位数高于 AIV 只有 9/20 轮；
- 每轮核内“晚到程度与 body”的 Spearman 中位数仅 -0.042，范围 -0.258～+0.211，进入稍晚的核并不会系统性执行更久；
- 最晚结束核与最长 body 核 20/20 相同，但 20 轮分散到 18 个不同 core；只有 c17 和 c77 各重复两次，类型为 AIC 13 次、AIV 7 次，不支持固定物理坏核；
- 运行序号与 G 的 Spearman 仅 +0.006，没有随连续运行时间单调升高或降低的趋势。

#### 8.5.5 当前成立的原因层级与下一步

本阶段把“独占设备仍波动”收敛到两个层级：

1. **主要统计载体是多数核完整 Submit body 的共同伸缩。** 这不是某个首核晚启动，也不是一个固定物理核偶发卡住；AIC/AIV 和绝大多数单核 body 都随整轮同向变化。
2. **统计残差 X 表现为迁移慢尾。** X 与 G 高度相关，但 P90-P10 只有 G 的 37.6%，没有达到预设的主要载体门禁，因此只能解释次级差异，不能替代共同伸缩结论，也不能仅凭该残差声称存在一种独立物理机制。

“设备独占”只排除了其他用户任务抢占，不会自动固定设备频率、温度、cache、内部运行时状态或真实 Kernel/flag/atomic 的时序。当前 perf-clock raw 没有 PMU、业务 phase、Kernel 或 atomic 字段，所以还不能在以下原因之间作选择：

- on-core PMU cycle 数本身随轮次增减；
- 每纳秒有效 PMU cycle 比例发生变化；
- scalar issue、I-cache request/miss、atomic 或 completion flag 等待变化；
- 真实 Vector/Cube Kernel 及其内部 PIPE wait 变化。

下一阶段先不加设备探针，也不修改业务代码：固定同一份 `submit-pmu-none` ELF，连续运行独立 Case1，在每一份同 ELF raw 内联合分析 SYS Submit elapsed、PMU total、scalar busy、I-cache request/miss。只有它能复现 P 的波动形态，才允许用这些字段解释共同伸缩；若不能复现，就明确认定 PMU ELF 改变了现象，再转向同一 swimlane ELF 的快慢轮结构或最小容量的 per-core Kernel 聚合。

另行查阅本机 CANN 9.1 的已安装实现，`asys profiling -r power -p <秒> -d 0` 在 Ascend950 路径会启用 `msprof --sys-lp=on`，其 low-power 数据可包含 AIC 平均频率、软件 DVFS 下发频率和 EDP 降频计数。这不是上述 20 份 raw 的结论，而且 msprof 采样不能与权威 perf-clock 混跑或跨模式相减。只有同 ELF PMU 仍不能解释波动时，才单独建立这条环境取证链。

### 8.6 同一 `submit-pmu-none` ELF 的 cycle、Scalar 与 I-cache 联合关联分析

**[PMU 波动定位，已完成；N ELF 的共同 body 变化主要落在 Scalar-busy cycle，但 N 没有完整复现 P 的 A/M/X 形态]**

#### 8.6.1 复用样本、连续补样与构建身份

第 8.4 节已经留下 12 份同一 N ELF 的严格 Case1 raw。先复算这些存量后发现，PMU total 与 Scalar-busy 已有明显同向信号，但旧样本来自 P/S/N 交错排列，I-cache miss 又呈现与前序 ELF 关联的两个状态。为避免只凭受前序状态影响的 12 轮下负结论，本阶段复用存量并补充：

1. 一次 N-only Case1 预热，明确排除出统计；
2. 8 个连续、独立的 N-only Case1 进程，中间不插入 P、S 或其他 profile；
3. 旧 12 + 新 8 合计 20 轮统一使用当前生产 `load_capture()` 重算。

冻结构建身份为：

| 对象 | 固定值 |
| --- | --- |
| AICore cache | `aa43623282e2a7db` |
| `aicore_kernel.o` SHA256 | `53934476e4a12f72f74da5e3035c9f066924ffbf36f1515e0baffdadad6bc3b9` |
| `aicore_kernel.o` 大小 / `.text` | 2,513,848 B / 138,064 B |
| AICore 定义 | `PTO_FDWIC_SUBMIT_PMU=1;PTO_FDWIC_TRACE_ENABLED=0` |
| `libhost_runtime.so` SHA256 | `5c2b6de4426e29a7aba6c65cbf83e9a3bebcc9fb19dd6bf126fa66f0aac7bd53` |

采样索引为：

| 样本 | 目录 |
| --- | --- |
| 历史 12 轮 | `..._045212`、`..._045441`、`..._045608`、`..._050000`、`..._050228`、`..._050807`、`..._051547`、`..._051632`、`..._052029`、`..._052259`、`..._052753`、`..._053148` |
| 本次预热 | `outputs/TestPagedAttentionUnroll_Case1_20260721_062309/`，不计入统计 |
| 本次 8 轮 | `..._062430`、`..._062512`、`..._062557`、`..._062641`、`..._062725`、`..._062809`、`..._062852`、`..._062936` |

表中的省略前缀均为 `outputs/TestPagedAttentionUnroll_Case1_20260721`。20/20 raw 通过生产消费者的 schema、mode、owner、selector、topology、96×1280、SYS 起止、status、summary、primary/shadow 与 counter 阈值全量重算；合计 1,920 条 core record、2,457,600 次 Submit，primary-shadow request/miss 差值均为 0。最大可编程计数为 8,978,154，只占风险阈值 `0x3fffffff` 的 0.836%。新增 8 份 HTML 与当前生产 `render_report(raw)` 逐字节一致。采样结束后的 AICore 和 host SHA 保持不变；raw 本身仍不内嵌 cache key 或 ELF SHA，构建身份依赖上述外部关联。

#### 8.6.2 20 轮原始核心指标

下表单位为 us。`body/core` 是 96 核逐核 SYS elapsed 的 mean；`Scalar/core` 和 `残余/core` 分别是每核 PMU `scalar_busy` 与 `total-scalar_busy`，按生产报告 ALL 口径的本机长窗系数 1.649844 cycles/ns 换算。它们是 cycle-equivalent，不是 G 的可加分区。`miss/core` 是 96 核事件数 mean：

| 轮次 | 来源 | G | body/core | Scalar/core | 残余/core | miss/core |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `045212` | 历史 | 5137.214 | 4638.058 | 4197.932 | 439.480 | 9253.7 |
| `045441` | 历史 | 5619.488 | 4854.759 | 4399.790 | 454.297 | 9198.4 |
| `045608` | 历史 | 5015.682 | 4333.754 | 3882.439 | 450.733 | 11621.8 |
| `050000` | 历史 | 5079.540 | 4330.985 | 3878.340 | 452.012 | 11431.9 |
| `050228` | 历史 | 5740.450 | 5260.871 | 4808.963 | 451.293 | 9258.1 |
| `050807` | 历史 | 4893.280 | 4339.138 | 3891.437 | 447.037 | 11497.0 |
| `051547` | 历史 | 5821.280 | 5253.866 | 4784.488 | 468.736 | 11545.3 |
| `051632` | 历史 | 4467.564 | 4305.209 | 3859.732 | 444.808 | 11647.9 |
| `052029` | 历史 | 5077.279 | 4577.454 | 4124.678 | 452.125 | 11530.6 |
| `052259` | 历史 | 4651.644 | 4330.944 | 3898.974 | 431.301 | 9202.2 |
| `052753` | 历史 | 4602.872 | 4339.941 | 3908.777 | 430.510 | 9083.0 |
| `053148` | 历史 | 4727.538 | 4316.797 | 3884.156 | 432.015 | 9248.5 |
| `062430` | 连续 N | 5326.155 | 4898.879 | 4437.649 | 460.560 | 11674.2 |
| `062512` | 连续 N | 5081.403 | 4494.263 | 4039.732 | 453.884 | 11627.3 |
| `062557` | 连续 N | 4500.138 | 4345.932 | 3898.434 | 446.869 | 11639.1 |
| `062641` | 连续 N | 4808.405 | 4480.150 | 4027.859 | 451.670 | 11614.8 |
| `062725` | 连续 N | 4552.993 | 4312.336 | 3859.504 | 452.236 | 11633.6 |
| `062809` | 连续 N | 5613.177 | 4729.055 | 4270.670 | 457.756 | 11699.7 |
| `062852` | 连续 N | 5058.346 | 4336.133 | 3884.605 | 450.878 | 11640.8 |
| `062936` | 连续 N | 4495.435 | 4306.473 | 3858.078 | 447.766 | 11658.3 |

#### 8.6.3 N 与 P 的波动形态只部分一致

N 的 G 中位数为 5037.014 us，范围 4467.564～5821.280 us，MAD 为 347.423 us，P90-P10 为 1131.917 us。N 的起跑偏斜同样只有微秒量级，但其 A/M/X 分解与 P 不同：

| 构建 | A：rho / 幅度占 G | M：rho / 幅度占 G | X：rho / 幅度占 G | 严格门禁结果 |
| --- | --- | --- | --- | --- |
| P 20 轮 | -0.395 / 0.8% | +0.895 / 68.5% | +0.850 / 37.6% | M 为主，X 为辅 |
| N 20 轮 | +0.152 / 1.0% | +0.863 / 47.0% | +0.776 / 50.18% | X 通过；M 的幅度略低于门槛 |

这里比较的是各自同 ELF 内的分布形态，没有对 P/N 绝对时间作减法。新增连续 8 轮单列时仍是 M 占 43.8%、X 占 67.4%，说明 N 的慢尾增强不是旧 12 轮交错状态单独造成。N 与 P 只共同复现了“起跑不是主因、body 聚合量有同向变化、慢尾会迁移”等部分现象，**没有完整复现 P 的 M 主导、X 次要形态**。逐核 body 对 G 虽为 96/96 正相关，但只有 52/96 核的 Spearman 不低于 0.7，明显弱于 P 的 95/96；N 的最晚结束核与最长 body 核为 19/20 重合，且分散在 17 个不同 core。因此后面的 PMU 结果只能解释 N 自身，不能反向冒充 P 的完整根因。

#### 8.6.4 PMU total、Scalar-busy 与非 Scalar-busy 残余

在 N 自身内部，对每轮 96 核 mean/core 使用与 G 相同的相关性和幅度双门禁，结果为：

| 序列 | Spearman(G, ·) | P90-P10 | 占 G 幅度 | 判定 |
| --- | ---: | ---: | ---: | --- |
| SYS body/core | +0.853 | 622.628 us | 55.0% | PASS，辅助共同 body 证据 |
| PMU total/core 等效 | +0.853 | 622.560 us | 55.0% | PASS |
| Scalar-busy/core 等效 | +0.812 | 612.624 us | 54.1% | PASS |
| 非 Scalar-busy 残余/core 等效 | +0.657 | 26.093 us | 2.3% | FAIL |

Scalar-busy 与同轮 SYS body/core 的 Spearman 为 +0.962；留一法的 20 个子样本中，Scalar-busy 对 G 有 18/20 仍通过双门禁，对 SYS body/core 则 20/20 通过。AIC Scalar-busy 对 G 为 +0.791、幅度为 G 的 74.8%；AIV 为 +0.830，但幅度为 G 的 43.3%。若分别与同角色 SYS body 比较，AIC/AIV Scalar-busy 的 Spearman 为 +0.994/+0.899；它们的 P90-P10 也分别为 846.964/490.170 us，接近同角色 body 的 865.774/498.548 us。两组 P90-P10 不是同一轮差值，不能相除后宣称“解释比例”。两类 residual 的幅度只有 G 的 2.9%/3.1%。按同一双门禁，AIC Scalar-busy 通过，AIV 则因 43.3% 的幅度不足而未通过。

因此 N ELF 自身可以收敛到：**完整 body 波动表现为执行了更多 PMU total cycle，其中主要幅度落在 Scalar instruction busy cycle；非 Scalar-busy 残余不是主要幅度载体。** 该事件不是纯算术指令数：受控微基准显示依赖返回的 atomic 等待大部分会进入 Scalar-busy，而 I-cache refill 大部分不进入；当前 A5 又没有已经核验的 `scalar_wait_ib_time`，所以还不能继续把 Scalar-busy 拆成普通控制、polling、atomic 或 `wait_flag`。`total-scalar_busy` 仍只能称非 Scalar-busy 残余，不能命名为空闲、I-cache stall 或 Vector/Cube 等待。现有 raw 也没有读取独立的 Vector/Cube wait 计数，不能据此排除这类等待。

#### 8.6.5 有效 cycle/time 幅度不足以解释波动

每轮按 `sum(PMU total) / sum(SYS body ticks)` 计算的长窗有效换算比为：

| 分组 | 最小值 | 中位数 | 最大值 | P90-P10 相对幅度 |
| --- | ---: | ---: | ---: | ---: |
| ALL | 1.649587828 | 1.649607947 | 1.649650965 | 21.35 ppm |
| AIC | 1.649581477 | 1.649609592 | 1.649655438 | 27.49 ppm |
| AIV | 1.649590009 | 1.649608471 | 1.649648649 | 18.32 ppm |

ALL 比值的全范围只有约 0.00383%，P90-P10 只有约 0.00213%；即使按 5 ms 感知，后者也只有约 0.107 us 数量级，无法解释 1.132 ms 的 G 跨度。这里允许的结论是“本机长窗有效 PMU cycle/time 比例不是 N 波动的主要幅度来源”；total gate 位于 SYS 首尾之内、比例只是窗口平均和本机等效换算，不能把它包装成瞬时硬件核频或无条件的频率契约。

#### 8.6.6 I-cache miss 的前序状态特征与非归因结论

20 轮中，miss/core 与 G 的 Spearman 为 ALL -0.141、AIC -0.048、AIV 0.000，均不支持 I-cache miss 随慢轮增加。新增连续 N-only 8 轮的 G 仍横跨 4495.435～5613.177 us，而 AIC miss/core 仅为 855.094～1114.094，AIV 更窄至 16973.828～17049.344；尤其 AIV miss 基本固定时仍存在 1.118 ms 墙钟范围。

历史 12 轮还揭示了此前 I-cache 数据易飘的直接特征：

| 前序状态 | 样本数 | AIC miss/core 范围 | AIV miss/core 范围 | G 范围 |
| --- | ---: | ---: | ---: | ---: |
| S→N | 6 | 1289.500～1302.969 | 12973.047～13241.281 | 4602.872～5740.450 us |
| P→N 或该状态延续 | 6 | 433.500～885.031 | 16852.203～17079.109 | 4467.564～5821.280 us |

两组 miss 完全不重叠，G 却大量重叠；这只能证明前序 ELF 与后续 I-cache 状态高度关联，不能仅凭顺序杜撰具体驱逐机制。request/core 的方向也分裂：AIC 对 G 的 Spearman 为 -0.576，AIV 为 +0.818，ALL 只有 -0.110。request 是事件量，可能是更长 Scalar/poll 路径的伴随结果，不能直接相加成墙钟。

所以当前结论是：**I-cache miss 次数不是 N 这 20 轮共同伸缩的主要关联变量；但这不能排除 miss 延迟变化或其他缓存机制，也不表示绝对 miss 成本为零。** 已有 90 ns/miss 仍只是一阶单核串行标尺，不能乘 96 核总 miss 后冒充 Submit 墙钟损失。

#### 8.6.7 阶段决定

本阶段不继续堆 N 样本，也不立即做频率/温度采样：PMU total 与 Scalar-busy 通过 N 内统计载体门禁，而全窗平均 cycle/time 比例和 miss 次数没有表现为主要关联变量；这仍不能排除 miss 延迟或其他缓存机制。N 的全局慢尾形态又与权威 P 不完全相同，继续用 N 解释 P 会越过证据边界。

下一步回到最接近权威基线的低容量路径：在 `perf-clock` 构建中沿真实 `execute_slot()` 的现有 Kernel 首尾边界，只增加**每核 Kernel 累计 SYS tick 和调用次数**，不生成逐事件 record、不启用泳道或 PMU。先验证该独立变体是否复现 P 的多数核共同伸缩形态；只有复现后，才在同一变体内部判断变化主要落在真实 Vector/Cube Kernel（包含内部 PIPE wait）的墙钟，还是其余 Scalar scheduler。若 Kernel 聚合稳定而其余部分继续拉长，再进入最新真实业务 span 的 Scalar 候选分析。该变体必须独立命名和闭合，不能冒充原 P，也不能与 P 绝对时间相减。

### 8.7 真实 PA 的低容量 `perf-clock-kernel` 聚合变体

**[工具实现与 B1/Case1 结构闭合已完成；波动分布采样待下一阶段冻结提交后进行]**

#### 8.7.1 为什么必须是独立变体

权威 P 只在每核首个 Submit 起点和末个 Submit 终点各读取一次 `SYS_CNT`。第 8.5、8.6 节已经证明，不能靠跨 ELF 时间差把泳道或 PMU 中看到的现象硬扣回 P；但 P 自身又没有字段可以区分真实 Kernel 与其余调度路径。为此新增独立 profile：

```text
--fdwic-profile perf-clock-kernel
PTO_FDWIC_PERF_CLOCK=1
PTO_FDWIC_PERF_CLOCK_KERNEL=1
PTO_FDWIC_TRACE_ENABLED=0
```

本文简称它为 K。K 使用独立 cache key、最终 ELF marker、文件名和 schema，不能覆盖或冒充 P：

```text
fdwic_perf_clock_kernel_summary.json
schema = fdwic-perf-clock-kernel-v1
mode   = perf-clock-kernel
```

P 继续输出原来的 `fdwic_perf_clock_summary.json` / `fdwic-perf-clock-v1`。后续首先比较 P、K 各自同 ELF 内的波动形态；只有 K 能复现 P 的 A/M/X 结构，才允许在 K 内部用 Kernel 与剩余调度区间做二分，绝不以 K-P 绝对时间差声称观察成本或业务收益。

#### 8.7.2 Kernel 边界与固定 64 B 布局

K 沿现有泳道的真实 Kernel 边界，在 `execute_slot()` 中紧贴 `dist_aicore_call_slot_kernel(s)` 前后各读取一次 `SYS_CNT`。该区间不包含函数返回后的 `store_barrier()`、完成标志发布、frontier 推进和 Commit。它包含 linked kernel 函数内部真实执行及其显式 `wait_flag` / pipe 同步，但不能命名为“纯 Vector/Cube 指令时间”，也不能证明函数返回后不存在仍在途的异步写回。

计时还受逐核 perf-clock 窗口约束：只有 `first_submit_start != 0` 且 `last_submit_end == 0` 时才累计。因此，本核末个 Submit 返回后的 FinalDrain Kernel 被有意排除。这样每核才能保持精确整数关系：

```text
elapsed_ticks
= kernel_elapsed_ticks
 + non_kernel_residual_ticks
```

这里的 `non_kernel_residual` 只是上述同核窗口的算术剩余，包含调度、同步及观察边界等尚未细分的时间，不能直接重命名为 Scalar 或空闲。

没有新增 sidecar，也没有逐 Kernel record。`FdwicSwimlaneCoreState` 继续保持每核独占 64 B，整个固定 header 仍为 6,976 B：

- 原 tail 的 32 B 增加 K 专属 union 视图，保存首末 Submit、实际/期望 Submit 数和 64-bit Kernel 累计 tick；
- trace 关闭后原前 20 B 中的 `count` 保存 Kernel 调用数，`dropped` 保存聚合错误状态，`poll_batch_records` 保存 K mode=2，另外两个字段必须为 0；
- tick 逆序、64-bit 累加溢出或 32-bit 调用数溢出都会设置状态，host 拒绝发布 raw；
- P 仍要求这五个外层字段全部为 0，并要求原 mode=1/final_seen=1。P/K 互相误载时不能静默通过。

#### 8.7.3 Host 发布前的闭合门禁

K 复用原 perf-clock 的 header-only 分配和一次 64 B/core flush，不打开普通泳道、atomic 或 PMU。host 只有在以下条件全部满足后才以 `.tmp` 原子发布正式 JSON：

1. 固定 32 AIC + 64 AIV、96 个 core 的 block/lane 拓扑完全一致；
2. `records_per_core=0`、header=6,976 B、时钟为 1 ns/tick；
3. 每核实际 Submit 数等于 orchestration 声明值，B1 为 5、Case1 为 1,280，且 K 的每核首末 tick 必须形成严格正区间；
4. K mode=2、聚合状态为 0、保留字段为 0；
5. 每核 `kernel_ticks <= elapsed`，calls/ticks 同为 0 或同为非 0；
6. 全局起止严格等于 96 核首 tick 最小值和末 tick 最大值；
7. 当前 PA 固定每 batch 一次 Alloc 和四次 Kernel Submit；无 fanin 的 QK 最迟会在后继 SF Submit 的 EfDrain 执行，所以 AIC 调用数至少为 `batch`，AIC/AIV 分别不超过 `2*batch`，总数不超过 `4*batch`。

第 7 条的最大值不是等式：末个 Submit 后才执行的任务属于 FinalDrain，必须从 K 中排除；QK 下界则防止 hook 失效后全零报告仍被发布。输出保留逐核整数、AIC/AIV 的 min/max/sum/mean、调用数以及 `sum(kernel_ticks)/sum(elapsed_ticks)`。最后一个比例只能称“聚合 core-time 份额”，不能当作跨核墙钟占比，也不能用全局 Submit span 减去跨核 Kernel tick 求调度时间。顶层 `min_kernel_calls_in_window` / `max_kernel_calls_in_window` 表示按 batch 推导的 **全局总调用合法范围**，不是逐核极值；逐核实际极值只在 `groups.*.kernel_calls_min/max` 中表达。

#### 8.7.4 B1 边界取证与 P 回归

最终源码的 K B1 闭合件为：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260721_073602/
  fdwic_perf_clock_kernel_summary.json
```

golden、host 门禁和 Python 末端产物契约均 PASS。96 核每核 5 次 Submit，完整 span 为 84.816 us；窗口内只出现 1 个 Kernel 调用（合法范围 1～4），位于 AIC，累计 61.289 us。AIV 的窗口内 Kernel 调用为 0。这个结果不是“漏采”：它与最近泳道中 B1 只有一个 Kernel 落在 Submit、其余任务进入 FinalDrain 的结构一致，直接证明 K 的窗口过滤生效。

随后用原 P 做 B1 回归：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260721_073736/
  fdwic_perf_clock_summary.json
```

该轮同样 PASS，仍输出原 schema、原字段和原文件名，不含任何 Kernel 聚合字段。P 的最终 `.text` 仍为 132,688 B，K 的 `.text` 为 138,320 B；P 只含 `dist_perf_clock_expect_submits`，K 还必须含 `dist_perf_clock_kernel_profile_marker`，两者均不含泳道、atomic 或 Submit-PMU 观察器符号。P 的完整 ELF 因调试行号变化不能只凭 SHA 断言字节相同，所以这里只把 `.text` 大小、宏门禁和最终符号表作为“热路径未编入 K hook”的证据。

最终构建身份如下，后续 K 分布采样必须保持不变：

| 对象 | cache / 大小 / SHA256 |
| --- | --- |
| P AICore | `138ce601ea506665` / 2,468,896 B / `98f4e3978cd145477be1865b489f6475545ed1dce6dda9d145768894438b57d1` |
| K AICore | `e9cebfc34cbed0e7` / 2,495,544 B / `8d23407aa0062534812846b7da03f5078a4843a261f134812c95f5dbc5155060` |
| 共用 host SO | 11,644,136 B / `5f2e9af0892f64f28aded87e013a5b32425d86940b2e08cd34dad49cab0c0f9d` |

#### 8.7.5 Case1/B256 首轮结构闭合

最终源码的 K Case1 闭合件为：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_073849/
  fdwic_perf_clock_kernel_summary.json
```

golden PASS，96×1280 Submit 与所有整数聚合均闭合：

| 指标 | ALL | AIC | AIV |
| --- | ---: | ---: | ---: |
| 完整 Submit span | 5079.557 us（整数 tick） | - | - |
| 窗口内 Kernel calls / 合法范围 | 1013 / [256,1024] | 509 / [256,512] | 504 / [0,512] |
| Kernel core-time sum | 32.905240 ms | 18.740346 ms | 14.164894 ms |
| 非 Kernel residual core-time sum | 413.683894 ms | 132.130850 ms | 281.553044 ms |
| Kernel core-time 份额 | 7.36812% | 12.4214% | 4.7900% |
| calls/core 范围 | 3～21 | 12～21 | 3～14 |

1013 而非 1024 表示该轮还有 11 个真实 Kernel 在对应 worker 的末次 Submit 结束后进入 FinalDrain；不能把它们补进 K 来追求“任务总数好看”。96 核均至少执行一次窗口内 Kernel，逐核 `elapsed = kernel + residual`、分组 sum 与顶层 sum 已独立从 JSON 复算通过。

这一轮只证明数据结构、边界和数量级可用，不能凭单样本断言 7.37% 就是权威 P 的 Kernel 墙钟比例，更不能据此直接提出 Scalar 优化。下一阶段需在冻结 K 提交后排除一次预热并连续采集足量独立 Case1，先判断 K 是否复现 P 的 M 主导、X 次要形态；同时比较 Kernel ticks、calls 和 ticks/call，避免把“更多任务迁入 Submit 窗口”误判成“单次 Kernel 变慢”。

K 每个已计入的 Kernel 都会新增两次 `SYS_CNT` 读取；门控判断、计数累加等开销还会落入同一 K ELF 的 residual。它们不从单轮结果中机械扣除，K 只用于同 ELF 波动载体分析，不能与 P 的字段逐项相减。

#### 8.7.6 当前验证记录

- profile/cache/compile definitions、ELF 正反门禁及 P/K JSON 末端契约定向单测：65/65 PASS；
- K B1、P B1、K Case1：golden、host raw 门禁和 Python 产物契约全部 PASS；
- Case1 JSON 的 96 核、起止、calls、Kernel tick、residual 及 AIC/AIV sum 独立复算全部闭合；
- 一次 K B1 在只重编 CMake cache、尚未通过 RuntimeBuilder 同步 `build/lib` 时仍加载旧 host SO，旧 JSON 缺少新增的全局调用下界字段，Python 末端契约按预期拒绝；该轮 `..._073315` 明确不计入有效结果。随后确认 cache 与 `build/lib` 的 host SO SHA256 同为上表值后才取得三份最终闭合件；
- 一次顺带触发的全平台 `test_runtime_builder.py` 运行中，A5 无本阶段失败，但 5 个 A2A3/A2A3sim 集成构建被该分支既有的 payload ABI 静态断言挡住；该结果不包装成本阶段 PASS，也不扩展到与真实 A5 PA 无关的修复。

### 8.8 同一 `perf-clock-kernel` ELF 的 Kernel、调用迁移与 residual 联合分析

**[波动定位，已完成；K 复现 P 的多数核共同伸缩，主要变化落在同一 K ELF 的 non-Kernel residual，不是 Kernel 数量或 Kernel core-time]**

#### 8.8.1 冻结对象与采样协议

第 8.7 节提交 `a17c188a` 后没有修改源码、触发重编或穿插其他 profile。冻结身份为：

| 对象 | 固定值 |
| --- | --- |
| commit | `a17c188a` |
| K AICore cache | `e9cebfc34cbed0e7` |
| `aicore_kernel.o` SHA256 | `8d23407aa0062534812846b7da03f5078a4843a261f134812c95f5dbc5155060` |
| `aicore_kernel.o` 大小 / `.text` | 2,495,544 B / 138,320 B |
| AICore 定义 | `PTO_FDWIC_PERF_CLOCK=1;PTO_FDWIC_PERF_CLOCK_KERNEL=1;PTO_FDWIC_TRACE_ENABLED=0` |
| `libhost_runtime.so` SHA256 | `5f2e9af0892f64f28aded87e013a5b32425d86940b2e08cd34dad49cab0c0f9d` |

先运行一次与正式样本同参数的 Case1 预热：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_074831/
```

该轮 4,480.978 us，只用于加载和预热，明确排除出统计。随后连续运行 20 个独立 pytest 进程，每个进程固定：

```text
--case Case1 --fdwic-profile perf-clock-kernel --rounds 1 --skip-golden
```

没有单进程 `--rounds 20`，没有人工 sleep，没有删除快慢极值。20/20 进程 PASS，生产消费者逐份复验 1,920 条 core record、2,457,600 次 Submit；96×1280、拓扑、严格正时间窗、AIC/AIV/ALL 调用范围、逐核/分组/顶层 Kernel-residual 整数关系全部闭合。采样前后 K AICore 和 host SHA 保持不变。

#### 8.8.2 原始分解

继续严格复用第 8.5 节已经冻结的 `G=A+M+X` 定义和门禁。下表时间单位均为 us；`K mean` 与 `R mean` 分别是该轮 96 核 Kernel 和 non-Kernel residual 累计 tick 除以 96，只是逐核平均工作量，不是跨核墙钟分解：

| 轮次 | 时间戳 | G | M | X | A | calls | K mean | R mean |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `075049` | 5487.491 | 5296.392 | 179.078 | 12.021 | 999 | 342.298 | 4949.224 |
| 2 | `075146` | 4558.225 | 4395.182 | 162.861 | 0.182 | 1003 | 335.283 | 3993.544 |
| 3 | `075243` | 4801.766 | 4410.283 | 379.067 | 12.416 | 1010 | 337.500 | 4087.562 |
| 4 | `075340` | 4948.371 | 4428.318 | 516.989 | 3.065 | 1009 | 338.888 | 4144.158 |
| 5 | `075438` | 4675.254 | 4403.558 | 266.603 | 5.093 | 1008 | 339.586 | 4012.933 |
| 6 | `075537` | 4462.847 | 4374.762 | 82.975 | 5.110 | 1006 | 337.467 | 3974.188 |
| 7 | `075622` | 5850.474 | 5350.149 | 499.995 | 0.330 | 1006 | 342.335 | 4916.046 |
| 8 | `075719` | 6052.962 | 5200.194 | 839.308 | 13.461 | 1006 | 345.278 | 4878.580 |
| 9 | `075816` | 4545.193 | 4378.726 | 165.007 | 1.459 | 1009 | 337.258 | 4006.942 |
| 10 | `075921` | 4431.545 | 4374.420 | 49.245 | 7.880 | 1009 | 337.067 | 3958.067 |
| 11 | `080019` | 5107.881 | 5007.164 | 88.431 | 12.285 | 1002 | 338.038 | 4575.078 |
| 12 | `080115` | 5351.425 | 4498.411 | 839.580 | 13.434 | 1007 | 342.476 | 4353.041 |
| 13 | `080213` | 4680.613 | 4383.765 | 292.122 | 4.727 | 1011 | 338.973 | 4001.718 |
| 14 | `080309` | 5148.840 | 4856.157 | 284.580 | 8.103 | 1003 | 338.359 | 4510.296 |
| 15 | `080407` | 5723.402 | 5325.057 | 386.435 | 11.910 | 1009 | 344.642 | 4985.772 |
| 16 | `080503` | 5027.479 | 4736.360 | 289.392 | 1.726 | 998 | 340.002 | 4382.374 |
| 17 | `080602` | 6526.893 | 5702.921 | 817.173 | 6.799 | 999 | 352.976 | 5326.335 |
| 18 | `080700` | 5309.780 | 4573.770 | 728.598 | 7.412 | 1011 | 343.431 | 4270.854 |
| 19 | `080756` | 5837.718 | 5254.138 | 573.874 | 9.706 | 1001 | 351.478 | 4901.364 |
| 20 | `080901` | 4695.202 | 4449.372 | 233.099 | 12.732 | 1007 | 340.036 | 4025.199 |

#### 8.8.3 K 是否复现 P 的波动形态

P10/P90 继续采用线性插值，MAD 为未缩放中位绝对偏差：

| 指标 | 最小值 | 中位数 | 最大值 | MAD | P10 | P90 | P90-P10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G | 4431.545 | 5067.680 | 6526.893 | 406.119 | 4536.958 | 5870.723 | 1333.764 |
| 首核到达跨度 | 12.482 | 13.811 | 14.186 | 0.121 | 13.264 | 14.124 | 0.860 |
| M | 4374.420 | 4536.090 | 5702.921 | 161.499 | 4378.330 | 5327.566 | 949.236 |
| X | 49.245 | 290.757 | 839.580 | 165.111 | 87.886 | 819.386 | 731.501 |
| A | 0.182 | 7.646 | 13.461 | 4.478 | 1.346 | 12.802 | 11.456 |

按第 8.5 节预先固定的“双门禁”，K 与 P 的对比如下：

| 构建/分量 | Pearson | Spearman | P90-P10 占 G | 判定 |
| --- | ---: | ---: | ---: | --- |
| K / M | +0.917 | +0.946 | 71.2% | PASS：主要共同伸缩 |
| K / X | +0.742 | +0.755 | 54.8% | PASS：本组还有显著迁移慢尾 |
| K / A | +0.293 | +0.353 | 0.9% | FAIL |
| K / 首核到达跨度 | +0.061 | -0.021 | 0.1% | FAIL |
| P / M（第 8.5 节） | +0.943 | +0.895 | 68.5% | PASS |
| P / X（第 8.5 节） | +0.771 | +0.850 | 37.6% | 相关但幅度不足 |

K 只**部分复现** P：最关键的 M 共同伸缩和 A 失败结论一致，因此可以在同一 K ELF 内部二分 K 自身的波动；但 K 的 X 也跨过门禁，而 P 的 X 幅度不足，不能宣称两个 ELF 的完整 A/M/X 形态相同，更不能把 K 内部归因直接外推成 P 的根因。交叉证据为：

- 96/96 核的 body 与 G 跨轮 Spearman 不小于 0.7，中位数 +0.876；
- body 最小值与 G 的 Spearman 为 +0.922，幅度达到 G 的 79.9%；
- AIC/AIV body 中位数的跨轮 Spearman 为 +0.889；
- 每轮核内“晚到程度与 body”的 Spearman 中位数 -0.142，范围 -0.325～+0.194；
- 最晚结束核分散到 18 个不同 core，AIC/AIV 为 11/9；运行序号与 G 的 Spearman 只有 +0.304，不支持固定坏核或简单线性升温解释。

K 中最晚结束核与最长 body 核只有 17/20 相同，而 P 为 20/20，也再次说明 K 的尾部形态已经改变。M 与 X 的幅度比例分别以各自分布计算，不是同轮可加分量，不能把 71.2% 与 54.8% 相加成所谓“解释 126%”。

#### 8.8.4 同一 K ELF 内的 Kernel 与 residual

逐轮都有精确关系：

```text
body mean/core = Kernel mean/core + residual mean/core
```

但 mean/core 与 G 不是同一个跨核墙钟定义，下面只用相关性和同量纲幅度判断“波动落在哪里”，不把三者机械相减成墙钟贡献：

| K 内部序列 | 中位数 | P90-P10 | Spearman(G, ·) | 占 G 幅度 | 双门禁 |
| --- | ---: | ---: | ---: | ---: | --- |
| body mean/core | 4654.901 us | 968.300 us | +0.941 | 72.6% | PASS |
| non-Kernel residual mean/core | 4311.948 us | 961.270 us | +0.941 | 72.1% | PASS |
| Kernel mean/core | 339.794 us | 8.659 us | +0.872 | 0.65% | FAIL：相关但幅度极小 |
| pooled Kernel 平均每调用 | 32.402 us | 0.938 us | +0.892 | 0.07% | FAIL：不是墙钟分量 |
| Kernel calls | 998～1011 次 | - | -0.392 | - | FAIL：数量与慢轮不同向 |

角色拆分也一致：AIC/AIV residual mean/core 与 G 的 Spearman 为 +0.961/+0.937，幅度为 G 的 91.7%/66.1%，两者都通过门禁；AIC/AIV Kernel mean/core 的幅度只有 2.49%/0.34%，均不通过。逐核看，95/96 核的 residual 与 G 的 Spearman 不小于 0.7，而 96 个核没有一个 Kernel tick 序列达到 0.7。

总 calls 为 998～1011，与 G 的 Spearman 为 -0.392；但 AIC/AIV calls 与 G 分别为 +0.755/-0.754，方向相反。AIC pooled 单次均值约 36.30～39.28 us，AIV 约 27.54～28.42 us，而且 AIC 内部还混合 QK/PV。因此总 `ticks/call` 的同向相关至少部分受角色与任务构成变化污染，不能解释成“某个 Kernel 变慢”。

因此当前成立的是：**K 自身约 1.33 ms 的整轮波动主要落在 non-Kernel residual，不是窗口内 Kernel 总数量，也不由 linked-kernel mean/core 的变化幅度主导。** 这不等于 Kernel 对绝对性能没有影响，也不能反推 P 已经由 residual 主导。

`non_kernel_residual` 仍只是 K 窗口内的算术剩余，包含 Scalar 调度、atomic/flag 轮询与等待、Kernel 调用之间的协议路径以及 K 聚合观察开销；它不能直接重命名为 Scalar busy、atomic、I-cache miss 或频率问题。下一阶段先对 K residual 做最小侵入归因，并最终由原 P 的候选交错 A/B 决策；不能因为本轮在 K 中排除了 Kernel 变化幅度主导，就直接修改业务协议或把结论外推到 P。

### 8.9 独立设备状态与低功耗定性取证

**[取证已完成；排除约 10.24 ms 粒度的持续 DVFS/EDP 主导，不外推到更短瞬态]**

第 8.5～8.8 节已经把 P、N、K 各自的同 ELF 波动形态摸清，但仍不能回答设备内部频率、EDP 降频或外部任务是否参与了某一批次。这里增加一次**独立、限时、只作定性判断**的设备状态取证；它不成为第四条性能证据链，也不与任何权威 perf-clock、Submit-PMU 或泳道采样同场比较。`msprof` 运行期间取得的 PA 时间一律作废。

#### 8.9.1 正式接口与字段语义

本机 CANN 9.1 的正式入口为：

```bash
asys info -r status -d 0
asys health -d 0
asys profiling -r power -p <seconds> -d 0 --output <dir>
```

本机实现确认 `asys profiling -r power` 最终调用 `msprof --sys-lp=on`。低功耗原始数据落在 `device_0/sqlite/lowpower.db` 的 `LowPower` 表；字段含义由本机 `tools/profiler/profiler_tool/analysis/viewer/stars/low_power_viewer.py` 核验：

| 字段 | 正式含义 | 本阶段用途 |
| --- | --- | --- |
| `data7_hard` | PPU 上报的 AIC 平均频率 | 看持续频率阶跃或硬件/目标分离 |
| `data0_soft` | 软件 DVFS 下发的 AIC 频率 | 看目标频率 |
| `data2_soft` | EDP POWERBRAKE 计数 | 只看同一采样期首末增量 |
| `data3_soft`～`data5_soft` | EDP IWARNING2/1/0 计数 | 只看同一采样期首末增量 |

该表没有 Scalar 独立频率；公开 HAL 头文件也没有 `MODULE_TYPE_SCALAR`。因此不能把 AIC 平均频率称为 Scalar 独立时钟，也不能臆造一个 Scalar 动态频率接口。N 构建中 PMU cycle 与 sys-counter 的比例仍只表示该工作负载窗口的有效 cycle/time，不是独立硬件频率读数。

#### 8.9.2 状态快照与告警边界

2026-07-21 的只读状态快照显示：设备为 `Ascend 950PR_958b V100`，功耗约 290.3～290.4 W、温度 40 C、采样时 AI Core usage 为 0%；idle 瞬时 AIC 频率为 `100, 100 MHz`。设备 health 同时为 `Alarm`，`asys health -d 0` 报告一个：

```text
0x80b78000 | node type=SLLC | sensor type=RAS State | event state=module error
```

这只证明采样时存在该设备状态；当前没有证据证明 SLLC 告警导致 Submit 波动，也没有执行清告警、复位或任何配置修改。后续性能结论若仍在该状态下取得，必须保留这个环境事实，不能把它隐去或提前归因。

#### 8.9.3 两秒 idle probe 的原始结果

独立 idle 产物为：

```text
/tmp/asys_power_idle_20260721_0815/
  asys_profiling_result_20260721082558007/
    PROF_000001_20260721082558023_03217614RIFDHMDG/
      device_0/sqlite/lowpower.db
```

数据库共 776 行，die 0/1/2/3 各 194 行；每个时间点四个 die 齐全。相邻时间戳的 193 个差值全部严格为 **10.240002 ms**，覆盖 **1.976320386 s**。有效 D-die 0/1 完全一致：

| 指标 | die 0 | die 1 |
| --- | ---: | ---: |
| `data7_hard` | 首点 100 MHz，后续 193 点全为 1650 MHz | 同左 |
| `data0_soft` | 首点 100 MHz，后续 193 点全为 1650 MHz | 同左 |
| POWERBRAKE 首末增量 | 0 | 0 |
| IWARNING2/1/0 首末增量 | 0/0/0 | 0/0/0 |

因此这条链路能看约 10 ms 以上的持续频率变化以及 EDP 计数增长，但不能解析单次约 5 ms Submit。更重要的是，纯 idle 采样也在第二个点后维持 1650 MHz，说明采集器本身或同期系统活动足以把频率拉高；不能把“采样期间为 1650 MHz”写成 PA 的独立频率结论。idle probe 只证明接口、粒度和字段可用，并且该两秒内未观察到 EDP 计数增长。

#### 8.9.4 K 分布存在潜在并发污染，不能继续假定独占

准备持续负载探针前的进程核验发现，一项 root 级系统评测从 `2026-07-21 07:17:38 UTC` 起持续运行：

```text
python3 -m pytest test_team_evals.py ... --ascend-platform A5 ... -n 10
```

它的生命周期完整覆盖 K 正式 20 轮的 `07:50:49`～`08:09:01`，其沙箱任务也在 `07:47`、`07:58`、`08:10` 等时刻陆续落盘。当前用户没有权限检查该 root 进程的设备文件描述符，因此尚未证明它实际占用 device 0；但在它退出或由外部确认不触碰 device 0 以前，同一时间段不能再写成“经过进程核验的设备独占”。这也为 K 与较早 P/N 的尾部形态不同提供了一个必须排除的环境变量，但**不是已经成立的根因**。

因此本阶段暂不启动新的 PA、power probe 或 phase-PMU：先等待这项潜在 A5 任务退出，再完成一次持续真实 PA 负载下的独立 power 取证。若负载期 10.24 ms 采样仍稳定在目标频率且 EDP 计数不增长，只能排除“粗粒度持续 DVFS/EDP”为主因，仍不能排除短于一个采样周期的瞬态。随后立即回到 Materialize 同 ELF 单阶段 PMU，不把低功耗采样扩张成长期路线。

#### 8.9.5 持续真实 PA 负载探针与路线收口

root 级系统评测在 `09:07 UTC` 退出。后续每个正式样本前后均未再发现该进程；本机又没有契约完整且已安装的 device-PID 枚举 CLI，所以这里准确表述为“已排除已知并发评测”，不把普通用户无法查看 root 文件描述符包装成更强的设备独占证明。

第一次主动负载尝试保留为失败记录：`/tmp/fdwic_power_probe_20260721_090907.log` 中的 100 轮真实 PA 本身 PASS，但探测脚本错误地只等待 `/dev/davinci0`；实际 Python 进程只持有 `/dev/davinci_manager`，所以脚本没有启动 power 采集。随后手工启动的 `/tmp/asys_power_pa_active_20260721_091152/` 已落在负载结束之后，不进入负载结论。没有用这份错位数据凑数。

第二次改为复用真实 PA 日志中已经存在的 AICPU 初始化标志，不再猜设备文件。20 轮无泳道、无 PMU 的 Case1 在 `09:13:14.721` 出现初始化标志，power 采集于 `09:13:15.191` 开始并持续约 3 s；测试随后完成 20/20 轮，单轮 device 时间为 77.206～84.442 ms，因此采集窗口与真实负载明确重叠。PA 时间因与 `msprof` 同场而全部作废，只保留低功耗字段：

```text
/tmp/fdwic_power_probe_aligned_20260721_091233.log
/tmp/asys_power_pa_aligned_20260721_091233/
  asys_profiling_result_20260721091315170/
    PROF_000001_20260721091315186_03293075NQEEOOGB/
      device_0/sqlite/lowpower.db
```

die 0/1 各有 291 个点，290 个间隔全部为 10.240002 ms，覆盖 2.969600580 s；两 die 的 `data7_hard` 和 `data0_soft` 每一点均为 1650 MHz，POWERBRAKE 与 IWARNING2/1/0 每一点均为 0。由此可以排除该 3 s 负载窗内可被 10.24 ms 采样解析的持续频率阶跃和 EDP 计数增长。它仍不能排除单个约 5 ms Submit 内更短的瞬态，也不能证明当前 SLLC Major/module-error 告警与波动无关。至此停止扩展 power 路线，后续性能样本不再与 `asys/msprof` 同场。

### 8.10 同一 Materialize ELF 的阶段与其余 Submit 波动分解

**[阶段归因已完成；Materialize 不通过主载体门禁，波动落在同核其余 Submit]**

#### 8.10.1 冻结对象与采样协议

本阶段直接复用已经闭合的 `submit-pmu-materialize`，不增加 selector、raw 字段或逐事件记录。它精确覆盖 task-cap 检查和 `dist_submit_materialize_args()`，排除后继 PrepareMap；Case1 每核固定 1,280 次 begin/end。正式取数前后身份完全一致：

| 对象 | 固定值 |
| --- | --- |
| commit | `6acebc8fca0b053a2c014cf40ffed0dced17d1fa` |
| AICore cache | `b28cf51da4d4f547` |
| AICore 定义 | `PTO_FDWIC_SUBMIT_PMU=1;PTO_FDWIC_SUBMIT_PMU_PHASE_ID=3;PTO_FDWIC_TRACE_ENABLED=0` |
| `aicore_kernel.o` | 2,499,680 B / SHA256 `6fea16b12c3f3dd46fa8417969042fa5114e0d06ddb3c3dea0efe3d30254a8b0` |
| AIC/AIV combined | `afa5206a2a5f8cb9a7f6fa9992f8531bb04b92b21791bbdd0c8599bc1bd42584` / `31a1ef188cb27718cbba3aab7f939a7dc823975049b49c57ea05c39c5fef0061` |
| `libhost_runtime.so` | SHA256 `5f2e9af0892f64f28aded87e013a5b32425d86940b2e08cd34dad49cab0c0f9d` |

power 探针结束后先运行一轮同参数 Case1 重新预热：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_091623/
```

该轮明确排除出统计。随后连续运行 12 个彼此独立的 pytest 进程，每个进程固定 `--case Case1 --fdwic-profile submit-pmu-materialize --rounds 1 --skip-golden`；没有单进程多轮、人工 sleep 或极值删除。12/12 均 PASS：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_091706/
outputs/TestPagedAttentionUnroll_Case1_20260721_091750/
outputs/TestPagedAttentionUnroll_Case1_20260721_091835/
outputs/TestPagedAttentionUnroll_Case1_20260721_091919/
outputs/TestPagedAttentionUnroll_Case1_20260721_092002/
outputs/TestPagedAttentionUnroll_Case1_20260721_092045/
outputs/TestPagedAttentionUnroll_Case1_20260721_092128/
outputs/TestPagedAttentionUnroll_Case1_20260721_092213/
outputs/TestPagedAttentionUnroll_Case1_20260721_092258/
outputs/TestPagedAttentionUnroll_Case1_20260721_092342/
outputs/TestPagedAttentionUnroll_Case1_20260721_092425/
outputs/TestPagedAttentionUnroll_Case1_20260721_092508/
```

生产 `load_capture()` 逐份复验通过：12 份均为 96 核、32 AIC + 64 AIV、每核 1,280 次 Submit 和 Materialize begin/end，1,152 条逐核记录全部满足 status `0x7ff`、phase status `0x3f`、owner 配置/恢复、拓扑、计数顺序和风险阈值；primary/shadow request/miss 1,152/1,152 精确相等。12 份 HTML 也与当前 `render_report(raw)` 逐字节一致。总计覆盖 1,474,560 次 Submit/Materialize 调用。

#### 8.10.2 整数分解与 12 轮原始结果

继续沿用第 8.5 节定义。对每轮最晚结束核 `z`，再把它自己的 body 拆为：

```text
Pz = phase_elapsed_ticks[z]
Rz = submit_elapsed_ticks[z] - Pz
G  = A + Pz + Rz
```

`Pz/Rz` 只在同一 Materialize ELF、同一核、同一轮中相加；不会把 phase core-time 或其他 ELF 的时间拿来扣墙钟。下表时间均为 us：

| 轮次 | 时间戳 | G | M | X | A | Pz | Rz | 最晚核 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `091706` | 5129.351 | 4584.886 | 534.023 | 10.442 | 1020.215 | 4098.694 | c18/AIC/b18/l0 |
| 2 | `091750` | 4890.960 | 4575.058 | 311.516 | 4.386 | 1005.842 | 3880.732 | c22/AIC/b22/l0 |
| 3 | `091835` | 4927.376 | 4596.597 | 330.618 | 0.161 | 970.163 | 3957.052 | c25/AIC/b25/l0 |
| 4 | `091919` | 4733.656 | 4594.046 | 129.707 | 9.904 | 1002.399 | 3721.353 | c16/AIC/b16/l0 |
| 5 | `092002` | 4665.849 | 4577.212 | 84.337 | 4.300 | 1020.395 | 3641.154 | c71/AIV/b19/l2 |
| 6 | `092045` | 4659.749 | 4579.614 | 72.424 | 7.711 | 1022.620 | 3629.418 | c41/AIV/b4/l2 |
| 7 | `092128` | 5088.129 | 4831.484 | 256.510 | 0.136 | 1032.260 | 4055.733 | c45/AIV/b6/l2 |
| 8 | `092213` | 4943.694 | 4593.721 | 344.279 | 5.693 | 1002.572 | 3935.429 | c27/AIC/b27/l0 |
| 9 | `092258` | 4653.506 | 4586.275 | 58.016 | 9.215 | 1056.466 | 3587.825 | c70/AIV/b19/l1 |
| 10 | `092342` | 4708.335 | 4595.176 | 103.877 | 9.281 | 1022.974 | 3676.080 | c72/AIV/b20/l1 |
| 11 | `092425` | 4731.198 | 4595.085 | 127.303 | 8.810 | 991.187 | 3731.201 | c11/AIC/b11/l0 |
| 12 | `092508` | 4755.224 | 4583.503 | 161.814 | 9.907 | 1010.032 | 3735.285 | c28/AIC/b28/l0 |

12 轮最晚核分散在 12 个不同 core，AIC/AIV 为 7/5，不支持固定坏核解释。分布使用整数 tick 派生，MAD 为未缩放中位绝对偏差，P10/P90 采用线性插值：

| 指标 | 最小值 | 中位数 | 最大值 | MAD | P10 | P90 | P90-P10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G | 4653.506 | 4744.440 | 5129.351 | 87.813 | 4660.359 | 5073.686 | 413.327 |
| M | 4575.058 | 4589.998 | 4831.484 | 5.836 | 4577.452 | 4596.455 | 19.003 |
| X | 58.016 | 145.760 | 534.023 | 80.540 | 73.616 | 342.913 | 269.298 |
| A | 0.136 | 8.261 | 10.442 | 1.914 | 0.575 | 9.907 | 9.332 |
| Pz | 970.163 | 1015.124 | 1056.466 | 10.917 | 992.308 | 1031.331 | 39.023 |
| Rz | 3587.825 | 3733.243 | 4098.694 | 124.621 | 3630.592 | 4045.865 | 415.273 |
| Materialize mean/core | 1020.461 | 1021.320 | 1022.028 | 0.234 | 1020.962 | 1021.957 | 0.994 |
| 其余 Submit mean/core | 3500.079 | 3523.462 | 3834.918 | 9.251 | 3506.308 | 3539.937 | 33.629 |

#### 8.10.3 主载体门禁与结论边界

沿用预先固定的双门禁：`abs(Spearman(G, component)) >= 0.7`，且 component 的 `P90-P10 >= 50% * G(P90-P10)`：

| 分量 | Spearman(G, ·) | P90-P10 | 占 G 波幅 | 判定 |
| --- | ---: | ---: | ---: | --- |
| M | +0.287 | 19.003 us | 4.60% | FAIL |
| X | +0.958 | 269.298 us | 65.15% | PASS：该 ELF 为慢尾形态 |
| A | -0.105 | 9.332 us | 2.26% | FAIL |
| Pz | -0.343 | 39.023 us | 9.44% | **FAIL：Materialize 不是主载体** |
| Rz | +0.986 | 415.273 us | 100.47% | **PASS：变化落在同核其余 Submit** |
| Materialize mean/core | +0.091 | 0.994 us | 0.24% | FAIL |
| 其余 Submit mean/core | +0.720 | 33.629 us | 8.14% | FAIL：相关但整体幅度不足 |

因此本阶段成立的结论是：**在当前 Materialize 诊断 ELF 中，最晚核的波动不由 Materialize 携带，而由该核其余 Submit 路径携带。** Materialize 每核平均 core-time 几乎不动，12 轮 P90-P10 只有 0.994 us；每次 request 中位 233.061、范围 232.911～233.243，每次 miss 中位 1.438、范围 1.424～1.447，同样没有随 G 同向的大幅变化。它在同 ELF 中仍占约 22.47% core-time，说明绝对成本重要，但“占比大” 不能替代波动门禁，也不能据此先改固定扫描逻辑。

本组 G 的 P90-P10 为 413.327 us，已有可分辨波形，而 Pz 的相关性和幅度都远离门槛，因此不扩到 20 轮。更重要的是，本组由 X 而不是 M 通过门禁，没有复现权威 P 的多数核 body 共同伸缩形态；所以 Rz 结论只限定于 Materialize ELF，不能直接外推成 P 的根因或可兑现收益。下一阶段按既定顺序切换到现有 `submit-pmu-claim`，优先检查包含 return-ready atomic 的 Claim 是否携带同 ELF 波动；仍不修改业务代码。

### 8.11 同一 Claim ELF 的阶段与其余 Submit 波动分解

**[阶段归因已完成；Claim 不通过主载体门禁，角色差异不能冒充跨轮波动]**

#### 8.11.1 B1、冻结对象与正式样本

本阶段复用第 8.3.5 节的 `submit-pmu-claim`，边界为 `claim_begin_to_claim_end`。它包含 task-cap 检查、实际 Claim 和 flags 构造；真实 FetchMax 返回等待位于该业务区间内，但诊断 ELF 编译掉 atomic 泳道观察代码。每个 Submit 固定进入一次外层 Claim，所以 Case1 shape 仍为每核 1,280 对 begin/end，不增加 winner 或 atomic 动态字段。

文档提交改变 source-v2 HEAD 后，先在新构建上运行 B1：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260721_093141/
```

结果为 96 核各 5 次 Submit/Claim、phase id `4`、status `0x3f`、primary/shadow 96/96 精确相等，HTML 与当前 renderer 字节一致；全局 243.386 us 仍只作结构证据。随后冻结：

| 对象 | 固定值 |
| --- | --- |
| commit | `52dc04c792cd5f6982fe7a6d1272dcc4f43231bc` |
| AICore cache | `88d4ef05843d4904` |
| AICore 定义 | `PTO_FDWIC_SUBMIT_PMU=1;PTO_FDWIC_SUBMIT_PMU_PHASE_ID=4;PTO_FDWIC_TRACE_ENABLED=0` |
| `aicore_kernel.o` | 2,590,880 B / `.text` 154,192 B / SHA256 `05f0c6e8edff1319770f9117ea17efb8c81a8822d6cbdaa199ebd339fb039703` |
| AIC/AIV combined | `fd0fa24ac1b117d4705f6405dc60497992efef17f71f268d46326e5fc2a1ac29` / `6a7bdf0221e15d750e9b531e3083caf728ad0527779e642bef41ef7f388e68f7` |
| `libhost_runtime.so` | SHA256 `5f2e9af0892f64f28aded87e013a5b32425d86940b2e08cd34dad49cab0c0f9d` |

Case1 预热件 `..._093327` 明确排除。12 个独立正式进程仍固定 `--rounds 1 --skip-golden`，没有 sleep、极值删除或其他 profile 穿插：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_093412/
outputs/TestPagedAttentionUnroll_Case1_20260721_093454/
outputs/TestPagedAttentionUnroll_Case1_20260721_093536/
outputs/TestPagedAttentionUnroll_Case1_20260721_093620/
outputs/TestPagedAttentionUnroll_Case1_20260721_093703/
outputs/TestPagedAttentionUnroll_Case1_20260721_093746/
outputs/TestPagedAttentionUnroll_Case1_20260721_093829/
outputs/TestPagedAttentionUnroll_Case1_20260721_093912/
outputs/TestPagedAttentionUnroll_Case1_20260721_093954/
outputs/TestPagedAttentionUnroll_Case1_20260721_094037/
outputs/TestPagedAttentionUnroll_Case1_20260721_094121/
outputs/TestPagedAttentionUnroll_Case1_20260721_094203/
```

12/12 pytest、生产 `load_capture()` 与 HTML 精确重建全部 PASS；1,152 条逐核记录均为 96×1280、phase id `4`、phase status `0x3f`、primary=shadow，owner、拓扑、计数顺序、phase 时间和风险阈值闭合。采样前后上述 AIC/AIV/host SHA 不变，已知 root A5 评测也未重新出现。

#### 8.11.2 原始分解与角色差异

继续定义 `Pz=Claim[z]`、`Rz=body[z]-Pz`，逐轮精确满足 `G=A+Pz+Rz=A+M+X`。下表单位均为 us：

| 轮次 | 时间戳 | G | M | X | A | Pz | Rz | 最晚核 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `093412` | 4620.233 | 4556.721 | 59.596 | 3.915 | 1024.912 | 3591.406 | c79/AIV/b23/l2 |
| 2 | `093454` | 4719.079 | 4524.864 | 181.847 | 12.368 | 415.432 | 4291.279 | c2/AIC/b2/l0 |
| 3 | `093536` | 4676.573 | 4535.911 | 140.432 | 0.230 | 370.210 | 4306.133 | c28/AIC/b28/l0 |
| 4 | `093620` | 4605.915 | 4542.673 | 62.686 | 0.556 | 1011.722 | 3593.637 | c66/AIV/b17/l1 |
| 5 | `093703` | 5015.477 | 4536.442 | 477.636 | 1.399 | 384.459 | 4629.619 | c24/AIC/b24/l0 |
| 6 | `093746` | 4952.543 | 4697.900 | 249.944 | 4.699 | 979.619 | 3968.225 | c39/AIV/b3/l2 |
| 7 | `093829` | 4657.082 | 4534.479 | 122.496 | 0.107 | 397.740 | 4259.235 | c22/AIC/b22/l0 |
| 8 | `093912` | 4676.111 | 4554.137 | 111.189 | 10.785 | 386.509 | 4278.817 | c26/AIC/b26/l0 |
| 9 | `093954` | 4726.002 | 4533.742 | 188.481 | 3.780 | 386.908 | 4335.314 | c24/AIC/b24/l0 |
| 10 | `094037` | 5238.567 | 4702.629 | 535.607 | 0.331 | 401.741 | 4836.495 | c23/AIC/b23/l0 |
| 11 | `094121` | 4725.767 | 4554.668 | 161.607 | 9.492 | 394.837 | 4321.438 | c0/AIC/b0/l0 |
| 12 | `094203` | 4633.868 | 4546.530 | 87.018 | 0.320 | 382.272 | 4251.276 | c31/AIC/b31/l0 |

最晚核分散到 11 个 core，AIC/AIV 为 9/3。`Pz` 的 370～1,025 us 大范围主要来自最晚核角色切换：三轮 AIV 均约 980～1,025 us，九轮 AIC 均约 370～415 us。这是同一 Claim ELF 内真实存在的角色成本差异，但它不随 G 单调变化；不能因数值范围大就称 Claim 携带跨轮波动。完整分布为：

| 指标 | 最小值 | 中位数 | 最大值 | MAD | P10 | P90 | P90-P10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G | 4605.915 | 4697.826 | 5238.567 | 52.351 | 4621.596 | 5009.184 | 387.587 |
| M | 4524.864 | 4544.601 | 4702.629 | 10.095 | 4533.815 | 4683.782 | 149.967 |
| X | 59.596 | 151.019 | 535.607 | 51.916 | 65.119 | 454.867 | 389.748 |
| A | 0.107 | 2.590 | 12.368 | 2.264 | 0.239 | 10.656 | 10.417 |
| Pz | 370.210 | 396.288 | 1024.912 | 12.923 | 382.491 | 1008.512 | 626.021 |
| Rz | 3591.406 | 4285.048 | 4836.495 | 43.328 | 3631.096 | 4600.189 | 969.093 |
| Claim mean/core | 778.854 | 830.892 | 856.279 | 7.354 | 813.978 | 842.240 | 28.262 |
| 其余 Submit mean/core | 3627.936 | 3676.057 | 3924.426 | 23.765 | 3642.722 | 3902.263 | 259.541 |

#### 8.11.3 主载体门禁与阶段决定

| 分量 | Spearman(G, ·) | P90-P10 | 占 G 波幅 | 判定 |
| --- | ---: | ---: | ---: | --- |
| M | +0.154 | 149.967 us | 38.69% | FAIL |
| X | +0.979 | 389.748 us | 100.56% | PASS：仍为慢尾形态 |
| A | +0.175 | 10.417 us | 2.69% | FAIL |
| Pz | -0.189 | 626.021 us | 161.52% | **FAIL：幅度大但方向不相关** |
| Rz | +0.790 | 969.093 us | 250.03% | **PASS：同核其余 Submit 携带变化** |
| Claim mean/core | -0.476 | 28.262 us | 7.29% | FAIL |
| 其余 Submit mean/core | +0.448 | 259.541 us | 66.96% | FAIL：幅度够、相关性不足 |

`Pz/Rz` 因 AIC/AIV 角色成本相反补偿，单个分量波幅可以大于 G；二者仍只按每轮同核整数关系闭合，不能把 161.52% 与 250.03% 相加。主门禁要求相关性和幅度同时成立，因此 Claim 明确失败，不能把 ClaimMax/return-ready atomic 当作本组慢轮原因。

Claim 在同 ELF 中的 core-time 份额中位为 18.40%，每次 request 中位 80.171、范围 80.097～80.236，每次 miss 中位 1.950、范围 1.948～1.951；这些稳定值说明 Claim 仍是绝对成本区域，但没有提供“慢轮执行了更多 Claim 取指或 miss”的证据。本组 G 有 387.587 us 的可分辨波形，Pz 相关性又远离阈值，所以不扩到 20 轮。其 M 仍未复现权威 P 的共同伸缩，结论继续严格限定在 Claim ELF。下一阶段切换到已经存在的 `submit-pmu-submit-transition`，检查相邻 Submit 之间的动态控制/回放间隙；不修改 Claim 或 atomic 业务逻辑。

### 8.12 同一 SubmitTransition ELF 的阶段与其余 Submit 波动分解

**[阶段归因已完成；相邻 Submit 间隙不通过主载体门禁]**

#### 8.12.1 N-1 结构闭合与冻结样本

`submit-pmu-submit-transition` 复用统一 Submit hook，聚合上一次 Submit end 到下一次 `dist_submit_begin()` 完成之间的返回、编排衔接和下一任务准备；它不生成末次 Submit 之后的伪间隙。因此 B1 每核固定 4 次、Case1 每核固定 1,279 次，不能沿用普通 phase 的 N 次 shape，也不能从聚合 raw 中还原不同 task-kind 转换。

新 HEAD 的 B1 为：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260721_094639/
```

96 核均为 5 个 Submit、4 对 transition，phase id `6`、status `0x3f`、primary/shadow 96/96 精确相等，HTML 精确重建；258.277 us 只作结构证据。冻结身份为：

| 对象 | 固定值 |
| --- | --- |
| commit | `7fa7399f57c0bb490cb762b74a0e184c8f2bdcf4` |
| AICore cache | `d17b87d79477c0e1` |
| AICore 定义 | `PTO_FDWIC_SUBMIT_PMU=1;PTO_FDWIC_SUBMIT_PMU_PHASE_ID=6;PTO_FDWIC_TRACE_ENABLED=0` |
| `aicore_kernel.o` | 2,584,560 B / `.text` 153,424 B / SHA256 `a5f96fa4c78cfd038cf7ef2bbc1f2c811cd4a2cf1cba6466b867ce24d49543f9` |
| AIC/AIV combined | `0badb10df88e36cb0782c6e0fee3787972cf8a403a4f4f671e70f716ab6a6be5` / `754510d18b6e09a41be7c5878e7f1cdf802e7b1f60bc83d035f341ad3475b394` |
| `libhost_runtime.so` | SHA256 `5f2e9af0892f64f28aded87e013a5b32425d86940b2e08cd34dad49cab0c0f9d` |

Case1 预热 `..._094821` 排除后，12 个独立正式进程为：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_094903/
outputs/TestPagedAttentionUnroll_Case1_20260721_094948/
outputs/TestPagedAttentionUnroll_Case1_20260721_095032/
outputs/TestPagedAttentionUnroll_Case1_20260721_095117/
outputs/TestPagedAttentionUnroll_Case1_20260721_095202/
outputs/TestPagedAttentionUnroll_Case1_20260721_095247/
outputs/TestPagedAttentionUnroll_Case1_20260721_095330/
outputs/TestPagedAttentionUnroll_Case1_20260721_095415/
outputs/TestPagedAttentionUnroll_Case1_20260721_095459/
outputs/TestPagedAttentionUnroll_Case1_20260721_095544/
outputs/TestPagedAttentionUnroll_Case1_20260721_095628/
outputs/TestPagedAttentionUnroll_Case1_20260721_095712/
```

12/12 pytest、生产 raw 消费和 HTML 重建均 PASS；1,152 条逐核记录全部满足 96×1280 Submit、每核 1,279 次 begin/end、phase id `6`、phase status `0x3f`、primary=shadow、owner/拓扑/计数和时间门禁，共覆盖 1,473,408 个真实相邻间隙。采样前后对象 SHA 不变，已知 root A5 评测未重新出现。

#### 8.12.2 原始分解与角色分层

继续以最晚核 `z` 定义 `Pz=Transition[z]` 和 `Rz=body[z]-Pz`。下表单位均为 us：

| 轮次 | 时间戳 | G | M | X | A | Pz | Rz | 最晚核 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `094903` | 4706.898 | 4589.310 | 110.019 | 7.569 | 523.401 | 4175.928 | c33/AIV/b0/l2 |
| 2 | `094948` | 4772.953 | 4585.756 | 186.890 | 0.307 | 483.429 | 4289.217 | c62/AIV/b15/l1 |
| 3 | `095032` | 4944.667 | 4554.933 | 389.734 | 0.000 | 385.113 | 4559.554 | c23/AIC/b23/l0 |
| 4 | `095117` | 4665.650 | 4561.532 | 98.951 | 5.168 | 521.598 | 4138.884 | c89/AIV/b28/l2 |
| 5 | `095202` | 4984.704 | 4567.462 | 416.864 | 0.378 | 363.112 | 4621.214 | c28/AIC/b28/l0 |
| 6 | `095247` | 5136.771 | 4559.525 | 571.823 | 5.423 | 421.619 | 4709.729 | c26/AIC/b26/l0 |
| 7 | `095330` | 4685.859 | 4586.337 | 87.935 | 11.587 | 460.224 | 4214.048 | c81/AIV/b24/l2 |
| 8 | `095415` | 4907.677 | 4597.949 | 305.505 | 4.223 | 374.434 | 4529.020 | c15/AIC/b15/l0 |
| 9 | `095459` | 4965.656 | 4578.873 | 381.935 | 4.848 | 420.177 | 4540.631 | c26/AIC/b26/l0 |
| 10 | `095544` | 4706.148 | 4559.293 | 139.362 | 7.493 | 381.920 | 4316.735 | c23/AIC/b23/l0 |
| 11 | `095628` | 4758.772 | 4581.688 | 169.271 | 7.813 | 458.108 | 4292.851 | c64/AIV/b16/l1 |
| 12 | `095712` | 4643.497 | 4549.686 | 92.862 | 0.949 | 456.329 | 4186.219 | c94/AIV/b31/l1 |

最晚核分散到 10 个 core，AIC/AIV 各 6 轮。AIC 的 `Pz` 为 363～422 us，AIV 为 456～523 us；整体 `rho=-0.531` 主要受“慢轮更常由 Pz 较短的 AIC 收尾”影响。角色内 `Spearman(G,Pz)` 只有 AIC `+0.371`、AIV `+0.257`，均不支持 transition 随慢轮增长。完整分布为：

| 指标 | 最小值 | 中位数 | 最大值 | MAD | P10 | P90 | P90-P10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G | 4643.497 | 4765.863 | 5136.771 | 111.289 | 4667.671 | 4982.799 | 315.128 |
| M | 4549.686 | 4573.168 | 4597.949 | 13.406 | 4555.369 | 4589.013 | 33.644 |
| X | 87.935 | 178.080 | 571.823 | 87.682 | 93.470 | 414.151 | 320.681 |
| A | 0.000 | 5.008 | 11.587 | 2.683 | 0.314 | 7.789 | 7.474 |
| Pz | 363.112 | 438.974 | 523.401 | 49.158 | 375.183 | 517.781 | 142.599 |
| Rz | 4138.884 | 4304.793 | 4709.729 | 147.387 | 4176.957 | 4615.048 | 438.091 |
| Transition mean/core | 447.180 | 448.720 | 451.075 | 0.553 | 447.434 | 449.755 | 2.321 |
| 其余 Submit mean/core | 4047.639 | 4087.141 | 4170.705 | 16.499 | 4061.148 | 4121.627 | 60.479 |

#### 8.12.3 门禁与阶段决定

| 分量 | Spearman(G, ·) | P90-P10 | 占 G 波幅 | 判定 |
| --- | ---: | ---: | ---: | --- |
| M | +0.070 | 33.644 us | 10.68% | FAIL |
| X | +0.965 | 320.681 us | 101.76% | PASS：仍为慢尾形态 |
| A | -0.343 | 7.474 us | 2.37% | FAIL |
| Pz | -0.531 | 142.599 us | 45.25% | **FAIL：相关性和幅度均不足** |
| Rz | +0.895 | 438.091 us | 139.02% | **PASS：同核其余 Submit 携带变化** |
| Transition mean/core | -0.175 | 2.321 us | 0.74% | FAIL |
| 其余 Submit mean/core | +0.399 | 60.479 us | 19.19% | FAIL |

Pz 的幅度比例接近但仍低于 50%，相关性又只有 -0.531；角色分层后相关性更低，Transition mean/core 也几乎不动。因此当前不是样本不足导致的临界结论，不扩到 20 轮。该 ELF 内 transition core-time 份额中位为 9.90%，每间隙 request 中位 134.226、范围 134.085～134.310，每间隙 miss 中位 4.375、范围 4.278～4.553；I-cache observed 占比高不等于它携带跨轮时延，更不能换算成可相减的墙钟损失。

本组仍由 X 而不是 M 通过，不能外推权威 P。下一阶段依序复用现有 `submit-pmu-arg-build`，检查 Claim.end 到 Materialize.begin 的同步构参与恢复路径；不修改 transition 或编排业务。

### 8.13 同一 ArgBuild ELF 的阶段与其余 Submit 波动分解

**[阶段归因已完成；ArgBuild 不通过主载体门禁]**

#### 8.13.1 B1、冻结对象与正式样本

`submit-pmu-arg-build` 的边界为 `claim_end_to_materialize_begin`：起点在真实 Claim 完成后，终点在匹配 Finish 恢复并校验 ticket 后、Materialize 入口前。它覆盖 Begin 返回、同步 eager callback 构参和 Finish 重入，不包含 Claim 与 Materialize。本阶段直接复用已经闭合的 phase id `1` 和 12,416 B 固定 ABI，没有增加 raw 字段。

先用 B1 验证每核 5 对边界：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260721_100220/
```

96 核均为 5 次 Submit/ArgBuild，phase status `0x3f`、primary/shadow 精确相等，HTML 精确重建；246.495 us 只作结构证据。随后冻结对象：

| 对象 | 固定值 |
| --- | --- |
| commit | `afeb515aaed182200d62f74f4cba0797aceeb2c2` |
| AICore cache | `8d3a86ebd090fe15` |
| AICore 定义 | `PTO_FDWIC_SUBMIT_PMU=1;PTO_FDWIC_SUBMIT_PMU_PHASE_ID=1;PTO_FDWIC_TRACE_ENABLED=0` |
| `aicore_kernel.o` | 2,567,472 B / `.text` 149,328 B / SHA256 `5172a04d0c2ffa05a511f9807b28fcf8be81d4a05783f095e036ba807238eea0` |
| AIC/AIV combined | `1dc77366f67f919594af753c5119b0c49af434cd30501c09e67be03e548cb20f` / `757c9db2528916c77443d6bcd197a5d45848ebf75c064ac874fbff52e445deb9` |
| `libhost_runtime.so` | SHA256 `5f2e9af0892f64f28aded87e013a5b32425d86940b2e08cd34dad49cab0c0f9d` |

Case1 预热件 `..._100401` 明确排除。随后连续运行 12 个彼此独立的 pytest 进程，每个固定 `--rounds 1 --skip-golden`，没有 sleep、极值删除或其他 profile 穿插：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_100445/
outputs/TestPagedAttentionUnroll_Case1_20260721_100529/
outputs/TestPagedAttentionUnroll_Case1_20260721_100613/
outputs/TestPagedAttentionUnroll_Case1_20260721_100656/
outputs/TestPagedAttentionUnroll_Case1_20260721_100739/
outputs/TestPagedAttentionUnroll_Case1_20260721_100824/
outputs/TestPagedAttentionUnroll_Case1_20260721_100909/
outputs/TestPagedAttentionUnroll_Case1_20260721_100953/
outputs/TestPagedAttentionUnroll_Case1_20260721_101036/
outputs/TestPagedAttentionUnroll_Case1_20260721_101121/
outputs/TestPagedAttentionUnroll_Case1_20260721_101204/
outputs/TestPagedAttentionUnroll_Case1_20260721_101249/
```

12/12 pytest、生产 `load_capture()` 和 HTML 重建全部 PASS。1,152 条逐核记录均为 96×1,280、phase id `1`、phase status `0x3f`、owner 配置/恢复 96/96、计数低于风险阈值；primary/shadow request/miss 1,152/1,152 逐核精确相等。共覆盖 1,474,560 次真实 ArgBuild 边界。raw 能证明 profile、shape、状态、拓扑、owner 和计数闭合；同一 ELF/HEAD 仍由上述外部冻结 SHA 证明，不能倒过来从 raw 杜撰。

#### 8.13.2 整数分解与角色双峰

继续以最晚结束核 `z` 定义 `Pz=ArgBuild[z]`、`Rz=body[z]-Pz`，逐轮精确满足 `G=A+M+X=A+Pz+Rz`。下表单位均为 us：

| 轮次 | 时间戳 | G | M | X | A | Pz | Rz | 最晚核 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `100445` | 4999.071 | 4549.461 | 439.759 | 9.851 | 198.725 | 4790.495 | c28/AIC/b28/l0 |
| 2 | `100529` | 4724.531 | 4557.943 | 160.395 | 6.193 | 292.721 | 4425.617 | c67/AIV/b17/l2 |
| 3 | `100613` | 4826.878 | 4637.162 | 178.973 | 10.743 | 276.317 | 4539.818 | c76/AIV/b22/l1 |
| 4 | `100656` | 4824.865 | 4577.124 | 235.010 | 12.731 | 199.131 | 4613.003 | c23/AIC/b23/l0 |
| 5 | `100739` | 4968.121 | 4543.257 | 412.807 | 12.057 | 199.732 | 4756.332 | c23/AIC/b23/l0 |
| 6 | `100824` | 4900.956 | 4548.284 | 342.089 | 10.583 | 198.530 | 4691.843 | c20/AIC/b20/l0 |
| 7 | `100909` | 4689.894 | 4575.105 | 100.829 | 13.959 | 279.883 | 4396.052 | c89/AIV/b28/l2 |
| 8 | `100953` | 4647.705 | 4565.437 | 71.148 | 11.120 | 286.475 | 4350.110 | c90/AIV/b29/l1 |
| 9 | `101036` | 4794.409 | 4594.467 | 199.771 | 0.171 | 198.698 | 4595.540 | c31/AIC/b31/l0 |
| 10 | `101121` | 4727.365 | 4589.965 | 129.642 | 7.758 | 298.120 | 4421.487 | c78/AIV/b23/l1 |
| 11 | `101204` | 4624.150 | 4556.818 | 66.997 | 0.335 | 198.700 | 4425.115 | c28/AIC/b28/l0 |
| 12 | `101249` | 4664.981 | 4554.060 | 100.759 | 10.162 | 198.868 | 4455.951 | c6/AIC/b6/l0 |

最晚核分散到 10 个 core，AIC/AIV 为 7/5。`Pz` 呈现约 199 us 的 AIC 与 276～298 us 的 AIV 双峰；按角色复算 `Spearman(G,Pz)` 也只有 AIC `+0.214`、AIV `-0.100`，而对应 `Rz` 为 `+1.000/+0.900`。角色双峰不是 ArgBuild 携带跨轮波动的证据。完整分布为：

| 指标 | 最小值 | 中位数 | 最大值 | MAD | P10 | P90 | P90-P10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G | 4624.150 | 4760.887 | 4999.071 | 83.450 | 4649.433 | 4961.405 | 311.972 |
| M | 4543.257 | 4561.690 | 4637.162 | 13.411 | 4548.402 | 4594.017 | 45.615 |
| X | 66.997 | 169.684 | 439.759 | 68.890 | 74.109 | 405.735 | 331.626 |
| A | 0.171 | 10.373 | 13.959 | 2.022 | 0.921 | 12.664 | 11.743 |
| Pz | 198.530 | 199.432 | 298.120 | 0.818 | 198.698 | 292.096 | 93.398 |
| Rz | 4350.110 | 4497.885 | 4790.495 | 99.744 | 4398.596 | 4749.883 | 351.288 |
| ArgBuild mean/core | 257.919 | 258.627 | 261.132 | 0.452 | 258.188 | 260.556 | 2.368 |
| 其余 Submit mean/core | 4246.566 | 4266.118 | 4387.781 | 17.411 | 4247.448 | 4328.366 | 80.918 |

#### 8.13.3 门禁、I-cache 观察值与结论边界

| 分量 | Spearman(G, ·) | P90-P10 | 占 G 波幅 | 判定 |
| --- | ---: | ---: | ---: | --- |
| M | -0.210 | 45.615 us | 14.62% | FAIL |
| X | +0.972 | 331.626 us | 106.30% | PASS：迁移慢尾形态 |
| A | +0.175 | 11.743 us | 3.76% | FAIL |
| Pz | -0.259 | 93.398 us | 29.94% | **FAIL：ArgBuild 不是主载体** |
| Rz | +0.867 | 351.288 us | 112.60% | **PASS：同核其余 Submit 携带变化** |
| ArgBuild mean/core | +0.615 | 2.368 us | 0.76% | FAIL |
| 其余 Submit mean/core | -0.021 | 80.918 us | 25.94% | FAIL |

同一 ELF 内 ArgBuild aggregate core-time 份额中位为 5.711%，范围 5.559%～5.792%。ALL 的 request/call 中位 119.891、范围 119.806～119.979，miss/call 中位 2.582、范围 2.571～2.612；AIC/AIV 的 miss/call 中位分别为 0.033/3.857。它们都是 running read-clear observed，不是无插桩净业务数；稳定的 observed 值也不能跨 ELF 换算成可直接消除的墙钟损失。

因此本阶段只成立一个受限结论：**当前 ArgBuild 诊断 ELF 的 311.972 us 波形不是 ArgBuild 携带，而是最晚核的非 ArgBuild 慢尾。** 本组由 X 而不是 M 通过，没有复现权威 perf-clock P 的约 1.408 ms 多数核共同伸缩；`Rz` 仍只是算术剩余，不能重命名为 atomic、flag wait、I-cache 或其他业务段。Pz 相关性和幅度都不临界，phase mean/core 波幅也只有 G 的 0.76%，所以不扩到 20 轮。下一阶段按既定顺序切换到现有 `submit-pmu-register`，完成最后一个存量 selector 的同 ELF 归因。

### 8.14 同一 Register ELF 的阶段与其余 Submit 波动分解

**[阶段归因已完成；Register 不通过主载体门禁]**

#### 8.14.1 真实契约、冻结对象与正式样本

Register observer 有三个互斥源码挂点，但每个正常 Submit 只命中其中一个，因此固定 shape 是每核 `N` 对而不是 `3N` 对：B1 为 5，Case1 为 1,280。边界从真实 `dist_submit_register_outputs()` 调用入口到返回，排除普通泳道 Register 的前序记录发布、结束 timestamp 与 caller 衔接；它只能命名为 RegisterOutputs 调用体，不能冒充完整 Register timestamp-to-timestamp span 或单次 TensorMap insert。

新 HEAD 上先运行 B1：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260721_102100/
```

96 核各 5 次 Submit/Register，phase id `5`、status `0x3f`、primary/shadow 96/96 精确相等，HTML 精确重建；83.517 us 只作结构证据。随后冻结：

| 对象 | 固定值 |
| --- | --- |
| commit | `57aedfee51883a5840e618037b49ac9c630d1cb6` |
| AICore cache | `32c26e06ad76d186` |
| source state | `source-v2:57aedfee51883a5840e618037b49ac9c630d1cb6:a799850ce15b3a0fcb6cd9dc6d89cc81a67f08c6d2a9a7afeb4c76b24ab75bea:ff85282e42fc1b03ba04d010a0b3b793349657e8b1e321fd7b31017015f1dee9` |
| AICore 定义 | `PTO_FDWIC_SUBMIT_PMU=1;PTO_FDWIC_SUBMIT_PMU_PHASE_ID=5;PTO_FDWIC_TRACE_ENABLED=0` |
| `aicore_kernel.o` | 2,573,816 B / `.text` 150,608 B / SHA256 `39760b286b14a3ec7a078a0d0a60981ce86ba3a58be68626df02bab056c107f1` |
| AIC/AIV combined | `78677b168454de544a708531781f00bcb1d3a2f65518affae262955f5f585933` / `8566136503bf6d7a60290a6f8776d515b13ec7469de4d4a084ced38d1ad24102` |
| `libhost_runtime.so` | SHA256 `5f2e9af0892f64f28aded87e013a5b32425d86940b2e08cd34dad49cab0c0f9d` |

Case1 预热 `..._102246` 排除后，12 个独立正式进程全部 PASS：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_102330/
outputs/TestPagedAttentionUnroll_Case1_20260721_102413/
outputs/TestPagedAttentionUnroll_Case1_20260721_102458/
outputs/TestPagedAttentionUnroll_Case1_20260721_102543/
outputs/TestPagedAttentionUnroll_Case1_20260721_102626/
outputs/TestPagedAttentionUnroll_Case1_20260721_102711/
outputs/TestPagedAttentionUnroll_Case1_20260721_102754/
outputs/TestPagedAttentionUnroll_Case1_20260721_102839/
outputs/TestPagedAttentionUnroll_Case1_20260721_102923/
outputs/TestPagedAttentionUnroll_Case1_20260721_103005/
outputs/TestPagedAttentionUnroll_Case1_20260721_103049/
outputs/TestPagedAttentionUnroll_Case1_20260721_103131/
```

12/12 生产 `load_capture()` 和 HTML 重建通过；1,152 条逐核记录全部满足 96×1,280、phase id `5`、phase status `0x3f`、owner 配置/恢复、拓扑、风险阈值和计数顺序，primary/shadow 1,152/1,152 精确相等。共覆盖 1,474,560 次真实 RegisterOutputs 调用体。采样前后上述实物 SHA 和 source state 不变，已知外部评测未出现。

#### 8.14.2 整数分解与 12 轮结果

继续定义 `Pz=Register[z]`、`Rz=body[z]-Pz`，12/12 精确满足 `G=A+M+X=A+Pz+Rz`。下表单位均为 us：

| 轮次 | 时间戳 | G | M | X | A | Pz | Rz | 最晚核 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `102330` | 5053.021 | 4784.164 | 268.581 | 0.276 | 233.630 | 4819.115 | c92/AIV/b30/l1 |
| 2 | `102413` | 4708.077 | 4587.936 | 116.475 | 3.666 | 231.358 | 4473.053 | c55/AIV/b11/l2 |
| 3 | `102458` | 4737.712 | 4593.329 | 136.797 | 7.586 | 250.913 | 4479.213 | c91/AIV/b29/l2 |
| 4 | `102543` | 4679.146 | 4594.077 | 77.060 | 8.009 | 231.034 | 4440.103 | c94/AIV/b31/l1 |
| 5 | `102626` | 4964.744 | 4608.269 | 354.148 | 2.327 | 212.237 | 4750.180 | c24/AIC/b24/l0 |
| 6 | `102711` | 5201.300 | 4590.639 | 606.702 | 3.960 | 232.572 | 4964.768 | c23/AIC/b23/l0 |
| 7 | `102754` | 5097.611 | 4818.618 | 278.777 | 0.216 | 245.937 | 4851.458 | c34/AIV/b1/l1 |
| 8 | `102839` | 4689.188 | 4604.074 | 78.356 | 6.757 | 236.377 | 4446.054 | c31/AIC/b31/l0 |
| 9 | `102923` | 4948.061 | 4612.592 | 329.669 | 5.800 | 244.657 | 4697.604 | c28/AIC/b28/l0 |
| 10 | `103005` | 4698.544 | 4598.600 | 98.540 | 1.404 | 244.352 | 4452.788 | c60/AIV/b14/l1 |
| 11 | `103049` | 4724.767 | 4595.187 | 126.730 | 2.850 | 265.833 | 4456.084 | c20/AIC/b20/l0 |
| 12 | `103131` | 4735.855 | 4604.962 | 123.287 | 7.606 | 241.715 | 4486.534 | c48/AIV/b8/l1 |

最晚核 12 轮落在 12 个不同 core，AIC/AIV 为 5/7。完整分布为：

| 指标 | 最小值 | 中位数 | 最大值 | MAD | P10 | P90 | P90-P10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G | 4679.146 | 4736.784 | 5201.300 | 52.617 | 4690.124 | 5093.152 | 403.028 |
| M | 4587.936 | 4601.337 | 4818.618 | 7.634 | 4590.908 | 4767.007 | 176.099 |
| X | 77.060 | 131.763 | 606.702 | 54.055 | 80.375 | 351.700 | 271.325 |
| A | 0.216 | 3.813 | 8.009 | 2.677 | 0.389 | 7.604 | 7.215 |
| Pz | 212.237 | 239.046 | 265.833 | 6.683 | 231.066 | 250.415 | 19.349 |
| Rz | 4440.103 | 4482.874 | 4964.768 | 39.795 | 4446.727 | 4848.224 | 401.496 |
| Register mean/core | 238.911 | 240.199 | 242.323 | 0.458 | 239.413 | 241.717 | 2.304 |
| 其余 Submit mean/core | 4297.259 | 4321.268 | 4572.133 | 20.016 | 4300.027 | 4549.033 | 249.006 |

#### 8.14.3 门禁、AIV miss 跃迁与结论边界

| 分量 | Spearman(G, ·) | P90-P10 | 占 G 波幅 | 判定 |
| --- | ---: | ---: | ---: | --- |
| M | +0.378 | 176.099 us | 43.69% | FAIL |
| X | +0.937 | 271.325 us | 67.32% | PASS：迁移慢尾 |
| A | -0.476 | 7.215 us | 1.79% | FAIL |
| Pz | +0.077 | 19.349 us | 4.80% | **FAIL：Register 不是主载体** |
| Rz | +0.986 | 401.496 us | 99.62% | **PASS：同核其余 Submit 携带变化** |
| Register mean/core | +0.462 | 2.304 us | 0.57% | FAIL |
| 其余 Submit mean/core | +0.720 | 249.006 us | 61.78% | PASS：辅助 core-work 序列 |

按最晚核角色复算，AIC/AIV 的 `Spearman(G,Pz)` 为 `-0.600/+0.571`，波幅只占各自 G 的 9.17%/4.39%；对应 Rz 为 `+1.000/+0.964`。Register 在角色内同样不携带波动。remaining mean/core 的通过说明本组除末核 X 外还存在部分多数核剩余工作联动，但它是辅助序列，不能与 X 相加成墙钟贡献，也不能继续命名为某个未测 phase。

Register core-time 份额中位为 5.267%，范围 5.025%～5.287%；request/call 中位 86.587，范围 86.414～86.709。miss/call 出现一个值得保留的 AIV 状态跃迁：前四轮 ALL 为 1.34～1.46、AIV 为 1.99～2.16，从第 5 轮起变为 ALL 约 1.79、AIV 约 2.66；AIC 始终约 0.047～0.053。该变化逐核 primary=shadow、构建身份和所有门禁闭合，因此不是 capture gap；但它与 G 的相关性只有 `+0.252`，Register phase 时间仍稳定，不能把 I-cache 状态变化包装成本组时延载体。这也说明正式报告必须同时展示同 ELF 的时间、request/miss 和逐核范围，而不能只看单轮 miss rate。

本阶段只排除当前 Register ELF 的 403.028 us 波形由 Register 主导；本组仍由 X 而不是权威 P 中的 M 主导，没有复现 P 的约 1.408 ms 宽波动。`Rz`/remaining 仍是算术剩余，不能改名为 atomic、flag wait 或 I-cache。Pz 与 mean/core 均远离门槛，所以不扩到 20 轮。至此所有存量 selector 的同 ELF 归因结束，下一步先完善报告端逐核 Submit/PMU/Scalar/非 busy 残余时间与构建 provenance，再实现固定容量的 EfDrain-control/依赖就绪观察；都不得扩展逐事件 raw。

### 8.15 真实 PA I-cache 报告的逐核时间闭合

**[观察工具已完成；不改 C++ producer、设备 ABI 或 raw schema]**

Register 正式样本出现了“primary=shadow 全部闭合，但 AIV miss/call 中途跃迁而 phase 时间不动”的真实反例。为了避免后续继续用单轮 miss rate 或不同聚合对象解释性能，本阶段只增强生产 Python consumer 和 HTML，不修改任何 AICore/host C++：

1. ALL/AIC/AIV 新增每核 `submit_elapsed_ticks` 的 mean/min/max，并按 1 ns/tick 显示为 us；它与顶部跨核 `global_submit_span` 明确分开；
2. PMU total、Scalar busy 和非 Scalar-busy 残余同时显示 cycles 与按各角色 1.649844/1.650062/1.649731 cycles/ns 校准的等效时间范围；
3. 残余严格先逐核计算 `total_cycles-scalar_busy` 再汇总，禁止拿来自不同核的 total/scalar 极值相减；
4. 新增逐核 `total_cycles/submit_elapsed_ticks` 的 mean/min/max，只命名为“同 ELF 长窗有效比”，不命名为瞬时频率或利用率；其典型值在最新 Register raw 中约为 ALL 1.649617、AIC 1.649614、AIV 1.649619 cycles/ns，与校准量级一致；
5. raw producer summary、`METRICS`、每核记录和 phase sidecar 全部保持不变，三个 derived summary 只存在于通过门禁后的 `SubmitPmuCapture`/HTML。

逐核 ratio 采用 arithmetic mean of ratios，不使用 `Σtotal/Σelapsed`；卡片纵向增加信息，不给已较宽的 phase/逐核表继续加列，原四张 SVG 也保持不变。全局说明继续强调等效时间不能与 perf-clock、swimlane 或另一个 phase ELF 相减，`total-scalar` 也不是 Scalar 空闲、I-cache stall 或 vector/cube wait。

单测新增专门的错法防线：让 total/scalar 极值落在不同核、构造 `mean(total_i/elapsed_i) != Σtotal/Σelapsed`、拒绝零 Submit elapsed、核对三种角色校准和 raw summary 未扩字段。生产报告 89 项、连同 profile/cache 合计 152 项 PASS，`ruff` 与 `git diff --check` 通过；最新 Register raw `outputs/TestPagedAttentionUnroll_Case1_20260721_103131/fdwic_submit_pmu_raw.json` 也已由增强后的生产 consumer 重新闭合并生成 82,656 B HTML。另从现存 raw 为 none、ArgBuild、Claim、EmptyBracket、Materialize、Register、SubmitTransition 各选最新一份重建 HTML，七份全部 PASS，大小为 79,838～82,812 B。该阶段完全发生在 case 返回后的 host 加工层，对 Submit 热路是零新增指令。

### 8.16 诊断 raw 与实际构建实物的 provenance 绑定

**[观察工具与真实 A5 B1 已闭合；不回写 raw，不给 AICore 增加指令]**

此前正式多轮依赖文档手工冻结 ELF SHA，raw 自身不能证明“这些轮次确实使用同一构建”。本阶段没有把身份字段塞进 C++ producer 或 AICore ELF，也没有在 consumer 校验后回写权威 raw，而是增加固定第三件产物：

```text
fdwic_submit_pmu_raw.json          # C++ producer 原子发布；始终只读
fdwic_submit_pmu_provenance.json   # case 返回后由实际构建路径生成并绑定 raw SHA
fdwic_submit_pmu_report.html       # 同次加载 raw + sidecar 后生成
```

provenance schema 为 `fdwic-submit-pmu-provenance-v1`，至少闭合：

1. 同一次 raw 读取冻结的固定文件名、字节数、SHA256 和 `capture.mode`；
2. 已 profile 化的 callable cache key、16 位 AICore extra cache key、严格匹配 profile 的编译宏；
3. 构建时 `.git_commit` 中的完整 `source-v2:<git-head>:<source-fingerprint>:<definitions-sha256>`，不拿采集后当前 HEAD 冒充构建身份；
4. worker 实际加载的 `build/lib/.../aicore_kernel.o`，以及 `build/cache/.../aicore/` 中的 AIC/AIV combined 和 source-state stamp；两类路径不能混用；
5. final AICore、AIC combined、AIV combined、host runtime 四件实物各自的完整文件 SHA/大小与 literal `.text` SHA/大小。

身份在 Submit-PMU ELF 符号门禁通过后、设备 case 开始前冻结；case 返回后再次从四个实际路径重算，任一文件或 source-state stamp 变化都拒绝发布 sidecar/HTML。scene test 按 `(class, platform, runtime, profile)` 严格查找同一 identity，缺失时在上板前拒绝，不会从 raw 自报字段反推。sidecar 自身再用 raw SHA 绑定；HTML 展示 provenance SHA、source-v2、宏和四件实物。已有无 sidecar 的历史 raw 仍可离线生成旧式报告，但新的正式采集必须自动产生三件套。

sidecar 与 HTML 先在同目录完整暂存，发布前后再次验证 raw 与已接受快照逐字节一致；首个/第二个最终替换失败或末次 raw 检查失败时，均恢复调用前的整对产物，不遗留半套或失配的正式文件。

实现中专门核实了 RuntimeBuilder 的真实目录结构：final ELF 只位于 `build/lib` 的 extra cache 目录；combined 和 `.git_commit` 位于对应的 `build/cache/.../aicore`。使用后一目录的 final 缓存文件冒充 worker 实际加载文件会造成证据错位，因此接口显式接收 output ELF 和 build directory 两条路径。`.text` 读取复用本机正式 `readelf -SW` 对 literal `.text` 的 offset/size，并单独 hash 字节范围，不把 `.rela.text` 当成正文。

纯 host 测试覆盖 raw 字节前后不变、sidecar/HTML 成组发布、四件实物与 `.text`、无 sidecar 历史路径，以及 raw binding、source-state/宏、schema、文件名和冻结后实物变化的 fail-closed；并覆盖首个/第二个最终替换失败、末次 raw 变化和旧产物整对恢复。report/cache 两组共 174 项 PASS，ruff 和 diff 检查通过；另用本机现存 Register 实物验证实际 `readelf` 路径，可得到 final/AIC/AIV/host `.text` 分别为 150,608/68,352/82,256/443,315 B。该工具全部运行在编译完成或 case 返回后，不会进入 Submit 性能窗口。

提交 `15c54b33` 后又依次完成两条真实 A5 B1：

1. `submit-pmu-none` 位于 `outputs/TestPagedAttentionUnroll_CaseB1_20260721_111118/`。96 核均为 5 次 Submit，primary/shadow request 与 miss 逐核完全相等。raw/provenance/HTML 大小为 44,339/3,050/80,808 B，SHA256 分别为 `dbdc10c7d9cc00b79d62f96600a8a91c07470dd402677de9758eb3653572ff57`、`ebeef32f7ef26de78dcdc0d8f49d2711863adc71c28ada00341d0d26951ee235`、`9bea11a9b58e7e8af47381808687c8a5afc168e23c37d270389aa9d38282214f`；
2. `submit-pmu-register` 位于 `outputs/TestPagedAttentionUnroll_CaseB1_20260721_111427/`。96 核均闭合 5 次 Submit 和 5 对 Register begin/end，primary/shadow 逐核完全相等。三件套大小为 69,638/3,103/83,714 B，SHA256 分别为 `be122416742c28ab138db04acf23165d9c73fd90851fd7121eb0c8e303ed2d3a`、`d943be4b76c719b3513b5d030b9eaaf8a223badfb46107b74bbcda016c32db14`、`7e5e40695b529671682dc8d493bd234df2a4870bab361a67e18116be49e85896`。

两轮 source-v2 的 Git head 都精确指向 `15c54b33`；完整窗与 Register 的 profiled key 末项分别为 `submit-pmu-none`、`submit-pmu-register`，AICore extra key 分别为 `aa43623282e2a7db`、`32c26e06ad76d186`。因此同一进程框架下的 profile/cache 隔离已由真实 ELF 和产物旁证，而不是仅由 host mock 证明。两轮 pytest 均为 1 PASS；报告重载与重新渲染逐字节一致。

### 8.17 EfDrain 控制段的固定容量 I-cache 归因

**[观察工具、真实 A5 B1 与首轮 Case1 稳态原因取数均已闭合]**

最新权威泳道中，完整 EfDrain 占 SubmitUnion 的 15.734%，但其中真实 KernelUnion 占 8.019%；直接包围整个 EfDrain 会把 Kernel 执行期间混入 Scalar 归因。扣除嵌套 Kernel 后的最大明确空缺是 EfDrain-control：30,754,207 aggregate core-ticks，占 SubmitUnion 的 7.715%。因此本阶段只实现这一项，不继续增加逐等待或逐事件记录。

四条真实 Submit 入口都在 `drain_block_won()` 前打开 phase，在 `drain_phase_b()` 后关闭。`execute_slot()` 只在 `dist_aicore_call_slot_kernel()` 紧邻前后 pause/resume；Kernel 返回后的 barrier、完成发布、frontier 和 slot 清理仍属于 control。背压与 FinalDrain 没有外层 phase，同一 helper 不会采集它们。

设逐核 Submit 数为 `N`、被排除的 linked-Kernel 调用数为 `K`，三层闭合公式是 `begin_reads=end_reads=N+K`，而报告 per-call 始终以外层 `N` 为分母。`K` 复用 phase record 的 `reserved[0]`，其余保留字保持 0；每核 record 继续是 64 B，总设备容量继续是 12,416 B。为避免污染其他 PMU profile，`K` 的 block-local 状态和清零写入只在 `PTO_FDWIC_SUBMIT_PMU_PHASE_ID=7` 编译；普通泳道、perf-clock、none 和其余 selector 不获得这份状态。

每次排除 Kernel 仍会产生四次 shadow MMIO read、两次 SYS_CNT 和少量 observer bookkeeping。完整 primary/shadow whole 会重建并包含 Kernel，局部 elapsed/request/miss 才排除 Kernel；所以它是同一诊断 ELF 内的控制段方向证据，不是零扰动业务净值，也不能跨 ELF 相减。生产 raw 新增的只是每核既有保留字对应的 `K` 字段与一项 96 核闭合结果，没有扩大设备 ABI 或改为逐事件 raw。

实现阶段已完成 C++ producer/consumer、CLI/profile/cache/provenance、HTML 和专属字段拒绝回归；同时在冻结 provenance 前核验实际 host ELF 的三个 hook 与精确 profile marker，旧 `libhost_runtime.so` 会在上板前直接报出重建要求，不再等到设备 init 返回 0。report/cache 共 209 项 PASS，ruff、clang-format 与 `git diff --check` 通过。

第一次上板在 `outputs/TestPagedAttentionUnroll_CaseB1_20260721_114357/` 于设备执行前失败：AICore override 已重编，但实际 host SO 是不含新 profile marker 的旧缓存，`fdwic_submit_pmu_host_init()` 返回 0，因此没有 raw。该失败没有被包装成设备或 Kernel 问题；补上 host ELF 能力门禁并按 `21e0414c` 重建完整 A5 FDWIC runtime 后，正式 B1 位于 `outputs/TestPagedAttentionUnroll_CaseB1_20260721_115019/`。

正式轮 96 核均为 5 次 Submit，实际 `ΣK=1`：95 核为 0，logical/physical core 9 的 AIC 为 1。全局 begin/end 恰为 481/481，等于 `96×5+1`；逐核也全部满足 `N+K`。phase status、专属 Kernel 排除、primary/shadow、owner Restore、拓扑、数值顺序与风险阈值均为 96/96。完整 B1 Submit 为 279.551 us；局部累计 elapsed/request/miss 为 64,751/56,187/3,414，相对本 ELF 逐核完整 Submit 累计值为 2.7441%/16.3993%/19.4021%。这些比例只证明真实不连续区间能闭合，不用于代替 Case1 稳态归因。

raw/provenance/HTML 大小为 72,951/3,124/84,377 B，SHA256 分别为 `79d6014e892b20823da039e3a2bea6c9761946b6c24444db71d7c61d16caca02`、`24be202086e9fda893012ca999073ceffa160aa9abba12861cee95414006e4d4`、`d8b2b4b77c243ac90e71fbb99084e7f5be7205f5d2a28ee224d4e56b9ed6c892`。provenance Git head 为 `21e0414c35ae7738a89f8994bfaf6870b733dea3`，extra key 为 `88075a1848686623`，host SHA 为 `441e54ac3d997e110de792d6597b8cf47d31a764ccf9bc63551387b9a597b919`；离线重载 raw+sidecar 后的 HTML 与正式文件逐字节一致。

随后从验证提交 `77df3959` 构建并运行首轮真实 Case1：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_115559/
```

golden 与全部门禁通过，完整 Submit 为 4,840.463 us。96 核均为 1,280 Submit；实际 `ΣK=936`，逐核 3～19，AIC/AIV 分别为 429/507。begin/end 精确闭合为 123,816/123,816，即 `96×1280+936`，primary/shadow 仍为 96/96 exact。

| 角色 | control ns/call | 时间占比 | request/call | request 占比 | miss/call | miss 占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 162.654 | 4.5920% | 82.110 | 14.0406% | 2.258 | 17.0455% |
| AIC | 199.792 | 5.7719% | 83.945 | 14.0450% | 0.110 | 28.6828% |
| AIV | 144.085 | 4.0219% | 81.193 | 14.0383% | 3.332 | 16.9316% |

完整 Submit 的 AIC/AIV 每核平均 miss 为 493/25,187.344，miss rate 为 0.06444%/3.40227%；AIV control 每核平均 miss 为 4,264.625。按 90 ns 标尺，后者是 383.816 us/core，却高于该角色实际 control elapsed 的 184.428 us/core；因此不能把标尺当成可相减的 stall。AIC 的 control 时间又高于 AIV，而 miss/call 低两个数量级，本轮同样不支持“I-cache miss 单独主导 EfDrain-control 时间”。下一步应结合既有 atomic 泳道和控制流证据选候选，不能仅按 miss 排名改代码。

Case1 raw/provenance/HTML 大小为 76,269/3,124/86,600 B，SHA256 为 `6d6aa06fbf972f36fbb9260485bb5ee41f5cbb97e9246cb15a398ba2497201ce`、`b4a535ec671f7416945b5205e11268308068759e75ecc324a10ea2a57fb3fa03`、`f5751e86f503273d1b8f8e386301d1d84dfdc49ee6de45d7372199ce90a68841`；provenance Git head 为 `77df395941f86b3b546a6f50d6288fb88acb7078`，离线重渲染逐字节一致。

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

仓库历史中没有可追溯的 `fanin-prefix` 提交或源码痕迹；该未提交过程态已经按要求去除，本文不为它杜撰 hash、数据或收益。

### 9.2 正确性或平台验证的已知缺口

- standalone 16 MiB tiny-ring 的 CPU b256，H1 与临时恢复原实现都在观察窗口内未结束；只能认定模型存在既有长程活性或 host 调度问题，不能把它记为 H1 的 PASS/FAIL；
- MB6 `Normal` 因当前 runtime 没有测试期望的 DEPSIG 而失败；
- MB6 `Heavy` 在 H1 和基线中走同一调用路径 abort，不能归因于 H1，也不能记 PASS；
- `FullCore36` 在 Heavy 基线失败后未继续运行；
- 上述缺口没有通过顺手修改测试契约或生产协议来掩盖。

perf-clock 的 Case2 负测试不属于上述缺口：其每核实际 576 次与期望 320 次不符，host 已按设计拒绝并且没有输出成功 summary。这个结果只覆盖 fail-closed 门禁，不能被改写成 Case2 正确性或性能 PASS。

### 9.3 被证明不适合的观察路线

- external task-based `msprof` 的 raw counter 不受 kernel 内 start/stop 缩窗控制，不能用于 Claim、EfDrain 等局部取数；
- A5 上把 I-cache miss selector `0x35` 复制到 CNT9 时计数恒为 0，因此现行 shadow miss 使用 CNT5，并明确牺牲诊断 ELF 的 MTE3 busy；
- A5 CANN 9.1 没有 `scalar_wait_ib_time` 的正式事件或派生字段；
- 只在运行时把泳道 level 设为 0/1，不能移除 ELF 中的冷诊断代码和布局污染；
- 旧 runtime cache 不理解新增 phase 时曾只生成 Kernel/Alloc 名称，该轮已排除，不能作为有效基线；
- running phase 的 PMU read-clear 会改变布局和时序，不能将局部 ELF 与 `none` 相减得到无扰动净时间。
- perf-clock 不能通过 `PTO2_PROFILING=0` 实现：该宏拥有公开 Arg 布局/ABI，首版尝试在 Arg cacheline `static_assert` 处失败、从未上板，并已撤回。最终实现保留该 ABI，只编译期移除 FDWIC 泳道/atomic 和平台 PMU 路径。

## 10. 本机证据产物索引

### 10.1 重要说明

`outputs/` 已被 Git 忽略。下列路径是本机采集证据索引，不属于源码提交，也不保证 fresh clone、其他 worktree 或后续清理后仍存在。文档中的数值必须与对应历史 commit 和采集配置一起理解，不能因为文件名存在就当作当前 HEAD 结果。

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
| empty-bracket PMU B1 首轮 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_020932/` | 96×5 对、257.430 us；冷启动记录开销闭合 |
| empty-bracket PMU B1 复验 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_021100/` | 96×5 对、303.032 us；冷启动记录开销闭合 |
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
| claim PMU B1 首轮 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_031756/` | 96×5 bracket、82.413 us、四边界/固定 shape 闭合 |
| claim PMU B1 复验 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_031954/` | 96×5 bracket、264.184 us；只作结构与波动样本 |
| claim PMU Case1 首轮 | `outputs/TestPagedAttentionUnroll_Case1_20260721_032101/` | 96×1280 bracket、4994.863 us；ALL 641.816 ns/次 |
| claim PMU Case1 复验 | `outputs/TestPagedAttentionUnroll_Case1_20260721_032244/` | 96×1280 bracket、4704.936 us；ALL 646.708 ns/次 |
| claim 后 materialize B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_032445/` | 96×5、status 0x3f、primary=shadow 96/96 |
| claim 后 arg-build B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_032536/` | 96×5、status 0x3f、primary=shadow 96/96 |
| claim 后 empty B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_032624/` | 96×5、status 0x3f、primary=shadow 96/96 |
| claim 后 none B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_032712/` | 无 phase 字段、primary=shadow 96/96 |
| claim 后 perf-clock B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_032802/` | 96×5 Submit、74.512 us、ELF 隔离 PASS |
| claim 后 level-4 B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_032914/` | 4552 records、87.664 us、dropped=0、排他闭合 PASS |
| register PMU B1 首轮 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_034517/` | 96×5 bracket、80.904 us、三挂点/固定 shape 闭合 |
| register PMU B1 复验 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_034904/` | 96×5 bracket、267.167 us；只作结构与波动样本 |
| register PMU Case1 首轮 | `outputs/TestPagedAttentionUnroll_Case1_20260721_035025/` | 96×1280 bracket、4688.752 us；ALL 187.688 ns/次 |
| register PMU Case1 复验 | `outputs/TestPagedAttentionUnroll_Case1_20260721_035136/` | 96×1280 bracket、5136.513 us；ALL 187.879 ns/次 |
| register 后 claim B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_035425/` | 96×5、status 0x3f、primary=shadow 96/96 |
| register 后 materialize B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_035505/` | 96×5、status 0x3f、primary=shadow 96/96 |
| register 后 arg-build B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_035545/` | 96×5、status 0x3f、primary=shadow 96/96 |
| register 后 empty B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_035625/` | 96×5、status 0x3f、primary=shadow 96/96 |
| register 后 none B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_035703/` | 无 phase 字段、primary=shadow 96/96 |
| register 后 perf-clock B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_035744/` | 96×5 Submit、74.282 us、ELF 隔离 PASS |
| register 后 level-4 B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_035843/` | 4550 records、292.159 us、dropped=0、排他闭合 PASS |
| submit-transition PMU B1 首轮 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_042627/` | 96×4 间隙、384/384、265.223 us；N-1 shape 闭合 |
| submit-transition PMU B1 复验 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_042750/` | 96×4 间隙、384/384、78.873 us；只作结构与波动样本 |
| submit-transition PMU Case1 首轮 | `outputs/TestPagedAttentionUnroll_Case1_20260721_042914/` | 96×1279 间隙、122784/122784、4708.545 us；ALL 354.560 ns/间隙 |
| submit-transition PMU Case1 复验 | `outputs/TestPagedAttentionUnroll_Case1_20260721_043036/` | 96×1279 间隙、122784/122784、4649.434 us；ALL 350.516 ns/间隙 |
| transition 后 register B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_043202/` | 原有 N 次 phase 仍为 96×5、480/480、`0x3f` |
| transition 后 none B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_043241/` | 无 phase 字段、primary=shadow 96/96 |
| transition 后 perf-clock B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_043322/` | 96×5 Submit、312.747 us、ELF 隔离 PASS |
| transition 后 level-4 B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_043553/` | 4548 FDWIC records、90.406 us、dropped=0、排他闭合 PASS |
| 观察代价 B1 预检 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_044523/`、`..._044616/`、`..._044712/` | P/N/S 三类身份与结构门禁闭合；不进入时间统计 |
| 观察代价首组六排列 | `outputs/TestPagedAttentionUnroll_Case1_20260721_044943/` 至 `..._050807/` | 18 个独立进程、每构建 6 样本；方向 3/6，触发扩样 |
| 观察代价反序扩样 | `outputs/TestPagedAttentionUnroll_Case1_20260721_051317/` 至 `..._053148/` | 再增 18 个独立进程；合计每构建 12 样本，差异仍不可分辨 |
| perf-clock 同 ELF 20 轮 | `outputs/TestPagedAttentionUnroll_Case1_20260721_055106/` 至 `..._060447/` | P 的 M 主导、X 次要波动证据 |
| submit-PMU-none 同 ELF 20 轮 | 历史 12 轮加 `outputs/TestPagedAttentionUnroll_Case1_20260721_062430/` 至 `..._062936/` | 仅限 N ELF 的 Scalar/I-cache 波动证据 |
| 最终 K B1 / P B1 回归 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_073602/`、`..._073736/` | K 边界与 P 未污染回归 |
| 最终 perf-clock-kernel Case1 | `outputs/TestPagedAttentionUnroll_Case1_20260721_073849/` | 1013 个窗口内 Kernel；逐核/分组/顶层整数闭合 |
| perf-clock-kernel 预热 | `outputs/TestPagedAttentionUnroll_Case1_20260721_074831/` | 4.480978 ms；明确排除出统计 |
| perf-clock-kernel 同 ELF 20 轮 | `outputs/TestPagedAttentionUnroll_Case1_20260721_075049/` 至 `..._080901/` | K 复现 P 的共同伸缩；波动主要落在 residual |
| EfDrain-control 首次旧 host 失败 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_114357/` | host 缺新 profile marker，init 前拒绝；无 raw，不作性能样本 |
| EfDrain-control 正式 B1 | `outputs/TestPagedAttentionUnroll_CaseB1_20260721_115019/` | 96×5、`ΣK=1`、481/481、primary=shadow、三件套闭合 |
| EfDrain-control Case1 | `outputs/TestPagedAttentionUnroll_Case1_20260721_115559/` | 4,840.463 us、`ΣK=936`、123816/123816、AIC/AIV 稳态原因数据 |

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

1. **不同 ELF 不相减。** `perf-clock`、`swimlane`、不同 selector 的 `submit-pmu` 都会改变代码布局；绝对时间不能机械相减。
2. **不同观察 level 不混用。** level-4 atomic 泳道、level-1 phase 泳道和 `--no-swimlane` 不是同一性能口径。
3. **跨核墙钟不等于累计工作量。** 96 核 span 求和、每核 PMU mean 和最早到最晚 Submit 都是不同量。
4. **PMU cycle 不等于 SYS_CNT tick。** 前者约 1.65 cycles/ns，后者 1 ns/tick。
5. **局部 phase 只能在本 ELF 内解释。** phase 时间、request、miss 只能除以同一进程、同一角色、同一 ELF 的完整区间；不同 phase 不相加，不与 `none` 相减。
6. **Atomic 边界按语义解释。** `return_ready` 不是全系统可见性屏障；`source_issue` 更不能解释为完成延迟。PollBatch 是等待区间，不是单次 atomic。
7. **90 ns/miss 不是墙钟损失。** `miss * 90 ns` 只给单核串行等效数量级；多核、预取、流水和等待会重叠。
8. **`total - scalar_busy` 不是 scalar 空闲。** 它还可能包含 I-cache refill、atomic 等待、Vector/Cube engine 等待和其他非 scalar-busy 周期。
9. **standalone 收益不外推真实 PA。** standalone compete-first 为 -3.9635%，真实 PA 只有波动重叠的 -0.263%，当前真实结论是基本持平。
10. **b1 与 b256 分工不同。** b1 用于结构、边界、调用次数和快速正确性门禁；b256 只在阶段性收口时用于规模和性能结论。
11. **不同 schema 的同名区域不直接相加。** schema-v2/v3 的 Build/Replay lap 与 schema-v4 的排他尾动作定义不同。
12. **观察构建不能冒充净性能。** 带观察的 5.066862、5.278401、5.774295 ms 都只能解释对应观察 ELF。
13. **中位数差不能脱离样本波动。** 三轮区间重叠时应写“基本持平”，不能只取小数点后的正向差宣布收益。
14. **固定次数与动态次数要分开。** Claim、H1 消减等可由拓扑推出；fanin retry、frontier helping 和 barrier poll 必须以动态记录为准。
15. **empty-bracket 不是可直接扣除的常数。** 它的 elapsed 使用外层 SYS_CNT，request/miss 使用 running read-clear 内边界，并且与业务 selector 属于不同 ELF；只能作为观察器自扰动的经验尺度，不能生成数学修正后的业务净值。

## 12. 后续阶段更新与提交模板

每个合理阶段只验证一个主要变量。完成该阶段的源码、正确性门禁、性能取数和本文记录后，再形成一条详细中文提交；不得把多个无法拆因果的候选堆进同一个性能结论。

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

### 12.1 2026-07-21 当时的下一阶段：只推进真实 simpler PA

当前只进入真实 PA 观察基础设施阶段，不继续修改 standalone，也不立即猜测新的业务优化：

1. **[观察工具，已实现] 真实 PA `perf-clock`**：保留 PTO2 公开 Arg ABI，编译期去除 FDWIC 泳道、atomic 观察和平台 PMU；首尾 Submit、96 核逐核调用数、6976 B header、ELF 双向身份、B1、Case1 五轮基线、Case2 fail-closed 和普通 level-4 同源边界均已验证，作为后续候选保留/撤销的权威低扰动 A/B 口径；
2. **[观察工具，已实现] 真实 PA `swimlane`**：普通业务 span 与 atomic 合并采集，已补最终 ELF 正向身份、schema-v4 fail-closed、父子/Kernel/整数闭合门禁，并由当前 HEAD 的 B1 与完整 Case1 上板验证；只用于定位业务区域和 atomic 变化，不与 perf-clock 绝对时间相减；
3. **[观察工具，已实现] 真实 PA `submit-pmu-none`**：编译期去除泳道、atomic 和通用逐 task PMU，每核完整 Submit 期只开关 PMU 一次；96 核 primary/shadow、owner Restore、AIC/AIV raw/HTML、两轮 B1 和一次 Case1 均已闭合；
4. **[观察工具，六个真实 selector 与空 bracket 均已实现] 真实 PA 单阶段 PMU**：`arg-build` 已完成两轮 B1 与一次 Case1；`empty-bracket` 复用同一 12,416 B ABI，已完成两轮 B1 与两轮 Case1，量出 running begin/end 的稳态记录开销并明确外层 elapsed 与 read-clear request/miss 的不同边界；`materialize` 又按最新真实 Case1 最大明确业务 span 建立 mode 4/phase 3，完成两轮 B1、两轮 Case1、五类互斥 B1 回归和最终 AICPU mode 重构回归；`claim` 按同一真实布局的下一明确热点建立 mode 5/phase 4，四条旧/新 Kernel/Alloc 边界、固定 96×1280 shape、两轮 B1、两轮 Case1 和六类互斥 B1 回归均已闭合；`register` 又建立 mode 6/phase 5，在三个真实 RegisterOutputs 调用点固定采集 96×1280 shape，完成两轮 B1、两轮 Case1 和七类互斥 B1 回归；`submit-transition` 最后建立 mode 7/phase 6，复用统一 Submit hook，以每核 `N-1` 独立 shape 聚合相邻 Submit 间隙，两轮 B1、两轮 Case1 和四类互斥构建回归均已闭合；`efdrain-control` 最后建立 mode 8/phase 7，用四条 Submit 外层边界和 Kernel 前后 pause/resume 聚合排除 linked Kernel 的不连续 Scalar 控制片段；真实 A5 B1 已闭合 `ΣK=1`、481/481 read、96 核状态与 provenance 三件套，Case1 又闭合 `ΣK=936`、123816/123816 read 和首轮 AIC/AIV 时间/request/miss 原因数据。六个 selector 都复用 12,416 B ABI，没有增加逐调用记录；至此停止继续增加 phase，也不依赖新的 standalone 实现；
5. 结构和边界迭代先使用最小有效真实 PA 用例完成正确性门禁；只有构建身份、容量、业务边界和统计闭合后，才运行完整 Case1 b256 性能样本，避免反复生成数百 MiB profiling 文件；
6. **[观察校准，已完成但不可分辨] 三构建交错量化**：在冻结的 `d1572c33` 上先做三类 B1 预检，再执行正序和反序各六个 Case1 排列块；P/S/N 各 12 个样本全部闭合。S-P 与 N-P 都只有 7/12 同方向，且中位差绝对值小于 `2×MAD`，因此没有把跨 ELF 差值包装成观察指令成本，也不继续扩样强求单值；
7. **[波动定位，已完成] 同一 perf-clock ELF 的 20 轮独立 Case1**：20/20 raw 均按 96×1280 和整数时钟严格闭合。完整 Submit 的中位数为 4898.184 us，P90-P10 跨度为 1407.736 us；M 通过预定相关性和幅度双门禁，A 与起跑偏斜均不通过，X 只作为次要迁移慢尾。96 核中 95 核的 body 与 G 的 Spearman 不小于 0.7，AIC/AIV 中位数联动，且不存在固定最慢核或时间单调漂移；
8. **[PMU 波动定位，已完成且限制为 N ELF] 同一 submit-pmu-none ELF 的 20 轮**：复用历史 12 轮并在一次排除预热后补 8 轮连续 N-only，20/20 通过生产消费者全量闭合。N 的 Scalar-busy 是其自身每轮 ALL mean/core 波动的主要观察载体，非 Scalar-busy 残余幅度很小，全窗平均有效 cycle/time 比例只有 21.35 ppm 的 P90-P10，I-cache miss 次数不随慢轮同向；但 N 的 A/M/X 由 X 通过门禁，未复现 P 的 M 主导形态，因此没有把 N 的 Scalar/I-cache 结论外推成 P 的根因；
9. **[低容量 Kernel 聚合，结构闭合已完成] 独立 perf-clock-kernel 变体**：保持 6,976 B 固定 header 和每核 64 B，只在真实 linked-kernel 调用前后读取 SYS_CNT，并用逐核首末 Submit 状态排除 FinalDrain。K B1、P B1 和 K Case1 均通过 golden 与 host/Python 双层产物门禁；Case1 窗口内记录 1013 个 Kernel，合法总范围为 `[256,1024]`，逐核 Kernel 与 residual 整数闭合。该单轮只证明工具可用，尚未证明 K 复现 P 的波动形态；
10. **[K 波动定位，已完成] 同一 perf-clock-kernel ELF 的 20 轮独立 Case1**：20/20 生产消费者闭合，K 的 M 以 `rho=+0.946`、71.2% 幅度复现 P 的多数核共同伸缩；但 K 的 X 也通过双门禁而 P 的 X 不通过，所以只能称部分复现。K 内部 residual mean/core 以 `rho=+0.941`、72.1% 幅度通过，Kernel mean/core 虽相关但幅度只有 G 的 0.65%，calls 仅 998～1011 且与 G 负相关。由此只在 K 内部排除 Kernel 总数量或 mean/core 变化幅度主导，不能外推 P 根因；residual 也还不能未经细分就命名为 Scalar、atomic、I-cache、flag 等待或频率问题。
11. **[环境取证，已完成] 持续真实 PA 负载 power probe**：在已知 root A5 评测退出后，用日志初始化标志对齐 20 轮真实 PA 和 3 s `asys --sys-lp`。die 0/1 各 291 个点均为 hard/soft 1650 MHz，POWERBRAKE 与三级 IWARNING 均为 0；只排除 10.24 ms 粒度的持续 DVFS/EDP，不外推到一个 Submit 内的更短瞬态。第一次错误等待 `/dev/davinci0` 的错位样本已明确作废。
12. **[Materialize 波动归因，已完成] 同一诊断 ELF 的 12 轮独立 Case1**：12/12 生产消费者和 HTML 闭合，最晚核 Materialize `Pz` 的 `rho=-0.343`、波幅只占 G 的 9.44%，不通过；同核其余 Submit `Rz` 的 `rho=+0.986`、波幅 100.47%，通过。Materialize mean/core 的 P90-P10 仅 0.994 us。本组由 X 慢尾而不是 M 共同伸缩通过门禁，因此只排除 Materialize 为该 ELF 的波动主载体，不外推 P 根因。
13. **[Claim 波动归因，已完成] 同一诊断 ELF 的 12 轮独立 Case1**：12/12 闭合；最晚核 Claim 因 AIC/AIV 角色切换形成 370～1,025 us 的大范围，但与 G 的 `rho=-0.189`，相关性失败；同核其余 Submit 的 `rho=+0.790` 并通过。Claim mean/core 波幅只占 G 的 7.29%，request/miss 也稳定。本组仍由 X 慢尾通过，因此不把 ClaimMax/return-ready atomic 包装成权威 P 的波动根因。
14. **[SubmitTransition 波动归因，已完成] 同一诊断 ELF 的 12 轮独立 Case1**：每核 1,279 个相邻间隙全部闭合。最晚核 Pz 的整体 `rho=-0.531` 受 AIC/AIV 角色混合影响，角色内仅 `+0.371/+0.257`；Transition mean/core 波幅只占 G 的 0.74%。同核其余 Submit 继续通过，因此相邻 Submit 间隙也不是本 ELF 的主载体。
15. **[ArgBuild 波动归因，已完成] 同一诊断 ELF 的 12 轮独立 Case1**：12/12 生产消费者、HTML、每核 1,280 个边界和 primary/shadow 全部闭合。AIC/AIV 的 Pz 形成约 199/276～298 us 角色双峰，但整体 `rho=-0.259`、波幅只占 G 的 29.94%，角色内也不通过；ArgBuild mean/core 波幅仅占 0.76%。同核非 ArgBuild 的 Rz 与迁移慢尾 X 通过。本组没有复现 P 的多数核共同伸缩，因此只排除当前 ArgBuild ELF 的实际波形由该 phase 主导。
16. **[Register 波动归因，已完成] 同一诊断 ELF 的 12 轮独立 Case1**：每个 Submit 只命中三个互斥挂点之一，1,152 条逐核记录和 1,474,560 次调用体全部闭合。Register Pz 的 `rho=+0.077`、波幅只占 G 的 4.80%，mean/core 波幅仅 0.57%；同核 Rz、迁移慢尾 X 和辅助 remaining mean/core 通过。AIV miss/call 在第 5 轮发生闭合但与 G 弱相关的状态跃迁，进一步证明单轮 miss 不能替代同 ELF 时间关联。结论仍不外推到权威 P 的宽波动。
17. **[I-cache 报告加工，已完成] 不改 raw/ABI 的逐核时间闭合**：从受信 records 派生 Submit SYS_CNT、PMU total、Scalar busy、逐核非 busy 残余和 PMU/SYS 长窗有效比的 mean/min/max；角色校准只生成当前 ELF 等效时间。测试专门拒绝 difference-of-extrema、ratio-of-sums 和零分母，相关 152 项及 ruff 通过。
18. **[构建 provenance，真实 A5 B1 已闭合] raw 保持只读**：新增 raw SHA 绑定的独立 sidecar，记录构建时 source-v2、profile/cache/宏，以及实际 final、AIC/AIV combined 和 host 的 whole/.text 身份；构建后与 case 后双重实物检查。缺 identity、绑定或文件变化均 fail-closed；成组发布失败会恢复旧产物。相关 174 项通过；`submit-pmu-none` 与 `submit-pmu-register` 各一轮真实 A5 B1 三件套均闭合，且使用不同 profiled/extra cache key。
19. **[EfDrain-control，B1 与首轮 Case1 已闭合]**：固定容量排除真实 linked Kernel，B1 实际 `ΣK=1`，Case1 实际 `ΣK=936`；后者完整 Submit 为 4,840.463 us，AIC/AIV control 分别为 199.792/144.085 ns/call、0.110/3.332 miss/call。AIC 时间更长但 miss 低两个数量级，因此当前证据不支持单独按 I-cache miss 选择生产优化。

下一步不再扩充 N、K 或已经完成归因的既有 selector，也不跨 ELF 扣减。独立构建 provenance 与 EfDrain-control 的 B1/Case1 已完成。不得恢复逐 Submit/逐等待的大 raw，也不在此基础上继续堆叠依赖就绪 selector。下一阶段只从现有泳道 atomic、排他 span、全窗 PMU 与六个 selector 的交叉证据中选择一个真实 Scalar 候选，再由 perf-clock 交错 A/B 决定保留或撤销。单阶段 observed 不机械扣除 empty，也不从单轮 I-cache 数直接提出生产优化；standalone 的绝对数继续不替代真实 PA 当前结果。

### 2026-07-21 / P1：BlockWon 慢路冷外提

状态：**[已撤回]**

#### 目标与观察前提

本阶段先复核现有 I-cache 观察覆盖，再验证一个不改变 BlockWon 协议的代码布局候选。最新真实 PA 排他泳道 `outputs/TestPagedAttentionUnroll_Case1_20260721_053003/` 中，五个 Submit 内业务 selector 已覆盖 SubmitUnion 的 75.4379%；扣除 EfDrain 内真实 Kernel 后，约覆盖 82.02% 的 Scalar/control core-work。尚未直接覆盖的最大纯 Scalar 区域是 PrepareMap（6.1493%）。当前完整 Submit、六个业务 selector、empty-bracket、primary/shadow、96 核 AIC/AIV raw、构建 provenance 和 HTML 已闭合，因而不为本候选继续增加设备字段或 selector；PrepareMap 只在后续候选明确落入该区时按需补齐。

Case1 是单 lane 图，权威泳道中 `DrainWon=0`，但 `drain_block_won()` 仍从 EfDrain、kernel loser replay 和背压路径被高频调用。候选只把 `g_fdwic_joint_submit_seen` 门闩保留在 inline wrapper，把原 BlockWon 扫描原样放入 `noinline` 冷函数；不删除 joint 路径，不改变 atomic、slot、fanin、完成发布或 FinalDrain 语义。

#### 正确性与布局证据

候选真实 A5 B1 位于 `outputs/TestPagedAttentionUnroll_CaseB1_20260721_121714/`，golden 通过，96 核均为 5 次 Submit，完整 Submit 为 80.746 us。该轮只作结构门禁，不进入 Case1 性能统计。

同一 perf-clock 构建下，候选确实形成独立 `drain_block_won_joint_slow()` 符号，布局变化如下：

| 对象 | 基线 | 候选 | 变化 |
| --- | ---: | ---: | ---: |
| AIC combined `.text` | 59,288 B | 52,960 B | -6,328 B |
| AIV combined `.text` | 73,296 B | 66,896 B | -6,400 B |
| AIC combined whole | 1,773,136 B | 1,659,096 B | -114,040 B |
| AIV combined whole | 2,032,584 B | 1,919,512 B | -113,072 B |
| final `aicore_kernel.o` | 2,469,600 B | 2,358,784 B | -110,816 B |

这只证明冷外提减少了热调用方的重复代码，不能自动推出 I-cache 或墙钟收益。

#### perf-clock 五对交错 A/B

基线 A 与候选 B 均使用独立 pytest 进程、`Case1 --fdwic-profile perf-clock --rounds 1 --skip-golden`。候选预热 `outputs/TestPagedAttentionUnroll_Case1_20260721_121858/` 明确排除。正式五对为：

| 对次 | A 基线产物 | A/us | B 候选产物 | B/us | B-A/us |
| ---: | --- | ---: | --- | ---: | ---: |
| 1 | `..._122032` | 4,883.659 | `..._122151` | 4,460.575 | -423.084 |
| 2 | `..._122319` | 4,886.473 | `..._122439` | 5,299.185 | +412.712 |
| 3 | `..._122611` | 4,488.058 | `..._122732` | 4,489.416 | +1.358 |
| 4 | `..._122905` | 4,553.267 | `..._123040` | 5,117.292 | +564.025 |
| 5 | `..._123229` | 6,139.517 | `..._123345` | 5,377.480 | -762.037 |

A/B 组中位数分别为 4,883.659/5,117.292 us；逐对差值中位数为 **+1.358 us（+0.0278%）**，候选 2 对更快、3 对更慢。布局缩小没有稳定兑现成完整 Submit 收益，结果仍被已知的多数核共同伸缩波动覆盖。

#### 决定

候选业务代码完整撤回，不以“ELF 更小”替代 perf-clock 结论，也不为证明预期收益继续扩样。保留本节负结果，避免后续重复进行相同的 BlockWon 冷外提。下一候选仍只改一个已由源码证明的 Scalar 冗余，并继续由完整 Submit perf-clock A/B 决定保留或撤销。

### 2026-07-21 / O8：补齐观察链离线组合回归

状态：**[观察工具，离线闭合]**

本阶段遵守“暂停上板”的要求，没有启动 A5，也没有修改 device/host 生产代码、PMU ABI 或 raw 字段。目标是复核当前三条证据链的现有正式产物，并补上仅靠单模块合成 raw 无法覆盖的组合门禁。

#### 正式产物复核

- `outputs/TestPagedAttentionUnroll_Case1_20260721_053003/` 的 schema-v4 泳道为 96 核、每核 1,280 个 Submit、945,264 个事件且 `dropped=0`；atomic 物理记录 105,963 条、PollBatch 337 条、折叠调用 4,481 次，逻辑调用严格闭合为 `105963 - 337 + 4481 = 110107`。当前 analyzer 重算的拓扑、父区间、Kernel containment、排他分区和整数 cycle closure 全部通过。
- Materialize、Claim、SubmitTransition、ArgBuild、Register 和 EfDrain-control 六份正式 phase raw 均通过当前严格 reader：96 个唯一物理核、32 AIC + 64 AIV、每核 1,280 个 Submit、owner 配置/恢复、primary/shadow、selector/status、风险阈值和 phase 时间/次数全部闭合；Transition 为 1,279 次/核，其他普通 phase 为 1,280 次/核，EfDrain-control 为 `N+K`，其中 `ΣK=936`、全局 begin/end 均为 123,816。
- 历史目录中只有 Register `..._103131` 和 EfDrain-control `..._115559` 具有 provenance sidecar；Materialize `..._092508`、Claim `..._094203`、Transition `..._095712` 和 ArgBuild `..._101249` 只能证明 raw 数据闭合，不能追溯当时实际 ELF/编译宏。`..._053003` 泳道同样没有构建 sidecar。该限制如实保留，不使用后来的二进制身份反向填充历史产物。

#### 新增离线门禁

1. `test_fdwic_submit_pmu_report.py` 对 phase 1～7 参数化执行完整 raw→build identity→provenance→HTML 发布链，逐项核对 capture mode、phase ID、三个编译宏、raw 不变性、provenance SHA 和 HTML 身份展示。这样可以离线拒绝 profile/phase/cache 身份串线，不再只由 none 和 EfDrain 两个特例间接代表其余 selector。
2. `test_fdwic_swimlane_converter.py` 在同一 schema-v4 业务 raw 中加入 ClaimMax `return_ready` direct 与 7 次 Fanin poll-batch。测试确认两类 atomic 与 Claim/Fanin 同处对应 Scalar lane，2 条物理记录加权为 8 次调用；atomic 保持非加和 overlay，加入前后的 aggregate core-work、residual breakdown 和 Submit 整数分区完全不变。

离线验证结果：

```text
tests/ut/py/test_fdwic_submit_pmu_report.py
tests/ut/py/test_fdwic_swimlane_converter.py
tests/ut/py/test_scene_test_cache.py
    288 passed

ruff check
    All checks passed

git diff --check
    PASS
```

#### 下一单变量候选

源码与现有泳道交叉审计后，下一候选收敛为 PrepareMap 的空 task-head 快路，而不再继续 ActiveMask 复用或重复增加 `drain_phase_b()` caller gate。Case1 每核 1,280 个 Submit、`H=64`，会清退 1,215 个历史 task id；五任务序列中只有 UP 的四个 INOUT 建立 map 链，因而每核可静态得到 243 个非空 head 和 972 个空 head。若只在 `cur == -1` 时跳过原本再次写入 `-1` 的 GM store，96 核理论上删除 93,312 次冗余写，且影响严格落在 PrepareMap（现有 aggregate 24,512,711 ticks、占 SubmitUnion 6.1493%）。截至 O8，该候选尚未写入生产源码、没有性能结论；其后续实现、离线门禁和真实 A5 待办由 P2 单独记录。

### 2026-07-21 / P2：PrepareMap 空 task-head 快路

状态：**[验证中：离线门禁完成，待真实 A5 裁决]**

#### 候选边界与机会量

真实实现位于 `src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/tensor_map.h` 的 `dist_tensor_map_advance_retire()`。原逻辑读取每个已退休 task 对应的 `task_heads[slot]`；即使值已经是空链哨兵 `-1`，也会在不进入释放循环后再次向同一 GM 地址写入 `-1`。

提交 `2dc49a13` 只增加以下精确快路：

```cpp
int32_t cur = self.task_heads[id & kTaskWindowMask];
if (cur == -1) continue;
```

这里不能写成 `cur < 0`：原实现会把小于 `-1` 的异常负值归一为 `-1`，精确比较才保持该防御行为。`continue` 只结束当前 id；循环后的 `cleaned_upto` 和 `alive_floor` 仍统一推进，非空 task 链、bucket 双向链和 free-list 完全走原路径。每个 worker 独占自己的 `DistCore::map`，真实路径也没有第二个执行流依赖“把已经是 `-1` 的普通字段再写一次”这个物理写事件。

Case1 每核 1,280 次 Submit，`H=64`，实际清退 task id `0..1214`。五任务序列中只有 UP 的四个 INOUT 建立 map 链，因此清退范围内有 243 个非空 head、972 个空 head；96 核静态机会量为 **93,312 次冗余 GM store**。该数字只证明候选有真实命中机会，不等于已经取得 93,312 次写延迟之和，更不是墙钟收益。

#### 直接生产语义单测

提交 `970e4fad` 新增 no-hardware C++ 测试，直接包含生产 `tensor_map.h`，没有复用 standalone 或旧 ring-per-bucket 模型。测试将生产 retire 与优化前控制流做完整状态差分，覆盖：

- 正常 `-1` 空 head 与异常 `-2` 归一化；
- 同 task 多 entry、bucket 头/中间摘链和 free-list 顺序；
- `kTaskWindow` 槽复用、重复 floor 和回退 floor；
- 所有未触及 entry/相邻字段的逐字节一致性。测试使用显式 `memcpy` 克隆并要求 `DistTensorMap` 可平凡复制，避免结构体 padding 导致跨编译器假失败。

本用户 GCC 15 的验证结果为：

```text
test_fdwic_swimlane_poll_batch
test_fdwic_tensor_map_retire
    2/2 passed
```

其中 CaseB1 不能替代这项门禁：它每核只有 5 次 Submit，而 `H=64`，不会进入 retire 循环。后续真实 A5 的 CaseB1 只可作为构建冒烟；候选正确性必须至少包含一次不跳过 golden 的 Case1。

#### 真实 CCEC 代码生成

基线和候选均通过真实 PA 的 `TestPagedAttentionUnroll.compile_chip_callable("a5")`、`perf-clock` 编译门构建；该入口只编译 orchestration、四个子 kernel 和 FDWIC AICore override，没有创建 Worker、调用 ACL 或连接设备。基线绑定提交 `970e4fad`，候选使用同一提交加单文件工作树差异；source-v2 指纹分别为：

```text
baseline  2732d4ad0ee67e920d24cc5c66d6a300c86fc9904c7ba65dc53168e8694f841d
candidate fae3c47f0bbf7562b3b1cbd264ef80b2df2c7f6750580327a64fa61a30f9c751
```

使用 CANN 9.1 `dav_3510` PEM decoder 检查 AIC/AIV 合并对象中的四类真实内联入口，八处结果全部闭合：

| 入口 | AIC 函数体 | AIV 函数体 | 控制流结论 |
| --- | ---: | ---: | --- |
| `dist_submit_impl` | 5,656→5,664 B | 5,660→5,668 B | `-1` 跳到 id 递增；其他负值跳到 reset store |
| `dist_alloc_tensors` | 5,344→5,352 B | 5,404→5,412 B | 同上 |
| `dist_submit_compete_first_finish` | 2,572→2,580 B | 2,608→2,620 B | 同上 |
| `dist_alloc_compete_first_finish` | 3,008→3,016 B | 3,024→3,044 B | 同上 |

基线中负值分支先落到 id 递增，随后仍顺序执行 reset store；候选把 reset store 移到递增之前，并增加精确 sentinel 比较：相等分支直接落到递增，非 `-1` 的负值分支仍落到 store。由此可以确认编译器没有把源码快路重新折叠回旧行为。

这项改动也有可见代码布局代价：AIC combined `.text` 增加 32 B；AIV combined 和最终 ELF 的 `.text` 因对齐保持不变，但各内联函数体仍有上述增加。布局变化只证明机器码形态，不能替代 perf-clock 性能结论。

#### A5Sim Case1

候选使用本用户 `.venv`、GCC 15 和当前 PTO-ISA 跑通真实 simpler PA 的 A5Sim Case1，完整执行 256 batch 和 golden：

```text
TestPagedAttentionUnroll::test_run
    1 passed in 81.02s
```

该结果证明模拟路径的 1,280 次 Submit、retire 和 TensorMap 最终输出没有回归；它不是 A5 性能数据。

#### 待完成的真实 A5 裁决

当前按要求停在上板之前，不把离线结果写成“已保留”。恢复上板后的固定顺序为：

1. Case1 不带 `--skip-golden`，闭合真实设备正确性；B1 仅作构建冒烟；
2. 同一环境交错运行基线 A / 候选 B 的 `perf-clock` Case1，由完整 Submit 墙钟决定保留或撤回；
3. 仅当 perf-clock 有稳定收益时，再用合并泳道确认变化落在 PrepareMap，业务 span、Submit 数和 atomic 次数不变；
4. 只有仍需解释 I-cache 方向时，才增加 PrepareMap 单阶段 PMU，不预先扩张观察面。

### 2026-07-22 / O9：纯 scalar-code Submit-PMU ABI/schema v2 口径重建

状态：**[观察工具：离线、CCEC 与首轮真实 A5 v2 门禁已闭合]**

#### 旧数据结论暂停

新口径要回答的是 scalar 代码本身的开销：Submit 时间分母既不含 linked Kernel，也不含已发出 atomic 后等待返回值可用的时间。旧采集和旧 scalar I-cache HTML 使用的分母与 v2 不同，因此暂停引用其绝对结论和阶段比例，不与 v2 混用。

linked Kernel 继续由调用前的 `metrics_prof_stop` 和返回后的 `metrics_prof_start` 排除，对应的 scalar SYS 时间段也在边界处闭合和重开；所以 linked Cube/Vector Kernel 既不进 PMU counter，也不进 scalar 时间分母。

#### atomic 分类与扣时方式

对 28 个 atomic site 做穷举分类：16 个使用返回值的 return-ready site 进入扣时，12 个只发出操作的 source-issue site 保留在 scalar 开销内。return-ready 边界是 atomic 前的 SYS_CNT 到结果数据依赖落实后的 SYS_CNT；v2 只从完整 Submit 和所属 active phase 的 scalar 时间中扣除这段 SYS bracket。

不在每个 atomic 周围 stop/start PMU，因为那会向每次采样注入 PMU 控制开销并扰动 PIPE_ALL；也不加 DSB，因为本地结果数据依赖已经给出 return-ready 边界，额外屏障会改写被观测路径。因此 PMU counter 仍然包含 atomic 指令及其观察钩子，v2 的“纯 scalar-code”仅指时间分母口径，不表示 counter 已排除 atomic。

#### 门禁与当前证据

核心记录 ABI 保持 64 B，没有增长；raw JSON 因必选字段、状态位和时间语义都已不兼容旧文件，同步升级为 `fdwic-submit-pmu-v2`，不伪装成旧 schema。bit 17 是 return-ready atomic 扣时有效性的 fail-closed 门禁，状态异常或 bracket 未闭合时不允许生成正式产物。当前已通过 263 项 Python 离线用例和 2 项 C++ 用例，并闭合 phase 0/10 在 AIC/AIV 下的 CCEC 编译；同一 CCEC 优化 IR 还证明 atomic 返回值进入 `MOV -> SYS_CNT` 的显式数据依赖。CANN 当前的 `llvm-objdump` 不能解码 HIIPU，因此这里不把 IR 证据误称为机器指令反汇编。这些都是离线正确性与代码生成证据，不是性能结论。

#### 首轮真实 A5 闭合

已产生四个 v2 产物：

- none B1：`outputs/TestPagedAttentionUnroll_CaseB1_20260722_211055`
- Winner B1：`outputs/TestPagedAttentionUnroll_CaseB1_20260722_211158`
- none Case1：`outputs/TestPagedAttentionUnroll_Case1_20260722_211333`
- Winner Case1：`outputs/TestPagedAttentionUnroll_Case1_20260722_211624`

四轮的 trusted record、linked Kernel gate 闭合、return-ready atomic 扣时有效、vector busy 为零和 cube busy 为零等正式门禁均为 `96/96`。Winner Case1 的业务 phase calls 为 `1024 = 512 AIC + 512 AIV`，排除的 53 次 linked Kernel 调用全部位于 AIC。在该 Winner Case1 的同一 ELF 内，phase scalar-code core-time 占完整 Submit 分母的比例为 ALL/AIC/AIV `3.47% / 7.93% / 1.42%`。

none Case1 的 `4.982069 ms` 与 Winner Case1 的 `4.599385 ms` 来自不同 ELF，不能相减或解读为优化收益，本轮不写收益结论。旧 v1 下的其余 profile 和 scalar I-cache HTML 仍需按 v2 口径全量重跑；在此之前不恢复旧数据结论。

#### v2 既有 profile 全量重采

在已记录的 none 和 Winner Case1 之外，补齐了下列 9 个 Case1 v2 产物：

- arg：`outputs/TestPagedAttentionUnroll_Case1_20260722_212608`
- empty：`outputs/TestPagedAttentionUnroll_Case1_20260722_212653`
- materialize：`outputs/TestPagedAttentionUnroll_Case1_20260722_212739`
- claim：`outputs/TestPagedAttentionUnroll_Case1_20260722_212857`
- register：`outputs/TestPagedAttentionUnroll_Case1_20260722_212943`
- transition：`outputs/TestPagedAttentionUnroll_Case1_20260722_213030`
- efdrain：`outputs/TestPagedAttentionUnroll_Case1_20260722_213152`
- prepare：`outputs/TestPagedAttentionUnroll_Case1_20260722_213240`
- fanin：`outputs/TestPagedAttentionUnroll_Case1_20260722_213326`

上述 9 种与 none、Winner 合计 11 种既有 Case1 profile。使用报告链本身的 `load_capture` + `load_provenance` 做机器化复验，`11/11` 全部通过；每轮 trusted record、linked Kernel gate 闭合和 return-ready atomic 扣时有效均为 `96/96`。`11/11` 的 HTML 均存在，其 I-cache 逐核摘要只显示最小值/最大值。

EfDrain 轮闭合了 962 次 linked Kernel 排除；Fanin 轮的动态业务 calls 为 1,024，全局 call-count 门禁闭合。每个 profile 的 phase 百分比都只属于自己的独立 ELF 及其本轮 Submit 分母，不在跨 ELF 之间求和；empty 仅用作 bracket 标定，不当作业务 phase。至此旧 v1 数据不再用于当前结论，当前口径只认 v2 全量重采产物。

### 2026-07-22 / O10：AllocComplete v2 阶段闭合

状态：**[观察工具：284 项 UT 与真实 A5 B1/Case1 v2 门禁已闭合]**

#### 真实业务边界

AllocComplete 只在真实 Alloc winner 上开窗。legacy/one-shot 路径从 `Claim.end` 开始，compete-first 路径因 Claim 已在 Begin 中完成，从 Finish 内的 `Register.end` 开始；两者都在 `dist_submit_complete_alloc()` 返回后立即闭合，与 schema-v4 泳道的完整 AllocComplete 尾动作一致。

该阶段使用 `dynamic_global`：协议只能确定全局调用数为 `B = expected_submits / 5`，Alloc winner 由多核竞争决定，不伪造 AIC/AIV 角色公式或逐核固定次数。B1 的唯一一次调用实际落在 AIV，Case1 也出现 1 次 AIV winner，直接证明角色必须保持动态。

AllocComplete 内的 HeapGuard 慢路可能回收并执行 linked Kernel，因此继续复用成对 pause/resume 把这些 Kernel 从阶段时间和 counter 中排除，不把本轮观测到的零次写成协议不可能。阶段时间同时复用 O9 的 return-ready atomic SYS bracket 扣时链；PMU counter 仍含 atomic 指令事件。新模式只复用现有 phase 字段和状态门禁，核心记录与 phase sidecar 仍各为 64 B，ABI 不增长。

#### 离线与真实 A5 证据

增量后的离线回归共 284 项 UT 全部通过。真实 A5 产物为：

- B1：`outputs/TestPagedAttentionUnroll_CaseB1_20260722_215826`
- Case1：`outputs/TestPagedAttentionUnroll_Case1_20260722_220027`

两轮都通过 v2 严格校验和 `96/96` trusted/kernel/return-ready atomic 门禁，全局动态 call-count 也闭合。Case1 的调用数为 `256 = 255 AIC + 1 AIV`，63 个核为零调用，排除的 linked Kernel 为 0 次；阶段 scalar-code 时间为 584,689 ticks，I-cache request 为 170,344，miss 为 8,790。同一 ELF 内，它占完整 Submit scalar-code core-time 分母的 `584,689 / 296,983,746 = 0.1968758%`；该轮全局 Submit 墙钟为 `4.648447 ms`。

这个百分比只能在 Case1 `20260722_220027` 的同一 ELF 内解释。AllocComplete 与 none、Winner 及其他 phase 均是独立 ELF，不对其百分比求和，也不用 `4.648447 ms` 与他轮墙钟相减或写成优化收益。

### 2026-07-22 / O11：LoserReplay v2 阶段闭合

状态：**[观察工具：305 项 Python UT、2 项 C++ 用例与真实 A5 B1/Case1 v2 门禁已闭合]**

#### 源码、泳道与动态 shape

LoserReplay 只存在于真实 Kernel loser 分支：它与 schema-v4 泳道共用 `Register.end` 起点，在 `drain_block_won()` 返回后立即闭合，对应报告边界 `register_end_to_drain_block_won_return`。Kernel winner 进入 WinnerBuild，Alloc winner 进入 AllocComplete，Alloc loser 只留在 Submit residual，都不生成 LoserReplay。

当前 PA 每个 batch 有 4 个 Kernel task，每个 task 由 96 核各回放一次并且全局只有 1 个 winner；其中每 `B` 的 Kernel winner 固定为 `2B AIC + 2B AIV`。因此 LoserReplay 的精确公式是：

- ALL：`(4 × 96 - 4)B = 380B`
- AIC：`(4 × 32 - 2)B = 126B`
- AIV：`(4 × 64 - 2)B = 254B`

这是全局和角色总数闭合，不伪造逐核固定次数；单核实际调用数为 `4B - 该核赢得的 Kernel task 数`。B1 闭合 `380 = 126 AIC + 254 AIV`，Case1 闭合 `97,280 = 32,256 AIC + 65,024 AIV`，与先前 schema-v4 泳道原始记录一致。

#### atomic、Kernel 与 ABI 语义

`drain_block_won()` 只做 BlockWon 状态轮询、lane claim 和本地 ring slot 构造，不调用 linked Kernel；取到待回放数据时也只由 `build_ring_slot()` 物化 slot，不在 LoserReplay 内执行该 slot 中的 linked Kernel。因此本阶段的 excluded Kernel 闭合值应为 0，而不是把业务 Kernel 时间算进 scalar control。

函数内消费 atomic 返回值的等待继续复用 O9 的 return-ready SYS bracket 从阶段时间中扣除；不消费返回值的 source-issue atomic 仍属于 scalar 发出开销，保留在时间分母内。PMU counter 仍包含两类 atomic 指令事件。新 phase 只复用已有累加器、状态位和 sidecar 字段，核心记录与 phase sidecar 仍各为 64 B，ABI 不增长。

#### 离线与真实 A5 证据

离线回归通过 305 项 Python UT 和 2 项 C++ 用例。真实 A5 产物为：

- B1：`outputs/TestPagedAttentionUnroll_CaseB1_20260722_222358`
- Case1：`outputs/TestPagedAttentionUnroll_Case1_20260722_222533`

两轮的 v2 严格校验、`96/96` trusted/linked-kernel/return-ready atomic 门禁和动态全局/角色 call-count 均闭合。B1 的逐核调用数为 `3–4`，Case1 的调用数为 `97,280 = 32,256 AIC + 65,024 AIV`，逐核 `1,005–1,020`（AIC `1,005–1,013`，AIV `1,014–1,020`），excluded Kernel 为 0。Case1 的阶段 scalar-code 时间为 5,021,979 ticks，I-cache request 为 8,218,072，miss 为 453,234；在同一 ELF 内，它占完整 Submit scalar-code core-time 分母的 `5,021,979 / 394,817,462 = 1.2719749%`。

Case1 执行了 97,280 对 begin/end observer，每对都含 shadow counter read-clear 和边界 bookkeeping，因此这是高频观察 ELF，不是无扰动业务 ELF。局部 SYS 时间边界故意位于 begin 读取之后、end 读取之前，但观察器仍会改变整体指令流、I-cache 和墙钟。该轮全局 Submit 墙钟 `5.256747 ms` 只用于证明采集闭合，不与 none、empty 或其他 phase 的独立 ELF 相减，不把跨 ELF 百分比求和，也不写任何收益结论。

### 2026-07-22 / O12：真实 PA 全 span 双证据链汇总

状态：**[观察工具：纯离线重算与现有 13 份 v2 产物闭合；未上板]**

#### 目标与实现边界

本阶段不再增加 device phase、record 字段或 raw 容量，也没有重新运行 A5。新增
`simpler_setup/tools/fdwic_submit_span_overview.py`，把现有泳道业务时间树与 13 份独立
Submit-PMU ELF 放进同一份离线报告，但从数据结构上禁止两条证据链混算。

泳道输入固定为 producer 的 `l2_swimlane_records.json`，不使用含 Perfetto 显示轨道和合成显示事件的
`merged_swimlane.json`，也不直接信任旁边已经生成的 analysis。工具先后读取 raw SHA，并复用现有
schema-v4 analyzer 当场重算：96 核物理拓扑、每核 1,280 个连续 Submit、父子包含、Kernel 唯一归属、
`dropped_records=0` 以及六组整数闭合全部通过后，才生成紧凑摘要。

PMU 输入固定为 13 个目录中的 raw/provenance 对。每份先通过现有 `load_capture()` 的 owner、selector、
status、拓扑、调用 shape、linked-Kernel gate 与 return-ready atomic 时间门禁，再用
`load_provenance()` 闭合 raw SHA、profile、编译宏和构建身份；缺 mode、重复 mode、场景键不一致或输入
生成期间变化都会拒绝发布。JSON/HTML 先成对暂存，再在输出目录独占锁下替换；第二份发布失败会恢复
旧 pair，不留下半新半旧的正式件。

本次没有给最新泳道反向伪造 provenance。13 份 PMU 实际来自四组 revision：

```text
9d3acca8  908f9adf  94172c64  d67f3f5e
```

13 个 `aicore_kernel` SHA 均不同；泳道 raw 又没有 build sidecar。因此 overview 明确发布
`swimlane_to_pmu_identity_bound=false`，只证明两条证据链各自严格闭合，并仅对齐 96 核拓扑和每核
Submit 数。mixed revision 可以并列展示，不能跨 ELF 相减、拼接或求和。

#### 时间与 PMU 的精确口径

真实源码的首末关系重新按 `submit_pmu.h` 核实：首个 Submit 先取 start tick、再启动 PMU；末个 Submit
先关闭 scalar segment、停止 PMU，再取 end tick。PMU gate 因而嵌在首末 SYS closure 内，不是比该
closure 更宽。每次 linked Kernel 又会成对 stop/start；`scalar_submit_elapsed_ticks` 是 gate-running
SYS 段的累计，再扣 result-used return-ready atomic 等待。PMU total/scalar-busy/primary I-cache 只在
gate 打开时计数，不含 linked Kernel，但不会为 return-ready atomic 停表，所以仍含 atomic 指令和最小
观察 hook 的事件；source-issue atomic 继续保留在时间与 counter 中。

泳道 `clock_freq_hz=1,000,000,000` 是 SYS counter 的换算频率，即 1 tick=1 ns；它与 AIC/AIV 约
1.65 GHz 的 PMU cycle 频率不是同一个量。泳道百分比只在同一 ELF 的排他 core-work 树中闭合；每张
PMU phase 卡则直接展示 `本 ELF 分子 / 本 ELF 分母 = 占比`。页面没有跨 PMU profile 的合计字段或
堆叠图。动态 phase 额外显示逐核 calls min/max 与零调用核数，避免把 AllocComplete AIV 的零值误读为
一次业务调用的零耗时。

#### 当前泳道同 ELF 分布

输入为：

```text
outputs/TestPagedAttentionUnroll_Case1_20260722_104657/l2_swimlane_records.json
```

schema-v4 重算得到全局 Submit 墙钟 4.844066 ms。96 核累计的 SubmitEnvelope 精确分成 SubmitUnion
`85.3839%` 与 BetweenSubmitResidual `14.6161%`；SubmitUnion 进一步精确闭合为：

| 区域 | 同一 SubmitUnion 占比 |
| --- | ---: |
| EfDrain | 16.1853% |
| Materialize | 25.3230% |
| PrepareMap | 5.3936% |
| Claim | 20.4422% |
| Fanin | 0.3989% |
| Register | 12.5173% |
| WinnerBuild | 1.6655% |
| AllocComplete | 0.2154% |
| LoserReplay | 3.1810% |
| SubmitInternalResidual | 10.8563% |
| SubmitTailResidual | 3.8217% |

EfDrain 内部又闭合为 KernelUnion `51.2020%` 与 control `48.7980%`。当前泳道还有 2 个 Kernel event
落在 WinnerBuild，但 analysis 没有发布该 child 的 Kernel union 时长；所以泳道表保持“原始业务
elapsed”名称，不假装为纯 Scalar。纯 Scalar 的 EfDrain/WinnerBuild/AllocComplete 归因只看分别排除
linked Kernel 的 PMU control ELF。

SubmitInternalResidual 本轮只有 `Claim->Materialize`，与 ArgBuild PMU 边界对应；SubmitTailResidual
按 LoserReplay/Register/WinnerBuild/AllocComplete 四种结尾列出，但当前没有独立 PMU phase。覆盖矩阵
同时列出 WorkerCompletion、OrchestrationReplay、Setup/Tail、FinalDrain 及其 Kernel/Residual 子项；
未覆盖项保持未覆盖，不以 residual 名称替代业务数据。Atomic、ClockBaseline、Commit、RingBp、DrainWon
均保持非加和 overlay。

#### 当前 PMU 独立 ELF 结果

13 份 Case1 v2 raw/provenance 全部重新严格加载。`none` 的完整 Scalar 时间分母、PMU total、
scalar-busy、primary request/miss 单列；empty-bracket 单列为 observer 校准。11 个业务 phase 的同 ELF
ALL 比例为：

| PMU phase | Scalar 时间 | request observed | miss observed |
| --- | ---: | ---: | ---: |
| ArgBuild | 5.927% | 19.119% | 27.021% |
| Materialize | 25.044% | 32.352% | 12.503% |
| Claim | 10.626% | 20.996% | 20.683% |
| Register | 6.028% | 14.119% | 14.486% |
| SubmitTransition | 14.231% | 22.198% | 27.843% |
| EfDrainControl | 7.092% | 16.735% | 12.784% |
| PrepareMap | 3.016% | 11.785% | 1.480% |
| Fanin | 0.358% | 0.486% | 1.027% |
| WinnerBuildControl | 3.472% | 5.047% | 1.815% |
| AllocCompleteControl | 0.197% | 0.244% | 0.403% |
| LoserReplay | 1.272% | 9.723% | 11.619% |

每一行的三个百分比只属于该行自己的 ELF。局部 request/miss 是 running read-clear observed，仍包含
边界附近 observer/bookkeeping；即使标出 capture gap，也不是无插桩业务事件的数学上下界。上述表只
用于找下一步应深入的区域，不是可相加的整窗解释，也不是性能收益。

#### 产物与离线门禁

正式离线加工件位于：

```text
outputs/down/fdwic_submit_span_overview_20260722_v2/
  fdwic_submit_span_overview.json
  fdwic_submit_span_overview.html
```

新增合成测试覆盖 schema-v4 分区闭合、13 profile 完整 cohort、缺失/重复 profile、泳道/PMU Submit
数不一致、mixed git head、所有 exclusive/residual/外围区域覆盖、零 miss 分母、HTML 只显示 min/max、
本 ELF 分母直显、成对发布回滚和并发发布锁。真实产物则完整执行一次 75 MB 泳道 raw 重算与 13 份
raw/provenance 加载；最终 HTML/JSON 约 46 KB/122 KB，只是离线加工件，不进入设备热路径或性能采集。

最终离线回归为：

```text
test_fdwic_submit_span_overview.py
test_fdwic_submit_pmu_report.py
test_fdwic_swimlane_converter.py
test_scene_test_cache.py
    386 passed

test_fdwic_swimlane_poll_batch
    2 passed

ruff check / format --check（本轮新增 Python 文件）
git diff --check
    PASS
```

### 2026-07-23 / O13：阶段计时改为 PMU total 1.65 GHz 口径

状态：**[观察工具：ABI/schema v3、336 项定向 UT、真实 A5 B1/Case1 及 13 个 profile 全部闭合]**

#### 为什么不能继续用 SYS tick 冒充阶段耗时

v2 的 `phase_elapsed_ticks` 是观察器边界内累计的 SYS_CNT：它适合确认 begin/end 是否闭合，
但不是阶段 local PMU total，也不能回答阶段内 scalar-busy 与非 scalar-busy 周期的关系。把它按
1 tick = 1 ns 直接呈现为阶段主时间，会把边界时间戳、观察器 bookkeeping 和 PMU 计数口径混在一起，
也无法与 whole PMU total/scalar 正确闭合。

v3 因此明确拆成两条数据：

1. 阶段主时间使用 `phase_total_cycles_observed`，由唯一 TOTAL PMU counter running read-clear，
   再在软件中重建 whole total；ALL/AIC/AIV 分别按
   `1.649844/1.650062/1.649731 cycles/ns` 换算等效时间。
2. `phase_elapsed_ticks` 只作为 raw SYS 边界闭合诊断保留。HTML 不把它换算成阶段时间，不用它做
   phase 占比，也不再出现“1 GHz phase time”的解释。

同一阶段还新增 `phase_scalar_busy_observed`。CNT3 配成与 CNT2 相同的 scalar selector `0x001`：
CNT2 保留 whole primary，CNT3 负责运行中 read-clear 的 phase local scalar。begin 顺序为
TOTAL → scalar → I-cache，end 按 I-cache → scalar → TOTAL 反序读取，使 scalar 观察窗嵌套在
total 观察窗内。逐核必须满足：

```text
phase_scalar_busy_observed <= phase_total_cycles_observed <= whole_total_cycles
phase_scalar_busy_observed <= shadow_scalar_busy <= whole_primary_scalar_busy
```

报告将逐核 `phase_total-phase_scalar` 后的结果命名为“非 Scalar-busy 残余”，不命名为空闲时间或
I-cache stall。linked vector/cube Kernel 通过 phase pause/resume 从两类 PMU counter 和 SYS 诊断中
共同排除；result-used return-ready atomic 只从 SYS 诊断扣除，仍进入 PMU total/scalar，因此 v3
并未伪造“去 atomic 的纯 scalar PMU 时间”。

#### ABI 与热路径边界

设备侧 phase sidecar/GM 容量仍为 64 B/核，没有逐调用增长；host raw 只增加固定 96 核字段。
v3 复用 sidecar 原空间，布局为：

```text
elapsed@0 u64
total@8 u64
request@16 u64
miss@24 u64
phase scalar@32 u32
shadow scalar@36 u32
shadow request@40 u32
shadow miss@44 u32
phase_id@48 u16
status@50 u16
begin@52 u32
end@56 u32
excluded_kernel_calls@60 u32
```

没有为阶段计数加入 DSB 或 phase-level `PIPE_ALL`。none ELF 禁止三个 phase reader 符号；phase ELF
必须同时包含 total/scalar/I-cache 三个 reader，避免 cache 复用错误变体。host raw schema 升为
`fdwic-submit-pmu-v3`，显式发布 PMU total、phase scalar、whole shadow scalar 和 SYS 边界诊断语义。

#### 真机校准

B1 正式闭合件：

- none：`outputs/TestPagedAttentionUnroll_CaseB1_20260723_043155/`
- empty：`outputs/TestPagedAttentionUnroll_CaseB1_20260723_043257/`
- Claim：`outputs/TestPagedAttentionUnroll_CaseB1_20260723_043400/`

none 的 96 个核均满足 CNT3 shadow scalar 与 CNT2 primary scalar 精确相等。phase 运行中
read-clear 后，empty 与 Claim 的 CNT3 shadow 均相对 CNT2 primary 固定少 `2 cycles/call`；
Case1 每核 1,280 次调用时正好少 2,560 cycles，证明误差随边界次数线性出现，raw 和 HTML 已显式
展示，未把它隐去。

Case1 empty：

| 角色 | PMU total cycles/call | 按角色频率换算 |
| --- | ---: | ---: |
| ALL | 1564.846 | 948.481 ns/call |
| AIC | 1485.029 | 899.984 ns/call |
| AIV | 1604.755 | 972.737 ns/call |

这说明当前 phase 观察器本身接近 0.95 us/call，并占 empty ELF whole PMU total 的 23.116%。
它是当前代码布局和固定调用点下测得的每组记录代码开销，不是可跨 ELF 精确扣除的常数。

#### Case1 v3 全量结果

13 个 profile 的 raw/provenance/HTML 全部重新采集并通过严格 loader。其中 none 没有 phase，
只提供 whole-window 分母；下表列出其余 12 个独立 phase ELF 的 ALL 聚合 96-core work。
“whole 占比”只以本行同一 ELF 的 whole PMU total 为分母，所有行禁止求和：

| phase | PMU total us | scalar us | 非 scalar 残余 us | whole total 占比 | scalar / phase total | request 占比 | miss 占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EmptyBracket | 116549.405 | 75065.241 | 41484.163 | 23.116% | 64.406% | 7.621% | 7.362% |
| EfDrainControl | 157130.274 | 107609.078 | 49521.196 | 31.409% | 68.484% | 16.750% | 16.327% |
| Claim | 195140.831 | 139943.418 | 55197.413 | 38.921% | 71.714% | 20.593% | 26.235% |
| ArgBuild | 155328.743 | 103572.005 | 51756.738 | 29.577% | 66.679% | 18.066% | 16.534% |
| Materialize | 223263.299 | 178164.487 | 45098.812 | 42.825% | 79.800% | 32.059% | 23.466% |
| PrepareMap | 127362.293 | 87292.749 | 40069.544 | 26.781% | 68.539% | 12.242% | 1.645% |
| Fanin | 2126.520 | 1642.250 | 484.269 | 0.538% | 77.227% | 0.450% | 0.591% |
| Register | 150738.181 | 101939.366 | 48798.814 | 29.845% | 67.627% | 13.798% | 12.785% |
| WinnerBuildControl | 37868.391 | 36978.703 | 889.687 | 7.514% | 97.651% | 6.589% | 1.993% |
| AllocCompleteControl | 34679.708 | 34324.720 | 354.988 | 6.872% | 98.976% | 0.747% | 0.965% |
| LoserReplay | 103511.458 | 69363.871 | 34147.587 | 21.941% | 67.011% | 9.778% | 7.325% |
| SubmitTransition | 190622.524 | 125218.596 | 65403.928 | 35.938% | 65.689% | 21.768% | 30.605% |

产物目录：

```text
none              outputs/TestPagedAttentionUnroll_Case1_20260723_043811/
arg-build         outputs/TestPagedAttentionUnroll_Case1_20260723_043857/
empty-bracket     outputs/TestPagedAttentionUnroll_Case1_20260723_043533/
materialize       outputs/TestPagedAttentionUnroll_Case1_20260723_043943/
claim             outputs/TestPagedAttentionUnroll_Case1_20260723_043639/
register          outputs/TestPagedAttentionUnroll_Case1_20260723_044029/
submit-transition outputs/TestPagedAttentionUnroll_Case1_20260723_044115/
efdrain-control   outputs/TestPagedAttentionUnroll_Case1_20260723_044202/
prepare-map       outputs/TestPagedAttentionUnroll_Case1_20260723_044248/
fanin             outputs/TestPagedAttentionUnroll_Case1_20260723_044336/
winner-build      outputs/TestPagedAttentionUnroll_Case1_20260723_044422/
alloc-complete    outputs/TestPagedAttentionUnroll_Case1_20260723_044508/
loser-replay      outputs/TestPagedAttentionUnroll_Case1_20260723_044555/
```

最新汇总件为：

```text
outputs/fdwic_submit_span_overview_20260723_v3/
  fdwic_submit_span_overview.json
  fdwic_submit_span_overview.html
```

目录名 `_v3` 表示输入采用 Submit-PMU v3 cohort；overview JSON 自身的 schema 为
`fdwic-submit-span-overview-v5`。v5 为每个业务 phase 固化了
`recording_cost_reference`，不再允许旧 payload 缺少参考值却沿用同一 schema。

总览中的泳道分区表新增 PMU 与 scalar-busy 两列。早期版本直接复用对应 phase ELF 的
`phase raw observed / 同 ELF raw whole`，该数值包含高频 begin/end 记录代码开销，只能称为
**raw 观测比例**，不能继续标成业务阶段占比。最终统一到单 phase 报告与 overview 共用的严格参考
公式：

```text
recording estimate =
    AIC empty 每组记录开销 × 本阶段 AIC 记录组数
    + AIV empty 每组记录开销 × 本阶段 AIV 记录组数

reference numerator = phase raw observed - recording estimate
reference share     = reference numerator / 同 phase ELF 的 raw whole
```

ALL 先汇总 AIC/AIV 的分子和分母再相除，不平均两个角色的百分比。分母明确保持 raw whole，不执行
`whole raw - recording estimate`：empty-bracket 只测得局部记录组，没有测量完整 Submit 整窗的
全部观察成本，不能制造一个“纯业务 whole”分母。页面同时保留 raw 比例、reference share 和
`recording estimate / phase raw`，后者用于提示两个大数相减时的校准敏感度。工具同时要求业务
phase 与 empty-bracket provenance 的场景和 Git revision 一致；两边都无 sidecar 时会明确标为
“只校验 raw 配置和核拓扑，revision 未证明”，不会静默宣称构建身份已闭合。

Materialize 的 PMU total 从 raw `42.825%` 得到 reference `20.469%`，scalar-busy 从 raw
`38.698%` 得到 reference `22.393%`；同一总览的泳道 Materialize 业务时间占比为 `25.323%`。
三类数据来自不同计数边界和不同 ELF，只能并列判断量级。曾经把分母也扣除记录估算得到的
`26.363%/26.756%` 已明确废弃，因为该算法虚构了未被 empty 校准覆盖的 whole 扣除量。全部业务
phase 单报告和 overview 已使用同一公式刷新，避免单报告继续显示 raw、overview 却显示另一种
参考口径。

映射只覆盖同业务边界、相邻 Submit 边界或明确 `control-only` 的行；SubmitInternalResidual
聚合、SubmitTailResidual 和 KernelUnion 等无等价 profile 的行保持“—”。

overview 的 11 个业务 phase 跨 ELF 合成诊断改为四层，不再把 raw observed 合计相对 none 的
膨胀直接命名为“观测偏差”或“插桩开销”：

1. 原始 observed 合计；
2. 空区间估算的记录代码开销，即
   `AIC/AIV 各自的 empty-bracket 每组 begin/end 开销 × 该角色实际 begin/end 记录组数`；
3. 扣除记录开销后的参考值，即第 1 层减第 2 层；
4. `submit-pmu-none` 独立基线。

11-phase ALL 的 PMU total 三层 phase 值为 `14.351794/9.492491/4.859303 ms/core`，
Scalar busy 为 `10.271346/6.113844/4.157502 ms/core`，非 Scalar-busy 残余为
`4.080448/3.378647/0.701800 ms/core`；none 基线三项分别为
`5.884932/5.733441/0.151491 ms/core`。I-cache request/miss 同样按四层展示，但保持
events/core。

第 2 层只是 empty-bracket ELF 在当前代码布局和运行状态下测得的估算，不是可跨 ELF 套用的精确
常数；第 3 层也只能用来检查量级。11 个业务 phase、empty-bracket 与 none 都是独立 ELF/进程，
还混有边界覆盖空洞、交叠、运行波动和 control-only 语义。因此不能把第 3 层冒充无记录代码下的
业务真值，不能拿它与 none 直接相减成优化收益；none 的非 Scalar-busy 残余分母很小，相关百分比
尤其容易放大。

SubmitUnion 表进一步改成全均值口径：泳道时间显示每核平均，泳道、PMU total 和 scalar-busy
三个占比都由同口径每核均值相除。表尾先将 11 个泳道分段的每核时间相加，得到
`4033.203 us/core`，与 SubmitUnion 父区间严格闭合为 100%；随后合计 9 个直接映射行与 ArgBuild
精确 residual 子段，共 10 个 SubmitUnion phase；SubmitTransition 位于 SubmitUnion 外，明确
不进入分子。表尾使用与全局 11-phase 相同的四层结构；真实 ALL 每核等效时间为：

| 指标 | 原始 observed | 记录开销估算 | 扣除后的参考值 | `submit-pmu-none` |
| --- | ---: | ---: | ---: | ---: |
| PMU total | 12.366143 ms/core | 8.279383 ms/core | 4.086759 ms/core | 5.884932 ms/core |
| Scalar busy | 8.966986 ms/core | 5.332525 ms/core | 3.634461 ms/core | 5.733441 ms/core |
| 非 Scalar-busy 残余 | 3.399157 ms/core | 2.946858 ms/core | 0.452299 ms/core | 0.151491 ms/core |

none 的范围是首个 Submit begin 到末个 Submit end，包含 BetweenSubmitResidual；10-phase 分子
没有覆盖其余 internal residual 和 tail residual。记录开销按真实 begin/end 记录组数估算，
linked Kernel pause/resume 产生的额外记录组也计入；AIC/AIV 分开计算后再按 32/64 核合并。
这组跨 ELF 合计只能描述“各 phase raw observed 合计减去局部记录估算”的诊断量。即使数值接近
正式泳道 SubmitUnion，也不能据此声称恢复了完整业务阶段、纯业务 whole 或无记录代码下的严格
闭合；phase 覆盖空洞、交叠、control-only 语义、代码布局和独立进程波动仍然存在。严格可提供的
占比只限于上式的 `reference numerator / 本 phase ELF raw whole`，并且必须标为参考值。动态
phase 的均值始终以全部 96 核为总体，零调用核保留为零值，不改成 active-core 或 per-call 均值。

该 overview 使用最新已有泳道 raw
`outputs/TestPagedAttentionUnroll_Case1_20260722_104657/l2_swimlane_records.json`，
重新闭合 schema-v4 排他树；泳道没有 provenance，因此只声明拓扑和每核 Submit 数对齐，不声明与
13 个 v3 PMU ELF 身份相同。

#### 验证与构建产物教训

Python 定向回归：

```text
test_fdwic_submit_pmu_report.py
test_fdwic_submit_span_overview.py
test_scene_test_cache.py
    341 passed
```

AICPU、Claim phase AICore、none AICore 和 host runtime 都完成真实 CANN/A5 编译；C++ clang-format、
`ruff check` 和 `git diff --check` 通过。`ruff format --check` 仍报告两处既有且与本轮无关的
格式差异。首次 B1 none 曾出现 96 核记录全零，最终查明不是硬件 counter
失效，而是 AICore cache 已为 v3、实际 `build/lib/.../libhost_runtime.so` 仍停留在 v2。通过正式
`RuntimeBuilder("a5").get_binaries("fully_distributed_within_core", build=True)` 重建并落盘后，
cache 与实际加载 host SO SHA 一致，所有真机门禁随即闭合。后续 ABI 变更必须核对实际加载件，
不能只看 build cache 或把 host/device ABI 不一致误判成 PMU 行为。

## 13. 2026-07-30 standalone shared loser 非 atomic 优化

### 13.1 固定合同与观察口径

本阶段只修改 `tests/atomic_probe/pa_scheduler` standalone shared 路径，
不迁移真实 simpler。固定合同为：

- 96 个 worker 全量回放同一任务计划；
- Alloc/QK-PV/SF-UP 分别保持 `96/32/64` 个 Claim 候选；
- ClaimMax 的类型、地址选择和候选人口冻结；
- perf-clock 决定端到端护栏，full-swimlane 负责区分 atomic、Kernel 与
  非 atomic loser 控制工作；
- 当前保留条件为 loser 非 atomic 工作有效下降，同时端到端回退不超过
  约 2%。

true-loser 定义为 `Claim attempted && !won`。EfDrain 非 atomic 时间从
父区间扣除 Kernel 与 Atomic 的区间并集；Claim outer 从 Claim 父区间
扣除 atomic；不把 atomic 总线等待冒充 scalar 指令消减。

### 13.2 shared DrainReady 两槽扫描

**[已保留] shared 正常 drain 只扫描两个普通可用 slot**

shared 单 lane PA 在 `occupied_count >= kUsableSlots == 2` 时先等待并
drain，`FindFreeSlot()` 又总取最低空槽，因此正常路径只能占用 slot 0/1。
候选只把 shared `DrainReady()` 的扫描上界从 4 收敛为 2；private 和
fatal 清理仍扫描全部 4 个物理 slot。CCEC AIC/AIV 优化后 IR 已确认终止
条件由 4 变为 2。

CPU shared/private、CCEC perf-clock/full-swimlane、A5 B256 完整语义与
后处理、97 项 Python 工具回归均通过。候选泳道：

```text
tests/atomic_probe/pa_scheduler/outputs/
  pa_scheduler_shared_swimlane_20260730_130222_3656488/ccec/
```

72,448 个 true-loser 的非 atomic core-time：

| 区域 | 基线 | 候选 | 变化 |
| --- | ---: | ---: | ---: |
| EfDrain control | 6.525702 ms | 6.137015 ms | -5.956% |
| Claim outer | 6.846398 ms | 6.374741 ms | -6.889% |
| post-Claim tail | 11.297971 ms | 11.615971 ms | +2.815% |
| SubmitTransition | 12.656433 ms | 11.877174 ms | -6.157% |
| 合计 | 37.326504 ms | 36.004901 ms | -3.541% |

平均每个 true-loser 从 `515.218 ns` 降至 `496.976 ns`。两份泳道的
true-loser 集合交集为 71,203；只比较交集仍下降 `3.504%`。

ClaimMax 调用保持 `73,728`，但动态轮询使所有 logical atomic calls 从
193,894 增至 200,902。因此该阶段没有宣称 atomic 次数整体下降。

12+12 个独立 perf-clock 样本中，基线/候选 mean 为
`2,292.263/2,304.011 us`，候选回退 `0.513%`；median 回退
`0.361%`。其中位置对称的后 4+4 样本 mean/median 回退
`0.246%/0.238%`。非 atomic loser 降幅明确且端到端低于 2% 护栏，
因此保留。

更完整的正确性证明、逐区域均值、动态 atomic 变化和产物说明见
[shared TensorMap 开发记录](pa_scheduler/shared_tensormap_record.md)。
