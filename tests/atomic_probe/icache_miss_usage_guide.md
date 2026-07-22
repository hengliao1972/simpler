# A5 PA Scheduler I-cache Miss 采集与分析指南

## 1. 目标与最终构建口径

本指南同时记录真实 simpler FDWIC PA 与 `tests/atomic_probe/pa_scheduler` standalone CCEC 的 I-cache 观察方法。目标是回答：在 Submit-all 整个调度回放期，32 个 AIC 和 64 个 AIV 每核发生了多少 I-cache request/miss，其中哪些 miss 值得继续优化。真实 PA 结论以真实 PA 独立诊断 ELF 为准；standalone 只保留历史方法、接口校准和模型边界证据，不能替代真实 PA profile。

standalone 当前保留两类正式观察构建：

| 构建 | 内容 | 是否包含 PMU |
| --- | --- | --- |
| `swimlane` | 普通阶段泳道 + 逐 atomic 泳道，在同一 AIC/AIV scalar lane 合并采集 | 否 |
| `submit-pmu` | 每物理子核的 Submit-all PMU 整窗，并可编译一个局部 phase | 是，仅 CCEC |

`run` 、`smoke` 和 phase 名是运行或编译选择，不是额外的第三类构建。

### 1.1 真实 PA `submit-pmu-none`

> **2026-07-22 ABI/schema v2 口径提示**：当前设备 ABI 与 raw JSON schema 已同步升级为
> `fdwic-submit-pmu-v2`。所有真实 PA submit-PMU profile
> 都在 linked vector/cube Kernel 调用前后统一门控，完整窗的 PMU counter 与
> `scalar_submit_elapsed_ticks`、以及命中的 phase 时间都排除 Kernel 整段。仅消费返回值的
> return-ready/result-used atomic 依赖区间从 Scalar SYS 时间分母及命中的 phase 时间中扣除，
> 不停 PMU；source-issue atomic 保留。因而 PMU total、scalar busy、I-cache request/miss
> 仍包含 atomic 指令及最小时间 hook 的指令事件。`wall - scalar_submit_elapsed_ticks` 同时混有
> linked Kernel、被扣除的 return-ready atomic 时间和门控边界间隙，不能命名为纯 Kernel 时间。
> 本章现存 v1 HTML 与数值只作历史记录；采用 v2 做新分析时必须重新上板采集，不能与旧结果
> 合并、相减或拼接比例。

#### 2026-07-22 Case1 v2 全量 HTML 索引

下表是同一轮 v2 口径下的完整入口。每个目录均包含
`fdwic_submit_pmu_raw.json`、`fdwic_submit_pmu_provenance.json` 和
`fdwic_submit_pmu_report.html`：

| profile | Case1 产物目录 |
| --- | --- |
| `submit-pmu-none` | `outputs/TestPagedAttentionUnroll_Case1_20260722_211333/` |
| `submit-pmu-arg-build` | `outputs/TestPagedAttentionUnroll_Case1_20260722_212608/` |
| `submit-pmu-empty-bracket` | `outputs/TestPagedAttentionUnroll_Case1_20260722_212653/` |
| `submit-pmu-materialize` | `outputs/TestPagedAttentionUnroll_Case1_20260722_212739/` |
| `submit-pmu-claim` | `outputs/TestPagedAttentionUnroll_Case1_20260722_212857/` |
| `submit-pmu-register` | `outputs/TestPagedAttentionUnroll_Case1_20260722_212943/` |
| `submit-pmu-submit-transition` | `outputs/TestPagedAttentionUnroll_Case1_20260722_213030/` |
| `submit-pmu-efdrain-control` | `outputs/TestPagedAttentionUnroll_Case1_20260722_213152/` |
| `submit-pmu-prepare-map` | `outputs/TestPagedAttentionUnroll_Case1_20260722_213240/` |
| `submit-pmu-fanin` | `outputs/TestPagedAttentionUnroll_Case1_20260722_213326/` |
| `submit-pmu-winner-build-control` | `outputs/TestPagedAttentionUnroll_Case1_20260722_211624/` |
| `submit-pmu-alloc-complete-control` | `outputs/TestPagedAttentionUnroll_Case1_20260722_220027/` |
| `submit-pmu-loser-replay` | `outputs/TestPagedAttentionUnroll_Case1_20260722_222533/` |

严格 loader 已复验每个 profile 的 96/96 trusted、linked-Kernel gate 和 return-ready atomic
时间门禁；所有 HTML 的 I-cache 卡片只显示逐核 `min/max`。旧 v1 各章节中的数表继续作为历史记录，
新的分析结论应以本索引中的 v2 三件套为入口。不同 profile 来自不同诊断 ELF 和独立进程，
时间及计数不得跨 profile 相加或相减；`submit-pmu-empty-bracket` 只用于校准观察器本身，不能解释为
业务阶段占比。

真实 PA 已建立第三条独立证据链 `--fdwic-profile submit-pmu-none`。它不是 standalone 的 `submit-pmu` 产物，也不与泳道 ELF 共用 I-cache 结论：

- 编译期固定 `PTO_FDWIC_TRACE_ENABLED=0`，去除普通泳道和逐 atomic 泳道记录；ABI v2 仍保留
  return-ready atomic 的最小 SYS 时间扣除 hook，该 hook 不生成 atomic 泳道记录，也不停止 PMU；
- 去除通用逐 task PMU ring，只保留每核首个 Submit 到末个 Submit 的一次 gate；
- CNT2 采集 scalar busy，CNT6/CNT7 采集 primary request/miss，CNT8/CNT5 做逐核 shadow 复核；
- AICPU owner 在 96 个物理子核上保存、配置、读回并恢复 PMU 寄存器；
- host 只有在 32 AIC、64 AIV、96 个唯一物理 ID、32 个 1:2 mixed triplet、每核实际 Submit 数与 orchestration 声明值一致（Case1 为 1280、B1 为 5）、主影子一致和 Restore 96/96 全部闭合时才发布正式 raw；
- 固定输出 `fdwic_submit_pmu_raw.json` 和 `fdwic_submit_pmu_report.html`，先写临时文件再原子替换正式文件。

真实 Case1/B256 命令为：

```bash
source /home/q00473782/Ascend/cann-9.1.0-weekly-20260708/cann/set_env.sh
source /home/q00473782/.venv/bin/activate
export PTO_ISA_ROOT=/home/q00473782/atomic/private/gpt/pto-isa-ddafa
export PYTHONPATH="$PWD:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest \
  examples/a5/fully_distributed_within_core/paged_attention_unroll/test_paged_attention_unroll.py \
  --platform a5 --runtime fully_distributed_within_core --level 2 \
  --case Case1 --manual include --fdwic-profile submit-pmu-none \
  --rounds 1 -s -v
```

2026-07-21 的正式产物位于：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_003335/
  fdwic_submit_pmu_raw.json       46,274 B
  fdwic_submit_pmu_report.html    76,933 B
```

该轮全局 Submit 为 **5,075.360 us**。AIC/AIV 的 request/core mean 分别为 646,963.94/594,542.61，miss/core mean 分别为 1,196.88/16,943.17，聚合 miss 率分别为 0.1850%/2.8498%。PMU total、scalar busy、request、miss 都是同一 ELF、同一物理核、同一首末 Submit 窗口的数据；仍不能把 `miss * 90 ns` 当作可直接消除的跨核墙钟损失。

真实 PA raw 顶部的“完整 Submit”表示**全部 worker 中最早的首个 Submit 起点到最晚的末个 Submit 终点**，是 96 核共同形成的全局 `SYS_CNT` 墙钟范围，不等于逐核 PMU gate 的平均值。对单个 worker，源码先读取 `first_submit_start_tick` 再调用 `metrics_prof_start()`，末次 Submit 则先调用 `metrics_prof_stop()` 再读取 `last_submit_end_tick`；因此该核的 `SYS_CNT` 首尾区间在两侧包住 PMU gate，并包含 start/stop 及其 `PIPE_ALL` 边界成本。PMU total 按本机约 1.65 cycles/ns 的长窗校准值换算，`SYS_CNT` 为 1 ns/tick；两者的边界、时钟口径和聚合对象均不同，只能在同一 ELF 内校验数量级和跨轮方向，不能直接相减来归因观察成本或业务耗时。

现行真实 PA raw summary 包含 `sum/min/mean/max`，生产 HTML 还会从通过 96 核门禁的 records 重新校验这些聚合。I-cache request/miss 在 HTML 中只展示逐核 `min/max`，不再展示 mean；raw 聚合口径保持不变。PMU total 与 scalar busy 各自独立取极值，不保证来自同一个物理核。

#### 1.1.1 HTML 的逐核时间与 PMU 加工口径

下述 2026-07-21 报告增强在**当时的 v1 producer/设备 ABI**上只改了 Python 加工层；“ABI 不变”仅描述那次历史提交，不适用于 2026-07-22 的 ABI v2。该版 ALL/AIC/AIV 卡片展示：

- `Submit SYS_CNT/core`：每核 `last_submit_end_tick-first_submit_start_tick` 的 mean/min/max，按 1 ns/tick 显示为 us；它不是顶部“最早核起点到最晚核终点”的跨核全局时间范围；
- PMU total、Scalar busy 与非 Scalar-busy 残余的 cycles mean/min/max，以及按 ALL/AIC/AIV 各自 1.649844/1.650062/1.649731 cycles/ns 换算的等效时间范围；
- `PMU-total / SYS-window/core`：先逐核计算 `total_cycles/submit_elapsed_ticks`，再展示 mean/min/max。它只表示当前 ELF 两套相近长窗的有效比，不是瞬时 AICore 频率，也不是利用率；
- primary I-cache request/miss 的 min/max、`Σmiss/Σrequest` 和按 miss 极值换算的 90 ns 直觉量尺。

非 Scalar-busy 残余必须先对每条 record 计算 `total_cycles-scalar_busy`，再求 min/mean/max；不能用 `min(total)-min(scalar)` 或 `max(total)-max(scalar)` 拼接，因为两个极值可能来自不同物理核。有效 cycle 比同样采用逐核 ratio 的算术均值，不是 `Σtotal/Σelapsed`。这些派生字段只存在于受信 `SubmitPmuCapture` 和 HTML，raw summary 仍保持 producer 原有字段，因此没有给设备热路径、GM 容量或 raw 合同增加成本。

上述等效时间都只解释当前诊断 ELF。不得拿它们与 perf-clock、swimlane 或另一个 phase ELF 相减，也不得把 `total-scalar` 改名为 Scalar 空闲、I-cache stall 或 vector/cube wait。Register 的 12 轮正式样本还出现过 primary=shadow 完整闭合、但 AIV miss/call 中途跃迁而阶段时间保持稳定的状态，说明报告必须把时间与 I-cache 计数并列展示；单轮 miss 或 miss rate 不能独立决定优化结论。

#### 1.1.2 新采集的构建 provenance 三件套

新的正式 Submit-PMU case 成功后应同时出现：

```text
fdwic_submit_pmu_raw.json
fdwic_submit_pmu_provenance.json
fdwic_submit_pmu_report.html
```

`raw` 仍由 C++ producer 原子发布，Python 只读，不允许为了加入构建信息而回写。`provenance` 使用固定 schema `fdwic-submit-pmu-provenance-v1`，通过同一次 raw 读取冻结的文件名、字节数、SHA256 和 capture mode 与该轮数据绑定。HTML 在同次闭合后展示 sidecar SHA、构建时 source-v2、profile 宏/cache key，以及下列四件实物的完整文件与 literal `.text` SHA/大小：

- worker 实际加载的 `build/lib/.../aicore_kernel.o`；
- 对应 `build/cache/.../aicore/aicore_aic_combined.o`；
- 对应 `build/cache/.../aicore/aicore_aiv_combined.o`；
- worker 使用的 `libhost_runtime.so`。

final ELF 与 AIC/AIV combined 不在同一目录。前者必须从实际 output path 取证，后两者和 `.git_commit` source-state stamp 必须从对应 build cache 取证，不能拿 cache 中间件冒充实际加载 ELF。身份在 ELF profile 符号门禁通过后、case 开始前冻结，case 返回后发布报告前再次重算；任一文件、stamp、raw binding、profile 宏或 cache key 不一致都会拒绝三件套闭合。sidecar 与 HTML 先完整生成到临时文件，发布前后都复核 raw 快照；任一步失败会恢复调用前的整对产物，不能留下只有一件更新或绑定旧 raw 的“正式”文件。

这条机制只在编译完成或 case 返回后运行，不进入 Submit/AICore 热路径，也不扩设备 GM 或 raw ABI。历史无 sidecar 的 raw 仍可离线生成基础 HTML，但从该机制落地后的新正式采集若缺少 provenance，应视为产物不完整，不能再靠采集时的当前 HEAD 或文档手工猜测 ELF 身份。

真实 A5 B1 已验证完整窗和分段窗两种 profile：

- `submit-pmu-none`：`outputs/TestPagedAttentionUnroll_CaseB1_20260721_111118/`，96 核均为 5 次 Submit，primary/shadow request 与 miss 逐核完全相等；raw/provenance/HTML 分别为 44,339/3,050/80,808 B；
- `submit-pmu-register`：`outputs/TestPagedAttentionUnroll_CaseB1_20260721_111427/`，96 核均为 5 次 Submit、5 对 Register begin/end，primary/shadow 同样逐核相等；三件套分别为 69,638/3,103/83,714 B。

两轮 provenance 都绑定构建提交 `15c54b33`，但 profiled cache key 和 AICore extra cache key 分别落到 `submit-pmu-none`/`aa43623282e2a7db` 与 `submit-pmu-register`/`32c26e06ad76d186`，据此确认完整窗与分段窗没有复用错误 ELF。报告可被 loader 重新严格解析，并与重新渲染的 HTML 逐字节一致。

### 1.2 真实 PA 首个单阶段 profile：`submit-pmu-arg-build`

真实 PA 已完成首个跟随最新泳道业务边界的单阶段 profile：

```bash
python -m pytest \
  examples/a5/fully_distributed_within_core/paged_attention_unroll/test_paged_attention_unroll.py \
  --platform a5 --runtime fully_distributed_within_core --level 2 \
  --case Case1 --manual include --fdwic-profile submit-pmu-arg-build \
  --rounds 1 -s -v
```

它仍保留本 ELF 的完整 Submit primary，只在 Kernel/Alloc 的 compete-first Claim 完成后开启局部 bracket，在匹配 Finish 恢复并校验 ticket 后、Materialize 入口前结束。实际覆盖 Begin 返回、同步 eager callback 构参和 Finish 重入，不包含 Claim 与 Materialize 本体。编译宏为：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=1
PTO_FDWIC_TRACE_ENABLED=0
```

none 的 ABI 仍为 6,272 B；arg-build 在相同 96 份整窗 64 B 记录后追加 96 份 64 B phase sidecar，总计 12,416 B。raw schema 仍为 `fdwic-submit-pmu-v1`，通过 `capture.mode` 区分：

- `submit-pmu-none` 不含 `configuration.phase`，每条 record 也不含 phase 字段；
- `submit-pmu-arg-build` 增加 `phase_elapsed_ticks`、`phase_icache_requests_observed`、`phase_icache_misses_observed`、begin/end 次数、最大 shadow 分段和 `phase_status`；
- phase 时间只与同一 ELF 的 `Σsubmit_elapsed_ticks` 比较；request/miss observed 只与同一 ELF、同一角色的 primary 比较；
- 不提供 phase-local PMU total、scalar busy 或 I-cache stall 时间。

CNT6/CNT7 在完整窗口中不读取，继续作为 primary；CNT8/CNT5 运行中 read-clear 并软件重建 shadow whole。`primary - shadow` 是分段重建的 capture gap。phase observed 会包含 counter 边界附近少量观测 bookkeeping 的取指，因此它不是原业务事件数的数学下界；`observed + capture gap` 也不是数学上界。HTML 对 ALL/AIC/AIV 各自展示 phase core-time 与时间占比；request/miss 展示 observed 总数、逐核 min/max、同 ELF primary 分母与占比，并明示 capture gap 敏感性量尺。I-cache per-call 加权均值仍保留在 Python 派生数据中，不再作为 HTML 主统计展示。不同 profile 的 ELF 绝对时间、request 和 miss 都不能相减。

首轮 Case1/B256 闭合件为：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_014355/
```

该轮 96 核每核 1,280 次 bracket 全部闭合，phase status 为 `0x3f`，primary/shadow 96/96 精确相等；同一 ELF 内 ALL 的 phase core-time、request observed 和 miss observed 份额分别为 5.557%、20.716% 和 21.334%。该数据首先证明采集链闭合，并提示后续需要空 bracket 校准观察 bookkeeping；不能直接把约 21% 写成零插桩业务区间的真实 I-cache 比例。

后续 selector 必须继续跟随真实泳道的排他 span，一次 ELF 只测一个区域，并保留该 ELF 自己的完整 Submit primary 作为比例分母。

### 1.3 真实 PA 空区间校准：`submit-pmu-empty-bracket`

局部 phase 的 begin/end 本身会执行 shadow counter read-clear、状态检查和累计 bookkeeping。为了先量出这套观察器在真实 simpler A5 PA 热路径上的经验指纹，已增加独立诊断 profile：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=2
PTO_FDWIC_TRACE_ENABLED=0
```

它不包围任何业务代码。在 Kernel/Alloc compete-first 路径的 Claim 结束处，每次 Submit 紧邻执行一次 phase begin 和 phase end；Case1/B256 每核固定 1,280 次，B1 每核固定 5 次。raw 中必须同时匹配：

```text
capture.mode                    = submit-pmu-empty-bracket
configuration.phase.id          = 2
configuration.phase.name        = empty-bracket
configuration.phase.boundary    = claim_end_adjacent_empty_bracket
configuration.phase.counter_semantics
                                = running_read_clear_empty_bracket_calibration
configuration.phase.time_semantics
                                = outer_sys_cnt_around_adjacent_begin_end_pair
```

真实 Case1 的构建和运行仍通过 pytest profile 入口完成；构建缓存会按 profile 宏生成独立 AIC/AIV ELF，不能拿其他 phase 的旧 ELF 拼接：

```bash
source /home/q00473782/Ascend/cann-9.1.0-weekly-20260708/cann/set_env.sh
source /home/q00473782/.venv/bin/activate
export PTO_ISA_ROOT=/home/q00473782/atomic/private/gpt/pto-isa-ddafa
export PYTHONPATH="$PWD:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest \
  examples/a5/fully_distributed_within_core/paged_attention_unroll/test_paged_attention_unroll.py \
  --platform a5 --runtime fully_distributed_within_core --level 2 \
  --case Case1 --manual include --fdwic-profile submit-pmu-empty-bracket \
  --rounds 1 -s -v
```

用例成功后，会在本轮 `outputs/TestPagedAttentionUnroll_Case1_<timestamp>/` 中自动生成并严格校验：

```text
fdwic_submit_pmu_raw.json
fdwic_submit_pmu_report.html
```

HTML 可直接用浏览器打开。若只需对已存在的 raw 重新生成报告，可在仓库根目录执行：

```bash
python -m simpler_setup.tools.fdwic_submit_pmu_report \
  outputs/TestPagedAttentionUnroll_Case1_<timestamp>/fdwic_submit_pmu_raw.json
```

输出文件名固定为同目录下的 `fdwic_submit_pmu_report.html`。分析器会重新核对 profile、phase 元数据、32 AIC + 64 AIV、每核调用次数、begin/end 闭合、状态位、primary/shadow 和 owner restore；不能把手工摘出的局部数字绕过这些门禁后当正式结果。

#### 两套边界必须分开解释

empty-bracket 的时间与 I-cache 事件故意使用两套不同边界：

1. `phase_elapsed_ticks` 由一对外层 SYS_CNT 包围相邻的完整 begin/end 调用，包含 shadow read-clear、begin/end 内部 SYS_CNT、状态检查和累计等观察器路径；它还带有外层时间戳自身的测量粒度，是“完整观察器调用对”的经验耗时，不是某段业务时间。
2. `phase_icache_requests_observed` 与 `phase_icache_misses_observed` 只统计 begin 的 shadow read-clear 到 end 的 shadow read-clear 之间被 CNT8/CNT5 读出的事件。它没有覆盖完整 begin/end 调用对，尤其不能把外层 SYS_CNT 时间边界等同为 I-cache request/miss 边界。

因此报告中的每 call 指标分别按同一角色聚合后计算：

```text
time/call    = Σphase_elapsed_ticks / Σphase_end_reads
request/call = Σphase_icache_requests_observed / Σphase_end_reads
miss/call    = Σphase_icache_misses_observed / Σphase_end_reads
```

它们是 ALL/AIC/AIV 各自的加权每次调用均值，不是 raw 为每次调用保存了一条记录。empty-bracket 继续复用每核 64 B phase sidecar，phase ABI 总大小仍为 12,416 B，没有为了逐调用校准扩充 raw。

#### Case1 稳态校准结果

两轮 Case1/B256 正式闭合件位于：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_021158/
outputs/TestPagedAttentionUnroll_Case1_20260721_021311/
```

两轮均为 96 核每核 1,280 次、phase status `0x3f`、primary/shadow 96/96 精确相等，request/miss capture gap 均为 0。每 call 结果如下：

| 轮次 | 角色 | 外层时间 ns/call | request observed/call | miss observed/call |
| --- | --- | ---: | ---: | ---: |
| `021158` | ALL | 640.465 | 49.340 | 1.342 |
| `021158` | AIC | 567.962 | 48.919 | 0.008 |
| `021158` | AIV | 676.717 | 49.550 | 2.009 |
| `021311` | ALL | 639.272 | 49.337 | 1.356 |
| `021311` | AIC | 567.619 | 48.870 | 0.008 |
| `021311` | AIV | 675.099 | 49.570 | 2.030 |

两轮全局 Submit 时间范围分别为 4,972.718 us 和 4,866.126 us。Case1 的每 call 校准值高度接近，可作为当前源码和工具版本下解释局部 phase 观察污染的稳态经验量尺。AIV 稳定出现约 2 次 miss/call，而 AIC 接近 0；这是观察器在不同 scalar 角色上的实测指纹，尚不能在没有进一步代码布局证据时归因为某一条具体指令。

当前 ELF 核验也不支持直接改 reader 来“压低校准值”：AIC/AIV reader 都是同一份 92 B noinline 实现，且在 128 B line 下都跨两行；本机 DAV3510 模型配置中的 AIV scalar I-cache 容量和 set 数只有 AIC 一半，而 AIV 角色代码更大。现阶段只能把约 2 miss/call 视为容量、角色代码与具体布局共同形成的稳定观察指纹，聚合 PMU 不能定位到某一条 cache line。因而保留当前 reader 和布局；若该底噪妨碍后续 selector，应另做只改对齐的 empty A/B，不能把布局变化夹带进业务 phase。

B1 闭合件位于：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260721_020932/
outputs/TestPagedAttentionUnroll_CaseB1_20260721_021100/
```

这两轮只有每核 5 次调用，ALL 分别为 725.192/719.304 ns、84.844/69.194 request 和 1.956/2.444 miss 每 call；AIC 的 request/miss 也明显比 Case1 稳态更易波动。B1 适合快速验证接口、次数和闭合，不适合代替 Case1 判断稳态观察器成本；少量调用会把 ELF 首次进入、取指预热和启动时序放大到每 call。

#### 使用边界

- empty-bracket 是观察器的**经验校准 profile**，不是观察成本的数学下界，也不是可以从业务 phase 中直接扣除的固定常数。
- `submit-pmu-empty-bracket` 与 `submit-pmu-arg-build` 使用不同诊断 ELF；phase 宏、代码布局、I-cache 状态和跨核到达时序都会改变。可以用 empty 的量级判断局部 observed 是否主要由观察器构成，不能逐项相减生成所谓“修正后的 arg-build”。
- 即使同一轮 capture gap 为 0，也只证明 primary/shadow 整窗重建闭合，不会把上述两套边界变成同一个区间。
- 现行单子核受控微基准给出的 AIC `77.376 ns/miss`、AIV `94.030 ns/miss` 只是一把 core-latency 直觉量尺；既有报告中的 90 ns 则是历史兼容口径。miss 可在单核内重叠、被流水隐藏，96 核之间也并行，其中还可能包含观察器自身带来的 miss。因此无论采用分角色标尺还是旧 `miss × 90 ns`，都不能解释为 Submit 墙钟损失，更不能当成候选优化的可兑现收益。

### 1.4 真实 PA `submit-pmu-materialize`

`submit-pmu-materialize` 是首个经过 empty-bracket 校准后落到真实业务 span 的局部 profile。它继续保留该 ELF 自己的完整 Submit primary，只把 running read-clear bracket 放到泳道 `Materialize` 的同一业务边界。编译宏为：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=3
PTO_FDWIC_TRACE_ENABLED=0
```

raw 中的 profile 元数据必须精确匹配：

```text
capture.mode                    = submit-pmu-materialize
configuration.phase.id          = 3
configuration.phase.name        = materialize
configuration.phase.boundary    = materialize_begin_to_materialize_end
configuration.phase.counter_semantics
                                = running_read_clear_observed_bracket
configuration.phase.time_semantics
                                = inner_sys_cnt_between_boundary_observers
```

当前真实 PA 使用 compete-first 路径：Kernel/Alloc 的 Finish 完成 ticket 恢复与校验、结束 arg-build 后，在泳道 `Materialize.begin` 对应位置打开 bracket；`dist_submit_materialize_and_prepare_map()` 内完成 task-cap 检查和 `dist_submit_materialize_args()` 后立即关闭，再进入 `PrepareMap`。因此该 profile 覆盖 Materialize 自身，不包含 Claim-to-Materialize 构参区间，也不包含后续 `dist_submit_prepare_map()`。仍受支持的 one-shot 入口使用相同的 Materialize 业务首尾边界。

每个成功 Submit 恰好执行一次 Materialize bracket，所以调用 shape 与每核 Submit 数完全一致：Case1/B256 为每核 1,280 次、全局 122,880 次，B1 为每核 5 次、全局 480 次。失败路径不会伪造 phase end；只要某核 begin/end 不平衡、次数不符或状态位不完整，host/分析器就拒绝发布受信结果。

真实 Case1 命令为：

```bash
source /home/q00473782/Ascend/cann-9.1.0-weekly-20260708/cann/set_env.sh
source /home/q00473782/.venv/bin/activate
export PTO_ISA_ROOT=/home/q00473782/atomic/private/gpt/pto-isa-ddafa
export PYTHONPATH="$PWD:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest \
  examples/a5/fully_distributed_within_core/paged_attention_unroll/test_paged_attention_unroll.py \
  --platform a5 --runtime fully_distributed_within_core --level 2 \
  --case Case1 --manual include --fdwic-profile submit-pmu-materialize \
  --rounds 1 -s -v
```

每轮测试会在对应输出目录自动生成并校验：

```text
fdwic_submit_pmu_raw.json       # 逐核权威原始数据
fdwic_submit_pmu_report.html    # 已加工的 ALL/AIC/AIV 报告
```

HTML 可直接用浏览器打开。需要从已有 raw 重建时，在仓库根目录执行：

```bash
python -m simpler_setup.tools.fdwic_submit_pmu_report \
  outputs/TestPagedAttentionUnroll_Case1_<timestamp>/fdwic_submit_pmu_raw.json
```

#### 两轮 Case1 稳态结果

两轮正式闭合件位于：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_024748/
outputs/TestPagedAttentionUnroll_Case1_20260721_024909/
```

两轮均为 96/96 受信记录、每核 1,280 次、begin/end 122,880/122,880、phase status `0x3f`，primary/shadow 的 request/miss capture gap 均为 0。下表的三个占比都只使用**本轮同一个 Materialize ELF** 的分母：时间以 `Σsubmit_elapsed_ticks` 为分母，request/miss 以各角色的完整 Submit primary 为分母。

| 轮次 | 角色 | 时间 ns/call | request observed/call | miss observed/call | 时间占比 | request 占比 | miss 占比 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `024748` | ALL | 797.814 | 233.197 | 1.474 | 22.560% | 37.688% | 12.056% |
| `024748` | AIC | 776.265 | 228.942 | 0.022 | 22.540% | 36.532% | 10.487% |
| `024748` | AIV | 808.588 | 235.325 | 2.201 | 22.570% | 38.278% | 12.065% |
| `024909` | ALL | 797.061 | 233.238 | 1.418 | 22.105% | 37.741% | 11.681% |
| `024909` | AIC | 775.653 | 228.170 | 0.021 | 21.882% | 36.442% | 10.997% |
| `024909` | AIV | 807.765 | 235.772 | 2.116 | 22.214% | 38.404% | 11.685% |

两轮全局 Submit 时间范围分别为 4,922.142 us 和 4,851.282 us。Materialize 的 ALL request observed 稳定在约 233.2 次/call，AIC/AIV 也都稳定在约 228～236 次/call；相对 empty-bracket 的约 49 次/call 经验指纹，这个 request 信号在量级上更明显。但两者来自不同诊断 ELF，代码布局、缓存状态和到达时序不同，这里只能作方向性判断，不能执行 `materialize - empty`。

AIV 的 Materialize miss observed 为约 2.20/2.12 次/call，与 empty-bracket 两轮约 2.01/2.03 次/call 处于同一量级。这说明当前数据尚不能证明 Materialize 业务体本身带来明显的 AIV miss 增量；也绝不能跨 ELF 相减后把约 0.1 次/call 冒充业务净 miss。AIC 的 miss observed 同样很小。现阶段可受信的结论是：Materialize 时间和 request 信号两轮稳定，而 miss 归因仍受到观察器经验指纹限制。

B1 只用于结构与 cold-path 核验：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260721_024533/
outputs/TestPagedAttentionUnroll_CaseB1_20260721_024641/
```

两轮都满足 96/96 记录、每核 5 次、480/480 begin/end、`0x3f` 状态和零 capture gap，证明 profile 选择、边界次数、ABI 与报告链路闭合。由于每核只有 5 次调用，首次进入 ELF、取指预热和启动时序会被放大到每 call；B1 的 per-call 值只作 cold 证据，不用于替代上述 Case1 稳态归因。

### 1.5 真实 PA `submit-pmu-claim`

#### 选型依据与 profile 身份

`submit-pmu-claim` 不是按历史 standalone phase 名单顺次补齐，而是从最新权威 Case1 泳道 `outputs/TestPagedAttentionUnroll_Case1_20260720_234305/` 的当前真实布局重新选型。该轮 Claim 为 **79,470,788 SYS_CNT ticks**，占 SubmitUnion 399,604,449 ticks 的 **19.887%**；在已经采集 arg-build 和 Materialize 后，它是剩余最大的明确 Submit 排他业务 span。该泳道中 Claim 同时为 96 核固定 1,280 次、全局 122,880 次，适合用严格固定 shape 验证局部 PMU 边界。

该 profile 的编译与设备身份固定为：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=4
PTO_FDWIC_TRACE_ENABLED=0

capture.mode                    = submit-pmu-claim
configuration.phase.id          = 4
configuration.phase.name        = claim
configuration.phase.boundary    = claim_begin_to_claim_end
configuration.phase.counter_semantics
                                = running_read_clear_observed_bracket
configuration.phase.time_semantics
                                = inner_sys_cnt_between_boundary_observers
```

公共二进制协议中的 mode 为 `5`、phase enum 为 `4`。它继续复用既有 phase sidecar：128 B header、`96 × 64 B` whole record 和 `96 × 64 B` phase record，总计 **12,416 B**；没有增加逐 Claim 记录、逐核字段或新的设备 raw 容量。

#### 四个真实边界

设备端直接在四条现存 Submit 入口上复用泳道 Claim 的业务首尾边界，没有另造近似调用：

1. 旧 Kernel `dist_submit_impl()`：在 PrepareMap 完成后的 `claim_begin` 打开，包围 `dist_submit_claim(Kernel, ...)`，在 `claim_end` 取时前关闭；
2. 旧 Alloc `dist_alloc_tensors()`：在 Register 完成后的 `claim_begin` 打开，包围 `dist_submit_claim(Alloc, ...)`，在 `claim_end` 取时前关闭；
3. compete-first Kernel begin：在 EfDrain 结束后的 `claim_begin` 打开，先执行 `dist_submit_check_task_cap()`，再执行条件 Claim，最后关闭；
4. compete-first Alloc begin：使用相同的 EfDrain.end 到 Claim.end 边界，同样包含 task-cap 检查与条件 Claim。

因此，当前真实 PA 使用的 compete-first 区间明确包含 **task-cap + Claim**；仍受支持的两个旧 API 区间只包含 Claim 主体，不应把两类入口的源码范围描述成完全相同。四条路径在正常 Case1 中都为每个 Submit 产生一次平衡的 begin/end；任一核次数、状态或边界不闭合都会被 host/分析器拒绝。

#### 运行与产物

真实 Case1 命令为：

```bash
source /home/q00473782/Ascend/cann-9.1.0-weekly-20260708/cann/set_env.sh
source /home/q00473782/.venv/bin/activate
export PTO_ISA_ROOT=/home/q00473782/atomic/private/gpt/pto-isa-ddafa
export PYTHONPATH="$PWD:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest \
  examples/a5/fully_distributed_within_core/paged_attention_unroll/test_paged_attention_unroll.py \
  --platform a5 --runtime fully_distributed_within_core --level 2 \
  --case Case1 --manual include --fdwic-profile submit-pmu-claim \
  --rounds 1 -s -v
```

每轮仍自动生成并校验：

```text
fdwic_submit_pmu_raw.json       # 逐核权威原始数据
fdwic_submit_pmu_report.html    # 同一 ELF 的阶段与整窗加工报告
```

从已有 raw 手工重建 HTML 时执行：

```bash
python -m simpler_setup.tools.fdwic_submit_pmu_report \
  outputs/TestPagedAttentionUnroll_Case1_<timestamp>/fdwic_submit_pmu_raw.json
```

#### B1 只作结构证据

两轮 B1 闭合件位于：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260721_031756/
outputs/TestPagedAttentionUnroll_CaseB1_20260721_031954/
```

两轮均为 96/96 受信记录、每核 5 次、480/480 begin/end、phase status `0x3f`，primary/shadow request/miss 精确相等、capture gap 为 0。全局 Submit 分别为 82.413 us 和 264.184 us，绝对时间明显波动；本阶段没有单独控制冷启动、取指预热和跨核到达，因此不对这段差值作原因归因。两轮只证明 mode/phase、四个挂点、固定 shape、ABI 与报告链路闭合，不作为 Claim 稳态性能结论。

#### 两轮 Case1 稳态结果

两轮完整 Case1 产物为：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_032101/
outputs/TestPagedAttentionUnroll_Case1_20260721_032244/
```

两轮均为 96 核每核 1,280 次、122,880/122,880 begin/end、phase status `0x3f`，owner 恢复、32 AIC + 64 AIV、固定 shape、数值顺序和风险阈值全部闭合；primary/shadow 96/96 精确相等，request/miss capture gap 均为 0。全局 Submit 时间范围分别为 4,994.863 us 和 4,704.936 us。

下表的时间、request、miss 都是原始 observed；三个占比只使用**同一轮、同一个 Claim ELF、同一角色**的分母。时间以 `Σsubmit_elapsed_ticks` 为分母，request/miss 分别以完整 Submit primary request/miss 为分母。

| 轮次 | 角色 | 时间 ns/call | request observed/call | miss observed/call | 时间占比 | request 占比 | miss 占比 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `032101` | ALL | 641.816 | 80.219 | 1.957 | 17.942% | 14.188% | 16.870% |
| `032101` | AIC | 311.706 | 79.437 | 0.025 | 8.862% | 13.636% | 4.324% |
| `032101` | AIV | 806.871 | 80.609 | 2.922 | 22.368% | 14.476% | 17.086% |
| `032244` | ALL | 646.708 | 80.202 | 1.951 | 18.493% | 14.182% | 16.917% |
| `032244` | AIC | 313.109 | 79.465 | 0.026 | 9.213% | 13.627% | 4.436% |
| `032244` | AIV | 813.507 | 80.571 | 2.914 | 22.940% | 14.473% | 17.133% |

两轮结果的角色差异稳定：AIV 为约 807～814 ns/call，AIC 为约 312～313 ns/call，而两者 request/call 都约为 79～81。结合独立泳道中 Claim 固定出现的 73,728 条返回型 ClaimMax Atomic，这一现象支持后续把 **AIV 角色和 atomic 等待**作为 Claim 时间重心继续核对，而不是先把差值归因于取指请求量。

这里仍有三条不可越过的解释边界：

- phase elapsed 还包含 task-cap、角色路由、条件控制、结果整理和 running bracket 的观察影响，不能把约 807 ns 或 AIC/AIV 差值命名为“纯 atomic 延迟”；
- Claim、swimlane、empty-bracket 分属不同 ELF。empty 的外层 elapsed 与 Claim 的内层 elapsed 本来也不是同一时间边界；即使 request/miss 都用 running read-clear，也受不同代码布局和缓存状态影响，绝不能跨 ELF 扣减；
- I-cache 结果只能报告本 Claim ELF 内的原始 observed、同角色占比和两轮方向性。AIV 的约 2.92 miss/call 不能减去 empty 的约 2.01～2.03 后冒充业务净 miss，`miss × 90 ns` 也仍然不是可兑现的 Submit 墙钟收益。

### 1.6 真实 PA `submit-pmu-register`

#### 选型依据与 profile 身份

`submit-pmu-register` 继续以最新权威 Case1 泳道 `outputs/TestPagedAttentionUnroll_Case1_20260720_234305/` 为选型依据。该轮普通泳道 Register 为 **47,991,560 SYS_CNT ticks**，占 SubmitUnion 399,604,449 ticks 的 **12.010%**，固定出现 122,880 次。在已经采集 arg-build、Materialize 和 Claim 后，它是下一个占比超过 10%、shape 固定且能映射到连续真实调用体的区域。

该 profile 的编译与设备身份固定为：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=5
PTO_FDWIC_TRACE_ENABLED=0

capture.mode                    = submit-pmu-register
configuration.phase.id          = 5
configuration.phase.name        = register
configuration.phase.boundary    = register_outputs_call_entry_to_return
configuration.phase.counter_semantics
                                = running_read_clear_observed_bracket
configuration.phase.time_semantics
                                = inner_sys_cnt_between_boundary_observers
```

公共二进制协议中的 mode 为 `6`，phase enum 为 `5`，两者不能混用。该构建仍复用 12,416 B 设备容量：128 B header、`96 × 64 B` whole record 和 `96 × 64 B` phase record；没有新增逐调用记录或 task-kind 字段。当前缓存身份为 `aicore-extra/32c26e06ad76d186`，最终 `aicore_kernel.o` 的 SHA256 为 `8264a6afd39815c825e0630dcbb69d9a3492d2e26989a61136f06d5e371fb750`，`.text` 为 150,608 B。该历史 v1 ELF 含 Submit PMU 整窗与 phase reader，且不含 perf-clock、普通泳道、逐 atomic 泳道记录或通用逐 task PMU 符号；这不能外推为 ABI v2 也删除了 return-ready atomic 的最小 SYS 时间 hook。raw 当前不内嵌 ELF SHA，因此该 SHA 只证明最终缓存构建身份，不把历史四轮产物表述为逐字节留档。

#### 三个真实挂点与调用体边界

设备端只在现有 `dist_submit_register_outputs()` 三个调用点的入口和返回处复用通用 phase begin/end：

1. `dist_submit_finish_kernel_tail()`：统一覆盖旧 Kernel 和 compete-first Kernel Finish，位于可选 Fanin 之后，传入 `include_existing=true`；
2. 旧 Alloc `dist_alloc_tensors()`：传入 `include_existing=false`；
3. compete-first Alloc `dist_alloc_compete_first_finish()`：同样传入 `include_existing=false`。

三处 end 都在 `TRACE_TIMESTAMP(register_end)` 之前。因此该边界只观察 `dist_submit_register_outputs()` 调用入口到返回，刻意排除前一阶段的记录发布、Register 结束时间戳和 caller 衔接；它对应普通泳道 Register 的核心调用体，**不是** 普通泳道 timestamp-to-timestamp span 的逐 tick 复制。

业务上，Kernel 的 `include_existing=true` 会按 `ctx.register_mask` 扫描 existing tensor 并插入 TensorMap；Alloc 的 `include_existing=false` 会在 helper 入口直接返回。当前 PA 每 batch 包含 1 个 Alloc 和 4 个 Kernel，所以固定 shape 中混合了 Alloc 空调用体、`register_mask=0` 的近空 Kernel 调用和真正的 TensorMap 工作。当前 raw 没有为此新增 task-kind 或逐 insert 字段，聚合结果不能命名为“单次 TensorMap insert 净成本”。

正常成功路径中，每个 Submit 恰好执行一次该 bracket：B1 为每核 5 次、全局 480 次；Case1 为每核 1,280 次、全局 122,880 次。若 ticket 或上游 Materialize 失败而提前返回，固定 calls、begin/end 和 phase status 门禁会拒绝整份 raw，不会静默发布缺边界的结果。

#### 运行与产物

```bash
source /home/q00473782/Ascend/cann-9.1.0-weekly-20260708/cann/set_env.sh
source /home/q00473782/.venv/bin/activate
export PTO_ISA_ROOT=/home/q00473782/atomic/private/gpt/pto-isa-ddafa
export PYTHONPATH="$PWD:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest \
  examples/a5/fully_distributed_within_core/paged_attention_unroll/test_paged_attention_unroll.py \
  --platform a5 --runtime fully_distributed_within_core --level 2 \
  --case Case1 --manual include --fdwic-profile submit-pmu-register \
  --rounds 1 -s -v
```

每轮自动生成并校验 `fdwic_submit_pmu_raw.json` 和 `fdwic_submit_pmu_report.html`；重建 HTML 的命令与第 1.5 节相同。

#### B1 只作结构证据

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260721_034517/
outputs/TestPagedAttentionUnroll_CaseB1_20260721_034904/
```

两轮均为 96/96 受信记录、480/480 begin/end、phase id `5`、phase status `0x3f`，primary/shadow request/miss 逐核精确相等，owner Restore 和风险阈值全部闭合。全局 Submit 分别为 80.904 us 和 267.167 us；本阶段没有分别控制冷启动、跨核到达或非 scalar-busy 等待，因而只记录 B1 绝对时间不稳定，不对差值作原因归因，也不用于 Register 稳态性能判断。

#### 两轮 Case1 稳态结果

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_035025/
outputs/TestPagedAttentionUnroll_Case1_20260721_035136/
```

两轮均为 96 核每核 1,280 次、122,880/122,880 begin/end、phase status `0x3f`，32 AIC + 64 AIV、owner Restore、mixed triplet、primary/shadow、数值顺序和计数器风险阈值全部闭合。下表所有占比只在**同一轮 Register ELF 内**计算：时间以逐核完整 Submit elapsed 为分母，request/miss 以本轮整窗 primary 为分母。

| 轮次 | 角色 | 时间 ns/call | request observed/call | miss observed/call | 时间占比 | request 占比 | miss 占比 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `035025` | ALL | 187.688 | 86.699 | 1.352 | 5.249% | 14.889% | 7.132% |
| `035025` | AIC | 187.833 | 88.301 | 0.045 | 5.390% | 14.902% | 9.492% |
| `035025` | AIV | 187.615 | 85.898 | 2.005 | 5.181% | 14.882% | 7.112% |
| `035136` | ALL | 187.879 | 86.517 | 1.392 | 5.055% | 14.884% | 7.335% |
| `035136` | AIC | 188.787 | 88.166 | 0.050 | 5.120% | 14.844% | 10.510% |
| `035136` | AIV | 187.425 | 85.692 | 2.063 | 5.023% | 14.905% | 7.308% |

两轮完整 Submit 分别为 4,688.752 us 和 5,136.513 us，而 Register 调用体的 per-call、request 和角色差异保持稳定。当前证据支持“该调用体在本诊断构建中约占逐核 Submit 时间 5.1%～5.2%，AIV 约 2.0 miss/call、AIC 约 0.05 miss/call”；不支持把普通泳道较宽 Register 的 12.010% 全部归给 RegisterOutputs。

该 profile 仍受三条解释边界约束：

- 普通泳道 Register 与本调用体边界不同，而且来自另一 ELF，不能逐 tick 对齐或相减；
- empty-bracket、Claim、Materialize 和 Register 均为独立诊断 ELF，不能用 empty 扣出“净 Register 时间/事件”；
- phase 没有局部 scalar busy，`miss × 90 ns` 只能作单核串行量级感知，不能当作可兑现的 Submit 墙钟收益。

### 1.7 真实 PA `submit-pmu-submit-transition`

#### 选型依据与 profile 身份

`submit-pmu-submit-transition` 对应最新泳道中相邻 Submit 之间的真实业务衔接：从上一次 Submit 的统一 end hook 打开，到下一次 `dist_submit_begin()` 完成并到达统一 begin hook 后关闭。它继续保留本 ELF 自己的完整 Submit primary，同时只用既有 begin/end hook 控制 running read-clear bracket；没有在各条业务 Submit 路径重复增加挂点。

该 profile 的编译与设备身份固定为：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=6
PTO_FDWIC_TRACE_ENABLED=0

capture.mode                    = submit-pmu-submit-transition
configuration.phase.id          = 6
configuration.phase.name        = submit-transition
configuration.phase.boundary    = previous_submit_end_to_next_submit_begin
configuration.phase.counter_semantics
                                = running_read_clear_observed_bracket
configuration.phase.time_semantics
                                = inner_sys_cnt_between_boundary_observers
```

公共二进制协议中的 mode 为 `7`、phase enum 为 `6`，两者不能混用。该构建仍复用 12,416 B 设备 ABI：128 B header、`96 × 64 B` whole record 和 `96 × 64 B` phase record；没有增加逐间隙记录、transition 类型或 task-kind 字段。

#### `N-1` 次数契约与聚合语义

首个 Submit 没有前驱，末个 Submit 没有后继，因此每核 `N` 次 Submit 只产生 `N-1` 个 transition bracket：

- task 0 的 begin 只启动完整 Submit PMU 整窗，不关闭不存在的前驱区间；
- 每个非末次 Submit 的 end 打开一个 transition bracket；
- 下一次非首个 Submit 完成 `dist_submit_begin()` 后，在统一 begin hook 关闭 bracket；
- 末次 Submit 的 end 只停止完整 PMU 整窗，不制造没有后继 Submit 的悬空区间。

设备 shape 状态、host 导出和 Python 分析器共用相同的预期次数口径。B1 每核 `N=5`，所以要求 4 次、全局 384/384 begin/end；Case1 每核 `N=1280`，所以要求 1,279 次、全局 122,784/122,784 begin/end。任一核仍按 `N` 次、少一次、多一次、begin/end 不平衡或 phase id 不是 6，整份结果都会被拒绝；每核少于 2 次 Submit 也不会发布 transition 结果。

当前 PA 编排中的 Kernel→Kernel、Kernel→Alloc 和 Alloc→Kernel 间隙都进入同一累计值。raw 只保存每核总 elapsed/request/miss 与总调用次数，因此报告中的 ns/request/miss per gap 是**所有相邻 Submit 间隙的加权均值**，不能进一步解释为上述任一种 transition 的独立成本。

#### 运行与产物

```bash
source /home/q00473782/Ascend/cann-9.1.0-weekly-20260708/cann/set_env.sh
source /home/q00473782/.venv/bin/activate
export PTO_ISA_ROOT=/home/q00473782/atomic/private/gpt/pto-isa-ddafa
export PYTHONPATH="$PWD:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest \
  examples/a5/fully_distributed_within_core/paged_attention_unroll/test_paged_attention_unroll.py \
  --platform a5 --runtime fully_distributed_within_core --level 2 \
  --case Case1 --manual include --fdwic-profile submit-pmu-submit-transition \
  --rounds 1 -s -v
```

每轮仍在对应输出目录自动生成并校验：

```text
fdwic_submit_pmu_raw.json       # 96 核整窗 primary 与 transition 聚合原始数据
fdwic_submit_pmu_report.html    # 同一 ELF 的 ALL/AIC/AIV 加工报告
```

已有 raw 的 HTML 重建命令与第 1.5 节相同。

#### B1 只作结构证据

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260721_042627/
outputs/TestPagedAttentionUnroll_CaseB1_20260721_042750/
```

两轮均为 96/96 受信记录、每核 5 次 Submit/4 次 gap、384/384 begin/end、phase id `6`、phase status `0x3f`，primary/shadow request/miss 逐核精确相等，owner Restore 与风险阈值全部闭合。B1 调用次数太少，会放大首次进入、取指预热和跨核到达差异；这两轮只证明 profile、`N-1` shape、ABI 和 raw→HTML 链路闭合，不用于判断 transition 的稳态性能。

#### 两轮 Case1 稳态结果

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_042914/
outputs/TestPagedAttentionUnroll_Case1_20260721_043036/
```

两轮均为 96 核每核 1,280 次 Submit/1,279 次 gap、122,784/122,784 begin/end、phase status `0x3f`，32 AIC + 64 AIV、owner Restore、mixed triplet、primary/shadow、数值顺序和计数器风险阈值全部闭合。完整 Submit 分别为 4,708.545 us 和 4,649.434 us。下表所有数据和占比都只来自**本轮同一个 SubmitTransition ELF**；每 gap 是按该角色累计 calls 加权的均值。

| 轮次 | 角色 | 时间 ns/gap | request observed/gap | miss observed/gap | 时间占比 | request 占比 | miss 占比 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `042914` | ALL | 354.560 | 134.101 | 4.815 | 10.046% | 23.349% | 28.624% |
| `042914` | AIC | 303.374 | 133.426 | 0.503 | 8.881% | 23.098% | 17.115% |
| `042914` | AIV | 380.153 | 134.438 | 6.972 | 10.601% | 23.475% | 29.337% |
| `043036` | ALL | 350.516 | 134.145 | 4.459 | 9.933% | 23.378% | 27.560% |
| `043036` | AIC | 303.185 | 134.200 | 0.494 | 8.843% | 23.214% | 16.889% |
| `043036` | AIV | 374.181 | 134.118 | 6.441 | 10.456% | 23.462% | 28.244% |

两轮的 per-gap 时间和 request 方向稳定；AIV 的时间与 miss 均高于 AIC，可作为后续核对跨 Submit scalar 衔接的观察信号。但该 phase 是三类 transition 的聚合，并且 bracket 本身会改变取指与多核到达，不能据此把 AIV/AIC 差值归给某一条业务调用或某一种间隙。

该 profile 还受以下解释边界约束：

- PMU phase 与泳道取自同源源码边界，但使用不同 ELF 和不同内部观察口径。泳道在 trace-on 构建中记录阶段时间戳，PMU 在 trace-off 构建中读取、清零 shadow counter，并累计两侧 observer 之间的内层 SYS_CNT；二者不能逐 tick 对齐或相减；
- empty-bracket 的 elapsed 是外层 SYS_CNT 包围相邻 begin/end 调用对，且来自另一个诊断 ELF；不能从 SubmitTransition 的内层 elapsed/request/miss 中扣除 empty；
- 本 phase 不提供局部 scalar busy，也不区分三种 transition。约 350～355 ns/gap 和 AIV 约 6.4～7.0 miss/gap 都是当前诊断 ELF 的原始聚合 observed，不能改写成零插桩业务净成本或可兑现的墙钟收益。

### 1.8 真实 PA `submit-pmu-efdrain-control`

#### 控制段定义与实现边界

最新排他泳道中，完整 EfDrain 同时包含 Scalar 调度控制与可能被回收执行的真实 linked Kernel。为了避免把 Kernel 执行期间混入 Scalar I-cache 归因，`submit-pmu-efdrain-control` 在四条真实 Submit 入口统一采用以下边界：

```text
drain_block_won() 前开始
    -> drain_block_won() + drain_phase_b()
drain_phase_b() 返回后结束
```

当 `drain_phase_b()` 进入 `execute_slot()` 时，观察器在 `dist_aicore_call_slot_kernel()` 紧邻前暂停、返回后恢复。因此局部结果是多个不连续控制片段之和：

```text
EfDrain-control = EfDrain 外层区间 - 真实 linked-Kernel 调用区间
```

它仍保留 fanin 检查、ring 扫描、atomic、完成发布、frontier 推进、slot 清理等 Scalar 控制逻辑。背压和 FinalDrain 也会调用 `execute_slot()`，但它们不在 EfDrain 外层 phase 内，pause helper 会在编译后的当前状态下直接返回 false，不会混入本 selector。

#### 固定容量与闭合公式

该 profile 的身份为：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=7
PTO_FDWIC_TRACE_ENABLED=0

capture.mode                    = submit-pmu-efdrain-control
configuration.phase.id          = 7
configuration.phase.name        = efdrain-control
configuration.phase.boundary    = efdrain_begin_to_end_excluding_linked_kernel_calls
```

公共 C++ capture mode 为 `8`，phase id 为 `7`，两者不能混用。设某核有 `N` 次 Submit，并在这些 EfDrain 中排除 `K` 次 linked-Kernel 调用，则设备、host 与报告端共同要求：

```text
phase_begin_reads = phase_end_reads = N + K
报告中的 phase per-call 分母 = N
```

历史 v1 中 `K` 保存在 phase record 的 `reserved[0]`，当时 `reserved[1..3]` 为零。ABI v2 继续用 `reserved[0]` 保存排除次数，同时用 `reserved[1]/[2]` 发布重建的 shadow request/miss，仅 `reserved[3]` 必须为零；每核 phase record 仍为 64 B，没有增加逐 Submit、逐 Kernel 或逐事件 raw。正式结果还必须同时闭合 96 核 phase boundary/shape/value/time/status 和 `phase_kernel_exclusion_closed_records=96`。

#### 观察效应与使用方式

每排除一次 Kernel，phase 外层会多一对 pause/resume 边界及相应 bookkeeping。ABI v2 的统一 gate 使 linked Kernel 不进入完整 Submit primary/shadow、`scalar_submit_elapsed_ticks`，也不进入命中的 phase elapsed/request/miss；return-ready atomic 则只扣 SYS 时间，不能把两种排除混为一谈。这个切分会改变诊断 ELF 布局和多核到达，只能在本 ELF 内分析同次采集的占比和方向；不能与 `none`、泳道或 perf-clock 绝对相减，也不能机械扣除 empty-bracket 后声称零观察开销下的业务净值。

运行命令沿用其他真实 phase，只替换 profile：

```bash
python -m pytest \
  examples/a5/fully_distributed_within_core/paged_attention_unroll/test_paged_attention_unroll.py \
  --platform a5 --runtime fully_distributed_within_core --level 2 \
  --case CaseB1 --manual include --fdwic-profile submit-pmu-efdrain-control \
  --rounds 1 -s -v
```

host 回归覆盖 mode/phase/provenance、`N+K` 读数闭合、旧 profile 拒绝专属字段、外层 `N` 分母和 HTML 语义。实现提交 `21e0414c` 后的真实 A5 B1 位于：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260721_115019/
  fdwic_submit_pmu_raw.json
  fdwic_submit_pmu_provenance.json
  fdwic_submit_pmu_report.html
```

该轮 96 核均为 5 次 Submit。实际只回收执行 1 次 linked Kernel：95 核 `K=0`，logical/physical core 9 的 AIC 为 `K=1`；所以全局 begin/end 均为 `96×5+1=481`，每核都满足 `N+K`。96/96 phase status 为 `0x3f`，primary/shadow request 与 miss 也逐核完全相等；owner Restore、数值顺序、风险阈值和专属 `phase_kernel_exclusion_closed_records=96` 全部通过。完整 B1 Submit 为 279.551 us，只作冷启动结构证据，不作为稳态性能。

本轮局部累计 elapsed/request/miss 分别为 64,751 ticks、56,187、3,414；相对同一 ELF 的逐核 Submit 累计值为 2.7441%/16.3993%/19.4021%。它证明不连续 control segments 已能采集且真实 Kernel 被单独排除，不证明这些 B1 比例能代表 Case1。

三件套大小为 72,951/3,124/84,377 B，SHA256 分别为 `79d6014e892b20823da039e3a2bea6c9761946b6c24444db71d7c61d16caca02`、`24be202086e9fda893012ca999073ceffa160aa9abba12861cee95414006e4d4`、`d8b2b4b77c243ac90e71fbb99084e7f5be7205f5d2a28ee224d4e56b9ed6c892`；重新加载 raw+sidecar 后渲染的 HTML 与正式文件逐字节一致。provenance 的 Git head 精确为 `21e0414c35ae7738a89f8994bfaf6870b733dea3`，extra cache key 为 `88075a1848686623`。

首次上板尝试 `..._114357/` 还暴露了一个必须保留的复现门禁：AICore 已按新 profile 重编，但预装 `libhost_runtime.so` 仍是只认识到 submit-transition 的旧缓存，host init 返回 0。最终实现因此在冻结 provenance 前先核验实际 host ELF 的三个 hook 和精确 profile marker；旧 host 现在会在设备执行前明确要求重建，而不是留下无 raw 的失败目录。重建后本轮 host SHA256 为 `441e54ac3d997e110de792d6597b8cf47d31a764ccf9bc63551387b9a597b919`。

#### Case1 首轮稳态原因数据

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_115559/
```

该轮 golden 通过，完整 Submit 为 4,840.463 us。96 核均为 1,280 次 Submit；EfDrain 内实际排除 936 次 linked Kernel，逐核 `K` 为 3～19，AIC/AIV 分别累计 429/507 次。全局 begin/end 都是 `96×1280+936=123816`，96/96 primary/shadow request 与 miss 逐核完全相等，其他 producer/consumer/provenance 门禁也全部闭合。

同一 ELF 内的局部归因如下；时间占比的分母是各角色逐核 Submit elapsed 之和，request/miss 占比的分母是各角色完整 Submit primary：

| 角色 | control 时间/call | 时间占比 | request/call | request 占比 | miss/call | miss 占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 162.654 ns | 4.5920% | 82.110 | 14.0406% | 2.258 | 17.0455% |
| AIC | 199.792 ns | 5.7719% | 83.945 | 14.0450% | 0.110 | 28.6828% |
| AIV | 144.085 ns | 4.0219% | 81.193 | 14.0383% | 3.332 | 16.9316% |

完整 Submit 的 AIC/AIV 每核平均 request 为 765,037/740,310，平均 miss 为 493/25,187.344，miss rate 为 0.06444%/3.40227%。AIV 的 EfDrain-control 每核平均 miss 为 4,264.625，也就是它贡献本轮 AIV 完整 Submit miss 的 16.93%。这回答的是 “平均每个 AIV 在本诊断 ELF 的完整 Submit 期看到多少 miss”，不是零观察构建的墙钟损失。

按 90 ns/miss 只作串行直觉量尺，AIV 完整窗约为 2,266.861 us/core，局部 control 约为 383.816 us/core；但局部实际 elapsed 只有 184.428 us/core。量尺大于被观察阶段时间本身，直接反证了把 `miss×90ns` 当作可相减停顿的做法：miss 可重叠、被流水隐藏，标尺也来自另一个隔离微基准。另一个重要现象是 AIC control 时间/call 反而高于 AIV，而 AIC miss/call 低两个数量级；因此当前单轮不支持“I-cache miss 主导 EfDrain-control 时间”，后续应把 atomic/同步等待和纯控制指令一并纳入解释。

Case1 三件套大小为 76,269/3,124/86,600 B，SHA256 分别为 `6d6aa06fbf972f36fbb9260485bb5ee41f5cbb97e9246cb15a398ba2497201ce`、`b4a535ec671f7416945b5205e11268308068759e75ecc324a10ea2a57fb3fa03`、`f5751e86f503273d1b8f8e386301d1d84dfdc49ee6de45d7372199ce90a68841`；provenance Git head 为 `77df395941f86b3b546a6f50d6288fb88acb7078`，实际 ELF SHA 与 B1 相同，离线重渲染也逐字节一致。

### 1.9 真实 PA `submit-pmu-prepare-map`

最新排他泳道把 PrepareMap 定义为 Materialize 结束后到 `dist_submit_prepare_map()` 返回并取得 `prepare_map_finish` 的区间。四条 Submit 入口都汇聚到 `dist_submit_materialize_and_prepare_map()`，因此 PMU profile 在该 helper 内只包围真实调用体：

```text
fdwic_submit_pmu_phase_begin<PrepareMap>()
dist_submit_prepare_map(self, task_id)
fdwic_submit_pmu_phase_end<PrepareMap>()
```

profile 身份为：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=8
PTO_FDWIC_TRACE_ENABLED=0

capture.mode                 = submit-pmu-prepare-map
configuration.phase.id       = 8
configuration.phase.name     = prepare-map
configuration.phase.boundary = dist_submit_prepare_map_call_entry_to_return
```

PrepareMap 每个 Submit 固定调用一次，因此 CaseB1 每核严格 5 对、Case1 每核严格 1,280 对 begin/end。它继续复用 64 B phase sidecar 和 12,416 B 固定总容量，没有增加 GM 字段、逐 Submit record 或逐事件 raw。运行方式为：

```bash
python -m pytest \
  examples/a5/fully_distributed_within_core/paged_attention_unroll/test_paged_attention_unroll.py \
  --platform a5 --runtime fully_distributed_within_core --level 2 \
  --case Case1 --manual include --fdwic-profile submit-pmu-prepare-map \
  --rounds 1 -s -v
```

B1 结构回归位于 `outputs/TestPagedAttentionUnroll_CaseB1_20260722_143131/`：golden 通过，96 核共 480 对边界，所有 phase status 为 `0x3f`，raw/provenance/HTML 闭合。完整 Case1 位于 `outputs/TestPagedAttentionUnroll_Case1_20260722_143242/`：golden 通过，96 核共 122,880 对边界，所有 producer/consumer/provenance 门禁通过。

Case1 同一 ELF 内的直接观察结果如下。时间占比以各角色 `Σsubmit_elapsed_ticks` 为分母；request/miss 占比以同次采集、同角色的 primary 为分母：

| 角色 | elapsed/call | 时间占比 | request observed 总数 | 逐核 request min–max | request 占比 | miss observed 总数 | 逐核 miss min–max | miss 占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 80.603 ns | 2.3078% | 10,095,628 | 103,633–106,941 | 13.9869% | 1,636 | 7–39 | 0.1767% |
| AIC | 81.468 ns | 2.3930% | 3,365,879 | 103,752–106,338 | 13.6735% | 314 | 7–15 | 4.0391% |
| AIV | 80.171 ns | 2.2668% | 6,729,749 | 103,633–106,941 | 14.1491% | 1,322 | 10–39 | 0.1440% |

这组数只说明独立 PMU ELF 中 `dist_submit_prepare_map()` 调用体的观察结果。泳道 PrepareMap 还包含相邻取时与 trace record 边界，而 submit-PMU 编译期去除了 trace；两类 ELF 的代码布局和多核时序也不同。因此不能拿本轮 2.3078% 与泳道占比直接相减来声称某段 record 成本，也不能把各 phase 的 request/miss 跨 ELF 相加成完整 Submit。

三件套 raw/provenance/HTML 的 SHA256 分别为 `62aa8f74aaa439f088963e4e3a768bbac4053ab292f392fc8c39a49b81995ae8`、`2fe946082ffbc9f4a91f9f5d1cdda38398be19e2c3a2f607df1a48f268b0c365`、`288cfc4238290ba11558f26f6e4798d43d13b2cca583e65002ca5a50a5dc2ebf`。

### 1.10 真实 PA `submit-pmu-fanin` 与动态调用闭合

Fanin 只在 Kernel winner 路径执行，winner 落在哪个 worker 由多核竞争决定，不能继续套用“每核固定 N 次”的 phase shape。该 profile 的源码边界与泳道继承边界一致：legacy Kernel winner 从 Claim.end 打开；compete-first Kernel winner 从 PrepareMap.end 打开；二者都在 `dist_submit_collect_fanin()` 返回、取得 fanin_end 前关闭。PMU ELF 编译掉中间 trace record，但仍保留同一业务起止位置。

身份为：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=9
PTO_FDWIC_TRACE_ENABLED=0

capture.mode                 = submit-pmu-fanin
configuration.phase.id       = 9
configuration.phase.name     = fanin
configuration.phase.boundary = fanin_begin_to_fanin_end
configuration.phase.call_shape = dynamic_balanced
```

设每核 Submit 数为 `N=5B`。当前 PA 每 batch 固定一个 Alloc 和四个 Kernel task，四个 Kernel task 的唯一 winner 分别有两个落在 AIC、两个落在 AIV。因此 producer、consumer 与报告端分别复算：

```text
逐核：begin_reads == end_reads，允许为 0，且不超过 2B
全局：Σcalls = 4B
AIC： Σcalls = 2B
AIV： Σcalls = 2B
```

零调用核必须满足 elapsed/request/miss observed 都为 0，但完整 Submit 末尾的 shadow tail read 仍可能更新 max chunk，因此不能错误要求 max chunk 为 0。实际调用数继续使用已有 `phase_begin_reads/end_reads`，没有新增 GM/header/record/reserved 字段；phase 构建仍为 12,416 B。host 只有在逐核平衡、逐核上界和全局/角色公式全部闭合后，才发布 `phase_global_call_count_closed=true` 的 raw。

B1 结构回归位于 `outputs/TestPagedAttentionUnroll_CaseB1_20260722_145237/`：golden 通过，实际 `4=2(AIC)+2(AIV)`，92 个 worker 为零调用，逐核 0～1 次，所有 96 核 phase status 为 `0x3f`。完整 Case1 位于 `outputs/TestPagedAttentionUnroll_Case1_20260722_145338/`：golden 通过，实际 `1024=512+512`，AIC 逐核 11～23 次、AIV 逐核 3～15 次，所有门禁闭合。

Case1 同一 ELF 内的观察结果为：

| 角色 | 调用数 | elapsed/call | 时间占比 | request observed | request 占比 | miss observed | miss 占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 1,024 | 990.228 ns | 0.2438% | 354,845 | 0.5837% | 8,284 | 1.2911% |
| AIC | 512 | 552.570 ns | 0.2122% | 126,334 | 0.6004% | 437 | 5.4212% |
| AIV | 512 | 1,427.885 ns | 0.2588% | 228,511 | 0.5749% | 7,847 | 1.2386% |

这里的百分比仍只使用 Fanin 这一独立 ELF 自己的 Submit elapsed/primary 作为分母。它不能与 PrepareMap、Claim 等其他 profile 横向求和；AIV per-call 明显高于 AIC 也只是当前诊断 ELF 的直接现象，尚不能仅凭单轮归因为 I-cache miss。

Case1 三件套 raw/provenance/HTML 的 SHA256 分别为 `0df28acdd7a429b92736447d25344e697cced0bb486de6b26518ce784c92a563`、`9d2d907d3225236e0c1645fe892a4030ed9929027e8a644983cf47a8068e75ab`、`48d5cb8f0781eedd43f55e437fbe340d68428531416f18da1d8dde8a19f1ef50`。

### 1.11 真实 PA `submit-pmu-winner-build-control`

WinnerBuild 只在 Kernel winner 路径执行。该 profile 在 Register 结束后、进入 winner 尾段前打开，
在 `dist_submit_build_winner_task()` 返回后关闭，对应完整 WinnerBuild 业务边界：

```text
capture.mode                    = submit-pmu-winner-build-control
configuration.phase.id          = 10
configuration.phase.name        = winner-build-control
configuration.phase.boundary    = winner_build_begin_to_end_excluding_linked_kernel_calls
configuration.phase.call_shape  = dynamic_balanced
```

设每核 Submit 数为 `N=5B`。它与 Fanin 使用同一组 Kernel winner 动态公式，但逐核不能伪造固定次数：

```text
逐核：outer_calls = phase_begin_reads - phase_excluded_kernel_calls
      outer_calls = phase_end_reads - phase_excluded_kernel_calls
      两侧相等，允许为 0，且不超过 2B
全局：Σouter_calls = 4B
AIC： Σouter_calls = 2B
AIV： Σouter_calls = 2B
```

WinnerBuild 内回收执行的 linked vector/cube Kernel 通过统一 gate 从 phase 时间和 PMU counter
同时排除，`phase_excluded_kernel_calls` 记录相应的 `K`。消费返回值的 return-ready/result-used
atomic 依赖区间只从 `scalar_submit_elapsed_ticks` 及命中的 phase elapsed 扣除，不停止 PMU；
I-cache/PMU counter 仍包含 atomic 与最小 hook 的指令事件，source-issue atomic 也继续保留。

B1 结构回归位于：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260722_211158/
```

该轮业务调用严格闭合为 `4=2(AIC)+2(AIV)`，排除 linked Kernel 为 `0`。96/96 记录的
owner/selector/status、linked-Kernel gate、return-ready atomic 时间、phase boundary/shape/value/time、
shadow-primary bounded 和动态全局调用数全部闭合；它只提供动态 shape 的上板结构证据。

完整 Case1 位于：

```text
outputs/TestPagedAttentionUnroll_Case1_20260722_211624/
```

该轮业务调用为 `1024=512(AIC)+512(AIV)`；phase 内共排除 53 次 linked Kernel，全部落在
AIC，AIV 为 0。严格 producer/consumer/provenance 门禁与 B1 相同，均已闭合。同一 ELF 内的
直接观察如下；时间列是 96 核累计 core-time，时间占比的分母是同角色
`Σscalar_submit_elapsed_ticks`，request/miss 占比的分母是同次采集、同角色的完整 primary：

| 角色 | calls | 排除 Kernel | phase core-time sum | Scalar 时间占比 | request observed / 占比 | miss observed / 占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 1,024 | 53 | 10,031.073 us | 3.4716228% | 3,561,057 / 5.0471511% | 33,047 / 1.8154869% |
| AIC | 512 | 53 | 7,213.506 us | 7.9254424% | 2,611,337 / 10.4338074% | 11,544 / 11.8110478% |
| AIV | 512 | 0 | 2,817.567 us | 1.4235334% | 949,720 / 2.0860070% | 21,503 / 1.2483281% |

`10,031.073 us` 是跨核累计的 phase core-time，不是 Submit 墙钟。该 WinnerBuild ELF 的全局
Submit 墙钟为 4.599385 ms；另一次 `submit-pmu-none` 为 4.982069 ms，但两者来自不同诊断
ELF 和独立进程，不能相减成 0.382684 ms 的 WinnerBuild 收益，也不能据此杜撰任何优化收益。
本节两轮均为 ABI/schema v2；旧 v1 HTML 和数值只能作为历史记录，不能与这里的计数或比例混用。

### 1.12 真实 PA `submit-pmu-alloc-complete-control`

AllocComplete 只在 Alloc winner 路径执行。legacy API 从 Claim.end 打开，compete-first API 从
Register.end 打开，均在 `dist_submit_complete_alloc()` 返回后关闭，对应完整 AllocComplete
Scalar 控制边界：

```text
capture.mode                    = submit-pmu-alloc-complete-control
configuration.phase.id          = 11
configuration.phase.name        = alloc-complete-control
configuration.phase.boundary    = alloc_complete_begin_to_end_excluding_linked_kernel_calls
configuration.phase.call_shape  = dynamic_global
```

设每核 Submit 数为 `N=5B`。每 batch 只有一个 Alloc winner，但其核角色由竞争结果决定，不能把
AIC/AIV 次数伪造成固定比例：

```text
逐核：outer_calls = phase_begin_reads - phase_excluded_kernel_calls
      outer_calls = phase_end_reads - phase_excluded_kernel_calls
      两侧相等，允许为 0，且不超过 B
全局：Σouter_calls = B
角色：AIC outer_calls + AIV outer_calls = B；两者分别不锁定
```

HeapGuard 慢路径若回收并执行 linked vector/cube Kernel，统一 gate 会同时从 phase elapsed 和
PMU counter 排除 Kernel 整段，并用 `phase_excluded_kernel_calls` 记录额外边界对。消费返回值的
return-ready/result-used atomic 依赖区间只从 `scalar_submit_elapsed_ticks` 和命中的 phase elapsed
扣除，不停止 PMU；因此 request/miss 仍包含 atomic 指令及最小 hook 的取指事件，source-issue
atomic 也继续保留。本次 B1 和 Case1 的 phase 内排除 Kernel 次数都为 0。

B1 与完整 Case1 的三件套分别位于：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260722_215826/
outputs/TestPagedAttentionUnroll_CaseB1_20260722_215826/fdwic_submit_pmu_report.html
outputs/TestPagedAttentionUnroll_Case1_20260722_220027/
outputs/TestPagedAttentionUnroll_Case1_20260722_220027/fdwic_submit_pmu_report.html
```

B1 严格闭合 `1=0(AIC)+1(AIV)`，95 个 worker 为零调用；这只是一次实际 winner 分布，不是角色
公式。Case1 严格闭合 `256=255(AIC)+1(AIV)`，同样不能把 255/1 固化成后续运行的期望值。两轮
96/96 trusted、linked-Kernel gate、return-ready atomic 时间、phase boundary/shape/value/time、
shadow-primary bounded 和动态全局调用数门禁全部闭合。

Case1 同一 ELF 内的直接观察如下。phase ticks 是跨核累计 core-time，Scalar 时间占比的分母是
同角色 `Σscalar_submit_elapsed_ticks`，不是全局 Submit 墙钟：

| 角色 | calls | phase elapsed ticks | request observed | miss observed | Scalar 时间占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ALL | 256 | 584,689 | 170,344 | 8,790 | 0.1968758%（约 0.197%） |
| AIC | 255 | 580,355 | 169,762 | 8,724 | 0.6376930% |
| AIV | 1 | 4,334 | 582 | 66 | 0.0021041% |

`584,689 ticks` 即 96 核累计的 584.689 us，不是 AllocComplete 墙钟耗时；0.197% 也只描述该
AllocComplete ELF 的 phase core-time 相对同一 ELF Scalar Submit 分母的比例。不得与 none、
WinnerBuild 或其他独立诊断 ELF 的时间和计数相加、相减，也不能据此杜撰独立性能收益。

### 1.13 真实 PA `submit-pmu-loser-replay`

LoserReplay 只在 Kernel loser 路径执行。legacy 与 compete-first Kernel Submit 最终共用同一个
tail：在 Register 结束边界打开 bracket，调用 `drain_block_won()`，函数返回后立即关闭。因此该
profile 的身份为：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=12
PTO_FDWIC_TRACE_ENABLED=0

capture.mode                    = submit-pmu-loser-replay
configuration.phase.id          = 12
configuration.phase.name        = loser-replay
configuration.phase.boundary    = register_end_to_drain_block_won_return
configuration.phase.call_shape  = dynamic_balanced
```

设每核 Submit 数为 `N=5B`。当前 PA 每 batch 有四个 Kernel task；96 个 worker 都回放这四次
Kernel Submit，每个 task 只有一个 winner，且四个 winner 中两个落在 AIC、两个落在 AIV。
LoserReplay 是对应 Kernel winner 集合的补集，因此 producer、consumer 和分析器共同复算：

```text
逐核：outer_calls = phase_begin_reads - phase_excluded_kernel_calls
      outer_calls = phase_end_reads - phase_excluded_kernel_calls
      两侧相等，允许动态分布，且不超过 4B
全局：Σouter_calls = (96 × 4 - 4)B = 380B
AIC： Σouter_calls = (32 × 4 - 2)B = 126B
AIV： Σouter_calls = (64 × 4 - 2)B = 254B
```

`drain_block_won()` 只把已经发布的 joint lane 搬入本核 RingSlot，不调用
`drain_phase_b()`/`execute_slot()`，所以当前 LoserReplay 边界内不会执行 linked vector/cube
Kernel，`phase_excluded_kernel_calls` 必须为 0。其通用 joint 路径仍可能执行消费返回值的
`WonAnyLoad`、`WonStateLoad` 和 `WonLaneClaimExchange`：这些 return-ready atomic 的依赖等待时间
沿 ABI v2 口径从 `phase_elapsed_ticks` 和 `scalar_submit_elapsed_ticks` 扣除，但 PMU gate 不停止，
request/miss 仍包含 atomic 指令及最小时间 hook 的取指事件。返回值不消费的 source-issue atomic
继续保留在 Scalar 控制口径中。

B1 与完整 Case1 的 raw/provenance/HTML 三件套分别位于：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260722_222358/fdwic_submit_pmu_raw.json
outputs/TestPagedAttentionUnroll_CaseB1_20260722_222358/fdwic_submit_pmu_provenance.json
outputs/TestPagedAttentionUnroll_CaseB1_20260722_222358/fdwic_submit_pmu_report.html

outputs/TestPagedAttentionUnroll_Case1_20260722_222533/fdwic_submit_pmu_raw.json
outputs/TestPagedAttentionUnroll_Case1_20260722_222533/fdwic_submit_pmu_provenance.json
outputs/TestPagedAttentionUnroll_Case1_20260722_222533/fdwic_submit_pmu_report.html
```

B1 严格闭合 `380=126(AIC)+254(AIV)`，逐核 3～4 次。Case1 严格闭合
`97,280=32,256(AIC)+65,024(AIV)`，全体逐核 1,005～1,020 次。两轮均为 96/96
trusted，owner/selector/status、linked-Kernel gate、return-ready atomic 时间、phase
boundary/shape/value/time、shadow-primary bounded 和动态全局/角色调用数门禁全部闭合。

Case1 同一 ELF 内的直接观察如下。phase ticks 是 96 核累计 core-time，Scalar 时间占比的分母是
同角色 `Σscalar_submit_elapsed_ticks`：

| 角色 | calls | phase elapsed ticks | request observed | miss observed | Scalar 时间占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ALL | 97,280 | 5,021,979 | 8,218,072 | 453,234 | 1.2719749% |
| AIC | 32,256 | 1,696,385 | 2,539,793 | 4,148 | 1.5188385% |
| AIV | 65,024 | 3,325,594 | 5,678,279 | 449,086 | 1.1745909% |

这是高频 observer 边界：Case1 执行 97,280 对 begin/end read-clear。`phase_elapsed_ticks` 使用
`inner_sys_cnt_between_boundary_observers`，不把两次 counter reader 自身当成业务时间；但
`phase_icache_requests/misses_observed` 是 `running_read_clear_observed_bracket`，会包含边界附近的
状态检查、累计 bookkeeping 和 observer 取指。高频固定开销会随调用次数累积，因此表中 request/miss
不能解释成无观察器 LoserReplay 函数体的数学下界，也不能与其他 phase ELF 的 observed 相加。

本轮全局首个 Submit 到末个 Submit 为 5.256747 ms，只用于证明该诊断运行的全局窗口闭合；它不是
上述 96 核累计 phase ticks 的分母，也不能与 none 或其他独立 ELF 的墙钟相减成 LoserReplay 收益。

## 2. 为什么 I-cache miss 必须独立重编译

I-cache 数据对代码布局极其敏感。泳道和逐 atomic 观察会增加：

- 阶段和 atomic 的 `SYS_CNT` 读取；
- atomic wrapper、ClockBaseline 与 record 发布分支；
- 更大的 scalar `.text` 和不同的函数/对齐布局；
- 由此引起的 worker 到达、轮询和跨核竞争时序变化。

因此，“代码仍在 ELF 中，只在运行时关闭 record”不足以得到干净的 I-cache 观察。`submit-pmu` 会独立重编译，编译掉泳道 record、逐 atomic 泳道记录慢体、ClockBaseline、runtime phase-profile 和旧 cold/warm 冲刷体。ABI v2 仍保留 return-ready/result-used atomic 的最小 SYS 时间扣除 hook；它不落泳道记录，也不门控 PMU counter。`swimlane` 则不构建 PMU owner。

两种数据不在同一进程采集：

- 看事件时序和 atomic bracket，使用 `swimlane`；
- 看 Submit-all 整窗每核 I-cache request/miss，使用 `submit-pmu`。

两份证据可以按同一源码版本交叉理解，不能做逐 tick 对齐，也不能把 PMU 的平均 miss 回填为某一条 atomic span 的属性。

### 2.1 运行时关闭不等于编译期去除：真实 PA 回退案例

真实 PA 已出现过一次完整反例：level 1 的 raw 中没有任何 Atomic 或 ClockBaseline 记录，但同一 ELF 为了允许运行时切到 level 4，仍把 atomic wrapper、PollBatch 遍历和记录发布慢体大量内联进 Submit 热函数。结果不是 “没有写 record 就没有代价”，而是未执行的诊断代码仍改变 `.text`、基本块布局和取指工作集。

当时真实 A5 Case1 的证据链为：

| 构建状态 | AIC/AIV `dist_engine .text` | AIC/AIV `dist_submit_impl` | 三轮首末 Submit 中位数 |
| --- | ---: | ---: | ---: |
| pre-atomic 历史构建 | 80,824 / 80,912 B | 42,136 / 42,152 B | 5.115620 ms |
| atomic 观测接入、level 1 不落 Atomic | 347,360 / 355,968 B | 100,724 / 102,588 B | 5.631038 ms |
| 两处低频 winner 调整为冷分支 | 347,536 / 357,112 B | 100,860 / 103,676 B | 5.192087 ms |
| atomic 冷代码共享外提 | 66,768 / 67,120 B | 18,812 / 18,872 B | 4.821897 ms |

最后一行正式三轮为 4.821897/4.890447/4.752956 ms。这里能证明的是 “代码复制和布局回退已被消除”；没有同时采集 I-cache PMU，因此不能把恢复量进一步写成某个确定的 miss 降幅。

冷代码外提遵守两条边界：

1. direct atomic 的 begin、真实 atomic、返回值地址依赖和 end 仍留在原 wrapper，只共享 end 之后的 record 发布。因此不把 source-issue 操作改成等待返回型，atomic span 口径也不变。
2. PollBatch 保留内联的 `level >= 4 && active_mask != 0` 快速判断，只把命中后的十类遍历与落盘外提。level 1 不新增 call/ret，level 4 的 `end_cycle` 仍在冷函数调用前取得。

保留此类修改前应依次检查：AIC/AIV 对象的 `.text` 和独立符号、level-1 多轮 Submit、level-4 logical/physical Atomic 公式、ClockBaseline 和 `dropped_records=0`。上述真实 PA level-4 复核得到 115,309 次 atomic 调用、107,608 条 Atomic、8,056 次批处理轮询和 355 条 PollBatch，满足 `107608 = 115309 - 8056 + 355`。

这个案例同时说明为什么 `submit-pmu` 不能只传运行时 level=0：专门分析 I-cache 时，普通泳道、atomic 泳道记录 wrapper、ClockBaseline 和相关慢体必须在编译期从待测 AIC/AIV ELF 中剔除；ABI v2 的最小 return-ready 时间 hook 是当前测量合同的一部分，不属于应删除的泳道慢体。运行时 gate 只控制“执行没有”，不能控制“代码存在没有”。

### 2.2 Claim-first eager 重编后的观察结论

真实 PA 切换到 compete-first eager 后，Submit 的阶段顺序变为 `EfDrain -> Claim -> Materialize -> PrepareMap -> Fanin/Register -> 尾阶段`。这类热路径重排会同时改变基本块与跨 TU 的代码布局，因此它的性能结论应记录在 I-cache 观察指南中，不归因为某个 atomic 本身的收益。

迁移后的 atomic 观察仍包围原位置的真实指令：消费返回值的调用继续使用 `return_ready` 边界，不消费返回值的 Exchange/FetchAdd 继续使用 `source_issue` 边界，PollBatch 的逻辑调用与物理压缩记录口径也没有改变。Claim 前移只改变业务阶段顺序，没有把 atomic 记录提前、延后或改写成另一种完成语义。

最终真实 A5 level-4 复核位于：

~~~text
outputs/TestPagedAttentionUnroll_Case1_20260720_104406/
~~~

该轮包含 122,880 个 Submit、945,653 条事件，`dropped_records=0`。Atomic 物理记录、逻辑调用、轮询调用和 PollBatch 记录满足：

~~~text
106355 = 109392 - 3361 + 324
~~~

raw 转换、阶段顺序和整数闭合均通过，schema 能完整解析 site/op、`result_used` 与 `return_ready`。这些结果只用于证明热路径重排后的观察能力和计数口径仍然正确；该轮没有同时采集专用 I-cache PMU，不能据此推导 I-cache miss 降幅或某个 atomic 的独立收益。

## 3. Standalone 历史 `none` 与局部 phase 如何选择

从本节到第 11 节主要描述 `tests/atomic_probe/pa_scheduler` standalone 的历史 schema-v5、构建目录、命令和字段；不能套用到上面的真实 PA profile。真实 PA 当前 profile 为：

- `submit-pmu-none`：完整 Submit 整窗；
- `submit-pmu-arg-build`：Claim.end 到 Materialize.begin 的同步构参区间；
- `submit-pmu-empty-bracket`：第 1.3 节定义的观察器经验校准，不是业务 phase；
- `submit-pmu-materialize`：第 1.4 节定义的真实 Materialize 业务 span；
- `submit-pmu-claim`：第 1.5 节定义的真实 Claim 业务 span；
- `submit-pmu-register`：第 1.6 节定义的 RegisterOutputs 调用体；
- `submit-pmu-submit-transition`：第 1.7 节定义的所有相邻 Submit 间隙聚合；
- `submit-pmu-efdrain-control`：第 1.8 节定义的排除 linked Kernel 的不连续 Scalar 控制段；
- `submit-pmu-prepare-map`：第 1.9 节定义的 `dist_submit_prepare_map()` 调用体；
- `submit-pmu-fanin`：第 1.10 节定义的 Kernel winner 动态 Fanin 区间；
- `submit-pmu-winner-build-control`：第 1.11 节定义的完整 WinnerBuild 业务边界内、排除 linked Kernel 的 Scalar 控制段；
- `submit-pmu-alloc-complete-control`：第 1.12 节定义的完整 AllocComplete 业务边界内、排除 linked Kernel 的 Scalar 控制段；
- `submit-pmu-loser-replay`：第 1.13 节定义的 Kernel loser `Register.end` 到 `drain_block_won()` 返回区间。

真实 phase id 依次为 ArgBuild `1`、EmptyBracket `2`、Materialize `3`、Claim `4`、Register `5`、SubmitTransition `6`、EfDrainControl `7`、PrepareMap `8`、Fanin `9`、WinnerBuild `10`、AllocComplete `11` 和 LoserReplay `12`；`none` 不含 phase。SubmitTransition 的 outer calls 为 `N-1`，所有 phase 的实际 begin/end 读数都要加本阶段内被统一排除的 linked Kernel 数 `K`；Fanin 与 WinnerBuild 使用 `4B/2B/2B` 的角色平衡动态公式，AllocComplete 只锁定全局 `B` 次而不锁定 AIC/AIV 分布，LoserReplay 使用每 batch `380/126/254` 的全局/AIC/AIV 补集公式且逐核不超过 `4B`，其余业务/校准 phase 都为每核 `N`。

当前十三种 profile 的新 raw schema 均为 `fdwic-submit-pmu-v2`，但分别来自独立诊断 ELF，不能跨 profile 相减或拼接。此前已经生成的 `fdwic-submit-pmu-v1` 文件只作历史记录，新 loader 不兼容读取。

standalone 历史 schema 中的 `lower/upper` 是已经固化的字段名，只表达 read-clear observed 与 `primary-shadow` capture gap；由于 bracket 两侧也有观察 bookkeeping，它们同样不能解释为零插桩业务事件数的数学上下界。

standalone 当前保留五个编译期 phase：

| phase | 边界 | 优先用途 |
| --- | --- | --- |
| `none` | Submit-all 整窗中不读局部 shadow counter | 回答整个调度回放期的 AIC/AIV 每核 request/miss；这是默认选择 |
| `claim` | `Claim()`、结果写回 context 及 claim 本地统计前后读局部 shadow counter | 当 `none` 已证明 miss 值得追踪时，试验 Claim 的 running read-clear 下界/上界归因链路 |
| `efdrain` | 每次 Submit 开头唯一的 `DrainReady(...EfDrain...)` 前后 | 观察 opportunistic drain；不混入 RingBackpressure 或 FinalDrain |
| `materialize` | `MaterializeTask()` 及成功路径 `materialized_outputs` 本地统计前后 | 观察输出 descriptor/layout、本地 register mask、输出字节数和 heap 游标等 scalar 工作；不包含后续 slot payload 拷贝 |
| `register` | 每次 Submit 统一的 `RegisterOutputs()`，非 Alloc 还包含 `map_inserts` 本地统计 | 观察输出注册语义体 |

普通泳道为了减少观察扰动，会让相邻阶段复用同一个 `SYS_CNT` 边界。PMU-only ELF 不生成这些泳道时间戳；局部 PMU bracket 只包围同一语义体，使用自己的 shadow read-clear 和 `SYS_CNT`，不做跨 ELF 的逐 tick 对齐。

running phase 的 begin/end 读取本身会执行 scalar 指令、占用取指并改变多核时序；schema-v5 还在每次 begin/end 各读取一次 1 ns/tick 的 SYS_CNT。因此：

- running phase 的 phase request/miss 是带局部边界扰动的观察值；
- running phase 的累计时间也是带边界扰动的直接观察值；起点在 begin 的 shadow read-clear 之后，终点在 end 的 shadow read-clear 之前，所以不包含两侧 `ld_dev`，但包含每次调用两次 SYS_CNT 的观察扰动；
- request/miss raw 观察值是下界，上界为该核下界加 primary-shadow loss；阶段时间是单点观察值，没有伪造的上下界；
- `none` 和任一 running phase 是不同 ELF、不同进程，不能以两者相减声称得到了零扰动的局部净值；
- 未来不同 phase ELF 的局部 request/miss 不可相加成 whole gate；Submit-all 整窗始终以每个 ELF 自己的 primary whole 为准。

`none`/`claim`/`efdrain` 与正式 swimlane 使用 block-local runtime state 和跨 TU noinline finish；Claim/EfDrain 边界在 finish 之前已闭合。`materialize`/`register` 为了在 finish 内继续操作同一份真实 `PmuContext`，当前使用 inline-finish 诊断 ELF。后两者与 `none` 不具备字节级相同的指令布局，它们的局部结果只能在各自 ELF 内解释。

该区间只描述同一插桩 ELF、当前边界定义下的局部事件，不是无插桩局部阶段的真实区间。Submit-all `none` 没有运行中 read-clear，仍执行 96/96 逐核严格闭合。

报告中有三个相关但不相同的时间窗，不能都简称为“完整 Submit”：

1. **PMU whole gate**：每核在 `InitPaOrchestration()` 之前 `metrics_prof_start()`，在末次 UP `SubmitCallbackTask()` 返回后立即 stop。它包含 orchestration 初始化、`EfDrain`、`Claim`、同步 eager 构参、finish、`AcceptTaskOutputs()` 和 Submit 间输出接收/调用衔接，排除 FinalDrain；与泳道 `OrchestrationReplay` 父区间接近但不做逐 tick 对齐。CNT6/CNT7 primary 和 PMU total/scalar-busy 都使用这个窗。
2. **`submit_elapsed_ticks`**：每核从首个 `BeginCallbackSubmit()` 之后到最后一个 Submit 的 `submits++` 之后，包含两端之间的 Submit 间衔接，但不包含 whole gate 首尾的 orchestration 外围。局部 phase 时间占比以它为分母。
3. **`configuration.submit_span_us`**：96 核中最早 `submit_begin` 到最晚 `submit_end` 的全局墙钟范围，不是逐核 PMU 时间的平均值。

## 4. 环境、构建与产物

命令在 `tests/atomic_probe/pa_scheduler` 目录下执行。非交互 shell 建议显式 source CANN 并选择本用户 GCC 15：

```bash
source /home/q00473782/Ascend/cann-9.1.0-weekly-20260708/cann-9.1.0/set_env.sh
export GCC15_ROOT=/home/q00473782/.local/gcc-15/root
export PATH="$GCC15_ROOT/usr/bin:$PATH"
export LD_LIBRARY_PATH="$GCC15_ROOT/usr/lib/x86_64-linux-gnu:$GCC15_ROOT/usr/lib/gcc/x86_64-linux-gnu/15${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CXX="$GCC15_ROOT/usr/bin/g++-15"
export PYTHON=/home/q00473782/.venv/bin/python
```

构建 Submit-all 整窗基准：

```bash
./run.sh build-submit-pmu ccec none
```

需要 Claim 局部归因时另行构建：

```bash
./run.sh build-submit-pmu ccec claim
```

需要 EfDrain 局部归因时独立构建：

```bash
./run.sh build-submit-pmu ccec efdrain
```

需要 Materialize 局部归因时独立构建：

```bash
./run.sh build-submit-pmu ccec materialize
```

需要 Register 局部归因时独立构建：

```bash
./run.sh build-submit-pmu ccec register
```

产物分别位于：

```text
build/ccec/submit-pmu/none/
build/ccec/submit-pmu/claim/
build/ccec/submit-pmu/efdrain/
build/ccec/submit-pmu/materialize/
build/ccec/submit-pmu/register/
```

每个 phase 目录中的 `pa_scheduler_host`、`pa_scheduler_kernel.o`、`libpa_scheduler_pmu_owner_aicpu.so` 和 `libpa_scheduler_pmu_owner_dispatcher.so` 是一个不可拆分的构建集。host 会按 kernel 所在目录加载两个 SO；不得从另一个 phase 目录复制或拼接产物。构建只在四件套全部完成后原子发布 `submit_pmu_artifacts.manifest`；`run.sh` 在启动 host 前核对 schema、phase、固定文件列表和四个 SHA256。

`swimlane` 的 CCEC 产物仍在 `build/ccec/`，不是 PMU 产物。

## 5. 采集 Submit-all 整窗与局部 phase

### 5.1 Submit-all 整窗 `none`

```bash
OUT="./outputs/submit_pmu_none_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"

./run.sh submit-pmu ccec none \
  --device 0 --batches 1 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1 \
  --pmu-json "$OUT/submit_icache_raw.json"
```

多轮比较必须使用多个独立进程和独立子目录；每个采集目录内部都保持同一组描述性文件名：

```bash
mkdir -p "$OUT/capture_02" "$OUT/capture_03"

./run.sh submit-pmu ccec none --device 0 --batches 1 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1 \
  --pmu-json "$OUT/capture_02/submit_icache_raw.json"

./run.sh submit-pmu ccec none --device 0 --batches 1 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1 \
  --pmu-json "$OUT/capture_03/submit_icache_raw.json"
```

### 5.2 Claim 局部归因

```bash
OUT_CLAIM="./outputs/submit_pmu_claim_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_CLAIM"

./run.sh submit-pmu ccec claim \
  --device 0 --batches 1 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1 \
  --pmu-json "$OUT_CLAIM/submit_icache_raw.json"
```

### 5.3 EfDrain 局部归因

```bash
OUT_EFDRAIN="./outputs/submit_pmu_efdrain_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_EFDRAIN"

./run.sh submit-pmu ccec efdrain \
  --device 0 --batches 1 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1 \
  --pmu-json "$OUT_EFDRAIN/submit_icache_raw.json"
```

`efdrain` 每核固定调用 `batches * 5` 次；b1 的 AIC/AIV/global calls 分别为 160/320/480。插点只位于 Submit 开头的 EfDrain 专属 call-site；复用的 `DrainReady()` 函数体不插桩。

### 5.4 Materialize 局部归因

后续边界和 96 核闭合验证固定使用 b1：

```bash
OUT_MAT_B1="./outputs/submit_pmu_materialize_$(date -u +%Y%m%dT%H%M%SZ)_b1"
mkdir -p "$OUT_MAT_B1"

./run.sh submit-pmu ccec materialize \
  --device 0 --batches 1 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1 \
  --pmu-json "$OUT_MAT_B1/submit_icache_raw.json"
```

边界位于 `MaterializeTask()` 唯一调用点前后。实现必须先保存返回值、关闭 phase，再处理失败返回，避免失败路径留下 begin/end 不平衡。每核固定 `5 * batches` 次；b1 的 AIC/AIV/global calls 为 160/320/480。

2026-07-19 A5 历史实测闭环如下；四轮均满足 capture accepted、语义、PMU 和 phase measurement PASS。b256 数据只作归档证据，不是后续迭代的重跑要求：

| phase | batches | calls/expected | begin/end 与 call shape | primary=shadow | shadow≤primary | request/miss loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `materialize` | 1 | 480/480 | 96/96 | 94/96 | 96/96 | 2/0 |
| `materialize` | 256 | 122,880/122,880 | 96/96 | 74/96 | 96/96 | 160/8 |
| `register` | 1 | 480/480 | 96/96 | 93/96 | 96/96 | 2/1 |
| `register` | 256 | 122,880/122,880 | 96/96 | 36/96 | 96/96 | 2,834/654 |

对应 raw/HTML 位于 `outputs/submit_pmu_{materialize,register}_20260719_b{1,256}/`；raw 是权威取数件，HTML 是同目录的加工展示件。

running read-clear 的硬门禁是 96/96 `shadow≤primary`，不是要求 96/96 逐值相等；表中的 loss 已进入每核局部 lower/upper 区间，不能被静默忽略。

### 5.5 Register 局部归因

```bash
OUT_REG_B1="./outputs/submit_pmu_register_$(date -u +%Y%m%dT%H%M%SZ)_b1"
mkdir -p "$OUT_REG_B1"

./run.sh submit-pmu ccec register \
  --device 0 --batches 1 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1 \
  --pmu-json "$OUT_REG_B1/submit_icache_raw.json"
```

Register 已合并为每次 Submit 唯一的 `RegisterOutputs()` 调用点；Alloc 通过 `include_existing=false` 保留语义差异，winner/loser 都经过该边界。因此仍是每核固定 `5 * batches`，调用规模与 Materialize 相同。它与普通泳道包围同一 Register 语义体，但两个 ELF 各自取时，不做逐 tick 对齐；其中较短或未实际插入 map 的调用仍会放大 PMU begin/end 边界扰动，解释结果时必须使用 lower/upper，而不能把 observed lower 当成无扰动净开销。

`submit-pmu` action 已固定：

```text
--runs 1 --no-swimlane --pmu-window submit-all
```

调用者不要重复传入这三项，也不能添加 `--profile-phases`、`--trace-atomics`、`--analyze-swimlane` 或 `--swimlane-json`。`--pmu-json` 可选；但要做 raw 复算、HTML 可视报告和多轮汇总时必须使用它。host 拒绝覆盖已有 JSON 或同名 `.tmp`。

raw 成功发布后，`run.sh` 会调用本目录的独立分析器并在同一目录生成：

```text
submit_icache_raw.json       # 96 核权威原始件及 host summary
submit_icache_report.html    # 可离线浏览的加工件
```

HTML 使用内联 CSS/SVG，不依赖外部前端库；浏览器直接打开即可。它包含 Submit-all PMU 整窗的 AIC/AIV 对比、逐物理核 request/miss/rate 分布、96 核明细和 90 ns core-equivalent 提示。报告同时展示 AIC/AIV/ALL 的 PMU raw `total_cycles`、CNT2 `scalar_busy`、`Σscalar/Σtotal` 以及 “非 Scalar-busy 残余”，并在保留 raw cycle 的同时给出校准后的每核等效时间。顶部的“完整 Submit”固定表示**最早一次 Submit 进入到最晚一次 Submit 返回**；它是 96 核共同形成的整体墙钟范围，不等于逐核 PMU whole gate 的平均值；对应到每个 worker，首末 Submit 边界也窄于该核的 PMU whole gate。ALL/AIC/AIV 响应式卡片中，I-cache request/miss 与 90 ns 直觉量尺只展示逐核 `min/max`，不展示 mean；其他 PMU 时间类字段仍保留现有聚合口径。HTML 从通过 96 核门禁的 `records` 对称推导 `min/max`；PMU total 与 scalar busy 各自独立取极值，不保证来自同一个物理核。较宽的 I-cache 对比表和 96 核明细只在表格内部横向滚动，不再撑宽整页。

本机 A5 受控 cold/warm 同窗校准得到：

```text
PMU cycle_delta = 1,817,457
SYS_CNT tick_delta = 1,101,593 ns
ALL = 1.649844 cycles/ns
AIC = 1.650062 cycles/ns
AIV = 1.649731 cycles/ns
time_us = PMU cycles / (cycles_per_ns * 1000)
```

ALL/AIC/AIV 汇总分别使用对应频率，逐核明细按该核角色使用 AIC 或 AIV 频率。其中 total 是每个物理子核在 PMU whole gate 内的累计周期，96 核求和是 core-work，不是 Submit 墙钟；“非 Scalar-busy 残余” 严格等于 `total−scalar_busy`，既不是 Scalar 空闲时间，也不是 I-cache stall，其中还包含同步等待、vector/cube engine 等待以及其他未归因周期。受控微基准中，依赖返回的 atomic 等待大部分进入 scalar busy，而 I-cache refill 的额外周期大部分只进入 total；这个现象不能把二者之差提升为 I-cache 专属计数器。

对于局部 phase，HTML 最前面先按 ALL/AIC/AIV 展示 calls、阶段时间占比、request/miss 占比；后面再列阶段时间/core、阶段时间/call、request/miss 下界—上界和 shadow loss。阶段时间占比是同一角色内 `Σphase_elapsed_ticks / Σsubmit_elapsed_ticks`，request/miss 则仍是占同一 ELF PMU whole-gate primary 的比例区间。两类占比的分母边界不同，只能分别用于时间和 I-cache 归因，不能把它们当成同一精确分区。`none` 明确显示“不适用”，历史 schema-v4 因没有阶段时间 raw 字段而显示“不可用”，不会伪造 0%。报告生成失败时 action 返回非零，但已成功发布的 raw 会保留用于排查。

## 6. Primary/shadow 计数和可信门禁

### 6.1 计数器分工

早期曾尝试用 external task-based `msprof` 汇总配合 kernel 内 start/stop 取得 Submit 子窗口，实测 raw counter 不受该门控缩窗，因此不能用于局部归因。现行链路由 standalone 自带的 Main AICPU owner 保存、配置、读回并恢复每个物理子核 PMU，kernel 只在同一 runtime TU 内控制 gate 和发布 worker 独占结果。任何 owner membership、selector readback 或 Restore 失败都会拒绝最终 JSON。

| 计数 | selector | 用途 |
| --- | --- | --- |
| PMU raw total | 固定 64-bit total low/high | PMU whole gate 内每个物理子核的累计周期；不是 96 核求和后的墙钟；HTML 另按实测频率显示等效时间 |
| CNT2 | `0x001` | scalar instruction busy cycle；不包含全部等待周期；HTML 另按实测频率显示等效时间 |
| CNT6 | `0x34` | PMU whole-gate primary I-cache request；局部边界从不读它 |
| CNT7 | `0x35` | PMU whole-gate primary I-cache miss；局部边界从不读它 |
| CNT8 | `0x34` | read-to-clear shadow request |
| CNT5 | `0x35` | read-to-clear shadow miss；诊断 ELF 因此不再提供 MTE3 busy |
| CNT9 | `0x0` | 未使用 |

A5 b1 实测已证明将 `0x35` 配置到 CNT9 时计数始终为 0，所以 CNT9 不能作 shadow miss。这个变化只影响 `submit-pmu` 诊断构建，不影响 `swimlane` 构建。

#### A5 `scalar_wait_ib_time` 的支持边界

2026-07-19 使用本机 CANN 9.1 在同一 A5/DAV3510 上对 standalone b1 依次验证了 `PipeUtilization`、`PipeUtilization,MemoryDetail` 和 `Default` 三种正式 `msopprof` 采集入口。三次均能生成 `PipeUtilization.csv`，但表头都不包含 `aic/aiv_scalar_wait_ib_time`，也不包含 `aic/aiv_scalar_wait_time`。本机 CANN 9.1 `CHIP_V6_MAP` 与本仓 DAV3510 正式事件表同样只给出已使用的 `scalar_busy(0x001)`、I-cache request `0x34`、I-cache miss `0x35` 等事件，没有 wait-IB/wait 的 selector 或派生公式。

三次原始证据分别保存在：

```text
outputs/wait_ib_official_msopprof_20260719_b1_probe2/
outputs/wait_ib_official_msopprof_20260719_b1_probe3_memory_detail/
outputs/wait_ib_official_msopprof_20260719_b1_probe4_default/
```

因此当前 A5 正式可编程路径的结论是：**不能采集这两个指标**。CANN 共享 `msopprof` 二进制包含相应字段字符串、官方文档也在 A2/A3 产品章节解释其含义，但这些证据不能推出 DAV3510 selector。不得把旧架构或其他产品的事件号套到 A5。若后续 CANN/A5 正式事件表新增这两项，必须重新用 scalar NOP、I-cache warm/cold、真实 Vector/Cube `PIPE_* -> PIPE_S` wait 和依赖 atomic 四组对照校准后再纳入报告。

#### A5 真实 Vector 流水的 `scalar_busy` 归类验证

2026-07-22 在同一台 A5/DAV3510、CANN 9.1 weekly 20260708 上新增了单 AIV
受控探针，专门回答“执行真实 Vector 计算时，PMU total 中是否以
`scalar_instr_busy(0x001)` 为主”。探针源码和独立 runner 为：

```text
tests/atomic_probe/ccec/vector_scalar_pmu.cpp
tests/atomic_probe/ccec/vector_scalar_pmu_host.cpp
tests/atomic_probe/ccec/vector_scalar_pmu_shared.h
tests/atomic_probe/ccec/run_vector_scalar_pmu.sh
```

运行命令为：

```bash
cd tests/atomic_probe/ccec
./run_vector_scalar_pmu.sh
```

测试基线是 `fdwic-swimlane-deps@7bd59a8f`。被测 `VECTOR_ADD` 循环与
`pa_scheduler/ccec/ccec_ops.h` 的 `RunRealVectorWorkload<false>` 保持相同
流水顺序，tile 为 `128 x 128 x float`：

```text
TLOAD(A) + TLOAD(B)
MTE2 -> V set/wait
TADD
V -> MTE3 set/wait
TSTORE
MTE3 -> S set/wait
```

每轮从 GM 读取 128 KiB、写回 64 KiB；最后的 `MTE3 -> S wait_flag`
保证本轮写回完成后才进入下一轮。PMU gate 只包住三种可替换工作负载：

- `EMPTY`：空窗口，用于确认 gate 固有成本；
- `LOOP_CONTROL`：相同运行时循环次数，每轮只有一个 scalar NOP，不向
  Vector/MTE 发工作；
- `VECTOR_ADD`：上面的完整真实流水。

探针直接复用 standalone 已验证的 10-slot PMU owner，而不是新增一套寄存器
协议。该构建固定 `PA_BUILD_SUBMIT_PMU=0`，所以同一窗口内的 selector 是
`CNT0=vector(0x501)`、`CNT2=scalar(0x001)`、`CNT4=MTE2(0x202)`、
`CNT5=MTE3(0x203)`、`CNT6=request(0x034)`、`CNT7=miss(0x035)`；kernel
逐项回读得到 `selector_status=0x3f` 后才接受样本。每个工作量先各模式预热
一次，再采 5 个 `EMPTY -> LOOP_CONTROL -> VECTOR_ADD` 配对样本。五轮都落在
同一物理 AIV 18，16,384 个输出元素都严格等于 `2.0 + 3.0 = 5.0`，I-cache
miss 都为 0，96 个 active owner slot、32 个完整 `1 AIC + 2 AIV` triplet
和最终 Restore 全部通过。

本机 `task-submit` 与 `npu-smi` 均不在 `PATH`，本轮按用户明确给出的
“没有 `npu-smi` 直接运行”许可执行，因此属于 unlocked 采样。五轮同窗计数
高度稳定，足以回答本节的 PMU 状态归类问题；这些绝对耗时不作为跨进程或跨设备
性能基线。

5 轮中位数如下。`non-scalar residual` 只按定义计算
`total - scalar_busy`：

| rounds | mode | SYS_CNT | PMU total | scalar busy | non-scalar residual | scalar/total | vector busy | MTE2 busy | MTE3 busy |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | `EMPTY` | 45 ns | 66 | 62 | 4 | 93.9394% | 0 | 0 | 0 |
| 16 | `LOOP_CONTROL` | 186 ns | 299 | 299 | 0 | 100.0000% | 0 | 0 | 0 |
| 16 | `VECTOR_ADD` | 31,312 ns | 51,659 | 262 | 51,397 | **0.5072%** | 14,976 | 23,233 | 12,869 |
| 128 | `EMPTY` | 45 ns | 66 | 62 | 4 | 93.9394% | 0 | 0 | 0 |
| 128 | `LOOP_CONTROL` | 1,212 ns | 1,993 | 1,993 | 0 | 100.0000% | 0 | 0 | 0 |
| 128 | `VECTOR_ADD` | 239,067 ns | 394,488 | 1,718 | 392,770 | **0.4355%** | 119,808 | 185,881 | 84,519 |

两个工作量的 `VECTOR_ADD - LOOP_CONTROL` 配对中位数斜率也一致：

| rounds | PMU total/轮 | scalar busy/轮 | vector busy/轮 | MTE2 busy/轮 | MTE3 busy/轮 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 3,210.0000 | -2.3125 | 936.0000 | 1,452.0625 | 804.3125 |
| 128 | 3,066.3672 | -2.1484 | 936.0000 | 1,452.1953 | 660.3047 |

这里的负 `scalar busy/轮` 不是“Vector 产生负周期”，而是
`LOOP_CONTROL` 的 scalar 循环在几乎整个短窗口内都 busy；真实 Vector 循环发出
异步指令后大部分时间不满足 `scalar_instr_busy` 条件，所以其 scalar 增量反而
略低。频率也得到独立交叉检查：16 轮的 `51,659 / 1.65 = 31,308.5 ns`，
128 轮的 `394,488 / 1.65 = 239,083.6 ns`，分别与 1 GHz SYS_CNT 的
31,312 ns 和 239,067 ns 对齐。

本次可以下的结论是：**对这条带每轮最终 `MTE3 -> S wait_flag` 的 TADD
真实流水，Scalar 并非大部分时间处于 `scalar_busy`；`scalar_busy` 只占
0.44%～0.51%，`total - scalar_busy` 占 99.49% 以上。** 因而该路径上的
Vector/MTE 执行和等待没有被 `0x001` 大量记入 scalar busy。完整 Submit 曾观察到
的约 70% AIV scalar ratio 不能外推到这个 Vector 循环，更不能据此判断 I-cache
miss；它还包含 PA 调度、atomic、轮询和其他 scalar 控制路径。

边界同样必须保留：本轮只验证了 TADD 分支，没有验证 TMUL/Cube；
`vector_busy`、`MTE2 busy`、`MTE3 busy` 可能彼此或与 scalar 状态重叠，不能求和
还原 total；`total - scalar_busy` 仍只能叫“非 Scalar-busy 残余”，其中同时包含
Vector/MTE 执行、DMA/存储等待、同步等待和流水空隙，不能进一步改名为
`wait_flag` 时间、Scalar idle 或 I-cache stall。

shadow PMU counter 是 read-to-clear。任一 running phase 在阶段 begin 读取 CNT8/CNT5，将之前的片段加入 shadow whole；在 end 再读一次，同时加入 shadow whole 和所选 phase；PMU whole gate stop 后读 tail。`none` 不做中途读取，只在 stop 后取 tail。

`none` 对每个物理子核必须精确满足：

```text
shadow_whole_icache_requests == icache_requests
shadow_whole_icache_misses   == icache_misses
```

即 stop 后读取的 CNT8/CNT5 分别等于同 selector、同 gate 的 CNT6/CNT7。这个 96/96 精确相等门禁验证 whole-gate 观察闭合；它不把 PMU 进程的 Submit span 变成无诊断墙钟基线，也不把 standalone 数据冒充真实 PA profile。

历史 A5 b1/b256 取证已证明运行中切片可能发生单向少计，接受规则为：

```text
shadow_request <= primary_request
shadow_miss    <= primary_miss

request_loss = primary_request - shadow_request
miss_loss    = primary_miss - shadow_miss

phase_request_lower = phase_request_observed
phase_request_upper = phase_request_observed + request_loss
phase_miss_lower    = phase_miss_observed
phase_miss_upper    = phase_miss_observed + miss_loss
```

上下界必须先逐核计算，再分别聚合；不能拿聚合后的 median 相减拼区间。CNT8/CNT5 是顺序 `ld_dev`，不是同一时刻的原子配对快照，因此局部 `phase_miss <= phase_request` 不是硬门禁。二者分别不超过对应 shadow，各自上界不超过对应 primary。

结合 `none` exact、running phase bounded 和调用次数门禁，可分别验证：

- 复制 selector 在本机 A5 上确实计数；
- 边界调用覆盖预定代码片段，且 begin/end/tail 次数闭合；边界竞态少计由 loss 和 lower/upper 区间显式保留；
- primary whole 没有被局部归因读取破坏。

### 6.2 正式 JSON 必须通过的门禁

只有下列条件全部成立，host 才发布最终 JSON：

- 语义、winner 真计算输出和 Submit placement/engine 闭合通过；
- 96 条记录可信，32 AIC + 64 AIV，物理子核 id 唯一；
- owner bitmap membership、worker slot、物理 role 和 32 个 1:2 triplet 全部匹配；
- 96 个核都真实执行 Submit-all PMU whole-gate start/stop，owner Restore 成功；
- build variant 和编译 phase id 在 96 条记录中全部匹配；
- `none` 的 shadow whole 与 primary whole 逐核精确相等；
- running phase 的 shadow whole 逐核不大于 primary，loss 与 upper-bound 公式逐核闭合；exact 核数只作诊断；
- `none` 的 phase calls/begin/end/request/miss 全部为 0；
- `claim`、`efdrain`、`materialize`、`register` 的 begin/end/calls 逐核平衡，每核 calls 为 `batches * 5`，全局 calls 为 `batches * 5 * 96`；
- 四个 running phase 必须分别命中自己的边界：Claim 调用、Submit 开头 EfDrain 专属 call-site、Materialize 唯一调用点，以及每次 Submit 统一的 Register 调用点；
- phase request/miss 分别不超过对应 shadow/primary，且可编程 counter 低于当前 25% 保守风险阈值。
- 96 个核的 `submit_elapsed_ticks` 都大于 0；running phase 的 `phase_elapsed_ticks` 大于 0 且不超过本核 `submit_elapsed_ticks`，`none` 则必须精确为 0；phase time 状态位和 host 复核都必须通过。

`metrics_prof_start/stop()` 在 PMU whole gate 前后各执行一次，其 `PIPE_ALL` 边界会改变流水和多核时序。PMU 结果只与相同构建、相同 phase、相同负载的独立进程比较，不把 PMU 进程的 Submit span 当作无诊断性能基线。

## 7. JSON 字段与 AIC/AIV 分析口径

当前 `submit-pmu` 输出 schema v5；分析器继续只读兼容历史 schema-v4。`records` 保留 96 个 worker 的 raw，`summary.all/aic/aiv` 分别对 96/32/64 个核统计：

```text
sum / mean / median / p95 / max
```

Submit-all 整窗优先查看：

- `configuration.submit_span_us`：本轮从第一个 Submit 进入到最后一个 Submit 返回的整体 span；HTML 顶部换算成毫秒展示；
- `total_cycles`：每核 PMU whole gate 内的 64-bit PMU raw total；按角色的 sum 是 core-work，raw 仍保留 mean/median/p95；HTML 只选 mean 表示典型单核，并辅以从逐核记录得到的 min/max，三者均不等于 host 看到的 Submit 墙钟。HTML 按 ALL/AIC/AIV 的实测频率将 raw cycle 换算为单核 cycle-equivalent；
- `scalar_busy`：CNT2 `scalar_instr_busy(0x001)`，表示 scalar instruction busy cycle；依赖返回的 atomic 等待可进入此项，但它不是“纯算术指令数”；换算后的时间也只是 scalar-busy cycle-equivalent；
- `icache_requests` / `icache_misses`：CNT6/CNT7 PMU whole-gate primary；
- `shadow_whole_icache_requests` / `shadow_whole_icache_misses`：闭合或分段 loss 用 shadow whole；
- `shadow_request_loss` / `shadow_miss_loss`：本核 primary-shadow residual；
- `phase_calls` / `phase_icache_requests` / `phase_icache_misses`：选定 phase 的 running read-clear lower；
- `submit_elapsed_ticks`：本 worker 从首个 `submit_begin` 计时点到末个 `submit_end` 计时点的 SYS_CNT 差值；起点位于首个 `BeginCallbackSubmit()` 上下文初始化之后，终点位于末个 Submit 返回之前；1 tick = 1 ns；
- `phase_elapsed_ticks`：所选 phase 所有调用的 SYS_CNT 差值累计；running phase 必须非零且不超过同核 `submit_elapsed_ticks`，`none` 必须为 0；
- `phase_icache_requests_upper_bound` / `phase_icache_misses_upper_bound`：lower 加本核对应 loss；
- `configuration.compiled_phase` 和 `validation.phase_measurement_valid`：确认文件口径。
- `validation.shadow_primary_match_records` / `shadow_primary_bounded_records`：区分逐值 exact 与单向 bounded 核数；
- `configuration.phase_values_are_running_read_clear_lower_bounds`：确认局部字段是否采用下界语义。

HTML 中展示的“非 Scalar-busy 残余/core”严格等于 `(total−scalar_busy)/core`，只用于观察未被 scalar busy 覆盖周期的数量级。它可能同时包含 I-cache refill、同步等待、vector/cube engine 等待和其他流水空隙，不能命名为 Scalar 空闲或 I-cache stall，也不能用它反推单次 miss 代价。

离线分析器与 HTML 会从这些 raw 字段继续派生 `phase_icache_request_lower_bound_share_of_submit`、`phase_icache_request_upper_bound_share_of_submit`、`phase_icache_miss_lower_bound_share_of_submit` 和 `phase_icache_miss_upper_bound_share_of_submit`：局部 lower/upper 分别除以同一角色、同一次采集的 PMU whole-gate primary 总数。

`phase_observed_read_clear_ratio` 只是 lower miss/lower request 的观测比值；分子和分母各有独立区间，因此它不是实际 phase miss rate 的数学下界。

每核平均值按角色求：

```text
AIC PMU total/core = summary.aic.total_cycles.sum / 32
AIC scalar/core    = summary.aic.scalar_busy.sum / 32
AIV PMU total/core = summary.aiv.total_cycles.sum / 64
AIV scalar/core    = summary.aiv.scalar_busy.sum / 64
AIC request/core = summary.aic.icache_requests.sum / 32
AIC miss/core    = summary.aic.icache_misses.sum / 32
AIV request/core = summary.aiv.icache_requests.sum / 64
AIV miss/core    = summary.aiv.icache_misses.sum / 64
```

PMU cycle 的时间换算通式和默认校准值为：

```text
time_us = cycles / cycles_per_ns / 1000
ALL cycles_per_ns = 1.649844
AIC cycles_per_ns = 1.650062
AIV cycles_per_ns = 1.649731
```

这里的 `time_us` 是每核 cycle-equivalent。`total−scalar_busy` 即使换算为时间，仍不是 Scalar 空闲时间或 I-cache stall；96 核 sum 换算后也仍是 core-work，不能冒充 Submit 墙钟。

`median` 和 `p95` 直接来自同角色逐核 raw 分布，用于观察典型核和高尾核。I-cache miss rate 只按组内加权口径计算：

```text
AIC miss rate = Σ(AIC miss) / Σ(AIC request)
AIV miss rate = Σ(AIV miss) / Σ(AIV request)
```

不平均 32 或 64 个逐核百分比。AIC/AIV 核数不同，比较每核强度时使用 mean/median/p95 或 miss rate，不直接比较两组 sum。

局部 phase 的时间和 I-cache 占比都使用组内总量，但分母边界不同：

```text
time share          = Σphase_elapsed_ticks / Σsubmit_elapsed_ticks
request share lower = Σphase_request_lower / Σprimary_request
request share upper = Σphase_request_upper / Σprimary_request
miss share lower    = Σphase_miss_lower / Σprimary_miss
miss share upper    = Σphase_miss_upper / Σprimary_miss
```

时间分子、分母来自同一个 1 ns SYS_CNT，是逐核累计 core-time 构成；不能用 `Σphase_elapsed_ticks` 除以 96 核共同形成的 `submit_span_us`。时间是直接观察单值，不仿造 request/miss 那样的 lower/upper。多个独立进程聚合时，分析器展示每轮上述 `Σ/Σ` 比值的分布，不把不同轮的 raw 重新拼成一次虚构运行。

分子与分母必须来自同一个 phase ELF、同一轮采集和同一角色。该比例回答“当前插桩 ELF 中局部窗口占自身对应分母的多少”，不能拿 `claim` 分子除以另一份 `none` 的分母。

## 8. HTML 报告与多轮分析命令

单份 raw 的 HTML 已由 `submit-pmu` action 自动生成。需要手工重建时：

```bash
PYTHON=/home/q00473782/.venv/bin/python

"$PYTHON" ./pmu_html_report.py \
  "$OUT/submit_icache_raw.json"
```

默认输出为同目录的 `submit_icache_report.html`；也可用 `-o` 指定路径。生成器先调用 `pmu_sidecar_analyzer.py` 的完整 raw 门禁，只有 96 核拓扑、raw→summary、primary/shadow、采集接受状态和 owner Restore 全部通过才发布 HTML。

使用本用户 Python 环境从 raw 重算 host summary，并聚合相同配置的多个独立进程：

```bash
PYTHON=/home/q00473782/.venv/bin/python

"$PYTHON" ./pmu_sidecar_analyzer.py \
  "$OUT/submit_icache_raw.json" "$OUT"/capture_*/submit_icache_raw.json
```

需要机器可读汇总时：

```bash
"$PYTHON" ./pmu_sidecar_analyzer.py --json \
  "$OUT/submit_icache_raw.json" "$OUT"/capture_*/submit_icache_raw.json \
  > "$OUT/summary.json"
```

分析器会先逐份复算 96 条 raw 与 host summary，然后拒绝聚合下列混用：

- 任意不同 phase，例如 `none` 与 `claim`，或 `materialize` 与 `register`；
- 不同 schema/build variant；
- 不同 batches、winner mode/count/pattern、selector 或观察开关。

建议将每次采集的 JSON 和分析器生成的 summary 放在同一唯一目录；`outputs/` 为本机证据目录，不作为源码提交的一部分。

## 9. 如何使用分角色的单次 I-cache miss 标尺

### 现行参考：2026-07-22 单子核严格 cold/warm 校准

此前统一使用的约 `90 ns/miss` 来自 96-worker mixed launch 的聚合试验。为消除各核同时执行 64 KiB capacity sweep 带来的取指争用，并区分 AIC/AIV 角色，2026-07-22 在提交 `b04f592b` 的干净 detached worktree 中复用 `tests/atomic_probe/ccec/icache_scalar_pmu` 的 PMU owner、host 校验和 cold/warm 配对框架，做了单物理子核校准。该校准没有修改真实 PA 热路，也没有把临时探针接入正式 `submit-pmu` 构建。

最终 AIC/AIV ELF 都通过以下运行前硬门禁：

- target 固定为 `8 B @ 0x0`，按 128 B I-cache line 对齐，只占一个 16 B fetch block；
- 布局固定为 `target -> prepare -> warm harness -> evictor`。AIC/AIV 的 harness 均为 `172 B @ 0x180`，evictor 均为 `65,604 B @ 0x280`；
- 64 KiB 以上的 evictor 同时超过本机模型配置中的 AIC 32 KiB、AIV 16 KiB scalar I-cache 容量；
- warm/cold 交替采用 `WARM,COLD` 和 `COLD,WARM` 顺序，每一对必须落在同一 physical subcore；任何一对不满足 `warm CNT7=0 && cold CNT7=1` 都立即判失败，不进入统计。

AIC 与 AIV 各执行一次 101 对试验，合计 404/404 个 cold/warm 样本的 checksum、mode echo、PMU 关窗、physical subcore 和 owner Restore 全部通过；两组都是 101/101 对严格增加一次 CNT7 miss。结果如下：

| 角色 | physical subcore | 有效配对 | SYS_CNT cold-warm 均值 | SYS_CNT 中位数 / 范围 | PMU total cold-warm 均值 | PMU total 中位数 / 范围 | PMU cycle 按 1.65 GHz 换算 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AIC | 0 | 101/101 | **77.376 ns** | 77 ns / 76～79 ns | **127.733 cycles** | 128 / 127～130 cycles | 77.414 ns |
| AIV | 18 | 101/101 | **94.030 ns** | 94 ns / 93～96 ns | **155.287 cycles** | 155 / 154～157 cycles | 94.113 ns |

1 ns/tick 的 SYS_CNT 原始分布为：AIC `76:5, 77:58, 78:33, 79:5`，AIV `93:15, 94:69, 95:16, 96:1`。PMU total cycle 分布为：AIC `127:34, 128:61, 129:5, 130:1`，AIV `154:7, 155:61, 156:30, 157:3`。两套时基换算后的差异均小于 0.1 ns，互相验证了约 1.65 cycles/ns 的当前频率口径。

每一对的额外 request 都严格为 4、额外 miss 严格为 1；额外 `scalar_instr_busy(0x1)` 则严格为 0。它证明本试验中的单次回填等待形成 PMU total/scalar-busy gap，而不是增加 scalar busy。request 的 `+4` 与本机模型配置的取指加三条顺序预取相符，但只有 target 所在行产生 CNT7 miss，后续 warm harness 行没有再 miss。

因此，当前受控口径不应再把 AIC/AIV 折成同一个 90 ns 常数。对真实 PA 已分别聚合出的角色 miss 数，应计算：

```text
隔离串行 core-work 等效量(ns) = 77.376 * M_AIC + 94.030 * M_AIV
分角色加权标尺(ns/miss) =
    (77.376 * M_AIC + 94.030 * M_AIV) / (M_AIC + M_AIV)
```

若仅为了与旧校准比较，假设 32 个 AIC 和 64 个 AIV 每核 miss 数相同，则 worker 加权结果为 `88.479 ns/miss`；对应 PMU 加权均值为 `146.102 cycles/miss`，按 1.65 GHz 为 `88.547 ns/miss`。旧 `90 ns` 相对此口径高 `1.521 ns`，约 `1.72%`。如果真实 PA 的 AIC/AIV miss 比例不是 1:2，就必须使用实际 `M_AIC/M_AIV`，不能套用 `88.5 ns`。

这里的“单次 miss 时间”严格限定为：**一个物理子核、容量驱逐后、CNT7 恰好增加 1 的 target 完成时间 cold-warm 增量**。本轮固定采样 physical AIC 0 和 AIV 18，没有轮转全部 96 个活跃子核，因此表中范围是同一子核的时间分布，不是核间 min/max。它也没有控制下一级 refill 命中位置，不等于连续顺序取指的 miss 吞吐，更不是 Submit 已经暴露的墙钟 stall。真实热路中的 compulsory/capacity/conflict miss、预取、同核流水重叠和跨核并行仍可能改变可见代价。

### 历史参考：96-worker 聚合得到的约 90 ns

2026-07-18 的旧 `icache-single` 在同一个 mixed launch 中让 32 个 AIC 和 64 个 AIV 各自执行 64 KiB sweep，再聚合全部 worker。64 trials/core × 10 轮和 128 trials/core × 5 轮都满足：

```text
cold CNT7 miss == trials
warm CNT7 miss == 0
miss_delta == 96 * trials
calibrated_cores == 96/96
```

前者 ALL 中位数为 86.596 ns/miss，范围 86.532～86.792；后者中位数为 89.629 ns/miss，范围 89.615～89.648。AIC/AIV 差值方向还随 trials 规模改变，所以这组数据只保留为“96 核并发驱逐条件下”的历史数量级证据，不再用来建立现行角色常数。历史原始日志为：

```text
pa_scheduler/outputs/pmu_validation/icache_single_64x10_20260718_085929_3232836_console.log
pa_scheduler/outputs/pmu_validation/icache_single_128x5_20260718_090151_3235468_console.log
```

当前 `pmu_sidecar_analyzer.py` 和 `pmu_html_report.py` 的单一 `--icache-miss-ns` 默认值仍为 `90.0`，用于保持已生成历史报告和命令接口的口径。本节只记录更严格的分角色校准结果，不静默回填旧 HTML、旧表格或旧 JSON，也不把单一 CLI 参数伪装成已经支持 AIC/AIV 两套常数。

无论使用历史 90 ns 还是现行分角色标尺，所得结果都只能称为隔离串行 core-work 等效量。不得从 Submit 墙钟中直接扣除，原因包括：

- 64 个 AIV、32 个 AIC 之间并行；
- 同一核的 miss、预取、其他流水和等待可能重叠；
- 隔离 cold miss 与真实热路 capacity/conflict/compulsory miss 不一定同价；
- 当前已核实的 A5 事件中没有可直接换算墙钟损失的 I-cache stall-cycle counter。

要测真正暴露的性能损失，必须对同语义代码做交错 A/B：

1. 用相同 `submit-pmu none` 口径确认 `ΔAIC/AIV miss/core`；
2. 另用不开 PMU/泳道的性能构建测 `ΔSubmit span`；
3. 只有第 2 项是实际暴露的墙钟收益；第 1 项只用于证明收益与 I-cache 变化同时出现，分角色单 miss 标尺仅提供数量级解释。

### 9.1 历史 b256 `none` 参考数据

2026-07-19 用当时的 `submit-pmu none` ELF、`real-compute/6,28,4,1` 在 A5 上采集一轮，全局首末 Submit span 为 `4.750810 ms`。96 核 raw、owner Restore、selector、counter 阈值和离线复算全部通过。该 b256 只作归档数据，不是后续重跑要求。

| 指标 | AIC（32 核） | AIV（64 核） |
| --- | ---: | ---: |
| request/core | 408,317.344 | 422,480.609 |
| miss/core | 38,664.344 | 55,098.625 |
| `Σmiss/Σrequest` | 9.4692% | 13.0417% |
| total/core | 7,471,385.531 | 7,085,178.734 |
| total/core 校准等效时间 | 4,527.942 us | 4,294.748 us |
| scalar busy/core | 5,944,751.250 | 5,603,587.469 |
| scalar busy/core 校准等效时间 | 3,602.744 us | 3,396.667 us |
| `Σscalar/Σtotal` | 79.5669% | 79.0889% |
| 非 Scalar-busy 残余/core | 1,526,634.281 | 1,481,591.266 |
| 非 Scalar-busy 残余/core 校准等效时间 | 925.198 us | 898.081 us |
| `miss/core × 90 ns` | 3,479.791 us | 4,958.876 us |

最后一行只是单核串行等效标尺，不能与 `4.750810 ms` 相减或解释成端到端损失；非 Scalar-busy 残余也不是空闲或 I-cache stall。原始件和自包含报告位于：

```text
outputs/submit_pmu_none_20260719_b256_final/submit_icache_raw.json
outputs/submit_pmu_none_20260719_b256_final/submit_icache_report.html
```

### 9.2 历史 schema-v5 b256 分段时间参考数据

2026-07-19 在同一 A5、`real-compute/6,28,4,1`、b256 配置下重新独立采集 `none|claim|efdrain|materialize|register`。五轮均为 96/96 有效记录；四个 running phase 都是 1,280 calls/core，时间均满足 `0 < phase <= 同核 Submit`。这些是历史归档件；后续边界迭代和重采默认只使用 b1。

| phase | Submit span | ALL 时间占比 | AIC 时间占比 | AIV 时间占比 |
| --- | ---: | ---: | ---: | ---: |
| `none` | 3.711584 ms | 不适用 | 不适用 | 不适用 |
| `claim` | 4.401747 ms | 12.0845% | 7.7096% | 14.2972% |
| `efdrain` | 3.592376 ms | 15.5068% | 20.2974% | 13.1974% |
| `materialize` | 6.770266 ms | 16.7186% | 15.4860% | 17.3556% |
| `register` | 4.086936 ms | 4.3089% | 3.6256% | 4.6568% |

每行都只解释自己的诊断 ELF。尤其 Materialize 的运行中边界读取显著改变了该轮 Submit 时序；这些时间占比不能跨行相加，Submit span 也不能与 `none` 相减成局部净开销。对应权威 raw 和加工 HTML 为：

```text
pa_scheduler/outputs/submit_pmu_phase_time_v5_20260719/none_b256/
pa_scheduler/outputs/submit_pmu_phase_time_v5_20260719/claim_b256/
pa_scheduler/outputs/submit_pmu_phase_time_v5_20260719/efdrain_b256/
pa_scheduler/outputs/submit_pmu_phase_time_v5_20260719/materialize_b256/
pa_scheduler/outputs/submit_pmu_phase_time_v5_20260719/register_b256/
```

### 9.3 当前边界联动版 b1 门禁

2026-07-19 在普通泳道相邻边界收敛后，重新构建五个 `submit-pmu` ELF，并只跑 A5 b1。五轮均使用 `real-compute/6,28,4,1`，都通过 96/96 物理核、真计算输出、mixed 引擎观察、PMU start/stop、owner Restore、phase call/time 和 primary/shadow 门禁：

| phase | 全局首末 Submit | calls | exact/bounded 核 | request/miss loss | phase 时间占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `none` | 113.821 us | 0/0 | 96/96 | 0/0 | 不适用 |
| `claim` | 61.689 us | 480/480 | 80/96 | 7/9 | 22.1744% |
| `efdrain` | 127.969 us | 480/480 | 92/96 | 4/0 | 9.7505% |
| `materialize` | 65.023 us | 480/480 | 96/96 | 0/0 | 41.7821% |
| `register` | 134.714 us | 480/480 | 90/96 | 6/0 | 8.5663% |

`exact/bounded` 中 exact 表示 shadow 与 primary 逐值相等，bounded 表示满足 `shadow <= primary`。五轮 bounded 都是 96/96，小幅 loss 已进入局部 lower/upper，没有被忽略。

该表只证明当前业务语义体与局部 PMU bracket 同步修正后仍严格闭合。b1 的全局 Submit 易受冷启动、多核到达和局部边界扰动影响，不用五行之间的时间差声称 phase 净成本或性能改善。权威 raw 和各自 HTML 位于：

```text
pa_scheduler/outputs/submit_pmu_boundary_sync_b1_20260719/{none,claim,efdrain,materialize,register}/
```

### 9.4 历史 O1 owner/PMU bring-up 证据

现行 `submit-pmu` schema 和 selector 分工建立前，直接 owner 曾用 empty、100,000 scalar NOP、2×100,000 scalar NOP 验证 gate 可以闭合并继续累计；96 核 PMU raw total 中位数约为 214、56,568、112,994。它们只证明链路响应和近似倍增，不表示同数值的纳秒或硬件 cycle。

同一历史版本还采过三个独立 b256 `submit-all` PMU-only 进程：Submit span 为 3.688236/4.089057/4.673237 ms，I-cache request 总和约 69.45M～70.07M，miss 总和约 5.83M～5.85M，整体 miss rate 为 8.32%～8.42%。这些旧统一 ELF 仍含当时的诊断代码，且 PMU gate 会改变多核时序；它们只保留为 owner、96 核拓扑和 raw→summary 演进证据，不覆盖 9.1～9.3 的现行构建口径。历史文件为：

```text
pa_scheduler/outputs/scalar_observation_final_20260718/pmu_submit_all_ccec_b256_v2/pmu_submit_all.json
pa_scheduler/outputs/scalar_observation_final_20260718/pmu_submit_all_ccec_b256_v2_run2/pmu_submit_all.json
pa_scheduler/outputs/scalar_observation_final_20260718/pmu_submit_all_ccec_b256_v2_run3/pmu_submit_all.json
```

## 10. 新增局部 phase 的修改清单

新 phase 不能只增加一个 CLI 字符串。最小完整修改包括：

1. `pa_scheduler/common/pa_model.h`：在 `SubmitPmuPhase` 尾部追加稳定 id，不重排已有 `None=0/Claim=1/EfDrain=2/Materialize=4/Register=5`。
2. `pa_scheduler/ccec/pmu_probe.h`：为 `SubmitPmuPhaseName()` 增加名称映射，并核对 phase status/边界闭合定义。
3. `pa_scheduler/ccec/build.sh`：在白名单中将 phase 名映射到稳定 `PA_SUBMIT_PMU_PHASE_ID`；先校验名称，再用于输出目录。
4. `pa_scheduler/run.sh`：同步 `build-submit-pmu/submit-pmu` 的 phase 白名单和 usage。
5. `pa_scheduler/common/pa_scheduler_core.h`：在真实目标代码段前后放置 `BeginSubmitPmuPhase<...>()` / `EndSubmitPmuPhase<...>()`。必须检查所有早退、winner/loser 和 Alloc/非 Alloc 分支，不得留下只 begin 不 end 的路径。
6. `pa_scheduler/ccec/host.cpp`：增加该 phase 的预期 calls/begin/end 形状。如果它不是每次 Submit 都调用，不能复用 `batches * 5 * 96`。
7. `pa_scheduler/pmu_sidecar_analyzer.py`：同步 phase 名/id 和配置指纹，保证不同 phase 输入不会被聚合。
8. 补充 host/analyzer 回归：`none` 验证 96/96 primary-shadow 精确相等；running phase 验证逐核 bounded、loss/upper 公式、begin/end/calls 和语义，任一 shadow 反向大于 primary 都必须拒绝。新增 phase 的构建、门禁和迭代默认只跑 A5 b1。普通泳道仍复用相邻既有 end，不为了对齐 PMU 而额外增加泳道 `SYS_CNT`。

每个 phase 必须是独立 ELF 和独立进程。不为了一次运行得到多个 phase，而在热路加运行时 phase switch 或多组 begin/end。

## 11. 常见问题与排错

### 提示缺少 submit-pmu 产物

确认 phase 名与构建命令一致：

```bash
./run.sh build-submit-pmu ccec none
./run.sh submit-pmu ccec none --device 0 --batches 1
```

不要用 `./run.sh build ccec` 代替；后者生成的是 `swimlane` 产物。

### host 提示 swimlane 构建不能采 PMU

这表示运行了 `build/ccec/pa_scheduler_host`。应通过 `./run.sh submit-pmu ...` 启动 phase 目录内的 host/kernel/SO 整套产物，不要手工指向根目录 kernel。

### shadow miss 始终为 0

先检查 selector 是否错把 `0x35` 放到 CNT9。本机 A5 b1 已经反证 CNT9 路径；正式配置应为 CNT5 shadow miss、CNT8 shadow request、CNT9 unused。

### `none` 不相等，或任意 phase 出现 `shadow > primary`

这两种情况都表示观察链路门禁失败，最终 JSON 不应发布。按下列顺序排查：

1. host/kernel/owner/dispatcher 是否来自同一 phase 目录；
2. owner 读回的 CNT5/CNT6/CNT7/CNT8 selector 是否与期望一致；
3. begin/end 数是否精确相等，是否有早退路径留下 armed phase；
4. 先缩到 b1；`none` 确认 96/96 exact，running phase 确认 96/96 bounded；
5. 检查可编程 counter 是否超过风险门槛。

running phase 出现小幅 `shadow < primary` 时，代码显式发布 loss 和局部 lower/upper；这不是 standalone 调度正确性异常。只有协议、数值输出、placement/engine 也失败时，才应转向调度代码排查。

### owner 或 Restore 失败

不要在同一设备上并发运行另一个 standalone PMU owner、`msprof` PMU 会话或其他会改 selector 的进程。检查 CANN 环境、两个 AArch64 SO 是否在 kernel 同目录，以及 96 个可用 slot/32 个完整 triplet 是否闭合。

### JSON 拒绝覆盖

每个独立进程使用新文件名，并处理上次失败留下的同名 `.tmp`。host 不会静默覆盖旧证据。

### 分析器报配置不一致

不要强行合并。逐项比较 phase、batches、winner workload/count/pattern、selector、schema 和 build variant。重采相同配置的独立进程。

### miss rate 高，但 Submit 没有同比例变慢

这不构成矛盾。miss rate 是事件比例，不是 stall 时间比例；多核、预取、流水重叠和资源等待都会改变实际暴露量。优先看 AIC/AIV 每核 miss、median/p95 和优化前后的 `Δmiss`，实际性能收益仍以无 PMU/泳道的交错 A/B 为准。
