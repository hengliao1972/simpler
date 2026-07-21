# A5 PA Scheduler I-cache Miss 采集与分析指南

## 1. 目标与最终构建口径

本指南同时记录真实 simpler FDWIC PA 与
`tests/atomic_probe/pa_scheduler` standalone CCEC 的 I-cache 观察方法。目标是回答：
在 Submit-all 整个调度回放期，32 个 AIC 和 64 个 AIV 每核发生了多少 I-cache
request/miss，其中哪些 miss 值得继续优化。真实 PA 结论以真实 PA 独立诊断 ELF 为准；
standalone 只保留历史方法、接口校准和模型边界证据，不能替代真实 PA profile。

standalone 当前保留两类正式观察构建：

| 构建 | 内容 | 是否包含 PMU |
| --- | --- | --- |
| `swimlane` | 普通阶段泳道 + 逐 atomic 泳道，在同一 AIC/AIV scalar lane 合并采集 | 否 |
| `submit-pmu` | 每物理子核的 Submit-all PMU 整窗，并可编译一个局部 phase | 是，仅 CCEC |

`run` 、`smoke` 和 phase 名是运行或编译选择，不是额外的第三类
构建。

### 1.1 真实 PA `submit-pmu-none`

真实 PA 已建立第三条独立证据链 `--fdwic-profile submit-pmu-none`。它不是
standalone 的 `submit-pmu` 产物，也不与泳道 ELF 共用 I-cache 结论：

- 编译期固定 `PTO_FDWIC_TRACE_ENABLED=0`，去除普通泳道和逐 atomic 观察；
- 去除通用逐 task PMU ring，只保留每核首个 Submit 到末个 Submit 的一次 gate；
- CNT2 采集 scalar busy，CNT6/CNT7 采集 primary request/miss，CNT8/CNT5 做
  逐核 shadow 复核；
- AICPU owner 在 96 个物理子核上保存、配置、读回并恢复 PMU 寄存器；
- host 只有在 32 AIC、64 AIV、96 个唯一物理 ID、32 个 1:2 mixed triplet、
  每核实际 Submit 数与 orchestration 声明值一致（Case1 为 1280、B1 为 5）、
  主影子一致和 Restore 96/96 全部闭合时才发布正式 raw；
- 固定输出 `fdwic_submit_pmu_raw.json` 和 `fdwic_submit_pmu_report.html`，先写临时文件
  再原子替换正式文件。

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

该轮全局 Submit 为 **5,075.360 us**。AIC/AIV 的 request/core mean 分别为
646,963.94/594,542.61，miss/core mean 分别为 1,196.88/16,943.17，聚合 miss 率
分别为 0.1850%/2.8498%。PMU total、scalar busy、request、miss 都是同一 ELF、
同一物理核、同一首末 Submit 窗口的数据；仍不能把 `miss * 90 ns` 当作可直接消除的
跨核墙钟损失。

### 1.2 真实 PA 首个单阶段 profile：`submit-pmu-arg-build`

真实 PA 已完成首个跟随最新泳道业务边界的单阶段 profile：

```bash
python -m pytest \
  examples/a5/fully_distributed_within_core/paged_attention_unroll/test_paged_attention_unroll.py \
  --platform a5 --runtime fully_distributed_within_core --level 2 \
  --case Case1 --manual include --fdwic-profile submit-pmu-arg-build \
  --rounds 1 -s -v
```

它仍保留本 ELF 的完整 Submit primary，只在 Kernel/Alloc 的 compete-first
Claim 完成后开启局部 bracket，在匹配 Finish 恢复并校验 ticket 后、Materialize
入口前结束。实际覆盖 Begin 返回、同步 eager callback 构参和 Finish 重入，不包含
Claim 与 Materialize 本体。编译宏为：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=1
PTO_FDWIC_TRACE_ENABLED=0
```

none 的 ABI 仍为 6,272 B；arg-build 在相同 96 份整窗 64 B 记录后追加
96 份 64 B phase sidecar，总计 12,416 B。raw schema 仍为
`fdwic-submit-pmu-v1`，通过 `capture.mode` 区分：

- `submit-pmu-none` 不含 `configuration.phase`，每条 record 也不含 phase 字段；
- `submit-pmu-arg-build` 增加 `phase_elapsed_ticks`、
  `phase_icache_requests_observed`、`phase_icache_misses_observed`、begin/end 次数、
  最大 shadow 分段和 `phase_status`；
- phase 时间只与同一 ELF 的 `Σsubmit_elapsed_ticks` 比较；request/miss observed
  只与同一 ELF、同一角色的 primary 比较；
- 不提供 phase-local PMU total、scalar busy 或 I-cache stall 时间。

CNT6/CNT7 在完整窗口中不读取，继续作为 primary；CNT8/CNT5 运行中
read-clear 并软件重建 shadow whole。`primary - shadow` 是分段重建的 capture gap。
phase observed 会包含 counter 边界附近少量观测 bookkeeping 的取指，因此它不是
原业务事件数的数学下界；`observed + capture gap` 也不是数学上界。HTML 同时展示
两者，只用于判断结论对 capture gap 是否敏感。不同 profile 的 ELF 绝对时间、
request 和 miss 都不能相减。

首轮 Case1/B256 闭合件为：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_014355/
```

该轮 96 核每核 1,280 次 bracket 全部闭合，phase status 为 `0x3f`，
primary/shadow 96/96 精确相等；同一 ELF 内 ALL 的 phase core-time、request
observed 和 miss observed 份额分别为 5.557%、20.716% 和 21.334%。该数据首先
证明采集链闭合，并提示后续需要空 bracket 校准观察 bookkeeping；不能直接把约
21% 写成零插桩业务区间的真实 I-cache 比例。

后续 selector 必须继续跟随真实泳道的排他 span，一次 ELF 只测一个区域，并保留
该 ELF 自己的完整 Submit primary 作为比例分母。

### 1.3 真实 PA 空区间校准：`submit-pmu-empty-bracket`

局部 phase 的 begin/end 本身会执行 shadow counter read-clear、状态检查和累计
bookkeeping。为了先量出这套观察器在真实 simpler A5 PA 热路径上的经验指纹，已增加
独立诊断 profile：

```text
PTO_FDWIC_SUBMIT_PMU=1
PTO_FDWIC_SUBMIT_PMU_PHASE_ID=2
PTO_FDWIC_TRACE_ENABLED=0
```

它不包围任何业务代码。在 Kernel/Alloc compete-first 路径的 Claim 结束处，每次
Submit 紧邻执行一次 phase begin 和 phase end；Case1/B256 每核固定 1,280 次，B1
每核固定 5 次。raw 中必须同时匹配：

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

真实 Case1 的构建和运行仍通过 pytest profile 入口完成；构建缓存会按 profile 宏生成
独立 AIC/AIV ELF，不能拿其他 phase 的旧 ELF 拼接：

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

用例成功后，会在本轮 `outputs/TestPagedAttentionUnroll_Case1_<timestamp>/` 中自动
生成并严格校验：

```text
fdwic_submit_pmu_raw.json
fdwic_submit_pmu_report.html
```

HTML 可直接用浏览器打开。若只需对已存在的 raw 重新生成报告，可在仓库根目录执行：

```bash
python -m simpler_setup.tools.fdwic_submit_pmu_report \
  outputs/TestPagedAttentionUnroll_Case1_<timestamp>/fdwic_submit_pmu_raw.json
```

输出文件名固定为同目录下的 `fdwic_submit_pmu_report.html`。分析器会重新核对
profile、phase 元数据、32 AIC + 64 AIV、每核调用次数、begin/end 闭合、状态位、
primary/shadow 和 owner restore；不能把手工摘出的局部数字绕过这些门禁后当正式结果。

#### 两套边界必须分开解释

empty-bracket 的时间与 I-cache 事件故意使用两套不同边界：

1. `phase_elapsed_ticks` 由一对外层 SYS_CNT 包围相邻的完整 begin/end 调用，包含
   shadow read-clear、begin/end 内部 SYS_CNT、状态检查和累计等观察器路径；它还带有
   外层时间戳自身的测量粒度，是“完整观察器调用对”的经验耗时，不是某段业务时间。
2. `phase_icache_requests_observed` 与 `phase_icache_misses_observed` 只统计 begin 的
   shadow read-clear 到 end 的 shadow read-clear 之间被 CNT8/CNT5 读出的事件。它没有
   覆盖完整 begin/end 调用对，尤其不能把外层 SYS_CNT 时间边界等同为 I-cache
   request/miss 边界。

因此报告中的每 call 指标分别按同一角色聚合后计算：

```text
time/call    = Σphase_elapsed_ticks / Σphase_end_reads
request/call = Σphase_icache_requests_observed / Σphase_end_reads
miss/call    = Σphase_icache_misses_observed / Σphase_end_reads
```

它们是 ALL/AIC/AIV 各自的加权每次调用均值，不是 raw 为每次调用保存了一条记录。
empty-bracket 继续复用每核 64 B phase sidecar，phase ABI 总大小仍为 12,416 B，没有
为了逐调用校准扩充 raw。

#### Case1 稳态校准结果

两轮 Case1/B256 正式闭合件位于：

```text
outputs/TestPagedAttentionUnroll_Case1_20260721_021158/
outputs/TestPagedAttentionUnroll_Case1_20260721_021311/
```

两轮均为 96 核每核 1,280 次、phase status `0x3f`、primary/shadow 96/96 精确
相等，request/miss capture gap 均为 0。每 call 结果如下：

| 轮次 | 角色 | 外层时间 ns/call | request observed/call | miss observed/call |
| --- | --- | ---: | ---: | ---: |
| `021158` | ALL | 640.465 | 49.340 | 1.342 |
| `021158` | AIC | 567.962 | 48.919 | 0.008 |
| `021158` | AIV | 676.717 | 49.550 | 2.009 |
| `021311` | ALL | 639.272 | 49.337 | 1.356 |
| `021311` | AIC | 567.619 | 48.870 | 0.008 |
| `021311` | AIV | 675.099 | 49.570 | 2.030 |

两轮全局 Submit 时间范围分别为 4,972.718 us 和 4,866.126 us。Case1 的每 call
校准值高度接近，可作为当前源码和工具版本下解释局部 phase 观察污染的稳态经验量尺。
AIV 稳定出现约 2 次 miss/call，而 AIC 接近 0；这是观察器在不同 scalar 角色上的
实测指纹，尚不能在没有进一步代码布局证据时归因为某一条具体指令。

当前 ELF 核验也不支持直接改 reader 来“压低校准值”：AIC/AIV reader 都是同一份
92 B noinline 实现，且在 128 B line 下都跨两行；本机 DAV3510 模型配置中的 AIV
scalar I-cache 容量和 set 数只有 AIC 一半，而 AIV 角色代码更大。现阶段只能把约
2 miss/call 视为容量、角色代码与具体布局共同形成的稳定观察指纹，聚合 PMU 不能
定位到某一条 cache line。因而保留当前 reader 和布局；若该底噪妨碍后续 selector，
应另做只改对齐的 empty A/B，不能把布局变化夹带进业务 phase。

B1 闭合件位于：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260721_020932/
outputs/TestPagedAttentionUnroll_CaseB1_20260721_021100/
```

这两轮只有每核 5 次调用，ALL 分别为 725.192/719.304 ns、
84.844/69.194 request 和 1.956/2.444 miss 每 call；AIC 的 request/miss 也明显
比 Case1 稳态更易波动。B1 适合快速验证接口、次数和闭合，不适合代替 Case1 判断
稳态观察器成本；少量调用会把 ELF 首次进入、取指预热和启动时序放大到每 call。

#### 使用边界

- empty-bracket 是观察器的**经验校准 profile**，不是观察成本的数学下界，也不是
  可以从业务 phase 中直接扣除的固定常数。
- `submit-pmu-empty-bracket` 与 `submit-pmu-arg-build` 使用不同诊断 ELF；phase 宏、
  代码布局、I-cache 状态和跨核到达时序都会改变。可以用 empty 的量级判断局部 observed
  是否主要由观察器构成，不能逐项相减生成所谓“修正后的 arg-build”。
- 即使同一轮 capture gap 为 0，也只证明 primary/shadow 整窗重建闭合，不会把上述
  两套边界变成同一个区间。
- 受控微基准给出的约 90 ns/miss 只是一把 core-latency 直觉量尺。miss 可在单核内
  重叠、被流水隐藏，96 核之间也并行；其中还可能包含观察器自身带来的 miss。因此
  `miss × 90 ns` 不能解释为 Submit 墙钟损失，更不能当成候选优化的可兑现收益。

### 1.4 真实 PA `submit-pmu-materialize`

`submit-pmu-materialize` 是首个经过 empty-bracket 校准后落到真实业务 span 的
局部 profile。它继续保留该 ELF 自己的完整 Submit primary，只把 running
read-clear bracket 放到泳道 `Materialize` 的同一业务边界。编译宏为：

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

当前真实 PA 使用 compete-first 路径：Kernel/Alloc 的 Finish 完成 ticket 恢复与校验、
结束 arg-build 后，在泳道 `Materialize.begin` 对应位置打开 bracket；
`dist_submit_materialize_and_prepare_map()` 内完成 task-cap 检查和
`dist_submit_materialize_args()` 后立即关闭，再进入 `PrepareMap`。因此该 profile
覆盖 Materialize 自身，不包含 Claim-to-Materialize 构参区间，也不包含后续
`dist_submit_prepare_map()`。仍受支持的 one-shot 入口使用相同的 Materialize
业务首尾边界。

每个成功 Submit 恰好执行一次 Materialize bracket，所以调用 shape 与每核 Submit
数完全一致：Case1/B256 为每核 1,280 次、全局 122,880 次，B1 为每核 5 次、全局
480 次。失败路径不会伪造 phase end；只要某核 begin/end 不平衡、次数不符或状态位
不完整，host/分析器就拒绝发布受信结果。

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

两轮均为 96/96 受信记录、每核 1,280 次、begin/end
122,880/122,880、phase status `0x3f`，primary/shadow 的 request/miss capture gap
均为 0。下表的三个占比都只使用**本轮同一个 Materialize ELF** 的分母：时间以
`Σsubmit_elapsed_ticks` 为分母，request/miss 以各角色的完整 Submit primary 为
分母。

| 轮次 | 角色 | 时间 ns/call | request observed/call | miss observed/call | 时间占比 | request 占比 | miss 占比 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `024748` | ALL | 797.814 | 233.197 | 1.474 | 22.560% | 37.688% | 12.056% |
| `024748` | AIC | 776.265 | 228.942 | 0.022 | 22.540% | 36.532% | 10.487% |
| `024748` | AIV | 808.588 | 235.325 | 2.201 | 22.570% | 38.278% | 12.065% |
| `024909` | ALL | 797.061 | 233.238 | 1.418 | 22.105% | 37.741% | 11.681% |
| `024909` | AIC | 775.653 | 228.170 | 0.021 | 21.882% | 36.442% | 10.997% |
| `024909` | AIV | 807.765 | 235.772 | 2.116 | 22.214% | 38.404% | 11.685% |

两轮全局 Submit 时间范围分别为 4,922.142 us 和 4,851.282 us。Materialize 的
ALL request observed 稳定在约 233.2 次/call，AIC/AIV 也都稳定在约
228～236 次/call；相对 empty-bracket 的约 49 次/call 经验指纹，这个 request
信号在量级上更明显。但两者来自不同诊断 ELF，代码布局、缓存状态和到达时序不同，
这里只能作方向性判断，不能执行 `materialize - empty`。

AIV 的 Materialize miss observed 为约 2.20/2.12 次/call，与 empty-bracket
两轮约 2.01/2.03 次/call 处于同一量级。这说明当前数据尚不能证明 Materialize
业务体本身带来明显的 AIV miss 增量；也绝不能跨 ELF 相减后把约 0.1 次/call
冒充业务净 miss。AIC 的 miss observed 同样很小。现阶段可受信的结论是：
Materialize 时间和 request 信号两轮稳定，而 miss 归因仍受到观察器经验指纹限制。

B1 只用于结构与 cold-path 核验：

```text
outputs/TestPagedAttentionUnroll_CaseB1_20260721_024533/
outputs/TestPagedAttentionUnroll_CaseB1_20260721_024641/
```

两轮都满足 96/96 记录、每核 5 次、480/480 begin/end、`0x3f` 状态和零 capture
gap，证明 profile 选择、边界次数、ABI 与报告链路闭合。由于每核只有 5 次调用，
首次进入 ELF、取指预热和启动时序会被放大到每 call；B1 的 per-call 值只作 cold
证据，不用于替代上述 Case1 稳态归因。

## 2. 为什么 I-cache miss 必须独立重编译

I-cache 数据对代码布局极其敏感。泳道和逐 atomic 观察会增加：

- 阶段和 atomic 的 `SYS_CNT` 读取；
- atomic wrapper、ClockBaseline 与 record 发布分支；
- 更大的 scalar `.text` 和不同的函数/对齐布局；
- 由此引起的 worker 到达、轮询和跨核竞争时序变化。

因此，“代码仍在 ELF 中，只在运行时关闭 record”不足以得到干净的
I-cache 观察。`submit-pmu` 会独立重编译，编译掉泳道 record、逐
atomic wrapper、ClockBaseline、runtime phase-profile 和旧 cold/warm 冲刷体。
`swimlane` 则不构建 PMU owner。

两种数据不在同一进程采集：

- 看事件时序和 atomic bracket，使用 `swimlane`；
- 看 Submit-all 整窗每核 I-cache request/miss，使用 `submit-pmu`。

两份证据可以按同一源码版本交叉理解，不能做逐 tick 对齐，也不能
把 PMU 的平均 miss 回填为某一条 atomic span 的属性。

### 2.1 运行时关闭不等于编译期去除：真实 PA 回退案例

真实 PA 已出现过一次完整反例：level 1 的 raw 中没有任何 Atomic 或
ClockBaseline 记录，但同一 ELF 为了允许运行时切到 level 4，仍把 atomic
wrapper、PollBatch 遍历和记录发布慢体大量内联进 Submit 热函数。结果不是
“没有写 record 就没有代价”，而是未执行的诊断代码仍改变 `.text`、基本块布局
和取指工作集。

当时真实 A5 Case1 的证据链为：

| 构建状态 | AIC/AIV `dist_engine .text` | AIC/AIV `dist_submit_impl` | 三轮首末 Submit 中位数 |
| --- | ---: | ---: | ---: |
| pre-atomic 历史构建 | 80,824 / 80,912 B | 42,136 / 42,152 B | 5.115620 ms |
| atomic 观测接入、level 1 不落 Atomic | 347,360 / 355,968 B | 100,724 / 102,588 B | 5.631038 ms |
| 两处低频 winner 调整为冷分支 | 347,536 / 357,112 B | 100,860 / 103,676 B | 5.192087 ms |
| atomic 冷代码共享外提 | 66,768 / 67,120 B | 18,812 / 18,872 B | 4.821897 ms |

最后一行正式三轮为 4.821897/4.890447/4.752956 ms。这里能证明的是
“代码复制和布局回退已被消除”；没有同时采集 I-cache PMU，因此不能把恢复量
进一步写成某个确定的 miss 降幅。

冷代码外提遵守两条边界：

1. direct atomic 的 begin、真实 atomic、返回值地址依赖和 end 仍留在原 wrapper，
   只共享 end 之后的 record 发布。因此不把 source-issue 操作改成等待返回型，
   atomic span 口径也不变。
2. PollBatch 保留内联的 `level >= 4 && active_mask != 0` 快速判断，只把命中后的
   十类遍历与落盘外提。level 1 不新增 call/ret，level 4 的 `end_cycle` 仍在
   冷函数调用前取得。

保留此类修改前应依次检查：AIC/AIV 对象的 `.text` 和独立符号、level-1 多轮
Submit、level-4 logical/physical Atomic 公式、ClockBaseline 和
`dropped_records=0`。上述真实 PA level-4 复核得到 115,309 次 atomic 调用、
107,608 条 Atomic、8,056 次批处理轮询和 355 条 PollBatch，满足
`107608 = 115309 - 8056 + 355`。

这个案例同时说明为什么 `submit-pmu` 不能只传运行时 level=0：专门分析
I-cache 时，普通泳道、atomic wrapper、ClockBaseline 和相关慢体必须在编译期从
待测 AIC/AIV ELF 中剔除。运行时 gate 只控制“执行没有”，不能控制“代码存在没有”。

### 2.2 Claim-first eager 重编后的观察结论

真实 PA 切换到 compete-first eager 后，Submit 的阶段顺序变为
`EfDrain -> Claim -> Materialize -> PrepareMap -> Fanin/Register -> 尾阶段`。
这类热路径重排会同时改变基本块与跨 TU 的代码布局，因此它的性能结论
应记录在 I-cache 观察指南中，不归因为某个 atomic 本身的收益。

迁移后的 atomic 观察仍包围原位置的真实指令：消费返回值的调用
继续使用 `return_ready` 边界，不消费返回值的 Exchange/FetchAdd
继续使用 `source_issue` 边界，PollBatch 的逻辑调用与物理压缩记录口径
也没有改变。Claim 前移只改变业务阶段顺序，没有把 atomic 记录
提前、延后或改写成另一种完成语义。

最终真实 A5 level-4 复核位于：

~~~text
outputs/TestPagedAttentionUnroll_Case1_20260720_104406/
~~~

该轮包含 122,880 个 Submit、945,653 条事件，`dropped_records=0`。Atomic
物理记录、逻辑调用、轮询调用和 PollBatch 记录满足：

~~~text
106355 = 109392 - 3361 + 324
~~~

raw 转换、阶段顺序和整数闭合均通过，schema 能完整解析
site/op、`result_used` 与 `return_ready`。这些结果只用于证明热路径
重排后的观察能力和计数口径仍然正确；该轮没有同时采集专用
I-cache PMU，不能据此推导 I-cache miss 降幅或某个 atomic 的独立收益。

## 3. Standalone 历史 `none` 与局部 phase 如何选择

从本节到第 11 节主要描述 `tests/atomic_probe/pa_scheduler` standalone 的历史
schema-v5、构建目录、命令和字段；不能套用到上面的真实 PA profile。真实 PA
当前 profile 为：

- `submit-pmu-none`：完整 Submit 整窗；
- `submit-pmu-arg-build`：Claim.end 到 Materialize.begin 的同步构参区间；
- `submit-pmu-empty-bracket`：第 1.3 节定义的观察器经验校准，不是业务 phase；
- `submit-pmu-materialize`：第 1.4 节定义的真实 Materialize 业务 span。

四者 raw schema 均为 `fdwic-submit-pmu-v1`，但分别来自独立诊断 ELF，不能跨
profile 相减或拼接。

standalone 历史 schema 中的 `lower/upper` 是已经固化的字段名，只表达
read-clear observed 与 `primary-shadow` capture gap；由于 bracket 两侧也有观察
bookkeeping，它们同样不能解释为零插桩业务事件数的数学上下界。

standalone 当前保留五个编译期 phase：

| phase | 边界 | 优先用途 |
| --- | --- | --- |
| `none` | Submit-all 整窗中不读局部 shadow counter | 回答整个调度回放期的 AIC/AIV 每核 request/miss；这是默认选择 |
| `claim` | `Claim()`、结果写回 context 及 claim 本地统计前后读局部 shadow counter | 当 `none` 已证明 miss 值得追踪时，试验 Claim 的 running read-clear 下界/上界归因链路 |
| `efdrain` | 每次 Submit 开头唯一的 `DrainReady(...EfDrain...)` 前后 | 观察 opportunistic drain；不混入 RingBackpressure 或 FinalDrain |
| `materialize` | `MaterializeTask()` 及成功路径 `materialized_outputs` 本地统计前后 | 观察输出 descriptor/layout、本地 register mask、输出字节数和 heap 游标等 scalar 工作；不包含后续 slot payload 拷贝 |
| `register` | 每次 Submit 统一的 `RegisterOutputs()`，非 Alloc 还包含 `map_inserts` 本地统计 | 观察输出注册语义体 |

普通泳道为了减少观察扰动，会让相邻阶段复用同一个
`SYS_CNT` 边界。PMU-only ELF 不生成这些泳道时间戳；局部 PMU bracket
只包围同一语义体，使用自己的 shadow read-clear 和 `SYS_CNT`，不做
跨 ELF 的逐 tick 对齐。

running phase 的 begin/end 读取本身会执行 scalar 指令、占用取指并改变多核
时序；schema-v5 还在每次 begin/end 各读取一次 1 ns/tick 的 SYS_CNT。因此：

- running phase 的 phase request/miss 是带局部边界扰动的观察值；
- running phase 的累计时间也是带边界扰动的直接观察值；起点在 begin 的
  shadow read-clear 之后，终点在 end 的 shadow read-clear 之前，所以不包含
  两侧 `ld_dev`，但包含每次调用两次 SYS_CNT 的观察扰动；
- request/miss raw 观察值是下界，上界为该核下界加
  primary-shadow loss；阶段时间是单点观察值，没有伪造的上下界；
- `none` 和任一 running phase 是不同 ELF、不同进程，不能以两者相减声称
  得到了零扰动的局部净值；
- 未来不同 phase ELF 的局部 request/miss 不可相加成 whole gate；
  Submit-all 整窗始终以每个 ELF 自己的 primary whole 为准。

`none`/`claim`/`efdrain` 与正式 swimlane 使用 block-local runtime
state 和跨 TU noinline finish；Claim/EfDrain 边界在 finish 之前已闭合。
`materialize`/`register` 为了在 finish 内继续操作同一份真实
`PmuContext`，当前使用 inline-finish 诊断 ELF。后两者与
`none` 不具备字节级相同的指令布局，它们的局部结果只能在各自
ELF 内解释。

该区间只描述同一插桩 ELF、当前边界定义下的局部事件，不是无插桩局部阶段
的真实区间。Submit-all `none` 没有运行中 read-clear，仍执行 96/96
逐核严格闭合。

报告中有三个相关但不相同的时间窗，不能都简称为“完整
Submit”：

1. **PMU whole gate**：每核在 `InitPaOrchestration()` 之前
   `metrics_prof_start()`，在末次 UP `SubmitCallbackTask()` 返回后立即 stop。它包含
   orchestration 初始化、`EfDrain`、`Claim`、同步 eager 构参、finish、
   `AcceptTaskOutputs()` 和 Submit 间输出接收/调用衔接，
   排除 FinalDrain；与泳道 `OrchestrationReplay` 父区间接近但不做逐 tick
   对齐。CNT6/CNT7 primary 和 PMU total/scalar-busy 都使用这个窗。
2. **`submit_elapsed_ticks`**：每核从首个 `BeginCallbackSubmit()` 之后到最后一个
   Submit 的 `submits++` 之后，包含两端之间的 Submit 间衔接，但不包含
   whole gate 首尾的 orchestration 外围。局部 phase 时间占比以它为分母。
3. **`configuration.submit_span_us`**：96 核中最早 `submit_begin` 到最晚
   `submit_end` 的全局墙钟范围，不是逐核 PMU 时间的平均值。

## 4. 环境、构建与产物

命令在 `tests/atomic_probe/pa_scheduler` 目录下执行。非交互 shell
建议显式 source CANN 并选择本用户 GCC 15：

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

每个 phase 目录中的 `pa_scheduler_host`、`pa_scheduler_kernel.o`、
`libpa_scheduler_pmu_owner_aicpu.so` 和
`libpa_scheduler_pmu_owner_dispatcher.so` 是一个不可拆分的构建集。host
会按 kernel 所在目录加载两个 SO；不得从另一个 phase 目录复制或
拼接产物。构建只在四件套全部完成后原子发布
`submit_pmu_artifacts.manifest`；`run.sh` 在启动 host 前核对 schema、phase、
固定文件列表和四个 SHA256。

`swimlane` 的 CCEC 产物仍在 `build/ccec/`，不是 PMU
产物。

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

多轮比较必须使用多个独立进程和独立子目录；每个采集目录内部都保持
同一组描述性文件名：

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

`efdrain` 每核固定调用 `batches * 5` 次；b1 的 AIC/AIV/global calls
分别为 160/320/480。插点只位于 Submit 开头的 EfDrain 专属
call-site；复用的 `DrainReady()` 函数体不插桩。

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

边界位于 `MaterializeTask()` 唯一调用点前后。实现必须先保存返回值、关闭
phase，再处理失败返回，避免失败路径留下 begin/end 不平衡。每核固定
`5 * batches` 次；b1 的 AIC/AIV/global calls 为 160/320/480。

2026-07-19 A5 历史实测闭环如下；四轮均满足 capture accepted、
语义、PMU 和 phase measurement PASS。b256 数据只作归档证据，不是后续迭代的
重跑要求：

| phase | batches | calls/expected | begin/end 与 call shape | primary=shadow | shadow≤primary | request/miss loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `materialize` | 1 | 480/480 | 96/96 | 94/96 | 96/96 | 2/0 |
| `materialize` | 256 | 122,880/122,880 | 96/96 | 74/96 | 96/96 | 160/8 |
| `register` | 1 | 480/480 | 96/96 | 93/96 | 96/96 | 2/1 |
| `register` | 256 | 122,880/122,880 | 96/96 | 36/96 | 96/96 | 2,834/654 |

对应 raw/HTML 位于
`outputs/submit_pmu_{materialize,register}_20260719_b{1,256}/`；raw 是权威
取数件，HTML 是同目录的加工展示件。

running read-clear 的硬门禁是 96/96 `shadow≤primary`，不是要求 96/96
逐值相等；表中的 loss 已进入每核局部 lower/upper 区间，不能被静默忽略。

### 5.5 Register 局部归因

```bash
OUT_REG_B1="./outputs/submit_pmu_register_$(date -u +%Y%m%dT%H%M%SZ)_b1"
mkdir -p "$OUT_REG_B1"

./run.sh submit-pmu ccec register \
  --device 0 --batches 1 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1 \
  --pmu-json "$OUT_REG_B1/submit_icache_raw.json"
```

Register 已合并为每次 Submit 唯一的 `RegisterOutputs()` 调用点；
Alloc 通过 `include_existing=false` 保留语义差异，winner/loser 都经过该边界。
因此仍是每核固定
`5 * batches`，调用规模与 Materialize 相同。它与普通泳道包围
同一 Register 语义体，但两个 ELF 各自取时，不做逐 tick 对齐；其中
较短或未实际插入 map 的调用仍会放大 PMU begin/end
边界扰动，解释结果时必须使用 lower/upper，而不能把 observed lower
当成无扰动净开销。

`submit-pmu` action 已固定：

```text
--runs 1 --no-swimlane --pmu-window submit-all
```

调用者不要重复传入这三项，也不能添加
`--profile-phases`、`--trace-atomics`、`--analyze-swimlane` 或
`--swimlane-json`。`--pmu-json` 可选；但要做 raw 复算、HTML 可视报告和
多轮汇总时必须使用它。host 拒绝覆盖已有 JSON 或同名 `.tmp`。

raw 成功发布后，`run.sh` 会调用本目录的独立分析器并在同一目录生成：

```text
submit_icache_raw.json       # 96 核权威原始件及 host summary
submit_icache_report.html    # 可离线浏览的加工件
```

HTML 使用内联 CSS/SVG，不依赖外部前端库；浏览器直接打开即可。
它包含 Submit-all PMU 整窗的 AIC/AIV 对比、逐物理核 request/miss/rate 分布、96 核
明细和 90 ns core-equivalent 提示。报告同时展示 AIC/AIV/ALL 的 PMU raw
`total_cycles`、CNT2 `scalar_busy`、`Σscalar/Σtotal` 以及
“非 Scalar-busy 残余”，并在保留 raw cycle 的同时给出校准后的每核等效时间。
顶部的“完整 Submit”固定表示**最早一次 Submit 进入到最晚一次 Submit 返回**；
它是 96 核共同形成的整体墙钟范围，不等于逐核 PMU whole gate
的平均值；对应到每个 worker，首末 Submit 边界也窄于该核的 PMU
whole gate。
ALL/AIC/AIV 分别用响应式卡片展示逐核 `min/mean/max`：`mean` 作为唯一的
典型值，`min/max` 只用于呈现核间范围。raw summary 没有 `min` 字段，HTML
从通过 96 核门禁的 `records` 对称推导 `min/max`；PMU total 与 scalar busy
各自独立取极值，不保证来自同一个物理核。较宽的 I-cache 对比表和 96 核明细
只在表格内部横向滚动，不再撑宽整页。

本机 A5 受控 cold/warm 同窗校准得到：

```text
PMU cycle_delta = 1,817,457
SYS_CNT tick_delta = 1,101,593 ns
ALL = 1.649844 cycles/ns
AIC = 1.650062 cycles/ns
AIV = 1.649731 cycles/ns
time_us = PMU cycles / (cycles_per_ns * 1000)
```

ALL/AIC/AIV 汇总分别使用对应频率，逐核明细按该核角色使用 AIC 或 AIV
频率。其中 total 是每个物理子核在 PMU whole gate 内的
累计周期，96 核求和是 core-work，不是 Submit 墙钟；“非 Scalar-busy 残余”
严格等于 `total−scalar_busy`，既不是 Scalar 空闲时间，也不是 I-cache stall，
其中还包含同步等待、vector/cube engine 等待以及其他未归因周期。受控微基准中，依赖返回的 atomic 等待大部分进入
scalar busy，而 I-cache refill 的额外周期大部分只进入 total；这个现象
不能把二者之差提升为 I-cache 专属计数器。

对于局部 phase，HTML 最前面先按 ALL/AIC/AIV 展示 calls、阶段时间占比、
request/miss 占比；后面再列阶段时间/core、阶段时间/call、request/miss
下界—上界和 shadow loss。阶段时间占比是同一角色内
`Σphase_elapsed_ticks / Σsubmit_elapsed_ticks`，request/miss 则仍是占同一
ELF PMU whole-gate primary 的比例区间。两类占比的分母边界不同，只能
分别用于时间和 I-cache 归因，不能把它们当成同一精确分区。`none`
明确显示“不适用”，历史 schema-v4
因没有阶段时间 raw 字段而显示“不可用”，不会伪造 0%。
报告生成失败时 action 返回非零，但已成功发布的 raw 会保留用于排查。

## 6. Primary/shadow 计数和可信门禁

### 6.1 计数器分工

早期曾尝试用 external task-based `msprof` 汇总配合 kernel 内 start/stop 取得
Submit 子窗口，实测 raw counter 不受该门控缩窗，因此不能用于局部归因。现行
链路由 standalone 自带的 Main AICPU owner 保存、配置、读回并恢复每个物理子核
PMU，kernel 只在同一 runtime TU 内控制 gate 和发布 worker 独占结果。任何
owner membership、selector readback 或 Restore 失败都会拒绝最终 JSON。

| 计数 | selector | 用途 |
| --- | --- | --- |
| PMU raw total | 固定 64-bit total low/high | PMU whole gate 内每个物理子核的累计周期；不是 96 核求和后的墙钟；HTML 另按实测频率显示等效时间 |
| CNT2 | `0x001` | scalar instruction busy cycle；不包含全部等待周期；HTML 另按实测频率显示等效时间 |
| CNT6 | `0x34` | PMU whole-gate primary I-cache request；局部边界从不读它 |
| CNT7 | `0x35` | PMU whole-gate primary I-cache miss；局部边界从不读它 |
| CNT8 | `0x34` | read-to-clear shadow request |
| CNT5 | `0x35` | read-to-clear shadow miss；诊断 ELF 因此不再提供 MTE3 busy |
| CNT9 | `0x0` | 未使用 |

A5 b1 实测已证明将 `0x35` 配置到 CNT9 时计数始终为 0，所以
CNT9 不能作 shadow miss。这个变化只影响 `submit-pmu` 诊断构建，
不影响 `swimlane` 构建。

#### A5 `scalar_wait_ib_time` 的支持边界

2026-07-19 使用本机 CANN 9.1 在同一 A5/DAV3510 上对 standalone b1 依次验证了
`PipeUtilization`、`PipeUtilization,MemoryDetail` 和 `Default` 三种正式
`msopprof` 采集入口。三次均能生成 `PipeUtilization.csv`，但表头都不包含
`aic/aiv_scalar_wait_ib_time`，也不包含 `aic/aiv_scalar_wait_time`。本机 CANN
9.1 `CHIP_V6_MAP` 与本仓 DAV3510 正式事件表同样只给出已使用的
`scalar_busy(0x001)`、I-cache request `0x34`、I-cache miss `0x35` 等事件，
没有 wait-IB/wait 的 selector 或派生公式。

三次原始证据分别保存在：

```text
outputs/wait_ib_official_msopprof_20260719_b1_probe2/
outputs/wait_ib_official_msopprof_20260719_b1_probe3_memory_detail/
outputs/wait_ib_official_msopprof_20260719_b1_probe4_default/
```

因此当前 A5 正式可编程路径的结论是：**不能采集这两个指标**。CANN 共享
`msopprof` 二进制包含相应字段字符串、官方文档也在 A2/A3 产品章节解释其
含义，但这些证据不能推出 DAV3510 selector。不得把旧架构或其他产品的事件号
套到 A5。若后续 CANN/A5 正式事件表新增这两项，必须重新用 scalar NOP、
I-cache warm/cold、真实 Vector/Cube `PIPE_* -> PIPE_S` wait 和依赖 atomic
四组对照校准后再纳入报告。

shadow PMU counter 是 read-to-clear。任一 running phase 在阶段 begin 读取 CNT8/CNT5，将
之前的片段加入 shadow whole；在 end 再读一次，同时加入 shadow
whole 和所选 phase；PMU whole gate stop 后读 tail。`none` 不做中途读取，
只在 stop 后取 tail。

`none` 对每个物理子核必须精确满足：

```text
shadow_whole_icache_requests == icache_requests
shadow_whole_icache_misses   == icache_misses
```

即 stop 后读取的 CNT8/CNT5 分别等于同 selector、同 gate 的 CNT6/CNT7。
这个 96/96 精确相等门禁验证 whole-gate 观察闭合；它不把 PMU 进程的
Submit span 变成无诊断墙钟基线，也不把 standalone 数据冒充真实 PA profile。

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

上下界必须先逐核计算，再分别聚合；不能拿聚合后的 median 相减拼区间。
CNT8/CNT5 是顺序 `ld_dev`，不是同一时刻的原子配对快照，因此局部
`phase_miss <= phase_request` 不是硬门禁。二者分别不超过对应 shadow，
各自上界不超过对应 primary。

结合 `none` exact、running phase bounded 和调用次数门禁，可分别验证：

- 复制 selector 在本机 A5 上确实计数；
- 边界调用覆盖预定代码片段，且 begin/end/tail 次数闭合；边界竞态少计
  由 loss 和 lower/upper 区间显式保留；
- primary whole 没有被局部归因读取破坏。

### 6.2 正式 JSON 必须通过的门禁

只有下列条件全部成立，host 才发布最终 JSON：

- 语义、winner 真计算输出和 Submit placement/engine 闭合通过；
- 96 条记录可信，32 AIC + 64 AIV，物理子核 id 唯一；
- owner bitmap membership、worker slot、物理 role 和 32 个 1:2 triplet 全部匹配；
- 96 个核都真实执行 Submit-all PMU whole-gate start/stop，owner Restore 成功；
- build variant 和编译 phase id 在 96 条记录中全部匹配；
- `none` 的 shadow whole 与 primary whole 逐核精确相等；
- running phase 的 shadow whole 逐核不大于 primary，loss 与 upper-bound
  公式逐核闭合；exact 核数只作诊断；
- `none` 的 phase calls/begin/end/request/miss 全部为 0；
- `claim`、`efdrain`、`materialize`、`register` 的 begin/end/calls 逐核
  平衡，每核 calls 为 `batches * 5`，全局 calls 为
  `batches * 5 * 96`；
- 四个 running phase 必须分别命中自己的边界：Claim 调用、Submit 开头
  EfDrain 专属 call-site、Materialize 唯一调用点，以及每次 Submit
  统一的 Register 调用点；
- phase request/miss 分别不超过对应 shadow/primary，且可编程 counter
  低于当前 25% 保守风险阈值。
- 96 个核的 `submit_elapsed_ticks` 都大于 0；running phase 的
  `phase_elapsed_ticks` 大于 0 且不超过本核 `submit_elapsed_ticks`，`none`
  则必须精确为 0；phase time 状态位和 host 复核都必须通过。

`metrics_prof_start/stop()` 在 PMU whole gate 前后各执行一次，其
`PIPE_ALL` 边界会改变流水和多核时序。PMU 结果只与相同构建、
相同 phase、相同负载的独立进程比较，不把 PMU 进程的 Submit
span 当作无诊断性能基线。

## 7. JSON 字段与 AIC/AIV 分析口径

当前 `submit-pmu` 输出 schema v5；分析器继续只读兼容历史 schema-v4。
`records` 保留 96 个 worker 的 raw，
`summary.all/aic/aiv` 分别对 96/32/64 个核统计：

```text
sum / mean / median / p95 / max
```

Submit-all 整窗优先查看：

- `configuration.submit_span_us`：本轮从第一个 Submit 进入到最后一个
  Submit 返回的整体 span；HTML 顶部换算成毫秒展示；
- `total_cycles`：每核 PMU whole gate 内的 64-bit PMU raw total；按角色的
  sum 是 core-work，raw 仍保留 mean/median/p95；HTML 只选 mean 表示典型
  单核，并辅以从逐核记录得到的 min/max，三者均不等于 host 看到的 Submit
  墙钟。HTML 按 ALL/AIC/AIV 的实测频率将 raw cycle 换算为单核
  cycle-equivalent；
- `scalar_busy`：CNT2 `scalar_instr_busy(0x001)`，表示 scalar instruction
  busy cycle；依赖返回的 atomic 等待可进入此项，但它不是“纯算术指令数”；
  换算后的时间也只是 scalar-busy cycle-equivalent；
- `icache_requests` / `icache_misses`：CNT6/CNT7 PMU whole-gate primary；
- `shadow_whole_icache_requests` / `shadow_whole_icache_misses`：闭合或分段
  loss 用 shadow whole；
- `shadow_request_loss` / `shadow_miss_loss`：本核 primary-shadow residual；
- `phase_calls` / `phase_icache_requests` / `phase_icache_misses`：选定 phase
  的 running read-clear lower；
- `submit_elapsed_ticks`：本 worker 从首个 `submit_begin` 计时点到末个
  `submit_end` 计时点的 SYS_CNT 差值；起点位于首个
  `BeginCallbackSubmit()` 上下文初始化之后，终点位于末个 Submit 返回之前；
  1 tick = 1 ns；
- `phase_elapsed_ticks`：所选 phase 所有调用的 SYS_CNT 差值累计；running
  phase 必须非零且不超过同核 `submit_elapsed_ticks`，`none` 必须为 0；
- `phase_icache_requests_upper_bound` / `phase_icache_misses_upper_bound`：
  lower 加本核对应 loss；
- `configuration.compiled_phase` 和 `validation.phase_measurement_valid`：确认文件口径。
- `validation.shadow_primary_match_records` / `shadow_primary_bounded_records`：
  区分逐值 exact 与单向 bounded 核数；
- `configuration.phase_values_are_running_read_clear_lower_bounds`：确认局部字段
  是否采用下界语义。

HTML 中展示的“非 Scalar-busy 残余/core”严格等于
`(total−scalar_busy)/core`，只用于观察未被 scalar busy 覆盖周期的数量级。
它可能同时包含 I-cache refill、同步等待、vector/cube engine 等待和其他
流水空隙，不能命名为 Scalar 空闲或 I-cache stall，也不能用它反推单次 miss
代价。

离线分析器与 HTML 会从这些 raw 字段继续派生
`phase_icache_request_lower_bound_share_of_submit`、
`phase_icache_request_upper_bound_share_of_submit`、
`phase_icache_miss_lower_bound_share_of_submit` 和
`phase_icache_miss_upper_bound_share_of_submit`：局部 lower/upper 分别除以
同一角色、同一次采集的 PMU whole-gate primary 总数。

`phase_observed_read_clear_ratio` 只是 lower miss/lower request 的观测比值；
分子和分母各有独立区间，因此它不是实际 phase miss rate 的数学下界。

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

这里的 `time_us` 是每核 cycle-equivalent。`total−scalar_busy` 即使换算为时间，
仍不是 Scalar 空闲时间或 I-cache stall；96 核 sum 换算后也仍是 core-work，
不能冒充 Submit 墙钟。

`median` 和 `p95` 直接来自同角色逐核 raw 分布，用于观察典型核和高
尾核。I-cache miss rate 只按组内加权口径计算：

```text
AIC miss rate = Σ(AIC miss) / Σ(AIC request)
AIV miss rate = Σ(AIV miss) / Σ(AIV request)
```

不平均 32 或 64 个逐核百分比。AIC/AIV 核数不同，比较每核强度时
使用 mean/median/p95 或 miss rate，不直接比较两组 sum。

局部 phase 的时间和 I-cache 占比都使用组内总量，但分母边界不同：

```text
time share          = Σphase_elapsed_ticks / Σsubmit_elapsed_ticks
request share lower = Σphase_request_lower / Σprimary_request
request share upper = Σphase_request_upper / Σprimary_request
miss share lower    = Σphase_miss_lower / Σprimary_miss
miss share upper    = Σphase_miss_upper / Σprimary_miss
```

时间分子、分母来自同一个 1 ns SYS_CNT，是逐核累计 core-time 构成；不能用
`Σphase_elapsed_ticks` 除以 96 核共同形成的 `submit_span_us`。时间是直接观察
单值，不仿造 request/miss 那样的 lower/upper。多个独立进程聚合时，分析器展示
每轮上述 `Σ/Σ` 比值的分布，不把不同轮的 raw 重新拼成一次虚构运行。

分子与分母必须来自同一个 phase ELF、同一轮采集和同一角色。该比例回答“当前
插桩 ELF 中局部窗口占自身对应分母的多少”，不能拿 `claim` 分子除以
另一份 `none` 的分母。

## 8. HTML 报告与多轮分析命令

单份 raw 的 HTML 已由 `submit-pmu` action 自动生成。需要手工重建时：

```bash
PYTHON=/home/q00473782/.venv/bin/python

"$PYTHON" ./pmu_html_report.py \
  "$OUT/submit_icache_raw.json"
```

默认输出为同目录的 `submit_icache_report.html`；也可用 `-o` 指定路径。
生成器先调用 `pmu_sidecar_analyzer.py` 的完整 raw 门禁，只有 96 核拓扑、
raw→summary、primary/shadow、采集接受状态和 owner Restore 全部通过才发布 HTML。

使用本用户 Python 环境从 raw 重算 host summary，并聚合相同配置的
多个独立进程：

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

建议将每次采集的 JSON 和分析器生成的 summary 放在同一唯一目录；
`outputs/` 为本机证据目录，不作为源码提交的一部分。

## 9. 如何使用约 90 ns/miss 标尺

当前约 `90 ns/miss` 来自隔离 cold/warm 微基准，只用于建立数量级感性。
目标函数只有 8 B，并按 128 B I-cache line 对齐；cold trial 在窗口外先执行
64 KiB 指令 capacity sweep，warm trial 在 PMU read-clear 前额外调用一次同一
目标。1 ns/tick 的 SYS_CNT 只包围最终目标调用，两条路径的 gate、harness 和
目标符号相同。

64 trials/core × 10 轮和 128 trials/core × 5 轮都满足：

```text
cold CNT7 miss == trials
warm CNT7 miss == 0
miss_delta == 96 * trials
calibrated_cores == 96/96
```

前者 ALL 中位数为 86.596 ns/miss，范围 86.532～86.792；后者中位数为
89.629 ns/miss，范围 89.615～89.648。AIC/AIV 差值方向在两组规模间改变，
因此不建立伪精确的角色常数，统一取 90 ns 只作一阶标尺。历史原始日志为：

```text
pa_scheduler/outputs/pmu_validation/icache_single_64x10_20260718_085929_3232836_console.log
pa_scheduler/outputs/pmu_validation/icache_single_128x5_20260718_090151_3235468_console.log
```

对某一角色，可以计算：

```text
角色总 core-work 串行等效量(us) = Σmiss * 0.09
该角色平均每核等效量(us/core) = (Σmiss / core_count) * 0.09
```

约 1.65 GHz 的频率只用于换算 PMU cycle 事件；它不改变由 1 ns SYS_CNT
cold/warm delta 得到的 `90 ns/miss` 标尺。

例如 AIV 平均 70,000 miss/核，感性等效量是约 6,300 us/核。这不表示
Submit 墙钟真的损失了 6.3 ms，原因包括：

- 64 个 AIV 之间并行；
- 同一核的 miss、预取、其他流水和等待可能重叠；
- 隔离 cold miss 与真实热路 capacity/conflict/compulsory miss 不一定同价；
- 当前已核实的 A5 事件中没有可直接换算墙钟损失的 I-cache
  stall-cycle counter。

把 AIC 和 AIV 的 `Σmiss * 90 ns` 相加，只能得到全部核的感性
core-work 等效总量，不是端到端 Submit 总损失。要测真正暴露的性能
损失，必须对同语义代码做交错 A/B：

1. 用相同 `submit-pmu none` 口径确认 `ΔAIC/AIV miss/core`；
2. 另用不开 PMU/泳道的性能构建测 `ΔSubmit span`；
3. 只有第 2 项是实际暴露的墙钟收益；第 1 项用于证明收益与 I-cache
   变化同时出现，`90 ns` 仅提供一阶数量级解释。

### 9.1 历史 b256 `none` 参考数据

2026-07-19 用当时的 `submit-pmu none` ELF、`real-compute/6,28,4,1`
在 A5 上采集一轮，全局首末 Submit span 为 `4.750810 ms`。96 核 raw、owner
Restore、selector、counter 阈值和离线复算全部通过。
该 b256 只作归档数据，不是后续重跑要求。

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

最后一行只是单核串行等效标尺，不能与 `4.750810 ms` 相减或解释成
端到端损失；非 Scalar-busy 残余也不是空闲或 I-cache stall。原始件和
自包含报告位于：

```text
outputs/submit_pmu_none_20260719_b256_final/submit_icache_raw.json
outputs/submit_pmu_none_20260719_b256_final/submit_icache_report.html
```

### 9.2 历史 schema-v5 b256 分段时间参考数据

2026-07-19 在同一 A5、`real-compute/6,28,4,1`、b256 配置下重新独立采集
`none|claim|efdrain|materialize|register`。五轮均为 96/96 有效记录；四个
running phase 都是 1,280 calls/core，时间均满足 `0 < phase <= 同核 Submit`。
这些是历史归档件；后续边界迭代和重采默认只使用 b1。

| phase | Submit span | ALL 时间占比 | AIC 时间占比 | AIV 时间占比 |
| --- | ---: | ---: | ---: | ---: |
| `none` | 3.711584 ms | 不适用 | 不适用 | 不适用 |
| `claim` | 4.401747 ms | 12.0845% | 7.7096% | 14.2972% |
| `efdrain` | 3.592376 ms | 15.5068% | 20.2974% | 13.1974% |
| `materialize` | 6.770266 ms | 16.7186% | 15.4860% | 17.3556% |
| `register` | 4.086936 ms | 4.3089% | 3.6256% | 4.6568% |

每行都只解释自己的诊断 ELF。尤其 Materialize 的运行中边界读取显著改变了
该轮 Submit 时序；这些时间占比不能跨行相加，Submit span 也不能与 `none`
相减成局部净开销。对应权威 raw 和加工 HTML 为：

```text
pa_scheduler/outputs/submit_pmu_phase_time_v5_20260719/none_b256/
pa_scheduler/outputs/submit_pmu_phase_time_v5_20260719/claim_b256/
pa_scheduler/outputs/submit_pmu_phase_time_v5_20260719/efdrain_b256/
pa_scheduler/outputs/submit_pmu_phase_time_v5_20260719/materialize_b256/
pa_scheduler/outputs/submit_pmu_phase_time_v5_20260719/register_b256/
```

### 9.3 当前边界联动版 b1 门禁

2026-07-19 在普通泳道相邻边界收敛后，重新构建五个
`submit-pmu` ELF，并只跑 A5 b1。五轮均使用
`real-compute/6,28,4,1`，都通过 96/96 物理核、真计算输出、mixed 引擎观察、
PMU start/stop、owner Restore、phase call/time 和 primary/shadow 门禁：

| phase | 全局首末 Submit | calls | exact/bounded 核 | request/miss loss | phase 时间占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `none` | 113.821 us | 0/0 | 96/96 | 0/0 | 不适用 |
| `claim` | 61.689 us | 480/480 | 80/96 | 7/9 | 22.1744% |
| `efdrain` | 127.969 us | 480/480 | 92/96 | 4/0 | 9.7505% |
| `materialize` | 65.023 us | 480/480 | 96/96 | 0/0 | 41.7821% |
| `register` | 134.714 us | 480/480 | 90/96 | 6/0 | 8.5663% |

`exact/bounded` 中 exact 表示 shadow 与 primary 逐值相等，bounded 表示满足
`shadow <= primary`。五轮 bounded 都是 96/96，小幅 loss 已进入局部
lower/upper，没有被忽略。

该表只证明当前业务语义体与局部 PMU bracket 同步修正后仍严格
闭合。b1 的全局 Submit 易受冷启动、多核到达和局部边界扰动影响，
不用五行之间的时间差声称 phase 净成本或性能改善。权威 raw 和各自
HTML 位于：

```text
pa_scheduler/outputs/submit_pmu_boundary_sync_b1_20260719/{none,claim,efdrain,materialize,register}/
```

### 9.4 历史 O1 owner/PMU bring-up 证据

现行 `submit-pmu` schema 和 selector 分工建立前，直接 owner 曾用
empty、100,000 scalar NOP、2×100,000 scalar NOP 验证 gate 可以闭合并继续累计；
96 核 PMU raw total 中位数约为 214、56,568、112,994。它们只证明链路响应和
近似倍增，不表示同数值的纳秒或硬件 cycle。

同一历史版本还采过三个独立 b256 `submit-all` PMU-only 进程：Submit span 为
3.688236/4.089057/4.673237 ms，I-cache request 总和约 69.45M～70.07M，
miss 总和约 5.83M～5.85M，整体 miss rate 为 8.32%～8.42%。这些旧统一 ELF
仍含当时的诊断代码，且 PMU gate 会改变多核时序；它们只保留为 owner、96 核
拓扑和 raw→summary 演进证据，不覆盖 9.1～9.3 的现行构建口径。历史文件为：

```text
pa_scheduler/outputs/scalar_observation_final_20260718/pmu_submit_all_ccec_b256_v2/pmu_submit_all.json
pa_scheduler/outputs/scalar_observation_final_20260718/pmu_submit_all_ccec_b256_v2_run2/pmu_submit_all.json
pa_scheduler/outputs/scalar_observation_final_20260718/pmu_submit_all_ccec_b256_v2_run3/pmu_submit_all.json
```

## 10. 新增局部 phase 的修改清单

新 phase 不能只增加一个 CLI 字符串。最小完整修改包括：

1. `pa_scheduler/common/pa_model.h`：在 `SubmitPmuPhase` 尾部追加稳定
   id，不重排已有 `None=0/Claim=1/EfDrain=2/Materialize=4/Register=5`。
2. `pa_scheduler/ccec/pmu_probe.h`：为 `SubmitPmuPhaseName()` 增加名称映射，
   并核对 phase status/边界闭合定义。
3. `pa_scheduler/ccec/build.sh`：在白名单中将 phase 名映射到稳定
   `PA_SUBMIT_PMU_PHASE_ID`；先校验名称，再用于输出目录。
4. `pa_scheduler/run.sh`：同步 `build-submit-pmu/submit-pmu` 的 phase
   白名单和 usage。
5. `pa_scheduler/common/pa_scheduler_core.h`：在真实目标代码段前后放置
   `BeginSubmitPmuPhase<...>()` / `EndSubmitPmuPhase<...>()`。必须检查所有
   早退、winner/loser 和 Alloc/非 Alloc 分支，不得留下只 begin 不 end
   的路径。
6. `pa_scheduler/ccec/host.cpp`：增加该 phase 的预期 calls/begin/end
   形状。如果它不是每次 Submit 都调用，不能复用
   `batches * 5 * 96`。
7. `pa_scheduler/pmu_sidecar_analyzer.py`：同步 phase 名/id 和配置指纹，
   保证不同 phase 输入不会被聚合。
8. 补充 host/analyzer 回归：`none` 验证 96/96 primary-shadow 精确相等；
   running phase 验证逐核 bounded、loss/upper 公式、begin/end/calls 和语义，
   任一 shadow 反向大于 primary 都必须拒绝。新增 phase 的构建、门禁和迭代
   默认只跑 A5 b1。普通泳道仍复用相邻既有 end，不为了对齐 PMU
   而额外增加泳道 `SYS_CNT`。

每个 phase 必须是独立 ELF 和独立进程。不为了一次运行得到多个
phase，而在热路加运行时 phase switch 或多组 begin/end。

## 11. 常见问题与排错

### 提示缺少 submit-pmu 产物

确认 phase 名与构建命令一致：

```bash
./run.sh build-submit-pmu ccec none
./run.sh submit-pmu ccec none --device 0 --batches 1
```

不要用 `./run.sh build ccec` 代替；后者生成的是 `swimlane` 产物。

### host 提示 swimlane 构建不能采 PMU

这表示运行了 `build/ccec/pa_scheduler_host`。应通过
`./run.sh submit-pmu ...` 启动 phase 目录内的 host/kernel/SO 整套产物，
不要手工指向根目录 kernel。

### shadow miss 始终为 0

先检查 selector 是否错把 `0x35` 放到 CNT9。本机 A5 b1 已经反证
CNT9 路径；正式配置应为 CNT5 shadow miss、CNT8 shadow request、
CNT9 unused。

### `none` 不相等，或任意 phase 出现 `shadow > primary`

这两种情况都表示观察链路门禁失败，最终 JSON 不应发布。按下列顺序排查：

1. host/kernel/owner/dispatcher 是否来自同一 phase 目录；
2. owner 读回的 CNT5/CNT6/CNT7/CNT8 selector 是否与期望一致；
3. begin/end 数是否精确相等，是否有早退路径留下 armed phase；
4. 先缩到 b1；`none` 确认 96/96 exact，running phase 确认 96/96 bounded；
5. 检查可编程 counter 是否超过风险门槛。

running phase 出现小幅 `shadow < primary` 时，代码显式发布 loss 和局部
lower/upper；这不是 standalone 调度正确性异常。只有协议、数值输出、
placement/engine 也失败时，才应转向调度代码排查。

### owner 或 Restore 失败

不要在同一设备上并发运行另一个 standalone PMU owner、`msprof` PMU
会话或其他会改 selector 的进程。检查 CANN 环境、两个 AArch64 SO
是否在 kernel 同目录，以及 96 个可用 slot/32 个完整 triplet 是否闭合。

### JSON 拒绝覆盖

每个独立进程使用新文件名，并处理上次失败留下的同名 `.tmp`。
host 不会静默覆盖旧证据。

### 分析器报配置不一致

不要强行合并。逐项比较 phase、batches、winner workload/count/pattern、
selector、schema 和 build variant。重采相同配置的独立进程。

### miss rate 高，但 Submit 没有同比例变慢

这不构成矛盾。miss rate 是事件比例，不是 stall 时间比例；多核、
预取、流水重叠和资源等待都会改变实际暴露量。优先看 AIC/AIV
每核 miss、median/p95 和优化前后的 `Δmiss`，实际性能收益仍以无
PMU/泳道的交错 A/B 为准。
