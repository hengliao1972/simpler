# PA 调度器独立复现与泳道使用指南

## 1. 目标与边界

本目录复现的是 A5 FDWIC Paged Attention Case1 的 **PA Submit 调度行为**，
不是把 PA 简化成普通 NOP 并发压测。整个目录复制到 `simpler` 代码仓之外后，
不再包含对 simpler 头文件、库、Python、PyTorch 或虚拟环境的编译和链接依赖。
泳道转换脚本也位于本目录，仅使用 Python 标准库，不 import `simpler_setup`
或任何目录外模块。

独立实现保留了当前 PA Case1 与调度性能相关的完整路径：

- 32 个 AIC worker、64 个 AIV worker，物理启动比例为 1:2；
- 默认 256 个 batch，每 batch 依次提交 Alloc、QK、SF、PV、UP 五个 task；
- 96 个 worker 各自回放 1,280 次 Submit，共 122,880 次 Submit；
- Alloc 由 96 个 worker 竞争，QK/PV 由 32 个 AIC 竞争，SF/UP 由 64 个 AIV 竞争；
- 4 路 Alloc/cube/vector Claim cursor，以及实际 `atomicMax` Claim；
- PA 的 TaskArgs、Tensor、TaskPayload、DistSubmitCtx、DistCore/DistGlobal 关键 ABI 布局；
- tensor tag 扫描、输出 layout、materialize、TensorMap retire/lookup/insert、register mask；
- fanin 收集、winner/loser、Replay、私有 ring slot 构造和 tensor/scalar payload 拷贝；
- EfDrain、WaitForSlot、HeapGuard、completion flag、vend、frontier、最终 drain；
- 与真实 PA 相同的单 lane 优化：Case1 不执行 BlockWon 轮询；
- 与真实泳道格式对齐的阶段记录及严格的结束状态校验。

默认工作量固定产生 73,728 次 Claim、1,280 个 winner 和 1,024 次 kernel
执行。每次运行都会校验这些数量以及最终 TensorMap、heap、cursor、flag、vend、
frontier 和 worker 状态，任一不符都会返回失败。

有意保留的替代只有两类：

1. QK/SF/PV/UP 的真实计算体由可控 NOP 数模拟，使每个 task 的占用时间接近
   真实 PA；调度前后路径没有用 NOP 补时。
2. simpler 的 AICPU/runtime 装载链路由本目录 host runner 代替；测试关注的
   首个 Submit 到最后一个 Submit 区间不含 AICPU 初始化和最终回收。

该对等范围只覆盖当前 PA Case1 的全单-lane 图，不宣称覆盖通用 FDWIC 的
joint/mixed task。若未来加入需要两个及以上 lane 联合执行的 task，必须补回
BlockWon 发布、claim、drain 和剩余计数协议后再谈语义对等。

因此，约 5 ms 是性能参考，不是通过条件。不能为了命中 5 ms 而在 Claim、
Register、PrepareMap 或等待路径中插入虚假延时。

## 2. 三种实现

三种后端共用 `common/` 中完全相同的 PA 模型和调度器，只分别实现原子指令、
时钟、NOP 和启动入口。

| 后端 | 启动形式 | atomic load | Claim fetch-max |
| --- | --- | --- | --- |
| CCEC | 1:2 mixed AIC/AIV ELF | `atomicAdd(addr, 0)` | `atomicMax` |
| AscendC | `__mix__(1, 2)` | `AtomicAdd(addr, 0)` | `AtomicMax` |
| CPU | 96 个 pthread | `fetch_add(0)` | C++17 CAS loop |

AscendC 对 64 位 vend 使用 signed `AtomicAdd(addr, 0)`。CANN 9.1 虽提供
unsigned overload，但它在本 mixed kernel 的 64 MiB heap wrap 位置发生过
稳定停滞；PA vend 小于 `INT64_MAX`，因此 signed add-zero 返回的位模式与比较
语义不变。CPU 版本用于协议和边界检查，host 线程调度耗时不能与 A5 比较。

## 3. 默认 kernel 时间

真实 PA 最好泳道中四类 kernel 的 1 GHz counter 均值和当前 A5 NOP 校准值为：

| Kernel | 目标时间 | 默认 NOP 数 |
| --- | ---: | ---: |
| QK | 44.170 us | 129,600 |
| SF | 53.729 us | 157,900 |
| PV | 27.626 us | 79,950 |
| UP | 1.565 us | 2,400 |

Alloc 没有模拟 kernel body。NOP 数是 A5 实测校准量，不应解释为 A5 cycle 数。
可以在运行时覆盖：

```bash
./run.sh run ccec --nop-count 100000
./run.sh run ccec --nop-counts 129600,157900,79950,2400
```

`--nop-count N` 同时设置四类 kernel；`--nop-counts` 的顺序固定为
QK、SF、PV、UP，允许范围为 0 到 10,000,000。

## 4. 本机依赖和构建

CCEC 与 AscendC 使用本用户安装的 CANN 9.1。非交互 shell 不保证读取
`~/.bashrc`，复现时建议显式执行：

```bash
cd /path/to/pa_scheduler

source /home/q00473782/Ascend/cann-9.1.0-weekly-20260708/cann/set_env.sh

export GCC15_ROOT=/home/q00473782/.local/gcc-15/root
export PATH="$GCC15_ROOT/usr/bin:$PATH"
export LD_LIBRARY_PATH="$GCC15_ROOT/usr/lib/x86_64-linux-gnu:$GCC15_ROOT/usr/lib/gcc/x86_64-linux-gnu/15${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CXX="$GCC15_ROOT/usr/bin/g++-15"

./run.sh build all
```

组合构建严格按 CCEC、AscendC、CPU 的顺序执行。CCEC 构建会检查 1:2 mixed
的两个入口和 metadata section；CANN 9.1 自带的 PTO 头可直接使用。若换用单独
安装的 PTO ISA，可把 `PTO_ISA_ROOT` 指向包含
`include/pto/common/kernel_meta.hpp` 的目录。

## 5. 使用说明：运行、测量与泳道查看

以下命令均在 `pa_scheduler` 目录执行。首次使用应先按第 4 节 source CANN
环境并完成构建。四个 action 的用途如下：

| action | 用途 | 是否生成泳道文件 |
| --- | --- | --- |
| `build` | 构建指定后端 | 否 |
| `smoke` | 1 batch、1 run、零 NOP 的快速语义回归 | 否，只做内存记录校验 |
| `run` | 自行控制 batch、run、NOP 和诊断参数 | 仅显式传入 `--swimlane-json` 时生成 raw |
| `swimlane` | 单轮运行并自动生成 raw 和 Perfetto merged JSON | 是 |

`ccec|ascendc|cpu|all` 用于选择后端；`all` 始终按 CCEC、AscendC、CPU
的顺序执行。

### 5.1 首次回归

先做三后端快速语义回归：

```bash
./run.sh smoke all --device 0
```

通过标准是所有 `[ASSERT]` 为 `PASS`，最终同时出现：

```text
semantic_status=PASS postprocess_status=PASS
```

`semantic_status` 表示 PA 调度协议和终态校验结果；`postprocess_status` 表示
泳道读取、导出或转换等后处理结果。任何一个为 `FAIL`，进程都会返回非零。

### 5.2 只运行 benchmark 和文字诊断

再在真实 A5 上测完整 CCEC 或 AscendC：

```bash
./run.sh run ccec \
  --device 0 --batches 256 --runs 5 \
  --profile-phases --analyze-swimlane

./run.sh run ascendc \
  --device 0 --batches 256 --runs 5 \
  --profile-phases --analyze-swimlane
```

此模式输出指标和泳道统计，但不会自动落盘 JSON。若只关注性能且不需要完整
泳道逐事件分析，可去掉 `--analyze-swimlane`；若也不需要记录泳道，可使用
`--no-swimlane` 进一步节省约 384 MiB device 内存。

### 5.3 生成并查看泳道

生成可直接载入 Perfetto 的泳道文件时使用独立的 `swimlane` action：

```bash
./run.sh swimlane ccec \
  --device 0 --batches 256 --profile-phases --analyze-swimlane
```

`swimlane` action 会管理 `--runs 1` 和输出路径，因此不要再传
`--runs`、`--swimlane-json` 或 `--no-swimlane`。转换器默认使用 `python3`；如需
固定到用户自己的 Python，可在命令前设置：

```bash
export PYTHON=/path/to/venv/bin/python
```

转换器只使用 Python 标准库，不需要 PyTorch，也不需要安装 simpler Python 包。

该 action 固定执行一轮，产物全部位于本目录的
`outputs/pa_scheduler_swimlane_<UTC时间>_<PID>/ccec/`：

- `l2_swimlane_records.json`：与真实 PA 相同的十列 `fdwic_events` 原始格式；
- `merged_swimlane.json`：Chrome Trace Event 格式，拖入
  <https://ui.perfetto.dev/> 即可查看泳道。

runner 结束时会打印准确目录：

```text
[SWIMLANE] output_root=.../outputs/pa_scheduler_swimlane_<UTC时间>_<PID>
```

查看步骤：

1. 打开 <https://ui.perfetto.dev/>；
2. 将 `merged_swimlane.json` 拖入页面，不要拖原始的
   `l2_swimlane_records.json`；
3. 每个 `block0` 至 `block31` 是一个物理 1AIC+2AIV block；
4. `AIC`、`AIV0`、`AIV1` 轨展示 Submit、Claim、EfDrain、Replay、RingBp 等
   runtime 阶段，带 `·kernel` 的轨展示 QK、SF、PV、UP；
5. 开启 `--trace-atomics` 时，Atomic 及 ClockBaseline 直接画在对应
   `AIC/AIV` scalar lane，名称与 category 显式区分 `return_ready` 和
   `source_issue`；不生成带 `·atomic` 的伪并行子轨；
6. 普通事件可查看 `task_id`、`func_id`、`core`、`mc` 和 `aux`；Atomic 的
   字段和解读边界见 5.6 节。

WaitForSlot 和 HeapGuard 没有可伪造的逐事件起止时间，因此不单独生成 Perfetto
事件；它们由 `--profile-phases` 的 `[PHASE]` 累计统计呈现。实际发生等待时，
泳道中会出现 RingBp 事件。

`outputs/` 已被 Git 忽略，生成的几十至数百 MiB 泳道文件不会被普通
`git add` 意外纳入提交。

### 5.4 手工导出或重新转换

转换完全由本目录脚本完成，不依赖 simpler 的 Python 包或虚拟环境。已有原始
文件也可单独转换：

```bash
python3 ./swimlane_converter.py \
  ./outputs/<capture>/ccec/l2_swimlane_records.json \
  -o ./outputs/<capture>/ccec/merged_swimlane.json
```

若只需要原始记录，可通过通用 `run` action 显式指定文件；导出为避免多轮覆盖
而要求 `--runs 1`：

```bash
mkdir -p ./outputs/manual
./run.sh run ccec --device 0 --batches 256 --runs 1 \
  --swimlane-json ./outputs/manual/l2_swimlane_records.json
```

手工 `--swimlane-json` 只生成 raw，不会自动生成 merged；需要随后调用上面的
`swimlane_converter.py`。该参数强制要求 `--runs 1`，避免多轮静默覆盖同一文件。

### 5.5 CPU 回归、参数与测量口径

CPU 完整协议回归建议关闭大泳道缓冲区：

```bash
./run.sh run cpu \
  --batches 256 --runs 1 --nop-count 0 \
  --profile-phases --no-swimlane
```

主要选项：

- `--profile-phases`：分别统计 Claim、EfDrain、WaitForSlot、HeapGuard；
- `--analyze-swimlane`：读取完整记录，输出各阶段的 per-worker 累计分布以及
  EfDrain/Materialize/Claim/Register 的 per-role、per-task-kind 单事件分布；
- `--trace-atomics`：在已开启的泳道中记录每次源码 atomic 调用括号；建议同时
  传 `--analyze-swimlane` 输出按 AIC/AIV、调用点分组的分布。不能与
  `--no-swimlane` 同用；
- `--swimlane-json FILE`：流式导出原始 `fdwic_events` JSON，要求单轮运行；
- `--no-swimlane`：关闭泳道记录，不能与 `--analyze-swimlane` 或
  `--swimlane-json` 同时使用；
- `--runs N`：同一进程和同一已装载 kernel 中连续运行 N 次；
- `smoke`：1 batch、1 run、零 NOP，但仍启动全部 96 个 worker 并执行全部校验。

`submit_span_us` 的定义与当前 PA 基线口径一致：

```text
96 个 worker 中最早的第一个 Submit.begin
    -> 96 个 worker 中最晚的最后一个 Submit.end
```

它排除启动屏障和最终 drain。`host_launch_us` 另外包含 host launch、最终 drain
和 stream/thread 同步。不同版本比较时，必须同时开启或同时关闭
`--profile-phases` 和泳道，因为计时与记录本身会影响竞争时序。

`[PHASE]` 的每个阶段都是每 worker 在 1,280 次 Submit 中的累计时间：

- Claim：当前 worker 实际参与或跳过对应 lane Claim 的完整 span；
- EfDrain：每次 Submit 开头执行已就绪私有 slot 的时间；
- WaitForSlot：仅 1,024 个 kernel winner 调用，额外给出发生等待的事件数；
- HeapGuard：Alloc/QK/SF/PV 的 1,024 个输出 winner 调用，额外给出 heap
  等待事件数。

WaitForSlot 和 HeapGuard 的 `calls_total` 会按 winner 所在角色分布，AIC/AIV
相加必须分别等于 1,024；等待事件则对应额外的 RingBp 泳道记录。

每轮还会输出一行不依赖泳道的动态原子分类：

```text
[ATOMIC] submit_completion_ops=... fanin_ready=... fanin_not_ready=... \
frontier_initial=... frontier_flag=... frontier_ready_fetch_max=... frontier_terminal=...
```

- `fanin_ready/not_ready` 分别是依赖 flag 返回 1/0 的次数，两者之和等于
  `[METRIC] fanin_loads`；
- `frontier_initial` 是每个 completion 对 frontier 的首次 load；
- `frontier_ready_fetch_max` 同时计数 ready flag 和紧随其后的 FetchMax，两者在
  这条控制流中一一对应；CCEC/AscendC 上它是一条真实 A5 atomicMax，CPU 上只是
  一次逻辑 FetchMax 调用；
- `frontier_terminal` 是每次扫描最终遇到的 not-ready flag，当前工作量下应与
  completion 数相等；`frontier_flag = ready + terminal`；
- `submit_completion_ops` 覆盖 Claim、第一圈 HeapGuard、fanin、completion 发布和
  frontier，不包含 started/replay_done 生命周期屏障。

这些字段在每个 worker 的私有 `LocalStats` 中递增，kernel 结束时才发布到独占
结果区，不为诊断新增共享 atomic。它们仍会增加少量 scalar 指令，因此优化 A/B
必须使用相同的计数布局；不能把启用分类后的绝对时间直接与旧二进制比较。

### 5.6 逐 atomic 语义边界泳道

`--trace-atomics` 记录 standalone 调度器中每一次动态 atomic 调用，不只记录
winner 或慢样本。生成带文字分析的 CCEC 泳道可直接执行：

```bash
./run.sh swimlane ccec \
  --device 0 --batches 256 \
  --trace-atomics --analyze-swimlane
```

转换后 Atomic 和 ClockBaseline 都放在对应 AIC/AIV 的原 scalar lane；
它们本来就是 scalar 指令，不再伪装成与 scalar 并行的独立子轨。
Kernel 仍放在独立计算单元轨。Atomic 事件名显式区分两种边界：

```text
atomic.return_ready.<site>.<op>#<task_id>
atomic.source_issue.<site>.<op>#<task_id>
```

category 也分别为 `atomic.return_ready` 与 `atomic.source_issue`，可在
Perfetto 中直接过滤，无需逐条点开 args 才能区分。

当前固定 schema 共有 15 个调用点：

| `site_id` | Perfetto `site` | `op` | 所属路径 |
| ---: | --- | --- | --- |
| 0 | `startup_increment` | `fetch_add` | 启动屏障到达计数 |
| 1 | `startup_poll` | `load` | 启动屏障轮询 |
| 2 | `fatal_poll` | `load` | fatal 状态检查 |
| 3 | `fatal_set` | `exchange` | fatal 状态发布 |
| 4 | `claim_max` | `fetch_max` | Submit lane Claim |
| 5 | `fanin_flag_load` | `load` | fanin 依赖 flag |
| 6 | `completion_vend_exchange` | `exchange` | completion vend 发布 |
| 7 | `completion_flag_exchange` | `exchange` | completion flag 发布 |
| 8 | `frontier_initial_load` | `load` | completion frontier 首次读取 |
| 9 | `frontier_flag_load` | `load` | frontier 扫描 flag |
| 10 | `frontier_max` | `fetch_max` | frontier 推进 |
| 11 | `heap_frontier_load` | `load` | HeapGuard frontier |
| 12 | `heap_vend_load` | `load` | HeapGuard vend |
| 13 | `replay_done_increment` | `fetch_add` | 回放完成屏障到达计数 |
| 14 | `replay_done_poll` | `load` | 最终 drain 中轮询回放完成 |

上表是源码调用点集合，不代表每轮都会出现 15 类事件；例如正常成功路径不应
执行 `fatal_set`。轮询点则会为每一次实际 load 生成独立事件。

原始 `fdwic_events` 仍是十列格式。对 `phase="Atomic"` 的记录，
`task_id` 是所属 PA task（启动/最终屏障等生命周期 atomic 为 -1），
`function_id` 固定为 -1，`auxiliary` 是上表 `site_id`，`flags` 使用独立 ABI：低 4 bit 是
`op_id`（Load/Exchange/FetchAdd/FetchMax 依次为 0/1/2/3），bit 4 表示
返回的旧值参与后续逻辑，bit 5 仅对 Load 表示本次读到零，bit 6
表示是否取得了“返回值本核可消费”边界，bits 8..31
仅对 FetchMax 保存饱和后的软件重试次数。CCEC/AscendC 的硬件
`atomicMax` 当前报告重试数 0；CPU CAS loop 才有可观察的软件重试。
merged 事件的 `args` 会显式导出 `site/site_id`、`op/op_id`、原始整数
`cycles`、`result_used`、`return_ready_observed`、`completion_boundary`，以及操作
适用时的 `value_zero` 或 `retries`。

边界必须按源码调用点语义解读：

- `source_issue_bracket`：返回旧值本来就丢弃的发布型调用。当前五处是
  `startup_increment/replay_done_increment/completion_vend_exchange/`
  `completion_flag_exchange/fatal_set`。结束时钟与旧值无依赖，只能表示
  源码发射包围区间。
- `return_value_ready`：协议本来就会判断返回值的 `Load/FetchMax`。CCEC
  在 atomic 后生成紧邻的 `dependent MOV -> MOV SYS_CNT`；这证明旧值已可被
  本核 scalar 消费，不证明其他核已看到发布的新值。

两种终点都早于本条 64-byte trace record 写入。当前不为每条 atomic
加 DSB/ISB/额外 GM load；这些操作要么后端不支持，要么会明显改写
被测路径。两种 bracket 都不能直接称为跨核可见或 atomic retire 延迟。

开启该诊断时，每个 worker 在最终 drain 之后还会写两条
`ClockBaseline`：`clock.consecutive_sys_cnt_reads` 和
`clock.atomic_return_dependency_hook`。前者量连续时钟读，后者量纯寄存器依赖 hook
的固定底噪。全局因此恰有 `96*2=192` 条，都只是分辨率参考，不是
可以从每条 Atomic 机械相减的校正常数。`[TRACE_ATOMIC]`
按 AIC/AIV、site 和 op 输出事件数、源码括号原始累计、中位数、p95 和最大值。

记录写本身不在它自己的 span 内，但会改变后续指令布局、cache、多核到达顺序和
atomic 争用；轮询次数也可能随之变化。所以不应将 `source_bracket_cycles_total`
与未插桩 Submit 时间相减，或用它计算 atomic 对 golden 的绝对占比。每核分区固定
容纳 65,536 条记录；通过必须同时满足 `dropped=0`、总记录数闭合，以及每 worker
新增诊断记录数精确等于 `atomic_trace_calls + 2` 闭合。CPU 线程调度可能放大启动轮询并撑满分区；这种
情况应视为本次 trace 无效，不能截断后继续分析。

### 5.7 CCEC 每核 PMU 与 I-cache sidecar

CCEC 后端提供与泳道分离的 PMU sidecar。正式取数由本目录自带的 Main AICPU
Path-A owner 配置 selector、保存并恢复 PMU 状态；kernel 在每个物理子核内门控并
读取 `CNT_TOTAL` 和 `CNT0..8`，再写入该 worker 独占的 `WorkerResult`。该链路不需要
目录外探针或整任务 profiler 的计数结果。完整字段为：

| sidecar 字段 | 寄存器 / selector | 原始事件含义 |
| --- | --- | --- |
| `total_cycles` | 64-bit PMU total | gate 窗口的 PMU total 原始计数 |
| `vector_busy` | `CNT0 = 0x501` | Vector pipe busy |
| `cube_busy` | `CNT1 = 0x301` | Cube pipe busy |
| `scalar_busy` | `CNT2 = 0x001` | Scalar pipe busy |
| `mte1_busy` | `CNT3 = 0x701` | MTE1 pipe busy |
| `mte2_busy` | `CNT4 = 0x202` | MTE2 pipe busy |
| `mte3_busy` | `CNT5 = 0x203` | MTE3 pipe busy |
| `icache_requests` | `CNT6 = 0x034` | I-cache request |
| `icache_misses` | `CNT7 = 0x035` | I-cache miss |
| `fix_busy` | `CNT8 = 0x714` | Fix pipe busy |

#### Main AICPU Path-A owner

owner 已自包含在 `ccec/`：构建会同时产出 dispatcher 和 owner AArch64 SO。host
通过 CANN 9.1 已验证的 Main AICPU Path-A 完成 bootstrap 和 mode-0 注册，运行时
调用 `simpler_aicpu_exec` 执行 Configure/Restore；不要求用户另行启动 PMU 配置进程。
owner 对本轮会话独占的 PMU 状态先保存、再配置并读回，结束时只按成功 bitmap 从
107 到 0 逆序恢复。

host 对同一 stream 调用 `aclrtGetStreamResLimit`，当前 A5 实报
`AIC=32/AIV=64/total=96`。owner 扫描 108 个物理 MMIO slot，只对完整读回一致的
96 个置 bitmap，跳过 12 个不可配置 slot；32 个可用物理组都必须形成
`1 AIC + 2 AIV` 完整 triplet。当前上板 bitmap 为：

```text
000003ff:fff3ff7f:f7cffcff:fffdf7ff
```

kernel 用真实 `get_coreid()` 查 host 通过 `halResMap` 建立的 MMIO 表，不用逻辑
`worker_id` 猜物理核。正式结果必须同时满足：九个 selector 全部匹配、96 条记录
可信且物理核 id 唯一、worker slot/role 与物理 triplet 对应、96 个核都实际执行过
start/stop、`miss <= request`，以及 owner Restore 成功。

`off` 是默认值，不建立 PMU owner 会话。四种非 off 模式中，前三种只用于观察链路
校准；`submit-all` 是唯一正式的 Submit 取数窗口：

| `--pmu-window` | 位置与窗口边界 | 用途 |
| --- | --- | --- |
| `empty` | `RunScheduler` 完成后，baseline read-clear → start → stop → 末尾 snapshot | 量一段空 gate 底噪 |
| `scalar` | `RunScheduler` 完成后，一个 gate 中执行 `--pmu-scalar-nops N` | 验证 scalar/I-cache 正向敏感性 |
| `scalar-double` | `RunScheduler` 完成后，两段相同 NOP 中间只 stop/start、不读 counter | 验证暂停后继续累计 |
| `submit-all` | 每 worker 通过启动屏障后，在首次 PA 参数构造前 start，最后一次 Submit 返回后 stop | 累计 orchestration、Submit 和模拟 kernel NOP |

`submit-all` 不含启动屏障，也不含 `replay_done`、最终 drain、末尾 ClockBaseline
或 PMU 结果发布。它是 **每 worker 从 orchestration 初始化前到本核最后一次 Submit
返回后的累计窗口**，不是全局“最早 `Submit.begin` 到最晚 `Submit.end`”墙钟 span。

每次 `metrics_prof_start/stop()` 都带 `PIPE_ALL` 流水屏障。该屏障用于明确门控边界，
也会收口流水并可能改变多核到达时序；因此 PMU 样本只能与相同 gate、相同 NOP、
相同构建配置的样本比较，不能把它当成无观察开销的端到端基线。当前模拟 QK/SF/PV/UP
的 NOP 循环实际在 scalar 上执行，`submit-all` 会把它计入 scalar 相关事件；这与真实
Vector/Cube task 运行时 scalar 等待的状态并不等价，所以 sidecar 不声称是绝对的
真实 PA profile。

#### 生成正式 PMU-only JSON

正式 sidecar 使用单轮、独立进程和唯一输出路径：

```bash
./run.sh build ccec

OUT="./outputs/pmu_submit_all_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
./run.sh run ccec \
  --device 0 --batches 256 --runs 1 --no-swimlane \
  --pmu-window submit-all \
  --pmu-json "$OUT/pmu_submit_all.json"
```

`--pmu-json` 只允许 CCEC 的非 off 窗口并强制 `--runs 1`。正式 JSON 还要求
`--no-swimlane`，且不能同时开启 `--trace-atomics`、`--profile-phases`、
`--analyze-swimlane` 或 `--swimlane-json`。已有目标文件或同名 `.tmp` 会被拒绝，
不会静默覆盖。host 只在协议、96 核 PMU/owner 门禁、Restore 和 runtime 清理全部
成功后，才把临时文件原子发布为最终 JSON。

`--no-swimlane` 会关闭 phase/atomic record 写和泳道后处理，但当前普通调度路径中的
阶段 `SYS_CNT` 调用点仍存在；JSON 会显式记录“有 phase timestamp call、无 phase
record write、无 atomic trace、无 profile accumulation”。因此该模式是 PMU-only
采集口径，不等于编译期删除所有时间戳指令的零观察二进制。

JSON 包含：

- `capture/configuration/validation`：采集边界、NOP 配置、`PIPE_ALL` 门控语义、九个
  selector、计数器位宽、实际 start/stop、96 条记录可信性及可编程 counter 风险门槛；
- `owner`：Main AICPU Path-A、配置 bitmap、AIC/AIV 数量、完整 triplet 和 Restore
  结果；
- `records`：每 worker 的物理子核 id、role、block/lane、原始 CNT0..8 和 total，
  以及本核 owner/slot/role、window started/stopped 断言；
- `summary.all/aic/aiv`：全部 96 核、32 个 AIC 和 64 个 AIV 分组后，各原始计数的
  `sum/mean/median/p95/max`。

控制台对应输出 `[PMU-ALL]`、`[PMU-AIC]`、`[PMU-AIV]`。I-cache miss rate 始终按
`sum(icache_misses) / sum(icache_requests)` 计算，不平均逐核百分比。AIC 与 AIV
核数不同，比较每核强度时应看 mean/median 或 miss rate，不能直接比较两组 sum。
`total_cycles` 是 PMU total 的原始值；96 核求和是 core-work，不是 Submit 墙钟
时间，在没有额外核实其时钟/事件语义前不应直接换算成微秒。CNT0..CNT8 是
32 bit，total 是 64 bit。正式门禁要求本轮最大可编程 counter 小于 `UINT32_MAX/4`
（25% 高水位），这只是缩短窗口后采用的保守风险阈值；最终 32-bit 值无法证明
计数器没有恰好回卷一圈或多圈，因此通过该门禁也不能声称“已证明无回卷”。

正式观察保留三类互不混用的样本：关闭所有诊断的性能 golden、PMU-only
`submit-all` sidecar、Atomic-trace-only 泳道。PMU sidecar 只有整个窗口的每核累计，
没有可与单条 Atomic span 对齐的子窗口；不能把 AIC/AIV 平均 miss rate 回填成
泳道中某条 atomic 的属性。优化前后需保持相同源码观察布局、NOP 和 owner 配置，
并用多个独立进程交错 A/B。

#### 2026-07-18 上板验收样本

自包含 owner 的本次验收中，`empty`、`scalar 100,000` 和
`scalar-double 2×100,000` 的 96 核 total 中位数分别约为 214、56,568 和
112,994；三个样本均为 `96/96 trusted`，owner 为 32 AIC + 64 AIV、32 个完整 triplet，
且每个样本 Restore PASS。它们是各模式的一次上板样本，不是多轮稳定性统计；只用于
确认空窗底噪、scalar 正向响应和双段近似倍增。

同一版本的 `batches=1,nop-count=0,submit-all` 单次样本也通过 96 核 start/stop、
selector、owner/slot/role/triplet 与 Restore 门禁：all/AIC/AIV 的 total 中位数约为
36,066.5/28,708/39,745，96 核 scalar busy 求和约 2,661,612，I-cache request/miss
求和为 210,399/30,283（约 14.3931%），host `submit_span_us` 约 47.770。该样本只
证明 `submit-all` 观察闭环可运行；`batches=1`、零模拟计算体和单次运行都不足以
支持 256 batch 性能归因或真实 PA 绝对结论。

按上述正式命令和默认 PA NOP 还完成了 3 个独立进程的 256 batch
PMU-only 验收。Submit span 为 3,688.236/4,089.057/4,673.237 us，中位数
4,089.057 us；I-cache request 总和为 69,812,583/69,451,706/70,065,443，miss
总和为 5,854,421/5,847,256/5,830,645，miss rate 为 8.3859%/8.4192%/8.3217%。
三轮均通过 96 核、owner/slot/role/triplet、start/stop、counter 风险门槛和 Restore，
并使用本用户 `.venv` 从 raw 记录独立重算 summary 一致。这三轮证明的是
同配置 PMU 取数可重复；由于 gate 包含 `PIPE_ALL` 且未与无 PMU 样本交错配对，
不应将它们与约 5 ms 无诊断基线直接相减。

## 6. 当前 A5 结果与真实 PA 的差异

2026-07-17 当前源码的一轮代表性结果如下。所有严格校验均为 PASS，kernel
时间使用上表附近的校准 NOP：

| 实现 | 首轮 `submit_span_us` | EfDrain/RingBp/FinalDrain |
| --- | ---: | ---: |
| 真实 simpler PA 最好泳道 | 5,096.685 us | 1011 / 0 / 13 |
| standalone CCEC，3 个独立进程首轮中位数 | 4,830.184 us | 961 / 50 / 12 |
| standalone AscendC，独立进程首轮 | 4,917.014 us | 1009 / 4 / 11 |

三次 CCEC 独立进程首轮分别为 4,846.431、4,798.260、4,830.184 us；这里
报告中位数，不挑最好值。`--runs N` 的后续轮次会复用进程、device binary 和
已分配内存，而真实 PA 的 5.1 ms 来自完整测试进程的一轮泳道，因此做基线
比较时应优先比较独立进程首轮，不能把热运行中位数混作同一口径。

四个重点阶段的一轮 CCEC 代表值为：

| role | Claim | EfDrain | WaitForSlot | HeapGuard |
| --- | ---: | ---: | ---: | ---: |
| AIC 每 worker 累计中位数 | 470.503 us | 711.005 us | 234.063 us，43 次等待 | 21.349 us，约 1 次等待 |
| AIV 每 worker 累计中位数 | 533.755 us | 401.962 us | 0.067 us，0 次等待 | 3.723 us，约 1 次等待 |

这里的“等待次数”是三轮中对应角色的全局中位数，不是每 worker 次数。Claim
和 EfDrain 已与真实 PA 同量级；HeapGuard 只有 0 至 3 次偶发等待。当前主要
残差来自编译边界：
真实 orchestration 和 `dist_submit_impl` 位于不同翻译单元，TaskArgs 对 Submit
编译器是运行时数据；standalone 为了保持可复制构建，共享实现会与固定 PA
任务图一起优化，导致 PrepareMap/Register 等前端阶段的指令生成不同。CCEC
的 AIC 到达顺序还会让约 39 至 54 个 kernel 进入 WaitForSlot/RingBp；AscendC
同轮只有 4 个，说明这部分主要是后端 codegen 触发的时序放大，不是缺少完成协议。

已经验证过的强制 `noinline`、拆设备目标文件和全局 compiler memory clobber
分别造成状态破坏、A5 device exception 或明显 RingBp，均未保留。它们不是
可靠的 PA 语义模拟。当前选择是保持源级协议和 ABI 对等，坦诚记录约
0.1 至 0.4 ms 的后端/冷热差异，不用虚假 NOP 填平调度阶段。

## 7. 内存占用和脱仓复制

为保持真实 DistGlobal/DistCore 偏移、65,536 个 task cell、每 worker payload
和 TensorMap，`SchedulerState` 为 1,007,092,544 bytes。默认泳道缓冲区另占
402,660,160 bytes，所以 A5 device 侧总计约 1.31 GiB，host 侧也需分配相近
内存。`smoke` 不缩小 State；只有 `--no-swimlane` 能省去泳道缓冲区。
256 batch 的正常采集约有 86 万条事件；原始 JSON 和 merged JSON 都可能达到
数十至数百 MiB。writer 与 converter 都使用临时文件后原子替换，失败时不会把
半截文件冒充完整产物。

脱离 simpler 时必须复制整个目录，因为三个后端共用 `common/`：

```bash
cp -a tests/atomic_probe/pa_scheduler /tmp/pa_scheduler
cd /tmp/pa_scheduler
./run.sh build cpu
./run.sh smoke cpu
```

CCEC/AscendC 只需再 source CANN 环境。本目录的构建脚本不会搜索 Git 根目录，
也不会引用 `simpler/src`、`simpler/examples` 或其他仓内文件。泳道转换只需
Python 3 标准库；复制后的 `./run.sh swimlane ...` 仍使用当前目录内的
`swimlane_converter.py`。
