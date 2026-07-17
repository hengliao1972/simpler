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
5. 点击事件可查看 `task_id`、`func_id`、`core`、`mc` 和 `aux`。

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
