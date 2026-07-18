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

Case1 的 task 不是五个彼此独立的占位符。standalone 会从 Tensor descriptor 的
owner 和本 worker 的 TensorMap 收集 producer，去重后构造下列 fanin 图：

| task | 直接 producer | fanin 数 |
| --- | --- | ---: |
| Alloc | 无 | 0 |
| QK | 无；query/key/block-table 是外部输入 | 0 |
| SF | QK | 1 |
| PV | SF | 1 |
| UP | Alloc、SF、PV；同一 producer 的多个 tensor 会去重 | 3 |

因此每个 batch 恰有 5 条 fanin 边，默认 256 batch 恰有 1,280 条。worker 在
EfDrain 中逐条读取 producer completion flag，只有全部 ready 才能执行 winner
负载；descriptor materialize、TensorMap lookup/register、fanin 去重、ring slot
拷贝和 completion 发布都走实际的 standalone 调度路径。

这里的“依赖对等”是 **Case1 调度依赖图对等**，不是 PA 数值数据流复刻。
`real-compute` workspace 的 QK/SF/PV/UP 都读取同一组受控输入，并写入各
worker-kind 的独占输出 tile；QK 的数值输出没有作为 SF 的真计算输入，SF/PV
的数值输出也没有继续串到后继真计算中。fanin 会真实约束执行次序，但 workspace
只校验每类 Cube/Vector 算术和角色路由。Case1 只有一个 block group 且
`q_loop=1`；通用 PA 的多 group、多 q-loop、跨迭代数据更新以及 joint/mixed
task 依赖当前均未覆盖，不能从本用例外推其 fanin 数量或时序。

有意保留的替代只有两类：

1. 三后端当前无参数默认使用 `real-compute`：CCEC 和 AscendC 在 A5 上让 QK/PV
   执行完整 Cube matmul，让 SF/UP 执行完整 Vector add/mul，并覆盖 GM load、
   引擎计算、GM store 和完成等待；CPU 使用同一 `128x128 float` 输入、输出布局
   执行普通浮点 matmul/add/mul，只用于回归算术、角色路由和输出闭环。
   `scalar-nop` 仍作为显式兼容/校准模式保留；历史 NOP 数据继续按当时口径解释。
2. simpler 的 AICPU/runtime 装载链路由本目录 host runner 代替；测试关注的
   首个 Submit 到最后一个 Submit 区间不含 AICPU 初始化和最终回收。

该对等范围只覆盖当前 PA Case1 的全单-lane 图，不宣称覆盖通用 FDWIC 的
joint/mixed task。若未来加入需要两个及以上 lane 联合执行的 task，必须补回
BlockWon 发布、claim、drain 和剩余计数协议后再谈语义对等。

因此，约 5 ms 是性能参考，不是通过条件。不能为了命中 5 ms 而在 Claim、
Register、PrepareMap 或等待路径中插入虚假延时。

## 2. 三种实现

三种后端共用 `common/` 中完全相同的 PA 模型和调度器，只分别实现原子指令、
时钟、winner 计算体和启动入口。

| 后端 | 启动形式 | atomic load | Claim fetch-max | 当前 winner 负载 |
| --- | --- | --- | --- | --- |
| CCEC | 1:2 mixed AIC/AIV ELF | `atomicAdd(addr, 0)` | `atomicMax` | `scalar-nop` 或 A5 Cube/Vector 真计算 |
| AscendC | `__mix__(1, 2)` | `AtomicAdd(addr, 0)` | `AtomicMax` | `scalar-nop` 或 A5 Cube/Vector 真计算 |
| CPU | 96 个 pthread | `fetch_add(0)` | C++17 CAS loop | `scalar-nop` 或 CPU 对等浮点算术 |

AscendC 对 64 位 vend 使用 signed `AtomicAdd(addr, 0)`。CANN 9.1 虽提供
unsigned overload，但它在本 mixed kernel 的 64 MiB heap wrap 位置发生过
稳定停滞；PA vend 小于 `INT64_MAX`，因此 signed add-zero 返回的位模式与比较
语义不变。CPU 版本用于协议、边界、算术和输出检查；它没有 A5
Cube/Vector 指令、流水线、GM 搬运和 PMU 语义，host pthread 耗时不能与 A5 比较或外推。

## 3. Winner 计算负载

### 3.1 三后端 `scalar-nop` 兼容/校准模式

真实 PA 最好泳道中四类 kernel 的 1 GHz counter 均值和当前 A5 NOP 校准值为：

| Kernel | 目标时间 | `scalar-nop` 标定数 |
| --- | ---: | ---: |
| QK | 44.170 us | 129,600 |
| SF | 53.729 us | 157,900 |
| PV | 27.626 us | 79,950 |
| UP | 1.565 us | 2,400 |

Alloc 没有模拟 kernel body。NOP 数是 A5 实测校准量，不应解释为 A5 cycle 数。
可以在运行时覆盖：

```bash
./run.sh run ccec --winner-workload scalar-nop --nop-count 100000
./run.sh run ccec --winner-workload scalar-nop \
  --nop-counts 129600,157900,79950,2400
```

`--nop-count N` 同时设置四类 kernel；`--nop-counts` 的顺序固定为
QK、SF、PV、UP，允许范围为 0 到 10,000,000。兼容旧命令时，显式提供
`--nop-count*` 而不写 `--winner-workload` 也会自动选择 `scalar-nop`；新脚本
建议像上面一样把模式写明，避免把校准样本误认成当前默认真计算。

### 3.2 三后端默认 `real-compute` 模式

三后端当前无参数运行即选择 `real-compute`。三后端共用同一组参数，可通过
`all` 按 CCEC、AscendC、CPU 的顺序统一运行；命令中显式写出模式仍然有效：

```bash
./run.sh run all \
  --winner-workload real-compute --batches 8 --runs 1 \
  --real-compute-count 1 --no-swimlane
```

也可单独选择后端：

```bash
./run.sh run ccec   --winner-workload real-compute --batches 256 --runs 1 --no-swimlane
./run.sh run ascendc --winner-workload real-compute --batches 256 --runs 1 --no-swimlane
./run.sh run cpu    --winner-workload real-compute --batches 1 --runs 1 \
  --real-compute-count 1 --no-swimlane
```

未传 count 时，QK/SF/PV/UP 默认使用 `6,28,4,1` 次完整迭代。这组默认值来自
CCEC 的 A5 标定；AscendC 需用自身实测解读，CPU 只复用参数含义而不复用
性能结论。每次迭代的输入、输出形状均为 `128x128 float`，不是 scalar NOP：

- CCEC/AscendC AIC 的 QK/PV：GM load → MTE2/MTE1 → Cube matmul → FIX
  → GM store → 完成等待；
- CCEC/AscendC AIV 的 SF/UP：GM load → Vector add/mul → GM store
  → 完成等待；
- CPU 的 QK/PV 执行三重循环 float matmul，SF/UP 执行 elementwise add/mul；
  这是数学与工作区布局对等，不是设备引擎对等。

可以统一或分类型覆盖，范围为 1 到 128：

```bash
./run.sh run all --winner-workload real-compute --real-compute-count 1 --no-swimlane
./run.sh run ascendc --winner-workload real-compute --real-compute-counts 6,28,4,1
```

`--real-compute-count*` 与 `--nop-count*` 不能混用；三后端全部使用相同的
范围 1 至 128、QK/SF/PV/UP 顺序和互斥规则。
真计算默认使用 `constant` 输入模式做性能测量；需要核验矩阵布局时使用：

```bash
./run.sh run all \
  --winner-workload real-compute --real-compute-count 1 \
  --real-compute-pattern layout-diagnostic --batches 1 --runs 1 --no-swimlane
```

`layout-diagnostic` 使用带权对角矩阵
`A[r,c]=(r==c ? r+1 : 0)` 和非对称稠密矩阵
`B[r,c]=1+((131r+17c+7rc) mod 251)`。QK/PV 逐元素校验
`(r+1)*B[r,c]`，SF/UP 分别校验 `A+B` 和 `A*B`，可直接发现 B 转置、
ND/NZ stride、分形重排或 FIXPIPE 输出重排错误。输入生成与校验都在 Submit
计时窗外，设备执行路径和 workspace ABI 不增加 pattern 分支。

workspace 包含两个只读输入 tile，并为每个 worker 按其 role 对应的两个 task kind
各保留一个独占输出 tile，共
12,713,984 bytes。CCEC/AscendC host 在计时前初始化并 H2D，计时后 D2H；
CPU 直接访问同布局的 host workspace。所有 active
worker-kind 的最终 tile 必须分别等于 QK/PV 的 768、SF 的 5、UP 的 6，未获胜
输出必须保留 sentinel。同一 worker-kind 的 repeat 会覆盖同一 tile，因此最终常量
结果只证明至少完成一次。CCEC 的 repeat 完整性另由受控 PMU count1→2
精确倍增取证；AscendC 不伪造 PMU，只将官方引擎循环、输出闭环与 count1/默认
`[KERNEL]` span 缩放合并作为证据；CPU 则在每次算术迭代后保留编译器物化边界。
性能默认输入使用常量 2 和 3，便于与既有 768/5/6 和 PMU 标定比较；它本身
不证明转置或 stride。正式布局闭环由上述 opt-in 诊断完成，实测见 6.4 节。

默认次数来自三个独立 CCEC b256 进程的标定：QK、SF、PV 中位数分别约
41.336、54.039、27.971 us，接近真实 PA 的 44.170、53.729、27.626 us。
UP 的一次完整 `128x128` 流水约 2.5 us，已经是正整数迭代下限；若后续要贴近
1.565 us，应缩小 UP tile，不能用 0 次掩盖执行。真实计算下 Cube/Vector 会在
不同物理子核并行；关闭泳道和开启标准泳道还会改变 worker 到达、fanin 重试与
RingBp，不能把两种观察布局的绝对时间直接相减。当前 `6,28,4,1` 优先贴近
真实 PA 的 **per-task core work**，不通过增加无关 repeat 硬凑 5.1 ms。
同观察口径的完整验收见 6.5 节；CPU 数值不进入 A5 性能对比。

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
`include/pto/common/kernel_meta.hpp` 的目录；同一 include tree 还必须具有
`pto/pto-inst.hpp`、`pto/common/constants.hpp` 和 `pto/common/pto_tile.hpp`。

## 5. 使用说明：运行、测量与泳道查看

以下命令均在 `pa_scheduler` 目录执行。首次使用应先按第 4 节 source CANN
环境并完成构建。四个 action 的用途如下：

| action | 用途 | 是否生成泳道文件 |
| --- | --- | --- |
| `build` | 构建指定后端 | 否 |
| `smoke` | 固定 b1/r1/`scalar-nop=0` 的快速语义回归 | 否，只做内存记录校验 |
| `run` | 自行控制 batch、run、winner 负载和诊断参数 | 仅显式传入 `--swimlane-json` 时生成 raw |
| `swimlane` | 单轮运行并自动生成 raw 和 Perfetto merged JSON | 是 |

`ccec|ascendc|cpu|all` 用于选择后端；`all` 始终按 CCEC、AscendC、CPU
的顺序执行。
CCEC、AscendC 和 CPU 的 `run/swimlane` 都可使用
`--winner-workload scalar-nop|real-compute`、
`--real-compute-count N`、`--real-compute-counts QK,SF,PV,UP` 或
`--real-compute-pattern constant|layout-diagnostic`；真计算 count/pattern
与 `--nop-count*` 互斥。`smoke` 有意固定 scalar-nop，不接受这些覆盖项。

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

这两条未传 `--winner-workload`，使用当前默认 `real-compute` 和
`6,28,4,1`。需要复现历史 NOP 基线时按 3.1 节显式传入
`--winner-workload scalar-nop`；CPU 使用同一选项。需同配置串行回归三后端
时，直接把 backend 换成 `all`，但 CPU 真计算只作协议/算术回归，不作 A5 性能值。

此模式输出指标和泳道统计，但不会自动落盘 JSON。若只关注性能且不需要完整
泳道逐事件分析，可去掉 `--analyze-swimlane`；若也不需要记录泳道，可使用
`--no-swimlane` 进一步节省约 384 MiB device 内存。

### 5.3 生成并查看泳道

生成可直接载入 Perfetto 的泳道文件时使用独立的 `swimlane` action：

```bash
./run.sh swimlane ccec \
  --device 0 --batches 256 --profile-phases --analyze-swimlane

./run.sh swimlane ccec \
  --device 0 --batches 1 --winner-workload real-compute --trace-atomics

./run.sh swimlane ascendc \
  --device 0 --batches 1 --winner-workload real-compute \
  --real-compute-count 1
```

`swimlane` action 会管理 `--runs 1` 和输出路径，因此不要再传
`--runs`、`--swimlane-json` 或 `--no-swimlane`。转换器默认使用 `python3`；如需
固定到用户自己的 Python，可在命令前设置：

```bash
export PYTHON=/path/to/venv/bin/python
```

转换器只使用 Python 标准库，不需要 PyTorch，也不需要安装 simpler Python 包。

该 action 固定执行一轮，产物全部位于本目录的
`outputs/pa_scheduler_swimlane_<UTC时间>_<PID>/<backend>/`。选择 `all` 时，
同一 output root 下会按顺序建立 `ccec/`、`ascendc/` 和 `cpu/` 三个子目录：

- `l2_swimlane_records.json`：与真实 PA 相同的十列 `fdwic_events` 原始格式；
- `merged_swimlane.json`：Chrome Trace Event 格式，拖入
  <https://ui.perfetto.dev/> 即可查看泳道。

runner 结束时会打印准确目录：

```text
[SWIMLANE] output_root=.../outputs/pa_scheduler_swimlane_<UTC时间>_<PID>
```

真计算泳道必须同时检查 raw/merged 顶层
`metadata.winner_workload`；其 `mode`、`counts`、`unit`、`input_pattern`
和 `engine_mapping`
分别应为 `real-compute`、实际 QK/SF/PV/UP 次数、
`complete_128x128_engine_pipeline_iteration`、实际输入模式以及
`qk/pv=cube_matmul、sf=vector_add、up=vector_mul`。逻辑
`·kernel` span 只说明 winner 执行区间，单凭轨道名称不能证明使用了硬件引擎。

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

`icache-single` 还在同一 worker 的诊断 sidecar 中保留下列配对字段；其他窗口
将它们写零，不应解读为额外的 Submit 计数：

| `WorkerResult` 原始字段 | 来源 | 含义 |
| --- | --- | --- |
| `pmu_window_ticks` | 1 GHz sys counter | cold 目标调用窗口，1 tick = 1 ns |
| `pmu_warm_total_cycles` | 64-bit PMU total | 同核 warm 对照窗口原始 total |
| `pmu_warm_window_ticks` | 1 GHz sys counter | warm 目标调用窗口 |
| `pmu_warm_icache_requests` | `CNT6 = 0x034` | 同核 warm 对照 request |
| `pmu_warm_icache_misses` | `CNT7 = 0x035` | 同核 warm 对照 miss |

#### Main AICPU Path-A owner

owner 已自包含在 `ccec/`：构建会同时产出 dispatcher 和 owner AArch64 SO。host
通过 CANN 9.1 已验证的 Main AICPU Path-A 完成 bootstrap 和 mode-0 注册，运行时
调用 `simpler_aicpu_exec` 执行 Configure/Restore；不要求用户另行启动 PMU 配置进程。
owner 对本轮会话独占的 PMU 状态先保存、再配置并读回，结束时只按成功 bitmap 从
107 到 0 逆序恢复。owner 当前没有跨进程互斥锁；同一设备上不得并发运行
另一个 PMU owner 会话或 `msprof` PMU 会话，否则 selector 和保存态可能互相覆盖，
本轮 Restore 也不再能代表恢复了启动前的状态。

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

`off` 是默认值，不建立 PMU owner 会话。五种非 off 模式中，前四种只用于
观察链路校准；`submit-all` 是唯一正式的 Submit 取数窗口：

| `--pmu-window` | 位置与窗口边界 | 用途 |
| --- | --- | --- |
| `empty` | `RunScheduler` 完成后，baseline read-clear → start → stop → 末尾 snapshot | 量一段空 gate 底噪 |
| `scalar` | `RunScheduler` 完成后，一个 gate 中执行 `--pmu-scalar-nops N` | 验证 scalar/I-cache 正向敏感性 |
| `scalar-double` | `RunScheduler` 完成后，两段相同 NOP 中间只 stop/start、不读 counter | 验证暂停后继续累计 |
| `icache-single` | `RunScheduler` 完成后，每 worker 做 cold/warm 配对试验；64 KiB sweep 或目标预热在目标 gate 外 | 标定受控单次 CNT7 miss 的一阶等效时间 |
| `submit-all` | 每 worker 通过启动屏障后，在首次 PA 参数构造前 start，最后一次 Submit 返回后 stop | 累计 orchestration、Submit，以及本 worker 在窗口内执行的 NOP 或真实引擎计算 |

`submit-all` 不含启动屏障，也不含 `replay_done`、最终 drain、末尾 ClockBaseline
或 PMU 结果发布。它是 **每 worker 从 orchestration 初始化前到本核最后一次 Submit
返回后的累计窗口**，不是全局“最早 `Submit.begin` 到最晚 `Submit.end`”墙钟 span。

每次 `metrics_prof_start/stop()` 都带 `PIPE_ALL` 流水屏障。该屏障用于明确门控边界，
也会收口流水并可能改变多核到达时序；因此 PMU 样本只能与相同 gate、相同负载模式、
相同构建配置的样本比较，不能把它当成无观察开销的端到端基线。`scalar-nop` 的循环
实际在 scalar 上执行并计入 scalar 事件；CCEC `real-compute` 则真实激活 Cube/Vector
及其搬运流水。两种口径不能混比，且 standalone sidecar 都不声称是完整真实 PA profile。

`icache-single` 的可执行标定命令为：

```bash
./run.sh run ccec \
  --device 0 --batches 1 --runs 1 --nop-count 0 --no-swimlane \
  --pmu-window icache-single --pmu-icache-trials 64
```

该命令用于校准并输出控制台结果，不生成正式 `submit-all` JSON。除通用
PMU/owner 断言外，每轮还必须看到：

```text
icache_pairs=96/96 calibrated_cores=96/96
[ASSERT] each cold trial adds exactly one CNT7 I-cache miss PASS
```

门禁逐核要求 cold 的 `CNT7 == trials`、warm 的 `CNT7 == 0`，并要求 cold-warm
时间差为正。也就是说，最终系数的分母不是推测的循环次数，而是严格闭合的
`CNT7` miss 差值。

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

上述命令没有选择 winner 模式，采集的是当前默认 `real-compute` 和
`6,28,4,1`。为了让 PMU 取证参数自描述，正式样本仍建议显式写出模式和次数；
例如已用于 count 倍增取证的 b8 命令：

```bash
./run.sh run ccec \
  --device 0 --batches 8 --runs 1 --no-swimlane \
  --winner-workload real-compute --real-compute-count 1 \
  --pmu-window submit-all --pmu-json "$OUT/pmu_real_b8_count1.json"
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

- `capture/configuration/validation`：采集边界、winner mode/count/unit/角色引擎映射、
  `PIPE_ALL` 门控语义、九个 selector、计数器位宽、实际 start/stop、96 条记录可信性
  及可编程 counter 风险门槛；real-compute 还记录数值输出与 placement/引擎闭合门禁；
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
泳道中某条 atomic 的属性。优化前后需保持相同源码观察布局、winner mode/count 和
owner 配置，
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

按当时的正式命令和当时默认 PA NOP，还完成了 3 个独立进程的 256 batch
PMU-only 验收。该段保留的是切换默认模式之前的历史样本。Submit span 为
3,688.236/4,089.057/4,673.237 us，中位数
4,089.057 us；I-cache request 总和为 69,812,583/69,451,706/70,065,443，miss
总和为 5,854,421/5,847,256/5,830,645，miss rate 为 8.3859%/8.4192%/8.3217%。
三轮均通过 96 核、owner/slot/role/triplet、start/stop、counter 风险门槛和 Restore，
并使用本用户 `.venv` 从 raw 记录独立重算 summary 一致。这三轮证明的是
同配置 PMU 取数可重复；由于 gate 包含 `PIPE_ALL` 且未与无 PMU 样本交错配对，
不应将它们与约 5 ms 无诊断基线直接相减。

#### 单次 CNT7 I-cache miss 的 scalar 一阶估算标尺

2026-07-18 的 96 核并发 cold/warm 配对中，15/15 轮均精确满足
`cold CNT7 == trials`、`warm CNT7 == 0`和 `calibrated_cores=96/96`。按每轮
`sum(cold_ticks-warm_ticks) / sum(cold_miss-warm_miss)` 计算，实测为：

| 规模 | ALL median（range） | AIC median（range） | AIV median（range） |
| --- | ---: | ---: | ---: |
| 64 trials/core × 10 | 86.596（86.532～86.792）ns/miss | 85.913（85.848～86.202）ns/miss | 86.938（86.861～87.086）ns/miss |
| 128 trials/core × 5 | 89.629（89.615～89.648）ns/miss | 92.100（91.984～92.267）ns/miss | 88.410（88.310～88.440）ns/miss |

同一时段的 64 与 128 trials，ALL 分别约为 86.6 与 89.6 ns/miss。统一取 90
只是便于总量级归因的保守取整，不是把两组实测改写成同一精确常数。64-trial 样本中
AIV 略高，128-trial 样本中则 AIC 更高；角色差值方向并不稳定，不建立
AIC/AIV 精确常数。只做总量级归因时，统一取整为：

```text
估算的 scalar I-cache miss 时间(ns) = CNT7_I-cache_miss_total × 90
估算的 scalar I-cache miss 时间(us) = CNT7_I-cache_miss_total × 0.09
```

例如 1,000 次 miss 约为 90 us，10,000 次约为 0.9 ms。若确实需要按角色
计算，应使用同一时段、同一运行中打印的 `[ICACHE-FORMULA-AIC/AIV]`，
不能跨时段套用上表的角色中位数。

`CNT7` 只报告 I-cache miss 总数，不区分 compulsory、capacity 和 conflict 原因；
三类都应计入上式。本探针用 64 KiB sweep 明确制造 capacity eviction，因此得到的
是 96 核并发条件下、cold 相对 warm 的一阶等效 miss penalty。它适合回答“这些
miss 大约能解释多少 scalar 时间”，不是硬件逐次给出的可加 stall；真实代码
中的预取、miss 重叠、不同下级命中位置和并发排队都会使实际关键路径偏离简单乘积。

本机未入库的原始输出保留在：

```text
tests/atomic_probe/pa_scheduler/outputs/pmu_validation/
  icache_single_64x10_20260718_085929_3232836_console.log
  icache_single_128x5_20260718_090151_3235468_console.log
```

这里验证的是观察手段，不是 PA 优化本身。`submit-all` 仍是正式调度取数的
唯一窗口；上述 90 ns/miss 只用于对其 CNT7 总量做一阶等效估算，不把校准
模式扩展成 Claim、EfDrain 等多个正式局部窗口。性能 A/B 仍要保持观察布局一致，
最终端到端收益以关闭 PMU 和泳道的独立进程结果为准。

## 6. 当前 A5 结果与真实 PA 的差异

2026-07-17 `scalar-nop` 阶段的一轮代表性结果如下。这是保留的历史
基线，不是后续 `real-compute` 数据；当时所有严格校验均为 PASS，kernel
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

表中阶段时长来自一轮代表值；括号内的“等待次数”则是三轮中对应角色的
全局中位数，不是每 worker 次数。Claim 和 EfDrain 已与真实 PA 同量级；
HeapGuard 只有 0 至 3 次偶发等待。当前主要
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

### 6.1 2026-07-18 CCEC 真实计算阶段结果

CCEC `real-compute` 已按“构建门禁 → b1 数值 → b256 标定 → PMU 倍增 →
泳道元数据”的顺序上板闭环：

- 最终 b256 默认 `6,28,4,1` 的 Submit 为 3,683.649 us，全部协议与
  192 个 active output tile 数值断言通过；此前三个独立进程 Submit 为
  3,808/3,555/3,706 ms，中位数 3.706 ms；
- b8、四类 count=1 时，Submit 窗口内 29 个 EfDrain 恰好对应 14 个非零
  AIC Cube worker 和 15 个非零 AIV Vector worker；AIC 每个 `cube_busy=8281`，
  AIV 每个 `vector_busy=936/937`；
- 独立 count=2 样本的 placement 为 28 个 EfDrain、4 个 FinalDrain，14 个 AIC
  和 14 个 AIV worker 的非零计数档位分别为 16562 与 1872/1874，恰为 count=1
  档位的两倍。获胜核会随调度变化，不能把两轮强行按同一 worker 配对；
- `submit-all` 明确排除 FinalDrain，所以 count1/count2 分别落在 FinalDrain 的
  3/4 个 kernel 不应出现在该窗口的引擎 PMU 中。每轮都由 placement 与非零引擎
  worker 数量闭合；
- scalar-NOP 与 real-compute 使用同一个最终 ELF 分别跑 b1，避免把代码布局变化
  误判为负载效果；最终 ELF 只暴露两个 mixed kernel 全局入口，冷路径 dispatcher
  以及 Cube/add/mul 三个执行 helper 均为非空 LOCAL 函数；
- real-compute b1 泳道 raw/merged 均记录 mode、`6,28,4,1`、完整迭代单位和
  QK/PV=Cube、SF/UP=Vector 的映射；4964 条 raw data event 转换为 4965 条
  data event（多一条 capture instant），最终 `traceEvents` 还含 256 条
  process/thread metadata，共 5221 条，且无 dropped record。

这组结果证明 standalone CCEC 的显式 `real-compute` 模式已不再用 scalar NOP
冒充 winner 计算，并能从数值、角色、PMU 和泳道四个方向闭环。这是
CCEC 后端当时的阶段结论；AscendC 和 CPU 后续已分步补齐，见 6.2 和 6.3 节。
三后端闭环仍不证明 standalone 的 3～5 ms 应等于真实 PA 的 5.1 ms；
真实引擎并行和代码生成差异都会改变调度时序。

### 6.2 2026-07-18 AscendC 真计算验证

AscendC 按 CCEC 之后独立接入同一 workspace 和参数规则。AIC 路径使用
`DataCopy/LoadData/Mmad/FIX` 完成 `128x128` matmul，AIV 路径使用
`DataCopy/Add/Mul` 完成 elementwise 计算；两条路径都在 task 发布前等待 GM
写回完成。分层上板结果为：

| 场景 | `submit_span_us` | QK/SF/PV/UP `[KERNEL] mean_us` | 数值输出 |
| --- | ---: | --- | --- |
| b1，count=`1,1,1,1` | 59.280 | 8.837 / 14.267 / 8.342 / 11.307 | 4 active + 188 sentinel，PASS |
| b8，count=`1,1,1,1` | 166.426 | 7.713 / 2.559 / 7.694 / 2.852 | 32 active + 160 sentinel，PASS |
| b256，count=`6,28,4,1` | 3810.471 | 41.232 / 48.047 / 27.940 / 2.528 | 191 active + 1 sentinel，PASS |

b256 三个独立进程的 Submit span 为 3810.471、4828.567 和 3777.371 us，
中位数为 **3810.471 us**。三轮都通过全部 PA 协议、角色路由和真计算输出
断言；数据离散也说明独立进程首轮需报告中位数，不能只挑最快值。

`[KERNEL] mean_us` 是公共调度器对每个 winner `ExecuteKernel` 前后的
1 GHz `SYS_CNT` span 求均值；b256 每类均为 256 个样本。它覆盖本 task 的
GM load、Cube/Vector 计算、GM store、完成等待及少量调用边界，因此是
**完整 winner 计算 span 均值**，不是纯 Cube/Vector busy counter，也不是 CCEC
PMU 计数。AscendC 当前不伪造 CCEC-only PMU sidecar；本阶段的取证是官方指令
接口、完整数值 tile、角色路由和上述 span 共同闭环。

已生成的 AscendC b1 count=1 泳道位于：

```text
tests/atomic_probe/pa_scheduler/outputs/
  pa_scheduler_swimlane_20260718_124326_3574886/ascendc/
    l2_swimlane_records.json
    merged_swimlane.json
```

该轮 raw 有 4652 条 data event，merged 增加一条 capture instant 后为 4653 条
data event，再加 256 条 process/thread metadata，`traceEvents` 合计 4909 条。
raw 和 merged 的 `metadata.winner_workload` 均为 `mode=real-compute`、
`counts=1,1,1,1`、`unit=complete_128x128_engine_pipeline_iteration`，且
QK/PV、SF、UP 分别映射到 `cube_matmul`、`vector_add`、`vector_mul`。
泳道轮次的协议、输出、record 数和 `dropped=0` 闭环全部 PASS。

### 6.3 2026-07-18 CPU 对等算术回归

CPU 在 AscendC 之后补齐了同一 `real-compute` CLI、workspace 编址、输入
2/3、active tile 结果 768/5/6 和 inactive sentinel 校验。b1 count=1 的
4 active + 188 sentinel 通过；b8 count=`2,3,2,1` 连续运行两轮，两轮的
25 active + 167 sentinel 均通过，同时验证了 runs 间输出会重置为 sentinel。

CPU 上的 matmul/add/mul 是普通 x86 浮点循环，`[KERNEL]`、`submit_span_us`
和泳道 `·kernel` 都会被 pthread 调度、CPU cache 和编译器影响。它们只用于
回归公共 PA 控制流与算术闭环，不得当作 A5 Cube/Vector 时间、PMU 或端到端
性能的估计。CPU 泳道里的 `engine_mapping` 是三后端共用的逻辑 workload
映射标签，不表示 x86 上存在对应物理引擎。

### 6.4 2026-07-18 非均匀布局诊断

常量 2/3 会掩盖 `B` 转置和分形重排错误，因此在不改变 `constant` 性能输入默认的
前提下增加了
`--real-compute-pattern layout-diagnostic`。按 CCEC → AscendC → CPU 的顺序，
三后端均以 b1、count=`1,1,1,1` 运行同一带权对角 A 和非对称 B，并逐元素扫描
4 个 active tile 与 188 个 inactive sentinel tile，全部 PASS：

| 后端 | Submit span | QK/SF/PV/UP `[KERNEL] mean_us` | 结论 |
| --- | ---: | --- | --- |
| CCEC | 37.682 us | 9.172 / 3.819 / 7.415 / 2.512 | PTO Cube/Vector 数学与布局基准 PASS |
| AscendC | 55.041 us | 8.478 / 4.211 / 20.801 / 3.008 | L1 错位、B 分形转置、ND/NZ 与 FIXPIPE 闭环 PASS |
| CPU | 51.958 ms | 3.457 ms / 26.118 us / 3.390 ms / 17.349 us | host 期望公式与路由回归 PASS；不作 A5 性能数据 |

AscendC 诊断泳道位于
`outputs/pa_scheduler_swimlane_20260718_125904_3613100/ascendc/`：raw 为
4584 条 data event，merged 增加一条 capture instant，`dropped=0`；raw/merged
的 `metadata.winner_workload.input_pattern` 都是 `layout-diagnostic`。CCEC 同模式的
`submit-all` PMU sidecar 也记录该字段且 `accepted/semantic_passed=true`。

诊断只证明单次完整迭代的数学与数据布局闭环；默认常量性能负载仍用于稳定比较，
CCEC 的 repeat 次数证明仍以 count1→count2 的 engine PMU 精确倍增为准。

### 6.5 2026-07-18 默认真负载与 5 ms 口径验收

三后端重编后，无 workload 参数的 b1 均打印
`mode=real-compute pattern=constant counts=6,28,4,1`，并通过 96 核、5 个
fanin/batch、唯一 winner、角色路由、TensorMap/heap/completion 与数值输出断言。
旧命令只给 `--nop-count 0` 时会显式打印 `mode=scalar-nop`，证明兼容
分支没有让 NOP override 静默失效。

CCEC b256 关闭泳道的 3 个独立进程为 4,411.760/4,297.704/4,677.634 us，
中位数 4,411.760 us；它只用于无观察热路对比。与真实 PA 5.1 ms 比较时，
必须同样开启标准泳道且不开逐 atomic；standalone 5 个独立进程为：

```text
5002.413 / 4875.193 / 4968.894 / 4992.477 / 4876.282 us
```

中位数为 **4,968.894 us**。真实 PA 最终三轮为
5,115.620/5,145.057/5,096.685 us，中位数 **5,115.620 us**；同口径差值
146.726 us，约 2.87%，已满足独立调度复现目标。五轮 standalone 的
QK/SF/PV/UP 每 task 均值中位数为 41.461/54.007/28.053/2.649 us，
与真实 44.170/53.729/27.626/1.565 us 的总 core work 接近，不再调整
repeat 追求逐微秒一致。

只保留一轮标准泳道原始证据：

```text
outputs/performance_gap_20260718/standalone_ccec_real_b256_raw.json
```

该轮有 863,237 条记录、`dropped=0`，比真实 PA 的 863,232 条只多 5 条
RingBp；两端的 122,880 个 Submit 与各前端阶段、1,024 个 Kernel/Fanin/Build
数量一致。这证明总体性能已接近，不等于真实 PA 数值数据流、代码生成与通用
多 group/joint 调度已完全相同。

## 7. 内存占用和脱仓复制

为保持真实 DistGlobal/DistCore 偏移、65,536 个 task cell、每 worker payload
和 TensorMap，当前 `WorkerResult` 为 832 bytes，用当前头文件实际编译得到的
`SchedulerState` 为 1,007,104,896 bytes。新增的 64 bytes 是独立、对齐的
winner workload 配置 cache line，生产 DistGlobal/DistCore 关键偏移保持不变。
默认泳道缓冲区另占 402,660,160 bytes；CCEC/AscendC `real-compute` 还在 device 分配
12,713,984 bytes workspace。因此 scalar-nop+trace 的 A5 device 占用约
1.313 GiB，real-compute+trace 约 1.325 GiB，host 侧也需分配相近内存。
CPU 后端只在 host 侧分配相同 workspace，不存在 device 内存口径。
`smoke` 不缩小 State；只有 `--no-swimlane` 能省去泳道缓冲区。
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
