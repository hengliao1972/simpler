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
- private 使用 production-prefix 的 4 路 Alloc/cube/vector Claim
  cursor；当前 shared 让 Vector 使用 sidecar 中全部 8 个 active
  shard，Cube/Alloc 仍使用 production-prefix 4 路；两种模式都执行
  实际 `atomicMax` Claim；
- PA 的 TaskArgs、Tensor、TaskPayload、DistSubmitCtx、DistCore/DistGlobal 关键 ABI 布局；
- tensor tag 扫描、输出 layout、materialize，以及按构建模式选择的 private
  每核有界桶环或 shared 有序桶环的 retire/lookup/insert、register mask；
- fanin 收集、winner/loser、Replay、私有 ring slot 构造和 tensor/scalar payload 拷贝；
- EfDrain、WaitForSlot、HeapGuard、completion flag、vend、frontier、最终 drain；
- 与真实 PA 相同的单 lane 优化：Case1 不执行 BlockWon 轮询；
- 与真实泳道格式对齐的阶段记录及严格的结束状态校验。

默认工作量固定产生 73,728 次 Claim、1,280 个 winner 和 1,024 次 kernel
执行。每次运行都会校验这些数量以及最终 TensorMap、heap、cursor、flag、vend、
frontier 和 worker 状态，任一不符都会返回失败。

两种 TensorMap 构建都先执行 `EfDrain` 和 `Claim`。private 随后保持
compete-first eager：每核构造五类完整 `TaskArgs` 并执行 per-worker
Materialize/map 前端。shared 则只保留 Alloc 的全员轻构参，QK/SF/PV/UP
只有 winner 构参和 Materialize；loser 只声明稳定 output symbol，并闭合
固定 finish/Submit 边界。CCEC 正式泳道构建将 orchestration caller、每核
runtime state 和 noinline finish 拆分为独立 TU；CPU 使用同一公共业务模板
做协议回归。本阶段只验收 CCEC 与 CPU，不把 AscendC 结果写进闭环证据。

Case1 的 task 不是五个彼此独立的占位符。standalone 会从 Tensor descriptor 的
owner 和当前构建模式的 TensorMap 收集 producer，去重后构造下列 fanin 图：

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

source /home/q00473782/Ascend/cann-9.1.0-weekly-20260708/cann-9.1.0/set_env.sh

export CXX=/usr/bin/g++

./run.sh build ccec --tensormap private
./run.sh build cpu --tensormap private
```

当前开发门槛只要求 CCEC 与 CPU，不要求 AscendC。用户目录中从 Ubuntu
plucky 解出的 GCC 15 会生成 binutils 2.42 不认识的 `.base64` 伪指令，
standalone host/CPU 回归必须显式使用已验证的系统 GCC 13；GCC 15 仍属于
A5sim 的独立工具链需求，不能把两种用途混为一谈。CCEC 构建会检查 1:2
mixed 的两个入口和 metadata section；CANN 9.1 自带的 PTO 头可直接使用。
若换用单独安装的 PTO ISA，可把 `PTO_ISA_ROOT` 指向包含
`include/pto/common/kernel_meta.hpp` 的目录；同一 include tree 还必须具有
`pto/pto-inst.hpp`、`pto/common/constants.hpp` 和 `pto/common/pto_tile.hpp`。

当前固定三条互不混算的证据链，不再生成同时夹带泳道、PMU 与性能基线
代码的统一 CCEC ELF：

| 构建 | 后端 | 内容 | 构建命令 | 产物目录 |
| --- | --- | --- | --- | --- |
| `swimlane` | CCEC/AscendC/CPU | schema-v4 普通阶段、业务父区间、真实 Submit 尾动作与 atomic（direct + PollBatch）合并采集；不配置 PMU | `./run.sh build ccec --tensormap <mode>` 或 `./run.sh build all --tensormap <mode>` | `build/<backend>/<mode>/swimlane/` |
| `perf-clock` | CCEC/CPU | 编译掉泳道、atomic 观察、phase-profile、PMU 和 kernel/lifecycle 计时；每核只新增首个 Submit 起点与末个 Submit 终点两个性能边界 | `./run.sh build-perf-clock ccec\|cpu --tensormap <mode>` | `build/<backend>/<mode>/perf-clock/` |
| `submit-pmu` | 仅 CCEC | 每核完整 Submit PMU，并在编译期可选一个局部阶段；当前有 `none|claim|efdrain|materialize|register` | `./run.sh build-submit-pmu ccec <phase> --tensormap <mode>` | `build/ccec/<mode>/submit-pmu/<phase>/` |

`./run.sh build all` 只构建三后端的 `swimlane` 产物；`perf-clock`
只支持当前验收范围内的 CCEC/CPU，必须逐后端构建；`submit-pmu` 必须按
phase 另行构建。`none` 是不做局部边界读取的完整 Submit PMU 窗口，
不能替代 `perf-clock`。

`--tensormap private|shared` 是由 `run.sh` 消费的构建身份，不会下传为
benchmark 运行时参数；省略时默认 `private`。它与 `swimlane`/
`perf-clock`/`submit-pmu` 正交，并进入产物目录、CCEC manifest 以及
host/device `magic + ABI version + mode + build variant +
sizeof(SchedulerState)` 握手。CCEC 的 swimlane/perf-clock 两件套和
submit-pmu 四件套都必须通过 manifest 的模式、变体、阶段和 SHA256
校验，才能启动 host。

`perf-clock` 的“两个时间边界”专指新增的性能观察：task 0 在 EfDrain 前
读取一次，末 task 完成 Submit 尾动作后读取一次。shared per-slot symbol
等待和 startup 屏障仍保留时间型 watchdog；每个等待窗口先读取一次超时起点，
随后只在每 1024 次未完成轮询时复查系统计数器。它属于防止协议永久挂死
的正确性机制，不应谎称整个 ELF 物理上只有两条 `SYS_CNT`。CPU 变体会
逐线程断言专用性能接口恰好调用两次，但 CPU 时间只验证协议和算术，
不能作为 A5 性能证据。

当前冻结并已保留的性能优化止于 S4.14b：在 S0～S4.6 的构建身份、
private/shared TensorMap 与 no-sequencer 基线上，shared no-wrap 完成
路径已移除无消费者的 frontier helping。S4.10～S4.13 的固定 owner、
早退、延迟解析、loser 快返和 `3×8` cursor 均已完成独立实验并因没有
稳定净收益而撤销。S4.14a 先把 shared Vector cursor 搬到物理容量 8、
active 仍为 4 的 sidecar；相对 S4.9 的冻结 ELF 配对为 6/6 区组更快、
百分差中位数 `-5.066%`。S4.14b 随后在相同地址、容量、state 大小和
寻址骨架下只改 active `4→8`，相对 S4.14a 为 6/6 更快、配对中位数
`-23.472%`；直接相对 S4.9 的净收益为 `-27.665%`。因此
standalone shared 冻结基线提交为 `319077a9`，其运行行为及冻结
S4.14b 性能口径与 `ee42b8c1` 一致。

S4.15a 曾仿照上述拆分方法，把 shared Cube 四分片迁到 sidecar
容量 8、active 4 的新字段。候选提交 `bab00e30` 的正确性全部闭合，
但相对 `ee42b8c1` 的六区组配对只有 1/6 更快，中位数
`+13.391us/+0.567%`，命中预登记的首轮撤销门槛且不追加第二轮。
该候选由 `319077a9` 回退；`319077a9` 回退后的冻结基线布局与
S4.14b `ee42b8c1` 一致。

S4.16 已按测量前预登记完成。S4.16a 曾把 shared Vector
物理容量从 8 扩成 16，active 仍为 8，
用来量化尾增 512B 带来的 state/GM/静态布局整体成本；它只是临时
布局控制，不单独长期保留。CPU shared/private、CCEC shared/private
构建和 A5 shared b1 已闭合正确性；相对重建 `319077a9` 的冻结
六区组配对为 4/6 更快、中位数 `-12.1595us/-0.5136%`。该布局成本
结果没有取消下一步。S4.16b 随后在相同地址、容量和 state 下只把
active 从 8 改为 16，并闭合正确性；但相对 S4.16a 仅 1/6 更快、
中位数 `+2.468us/+0.1049%`，第一层即失败。因此没有追加第二轮，
也没有执行相对 `319077a9` 的第二层，整套 S4.16 已回退。

完整历史证据和互斥判据见 `shared_tensormap_record.md`。后续 shared
TensorMap 开发与阶段门禁固定覆盖 CPU/CCEC。
`--tensormap private|shared` 都会生成对应模式的真实可执行文件；S0 用于禁止
伪 shared 产物的临时编译门禁已在 S2 接入 shared sidecar 后删除。两种模式仍
使用相互隔离的产物目录、manifest 和 host/device ABI 握手，不能混用镜像。

S0 还修复了一处独立问题：旧 CCEC PMU 配置以
`RunConfig::reserved[4]` 访问只有四项的数组，越界落入相邻 winner workload。
现在 PMU mode、work amount、寄存器表地址和 magic 位于独立的 64B
`PmuProbeConfig`，不再占用 `RunConfig` 尾部或覆盖业务配置。

### 4.1 当前 private TensorMap：128×128 有界桶环

S1 已把 standalone private TensorMap 从旧的 bucket linked map 同构为
ring-per-bucket，但没有同时改构参、heap、output ref 或真实 simpler runtime：

- 128 个 hash bucket，每桶 128 个连续 `MapEntry`，总容量仍为 16,384；
- 每桶使用普通 `uint64_t head/tail`，因为 map 仍由单 worker 独占，不使用
  atomic、per-slot `seq`、flush 或 invalidate；
- `MapEntry` 仍为 48 bytes，前 32 bytes 保存
  `(buffer_addr, lo, hi, producer)`，后 16 bytes 只保留 ABI，不提前塞入
  shared 发布状态；
- `TensorMap` 仍为 823,312 bytes，`WorkerState` 仍为 9,231,296 bytes；
  `WorkerState::map`、后续 ring slot 以及 payload 的 size/offset 均未移动；
- `AdvanceTensorMap()` 用 1,024 项 `task_entry_counts` 精确推进
  `alive_floor/cleaned_upto` 并扣减 `live_count`；桶的物理 `head` 在下一次
  lookup/insert 触达该桶时由 `RetireBucket()` 惰性推进，避免每个 task
  固定扫描 128 个桶；
- lookup 扫描该桶 `[head, tail)` 内全部合法槽，只接受
  `producer >= alive_floor` 的重叠区间，并返回最大的 producer；
- insert、existing insert 和 Register 均返回成功状态。满桶时不覆写 live
  槽、不推进 tail，Submit 把失败发布为 fatal 并终止，不能静默漏掉依赖。

PA Case1 在 `H=64` 下全 map 最多保留 52 个 live entry，所以当前每桶 128
槽有充分余量；这只证明当前 Case1，不是任意任务图的通用定容结论。shared
模式使用独立的并发发布、时序过滤、reclaim 与容量协议，见 4.2 节。

独立 ring 自测覆盖半开区间、最新 producer、`alive_floor` 边界、跨
`task_entry_counts` 多次回绕、满桶不覆写以及固定种子 differential，并通过
ASan/UBSan。CPU b1、CPU b256 的完整调度断言和 CCEC private 三镜像编译也已
通过。A5 本轮使用 CCEC private、关闭泳道、`real-compute`；b1 的 S0/S1
来自同一构建变体，Submit host span median 分别为 64.173/61.666 us。样本
太小，只能说明没有观察到回退，不能声称 2.507 us 是稳定收益。S1 b256
同口径单次为 3,862.246 us，只作协议和规模回归记录，不是性能基线，也不
替代后续配对多轮性能验证。

### 4.2 当前 shared TensorMap：per-slot symbol、shared heap 与隔离的 region 原型

S3.2a 在 S3.1 的 4,735,104B output table 尾部追加 8 条 cache-line
heap cursor 和 1 条 aggregate vend，因此 `SharedTensorMapSidecar`
在 S4.9 为 4,735,680B；S4.14a 再在尾部追加 8 条物理 shared Vector
Claim cursor，S4.14b 已启用全部 8 条。S4.16a/S4.16b 曾在数组
尾部追加并启用另外 8 条，但因性能门槛失败已整体撤销。S6.3 将 shared
`kMaxTasks` 从 1,280 扩为 4,352，使 generation-6 sidecar 增至
11,027,648B；region ring 和 `shared_outputs` 起点不变，heap/vend/Vector
尾字段随 output table 扩容顺延。R4c 在 offset 11,027,648B 追加
`writer_history[4352]`，每 cell 320B，共 1,392,640B；R4c sidecar 为
12,420,288B，默认 CAP=128 的构建身份当时为 ABI generation 7。

R4e-a 又在该 offset 追加 `reader_done[96]`，每个 worker 独占 64B，
合计 6,144B。当前 generation-8 sidecar 为 12,426,432B；CPU 非 split
`SchedulerState` 为 1,019,542,400B，定义
`PA_COMPETE_FIRST_SPLIT_FINISH` 的 CCEC split 布局为 1,019,548,544B。
private sidecar 仍为 2,113,664B，private non-split/split
`SchedulerState` 仍分别为 1,007,115,968B 和 1,007,122,112B。两次尾部
追加都不移动此前字段，Vector 热路径仍使用 `task_id%8`；sidecar 仍位于
standalone 控制区和 `results` 之后，不移动 `WorkerState`、`RunConfig`
或既有结果字段。

`reader_done[worker]=D` 表示该 worker 已经结束 task `[0,D]` 的全部
ordinary-ring 读取，初值为 -1，只允许 CAS `D-1 -> D`。最慢完成值为
`Dmin` 时，inclusive 回收候选为 `max(-1,Dmin-H)`。这与设计文档写的
`R=min_progress-H-1` 并不矛盾：这里的下一任务进度
`min_progress=Dmin+1`。R4e-a 建立独立状态、严格 CAS 和纯公式 CPU
门槛；R4e-b 又在隔离 ring driver 中把 exact writer turn、reader 候选和
单调 `reclaim_upto` 发布组合起来，并以满桶慢 reader 交错证明：task 2
读取尚未结束时 future writer 只能得到 `CapacityBlocked`，读取返回并发布
完成前沿后才可回收 producer 0、复用对应物理槽。该组合 helper 仍没有
PA/Submit 调用者；其 `active_workers` 连续前缀和 `H` 也必须是整个 ring
生命周期的固定权威配置，不能在已经发布回收边界后动态改变。普通 PA
继续走专用 writer-chain，不发布
`reader_done`，host 反向要求 96 条线保持 -1。A5 上的
“ordinary 读取完成 -> reader_done 发布”编译器/设备顺序及跨核可见性
仍要由后续独立门槛闭合，不能由当前 CPU 结果外推。

S4.15a 历史候选曾在末尾追加 512B Cube cursor，但已因性能门槛未通过而
撤销，不属于当前传输布局。

S4.16a 的历史布局曾保持
`shared_vector_cursor` 起点 4,735,680B、前八条线地址和热路径
`task_id%8` 不变，只在数组尾部追加八条 inactive 物理线。
sidecar 为 4,736,704B，CPU non-split/CCEC split `SchedulerState`
分别为 1,011,852,672B/1,011,858,816B；后八条线必须零 attempted
且终值为 -1。S4.16b 在完全相同布局中启用全部十六条线，b256
每线 2,048 次 attempted。上述 ABI 和语义均实际验收通过，但
S4.16b 第一层性能门槛失败；这些数字只属于历史候选，不是当前传输
布局。

每 task 最多八个 fresh output，以 16B
`FdwicOutputRef` 表达 `(producer_task_id, output_slot)`，返回句柄为
8B `SharedTaskOutputs`。当前构建身份 ABI generation 为 8。通用
history 能力由 generation 7 引入，generation 8 只在其后追加
`reader_done` 并重建同一 history 门槛；两项仍都是独立门槛能力，不能
据此声称 PA 热路径已经迁到通用 WriterIntentSet 或 reader-progress
reclaim。

当前 shared 已完成 winner-only 重构参与 Materialize：所有 worker 仍先
Claim 并独立声明同一组 output symbol；Alloc 暂时保留每核三个静态 Output
参数，以对齐参考 `alloc_tensors(args)` 的调用形状。QK/SF/PV/UP 只有 winner
执行 reset、view/CreateInfo 构造以及 tensor/scalar 参数添加。winner 随后
独立预留 shared heap、生成 descriptor，只在自己实际依赖的
`(producer_task_id, output_slot).published` 上等待并只读解析 fanin；本地
执行状态建立后再提交 INOUT writer，最后发布 descriptor。PA Case1 不再
读取或推进全局 `committed_tasks/reclaim_upto`。loser 仍以非空地址把上一
task 的陈旧
`TaskArgs` 传过 split ABI，但 finish 只闭合固定泳道/Submit 边界，不读参数
内容，也不触碰 heap/map/payload。

CPU guard-page 定向测试会把 loser 的 `TaskArgs` 页改成 `PROT_NONE`，依次
通过 Alloc/QK/SF/PV/UP 五次 split finish；任一字段读取都会立即失败。逐核
host oracle 还按实际 wins 精确核对四类重构参次数，防止只看全局总数而漏掉
某个 loser 回退。S4.6 另有三组定向门禁：预置 terminal `fatal` 后 winner
不得产生 heap/slot/completion/symbol 副作用；空 region 验证分别覆盖
Local/GM 的 manual 与 ordinary writer；shared `PrepareMap` raw marker 的
负向自测会拒绝非零时长、task 序列或身份漂移、未锚定 matching
`Materialize.end`、非法 flags/aux、缺失和重复记录。后者复用 host 已有 raw
扫描，并与既有逐核 Submit 数、精确记录数门禁共同闭合，不增加任何 device
字段。

shared heap 使用 `task_id % 8` 的有界绝对 cursor，物理 shard span 默认为
32MiB；b256 每 shard 精确使用 25,821,184B，不 wrap、不复用 generation。
非空 task 分别推进 shard cursor 和 aggregate vend，零输出 task 只读取 vend。
shared 不调用 private per-worker ring 的 HeapGuard。host 直接核对 8-shard
实际 descriptor 地址，再以 canonical 连续地址比较 private/shared 的
normalized writer signature。

Materialize 只在 winner 上预留 heap 并生成本地 descriptor，不再提前发布。
本地 `CompleteTask`/`BuildWinner` 成功后，winner 用 FetchMax 提交本 task
消费的唯一 INOUT writer，再把本 task 的全部 output descriptor 复制到共享
cell，执行 `FlushRegion`、存储屏障并发布 `published`。因此当前
`published=N` 表示 producer Submit 已封口，而不只是 descriptor 已构造。
没有依赖关系的 future task 可以先行封口；有依赖的 task 只受自己的 per-slot
发布位和执行期 task completion flag 约束。

重复发布或后槽异常不会覆盖既有 descriptor，也不会留下前槽的部分发布；
若发布位 Exchange 观察到异常旧值，冷路径会撤回本 producer cell 的全部
控制字并清空、flush descriptor。QK/SF/PV/UP 的构建后封口失败还会撤销本地
built slot，避免 FinalDrain 执行失败任务；Alloc ready flag 已不可逆，只能
广播 fatal 使整轮结果无效。loser 不读取 shared sidecar，也不等待 commit。

shared symbol resolver 只接受 flags/view 字段全零的 plain ref；其他形态显式
失败，不能被悄悄降级为普通 descriptor。resolver 第一遍等待并校验全部引用，
读取每个 symbol 的 `published` 和 `last_writer`，但不修改 writer、payload、
统计或输出 fanin。PA Case1 当前每个 fresh symbol 只有一个后继 writer，
所以 Input/Inout/OutputExisting 看到的 writer 都必须精确等于句柄声明的
producer；同一 task 的重复写引用也会被拒绝。全部通过后才失效并复制
descriptor 到本 task payload scratch，一次性提交 input-load 计数和按既有
`AddFanin` 去重的 fanin。

INOUT/OutputExisting 的 writer 更新单独发生在本地执行状态成功建立后。
FetchMax 的返回旧值必须精确等于 producer；成功项计入
`inout_writer_commits`。异常 FetchMax 的终态不做负向 RMW 回滚，而是保留
现场并广播 fatal；该 RMW 已经线性化，多 symbol 提交不是事务，不能伪造
负向 RMW 抹掉故障现场。当前固定拓扑不支持
`producer -> INOUT -> 后继 INPUT` 的多级 writer 链，定向测试要求该形态显式
失败，不能把它误写成通用能力。

符号和 `manual_dep=true` 的 output view 都不进入 region lookup/register。
因此 PA Case1 完全绕过 shared region raw ring；host 要求
`committed_tasks=0`、`reclaim_upto=-1`，全部 bucket/slot 保持初始化状态。
Register 仍逐项验证 ordinary region entry 必须为零，发现非
`manual_dep` 的普通 writer 会 fail-closed，而不是悄悄回退到未接线的 ring。
host 将实际读取到的 fresh symbol 最终 writer
投影为与 private region 相同的规范化 writer 序列，并按 Case1 约定补入
manual output view；所以 raw 存储不同不妨碍两种模式使用同一规范化签名
核对已观测 fresh-symbol writer。manual view 是约定投影，不是从 shared
执行态独立取证，不能用该签名单独证明 manual view 对等。b1 的
dependency/normalized writer 签名为
`5cb454393ed48dcb`/`3a3d526c9b23c3db`，b256 为
`b7d985d6edb07078`/`556bec7ec8d0f323`。

以下是 S3.1 当时的历史门禁，不是当前保留路径的提交顺序或统计口径：

| 门禁 | 结果 |
| --- | --- |
| CPU private/shared b1、b256 | 全部调度与终态断言 PASS |
| shared symbol、shared ring ASan/UBSan/leak | PASS |
| Python | 100 项 PASS |
| CCEC shared submit-pmu none b1 | 调度、symbol、PMU owner 恢复均 PASS |
| CCEC private b1/b256 | 73.318 us / 3.808011 ms |
| CCEC shared b1/b256（fail-closed 修正后） | 86.552 us / 27.094219 ms |
| S3.1 历史 symbol publish/INPUT-load/exchange，b1 | 8 / 5 / 3 |
| S3.1 历史 symbol publish/INPUT-load/exchange，b256 | 2,048 / 1,280 / 768 |
| S3.1 历史 shared b256 ordered 终态 | `committed_tasks=1280`，`reclaim_upto=1214` |

S2.5 shared b256 的同类单样本为 26.556193 ms。S3.1 在正确性审计前曾取得
23.562916 ms，但当时重复发布和后置非法引用的失败路径可能留下部分共享
状态；补齐全量发布预检与两遍 resolver 后，最终同类单样本为 27.094219 ms。
两者只差约 +2.03%，且都不是多轮稳定基线，不能据此宣称稳定回退或收益；
被撤销的 23.562916 ms 也不能作为有效 S3.1 基线。S3.2a 已把 Materialize
与 heap 收敛到 winner，S3.2b 又只收敛 QK/SF/PV/UP 重构参；两步分别提交，
没有把分配主体和构参主体混成一个不可归因的性能变量。

S4.6 提交前审查补回 winner 的 terminal-fatal 读取后，最终 CCEC
shared b1 real-compute perf-clock 单样本为 `70.279 us`。同阶段较早 ELF
的 shared b256 scalar-nop0 / real-compute 为 `3,228.844 us` /
`5,982.840 us`；private b1 real-compute / b256 scalar-nop0 为
`70.707 us` / `3,300.478 us`。所有模式均通过完整语义断言，shared b256
的 96 核全部活跃且 region sequencer 保持初值。b256 数字不是补回 fatal
读取后的最终 ELF，只作 convoy 消失和同量级证据；不同阶段、负载和 ELF
不能直接相减，稳定收益仍需冻结最终 ELF 后配对多轮。

实现中验证了一项 CCEC 约束：`[[block_local]]` runtime state 不能包含具有
非平凡构造函数的类型。不能用 `block-local-init` 绕过该限制；当前
`FdwicOutputRef`/`SharedTaskOutputs` 保持 trivial POD，非法引用由显式
`InvalidSharedOutputRef()` 工厂生成，并用 `static_assert` 固化这一 ABI
前提。

S2 的 2,119,808B sidecar（含 96 条 per-core progress）和 S2.5 的
2,113,664B sidecar（删除 progress）均只属于明确标注的历史阶段。S2.5
确立的 ordered-winner ring helper 与定向自测仍保留为未来非空
ordinary-region 原型，但已经从 PA Case1 运行路径断开；它的测试结果不能
冒充当前 PA 热路径证据。S2/S2.5 的历史结构、失败实验和上板结果见
`shared_tensormap_record.md`。

#### shared protocol 的独立 A5 门槛

R4c 的通用 WriterIntentSet 为 future writer 覆盖 latest-cache 的场景追加
task-indexed immutable history；R4d 用独立 mixed AIC/AIV ELF 验证跨物理核
失效与回溯。R4e-c 将该门槛泛化为 shared protocol 多场景载体，当前第一步
只迁移原 history 场景，不改变其设备算法。它不启动普通 PA benchmark，也
不改变 PA kernel/host。构建和运行前先 source 本用户 CANN 9.1，再在本目录
执行：

```bash
./run.sh build-shared-protocol-litmus ccec
./run.sh shared-protocol-litmus ccec \
  --scenario history --device 0 --runs 20
```

该 action 已固定为 shared-only，不能附加 `--tensormap shared`。scenario
必须显式选择，避免后续增加协议门槛后默认运行错误路径；`--runs 20` 表示
history 的两个方向各启动 20 个全新 host 进程，共 40 个：

```text
AIC writers -> AIV reader
AIV writers -> AIC reader
```

每个方向都先让 reader 普通读取并预热尚为零的 future-writer history 两条
cache line，再由两个未来 writer 发布，最后调用真实
`CollectSharedFanin()` 沿 `E -> D -> B` 回溯。host 会逐项校验预热值、
21 条 history record 及 21 次成功 CAS、7 个 latest、最终 fanin、控制门
和未触碰的 ordinary ring。host 内的 history 初始化与验证保持为独立函数，
不能因后续场景共用 ACL/build 骨架而弱化断言。产物固定在：

```text
build/ccec/shared/shared-protocol-litmus/
```

普通 shared 构建还会把 AIC/AIV 的 generic shared-protocol probe 各自实际
静态链接。该 probe 同时显式实例化 WriterIntentSet、`reader_done` CAS 和
reader-based reclaim refresh，拒绝 `__multi3` 或其他未解析 device
builtin；检查后删除，不会进入正式 mixed ELF。静态链接只证明两种 CCEC
后端能生成完整设备代码，不证明 ordinary region 的跨核
reader-progress/reclaim 可见性已经闭合。
shared-protocol-litmus 自身虽是 CCEC mixed ELF，但没有定义 split-finish，
因此其
GM `SchedulerState` 使用当前 generation-8 non-split 大小
1,019,542,400B；它会校验新追加的 96 条 `reader_done` 始终保持 -1。

shared sidecar 的 atomic 当前会计入既有 Submit/业务阶段时间，但 heap
cursor/vend、symbol writer/published 等尚未逐条接入 atomic 泳道 wrapper；
所以这一版泳道不能声称完整列出了 shared 协议 atomic，也不能从现有事件拆出
shared heap 或 symbol 单指令成本。这不影响 host 对 cursor/vend、descriptor、
输出符号、依赖边和规范化 writer 签名的独立校验。为复用现有 analyzer，
shared 泳道仍为每个 Submit 写一条零时长 `PrepareMap` 结构 marker；它不
读钟、不访问 region sidecar，且在 perf-clock/submit-PMU 构建中编译消除，
不能把该 marker 解读为 shared 模式仍有 PrepareMap 业务开销。

## 5. 使用说明：运行、测量与泳道查看

以下命令均在 `pa_scheduler` 目录执行。首次使用应先按第 4 节 source CANN
环境并完成构建。主要 action 的用途如下：

| action | 用途 | 是否生成泳道文件 |
| --- | --- | --- |
| `build` | 构建指定后端 | 否 |
| `smoke` | 固定 b1/r1/`scalar-nop=0` 的快速语义回归 | 否，只做内存记录校验 |
| `run` | 自行控制 batch、run、winner 负载和诊断参数 | 仅显式传入 `--swimlane-json` 时生成 raw |
| `swimlane` | 单轮运行并自动生成 raw、Perfetto merged JSON 和排他闭合分析 JSON | 是 |
| `build-perf-clock` | 单独构建 CCEC 或 CPU 的低扰动首末 Submit 计时产物 | 否 |
| `perf-clock` | 强制单进程单轮、关闭全部其他观察器，输出完整 Submit 全局跨度 | 否 |
| `build-submit-pmu` | 构建指定 `none|claim|efdrain|materialize|register` 的 CCEC PMU-only ELF | 否 |
| `submit-pmu` | 单轮采集完整 Submit PMU，可选导出 JSON | 否，与泳道隔离 |

`ccec|ascendc|cpu|all` 用于选择后端；`all` 始终按 CCEC、AscendC、CPU
的顺序执行。所有 action 都接受一次
`--tensormap private|shared`，位置可在 backend 后的其余参数中任意放置；
两种模式均可运行，省略时仍默认 `private`。
CCEC、AscendC 和 CPU 的 `run/swimlane` 都可使用
`--winner-workload scalar-nop|real-compute`、
`--real-compute-count N`、`--real-compute-counts QK,SF,PV,UP` 或
`--real-compute-pattern constant|layout-diagnostic`；真计算 count/pattern
与 `--nop-count*` 互斥。`smoke` 有意固定 scalar-nop，不接受这些覆盖项。

低扰动 b1 门禁使用：

```bash
./run.sh build-perf-clock cpu  --tensormap private
./run.sh perf-clock       cpu  --tensormap private --batches 1
./run.sh build-perf-clock ccec --tensormap private
./run.sh perf-clock       ccec --tensormap private --batches 1
```

shared 时只替换 `--tensormap shared`。`perf-clock` action 自己固定
`--runs 1 --no-swimlane`，并拒绝泳道、atomic、phase-profile 与 PMU
参数；需要多样本时应由外层启动多个独立进程并平衡 private/shared 顺序。

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
  --analyze-swimlane

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
  --device 0 --batches 1 --winner-workload real-compute

./run.sh swimlane ascendc \
  --device 0 --batches 1 --winner-workload real-compute \
  --real-compute-count 1
```

`swimlane` 是唯一的正式泳道构建口径，固定合并 schema-v4
普通阶段、业务父区间、真实 Submit 尾动作与 atomic 记录（direct
Atomic 加 PollBatch）；
无需再显式传 `--trace-atomics`。该 action 会管理 `--runs 1` 和输出路径，因此不要再传
`--runs`、`--swimlane-json` 或 `--no-swimlane`。转换器默认使用 `python3`；如需
固定到用户自己的 Python，可在命令前设置：

```bash
export PYTHON=/path/to/venv/bin/python
```

转换器和排他分析器都只使用 Python 标准库，不需要 PyTorch，
也不需要安装 simpler Python 包。`--analyze-swimlane` 仍只控制 runner
终端中的传统分组文字统计；无论是否传它，`swimlane` action 都会生成
排他闭合报告。

修改 C++ 头文件或 kernel 后必须先执行
`./run.sh build ccec --tensormap private`，
`swimlane` action 只消费已有构建件，不会隐式重编译。日常边界迭代默认只跑
A5 b1；b256 只用于阶段性规模/容量收口或明确指定的长负载结论。

该 action 固定执行一轮，产物全部位于本目录的
`outputs/pa_scheduler_<mode>_swimlane_<UTC时间>_<PID>/<backend>/`。选择 `all` 时，
同一 output root 下会按顺序建立 `ccec/`、`ascendc/` 和 `cpu/` 三个子目录：

- `l2_swimlane_records.json`：与真实 PA 相同的十列 `fdwic_events`
  权威原始件，所有字段复算以它为准；
- `merged_swimlane.json`：只用于 Perfetto 可视化。schema-v4 的 duration
  事件只保留 `ph/name/pid/tid/ts/dur` 六个必需字段，不再逐事件
  复制 raw 中的 `args/cat`；拖入 <https://ui.perfetto.dev/> 即可查看；
- `swimlane_exclusive_analysis.json`：以原始整数 cycle 校验并汇总
  Submit、EfDrain、OrchestrationReplay、FinalDrain 和 WorkerCompletion
  排他闭合关系，并将 Submit 内和 Submit 间的 residual 按相邻边界小表
  聚合。完整父子层级、排他角色和数值统计以该报告为准，不再逐事件
  塞入 merged。

schema-v4 禁止产生历史 `Alloc/Build/Replay` lap 与未使用的
`DrainWon`，只用 `AllocComplete/WinnerBuild` 表达真实 Submit winner
尾动作。loser 没有可单独计时的业务动作，不生成 `LoserReplay`
伪阶段；其未归因尾段只由离线 residual 展示。每个 Kernel 还必须唯一归入 EfDrain、WinnerBuild、AllocComplete 或
FinalDrain；孤儿、越界或多重归属都会使排他分析失败。
除 Kernel 可以执行前序 task 外，所有 Submit 前端和尾动作的 `task_id` 必须与
包含它的 Submit 一致。

runner 结束时会打印准确目录：

```text
[SWIMLANE] output_root=.../outputs/pa_scheduler_private_swimlane_<UTC时间>_<PID>
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
4. `AIC`、`AIV0`、`AIV1` 轨展示 OrchestrationReplay、Submit、Claim、
   EfDrain、WinnerBuild、AllocComplete、FinalDrain、Residual 和 RingBp
   等 runtime 阶段，带 `·kernel` 的轨展示 QK、SF、PV、UP；
5. direct Atomic、PollBatch 及 ClockBaseline 固定画在对应
   `AIC/AIV` scalar lane；direct 名称显式区分 `return_ready` 和
   `source_issue`，PollBatch 单独标识逻辑轮询 episode；不生成带 `·atomic` 的
   伪并行子轨；
6. merged 事件名保留 phase/task，atomic 名还保留
   `site/op/boundary/call_count`；需要 `func_id/core/flags/aux` 等精确字段时
   查同目录 raw，Atomic 的解读边界见 5.6 节。

WaitForSlot 和 HeapGuard 没有可伪造的逐事件起止时间，因此不单独生成 Perfetto
事件；实际发生等待时，泳道中会出现 RingBp 事件。现行 CCEC 局部 PMU
只支持 Claim、EfDrain、Materialize 和 Register，使用第 5.7 节的独立
`submit-pmu` phase ELF。

`outputs/` 已被 Git 忽略，生成的几十至数百 MiB 泳道文件不会被普通
`git add` 意外纳入提交。

### 5.4 手工导出或重新转换

转换完全由本目录脚本完成，不依赖 simpler 的 Python 包或虚拟环境。已有原始
文件也可单独转换：

```bash
python3 ./swimlane_converter.py \
  ./outputs/<capture>/ccec/l2_swimlane_records.json \
  -o ./outputs/<capture>/ccec/merged_swimlane.json

python3 ./swimlane_exclusive_analyzer.py \
  ./outputs/<capture>/ccec/l2_swimlane_records.json \
  -o ./outputs/<capture>/ccec/swimlane_exclusive_analysis.json
```

若只需要原始记录，可通过通用 `run` action 显式指定文件；导出为避免多轮覆盖
而要求 `--runs 1`：

```bash
mkdir -p ./outputs/manual
./run.sh run ccec --device 0 --batches 256 --runs 1 \
  --swimlane-json ./outputs/manual/l2_swimlane_records.json
```

手工 `--swimlane-json` 只生成 raw，不会自动生成 merged 或排他报告；
需要随后调用上面两个脚本。该参数强制要求 `--runs 1`，避免多轮
静默覆盖同一文件。runner、converter 和 analyzer 都只在单件写完后
原子替换各自目标；这是逐文件发布而不是三件整体事务：runner 失败时
不启动后处理，converter 失败时保留已完成的 raw，analyzer 失败时保留
已完成的 raw 与 merged，任何阶段都不会把半截 JSON 冒充完整产物。

### 5.5 CPU 回归、参数与测量口径

CPU 完整协议回归建议关闭大泳道缓冲区：

```bash
./run.sh run cpu \
  --batches 256 --runs 1 --nop-count 0 \
  --profile-phases --no-swimlane
```

主要选项：

- `--profile-phases`：CPU/AscendC 兼容诊断中分别统计 Claim、EfDrain、
  WaitForSlot、HeapGuard；最终 CCEC `swimlane` 构建不接受该选项，
  CCEC 的独立 `submit-pmu` 局部归因只支持 Claim、EfDrain、Materialize、Register；
- `--analyze-swimlane`：读取完整记录，输出各阶段的 per-worker 累计分布以及
  EfDrain/Materialize/Claim/Register 的 per-role、per-task-kind 单事件分布；
- `--trace-atomics`：在已开启的泳道中记录 atomic 逻辑调用；direct 调用逐条记录，
  显式等待区内六类 observation load 用带精确 `call_count` 的 PollBatch 聚合。
  建议同时传 `--analyze-swimlane` 输出按 AIC/AIV、调用点分组的分布。不能与
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
和 stream/thread 同步。不同版本比较时，必须使用相同构建、相同 phase
和相同泳道/PMU 开关，因为计时、记录和 PMU 边界本身都会影响竞争时序。

CPU/AscendC 兼容诊断中，`[PHASE]` 的每个阶段都是每 worker 在
1,280 次 Submit 中的累计时间：

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
- private 模式下，`frontier_initial` 是每个 completion 对 frontier 的首次
  load；shared Case1 使用严格 no-wrap heap，完成只发布 vend/flag，
  `frontier_initial`、`frontier_flag`、`frontier_ready_fetch_max` 和
  `frontier_terminal` 均应为 0；
- `frontier_ready_fetch_max` 同时计数 ready flag 和紧随其后的 FetchMax，两者在
  这条控制流中一一对应；CCEC/AscendC 上它是一条真实 A5 atomicMax，CPU 上只是
  一次逻辑 FetchMax 调用；
- private 模式下，`frontier_terminal` 是每次扫描最终遇到的 not-ready flag，
  当前工作量下应与 completion 数相等；
  `frontier_flag = ready + terminal`；
- `submit_completion_ops` 覆盖 Claim、第一圈 HeapGuard、fanin、completion 发布和
  frontier，不包含 started/replay_done 生命周期屏障。

shared 模式仍保留上述 frontier 字段、AtomicSite 编号和 ABI，以便与 private
共用 converter/schema。只有在 `--trace-atomics` 已开启、
`dropped_records=0` 且 logical/physical/batch 闭合通过时，泳道没有对应
记录才能证明热路径没有执行调用，而不是采集丢失。如果后续 shared heap
允许 wrap 或复用 task cell，必须恢复 frontier 或等价
generation/reclaim 协议，不能沿用 no-wrap 结论。

这些字段在每个 worker 的私有 `LocalStats` 中递增，kernel 结束时才发布到独占
结果区，不为诊断新增共享 atomic。它们仍会增加少量 scalar 指令，因此优化 A/B
必须使用相同的计数布局；不能把启用分类后的绝对时间直接与旧二进制比较。

### 5.6 合并泳道中的 atomic schema-v4 语义边界

正式 `swimlane` action 固定记录 standalone 调度器中的 atomic **逻辑调用**，
不只记录 winner 或慢样本。普通 direct Atomic 仍是一条源码调用对应一条物理记录；
显式等待区内允许聚合的 observation load 则用一条带精确调用次数的 PollBatch 表示。
生成带文字分析的 CCEC 合并泳道可直接执行：

```bash
./run.sh swimlane ccec \
  --device 0 --batches 1 \
  --analyze-swimlane
```

只有使用低层 `run` action 手工导出 raw 时，才需要显式传入
`--trace-atomics`；该兼容入口不代表存在第二种 atomic-swimlane 构建。

schema-v4 raw 的 `metadata.trace_schema_version` 必须为 4，且顶层
`l2_swimlane_level` 必须为 4。转换后 direct Atomic、PollBatch 和
ClockBaseline 都放在对应 AIC/AIV 的原 scalar lane；它们属于 scalar
调度观察，不再伪装成与 scalar 并行的独立子轨。Kernel 仍放在独立计算单元轨。
direct Atomic 事件名显式区分两种边界：

```text
atomic.return_ready.<site>.<op>#<task_id>
atomic.source_issue.<site>.<op>#<task_id>
```

边界已编码在事件名中，可在 Perfetto 中按名称搜索或过滤，
无需为每条事件保留 category/args。

PollBatch 转换为：

```text
atomic.poll_batch.<site>.load×<call_count>
```

名称中的 `call_count` 是实际执行的源码 wrapper 调用次数，
不是采样或估算值。

当前固定 schema 共有 19 个调用点。0～14 是既有 common/private
调用点，15～18 是 shared heap 第一批已接入的调用点：

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
| 15 | `shared_heap_vend_load` | `load` | shared aggregate vend 预检 |
| 16 | `shared_heap_cursor_load` | `load` | shared 分片 cursor 预检 |
| 17 | `shared_heap_cursor_reserve` | `fetch_add` | 取得本 task 分片物理区间 |
| 18 | `shared_heap_vend_advance` | `fetch_add` | 推进并取得 aggregate vend |

上表是源码调用点集合，不代表每轮都会出现全部 19 类事件；例如正常成功路径不应
执行 `fatal_set`。standalone 也没有真实 PA 后续追加的 BlockWon site，不能把真实
PA 的九类 load 加一类 exchange allowlist 照搬到这里。

shared heap 四个站点的返回值都参与协议判断：vend/cursor Load 用于合法性
与容量检查，两个 FetchAdd 的旧值分别决定物理地址和累计进度。因此 CCEC
direct 记录均使用 return-ready 边界；它们不是发布后即丢弃返回值的
source-issue 操作。PA Case1 每 batch 固定执行 5 次 vend load，以及各 4 次
cursor load、cursor reserve 和 vend advance。output publication/last-writer
仍按后续 S5.2 小步接入，不能把当前 19-site schema 宣称为 shared 全覆盖。

standalone 只允许以下六类 observation load 在**匹配的显式等待区内**进入
PollBatch；同一 site 在等待区外的一次性或 opportunistic 读取仍是 direct Atomic：

| `site_id` | `site` | `op` |
| ---: | --- | --- |
| 1 | `startup_poll` | `load` |
| 2 | `fatal_poll` | `load` |
| 5 | `fanin_flag_load` | `load` |
| 11 | `heap_frontier_load` | `load` |
| 12 | `heap_vend_load` | `load` |
| 14 | `replay_done_poll` | `load` |

一个等待区可以同时累积多个 site，所以不同 site 的 PollBatch 时间窗可以重叠；
等待区内也可以交错 direct Atomic。direct record 写入不能隐式关闭 PollBatch，
否则这些自然交错会把同一个等待 episode 人为切碎。

原始 `fdwic_events` 仍是十列格式。对 `phase="Atomic"` 的记录，`auxiliary` 是
`site_id`，`flags` 使用以下 ABI：

- direct Atomic：bit 7 为 0；低 4 bit 是 `op_id`
  （Load/Exchange/FetchAdd/FetchMax 依次为 0/1/2/3），bit 4 表示返回旧值参与
  后续逻辑，bit 5 仅对 Load 表示本次读到零，bit 6 表示是否取得
  “返回值本核可消费”边界；bits 8..31 只对 direct FetchMax 表示软件 retry，
  不能解释为调用次数；
- PollBatch：低 4 bit 必须为 `load(0)`，bit 4 必须为 1，bit 5/6 必须为 0，
  bit 7 必须为 1；bits 8..31 保存无符号 24 bit `call_count`，有效范围为
  `1..0xFFFFFF`。达到上限时先落盘，再从 1 开启下一条 batch，不允许饱和后
  丢失调用数；`task_id=-1`、`function_id=-1`。

schema-v4 merged 只保留可视化必需的六个 duration 字段。direct 的
`site/op/boundary/task_id` 和 PollBatch 的 `site/op/call_count` 均在名称中；
`site_id/op_id`、原始整数 cycle、flags、retry 和 value-zero 等精确值从同目录
raw 十列记录复算。不把这些重复复制到 merged，是为了控制数百万
事件时的文件大小和观察工具内存。

边界必须按源码调用点语义解读：

- `source_issue_bracket`：返回旧值本来就丢弃的发布型调用。当前五处是
  `startup_increment/replay_done_increment/completion_vend_exchange/`
  `completion_flag_exchange/fatal_set`。结束时钟与旧值无依赖，只能表示
  源码发射包围区间。
- `return_value_ready`：协议本来就会判断返回值的
  `Load/FetchAdd/FetchMax/Exchange`。CCEC
  在 atomic 后生成紧邻的 `dependent MOV -> MOV SYS_CNT`；这证明旧值已可被
  本核 scalar 消费，不证明其他核已看到发布的新值。

两种终点都早于本条 64-byte trace record 写入。当前不为每条 atomic
加 DSB/ISB/额外 GM load；这些操作要么后端不支持，要么会明显改写
被测路径。两种 bracket 都不能直接称为跨核可见或 atomic retire 延迟。

PollBatch 的 `duration`/`poll_window_cycles` 是从该 site 在显式等待区内首次累计
调用到边界关闭的**逻辑等待 episode 包络**。它不是独占 scalar 时间，不是
`call_count` 次 atomic 延迟之和，也不是其中任意一次 load 的单次延迟；因此不能把
它放进 direct atomic 的 median/p95，或用 `duration/call_count` 推导单次成本。

边界关闭规则与 phase/lap 共用同一次 cycle 采样：

- 显式等待区退出时关闭匹配的 PollBatch；
- `TraceTimestamp` 在写 phase begin/end 前关闭全部活跃 batch；
- schema-v4 producer 不再生成旧 `Alloc/Build/Replay` lap；历史 helper
  仍有自身关闭规则，但不得出现在当前 raw 中；
- Kernel begin/end 也通过 `TraceTimestamp`，所以 PollBatch 不能跨入或跨出 Kernel；
- 最终 flush 只作防御性兜底，不能替代上述语义边界。

开启该诊断时，每个 worker 在最终 drain 之后还会写两条
`ClockBaseline`：`clock.consecutive_sys_cnt_reads` 和
`clock.atomic_return_dependency_hook`。前者量连续时钟读，后者量纯寄存器依赖 hook
的固定底噪。全局因此恰有 `96*2=192` 条，都只是分辨率参考，不是
可以从每条 Atomic 机械相减的校正常数。`[TRACE_ATOMIC]` 只按 AIC/AIV、site 和 op
输出 direct 事件数、源码 bracket 原始累计、中位数、p95 和最大值；
`[TRACE_ATOMIC_POLL]` 单独输出 PollBatch 的 episode 数、精确逻辑调用数和等待包络
分布，二者不会混算。

schema-v4 level-4 raw 必须按逻辑调用与物理记录两套口径闭合。设 direct 物理记录数为
`direct_atomic_records`，则逐核和全局都必须满足：

```text
logical_atomic_calls = direct_atomic_records + Σ(PollBatch.call_count)
physical_atomic_records = direct_atomic_records + poll_batch_records
physical_atomic_records
    = logical_atomic_calls - batched_poll_calls + poll_batch_records
```

raw metadata 的 `fdwic_summary` 七项
`records/atomic_records/clock_baseline_records/atomic_calls/`
`batched_poll_calls/poll_batch_records/dropped_records` 必须与 producer、逐核 state、
raw 行重算和 converter 重算逐项一致；`dropped_records` 必须为 0，每个 worker
必须恰有 2 条 ClockBaseline。raw 到 merged 后物理 Atomic 条数保持不变；正式报告
源码调用总数必须使用 `atomic_calls`，不能使用压缩后的 `atomic_records`。

记录写和 PollBatch 维护本身仍会改变后续指令布局、cache、多核到达顺序、atomic
争用与轮询次数。所以不应将 bracket 或 PollBatch duration 与未插桩 Submit 时间相减，
也不能用它们计算 atomic 对 golden 的绝对占比。每核分区固定容纳 65,536 条记录；
任何容量溢出或闭合失败都使该轮 trace 无效，不能截断后继续分析。

#### schema-v4 排他闭合版 A5 验收

2026-07-19 早期曾完成 CPU/CCEC/AscendC b1 语义门禁，但该过程态
raw 仍含现已删除的 `LoserReplay`，当前 converter 会拒绝它们。下面
只列当前无 loser 记录的 CCEC 证据。删除无业务实体的 loser 标记后，
b256 规模样本位于：

```text
outputs/pa_scheduler_swimlane_20260719_103435_542368/ccec/l2_swimlane_records.json
outputs/pa_scheduler_swimlane_20260719_103435_542368/ccec/merged_swimlane_thin.json
outputs/pa_scheduler_swimlane_20260719_103435_542368/ccec/swimlane_exclusive_analysis.json
```

该轮使用 `real-compute/6,28,4,1`，每核 1,280 个 Submit，raw
845,813 条事件、`dropped=0`，首末 Submit 为 5,326.055 us，六类整数
cycle 闭合全部精确相等。raw 为 56,212,672 bytes；旧格式 merged 为
248,767,986 bytes，同一 raw 经当前六字段 converter 生成的
`merged_swimlane_thin.json` 为 138,349,686 bytes，减少 44.4%。这是离线
可视化瘦身，不改变该轮设备采集。同目录的
`merged_swimlane.json` 是旧胖版，查看该规模样本时应打开上述 thin 文件。
该轮早于最终相邻边界复用，只作规模/容量门禁，不代表当前 residual
构成。

边界收敛后的最新日常验证只跑 CCEC b1：

```text
outputs/pa_scheduler_swimlane_20260719_110756_584549/ccec/
```

该轮 raw 4,118 条事件、`dropped=0`，全局 Submit 89.313 us，merged
428,455 bytes，六类闭合全部通过。Submit 内所有 child-to-child
gap 已为零，即 `submit_internal_residual=0`；最后一个真实 child 到
`SubmitEnd` 的 `submit_tail_residual` 为 184,788/2,241,892 cycle
（8.2425%）。Submit 间 residual 为 155,679/2,397,571 cycle（首末
Submit 包络的 6.493%）。Perfetto 分别用 `submit_tail_gap` 与
`between_submit_residual` 展示两类补集，不新增 raw 记录或字段。

按明确要求完成的当前 CCEC b256 规模复核位于：

```text
outputs/pa_scheduler_swimlane_20260719_114815_617346/ccec/
```

该轮使用 `real-compute/6,28,4,1`，全局 Submit 为 5,360.061 us；raw
839,526 条、`dropped=0`，merged 1,085,191 条事件。raw/merged 分别为
55,791,947/88,775,668 bytes，全部语义断言与整数闭合通过。
Submit aggregate core-work 为 433,383,588 cycle：内部 residual 为 0，
尾部 residual 为 41,008,786 cycle（9.4625%）；逐核首末 Submit 包络为
500,448,909 cycle，Submit 间 residual 为 67,065,321 cycle（13.4010%）。
122,880 条 `submit_tail_gap` 与 122,784 条 `between_submit_residual`
分别精确对应 `96*1280` 和 `96*(1280-1)`，没有增加设备事件。

后续边界迭代默认只跑 A5 b1；b256 仅在阶段性规模/容量收口或
明确要求时重跑。历史 level-4 b256 多轮波动证据仍保留在本文后续
章节，不将 b1 单轮时间宣称为性能改善。

#### 历史：schema-v3 边界修复版 A5 验收

2026-07-18 已用边界修复后的同一版 standalone CCEC 依次完成 b1 与 b256
真机重测。下列是升级到当前 schema-v4 之前的历史样本，不应修改其
metadata 或用当前父区间口径强行解释：

```text
outputs/pa_scheduler_swimlane_20260718_182649_4060527/ccec/
outputs/pa_scheduler_swimlane_20260718_182725_4061524/ccec/
```

每个目录都包含 `l2_swimlane_records.json` 与 `merged_swimlane.json`。raw metadata
与事件行复算结果如下；`records` 是包含普通 phase、Atomic 和 ClockBaseline 的总物理
记录数，不能与 `atomic_records` 混用：

| 样本 | winner 负载 | `records` | 逻辑 `atomic_calls` | direct Atomic | 物理 `atomic_records` | `batched_poll_calls` | PollBatch | 单核记录峰值 | ClockBaseline | dropped | 首末 Submit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| b1 | `scalar-nop=0` | 4,414 | 1,031 | 613 | 850 | 418 | 237 | 57/65,536 | 192 | 0 | 54.056 us |
| b256 | `real-compute/6,28,4,1` | 967,307 | 105,580 | 103,618 | 103,883 | 1,962 | 265 | 10,252/65,536 | 192 | 0 | 5,774.295 us |

两轮逐核 producer state 与 raw 扫描均为 96/96 闭合、异常核为 0；全局公式展开为：

```text
b1:
logical  = 613 + 418 = 1,031
physical = 613 + 237 = 850 = 1,031 - 418 + 237

b256:
logical  = 103,618 + 1,962 = 105,580
physical = 103,618 + 265 = 103,883 = 105,580 - 1,962 + 265
```

PollBatch 只出现在本轮实际进入的四类 allowlist site，下面每项依次为
`物理 episode/逻辑调用`：

- b1：`startup_poll=96/143`、`fatal_poll=42/47`、
  `fanin_flag_load=2/6`、`replay_done_poll=97/222`；
- b256：`startup_poll=96/143`、`fatal_poll=42/47`、
  `fanin_flag_load=16/467`、`replay_done_poll=111/1305`。

`heap_frontier_load/heap_vend_load` 在这两轮均未进入相应 slow path，计数为 0，
不表示 allowlist 漏实现。按同一物理核区间严格检查，b1 的 237 个 PollBatch 与
4 个 Kernel、b256 的 265 个 PollBatch 与 1,024 个 Kernel 都是严格 overlap 0；
分别有 3 和 31 处仅 `end==begin` 的端点相接。这直接验证 PollBatch 没有跨入或跨出
Kernel。b256 的真实 Cube/Vector 计算、调度终态和全部语义断言均 PASS。

converter 的 schema-v3 静态回归同时为 19/19 PASS。b1 是零 winner 负载的快速验收，
b256 是开启 atomic 泳道的诊断运行；上表 Submit 只能证明当前观察构建的运行量级，
不能替代关闭 trace 的性能基线或与历史样本做单轮减法。

#### 历史 schema-v2 样本（仅保留旧口径）

2026-07-18 的旧 CCEC b256 文件
`outputs/scalar_observation_final_20260718/atomic_inlineasm_ccec_b256/raw.json`
记录了 963,368 条物理记录，其中逐条 Atomic 99,944 条、ClockBaseline 192 条，
逐核峰值 10,308/65,536，且当轮 `dropped=0`。这些数字来自引入 PollBatch 与上述
phase/lap/Kernel 边界修复之前的 schema-v2 逐调用模型，只能用于追溯旧版观察结果；
不能拿 99,944 当作当前 schema-v4 的物理容量、逻辑调用数或闭合证据。

### 5.7 两类正式构建与 CCEC Submit PMU

`swimlane` 和 `submit-pmu` 是两个独立重编译的观察产物：

- `swimlane` 编译普通阶段和逐 atomic record，把 atomic 画在对应
  AIC/AIV scalar lane；不生成 PMU owner，也不输出 PMU JSON。
- `submit-pmu` 编译掉泳道 record、逐 atomic wrapper、ClockBaseline、
  runtime phase-profile 和旧 cold/warm 冲刷体，只保留完整 Submit PMU
  与一个编译期 phase。

两者不能在同一进程同时采集。这不只是 CLI 限制：泳道/逐 atomic
代码会改变 scalar 指令布局和 I-cache 本身，将其保留在 PMU ELF 里即使
运行时关闭 record，也会污染要观察的取指环境。

`none`/`claim`/`efdrain` 保持与泳道版一致的跨 TU split-finish
形状；这两个局部阶段的边界都在 finish 之前。`materialize`/
`register` 边界位于 finish 内，当前为了使用同一份真实
PMU context 而采用 inline-finish 诊断 ELF。因此后两者与 `none`
的代码布局并不相同，只能解释各自 ELF 内的观测结果，不能与
`none` 机械相减或当作无扰动的阶段净值。

当前 `submit-pmu` 只支持：

| phase | 编译期 ID | 局部边界 | 用途 |
| --- | ---: | --- | --- |
| `none` | 0 | 不做任何中途 shadow counter 读取 | 完整 Submit 主基准，优先用于回答 AIC/AIV 每核 request/miss |
| `claim` | 1 | 每次 `Claim()` 调用前后读取 shadow counter | 验证局部归因链路，输出带观察扰动的 running read-clear 下界和保守上界 |
| `efdrain` | 2 | Submit 开头唯一的 EfDrain call-site 前后 | 归因 opportunistic drain，不包含 RingBackpressure/FinalDrain |
| `materialize` | 4 | 每次 `MaterializeTask()` 调用前后 | 归因 descriptor materialize；成功和失败出口都由同一闭合边界覆盖 |
| `register` | 5 | 每次 `RegisterOutputs()` 调用前后 | 归因输出注册；Alloc 与非 Alloc 两个互斥调用点合起来仍是每次 Submit 一次 |

分别构建：

```bash
./run.sh build-submit-pmu ccec none
./run.sh build-submit-pmu ccec claim
./run.sh build-submit-pmu ccec efdrain
./run.sh build-submit-pmu ccec materialize
./run.sh build-submit-pmu ccec register
```

上面省略了默认的 `--tensormap private`；显式写法例如：

```bash
./run.sh build-submit-pmu ccec none --tensormap private
```

产物完全分开：

```text
build/ccec/private/submit-pmu/none/
build/ccec/private/submit-pmu/claim/
build/ccec/private/submit-pmu/efdrain/
build/ccec/private/submit-pmu/materialize/
build/ccec/private/submit-pmu/register/
```

每个目录都自包含同 phase 的 host、mixed kernel、PMU owner 和 dispatcher，
不得跨 mode 或 phase 拼装。构建完成后才会原子发布包含 mode、variant、phase
身份和四个 SHA256 的统一 manifest；`submit-pmu` action 在启动 host 前逐项
复核，缺件、串 mode/phase 或内容变化都会直接拒绝。一次正式采集示例：

```bash
export PYTHON=/home/q00473782/.venv/bin/python
OUT="./outputs/submit_pmu_none_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
./run.sh submit-pmu ccec none \
  --tensormap private \
  --device 0 --batches 1 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1 \
  --pmu-json "$OUT/submit_icache_raw.json"
```

raw 成功发布后，`run.sh` 会在同目录自动生成一份自包含的加工件：

```text
submit_icache_raw.json       # 96 核权威原始数据
submit_icache_report.html    # 浏览器直接打开的离线图表和汇总
```

compete-first eager 移植后的五种 CCEC A5 b1 回归位于
`outputs/compete_first_submit_pmu_b1_20260720/{none,claim,efdrain,materialize,register}/`。
五轮均为 96/96 trusted，语义、真计算输出、role/triplet、owner Restore、
phase status/time 与 primary/shadow 门禁全部通过；`none` 的 phase calls
为 0，四个局部 phase 均精确为 `96 * 5 = 480` 次。

HTML 中包含 AIC/AIV 的每核 request、miss、miss rate、p95、96 核散点和
局部 phase 的 lower/upper 区间。schema-v5 的 running phase 还在页面
最前面按 ALL/AIC/AIV 并列阶段时间、request 和 miss 占比；阶段时间
是 `Σphase_elapsed_ticks / Σsubmit_elapsed_ticks`，request/miss 是占同一
ELF PMU whole-gate primary 的比例区间。两者分母边界不同，不是同一
精确分区。`none` 显示“不适用”，历史
schema-v4 因没有阶段时间 raw 字段而显示“不可用”。报告也展示
ALL/AIC/AIV 的逐核 PMU `total_cycles`
与 `scalar_busy`：三个响应式角色卡只选 mean 作为典型值，并补充逐核
min/max 显示核间范围；PMU total 与 scalar busy 的极值分别独立计算，不保证
来自同一个物理核。顶部“完整 Submit（最早开始 → 最晚结束）”是 96 核整体
墙钟范围，不能与逐核 PMU mean 混为一个统计量。报告还展示 scalar/total 比例
和逐核散点，并同时保留 raw cycle 与本机校准后的每核等效时间；宽表只在表格
内部横向滚动。受控 cold/warm 同窗实测
`1,817,457 PMU cycles / 1,101,593 ns = 1.649844 cycles/ns`；AIC/AIV
分别使用 `1.650062/1.649731 cycles/ns`，换算式为
`time_us = cycles / cycles_per_ns / 1000`。报告将
`total_cycles-scalar_busy` 明确标为“非 Scalar-busy 残余”。`total_cycles`
是每个物理子核在 PMU whole gate 内的 64-bit raw total，96 核求和是 core-work，
不是约 5 ms 的 Submit 墙钟；`scalar_busy` 是 CNT2 的
`scalar_instr_busy(0x001)`。依赖返回值的 atomic 等待可以落入 scalar busy，
而 I-cache refill 的额外周期可能主要只增加 total，但
**`total_cycles - scalar_busy` 既不能解释为 Scalar 空闲，也不能解释为
I-cache stall**：差值还混有同步等待、Cube/Vector/MTE 等 engine 等待
及其他非 scalar-busy 周期。2026-07-19 用本机 CANN 9.1 在 A5/DAV3510
上依次实测 `PipeUtilization`、`PipeUtilization,MemoryDetail` 和 `Default`，
三份 `PipeUtilization.csv` 均没有 `scalar_wait_ib_time` 或
`scalar_wait_time`；DAV3510 正式事件表也没有对应 selector/公式。因此
当前 A5 正式可编程路径不采这两项，不套用其他产品的事件号。
原始证据位于 `outputs/wait_ib_official_msopprof_20260719_b1_probe2/`、
`outputs/wait_ib_official_msopprof_20260719_b1_probe3_memory_detail/` 和
`outputs/wait_ib_official_msopprof_20260719_b1_probe4_default/`。
报告只复用 `pmu_sidecar_analyzer.py` 已校验的统计口径；生成失败不会删除已经发布
的 raw，但本次 action 会返回非零。

这里的 PMU whole gate 从 orchestration 初始化前开始，到末次
Submit 返回后停止，包含 Submit 内的 EfDrain、Claim、当前模式实际构参
（private 全员 eager；shared Alloc 全员、其余 task winner-only）和
finish，也包含 Submit 间的 `AcceptTaskOutputs()`/调用衔接，排除 FinalDrain。
`submit_elapsed_ticks` 是每核首末 Submit 时间；顶部
`submit_span_us` 则是 96 核共同墙钟范围。三者不能混用，详细定义见
`../icache_miss_usage_guide.md`。

当前边界联动版已对 `none|claim|efdrain|materialize|register` 五个独立
ELF 完成 A5 b1 门禁，五轮均为 96/96 有效记录并通过语义、真计算、
phase call/time、primary/shadow 和 owner Restore。产物位于：

```text
outputs/submit_pmu_boundary_sync_b1_20260719/{none,claim,efdrain,materialize,register}/
```

该 b1 只作源码边界与工具闭合证据，不用不同 phase ELF 的单轮时间差
宣称性能改善。

两类新增局部 phase 可按与 `none` 相同的参数分别运行；输出文件名应体现 phase，
避免误把不同 ELF 的结果放进同一组：

```bash
for phase in materialize register; do
  OUT="./outputs/submit_pmu_${phase}_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$OUT"
  ./run.sh submit-pmu ccec "$phase" \
    --device 0 --batches 1 \
    --winner-workload real-compute --real-compute-counts 6,28,4,1 \
    --pmu-json "$OUT/submit_icache_raw.json"
done
```

`submit-pmu` action 自己固定 `--runs 1 --no-swimlane --pmu-window submit-all`，
不要重复传入这三项，也不能传入 `--profile-phases`、
`--trace-atomics`、`--analyze-swimlane` 或 `--swimlane-json`。

后续 submit-pmu 构建、门禁和边界迭代默认只跑 A5 b1；b256 只用于
阶段性规模/容量收口或明确要求的长负载结论。

Submit-all PMU 整窗的权威 I-cache 主计数是从不在局部边界读取的
`CNT6=request` 和 `CNT7=miss`。A5 b1 实测已反证 `CNT9=0x35`
可作有效计数槽：它始终为 0。因此正式 `submit-pmu` 用
`CNT8=0x34` 作 shadow request、`CNT5=0x35` 作 shadow miss，`CNT9`
保持未使用；这会牺牲 PMU 诊断版的 `mte3_busy`，不影响标准
`swimlane` 构建。

shadow 计数器是 read-to-clear：选中的局部 phase 在 begin/end 切分片段，stop 时
再加 tail，从而软件重建 PMU whole-gate shadow whole。schema-v5 同时在
begin read-clear 之后取阶段起点，在 end read-clear 之前先取终点；因此
阶段时间不包含两侧 `ld_dev`，但包含每次调用两次 SYS_CNT 的观察
扰动。`none` 没有运行中读取，必须
在每个物理子核精确满足：

```text
CNT8 shadow whole request == CNT6 primary whole request
CNT5 shadow whole miss    == CNT7 primary whole miss
```

`claim/efdrain/materialize/register` 在 A5 上运行中反复 read-clear 时，shadow
可能在边界处单向少计，
因此接受条件改为逐核：

```text
shadow request <= primary request
shadow miss    <= primary miss

request loss = primary request - shadow request
miss loss    = primary miss - shadow miss

phase request ∈ [observed request, observed request + request loss]
phase miss    ∈ [observed miss,    observed miss    + miss loss]
```

区间必须逐核构造后再聚合。CNT8/CNT5 是顺序 `ld_dev` 而非原子配对快照，
不要求局部 `phase miss <= phase request`；只要求二者分别不超过对应 shadow，
上界分别不超过对应 primary。`none` 中 phase calls/begin/end/request/miss 必须
全为 0；其余四个 phase 的每核 begin/end/calls 都必须配对，且每核 calls 固定为
`batches * 5`，全局为 `batches * 5 * 96`。原因是每个 batch 固定提交
Alloc/QK/SF/PV/UP 五个 task，每次 Submit 都恰好执行一次 Claim、开头 EfDrain、
Materialize 和 Register 边界。`efdrain` 插点只允许包围 Submit 开头的专属调用，
不能插入复用的 `DrainReady()` 函数体；`materialize` 必须先保存真实返回值再关闭
边界，保证失败出口也闭合；`register` 的 Alloc 与非 Alloc 两个源码调用点互斥，
不能误算成每次 Submit 两次。每个 running phase 还要求每核
`phase_elapsed_ticks > 0`且不超过同核从首个 `submit_begin` 计时点
到末个 `submit_end` 计时点的 `submit_elapsed_ticks`；前者位于首个
`BeginCallbackSubmit()` 上下文初始化之后，后者位于末个 Submit 返回之前。
`none` 的 phase elapsed 必须精确为 0。
当前 Case1 中真实 TensorMap insert 工作主要发生在
UP 的输出注册，其他 task 的 Register 可能很短或没有 insert；因此该 phase 的
固定调用数只证明边界覆盖完整，不能解释为五类 task 拥有等量注册工作。

局部边界读取本身会增加 scalar 指令、改变 I-cache 布局和多核时序，
因此所有非 `none` phase 都是带边界扰动的归因结果。`none` 与每个局部 phase
是不同 ELF/不同进程；不同 phase 必须各自单独采集。不同 phase
ELF 的局部 request/miss **不可相加**，也不能与 `none` 相减后宣称
得到了零扰动的阶段净值。这个区间只约束同一插桩 ELF、当前边界定义下的
局部事件；它不是对应阶段在无插桩构建中的真实区间。调度语义、真计算输出和
placement/engine 门禁都通过时，running shadow 的负差属于观测边界行为，
不得描述为 standalone scheduler 异常。

JSON 保留 96 条 raw record，并按 ALL/AIC/AIV 输出 authoritative whole、
shadow loss 和 phase lower/upper。raw 中包含 `shadow_request_loss`、
`shadow_miss_loss`、`phase_icache_requests_upper_bound` 与
`phase_icache_misses_upper_bound`；schema-v5 还包含逐核
`submit_elapsed_ticks`、`phase_elapsed_ticks`和 `phase_time_valid`。host 还分别
报告 exact/bounded 核数。时间占比必须在 ALL/AIC/AIV 各自范围内按

```text
Σphase_elapsed_ticks / Σsubmit_elapsed_ticks
```

计算；分子、分母都是 1 ns/tick 的逐核 SYS_CNT core-time，不能改用
96 核整体 `submit_span_us`，也不能平均 96 个逐核百分比。
完整 Submit 的组内 miss rate 才按 `sum(misses)/sum(requests)` 计算，不平均
逐核百分比；局部 lower miss/lower request 之比只是 observed read-clear
ratio，不是实际 miss rate 的数学下界。更完整的 I-cache 采集、分析、估算与排错见
[`../icache_miss_usage_guide.md`](../icache_miss_usage_guide.md)。

#### 历史 PMU 校准资料（不属于当前两类正式构建）

以下 `empty/scalar/scalar-double/icache-single`、CNT8 fix-busy 和 schema-v3
文字保留为 2026-07-18 观察链路的建设过程与历史数据。当前
`swimlane` 构建不提供 PMU，当前 `submit-pmu` 也只接受完整
Submit 的 `none|claim|efdrain|materialize|register`；不应继续照抄下文的历史校准命令作为当前用法。

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

#### 历史：Main AICPU Path-A owner

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

#### 历史：生成旧 PMU-only JSON

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
时间。本机现已通过 PMU/SYS_CNT 同窗校准核实其频率，HTML 可按
ALL/AIC/AIV 的 `1.649844/1.650062/1.649731 cycles/ns` 显示每核
cycle-equivalent；该换算仍不能把 96 核 core-work 冒充墙钟。CNT0..CNT8 是
32 bit，total 是 64 bit。正式门禁要求本轮最大可编程 counter 小于 `UINT32_MAX/4`
（25% 高水位），这只是缩短窗口后采用的保守风险阈值；最终 32-bit 值无法证明
计数器没有恰好回卷一圈或多圈，因此通过该门禁也不能声称“已证明无回卷”。

正式观察保留三类互不混用的样本：关闭所有诊断的性能 golden、PMU-only
`submit-all` sidecar、Atomic-trace-only 泳道。PMU sidecar 只有整个窗口的每核累计，
没有可与单条 Atomic span 对齐的子窗口；不能把 AIC/AIV 平均 miss rate 回填成
泳道中某条 atomic 的属性。优化前后需保持相同源码观察布局、winner mode/count 和
owner 配置，
并用多个独立进程交错 A/B。

#### 历史：2026-07-18 上板验收样本

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

#### 历史：单次 CNT7 I-cache miss 的 scalar 一阶估算标尺

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

#### 历史：校验并聚合多轮 PMU sidecar

`pmu_sidecar_analyzer.py` 在该历史流程中只读消费当时的 schema-v3 JSON。它不信任单轮 host 已写好的
summary，而是从每份文件的 worker raw 记录重新计算 ALL/AIC/AIV 的
`sum/mean/median/p95/max` 与 `sum(miss)/sum(request)`；同时检查 accepted、
96 核 start/stop、物理核唯一性、owner membership、角色、counter 门槛和 Restore。
任一字段不一致即拒绝整组输入。

同一次构建、同一观察参数的多个独立进程可直接聚合：

```bash
source /home/q00473782/.venv/bin/activate
python pmu_sidecar_analyzer.py "$OUT"/run*.json
python pmu_sidecar_analyzer.py --json "$OUT"/run*.json
```

工具把 device、batch、AIC/AIV 数、PMU window、selector、trace 配置和完整
winner workload 纳入配置指纹。`submit-all` 与 `empty`、scalar-NOP 与
real-compute 等不同口径不能混合聚合。当前 JSON 没有记录 ELF 内容哈希，所以
调用者仍必须用独立输出目录隔离不同构建，不能只因 kernel 路径字符串相同就认为
是同一二进制。

默认 `--icache-miss-ns 90` 只打印受控 cold/warm 标尺下的一阶 core-work 等效量。
输出明确标记 `not_wall_or_additive_stall`：96 核总和不是 Submit 墙钟，逐核估算也
可能因 miss 重叠、事件来源和真实层级差异超过窗口时间，不能据此做绝对减法。
文本输出的 `[PRIMARY]` 以 AIV 平均 request/miss、AIV 逐核 p95 和组内 miss rate
为主；`[ACTUAL-EXPOSED-LOSS]` 在没有同语义配对 A/B 前固定报告 `UNMEASURED`，
避免把 90 ns 标尺误写成约 5 ms Submit 中已经暴露的损失。
分析器回归可独立执行：

```bash
python -m unittest -v \
  test_pmu_sidecar_analyzer.py \
  test_pmu_html_report.py
```

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
当时必须同样开启 phase-only 泳道且不开逐 atomic；standalone 5 个独立进程为：

```text
5002.413 / 4875.193 / 4968.894 / 4992.477 / 4876.282 us
```

中位数为 **4,968.894 us**。真实 PA 最终三轮为
5,115.620/5,145.057/5,096.685 us，中位数 **5,115.620 us**；同口径差值
146.726 us，约 2.87%，已满足独立调度复现目标。五轮 standalone 的
QK/SF/PV/UP 每 task 均值中位数为 41.461/54.007/28.053/2.649 us，
与真实 44.170/53.729/27.626/1.565 us 的总 core work 接近，不再调整
repeat 追求逐微秒一致。

只保留一轮历史 phase-only 泳道原始证据：

```text
outputs/performance_gap_20260718/standalone_ccec_real_b256_raw.json
```

该轮有 863,237 条记录、`dropped=0`，比真实 PA 的 863,232 条只多 5 条
RingBp；两端的 122,880 个 Submit 与各前端阶段、1,024 个 Kernel/Fanin/Build
数量一致。这证明总体性能已接近，不等于真实 PA 数值数据流、代码生成与通用
多 group/joint 调度已完全相同。该文件用于保留历史 5 ms 同口径性能证据，
不能冒充当前 `swimlane` action 的合并记录；当前 action 还会同时加入 Atomic 与
ClockBaseline，并继续以逐核容量、调用数、总记录数和 `dropped=0` 闭环。

## 7. 内存占用和脱仓复制

为保持真实 DistGlobal/DistCore 偏移、65,536 个 task cell、每 worker payload
和 TensorMap，非 split 的 CPU 构建中 `WorkerResult` 为 896 bytes、
`SchedulerState` 为 1,007,115,968 bytes。CCEC swimlane 以及使用 split-finish
的 submit-PMU 构建为了发布跨 TU 正确性诊断，`WorkerResult` 为
960 bytes、`SchedulerState` 为 1,007,122,112 bytes，增量精确为
`64 * 96 = 6,144` bytes。split ELF 另预留 AIC/AIV 两个 role-specific
block-local runtime state，每个精确 1,664 bytes、最终 section 合计
3,328 bytes；它们不属于 GM `SchedulerState`。以上 `SchedulerState` 数字
是 private 模式。R4c 的 shared sidecar 历史大小为 12,420,288 bytes；
R4e-a 追加 96 条 reader-progress cache line 后，当前 generation-8
sidecar 为 12,426,432 bytes。因此 CPU non-split 与定义
`PA_COMPETE_FIRST_SPLIT_FINISH` 的 CCEC 变体总大小分别为
1,019,542,400/1,019,548,544 bytes；swimlane、perf-clock 以及
submit-PMU none/claim/efdrain 使用后者，submit-PMU
materialize/register 和独立 shared-protocol-litmus 使用 non-split 大小。
既有
production prefix 和
standalone 控制字段 offset 不变。S4.15a/S4.16 历史候选都曾得到
4,736,704B，但末尾 512B 分别是 Cube cursor 和
`shared_vector_cursor[8..15]`，且均已撤销。S4.9 的
4,735,680B、历史 S2.5 的 2,113,664B 和 S3.1 的 4,735,104B 也都
不是当前 shared 构建的传输或分配口径。

S4.16a 的 host/device `sizeof(SchedulerState)` 握手、manifest 校验和
相对重建 `319077a9` 的正式 b256 六区组配对已经完成。不能因历史候选
字节数碰巧相同就混用产物；冻结件及配对路径见
`shared_tensormap_record.md`。S4.16b 沿用相同历史大小并闭合正确性，
但第一层性能门槛失败；当前源码保留 S4.14b 的八分片热路径，同时在尾部
纳入 S6.3 的 output table 扩容、R4c history 和 R4e-a
`reader_done[96]`，不能再把当前传输长度写成“恢复 S4.14b”。

独立的 64 bytes PMU 配置和 64 bytes winner workload 配置各占一条
cache line；二者都位于完整生产 DistGlobal 镜像之后，生产 DistGlobal/
DistCore 关键偏移保持不变。
默认泳道缓冲区另占 402,660,160 bytes；CCEC/AscendC `real-compute` 还在 device 分配
12,713,984 bytes workspace。因此 scalar-nop+trace 的 A5 device 占用约
1.313 GiB，real-compute+trace 约 1.325 GiB，host 侧也需分配相近内存。
CPU 后端只在 host 侧分配相同 workspace，不存在 device 内存口径。
`smoke` 不缩小 State；只有 `--no-swimlane` 能省去泳道缓冲区。
256 batch 的历史 phase-only 采集约有 86.3 万条事件。删除
`LoserReplay` 过程态后，当前 schema-v4 level-4 b256 规模门禁为
845,813 条 raw 事件、56,212,672 bytes；同一 raw 使用旧 merged
字段布局为 248,767,986 bytes，使用当前六字段 duration 布局为
138,349,686 bytes，减少 44.4%。按 96 核聚合而不复制每个 gap
属性的排他报告约 117 KiB。文件瘦身不改变 raw 记录写入数，
也不改变固定 trace buffer 分配；三者是独立口径。
runner、converter 与 analyzer 都使用临时文件后原子替换自己的目标，失败时
不会把半截文件冒充完整产物。

脱离 simpler 时必须复制整个目录，因为三个后端共用 `common/`：

```bash
cp -a tests/atomic_probe/pa_scheduler /tmp/pa_scheduler
cd /tmp/pa_scheduler
./run.sh build cpu
./run.sh smoke cpu
```

上面两条省略 `--tensormap private`，与显式写出 private 等价。shared
模式可在复制目录后显式执行
`./run.sh build cpu --tensormap shared`；构建身份和产物目录会保持 shared，
不会回退到 private。

CCEC/AscendC 只需再 source CANN 环境。本目录的构建脚本不会搜索 Git 根目录，
也不会引用 `simpler/src`、`simpler/examples` 或其他仓内文件。泳道转换与排他
分析都只需 Python 3 标准库；复制后的 `./run.sh swimlane ...` 仍使用当前目录内的
`swimlane_converter.py` 和 `swimlane_exclusive_analyzer.py`。
