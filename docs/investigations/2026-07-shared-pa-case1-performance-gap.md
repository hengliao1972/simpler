# Shared PA Case1：standalone 2.3 ms 不是等价基线，六个微优化候选均未保留

**Date**: 2026-07-30
**Verdict**: deferred-pending-equivalent-benchmark

## Question

把 standalone 的 shared TensorMap 与 schema-v5 泳道图迁入 simpler 的真实
Paged Attention（PA）Case1 后，是否还存在一项可删除的软件开销，能够把
`perf-clock` 的完整 Submit 窗口从约 2.7 ms 降到 standalone 的约 2.3 ms？

这个问题容易把“相同的 96 个 worker 和 1,280 个 task”误当成“相同的
benchmark”。本调查先核对两边实际执行的工作和 Claim 拓扑，再只验证能够由
生成代码或泳道图支持的候选，不通过反复跑设备碰运气。

## Measurement scope

迁移基线为本地提交 `18a88a41`：

```text
Update: 迁移 PA 专用 shared TensorMap 与 schema-v5 泳道图
```

真实 PA 的锁定口径为：

- Case1、单 group、B256；
- 96 workers、每核 1,280 次 Submit；
- Alloc 仅 8 个 AIC 候选，QK/PV 为全部 32 个 AIC，
  SF/UP 为全部 64 个 AIV；
- 256 MiB shared heap、8 个 heap shard、8 个 vector Claim cursor shard；
- UP writer history 含 3 条 symbol record，并使用 group CAS 和
  predecessor handoff；
- `perf-clock` 设备计数频率为 1 GHz。

机器上没有可用的 `task-submit` / `perf_lock` 执行面，因此所有 A5 数字都是
**未锁频、未隔离**的 fresh-process 结果。它们只适合判断明显回归或稳定方向，
不适合作为亚百分比差异的最终验收。

冻结基线的五个 fresh-process Submit 样本为：

```text
2665.43, 2679.92, 2746.03, 2818.71, 2968.16 us
median = 2746.03 us
mean   = 2775.65 us
range  = 302.73 us
```

## Why the standalone number is not comparable

standalone 的十个 fresh-process `perf-clock` 样本中位数为
`2304.520 us`，但它与真实 PA 只共享 worker 数和 task 数：

| Dimension | standalone | simpler real PA |
| --------- | ---------- | --------------- |
| Workload | fixed synthetic tiles | real BF16 PA tensors/deps |
| Alloc candidates | 96 workers | 8 AIC workers |
| Candidate topology | fixed masks | A8 / Q32 / S64 / P32 / U64 |
| Total Claim attempts | 73,728 | 51,200 |
| Arrival shape | synthetic orchestration | data-dependent PA orchestration |

因此 `2746.03 - 2304.52 us` 不是已经被证明的“迁移损失”。改变真实 PA 的
候选掩码或业务工作量去复现 2.3 ms，会违反本阶段已经锁定的调度协议，也会把
benchmark 换成另一个问题。

完整 schema-v5 泳道图也必须同口径比较：真实 PA 为 `2.853980 ms`，
standalone full trace 为 `2.367119 ms`。不能把带完整泳道图的一个数字与
standalone 的低观测 `perf-clock` 数字混用。

## Migration parity audit

standalone 的性能分支在迁移基点之后还有 23 条提交；提交数量本身不能作为
“漏迁 23 项优化”的证据。逐条按最终源码审计后，实际构成为：

- 14 条保留代码；
- 9 条纯文档或已经撤回的候选；
- 14 条保留代码中，12 条已经在 simpler-PA 中字面对等实现，或被更轻的
  架构实现覆盖；
- 真正没有保留的只有 occupied 快照提前停止 slot 扫描，以及 PV 复用 SF
  task-id 两项。

前一项已在真实 PA 单独移植、通过 focused test 后实测并撤回：目标 drain phase
没有同步改善，全局中位数变化也落在未锁频噪声内。后一项在 standalone 的直接
AIC 路径只节省约 `5.791 ns/group`，即使把 256 个 group 全部串在关键 lane
上，极乐观上限也只有：

```text
5.791 ns × 256 = 1.48 us
```

完整五阶段 standalone 测量为约 `-0.320%`，同时 AIV 约 `+0.218%`。它可以
作为低优先级的源码对等项，但不能解释 `0.4–0.5 ms`。

已经核对的主要阶段包括：

- Begin/Claim：replay role/block 只在入口取得一次，loser 在 Begin TU 内闭合；
- Materialize/Publish：仅 1,280 个 winner 执行，descriptor 和 UP writer
  history 在 ordered Register 前准备；
- Register：predecessor 等待、group writer、store barrier、CAS handoff
  顺序与 standalone 对等；
- FanIn：按 SharedOutputRef 的 published、last-writer 和 history 解算；
- EfDrain：shared 正常路径只扫描两个可用 slot；
- FinalDrain：两边均使用 16 个叶组的两层 barrier；
- orchestration：winner-only 构参，并按 AIC/AIV 角色只保留后续会消费的
  output symbol。

所以后续性能判断以最终生成代码和墙钟关键链为准，不再以“还有多少提交没
cherry-pick”作为判断依据。

## Wall-clock critical-chain evidence

下式直接计算完整 Submit envelope，不把多个 core 的 duration 相加：

```text
T = max(Submit.end) - min(Submit.begin)
```

两份 schema-v5 trace 均为 1 GHz、96 workers、每核 1,280 Submit、
drop 0，并通过整数闭合。结果为：

```text
real PA     2,853,980 cycles = 2853.980 us
standalone  2,367,119 cycles = 2367.119 us
difference    486,861 cycles =  486.861 us
```

最后完成 scalar lane 的连续排他分区如下。表中的 Kernel 是 EfDrain 内本核
实际执行的 linked kernel union，不是另外再加一次的 duration：

| Partition | Real PA | standalone | Difference |
| --------- | ------: | ---------: | ---------: |
| EfDrain | 1178.338 us | 802.507 us | +375.831 us |
| └ local Kernel union | 939.101 us | 611.968 us | +327.133 us |
| EfDrain excluding local Kernel | 239.237 us | 190.539 us | +48.698 us |
| Register | 186.270 us | 12.702 us | +173.568 us |
| └ predecessor poll | 165.141 us | 5.207 us | +159.934 us |
| Register excluding predecessor poll | 21.129 us | 7.495 us | +13.634 us |
| BetweenSubmit | 285.557 us | 183.988 us | +101.569 us |
| Materialize | 190.028 us | 130.798 us | +59.230 us |
| Claim | 778.344 us | 840.062 us | -61.718 us |
| Submit residual | 111.715 us | 250.650 us | -138.935 us |
| FanIn | 36.097 us | 58.405 us | -22.308 us |

EfDrain 正差的 `87.04%` 是真实 Kernel，Register 正差的 `92.14%` 是
predecessor 等待。两项差值会随一次运行中的执行到达顺序，在 EfDrain 与
Register 之间迁移，不能分别解释成两份可删除的软件开销。

按每个 batch 的 UP Submit wall frontier 重新分区后，还有两个重要边界：

- 生产 batch 中位数为 `6.042 us`，standalone 为 `6.755 us`；
- 256 个配对 batch 中，生产只在 127 个 batch 更慢；
- 最后 64 个 batch 贡献总差值的 `56.82%`；
- 最大的两个生产前沿增量出现在 batch 251 和 255，分别为
  `146.087 us` 和 `103.859 us`。

这不是每 batch 固定增加约 `1.9 us` 的软件税，而是少数执行长尾及其等待传播。
主 trace 的两个直接证据是：

```text
SF#1257: production 127.111 us, standalone 55.714 us
QK#1276: production 256.150 us, standalone 40.595 us
```

其中 `QK#1276` 单事件差 `215.555 us`，但与其他 lane 流水重叠后，对最后
batch Submit frontier 的净差只有 `103.859 us`。它仍然是业务 Kernel/设备
长尾，不能改写成 scheduler exclusive 软件耗时。

若比较完整 worker completion，而不是只看 Submit envelope，真实 PA 为
`2987.504 us`，standalone 为 `2462.685 us`，差值为 `524.819 us`。

### Ordered Register frontier is a different metric

按 task-id 的有序 Register 端点观察，真实 PA 的上一批 UP 到下一批 Alloc
端点间隔均值为 `5.128 us`，standalone 为 `0.602 us`。它不能与每核
`between_submit_residual` 的 `550.109 ns` / `252.780 ns` 混为一谈：
前者包含跨核 winner 到达、Materialize 和 predecessor handoff，后者只是一条
连续 replay lane 上两个 Submit 之间的排他间隙。

进一步分解 Alloc winner 的到达：

| Alloc winner timing | Real PA | standalone |
| ------------------- | ------: | ---------: |
| Submit begin relative to previous UP Register end | -11.941 us | -33.113 us |
| Submit begin to Alloc Register begin | 13.385 us | 15.675 us |
| Register begin relative to previous UP Register end | +3.843 us | -17.450 us |

真实 PA 的 Alloc winner 路径本身反而更短，但 8 个 AIC 候选没有 standalone
96-worker 候选人口所带来的提前到达量。因此 `5.128 us` 主要说明已经锁定的
Alloc8/AIC 到达拓扑不同，不能归因为 Register metadata body 回归。

## Final ELF and I-cache layout audit

使用最终 perf-clock AICore execution sections 对比，而不是比较源码行数：

| Artifact | `.text` | `.rodata` |
| -------- | ------: | --------: |
| simpler real PA | 96,984 B | 532 B |
| standalone | 130,104 B | 432 B |

standalone 的 `.text` 反而大 `33,120 B`，即 `34.15%`。两边热函数的
CFA 都为 `1,952 B`，保存相同的 callee-saved 寄存器；最终 ELF 都没有
relocation、thunk 或 veneer，代表 Finish 调用均为直接 branch。因此当前没有
证据支持“simpler 因代码更大或远跳转多而慢 0.2 ms”。

生产版 AIV orchestration 与 common Finish 的物理间隔更大。交换 AIC/AIV
COMDAT 顺序可以在不改变 section 大小和 rodata 的情况下把 AIV orchestration
前移，但物理间隔只可能改变 cache-set 映射或顺序预取，不能证明中间代码会被
取指。按已测单次 AIV I-cache miss 约 `94.030 ns` 估算，解释 `0.2 ms`
需要关键路径少约 2,127 次串行等效 miss；当前同一 perf-clock ELF 没有对应
PMU 证据。因此没有为了这个未证假设反复跑 A5 或扩大代码体积。

## Why EfDrain-before-Claim was retained

production 与 standalone 都按 `EfDrain -> Claim` 执行。这个顺序是
execute-first 负载均衡与执行进展点，不是单纯的控制开销。

真实 PA 主 trace 中，71,680 个 `claim.not_attempted` Submit 的 EfDrain
一共承载 790/1,024 个 Kernel，即约 `77%` 的 Kernel：

```text
AIC during SF       320
AIC during UP        63
AIC noncandidate
    during Alloc     17
AIV during Alloc    185
AIV during QK        50
AIV during PV       155
```

因此按 role 不匹配跳过 EfDrain，或把所有 Claim 挪到 EfDrain 之前，并不会
消除这些 Kernel。shared 正常路径只有两个可用 ring slot，winner build 会在
满槽时进入 RingBp 并继续 drain；FinalDrain 又必须同时观察全局 release 和
`occupied_count == 0` 才能退出。重排只会把大部分时间移到 WinnerBuild、
RingBp 或 FinalDrain，并改变 first-winner 的自然负载均衡。

只把当前 Alloc 的 8 个候选改为 claim-before-drain，在协议上可以保持 ring
和 FinalDrain 正确，但当前 Alloc winner 的 EfDrain 只有约
`0.232 us × 256 = 59 us` 的 trace 极乐观上限。把同一 future-Alloc 候选的
前序 Alloc/SF/UP drain 一起延后，观察构建的上限约 `0.24 ms`，却会延迟旧
QK/PV 完成并把耗时搬到其他阶段，不能作为完整 worker 优化。基于这些证据，
本调查没有为重排候选执行无锁频 A5 试跑。

## What was tried

所有候选都先检查协议不变量。候选 1、2 由泳道图提出并做有界 A5 测量；
候选 3–5 还用 whole-program O3 IR 证明预期指令或控制流确实发生变化；
候选 6 先由等价物理模型筛选，再用最终 CCEC object 尺寸证明 hint 进入热函数。
代码候选最终均已撤回。

### Common validation and artifact identity

focused protocol test：

```bash
cmake --build build/ut_cpp --target test_fdwic_shared_pa_submit -j2
build/ut_cpp/test_fdwic_shared_pa_submit --gtest_color=no
```

A5 fresh-process 测量命令：

```bash
PTO_ISA_ROOT=<pto-isa-build> \
PYTHONPATH="$PWD/python:$PWD/build/cp312-cp312-linux_x86_64/python/bindings" \
.venv/bin/python -m pytest \
  examples/a5/fully_distributed_within_core/paged_attention_unroll/\
test_paged_attention_unroll.py \
  --platform=a5 --device=0 \
  --case=TestPagedAttentionUnroll::Case1 \
  --fdwic-tensormap=shared --fdwic-profile=perf-clock -q
```

CCEC 检查使用与 `CMakeLists.txt` 相同的 `-O3`、dav-c310 AIC/AIV 架构和以下
关键宏编译 unity orchestration，再把 driver 打印的 `-cc1` 命令从
`-emit-obj` 改成 `-emit-llvm-bc`：

```text
PTO_FDWIC_SHARED_MAP=1
PTO_FDWIC_SHARED_PA_UNITY=1
PTO_FDWIC_PERF_CLOCK=1
PTO_FDWIC_TRACE_ENABLED=0
```

可核对的构建身份：

```text
baseline source: 18a88a419b0c2d90012517106f6229fff3437f00
baseline aicore_kernel.o sha256:
  9e6b0c5303d7fc72ae8c5277e597ee2dbf9342ae6c37cdf4af05e39e0bb27a78
replay-pointer candidate aicore_kernel.o sha256:
  5becdffac8543fb10f18018871e48cb7d5bb7a4fabb3da4c887562dd55e7511c
fresh-output preload candidate aicore_kernel.o sha256:
  15f968a126550bf405817dc57da4420c2d05353933d311549c3a0fb9f95ac215
```

候选 3、4、5 的 A5 输出目录分别为：

```text
typed Claim:
  TestPagedAttentionUnroll_Case1_20260730_220753
  TestPagedAttentionUnroll_Case1_20260730_221016
  TestPagedAttentionUnroll_Case1_20260730_221104
  TestPagedAttentionUnroll_Case1_20260730_221151
  TestPagedAttentionUnroll_Case1_20260730_221237
fixed geometry:
  TestPagedAttentionUnroll_Case1_20260730_221905
  TestPagedAttentionUnroll_Case1_20260730_222049
  TestPagedAttentionUnroll_Case1_20260730_222135
  TestPagedAttentionUnroll_Case1_20260730_222221
  TestPagedAttentionUnroll_Case1_20260730_222308
replay pointer:
  TestPagedAttentionUnroll_Case1_20260730_223731
  TestPagedAttentionUnroll_Case1_20260730_223900
  TestPagedAttentionUnroll_Case1_20260730_223957
  TestPagedAttentionUnroll_Case1_20260730_224054
  TestPagedAttentionUnroll_Case1_20260730_224148
fresh-output preload:
  TestPagedAttentionUnroll_Case1_20260730_235402
  TestPagedAttentionUnroll_Case1_20260730_235516
  TestPagedAttentionUnroll_Case1_20260730_235602
  TestPagedAttentionUnroll_Case1_20260730_235649
  TestPagedAttentionUnroll_Case1_20260730_235736
```

候选 1、2 是未提交的早期原型，撤回时没有保留 source snapshot 与稳定的输出
目录映射；候选 3、4 也没有保留可链接 object。这里的描述和数值足以避免重复
追逐已否定的方向，但不能把这些临时 artifact 当作可复现性能基线。若要重开，
必须从 `18a88a41` 按下文描述重新实现，并重新建立完整 artifact identity。

### 1. Relocate the shared Claim cursor

原理是假设 writer-history 发布前后的 cache-line 竞争影响下一次 Claim，因此
原型把 cursor 更新挪到 writer-history 之后，并配套调整 padding/layout。

协议和 golden 校验通过，但五次测量的中位数约从 `2746.03 us` 上升到
`2861.6 us`，约回归 `4.2%`。这个方向被撤回。

### 2. Snapshot occupied state in drain phase B

原理是在一次 drain 中缓存 `occupied_count`，避免无变化时重复读取，并在已知
为空时提前退出。

focused tests 通过；全局 Submit 中位数只变化约 `-1%`，落在未锁频噪声内，
而泳道图中的目标 drain phase 没有同步改善。这个方向被撤回。

### 3. Specialize Claim and `MixedKernels` by task kind

先核对整程序 IR 后发现，原始 Claim wrapper 已经折叠成直接的
`llvm.hivm.atom.MAX.G.s64` 与结果比较，并不存在额外函数调用层。随后仍验证了
按 task kind 固化 role/kernel 字段的原型：

- AIC/AIV replay body 的 `ctpop` 从 `4/4` 变为 `0/0`；
- `MAX.G.s64` 数量不变；
- 最终 `.text` 均为 96,984 B。

候选五次样本为：

```text
2660.82, 2765.74, 2786.35, 2844.70, 2868.38 us
median = 2786.35 us
```

相对冻结基线中位数回归 `40.32 us`（`+1.47%`），且区间重叠。这个方向被撤回。

### 4. Constant-fold the phase-1 geometry

shared Case1 gate 已限定 block size 128 和单 group，因此原型把 runtime 除法
改为右移，并把只执行一次的 group loop 展开。

整程序 IR 证明：

- AIC/AIV 中相关 `udiv` 均变成 `lshr 7`；
- `bn += 64` 的 loop latch 被删除；
- Claim 原子和 `ctpop` 数量不变；
- 最终 `.text` 仍为 96,984 B。

候选五次样本为：

```text
2693.90, 2702.31, 2806.76, 2849.86, 2881.37 us
median = 2806.76 us
```

相对冻结基线中位数回归 `60.73 us`（`+2.21%`）。这个方向被撤回。

### 5. Carry the attached `DistCore` pointer in the replay capability

原实现的 8-byte replay token 只保存 role 和 block id。每个静态 Submit 点仍
读取一次 block-local `g_self`。原型把 attach 时已经验证的 `DistCore` 指针也
放入 token，并在 PA replay 入口一次性验证编译角色。

整程序 IR 证明该原型确实删除了目标工作：

- 两个 replay body 的十个正常 Begin `g_self` load 全部消失；
- 全模块 `g_self` load 从 23 降到 13；
- `MAX.G.s64` 数量保持 15（含声明）；
- 没有新增 `memcpy` / `memmove`；
- 最终 `.text` 从 96,984 B 降到 95,704 B。

但 token ABI 从 8 B 扩到 16 B，并引入额外 context materialization。A5 五次
样本为：

```text
2694.15, 2703.60, 2739.43, 2739.90, 2873.22 us
median = 2739.43 us
mean   = 2750.06 us
```

相对冻结基线中位数只有 `-6.60 us`（`-0.24%`），远小于两组未锁频样本的
波动，不能证明收益足以承担 ABI 和能力生命周期复杂度。这个方向被撤回。

若未来重启该原型，还必须同时满足以下安全条件：

- 捕获的 pointer 只能作为同步 replay 动态范围内的非逃逸 capability，不能向
  orchestration 暴露可修改的 `DistCore *`；
- winner Finish 除了重读 role/block，还必须比较当前 `g_self` 与捕获指针完全
  相同，避免另一份相同 role/block 的 `DistCore` 通过；
- 8 B/align-4 到 16 B/align-8 的 ABI 变化必须触发 caller/callee 全量重编；
  C++ mangled name 本身不编码类尺寸，不能手工混用旧新 object；
- IR 验收必须沿 CFG 确认成功 Begin 路径无加载，不能要求整个文件零
  `g_self`，因为 getter、winner revalidation 和冷错误路径仍应保留。

### 6. Preload fresh shared-output destination lines

独立 shared 物理模型已经证明：保持 descriptor clean-out 与 barrier 不变时，
128 B 和 384 B fresh destination 的无额外 gap publish 分别稳定下降约
`9%` 和 `25%`。这只能支持业务 A/B，不能直接外推为 Submit 收益。

原型在参数校验后、authoritative writer reservation 前，按 64 B cacheline 对
`shared_outputs[task_id].tensors[]` 发起 DCache preload。放置依据为：

- 当前 task 的 Claim winner 独占整个 fresh descriptor cell；
- `Tensor::copy` 会覆盖每个 128 B descriptor 的全部字节；
- reservation 的一个或三个 FetchMax 可以给 hint 提供提前量；
- 原有全 slot reservation、descriptor copy、DCCI clean-out、StoreBarrier 和
  published Exchange 的顺序完全不变；
- preload 只是 hint，失败路径和正确性都不消费其结果。

focused heap/协议测试共 19 项通过，其中单测精确校验三个 descriptor 对应六条
连续 cacheline。用同一 CCEC O3/Case1 unity 命令编译纯基线与候选：

```text
paged_attention_orch_aic.o generic .text: 0x8ad8 -> 0x8af8 (+32 B)
dist_shared_pa_finish_winner:             0x5864 -> 0x5880 (+28 B)
AIC/AIV orchestration entry size:         unchanged
```

所以该原型不是被编译器删除的空改动。A5 五个 fresh-process 样本为：

```text
2667.51, 2705.48, 2717.62, 2740.01, 2752.11 us
median = 2717.62 us
mean   = 2716.55 us
range  =   84.60 us
```

相对冻结基线中位数只变化 `-28.41 us`（`-1.04%`）。当前既没有
`perf_lock`，也没有同时段交错 baseline；这个差值小于既定的 `0.1 ms`
正常波动边界，不能证明端到端收益。实验提交 `ff12278e` 已由
`b3665903` 完整撤回。

a5sim 还暴露了一个独立门禁问题：候选与撤回前基线的镜像
`sizeof(Runtime)` 均为 71,104 B，但现有缓存组合仍由 AICore 报 build identity
mismatch；只读基线 worktree 又没有已安装 runtime，无法完成同命令判别。
真实 A5 的五次运行均通过 build identity、协议和 golden。因为 a5sim 失败没有
证据指向本原型，且源码已完整撤回，本调查没有通过搬运或删除缓存掩盖该问题。

## Result

| Candidate | Product signal | A5 delta | Verdict |
| --------- | -------------- | -------- | ------- |
| Cursor relocation | Work moved | About `+4.2%` | Dropped |
| Drain snapshot | Target phase unchanged | About `-1%` | Dropped |
| Typed Claim | Removed `ctpop` | `+40.32 us` | Dropped |
| Fixed geometry | Removed `udiv`/latch | `+60.73 us` | Dropped |
| Replay pointer | Removed 10 loads | `-6.60 us` | Dropped |
| Fresh-output preload | Hot Finish +28 B | `-28.41 us` | Dropped |

结论不是“生成代码没有变化”：候选 3–6 都有明确的产物变化。结论是这些变化在
当前真实 PA 和未锁频 A5 上没有形成可重复、足以承担复杂度的 makespan 收益。

## Why winner balancing and tail drain are not local fixes

对主 trace 的 1,280 个 task 逐个排序后，winner 的 FetchMax 返回顺序
`1280/1280` 都是第一名；开始顺序也有 `1186/1280` 是第一名。最明显的
`QK#1276` 长尾为 `256.150 us`，但获胜 block22 的 FetchMax 开始时间仍比第二
候选早 `2.780 us`。该核此前 11 个 QK 都只用 `44.670–47.334 us`，异常发生在
获胜之后，Claim 前无法由在线累计负载预测。

三份 full trace 的正常区间中位数稳定，末段最大值却漂移：

| Kernel | Three-run median range | Three-run max range |
| ------ | ---------------------: | ------------------: |
| QK | `45.797–46.171 us` | `69.902–256.150 us` |
| SF | `51.936–54.139 us` | `107.996–162.577 us` |
| PV | `28.373–28.531 us` | `104.681–176.141 us` |

长尾集中在最后约 62 个 task，但具体 task/core/幅度跨运行变化。真实 PA 每个
QK/PV 会经随机 block table 读取大规模 K/V backing；standalone synthetic
反复复用两块固定 input tile。这进一步说明末段设备/GM 竞争不能被归成一条
稳定的 scheduler 软件税。

当前 shared 正常路径只有两个可用 won slot；每次 Submit 在 Claim 前已经执行
一次 drain，满槽时继续 drain，FinalDrain 又必须等待 global release、ring empty
和无 pending won。代码中没有 last-N watermark、batch-aware throttle 或
eager-drain 旋钮。额外末段 drain 只会移动 FinalDrain 边界；改变 slot、
admission 或 winner 委托则会改变已经锁定的 first-winner 合同。

Materialize 的 whole-program O3 审计确实发现重复扫描、第二次 `%5`、动态
output-mask walk 和可折叠的 validated-no-init 冷分支。它适合以后合并成一次
compact per-kind plan，但最终 lane 的整个 Materialize 也只有 `190.028 us`，
其中昂贵的 heap atomic、完整 128 B descriptor、FetchMax、DCCI、barrier、
Exchange 和 writer history 都不可删除。因此该方向不能静态支持
`>=0.2 ms` 的墙钟候选，本轮没有为它继续跑设备。

## Trace evidence

full-trace 的 aggregate core-time 差值用于定位，不等于 wall time，也不能跨
lane 直接相加：

| Partition | Real PA minus standalone aggregate core-time |
| --------- | -------------------------------------------: |
| Claim | +10.190 ms |
| Register | +13.678 ms |
| BetweenSubmit | +9.948 ms |
| EfDrain | +8.884 ms |
| FinalDrain | +24.252 ms |

进一步核对后：

- Register 差值的 `96.42%` 是既定 predecessor handoff 的等待，不是可删除的
  TensorMap 插入代码；
- EfDrain 和 FinalDrain 主要反映依赖完成与 replay 到达分散；
- UP 到下一 Alloc 的平均间隙为真实 PA `550.109 ns`、standalone
  `252.780 ns`，AIC/AIV 的额外部分都约 0.30 us，更像共同的 orchestration
  边界工作，而不是 UP winner 独有的 publish 行为；
- 只对这个 per-lane 边界做极乐观估算：
  `(550.109 - 252.780) ns × 255 transitions = 75.82 us/critical lane`。
  这假设每次差值都串行落在同一关键 lane 且可被全部删除；它不是由上表
  aggregate core-time 相加得到的 wall-time。即使达到这个不现实的上界，也
  不足以把真实 PA 的 2.7–2.8 ms 变成 2.3 ms。

## Why not now

候选 1 是工作搬移，候选 2 的目标 phase 没有改善，候选 3–6 的生成代码按预期
变化；但没有一个得到稳定的 A5 cross-core Submit makespan 收益。继续在当前
未锁频环境中堆叠微优化或增加重复次数，会把噪声当成结论。

更重要的是，standalone 2.3 ms 和真实 PA 基线不是同一个 workload/Claim
协议。当前证据不支持把两者的差值定义为待修复回归，也没有发现足以解释
0.4–0.5 ms 的单项纯软件开销。

## When to reconsider

满足以下任一条件时再打开本调查：

1. 建立 standalone 与 simpler 完全等价的 benchmark：相同业务工作、相同
   candidate mask、相同 Claim 次数与到达拓扑、相同宏和工具链；
2. 明确决定把目标改为复现 standalone synthetic workload，而不是保持真实 PA
   协议；
3. `task-submit` / `perf_lock` 恢复后，用至少 12 个 fresh process、固定构建
   artifact 和 source hash 建立真实 PA 自身基线；
4. 新泳道图把某个纯软件 exclusive partition 证明为稳定的关键路径热点，再为
   该热点设计单一候选。

若选择真实 PA 口径，应以它自己的锁频基线和 correctness/closure 作为验收，
而不是继续把 standalone 2.3 ms 当作硬门槛。

## References

- Migration baseline: `18a88a41`
- `docs/fully_distributed_within_core.md`
- `docs/dfx/l2-swimlane-profiling.md`
- `docs/dfx/l2-timing.md`
- `tests/ut/cpp/a5/test_fdwic_shared_pa_submit.cpp`
- `examples/a5/fully_distributed_within_core/paged_attention_unroll/`
