# A5 Cache-Line Cross-Core Probe

## Goal

A5 每个核的 scalar data cache 没有 CPU 式多核 cache coherence。测试需要回答：多个核并发访问同一
或不同 64B cache line 时，哪些 scalar 访问方式能保持精确值，哪些 cache 管理路径会覆盖或遗漏修改。

测试分别覆盖 AscendC API、CCEC 原始 intrinsic，并以无数据竞争的 CPU 多线程程序作为 coherent
control。结论只能来自精确判定标准或明确标为观察项的统计，不能用一次随机现象替代契约。

DCCI、`st_dev` 与 atomic 的 API 功能、隔离规则和代码评审清单见
[`ATOMIC_USAGE_GUIDE.md`](ATOMIC_USAGE_GUIDE.md)。

## 当前验证状态

- 2026-07-11：所有 AscendC source 已使用本机 CANN 9.1 `bisheng`、`dav-3510` 编译通过。
- 2026-07-11：CCEC runner 默认编译 AIV-only kernel；已验证 kernel、link 与 host 构建链路。
- 2026-07-11：CPU control 在 `-Wall -Wextra -Werror` 下编译并运行通过；pytest 节点通过。
- 2026-07-11：经用户授权直接使用 device 0；AIV-only 权威矩阵与 AscendC/CCEC 同-line 最简对照已上板。
- 2026-07-11：两个 `st_dev_same_line` 用例按正确性契约断言同-line mismatch 必须为 0；当前设备会
  复现 mismatch 并返回非零。其余 control 与完整入口结果见上板记录。
- 2026-07-13：新增 AscendC/CCEC `atomic_exch_same_line` 同构对照；三组路径均为 `0/4000` mismatch。
- 2026-07-13：CCEC `st_dev_same_line` 新增三 AIV 拓扑模式与低压力单 AIV 同址 mode；记录 raw
  `core/subblock` 与 PTO 公式派生的 `comm_slot` 后，AIV0+AIV2 同-line 路径仍复现 mismatch。当时
  单 AIV 同址 `0/2000` 只是一轮历史低压力样本，已被下述独立压力结果取代。`comm_slot` 不是物理组号。
- 2026-07-13：新增 AscendC/CCEC `st_dev_single_core_stress`。只启动一个 AIV，不调用跨核同步；
  line1 单址、仅 loop-end DSB 分别复现 `3/2000000` 与 `4/2000000` mismatch，line1/line2 双址分别
  复现 `83821/1000000` 与 `78130/1000000`。同址和双址的逐写 DSB 控制均为 0 mismatch。
- 2026-07-13：新增 AscendC/CCEC `st_dev_separate_line_stress` 四模式独立压力。两个 AIV 始终写不同
  64B line；line0/1 布局高频复现，line1/2 布局低频复现，证明旧 `0/4000` 分-line 样本不能作为
  “不同 cacheline 必然安全”的证据。
- 2026-07-13：新增 DCCI selector 五模式与 AtomicExch 同/分-line 八模式；两种前端结果一致，
  同-line 的 ALL/OUT/ATOMIC 三项按正确性门禁返回非零，另外五项 control 全部通过。
- 2026-07-13：新增 `ATOMIC_USAGE_GUIDE.md`，汇总 DCCI、无参 DCI、bypass load/store、atomic、
  DSB 的本机 API 边界、实测状态和 64B cacheline 隔离规则。
- 2026-07-14：新增 CCEC `ld_dev_fanout_publish`。24 AIV 受控对照中，ordinary+DSB 为
  `0/4416` 可见，st_dev+DSB 与 AtomicExch 均为 `4416/4416`；72 AIV 持续读压力同时破坏独立
  control 对照，单列记录为高压力进展失败，不能外推为某个 data writer 的独立语义结论。
- 2026-07-18：新增 CCEC 单 AIV `atomic_scalar_pmu`，以 EMPTY/SCALAR_CONTROL 扣除
  gate 和同构标量递推开销。三个独立会话的 8192 次 dependent `atomicAdd` 均显示：
  atomic 额外 PMU total 几乎 100% 同步增加到 `scalar_instr_busy(0x1)`。
- 2026-07-18：新增 CCEC 单 AIV `icache_scalar_pmu`，在同一静态调用点配对执行
  WARM/COLD 同一 target。三个独立会话共 33 对都是 `WARM miss=0`、
  `COLD miss=68`；COLD-WARM 只增加 48 scalar busy cycle，但增加 `2309..2312`
  total cycle，证明本场景中 I-cache refill 等待的绝大多数周期不计入 scalar busy。
- 原始环境与定量结果记录在 `tests/ATOMIC_MINIBENCH_ONBOARD_LOG.md` 的 2026-07-11 与 2026-07-13 小节。

## 权威覆盖矩阵

`ascendc/cacheline_matrix.asc` 与 `ccec/cacheline_matrix.cpp` 使用相同数据布局和相同值生成公式。

| Mode | 宽度 | 布局 | 参与者 | 精确判定标准 |
|---:|---:|---|---|---|
| 0 | 1B | 同一 cache line、不同 slot | 2/4 blocks | 每个 slot 等于最后一轮精确值 |
| 1 | 2B | 同一 cache line、不同 slot | 2/4 blocks | 同上 |
| 2 | 4B | 同一 cache line、不同 slot | 2/4 blocks | 同上 |
| 3 | 8B | 同一 cache line、不同 slot | 2/4 blocks | 同上 |
| 4 | 1B | 每个参与者独占 cache line | 2/4 blocks | 同上 |
| 5 | 2B | 每个参与者独占 cache line | 2/4 blocks | 同上 |
| 6 | 4B | 每个参与者独占 cache line | 2/4 blocks | 同上 |
| 7 | 8B | 每个参与者独占 cache line | 2/4 blocks | 同上 |

权威门禁使用 AIV-only binary：2/4 blocks 表示 2/4 个 vector 核，不是单核测试。每个参与者通过 kernel
内计数和 marker 精确验证。AIC+AIV MIX 不是本 goal 的必要条件；AscendC runner 仅在显式设置
`ATOMIC_PROBE_RUN_MIX=1` 时运行补充 MIX 覆盖，且 MIX 比例不参与充分性判定。

### AscendC 与 CCEC 同构关系

| 语义 | AscendC | CCEC | 本机 CANN 9.1 依据 |
|---|---|---|---|
| bypass write | `WriteGmByPassDCache<T>` | `st_dev` | `kernel_scalar.h` 的 1/2/4/8B 分支 |
| bypass read | `ReadGmByPassDCache<T>` | `ld_dev` | 同上 |
| AIV-only barrier | `SyncAll<true>()` | flag 14 FFTS 协议 | `dav_3510/kernel_operator_sync_impl.h` |
| atomic | `AtomicAdd/AtomicMax` | `atomicAdd/atomicMax` | `dav_3510/kernel_operator_atomic_impl.h` |

CCEC 不再使用 GM atomic counter 模拟 `SyncAll`，避免 barrier 本身污染待测 cache line 或增加额外 GM
竞争。`ccec_utils.h` 按上述本机实现映射 FFTS 同步。

### DSB 的本机实现边界

DSB 是 Data Synchronization Barrier，不是 Data Store Barrier。本机 CANN 9.1 / dav-3510 的调用链为：

```text
DataSyncBarrier<MemDsbT::ALL>()
  -> DataSyncBarrierImpl<MemDsbT::ALL>()
  -> dsb(DSB_ALL)
  -> __builtin_cce_dsb
```

本机 `cce_aicore_intrinsics.h` 对 `DSB_ALL` 的注释为 “Wait for all memory access instructions.”。它等待
当前核此前的相应 memory access，不是跨核 rendezvous，也不提供 cache coherence；跨 AIV 会合另由
`SyncAll<true>()`/FFTS flag 14 完成。

## DCCI 参数的本机定义边界

本机 CANN 9.1 `cce_aicore_intrinsics.h` 对第二参数有直接注释：

- `SINGLE_CACHE_LINE=0`：处理地址对应的单个 data-cache cacheline entry；
- `ENTIRE_DATA_CACHE=1`：处理整个 data cache，此时本机 C API 的封装传空地址。

第三参数类型为 `dcci_dst_t` / `DcciDst`，A5 可用编码为 `CACHELINE_ALL=0`、
`CACHELINE_OUT=2`、`CACHELINE_ATOMIC=3`。两参数调用在 A5 SIMT intrinsic 中的默认第三参数是 0，
即与 `CACHELINE_ALL` 同编码。AscendC 公共接口对三种 selector 都统一命名为
`DataCacheCleanAndInvalid`，因此不能把 OUT、ATOMIC 解释为 clean-only / invalidate-only 动作开关。

本机代码没有进一步公开 OUT/ATOMIC 的硬件 entry 分类规则，也不能仅凭名称证明
`ALL == OUT ∪ ATOMIC`。CANN 业务中 OUT 同时用于普通 GM 读取前和 scalar 写入后；ATOMIC 仅找到
一处在普通 `GlobalTensor::SetValue` 后的调用。这些只能说明实际使用方式，三种编码在本机 A5
当前场景中的行为由以下两组精确用例确定。

A5 C API 另外声明了无参 `asc_dci()`，底层调用 `dci()`/`__builtin_cce_dci`。本机头文件没有给出
它的作用域和同步语义，当前也没有精确上板用例，因此只能确认该 primitive 存在，不能把它解释为
“按指定地址失效单条 64B line”，也不能用它替代以下 DCCI 所有权协议。

## clean reader 的 DCCI selector 对照

`ascendc/mb8_dcci_seam.asc` 与 `ccec/dcci_seam.cpp` 固定启动两个 AIV，验证读核从未写过
目标 64B line 时，两参数 DEFAULT、显式 ALL/OUT/ATOMIC 与 no-DCCI 的行为。data、ready、done、writer marker
和 reader result 各自独占 cache line；ready/done 使用单调 atomic phase 严格串行以下时序：

1. 读核用普通 scalar load 预读旧 data，使本核持有 clean cache line，DSB 后 atomic 发布 ready；
2. 写核看到 ready 后用 bypass store 更新整条 data，DSB 后 atomic 发布 done；
3. 读核看到 done 后执行对应 DCCI 和 DSB，再用普通 load 观察本核 cache，用 bypass load 观察 GM；
4. 下一轮 ready 只能在本轮检查结束后发布，因此两个核对 data line 没有并发访问。

no-DCCI mode 只用于证明普通 load 确实持有 stale clean line，不是一种 DCCI 方案。每轮普通读取
只能精确等于“写核本轮新 line”或“本轮 DCCI 前实际预读的完整 line”，拒绝任意历史轮次、torn
或其他值。精确判定标准如下：

| Mode | 精确判定标准 |
|---|---|
| 两参数 DEFAULT | 100 轮普通/bypass 读取均为写核本轮新 line；GM、phase、marker 全精确 |
| 显式 CACHELINE_ALL | 与 DEFAULT 完全相同，用于核对默认编码路径 |
| CACHELINE_OUT | 100 轮普通/bypass 均为本轮完整新 line；`gm_bad=0`；最终 16-word line 等于最后一轮值 |
| CACHELINE_ATOMIC | 100 轮普通/bypass 均为本轮完整新 line；`gm_bad=0`；最终 16-word line 等于最后一轮值 |
| NO DCCI control | 100 轮普通读取均等于本轮实际预读旧 line；bypass 均为写核本轮新 line |

device 0 直接上板结果：AscendC 与 CCEC 完全一致，DEFAULT、ALL、OUT、ATOMIC 均为
`100 fresh / 0 stale / 0 other / 0 gm_bad`，no-DCCI 为
`0 fresh / 100 stale / 0 other / 0 gm_bad`；phase、marker、轮数全部精确。ATOMIC 又在两端各用独立
进程重复 5 次，结果不变。no-DCCI 排除了“目标 line 在 DCCI 前自然失驻”的解释。

该结果直接证明：在本机 A5 的 `SINGLE_CACHE_LINE`、ordinary clean scalar cache entry 上，三个显式
selector 执行后，后续普通 load 都取得了 GM 新值；clean line 上的 clean 动作本身不可由该组观测。
结合下一组 dirty-line 结果，可确认三者也都会发布该场景的 ordinary dirty line。仍不能证明三种
selector 对所有 entry 类别都等价，也不能证明第三参数在其他 scope、地址属性或芯片上会被忽略。

## dirty line、AtomicExch 与 DCCI 的同/分 line 对照

`ascendc/dcci_atomic_clobber.asc` 与 `ccec/dcci_atomic_clobber.cpp` 固定启动两个 AIV，并把时序写死：

1. 核0普通预读整条 data，再 scalar store 邻接 word，使本核持有包含 atomic 目标旧值的 dirty line；
2. 核0 DSB 后通过独立 ready line 交权；核1 `AtomicExch` 目标 word，DSB 后通过独立 done line 交回；
3. 核0看到 done 后先 DSB，再执行 ALL/OUT/ATOMIC DCCI 或 no-DCCI，随后 DSB 并用 bypass load
   逐字保存 data 与独立 atomic line；
4. 每个 mode 独立进程执行，完整检查两条 16-word line、AtomicExch 返回旧值、phase 与两核 marker。

因此该用例没有“两个核同时操作被测 data line”的调度竞态。device 0 上 AscendC 与 CCEC 的八个
mode 逐项一致：

| DCCI | atomic 与 dirty data 同 64B line | atomic 与 dirty data 分 64B line |
|---|---|---|
| CACHELINE_ALL | dirty line 被发布；AtomicExch 新值被旧快照覆盖 | dirty data 被发布；AtomicExch 新值保留 |
| CACHELINE_OUT | dirty line 被发布；AtomicExch 新值被旧快照覆盖 | dirty data 被发布；AtomicExch 新值保留 |
| CACHELINE_ATOMIC | dirty line 被发布；AtomicExch 新值被旧快照覆盖 | dirty data 被发布；AtomicExch 新值保留 |
| NO DCCI | dirty scalar 值未发布；AtomicExch 新值保留 | dirty scalar 值未发布；AtomicExch 新值保留 |

上表记录实际状态，不把问题存在本身当作成功条件。同-line 的三个 selector mode 使用正确性门禁：
AtomicExch 新值必须保留；device 0 当前均因新值被覆盖而返回非零。分-line 与 no-DCCI 五个 mode
全部通过，作为精确 control。两个 runner 都会跑完本 probe 的八个 mode 后再汇总失败。

结论限定为：atomic 指令本身已经成功完成，仍不能阻止另一个核随后把同一 64B line 的 stale dirty
快照通过 DCCI 写回。因此工程与测试都要求 atomic 控制量和 DCCI 数据位于不同 64B line；否则
stale dirty line 的后续 clean 仍可能覆盖
已经成功完成的 atomic 更新。多个纯 atomic 控制字能否共 line 不由本用例外推。

## 同 line `st_dev` 最简对照

`ascendc/st_dev_same_line.asc` 固定启动两个 AIV；`ccec/st_dev_same_line.cpp` 的 mode 0 保留同构双 AIV
路径。每个活跃 AIV 只写自己的 4B slot，使用相同值公式和三组路径：

| 路径 | 类型 | 精确判定标准 |
|---|---|---|
| 不同 slot、同一 64B line、仅 loop-end DSB | regression gating | 4000 次必须全部等于最后一轮值；任一 mismatch 都使测试失败 |
| 每个 AIV 独占 64B line、仅 loop-end DSB | 低压力正确性门禁 | 4000 次全部等于最后一轮值；失败同样暴露问题 |
| 不同 slot、同一 64B line、逐轮 DSB | gating control | 4000 次全部等于最后一轮值 |

CCEC 另增加 mode 1/2，并由 runner 在独立 host 进程中分别执行三个 mode。每个 block 使用独占
64B marker 记录 `get_coreid()`、`get_subblockid()` 和按本机 PTO-ISA A5 `TSYNC_CVID` 公式计算的
`comm_slot`。该值只是软件 CV 通信配对编号，不能解释为硬件物理组。三 AIV mode 中 AIV1 不访问
任何被测数据，只写自己的参与计数和独占 marker，并按相同顺序参加每次 `SyncAll`：

| CCEC mode | 启动与活跃者 | 本轮每次 launch 的记录 | 同-line loop-end DSB | 旧分-line 低压路径 | 同-line 逐轮 DSB |
|---:|---|---|---:|---:|---:|
| 0 | 启动 2；block0+block1 活跃 | block0 `(core18,sub0,comm0)`；block1 `(core72,sub0,comm18)` | 1679/4000 mismatch | 0/4000 | 0/4000 |
| 1 | 启动 3；block0+block2 活跃，block1 data-idle | block0 `(18,0,0)`；block1 `(19,0,0)`；block2 `(72,0,18)` | 1736/4000 mismatch | 0/4000 | 0/4000 |
| 2 | 只启动 block0 | block0 `(18,0,0)` | 同址 0/2000 | 不适用 | 同址逐写 DSB 0/2000 |

表中数字来自当时 device 0 带首错诊断的 20-launch 复跑；mode 0/1 因同-line 正确性门禁分别 exit 1，
mode 2 exit 0。AscendC 原双 AIV 路径为同-line `1792/4000`、两个旧对照均 `0/4000`、exit 1。mode 2
的 `0/2000` 只说明该轮低压力没有命中，不能支持单 AIV 同址安全结论；当前结论以下一节独立压力为准。

原双 AIV/AscendC 与 CCEC mode 0 的旧分-line 路径位于 allocation 内部 line1/line2；CCEC mode 1
的 block0+block2 路径位于 line5/line6。它们都只有 4000 次检查，并嵌在同-line 与逐轮 DSB 路径
之中。在加入首错记录前，CCEC mode 0/1 曾各出现 `1/4000`；随后一轮为 0，而当前同版本 mode 0
复跑又得到 `3/4000`。因此这些 `0/4000` 只能作为历史低压力样本，不能继续支持“问题只限同一
cacheline”或“分-line 必然安全”。分-line 的当前主证据见下一节独立压力用例。

本用例的目标是让同-line 问题以正确性失败显式暴露。问题路径必须保留 loop-end DSB，不能通过改成
逐轮 DSB、替换成分-line 布局，或把 `mismatch > 0` 写成成功条件来让测试通过。逐轮 DSB 只保留为
同-line 对照；分-line 本身由下一节独立压测，不能再当作规避问题的安全修改。

## 单 AIV `st_dev` 独立压力

`ascendc/st_dev_single_core_stress.asc` 与 `ccec/st_dev_single_core_stress.cpp` 是同构用例。每次 kernel
只启动一个 AIV；写者和 bypass/`ld_dev` 读者都是 block0，不调用 `SyncAll`，不存在核间交权、核间
数据共享或多个 writer。data line0..2、result、participation、topology marker、tail guard 各自独占
64B；控制区使用 atomic 写，避免把控制发布本身混入 repeated `st_dev` 数据路径。

七个 mode 将 allocation 内偏移、单/双地址和 DSB 位置显式分开：

| Mode | data 地址 | DSB 位置 | runner 默认执行 |
|---:|---|---|---|
| 0 | line0 单址 | 257 次写后一次 | 否；可用 `ATOMIC_PROBE_MODE=0` 单独执行 |
| 1 | line1 单址 | 257 次写后一次 | 是 |
| 2 | line2 单址 | 257 次写后一次 | 否；可单独执行 |
| 3 | 每轮依次写 line0 / line1 | 257 轮后一次 | 否；可单独执行 |
| 4 | 每轮依次写 line1 / line2 | 257 轮后一次 | 是 |
| 5 | line1 单址 | 每次写后立即执行 | 是，mode 1 的控制 |
| 6 | 每轮依次写 line1 / line2 | 每个 slot 每次写后立即执行 | 是，mode 4 的控制 |

每个 launch 含 100 trial，每个 slot 每 trial 连续写 257 次，然后同一 AIV bypass-read 精确终值。
2026-07-13 device 0 未加设备锁直跑，单址 mode 使用 20000 launch，双址 mode 使用 5000 launch：

| Mode | CCEC mismatch | AscendC mismatch | host 最终快照错误（CCEC / AscendC） |
|---:|---:|---:|---:|
| 1：line1 单址、loop-end DSB | 4/2000000 | 3/2000000 | 0 / 0 |
| 4：line1/line2 双址、loop-end DSB | 78130/1000000 | 83821/1000000 | 784 / 838 |
| 5：line1 单址、逐写 DSB | 0/2000000 | 0/2000000 | 0 / 0 |
| 6：line1/line2 双址、逐写 DSB | 0/1000000 | 0/1000000 | 0 / 0 |

两端 allocation 均为 `mod128=0`、`mod256=0`、`mod512=0`，且 marker 都记录唯一参与者为
`block0(core18,sub0,comm_slot0)`。所有参与计数、result header、marker、非目标 data 和 guard 检查
均精确。mode 1/4 按正确性契约返回非零；mode 5/6 返回 0。mode 4 的 host 最终快照错误也是目标值
不精确，单独计数并参与失败，不能归入 protocol/guard 错误。

因此已证实：多 AIV、跨核同步和跨核数据共享都不是本现象出现的必要条件；同一 AIV 对同一 4B
地址 repeated `st_dev`，只在 257 次写后执行一次 DSB，仍可能读到前一轮值。首错均表现为期望
round 256、实际为 round 255 或更早，但当前用例只证明精确终值回退，没有证明底层机制必然是
编译器重排、硬件 store queue、cache bank/set 或其他具体原因。逐写 DSB 在本轮样本中把错误压到 0，
这是实测控制结果，不是所有芯片、地址、宽度和时序的通用充分性证明。

## 分 line `st_dev` 独立压力

`ascendc/st_dev_separate_line_stress.asc` 与 `ccec/st_dev_separate_line_stress.cpp` 不包含任何同-line
数据路径。每个 mode 由独立 host 进程执行 500 launch；每个 launch 有 100 trial，每个活跃 AIV 对
自己的 4B 地址连续执行 257 次 bypass store，只在循环末执行一次 DSB。两个目标地址始终分属不同
64B line，block0 在 `SyncAll` 后用 bypass load 检查各自最后一次写入值，总计 100,000 次终值检查。

四个 mode 同时区分活跃 block 映射和 allocation 内部 line offset：

| Mode | 活跃 block | 被测 data line | CCEC mismatch | AscendC mismatch |
|---:|---|---|---:|---:|
| 0 | block0 + block1 | line0 / line1 | 41484/100000 | 42165/100000 |
| 1 | block0 + block2；block1 data-idle | line0 / line1 | 39974/100000 | 37320/100000 |
| 2 | block0 + block1 | line1 / line2 | 0/100000 | 46/100000 |
| 3 | block0 + block2；block1 data-idle | line1 / line2 | 10/100000 | 5/100000 |

两端本轮 allocation 首地址均满足 `mod128=0`、`mod256=0`、`mod512=0`。mode 0/2 记录 block0
`(core18,sub0,comm_slot0)`、block1 `(core72,sub0,comm_slot18)`；mode 1/3 另记录 data-idle block1
`(core19,sub0,comm_slot0)`，第二个活跃者 block2 为 `(core72,sub0,comm_slot18)`。所有 protocol、
participation、marker、guard 和 `comm_slot` 公式检查均为 0 failure；data-idle block1 不访问两条
被测 data line。

当前证据足以否定“两个 AIV 写不同 cacheline 就必然安全”：line0/line1 在两种前端均高频复现，
line1/line2 也在三组运行中低频复现。CCEC mode 2 的单次 `0/100000` 不能被解释为安全保证。
allocation 内部 line offset 与本轮失败频率呈显著相关，但测试没有证明 cache bank、set、store
队列或其他底层机制，不能据此指定根因；CCEC 与 AscendC 的具体失败 slot 也不一致，不能扩大为某个
固定 writer 必然失败。该分-line 用例自身只覆盖多个 AIV、repeated bypass store、loop-end DSB 和
随后跨 AIV 会合，不能单独用于判断单 AIV；上节独立单核用例已另行证明跨 AIV 不是问题复现的必要条件。

旧双 AIV/AscendC 与 CCEC mode 0 路径使用本轮低频的 line1/line2 布局，且只有 4000 次检查；CCEC
mode 1 另用 line5/line6，同样只有 4000 次。旧路径还与同-line、逐轮 DSB 路径共处一个 kernel
时序。独立用例只保留分-line 数据路径，并把样本提高到 100,000 次；旧 CCEC mode 0 随后也复现了
`3/4000`。这些数据足以解释为什么 4000 次旧样本经常为 0，也证明新用例没有引入同-line 数据交互；
line5/line6 尚未由新独立用例同压复测，且两者完整指令序列并非逐条相同，不能把频率差异进一步
归因到某个底层机制。

## 同 line `AtomicExch` 对照

`ascendc/atomic_exch_same_line.asc` 与 `ccec/atomic_exch_same_line.cpp` 完全复用上述两个 AIV、
4B slot、20 launch × 100 trial × 257 round、三组布局和同步点，仅把测试数据写替换为
`AtomicExch<uint32_t>` / `atomicExch`。选择 exchange 是为了保留任意轮次值；`AtomicMax` 与
`AtomicAdd` 的终值会天然掩盖执行顺序。

device 0 实测：CCEC 与 AscendC 的同-line loop-end DSB、分-line loop-end DSB、同-line 逐轮 DSB
均为 `0/4000` mismatch，参与计数与 marker 精确，两个用例均 exit 0。当前证据说明同构压力下
AtomicExch 没有复现 st_dev 的末值回退。该判定只检查每个 trial 的最终值，不证明中间 AtomicExch
绝无重排；本用例仍是两个核写同一 cacheline 的不同 4B slot，不覆盖两个核写同一个 4B 地址，也不能
外推到其他 atomic 类型或其他内存序场景。

## `ld_dev` 多读者 fanout 发布对照

`ccec/ld_dev_fanout_publish.cpp` 固定 block0 为唯一 writer，其余所有实际启动的 AIV 都是 reader，
每个 reader 从始至终只用 raw `ld_dev` 读取同一个 data word。data、host-only launch config、
control epoch、ready、ack、timing 和每核结果分别从独立 64B cacheline 开始；data line 不含任何
atomic 控制字，全用例不执行 DCCI。runner 默认启动 72 个 AIV，也可用
`ATOMIC_PROBE_AIVS=2..72` 缩小并发数；host 会逐核检查
marker，并要求每个 block 的 `(get_coreid(), get_subblockid())` 二元组唯一，不能把请求 block 数当作
实际多核参与证明。

三个 mode 的精确 writer 序列为：

| Mode | writer 发布序列 | 单指令 timing | 发布 timing |
|---:|---|---|---|
| 0 | `volatile ordinary scalar GM store -> DSB`；明确无 DCCI | 只包围 ordinary store | 包含其后的 DSB |
| 1 | `st_dev -> DSB` | 只包围 `st_dev` | 包含其后的 DSB |
| 2 | `AtomicExch`；不额外补 DSB | 包围 AtomicExch | 与单指令 timing 相同 |

每个 kernel 执行 64 轮。每轮 reader 先在独立 ready line 上 atomic 加一，再用 `ld_dev` 等待 writer
通过独立 control line 发布 epoch；writer 确认全部 reader ready 后才写本轮唯一序列值。reader
只有精确看到该值或设备侧 20ms 有限超时后才 atomic ack；writer 等到全部 ack 后才进入下一轮。
因此正常路径不允许 writer 跳过中间值，判定的是“每个 reader 逐轮看到完整序列”，不是只在 kernel
结束时碰巧读到最终值。失败 reader 仍会 ack，所有控制与 data 轮询都有 `get_sys_cnt()` 超时，错误
mode 必须返回首错和 timeout，不能死循环。

用例同时记录三类 A5 sys-counter 指标：单条 writer 写指令周期、包含必要 DSB 的 writer 发布序列
周期、从 write start 到全部 reader ack 的端到端周期。host 输出 min/p50/p95/max/mean；单指令数据
包含两次计数器读取的固定开销，跨 mode 比较必须使用同一并发数和独占设备。计时 record、每核结果
均在被测轮次结束后用 atomic 写到非 data line，不参与本轮端到端时间。

三个 mode 都使用同一正确性门禁：所有 `reader * 64` 次读取都必须精确，control 不得 timeout，
writer 每轮 `ld_dev` 快照与 kernel 返回后的最终 GM 值也必须精确。ordinary mode 不会为了让门禁通过
而补 DCCI；它另用 writer 本核 ordinary load 检查最终 store，区分“写者本核 store 已执行但没有
发布到 GM”和“store 根本没执行”。若 ordinary dirty line 对 `ld_dev` 不可见，就应以非零退出码和
明确 timeout 暴露。

device 0 独占执行的 24 AIV 受控结果如下；每项包含 3 个独立 kernel launch、每次 64 轮，即每个 mode
共检查 `23 * 64 * 3 = 4416` 个 reader/round 观察点：

| Writer mode | reader 精确看到本轮值 | writer 单指令 mean | 完整发布 mean | write-start 到全部 ack mean |
|---|---:|---:|---:|---:|
| ordinary+DSB，无 DCCI | `0/4416` | 16.520 us | 16.530 us | 20018.053 us（timeout 路径） |
| st_dev+DSB | `4416/4416` | 5.525 us | 5.707 us | 7.193 us |
| AtomicExch | `4416/4416` | 15.597 us | 15.597 us | 17.478 us |

ordinary writer 的本核 ordinary load 始终看到最终 store，但 writer 的 `ld_dev`、所有 reader 的
`ld_dev` 和 kernel 返回后的 GM 均仍为 0；因此该路径精确暴露“写入本核 scalar cache、DSB 完成，
但没有发布到 bypass reader 所见 GM”。st_dev 的本用例结果只覆盖每轮一次写且逐轮 DSB 的发布序列，
不能覆盖或推翻重复 st_dev 压力用例已经记录的末值回退。

72 AIV 独占压力确认 72 个 block 的 `(core, subblock)` 均唯一，但三种 mode 的独立 control line 也
分别出现 `5263/13632`、`10072/13632`、`5967/13632` 次 timeout；data 精确观察分别为
`0/13632`、`2986/13632`、`5921/13632`。把 watchdog 从 2ms 提高到 20ms 后，写指令或发布序列的
长尾也从约 2ms 移到约 20ms，仍未恢复 control。现有证据只说明 36+ AIV（35+ 个无退避
`ld_dev` reader）的持续读压力会造成与 watchdog 边界绑定的进展失败；在 control 已失败时，三种
data 结果不是隔离的写入语义对照，不能据此声称 st_dev 或 AtomicExch 本身在 72 AIV 下丢写。默认
72 AIV 保留这一高压力
看护；需要比较三种 writer 的可见性与 timing 时，使用已验证的 `ATOMIC_PROBE_AIVS=24` 受控配置。

```bash
ATOMIC_PROBE_AIVS=24 ATOMIC_PROBE_FANOUT_LAUNCHES=3 \
  tests/atomic_probe/ccec/run_all.sh ld_dev_fanout_publish
```

## PMU 对 atomic 与 I-cache miss 等待周期的精确归类

`ccec/atomic_scalar_pmu.cpp` 与 `ccec/icache_scalar_pmu.cpp` 是两个独立单 AIV 微基准。
它们共用 `pmu_probe_control.h` / `pmu_probe_aicpu.cpp` 的 108 physical sub-core
MMIO 表与 PMU 所有权协议；I-cache host 另用 `pmu_probe_host_support.h` 封装同一协议。
AICPU helper 会保存并读回核验
CTRL、slot 0/1/2 selector 和 START/STOP range，配置：

| PMU 计数 | 事件 | 用途 |
|---|---:|---|
| slot 0 | `0x1` | `scalar_instr_busy` |
| slot 1 | `0x34` | `icache_req` |
| slot 2 | `0x35` | `icache_miss` |
| total | 固定总周期计数器 | PMU gate 内的总 AICore cycle |

每个用例在待测段前后只执行 `metrics_prof_start/stop`，关窗后才用
`ld_dev` 读取 total/scalar/request/miss，最后恢复进程进入用例前的 PMU 配置。
它们不依赖 `msprof task-based` 的整任务 context 计数，也不与另一个 PMU session
并发执行。

### dependent atomic 等待计入 scalar busy

Atomic 用例对每个 rounds 依次执行：

1. `EMPTY`：只量 gate/read 固定开销；
2. `SCALAR_CONTROL`：使用 scalar 寄存器执行与 atomic 路径同构的
   `old/delta/checksum` 递推；
3. `DEPENDENT_ATOMIC_ADD`：`old` 改由 64-bit `atomicAdd` 返回，下一次 addend
   依赖上一次返回值，强制测量 atomic 完成延迟而不是无依赖吞吐。

Host 对 CONTROL/ATOMIC 复算完全相同的 checksum，并检查 atomic 终值、
physical core id 和 `CTRL.bit0 == 0`。2026-07-18 在 A5 device 0 上的三个独立
进程会话均执行 `8192 次 × 7 组`：

| 会话 | `(ATOMIC-CONTROL)` 完成延迟 | PMU total cycle/op | scalar busy cycle/op | scalar / total |
|---:|---:|---:|---:|---:|
| 1 | 182.922729 ns | 301.804321 | 301.803955 | 0.999998787 |
| 2 | 251.687622 ns | 415.253540 | 415.253174 | 0.999999119 |
| 3 | 271.009888 ns | 447.110718 | 447.110352 | 0.999999181 |

完成延迟在三个会话中处于不同档位，当前证据不足以归因；但计数归类完全一致：
**dependent `atomicAdd` 增加的 PMU total 周期几乎 100% 同步计入
`scalar_instr_busy(0x1)`。** `get_sys_cnt` 是 1 GHz 时基，表中 PMU total/scalar
则是 AICore 核时钟 cycle，两者不能直接按同一单位比较。

### I-cache miss 回填等待的绝大多数周期不计入 scalar busy

I-cache 用例的 WARM/COLD 两条路径在 PMU 开窗前汇合，窗口内只从同一个
静态调用点执行一次同一个 target：

- WARM 在窗外先调用一次 target；
- COLD 在窗外先执行超过 16 KiB AIV scalar I-cache 容量的 evictor；
- 最终 ELF 硬校验 target 为 `8280 B @ 0x0`、evictor 为 `32836 B @ 0x2080`，
  两者均按 128 B 对齐且区间不重叠；
- Host 逐样本复算 target/prepare checksum，并要求每对 WARM/COLD 使用同一
  physical AIV、`COLD miss > WARM miss`。

2026-07-18 在 A5 device 0 执行三个独立进程会话，每个会话 11 对，
交替使用 WARM,COLD 和 COLD,WARM 顺序。全部 33 对均为
`WARM miss=0`、`COLD miss=68`，且 checksum、mode echo、PMU 关窗与恢复全部通过：

| 会话 | WARM `total/scalar/req/miss` | COLD `total/scalar/req/miss` | total 增量 | scalar 增量 | miss 增量 | scalar / total 增量 |
|---:|---|---|---:|---:|---:|---:|
| 1 | `1068/1060/520/0` | `3377/1108/588/68` | 2309 | 48 | 68 | 2.078822% |
| 2 | `1068/1060/520/0` | `3379/1108/588/68` | 2311 | 48 | 68 | 2.077023% |
| 3 | `1068/1060/520/0` | `3380/1108/588/68` | 2312 | 48 | 68 | 2.076125% |

因此准确结论是：**本场景中 I-cache miss 回填等待的绝大多数周期不计入
scalar busy，但不是 scalar 增量严格为零。** 额外 68 次 miss 产生 `2309..2312`
total cycle，scalar busy 只增加 48 cycle，其余 `2261..2264` cycle 形成
scalar-busy gap。target 实际覆盖 65 条 cache line，另 3 次 miss 与顺序预取相符；
因此 `total_delta / 68` 只能作为本窗口归一化值，不能称为单次阻塞
I-cache miss 的精确延迟。

两个用例均只使用本机 CANN/PTO-ISA，不下载外部 PTO-ISA：

```bash
source /home/q00473782/cann/cann-9.1.0/set_env.sh
cd tests/atomic_probe/ccec
./run_atomic_scalar_pmu.sh
./run_icache_scalar_pmu.sh
```

## 其余探针

| 文件 | 类型 | 验证内容 |
|---|---|---|
| `ascendc/atomic.asc` / `ccec/atomic_cas_probe.cpp` | gating | CAS 最终值 2000，且全局恰好一次成功 |
| `ascendc/atomic64_verify.asc` | gating | 32/64-bit add/max 精确终值 |
| `ascendc/cacheline_blast.asc` / `ccec/atomic_blast.cpp` | gating + observation | atomic 目标确实改变且邻接字节不变；dcci 反向覆盖单列观察 |
| `ascendc/bypass_dcache_probe.asc` / `ccec/bypass_dcache_ccec.cpp` | gating | 1/2/4/8B ld_dev 共 120 次精确读取、st_dev/atomic、publish/observe |
| `ascendc/concurrent_cacheline.asc` | gating + observation | 多 block st_dev、producer/consumer、持续读；store+dcci race 单列观察 |
| `ascendc/cacheline_stress.asc` / `ccec/concurrent_stress.cpp` | observation + control | tight-loop dcci hazard；CCEC st_dev control 精确终值 |
| `ascendc/st_dev_same_line.asc` / `ccec/st_dev_same_line.cpp` | regression gating + comparison | 多 AIV 同 line 使用精确终值断言；CCEC 另覆盖 block0+block2、raw core/subblock/`comm_slot` 与低压力单 AIV 同址历史样本；原分-line 路径同样只作历史对照 |
| `ascendc/st_dev_single_core_stress.asc` / `ccec/st_dev_single_core_stress.cpp` | regression gating + control | 只启动一个 AIV；覆盖三个 allocation line 偏移、单/双地址 loop-end DSB，以及同址逐写 DSB 控制 |
| `ascendc/st_dev_separate_line_stress.asc` / `ccec/st_dev_separate_line_stress.cpp` | regression gating | 只含分-line 数据路径；四模式覆盖两组活跃 block 与两种 allocation 内 line offset，100000 次精确终值检查 |
| `ascendc/atomic_exch_same_line.asc` / `ccec/atomic_exch_same_line.cpp` | gating + control | 与 st_dev 同构的 AtomicExch 末值顺序对照；三组路径均精确通过 |
| `ccec/ld_dev_fanout_publish.cpp` | regression gating + timing | 唯一 writer 以 ordinary+DSB、st_dev+DSB、AtomicExch 三种方式逐轮发布；其余全部 AIV 只用 ld_dev 读取完整序列，并记录 writer/全读者周期 |
| `ccec/atomic_scalar_pmu.cpp` | gating + PMU classification | 单 AIV dependent atomicAdd 完成延迟与同构 scalar control 对照；核实 atomic 等待是否计入 scalar busy |
| `ccec/icache_scalar_pmu.cpp` | gating + PMU classification | 单 AIV 同一 target 的 WARM/COLD I-cache 对照；核实 miss 回填等待是否计入 scalar busy |
| `ascendc/dcci_atomic_stress.asc` | legacy observation | 旧的混合 stress；不再作为 DCCI selector 语义证据 |
| `ccec/dcci_clean_clobber.cpp` | gating | 有序 dirty/clean line 的 dcci clobber 与 control |
| `ascendc/mb2_flags_clobber.asc` | gating + observation | AtomicMax flags 无丢失；store+dcci 仅统计 |
| `ascendc/mb8_dcci_seam.asc` / `ccec/dcci_seam.cpp` | gating | clean reader 的 DEFAULT/ALL/OUT/ATOMIC/no-DCCI 五模式精确对照 |
| `ascendc/dcci_atomic_clobber.asc` / `ccec/dcci_atomic_clobber.cpp` | regression gating + control | 同-line 三 selector 当前明确失败；分-line 与 no-DCCI 五模式精确通过 |
| `pa_scheduler/ccec/kernel.cpp` | calibration | cold/warm 同核配对；每个 cold trial 严格增加一个 CNT7 I-cache miss，建立 scalar 时间标尺 |
| `cpu/cpu_atomicity.cpp` | gating + observation | coherent CPU 同/异 cacheline 同构 control、atomic、snapshot、spinlock |

### PA I-cache 单 miss 实测数据

2026-07-18 在 device 0、32 AIC + 64 AIV 并发、`msprof PipeUtilization` 下，
`icache-single` 得到以下结果。时间列为多轮 `ns/miss` 中位数，括号内是最小值～最大值：

| 配置 | 每轮 cold/warm CNT7 miss（ALL） | 严格门禁 | ALL | AIC | AIV |
|---|---:|---:|---:|---:|---:|
| 64 trials/core × 10 | 6,144 / 0（2,048 AIC + 4,096 AIV） | 10/10 PASS | 86.596（86.532～86.792） | 85.913（85.848～86.202） | 86.938（86.861～87.086） |
| 128 trials/core × 5 | 12,288 / 0（4,096 AIC + 8,192 AIV） | 5/5 PASS | 89.629（89.615～89.648） | 92.100（91.984～92.267） | 88.410（88.310～88.440） |

两组每轮均为 `calibrated_cores=96/96`，并通过 “each cold trial adds exactly
one CNT7 I-cache miss” 断言。AIC/AIV 差值只有数 ns 且方向随运行时段变化，
因此不建立两个伪精确常数。原始日志为
[`64×10`](pa_scheduler/outputs/pmu_validation/icache_single_64x10_20260718_085929_3232836_console.log)
和
[`128×5`](pa_scheduler/outputs/pmu_validation/icache_single_128x5_20260718_090151_3235468_console.log)。

PA scalar 分析只需要数量级时，使用 `T_icache_est_ns = CNT7_miss_total * 90`；例如
1,000 个 I-cache miss 约为 90 us。compulsory、capacity、conflict miss 都包含在
`CNT7_miss_total` 内。该乘积是 cold/warm 校准得到的一阶等效时间，不是逐次精确
可加的 stall；方法、角色分项和原始日志见
[`PA调度器独立复现与泳道使用指南.md`](pa_scheduler/PA调度器独立复现与泳道使用指南.md#单次-cnt7-i-cache-miss-的-scalar-估算标尺)。

## 判定标准与退出码规则

1. **确定性安全契约必须 gating**：目标值、邻居值、参与核数、执行 marker 全部精确匹配；任一失败返回非零。
2. **风险状态若有严格 phase ordering，可以精确分类和记录**；PASS/FAIL 仍由该用例的安全契约决定。
   例如 dirty line 在 atomic 更新后执行 DCCI，可以精确识别旧快照覆盖，但要求 atomic 新值保留的
   regression mode 仍必须判失败。
3. **一般无 ordering 竞态只 observational**：输出 survived/clobbered 分布，不硬编码某次调度结果。
   但 `st_dev_same_line` 验证的是各核独占 slot 的正确性契约，不是把已知错误当成功条件的
   characterization；它要求 mismatch 为 0，同时打印实际 mismatch 数量用于复现与诊断。
4. ACL、编译、link、timeout、kernel sync 或 semantic assertion 失败均向 runner 传播非零退出码。
5. 结果槽用 bypass store 发布，并在 host 消费前显式完成；CCEC 不依赖 scalar 自动 dcci。

原 stress 判定条件已从 `actual >= base` 改为 `actual == base + last_round`。byte-width 探针同时检查原子或
st_dev 的目标字节确实改变，防止 no-op kernel 因“邻居没坏”而假通过。

## CCEC 编译约束

本机 `ccec -mllvm -print-all-options` 显示：

```text
cce-aicore-dcci-insert-for-scalar = 1 (default: 1)
cce-aicore-dcci-before-kernel-end = 1 (default: 1)
```

所有 CCEC runner 显式设置：

```text
-mllvm -cce-aicore-dcci-insert-for-scalar=false
-mllvm -cce-aicore-dcci-before-kernel-end=false
```

普通 scalar store + 显式 dcci 只保留在专门制造 cache-line hazard 的 mode；需要绕过 DCache 的路径
与结果发布使用 `st_dev`。`st_dev` 本身不提供跨核终值正确性保证，repeated/multi-AIV 路径仍需单独
设置 DSB、同步和精确判定。这两个选项关闭的是编译器插入，不改变显式 dcci。

## 执行入口

### CPU control

本机已有 PTO-ISA 可直接复用，无需下载：

```bash
export PTO_ISA_ROOT=/home/q00473782/atomic/codex/simpler-fully_distributed/build/pto-isa
export PYTHONPATH="$PWD/python:$PWD"
.venv/bin/python -m pytest tests/atomic_probe/test_atomic_probe.py -k cpu -q --clone-protocol https
```

### Build-only

```bash
bisheng -xasc tests/atomic_probe/ascendc/cacheline_matrix.asc \
  --npu-arch=dav-3510 -DPROBE_CORE_VARIANT=0 -o /tmp/cacheline_matrix_aiv
tests/atomic_probe/ccec/run_all.sh build
```

### A5 onboard

CI/共享环境优先由 `task-submit` 独占设备：

```bash
.claude/skills/onboard-arch-precheck/check.sh a5 || exit 1
command -v task-submit >/dev/null || exit 1

task-submit --timeout 1800 --max-time 1800 --device auto --device-num 1 \
  --run "cd $PWD && \
    PYTHONPATH=$PWD/python:$PWD \
    PTO_ISA_ROOT=$PTO_ISA_ROOT \
    .venv/bin/python -m pytest tests/atomic_probe/test_atomic_probe.py \
      -m requires_hardware --platform a5 --device \$TASK_DEVICE -v --clone-protocol https"
```

`ascendc/_run_asc_probe.sh` 和 `ccec/run_all.sh` 都输出 UTC 时间、git SHA、CANN 路径、编译器版本、
device 与 timeout。保存原始证据时直接对上面的 `task-submit` 命令做 `2>&1 | tee <log>`，并保留退出码。
经用户明确授权直接占用设备时，也可设置 `ATOMIC_PROBE_DEVICE=<id>` 后运行这两个 runner；日志必须注明
未经过 `task-submit`，并记录 dirty worktree，不能只记录 base SHA。

## 充分性判定

套件只有同时满足以下条件，才能称为“足够”：

- AscendC 与 CCEC 的 AIV-only 参与计数和 marker 均符合预期；
- 1/2/4/8B × same/separate-line 全部精确通过；
- 同-line regression、dedicated repeated-`st_dev` separate-line 压力与逐轮 DSB control 明确分离；
- 已通过路径、正确性失败路径和无序观察项三类不混淆；
- CPU control 无 C++ data race，并验证同 line 不影响正确性；
- 全量 runner 不漏文件、不吞错误；
- 新提交对应的 A5 原始日志与环境元数据可追溯。

当前工作区尚未通过上述充分性判定：AscendC/CCEC 的同-line regression，以及 dedicated
separate-line repeated-`st_dev` 的多个 mode，均在 device 0 复现 mismatch 并按正确性契约返回
非零；这正是测试要暴露的问题。AtomicExch、逐轮 DSB、DCCI 特定分-line/no-DCCI 模式、既有矩阵与
CPU control 的已执行结果仍按各自场景记录，不能拿来覆盖 repeated-`st_dev` 的失败。结论只覆盖本
文件 goal；AIC/MIX 比例和未测试的数据宽度/拓扑不能由此外推。
