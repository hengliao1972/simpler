<!-- markdownlint-disable MD060 -->

# A5 AICPU 与 AICore atomic/cache 单点能力验证

## 1. 目的与结论

本目录不运行 PA 调度器，只保留 AICPU、AIV Scalar、GM、atomic 和 cache
maintenance 的最小交互，用于回答长期控制面最容易混淆的几个问题：

1. AICPU 写 control 后，AICore Scalar 用什么方式才能稳定看到新值；
2. AICPU 写 ordinary payload 后，AICore 已缓存旧 line 时如何取得新值；
3. AICore 写 control/payload 后，AICPU 是否需要 `dc civac`；
4. AICore 连续发布 dirty line 时，DCCI 与随后 atomic control 之间是否必须显式 DSB；
5. AICore 的 `atomicAdd/Max/CAS/Exch` 与 ordinary load/store、`ld_dev/st_dev`
   在这条跨域路径上的实际区别；
6. 哪些结果是可以作为当前工程协议的正确性门禁，哪些只是当前 A5 的观察值。

当前 device 0 五个独立进程、每项 4096 轮、同一地址反复复用的结果为：

- 新的正式 AICPU → AICore Path-A 协议全部 `20480 fresh / 0 stale / 0 other`：
  AICPU ordinary-store payload 后，直接以 `__atomic_store_n(RELEASE)` 发布独占
  control；中间没有 `dc cvac`、显式 DMB/DSB、ISB 或独立 clean-done。AICore 用
  返回型 atomic 观察 control，并对 ordinary payload 执行单 line DCCI + DSB；
- AICPU ordinary control 无 clean、无 barrier、control 自己充当 doorbell 的更弱
  对照也是 `20480/0/0`，但只作为当前硬件观察；工程协议仍使用 release control，
  不依赖普通 store 的跨域顺序；
- AICore 已缓存 control 旧值时，AICore ordinary load 为
  `0 fresh / 20480 stale / 0 other`，同轮返回型 atomic 为 `20480 fresh`；
- AICore 已缓存 payload 旧值且不执行 DCCI 时，ordinary load 为
  `0/20480/0`，同轮 `ld_dev` 为 `20480 fresh`；producer 侧不再需要 cvac，
  但 consumer 侧 DCCI 仍然不可删除；
- AICore → AICPU 的 32-line direct 强化门禁中，default DCCI 和
  `CACHELINE_OUT` 在**不显式执行 DSB、随后直接 `atomicExch` 发布 control**时均为
  `20480/0/0`，与带一个尾随 DSB 的对照完全一致；当前 A5 这条精确 atomic 发布
  路径不要求额外 DSB；
- 只有 DSB、没有 DCCI 的 32-line 负对照无论是否执行 DSB 都为
  `0 fresh / 20480 stale / 0 other`；DSB 本身不会把 Scalar dirty line 发布出去；
- AICore 不执行 DCCI 时，AICPU 普通读取和随后 `dc civac + ordinary load` 都是
  `0/20480/0`；consumer invalidate 不能补救 producer 未发布的 dirty line。

“当前精确 atomic 发布路径不要求额外 DSB”不是“DCCI 天生隐含完成屏障”。本机公开
头文件只说明 `DSB_ALL` 等待 memory access，没有找到 DCCI 返回即完成或任意后续指令
都能替代 DSB 的契约。因此非 `DCCI lines -> atomicExch(control)` 路径继续保留 DSB；
换芯片、CANN、atomic primitive 或发布拓扑也必须同构复测。

该结论限定于本目录与 `cross_core_aicpu_plan` 共用的 main `aicpu_scheduler`
Path-A。仓库记录过 cust AICPU 子进程被绑到另一 cluster 后不在 AICore snoop domain
的历史故障，不能把这里的结果外推给不同 AICPU launch/affinity 路径。

还必须与 Host/SDMA → AICPU 场景分开：本探针验证的是 **AICPU ↔ AICore
Scalar**，不验证 DMA writer 对 AICPU cache 的一致性。Host DMA 写 GM 后 AICPU 读
仍需 `dc civac`；SDMA 在得到专项硬件结论前也继续按非一致路径处理。这里删的是当前
Path-A 中 AICPU producer 面向 Scalar 的 `dc cvac`，不是删除所有 AICPU cache 操作。

## 2. 与 `cross_core_aicpu_plan` 的关系

正式实现位于
`tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan`。原有
`common/protocol_probe` 已经验证完整 PlanCell：

```text
AICPU ordinary payload store
  -> exact dc cvac range
  -> dsb sy
  -> ordinary control store
  -> dc cvac control
  -> dsb sy / isb

AICore return-ready atomic control observe
  -> exact payload DCCI range
  -> DSB
  -> second atomic control observe
  -> payload decode
```

该协议是此前的保守基线，但把 producer clean、consumer invalidate、atomic observe
和布局隔离同时打开，无法单独回答 `dc cvac` 是否必要。本目录复用仓库正式的
`src/common/aicpu_loader` Path-A 启动实现，独立拆分每个变量；不复制 PA 调度器，
也不依赖 PA task 数量、TensorMap 或算子负载。新增 direct case 已验证可将发布收缩为
`ordinary payload stores -> release atomic control`，不再执行 AICPU `dc cvac`。

## 3. 测试结构

### 3.1 真实参与者

- AICPU：HCC 编译的 `simpler_aicpu_exec`，通过仓库通用 dispatcher 和
  `rtsLaunchCpuKernel` 启动；
- AICore：两个独立的 `dav-c310-vec` CCEC AIV kernel；只执行 Scalar 代码；
- Host：只负责同一块 GM 的初始化、双 stream 并发 launch、同步和最终 D2H
  精确校验，不参与轮内握手。

### 3.2 布局

每个 case 固定预留以下 4736 B，连续 4096 轮复用同一地址：

| 区域 | 大小 | 用途 |
|---|---:|---|
| `ready` | 128 B | harness 交权；与被测 control 分离 |
| `tested_control` | 128 B | 被测 ordinary/atomic/ld_dev 控制字 |
| `done` | 128 B | 旧基线用作 AICore doorbell；direct case 只允许 AICPU 在 primary 读取后写 ACK |
| `payload[32]` | 4096 B | 每项 128 B；前 64 B 为 8 个精确校验 word，后 64 B 为隔离 padding |
| `result` | 256 B | fresh/stale/other、首值、轮数和 atomic 返回值诊断 |

所有 atomic control 都独占 128 B；被 DCCI 的 payload 与 atomic control 不共
64 B cache line。case 0～13 只使用 `payload[0]`，DSB direct case 使用全部 32 项；
一轮只有 32 条 data cacheline 必须同时全 fresh 才记为一次 fresh。测试不借同-line
false sharing 制造假故障。

### 3.3 每轮判定

读者先 ordinary-load 当前 control/payload，使旧 line 驻留本地 cache，然后才发布
`ready`。direct case 中 writer 写 payload 后直接发布被测 control，读者在该 control
上做返回型 atomic polling；没有额外的 producer `done` doorbell。no-DSB direct case
每 64 轮还增加一次 post-primary ACK：AICPU 必须先保存全部 primary payload，之后才
release-store `done`；Scalar 发布 control 后先执行 1048576 个纯 NOP，再以本轮第一条
后续 GM/atomic 指令检查 ACK。最后一轮读完后，AICore 还会发布 `rounds+1` ACK，AICPU
收到 ACK 后才允许执行结果 clean/退出，排除末轮隐藏 clean。
每次结果严格归入：

- `fresh`：本 case 使用的每条 line 都等于本轮新 generation；
- `stale`：本 case 使用的每条 line 都等于本轮写入前实际预读值；
- `other`：存在混合 line，或任一 line 既不是完整新值也不是完整预读值，包括历史代、
  torn 或非法值。

`fresh/stale/other` 的和必须精确等于 4096。正式协议 case 还要求分布与预期完全
一致；`observe` case 只要求轮数、状态、返回值和分类守恒，不把某次可见性变成
契约。

## 4. AICPU → AICore 矩阵

每个 case 的 control primary 是被测读取；control reference 固定为
`atomicAdd(ptr, 0)`。payload primary 是被测读取；payload reference 固定为
`ld_dev`。五轮累计结果如下。

| ID | producer / consumer 差异 | control primary | payload primary | reference | 口径 |
|---:|---|---:|---:|---:|---|
| 0 | control/payload clean；`atomicAdd(0)` + default DCCI | 20480/0/0 | 20480/0/0 | 全 fresh | 旧保守基线 |
| 1 | control 改为恒等 `atomicMax(INT64_MIN)` | 20480/0/0 | 20480/0/0 | 全 fresh | atomic 对照 |
| 2 | control 改为恒等 `atomicCAS(new,new)` | 20480/0/0 | 20480/0/0 | 全 fresh | atomic 对照 |
| 3 | control/payload 都用一次 `ld_dev` | 20480/0/0 | 20480/0/0 | 全 fresh | bypass 读取对照；不做自旋 |
| 4 | control 用已缓存的 ordinary load | 0/20480/0 | 20480/0/0 | control atomic reference fresh | 必须 stale 的负例 |
| 5 | payload 不 DCCI，直接 ordinary load | 20480/0/0 | 0/20480/0 | payload `ld_dev` reference fresh | 必须 stale 的负例 |
| 6 | payload 显式 `CACHELINE_ALL` | 20480/0/0 | 20480/0/0 | 全 fresh | DCCI selector |
| 7 | payload `CACHELINE_OUT` | 20480/0/0 | 20480/0/0 | 全 fresh | DCCI selector |
| 8 | payload `CACHELINE_ATOMIC` | 20480/0/0 | 20480/0/0 | 全 fresh | DCCI selector |
| 9 | AICPU clean + DSB，不执行 ISB | 20480/0/0 | 20480/0/0 | 全 fresh | ISB 功能对照 |
| 10 | control release store 不 cvac；payload clean；独立 clean-done | 20480/0/0 | 20480/0/0 | 全 fresh | 旧 observe；存在独立 done |
| 11 | control exchange 不 cvac；payload clean；独立 clean-done | 20480/0/0 | 20480/0/0 | 全 fresh | 旧 observe；存在独立 done |
| 12 | ordinary control 不 clean；payload clean；独立 clean-done | 20480/0/0 | 20480/0/0 | 全 fresh | 旧 observe；存在独立 done |
| 13 | payload 不 clean；control/done clean | 20480/0/0 | 20480/0/0 | 全 fresh | 旧 observe；存在 clean control/done |
| 14 | control fetch-add 不 cvac；payload clean；独立 clean-done | 20480/0/0 | 20480/0/0 | 全 fresh | 旧 observe；存在独立 done |
| 15 | control compare-exchange 不 cvac；payload clean；独立 clean-done | 20480/0/0 | 20480/0/0 | 全 fresh | 旧 observe；存在独立 done |
| 16 | ordinary control 不 clean；AICore `ld_dev`；独立 clean-done | 20480/0/0 | 20480/0/0 | 全 fresh | 旧 observe；存在独立 done |
| 17 | payload 不 clean；AICore `ld_dev`；control/done clean | 20480/0/0 | 20480/0/0 | 全 fresh | 旧 observe；存在 clean control/done |
| 18 | payload 后 ordinary control；均不 clean/barrier；control 即 doorbell | 20480/0/0 | 20480/0/0 | 全 fresh | 仅 A5 观察；无顺序契约 |
| 19 | payload ordinary store 后 release control；均不 clean；control 即 doorbell | 20480/0/0 | 20480/0/0 | 全 fresh | **当前 Path-A 正式门禁** |
| 20 | payload 后 seq_cst exchange control；均不 clean；control 即 doorbell | 20480/0/0 | 20480/0/0 | 全 fresh | RMW 对照 |
| 21 | payload 后 seq_cst fetch-add control；均不 clean；control 即 doorbell | 20480/0/0 | 20480/0/0 | 全 fresh | RMW 对照 |
| 22 | payload 后 seq_cst CAS control；均不 clean；control 即 doorbell | 20480/0/0 | 20480/0/0 | 全 fresh | RMW 对照 |
| 23 | ordinary payload/control 后独立 release atomic doorbell；均不 clean | 20480/0/0 | 20480/0/0 | 全 fresh | **release 排序门禁** |
| 24 | ordinary payload/control 后独立 relaxed atomic doorbell；均不 clean | 20480/0/0 | 20480/0/0 | 全 fresh | 仅观察 |
| 25 | ordinary payload/control 后 DMB + relaxed doorbell；均不 clean | 20480/0/0 | 20480/0/0 | 全 fresh | DMB 对照 |
| 26 | ordinary payload/control 后 DSB + relaxed doorbell；均不 clean | 20480/0/0 | 20480/0/0 | 全 fresh | DSB 对照 |
| 27 | 与 19 相同，但 payload 不 DCCI | 20480/0/0 | 0/20480/0 | payload `ld_dev` reference fresh | 必须 stale 的负例 |
| 28 | 与 19 相同，但 payload 用 `ld_dev` | 20480/0/0 | 20480/0/0 | 全 fresh | bypass 读取对照 |
| 29 | ordinary control + release doorbell；AICore 读 cached control | 0/20480/0 | 20480/0/0 | control atomic reference fresh | 必须 stale 的负例 |
| 30 | payload 后 ordinary control，之前 DMB；不 clean；control 即 doorbell | 20480/0/0 | 20480/0/0 | 全 fresh | 仅观察 |
| 31 | payload 后 ordinary control，之前 DSB；不 clean；control 即 doorbell | 20480/0/0 | 20480/0/0 | 全 fresh | 仅观察 |

表中三元组均为 `fresh/stale/other`。

### 4.1 `dc cvac + dsb sy` 是否真的需要保留

对**当前 A5 main `aicpu_scheduler` Path-A**，结论是：每次 AICPU → AICore 发布不需要
保留 `dc cvac + dsb sy + isb`；应把 control 发布改成 release atomic store，而不是
直接把旧 ordinary-control 协议中的 clean/barrier 裸删掉。

case 19 是这项结论的主门禁：

```text
AICPU ordinary store payload
  -> __atomic_store_n(isolated_control, generation, __ATOMIC_RELEASE)
     // HCC 产物为 stlr；无 dc cvac、dmb、dsb、isb

AICore return-value atomic poll isolated_control
  -> DSB
  -> DCCI payload line
  -> DSB
  -> ordinary load payload
```

它与旧 case 10 的关键差别不是轮数，而是排除了隐藏发布动作：`tested_control` 自己就是
doorbell，AICPU 不再写独立 `done`；AICore 完成末轮读取后再以 `rounds+1` ACK，AICPU
收到 ACK 后才允许 clean 结果或退出。五进程共 20480 轮全部 fresh，doorbell atomic
poll 无超时，单轮最大 43 次尝试。owner ELF 反汇编确认 payload 为 ordinary `str`，
control 为 `stlr`，两者之间及其后没有 `dc cvac`、DMB、DSB 或 ISB。

case 23 进一步把 ordinary payload/control 与独立 doorbell 分开，以 release atomic
doorbell 排序，仍为 `20480/0/0`。case 27 则证明删掉 producer cvac 不等于删掉
consumer DCCI：同一 release 发布下，AICore 不 DCCI 时 ordinary payload 连续
20480 轮全部 stale，而同轮 `ld_dev` 全 fresh。

case 18 的 ordinary control 无 clean、无 barrier 也观察到 `20480/0/0`，但它仍不是
工程协议：普通 store 没有明确的 payload-before-control 发布顺序。当前放行的是
**ordinary payload + release atomic control** 这一组合，不是“所有 AICPU store 都可
随意去掉 cache/order 操作”。不同 cust AICPU launch、cluster affinity、芯片或 CANN
版本仍需同构复测；仓库已记录另一 cluster 的 cust AICPU 不在该 snoop domain 的历史
故障，不能外推本结论。

## 5. AICore → AICPU 矩阵

AICPU 先预读旧 control/payload，再通过正确的 AICPU ordinary-store + cvac 控制线
发布 `ready`。旧 case 0～13 由 AICore 另发 atomic `done`；direct case 14～19 只用
`tested_control` 作为唯一 producer doorbell。AICPU 观察 doorbell 后先读取并分类全部
payload，随后才 release-store post-primary ACK，并对被测地址执行
`dc civac + dsb sy + isb` 做 reference 读取。最后一轮还会先发布 ACK，AICore 收到 ACK
后才退出，避免 kernel-end 行为参与被测窗口。

| ID | AICore producer / AICPU consumer 差异 | control primary | payload primary | civac reference | 口径 |
|---:|---|---:|---:|---:|---|
| 0 | `atomicExch`；payload default DCCI；AICPU acquire | 20480/0/0 | 20480/0/0 | 全 fresh | 正式门禁 |
| 1 | control `atomicAdd` | 20480/0/0 | 20480/0/0 | 全 fresh | atomic 返回值门禁 |
| 2 | control `atomicMax` | 20480/0/0 | 20480/0/0 | 全 fresh | atomic 返回值门禁 |
| 3 | control `atomicCAS` | 20480/0/0 | 20480/0/0 | 全 fresh | atomic 返回值门禁 |
| 4 | control ordinary store + DCCI + DSB | 20480/0/0 | 20480/0/0 | 全 fresh | ordinary 发布门禁 |
| 5 | payload 显式 `CACHELINE_ALL` | 20480/0/0 | 20480/0/0 | 全 fresh | DCCI selector |
| 6 | payload `CACHELINE_OUT` | 20480/0/0 | 20480/0/0 | 全 fresh | DCCI selector |
| 7 | payload `CACHELINE_ATOMIC` | 20480/0/0 | 20480/0/0 | 全 fresh | DCCI selector |
| 8 | AICPU 已缓存旧值后直接 ordinary load | 20480/0/0 | 20480/0/0 | 全 fresh | AICore DCCI 后 AICPU cache 自动更新/失效的实测 |
| 9 | 被测 control 用 AICPU relaxed load | 20480/0/0 | 20480/0/0 | 全 fresh | 只测 control 值；harness done 仍为 acquire |
| 10 | AICPU primary 先做 civac | 20480/0/0 | 20480/0/0 | 全 fresh | 冗余 invalidate 对照 |
| 11 | AICore payload 不 DCCI | 20480/0/0 | 0/20480/0 | 0/20480/0 | observe，稳定负例 |
| 12 | AICore ordinary control 不 DCCI | 0/20480/0 | 20480/0/0 | 0/20480/0 | observe，稳定负例 |
| 13 | AICore 单次 `st_dev` control | 20480/0/0 | 20480/0/0 | 全 fresh | observe，不作为业务放行依据 |
| 14 | 32 条 dirty line；default DCCI 全部发出后执行一个 DSB，再 `atomicExch` doorbell | 20480/0/0 | 20480/0/0 | 全 fresh | DSB 正对照 |
| 15 | 与 14 相同，但 DCCI 后不显式 DSB，直接 `atomicExch` doorbell | 20480/0/0 | 20480/0/0 | 全 fresh | **当前 A5 精确路径门禁** |
| 16 | 32 条 dirty line；OUT DCCI 全部发出后执行一个 DSB，再 `atomicExch` doorbell | 20480/0/0 | 20480/0/0 | 全 fresh | selector/DSB 正对照 |
| 17 | 与 16 相同，但 OUT DCCI 后不显式 DSB，直接 `atomicExch` doorbell | 20480/0/0 | 20480/0/0 | 全 fresh | **当前 A5 精确路径门禁** |
| 18 | 32 条 dirty line；不执行 DCCI，只执行一个 DSB，再 `atomicExch` doorbell | 20480/0/0 | 0/20480/0 | 0/20480/0 | 必须 stale 的负例 |
| 19 | 32 条 dirty line；既不执行 DCCI，也不执行 DSB，直接 `atomicExch` doorbell | 20480/0/0 | 0/20480/0 | 0/20480/0 | 必须 stale 的负例 |

这条方向与 AICPU → AICore 不对称：AICore 正确 DCCI 后，AICPU 不需要对同一
payload 再做 `dc civac`；但 AICore 不 DCCI 时，AICPU 自己做 civac 也无法把仍留在
AICore local cache 的 dirty line 变出来。

### 5.1 DCCI 后的显式 DSB 是否必须

对本测试精确覆盖的序列，答案是：**当前 A5 上不必须**。

```text
Scalar ordinary-store 32 条相互隔离的 data cacheline
  -> 对 32 条 line 逐条执行 default DCCI 或 SINGLE CACHELINE_OUT DCCI
  -> 不执行显式 DSB
  -> atomicExch(isolated_control, generation)
  -> AICPU acquire 观察 control
  -> AICPU ordinary-load 并精确校验全部 32 条 line
```

五个独立进程中，两种 no-DSB case 各有 20480 轮，每一轮的 32 条 line 都全部
fresh，累计检查每个 case `20480 × 32 = 655360` 条 line，`stale=0`、`other=0`；
带 DSB 的同构对照完全相同。两个 no-DCCI 负例则不论有没有 DSB 都连续 20480 轮
全部 stale，证明结果不是 AICPU 自然看见 Scalar ordinary dirty store，也证明 DSB
不能替代 DCCI。

测试构建显式关闭编译器自动 Scalar DCCI 和 kernel-end DCCI。优化后的 device LLVM
IR 中，no-DSB 分支从最后一个 `llvm.hivm.DCCI.DST` 直接进入
`llvm.hivm.atom.EXCH.G.s64`，中间没有 `llvm.hivm.DSB`；带 DSB 分支才经过
`llvm.hivm.DSB`。当前工具无法反汇编 hiipu64 最终机器码，因此证据口径是“优化后
device IR + 真实 A5 运行”，不冒充最终指令反汇编。

AICPU consumer 也排除了隐藏等待：它在 acquire control 后立即逆序读取 payload，先读
最后一条 DCCI 目标；payload primary 已保存并分类后，才允许执行 control/payload
reference 的 `dc civac + dsb sy + isb`。AArch64 owner ELF 反汇编确认 direct ordinary
payload mode 从 `ldar` 进入 payload `ldr` 循环，随后才以 `stlr` 写 post-primary ACK；
首个后续 `dc civac`/`dsb sy` 位于 ACK 之后，不会替 Scalar producer 补一个被测窗口内
的 DSB。

Scalar 侧也不允许下一轮 atomic 暗中助攻。两个 no-DSB case 每 64 轮抽取一轮：
`atomicExch` 后执行 1048576 个 NOP，期间没有 GM/atomic/DSB，然后第一条后续内存指令
才用 `atomicAdd(done, 0)` 检查 AICPU 的 post-primary ACK。优化后 device IR 精确呈现
`atomicExch -> NOP loop -> atomicAdd ACK`。五进程中每个 selector 共 320 个静默窗口，
ACK 全部在检查前到达；因此这些样本的 payload 已在 Scalar 下一条内存指令之前被读取，
不能归因于下一轮 ready atomic 促成 DCCI 完成。

边界同样明确：本机头文件只给出 `DSB_ALL` 等待全部 memory access 的定义，没有找到
“DCCI 自身完成”或“任意 atomic 都替代 DSB”的公开契约。实测只能说明紧随其后的
`atomicExch` 在当前 A5/CANN/编译产物中形成了足够的发布边界。非 atomic 后继、不同
atomic primitive、reader 侧 DCCI、不同芯片/CANN 或不同 launch 路径仍保留显式 DSB，
直到有对应契约或同构门禁。

## 6. atomic 命令差异

### 6.1 AICore CCEC atomic

| 命令 | 本用例用法 | 返回值用途 | 不能替代的能力 |
|---|---|---|---|
| `atomicAdd(ptr, 0)` | 不改变值的返回型观察 | 返回线性化前旧值，即本次读取结果 | 不替代 payload DCCI |
| `atomicMax(ptr, INT64_MIN)` | int64 恒等 RMW | 同上 | 不等于普通 cached load |
| `atomicCAS(ptr, new, new)` | 已知目标值时恒等 CAS | 返回 CAS 前旧值 | 不提供邻接 bytes 的可见性 |
| `atomicExch(ptr, new)` | 单 writer 发布 control | 返回发布前旧值并逐轮精确校验 | 不保护同 line ordinary dirty writeback |
| `atomicAdd/Max/CAS` 写入 | AICore → AICPU 发布对照 | 4096 轮返回值必须等于上一代 | 仍需独立 payload 发布顺序 |

生产 control 应继续放在 atomic-only 独占 line；不要对其中混入 ordinary payload，
也不要依赖随后对同 line 的 DCCI 保护 atomic 更新。

### 6.2 AICPU ARM atomic

本机 HCC 产物反汇编确认：

- `__atomic_load_n(..., ACQUIRE)` 生成 `ldar`；
- `__atomic_store_n(..., RELEASE)` 生成 `stlr`；
- seq_cst exchange/fetch-add/compare-exchange 生成 `ldaxr/stlxr` 重试循环。

它们在当前 A5 样本中都能被 AICore atomic observe 看到。其中 release store 已通过
无隐藏 clean 的 direct 门禁，作为当前 main `aicpu_scheduler` Path-A 的发布方式；
RMW 变体仍只作互操作观察。这些结果不等于已经获得跨芯片、跨 CANN 的公开 ABI/ISA
契约，换 AICPU launch/affinity 路径仍需复测。

### 6.3 `ld_dev/st_dev`

- `ld_dev` 在本目录只做写完成后的单次 reference，不在 writer 尚未完成时对同一
  地址死循环，避免已知的同址 bypass 读取压力阻塞 writer；
- `ld_dev` 绕过 AICore Scalar DCache，所以在 ordinary load 保持 stale 时仍能看到
  GM 新值；
- AICore 单次 `st_dev` control 当前为 20480 fresh，但仓库已有 repeated-store 精确
  终值回退证据，因此这里只是观察项，不改变 `ATOMIC_USAGE_GUIDE.md` 对业务
  `st_dev` 写路径的禁用规则。

## 7. 可直接采用的双向协议

### 7.1 AICPU 发布，AICore 消费

```text
AICPU
  ordinary store payload
  __atomic_store_n(isolated_control, generation, __ATOMIC_RELEASE)
  [当前 A5 Path-A 不执行 dc cvac / 显式 dmb/dsb / isb]

AICore Scalar
  return-ready atomic observe isolated control
  DCCI each ordinary payload cache line
  DSB_ALL
  compiler memory barrier
  optional second atomic observe for immutable/version check
  ordinary load payload
```

不能把 release control 退化为无顺序保证的 ordinary store；AICore 也不能把 control
换成已缓存 ordinary load，更不能只因为 control atomic 已 fresh 就省略 payload DCCI。

### 7.2 AICore 发布，AICPU 消费

```text
AICore Scalar
  ordinary store payload
  DCCI payload lines
  [当前 A5、紧随 atomicExch 发布的精确路径：显式 DSB 可省]
  [其他路径：保留 DSB_ALL]
  atomic publish isolated control

AICPU
  acquire-load/poll control
  [若前序通知来自 Device MMIO，则另按协议执行 rmb/dsb ld]
  ordinary load payload
```

这条路径中，AICPU 不需要再 `dc civac` AICore 已 DCCI 的 payload。若生产者省略
DCCI，AICPU civac 无法补救。省略 DSB 的放行范围仅限上述经过 direct 门禁的
`DCCI lines -> atomicExch` 序列，不应改写为通用 DCCI 规则。

## 8. 构建与运行

```bash
cd tests/atomic_probe/aicpu_aicore_cache
./build.sh build
ATOMIC_PROBE_DEVICE=0 ./build.sh run
```

`build` 会执行：

1. Host layout/matrix contract；
2. 两个 `dav-c310-vec` CCEC kernel 构建与 metadata/symbol 检查；
3. 仓库通用 AICPU dispatcher 构建；
4. AICPU owner 构建；
5. 复用 `src/common/aicpu_loader` 的 Host runner 构建。

`run` 外层固定 180 秒超时；AICPU 每个等待有 5 秒单点超时，AICore 每个等待有
system-counter 超时。没有无限自旋。输出中的四组 `fresh/stale/other` 分别对应
control primary、control reference、payload primary、payload reference。

## 9. 本次环境与复测边界

- 日期：2026-08-12；
- CANN：`cann-9.1.0-weekly-20260708/cann-9.1.0`；
- CCEC target：`dav-c310-vec`；
- device：`/dev/davinci0`；
- 复测：五个独立 Host 进程，每个方向每项 4096 轮；AICore→AICPU 新增 32-line
  DSB 对照共检查每个 case 655360 条 data line；每个 no-DSB selector 另有 320 个
  post-publish 静默窗口 ACK；
- 结果：五轮均为 `AICPU->AICore=PASS AICore->AICPU=PASS`，所有
  `other=0`，所有受检 atomic 返回旧值精确；
- 机器未提供 `npu-smi`，仓库架构预检脚本无法读取 silicon 名称；设备节点、编译
  target 和真实 ACL/RTS launch 均正常。本结论限定为当前 A5 开发环境，不冒充其他
  芯片或未来 CANN 的公开内存模型。

本测试是正确性/能力探针，不是性能 benchmark。扩展后的整组为 32×4096 与
20×4096 次握手；加入静默窗口后 AICore→AICPU 单次 wall 约 1.9 s，包含初始化、
32-line direct 压力、NOP 静默门禁、逐轮 barrier、atomic poll、reference 读取和结果
记录，不能拆成单条 atomic、DCCI、DSB 或发布协议的延迟。
