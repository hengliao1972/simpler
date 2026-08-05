# SIMT 调度设计

本文定义 `simt_cross_core` 的实施边界、设备拓扑、协议合同、分阶段交付顺序
和验证门槛。它是第四种 PA standalone 调度实现的设计基线，不记录尚未运行
的结果；实际命令、结果和问题统一追加到
[`simt调度实现过程.md`](simt调度实现过程.md)。

## 1. 目标、范围与已对齐事项

### 1.1 目标

使用少量固定 AIV 上的 SIMT 线程构建 PA task，其余 AIC/AIV 只领取并执行
task。第一版固定 AIV0 为 builder，后续扩展为 AIV0、AIV1 两个 builder。
builder 与 executor 必须严格分工：

- builder AIV 只构建、发布 task，不执行任何 PA task；
- executor AIC/AIV 只领取、执行和完成 task，不参与 task 构建；
- 完整 PA 保持 `cross_core` 的 task DAG、task 数量、执行包 ABI 和状态机语义；
- 先完成可校验的小型协议探针，再接入 shared TensorMap standalone PA；
- `__gm__` 与 `__ubuf__` 是长期共存的两条实现路径，不相互替换。

### 1.2 二十项需求对齐结论

| 序号 | 已冻结结论 |
| ---: | ---------- |
| 1 | 工作分支固定为 `fdwic-swimlane-deps`。所有新增和修改代码只能位于本目录。 |
| 2 | 本实现独立演进，不与 `cross_core` 共享源码。 |
| 3 | CPU 只模拟协议语义；设备实现直接由 CCEC 编译，不新增 AscendC 实现。 |
| 4 | SIMT 用法以本机 `ops-nn` 和 CANN 官方资料为查证来源。 |
| 5 | 第一版 builder 为 AIV0，后续才扩展 AIV0、AIV1。 |
| 6 | builder 负责生成完整可发布执行包；executor 负责领取、执行和完成。 |
| 7 | builder 和 executor 的角色互斥，运行中不能互相补位。 |
| 8 | 分别验证 SIMT VF 的 `__gm__` 与 `__ubuf__` 指针入参。 |
| 9 | 最小闭环覆盖发布顺序、唯一领取、半包不可见和超时退出。 |
| 10 | 每个小阶段都要有 CPU、CCEC、真实 A5、中文记录和独立提交；泳道不要求，性能数据不强制。 |
| 11 | 首个完整 PA 用例复用 shared TensorMap standalone 主 Case。 |
| 12 | launch 形态由实现查证决定；本设计选择一次 mixed kernel launch 为主路线。 |
| 13 | 不再使用容易混淆的 `block0/block1`，统一称为 AIV0/AIV1。 |
| 14 | 不产生对 `cross_core` 或 `ops-nn` 的源码依赖；公共泳道解析工具可按需调用。 |
| 15 | 状态机与 `cross_core` 一致；GM 可见性由最小 A5 对照实验决定。当前 CCEC 没有 SIMT-native MTE3 接口，因此 U0 先验证 SIMT 自管 UB 单槽生命周期和 `UBUF -> SIMT load -> GM store`，不伪称 MTE3。 |
| 16 | 单 Vector/Cube 阶段使用可校验的最小任务，不直接搬入完整 PA 计算。 |
| 17 | 完整 PA 阶段只替换构建侧，原有 task DAG 和 task 数量保持不变。 |
| 18 | `gm/` 与 `ubuf/` 分目录长期保留。 |
| 19 | 双 builder 首版允许竞争领取构建权，性能优化后置。 |
| 20 | 每阶段完成后自动进入下一阶段，不等待人工确认。 |

### 1.3 明确不做的事情

- 不修改 `cross_core/`、`same_core/`、Simpler runtime 或真实 PA 目录；
- 不把 CPU 时延解释为 A5 性能；
- 不用分离的前后两个 kernel launch 冒充构建与执行并行；
- 不因性能方便而改变 task DAG、任务数、完成顺序或 golden；
- 不在没有用户授权时新增行为开关、环境变量或条件编译宏；
- 不把尚未通过真实 A5 的内存序推断写成硬件结论。

## 2. 查证依据与总体架构

### 2.1 已查证的存量事实

1. `cross_core` 的共享执行包使用
   `EMPTY -> BUILDING -> BUILT -> CLAIMED -> DONE`，control 独占 64 B，
   payload 从下一条 cacheline 开始。参考
   [`shared_exec_protocol.h`](../cross_core/common/shared_exec_protocol.h)。
2. 当前完整 PA 每 batch 为 `Alloc + QK + SF + PV + UP` 五个 task；Alloc
   不执行 kernel，QK/PV 进入 AIC，SF/UP 进入 AIV。参考
   [`pa_model.h`](../cross_core/common/pa_model.h) 和
   [`pa_exec_adapter.h`](../cross_core/common/pa_exec_adapter.h)。
3. A5 mixed 拓扑为每个物理 block 配置 1 个 AIC 和 2 个 AIV。AIV 的稳定
   逻辑编号应由
   `get_block_idx() * get_subblockdim() + get_subblockid()` 展平，不能把
   AIC 与 AIV 各自从 0 开始的 `block_idx` 混为一个编号空间。
4. CCEC 自带 `cce::async_invoke`、`cce::dim3`、`__simt_vf__`、
   `threadIdx` 等编译器接口；现有 A5 dispatcher 已用这些接口解决
   SIMT 元数据分类，参考
   [`docs/simt-launch.md`](../../../../docs/simt-launch.md)。
5. 官方接口允许 SIMT VF 接收 `__gm__ *`、`__ubuf__ *` 和基础标量，
   但不允许把结构体、数组或未标地址空间的指针作为 VF 参数，也不允许
   间接函数调用。实现必须逐基础字段或逐 word 写 payload，不能直接对
   GM/UBUF 中的结构体赋值。
6. 本机 `ops-nn` 的
   `hash/embedding_hash_table_export/op_kernel/arch35/`
   `embedding_hash_table_export.h` 已展示同一 SIMT VF 同时接收 GM 与 UBUF
   指针；adaptive-pool 示例还展示了调用前建立 UB 可见性边界。
7. 本机 CCEC 的 `copy_ubuf_to_gm_align_v2` 位于
   `namespace __cce_scalar`，包装层通过
   `CCE_SCALAR(copy_ubuf_to_gm_align_v2)` 调用；SIMT API 中没有对应
   UBUF→GM/MTE3 接口。将该 Scalar intrinsic 强行保留在
   `__simt_vf__` 内的最小 CCEC 探针不能生成合法代码，因此不能把
   普通 aicore 代理 MTE3 冒充为纯 SIMT 实现。
8. 本机 `ops-nn` 的 `embedding_hash_table_export.h` 给出了当前可行
   边界：普通 aicore 仅分配 UB 并将 `__ubuf__ *` 传给 VF，SIMT thread
   从 UBUF 读取中间值后直接写 GM。这是逐线程 GM store，不是 MTE3。
   U0 只使用这条已有真实代码证据的路径，并继续为 SIMT Data Cache
   至少保留 32 KB。

官方接口参考：

- [asc_vf_call](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_10303.html)
- [SIMD 与 SIMT 混合编程](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10052.html)
- [SIMT VF 函数限制](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/guide/%E6%8A%80%E6%9C%AF%E9%99%84%E5%BD%95/CPP%E6%A0%87%E5%87%86%E6%94%AF%E6%8C%81/%E8%AF%AD%E6%B3%95%E9%99%90%E5%88%B6/%E5%87%BD%E6%95%B0.md)

### 2.2 选择一次 mixed kernel launch

正式协议使用一个静态 1:2 mixed AICore ELF 和一次 kernel launch：

```text
AIV0 __aicore__ entry  只启动/等待 VF，不承担任何 task 语义
SIMT threads(AIV0)       领取构建权 -> 写 payload -> 发布 BUILT ...
AIC0..AIC31            观察 BUILT -> Claim -> 执行 AIC task -> DONE
AIV1..AIV63            观察 BUILT -> Claim -> 执行 AIV task -> DONE
```

这样 builder 发布第一个 task 后，其他核即可开始执行，不需要等全部 task
构建结束。两个顺序 kernel launch 只允许作为早期 SIMT 基础功能诊断，不能成为
最终调度方案，因为它无法证明 task build 与 task execute 的设备内重叠。

### 2.3 角色与 owner 编号

- AIC executor owner 保持 `0..31`；
- AIV owner 保持 `32 + aiv_id`；
- 单 builder 时 owner 32 对应 AIV0，但该 owner 不持有 execution token；
- 双 builder 时 owner 32、33 对应 AIV0、AIV1，二者都不持有 token；
- 单 builder 的 AIV executor 为 AIV1..AIV63；双 builder 时为
  AIV2..AIV63；AIC executor 数量始终不变；
- host 终态必须分别证明 builder 的执行次数为 0、executor 的构建次数为 0。

AIV0 的 `__aicore__` entry 是工具链要求的 VF 启动壳，不是
builder、executor 或调度角色。payload 构造、严格插入、发布和
归因全部发生在 SIMT 执行域。

### 2.4 AIV ELF 分类约束

AIV entry 同时承载 SIMT builder 和普通 AIV task executor，最终 ELF 必须被
编译器分类为 `SIMD_SIMT_MIX_VF`，不能是 `SIMT_VF_ONLY`。构建门槛必须检查：

- AIV metadata 的 VF 类型为 MIX；
- AIC entry 没有错误的 SIMT VF 分类；
- 只有预期的 AIC/AIV 两个全局 device entry；
- 内部 SIMT VF 使用 internal linkage，不额外导出 GLOBAL
  `_simt_entry`；
- UB metadata 与静态/动态 UB 预算满足至少 32 KB Data Cache 的要求。

## 3. 协议与内存合同

### 3.1 共同状态机

GM 与 UBUF 两条路径都保留同一共享终态：

| 转换 | 唯一责任方 | 合同 |
| ---- | ---------- | ---- |
| `EMPTY -> BUILDING` | builder | CAS 成功者取得该 task 的唯一构建权。 |
| `BUILDING -> BUILT` | builder | payload 全部可见后才能发布。 |
| `BUILT -> CLAIMED` | compatible executor | 只能有一个 CAS winner。 |
| `CLAIMED -> DONE` | Claim winner | kernel 完成、vend 和 flag 发布后才能完成。 |

control、fatal 和 drain word 使用独立 atomic-only cacheline；普通 payload
不得与原子控制字共行。Claim loser 不读取 payload，也不执行可见性操作。
`BUILDING` 只保留 builder owner 与 task id，executor 为 unbound，engine 为
`None`，payload lines 为 0；engine 与非零 payload lines 只能随 `BUILT` 一起
发布，executor owner 只能从 `CLAIMED` 开始绑定。decoder 对每个 phase 分别
校验，任何提前发布或畸形组合都 fail-closed。

### 3.2 GM 直接构建路径

每个 SIMT thread 独立负责完整 task，避免多 thread 拼同一 payload 所需的
额外 barrier。初始扫描采用 thread-stride task-id 分配；双 builder 时两个 AIV
可以扫描同一 task 范围，并通过 `EMPTY -> BUILDING` CAS 竞争构建权。

发布顺序固定为：

```text
SIMT 64-bit CAS 取得 BUILDING
  -> 逐基础字段/uint64 word 写 GM payload
  -> SIMT thread fence
  -> SIMT 64-bit CAS 发布 BUILT
```

`thread fence` 只建立 SIMT 普通 GM store 先于发布 CAS 的顺序，不能先验地
等价为 DCCI，也不能先验地断言 GM 路径一定不需要 DCCI。S0～S2 用同一地址
重复复用的最小 A5 探针对照以下四种可见性组合：writer/reader 都不做 DCCI、
SIMT writer 做单行 DCCI、Claim winner 做 payload 单行 DCCI、两侧都做。随后
分别覆盖同 AIV、AIV0 到其他 AIV、AIV0 到 AIC 的读取。

正式 GM 路径只采用硬件证据支持的最小序列：如果无 DCCI 组合在重复 launch、
地址复用和两类跨核方向都稳定通过，则保留纯 thread-fence 路径；如果失败，
就保留能闭合正确性的最小 writer/reader DCCI，不能为了减少指令而省略。

截至 S2，已有三条各 100 轮、每轮四模式的 A5 证据：S0 的同 AIV读取、
S1 的 AIV0 builder 到 AIV1 executor 跨 AIV 读取，以及 S2 的 AIV0 builder
到 AIC executor 跨引擎读取都得到
`NO_DCCI=1/100`、`WRITER_DCCI=0/100`、`READER_DCCI=100/100`、双侧
DCCI `100/100`。因此当前 GM 路径在已覆盖的三种方向上统一冻结为 Claim
winner 成功后对 payload 做 reader DCCI + DSB；writer DCCI 不是必要条件。
这个结论只覆盖 A5、当前普通 GM payload、同地址重复复用和现有 compiler
DCCI 配置，不能外推为所有 GM 访问都使用相同 cache 协议。

这组数据能支持的一致性口径是“SIMT/V 侧普通 GM 写”与“Main
Scalar/S 侧普通 GM 读”不能按单一自动一致 DCache 使用，不是对未公开的
物理 cache 层次做推测。`asc_threadfence()` 只建立 SIMT 普通 store 与后续
atomic 发布的顺序；Scalar reader 仍需对 payload line 执行 DCCI + DSB。
atomic-only control line 依赖原子访问的同地址顺序，不能把这个性质外推到旁边的
普通 payload line。`ld_dev` 仅在探针诊断路径中用于绕开 Scalar 普通
DCache 读取，不代替正式 executor 的 reader DCCI 合同。

### 3.3 UBUF 构建路径

UBUF 是每个 AIV 私有、不可跨核共享的暂存区。第一版只做单槽，之后扩展
双槽或小型 ring。当前工具链只支持 SIMT 从 UBUF 读取后直接写
GM，不支持 VF 内发起 MTE3；因此 U0 的每个槽在有效 payload 全部直接
GM store、`asc_threadfence()` 和 `BUILT` 发布完成前，都不能被下一
task 复用。

U0 固定发射 2048 thread/64 warp，仅每个 warp 的 lane0 工作。
64 个 leader 分别负责一个独立 task，先并发竞争各自的
`EMPTY -> BUILDING`，然后竞争同一个 UBUF staging slot。slot owner
必须位于独立 GM atomic-only cacheline，不得假设 UBUF 支持所需的
跨 warp atomic。单槽的基础顺序固定为：

```text
SIMT warp leader: EMPTY --CAS--> BUILDING
  -> 用 GM atomic CAS 取得唯一 UBUF slot
  -> 逐 word 写本 AIV 的 __ubuf__ staging slot
  -> 同一 leader 逐 word 从 __ubuf__ 读回并直接写目标 GM payload
  -> asc_threadfence()
  -> 同一 leader CAS 发布 BUILT
  -> 同一 leader 用 GM atomic CAS 释放 UBUF slot
```

AIV0 `__aicore__` entry 仅允许提供固定、对齐的 UB 基址，将基础指针传入 VF，
并执行 `async_invoke -> join -> drain`；它不能预留 task cell、填充 UBUF、
搬运 payload、发布 `BUILT` 或释放 slot。role report 的聚合
`main_scalar_build_action_count`、task claim/finish 必须全为 0，task 构建
细分阶段另由 SIMT report 和源码/bitcode 门槛取证。UB region 大小、payload
偏移、64 B 对齐、slot 数、slot owner、有效 cacheline 数和搬运方式必须是
显式 ABI；`slot_ticket + launch_nonce` 共同标识槽的本轮复用次序。U0 用
`1/10/16/68` 条 64 B payload 覆盖最小、当前 PA 常用规模和最大执行包边界，
每个 task 只复制自己的有效行，不得固定搬满 68 行。

U0 的诊断必须显式记录
`transport=SIMT_UBUF_READ_TO_GM_WORD_STORE`、
`mte3_count=0`、slot 最大 busy depth 为 1，并验证 slot acquire/release
各恰好 64 次。这一阶段只证明 UBUF 指针、单槽生命周期和 SIMT
直接 GM 发布正确，不证明 MTE3 能力，也不把它写成性能优化。若后续
必须验证批量 UBUF→GM MTE3，需要先获得 SIMT-native 工具链接口，
或另行对齐“普通 aicore 仅作 transport engine”的新角色边界；本设计
不默认引入该回退。

### 3.4 executor 与结束条件

executor 复用 `cross_core` 的兼容 engine 判断、Claim、fanin、dispatch
binding、engine completion、vend、completion flag 和 DONE 语义，但在本目录
独立实现。首版 task-indexed cell 不回收，避免 generation/reclaim 混入验证。

完整 PA 的 task 数由配置确定。builder 发布完已知 task 集后发布 builder
完成证据；executor 只有在以下条件同时满足时才能退出：

- 所有 builder 已结束或 global fatal 已发布；
- 本核没有 busy execution token；
- 全局 DONE 数与计划 kernel task 数一致；
- drain root 已核对所有 executor 的到达与完成数。

任意超时或非法状态都发布首错 fatal，并让 host 得到可定位的 owner、task id
和阶段；不得用无限轮询把协议错误变成设备超时。

### 3.5 双 task 路由与 drain 合同

S3 使用两个互不共享 control/payload cacheline 的 task slot。AIV0 的 SIMT
thread 0 只构建 Vector slot，thread 1 只构建 Cube slot；一个 slot 仍处于
`BUILDING` 时，另一个 slot 可以独立进入 `BUILT -> CLAIMED -> DONE`。AIV1
只观察并领取 Vector slot，AIC 只观察并领取 Cube slot，不能通过遍历另一个
engine 的 slot 形成隐式 Claim 竞争。

SIMT invoke 完成且两个 report 都证明发布成功后，AIV0 才把
`builder_finished` 从 0 原子发布为 1。每个 executor 必须先等本 engine 的
MTE3/FIX 写回边界，再把自己的 control 从 `CLAIMED` 改为 `DONE`，最后对
`done_count` 原子加一。三个角色只有同时观察到
`builder_finished == 1 && done_count == 2` 才能通过全局 drain；因此
`done_count == 2` 不能由“已发射两个 workload”或重复完成同一个 task 代替。

S0～S2 已分别闭合同 AIV、跨 AIV 和 AIV→AIC 三种 GM descriptor 读取方向，
S3 起不再重复四种已淘汰的 DCCI 模式。两个 Claim winner 都固定在 Claim
成功后对自己的 payload line 执行 reader DCCI + DSB，再做普通 GM load。
每个 stage 仍必须用同地址重复运行和完整 golden 证明这条冻结规则没有被新
协议破坏。

### 3.6 SIMT atomic 同地址竞争合同

S0 只由 SIMT thread 0 执行 CAS，证明了指令可编译且单线程路径可用，
但没有覆盖多 warp 对同一 GM 地址竞争。A0 独立探针只验证当前调度会
用到的 GM `uint64_t asc_atomic_cas` 和 `asc_atomic_add`，不扩展到 UBUF、
其他数据类型或其他 atomic 操作。

本机 CANN dav_3510 头文件定义 warp size 为 32、SIMT 最大线程数为
2048；本机 `ops-nn` 同时存在 1024-thread 常用算子和 2048-thread
`sparse_tensor_dense_mat_mul` 实现。因此 A0 固定验证 32/64/1024/2048
四档，分别覆盖 1/2/32/64 warp。每档必须同时满足：

- 同地址 CAS 恰好一个 winner，winner 返回 64-bit initial value；
- 其余 CAS loser 全部返回 winner 写入的最终 64-bit desired value；
- 同地址 atomic-add 最终值精确增加 thread count，返回 ticket 是
  `[initial, initial + thread_count)` 的不重不漏排列；
- active thread marker 全部匹配，inactive tail 保持 sentinel，所有 guard 不变；
- 同一 device allocation 重复使用，每轮改变 nonce 且验证 64-bit 高位。

Main Scalar 在 V→S completion 之后用 `ld_dev` 建立设备侧诊断摘要，host
对所有逐线程返回值再做一次精确 oracle。这里故意不使用 Scalar 普通
GM load，避免把 SIMT/Scalar 普通 DCache 可见性与 atomic 返回值语义混在一起。

### 3.6.1 同 warp 串行与跨 warp 独立推进合同

A1 不用“发射了 64 个 thread”推断两个 warp 一定并行，而是把 forward
progress 和墙钟区间分别做成可证伪的设备 oracle。公开 SIMT `clock()` 在
当前 CANN 9.1 实现中落到 `__cce_simt_get_CLOCK64()`；warp id 没有独立 API，
固定由 `threadIdx.x / 32` 推导，lane 由 `threadIdx.x % 32` 推导。

四种模式复用同一 A/B 工作代码和同一份 GM 状态：

- `A-only`、`B-only` 分别记录两条不同代码路径的 CLOCK64 区间与 checksum；
- `same-warp` 只让 warp0 的 lanes 0..15 和 lanes 16..31 进入互斥外层
  分支，只有 leader tid0/tid16 执行握手和 A/B work；
- `cross-warp` 把 A/B 外层分支分别放在 warp0/warp1，同样只有 leader
  tid0/tid32 执行握手和 work；
- A/B leader 先 CAS 发布自己的 ready，再用 `atomic-add(0)` 有界读取对方
  ready，所有等待同时受最大 poll 数和 CLOCK64 deadline 限制。

每份 CLOCK64 区间从对应 `RunA/RunB` 入口开始，包含握手、poll 和 work。
同 warp 的一条分歧路径在另一条路径尚未执行时无法完成双向握手，因此先执行
的一方必须 timeout，后一方只能单向观察到前者；两份总区间必须不重叠。
跨 warp 只有双方都在对方的 bounded poll 窗口内取得过 forward progress，
才可能同时成功；对应总区间必须重叠。这里的“跨 warp 独立推进”不等价于两条
指令每周期同时 issue：warp 仍共享执行管线，所以 A1 不把设备总耗时强行断言为
`max(A,B)`。CPU 的 step oracle 另行验证理想调度关系
`same=A+B`、`cross=max(A,B)`，设备结论以握手因果和 CLOCK64 区间为准。

每轮四种模式使用不同 nonce，但复用同一 device allocation；host 必须核对
active/inactive report、精确 tid/warp/lane、A/B checksum、ready、guard 和
Main Scalar 的 `ld_dev` 摘要。A1 是调度语义探针，不生成泳道图，也不作为
完整 PA 的性能数字。

### 3.7 多 task 扫描、busy token 与 fan-in 合同

S4 使用 16 个交错编号的 task：偶数 task 为 Vector，奇数 task 为
Cube，各 8 个。AIV0 发射 128 个 SIMT thread，即 4 个 32-thread warp。
task `i` 映射到
`tid=(i%4)*32+((i/4)%32)`；等价地，thread 从
`lane*4+warp` 开始并按 128 递增。这样 16 个 task 均匀落到 4 个 warp，
而不是由同一 warp 的相邻 lane 完成。每个 task 仍由单 thread 完整写一条
descriptor，不在多 thread 之间拼包。

AIV1 只扫描偶数 task，AIC 只扫描奇数 task。两个 executor 各自只有一个
busy token：只有 token free 时才能 CAS `BUILT -> CLAIMED`；真实 MTE3/FIX
写回完成、`CLAIMED -> DONE` 成功且完成计数发布后才能释放 token。
CPU 模型必须受控暂停在第一个 task 的 busy 区间，并证明第二个 task 不会被
提前 Claim；CCEC 源码和设备结果同时检查最大 busy depth 恰好为 1。

drain cacheline 独立记录 `builder_finished`、`vector_done`、`cube_done`
和 `done_count`。三个角色只有同时观察到 `1/8/8/16` 才能退出；每个
task 的 `DONE` CAS 必须早于分 engine 计数，分 engine 计数必须早于全局
`done_count`。这是 S4 的完成 fan-in，不代替 G0 中真实 PA DAG 的依赖 fan-in。

每个 task 使用独立的 16×16 FP32 input/output tile；Vector 做逐元素 add，
Cube 做对角左矩阵的 matmul。不同 task 的输入包含 task ordinal 且输出地址
不同，host 必须逐 task、逐元素核对 golden，不允许用一块共享输出伪装
多 task 执行。

### 3.8 G0 纯 SIMT 多 warp 构建与严格插入链

S4 的 128-thread 映射用于 16 个互不依赖的探针 task，不能直接搬到 G0。
完整 PA 的 shared TensorMap writer metadata 必须按 task id 严格提交；若同一
warp 的多个 lane 分别负责相邻 task，并让后继 lane 在分歧路径中等待前驱，
A1 已经证明先进入等待的分支可能阻止同 warp 的前驱分支取得 forward
progress。因此 G0 不允许用 Main Scalar 代替 SIMT 顺序提交，也不允许同一
warp 内有两个 task builder。

G0 固定发射 2048 个 SIMT thread，即已经由 A0 验证过的 64 个 warp。每个
warp 只有 lane 0 是 builder，其他 31 个 lane 不得读取、预留、构造或发布
任何 task 状态。task 映射固定为：

```text
builder_warp(task_id) = task_id % 64
builder_tid(task_id)  = builder_warp(task_id) * 32
first_task(warp)      = warp
next_task             = current_task + 64
```

B256/context8192 共 1280 个 task，因此 64 个有效 warp leader 各构建 20 个
task。任意 `task[N]` 与 `task[N-1]` 都位于不同 warp，包含 `63 -> 64` 的
回绕边界；所以严格插入等待只发生在不同 warp 之间。host 必须逐 task 核对
实际 builder tid，并证明 1984 个 inactive lane 的整份诊断 report 仍为 host
poison，即它们既不访问/构建 task，也不产生诊断 GM store。

每个 warp leader 对自己的 task 完整执行以下流程：

```text
可执行 task: EMPTY --CAS--> BUILDING
  -> 在 8-shard shared heap 上并发预留输出区间
  -> 写完整 fresh-output descriptor 并发布每个 output
  -> 写完整 inline execution payload，但保持 BUILDING
  -> 原子等待 task[N-1].insert_completion（task0 无前驱）
  -> 提交 writer history 与 last_writer
  -> 发布本 task 的 insert_completion
  -> Alloc 发布 vend/flag；kernel task 发布 BUILT
```

heap task base 由真实的分 shard atomic reservation 决定，不能按 task id
伪造一个串行地址。后继 task 若需要前驱输出 descriptor，只读取前驱独占的
atomic task-base 报告，再按已冻结 PA shape 重新构造 descriptor；它不能在
发布位之前普通读取另一个 SIMT warp 尚未完成的 descriptor。这样既保留并发
heap 分配，又不新增未经验证的 SIMT 普通 DCache 一致性假设。

最后一个 task 的 SIMT builder 在观察到完整严格前缀后发布
`builder_finished`。AIV0 `__aicore__` entry 只允许发起 VF、等待 V→S 完成并以零
执行数参加最终 drain；它不能构造 descriptor/payload，不能提交 history、
last_writer 或 insert-completion，也不能发布任何 task 的 `BUILT`。
其 role build/commit/claim/execute 计数也必须全部为 0；构建总数只从 64 份
SIMT thread report 求和，不能归因给 entry 壳。executor
仍使用 AIC/AIV 两条 immutable ticket 表和每 worker 四个 token，在 task
尚未发布时停留于 `WaitingBuilt`，不能因一次观察到 `EMPTY/BUILDING` 就丢失
该 task。

### 3.9 G1 两个独立 VF 的全量竞争合同

G1 不把 G0 的 64 个 leader 拆成两半，也不允许 Main Scalar 代替第二组
builder。AIV0 和 AIV1 各自发射一份完整的 2048-thread VF；每份 VF 都有
64 个 warp，仍只有每个 warp 的 lane0 有效。因此 G1 有 128 个有效 worker，
但任一 warp 内始终只有一个 worker：

```text
AIV0: global_tid = local_tid,        owner = 32, global_warp = 0..63
AIV1: global_tid = 2048 + local_tid, owner = 33, global_warp = 64..127
active(local_tid) = (local_tid % 32 == 0)
task scan          = local_warp + 64*k
```

两个 VF 扫描同一组 task，而不是静态分片。每个 VF 的 thread0 先对
`builder_started` 各原子到达一次，所有 active leader 必须观察到
`builder_started == builder_count` 后才能发起第一个 claim。这个闸门只能证明
两份 VF 已经进入；逐 task 的参与尝试由独立设备证据验证：每个 leader 在
claim 前对本 task 的 `build_attempt_count` 原子加一，唯一 CAS winner 再对
`build_win_count` 原子加一。attempt/win 证明两个对应 leader 最终都尝试，
不单独声称两次尝试的时间区间必然重叠。最终每个 task 必须精确满足：

```text
build_attempt_count == builder_count  # G0 为 1，G1 为 2
build_win_count     == 1
```

可执行 task 竞争 `EMPTY -> BUILDING(actual_build_owner)`；Alloc 没有 execution
control，复用其最终 completion flag 做临时 claim：
`0 -> ALLOC_BUILDING(actual_build_owner) -> 1`。CAS loser 只允许观察另一合法
builder 的临时/后续状态，不得预留 heap、写 descriptor/payload 或进入严格
insert chain；同 owner 的重复 claim 和畸形状态必须报 fatal。winner 独立完成
G0 的全部 Prepare/Commit，实际 build owner 必须同时保存在 plan、task report
和 kernel task 的 BUILDING/BUILT/CLAIMED/DONE 状态中。Alloc 虽然终态 flag
只有 1，也必须由 plan/report 保留 winner owner。

两份 VF 的 4096 份 thread report 使用不相交下标。每个 active leader 的
`attempt = win + loss`，所有 leader 汇总必须为：

```text
attempt = builder_count * task_count
win     = task_count
loss    = (builder_count - 1) * task_count
```

inactive lane 的整份 thread report 必须保持 host poison；G0 未发射的第二实例
`[2048,4096)` 同样必须保持 host poison。两份 `__aicore__` entry 仅允许
`async_invoke -> V/S join -> drain`，其 role 的 build/commit/claim/execute/
ticket 均为 0。AIV executor 因此从 owner34 开始，G1 总 executor 为
32 AIC + 62 AIV = 94；16 个 drain group 仍各有 6 个物理参与者，owner32
仍是唯一 root。

task build report 只有一条 64 B cacheline，其中 word6 保存 attempt/win。
为避免另一个 VF 的 atomic 与 winner 的普通 DCache store 同行竞争，整条 report
禁止普通 SIMT store：word6 只用 atomic-add，其他 word 只允许从 host poison
通过 atomic-CAS 发布。CPU 模型中的独立 `std::atomic` 不能替代这条设备规则。

### 3.10 U1 四槽、128 task 与 generation 合同

U1 不为了强行保留 U0 的 68-line 边界而臆测动态 UB 接口。
G0 的真实 PA payload 中 QK/SF/PV 都是 10 行，UP 是 16 行，因此
U1 将每槽最大 payload 固定为 16 行。每槽布局为前 guard 1 行、
payload 16 行、后 guard 1 行，即 1152 B；四槽合计 4608 B，低于
已由 U0 产物证明的 TLV7 8192 B 静态 share 预算，不需要新的
launch attribute，并继续保留至少 32 KiB SIMT DCache。但公开
文档没有完整说明裸 UBUF offset 与 VF stack 在 TLV7 内的物理分区；
U0 真机只触达 `0..4479`，U1 会首次触达 `4480..4607`。因此
4608 B 只是静态容量入场条件，四槽前后 guard 和真实 A5 结果仍是
必须通过的地址边界门槛。U0 仍独立覆盖 68 行
单槽边界；U1 只声称四槽的 `1/4/10/16` 行。

U1 仍发射 2048 thread/64 warp，只允许每个 warp 的 lane0 工作。
任务数增为 128，每个 leader 精确构建 `warp` 和 `warp+64`
两个 task。槽与长度映射固定为：

```text
slot_id(task)       = task_id % 4
payload_class(task) = ((task_id / 4) + 3) % 4
payload_lines       = {1, 4, 10, 16}[payload_class]
```

这样 task0..3 都是 16 行 anchor，而每个物理槽在 32 次复用中都
精确经历八次 `1/4/10/16`，不会把槽号与 payload 长度永久绑定。

槽状态为独占 64 B 的 GM atomic-only cacheline。低 32 位为
`task_id+1`，0 表示 free；高 32 位为 generation。acquire 只能将
`FREE(g)` CAS 为 `BUSY(g, task)`，release 只能将原值 CAS 为
`FREE(g+1)`。build report 必须记录 slot id 和取得时的 generation；
host 对每槽独立验证 generation 恰好是 `0..31` 且无重复，终态为
`FREE(32)`。这是防止提前释放和 ABA 复用的主要协议证据。

四个 anchor leader 先各自取得不同槽，完整写入 16 行并检查
前后 guard，然后各自用 CAS 在 `anchor_staged_mask` 中只置自己的
task bit，再累加 `anchor_staged_count`。重复 bit 必须报 fatal；四者在
保持槽所有权的状态下同时等到 `count=4 && mask=0xf`。其他
60 个 leader 在此之前不能 acquire。因此真机终态必须观察到
`anchor_staged_count=4`、`anchor_staged_mask=0xf` 和 global
`max_busy_depth=4`，证明四个不同 anchor 的完整
staging payload 曾同时驻留；这仍不外推为所有 64 个 leader 的
指令区间全部重叠。

count 与 mask 是两条不同的 GM atomic cacheline。发布顺序虽然固定为
先置身份 bit、再加 count，但没有把跨地址可见顺序当成未经验证的硬件
前提：reader 只有同时读到精确的 `4/0xf` 才开门；所有取值范围合法但
暂时不匹配的组合都继续有界轮询。只有 count 越界、mask 出现非法 bit、
重复置同一身份 bit，或最终 watchdog 超时才报错；身份位图自身的 CAS
竞争也有固定尝试上限。

slot CAS 成功取得 `BUSY` 后必须立即增加 global busy depth，然后
才能开始 staging。释放时先用有界 CAS 将 global busy depth 减一，
再执行精确的 `BUSY(g,task)->FREE(g+1)`；若第二步失败，必须
先回滚 busy depth 再报 fatal。这避免新 owner 在旧 owner 减计数前
已重新 acquire 同一槽，从而把 `max_busy_depth` 伪增到 5。

每个 winner 的正常顺序继续是纯 SIMT：取得 task、取得槽、只写
有效 UBUF word、检查 guard、同 leader 读 UBUF 后直接写 GM、fence、
发布 `BUILT`、推进 generation 并释放槽。AIV0 `__aicore__` entry 壳
仍只能 invoke/join/drain；AIV1 作为与 U0 一致的独立 executor 进行
Claim/DCCI/校验/DONE，不参与 task 构建。任何 pre-publish 异常都
必须先用精确 BUSY 值释放所属槽、推进 generation，尚未发布
的 task 再尝试 `BUILDING->EMPTY`，最后让全局 fatal 使其他角色有界
收口。持槽 leader 在等待中观察到别的线程已发 fatal 时也必须走
同一 cleanup epilogue；不得留下 busy 槽或伪造已发布 task。
GM copy 与 fence 之后、`BUILDING->BUILT` 紧邻之前还必须再读一次
global fatal，封住 copy 期间由其他角色发布首错的窗口。busy decrement
最多尝试固定次数；超过上限保留当前可解释的 BUSY/busy-depth 状态并
发布或保留首个 fatal，不能在 cleanup 内无限自旋。

AIV1 executor 同样必须在每次 `BUILT->CLAIMED` 之前检查 fatal，并在
claim 成功后、读取 payload 前再次检查。第二个窗口若观察到首错，owner33
必须用精确 CAS 将自己的 `CLAIMED` 恢复为 `BUILT`，不得增加 claim/done
计数；payload 校验完成到 `DONE` 之前再做一次同样的 fatal 边界检查。

## 4. 目录与分阶段实施

### 4.1 计划目录

```text
simt_cross_core/
  simt调度设计.md
  simt调度实现过程.md
  common/                 # 独立 ABI、状态机、host/device 公共定义
  protocol_probe/
    cpu/                  # 协议语义和受控交错
    ccec/                 # 最小 mixed A5 探针
    test/
    simt_atomic/          # GM uint64 CAS/add 多 warp 同地址独立探针
    warp_concurrency/     # 同 warp 分歧串行与跨 warp 独立推进探针
  gm/
    common/
    cpu/
    ccec/
    test/
  ubuf/
    common/
    cpu/
    ccec/
    test/
  test_record/            # 仅保存约定的设备结果；泳道不是阶段门槛
  run.sh                  # 统一构建/运行入口，内部不 include cross_core
```

系统 C/C++ 头、CCEC builtin 头和 ACL runtime 属于工具链依赖，不算
`cross_core` 源码依赖。若以后生成泳道，允许调用 `pa_scheduler` 现有 converter
和 analyzer，但不得复制后形成第二套解析规则。

### 4.2 阶段顺序

| 阶段 | 交付内容 | 关键通过条件 |
| ---- | -------- | ------------ |
| D0 | 两份设计/过程文档 | 边界、风险和门槛完整，不写伪结果。 |
| S0 | 基础协议与 SIMT 自检 | CPU 状态机及 AIV0 SIMT 的线程、GM 写入、完成等待在 A5 闭合。 |
| S1 | 单 Vector task | AIV0 构建，AIV executor 唯一领取并通过 golden。 |
| S2 | 单 Cube task | AIV0 构建，AIC executor 唯一领取并通过 golden。 |
| S3 | Vector + Cube | 两个 task 同时发布，engine 路由、完成和 drain 正确。 |
| A0 | SIMT atomic 竞争 | 32/64/1024/2048 thread 的 GM uint64 CAS/add 返回值与终值精确。 |
| A1 | warp 推进语义 | 同 warp 握手不能双向完成且区间串行；跨 warp 双向完成且区间重叠。 |
| S4 | 多 task、单 builder | task-id 扫描、fanin、token busy 和无遗失。 |
| G0 | GM 完整 PA | shared TensorMap 主 Case 的五类 task、DAG 和 golden 闭合。 |
| G1 | AIV0+AIV1 GM | 两 builder 竞争构建；两者仍零 task execute。 |
| U0 | UBUF 单槽探针 | 64 个 warp leader 的 VF→UB、SIMT 直接 GM store、publish 顺序和重复复用正确；明确 `mte3_count=0`。 |
| U1 | UBUF 多槽/多 task | 纯 SIMT 槽所有权、无提前复用、无覆盖、异常可收口。 |
| U2 | UBUF 完整 PA | 与 G0 相同 task/DAG/golden，GM 与当前可用的 UBUF 路径长期共存。 |

S0 允许使用单独 AIV-only launch 先确认 SIMT 语法和线程行为；从 S1 开始必须
进入同一次 mixed kernel launch。G1 的首版只做正确性竞争，静态分片、批次分片
和减少重复扫描都属于后续有数据支撑的优化。

## 5. 验收、交付与停止线

### 5.1 每阶段固定验收

每个阶段完成后不等待人工确认，依次完成：

1. CPU 严格告警构建与协议测试；涉及并发时增加确定性交错和随机压力，
   可运行的 portable 代码还要经过 ASan/UBSan，TSan 只作 CPU 证据；
2. CCEC 分别按 `dav-c310-cube`、`dav-c310-vec` 构建需要的对象，静态检查
   entry、metadata、未定义符号、UB 大小以及本阶段实际 transport 的关键
   atomic/fence 顺序；使用 MTE3 的阶段才检查 MTE3 顺序，不使用的阶段反向
   检查对应 intrinsic 不存在；
3. 按仓库规范先执行 A5 arch precheck；环境提供 `task-submit` 时通过它运行
   真实 A5，否则明确记录为用户授权的 unlocked 单卡功能验证；
4. host oracle 检查 golden、task 数、唯一 builder/executor、最终 control、
   guard、inactive tail、fatal 和超时；
5. 将命令、结果、失败过程和边界写入实现过程文档；
6. 只提交本阶段文件并使用详细中文 commit；只有用户明确授权后才能 push；
7. 自动进入下一阶段。

泳道图不是阶段门槛。性能数据也不是强制门槛；如果顺手测量，必须记录
构建身份、参数、运行轮数和原始结果，且不能用带泳道时间与无泳道时间相减。

### 5.2 完整 PA 终态 oracle

G0、G1、U2 至少检查：

- B1 的逐 task 状态和数值 golden；
- shared TensorMap 主 Case 的完整规模；
- 每 batch 精确 5 个 task，Alloc 无 kernel，四个 kernel task 各执行一次；
- QK/PV 只由 AIC 执行，SF/UP 只由非 builder AIV 执行；
- fanin、vend、completion flag 和 DONE 顺序正确；
- execution witness 记录 executor 实际推进到的 `fanin_ready_prefix`，host 按
  task 的 0/1/1/3 条 fanin 精确核对；
- builder AIV 零执行，executor 零构建；
- `[task_count, capacity)` 未被写入，所有 guard 和 padding 保持不变；
- 多次 fresh launch 与同地址复用都不读到上一轮 payload。

### 5.3 必须停下定位的情况

以下任一情况发生时，不进入下一阶段：

- AIV ELF 被标记为 SIMT-only，或出现额外全局 SIMT entry；
- 无法证明 `async_invoke` 后的可靠完成边界；
- GM 的所有候选可见性序列都出现旧 payload、部分 payload或重复 launch 不稳定；
- 64-bit SIMT CAS 与 executor Scalar CAS 对同一 control 的结果不一致；
- UBUF slot 在有效 UBUF→GM 直接 store、fence 和 `BUILT` 发布完成前被复用，
  或 UB 预算侵占最小 Data Cache；
- builder 执行 task、executor 构建 task，或 host 只能靠推断而不能取证；
- fatal/timeout 无法让 kernel 有界退出；
- 完整 PA 的 DAG、task 数或 golden 与基线不一致。

上述问题先记录真实复现、产物和根因，再做最小修正；不能用扩大超时、减少
检查或改变 workload 掩盖。
