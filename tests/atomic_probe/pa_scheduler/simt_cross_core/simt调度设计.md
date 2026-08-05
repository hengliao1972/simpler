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
| ---: | --- |
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
| 15 | 状态机与 `cross_core` 一致；GM 是否需要 DCCI 先由最小 A5 对照实验决定，UBUF 路径自行管理 UB 和 MTE3。 |
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
7. 官方混合编程示例展示了 `SIMT -> UBUF -> MTE3 -> GM` 的基本链路，
   并要求为 SIMT Data Cache 至少保留 32 KB，同时保留编译器预留空间。

官方接口参考：

- [asc_vf_call](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_10303.html)
- [SIMD 与 SIMT 混合编程](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10052.html)
- [SIMT VF 函数限制](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/guide/%E6%8A%80%E6%9C%AF%E9%99%84%E5%BD%95/CPP%E6%A0%87%E5%87%86%E6%94%AF%E6%8C%81/%E8%AF%AD%E6%B3%95%E9%99%90%E5%88%B6/%E5%87%BD%E6%95%B0.md)

### 2.2 选择一次 mixed kernel launch

正式协议使用一个静态 1:2 mixed AICore ELF 和一次 kernel launch：

```text
AIV0 Main Scalar       启动 SIMT builder ───────────── 等待并收口 builder
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

AIV0 的 `__aicore__` Main Scalar 负责发起 SIMT VF，但真正 payload 构造发生在
SIMT 执行域。AIV0 不因为保留了一段 Main Scalar 控制代码就被算作 executor。

### 2.4 AIV ELF 分类约束

AIV entry 同时承载 SIMT builder 和普通 AIV task executor，最终 ELF 必须被
编译器分类为 `SIMD_SIMT_MIX_VF`，不能是 `SIMT_VF_ONLY`。构建门槛必须检查：

- AIV metadata 的 VF 类型为 MIX；
- AIC entry 没有错误的 SIMT VF 分类；
- 只有预期的 AIC/AIV 两个全局 device entry；
- 内部 SIMT VF 使用 internal linkage，不额外注册 `_simt_entry`；
- UB metadata 与静态/动态 UB 预算满足至少 32 KB Data Cache 的要求。

## 3. 协议与内存合同

### 3.1 共同状态机

GM 与 UBUF 两条路径都保留同一共享终态：

| 转换 | 唯一责任方 | 合同 |
| --- | --- | --- |
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
等价为 DCCI，也不能先验地断言 GM 路径一定不需要 DCCI。S0/S1 先用同一地址
重复复用的最小 A5 探针对照以下四种可见性组合：writer/reader 都不做 DCCI、
SIMT writer 做单行 DCCI、Claim winner 做 payload 单行 DCCI、两侧都做。随后
还要分别覆盖 AIV0 到其他 AIV、AIV0 到 AIC 的跨核读取。

正式 GM 路径只采用硬件证据支持的最小序列：如果无 DCCI 组合在重复 launch、
地址复用和两类跨核方向都稳定通过，则保留纯 thread-fence 路径；如果失败，
就保留能闭合正确性的最小 writer/reader DCCI，不能为了减少指令而省略。

### 3.3 UBUF 构建路径

UBUF 是每个 AIV 私有、不可跨核共享的暂存区。第一版只做单槽，之后扩展
双槽或小型 ring；任何槽在 MTE3 完成前都不能被下一 task 复用。

基础顺序固定为：

```text
AIV Main Scalar 预留 cell，置为 BUILDING
  -> SIMT VF 逐 word 写本 AIV 的 __ubuf__ staging slot
  -> 建立 VF/UB 完成边界
  -> MTE3 将有效 payload 精确复制到 GM cell
  -> 等待 MTE3 完成
  -> Main Scalar CAS 发布 BUILT
  -> 释放并复用 UBUF slot
```

直接 CCEC 路径优先使用编译器的 `copy_ubuf_to_gm_align_v2` 一类 MTE3
intrinsic，并显式检查 V/MTE3/S 事件顺序。不得通过引入 AscendC `TPipe`、
`LocalTensor` 或 `DataCopy` 来绕过生命周期设计。UB 大小、对齐、slot owner、
有效字节数和复用状态都必须是显式 ABI。

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
| --- | --- | --- |
| D0 | 两份设计/过程文档 | 边界、风险和门槛完整，不写伪结果。 |
| S0 | 基础协议与 SIMT 自检 | CPU 状态机及 AIV0 SIMT 的线程、GM 写入、完成等待在 A5 闭合。 |
| S1 | 单 Vector task | AIV0 构建，AIV executor 唯一领取并通过 golden。 |
| S2 | 单 Cube task | AIV0 构建，AIC executor 唯一领取并通过 golden。 |
| S3 | Vector + Cube | 两个 task 同时发布，engine 路由、完成和 drain 正确。 |
| S4 | 多 task、单 builder | task-id 扫描、fanin、token busy 和无遗失。 |
| G0 | GM 完整 PA | shared TensorMap 主 Case 的五类 task、DAG 和 golden 闭合。 |
| G1 | AIV0+AIV1 GM | 两 builder 竞争构建；两者仍零 task execute。 |
| U0 | UBUF 单槽探针 | VF->UB、MTE3、publish 顺序和重复复用正确。 |
| U1 | UBUF 多槽/多 task | 无提前复用、无覆盖、异常可收口。 |
| U2 | UBUF 完整 PA | 与 G0 相同 task/DAG/golden，GM 与 UBUF 长期共存。 |

S0 允许使用单独 AIV-only launch 先确认 SIMT 语法和线程行为；从 S1 开始必须
进入同一次 mixed kernel launch。G1 的首版只做正确性竞争，静态分片、批次分片
和减少重复扫描都属于后续有数据支撑的优化。

## 5. 验收、交付与停止线

### 5.1 每阶段固定验收

每个阶段完成后不等待人工确认，依次完成：

1. CPU 严格告警构建与协议测试；涉及并发时增加确定性交错和随机压力，
   可运行的 portable 代码还要经过 ASan/UBSan，TSan 只作 CPU 证据；
2. CCEC 分别按 `dav-c310-cube`、`dav-c310-vec` 构建需要的对象，静态检查
   entry、metadata、未定义符号、UB 大小以及关键 atomic/fence/MTE3 顺序；
3. 按仓库规范先执行 A5 arch precheck，再通过 `task-submit` 运行真实 A5；
4. host oracle 检查 golden、task 数、唯一 builder/executor、最终 control、
   guard、inactive tail、fatal 和超时；
5. 将命令、结果、失败过程和边界写入实现过程文档；
6. 只提交本阶段文件，使用详细中文 commit，并 push 当前分支；
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
- builder AIV 零执行，executor 零构建；
- `[task_count, capacity)` 未被写入，所有 guard 和 padding 保持不变；
- 多次 fresh launch 与同地址复用都不读到上一轮 payload。

### 5.3 必须停下定位的情况

以下任一情况发生时，不进入下一阶段：

- AIV ELF 被标记为 SIMT-only，或出现额外全局 SIMT entry；
- 无法证明 `async_invoke` 后的可靠完成边界；
- GM 的所有候选可见性序列都出现旧 payload、部分 payload或重复 launch 不稳定；
- 64-bit SIMT CAS 与 Main Scalar CAS 对同一 control 的结果不一致；
- UBUF slot 在 MTE3 完成前被复用，或 UB 预算侵占最小 Data Cache；
- builder 执行 task、executor 构建 task，或 host 只能靠推断而不能取证；
- fatal/timeout 无法让 kernel 有界退出；
- 完整 PA 的 DAG、task 数或 golden 与基线不一致。

上述问题先记录真实复现、产物和根因，再做最小修正；不能用扩大超时、减少
检查或改变 workload 掩盖。
