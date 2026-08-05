# SIMT 调度实现过程

本文只记录 `simt_cross_core` 已经发生的开发动作、可复现命令、测试结果、
失败证据和提交。未来计划以
[`simt调度设计.md`](simt调度设计.md) 为准；计划本身不能当作完成证据。

## 1. 记录规则

- 每个阶段单独记录目标、修改文件、CPU 结果、CCEC 静态结果、真实 A5
  结果、未闭合问题和 commit SHA；
- CPU、CCEC IR/ELF、A5 正确性和性能是不同证据，不能互相替代；
- 未实际执行的命令只能写在“下一步”，不能写成 PASS；
- 失败尝试同样记录，说明保留、修正或撤回了什么；
- 性能不是阶段强制项；一旦给出性能数字，必须同时给出对照、参数、轮数、
  构建身份和结果文件；
- 每个阶段只提交 `simt_cross_core/` 内的文件；
- 阶段提交和 push 完成后自动继续，不等待人工确认；
- 若遇到设计停止线，先停下定位，不能用减少校验或扩大超时继续推进。

阶段记录统一使用以下结构：

```text
## 日期：阶段编号与名称
### 目标和边界
### 修改文件
### CPU 命令与结果
### CCEC 命令、IR/ELF 检查与结果
### 真实 A5 命令与结果
### 可选性能数据
### 失败、修正和仍未闭合项
### 阶段结论与 commit
```

## 2. 2026-08-05：D0 需求对齐与接口查证

### 2.1 工作身份

- 分支：`fdwic-swimlane-deps`；
- 目录：`tests/atomic_probe/pa_scheduler/simt_cross_core/`；
- 本阶段只建立设计与过程文档，没有写协议或设备实现；
- 本阶段没有修改 `cross_core/`、`same_core/`、runtime 或真实 PA 路径；
- 后续代码不 include、link 或运行时加载 `cross_core` 源码。

### 2.2 需求对齐结果

本轮把 builder 固定为 AIV 身份而不是含义模糊的 block 身份：第一版只有
AIV0，后续扩展为 AIV0、AIV1。builder 与 executor 完全互斥。CPU 负责协议
语义模拟，设备代码直接用 CCEC 构建。GM 和 UBUF 两条路径长期共存：GM
先用于建立最简单的正确性基线，UBUF 后续用于 MTE3 暂存和性能探索。

完整 PA 选择 shared TensorMap standalone 主 Case，保持现有五类 task、DAG、
task 数量和执行 engine 不变。每个小阶段要求 CPU、CCEC、真实 A5、中文记录
和独立提交；泳道和性能数字不作为阶段必需条件。

launch 方式在需求对齐时没有现成答案，用户授权根据查证结果选择。本轮设计
选择一次 mixed kernel launch：AIV0 在 entry 内发起 SIMT builder，其余 AIC
和 AIV 同时进入 executor。分离的前后两个 launch 只保留为 P1 基础诊断，
不作为正式 build/execute overlap 方案。

### 2.3 仓库协议查证

查阅了以下存量实现：

- `cross_core/common/shared_exec_protocol.h`；
- `cross_core/protocol_probe/` 的 CPU、CCEC 与 host oracle；
- `cross_core/ccec/kernel.cpp`、`build.sh` 和 `run.sh`；
- `cross_core/common/pa_model.h`、`pa_frontend.h`、
  `pa_exec_adapter.h`；
- `cross_core/PA调度器分离版实现过程.md`；
- `docs/simt-launch.md`；
- `docs/investigations/2026-07-shared-pa-case1-performance-gap.md`。

得到的可复用语义如下：

1. execution cell 的 control 独占 64 B，payload 从第二条 cacheline 开始；
2. 状态机为 `EMPTY -> BUILDING -> BUILT -> CLAIMED -> DONE`；
3. builder owner、executor owner、task id、engine 和 payload 长度都进入
   可审计 control/header；
4. Claim loser 不读 payload；唯一 winner 负责 fanin、engine、vend、flag
   和 DONE；
5. 完整 PA 每 batch 有 Alloc、QK、SF、PV、UP 五个 task，QK/PV 属于 AIC，
   SF/UP 属于 AIV；
6. A5 mixed entry 当前按 1 AIC + 2 AIV/物理 block 编译；AIV id 由 block
   和 subblock 展平；
7. 现有 `cross_core` 的 CPU 模型、CCEC 混合 ELF、host oracle 和构建产物
   检查方式可以参考，但新目录必须重新形成独立源码闭包。

逐字段核对还确认了 phase 约束：`BUILDING` 只有合法 builder owner 和 task id，
executor 必须为 unbound，engine 必须为 `None`，payload lines 必须为 0；只有
`BUILT` 才发布非空 engine 和 1～68 条 payload line。`CLAIMED/DONE` 才允许
绑定合法 executor owner。S0 的独立 decoder 按这套约束 fail-closed，而不是
只检查字段位宽。

`docs/investigations/` 没有发现已经否决“固定 AIV 上 SIMT 构建 task”的记录。
已有 Case1 性能调查提醒：standalone 与真实 PA 不等价，不能在正确性机制尚未
闭合时用单个时延数字驱动微优化。本路线因此先做协议和完整 PA golden，
性能不提前设收益结论。

### 2.4 CCEC 与官方 SIMT 接口查证

本机 CANN 9.1 工具链提供并自动包含以下基础能力：

- `__simt_vf__` 将函数标记为 SIMT VF entry；
- `cce::async_invoke<Func>(cce::dim3{...}, args...)` 配置线程维度并发起 VF；
- `threadIdx`、`blockDim` 等 SIMT builtin；
- 64-bit GM SIMT atomic CAS；
- SIMT thread fence；
- `copy_ubuf_to_gm_align_v2` 等直接 MTE3 intrinsic；
- V、MTE3、S pipeline 的 `set_flag`/`wait_flag`。

官方 `asc_vf_call` 文档确认：混合编程中的 SIMT VF 由 `__aicore__` 调用，
线程总数不超过 2048，VF 只能接收 raw pointer 和基础标量。官方语法限制还
确认：

- 指针形参必须明确为 `__gm__` 或 `__ubuf__`；
- 不能把栈数组、结构体或间接函数指针传入 SIMT VF；
- 混合场景不支持直接对 GM/UBUF 结构体整体赋值；
- 因此 task payload 必须逐字段或逐基础 word 构造。

官方混合编程示例给出的 UB 预算为 256 KB 总量，除编译器预留空间外，
SIMT 至少需要 32 KB Data Cache。示例还展示了 SIMT 写 UBUF、建立事件边界、
再由 MTE3 搬到 GM 的基本链路。

### 2.5 本机 `ops-nn` 查证

重点阅读了：

- `hash/embedding_hash_table_export/op_kernel/arch35/`
  `embedding_hash_table_export.h`；
- `hash/init_embedding_hash_table/op_kernel/arch35/`
  `init_embedding_hash_table.h`；
- `pooling/adaptive_max_pool2d/op_kernel/arch35/`
  `adaptive_max_pool2d_simt.h`；
- `pooling/adaptive_pool3d_common/op_kernel/arch35/`
  `adaptive_max_pool3d_simt.h`。

这些实现证明同一 SIMT VF 可以同时接收 GM 与 UBUF 指针，SIMT thread 可以
直接遍历和写 GM，也可以消费 Main Scalar 预先填好的 UBUF 参数。它们是接口
和时序参考，不会成为新目录的源码依赖；其中 AscendC 的 `TPipe`、
`LocalTensor`、`DataCopy` 也不会搬入直接 CCEC 实现。

### 2.6 当前只形成设计、尚未形成硬件结论的事项

以下内容必须由 S0/S1/U0 的 CCEC 产物和真实 A5 动态结果回答：

1. AIV entry 发起真实 SIMT VF 后，哪一组 V/S event 能可靠证明 VF 完成；
2. SIMT 普通 GM store、thread fence、64-bit CAS 发布后，其他 AIC/AIV
   Main Scalar 是否能稳定读取新 payload；writer/reader 哪一侧需要 DCCI
   不能预设，由四组最小对照决定；
3. 上述 GM 可见性在多次 kernel launch、复用同一地址时是否仍成立；
4. SIMT 64-bit CAS 与 Main Scalar 64-bit CAS 竞争同一 control 时的返回值
   和全序是否一致；
5. 实际执行 SIMT builder 后，最终 AIV ELF 是否稳定产生 MIX VF metadata，
   且不会额外导出 SIMT entry；
6. UBUF 写完成到 MTE3 读取需要的精确 event、barrier 和等待序列；
7. 直接 MTE3 的有效字节、对齐和尾部行为是否满足最大执行包；
8. AIV0 退出 builder 后，其余 executor 的 drain 是否能有界完成。

这些问题都能通过小型探针逐项查证，不需要在 D0 阶段继续脑补接口，因此
本轮没有追加新的需求问题。

### 2.7 D0 修改与验证

新增文件：

- `simt调度设计.md`；
- `simt调度实现过程.md`。

本阶段是纯文档变更，按仓库 testing 工作流不运行 CPU、CCEC 或 A5 测试。
已完成以下检查：

- 当前分支确认为 `fdwic-swimlane-deps`；
- 本目录只有两份约定的 Markdown 文件；
- 文档引用的 `cross_core` 协议、PA model、adapter 和
  `docs/simt-launch.md` 相对路径全部存在；
- 两个新文件分别执行 `git diff --check --no-index`，没有空白错误；
- 当前环境没有 `markdownlint-cli2` 或 `markdownlint` 可执行文件，因此没有
  伪写 markdownlint PASS。

### 2.8 下一阶段

S0 将在同一阶段建立独立 portable ABI、CPU 受控交错和最小 CCEC SIMT
基础算子。CPU 部分只证明角色、状态机、唯一 Claim、半包不可见、fatal 和
timeout，不模拟 A5 cache 或 SIMT 指令时延；CCEC 和真实 A5 部分负责证明
SIMT 线程、GM 写入与完成边界。S0 通过并提交后自动进入单 Vector task。

## 3. 2026-08-05：S0 基础协议与 SIMT 自检

### 3.1 目标、范围与需求校正

S0 只回答两类问题：portable CPU 状态机是否闭合；AIV0 能否用直接 CCEC
发起 64-thread SIMT VF，并完成 GM 写入、64-bit CAS、V/S completion 和
Main Scalar 领取。S0 使用单 AIV launch，不包含真实 Vector/Cube task，也不
对跨 AIV、跨 AIC 可见性下结论。

开发过程中用户进一步指出，GM 经过 Data Cache 时不能先验地断言“不需要
DCCI”。本阶段据此把原设计中的绝对口径改为四组硬件对照：

| 模式 | SIMT writer payload DCCI | Claim winner payload DCCI |
| --- | --- | --- |
| `NO_DCCI` | 无 | 无 |
| `WRITER_DCCI` | 有 | 无 |
| `READER_DCCI` | 无 | 有 |
| `WRITER_AND_READER_DCCI` | 有 | 有 |

四种模式共用一份 kernel 机器码和同一 GM 地址。每次 launch 更换 nonce 并由
host 重置整个状态，专门暴露冷 cache 首轮成功、地址复用后读旧值这一类问题。
只有在全部重复轮次通过的模式才有资格成为后续协议候选；候选模式失败是
实验结论，不会被误报成探针框架失败。

### 3.2 修改文件与独立源码闭包

新增：

- `common/shared_protocol.h`：独立定义与 `cross_core` 数值一致的
  `EMPTY -> BUILDING -> BUILT -> CLAIMED -> DONE` 64-bit control 编码，
  以及首错 fatal 编码；
- `protocol_probe/common/s0_probe.h`：64 B 对齐的 S0 host/device ABI、
  四种 DCCI 模式、固定 payload/thread oracle；
- `protocol_probe/test/test_s0_protocol.cpp`：CPU 受控半包交错、16 actor
  Claim 竞争、首错 fatal 和有界 timeout；
- `protocol_probe/cpu/build.sh`：optimized、ASan/UBSan、TSan 三套构建；
- `protocol_probe/ccec/kernel.cpp`：AIV0 Main Scalar、内部 SIMT VF、GM
  可见性矩阵和 `st_dev` 结果发布；
- `protocol_probe/ccec/host.cpp`：独立 ACL loader、重复地址复用、四模式
  分类和 SoC 身份校验；
- `protocol_probe/ccec/build.sh`：CCEC、bitcode inventory、ELF metadata、
  symbol 和 ACL host 构建门槛；
- `run.sh`：S0 的统一 build/run 入口。

修改：

- `simt调度设计.md`：把“GM 不做 DCCI”修正为“小 case 先决定最小
  writer/reader DCCI”；
- 本过程文档：记录实际构建和 A5 结果。

所有新增源码都位于 `simt_cross_core/`。构建脚本扫描 C/C++ include，确认
没有 include `cross_core` 或本机 `ops-nn` 源码；PTO metadata、CCEC builtin、
SIMT API 和 ACL 是工具链依赖，不是其他调度实现的源码依赖。

### 3.3 CPU 命令与结果

统一命令：

```bash
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-s0
```

其中 CPU 部分得到：

```text
[PASS] S0 CPU protocol rounds=128: half-packet hidden, unique claim,
       first fatal and bounded timeout
[PASS] S0 CPU protocol rounds=64:  ASan+UBSan
[PASS] S0 CPU protocol rounds=32:  TSan
```

CPU oracle 使用显式暂停点让 builder 在写完 4/8 payload word 后停住，确认：

1. control 保持 `BUILDING` 时 executor 不得 Claim，payload read 计数为 0；
2. 第二个 builder 的 `EMPTY -> BUILDING` CAS 失败且不能改写半包；
3. 发布 `BUILT` 后 16 个并发 executor 只有一个 Claim winner；
4. winner 精确读取 8 个 word 并发布 `DONE`，15 个 loser 不读 payload；
5. 8 个 fatal reporter 只有一个成功发布可解码的首错；
6. 固定 64 次 poll 能返回 timeout，不使用无界死循环。
7. phase-specific decoder 拒绝 `BUILDING` 提前携带 engine/payload、`BUILT`
   提前绑定 executor、`DONE` 未绑定 executor、payload 超过 68 line 和保留位
   非零等畸形状态。

CPU 只证明 C++ 内存模型下的协议角色和状态转换，不模拟 A5 DCache、SIMT
指令时延或跨核可见性。

### 3.4 CCEC、bitcode 与 ELF 静态结果

CCEC 使用 `dav-c310-vec`，并显式关闭 compiler 自动 scalar DCCI 和
kernel-end DCCI。构建脚本检查并得到：

```text
[CHECK] S0 source closure and publication sequence
[CHECK] bitcode contains SIMT launch, 64-bit CAS, fence, DCCI matrix
        and V/S completion intrinsics
[CHECK] ELF exports only AIV Main entry; SIMT entry is local;
        metadata is MIX_AIV_MAIN [0:1]
```

具体证据为：

- source publication 顺序固定为 payload stores、可选 writer DCCI、
  `asc_threadfence`、`BUILDING -> BUILT` CAS；
- optimized bitcode 可以由 `llvm-bcanalyzer` 完整解析，包含
  `store.vfsimt.info`、`get.TID.X`、`atom.CAS.G.u64`、
  `fence.workitems`、`DCCI.DST`、`SET/WAIT.FLAG.IMM`、`LD/ST.DEV`；
- 最终 ELF 只有一个非空 GLOBAL entry：
  `simt_cross_core_s0_0_mix_aiv`；
- `S0SimtBuild..._simt_entry` 是唯一非空 LOCAL SIMT function，没有被注册为
  第二个 device entry；
- ELF 无 undefined GLOBAL、无 relocation，且只有预期 metadata section；
- CANN `msobjdump` 将 metadata 解码为 `MIX_AIV_MAIN`、ratio `[0:1]`。

当前环境的 `/opt/mlir-debug/llvm-dis` 不能读取 CCEC 新 SIMT bitcode 中的
`amdgpu_cs_chain`，会以“需要 `llvm.amdgpu.cs.chain`”为由拒绝模块。这里没有
把工具版本不兼容伪写成逐 SSA IR PASS；S0 静态门槛使用可成功解析的
`llvm-bcanalyzer` intrinsic inventory、source 顺序、ELF symbol/metadata 和
真实 A5 动态结果。后续若取得匹配 CCEC 的 `llvm-dis`，再补逐 SSA 检查。

### 3.5 真实 A5 命令、环境与结果

运行硬件前已执行：

```bash
.claude/skills/onboard-arch-precheck/check.sh a5
command -v task-submit
```

当前 shell 没有 `npu-smi`，precheck 因而无法自动识别 silicon；也没有
`task-submit`。检查时只有 `/dev/davinci0`，没有发现当前用户可见的计算进程，
因此按仓库无队列降级规则在 device 0 未加锁串行运行。ACL runtime 在实际
打开 device 后报告 `Ascend950PR_958b`，host 也显式拒绝非 `Ascend950*`
设备，补上了真实 A5 身份证据。

最终命令：

```bash
timeout --foreground 180s \
  tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-s0 --device 0 --runs 100
```

100 次地址复用、每次四种模式的完整结果：

| 模式 | control/CAS/64-thread | payload | 是否稳定 | S0 结论 |
| --- | ---: | ---: | --- | --- |
| `NO_DCCI` | 100/100 | 1/100 | 否 | 只有冷 cache 首轮成功，地址复用后不可靠，淘汰。 |
| `WRITER_DCCI` | 100/100 | 0/100 | 稳定失败 | 只处理 writer 不能让 Scalar reader 的旧 line 失效，淘汰。 |
| `READER_DCCI` | 100/100 | 100/100 | 是 | 当前同 AIV 的最小可靠序列。 |
| `WRITER_AND_READER_DCCI` | 100/100 | 100/100 | 是 | 保守序列通过，但比最小序列多 writer DCCI。 |

最终输出：

```text
[DEVICE] id=0 soc=Ascend950PR_958b
[CHARACTERIZATION] same-AIV minimum=READER_DCCI;
                   cross-AIV/AIC remains unresolved
[PASS] S0 A5 SIMT-GM visibility runs=100 modes=4 reused_address=yes
```

四种模式的 control、SIMT reserve/publish CAS、Main Scalar claim/done CAS、
duplicate reserve 拒绝和 64 个 thread oracle 全部 100/100。thread/report 由
Main Scalar 用 `ld_dev` 读取，payload 才走被测的普通 GM load，避免诊断数据
自己的 cache 行为污染四组 payload 结论。control 独占第一条 64 B，payload
独占第二条 64 B，reader DCCI 不接触 atomic-only control line。

### 3.6 失败现象的边界解释

`NO_DCCI` 的 1/100 不是“偶尔大概率能用”：成功样本正好是同地址尚未被
Scalar reader 缓存的首轮，之后更换 nonce 便持续读到旧 line。因此重复地址
场景必须淘汰该路径。`WRITER_DCCI` 的 0/100 也不能外推为“writer DCCI 在
所有跨核方向都无效”；它只证明 writer 操作不能替代当前 Scalar reader 对
自身旧 payload line 的失效处理。

S0 唯一能冻结的硬件规则是：同 AIV、同 GM 地址重复复用时，Claim winner
成功把 `BUILT -> CLAIMED` 线性化后，对 payload 单行执行 reader DCCI 和 DSB，
再做普通 GM load，可以稳定得到新 payload。AIV0 到其他 AIV、AIV0 到 AIC
可能具有不同 cache 拓扑，S1/S2 必须各自重复四模式实验，不能直接复用这个
结论。

### 3.7 阶段结论与提交

S0 已闭合 CPU 状态机、直接 CCEC SIMT 语法、64-thread execution、64-bit GM
CAS 返回值、V->S completion、同 AIV 地址复用和 DCCI 最小序列。没有写入
性能数字，也没有生成泳道文件。阶段作为本次独立中文提交交付；提交 SHA 以
本分支紧随 D0 的 S0 commit 为准，下一阶段记录会回填该固定 SHA。

S1 将进入同一次 mixed launch：AIV0 仍只做 builder，至少一个其他 AIV 只做
单 Vector task executor，并在真实跨 AIV 方向重新执行 DCCI 可见性矩阵和
golden 校验。

## 4. 阶段状态索引

| 阶段 | 状态 | 结果/提交 |
| --- | --- | --- |
| D0 文档与查证 | 完成 | `64e3d5d5`：范围、链接和空白检查通过。 |
| S0 基础协议与 SIMT 自检 | 完成 | CPU 三套 PASS；A5 同 AIV 100×4 模式完成，reader DCCI 为最小可靠序列；随本次 S0 提交交付。 |
| S1 单 Vector task | 未开始 | - |
| S2 单 Cube task | 未开始 | - |
| S3 Vector + Cube | 未开始 | - |
| S4 多 task、单 builder | 未开始 | - |
| G0 GM 完整 PA | 未开始 | - |
| G1 双 builder GM | 未开始 | - |
| U0 UBUF 单槽 | 未开始 | - |
| U1 UBUF 多槽/多 task | 未开始 | - |
| U2 UBUF 完整 PA | 未开始 | - |
