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
- 阶段性本地 commit 完成后自动继续，不等待人工确认；未经用户明确授权不得
  push；
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
先用于建立最简单的正确性基线。D0 原设想将 UBUF 用于 MTE3 暂存
和性能探索；U0 实际查证后确认当前 VF 无 SIMT-native MTE3，
改为了纯 SIMT 读 UBUF 后直接写 GM，详见 12.1。

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
- 普通 `__aicore__` Scalar 域的
  `copy_ubuf_to_gm_align_v2` 与 V/MTE3/S event intrinsic。

最后一项不是 SIMT VF 能力。D0 当时只证明本机头文件有该
intrinsic；U0 后续用最小 CCEC 探针确认 VF 不能调用它，详见
12.1。

官方 `asc_vf_call` 文档确认：混合编程中的 SIMT VF 由 `__aicore__` 调用，
线程总数不超过 2048，VF 只能接收 raw pointer 和基础标量。官方语法限制还
确认：

- 指针形参必须明确为 `__gm__` 或 `__ubuf__`；
- 不能把栈数组、结构体或间接函数指针传入 SIMT VF；
- 混合场景不支持直接对 GM/UBUF 结构体整体赋值；
- 因此 task payload 必须逐字段或逐基础 word 构造。

官方混合编程示例给出的 UB 预算为 256 KB 总量，除编译器预留空间外，
SIMT 至少需要 32 KB Data Cache。官方文档证明了 `__ubuf__ *`
参数与 VF 启动边界；它没有证明 SIMT VF 可以发起 MTE3。

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
直接遍历和写 GM，也可以消费 kernel entry 在 VF 发起前填好的 UBUF
参数。该 entry 只是工具链要求的启动载体，不是 task builder。它们是接口
和时序参考，不会成为新目录的源码依赖；其中 AscendC 的 `TPipe`、
`LocalTensor`、`DataCopy` 也不会搬入直接 CCEC 实现。

### 2.6 当前只形成设计、尚未形成硬件结论的事项

以下内容必须由 S0/S1/U0 的 CCEC 产物和真实 A5 动态结果回答：

1. AIV entry 发起真实 SIMT VF 后，哪一组 V/S event 能可靠证明 VF 完成；
2. SIMT 普通 GM store、thread fence、64-bit CAS 发布后，其他 AIC/AIV
   executor 是否能稳定读取新 payload；writer/reader 哪一侧需要 DCCI
   不能预设，由四组最小对照决定；
3. 上述 GM 可见性在多次 kernel launch、复用同一地址时是否仍成立；
4. SIMT 64-bit CAS 与 executor Scalar 64-bit CAS 竞争同一 control 时的返回值
   和全序是否一致；
5. 实际执行 SIMT builder 后，最终 AIV ELF 是否稳定产生 MIX VF metadata，
   且不会额外导出 SIMT entry；
6. UBUF 发布 transport 的真实工具链能力；U0 已确认当前 VF
   无 SIMT-native MTE3，改用同一 leader 读 UBUF 后直接写 GM；
7. 直接 GM word-store transport 的有效字节、对齐和尾部行为
   是否满足最大执行包；
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
| ---- | ------------------------ | ------------------------- |
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
| ---- | --------------------: | ------: | -------- | ------- |
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
性能数字，也没有生成泳道文件。阶段已由提交 `399d5704` 独立交付并推送。

S1 随后进入同一次 mixed launch：AIV0 仍只做 builder，AIV1 只做单 Vector
task executor，并在真实跨 AIV 方向重新执行 DCCI 可见性矩阵和 golden 校验。

## 4. 2026-08-05：S1 单 Vector task

### 4.1 目标、拓扑与任务合同

S1 不扩展到多 task，也不引入 Cube。host 固定 launch 一个 mixed block，
metadata 比例为 1 AIC + 2 AIV：

| 参与者 | 固定职责 | 明确禁止 |
| ------ | -------- | -------- |
| AIC | 只观察 `DONE` 并报告身份。 | 不 Build、不 Claim、不执行 Vector。 |
| AIV0 / owner 32 | 启动 64-thread SIMT VF；thread 0 构建唯一 task。 | 不 Claim、不执行 task。 |
| AIV1 / owner 33 | 轮询 `BUILT`，唯一 Claim，校验 payload，执行 Vector add，完成后发布 `DONE`。 | 不参与 Build。 |

task payload 独占一条 64 B cacheline，共 8 个 `uint64_t`：magic、version、
launch nonce、input A/B/output 的 GM 地址、`task_id + element_count` 和覆盖前
7 个 word 的 checksum。状态仍严格使用
`EMPTY -> BUILDING -> BUILT -> CLAIMED -> DONE`；`BUILT` 标记 AIV engine
和一条 payload line。AIV1 只有 CAS 成功后才能按被测 DCCI 模式读取 payload。

真实任务采用 128×128 float tile，共 16,384 个元素。AIV1 按以下顺序执行：

```text
TLOAD input A/B
  -> MTE2->V wait
  -> TADD
  -> V->MTE3 wait
  -> TSTORE output
  -> MTE3->S wait
  -> CLAIMED->DONE CAS
```

因此 `DONE` 是 GM output 已完成写回后的边界，不是仅完成 Vector 指令发射。
四条 64 B guard 分隔 control、输入与输出；host 还逐元素检查两个输入未改写。

### 4.2 新增文件与入口

- `gm/common/s1_vector.h`：S1 独立 ABI、角色/状态常量、8-word payload、
  128 B role result、128×128 tile 和 host golden；
- `gm/test/test_s1_vector.cpp`：CPU 半包暂停、角色互斥、Claim、畸形
  descriptor fail-closed 与逐元素 golden；
- `gm/cpu/build_s1.sh`：optimized、ASan/UBSan、TSan 三套门槛；
- `gm/ccec/s1_vector_kernel.cpp`：AIC observer、AIV0 SIMT builder、AIV1
  Vector executor 两个 mixed entry；
- `gm/ccec/s1_vector_host.cpp`：ACL loader、同地址复用、角色/guard/payload/
  golden oracle 和四模式分类；
- `gm/ccec/build_s1.sh`：源码顺序、bitcode intrinsic、mixed ELF、metadata、
  UB 预算与 host 构建检查；
- `run.sh`：新增 `build-s1` 与 `run-s1` 统一入口。

所有源文件仍只位于 `simt_cross_core/`。构建脚本扫描 include，确认没有引用
`cross_core` 或 `ops-nn` 源码；PTO/CANN 只作为已安装工具链头使用。

### 4.3 CPU 结果

统一命令：

```bash
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-s1
```

CPU 部分结果：

```text
[PASS] S1 CPU vector rounds=64: optimized
[PASS] S1 CPU vector rounds=32: ASan+UBSan
[PASS] S1 CPU vector rounds=16: TSan
```

每轮使用显式暂停点让 AIV0 在写完 4/8 payload word 后停住，验证：

1. `BUILDING` 期间 AIV1 Claim 失败且 payload read 为 0；
2. AIC、AIV0 调用 execute route 均被拒绝，AIC、AIV1 调用 build route 均被
   拒绝；
3. AIV0 恢复后唯一发布 `BUILT`，AIV1 唯一 Claim 并只读取 8 个 word；
4. AIV1 精确执行一次 16,384-element add，逐元素 golden 相等；
5. checksum 被破坏的已领取 descriptor 不执行 Vector，output 全部保持 sentinel，
   但探针可有界发布 `DONE`，便于将候选可见性失败归类而不制造设备死锁。

CPU 仍只证明 C++ 语义和角色路由，不替代 A5 DCache 证据。

### 4.4 CCEC 与 mixed ELF 静态门槛

CCEC 分别使用 `dav-c310-cube` 和 `dav-c310-vec` 编译同一源码，显式关闭
compiler 自动 scalar DCCI 与 kernel-end DCCI。检查结果为：

```text
[CHECK] S1 source closure and publication/completion order
[CHECK] bitcode contains SIMT publication, atomic polling, DCCI and
        real Vector-add intrinsics
[CHECK] ELF exports only two mixed entries; AIV is SIMD_SIMT_MIX_VF
        and UB budget is 200/224 KiB
[BUILD] S1 CCEC complete
```

具体证据：

- source 顺序锁定 payload store、可选 writer DCCI、SIMT fence、`BUILT` CAS；
- source 顺序还锁定 `TLOAD < TADD < TSTORE < MTE3->S wait`，且
  `RunVectorAdd < DONE CAS`；
- AIV bitcode inventory 同时包含 SIMT launch/TID、64-bit CAS、atomic load、
  work-item fence、DCCI/DSB、V/S event、`vldsx1/vadd/vstsx1` 和 `st_dev`；
- 最终 ELF 只有 `simt_cross_core_s1_0_mix_aic` 与
  `simt_cross_core_s1_0_mix_aiv` 两个 GLOBAL function；SIMT builder 与
  Vector add 都是非空 LOCAL function，且无 undefined GLOBAL 与 relocation；
- 两个 metadata 都是 `MIX_AIC_MAIN [1:2]`；AIV metadata 的 TLV12 为 4，
  即 `SIMD_SIMT_MIX_VF`，不是会拒绝 SIMD 的 `SIMT_VF_ONLY`；
- AIV metadata 的 SIMT share memory 为 8 KiB；三块 64 KiB Vector tile 共
  192 KiB，总计 200 KiB，不超过保留 32 KiB DCache 后的 224 KiB 上限。

与 S0 一样，新 SIMT bitcode 使用当前 `/opt/mlir-debug` 的 `llvm-dis` 不支持的
`amdgpu_cs_chain`。因此仍使用可成功解析的 `llvm-bcanalyzer` 做 intrinsic
inventory，并用源码顺序检查补足发布/完成顺序；没有伪称完成逐 SSA IR 审计。

### 4.5 上板前发现并修正的诊断发布问题

首版 role result 使用 256 B 局部 aggregate，再将其 `reinterpret_cast` 为
`uint64_t*` 循环执行 `st_dev`。真实 A5 上 task 主链和 Vector golden 已完成，
但 D2H 后 role 字段发生重排，例如 magic 位置读到 status，说明该通用局部地址
遍历不能作为 CCEC Scalar 结果 ABI。把结构单纯缩到 128 B 后问题仍存在，因而
“只是结构太大”被排除。

最终保留 128 B 打包 ABI，并把 16 个命名字段逐一 `st_dev`，不再线性遍历
Scalar 局部 aggregate。修正后所有字段连续稳定。现有证据能确定故障边界和
有效规避方式，但没有编译器内部证据精确断言是哪一步 scalar replacement 或
spill 造成重排；过程文档不把推测写成根因。该修正只影响诊断发布，不改变
task 状态机、DCCI 矩阵或 Vector 执行。

### 4.6 真实 A5 四模式结果

precheck 再次报告当前 shell 无 `npu-smi`，且没有 `task-submit`；
`/dev/davinci0` 存在，按仓库无队列降级规则直接在 device 0 串行运行。ACL
实际报告 `Ascend950PR_958b`。最终命令：

```bash
timeout --foreground 180s \
  tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-s1 --device 0 --runs 100
```

同一块约 193 KiB device state 地址复用 100 次、每次四模式：

| 模式 | 状态机/角色/guard | payload + 16,384 元素 golden | 稳定性 | S1 结论 |
| ---- | ----------------: | ---------------------------: | ------ | ------- |
| `NO_DCCI` | 100/100 | 1/100 | 不稳定 | 仅冷 cache 首轮成功，淘汰。 |
| `WRITER_DCCI` | 100/100 | 0/100 | 稳定失败 | writer clean 不能替代 AIV1 reader 失效，淘汰。 |
| `READER_DCCI` | 100/100 | 100/100 | 稳定通过 | AIV0→AIV1 的最小可靠序列。 |
| `WRITER_AND_READER_DCCI` | 100/100 | 100/100 | 稳定通过 | 保守通过，但 writer DCCI 冗余。 |

最终输出：

```text
[DEVICE] id=0 soc=Ascend950PR_958b topology=1AIC+2AIV
[SUMMARY] mode=NO_DCCI                protocol=100/100 payload+golden=1/100
[SUMMARY] mode=WRITER_DCCI            protocol=100/100 payload+golden=0/100
[SUMMARY] mode=READER_DCCI            protocol=100/100 payload+golden=100/100
[SUMMARY] mode=WRITER_AND_READER_DCCI protocol=100/100 payload+golden=100/100
[CHARACTERIZATION] AIV0-builder -> AIV1-executor minimum=READER_DCCI;
                   AIV-to-AIC remains unresolved
[PASS] S1 mixed single-Vector runs=100 modes=4 reused_address=yes
       golden_elements=16384
```

四模式中 AIV0 都精确 Build 一次、Claim/Vector 为 0；AIV1 都精确 Claim 一次、
Build 为 0，仅在 payload 有效时执行一次 Vector；AIC 的 Build/Claim/Vector 均
为 0。所有 control 最终为 `DONE`、fatal 为 0、guard 和输入未改写。候选模式
payload 失败时 output 保持 sentinel，因此不会把“误用旧 descriptor 后碰巧算
对”计为通过。

### 4.7 阶段结论

S1 已在同一次 1:2 mixed launch 内证明 AIV0 builder-only、AIV1
executor-only 和真实 Vector task 完成边界。跨 AIV 的 DCache 行为与 S0 同 AIV
一致：正式 GM 路径必须在 Claim winner 侧对 payload 执行 reader DCCI + DSB，
writer DCCI 不是必要条件。这个结论仍不能替代 AIV→AIC 证据；S2 将用单 Cube
task 和同一四模式矩阵独立验证。

## 5. 2026-08-05：S2 单 Cube task

### 5.1 目标、拓扑与任务合同

S2 仍只启动一个 `1 AIC + 2 AIV` mixed block，但把执行方向从 S1 的跨 AIV
改为 AIV0 到 AIC：

| 参与者 | 固定职责 | 明确禁止 |
| ------ | -------- | -------- |
| AIC / owner 0 | 等待 `BUILT`，唯一 Claim，校验 payload，执行 Cube matmul，等待 FIX 写回后发布 `DONE`。 | 不参与 Build。 |
| AIV0 / owner 32 | 启动 64-thread SIMT VF，由 thread 0 构建唯一 Cube task。 | 不 Claim、不执行 Cube/Vector task。 |
| AIV1 / owner 33 | 只等待并观察 `DONE`。 | 不 Build、不 Claim、不执行 task。 |

task descriptor 继续使用独占一条 64 B cacheline 的 8 个 `uint64_t`，字段为
magic、version、launch nonce、input A/B/output 的 GM 地址、task shape 和
checksum。状态严格保持
`EMPTY -> BUILDING -> BUILT -> CLAIMED -> DONE`；`BUILT` 的 engine 明确为
AIC，AIC 只有在 Claim CAS 成功后才允许读取 descriptor。

真实 Cube workload 为 128×128 float matmul。输入 A 是行权重不同的对角矩阵，
`A[row,row] = row + 1`；输入 B 同时依赖 row、column 和 launch nonce，避免全
1 输入掩盖转置、错行或复用旧输入。因而 host 可用
`output[row,column] = (row + 1) * B[row,column]` 做逐元素精确 golden，
同时检查 A/B 保持不变。AIC 的完成顺序为：

```text
TLOAD A/B
  -> MTE2->MTE1 wait
  -> TMOV A/B to L0A/L0B
  -> MTE1->M wait
  -> TMATMUL
  -> M->FIX wait
  -> TSTORE output
  -> FIX->S wait
  -> CLAIMED->DONE CAS
```

所以 `DONE` 位于真实 Cube 结果写回 GM 之后，不能把指令发射完成误当成 task
完成。

### 5.2 修改文件与公共 ABI 收敛

新增：

- `gm/common/gm_probe_support.h`：抽取 S1/S2 共用的 DCCI 模式、identity、
  role count、descriptor checksum、guard 和 64/128 B 诊断 ABI；
- `gm/common/s2_cube.h`：S2 角色、状态、Cube descriptor、三块 64 KiB tile
  和非对称 golden；
- `gm/test/test_s2_cube.cpp`：CPU 半包暂停、角色互斥、唯一 Claim、畸形
  descriptor fail-closed 和真实 CPU matmul；
- `gm/cpu/build_s2.sh`：optimized、ASan/UBSan、TSan 三套 CPU 门槛；
- `gm/ccec/s2_cube_kernel.cpp`：AIV0 SIMT builder、AIC Cube executor、AIV1
  observer 和两个 mixed entry；
- `gm/ccec/s2_cube_host.cpp`：ACL loader、四模式重复地址探针、角色/guard/
  输入/descriptor/Cube golden oracle；
- `gm/ccec/build_s2.sh`：发布及完成顺序、双核型 bitcode、ELF symbol、
  metadata 和 host 构建门槛。

修改：

- `gm/common/s1_vector.h` 改为复用 `gm_probe_support.h`，保留 S1 自己的状态、
  Vector tile 和 golden，不复制第二套公共 ABI；
- `gm/ccec/s1_vector_kernel.cpp` 适配公共 identity 的显式整数入参；
- `gm/ccec/s1_vector_host.cpp` 把阶段边界改成“S1 本身不测试 AIC read”，
  不再把已经由 S2 闭合的方向写成全局未决；
- `run.sh` 增加 `build-s2`、`run-s2`，并把 S1/S2 共用路径命名收敛为
  `GM_ROOT`、`GM_BUILD`；
- 两份中文文档记录 S2 的实现、证据和最终 GM 可见性规则。

公共抽取没有改变 S1 的 state layout、状态值、任务数或行为。S2 的
`ProbeState` 为 197,568 B，其中 A、B、output 各为 65,536 B，并用四条 64 B
guard 隔离。所有源码仍位于 `simt_cross_core/`；构建门槛确认没有 include
`cross_core` 或本机 `ops-nn` 源码。

S2 的 AIV 运行时没有普通 Vector task，但 1:2 mixed entry 仍必须由 compiler
标记为 `SIMD_SIMT_MIX_VF`，不能退化成 `SIMT_VF_ONLY`。因此源码保留一个
本地 SIMD metadata anchor；它只位于运行时不进入的诊断分支，不改变 AIV0
builder-only 与 AIV1 observer-only 的角色计数。最终 ELF 同时检查该本地
SIMD function 和本地 SIMT entry 都未成为额外全局入口。

### 5.3 CPU 命令与结果

统一命令：

```bash
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-s2
```

CPU 部分实际结果：

```text
[PASS] S2 CPU cube rounds=16: optimized
[PASS] S2 CPU cube rounds=8:  ASan+UBSan
[PASS] S2 CPU cube rounds=4:  TSan
```

每轮由 AIV0 builder 在写完 4/8 descriptor word 时进入显式暂停点，依次证明：

1. `BUILDING` 期间 AIC Claim 失败，且 payload read 计数保持 0；
2. AIC 和 AIV1 不能 Build，AIV0 和 AIV1 不能执行 Cube；
3. AIV0 恢复后唯一发布 `BUILT`，只有 AIC 能唯一 Claim；
4. AIC 精确读取 8 个 descriptor word、执行一次 128×128 CPU matmul，并
   得到 16,384 个精确 golden；
5. checksum 被破坏，或者 output 地址被替换后重新计算出合法 checksum 时，
   AIC 都可以有界收口到 `DONE`，但 Cube execute 计数为 0，完整 output 仍为
   sentinel；checksum 不能替代执行地址白名单校验。

CPU triple-loop 只用于协议和 golden 的 portable oracle，不代表 A5 Cube
实现，也没有拿 CPU 时间估算设备性能。

### 5.4 CCEC、bitcode 与 mixed ELF 静态门槛

同一 `s2_cube_kernel.cpp` 分别用 `dav-c310-cube`、`dav-c310-vec` 编译，并
显式关闭 compiler 自动 scalar DCCI 和 kernel-end DCCI。完整门槛通过：

```text
[CHECK] S2 source closure and publication/completion order
[CHECK] bitcode contains SIMT publication and real Cube
        load/move/matmul/fix intrinsics
[CHECK] ELF exports only two mixed entries; AIV metadata is
        SIMD_SIMT_MIX_VF with 8 KiB SIMT share
[BUILD] S2 CCEC complete
```

静态证据包括：

- source 顺序锁定 payload stores、可选 writer DCCI、SIMT thread fence、
  `BUILDING -> BUILT` CAS；
- Cube source 顺序锁定 `TLOAD < TMOV < TMATMUL < TSTORE < FIX->S wait`，
  且 `RunCubeMatmul < DONE CAS`；
- AIV optimized bitcode 包含 SIMT launch/TID、64-bit CAS/atomic load、
  work-item fence、DCCI/DSB、V/S event、`st_dev` 和 SIMD anchor；
- AIC optimized bitcode 包含 64-bit CAS/atomic load、DCCI/DSB、MTE2、
  ND2NZ、L1→L0A/L0B、`MAD.f322f32.c310`、FIX 写回和 event intrinsic；
- 最终 ELF 只导出 `simt_cross_core_s2_0_mix_aic` 与
  `simt_cross_core_s2_0_mix_aiv` 两个非空 GLOBAL function；
- `RunCubeMatmul`、SIMT builder `_simt_entry` 和 SIMD anchor 都是非空
  LOCAL function；ELF 没有 undefined GLOBAL 或 relocation；
- 两份 metadata 均为 `MIX_AIC_MAIN [1:2]`；AIV metadata 的 TLV12 为 4，
  即 `SIMD_SIMT_MIX_VF`，SIMT share memory 为 8 KiB。

新 SIMT bitcode 仍由能完整解析它的 `llvm-bcanalyzer` 做 intrinsic inventory；
当前 `/opt/mlir-debug/llvm-dis` 的版本限制与 S0/S1 相同，本文没有把 inventory
冒充为逐 SSA IR 证明。

### 5.5 真实 A5 四模式结果

按仓库流程先运行 A5 precheck。当前 shell 仍没有 `npu-smi` 和
`task-submit`，因此 precheck 无法从 CLI 自动识别 silicon；本轮沿用用户已经
明确授权的 device 0 未加锁直跑路径。ACL 实际打开设备后报告
`Ascend950PR_958b`，host 会拒绝非 `Ascend950*` SoC。命令为：

```bash
timeout --foreground 300s \
  tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-s2 --device 0 --runs 100
```

同一块 197,568 B device state 地址复用 100 次、每次四模式：

| 模式 | 状态机/角色/guard/输入 | payload + 16,384 元素 Cube golden | S2 结论 |
| ---- | ---------------------: | --------------------------------: | ------- |
| `NO_DCCI` | 100/100 | 1/100 | 仅冷 cache 首轮成功，重复地址不可靠。 |
| `WRITER_DCCI` | 100/100 | 0/100 | writer clean 不能替代 AIC reader 失效。 |
| `READER_DCCI` | 100/100 | 100/100 | AIV0→AIC 的最小可靠序列。 |
| `WRITER_AND_READER_DCCI` | 100/100 | 100/100 | 保守通过，但 writer DCCI 冗余。 |

最终输出：

```text
[DEVICE] id=0 soc=Ascend950PR_958b topology=1AIC+2AIV
[SUMMARY] mode=NO_DCCI                protocol=100/100 payload+golden=1/100
[SUMMARY] mode=WRITER_DCCI            protocol=100/100 payload+golden=0/100
[SUMMARY] mode=READER_DCCI            protocol=100/100 payload+golden=100/100
[SUMMARY] mode=WRITER_AND_READER_DCCI protocol=100/100 payload+golden=100/100
[CHARACTERIZATION] AIV0-builder -> AIC-executor minimum=READER_DCCI
[PASS] S2 mixed single-Cube runs=100 modes=4 reused_address=yes
       golden_elements=16384
```

所有模式中 AIV0 都精确 Build 一次且 Claim/task execute 为 0；AIC 都精确
Claim 一次且 Build 为 0，只在 descriptor 有效时执行一次 Cube；AIV1 的
Build/Claim/task execute 均为 0。control 全部收口到 `DONE`，fatal 为 0，
descriptor 的 host 最终值、SIMT report、四条 guard 和两块输入逐项通过。
候选模式读取旧 descriptor 时不执行 Cube，output 保持 sentinel，因此
`payload+golden` 失败不是矩阵数值误差。

### 5.6 S1 公共抽取回归

公共 ABI 抽取后重新执行完整 `build-s1`，optimized、ASan/UBSan、TSan、
CCEC bitcode、mixed ELF、metadata 和 UB 预算门槛均通过；随后在同一设备上
重新运行 S1 的 100×4 模式。结果仍为：

| 模式 | S1 状态机/角色/guard | S1 payload + Vector golden |
| ---- | -------------------: | -------------------------: |
| `NO_DCCI` | 100/100 | 1/100 |
| `WRITER_DCCI` | 100/100 | 0/100 |
| `READER_DCCI` | 100/100 | 100/100 |
| `WRITER_AND_READER_DCCI` | 100/100 | 100/100 |

这证明公共抽取没有改变 S1 的设备行为，也让同 AIV、跨 AIV、AIV→AIC 三条
证据获得同一组 DCCI 模式定义和诊断 ABI。

### 5.7 阶段结论与边界

S2 已闭合 AIV0 builder-only、AIC executor-only、AIV1 observer-only、真实
Cube 完成边界和跨引擎 descriptor 可见性。三类现有方向在当前 A5 上得到完全
一致的四模式计数，因此 GM 后续阶段统一使用 Claim winner 侧
`reader DCCI + DSB`；不保留无收益的 writer DCCI。

该结论不是“GM 一定不需要/一定需要 DCCI”的通用硬件定律，只适用于本探针
已经覆盖的 compiler 配置、普通 GM payload、发布 CAS、同地址复用和 reader
类型。S2 没有采性能数据，也没有生成泳道图。设备运行未经过 `task-submit`
锁，正确性计数完整，但不能把本轮墙钟用作性能基线。

S2 随本节所在提交独立交付；下一阶段 S3 将在同一次 mixed launch 中同时
发布一个 Vector task 和一个 Cube task，验证 engine 路由、两个唯一 Claim、
完成计数和 drain，而不直接跳到完整 PA。

## 6. 2026-08-05：S3 Vector + Cube 双 task

### 6.1 目标、拓扑与双 slot 合同

S3 首次在同一次 mixed launch 中同时构建和执行两个不同 engine 的 task：

| 参与者 | 构建 | 执行 | drain 前必须证明 |
| ------ | ---- | ---- | ---------------- |
| AIV0 / owner 32 | SIMT thread 0 构建 Vector，thread 1 构建 Cube。 | 0 个 task。 | 两个 SIMT report 均成功，随后发布 `builder_finished=1`。 |
| AIV1 / owner 33 | 0 个 task。 | 只 Claim 并执行 Vector add。 | MTE3→S、Vector `DONE`、`done_count + 1`。 |
| AIC / owner 0 | 0 个 task。 | 只 Claim 并执行 Cube matmul。 | FIX→S、Cube `DONE`、`done_count + 1`。 |

Vector 和 Cube 各有独立的 64 B control 与 64 B payload。SIMT 两个 thread
分别执行自己的 `EMPTY -> BUILDING -> BUILT`，不拼接同一个 descriptor；
所以一个 slot 停在半包时，另一 slot 可以独立发布和执行。两个 executor
使用 S0～S2 已冻结的 reader DCCI + DSB，不再重复运行已经淘汰的无 DCCI 和
writer-only 模式。

全局 drain 独占一条 64 B cacheline。AIV0 只有在 SIMT invoke 的 V→S wait
完成并检查两份 report 后才能发布 `builder_finished=1`；每个 executor 只有
在 workload 写回、task `DONE` 后才能原子增加 `done_count`。AIV0、AIV1、
AIC 最终都必须观察到：

```text
builder_finished == 1 && done_count == 2
```

任一计数越界、状态错误、payload 错误或 timeout 都发布首错 fatal，不能靠
放宽 drain 让错误路径伪装成通过。

### 6.2 新增文件与入口

- `gm/common/s3_dual_task.h`：两条 task 状态、双 descriptor、drain、角色、
  六块 64 KiB 数据区、Vector/Cube 非对称 golden 和完整 GM ABI；
- `gm/test/test_s3_dual_task.cpp`：CPU 半包交错、engine 路由、双执行、精确
  drain、地址替换拒绝和双 golden；
- `gm/cpu/build_s3.sh`：optimized、ASan/UBSan、TSan 三套 CPU 门槛；
- `gm/ccec/s3_dual_task_kernel.cpp`：两 thread SIMT builder、AIV1 Vector、
  AIC Cube、完成计数和 mixed entry；
- `gm/ccec/s3_dual_task_host.cpp`：ACL loader、同地址重复运行、两份 report、
  两个 descriptor、三角色、drain、guard、输入和双 golden oracle；
- `gm/ccec/build_s3.sh`：双发布/完成顺序、双核型 bitcode、ELF、metadata、
  AIV 本地内存预算和 host 构建门槛；
- `run.sh`：增加 `build-s3`、`run-s3` 统一入口；
- 两份中文文档：冻结 S3 合同并记录实际证据。

所有文件仍位于 `simt_cross_core/`。S3 复用 `gm_probe_support.h` 的 control、
fatal、SIMT report、role result、identity、计数和 checksum 基础函数，没有复制
第三套公共诊断 ABI，也没有 include `cross_core` 或本机 `ops-nn` 源码。

设备 `ProbeState` 为 394,624 B，包含六块 65,536 B 输入/输出 tile、七条
64 B guard、两个 128 B task slot、两份 SIMT report、三份 role result 和
一条 drain cacheline。host 每轮完整重置并复用同一 device 地址。

### 6.3 CPU 命令与结果

统一命令：

```bash
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-s3
```

CPU 三套实际结果：

```text
[PASS] S3 CPU dual-task rounds=16: optimized
[PASS] S3 CPU dual-task rounds=8:  ASan+UBSan
[PASS] S3 CPU dual-task rounds=4:  TSan
```

受控交错先让 Vector builder 写完 4/8 word 后暂停，再完整发布 Cube。此时验证：

1. Vector control 仍为 `BUILDING`，AIV1 对它的 Claim 失败且不读半包；
2. Cube 已为 `BUILT`，AIC 可以独立 Claim、执行一次 matmul、发布 Cube
   `DONE`，`done_count` 精确为 1；
3. AIC 不能执行 Vector，AIV1 不能执行 Cube，错误 engine 不产生 Claim；
4. 即使 Cube 已完成，`builder_finished=0` 时 drain 仍不能通过；
5. 恢复 Vector 发布并设置 `builder_finished=1` 后，drain 仍等待第二个 task；
6. AIV1 完成 Vector 并把 `done_count` 增至 2 后，drain 才通过；
7. 两个 task 合计 Build 2/2、Claim 2/2、payload read 16 word、Vector/Cube
   execute 各一次，两个 output 共 32,768 个元素逐项等于 golden；
8. Vector、Cube 各自的 output 地址被替换并重新计算合法 checksum 时仍被
   拒绝，证明 checksum 不能取代执行地址白名单。

### 6.4 CCEC、bitcode 与 mixed ELF 门槛

CCEC 分别以 `dav-c310-cube`、`dav-c310-vec` 编译同一内核源码。统一构建
实际通过：

```text
[CHECK] S3 source closure and dual-task publication/drain order
[CHECK] bitcode contains two-thread SIMT publication, Vector add,
        Cube matmul and drain atomics
[CHECK] ELF has two mixed entries, local SIMT/Vector/Cube functions
        and 200/224 KiB AIV budget
[BUILD] S3 CCEC complete
```

门槛具体锁定：

- 两个 SIMT thread 各自写完 8-word payload、执行 thread fence 后才发布
  对应 `BUILT`；SIMT invoke 的 V→S wait 先于 `builder_finished`；
- Vector 的 `TLOAD < TADD < TSTORE < MTE3->S`，Cube 的
  `TLOAD < TMOV < TMATMUL < TSTORE < FIX->S` 都先于各自 `CompleteTask`；
- `CompleteTask` 内严格为 `CLAIMED -> DONE` CAS 在前、`done_count` 原子
  加一在后；
- AIV bitcode 同时包含 SIMT launch/TID、CAS/atomic-add、thread fence、
  reader DCCI、Vector load/add/store 和 event intrinsic；
- AIC bitcode 同时包含 CAS/atomic-add、reader DCCI、MTE2、ND2NZ、
  L1→L0A/L0B、Cube MAD、FIX 写回和 event intrinsic；
- 最终 ELF 只有两个非空 GLOBAL entry；SIMT builder、Vector add、Cube
  matmul 都是非空 LOCAL function，无 undefined GLOBAL 和 relocation；
- 两份 metadata 均为 `MIX_AIC_MAIN [1:2]`；AIV 为
  `SIMD_SIMT_MIX_VF`，SIMT share 为 8 KiB，三块 Vector UB 加 share 仍为
  200/224 KiB。

### 6.5 真实 A5 结果

本轮继续先执行仓库 A5 precheck；当前 shell 无 `npu-smi`、无
`task-submit`，因此沿用用户明确授权的 device 0 未加锁路径。ACL 实际设备
身份为 `Ascend950PR_958b`，host 同样拒绝非 `Ascend950*` SoC。命令：

```bash
timeout --foreground 300s \
  tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-s3 --device 0 --runs 100
```

真实输出：

```text
[DEVICE] id=0 soc=Ascend950PR_958b topology=1AIC+2AIV
         state_bytes=394624 tasks=2
[SUMMARY] reader_dcci protocol+roles+drain+vector+cube=100/100
[PASS] S3 mixed dual-task runs=100 reused_address=yes
       vector_elements=16384 cube_elements=16384
```

100 次同地址复用中，每轮两份 SIMT report 都证明 thread 0/1 分别完成一个
8-word descriptor；AIV0 Build 2/2 且 task execute 为 0；AIV1 和 AIC 各自
Build 为 0、Claim 1/1、execute 一次。两个 control 分别精确为 Vector/Cube
`DONE`，`builder_finished=1`、`done_count=2`、fatal=0，七条 guard、六块
输入/输出和两份 checksum 全部通过。

### 6.6 阶段结论

S3 已证明同一个 AIV0 SIMT launch 可以用不同 thread 独立构建 Vector/Cube
task，并让 AIV1/AIC 在同一次 mixed kernel launch 中按 engine 路由执行。
全局 drain 依赖“builder 已完成发布”和“两个 task 已在真实写回边界后完成”
两项独立事实，不会把单 task 重复完成或 workload 仅发射计作通过。

本阶段没有测量性能，也没有生成泳道图；未加设备锁的墙钟不进入性能结论。
S3 随本节所在提交独立交付。下一阶段 S4 扩展为多 task、单 builder，重点验证
SIMT thread-stride task 扫描、executor token busy、无遗失和更大完成计数。

## 7. 2026-08-05：A0 SIMT atomic 同地址竞争探针

### 7.1 为什么需要独立验证

S0～S3 已经使用 SIMT `asc_atomic_cas`，S3 还使用了 atomic-add，但那些
用例的目标是调度协议：S0 的 CAS 只由 thread 0 执行，S3 的构建线程数也只有
2。它们能证明指令在当前组合路径中可用，但不能单独回答以下问题：

1. 多个 warp 同时 CAS 同一 GM `uint64_t` 时是否恰好一个 winner；
2. winner 和 loser 返回的 old value 是否保留完整 64 bit；
3. 同地址 atomic-add 的返回 ticket 是否不重不漏；
4. 语义能否从单 warp 一直保持到工具链声明的最大线程数。

因此在 S4 之前增加 A0 独立项，只验证调度会用到的 GM `uint64_t`
CAS/add，不把结论外推到 UBUF、其他数据类型或其他 atomic 操作。

### 7.2 接口、warp 与线程上限查证

本阶段先查本机 CANN 9.1.0 weekly 20260708，没有臆想接口：

- `simt_api/device_atomic_functions.h` 明确声明 GM
  `uint64_t asc_atomic_cas(...)` 和 `asc_atomic_add(...)`；
- `device_atomic_functions_impl.h` 分别下降到 `atomicCAS` 与 `atomicAdd`；
- dav_3510 `kernel_simt_warp_level_impl.h` 的 `WARP_SIZE` 为 32；
- dav_3510 `kernel_simt_constant.h` 的 `SIMT_MAX_THREAD_NUM` 为 2048；
- 本机 `ops-nn` 存在多个 1024-thread 实现，也有
  `sparse_tensor_dense_mat_mul` 以 2048 作为 `LAUNCH_BOUND` 和实际
  `dim3` 线程数。

因此没有继续沿用 S0 的 64-thread 最小配置，而是固定四档：

| 线程数 | warp 数 | 覆盖目的 |
| -----: | ------: | -------- |
| 32 | 1 | 单 warp 基本语义。 |
| 64 | 2 | 最小跨 warp 竞争。 |
| 1024 | 32 | 本机算子常用的大并发配置。 |
| 2048 | 64 | 当前 dav_3510 头文件声明上限。 |

### 7.3 探针 ABI 和精确 oracle

`protocol_probe/simt_atomic/` 是独立子目录，但复用本目录已有的
`common/shared_protocol.h` cacheline 对齐定义：

- `common/atomic_probe.h`：固定 32/64/1024/2048 配置、完整
  64-bit initial/desired/ticket/marker 生成规则与 49,856 B host/device ABI；
- `test/test_simt_atomic.cpp`：用 C++ atomic 建立与设备 oracle 一致的
  portable 语义模型；
- `cpu/build.sh`：optimized、ASan/UBSan、TSan 三套门槛；
- `ccec/kernel.cpp`：AIV Main Scalar 发射最多 2048 个 SIMT thread，
  所有 active thread 对同一 CAS cell 和同一 add cell 执行原子指令；
- `ccec/host.cpp`：单次分配 device state，四档与多轮全部复用同一地址，
  逐线程核对 CAS old value、add ticket、marker、inactive tail 和 guard；
- `ccec/build.sh`：锁定源码顺序、CCEC bitcode intrinsic、ELF symbol、
  relocation 和 `MIX_AIV_MAIN [0:1]` metadata；
- `run.sh`：增加 `build-atomic` 与 `run-atomic` 统一入口。

每个 SIMT thread 按以下顺序执行：

```text
same-address uint64 CAS
  -> same-address uint64 atomic-add
  -> 写本 thread 的 CAS old value / add ticket / marker
  -> asc_threadfence
```

CAS initial 与每个 thread 的 desired 有不同的固定高位，且低 12 bit 唯一编码
thread id。因此 host 不只统计 winner 数，还可以验证“返回 initial 的线程”
与“最终 desired 的 owner”必须是同一个 thread。add 结果在 host 排序后必须
精确等于 `[initial, initial + thread_count)`，不允许只用 final count 推断。

Main Scalar 在 V→S wait 后用 `ld_dev` 读取 atomic cell 和逐线程诊断数据，
不使用 Scalar 普通 GM load。这一选择是为了隔离 atomic 语义与 S0～S2
已证实存在差异的 SIMT/Scalar 普通 DCache 可见性，不表示正式 executor
可以省略 payload reader DCCI。

### 7.4 CPU 与 CCEC 结果

统一构建命令：

```bash
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-atomic
```

CPU 模型不用 OS thread 数伪装硬件上限，只对 32/64/128 actor 验证相同语义。
为避免 sanitizer 无意义地反复创建大量 OS thread，最终统一入口轮数为：

```text
[PASS] SIMT atomic CPU rounds=16 thread_configs=32/64/128: optimized
[PASS] SIMT atomic CPU rounds=8  thread_configs=32/64/128: ASan+UBSan
[PASS] SIMT atomic CPU rounds=4  thread_configs=32/64/128: TSan
```

CCEC 和静态门槛实际通过：

```text
[CHECK] SIMT atomic source closure and full-contention sequence
[CHECK] bitcode contains 2048-thread SIMT launch, GM uint64 CAS/add,
        return stores, fence and V/S wait
[CHECK] ELF exports only AIV Main entry; SIMT atomic entry is local;
        metadata is MIX_AIV_MAIN [0:1]
[BUILD] SIMT atomic CCEC complete
```

optimized bitcode 中同时存在 `llvm.hivm.atom.CAS.G.u64` 和
`llvm.hivm.atom.ADD.G.u64`，以及 TID、SIMT info、workitem fence、V/S event、
`LD.DEV.u64.GM` 和 `ST.DEV.u64`。最终 ELF 只导出一个非空 AIV Main
entry，并保留一个非空 LOCAL SIMT entry，无 undefined GLOBAL 和 relocation。

### 7.5 真实 A5 结果

当前 shell 仍无 `npu-smi`、无 `task-submit`，因此按用户已授权的 device 0
未加锁路径运行。ACL 报告 `Ascend950PR_958b`，本轮不记录性能。命令：

```bash
timeout --foreground 300s \
  tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-atomic --device 0 --runs 100
```

真实输出：

```text
[DEVICE] id=0 soc=Ascend950PR_958b state_bytes=49856
         max_threads=2048 warp_size=32
[SUMMARY] threads=  32 warps= 1 CAS+add+returns+guards=100/100
[SUMMARY] threads=  64 warps= 2 CAS+add+returns+guards=100/100
[SUMMARY] threads=1024 warps=32 CAS+add+returns+guards=100/100
[SUMMARY] threads=2048 warps=64 CAS+add+returns+guards=100/100
[PASS] A5 SIMT GM uint64 atomic runs=100
       configs=32/64/1024/2048 reused_address=yes
```

| 线程 | warp | 同地址 CAS | 同地址 add ticket | marker/inactive/guard | 结果 |
| ---: | ---: | ---------: | ----------------: | --------------------: | ---: |
| 32 | 1 | 100/100 | 100/100 | 100/100 | PASS |
| 64 | 2 | 100/100 | 100/100 | 100/100 | PASS |
| 1024 | 32 | 100/100 | 100/100 | 100/100 | PASS |
| 2048 | 64 | 100/100 | 100/100 | 100/100 | PASS |

四档都是同一个 device allocation 上重复地址运行；每轮都检查恰好一个 CAS
winner、全部 loser 的 old value、最终 CAS owner、完整 add ticket 排列、最终
add count、active marker、inactive tail、6 条 guard 和 64-bit 高位。

### 7.6 阶段结论与边界

A0 证明在当前 `Ascend950PR_958b` 和 CANN 20260708 组合上，SIMT
thread 可继续使用 GM `uint64_t asc_atomic_cas/add`，且同地址语义在
1、2、32、64 warp 下均精确。当前工具链和真实 A5 都接受 2048-thread，
因此 1024 是常用规模，不是本环境的已证实硬上限。

这个结论不包含 UBUF atomic、非 `uint64_t` 类型、CAS/add 之外的指令、
多 AIV 同时对同一 atomic 地址竞争，也不改变普通 payload 的 reader DCCI
规则。本阶段没有测量性能，没有生成泳道图。A0 随本节所在提交独立交付，
然后继续 S4。

## 8. 2026-08-05：S4 多 task、单 builder

### 8.1 目标与冻结合同

S4 不再用一个 Vector task 和一个 Cube task 代表调度，而是在同一次
`1 AIC + 2 AIV` mixed kernel launch 中完成 16 个相互独立的 task：

- 偶数 task 为 Vector，共 8 个；奇数 task 为 Cube，共 8 个；
- AIV0 只构建 task，AIV1 只执行 Vector，AIC 只执行 Cube；
- AIV0 发射 128 个 SIMT thread，即 4 个 warp；task `i` 映射到
  `tid=(i%4)*32+((i/4)%32)`，16 个 task 均匀分布到 4 个 warp，
  不跨 thread 拼 descriptor；
- AIV1 和 AIC 各只有一个 busy token，完成当前 workload、发布 `DONE` 和
  fan-in 计数后，才允许处理下一个兼容 task；
- 最终 drain 必须精确等于 `builder_finished/vector_done/cube_done/done_count
  = 1/8/8/16`，16 个 task 必须分别到达合法 `DONE`；
- 每个 task 使用独立的 16×16 FP32 输入和输出，不能通过重复写同一个地址
  假装完成多 task 调度。

这仍是独立 GM 探针。它验证的是多 task 扫描、路由、单 token 占用和完成
fan-in，不把当前线性扫描声称为正式 PA DAG scheduler，也没有引入
`cross_core` 或 `ops-nn` 源码依赖。

### 8.2 修改文件与作用

- `gm/common/s4_multi_task.h`：冻结 16 task ABI、动态 state 编码、每 task
  payload/checksum、executor busy 统计、独立 tile 和 7 条 guard；
- `gm/test/test_s4_multi_task.cpp`：CPU 协议模型，包含半包不可见、4-warp
  交错分工、错误 engine 路由拒绝、busy 时第二次 Claim 拒绝、完整 fan-in
  和逐元素 golden；
- `gm/cpu/build_s4.sh`：optimized、ASan/UBSan、TSan 三套测试；
- `gm/ccec/s4_multi_task_kernel.cpp`：AIV0 SIMT builder、AIV1 Vector executor、
  AIC Cube executor 和三个角色共同参与的 drain；
- `gm/ccec/s4_multi_task_host.cpp`：真实 A5 host，复用同一 device allocation，
  对 16 份 state/payload/report 和全部输入输出做精确校验；
- `gm/ccec/build_s4.sh`：源码时序、bitcode intrinsic、ELF symbol/relocation、
  mixed metadata 和 AIV 本地内存预算门槛；
- `run.sh`：增加 `build-s4`、`run-s4` 统一入口。

### 8.3 协议与 workload 边界

每个 builder thread 对自己负责的 task 执行：

```text
EMPTY --CAS--> BUILDING
  -> 写完整 8-word descriptor
  -> asc_threadfence
  -> BUILDING --CAS--> BUILT
```

AIV0 等待 SIMT launch 的 V→S completion 后，逐 task 用 `ld_dev` 校验
reserve/publish 返回值和 thread 归属，全部 16 项成功才发布
`builder_finished=1`。writer 不执行 DCCI；与 S0～S3 的实测结论保持一致，
AIV1/AIC Claim 后对 payload cacheline 执行 reader `dcci + dsb`。

两个 executor 只扫描自己的奇偶 task。每个 task 严格执行：

```text
等待 BUILT -> 确认 token free -> CAS Claim -> reader DCCI
  -> 真实 Vector add 或 Cube matmul并等待写回
  -> CAS DONE -> engine_done atomic-add -> done_count atomic-add
  -> 释放 token
```

CPU 模型在首个 task 的 busy 区间主动尝试第二次 Claim，并要求它被拒绝、
第二个 task 仍为 `BUILT`；所以 CPU 的 `busy_blocked=1` 是负向测试证据。
设备 kernel 不主动制造违规 Claim，host 要求设备侧 `max_busy=1` 且
`busy_blocked=0`。两者验证的是同一个合同，不是数据不一致。

### 8.4 CPU、CCEC 与 ELF 结果

统一构建命令：

```bash
export ASCEND_HOME_PATH=/home/q00473782/Ascend/cann-9.1.0-weekly-20260708/cann-9.1.0
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-s4
```

CPU 三套结果：

```text
[PASS] S4 CPU multi-task rounds=16: 4-warp/128-thread mapping builds 16 tasks,
       busy depth 1, fan-in and goldens
[PASS] S4 CPU multi-task rounds=8:  4-warp/128-thread mapping builds 16 tasks,
       busy depth 1, fan-in and goldens (ASan+UBSan)
[PASS] S4 CPU multi-task rounds=4:  4-warp/128-thread mapping builds 16 tasks,
       busy depth 1, fan-in and goldens (TSan)
```

CCEC/ELF 门槛实际通过：

- dav-c310-vec bitcode 包含 4-warp/128-thread SIMT launch/TID、GM uint64 CAS、
  atomic-add、thread fence、reader DCCI、Vector load/add/store 和 V/S event；
- dav-c310-cube bitcode 包含 CAS、atomic-add、reader DCCI、MTE2、ND2NZ、
  L1→L0A/L0B、MAD、FIX 写回和 event；
- 最终 ELF 只导出两个非空 mixed GLOBAL entry，并保留非空 LOCAL
  `S4SimtBuildTasks`、`RunVectorAdd`、`RunCubeMatmul`；无 undefined GLOBAL
  和 relocation；
- 两份 metadata 都是 `MIX_AIC_MAIN [1:2]`；AIV metadata 为
  `SIMD_SIMT_MIX_VF` 且 SIMT share 为 8 KiB；
- 三块 1024 B Vector tile 加 8 KiB SIMT share 合计 11 KiB，未超过
  224 KiB 上限。

编译过程中实际发现 `shared_protocol.h` 的 `constexpr EncodeExecState` 是
host-only，CCEC 禁止从 `__aicore__` 调用。修正方式不是放宽检查，而是在 S4
公共头中按同一位域常量实现设备可调用的编码器；SIMT VF 内则直接按已查证的
位域编码，CPU 继续用同一个协议解码器检查最终 state。

### 8.5 真实 A5 结果

当前 shell 无 `npu-smi`、无 `task-submit`，本轮沿用用户已经明确授权的
device 0 未加锁验证；结果只用于功能正确性，不进入性能结论。先跑 1 次
冒烟，再在同一个 device allocation 上复用相同地址 100 次：

```bash
timeout 60s tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-s4 --device 0 --runs 1
timeout 120s tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-s4 --device 0 --runs 100
```

真实输出：

```text
[DEVICE] id=0 soc=Ascend950PR_958b topology=1AIC+2AIV
         state_bytes=53248 builder_warps=4 builder_threads=128 tasks=16
[SUMMARY] 4-warp-build+16-state+drain+8-vector+8-cube=100/100
[PASS] S4 mixed multi-task runs=100 reused_address=yes
       vector_tasks=8 cube_tasks=8 busy_depth=1
```

每一轮均逐项满足：

- 16/16 reserve 从 `EMPTY` 成功，publish 返回对应 `BUILDING`；
- 每个 report 的 builder thread 精确等于
  `(task_index%4)*32+((task_index/4)%32)`；
- 16 份 payload 的 magic/version/nonce/三个 GM 地址/shape/checksum 全匹配；
- AIV0 Build 16/16、Claim 0；AIV1 和 AIC 都是 Build 0、Claim 8/8、
  execute 8、最大 busy depth 1；
- 16 个 control 全部到达匹配 engine/owner/task id 的 `DONE`；
- drain 精确为 `1/8/8/16`，fatal=0；
- 8 份 Vector 输出、8 份 Cube 输出、全部输入和 7 条 guard 逐元素通过。

### 8.6 同 warp 缺口与跨 warp 修正

首个 S4 提交 `a29fa08e` 使用 `tid=0..3`。这四个 lane 都属于 warp 0，
能够证明同一 warp 内的多 lane 分工，但不能证明多个 warp 独立推进 task
构建。用户指出该缺口后，本阶段没有把“4 thread”继续解释成充分的并发证据，
而是把 launch 扩为 128 thread/4 warp，并采用 warp-interleaved 映射。

映射不是简单让 `tid=0..15` 构建 16 个 task，因为那样仍全部落在 warp 0；
task 0..3 分别由 tid 0/32/64/96 构建，task 4..7 分别由
tid 1/33/65/97 构建，依此类推。host 逐 task 校验实际 thread id，因而不能
由某一个 warp 代替其他 warp 完成后仍误判通过。该映射只用于 S4 的无前后继
探针 task，不直接扩到有严格插入顺序的 G0。G0 改为发射 2048 thread/
64 warp，每个 warp 仅 lane 0 工作；1280 个 task 按 `task_id%64`
分给 64 个 leader，每个 leader 构建 20 个 task。

### 8.7 阶段结论

S4 已证明一个固定 AIV0 可以用 4 个 SIMT warp 构建多份独立 task，AIV1
和 AIC 在同一次 kernel launch 中按 engine 过滤并各自以 busy depth 1 连续
执行，且完整 workload 写回后才能发布 per-engine/global fan-in。真实 A5
100/100 说明结果不是 CPU 模型或静态 IR 推断。

本阶段没有测性能、没有生成泳道图；未加锁设备墙钟不作性能结论。S4 随本次
阶段提交交付，不 push。下一阶段 G0 才把这一能力接入 GM 版完整 PA DAG。

## 9. A1：同 warp 串行与跨 warp 独立推进

### 9.1 验证问题与接口查证

S4 修正为 4 warp 后，`tid` 分布只能证明任务确实落在不同 warp，仍不能直接
回答“同 warp 的不同分支是否串行、不同 warp 的不同代码能否独立推进”。A1
因此建立独立的 `protocol_probe/warp_concurrency/`，不改 S4 协议，也不把
thread 数量当作并行证据。

先查当前 CANN 9.1 安装和本机 `ops-nn/control/sleep` 实现，确认：

- `simt_api/asc_simt.h` 提供的 SIMT `clock()` 实际使用
  `__cce_simt_get_CLOCK64()`；
- thread id 使用 `threadIdx.x`，warp/lane 分别按 `/32` 和 `%32` 推导；
- GM `uint64_t` 没有单独 atomic-load API，本探针复用已经由 A0 证明的
  `asc_atomic_cas` 发布和 `asc_atomic_add(address, 0)` 读取；
- CCEC optimized bitcode 中对应符号必须包含
  `llvm.hivm.get.CLOCK64`、`llvm.hivm.atom.CAS.G.u64` 和
  `llvm.hivm.atom.ADD.G.u64`。

### 9.2 CPU 与设备 oracle

CPU 模型显式区分 outer-branch mask 和真正执行 actor 的 leader mask，并保存
warp id、leader lane 和 cooperative dispatch epoch。同 warp 使用 warp0 的
`0x0000ffff/0xffff0000` 两个互斥外层 mask，executing mask 分别只有
lane0/lane16。模型同时覆盖 A-first 和 B-first：先执行的路径发布后有界等待
4 次并 timeout，之后另一条路径才能开始并观察前者。跨 warp 使用 warp0/warp1
两个 outer actor 每 epoch 各推进一次，executing mask 都只有 lane0，双方在
第二个 epoch 都观察到对方。non-leader mailbox 访问次数必须为 0。

另设不依赖 host wall time的 step oracle，覆盖多组正常值和零边界：

| A steps | B steps | A-only | B-only | same warp | cross warp |
| ------- | ------- | ------ | ------ | --------- | ---------- |
| 7 | 11 | 7 | 11 | 18 | 11 |

所有用例都要求 `same=A+B`、`cross=max(A,B)`。这只是理想 cooperative
调度模型，不拿它替代 A5 指令管线事实。

设备统一发射 64 thread，四种模式复用相同 A/B 工作函数：

- `AOnly`：tid0；
- `BOnly`：tid16；
- `SameWarp`：仅 warp0，A/B 分支 leader 为 tid0/tid16；
- `CrossWarp`：A/B 分别位于 warp0/warp1，leader 为 tid0/tid32。

实现检查中曾发现 `SameWarp` 首版只判断 lane，导致 64-thread launch 的
warp1 也会重复写 ready/report；在设备运行前已修为先限定 `warp==0`，host
仍逐字段要求精确 tid0/tid16，不能容忍重复 writer。A/B 都先 CAS 发布 ready，
再进行最多 200000 次、同时受 CLOCK64 deadline 约束的 atomic poll。Main
Scalar 等 V→S completion 后只用 `ld_dev` 汇总报告，host 再核对全部 control、
ready、report、checksum、inactive sentinel、atomic padding 和五条 guard。

### 9.3 构建与静态证据

统一入口：

```bash
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-warp
```

结果：

- CPU optimized、ASan+UBSan、TSan 全部 PASS；
- dav-c310-vec CCEC object 和 optimized bitcode 构建通过；
- bitcode 同时包含 64-thread SIMT launch/TID、CLOCK64、GM uint64 CAS/add、
  workitem fence、V/S flag 和 Scalar `ld_dev/st_dev`；
- 最终 ELF 只导出一个 AIV Main global entry，SIMT entry 为 local，metadata
  为 `MIX_AIV_MAIN [0:1]`，无 undefined global 和 relocation；
- GCC 15 ACL host 在 `-Wall -Wextra -Werror` 下通过。

### 9.4 真实 A5 结果

本轮再次执行仓库 A5 precheck；当前 shell 没有 `npu-smi` 和 `task-submit`，
但 `/dev/davinci0` 存在，因此沿用用户已经明确授权的 device 0 未加锁功能验证
路径。ACL 实际报告 `Ascend950PR_958b`。先 smoke 1 轮，再以同一个 896-byte
device allocation 连续复用 100 轮：

```bash
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-warp --device 0 --runs 100
```

100 轮全部 PASS。CLOCK64 是原始 tick，不换算 ns；下表为 100 轮摘要：

| 模式 | PASS | span 平均 | A 区间平均 | B 区间平均 | poll | handshake | 区间关系 |
| ---- | ---- | --------: | ---------: | ---------: | ---- | --------- | -------- |
| AOnly | 100/100 | 1,060,021.5 | 1,060,021.5 | - | 0/- | N/- | single |
| BOnly | 100/100 | 1,120,029.0 | - | 1,120,029.0 | -/0 | -/N | single |
| SameWarp | 100/100 | 97,630,251.2 | 96,508,663.3 | 1,121,012.6 | 200000/1 | T/S | disjoint |
| CrossWarp | 100/100 | 1,121,257.7 | 1,062,052.3 | 1,121,125.8 | 1/1 | S/S | overlap |

`SameWarp` 的 A 区间故意包含 200000 次超时 poll，所以不能拿它当作纯 A
workload 性能；它的用途是制造因果判据。A timeout 后 B 才开始，100 轮始终
是 `T/S`，span 与本轮 A/B 两段区间之和只差约 575 tick（0.000589%），符合
分歧路径串行执行。

`CrossWarp` 100 轮始终是 `S/S`：双方都在对方 bounded poll 尚未结束时发布
ready，形成双向 forward-progress 证据，而且包含握手、poll 和 work 的
`RunA/RunB` 总区间始终重叠。其平均 span 只比较慢的 B-only 高 0.1097%；
cross A/B 自身相对单独运行分别只高 0.1916%/0.0979%。结合 `poll=1/1`、
span 接近较慢单分支和两段总区间重叠，可以推断本例 work 阶段获得了有效的
并发推进；不能仅凭区间字段宣称纯 work 的起止点完全重叠。本次真实设备数据
因此同时区分了：

1. 同 warp 的互斥分支不能在等待期间推进另一分支；
2. 不同 warp 能独立取得 forward progress，并且本例两条工作区间实际重叠；
3. “区间重叠”不外推为任意两条 SIMT 指令都能同周期 issue，执行管线仍可能共享。

A1 不生成泳道图，不把未加锁 host 墙钟写成性能结论。本阶段只做本地阶段性
commit，不 push。

## 10. G0：纯 SIMT 多 warp 构建完整 PA

### 10.1 阶段目标与不可越过的角色边界

G0 首次把前面探针证明的原语接入完整 shared TensorMap PA 流程。源码继续在
`simt_cross_core` 内独立闭合，不 include `cross_core` 或 `ops-nn` 源码；但
task ABI、五类 task、DAG、payload、writer history、四 token 执行和最终
drain 均逐项对照现有生产实现，而不是另造一套简化协议。

本阶段采用以下固定分工：

| 设备角色 | 数量 | 允许的工作 | 明确禁止 |
| -------- | ---: | ---------- | -------- |
| AIV0 SIMT | 2048 thread/64 warp | 每个 warp 仅 lane 0 构建、发布并提交自己的 task。 | inactive lane 访问 task；执行任何 PA task。 |
| AIV0 `__aicore__` entry 壳 | 1 | 发起 VF、等待 VF 并参加 drain；不是调度角色。 | 获得 task build/commit 计数；构造 descriptor/payload；提交 history/last-writer/insert-completion；发布 BUILT；Claim 或执行 task。 |
| AIC executor | 32 | 从 AIC ticket 表 Claim、执行 QK/PV、发布完成。 | 构建 task。 |
| AIV executor | 63 | 从 AIV ticket 表 Claim、执行 SF/UP、发布完成。 | 构建 task。 |

entry 壳的 role build/commit/execute/claim 计数全部为 0。实际构建数由
64 份 SIMT thread report 独立求和，`builder_finished` 也由最后一个 SIMT task
在 VF 内发布，entry 壳不能代发或取得 task 归因。

### 10.2 完整 PA task、payload 与 heap 口径

每个 batch 固定五个 task，`task_id=5*batch+kind`：

| kind | 名称 | engine | tensor/scalar/fanin | payload | output reserve |
| ---: | ---- | ------ | ------------------: | ------: | -------------: |
| 0 | Alloc | 不进入执行器 | 0/0/0 | 无执行 payload | 10,240 B |
| 1 | QK | AIC | 4/2/0 | 592 B，10 line | 524,288 B |
| 2 | SF | AIV | 4/3/1 | 604 B，10 line | 264,192 B |
| 3 | PV | AIC | 4/2/1 | 596 B，10 line | 8,192 B |
| 4 | UP | AIV | 7/2/3 | 988 B，16 line | 0 B |

DAG 固定为 `SF<-QK`、`PV<-SF`、`UP<-SF,PV,Alloc`，即每 batch 五条
fanin edge。UP writer history 固定记录 Alloc 的 output key `3/2/1`；最终仅
Alloc slot0 的 last-writer 从 Alloc 更新为 UP，slot1/2 仍保持 Alloc。

B1 共 5 个 task、4 个可执行 task，heap reserve 为 806,912 B。B256 共
1,280 个 task、1,024 个可执行 task，heap reserve 为 206,569,472 B。heap
分成 8 个 25,821,184 B shard；SIMT leader 真实执行分 shard atomic reserve，
所以 task base 的先后次序允许随并发变化，host 改为按 shard 排序检查区间，
不把某一种偶然分配顺序写成 golden。

### 10.3 64 warp 全并发构建与严格 insert

G0 发射 2048 个 thread，但只让每个 warp 的 lane0 工作。task 映射为
`warp=task_id%64`、`tid=warp*32`、下一 task 为 `task_id+64`。B256 中每个
leader 精确构建 20 个 task，另外 1,984 个 lane 的 prepare/commit/state
访问计数必须全为 0。

prepare 与严格 insert 被有意分开：

1. leader 先 CAS `EMPTY->BUILDING`，预留 heap，写 plan 和完整 fresh-output
   descriptor；
2. descriptor 后执行 SIMT thread fence，立即 CAS 发布 fresh output 及初始
   last-writer；该步骤不进入全局严格链；
3. 构造完整 inline payload，仍保持 `BUILDING`；需要前驱输出时只原子等待
   前驱独占的 task-base report，再按冻结 shape 重建 descriptor；
4. 等待 `task[N-1].insert_completion==N-1` 后，才提交 UP history、跨 task
   last-writer、本 task insert-completion，以及 Alloc completion 或 kernel
   task 的 `BUILT`。

相邻 task 总在不同 warp，包括 `63->64` 回绕，因此不会出现同 warp 的一条
分歧路径等待另一条路径。等待值只允许从 `N-2` 变为 `N-1`；观察到第三种值
立即报告 `InsertProtocolFailed`，不能伪装成普通超时。构建报告逐 thread 保存
tid/warp/lane、首尾 task、prepare/commit/等待次数、nonce 和 checksum，host
逐项检查 64 个 active leader，并要求 1,984 个 inactive lane 的整份 report
保持 host poison；因此同一 warp 除 lane0 外连诊断 GM store 也没有。

### 10.4 四 token 执行、真实 workload 与终态发布

AIC/AIV 各自使用 immutable ticket 表，每个 executor owner 固定四个
`ExecutionToken`。扫描器即使先看到 `EMPTY/BUILDING`，也保留该 ticket 并停在
`WaitingBuilt`，而不是丢弃 task。B1 的最终 cursor 为 AIC/AIV `34/65`，
B256 为 `544/575`，即有效 task 数再加每个 executor 精确一次越界 ticket。

四类 kernel task 都执行真实 128×128 FP32 workload：QK/PV 做 128 项
`2*3` 累加，抽样结果为 768；SF 做 `2+3`，结果为 5；UP 做 `2*3`，结果为
6。每个 owner 使用独立输出 tile，host 检查实际结果且确认两个输入 tile 未被
改写。完成顺序固定为：

```text
workload -> execution witness -> vend -> flag -> DONE
```

DONE CAS 冲突单独报告 `CompletionStateConflict`，不再与普通 vend/flag/witness
发布失败混成一个原因。

executor 进入 workload 前逐条原子读取 producer completion，并只在真实
`fanin_ready_prefix` 推进到 task 所需的 0/1/1/3 后继续。该运行时 prefix
随 execution witness 一起发布，host 按 task kind 精确核对；因此 fanin
不再只由 payload 和终态 completion 自洽证明。`completion_sequence` 仍是
终态路径标记，不把它误称为独立的逐阶段时间线。

生产 `ResetExecutionToken` 只清 control、保留 dispatch，G0 也保持这一语义。
首版终态检查只验证 control 恢复初值，无法证明中间确实发生过 binding；最终版
还检查每个实际用过 token 保留的最后一份 tensor/scalar args、local/global
context、自指针、task/owner/layout/vend。为避免给每个 task 增加 DCCI，executor
只在彻底排空后，对最多四个已用 token 集中 clean+invalidate control，以及
dispatch 的有效 line 0/1/6/7，最后执行一次 DSB。未用 token、args 9..47、
padding 和 builder owner32 的四个 token 必须保持初始化 poison。

最终 drain 仍为 16 组、每组 6 个物理参与者；每组 atomic 值同时编码到达数
和该组完成数。固定 root 只有在 96 个角色全部到达后，才核对 1,024 个完成、
AIC/AIV 各 512 个完成、256 个 Alloc completion、精确 cursor 和所有 token
排空，并发布 `root_finished`。

### 10.5 CPU 语义模型与独立 host oracle

CPU 模型覆盖 B1/B256、64 个 builder leader、8-shard heap、精确 payload/DAG、
四 token、fanin、completion、16 组 drain、inactive tail 与同地址复用。除完整
case 外还包含：

- 半包 payload 在 BUILT 前不可 Claim；
- 多 owner 对同一 task 只能唯一 Claim；
- 四个 token 同时停在 `WaitingBuilt` 后仍能继续推进；
- strict insert prefix 必须从 task0 连续增长，不能跳号；
- retained dispatch 与 control-only reset 必须同时成立；
- 两份不同 nonce 在同一 materialized state 上复用，旧 witness/poison 不得泄漏。

完整 CPU case 让 64 个 builder 与 95 个 executor owner 从同一闸门并发启动，
executor 按 owner 推进 `WaitingBuilt` 和单调 `fanin_ready_prefix`，不再采用
“BuildAll 后 ExecuteAll”的串行自洽模型。95 个 executor 各自原子到达 drain，
64 个 builder join 后 AIV0 只发布 `builder_finished` 并作为零 task 角色到达；
root 必须实际等齐 16 组各 6 个参与者。AIV0 entry 壳的 role
build/commit/claim/execute 始终为 0，1280 个 prepare/commit 只由 64 份 SIMT
leader report 求和。

materialize 后再次从独立 ABI 对象复核 active task、heap、role、token、builder
report 和 drain；retained token 中的 tensor/local/global/self pointer 全部重绑
到目标 `FullPaState`，不能继续指向 CPU 模型的源对象。DONE control、运行时
fanin witness、奇数 fanin 高 32 位、plan 保留字段、history 尾部及连续 vend
区间也纳入检查。`completion_sequence` 仍只是终态见证，真实指令先后另外由
设备源码顺序、atomic 发布和 host 终态共同约束，不把一个常量夸大为完整
时间线证明。

统一构建入口：

```bash
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-g0
```

它依次运行 optimized、ASan+UBSan、TSan，并构建 AIC/AIV CCEC object、optimized
bitcode、1:2 mixed ELF 和 GCC 15 ACL host。

### 10.6 CCEC、bitcode 与 ELF 静态门槛

构建脚本除编译外还锁定以下事实：

- 源码闭包不 include `cross_core`/`ops-nn`；发射 2048 thread、64 warp，且
  仅 lane0 按 `warp+64*k` 构建；
- fresh descriptor store、SIMT fence、fresh publish 均位于严格前驱等待之前；
- `builder_finished` 的 CAS 位于 VF 内，entry 壳只能 invoke/join；
- reader 对 payload 每条 cacheline 执行 DCCI+DSB 后才普通读取；
- workload 前先按 task/nonce 毒化抽样输出并 DSB，随后真实 TSTORE 必须覆盖；
- execution witness 保存运行时 token 的 `fanin_ready_prefix`，host 按 task
  的 fanin 数核对；
- optimized bitcode 含 SIMT TID、GM uint64 CAS/add、workitem fence、逐行 DCCI、
  Vector add/multiply、Cube matmul 和 drain atomic；
- 最终 ELF 仅有两个 global kernel entry，无 relocation；metadata 为
  `MIX_AIC_MAIN [1:2]`，AIV entry 为 `SIMD_SIMT_MIX_VF`、share memory 8 KiB；
- AIV metadata 给出 8 KiB SIMT share memory；源码门槛同时冻结 128×128 FP32
  三个不重叠 Vector tile，据此计算 `192+8 KiB`，不超过 224 KiB 上限；
- workload、witness、vend、flag、DONE 的源码顺序固定。
- 成功构建后原子生成 SHA-256 清单，覆盖 G0 device/host 构建输入及配对的
  kernel/host；
  `run-g0` 遇到缺失或任一哈希不匹配时拒绝访问设备。

### 10.7 真机暴露并修复的问题

真实 A5 首轮没有死锁，但独立 oracle 连续暴露了两个仅靠 CPU 终态不容易发现的
问题：

1. B1 首轮 AIC0 role 的 kind 计数出现 `0x4c600000/0x1200`。它们正是
   `0x12004c600000` GM state 地址的片段。原因是 CCEC 会标量替换 128 B
   Scalar 局部 aggregate，不能用 `reinterpret_cast<uint64_t*>(&result)` 假定
   它仍是连续对象。修复为从每个命名字段显式打包 16 个 ABI word 并逐 word
   `st_dev`，同时增加字段 offset static_assert；随后 B1 通过。
2. B1 通过后，B256 在 owner37/slot0 报 token control 未复位，但 phase 已是
   Idle。O3 IR 证明成功和错误清理路径都写了全部 control 字段；真正原因是
   B256 cache 压力曾把 busy 中间态逐出到 GM，最终普通 reset 留在 Scalar
   DCache，而构建明确关闭 kernel-end DCCI。B1 因 backing memory 仍是初始
   Idle，反而可能假通过。修复为上一节的 executor 终态集中 DCCI，并把 host
   诊断细化为 control 8 个 64-bit word 的 actual/expected/xor；B256 随后通过。

静态复核还在真机前修正了 `shape_and_scalar_offset` 位域打包，以及 fresh output
曾被错误放入严格 insert 链的问题。后者虽然终态值相同，却会把本应并发的
descriptor publication 串行化，因此增加了源码顺序门槛防止回归。

终审又发现 owner 会复用同 kind output tile，而所有同 kind task 的 golden
相同；若某次 workload/TSTORE 被跳过，旧 tile 仍可能产生合法 witness。最终版
在每个 task 的 workload 紧前，用 task/nonce 派生值 `st_dev` 毒化首 64 bit
并执行 DSB，随后只有真实 TSTORE 覆盖后 checksum 才能通过。该动作只服务功能
见证，所以本阶段不报告其性能数据。

### 10.8 真实 A5 结果与结论边界

本轮 A5 precheck 因 shell 无 `npu-smi` 无法检测 silicon；独立的
`command -v task-submit` 检查也确认队列工具不在 PATH，但 device0 存在。按本
会话已有授权，每次运行前均输出 precheck-unavailable 与 unlocked 两条警告，
只做功能验证。ACL 报告 SoC 为 `Ascend950PR_958b`；
`32*(1AIC+2AIV)` 是本探针 launch/ELF 固定并由 host 输出的拓扑配置，不归因
给 ACL 自动探测。最终 state 为 31,876,160 B，workspace 为 12,713,984 B；
同一进程的多轮测试保持 device state/workspace 地址不变。

若环境提供 `task-submit`，标准调用必须先在锁外完成 precheck，再提交；锁内
`run-g0` 自动把 `$TASK_DEVICE` 注入 host，禁止再次传 `--device`：

```bash
.claude/skills/onboard-arch-precheck/check.sh a5 && \
task-submit --timeout 600 --max-time 600 --device auto --device-num 1 \
  --run "cd $PWD && tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
    run-g0 --batches 256 --runs 10"
```

入口还会做一次锁内 precheck 作为二次防护：exit 2 的确定架构不匹配无条件
拒绝；只有 exit 1 的无法检测、`task-submit` 又确实缺失时，才走本会话已经
明确授权的 unlocked fallback。入口另有 600 秒 host timeout 和成功构建清单
校验，不能裸跑陈旧或不配对的产物。

最终复现命令为：

```bash
timeout --foreground 120s tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-g0 --device 0 --batches 1 --runs 10
timeout --foreground 180s tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-g0 --device 0 --batches 256 --runs 10
```

当前环境的 `run-g0` 先输出 precheck-unavailable warning，再输出 unlocked
warning。两条命令的原始汇总分别为：

```text
[SUMMARY] G0 B1 passes=10/10 fresh_initialization=yes same_address_reuse=validated
[SUMMARY] G0 B256 passes=10/10 fresh_initialization=yes same_address_reuse=validated
```

| case | task/kernel task | heap reserve | 结果 | 覆盖 |
| ---- | ---------------: | -----------: | ---- | ---- |
| B1 | 5/4 | 806,912 B | 10/10 PASS | 首轮 fresh allocation；随后同地址、逐轮 nonce/poison 重置。 |
| B256 | 1280/1024 | 206,569,472 B | 10/10 PASS | 首轮 fresh allocation；64 leader 各 20 task 及同地址复用。 |

这些运行没有生成泳道图，也不把未加锁 host 墙钟当作性能数据。G0 的结论限于：
纯 SIMT 64-warp/lane0 builder 与 AIC/AIV executor 能在同一次 mixed kernel
中协同完成 B1 和 B256 的完整 PA 语义。代码允许构建和执行并发推进，但本阶段
没有设备侧 overlap 时间见证，因此不把终态结果夸大为时间区间重叠证据；也不
声称 64 个 warp 每周期同时 issue 或当前实现已经性能最优。阶段只做本地
commit，不 push。

## 11. G1：AIV0+AIV1 双 VF 全量竞争构建

### 11.1 用户纠正与最终角色边界

G1 最初讨论过“SIMT 只 Prepare、AIV0 Scalar 串行 Commit”的两段式方案，
但这会把最关键的严格 insert chain 重新归因给 Scalar，也不能模拟两个 builder
完整竞争。最终按用户纠正冻结为纯 SIMT：AIV0/AIV1 各发射 2048 thread，
各 64 个 warp 只有 lane0 工作，共 128 个有效 leader。任一 task 只由 AIV0/AIV1
中相同 local-warp 的两个 leader 竞争唯一 winner，不是 128 个 leader 同抢一个
task。`__aicore__` entry 壳只做 VF invoke/join 和最终 drain，不构造、
提交或执行 task。

A1 已经用真实 A5 区分了两种情况：同 warp 的 tid0/tid16 分歧路径 100 轮始终
`T/S + disjoint`，不同 warp 的 tid0/tid32 100 轮始终 `S/S + overlap`。因此
G1 没有在一个 warp 内安排两个有效 lane；AIV0/AIV1 的相同 local warp 也是
两个独立 VF 的不同 global warp，不依赖同 warp 分歧取得 forward progress。

### 11.2 参数化复用而不是复制 G0

G0/G1 共用同一套 ABI、CPU 模型、kernel、ACL host 和构建入口，运行时只通过
`builder_count=1|2` 选择拓扑。`builder_thread_count` 仍表示每份 VF 的 2048，
总 report 容量单独扩为 4096，避免把 AIV1 误发射成 4096 thread。主要变化为：

- 全局 thread/warp 分别为 `instance*2048+local_tid` 和
  `instance*64+local_warp`，两实例写入互不重叠的 report；
- AIV0/owner32、AIV1/owner33 为 builder，owner34..95 才是 AIV executor；
- 每个 VF 的 thread0 对 `builder_started` 到达一次，所有 active leader 等齐
  1/2 个实例后再 claim；
- executable task 用实际 owner 竞争 `EMPTY -> BUILDING`，Alloc 用
  `completion.flag: 0 -> ALLOC_BUILDING(owner) -> 1`，loser 不做任何 task 写入；
- winner owner 贯穿 plan、build report、BUILT、token、CLAIMED 和 DONE；
- 每 task 的 build report word6 用两个 32-bit 原子计数直接取证 attempt/win；
  为避免同行的 DCache 普通回写与 atomic 竞争，winner 对其余 word 也只能从
  host poison 用 atomic-CAS 发布，整个 64 B report 禁止普通 SIMT store；
- thread report 的 `task_count/prepare/commit` 只统计 win，
  `task_state_access_count` 统计 attempt，`claim_lost_count` 统计 loss；
- AIV dispatch 尾 cursor 从 G0 的 `2*batches+63` 变为 G1 的
  `2*batches+62`，AIC cursor 不变；
- 两个 builder role 的 build/commit/claim/execute/ticket/exhausted 全为 0，
  但都按物理拓扑各到达一次 drain；owner32 仍是唯一 root。

ABI 版本升为 2。state 从 G0 阶段的 31,876,160 B 增至 32,007,296 B，精确
增加 131,136 B：第二组 2048×64 B thread report 加一条 64 B
`builder_started` cacheline；没有逐轮 raw trace 或按 poll 次数扩张的数组。

### 11.3 CPU 并发模型和独立 host oracle

CPU 模型同时运行 builder_count 1/2。每个实例的 64 个 leader 与全部 executor
从同一 start gate 出发，两个 instance 的 thread0 另行完成 builder-start
闸门。每个 task 都有独立 atomic attempt/win；Alloc 和 kernel task 分别走与
设备一致的两种 claim，winner 才能 reserve/Prepare/Commit，loser 只记 loss。
CPU 定向负例还逐项拒绝同 owner 重入、错误 task/engine/payload-lines、非法
executor route、未知状态位，以及 Alloc 的错误 nonce/task/owner/magic，避免
CPU oracle 比设备 loser 协议更宽松。

materialize 和 host oracle 不预设 owner32 获胜，而是从每个 task 的实际
plan/report 反推出 winner instance、global thread/warp，再逐线程重建
win/first/last/wait/loss/checksum。固定总量为：

| 配置 | task | attempt | win | loss | executor | AIC/AIV cursor |
| ---- | ---: | ------: | --: | ---: | -------: | -------------: |
| G0 B1 | 5 | 5 | 5 | 0 | 95 | 34 / 65 |
| G0 B256 | 1280 | 1280 | 1280 | 0 | 95 | 544 / 575 |
| G1 B1 | 5 | 10 | 5 | 5 | 94 | 34 / 64 |
| G1 B256 | 1280 | 2560 | 1280 | 1280 | 94 | 544 / 574 |

统一 CPU 入口最终重跑结果：

```text
[PASS] G0 CPU complete: builders=1/2, B1/B256, 64 leaders/builder,
       unique build claim, 8-shard heap, exact DAG/payload, 4-token tickets,
       fanin/completion/drain/tail, same-address reuse rounds=4
[PASS] ... ASan+UBSan ... rounds=2
[PASS] ... TSan ... rounds=2
```

三套测试都覆盖单/双 builder、B1/B256，并在同一个 `FullPaState` 地址连续
materialize 两个 nonce。G0 的 `[2048,4096)` report 必须保持初始值；G1
两组各只允许 64 个 lane0 leader 写 report，其余 lane 必须保持 host poison。
host 还逐 task 检查 `attempt==builder_count && win==1`，
所以“只启动 AIV1、实际所有 task 都没尝试”不能通过。

### 11.4 CCEC、bitcode、ELF 与运行入口门槛

设备实现没有新增第二份 kernel 源码。AIV entry 先严格检查
`block<32 && subblock_dim==2 && subblock<2`，再用
`block*subblock_dim+subblock` 得到 dense AIV id；不使用未经查证的 core-id
映射。AIV id 小于 builder_count 才发射 VF，否则进入 executor。

构建脚本新增了以下 fail-closed 检查：每 VF 必须是 2048 thread/64 warp/lane0；
builder gate 必须早于 attempt/claim；claim 必须早于 Prepare，Prepare 早于
Commit，`builder_finished` 只能在 VF 内发布；实际 owner 必须贯穿 state/token；
task build-report 整条 cacheline 必须只使用 atomic-add/atomic-CAS，entry 壳
不得获得 task build/commit 归因；inactive lane 的 thread report 必须保持 poison。
最终 CCEC AIC/AIV、optimized
bitcode inventory、1:2 mixed ELF symbol/metadata/relocation 和 GCC15 host 均
通过。成功清单同时覆盖 `run.sh`，因此 G0/G1 的固定 `--builders` 注入规则被
修改后，旧产物会被拒绝。

统一入口为：

```bash
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-g1
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh run-g1 \
  --device 0 --batches 1 --runs 10
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh run-g1 \
  --device 0 --batches 256 --runs 10
```

`build-g1` 复用参数化的 G0/G1 构建；`run-g0` 强制注入 `--builders 1`，
`run-g1` 强制注入 `--builders 2`。两者都拒绝用户覆盖 kernel/builders，执行
A5 precheck、SHA-256 清单校验和 600 秒 timeout，并在 task-submit 内只接受
`$TASK_DEVICE` 注入的 device。

### 11.5 真实 A5 结果与结论边界

本 shell 仍无 `npu-smi` 和 `task-submit`，故沿用本会话已明确授权的 device0
unlocked 功能验证；每次均输出两条 warning，不把 host 墙钟作为性能数据。
ACL 报告 `Ascend950PR_958b`。G1 先跑 B1，再跑 B256；随后用同一产物和
builder_count=1 回归 G0。四组均首轮 fresh initialization，随后复用相同
device state/workspace 地址并逐轮改变 nonce：

```text
[SUMMARY] G1 builders=2 B1   passes=10/10 fresh_initialization=yes same_address_reuse=validated
[SUMMARY] G1 builders=2 B256 passes=10/10 fresh_initialization=yes same_address_reuse=validated
[SUMMARY] G0 builders=1 B1   passes=10/10 fresh_initialization=yes same_address_reuse=validated
[SUMMARY] G0 builders=1 B256 passes=10/10 fresh_initialization=yes same_address_reuse=validated
```

逐轮 winner 分布也由 host 在完整 oracle 通过后直接打印。G1 B1 的
`AIV0/AIV1` 依次为
`0/5, 2/3, 5/0, 2/3, 3/2, 4/1, 1/4, 0/5, 0/5, 5/0`；G1 B256
十轮均为 `640/640`。B1 第 1 轮 owner32 没赢到 task，但该轮仍逐 task 验证
owner32 的五次 attempt 和五次 loss。这正是区分“参与竞争”与“最终获胜”的
原因，G1 不以两个 builder 都必须赢到 task 作为伪并发门槛。

G1 的每轮 B1 都直接证明 10 次 attempt/5 个唯一 winner/5 次 loss；B256 证明
2560 次 attempt/1280 个唯一 winner/1280 次 loss。所有 1024 个 kernel task
仍各执行一次，两个 builder entry 壳的 role 均为零 task，94 个 executor
完整排空，16 组 drain 各 6 arrivals；descriptor、payload、history、last-writer、
fanin、vend/flag、execution witness、DONE、workspace golden、guard、padding 和
inactive tail 全部通过。

本阶段没有生成泳道图，也没有测性能。结论限于：两个独立 VF 中相同
local-warp 的两个 lane0 leader 最终都对其负责的每个 task 发起尝试，唯一 winner
完成完整 Prepare/Commit，并能与 94 个 executor 在同一次 mixed kernel 中完成
真实 PA DAG。start gate 与 attempt/win 不能单独证明两个对应 leader 对每个 task
的尝试区间必然重叠；A1 证明的是不同 warp 具备独立推进和区间重叠能力，B256
的实际 winner 分布是调度现象而非协议定理。这里不声称胜率均匀、每 task 必然
时间重叠或每周期双发射。阶段只做本地 commit，不 push。

## 12. U0：纯 SIMT UBUF 单槽与直接 GM 发布

### 12.1 开工前的 MTE3 接口核查

原设计的 U0 顺序是“SIMT 写 UBUF，Main Scalar 用 MTE3 搬到
GM 后发布 `BUILT`”。这会让 Scalar 重新获得 task 的搬运和发布
语义，与 G0/G1 已冻结的“`__aicore__` entry 壳只
invoke/join/drain”边界
冲突。因此 U0 在写实现前先核查了 CCEC 和本机 `ops-nn`，
没有臆测 VF 内的 MTE3 接口。

查证结果为：

1. CANN 9.1 当前头文件的 `copy_ubuf_to_gm_align_v2` 定义在
   `namespace __cce_scalar`；`__clang_cce_aicore_functions.h` 还明确用
   `#define CCE_SCALAR(FUNC) __cce_scalar::FUNC` 包装该指令。
2. 在 `__clang_cce_simt*` 和 `asc/.../simt_api` 中搜索
   `copy_ubuf_to_gm`/`MTE3` 没有得到 SIMT-native 对应物。
3. 临时最小 CCEC 探针中，VF 内的未限定名调用直接报
   undeclared identifier；显式调用 `__cce_scalar::` 并强制保留函数
   后，编译器不能生成合法 SIMT 代码。该探针只放在 `/tmp`，
   没有进入仓库。
4. `ops-nn/hash/embedding_hash_table_export/op_kernel/arch35/`
   `embedding_hash_table_export.h` 的真实实现是：普通 aicore 分配
   `TBuf`，把 `GetPhyAddr()` 转为 `__ubuf__ *` 传给
   `asc_vf_call`；`ExportPerThread` 在 VF 内读 UBUF 后直接逐线程写
   `__gm__` 输出。这条路径不发起 MTE3。
5. `ops-nn` 全仓的 `copy_ubuf_to_gm_align_v2` 只出现在普通
   `__aicore__` matmul epilogue 中；“含 `__simt_vf__` 的源文件”与
   “含 GM↔UB copy intrinsic 的源文件”交集为空。

所以 U0 不会将 Scalar MTE3 写成 SIMT MTE3，也不会为了保留
旧流程而让 Scalar 发布 task。当前实现边界改为
`SIMT write UBUF -> same leader load UBUF -> direct GM store -> threadfence ->`
`SIMT publish BUILT`，并在 ABI/诊断中硬性记录
`transport=SIMT_UBUF_READ_TO_GM_WORD_STORE` 和 `mte3_count=0`。该结论只表示当前
工具链的可用边界，不否定未来工具链新增 SIMT-native MTE3
的可能。

### 12.2 纯 SIMT 单槽合同

U0 固定一份 2048-thread VF，即 64 个 warp；仅 lane0 是有效
leader，其余 1984 个 lane 必须保持 host poison。64 个 leader
分别对应 64 个 task，可以并发将各自 control 从 `EMPTY` 竞争到
`BUILDING`；随后它们用独立 GM atomic-only cacheline 竞争同一个
AIV0-private UBUF slot。

单槽 ABI 固定为 4480 B：前 guard 64 B、payload 容量
4352 B（68 条 cacheline）、后 guard 64 B。region 与 payload 均按
64 B 对齐，payload offset 为 64 B，slot 数为 1。control 显式带有
transport 和 launch nonce，build report 带有 slot ticket，不从源码
布局暗推 ABI。

只有 slot CAS winner 能写 UBUF、检查两侧 guard、从 UBUF 读回并
写自己的 GM payload，再由同一 leader 完成 fence、
`BUILDING -> BUILT` 和 slot release。payload 有效长度按 task id
循环覆盖 `1/10/16/68` 个 cacheline，实现只读写有效 word，不会
先填满 68 行再用相同 poison 遮住越界。AIV1 是唯一 executor，
只在 `BUILT` 后 Claim，对有效行做 reader DCCI+DSB 后校验
payload，最后发布 `DONE`；AIC 仅观察终态，不 Claim。

工具链要求的 AIV0 `__aicore__` entry 只是 VF 启动壳：传入
固定对齐 UB 基址，执行 `async_invoke -> join -> drain`。它不是
builder 或调度角色，不能 build/commit/publish/claim/execute 任何 task；
聚合的 `main_scalar_build_action_count` 以及 role claim/finish 必须全为 0。
真正的 64 个 task 构建只能归因到 64 份 SIMT lane0 report。

CPU 和设备共用同一套 fatal ABI，低 8 位是 reason，随后是 owner
与 task id。executor 轮询时只允许精确的 `EMPTY/BUILDING`
继续等待，精确 `BUILT` 才能 Claim，其他组合立即以
`InvalidTaskState` fail-closed，不把协议错误伪装成 timeout。设备内
watchdog 固定为 `1e9` ticks，角色 timeout 计数与 fatal reason 分开
记录。

### 12.3 实现文件与静态门槛

U0 保持在 `ubuf/` 独立闭包内：

- `common/u0_single_slot.h` 定义 task、slot、UBUF region、fatal 和
  host/device 共用 ABI；
- `common/u0_single_slot_cpu_model.h` 实现可控交错、负向注入和
  有界退出的 CPU 协议模型；
- `test/test_u0_single_slot.cpp` 覆盖发布边界、单槽复用、随机压力、
  非法 control、guard、timeout 和异常清理；
- `ccec/u0_single_slot_kernel.cpp` 实现 64-warp/lane0 纯 SIMT builder、
  AIV1 executor 和 AIC observer；
- `ccec/u0_single_slot_host.cpp` 重建 payload/checksum/tail/GM guard/report
  全量 oracle，并核对 SIMT 自检发布的 UBUF guard 计数，于同一
  设备地址复用 100 轮；
- `cpu/build_u0.sh`、`ccec/build_u0.sh` 和总入口 `run.sh` 分别封闭
  CPU 三套、CCEC/bitcode/ELF/host 与真机运行。

构建门槛除了检查 2048 threads/64 warps/lane0，还锁定以下源码
顺序：

```text
UBUF store < UBUF load < GM store < threadfence < BUILT CAS
           < busy release < slot-owner release
Claim CAS < reader DCCI < payload read < DONE CAS
```

同时拒绝固定遍历 `kMaxPayloadWords`、MTE3/UBTOOUT intrinsic、额外
SIMT entry、未定义全局符号或 relocation。SHA-256 清单覆盖 `run.sh`、
ABI、kernel、host 和构建脚本，修改任一运行时输入都会拒绝旧产物。

### 12.4 审计发现与修正

首轮实现通过后没有直接收口，而是按 ABI、负向路径和真实
UBUF 边界再做一次审计，修正了：

1. CPU、device、host 原先各自编码 fatal 的布局偏差，改为公共
   `EncodeU0Fatal/DecodeU0Fatal`；
2. executor 将畸形 control 反复轮询到 timeout 的问题，改为立即
   `InvalidTaskState`；
3. `1<<42` 内部 watchdog 可能晚于外层 300 s timeout 的问题，
   改为 `1e9` ticks，并让 role timeout 实际可计数；
4. 设备路径没有 UBUF guard、且先写满 68 行 poison 会遮住固定
   搬满错误的问题，改为前后 guard 与仅处理有效 word；
5. CPU 只停在“GM 写一半”的覆盖缺口，补了“GM 已完整写入但
   仍为 `BUILDING`”的精确暂停点；
6. 64 builder 压力只有固定启动顺序的缺口，补了固定 seed 的
   shuffle 和预生成 yield 扰动，避免测试本身引入 RNG 数据竞争。

### 12.5 CPU 结果

最终入口 `run.sh build-u0` 中的 CPU 三套全部通过：

```text
[PASS] U0 CPU rounds=8: optimized
[PASS] U0 CPU rounds=2: ASan+UBSan
[PASS] U0 CPU rounds=2: TSan
```

每套都覆盖 64 个 SIMT leader、一个 executor、single-slot depth=1、
seeded shuffle、pre-BUILT hidden、fail-closed control/guards、bounded faults、
direct GM store 和 `mte3=0`。定向交错确认：UBUF 只写一半时
GM 仍为 poison；GM 已全部写完但尚未发布 `BUILT` 时 executor
仍不能 Claim；第二 leader 在首个 task publish/release 之前不能取得
单槽。CPU 负向测试另外验证了 timeout、异常、畸形 control 和
UBUF guard 破坏均能有界退出。

### 12.6 CCEC、bitcode 与 mixed ELF 结果

AIC/AIV 均用 CANN 9.1 CCEC `-O3` 编译通过，optimized AIV bitcode
保留 AS6 volatile UBUF load/store、SIMT CAS/add/fence、reader DCCI/DSB 与
V→S event，且不含 MTE3/UBTOOUT intrinsic。最终 ELF 只有预期的
AIC/AIV 两个 GLOBAL entry，metadata 为 1:2 `MIX_AIC_MAIN`，AIV 为
`SIMD_SIMT_MIX_VF`。编译器产物的 share memory 为 8 KiB，4480 B
带 guard region 完整位于该 share region 内，同时保留不少于
32 KiB SIMT DCache。最终 kernel ELF 为 224104 B，GCC15 ACL host
为 30936 B。

### 12.7 真实 A5 与旧阶段回归

当前 shell 仍无 `npu-smi` 和 `task-submit`，onboard precheck 返回 1。
因本会话已明确授权 device0 unlocked 单卡功能验证，入口在打印
precheck-unavailable 与 unlocked warning 后继续；下列结果不是受管
性能数据。

最终复现命令为：

```bash
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-u0
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh run-u0 --device 0 --runs 100
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-g1
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh run-g0 --device 0 --batches 1 --runs 1
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh run-g0 --device 0 --batches 256 --runs 1
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh run-g1 --device 0 --batches 1 --runs 1
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh run-g1 --device 0 --batches 256 --runs 1
```

ACL 报告 SoC 为 `Ascend950PR_958b`，state 大小 423424 B，地址
`0x120000019000`。最终重建后在同一地址连续复用 100 轮：

```text
[SUMMARY] U0 passes=100/100 same_address_reuse=validated
          builder_scalar_build_actions=0 executor_owner=33
          transport=SIMT_UBUF_READ_TO_GM_WORD_STORE mte3=0
```

每轮都精确为 64 task，slot acquire/release=`64/64`，builder/executor/
done=`64/64/64`，UBUF guard check=64，busy depth=`0/1`，`fatal=0`、
`timeout_count=0`。host 同时逐 task 验证 `1/10/16/68` 有效长度、
payload/checksum/tail/GM guard、slot ticket 唯一性、launch nonce 和 1984 份
inactive-lane poison。

`run.sh` 改动会让 G0/G1 旧清单失效，因此重跑 `build-g1`，
G0/G1 CPU optimized、ASan+UBSan、TSan 和 CCEC/ELF/host 全部通过；
真机再回归四组：

| 配置 | 结果 | 关键证据 |
| ---- | ---- | -------- |
| G0 B1 | 1/1 | 5 task，单 builder 每 task 1 attempt。 |
| G0 B256 | 1/1 | 1280 task，1280 unique win。 |
| G1 B1 | 1/1 | 10 attempt/5 win，winner `3/2`。 |
| G1 B256 | 1/1 | 2560 attempt/1280 win，winner `640/640`。 |

### 12.8 结论边界

U0 已证明：在一份 2048-thread VF 中，64 个不同 warp 的 lane0
都在同一 VF 中具备并发竞争单个 UBUF slot 的资格，winner 在不引入
任何 Scalar task
构建/提交/发布逻辑的前提下，完成 UBUF 暂存、同 leader 读回、
直接 GM store 和 `BUILT` 发布，且 executor 只会在发布后读取。
本阶段不通过逐 task 时间区间声称 64 个 leader 的竞争区间全部
重叠；跨 warp 独立 forward progress 与区间重叠能力由 A1 单独取证。

本阶段不证明 SIMT-native MTE3，不证明多槽性能收益，不生成泳道图，
也不将 unlocked host 墙钟写成性能结论。设备动态测试覆盖的是
100 轮成功路径；fatal/guard 负向注入由 CPU 模型和 CCEC 静态顺序
门槛验证，不夸大成已经在真机动态注入。阶段只做本地
commit，不 push。

## 13. U1：纯 SIMT 四槽与 generation 复用

### 13.1 容量与接口查证

U1 开工前先重新核对了“四个最大 U0 槽”的容量。U0 每槽
4480 B，两槽就会超过当前 ELF 的 TLV7 8192 B，不能直接把 offset
倍增后就宣称合法。本机 ACL 头文件确实存在
`ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE`，`aclrtLaunchKernelWithHostArgs`
也接受 launch cfg；本机 `ops-nn` 还有 12 KiB dynamic local-memory 的
AscendC 实例。但这些证据尚未回答 standalone mixed ELF 中的真机
参数、对齐和 DCache 分区，所以 U1 不将 dynamic UB 当作未验证前提。

真实 G0 PA shape 继续提供了更小、可直接验证的边界：QK/SF/PV
分别是 592/604/596 B，均向上取整为 10 条 cacheline；UP 是
988 B，取整为 16 条。数值同时由 `full_pa_model.h` 的 shape、
`full_pa_exec_protocol.h` 的 layout 公式和设备 executor 的同一公式
锁定。这是当前 PA Case 的上限，不是通用协议 68 行容量的替代。

因此 U1 固定四槽，每槽为 `64 B guard + 16*64 B payload +
64 B guard = 1152 B`，合计 4608 B，在 TLV7 8192 B 内剩余 3584 B。
U0 继续保留 68 行单槽边界，U1 只验证与 U2 当前 PA Case 直接
衔接的 `1/4/10/16` 行四槽路径。TLV7 容量检查只是静态门槛：
公开接口没有完整说明裸 offset 与 VF stack 的内部分区，而 U1 会
比 U0 多触及 `4480..4607`。所以四槽 guard 的真机自检是必须
的最终地址边界证据，不能被 metadata 算术代替。

### 13.2 冻结协议

U1 不引入 PA descriptor 或严格 insert chain，这些属于 U2。本阶段用
128 个互相独立、可全量校验的 payload task 隔离多槽生命周期问题。
2048 thread/64 warp 仍然只有 lane0 有效，每个 leader 精确处理
`warp` 和 `warp+64`两个 task。slot 与 payload 映射为：

```text
slot_id(task)       = task_id % 4
payload_class(task) = ((task_id / 4) + 3) % 4
payload_lines       = {1, 4, 10, 16}[payload_class]
```

每槽因此都经历 32 次复用，且四种长度各 8 次。task0..3 是
16-line anchor：四个 leader 各自取得一个不同槽，完整写入并检查
guard 后，各自用 CAS 只置 `anchor_staged_mask` 中与 task0..3 对应的
bit，在保持所有权的情况下同时等到
`anchor_staged_count=4 && anchor_staged_mask=0xf`。
其他 60 个 leader 在此之前不能 acquire，使真机必须形成
`max_busy_depth=4`。

每槽的 atomic state 用高 32 位保存 generation，低 32 位保存
`task_id+1`，0 表示 free。只允许 `FREE(g)->BUSY(g,task)` 和
`BUSY(g,task)->FREE(g+1)` 两种 CAS。host 必须对每槽验证 report 中
generation 集合恰好为 `0..31`、无重复，终态恰好为 `FREE(32)`。
异常如果发生在 acquire 之后，必须用原 BUSY 值精确释放且推进
generation，未发布 task 再尝试恢复 `EMPTY`，然后通过 fatal 让全部
角色有界退出。release 固定先有界减 global busy，再 exact-CAS 槽状态；
后者失败必须恢复 global busy。

AIV0 `__aicore__` entry 壳仍只做 VF invoke/join/drain，所有 build
claim（`EMPTY->BUILDING`）、UBUF 写读、GM store、fence、`BUILT` 和槽释放
都在 SIMT leader 内。AIV1 owner33 延续 U0 的独立 executor，只负责
execute claim（`BUILT->CLAIMED`）/DCCI/校验/DONE，不参与构建。本阶段要
动态证明四个不同 anchor 的完整
staging payload 曾同时驻留，不声称 64 个 leader 的时间区间全部重叠，
也不记录性能收益。

### 13.3 实现与审查修正

U1 新增了以下实现：

- `ubuf/common/u1_multi_slot.h`：128 task、四槽布局、generation 状态、
  task/build/exec/thread/role report 和统一 fatal ABI；
- `ubuf/common/u1_multi_slot_cpu_model.h` 与
  `ubuf/test/test_u1_multi_slot.cpp`：受控交错、故障注入和 64-warp
  压力模型；
- `ubuf/ccec/u1_multi_slot_kernel.cpp`：AIV0 的 2048-thread VF、四槽
  UBUF 直接 GM transport、AIV1 executor 和 AIC observer；
- `ubuf/ccec/u1_multi_slot_host.cpp`：真机逐 task、逐 generation、逐
  inactive lane 和 guard oracle；
- CPU/CCEC 的 `build_u1.sh` 及统一 `run.sh build-u1/run-u1` 入口。

首轮审查发现，若用 `task_id/4` 预先推导 generation，就隐含了
同槽 task 按 id 串行的错误前提。真实并发下，generation 只能由
`FREE(g)->BUSY(g,task)` 成功的实际 CAS 次序产生。最终删除了
task-id-to-generation helper，build report 记录实际 `g`，host 只验证每槽
generation 集合精确为 `0..31`。

第二轮审查发现，仅有 `anchor_staged_count=4` 不能独立排除
某一 anchor 重复到达。因此新增了身份位图：task0..3 只能用
CAS 各自置 bit0..3，门槛同时要求 `count=4 && mask=0xf`。另外将
所有持槽的 pre-publish 错误收口统一为：有界减 busy、exact-CAS
推进 generation、必要时回滚 busy、尝试 `BUILDING->EMPTY`，最后
才发布本地 fatal。另一线程已发布 fatal 时，当前 holder 也走同一
cleanup epilogue。

最终只读审查又封住了三个边界。第一，mask 与 count 位于不同 GM
atomic cacheline，设备端不能把 `count=4/mask!=0xf` 的瞬态可见性偏差
直接判成永久错误；现在所有范围内不一致都继续有界轮询，CPU 增加
确定性偏差用例，证明它以 timeout 而非 invariant 收口。第二，在 GM
copy/fence 之后、发布 `BUILT` 之前增加紧邻 fatal 检查；CPU 的受控交错
改为断言 holder 精确释放、恢复 EMPTY、保留外部首错且绝不发布。
第三，设备 `busy_depth` 递减由无限 CAS 改成最多 4096 次尝试，cleanup
本身不再可能永久自旋；anchor 身份位图 CAS 同样复用该上限。

原先用于保留 SIMD/SIMT mixed metadata 的 anchor 位于 config 校验之前，
畸形 version 理论上可触达 UBUF。现在它只在完整 `ConfigValid` 成功后，
由保留的诊断 nonce `UINT64_MAX` 触发；该单 word 写发生在 SIMT launch
之前，并会被 slot0 guard 初始化覆盖。正常 probe 路径不执行它。role
report 中的 `main_scalar_build_action_count=0` 只作为运行时 telemetry，
不再声称能够自证“没有 Scalar 构建代码”；构建脚本另行截取
`RunBuilder` 的完整作用域并拒绝 task CAS/state/payload/report 写入模式，
与 host telemetry 形成两条独立门槛。

后续 executor 复审发现，原循环会先尝试 `BUILT->CLAIMED`，仅在 CAS
失败的等待分支检查 fatal；如果首错已经存在而后续 task 已是 BUILT，
owner33 仍可能继续执行。现在每次 claim 前先查 fatal，claim 成功后、
payload 前以及 DONE 前各复查一次；后两个窗口观察到首错时用精确 CAS
恢复 `CLAIMED->BUILT`，不再推进 DONE。CPU 新增两份预置 BUILT task，
先由畸形第三 task 发布 foreign fatal，再断言两者保持 BUILT 且
claim/done 都为 0，首错也未被覆盖。

### 13.4 CPU 验证

统一命令：

```bash
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-u1
```

三套最终结果全部 PASS：

| 构建 | 压力轮数 | 结果 |
| ---- | -------: | ---- |
| optimized | 8 | PASS |
| ASan + UBSan | 2 | PASS |
| TSan | 2 | PASS |

CPU 用例动态覆盖了：四个 anchor 同时持有四个完整 16-line
payload、发布前不可见、同槽不提前复用、长短 payload 交替不写
GM tail、异常/guard/timeout 精确释放、旧 generation 不能释放新 owner、
畸形 control fail-closed、executor 逐 word 校验和 seeded 64-warp 竞争。
还有一组确定性交错专门让 anchor0 持有 slot0 等 gate，再由 AIV1
畸形 control 路径发布首个 fatal；anchor0 最终精确收口到
`FREE(1)`、task 恢复 `EMPTY`、global busy 回到 0，且不覆盖首错。

### 13.5 CCEC、bitcode 与 ELF 门槛

`build-u1` 后半段的结果如下：

- dav-c310-cube AIC observer 和 dav-c310-vec AIV builder/executor 都通过
  CCEC `-O3`；
- optimized AIV bitcode 保留 AS6 volatile UBUF load/store、GM uint64 CAS/add、
  workitem fence、Scalar reader DCCI/DSB，且 builder transport 中没有
  MTE3/UBTOOUT；
- 最终 ELF 只有 AIC/AIV 两个 global entry，内部 SIMT entry 为 local，
  无 undefined global 和 relocation；
- metadata 为两个 `MIX_AIC_MAIN [1:2]`，AIV VF 类型为
  `SIMD_SIMT_MIX_VF=4`，TLV7 为 `0x2000`/8192 B；
- 4608 B 四槽区域通过静态预算，kernel ELF 为 273952 B，
  GCC 15 ACL host 为 30968 B；successful-build manifest 也已生成并校验。

静态容量仍只是入场条件，它本身不证明 UBUF `4480..4607` 可用；
这一部分由下一节真机 guard 结果闭合。

### 13.6 真实 A5 结果

本轮仍先执行仓库 A5 precheck。当前 shell 没有 `npu-smi` 和
`task-submit`，但 `/dev/davinci0` 存在，因此沿用用户已授权的
device 0 unlocked 单卡功能验证，不记录 host 墙钟性能。ACL 实际报告
`Ascend950PR_958b`，device state 为 288640 B，地址
`0x120000019000`。

先执行 1 轮 smoke，再对同一 device allocation 连续复用 100 轮：

```bash
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-u1 --device 0 --runs 1
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-u1 --device 0 --runs 100
```

| 项目 | 1 轮 smoke | 100 轮同地址复用 |
| ---- | ---------: | ---------------: |
| PASS | 1/1 | 100/100 |
| task BUILT/DONE | 128/128 | 每轮 128/128 |
| 每槽 acquire/release | 32/32 | 每轮 32/32 |
| 每槽终态 | `FREE(32)` | 每轮 `FREE(32)` |
| anchor count/mask | `4/0xf` | 每轮 `4/0xf` |
| global max busy | 4 | 每轮 4 |
| UBUF guard check | 128 | 每轮 128 |
| inactive lane | 1984 份 poison | 每轮 1984 份 poison |
| builder Main Scalar build telemetry | 0 | 每轮 0 |
| builder transport MTE3 | 0 | 每轮 0 |

四个 anchor 的 slot/generation report 还分别为 `0/0`、`1/0`、`2/0`、
`3/0`，同时 `mask=0xf` 和 `max_busy=4`，所以这里的四槽同驻留
不是仅从一个累加计数推测。每个 task 的 build report 亦证明每槽
generation 集合精确为 `0..31`。后 guard 位于过去 U0 未触及的
`4480..4607`；100 轮全部 guard 和 GM 邻接区域校验通过，所以
当前 A5 上的 4608 B 裸 UBUF 边界已经获得动态证据。

### 13.7 U0 与 G0/G1 回归

由于统一 `run.sh` 加入了 U1 入口，U0 和 G0/G1 的 successful-build
manifest 都必须重建，不能继续使用旧产物。本轮重新执行：

```bash
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-u0
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-g1
```

U0 CPU optimized 8 轮、ASan+UBSan 2 轮、TSan 2 轮全部 PASS，
CCEC/bitcode/mixed ELF/host 和新 manifest 通过。G0/G1 参数化 CPU 模型
optimized 4 轮、ASan+UBSan 2 轮、TSan 2 轮全部 PASS，完整 PA
CCEC/bitcode/mixed ELF/host 和新 manifest 也通过。随后执行真机功能
回归：

| 路径 | 配置 | A5 结果 | 关键终态 |
| ---- | ---- | ------- | -------- |
| U0 | 64 task/单槽 | 1/1 PASS | acquire/release `64/64`，BUILT/DONE `64/64`，mte3=0 |
| G0 | B1/1 builder | 1/1 PASS | 5 task/4 kernel task，attempt=1 |
| G0 | B256/1 builder | 1/1 PASS | 1280 task/1024 kernel task，attempt=1 |
| G1 | B1/2 builders | 1/1 PASS | 每 task attempt=2，唯一 winner |
| G1 | B256/2 builders | 1/1 PASS | 1280 task 的 winner 为 `640/640` |

回归仍是用户授权的 unlocked device 0 功能运行，不作性能取样。

### 13.8 本阶段结论与边界

U1 已证明：一份 2048-thread VF 中 64 个独立 warp 的 lane0 可以
以纯 SIMT 语义竞争和复用四个 AIV0 私有 UBUF slot，并将 128 份
`1/4/10/16` 行 payload 逐有效 word 直接写入 GM，再由独立 AIV1
executor 收口。该结论同时覆盖实际 generation、无提前复用、四槽
同驻留、异常可收口和 100 轮同地址复用。

U1 仍然是独立 payload 诊断 task，没有 PA descriptor、shared TensorMap
history/last-writer 或严格 insert chain；这些属于 U2。本阶段也没有
SIMT-native MTE3，transport 仍是 UBUF volatile load 后的普通 GM word store。
`max_busy=4` 只证明四个 anchor staging 区间曾同时存在，不声称
64 个 warp 的所有指令都同周期并行，也不把 unlocked 运行写成性能数据。

## 14. U2：四槽 UBUF 迁移到完整 PA

### 14.1 开始前复用审计

U2 的第一原则是复用 G0，而不是在 `ubuf/` 下复制一套完整 PA 调度器。
已确认可直接保留的存量包括：

- `common/full_pa_exec_protocol.h` 的 exec state、payload layout、token、
  completion 和 fatal ABI；
- `common/full_pa_model.h` 的五类 task、每 batch 五 task/四 kernel task、
  DAG、heap、fanin、engine route 和 drain 映射；
- `gm/common/g0_full_pa.h` 的完整 host/device state、task plan、build report、
  writer history 和 execution witness；
- `gm/ccec/g0_full_pa_kernel.cpp` 的纯 SIMT claim/prepare/strict commit，以及
  AIC/AIV executor/workload；
- 同一份 G0 CPU model、ACL host 和 oracle。

因此实现采用同源编译期 transport policy：Direct-GM 继续生成 G0/G1，
Ubuf-Staged 生成独立 U2 产物。差异只能进入 payload sink、四槽状态/报告、
guard 和对应 oracle，不允许分叉 task/DAG/insert/executor 主逻辑。

### 14.2 已冻结的死锁规避规则

U1 可以让任意 task 抢当前 FREE generation，因为 128 个诊断 task 相互
独立；完整 PA 不可以。U2 的后继 task 会在持槽时进入严格
`task[N-1].insert_completion` 等待。若后继抢走前驱所需的旧 generation，
会形成真环：例如 task6 若抢到 slot2 的 `FREE(0)` 并等待 task5，严格链
最终依赖 task2，而 task2 又在等待同一 slot2。

因此只允许 executable task 使用：

```text
slot = task_id % 4
generation = TaskBatch(task_id)
FREE(generation) -> BUSY(generation, task_id) -> FREE(generation + 1)
```

Alloc 不占槽；同 batch 的 QK/SF/PV/UP 分别占四个不同槽。U2 首版只能
使用一个 AIV0 builder：G1 的 AIV0/AIV1 各有私有 UBUF，不能拿一份 GM
slot state 假装两份物理 UBUF 是同一四槽。双 builder 仍只保留在 Direct-GM
G1 回归中。

### 14.3 Payload 精确搬运口径

实际 layout 为 QK `592 B/74 words/10 lines`、SF
`604 B/76 words/10 lines`、PV `596 B/75 words/10 lines`、UP
`988 B/124 words/16 lines`。SF/PV 的最后一个 64-bit word 只部分承载
语义字节，但仍必须作为完整 word 写入；其后所有 GM word 保持 task 专属
poison。U2 只复制 `written_words`，不能用 payload_bytes、payload_lines
或固定 16 行代替。

第一 batch 的 task1..4 作为四类真实 payload anchor，在各自槽内完成
staging/guard 后同时持槽，等 `count=4 && mask=0xf` 才进入 GM copy 和
原有 strict commit；四个 identity bit 使用 `1 << (task_id - 1)`，避免
误写成 task-id bit 后得到 `0x1e`。B1 每槽只允许 generation0，终态 `FREE(1)`；B256
每槽必须完整经历 generation `0..255`，终态 `FREE(256)`。

### 14.4 实施顺序

1. 抽取 U1/U2 共用的四槽几何和 64-bit slot state 编解码；先回归 U1。
2. 给同一 CPU full-PA model 注入 Direct-GM/Ubuf-Staged payload sink，
   先完成 B1/B256、anchor、tail、ordered-generation 和故障 release。
3. 参数化同一 CCEC kernel/host，生成独立 U2 mixed ELF；静态检查只限定
   builder payload transport 不含 MTE3，不误伤真实 workload 的 MTE3。
4. 跑真实 A5 U2 B1/B256，再回归 G0/G1/U0/U1并阶段性本地提交。

本节当前只记录冻结设计，没有提前填写性能或设备 PASS。

## 15. 2026-08-06：暂停 U2，补齐 Direct-GM 性能与泳道图

### 15.1 工作切换、GM/UB 边界与栈配置

用户要求先停止 U2，不再依据 stack-overflow 现象继续改 UBUF；本阶段只处理
已经通过功能验证的 Direct-GM G0 性能和泳道。这里的“GM 路径”限定为
descriptor/payload 的跨核 transport 直接落 GM，不表示编译后的 AIV 完全不
访问 UB：函数局部量、寄存器 spill、SIMT stack 和 divergence stack 仍由 CCEC
映射到 AIV 本地存储，编译器也可能为聚合初始化生成 VEC-UB copy。这类 UB
访问不是 UBUF payload staging，更没有引入 MTE3。

按用户指出的 ACL 初始化路径重新查证后，profiling host 把非空 JSON 路径直接
传给第一次 `aclInit(configPath)`。当前真机闭合配置为：

```json
{
  "StackSize": {
    "simt_stack_size": 1536,
    "simt_divergence_stack_size": 4608
  }
}
```

两项单位都是 byte，`simt_stack_size=1536` 使用 512 B stride 对齐；swimlane
AIV 另外使用 `-mllvm -cce-vf-stack-size=0x3800`，最终 metadata 的 VF 总保留为
16 KiB。生产 G0 不带这些 profiling 配置。此前 U2 在 `aclInit` 之后调用
`rtDeviceSetLimit` 的实验不是同一接口，本阶段没有据此恢复 U2。

### 15.2 atomic/DCCI 泳道实现与一致性口径

G0 增加了与生产产物分离的 `swimlane` 编译变体。新版导出命令使用独立文件名：

```bash
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-g0-swimlane
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-g0-swimlane --device 0 --batches 256 --runs 1 \
  --swimlane-json \
  tests/atomic_probe/pa_scheduler/simt_cross_core/test_record/2026-8-6/\
gm_g0_b256_atomic_dcci_swimlane.json
```

导出器现在在参数解析和真正写文件前都检查目标路径，路径已存在时直接拒绝，
不再静默 truncate。生产 G0 的 task/DAG/payload/executor 语义不变；profiling
sidecar 固定追加 3,938,432 B，包含：

- 每个 task 独立的 builder/executor 区间和每个物理 owner 的 Scalar role；
- 64 个 SIMT writer，每 writer 最多 1024 条 32 B raw record；
- 96 个 Scalar writer，每 writer 最多 512 条 32 B raw record；
- 每 writer 独占 control 和 record 区，避免不同执行单元抢同一记录槽。

每次非 poll atomic 都保留一条记录；连续 poll 只保留一个完整区间和精确
`call_count`，所以 526 万余次 poll 不会展开成 526 万个 JSON event。Scalar
atomic 通过消费返回寄存器后再读 `SYS_CNT`，标为 `return_ready`；当前 CCEC
无法为 SIMT atomic 建立同样的返回寄存器依赖，因此只诚实标为
`source_issue`，不把两个 `CLOCK64` 之间的区间冒充单次 atomic 返回延迟。

SIMT GM 读写规则按实际接口区分：`asc_stcg` 是 L1 non-cacheable 的直接 GM
store；普通 SIMT `__gm__` load 会经过 SIMT DCache，读跨 writer 或复用地址前
必须先 `asc_dcci_single` invalidate。当前 G0 builder 对共享状态的读取全部是
`asc_atomic_add(..., 0)` 或 CAS，没有普通 SIMT GM load，所以这份图中 SIMT
DCCI 为 0 是真实调用点为 0，不是漏记。Scalar executor 的普通 GM 读仍在
claim 后执行 DCCI+DSB，四类实际 DCCI 全部记录。

时钟继续严格分域。Scalar 使用本机已校准的 `get_sys_cnt()` 1 ns/tick；SIMT
使用 raw `CLOCK64`。两者 epoch 不同，只为显示把本 launch 的 SIMT 最早/最晚
点仿射映射到 AIV0 VF invoke/join 包络。JSON 顶层同时写入
`simt_alignment=affine_to_builder_scalar_vf_envelope_for_display_only` 和
`simt_atomic_boundary=source_issue`；映射后的 SIMT `dur` 不能当真实 ns。

### 15.3 真机暴露的记录损坏与最小修正

atomic/DCCI raw trace 首版在真实 A5 暴露了两类 CCEC 本地存储问题：

1. SIMT record 若在 atomic 之后继续携带 task/site/flags 等多个活跃值，CCEC
   会生成未对齐的 VEC-UB spill，运行报 error 340。最终将 metadata/attributes
   用 `asc_stcg` 在 atomic 前写完，atomic 后只保留 begin/end 两个 64-bit
   endpoint；这不改变被测 atomic 顺序。
2. Scalar executor 原先使用 `ScalarPollEpisode built[4]/fanin[4]`。没有领取
   task 的 owner 在退出 flush 时会偶发读到 `task=0x4e408bc0`、
   `call_count=4608`、`site=2` 的本地垃圾值。单纯把 ACL SIMT stack 从
   1536 B 提到 2048 B 仍只有 4/10 PASS，证明不是容量不足；逐字段初始化后
   为 7/10。最终去掉本地聚合数组的动态下标，把四个 token 的 built/fanin
   episode 变成八个具名局部量，并显式展开四次 advance/flush，B1 达到
   20/20 PASS。

host trace oracle 会逐 writer 核对 nonce/domain/count、每条 task/site/op/flags、
poll 精确计数、DCCI line 数、未用 tail poison 和汇总值；失败时额外打印四个
raw word，避免把记录损坏误判成调度协议失败。修复后 B256 同地址复用 5/5，
随后独立导出 1/1 PASS。

### 15.4 真实 A5 atomic/DCCI 泳道结果

设备为用户授权的单卡 device 0，ACL 报告 `Ascend950PR_958b`；当前 shell
没有 `npu-smi`/`task-submit`，所以是 unlocked 运行。最终文件为：

`test_record/2026-8-6/gm_g0_b256_atomic_dcci_swimlane.json`

文件为 26,350,208 B，并通过 `python3 -m json.tool` 完整解析：

| 事件类别 | event 数 |
| -------- | -------: |
| 全部事件 | 58,392 |
| complete event | 58,230 |
| metadata event | 162 |
| `atomic.source_issue` | 28,514 |
| `atomic.return_ready` | 11,413 |
| `atomic.poll_batch` | 5,184 |
| `dcci` | 2,239 |
| Scalar role/setup/loop/drain | 384 |
| SIMT task build 及五个子层 | 6,400 |
| task lifecycle/wait/execute | 1,024 / 2,048 / 1,024 |

raw 汇总既保存 event 数，也保存被合并后的真实调用次数：

| 执行域 | atomic 调用 | 其中 poll 调用 | raw record | poll record | DCCI 调用/行 |
| ------ | ----------: | -------------: | ---------: | ----------: | -----------: |
| SIMT | 253,345 | 226,975 | 29,761 | 3,391 | 0 / 0 |
| Scalar | 5,053,126 | 5,039,569 | 17,589 | 1,793 | 14,988 / 14,988 |

六类合并 poll 的拆分如下，`record` 表示图上区间数，`call` 才是实际 atomic
load 次数：

| poll site | record | call |
| --------- | -----: | ---: |
| SIMT builder-start | 64 | 64 |
| SIMT producer-task-base | 2,048 | 2,048 |
| SIMT strict insert predecessor | 1,279 | 224,863 |
| Scalar exec-state/BUILT | 1,024 | 5,038,146 |
| Scalar fanin flag | 768 | 1,359 |
| Scalar root drain arrival | 1 | 64 |

DCCI 按真实调用点拆分为：

| DCCI site | record | 调用/行 |
| --------- | -----: | ------: |
| startup config | 96 | 288 / 288 |
| dispatch task-id | 1,024 | 1,024 / 1,024 |
| exec payload | 1,024 | 11,776 / 11,776 |
| terminal token | 95 | 1,900 / 1,900 |

图上每个 builder task 为上层 `task[N] build`，下面依次包含 claim、prepare、
ordered_insert 和 build-report publish；atomic/poll 作为更下层区间落在对应
SIMT writer 泳道。每个 executor task 保留 lifecycle、wait、fanin 和 execute
包含层；Scalar atomic/DCCI 落在各 owner 的独立底层泳道。

可定量的 Scalar task 区间分布为：

| 区间 | 最小/us | 中位/us | 平均/us | 最大/us | 1024 task 累加/us |
| ---- | ------: | ------: | ------: | ------: | ----------------: |
| 完整 lifecycle | 646.868 | 5376.107 | 6623.095 | 11374.393 | 6782049.322 |
| wait_built + claim | 631.044 | 5360.574 | 6608.172 | 11358.546 | 6766768.503 |
| bind + fanin wait | 4.293 | 6.674 | 7.327 | 14.379 | 7502.523 |
| task execute | 4.644 | 7.782 | 7.559 | 11.196 | 7740.503 |

本次 trace-on 的 Scalar device span 为 21,838.877 us，AIV0 VF 包络为
21,813.620 us；导出轮 ACL event 为 22,256.912 us。最终产物另做 B256 5/5，
event 中位为 21,887.661 us。逐 atomic 记录明显扰动调度，因此这些数值只
用于说明这张图自身的时间范围，不能替代关闭埋点的性能结果。

### 15.5 关闭泳道后的 Direct-GM 性能

生产 G0 host 在 H2D 完成后记录 start event，在 mixed kernel 后记录 end
event；`aclrtEventElapsedTime` 只覆盖 kernel，不包含 32 MB state 的 H2D/D2H、
host oracle 和 JSON 导出。参数固定为：G0、1 个 AIV builder、B256、1280
task/1024 kernel task、QK/SF/PV/UP repeats 均为 1。

主取样命令：

```bash
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-g0 --device 0 --batches 256 --runs 21
```

21/21 全部 PASS，同一 device allocation 重复使用。结果：

| trace | 样本 | min/us | median/us | avg/us | max/us |
| ----- | ---: | -----: | --------: | -----: | -----: |
| off | 21 | 14916.198 | 14976.292 | 15006.951 | 15669.657 |

同一最终产物另以 B1 同地址复用 10/10 回归。因设备未锁且首样本较高，本阶段
只给出直观结论：**当前 Direct-GM G0 B256 的完整 mixed kernel 性能约为
15.0 ms，不是 1.5 ms。**

旧 overview-only 埋点曾得到 7/7 中位 14,411.556 us；新版逐 atomic/DCCI
埋点 5/5 中位为 21,851.803 us。两者记录密度不同，都不能与 trace-off 直接
相减成“固定埋点成本”。性能结论只采用 trace-off 数据。

同一 base 产物还回归双 builder G1：B1 3/3、B256 3/3，后者每轮 winner
精确为 `640/640`。因此 profiling 条件编译没有破坏 G0/G1 共用的生产路径。

### 15.6 文件覆盖失误与防回归

第一次按新增 atomic/DCCI 口径导出时，误用了已有 overview 图的同名路径，
覆盖了未纳入 Git 的旧 1.4 MiB JSON；搜索工作区和临时目录后没有找到副本，
不能声称原样恢复。新版随即改名为
`gm_g0_b256_atomic_dcci_swimlane.json`。为避免再次依赖人工记忆，host 现在
默认拒绝任何已存在的 `--swimlane-json` 目标，构建门槛也检查这条规则。

### 15.7 本阶段结论

Direct-GM 功能路径此前已经通过，本阶段补齐了此前明确缺失的两类证据：

1. 一份包含 SIMT/Scalar atomic、合并 poll 精确次数、全部实际 DCCI、builder、
   AIC/AIV task 和 Scalar role 的真实 B256 Chrome Trace；
2. 关闭泳道后的 21 轮 ACL device-event 性能，稳定量级约 15.0 ms。

泳道显示 executor 时间几乎都消耗在等待 builder 的 BUILT/claim，而
final_drain 中位仅 1.779 us；后续若优化 GM，应先处理严格构建/插入
关键路径，不能把重点放在 UBUF 或 final_drain。按用户要求，U2 在本阶段保持
暂停。最终验证已覆盖 CPU optimized/ASan+UBSan/TSan、CCEC/bitcode/ELF、
目标文件拒绝覆盖、泳道 B1 20/20 与 B256 5/5、base G0 B1 10/10 与 B256
21/21，以及 base G1 B1/B256 各 3/3。

## 16. 2026-08-06：Direct-GM 双 builder 性能收敛与泳道修正

### 16.1 优化目标和固定口径

本轮只优化已经功能闭合的 Direct-GM 路径，U2 继续暂停。固定测试口径为：

- 真实 A5 单卡 device 0，G1 双 builder，B256；
- AIV0/AIV1 只运行 SIMT builder VF，完全不执行 task；其余 94 个
  AIC/AIV owner 只执行 task，不参与构建；
- 每个 warp 仍只有 lane0 工作，不引入 Main Scalar 构建；
- 性能只采用关闭泳道后的 `aclrtEventElapsedTime` kernel event；
- 每个保留或否决的性能点都先过完整 host oracle，性能中位数不拿 trace-on
  结果替代；
- 每次保留的性能扫描都单独保存泳道图，文件名写入对应 trace-off 中位数，
  已存在文件仍拒绝覆盖。

最初的 G0 单 builder 配置为 64 warp/2048 thread，B256 中位
14,990.505 us；开始本轮时双 builder 仍让 AIV0/AIV1 对每个 task 都发起
竞争，现场中位为 8,025.589 us。优化不改变 PA task、DAG、payload、严格
insert 顺序和 executor 协议，只处理 builder 的无效竞争、诊断写入和并发度。

### 16.2 静态唯一分片：先消除双 builder 的 loser 路径

原 G1 的两个 VF 都遍历全量 task，并用同地址 CAS 决定 winner。即使最终
winner 为 `640/640`，每个 task 仍可能产生 loser claim、额外 report atomic
和缓存一致性流量。新映射把两个物理 builder 的 warp 组成一个逻辑 leader
集合：

```text
logical_leader = task_id % (builder_count * warp_count_per_builder)
builder_instance = logical_leader / warp_count_per_builder
local_warp = logical_leader % warp_count_per_builder
first_task = builder_instance * warp_count_per_builder + local_warp
task_stride = builder_count * warp_count_per_builder
```

因此每个 task 从入口就只有一个固定 writer，AIV0/AIV1 不再抢同一 task。
CPU oracle、host oracle 和 build-time source gate 都按同一公式验证
`task -> builder -> local warp -> global thread`；双 builder B256 的 winner
仍精确为 `640/640`，但 `builder_attempts_per_task` 从竞争语义收敛为 1。

在当时的 64 warp/侧配置下，静态分片把中位从 8,025.589 us 降到
7,525.793 us，减少 499.796 us（6.23%），因此保留。

### 16.3 去掉唯一 writer 上的诊断 atomic

`build_report[0..7]` 只供 kernel 结束后的 host oracle 读取，设备 executor
不消费；静态分片后每个 report 又只有一个确定 writer。生产 G0/G1 因而用
8 次 `asc_stcg` 直接发布报告，不再用 atomic CAS/add 维护 attempt/prepared
证据。task descriptor、payload、exec BUILT 和 insert-completion 等真正的
跨核发布仍保留原 atomic/DCCI 协议，不能与诊断 report 混为一谈。

单 builder 对照从 14,990.505 us 降到 13,970.151 us，减少
1,020.354 us（6.81%）；静态双 builder 的最终 64-warp 复测中位为
7,443.780 us。host 同时增加 `insert_polls` 与 `max_insert_polls`，用于区分
“构建计算慢”和“严格前驱等待多”，不再只看总 kernel 时间猜原因。

### 16.4 builder warp 数扫描

双 builder 静态唯一分片后，继续扫描每侧活跃 warp 数；launch thread 数等于
`warp_count * 32`。每组 B256 均通过完整 oracle，性能使用 trace-off 的
11 轮中位数；表中的 poll 是同组真机输出的典型量级，不把一次调度波动写成
协议常量。

| 每侧 warp | 每侧 SIMT thread | 总 leader | task 分布 | insert poll 典型总数/最大值 | B256 median/us | 相对 64-warp |
| --------: | ---------------: | --------: | --------- | --------------------------: | -------------: | -----------: |
| 64 | 2048 | 128 | 640 / 640 | 约 31.3 万～32.2 万 / 994 | 7443.780 | 基准 |
| 32 | 1024 | 64 | 640 / 640 | 约 13.8 万 / 400 | 3944.306 | -47.01% |
| **16** | **512** | **32** | **640 / 640** | **约 6.0 万 / 171** | **3636.886** | **-51.14%** |
| 12 | 384 | 24 | 644 / 636 | 约 6.8 万 / 189 | 4488.931 | -39.69% |
| 8 | 256 | 16 | 640 / 640 | 约 3.1 万 / 157 | 4109.439 | -44.79% |

最终保留每侧 16 warp/512 thread。它相对优化前的双 64-warp 现场基线
8,025.589 us 减少 4,388.703 us（54.68%），相对最初单 builder
14,990.505 us 减少 11,353.619 us（75.74%）。B1 的 5 轮中位为
148.406 us；第一轮约 796 us 的设备 warm-up 不影响中位，但原始输出仍保留。

提交前以最终同一 16-warp 源码再跑 B1 5/5、B256 21/21：B1 中位
148.936 us，B256 中位 3,742.545 us（min 3,709.436、avg 3,770.752、
max 4,404.704 us），winner 仍为 `640/640`。当前环境没有 `task-submit`，
属于 unlocked 复测；它相对扫描轮的 3,636.886 us 高 2.91%，记录为运行波动，
不改写各独立泳道文件名所对应的原始 trace-off 样本。

这个扫描说明“更多 SIMT worker”并不等于更快。严格
`task[N-1].insert_completion` 让过量 warp 同时轮询不同前驱，大量 outstanding
poll 和同一调度状态上的 atomic 压力反而拖慢真正能推进链头的 warp。64→16
时 poll 总量与尾部最大 poll 同时大幅下降。继续降到 8 时构造并行度不足，
虽然 poll 更少，关键链的 descriptor/payload 生产变慢；16 是当前两者的平衡点。
12-warp 还引入 24 个总 leader 的 `%24` 非二次幂映射，1280 个 task 不能均分，
并与 8 分片 heap 的相位不对齐，因此不能根据“12 介于 8 和 16”预设其时间也
必然居中。

### 16.5 已否决的优化及真机证据

| 实验 | 结果 | 结论 |
| ---- | ---- | ---- |
| 把 `SimtPrepareTask` 拆成 `noinline` callee | 14,990.505 → 16,921.961 us，+12.88%；入口约 17,912 B、callee 约 19,848 B | 减小单符号不能抵消 call/stack/活跃状态搬运，已回退 |
| predecessor poll 固定等待 128 tick | poll 约少 18%，中位约 7.50 → 8.231 ms | 固定退避延误链头接棒，已回退 |
| task-base 和 vend 改直接 store | 功能通过，中位约 7.648 ms | task-base 会被对端立即 atomic 读取，仍需原发布语义，已回退 |
| 只把 vend 改直接 store | 7.449 vs 7.444 ms，基本无差异 | 收益为 0 且削弱初始化冲突检查，已回退 |
| published/last-writer 改直接 store | 功能通过，中位约 8.441 ms | 跨核消费字段不能因为单 writer 就去掉原 atomic，已回退 |
| 双 builder 相邻 task 交替归属 | 7.634 ms，对照 chunk 7.526 ms，+1.43% | 更频繁的跨 VF 前驱交接更慢，已回退 |
| 同一 lane 连续批量处理自身 task | B256 136,353.439 us，约慢 18 倍 | 远端活跃 poller 会长期占压可推进 writer，已回退 |

失败实验均未进入最终源码；文档保留数据，是为了避免后续再次把“atomic 次数
少了”直接等同于“墙钟一定下降”。

### 16.6 双 builder 泳道图错误与修复

初版双 builder JSON 错把 AIV0、AIV1 两个独立 `CLOCK64` epoch 共用一组
全局 affine 参数。直接症状是 1280 个 SIMT build 区间全部被压成 0，AIV0
堆在图尾、AIV1 堆在图头；随后把这些错误 build 与 Scalar task_execute
叠在一起，就会同时造成“SIMT 构建不对”和“SIMD 执行不对”的假象。

v4 修复为每个 builder 独立对齐到自己的 Scalar VF `work_begin/work_end`
包络，并明确写入：

```text
simt_alignment = per_builder_affine_to_own_scalar_vf_envelope_for_display_only
simt_atomic_boundary = source_issue
```

同时完成以下收口：

- trace writer 数由固定单 builder 改为本次实际的 16/32 个活跃 writer；数组
  仍按双 builder 上界预留，未启动 writer 的 control/record tail 必须保持 poison；
- 每 writer 容量扩大到 8192 条，覆盖最低 8 warp 的扫描；连续 poll 仍只生成
  一个有区间的 `atomic.poll_batch`，并保留精确 `call_count`；
- writer 校验、task 归属校验和泳道名全部携带 builder instance，显示为
  `AIV0 SIMT warp N` / `AIV1 SIMT warp N`；
- `run.sh` 增加独立的 `build-g1-swimlane` / `run-g1-swimlane`，后者固定
  `--builders=2`，继续拒绝覆盖已有 JSON。

最佳 16-warp 图的自动核验结果为：1280/1280 个 build 区间均非 0，1024/1024
个 `task_execute` 区间均非 0；每个 execute 都晚于同 task 的
`ordered_insert` 结束点，违规 0 个，最小/最大间隔为 2.773/22.799 us。
build 区间为 47.887～205.594 us，execute 区间为 4.619～10.980 us。
`build_report_publish` 是 BUILT 之后的 host-only 诊断写，允许与后续 execute
重叠；执行因果必须以 `ordered_insert`/BUILT 结束点判断，不能错误地等到
report 结束。

该图本身的 trace-on device span 为 5,004.892 us；raw 汇总为 SIMT atomic
97,219 次（poll 82,368 次）、Scalar atomic 1,117,208 次（poll
1,104,982 次）、Scalar DCCI 14,968 次/14,968 行。trace-on 只用于观察，
文件名中的 `3637us` 来自独立 trace-off 中位数。

### 16.7 本轮保存的独立泳道图

目录固定为 `test_record/2026-8-6/`，五份图都采用修正后的 per-builder clock
v4 schema，没有覆盖原有单 builder `gm_g0_b256_atomic_dcci_swimlane.json`：

| 配置 | trace-off median/us | 泳道文件 |
| ---- | ------------------: | -------- |
| 双 builder，每侧 64 warp | 7443.780 | `gm_g1_b256_warp64_traceoff_7444us_atomic_dcci_per_builder_clock_swimlane.json` |
| 双 builder，每侧 32 warp | 3944.306 | `gm_g1_b256_warp32_traceoff_3944us_atomic_dcci_per_builder_clock_swimlane.json` |
| 双 builder，每侧 16 warp | 3636.886 | `gm_g1_b256_warp16_traceoff_3637us_atomic_dcci_per_builder_clock_swimlane.json` |
| 双 builder，每侧 12 warp | 4488.931 | `gm_g1_b256_warp12_traceoff_4489us_atomic_dcci_per_builder_clock_swimlane.json` |
| 双 builder，每侧 8 warp | 4109.439 | `gm_g1_b256_warp8_traceoff_4109us_atomic_dcci_per_builder_clock_swimlane.json` |

最终复现入口为：

```bash
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-g1
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-g1 --device 0 --batches 256 --runs 21

tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-g1-swimlane
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-g1-swimlane --device 0 --batches 256 --runs 1 \
  --swimlane-json <新的、不重复的输出文件名>
```

本阶段的 `kMaxBuilderCount=2` 只是已经实现和实测的边界，不是 SIMT 调度
只能占用两个 Vector 的架构结论。后续性能扫描应把 builder AIV 数量独立参数化
为 `1..N`：每增加一个 builder 都要同步扣减一个 AIV executor，按总 leader
数重算唯一 task 分片，并分别量化严格 insert poll、构建关键链、Vector task
排队和最终端到端时间。只有端到端最优点才能决定应占用几个 Vector，不能从
本轮“双 builder 优于单 builder”外推“双 builder 永远最优”。

本轮结论是：GM 当前最佳实测为双 AIV builder、每侧 16 warp、静态唯一
task 分片和直接诊断 report store，B256 最佳扫描中位 **3.637 ms**、提交前
独立复测中位 **3.743 ms**。相对 15 ms 基线已经消除了最显著的异常 gap，
但仍明显高于约 1.5 ms 的 same-core 参考；按用户要求先提交这一版证据和
可复现状态，再决定是否继续缩短严格 insert 关键链。

## 17. 2026-08-06：Direct-GM 的 1～8 Builder 与联合 warp 扫描

### 17.1 本轮目标和边界

用户明确指出：用于 SIMT 调度的 Vector 数量也是可调参数，不能把双 AIV
builder 当成最终拓扑。因此本轮仍只处理已经功能闭合的 Direct-GM，U2 的
源码草案完整保留但不继续开发，也不把它混入本轮结论。保持不变的合同包括：

- builder AIV 完全不执行 task，executor AIV 完全不参与构建；
- 每个 SIMT warp 仍只有 lane0 工作，不回退到 Main Scalar 构建；
- 每个 task 只有一个静态 writer，五类 task、DAG、payload、strict insert、
  四 token 执行、fanin、completion 和 final drain 均不改变；
- 性能只使用 trace-off 的 ACL kernel event；trace-on 只解释自身泳道；
- 每个性能候选单独保存 JSON，继续拒绝覆盖已有文件。

开始实现前先复查本机 `ops-nn`。`hash/embedding_hash_table_export` 的 arch35
SIMT kernel 在固定 `LAUNCH_BOUND` 下使用运行时 `dim3{maxThreadNum_}`，
`init_embedding_hash_table` 也存在独立的实际 thread 数选择，证明活跃 SIMT
thread 数并非必须等于单一固定值。本阶段没有照抄源码，也没有新增仓间依赖；
为了让 CPU、device、host、LAUNCH_BOUND、trace 容量和构建清单始终同源，
先采用构建期参数，而不是临时扩展 PA control ABI。

### 17.2 参数化实现

首轮 builder 数上界从 2 扩到 8。这个 8 只是有界搜索范围，不是 A5 硬件
上限；B=8 时仍有 56 个 AIV executor。如果最优点落在 B=8，才需要继续
扩大边界。本轮的实际映射为：

```text
W = SIMT_CROSS_CORE_G0_BUILDER_WARP_COUNT，默认 16
logical_leader = builder_instance * W + local_warp
task_owner = task_id % (B * W)
task_stride = B * W
AIV executor count = 64 - B
```

实现没有复制第三套 kernel：

- `common/full_pa_model.h` 将诊断/host/device builder 上界同步扩到 8，增加
  `SIMT_CROSS_CORE_G0_BUILDER_WARP_COUNT`，合法范围为 1～64；
- `gm/cpu/build_g0.sh` 与 `gm/ccec/build_g0.sh` 从
  `SIMT_CROSS_CORE_GM_BUILDER_WARPS` 读取同一构建值并同步传给所有产物；
- host、CPU oracle、设备入口和报告校验全部接受 1～8 个 builder；B1/B256
  winner 分布必须与静态映射精确一致；
- `run.sh` 新增通用 `build-gm`、`run-gm --builders N`、
  `build-gm-swimlane` 和 `run-gm-swimlane --builders N`，原有固定 B=1 的
  G0 与固定 B=2 的 G1 入口继续兼容；
- profiling 的 SIMT writer 上界随 `8 * W` 推导，每 writer record 容量从
  固定 8192 改为 `ceil(1280 / W) * 40 + 16`。默认 W=16 时 trace state 为
  47,025,408 B，生产 state 为 32,007,296 B；连续 poll 仍合并成一个有区间
  的事件并保留精确次数。

### 17.3 完整构建与静态验证

默认 W=16 的完整入口为：

```bash
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-gm
```

它覆盖 CPU optimized 的 B=1..8、B1/B256 和四轮同地址复用，并覆盖
ASan+UBSan、TSan；随后分别构建 AIC/AIV bitcode、检查 SIMT atomic/fence、
Scalar DCCI、Vector add/multiply、Cube matmul、drain atomic，最后检查静态
1:2 mixed ELF、metadata、本地内存预算和 ACL host。W=8、12、24 的 B=4
候选也分别重新执行了同一完整构建门槛，不能把只改宏后的一次上板当成有效
性能点。所有构建和静态门槛均通过。

默认 W=16 的 B1 真实 A5 smoke 为 B=1..8 各 1/1 PASS。B1 只有 5 个 task，
按当前映射都落在 builder0，因此它只证明参数化 launch/oracle 闭合，不用于
判断多 builder 负载均衡。B256 才是性能与 winner 分布的主口径。

### 17.4 固定 16 warp 的 B=1..8 扫描

设备为 ACL 报告的 `Ascend950PR_958b` device0；环境仍没有
`npu-smi`/`task-submit`，是用户授权的 unlocked 单卡运行。每组 B256 均为
11/11 PASS，1280 个 task 都只有一次 builder attempt；表中 poll 为该组
多轮输出的典型总量，只用于解释趋势，不冒充稳定常量。

| Builder 数 B | AIV executor | builder_wins | insert poll 典型值 | min/us | median/us | avg/us | max/us | 相对 B1 |
| -----------: | -----------: | ------------ | -----------------: | -----: | --------: | -----: | -----: | ------: |
| 1 | 63 | 1280 | 约 4.08 万 | 6814.707 | 6871.043 | 6937.757 | 7583.894 | 基线 |
| 2 | 62 | 640/640 | 约 6.4 万 | 3709.292 | 3747.599 | 3796.482 | 4383.371 | -45.46% |
| 3 | 61 | 432/432/416 | 约 6.5 万 | 2419.847 | 2435.067 | 2493.910 | 3091.143 | -64.56% |
| **4** | **60** | **320/320/320/320** | **约 10.3 万** | **2060.366** | **2076.408** | **2135.426** | **2727.320** | **-69.78%** |
| 5 | 59 | 256/256/256/256/256 | 约 16.8 万 | 2070.682 | 2082.290 | 2137.793 | 2722.898 | -69.69% |
| 6 | 58 | 224/224/208/208/208/208 | 约 23.5 万 | 2101.184 | 2117.766 | 2178.894 | 2795.951 | -69.18% |
| 7 | 57 | 192/192/192/176/176/176/176 | 约 30.4 万 | 2142.236 | 2151.302 | 2220.668 | 2828.489 | -68.69% |
| 8 | 56 | 160/160/160/160/160/160/160/160 | 约 35.4 万 | 2087.441 | 2099.708 | 2158.992 | 2739.923 | -69.44% |

B=4 是明确的内部最优点，不在 B=8 搜索边界，因此本轮没有理由继续占用更多
Vector。B=1→4 把中位缩短 4,794.635 us；B=4→8 没有继续缩短，反而多
23.300 us。新增 builder 同时发生三件事：descriptor/payload 构造继续分摊，
strict insert 仍是一条全局串行链，且每增加一个 builder 就少一个 AIV
executor。B≥4 后第一项基本饱和，跨 AIV 的前驱轮询和 executor 损失开始抵消
收益；这也是不能用“SIMT 数翻倍”直接外推墙钟减半的具体数据。

八份对应泳道的 raw 汇总如下。这里的 span 和 atomic 都来自 trace-on，记录
本身会显著扰动运行，只能对图内事件负责；DCCI 随 builder 增加每次少 20，
是因为被占作 builder 的 AIV 不再执行 terminal-token 等 Scalar DCCI。

| Builder B | trace-on device span/us | SIMT atomic call | Scalar atomic call | Scalar DCCI call |
| --------: | ----------------------: | ---------------: | -----------------: | ---------------: |
| 1 | 约 9226 | 58,649 | 2,166,119 | 14,988 |
| 2 | 约 4986 | 95,647 | 1,098,082 | 14,968 |
| 3 | 约 4969 | 257,221 | 1,073,236 | 14,948 |
| 4 | 约 4980 | 411,377 | 1,037,409 | 14,928 |
| 5 | 约 4994 | 561,001 | 1,018,887 | 14,908 |
| 6 | 约 4974 | 704,636 | 989,791 | 14,888 |
| 7 | 约 4981 | 852,415 | 952,284 | 14,868 |
| 8 | 约 4952 | 991,431 | 923,974 | 14,848 |

### 17.5 固定四 builder 的 warp 联合扫描

只扫 builder 数仍可能把“B=4 最好”与每个 builder 的活跃 warp 数混在一起。
因此固定 B=4，再对 W=8/12/16/24 做有界扫描；每个点同样 11/11 PASS：

| 每 builder warp W | 每 builder thread | 总 leader | builder_wins | insert poll 典型值 | min/us | median/us | avg/us | max/us | 相对 W16 |
| ----------------: | ----------------: | --------: | ------------ | -----------------: | -----: | --------: | -----: | -----: | -------: |
| 8 | 256 | 32 | 320/320/320/320 | 约 4.6 万 | 2181.513 | 2239.565 | 2286.970 | 2890.144 | +7.86% |
| 12 | 384 | 48 | 324/324/320/312 | 约 6.95 万 | 2216.341 | 2230.759 | 2286.503 | 2853.491 | +7.43% |
| **16** | **512** | **64** | **320/320/320/320** | **约 10.3 万** | **2060.366** | **2076.408** | **2135.426** | **2727.320** | **最优** |
| 24 | 768 | 96 | 336/320/312/312 | 约 13.2 万 | 2179.024 | 2191.867 | 2252.824 | 2864.483 | +5.56% |

W=8/12 虽然 poll 更少，但用于 descriptor/payload 构造的独立 leader 不足；
W=24 又产生更多 outstanding strict-insert poll 和 SIMT atomic 压力。W=16
同时被更低和更高样本包围，因此当前联合最优为 B=4、W=16。W=12/24 的
1280 task 不能在各 builder 完全均分，winner 分布由静态映射精确解释，并非
丢 task 或重复构建。

### 17.6 最终复测、泳道与自动校验

恢复默认 W=16 并重新完整构建后，最终生产产物结果为：

| 配置 | 样本 | min/us | median/us | avg/us | max/us | 结果 |
| ---- | ---: | -----: | --------: | -----: | -----: | ---- |
| B4/W16，B1 | 5 | 148.926 | 149.701 | 287.256 | 821.506 | 5/5 PASS；首轮 warm-up 821.506 us |
| B4/W16，B256 | 21 | 2051.269 | **2067.660** | 2096.794 | 2734.328 | 21/21 PASS；每轮 wins=320×4 |

最终 21 轮中位只比扫描轮低 8.748 us（0.42%），说明 2.07 ms 结论可复现。
当前 Direct-GM 的约 1.5 ms same-core 参考仍有约 0.57 ms 差距，但原双 builder
阶段的 3.64～3.74 ms 异常 gap 已继续缩到 2.07 ms。下一步若继续优化 GM，
应针对全局 strict-insert 链、跨 builder publication 与 builder/executor
重叠找证据，不能靠继续盲加 AIV。

本轮在 `test_record/2026-8-6/` 保存 12 份互不覆盖的泳道：

| 扫描 | trace-off median/us | 文件 |
| ---- | ------------------: | ---- |
| B1/W16 | 6871.043 | `gm_b1_b256_warp16_traceoff_6871us_atomic_dcci_per_builder_clock_swimlane.json` |
| B2/W16 | 3747.599 | `gm_b2_b256_warp16_traceoff_3748us_atomic_dcci_per_builder_clock_swimlane.json` |
| B3/W16 | 2435.067 | `gm_b3_b256_warp16_traceoff_2435us_atomic_dcci_per_builder_clock_swimlane.json` |
| B4/W16 扫描轮 | 2076.408 | `gm_b4_b256_warp16_traceoff_2076us_atomic_dcci_per_builder_clock_swimlane.json` |
| B5/W16 | 2082.290 | `gm_b5_b256_warp16_traceoff_2082us_atomic_dcci_per_builder_clock_swimlane.json` |
| B6/W16 | 2117.766 | `gm_b6_b256_warp16_traceoff_2118us_atomic_dcci_per_builder_clock_swimlane.json` |
| B7/W16 | 2151.302 | `gm_b7_b256_warp16_traceoff_2151us_atomic_dcci_per_builder_clock_swimlane.json` |
| B8/W16 | 2099.708 | `gm_b8_b256_warp16_traceoff_2100us_atomic_dcci_per_builder_clock_swimlane.json` |
| B4/W8 | 2239.565 | `gm_b4_b256_warp8_traceoff_2240us_atomic_dcci_per_builder_clock_swimlane.json` |
| B4/W12 | 2230.759 | `gm_b4_b256_warp12_traceoff_2231us_atomic_dcci_per_builder_clock_swimlane.json` |
| B4/W24 | 2191.867 | `gm_b4_b256_warp24_traceoff_2192us_atomic_dcci_per_builder_clock_swimlane.json` |
| B4/W16 最终复测 | 2067.660 | `gm_b4_b256_warp16_traceoff_2068us_bounded_trace_capacity_atomic_dcci_per_builder_clock_swimlane.json` |

最后一份新图没有覆盖扫描轮 W16 图。它的 trace-on ACL event 为
5526.665 us、Scalar device span 为 4927.921 us，SIMT/Scalar atomic call
分别为 408,502/1,044,639，DCCI 为 14,928 次/行。文件名中的 2068 us 来自
独立 trace-off 21 轮中位，不是 trace-on 数值。

12 份图全部用 `jq` 自动核验：schema 均为
`simt_cross_core_g0_swimlane_v4`；每份 builder clock span 数与文件中的 B
一致；每份都有 6400 个 SIMT build 完整区间和 1024 个 task_execute；零/负
区间为 0；逐 task 检查 execute 早于自身 ordered_insert 结束的违规为 0。

### 17.7 复现命令与阶段结论

默认最优配置的生产复现命令为：

```bash
SIMT_CROSS_CORE_GM_BUILDER_WARPS=16 \
  tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-gm
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-gm --builders 4 --device 0 --batches 256 --runs 21
```

warp 候选必须先按相同值重建，例如：

```bash
SIMT_CROSS_CORE_GM_BUILDER_WARPS=8 \
  tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-gm
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-gm --builders 4 --device 0 --batches 256 --runs 11
```

泳道使用同一 warp 构建参数和独立输出名：

```bash
SIMT_CROSS_CORE_GM_BUILDER_WARPS=16 \
  tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-gm-swimlane
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-gm-swimlane --builders 4 --device 0 --batches 256 --runs 1 \
  --swimlane-json <新的、不重复的输出文件名>
```

阶段结论：**当前 Direct-GM 首轮有界最优配置为 4 个 builder AIV、每个
16 warp/512 thread，B256 trace-off 中位 2.068 ms。** B=4 与 W=16 都是
内部最优点，不是搜索边界；这是一份有真实 A5 功能 oracle、完整构建门槛和
独立泳道支撑的当前结论。U2 仍保持进行中，本阶段没有把它写成完成。

## 18. 2026-08-06：Direct-GM 单代表地址候选降至 0.710 ms（历史、已淘汰）

> 本章保留当时的实验过程和原始数据，但不再代表最终实现。该候选
> 只更新一个代表 `last_writer` 地址，依赖 PA 三个 accumulator 同步推进的
> 算子性质，既不泛化，也没有完整发布三条 symbol writer 链。第 20 章已将其
> 替换为由 tensor access/ref 驱动的通用 writer-intent 合同。

### 18.1 性能目标与实际瓶颈

本阶段的目标是将已经功能闭合的 Direct-GM B256 生产路径降到
0.8 ms 以内。性能口径仍是 trace-off ACL kernel event 多轮中位数；
trace-on 只用于解释事件数和因果关系。本轮只修改 Direct-GM，没有继续
UBUF/U2 功能开发，U2 继续保留原有全 task 严格插入链。

两个局部构建优化先确认了非串行部分的上限：

| Direct-GM B4/W16 阶段 | B256 样本 | trace-off median/us | 说明 |
| --------------------- | --------: | ------------------: | ---- |
| 上一阶段已提交基线 | 21 | 2067.660 | 全部 1280 task 参与 strict insert |
| 相邻 producer base 复用 | 11 | 1809.952 | producer-base atomic load 由 2048 次降为 1280 次 |
| descriptor 单次解码 | 11 | 1622.330 | type/shape/address 只计算一次，再写 16 words |
| 同源复测 | 7 | 1629.391 | 作为修改 strict insert 前的直接对照 |

这些改动能从 2.068 ms 降到约 1.63 ms，但仍然无法接近 0.8 ms。
泳道和 report 都指向同一个瓶颈：1280 个 task 都在等待
`task[N-1].insert_completion`，本来可并行的 descriptor/payload 构建被
一条全局串行链重新排队。

曾实现并完整验证“63 个 prepare leader + AIV0 warp0 单一顺序 committer”
候选，功能 7/7 PASS，但 B256 中位反而为 **4685.685 us**。它把
1280 次 commit 完全集中到一个 leader，直接丢掉了现有的跨 warp/AIV
并行性，因此已回退，也没有保留该候选的泳道。

### 18.2 用真实 writer 集合收缩串行链

重新按五类 PA task 的实际写集合审核后，只有每个 batch 的 UP task
会修改 shared TensorMap writer metadata：

- UP 写自身 `writer_history`，并将同 batch Alloc 的 `last_writer[0]`
  从 Alloc task id 更新为 UP task id；
- Alloc/QK/PV/SF 不修改这组 writer metadata，它们没有语义上的
  metadata 前驱；
- 因此 Direct-GM 的严格链只需要包含 256 个 UP，而不是 1280 个
  task。

新协议的次序为：

```text
所有 task：并行完成 descriptor / output publication / payload

UP[batch]：
  atomic 等待本 batch Alloc.output[0].published == Alloc task id
  -> batch>0 时等待 UP[batch-1].insert_completion
  -> 写 UP.writer_history
  -> CAS Alloc.last_writer[0]: Alloc task id -> UP task id
  -> 发布本 UP.insert_completion

Alloc/QK/PV/SF：
  不进入 metadata predecessor 链
  不发布自身 insert_completion
  -> 直接发布 completion 或 BUILT
```

UP 不能因为稀疏化就在 Alloc descriptor 尚未发布时提前改
`last_writer`。因此在进入 UP 前驱链之前，新增了对本 batch Alloc
`published[0]` 的 atomic acquire poll。这个目标等待是真实数据依赖，不是
用隐式 SIMT DCache 一致性假设代替。首个 UP 没有 metadata 前驱；后续
UP 的前驱从 `N-1` 变为 `N-5`。

`kBuildInsertCommittedBit` 在 Direct-GM 非 UP task 上现在表示“该 task 的
commit/metadata 决策阶段已完成”，不再表示该 task 一定写过
`insert_completion`。host 和 CPU oracle 均按这一最终语义检查：UP 最终
completion 必须等于自身 id，非 UP 必须保持初始值；稀疏前缀最终必须
等于 batch 数 256。

### 18.3 重新扫描 builder 数

串行链改变后，旧的 B4 最优结论不再可直接沿用。固定 W=16，重新在
真实 A5 `Ascend950PR_958b` device0 扫描 B=2..8。各点 B256 均 7/7
PASS（B4 额外执行 11 轮），1280 个 task 的 winner 分布与静态映射精确
一致：

| Builder B | AIV executor | trace-off median/us | 相对 0.8 ms | 结论 |
| --------: | -----------: | ------------------: | ----------: | ---- |
| 2 | 62 | 1357.270 | +557.270 | builder 构建仍在关键路径 |
| 3 | 61 | 1054.585 | +254.585 | 仍未达标 |
| 4 | 60 | 818.473 | +18.473 | 接近目标，11 轮 min/avg/max=805.078/875.354/1453.989 |
| **5** | **59** | **703.530** | **-96.470** | **扫描最优** |
| 6 | 58 | 782.534 | -17.466 | 达标，但已开始损失 executor |
| 7 | 57 | 805.833 | +5.833 | 越过最优点 |
| 8 | 56 | 825.443 | +25.443 | 继续增加 builder 无收益 |

B=5 不在搜索边界，且同时被 B4/B6 包围。稀疏链取消了大量串行
poll 后，多一个 builder 能继续分摊 descriptor/payload 构建；超过 B5 后，
少一个 AIV executor 的代价开始超过构建收益。

最终对 B5/W16 执行 21 轮独立 trace-off 复测：

| 配置 | PASS | min/us | median/us | avg/us | max/us | builder_wins |
| ---- | ---: | -----: | --------: | -----: | -----: | ------------ |
| B5/W16，B256 | 21/21 | 698.815 | **709.769** | 741.563 | 1371.725 | 256×5 |

max 来自首轮冷启动；后续 warm 样本稳定在约 0.699～0.722 ms。以中位数
计，相对稀疏化前的同源 1.629 ms 缩短 **56.4%**，相对上一阶段已
提交的 2.068 ms 基线缩短 **65.7%**。

### 18.4 atomic 数量和泳道证据

生产 report 中的 predecessor poll 从稀疏化前每轮约 13.3 万次降到
B5 复测的约 3215～3971 次。新 v5 泳道另外把 UP 获取本 batch Alloc
publication 的等待记录为独立站点
`simt_metadata_output_published_poll`，不把它混入 predecessor poll。

| trace-on 证据 | B4/W16 旧全 task 链 | B5/W16 稀疏 UP 链 |
| ------------- | ------------------: | ----------------: |
| schema | v4 | v5 |
| device span/us | 5220.165 | 1068.623 |
| SIMT atomic calls | 565,148 | 28,056 |
| SIMT poll calls | 550,295 | 14,226 |
| predecessor poll records | 1,279 | 255 |
| predecessor poll calls | 548,951 | 12,038 |
| metadata-target poll records/calls | 无独立站点 | 256 / 256 |
| Scalar atomic calls | 1,090,857 | 146,379 |
| Scalar poll calls | 1,078,644 | 134,474 |
| Scalar DCCI calls/lines | 14,928 | 14,908 |

两份 trace-on 因 builder 数和协议均不同，上表用于证明 poll 量级和事件
归因，不用 device span 代替 trace-off 生产性能 A/B。新图保存为：

`test_record/2026-8-6/gm_b5_b256_warp16_traceoff_710us_sparse_metadata_writer_atomic_dcci_per_builder_clock_swimlane.json`

该文件约 18 MiB，没有覆盖任何旧泳道；文件名中 710 us 来自独立
trace-off 21 轮中位，文件内 1068.623 us 是加上 atomic/DCCI 记录后的
trace-on device span。

### 18.5 验证和复现

已通过的门槛包括：

- CPU optimized 的 builder=1..8、B1/B256 和同地址复用；
- CPU ASan+UBSan 和 TSan；
- AIC/AIV CCEC bitcode、SIMT atomic/fence、Scalar DCCI、Vector/Cube
  intrinsic、mixed ELF/metadata 以及 host 构建；
- 真实 A5 B256 的 B2..8 扫描、B5 21/21 复测和完整 host oracle；
- v5 JSON 可由 `jq` 解析，256 个 metadata target poll 与 255 个前驱
  poll record 精确闭合。

最后只将无前驱 task 的 `predecessor_task` 明确设为 `UINT32_MAX`，避免
在未使用路径中依赖无符号回绕；该收尾不改协议。之后重新执行了全量
CPU optimized/ASan+UBSan/TSan、CCEC/ELF/host 构建以及真机短回归：
B1 3/3 PASS；B256 7/7 PASS，中位 686.169 us，首轮冷启动
1328.394 us，其余 warm 样本为 683.158～694.717 us。这个短回归用于
证明最终源码未回退，正式性能结论仍使用样本更多的 21 轮 709.769 us。

生产复现：

```bash
SIMT_CROSS_CORE_GM_BUILDER_WARPS=16 \
  tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-gm
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-gm --builders 5 --device 0 --batches 256 --runs 21
```

泳道复现（必须使用新文件名，不覆盖旧图）：

```bash
SIMT_CROSS_CORE_GM_BUILDER_WARPS=16 \
  tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-gm-swimlane
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-gm-swimlane --builders 5 --device 0 --batches 256 --runs 1 \
  --swimlane-json <新的、不重复的输出文件名>
```

当阶段候选结论为 B5/W16、B256 trace-off 21 轮中位 0.710 ms。
这个数据只描述上述不完整候选，已被第 20 章的通用实现取代；不能再称为
当前最优或最终性能。trace-on 1.069 ms 也始终不是生产性能口径。

## 19. 2026-08-06：U2 四槽 UBUF 完整 PA 功能闭合

### 19.1 同源实现边界

U2 没有复制第二套 PA 调度器。本阶段提交时继续复用 G0 的五类 task、DAG、
heap/output publication、writer metadata、当时的全 task 严格 insert、AIC/AIV
executor、token、completion 和 drain；同一份 kernel、CPU model 和 ACL host
通过编译期 transport policy 生成独立 U2 产物。新增内容只包括：

- U1/U2 共用的四槽几何与 64-bit generation 状态编解码；
- U2 control、四槽状态、前后 guard、逐 task staging report 和 host oracle；
- 可执行 task 的 UBUF payload sink，以及同一 SIMT leader 的 UBUF 读取到
  普通 GM 逐 word store；
- `build-u2`、`run-u2` 和独立 ACL 初始化配置。

Alloc 不占槽。QK/SF/PV/UP 固定使用 `task_id % 4`，generation 固定为
`TaskBatch(task_id)`；第一 batch 的 task1..4 同时持有四个槽，等
`count=4 && mask=0xf` 后才继续。每个 payload 只搬运精确的 74/76/75/124
个 64-bit word，SF/PV 尾部和其余 GM tail poison 都由 host 独立检查。
payload transport 本身不使用 MTE3；完整 PA workload 原有的合法 MTE3 不受
这个局部门槛影响。

### 19.2 真机栈溢出的定位与修复

初版正式配置为 SIMT 1024 B、DVG 1536 B。CPU、CCEC、bitcode、mixed ELF
和 host 都通过，但真实 A5 在第一次 B1 launch 的
`aclrtSynchronizeStream` 返回 507015。debug plog 中只有 AIV0 给出非零
硬件错误码 354，运行时原文为 `The VEC SIMT stack overflows`；其余 AIC/AIV
停在等待 BUILT 或 drain 的位置，是 builder 未完成后的连带超时，不是多个
独立根因。

保持同一个 U2 ELF，只改变第一次 `aclInit(configPath)` 的两项容量，得到：

| SIMT stack/B | DVG stack/B | A5 B1 | 结论 |
| -----------: | ----------: | ----- | ---- |
| 1024 | 1536 | 507015，AIV0 error 354 | 初版失败 |
| 1024 | 4608 | 507015，AIV0 error 354 | 增大 DVG 无效 |
| 1536 | 1536 | PASS | SIMT 容量修复根因 |
| 1536 | 4608 | PASS | DVG 不是本次瓶颈 |

CCEC metadata 同时报告 SU 808 B、SIMT 456 B、DVG 1280 B。456 B 是产物记录
值，但没有覆盖本次动态调用链的真机容量结论，不能用它反推 1024 B 一定够。
独立最小 stack probe 的 1024/512 B 已在 A5 20/20 PASS，也只证明配置入口和
那个最小 kernel 成立。最终 U2 因此使用 SIMT 1536 B、DVG 1536 B；没有沿用
GM profiling 的 1536/4608 B，也没有恢复初始化后的 `rtDeviceSetLimit`。

### 19.3 CPU、编译产物与静态门槛

正式 `build-u2` 已重新完整通过：

- CPU optimized：B1/B256、四槽 ordered generation、四 anchor、精确 payload
  word/tail、严格 insert、完整 PA oracle、持槽故障 cleanup 和同地址复用四轮；
- CPU ASan+UBSan：B1/B256 与同地址复用两轮；
- CPU TSan：B1/B256 与同地址复用两轮；
- CCEC AIC/AIV、optimized bitcode、transport-only bitcode、静态 1:2 mixed
  ELF、metadata 和 GCC15 ACL host；
- transport-only bitcode 保留 AS6 volatile UBUF load/store 和普通 GM store，
  且不含 MTE3/UBTOOUT；完整 bitcode 保留合法 workload MTE3；
- 4608 B staging、16 KiB compiler UB、192+16/224 KiB AIV local budget，以及
  ACL 1536/1536 B 配置均由构建门槛检查。

### 19.4 真实 A5 完整 PA 结果

正式 `run-u2` 在 `Ascend950PR_958b` device0 上得到：

| 配置 | PASS | kernel event/us | insert polls | max insert polls |
| ---- | ---: | --------------- | -----------: | ---------------: |
| B1 run1 | 1/1 | 906.693 | 18 | 8 |
| B1 run2 | 1/1 | 247.425 | 16 | 7 |
| B1 run3 | 1/1 | 234.634 | 18 | 8 |
| **B1 汇总** | **3/3** | **median 247.425** | - | - |
| B256 run1 | 1/1 | 42564.777 | 100227 | 549 |
| B256 run2 | 1/1 | 41966.263 | 100473 | 531 |
| B256 run3 | 1/1 | 41911.259 | 100328 | 529 |
| **B256 汇总** | **3/3** | **median 41966.263** | - | - |

每轮均通过 5/1280 task、4/1024 executable task、完整 DAG/golden、四槽
最终 `FREE(1/256)`、每槽 acquire/release `1/256`、anchor `4/0xf`、
`maxbusy=4`、guard、report checksum、SF/PV 尾部和同地址复用检查。这里的
约 42 ms 是 U2 首版全 task 严格链的功能数据，不作为 Direct-GM 的性能替代，
也不声称 UBUF 已完成性能优化。
提交前在最终 host 产物上再次补跑 B1 1/1，功能校验全部通过，
`[SUMMARY] U2` 标签与独立 U2 产物一致；该次 unlocked event 为 886.081 us，
只用于确认最终产物，不替换上表的三轮数据。

### 19.5 既有路径回归与复现

重新构建后，真实 A5 的回归结果为：

| 路径 | 配置 | 结果 | kernel event/us 或关键终态 |
| ---- | ---- | ---- | -------------------------- |
| G0 | B1/1 builder | 1/1 PASS | 774.073 us |
| G0 | B256/1 builder | 1/1 PASS | 3013.275 us |
| G1 | B1/2 builders | 1/1 PASS | 769.413 us，wins `5/0` |
| G1 | B256/2 builders | 1/1 PASS | 1974.994 us，wins `640/640` |
| U0 | 64 task/单槽/3 runs | 3/3 PASS | acquire/release `64/64`，mte3=0 |
| U1 | 128 task/四槽/3 runs | 3/3 PASS | generation32，maxbusy4，mte3=0 |

设备环境没有 `task-submit`，`npu-smi` 也不在 PATH，因此这些功能回归按已授权
方式在 device0 unlocked 运行；host 实际识别到的 SoC 为
`Ascend950PR_958b`。性能数字只用于记录本次功能运行，不当作隔离 benchmark。

复现命令：

```bash
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-u2
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-u2 --device 0 --batches 1 --runs 3
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-u2 --device 0 --batches 256 --runs 3
```

阶段结论：U2 已完成与 G0 同 task/DAG/golden 的四槽 UBUF 完整 PA 功能迁移；
Direct-GM 与 UBUF 两条路径继续独立共存。本章的全 task 严格链和
ACL 1536/1536 B 是该历史阶段的基线；第 20 章已用通用 writer-intent 合同
替换严格链，并根据新产物 metadata 将 DVG 调整为 2048 B。

## 20. 2026-08-06：取消算子特化，GM/U2 共用通用 writer-intent 合同

### 20.1 不可越过的泛化边界

第 18 章的 0.710 ms 候选依赖“PA 的三个 accumulator 永远同步推进”，
只用 Alloc 的一个 `last_writer` 代表三个 symbol。这是针对特定算子的快捷路径，
不能作为最终实现。本轮先明确两层边界：

- PA 测试 workload 层可以定义 task shape、tensor access tag 和 tensor ref；
- 调度协议层只能消费 access/ref 导出的 writer intent，不得识别
  `TaskKind::Up`、Alloc、固定 `N-5`、固定三个 slot 或代表地址。

构建脚本对 GM 和 U2 都增加了静态门槛：`SimtCommitTask` 的 metadata 提交段
必须读取 generic contract、遍历 `writer_count` 和解码 symbol key，该区间一旦出现
`TaskKind::Up` 分支就直接构建失败。

### 20.2 通用协议实现

task schema 现在显式提供五种 access：`Input`、`Output`、`Inout`、
`OutputExisting` 和 `NoDependency`。通用判定规则只有一条：
`Inout/OutputExisting + SharedOutputRef` 才是 metadata writer intent。

每个有 writer intent 的 task 在 `FullPaTaskPlan.metadata_insert_contract` 中保存：

- present bit 和 writer 数；
- task-id 序中上一个真正含 writer intent 的 task；
- `writer_history` 中每个 intent 的 packed symbol key。

SIMT commit 按以下顺序处理任意 writer 数和任意 symbol 集合：

```text
校验 generic contract / symbol key
  -> 等待每个目标 output.published
  -> 等待上一个实际 metadata-writer task.insert_completion
  -> 逐 symbol atomic-load last_writer
  -> 写入所有 {symbol_key, previous_writer} history
  -> asc_threadfence
  -> 逐 symbol CAS last_writer: previous_writer -> task_id
  -> 发布当前 task.insert_completion
```

本 PA schema 导出 256 个 writer task，每个恰好有 3 个 writer intent；CPU model 和
host 独立 oracle 均校验全部 768 个 symbol 的 key、previous writer、history 和
`last_writer`。“256×3”是这份 workload schema 的输出，不是协议常量。

曾考察用 `asc_atomic_exch` 合并“读 previous + CAS 发布”。查证 A5 SIMT API 后
确认指令存在，但协议上不成立：exchange 返回 previous 时已经把新 writer 发布
到 `last_writer`，而完整 history 尚未写回。reader 可能看到新 writer 却读不到它的
history，违反“history 先完整、CAS 后发布”的边界，因此没有实装。

### 20.3 Direct-GM 通用版重新扫描

通用版每个 writer 必须完整执行 3 次 last-writer load 和 3 次 CAS，所以第 18 章
的 B5/W16 0.710 ms 不再可比作正确实现。协议改变后重新扫描，结果如下：

| warps/builder | builders | B256 trace-off median/us | 说明 |
| ------------: | -------: | -----------------------: | ---- |
| 16 | 1 | 2828.050 | builder 不足 |
| 16 | 2 | 2155.391 | - |
| 16 | 3 | 1960.322 | - |
| 16 | 4 | 1706.745 | - |
| 16 | 5 | 1407.327 | W16 最优 |
| 16 | 6 | 1774.211 | executor 减少后回退 |
| 16 | 7 | 1916.283 | - |
| 16 | 8 | 1927.498 | - |
| 8 | 4 | 1447.557 | 缩小 builder 内活跃 warp |
| 8 | 5 | 1251.393 | W8 最优 |
| 8 | 6 | 1313.846 | - |
| 4 | 5 | 1537.262 | - |
| **4** | **6** | **1206.623** | **7 轮扫描最优** |
| 4 | 7 | 1261.554 | - |
| 4 | 8 | 1263.407 | - |

最终对 B6/W4 执行了 21 轮独立 trace-off 复测：

| 配置 | PASS | min/us | median/us | avg/us | max/us | builder wins |
| ---- | ---: | -----: | --------: | -----: | -----: | ------------ |
| B6/W4，B256 | 21/21 | 1226.394 | **1233.298** | 1265.972 | 1905.112 | 216/216/212/212/212/212 |

设备上另一 session 同时运行，且环境没有 `task-submit`，因此数字是已授权的
unlocked 结果，不声称为独占设备峰值。但 21 轮均通过完整 PA oracle，中位数
不受首轮冷启动 max 代替。

新的通用版泳道保存为：

`test_record/2026-8-6/gm_b6_b256_warp4_traceoff_1233us_generic_writer_intent_atomic_dcci_per_builder_clock_swimlane.json`

文件约 19 MiB，`jq` 解析通过，没有覆盖旧图。文件名的 1233 us 来自上述
trace-off 21 轮中位；该 trace-on 运行的 kernel event 为 2087.661 us、图内
device span 为 1669.084 us，不用来替换生产口径。

| 通用 writer 泳道证据 | records | atomic calls |
| -------------------- | ------: | -----------: |
| output publication poll | 768 | 768 |
| metadata predecessor poll | 255 | 2447 |
| last-writer load | 768 | 768 |
| last-writer CAS | 768 | 768 |
| 全部 SIMT atomic | 17445 | 42757 |
| 全部 Scalar atomic | - | 315297 |
| Scalar DCCI | 2234 records | 14888 calls/lines |

768 次 load 和 768 次 CAS 与 `256 writer task × 3 symbol`精确闭合，证明最终
版没有再使用单代表地址。

### 20.4 U2 通用化、候选回退与真机数据

U2 与 GM 现在消费同一份 writer-intent 合同。U2 B256 的 predecessor poll 由
第 19 章全 task 链的约 10 万次降到约 1.5～1.8 千次，结构性去掉了非 writer
的假串行等待。但 UBUF payload 运输仍为主要开销，所以墙钟没有按 poll 数
等比收益。

两类不依赖算子的 UBUF 运输候选也做了真机验证，但均明确回退：

| U2 B256 候选 | trace-off median/us | 结论 |
| ------------ | ------------------: | ---- |
| 32 lane warp 协作 UBUF→GM | 49994 | 比串行 leader 更慢，回退 |
| 8 lane 协作搬运 | 53354 | 更慢，回退 |
| 同一重排循环限制为 1 lane | 49947 | 说明重排循环本身有回退，回退 |
| 在 UBUF staging 内提前折叠 checksum | 53705 | 把校验计算搬到 BUILT 前关键路径，回退 |

最终源码上的 U2 真机数据：

| 配置 | PASS | min/us | median/us | avg/us | max/us | predecessor polls |
| ---- | ---: | -----: | --------: | -----: | -----: | ----------------: |
| B1 | 3/3 | 241.659 | **254.465** | 466.573 | 903.596 | 小样本 |
| B256，较少干扰时段 | 11/11 | 39043.438 | **39198.059** | 39234.888 | 39865.471 | 约 1.5～1.8k |
| B256，同机其他 session 活跃时 | 11/11 | 46227.093 | **46369.125** | 46813.481 | 49214.775 | 约 1.5～1.8k |

两组 B256 使用同一协议代码，差异与 unlocked 设备竞争同时出现，因此不把
39.198 ms 宣称为稳定独占性能，也不把 46.369 ms 误判为代码回退。可确定的
是完整 oracle 均 PASS，且 metadata predecessor poll 稳定从约 10 万降到千级。

通用 helper 使 CCEC 最终报告 SU 808 B、SIMT 496 B、DVG 1920 B。因此 U2
ACL-init 配置从历史的 1536/1536 B 调整为 **SIMT 1536 B / DVG 2048 B**，
继续使用 512 B 对齐容量，不使用初始化后的 runtime limit 修改。

### 20.5 完整验证与复现

本轮已通过：

- GM CPU optimized：builder=1..8 的 B1/B256 与同地址复用；
- GM CPU ASan+UBSan 和 TSan；
- GM AIC/AIV CCEC bitcode、mixed ELF/metadata、GCC15 host；
- U2 CPU optimized/ASan+UBSan/TSan 的 B1/B256 和同地址复用；
- U2 完整 bitcode、transport-only bitcode、mixed ELF/metadata、GCC15 host；
- 真实 A5 GM B6/W4 B256 21/21，U2 B1 3/3 与 B256 两组 11/11；
- 通用 GM 泳道 host 域校验、完整 oracle 和 `jq` JSON 校验。

最后将 `TaskKind` 限制在 full-PA schema adapter、让设备侧 writer-intent 显式改为消费
`TensorAccess` 后，又重新通过 GM B6/W4 B256 3/3（中位 1260.971 us）和 U2
B1 1/1（908.501 us）。这是最终源码的功能回归；正式 GM 性能口径仍使用
样本更多的 21 轮中位 1233.298 us。

GM 生产性能复现：

```bash
SIMT_CROSS_CORE_GM_BUILDER_WARPS=4 \
  tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-gm
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-gm --builders 6 --device 0 --batches 256 --runs 21
```

GM 通用泳道复现（必须使用新文件名）：

```bash
SIMT_CROSS_CORE_GM_BUILDER_WARPS=4 \
  tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-gm-swimlane
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-gm-swimlane --builders 6 --device 0 --batches 256 --runs 1 \
  --swimlane-json <新的、不重复的输出文件名>
```

U2 复现：

```bash
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-u2
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-u2 --device 0 --batches 1 --runs 3
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-u2 --device 0 --batches 256 --runs 11
```

最终结论：**保留的优化是与算子无关的 writer-intent 稀疏链，而不是 PA
单代表地址快捷路径。Direct-GM 当前正确实现的最优实测配置为 B6/W4，
B256 trace-off 21 轮中位 1.233 ms。U2 没有保留任何真机回退的搬运候选，
只保留同一份通用 metadata 协议优化。**

## 21. 2026-08-06：按 symbol 解开 writer 假串行并将 GM 降至 0.495 ms

### 21.1 问题与泛化边界

第 20 章的正确通用基线为 B6/W4、B256 trace-off 21 轮中位
1233.298 us。泳道中的 256 个 metadata writer 虽然各自写三个真实 symbol，
但旧 contract 只保存“上一个任意 metadata writer”，因而把不同 batch、不同
`(producer_task, output_slot)` 的 writer 也串入同一条链。与此同时，每个
writer 在等待全局前驱后还要对三个 `last_writer` 分别执行一次 atomic add 0，
再把读到的值写入 history；总数为 768 次 load 和 768 次 CAS。

本轮没有恢复已经淘汰的 PA 单代表地址，也没有在调度 commit 中识别
`TaskKind::Up`。保留的通用规则是：

1. workload schema adapter 继续只把 tensor access tag 和
   `SharedOutputRef` 转成 writer intent；
2. 调度层以 `(producer_task, output_slot)` 生成稳定 symbol key；
3. prepare 向前查找同一 symbol 的最近 writer，把精确 expected writer 与
   symbol key 一次性写入 history；
4. commit 先等待每个目标 output 发布；只有 expected writer 大于原始 producer
   时，才等待该真实前驱 task 的 `insert_completion`，相同前驱去重；
5. history 完整后执行 fence，再用 history 中的 expected writer 对每个
   `last_writer` 做 CAS；CAS 仍是校验和发布边界，不能换成提前发布的 exchange；
6. 当前 writer 的所有 symbol 都提交完成后，才发布自己的
   `insert_completion`。

因此，如果多个 task 写同一 symbol，仍保持严格的同地址顺序；互不相干的
symbol 不再产生假依赖。full-PA 当前 256 个 writer 的 768 个 symbol 都只在
本 batch 内使用，精确前驱均为各自的原始 producer，所以 metadata 前驱等待
由 255 个 episode 降为 0；768 次 `last_writer` atomic load 也降为 0，768 次
校验 CAS 全部保留。

CPU 语义模型同步取消了只为旧全局链服务的 `committed_prefix`，改为无顺序含义
的 writer 完成计数。独立 ACL host oracle 仍按 PA 参数 schema 自己重建所有
writer intent、history、最终 last-writer 和 builder report，不复用 device
结果作为 golden。GM/U2 构建脚本新增静态门槛：history 必须在 prepare 中携带
精确前驱；commit 中不得出现 `SimtMetadataLastWriterLoad`；metadata commit
仍禁止 `TaskKind::Up` 分支。

### 21.2 去掉重复反推

第一版虽然已经令 `insert_polls=0`，但 prepare、commit 校验和 builder report
三处重复扫描 schema，B6/W4 的 7 轮中位反而为 1338.412 us。随后只在 prepare
计算一次精确前驱；commit 直接消费已发布 history，并在实际等待位置累计
report，不再第二、第三次反推。相同 B6/W4 的 7 轮中位降到 1004.599 us。

这一步说明收益来自协议关键路径缩短，而不是仅把 poll 计数改成 0。相对第 20
章的 1233.298 us 正确基线，同配置 B6/W4 减少约 228.7 us。

### 21.3 Builder Vector 范围扩展与完整扫描

W4 下 B8 的五轮中位已到 853.362 us，且最优点落在原 `1..8` 人工边界。该上限
不是硬件或协议限制，因此将生产、CPU model、host 参数、builder report、
profiling writer 容量和命令行统一扩为 `1..16`。B16 使用 16 个 AIV builder，
每个 builder 为 4 warp/128 thread、仍只有四个 lane0 构建；总计 64 个活跃
builder lane，每个 AIV 精确构建 80 个 task，同时保留 48 个 AIV executor。

同一 W4 产物、B256、trace-off、完整 oracle 的五轮扫描结果如下。每组第一轮
包含明显冷启动，表中使用五轮中位；设备没有 `task-submit`，属于已授权的
unlocked 单卡运行。

| 配置 | builder wins | min/us | median/us | 结论 |
| ---- | ------------ | -----: | --------: | ---- |
| B4/W4 | `320×4` | 1397.277 | 1411.036 | 慢于基线候选 |
| B5/W4 | `256×5` | 1715.496 | 1722.056 | 非单调回退 |
| B6/W4 | `216/216/212/212/212/212` | 1001.950 | 1010.067 | 精确前驱后的同配置参考 |
| B7/W4 | `184×5/180×2` | 955.423 | 963.980 | 继续下降 |
| B8/W4 | `160×8` | 843.140 | 853.362 | 撞到原上限 |
| B9/W4 | `144×5/140×4` | 679.917 | 698.637 | 首次稳定低于 0.8 ms |
| B10/W4 | `128×10` | 906.192 | 914.448 | 拓扑/执行资源折中回退 |
| B11/W4 | `120/116×10` | 595.367 | 612.900 | 低于 0.8 ms |
| B12/W4 | `108×8/104×4` | 625.430 | 632.195 | 低于 0.8 ms |
| B13/W4 | `100×8/96×5` | 592.169 | 601.497 | 低于 0.8 ms |
| B14/W4 | `92×12/88×2` | 510.459 | 514.598 | 继续下降 |
| B15/W4 | `88×5/84×10` | 650.978 | 665.864 | 非单调回退 |
| B16/W4 | `80×16` | 478.333 | **486.552** | 五轮扫描最优 |

B16/W4 随后完成两组 21 轮确认。协议实现后的首组为 min 474.400 us、median
487.891 us；最终格式化、静态门槛和全量构建后的当前源码复测为：

| 配置 | PASS | min/us | median/us | avg/us | max/us | insert polls |
| ---- | ---: | -----: | --------: | -----: | -----: | -----------: |
| B16/W4，B256 | 21/21 | 480.000 | **494.528** | 524.730 | 1153.834 | 0 |

首轮 1153.834 us 是同一进程的新鲜初始化冷样本；其余 20 轮为
480.000～513.583 us。正式结论仍使用预先约定的 ACL device-event 21 轮中位，
即 **0.495 ms**，已经低于 0.8 ms 目标。相对第 20 章 1.233 ms 正确基线，
中位减少 738.770 us，约 59.9%。其中从 B6/W4 约 1.005 ms 到 B16/W4 约
0.495 ms 的多 builder 扩展贡献约 0.510 ms，是本轮最大单项墙钟收益。

### 21.4 独立泳道图

没有覆盖任何旧 JSON。新图为：

`test_record/2026-8-6/gm_b16_b256_warp4_traceoff_488us_per_symbol_writer_expected_cas_atomic_dcci_per_builder_clock_swimlane.json`

文件约 18 MiB，`jq` 解析通过。文件名的 488 us 来自同一协议实现第一组 21 轮
trace-off 中位 487.891 us；最终当前源码的复测口径采用上一节 494.528 us。
trace-on 自身的 kernel event 为 1227.949 us，只用于解释埋点图，不代替
trace-off 性能。图内关键数据为：

| 项目 | 数值 |
| ---- | ---: |
| trace-on device span | 630.996 us |
| SIMT build span | 565.319 us |
| SIMT / Scalar atomic calls | 24053 / 68199 |
| Scalar DCCI calls / cache lines | 14688 / 14688 |
| metadata output publication poll | 768 |
| metadata last-writer load | **0** |
| metadata last-writer CAS | **768** |
| metadata predecessor poll | **0** |
| metadata insert-completion publish | 256 |

这组计数证明图与最终协议一致：没有用单代表地址减少真实 symbol 数，也没有用
exchange 取消 CAS；消失的是不同 symbol 的假等待和可静态确定的 atomic load。

### 21.5 验证、复现与 UBUF 决策

最终修改已通过：

- GM CPU optimized、ASan+UBSan、TSan：builder=1..16，B1/B256，同地址复用；
- GM AIC/AIV CCEC、optimized bitcode、1:2 mixed ELF/metadata、GCC15 host；
- 真实 A5 B16/W4 B256 21/21，完整 task/DAG/payload/history/last-writer/
  builder report/oracle 和同地址复用全部 PASS；
- profiling 变体 CCEC/ELF/host、真实 A5 1/1 和 `jq` JSON 校验；
- U2 共享协议的 CPU 三套、CCEC/transport bitcode/mixed ELF，以及 A5 B1
  3/3 回归；U2 B1 最终中位 260.008 us，`insert_polls=0`。

GM 生产性能复现：

```bash
SIMT_CROSS_CORE_GM_BUILDER_WARPS=4 \
  tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-gm
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-gm --builders 16 --device 0 --batches 256 --runs 21
```

GM 泳道复现时必须使用新的、不重复的文件名：

```bash
SIMT_CROSS_CORE_GM_BUILDER_WARPS=4 \
  tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh build-gm-swimlane
tests/atomic_probe/pa_scheduler/simt_cross_core/run.sh \
  run-gm-swimlane --builders 16 --device 0 --batches 256 --runs 1 \
  --swimlane-json <新的输出文件名>
```

根据当前实现和本 workload 的数据流，UBUF 不能消除最终跨核 GM 写，反而增加
`写 UBUF -> 读 UBUF -> 写 GM`、slot CAS、guard 和生命周期管理。用户据此决定：
**后续停止 UBUF 性能优化**。U0/U1/U2 保留为功能与一致性验证路径；共享协议
变化时只做必要回归，不再扫描或优化 UBUF 性能。

## 22. 2026-08-07：增加 trace-off 构建包络并完成 B32/W4 收尾扫描

### 22.1 为什么需要新的构建时间口径

第 21 章的 B16/W4 trace-off ACL kernel-event 中位为 494.528 us，但该数值
覆盖整个 mixed kernel；已有泳道中的 `SIMT VF build` 为 trace-on 显示区间，
又受到逐 atomic/DCCI 记录和跨时钟域展示映射影响。两者都不能直接回答
“SIMT 构建本身是否低于 0.3 ms”。

本轮没有把逐 task trace 带入生产产物，而是在每个 builder AIV 的 Main
Scalar 壳中只增加两个 `get_sys_cnt()` 采样：

```text
reserved[0] = pre_async_invoke
  -> async_invoke<SIMT VF>
  -> set_flag / wait_flag
reserved[1] = post_wait_flag
```

`get_sys_cnt()` 在本机 A5 上是跨 AIV 共用的 1 ns 时基。host 取所有 builder
的最早 `reserved[0]` 到最晚 `reserved[1]`，输出
`BUILD_PERF/BUILD_PERF_SUMMARY`；该包络包含 VF 启动偏斜、完整 SIMT 构建和
VF join，是 trace-off 下可直接用于判断 0.3 ms 目标的有效墙钟。host 同时
验证 builder 的两个采样严格递增、非 builder 对应字段保持 0，ABI 大小没有
变化。共享该 kernel/host 的 U2 已重新通过 CPU optimized、ASan+UBSan、TSan、
CCEC/bitcode/mixed ELF/host 全套门槛，并在真实 A5 上通过 B1 1/1；该轮构建
包络为 228.932 us，只作为共享计时代码的功能回归，不重启 UBUF 性能优化。

### 22.2 W4 扩展到 32 个 builder

builder 软件搜索上限由 16 扩到 32，CPU model、device、host 参数、命令行、
builder report 和泳道 writer 容量使用同一个 `kMaxBuilderCount`。B32 占用
32 个 AIV builder，每个 builder 仍为 4 warp/128 thread、每个 warp 仍只有
lane0 工作；总计 128 个活跃 leader，每个 leader 精确构建 10 个 task，并
保留 32 个 AIV executor 和全部 32 个 AIC executor。

W4 完整产物通过 builder=1..32 的 CPU optimized、ASan+UBSan、TSan，随后
通过 AIC/AIV CCEC bitcode、1:2 mixed ELF/metadata 和 GCC15 host 构建。真实
A5 使用用户授权的 unlocked device0；除明确列出的噪声失败外，完整
task/DAG/payload/history/last-writer/golden oracle 均通过，`insert_polls=0`。

保留的关键 trace-off 数据如下。每组 3 轮的第一轮包含明显 ACL 冷样本，表中
仍按程序固定口径报告全部有效样本的中位，不擅自删除冷样本：

| 配置 | PASS | kernel event median/us | build envelope median/us | 说明 |
| ---- | ---: | ---------------------: | -----------------------: | ---- |
| B16/W4 | 3/3 | 566.337 | 420.559 | 新构建口径下的 B16 端点 |
| B19/W4 | 3/3 | 459.812 | 382.242 | 构建继续下降 |
| B21/W4 | 3/3 | 440.432 | 369.842 | 首次低于 0.37 ms |
| B23/W4 | 3/3 | 441.123 | 361.719 | 接近后段平台 |
| B29/W4 | 2/3 | 753.159 | 337.802 | 一轮外部噪声触发 global fatal，不作为候选 |
| B31/W4 | 3/3 | 431.227 | 354.826 | 端到端较快但构建未达 0.3 ms |
| B32/W4 | 3/3 | **436.673** | **348.332** | 当前可复现端点；稳定两轮为 410.717/436.673 us |

B32/W4 的三轮原始 kernel-event 为
`1092.371/436.673/410.717 us`，构建包络为
`353.991/348.332/343.668 us`。因此本轮只能得出：端到端稳定区间约
0.41～0.44 ms，trace-off SIMT 构建约 0.35 ms；**0.3 ms 构建目标没有
达到**，不能用较短的总 kernel 样本冒充构建达标。

### 22.3 W8 对照与停止结论

为了区分“总 leader 数”与“占用 builder AIV 数”，另构建了 W8 完整产物，
同样通过 builder=1..32 的 CPU 三套和完整 CCEC/ELF/host 门槛。每个 warp
仍只有 lane0 工作，没有引入同 warp 多 lane 协作。真实 A5 对照为：

| 配置 | 总活跃 leader | PASS | kernel event median/us | build envelope median/us |
| ---- | ------------: | ---: | ---------------------: | -----------------------: |
| B16/W8 | 128 | 5/5 | 441.221 | 381.533 |
| B24/W8 | 192 | 5/5 | 451.232 | 377.741 |
| B32/W8 | 256 | 2/5 | 468.790 | 371.299 |

B16/W8 与 B32/W4 都有 128 个活跃 leader，但构建包络分别为
381.533 us 和 348.332 us，说明把更多 warp 堆到更少 AIV 上更慢；B32/W8
还出现 3/5 global fatal，不能作为有效性能候选。W8 路线据此否决，不固化为
默认配置。

### 22.4 B32/W4 独立泳道

没有覆盖旧图。B32/W4 使用单独的 W4 profiling 产物，真实 A5 1/1 完整
oracle 通过，JSON 已通过 `jq` 解析：

`test_record/2026-8-6/gm_b32_b256_warp4_traceoff_437us_build348us_atomic_dcci_per_builder_clock_swimlane.json`

文件名中的 437 us 和 348 us 分别来自上一节独立 trace-off 的 kernel-event
中位 436.673 us 与 build-envelope 中位 348.332 us。泳道本身开启埋点，不能
与 trace-off 混用；图内数据为：

| 项目 | 数值 |
| ---- | ---: |
| trace-on kernel event | 1069.007 us |
| trace-on device span | 481.350 us |
| trace-on SIMT build 显示包络 | 404.694 us |
| trace-on `get_sys_cnt` build envelope | 404.685 us |
| SIMT / Scalar atomic calls | 30795 / 29272 |
| Scalar DCCI calls / cache lines | 14368 / 14368 |
| builder wins | `40×32` |
| metadata insert poll | 0 |

这份图只用于保存 B32/W4 的 atomic、DCCI、builder/executor 重叠证据；是否达到
0.3 ms 仍只看 trace-off build-envelope，结论仍是未达到。

### 22.5 补齐 Startup 到 FinalDrain 的同口径设备时钟

为了与 Scalar `cross_core_DAG` 的端到端性能直接比较，trace-off 产物复用
`FullPaRoleResult::reserved[2:3]` 新增两个低频 `get_sys_cnt()` 边界：96 个
角色在配置取得后记录 startup，root 在所有角色到达、completion 校验和
`root_finished` 发布后记录 FinalDrain 终点。host 取最早 startup 到 root
终点；不新增 ABI、不增加逐 task 记录，终点只旁路写一个既有 64-bit 槽。

用 `SIMT_CROSS_CORE_GM_BUILDER_WARPS=4` 重建 B32/W4 后，CPU builders=1..32
三套、CCEC/bitcode/mixed ELF/GCC15 host 全部 PASS。正式 A5 B256 十轮
10/10 PASS：

- startup 到 root FinalDrain 中位 `394.563 us`；
- ACL kernel event 中位 `422.813 us`；
- builder 包络中位 `343.760 us`。

这证明旧 `436.673 us` ACL event 与完整设备周期处于同一量级；它不能与
builder 包络混称，但也没有隐藏毫秒级执行尾部。一次 W16 误构建得到的
`606.719 us` 已由设备输出的 `warps_per_builder=16` 识别并排除，不作为 W4
结论。

用户在此阶段决定性能优化告一段落。本轮保留的是：

- 可复用的 trace-off 构建包络测量与 host 校验；
- builder=1..32 的统一有界配置能力；
- B32/W4 与 W8 否决数据，以及“构建仍未低于 0.3 ms”的诚实结论。

不再继续扩展 builder、不再增加 warp、不恢复 PA 特例，也不继续 UBUF 性能
优化。后续若重新开启性能工作，必须从本节的独立 build-envelope 口径继续，
不能只凭 ACL kernel-event 猜测构建时间。

### 22.5 Scalar task 展示与真实 workload 口径纠正

用户检查新图时指出 AIV task 只有约 5 us，而此前 cross_core 的 SF 应约
50 us。对照实际 JSON 和源码后确认这不是绘图缩放误差：SIMT G0 host 一直把
QK/SF/PV/UP 的 `repeats` 全部固定为 1；standalone shared same-core/
cross-core 的 A5 标定默认值是 `6/28/4/1`，其中 SF 的 28 次完整 128x128
Vector add 正是约 50 us 的来源。

本轮在 SIMT 自有 model 中冻结同一组 `6/28/4/1` 常量，没有引入对 cross_core
源码的依赖。base B32/W4、B256 在真实 A5 上 5/5 通过，纠正后的性能为：

| 口径 | min/us | median/us | avg/us | max/us |
| ---- | -----: | --------: | -----: | -----: |
| kernel event | 984.649 | **1014.989** | 1129.106 | 1634.115 |
| SIMT build envelope | 346.248 | **347.307** | 348.696 | 355.491 |

首轮 1634.115 us 是未锁设备上的冷/噪声样本，但仍保留在全部 5 轮统计中。
旧的 436.673 us kernel 中位和 348.332 us build 中位是
`repeats=1/1/1/1` 的历史事实；构建时间仍有效，端到端时间不再作为与
same-core/cross-core 可比的结论。

展示方式也按 cross_core 的物理 block 合同重做，不再把 96 条 Scalar 和
全部 SIMT warp 拆成两个全局 process：

- 未参与 SIMT 的 block 严格保留 1C2V 顺序：AIC/AIV0/AIV1 Scalar 在上，
  AIC/AIV0/AIV1 kernel 在下；
- 参与 SIMT 的 block 中，每个 builder AIV 单独形成 `SIMT Scalar host + N 条
  SIMT warp` 调度组；调度组之后，AIC 仍按 `AIC Scalar + AIC kernel` 的旧
  方式展示；
- 若只有一个 AIV 是 builder，另一个普通 AIV 会和 AIC 一样保留 Scalar 和
  kernel 两条轨，因此布局逻辑不依赖 B32；
- Scalar 执行区间命名为 `task.execute.QK/SF/PV/UP#task_id`，kernel 轨命名为
  `QK/SF/PV/UP#task_id`；两者是相同 workload 起止时间的重复投影，不能相加；
- direct atomic、DCCI 和 workload 只在真实 Scalar 轨展示，不复制到 kernel
  轨；完整 lifecycle/fanin 时间戳保留在 raw trace 中供 host 校验。

首版边界仍包住了 workload 后的 witness/completion/DONE atomic，SF 中位为
60.233 us，尚不等价于 cross_core 的 `TracePhase::Kernel`。最终版把
workload begin/end 与完整 task end 分开：`task.execute.*` 只显示 engine
workload，外层 lifecycle 继续覆盖完成发布。为避免扩大 profiling 写竞争，
没有把 `ExecutorTaskTrace` 扩成两条 cache line；task id 由数组下标确定，
kind/phase 压入一个 32-bit 字段，因而新增 workload end 后记录仍严格为原来
的 64 B。

随后用户指出 AIC Scalar 明明只有一个线程，图中却被展开成很多层。检查事件
和源码后确认：每个 executor Scalar 确实只有一个物理线程，但
`kTokensPerOwner=4`，该线程会交错推进四个 token。上一版把每个 token 从首次
poll 到就绪的逻辑 episode，以及 ticket 到 task end 的 lifecycle，都作为
连续 `X` 区间放进同一个 `tid`。block0 AIC 因此出现最多 4 个互相重叠的 poll
episode；叠加 role/lifecycle/wait 后，viewer 最大展开深度达到 14。这不代表
14 个线程，而是错误地把“逻辑 pending 时间”当成了“物理线程连续占用时间”。

v8 改为严格单物理线程展示：

- executor Scalar 持续区间只保留 workload、direct atomic 和 DCCI；
- 不再导出 role、executor loop、task lifecycle 和宽泛 wait 包络；这些原始
  时间戳仍在 64 B trace 记录中，并继续参与 host 时序校验；
- 合并 poll 改成 episode 结束处的 instant marker，`call_count`、逻辑包络起点
  和包络时长完整保存在 args，既不丢统计，也不再占据四条重叠显示行；
- SIMT warp 内的 build/claim/prepare/insert 分层仍保留，因为那是用户要求的
  SIMT 调度结构，不冒充 AIC/AIV Scalar 线程。

v8 虽然消除了假多线程层级，但用户继续指出 AIC Scalar 出现大量空白。对 v8
真机图量化后，32 条 AIC 平均内部空白为 304.268 us，其中 280.876 us 位于
至少一个 pending poll episode 内，占 92.3%；block0 分别为 326.788 us、
294.701 us。空白不是 idle 的可靠证据，而是 poll 只保留结束 marker、普通
Scalar 控制代码和 trace record 写入没有独立持续区间造成的展示缺口。

v9 因此继续在 host 导出阶段计算“精确持续区间的补集”：

- workload、direct atomic、DCCI 保持原始精确边界并拥有最高优先级；
- 补集处若有 pending poll episode，生成单层 `scalar.poll/control`，args 记录
  pending episode 数；否则按 role phase 生成 scheduler/setup/drain control；
- 每个补位段明确标为 host synthesized，表示 poll/control 未逐条定时，并且
  可能包含 profiling 写 record 的开销，不冒充单次 atomic；
- 补位算法只读现有 raw trace 并生成 JSON，不增加 device record、atomic、DCCI
  或 kernel 执行开销；
- 所有补位与精确事件互斥，因此 Scalar 始终是一条连续且不重叠的物理时间线。

最终按物理 block 重排后的真实 A5 图中，各类均为 256 个样本：

| kind | cross_core median/us | SIMT median/us | SIMT min/p95/max us |
| ---- | -------------------: | -------------: | ------------------: |
| QK | 40.954 | 41.081 | 40.216 / 49.361 / 51.982 |
| SF | 53.575 | **56.920** | 52.077 / 61.878 / 67.431 |
| PV | 27.460 | 27.306 | 26.792 / 34.740 / 36.746 |
| UP | 2.533 | 2.050 | 1.872 / 2.992 / 3.244 |

最终 B32 图的 block0..15 各有 12 轨（两个 builder 调度组和两条 AIC 轨），
block16..31 各有 6 轨（标准 1C2V），共 288 个唯一 `(pid, tid)`。Scalar 与
kernel 各有 1024 个 workload 区间，按 task 对比后的起点、时长、owner、
kind 和 engine 差异为 0；SIMT/Scalar atomic 与 DCCI 的错轨数也为 0。
64 条 executor Scalar 的全部持续区间最大重叠深度均为 1，重叠轨数为 0；
相邻持续区间之间的空白轨数和空白总时长也均为 0。1793 个 Scalar poll
record 全部保持 instant marker，与 summary 精确一致。host 另外生成 15918
段 `scalar.poll/control`（合计 20921.962 us）和 2942 段其他 control（合计
1204.904 us），全部只占 exact 事件的补集。
另用 B1/W4/B1 真机覆盖了奇数 builder 分支：block0 为一个五轨 SIMT 组加
AIC/AIV1 的两条 Scalar 和两条 kernel，block1 仍是标准六轨 1C2V，完整
oracle 和 4 对 Scalar/kernel workload 均通过；95 条 executor Scalar 的
重叠轨数与空白轨数同样都是 0。
trace-on device span 为 1082.567 us、SIMT build span 为 403.544 us、kernel
event 为 1678.656 us；完整 PA oracle 1/1 通过。该 v9 JSON 是展示实现过程中的
中间证据，最终已在第 22.8 节收敛为 v13 后删除，不再作为当前入口。

本轮 base/swimlane 共同通过 CPU optimized、ASan+UBSan、TSan、AIC/AIV CCEC、
bitcode、mixed ELF 和 GCC15 host 门槛。两个已判定错误的 2026-08-07 中间
JSON 以及后续错误的全局 Scalar 中间图均已删除；上述 v9 图作为“首次连续
Executor Scalar”历史版本保留，后续 v10/v11 也使用新文件名，没有覆盖它；
2026-08-06 的历史 v5 图同样未覆盖。

### 22.6 Scalar 等待对象与普通代码的完整归因

用户继续指出 `scalar.poll/control` 只能说明“这里有空白”，看不出 Scalar
究竟在等待什么、poll 本体开销多少，也把本来可以由控制流确定的普通 Scalar
代码混成了一个兜底类别。最终 v11 仅修改 host JSON 导出，device raw ABI 仍为
6，没有增加设备记录和 profiling 热路径开销。

新的单物理线程展示同时保留三层证据：

1. workload、direct atomic、DCCI 继续使用设备记录的精确起止时间；
2. 三个合并 poll site 分别显示为 `exec_state`、`fanin_flag`、`drain_arrival`；
   每个 episode 仍在结束点保存精确 `call_count` 和逻辑 envelope；
3. 精确事件的剩余补集，根据 role phase 与相邻的 `previous_exact/next_exact`
   拆为 payload bind、claim、dispatch、engine prepare、completion、token rotation、
   config 和 drain 等阶段，不再生成 `scalar.control`。

poll 宽条命名为 `[AtomicPoll+GM+Scalar]`，表示“这一物理 Scalar 墙钟区间内至少
一个相应 poll episode 处于 pending”，而不是宣称整条都是 atomic 指令。
marker 另给出 `call_count * 160 ns` 的 Atomic 本体参考值。这样能同时回答：

- 等待对象是谁；
- episode 墙钟占了多少；
- 实际执行了多少次 atomic load；
- 若按 standalone probe 的约 160 ns/次估算，Atomic 本体约占多少；
- 两者之间为何不能直接画等号。

最终 B32/W4/B256 v11 真机图完整 PA oracle 1/1 通过：

| 项目 | v11 数据 |
| ---- | -------: |
| trace-on device span | 1073.060 us |
| trace-on kernel event | 1673.578 us |
| trace-on SIMT build span | 405.684 us |
| Scalar poll record / call | 1793 / 31973 |
| `call_count * 160 ns` | 5115.680 us |
| 分类后的 poll 墙钟条，96 轨累加 | 21533.369 us |
| 普通 Scalar 具体阶段，96 轨累加 | 1397.176 us |
| `scalar.control` | 0 段 / 0 us |

poll 墙钟条进一步分解为 fanin 13353.351 us、exec-state 4807.583 us、mixed
2672.722 us、root drain 699.713 us。普通 Scalar 中较大的部分是 payload bind
440.233 us、task completion 267.020 us、token rotation 155.187 us、config
validate 129.195 us、engine prepare 115.291 us 和 dispatch 105.613 us。所有数字
都是跨物理 Scalar 轨累加，不能当成端到端墙钟。

机械校验覆盖：全部 96 条物理 Scalar 连续且最大重叠深度为 1，空白轨和重叠
轨均为 0；1024 对 Scalar/kernel workload 的 task、owner、kind、engine、起止
和时长逐项一致；1793 个 marker 的 call count 与 31973 次 raw poll 完全闭合，
估算值严格等于 `31973 * 160 ns`。额外真实 A5 B1/W4/builder=1 也通过完整
oracle：block0 为 9 轨、其余 block 为标准 6 轨，全部 96 条 Scalar 仍为 0
空白、0 重叠，4 对 workload 投影完全一致。

v11 当时使用独立文件完成验证；最终已在第 22.8 节收敛为 v13 后删除。

### 22.7 poll 等待对象与当前 Scalar 阶段双标签

抽查 v11 的 block0 AIC 后发现，仅把补集优先画成 `wait.exec_state x3` 仍可能
隐藏四 token 交错：三个 token 的 exec-state episode 虽然 pending，当前物理
Scalar 可能正在为第四个 token 绑定 payload。v12 因此不改变区间切分，只把
已经用于普通区间归因的相邻精确事件结果也附到 poll 条上。

现在的名称直接同时显示两种信息，例如：

```text
wait.exec_state[AtomicPoll+GM+Scalar] x3 pending |
scalar.bind_and_validate_payload[GM+Scalar]#83
```

args 同时保存 `coexecuted_scalar_stage/category` 和
`previous_exact/next_exact`。poll category 仍用于按等待对象着色，`|` 后半段
说明该物理补集位于哪类 Scalar 代码的相邻精确边界之间。它不会把 episode
墙钟冒充成纯 Atomic，也不会因为另一个 token 正在等待就隐藏当前 token 的
payload/claim/completion 阶段。该变化仍只发生在 host JSON 导出，raw ABI 和
device trace 数均不变。

v12 的真实 A5 B32/W4/B256 完整 PA oracle 1/1 通过：device span
1086.576 us、kernel event 1676.610 us、SIMT build 400.234 us。全部 96 条物理
Scalar 仍是单层连续轨，空白和重叠均为 0；1024 对 Scalar/kernel workload
逐字段一致；1793 个 poll marker 合计 30673 次 load，`30673 * 160 ns =
4907.680 us`。分类 poll 墙钟条跨轨累计 21149.803 us，普通 Scalar 具体阶段
累计 1287.404 us，`scalar.control` 仍为 0。v12 当时使用独立文件完成验证；
最终已在第 22.8 节收敛为 v13 后删除。

### 22.8 固定 6/28/4/1、补齐完整 E2E 并收敛最终文件

最终口径统一为 cross_core 正式 real-compute 默认值：QK/SF/PV/UP 分别执行
`6/28/4/1` 次完整 128x128 engine pipeline。SIMT host 初始化直接写入四个
编译期常量，没有保留运行时 workload 覆盖入口；启动日志和 JSON 顶层均再次
保存实际值，避免把轻量 smoke 数据混入正式性能结果。

同一份 W4 生产产物在真实 A5、B256、关闭埋点下分别复测 B16/B32 五轮：

| 配置 | PASS | min/us | median/us | avg/us | max/us | build median/us |
| ---- | ---: | -----: | --------: | -----: | -----: | --------------: |
| B16/W4 | 5/5 | 946.581 | **956.976** | 1086.910 | 1620.327 | 418.222 |
| B32/W4 | 5/5 | 955.730 | **972.090** | 1108.654 | 1673.095 | 338.831 |

v13 在 JSON 顶部新增单独的 `Kernel end-to-end total` 轨，duration 直接来自
kernel launch 前后的 ACL event，覆盖 kernel final drain 和返回。ACL event 与
device `get_sys_cnt()` 没有公开的共同绝对 epoch，因此该轨明确标记为
`duration_reference_only`，不伪造它与设备子阶段的绝对相位关系。

每种配置生成两个 trace-on 候选，保留 ACL E2E 更短且完整 oracle 通过的一份：

| 配置 | trace-on ACL E2E/us | device span/us | build envelope/us | 最终文件 |
| ---- | ------------------: | -------------: | ----------------: | -------- |
| B16/W4 | 1600.264 | 1015.929 | 563.086 | `test_record/2026-8-7/gm_b16_b256_warp4_traceoff_957us_traceon_e2e1600us_device1016us_build563us_per_block_full_scalar_wait_stage_v13_simt_aic_atomic_dcci_swimlane.json` |
| B32/W4 | 1659.830 | 1061.737 | 396.792 | `test_record/2026-8-7/gm_b32_b256_warp4_traceoff_972us_traceon_e2e1660us_device1062us_build397us_per_block_full_scalar_wait_stage_v13_simt_aic_atomic_dcci_swimlane.json` |

两份图均有 96 条非空物理 Scalar 轨，空白轨与重叠轨均为 0；Scalar/kernel
各 1024 个 workload 投影逐字段一致；E2E 事件恰好一个且 duration 与顶层值
完全相等；1793 个 Scalar poll episode 继续合并并保留精确 call count；
`scalar.control` 为 0。B16/B32 的 SF 中位分别为 54.639/57.025 us，符合统一
workload 口径。

在 22.8 阶段，`2026-8-7` 当时只保留上述两份 v13 JSON 和一份 README；
随后 22.9 按新的六种 B、每种三种 W 要求替换为 18 份最终图。v9/v10/v11/v12
展示中间图已删除；文件名也不再重复携带 `realcompute_6_28_4_1`，固定 workload
由 JSON 元数据和文档说明承担。

### 22.9 B=1/2/4/8/16/32 的 W 扫描与 18 份最终泳道图

用户进一步要求同时观察 B=`1/2/4/8/16/32`，并让每种 B 都保留三种实测较优
的 W。该轮不修改调度协议和 workload，只对同一份通用源码改变编译期
`SIMT_CROSS_CORE_GM_BUILDER_WARPS`，继续固定 QK/SF/PV/UP=`6/28/4/1`。

测试分两层进行：先以 trace-off 五轮中位数做宽范围候选扫描，再对每种 B 的
近邻入围项统一复测 11 轮。第一轮新鲜初始化没有剔除，最终排序只采用 11/11
通过配置的 ACL kernel event 中位数。入围结果如下：

| B | 第 1 名 | median/us | 第 2 名 | median/us | 第 3 名 | median/us |
| -: | ------- | --------: | ------- | --------: | ------- | --------: |
| 1 | W16 | 2946.263 | W14 | 3303.954 | W26 | 3307.122 |
| 2 | W32 | 1646.371 | W28 | 1697.087 | W8 | 1784.520 |
| 4 | W30 | 1013.554 | W16 | 1041.676 | W15 | 1047.185 |
| 8 | W5 | **926.038** | W7 | 945.652 | W6 | 952.794 |
| 16 | W5 | 946.850 | W3 | 947.485 | W4 | 954.408 |
| 32 | W2 | 950.701 | W1 | 988.489 | W5 | 1006.230 |

本轮扫描范围内的全局最优是 B8/W5 的 926.038 us。B16/W4 与未入选 W6
只差 0.469 us，B1/W14 与 W26 只差 3.168 us，属于 unlocked device 上的
近似并列，不解释为确定的架构收益。

B32 暴露了不能仅凭性能数选配置的稳定性问题：W3 初筛为 0/5，W6、W8 为
4/5；W4 早期虽曾得到 5/5，但 11 轮复测出现一次 `fatal-nonzero`，只有 10/11。
因此 B32 最终只保留 11/11 的 W2/W1/W5，旧 W4 图退出最终目录。

复测自动化最初还捕获到一次测试基础设施问题：W1 的 CPU optimized 校验失败后，
日志 `tee` 掩盖了构建退出码，设备启动行显示实际仍是上一轮
`warps_per_builder=33`。该批数据立即作废；后续构建启用 `pipefail`，每次运行
都强制核对设备打印的 W、`workload_repeats=6/28/4/1`、PASS 分母和 build
manifest，避免把旧产物误记到新配置。这个失败没有进入上表和最终 JSON。

每个 Top-3 配置又生成两个 trace-on 候选，只保留 ACL E2E 更短且完整 PA oracle
通过的一份。最终 `test_record/2026-8-7` 恰好包含 18 份 v13 JSON，每个 B 三份；
文件名包含 B、W、trace-off 中位数以及该图的 trace-on E2E/device/build 整数 us，
完整小数和配置表记录在同目录 README。

18 份图统一通过机械校验：schema/raw ABI 和 workload 正确；96 条物理 Scalar
连续且无空白、无重叠；Scalar/kernel 各 1024 个 workload 区间按 task、pid、
owner、kind、engine、起点和时长逐项一致；唯一 E2E 轨与顶层 duration 相等；
1793 个 Scalar poll marker 的 call count 与 summary 闭合；`scalar.control` 为 0。
泳道图编译仍通过 AIC/AIV CCEC、bitcode、mixed ELF 和 GCC15 host 静态门槛，
该轮没有为特定 PA task kind 增加路径或调度特判。

## 23. 阶段状态索引

| 阶段 | 状态 | 结果/提交 |
| ---- | ---- | --------- |
| D0 文档与查证 | 完成 | `64e3d5d5`：范围、链接和空白检查通过。 |
| S0 基础协议与 SIMT 自检 | 完成 | `399d5704`：CPU 三套 PASS；A5 同 AIV 100×4 模式完成，reader DCCI 为最小可靠序列。 |
| S1 单 Vector task | 完成 | `975da5ea`：CPU 三套 PASS；1:2 mixed ELF 静态门槛 PASS；A5 跨 AIV 100×4 模式完成，reader DCCI 为最小可靠序列。 |
| S2 单 Cube task | 完成 | `2e3034d7`：CPU 三套 PASS；1:2 mixed ELF/Cube intrinsic 门槛 PASS；A5 AIV→AIC 100×4 模式完成，reader DCCI 为最小可靠序列。 |
| S3 Vector + Cube | 完成 | `5dff1124`；CPU 三套 PASS；双 task mixed ELF/Vector/Cube 门槛 PASS；A5 同地址复用 100/100，完成计数和 drain 精确闭合。 |
| A0 SIMT atomic 竞争 | 完成 | `dc61c014`：CPU 三套 PASS；CCEC/ELF 门槛 PASS；A5 32/64/1024/2048 线程各 100/100。 |
| S4 多 task、单 builder | 完成 | 初版 `a29fa08e` 完成 4-lane 基线；随后修正为 4-warp/128-thread 交错映射，CPU/CCEC/ELF PASS，A5 同地址复用 100/100，完成计数 `1/8/8/16` 精确闭合。 |
| A1 warp 推进语义 | 完成 | CPU 三套和 CCEC/ELF 门槛 PASS；A5 同地址复用 100/100，SameWarp 始终 T/S+disjoint，CrossWarp 始终 S/S+overlap。 |
| G0 GM 完整 PA | 完成 | 16-warp/lane0 纯 SIMT 构建；CPU 三套与 CCEC/ELF 门槛 PASS；A5 B1/B256 功能闭合；原始 64-warp B256 trace-off 中位约 15.0 ms。 |
| G1 双 builder GM | 完成 | 双 VF 各 512-thread/16-warp/lane0 静态唯一分片；B256 trace-off 中位 3.637 ms；五组独立 v4 atomic/DCCI 泳道已保存。 |
| GN 多 builder GM 扫描 | 完成，性能优化告一段落 | 通用版按 symbol 建立精确 writer 前驱，保留 768 次真实 CAS、去掉 255 个假等待 episode 和 768 次 atomic load。最终统一固定 `6/28/4/1`，完成 B=`1/2/4/8/16/32` 的 W 扫描和 11 轮入围复测；当前扫描范围内 B8/W5 最优中位为 0.926 ms。`2026-8-7` 按每种 B 的稳定 Top-3 保存 18 份 v13 泳道图，全部带完整 ACL E2E 参考轨。0.3 ms 构建目标仍未达到。 |
| U0 UBUF 单槽 | 完成 | 64-warp/lane0 纯 SIMT 单槽；CPU 三套、CCEC/ELF 门槛和 A5 同地址 100/100 全部 PASS；G0/G1 四组真机回归 PASS。 |
| U1 UBUF 多槽/多 task | 完成 | `a20a29e2`；CPU 三套、CCEC/bitcode/mixed ELF 门槛全部 PASS；A5 smoke 1/1 与同地址复用 100/100，四槽 `maxbusy=4`、每槽 generation `0..31`精确闭合。 |
| U2 UBUF 完整 PA | 功能完成，停止性能优化 | 同源 transport policy、四槽 ordered generation、真实 payload word/tail 与完整 PA oracle 已闭合；共享的按-symbol writer 协议在最终 B1 令 predecessor poll 为 0；CPU 三套和 CCEC/bitcode/mixed ELF 全部 PASS，真机 B1 3/3。由于 UBUF staging 不能消除最终 GM 写，后续只保留功能与共享协议回归。 |
