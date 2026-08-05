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

## 8. 阶段状态索引

| 阶段 | 状态 | 结果/提交 |
| ---- | ---- | --------- |
| D0 文档与查证 | 完成 | `64e3d5d5`：范围、链接和空白检查通过。 |
| S0 基础协议与 SIMT 自检 | 完成 | `399d5704`：CPU 三套 PASS；A5 同 AIV 100×4 模式完成，reader DCCI 为最小可靠序列。 |
| S1 单 Vector task | 完成 | `975da5ea`：CPU 三套 PASS；1:2 mixed ELF 静态门槛 PASS；A5 跨 AIV 100×4 模式完成，reader DCCI 为最小可靠序列。 |
| S2 单 Cube task | 完成 | `2e3034d7`：CPU 三套 PASS；1:2 mixed ELF/Cube intrinsic 门槛 PASS；A5 AIV→AIC 100×4 模式完成，reader DCCI 为最小可靠序列。 |
| S3 Vector + Cube | 完成 | `5dff1124`；CPU 三套 PASS；双 task mixed ELF/Vector/Cube 门槛 PASS；A5 同地址复用 100/100，完成计数和 drain 精确闭合。 |
| A0 SIMT atomic 竞争 | 完成 | CPU 三套 PASS；CCEC/ELF 门槛 PASS；A5 32/64/1024/2048 线程各 100/100；随本次 A0 提交交付。 |
| S4 多 task、单 builder | 未开始 | - |
| G0 GM 完整 PA | 未开始 | - |
| G1 双 builder GM | 未开始 | - |
| U0 UBUF 单槽 | 未开始 | - |
| U1 UBUF 多槽/多 task | 未开始 | - |
| U2 UBUF 完整 PA | 未开始 | - |
