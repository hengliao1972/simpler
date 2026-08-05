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
   Main Scalar 是否能在完全无 DCCI 的情况下稳定读取新 payload；
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

## 3. 阶段状态索引

| 阶段 | 状态 | 结果/提交 |
| --- | --- | --- |
| D0 文档与查证 | 文档完成 | 范围/链接/空白检查通过，随本次 D0 提交交付。 |
| S0 基础协议与 SIMT 自检 | 未开始 | - |
| S1 单 Vector task | 未开始 | - |
| S2 单 Cube task | 未开始 | - |
| S3 Vector + Cube | 未开始 | - |
| S4 多 task、单 builder | 未开始 | - |
| G0 GM 完整 PA | 未开始 | - |
| G1 双 builder GM | 未开始 | - |
| U0 UBUF 单槽 | 未开始 | - |
| U1 UBUF 多槽/多 task | 未开始 | - |
| U2 UBUF 完整 PA | 未开始 | - |
