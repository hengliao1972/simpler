# Scalar Cross-Core DAG 调度实现过程

本文记录 `cross_core_DAG` 的实际开发、验证、性能结果和失败候选。架构合同
以 [`DAG调度设计.md`](DAG调度设计.md) 为准；没有运行的项目必须明确写成
`NOT RUN`，不能把设计推断写成测试结果。

## 1. 2026-08-07：D0 需求对齐与源码查证

### 1.1 GitHub 同步

开始前将当前分支快进到 `8dee3431`。工作树同步前后均干净，没有 rebase、
冲突或未提交文件。

### 1.2 三轮需求结论

本实现冻结为第五套 standalone 方案：

- 从现有 Scalar `cross_core` 改造，而不是引入 SIMT 指令代码；
- 新代码全部位于 `cross_core_DAG/`，不依赖旧实现源码；
- 96 个 Main Scalar 都能动态 Build 任意 task，不设专职 builder；
- 每次中央 Build ticket 只领取一个 task；
- Build 时根据 tensor access schema 动态推导 per-symbol writer DAG；
- 同 symbol 严格串行，不同 symbol 取消全局假串行；
- 一个 writer task 在全部 symbol 提交后发布一个 completion；
- ordinary region/view/alias 保留保守 TensorMap 回退；
- task-indexed execution cell 单轮不复用；
- Execute 使用 AIC/AIV 双 cursor、两项 ticket 和每核四 token；
- Execute 不等待 96 核全部到达，只要合法工作出现就尽早推进；
- CPU 与 CCEC/A5 都要验证，不新增 AscendC；
- `perf` 与 `swimlane` 使用独立产物；
- B256 性能固定运行 10 次，以 startup 到 FinalDrain 中位数裁决；
- 每阶段更新本文并独立提交，未经授权不 push。

### 1.3 SIMT DAG 来源查证

当前 SIMT 实现没有由 host 预计算逐 task DAG。设备 builder 在
`SimtPrepareTask()` 内完成以下工作：

1. 从 PA schema adapter 取得 `TensorAccess` 与 `SharedOutputRef`；
2. 动态生成 writer intent；
3. 对每个 symbol 从当前 task 向前扫描同 symbol 最近 writer；
4. 将 `(symbol_key, previous_writer)` 写入 immutable history；
5. commit 时等待精确前驱并执行 expected CAS；
6. 全部 symbol 完成后发布当前 task 的 `insert_completion`。

host 只初始化状态并通过独立 oracle 校验最终 history、last-writer 和 DAG。
因此本 Scalar 版本也采用 Build 时动态推导，不采用此前讨论过的 host 预计算
DAG 建议。

### 1.4 Heap 与 output 查证

现有 Scalar C 与 SIMT G0 都使用 8 路 shared heap shard、task-indexed final
output descriptor 和 aggregate completion vend。Scalar C 已经把 fresh
descriptor 直接构造在最终 `shared_outputs[task_id]`，不存在必须保留的中间
descriptor 搬运。

新实现因此复用 Scalar C 的成熟 heap/output 语义，并让 host oracle 对齐 SIMT
的 interval、descriptor 和 vend 结果，不重写第二套内存所有权协议。

### 1.5 当前阶段状态

- 设计文档：完成。
- 目录合同：完成。
- CPU 代码迁移：NOT RUN。
- CCEC 构建：NOT RUN。
- 真实 A5：NOT RUN。
- 性能与泳道：NOT RUN。

下一阶段先复制现有 Scalar `cross_core` 的最小 CPU/公共实现到独立目录，删除
旧的 host writer bitset 前驱合同，增加动态 per-symbol 前驱和乱序交错测试。

## 2. 2026-08-07：S0a 独立 Scalar 基线

### 2.1 迁移范围

只复制原 `cross_core` 已跟踪的以下内容：

- `common/`；
- `cpu/`；
- `ccec/`；
- `test/`；
- `run.sh`。

没有复制 `build/`、`outputs/`、旧过程文档或方案提案。C++ 命名空间从
`pa_scheduler::cross_core` 机械改为 `pa_scheduler::cross_core_dag`，执行扫描
测试和泳道输出也使用 DAG 独立名称，防止两套 standalone 产物混淆。

Atomic/DCCI 源码覆盖脚本已复制到本目录，CPU/CCEC 构建不再调用
`same_core/` 内的脚本。源码扫描确认，除设计文档用于说明来源的链接外，
当前文件没有 include 或运行时访问 `cross_core/`、`same_core/`、
`simt_cross_core/`。

### 2.2 CPU 工具链记录

首次显式指定用户态 `g++-15` 时，GCC 15 生成了 binutils 2.42 不识别的
`.base64` 汇编伪指令，构建在首个独立测试的汇编阶段停止，尚未运行协议代码。

CPU 语义模型不依赖 GCC 15 特性，因此按脚本默认值改用系统
`g++ 13.3.0`。CCEC 和 ACL host 后续仍使用各自既有固定工具链，不由这次
CPU 选择外推。

### 2.3 行为不变基线

执行命令：

```bash
source .venv/bin/activate
PYTHON=.venv/bin/python \
  tests/atomic_probe/pa_scheduler/cross_core_DAG/run.sh \
  build-perf-clock cpu
```

结果为 PASS，覆盖：

- Atomic/DCCI 源码覆盖；
- `PollBatch`；
- 稀疏 metadata-writer completion；
- host task plan 和 random-access args；
- 96 worker 中央 Build dispatch 10 轮；
- ordinary region ring 的 32/64/128/256/16384 五档容量；
- compact/sparse trace；
- shared output、writer intent 和 8 路 shared heap；
- execution payload adapter；
- AIC/AIV 双 Execute cursor、两项 ticket 和 FinalDrain；
- 完整 ordered-submit 交错。

动态 Build 测试每轮均闭合 1,280 个 task、1,024 次 Execute 和 256 个旧式
全局 metadata writer。该结果只证明机械隔离没有改变旧协议；256 个 writer
仍由全局 writer 链串行，尚未实现本目录目标的 per-symbol DAG。

### 2.4 当前阶段状态

- 独立文件与构建目录：完成。
- 旧行为 CPU 基线：完成。
- 动态 per-symbol DAG：NOT RUN。
- CCEC 构建：NOT RUN。
- 真实 A5：NOT RUN。

## 3. 2026-08-07：S0b 动态 DAG 通用原语

### 3.1 先失败再实现

先增加不依赖 PA `TaskKind` 的 `test_shared_metadata_dag.cpp`。旧基线因缺少
`SharedDagTensor`、`SharedMetadataDag` 和 `BuildSharedMetadataDag()` 无法
编译，证明测试确实命中了尚未实现的能力，而不是重复覆盖旧全局 writer
bitset。

随后实现只消费 adapter schema 的动态 DAG 原语。schema 对每个 task 只提供
tensor 数及 `(TensorAccess, TensorRefKind, symbol_key, manual_dep)`；公共算法
不读取 PA kind、固定 task 间距或设备 `last_writer`。

### 3.2 已闭合语义

定向测试已证明：

- task 6 在 task 2/5 尚未实际 Build 时仍从只读 schema 找到 task 5；
- 同 symbol A 严格得到最近逻辑 writer，夹在中间的 symbol B writer 不参与；
- 首个 symbol B writer 直接以前序 descriptor producer 为基线，不等待
  symbol A；
- 纯 reader 也得到自己查询所需的精确早期 writer；
- ordinary input/writer 只沿 ordinary writer 保守链推进，不被 symbol-only
  task 串行；
- 同 task 重复 symbol、future producer 和越界 task 全部 fail closed。

### 3.3 回归结果与边界

`build-perf-clock cpu` 全套回归 PASS，包括新增 DAG 门槛以及 S0a 已列出的
全部旧协议测试。当前只完成“逻辑 DAG 推导”原语；生产 Build/Register 仍在
读取 host metadata-writer plan 并走旧全局 writer 链，因此本阶段不宣称运行
路径已经获得异 symbol 并行，也没有 CCEC/A5 性能结论。

## 4. 2026-08-07：S0c 动态 DAG 接入 Scalar Build/Register

### 4.1 PA schema adapter 与运行时交叉校验

新增的 PA adapter 把随机访问 Build 已有的 task identity 翻译为公共 DAG
schema：adapter 可以识别 `TaskKind`，但 `BuildSharedMetadataDag()`、等待和
commit 只消费 `TensorAccess`、引用种类及 symbol key。进入 Materialize 前，
运行时逐项比较 adapter schema 与真正构造出的 `TaskArgs`，任意 tensor 数、
access、引用种类、symbol 或 manual-dependency 漂移都在共享副作用发生前
终止。

这一步没有把 PA 固定三 accumulator 写进公共 DAG 算法；PA 只是当前
workload adapter，后续算子必须提供自己的 schema adapter。

### 4.2 Register 从全局 writer baton 改为精确前驱

生产 Build/Register 已完成以下替换：

- 不再用 `DecodeSharedMetadataWriterPlan()` 给当前 task 找一条全局
  `previous_metadata_writer`；
- 对每个 SharedOutputRef reader/writer 等待动态 DAG 去重后的精确前驱；
- UP 的三个 accumulator 不再压缩为 Alloc slot0 的 PA 专用 group word，
  而是按三个独立 symbol 分别执行 expected CAS；
- immutable history 保存每个 symbol 自己的 `previous_writer`；
- 只有全部 symbol CAS 和 ordinary metadata 提交成功，才发布本 task 唯一
  `insert_completion`；
- 任一 CAS 冲突保留已经线性化的故障前缀并进入 fatal，不伪造事务回滚，
  也不发布 completion。

不同 batch 使用不同 Alloc producer，因此不会再因为旧全局 writer 计划而
产生跨 batch 假依赖。同一 batch 内，三个 accumulator 虽由同一批 UP task
更新，协议和 host oracle 仍逐 slot 独立校验，不能再把 slot1/2 当成永不
变化。

### 4.3 CPU 验证结果

定向门槛已通过：

- `test_shared_metadata_dag`：乱序 Build、同/异 symbol、纯 reader、ordinary
  回退及非法 schema；
- `test_shared_output_symbols`：三条不同 previous 的动态数组 commit，以及
  第二条 CAS 冲突后的终止前缀；
- `test_shared_ordered_submit`：不同 batch 无伪 completion load，末组 UP
  独立推进 Alloc 的 slot0/1/2；
- `build-perf-clock cpu`：源码覆盖、host plan、random-access args、96 worker
  dispatch、heap、payload、Execute/FinalDrain 和 ordered-submit 全套 PASS；
- CPU B256：1,280 个 Build、1,024 个 kernel、2,048 个 published output、
  1,280 条 fanin edge、8 路 heap 和最终 normalized writer signature 全部
  PASS。

CPU B256 单次端到端约 `177.305 ms`，只受 host thread 调度影响，用于协议
闭合而不作为 A5 性能结论。

### 4.4 尚未完成的收口

`SharedBuildDispatchState` 中旧的 metadata writer count/bitset 当前已不被
生产 Build/Register 热路消费，但字段、host 初始化和旧门槛仍然存在。下一
阶段要删除这份死计划及其 `Decode/Validate` 接口，避免 host 数据继续冒充
DAG 权威；task identity 与 AIC/AIV Execute route 仍作为 launch 级计划单独
审视，不能与 metadata DAG 混为一谈。

CCEC 构建、A5 B1/B256、A5 端到端性能和泳道仍为 `NOT RUN`。

## 5. 2026-08-07：S0d 删除 host metadata DAG 权威

### 5.1 删除范围

`SharedBuildDispatchState` 已删除：

- metadata writer union count；
- ordinary/symbol writer 分类 count；
- 每 task metadata writer bitset；
- `DecodeSharedMetadataWriterPlan()`；
- `ValidateSharedMetadataWriterSummary()`。

host 不再初始化或校验上述字段，设备入口也不再读取 writer 摘要。Build
dispatch 只剩 task identity、task/batch/executable 数以及独立的 Execute
route；它们用于随机访问构参和 engine 路由，不提供 DAG 前驱。

删除 69 个 `uint64_t` bitset word 和行尾对齐后，cross-core execution tail
从 `19,493,824 B` 减为 `19,493,248 B`，减少 `576 B`。

### 5.2 门槛修正

- host-plan 测试只验证 task identity 与 Execute route，不再复刻一份 writer
  bitset decoder；
- random-access G1 测试直接用真实 `TaskArgs` 与 PA schema adapter 构建
  动态 DAG，四个 UP 各有三个 writer，跨 batch dependency task 数为 0；
- 96-thread Build dispatch 每个 ticket 动态构建 DAG，B256 闭合 1,280 个
  Build、1,024 个 Execute 和 256 个 writer completion；
- ordered-submit 的 host oracle 按独立 workload plan 判断 UP writer，不读取
  device 计划形成同错校验；
- 源码门槛新增禁止标识符列表，防止 count/bitset 及旧 Decode/Validate
  接口以后悄悄回流生产头。

完整 `build-perf-clock cpu` 和 CPU B256 再次 PASS。B256 单次约
`75.202 ms`，仍只作为 CPU 协议证据，不与前一轮 host 调度时间比较，也不
冒充 A5 收益。

CCEC 构建、A5 B1/B256、A5 端到端性能和泳道仍为 `NOT RUN`。

## 6. 2026-08-07：S1 CCEC 双产物闭合

### 6.1 固定环境

- CANN：`9.1.0-weekly-20260708`；
- PTO-ISA：`ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8`；
- AIC：`dav-c310-cube`；
- AIV：`dav-c310-vec`；
- shared insert-turn identity：`groups=1`；
- Scalar 自动 DCCI 与 kernel-end DCCI 均关闭，只保留源码显式协议。

### 6.2 构建结果

分别执行：

```bash
tests/atomic_probe/pa_scheduler/cross_core_DAG/run.sh \
  build-perf-clock ccec
tests/atomic_probe/pa_scheduler/cross_core_DAG/run.sh \
  build ccec
```

`perf-clock` 与 `swimlane` 两类独立产物均 PASS，覆盖：

- AIC/AIV generic shared protocol 的真实 CCEC 模板实例化；
- 两种架构各自的 entry、block-local runtime 和 noinline finish；
- 1:2 mixed AICore ELF；
- 两个入口及 kernel metadata section；
- role-specific real-compute helper 非空且不跨角色引用；
- 最终 ELF 无未解析 relocation；
- host runner；
- variant、trace ABI、insert-turn identity 与 SHA256 manifest。

`perf-clock` mixed kernel 约 `2.3 MiB`，`swimlane` mixed kernel 约
`4.1 MiB`。这里的大小只用于确认两种编译身份确实隔离，不据此推断
I-cache 或端到端收益。

CCEC 编译日志中的 warning 来自 PTO-ISA 头部的既有未使用变量和 pragma
兼容提示；构建退出码、链接检查和最终 manifest 均成功，没有把 warning
误报为错误。

### 6.3 当前边界

本阶段只证明设备源码能以 AIC/AIV 两种架构编译、静态链接并形成可校验
产物。真实 A5 B1/B256、golden、端到端性能和泳道仍为 `NOT RUN`。

## 7. 2026-08-07：S2 A5 B1/B256 正确性与 perf

### 7.1 设备入口

本机没有 `npu-smi` 和 `task-submit`，与仓库既有 A5 复现记录一致。设备
节点 `/dev/davinci0`、`/dev/davinci_manager` 和 `devmm_svm` 存在，PCI
设备为 Huawei d806；使用 CANN ACL host 对 device 0 直接同步 launch。

这属于本机已记录的设备入口约束，不能写成完成了 `task-submit` 预约，也
不能用设备节点本身替代上板执行证据。实际 B1/B256 ACL launch 均成功，
CCEC 目标继续固定为 `dav-c310-cube/vec`。

### 7.2 B1 真计算

B1 使用默认 `real-compute/6,28,4,1`，结果：

- 5 个 task 全部 Build exactly once；
- QK/SF/PV/UP 四个 kernel 各执行一次；
- 8 个 fresh output、5 次 symbol input load、3 次 INOUT symbol commit；
- descriptor、history、fanin、8 路 heap、execution payload、golden 和
  FinalDrain 全部 PASS；
- startup 到 FinalDrain 为 `173.918 us`。

### 7.3 B256 十轮 perf

B256 同样使用默认真计算，每轮都是独立 ACL launch。1,280 个 Build、1,024
个 kernel、2,048 个 published output、1,280 条 fanin edge、256 个 writer
completion 和 `206,569,472 B` heap 全部闭合；每轮
`execution/semantic/postprocess` 均为 PASS。

startup 到 FinalDrain 的十轮结果为：

```text
2.341248  2.330751  2.321785  2.320387  2.344957 ms
2.347033  2.311389  2.321238  2.337646  2.319769 ms
```

- 最快：`2.311389 ms`；
- 中位数：`2.326268 ms`；
- 均值：`2.329620 ms`；
- 最慢：`2.347033 ms`。

这是无泳道 `perf-clock` 产物的唯一性能裁决口径。swimlane 尚未运行，不能
把之后的带观察时间与本节绝对值直接相减。

## 8. 2026-08-07：S3 A5 B256 DAG 泳道闭合

### 8.1 先修正观察协议，而不是放宽错误

第一次真实 B256 采集的设备执行、语义和 golden 已全部 PASS，但公共
converter 仍按旧 `global_writer_chain` 假设，要求 host 为每个 task 声明
metadata writer 前缀，并据此断言每个前缀 task 必有一条
`SharedInsertTurnPoll`。新实现的 DAG 在 device Build 中动态推导；没有精确
同 symbol 前驱的 task 本来就不应产生该 PollBatch，因此旧断言拒绝了正确
raw。

本阶段没有用 PA 固定 task 间距在 host 侧重建 DAG，也没有简单删除结构
门槛，而是：

- DAG host raw 明确写入
  `shared_metadata_ordering=per_symbol_dag`；
- DAG raw 不再导出 `shared_metadata_prefix_tasks`，避免 host 重新成为逐 task
  DAG 权威；
- `shared_metadata_writer_tasks` 只说明哪些 task 确实发布 completion，用于
  闭合 writer handoff 数量，不描述其前驱；
- 每个 Build winner 允许零或一条聚合 `SharedInsertTurnPoll`，若存在则必须
  精确覆盖 `Register.start -> metadata.start`；重复、越界和孤儿记录继续
  fail-closed；
- producer summary 继续闭合全部物理记录、逻辑 atomic 调用、PollBatch 和
  DCCI 调用/行数；旧 `global_writer_chain` raw 仍走原有严格前缀校验。

analyzer 同步读取该协议身份。`register_wait_predecessor_insert` 字段名为保持
报告 schema 稳定而保留，但其说明已改为“等待 device 动态 DAG 的去重精确
前驱”，不再错误描述成 `task N` 等待 `task N-1`。

### 8.2 最终产物与闭合结果

最终 B256 真计算泳道目录为：

```text
tests/atomic_probe/pa_scheduler/outputs/
  pa_scheduler_cross_core_dag_swimlane_20260807_074759_2456882/ccec/
```

其中：

- raw：`l2_swimlane_records.json`，约 `3.7 MiB`；
- merged：`merged_swimlane.json`，约 `8.0 MiB`；
- 排他分析：`swimlane_exclusive_analysis.json`，约 `340 KiB`。

设备与 host 门槛全部 PASS：

- 1,280 个 Build、1,024 个 kernel、2,048 个 published output；
- 1,280 条 fanin edge、256 个 writer completion；
- AIC 只执行 QK/PV，AIV 只执行 SF/UP；
- heap、descriptor、history、execution payload、golden 与 FinalDrain 全部
  闭合；
- raw `57,105` 条、merged `72,538` 个事件，`dropped_records=0`；
- 1,024 个 kernel 均有唯一父区间：EfDrain 814 个、FinalDrain 117 个、
  OrchestrationTail 93 个，孤儿为 0。

本次 PA G1 的 DAG 关键原语数量为：

- `SharedInsertTurnPoll`：0 条物理记录、0 次逻辑调用。各 batch 的三个
  writer symbol 均只回到本组 producer，没有跨 task writer 前驱；
- `SharedMetadataOutputPublishedLoad`：768 条物理记录、1,246 次逻辑调用。
  256 个 UP 各有 3 个目标 output，轮询次数按真实可见时序累计；
- `SharedMetadataLastWriterCommit`：768 条物理记录、768 次逻辑调用，即
  256 个 UP × 3 个独立 symbol CAS；
- `SharedInsertTurnHandoff`：256 条物理记录、256 次逻辑调用，每个 metadata
  writer 在全部 symbol commit 后发布一次 task completion；
- `SharedWriterHistoryFlush` DCCI：256 条物理记录、256 次调用、256 条 cache
  line，每个 writer 精确发布一条 immutable history cache line。

`SharedInsertTurnPoll=0` 不是漏采：本用例不同 batch 使用不同 symbol，且
producer output 的取得由 `SharedMetadataOutputPublishedLoad` 单独证明。以后
加入同 symbol 多 writer 或 ordinary writer 门槛时，动态 DAG 才应产生对应
的 predecessor PollBatch。

### 8.3 时间口径

这次带完整 atomic/DCCI 观察的单轮结果为：

- host 摘要 Submit span：`1.984255 ms`；
- startup 到 FinalDrain lifecycle：`2.047027 ms`；
- analyzer 的全局 Submit raw 区间：`1.899758 ms`。

三者边界不同；此外 swimlane ELF 含观察代码，不能与无泳道 ELF 做绝对时间
相减。性能裁决仍以 S2 的十轮 `perf-clock` startup-to-FinalDrain 中位数
`2.326268 ms` 为准，本节只作为业务阶段、atomic、DCCI 与 DAG 协议证据。

## 9. 2026-08-07：S4 对齐 SIMT 端到端计时基线

### 9.1 先消除比较口径歧义

历史 SIMT B32/W4 的 `436.673 us` 是 ACL kernel event，`348.332 us` 是
builder 包络；Scalar 的 `2326.268 us` 则是设备内最早 startup 到最后
FinalDrain。三者原先不能直接相减。

本轮没有增加逐 task 记录，而是复用 SIMT `FullPaRoleResult` 的既有保留槽：

- 每个 AIC/AIV 在配置 DCCI 完成、进入 Build/Execute 前记录
  `reserved[2]`；
- root 在观察全部角色到达、验证 1,024 个 kernel completion 并发布
  `root_finished` 后记录 `reserved[3]`；
- host 取 96 个角色最早的 `reserved[2]` 到 root `reserved[3]`，作为
  `earliest_role_startup_to_root_final_drain_end`；
- root 终点只旁路写一个既有 64-bit 槽，不重写整份 128B role result，
  不把泳道或逐 task PMU 带入性能产物。

host 同时严格校验所有角色起点非零、只有 root 拥有递增终点、最后一个保留
槽保持零。ABI 大小与 task/DAG/payload 协议均未变化。

### 9.2 构建配置查证

第一次校准误用了脚本默认 W16，设备输出明确显示
`warps_per_builder=16`；其 10/10 PASS 的完整周期中位数为
`606.719 us`，不能冒充历史最优 W4。随后通过已存在的
`SIMT_CROSS_CORE_GM_BUILDER_WARPS=4` 重建 B32/W4：

- CPU builders=1..32 的 optimized、ASan+UBSan、TSan 全部 PASS；
- AIC/AIV CCEC、bitcode inventory、1:2 mixed ELF 和 GCC15 host PASS；
- 每个 builder 4 warp/128 thread，32 个 builder 共 128 个活跃 lane0
  leader，每个 builder 精确构建 40 个 task。

### 9.3 W4 十轮 A5 结果

第一组十轮出现一次 `fatal-nonzero`，只形成 9/10，不作为正式基线。未修改
代码后重新运行，得到 10/10 PASS；每轮均闭合 1,280 个 Build、1,024 个
kernel、8 路 heap、per-symbol DAG、payload、golden 和 FinalDrain。

正式十轮完整周期为：

```text
401.243  393.017  391.942  399.137  398.913 us
395.434  394.909  394.217  393.187  391.591 us
```

汇总为：

- startup 到 root FinalDrain：最小 `391.591 us`，中位 `394.563 us`，
  均值 `395.359 us`，最大 `401.243 us`；
- ACL kernel event：中位 `422.813 us`；
- builder 包络：中位 `343.760 us`；
- max single builder VF：中位 `343.505 us`。

因此历史约 `0.43 ms` 并没有遗漏一个毫秒级 FinalDrain 尾部；同口径设备内
完整周期反而约为 `0.395 ms`。Scalar DAG 当前 `2326.268 us` 比它多
`1931.705 us`，约慢 `5.90` 倍。下一阶段不再讨论口径，直接以这条 W4
基线拆解并优化 Scalar 关键路径。

## 10. 2026-08-07：S5 去除 `TensorAt` 内部重复 GM 解码

### 10.1 先修正性能裁决口径

复查 host 初始化后确认，SIMT B32/W4 将 QK/SF/PV/UP 的完整
128x128 计算流水重复次数固定为 `1,1,1,1`；Scalar 约 `0.82 ms`
和 DAG 约 `2.33 ms` 基线使用默认 `6,28,4,1`。前者不再作为后者
的绝对性能裁决线，只借鉴 SIMT 已验证的 DAG 数据流和并行构建
机制。后续统一使用 B256、`6,28,4,1`、startup 到 FinalDrain 的
10 次中位数裁决候选；先低于成熟 `cross_core` 的约 `0.82 ms`，
再向 `0.60 ms` 收敛。

### 10.2 根因与修改

`SharedPaDagSchema::TensorAt()` 已经通过
`DecodeSharedBuildDispatchTask()` 从 GM 计划项得到 `TaskKind`，却又调用
`TensorCount()` 对同一 task 完整解码第二次。这不仅发生于当前
task 的 Validate/Build，也会放大每次反向搜索 candidate writer 的 GM
访问。

本轮只做一项机械消减：

- 新增纯 `TaskKind -> tensor_count` 的 `TensorCountForKind()`；
- `TensorCount()` 仍独立解码并保留原有 fail-closed 校验；
- `TensorAt()` 直接消费本调用已解码的 `task.meta.kind`，不再读取
  第二次 GM 计划项。

本轮没有改动 per-symbol 前驱算法、writer 数量、atomic、DCCI、
TensorMap 顺序或 Execute 调度。

### 10.3 校验与性能

- `build-perf-clock cpu` 全量门槛 PASS，包括动态 DAG、乱序 Build、
  ordinary 回退、writer intent、execution payload 和 FinalDrain；
- AIC/AIV `build-perf-clock ccec` PASS，mixed ELF、ABI、强符号、无 relocation
  和 manifest 全部闭合；
- A5 B256 默认真计算 10/10 `execution/semantic/postprocess` PASS。

startup 到 FinalDrain 的十次结果为：

```text
2025.531  2052.315  2041.195  2057.606  2034.688 us
2020.123  2068.275  2048.935  2025.603  2066.209 us
```

- 最快：`2020.123 us`；
- 中位数：`2045.065 us`；
- 均值：`2044.048 us`；
- 最慢：`2068.275 us`；
- 相对 S2 的 `2326.268 us` 中位数减少 `281.203 us`，改善
  `12.09%`。

这证明重复 schema GM 解码是一项真实主开销，但当前仍比
`0.82 ms` 门槛慢约 `1.23 ms`。下一轮继续合并当前 task 的 schema
Validate 和 DAG 构建，并让后续 writer delta 直接消费一次推导结果。

## 11. 2026-08-07：S6 用紧凑 writer-intent 集合取代 candidate 全参数重建

### 11.1 理论与 SIMT 可复用部分

SIMT 已验证的有效机制不是 PA 固定图形本身，而是：每个 candidate
一次生成最小 writer intent，反向前驱搜索只比较 symbol，commit 再直接
消费已求得的 exact predecessor。

Scalar DAG 此前的 `SharedDagTaskHasSymbolWriter()` 对每个待查 symbol
都执行：

```text
TensorCount(candidate)
-> TensorAt(candidate, 0..N)
-> 再从全部 tensor 里筛 writer
```

UP 一次 Build 会对多个 input/INOUT symbol 搜索多个 candidate，因而同一
candidate 的 GM 计划项和全量 tensor schema 被重建多次。

### 11.2 通用改造

新增通用 `SharedDagWriterIntents`，固定容量保存：

- whole-object `symbol_keys[]`；
- `symbol_count`；
- `ordinary_writer`。

schema adapter 新增 `WriterIntentsAt(task_id, intents)`：

- PA adapter 可以读取 `TaskKind`，一次把当前 workload schema 翻译成
  紧凑 writer 集合；
- 公共 `SharedDagTaskHasSymbolWriter()` 不读 `TaskKind`，只检查 key、
  producer 边界和同 task 重复 symbol；
- ordinary 前驱搜索复用同一 adapter 结果；
- CPU 的任意 schema 模型也实现同一接口，继续覆盖同/异 symbol、
  重复 writer、future producer 和 ordinary 回退。

这一接口是每个算子 schema adapter 都可以实现的通用合同，公共路径
没有 PA task-id 间距、固定 batch 形状或特例分支。atomic、DCCI、
per-symbol CAS 和 TensorMap 顺序均未改动。

### 11.3 校验与性能

- `build-perf-clock cpu` 全量门槛 PASS；
- AIC/AIV CCEC perf-clock 产物和 manifest PASS；
- A5 B256、`6,28,4,1` 十轮全部正确。

startup 到 FinalDrain 的十次结果为：

```text
1131.659  1121.307  1108.034  1131.233  1127.642 us
1107.674  1111.810  1124.130  1099.495  1117.145 us
```

- 最快：`1099.495 us`；
- 中位数：`1119.226 us`；
- 均值：`1118.013 us`；
- 最慢：`1131.659 us`；
- 相对 S5 `2045.065 us` 减少 `925.839 us`，改善 `45.27%`；
- 相对初始 `2326.268 us` 累计改善 `51.89%`。

结果证明 candidate 全参数重建是当前最大的 DAG 纯 Scalar 开销。目前
距 `0.82 ms` 门槛约 `299 us`；下一轮将当前 task 的 Validate、
DAG 分类和 writer-delta 构造合并为一次 `TaskArgs` 扫描。

## 12. 2026-08-07：S7 当前 `TaskArgs` 直接构建 DAG

### 12.1 问题与合同

S6 后的生产 Build 对当前 task 仍然做两份等价工作：

1. adapter 已经根据 ticket 构造真实 `TaskArgs`；
2. 独立 Validate 又从 GM dispatch plan 重建 schema 并逐 tensor 比较；
3. DAG Build 第三次遍历 schema 并分类 reader/writer。

本轮借鉴 SIMT “实际参数一次推导、后续直接消费”的数据流，
但不复制 PA 固定图形：

- 当前 task 的 `TaskArgs` 是唯一权威 DAG 输入；
- 同一次 tensor 遍历检查 tag 范围、reference kind、空指针、
  SharedOutputRef key/producer、重复 writer 与 ordinary 回退；
- adapter schema 只为历史 candidate 提供紧凑 writer intent，不再
  重建当前 task 的第二份 schema；
- 没有改动 atomic、DCCI、per-symbol CAS、TensorMap 顺序、task
  数或 Execute 调度。

`SharedDagTaskArgsSchema` 是通用 adapter 包装，公共路径不读
`TaskKind`。每个新算子 adapter 必须保证 `WriterIntentsAt()` 与它
构造的 `TaskArgs` writer 语义一致，CPU schema 模型继续覆盖
同/异 symbol、future producer、重复 writer 和 ordinary 回退。

### 12.2 校验与性能

- `build-perf-clock cpu` 全量门槛 PASS；
- AIC/AIV CCEC 编译、mixed ELF、ABI、强符号、无 relocation 和
  manifest PASS；
- A5 B256、`6,28,4,1` 十轮全部
  `execution/semantic/postprocess` PASS。

startup 到 FinalDrain 的十次结果为：

```text
1119.658  1134.944  1113.214  1094.545  1100.290 us
1099.653  1093.872  1104.449  1116.927  1130.450 us
```

- 最快：`1093.872 us`；
- 中位数：`1108.832 us`；
- 均值：`1110.800 us`；
- 最慢：`1134.944 us`；
- 相对 S6 `1119.226 us` 减少 `10.394 us`，改善 `0.93%`；
- 相对初始 `2326.268 us` 累计改善 `52.33%`。

收益稳定但不大，说明只消除当前 task 的重复 schema 重建不足以
突破 `0.82 ms`。下一轮将同一当前 task 的全部 symbol 合并为
一次反向 candidate 扫描，避免对同一 candidate 重复解码。

## 13. 2026-08-07：失败候选——合并多 symbol 反向扫描

S7 后曾尝试把当前 task 的全部 symbol 先收集起来，再从
`task_id - 1` 向前扫描一次 candidate，一次 `WriterIntentsAt()`
同时解析多个 symbol 和 ordinary predecessor。CPU 证据确认同一
candidate 从旧算法的多次解码降为一次，所有正确性门槛
和 CCEC 构建也全部 PASS。

但 A5 B256、`6,28,4,1` 端到端前五次为：

```text
1644.693  1611.894  1588.758  1608.016  1602.784 us
```

随后又将新增的 32-entry 本地 symbol 数组删除，改为原地
复用 `SharedMetadataDag` 已有数组，以排除 Scalar 栈/寄存器压力。
第二版五次仍为：

```text
1602.470  1627.251  1607.983  1579.952  1632.277 us
```

因此回退不是大数组导致，而是“多 symbol 批量解析”增加的
通用控制流、循环和代码形态在 A5 Scalar 上明显重于它减少的
GM 解码。两版失败实现均已完整撤回，未提交。后续不再
扩展热路反向扫描，改为验证“前驱推导一次后持久化，Build 直接
消费紧凑结果”的 SIMT 机制。

## 14. 2026-08-07：S8 历史 writer intent 只解码必要投影

### 14.1 查证与修改

失败的批量扫描说明，A5 Scalar 上不能为了减少读取而引入
更重的通用控制流。继续检查单次 `WriterIntentsAt()` 后发现：
历史 candidate 只需要 task-local metadata writer schema，但 PA adapter
每次都调用 `DecodeSharedBuildDispatchTask()`，额外解码和校验：

- Execute route；
- executable 标志；
- AIC/AIV engine class；
- 与 writer intent 无关的完整 Build task 结构。

本轮不改反向搜索算法和调用次数，只改 PA schema adapter 的
投影边界：

```text
WriterIntentsAt(candidate)
  -> 检查 dispatch task/batch 边界
  -> 读 candidate 的 4B immutable identity
  -> 只解码 encoded PA metadata
  -> 校验 batch/last-task 合同
  -> 输出紧凑 writer symbol 集
```

Execute route 由 launch 前 host plan 和每个当前 task 的完整 dispatch
解码独立校验，不再在每次历史 writer 查询时重复执行。
这是“adapter 只解码调用者需要的 schema 投影”的通用原则；
公共 DAG 层没有新增 PA `TaskKind` 或固定图形分支。

### 14.2 校验与性能

- CPU perf-clock 全量门槛 PASS；
- AIC/AIV CCEC、mixed ELF、ABI、强符号、无 relocation 和 manifest
  PASS；
- A5 B256、`6,28,4,1` 十轮全部
  `execution/semantic/postprocess` PASS。

startup 到 FinalDrain 的十次结果为：

```text
1020.537  1003.856  1010.410  1015.094  1009.150 us
1013.102   999.118   991.684   992.143  1003.892 us
```

- 最快：`991.684 us`；
- 中位数：`1006.521 us`；
- 均值：`1005.899 us`；
- 最慢：`1020.537 us`；
- 相对 S7 `1108.832 us` 减少 `102.311 us`，改善 `9.23%`；
- 相对初始 `2326.268 us` 累计改善 `56.73%`。

目前距成熟 Scalar `0.82 ms` 门槛仍有约 `187 us`，距 `0.60 ms`
目标约 `407 us`。下一步继续减少历史 schema 的重复 GM 读取，但
不再扩大 DAG 热路控制流。

## 15. 2026-08-07：S9 无 writer candidate 最小快速拒绝

### 15.1 原理与边界

S8 已把历史查询从完整 Build dispatch 缩小到 PA metadata，但
仍然对每个 candidate 完整调用 `DecodeSharedPaTaskMeta()` 后才
判断 writer 集是否为空。writer intent adapter 的输出只有两类：

- 明确无 writer：返回空集；
- 可能有 writer：继续完整解码并构造 symbol key。

因此本轮先从 immutable identity 的 `encoded_meta` 投影
`present + kind`，同时检查 task/batch 边界；若 adapter 可当场证明
该 kind 无 metadata writer，直接返回空集。只有可能写 metadata
的 task 才进入完整 PA metadata 解码。

这个快速拒绝位于 PA schema adapter，公共 DAG 仍只消费通用
`WriterIntentsAt()`；新算子可以在自己的 adapter 中使用同一
“先证明空集，再解码完整 schema”原则。host plan 发布前仍校验
全部 identity，每个 task 自身取得 Build ticket 时仍执行完整
device dispatch 解码；没有放松副作用发布边界。

### 15.2 校验与性能

- CPU perf-clock 全量门槛 PASS；
- AIC/AIV CCEC、mixed ELF、ABI、强符号、无 relocation 和 manifest
  PASS；
- A5 B256、`6,28,4,1` 十轮全部
  `execution/semantic/postprocess` PASS。

startup 到 FinalDrain 的十次结果为：

```text
953.852  949.250  943.093  932.885  933.149 us
943.707  933.897  957.536  945.365  938.186 us
```

- 最快：`932.885 us`；
- 中位数：`943.400 us`；
- 均值：`943.092 us`；
- 最慢：`957.536 us`；
- 相对 S8 `1006.521 us` 减少 `63.121 us`，改善 `6.27%`；
- 相对初始 `2326.268 us` 累计改善 `59.45%`。

目前距 `0.82 ms` 门槛约 `123 us`，距 `0.60 ms` 目标约 `343 us`。

### 15.3 无明确收益的 immutable header 快照

后续候选将 `task_count/batch_count` 在构造 schema 时读入 8B 本地
快照，避免每个 candidate 重读 dispatch header。CPU/CCEC 门槛和
A5 正确性均 PASS，但十轮中位数仅从 `943.400 us` 变为
`942.356 us`，差值 `1.044 us / 0.11%` 小于波动，不足以证明收益。
该候选已完整撤回，不增加 schema 本地状态。

## 16. 2026-08-07：S10 DAG 推导结果直接生成 writer delta

### 16.1 重复工作

S9 后当前 task 的 DAG 已经从真实 `TaskArgs` 得到：

- writer symbol key；
- ordinary writer 分类；
- exact predecessor 与 fanin；
- 同 task 内重复 symbol 和引用合法性结论。

但 Materialize 后仍调用 `PrepareSharedTaskWriterDelta()` 第二次遍历
整份 1,280B `TaskArgs`，重新计算 symbol key、重复检查和 writer
分类，再逐 key 与 DAG 比较。这些工作对 symbol-only 路径没有新的
正确性证明。

### 16.2 通用 derive-once 改造

`SharedMetadataDag` 新增 `writer_tensor_mask`，首次 DAG 遍历同时记录
所有 `Inout/OutputExisting` tensor 位。Materialize 后的新流程为：

```text
actual register_mask == DAG writer_tensor_mask
-> symbol keys 直接从 DAG 复制到 writer delta
-> ordinary_writer == false: 立即完成
-> ordinary_writer == true:
     进入 noinline 保守回退
     只为真实 ordinary writer 读 TensorDesc/构造 region
```

旧 `PrepareSharedTaskWriterDelta()` 和它的独立通用门槛保留；生产
DAG 路径使用 `FinalizeSharedTaskWriterDeltaFromDag()`。ordinary 的
GM/local address-space 由模板引用保留，不把 `__gm__ TensorDesc`
强制转成 local 引用。

首版把 ordinary 循环内联进热路后，前五次中位数约 `949 us`，
未取得收益。将它隔离为 noinline 回退后，symbol-only 主路不再
承担该代码形态，才获得下述稳定改善。

### 16.3 门槛与性能

新增/加强的 CPU 证据包括：

- 动态 DAG 保留 writer tensor mask；
- PA 随机访问构参的 symbol-only delta 与 DAG 一致；
- ordinary + symbol 混合回退仍构造正确 bucket/region；
- Materialize register mask 与 DAG 不一致时 fail closed。

CPU perf-clock 全量门槛、AIC/AIV CCEC、mixed ELF、ABI、强符号、
无 relocation 和 manifest 全部 PASS。A5 B256、`6,28,4,1` 十轮
`execution/semantic/postprocess` 全部 PASS：

```text
931.088  922.405  923.536  942.034  929.284 us
978.265  935.013  933.532  920.605  944.017 us
```

- 最快：`920.605 us`；
- 中位数：`932.310 us`；
- 均值：`935.978 us`；
- 最慢：`978.265 us`；
- 相对 S9 `943.400 us` 减少 `11.090 us`，改善 `1.18%`；
- 相对初始 `2326.268 us` 累计改善 `59.92%`。

目前距 `0.82 ms` 门槛约 `112 us`，距 `0.60 ms` 目标约 `332 us`。

## 17. 2026-08-07：S11 固定容量对象只初始化有效前缀

### 17.1 泳道证据与问题定位

S10 的 B256、`6,28,4,1` 完整泳道为：

```text
outputs/pa_scheduler_cross_core_dag_swimlane_20260807_112721_3417151/ccec/
```

该次设备端 startup 到 FinalDrain 为 `1038.075 us`。按 task kind 统计
Claim 结束到 Materialize 开始之间的未标记区间，Alloc/SF/UP 平均约
`2.4/2.7/4.0 us`，而 QK/PV 分别达到约 `17.0/16.9 us`。同负载成熟
`cross_core` 的五类 task 均约 `1.7--2.1 us`，说明 DAG 版本在参数构造与
局部临时对象处理上还有明显额外 Scalar 工作，不能只归咎于 atomic。

代码核查发现，每次 DAG 推导都会对若干固定 32 项的本地对象执行 `{}`
值初始化：

- 历史 writer intent 查询；
- 单项 tensor 投影；
- 当前 task 的 metadata DAG；
- Materialize 后的 writer delta。

这些生产函数本来就先写 `count`/布尔边界，并完整写出随后会读取的
`[0, count)` 前缀。清零未使用容量既不提供新的正确性证明，也会放大
QK/PV 热路的指令和栈访问。

### 17.2 改造与合同

本轮取消上述四类本地对象的整对象清零，并把合同收紧为：

1. 生产者先写全部标量边界；
2. 生产者只发布 `[0, count)` 有效前缀；
3. 消费者先校验 `count`，只读取有效前缀；
4. 未使用容量没有可观察语义，任何路径不得依赖其为零。

没有减少 task、tensor、DAG candidate、atomic、DCCI 或 kernel workload，
也没有识别 PA 固定拓扑。该原则可由其他算子的 adapter 复用。

### 17.3 门槛与性能

- `git diff --check` PASS；
- CPU perf-clock 全量门槛 PASS；
- AIC/AIV CCEC、mixed ELF、ABI、强符号、无 relocation 和 manifest PASS；
- A5 B256、`6,28,4,1` 十轮全部
  `execution/semantic/postprocess` PASS。

startup 到 FinalDrain 的十次结果为：

```text
918.006  914.945  904.198  914.929  906.275 us
929.908  904.425  925.128  911.470  957.381 us
```

- 最快：`904.198 us`；
- 中位数：`914.937 us`；
- 最慢：`957.381 us`；
- 相对 S10 `932.310 us` 减少 `17.373 us`，改善 `1.86%`；
- 相对初始 `2326.268 us` 累计改善 `60.67%`。

目前距 `0.82 ms` 门槛约 `95 us`，距 `0.60 ms` 目标约 `315 us`。

## 18. 2026-08-07：S12 schema 直接求解 ordinary writer 前驱

### 18.1 根因证据

S11 的旧 swimlane ELF 中，QK/PV 含普通 GM/local 输入，公共 DAG 为证明
此前没有 ordinary writer，会从 `task_id - 1` 扫描到 0。按 task-id 四等分后，
两类 task 的 Claim→Materialize 平均耗时随历史长度近似线性增长：

| task | 前 1/4 | 第 2/4 | 第 3/4 | 后 1/4 |
| ---- | ------: | -----: | -----: | -----: |
| QK | `6.46 us` | `12.72 us` | `20.00 us` | `26.01 us` |
| PV | `6.97 us` | `12.86 us` | `19.64 us` | `26.29 us` |

PA 本轮 1,280 个 task 没有任何 ordinary writer，因此这段二次复杂度工作只是在
重复证明空集。它不是 atomic、DCCI 或真实 kernel 开销。

### 18.2 通用 schema 合同

借鉴 SIMT “前驱由访问 schema 一次求解、Build 直接消费”的原则，公共 DAG
新增 `PreviousOrdinaryWriter(task_id, previous)` adapter 合同：

- 任意算子 adapter 都必须从自己的只读 tensor access schema 求最近 ordinary
  writer；能紧凑求解的无需逐 task 枚举，不能紧凑求解的可在 adapter 内回退扫描；
- 公共层继续校验 `-1 <= previous < task_id`；
- 非负 candidate 必须再经 `WriterIntentsAt()` 复核确实为 ordinary writer；
- 当前 task 是否读取/写入 ordinary region 仍从真实 `TaskArgs` 动态判定；
- 不新增 host DAG、writer bitset 或 PA 固定 task 间距。

PA adapter 的 schema 本身已经证明所有 metadata writer 都是
`SharedOutputRef` symbol，因此直接返回 `-1`。CPU 任意 schema adapter 仍动态
扫描自己的 tensor 定义，并覆盖 writer 与 reader 混排。新增负例还验证了
返回当前 task、symbol-only task 或小于 `-1` 时必须 fail closed。

### 18.3 门槛、泳道与性能

- `git diff --check` PASS；
- CPU perf-clock 全量门槛 PASS；
- AIC/AIV CCEC、mixed ELF、ABI、强符号、无 relocation 和 manifest PASS；
- A5 B256、`6,28,4,1` 十轮全部
  `execution/semantic/postprocess` PASS。

startup 到 FinalDrain 的十次结果为：

```text
808.007  831.690  817.445  815.316  819.482 us
818.082  806.511  821.768  819.610  820.659 us
```

- 最快：`806.511 us`；
- 中位数：`818.782 us`；
- 最慢：`831.690 us`；
- 相对 S11 `914.937 us` 减少 `96.155 us`，改善 `10.51%`；
- 相对初始 `2326.268 us` 累计改善 `64.80%`；
- 首次越过 `0.82 ms` 门槛，距 `0.60 ms` 仍约 `219 us`。

重新编译后的完整泳道位于：

```text
outputs/pa_scheduler_cross_core_dag_swimlane_20260807_115606_3447126/ccec/
```

其 global Submit makespan 为 `758.765 us`。Claim→Materialize 聚合 core-time
从旧 ELF 的 `10.689 ms` 降到 `3.672 ms`；QK/PV 平均分别从
`16.30/16.44 us` 降到 `2.45/2.82 us`，且不再随 task-id 线性增长。

## 19. 2026-08-07：S13 shared heap 直接消费 RMW 旧值

### 19.1 先排除 Execute 推进次数不足

S12 后先验证了一个不改协议的调度候选：若 Build 边界上的
`ProgressCrossCoreExec()` 完成过 task，就立即重复调用，直到本次不再完成。
A5 B256、`6,28,4,1` 十轮为：

```text
828.861  811.766  829.337  817.519  824.746 us
825.684  821.341  829.017  815.092  808.333 us
```

中位数 `823.044 us`，比 S12 的 `818.782 us` 回退 `0.52%`。代码核查同时
确认，一次 `ProgressCrossCoreExec()` 已经会消费本核所有当前 ready token；
外层重复只增加一次无效 Execute 扫描。该候选已完整回退，不把调度频率误判为
当前主要矛盾。

### 19.2 与 SIMT heap 合同的差异

Scalar 的非空 output reservation 原来依次执行：

```text
Load(aggregate vend)
Load(shard cursor)
FetchAdd(shard cursor)
FetchAdd(aggregate vend)
```

前两次 Load 的值随时可能被并发 builder 改写，只能提前发现部分明显异常，
不能证明随后 reservation 成功；真正确定唯一物理区间和累计进度的是两次
`FetchAdd` 返回的旧值。SIMT builder 已采用后者的直接 RMW 合同。

本轮令 Scalar 非空 task 也直接执行两次 `FetchAdd`，并继续完整消费返回旧值：

- shard old cursor 决定 task 的唯一连续物理区间；
- aggregate old vend 决定本次发布的完成进度；
- 静态非法 task、heap、alignment 和单次 reserve 大小仍在 RMW 前拒绝；
- RMW 后发现越界属于 terminal failure，保留已线性化控制字供诊断，绝不
  回滚覆盖其他 builder 的进度；
- 零输出 task 仍只 Load aggregate vend，不推进任何分配控制字。

PA B256 中共有 `1024` 个非空 output task，因此热路径减少 `2048` 次返回型
atomic Load；task 数、输出字节、8 路 shard、DCCI、TensorMap、DAG、payload
及 kernel workload 均未改变。这是 allocator 通用优化，不依赖 PA 固定 DAG。

### 19.3 正确性门槛与性能

- 原子泳道门槛更新为：非空 reservation 仅记录
  `SharedHeapCursorReserve`、`SharedHeapVendAdvance` 两次 return-ready RMW；
- 单测覆盖静态失败零写入、合法并发唯一无缝区间、cursor/vend 交错，以及
  terminal 容量竞争保留 overrun 现场；
- CPU perf-clock 全量门槛 PASS；
- A5 B256、`6,28,4,1` 十轮全部
  `execution/semantic/postprocess` PASS。

startup 到 FinalDrain 的十次结果为：

```text
786.785  793.558  793.245  799.376  809.046 us
792.927  789.551  792.300  787.899  806.446 us
```

- 最快：`786.785 us`；
- 中位数：`793.086 us`；
- 均值：`795.113 us`；
- 最慢：`809.046 us`；
- 相对 S12 `818.782 us` 减少 `25.696 us`，改善 `3.14%`；
- 相对初始 `2326.268 us` 累计改善 `65.91%`；
- 距 `0.60 ms` 目标仍约 `193 us`。

## 20. 2026-08-07：S14 execution payload 引用不可变共享 descriptor

### 20.1 从 SIMT 最终对象构造合同提炼改造点

SIMT 路径的重要差异不是简单把 Scalar Build 搬给 AIV，而是 builder 直接形成
最终可消费对象，避免同一 descriptor 在多个中间表示之间反复搬运。S13 的
Scalar Build 已经把 fresh output 直接构造到 task-indexed `shared_outputs`，但
随后又把每个 128B `TensorDesc` 从该共享位置读回，并完整复制到 execution
payload；Execute 再从 payload 绑定一次。plain `SharedOutputRef` 也有同样重复。

本轮复用 portable execution payload 已有的 `tensor_reference_mask`：

- fresh output descriptor 与 plain `SharedOutputRef` 指向的 descriptor 在发布后
  本轮不再修改，payload 只保存 64-bit GM 地址；
- external tensor、local/view 以及无法证明不可变生命周期的 descriptor 继续
  128B 内联；
- builder 只 clean-out compact payload；executor 取得 `CLAIMED` 后先 invalidate
  payload，再逐个 invalidate 引用目标的两条 cache line；
- reference mask 必须是 GM tensor mask 的子集且不得越过 `tensor_count`；
- 公共协议只消费 adapter 给出的不可变性判定，不识别 PA 固定 task id、fanin
  或 DAG 形态。

四种 PA execution payload 的物理形状由公共 layout 精确变为：

| task | reference mask | 旧 payload | 新 payload |
| ---- | -------------: | ----------: | ----------: |
| QK | `0x08` | `592 B / 10 lines` | `472 B / 8 lines` |
| SF | `0x0f` | `604 B / 10 lines` | `124 B / 2 lines` |
| PV | `0x09` | `596 B / 10 lines` | `356 B / 6 lines` |
| UP | `0x3f` | `988 B / 16 lines` | `268 B / 5 lines` |

### 20.2 独立 oracle 暴露并修正旧假设

第一轮 A5 执行的 1,024 个 kernel、FinalDrain 和 real-compute 输出均闭合，但
host payload oracle 正确拒绝了结果。原因不是放宽后即可忽略的“验证器问题”，
而是旧 oracle 固定假设 `tensor_reference_mask == 0`，并用
`tensor_index * 16 words` 解码所有 inline descriptor。

修正后的 host oracle 不解引用 D2H 快照中的 device GM 地址，而是：

1. 按 task 访问语义独立得到期望 reference mask；
2. 用 `runtime SchedulerState` 的 device 基址加 host 对象 offset，精确核对 payload
   中的 GM 地址；
3. 在 host 快照的同一 offset 比较完整 `TensorDesc`；
4. 对 inline tensor 继续按 compact layout 的实际 word offset 解码；
5. 同时检查 header、payload lines、scalar、fanin、route 和 terminal state。

CPU adapter 门槛还覆盖了四种实际 shape、引用地址、builder 源污染后的 payload
不变性，以及“portable shape 合法但不符合 PA function shape”时绑定必须拒绝。

### 20.3 正确性与端到端性能

- `git diff --check` PASS；
- CPU perf-clock 全量门槛 PASS；
- AIC/AIV CCEC、mixed ELF、ABI、强符号、无 relocation 和 manifest PASS；
- A5 B256、默认 `6,28,4,1` 十轮全部
  `execution/semantic/postprocess` PASS。

startup 到 FinalDrain 的十次结果为：

```text
785.527  795.271  789.765  780.751  787.438 us
788.099  775.384  799.326  790.455  791.748 us
```

- 最快：`775.384 us`；
- 中位数：`788.932 us`；
- 最慢：`799.326 us`；
- 相对 S13 `793.086 us` 减少 `4.154 us`，改善 `0.52%`；
- 相对初始 `2326.268 us` 累计改善 `66.09%`；
- 距 `0.60 ms` 目标仍约 `189 us`。

### 20.4 泳道解释

重新编译后的完整泳道位于：

```text
outputs/pa_scheduler_cross_core_dag_swimlane_20260807_132840_3887751/ccec/
```

该次观察构建的 startup 到 FinalDrain 为 `860.809 us`。与 S13 的观察构建比较，
聚合 `WinnerBuild` core-time 从 `9.238 ms` 降到 `5.735 ms`，减少约
`3.503 ms / 37.9%`；`Submit union` 从 `62.842 ms` 降到 `54.497 ms`。
这证明 builder 侧确实删除了重复 descriptor copy 和 payload clean-out，而不是
只靠端到端噪声得出结论。

端到端只改善 `0.52%` 的原因也很明确：executor 仍必须 invalidate 并读取相同
descriptor，且 AIC kernel compute 已接近关键路径下限；缩短的是 96 核聚合
Build 工作，不会等比例缩短最慢 AIC 的串行关键链。该结果仍保留，因为合同
通用、删除了真实重复搬运且所有 oracle 闭合，但它不能单独通向 `0.60 ms`。

## 21. 2026-08-07：S15 去除 dispatched metadata 的重复 fatal 读取

### 21.1 先排除三个不成立的 SIMT 候选

S14 之后先从 SIMT 的「分散控制字」和「局部完成」中提炼了三个
候选，但都不符合 Scalar 当前的动态均衡或可接管合同：

1. **16 路动态争抢、固定 task shard 的 Build cursor**：CPU 正确性通过，
   但 A5 十轮中位数 `857.808 us`，相对 S14 回退 `8.73%`。它虽然
   降低了单地址 atomic 竞争，却把任务尾部和快慢核差异固化在 shard
   内，破坏了中央 ticket 的全局动态均衡。
2. **Build `FetchAdd(2)` 并把第二个 task 私藏在 owner 栈上**：目标是把
   Build ticket 物理调用从 `1376` 降至 `736`，但 CPU
   `release_before_build` 只完成 `17` 个 task。owner 停在第一个 Build 时，
   第二个 task 也被私有化，其他 Scalar 无法接管；因此未上板。
3. **16 组 FinalDrain 最后到达者动态收口**：CPU 全套通过，A5 十轮
   中位数 `794.767 us`，回退 `0.74%`。原固定收口核的成功路径只有
   `70` 次返回型 load、观察构建聚合约 `29.951 us`；新增的 16 次
   RMW、分支和指令工作集得不偿失。

三个候选均已完整撤回，没有进入 S15 代码。

### 21.2 重复读取的合同依据

shared Build 主循环已有两层 terminal-fatal 合同：

- worker 进入 scheduler 时立即观察一次；
- 此后每个 worker 每完成 `16` 个 Build，在领取下一张 ticket 前再观察
  一次，因而远端错误后最多额外发放 `15` 个本核 task。

同一代码的合同也明确：已取得的合法工作单允许闭合，不应在
Materialize/Register 中又对全局 fatal cache line 逐 task 做返回型 Load。
但正式 `FinishSharedWinnerSubmitBody()` 仍以 `CheckFatal=true` 实例化
`PublishSharedTaskWriterMetadata()`，B256 因此产生 `1280` 次
`SharedMetadataFatalGuardLoad`；S14 泳道聚合约 `0.647 ms` core-time。

S15 只对正式 dispatched 调用点传入 `CheckFatal=false`：

- 通用组合入口和隔离测试仍使用 helper 默认的逐次检查；
- metadata DAG 前驱、writer CAS、DCCI、insert completion 和 FinalDrain 均不变；
- 不读取 task kind、batch、fanin 或 PA 固定 DAG，对任意算子的中央
  Build 发放合同都成立。

### 21.3 正确性与性能

- `git diff --check` PASS；
- CPU 全量门槛 PASS，包括 `remote_fatal_cadence_closure`、乱序 Build、
  严格 metadata writer 前驱、Execute drain 与独立 kernel overlap；
- AIC/AIV CCEC、mixed ELF、ABI、强符号、无 relocation 和 manifest PASS；
- A5 B256、`6,28,4,1` 十轮全部
  `execution/semantic/postprocess` PASS。

startup 到 FinalDrain 的十次结果为：

```text
782.196  784.846  785.561  789.696  793.672 us
779.835  793.219  781.152  787.521  783.710 us
```

- 最快：`779.835 us`；
- 中位数：`785.204 us`；
- 最慢：`793.672 us`；
- 相对 S14 `788.932 us` 减少 `3.729 us`，改善 `0.47%`；
- AIV perf-clock 对象的 text 从 S14 约 `85,448 B` 降至
  `84,424 B`，说明编译期确实删除了重复的返回型原子路径；
- 距 `0.60 ms` 目标仍约 `185 us`。

这是一个通用但幅度很小的热路减法。它不能支撑「继续微调单次
atomic 即可到达 0.60 ms」的结论，后续仍需降低 Build/Execute 的通用
非 kernel 工作量或关键路径指令工作集。
