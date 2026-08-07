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
