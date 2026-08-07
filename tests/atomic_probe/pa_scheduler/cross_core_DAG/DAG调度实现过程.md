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
