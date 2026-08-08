# SIMT 串行 TensorMap 调度设计

## 1. 目标与边界

本目录验证第五种组合：保留 `simt_cross_core` 的 SIMT Build / Scalar Execute
分工，但把 metadata 插入顺序恢复为 `cross_core` 已验证的全局稀疏 writer
顺序。为避免与现有 symbol 快路互相污染，本目录独立演进，不从
`simt_cross_core` 引用源码；只有协议形状和泳道图表达方式沿用已经验证的设计。

本阶段固定边界如下：

- 只实现 GM execution cell，不实现 UBUF；
- 真实 PA workload 固定为 `QK/SF/PV/UP=6/28/4/1`；
- builder 数仍可取 `1..32`，每个 builder 的 warp 数仍可构建时配置；
- builder AIV 只构建，executor AIC/AIV 只执行；
- task id、1280-task DAG、1024 个 kernel task、engine 类型、payload、fanin、
  completion 和 FinalDrain 均不得改变；
- 新目录当前是 PA schema 的验证实现，不宣称已经覆盖任意 ordinary
  region/view/alias TensorMap。

这里的 `ordinary` 指“不使用 per-symbol 解链、恢复普通的全局 writer 插入
顺序”，不是说当前 PA case 新增了 ordinary-region writer。当前真实 PA case
仍是 256 个 whole-object metadata writer。

## 2. 与现有两条路径的准确关系

参考 `../simt_cross_core/simt调度设计.md` 第 2.5 节，现有两条路径为：

| 路径 | Build 域 | metadata writer 顺序 | Execute 域 |
| ---- | -------- | -------------------- | ---------- |
| `cross_core` | Main Scalar 动态 Build ticket | 256 个真实 writer 的全局 task-id 稀疏链 | 兼容 Main Scalar |
| `simt_cross_core` | 专职 AIV 的 SIMT warp leader 静态分片 | 每个 symbol 的精确 previous writer；不同 symbol 可并行 | 非 builder Main Scalar |
| 本目录 | 专职 AIV 的 SIMT warp leader 静态分片 | 与 `cross_core` 相同的全局 task-id 稀疏 writer 链 | 非 builder Main Scalar |

所以本目录只替换 `simt_cross_core` 的 metadata 排序轴，不把 Scalar Build
重新搬回来，也不退回 `shared_same_core` 的 1280-task 全链。

## 3. 串行插入合同

对 task `N`，只读计划先给出通用 writer intent：

```text
publishes_metadata(N) = writer_intent_count(N) > 0
previous_metadata_writer(N) = max(W | W < N && publishes_metadata(W))
```

真实 writer 执行以下流程：

```text
SIMT leader 唯一 Claim task N
  -> 构造 descriptor / payload
  -> 等待 writer intent 引用的 producer output published
  -> 等待 insert_completion[previous_metadata_writer(N)]
  -> 发布 immutable per-symbol history
  -> 对每个 symbol 以 non-cacheable store 发布派生 last_writer
  -> fence，确保 history / last_writer 先于 completion 可见
  -> 发布 insert_completion[N]
  -> 发布 BUILT
```

非 writer 不等待也不发布 `insert_completion` baton。固定 B256 case 中：

```text
1280 tasks
256 metadata writer tasks
255 global sparse predecessor edges
1024 non-writer tasks
```

这与 `cross_core` 的核心顺序一致：`W0 -> W1 -> ... -> W255`。它与
`shared_same_core` 的 `task[0] -> ... -> task[1279]` 不同。

## 4. 为什么保留 per-symbol history，但不重复做 last-writer CAS

全局 writer 链只回答“哪个 metadata task 先提交”，不能替代 Tensor 依赖本身。
每个 writer 仍必须记录每个 symbol 的精确 logical previous writer，host/CPU oracle
也继续逐条校验 history 与最终 last-writer。这样才能同时保证：

- history 不会丢失真实的逻辑前驱；
- reader 仍可沿 history 找到正确 producer；
- 全局链只增加跨 symbol 顺序，不改变任何真实 symbol 依赖；
- 后续可以直接和 per-symbol 快路做 A/B，而不改变 DAG 或终态 oracle。

但是，取得全局 predecessor turn 后，整个 ordinary writer commit 区间在协议上
已经只有一个合法 writer。此时再对每个 symbol 做 expected-previous CAS，只是在
同一条全局顺序中重复执行一次原子串行化，不再提供额外互斥。因此最终实现为：

1. history 仍写入精确的 per-symbol previous writer；
2. last-writer 作为该串行插入的派生终态，用 `asc_stcg` non-cacheable store 发布；
3. fence 后才原子发布当前 writer 的 `insert_completion`。

这个化简只依赖“全局稀疏 writer 链已经证明 commit 区间单写”这一通用协议事实，
不依赖 PA `TaskKind` 或特定 symbol。如果将来恢复不同 symbol 的并行 commit，
单写前提就不再成立，必须恢复 expected-previous CAS 或提供等价的并发校验。

## 5. 一致性与发布边界

本目录沿用已经上板验证的 GM 合同：

- SIMT 写 descriptor、payload 和 history；
- writer 在提交前等待 producer `published`；
- `insert_completion` 只由真实 writer 发布，并由后续 writer 原子读取；
- history 和派生 last-writer 在 `insert_completion` 前由 fence 建立发布顺序；
- BUILT 在 descriptor、payload、history 和 writer side effect 完成后发布；
- Scalar executor 按原路径执行 GM invalidate、Claim、payload 读取和 completion
  发布；
- FinalDrain 必须收口全部 1280 task。

GM 读取所需的 DCCI、SIMT fence 和 Scalar invalidate 均保留。串行 writer 链
不是省略一致性操作的理由。

## 6. 泛化约束

调度协议只消费 `writer_intent_count`、producer/slot 引用和 task id，不允许在
全局前驱扫描中直接检查 PA `TaskKind`。当前 PA workload 的 intent 由单独的
schema adapter 生成；未来接入其他算子时应替换 adapter，而不是修改全局链。

对于真正的 ordinary region/view/alias：在 shared region writer plan 与
max-overlap oracle 被证明前，不能把 whole-object symbol history 宣称为等价
实现。该场景仍需 `cross_core` 的保守 metadata-prefix 规则或 private TensorMap
回退。

## 7. 目录与验收

```text
simt_cross_core_ordinary/
  common/                 独立协议与 PA 模型
  gm/common/              GM 状态、writer contract、泳道 ABI
  gm/cpu/                 optimized / sanitizer 构建
  gm/test/                CPU 并发语义与 host ABI 测试
  gm/ccec/                AIC/AIV kernel、ACL host、静态校验
  test_record/2026-8-7/   真实 A5 泳道图与结果说明
  run.sh                  仅暴露 GM 构建和运行入口
```

验收必须同时满足：CPU optimized、ASan/UBSan、TSan、CCEC AIC/AIV、bitcode、
1:2 mixed ELF、ACL host、真实 A5 B1/B256、完整 host oracle 和泳道图结构校验。
性能结论使用 trace-off，泳道图只用于解释时序。
