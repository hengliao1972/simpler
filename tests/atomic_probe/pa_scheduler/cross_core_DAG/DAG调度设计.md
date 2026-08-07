# Scalar Cross-Core DAG 调度设计

本文定义 `cross_core_DAG` standalone 的架构、内存合同、实施边界和验收
口径。实现从现有 Scalar [`cross_core`](../cross_core/) 的成熟代码改造，
但复制到本目录后独立演进，不产生对 `cross_core/` 或
[`simt_cross_core`](../simt_cross_core/) 的源码依赖。

实际开发命令、阶段结果、失败候选和提交信息记录在
[`DAG调度实现过程.md`](DAG调度实现过程.md)。

## 1. 目标与边界

目标是在同一个 PA workload 上比较两种 Build 执行域：

- 本实现由 96 个 Main Scalar 动态领取 Build；
- SIMT 方案由专职 AIV SIMT warp leader 构建；
- 两者都在 Build 时从 tensor access schema 动态推导 per-symbol DAG；
- 两者保持相同 task 数、task id、fanin、payload、kernel、completion 和
  FinalDrain 语义；
- 最终性能只以最早 startup 到最后 FinalDrain 的端到端时间裁决。

本目录只实现 CPU 语义模型与 CCEC/A5 路径，不新增 AscendC 版本。第一版
task-indexed cell 单轮不复用，不引入 generation、回收或容量反压。

以下内容明确不做：

- 不让 host 或 AICPU参与运行期逐 task 调度；
- 不由 host 预计算逐 task DAG 或 metadata writer bitset；
- 不在公共调度代码中识别 PA `TaskKind`、固定五 task 间距或固定三个
  accumulator；
- 不用物理到达顺序代替逻辑 task-id 顺序；
- 不为了性能省略 ordinary payload 的 DCCI 发布/取得合同；
- 不强行复用旧目录的热代码，也不在同一代码块堆叠模式宏。

## 2. 已冻结的总体架构

```text
host
  -> 发布 launch 级只读配置、workspace 和初始状态
  -> 不发布逐 task DAG

96 Main Scalar
  -> 发布 startup arrival
  -> 从中央 Build cursor 每次领取一个 task id
  -> adapter 根据 task id 构造 args 和 tensor access schema
  -> 动态推导 writer intent 与同 symbol 最近前驱
  -> Materialize output / 注册 per-symbol metadata
  -> 构造 task-indexed execution payload
  -> DCCI 发布 BUILT

32 AIC Scalar / 64 AIV Scalar
  -> 分别从 function-striped Execute cursor 领取兼容 task
  -> 绑定最多四个本核 token
  -> BUILT 后 Claim、invalidate payload、检查 fanin
  -> ready token 优先执行对应 Cube / Vector kernel
  -> engine 完成后发布 vend、completion 和 DONE

FinalDrain
  -> Build 与两条 Execute cursor 全部耗尽
  -> 全部 execution cell 闭合
  -> 所有 token IDLE、无 engine in-flight
  -> 96 个 startup arrival 与分组到达计数完整
```

Build owner 与 Execute owner 完全解耦，但不强制物理核不同。任何 Scalar
都能 Build 任意 task；只有 Execute 受 engine class 约束。

## 3. Build 时动态推导 per-symbol DAG

### 3.1 Schema adapter 与通用调度层

PA adapter 可以根据 `TaskKind` 构造参数并给出每个 tensor 的：

- `TensorAccess`；
- `SharedOutputRef(producer_task, output_slot)`；
- ordinary region 描述；
- engine class、function id 和 payload shape。

adapter 还必须能对任意 task 一次导出最小 writer-intent 集合：

- whole-object symbol key 列表；
- 是否存在 ordinary writer。

这是 schema adapter 的通用能力，不是 PA 快路：新算子提供自己的
schema 翻译。公共反向搜索每个 candidate 只取这份 writer 集合，
不得为待查的每个 symbol 反复重建 candidate 的全部 tensor 参数。

当前 task 已经由 adapter 构造成真实 `TaskArgs`，公共 DAG 层将
它作为当前 task 的唯一权威 schema：在一次 tensor 遍历中校验
tag、reference kind、指针、symbol producer 和重复 writer，不再从 GM
dispatch plan 重建第二份当前 task schema。`WriterIntentsAt()` 则是
adapter 对历史 candidate 的稳定合同，必须与该 adapter 生成的
`TaskArgs` writer 语义一致；CPU adapter 门槛必须覆盖这一点。

从 writer-intent 生成开始，公共 DAG 代码只消费上述通用信息。调度层不得
读取 `TaskKind` 或 PA 固定图形。

### 3.2 Writer intent

只有满足以下条件的 tensor 才形成 whole-object writer intent：

```text
access in {Inout, OutputExisting}
&& reference is SharedOutputRef
```

symbol key 固定由 `(producer_task, output_slot)` 构造。一个 task 内不能提交
重复 symbol；writer 数超过固定 history 容量时显式 fatal。

### 3.3 精确逻辑前驱

对 task `N` 的每个 writer symbol，Build owner 执行与 SIMT 版本等价的动态
反向扫描：

```text
candidate = N - 1
while candidate > producer_task:
    从 candidate 的只读 schema 动态重建 writer intent
    若存在相同 symbol:
        previous_writer = candidate
        break
    candidate--

若不存在更晚 writer:
    previous_writer = producer_task
```

扫描只读取只读 workload schema，不要求 candidate 已被 Build，也不读取其
未发布 payload。不能用当前 `last_writer` 的物理值猜逻辑前驱。

### 3.4 History 与 commit

Build owner把每个 `(symbol_key, previous_writer)` 一次性写入当前 task 的
immutable writer history。随后按以下顺序 commit：

```text
等待每个目标 SharedOutput descriptor 已发布
-> 对去重后的 previous_writer：
     若 previous_writer != producer_task，等待其 insert_completion
-> history ordinary store 全部完成
-> 精确 DCCI clean-out + DSB
-> 对每个 symbol：last_writer CAS(previous_writer -> N)
-> 所有 symbol CAS 成功
-> 发布本 task 唯一 insert_completion
```

同一 symbol 的 writer 因此严格按 task id 串行；不同 symbol 不产生全局假
串行。一个 task 即使写多个 symbol，也只在全部 symbol 提交后发布一个
`insert_completion`。

### 3.5 Ordinary region 回退

whole-object 快路不能替代原始 max-overlap region TensorMap。遇到 ordinary
region、view 或 alias 时，首版保留 shared TensorMap 的保守全局 writer 顺序，
保证查询看到所有可能影响结果的早期 writer。

PA 主 Case 的 ordinary writer 为零，因此 A5 性能路径只走 per-symbol 快路；
CPU 与 CCEC 门槛仍必须覆盖 ordinary 回退，未知形态不能静默当成 symbol。

## 4. Build 与 Execute 调度

### 4.1 Build ticket

全部 96 个 Main Scalar 共享一条中央 Build cursor。每次返回型 FetchAdd 只
领取一个 task id：

- 每个 task 恰好一个 Build owner；
- Build owner领取后必须完整发布该 task，处理中途不切换；
- task args 必须可按 task id 随机重建，不依赖本核先前 replay 状态；
- 第一版不批量领取 Build ticket，避免把批次效果混入 Scalar/SIMT 比较。

### 4.2 Execute 尽早开放

每个 worker 发布 startup arrival 后，不等待其他 worker 到齐即可尝试 Build
和 Execute。全员到达不是 Execute 的正确性条件，只在 FinalDrain 中证明参与者
完整。

Execute 仍使用两条 function-striped cursor：

- AIC cursor 只包含 Cube task；
- AIV cursor 只包含 Vector task；
- 一次 FetchAdd 最多领取两个连续 ordinal；
- 每核最多四个 owner-local token；
- 领取后可以处于 `WAITING_BUILT`，但不能读取半包；
- 任一 token ready 时，必须先执行，不得继续领取 Execute 或 Build；
- 新取得的 token 仍未 BUILT 时，本调度边界停止继续取票。

Execute 尽早开放必须通过乱序启动和慢 builder 测试证明不会造成 task 遗失、
token 永久占用或 FinalDrain 提前退出。

## 5. Execution cell 与内存合同

### 5.1 状态机

task-indexed `SharedExecCell[task_id]` 保持：

```text
EMPTY -> BUILDING -> BUILT -> CLAIMED -> DONE
```

- Build ticket owner取得 `EMPTY -> BUILDING`；
- payload 完整发布后才能进入 `BUILT`；
- Execute ticket 唯一发放 task，`BUILT -> CLAIMED` CAS 是状态校验；
- 只有 Execute owner能在 kernel 完成后发布 `DONE`；
- Alloc 没有 Execute ticket，由 Build owner直接闭合 completion。

control 独占 atomic-only cacheline；immutable payload 从下一条 cacheline
开始。payload 不得包含 builder 私有指针或本核上下文地址。

### 5.2 Scalar 发布与取得

Scalar 普通 GM store 与 SIMT `asc_stcg` 的具体指令不同，不能为了源码外观
一致而省略已经验证的 Scalar cache 合同。

Build owner：

```text
写 descriptor / history / payload ordinary fields
-> 对精确有效 cacheline 执行 clean-out
-> DSB
-> atomic 发布 published / BUILT
```

Execute owner：

```text
atomic 取得 CLAIMED
-> 对有效 payload cacheline 执行 invalidate
-> DSB
-> ordinary load 并绑定本核 token
```

atomic-only line 不执行 payload DCCI。泳道记录必须覆盖真实 atomic 与 DCCI
调用点，但观察字段不能写入协议 cacheline。

### 5.3 Heap 与 output

保留现有 Scalar C 已验证的 8 路 shared heap shard：

- shard 由 task id 确定；
- task 的所有 fresh output 在一个连续 reservation 中分配；
- aggregate vend 只表示全局完成进度，不是物理地址；
- fresh `TensorDesc` 直接构造在最终 task-indexed shared output cell；
- 不先写 worker descriptor 再跨核复制第二次。

该语义与 SIMT G0 一致，允许 host 使用同一套 heap interval、descriptor 和
completion-vend oracle 比较两种实现。

## 6. 容量与失败合同

首版容量与 SIMT G0 保持一致：task、output、tensor、scalar、fanin、writer
history 和 payload 都使用固定上限。任何越界、重复 symbol、非法 producer、
future producer、错误 engine 或状态跳变均发布首错 fatal。

所有等待必须同时受 poll 上限和设备时钟 deadline 约束。fatal 后停止领取新
Build/Execute ticket，但已取得的合法状态必须按可证明路径闭合或保留可定位
终态，不能无限轮询。

## 7. FinalDrain 合同

FinalDrain 至少证明：

1. 96 个 worker 全部发布 startup arrival；
2. Build cursor 耗尽且恰好构建全部 task；
3. AIC/AIV Execute cursor 各自耗尽且无重复 ordinal；
4. 所有 kernel task 恰好执行一次；
5. 全部 execution cell 为合法终态；
6. 每核四个 token 均为 `IDLE`；
7. 无 engine in-flight；
8. DONE、completion、vend 和分组 arrival 数量全部匹配计划；
9. fatal 为零，所有 guard、padding 和未使用 cell 保持初值。

## 8. 文件隔离与构建身份

```text
cross_core_DAG/
  DAG调度设计.md
  DAG调度实现过程.md
  common/             # 独立 ABI、动态 DAG、调度与 host 公共模型
  cpu/                # 同源语义模型和构建入口
  ccec/               # 独立 AIC/AIV kernel、ACL host 与构建入口
  test/               # 定向交错、压力和 oracle
  test_record/        # 仅保存用户要求留档的结果
  run.sh              # 独立 build/run/perf/swimlane 入口
```

正式只保留两个产物身份：

- `perf`：编译期不包含泳道记录；
- `swimlane`：普通阶段、atomic 和 DCCI 合并记录。

两者使用独立源入口和独立产物目录，不通过大量条件宏共享一条热代码。

## 9. 分阶段实施与验收

| 阶段 | 内容 | 通过条件 |
| ---- | ---- | -------- |
| D0 | 设计和过程文档 | 三轮需求、动态 DAG 和公平比较口径完整 |
| S0 | 独立 CPU 模型 | 乱序 Build、同/异 symbol、INOUT、ordinary 回退和早期 Execute 全部通过 |
| S1 | 独立 CCEC 产物 | AIC/AIV 编译、ELF/ABI、atomic/DCCI 和无旧目录依赖门槛通过 |
| S2 | A5 B1/B256 | task/DAG/payload/history/golden/FinalDrain 全部闭合 |
| S3 | perf/swimlane | 10 轮端到端中位数、Build 完成时间和完整泳道可解释 |

每个阶段完成后更新过程文档并独立提交。未经用户明确授权不得 push。

## 10. 性能口径与目标

唯一裁决指标为同一 PA、默认真实计算负载
`QK/SF/PV/UP=6,28,4,1`、B256、10 次运行的：

```text
最早 startup timestamp -> 最后 FinalDrain timestamp
```

A5 跨核共享的 1 ns `get_sys_cnt()` 取设备内边界，Scalar 复用
`WorkerResult::submit_begin/finish_cycle`。ACL kernel event、Build 包络和泳道时间
只用于解释，不得替代上述端到端指标。

现有 SIMT B32/W4 产物将 workload 固定为 `1,1,1,1`，而下列
Scalar 基线使用 `6,28,4,1`。因此 SIMT 的约 `0.39 ms` 只能用来
查证“per-symbol 前驱一次推导、后续直接消费”等机制，不得与
Scalar 真实负载时间直接相减。

默认真实负载的权威参考线为：

- 成熟 `cross_core`：历史 12 组中位数 `815.937 us`；
- `cross_core_DAG` 初始基线：10 次中位数 `2326.268 us`；
- 紧凑 writer-intent adapter 后的阶段值：`1119.226 us`；
- 当前 TaskArgs 直接构 DAG 后的阶段值：`1108.832 us`；
- 历史 writer intent 最小投影解码后：`1006.521 us`；
- 第一门槛：DAG 实现必须低于 `0.82 ms`；
- 最终目标：达到 `0.60 ms`。

辅助指标不作为第二套性能结论，只解释端到端差异：

- 最早 Build 开始到最后一个 task `BUILT`；
- 最后一个 `BUILT` 到 FinalDrain 的执行尾部；
- Build/Execute owner 的 task 分布；
- per-symbol predecessor、last-writer CAS、atomic 和 DCCI 数量；
- Scalar 与 SIMT 实际占用的 builder/executor 资源数。

`perf` 决定候选保留或撤销；`swimlane` 只解释时序。带泳道与无泳道产物的
绝对时间不得互相相减。

## 11. 不变量清单

1. task id、task 数、engine、function、参数 ABI 和 kernel workload 不变；
2. fanin 只能引用严格更早 producer；
3. 每个 task 恰好 Build 一次，每个 kernel task 恰好 Execute 一次；
4. descriptor/history/payload 完整后才能发布可见状态；
5. 同 symbol writer 严格按 task id，异 symbol 才允许并行；
6. ordinary region 不得误走 whole-object 快路；
7. AIC 只执行 Cube，AIV 只执行 Vector；
8. Build owner、Execute owner和 Completion owner责任明确；
9. host oracle 必须独立重建结果，不能复用设备结果充当 golden；
10. 任何性能优化不得识别 PA 固定 DAG 或减少工作语义。
