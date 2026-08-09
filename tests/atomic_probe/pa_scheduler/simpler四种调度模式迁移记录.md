# Simpler 四种调度模式迁移记录

## 1. 目标与边界

本文记录以独立调度器为参考，将下列四种模式接入 Simpler 真实
`fully_distributed_within_core` 路径的过程：

1. `cross_core_ordinary`；
2. `cross_core_dag`；
3. `simt_cross_core_ordinary`；
4. `simt_cross_core_dag`。

功能迁移优先，之后再使用同一业务负载、同一起止边界与同一验收
条件对比独立调度器。不用不同端点的绝对时间直接相减。

## 2. 功能迁移状态

| 模式 | 真实动态 Submit | 独立状态/复位 | 真实 A5 PA B1/B256 | 阶段提交 |
| ---- | --------------- | ------------- | ------------------ | -------- |
| `cross_core_ordinary` | 已接通 | 已闭合 | PASS | `405c060f` 及前置提交 |
| `cross_core_dag` | 已接通 | 已闭合 | PASS | `4ecc6c2f` / `197d8003` / `2d0e0426` |
| `simt_cross_core_ordinary` | 已接通 | 已闭合 | PASS | `2843d82a` / `7cf087c4` / `b13f1b19` / `f20522e4` |
| `simt_cross_core_dag` | 已接通 | 已闭合 | PASS | `ba0abef6` / `8bd60f35` / `8d464955` |

SIMT 两种模式都使 Build owner 与 Execute owner 保持独立，但 builder
拓扑按算法特性区分：

- `simt_cross_core_ordinary` 由唯一 `block0/AIV0` Main Scalar 承载持久
  builder VF，其他 95 个 Scalar 继续 replay 和竞争执行；
- `simt_cross_core_dag` 按物理 block id 选择最多 16 个 AIV0 承载 builder
  VF；32 block 时为 16 builder + 80 replay，小于 16 block 时按实际
  block 数收缩。

这一区分只依赖 scheduler mode 与物理 block/lane，不读 PA `TaskKind`、
batch 或固定 task 图形。

## 3. 泛化正确性证据

除 PA 外，已在真实 A5 上运行：

```text
examples/a5/fully_distributed_within_core/submit_dependency_smoke/
test_submit_dependency_smoke.py
case: A5OnboardBd24ExistingInoutChain
```

该用例使用 `block_dim=24` 的 72 个 worker，验证 AIC writer 到 AIV INOUT
consumer 的真实依赖链。四种调度模式均 PASS。这项证据表明通用路径不
依赖 PA 的 `TaskKind` 、QK/SF/PV/UP 固定图形或固定 batch 大小。

## 4. 端到端性能口径修正

### 4.1 旧 `perf-clock` 不能用于四模式横比

旧边界是每核首个 Submit 起点到末个 Submit 返回，它有两个确定缺口：

- 末次 Submit 后的 FinalDrain 与其中 Kernel 没有被统计；
- SIMT 专职 builder 不 replay Submit，其计数合法地为 `0/0`，旧 host
  却要求 96 核全部为相同的正 Submit 数。

AICPU `orch_start/orch_end` 也不能替代该口径：它还包含 runtime 交接、
PMU 状态恢复和结果检查，而现有解析器又要求传统 scheduler 日志。

### 4.2 新的 cross-core 合同

`perf-clock` 仍然只复用每核固定 64B 状态，不新增逐 task 事件：

```text
global min(每核 startup increment 前起点)
    -> global max(每核 FinalDrain 完成终点)
```

- 模式 1/2：96 个 replay worker，0 builder；
- 模式 3：95 个 replay worker，唯一 `block0/AIV0` builder；
- 模式 4：按 block id 选择最多 16 个 AIV0 builder；32 block 时为
  80 replay + 16 builder；
- replay worker 必须满足 `submit_count == expected_submit_count > 0`；
- builder 必须满足 `submit_count == expected_submit_count == 0`；
- 设备 raw 使用独立 mode 值，host 使用独立 schema
  `fdwic-cross-core-e2e-clock-v1`，不会把新数据静默解释为旧 Submit 窗。

`perf-clock-kernel` 在 cross-core 中使用相同端到端窗，不再套用
`5 * batches` 的 PA 专用 Kernel 数量上下界。

### 4.3 实测门槛

| 模式 | 用例 | 结果 | worker 闭合 | 单次端到端时间 |
| ---- | ---- | ---- | ----------- | -------------: |
| `cross_core_ordinary` | A5 PA CaseB1 | PASS | 96 replay + 0 builder | 398.216 us |
| `simt_cross_core_ordinary` | A5 PA CaseB1 | PASS | 95 replay + 1 builder | 1368.150 us |

这两个 B1 数字只用来证明端点和角色合同真正在设备上生效，不用来
排名。正式横比必须改用 B256，且用独立进程重复采样。

`cross_core_ordinary` 的 A5 B1 `perf-clock-kernel` 也已 PASS：同一窗口中
统计到 4 次真实 Kernel 调用，总端到端时间为 400.537 us。该产物没有
输出 PA 专用的 `5 * batches` 调用数范围字段。

### 4.4 构建注意点

pytest 会按源码指纹重编 AICore override，但不会代替安装/运行时构建去
重编长期驻留的 `libhost_runtime.so`。修改 host 的合同或 schema 后，每个
scheduler mode 必须先显式调用存量 `RuntimeBuilder` 重建对应 artifact family，
再跑上板用例。不允许用新 AICore 与旧 host 的组合解释 raw。

## 5. 四模式 B256 首轮基线

### 5.1 测试条件与可比边界

四种模式均使用真实 A5 `Case1`、shared TensorMap、32 个物理 block，
每次由独立 pytest 进程运行。命令只选择 `perf-clock` 或
`perf-clock-kernel`，没有同时开启泳道、atomic 或 PMU。

`--use-example-exec-time` 明确只允许 A5Sim，不能用于真实 A5；因此本节
执行的是 Simpler PA 的真实 QK/SF/PV/UP Kernel，并保留数值 golden。它可用于
四种 Simpler 集成模式之间的同业务横比，但不能与 standalone 的
`6/28/4/1` 合成负载绝对时间直接相减。

本节记录的是多 builder 改造前基线，四种模式均满足：

- `Case1` 数值 golden PASS；
- 1280 个动态 task 闭合；
- replay worker 每核恰好 1280 次 Submit；
- `perf-clock-kernel` 恰好统计到 1024 次 Kernel，其中 AIC/AIV 各 512 次；
- 模式 1/2 为 96 replay + 0 builder，模式 3/4 为 95 replay + 1 builder。

### 5.2 无 Kernel 聚合的低扰动首样本

| 模式 | startup 到 FinalDrain | 角色闭合 | 产物目录 |
| ---- | --------------------: | -------- | -------- |
| `cross_core_ordinary` | 71.365 ms | 96 replay + 0 builder | `TestPagedAttentionUnroll_Case1_20260809_054624` |
| `cross_core_dag` | 70.748 ms | 96 replay + 0 builder | `TestPagedAttentionUnroll_Case1_20260809_054849` |
| `simt_cross_core_ordinary` | 144.191 ms | 95 replay + 1 builder | `TestPagedAttentionUnroll_Case1_20260809_055000` |
| `simt_cross_core_dag` | 1321.880 ms | 95 replay + 1 builder | `TestPagedAttentionUnroll_Case1_20260809_055116` |

这是每种模式的首个正确性闭合样本，只用于识别数量级，不把单样本当稳定
中位数。尤其 `simt_cross_core_dag` 已比其他模式慢一个数量级，继续盲目跑十轮
不会增加定位价值，必须先修复协议热点再重复采样。

### 5.3 Kernel 聚合诊断

`perf-clock-kernel` 是另一种 ELF，不能与上一表逐项相减。下表只在同一诊断
构建族内比较；Kernel 时间与 residual 都是 96 核各自累计后的 core-time，
不是端到端墙钟。

| 模式 | 端到端 | Kernel 调用 | Kernel core-time | 非 Kernel core-time | Kernel 占比 |
| ---- | -----: | ----------: | ---------------: | ------------------: | ----------: |
| `cross_core_ordinary` | 71.442 ms | 1024 | 50.218 ms | 6804.800 ms | 0.733% |
| `cross_core_dag` | 70.907 ms | 1024 | 48.709 ms | 6755.869 ms | 0.716% |
| `simt_cross_core_ordinary` | 110.152 ms | 1024 | 51.829 ms | 10520.190 ms | 0.490% |
| `simt_cross_core_dag` | 1327.920 ms | 1024 | 51.613 ms | 127426.108 ms | 0.040% |

四种模式的 Kernel 调用数完全相同，Kernel core-time 也都在 48.7～51.8 ms；
SIMT DAG 的数量级回退几乎全部位于调度残差，不能归因于多执行 Kernel 或
Kernel workload 改变。

### 5.4 当前可证实的性能根因

第一处差距是 builder 拓扑。改造前 Simpler 的两种 SIMT 模式只让唯一
`block0/AIV0` 启动一个 128-thread VF，实际只有 4 个 warp leader 工作；
standalone 的有效配置可以让多个 AIV builder 并行，二者不是同一供给能力。

第二处、也是当前最大的差距，是动态 DAG writer 查询。当前
`dist_simt_lookup_dag()` 对每个 INPUT/INOUT 都从 `N-1` 向后扫描
`[N-H,N)`；默认 `H=64`。每个候选至少包含返回型 atomic control load，
命中前还会再次读取 control 验证不变；候选未发布时，当前 warp 继续 poll。
四个 leader 又以 task id 交错前进，因此一次慢 metadata 发布会把同一 wave
中的其他 lookup 变成等待。

Scalar `cross_core_dag` 也使用同类逆向查询，但 Build 分散在 96 个 Scalar，
所以首轮没有出现 SIMT 单 builder 下的数量级放大。standalone 的最佳泛化
记录采用按 symbol 精确前驱和更多 builder，已经把大量 predecessor poll
消除；它提供优化方向，但其固定 PA workload、builder 数和端点必须先与
Simpler 动态 Submit 合同逐项对齐，不能直接照抄性能数字。

## 6. 后续收敛顺序

1. 先把 DAG SIMT 的唯一 builder 改成按真实 AIV 拓扑扩展的通用多
   builder 合同；builder rank、leader stride、启动/完成计数和 host
   角色闭合必须一致。普通 SIMT 保留单 builder，避免无故占用执行核。
2. 再独立消减 DAG 的 `O(H × 输入数)` writer 扫描；不得编码 PA task kind
   或固定 `5 × batch` 图形。
3. 每一阶段先跑 CPU/协议门槛，再跑 A5 B1/B256 功能和单轮低扰动性能；
   数量级回退修复后才恢复独立进程多轮采样。
4. 最后只与同 workload、同 startup→FinalDrain 端点的 standalone 结果横比。

## 7. 第一阶段：按算法特性收敛 builder 拓扑

### 7.1 先做全 AIV0 受控对照

首先不猜测多 builder 是否对两种 SIMT 都有利，而是让 32 个 block 的
AIV0 全部启动 builder VF，保持其他业务合同不变：

| 模式 | builder/replay | B256 startup→FinalDrain | 相对单 builder 首样本 |
| ---- | -------------- | ----------------------: | --------------------: |
| `simt_cross_core_ordinary` | 32 / 64 | 164.629 ms | 由 144.191 ms 回退约 14.2% |
| `simt_cross_core_dag` | 32 / 64 | 188.424 ms | 由 1321.880 ms 改善约 85.7% |

结论很明确：

- ordinary Build 足够轻，额外 31 个 builder 的供给收益小于丢失 31 个
  AIV0 replay/执行候选者的代价；
- DAG Build 存在高频动态 writer 扫描，把 4 个 warp leader 扩展到 128 个
  leader 后，原先的单 builder 串行瓶颈被大幅摊平。

因此不保留“两种模式一律 32 builder”，而是收敛为模式内部的通用
拓扑合同：ordinary 单 builder，DAG 每 block 一个 AIV0 builder。

### 7.2 协议改动

多 builder 不是简单地多启动几个 VF，完整合同包括：

1. 以物理 `block_id` 生成 builder rank，不把动态 worker 注册顺序当 ABI；
2. 每个 builder 的 4 个 warp leader 按
   `rank * 4 + warp` 取首 task，以 `builder_count * 4` 为 stride；
3. `builder_started` 计数所有 VF 启动，leader 只在拓扑闭合后消费动态
   request；
4. `builder_finished` 以全局 leader 数闭合，FinalDrain 不能在部分 builder
   退出时提前结束；
5. builder 传入物理 worker id 作为 Build owner，host 和 Python 产物校验使用
   相同的 builder/replay 拓扑。

调度代码不根据 PA 的 Alloc/QK/SF/PV/UP 选 builder，也不依赖 1280
个 task 或 32 block 常量；32 只出现在当前 PA 性能产物的闭合层。

### 7.3 收敛后实测

| 模式/用例 | builder/replay | startup→FinalDrain | 结果 |
| --------- | -------------- | -----------------: | ---- |
| ordinary B1 | 1 / 95 | 1.280 ms | golden PASS |
| DAG B1 | 32 / 64 | 1.692 ms | golden PASS |
| ordinary B256 独立进程样本 1 | 1 / 95 | 151.831 ms | golden PASS |
| ordinary B256 独立进程样本 2 | 1 / 95 | 106.935 ms | golden PASS |
| DAG B256 | 32 / 64 | 188.308 ms | golden PASS |

ordinary B256 两次测量相差约 44.9 ms，证明当前设备上单轮绝对时间
不适合用于声称小比例收益。本阶段保留 builder 拓扑的依据是 DAG
模式数量级改善以及两种模式的角色/协议闭合，不是 ordinary 单样本排名。

DAG B256 的 `perf-clock-kernel` 进一步记录到：

- startup→FinalDrain 为 187.592 ms；
- Kernel 调用数恰好 1024，AIC/AIV 各 512；
- Kernel core-time 合计 61.231 ms，非 Kernel residual 仍为 17944.972 ms。

因此 32 builder 没有通过少执行 Kernel 伪造收益。但 188 ms 仍显著高于
Scalar 两种模式的约 71 ms 首基线，下一阶段必须继续消减 DAG 中
`O(H × 输入数)` 的返回型 atomic writer 扫描，而不是继续增加 builder。

收敛后还在真实 A5 上重跑了非 PA 的
`A5OnboardBd24ExistingInoutChain`：`simt_cross_core_dag` 数值 golden PASS。
该用例的 24 block 会动态得到 24 builder，证明设备协议没有把 PA 的
32 block 当作调度常量。

## 8. 第二阶段候选：否决 Scalar 有序链内的精确依赖索引

### 8.1 验证目的

为消除 SIMT builder 对每个 INPUT/INOUT 执行的
`O(H × 输入数)` metadata 逆向扫描，曾验证一条通用候选路径：由严格有序的
Scalar request publisher 在发布 task N 时，同时发布 writer metadata、查询
精确前驱并更新按 buffer address 索引；SIMT builder 只消费 request 尾部已经
解析好的 fanin。

该候选不编码 PA task kind，并处理了三类通用情况：

- 同一 buffer 始终使用相同 region 时，索引直接返回最新 producer；
- 同一 buffer 出现不同 region 时标记为 ambiguous，回退到有界 metadata 扫描；
- hash 冲突使用线性探测，索引耗尽同样回退，不改变依赖正确性。

协议单测、CCEC 构建、A5Sim PA B1，以及真实 A5 非 PA 的
`A5OnboardBd1ExistingInoutChain`、`A5OnboardBd24ExistingInoutChain` 均已通过。
验证期间还发现：把 writer 与 query 同时保存为两个局部数组会令 CCEC 热函数的
临时栈接近 2 KiB，造成真实 A5 卡在 request 发布/输出等待；改成两次直接遍历、
不保存 query 数组后，非 PA 上板正确性恢复。

### 8.2 性能结果

相同 PA B256、shared TensorMap、`simt_cross_core_dag`、startup 到
FinalDrain 的 `perf-clock` 口径下：

| 实现 | 单次端到端时间 | 相对 188.308 ms 基线 |
| ---- | -------------: | -------------------: |
| 32 builder 原 metadata 扫描 | 188.308 ms | 基线 |
| 精确索引，逐字段跨地址空间读取 | 281.592 ms | 回退约 49.5% |
| 精确索引，每 Tensor 合并为一次快照 | 282.252 ms | 回退约 49.9% |

把每个 Tensor 的五次 CCEC `noinline` 字段读取合并为一次快照后，端到端时间
没有改善，说明主要损耗不是函数调用次数，而是把 Tensor region 解析、索引
invalidate/update、metadata 发布和 fanin 查询放进了所有 task 共用的 Scalar
有序发布链。该链原本只负责轻量 immutable request 发布，新增工作无法被 32 个
SIMT builder 并行摊分。

### 8.3 裁决与后续约束

这条候选已完整撤回，不提交实现代码。后续消减 DAG writer 查询必须满足：

1. 不把按 Tensor 的依赖解析搬进全局串行 request 发布链；
2. 保留通用 region/alias/manual dependency 语义，不以 PA 固定图替代查询；
3. 优先让索引构建或查询留在多 builder 可并行的位置；
4. 每个候选先通过非 PA INOUT 链正确性，再以 188.308 ms 为端到端保留门槛。

## 9. 第二阶段保留项：不可变控制字只做一次返回型读取

### 9.1 冗余来源与改动边界

mode4 的 request 与 DAG metadata 都遵循 publish-once 合同：发布者先写完整
payload，再用 control atomic 发布；本轮运行中不再修改，只有下一轮启动前由
AICPU 复位。原 SIMT 热路径仍存在三类重复返回型 atomic：

- builder 轮询到 request `Published` 后，decode 再读一次 control；
- decode 读取 immutable payload 后，第三次读取相同 control；
- DAG lookup 首次读到非零 metadata control 并检查 payload 后，再次读取同一
  control；metadata 发布前还先 load，再用 CAS 检查一次冲突。

收敛后，轮询取得的首个 published control 直接传给 decode；metadata lookup
也只保留首次非零 acquire。发布冲突仍由最终 CAS 的返回值检查。改动没有减少
任何 Tensor region、writer、fanin 或 payload 校验，也没有改变 builder/execute
拓扑和 TensorMap 顺序。

### 9.2 同时段 A/B 实测

相同 PA B256、真实 A5、shared TensorMap、startup 到 FinalDrain 口径，每次为
独立 pytest 进程：

| 实现 | 三次端到端时间 | 中位数 |
| ---- | -------------- | -----: |
| 单次读取候选 | 183.221 / 185.231 / 189.940 ms | 185.231 ms |
| 恢复旧重复读取 | 185.215 / 186.827 / 187.642 ms | 186.827 ms |

候选中位数降低 1.596 ms，约 0.85%；两组范围仍有重叠，因此只认定为小幅收益，
不把最快样本当作稳定提升。通用正确性方面，相关协议单测全部通过，AIC/AIV
CCEC clean build 通过，真实 A5 非 PA 的
`A5OnboardBd24ExistingInoutChain` 也通过。该项建立在明确的不可变发布合同上，
不含 PA 特例，予以保留。

## 10. 第二阶段候选：否决按 owner 缩短 metadata 扫描窗

### 10.1 候选与正确性证据

Tensor descriptor 的有效 `owner_task_id` 已作为 fanin 加入，因此从依赖集合
语义看，region 查询只需寻找 owner 之后、consumer 之前的更新 writer。候选
据此把 lookup history 收窄为：

```text
min(H, consumer_task - owner_task - 1)
```

owner 无效的外部 Tensor 仍使用完整 H。该规则不识别算子类型，公共边界单测、
Scalar/SIMT DAG CCEC 构建，以及两种 DAG 模式的真实 A5 非 PA Bd24 INOUT 链
均通过。

### 10.2 端到端否决

mode4 PA B256 三次独立进程结果为 206.913 / 205.614 / 213.672 ms，中位
206.913 ms；当前保留版三次中位为 185.231 ms，候选回退约 11.7%。这已经超出
同期波动范围，因此实现代码已完整撤回。

候选确实减少了 metadata control 查询，但本轮没有 builder 分阶段计数，不能把
回退原因直接断言为某一个后续 atomic。可验证的事实只有：加速 lookup 后，大量
builder 更早进入 heap/output/exec 共享发布阶段，整体时间反而恶化；“原扫描形成
自然错峰、消除后放大后段竞争”是下一轮需要用计数验证的假设，不作为既成结论。
后续若再减少扫描，必须同时观测 lookup 与后续共享发布，而不能只按减少 GM/atomic
次数判断收益。

## 11. 第二阶段保留项：唯一 SIMT builder 直接发布 Built

SIMT builder 以 `builder_rank * warps + warp` 取得首 task，并以全局 leader 数为
stride；同一 task 只可能由一个 `(builder, warp)` 构建。旧路径仍沿用多方 Build
竞争协议，先做 `Empty -> Building` CAS，写 payload 后再做
`Building -> Built` CAS。

保留版在 control 仍为 Empty 时构造不可见 payload，最后用一次
`Empty -> Built` CAS 同时完成发布与唯一 owner 校验。如果拓扑错误导致第二个
builder 到达，CAS 返回值仍会令其 fail-closed。Immediate task 继续执行
`Built -> Done`，task completion 与依赖可见顺序不变。Scalar 构建路径仍保留
通用 Building reservation，不受此优化影响。

真实 A5 非 PA Bd24 INOUT 链通过；PA B256 三次独立进程结果为
183.457 / 184.400 / 188.445 ms，中位 184.400 ms。相对上一保留版 185.231 ms
中位降低 0.831 ms，约 0.45%。两组范围重叠，仍只认定为符合协议的一次明确
atomic 消减，不将其描述为主要瓶颈突破。

## 12. 第三阶段保留项：在 Build 供给与 Execute 供给之间收缩 builder 拓扑

### 12.1 问题与扫描方法

全 32 个 AIV0 承载 builder 能消除单 builder 的数量级瓶颈，但也会把
AIV replay/Execute 候选核从 64 个减少到 32 个。因此不能把“builder
越多越好”当作协议结论。本轮仅改变 builder 数量，保持 request、
DAG lookup、TensorMap、exec ticket 和 startup→FinalDrain 端点不变，依次
测量 K=8/16/24，并与已保留的 K=32 对照。

首轮取数如下：

| builder K | replay 核 | B256 startup→FinalDrain | 判断 |
| --------: | ----------: | ------------------------: | ---- |
| 8 | 88 | 208.016 ms | Build 供给不足，明显回退 |
| 16 | 80 | 172.833 ms | 首轮最优，进入重复验证 |
| 24 | 72 | 182.619 ms | 优于 K=32，但不及 K=16 |
| 32 | 64 | 184.400 ms | 上一保留版三次中位数 |

K=8/16/24 扫描初期，设备 raw 已闭合对应 builder/replay 数量，但
Python 产物校验器仍固定期望 32 builder，因此 pytest 在设备完成后按
角色数不匹配拒绝产物。选出 K=16 后，校验器与测试已同步到新拓扑，
后续三次结果都是完整 PASS，不使用被旧校验器拒绝的扫描产物充当
最终正确性证据。

### 12.2 保留协议

保留版的拓扑不读取 PA task kind、batch 或固定 DAG：

1. `simt_cross_core_ordinary` 仍保留 1 个 builder；
2. `simt_cross_core_dag` 选择 `block_id < min(block_count, 16)` 的 AIV0；
3. builder rank 仍与连续 block id 一致，task stride 仍由实际
   `builder_count * 4 warps` 计算；
4. Host 泳道角色、AICore 启动/完成闭合和 Python `perf-clock`
   校验使用同一拓扑规则；
5. `FdwicBuildIdentity` 的稳定 64 B 前缀消耗一个既有 reserved word
   记录编译期 builder limit，Host/AICPU/AICore 任一镜像不一致都
   fail-closed。

这个数量是调度器拓扑配置，不是 PA 语义特例。但当前性能取舍只在
PA B256 上完成了稳定多轮取数；其他 DAG 工作负载可能有不同的最优
供给比。因此后续接入其他算子时必须复核端到端性能；若需要可调，
应将 builder limit 提升为明确的构建身份，不得根据 PA 内部 task
类型在热路径中动态分支。

### 12.3 最终实测与验证

K=16 重建 Host/AICPU/AICore 后，PA B256 三个独立进程完整 PASS：

```text
173.206 ms / 172.669 ms / 173.976 ms
median = 173.206 ms
```

相对 K=32 保留版 184.400 ms 中位数降低 11.194 ms，约 6.1%。
此外，真实 A5 非 PA `A5OnboardBd24ExistingInoutChain` 数值 golden PASS；
16 个定向 FDWIC C++ 身份/协议测试全部 PASS；`test_scene_test_cache.py`
140 项 PASS。全仓 C++/Python 构建仍被 A2/A3 `PTO2TaskPayload` 既有布局
断言（期望 576，当前 568）阻断，本轮没有改动该架构。

### 12.4 否决每 builder 增加 warp

K=16 固定后，又将每个 builder 从 128 thread/4 warp 单变量提高到
160 thread/5 warp。该配置能正常 CCEC 构建，PA B256 也完整 PASS，但
startup→FinalDrain 为 254.568 ms，相对 W4 的 173.206 ms 中位数回退
约 47%。这不在同期 W4 波动范围内。

本轮只能证明：在 Simpler 真实动态 Submit、K=16 和当前 GM/atomic
协议下，增加 leader 数量会大幅恶化端到端性能。没有分阶段计数时，
不把回退唯一归因为某条 atomic 或 VF 资源。W5 已完整撤回，保留
W4；既然 W5 已是数量级回退，不继续盲测 W8。
