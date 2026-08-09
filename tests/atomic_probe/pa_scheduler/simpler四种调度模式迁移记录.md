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
| --------: | --------: | ----------------------: | ---- |
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

## 13. 第四阶段候选：否决 metadata 全任务前缀发布

### 13.1 候选协议

mode4 的 DAG lookup 会对每个 INPUT/INOUT 在最近 `H=64` 个 task 中逆向
扫描；每个候选 metadata control 都通过需要返回值的 `atomicAdd(0)` 读取。
本轮尝试把“候选 control 已发布”收敛为一个连续 task 前缀：

1. task N 的 builder 先写完整 writer metadata；
2. N 等待 task N-1 的 prefix-ready 位，再原子发布自己的 control；
3. lookup 依靠该前缀证明 `[0, N)` 的 control 已经发布；
4. 候选 control 改用 CANN 正式 `asc_ldcg` 读取。该接口绕过 L1，保留
   正常 L2 缓存，不要求返回型 atomic。

实现复用了既有 64 bit control 的保留位，没有增加额外 GM 字段。曾先验证
`__builtin_cce_ld_dev`，但它在当前 SIMT VF 的 dav-c310-vec 编译前端触发
退出码 139，因此未进入上板版本；上板候选只使用已查证实现的 `asc_ldcg`。

### 13.2 正确性证据

开发顺序是先补 CPU 协议测试，证明 prefix 不允许跨过尚未发布的前序 task，
再修改 CCEC 热路径。以下检查均通过：

- 四个 DAG build identity、protocol 和 state 定向测试；
- mode4 AIC/AIV CCEC 完整构建；
- 真实 A5 非 PA `A5OnboardBd24ExistingInoutChain`；
- PA B1 与 PA B256 完整数值 golden。

因此否决原因是性能，不是功能没有闭合。

### 13.3 A5 性能与裁决

相同 PA B256、shared TensorMap、K16/W4、startup 到 FinalDrain 的
`perf-clock` 口径，三个独立进程结果为：

```text
178.674 ms / 175.802 ms / 172.999 ms
median = 175.802 ms
```

当前保留版中位数为 173.206 ms，候选回退 2.596 ms，约 1.50%；候选前两次
也都慢于保留版三次结果中的最大值 173.976 ms，不能认定有稳定收益。实现代码
与临时协议测试已完整撤回，只保留本节与 investigation 记录。

该候选减少了 lookup 的返回型 atomic，但同时把原本由 K16 builder 并行发布的
metadata 变成完整 task 顺序链。它也可能消除了 builder 到达后续共享阶段的
自然错峰；当前没有分阶段计数，后一点只作为待验证解释，不写成既定根因。

后续若继续消减 metadata lookup，应优先考虑不引入全任务串行前缀的局部索引、
按 symbol/region 缩小候选集合，或先补 builder 分阶段计数定位真实占比；不要
为了替换候选 atomic 再引入同类全局 prefix chain。

## 14. 第四阶段保留项：合并同一 task 的 DAG 查询窗口

### 14.1 重复工作与等价改写

mode4 原实现按 Tensor 分别调用 `dist_simt_lookup_dag()`。若一个 task 有
`Q` 个自动依赖的 INPUT/INOUT，每个查询都从 `N-1` 逆向扫描至命中或
`N-H`，因此同一批 immutable metadata control 最坏会被返回型 atomic
读取 `Q × H` 次。

保留版改为一次 task 级扫描：

1. 先用 32 bit 本地 mask 标记最多 32 个自动依赖查询；
2. 按 task id 从近到远，每个候选 control 只 acquire 一次；
3. 将该候选的 writer regions 与所有尚未命中的查询比较；
4. 每个查询的首次命中仍是逻辑上最近的 writer；
5. 扫描完成后再按原 Tensor 顺序把 owner、lookup 与显式依赖加入 fanin。

因此最坏 control 读取上界由 `Q × H` 降为 `H`。这项改动没有新增共享
字段、DCCI、atomic 地址或全局顺序链，也没有改变 metadata 发布、K16/W4
builder 拓扑和执行包协议。代价仅是每个活跃 warp leader 保存一个最多
32 项的本地 producer 数组。

### 14.2 正确性与构建验证

以下验证均通过：

- 四个 DAG build identity、protocol 与 state 定向 C++ 测试；
- mode4 AIC/AIV CCEC 完整构建；
- 真实 A5 非 PA `A5OnboardBd24ExistingInoutChain`；
- PA B1 数值 golden，端到端 1.532 ms；
- 四次 PA B256 数值 golden 与 worker/builder 角色闭合。

该算法只依赖 Tensor tag、manual dependency、region overlap、task id 与
history，不读取 PA task kind、batch 或固定 DAG。

### 14.3 A5 性能与保留判断

相同 PA B256、shared TensorMap、K16/W4、startup 到 FinalDrain 口径：

| 实现 | 独立进程结果 | 中位数 |
| ---- | ------------ | -----: |
| 原逐 Tensor 扫描 | 173.206 / 172.669 / 173.976 ms | 173.206 ms |
| task 级批量扫描，前三次 | 172.157 / 173.895 / 172.656 ms | 172.656 ms |
| task 级批量扫描，补充一次 | 172.372 ms | 四样本中位 172.514 ms |

按前三次同样本数比较，中位改善 0.550 ms，约 0.32%；加入补充样本后，
四样本中位相对原基线改善 0.692 ms，约 0.40%。两组区间仍重叠，所以只
认定为小幅收益；保留依据同时包括明确减少重复返回型 atomic、共享协议
完全不变和非 PA 正确性闭合，不把该结果描述为数量级突破。

### 14.4 否决同轮的 region 快捷分支

在批量扫描上又尝试过两个源码级微调：先比较 descriptor address 再计算
region，以及让 caller 跳过已经在 lookup 中完成的第二次 region 校验。
它们保持数值正确，但两次 B256 分别回退到 197.612 / 187.161 ms。完整撤回
后立即恢复到 172.372 ms，已排除单次设备波动解释。

当前没有 VF 汇编与分阶段计数，不能把回退武断归因于某个寄存器或分支；只
记录该 CCEC 代码形态不可保留。对应 investigation 保存重新评估条件。

## 15. mode3 中间 builder 拓扑扫描：继续保留 K1

### 15.1 扫描动机与协议修正

早期 `simt_cross_core_ordinary` 只比较过 K1 与 K32，没有覆盖 K2/K4/K8
这类中间供给。由于 mode3 的端到端仍明显慢于 Scalar 两种模式，本轮保持
ordinary TensorMap、request、execute ticket、W4 和 startup 到 FinalDrain
端点不变，只扫描 builder block 数量。

把 K1 临时扩为 K2 时首先触发 fail-closed。根因不是 TensorMap：K1 收敛后
mode3 的 `builder_rank` 固定为 0，多 VF 会重复消费同一 task 流。实验版将
rank 恢复为物理 block id 后，K2/K4/K8/K16 都完成 PA B256 数值 golden，
builder/replay 数量也与拓扑一致。该 rank 改动在恢复 K1 后没有独立功能价值，
最终一并撤回。

### 15.2 同窗口扫描结果

| ordinary builder K | replay 核 | B256 startup→FinalDrain |
| -----------------: | --------: | ----------------------: |
| 1 | 95 | 94.732 ms |
| 2 | 94 | 136.467 ms |
| 4 | 92 | 118.651 ms |
| 8 | 88 | 106.304 ms |
| 16 | 80 | 93.843 ms |

K16 首样本只比 K1 快约 0.94%，因此又分别重复：

```text
K16: 93.843 / 100.282 / 97.869 ms, median = 97.869 ms
K1 : 104.105 / 93.608 / 138.501 ms, median = 104.105 ms
```

K1 在同一轮内出现 93.608～138.501 ms 的巨大波动。若把扫描前的 94.732 ms
也计入，K1 四样本中位约 99.419 ms，K16 的中位优势只剩约 1.56%，且两组
范围明显重叠。

### 15.3 裁决

本轮不把 K16 固化为公共默认值，继续保留 K1，原因是：

1. 端到端收益没有超过当前 mode3 波动；
2. K16 固定减少 15 个 AIV replay/Execute worker，这一取舍可能随算子 Build
   重量变化，不能只凭 PA 单负载设为通用合同；
3. K2/K4/K8 都未超过 K1，说明“增加 builder 必然更快”不成立；
4. 当前没有 Build 与 Execute 分阶段计数，无法解释 K1 长尾究竟来自 builder
   供给、设备状态还是后段执行竞争。

后续若重启该方向，应先让 builder limit 成为明确的构建身份并采集至少两个
非 PA 工作负载，再用交错 A/B 和 Build/Execute 分段数据选择默认值。不得把
PA task kind、batch 或固定 task 图写进 builder 选择逻辑。

## 16. 通用延迟输出引用：去除 replay 全核逐 task 等待

### 16.1 根因与修正边界

前述约 71 ms 的 Scalar cross-core 结果不是 DAG 查询本身造成的：
`cross_core_ordinary` 和 `cross_core_dag` 分别为 71.442 / 70.907 ms，
已经证明两者共有的 Submit 语义才是首要问题。源码核对后确认：

1. 旧 `TaskOutputTensors` 合同要求 Finish 返回时 TensorDesc 已完整可读；
2. 迁移版让 96 个 replay actor 都在每个 task 的 Finish 中取得
   `SharedTaskOutputCell`；
3. 因而 loser/非 builder 也会轮询 builder 发布，把原本的延迟输出
   引用错做成了逐 task 的全核屏障。

修正不改动旧 API，而是增加一组通用 deferred compete-first API：

- 调用方显式给出 task schema 中的预期输出数；
- 所有 replay actor 立即获得稳定 `(task_id, output_slot)` 引用；
- 只有唯一 builder 构造 Args、校验实际输出数、Materialize 并发布
  TensorDesc；
- 后续 task 的 builder 在真正解析 fanin 时，才等待它实际依赖的
  producer output cell；
- 非 builder 仍进入 Finish，保留与 engine 类型匹配的 Execute owner
  竞争，不把 Build 与 Execute 重新绑回同一个核。

旧 `TaskOutputTensors/get_ref()` 的“返回时已物化”语义完全保留，
same-core 路径不变。新合同只在调用方显式选择 deferred API 时生效，
避免用性能修正暗中改写全仓接口语义。

### 16.2 内存与顺序合同

该修正只把“谁需要等待 TensorDesc”从全体 replay actor 收窄到真正的
fanin consumer，不改以下不变量：

- TensorMap writer metadata 仍严格按 task id 插入；
- output descriptor 仍是“写 payload → clean-out DCCI → atomic publication”；
- consumer 仍是“atomic acquire → invalidate descriptor lines → copy”；
- Build N 仍拒绝 `producer_task_id >= N` 的未来引用；
- builder 在发布任何共享状态前校验实际 output arity，不让调用方
  schema 与 Args 不一致时带着半发布状态继续运行。

`SharedTaskOutputs` 仍为 8 byte trivial POD，只保存 task id 与 output count；
新增 `reset_deferred()` 支持通用 `MAX_TENSOR_ARGS` 上限，旧 PA helper 的
8-output 限制与 ABI 均未改动。

### 16.3 真实 A5 证据

本阶段只以 CCEC 和真实 A5 为有效证据，不使用 A5Sim 裁决
SIMT/cross-core 路径。

- 四组 shared/cross-core C++ 布局与协议测试全部 PASS；
- 非 PA `A5OnboardBd24CompeteFirstFreshOutput` 在 24 worker 下数值验证 PASS，
  证明通用 fresh output 能被后续 task 作为 fanin 消费；
- PA B1 数值 golden PASS；
- PA B256 数值 golden PASS，96 个 replay core 均精确完成 1280 次
  Submit；
- B256 `perf-clock` 的 startup→FinalDrain 为 **24.188 ms**，原
  `cross_core_ordinary` 基线为 **71.442 ms**，单阶段降低
  **47.254 ms / 66.1%**。

新数据位于
`outputs/TestPagedAttentionUnroll_Case1_20260809_145059/fdwic_perf_clock_summary.json`。
这个对比证明 replay 全核逐 task 输出等待是主要回退源，但 24.188 ms
仍明显高于 same-core 的毫秒级目标，不宣称问题已完全解决。下一阶段
应分别定位 Build/Execute owner 绑定等待、严格 TensorMap 插入链与
Build dispatch 供给，不再把它们混成“DAG 太慢”。

### 16.4 `cross_core_dag` 复用同一合同

mode2 没有新增 DAG 专用输出状态，只把 PA 和非 PA smoke 的调用条件
扩展到已有 deferred API。严格 TensorMap 插入、DAG metadata 发布与
fanin lookup 均继续由 mode2 原协议处理。

真实 A5 验证结果：

- 非 PA `A5OnboardBd24CompeteFirstFreshOutput` PASS；
- PA B1 数值 golden PASS；
- PA B256 数值 golden PASS，96 核均精确完成 1280 次 Submit；
- startup→FinalDrain 为 **24.459 ms**，相对旧 mode2 基线
  **70.907 ms** 降低 **46.448 ms / 65.5%**。

新数据位于
`outputs/TestPagedAttentionUnroll_Case1_20260809_150934/fdwic_perf_clock_summary.json`。
mode1/mode2 分别为 24.188 / 24.459 ms，只相差 0.271 ms；它们共同从
约 71 ms 降到约 24 ms，进一步证明主要改善来自去除共有的全核输出
等待，而不是削弱 DAG 语义。

### 16.5 SIMT ordinary 中将描述符解析收窄到唯一 publisher

SIMT 路径不能直接复用 Scalar builder 的 Finish：Scalar replay actor 先竞争
request publisher，再由持久 SIMT VF 消费不可变 request 并构建 execution
payload。本轮保留该分工，没有把 Build 退回 Scalar，也没有扩大 request ABI：

1. `ValidateSimtL0TaskArgs()` 接受 plain `FdwicOutputRef`，但拒绝未来
   task、非 plain 引用、越界 slot 和预期 output arity 不一致；
2. 唯一 request publisher 只等待当前 task 实际使用的 producer output cell，
   acquire 后把完整 TensorDesc 复制到本核 task payload 暂存；
3. request source 将该完整描述符按旧格式打包，SIMT builder 继续只看
   TensorDesc，因此 ordinary TensorMap、fanin 和 execution payload 协议不变；
4. 其余 replay actor 不再取得本 task 的 output cell，但与 engine 匹配的
   actor 仍参与独立 Execute owner 竞争。

这个版本仍会让唯一 Scalar publisher 支付真实 producer 描述符的等待与
copy，但已把代价从“95 个 replay actor 每 task 都等”收窄为“1 个 publisher
只等实际 fanin”。将符号引用编入 request、再由 SIMT builder 直接解析可能
进一步减少 Scalar 工作，但会扩大 request 协议和控制面，不在本次功能
收敛中冒进引入。

验证结果：

- SIMT request 协议 C++ 单测 8 项全部 PASS，包括 shared ref 解析、
  output arity 和未来引用拒绝；
- 真实 A5 非 PA `A5OnboardBd24CompeteFirstFreshOutput` PASS；
- PA B1 数值 golden PASS；
- PA B256 三个独立进程均 PASS，`95 replay + 1 builder` 角色与每核
  1280 次 Submit 均闭合；
- startup→FinalDrain 为 **63.613 / 64.551 / 64.742 ms**，中位数
  **64.551 ms**。相对第 15 章保留的旧 K1 四样本中位数
  **99.419 ms** 下降 **34.868 ms / 35.1%**。

三份新数据分别位于：

- `outputs/TestPagedAttentionUnroll_Case1_20260809_151947/`；
- `outputs/TestPagedAttentionUnroll_Case1_20260809_152101/`；
- `outputs/TestPagedAttentionUnroll_Case1_20260809_152206/`。

### 16.6 SIMT DAG 不另造输出协议

mode4 只在 orchestration 调用层启用第 16.5 节已验证的 deferred API。
它继续使用 K16/W4 builder 拓扑、现有 DAG metadata 发布和同 task 批量
fanin lookup；没有添加 mode4 专用的 output cell、新 atomic 或新顺序链。

真实 A5 验证结果：

- 非 PA `A5OnboardBd24CompeteFirstFreshOutput` PASS；
- PA B1 数值 golden PASS；
- PA B256 三个独立进程均 PASS，`80 replay + 16 builder` 角色和每核
  1280 次 Submit 均闭合；
- startup→FinalDrain 为 **39.792 / 39.361 / 39.224 ms**，中位数
  **39.361 ms**。相对第 14 章保留版的四样本中位数
  **172.514 ms** 下降 **133.153 ms / 77.2%**。

三份新数据分别位于：

- `outputs/TestPagedAttentionUnroll_Case1_20260809_152749/`；
- `outputs/TestPagedAttentionUnroll_Case1_20260809_152852/`；
- `outputs/TestPagedAttentionUnroll_Case1_20260809_153003/`。

mode4 的 DAG 语义与 builder 并行度没有变，仅去除共有的 replay 全核输出
等待就从 172.514 ms 降至 39.361 ms。因此之前把 mode4 数量级回退主要
归因于 DAG lookup 是不完整的；当时更大的共同因素是错误的输出合同。
当前 39.361 ms 仍不是目标性能，后续再在新基线上分析 publisher 描述符等待、
SIMT Build 供给和 Execute 进度，不应回到旧基线上微调 DAG 分支。

## 17. 通用 Build 仲裁与非 PA 边界矩阵

### 17.1 两级仲裁不读取算子语义

Scalar mode1/mode2 原先让所有 replay worker 直接竞争同一个 task 的
`SharedExecCell`。A5 对同地址返回型 atomic 的并发代价较高，因此复用
same-core 已验证的 task-private tournament 布局，把一次 Build owner 仲裁
拆成两层：

1. 全部 worker 按 `core_idx % min(num_workers, 8)` 竞争本组 owner；
2. 只有各组唯一 winner 再竞争 root owner；
3. root winner 继续调用原有 `ReserveExecBuild()`，之后的 Materialize、
   TensorMap 插入、Build payload 发布和 Execute owner 协议不变。

该路径只读取 worker 数、core id 和 task id；不读取 PA `TaskKind`、batch、
QK/SF/PV/UP，也不裁剪某类 task 的候选核。8 是 tournament 布局已有的最大
组数，不是 PA 的 C/V 候选数量。

两层仲裁引入了一个必须显式支持的通用时序：local loser 可能先到达 Execute
绑定，而 root winner 尚未把 execution cell 从 `Empty` 推进到 `Building`。
Build owner 与 Execute owner 本来就是独立角色，因此 `Empty` 在这里表示
Build 尚未发布，不是控制字损坏；Execute 侧继续等待并定期检查 fatal。

### 17.2 新增通用边界用例

所有用例都放在存量 `submit_dependency_smoke`，复用已有 AIC/AIV kernel，
没有新建 PA adapter 或测试专用运行时分支。

- `Bd1BuildExecuteSkewNoFreshOutput`：3 worker、1 task，验证 worker 少于
  8 时的动态分组和零 fresh output；
- `Bd4BuildExecuteSkewNoFreshOutput`：12 worker、16 task，验证 worker
  多于 8 且各组人数不均；
- `Bd32BuildExecuteSkewNoFreshOutput`：96 worker、128 task，验证
  Build/Execute 到达次序和独立 region；
- `Bd32DeferredCrossRoleChain`：32 条链、96 task，验证 AIC→AIV→AIC
  deferred output 消费；
- `Bd32DeferredMultiOutput`：16 组、64 task，验证每 task 两个 output、
  分槽发布和 INOUT 消费；
- `Bd32RepeatedInoutChain`：96 worker、32 writer，验证同一 region 的
  严格 writer 前驱链。
- `Bd32ZeroSubmit`：96 worker、零 task，验证只有 startup/FinalDrain 时
  不臆造 owner、不错误等待不存在的执行工作；
- `Bd7AicOnlyPrimeTasks`：21 worker、17 个独立 AIC task，验证非 2 次幂
  task 数、非均匀仲裁分组，以及 AIV 完全无任务时的闭合；
- `Bd7AivOnlyPrimeTasks`：21 worker、17 个独立 AIV task，反向验证 AIC
  完全无任务时的闭合；
- `Bd7AlternatingCrossEngineInout`：17 组 AIC→AIV writer 交接，验证同一
  region 的跨引擎严格前驱顺序，不依赖 PA 的固定五任务 DAG。
- `Bd32TaskCapacityMinusOne`：96 worker、2047 个零参数 task，
  验证 task 状态容量上限前一项；
- `Bd32ExactTaskCapacity`：96 worker、2048 个零参数 task，验证
  末个合法 request、SIMT Builder 封口和并发 Execute 尾票退出。

`Bd32BuildExecuteSkewNoFreshOutput` 交替提交 AIC/AIV，并让每个 task 写独立
subview；它不会因为业务依赖把 Build 人为串行化，能放大 Execute 先看到
`Empty` 的时序。多输出用例通过通用 deferred alloc 一次发布两个 output，
后续 AIC/AIV 分别写入并合并，不依赖 PA 的三输出 Alloc 图形。

两个 task 容量用例交替提交 AIC/AIV 零参数 kernel，不生成
TensorMap writer，也不申请 heap。这使 2047/2048 验证只针对跨核
task 状态容量，不会像输出 view 链那样先命中 TensorMap bucket 的
独立容量合同。

### 17.3 四模式真实 A5 结果

以下四种 scheduler 均运行上述十二个 case，合计 48 个模式/场景组合：

- `cross_core_ordinary`：全部数值 golden PASS；
- `cross_core_dag`：全部数值 golden PASS；
- `simt_cross_core_ordinary`：全部数值 golden PASS；
- `simt_cross_core_dag`：全部数值 golden PASS。

2026-08-09 在同一最终源码状态上逐一重跑三镜像和真机数值
golden，48 个组合全部 PASS。零 Submit、单引擎空闲、17-task 非均匀
数量和 2047/2048 容量边界均由同一公开 orchestration API 产生，
没有增加调度器测试专用分支。

2048-task 首次真机运行暴露了 SIMT Builder 的通用封口时序：
leader 完成 task 2047 后可能先于 replay sealer 走到 request 数组尾部。
修正后 Builder 不访问越界 cell，而是等待 seal/fatal 后再区分正常穷尽和
真实容量错误。该分支只使用通用 task 容量和 lifecycle 状态。

同时，cross-core execution/output/TensorMap 协议与 ordinary/DAG state 五组
C++ 单测全部 PASS。没有运行 A5Sim；SIMT/cross-core 的裁决继续只使用 CCEC
构建和真实 A5。

Scalar ordinary 的 PA B256 回归也数值 PASS。该版本第一份
startup→FinalDrain 样本为 **11.058 ms**，相对第 16.3 节的 **24.188 ms**
单样本改善约 **54.3%**。这里仅证明候选没有以正确性换性能，并记录量级；
尚未取得多次中位数，且距离 2 ms 目标仍远，不把该数字写成稳定性能结论。

## 18. SIMT request publisher 复用通用两级仲裁

### 18.1 直接竞争的通用问题

mode3/mode4 的唯一 request publisher 原先由全部 replay worker
直接竞争同一个 task 的 `SimtBuildRequestCell::control`。PA B256 中，
mode3/mode4 分别有 95/80 个 replay worker；因而每个 task 都会在同一
地址产生 95/80 路返回型 CAS。这与 Scalar 模式第 17 章已处理的
Build owner 竞争是同一类硬件约束，没有必要再造一套仲裁布局。

保留版把 `CrossCoreRuntimeState::build_tournament` 变为四种 cross-core
模式的公共状态，并直接复用
`dist_cross_core_win_build_tournament()`：

1. 所有合法 replay worker 仍是候选者，不固定 PA 核集合；
2. 分组只由 `num_workers` 和 `core_idx` 决定；
3. 每组唯一 winner 再竞争 root，root winner 才会触碰 request control；
4. SIMT builder 数量、request ABI、TensorMap 插入、fanin、execution payload
   和 Execute owner 协议均不变。

该改动不读取 PA task kind、batch、Tensor 形状或固定 DAG；小 block
下候选者少于分组上限时，分组数会按实际 worker 数收缩。

### 18.2 非 PA 边界回归

受影响的 mode3/mode4 分别运行第 17.2 节六类通用边界，合计
12 个真实 A5 模式/场景组合，全部数值 golden PASS。覆盖范围包括：

- 3/12/96 worker 和非均匀分组；
- 无 fresh output 的 Build/Execute 到达偏斜；
- AIC→AIV→AIC deferred 依赖链；
- 一个 task 的多 output 分槽消费；
- 同一 region 的重复 INOUT writer 链。

SIMT ordinary/DAG state 和 request protocol 定向 C++ 测试也全部 PASS。
本阶段没有使用 A5Sim。

### 18.3 PA B256 端到端结果

相同 shared TensorMap、PA B256、startup→FinalDrain `perf-clock`
口径，每组都是三个独立 pytest 进程：

| 模式 | 保留版三次结果 | 中位数 | 本轮前对照 | 改善 |
| ---- | -------------- | -----: | ---------: | ---: |
| `simt_cross_core_ordinary` | 55.534 / 55.914 / 55.332 ms | 55.534 ms | 64.551 ms | 14.0% |
| `simt_cross_core_dag` | 24.813 / 24.544 / 24.512 ms | 24.544 ms | 39.361 ms | 37.6% |

这些数据证明 request control 的同地址并发是 SIMT 路径的真实大头。
同时，24.5/55.5 ms 仍远高于 2 ms 目标：两级仲裁只降低了竞争
扇入，没有消除 80～95 个 worker 对 1,280 个 Submit 的全量 replay。
后续需要改造 replay/Build/Execute 的任务发放方式，不应继续把同一
收益误解为 PA DAG 或某种 Tensor 特例。

## 19. 只为 SIMT DAG 保留 Execute 两级仲裁

### 19.1 全模式候选扫描

Build/request owner 收敛后，每个可执行 task 仍会让所有同 engine
worker 直接 CAS 同一个 `execute_owner`。实验版为 Execute 增加了与
Build/request 独立的 task-private tournament，避免两类 owner 在同一份
状态上相互覆盖。所有同 engine worker 仍是候选者，不改执行资格。

首轮 PA B256 单样本扫描为：

- `cross_core_ordinary`：11.123 ms，对照 11.058 ms，无收益；
- `cross_core_dag`：5.120 ms，对照 5.151 ms，约 0.6%，在噪声内；
- `simt_cross_core_ordinary`：55.201 ms，对照 55.534 ms，约 0.6%，
  在噪声内；
- `simt_cross_core_dag`：22.637 ms，对照 24.544 ms，需要复测。

因此没有把“一个模式有效”扩大为“四种模式都增加状态”。
mode1/mode2/mode3 已撤回 Execute tournament，不为无可证收益的路径
每个 task 额外保留 4,608 B。

### 19.2 mode4 的结构性适用边界

`simt_cross_core_dag` 使用 16 个 AIV0 builder，与其他三种后端的
replay/Build/Execute 重叠拓扑不同。该后端的 Execute 两级仲裁三次
独立进程为：

```text
22.637 / 22.806 / 22.407 ms
median = 22.637 ms
```

相对第 18 章 24.544 ms 中位数改善约 7.8%，三次都低于对照
最快样本 24.512 ms。将状态和热路缩回为 mode4 编译期独有后，
最终确认样本为 **22.356 ms**。

保留条件是 scheduler backend 的 builder/replay 拓扑，不是 PA task kind、
batch、Tensor 形状或 DAG 内容。同 engine 的全部非 builder worker 仍参与
仲裁，Build owner 与 Execute owner 仍完全解耦。

### 19.3 正确性和观察闭合

- mode4 的六类非 PA 边界全部数值 golden PASS；
- 四种 mode 的 AIC/AIV、AICPU 和 Host 均完整 CCEC 构建通过；
- ordinary/DAG/SIMT state 与 execution protocol 五组 C++ 测试通过；
- 泳道 atomic site 增加 Execute local/root 名称，两者均按
  `return_ready` CAS 观察；
- 本阶段不使用 A5Sim。

该优化仍只是降低 owner CAS 的并发扇入，没有改变 80 个 replay
worker 全量运行 1,280 次 Submit 的基本架构，因而不会单独把 mode4
推到 2 ms 目标。

## 20. 已否决的“缩减 replay 后统一 Execute”结构

### 20.1 单 replay worker

首个候选只让一个 Scalar 回放动态 orchestration，其余 95 个 Scalar 等待
回放完成后再执行。该 Scalar 同时承担 Materialize、DAG history 查询、
TensorMap metadata 发布和 Build payload 发布，PA B256 虽然数值正确，
startup→FinalDrain 却达到 **163.880 ms**。这证明减少 replay 人口本身不等于
减少 Build 工作；只要“计划生成”和“完整 Build”仍绑在同一 Scalar，方案就把
原有并行 Build 退化成串行。

### 20.2 16-worker 通用 replay cohort

第二个候选按 96 个 worker 的物理编号均匀选择 16 个 replay worker，不读取
PA task kind、batch 或固定五任务形状。16 个 worker 仍通过逐 task Build
tournament 并行构建；全部 replay 完成后，由一个 sealer 扫描不可变 exec cell，
生成 AIC/AIV task-id 队列，最后由 96 个 Scalar 统一 Execute。

该候选通过本轮六类通用边界和 PA 数值 golden，但 PA B256
startup→FinalDrain 为 **13.1668 ms**，仍比保留流水的约 **5.15 ms** 基线
回退约 **155.7%**。根因不是 16 这个 PA 特例参数，而是全局 replay 闭合屏障
把可重叠的 Build/Execute 强制串成前后两段。

### 20.3 结论

两版代码均已完整撤回，不作为性能候选保留。后续若要缩减全核重复 replay，
必须同时满足：

1. orchestration 计划生产与完整 Build 解耦，计划生产者不得承担全部 Build；
2. Build 发布后立即允许同 engine executor 取得工作，不能等待全量计划封口；
3. 动态输出用稳定 `(producer_task_id, output_slot)` 引用跨过计划阶段，不能迫使
   计划生产者等待真实 TensorDesc；
4. TensorMap metadata 仍严格按 task-id writer 前驱发布，而 Fanin/Build/Execute
   继续允许并行；
5. 新协议必须通过第 17 章十二类、四模式共 48 个非 PA 边界组合。

因此下一条可行主线是复用已有通用 build-request/payload 协议，形成“轻量计划
发布 → 多 Scalar 动态 Build → Build 后即时 Execute”的流水，而不是继续微调
replay worker 数量。

## 21. SIMT request 延迟携带 shared output 引用

### 21.1 旧 publisher 等待不是计划发布的必要工作

mode3/mode4 虽然已经把完整 Build 交给 SIMT builder，但 request publisher 在
写 request 前仍逐个调用 `dist_cross_core_copy_existing_tensor()`。只要参数中有
`FdwicOutputRef`，publisher 就必须等待 producer 发布真实 `TensorDesc`，再把
整份 descriptor 复制进 request。结果是 replay 端仍沿动态依赖链等待，轻量
计划发布和完整 Build 没有真正解耦。

本阶段只改变通用 request wire contract，不读取 PA `TaskKind`、batch、固定五
task 次序、C/V 比例或 Tensor 形状：

1. request header 第 7 个 word 的低 32 位保存 `tensor_reference_mask`，高
   32 位继续固定为零；header 仍为 64 B；
2. 普通 existing Tensor 继续内联完整 descriptor，OUTPUT 继续内联
   `TensorCreateInfo`，shared output 只携带一个 64-bit
   `(producer_task_id, output_slot, flags, view_ndims)`；
3. source 声明的引用种类必须与 mask 逐 tensor 完全一致，引用不能标成
   OUTPUT，producer 必须严格小于 consumer task id，slot 和保留字段均校验；
4. SIMT builder 取得 request 后等待对应 output control 到 Published，校验
   task id、output count、build owner 和 descriptor owner，再从 producer 的
   task-indexed output cell 读取不可变 descriptor；
5. TensorMap writer 顺序、ordinary/DAG fanin 推导、execution payload、Execute
   owner 和 FinalDrain 合同均不变。

引用只占一个 word，因此混合 inline descriptor、OUTPUT create-info、shared
reference、scalar 和 explicit dependency 时，所有后续 offset 都由同一个
layout 函数计算，不再假设每个 tensor 固定占一份完整 descriptor。

### 21.2 首版局部状态过重及修正

首版为了避免重复解析，在每个 SIMT leader 的局部 request view 中保存了
32 个 descriptor GM 指针。代码能够编译，但真实 A5 上连 B1 都触发 AICPU
异常；零 Submit 也复现，说明问题发生在 builder VF 的局部状态形状，而不是
PA output 依赖本身。

该实现未保留。修正版只在 request view 中保存一个 shared output 数组基址和
一个“引用已验证”标志：先逐引用完成 Published/identity 校验，之后按紧凑 ref
现场计算 descriptor 地址。修正后零 Submit 不再出现设备异常，PA B1 在
ordinary/DAG 两模式均数值 PASS。这个过程也说明，SIMT 路径不能为了少几次
地址计算而无界增加每 lane 的局部数组。

### 21.3 通用正确性门槛

request protocol C++ 测试扩展到 11 项，新增覆盖：

- shared reference 的紧凑布局与 payload line 数；
- inline INPUT、OUTPUT、INOUT reference、scalar、explicit dependency 混合
  时的全部 offset；
- source/reference mask 不一致、OUTPUT reference、越界 mask、future/self
  producer 和非 plain ref 的 fail-closed 行为。

真实 A5 上，`simt_cross_core_ordinary` 与 `simt_cross_core_dag` 分别完整重跑
第 17.2 节十类边界，合计 20 个模式/场景组合全部数值 golden PASS。门槛包括
零 Submit、纯 AIC、纯 AIV、3/12/21/96 worker、17 个非 2 次幂 task、无 fresh
output、多 output、AIC→AIV→AIC deferred 链、重复 INOUT 和交替跨引擎 INOUT。
因此本阶段不是只对 PA 固定图成立。

### 21.4 PA B256 端到端结果

相同 shared TensorMap、PA B256、startup→FinalDrain `perf-clock` 口径，三个
独立 pytest 进程结果如下：

| 模式 | 本阶段三次结果 | 中位数 | 第 18/19 章中位数 | 改善 |
| ---- | -------------- | -----: | ----------------: | ---: |
| `simt_cross_core_ordinary` | 26.006 / 26.063 / 26.081 ms | 26.063 ms | 55.534 ms | 53.1% |
| `simt_cross_core_dag` | 8.722 / 8.739 / 8.745 ms | 8.739 ms | 22.637 ms | 61.4% |

对应 raw 目录：

- ordinary：`TestPagedAttentionUnroll_Case1_20260809_185304`、
  `TestPagedAttentionUnroll_Case1_20260809_190211`、
  `TestPagedAttentionUnroll_Case1_20260809_190314`；
- DAG：`TestPagedAttentionUnroll_Case1_20260809_185420`、
  `TestPagedAttentionUnroll_Case1_20260809_190014`、
  `TestPagedAttentionUnroll_Case1_20260809_190112`。

收益证明 publisher 的逐 output descriptor 等待是两种 SIMT simpler 路径的
通用串行链路。26.1/8.7 ms 仍未达到 2 ms 目标，下一阶段应继续减少全量 replay
和 Build 任务发放成本，而不是加入 PA task 特例。

## 22. SIMT DAG 改用按执行引擎划分的中央 Execute ticket

### 22.1 为什么只修改 mode4

第 21 章之后，mode4 已有 16 路持久 Builder，Build 供给可以和 replay
并行。旧 Execute 协议仍在每次 Submit 中让所有同引擎 Scalar 先经过逐 task
两级 tournament，再由唯一 owner 等待 Built。它同时保留了大量 owner
竞争和逐 Submit 等待。

本阶段分别实测了两个结构候选：

1. mode3/mode4 都在 replay 结束后集中 Execute：mode3 从约 26.1 ms 回退到
   38.894 ms，因为唯一 Builder 无法再和逐 Submit Execute 流水重叠；
2. 只让 mode4 使用集中 Execute：mode3 继续保留原流水，mode4 的多 Builder
   则能够持续供给 executor。

因此最终只对 mode4 生效。适用条件是“已有多路持久 Builder”这一 scheduler
拓扑，不是 PA task kind、batch、固定 C/V 序列或 Tensor 形状。

### 22.2 保留协议

mode4 在 runtime 既有 control storage 中复用两个独立的单调 cursor，分别供
AIC 和 AIV executor 使用：

1. 非 builder Scalar 完成自己的动态 replay 并参与 request seal；
2. 随即加入自身执行引擎的 ticket 流，与仍在 replay 的 worker 和持久
   Builder 流式并行；
3. AIC/AIV 两条流都按 task id 扫描，Builder 发布真实 engine 后，不匹配的
   一侧立即跳过，匹配的一侧才 claim immutable execution payload；
4. task cell 尚未发布或处于 Building 时只等待该 cell；已封口且 ticket
   超过真实 task 数时正常退出；
5. 恰好 2048 个 task 时，尾部 worker 可以取得 `capacity + k` 的退出票，但
   不得越界访问 task cell；允许的尾票上界只由 task capacity 和真实 worker
   数决定；
6. TensorMap writer 顺序、Build owner、fanin、payload 发布、ring slot 和
   FinalDrain 合同保持不变。

Submit 热路径不再把最早到达的 Scalar 固定成 Execute owner。Build owner 与
Execute owner 仍完全独立；只是 Execute 的领取从逐 task 全体竞争变为每个
引擎的一次单调 ticket。

### 22.3 通用正确性门槛

当前门槛已扩展为十二类，不依赖 PA：

- 1/4/32 block 下的 Build/Execute 到达偏斜；
- 2047 与 2048 个零参数 AIC/AIV 交替 task；
- AIC→AIV→AIC deferred 跨角色依赖；
- 单 task 多 output 分槽消费；
- 连续 32 个 INOUT writer；
- 零 Submit；
- 7 block、17 个质数 task 的纯 AIC、纯 AIV；
- 7 block 下 AIC/AIV 交替写同一 INOUT region。

四种 scheduler 均完整运行这十二类，合计 **48/48** 个真实 A5
模式/场景组合数值 golden PASS。2048 满容量用例还验证了并发尾票退出；它的
零参数 kernel 不访问 TensorMap 或 heap，避免把 task 状态容量与其他容量合同
混为一谈。本阶段不使用 A5Sim。

### 22.4 PA B256 性能

相同 shared TensorMap、真实 PA B256、startup→FinalDrain `perf-clock` 口径：

| 模式 | 本阶段结果 | 中位数/单样本 | 第 21 章对照 | 变化 |
| ---- | ---------- | ------------: | -----------: | ---: |
| `simt_cross_core_ordinary` | 25.894 ms | 单样本 | 26.063 ms | 同量级，未套用新协议 |
| `simt_cross_core_dag` | 8.078 / 7.976 / 8.179 ms | 8.078 ms | 8.739 ms | 改善约 7.6% |

mode4 三个独立进程都低于旧中位数。收益来自减少必要的返回型 Execute owner
竞争和解除 Execute 与单次 Submit 的绑定，不代表已经达到 2 ms 目标。

### 22.5 观察合同的独立问题

复核时发现四种新增 cross-core 路径的 shared full-swimlane 尚未接通完整
Submit 父记录，而 schema-v5 又固定要求每核 1280 条；SIMT builder 合法地
replay 0 条 Submit，也没有被该旧检查表达。mode3 在 level-1 与 level-4 都能
独立复现 `dropped=1`，证明它不是本阶段中央 ticket 引入的业务错误。

该问题必须作为单独的观察协议改造：让 Submit 数量和 builder/replay 角色按
真实 orchestration 动态闭合，并为 Execute ticket 增加准确 atomic site。在
观察协议完成前，不把半成品 schema 或误导性 atomic 名称混入本性能提交。

### 22.6 已撤销：单 replay 规划核

为验证“消除重复 replay”是否能继续降低 mode4 端到端时间，曾实现一个完全
通用的候选：只让 `block0/AIC` 顺序生成 request，保留 16 个 Builder，其余
79 个非 Builder Scalar 直接加入分引擎 Execute ticket。角色只由物理拓扑和
scheduler mode 决定，不读取 PA task kind、batch、C/V 序列或 Tensor 形状。

该候选先通过了本章十二类真实 A5 通用边界，包括零 Submit、2048 满容量、
跨引擎 INOUT、多 output 和 Build/Execute 到达偏斜；因此它在正确性上成立。
但 PA B256 三个独立进程结果为：

- 8.942 ms：`TestPagedAttentionUnroll_Case1_20260809_213014`；
- 8.964 ms：`TestPagedAttentionUnroll_Case1_20260809_213147`；
- 8.853 ms：`TestPagedAttentionUnroll_Case1_20260809_213233`。

中位数 8.942 ms，相对第 22.4 节保留基线 8.078 ms 回退约 10.7%。单规划核
虽然删除了 79 份重复 replay，却同时把真实参数构造与 request 生产变成一条
串行供给链，Builder 和 executor 更容易断粮。因此实现、host 角色合同和测试
夹具均已完整撤销，只保留本条负结果。

后续若继续消减 replay，必须同时保留并行 request/参数生产能力，例如研究
可证明完备的多 planner 分工；不能简单把 replay worker 数量降到 1，也不能
用 PA 固定五 task 或固定 task 类型为任务做静态分片。

### 22.7 已撤销：按 replay rank 确定 request publisher

为同时保留 80 路并行 replay 并删除逐 task tournament，曾按完整物理 block
拓扑为所有非 Builder worker 建立连续 replay rank，再令
`publisher_rank = task_id % replay_count`。映射覆盖 1/7/32 block 和动态
Builder 数，不依赖 PA task 语义；纯协议测试证明每个 task 恰好一个
publisher，真实 A5 的单 block、2048 满容量和 7 block 跨引擎 INOUT 也通过。

PA B256 五次为 8.013 / 8.051 / 8.095 / 8.068 / 8.173 ms，中位数
8.068 ms。相对第 22.4 节 8.078 ms 基线只改善约 0.13%，完全落在原样本
波动内；同时固定映射失去了 first-arriver 自动绕开慢 replay worker 的能力。
因此实现和新增协议测试均撤销，只保留结果。

这组数据说明，mode4 的 Build tournament 已不是当前端到端主要矛盾。后续应
先分别测清 request 发布完成前沿、Builder 完成前沿和 Execute 完成前沿，再
选择优化对象，不能继续凭 atomic 调用数量猜测收益。

## 23. SIMT DAG 查询 Tensor 区域只解析一次

### 23.1 重复工作的来源

`dist_simt_lookup_dag_fanins()` 原先在扫描每个历史 writer 时，都会为每个尚未
解析的 Tensor 重新读取 request 中的 `TensorDesc`，并重复计算 buffer 地址和
`[lo, hi)` 区域。对 32 个 Tensor、64 个历史 task 的最坏组合，这部分工作处在
`history × writer × tensor` 内层；同一个 immutable request 的查询区域实际上
从未改变。

保留实现先在 leader 本地为每个 Tensor 计算一次 `address/lo/hi`，历史扫描只
读取这三组紧凑数组进行重叠比较。manual dependency、writer 发布验证、producer
选择顺序、fanin 去重和错误收敛均保持原协议。实现不读取 PA task kind、batch、
固定五 task 次序或 Tensor 形状，因此是动态 DAG 查询的通用消减。

### 23.2 真实 A5 结果

相同 shared TensorMap、SIMT DAG、PA B256、startup 到 FinalDrain 的
`perf-clock` 口径，五个独立进程结果为：

- 3.27452 ms；
- 3.23714 ms；
- 3.26356 ms；
- 3.30605 ms；
- 3.40025 ms。

中位数为 **3.27452 ms**，相对第 22.4 节 8.078 ms 中位数改善约 **59.5%**。
这仍未达到 2 ms 目标，但已证明重复解析 immutable Tensor 区域是 mode4 的主要
通用热点之一。

改动后重新运行第 22.3 节十二类边界，四种 scheduler 合计 **48/48** 个真实
A5 模式/场景组合数值 golden PASS；本阶段未运行 A5Sim。
