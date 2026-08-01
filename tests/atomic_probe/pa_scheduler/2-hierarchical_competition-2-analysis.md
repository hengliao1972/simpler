# Claim 分层竞争复审：方案 A、C、D 在 A5 上的正确性与性能边界

> **状态：历史候选复审，结论继续有效。** 方案 A、C、D 均未进入当前
> shared PA 主路径；当前保留的是 per-task CAS Tournament，详见
> [`two_level_per_task_cas_tournament_summary.md`](two_level_per_task_cas_tournament_summary.md)。

## 1. 复审对象与结论

本文首先复审 `2-hierarchical_competition-2.md` 中的方案 D：在现有
flat `atomicMax` Claim 前先读取同一条全局 cursor，若已经满足
`global_cursor >= task_id`，则跳过 `atomicMax`。在同一事实基础上，
本文后半部分继续复审方案 A 的全局水位等待，以及方案 C 的 per-shard
task gate。

复审结论分为四部分：

1. **条件正确性基本成立**：在当前固定 PA 任务图上，如果预筛读取继续使用
   A5 的线性化原子读取，并且同一 cursor 的后继候选集合保持前缀闭合，D
   不会引入朴素分层方案的组内 loser 跑飞漏洞。
2. **当前性能论证不成立**：A5 的 `Ops::Load()` 实际是
   `atomicAdd(address, 0)`，仍是作用于同一热点 AtomicLine 的返回型 RMW。
   D 没有把迟到 loser 摘出原子总线，早到 contender 还会从一次 RMW
   增加到两次。因此不能把 D 称为“廉价 Load”“平均 atomic 小于 1”或
   “零风险、总应先加”的优化。
3. **A 的正确性有条件成立，但原性能模型不成立**：当前固定 PA 候选集合
   满足前缀闭合时，组内 loser 等到 `global >= task_id` 足以封住零 winner
   跑飞窗口；但每个 loser 的等待 Load 仍是同一 global AtomicLine 上的
   返回型 `atomicAdd(0)`。A 并没有结构性消除全局线总 RMW 流量。
4. **C 的正确性最直观，但性能和活性代价最重**：task gate 可以禁止后继
   task 在当前 task 产生 winner 前进入；代价是把所有候选重新汇聚到 gate
   AtomicLine，并引入“global 已发布、gate 尚未推进”的单 winner 交接窗口。

尤其需要撤回原文中的下列推论：

- D 的 Load 是 read-shared，基本不向热点线注入写流量；
- D 的平均 atomic 次数可以小于 1；
- D 最坏裁定成本仍近似 Flat 的 `N * t_atomic`；
- D 与分层 A 叠加后可以同时降低全局峰值和平均开销；
- D 可以不经独立 A5 取证直接作为第一落地项。

本文只分析协议与性能模型，不修改生产 Claim 实现。

## 2. 方案 D 的准确语义

原方案可以写成：

```text
FUNCTION ClaimD(task_id, cursor):
    observed = AtomicLoad(cursor)
    IF observed >= task_id:
        RETURN loser_without_fetch_max

    old = AtomicMax(cursor, task_id)
    RETURN old < task_id ? winner : loser
```

这里的 `AtomicLoad` 不能被含混地理解成普通 scalar load。正确性和性能
必须分别基于其真实实现讨论。

当前 CCEC 后端在 `same_core/ccec/ccec_ops.h` 中明确实现为：

```cpp
Load(address) == atomicAdd(address, 0)
```

CPU 后端也故意使用 `__atomic_fetch_add(address, 0)` 模拟同一逻辑，而不是
普通 load。原因是当前 PA 的共享控制字读取需要保持 A5 原子 RMW 语义。

因此，本文将两个可能实现区分为：

| 名称 | 预筛原语 | 当前状态 |
| ---- | -------- | -------- |
| `D-atomic` | `atomicAdd(address, 0)` | 正确性语义已存在，性能收益未证明 |
| `D-plain` | 普通 GM/scalar load | 可能更轻，但一致性与缓存正确性未证明 |

不能用 `D-atomic` 的正确性证明为 `D-plain` 背书，也不能用对普通 load 的
性能想象估算 `D-atomic`。

## 3. 当前 PA 下的正确性论证

### 3.1 至多一个 winner

预筛只删除已经确定失败的路径，真正 winner 仍由同一条 cursor 上的
`atomicMax` 线性化：

```text
old = atomicMax(cursor, task_id)
won = old < task_id
```

同一 `task_id` 最多只有一个调用观察到旧值小于 `task_id`。D 不增加新的
写入方式，因此“至多一个 winner”保持不变。

### 3.2 预筛命中为何可以判负

在一次运行中，Claim cursor 只通过 `atomicMax` 单调增加。若线性化读取到：

```text
observed >= task_id
```

那么随后再执行 `atomicMax(cursor, task_id)` 也只能得到
`old >= task_id`，结果必然为 loser。对这个调用而言，跳过 FetchMax 与
真正执行 FetchMax 的裁定结果一致。

### 3.3 无洞证明需要写明两个前提

原文把“观察到全局已覆盖 T”直接等同于“T 已有 winner”，这个表述还不够
严谨。cursor 值大于等于 T，只能直接证明数值水位覆盖了 T；要进一步证明
task T 没被更高 task 跳过，还依赖下面两个前提。

#### 前提一：候选核按 task 顺序回放

任何可能对同一 cursor 的后继 task `T'` 执行 Claim 的核，在到达 `T'`
之前，都必须已经处理完逻辑上更早的 T。

#### 前提二：同一 cursor 的候选集合前缀闭合

如果某个核有资格 Claim 后继 `T'`，它不能因为候选路由变化而完全绕开
前驱 T 的全局 cursor 协议。等价地说，同一 cursor 上的后继候选不能来自
一批从未经过前驱的全新核。

当前 standalone PA 满足这两个前提：

- Alloc 的同一 shard 使用固定 worker 子集；
- QK/PV 都由完整 AIC 集合参与并共享 cube cursor；
- SF/UP 都由完整 AIV 集合参与并共享 vector cursor；
- 每个 worker 按权威 task plan 单调回放 Submit。

在这些条件下，可以用“第一个可能越过 T 的全局操作”作归纳：该操作所属
核在到达后继前已经处理 T；处理 T 时，要么自己执行 `atomicMax(T)`，要么
已经线性化观察到 `global >= T`。由此不能凭空产生第一个跳过 T 的后继
写入。

### 3.4 动态候选集合下仍有零 winner 反例

若未来算子的 active mask、lane 路由或候选资格随 task 改变，D 的无洞
结论不再自动成立：

```text
global = pred(T)

task T:
  只有核 A 有资格 Claim；A 在发射原子前被延迟
  核 B 对 T 无资格，直接越过，不观察 global

task T' > T，且与 T 共用 cursor:
  B 在 T' 变成合法候选
  B 读取 global < T'，随后 atomicMax(T') 成功

A 恢复处理 T:
  A 读取 global == T' >= T，预筛判负

结果：T 没有 winner
```

这个反例并非 D 独有；现有 Flat Claim 的无洞证明同样依赖候选集合前缀
闭合。但它证明了 D 不能被描述为对任意动态任务图都“天然无洞”。后续泛化
必须把候选集合合同写入协议，或增加独立的逐 task 完成藩篱。

## 4. A5 性能模型中的决定性漏洞

### 4.1 Atomic Load 仍是热点 RMW

原文把预筛命中描述成“1 次廉价 read、0 次 atomic”，这与 A5 实际原语
不符。真实成本是：

| 结局 | Flat | `D-atomic` |
| ---- | ---- | ---------- |
| 迟到 loser | 1 次 `atomicMax` | 1 次 `atomicAdd(0)` |
| 早到 loser | 1 次 `atomicMax` | 1 次 `atomicAdd(0)` + 1 次 `atomicMax` |
| winner | 1 次 `atomicMax` | 1 次 `atomicAdd(0)` + 1 次 `atomicMax` |

因此：

- 每个合法候选仍至少向全局 AtomicLine 发射一次 RMW；
- `D-atomic` 只能减少 FetchMax 数，不能令原子总数小于候选数；
- 性能是否改善，只取决于 `atomicAdd(0)` 是否显著便宜于被替换掉的
  `atomicMax`，必须用同地址并发微基准取证；
- 不能把 FetchMax 次数下降直接解释为原子总线工作量同比下降。

### 4.2 返回值依赖拉长 winner 关键路径

预筛结果决定是否执行 `atomicMax`，所以 winner 必须等待
`atomicAdd(0)` 的返回值 ready 后才能发射第二条原子。两条操作不能并行，
也不能把第一条当作无需返回的 source-issue 指令。

Flat winner 的关键路径只有一次返回型 `atomicMax`；`D-atomic` winner 的
关键路径变成：

```text
atomicAdd(0) return-ready
  -> 分支判断
  -> atomicMax return-ready
```

即使大量迟到 loser 能跳过 FetchMax，也必须确认 winner 的额外串行延迟
没有拖长后续 Materialize/Register/Build 的启动时间。

### 4.3 同时到达时最坏可能接近两轮 N 路 RMW

若 N 个核近似同时到达，它们会先把 N 个 `atomicAdd(0)` 发往同一条全局
cursor。因为每个核必须等待 Load 返回，硬件队列可能先处理大量 add-zero，
这些调用都观察到旧水位；随后这些核再发射 N 个 `atomicMax`。

方向性的最坏上界应写成：

```text
N * t_atomic_add_zero + N * t_atomic_max
```

而不是原文的 `N * t_atomic_max`。这会把 D 从“峰值基本等于 Flat”改成
“峰值可能显著差于 Flat”。实际交错由 A5 原子流水决定，需要微基准而不能
靠模型认定。

### 4.4 迟到场景也没有消除全局热线访问

即使 winner 已经发布 T，后到核仍然要对同一条 cursor 发射
`atomicAdd(0)`，等待返回后才知道可以跳过。因此 D 只把某些热线操作从
Max 换成 Add-zero，没有像候选过滤那样真正让这些 actor 完全不访问原子
线。

这也是 D 与候选人口收窄的根本区别：候选过滤的 `not_attempted` actor
不发任何原子；D 的 `prefilter_skip` actor 仍发一条原子 RMW。两者的
性能收益不能互相类推。

## 5. D 与分层 A 叠加的问题

分层 A 的主要价值是先把 N 个候选分散到多条组 cursor，只有每组胜者才
触碰全局 cursor，从而把全局热点人口从 N 降到约 `N/K`。

如果按原文建议把 D 放在组轮之前：

```text
全部 N 个候选
  -> 对同一 global cursor 执行 atomicAdd(0)
  -> 未命中者再进入 K 大小的各组竞争
```

那么全部 N 个核在分组前已经重新汇聚到同一条全局 RMW 线。D+A 由此
重新制造 A 想消除的入口瓶颈：

- 全局线峰值人口仍是 N，不是 `N/K`；
- 组层只能优化第二阶段，无法收回入口 add-zero 的排队时间；
- winner 还要支付 global load、group max、global max 三段串行路径；
- “D+A 同时降低峰值和均值”在当前 A5 原语下没有理论依据。

因此，在没有真正的轻量、线性化 read-only 原语之前，不应把 D 放在组轮
之前。若未来存在经过验证的 read-only 预筛，也必须单独测量大量共享读者
对随后 global RMW 的失效/升级成本。

## 6. 普通 load 不能直接替换 Atomic Load

为了获得原文设想的低成本，有可能尝试把 `atomicAdd(0)` 换成普通 GM 或
scalar load。但此时正确性证明发生了实质变化。

### 6.1 stale-low 只影响性能

若普通 load 读到比真实 cursor 更小的旧值，本核只是多执行一次
`atomicMax`。最终 winner/loser 仍由原子返回值裁定，因此不会直接破坏
正确性。

### 6.2 stale-high 可以直接漏 task

若 cursor 在新一轮被重置或复用，而某核 cache 中仍保留上一轮较大的值，
普通 load 可能错误读到：

```text
stale_cursor >= current_task_id
```

本核会直接跳过 `atomicMax`。如果所有合法候选都命中 stale-high，本 task
就没有 winner。

此外，仓库已有 atomic/DCCI 门槛证明，普通 dirty cache line、atomic 更新
和整行 DCCI 的混用可能产生覆盖问题。Claim cursor 当前是 atomic-only
cache line，不能为了预筛性能未经验证地引入普通 cached 访问。

### 6.3 `D-plain` 的准入门槛

普通 load 只有在以下事项全部经 A5 门槛证明后才能进入 Claim：

1. 与同地址 `atomicMax` 处在可用的一致性域；
2. 运行初始化、重复运行和地址复用后不会读到 stale-high；
3. 不要求在 Claim 热路径增加 DCCI 或 DSB；
4. 不会生成可随后回写的 dirty cache line；
5. 编译器不会合并、提升或删除该同步读取；
6. AIC/AIV 都通过同地址并发读写压力测试。

在这些门槛完成前，`D-plain` 只能是隔离微基准候选，不能成为调度协议。

## 7. `attempted` 与观察合同存在冲突

原方案将预筛命中返回为：

```text
attempted = false
won = false
```

但当前 `ClaimOutcome::attempted` 和 `WorkerResult::claim_attempts` 用于验证固定
候选拓扑；它不是“是否执行了 FetchMax”的通用计数。当前 host 还会根据
worker、task 和 role 独立重建 compact raw 中的 attempted 状态，因为
32B `SharedSubmitClaimTraceRecord` 只保存时间端点与 winner bit。

若 D 动态把预筛命中改成 `attempted=false`：

- `claim_attempts == expected_claims` 的正确性断言会动态失败；
- compact raw 仍会把该 actor 重建为 attempted，和设备计数矛盾；
- `not_attempted` 会混入“角色/候选不合法”和“合法但预筛命中”两种语义；
- 不同运行的 Claim 数随时序变化，固定拓扑门槛失去证明力。

更合理的状态拆分是：

```text
eligible              // 角色与候选合同允许参与
prefilter_skipped     // 合法参与者已观察到全局覆盖
fetch_max_issued      // 实际发射了 ClaimMax
won                   // FetchMax 返回值裁定为 winner
```

其中 `attempted` 若继续代表合法 Claim 参与，就应在 prefilter skip 时保持
true；另用聚合计数或已有 atomic 记录统计 `prefilter_skipped` 与
`fetch_max_issued`。不能为了少扩一个观察字段而污染原有正确性口径。

另一个容易遗漏的实现细节是：非 Alloc 路由会在原子前填写
`outcome.function_id`。D 若在预筛命中后提前返回，必须把 loser 的
`function_id` 恢复为 `-1`，否则会破坏现有 loser outcome 不变量。

## 8. 建议的验证顺序

### 8.1 第一阶段：只验证 `D-atomic` 的协议

CPU 门槛应覆盖：

- 同 task 多核同时到达，仍恰好一个 winner；
- winner 已发布后，迟到核预筛命中；
- 预筛读到旧的小值后继续 FetchMax，结果仍正确；
- 后继核先跑、前序 winner 延迟的强制交错；
- 候选集合前缀闭合时无洞；
- 候选集合变化时复现本文的零 winner 反例，明确记录适用边界；
- prefilter loser 的 `function_id == -1`；
- eligible、prefilter、FetchMax 和 winner 四类计数闭合。

CPU 只验证协议，不得据其耗时判断 A5 性能。

### 8.2 第二阶段：A5 同地址原子微基准

先脱离完整 PA，在 AIC/AIV 分别测试：

1. Flat：每核一次 `atomicMax`；
2. D-atomic：每核 `atomicAdd(0)`，未命中再 `atomicMax`；
3. winner 已预先发布的全预筛命中场景；
4. 所有核同时起跑的全预筛未命中场景；
5. 逐级增加到达倾斜，扫描实际 `p_skip`；
6. N 取 8、24、32、64、96，覆盖组宽与当前角色人口。

必须同时报告：

- 整体完成墙钟时间；
- add-zero 与 max 的调用数和 return-ready 时间；
- winner 出现时间；
- 最后一个 loser 返回时间；
- 同一 AtomicLine 的 mean、p95、max；
- 所有最终值和 winner 数。

### 8.3 第三阶段：standalone PA 消融

只有微基准显示 `atomicAdd(0)` 明显便宜时，才进入完整 B256：

- 用 perf-clock 决定是否有端到端收益；
- 用泳道分别展示 `ClaimPrefilterLoad` 与 `ClaimMax`；
- 比较 Claim 父区间，而不是只比较 FetchMax 子区间；
- 比较总 atomic return-ready 时间和调用数；
- 保持 task winner、completion、fanin、依赖签名与计算结果完全一致；
- 独立报告 prefilter 命中率，不能用命中率替代性能收益。

`D+A` 必须在 D 和 A 各自单独验证后才允许进入组合实验，并重点检查 D
是否重新把全局 AtomicLine 变成 N 路入口热点。

## 9. 建议修正文档中的结论

方案 D 当前应被重新定位为：

> 一个在固定候选合同下保持 Flat Claim 裁定语义的实验性 FetchMax
> 预筛；使用 A5 当前原子读取时，它不会减少全局 RMW 访问人口，只可能把
> 部分 `atomicMax` 替换成 `atomicAdd(0)`。是否有收益完全依赖两种原子在
> 同地址竞争下的真实代价与到达倾斜，必须先做隔离微基准。

对应决策为：

- **不直接落 D**；
- **不把 D 标成零成本或零风险**；
- **不默认推荐 D+A**；
- 先完成 A5 `atomicAdd(0)`/`atomicMax` 同线并发微基准；
- 若 `D-atomic` 无收益，立即停止，不为降低 FetchMax 数而保留；
- 若希望尝试 `D-plain`，必须另立缓存一致性正确性目标，不与 D-atomic
  混为同一方案。

## 10. 与本轮 Alloc 候选实验的关系

2026-07-31 的 standalone shared B256 实验只把 Alloc 候选从 96 收窄到
24，Cube/Vector 继续保持 32/64：

- Claim 总数从 73,728 降到 55,296；
- perf-clock 10 次 mean/median 为 `1762.728/1757.651 us`；
- 原 96/32/64 基线 mean/median 为 `2305.596/2304.520 us`；
- 中位改善约 `546.870 us / 23.73%`。

该收益来自 72 个非 Alloc 候选完全不访问 Claim 原子线。D 与它不同：D
的 prefilter actor 仍要执行 `atomicAdd(0)`，所以不能依据这组 24/32/64
结果推断 D 也能获得相近收益。

这组候选实验本身改变 winner 资格，只用于理解人口与原子竞争的性能关系；
它也不能替代 D 的等候选 A/B 验证。

## 11. 方案 A、C 复审共同依据

原方案在性能建模时把等待 `Load` 视为 read-shared，并假定它基本不向
热点 RMW 线注入写流量。该假设不仅影响 D，也直接影响 A 和 C：

- A 的组内 loser 在 `global < task_id` 时循环 Load global cursor；
- C 的所有 contender 在进入 Claim 前循环 Load task gate；
- 两种 Load 的返回值都参与比较和控制流，不能编译成不等待返回值的
  source-issue 发布指令；
- 当前 CCEC `Ops::Load()` 是 `atomicAdd(address, 0)`，CPU 后端也用
  `fetch_add(0)` 保持相同协议含义。

因此，A/C 的等待路径都必须按“返回型 RMW + 分支依赖”建模。`PAUSE`
只能改变下一次轮询的发射间隔，不能把已经发出的 `atomicAdd(0)` 变成
普通共享读。

现有证据只能支持下面的有限结论：

1. 单 AIV dependent `atomicAdd` 微基准证明，消费返回值的完成等待会进入
   scalar busy，三个独立会话测得约 `182.9～271.0 ns/op`；该用例的 addend
   在 1/2 间变化，不是严格的 add-zero 对照。
2. 2026-07-30 Claim 预读实验直接使用返回型 `atomicAdd(0)` 做分支，
   B256 perf-clock 中位数回退 `2.916%`。该实验是“Load + 条件 FetchMax”
   对原 FetchMax，不是只切换返回值消费方式的同指令 A/B。
3. 历史 source-issue 的 Exchange/FetchAdd 可以很快结束源码发射包围区间，
   但它们不消费返回值，也不是 `atomicAdd(0)`，不能拿来给 A/C 的轮询估价。

在完成同地址、同竞争人口的 `atomicAdd(0)`/`atomicMax` 微基准前，下面的
RMW 数量重算只用于揭示结构性流量，不把每种 RMW 武断换算成相同纳秒数。

## 12. 方案 A：组内 loser 等全局水位

### 12.1 准确协议

方案 A 的关键路径是：

```text
old_group = AtomicMax(group_cursor, task_id)

if old_group >= task_id:
    while AtomicLoad(global_cursor) < task_id:
        pause
    return loser

old_global = AtomicMax(global_cursor, task_id)
return old_global < task_id ? winner : global_loser
```

它修复朴素两轮协议的方式不是等待“本组代表完成”，而是等待任意组已经
把全局水位推进到当前 task。global cursor 同时承担全局 winner 裁定和
loser release 两种语义。

### 12.2 当前固定 PA 拓扑下为何可以闭合零 winner

在当前 standalone PA 中，下列条件成立：

- 同一 cursor 的候选 worker 按权威 task plan 单调回放；
- Alloc 的每个 shard 使用固定 worker 子集；
- QK/PV 共用完整 AIC 候选集合；
- SF/UP 共用完整 AIV 候选集合；
- 任何候选 worker 离开 task T 前，要么自己完成
  `GlobalAtomicMax(T)`，要么已经原子观察到 `global >= T`。

因此可以对同一 shard 的 task 序列归纳：任何 worker 到达后继 T' 前，
已经证明全局覆盖 T；第一个可能发布 T' 的操作不可能凭空越过尚未覆盖的
T。第一个完成的 `GlobalAtomicMax(T)` 仍会观察到 `old < T`，产生唯一
winner。

这里的准确结论是：

> A 在当前固定、前缀闭合候选合同下没有发现朴素分层的零 winner 漏洞。

它不是对任意动态算子图都无条件正确。

### 12.3 动态候选集合仍可破坏无洞

若同一 cursor 上的后继候选来自一批没有经过当前 task 的新 worker，A 的
等待不会被执行：

```text
global = pred(T)

task T:
  只有 A 是候选；A 在 GroupMax(T) 或 GlobalMax(T) 前延迟
  B 不是 T 的候选，因此不执行 A 的 global wait

task T' > T，且与 T 共用 cursor:
  B 变成候选
  B 赢得 GroupMax(T') 和 GlobalMax(T')

A 恢复：
  GlobalMax(T) 看到 global == T'，判为 loser

结果：T 没有 winner
```

所以 A 至少要求：同一 cursor 的后继候选集合对前驱 task 保持前缀闭合，
或由没有候选的 task 显式发布等价的 resolved/skip 状态。active mask、动态
lane 路由和随 task 改变的候选裁剪都必须重新证明，不能沿用当前 PA 结论。

### 12.4 A5 上的 RMW 流量重算

设：

```text
N = 当前 task 的候选数
K = 组大小
G = ceil(N / K)
```

在每组都有候选、每个组内 loser 只检查一次 global 就看到已覆盖的理想
情况下，方案 A 仍至少包含：

| 操作 | 数量 | 所在线 |
| ---- | ---: | ------ |
| Group `atomicMax` | `N` | 分散到 G 条 group line |
| Global `atomicMax` | `G` | 同一 global line |
| loser `atomicAdd(0)` | `N-G` | 同一 global line |
| global-loser 回写 | 最多 `G-1` | 分散到 group line |

因此方向性下界为：

```text
total RMW >= N + G + (N-G) = 2N
包含最大回写时约为 2N + G - 1

global-line RMW >= G + (N-G) = N
```

以 Vector `N=64、K=8、G=8` 为例：

```text
64 次 group atomicMax
 8 次 global atomicMax
56 次 global atomicAdd(0)
最多 7 次 group 回写

合计约 128～135 次 RMW；global line 至少仍承受 64 次 RMW。
```

Flat 在同一 global line 上是 64 次 `atomicMax`。A 可能让每组最早返回的
代表更早发起 global Max，从而改善“首个 winner 出现时间”；但它没有在
结构上降低 global line 的总 RMW 数。原文的“global 只剩 G 个竞争者”只
统计了 Max，漏掉了 `N-G` 个返回型 add-zero。

如果组代表尚未完成，loser 会重复轮询，实际 global-line 流量为：

```text
G 次 global atomicMax
+ Σ 每个组内 loser 的 poll 次数
```

它可以显著大于 N。由此不能继续声称 A 的单 shard 吞吐结构性提升 K 倍，
也不能用 `1 + 1/K + 少量回写` 表示真实平均原子成本。

### 12.5 进展与调度副作用

A 还存在下列活性和调度风险：

1. **代表延迟被放大**：组内 winner 在 global Max 前延迟时，该组其他
   worker 已失去直接参与 global 裁定的资格，只能等待其他组或该代表推进。
   `G=1` 时，原 Flat 的 N 个推进者退化为一个代表。
2. **global poll 可能干扰发布者**：等待者和真正的 global Max 共享同一
   AtomicLine；是否公平排队、会不会增加 winner P99，必须由 A5 实测回答。
3. **等待期间不做有效工作**：原伪代码只 `PAUSE`，不会 EfDrain 本核已有
   winning slot，可能把原本的 replay 错峰变成集中空等，并改变 slot 压力、
   fanin 轮询和 winner 分布。
4. **代际初始化必须严格**：group cursor 若在重置或地址复用后保留 stale-high，
   全组可能直接判负并等待尚未有人发布的 global，形成永久停滞。
5. **观察必须分项**：只看 GlobalClaimMax 变短会遗漏 GroupClaimMax、
   global poll 和回写；端到端保留仍应由 perf-clock 裁决。

### 12.6 对方案 A 的判断

方案 A 不是已发现 correctness hole 的废案。它的准确定位是：

> 在当前固定 PA 候选合同下，A 是一个正确性可以闭合、但 A5 性能模型尚未
> 成立的峰值候选。它可能提前产生 winner，却可能让 loser 和 global-line
> 总流量明显变重。

如果继续研究分层，优先考虑把 ELECTED 与 RESOLVED 分开：每组代表完成
global Claim 后发布本组 `group_resolved_max`，组内 loser 只轮询本组
resolved line。这样 add-zero 轮询被分散到 G 条线，不再全部回灌 global
热点；代价是某组代表延迟会单独阻塞本组，并增加每组一次 resolved 发布。
该变体必须先复用 `hierarchical_competition_analysis.md` 中的 CPU 乱序证明，
再进入 A5。

## 13. 方案 C：每 shard 的 task gate

### 13.1 准确协议与正确性强度

方案 C 在每个 shard 增加 `open_task`：只有 gate 等于当前 task 时才允许
进入组内和全局 Claim；唯一 global winner 再把 gate 推到下一合法 task。

只要下面的条件全部成立，C 能直接保证后继 task 不会在当前 task 产生
winner 前进入：

- gate 初值对应 shard 的首个合法 task；
- `NextTaskIdOnShard(T)` 精确给出真实后继；
- 每个 gate task 至少有一个最终能进展的合法候选；
- global winner 在成功后最终能够发布下一 gate；
- gate 的初始化、复用和发布具有经过验证的原子语义。

C 不像 A 那样要求后继 worker 自己曾经参与前驱 task；新候选会被 gate
挡住。因此它对动态候选集合更强，但把风险转化成“每个逻辑 task 必须有人
推进 gate”的稠密序列合同。

### 13.2 gate Load 在 A5 上是新的全局 RMW 热点

原伪代码为：

```text
while Load(gate) < task_id:
    pause
if Load(gate) > task_id:
    return loser
```

当 gate 正好等于 task_id 时，每个 contender 通常先执行一次 while 条件
Load，再执行一次 if 条件 Load。两次返回值都参与比较，因此在 A5 上都是
返回型 `atomicAdd(0)`。第二次 Load 可以在具体实现中通过保存第一次观察值
消减，但即使只保留一次，所有候选仍会访问同一 gate AtomicLine。

在“64 个候选都于 `gate == T` 时进入、两次 gate Load 均执行、每组一名
代表”的方向性场景中：

| 操作 | 数量 |
| ---- | ---: |
| gate `atomicAdd(0)` | 约 `2N = 128` |
| Group `atomicMax` | `N = 64` |
| Global `atomicMax` | `G = 8` |
| global-loser 回写 | 最多 `G-1 = 7` |
| winner 推进 gate | 1 |

合计约 208 次 RMW，且尚未计算后继 task 提前到达后的重复 gate poll。实际
执行可能因为 gate 在两次读取之间推进而少做部分工作，也可能因为等待而远
高于该数；该计算的作用是说明原表中的“winner 三次 atomic、组 loser 一次
atomic”漏掉了 gate 的返回型 RMW，不能作为 A5 成本模型。

gate 会形成新的集中点：

- 所有候选先碰 gate，组线尚未起到分流作用；
- winner 的 gate 发布与后继核的 gate poll 作用于同一 AtomicLine；
- gate 打开后，等待者集中涌入 group line，削弱原 PA 自然到达倾斜，形成
  新一轮同步浪涌；
- 原文假设的 read-shared gate 不存在时，不能认定 C 仍比 Flat 快。

### 13.3 global 发布与 gate 发布之间的单 owner 窗口

C 把一个 task 的放行拆成两次独立原子：

```text
old = AtomicMax(global, T)       // 唯一 winner 已经产生
AtomicMax(gate, Next(T))         // 只有该 winner 推门
```

如果 winner 在两条操作之间延迟：

- global 已经覆盖 T，其他 global contender 都只能判负；
- gate 仍停在 T，后继 task 全部等待；
- 原伪代码中的 global loser 不会帮助推进 gate；
- 整个 shard 的进展依赖这一个 winner 恢复执行。

这不会凭空制造第二个 winner，但会产生明显的队头阻塞和活性脆弱点。若
winner 永久失败，shard 永久停滞。增加 helping 可以缓解：任何核在确认
`global >= T` 后都尝试推进 gate；但这会增加 gate RMW、状态判定和代际
证明，已经不是原 C 的简单协议。

### 13.4 “task 窗口完全串行”并不精确

C 真正保证的是：

> 后继 task 只有在当前 task 已经产生 global winner 后才能进入。

它不保证所有当前 task contender 都已退出。某些核可能已经在 gate==T 时
通过门，随后停在 GroupMax 或 GlobalMax；winner 推门后，T' 的新 contender
可以进入，而这些迟到的 T contender 仍在完成自己的 loser 路径。

所以 C 是“按 winner 发布串行化 task 入口”，不是严格的全窗口 barrier。
这不破坏无洞正确性，却说明原文两处表述都需要修正：

- 不能声称任意时刻绝对只有一个 task 的 Claim 指令在执行；
- 也不能直接按完全无重叠的 `(K + N/K) * t_atomic` 推导吞吐。

如果真的等待所有 T contender 收敛后才推门，还需要 arrival/completion
计数或 release 状态，成本会进一步接近每 task barrier。

### 13.5 稀疏 task、动态路由与复用风险

C 的另一组缺陷来自 gate 的精确序列依赖：

1. **无人 Attempt**：某个逻辑 task 因 active mask、角色过滤或条件执行没有
   候选，gate 永远不会推进。
2. **后继计算错误**：`task_id + S` 只有在该 kind/shard 的真实 task 集合
   连续时才成立；PA 中 QK/PV、SF/UP 的交错以及未来通用算子都必须使用
   权威实际后继，而不能根据整数步长猜测。
3. **地址复用或重置错误**：stale-high 会让当前 task 被所有核直接跳过；
   stale-low 会让后继永久等待。gate 必须拥有显式 generation/reset 合同。
4. **异常路径**：global winner 在 Materialize 之前就已存在，但若 gate 推进
   与 fatal 处理顺序不清晰，可能放行后继后才报告协议失败；需要单独定义
   Claim 成功、gate 发布和后续业务失败的关系。
5. **负载集中**：即使不同 task 的 group/global cursor 已分片，同一 shard
   的全部入口仍被 gate 串行协调，削弱 shared scheduler 允许无依赖 task
   并行推进的架构目标。

### 13.6 对方案 C 的判断

方案 C 的准确定位是：

> 一个正确性易证明、适合构造无洞对照的诊断协议；它不是当前 A5 上有说服力
> 的性能候选。gate 的返回型 add-zero 热点、单 winner 二次发布窗口、精确
> task 序列前提和同步浪涌，都可能超过分层减少的 GlobalMax 收益。

因此不建议为了性能直接实现 C。若需要它验证 A/B 的无洞语义，应把它作为
CPU/小规模 A5 correctness oracle，不能把其结果当成最终调度架构。

## 14. A、C、D 的统一结论

| 方案 | 当前正确性边界 | A5 上的决定性缺陷 | 当前定位 |
| ---- | -------------- | ----------------- | -------- |
| A | 固定、前缀闭合候选下可闭合 | loser 全部以 `atomicAdd(0)` 回到 global 热线 | 修改等待结构后再验证 |
| C | task 序列准确且每 task 有推进者时可闭合 | gate 成为新热点，且依赖 winner 二次发布 | 仅作正确性对照 |
| D | 与 Flat 相同的候选前提下可闭合 | 每核至少一次 global RMW，winner 串行两次 RMW | 微基准未证明前不落地 |

从支持通用算子的角度，三者都不能直接按原文推荐顺序进入正式 Claim：

1. D 已有等候选端到端回退记录，先停止；
2. C 的架构代价最大，不作为性能主线；
3. A 若继续，先改成分布式 per-group resolved 等待，避免所有 loser 轮询
   global；
4. CPU 必须构造“代表在组选举后延迟”的确定性交错，证明每 task 恰好一个
   winner；
5. A5 先做同线并发微基准，再做 B256 perf-clock；泳道只用于解释 GroupMax、
   GlobalMax、ResolvedPoll 和 winner/loser 时间迁移；
6. 任一版本都必须保持候选人口、winner 数、依赖签名、completion、fanin、
   heap 和 TensorMap 终态不变。

最终判断不能只看 GlobalClaimMax 是否缩短。正确的裁决顺序仍是：协议闭合、
CPU 确定性交错、A5 正确性、无观察 perf-clock、最后才用完整泳道解释收益
来自哪里。
