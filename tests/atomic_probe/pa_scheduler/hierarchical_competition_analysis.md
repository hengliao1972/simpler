# ClaimMax 分层竞争正确性分析

> **状态：历史反例，结论继续有效。** 本文否定的是 group/global
> 高水位两轮协议；当前 per-task CAS Tournament 不复用该高水位协议，
> 后继 task 也不会覆盖前驱 task 的仲裁状态。

## 1. 结论

[`hierarchical_competition.md`](hierarchical_competition.md) 中当前的两轮
Claim 伪代码能够证明：

- 同一 task **至多一个 winner**；
- 组级和全局 cursor 都单调不减。

但它还不能证明调度协议真正需要的：

- 同一 task **至少一个 winner**；
- 因而同一 task **恰好一个 winner**。

问题不在 `atomicMax` 的原子性，而在两轮之间缺少“全局裁定已完成”的
发布和等待。组内 loser 看到的只是“本组已经选出代表”，却把它解释成
“本 task 已完成全局 Claim”，随后继续回放后继 task。后继 task 因此
可能先把全局高水位越过当前 task，使当前 task 永久没有 winner。

在补齐这个状态之前，当前伪代码不能直接进入 standalone 或真实路径。

## 2. 历史背景与讨论范围

本分析对应的是 private TensorMap 阶段曾经讨论过的 Claim 优化方向：

1. 先在 startup/final 同步阶段验证树形汇合；
2. 再研究能否把 Submit 中间的 Claim `FetchMax` 也改成
   “组内竞争、组代表竞争全局”。

它不是后来 shared TensorMap 中按 `task_id` 增加 cursor 分片数的优化。
两者处理的是不同问题：

- task-id 分片缓解不同 task 对同一 AtomicLine 的冲突；
- 分层竞争试图缓解同一 task 的全部候选核对同一全局 AtomicLine 的冲突。

此前 final 分层能够成立，是因为它具有完整的向上汇合和向下 release：

```text
worker -> leaf arrival -> root arrival
worker <- leaf release <- root release
```

任何 worker 都不能只因本组已有代表就越过 final。相关实验和协议见
[`PA-atomic情况分析.md`](PA-atomic情况分析.md) 的
“只改 final 的分层全局汇合实验”。

Claim 与 final 不同。Claim loser 原本允许立即进入后续 task，这正是
当前漏洞出现的条件。

## 3. 扁平 Claim 的完整正确性来自哪里

现有扁平 Claim 可抽象为：

```text
old = FetchMax(global_cursor[shard], task_id)
won = old < task_id
```

`FetchMax` 本身直接提供“至多一个 winner”：

- 同一全局原子变量上的调用具有线性化顺序；
- 只有第一个把 cursor 从 `< task_id` 推进到 `>= task_id` 的调用，
  才能观察到 `old < task_id`；
- 后续调用观察到的旧值都不小于 `task_id`。

“至少一个 winner”还依赖调度回放顺序，而不是由 `FetchMax` 单独提供。
对于同一 shard 上的当前 task `t` 和后继 task `t+S`：

1. 每个 eligible worker 都按 task 顺序回放；
2. worker 在进入 `t+S` 前，必须完成自己对 `t` 的全局 Claim；
3. 因此任何 `GlobalFetchMax(t+S)` 之前，必然已经有
   `GlobalFetchMax(t)` 完成；
4. 第一个完成的 `GlobalFetchMax(t)` 观察到的旧值小于 `t`，
   从而产生 task `t` 的 winner。

扁平协议实际依赖以下不变量：

> 任何 worker 都不能绕过当前 task 的全局 Claim，进入同一全局
> cursor 链上的后继 task。

这使全局 cursor 虽然是允许跳跃的高水位，实际执行中却不会在没有
task `t` winner 的情况下从 `<t` 跳到 `>t`。

## 4. 两轮协议混淆了两个不同状态

当前分层伪代码的第一轮是：

```text
old_group = FetchMax(group_cursor[shard][group], task_id)
if old_group >= task_id:
    return loser
```

`group_cursor >= task_id` 只能证明：

```text
ELECTED(task_id)
本组已经选出一个负责全局 Claim 的代表
```

真正允许本组其他 worker 离开的条件应当是：

```text
RESOLVED(task_id)
该代表已经完成全局 Claim，task_id 的全局裁定已经发生
```

两个状态之间存在时间窗口：

```text
组内代表产生
    |
    | 代表尚未完成全局 FetchMax
    v
全局裁定完成
```

当前单一 `group_cursor` 只记录 task 高水位，不能表达
`ELECTED` 与 `RESOLVED` 的区别。组内 loser 在 `ELECTED` 状态便被
允许返回，破坏了扁平协议的“不可绕过全局 Claim”不变量。

## 5. 两核最小反例

只需要一个组、两个 worker 就能构造反例。初始状态为：

```text
group_cursor  = t-S
global_cursor = t-S
```

执行顺序如下：

| 顺序 | worker | 操作 | group | global | 结果 |
| ---: | ------ | ---- | ----: | -----: | ---- |
| 1 | A | `GroupFetchMax(t)` | `t` | `t-S` | A 成为组代表 |
| 2 | A | 在全局 Claim 前延迟 | `t` | `t-S` | `t` 尚无全局 winner |
| 3 | B | `GroupFetchMax(t)` | `t` | `t-S` | B 组内失败并返回 |
| 4 | B | `GroupFetchMax(t+S)` | `t+S` | `t-S` | B 成为新组代表 |
| 5 | B | `GlobalFetchMax(t+S)` | `t+S` | `t+S` | B 赢得 `t+S` |
| 6 | A | `GlobalFetchMax(t)` | `t+S` | `t+S` | A 全局失败 |

最终结果为：

```text
task t   : 0 个 winner
task t+S : 1 个 winner
```

错误无法由后续 replay 自动修复：

- `global_cursor` 已经大于 `t`；
- 任何迟到的 `GlobalFetchMax(t)` 都只会失败；
- task `t` 不会进入 Materialize、Register 和 Build；
- 依赖 task `t` 的任务可能一直不 ready；
- 最终可能表现为 winning slot 积压、FinalDrain 无法收敛或语义校验失败。

在多组场景中，只需让各组关于 task `t` 的代表都在全局 Claim 前延迟，
再让任意组的 follower 先把后继 task 推进到全局，即可形成同样结果。
该时序不一定经常发生，但正确性合同必须覆盖所有合法的核间交错。

## 6. 为什么“下一条指令就是全局 FetchMax”不构成证明

组代表在源码中紧接着执行全局 `FetchMax`，只能证明本核程序顺序：

```text
GroupFetchMax(t)
program-order-before
GlobalFetchMax(t)
```

它不能建立下面的跨核关系：

```text
A 的 GlobalFetchMax(t) 完成
happens-before
B 离开 task t
```

即使 AICore scalar 不被操作系统抢占，以下因素仍可产生核间交错：

- 组级和全局 atomic 位于不同 cache line；
- 全局 AtomicLine 正被其他组争用；
- atomic 请求在互连或原子单元中排队；
- 不同核的请求以不同顺序到达全局线。

因此不能依赖“代表通常会先完成”作为正确性条件。

以下手段也不能单独补上该关系：

- 地址依赖只约束本核对返回值的消费；
- 在代表侧增加 DSB 只约束代表自身已有操作；
- 编译器屏障不能让 follower 等待代表未来的全局操作；
- 指望两个不同地址上的 atomic 自然形成全局先后顺序没有协议依据。

需要由代表在全局 Claim 完成后显式发布状态，再由 follower 消费该状态。

## 7. 现有回写为什么不能修复

当前方案让全局 loser 执行：

```text
FetchMax(group_cursor, global_watermark)
```

这项回写解决的是组水位落后：

- 全局 cursor 已经推进到更高 task；
- 本组 cursor 仍停留在较低 task；
- 后续 worker 可能重复成为组代表并冲击全局线。

它没有解决 `ELECTED` 与 `RESOLVED` 的混淆：

- 组内 loser 在回写发生前已经返回；
- 全局 winner 路径没有独立的 resolved 发布；
- follower 也没有等待任何 resolved 状态；
- 在反例中，迟到代表只会把错误的更高水位再次回写到组内。

因此该回写是减少重复全局流量的性能动作，不是“task 已完成全局裁定”
的充分证明。

## 8. 为什么 final 分层可以成立

final 分层不会出现上述越序，原因不是树形结构本身更特殊，而是它保留了
完整屏障语义：

1. worker 到达 leaf；
2. leaf 代表在本组全部到达后推进 root；
3. root 满足全局条件后发布各 leaf release；
4. 每个 worker 观察到 leaf release 后才能退出。

其中 leaf arrival 表示“本组正在汇合”，leaf release 才表示
“全局条件已经成立”。二者没有混用。

此外，final 每轮只执行一次，不存在下一代 task 越过当前 final；等待期间
还能执行 `FinalDrain`，用实际工作覆盖部分等待成本。

若把同样语义用于 Claim，就必须为每个 task 提供一次可复用的组内
resolve/release。它不再只是两次 `FetchMax`，而是每个 task 上的一次微型
可复用屏障。

## 9. 最直接的正确协议

最容易验证的修正是把组选举水位和全局裁定完成水位分开：

```text
group_elected_max[shard][group]
group_resolved_max[shard][group]
```

伪代码如下：

```text
old_group = FetchMax(group_elected_max, task_id)

if old_group < task_id:
    old_global = FetchMax(global_cursor, task_id)
    won = old_global < task_id

    // 无论全局赢或输，都在全局操作返回后发布本组裁定完成。
    AtomicPublish(group_resolved_max, task_id)
    return {attempted: true, won: won}

while AtomicLoad(group_resolved_max) < task_id:
    if FatalObserved():
        return failure

return {attempted: true, won: false}
```

该协议要求：

- 所有组代表都在全局 Claim 返回后发布 resolved；
- 全局 winner 所在组也不能省略 resolved 发布；
- 组内 loser 在 `resolved >= task_id` 前不能进入同 shard 后继 task；
- resolved 状态必须具有经过验证的跨核发布和消费语义；
- 代表失败时必须有 fatal 终止或明确的接管协议，不能永久静默等待。

### 9.1 “至少一个 winner”的证明草图

假设存在最早的后继全局操作 `GlobalFetchMax(t+S)`。

发起该操作的 worker 在离开 task `t` 前，必须满足本组
`group_resolved_max >= t`。该 resolved 值只能由一个已经完成
`GlobalFetchMax(t)` 的组代表发布。

因此：

```text
某次 GlobalFetchMax(t)
happens-before
最早的 GlobalFetchMax(t+S)
```

在任何后继 task 把全局 cursor 推过 `t` 之前，至少已经有一次
`GlobalFetchMax(t)` 完成。第一条这样的操作观察到旧值小于 `t`，
所以 task `t` 至少有一个 winner。

全局 `FetchMax` 的线性化又保证至多一个 winner，二者合起来得到
恰好一个 winner。

## 10. 修正后的性能代价

原始未闭合方案的理想路径为：

- 组内 loser：一次组内 `FetchMax` 后立即返回；
- 全局 winner：组内和全局各一次 `FetchMax`；
- 全局 loser：再增加一次组水位回写。

增加 resolved 后，至少还会出现：

- 每个组代表一次 resolved 发布；
- 每个组内 loser 至少一次 resolved load；
- 代表较慢时，组内 loser 会重复 poll；
- 同组 worker 在 resolved 前不能继续向前回放。

因此正确版本实际是在做如下交换：

> 用低竞争的组内 atomic、组内轮询和部分 worker 等待，换取全局
> AtomicLine 上竞争者数量下降。

这与 final 屏障不同：Claim loser 原本很轻，等待 resolved 会减少 worker
间的自然错峰和 run-ahead，也可能改变 fanin 轮询、winner 分布和
winning-slot 压力。

此前 startup 分层在没有可重叠工作的情况下明显回退，也说明“减少全局
cache-line 竞争”并不自动等于端到端更快。Claim 正确版本必须单独测量，
不能沿用未闭合版本的 `1 + 1/K + 少量回写` 成本模型。

## 11. 其他可选协议

### 11.1 单控制字编码 pending/resolved

可尝试在一个 64 位组状态中编码：

```text
(task_id, PENDING)
(task_id, RESOLVED)
```

它能减少独立 cache line，但状态转换需要 CAS、代际和越序证明。未来 task
不得在当前 task 仍为 `PENDING` 时覆盖状态。相比双水位方案，它更紧凑，
但首版正确性验证更复杂。

### 11.2 每 task 独立的全局 Claim 单元

如果全局 ownership 也改为 per-task 状态，后继 task 不会覆盖当前 task，
迟到的 task `t` 代表仍可在自己的单元中获胜。

该方案会失去原类型高水位 cursor 提供的 run-ahead 约束。历史 per-task
Claim 实验虽然降低了单条 ClaimMax 延迟，却伴随 fanin 轮询增加和
perf-clock 回退。因此它只能作为独立协议重新评估，不能视作简单修复。

### 11.3 组内 loser 回退到全局 Claim

组内 loser 若发现 resolved 尚未完成，也可以回退执行原全局 Claim。
这容易恢复正确性，却会在竞争最强时重新让大量 worker 冲击全局线，
可能抵消分层目的。它适合作为诊断对照，不宜未经测量作为最终协议。

## 12. 必须补充的验证

### 12.1 CPU 确定性乱序测试

测试必须能够在两轮之间主动暂停组代表，而不是依赖概率复现：

1. A 赢得 `GroupFetchMax(t)` 后停在全局 Claim 前；
2. B 对 `t` 组内失败；
3. 尝试让 B 进入同 shard 后继 task；
4. 原协议应稳定复现 task `t` 零 winner；
5. 修正协议中 B 必须停在 resolved 等待；
6. 释放 A 后，断言 `t` 和后继 task 各恰好一个 winner。

还要覆盖：

- 多组代表同时延迟；
- 全局 winner 组和全局 loser 组的 resolved 发布；
- 迟到 follower 观察到更高 resolved 水位；
- fatal 发生在组选举后、全局 Claim 前；
- 同 shard 多代连续推进；
- 不同 shard 互不阻塞。

### 12.2 CCEC/A5 正确性门槛

完成 CPU 协议证明后，A5 至少需要核对：

- 每 task 恰好一个 winner；
- winner、loser、not-attempted 总数闭合；
- 各组 elected/resolved 最终水位；
- Materialize/Register/Build/Kernel 次数不变；
- fanin、completion、heap 和 TensorMap 终态不变；
- fatal、超时和泳道 dropped record 均为零。

### 12.3 性能证据

正确性闭合后再比较：

- flat Claim；
- 原始未闭合层级版本仅作为性能上界参考；
- 带 resolved 的正确层级版本。

观察项至少包括：

- GroupClaimMax、GlobalClaimMax、GroupResolvedPublish；
- resolved poll 次数和等待时间；
- Claim 外层非 atomic 时间；
- true-loser 完整控制时间；
- fanin load、winning-slot 占用和 FinalDrain；
- 无泳道 perf-clock 端到端时间。

局部 atomic 变快不能替代完整调度性能结论。

## 13. 当前决策

当前两轮伪代码可继续作为“降低同 task 全局线争用”的性能构想，但不能
标记为正确性已经闭合。继续工作的顺序应为：

1. 在设计文档中显式区分 `ELECTED` 与 `RESOLVED`；
2. 用确定性乱序测试复现原协议零 winner；
3. 实现最小 resolved 协议并先完成 CPU 证明；
4. 再构建 CCEC 并运行 A5 正确性；
5. 最后用无观察 perf-clock 判断分层收益能否覆盖新增等待。

在完成前四步之前，不应依据理论上的全局竞争者下降直接修改正式 Claim。
