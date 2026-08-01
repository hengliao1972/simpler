# 当前候选方案总结：两级 Per-Task CAS Tournament + `deps_prepared` 顺序提交链

> **状态：** 已在 standalone shared PA 落地并保留
>
> **首次落地：** `a57f5a57`；Alloc 全候选与最终 G8 形态：`9c8d1baa`
>
> **当前合同：** Alloc/AIC/AIV 候选为 `96/32/64`，local 组数为 `8/6/8`
>
> **目标：** 在保留全部合法候选和动态负载均衡能力的前提下，消除 Flat Claim 的同地址高并发热点，同时保持 shared TensorMap 的严格 task-id 顺序插入。
>
> 当前明确不纳入范围：per-task 节点的 ABA / generation / 长窗口复用设计。
>
> 实现后的完整性能与回归证据见 [`../perf_opt_record.md`](../perf_opt_record.md) 第 14.2～14.7 节。

---

## 1. 问题定义

当前 Flat Claim 使用单调 cursor：

```cpp
old = atomicMax(cursor, task_id);
won = old < task_id;
```

同一 task 的全部合法候选同时访问同一个 GM atomic word。Claim 必须消费原子返回值，因此 winner 和 loser 都处于 return-ready 路径。候选人口增大时，同地址 RMW 近似串行排队。

当前讨论采用的硬件模型是：

```text
单次 return-ready atomic 延迟约为 α = 160 ns
同地址 N 路并发近似线性串行：约 α × N
不同地址之间存在可利用的并行度
```

现有 A5 探针还表明，在相同 Flat 或两级拓扑下：

- CAS；
- Exchange；
- FetchMax；

三种原语的 return-ready 时间几乎相同，差异约在 `0.1%` 量级。由此，优化重点不是替换 atomic 指令，而是改变竞争拓扑。

另一方面，shared TensorMap 要求 ordinary / symbol metadata 严格按照 task id 插入：

```text
task 0 metadata
  happens-before
task 1 metadata
  happens-before
task 2 metadata
  ...
```

因此必须把两个职责明确分开：

1. **Owner election**：task N 由哪个合法候选负责；
2. **Metadata commit order**：task N 何时允许把 TensorMap metadata 写入共享历史。

---

## 2. 当前架构决策

采用：

```text
每 task 独立的两级 CAS Tournament
        ↓
产生唯一 owner
        ↓
只有 owner 进入 Register / metadata prepare
        ↓
等待 deps_prepared[N-1]
        ↓
发布 task N 的 ordinary / symbol / TensorMap metadata
        ↓
提交 deps_prepared[N]
```

职责划分如下：

| 机制 | 唯一职责 |
| ---- | -------- |
| Per-task local CAS nodes | 每个仲裁组产生至多一个代表 |
| Per-task root CAS node | 从所有组代表中产生唯一 task owner |
| `deps_prepared` 前驱链 | 保证 shared TensorMap metadata 严格按 task id 提交 |
| 现有 metadata publication seam | 保证 ordinary payload 在提交完成字之前对后继可见 |
| Completion / task flag | 表示 task kernel 或逻辑任务执行完成；不与 `deps_prepared` 混用 |

核心原则：

```text
owner elected != metadata committed != task completed
```

---

## 3. Per-task Tournament 状态

对 task `T`，逻辑状态为：

```text
local[T][0 .. G-1] = -1
root[T]             = -1
```

节点只承载一次性的仲裁状态，不承载普通 payload：

```text
local[group]: -1 → task_id
root:         -1 → task_id
```

CAS 返回值决定调用者是否成功：

- `local` CAS 成功：成为本组代表；
- `local` CAS 失败：普通 loser，立即退出本 task 的 Claim；
- `root` CAS 成功：成为 task 的唯一 owner；
- `root` CAS 失败：组代表成为 global loser，立即退出。

节点不需要存 owner 的复杂元数据。owner 身份由“哪一个调用者观察到 CAS 成功”确定。

---

## 4. Claim 伪代码

```cpp
ClaimOutcome ClaimTwoLevel(
    int32_t task_id,
    int32_t arbitration_group,
    AtomicNode* local_node,
    AtomicNode* root_node) {

    // 第一轮：组内选代表
    const int32_t local_old =
        CAS(&local_node[arbitration_group], -1, task_id);

    if (local_old != -1) {
        return {
            .eligible = true,
            .local_attempted = true,
            .local_won = false,
            .root_attempted = false,
            .won = false,
        };
    }

    // 第二轮：组代表竞争唯一 owner
    const int32_t root_old =
        CAS(root_node, -1, task_id);

    if (root_old != -1) {
        return {
            .eligible = true,
            .local_attempted = true,
            .local_won = true,
            .root_attempted = true,
            .won = false,
        };
    }

    return {
        .eligible = true,
        .local_attempted = true,
        .local_won = true,
        .root_attempted = true,
        .won = true,
    };
}
```

该 Claim 不轮询：

- 不轮询 global cursor；
- 不轮询 group resolved；
- 不等待 `deps_prepared`；
- loser 在本层失败后立即返回并继续 replay。

只有 root winner 进入后续 Register / commit 路径。

---

## 5. Owner 到 metadata commit 的路径

owner 被选出后，执行：

```text
owner(T)
    ↓
可以提前准备不产生共享可见副作用的 Materialize / writer delta
    ↓
等待 deps_prepared[T-1] == T-1
    ↓
发布 task T 的 ordinary / symbol / TensorMap metadata
    ↓
确保 metadata 对后继观察者可见
    ↓
CAS deps_prepared[T]: -1 → T
    ↓
task T+1 的 owner 才能进入其 metadata 写入区
```

伪代码：

```cpp
void CommitTaskMetadata(int32_t task_id, PreparedDelta& delta) {
    // Acquire/线性化地观察前驱 metadata 已提交。
    WaitUntilDepsPrepared(task_id - 1);

    // 只允许唯一 owner 执行。
    PublishOrdinaryAndSymbolMetadata(delta);

    // A5 非一致 cache 下，必须沿用现有、已验证的 payload-before-control
    // publication seam，不能把这一步简化为普通 store + CAS 的语言级推理。
    MakeMetadataGloballyVisible(delta);

    const int32_t old =
        CAS(&task[task_id].deps_prepared, -1, task_id);

    Assert(old == -1);
}
```

`deps_prepared[T] == T` 的语义只能是：

> task T 的全部相关 metadata 已经按协议发布，后继 task 可以安全进入其 metadata 写入区。

它不表示：

- task T 的 owner 刚刚产生；
- task T kernel 已经执行完成；
- task T completion flag 已经发布。

---

## 6. 正确性合同

### 6.1 至多一个 owner

每个 group 的 local CAS 至多有一个成功者；root CAS 也至多有一个成功者。因此同一 task 至多一个 owner。

### 6.2 至少一个 owner

在以下正常运行前提下：

1. task 至少存在一个合法候选；
2. per-task local/root 节点正确初始化为 `-1`；
3. 成功的 local representative 最终能够继续执行 root CAS；
4. 正常路径不考虑 core 永久失效；

至少会有一个 local CAS 成功，并且至少一个 representative 会调用 root CAS。root 的第一次 CAS 必然从 `-1` 成功，因此 task 至少一个 owner。

由此得到：每个 task 恰好一个 owner。

### 6.3 Metadata 严格按 task-id 提交

owner `T` 在发布 metadata 前必须观察：

```text
deps_prepared[T-1] == T-1
```

而 `deps_prepared[T-1]` 只能在 task `T-1` 的 metadata 已全部发布并对后继可见后提交。因此：

```text
metadata(T-1)
  happens-before
deps_prepared[T-1]
  happens-before
metadata(T)
```

归纳可得 shared TensorMap metadata 严格按 task id 插入。

### 6.4 多 owner 不能靠最后一次 CAS 补救

owner 唯一性必须在任何共享副作用之前成立。不能允许多个候选先重复写 metadata，再依赖最后的 `deps_prepared` CAS 检测错误；此时共享历史已经被污染。

因此：

```text
只有 root CAS 成功者可以进入 Register / metadata publication
```

是硬性合同。

---

## 7. 为什么不会重现旧 hierarchical Claim 的零-winner 漏洞

旧的两轮高水位方案混淆：

```text
ELECTED(T)  !=  RESOLVED(T)
```

组内 loser 在代表完成 global Claim 前进入后继 task，后继 `T+S` 可能把共享 global cursor 推过 `T`，造成 `T` 永久零 winner。

当前方案不同：

1. task `T` 与 `T+S` 使用独立的 per-task local/root 节点；
2. `T+S` 的仲裁结果不能覆盖 `T` 的仲裁状态；
3. loser 不需要等待代表回传，因为它在本层失败后已经确定自己不是 owner；
4. shared TensorMap 顺序不依赖 Claim cursor，而由独立的 `deps_prepared` 链保证；
5. 即使 `T+S` 更早选出 owner，其 owner 也只能阻塞在 `deps_prepared[T+S-1]`，不能越序写 metadata。

因此不需要新增 `group_resolved` 轮询。

---

## 8. 与既有 A / C / D 方案的区别

| 方案 | 主要问题 | 当前方案如何避免 |
| ---- | -------- | ---------------- |
| A：group Max + global Max + global 等待 | loser 通过 `atomicAdd(0)` 回流 global 热线；或缺 resolved 时出现零 winner | loser 失败即返回；无 global cursor；per-task 状态不被后继覆盖 |
| A+：group elected + group resolved | 正确但增加 resolved publish / poll，代表慢时本组被阻塞 | 不需要 resolved；owner 顺序由 `deps_prepared` commit chain 独立保证 |
| C：per-shard task gate | gate 成为新的同地址 RMW 热点，并产生同步浪涌 | 没有入口 gate；不同 task 可以提前完成 owner election |
| D：global prefilter load | A5 Load 是 `atomicAdd(0)`，仍访问热点且拉长 winner 路径 | 不做 global pre-read |
| Flat per-task CAS | 每个 task 仍有 N 个 candidate 竞争一个地址 | 两级拆分，同一地址 fan-in 显著降低 |
| 完整二叉树 tournament | winner 串行经过约 `log₂N` 层，节点多、固定延迟深 | 当前优先两级，winner 只经过两次 atomic |

---

## 9. 为什么选择 CAS

A5 原语对照显示 CAS、Exchange、FetchMax 在相同拓扑下的 return-ready 时间基本相同，不能用性能选择原语。

选择 CAS 的理由是语义：

### CAS

```text
-1 → task_id
```

- 一次性 ownership transfer；
- 失败 contender 不改写节点；
- 成功/失败由返回旧值直接判断；
- 与 per-task one-shot node 的状态机完全匹配。

### Exchange

所有 contender 都重写相同的 `task_id`，产生不必要写流量，并且不能自然表达“只有第一个调用者拥有节点”。

### FetchMax

适合单调高水位，但 per-task node 没有跨代水位推进需求；其语义过强，没有提供额外价值。

---

## 10. 理论性能模型

设：

```text
N = 合法候选数
K = 最大 local group size
G = ceil(N / K)
α = 单次 return-ready atomic 延迟，当前假设约 160 ns
```

### 10.1 Flat Claim

所有候选访问同一地址：

```text
全体 candidate 退出 Claim 的理想时间 ≈ α × N
```

第一个 winner 可以较早产生，但所有 true loser 仍需从同一地址的串行队列返回。

### 10.2 两级 Tournament

- G 条 local node 可并行；
- 每条 local node 最多 K 个 contender；
- 每组的第一个成功者约在 `α` 后进入 root；
- root 接收 G 个 representative。

因此在理想同步到达模型下：

```text
owner 产生时间 ≈ 2α
local 层尾部完成 ≈ Kα
root 层尾部完成 ≈ (G + 1)α
全体 Claim 参与者退出时间
    ≈ α × max(K, G + 1)
```

注意：

```text
α × max(K, G + 1)
```

比把两层简单相加为 `α × (K + G)` 更准确，因为 root 竞争会与 local loser 的尾部排队重叠。

物理 atomic 数：

```text
N 次 local CAS + G 次 root CAS
```

总数略高于 Flat，但热点宽度显著下降。

### 10.3 理论最佳 group size

目标近似是最小化：

```text
max(K, ceil(N/K) + 1)
```

因此最佳 K 通常在 `sqrt(N)` 附近。

在完全理想化模型下：

| 候选数 | 较优 `(K, G)` | Flat 尾部 | 两级尾部 | 理论尾部下降 |
| ------ | ------------- | --------- | -------- | ------------ |
| 24 | `(5,5)` 或 `(6,4)` | `3,840 ns` | `960 ns` | `75.0%` |
| 32 | `(6,6)` 或 `(7,5)` | `5,120 ns` | `1,120 ns` | `78.1%` |
| 64 | `(8,8)` 或 `(9,8)` | `10,240 ns` | `1,440 ns` | `85.9%` |
| 96 | `(10,10)` 或 `(11,9)` | `15,360 ns` | `1,760 ns` | `88.5%` |

这些数字只用于选择探针参数；真实结果还受：

- 到达错峰；
- 原子单元地址并行度；
- node 地址是否同 cache line；
- 分支和索引；
- AIC/AIV 调度；
- root/local 请求交错；

影响，不能替代 A5 实测。

---

## 11. A5 探针与集成证据

以下是集成前用于选择拓扑的同人口 A5 微探针结果：

| 候选核数 | Flat 同地址 | 两级分组 | 循环耗时下降 | 加速比 |
| -------- | ----------- | -------- | ------------ | ------ |
| 24 | `4,392 ticks` | `1,270 ticks` | `71.08%` | `3.46×` |
| 32 | `5,794 ticks` | `1,420 ticks` | `75.49%` | `4.08×` |
| 64 | `11,410 ticks` | `1,756 ticks` | `84.61%` | `6.50×` |

这些数据支持：

1. 原语差异不是主因；
2. 拆散地址竞争是主收益来源；
3. 候选人口越大，两级拓扑相对 Flat 的收益越明显。

微探针的证据边界：

- 它只证明同地址竞争拆分的方向，不单独证明完整 scheduler 收益；
- 完整 scheduler 已补齐 `N=96` Alloc、地址间距、group layout、语义门槛和
  B256 perf-clock/泳道取证；
- 当前 `N96/G8` Alloc 与 `N32/G6` AIC、`N64/G8` AIV 的最终结果和产物路径，
  以 [`../perf_opt_record.md`](../perf_opt_record.md) 第 14.4～14.7 节为准。

---

## 12. 动态负载均衡与候选人口

当前方案保留 arrival-based dynamic arbitration：

```text
谁更早到达 local CAS，谁更可能成为组代表；
哪个组代表更早到达 root CAS，谁更可能成为 owner。
```

因此它保留了根据 worker 实际进度动态选择 owner 的能力，不要求 deterministic owner。

重要边界：

- 是否把 Alloc 候选从 `96` 缩到 `24` 是独立策略，不是 tournament 协议的一部分；
- 固定缩小候选集合可能降低调度自由度；
- tournament 的长期价值之一，是在保留 `96/32/64` 等完整合法候选人口的同时降低同址热点；
- 最终应分别验证候选人口、winner 分布、core 利用率与 perf-clock，不能只看 Claim 微基准。

---

## 13. Group layout 与节点布局

### 13.1 Group size

设计阶段使用固定、可复现布局，并围绕 `sqrt(N)` 扫描：

```text
N=24：优先测试 K=4/5/6/8
N=32：优先测试 K=4/6/8
N=64：优先测试 K=8
N=96：优先测试 K=8/10/11/12
```

最终没有仅凭理论选择：A5 对照后保留 `96/32/64` 候选和 `8/6/8`
local 组；Alloc 的 G12 因相对 G8 没有可辨识收益且增加约 8.9 MiB 状态而撤回。

### 13.2 Group membership

第一版可按稳定 candidate index 划组，以降低实现和验证复杂度。是否需要 task-dependent rotation，应在确认固定分组造成 winner bias 后再研究。

### 13.3 Atomic node 地址

必须分别验证：

1. 同一 word；
2. 同一 64 B cache line 内不同 word；
3. 不同 64 B cache line。

在 A5 没有证据证明同 line 不会相互干扰前，local node 和 root node 应优先按独立 `AtomicLine` 布局。不能让多个逻辑节点因节省空间重新形成物理热线。

---

## 14. 可观测性口径

不得把逻辑候选人口和物理 atomic 数混为一谈。

建议记录：

```text
eligible                // 合法候选
local_cas_issued        // 第一轮物理 CAS
local_cas_won
root_cas_issued         // 第二轮物理 CAS
owner_won
metadata_commit_started
metadata_committed
```

合同：

```text
eligible == 原候选人口
local_cas_issued == eligible
local_cas_won == root_cas_issued
owner_won == task 数
metadata_committed == task 数
```

现有 `attempted` 若代表合法 Claim 参与，应继续覆盖所有 eligible candidate，不应因某个 candidate 没进入 root 而变为 false。

---

## 15. 回归验证合同

以下项目已在首次集成、Alloc 恢复 96 候选及最终 G8 定型时执行；后续修改
Tournament、候选人口、节点布局或 `deps_prepared` 时仍应作为回归门槛。

### 15.1 CPU 确定性交错

必须覆盖：

1. 同 task 多组同时竞争，恰好一个 root owner；
2. local winner 在 root CAS 前延迟；
3. task `T+1` 先产生 owner，但不能越过 `deps_prepared[T]` 写 metadata；
4. task `T` owner 在 predecessor gate 前延迟；
5. 多个后继 owner 同时等待 commit chain；
6. root loser 不进入 Register；
7. 注入双写检测，证明任何 metadata 副作用前已完成唯一 owner 仲裁；
8. 无合法候选时进入明确 fatal，而不是永久静默等待。

### 15.2 A5 隔离探针

至少比较：

- Flat FetchMax；
- Flat CAS；
- 两级 CAS；
- 不同 K/G；
- 同 word / 同 line / 独立 line；
- 同步突发与固定错峰；
- AIC 与 AIV 分开；
- `N=24/32/64/96`。

记录：

- owner 产生时间；
- local loser 最晚返回；
- root loser 最晚返回；
- 全体参与者退出时间；
- 每类 CAS 精确次数；
- 最终 owner 数。

### 15.3 standalone shared scheduler

必须保持：

- 合法候选人口；
- 每 task 恰好一个 owner；
- metadata task-id 顺序；
- DEPSIG；
- TMOPS；
- Materialize / Register / Build 次数；
- completion / fanin / heap / TensorMap 终态；
- 真实计算结果。

性能保留由无观察 `perf-clock` 决定；泳道只用于解释 local CAS、root CAS、commit wait 和 Register 的时间迁移。

---

## 16. 当前不做的事项

当前版本明确不包含：

- per-task node 的 ABA / generation / 长窗口复用；
- full binary tournament；
- group resolved；
- global cursor pre-read；
- task gate；
- deterministic reduce；
- ordinary-load mailbox；
- deterministic owner；
- load-aware score arbitration；
- 修改 `deps_prepared` commit 语义。

---

## 17. 当前结论

当前第一候选为：

> **Two-level per-task CAS owner election + existing `deps_prepared` ordered metadata commit chain**

其关键价值不是减少 atomic 总数，而是：

1. 将同地址 N 路热点拆成多条较窄 local 热点和一条较窄 root 热点；
2. 保留全部合法候选与 arrival-based 动态负载均衡；
3. loser 在本层失败后直接返回，不需要 resolved poll；
4. per-task 节点避免后继 task 覆盖前驱 arbitration state；
5. shared TensorMap 的严格插入顺序继续由现有 `deps_prepared` 链承担；
6. CAS 与一次性节点的所有权语义匹配，并避免 Exchange 的重复写和 FetchMax 的无用高水位语义。

当前落地状态：

- 已覆盖 `N=24/32/64/96` 与 64/128/256/512 B 节点间距探针；
- 已通过 CPU 96 线程确定性交错、重复 replay、future-before-earlier 与
  唯一 owner 门槛；
- 已通过 shared scheduler 的 DEPSIG、TMOPS、候选/owner 数和终态检查；
- 已完成 B256 无泳道 perf-clock 与完整 atomic/DCCI 泳道；
- 最终保留 `96/32/64` 候选、`8/6/8` local 组和 512 B node stride。

per-task 节点的 ABA、generation 与长窗口复用仍是明确非目标；若未来允许
task-id 回绕或节点跨调度复用，必须先扩展协议并重新完成本节全部门槛。
