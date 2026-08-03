# 当前候选方案总结：两级 Per-Task CAS Tournament + `deps_prepared` 顺序提交链

> 状态：架构候选 / 待集成验证  
> 目标：在保留全部合法候选和动态负载均衡能力的前提下，消除 Flat Claim 的同地址高并发热点，同时保持 shared TensorMap 的严格 task-id 顺序插入。  
> 当前明确不纳入范围：per-task 节点的 ABA / generation / 长窗口复用设计。

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
|---|---|
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
|---|---|---|
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

| N | 较优 `(K, G)` | Flat 尾部 | 两级尾部 | 理论尾部下降 |
|---:|---:|---:|---:|---:|
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

## 11. 当前 A5 探针证据

本轮对话给出的同人口 A5 结果为：

| 候选核数 | Flat 同地址 | 两级分组 | 循环耗时下降 | 加速比 |
|---:|---:|---:|---:|---:|
| 24 | `4,392 ticks` | `1,270 ticks` | `71.08%` | `3.46×` |
| 32 | `5,794 ticks` | `1,420 ticks` | `75.49%` | `4.08×` |
| 64 | `11,410 ticks` | `1,756 ticks` | `84.61%` | `6.50×` |

这些数据支持：

1. 原语差异不是主因；
2. 拆散地址竞争是主收益来源；
3. 候选人口越大，两级拓扑相对 Flat 的收益越明显。

证据边界：

- 这些是本轮对话提供的探针结果；
- 需要在仓内保留原始输出、参数、group layout 和地址间距，才能作为可复现实验；
- `N=96` 尚需同规格 A5 取证。

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

第一版应使用固定、可复现布局，并围绕 `sqrt(N)` 扫描：

```text
N=24：优先测试 K=4/5/6/8
N=32：优先测试 K=4/6/8
N=64：优先测试 K=8
N=96：优先测试 K=8/10/11/12
```

不要仅凭理论选择；A5 实测决定最终参数。

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

## 15. 验证计划

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

进入生产路径前，仍需补齐：

- `N=96` 和地址间距 A5 探针；
- group size / layout 的实测选择；
- CPU 确定性交错；
- shared scheduler 的 DEPSIG / TMOPS / 终态对照；
- B256 无观察 perf-clock；
- 候选人口和 winner 分布的动态均衡验证。

---

## 18. Selective Participation（选择性参与）

### 18.1 机制

Selective Participation 是一个**正交于两级 tournament 的候选削减
层**：在进入 local CAS 之前，用一个固定的取模过滤，让每个 task
只由**一部分** worker 参与竞争，其余 worker 直接跳过本 task 的
Claim。

引入常数：

```text
PARTICIPATION_INTERVAL = 4   // 决定一个 core 参与多少比例的 task
```

参与判据：

```text
if (task_id % PARTICIPATION_INTERVAL == worker_id % PARTICIPATION_INTERVAL)
    // 本 worker 参与该 task 的 Claim 竞争
else
    // 本 worker 跳过本 task 的 tournament，直接继续 replay
```

即：对 task `T`，只有**余数类**
`worker_id % PARTICIPATION_INTERVAL == T % PARTICIPATION_INTERVAL`
的 worker 才竞争；其余 worker 连 local CAS 都不发。于是每个 task
的实际竞争者从 `N` 降到约 `N / PARTICIPATION_INTERVAL`，**同址
fan-in 直接按 `1/PARTICIPATION_INTERVAL` 收缩**。

> `worker_id` 应取**角色内编号**（AIC / AIV / Alloc 各自的
> role-local id），否则余数类无法覆盖该 kind 的合法候选集合，见
> §18.4。

### 18.2 伪代码

过滤放在 `ClaimTwoLevel`（§4）之前，构成一层预筛：

```cpp
constexpr int32_t PARTICIPATION_INTERVAL = 4;

bool WorkerParticipates(int32_t task_id, int32_t worker_id_in_role) {
    return (task_id % PARTICIPATION_INTERVAL)
        == (worker_id_in_role % PARTICIPATION_INTERVAL);
}

ClaimOutcome ClaimSelective(
    int32_t task_id,
    int32_t worker_id_in_role,
    int32_t arbitration_group,
    AtomicNode* local_node,
    AtomicNode* root_node) {

    // 选择性参与预筛：不属于本 task 余数类的 worker 直接退出
    if (!WorkerParticipates(task_id, worker_id_in_role)) {
        return {
            .eligible = true,          // 仍是合法候选
            .participating = false,    // 但本 task 选择不参与
            .local_attempted = false,
            .root_attempted = false,
            .won = false,
        };
    }

    // 参与者进入原有两级 tournament（§4），语义不变
    return ClaimTwoLevel(
        task_id, arbitration_group, local_node, root_node);
}
```

被过滤掉的 worker `participating == false`，**不发任何 CAS**、不
轮询、不等待，立即继续 replay 后续 task。它仍是逻辑上的
`eligible` 候选，只是本 task 主动弃权（§18.7 的口径）。

### 18.3 竞争规模与性能

设过滤后每 task 参与者 `N' ≈ N / PARTICIPATION_INTERVAL`。

| 层 | Flat | 两级 | 两级 + Selective |
| -- | ---- | ---- | ---------------- |
| 同址 fan-in | `N` | local `K` / root `G` | local `K` / root `G' = ⌈N'/K⌉` |
| 物理 CAS 数 | `N` | `N` local + `G` root | `N'` local + `G'` root |
| 尾部（§10 模型） | `α·N` | `α·max(K, G+1)` | `α·max(K, G'+1)` |

要点：

- Selective 与两级 tournament **叠乘**：先把 `N` 降到 `N'`，再由
  两级把 `N'` 拆成 local/root 两条窄热线。二者收益相乘。
- 也可**单独**叠在 Flat 上：Flat 同址 fan-in 从 `N` 降到 `N'`，
  尾部从 `α·N` 降到 `α·N'`，即 `1/PARTICIPATION_INTERVAL`。
- 物理 CAS 总数同样按 `1/PARTICIPATION_INTERVAL` 下降——这是它
  与 §10 两级方案的差别：两级降**热点宽度**但不降 CAS 总数，
  Selective **直接降 CAS 总数与候选人口**。

### 18.4 正确性必须守住的前提

Selective **不改变** §6 的三条合同的证明结构，但给「至少一个
owner」新增一个**必须显式守住**的前提：

1. **每个 task 至少一个参与者**。对 task `T`，参与者是余数类
   `r = T % PARTICIPATION_INTERVAL`。必须保证该 kind 的合法候选
   里**至少有一个** role-local id 满足 `id % I == r`，否则该 task
   参与者为空 ⇒ 零 owner，直接违反 §6.2。
   - 由于 role-local id 连续（`0 .. num_role_workers-1`）且通常
     `num_role_workers ≫ PARTICIPATION_INTERVAL`，每个余数类都被
     覆盖；但这要作为**前提断言**，并按 §15.1 第 8 项「无参与者
     即 fatal，不静默等待」处理。
   - 若某 kind 的合法候选因 role/lane 过滤后**不连续**（例如只有
     少数几个 worker 合法），必须确认它们的余数类仍覆盖所有会被
     Attempt 的 task 残差，否则 Selective 不适用于该 kind。
2. **至多一个 owner** 不受影响：参与者子集内仍跑原两级 CAS，
   local/root 各至多一个成功者。
3. **metadata 有序** 不受影响：顺序由独立的 `deps_prepared` 链
   承担（§5/§6.3），与谁参与、参与多少无关。

### 18.5 开销：利用率浪费与 winner bias

Selective 用「牺牲一部分调度自由度」换「更小竞争规模」：

- **worker 利用率浪费**：一个 worker 在 `I` 个 task 里只参与
  `1` 个，其余 `I-1` 个即使空闲、即使能更快到达，也**不允许**
  参与竞争。若被指定余数类的参与者恰好都较慢，而其它空闲 worker
  被过滤在外，该 task 的 owner election 无法利用那部分算力——
  这正是本方案的主要代价（对应 §12 的动态负载均衡边界）。
- **winner 分布收窄 / bias**：task `T` 的 owner 只能来自余数类
  `T % I`，owner 集合被结构性地绑定到 worker 的静态余数。若后续
  Register / metadata prepare 负载在 owner 上，可能造成余数类之间
  的负载倾斜。与 §13.2 的固定分组 bias 是同类问题，需实测
  winner 分布与 core 利用率确认。
- **动态性下降**：§12 的 arrival-based 仲裁只在**参与者内部**
  仍然成立；跨余数类的动态迁移被禁止。`PARTICIPATION_INTERVAL`
  越大，竞争越小、但弃权越多、利用率浪费与 bias 越重——这是一个
  需要用 A5 实测标定的权衡旋钮，而非越大越好。

### 18.6 与两级 tournament / `deps_prepared` 的关系

- **与两级正交且可叠乘**：Selective 只改「谁进入 Claim」，两级只
  改「进入者如何收敛成唯一 owner」。二者互不依赖，可单独或叠加
  启用。
- **与 `deps_prepared` 无关**：commit chain 只认 owner 与前驱门，
  不关心 owner 来自哪个余数类，因此严格插入顺序保持不变。
- **与 §16 边界一致**：Selective 是**候选人口削减策略**，与「是否
  把 Alloc 候选从 96 缩到 24」属同一类独立策略（§12），不是
  tournament 协议的一部分；启用与否应能独立开关并单独取证。

### 18.7 可观测性与验证补充

在 §14 口径上新增区分「合法候选」与「实际参与者」：

```text
eligible                 // 合法候选，不因弃权变 false
participating            // 通过 Selective 过滤、真正进入 Claim 的候选
local_cas_issued         // == participating（每个参与者一次 local CAS）
participation_skipped    // == eligible - participating
```

新增合同：

```text
participating          ≈ eligible / PARTICIPATION_INTERVAL
local_cas_issued       == participating
owner_won              == task 数        // 弃权不得导致零 owner
metadata_committed     == task 数
```

验证需在 §15 基础上补：

1. 构造某余数类**无合法候选**的 task，确认进入明确 fatal 而非
   静默零 owner（§18.4 前提 1）；
2. 扫描 `PARTICIPATION_INTERVAL ∈ {1,2,4,8}`，记录每 task 参与者
   数、尾部时间、owner 余数类分布与 core 利用率；
3. 确认 `PARTICIPATION_INTERVAL=1` 时行为与未启用 Selective 逐位
   一致（退化对照）；
4. shared scheduler 终态（DEPSIG / TMOPS / owner 数 / metadata
   顺序）相对基线不变，仅 `participating` / CAS 计数下降。

---

## 19. Same-Core Scalar Coroutine：执行期交错下一轮 Claim

### 19.1 动机：编排优化逼近上限后的下一瓶颈

当前方案族（两级 per-task CAS tournament、Selective Participation、
`deps_prepared` 提交链）的主目标是压缩 **orchestration** 开销：
task claim 竞争与依赖构建。实测与模型都显示这条路径已明显变窄，
继续在 Claim 拓扑上抠收益会进入边际递减。

每个物理核上的 worker 大致交替两段工作：

```text
orchestration（Claim / Register / deps / Submit 外壳）
        ↕ 串行切换
execution（in-core QK/SF/PV/UP …：Cube / Vector / MTE 主导）
```

关键观察：

1. **执行阶段标量单元并不总是满载**。In-core kernel 的主体负载在
   Cube / Vector / MTE；scalar 主要发指令、维护循环索引、做
   `SetFlag` / `WaitFlag` 与少量地址算术。长算子在异步管线上跑时，
   scalar 常处于「等 flag / 等 pipe」的轻载或阻塞态。
2. **编排阶段几乎吃满 scalar**，且大量时间花在 GM 原子与控制流上，
   与 Cube/Vector 重叠度低。
3. 若能在**同一物理核**上，让 scalar 在执行期的空档去推进**下一
   task 的 Claim 竞争**，则可把编排延迟藏进已有的执行气泡，而不把
   工作扔到别的核（因而**不增加跨核 GM 热线流量**）。

本方案是相对 Claim 拓扑的**正交下一阶段**：不改「谁赢」的合同，
改「何时用本核 scalar 做 Claim」。

> 与 `shared构建执行分离.md` 刻意不引入 engine coroutine 的边界
> 不冲突：那份文档排除的是把跨核发布 / owner 仲裁 / 回收与另一套
> 调度混做；本章讨论的是**同核、协作式、仅交错「已在跑的
> in-core」与「本核下一轮 Claim」**，且默认不改变跨核 publication
> 合同。若落地，应作为独立开关与探针，不并入首版 build/execute
> 分离路径。

### 19.2 方案概述

把每个 worker 拆成两个协作式协程（cooperative coroutine），共享
同一物理核、同一 GM 视图、同一本地 ring/slot，但分时占用 **scalar
控制流**：

| 协程 | 职责 | 主要占用 |
| ---- | ---- | -------- |
| `ExecCoro` | 跑当前已 claim 且 fanin ready 的 in-core kernel | Cube / Vector / MTE + 间歇 scalar |
| `OrchCoro` | 对后续 task 做 Selective / 两级 CAS Claim、轻量编排 | 几乎纯 scalar + GM atomic |

调度模型（协作式，非抢占）：

```text
ExecCoro 发出长时 Cube/Vector/MTE 操作
    → 到达 yield 点（见 §19.3）
    → 保存 Exec 标量上下文，切换到 OrchCoro
OrchCoro 推进下一 task 的 Claim（local/root CAS 等）
    → 自己的原子返回或到达预算上限
    → 切回 ExecCoro
ExecCoro 恢复 WaitFlag / 后续 tile
    → …
```

约束（第一版建议写死）：

1. **同核 only**：OrchCoro 只服务本 `worker_id` 的 Claim 参与，不
   代替异核 executor、不跨核偷任务执行。
2. **只交错 Claim / 极轻编排**：第一版禁止在 yield 窗口内做
   Register、metadata publish、`deps_prepared` CAS、大块 GM 拷贝；
   这些仍留在传统编排段，避免与 in-core 的 UB/L1/GM 流水纠缠。
3. **协作式 yield**：只在明确插入的调度点切换；不依赖硬件抢占
   scalar。
4. **Cube/Vector 不中断**：切换发生在「算子已下发、scalar 本可空等」
   的窗口；不假设能抢 Cube/Vector 流水线。

收益假设：

```text
隐藏延迟 ≈ min( Claim_scalar_work , Σ bubble_i )
bubble_i = 第 i 个 yield 窗口内 Cube/Vector/MTE 仍在飞、scalar 本会空等的时间
```

且 Claim 的 GM 原子仍由**本核**发出，不引入新的跨核编排角色，
因此不额外制造「为编排而编排」的 GM 多播；只是把本就会发生的
本核原子时间前移到执行气泡里。

### 19.3 可插入的协程调度点（scheduling points）

Ascend in-core 函数里，scalar 是控制平面：它 **issue** 异步管线
操作，再 **WaitFlag** 回收完成。真正可让出 scalar 的窗口，是
「长操作已 issue、完成尚未回收」之间——或等价的、已知 Cube/Vector
仍忙碌的间隙。

推荐按**安全与收益**分级插入：

#### A 级（首选）：异步 issue 之后、对应 `WaitFlag` 之前

```text
// 伪代码：tile / 流水节拍内
IssueCubeOrVectorOp(...);          // 或 MTE copy，长延迟
SetFlag<PIPE_M / PIPE_V / ...>(...);

// >>> SCHED_POINT: scalar 本将空等管线 <<<
CoroutineYieldToOrch(budget);      // OrchCoro 做下一 task Claim

WaitFlag<PIPE_M / PIPE_V / ...>(...);
```

这是最干净的点：管线已有活，scalar 若不 yield 也会在 `WaitFlag`
上阻塞；把阻塞换成「有限预算的 Claim 工作」是直接的气泡填充。

#### B 级：双缓冲 / 多 tile 流水的「另一侧仍在飞」处

例如 ping-pong：buffer0 上 Vector 仍在算，scalar 已可启动 buffer1
的 MTE；在启动下一拍之前或之后插入 yield，用「另一侧未完成」作
为气泡证明。

#### C 级：显式软件预算点（无硬件 flag 时）

在已知耗时的大循环边界（如按 block-group / head 维的外环）插入：

```text
for (tile = ...) {
    RunTile(...);
    if ((tile % YIELD_EVERY) == 0)
        CoroutineYieldToOrch(budget);
}
```

收益依赖「tile 内 Cube/Vector 是否仍有未回收工作」。若 tile 结尾
已经 `WaitFlag` 排空所有 pipe，则此处**没有**硬件气泡，yield 只会
拉长执行关键路径——只适合探针对比，不适合作为默认点。

#### 不推荐作为默认点

| 点 | 原因 |
| -- | ---- |
| 任意算术指令之间 | 无异步重叠；纯拉长 kernel |
| UB/L1 上正在被 MTE/Cube 使用的数据的标量读写附近 | 易破坏 pipe 依赖与 hazard 假设 |
| kernel 内已有的 GM atomic / 完成字发布前后 | 与 OrchCoro 的 Claim 原子可能形成同核重入与顺序难题 |
| `deps_prepared` / metadata publish 路径 | 有跨核可见性合同，不应放进执行气泡 |

**插桩合同（给 codegen / kernel 作者）：**

```text
在 CoroutineYieldToOrch 调用处，必须成立：
1. 本核至少一条非 scalar 管线仍有未 Wait 的未完成操作；或
2. 调用方能证明 yield 预算 << 该未完成操作的剩余延迟；
3. kernel 的标量现场（栈、自动变量、pipe 计数、flag id）可被保存/
   恢复，且 OrchCoro 不得破坏 ExecCoro 的 UB/L1/PIPE 状态；
4. yield 窗口内禁止假设「当前 tile 的输出已对后继可见」。
```

### 19.4 运行时草图

```cpp
struct WorkerCoroState {
    // Exec
    void* exec_stack;
    KernelFrame exec_frame;     // PC / 标定的可恢复断点 id
    PipeSnapshot pipes;         // 哪些 flag 已 Set 未 Wait

    // Orch
    void* orch_stack;
    ClaimCursor orch_claim;     // 下一个要尝试的 task_id 等
    uint32_t orch_budget_atomics;

    enum { EXEC, ORCH } running;
};

// ExecCoro 内由 kernel 调用
void CoroutineYieldToOrch(uint32_t budget) {
    SaveExecScalarContext();
    worker.orch_budget_atomics = budget;
    SwitchTo(ORCH);
    // Orch 返回后：
    RestoreExecScalarContext();
}

// OrchCoro 主循环片段
void OrchCoroMain(WorkerCoroState& w) {
    while (w.orch_budget_atomics > 0 && HasPendingClaimWork(w)) {
        ClaimOutcome o = ClaimSelective(...);   // §18 + §4
        --w.orch_budget_atomics;
        if (o.won) {
            ArmLocalSlotForLaterDrain(o);       // 只写本核私有状态
            break;                              // 赢了就尽快还 Exec
        }
        if (ShouldReturnToExec(w)) break;
    }
    SwitchTo(EXEC);
}
```

第一版建议的 **预算** 以「允许的 GM CAS 次数」计量（例如 1～2 次
local/root），而不是墙钟时间——A5 上墙钟难在 kernel 内可靠读取，
且 CAS 次数与 §10 的 `α` 模型直接对应。

### 19.5 与现有 Claim / Selective / Drain 的衔接

```text
时间线（单核）：

Claim(T) 胜 → Build/Register（传统编排段）→ Drain 启动 Kernel(T)
    → Kernel(T) 中多次 Yield → 其间 Orch 可能 Claim(T+k) 胜/负
    → Kernel(T) 结束 → Drain 收尾 / completion
    → 若 T+k 已 claim 且 fanin ready，可立刻 Drain；否则再 Orch
```

要点：

- **Yield 中的 Claim 不得要求当前 kernel 已完成**；它只提前做
  owner election。`deps_prepared` 与 metadata 顺序仍走 §5 链，
  不因 Claim 提前而越序写共享历史。
- **winner 在 yield 窗口内**只应把结果落到**本核私有**结构（例如
  「下一 slot 已选举」标记）；完整 Register / publish 仍回传统
  编排段，避免与 in-core GM/UB 并发。
- Selective（§18）仍然适用：OrchCoro 只对参与余数类发 CAS，进一
  步减少 yield 窗口内的原子次数。

### 19.6 预期收益与不收益的情形

| 情形 | 预期 |
| ---- | ---- |
| 长 Cube/Vector tile + A 级 WaitFlag 前 yield；Claim 已是短路径（两级+Selective） | Claim 延迟被气泡吃掉，核间等待 fanin/owner 的尾部下降 |
| kernel 标量密集、几乎无异步气泡（短 kernel / 纯 scalar） | **负收益**：切换成本直接加在关键路径 |
| Claim 仍极热、单次 CAS ≈ 数百 ns，而单次气泡 < CAS | 需要多次 yield 才能藏完；否则只藏一部分 |
| 执行已是全局瓶颈、编排已不是关键路径 | 端到端加速有限；应用 PMU/泳道先确认 scalar 空等占比 |

验收应看三类信号，而不是只看 Claim 微基准：

1. kernel 内 `yield` 次数与每次 Orch 完成的 CAS 数；
2. 从「Kernel(T) 结束 → Claim(T+k) 完成」的间隔是否下降；
3. 端到端 perf-clock 与 scalar 空等 PMU（若有）是否同向改善。

### 19.7 挑战与风险

#### 19.7.1 硬件 / 编程模型是否允许「Wait 中干别的」

协作式 yield 的前提是：长操作 issue 之后，scalar **可以**在
`WaitFlag` 之前跑别的代码，且管线继续推进。若某工具链把
issue+wait 固化成不可拆开的 intrinsic，或 wait 期间不允许触碰
GM 原子，则 A 级调度点不成立，方案退化为 C 级软件猜测。

需要 A5 最小探针证明：

```text
Issue long Vector/Cube
  → 中间插入 N 次 GM CAS（或 dummy scalar 工作）
  → WaitFlag
结果正确，且总时间 < Issue+Wait 与 CAS 串行相加
```

#### 19.7.2 上下文切换成本

Exec/Orch 切换需保存：scalar 寄存器、栈指针、（可选）PC/断点 id、
pipe/flag 记账。若由软件完整保存大量寄存器，切换可能吃掉短气泡。
第一版应：

- 限制 `KernelFrame` 为**显式断点**（标注可恢复的 yield id），避免
  任意指令级续跑；
- 测量「空 yield」（Orch 立即返回）的成本，作为气泡准入阈值。

#### 19.7.3 同核重入与状态机

当前 runtime 默认「单核单线程栈」：Claim、Drain、kernel 顺序占用
scalar。引入协程后：

- kernel 未完成时 Orch 可能改 `task_id`、slot、统计、trace；
- 必须证明 Orch 触摸的集合与 Exec 触摸的集合在第一版**不相交**
  （Exec：UB/L1/pipe/当前 slot；Orch：tournament 节点 + 下一
  slot 的选举结果字）；
- trace / PMU bracket 在切换点可能嵌套错误，泳道需新的
  `coro_yield` / `coro_orch` 事件。

#### 19.7.4 GM 原子与 in-core GM 流量的干扰

即便不增加**跨核编排角色**，本核在 kernel 中段发 Claim CAS 仍会：

- 占用 GM / 原子单元带宽，可能干扰同核 kernel 的 GM load/store；
- 拉长其它核在同一 tournament 节点上的排队。

因此「不增加跨核编排流量」≠「对 GM 零影响」。需要对「执行中段
CAS」与「执行结束后 CAS」做对比探针。

#### 19.7.5 正确性合同不自动继承

§6 的「恰好一个 owner」与「metadata 有序」在逻辑上仍由 CAS 与
`deps_prepared` 保证，但实现上要防：

- yield 窗口内误调用 publish / completion；
- fanin 未 ready 时提前 Drain；
- kernel 失败路径与 Orch 已写入的「下一 winner」之间的回滚。

#### 19.7.6 Codegen / 内核源侵入

调度点必须打进**每个**目标 in-core 函数（或公共 tile 模板）。这是
跨 runtime 与 codegen 的契约：

- 手写 kernel：易漏插、易插在无气泡处；
- 模板/codegen：需要统一的 `YIELD_IF_CORO_` 宏与静态检查
  （「yield 前存在未 wait 的 issue」）；
- 第三方/闭源 kernel：无法插桩则该核无法受益。

#### 19.7.7 与「标量已满」现实的偏差

本章的前提是「执行期 scalar 相对轻」。若泳道/PMU 显示某类
kernel（例如大量地址计算 + 短向量）scalar 已是瓶颈，则 coroutine
会加剧争用。必须以 kind 为单位门控（例如仅 PV/QK 长大 tile 开启）。

### 19.8 建议推进顺序

1. **PMU / 泳道取证**：量化执行期 scalar 空等 vs Cube/Vector 活跃
   的比例；无气泡则停。
2. **A5 微探针**：issue → N×CAS → wait 的正确性与重叠收益。
3. **空 yield 成本探针**：标定最小有用气泡长度。
4. **单 kind 试点**：在一个 tile 模板的 A 级点接入
   `CoroutineYieldToOrch(1)`，Orch 只做 Selective+local CAS。
5. **合同测试**：切换点注入、禁止 publish、owner 唯一、
   DEPSIG/TMOPS 不变。
6. **再扩 root CAS / 多预算**；最后才考虑在 yield 窗口做更多编排。

### 19.9 小结

| 维度 | 内容 |
| ---- | ---- |
| 想法 | 同核协作式协程：Exec 在管线气泡让出 scalar，Orch 提前做下一 task Claim |
| 调度点 | 优先「异步 issue 之后、WaitFlag 之前」；其次双缓冲间隙；避免无气泡处乱插 |
| 收益 | 隐藏编排延迟；不引入异核编排角色、不额外制造跨核 GM 编排多播 |
| 代价 / 风险 | 切换成本、同核重入、GM 带宽干扰、kernel 侵入、气泡前提可能不成立 |
| 定位 | Claim 拓扑优化见顶后的**下一正交阶段**；独立开关，不并入首版跨核 build/execute 分离 |

当前结论：该方向**值得作为研究候选**，但必须先用 A5 证明
「issue–CAS–wait」真能重叠，再用单一 kind 证明端到端收益；在此
之前不应假设它能继续兑现与两级 tournament 同量级的加速。
