# ClaimMax 分层竞争（Hierarchical Competition）方案

> **状态：历史候选，未采用。** 本文 §3.3、§4.9 和 §5 对方案 D
> “廉价 Load”“零风险”的判断，已被
> [`2-hierarchical_competition-2-analysis.md`](2-hierarchical_competition-2-analysis.md)
> 基于 A5 实际 `Ops::Load()==atomicAdd(0)` 的复审否定。当前保留实现是
> [`two_level_per_task_cas_tournament_summary.md`](two_level_per_task_cas_tournament_summary.md)
> 总结的 per-task CAS Tournament；本文仅作为候选演进记录，不代表当前实现。

本文描述 shared tensormap runtime 中，在现有 **ClaimMax cursor
分片**之上进一步降低同 task `atomicMax` 冲突的 **两轮分层竞争**
方案。目标：缩短 Claim 热路径上 `atomicMax` 的尾延迟，同时保持
「同一 cursor 上每个 `task_id` 至多一个 winner」的语义。

相关现状代码：`common/pa_scheduler_core.h` 的 `Claim()`；cursor
定义见 `common/pa_model.h`
（`alloc_cursor` / `cube_cursor` /
`shared_vector_cursor` 等）。

---

## 1. 背景与动机

### 1.1 原始单 cursor 竞争

Claim 用单调 cursor 上的 `atomicMax` 裁定 winner / loser：

```text
old = atomicMax(&claim_max, task_id)
won = (old < task_id)
```

凡参与该 task 提交的核都要对**同一**全局 `claim_max` 做
`atomicMax`。最坏情况下全部候选核同时撞同一 cache line，
`atomicMax` 延迟随并发竞争者近似恶化，成为 Claim 的主要开销。

### 1.2 现状：ClaimMax 分片

当前实现对不同 kind 使用分片 cursor，例如：

| Kind | Cursor | 分片数 |
| ---- | ------ | ------ |
| Alloc | `alloc_cursor[task_id % kCursorShards]` | 4 |
| QK / PV | `cube_cursor[task_id % kCursorShards]` | 4 |
| SF / UP（shared） | `shared_vector_cursor[task_id % kSharedVectorCursorShards]` | 8 |

分片把**不同 `task_id`** 的 Claim 分散到多条 AtomicLine，降低
跨 task 的 false sharing，实测有收益。

但分片键是 `task_id % S`：**同一个 `task_id` 的全部候选核仍落在
同一 shard**。例如 64 个 AIV 争同一个 SF/UP task 时，仍会全部
`atomicMax` 到同一条 `shared_vector_cursor[shard]`。因此在同
task 高并发 Claim 下，仍能观察到偏长的 `ClaimMax` /
`atomicMax` 时间。

### 1.3 下一步：分层竞争

在（可选保留的）shard cursor 之上，再按核分组做两轮竞争：

1. **组内轮**：先在 `claim_group_max[shard][group]` 上竞争；
2. **全局轮**：仅组内胜者再对 `claim_max[shard]`（即现有 shard
   cursor）做 `atomicMax`；
3. **回写**：全局轮失败者把已观察到的全局水位
   `atomicMax` 回写到本组 cursor，避免组内后续核重复冲击全局线。

常数：`COMPETITION_GROUP_SIZE = 8`。

---

## 2. 方案概述

### 2.1 数据结构

在现有 per-shard 全局 cursor 旁增加组级 cursor（每条
`AtomicLine`，与现网 Claim cursor 同宽）：

```text
// 以 Vector shared 为例；Cube/Alloc 同构
// S = kSharedVectorCursorShards（或对应 kind 的分片数）
// G = ceil(num_eligible_workers / COMPETITION_GROUP_SIZE)

AtomicLine claim_max[S];                 // 现有 shard cursor（全局轮）
AtomicLine claim_group_max[S][G];        // 组内轮 cursor
```

核到组的映射（固定、无动态除法即可）：

```text
COMPETITION_GROUP_SIZE = 8
group_id = worker_id_in_role / COMPETITION_GROUP_SIZE
// 例：64 AIV → G=8；32 AIC → G=4；96 Alloc → G=12
```

`worker_id_in_role` 取该 Claim 路由下的角色内编号（AIV 用
`worker_id - kAicWorkers`，AIC 用 `worker_id`，Alloc 用全局
`worker_id`），保证同 role 候选落在连续组内。

### 2.2 两轮裁定规则

| 轮次 | 操作 | 结果 |
| ---- | ---- | ---- |
| 组内 | `atomicMax(claim_group_max[s][g], task_id)` | `old_g >= task_id` → **loser**（不再碰全局） |
| 全局 | 组内胜者再 `atomicMax(claim_max[s], task_id)` | `old < task_id` → **唯一 winner** |
| 回写 | 全局失败者 | `atomicMax(claim_group_max[s][g], global_watermark)` 后仍为 **loser** |

不变量（与现网一致）：

- 对给定 shard cursor，推进到 `task_id` 时至多一个核观察到
  `old < task_id`，该核为 winner；
- 组内失败、全局失败均为 loser；loser 不进入 Materialize /
  Register / Build；
- cursor 单调不减；回写只把**更高的全局水位**灌回组内，不降低
  任何 cursor。

### 2.3 为何全局失败者要回写组 cursor

组内胜者已把 `claim_group_max[s][g]` 至少推到 `task_id`，但全局
可能已被其他组推到更高值 `W >= task_id`。若不回写：

- 同组后续核仍可能对 `(task_id, W]` 再次「组内获胜」，反复冲击
  全局线，抵消分层收益。

因此第二轮失败者执行：

```text
atomicMax(&claim_group_max[s][g], W)
```

其中 `W` 取全局轮返回的 `old`（已是写入前全局值，且
`old >= task_id`），或再 load 一次全局 cursor；语义上都是把组水位
抬到「至少不小于已见全局水位」。

> 注意：全局轮的 `atomicMax(claim_max, task_id)` **已经**把
> `task_id` 合并进全局 cursor。失败者**不必**再把组水位单独写回
> 全局；需要的是 **全局 → 组** 的下行同步。

---

## 3. 伪代码

下列伪代码嵌在现有 `Claim()` 的 kind / role 路由之后：已选好
`shard` 与对应的全局 cursor 指针，且本核 `attempted == true`。

```text
CONST COMPETITION_GROUP_SIZE = 8

FUNCTION HierarchicalClaim(task_id, worker_id_in_role, shard):
    g = worker_id_in_role / COMPETITION_GROUP_SIZE
    group_cursor  = &claim_group_max[shard][g]
    global_cursor = &claim_max[shard]          // 现网 shard cursor

    // ---- Round 1: 组内竞争 ----
    old_g = AtomicMax(group_cursor, task_id)
    IF old_g >= task_id:
        RETURN { attempted: true, won: false }   // 组内 loser

    // ---- Round 2: 仅组内胜者冲全局 ----
    old = AtomicMax(global_cursor, task_id)
    IF old < task_id:
        RETURN { attempted: true, won: true }    // 全局 winner

    // ---- 全局 loser：把全局水位灌回本组 ----
    // old 已是写入前的全局值，且 old >= task_id
    AtomicMax(group_cursor, old)
    RETURN { attempted: true, won: false }
```

与现网单轮 Claim 的对照：

```text
FUNCTION FlatClaim(task_id, shard):             // 现状
    old = AtomicMax(&claim_max[shard], task_id)
    RETURN { attempted: true, won: (old < task_id) }
```

角色过滤（非本 lane / 非候选）仍在分层之前返回
`attempted: false`，与现网一致；分层只替换
`TraceAtomicFetchMax(... ClaimMax ...)` 这一段竞争。

### 3.1 正确性要点（草图）

1. **全局至多一个 winner**：只有组内胜者才执行全局 `atomicMax`；
   全局仍是单线单调 `atomicMax`，故对同一 `task_id` 至多一个核
   看到 `old < task_id`。
2. **回写不破坏单调性**：只对组 cursor 做 `atomicMax` 抬升。
3. **与 task_id 分片正交**：`shard = task_id % S` 仍先选线。

> 下列「每个 task 必有 winner / cursor 不跳号」**不能**由 §3 伪代码
> 单独推出，见 §3.2。

### 3.2 每个 task_id 是否必有 winner？能否保证 +1？

#### 先澄清现网不变量

现网 `FlatClaim` 用的也是 `atomicMax`，**并不**保证
`claim_max` 每次只 `+1`：

- 分片后同一 cursor 上的 task 序列是
  `…, t, t+S, t+2S, …`，成功 Claim 时 cursor 从 `t` 跳到
  `t+S`（步长为 `S`，不是 `1`）；
- 真正要守的是：**该 cursor 负责的每个 `task_id` 恰好有一个
  winner**，即序列上不出现「洞」——更高 `task_id` 先写入全局后，
  更低的 `task_id` 再 Claim 会全员 `old >= task_id` 而**永无
  winner**。

Flat 路径能守住「无洞」，靠的是：每个候选核在进入更高
`task_id` 之前，**已经在同一全局线上完成过**当前
`task_id` 的 `atomicMax`。Loser 返回时全局必已 `>= task_id`，
因此不可能在全局仍停在 `pred` 时去 `atomicMax(next)`。

#### 朴素分层会破洞（可跳号）

§3 伪代码里，**组内 loser 立刻返回**，并不等待组胜者完成全局
`atomicMax(task_id)`。Loser 路径很轻，可以继续回放并 Claim 同
shard 上更大的 `task_id'`。于是：

```text
全局 claim_max = pred          // 尚无 T 的 winner
组 g：W1 组内胜 T，正在去全局（未完成）
组 g：W2 组内负 T，立刻返回 loser
W2 跑到同 shard 的 T' = next(T)（如 T+S）
W2 组内胜 T'，再 AtomicMax(global, T')
  → old = pred < T' ⇒ W2 成为 T' 的 winner，全局跳到 T'
W1 稍后 AtomicMax(global, T)
  → old = T' >= T ⇒ W1 也输
⇒ task T 没有任何 winner（被跳过）
```

跨组同样成立：任一核在「全局尚未覆盖 T」时对同 shard 的
`T' > T` 做全局 `atomicMax`，即可制造空洞。

因此：

| 问题 | §3 朴素分层 | 现网 Flat |
| ---- | ----------- | --------- |
| 同一 `task_id` 至多一 winner | 是 | 是 |
| 每一 `task_id` 至少一 winner | **否（可跳号）** | 是* |
| `claim_max` 每次字面 `+1` | 否 | 否（分片步长 `S`） |
| 无洞推进（到 next shard task） | **否** | 是* |

\*前提：候选核按 task 序 Attempt，且每次 Attempt 都碰全局线。

§3.1 旧表述「组内失败 ⇒ 全局必败 / 组胜者会（或已）写全局」里，
「会写」是异步的；在写完之前组内失败者已可跑飞——这是漏洞。

#### 若要恢复「无洞」，分层必须加藩篱

任选其一（或等价物），否则不能上生产：

##### 方案 A — 组内负后等全局水位（推荐与 atomicMax 兼容）

```text
old_g = AtomicMax(group_cursor, task_id)
IF old_g >= task_id:
    WHILE Load(global_cursor) < task_id:
        // 等待组胜者（或其他组胜者）提交全局
        PAUSE
    // 可选：AtomicMax(group_cursor, Load(global)) 提前回写
    RETURN { attempted: true, won: false }

old = AtomicMax(global_cursor, task_id)
...
```

代价：组内 loser 从「1 次低争用原子」变为「可能自旋等全局
可见」；延迟通常仍远小于 N 路 `atomicMax`，但要用泳道量 P99。

##### 方案 B — 全局改为前驱 CAS（强制逐步推进）

核心思想：把全局层从「取最大值」换成「只接受严格前驱」。全局
cursor 只有当前值恰为 `task_id` 在本 shard 上的前驱 `pred` 时，
才允许被推进到 `task_id`。这样跳号在硬件语义上被禁止：`T'` 想
发布，必须先看到全局 `== pred(T') == T`，而 `T` 未发布则
`pred(T')` 不成立，CAS 必然失败。

前驱定义（与分片步长一致）：

```text
FUNCTION PredOnShard(task_id):
    // 同一 shard 上的合法 task 序列是 base, base+S, base+2S, ...
    // 其中 base = shard，S = 该 kind 的分片数
    IF task_id < S:              // 本 shard 的首个 task
        RETURN SENTINEL_EMPTY    // 例如 -1
    RETURN task_id - S
```

组内筛选保留，组胜者改打前驱 CAS；组内失败者仍要等到全局覆盖
本 task 才能离开，否则跑飞窗口未闭合：

```text
CONST COMPETITION_GROUP_SIZE = 8

FUNCTION HierarchicalClaim_B(task_id, worker_id_in_role, shard):
    g = worker_id_in_role / COMPETITION_GROUP_SIZE
    group_cursor  = &claim_group_max[shard][g]
    global_cursor = &claim_max[shard]
    pred = PredOnShard(task_id)

    // ---- Round 1: 组内竞争 ----
    old_g = AtomicMax(group_cursor, task_id)
    IF old_g >= task_id:
        // 组内 loser：仍须等全局覆盖本 task，堵住 §3.2 跑飞窗口
        WHILE Load(global_cursor) < task_id:
            PAUSE
        RETURN { attempted: true, won: false }

    // ---- Round 2: 组胜者用前驱 CAS 抢全局 ----
    // 只有全局恰停在 pred 时才允许发布 task_id
    WHILE TRUE:
        cur = Load(global_cursor)
        IF cur >= task_id:
            // 已被他人（不可能是本 task 的更晚者，见正确性）覆盖
            RETURN { attempted: true, won: false }
        IF cur == pred:
            IF CompareExchange(global_cursor, expected=pred, task_id):
                RETURN { attempted: true, won: true }   // 唯一 winner
            // CAS 失败：有人并发改了 cur，重读
            CONTINUE
        // cur < pred：前驱尚未发布，等待前一个 task 的 winner
        PAUSE
```

正确性（为何 B 保证「每 task 恰一个 winner、无洞」）：

1. **无洞**：`task_id` 只能在 `global == pred` 时发布。归纳：
   `pred` 发布过 ⇒ 之前每个更小 task 都发布过；否则 CAS 永远
   等不到 `cur == pred`，只能停在 `cur < pred` 自旋，直到前驱
   winner 出现。故不存在「T 未发布而 T' 已发布」。
2. **至多一个 winner**：CAS 的 `expected=pred` 只可能成功一次；
   一旦某核把 `pred → task_id`，其余组胜者读到 `cur == task_id
   >= task_id` 直接判负。
3. **至少一个 winner**：`task_id` 的候选核里必有至少一个组胜者
   （组内 `atomicMax` 单调，最大者胜其组），且回放保证该
   `task_id` 一定被某物理核 Attempt；该组胜者会自旋到
   `cur == pred` 并成功 CAS（除非已被同 task 另一组胜者抢先，
   那仍是一个 winner）。

代价与风险：

| 维度 | 说明 |
| ---- | ---- |
| 原子类型 | 全局从 `atomicMax` 换成 `CAS + Load` 自旋；单次成功 CAS 成本与 `atomicMax` 同量级，但**失败重试**会增加流量 |
| 串行化 | 全局被强制成严格顺序发布，本 shard 上 winner 之间形成生产者链；若某 task winner 迟迟不出现，后续 task 组胜者全部自旋等待（隐性 barrier） |
| 组 loser 自旋 | 与方案 A 相同，仍需 `WHILE Load(global) < task_id` |
| 前驱可计算 | 依赖「同 shard task 序列 = `base + kS`」且步长恒定；若 kind 的路由让某些 `task_id` 不落到该 shard（如角色过滤后并非每个 `+S` 都真的会被 Attempt），`pred` 会指向一个**永不发布**的 task，导致死锁。**必须**确认：本 shard 上被 Attempt 的 task 集合就是 `{base, base+S, …}` 连续无缺，否则 B 不适用或需改用「实际前驱」映射 |
| 好处 | 唯一能给出「字面逐步推进（按 shard 步长）」强保证的方案；调试 / 校验 cursor 终值最干净 |

适用判断：当 PA 图对某 kind 保证「每个 shard 的 task 连续且必被
Attempt」时，B 提供最强不变量；否则优先方案 A（`atomicMax` +
组 loser 等水位），A 不要求前驱连续，只要求「更高 task 碰全局前
全局已覆盖当前 task」。

##### 方案 C — 禁止同 shard 上多 task 的 Claim 重叠

核心思想：从源头消除 §3.2 的跑飞——同一 shard 在任一时刻只允许
**一个** `task_id` 处于「正在 Claim」状态。任何核想 Claim 本
shard 的下一个 task，必须先看到上一个 task 已经产生 winner 并
关闭。等价于给每条 shard cursor 加一把「当前 open task」的门。

一种实现（每 shard 一个 `open_task` 门 + 组内两轮）：

```text
FUNCTION HierarchicalClaim_C(task_id, worker_id_in_role, shard):
    g = worker_id_in_role / COMPETITION_GROUP_SIZE
    gate          = &shard_open_task[shard]   // 当前允许 Claim 的 task_id
    group_cursor  = &claim_group_max[shard][g]
    global_cursor = &claim_max[shard]

    // 门禁：只有轮到本 task 时才参与竞争
    WHILE Load(gate) < task_id:
        PAUSE                                  // 前序 task 尚未收敛
    IF Load(gate) > task_id:
        RETURN { attempted: true, won: false } // 本 task 窗口已关闭且已有 winner

    // 此刻 shard 上只有 task_id 在竞争，可安全跑两轮 atomicMax
    old_g = AtomicMax(group_cursor, task_id)
    IF old_g >= task_id:
        RETURN { attempted: true, won: false } // 组内 loser（同 task，无跑飞）

    old = AtomicMax(global_cursor, task_id)
    IF old < task_id:
        // 唯一 winner：推进门到下一个 task，放行后继
        AtomicMax(gate, NextTaskIdOnShard(task_id))
        RETURN { attempted: true, won: true }

    AtomicMax(group_cursor, old)
    RETURN { attempted: true, won: false }     // 全局 loser（仍是同 task）
```

正确性直观：门 `shard_open_task` 保证任一时刻 shard 上只有一个
活跃 `task_id`，于是「组内 loser 立刻返回后抢下一个 task」的场景
根本不存在——下一个 task 的门尚未打开，loser 会卡在 `WHILE
Load(gate) < task_id`。winner 出现后才推门，天然逐 task 无洞。

代价与风险：

| 维度 | 说明 |
| ---- | ---- |
| 并发度 | 最严格：shard 内 task **完全串行**，同一时刻只有一个 task 在竞争；分层的组内并行只发生在「同一个 task 的候选核之间」 |
| 分层收益 | 大幅缩水：既然同时只有一个 task，组内轮筛掉的仍只是该 task 的 N 个候选核；跨 task 的重叠被门禁掉，等于放弃了「多 task 并行 Claim」这条并行度 |
| 门更新成本 | winner 额外一次 `AtomicMax(gate, next)`；后继核自旋等门 |
| 死锁风险 | 若某 task 因角色过滤等原因**无人 Attempt**，门永不推进，全 shard 卡死。C 同样要求 shard task 序列连续可发布；对「稀疏 task」需要 winner 之外的推门者或超时 |
| 好处 | 正确性最易证明（活跃 task 唯一）；适合作为「先正确、后放开」的保底实现，或跳号问题定位期的对照基线 |

C 与 B 的关系：B 只串行化**全局发布顺序**，组内两轮和不同 task
的组内筛选仍可并行；C 串行化**整个 task 的竞争窗口**，比 B 更
保守。性能预期 B > C，正确性论证 C 最简单。二者都要求 shard 上
的 task 序列连续且必被 Attempt。

#### 小节

1. 用户期望的「每次只能 +1」应弱化为：**同一 cursor 上的 task
   序列无洞推进**（步长为分片步长 `S`，不是字面 `1`）。
2. **§3 朴素两轮 atomicMax 不能保障每个 task_id 都有 winner**；
   存在组内/跨组跑飞导致的跳号反例。
3. 落地分层时必须加上 **「更高 task 碰全局之前，全局已覆盖当前
   task」** 的藩篱（至少方案 A）；并在定向测试中构造
   「组胜者延迟全局提交 + loser 抢先 Claim next」反例作回归。

### 3.3 方案 D — Flat + 投机 load 预筛（不分层）

方案 D 走另一条路：**不引入组层**，保留现网 flat 单 cursor 竞争，
只在每次 `atomicMax` **之前先 `Load` 一次** `claim_max`。若
`task_id <= claim_max`，说明本 task 已被更高（或同）水位覆盖，
**直接判负并跳过 `atomicMax`**；只有 `task_id > claim_max` 才真正
发一次 `atomicMax`。核心目标：**减少 `atomicMax` 的次数**，把
「注定失败」的核从热 RMW 线上摘掉，只留一次廉价的 read。

```text
FUNCTION FlatClaim_D(task_id, shard):
    global_cursor = &claim_max[shard]

    // 投机预筛：先读一次
    IF Load(global_cursor) >= task_id:
        // claim_max 单调，只会更大；真做 atomicMax 也必输
        RETURN { attempted: false, won: false }   // 跳过 atomicMax

    // 仅当可能获胜时才碰热线
    old = AtomicMax(global_cursor, task_id)
    RETURN { attempted: true, won: (old < task_id) }
```

#### 为什么 D 仍然正确（且不破洞）

1. **跳过等价于判负**：`claim_max` 单调不减。若读到
   `claim_max >= task_id`，则此后恒 `>= task_id`，真做
   `atomicMax` 也必得 `old >= task_id` → 输。跳过与做原子结论
   相同。
2. **至多一个 winner**：winner 仍走真正的 `atomicMax`，全局单线
   单调语义不变；预筛只删「必输」分支，不新增 winner。
3. **无洞（关键）**：D **不需要任何藩篱**就守住无洞——因为核离开
   task `T` 时，两条分支都已保证 `global >= T`：
   - 跳过分支：正因为 `Load >= T` 才跳过，全局已 `>= T`；
   - RMW 分支：`atomicMax` 已把全局推到 `>= T`。
   于是不存在「全局仍停在 `pred` 却已去 Claim `T'`」的
   §3.2 跑飞窗口。这与分层朴素版不同：分层的组 loser 可能在**组
   线**上早退而**没碰过全局**，D 的预筛读的正是**全局**线，天然
   闭合。

> 与 §3.2 对照：分层朴素版把「早退判断」放在**组** cursor 上，
> 与全局脱耦，才需要 A/B/C 补藩篱；D 把早退判断放在**全局**
> cursor 上，判负即已观测到全局覆盖，无需额外藩篱。

#### 统计口径

预筛跳过的核 `attempted == false`，应单列一类（例如
`claim.prefilter_skip`），区别于现网角色过滤的 `not_attempted`
与真正发起的 `claim.lost`；否则 ClaimMax 计数会与基线不可比。

---

## 4. 性能对比：Flat 分片 vs 分层 A / B / C vs 投机预筛 D

本章把**现网 Flat + 分片**、三种「可上生产」的分层方案
（A / B / C，均叠加在现有分片上）、以及不分层的 **投机预筛 D**
放在同一口径下比较。§3 的朴素分层因会跳号（§3.2），只作为
**成本下界参考**，不进入选型。

D 与 A/B/C 正交：A/B/C 靠**分组**缩小同 task 争用半径，D 靠
**先读后写**删掉必输核的 `atomicMax`。二者可叠加（见 §4.9）。

### 4.1 参数、模型与被比方案

符号：

| 符号 | 含义 | Vector 取值 | Cube | Alloc |
| ---- | ---- | ----------- | ---- | ----- |
| `N` | 同一 `task_id` 的候选核数 | ≤ 64 | ≤ 32 | ≤ 96 |
| `K` | 组大小 `COMPETITION_GROUP_SIZE` | 8 | 8 | 8 |
| `G` | 组数 `⌈N/K⌉` | 8 | 4 | 12 |
| `S` | 分片数 | 8 | 4 | 4 |
| `t_a` | 一次**无争用**原子 RMW 的时间 | — | — | — |

**争用模型（仅作方向性上下界，非硬件精确模型）**：

- 一条 AtomicLine 上有 `p` 个并发 RMW 时，近似串行化，排空耗时
  `≈ p · t_a`；
- 自旋读（`Load` 等待某事件）在等待期基本是 read-shared，
  **不往热 RMW 线注入额外写流量**，其代价是「等到事件」的墙钟
  时间，可与他核的 RMW 重叠；
- 下面用两个量刻画性能：
  - **同 task 裁定延迟 `L₁`**：一个 task 的 winner 被定下来所需
    时间（尾延迟视角，对应 loser_overhead 里的 ClaimMax）；
  - **单 shard 吞吐 `Θ`**：该 shard 每秒能 retire 多少个 task
    （流水线视角）。

被比方案（都含分片 `S`）：

| 代号 | 全局层机制 | 组 loser 行为 | 跨 task 是否串行 | 无洞正确性 |
| ---- | ---------- | ------------- | ---------------- | ---------- |
| **Flat** | 单轮 `atomicMax`（无组层） | 立即返回 | 否（自由流水） | 是 |
| **A** | 组内 + 全局 `atomicMax` | 自旋等 `global ≥ T` | 否（每 task 轻同步） | 是 |
| **B** | 组内 `atomicMax` + 全局前驱 CAS | 自旋等 `global ≥ T` | winner 链串行 | 是（最强） |
| **C** | 门 + 组内 + 全局 `atomicMax` | 门后立即返回 | task 窗口完全串行 | 是 |
| **D** | 单轮：`Load` 预筛 + 可能 `atomicMax`（无组层） | 无组层；必输核只读不写 | 否（自由流水） | 是（预筛读全局，天然闭合） |

### 4.2 同 task 争用半径（每条线并发 RMW 数）

分层的收益本质：把「一条线 N 路碰撞」拆成「组线 K 路 × 并行」
加「全局线 `N/K` 路」。

| 方案 | 组线并发 | 全局线并发 | 同 task 峰值单线并发 |
| ---- | -------- | ---------- | -------------------- |
| Flat | — | `N` | **`N`**（64 / 32 / 96） |
| A | `K`（G 条并行） | `⌈N/K⌉` | **`max(K, N/K)`**（Vector 8） |
| B | `K` | `⌈N/K⌉`（CAS，含失败重试） | `max(K, N/K)`，重试抬高全局线 |
| C | `K` | `⌈N/K⌉` | `max(K, N/K)`，但同刻只有一个 task |
| D | — | 峰值仍 `N`；只有「读到 `<T`」的核发 RMW | **`N`**（真同时到达）／ 随到达倾斜显著下降 |

Vector（`N=64, K=8`）：Flat 单线 **64** 路 → 分层任一线 **8** 路，
峰值争用半径约降 **8×**。Cube（32→ max(8,4)=8，约 4×）、
Alloc（96→ max(8,12)=12，约 8×）。

**D 与 A/B/C 的本质区别**：A/B/C 降低**峰值争用半径**（把一条线
拆成组线 + 全局线）；D **不改峰值**——若 N 个核真的同时到达，都
读到 `claim_max = pred < T`，于是全都发 `atomicMax`，峰值仍是
`N`（还各多一次 `Load`）。D 削减的是**有到达倾斜时的平均 RMW
数**：winner 一旦发布 `T`，之后才到的核读到 `>= T` 直接跳过。PA
回放里 loser 轻、进度不齐，同 task 候选核往往**不是**同时到达，
故 D 能摘掉大量「迟到必输」核的 RMW，但对「真并发浪涌」的峰值
尾延迟无能为力。

### 4.3 每核每 task 的原子 / 自旋成本

按结局拆分（一个 task 内，`p`=同线并发）：

| 结局（占比量级） | Flat | A | B | C | D |
| ---------------- | ---- | - | - | - | - |
| 组内 loser（`≈(K-1)/K`） | 1 atomic | 1 atomic **+ 自旋等 `global≥T`** | 1 atomic **+ 自旋等 `global≥T`** | 门自旋 **+** 1 atomic | 迟到必输：**仅 1 Load，0 atomic** |
| 全局 winner（每 task 1 个） | 1 atomic | 2 atomic（组+全局） | 1 atomic + 自旋到 `pred` + 1 成功 CAS | 门自旋 + 2 atomic **+ 推门 1 atomic** | 1 Load + 1 atomic |
| 全局 loser（组胜但全局败） | —（Flat 无组层，等同上面 loser） | 3 atomic（组+全局+回写） | 1 atomic + 失败 CAS/读若干 | 门自旋 + 2 atomic + 回写 | 早到但输：1 Load + 1 atomic |

要点：

- **平均原子次数**：Flat ≈ 1；A ≈ `1 + 1/K + 少量回写`；B 与 A
  相近但把全局换成 CAS（失败重试是主要变量）；C 最多（winner 多
  一次推门）。分层用「略增原子条数」换「大幅缩小单线争用半径」。
- **D 反向**：不缩争用半径，而是**直接减少 atomic 条数**——迟到
  必输核从「1 atomic」降为「1 Load」。平均 atomic 数
  `≈ 1 − p_skip`（`p_skip` 为预筛命中比例），是四个方案里唯一
  **平均 atomic < 1** 的；代价是每个真发 RMW 的核多一次 `Load`。
- **自旋**是 A/B/C 为**正确性**（无洞）付出的新成本，Flat 与 D
  都没有自旋（D 的预筛是一次非阻塞 `Load`）。

### 4.4 同 task 裁定延迟 `L₁`

用 §4.1 模型，取「同 task 全员同时到达」这一最坏并发：

| 方案 | `L₁` 量级 | Vector 代入（单位 `t_a`） |
| ---- | --------- | ------------------------- |
| Flat | `N · t_a` | **64** |
| A | `(K + N/K) · t_a`（组轮 + 全局轮串行） | **16** |
| B | `(K + N/K) · t_a` + 前驱依赖等待 | 16 + 链等待（前驱慢则拖尾） |
| C | 门等待 + `(K + N/K) · t_a` | 16 + 门等待（前序 task 未收敛则更久） |
| D | 真同时到达 `≈ N · t_a`；有倾斜时 `≈ m · t_a`（`m`=首个发布前的并发到达数） | 最坏 **64**，实际随倾斜下降 |

同 task 尾延迟排序：**A ≈ B ≈ C ≈ 16 ≪ Flat 64**（约 **4×**）。
差别在「附加等待」：A 只有组 loser 的轻自旋且与全局轮重叠，尾部
最干净；B 多了 winner 之间的前驱链，若某前驱 winner 慢会顺链
放大 P99；C 的门等待最重（要等整个前序 task 收敛）。

**D 不保证改善最坏 `L₁`**：真并发浪涌下 D ≈ Flat（甚至略差，多
一次 `Load`）。D 的 `L₁` 取决于「首个 winner 发布前挤进来的核数
`m`」：只有当 `m < N`（有到达倾斜）时才 `< N·t_a`。因此 D 是
**均值型**优化，A/B/C 是**峰值型**优化——这决定了二者在
loser_overhead 尾延迟问题上的定位不同。

### 4.5 跨 task 吞吐 `Θ`（单 shard）

关键差异在这里，不在 `L₁`：

| 方案 | 吞吐瓶颈 | `Θ` 量级 | 相对 Flat |
| ---- | -------- | -------- | --------- |
| Flat | 唯一全局线，每 task `N` 次 RMW 全走它 | `1 / (N · t_a)` | **1×** |
| A | 全局线每 task `N/K` 次；组线并行不成瓶颈 | `1 / ((N/K) · t_a) = K/(N·t_a)` | **≈ K×**（8×） |
| B | 全局 winner 链串行，每 task 至少一次有序 CAS + 重试 | `≈ 1 / ((N/K) · t_a)`，受前驱链约束 | **≈ K×**，尾部差于 A |
| C | task 窗口完全串行，每 task 付 `L₁` | `1 / ((K + N/K) · t_a)` | **≈ N/(K+N/K)×**（4×） |
| D | 唯一全局线，但每 task 只有 `m` 次 RMW（`m`=发布前到达数）+ `N` 次 `Load` | `≈ 1 / (m · t_a + N · t_load)` | **1× ~ 高倾斜时 `N/m`×** |

直觉：

- Flat 即便跨 task 自由流水，所有原子仍挤**同一条**全局线，
  线吞吐 `1/t_a` 被每 task `N` 次占用 → `Θ≈1/(N t_a)`。
- **A** 把全局线每 task 占用降到 `N/K`，且组线分散并行，因此吞吐
  约 **K×**——这是分层相对 Flat 的主要吞吐收益。
- **B** 吞吐量级同 A，但全局被强制有序发布，winner 链上任一慢核
  会阻塞后继，稳态吞吐略低于 A、尾部更抖。
- **C** 虽然每 task 的 `L₁` 已从 64 降到 16，但 task 之间**不能
  重叠**，丢掉了 A 的「组线跨 task 并行」，吞吐只有 `≈4×`
  Flat——仍胜 Flat（因为 Flat 单线本就把 task 串行化在一条线
  上），但明显低于 A/B。
- **D** 把每 task 的 RMW 从 `N` 降到 `m`（发布前挤进的核数），但
  同一条线上仍要跑 `N` 次 `Load`。若 `Load` 远比 RMW 廉价且倾斜
  大（`m ≪ N`），吞吐可逼近 `N/m ×`；若 `Load` 在热线上也接近
  RMW 代价、或近似同时到达（`m ≈ N`），收益趋近 0。D 的收益是
  **数据依赖**（取决于到达倾斜与 read/RMW 代价比），不像 A 的
  `K×` 那样结构性保证。

### 4.6 综合对比（Vector `N=64, K=8`）

| 指标 | Flat+分片 | A | B | C | D |
| ---- | --------- | - | - | - | - |
| 同 task 单线争用半径 | 64 | 8 | 8（+重试） | 8 | **64（峰值不变）** |
| 同 task 裁定延迟 `L₁` | 64 `t_a` | 16 `t_a` | 16 `t_a`+链 | 16 `t_a`+门 | 64 ~ `m·t_a`（数据依赖） |
| 单 shard 吞吐 `Θ`（相对） | 1× | **≈8×** | ≈8×（尾抖） | ≈4× | 1× ~ `N/m`×（倾斜相关） |
| winner 原子次数 | 1 | 2 | 2（+失败重试） | 3 | 1（+1 Load） |
| 平均 loser 原子次数 | 1 | ≈1 | ≈1 | ≈1 | **`1−p_skip`（可 <1）** |
| loser 新增自旋 | 无 | 组 loser 轻自旋 | 组 loser + winner 链 | 全核门自旋 | 无（仅 1 非阻塞 Load） |
| 无洞 / 每 task 有 winner | 是 | 是 | 是（最强，字面有序） | 是 | 是（预筛读全局，无需藩篱） |
| 额外 GM | 0 | `S×G` 组线 | `S×G` 组线 | `S×G` 组线 + `S` 门 | **0** |
| 死锁风险 | 无 | 无（只等水位） | 有（前驱须连续可发布） | 有（task 须必被 Attempt） | 无 |
| 实现 / 验证复杂度 | 最低 | 中 | 高 | 中（正确性最易证） | **最低（仅加一次 Load）** |

**选型建议**：

1. **默认 A**：在保住无洞正确性的前提下，同 task 尾延迟约 4×、
   单 shard 吞吐约 K× 改善，且不引入前驱连续性或 task 稠密性的
   死锁前提；新增成本只是组 loser 的轻自旋。
2. **B 仅当需要「字面按 shard 步长有序推进」的强不变量**（例如
   要用 cursor 终值做严格校验），且已证明该 shard 的 task 连续
   必被 Attempt；否则前驱链的尾部抖动与死锁前提得不偿失。
3. **C 作保底 / 定位基线**：正确性最易证明，但放弃跨 task 并行，
   吞吐只到约 4×；适合先上线拿正确性，再切 A 拿吞吐。
4. **D 作零成本首选叠加项**：改动最小（一次 `Load`）、零 GM、零
   死锁、正确性无需藩篱，**总应先加上**。但它是**均值型**优化，
   对「真并发浪涌」的峰值 `L₁` 无保证——不能替代 A/B/C。最佳组合
   是 **D + A**：D 摘掉迟到必输核的 RMW，A 把仍并发的核拆到组线
   / 全局线（见 §4.9）。

### 4.7 分层之后还要不要分片？

**要。** 分层与分片正交：

- **分片 `task_id % S`** 降低**不同 task** 撞同一线（跨 task
  false sharing）；
- **分层** 降低**同一 task** 的 N 路碰撞。

同 `task_id` 必落同 shard，故分片对同 task 的 N 路碰撞几乎无效；
反过来若把 `S=1`（只分层不分片），则**所有 task** 的组胜者又汇回
**一条**全局线，§4.5 的吞吐 `Θ` 会从 `≈K×` 掉回 Flat 量级，组线
也被跨 task 流量打满。加大组数 `G` 或加更多层都只优化「同 task
汇聚」，无法把不同 task 分到不同全局线——那本质就是分片。

因此以上 A/B/C 的 `Θ` 数字都**默认叠在现网 `S` 上**。若要动
`S`，须单独做消融并证明跨 task 全局线 / 组线 P99 不回退，不能
假设「有分层即可去分片」。

### 4.8 额外成本、观测与验收

**成本**：

1. **GM**：每 shard `G` 条 `AtomicLine`（cache-line 对齐）。
   Vector `S=8, G=8` ⇒ +64 条；C 再加 `S` 条门。放 shared
   sidecar 并保持对齐惯例。
2. **winner 多付原子**：A 2 次 / B 2 次+重试 / C 3 次；需确认不
   拖长 winner Submit（winner 才进 Materialize/Build）。
3. **自旋**：A/B/C 引入的等待要计入 Claim 父区间；用泳道量 P99，
   别只看中位。

**观测**：泳道若仍只打一个 `ClaimMax` 括号，需拆成
`GroupClaimMax` / `GlobalClaimMax` / `GroupSyncMax`（B 再加
`PredCas`、C 加 `GateWait`），否则无法区分「争用真的下降」还是
「时间搬到了自旋」。

**验收（对照 `loser_overhead.md`）**：空 `claim.lost` 的 Claim
中位约 0.7µs、其中 `claim_max` 约 0.3µs 且含插桩。分层要切的是
**高并发同 task** 时 ClaimMax 的尾部，而非把无冲突 loser 再压一个
数量级。门禁指标：

- 同 batch 多核同时 Claim 时 `GlobalClaimMax` / `PredCas` 的 P99
  是否按 `N → N/K` 下降；
- 组内早退比例是否接近 `(K-1)/K`；
- 自旋（`GroupSyncMax` / `GateWait`）没有把节省的时间吃回去；
- winner 数 / completion / 依赖签名相对基线**逐位不变**（正确性
  回归，尤其针对 §3.2 跳号反例）。
- D：`claim.prefilter_skip` 计数是否显著（`p_skip` 高才说明 D
  真省了 RMW）；同时确认预筛 `Load` 没有把热线读流量抬成新瓶颈。

### 4.9 方案 D 与 A/B/C 的叠加

D（先读后写）与 A/B/C（分组）互不冲突，可组合：

| 组合 | 迟到必输核 | 首个 winner 前的并发核 | 综合效果 |
| ---- | ---------- | ---------------------- | -------- |
| 仅 D | 只 Load，省 RMW | 仍 `N` 路 RMW 峰值 | 均值降、峰值不降 |
| 仅 A | 仍发组 RMW | 拆成组线 `K` + 全局 `N/K` | 峰值降、均值≈1 |
| **D + A** | 只 Load，省 RMW | 组线 / 全局线各再摘掉迟到核 | **峰值与均值同时降** |

在 D+A 里，预筛 `Load` 放在**组轮之前**：读全局 `claim_max`，
`>= task_id` 直接判负（连组 `atomicMax` 都省）；否则再进 A 的组内
/ 全局两轮。这样：

- 有到达倾斜时，迟到核连组线都不碰（D 的均值收益）；
- 仍并发的核被 A 拆到组线 + 全局线（A 的峰值收益）；
- 正确性：预筛读的是全局线，判负即已 `global >= T`，与 A 的组
  loser 等水位藩篱一致，不破洞。

因此推荐落地顺序：**先无条件加 D**（零成本、零风险），量
`p_skip` 与尾延迟；若峰值 `L₁` / ClaimMax 尾部仍高，再叠 **A**。

---

## 5. 实施要点（尚未落码）

1. **先落 D**（§3.3）：`Claim()` 在原子前加一次 `Load` 预筛，新增
   `claim.prefilter_skip` 统计。零 GM、零风险，可独立上线。
2. 在 `pa_model.h` 增加 `COMPETITION_GROUP_SIZE`、`claim_group_max`
   布局与 AICPU reset；选 B/C 时再加前驱映射 / `shard_open_task`。
3. `Claim()` 内用 §3 + 选定藩篱（A/B/C）替换单次
   `TraceAtomicFetchMax`；保留 kind / role 路由与
   `outcome.won = old < task_id` 语义。D 的预筛置于组轮之前
   （§4.9），构成推荐的 D+A 叠加。
4. 泳道按 §4.8 拆分组 / 全局 / 回写 / 自旋 / `prefilter_skip`
   站点，便于与扁平 ClaimMax A/B。
5. 定向测试：同组双核 `task_id` 序与乱序、跨组唯一 winner、回写后
   组内不再误冲、**§3.2 跳号反例回归**、与现网分片终态 cursor
   一致。
6. 性能门禁：在 R5ij b256 等同配置下对比 Claim / ClaimMax 中位与
   P99，并报告 winner Submit 是否回退。

---

## 6. 参考

- 现网 Claim：`common/pa_scheduler_core.h` → `Claim()`
- 分片与历史对照：`shared_tensormap_record.md`（S4.14 Vector
  cursor 等）
- Loser Submit 中 ClaimMax 占比：`loser_overhead.md`
