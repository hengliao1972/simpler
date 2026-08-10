# AIC 执行提前受阻问题分析

## 0. 结论先行

本文主体记录的是旧中央 Build ticket 阶段为提前执行 AIC task 所做的探索。
当前 S7 已删除该 ticket 和 Host Execute task-id 表，96 个 Scalar 先完整回放
真实 callback 并完成 Build，等 `replay_done == 96` 后才由 AIC/AIV runtime
cursor 扫描 task-indexed cell。因此，“每次领取新 Build ticket 前推进
Execute token”不再是现行路径，旧泳道与失败候选只能作为历史证据。

S7 的当前限制更加明确：Build 与 Execute 尚未 overlap，AIC task 会有意等到
真实 replay 封口后才开始执行。若后续用 winner 动态发布的通用 role queue
恢复 overlap，仍需面对本文识别出的非抢占边界：

> Execute owner 一旦进入另一个 task 的 Build，就必须连续完成该 Build 的
> Materialize、严格 Register、Fanin 和 WinnerBuild，期间没有新的调度点。
> 如果它持有的 AIC task 在这段时间内变为 `BUILT`，该 task 仍只能等到当前
> Build 完整返回后才会被再次观察和执行。

因此，当前不是“Execute 优先级没有写出来”，而是“Execute 优先级只能在
Build 之间生效，不能在一个长 Build 内生效”。

现有 B256 泳道中的 task 6 是直接证据：

```text
40.394 us  AIC Execute owner 首次观察 task 6，cell 尚未 BUILT
41.965 us  同一个 Scalar 领取并开始 Build task 36
76.457 us  task 36 进入严格 Register 等待
81.307 us  另一个 Build owner 已经发布 task 6 BUILT
101.313 us task 36 才结束 Register，继续 Fanin 和 WinnerBuild
113.502 us task 36 完成 BUILT 发布，当前 Build 即将返回
114.426 us AIC Execute owner 再次观察 task 6
118.135 us QK#6 kernel 开始执行
```

task 6 从 `BUILT` 发布到 owner 再次观察，相隔约 `33.119 us`；从 `BUILT`
发布到 kernel 开始，相隔约 `36.828 us`。task 6 是 QK，不存在需要等待前序
fanin 的原因。这段延迟也不是 AIC engine 启动指令本身造成的，而是其
Execute owner 正在不可抢占地完成 Build task 36。

后续若要真正让 AIC 执行提前，理论上只有两类主路径：

1. 在长 Build 内增加可证明正确的 Scalar 调度点；
2. 不在 task 尚未 `BUILT` 时过早绑定唯一 Execute owner，使已 `BUILT` task
   可以被当时空闲的同角色 Scalar 领取。

单纯增加 token、重复强调“Execute 优先”、减少某次轮询，均无法独立解决
上述根因。

## 1. 问题本身的现象

### 1.1 当前调度模型

当前 cross-core 版本有三条相互独立的工作发放流：

- 一条全 96 Scalar 共享的中央 Build ticket；
- 一条供 32 个 AIC Scalar 消费的 AIC Execute ticket；
- 一条供 64 个 AIV Scalar 消费的 AIV Execute ticket。

Build owner 与 Execute owner 完全解耦，但允许恰好是同一个核。每个 Scalar
最多保存四个 owner-local Execute token。Execute ticket 被领取后，对应 task
可能还没有完成 Build，此时 token 保持 `WAITING_BUILT`；看到 cell 进入
`BUILT` 后，owner 才执行 Claim、payload acquire、fanin 检查和 kernel。

这套模型的正确性边界记录在
[shared构建执行分离.md](shared构建执行分离.md)，主要实现位于
[pa_scheduler_core.h](common/pa_scheduler_core.h) 和
[pa_shared_submit_path.h](common/pa_shared_submit_path.h)。

### 1.2 代码已经实现了什么优先级

当前外层 Build dispatch 循环并不是无条件先 Build：

1. `DispatchOneSharedBuildTask()` 进入时，先判断本核是否有 Execute 推进工作；
2. 若有，则调用 `ProgressCrossCoreExec()`；
3. `ProgressCrossCoreExec()` 先遍历本核已有 token；
4. 任一 token 已可执行时，先执行 kernel；
5. 只有这次 Execute 推进返回后，才领取下一张中央 Build ticket。

所以，“如果已经站在一次 Build 的入口，应先 Execute 再 Build”已经落实。

缺口发生在下一层：中央 Build ticket 一旦领取，当前调用会同步、连续完成该
task 的整个 Build。Build 中间没有返回到外层调度循环，也没有调用
`ProgressCrossCoreExec()`。

简化后的控制流是：

```text
while Build 尚未发完:
    推进本核已有 Execute token
    领取一张 Build ticket
    Materialize
    等待严格 TensorMap 插入前驱
    Register
    Fanin
    WinnerBuild
    发布 BUILT
    返回外层循环

FinalDrain:
    排空所有 Execute token
```

其中 `Materialize -> Register -> Fanin -> WinnerBuild` 是当前不可抢占区间。

### 1.3 AIC 首次执行滞后的实测现象

当前诊断依据是
[cross_waitbuilt_1p084ms.json](../test_record/2026-8-4/cross_waitbuilt_1p084ms.json)。
该 B256 full-swimlane 的主要结果为：

- lifecycle：`1084.321 us`；
- Submit：`1022.595 us`；
- 1280 个 Build、1024 个 kernel、6528 次 DCCI 全部闭合；
- 最早 AIC kernel 为 `QK#16`，开始于 `113.609 us`；
- task 6 的 `QK#6` 开始于 `118.135 us`。

task 6 的完整关键路径如下。

#### 第一步：Execute owner 预领了尚未发布的 task

```text
40.394--40.707 us
core 12 / AIC Scalar 读取 task 6 execution cell
结果：尚未 BUILT，token 保持 WAITING_BUILT
```

此时 core 12 已经取得 task 6 的唯一 Execute 消费权。其他 AIC 即使空闲，也
不能再执行 task 6，否则会破坏 exactly-once。

#### 第二步：同一个 Scalar 开始另一个 task 的 Build

```text
41.965 us
core 12 取得 Build task 36
```

task 36 的主要区间为：

```text
48.493--51.930 us   ArgBuild
51.930--76.457 us   Materialize，约 24.527 us
76.457--101.313 us  Register，约 24.856 us
101.313--105.116 us Fanin
105.116--113.502 us WinnerBuild 与 BUILT 发布
```

Register 中的大头是严格插入前驱轮询。它保证 TensorMap metadata side effect
严格按 task id 发布，不能直接删除。

#### 第三步：task 6 已就绪，但 owner 没有调度机会

```text
81.029--81.307 us
另一个 Build owner 发布 task 6 BUILT
```

此时 task 6 已具备执行 payload。对于 QK，PA 当前依赖图中也没有额外 fanin
需要等待。但是 core 12 正在 task 36 的同步 Build 调用内，直到 `113.502 us`
才完成当前 Build。

因此，`81.307--114.426 us` 不是 task 6 自己的 Build、fanin 或 kernel 工作，
而是其 Execute owner 被另一项不可抢占 Build 占用的等待。

#### 第四步：返回调度边界后才执行

```text
114.426--114.656 us  owner 再次读取 task 6 cell
114.999 us           BUILT -> CLAIMED
115.286 us           payload invalidate
115.362--118.135 us  owner-local dispatch 重建
118.135 us           QK#6 kernel 开始
```

这证明执行提前的必要条件不是再加一个“Execute-first”外层判断，而是让
`81.307 us` 之后出现新的调度机会，或者让 task 6 在 `BUILT` 后由另一个空闲
AIC 取得执行权。

### 1.4 为什么其他 AIC 不能接手

AIC Execute ticket 是中央动态发放的，所以 task 6 最初并不绑定某个固定
候选核。但是一旦 core 12 领取了对应 ticket，唯一消费权就保存在 core 12 的
owner-local token 中。

这个 ticket 当前不能归还、转让或重复领取，原因包括：

- cursor 是单调 FetchAdd 发放，不能简单回退；
- 重复发放会产生两个 Execute owner；
- `BUILT -> CLAIMED` CAS 虽然能阻止双执行，但会把正常路径变成冲突恢复路径；
- token 还承载 FinalDrain 收口和错误归因，不能只丢 task id；
- A5 Scalar 间没有 cache coherence，跨核迁移 token 需要新的显式发布、获取和
  DCCI/Atomic 合同。

所以，中央 Execute ticket 解决的是“由全部 32/64 个同角色 Scalar 动态领取”，
不等于“领取后还能在任意时刻被任意同角色 Scalar 抢走”。

### 1.5 这不是什么问题

已有证据可以排除以下解释：

- **不是 AIC engine 启动指令慢。** 延迟发生在发射 kernel 之前，Scalar 正在
  执行另一个 Build。
- **不是 task 6 自身 fanin 未满足。** QK#6 没有需要继续等待的 producer。
- **不是同一 EfDrain 连续领取四张未来 Execute ticket。** S6.72 已把
  `EfDrain#80` 从 `39.361 us` 降为 `2.595 us`，最早 AIC 仍为
  `113.609 us`。
- **不是仅由 TensorMap 串行插入本身决定。** 严格 Register 确实拉长了 task 36
  的不可抢占 Build，但真正让 task 6 无法执行的是 owner 在此期间没有调度点。
- **不是 Build 和 Execute 必须同核。** 两者已经独立；反例只是动态调度下合法
  地落到同一核。
- **不是 FinalDrain 口径造成的假象。** 首个 AIC 启动时间直接来自泳道事件，
  与端到端终点选在哪里无关。

### 1.6 根因归纳

根因可以写成一条完整因果链：

```text
Execute ticket 可在 payload BUILT 前领取
-> 唯一 Execute owner 先被固定
-> owner 随后领取另一个 Build ticket
-> Build 是不可抢占的同步调用
-> 其他核发布原 Execute task 的 BUILT
-> owner 仍在 Materialize/Register/Fanin/WinnerBuild
-> 已 BUILT task 只能等待 owner 返回外层调度边界
-> AIC kernel 启动滞后
```

这里有两个结构性选择：

- **保留未来 ticket：** 必须让 owner 在长 Build 内有机会推进已有 token；
- **不保留未来 ticket：** 必须设计只发布 ready task、且不引入更大共享竞争的
  新领取协议。

## 2. 已尝试方案、收益与遗留问题

完整阶段记录位于
[PA调度器分离版实现过程.md](PA调度器分离版实现过程.md)。本节只抽取与
“AIC 执行能否提前”直接相关的尝试。

### 2.1 K2 固定候选改为 AIC/AIV 双中央 Execute ticket

#### 原问题

早期每个 task 只允许两个预定同角色候选核观察和执行。即使全局有 32 个 AIC
或 64 个 AIV，某个 ready task 仍可能等待自己的两个候选核，角色内部无法
动态均衡。

#### 修改

改为两条中央 Execute ticket：

- 所有 AIC Scalar 竞争 AIC task 流；
- 所有 AIV Scalar 竞争 AIV task 流；
- Build 与 Execute owner 独立；
- 每个表项由单调 ticket 唯一发放。

#### 结果

冻结 A/B 中，K2 基线中位数 `1.410142 ms`，双中央 ticket 中位数
`1.281905 ms`，改善 `9.094%`，12/12 对候选更快。

#### 仍未解决

它解决了“task 由哪一个同角色核领取”的负载不均，但 ticket 一旦被领取，
task 仍固定在该 owner 上。owner 后续进入长 Build 时，其他同角色核仍不能
接手。因此它是必要的基础改造，不是 AIC 首次执行滞后的完整解法。

### 2.2 owner-local token 从单槽扩展到四槽

#### 动机

单个 Execute token 若等待 `BUILT` 或 fanin，会阻止同核观察后续可能独立
ready 的 task。增加有界 token 可以让一个 Scalar 保存多个独立执行候选，
避免队首等待完全阻塞本核。

#### 已验证过程

- 两槽扩三槽：中位改善 `4.924%`，12/12 对更快；
- 三槽扩四槽：中位从 `1.076359 ms` 降至 `1.036396 ms`，改善
  `3.713%`，11/12 对更快；
- 三槽泳道中进入 FinalDrain 的 kernel 为 170 个，四槽降到 145 个。

#### 仍未解决

四槽改善的是 owner-local 有界前视和 backlog 排空。它无法让一个已经
`BUILT` 的 token 抢占正在运行的 Build。task 6 已在 core 12 的四槽之一，
问题不是“没有槽保存它”，而是“保存它的 owner 没有新的调度点”。

继续机械增加到五槽或更多，也不能从结构上解决这一点，还会增加每核状态、
扫描工作和代码体积。

### 2.3 新 token 未 BUILT 时停止同边界继续预领

#### 原问题

四槽版本曾在一次 EfDrain 中连续领取四张未来 Execute ticket。旧
`EfDrain#80` 的四次返回型 FetchAdd 分别为
`7.554/11.276/9.434/5.460 us`，合计 `33.724 us`；四个 cell 均未
`BUILT`，当次没有执行任何 kernel。

#### 修改

新领取的 token 第一次观察后若仍为 `WAITING_BUILT`：

- 保留该 token 的唯一消费权；
- 结束本次调度边界的继续取票；
- 后续独立边界仍可逐步填满四槽。

#### 结果

- `EfDrain#80` 从 `39.361 us` 降到 `2.595 us`；
- 冻结 A/B 中位从 `1.047032 ms` 降到 `1.010589 ms`；
- 中位改善 `3.481%`，11/12 对更快。

#### 仍未解决

这项修改删除了同一边界内无意义的未来票竞争，但 task 6 反例发生在“已经
保留一张未来票后，owner 去做另一个 Build”。因此 AIC 最早启动仍为
`113.609 us`。

### 2.4 删除新 Claim 后的立即 token 复扫

#### 思路

曾尝试减少一次看似重复的 owner-local token 扫描，希望降低 Scalar 控制开销。

#### 结果与问题

两种实现均未成立：

- 第一种使代码体积增加约 `10.32%`，8 对中仅 4 对改善，信号约
  `0.083%`；
- 第二种避免代码膨胀后，中位仍从 `1.385292 ms` 回退到
  `1.404504 ms`，回退 `1.387%`，8 对中仅 2 对改善；
- fanin load 中位还从 `30138.5` 增到 `30567`。

#### 结论

立即复扫不是纯冗余。一个 token 完成可能解开另一个 owner-local token；把
消费推迟到下一个外层边界或 FinalDrain，反而会扩大执行 backlog。这项修改已
撤回，也说明不能只按“少一次轮询”判断 AIC 是否会更早执行。

### 2.5 BUILT-ready 位图与批量 bit-CAS

#### 思路

未来 ticket 的根本问题是 task 尚未 `BUILT` 时就固定 Execute owner。
BUILT-ready 位图尝试反过来：只有 Build 已完成的 task 才进入可领取集合，
executor 从 ready 位中 Claim，理论上不会存在 `WAITING_BUILT` owner 被长
Build 占住的问题。

#### 优点

- Execute owner 只领取真正已发布的 task；
- ready task 可由当时空闲的同角色 Scalar 消费；
- 直接打到了当前根因的第二条主路径。

#### 实际问题

首版生产路径固定扫描 `4 × active_words`，并引入共享 ready word 的发布、
扫描和 bit-CAS：

- 工作量随 active word 数和 task 分布变化；
- 多核会反复触碰共享位图 cache line；
- Build closure 与 scanner 终止之间还需要额外合同，避免漏掉最后发布的 ready；
- builder hint 与 closure 后重读只能修复该协议内部问题，脱离 ready 位图没有
  独立价值；
- B256 首版端到端回退到 `1.834--1.924 ms`。

#### 结论

该版本已撤回。它证明“延迟 Execute ownership”在协议上可行，但当前位图扫描
实现的共享访问和固定扫描成本远大于收益。未来如果重做，不能原样恢复旧提交。

### 2.6 每 task 严格插入完成字独占 128B

#### 思路

严格 Register 会等待 task `N-1` 的 insert-completion。将每 task 完成字隔离到
独占 128B Atomic 冲突单元，减少相邻 task 完成字之间的地址冲突。

#### 结果

冻结 A/B 中位改善 `12.282%`，已经保留。它同时缩短了部分 Register 等待，
间接减小 Build 不可抢占窗口。

#### 仍未解决

它只让当前同步 Build 更短，没有在 Build 内产生调度点。只要 Register 仍可能
等待数微秒到数十微秒，已 `BUILT` 的 owner-local AIC task 仍可能被挡住。

### 2.7 双 Execute cursor 独占 128B

#### 思路

将 AIC/AIV 两条中央 Execute cursor 分别隔离，希望降低不同角色之间的 Atomic
地址干扰。

#### 结果

12 对端到端 A/B 的置信区间跨零，没有稳定收益，已恢复为两个 64B cursor。

#### 结论

这属于 Atomic 布局微调，也不改变 Execute owner 被长 Build 占用的控制流，
不是当前根因的解法。

### 2.8 当前正式保留的状态

截至 S6.73，设备代码只保留三项已完成单变量冻结 A/B、且不读取 PA task kind
或固定 DAG 的通用优化：

1. 每 task insert-completion 独占 128B；
2. 每 Scalar 四个 owner-local Execute token；
3. 新 token 未 `BUILT` 时停止本调度边界的连续预领。

收敛后三次 B256、real-compute `6,28,4,1`、startup 到 FinalDrain 结束的
perf-clock 为：

```text
982.876 us
997.605 us
1023.453 us
```

中位数为 `997.605 us`。这说明整体端到端已经接近 1 ms，但不能据此声称
AIC 首次执行问题已经解决。当前最直接的 full-swimlane 证据仍显示最早 AIC
到 `113.609 us`，且 task 6 存在完整的 owner 被 Build 占用反例。

## 3. 未来可能尝试的方案

### 3.1 任何新方案都必须保持的合同

后续不能只追求“让第一条 AIC 更早”，还必须保持公共调度器的正确性与泛化性：

1. 每个逻辑 task 恰好 Build 一次；
2. 每个可执行 task 恰好 Execute 一次；
3. TensorMap metadata 仍严格按 task id 插入；
4. Build payload 发布顺序继续保持：普通写、DCCI OUT + DSB、Atomic 发布
   `BUILT`；
5. executor 继续保持：取得唯一所有权、invalidate/acquire、再读取 payload；
6. A5 Scalar 间没有 cache coherence，跨核共享状态不能依赖普通 cache 可见性；
7. FinalDrain 必须证明所有 task 为 `DONE`、所有 token 回到 `IDLE`；
8. 公共调度器不能识别 Alloc/QK/SF/PV/UP 等 PA task kind；
9. 不把 Scalar 调度与 engine 执行 overlap 重新带回目标。本问题只讨论
   Scalar Build 与 Scalar 发起 Execute 之间的调度。

还要继续使用同一证据链：CPU 协议门槛、CCEC 构建与代码体积、A5 B1/B256
正确性、full-swimlane 归因、冻结 A/B 端到端性能。不能只看第一条 kernel
时间而忽略总周期和共享访问增量。

### 3.2 已实测否决：在严格 Register 纯等待中同步执行 kernel

边界已重新对齐：不允许 kernel 执行过程中让出 Scalar，也不在
Materialize/Fanin 等实际计算段插调度点；但 Register 确认前序 metadata 尚未
发布后的纯 Atomic 轮询间隙，可以同步执行一个本核已经持有的 Execute token，
kernel 与 completion 完成后再返回当前 Build。

#### 基本思路

当前 `WaitForSharedTaskInsertTurn()` 在一个循环内持续 Atomic load 前驱完成字。
可以把无限连续轮询改为有界批次：

```text
轮询 predecessor 最多 K 次
若 predecessor 已完成:
    继续当前 Build 的 Register
若尚未完成:
    每 K 次 pending 轮询只检查一个本核已经持有的 Execute token
    不领取新的 Execute ticket
    再继续轮询 predecessor
```

它针对 task 6 的反例：core 12 在等待 task 35 insert-completion 时，若发现自己
持有的 task 6 已 `BUILT`，可以先执行 QK#6，再回来继续 task 36 Register。

#### 为什么第一版只推进已有 token

在 Register 内继续领取新的 Execute ticket 会扩大重入范围，并可能再次囤积
未来 task。第一版只服务已经归本核所有的 token，目标单一：缩短
“task 已 BUILT 到 owner 再观察”的间隔。

#### 需要证明的问题

- Register 等待期间执行 kernel，不得修改当前 Build 的 `TaskArgs`、
  `SubmitContext`、`SharedTaskWriterDelta` 和 materialize reservation；
- 当前 Build 的局部对象都还在栈上。嵌套调用 Execute 会增加栈深和代码体积，
  必须检查 CCEC stack 与 `.text`；
- Execute completion 可能访问 shared heap、completion flag 和 worker state，
  要证明不会覆盖当前 Build owner 尚未发布的 task-local 状态；
- 轮询批次 K 不能通过 PA 时间常量脑补，应先以现有 Register poll count 分布
  选择少量候选；
- 新调度点本身会增加分支、函数调用和 owner-local token 扫描，若等待本来很短，
  可能得不偿失；
- 不能在持有已经对其他核可见、但尚未闭合的共享 metadata 临界状态时让出。
  当前最合适的点是“writer delta 已在本地准备完成、前驱尚未 ready、当前 task
  尚未开始 metadata commit”的等待段。

现有 B256 的 1,279 次等待中，pending-load 中位数为 `20`、p75 为 `32`；
达到 16 次的等待覆盖 `93.6%` 的轮询 load。首轮因此只试 `K=16`，且每次
轮转检查四个 owner-local token 中的一个，不调用完整 token 扫描器。

该版 CPU、CCEC 与 A5 正确性全部通过，也确实把首个 AIC kernel 从旧证据约
`113.609 us` 提前到 `56.909 us`。但 826/1024 个 kernel 被搬入 Register
等待后，前序等待中位从约 `5.647 us` 放大到 `178.413 us`，p95 达
`344.486 us`；trace-free 端到端从同窗 `996.663 us` 回退到
`2985.751 us`。

原因是同步 kernel 通常长于触发点之后剩余的 predecessor 等待。kernel 运行
期间前序即使发布，当前 owner 也不能交出自己的 insert completion，延迟会沿
严格插入链传播。固定 K 不能预测剩余等待，因此该方向已撤回，不再通过枚举
阈值延续。否决泳道保存在
`test_record/2026-8-6/register_wait_exec_k16_rejected.json`。

#### 首轮验收指标

- task 6 的 `BUILT -> owner reobserve` 是否从约 `33.119 us` 明显下降；
- 最早 AIC kernel 是否早于当前 `113.609 us`；
- TensorMap 插入顺序、Build/Execute exactly-once 与 FinalDrain 是否全通过；
- Atomic/DCCI 次数是否不增加，或增加量能被明确解释；
- CCEC `.text` 和 stack 是否出现明显膨胀；
- B256 startup 到 FinalDrain 的冻结 A/B 是否有净收益。

### 3.3 已排除：把 Build 改为可恢复状态机

这一节同样只保留为历史设计空间。显式保存 Build continuation 仍属于给长任务
增加协作式让出能力，当前不实施。

#### 最小状态划分

可以只在第一个真实阻塞点切分，而不是一开始就把全部 Build 阶段协程化：

```text
NoPendingBuild
-> MaterializedAwaitInsertTurn
-> PublishWriterMetadata
-> Fanin
-> WinnerBuild
-> PublishBuilt
-> NoPendingBuild
```

一个 Scalar 在 `MaterializedAwaitInsertTurn` 发现前驱未完成时：

1. 把恢复所需的最小状态保存到 owner-private `PendingBuild`；
2. 返回外层调度器；
3. 优先推进已有 Execute token；
4. 后续再恢复当前 Build，而不是领取第二张 Build ticket。

#### 必须保存什么

只保存恢复确实需要的值，例如：

- task id 与静态 plan 索引；
- materialize 结果和 completion-vend 快照；
- 已准备完成的 writer delta；
- 当前阶段枚举；
- 后续 Fanin/Build 所需但不能从 immutable plan 重建的最小上下文。

不应保存栈地址、指向临时对象的指针或隐含 builder 核身份的引用。

#### A5 内存模型注意点

`PendingBuild` 若只有原 owner 访问，可以是 owner-private GM 状态，不需要为了
跨核可见性发布；但它仍是 GM 访问，可能引入 D-cache miss。应尽量压缩到少量
cache line，并在恢复前考虑有证据的 preload。若未来允许其他核接管 Build，
则必须重新定义 DCCI publish/acquire，不能直接复用 owner-private 版本。

#### 优点与风险

优点是调度边界清晰，不需要在深层 helper 中重入完整执行路径；风险是新增
持久状态、分支和恢复逻辑，容易扩大代码体积，并使错误收敛和 FinalDrain
多出一个“尚未完成 Build”的状态。

### 3.4 第三候选：只在 BUILT 后分配 Execute owner

这个方向从根上避免“未来 task 先绑定忙 owner”。旧 ready bitmap 已证明原理
可行，但实现成本不合适。未来若重做，应换一种数据结构。

#### 可研究的最简原型

在 standalone 中先做不复用容量的 per-role ready queue：

- AIC 与 AIV 各有一份追加数组；
- Build owner 发布 immutable payload 后，将 task id 追加到对应角色队列；
- producer 对队列元素完成 DCCI，再用 Atomic 发布 tail/ready；
- 任一同角色 executor 通过中央消费 ordinal 取得一个已发布条目；
- consumer acquire/invalidate 后读取 task id 和 payload；
- 本轮先固定容量，不处理复用、环回和反压。

#### 主要难点

- Build 完成可以乱序，不能假设队列 ordinal 等于 task id；
- 多 producer 追加必须避免“先拿到位置但 payload 尚未发布”形成 hole；
- 若为每个条目增加 ready word，会增加 Atomic、DCCI 和内存占用；
- 若用单一 tail 同时表达 reservation 和 publication，会再次出现发布空洞；
- 多个 executor 竞争中央消费 cursor，可能把当前未来-ticket 竞争换成新的热点；
- 错误关闭时必须证明最后一个 ready 不会被漏消费；
- 旧位图版本的 `1.834--1.924 ms` 回退说明，共享扫描或发布成本不能被忽略。

该方向只有在新协议能用 O(1) 消费、避免全量扫描，并保持较少共享 cache line
访问时，才可能打败当前中央 future-ticket 方案。

### 3.5 第四候选：调整 Build admission，给已有 Execute owner 留出响应机会

可以研究一种通用 admission 规则：某核持有 Execute token 时，是否应限制它
领取新的长 Build。

但简单规则都存在明显问题：

- “只要有 `WAITING_BUILT` 就不 Build”可能让 32 个 AIC 很快全部停工；
- Build 仍需要 96 Scalar 并行推进，否则 payload 生产变慢，AIC 反而更晚 ready；
- 如果所有持票核都等 Build、所有 Build 又被限流，可能形成活性问题；
- 固定保留若干 AIC 只 Execute 会降低通用 Build 能力，并依赖角色比例和工作负载；
- 以 PA task kind 决定谁 Build 不具备公共调度器的泛化性。

因此该方向更适合作为诊断上界：例如暂时保留少量 AIC 不领取 Build，观察首个
AIC 与端到端变化，用来估计“执行响应延迟”的理论收益；不能在没有进度证明和
通用负载信号时直接成为正式策略。

### 3.6 第五候选：继续缩短严格 Register 的不可抢占时间

每 task insert-completion 独占 128B 已经证明有效。还可以继续从通用角度减少
严格串行段内的冗余 GM 访问和返回型 Atomic，但必须清楚：

- 这会缩短阻塞窗口；
- 它不会创建新的调度点；
- 即使平均 Register 很短，尾部等待仍可能挡住已经 `BUILT` 的 AIC task。

可以继续审计的项目包括：

- 前驱完成字是否存在重复 load；
- writer delta 中能否把纯本地计算全部提前到串行区外；
- 空 writer 集合是否仍只做维持严格序号所需的最小发布；
- metadata commit 后的统计是否已经全部移出串行区；
- Register 前是否能根据一次普通、非阻塞观察选择更合适的 Build 工作，但不能
  跳过严格插入合同。

用户已经否决 grouped drainer，因此后续不把它作为当前实施候选重新提出。

### 3.7 已排除：完整 Scalar continuation

完整 continuation 可以在 Materialize、Register、Fanin 等多个边界保存上下文，
让 Scalar 随时在 Build 与 Execute 间切换。这是表达力最强的方案，也是当前
复杂度最高的方案：

- 每个阶段都需要明确可恢复状态；
- owner-private GM 上下文会增大；
- stack-local 优势会减少；
- D-cache、I-cache 和分支开销可能抵消调度收益；
- fatal、FinalDrain、heap reservation 与 partially published metadata 都要新增
  状态机证明。

当前不实现完整 continuation，也不先实现 Register 单点让出；这两条路径均已
从当前性能优化计划删除。

### 3.8 当前实施顺序

当前只推进不引入协作式让出或 Build continuation 的方向：

1. **先消减已有同步路径的确定性常数。** 只处理重复 GM 访问、重复诊断统计和
   可有界降频的正常态 Atomic；每项单变量构建并做冻结 A/B。
2. **继续缩短严格 Register 串行段。** 只移动或删除不属于顺序提交合同的工作，
   不在轮询中嵌套 Execute，也不保存待恢复 Build。
3. **若需要改变 Execute ownership，再独立设计 ready 发布协议。** 先在
   standalone 证明乱序 producer、无发布空洞和同角色动态消费；不能借此夹带
   长任务让出机制。
4. **每项均检查 CCEC 和 A5。** CCEC 精确核对 `.text`、stack、finish relocation；
   A5 先闭合业务和终态，再用 startup 到 FinalDrain 的交错 A/B 裁决。
5. **泳道只做归因。** 继续记录 task 6 类 `BUILT -> reobserve` 间隔，但不以
   “首条 AIC 更早”代替完整周期收益。

每一步都必须同时记录：

- 最早 AIC kernel 时间；
- `BUILT -> owner reobserve -> kernel start` 三段延迟；
- Register 等待与 Build 不可抢占区间；
- 进入 FinalDrain 的 kernel 数量；
- Atomic 与 DCCI 次数是否变化；
- startup 到 FinalDrain 的端到端分布；
- CPU/CCEC/A5 正确性门槛和代码体积。

## 4. 当前判断

当前 AIC 无法进一步提前，不是单条 Atomic 或某个 PA 特例导致的孤立问题，而是
调度粒度与 Execute ownership 时机共同造成的结构性问题。

当前不通过 Register 内让出或 `PendingBuild` continuation 解决该问题。近期只做
确定性的同步常数消减与严格串行段瘦身；若这些方向到达收益上限，再单独评估
Build 完成后发布到同角色动态 Execute 队列的协议。

无论采用哪一条路径，都应牢记：外层的 Execute-first 已经存在。若未来继续处理
首个 AIC 滞后，必须设计一种协议来改变“task 在 `BUILT` 前就绑定唯一 owner”
的现状；否则外围常数优化只能改善端到端时间，不能消除 AIC 已 ready 但仍被
长 Build 阻塞的核心现象。当前接受这一边界，不用协作式让出点强行解决。

## 5. 2026-08-06 补充：Register 后有限机会也已实测否决

在不触碰严格插入链的前提下，又验证了一个比 Register-wait 更克制的候选：
task N 发布 insert completion 后、Fanin/BUILT 前，推进至多一个已有 token。
它不领取新 Execute ticket，固定四 token 中选择最老 task，理论上不会把 kernel
阻塞传播给 N+1 的 Register。

机制按预期工作：B256 泳道有 208 个 kernel 在该机会点完成；但它们全部来自
常规 EfDrain 的位置迁移，FinalDrain 仍为 122。每个 1280 Build 都新增二次
split Finish、continuation 交接和 token 扫描，且执行旧 task 会延后当前 task
的 Fanin/BUILT。6 对冻结反转 A/B 的端到端中位由 `1011.904 us` 回退到
`1035.376 us`，候选全部六对均更慢。

这进一步收窄了问题边界：既不能在严格 Register 等待中同步执行，也不能在
Register 后对每个 Build 无条件插入一次检查。后续若继续解决 AIC 提前问题，
候选必须做到至少一项：

- 只在高置信 ready 且会进入 FinalDrain 的任务上付费；
- 把机会触发成本从“每 Build”降为“每实际 ready 事件”；
- 改变 BUILT 前绑定唯一 Execute owner 的协议，使任意空闲同角色 Scalar 能
  O(1) 取得已发布任务。

在出现这种新协议前，现行实现保持外层 Execute-first，不再恢复两种 Build 内
检查点。完整实验见
`PA_Build_Phase_Aware_Execute_Opportunity_Proposal.md`。
