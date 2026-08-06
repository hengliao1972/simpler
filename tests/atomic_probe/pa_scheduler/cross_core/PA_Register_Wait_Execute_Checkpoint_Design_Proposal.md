# PA Cross-Core Scheduler 优化方案设计

## Register Wait Execute Checkpoint

版本：Proposal v6（A5 实测后否决）

状态：**REJECTED，不属于现行生产协议。**

## 1. 背景

当前 cross-core scheduler 已完成 Build / Execute 分离、Execute
token、AIC/AIV Execute cursor 以及 BUILT -\> CLAIMED 仲裁。

剩余问题：

> Execute obligation 已经存在，但持有该 obligation 的 Scalar 随后进入长
> Build，导致 BUILT 后无法及时推进 Execute。

典型路径：

``` text
Execute ticket
      |
      v
WAITING_BUILT(task6)
      |
      v
Build task36
      |
      v
task6 BUILT
      |
      v
等待调度边界
```

问题本质：

> Build 内部等待期间，没有利用已有 Execute obligation 推进机会。

## 2. 目标

保持：

-   Execute token；
-   BUILT -\> CLAIMED CAS；
-   payload publication；
-   TensorMap；
-   deterministic replay。

不引入：

-   ready queue；
-   bitmap；
-   ownership transfer；
-   task discovery scanner；
-   Build continuation。

## 3. 核心方案

增加：

# Execute Checkpoint

在 Build 内部已有等待点提供轻量 Execute 推进。

从调度语义看，它确实是一次有边界的：

``` text
Build(Register 等待)
    -> Execute 一个本核已持有的 token
    -> 返回同一 Build 的 Register
```

但它不需要把 Build continuation 发布到 GM：当前 `TaskArgs`、
`SubmitContext` 和 writer delta 仍保留在同一 Scalar 调用栈中。kernel 一旦
开始必须同步执行到 completion 发布结束，中间不再让出 Scalar。

## 4. 首选位置

首选：

``` text
WaitForSharedTaskInsertTurn()
```

当前：

``` cpp
while (!previous_task_done)
{
    atomic_load();
}
```

修改后的顺序必须是：

``` cpp
while (!previous_task_done)
{
    observed = atomic_load(previous_task_done);
    if (observed == ready)
        break;
    if (observed != pending)
        fail_closed();

    if (++pending_polls % K == 0)
        ProgressOneOwnedExecuteToken();
}
```

不能把 Execute 检查放在前序 atomic load 之前。前序已经 ready 时必须直接
进入 Register，不为短等待额外读取 Execute cell；只有刚刚确认 pending 后，
才允许使用这一段本来会继续空转的时间。

## 5. ProgressWaitingExecute 范围

允许：

-   每次只选择一个 owner-local token；
-   对已有 `WAITING_BUILT` token 检查一次 `SharedExecCell`；
-   `BUILT` 后执行 `CLAIM`；
-   对已经进入 `WAITING_FANIN` 的 token 只检查一次 ready 前缀；
-   fanin 已 ready 时同步执行 kernel 和 completion 发布。

禁止：

-   创建新 Execute ticket；
-   创建新 Build ticket；
-   在 checkpoint 内循环等待 fanin；
-   扫描全局 task/cell；
-   批量 drain 全部 owner-local token；
-   递归进入完整 scheduler。

第一次实现使用四个 owner-local token 的轮转游标；一次检查点只访问一个
slot。它不能调用会反复扫描四槽、并在完成后重扫的
`ProgressCrossCoreOwnedTokens()`。

## 6. 状态模型

保持：

``` text
BUILDING
BUILT
CLAIMED
DONE
```

不新增：

``` text
TRANSFERABLE
EXEC_PENDING
```

原因：

当前问题不是 ownership transfer，而是调度机会不足。

## 7. task6 预期

旧：

``` text
40us WAITING_BUILT

41us Build task36

81us task6 BUILT

114us 返回

118us kernel
```

新：

``` text
40us WAITING_BUILT

41us Build task36

Register wait 中执行 checkpoint

81us task6 BUILT

82~90us claim + kernel
```

现有 B256 泳道中 1,279 次前序等待的 pending-load 数量为：中位 `20`、
p75 `32`、p90 `90`。达到 16 次的等待占事件数 `64.0%`，覆盖全部轮询
load 的 `93.6%`。因此第一版固定 `K=16`：短等待不检查 Execute，长等待
约每 4--5 us 获得一次单槽推进机会。该值是首轮实测参数，不是最终常量。

## 8. 正确性

Exactly once：

``` text
BUILT -> CLAIMED
```

继续由 CAS 保证。

Payload：

``` text
payload write
-> DCCI flush
-> BUILT publish
-> CLAIM
-> invalidate/acquire
-> kernel
```

不改变。

当前 Build 的 writer metadata 在等待期间尚未开始提交；已经完成的
Materialize output publication 也有独立 task cell。执行另一个 token 不会
让出一个“已写一半”的 TensorMap 事务。返回后仍由当前 Build owner 完成
metadata commit 和本 task insert-completion handoff。

Deterministic replay：

不改变 task id、TensorMap、heap。

## 9. 不采用其他方案原因

### Ownership Transfer

需要额外 task discovery。

### Ready Queue

引入新的共享调度结构和 publication 协议。

### Bounded Revisit Window

需要定义 window、revisit、retire、FinalDrain。

### Build Admission Control

需要 builder quota 和活性合同。

## 10. 实施计划

Phase 0：

-   用已有 B256 raw 确定首轮 `K=16`；
-   复用已有 cell-state、CLAIM、fanin、kernel、completion 事件；
-   复用已有 Kernel/Commit 作为成功检查点边界，不增加新的 TracePhase，
    也不增加逐 poll raw 字段；
-   Register 父区间允许包含该同步 Execute，但前序 poll 的解释必须与嵌套
    Execute 分开，不能把 kernel 时间冒充 atomic latency。

Phase 1：

CPU standalone：

验证 checkpoint 不破坏：

-   Build；
-   Execute；
-   FinalDrain。

还要定向验证：前序首读已经 ready 时零检查；第 16 个 pending load 才检查
一个 token；fanin 未 ready 只保留 token 并立即返回 Register 轮询；检查点
不得推进 Execute 中央 ticket cursor。

Phase 2：

CCEC/A5：

验证：

-   BUILT publication；
-   CLAIM CAS；
-   payload acquire。

CCEC 当前将 `FinishSharedWinnerSubmitBody()` 放在 split finish TU，而真实
Cube/Vector dispatcher 由 caller TU 定义。检查点进入 Finish 后，finish object
将合法导入同角色 dispatcher；构建脚本必须验证这一新依赖只出现一次、核型
匹配，并继续拒绝 finish 拥有 launch metadata 或 orchestration。

Phase 3：

B256 full-swimlane：

比较：

-   first AIC kernel；
-   BUILT -\> kernel latency；
-   total runtime。

## 11. 风险

### checkpoint 不覆盖主要等待

本轮不把检查点扩展到 Materialize 或 Fanin preparation。二者是实际 Scalar
计算/GM 工作，不是纯等待；若 Register 等待没有覆盖主要滞后，应撤销或另做
调度设计，不能靠向正常计算段扩散检查点来掩盖。

### checkpoint 过频

`K=16` 会增加 owner-local token control 读取及偶发 cell-state atomic。必须用
冻结 A/B 同时比较端到端、Register 等待、fanin load 和 Atomic 次数；第一条
AIC kernel 提前但端到端稳定回退，不能判为有效。

### 严格插入链被 kernel 延迟

检查点执行 kernel 时，前序 TensorMap 插入可能已经完成；当前 Build owner
仍要等同步 kernel 返回后才能继续 Register。因此收益是“用原空转时间做必要
Execute”，代价是可能把一部分等待转移到后继 insert owner。必须同时观察
严格插入链尾和完整 FinalDrain，不能只看 task 6。

### CCEC 栈和代码体积

Finish 调用栈上仍保留 Materialize/SubmitContext/writer delta，再进入完整
Execute adapter。CPU 通过不能证明 A5 栈足够；CCEC 构建需继续固定 32 KiB
stack，并记录 AIC/AIV finish `.text` 及是否出现 stack overflow。

### 递归调度

必须禁止：

``` text
checkpoint
 -> execute
 -> build
 -> checkpoint
```

## 12. 最终判断

当前首轮假设不是：

-   Execute ticket 数量不足；
-   task discovery 不足；
-   ownership transfer 缺失。

而是：

> Build 长等待阶段没有利用已有 Execute obligation。

因此待验证的最小结构性修正是：

``` text
Build wait
    |
    v
Execute checkpoint
    |
    v
BUILT -> CLAIMED -> kernel
```

一句话：

> 不重新设计 scheduler，只让现有 scheduler 在已有等待点获得推进 Execute
> 的机会。

## 13. A5 验证结论

第一版按本文合同实现：先读 predecessor、确认 pending、`K=16`、一次只检查
一个本地 token、不领新票、kernel 同步执行到底。CPU 定向交错证明了 ready
首读零检查、第 16 次 pending 才推进、fanin 未 ready 立即返回；CCEC 也以
唯一 noinline helper 闭合了 split finish 到同核型 dispatcher 的调用。

B256 功能全部通过，但性能明确失败：

```text
修改前 trace-free 单次： 996.663 us
K=16 候选 trace-free：  2985.751 us
K=16 full-swimlane：     3292.715 us

RegisterCheckpoint kernel：826 / 1024
首个 AIC kernel：约 113.609 us -> 56.909 us
Register wait 中位：约 5.647 us -> 178.413 us
Register wait p95：约 30.056 us -> 344.486 us
```

QK/SF/PV 同步 kernel 约 28--54 us；检查点触发后，前序等待通常只剩数微秒。
kernel 期间 predecessor 即使已经发布，当前 Build 也必须等 kernel 返回才能
交出本 task completion，延迟因而沿严格插入链传播。该方案局部改善首个 AIC，
却把必要执行工作串进全局 metadata 顺序链，端到端约扩大到三倍。

固定 `K` 无法预测剩余等待时间。继续增大 `K` 只会减少触发并逐渐退化成旧
路径，不能消除“同步 kernel 阻塞严格串行链”的结构矛盾。因此不再枚举
32/64/128，生产代码、专用 placement 和 split-TU 依赖全部撤回；仅保留本文
与 `test_record/2026-8-6/register_wait_exec_k16_rejected.json` 供后续分析。
