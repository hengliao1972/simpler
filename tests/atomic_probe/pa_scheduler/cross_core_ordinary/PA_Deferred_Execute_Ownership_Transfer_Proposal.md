# PA Cross-Core Scheduler 优化方案设计

## Deferred Execute Ownership Transfer

版本：Proposal v1

------------------------------------------------------------------------

## 1. 背景

当前 cross-core scheduler 已经完成：

-   Build / Execute 分离；
-   Build owner 与 Execute owner 解耦；
-   AIC/AIV 分角色 Execute cursor；
-   owner-local Execute token；
-   四槽 token 前视；
-   未 BUILT 时停止继续预领。

仍存在的问题：

> Execute ownership 发生在 task BUILT 之前。

当前流程：

    Execute ticket
          |
          v
    Assign owner
          |
          v
    WAITING_BUILT
          |
          v
    BUILT
          |
          v
    Execute

导致：

    core12:

    领取 task6 execute token

    task6 未 BUILT

    进入 Build task36

    task6 BUILT

    core12 无调度机会

    task6 等待 owner 返回

典型时间线：

    40.394 us
    core12 观察 task6

    41.965 us
    core12 Build task36

    81.307 us
    task6 BUILT

    114.426 us
    core12 返回

    118.135 us
    kernel start

核心问题：

> Execute owner 生命周期早于 Execute readiness。

------------------------------------------------------------------------

# 2. 设计目标

必须保持：

-   deterministic replay；
-   Build 自由竞争；
-   TensorMap strict register；
-   AIC/AIV Execute cursor；
-   exactly once execution；
-   FinalDrain。

不引入：

-   全局 ready queue；
-   bitmap scheduler；
-   Build continuation；
-   kernel/scheduler overlap。

------------------------------------------------------------------------

# 3. 核心思想

当前：

    token = Execute ownership

改为：

    token = speculative observation

即：

> token 可以提前观察未来 task，但不能提前获得唯一执行权。

真正 Execute ownership 必须在 BUILT 后产生。

------------------------------------------------------------------------

# 4. 状态模型

## Execute token

旧：

    IDLE
     |
    WAITING_BUILT
     |
    CLAIMED
     |
    DONE

新：

    IDLE

     |
     v

    OBSERVING

     |
     +----------------+
     |                |
     v                v

    BUILT_READY    OWNER_BUSY

     |
     v

    CLAIMABLE

     |
     v

    CLAIMED

     |
     v

    DONE

------------------------------------------------------------------------

## TaskCell

增加：

    EXEC_PENDING

状态：

    EMPTY

    BUILDING

    BUILT

    EXEC_PENDING

    CLAIMED

    DONE

关键不变量：

    BUILT != owner assigned

------------------------------------------------------------------------

# 5. 新执行流程

## Execute token 获取

保持：

    AIC Execute cursor
    AIV Execute cursor

不变。

旧：

    cursor
     |
    claim owner
     |
    wait BUILT

新：

    cursor
     |
    observe task
     |
    if BUILT:
           claim
    else:
           OBSERVING

------------------------------------------------------------------------

# 6. Owner 转移机制

问题：

    core12 observing task6

    随后进入 Build task36

增加：

    exec_epoch

每个 Scalar 在外层 scheduler loop 推进时更新。

Token 保存：

    owner_epoch

判断：

    current_epoch != owner_epoch

表示：

owner 已离开原调度上下文。

状态：

    OBSERVING
          |
          v
    TRANSFERABLE

其他 Scalar 可以：

    CAS
    TRANSFERABLE -> CLAIMED

------------------------------------------------------------------------

# 7. 为什么不用 Ready Queue

Ready Queue 会引入：

-   producer/consumer；
-   head/tail；
-   ABA；
-   cacheline contention；
-   overflow。

当前已有：

-   TaskCell；
-   Execute cursor；
-   CAS claim。

无需新增第二套调度协议。

------------------------------------------------------------------------

# 8. 为什么不用 Build continuation

Continuation 需要保存：

-   Materialize 状态；
-   Register 状态；
-   Fanin 状态；
-   stack context。

会增加：

-   ABI；
-   GM state；
-   recovery；
-   FinalDrain 复杂度。

当前问题不是 Build 不可暂停，而是 Execute 不应该过早绑定。

------------------------------------------------------------------------

# 9. 实施步骤

## Phase 0：观测

增加 trace：

    task_id

    token state

    epoch

    BUILT timestamp

    CLAIM timestamp

    kernel timestamp

验证：

    BUILT -> Execute

等待占比。

------------------------------------------------------------------------

## Phase 1：引入 OBSERVING

目标：

-   不产生提前 ownership；
-   不改变 transfer。

验证：

-   correctness；
-   no regression。

------------------------------------------------------------------------

## Phase 2：引入 TRANSFERABLE

允许：

    OBSERVING
        |
        v
    TRANSFERABLE
        |
        v
    CLAIMED

重点观察：

-   task6 类反例；
-   BUILT 到 CLAIM 延迟。

------------------------------------------------------------------------

# 10. 验收指标

## 正确性

必须保持：

-   每 task Build 一次；
-   每 task Execute 一次；
-   TensorMap 依赖不变；
-   fanin 不变；
-   FinalDrain 正常。

------------------------------------------------------------------------

## 性能

重点：

    BUILT timestamp
            |
            v
    CLAIM timestamp
            |
            v
    kernel start

目标：

降低：

    BUILT -> kernel

延迟。

------------------------------------------------------------------------

# 11. 最终判断

不建议：

-   Ready Queue；
-   Runnable bitmap；
-   Full continuation。

推荐：

> 保留 Execute cursor 和 token，只改变 token 的语义。

从：

    Execute owns future task

改成：

    Execute observes future task,
    owns only ready task

这是对现有 cross-core scheduler 最小侵入、最高兼容性的优化方向。
