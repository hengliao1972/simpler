# PA Cross-Core Scheduler 优化方案设计

## Execute-Aware Build Admission Control

版本：Proposal v3

------------------------------------------------------------------------

# 1. 背景

当前 cross-core scheduler 已经完成：

-   Build / Execute 分离；
-   Build owner 与 Execute owner 解耦；
-   AIC/AIV 分角色 Execute cursor；
-   owner-local Execute token；
-   多槽 token 前视；
-   未 BUILT 时停止继续预领。

当前剩余问题：

> Execute obligation 会被绑定在某个 Scalar 上，而该 Scalar
> 随后可能进入长 Build，导致已经 BUILT 的 task 无法及时执行。

典型路径：

``` text
Execute ticket

      |

      v

WAITING_BUILT token

      |

      v

同一 Scalar 进入长 Build

      |

      v

task BUILT

      |

      v

等待 owner 返回调度边界
```

典型时间线：

``` text
40.394 us
core12 观察 task6

41.965 us
core12 开始 Build task36

81.307 us
task6 BUILT

114.426 us
core12 返回调度边界

118.135 us
kernel start
```

核心问题：

> Execute obligation 与 Build admission 之间缺少协调。

------------------------------------------------------------------------

# 2. 前序方案审视

## 2.1 Ownership Transfer

不采用。

原因：

如果 task 已经被 Execute cursor 消费：

``` text
cursor -> task N
```

其他 Scalar 没有天然发现该 task 的机制。

因此 transfer 需要额外：

-   ready queue；
-   bitmap；
-   transferable list；
-   scanner。

会引入新的 discovery 协议。

------------------------------------------------------------------------

## 2.2 Delayed Execute Claim + Bounded Revisit Window

方向正确，但实现复杂。

主要问题：

需要定义：

-   window 如何覆盖；
-   未 BUILT task 如何保存；
-   revisit 谁负责；
-   retire frontier；
-   FinalDrain。

否则无法证明：

-   at-least-once discovery；
-   不漏 BUILT task；
-   不产生大量 Atomic scan。

------------------------------------------------------------------------

## 2.3 Ready Queue

不采用。

原因：

新增：

-   producer/consumer；
-   head/tail；
-   ABA；
-   cacheline contention；
-   overflow。

当前已有：

-   TaskCell；
-   Execute cursor；
-   CAS claim。

无需新增第二套任务发布结构。

------------------------------------------------------------------------

# 3. 核心方案

## Execute-Aware Build Admission Control

核心思想：

> 不增加 Execute discovery，而减少 Execute obligation 被长 Build
> 隔离的概率。

即：

当前：

``` text
Execute token
      |
      v
未来 task obligation
      |
      v
允许进入长 Build
```

改：

``` text
Execute token
      |
      v
检测 pending execute obligation

      |

      +----------------+
      |                |
      v                v

优先推进 Execute     允许 Build
```

------------------------------------------------------------------------

# 4. 核心原则

## 原则1

保留：

-   Execute cursor；
-   Execute token；
-   BUILT -\> CLAIMED CAS。

不改变已有 ownership 机制。

------------------------------------------------------------------------

## 原则2

Build admission 必须感知本核未完成 Execute obligation。

如果：

``` text
本核存在 WAITING_BUILT token
```

则不能无条件领取长 Build。

------------------------------------------------------------------------

# 5. 新调度规则

当前：

``` cpp
ProgressExecute();

BuildTask();
```

修改：

``` cpp
ProgressExecute();

if (HasBlockedExecuteToken()) {

    RetryExecuteProgress();

    if (still_blocked)
        delay_build_admission();

} else {

    BuildTask();

}
```

------------------------------------------------------------------------

# 6. Pending Execute 判断

不使用：

-   时间；
-   wall clock。

原因：

A5 多核时间不可作为可靠协议。

------------------------------------------------------------------------

采用：

## scheduler round

每次进入外层调度循环：

``` text
scheduler_epoch++
```

token 保存：

``` text
last_progress_epoch
```

如果：

``` text
current_epoch - last_progress_epoch > threshold
```

说明：

该 Execute obligation 长时间没有推进。

此时：

降低 Build admission 优先级。

------------------------------------------------------------------------

# 7. Token 语义

保持：

``` text
IDLE

WAITING_BUILT

CLAIMED

DONE
```

不新增：

``` text
TRANSFERABLE
EXEC_PENDING
```

原因：

这些状态无法解决 task discovery。

------------------------------------------------------------------------

# 8. 对 task6 的行为

旧：

``` text
40us

core12:
task6 WAITING


42us

core12:
Build task36


81us

task6 BUILT


114us

core12 返回


118us

kernel
```

新：

``` text
40us

core12:
task6 WAITING


42us

发现 pending execute obligation

禁止进入长 Build


优先等待/推进 task6


81us

task6 BUILT


83~90us

kernel
```

目标：

减少：

``` text
BUILT -> Execute
```

等待。

------------------------------------------------------------------------

# 9. 正确性

## Exactly once

保持：

``` text
BUILT -> CLAIMED
```

CAS。

多个 Scalar：

``` text
winner:
CAS success

loser:
CAS fail
```

------------------------------------------------------------------------

## Payload visibility

保持：

``` text
payload write

DCCI flush

BUILT publish

CLAIM

invalidate/acquire

kernel
```

------------------------------------------------------------------------

## Deterministic replay

不改变：

-   task id；
-   TensorMap；
-   heap；
-   Build 顺序。

------------------------------------------------------------------------

# 10. 风险

## 风险1：Build 利用率下降

如果简单限制：

``` text
有 WAITING_BUILT 就不能 Build
```

可能造成：

所有 Scalar 等 Execute。

因此必须使用：

-   quota；
-   threshold；
-   round-based admission。

------------------------------------------------------------------------

## 风险2：慢 Execute 阻塞 Build

需要限制：

单个 pending token 对 Build 的影响范围。

建议：

初始：

-   只影响 owner Scalar；
-   不影响其他 Scalar。

------------------------------------------------------------------------

# 11. 实施计划

## Phase 0：观测

增加：

``` text
task_id

token state

scheduler epoch

Build admission decision

BUILT timestamp

kernel timestamp
```

统计：

-   pending token 时间；
-   被 Build 阻塞次数。

------------------------------------------------------------------------

## Phase 1：本核 Build admission

只改：

``` text
DispatchOneSharedBuildTask()
```

不改：

-   TaskCell；
-   Execute cursor；
-   token ABI。

------------------------------------------------------------------------

## Phase 2：参数扫描

验证：

-   threshold；
-   quota；
-   pending token 数。

------------------------------------------------------------------------

## Phase 3：B256 full-swimlane

观察：

-   first AIC kernel；
-   BUILT -\> kernel 延迟；
-   startup 到 FinalDrain；
-   Build 吞吐。

------------------------------------------------------------------------

# 12. 最终判断

当前问题不是：

-   Execute token 数量不足；
-   Execute claim 不正确；
-   缺少 ready queue。

而是：

> 一个 Scalar 在承担未来 Execute obligation 时，仍然允许进入不可响应的长
> Build。

因此最小修正：

``` text
Execute-aware Build Admission

而不是

重新设计 Execute discovery
```

目标：

让负责观察 Execute readiness 的 Scalar 保持响应能力。

一句话：

> 不让 Execute 找更多 task，而是不让已经负责 Execute 的 Scalar 消失在
> Build 中。
