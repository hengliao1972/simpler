# PA Cross-Core Scheduler 优化方案设计

## Delayed Execute Claim with Bounded Revisit Window

版本：Proposal v2

------------------------------------------------------------------------

## 1. 背景与问题重新定位

当前 cross-core scheduler 已经完成：

-   Build / Execute 分离；
-   Build owner 与 Execute owner 解耦；
-   AIC/AIV 分角色 Execute cursor；
-   owner-local Execute token；
-   多槽 token 前视；
-   未 BUILT 时停止继续预领。

当前仍存在的问题：

> Execute claim 发生在 task BUILT 之前，导致未来执行资格被提前消耗。

典型流程：

``` text
Execute cursor
      |
      v
task N candidate
      |
      v
owner-local token
      |
      v
WAITING_BUILT
      |
      v
BUILT
      |
      v
kernel
```

根因：

> Execute owner 在 task ready 之前产生。

------------------------------------------------------------------------

## 2. 方案重新审视

### 不采用 Ready Queue

原因：

-   新增 producer/consumer 协议；
-   引入 head/tail 共享热点；
-   增加 ABA、overflow、publication 问题。

当前 runtime 已有 TaskCell、Execute cursor 和 CAS
claim，不需要新增第二套调度结构。

### 不采用 ownership transfer

如果 task 已被 Execute cursor 消费：

``` text
cursor -> task N
```

其他核没有天然方式知道 task N 可接管。

因此 transfer 机制必须额外引入 discovery：

-   ready queue；
-   bitmap；
-   scanner；
-   transferable list。

否则状态不可达。

### 不采用 Build continuation

需要保存：

-   Materialize 状态；
-   Register 状态；
-   Fanin 状态；
-   栈上下文。

会增加 ABI、GM 状态和 FinalDrain 复杂度。

------------------------------------------------------------------------

## 3. 核心方案

# Delayed Execute Claim

核心原则：

> Execute cursor 不负责产生 ownership，只负责提供候选顺序。

真正 ownership 只在：

``` text
task == BUILT
```

之后产生。

------------------------------------------------------------------------

## 4. 状态模型

保持：

``` text
BUILDING

BUILT

CLAIMED

DONE
```

不增加：

``` text
EXEC_PENDING
TRANSFERABLE
```

原因：

当前 BUILT 已经表示：

-   payload 完成；
-   publish 完成；
-   可被执行端 acquire。

真正缺失的是：

> BUILT task 如何被发现。

------------------------------------------------------------------------

## 5. Bounded Revisit Window

Execute worker 不等待未来 task。

而是在有限窗口内重新检查候选任务。

例如：

``` text
execute_cursor = 100

window:

100 ~ 163
```

扫描：

``` text
task.state == BUILT
```

若：

``` text
BUILT
 |
 CAS
 |
 CLAIMED
```

成功后执行。

------------------------------------------------------------------------

## 6. Execute 流程变化

旧：

``` text
fetch ticket

claim owner

if !BUILT:
    wait

execute
```

新：

``` text
advance candidate cursor

scan bounded window

for task in window:

    if state != BUILT:
        continue

    if CAS(BUILT -> CLAIMED):

        build private token

        acquire payload

        execute
```

------------------------------------------------------------------------

## 7. 为什么保留 Execute Cursor

Execute cursor 已解决：

-   role 内动态均衡；
-   固定候选核不均衡；
-   exactly-once claim。

因此：

不是删除 cursor。

而是改变语义：

旧：

``` text
cursor = ownership generator
```

新：

``` text
cursor = discovery order
```

------------------------------------------------------------------------

## 8. Window 设计

窗口不能无限扩大。

建议：

初始：

``` text
32 ~ 128 task
```

窗口大小关联：

-   private execute slots；
-   run-ahead window；
-   H。

------------------------------------------------------------------------

## 9. 正确性

### Exactly once

仍由：

``` text
BUILT -> CLAIMED
```

CAS 保证。

多个 executor：

``` text
A: CAS success

B: CAS fail
```

只有一个执行者。

### Payload visibility

保持：

``` text
payload write

DCCI flush

BUILT publish

claim

invalidate/acquire

kernel
```

不改变已有 publication contract。

### Deterministic replay

保持：

-   task id；
-   TensorMap；
-   heap；
-   Build 顺序。

只改变 Execute discovery。

------------------------------------------------------------------------

## 10. 验证计划

### Phase 0

增加 trace：

-   task_id；
-   candidate cursor；
-   scan count；
-   BUILT timestamp；
-   CLAIM timestamp；
-   kernel timestamp。

### Phase 1

CPU standalone：

验证：

-   window scan；
-   CAS claim；
-   无重复执行；
-   无遗漏。

### Phase 2

CCEC/A5：

验证：

-   BUILT publication；
-   claim CAS；
-   invalidate/acquire。

### Phase 3

B256 full-swimlane：

观察：

-   first AIC kernel；
-   BUILT-\>CLAIM 延迟；
-   startup 到 FinalDrain。

------------------------------------------------------------------------

## 11. 风险

### 扫描成本

使用：

-   bounded window；
-   role 分离；
-   小范围扫描。

### Starvation

需要：

-   window overlap；
-   cursor 推进策略；
-   FinalDrain 特殊处理。

### CAS 竞争

只允许 BUILT task 参与竞争。

------------------------------------------------------------------------

## 12. 最终判断

当前问题不是：

-   Execute token 不够多；
-   Build 内没有调度点。

而是：

> Execute ownership 产生时机错误。

最终方向：

``` text
Build

task

BUILT

        |
        v

Delayed Execute Claim

        |
        v

CLAIMED

        |
        v

kernel
```

一句话：

> 不让 Execute 提前拥有未来 task，而让 Execute 在有限窗口内发现已经
> ready 的 task。
