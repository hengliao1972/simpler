# PA Cross-Core Scheduler 优化方案设计

## Build Phase-Aware Execute Opportunity

版本：Proposal v6

状态：**已完成 standalone CPU、CCEC 与 A5 验证，性能否决，代码已撤回。**
本文保留方案合同、实测证据和否决原因，不代表现行实现。

------------------------------------------------------------------------

## 1. 背景

当前 cross-core scheduler 已完成：

-   Build / Execute 分离；
-   Execute token；
-   AIC/AIV Execute cursor；
-   BUILT -\> CLAIMED exactly-once 仲裁；
-   TensorMap 严格插入链。

剩余问题：

> Execute opportunity 出现时，持有 Execute obligation 的 Scalar
> 可能已经进入严格 Build commit 路径，导致无法及时推进 Execute。

典型：

``` text
core18:
    Build task6

core12:
    Execute token(task6)
        |
        v
    未 BUILT

        |
        v

    Build task36

task6:
    BUILT

但 core12 仍在 Register/commit 路径
```

------------------------------------------------------------------------

## 2. 已否决方向

### Ownership Transfer

否决：

-   缺少低成本 discovery；
-   需要额外共享协议。

### Ready Queue / Bitmap / Revisit Window

否决作为第一方案：

-   引入新的 scheduler 状态；
-   增加 publication、scan、retire 合同。

### Register Wait Execute Checkpoint

实验否决：

虽然首个 AIC kernel 提前，但同步 kernel 插入严格 Register 链：

``` text
Register wait
    |
    v
kernel
    |
    v
resume Register
```

造成：

-   Register wait 放大；
-   metadata chain 延迟传播；
-   端到端退化。

------------------------------------------------------------------------

## 3. 核心方案

# Build Phase-Aware Execute Opportunity

核心原则：

> Execute 只能插入已经释放 TensorMap 严格提交链、但尚未发布当前
> execution payload 的 Build 间隙。

Build 分为：

``` text
Prepare Phase（Materialize）

      |

Commit Phase（Register + insert completion）

      |

Execute Opportunity

      |

Fanin + BUILT Publish
```

允许：

``` text
Register completion
        |
        v
Execute Opportunity
        |
        v
Fanin + BUILT Publish
```

禁止：

``` text
Register wait / metadata commit
        |
        v
Kernel
```

------------------------------------------------------------------------

## 4. 插入点

第一候选：

``` text
Register 完成并发布 task N insert completion
Fanin 开始之前
```

原因：

-   task N 已经释放严格 task-id metadata commit 链；
-   N+1 owner 已可独立进入 Register；
-   Execute 只会延后当前 task N 的 Fanin/BUILT，不会把完整 kernel
    串进 TensorMap 插入链。

原候选 `Materialize -> Execute -> Register` 已被现有泳道证伪。以
`cross_waitbuilt_1p084ms.json` 的 task 6 为例：

``` text
core12 Materialize task36 end  76.457 us
task6 BUILT publish end         81.029 us
core12 Register task36 end     101.313 us
task6 actual kernel start      118.135 us
```

Materialize 后的单次检查不仅会错过 task 6，而且若当时命中一个
28--54 us kernel，插入前沿追到 task36 后仍会等待该 Build owner。物理上
没有嵌套在 Register span 内，不代表没有延迟 Register completion。

------------------------------------------------------------------------

## 5. 调度流程

旧：

``` text
Build ticket

Materialize

Register

Fanin

Publish
```

新：

``` text
Build ticket

Materialize

Register

publish insert completion

Execute Opportunity Check

Fanin

Publish BUILT
```

------------------------------------------------------------------------

## 6. Execute Opportunity Check

只处理已有 Execute obligation，并且一次 Build 最多推进一个 owner-local
token。

允许：

``` text
选择一个 owner-local occupied token

if BUILT:

    CLAIM

    acquire payload

    fanin probe once
```

禁止：

-   新 Execute ticket；
-   新 Build ticket；
-   全局 token scan；
-   fanin spin。

选择过程只读取本 worker 的固定四个 token control，优先最老 task id，
随后只对该 token 调用一次现有推进原语。没有 occupied token 时完全跳过；
本次机会不得因为成功完成一个 token 而继续 drain 其余 token。

------------------------------------------------------------------------

## 7. Fanin 合同

允许：

``` text
BUILT
 |
CLAIM
 |
fanin probe
```

如果 ready：

执行。

如果 not ready：

立即返回。

不能在 checkpoint 中等待 fanin。

------------------------------------------------------------------------

## 8. CCEC 实现约束

当前真实 dispatcher 只由 orchestration caller TU 定义，finish object
不得引用 dispatcher。不能把 `ExecuteOpportunityHook()` 直接编译进 Finish；
前一轮 Register-wait 实验已经证明这种形态会把最终 ELF 从约 2.41 MiB
膨胀到 2.56 MiB，并破坏现有 split-object 依赖合同。

第一版采用两段式 Finish：

``` text
Finish Stage A（finish TU）
    Materialize
    Register
    publish insert completion
    return Registered

orchestration caller TU
    progress at most one existing owner-local token

Finish Stage B（finish TU）
    Fanin
    publish BUILT / Complete Alloc
    close Submit
```

两次调用继续复用同一个 block-local `CompeteFirstSplitRuntimeState`、
`SubmitContext` 和 caller 栈上的 `TaskArgs`。continuation 身份编码进现有
一次性交接字，第二次进入后消费并清零；不新增跨核 SchedulerState、atomic、
DCCI 或行为宏。逻辑 Finish 次数仍按 task 计一次，不能因 Stage B 重入翻倍。

结果使用固定枚举表达：

``` cpp
enum class SplitFinishResult : uint32_t {
    Failed = 0,
    Completed = 1,
    Registered = 2,
};
```

------------------------------------------------------------------------

## 9. 正确性

Exactly once：

``` text
BUILT -> CLAIMED
```

CAS 保持不变。

Payload：

``` text
payload write
-> DCCI flush
-> BUILT publish
-> CLAIM
-> invalidate/acquire
-> kernel
```

Deterministic replay：

不改变：

-   task id；
-   TensorMap；
-   heap；
-   Register 顺序。

对于当前 Build task N，Stage A 成功意味着其 TensorMap writer 已经提交，
`insert_completion[N]` 已发布。Stage B 尚未完成只允许 N 的 execution cell
保持 EMPTY/BUILDING，不能撤回已经提交的 TensorMap 元数据；失败仍沿现有
terminal fatal 合同收敛。

被机会点执行的 task M 必然已经 BUILT，而 BUILT 发布发生在 M 自身 Register
完成之后。严格 task-id 插入又意味着 M 不可能依赖尚未 Register 的未来 N；
执行仍只消费 token-private payload，并沿用既有 DCCI acquire 合同。

------------------------------------------------------------------------

## 10. 验证计划

Phase 0：

先复用现有泳道中的 task id、Materialize、Register、BUILT/CLAIM、Kernel 和
Commit 事件，不增加新的逐 task raw 字段。cross-core 未使用的第二个 physical
placement counter 可以承载 BuildOpportunity 完成数，避免扩大 profiling
文件。

Phase 1：

CPU：

验证：

-   hook 位置；
-   exactly-once；
-   无递归 scheduler。
-   Stage A 已发布 insert completion，Stage B 尚未发布当前 BUILT；
-   caller 只执行一个已有 token，不领取新 Execute ticket；
-   Stage B 必须消费同 task continuation，重复、错 task 和漏调用均失败闭合。

Phase 2：

CCEC/A5：

验证：

-   split TU；
-   callback；
-   DCCI；
-   CLAIM；
-   completion。
-   finish object 不引用 AIC/AIV dispatcher；
-   32 KiB stack、最终 ELF relocation 和代码体积保持受控。

Phase 3：

B256：

观察：

``` text
first AIC kernel

BUILT -> kernel

last insert completion

FinalDrain

total runtime
```

------------------------------------------------------------------------

## 11. 成功标准

同时满足：

降低：

``` text
BUILT -> kernel latency
```

且不能用局部指标掩盖：

``` text
startup -> last insert completion
startup -> FinalDrain end
```

不破坏：

``` text
TensorMap strict ordering
```

Register span 自身可能保持不变，即使 hook 已把其绝对完成时刻整体后移；因此
不能只比较 Register duration。候选只有在同设备顺序 A/B 中降低完整端到端
中位数，且所有终态与 split-object 检查通过时才保留。首个 AIC 提前、
FinalDrain kernel 减少都只是解释证据，不是单独的保留条件。

------------------------------------------------------------------------

## 12. 最终判断

当前问题不是：

-   Execute claim 错误；
-   token 数不足；
-   discovery 不足。

而是：

> Execute opportunity 出现时，Scalar 正持有一个 Build；必须先释放
> TensorMap 严格提交链，再在当前 task 的 Fanin/BUILT 之前有限推进旧任务。

最终方向：

``` text
Build Materialize

      |

Register + insert completion

      |

Execute Opportunity

      |

Fanin + BUILT Publish
```

一句话：

> 不在严格提交链中执行 kernel；先发布当前 task 的插入完成字，再由 caller
> 推进至多一个已有 Execute token，最后恢复当前 task 的 Fanin 与 BUILT。

------------------------------------------------------------------------

## 13. 实现与验证结果

### 13.1 正确性和代码形状

第一版按本文合同实现了两段式 Finish，并先用 CPU 门槛验证：

- Stage A 结束时 `insert_completion[N]` 已发布，当前 task 尚未 `BUILT`；
- caller 只推进一个已经持有的 owner-local token，不读取中央 Execute cursor；
- Stage B 必须消费同一 task 的 continuation，随后完成 Fanin、BUILT 和 Submit；
- 完整 CPU 协议、96 worker 并发、严格插入、payload、fatal 和 FinalDrain 均通过。

最初把推进逻辑直接展开到 caller 后，最终 CCEC ELF 从约 `2.407 MiB` 增至
`3.021 MiB`。随后把正常 token drain 与单次机会合并进同一个 noinline helper，
最终 perf-clock ELF 为 `2,195,712 B`，相对冻结基线 `2,407,192 B` 反而减少
`8.785%`；AIC/AIV caller object 分别缩小 `14.922%/13.253%`，Finish object
仅增加约 `2.8%`，且没有引入 dispatcher 依赖或最终 relocation。因此最后的
A5 回退不能简单归因于最终 ELF 总体膨胀。

B1 实测 `204.528 us`，B256 的 Build/Execute exactly-once、TensorMap 严格
插入、payload、fanin、DCCI、completion 和 FinalDrain 断言全部通过。

### 13.2 冻结 A/B

用修改前冻结产物和最终候选产物做 6 对 B-C/C-B 反转交错，统一统计 Startup
起点到 FinalDrain 结束：

```text
基线      = 1015.449, 1000.084, 1008.358,
            982.198, 1024.745, 1029.086 us
候选      = 1035.949, 1042.194, 1034.206,
            1033.856, 1047.252, 1034.803 us

基线中位数 = 1011.904 us
候选中位数 = 1035.376 us
回退       = 23.473 us / 2.320%
候选获胜   = 0 / 6 对
配对差中位 = +24.178 us
```

候选在两种执行顺序中都没有赢过冻结基线，因此不能把结果解释成一次偶然长尾。

### 13.3 保留的候选泳道

过程泳道保留在：

```text
test_record/2026-8-6/build_phase_opportunity_rejected/
├── l2_swimlane_records.json
└── merged_swimlane.json
```

该次 B256 结果为：

```text
lifecycle       = 1104.222 us
Submit          = 1011.970 us
EfDrain         = 694 kernels
BuildOpportunity= 208 kernels
FinalDrain      = 122 kernels
```

所有设备业务断言通过，raw 无 dropped record。正式 sparse validator 原合同要求
Register 后立即进入 Fanin/AllocComplete，因而不认识中间插入的 Kernel。为了
保存被否决候选的过程证据，只用临时 host exporter 把这两个相邻端点约束从
“严格相等”放宽为“不得重叠”，生成 raw 和 Perfetto 合并泳道；临时改动没有
进入设备代码，也已撤回。正式 exclusive analyzer 仍按现行生产结构拒绝该
过程态，不为已否决方案扩张长期兼容分支。

------------------------------------------------------------------------

## 14. 为什么局部执行提前，端到端反而下降

### 14.1 机会点大多只付费，没有完成 kernel

机会点对全部 `1,280` 个 Build task 都执行一次固定四 token 检查，并使每个
Build 的 split Finish 从一次跨 TU 调用变成 Stage A、Stage B 两次。但保留泳道
中只有 `208/1280 = 16.25%` 的 Build 机会真正完成了 kernel；换成全部 `1,024`
个 kernel 的口径，也只有 `20.31%` 在该位置完成。其余约五分之四的 Build
仍支付二次 Finish、continuation 校验和 token control 读取，却没有得到执行
推进收益。

### 14.2 它主要搬运既有 EfDrain 工作，没有消减尾部

同底座撤回后的典型 placement 为：

```text
EfDrain=902, BuildOpportunity=0, FinalDrain=122
```

候选保留泳道为：

```text
EfDrain=694, BuildOpportunity=208, FinalDrain=122
```

三项总数都为 `1024`。候选恰好把 `208` 个原本会在常规 EfDrain 推进的 kernel
搬到了 Register 后机会点，而 FinalDrain 没有减少。也就是说，它改善了部分
kernel 的局部时间位置，却没有消除真正的尾部执行量，不能抵消新增控制成本。

### 14.3 Execute 提前会反向延后后续 BUILT 供给

机会点已经释放 task N 的 TensorMap 插入链，因此不会像 Register-wait 候选那样
把完整 kernel 直接串进全局有序链；但它仍位于 N 的 Fanin 和 BUILT 之前：

```text
insert_completion[N]
-> execute old task M
-> Fanin[N]
-> publish BUILT[N]
```

Scalar 在同步执行 M 的 2--56 us 期间不能继续生成 N 的 portable payload。
所以 M 可能更早，而 N 及其后续可执行供给相应变晚。在 Build 与 Execute 都消耗
同一 Scalar 时间、FinalDrain 又没有减少的本轮负载上，这只是把工作在时间轴上
重新排布，并额外增加状态机成本，不会凭空产生 overlap。

### 14.4 二次 Finish 是每 task 的确定性税

即使没有 occupied token，每个 Build 仍新增：

- 第二次 caller→finish 跨 TU 调用；
- continuation binding 的写入、校验、消费和分支；
- 固定四个 owner-local token control 的选择扫描；
- Stage A 返回值与 Stage B 完成值的两次协议判断。

这些成本按 `1,280` 个 task 固定发生，而有效机会只出现 208 次。最终 ELF 虽已
缩小，动态执行的指令数和 GM token-control 访问仍然增加；因此“代码体积变小”
与“热路径执行更少”不是同一件事。

### 14.5 结论

该方案的理论前提是：Build 中间存在大量会错过常规推进点、并最终滞留
FinalDrain 的 ready token。实测却表明，新增机会主要抢先消费了下一次 EfDrain
本来就会处理的任务，FinalDrain 没有下降。于是收益上限只是局部时序搬移，成本
却是每个 Build 都支付，最终形成稳定的约 `2.32%` 回退。

因此本候选正式否决：生产代码、临时 CPU 门槛、placement 别名和 split Finish
continuation 均已撤回；只保留本文与过程泳道，防止以后在缺少新证据时重复试验。
