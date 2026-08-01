# Scalar 调度与算子执行重叠机制验证

## 1. 目的与边界

这组探针先独立验证一个基础合同：当 AIC/AIV engine 已经发射、但最终
`FIX/MTE3 -> S` 完成事件尚未等待时，scalar 能否保存当前 continuation，
继续推进与该输出无关的 loser Submit；遇到第二个 winner 后，再保存新的
continuation，按 task-age 顺序恢复旧工作，完成最终 wait 和 completion 发布，
最后恢复第二个 winner。

探针位于 `tests/atomic_probe/ccec`，不 include `pa_scheduler`，也不修改
standalone 或 simpler 的正式调度路径。这里验证的是协程所需的上下文转移、
完成边界和基础收益，不宣称已经实现通用调度器。

## 2. 已求证的 A5 同步能力

本机 CANN 9.1 的 `try_wait(id, sync_mode)` 对应
`TRY_WAIT/TRY_WAITI`，公开 mode 为 `CROSS_CORE`、`INTRA_BLOCK`、
`BUFFER_ID` 和保留值。直接把 `EVENT_ID7` 作为第一个参数时，三种有效 mode
在 AIC/AIV 的发射前、发射后、scalar replay 后和最终 wait 后都返回 0；
所以它不能直接查询 `set_flag/wait_flag` 的 pipeline event。

继续审计 A5 simulator 并上板验证后，确认 `BUFFER_ID` 的正确对象是
`get_buf/rls_buf` 协议。该 mode 读取某个 buffer id 的已派发 acquire 与已完成
release 之差；单个在途工作返回 1，release 完成后返回 0。最终 pipeline 工作
必须按同一 pipe 顺序显式标记：

```cpp
get_buf(PIPE_MTE3, buffer_id, false);  // AIV；AIC 使用 PIPE_FIX
TSTORE(output, tile);
rls_buf(PIPE_MTE3, buffer_id, false);

const int64_t pending = try_wait(buffer_id, BUFFER_ID);
```

常量和运行时 buffer id 均已编译并上板通过。由此可采用真正可动态检查的
两阶段方式：

```text
保存 engine continuation E
  -> 用 get_buf/rls_buf 标记最终 FIX/MTE3 工作并保留 EVENT_ID7
  -> scalar 回放 loser Submit，机会式 try_wait(buffer_id, BUFFER_ID)
     ├─ pending != 0 且仍是 loser：继续做独立调度工作
     ├─ pending == 0：恢复 E，执行原 wait 并发布 completion
     └─ pending != 0 且到达第二个 winner：
          保存 continuation W，停止继续前进
          -> 恢复 E，执行原 wait 并发布 completion
          -> 恢复 W
```

`try_wait` 不替代最终 `wait_flag`：它只决定何时恢复旧 continuation；恢复后
仍执行原 wait，以保持原事件消费和 completion 边界。返回 0 同时表示“尚未
发射”和“已经完成”，因此必须与 slot 的 `Issued` 状态联合判断。TLOAD、
TMOV、TMATMUL、TADD、TSTORE 之间继续维持原 pipeline 依赖，不能把本结果
解释成整个算子执行时间都可以隐藏。

## 3. 探针实现

### 3.1 显式 continuation

`ContinuationFrame` 是 128B 的显式结构，保存 8 个 64-bit 上下文字段、
signature、generation 和状态。`SaveContinuation`、`RestoreContinuation`
均为 `noinline, used`，frame 使用 volatile 本地存储。

第一份上下文保存后，源码会主动用另一组值覆盖原 `RuntimeContext`。因此最终
校验不能依赖编译器碰巧保留的 SSA 值或寄存器，只能从 frame 恢复。

host 使用独立 oracle 重算两份上下文 signature、loser replay checksum 和
第二个 winner 上下文；不是只相信 device 自己写出的 PASS 位。

恢复顺序编码为：

```text
resume_sequence = ((0 << 2) | 1) << 2 | 2 = 6
```

即先恢复 generation 1，再恢复 generation 2。任一 frame 状态、generation、
signature、host oracle 或顺序不匹配，用例均失败。

### 3.2 三种对照模式

| 模式 | 执行顺序 | 作用 |
| --- | --- | --- |
| `ContextFifo` | save E -> replay -> save W -> restore E -> restore W | 单独验证上下文与 FIFO，不发射 engine |
| `EngineThenSchedule` | issue -> final wait -> replay -> restore E/W | 串行性能对照 |
| `EngineOverlapSchedule` | issue -> replay -> restore E -> final wait/commit -> restore W | 验证 scalar 调度覆盖 engine 尾部 |

串行与重叠模式调用完全相同的一份 `noinline` loser replay 机器码，运行顺序按
奇偶轮交替，避免再次把 O3 展开差异或单向漂移误判成 overlap。

### 3.3 `try_wait` 定向模式

| 模式 | 标记与查询 | 验证目标 |
| --- | --- | --- |
| `TryWaitCrossCore/IntraBlock/BufferId` | 直接查询 `EVENT_ID7` | 证明 event id 不是正确查询对象 |
| `TryWaitTaggedBufferId` | 常量 buffer id 31 | 验证单个 buffer-token 的 `0 -> 1 -> 0` |
| `TryWaitTaggedDynamicBufferId` | 运行时 buffer id 24/25 | 验证 get、release、query 均支持运行时 ID |
| `TryWaitPollUntilDone` | 连续按返回值轮询 | 验证查询不阻塞且最终能观察完成 |
| `TryWaitFourSlots` | buffer id 24--27 | 验证四个独立 ID 可合成为 pending bitmask |
| `TryWaitIdleCost` | 已完成 buffer 上查询 4,096 次 | 估算循环中机会式查询的稳态摊销成本 |

这些模式都有 host 强断言，不是只打印取数。四 ID 模式目前让四个 token 标记
同一份最终 TSTORE，只证明 ID 独立性和 bitmask 查询；它尚不等于四个 engine
task 同时在途。

### 3.4 AIC 与 AIV 负载

- AIC：128×128 FP32 matmul，输入恒为 2 和 3，逐元素期望输出 768；最终延后
  `FIX -> S EVENT_ID7`。
- AIV：128×128 FP32 add，输入恒为 2 和 3，逐元素期望输出 5；最终延后
  `MTE3 -> S EVENT_ID7`。
- loser replay：循环读取 64 个独占 64B 的 synthetic task record，执行
  task cursor、ready 分支和参数摘要更新；`iterations` 控制回放长度。

PMU 复用已有十槽 owner，一次配置后读取 total、scalar、Vector、Cube、
MTE1/2/3、FIX、I-cache request/miss，退出时恢复原寄存器配置。PMU cycle 按
本机校准的 1.65 GHz 换算，SYS_CNT 仍按 1 GHz 直接作为 ns 交叉验证。

## 4. 运行方法

```bash
cd /home/q00473782/atomic/private/gpt/simpler
source /home/q00473782/Ascend/cann-9.1.0-weekly-20260708/cann/set_env.sh
source .venv/bin/activate
tests/atomic_probe/ccec/run_scalar_coroutine_probe.sh
```

也可以分别执行 `build` 或 `run`。脚本优先使用
`~/.local/gcc-15/root/usr/bin/g++-15`，AIC/AIV 分别由 CCEC cube/vec 后端编译。

## 5. 2026-08-01 A5 实测

运行条件：device 0、CANN 9.1、每组 5 个进程内配对样本。两种 role 的
`ContextFifo`、两种 engine 顺序、6 档 replay 长度、所有输出、host oracle、
`resume_sequence=6`、`protocol_status=0xf`、PMU stop/restore/cleanup 全部 PASS。

### 5.1 显式上下文基础成本

这里的 0-iteration 窗口完整包含两次 save、两次 restore、上下文构造/毒化、
signature 与协议校验；它不是单次 context switch 的成本。

| role | replay iterations | PMU total 中位数 | 1.65 GHz 换算 | scalar busy |
| --- | ---: | ---: | ---: | ---: |
| AIC | 0 | 1,284 cycles | 778.182 ns | 1,284 cycles |
| AIV | 0 | 1,282 cycles | 776.970 ns | 1,282 cycles |
| AIC | 64 | 29,126 cycles | 17,652.121 ns | 29,124 cycles |
| AIV | 64 | 29,128 cycles | 17,653.333 ns | 29,124 cycles |

0-iteration 结果只能作为当前显式 frame 实现的整段基础成本。正式调度器还会
改变 frame 字段数量和控制流，不能直接把约 0.78 us 当成最终成本。

### 5.2 调度与 engine 尾部重叠

表中“收益”均为同 seed 串行减重叠的 5 组配对中位数。

| role | replay iterations | 串行 total | 重叠 total | PMU 收益 | SYS_CNT 收益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AIC | 0 | 15,040 | 14,019 | 476.364 ns | 466 ns |
| AIC | 16 | 22,028 | 14,594 | 4,507.879 ns | 4,513 ns |
| AIC | 64 | 42,974 | 29,187 | 8,355.758 ns | 8,357 ns |
| AIC | 256 | 126,622 | 112,707 | 8,433.333 ns | 8,435 ns |
| AIC | 1,024 | 460,601 | 446,787 | 8,372.121 ns | 8,373 ns |
| AIC | 4,096 | 1,799,907 | 1,786,179 | 8,320.000 ns | 8,353 ns |
| AIV | 0 | 7,189 | 6,759 | 235.758 ns | 229 ns |
| AIV | 16 | 14,253 | 8,307 | 3,607.273 ns | 3,605 ns |
| AIV | 64 | 34,955 | 29,187 | 3,499.394 ns | 3,502 ns |
| AIV | 256 | 118,485 | 112,707 | 3,506.061 ns | 3,513 ns |
| AIV | 1,024 | 452,659 | 446,787 | 3,558.788 ns | 3,555 ns |
| AIV | 4,096 | 1,792,007 | 1,786,179 | 3,536.364 ns | 3,481 ns |

有实际 replay 工作后，收益随可覆盖 scalar 工作增加而上升并饱和：AIC 约
8.3--8.4 us，AIV 约 3.5--3.6 us。PMU 与 SYS_CNT 同向，且饱和档的
`scalar busy` 配对差只有约 -6 到 -15 cycles，说明 scalar 工作没有被删掉，
主要变化确实是原本串行暴露的 engine 尾部被覆盖。

`iterations=0` 仍有约 0.2--0.5 us 的顺序差，只反映 wait、恢复点和流水发射
边界的固定差异，不能作为“有用调度被隐藏”的证据。判断重叠能力应看
`iterations>=16` 以及饱和平台。

在把第一份 save 纳入窗口之前的两次完整进程复测中，饱和收益已稳定落在
AIC 约 8.28--8.46 us、AIV 约 3.47--3.60 us；窗口修正只给两条路径增加
共同上下文成本，最终饱和值保持一致。

### 5.3 `try_wait` 返回值与成本

单 buffer 的稳定状态序列如下，AIC/AIV 一致：

| 场景 | 发射前 | 发射后 | 0 次 replay 后 | 4,096 次 replay 后 | 最终 wait 后 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 常量 buffer id 31 | 0 | 1 | 1 | 0 | 0 |
| 运行时 buffer id 24/25 | 0 | 1 | 1 | 0 | 0 |

连续轮询中，AIC 通常经历约 3.4K 次返回 1 后变为 0，AIV 通常约 1.4K 次；
这两个次数只受当前 engine 尾部和紧密轮询吞吐共同决定，不能解释成时间。
四 ID 模式在发射后稳定得到 pending mask `0xf`，长 replay 后为 `0x0`。
上述模式连续三次完整进程复测均 PASS。

在未 acquire 的 buffer 上执行 4,096 次动态 ID 查询，扣除同模式 0 次窗口后，
AIC/AIV 都约为每次 4 PMU cycles，即按 1.65 GHz 约 2.43 ns。该数是当前
`noinline` 紧循环中的稳态摊销吞吐成本，包含循环控制；不是孤立一条
`TRY_WAIT` 的完成延迟，也不能直接乘真实 Submit 的检查次数预测端到端开销。

## 6. 当前能够确认的合同

1. A5 CCEC 可以用显式 frame 保存两份 scalar continuation，并在原活跃
   context 被覆盖后正确恢复。
2. engine continuation 与第二个 winner continuation 可以按 task-age FIFO
   恢复；不需要同时保留两套隐式 C++ 调用栈。
3. scalar 可以在最终 `FIX/MTE3 -> S` wait 之前执行独立调度工作。
4. completion 必须保持未发布，直到旧 engine continuation 恢复并完成最终
   wait；之后才能恢复第二个 winner。
5. 当前负载可覆盖的只是约 8.3 us AIC、3.5 us AIV 尾部，不是整个约 50 us
   kernel。是否有更大 overlap 取决于真实 kernel 的发射边界和可独立调度量。
6. `get_buf/rls_buf + try_wait(buffer_id, BUFFER_ID)` 可以非阻塞观察最终
   FIX/MTE3 工作是否仍在途；直接查询 `EVENT_ID7` 不可用。
7. buffer id 可运行时传入；固定数量的 continuation slot 可以各自持有一个
   ID，并通过逐 slot 查询形成 pending/ready bitmask。

## 7. 尚未证明、正式修改前必须保留的约束

- 探针只验证“一份在途 engine + 遇到第二个 winner 后挂起”的最小闭环，
  尚未验证任意深度或多个同时在途 engine。
- 四 buffer-id 模式只给同一最终 TSTORE 叠加四个 token，尚未证明四个不同
  engine task 同时在途时的 ID、UB/L1/L0 和输出所有权合同。
- `try_wait == 0` 不能单独区分“未发射”和“已完成”，正式状态机必须先确认
  slot 处于 `Issued`，再把 0 解释为可恢复。
- 目前仍保留并执行原 `wait_flag`；尚未验证也没有必要让 `try_wait` 直接替代
  原事件消费。buffer-token 标记本身的端到端代价也需在正式接入前单独量取。
- synthetic replay 没有 Claim atomic、TensorMap 插入、fanin 和 EfDrain，不能
  用本探针的绝对时间预测 PA 端到端收益。
- 正式 frame 必须明确保存 task id/kind、args 或其稳定所有权、winning slot、
  engine role、event id 和下一状态；不得保存离开作用域后失效的栈地址。
- loser 可以在旧 engine 在途时继续前进；一旦再次 win，必须先保存新 winner
  continuation，并优先收口更老的 engine，不能继续无界回放。
- event id、UB/L1/L0 buffer 和 winning slot 在 continuation 完成前都仍被占用，
  不能因为 launch 已发射就提前复用或释放。
- shared TensorMap 的严格插入顺序、Claim 仲裁和 fanin 语义不因该机制改变。

只有上述约束在下一轮独立状态机门槛中继续闭合后，才适合把机制接入
`pa_scheduler` 的正式路径。
