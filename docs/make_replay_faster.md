# 让 Replay 更快 —— Compete-First 编排

本文档为 [fully_distributed_within_core.md](fully_distributed_within_core.md) 的 runtime 提出一项
性能增强：**把 kernel 的参数块（param-block）构建移出 replay 关键路径**，让绝大多数“败者”核在
认领失败后**立即跳过**昂贵的参数打包，从而显著降低每个 AICore 花在 “replay” 阶段的 cycle。

本增强同时已并入主设计文档的 [§6.8](fully_distributed_within_core.md)。本文提供更完整的动机、
正确性论证与 API/codegen 落地方案。

---

## 1. 背景：replay 阶段在做什么

全分布式模式下**没有中心调度器**：编排函数被加载并**同时运行在每一个 AICore 上**（SPMD），
每个核完整重放（replay）同一段编排程序，逐个 submit 点竞争任务所有权（[§1、§2](fully_distributed_within_core.md)）。

编排函数由两部分组成：

1. **高层控制/数据流**——描述 kernel 之间关系的循环与分支（batch/head/block 迭代、`PTO2_SCOPE`
   等）。
2. **每个 kernel 的参数块构建**——在每次 `rt_submit_*` 之前，把该 kernel 的输入/输出/标量参数
   打包进一个 `L0TaskArgs`（`add_input` / `add_output` / `add_scalar`），其中还包括构造
   tensor view（`tensor.view(...)`）等。

以 `paged_attention_orch.cpp` 的 SplitK PV matmul 为例（简化）：

```cpp
// === Task 3: SplitK PV matmul (accumulated P @ V) ===
L0TaskArgs params_pv;
params_pv.add_input(pij_f16);          // 打包输入
params_pv.add_input(vj);               // 打包输入（vj 来自 value_cache.view(...)）
params_pv.add_output(tile2d_ci);       // 打包输出 create-info
CYCLE_COUNT_LAP(prof_param_setup);     // ← 这一段就是 param-block 构建
TaskOutputTensors pv_outs = rt_submit_aic_task(FUNC_PV_MATMUL, params_pv);
```

在该文件自带的 profiling（`ENABLE_PROFILING`）里，`prof_param_setup` 与 `prof_tensor_view`
两项**合计占据了片上编排墙钟的大头**。

## 2. 问题：参数块构建被每个核对每个任务无条件执行

关键事实：**上面这段参数块构建，在每个核上、对每个任务都会执行——无论该核是否会赢得该任务。**

而参数块**真正被谁用到**？

| 角色 | 是否需要完整参数块 | 用途 |
| ---- | ------------------ | ---- |
| **winner**（认领成功） | **需要**全部（input + output + scalar） | 更新 TensorMap、构建任务、执行 kernel |
| **loser**，`tensormap == private`（每核复制，即本设计的默认模型 [§4](fully_distributed_within_core.md)） | 只需要 **output** 部分 | 把 output 作为 producer 登记进本核私有 TensorMap，以保持每核副本一致 |
| **loser**，`tensormap == shared`（全局共享 map 变体） | **完全不需要** | 无 —— 全局 map 由 winner 维护 |

由于任一任务只有约 `1 / 参与核数` 的核会成为 winner，**绝大多数核都是 loser**。它们却照样付出了
完整参数块构建（尤其是 input 侧的 tensor view + `add_input`、以及 scalar 打包）的开销——这部分对
loser 而言是**纯浪费**，正是 replay 阶段偏慢的主因。

> 已有的 runtime 内部优化（[§6.4](fully_distributed_within_core.md) 的 winner-only fan-in +
> `built[]` 后置）**无法覆盖这里**：那些优化发生在 `rt_submit_*` 的**内部**；而参数块构建发生在
> `rt_submit_*` 被调用**之前**的编排代码里，等 runtime 拿到 `L0TaskArgs` 时，昂贵的打包早已完成。
> 要省掉它，必须让**竞争（认领）发生在参数块构建之前**，并让编排代码按胜负**条件化**地构建参数。

## 3. 方案：先竞争，后按胜负条件化构建参数

### 3.1 `local_cursor` + `compete_cursor` 原语（**runtime 内部，不暴露给编排**）

1. **`local_cursor` 是每核私有计数器**，带确定性初值——它就是现有的 `local_current_task_index`
   （[§2](fully_distributed_within_core.md)）。每次提交时先 `++`，得到确定性任务 id `N`（各核一致，
   与最终谁执行无关）。**它是 runtime 内部状态，编排层不需要、也不应该看到它**（见 §3.3 对“为何可以
   隐藏”的说明）。
2. runtime 在每次提交内部先做一次竞争：

   ```text
   bool compete_cursor(T, local_cursor):
       # 原子地比较 local_cursor 与该类型的全局 cursor[T]（cube / vector）
       old = atomic_fetch_max(global_cursor[T], local_cursor)
       return local_cursor > old      # TRUE=winner（并已把 global_cursor 推到 local_cursor）；FALSE=loser
   ```

   这**正是**现有 `claim()` / `atomic_fetch_max` 的语义（[§11.1](fully_distributed_within_core.md)），
   区别仅在于：把它**提前到参数块构建之前**执行，使竞争先于（且门控）昂贵的参数打包。分片 cursor
   （[§6.6](fully_distributed_within_core.md)，`N % G`）与两条 cursor（cube/vector）语义不变。

   > **`compete_cursor` 不必作为公开 API 暴露给编排。** 它是 `rt_submit_*` 提交路径的一个内部步骤：
   > runtime 在收到本次提交后，自行 `local_cursor++`、调用 `compete_cursor` 得到 winner/loser，再据此
   > 决定是否/如何回放参数（§3.3）。编排只描述“这个任务要哪些 input/output/inout”，不感知 cursor 与
   > 竞争。

### 3.2 条件化编排（由 codegen 生成）

编排在每个 submit 点改为生成如下结构：

```text
local_cursor++                                   # 任务 id N（确定性、各核一致）
win = compete_cursor(T, local_cursor)            # 先竞争，再决定要不要打包参数

if win:
    # —— winner 路径：与当前 rt_submit_* 完全等价，唯一区别是不再重复 claim ——
    #     （认领已在上面的 compete_cursor 里完成，此处直接进入构建）
    构建【完整】参数块：add_input / add_output / add_scalar（含 input 侧 tensor.view）
    update_tensormap(task)                        # 查 INPUT/INOUT→fanin；插 OUTPUT/INOUT→producer=N
    构建任务进本核私有环（带 fanin）
    ... 任务检查 / 执行（Phase B，多核则 winner-gated launch，§3.1）
else if tensormap == shared:
    # —— loser + 共享 map：什么都不做 ——
    pass
else:  # loser + tensormap == private（本设计默认的每核复制模型）
    构建【仅 output】参数块：output create-info + 确定性分配（§9.3）
    仅把 OUTPUT/INOUT 作为 producer=N 插入本核私有 TensorMap
    # 跳过：input 侧 tensor.view、add_input、add_scalar、fan-in lookup、任务构建/执行
```

- **winner 路径**行为与今天的 `rt_submit_*` 逐字等价（TensorMap → 私有环构建 → 执行），**唯一区别是
  不再重复执行 claim**——认领这一步已经由前面的 `compete_cursor` 完成（`compete_cursor` 本身就是那次
  claim），winner 分支直接从“已认领成功”的状态进入构建，不再碰 cursor。
- **loser + shared**：完全跳过。
- **loser + private**：只构建并登记 output，跳过 input 侧打包与所有 winner 专属工作。

省掉的正是 loser 身上最贵的部分——input 侧的 `tensor.view` + `add_input` + `add_scalar`
（`prof_tensor_view` + `prof_param_setup`），而这些对 loser 本就没有任何用途。

### 3.3 API 形态：单个 builder 回调 + runtime 自行决定

**目标 API（推荐）。** 编排只提供**一份**参数清单——通过 builder 调用 `add_input` / `add_output` /
`add_inout`——由 `rt_submit_aic_task` **在内部**完成 `local_cursor++` → `compete_cursor` →
按 winner/loser × tensormap-mode **自行决定回放哪些项**。`local_cursor` 与竞争都藏在 API 后面，编排
不感知：

```cpp
// 单个 builder 回调；runtime 内部 local_cursor++ → compete → 决定回放策略。
// input / output / inout 全部以惰性 thunk 登记，runtime 按角色 × map-mode 选择性求值。
rt_submit_aic_task(FUNC_PV_MATMUL, [&](SubmitBuilder &b) {
    b.add_input([&] { return pij_f16; });                                 // 仅 winner 求值
    b.add_input([&] { return value_cache.view(kv_shapes, kv_offsets); }); // 仅 winner 求值
    b.add_output([&] { return tile2d_ci; });                              // winner + loser(private) 求值；shared 跳过
    // b.add_inout([&] { return x; });                                    // produce: winner+private；consume: winner；shared 全跳
    // b.add_scalar([&] { return scale_value; });                         // 仅 winner 求值（同 input）
});
```

runtime 侧回放策略（一次回调即可覆盖三种情况）：

| 结果 | `add_input` / `add_scalar`（惰性项） | `add_output` / `add_inout` 的 produce 侧（惰性项） | `add_inout` 的 consume 侧 |
| ---- | ------------------------------------ | ------------------------------------------------- | ------------------------- |
| **winner** | 求值 + 打包（用于 build/exec） | 求值 + 分配 + build | 求值（fan-in lookup） |
| **loser + private** | **跳过（不求值）** | 求值 + 确定性分配 + 插入 map（§9.3、§4） | 跳过 |
| **loser + shared** | **跳过** | **跳过**（连 output/inout thunk 都不求值，零开销；map/堆全局共享，见 §4 第 4 条约束） | 跳过 |

**为什么 input / output / inout 全部要以惰性项（lambda/thunk）登记——这是能否省下开销的关键。** 昂贵
的构建（如 input 侧 `tensor.view`、output 的 create-info 组装）若写成 `b.add_output(tile2d_ci)` 这类
**即时求值**，C++ 会在调用 `add_output` **之前**就把参数求值掉；等 runtime 决定“这是 loser，跳过”时，
构建早已执行——**优化落空**。因此三类参数都必须以 `[&]{ return <表达式>; }` 交给 builder，runtime 才能
对不需要的项**根本不求值**。分角色看：

- **input**：只有 winner 需要（build/exec）；loser 一律跳过。
- **output / inout 的 produce 侧**：winner 与 **loser+private** 需要（确定性分配 + 插入 map，§9.3/§4）；
  但 **loser+shared 不需要**（map/堆全局共享）。**正因为要让 loser+shared 也跳过 output/inout 的构建、
  省下这一档 replay 开销，output/inout 同样必须惰性**——否则即时求值会在 shared 场景下白白付出构建成本。
- **inout 的 consume 侧**（fan-in lookup）：winner-only。`add_inout` 的双重身份（produce + consume）由
  builder 内部按角色区分，编排只写一次。

**这样就同时回答了两个问题：**

- **单一清单 + runtime 决定？可以。** 只要 input 项惰性化，runtime 就能对 winner 全量回放、对
  loser+private 只回放 output/inout-produce、对 loser+shared 全部跳过——**编排只写一份清单**，无需
  `build_full` / `build_outputs` 两个闭包。
- **`local_cursor` 可以隐藏？可以。** 它就是每核私有的 `local_current_task_index`，由 `rt_submit_*`
  内部推进；竞争也在内部完成。编排层完全不需要看到 cursor。

**落地。** runtime 侧新增一个 ops 表项承接这种“单 builder 回调 + 竞争优先 + 条件回放”的提交路径；
`SubmitBuilder` 的 `add_input` / `add_output` / `add_inout` 均接收惰性 thunk，runtime 按角色 × map-mode
选择性求值（input=winner；output/inout-produce=winner+private；inout-consume=winner；shared 全跳）。旧的
`rt_submit_*(kernel_id, L0TaskArgs)` 保留兼容（相当于所有项立即求值、winner 全量、无 loser 优化）。

> **权衡与退化行为。** 若某项图省事仍写成即时求值（如 `b.add_output(tile2d_ci)`），API 仍然**正确**
> ——只是该项对本可跳过它的 loser 也执行了，退回到“省不掉这一项”的旧成本。换言之，惰性化是**逐项可选
> 的性能手段**，不影响正确性；codegen 应默认对 input / output / inout 三类都生成惰性形式。

> **codegen 视角。** 用户所说的 “generating more optimized orchestration function” 即由代码生成器
> 直接产出上述 compete-first 结构：为每个 kernel 生成一份 `SubmitBuilder` 回调，**把 input / output /
> inout 三类参数都自动包成惰性 thunk**（`[&]{ return <表达式>; }`）；竞争与 `local_cursor` 全部交给
> runtime。手写编排也可按同一形态改写。完整的 codegen 改进方案见 §8。

### 3.4 所有参数类型的惰性归类（`add_scalar` / 显式依赖 / launch_spec 等）

`L0TaskArgs`（`Arg`）当前提供的参数登记 API 不止 `add_input/output/inout`。要把 replay 成本压到最低，
**除“kernel 身份”外的所有参数都应以惰性 thunk 登记**，再由 runtime 按“谁需要”分三档求值。归类如下：

| 参数 API | 类别 | 求值时机（惰性档） | 理由 |
| -------- | ---- | ------------------ | ---- |
| kernel 身份：`MixedKernels`（`aic/aiv0/aiv1_kernel_id` + `active_mask`） | **Tier 0：eager，不惰性** | **compete 之前**（所有核） | compete 要靠它判定任务**类型**（cube/vector）以选 `cursor[T]` 与分片 `N%G`（§3.1、§6.6）；必须先于认领可知。它作为 `rt_submit_*` 的直接实参传入，不进 builder |
| `add_output`（`TensorCreateInfo`→分配 / 既有 Tensor 写目标） | **Tier 1：winner + loser(private)** | winner 或 loser+private 求值；loser+shared 跳过 | 需据其大小做确定性分配（§9.3）并把 producer=N 插入本核私有 map（§4）；shared 变体由 winner 维护全局 map，loser 免 |
| `add_inout` 的 **produce 侧** | **Tier 1：winner + loser(private)** | 同上 | INOUT 既产出新版本（登记 producer）又消费旧版本；产出侧同 output |
| `add_input` | **Tier 2：winner-only** | 仅 winner | 只用于 build/exec；loser 从不消费输入 |
| `add_inout` 的 **consume 侧**（fan-in lookup） | **Tier 2：winner-only** | 仅 winner | 消费侧的依赖解析是 winner-only（§6.4） |
| `add_scalar` / `add_scalars` / `add_scalars_i32` / `copy_scalars_from` | **Tier 2：winner-only** | 仅 winner | 标量是 kernel 执行参数，不参与 map/分配；loser 不需要 |
| 显式依赖 `add_dep` / `set_explicit_deps` | **Tier 2：winner-only** | 仅 winner | 显式 fan-in producer id，仅供 winner 依赖轮询 |
| `add_no_dep`（输入角色的张量） | **Tier 2：winner-only** | 仅 winner | 不建依赖的输入，仅 build/exec 用 |
| `launch_spec`（SPMD `block_num` 等）/ `set_allow_early_resolve` | **Tier 2：winner-only**（若纯执行元数据） | 仅 winner | 执行期元数据。**例外**：若某字段会影响任务类型/所有权判定，则须上提到 Tier 0 |

**结论（直接回答“`add_scalar` 等是否也要惰性”）：是。** 除 kernel 身份（Tier 0，必须 eager 以供 compete
选 cursor）外，**其余全部参数——包括 `add_scalar`、显式依赖、`add_no_dep`、`launch_spec`——都应以惰性
thunk 登记**，让 loser+shared 能整个跳过、loser+private 只求值 Tier 1。标量本身打包很廉价，惰性化对它
的直接收益有限，但纳入是为了：(a) 让 loser+shared 真正**零开销**；(b) API 统一，codegen 无需对参数类型
特判。

> **一处必须遵守的约束：输出句柄的数据流。** `add_output` 返回的 Tensor 句柄常被**后续 submit 点**当作
> 输入引用（如 `oi_tmp` 喂给 online_update）。其地址是任务 id 的**纯函数**（§9.3 确定性布局），因此可在
> 任意核上**确定性重建**。为使 loser+shared 跳过 output 后、下游仍能正确引用：**输出句柄只能在惰性参数
> thunk 内被消费，不得进入控制流**；下游对该张量的解析要么走（共享/私有）TensorMap 按区域查找，要么由
> codegen 生成确定性重建（§8.3）。这样，哪个核赢得下游任务，它的 input thunk 就在该核上按确定性地址取到
> 句柄——无需本核曾经 build 过上游 output。

## 4. 正确性论证

1. **任务 id 不变、各核一致。** `local_cursor` 仍是每核私有、单调 `++`，竞争只决定“谁执行”，
   **不改变任务 id**（[§2](fully_distributed_within_core.md)）。所有核仍走完全相同的确定性 submit
   序列——把竞争提前不影响 id 分配。
2. **winner 行为不变（但不重复 claim）。** winner 分支就是今天的 `rt_submit_*`，TensorMap 更新、
   私有环构建、winner-gated launch（[§3.1](fully_distributed_within_core.md)）、执行全部保留；**唯一
   差别是不再执行 claim**——认领已由前面的 `compete_cursor` 完成（它就是那次 `atomic_fetch_max`
   claim，§11.1），winner 分支不再碰 cursor，避免对同一任务做第二次原子认领。
3. **private TensorMap 每核副本一致（关键）。** producer 条目只在处理 `OUTPUT`/`INOUT` 时创建，
   且**必须每个核都创建**，否则本核上的下游消费者会查不到（[§4](fully_distributed_within_core.md) 的
   “为什么部分 map 是错的”）。因此 loser+private **仍求值 output/inout-produce thunk 并登记**——这正是
   本方案对 loser+private 仍回放 output/inout 项的原因。相对地，**input 侧 lookup 只有 winner 需要**（消费者的 fan-in 只
   给 winner 用于依赖轮询），loser 跳过 input 打包与 lookup 安全无误——这与
   [§6.4](fully_distributed_within_core.md) 的 winner-only fan-in 完全同源，只是把它从 runtime 内部
   进一步上推到了编排层。
4. **确定性 heap 布局。** `heap_top` 由每个核对每个任务**无条件**确定性推进
   （[§9.3](fully_distributed_within_core.md)），依赖 output 的大小。loser+private 构建 output
   create-info 后照常做确定性分配，布局不漂移。**loser+shared 若要整体跳过**，前提是该变体的输出堆
   也是**全局共享分配**（而非每核复制 bump）——这是 `tensormap == shared` 变体的配置约束，需与
   [§9](fully_distributed_within_core.md) 的分配模型配套。
5. **多核任务不受影响。** anchor/follower、launch、joint 完成计数
   （[§3.1](fully_distributed_within_core.md)）都在 winner 分支内，loser 不参与，语义不变。

## 5. 一处重要 caveat：区分“控制流读取”与“纯参数打包”

编排里有些量来自 `get_tensor_data(...)`（如 `paged_attention_orch.cpp` 读 `context_lens` /
`block_table` 得到 `cur_seq`、`cur_block_idx`、循环上界等）。这些是**驱动控制流**的读取——决定了
后续会 submit 哪些任务、循环走多少次——因此**必须由所有核执行**，不能移进 winner-only 闭包，否则
各核的 submit 序列会分叉、任务 id 不再一致。

因此 codegen（或手写改写）必须严格区分：

- **控制流读取 / 索引计算**：保留在提交调用**之前**、所有核都执行的公共路径里（在 `SubmitBuilder`
  回调**之外**）。
- **纯参数打包**（`tensor.view` 后 `add_input` 等）：才可放进 builder 回调、以惰性 input 项交给
  runtime 条件求值。

对 loser+private，回调里 output 的形状若依赖某控制流量，该量已在公共路径算好，惰性/直接项只做纯打包，
无副作用；被跳过的 input thunk 同理不含控制流读取。

## 6. 预期收益

- **loser 的 replay 成本**：从“完整参数块”降到 **outputs-only（private）** 或 **零（shared）**，省掉
  input 侧 `tensor.view` + `add_input` + `add_scalar`（profiling 中 `prof_tensor_view` +
  `prof_param_setup` 的主要部分）。
- **摊销随核数放大**：核越多，单核赢得的任务越少、走 loser 快路径的比例越高——省得越多，正好补上
  [§6.2](fully_distributed_within_core.md) 里“SPMD 冗余重放随核数近线性增长”的开销。
- **与 [§6.4](fully_distributed_within_core.md) 正交叠加**：§6.4 省的是 `rt_submit_*` **内部**的
  fan-in lookup 与 `built[]` 拷贝；本方案省的是 `rt_submit_*` **之前**的参数打包（tensor view +
  `add_*`），量级更大。二者相加把 loser 的整条 submit 路径压到接近“只推进 cursor + 登记 output”。

## 7. 落地清单

| 改动 | 位置 | 说明 |
| ---- | ---- | ---- |
| 新增“单 builder 回调 + 竞争优先 + 条件回放”提交路径 | orchestration API（`PTO2RuntimeOps` 加 ops 项）+ dist_engine submit runtime | `rt_submit_*` 内部：`local_cursor++` → `compete_cursor`（= 现有 `claim()`/`atomic_fetch_max`，§11.1）→ 按 winner/loser × map-mode 回放。`local_cursor` 与竞争**不暴露**给编排 |
| `SubmitBuilder`：input / output / inout 全惰性 | orchestration API | 三类均以 thunk 登记；求值策略：input=winner；output/inout-produce=winner+private；inout-consume=winner；loser+shared 全跳 |
| 保留旧 `rt_submit_*(kernel_id, L0TaskArgs)` | orchestration API | 兼容路径：所有项立即求值、winner 全量、无 loser 优化 |
| codegen 生成 builder 回调、input 自动惰性化 | Codegen（`examples/`） | 把 input 参数包成 `[&]{ return <view>; }`；控制流读取留在回调之外的公共路径（§5） |
| `tensormap == shared` 变体的堆分配配套 | dist_engine 内存管理（§9） | shared 变体需全局共享分配，loser 方可整体跳过（§4 第 4 条约束） |

## 8. PYPTO 前端 / orchestration codegen 改进方案

本章给出**完整**的代码生成侧改进方案：PYPTO 前端把用户的图/DSL 降级（lower）为 AICore 上重放的
orchestration function（`aicpu_orchestration_entry`）。要落地 §3 的 compete-first + 单 builder 惰性
提交，codegen 的产物形态必须改变。本章描述改动目标、核心变换、所需分析、边界情况、分阶段落地与验证。

### 8.1 现状：codegen 生成的 orchestration 形态

今天 codegen 为每个任务生成**直线式（straight-line）**代码，且在 `rt_submit_*` **之前内联**完成参数
打包（以 `paged_attention_orch.cpp` 的 PV matmul 为原型）：

```cpp
// 控制流读取（索引/边界，来自 tensor 数据）
uint64_t cur_block_idx = get_tensor_data<int32_t>(block_table, 2, bt_idx);
Tensor vj = value_cache.view(kv_shapes, kv_offsets);   // ← 纯参数打包（view）
// 参数块内联构建（每个核都执行）
L0TaskArgs params_pv;
params_pv.add_input(pij_f16);
params_pv.add_input(vj);
params_pv.add_output(tile2d_ci);
TaskOutputTensors pv_outs = rt_submit_aic_task(FUNC_PV_MATMUL, params_pv);
const Tensor &oi_tmp = pv_outs.get_ref(0);             // ← 输出句柄，喂给下游任务
```

问题（§1–§2）：`view` + `add_*` 这段**纯参数打包**被每个核无条件执行，而它对 loser 是浪费。

### 8.2 目标形态：compete-first + 单 builder 惰性回调

codegen 改为对每个任务生成一次 `rt_submit_*(FUNC, builder_lambda)`，其中：

- **kernel 身份**（`FUNC` / `MixedKernels`）作为直接实参（Tier 0，eager，供内部 compete，§3.4）。
- **参数打包**全部搬进 `builder_lambda`，并按方向以**惰性 thunk** 登记（§3.3、§3.4）。
- **控制流读取 / 索引计算**留在 lambda **之外**的公共路径（所有核都执行，§5）。

```cpp
// 控制流读取仍在公共路径（所有核执行）
uint64_t cur_block_idx = get_tensor_data<int32_t>(block_table, 2, bt_idx);
uint32_t kv_offsets[2] = {static_cast<uint32_t>(cur_block_idx * block_size), 0};
// 参数打包进单 builder 回调，全部惰性
auto pv = rt_submit_aic_task(FUNC_PV_MATMUL, [&](SubmitBuilder &b) {
    b.add_input([&] { return pij_f16; });
    b.add_input([&] { return value_cache.view(kv_shapes, kv_offsets); });  // view 惰性，仅 winner 求值
    b.add_output([&] { return tile2d_ci; });
});
const Tensor &oi_tmp = pv.get_ref(0);   // 确定性句柄（§8.3）：地址 = f(task_id)
```

### 8.3 codegen 的核心变换

| 变换 | 内容 | 依据 |
| ---- | ---- | ---- |
| **T1 参数惰性化** | 把每条 `add_input/output/inout/scalar/dep/no_dep` 连同其实参表达式（尤其 `tensor.view(...)`）包成 `[&]{ return <expr>; }` 交给 `SubmitBuilder`；按 §3.4 归类到 Tier 1/2 | §3.3、§3.4 |
| **T2 控制流/打包分离** | 对每个任务的输入做 def-use 分析：凡是**驱动后续控制流或索引**的值（`get_tensor_data`、循环边界、`view` 的 offset/shape 计算）留在 lambda 外的公共路径；只有**最终打包**（`view` 调用 + `add_*`）进 thunk | §5 |
| **T3 输出句柄确定性化** | 上游 `add_output` 返回的句柄若被下游任务引用，改为**确定性重建**（地址 = `f(task_id)`，shape = create-info）或按 TensorMap 逻辑区域解析，并保证句柄**只在下游的惰性 thunk 内被消费**、不进控制流 | §3.4 约束、§9.3 |
| **T4 kernel 身份前置** | 把 `MixedKernels`（`active_mask`）作为 `rt_submit_*` 直接实参，确保 compete 在参数打包前可判定任务类型 | §3.1、§3.4 |
| **T5 scope/循环不变量外提** | 保持 `PTO2_SCOPE` 结构不变；把 loop-invariant 的 `TensorCreateInfo`（如 `tile2d_ci`）仍在循环外构造一次，thunk 内只引用 | §9.4 |

### 8.4 所需的前端分析

- **def-use / SSA**：识别每个任务参数表达式的依赖链，判定它是否**流入控制流**（决定后续 submit 序列）。
  流入控制流 → 归公共路径（Tier 0 语义，所有核执行）；否则 → 可进惰性 thunk。**保守规则**：拿不准时归
  公共路径（宁可不省，不可让 submit 序列分叉）。
- **方向标注**：codegen 本就知道每个张量参数是 INPUT/OUTPUT/INOUT（它据此生成 `add_*`），直接映射到
  §3.4 的 Tier 分类，无需额外推断。
- **句柄生存期**：追踪 `add_output` 返回句柄的所有使用点；若存在**控制流使用**（极少见），必须把该输出的
  确定性重建上提为无条件执行（退化为不省该项），并告警。

### 8.5 边界情况

- **控制流依赖 tensor 读**（如 `context_lens` 决定循环次数）：始终在公共路径、所有核执行——这是 §5 的
  硬约束，codegen 不得下沉进 thunk。
- **动态 shape**：shape 计算若依赖控制流读，其结果已在公共路径可得；thunk 内 `view` 只做纯构造。
- **INOUT 的双侧**：codegen 对一个 `add_inout` 生成一个 thunk，runtime 内部按角色分别用于 produce 登记
  （Tier 1）与 consume lookup（Tier 2）；codegen 不需要拆成两个。
- **`tensormap == shared` 变体**：codegen 产物**不变**（同一份惰性 builder）；private/shared 的差异完全在
  runtime 回放策略里（§3.3 表）。但 shared 变体要求输出堆全局共享分配（§4 第 4 条），且下游解析必须走
  共享 map / 确定性地址（T3）。
- **错误处理**：`SubmitBuilder` 沿用现有 `has_error`/`report_fatal` 路径；thunk 内构造失败仍能上报。

### 8.6 分阶段落地

1. **runtime 先行**：新增 `SubmitBuilder` + “单 builder 回调”ops 项与 compete-first 提交路径；旧
   `rt_submit_*(kernel_id, L0TaskArgs)` 保留（§7）。此步不改 codegen，用手写用例验证语义。
2. **codegen 双模式**：codegen 增加一个开关，可继续输出旧的直线式，也可输出新的 builder 形态；先对
   `paged_attention` 等基准用例产出新形态。
3. **T2/T3 分析接入**：实现 def-use 分离与输出句柄确定性化；对无法安全惰性化的项自动退化为即时求值
   （正确性优先，§3.3 退化说明）。
4. **全量切换 + 清理**：所有用例校验通过后，将新形态设为默认；旧路径转为兼容/回归对照。

### 8.7 验证

- **Golden 一致**：新旧 codegen 产物在全部用例（bgemm / paged_attention / paged_attention_ringbuffer /
  mix_coown 等）上结果逐位一致——因为惰性化只改“何时/是否求值”，不改任务 id 与 map 内容（§4）。
- **确定性不变量**：抽查各核的 per-core map 与 `heap_top` 演化一致（§4、§9.3）。
- **性能**：用 §6.2/§6.3 的 skip-exec 口径与 `paged_attention_orch.cpp` 自带 profiling，对比新旧
  `prof_param_setup` + `prof_tensor_view` 及片上编排墙钟，验证 loser 快路径确实压低了 replay。

## 9. 相关文档

| 文档 | 关联性 |
| ---- | ------ |
| [fully_distributed_within_core.md](fully_distributed_within_core.md) | 本增强的主设计文档；见 §6.8（本方案的并入版）、§4（TensorMap 一致性）、§6.4（winner-only fan-in）、§9.3（确定性布局） |
