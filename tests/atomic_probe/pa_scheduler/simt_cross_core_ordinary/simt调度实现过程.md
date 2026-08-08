# SIMT 串行 TensorMap 调度实现过程

## 1. 基线与用户约束

- 分支：`fdwic-swimlane-deps`
- 起点：`678987515baec7fa7dee0adef72050ece66dfb1e`
- 新目录必须独立，不产生对 `simt_cross_core_dag` 的源码依赖；
- metadata 串行语义必须参考真实 `cross_core_ordinary`，不能臆造接口；
- 只做 GM，不做 UBUF；
- workload 固定 `QK/SF/PV/UP=6/28/4/1`；
- 需要真实 PA、泳道图和与 symbol 路径的性能对比。

## 2. 查证得到的真实 `cross_core_ordinary` 规则

`cross_core_ordinary/common/pa_shared_submit_path.h` 的
`DecodeSharedMetadataWriterPlan()` 从只读 writer bitset 中求
`previous_metadata_writer=max(writer<N)`；`FinishSharedWinnerSubmitBody()` 只在
Register 阶段等待该前驱。非 writer 不交接 baton，真实 writer 才发布自己的
insert completion。

因此新路径实现的是 256-writer 稀疏链，不是 1280-task 全链，也不是按 PA
五类 task 直接硬编码前驱。

## 3. 先失败再实现的 CPU 合同

先在复制后的独立 CPU 测试中增加 `TestOrdinaryGlobalWriterContract`，要求：

- contract 能编码/解码 previous global metadata writer；
- 每个真实 writer 最多只有一个全局 writer 前驱；
- B256 恰有 256 个 writer 和 255 条 predecessor edge；
- history 仍保存每个 symbol 的精确 previous writer；
- 最终 last-writer、history、execution、completion 和 FinalDrain oracle 不变。

测试最初因缺少 `PreviousMetadataWriterTask()`、
`MetadataInsertPreviousWriter()` 及对应 contract bit field 而编译失败。随后才补充
协议和 device/CPU 实现。

## 4. 实现内容

### 4.1 公共合同

在 metadata insert contract 中保留 present bit 与 writer count，并新增 16-bit
`previous_writer+1`。`0` 表示没有前驱。前驱通过通用 writer intent 反向扫描，
不检查 PA `TaskKind`。

### 4.2 CPU 语义模型

每个 writer 先等待 intent 引用的 producer output，再等待一个 global previous
writer 的 `insert_completion`。取得 turn 后仍写入逐 symbol history；初始闭环用
精确 logical previous writer 做 last-writer CAS，后续优化在证明全局 commit 区间
单写后改为 release store，最后发布自身 completion。

### 4.3 CCEC device

SIMT leader 使用与 CPU 相同的 contract：

- `SimtPreviousMetadataWriterTask()` 计算全局稀疏前驱；
- `SimtCommitTask()` 等 producer published；
- 仅在有前驱时轮询 `insert_completion[prev]`；
- history、last-writer 发布、fence、BUILT 和 executor 路径保持不变。

独立 kernel entry 为：

- `simt_cross_core_ordinary_g0_0_mix_aiv`
- `simt_cross_core_ordinary_g0_0_mix_aic`

### 4.4 泳道图

schema 为 `simt_cross_core_ordinary_g0_swimlane_v1`，并显式记录：

- `metadata_insert_order=global_sparse_task_id_writer_chain`
- `metadata_writer_tasks=256`
- `metadata_insert_predecessor_edges=255`
- writer phase：`serial_metadata_insert`
- non-writer phase：`publish_without_metadata_insert`
- atomic site：`simt_global_insert_predecessor_poll`

这样不会把 1280 个 task 都误画成 TensorMap insert。

## 5. 初始闭环验证结果

### 5.1 CPU 与静态构建

`./run.sh build-gm` 已通过：

| 验证 | 覆盖 | 结果 |
| ---- | ---- | ---- |
| optimized CPU | builders 1..32，B1/B256，4 rounds | PASS |
| ASan/UBSan | builders 1..32，B1/B256，2 rounds | PASS |
| TSan | builders 1..32，B1/B256，2 rounds | PASS |
| CCEC AIC | `dav-c310-cube` | PASS |
| CCEC AIV | `dav-c310-vec` | PASS |
| GM-only 源码审计 | 无 `__ubuf__`、U2 宏、UBUF include/entry | PASS |
| optimized bitcode | SIMT atomic/fence、DCCI、Vector/Cube 指令 | PASS |
| 1:2 mixed ELF | 精确双 entry/function/metadata | PASS |
| AIV local memory | base 192+8/224 KiB；trace 192+16/224 KiB | PASS |
| GCC15 ACL host | host 与 manifest | PASS |

### 5.2 真实 A5 功能

设备：`Ascend950PR_958b`，device 0。环境没有 `npu-smi` 与 `task-submit`，按本机
既有显式授权采用 unlocked 运行，因此性能数据可能受同卡其他进程干扰。

| Case | 配置 | 结果 |
| ---- | ---- | ---- |
| 最小闭环 | B1、W16、1 batch、3 runs | 3/3 PASS |
| 完整 PA | B8、W5、256 batch、5 runs | 5/5 PASS |
| 公平性能样本 | B8、W5、256 batch；每条路径 3×21 runs | 126/126 PASS |
| 泳道图 | B8、W5、256 batch、1 run、trace-on | PASS |

完整 PA 每轮均验证：1280 active tasks、1024 kernel tasks、每 task 一次 Build、
每 builder 160 个 winner、同设备地址、206569472 heap bytes 和完整 golden。

## 6. 与 per-symbol 路径的公平性能对比

两条路径在同一台设备上交替运行；均为 B8/W5、B256、固定 `6/28/4/1`、
trace-off。最初一组 11 次 A/B 得到 943.790/950.283 µs，但后续复跑发现同一个
binary 的 builder 包络也会跨窗口漂移约 15 µs，因此最终采用每条路径 3 个
独立 21-sample 窗口，并只比较窗口中位数。

<!-- markdownlint-disable MD013 MD060 -->

| 21-sample 窗口 | per-symbol kernel median | per-symbol builder median | global-writer kernel median | global-writer builder median |
| -------------- | ------------------------ | ------------------------- | --------------------------- | ---------------------------- |
| 1              | 934.874 µs               | 846.130 µs                | 931.939 µs                  | 850.042 µs                   |
| 2              | 931.673 µs               | 826.004 µs                | 940.614 µs                  | 864.150 µs                   |
| 3              | 935.272 µs               | 826.058 µs                | 938.615 µs                  | 850.151 µs                   |
| 三个窗口中位数的中位数 | 934.874 µs               | 826.058 µs                | 938.615 µs                  | 850.151 µs                   |

<!-- markdownlint-enable MD013 MD060 -->

按最后一行计算，global-writer 的 kernel E2E 为 `+3.741 µs`（`+0.40%`），
builder 包络为 `+24.093 µs`（`+2.92%`）。但更重要的是区间关系：

- per-symbol E2E 窗口为 931.673..935.272 µs；global-writer 为
  931.939..940.614 µs，区间重叠，当前不能证明端到端存在稳定回退；
- per-symbol builder 为 826.004..846.130 µs；global-writer 为
  850.042..864.150 µs，区间不重叠，串行 writer 等待确实增加了 Build 时间；
- ordinary 的真实运行记录到数百次 `insert_polls`，symbol 路径为 0；新增
  Build 成本大部分与 executor 工作重叠，没有等量落到 kernel E2E；
- 功能上仍只有 255 条 writer edge，没有扩散成 1280-task baton。

以上是 unlocked 单卡上的工程数据，不是隔离实验。若要给出统计显著的 E2E
结论，需要在独占卡上用同一套交替采样重新测量。

历史目录中的 B8/W5 symbol 记录曾得到 926 µs；它不是本次相邻 A/B 的同一
采样窗口，因此只作历史参考，不拿来计算上表差值。

## 7. 泳道图结果

修正版文件位于 `test_record/2026-8-7/`。结构校验结果：

| 字段/事件 | 数值 |
| --------- | ---- |
| workload | `6/28/4/1` |
| tasks / kernel tasks | 1280 / 1024 |
| metadata writer / edge | 256 / 255 |
| `serial_metadata_insert` | 256 |
| `publish_without_metadata_insert` | 1024 |
| `task_execute` / kernel projection | 1024 / 1024 |
| trace-on kernel E2E | 1818.676 µs |
| device trace span | 1230.557 µs |
| SIMT build span | 1200.524 µs |
| SIMT/Scalar atomic calls | 18661 / 109143 |
| Scalar DCCI calls/lines | 14848 / 14848 |

## 8. 初始闭环结论

新目录已闭合“SIMT 构建 + `cross_core_ordinary` 全局稀疏 writer 顺序 + Scalar 执行”的
真实 GM PA 流程。它与 per-symbol 路径只在 metadata 排序粒度上不同，适合作为
后续 correctness/performance A/B 基线。当前没有实现 UBUF，也没有把 PA
whole-object 结果外推成通用 ordinary-region TensorMap 能力。

## 9. 0.8 ms 目标的优化过程

后续优化始终保持固定 workload `6/28/4/1`、1280-task DAG、1024 个 kernel
task、GM execution cell 和完整 host oracle。没有按 PA task id 写特例，也没有
减少计算量。性能口径均为 ACL event 覆盖的完整 kernel、trace-off；设备为
unlocked device 0，因此相邻采样仍可能有少量漂移。

### 9.1 上游优先派发，消除 token 队头阻塞

初始派发表在每个 engine 内按 batch 交错：AIC 为 QK/PV 交错，AIV 为 SF/UP
交错。下游 PV/UP 很早占住执行 token 后，会等待上游 completion；此时 executor
不能继续领取尚未执行的 QK/SF，形成控制层队头阻塞。

优化后在同一个 engine 内先派发全部上游任务，再派发全部下游任务：

```text
AIC: all QK -> all PV
AIV: all SF -> all UP
```

这只改变 ready task 的派发顺序，不改变 task id、fanin 或 TensorMap 插入顺序。
B12/W5/token4 从约 939 µs 降到 842..850 µs；保存的阶段泳道图对应 trace-off
843 µs。收益来自 executor 不再让未 ready 的下游任务长期占住 token。

### 9.2 去掉全局单写区间中的冗余 per-symbol CAS

ordinary 协议已用 `previous_metadata_writer` 把 256 个真实 writer 串成唯一的
全局 commit 顺序。writer 取得 turn 后，commit 区间不存在第二个合法 writer，
所以 last-writer CAS 不再提供互斥，只重复增加 atomic 串行化。

最终实现仍保留精确 per-symbol history，并由 host/CPU oracle 逐条验证；派生
last-writer 改用 `asc_stcg` non-cacheable store，随后 fence，再发布
`insert_completion`。Build 包络由约 650 µs 降到约 600 µs，writer poll 量由约
2100 次降到约 800 次。该化简基于全局 writer 单写合同，不依赖算子类型；若未来
恢复不同 symbol 并发 commit，则必须恢复 CAS 或等价并发校验。

### 9.3 token 从 4 减到 1

上游优先派发后，executor 不再需要用多个 token 容纳被下游依赖阻塞的 task。
继续保留过多 token 只会增加并发 poll 和 Scalar 控制压力。B12/W5 的相邻结果为：

| token/owner | kernel 中位数 | 结论 |
| ----------- | ------------- | ---- |
| 8 | 978.908 µs | atomic/poll 压力显著变大，回退 |
| 4 | 约 850 µs | 上游优先阶段基线 |
| 2 | 821.942 µs | 继续下降 |
| 1 | 约 802..805 µs | 最优，作为最终默认值 |

### 9.4 builder 数与 warp 数收敛

固定 W5/token1/window256 后扫描 builder 数。不同点位来自 unlocked 设备上的
相邻短窗口，只用于找甜点区，不把 1..10 µs 的差异解释为稳定统计结论。

| builder 数 | 端到端中位数 | 观察 |
| ---------- | ------------ | ---- |
| 6 | 1106.347 µs | Build 供给不足 |
| 7 | 975.054 µs | Build 仍在关键路径 |
| 8 | 858.418 µs | 接近甜点区但仍明显偏慢 |
| 9 | 797.772 µs（首个 31 次窗口）；重建后 798.371 µs | 最优且两轮达到目标 |
| 10 | 约 802..809 µs | 与甜点区接近 |
| 11 | 808.493 µs | 略慢 |
| 12 | 约 802..805 µs | 先前默认候选 |
| 13 | 约 803..807 µs | 略慢 |
| 14 | 806.504 µs | 略慢 |
| 15 | 817.915 µs | builder 占用开始挤压 AIV executor |
| 16 | 811.576 µs | 无继续增加收益 |

W5 是当前最优。额外用 B8/W6 做了 21 次检查：Build 中位数确实从 B8/W5 的
821.054 µs 缩短到 603.536 µs，但 kernel E2E 反而从 858.418 µs 增到
932.337 µs。也就是说，更多 SIMT leader 能更早结束 Build，却会在 Build/Execute
重叠区增加全局 writer 等待与 GM atomic 压力，使 Scalar executor 变慢。
builder/warp 不是越多越好；最终仍保持 B9/W5。

### 9.5 派发表改为一次性 non-cacheable 读取

每个 ticket 的 dispatch pair 在 kernel 开始前已经不可变。最终用一次
`LoadDev64` 读取成对的 task id，替代逐 ticket 的 DCCI+DSB+普通 load；这使最终
泳道图中的 dispatch DCCI 为 0，并避免 1024 次重复 invalidate。该项没有观察到
稳定、可单独归因的端到端降幅，但一致性合同更直接：immutable GM 表只读一次，
不再伪装成轮询数据。

## 10. 验证过但未保留的路线

<!-- markdownlint-disable MD013 -->

| 实验 | 结果 | 未保留原因 |
| ---- | ---- | ---------- |
| dispatch window 128 | 877.560 µs | 上游窗口过小，重新引入下游占 token |
| dispatch window 192 | 962.060 µs | 本窗口明显回退；最终保持 256 |
| writer poll backoff 16/64 | 无改善 | 延迟 baton 接力，没有减少关键路径 |
| 所有 writer 先 prepare、再分布式 commit | 约 1.012 ms，poll 约 1.16 万次 | 大量 writer 同时等待同一串行链 |
| 单一 SIMT sequencer 提交全部 writer | 约 2.214 ms，Build 约 2.172 ms | 单 SIMT thread 上的 atomic/store 延迟完全暴露 |

<!-- markdownlint-enable MD013 -->

上述实验代码均已从最终实现删除；保留这些数据是为了避免后续重复走同一条回退
路线。

## 11. 最终性能与验收

最终配置为 B9/W5/token1/dispatch-window256，workload 固定 `6/28/4/1`。
修改后完整重建产物并在真实 A5 连续运行 31 次，结果为：

| 指标 | 数值 |
| ---- | ---- |
| kernel E2E min | 793.452 µs |
| kernel E2E median | **798.371 µs** |
| kernel E2E average | 819.712 µs |
| kernel E2E max | 1442.208 µs |
| Build envelope min / median / max | 734.679 / 742.404 / 749.442 µs |
| correctness | 31/31 PASS |

average/max 被一次 unlocked 设备抖动拉高，所以以预先采用的中位数口径判断目标。
中位数比 0.8 ms 目标低 1.629 µs；前一个独立 31 次窗口的中位数为
797.772 µs，两轮都达到目标。这是“已达到目标”，不是宣称在独占设备上有显著的
0.8 ms 以下裕量。

最终配置最新保存的自描述 trace-on 样本为 1824.008 µs E2E、1050.913 µs
device span、1021.123 µs SIMT Build；插桩本身会显著放大时间，只能用于解释
时序，不能代替上述 trace-off 性能结论。JSON 顶层直接写出 builder 数、每
builder warp、token、dispatch window 和 last-writer 发布方式，避免只凭文件名
猜配置。

最终回归结果：

<!-- markdownlint-disable MD013 -->

| 验证 | 最终配置结果 |
| ---- | ------------ |
| CPU optimized | builder 1..32、B1/B256、4 rounds，PASS |
| ASan/UBSan | builder 1..32、B1/B256、2 rounds，PASS |
| TSan | builder 1..32、B1/B256、2 rounds，PASS |
| CCEC base / swimlane | AIC、AIV、bitcode、1:2 mixed ELF、GCC15 host 全部 PASS |
| AIV local memory | base 192+8/224 KiB；swimlane 192+16/224 KiB，PASS |
| 真实 A5 | B1 3/3、B9 性能窗口 31/31、最终重建短回归 5/5、swimlane 1/1，PASS |
| 最终 JSON | 1280 Build、256 serial insert、1024 non-writer、1024 execute；配置自描述字段与 oracle 一致 |
| 脚本/产物 | `bash -n`、4 份 JSON 解析、base/swimlane manifest hash，PASS |

<!-- markdownlint-enable MD013 -->

## 12. 最终结论

0.8 ms 目标已在固定真实 PA workload 和完整 oracle 下达到。最大收益不是缩短
单次 TensorMap 串行插入，而是先消除 executor 的 token 队头阻塞；其后通过全局
单写合同去掉冗余 CAS、把 token 收敛到 1，并在 builder/AIV executor 之间找到
B9/W5 的平衡点。最终路径仍是通用的“SIMT Build + ordinary 全局稀疏 writer
链 + Scalar Execute”协议，没有 UBUF，也没有算子定制。

## 13. 无效优化记录：错误地流水化模拟 task 负载（已回退）

### 13.1 错误是什么

曾经把 AIC workload 的相邻 repeat 改成 L1 ping-pong：在 repeat N 执行
MTE1/M/FIX 时提前搬运 repeat N+1。表面上每个 repeat 仍有两次 TLOAD、两次
TMOV、一次 TMATMUL 和一次 TSTORE，实际却把 B11/W5 的中位数从 696.268 µs
降到了 585.865 µs。

这是一项无效且愚蠢的“优化”。本测试里的串行 repeat 不是待优化的真实算子实现，
而是用来把单个 task 校准到约 50 µs 的负载发生器。流水重叠虽然没有删指令，却
直接缩短了被测 task 的服务时间；对于调度性能基准，它与把 workload 改成更小的
`1/1/1/1` 没有本质区别。585.865 µs 只能说明负载变轻，不能说明调度协议变快。

无效实验的泳道图保留作反例，不作为任何性能结论：

`test_record/2026-8-7/ordinary_gm_b11_b256_warp5_token1_traceoff586us_aic_l1_pingpong_pipeline_swimlane.json`

### 13.2 回退与防复发

`full_pa_workloads.h` 已恢复原来的逐 repeat 完全串行路径：

```text
TLOAD A/B -> wait -> TMOV -> wait -> TMATMUL -> wait -> TSTORE -> wait Scalar
```

QK/SF/PV/UP 仍固定为 `6/28/4/1`。构建脚本现在明确审计单份 L1 tile、逐 repeat
FIX→Scalar wait，并拒绝先前的双 L1/ping-pong 结构。后续除非明确要求改变 task
负载，否则不得以算子流水、减少 repeat、减少搬运或缩小矩阵的方式优化调度数据。

回退后的 B1 真机验证为 3/3 PASS；base 与 swimlane CCEC、bitcode、mixed ELF、
host 构建均通过。

### 13.3 当前有效性能

所有保留的优化都只作用于构建、派发、payload 绑定、完成发布和 final drain，
不改变 task workload。历史同一有效代码的最佳 21 次窗口为 B11/W5/token1/
window256，中位数 696.268 µs、最小 692.642 µs。

2026-08-08 在 unlocked device 0 重新构建并复测，结果如下：

| 配置 | 样本 | trace-off median | min | Build median | correctness |
| ---- | ---- | ---------------- | --- | ------------ | ----------- |
| B9/W5 | 21 | **698.949 µs** | 688.774 µs | 616.387 µs | 21/21 PASS |
| B11/W5 | 21 | 709.624 µs | 695.044 µs | 520.113 µs | 21/21 PASS |

同日 B9..B14 各 7 次短窗的中位数分别为 692.386、717.861、699.119、700.622、
698.315、704.241 µs。B11 的历史最佳与 B9 的当前长窗只差 2.681 µs，不能在
unlocked 环境下解释成稳定架构差异。可信结论是：固定 task 负载下当前最优约为
**0.70 ms**，配置甜点区为 B9..B11；585.865 µs 已作废。

当前版本的新泳道图为：

`test_record/2026-8-8/ordinary_gm_b9_b256_warp5_token1_traceoff699us_restored_serial_task_duration_swimlane.json`

### 13.4 当前瓶颈量化

新泳道图仍有插桩放大，只用于分解结构，不用它替代 trace-off E2E。256 个 QK 与
256 个 PV 的 AIC workload 数据为：

<!-- markdownlint-disable MD060 -->

| task 类别 | 数量 | 平均时间  | 总 AIC 时间  |
| --------- | ---- | --------- | ------------- |
| QK        | 256  | 45.395 µs | 11621.072 µs |
| PV        | 256  | 27.681 µs | 7086.220 µs  |
| 合计      | 512  | -         | 18707.292 µs |

<!-- markdownlint-enable MD060 -->

32 个 AIC 的平均工作量下界已经是 `18707.292 / 32 = 584.603 µs`。实际关键 AIC
owner 28 执行 8 个 QK 和 8 个 PV：workload 589.352 µs，task 间 Scalar 间隙
51.231 µs，连续活跃区间共 640.583 µs。其 51.231 µs 间隙完整分解为：

<!-- markdownlint-disable MD060 -->

| 关键 AIC 非 workload 项        | 时间      |
| ------------------------------ | --------- |
| `scalar.task_prepare_engine`   | 19.385 µs |
| `scalar.task_complete`         | 10.082 µs |
| payload DCCI                   | 6.139 µs  |
| BUILT/exec-state wait          | 6.026 µs  |
| payload bind                   | 3.727 µs  |
| dispatch atomic               | 3.689 µs  |
| fan-in flag wait              | 2.183 µs  |
| 合计                           | 51.231 µs |

<!-- markdownlint-enable MD060 -->

因此第一瓶颈是固定的 AIC task 服务时间，而不是 TensorMap writer、atomic 或 DCCI。
关键 AIC 活跃区间约 92% 是 workload、8% 是 Scalar 调度间隙。AIC task 数量也有
轻微不均：5 个 owner 执行 15 个、22 个执行 16 个、5 个执行 17 个；per-owner
workload 为 554.650..611.167 µs，说明通用的异构时长负载均衡仍可能有少量空间。

第二瓶颈是 Build 供给与 Execute 的平衡。B9 的 Build 中位数 616.387 µs，距离
E2E 只剩约 82.562 µs；增加到 B11 后 Build 缩短到 520.113 µs，但 E2E 没有跟着
下降，证明 B11 以后 Build 已退出关键路径，多占 builder 反而减少 AIV executor。

AIV 尾部虽然最终执行最后的 UP，但 UP 自身约 2 µs；泳道图里的大段 fan-in poll
是在等待上游 PV/Build，不是可直接删除的 atomic 计算。真正可继续优化的调度部分，
主要是关键 AIC 的约 51 µs Scalar 间隙和 15/16/17 task 的负载不均；不能再通过
缩短那约 589 µs 的模拟 workload 来达标。

## 14. 真实 Simpler 迁移：动态 Build Request 协议

### 14.1 通用动态请求 ABI

standalone 的 SIMT builder 可以随机访问预先生成的 PA task plan；真实 Simpler 的
`Submit()` 参数则由 Scalar 在运行过程中动态构造。生产迁移不能把 PA 的五类 task
或 `task_id % warp` 静态映射带入公共 runtime，因此先建立如下通用桥接：

```text
动态 Submit
  -> 唯一 Scalar publisher 保留 task-indexed request
  -> 复制 L0TaskArgs 的 tensor/create-info、scalar 和 explicit dependency
  -> 只刷新实际写入的完整 cache line
  -> Published atomic 交接
  -> SIMT builder invalidate 后获取不可变 request
```

首阶段只锁定 ABI 和发布合同，尚未把 persistent SIMT builder 接入 A5 kernel：

- 控制字单独占用 64 B，区分 `Empty / Reserved / Published`；
- payload 首 cache line 保存 task/function/engine/count 和 32 个 tensor tag；
- 每个 tensor 固定占 128 B：已有 Tensor 保存完整描述符，`OUTPUT` 只保存 64 B
  `TensorCreateInfo` 并清零其余 64 B；
- scalar 与 explicit dependency 紧随 tensor 区域，最大 payload 为 4416 B；
- 发布者只 flush 实际 payload line，消费者按控制字记录的 line 数 invalidate；
- immediate task 与 AIC/AIV kernel task共用同一请求格式；
- 协议不含 PA task kind、固定 task 数或固定 TensorMap writer 位置。

验证结果：

| 验证 | 结果 |
| ---- | ---- |
| mode 3 Host/AICPU/AICore 构建身份 | PASS |
| 单一 Scalar publisher 保留 | PASS |
| payload 打包、flush、Published 交接 | PASS |
| SIMT 侧 invalidate、header/layout 校验 | PASS |
| OUTPUT create-info 64 B 复制与后半清零 | PASS |
| immediate task 与错误输入拒绝 | PASS |
| A5 动态功能与性能 | NOT RUN（本阶段尚未接入 builder） |

上述 ABI 先作为独立协议落地；状态布局和复位在下一小节闭合后，才允许接入真实
`cce::async_invoke`。

### 14.2 独立状态与复位合同

mode 3 复用 mode 1/2 已验证的跨核执行、输出、heap 与 ordinary TensorMap 状态，
随后追加三条彼此隔离的 builder 生命周期控制和 2048 个 Build Request：

```text
CrossCoreRuntimeState
SimtBuilderLifecycleState
  builder_started
  sealed_task_count     // -1 表示动态 Submit 尚未封口
  builder_finished
SimtBuildRequestCell requests[2048]
```

这里没有新增全局 build ticket。后续每个 SIMT warp 按
`global_warp + k * total_warps` 消费固定请求流；请求尚未发布时只需同时观察本 cell
与 sealed task count。这样保留 standalone 的分布式 builder 归属，又不会在动态
Submit 路径制造一个所有 warp 都竞争的返回型 atomic。

AICPU 是唯一复位方：公共 runtime 控制复用既有 reset，三条生命周期控制分别置为
`0/-1/0`，每个 request control 置零。不可变 payload 不清零；只有对应 control 在
新一轮重新发布后才允许读取，旧字节可继续用于错误诊断。

验证结果：

| 验证 | 结果 |
| ---- | ---- |
| mode 3 状态偏移、64 B 对齐与 arena 上界 | PASS |
| 2048 个 request control 完整复位 | PASS |
| request payload 在复位时保持不变 | PASS |
| mode 1 ordinary 与 mode 2 DAG 状态回归 | PASS |
| A5 builder 启动 | NOT RUN（下一阶段） |

### 14.3 真实 L0TaskArgs 快照

动态请求发布端已直接适配生产 `L0TaskArgs`，不保存其中的 `TensorRef`、create-info
指针或显式依赖数组指针：

- `INPUT/INOUT/OUTPUT_EXISTING/NO_DEP` 复制完整 128 B Tensor；
- `OUTPUT` 复制 64 B `TensorCreateInfo`，协议层清零固定 128 B 槽的后半；
- scalar 和 `PTO2TaskId::raw` 按值复制；
- 发布前拒绝错误 args、错误 tag、空引用、shared-output 专用引用、未来依赖和非零
  ring 依赖；
- AIC/AIV 的 function id、function address 和 engine 分类继续复用现有单 lane
  `MixedKernels` 合同；Alloc 使用 immediate engine。

mode 1/2 与 mode 3 的 kernel 分类已收敛到同一公共 helper，避免两份逻辑随后漂移。
mode 3 同时具备请求保留和发布 helper，但本阶段没有打开 `submit_runtime.h` 的 mode 3
路由；在 SIMT consumer 闭合前，不能让真实 Submit 发布后无人消费。

验证结果：

| 验证 | 结果 |
| ---- | ---- |
| 真实 L0TaskArgs 已有 Tensor 128 B 值复制 | PASS |
| OUTPUT create-info 64 B 复制与后半清零 | PASS |
| scalar 与两个显式依赖原值复制 | PASS |
| 未来依赖与跨 ring 依赖拒绝 | PASS |
| mode 1/2 状态及 Python 构建选择回归 | PASS |
| mode 3 A5 动态 Submit | NOT RUN（consumer 尚未接通） |
