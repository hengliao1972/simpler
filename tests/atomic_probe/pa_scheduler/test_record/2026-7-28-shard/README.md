# 2026-07-28 shared TensorMap 分组与 per-task 插入完成链

## 记录范围

本目录先归档 PA Scheduler CCEC `swimlane` 用例在以下参数矩阵中的完整泳道图：

- `shared_insert_turn_groups`（下文记为 `G`）：`1 / 8 / 32 / 64 / 128`
- `batches`（目录名中记为 `B`）：`256 / 512`
- 每个 batch 固定生成 5 个任务，因此 `B256` 为 1280 tasks，`B512` 为 2560 tasks

上述矩阵共 10 组，均为基于 generation 10、shared TensorMap、ring
capacity 128 重新构建和实机运行的历史结果。目录随后增加一组
generation 11 的新协议结果：Claim 恢复原 Cube/Alloc 四分片与 shared
Vector 八分片 cursor；TensorMap 插入完成链改为每 task 独立的
`TaskCell::deps_prepared`。task 0 无前驱，task N 只轮询
`task[N-1]`，发布完整 writer 元数据后再以 CAS 发布 `task[N]=N`。
旧 sidecar turn 仅作为历史 ABI 保留，运行期间保持初值。

在同一 generation 11 协议上又归档一组 Register 细分结果。该版本为
每个成功 winner 增加一条
`SharedRegisterPublishTaskOutputs` raw detail，并与原有
`SharedRegisterPublishMetadata` 组成 `Register -> metadata -> outputs`
两级父子区间；loser 不产生这两条 detail。离线加工仅复用三个区间的
端点，恢复前序等待、writer metadata、`PublishSharedTaskOutputs`、
metadata 收尾和插入完成发布五段，不逐次记录 atomic poll。

随后再归档一组 `PublishSharedTaskOutputs` 内层拆分结果。该版本把
outputs 内的 descriptor 发布拆成相邻两层 raw detail：
`SharedRegisterPublishTaskOutputsCopy`（整批 `TensorDesc` copy）与
`SharedRegisterPublishTaskOutputsFlush`（整批 `FlushRegion`）；二者
严格嵌在 `SharedRegisterPublishTaskOutputs` 内且 `copy.end == flush.start`。
每个成功 winner 固定 4 条 Register detail
（metadata + outputs + copy + flush），loser 仍不产生这些 detail。
exclusive 分析在 task-outputs 包络下再闭合
`copy + flush + residual`（预检 / last_writer / barrier / published）。

之后归档 `PublishSharedTaskOutputs` 移出 task-ID 串行插入区后的结果。
fresh output cell 由唯一 winner 在 Materialize 尾部独占发布，Register
只保留前驱 `deps_prepared` 等待、ordinary/symbol writer metadata 和
本 task completion 发布。对应 raw detail 更名为
`SharedMaterializePublishTaskOutputs`、`Copy` 和 `Flush`，并严格嵌套在
Materialize；Register 中三类 task-output detail 的计数必须为零。

本次新增 R5i/R5j 串行区消减后的最终 B256 泳道：在等待前驱前预计算
ordinary bucket、同 bucket 序号和 symbol key；Register 内不再重复
Inspect/Validate/producer 扫描，不构造 `ignored_fanin`，并将已经由
per-task completion 链证明就绪的 published 等待收敛为一次精确检查。
泳道结构不增加新区域，继续使用上一版 Materialize/Register 边界。

历史矩阵中的 `G` 在 generation 10 代码和 artifact manifest 中表示
`shared_insert_turn_groups`，不要误读成 batch 数或 worker 数。
generation 11 新协议不再用 G 控制热路径；manifest 中保留的默认 G1
只是历史 ABI 身份字段，不表示仍用单线 baton。

> **Git 归档边界：** 本次提交只保留 README、`SHA256SUMS` 和构建
> manifest。约 1.8 GiB 的泳道 JSON 与运行日志仍保留在本机同名目录，
> 分别受仓库 `test_record/**/*.json` 与 `*.log` 忽略规则约束，不使用
> `git add -f` 强行提交。fresh clone 可审阅结论和构建身份，但不能仅凭
> 本提交重放下文列出的 raw/merged 泳道。

## 固定测试口径

- CCEC，device 0，96 workers（32 AIC + 64 AIV）
- shared TensorMap；两代均使用原 Claim cursor。generation 11 仅将
  TensorMap 插入完成等待/发布迁到 per-task `deps_prepared`
- context length：8192
- winner real-compute counts：`6,28,4,1`
- final barrier：`two-16`
- swimlane：开启
- atomic trace：开启
- 每个组合运行 1 次

泳道图与 atomic trace 会明显放大绝对耗时。本目录用于观察流程结构、并发关系和
exclusive span 完整性；性能基线仍应使用关闭泳道图/atomic trace 的 `perf-clock`
测试结果，不能用本目录中的 Submit 时间替换性能基线。

## 结果总览

| G 值 | B 值 | tasks | raw records | dropped | Submit (us) | execution | semantic | postprocess |
| ---- | ---- | ----- | ----------- | ------- | ----------- | --------- | -------- | ----------- |
| 1 | 256 | 1280 | 488432 | 0 | 58182.263 | PASS | PASS | PASS |
| 1 | 512 | 2560 | 979273 | 0 | 122146.096 | PASS | PASS | PASS |
| 8 | 256 | 1280 | 474626 | 0 | 25089.932 | PASS | PASS | PASS |
| 8 | 512 | 2560 | 948414 | 0 | 50556.887 | PASS | PASS | PASS |
| 32 | 256 | 1280 | 479052 | 0 | 13256.569 | PASS | PASS | PASS |
| 32 | 512 | 2560 | 957198 | 0 | 26830.159 | PASS | PASS | PASS |
| 64 | 256 | 1280 | 480879 | 0 | 11117.321 | PASS | PASS | PASS |
| 64 | 512 | 2560 | 961093 | 0 | 22130.270 | PASS | PASS | PASS |
| 128 | 256 | 1280 | 480356 | 0 | 10193.173 | PASS | PASS | PASS |
| 128 | 512 | 2560 | 960282 | 0 | 20288.731 | PASS | PASS | PASS |
| per-task `deps_prepared` | 256 | 1280 | 481198 | 0 | 9091.529 | PASS | PASS | PASS |
| per-task + Register detail | 256 | 1280 | 482387 | 0 | 9405.962 | PASS | PASS | PASS |
| per-task + outputs copy/flush | 256 | 1280 | 485028 | 0 | 9466.451 | PASS | PASS | PASS |
| per-task + outputs in Materialize | 256 | 1280 | 486416 | 0 | 3464.587 | PASS | PASS | PASS |
| per-task + serial reduction | 256 | 1280 | 485925 | 0 | 3392.893 | PASS | PASS | PASS |

这里的 `dropped=0` 表示 trace buffer 没有丢记录；每组
`swimlane_exclusive_analysis.json` 的 validation 也均为 `PASS`。
新协议 B256 恰好生成 1,279 条前驱 PollBatch 和 1,280 条完成 CAS，
分别对应 task 1…1279 的一次等待 episode 与每个 task 的一次完成发布；
task 0 不伪造前驱原子访问。

新协议另以三个独立进程运行 trace-free `perf-clock`，Submit 分别为
8,406.504 / 8,326.011 / 8,376.102 us，中位数 8,376.102 us。与迁移前
同设备、同业务参数的历史 G128 中位数 9,371.635 us 相比，减少
995.533 us（10.6228%）。该对照用于判断候选是否值得保留；泳道构建和
trace-free 构建的绝对值仍不能互相相减。

Register 细分图的 analysis schema 为 3、raw trace schema 为 5，
1,280 个 Register 均恰好关联一条 metadata 和一条 task-output detail。
两级闭合与五段扁平闭合全部精确通过。按 96 核累计 core-work：

| Register 区域 | cycles | 在父区间中的比例 |
| ------------- | ------ | ---------------- |
| 前序插入完成等待 | 578,829,069 | Register 的 98.460% |
| metadata 总区间 | 8,365,557 | Register 的 1.423% |
| 插入完成发布 | 690,424 | Register 的 0.117% |
| writer metadata | 2,311,441 | metadata 的 27.630% |
| `PublishSharedTaskOutputs` | 5,963,620 | metadata 的 71.288% |
| metadata 收尾 | 90,496 | metadata 的 1.082% |

这里的比例是所有核区间时长之和，不是端到端 wall-clock。新增版本比
上一张泳道多一次 output 边界取时和一条 raw，因此两张泳道的 Submit
绝对值不能用来推导业务性能变化；本图只用于定位 Register 内部构成。

outputs copy/flush 拆分图同样为 analysis schema 3、raw trace schema 5；
1,280 个 Register 均恰好关联 metadata、outputs、copy、flush 各一条。
Register 两级闭合、metadata 内三段闭合与 task-outputs 内
`copy + flush + residual` 闭合全部 exact。按 96 核累计 core-work：

| outputs 内区域 | cycles | 在 task_outputs 中的比例 |
| -------------- | ------ | ------------------------ |
| task_outputs 总包络 | 6,067,592 | 100% |
| copy（`TensorDesc` 批拷贝） | 2,094,172 | 34.514% |
| flush（`FlushRegion`） | 588,668 | 9.702% |
| residual（预检 / last_writer / barrier / published） | 3,384,752 | 55.784% |

该版本相对 Register detail 泳道再多两条 raw 与两次内部取时，Submit
绝对值同样不能与业务 perf-clock 或上一张泳道直接相减；只用于拆开
`PublishSharedTaskOutputs` 内 copy 与 flush 的相对成本。

outputs 移入 Materialize 后，analysis schema 仍为 3、raw trace schema
仍为 5。1,280 个 winner 各有一组 Materialize outputs/copy/flush，
Register 只保留 1,280 条 metadata detail。两级 Materialize、
task-outputs、Register 和 metadata 闭合全部 exact。按 96 核累计
core-work：

| Materialize 区域 | cycles | 在父区间中的比例 |
| ---------------- | ------ | ---------------- |
| Materialize 总区间 | 12,743,376 | 100% |
| output 发布前 | 6,591,184 | 51.722% |
| `PublishSharedTaskOutputs` | 6,026,512 | 47.291% |
| output 发布后 | 125,680 | 0.986% |

| outputs 内区域 | cycles | 在 task_outputs 中的比例 |
| -------------- | ------ | ------------------------ |
| copy | 2,071,279 | 34.369% |
| flush | 603,020 | 10.006% |
| residual | 3,352,213 | 55.624% |

| Register 区域 | cycles | 在父区间中的比例 |
| ------------- | ------ | ---------------- |
| Register 总区间 | 15,678,761 | 100% |
| 前序插入完成等待 | 12,665,626 | 80.782% |
| writer metadata | 2,409,396 | 15.367% |
| 插入完成发布 | 603,739 | 3.851% |
| Register 内 task outputs | 0 | 0% |

新的 output 总包络 `6,026,512 cycles` 与移出前的
`6,067,592 cycles` 同量级，说明工作没有被删掉，而是改变了所在层级和
可重叠关系。泳道 Submit 含 level 4/atomic trace 开销；独立的
trace-free `perf-clock` 交错 A/B 各有四个正式样本：移出前中位数
8.3559295 ms，移出后中位数 2.9402265 ms，减少 5.4157030 ms
（64.8127%，前后比 2.8419×）。after 四次为
2.906899/2.941103/2.939350/2.951372 ms，全部 execution、semantic 和
postprocess PASS。原始日志位于
`outputs/output_publish_move_ab_20260728_104448/`。

串行区消减后的 analysis schema 为 3、raw trace schema 为 5；1,280
个 Register 均恰好关联一条 metadata，Materialize 中仍各有一组
outputs/copy/flush，所有父子分区和任务身份检查均精确闭合。按 96 核
累计 core-work：

| Register 区域 | cycles | 在父区间中的比例 |
| ------------- | ------ | ---------------- |
| Register 总区间 | 8,365,867 | 100% |
| 前序插入完成等待 | 6,169,187 | 73.742% |
| writer metadata | 1,537,416 | 18.377% |
| 插入完成发布 | 659,264 | 7.880% |

与紧邻的 outputs-in-Materialize 泳道相比，Submit 从 3,464.587 us 降到
3,392.893 us（-2.069%），Register 累计 core-work 从 15,678,761
cycles 降到 8,365,867 cycles（-46.642%）；其中 predecessor wait
下降 51.292%，writer metadata 下降 36.191%。这两张图使用相同的
raw 区域集合，因此可用于确认消减落点；绝对净性能仍以关闭泳道与
atomic trace 的 perf-clock 为准。最终源码的单次无锁 perf-clock B256
为 2,482.874 us，全部运行断言 PASS；它只有一个样本，不把相对历史
四样本中位数的差值声明成稳定收益。

## 目录与文件语义

每个 generation 10 的 `turn_g<G>_b<B>/` 目录包含四个统一重命名后的文件：

- `turn_g<G>_b<B>_merged_swimlane.json`：可直接载入
  [Perfetto UI](https://ui.perfetto.dev/) 的完整泳道图。
- `turn_g<G>_b<B>_l2_swimlane_records.json`：设备侧原始 L2 trace 记录，是泳道图加工
  和 exclusive 分析的输入。
- `turn_g<G>_b<B>_swimlane_exclusive_analysis.json`：排他分区分析、记录覆盖和
  dropped-record 校验结果。
- `turn_g<G>_b<B>_run.log`：该组合的运行参数、断言、计数器、Submit 时间及
  后处理状态。

`manifests/turn_g<G>_artifacts.manifest` 保存每个 G 对应的 host/kernel 哈希与
构建参数。同一 G 的 B256、B512 使用同一套构建产物。

generation 11 结果位于 `per_task_deps_prepared_b256/`，文件使用相同前缀：

- `per_task_deps_prepared_b256_merged_swimlane.json`
- `per_task_deps_prepared_b256_l2_swimlane_records.json`
- `per_task_deps_prepared_b256_swimlane_exclusive_analysis.json`
- `per_task_deps_prepared_b256_run.log`
- `per_task_deps_prepared_b256_perf_clock_run1.log` 至 `run3.log`

泳道与 trace-free 性能构建身份分别保存在
`manifests/per_task_deps_prepared_swimlane_artifacts.manifest` 和
`manifests/per_task_deps_prepared_perf_clock_artifacts.manifest`。

Register 细分结果位于
`per_task_deps_prepared_register_detail_b256/`，包含 raw、merged、
exclusive analysis 和运行日志；对应构建身份保存在
`manifests/per_task_deps_prepared_register_detail_swimlane_artifacts.manifest`。

outputs copy/flush 拆分结果位于
`per_task_deps_prepared_task_outputs_copy_flush_b256/`，文件前缀相同：

- `per_task_deps_prepared_task_outputs_copy_flush_b256_merged_swimlane.json`
- `per_task_deps_prepared_task_outputs_copy_flush_b256_l2_swimlane_records.json`
- `per_task_deps_prepared_task_outputs_copy_flush_b256_swimlane_exclusive_analysis.json`
- `per_task_deps_prepared_task_outputs_copy_flush_b256_run.log`

对应构建身份保存在
`manifests/per_task_deps_prepared_task_outputs_copy_flush_swimlane_artifacts.manifest`。

outputs 移入 Materialize 的结果位于
`per_task_deps_prepared_materialize_task_outputs_b256/`：

- `per_task_deps_prepared_materialize_task_outputs_b256_merged_swimlane.json`
- `per_task_deps_prepared_materialize_task_outputs_b256_l2_swimlane_records.json`
- `per_task_deps_prepared_materialize_task_outputs_b256_swimlane_exclusive_analysis.json`

该次运行的终端输出没有单独落盘，因此不伪造 `run.log`；运行参数、trace
元数据和校验结论分别由 raw capture 与 exclusive analysis 保存。该组只
归档三份泳道结果，不更新历史 `SHA256SUMS`。

串行区消减后的最终结果位于
`per_task_deps_prepared_serial_reduction_b256/`：

- `per_task_deps_prepared_serial_reduction_b256_merged_swimlane.json`
- `per_task_deps_prepared_serial_reduction_b256_l2_swimlane_records.json`
- `per_task_deps_prepared_serial_reduction_b256_swimlane_exclusive_analysis.json`

本次终端输出同样没有单独落盘，因此只归档上述三份真实产物，不伪造
`run.log`，也不修改此前历史矩阵的 `SHA256SUMS`。

## 原始输出来源

| 归档目录 | 原始输出目录 |
| -------- | ------------ |
| `turn_g1_b256` | `outputs/pa_scheduler_shared_swimlane_20260728_050035_1380442/ccec` |
| `turn_g1_b512` | `outputs/pa_scheduler_shared_swimlane_20260728_050126_1381547/ccec` |
| `turn_g8_b256` | `outputs/pa_scheduler_shared_swimlane_20260728_050428_1387281/ccec` |
| `turn_g8_b512` | `outputs/pa_scheduler_shared_swimlane_20260728_050517_1388650/ccec` |
| `turn_g32_b256` | `outputs/pa_scheduler_shared_swimlane_20260728_050812_1393228/ccec` |
| `turn_g32_b512` | `outputs/pa_scheduler_shared_swimlane_20260728_050901_1394397/ccec` |
| `turn_g64_b256` | `outputs/pa_scheduler_shared_swimlane_20260728_051158_1398883/ccec` |
| `turn_g64_b512` | `outputs/pa_scheduler_shared_swimlane_20260728_051248_1400250/ccec` |
| `turn_g128_b256` | `outputs/pa_scheduler_shared_swimlane_20260728_051814_1410271/ccec` |
| `turn_g128_b512` | `outputs/pa_scheduler_shared_swimlane_20260728_051907_1412861/ccec` |
| `per_task_deps_prepared_b256` | `outputs/pa_scheduler_shared_swimlane_20260728_064905_1478156/ccec` |
| `per_task_deps_prepared_register_detail_b256` | `outputs/pa_scheduler_shared_swimlane_20260728_074105_1519701/ccec` |
| `per_task_deps_prepared_task_outputs_copy_flush_b256` | `outputs/pa_scheduler_shared_swimlane_20260728_085342_1556695/ccec` |
| `per_task_deps_prepared_materialize_task_outputs_b256` | `outputs/pa_scheduler_shared_swimlane_20260728_105926_1598639/ccec` |
| `per_task_deps_prepared_serial_reduction_b256` | `outputs/pa_scheduler_shared_swimlane_20260728_131823_1696692/ccec` |

## 环境边界与异常记录

运行环境中没有 `npu-smi` 和 `task-submit`，因此本次是在 device 0 上直接串行运行，
未获得设备锁；这里不据此声明芯片身份，也不声明测试期间不存在环境级干扰。

G128 首次完整构建时 CCEC frontend 以 exit 135 异常退出，未启动设备任务。使用
正式构建参数（包括 1664 B block-local reserve）复现编译成功，随后完整构建重试和
B256/B512 两组运行均通过。现有证据支持将首次失败记录为一次 frontend 瞬态异常，
不支持把它判断为稳定的 G128 编译边界。
