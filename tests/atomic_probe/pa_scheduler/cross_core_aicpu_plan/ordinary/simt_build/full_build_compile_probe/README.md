# Scalar Build 全链路复用的 SIMT 编译探针

## 结论

现有 ordinary/scalar_build 的完整
`BuildRuntimePlanTask<SimtOps, false>` **不能只靠重定义
`PA_DEVICE`/`PA_DEVICE_NOINLINE` 就直接复用为 SIMT Build**。当前 CCEC
能够生成包含完整调用图的优化 LLVM bitcode，但不能把它继续生成可链接的
VF machine object。因此本探针是负向结论，不能把 mixed 壳的成功误报成
“完整 Build 已可运行”。

本轮定位到三个彼此独立的硬阻塞：

1. 只改两个公共 device 宏时，前端在模板定义阶段就拒绝
   `BuildCallbackSubmitArgs`。该函数内的 callback lambda 使用硬编码
   `PA_CALLBACK_LAMBDA_DEVICE __aicore__`，仍是 Scalar-only，却调用已经变成
   `simt_callee` 的 `MakeCallbackQueryView`、`InitCreateInfo` 和
   `MakeCallbackOutputView`。首个错误是
   `simt_callee function can only be called by simt_vf/simt_callee function`。
2. 仅在探针 include 窗口把公共头中的全部显式 `__aicore__` 连同 lambda
   提升为 `simt_callee` 后，正式 `TaskArgs` writer-delta 扫描仍无法生成
   machine object。`TensorPointer` union 同时保存 owner-local
   `TensorDesc *` 与 GM `__gm__ TensorDesc *`，而 `TensorRefKind` 在运行期
   决定读取哪一地址空间；后端报
   `ERROR: error pointer address space cast`。把同一隔离用例固定成 GM 分支后，
   地址空间错误消失，但后端继续报
   `Copy one register into another with a different width`，说明它不只是
   inline/noinline 或一次偶然的优化器折叠问题。
3. 即使绕过前述 writer-delta 形状，正式 exec publication 还有第二个独立
   的异地址空间容器：`PaExecPayloadSource::tensors[]` 的
   `PaExecTensorAddress` union。运行期在 local/GM descriptor 间选择会复现
   同一个 `error pointer address space cast`；固定为 GM reference 后可成功
   生成 object，证明 GM descriptor 引用转换为 wire address 本身不是问题。

把历史 `PA_DEVICE_NOINLINE` 改成真正的
`static __simt_callee__ __aicore__ __attribute__((noinline))` 也不能绕过完整
Build 的后端错误，仍首先停在 `error pointer address space cast`。

## 已经通过的边界

- 探针专用的全头 attribute overlay 可以让完整 Build 调用图进入 `-O3`
  optimized LLVM bitcode。bitcode 中保留 static VF、返回型 ADD/CAS/MAX
  atomic、DCCI、fence、uncached payload store，以及 mixed entry 的
  async/wait。
- Runtime Plan acquire、payload DCCI、canonical validation 与 PA adapter
  decode 可单独生成 VF object；问题不是一进入 Plan decoder 就发生。
- `PublishSharedTaskWriterMetadata<SimtOps>` 使用真实
  `ordinary_count=1`、运行期 bucket/entry 并生成 preflight、slot payload、
  DCCI、seq/tail atomic publication 的完整链路，可以生成 VF object。因此
  ordinary TensorMap metadata publication 本身不是当前阻塞点。
- 同 TU 的 static `__simt_vf__`、mixed AIV `async_invoke + wait_flag` 壳可
  编译和链接；最终 ELF 有一个 LOCAL VF entry、一个 GLOBAL kernel entry，
  metadata 编码 `SIMD_SIMT_MIX_VF=4`。这排除了最外层 mixed 工具链壳失败。

## 探针边界

本目录不创建第二份 TaskPlan ABI，不用 `task_id % 5` 或固定 PA 公式重建
任务，也没有删掉正式 full-Build 的 Materialize、Register、Fanin 或 exec
publication 调用。为了越过第一个 lambda 属性错误，`kernel.cpp` 只在公共头
include 窗口覆盖预定义 `__aicore__`；这会把整个头的显式 Scalar 标注一起
提升，**只是继续定位后端的探针手段，不是生产修复方案**。

探针只做 `dav-c310-vec` 编译和静态链接，没有 Host、a5sim 或 A5 上板验证。
`SimtOps` 中的 DCCI/fence 映射因此也不能作为已经验证的跨核内存模型合同。

## 复现

```bash
./build.sh
```

脚本始终使用 `-Werror`，并同时验证：

- 直接双宏替换稳定暴露 callback lambda 属性错误；
- probe-only 全头 overlay 后完整 Build 能到 optimized bitcode、但 machine
  code 稳定暴露地址空间错误；
- 真正的 `simt_callee noinline` 不能消除该错误；
- 两个动态 local/GM 地址空间隔离用例及 GM-only 对照；
- ordinary_count=1 writer metadata 正向门槛；
- VF 为 LOCAL、mixed entry 为 GLOBAL、metadata 为 MIX_VF=4。

脚本成功结束表示上述“正向门槛通过、负向阻塞精确复现”，不表示完整 Build
已经编译成功。

## 窄 canonical Plan builder 正向门槛

`simt_plan_task_builder.h` 与 `narrow_builder_kernel.cpp` 不再把完整 Scalar
头整体提升为 `simt_callee`，也不 Decode 到带 local/GM pointer union 的
`TaskArgs`。四个 warp leader 直接消费 immutable canonical Plan v2 wire，
按值保存 descriptor/create-info/scalar/dependency，并真实实例化下列完整链：

1. Plan control acquire、payload 逐行 DCCI、control 二次确认和 canonical
   payload validation；
2. fresh Output heap reservation、128B descriptor 原位构造与
   `published/last_writer` 发布；
3. `ordinary_count>0` region preflight/append、symbol history/last-writer
   CAS，以及严格 `task[N-1] -> metadata -> completion[N]`；
4. ordinary/symbol/explicit fanin 且所有 producer 严格小于 consumer；
5. metadata-only completion，或完整 immutable exec payload 与
   `SharedExecCell::BUILT` CAS 发布。

窄 VF 的 payload 发布固定使用同一内存模型：writer 对 output descriptor、
writer history、ordinary region、metadata vend 和 exec payload 都执行
`asc_stcg` bypass store，随后 `asc_threadfence`，最后才以 CAS/Exchange
发布对应 control。writer 不执行 DCCI；`asc_dcci_single + threadfence` 只
存在于 reader acquire/invalidate。尤其 history header 与 record 按 canonical
64-bit word 写入，禁止 ordinary GM store 后再用 reader DCCI 伪装 clean-out。
heap cursor/vend 的返回型 FetchAdd 若在合法并发下取得越界旧值，同样进入
terminal fatal；已经推进的控制字不得局部 Exchange 回滚，下一轮只能由 Host
按现有启动合同重置整份 sidecar/heap control。

fresh Output 另有一条只覆盖 **insert completion 交棒前** 的最小回滚合同。
窄 builder 在 scratch 中分别记录已成功 `last_writer: -1 -> task` 的连续
reservation 前缀和已成功 `published: -1 -> task` 的连续 publication 前缀；
后续任何失败都先按逆序尽力执行 `published: task -> -1`，再按逆序执行
`last_writer: task -> -1`。CAS 只撤本 owner 的值，不覆盖冲突者；任一撤回失败
仍保持 terminal fatal，不能假装本轮可继续。descriptor payload、heap cursor
和 aggregate vend 均不做局部回滚，因为它们没有能在其他并发 reservation
面前安全撤销的所有权证明。CPU 门槛固定覆盖第二个 Output publication 冲突
和第二个 descriptor 地址失败，并验证本 owner 的已发布/已预留控制前缀恢复
为 `-1`、insert completion/build release 均未发生，而 descriptor 失败后的
heap advance 明确保留给整轮 fatal reset。

SharedOutputRef 的 tag 合同也在 canonical acquire、任何副作用之前闭合：只
允许 `Input/Inout/OutputExisting`，`NoDependency + reference` 与 fresh
`Output + reference` 都 fail-closed。fanin 则由 Runtime 提供统一的
`FaninLowerBound(N)`，窄 header 的唯一插入入口把 symbol latest-writer、
descriptor owner、ordinary lookup 和 explicit dependency 全部裁到同一个
半开窗口 `[N-H,N)`；窗口外旧 producer 视作 external，不进入 immutable
Exec payload，self/future producer 仍是协议错误。production Runtime 必须从
真实 `SchedulerState::heap_window` 计算该下界，不能让 PA adapter 私下使用
另一套过滤规则。

这里的 `NarrowRuntime`、ordinary ring 和 metadata cell 只是强制所有模板
分支进入 machine code 的 compile harness，**不是第二份 production ABI**。
production 固定采用双 TU 身份隔离：独立 AIV TU 先以 `simt_callee` 身份
include canonical Plan/shared-exec 协议并启动/join VF；join 后调用另一 Scalar
TU 导出的 scheduler continuation。现有 Scalar scheduler TU 不能再次 include
窄 SIMT 头，否则 include guard 会把同一 helper 的调用身份锁错。正向脚本
分别编译这两个 TU，再静态链接并验证 continuation 已解析、无 undefined。
正式接线必须让 Runtime accessor 直接返回现有 `SharedTensorMapSidecar`、
`SharedRegionValue`、`SharedWriterHistoryCell`、per-task insert-completion 和
`SharedExecCell` 的真实地址；不得在 production GM 中分配 mirror 状态。接线
TU 还必须交叉 `static_assert`：

- `SimtWriterRegion` 与 `SharedRegionValue` 的 size/alignment，以及
  `buffer_addr/lo/hi/producer/reserved` 全部 offset；
- `SimtWriterHistoryRecord` 与真实 record 的 size/alignment/两个字段 offset；
- `SimtWriterHistoryCell` 与真实 cell 的 320B/64B 对齐，以及
  `magic/writer_task/count/reserved/entries` 全部 offset。

这些断言缺少任何一项，都不能用 `reinterpret_cast` 把窄 builder 接到真实
shared 状态。1D view 的 wire 目前只有结构合同，没有可求证的 offset 单位
合同，因此窄 builder 在任何 heap/metadata side effect 前明确 fail-closed；
不能把它静默降级为 plain reference。

同理，compile harness 的 `NarrowMetadataCell` 只用于强制生成
`stcg(vend) -> fence -> completion CAS` 机器码，并不证明重复 metadata Build
时 vend 不会被覆盖。production Runtime 必须直接映射真实 `TaskCell`，对 vend
使用现有原子 Exchange、校验 `prior_vend == 0`，再 CAS flag；CPU 动态门槛已
按这一真实顺序实现。唯一 Build ticket 是正常运行前提，harness 不能被当成
第二套 metadata ABI 或 duplicate-owner 正确性证明。

正向门槛复现：

```bash
./narrow_builder_build.sh
```

该脚本在 `dav-c310-vec -O3 -Wall -Werror` 下强制实例化完整
`BuildCanonicalPlanTask<SimtOps>`，检查 optimized IR 中的 SIMT entry、返回型
ADD/CAS、DCCI、fence、uncached store 和 V/S join，然后静态链接 machine
object，并要求：

- 完整 VF 实体是唯一非空 LOCAL `_simt_entry`；
- mixed wrapper 是唯一非空 GLOBAL function；
- 没有 undefined GLOBAL；
- source/header 不含 `task_id % 5`、`FullPaTaskPlan`、`SimtPrepareTask` 或
  Scalar Plan decoder。
- source gate 逐函数检查 writer `FlushRegion` 只有 fence、reader
  `InvalidateRegion` 保留逐行 DCCI，并拒绝 history 的 ordinary GM 赋值。
- source gate 同时锁定 output 逆序 CAS rollback、统一 fanin lower-bound 和
  SharedOutputRef tag 白名单；对应动态门槛由 `../cpu/build_task_builder.sh`
  在普通与 ASan/UBSan 两种构建中执行。

这只闭合 CCEC machine-object/link gate；A5 可见性、真实 shared ABI 接线和
动态正确性仍需后续单独验证。
