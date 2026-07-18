# A5 PA Scheduler I-cache Miss 采集与分析指南

## 1. 目标与最终构建口径

本指南面向
`tests/atomic_probe/pa_scheduler`
的 standalone CCEC 分支，目标是回答：在完整 Submit 期间，32 个 AIC
和 64 个 AIV 每核发生了多少 I-cache request/miss，其中哪些 miss
值得继续优化。它不依赖 simpler 生产代码，也不将 standalone 结果
冒充为真实 PA 的绝对 profile。

当前最终只保留两类正式观察构建：

| 构建 | 内容 | 是否包含 PMU |
| --- | --- | --- |
| `swimlane` | 普通阶段泳道 + 逐 atomic 泳道，在同一 AIC/AIV scalar lane 合并采集 | 否 |
| `submit-pmu` | 每物理子核的完整 Submit PMU，并可编译一个局部 phase | 是，仅 CCEC |

`run` 、`smoke` 和 phase 名是运行或编译选择，不是额外的第三类
构建。

## 2. 为什么 I-cache miss 必须独立重编译

I-cache 数据对代码布局极其敏感。泳道和逐 atomic 观察会增加：

- 阶段和 atomic 的 `SYS_CNT` 读取；
- atomic wrapper、ClockBaseline 与 record 发布分支；
- 更大的 scalar `.text` 和不同的函数/对齐布局；
- 由此引起的 worker 到达、轮询和跨核竞争时序变化。

因此，“代码仍在 ELF 中，只在运行时关闭 record”不足以得到干净的
I-cache 观察。`submit-pmu` 会独立重编译，编译掉泳道 record、逐
atomic wrapper、ClockBaseline、runtime phase-profile 和旧 cold/warm 冲刷体。
`swimlane` 则不构建 PMU owner。

两种数据不在同一进程采集：

- 看事件时序和 atomic bracket，使用 `swimlane`；
- 看完整 Submit 每核 I-cache request/miss，使用 `submit-pmu`。

两份证据可以按同一源码版本交叉理解，不能做逐 tick 对齐，也不能
把 PMU 的平均 miss 回填为某一条 atomic span 的属性。

## 3. `none` 与 `claim` 如何选择

当前只有两个编译期 phase：

| phase | 边界 | 优先用途 |
| --- | --- | --- |
| `none` | 完整 Submit 中不读局部 shadow counter | 回答完整 Submit 的 AIC/AIV 每核 request/miss；这是默认选择 |
| `claim` | 每次 `Claim()` 调用前后读局部 shadow counter | 当 `none` 已证明 miss 值得追踪时，试验 Claim 的 running read-clear 下界/上界归因链路 |

`claim` 的 begin/end 读取本身会执行 scalar 指令、占用取指并改变多核
时序。因此：

- `claim` 的 phase request/miss 是带局部边界扰动的观察值；
- `claim` 的 raw 观察值是下界，上界为该核下界加 primary-shadow loss；
- `none` 和 `claim` 是不同 ELF、不同进程，不能以两者相减声称得到
  了零扰动 Claim 净值；
- 未来不同 phase ELF 的局部 request/miss 不可相加成“完整 Submit”；
  完整 Submit 始终以每个 ELF 自己的 primary whole 为准。

该区间只描述同一插桩 ELF、当前边界定义下的局部事件，不是无插桩 Claim
的真实区间。完整 Submit `none` 没有运行中 read-clear，仍执行 96/96
逐核严格闭合。

## 4. 环境、构建与产物

命令在 `tests/atomic_probe/pa_scheduler` 目录下执行。非交互 shell
建议显式 source CANN 并选择本用户 GCC 15：

```bash
source /home/q00473782/Ascend/cann-9.1.0-weekly-20260708/cann-9.1.0/set_env.sh
export GCC15_ROOT=/home/q00473782/.local/gcc-15/root
export PATH="$GCC15_ROOT/usr/bin:$PATH"
export LD_LIBRARY_PATH="$GCC15_ROOT/usr/lib/x86_64-linux-gnu:$GCC15_ROOT/usr/lib/gcc/x86_64-linux-gnu/15${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CXX="$GCC15_ROOT/usr/bin/g++-15"
```

构建完整 Submit 基准：

```bash
./run.sh build-submit-pmu ccec none
```

需要 Claim 局部归因时另行构建：

```bash
./run.sh build-submit-pmu ccec claim
```

产物分别位于：

```text
pa_scheduler/build/ccec/submit-pmu/none/
pa_scheduler/build/ccec/submit-pmu/claim/
```

每个 phase 目录中的 `pa_scheduler_host`、`pa_scheduler_kernel.o`、
`libpa_scheduler_pmu_owner_aicpu.so` 和
`libpa_scheduler_pmu_owner_dispatcher.so` 是一个不可拆分的构建集。host
会按 kernel 所在目录加载两个 SO；不得从另一个 phase 目录复制或
拼接产物。构建只在四件套全部完成后原子发布
`submit_pmu_artifacts.manifest`；`run.sh` 在启动 host 前核对 schema、phase、
固定文件列表和四个 SHA256。

`swimlane` 的 CCEC 产物仍在 `pa_scheduler/build/ccec/`，不是 PMU
产物。

## 5. 采集完整 Submit 和 Claim

### 5.1 完整 Submit `none`

```bash
OUT="./outputs/submit_pmu_none_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"

./run.sh submit-pmu ccec none \
  --device 0 --batches 256 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1 \
  --pmu-json "$OUT/run1.json"
```

多轮比较必须使用多个独立进程和唯一文件名：

```bash
./run.sh submit-pmu ccec none --device 0 --batches 256 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1 \
  --pmu-json "$OUT/run2.json"

./run.sh submit-pmu ccec none --device 0 --batches 256 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1 \
  --pmu-json "$OUT/run3.json"
```

### 5.2 Claim 局部归因

```bash
OUT_CLAIM="./outputs/submit_pmu_claim_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_CLAIM"

./run.sh submit-pmu ccec claim \
  --device 0 --batches 256 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1 \
  --pmu-json "$OUT_CLAIM/run1.json"
```

`submit-pmu` action 已固定：

```text
--runs 1 --no-swimlane --pmu-window submit-all
```

调用者不要重复传入这三项，也不能添加
`--profile-phases`、`--trace-atomics`、`--analyze-swimlane` 或
`--swimlane-json`。`--pmu-json` 可选；但要做 raw 复算和多轮汇总时必须
使用它。host 拒绝覆盖已有 JSON 或同名 `.tmp`。

## 6. Primary/shadow 计数和可信门禁

### 6.1 计数器分工

| 计数 | selector | 用途 |
| --- | --- | --- |
| CNT6 | `0x34` | 完整 Submit primary I-cache request；局部边界从不读它 |
| CNT7 | `0x35` | 完整 Submit primary I-cache miss；局部边界从不读它 |
| CNT8 | `0x34` | read-to-clear shadow request |
| CNT5 | `0x35` | read-to-clear shadow miss；诊断 ELF 因此不再提供 MTE3 busy |
| CNT9 | `0x0` | 未使用 |

A5 b1 实测已证明将 `0x35` 配置到 CNT9 时计数始终为 0，所以
CNT9 不能作 shadow miss。这个变化只影响 `submit-pmu` 诊断构建，
不影响 `swimlane` 构建。

shadow PMU counter 是 read-to-clear。`claim` 在阶段 begin 读取 CNT8/CNT5，将
之前的片段加入 shadow whole；在 end 再读一次，同时加入 shadow
whole 和 Claim phase；完整 Submit stop 后读 tail。`none` 不做中途读取，
只在 stop 后取 tail。

`none` 对每个物理子核必须精确满足：

```text
shadow_whole_icache_requests == icache_requests
shadow_whole_icache_misses   == icache_misses
```

即 stop 后读取的 CNT8/CNT5 分别等于同 selector、同 gate 的 CNT6/CNT7。
这个 96/96 精确相等门禁验证完整 Submit 观察闭合；它不把 PMU 进程的
Submit span 变成无诊断墙钟基线，也不把 standalone 数据冒充真实 PA profile。

运行中切片的 `claim` 已在 A5 b1/b256 上证明可能发生单向少计，接受规则为：

```text
shadow_request <= primary_request
shadow_miss    <= primary_miss

request_loss = primary_request - shadow_request
miss_loss    = primary_miss - shadow_miss

phase_request_lower = phase_request_observed
phase_request_upper = phase_request_observed + request_loss
phase_miss_lower    = phase_miss_observed
phase_miss_upper    = phase_miss_observed + miss_loss
```

上下界必须先逐核计算，再分别聚合；不能拿聚合后的 median 相减拼区间。
CNT8/CNT5 是顺序 `ld_dev`，不是同一时刻的原子配对快照，因此局部
`phase_miss <= phase_request` 不是硬门禁。二者分别不超过对应 shadow，
各自上界不超过对应 primary。

结合 `none` exact、running phase bounded 和调用次数门禁，可分别验证：

- 复制 selector 在本机 A5 上确实计数；
- 边界调用覆盖预定代码片段，且 begin/end/tail 次数闭合；边界竞态少计
  由 loss 和 lower/upper 区间显式保留；
- primary whole 没有被局部归因读取破坏。

### 6.2 正式 JSON 必须通过的门禁

只有下列条件全部成立，host 才发布最终 JSON：

- 语义、winner 真计算输出和 Submit placement/engine 闭合通过；
- 96 条记录可信，32 AIC + 64 AIV，物理子核 id 唯一；
- owner bitmap membership、worker slot、物理 role 和 32 个 1:2 triplet 全部匹配；
- 96 个核都真实执行完整 Submit PMU start/stop，owner Restore 成功；
- build variant 和编译 phase id 在 96 条记录中全部匹配；
- `none` 的 shadow whole 与 primary whole 逐核精确相等；
- running phase 的 shadow whole 逐核不大于 primary，loss 与 upper-bound
  公式逐核闭合；exact 核数只作诊断；
- `none` 的 phase calls/begin/end/request/miss 全部为 0；
- `claim` 的 begin/end/calls 逐核平衡，每核 calls 为 `batches * 5`，
  全局 calls 为 `batches * 5 * 96`；
- phase request/miss 分别不超过对应 shadow/primary，且可编程 counter
  低于当前 25% 保守风险阈值。

`metrics_prof_start/stop()` 在完整 Submit 前后各执行一次，其
`PIPE_ALL` 边界会改变流水和多核时序。PMU 结果只与相同构建、
相同 phase、相同负载的独立进程比较，不把 PMU 进程的 Submit
span 当作无诊断性能基线。

## 7. JSON 字段与 AIC/AIV 分析口径

当前 `submit-pmu` 输出 schema v4。`records` 保留 96 个 worker 的 raw，
`summary.all/aic/aiv` 分别对 96/32/64 个核统计：

```text
sum / mean / median / p95 / max
```

完整 Submit 优先查看：

- `icache_requests` / `icache_misses`：CNT6/CNT7 primary whole；
- `shadow_whole_icache_requests` / `shadow_whole_icache_misses`：闭合或分段
  loss 用 shadow whole；
- `shadow_request_loss` / `shadow_miss_loss`：本核 primary-shadow residual；
- `phase_calls` / `phase_icache_requests` / `phase_icache_misses`：选定 phase
  的 running read-clear lower；
- `phase_icache_requests_upper_bound` / `phase_icache_misses_upper_bound`：
  lower 加本核对应 loss；
- `configuration.compiled_phase` 和 `validation.phase_measurement_valid`：确认文件口径。
- `validation.shadow_primary_match_records` / `shadow_primary_bounded_records`：
  区分逐值 exact 与单向 bounded 核数；
- `configuration.phase_values_are_running_read_clear_lower_bounds`：确认局部字段
  是否采用下界语义。

`phase_observed_read_clear_ratio` 只是 lower miss/lower request 的观测比值；
分子和分母各有独立区间，因此它不是实际 phase miss rate 的数学下界。

每核平均值按角色求：

```text
AIC request/core = summary.aic.icache_requests.sum / 32
AIC miss/core    = summary.aic.icache_misses.sum / 32
AIV request/core = summary.aiv.icache_requests.sum / 64
AIV miss/core    = summary.aiv.icache_misses.sum / 64
```

`median` 和 `p95` 直接来自同角色逐核 raw 分布，用于观察典型核和高
尾核。I-cache miss rate 只按组内加权口径计算：

```text
AIC miss rate = Σ(AIC miss) / Σ(AIC request)
AIV miss rate = Σ(AIV miss) / Σ(AIV request)
```

不平均 32 或 64 个逐核百分比。AIC/AIV 核数不同，比较每核强度时
使用 mean/median/p95 或 miss rate，不直接比较两组 sum。

## 8. 分析命令与结果文件

使用本用户 Python 环境从 raw 重算 host summary，并聚合相同配置的
多个独立进程：

```bash
PYTHON=/home/q00473782/.venv/bin/python

"$PYTHON" ./pmu_sidecar_analyzer.py \
  "$OUT/run1.json" "$OUT/run2.json" "$OUT/run3.json"
```

需要机器可读汇总时：

```bash
"$PYTHON" ./pmu_sidecar_analyzer.py --json \
  "$OUT/run1.json" "$OUT/run2.json" "$OUT/run3.json" \
  > "$OUT/summary.json"
```

分析器会先逐份复算 96 条 raw 与 host summary，然后拒绝聚合下列混用：

- `none` 与 `claim`；
- 不同 schema/build variant；
- 不同 batches、winner mode/count/pattern、selector 或观察开关。

建议将每次采集的 JSON 和分析器生成的 summary 放在同一唯一目录；
`outputs/` 为本机证据目录，不作为源码提交的一部分。

## 9. 如何使用约 90 ns/miss 标尺

当前约 `90 ns/miss` 来自已有隔离 cold/warm 微基准，只用于建立
数量级感性。对某一角色，可以计算：

```text
角色总 core-work 串行等效量(us) = Σmiss * 0.09
该角色平均每核等效量(us/core) = (Σmiss / core_count) * 0.09
```

例如 AIV 平均 70,000 miss/核，感性等效量是约 6,300 us/核。这不表示
Submit 墙钟真的损失了 6.3 ms，原因包括：

- 64 个 AIV 之间并行；
- 同一核的 miss、预取、其他流水和等待可能重叠；
- 隔离 cold miss 与真实热路 capacity/conflict/compulsory miss 不一定同价；
- 当前已核实的 A5 事件中没有可直接换算墙钟损失的 I-cache
  stall-cycle counter。

把 AIC 和 AIV 的 `Σmiss * 90 ns` 相加，只能得到全部核的感性
core-work 等效总量，不是端到端 Submit 总损失。要测真正暴露的性能
损失，必须对同语义代码做交错 A/B：

1. 用相同 `submit-pmu none` 口径确认 `ΔAIC/AIV miss/core`；
2. 另用不开 PMU/泳道的性能构建测 `ΔSubmit span`；
3. 只有第 2 项是实际暴露的墙钟收益；第 1 项用于证明收益与 I-cache
   变化同时出现，`90 ns` 仅提供一阶数量级解释。

## 10. 新增局部 phase 的修改清单

新 phase 不能只增加一个 CLI 字符串。最小完整修改包括：

1. `pa_scheduler/common/pa_model.h`：在 `SubmitPmuPhase` 尾部追加稳定
   id，不重排已有 `None=0/Claim=1`。
2. `pa_scheduler/ccec/pmu_probe.h`：为 `SubmitPmuPhaseName()` 增加名称映射，
   并核对 phase status/边界闭合定义。
3. `pa_scheduler/ccec/build.sh`：在白名单中将 phase 名映射到稳定
   `PA_SUBMIT_PMU_PHASE_ID`；先校验名称，再用于输出目录。
4. `pa_scheduler/run.sh`：同步 `build-submit-pmu/submit-pmu` 的 phase
   白名单和 usage。
5. `pa_scheduler/common/pa_scheduler_core.h`：在真实目标代码段前后放置
   `BeginSubmitPmuPhase<...>()` / `EndSubmitPmuPhase<...>()`。必须检查所有
   早退、winner/loser 和 Alloc/非 Alloc 分支，不得留下只 begin 不 end
   的路径。
6. `pa_scheduler/ccec/host.cpp`：增加该 phase 的预期 calls/begin/end
   形状。如果它不是每次 Submit 都调用，不能复用
   `batches * 5 * 96`。
7. `pa_scheduler/pmu_sidecar_analyzer.py`：同步 phase 名/id 和配置指纹，
   保证不同 phase 输入不会被聚合。
8. 补充 host/analyzer 回归：`none` 验证 96/96 primary-shadow 精确相等；
   running phase 验证逐核 bounded、loss/upper 公式、begin/end/calls 和语义，
   任一 shadow 反向大于 primary 都必须拒绝。先跑 A5 b1，再进入 b256。

每个 phase 必须是独立 ELF 和独立进程。不为了一次运行得到多个
phase，而在热路加运行时 phase switch 或多组 begin/end。

## 11. 常见问题与排错

### 提示缺少 submit-pmu 产物

确认 phase 名与构建命令一致：

```bash
./run.sh build-submit-pmu ccec none
./run.sh submit-pmu ccec none --device 0 --batches 1
```

不要用 `./run.sh build ccec` 代替；后者生成的是 `swimlane` 产物。

### host 提示 swimlane 构建不能采 PMU

这表示运行了 `build/ccec/pa_scheduler_host`。应通过
`./run.sh submit-pmu ...` 启动 phase 目录内的 host/kernel/SO 整套产物，
不要手工指向根目录 kernel。

### shadow miss 始终为 0

先检查 selector 是否错把 `0x35` 放到 CNT9。本机 A5 b1 已经反证
CNT9 路径；正式配置应为 CNT5 shadow miss、CNT8 shadow request、
CNT9 unused。

### `none` 不相等，或任意 phase 出现 `shadow > primary`

这两种情况都表示观察链路门禁失败，最终 JSON 不应发布。按下列顺序排查：

1. host/kernel/owner/dispatcher 是否来自同一 phase 目录；
2. owner 读回的 CNT5/CNT6/CNT7/CNT8 selector 是否与期望一致；
3. begin/end 数是否精确相等，是否有早退路径留下 armed phase；
4. 先缩到 b1；`none` 确认 96/96 exact，running phase 确认 96/96 bounded；
5. 检查可编程 counter 是否超过风险门槛。

running phase 出现小幅 `shadow < primary` 时，代码显式发布 loss 和局部
lower/upper；这不是 standalone 调度正确性异常。只有协议、数值输出、
placement/engine 也失败时，才应转向调度代码排查。

### owner 或 Restore 失败

不要在同一设备上并发运行另一个 standalone PMU owner、`msprof` PMU
会话或其他会改 selector 的进程。检查 CANN 环境、两个 AArch64 SO
是否在 kernel 同目录，以及 96 个可用 slot/32 个完整 triplet 是否闭合。

### JSON 拒绝覆盖

每个独立进程使用新文件名，并处理上次失败留下的同名 `.tmp`。
host 不会静默覆盖旧证据。

### 分析器报配置不一致

不要强行合并。逐项比较 phase、batches、winner workload/count/pattern、
selector、schema 和 build variant。重采相同配置的独立进程。

### miss rate 高，但 Submit 没有同比例变慢

这不构成矛盾。miss rate 是事件比例，不是 stall 时间比例；多核、
预取、流水重叠和资源等待都会改变实际暴露量。优先看 AIC/AIV
每核 miss、median/p95 和优化前后的 `Δmiss`，实际性能收益仍以无
PMU/泳道的交错 A/B 为准。
