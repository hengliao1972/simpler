# Atomic Minibench 上板移植与修复记录（2026-07-10，更新 2026-07-13 v9）

## 2026-07-13 DCCI selector 五模式与 AtomicExch 八模式对照

执行环境：base HEAD `c2739e23ed3a6729eeaaa4a3fa336875319d8c17`、dirty worktree、CANN 9.1、
`dav-3510`、device 0、PTO-ISA `ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8`。本轮经用户既有授权
直接占用 device 0，没有经过 `task-submit`；结果必须连同本节对应 diff 使用，不能只按 base HEAD 复现。

### ordinary clean line：DEFAULT/ALL/OUT/ATOMIC/no-DCCI

AscendC `mb8_dcci_seam.asc` 与 CCEC `dcci_seam.cpp` 使用两个 AIV。读核仅 normal-load data，
从不写 data line；写核仅 bypass-store data。data、两个 atomic phase、writer marker、reader result
均独占 64B line。每轮由 `ready`/`done` atomic 严格排序“读核预读旧值 → 写核写新值并 DSB →
读核 DCCI+DSB 后 normal/bypass 双读”，不存在 data line 的并发访问。

| 路径 | DEFAULT | ALL | OUT | ATOMIC | NO DCCI | GM/协议 |
|---|---:|---:|---:|---:|---:|---|
| CCEC | 100/0 | 100/0 | 100/0 | 100/0 | 0/100 | `other=0`、`gm_bad=0`，phase/marker/round 全精确 |
| AscendC | 100/0 | 100/0 | 100/0 | 100/0 | 0/100 | `other=0`、`gm_bad=0`，phase/marker/round 全精确 |

表中数字为 `fresh/stale`。ATOMIC 又在 AscendC、CCEC 各用独立进程重复 5 次，每次均为
`100/0`。no-DCCI 的 `0/100` 证明 DCCI 前 clean stale line 确实驻留，排除“自然 eviction 导致
假 fresh”。当前精确门禁说明三个 selector 在本机 A5、single-line ordinary clean entry 上都会使
后续 normal load 取得 GM 新值；不外推为其他 entry 类别或芯片的通用 ISA 契约。

两个 runner 将五个 mode 放在独立 host 进程中执行。初版同进程连续 launch 时第二次 launch 的
phase/marker 全为 0；该次没有留下能够证明 kernel 执行的标记，因此不能把结果归因于 OUT 的硬件行为。
CCEC runner 继续关闭 scalar auto-DCCI 与 kernel-end DCCI；AscendC runner 仅对本 DCCI probe 定向
显式关闭这两项，避免自动插入污染对照。

### ordinary dirty line：DCCI 与 AtomicExch 同/分 line

新增 AscendC/CCEC `dcci_atomic_clobber` 同构用例。核0先普通预读完整 data line，再 scalar store
邻接 word 形成 stale dirty line；DSB/独立 atomic phase 后，核1完成一次 AtomicExch；核0再次经
DSB/phase 取得权限后才执行 DCCI。整个时序严格串行，不存在两个核同时访问被测 data line。

| DCCI | atomic 与 dirty data 同 64B line | atomic 与 dirty data 分 64B line |
|---|---|---|
| ALL | dirty line 发布，AtomicExch 新值被旧快照覆盖 | dirty data 发布，AtomicExch 新值保留 |
| OUT | dirty line 发布，AtomicExch 新值被旧快照覆盖 | dirty data 发布，AtomicExch 新值保留 |
| ATOMIC | dirty line 发布，AtomicExch 新值被旧快照覆盖 | dirty data 发布，AtomicExch 新值保留 |
| NO DCCI | dirty scalar 值未发布，AtomicExch 新值保留 | dirty scalar 值未发布，AtomicExch 新值保留 |

AscendC 与 CCEC 八个 mode 的实际状态逐项一致，完整检查两条 16-word line、AtomicExch 返回旧值、
ready/done、两核 marker 与未使用 slot。同-line 的 ALL/OUT/ATOMIC 三个 mode 均触发正确性门禁并
以非零退出；分-line 与 no-DCCI 五个 control 均通过。结论是：atomic RMW 已完成不等于能阻止另一个
核随后进行的 stale dirty 整-line DCCI writeback；在本用例覆盖的 ordinary dirty line 场景中，
atomic 控制量必须与可能被 DCCI 的 data 分 64B line 管理。

复现入口：

```bash
source /home/q00473782/cann/cann-9.1.0/bin/setenv.bash
export PTO_ISA_ROOT=/home/q00473782/atomic/codex/simpler-fully_distributed/build/pto-isa
export ATOMIC_PROBE_DEVICE=0
tests/atomic_probe/ccec/run_all.sh dcci_seam
tests/atomic_probe/ascendc/_run_asc_probe.sh mb8_dcci_seam
tests/atomic_probe/ccec/run_all.sh dcci_atomic_clobber
tests/atomic_probe/ascendc/_run_asc_probe.sh dcci_atomic_clobber

# CACHELINE_ATOMIC clean-reader 路径的 5 次独立进程复测
for i in {1..5}; do ATOMIC_PROBE_MODE=3 tests/atomic_probe/ccec/run_all.sh dcci_seam; done
for i in {1..5}; do ATOMIC_PROBE_MODE=3 tests/atomic_probe/ascendc/_run_asc_probe.sh mb8_dcci_seam; done
```

## 2026-07-13 `st_dev` 分-line 独立压力

执行环境：base HEAD `c2739e23ed3a6729eeaaa4a3fa336875319d8c17`、dirty worktree、CANN 9.1、
`dav-3510`、device 0、PTO-ISA `ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8`。新增
`atomic_probe/ascendc/st_dev_separate_line_stress.asc` 与
`atomic_probe/ccec/st_dev_separate_line_stress.cpp`。本节是当前分-line 主证据；下方旧
`st_dev_same_line` 中的 `0/4000` 分-line 数据只是历史低压力样本。

该用例不包含任何两个 AIV 写同一 cacheline 的路径。每个 mode 在独立 host 进程中执行 500 launch；
每个 launch 有 100 trial，两个活跃 AIV 各自对独占 64B line 的 word0 连续执行 257 次 bypass store，
仅在循环末执行一次 DSB。随后所有已启动 AIV 会合，由 block0 bypass-read 两个终值。每个 mode 共
检查 100,000 个终值，并独立核对首错、slot 计数、最终 host snapshot、参与数、marker、未用 data
line 和 tail guard。

四模式结果如下：

| Mode | 活跃 block | data line | CCEC mismatch | AscendC mismatch |
|---:|---|---|---:|---:|
| 0 | block0 + block1 | line0 / line1 | 41484/100000 | 42165/100000 |
| 1 | block0 + block2；block1 data-idle | line0 / line1 | 39974/100000 | 37320/100000 |
| 2 | block0 + block1 | line1 / line2 | 0/100000 | 46/100000 |
| 3 | block0 + block2；block1 data-idle | line1 / line2 | 10/100000 | 5/100000 |

CCEC 的 slot 分解分别为 mode 0 `0/41484`、mode 1 `5/39969`、mode 2 `0/0`、mode 3 `5/5`；
AscendC 分别为 `42134/31`、`37301/19`、`25/21`、`0/5`。两种前端的具体失败 slot 不一致，
因此不能把问题归因到某个固定逻辑 block 或固定 writer。

两端本轮 allocation 首地址均输出 `mod128=0`、`mod256=0`、`mod512=0`。mode 0/2 的记录为
block0 `(core18,sub0,comm_slot0)`、block1 `(core72,sub0,comm_slot18)`；mode 1/3 另有 data-idle
block1 `(core19,sub0,comm_slot0)`，第二个活跃者 block2 为 `(core72,sub0,comm_slot18)`。
`comm_slot` 严格按 PTO `TSYNC_CVID` 的 signed 公式派生，只是软件 CV 通信配对编号，不是硬件物理
组号。所有 protocol、participation、marker、guard 与 `comm_slot` 公式检查均为 0 failure。

因此当前证据足以否定“两个 AIV 写不同 64B cacheline 就必然安全”。line0/line1 布局在两种前端
均高频复现，line1/line2 布局也出现低频 mismatch；CCEC mode 2 的单次 `0/100000` 不能作为安全
保证。allocation 内部 line offset 与本轮失败频率相关，但当前用例没有区分 cache bank、set、store
队列或其他底层机制，不能指定根因；结论也不外推为单 AIV 同址 store 自身乱序。

旧双 AIV/AscendC 与 CCEC mode 0 分-line 路径使用 allocation 内 line1/line2；CCEC mode 1 使用
line5/line6。它们都只有 4000 次检查，而且与同-line、逐轮 DSB 路径共处一个 kernel 时序。旧路径
曾跑出 `0/4000`，也曾出现 `1/4000`；当前同版本 CCEC mode 0 又得到 `3/4000`。新用例只保留
分-line 数据路径、覆盖 line0/line1 与 line1/line2，并把样本提高到 100,000 次，因此能更稳定地
暴露问题且没有引入同-line 数据交互。line5/line6 尚未由新独立用例同压复测，两类用例的完整指令
序列也并非逐条相同，当前证据不能把频率差异进一步归因到某个底层机制。

复现入口：

```bash
source /home/q00473782/cann/cann-9.1.0/bin/setenv.bash
export PTO_ISA_ROOT=/home/q00473782/atomic/codex/simpler-fully_distributed/build/pto-isa
export ATOMIC_PROBE_DEVICE=0
tests/atomic_probe/ccec/run_all.sh st_dev_separate_line_stress
tests/atomic_probe/ascendc/_run_asc_probe.sh st_dev_separate_line_stress

# 可缩短或扩大每个 mode 的独立 launch 数
ATOMIC_PROBE_STRESS_LAUNCHES=500 ATOMIC_PROBE_MODE=0 \
    tests/atomic_probe/ccec/run_all.sh st_dev_separate_line_stress
ATOMIC_PROBE_STRESS_LAUNCHES=500 ATOMIC_PROBE_MODE=0 \
    tests/atomic_probe/ascendc/_run_asc_probe.sh st_dev_separate_line_stress
```

## 2026-07-13 AtomicExch 同-line 对照

执行环境：base HEAD `ea1639f19f0a97ea2e7f98ff56ff6c9a935373c1`、CANN 9.1、`dav-3510`、
device 0、PTO-ISA `ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8`。新增
`atomic_probe/ascendc/atomic_exch_same_line.asc` 与
`atomic_probe/ccec/atomic_exch_same_line.cpp`；CCEC 复用同布局 host。

两条路径固定两个 AIV，每核独占一个 4B slot，执行 20 launch × 100 trial × 257 round。除把
`st_dev` / `WriteGmByPassDCache<uint32_t>` 替换为 `atomicExch` / `AtomicExch<uint32_t>` 外，
布局、值公式、DSB 和跨 AIV 同步点均与 `st_dev_same_line` 相同。结果：

| 路径 | 同 line、仅 loop-end DSB | 分 line、仅 loop-end DSB | 同 line、逐轮 DSB |
|---|---:|---:|---:|
| CCEC `atomicExch` | 0/4000 mismatch | 0/4000 | 0/4000 |
| AscendC `AtomicExch<uint32_t>` | 0/4000 mismatch | 0/4000 | 0/4000 |

两个 runner 均 exit 0，参与计数和 marker 全部精确。该证据仅说明 AtomicExch 在本同构压力下没有
复现 st_dev 的末值回退，不外推到 AtomicAdd/AtomicMax/CAS 或其他数据宽度、核拓扑和内存序场景。

复现入口：

```bash
tests/atomic_probe/ccec/run_all.sh atomic_exch_same_line
tests/atomic_probe/ascendc/_run_asc_probe.sh atomic_exch_same_line
```

## 2026-07-13 CCEC 三 AIV 拓扑与正式单 AIV `st_dev` 对照

执行环境：base HEAD `c2739e23ed3a6729eeaaa4a3fa336875319d8c17`、dirty worktree、CANN 9.1、
`dav-3510`、device 0、PTO-ISA `ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8`。扩展
`atomic_probe/ccec/st_dev_same_line.cpp`，runner 以三个独立 host 进程执行：

- mode 0：启动两个 block，block0 与 block1 写数据，保留原双 AIV 回归路径；
- mode 1：启动三个 block，仅 block0 与 block2 写数据；block1 只写独占拓扑 marker，并参加每个
  flag-14 `SyncAll`，从始至终不访问任何被测 data line；
- mode 2：只启动一个 block，同一地址连续执行 257 次 `st_dev`，分别只在循环末 DSB 和逐写 DSB。

每个 marker 独占 64B，记录逻辑 block、`get_coreid()`、`get_subblockid()` 以及按本机 PTO-ISA
`TSYNC_CVID` A5 公式计算的 `comm_slot`。该值是软件 CV 通信配对编号，不是硬件物理组号。20 次
launch 中记录保持一致：

| mode | core/subblock/comm_slot 记录 | 同-line、仅 loop-end DSB | 旧分-line 低压路径 | 同-line、逐写 DSB |
|---:|---|---:|---:|---:|
| 0 | block0 `(core18,sub0,comm0)`；block1 `(core72,sub0,comm18)` | 1679/4000 mismatch | 0/4000 | 0/4000 |
| 1 | block0 `(18,0,0)`；idle block1 `(19,0,0)`；block2 `(72,0,18)` | 1736/4000 mismatch | 0/4000 | 0/4000 |
| 2 | block0 `(18,0,0)` | 同址 0/2000 | 不适用 | 同址 0/2000 |

mode 0/1 的 active writer 记录为不同 `comm_slot`；该派生编号不能用于解释硬件物理位置。两者仍
复现同-line 终值错误并按正确性契约 exit 1。mode 2 exit 0，说明本轮正式单 AIV 同址用例没有复现
末值回退。结论只限定为该设备与当前压力下的观测，不把它扩大为底层乱序机制说明。

加入首错 actual/expected 诊断前的一次独立运行中，mode 0/1 的分-line 路径各出现过
`1/4000` mismatch；表中带诊断复跑当时均为 `0/4000`，当前同版本 mode 0 又得到 `3/4000`。永久
用例仍要求 mismatch 精确为 0，出现非零必须作为正确性失败暴露。该低压力路径的安全性结论已被
上方 `st_dev_separate_line_stress` 的 100,000 次独立压力结果取代。

由于 `st_dev_same_line_host.cpp` 同时被原有 `atomic_exch_same_line` 复用，扩展后又执行兼容回归：
CCEC AtomicExch 的同-line、分-line、同-line 逐轮 DSB 均为 `0/4000`，20 次参与计数和 marker 全精确，
runner exit 0。

复现入口：

```bash
source /home/q00473782/cann/cann-9.1.0/bin/setenv.bash
export PTO_ISA_ROOT=/home/q00473782/atomic/codex/simpler-fully_distributed/build/pto-isa
export ATOMIC_PROBE_DEVICE=0
tests/atomic_probe/ccec/run_all.sh st_dev_same_line
tests/atomic_probe/ccec/run_all.sh atomic_exch_same_line
```

## 2026-07-11 atomic probe 两 AIV 原始上板证据（历史）

本节记录当时 `tests/atomic_probe/` 未提交工作区的直接设备验证；CCEC 最新三 mode 结果以上方
2026-07-13 小节为准，分-line 结论以最上方独立压力小节为准，不用本节旧数字替代。
执行环境：base HEAD `57841544fe4c2360ea703eeb583d6112cd379037`、CANN 9.1、
`dav-3510`、device 0、PTO-ISA `ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8`。本轮经用户授权直接
占用设备，没有经过 `task-submit`，所以结果必须连同 dirty worktree diff 使用，不能只按 base HEAD 复现。

### 两个 AIV 并发 `st_dev` / `WriteGmByPassDCache` 同 line 最简对照

永久用例：

- `atomic_probe/ccec/st_dev_same_line.cpp`
- `atomic_probe/ascendc/st_dev_same_line.asc`

每次 launch 精确验证两个 AIV 的参与计数和 marker；每个 AIV 只写自己的 4B slot。20 次 launch、每次
100 trial、每个 trial 257 轮，结果如下：

| 路径 | 同 line、仅 loop-end DSB | 分 line、仅 loop-end DSB | 同 line、逐轮 DSB |
|---|---:|---:|---:|
| CCEC `st_dev` | 1589/4000 mismatch（正确性断言失败，exit 1） | 0/4000 | 0/4000 |
| AscendC `WriteGmByPassDCache<uint32_t>` | 1792/4000 mismatch（正确性断言失败，exit 1） | 0/4000 | 0/4000 |

当时的临时单 AIV 隔离 control 为 20 launch × 100 trial：同址连续 257 次 `st_dev`、仅 loop-end DSB，
`0/2000` mismatch；逐写 DSB 同样 `0/2000`。该路径现已由上方 2026-07-13 CCEC mode 2 固化。

复现入口：

```bash
source /home/q00473782/cann/cann-9.1.0/bin/setenv.bash
export PTO_ISA_ROOT=/home/q00473782/atomic/codex/simpler-fully_distributed/build/pto-isa
export ATOMIC_PROBE_DEVICE=0
tests/atomic_probe/ccec/run_all.sh st_dev_same_line
tests/atomic_probe/ascendc/_run_asc_probe.sh st_dev_same_line
```

### 当时工作区入口验证记录

同一工作区、同一环境在当时版本上的目标用例与修改判定条件前的完整入口记录如下。这些
`0/4000` 分-line 样本不代表上方 dedicated separate-line 压力的当前结论：

| 入口 | 结果 | 关键证据 |
|---|---|---|
| `tests/atomic_probe/ccec/run_all.sh st_dev_same_line` | exit 1 | 当时 20-launch 判定：same-line mismatch 1589/4000；两个低压对照 0/4000；`semantic_failures=1` |
| `tests/atomic_probe/ascendc/_run_asc_probe.sh st_dev_same_line` | exit 1 | 当时 20-launch 判定：same-line mismatch 1792/4000；两个低压对照 0/4000；`failures=1` |
| `tests/atomic_probe/ccec/run_all.sh` | 历史验证 exit 0 | 改为正确性断言前，所有 AIV-only probe 与 8-mode matrix 均完成；same-line mismatch 1634/4000，两个 control 0/4000 |
| `tests/atomic_probe/ascendc/_run_asc_probe.sh` | 历史验证 exit 0 | 改为正确性断言前，`failures=0 executables=11 sources=11`；same-line mismatch 1738/4000，两个 control 0/4000 |
| `.venv/bin/python -m pytest tests/atomic_probe/test_atomic_probe.py -k cpu -q` | exit 0 | `1 passed, 2 deselected` |

上述两个完整入口的 exit 0 来自旧的 `mismatch > 0` 成功条件，不再代表当前判定标准。
`st_dev_same_line` 与 `st_dev_separate_line_stress` 都要求各自终值 mismatch 为 0；设备复现任一
mismatch 时，目标用例及包含它的完整入口都必须返回非零，不能把问题存在本身包装成 PASS。

CCEC publish/observe seam 曾在前序压力后出现 40/100 stale reads。producer 在 16 个 payload `st_dev` 完成后、
发布 atomic flag 前加入一处 `dsb(DSB_ALL)`；保持 consumer 不变后，单项压力后与完整 runner 均为
flag=100、errors=0。

## 当前 pull 验证（`e63f072d`）

本轮测试仓库为 `/home/q00473782/atomic/sfd/gpt/simpler`，HEAD 为
`e63f072d`（`Fix: resolve CCEC Stride ambiguity, symbol conflicts, and scalar dcci for all MB onboard`）。
测试开始前工作区 clean；该提交已包含 PTO 双 include、`pto::Stride` 完全限定、AIV 重名函数修复，
以及 atomic minibench scalar kernel 的 `dcci CACHELINE_OUT`。

### 当前 HEAD a5 上板结果

| 测试 | 结果 |
|------|------|
| MB-2 flags 512 | ✅ 1 passed（与 MB-8 合并运行） |
| MB-4 block.won | ✅ 1 passed，34.86s |
| MB-5 shared_map | ✅ 1 passed，27.68s |
| MB-6 heap | ✅ 1 passed，27.31s |
| MB-7 runahead | ✅ 1 passed，27.71s |
| MB-8 DCCI 50 rounds | ✅ 1 passed（与 MB-2 合并运行，总 41.16s） |
| benchmark_bgemm | ✅ 1 passed，42.93s |
| vector_example | ✅ 1 passed，30.05s |
| simple + submit dependency smoke | ✅ 2 passed，142.27s |

MB-4/5/6/7 已越过此前的 CCEC 编译阻塞并完成真硬件运行。下方旧验证矩阵中的
`01a706e3` 结果是历史基线，不代表本次 `e63f072d`。

## 当前状态（基线 `01a706e3`，dist_engine 重构后）

基线代码将 `dist_engine.cpp` 拆分为 `dist_engine/aicore/`、`dist_engine/aicpu/`、
`dist_engine/common/` 多文件结构。本仓移植的测试用例无需修改即与新结构兼容。

## 背景

将 `simpler-fully_distributed` 分支的 atomic minibench 测试用例（MB-1~MB-9）
移植到本仓（`sfd/glm/simpler`），验证 `fully_distributed_within_core` runtime
在 A5 上板的跨核数据竞争防护。

## 移植的文件

### C++ UT（纯逻辑，host 多线程）

| 文件 | MB | 位置 |
|------|----|------|
| `test_dist_atomic_mb1_claim.cpp` | MB-1 | `tests/ut/cpp/a5/` |
| `test_dist_atomic_mb3_frontier.cpp` | MB-3 | `tests/ut/cpp/a5/` |
| `test_dist_atomic_mb5_shared_map.cpp` | MB-5 | `tests/ut/cpp/a5/` |
| `test_dist_atomic_mb9_private_det.cpp` | MB-9 | `tests/ut/cpp/a5/` |
| `test_dist_tensormap_ring.cpp` | 参考 | `tests/ut/cpp/a2a3/` |
| `CMakeLists.txt` 新增 | — | `add_standalone_dist_test()` + 5 行注册 |

### Python ST（端到端，真硬件）

| 文件 | MB | 位置 |
|------|----|------|
| `test_mb2_flags.py` | MB-2 | `tests/st/a5/.../atomic_minibench/` |
| `test_mb4_block_won.py` | MB-4 | 同上 |
| `test_mb5_shared_map.py` | MB-5 | 同上 |
| `test_mb6_heap.py` | MB-6 | 同上 |
| `test_mb7_runahead.py` | MB-7 | 同上 |
| `test_mb8_dcci.py` | MB-8 | 同上 |
| MB-2/MB-8 orch + kernels | — | 同上 `kernels/` |
| `mix_coown/`（MB-4 依赖） | — | `tests/st/a5/.../mix_coown/` |
| `vector_example/` | — | `tests/st/a5/.../vector_example/` |
| `atomic_probe/` | 硬件探针 | `tests/atomic_probe/` |

## 适配修复

### 1. 环境（editable install 路径修正）

`_simpler_editable.py` / `.pth` 硬编码旧目录，已替换为当前目录。

本机已有 PTO-ISA checkout，测试和 editable build 应直接复用，避免再次 clone：

```bash
export PTO_ISA_ROOT=/home/q00473782/atomic/codex/simpler-fully_distributed/build/pto-isa
git -C "$PTO_ISA_ROOT" rev-parse HEAD
# ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8
test -f "$PTO_ISA_ROOT/include/pto/pto-inst.hpp"
```

`PTO_ISA_ROOT` 必须指向仓库根目录，构建器会使用其 `include/`；无需把 PTO-ISA 复制进本仓。

### 2. Build 管道

| 文件 | 改动 | 原因 |
|------|------|------|
| `runtime_compiler.py` | cross-arch strip 改为 non-fatal | AICPU SO 是 aarch64，host strip 无法读取 |
| `runtime_builder.py` | `build_aicore_with_extra_sources` 加入 `PTO_ISA_ROOT/include` 到 CCEC include path | PTO-ISA tile kernel 需要 `pto/pto-inst.hpp` 等头文件 |
| `scene_test.py` | fdwic onboard 跳过 `extract_text_section` | kernel 已通过 incore wrapper 编入 aicore image，单独 text extraction 不必要且 PTO-ISA tile API 会产生 `.text` 重定位 |

### 3. Kernel/orch 适配（源仓 → 本仓 API 差异）

| 改动 | 原因 |
|------|------|
| `#include <pto/pto-inst.hpp>` → 条件 `__has_include("inner_kernel.h")` + `__has_include(<pto/pto-inst.hpp>)` 双 include | CCEC 下 `inner_kernel.h` 提供平台宏，`pto-inst.hpp` 提供 tile API，两者不互斥 |
| `using namespace pto;` → `using namespace pto; using ::pto::Stride;` | CCEC 的 CANN 头文件定义了全局 `Stride` enum，与 `pto::Stride` 冲突 |
| orch 函数属性加 `weak` + `PTO_DEVICE_FUNC` | CCEC 下无 `PTO_DEVICE_FUNC` 的函数被视为 host 函数，无法访问 device API |
| `ext_out.view()` → `Tensor::view()` | CCEC 不允许在 `__gm__` 引用上调非 `__gm__` this 的 member |
| `ext_config.data_as<T>()` → `orch_args.tensor(i).ref().data_as<T>()` | `TensorRef` 无 `data_as`，需 `.ref()` 先获取 `Tensor&` |
| `PTO_ORCH_ENTRY` → `weak PTO_DEVICE_FUNC` | 源仓宏在本仓不存在 |
| `prod_outs.get_ref(0)` → `add_input(ext_buf)` | 本仓 INOUT→INPUT 通过 TensorMap 自动建依赖，不需要 `TaskOutputTensors::get_ref()` |
| `ELEMS` 常量重命名为 `PRODUCER_ELEMS`/`CONSUMER_ELEMS` | CCEC 把同目录所有 .cpp 编在一起，同名 namespace-scope constexpr 冲突 |

### 4. Runtime 修复（上板根因）——已回退，仅记录发现

> 以下 dist_engine.cpp 改动在 `a83196e4` 中引入，在 `b422c48f` 中回退。
> 原因：A/B 对照实验证明现有主线测试（vector TSTORE kernel）不需要这些改动。

**历史发现 A：该 scalar 用例需要显式发布，vector 用例不需要**

> 这是早期 A/B 记录中的用语。当前 selector 专项测试只证明普通 scalar dirty line 在执行
> DEFAULT/ALL/OUT/ATOMIC 后均被发布；不能据此声称只有 OUT 才能发布，也不把 GM 可见性等同为
> 已追踪到物理 HBM。

A/B 对照实验（去掉 flush，分别跑 vector_example 和 MB-2）：

| 写类型 | 硬件路径 | host/ld_dev 可见 | 该旧用例采用的发布手段 |
|--------|---------|--------|----------------|
| Scalar store（`out[i]=val`） | L1 data cache | ❌ 不自动 | `dcci CACHELINE_OUT` |
| Vector TSTORE | L2 cache | ✅ | 不执行 DCCI |

vector_example（纯 TSTORE）去掉 DCCI 后仍 PASS；MB-2（scalar 裸写）去掉 DCCI
后 FAIL。当前主线 runtime 代码不含这一步显式 DCCI，MB-2 scalar-write 场景需要 kernel
侧自行发布，或另行论证是否应由 runtime 处理。

**发现 B：完成标志 publish/read 的 cacheline clobber / TOCTOU**

`publish_task_flag` 用 `cell.flag = 1; ccec_flush_region(...)` 有 cacheline
clobber 风险。`task_flag_ready` 用 `invalidate + plain load` 有 TOCTOU 窗口。
改用 `atomicMax` 可避免。此改动也已回退，当前主线仍用 store+flush / invalidate+load。

## 验证矩阵

### C++ UT（`ctest`，纯 host 逻辑）

| 测试 | 结果 |
|------|------|
| test_dist_atomic_mb1_claim | ✅ PASS |
| test_dist_atomic_mb3_frontier | ✅ PASS |
| test_dist_atomic_mb5_shared_map | ✅ PASS |
| test_dist_atomic_mb9_private_det | ✅ PASS |
| test_dist_tensormap_ring | ✅ PASS |

### Python ST — a5sim

| 测试 | 结果 |
|------|------|
| simple_orch_smoke | ✅ |
| submit_dependency_smoke | ✅ |
| benchmark_bgemm | ✅ |
| vector_example | ✅ |
| MB-2 flags 512 | ✅ |
| MB-4 block.won | ✅ |
| MB-5 shared_map | ✅ |
| MB-6 heap | ✅ |
| MB-7 runahead | ✅ |
| MB-8 dcci 50 rounds | ✅ |
| mix_coown | ✅ |

### Python ST — a5 onboard（真硬件，基线 `01a706e3`）

| 测试 | 结果 | 说明 |
|------|------|------|
| simple_orch_smoke | ✅ | 不回归 |
| submit_dependency_smoke | ✅ | 不回归 |
| vector_example | ✅ | 纯 vector TSTORE，L2 write-through |
| MB-2 flags 512 | ❌ | scalar 裸写需 flush，主线无 flush |
| MB-8 dcci 50 rounds | ✅ | 纯 vector，不需要 flush |
| MB-4 block.won | ❌ CCEC 编译 | PTO-ISA `Stride` 歧义 |
| MB-5 shared_map | ❌ CCEC 编译 | 同上 |
| MB-6 heap | ❌ CCEC 编译 | 同上 |
| MB-7 runahead | ❌ CCEC 编译 | 同上 |

## 遗留问题

### MB-4/5/6/7 CCEC 编译失败：`Stride` 名称歧义

**根因**：PTO-ISA 头文件定义 `pto::Stride`（struct 模板），CANN 头文件定义
全局 `::Stride`（enum class）。当 `using namespace pto;` 后使用 `Stride`
时，CCEC 无法消歧。

**已尝试**：
- `using ::pto::Stride;` 局部覆盖 —— g++ 通过，CCEC 仍报歧义
- `using Stride = pto::Stride;` 类型别名 —— g++ 报 "auto not allowed"（因为
  `pto::Stride` 在 sim g++ 上下文中可见性不同）
- 去掉 `using namespace pto` 改用逐类型 `using pto::TileData;` 等 ——
  `TileData`/`GlobalMatrix` 等是 kernel 内 local typedef，不在 `pto::` 里

**建议方向**：
1. 在 kernel 函数体内用完全限定名（`pto::Shape<...>`、`pto::Tile<...>`），
   不依赖 `using namespace pto`
2. 或在 PTO-ISA 头文件中 rename `Stride` 为 `TensorStride` 等
3. 或用 CCEC `-DStride=pto::Stride` 宏覆盖（hack）

### MB-2 scalar 写需要显式发布（应在 kernel 侧自行处理）

MB-2 的 `kernel_write_index` 用裸 scalar 写（`out[0] = float(index) + 1`）。
A5 上该 scalar store 不会自动变成 host/ld_dev 可见值；旧实现用 kernel 内的
`dcci CACHELINE_OUT` 发布。**不应在 runtime 层泛化处理**——vector TSTORE 的旧对照不需要
DCCI，统一插入会引入冗余开销。具体选择哪个 DcciDst 必须按目标 entry 类型另行验证。

### MB-8 过程中曾出现 kernel hang

MB-8（50 轮 producer→consumer）在 post-kernel flush 修复前会 timeout。
修复后通过。这验证了 ONBOARD_ISSUES.md Issue 8 中描述的 "kernel call hangs"
在本仓的根因是 **output data 未 flush** 而非 kernel_entry 符号冲突。
