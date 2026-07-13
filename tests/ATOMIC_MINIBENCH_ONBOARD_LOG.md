# Atomic Minibench 上板移植与修复记录（2026-07-10，更新 2026-07-13 v5）

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

## 2026-07-11 atomic probe 当前工作区直接上板证据

本节只记录 `tests/atomic_probe/` 当前未提交工作区的直接设备验证，不用下方旧提交结果替代。
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

临时单 AIV 隔离 control 为 20 launch × 100 trial：同址连续 257 次 `st_dev`、仅 loop-end DSB，
`0/2000` mismatch；逐写 DSB 同样 `0/2000`。因此当前证据支持的是“多个 AIV 并发写同一 64B line
的不同 slot 时存在干扰”，不支持“单 AIV 同址 store 自身乱序”。底层机制尚未由本测试判定。

复现入口：

```bash
source /home/q00473782/cann/cann-9.1.0/bin/setenv.bash
export PTO_ISA_ROOT=/home/q00473782/atomic/codex/simpler-fully_distributed/build/pto-isa
export ATOMIC_PROBE_DEVICE=0
tests/atomic_probe/ccec/run_all.sh st_dev_same_line
tests/atomic_probe/ascendc/_run_asc_probe.sh st_dev_same_line
```

### 当前工作区入口验证记录

同一工作区、同一环境的当前目标用例与改 oracle 前完整入口记录如下：

| 入口 | 结果 | 关键证据 |
|---|---|---|
| `tests/atomic_probe/ccec/run_all.sh st_dev_same_line` | exit 1 | 当前正确性 oracle：same-line mismatch 1589/4000；两个 control 0/4000；`semantic_failures=1` |
| `tests/atomic_probe/ascendc/_run_asc_probe.sh st_dev_same_line` | exit 1 | 当前正确性 oracle：same-line mismatch 1792/4000；两个 control 0/4000；`failures=1` |
| `tests/atomic_probe/ccec/run_all.sh` | 历史验证 exit 0 | 改为正确性断言前，所有 AIV-only probe 与 8-mode matrix 均完成；same-line mismatch 1634/4000，两个 control 0/4000 |
| `tests/atomic_probe/ascendc/_run_asc_probe.sh` | 历史验证 exit 0 | 改为正确性断言前，`failures=0 executables=11 sources=11`；same-line mismatch 1738/4000，两个 control 0/4000 |
| `.venv/bin/python -m pytest tests/atomic_probe/test_atomic_probe.py -k cpu -q` | exit 0 | `1 passed, 2 deselected` |

上述两个完整入口的 exit 0 来自旧的 `mismatch > 0` 成功条件，不再代表当前 oracle。当前
`st_dev_same_line` 要求同-line mismatch 为 0；设备复现任一 mismatch 时，目标用例及包含它的完整入口
都必须返回非零，不能把问题存在本身包装成 PASS。

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

**发现 A：scalar 写需要 dcci flush，vector 写不需要**

A/B 对照实验（去掉 flush，分别跑 vector_example 和 MB-2）：

| 写类型 | 硬件路径 | 到 HBM | 需要 dcci flush |
|--------|---------|--------|----------------|
| Scalar store（`out[i]=val`） | L1 data cache | ❌ 不自动 | ✅ 必须 `dcci CACHELINE_OUT` |
| Vector TSTORE | L2 cache | ✅ write-through | ❌ 不需要 |

vector_example（纯 TSTORE）去掉 flush 后仍 PASS；MB-2（scalar 裸写）去掉 flush
后 FAIL。当前主线 runtime 代码不含 flush，MB-2 scalar-write 场景需要 kernel
侧自行 flush 或在 runtime 层补 flush。

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

### MB-2 scalar 写需要 dcci flush（应在 kernel 侧自行处理）

MB-2 的 `kernel_write_index` 用裸 scalar 写（`out[0] = float(index) + 1`）。
A5 上 scalar store 只到 L1 data cache，需要 kernel 自行调用 `dcci CACHELINE_OUT`
刷到 HBM。**不应在 runtime 层泛化处理**——vector TSTORE 走 L2 write-through 不
需要 flush，统一 flush 会引入冗余开销。正确做法：谁写谁 flush。

### MB-8 过程中曾出现 kernel hang

MB-8（50 轮 producer→consumer）在 post-kernel flush 修复前会 timeout。
修复后通过。这验证了 ONBOARD_ISSUES.md Issue 8 中描述的 "kernel call hangs"
在本仓的根因是 **output data 未 flush** 而非 kernel_entry 符号冲突。
