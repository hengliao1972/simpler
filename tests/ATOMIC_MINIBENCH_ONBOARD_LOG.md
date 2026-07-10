# Atomic Minibench 上板移植与修复记录（2026-07-10，更新 2026-07-10 v2）

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
