# A5 CCEC/AscendC ICache/DCache Preload 用法与真机验证

## 1. 当前交付边界

本目录现在提供两套可独立构建和运行的对等用例：

- CCEC 用例直接调用编译器 intrinsic，不包含 `kernel_operator.h`；
- AscendC 用例只调用公开的 `kernel_operator.h` API；
- 两者复用同一套 mode、host/device ABI、输入、checksum oracle、冷态构造和
  raw `SYS_CNT` 统计口径，以便区分“API 用法差异”和“测试模型差异”。

用户口语中的 `icache_pretch`、`dcache_pretch` 不是当前 CANN 头文件中的符号。正确检索词是：

| 目的 | CCEC 接口 | AscendC 接口 | 说明 |
| --- | --- | --- | --- |
| 从当前 PC 预取后续指令 | `icache_preload(len)` | `ICachePreLoad(len)` | `len` 在 Atlas 350 上以 2 KiB 为单位 |
| 查询 ICache preload 状态 | `get_icache_prl_st()` | `GetICachePreloadStatus()` | `0=idle`，`1=busy` |
| 从 GM 预取数据到 DCache | `dc_preload(base, byteOffset)` | `DataCachePreload(tensor, byteOffset)` | offset 单位为 byte |
| 指定 ICache 地址的底层接口 | `preload(addr, len)` | 无需在业务中直接使用 | 本探针只测试 current-PC 接口 |

本次 A5 真机验证观察到：

- CCEC DCache 普通 GM load 中位数为 `308 -> 5`，AscendC 为 `349 -> 5` raw ticks；
- CCEC ICache 同一物理指令区中位数为 `995 -> 501`，AscendC 为
  `1162 -> 792` raw ticks；
- 两套实现的两个 ICache preload mode 都在发起后立即得到 `9/9 busy`，独立 gap
  之后均得到 `9/9 idle`；
- 立即轮询等待没有降低目标工作区间：CCEC 完整区间为 `1451 -> 1454`，
  AscendC 为 `1632 -> 1648` raw ticks。

这些是当前 microprobe 的真机观察，不是 PA 或其他业务的固定收益承诺。

## 2. 文件与运行方式

| 文件 | 用途 |
| --- | --- |
| `ccec/cache_preload.cpp` | 纯 CCEC AIV kernel，直接调用 preload intrinsic |
| `cache_preload_shared.h` | CCEC/AscendC/host 共用 ABI、模式和 checksum oracle |
| `ccec/cache_preload_host.cpp` | 真机 launcher、逐样本校验和中位数汇总 |
| `ccec/run_cache_preload.sh` | CCEC 独立构建、最终 ELF 门禁和定向运行 |
| `ascendc/cache_preload_probe.asc` | AscendC AIV kernel、host launcher 和校验 |
| `ascendc/run_cache_preload.sh` | AscendC 独立构建、嵌入 AICore ELF 门禁和定向运行 |

从仓库根目录运行：

```bash
tests/atomic_probe/ccec/run_cache_preload.sh
tests/atomic_probe/ascendc/run_cache_preload.sh
```

两套脚本都可以拆开：

```bash
tests/atomic_probe/ccec/run_cache_preload.sh build
tests/atomic_probe/ccec/run_cache_preload.sh run

tests/atomic_probe/ascendc/run_cache_preload.sh build
tests/atomic_probe/ascendc/run_cache_preload.sh run
```

脚本是独立入口，没有修改 `ccec/run_all.sh`，也不会误跑目录内其他探针。AscendC
runner 会从可执行文件提取 `.aicore_binary`，再对实际执行的 `.vector` 符号做尺寸、
对齐和地址不重叠检查，不能用 host ELF 的表面尺寸替代该门禁。

## 3. 已查证的接口语义

### 3.1 `dc_preload`

CCEC 业务参考形态：

```cpp
__gm__ uint8_t *metadata_bytes = /* GM base */;
const int64_t byte_offset =
    static_cast<int64_t>(entry_index) * entry_stride_bytes;

dc_preload(
    reinterpret_cast<__gm__ uint64_t *>(metadata_bytes),
    byte_offset);

// 必须放已有且与目标 load 无数据依赖的工作，创造隐藏 miss 的窗口。
DoIndependentWork();

const uint64_t value =
    *reinterpret_cast<volatile __gm__ uint64_t *>(
        metadata_bytes + byte_offset);
```

AscendC 对等形态：

```cpp
AscendC::GlobalTensor<uint64_t> metadata;
metadata.SetGlobalBuffer(metadata_base, metadata_word_count);

const int64_t byte_offset =
    static_cast<int64_t>(entry_index) * entry_stride_bytes;
AscendC::DataCachePreload(metadata, byte_offset);

DoIndependentWork();
const uint64_t value =
    *reinterpret_cast<volatile __gm__ uint64_t *>(
        reinterpret_cast<__gm__ uint8_t *>(metadata_base) + byte_offset);
```

当前 CANN 9.1 weekly 的 `dav_3510/kernel_operator_cache_impl.h` 也把上层
`DataCachePreload` 直接下沉为：

```cpp
dc_preload(src, cacheOffset);
```

官方接口契约确认：

- 源地址按 `uint64_t` GM tensor/指针描述；
- `cacheOffset` 单位是 byte，上层 API 支持 `int16_t`/`int64_t`；
- 作用是从特定 GM 地址预加载到 data cache；
- 频繁调用可能造成保留站拥塞，此时指令可能被当作 NOP，并阻塞 Scalar 流水。

因此不能在每个短小 load 前机械调用。只有“地址能提前确定、后面有独立工作、目标很可能被消费”
的点才值得尝试。

当前 CANN ListTensor 内部实现有“从 miss offset 预取 256 Bytes”的注释，但公开接口文档没有把
256 B 定义为跨版本契约。业务正确性不能依赖固定预取范围。

官方文档：
[DataCachePreload](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0176.html)。

### 3.2 `icache_preload`

当前 Bisheng CCEC 头文件中的实现是：

```cpp
inline void icache_preload(int64_t len)
{
    preload(reinterpret_cast<const void *>(get_pc()), len);
}
```

也就是说，它不是“按 C++ 函数名预取”，而是从调用点当前 PC 所在的指令地址开始预取。
两套发起方式分别为：

```cpp
icache_preload(2);          // CCEC
AscendC::ICachePreLoad(2);  // AscendC；业务只选与当前实现对应的一行

// 最好执行会跳到别处的小段已有工作，让当前 PC 后方代码在后台预取。
DoIndependentControlWork();

RunUpcomingSequentialHotPath();
```

调用点应位于即将执行的顺序热点代码之前。若调用后马上跳到无关且很远的分支，预取到的代码可能不被
消费；若调用太晚，miss 已经发生，也无法隐藏延迟。

官方文档对 Atlas 350 说明：

- `len`/`preFetchLen` 单位为 2 KiB；
- 取值应小于 `ICache size / 2 KiB`；
- 文档列出的 AIC/AIV ICache 大小分别是 32 KiB/16 KiB。

本轮两套实现都使用 2 units，即 4 KiB。最终 ELF 中，CCEC current-PC 被测函数为
4,792 B，AscendC 实际执行的 `.vector` 被测函数为 4,808 B。

当前 CANN OPP 的 arch35 存量代码也采用相同模式。例如
`mat_mul_v3/arch35/mat_mul_stream_k_kernel.h` 在 AIV 分支调用 `ICachePreLoad(2)` 后直接进入后续
处理，没有立即轮询；这与“发起后继续可重叠工作”的用法一致。

官方文档：
[ICachePreLoad](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0276.html)。

### 3.3 `get_icache_prl_st`

两套查询方式分别为：

```cpp
const int64_t ccec_status = get_icache_prl_st();
const int64_t ascendc_status = AscendC::GetICachePreloadStatus();
```

返回值：

- `0`：idle；
- `1`：busy。

两套探针都保留了 wait 对照，下面展示 AscendC 写法；CCEC 只需替换成对应的
小写 intrinsic：

```cpp
AscendC::ICachePreLoad(2);
while (AscendC::GetICachePreloadStatus() != 0) {
    asm volatile("nop");
}
RunUpcomingSequentialHotPath();
```

这只是状态契约验证，不是推荐的性能写法。为了隐藏 miss，通常应让其他有用工作和预取重叠；立即轮询
会把异步行为重新串行化，并增加 Scalar 指令。

官方上层状态文档：
[GetICachePreloadStatus](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0277.html)。

## 4. 真机探针如何避免错误归因

### 4.1 DCache 冷读对照

每个样本使用相隔 512 B 的新目标地址，baseline 与 preload 执行：

1. 在计时区间外对目标行执行单行 DCCI invalidation 和完成等待；
2. baseline 不发起 hint；CCEC preload 调用
   `dc_preload(state->data, targetWord * sizeof(uint64_t))`，AscendC 调用
   `DataCachePreload(data, targetWord * sizeof(uint64_t))`；
3. 两条路径调用完全相同的 noinline 64-round Scalar gap；
4. gap 的 checksum 通过 opaque register dependency 约束目标地址计算；
5. 执行普通 volatile GM load；
6. host 校验 load 值、gap checksum、mode/offset echo。

DCCI 只用于建立 probe 的冷态起点，位于测量区间之外；它不是 preload 业务写法的一部分。

### 4.2 ICache 冷态与同一物理代码区

为了避免“源码写了很多 NOP，但最终 ELF 并没有形成足够指令 footprint”的假测试，构建脚本直接检查
链接后实际 AIV ELF：

| 实现 | 符号 | 最终地址 | 最终大小 | 门禁 |
| --- | --- | ---: | ---: | --- |
| CCEC | `cache_preload_icache_evictor` | `0x100` | 32,836 B | ≥32 KiB、128 B 对齐 |
| CCEC | `cache_preload_icache_path` | `0x8180` | 4,792 B | ≥4 KiB、128 B 对齐 |
| AscendC | `cache_preload_ascendc_icache_evictor.vector` | `0x9580` | 32,836 B | ≥32 KiB、128 B 对齐 |
| AscendC | `cache_preload_ascendc_icache_path.vector` | `0x11600` | 4,808 B | ≥4 KiB、128 B 对齐 |

每套实现内部的 evictor/path 地址范围互不重叠。每个 ICache 样本先完整执行该实现的
evictor，再从同一个 current-PC path 进入以下三种动态路径：

- `icache-cold`：不发起 preload；
- `icache-current-pc`：调用各自 current-PC preload API 后进入 noinline gap；
- `icache-wait`：发起后先轮询 idle，再调用同一 gap。

三条路径最终汇合到同一份 1,024 条 volatile NOP 指令区，而不是各自复制一份 target。host 同时复算：

- evictor checksum；
- gap checksum；
- target checksum；
- preload immediate/final status；
- mode、target 和 gap-round echo。

### 4.3 计时边界

两套实现的 `SYS_CNT` 读取和被测值都放在同一个 inline asm 数据依赖中，防止 O3
把普通 load 或纯 Scalar checksum 移出计时区间。这个 dependency 只约束编译器，
不是 DSB、DCCI 或跨核同步。

所有结果都报告 raw `SYS_CNT` delta。本轮没有把 raw tick 按 1 GHz、1.65 GHz 或其他假设频率换算
成 ns。

## 5. 2026-07-29 A5 实测结果

环境：

| 项目 | 值 |
| --- | --- |
| Git 基线 | `9e97dbba1a0ba4bfdda9d13f3d3ac973776f0b24` |
| CANN | `9.1.0-weekly-20260708` |
| Bisheng/CCEC | clang 15.0.5，构建时间 `2026-07-07T20:35:46+08:00` |
| CCEC arch | `dav-c310-vec` |
| AscendC arch | `--npu-arch=dav-3510`，统计实际执行的 AIV `.vector` 符号 |
| device | `0` |
| 每种模式 | 9 个样本，中位数汇总 |
| 独立 gap | 64 rounds |
| ICache preload | 2 units，即 4 KiB |

同一次最终复测的 CCEC 原始结果：

| 模式 | issue | access/work | total | polls | immediate busy | final busy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dcache-baseline` | 0 | 308 | 1051 | 0 | 0/9 | 0/9 |
| `dcache-preload` | 2 | 5 | 742 | 0 | 0/9 | 0/9 |
| `icache-cold` | 0 | 995 | 1908 | 0 | 0/9 | 0/9 |
| `icache-current-pc` | 2 | 501 | 1451 | 0 | 9/9 | 0/9 |
| `icache-wait` | 2 | 501 | 1454 | 2 | 9/9 | 0/9 |

同一次最终复测的 AscendC 原始结果：

| 模式 | issue | access/work | total | polls | immediate busy | final busy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dcache-baseline` | 0 | 349 | 1159 | 0 | 0/9 | 0/9 |
| `dcache-preload` | 2 | 5 | 741 | 0 | 0/9 | 0/9 |
| `icache-cold` | 0 | 1162 | 1992 | 0 | 0/9 | 0/9 |
| `icache-current-pc` | 3 | 792 | 1632 | 0 | 9/9 | 0/9 |
| `icache-wait` | 3 | 793 | 1648 | 1 | 9/9 | 0/9 |

两套实现的 kernel launch、D2H 回读、checksum/echo/status 校验和 cleanup 均为
PASS。

### 5.1 可以下的结论

**接口和真机事实：**

- CCEC 三个 intrinsic 与 AscendC 三个公开 API 在当前 A5 编译栈均可编译、可运行；
- 两套实现发起 ICache preload 后立即观察到 busy，64-round gap 结束时均已 idle；
- baseline/preload 使用同一 gap，ICache 三种模式使用同一物理目标指令区；
- DCache 普通读值和所有 ICache checksum 都保持正确；
- AscendC 构建会同时产生不同 helper 变体，ELF 门禁检查的是本用例真正执行的
  `.vector` 符号。

**本探针内的性能观察：**

- CCEC DCache access 为 `308 -> 5`，完整区间为 `1051 -> 742` raw ticks；
- AscendC DCache access 为 `349 -> 5`，完整区间为 `1159 -> 741` raw ticks；
- CCEC ICache work 为 `995 -> 501`，完整区间为 `1908 -> 1451` raw ticks；
- AscendC ICache work 为 `1162 -> 792`，完整区间为 `1992 -> 1632` raw ticks；
- wait 相对 async 没有降低 target work；完整区间在 CCEC 中多 `3` raw ticks，
  在 AscendC 中多 `16` raw ticks。

两套实现的变化方向一致，支持“提前发起，并让独立工作与预取重叠”的使用方向；
本轮数据不支持 preload 后立即等待。

CCEC 与 AscendC 的绝对 tick 不应彼此直接做 API 成本归因。两者的 wrapper、kernel
入口和最终 `.text` 地址/大小不同，ICache 冷态本身也对代码布局敏感。这里的对等关系是
测试模型、语义校验和 A/B 方向对等，不是要求两个二进制得到相同 tick。

### 5.2 不能下的结论

- 不能把任一 microprobe 的下降幅度外推成 PA Submit 的固定收益；
- 不能假设每次 preload 都会执行；官方明确允许 DCache hint 在拥塞时按 NOP 处理；
- 不能把 preload 当成数据一致性、发布、内存顺序或跨核同步；
- 不能把 raw tick 未经计数器频率校准直接换成时间；
- 不能把 CCEC/AscendC 绝对值之差只归因于 API wrapper；
- 不能只看 ICache miss 降低而忽略新增指令和 `.text` 布局变化。

## 6. Preload 不能替代什么

| 业务要求 | 应使用的机制 | Preload 不满足的原因 |
| --- | --- | --- |
| 让其他核看见最新 GM 写入 | 已定义的 DCCI/发布协议 | preload 不 clean 写回，也不建立 happens-before |
| 等待前序流水完成 | pipe flag、barrier 或既有同步协议 | preload 不是完成屏障 |
| 跨核启动/结束汇聚 | `SyncAll` 或 atomic barrier | preload 没有参与者和 release 语义 |
| 保证目标一定进 cache | 普通 load/取指的正确性路径 | preload 是可丢弃的性能 hint |
| 修复错误的代码布局 | `.text`、分支与热点工作集优化 | preload 只能覆盖有限地址窗口 |

尤其不能把 `dc_preload` 混入 tensormap/symbol 发布协议来替代 DCCI，也不能由
`get_icache_prl_st()==0` 或 `GetICachePreloadStatus()==0` 推导任何 GM 数据已经跨核可见。

## 7. 接入业务的验收标准

只在同时满足以下条件时建议保留业务改动：

1. PMU、泳道或稳定 A/B 已确认目标区间存在 ICache miss 或 GM Scalar 冷读；
2. DCache 地址能提前确定，或 ICache 调用点确实位于即将执行的顺序热点代码前；
3. 发起后存在真实独立工作，不是 preload 后立即 load/立即轮询；
4. 删除 preload 后业务仍完全正确；
5. 同输入、同环境、同构建口径下，Submit/算子端到端性能稳定改善；
6. 同时检查 Scalar busy、I-cache miss、`.text` 大小/布局和其他阶段是否回退；
7. 默认保留 raw `SYS_CNT`，只有频率已经校准时才换算时间；
8. 若收益处于正常波动，或 miss 降低但端到端时间不降，则删除该 hint。

ICache A/B 要特别检查 `.text` 扰动。新增 preload、状态读取或诊断打点都会改变指令数和分支地址；
不能用还包含其他代码差异的两个 ELF，把全部变化归因给 preload。

## 8. 查证来源

本次本地 CANN 9.1 weekly 查证路径如下。`impl/` 是内部实现，只用于核对当前版本下沉关系，业务代码
不应直接 include：

```text
tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_aicore_functions.h
tools/bisheng_compiler/lib/clang/15.0.5/include/cce_aicore_intrinsics.h
compiler/asc/impl/basic_api/dav_3510/kernel_operator_cache_impl.h
compiler/asc/impl/basic_api/dav_3510/kernel_operator_list_tensor_impl.h
x86_64-linux/asc/include/c_api/cache_ctrl/cache_ctrl.h
opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/flash_attn/flash_attn.cpp
opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/mat_mul_v3/arch35/
```

其中 CANN OPP 的 `flash_attn.cpp` 已有 raw `dc_preload` 的生产实现示例；
`mat_mul_v3/arch35/` 已有 AscendC `ICachePreLoad(2)` 的存量用例；Bisheng 头文件则
直接给出 `icache_preload(len) -> preload(get_pc(), len)` 的 CCEC 关系。
