# A5 CCEC/AscendC ICache/DCache Preload 用法与真机验证

## 1. 当前交付边界

本目录现在提供三套可独立构建和运行的用例：

- CCEC 用例直接调用编译器 intrinsic，不包含 `kernel_operator.h`；
- AscendC 的被测 preload 调用使用公开 `kernel_operator.h` API；冷态构造、发布边界和
  校验还显式使用 `dcci`、`dsb` 与 bypass load/store；
- 上述两者复用同一套 mode、host/device ABI、输入、checksum oracle、冷态构造和
  raw `SYS_CNT` 统计口径，以便区分“API 用法差异”和“测试模型差异”；
- 第三套是纯 CCEC 的 `1:2` mixed 持续写探针，用与 PA 泳道相同的 32 B
  七字段 record、独占 cacheline 和最终逐行发布口径，专门回答单条 store
  microprobe 没有覆盖的连续写问题。

用户口语中的 `icache_pretch`、`dcache_pretch` 不是当前 CANN 头文件中的符号。正确检索词是：

| 目的 | CCEC 接口 | AscendC 接口 | 说明 |
| --- | --- | --- | --- |
| 从当前 PC 预取后续指令 | `icache_preload(len)` | `ICachePreLoad(len)` | `len` 在 Atlas 350 上以 2 KiB 为单位 |
| 查询 ICache preload 状态 | `get_icache_prl_st()` | `GetICachePreloadStatus()` | `0=idle`，`1=busy` |
| 从 GM 预取数据到 DCache | `dc_preload(base, byteOffset)` | `DataCachePreload(tensor, byteOffset)` | offset 单位为 byte |
| 指定 ICache 地址的底层接口 | `preload(addr, len)` | 无需在业务中直接使用 | 本探针只测试 current-PC 接口 |

本次 A5 真机验证观察到：

- CCEC DCache 普通 GM load 中位数为 `316 -> 5`，AscendC 为 `295 -> 5` raw ticks；
- store-only 场景只计到普通 store 被接受：CCEC、AscendC 均为 `2 -> 2`
  raw ticks，完整区间没有收益；
- 新增的持续顺序写探针复刻泳道记录的物理形态：96 核各写 120 KiB、每条
  32 B、每 cacheline 两条记录；提前 16 条 cacheline 的 `dc_preload`
  使最慢核 store 发射窗口下降 `40.436%`，包含最终逐行
  `CACHELINE_OUT + DSB` 的总窗口下降 `30.460%`；
- publish-to-GM 场景包含 `CACHELINE_OUT + DSB`：CCEC 的 `store->GM` 为
  `512 -> 206`，AscendC 为 `343 -> 121` raw ticks；
- CCEC ICache 同一物理指令区中位数为 `975 -> 484`，AscendC 为
  `1085 -> 743` raw ticks；
- 两套实现的两个 ICache preload mode 都在发起后立即得到 `9/9 busy`，独立 gap
  之后均得到 `9/9 idle`；
- 立即轮询等待没有降低目标工作区间；两次有效运行中完整区间的微小差异方向并不
  一致，不能解释为 wait 收益。

因此当前证据必须按场景表述：单条 cold-line store 被 Scalar/store queue 接受的
窗口没有观察到收益；这不能外推为持续写也无收益。持续顺序写超过每核 DCache
容量后，提前预取未来 cacheline 可以减少 store-side 停顿，但没有降低最终
`DCCI + DSB` 发布段。这些仍是 microprobe 的真机观察，不是 PA Submit 的固定收益
承诺；真实泳道路径还需要带编译开关的端到端 A/B。

## 2. 文件与运行方式

| 文件 | 用途 |
| --- | --- |
| `ccec/cache_preload.cpp` | 纯 CCEC AIV kernel，直接调用 preload intrinsic |
| `cache_preload_shared.h` | CCEC/AscendC/host 共用 ABI、模式和 checksum oracle |
| `ccec/cache_preload_host.cpp` | 真机 launcher、逐样本校验和中位数汇总 |
| `ccec/run_cache_preload.sh` | CCEC 独立构建、最终 ELF 门禁和定向运行 |
| `ascendc/cache_preload_probe.asc` | AscendC AIV kernel、host launcher 和校验 |
| `ascendc/run_cache_preload.sh` | AscendC 独立构建、嵌入 AICore ELF 门禁和定向运行 |
| `trace_write_preload_shared.h` | 持续泳道写探针的 32 B record、控制块和结果 ABI |
| `ccec/trace_write_preload.cpp` | 1:2 mixed CCEC kernel；复刻七字段顺序写与最终逐行发布 |
| `ccec/trace_write_preload_host.cpp` | 96 核/单核策略矩阵、payload 全量回读和容量拐点校验 |
| `ccec/run_trace_write_preload.sh` | 持续泳道写探针的独立构建、mixed ELF 门禁和运行入口 |

从仓库根目录运行：

```bash
tests/atomic_probe/ccec/run_cache_preload.sh
tests/atomic_probe/ascendc/run_cache_preload.sh
tests/atomic_probe/ccec/run_trace_write_preload.sh
```

三套脚本都可以拆开：

```bash
tests/atomic_probe/ccec/run_cache_preload.sh build
tests/atomic_probe/ccec/run_cache_preload.sh run

tests/atomic_probe/ascendc/run_cache_preload.sh build
tests/atomic_probe/ascendc/run_cache_preload.sh run

tests/atomic_probe/ccec/run_trace_write_preload.sh build
tests/atomic_probe/ccec/run_trace_write_preload.sh run
```

三个脚本都是独立入口，没有修改 `ccec/run_all.sh`，也不会误跑目录内其他探针。
AscendC runner 会从可执行文件提取 `.aicore_binary`，再对实际执行的 `.vector`
符号做尺寸、对齐和地址不重叠检查，不能用 host ELF 的表面尺寸替代该门禁。持续写
runner 则检查最终 ELF 同时存在 AIC/AIV 入口及其 `1:2` mixed metadata，32 个物理
block 实际形成 32 AIC + 64 AIV。

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

官方契约只说明“把 GM 数据预加载到 DCache”，没有承诺后续 store 或写回一定加速。
因此不能在每个短小 load/store 前机械调用。只有“地址能提前确定、后面有独立工作、
目标很可能被消费”的点才值得尝试；写场景必须再区分 store-only 和发布到 GM。

当前 CANN ListTensor 内部实现有“从 miss offset 预取 256 Bytes”的注释，但公开接口文档没有把
256 B 定义为跨版本契约。业务正确性不能依赖固定预取范围。

官方文档：
[DataCachePreload](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0176.html)。

### 3.2 普通 store 与发布到 GM

普通 scalar store、把 dirty DCache line 发布到 GM 是两个不同业务边界：

```cpp
*volatile_target = value;
const uint64_t store_only_end = ReadSysCnt();

// 只有业务要求本次把 dirty line 刷到 GM 时，才把下面两步纳入关键路径。
dcci(target, SINGLE_CACHE_LINE, CACHELINE_OUT);
dsb(DSB_ALL);
const uint64_t publish_to_gm_end = ReadSysCnt();
```

当前探针据此提供四个独立 mode：

| mode | preload | 被测终点 |
| --- | --- | --- |
| `dstore-only-baseline` | 无 | 普通 volatile store 后 |
| `dstore-only-preload` | 有 | 普通 volatile store 后 |
| `dpublish-gm-baseline` | 无 | store 后的 `CACHELINE_OUT + DSB` 完成后 |
| `dpublish-gm-preload` | 有 | store 后的 `CACHELINE_OUT + DSB` 完成后 |

store-only 不插 `PipeBarrier<PIPE_S>()`。官方明确说明 Scalar 流水间顺序由硬件自动
保证，调用 `PipeBarrier<PIPE_S>()` 会触发硬件错误。store 后的 `SYS_CNT` 本身是后续
Scalar 指令，因此这里测的是 store 在 Scalar 顺序中的指令窗口，不是 dirty line 已经
写回 GM。若业务语义是 `store + DSB`、但不做 DCCI，那是第三种边界，当前两场景探针
没有把它混入 store-only。官方文档：
[PipeBarrier](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendcopapi/atlasascendc_api_07_0271.html)。

store-only mode 为了校验与清理，也会在计时终点之后执行本核普通回读、
`CACHELINE_OUT + DSB`、bypass read 和 host D2H。这些动作不计入 store-only 的
`access/work` 或 `total`，因此后续 host 能看到新值不能被误解成“store-only 被测区间
已经完成 GM 发布”。

官方 `CACHELINE_OUT` 契约是让 Data Cache 与 Global Memory 保持一致；preload 本身不
clean dirty line，也不提供发布或同步语义。官方文档：
[DataCacheCleanAndInvalid](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0177.html)。

### 3.3 `icache_preload`

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
4,804 B，AscendC 实际执行的 `.vector` 被测函数为 4,820 B。

当前 CANN OPP 的 arch35 存量代码也采用相同模式。例如
`mat_mul_v3/arch35/mat_mul_stream_k_kernel.h` 在 AIV 分支调用 `ICachePreLoad(2)` 后直接进入后续
处理，没有立即轮询；这与“发起后继续可重叠工作”的用法一致。

官方文档：
[ICachePreLoad](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0276.html)。

### 3.4 `get_icache_prl_st`

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

### 4.2 DCache 写对照

四个写 mode 在每次 launch 前都由 host 把目标 word 恢复成同一初值。kernel 随后在
计时区间外对目标 64 B line 执行 DCCI 和 DSB，确保 baseline/preload 都从 clean、
cold line 开始。目标 word 位于 cacheline 首地址，不存在跨 line 的 DCCI 歧义。

两类场景复用相同地址、写入值和 64-round gap：

1. `dstore-only-*` 的 `access/work` 只包围普通 volatile store，`total` 从可选
   preload 发起前开始、在 store 后结束；
2. `dpublish-gm-*` 的 `access/work` 仍只包围 store，`store->GM` 从 store 前开始、
   在 `CACHELINE_OUT + DSB` 后结束，`total` 也在该发布序列后结束；
3. preload mode 都在 gap 前发起 `dc_preload`/`DataCachePreload`，baseline 不发起；
4. 写入值依赖 gap checksum，目标地址也通过 opaque dependency 依赖 gap，防止 O3
   把 store 提前；
5. 计时结束后，本核普通回读、bypass read 与 host D2H 必须三者都等于预期新值。

第 5 点只做正确性门禁。尤其对 store-only mode，bypass/host 校验发生在计时结束后的
清理性 `CACHELINE_OUT + DSB` 之后，不属于被测业务路径。

### 4.3 ICache 冷态与同一物理代码区

为了避免“源码写了很多 NOP，但最终 ELF 并没有形成足够指令 footprint”的假测试，构建脚本直接检查
链接后实际 AIV ELF：

| 实现 | 符号 | 最终地址 | 最终大小 | 门禁 |
| --- | --- | ---: | ---: | --- |
| CCEC | `cache_preload_icache_evictor` | `0x100` | 32,836 B | ≥32 KiB、128 B 对齐 |
| CCEC | `cache_preload_icache_path` | `0x8180` | 4,804 B | ≥4 KiB、128 B 对齐 |
| AscendC | `cache_preload_ascendc_icache_evictor.vector` | `0x9580` | 32,836 B | ≥32 KiB、128 B 对齐 |
| AscendC | `cache_preload_ascendc_icache_path.vector` | `0x11600` | 4,820 B | ≥4 KiB、128 B 对齐 |

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

### 4.4 计时边界

两套实现的 `SYS_CNT` 读取和被测值都放在同一个 inline asm 数据依赖中，防止 O3
把普通 load、store 或纯 Scalar checksum 移出计时区间。这个 dependency 只约束
编译器，不是 DSB、DCCI 或跨核同步。

所有结果都报告 raw `SYS_CNT` delta。本轮没有把 raw tick 按 1 GHz、1.65 GHz 或其他假设频率换算
成 ns。

### 4.5 持续泳道写与 DCache 容量对照

单条 store 的 `2 -> 2` 只能回答“一个 ordinary store 何时能继续执行后续 Scalar
指令”，不能覆盖以下持续写行为：

- 连续触及新的 cold cacheline；
- dirty line 超出 DCache 容量后逐步被替换；
- store queue、DCache miss 和写回流量形成反压；
- 最后对完整有效区间逐行执行 `DCCI(CACHELINE_OUT)`，再以一次 DSB 收口。

因此持续写探针不复用单 word mode，而是建立与当前 PA trace 相同的物理 record：

```cpp
struct alignas(32) TraceRecord {
    uint64_t start_cycle;
    uint64_t end_cycle;
    int32_t task_id;
    int32_t function_id;
    uint32_t flags;
    uint16_t phase;
    uint16_t auxiliary;
};
static_assert(sizeof(TraceRecord) == 32);
```

每条 record 分别执行 7 个 ordinary scalar GM 字段 store；两个连续 record 恰好填满
一条 64 B cacheline。每核拥有独立且 64 B 对齐的 128 KiB 区间，不存在跨核同 line
竞争。被测窗口分成：

1. `issue`：从第一条 record 写入前到最后一条 ordinary store 之后；
2. `flush`：对全部有效 cacheline 逐行 `CACHELINE_OUT`，再执行 DSB；
3. `total = issue + flush`。

preload 策略写成 `dN-cM`：`dN` 表示预取当前写位置之后 N 条 cacheline，`cM`
表示每 M 条 cacheline 发起一次 hint。例如 `d16-c1` 是每写一条 line 时，为未来
第 16 条 line 发起一次 `dc_preload`。预取指令本身包含在 `issue` 窗口内，不能靠
移出计时区间制造收益。每个策略的首轮还会把所有有效 record 从 GM 全量回读，逐字段
验证预期值；每次 launch 都校验 96 个 mixed worker 的结果或未参与者零值。

同一探针还提供随机依赖 pointer-cycle 容量扫描。每条 cacheline 只保存下一条 line
编号，下一次 load 地址依赖上一次结果，从而避免把顺序硬件预取误认为 DCache 容量。
每个 working set 先 cold 遍历一次，再立即重复遍历 8 次；这里只用复用延迟的容量
拐点判断有效 resident set，不把单次 raw tick 直接换算为 ns。

## 5. 2026-07-29 A5 实测结果

环境：

| 项目 | 值 |
| --- | --- |
| Git 基线 | `ad52c018e323511b9f3a9dfe17c95aadbf9ddab9` |
| CANN | `9.1.0-weekly-20260708` |
| Bisheng/CCEC | clang 15.0.5，构建时间 `2026-07-07T20:35:46+08:00` |
| CCEC arch | `dav-c310-vec` |
| AscendC arch | `--npu-arch=dav-3510`，统计实际执行的 AIV `.vector` 符号 |
| device | `0` |
| 模式数 | 9：DCache read 2、store-only 2、publish-to-GM 2、ICache 3 |
| 每种模式 | 9 个样本，中位数汇总 |
| 独立 gap | 64 rounds |
| ICache preload | 2 units，即 4 KiB |

同一次最终复测的 CCEC 原始结果：

| 模式 | issue | access/work | store->GM | total | polls | immediate busy | final busy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dcache-baseline` | 0 | 316 | 0 | 1052 | 0 | 0/9 | 0/9 |
| `dcache-preload` | 3 | 5 | 0 | 742 | 0 | 0/9 | 0/9 |
| `dstore-only-baseline` | 0 | 2 | 0 | 741 | 0 | 0/9 | 0/9 |
| `dstore-only-preload` | 2 | 2 | 0 | 742 | 0 | 0/9 | 0/9 |
| `dpublish-gm-baseline` | 0 | 2 | 512 | 1252 | 0 | 0/9 | 0/9 |
| `dpublish-gm-preload` | 1 | 2 | 206 | 942 | 0 | 0/9 | 0/9 |
| `icache-cold` | 0 | 975 | 0 | 1892 | 0 | 0/9 | 0/9 |
| `icache-current-pc` | 2 | 484 | 0 | 1423 | 0 | 9/9 | 0/9 |
| `icache-wait` | 2 | 484 | 0 | 1439 | 2 | 9/9 | 0/9 |

同一次最终复测的 AscendC 原始结果：

| 模式 | issue | access/work | store->GM | total | polls | immediate busy | final busy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dcache-baseline` | 0 | 295 | 0 | 1147 | 0 | 0/9 | 0/9 |
| `dcache-preload` | 2 | 5 | 0 | 739 | 0 | 0/9 | 0/9 |
| `dstore-only-baseline` | 0 | 2 | 0 | 745 | 0 | 0/9 | 0/9 |
| `dstore-only-preload` | 4 | 2 | 0 | 749 | 0 | 0/9 | 0/9 |
| `dpublish-gm-baseline` | 0 | 2 | 343 | 1083 | 0 | 0/9 | 0/9 |
| `dpublish-gm-preload` | 3 | 2 | 121 | 863 | 0 | 0/9 | 0/9 |
| `icache-cold` | 0 | 1085 | 0 | 1912 | 0 | 0/9 | 0/9 |
| `icache-current-pc` | 3 | 743 | 0 | 1602 | 0 | 9/9 | 0/9 |
| `icache-wait` | 3 | 743 | 0 | 1596 | 1 | 9/9 | 0/9 |

两套实现的 kernel launch、D2H 回读、checksum/echo/status 校验和 cleanup 均为
PASS。

同一逻辑在最终重建前还有一次有效运行。写场景两次中位数如下，用于区分稳定方向和
绝对值波动：

| 实现/指标 | 前一次 baseline -> preload | 最终 baseline -> preload | 判断 |
| --- | ---: | ---: | --- |
| CCEC store-only access | `2 -> 1` | `2 -> 2` | 没有稳定下降 |
| AscendC store-only access | `2 -> 2` | `2 -> 2` | 没有下降 |
| CCEC publish `store->GM` | `319 -> 110` | `512 -> 206` | 两次均明显下降 |
| AscendC publish `store->GM` | `526 -> 213` | `343 -> 121` | 两次均明显下降 |

### 5.1 可以下的结论

**接口和真机事实：**

- CCEC 三个 intrinsic 与 AscendC 三个公开 API 在当前 A5 编译栈均可编译、可运行；
- 两套实现发起 ICache preload 后立即观察到 busy，64-round gap 结束时均已 idle；
- baseline/preload 使用同一 gap，写 mode 在每次 launch 前恢复同一目标初值，
  ICache 三种模式使用同一物理目标指令区；
- DCache 普通读值、写者本核普通回读、清理后的 bypass read、host D2H 和所有
  ICache checksum 都保持正确；
- AscendC 构建会同时产生不同 helper 变体，ELF 门禁检查的是本用例真正执行的
  `.vector` 符号。

**本探针内的性能观察：**

- CCEC DCache load 为 `316 -> 5`，完整区间为 `1052 -> 742` raw ticks；
- AscendC DCache load 为 `295 -> 5`，完整区间为 `1147 -> 739` raw ticks；
- 最终 store-only 的 store 指令窗口两套均为 `2 -> 2`；完整区间分别为
  `741 -> 742`、`745 -> 749`，没有净收益；
- publish-to-GM 的 `store->GM` 在 CCEC 中为 `512 -> 206`、AscendC 中为
  `343 -> 121`；完整区间分别为 `1252 -> 942`、`1083 -> 863`；
- CCEC ICache work 为 `975 -> 484`，完整区间为 `1892 -> 1423` raw ticks；
- AscendC ICache work 为 `1085 -> 743`，完整区间为 `1912 -> 1602` raw ticks；
- wait 相对 async 没有降低 target work。两次 AscendC 完整区间差值分别为
  `+16`、`-6` raw ticks，方向反转；CCEC 分别为 `+53`、`+16`，只能判为小开销或波动，
  不能证明 wait 有收益。

两套单 word 实现的变化方向一致：提前 preload 对普通读和本探针的 publish-to-GM
序列有收益，但对单条 store-only 没有可见净收益。本轮数据也不支持 ICache
preload 后立即等待。这里不能替代 5.3 节对持续顺序写的独立结论。

publish-to-GM 的下降与“预先把 line 带入 DCache，随后普通 store 和 dirty-line clean
不再从 cold line 起步”这一解释相符，但探针没有直接观测内部 write-allocate/store
queue 状态，因此这里只能作为机制推测，不能写成已证硬件原因。

CCEC 与 AscendC 的绝对 tick 不应彼此直接做 API 成本归因。两者的 wrapper、kernel
入口和最终 `.text` 地址/大小不同，ICache 冷态本身也对代码布局敏感。这里的对等关系是
测试模型、语义校验和 A/B 方向对等，不是要求两个二进制得到相同 tick。

### 5.2 不能下的结论

- 不能把任一 microprobe 的下降幅度外推成 PA Submit 的固定收益；
- 不能只用单 word publish-to-GM 的收益证明 store-only、bypass store、atomic、整
  line 写或多次连续写也会受益；持续顺序写的证据来自 5.3 节独立探针；
- 不能假设每次 preload 都会执行；官方明确允许 DCache hint 在拥塞时按 NOP 处理；
- 不能把 preload 当成数据一致性、发布、内存顺序或跨核同步；
- 不能把 raw tick 未经计数器频率校准直接换成时间；
- 不能把 CCEC/AscendC 绝对值之差只归因于 API wrapper；
- 不能只看 ICache miss 降低而忽略新增指令和 `.text` 布局变化。

### 5.3 持续顺序写的补充实验

运行环境和证据身份：

| 项目 | 值 |
| --- | --- |
| 实验日期 | `2026-07-29` |
| 运行时 Git HEAD | `f0903f6f68743606e70bba983923aec4a476f3b6` |
| 分支 | `fdwic-swimlane-exclusive`，跟踪 `origin/fdwic-swimlane-deps` |
| CANN/CCEC | `9.1.0-weekly-20260708` / clang 15.0.5 |
| 最终 mixed kernel SHA256 | `b98c6c912d3533d19705e23e13c9b298a5915a62651f9ec02d19949a4d8a34c6` |
| topology | 32 个物理 block，`1 AIC : 2 AIV`，总计 96 worker |
| 主工作集 | 每核 120 KiB，即 3,840 records / 1,920 cachelines |
| 计时单位 | raw `SYS_CNT`；百分比不依赖频率换算 |
| 设备隔离 | 当前 shell 没有 `task-submit`/`npu-smi`；按已有用户授权在 device 0 未加锁直跑 |

第一轮用 9 个交错样本扫描预取距离和发起密度。下表的 `critical` 是每次 launch
先取 96 核最大值、再跨样本取中位数；它不会把所有核的 duration 相加。

| 策略 | critical issue | critical total | 相对 baseline total |
| --- | ---: | ---: | ---: |
| baseline | 85,903 | 114,491 | 0 |
| `d1-c1` | 89,498 | 117,812 | +2.901% |
| `d2-c1` | 67,650 | 95,821 | -16.307% |
| `d4-c1` | 54,502 | 83,081 | -27.435% |
| `d8-c1` | 53,912 | 82,248 | -28.162% |
| `d16-c1` | 51,097 | 79,561 | **-30.509%** |
| `d4-c4` | 63,771 | 92,264 | -19.414% |
| `d8-c4` | 64,811 | 93,471 | -18.359% |
| `d16-c4` | 67,469 | 95,979 | -16.169% |

`d1-c1` 说明“调用了 preload”不等于会加速：只提前一条 line 时，hint 成本已经进入
关键路径，却没有留下足够的重叠窗口。`d16-c1` 是本次已测试集合中的最优点，不是
所有代码布局、记录密度或硬件负载下的固定参数。

随后将 baseline 与 `d16-c1` 交错运行 21 个确认样本：

| 96 核、每核 120 KiB | baseline | `d16-c1` | 变化 |
| --- | ---: | ---: | ---: |
| critical issue | 85,888 | 51,158 | **-40.436%** |
| 每核 issue 中位数 | 83,552 | 49,589 | -40.649% |
| 每核最终 flush 中位数 | 27,787 | 27,735 | -0.187% |
| critical total | 114,519 | 79,636 | **-30.460%** |
| 设备阶段完整 span | 115,171 | 80,254 | -30.318% |
| host launch + stream sync 中位数 | 132.268 us | 99.988 us | -24.405% |

最终 flush 段基本不变，把收益边界定位在 ordinary store 发射阶段，而不是
`DCCI(CACHELINE_OUT)+DSB`。这与“提前把未来 cold line 带入本核 DCache，持续写时
减少 store-side 等待/反压”的机制解释一致；本探针没有采集 DCache miss 或 store
queue PMU，因此不能把该内部原因写成直接观测事实。

工作集和核类型对照同样保持相同方向：

| 对照 | critical issue 变化 | critical total 变化 |
| --- | ---: | ---: |
| 96 核、每核 96 KiB，13 样本 | `69,141 -> 40,696`，-41.141% | `96,873 -> 68,251`，-29.546% |
| 单 AIC、每核 120 KiB，13 样本 | `84,695 -> 50,163`，-40.772% | `112,232 -> 77,630`，-30.831% |
| 单 AIV、每核 120 KiB，13 样本 | `85,329 -> 50,491`，-40.828% | `113,024 -> 78,114`，-30.887% |

每个策略首个样本都全量回读有效 record 并校验七个字段；所有 launch 的 mixed
topology、结果 echo、GM publication、pointer cycle 和 cleanup 均为 PASS。

随机依赖容量扫描得到：

| working set | AIC reuse ticks/load | AIV reuse ticks/load |
| --- | ---: | ---: |
| 4 KiB | 246.148 | 246.133 |
| 8 KiB | 246.103 | 246.100 |
| 12 KiB | 246.083 | 246.087 |
| 16 KiB | 246.076 | 246.078 |
| 20 KiB | 295.791 | 296.135 |
| 24 KiB | 328.494 | 328.778 |
| 32 KiB | 329.346 | 331.048 |
| 64 KiB | 328.449 | 329.654 |

16 KiB 以内复用延迟稳定，20 KiB 开始退化，24 KiB 后已经接近本探针约
`325-331 ticks/load` 的 cold traversal。因此本机 A5 的 AIC/AIV Scalar DCache 在该
随机依赖访问模型下都呈现约 **16 KiB 有效容量拐点**。这是实测 effective resident
set，不等价于对所有地址映射、关联冲突和未来 SKU 声明精确架构容量。

这组实验修正了“纯写不能 preload”的过度结论：

- 单条 ordinary store 没有收益，只代表其立即接受窗口；
- 大于 DCache 容量的持续、独占、顺序 cold-line 写可以从足够提前的 preload 获益；
- preload 不减少最终发布成本，也不能替代 DCCI/DSB；
- 当前约 30% 是隔离的紧凑连续写模型收益，不能直接写成 PA Submit 收益。真实
  TraceWriter 仍需以编译开关接入相同策略，并用同一业务输入做 level 4 端到端 A/B。

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

写路径还必须先确认真实关键路径：

- 只要求 ordinary store 被本核接受时，应看 `dstore-only-*` 的 `access/work` 和
  `total`；不能拿 publish 数据替它证明收益；
- 要求 dirty line 在本次发布到 GM 时，应看 `dpublish-gm-*` 的 `store->GM` 和
  `total`，且业务仍必须保留既有 DCCI/DSB；
- 单次 store 探针不覆盖连续写把 store queue 压满后的吞吐，也不覆盖 bypass store、
  atomic 或多核同 line 写；连续、独占、顺序 cacheline 写应使用本目录的
  `trace_write_preload` 探针，但它仍不覆盖跨核同 line 竞争。

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
