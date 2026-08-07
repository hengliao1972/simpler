# 2026-08-06 SIMT Cross-Core 泳道图配置与性能索引

## 先看结论

本目录现有 24 份 JSON 都是真实 A5、B256（256 batch、1280 个构建
task、1024 个执行 task）的 Direct-GM 泳道图。文件名中的
`traceoff_Nus` 是对应生产变体的独立 trace-off ACL kernel-event
中位数；JSON 内的 `device_span_ns` 和 `SIMT VF build` 则是开启逐
atomic/DCCI 埋点后的本次泳道图时间。两种时间不能混用。

从配置与性能的对应关系看：

- 早期严格全局 insert 链下，单 builder 从 64 warp 收缩并扩展到
  多 builder 后，trace-off 从约 **14.976 ms** 降到 B4/W16 的
  **2.068 ms**；B5～B8 继续增加 builder 已基本不再收益，说明当时
  的全局串行 insert 链限制了扩展。
- 只看双 builder 的 warp 扫描，W16 最好（**3.637 ms**）；W8/W12
  leader 不足，W32/W64 又引入过多 SIMT 并发压力。
- 后续先用 producer-base 复用和 descriptor 单次解码将 B4/W16 降到
  **1.622 ms**，再将 writer 提交改为按 symbol 的精确前驱，并扩展为
  B16/W4，当时配对的 trace-off 中位数为 **0.488 ms**，最终源码
  复测为 **0.495 ms**。
- `B5/W16 sparse metadata writer` 的 **0.710 ms** 依赖 PA 三个
  accumulator 同步推进，是已淘汰的非泛化候选，不得当作当前
  协议或通用调度器的性能。
- B32/W4 将软件搜索上限扩到 32 个 builder，并加入两个
  `get_sys_cnt()` 的轻量 trace-off 构建包络。独立 trace-off 三轮得到总
  kernel 中位 **436.673 us**、构建包络中位 **348.332 us**；因此总
  kernel 已进入约 0.42～0.44 ms 稳定区间，但 **0.3 ms 构建目标没有达到**。
  对应图内 trace-on device span 为 **481.350 us**、`SIMT VF build` 显示
  包络为 **404.694 us**，只能解释该图自身，不能替代 trace-off 口径。

## 配置字段怎么看

- `B×W`：`B` 是专职 builder AIV 数，`W` 是每个 builder AIV 启动的
  SIMT warp 数。例如 `16×4` 表示 16 个 builder AIV，每个 AIV 4 warp。
- 每个 warp 固定 32 thread，但当前构建代码只让 lane0 工作。因此：
  `每 AIV thread = W×32`，`总活跃 leader = B×W`。
- A5 共有 64 个 AIV。builder 完全不执行 task，所以
  `AIV executor = 64-B`；32 个 AIC 始终全部作为 executor。
- `trace-off median` 是端到端 mixed kernel 性能的主口径。
- `trace-on device span` 是 JSON 内 Scalar 全局时域的泳道图范围。
- `trace-on VF 包络` 是所有 `SIMT VF build` 事件的
  `min(ts) → max(ts+dur)`。v4/v5 将各 builder 的 SIMT `CLOCK64` 仿射到各自
  Scalar VF 包络，只用于看图，不是 trace-off 下的精确物理耗时。

## 全部泳道图对照表

`thread/AIV` 一列的格式是“每个 builder AIV / 全部 builder AIV 合计”。

| ID | 阶段/用途 | B×W | thread/AIV | 总活跃 lane0 | AIV executor | trace-off median/us | trace-on device/us | trace-on VF 包络/us | 口径 |
| -- | --------- | ---: | ---------: | -------------: | -----------: | ------------------: | -----------------: | -------------------: | ---- |
| A0 | 最初单 builder、全局严格 insert 链 | 1×64 | 2048 / 2048 | 64 | 63 | 14976.292 | 21838.877 | 21813.620 | 历史基线，v2 |
| A1 | 双 builder warp 扫描 | 2×64 | 2048 / 4096 | 128 | 62 | 7443.780 | 17208.114 | 17174.814 | 拓扑扫描，v4 |
| A2 | 双 builder warp 扫描 | 2×32 | 1024 / 2048 | 64 | 62 | 3944.306 | 6562.730 | 6524.209 | 拓扑扫描，v4 |
| A3 | 双 builder warp 扫描 | 2×16 | 512 / 1024 | 32 | 62 | **3636.886** | 5004.892 | 4963.811 | 双 builder 最优，v4 |
| A4 | 双 builder warp 扫描 | 2×12 | 384 / 768 | 24 | 62 | 4488.931 | 5979.972 | 5951.138 | 拓扑扫描，v4 |
| A5 | 双 builder warp 扫描 | 2×8 | 256 / 512 | 16 | 62 | 4109.439 | 6079.109 | 6050.192 | 拓扑扫描，v4 |
| B1 | 固定 W16，builder 数扫描 | 1×16 | 512 / 512 | 16 | 63 | 6871.043 | 9226.084 | 9193.622 | 全局链，v4 |
| B2 | 固定 W16，builder 数扫描 | 2×16 | 512 / 1024 | 32 | 62 | 3747.599 | 4985.809 | 4946.329 | 全局链，v4 |
| B3 | 固定 W16，builder 数扫描 | 3×16 | 512 / 1536 | 48 | 61 | 2435.067 | 4969.272 | 4928.400 | 全局链，v4 |
| B4 | 固定 W16，builder 数扫描 | 4×16 | 512 / 2048 | 64 | 60 | 2076.408 | 4980.262 | 4939.630 | 扫描轮，v4 |
| B5 | 固定 W16，builder 数扫描 | 5×16 | 512 / 2560 | 80 | 59 | 2082.290 | 4994.315 | 4952.828 | 全局链，v4 |
| B6 | 固定 W16，builder 数扫描 | 6×16 | 512 / 3072 | 96 | 58 | 2117.766 | 4973.567 | 4932.403 | 全局链，v4 |
| B7 | 固定 W16，builder 数扫描 | 7×16 | 512 / 3584 | 112 | 57 | 2151.302 | 4980.977 | 4943.001 | 全局链，v4 |
| B8 | 固定 W16，builder 数扫描 | 8×16 | 512 / 4096 | 128 | 56 | 2099.708 | 4952.476 | 4911.417 | 全局链，v4 |
| C1 | 固定 B4，warp 扫描 | 4×8 | 256 / 1024 | 32 | 60 | 2239.565 | 3035.473 | 2991.825 | 全局链，v4 |
| C2 | 固定 B4，warp 扫描 | 4×12 | 384 / 1536 | 48 | 60 | 2230.759 | 4681.034 | 4633.122 | 全局链，v4 |
| C3 | 固定 B4，warp 扫描最终复测 | 4×16 | 512 / 2048 | 64 | 60 | **2067.660** | 4927.921 | 4886.124 | 当时最优，v4 |
| C4 | 固定 B4，warp 扫描 | 4×24 | 768 / 3072 | 96 | 60 | 2191.867 | 5903.263 | 5863.166 | 全局链，v4 |
| D1 | 复用相邻 producer base | 4×16 | 512 / 2048 | 64 | 60 | 1809.952 | 5118.619 | 5080.486 | 通用局部优化，v4 |
| D2 | descriptor 单次解码后写 16 word | 4×16 | 512 / 2048 | 64 | 60 | **1622.330** | 5220.165 | 5178.822 | 通用局部优化，v4 |
| E1 | 单代表地址 sparse writer | 5×16 | 512 / 2560 | 80 | 59 | 709.769 | 1068.623 | 1036.895 | **已淘汰、非泛化**，v5 |
| E2 | 完整通用 writer-intent：3 load + 3 CAS | 6×4 | 128 / 768 | 24 | 58 | 1233.298 | 1669.084 | 1639.980 | 泛化正确基线，v5 |
| E3 | 按 symbol 精确前驱：0 load + 3 CAS | 16×4 | 128 / 2048 | 64 | 48 | **487.891** | **630.996** | **565.319** | 目录内最佳泛化图，v5；最终源码复测 494.528 us |
| E4 | trace-off 构建包络、builder 扩至 32 | 32×4 | 128 / 4096 | 128 | 32 | **436.673** | **481.350** | **404.694** | 同一通用协议，v5；build-envelope 中位 348.332 us，未达 0.3 ms |

## 文件映射

| ID | 泳道图文件 |
| -- | ---------- |
| A0 | [`gm_g0_b256_atomic_dcci_swimlane.json`](gm_g0_b256_atomic_dcci_swimlane.json) |
| A1 | [`gm_g1_b256_warp64_traceoff_7444us_atomic_dcci_per_builder_clock_swimlane.json`](gm_g1_b256_warp64_traceoff_7444us_atomic_dcci_per_builder_clock_swimlane.json) |
| A2 | [`gm_g1_b256_warp32_traceoff_3944us_atomic_dcci_per_builder_clock_swimlane.json`](gm_g1_b256_warp32_traceoff_3944us_atomic_dcci_per_builder_clock_swimlane.json) |
| A3 | [`gm_g1_b256_warp16_traceoff_3637us_atomic_dcci_per_builder_clock_swimlane.json`](gm_g1_b256_warp16_traceoff_3637us_atomic_dcci_per_builder_clock_swimlane.json) |
| A4 | [`gm_g1_b256_warp12_traceoff_4489us_atomic_dcci_per_builder_clock_swimlane.json`](gm_g1_b256_warp12_traceoff_4489us_atomic_dcci_per_builder_clock_swimlane.json) |
| A5 | [`gm_g1_b256_warp8_traceoff_4109us_atomic_dcci_per_builder_clock_swimlane.json`](gm_g1_b256_warp8_traceoff_4109us_atomic_dcci_per_builder_clock_swimlane.json) |
| B1 | [`gm_b1_b256_warp16_traceoff_6871us_atomic_dcci_per_builder_clock_swimlane.json`](gm_b1_b256_warp16_traceoff_6871us_atomic_dcci_per_builder_clock_swimlane.json) |
| B2 | [`gm_b2_b256_warp16_traceoff_3748us_atomic_dcci_per_builder_clock_swimlane.json`](gm_b2_b256_warp16_traceoff_3748us_atomic_dcci_per_builder_clock_swimlane.json) |
| B3 | [`gm_b3_b256_warp16_traceoff_2435us_atomic_dcci_per_builder_clock_swimlane.json`](gm_b3_b256_warp16_traceoff_2435us_atomic_dcci_per_builder_clock_swimlane.json) |
| B4 | [`gm_b4_b256_warp16_traceoff_2076us_atomic_dcci_per_builder_clock_swimlane.json`](gm_b4_b256_warp16_traceoff_2076us_atomic_dcci_per_builder_clock_swimlane.json) |
| B5 | [`gm_b5_b256_warp16_traceoff_2082us_atomic_dcci_per_builder_clock_swimlane.json`](gm_b5_b256_warp16_traceoff_2082us_atomic_dcci_per_builder_clock_swimlane.json) |
| B6 | [`gm_b6_b256_warp16_traceoff_2118us_atomic_dcci_per_builder_clock_swimlane.json`](gm_b6_b256_warp16_traceoff_2118us_atomic_dcci_per_builder_clock_swimlane.json) |
| B7 | [`gm_b7_b256_warp16_traceoff_2151us_atomic_dcci_per_builder_clock_swimlane.json`](gm_b7_b256_warp16_traceoff_2151us_atomic_dcci_per_builder_clock_swimlane.json) |
| B8 | [`gm_b8_b256_warp16_traceoff_2100us_atomic_dcci_per_builder_clock_swimlane.json`](gm_b8_b256_warp16_traceoff_2100us_atomic_dcci_per_builder_clock_swimlane.json) |
| C1 | [`gm_b4_b256_warp8_traceoff_2240us_atomic_dcci_per_builder_clock_swimlane.json`](gm_b4_b256_warp8_traceoff_2240us_atomic_dcci_per_builder_clock_swimlane.json) |
| C2 | [`gm_b4_b256_warp12_traceoff_2231us_atomic_dcci_per_builder_clock_swimlane.json`](gm_b4_b256_warp12_traceoff_2231us_atomic_dcci_per_builder_clock_swimlane.json) |
| C3 | [`gm_b4_b256_warp16_traceoff_2068us_bounded_trace_capacity_atomic_dcci_per_builder_clock_swimlane.json`](gm_b4_b256_warp16_traceoff_2068us_bounded_trace_capacity_atomic_dcci_per_builder_clock_swimlane.json) |
| C4 | [`gm_b4_b256_warp24_traceoff_2192us_atomic_dcci_per_builder_clock_swimlane.json`](gm_b4_b256_warp24_traceoff_2192us_atomic_dcci_per_builder_clock_swimlane.json) |
| D1 | [`gm_b4_b256_warp16_traceoff_1810us_producer_base_reuse_atomic_dcci_per_builder_clock_swimlane.json`](gm_b4_b256_warp16_traceoff_1810us_producer_base_reuse_atomic_dcci_per_builder_clock_swimlane.json) |
| D2 | [`gm_b4_b256_warp16_traceoff_1622us_single_decode_descriptor_atomic_dcci_per_builder_clock_swimlane.json`](gm_b4_b256_warp16_traceoff_1622us_single_decode_descriptor_atomic_dcci_per_builder_clock_swimlane.json) |
| E1 | [`gm_b5_b256_warp16_traceoff_710us_sparse_metadata_writer_atomic_dcci_per_builder_clock_swimlane.json`](gm_b5_b256_warp16_traceoff_710us_sparse_metadata_writer_atomic_dcci_per_builder_clock_swimlane.json) |
| E2 | [`gm_b6_b256_warp4_traceoff_1233us_generic_writer_intent_atomic_dcci_per_builder_clock_swimlane.json`](gm_b6_b256_warp4_traceoff_1233us_generic_writer_intent_atomic_dcci_per_builder_clock_swimlane.json) |
| E3 | [`gm_b16_b256_warp4_traceoff_488us_per_symbol_writer_expected_cas_atomic_dcci_per_builder_clock_swimlane.json`](gm_b16_b256_warp4_traceoff_488us_per_symbol_writer_expected_cas_atomic_dcci_per_builder_clock_swimlane.json) |
| E4 | [`gm_b32_b256_warp4_traceoff_437us_build348us_atomic_dcci_per_builder_clock_swimlane.json`](gm_b32_b256_warp4_traceoff_437us_build348us_atomic_dcci_per_builder_clock_swimlane.json) |

## 比较时的限制

1. 只有同一阶段、同一协议的 B/W 扫描才能把差异主要归因于拓扑。
   A、B、C 组内可以做这种对比；D/E 组同时修改了构建或 writer
   协议，不能把全部收益归因于 builder 数。
2. 所有数据来自用户授权的 unlocked device0，其他 session 可能产生噪声。
   中位数比单次样本更稳健，但不声称为独占设备峰值。
3. v2/v4/v5 的记录密度和协议不同，trace-on 时间只说明对应 JSON
   本身的显示范围，不能跨版本当作生产性能 A/B。
4. 表中精确 trace-off 中位数来自
   [`simt调度实现过程.md`](../../simt调度实现过程.md) 第 15～22 节；
   trace-on 数值直接由本目录 JSON 顶层字段和 `SIMT VF build` 事件计算。
