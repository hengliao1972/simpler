# 2026-08-07 SIMT Cross-Core B/W 对比泳道图

本目录保存真实 A5、Direct-GM、B256、固定 workload `6/28/4/1` 的最终
v13 泳道图。B 固定覆盖 `1/2/4/8/16/32`，每个 B 按关闭埋点后的 11 轮
ACL kernel event 中位数保留 3 种 W，共 **18 份 JSON**。

当前扫描范围内的全局最优是 **B8/W5：926.038 us**。这里的“最优三组”是
本轮已扫描候选中的实测排序，不声称穷举了 W=1..64 的全部整数。

## 一眼看清配置

- `B`：专职 builder AIV 数；这些 AIV 只构建，不执行 task。
- `W`：每个 builder AIV 启动的 SIMT warp 数；每个 warp 只有 lane0 构建。
- 每个 builder 的 SIMT thread 数为 `W * 32`，活跃 lane0 总数为 `B * W`。
- AIC executor 始终为 32 个；AIV executor 为 `64 - B` 个，executor 不参与构建。
- 每个配置均构建 1280 个 descriptor，执行其中 1024 个 QK/SF/PV/UP task。

| B | 保留的 W（按性能排序） | 最优 trace-off median/us | 最优配置的 build median/us |
| -: | ----------------------- | -----------------------: | ----------------------------: |
| 1 | 16 / 14 / 26 | 2946.263 | 2796.115 |
| 2 | 32 / 28 / 8 | 1646.371 | 1480.959 |
| 4 | 30 / 16 / 15 | 1013.554 | 974.698 |
| 8 | 5 / 7 / 6 | **926.038** | 825.004 |
| 16 | 5 / 3 / 4 | 946.850 | 480.852 |
| 32 | 2 / 1 / 5 | 950.701 | 362.501 |

## 完整性能数据

性能主口径是关闭泳道图埋点后的 ACL kernel event。每个入围配置复测 11 轮，
第一轮新鲜初始化样本仍纳入 min/median/avg/max，没有人为丢弃。`trace-on` 三列
来自对应泳道图的两次独立采样中 E2E 更短且 oracle 通过的一份，只用于解释该图，
不能与 `trace-off median` 混为同一性能口径。

| B | 排名 | W | thread/builder | lane0 总数 | AIC/AIV executor | PASS | trace-off min/us | median/us | avg/us | max/us | build median/us | trace-on E2E/us | device span/us | build envelope/us |
| -: | ---: | -: | -------------: | ---------: | ----------------: | ---: | ---------------: | --------: | -----: | -----: | --------------: | --------------: | -------------: | ----------------: |
| 1 | 1 | 16 | 512 | 16 | 32/63 | 11/11 | 2929.671 | **2946.263** | 3011.180 | 3602.412 | 2796.115 | 4475.036 | 3892.349 | 3757.616 |
| 1 | 2 | 14 | 448 | 14 | 32/63 | 11/11 | 3295.669 | **3303.954** | 3363.138 | 3960.403 | 3151.599 | 5182.962 | 4599.121 | 4471.386 |
| 1 | 3 | 26 | 832 | 26 | 32/63 | 11/11 | 3290.661 | **3307.122** | 3369.582 | 3980.906 | 3150.409 | 4696.428 | 4103.995 | 3974.210 |
| 2 | 1 | 32 | 1024 | 64 | 32/62 | 11/11 | 1625.148 | **1646.371** | 1706.564 | 2316.101 | 1480.959 | 2876.449 | 2295.846 | 2159.782 |
| 2 | 2 | 28 | 896 | 56 | 32/62 | 11/11 | 1667.557 | **1697.087** | 1756.286 | 2360.840 | 1541.759 | 2750.224 | 2160.478 | 2028.969 |
| 2 | 3 | 8 | 256 | 16 | 32/62 | 11/11 | 1779.160 | **1784.520** | 1848.105 | 2428.918 | 1621.151 | 3109.412 | 2529.161 | 2389.405 |
| 4 | 1 | 30 | 960 | 120 | 32/60 | 11/11 | 991.515 | **1013.554** | 1079.995 | 1683.234 | 974.698 | 1888.448 | 1297.003 | 1267.593 |
| 4 | 2 | 16 | 512 | 64 | 32/60 | 11/11 | 1014.725 | **1041.676** | 1100.783 | 1706.088 | 814.734 | 1838.159 | 1245.133 | 1081.563 |
| 4 | 3 | 15 | 480 | 60 | 32/60 | 11/11 | 1029.380 | **1047.185** | 1108.004 | 1710.233 | 1009.061 | 2013.422 | 1426.318 | 1395.649 |
| 8 | 1 | 5 | 160 | 40 | 32/56 | 11/11 | 915.465 | **926.038** | 990.479 | 1585.222 | 825.004 | 1758.027 | 1176.586 | 1141.773 |
| 8 | 2 | 7 | 224 | 56 | 32/56 | 11/11 | 922.049 | **945.652** | 1006.657 | 1633.698 | 499.392 | 1650.687 | 1054.750 | 727.845 |
| 8 | 3 | 6 | 192 | 48 | 32/56 | 11/11 | 909.270 | **952.794** | 1004.310 | 1613.375 | 604.546 | 1695.683 | 1106.171 | 898.296 |
| 16 | 1 | 5 | 160 | 80 | 32/48 | 11/11 | 917.048 | **946.850** | 1002.954 | 1567.795 | 480.852 | 1583.572 | 1002.500 | 632.520 |
| 16 | 2 | 3 | 96 | 48 | 32/48 | 11/11 | 920.504 | **947.485** | 1008.123 | 1580.593 | 474.283 | 1616.191 | 1016.171 | 672.426 |
| 16 | 3 | 4 | 128 | 64 | 32/48 | 11/11 | 924.020 | **954.408** | 1019.624 | 1716.267 | 422.751 | 1614.128 | 1026.964 | 556.698 |
| 32 | 1 | 2 | 64 | 64 | 32/32 | 11/11 | 924.433 | **950.701** | 1017.533 | 1620.403 | 362.501 | 1660.314 | 1080.561 | 456.118 |
| 32 | 2 | 1 | 32 | 32 | 32/32 | 11/11 | 954.206 | **988.489** | 1044.029 | 1590.828 | 587.956 | 1642.014 | 1054.233 | 770.554 |
| 32 | 3 | 5 | 160 | 160 | 32/32 | 11/11 | 965.725 | **1006.230** | 1057.051 | 1627.858 | 396.174 | 1647.185 | 1055.719 | 457.129 |

文件名中的 us 均四舍五入到整数，精确值以上表及 JSON 顶层元数据为准。

## 文件索引

| 配置 | 泳道图 |
| ---- | ------ |
| B1/W16 | `gm_b1_b256_warp16_traceoff_2946us_traceon_e2e4475us_device3892us_build3758us_per_block_full_scalar_wait_stage_v13_simt_aic_atomic_dcci_swimlane.json` |
| B1/W14 | `gm_b1_b256_warp14_traceoff_3304us_traceon_e2e5183us_device4599us_build4471us_per_block_full_scalar_wait_stage_v13_simt_aic_atomic_dcci_swimlane.json` |
| B1/W26 | `gm_b1_b256_warp26_traceoff_3307us_traceon_e2e4696us_device4104us_build3974us_per_block_full_scalar_wait_stage_v13_simt_aic_atomic_dcci_swimlane.json` |
| B2/W32 | `gm_b2_b256_warp32_traceoff_1646us_traceon_e2e2876us_device2296us_build2160us_per_block_full_scalar_wait_stage_v13_simt_aic_atomic_dcci_swimlane.json` |
| B2/W28 | `gm_b2_b256_warp28_traceoff_1697us_traceon_e2e2750us_device2160us_build2029us_per_block_full_scalar_wait_stage_v13_simt_aic_atomic_dcci_swimlane.json` |
| B2/W8 | `gm_b2_b256_warp8_traceoff_1785us_traceon_e2e3109us_device2529us_build2389us_per_block_full_scalar_wait_stage_v13_simt_aic_atomic_dcci_swimlane.json` |
| B4/W30 | `gm_b4_b256_warp30_traceoff_1014us_traceon_e2e1888us_device1297us_build1268us_per_block_full_scalar_wait_stage_v13_simt_aic_atomic_dcci_swimlane.json` |
| B4/W16 | `gm_b4_b256_warp16_traceoff_1042us_traceon_e2e1838us_device1245us_build1082us_per_block_full_scalar_wait_stage_v13_simt_aic_atomic_dcci_swimlane.json` |
| B4/W15 | `gm_b4_b256_warp15_traceoff_1047us_traceon_e2e2013us_device1426us_build1396us_per_block_full_scalar_wait_stage_v13_simt_aic_atomic_dcci_swimlane.json` |
| B8/W5 | `gm_b8_b256_warp5_traceoff_926us_traceon_e2e1758us_device1177us_build1142us_per_block_full_scalar_wait_stage_v13_simt_aic_atomic_dcci_swimlane.json` |
| B8/W7 | `gm_b8_b256_warp7_traceoff_946us_traceon_e2e1651us_device1055us_build728us_per_block_full_scalar_wait_stage_v13_simt_aic_atomic_dcci_swimlane.json` |
| B8/W6 | `gm_b8_b256_warp6_traceoff_953us_traceon_e2e1696us_device1106us_build898us_per_block_full_scalar_wait_stage_v13_simt_aic_atomic_dcci_swimlane.json` |
| B16/W5 | `gm_b16_b256_warp5_traceoff_947us_traceon_e2e1584us_device1002us_build633us_per_block_full_scalar_wait_stage_v13_simt_aic_atomic_dcci_swimlane.json` |
| B16/W3 | `gm_b16_b256_warp3_traceoff_947us_traceon_e2e1616us_device1016us_build672us_per_block_full_scalar_wait_stage_v13_simt_aic_atomic_dcci_swimlane.json` |
| B16/W4 | `gm_b16_b256_warp4_traceoff_954us_traceon_e2e1614us_device1027us_build557us_per_block_full_scalar_wait_stage_v13_simt_aic_atomic_dcci_swimlane.json` |
| B32/W2 | `gm_b32_b256_warp2_traceoff_951us_traceon_e2e1660us_device1081us_build456us_per_block_full_scalar_wait_stage_v13_simt_aic_atomic_dcci_swimlane.json` |
| B32/W1 | `gm_b32_b256_warp1_traceoff_988us_traceon_e2e1642us_device1054us_build771us_per_block_full_scalar_wait_stage_v13_simt_aic_atomic_dcci_swimlane.json` |
| B32/W5 | `gm_b32_b256_warp5_traceoff_1006us_traceon_e2e1647us_device1056us_build457us_per_block_full_scalar_wait_stage_v13_simt_aic_atomic_dcci_swimlane.json` |

## 候选范围与稳定性处理

先用 5 轮 trace-off 做宽范围扫描，再把各 B 的近邻候选统一复测 11 轮：

- B1：扫描 W=`4/8/12/14/15/16/17/18/20/24/26/28/30/32/64`；
- B2：扫描 W=`4/7/8/9/12/14/15/16/17/18/20/24/26..33/64`；
- B4：扫描 W=`4/8/12/14/15/16/17/18/20/24/26/28..32/64`；
- B8：扫描 W=`4..10/12..16/18/20/24/28/32/64`；
- B16：扫描 W=`1..10/12/14/16/18`；
- B32：扫描 W=`1..6/8/10/12/14/16/18`。

B32/W3 在初筛中为 0/5，W6、W8 为 4/5；B32/W4 虽曾有 5/5，11 轮复测
出现一次 `fatal-nonzero`，最终为 10/11。因此这些配置均不进入稳定 Top-3，
B32 最终保留 W2/W1/W5。旧 B32/W4 图也不再留在本目录。

B16/W4 与 W6 的 11 轮中位数只差 0.469 us，B1/W14 与 W26 只差 3.168 us，
都属于容易受 unlocked device 噪声影响的近似并列区间。本目录严格按本次中位数
排序选文件，不把这种微小差异解释为架构级确定收益。运行时没有 `task-submit`
锁，日志已明确标记为 unlocked；全部数据都来自同一 device0，但不声称系统级独占。

## 固定 workload 口径

每个 batch 有一个 QK、SF、PV、UP 执行 task。`6/28/4/1` 表示每个 task
内部执行完整 128x128 engine pipeline 的次数：

| task kind | engine workload | repeats |
| --------- | --------------- | ------: |
| QK | Cube matmul | 6 |
| SF | Vector add | 28 |
| PV | Cube matmul | 4 |
| UP | Vector multiply | 1 |

该值与 cross_core_ordinary 正式 real-compute 默认值一致。当前源码没有 workload 命令行
覆盖入口；host 启动日志打印 `workload_repeats=6/28/4/1`，18 份 JSON 顶层也都
保存 `{"qk":6,"sf":28,"pv":4,"up":1}`。

## v13 展示合同

- 图顶端唯一的 `kernel.end_to_end[ACL event]` 覆盖 launch 前到 final drain、
  kernel 返回后的完整 E2E。ACL event 与 device `get_sys_cnt()` 没有公开的共同
  epoch，所以该轨只作为精确 duration 参考，不伪造绝对相位对齐。
- 未参与 SIMT 的 block 保持 `AIC/AIV0/AIV1 Scalar` 在上、对应 kernel 在下的
  1C2V 六轨布局；builder AIV 单独显示 Scalar host 和 W 条 SIMT warp。
- 每个物理 AIC/AIV 只有一条连续 Scalar 轨。`task.execute.*` 与下方 kernel 是
  同一个真实 workload 区间的两种投影，不能相加。
- atomic、DCCI 和 workload 使用设备精确起止时间；连续 poll 合并为 episode，
  instant marker 保存精确 `call_count`，避免逐次记录导致数 GiB raw trace。
- poll 墙钟条同时显示等待对象和邻接精确事件推导出的 Scalar 阶段；marker 另存
  `call_count * 160 ns` 的 atomic 本体参考值，不把整个等待墙钟冒充成指令 latency。
- 不生成 `scalar.control` 兜底段；普通补集按具名 Scalar 阶段展示。

## 机械校验

18 份最终 JSON 全部通过同一组检查：

- schema=`simt_cross_core_g0_swimlane_v13`，raw trace ABI=6；
- workload 严格等于 `6/28/4/1`；
- 真机完整 PA oracle 1/1 通过，构建 1280 task、执行 1024 task；
- 96 条物理 Scalar 均非空，相邻持续区间连续，空白轨和重叠轨均为 0；
- Scalar 与 kernel 各有 1024 个 workload 区间，task、pid、owner、kind、engine、
  起点和时长逐项一致；
- `kernel_end_to_end` 事件恰好一个，duration 与顶层 `kernel_end_to_end_us` 相等；
- 1793 个 Scalar poll marker 的 `call_count` 合计与顶层 summary 精确闭合；
- `scalar.control` 事件数为 0，`jq` 完整解析通过；
- 目录中恰好 18 个 JSON，每个 B 恰好 3 个 W，没有覆盖旧路径。

## 复现命令

下面以 B8/W5 为例。其他配置替换 `SIMT_CROSS_CORE_GM_BUILDER_WARPS` 和
`--builders` 即可；workload 固定在源码中，无需也不能传 `1/1/1/1` 等覆盖值。

```bash
SIMT_CROSS_CORE_GM_BUILDER_WARPS=5 \
  ./tests/atomic_probe/pa_scheduler/simt_cross_core_dag/run.sh build-gm

./tests/atomic_probe/pa_scheduler/simt_cross_core_dag/run.sh run-gm \
  --builders 8 --device 0 --batches 256 --runs 11

SIMT_CROSS_CORE_GM_BUILDER_WARPS=5 \
  ./tests/atomic_probe/pa_scheduler/simt_cross_core_dag/run.sh build-gm-swimlane

./tests/atomic_probe/pa_scheduler/simt_cross_core_dag/run.sh run-gm-swimlane \
  --builders 8 --device 0 --batches 256 --runs 1 \
  --swimlane-json <新的、不存在的输出路径>
```
