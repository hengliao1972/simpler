# 2026-08-11 Host 预制 TaskPlan A5 记录

## 固定口径

- A5 `device=0`，B256，96 个 Scalar worker；
- shared TensorMap，Host 在 kernel launch 前预制 `SharedHostTaskPlan`；
- 主 `perf_off` 与首份泳道使用 QK/SF/PV/UP=`6/28/4/1`；两种负载均为
  1280 个 task、1024 个 kernel task；
- 文中的 `perf_off` 指独立 `perf-clock` ELF：PMU、泳道、atomic/DCCI 和阶段观察
  全部关闭，统计全局最早 startup begin 到最晚 FinalDrain end；
- 先完成一次预热，再用五个独立进程各运行一次。

## perf_off 结果

五次结果为 `812.157 / 811.132 / 818.405 / 811.360 / 805.209 us`：

- 中位数：**811.360 us**；
- 均值：811.653 us；
- 最小/最大：805.209 / 818.405 us。

所有样本均通过执行、语义和后处理校验。该实现使用 central Build ticket、稀疏
metadata-writer 插入链；泳道元数据为 `central_ticket/fetch_add`。

## 泳道图

- `host_prebuilt_b256_realcompute_6_28_4_1_perfoff811us_atomic_dcci_swimlane.json`
- trace-on lifecycle：903.125 us；raw 记录 53,600 条，`dropped=0`；
- 文件只保留 merged Perfetto 泳道图，不保存 raw 和 `exclusive_analysis.json`；
- 文件名中的 `811us` 是无插桩 `perf_off` 中位数，不是 trace-on 时间。

## `1/1/1/1` 附加泳道

- `host_prebuilt_b256_realcompute_1_1_1_1_traceon564us_atomic_dcci_swimlane.json`
- workload：QK/SF/PV/UP=`1/1/1/1`；
- trace-on lifecycle：564.396 us，Submit span：556.008 us；
- raw 记录 53,938 条，`dropped=0`，全部正确性检查 PASS；
- 本次只补泳道，没有另做五轮 `perf_off`，因此文件名明确使用 `traceon564us`。

复现命令：

```bash
./run.sh build-perf-clock ccec
./run.sh perf-clock ccec --device 0 --batches 256 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1
./run.sh swimlane ccec --device 0 --batches 256 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1
```

采集附加泳道时将最后一个参数替换为 `1,1,1,1`。
