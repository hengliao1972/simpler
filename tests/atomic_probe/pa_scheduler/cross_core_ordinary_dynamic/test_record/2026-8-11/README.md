# 2026-08-11 动态 TaskPlan A5 记录

## 固定口径

- A5 `device=0`，B256，96 个 Scalar worker；
- shared TensorMap，无 Host 预制计划的全员真实 replay/per-task tournament 路径；
- 主 `perf_off` 与首份泳道使用 QK/SF/PV/UP=`6/28/4/1`；两种负载均为
  1280 个 task、1024 个 kernel task；
- 文中的 `perf_off` 指独立 `perf-clock` ELF：PMU、泳道、atomic/DCCI 和阶段观察
  全部关闭，统计全局最早 startup begin 到最晚 FinalDrain end；
- 先完成一次预热，再用五个独立进程各运行一次。

## perf_off 结果

五次结果为 `2417.045 / 2453.597 / 2444.629 / 2439.774 / 2428.729 us`：

- 中位数：**2439.774 us**；
- 均值：2436.755 us；
- 最小/最大：2417.045 / 2453.597 us。

所有样本均通过执行、语义和后处理校验。该实现让 96 个 worker 各自 replay
1280 个 Submit，泳道元数据为 `all_worker_replay/compare_exchange`。相同口径下，
它的中位数是 Host 预制版的约 3.01 倍，多 1628.414 us。

## 泳道图

- `dynamic_b256_realcompute_6_28_4_1_perfoff2440us_atomic_dcci_swimlane.json`
- trace-on lifecycle：2898.234 us；raw 记录 428,923 条，`dropped=0`；
- 文件只保留 merged Perfetto 泳道图，不保存 raw 和 `exclusive_analysis.json`；
- 文件名中的 `2440us` 是无插桩 `perf_off` 中位数，不是 trace-on 时间。

## `1/1/1/1` 附加泳道

- `dynamic_b256_realcompute_1_1_1_1_traceon2205us_atomic_dcci_swimlane.json`
- workload：QK/SF/PV/UP=`1/1/1/1`；
- trace-on lifecycle：2205.048 us，Submit span：1056.038 us；
- raw 记录 428,927 条，`dropped=0`，全部正确性检查 PASS；
- 本次只补泳道，没有另做五轮 `perf_off`，因此文件名明确使用 `traceon2205us`。

复现命令：

```bash
./run.sh build-perf-clock ccec
./run.sh perf-clock ccec --device 0 --batches 256 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1
./run.sh swimlane ccec --device 0 --batches 256 \
  --winner-workload real-compute --real-compute-counts 6,28,4,1
```

采集附加泳道时将最后一个参数替换为 `1,1,1,1`。
