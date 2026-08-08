# 2026-08-08 恢复固定 task 时长后的真实 A5 泳道图

本目录只保存恢复串行 task workload 后的有效结果。固定条件为：

- workload：`QK/SF/PV/UP=6/28/4/1`；
- 1280 个 active task、1024 个 kernel task；
- 256 个真实 metadata writer、255 条串行 writer edge；
- GM 路径，token1，dispatch-window256；
- 完整 host oracle，不减少 repeat、矩阵规模、搬运或 task/DAG。

## 当前记录

| 文件 | 配置 | trace-off 性能 | trace-on E2E | device span | trace-on Build span |
| ---- | ---- | -------------- | ------------ | ----------- | ------------------- |
| `ordinary_gm_b9_b256_warp5_token1_traceoff699us_restored_serial_task_duration_swimlane.json` | B9/W5；9 个 AIV builder × 5 warp；55 个 AIV + 32 个 AIC executor | 21 次 min 688.774 µs；median **698.949 µs** | 1481.580 µs | 794.663 µs | 778.639 µs |

trace-off 与 trace-on 必须分开理解：698.949 µs 是未插桩二进制的 21 次中位数；
JSON 对应的 1481.580 µs 是插入 atomic/DCCI/Scalar 记录后的单次结果，只用于分析
结构。trace-on 运行 correctness 1/1 PASS，SIMT/Scalar atomic call 为
4120/62827，DCCI call/line 均为 12499。

同日相邻配置复测中，B11/W5 的 21 次中位数为 709.624 µs、Build 中位数为
520.113 µs。B9 的 Build 中位数为 616.387 µs。B11 虽显著缩短 Build，却没有缩短
E2E，说明增加 builder 后 Build 已不再是首要关键路径。历史同一有效协议的最佳
窗口为 B11/W5 的 696.268 µs；与当前 B9 的 698.949 µs 相差 2.681 µs，属于
unlocked 环境噪声范围。有效性能应表述为约 **0.70 ms**。

## 关键瓶颈

泳道图中 512 个 AIC task 的 workload 总量为：

| 类别 | 数量 | 平均时间 | 总时间 |
| ---- | ---- | -------- | ------ |
| QK | 256 | 45.395 µs | 11621.072 µs |
| PV | 256 | 27.681 µs | 7086.220 µs |
| 合计 | 512 | - | 18707.292 µs |

仅 workload 的 32-AIC 平均下界就是 584.603 µs。关键 AIC owner 28 执行 8 个 QK
和 8 个 PV，workload 为 589.352 µs，task 间 Scalar 间隙为 51.231 µs，连续
活跃区间为 640.583 µs。51.231 µs 间隙分解如下：

| 非 workload 项 | 时间 |
| --------------- | ---- |
| task prepare | 19.385 µs |
| task complete | 10.082 µs |
| payload DCCI | 6.139 µs |
| BUILT/exec-state wait | 6.026 µs |
| payload bind | 3.727 µs |
| dispatch atomic | 3.689 µs |
| fan-in wait | 2.183 µs |

所以主要瓶颈是固定 AIC task 服务时间；关键 AIC 活跃区间约 92% 是 workload，约
8% 才是可优化的 Scalar 间隙。AIV 最后的 UP 约 2 µs，其前方大段 poll 是在等待
上游 PV/Build，不能把等待区间直接当作 atomic 可删除开销。

## 无效结果隔离

`2026-8-7/ordinary_gm_b11_b256_warp5_token1_traceoff586us_aic_l1_pingpong_pipeline_swimlane.json`
是已经回退的无效实验。它错误地把相邻 AIC repeat 流水化，缩短了用于模拟约
50 µs task 的负载时间。585.865 µs 不计入当前性能，不得用于方案对比。

## 复现命令

```bash
SIMT_CROSS_CORE_GM_BUILDER_WARPS=5 \
SIMT_CROSS_CORE_GM_TOKENS_PER_OWNER=1 \
SIMT_CROSS_CORE_GM_DISPATCH_WINDOW_BATCHES=256 \
  ./run.sh build-gm-swimlane
./run.sh run-gm-swimlane \
  --builders 9 --device 0 --batches 256 --runs 1 \
  --swimlane-json test_record/2026-8-8/<new-file>.json
```

host 会拒绝覆盖已有 JSON；新采样必须使用新文件名。
