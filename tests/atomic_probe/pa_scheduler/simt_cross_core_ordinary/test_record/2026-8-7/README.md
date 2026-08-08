# 2026-08-07 真实 A5 泳道图说明

本目录只记录 `simt_cross_core_ordinary` 的 GM 路径。固定真实 workload 为
`QK/SF/PV/UP=6/28/4/1`。

| 文件 | builder / executor 配置 | token / 派发 | last-writer 发布 | trace-off 中位数 | trace-on E2E | device / Build span |
| ---- | ----------------------- | ------------ | ------------------ | ---------------- | ------------ | ------------------- |
| `ordinary_gm_b8_b256_warp5_traceoff939us_traceon_e2e1819us_device1231us_build1201us_global_sparse_writer_swimlane.json` | 8 个 builder AIV × 5 warp；56 AIV + 32 AIC executor | token4；batch 内上下游交错 | per-symbol CAS | 938.615 µs | 1818.676 µs | 1230.557 / 1200.524 µs |
| `ordinary_gm_b12_b256_warp5_token4_traceoff843us_traceon_e2e1601us_device1019us_build987us_upstream_first_dispatch_global_sparse_writer_swimlane.json` | 12 个 builder AIV × 5 warp；52 AIV + 32 AIC executor | token4；window256，上游优先 | per-symbol CAS | 约 843 µs | 1600.544 µs | 1019.012 / 约 987 µs |
| `ordinary_gm_b9_b256_warp5_token1_traceoff798us_traceon_e2e1743us_device1059us_build1030us_upstream_first_serial_store_swimlane.json` | 9 个 builder AIV × 5 warp；55 AIV + 32 AIC executor | token1；window256，上游优先 | 串行区 non-cacheable store | 797.772 µs | 1743.330 µs | 1059.325 / 1029.609 µs |
| `ordinary_gm_b9_b256_warp5_token1_traceoff798us_traceon_e2e1824us_device1051us_build1021us_upstream_first_serial_store_selfdescribing_swimlane.json` | 9 个 builder AIV × 5 warp；55 AIV + 32 AIC executor | token1；window256，上游优先 | 串行区 non-cacheable store | 798.371 µs | 1824.008 µs | 1050.913 / 1021.123 µs |
| `ordinary_gm_b9_b256_warp5_token1_traceoff719us_traceon_e2e1532us_device804us_build788us_published_payload_fast_bind_swimlane.json` | 9 个 builder AIV × 5 warp；55 AIV + 32 AIC executor | token1；window256 | 同上；executor 直接绑定已发布 payload | 约 719 µs | 1532 µs | 804 / 788 µs |
| `ordinary_gm_b11_b256_warp5_token1_traceoff702us_traceon_e2e1416us_device715us_build649us_dynamic_witness_prebuilt_vend_no_claimed_store_swimlane.json` | 11 个 builder AIV × 5 warp；53 AIV + 32 AIC executor | token1；window256 | 动态 witness、预发布 vend、取消 CLAIMED store | 约 702 µs | 1416 µs | 715 / 649 µs |
| `ordinary_gm_b11_b256_warp5_token1_traceoff700us_traceon_e2e1400us_device717us_build662us_no_alloc_done_swimlane.json` | B11/W5；53 AIV + 32 AIC executor | token1；window256 | final drain 作为唯一完成计数 | 约 700 µs | 1400 µs | 717 / 662 µs |
| `ordinary_gm_b11_b256_warp5_token1_traceoff696us_traceon_e2e1383us_device709us_build660us_compact_terminal_witness_swimlane.json` | B11/W5；53 AIV + 32 AIC executor | token1；window256 | 单词终态 witness hash | 696.268 µs | 1383.123 µs | 709.384 / 659.596 µs |
| `ordinary_gm_b11_b256_warp5_token1_traceoff586us_aic_l1_pingpong_pipeline_swimlane.json` | B11/W5；53 AIV + 32 AIC executor | token1；window256 | **无效实验，已回退**：错误地流水化模拟 task 负载 | ~~585.865 µs~~ | 1422.971 µs | 679.624 / 661.226 µs |

最后一行只保留作错误实验记录，不能作为调度性能。它虽然没有减少指令条数，却
通过 overlap 缩短了原本用于模拟约 50 µs task 的 workload，与直接减小 workload
没有本质区别，当前源码已经回退。当前有效的最后版本是上一行的 compact terminal
witness；其 JSON 顶层自描述字段为：

```text
builder_count=11
builder_warps_per_aiv=5
execution_tokens_per_owner=1
dispatch_window_batches=256
metadata_last_writer_publish=serialized_noncacheable_store
```

`B11` 只占用 11 个 AIV 做 SIMT Build，不占用 AIC；余下 53 个 AIV 与全部 32 个
AIC 负责执行。每个 builder 的 5 个 warp 只有 lane0 参与任务构建，共 55 个并行
leader。

泳道图顶层字段明确给出 `metadata_writer_tasks=256` 和
`metadata_insert_predecessor_edges=255`。SIMT builder 中：

- `serial_metadata_insert` 只出现 256 次；
- `publish_without_metadata_insert` 出现 1024 次；
- `simt_global_insert_predecessor_poll` 展示真实 writer 等待；
- task execute 与 kernel projection 均为 1024 次；QK/SF/PV/UP 各 256 次。

最终自描述文件还验证了：dispatch DCCI 事件为 0，metadata last-writer atomic
事件为 0；这分别对应一次性 `LoadDev64` 派发表读取和全局单写区间中的
non-cacheable store。其余 producer/payload/Scalar 一致性 DCCI 仍按协议保留。

当前有效历史最佳来自 compact terminal witness 的 21 次 trace-off 窗口：21/21
correctness PASS，最小值 692.642 µs，中位数 696.268 µs。2026-08-08 恢复串行
task 时长后的最新复测与泳道图移至相邻的 `2026-8-8/` 目录。trace-on 只用于观察
时序，不能直接与 trace-off 比性能。设备为 unlocked device 0，环境没有
`task-submit`，数据可能受到同卡其他进程干扰。

生成命令：

```bash
SIMT_CROSS_CORE_GM_BUILDER_WARPS=5 \
SIMT_CROSS_CORE_GM_TOKENS_PER_OWNER=1 \
SIMT_CROSS_CORE_GM_DISPATCH_WINDOW_BATCHES=256 \
  ./run.sh build-gm-swimlane
./run.sh run-gm-swimlane \
  --builders 11 --device 0 --batches 256 --runs 1 \
  --swimlane-json test_record/2026-8-7/<new-file>.json
```

host 拒绝覆盖已存在的 JSON；每次采样必须使用新文件名。
