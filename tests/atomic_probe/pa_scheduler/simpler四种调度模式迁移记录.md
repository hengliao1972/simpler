# Simpler 四种调度模式迁移记录

## 1. 目标与边界

本文记录以独立调度器为参考，将下列四种模式接入 Simpler 真实
`fully_distributed_within_core` 路径的过程：

1. `cross_core_ordinary`；
2. `cross_core_dag`；
3. `simt_cross_core_ordinary`；
4. `simt_cross_core_dag`。

功能迁移优先，之后再使用同一业务负载、同一起止边界与同一验收
条件对比独立调度器。不用不同端点的绝对时间直接相减。

## 2. 功能迁移状态

| 模式 | 真实动态 Submit | 独立状态/复位 | 真实 A5 PA B1/B256 | 阶段提交 |
| --- | --- | --- | --- | --- |
| `cross_core_ordinary` | 已接通 | 已闭合 | PASS | `405c060f` 及前置提交 |
| `cross_core_dag` | 已接通 | 已闭合 | PASS | `4ecc6c2f` / `197d8003` / `2d0e0426` |
| `simt_cross_core_ordinary` | 已接通 | 已闭合 | PASS | `2843d82a` / `7cf087c4` / `b13f1b19` / `f20522e4` |
| `simt_cross_core_dag` | 已接通 | 已闭合 | PASS | `ba0abef6` / `8bd60f35` / `8d464955` |

SIMT 两种模式均由唯一 `block0/AIV0` Main Scalar 承载持久 builder VF；
该 Scalar 不 replay orchestration。其他 95 个 Scalar 发布动态请求，Build owner
与 Execute owner 仍独立。

## 3. 泛化正确性证据

除 PA 外，已在真实 A5 上运行：

```text
examples/a5/fully_distributed_within_core/submit_dependency_smoke/
test_submit_dependency_smoke.py
case: A5OnboardBd24ExistingInoutChain
```

该用例使用 `block_dim=24` 的 72 个 worker，验证 AIC writer 到 AIV INOUT
consumer 的真实依赖链。四种调度模式均 PASS。这项证据表明通用路径不
依赖 PA 的 `TaskKind` 、QK/SF/PV/UP 固定图形或固定 batch 大小。

## 4. 端到端性能口径修正

### 4.1 旧 `perf-clock` 不能用于四模式横比

旧边界是每核首个 Submit 起点到末个 Submit 返回，它有两个确定缺口：

- 末次 Submit 后的 FinalDrain 与其中 Kernel 没有被统计；
- SIMT 专职 builder 不 replay Submit，其计数合法地为 `0/0`，旧 host
  却要求 96 核全部为相同的正 Submit 数。

AICPU `orch_start/orch_end` 也不能替代该口径：它还包含 runtime 交接、
PMU 状态恢复和结果检查，而现有解析器又要求传统 scheduler 日志。

### 4.2 新的 cross-core 合同

`perf-clock` 仍然只复用每核固定 64B 状态，不新增逐 task 事件：

```text
global min(每核 startup increment 前起点)
    -> global max(每核 FinalDrain 完成终点)
```

- 模式 1/2：96 个 replay worker，0 builder；
- 模式 3/4：95 个 replay worker，唯一 `block0/AIV0` builder；
- replay worker 必须满足 `submit_count == expected_submit_count > 0`；
- builder 必须满足 `submit_count == expected_submit_count == 0`；
- 设备 raw 使用独立 mode 值，host 使用独立 schema
  `fdwic-cross-core-e2e-clock-v1`，不会把新数据静默解释为旧 Submit 窗。

`perf-clock-kernel` 在 cross-core 中使用相同端到端窗，不再套用
`5 * batches` 的 PA 专用 Kernel 数量上下界。

### 4.3 实测门槛

| 模式 | 用例 | 结果 | worker 闭合 | 单次端到端时间 |
| --- | --- | --- | --- | ---: |
| `cross_core_ordinary` | A5 PA CaseB1 | PASS | 96 replay + 0 builder | 398.216 us |
| `simt_cross_core_ordinary` | A5 PA CaseB1 | PASS | 95 replay + 1 builder | 1368.150 us |

这两个 B1 数字只用来证明端点和角色合同真正在设备上生效，不用来
排名。正式横比必须改用 B256，且用独立进程重复采样。

`cross_core_ordinary` 的 A5 B1 `perf-clock-kernel` 也已 PASS：同一窗口中
统计到 4 次真实 Kernel 调用，总端到端时间为 400.537 us。该产物没有
输出 PA 专用的 `5 * batches` 调用数范围字段。

### 4.4 构建注意点

pytest 会按源码指纹重编 AICore override，但不会代替安装/运行时构建去
重编长期驻留的 `libhost_runtime.so`。修改 host 的合同或 schema 后，每个
scheduler mode 必须先显式调用存量 `RuntimeBuilder` 重建对应 artifact family，
再跑上板用例。不允许用新 AICore 与旧 host 的组合解释 raw。

## 5. 下一阶段

1. 重建四种 scheduler mode 的 A5 shared runtime；
2. 分别运行同一 PA B256，采集独立进程的 startup 到 FinalDrain 数据；
3. 核对 1280 task、1024 Kernel、金标、replay/builder 角色和终态；
4. 再与同负载、同端点的独立调度器数据对比，分析 Simpler 集成差距。
