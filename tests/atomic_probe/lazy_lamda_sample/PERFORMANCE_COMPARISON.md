# A/B/C 独立运行性能对比

## 结论

最终口径是 72 次独立 host 启动：A/B/C 各 24 个 device 样本，每次均为 `--runs 1`。
去除统一 Hampel 规则识别的异常后，lazy 的直接对照 C-B 有两个读数：

- 不考虑采集时间配对的总体中位差为 `+74.364 us / +1.986%`；
- 以相邻 A/B/C 排列块抵消时间漂移后，中位差只有 `+1.503 us / +0.040%`，C 与 B
  各自更快 `11/22` 个有效块。

因此，本批证据不支持 lazy lambda 存在稳定的性能收益或回退。C-B 远低于 5% 门槛，不启动
I-cache PMU 对比，也不把静态 `.text` 差异写成 I-cache 因果。

B-A 的变化则稳定：去异常配对中 B 快 `206.270 us / 5.214%`，且 B 在 `22/22` 个有效块更快。
但 A-B 同时包含 compete-first 流程、split finish/outlining 和代码布局变化，不能称为 lazy lambda
收益。C 相对 A 的去异常配对中位收益为 `144.819 us / 3.660%`。

## 固定测量口径

- 采集时间：2026-07-20 07:02:23–07:12:45 UTC；
- CCEC device0，32 AIC + 64 AIV，共 96 worker；
- 完整 `Alloc -> QK -> SF -> PV -> UP` task 流；
- b256、real-compute；
- `PA_BUILD_SUBMIT_PMU=0`，每份原始输出均显示 `[PMU-CONFIG] window=off`；
- ELF 以 `PA_BUILD_SWIMLANE=1` 构建，运行时显式 `--no-swimlane`；
- 每次 host 启动只做一个 device run，避免在同一进程中混合冷/暖运行状态；
- 六种 A/B/C 启动排列各重复四次，每版启动位置 1/2/3 均为 8/8/8；
- 72 次 host 启动全部通过 execution、semantic 和 postprocess 门禁。

采集 shell 中 `task-submit` 与 `npu-smi` 均不可用。依据用户明确授权直接访问 device0，本次并非
队列隔离运行；因此可以证明配置一致和程序门禁通过，不能证明采集期间没有外部负载干扰。

## 异常值规则

原始日志不删除。统计前按每个版本自己的 24 个样本固定使用 Hampel 规则：

`中位数 ± 3 × 1.4826 × MAD`

越界样本不进入“去异常”总体统计；配对比较中，只要 A/B/C 对应成员之一越界，该比较涉及的
整个排列块就不进入相应配对统计。规则和逐样本 `included` 标记可由
`./analyze_performance.py` 复算。

| 形态 | Hampel 下界 / 上界 | 排除样本 |
| --- | ---: | --- |
| A original | 3854.127 / 4055.564 us | block11 `3832.087`、block23 `3788.296` us |
| B compete-first eager | 3338.151 / 4150.683 us | 无 |
| C compete-first + lazy | 3590.015 / 4066.068 us | block01 `4084.775`、block09 `4370.394` us |

## Submit span

| 形态 | 原始 N / 中位数 | 去异常 N / 中位数 | 去异常均值 | 去异常 min / max | 去异常标准差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A original | 24 / 3954.845 us | 22 / 3956.298 us | 3941.249 us | 3856.072 / 4000.058 us | 38.004 us |
| B compete-first eager | 24 / 3744.417 us | 24 / 3744.417 us | 3747.214 us | 3625.727 / 3866.154 us | 81.935 us |
| C compete-first + lazy | 24 / 3828.042 us | 22 / 3818.781 us | 3788.503 us | 3643.199 / 3950.280 us | 83.869 us |

| 对比 | 去异常总体中位差 | 去异常配对中位差 | 候选更快块数 | 被排除块 |
| --- | ---: | ---: | ---: | --- |
| B - A | -211.881 us / -5.356% | -206.270 us / -5.214% | B 22/22 | 11、23 |
| C - B（lazy） | +74.364 us / +1.986% | +1.503 us / +0.040% | C 11/22 | 01、09 |
| C - A（总变化） | -137.517 us / -3.476% | -144.819 us / -3.660% | C 20/20 | 01、09、11、23 |

负值表示候选版本更快。配对百分比使用对应 baseline 的去异常总体中位数作为分母。

## 为什么不配对时看起来差很多

设备状态在约十分钟采集窗口内并非平稳。即使六种排列和位置完全平衡，三个版本在采集前后半段
仍发生不同方向的漂移：

| 形态 | block01–12 原始中位数 | block13–24 原始中位数 | 后半 - 前半 |
| --- | ---: | ---: | ---: |
| A | 3939.252 us | 3956.298 us | +17.046 us |
| B | 3668.446 us | 3788.252 us | +119.806 us |
| C | 3840.722 us | 3786.717 us | -54.004 us |

相应地，C-B 的原始配对中位差在前半段为 `+101.951 us`，后半段变成 `-6.089 us`；合并 24 块
后为 `+8.065 us`，统一去异常后为 `+1.503 us`。这说明不按相邻时间块比较时，B 后半段变慢、
C 后半段变快会把总体中位数拉成一个看似明确的 `+74.364 us` 差异。

明显长尾也有可见的动态调度特征。C 的两个高异常样本中，`frontier_flag` 中位数为 `12969`、
`RingBp` 中位数为 `53`；C 的保留样本对应值只有 `3322.5` 和 `14`。C 内部 Submit span 与
`frontier_flag`、`RingBp` 的 Pearson 相关系数分别为 `0.860`、`0.797`。A 的两个低异常样本则
伴随很小的 SF `max_us`：中位数 `96.863 us`，保留样本为 `226.224 us`，A 内两者相关系数为
`0.941`。

这些证据把大波动定位到运行时调度等待和 kernel 尾时延，而不是配置错误：异常日志依然全部
通过功能门禁，PMU 和泳道配置也没有变化。由于本次是未隔离的 device0 直跑，现有证据不能继续
区分这些动态变化是设备外部负载、频率状态还是 PA 内部竞争的哪一种组合。

## lazy 动态证据

三份包的 device0 b1 功能门禁均通过。B/C 的 `[LAZY_SAMPLE_FRONTEND]` 同时打印实测值和精确
期望值：

| 形态 | views | tensor args | scalar args | resets |
| --- | ---: | ---: | ---: | ---: |
| B compete-first eager | 192 | 2112 | 864 | 384 |
| C compete-first lazy | 97 | 1162 | 9 | 384 |

C 的计数下降证明 loser 没有执行 input/scalar thunk。A 没有编译该专用 banner，因此不把源码
推算值冒充实测值。

## `.text`、实测 ELF 与反汇编

| 形态 | final `.text` | `.text` SHA256 | 实测 final ELF SHA256 |
| --- | ---: | --- | --- |
| A original | 780344 B | `018ae7dc29ce78249b2e3bff84b0faa2e671e2448c9316719fb48bc59139d2fc` | `76c961f846c7efc89b40942c3a8113530a6da2c7bfbdc601615852f8c5b79cfc` |
| B compete-first eager | 547640 B | `8fd6209b2f0f9d1fb48d4d8a16cf27649e26071153b908e1e0e39ca107d63589` | `82a27e206ee1b2411964c0530c73adf92a2b032ac202a7c65c6b5f7d76a4571b` |
| C compete-first lazy | 547640 B | `866ed1081ebb94038a8feca87ccedf95e67aebfdaef11651c109f86672be2bdd` | `8d293c52312429d13efe89592ed705c672ceef1f2875b5e3bd25582419d37893` |

带源码注释的反汇编是从表中当前 `artifacts/measured/pa_scheduler_kernel.o` 重新生成的，文件头记录
完整 ELF SHA256。最终性能复测直接执行相同 clean-build host/final ELF，期间没有改源码或重编译，
所以反汇编无需因性能日志更新而修改。只有源码或 ELF 改变时才需要重新运行各包的
`./generate_disassembly.sh`。

可复算数据位于：

- `performance_summary.json`：统计、阈值、配对结果和诊断量；
- `performance_samples.tsv`：72 个样本及异常标记；
- 各包 `output/performance/raw/`：完整设备原始输出；
- 各包 `output/performance/SUMMARY.md`、`samples.tsv`、`summary.json`：单包独立结果。
