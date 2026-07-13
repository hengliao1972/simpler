# A5 Cache-Line Cross-Core Probe

## Goal

A5 每个核的 scalar data cache 没有 CPU 式多核 cache coherence。测试需要回答：多个核并发读写同一
64B cache line 时，哪些 scalar 访问方式能保持精确值，哪些 cache 管理路径会覆盖其他核的修改。

测试分别覆盖 AscendC API、CCEC 原始 intrinsic，并以无数据竞争的 CPU 多线程程序作为 coherent
control。结论只能来自精确 oracle 或明确标为 observational 的统计，不能用一次随机现象替代契约。

## 当前验证状态

- 2026-07-11：所有 AscendC source 已使用本机 CANN 9.1 `bisheng`、`dav-3510` 编译通过。
- 2026-07-11：CCEC runner 默认编译 AIV-only kernel；已验证 kernel、link 与 host 构建链路。
- 2026-07-11：CPU control 在 `-Wall -Wextra -Werror` 下编译并运行通过；pytest 节点通过。
- 2026-07-11：经用户授权直接使用 device 0；AIV-only 权威矩阵与 AscendC/CCEC 同-line 最简对照已上板。
- 2026-07-11：两个 `st_dev_same_line` 用例按正确性契约断言同-line mismatch 必须为 0；当前设备会
  复现 mismatch 并返回非零。其余 control 与完整入口结果见上板记录。
- 2026-07-13：新增 AscendC/CCEC `atomic_exch_same_line` 同构对照；三组路径均为 `0/4000` mismatch。
- 原始环境与定量结果记录在 `tests/ATOMIC_MINIBENCH_ONBOARD_LOG.md` 的 2026-07-11 与 2026-07-13 小节。

## 权威覆盖矩阵

`ascendc/cacheline_matrix.asc` 与 `ccec/cacheline_matrix.cpp` 使用相同数据布局和相同值生成公式。

| Mode | 宽度 | 布局 | 参与者 | Oracle |
|---:|---:|---|---|---|
| 0 | 1B | 同一 cache line、不同 slot | 2/4 blocks | 每个 slot 等于最后一轮精确值 |
| 1 | 2B | 同一 cache line、不同 slot | 2/4 blocks | 同上 |
| 2 | 4B | 同一 cache line、不同 slot | 2/4 blocks | 同上 |
| 3 | 8B | 同一 cache line、不同 slot | 2/4 blocks | 同上 |
| 4 | 1B | 每个参与者独占 cache line | 2/4 blocks | 同上 |
| 5 | 2B | 每个参与者独占 cache line | 2/4 blocks | 同上 |
| 6 | 4B | 每个参与者独占 cache line | 2/4 blocks | 同上 |
| 7 | 8B | 每个参与者独占 cache line | 2/4 blocks | 同上 |

权威门禁使用 AIV-only binary：2/4 blocks 表示 2/4 个 vector 核，不是单核测试。每个参与者通过 kernel
内计数和 marker 精确验证。AIC+AIV MIX 不是本 goal 的必要条件；AscendC runner 仅在显式设置
`ATOMIC_PROBE_RUN_MIX=1` 时运行补充 MIX 覆盖，且 MIX 比例不参与充分性判定。

### AscendC 与 CCEC 同构关系

| 语义 | AscendC | CCEC | 本机 CANN 9.1 依据 |
|---|---|---|---|
| bypass write | `WriteGmByPassDCache<T>` | `st_dev` | `kernel_scalar.h` 的 1/2/4/8B 分支 |
| bypass read | `ReadGmByPassDCache<T>` | `ld_dev` | 同上 |
| AIV-only barrier | `SyncAll<true>()` | flag 14 FFTS 协议 | `dav_3510/kernel_operator_sync_impl.h` |
| atomic | `AtomicAdd/AtomicMax` | `atomicAdd/atomicMax` | `dav_3510/kernel_operator_atomic_impl.h` |

CCEC 不再使用 GM atomic counter 模拟 `SyncAll`，避免 barrier 本身污染待测 cache line 或增加额外 GM
竞争。`ccec_utils.h` 按上述本机实现映射 FFTS 同步。

### DSB 的本机实现边界

DSB 是 Data Synchronization Barrier，不是 Data Store Barrier。本机 CANN 9.1 / dav-3510 的调用链为：

```text
DataSyncBarrier<MemDsbT::ALL>()
  -> DataSyncBarrierImpl<MemDsbT::ALL>()
  -> dsb(DSB_ALL)
  -> __builtin_cce_dsb
```

本机 `cce_aicore_intrinsics.h` 对 `DSB_ALL` 的注释为 “Wait for all memory access instructions.”。它等待
当前核此前的相应 memory access，不是跨核 rendezvous，也不提供 cache coherence；跨 AIV 会合另由
`SyncAll<true>()`/FFTS flag 14 完成。

## 同 line `st_dev` 最简对照

`ascendc/st_dev_same_line.asc` 与 `ccec/st_dev_same_line.cpp` 固定启动两个 AIV，每个 AIV 只写自己的
4B slot，使用相同值公式和三组路径：

| 路径 | 类型 | Oracle |
|---|---|---|
| 不同 slot、同一 64B line、仅 loop-end DSB | regression gating | 4000 次必须全部等于最后一轮值；任一 mismatch 都使测试失败 |
| 每个 AIV 独占 64B line、仅 loop-end DSB | gating control | 4000 次全部等于最后一轮值 |
| 不同 slot、同一 64B line、逐轮 DSB | gating control | 4000 次全部等于最后一轮值 |

device 0 当前正确性断言实测：CCEC 同-line mismatch `1589/4000`、exit 1，AscendC
`1792/4000`、exit 1；两个 control 在两条路径均为 `0/4000`。另做的单 AIV 同址隔离 control 为
`0/2000`，所以当前结论限定为多 AIV 同 cacheline 干扰，不能写成单 AIV 同址 `st_dev` 自身不保序，
也不推断尚未验证的底层实现机制。

本用例的目标是让该问题以正确性失败显式暴露。问题路径必须保留 loop-end DSB，不能通过改成逐轮
DSB、拆到不同 cache line，或把 `mismatch > 0` 写成成功条件来让测试通过；后两种安全路径只作为对照。

## 同 line `AtomicExch` 对照

`ascendc/atomic_exch_same_line.asc` 与 `ccec/atomic_exch_same_line.cpp` 完全复用上述两个 AIV、
4B slot、20 launch × 100 trial × 257 round、三组布局和同步点，仅把测试数据写替换为
`AtomicExch<uint32_t>` / `atomicExch`。选择 exchange 是为了保留任意轮次值；`AtomicMax` 与
`AtomicAdd` 的终值会天然掩盖执行顺序。

device 0 实测：CCEC 与 AscendC 的同-line loop-end DSB、分-line loop-end DSB、同-line 逐轮 DSB
均为 `0/4000` mismatch，参与计数与 marker 精确，两个用例均 exit 0。当前证据说明同构压力下
AtomicExch 没有复现 st_dev 的末值回退。该 oracle 只检查每个 trial 的最终值，不证明中间 AtomicExch
绝无重排；本用例仍是两个核写同一 cacheline 的不同 4B slot，不覆盖两个核写同一个 4B 地址，也不能
外推到其他 atomic 类型或其他内存序场景。

## 其余探针

| 文件 | 类型 | 验证内容 |
|---|---|---|
| `ascendc/atomic.asc` / `ccec/atomic_cas_probe.cpp` | gating | CAS 最终值 2000，且全局恰好一次成功 |
| `ascendc/atomic64_verify.asc` | gating | 32/64-bit add/max 精确终值 |
| `ascendc/cacheline_blast.asc` / `ccec/atomic_blast.cpp` | gating + observation | atomic 目标确实改变且邻接字节不变；dcci 反向覆盖单列观察 |
| `ascendc/bypass_dcache_probe.asc` / `ccec/bypass_dcache_ccec.cpp` | gating | 1/2/4/8B ld_dev 共 120 次精确读取、st_dev/atomic、publish/observe |
| `ascendc/concurrent_cacheline.asc` | gating + observation | 多 block st_dev、producer/consumer、持续读；store+dcci race 单列观察 |
| `ascendc/cacheline_stress.asc` / `ccec/concurrent_stress.cpp` | observation + control | tight-loop dcci hazard；CCEC st_dev control 精确终值 |
| `ascendc/st_dev_same_line.asc` / `ccec/st_dev_same_line.cpp` | regression gating + control | 两 AIV 同 line 使用精确终值断言，当前问题以非零退出码暴露；分 line 与逐轮 DSB 为精确 control |
| `ascendc/atomic_exch_same_line.asc` / `ccec/atomic_exch_same_line.cpp` | gating + control | 与 st_dev 同构的 AtomicExch 末值顺序对照；三组路径均精确通过 |
| `ascendc/dcci_atomic_stress.asc` | gating + observation | dcci type 调查、10K st_dev、ld_dev snapshot、atomic 并发 |
| `ccec/dcci_clean_clobber.cpp` | gating | 有序 dirty/clean line 的 dcci clobber 与 control |
| `ascendc/mb2_flags_clobber.asc` | gating + observation | AtomicMax flags 无丢失；store+dcci 仅统计 |
| `ascendc/mb8_dcci_seam.asc` / `ccec/dcci_seam.cpp` | gating | 数据、flag round count、error count 全精确 |
| `cpu/cpu_atomicity.cpp` | gating + observation | coherent CPU 同/异 cacheline 同构 control、atomic、snapshot、spinlock |

## Oracle 与退出码规则

1. **确定性安全契约必须 gating**：目标值、邻居值、参与核数、执行 marker 全部精确匹配；任一失败返回非零。
2. **风险场景若有严格 phase ordering，可 gating**：例如 dirty line 在 st_dev 后执行 dcci，预期覆盖值可精确断言。
3. **一般无 ordering 竞态只 observational**：输出 survived/clobbered 分布，不硬编码某次调度结果。
   但 `st_dev_same_line` 验证的是各核独占 slot 的正确性契约，不是把已知错误当成功条件的
   characterization；它要求 mismatch 为 0，同时打印实际 mismatch 数量用于复现与诊断。
4. ACL、编译、link、timeout、kernel sync 或 semantic assertion 失败均向 runner 传播非零退出码。
5. 结果槽用 bypass store 发布，并在 host 消费前显式完成；CCEC 不依赖 scalar 自动 dcci。

原 stress oracle 已从 `actual >= base` 改为 `actual == base + last_round`。byte-width 探针同时检查原子或
st_dev 的目标字节确实改变，防止 no-op kernel 因“邻居没坏”而假通过。

## CCEC 编译约束

本机 `ccec -mllvm -print-all-options` 显示：

```text
cce-aicore-dcci-insert-for-scalar = 1 (default: 1)
cce-aicore-dcci-before-kernel-end = 1 (default: 1)
```

所有 CCEC runner 显式设置：

```text
-mllvm -cce-aicore-dcci-insert-for-scalar=false
-mllvm -cce-aicore-dcci-before-kernel-end=false
```

普通 scalar store + 显式 dcci 只保留在专门制造 cache-line hazard 的 mode；安全数据路径与结果发布使用
`st_dev`。这两个选项关闭的是编译器插入，不改变显式 dcci。

## 执行入口

### CPU control

本机已有 PTO-ISA 可直接复用，无需下载：

```bash
export PTO_ISA_ROOT=/home/q00473782/atomic/codex/simpler-fully_distributed/build/pto-isa
export PYTHONPATH="$PWD/python:$PWD"
.venv/bin/python -m pytest tests/atomic_probe/test_atomic_probe.py -k cpu -q --clone-protocol https
```

### Build-only

```bash
bisheng -xasc tests/atomic_probe/ascendc/cacheline_matrix.asc \
  --npu-arch=dav-3510 -DPROBE_CORE_VARIANT=0 -o /tmp/cacheline_matrix_aiv
tests/atomic_probe/ccec/run_all.sh build
```

### A5 onboard

CI/共享环境优先由 `task-submit` 独占设备：

```bash
.claude/skills/onboard-arch-precheck/check.sh a5 || exit 1
command -v task-submit >/dev/null || exit 1

task-submit --timeout 1800 --max-time 1800 --device auto --device-num 1 \
  --run "cd $PWD && \
    PYTHONPATH=$PWD/python:$PWD \
    PTO_ISA_ROOT=$PTO_ISA_ROOT \
    .venv/bin/python -m pytest tests/atomic_probe/test_atomic_probe.py \
      -m requires_hardware --platform a5 --device \$TASK_DEVICE -v --clone-protocol https"
```

`ascendc/_run_asc_probe.sh` 和 `ccec/run_all.sh` 都输出 UTC 时间、git SHA、CANN 路径、编译器版本、
device 与 timeout。保存原始证据时直接对上面的 `task-submit` 命令做 `2>&1 | tee <log>`，并保留退出码。
经用户明确授权直接占用设备时，也可设置 `ATOMIC_PROBE_DEVICE=<id>` 后运行这两个 runner；日志必须注明
未经过 `task-submit`，并记录 dirty worktree，不能只记录 base SHA。

## 充分性判定

套件只有同时满足以下条件，才能称为“足够”：

- AscendC 与 CCEC 的 AIV-only 参与计数和 marker 均符合预期；
- 1/2/4/8B × same/separate-line 全部精确通过；
- 同-line regression gating 与 separate-line、逐轮 DSB control 明确分离；
- safe path、ordered hazard、unordered observation 三类不混淆；
- CPU control 无 C++ data race，并验证同 line 不影响正确性；
- 全量 runner 不漏文件、不吞错误；
- 新提交对应的 A5 原始日志与环境元数据可追溯。

当前工作区尚未通过上述充分性判定：AscendC/CCEC 的同-line regression gating 均在 device 0
稳定复现 mismatch 并返回非零；这正是该测试当前要暴露的问题。其余已执行的精确 control、矩阵与
CPU control 通过。结论只覆盖本文件 goal；AIC/MIX 比例和未测试的数据宽度/拓扑不能由此外推。
