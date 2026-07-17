# AICore 嵌套 Lambda、捕获与 caller 栈地址验证

> 最后验证日期：2026-07-17
>
> 验证仓库：`simpler`，基线 HEAD `52ca4f5eba343c2f7b7a3a743e575cb9308d128f`
>
> 测试目录：`tests/atomic_probe`
>
> 验证平台：A5 `Ascend950PR_9599`、CANN 9.1.0

## 1. 文档目的和最终结论

本文是一份可以独立阅读和执行的复现手册。读者不需要先了解此前的排查过程，
按本文即可完成以下验证：

1. CPU、AscendC 和纯 CCEC 是否支持嵌套 lambda、捕获和模板调用；
2. 是否能在 A5 上复现 caller 栈地址经过 runtime 调用后使用时的异常；
3. “删除独立 context，把 capture 写入现有 `L0TaskArgs`，由 runtime TU
   直接读取”的数据驱动方案是否可行；
4. CPU 对等实现是否具有相同功能，以及是否存在明显生命周期或未定义行为。

本次结果为：

- CPU、AscendC AIV、纯 CCEC AIV 的嵌套 lambda/捕获/模板基线全部通过；
- 纯 CCEC AIC 的 `weak-context-materialize-0` 在最终测试版本上 5/5 次触发
  `507015 AICore exception`；
- 同一版探针中的 `args-runtime-read` 在 A5 上 11/11 次通过，累计
  704 轮、2816 次 submit；
- CPU caller 与 AIC 共用同一份 runtime TU 源码，功能验证通过；GCC
  `-O0/-O2/-O3` 各 20/20 次通过，ASan+UBSan 通过。

必须保留的结论边界：业务用例是 1/2 条无业务地址物化仍失败，第 3 条
`block_table` 地址物化后通过；当前纯 CCEC 探针是 0 条失败、1/2/3 条通过。
因此本文已经复现“caller 栈地址传递对 HiIPU 最终 codegen 敏感”的核心现象，
但不是与业务阈值和机器码完全相同的最小复现。

当前设备状态也必须单独记录：最后一次负向复核之后，后续正向检查在
`aclInit(nullptr)` 阶段返回 500000；用户已确认当前 NPU 设备不存在。本文此前
的 A5 数据是在设备仍存在时取得，现在不能继续上板验证。没有芯片或驱动侧
故障日志，不能仅凭时间先后断言是本 probe 导致设备物理掉线。

## 2. 最短复现路径

如果只想最快确认结果，执行本节即可。后续章节解释原理、源码和每个测试的
完整判定规则。

### 2.1 设置路径

将前两行替换为本机实际路径：

```bash
export REPO=/path/to/simpler
export CANN_ROOT=/path/to/cann-9.1.0

cd "$REPO"
source "$CANN_ROOT/set_env.sh"
export PTO_ISA_ROOT="$CANN_ROOT/x86_64-linux"
export ATOMIC_PROBE_DEVICE=0
```

确认关键文件和工具存在：

```bash
test -x "$ASCEND_HOME_PATH/bin/ccec"
test -x "$ASCEND_HOME_PATH/bin/bisheng"
test -x "$ASCEND_HOME_PATH/bin/ld.lld"
test -f "$PTO_ISA_ROOT/include/pto/common/kernel_meta.hpp"
test -f tests/atomic_probe/ccec/nested_lambda_cross_tu.cpp
test -f tests/atomic_probe/ccec/nested_lambda_cross_tu_runtime.cpp
test -f tests/atomic_probe/cpu/nested_lambda_args_runtime_read.cpp
```

### 2.2 CPU 对等功能验证

```bash
tests/atomic_probe/run_nested_lambda.sh cpu
```

关键期望输出：

```text
[ASSERT] CPU nested capture/template semantics                PASS
[VALUES] rounds=64 mismatches=0 checksum=0x3e6cd1b792bff0e0 L0TaskArgs=1024B
[ASSERT] CPU L0TaskArgs args-runtime-read semantics PASS
[SUMMARY] semantic_failures=0
```

### 2.3 编译 AIC 双 TU 探针

```bash
tests/atomic_probe/ccec/run_all.sh nested_lambda_cross_tu build
```

命令必须以 0 退出，并生成：

```text
tests/atomic_probe/ccec/build/nested_lambda_cross_tu_kernel_caller_aic.o
tests/atomic_probe/ccec/build/nested_lambda_cross_tu_kernel_runtime_aic.o
tests/atomic_probe/ccec/build/nested_lambda_cross_tu_kernel.o
tests/atomic_probe/ccec/build/nested_lambda_cross_tu_host
```

### 2.4 运行可落地方案组

```bash
ATOMIC_PROBE_MODE=args-runtime-read \
  tests/atomic_probe/ccec/run_all.sh nested_lambda_cross_tu run
```

关键期望输出和退出码：

```text
[ASSERT] CCEC AIC cross-TU ABI variant=args-runtime-read PASS
[VALUES] rounds=64 mismatches=0 dispatches=0 materializations=3 \
checksum=0x3e6cd1b792bff0e0 L0TaskArgs=1024B
[SUMMARY] semantic_failures=0
```

退出码必须为 0。

### 2.5 运行故障对照组

该组会故意触发 AICore 异常。必须使用独立进程，并遵守所在环境的设备独占和
异常恢复规则。

```bash
set +e
ATOMIC_PROBE_MODE=weak-context-materialize-0 \
  tests/atomic_probe/ccec/run_all.sh nested_lambda_cross_tu run
bad_rc=$?
set -e
echo "bad_rc=$bad_rc"
```

本次受影响编译器上的期望结果：

```text
ACL error 507015 from aclrtSynchronizeStream(stream) ...
CCEC [nested_lambda_cross_tu_kernel] failed runs: 1
=== Done. run_failures=1 ===
bad_rc=1
```

此处退出码 1 是“故障复现成功”，不是 launcher 构建失败。如果该组返回 0，
说明当前编译器或当前代码布局没有命中此复现边界，不能据此认为方案组失败。

## 3. 被验证的问题是什么

### 3.1 原始业务现象

业务中的 private-lazy 调用会在 orchestration 栈上构造多个 `Tensor`，再把
这些对象的地址放入一个 caller context，经过 runtime 调用后继续使用：

```text
orchestration caller
  -> runtime submit
  -> dispatcher/recipe bind
  -> 使用 caller 栈上的 Tensor*
```

已有业务证据为：

- 优化后的 LLVM IR 正确写入三个 `Tensor *` context 字段；
- orchestration CFA 为 `reg93 + 1952`，栈帧只有 1952B；
- 32 KiB 和 64 KiB 栈配置都失败，所以不是普通栈容量不足；
- 不参与业务的地址写会改变 PASS/FAIL；
- 去掉 `-cce-aicore-addr-transform` 后业务仍失败，不能归罪于单一 pass；
- 同 TU、可内联 dispatcher 仍失败，说明跨 TU 外部回调不是根因；
- 业务原发日志包含 code 264：scalar 访问使用了无效 GM 地址；外层最终表现为
  `507018 AICPU exception`。

### 3.2 当前最合理的出错原理

坏路径可以简化为：

```text
reg93 栈
  ├─ Tensor first
  ├─ Tensor second
  ├─ Tensor third
  ├─ CallerContext {&first, &second, &third, salt}
  └─ L0TaskArgs args
            │
            └─ runtime(site_id, &context, &args)
                         │
                         └─ 再解引用 context 中的 Tensor*
```

LLVM IR 已经正确，栈容量也足够。可疑区间位于优化后 LLVM IR 到 HiIPU 最终
机器码之间，包括栈地址物化、指令选择、活跃区间、调度和寄存器分配。
`&Tensor` 来自 `reg93 + offset`；如果其地址寄存器在调用边界被错误复用、
覆盖或以错误偏移传递，runtime 后续解引用就会访问无效地址。

无业务用途的 `ptrtoint/store` 会改变地址的活跃区间和寄存器选择，所以即使
LLVM 业务语义完全不变，最终机器码也可能从失败形态切换到通过形态。这是
Heisenbug 诊断特征，不是合法修复方式。

目前尚未证明是某一个具体 CCEC pass。本文只把问题边界收敛到 HiIPU 后端的
地址物化/调度/寄存器分配组合，不声称已经完成编译器根因定位。

### 3.3 数据驱动方案为什么可能绕开问题

方案组不再构造独立 `CallerContext`：

```text
caller 栈上的 Tensor 地址
  -> 写入既有 L0TaskArgs 固定 slot
  -> 跨调用只传稳定使用的 args_ptr
  -> runtime TU 直接读取 slot
  -> submit 返回前 add_input 并完成物化
```

它减少了一条独立 caller 栈指针传参链，并迫使三个地址在调用前形成明确的
内存表示。它是对易错 codegen 形态的规避，不是对 CCEC 后端的修复。

## 4. 测试分层与调用结构

### 4.1 第一层：嵌套 lambda/捕获/模板语言基线

共享测试形态为：

```text
自由函数模板 Submit(outer_lambda)
  -> outer lambda 按引用捕获 caller 状态
  -> SubmitBuilder::AddInput(inner_lambda)
  -> SubmitBuilder::AddOutput(inner_lambda)
  -> SubmitBuilder::AddScalar(inner_lambda)
```

具体覆盖：

- outer lambda：`[&]`；
- input lambda：`[&]`，修改引用捕获；
- output lambda：`[outer_local, &output_calls]`，混合按值/按引用捕获；
- scalar lambda：按值和按引用混合捕获；
- 自由函数模板 `Submit<BuildCallback>`；
- 成员函数模板 `AddInput/AddOutput/AddScalar<Thunk>`；
- AICore 版本的 lambda 显式标注 `__aicore__`。

这一层分别由 CPU、AscendC AIV 和纯 CCEC AIV 实现。它只能回答“语言和
单 TU 语义是否支持”，不能单独复现 caller 栈地址问题。

### 4.2 第二层：caller capture 传输 A/B

第二层直接复用仓库中的真实类型：

```text
L0TaskArgs
Tensor
TaskOutputTensors
```

没有另造简化容器，也没有链接完整 Simpler runtime。探针只复用类型定义，
用单独的 `nested_lambda_cross_tu_runtime.cpp` 模拟待测 runtime 调用边界。
ACL 只负责加载 raw AICore ELF、启动 kernel 和回读结果，不参与被测 ABI。

每个变体执行：

```text
64 轮
  每轮在 caller 栈上新建 3 个 Tensor
  复用同一个 L0TaskArgs
  第 1 次 submit：待测 lazy/capture 路径
  第 2~4 次 submit：控制路径
```

这保留了“同一容器四次 submit”和 caller 栈对象反复创建/销毁的关键形态。

### 4.3 `L0TaskArgs` recipe slot 布局

测试专用布局为：

| slot | 内容 | 读取者 |
| ---: | --- | --- |
| scalar 8 | `&first` | runtime 或 dispatcher |
| scalar 9 | `&second` | runtime 或 dispatcher |
| scalar 10 | `&third` | runtime 或 dispatcher |
| scalar 11 | `salt` | runtime 或 dispatcher |
| scalar 0 | runtime 计算结果 | caller oracle |
| scalar 5 | dispatcher 次数 | caller oracle |
| scalar 6 | dispatcher 缺失标志 | caller oracle |

8~11 是本探针的 recipe/诊断 slot，不属于生产 ABI。真实落地必须为 recipe
定义正式布局、容量和版本，不能直接把这些编号当成业务规范。

### 4.4 七个 AIC 变体

所有变体都是独立 global AIC kernel，地址物化数量在编译期固定，不是运行时
分支。数字 entry 必须唯一。

| entry | 运行参数 | 独立 context | capture 读取位置 | dispatcher | 地址写 | 本次 A5 |
| ---: | --- | --- | --- | --- | ---: | --- |
| 0 | `weak-context-materialize-0` | 有 | caller dispatcher | weak | 0 | 507015 |
| 1 | `weak-context-materialize-1` | 有 | caller dispatcher | weak | 1 | PASS |
| 2 | `weak-context-materialize-2` | 有 | caller dispatcher | weak | 2 | PASS |
| 3 | `weak-context-materialize-3` | 有 | caller dispatcher | weak | 3 | PASS |
| 4 | `weak-args-storage` | 无 | caller dispatcher | weak | 3 个实际 capture | PASS |
| 5 | `strong-context` | 有 | caller dispatcher | strong | 0 | PASS |
| 6 | `args-runtime-read` | 无 | runtime TU 直接读取 | 无回调 | 3 个实际 capture | PASS |

四个 `materialize-N` 组中的 scalar 8~10 不参与业务读取，只用于改变地址物化。
`weak-args-storage` 中这些 slot 是实际 capture，但 runtime 仍回调 weak
dispatcher。`args-runtime-read` 才是最终待验证方案：没有第二个 context，
也没有反向回调。

### 4.5 精确 oracle

语言基线固定 `seed=0x120`：

| 字段 | 期望值 |
| --- | ---: |
| outer/input/output/scalar 调用次数 | 各 1 |
| `reference_state` | 300 |
| `input.value` | 591 |
| `output.value` | 323 |
| `scalar` | 337 |
| `combined` | 1251 |

caller capture 组要求：

- `completed_rounds == 64`；
- `mismatches == 0`；
- 64×4 次 submit 的 checksum 为 `0x3e6cd1b792bff0e0`；
- 回调组 dispatcher 次数为 128；`args-runtime-read` 为 0；
- 地址写数量和 variant echo 与 entry 完全一致；
- `sizeof(L0TaskArgs) == 1024`，小于 32 KiB 且按 64B 对齐。

## 5. 源码清单和复用关系

所有测试代码位于 `simpler/tests/atomic_probe`。

### 5.1 语言基线

| 文件 | 作用 |
| --- | --- |
| `nested_lambda_probe.h` | 三端共用的 builder、模板、字段布局和 oracle |
| `cpu/nested_lambda.cpp` | 标准 C++17 语言对照 |
| `ascendc/nested_lambda.asc` | AscendC AIV kernel 和 ACL host |
| `ccec/nested_lambda.cpp` | 纯 CCEC AIV kernel，无 AscendC API |
| `ccec/nested_lambda_host.cpp` | 纯 CCEC raw ELF launcher 和校验 |

### 5.2 caller capture 与 CPU 对等用例

| 文件 | 作用 |
| --- | --- |
| `ccec/nested_lambda_cross_tu.cpp` | AIC caller、dispatcher、七个 kernel entry |
| `ccec/nested_lambda_cross_tu_runtime.cpp` | 独立 runtime TU；含回调组和直接读取组 |
| `ccec/nested_lambda_cross_tu_api.h` | caller/runtime 共用函数签名 |
| `ccec/nested_lambda_cross_tu_layout.h` | 变体、entry、结果字段和 checksum oracle |
| `ccec/nested_lambda_cross_tu_host.cpp` | 按 entry 启动 raw ELF 并回读校验 |
| `cpu/nested_lambda_args_runtime_read.cpp` | CPU caller 对等实现 |
| `run_nested_lambda.sh` | CPU、AscendC、CCEC 统一入口 |
| `ccec/run_all.sh` | CCEC 编译、链接和独立进程矩阵 runner |
| `test_atomic_probe.py` | CPU 和 A5 pytest 入口 |

CPU 用例没有复制 runtime 实现。g++ 把
`cpu/nested_lambda_args_runtime_read.cpp` 与同一份
`ccec/nested_lambda_cross_tu_runtime.cpp` 作为两个 TU 编译，再通过
`--gc-sections` 裁掉 CPU 未调用的 dispatcher 控制函数。CPU 和 AIC 实际调用
的是同一个 `nested_probe_submit_args_runtime_read()` 源码函数。

## 6. 环境与构建前置条件

### 6.1 CPU

最低要求：

```text
bash
git
g++，支持 C++17
```

CPU 功能测试不需要 CANN runtime，也不访问设备，但会包含仓库现有的
`L0TaskArgs/Tensor` 头文件。

### 6.2 A5/CANN

要求：

```text
CANN 9.1.0
ccec
bisheng
ld.lld
g++
libascendcl.so
PTO kernel_meta.hpp
至少 1 个可用 A5 device
```

推荐检查：

```bash
echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
"$ASCEND_HOME_PATH/bin/ccec" --version | head -n 3
"$ASCEND_HOME_PATH/bin/bisheng" --version | head -n 1
g++ --version | head -n 1
test -f "$ASCEND_HOME_PATH/x86_64-linux/lib64/libascendcl.so"
test -f "$PTO_ISA_ROOT/include/pto/common/kernel_meta.hpp"
```

### 6.3 负向用例执行约束

故障组会触发 AICore exception，必须一次只运行一个变体，并在独立 host 进程
中启动。host 强制要求 `<kernel.o> <variant>` 两个参数；runner 也为每个变体
创建独立进程，避免一个异常污染其他控制组的判定。

早期两组交替实验中，故障进程退出后方案组可以立即再次通过；但最后一次
负向复核后，稍后的正向检查在 `aclInit(nullptr)` 就返回 500000，用户随后
确认 NPU 设备已经不存在。因此不能把“进程退出即可恢复”写成通用结论；没有
故障恢复条件时不要执行负向组。当前环境不再执行任何硬件命令。

### 6.4 本探针与 32K/64K 配置的关系

本探针不是 `aclInit` 栈配置测试：host 使用 `aclInit(nullptr)`，不读取额外
ACL JSON。CCEC runner 的 AIC 编译参数包含：

```text
-O3 -g -x cce -std=c++17 --cce-aicore-only
-mllvm -cce-aicore-stack-size=0x8000
-mllvm -cce-aicore-function-stack-size=0x8000
-mllvm -cce-aicore-record-overflow=false
-mllvm -cce-aicore-addr-transform
-mllvm -cce-aicore-dcci-insert-for-scalar=false
-mllvm -cce-aicore-dcci-before-kernel-end=false
--cce-aicore-arch=dav-c310-cube
```

`0x8000` 是 32 KiB 上限，但实际 orchestration CFA 只有 1952B。业务侧已经
独立验证 32K/64K 都失败，因此本文不再通过增大栈来解释或掩盖问题。

## 7. 逐项测试用例

以下命令除特别说明外，都从 `simpler` 仓库根目录执行。

### TC-CPU-01：CPU 语言基线与 runtime-read 对等实现

目的：

- 验证标准 C++17 的嵌套 lambda/捕获/模板语义；
- 验证真实 `L0TaskArgs/Tensor/TaskOutputTensors`；
- 用与 AIC 相同的 runtime TU 验证数据驱动 recipe 功能。

命令：

```bash
tests/atomic_probe/run_nested_lambda.sh cpu
```

判定：两个 `[ASSERT]` 均为 PASS，两个 summary 均为 0；runtime-read 的
rounds、mismatch、checksum 和 `L0TaskArgs` 大小必须与第 2.2 节一致。

pytest：

```bash
export PYTHONPATH=python
.venv/bin/python -m pytest \
  tests/atomic_probe/test_atomic_probe.py::test_cpu_nested_lambda_compiler_probe \
  -q -s
```

本次结果：`1 passed in 1.73s`。

### TC-CPU-02：CPU 优化级别和 sanitizer

目的：排除该 CPU 实现仅在某个优化级别偶然通过，以及明显的越界、
use-after-scope 或未定义行为。

下面的命令与本次实测一致：

```bash
INCLUDES=(
  -Itests/atomic_probe/ccec
  -Isrc/a5/platform/onboard/aicore
  -Isrc/a5/platform/include
  -Isrc/common/platform/include
  -Isrc/common/task_interface
  -Isrc/common/log/include
  -Isrc/common
  -Isrc/a5/runtime/fully_distributed_within_core/runtime
  -Isrc/a5/runtime/fully_distributed_within_core/common
  -Isrc/a5/runtime/fully_distributed_within_core/orchestration
  -Isrc/a5/runtime
)
SOURCES=(
  tests/atomic_probe/cpu/nested_lambda_args_runtime_read.cpp
  tests/atomic_probe/ccec/nested_lambda_cross_tu_runtime.cpp
)

for opt in 0 2 3; do
  out="/tmp/cpu_args_runtime_read_O${opt}"
  g++ "-O${opt}" -std=c++17 -Wall -Wextra -Werror -ffunction-sections \
    "${INCLUDES[@]}" "${SOURCES[@]}" -Wl,--gc-sections -o "$out"
  for i in $(seq 1 20); do "$out" >/dev/null; done
  echo "g++ -O${opt}: 20/20 PASS"
done

out=/tmp/cpu_args_runtime_read_sanitize
g++ -O2 -std=c++17 -Wall -Wextra -Werror -ffunction-sections \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  "${INCLUDES[@]}" "${SOURCES[@]}" -Wl,--gc-sections -o "$out"
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 "$out"
```

本次 GCC 13.3.0 结果：O0/O2/O3 各 20/20 PASS；ASan+UBSan PASS。本机没有
`clang++`，所以未完成第二种 CPU 后端交叉验证。

### TC-ASCENDC-01：AscendC AIV 语言基线

目的：验证 `bisheng -xasc` 对设备 lambda、捕获和模板调用的支持。

编译并上板：

```bash
ATOMIC_PROBE_DEVICE=0 tests/atomic_probe/run_nested_lambda.sh ascendc
```

仅编译、不访问设备：

```bash
bisheng -xasc tests/atomic_probe/ascendc/nested_lambda.asc \
  --npu-arch=dav-3510 \
  -o /tmp/nested_lambda_ascendc
```

上板期望：

```text
[ASSERT] AscendC nested capture/template semantics PASS
[ASSERT] AscendC ACL cleanup                      PASS
[SUMMARY] semantic_failures=0
```

### TC-CCEC-AIV-01：纯 CCEC AIV 语言基线

目的：不包含 AscendC header，只用 CCEC 和 lowercase builtin 验证同一语言
形态。

编译并上板：

```bash
ATOMIC_PROBE_DEVICE=0 \
  tests/atomic_probe/ccec/run_all.sh nested_lambda
```

拆分执行：

```bash
tests/atomic_probe/ccec/run_all.sh nested_lambda build
ATOMIC_PROBE_DEVICE=0 \
  tests/atomic_probe/ccec/run_all.sh nested_lambda run
```

期望：

```text
[ASSERT] CCEC nested capture/template semantics   PASS
[ASSERT] CCEC ACL cleanup                         PASS
[SUMMARY] semantic_failures=0
```

### TC-CCEC-AIC-01：双 TU build-only 和 ELF 结构

目的：确认 caller/runtime 分开编译、weak/strong 符号和七个入口都真实保留。

构建：

```bash
tests/atomic_probe/ccec/run_all.sh nested_lambda_cross_tu build
```

静态核对：

```bash
BUILD=tests/atomic_probe/ccec/build

readelf -S -W "$BUILD/nested_lambda_cross_tu_kernel.o" \
  | rg '\.ascend\.meta\.'

readelf -Ws -W "$BUILD/nested_lambda_cross_tu_kernel_runtime_aic.o" \
  | rg 'nested_probe_(weak_.*dispatch|strong_context_dispatch)'

readelf -Ws -W "$BUILD/nested_lambda_cross_tu_kernel.o" \
  | rg 'nested_probe_orchestration|nested_lambda_cross_tu_.*mix_aic'

"$ASCEND_HOME_PATH/bin/llvm-objdump" --dwarf=frames \
  "$BUILD/nested_lambda_cross_tu_kernel.o" \
  | rg 'reg93 \+1952'
```

期望：

- 七个 `.ascend.meta.*` section；
- runtime object 中两个 dispatcher 是 `WEAK UND`，strong dispatcher 是
  `GLOBAL UND`；
- context orchestration text 为 1432/1436/1440/1444B；
- `args-runtime-read` orchestration text 为 1484B；
- 七个 orchestration 的 CFA 均为 `reg93 + 1952`；
- 七个 global wrapper 均为 128B。

### TC-CCEC-AIC-02：数据驱动方案正向验证

目的：验证没有独立 context、没有反向 dispatcher 的 runtime-read 路径。

```bash
ATOMIC_PROBE_MODE=args-runtime-read \
ATOMIC_PROBE_DEVICE=0 \
  tests/atomic_probe/ccec/run_all.sh nested_lambda_cross_tu run
```

严格判定：

```text
rounds=64
mismatches=0
dispatches=0
materializations=3
checksum=0x3e6cd1b792bff0e0
L0TaskArgs=1024B
semantic_failures=0
进程退出码=0
```

pytest 会重新 build 再运行方案组：

```bash
export PYTHONPATH=python
ATOMIC_PROBE_DEVICE=0 .venv/bin/python -m pytest \
  tests/atomic_probe/test_atomic_probe.py::test_a5_ccec_nested_lambda_args_runtime_read \
  -q -s
```

### TC-CCEC-AIC-03：独立 context 故障复现

目的：证明同一编译器和同一测试版本仍能触发 caller-context 异常，避免因为
所有变体都通过而错误宣称方案有效。

```bash
set +e
ATOMIC_PROBE_MODE=weak-context-materialize-0 \
ATOMIC_PROBE_DEVICE=0 \
  tests/atomic_probe/ccec/run_all.sh nested_lambda_cross_tu run
rc=$?
set -e
echo "rc=$rc"
```

本机期望：`aclrtSynchronizeStream` 返回 507015，runner 和最终命令退出码都为
1。由于异常发生在同步阶段，host 的普通成功清理路径不会执行；必须依赖独立
进程和平台认可的异常恢复流程。

最新一次文档复核确实再次得到 507015；之后的正向恢复检查在
`aclInit(nullptr)` 返回 500000，当前 NPU 已不再存在。只有具备平台恢复能力
时才应继续执行本负向用例。

### TC-CCEC-AIC-04：完整七变体矩阵

runner 会把每个变体放在独立进程中，并把故障组放到最后，防止一个 AICore
exception 遮住其他控制组。

```bash
unset ATOMIC_PROBE_MODE
set +e
ATOMIC_PROBE_DEVICE=0 \
  tests/atomic_probe/ccec/run_all.sh nested_lambda_cross_tu run
matrix_rc=$?
set -e
echo "matrix_rc=$matrix_rc"
```

固定运行顺序：

```text
strong-context                    PASS
args-runtime-read                 PASS
weak-args-storage                 PASS
weak-context-materialize-3        PASS
weak-context-materialize-2        PASS
weak-context-materialize-1        PASS
weak-context-materialize-0        507015
```

本机 `matrix_rc=1` 是期望结果。该显式诊断 probe 不在 `ccec/run_all.sh` 默认
cache-line suite 中。

### TC-CCEC-AIC-05：方案组稳定性

先完成 build-only，再用独立进程重复运行：

```bash
for i in $(seq 1 10); do
  echo "=== args-runtime-read iteration=$i/10 ==="
  ATOMIC_PROBE_MODE=args-runtime-read \
  ATOMIC_PROBE_DEVICE=0 \
    tests/atomic_probe/ccec/run_all.sh nested_lambda_cross_tu run
done
```

要求 10/10 进程退出码为 0，且每次 checksum 完全一致。正式稳定性批次为
10/10；文档复核又成功运行 1 次，因此累计记录为 11/11、704 轮、2816 次
submit。

可选的交替 A/B：

```bash
for cycle in 1 2; do
  set +e
  ATOMIC_PROBE_MODE=weak-context-materialize-0 \
  ATOMIC_PROBE_DEVICE=0 \
    tests/atomic_probe/ccec/run_all.sh nested_lambda_cross_tu run
  bad_rc=$?
  set -e

  if [ "$bad_rc" -eq 0 ]; then
    echo "bad control unexpectedly passed" >&2
    exit 1
  fi

  # 若所在平台要求 reset，请在这里执行平台批准的恢复流程。
  ATOMIC_PROBE_MODE=args-runtime-read \
  ATOMIC_PROBE_DEVICE=0 \
    tests/atomic_probe/ccec/run_all.sh nested_lambda_cross_tu run
done
```

早期两组交替均为“context 507015，随后方案组立即 PASS”。这只证明当时两次
可以恢复，不覆盖最后一次设备消失的状态。

## 8. 实测环境和结果记录

### 8.1 软件和设备

```text
Repo HEAD: 52ca4f5eba343c2f7b7a3a743e575cb9308d128f
Device: Ascend950PR_9599
Short SoC: Ascend950
CANN: 9.1.0
ccec: clang 15.0.5, build 2026-06-10T11:29:46+08:00
GCC: 13.3.0
AscendC arch: dav-3510
CCEC AIV arch: dav-c310-vec
CCEC AIC arch: dav-c310-cube
```

### 8.2 A5 矩阵

| 变体 | 次数 | 结果 | rounds | mismatches | dispatcher | checksum |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `args-runtime-read` | 11 | 11/11 PASS | 64/次 | 0 | 0 | 固定正确 |
| `weak-context-materialize-0` | 5 | 5/5 507015 | 未回读 | 未回读 | 未回读 | 未回读 |
| 其他五个控制组 | 各至少 1 | PASS | 64 | 0 | 128 | 固定正确 |

故障组在 `aclrtSynchronizeStream` 就返回，因此不能伪造 rounds/mismatch 等设备
结果；表中明确写“未回读”。

最新状态：第 5 次负向命中后，后续 `args-runtime-read` 没有进入 kernel，
而是在 `aclInit(nullptr)` 返回 500000。用户确认当前 NPU 设备已经不存在。
因此 A5 数据到此冻结，后续仅做离线审查。

### 8.3 最终 build 产物哈希

哈希用于确认当前机器上的复现产物，不应假设不同绝对源码路径或不同工具版本
一定生成同一哈希。

```text
caller AIC object:
2412dd11bf0d8b07d8c9f9fd08179cc4821e95335616f314a2f55b4b710f52b0

runtime AIC object:
ec365301d026dbefec93d562102f7b85b44c72aa8ade2cea4956eb6e3f09470e

linked raw AICore ELF:
74167ad081799b93daa29a7b974746339cc4a3cd488dd2463514097696fb900b

host launcher:
未记录。host 入口已收紧为必须显式指定单个变体，修改后未重新执行 build-only。
```

## 9. 如何解释结果

| 观察 | 可以支持的结论 | 不能推出的结论 |
| --- | --- | --- |
| CPU/AscendC/CCEC AIV 全通过 | 前端和单 TU 语言语义支持 | 真实跨调用 ABI 一定正确 |
| context 组 507015、方案组通过 | 数据驱动方案避开当前易错形态 | 已定位某个具体 CCEC pass |
| 地址写数量改变 PASS/FAIL | 最终 codegen 对地址活跃区间敏感 | “加一条/三条 store”是合法修复 |
| strong 通过、weak 失败 | weak 会影响当前最小代码布局 | weak 是业务根因；同 TU 实验已反证 |
| 所有七组都通过 | 当前工具链/布局未命中 probe | 原业务问题不存在 |
| `args-runtime-read` 失败 | 当前方案在该工具链下不可用或测试有回归 | 一定与原业务是同一 fault PC |

### 9.1 507015 与 507018

CANN 安装头文件 `include/acl/error_codes/rt_error_codes.h` 定义：

```text
507015 = ACL_ERROR_RT_AICORE_EXCEPTION
507018 = ACL_ERROR_RT_AICPU_EXCEPTION
```

本文探针由 ACL 直接启动 AIC kernel，所以原发异常在 host 侧表现为 507015。
业务经过 AICPU 外层时报告 507018。错误层级符合调用路径差异，但仅凭两个
错误码不能证明 fault PC 完全相同。

### 9.2 CPU 是否也有栈地址物化缺陷

CPU 同样需要物化栈地址，但当前 x86-64 GCC 后端在这条双 TU 调用链上没有
观察到缺陷：多优化级别和 sanitizer 都通过。这支持问题是 CCEC HiIPU 后端
特有，而不是通用 C++ 语义错误。

这不是对所有 CPU 编译器的普遍证明。本机没有 `clang++`，也没有覆盖所有
代码布局。如果 runtime 在 submit 返回后仍保存并异步解引用 caller 栈地址，
CPU 也会发生 use-after-return；那是生命周期错误，不是地址物化缺陷。

## 10. 方案落地约束

当前结果证明“机制可行”，不等于真实 PA 业务修复已经完成。生产改造至少要
满足：

1. runtime 必须在 submit 返回前把 recipe/capture 复制或物化到自己的稳定
   存储，不能把 caller 栈裸地址留给异步阶段；
2. recipe slot 必须定义正式布局、容量、版本和边界检查；
3. 不能让 recipe slot 与正常 `scalar_count`、tensor slot 或后续 ABI 演进冲突；
4. 必须在真实 PA 上保留 eager control、原 lazy bad 和数据驱动 recipe 三组
   A/B，并检查 golden；
5. 必须覆盖真实的 64 轮、同容器四次 submit、多进程稳定性和异常恢复；
6. 无业务地址写只能用于诊断，不能作为正式修复提交；
7. 负向 507015 probe 只留在显式诊断矩阵，默认 CI 只运行正向
   `args-runtime-read`。

## 11. 常见问题与排查

### 11.1 `ASCEND_HOME_PATH` 或 `PTO_ISA_ROOT` 未设置

重新执行：

```bash
source "$CANN_ROOT/set_env.sh"
export PTO_ISA_ROOT="$CANN_ROOT/x86_64-linux"
```

不要把 `PTO_ISA_ROOT` 指到不含
`include/pto/common/kernel_meta.hpp` 的目录。

### 11.2 `aclrtBinaryGetFunction` 返回 107000

raw AICore ELF 的多入口不要使用带 `_mix_aic` 的完整符号名查找。本文已经按
CANN runtime 头文件的定义处理：`funcEntry` 是 kernel 名中的数字后缀，七个
入口使用唯一的 `0..6`，host 调用 `aclrtBinaryGetFunctionByEntry`。

当前正确命名示例：

```text
nested_lambda_cross_tu_ctx_m0_0_mix_aic
nested_lambda_cross_tu_ctx_m1_1_mix_aic
...
nested_lambda_cross_tu_runtime_args_6_mix_aic
```

本机官方依据：

```text
$ASCEND_HOME_PATH/x86_64-linux/pkg_inc/runtime/runtime/rts/rts_kernel.h
funcEntry: the suffix number; kernel_foo_123 -> 123
```

### 11.3 完整矩阵退出码为 1

先检查是否只有最后的 `weak-context-materialize-0` 返回 507015。如果前六组
都是 PASS，那么退出码 1 是预期诊断结果。若 build、入口获取或方案组也失败，
则不是预期结果。

### 11.4 CCEC 报 `tensor.h` 的 unused variable warning

当前构建会从仓库已有 `tensor.h` 报一个 `buffer_elems` unused warning；本次
编译仍成功。不要把 warning 当作本 probe 的 semantic failure，也不要为了
本测试批量修改无关生产头文件。

### 11.5 故障组后设备不可用

停止继续运行，执行所在平台批准的设备 reset/recovery，再先跑
`strong-context` 或 `args-runtime-read` 控制组。本文不提供未经本机文档验证的
reset 命令。本文最终一次复核后正处于该状态：`aclInit=500000`，NPU 已不再
存在/暴露，所以没有继续执行任何硬件命令。

### 11.6 故障组意外通过

先确认：

- 使用的是 `dav-c310-cube`，不是 AIV；
- caller/runtime 确实分开编译；
- 没有复用旧 build；
- CCEC 版本和 flags 与第 6.4 节一致；
- entry 0 启动的是 `weak-context-materialize-0`；
- source tree 包含本文列出的测试版本。

重新执行 build-only 后再运行。如果仍通过，应记录为“当前环境未复现”，不能
人为添加无关代码强迫其失败。

## 12. 最终复现检查清单

完成下列检查即可认为复现记录完整：

- [ ] 记录 repo HEAD、CANN/ccec/GCC 版本和设备型号；
- [ ] CPU 嵌套 lambda 基线 PASS；
- [ ] CPU runtime-read 对等实现 PASS；
- [ ] AscendC AIV 基线 PASS；
- [ ] 纯 CCEC AIV 基线 PASS；
- [ ] AIC caller/runtime 双 TU build PASS；
- [ ] 七个 metadata entry 和 weak/strong 符号符合预期；
- [ ] orchestration CFA 为 `reg93 + 1952`；
- [ ] `args-runtime-read` 为 PASS、checksum 精确匹配；
- [ ] `weak-context-materialize-0` 在受影响环境返回 507015；
- [ ] 明确记录完整矩阵退出码 1 是预期负向命中；
- [ ] 异常后按平台规范完成恢复；
- [ ] 没有把地址物化条数或 weak 属性误写成最终根因；
- [ ] 没有把机制验证误写成真实业务修复完成。
