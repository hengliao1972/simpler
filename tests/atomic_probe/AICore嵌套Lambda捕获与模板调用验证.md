# AICore 嵌套 Lambda、caller 栈地址与 inline 调用边界验证

> 最后验证日期：2026-07-17
>
> 验证基线 HEAD：`3a3de54db0a900e489a5a5496cbee1dc5b76be7a`
>
> 测试目录：`tests/atomic_probe`
>
> 验证平台：A5 device 0、CANN 9.1.0

## 1. 文档目的和最终结论

本文验证四件事：

1. CPU、AscendC 和纯 CCEC 是否支持嵌套 lambda、捕获和模板调用；
2. caller 栈上的 `Tensor` 地址经过未内联 submit 调用后，是否会触发
   AICore 异常；
3. 异常是否由“链接两个 `.o`”或 `ld.lld` 跨对象重定位直接导致；
4. runtime submit 全 inline，以及把 capture 写入既有 `L0TaskArgs`，
   是否可作为当前工具链的规避形态。

本轮得到的核心矩阵如下：

| AIC 构建形态 | submit 形态 | 语义变体 | 本轮结果 |
| ------------ | ----------- | -------- | -------- |
| 原始双 `.o` | 外部 submit | m0 | 5/5 507015 |
| 双 `.o`，第二个 `.text=0` | 全 inline | m0 | 5/5 PASS |
| 单 `.o` | 全 inline | m0 | 5/5 PASS |
| 单 `.o` | 仅 weak-context submit noinline | m0 | 5/5 507015 |
| 同一 noinline fixture | 仅 weak-context submit noinline | m1 | 3/3 PASS |
| 同一 noinline fixture | strong 路径保持 inline | strong | 3/3 PASS |

因此可以排除“只要链接两个 `.o` 就会失败”。单 `.o` 保留
`nested_probe_submit_weak_context` 未内联时仍然失败，而双 `.o`
在第二个对象没有代码、submit 已完全内联时通过。

当前最小、可重复的触发边界是：

```text
weak-context-materialize-0
+ nested_probe_submit_weak_context 未内联
+ caller 栈地址跨该调用边界
-> 507015 AICore exception
```

全 inline 后 m0 通过；同一 noinline 产物增加一次无业务地址物化后也通过。
这仍然是对 HiIPU 最终 codegen 形态的收敛，不是对某个具体后端 pass 或
某条机器指令的根因定位。

需要严格保留以下结论边界：

- 不是 C++ lambda、捕获或模板语言语义不受支持；
- 没有证据表明 `ld.lld` 链接两个对象本身有错误；
- 不能把 weak、noinline 或零地址物化任一单因素写成充分根因；
- all-inline 是已验证规避，不是 CCEC 编译器修复；
- `args-runtime-read` 仍是避免独立 caller context 的数据驱动方案，
  与“submit 全 inline”是两个不同维度。

## 2. 最短复现路径

以下命令从仓库根目录执行。

### 2.1 环境

```bash
export REPO=/path/to/simpler
export CANN_ROOT=/path/to/cann-9.1.0

cd "$REPO"
source "$CANN_ROOT/set_env.sh"
export PTO_ISA_ROOT="$CANN_ROOT/x86_64-linux"
export ATOMIC_PROBE_DEVICE=0
```

确认工具和源码存在：

```bash
test -x "$ASCEND_HOME_PATH/bin/ccec"
test -x "$ASCEND_HOME_PATH/bin/ld.lld"
test -f "$PTO_ISA_ROOT/include/pto/common/kernel_meta.hpp"
test -f tests/atomic_probe/ccec/nested_lambda_cross_tu.cpp
test -f tests/atomic_probe/ccec/nested_lambda_cross_tu_api.h
test -f tests/atomic_probe/ccec/nested_lambda_only_weak_submit_noinline.cpp
```

### 2.2 CPU 语义和 inline runtime-read

```bash
tests/atomic_probe/run_nested_lambda.sh cpu
```

关键期望输出：

```text
[ASSERT] CPU nested capture/template semantics                PASS
[VALUES] rounds=64 mismatches=0 checksum=0x3e6cd1b792bff0e0
[ASSERT] CPU L0TaskArgs args-runtime-read semantics PASS
[SUMMARY] semantic_failures=0
```

CPU 和 AIC 都从 `nested_lambda_cross_tu_api.h` 使用同一份 inline
runtime 实现。CPU 不再链接独立 runtime TU。

### 2.3 默认单对象、全 inline 正向组

构建：

```bash
tests/atomic_probe/ccec/run_all.sh nested_lambda_cross_tu build
```

构建过程会检查：

```text
[ASSERT] CCEC caller-capture runtime symbol shape PASS
[VALUES] aic_input_objects=1 runtime_text=n/a submit_symbols=none
```

运行默认正向组：

```bash
tests/atomic_probe/ccec/run_all.sh nested_lambda_cross_tu run
```

固定顺序和期望结果：

```text
args-runtime-read                 PASS
weak-context-materialize-0        PASS
run_failures=0
```

也可以单独运行 m0：

```bash
ATOMIC_PROBE_MODE=weak-context-materialize-0 \
  tests/atomic_probe/ccec/run_all.sh nested_lambda_cross_tu run
```

严格 oracle 为：

```text
rounds=64
mismatches=0
dispatches=128
materializations=0
checksum=0x3e6cd1b792bff0e0
L0TaskArgs=1024B
进程退出码=0
```

### 2.4 双对象数量控制

该 fixture 仍向 `ld.lld` 传入两个 AIC 对象，但第二个对象的
`.text` 大小必须为零，且所有 submit 已进入 caller 对象并被内联。

```bash
tests/atomic_probe/ccec/run_all.sh \
  nested_lambda_inline_plus_empty_runtime build

tests/atomic_probe/ccec/run_all.sh \
  nested_lambda_inline_plus_empty_runtime run
```

构建期望：

```text
[VALUES] aic_input_objects=2 runtime_text=000000 submit_symbols=none
```

运行期望：m0 PASS、退出码 0。该控制证明两个链接输入对象本身不足以触发
异常。

### 2.5 only-weak-submit-noinline 故障控制

该 fixture 仍只编译一个有代码的 AIC 对象，但仅保留
`nested_probe_submit_weak_context` 为 noinline。其他 runtime submit、
digest、consume 和 control 均保持 `always_inline`。

```bash
tests/atomic_probe/ccec/run_all.sh \
  nested_lambda_only_weak_submit_noinline build
```

构建期望：

```text
[VALUES] aic_input_objects=1 runtime_text=n/a
submit_symbols=*nested_probe_submit_weak_context*
```

先运行通过控制：

```bash
ATOMIC_PROBE_MODE=strong-context \
  tests/atomic_probe/ccec/run_all.sh \
  nested_lambda_only_weak_submit_noinline run

ATOMIC_PROBE_MODE=weak-context-materialize-1 \
  tests/atomic_probe/ccec/run_all.sh \
  nested_lambda_only_weak_submit_noinline run
```

两条命令都必须退出 0。

最后运行故障控制：

```bash
set +e
ATOMIC_PROBE_MODE=weak-context-materialize-0 \
  tests/atomic_probe/ccec/run_all.sh \
  nested_lambda_only_weak_submit_noinline run
bad_rc=$?
set -e
echo "bad_rc=$bad_rc"
```

受影响 CCEC 上的期望结果：

```text
ACL error 507015 from aclrtSynchronizeStream(stream) ...
CCEC [...] failed runs: 1
bad_rc=1
```

这里的退出码 1 表示故障控制命中，不是构建失败。该组必须在独立 host
进程中运行，并放在所有通过控制之后。

## 3. 被验证的问题

### 3.1 原始调用形态

业务中的 private-lazy 路径会在 orchestration 栈上构造多个 `Tensor`，
再把对象地址写入 caller context：

```text
orchestration caller
  -> caller 栈上的 Tensor first/second/third
  -> CallerContext {&first, &second, &third, salt}
  -> submit(context, args)
  -> dispatcher 解引用 context 中的 Tensor*
```

已有证据包括：

- 优化后的 LLVM IR 正确写入三个 `Tensor *`；
- orchestration CFA 为 `reg93 + 1952`，不是普通栈容量不足；
- 32 KiB 和 64 KiB 栈配置都失败；
- 无业务用途的地址写会改变 PASS/FAIL；
- 去掉 `-cce-aicore-addr-transform` 后业务仍失败；
- 业务原发日志包含无效 GM 地址访问。

### 3.2 本轮新增的判别证据

本轮把“对象数量”和“函数调用边界”拆开：

```text
两个对象 + 空 runtime + submit 全 inline
  -> PASS

一个对象 + weak-context submit noinline
  -> 507015
```

因此多对象链接不是必要条件，未内联 submit 才是当前最小 fixture 中的
必要构建形态。它仍不是单独的充分解释，因为：

- 同一 noinline fixture 的 m1 通过；
- 同一 fixture 的 strong 路径通过；
- 全 inline 的 weak m0 通过。

最合理的范围仍是优化 LLVM IR 到 HiIPU 最终机器码之间的地址物化、
活跃区间、调度和寄存器分配组合。

当前安装的 `llvm-objdump` 能读取 `elf64-hiipu` 符号和 DWARF，但指令
反汇编显示 `<not available>`。没有可用 fault PC，因此本文不声称已经定位
具体机器指令。

### 3.3 两种规避形态

全 inline 形态：

```text
caller 栈地址
  -> inline submit/helper
  -> 不跨 nested_probe_submit_weak_context 函数边界
  -> dispatcher/consume
```

数据驱动形态：

```text
caller 栈地址
  -> 写入既有 L0TaskArgs 固定 slot
  -> inline args-runtime-read 直接读取
  -> submit 返回前完成 add_input 和结果物化
```

两者都在当前工具链上通过。生产实现仍必须保证 submit 返回前完成复制或
物化，不能让异步阶段继续保存 caller 栈裸地址。

## 4. 测试结构和 oracle

### 4.1 语言语义层

CPU、AscendC AIV 和纯 CCEC AIV 共同覆盖：

- outer lambda 按引用捕获；
- nested lambda 按值、按引用和混合捕获；
- 自由函数模板 `Submit<BuildCallback>`；
- 成员函数模板 `AddInput/AddOutput/AddScalar<Thunk>`；
- AICore lambda 的 `__aicore__` 标注。

固定 `seed=0x120` 的期望值为：

| 字段 | 期望值 |
| ---- | -----: |
| outer/input/output/scalar 调用次数 | 各 1 |
| `reference_state` | 300 |
| `input.value` | 591 |
| `output.value` | 323 |
| `scalar` | 337 |
| `combined` | 1251 |

### 4.2 caller-capture 语义层

每个变体执行 64 轮，每轮执行 4 次 submit：

- 第一次为待测 caller-capture 或 args-runtime-read 路径；
- 后三次为固定 control 路径；
- 每轮重新创建三个 caller 栈 `Tensor`；
- 同一轮复用一个 `L0TaskArgs`。

精确 oracle：

| 字段 | 期望 |
| ---- | ---: |
| completed rounds | 64 |
| mismatches | 0 |
| submits | 256 |
| checksum | `0x3e6cd1b792bff0e0` |
| `sizeof(L0TaskArgs)` | 1024B |
| callback dispatcher 次数 | 128 |
| args-runtime-read dispatcher 次数 | 0 |

`L0TaskArgs` 的测试 slot：

| slot | 内容 |
| ---: | ---- |
| scalar 8 | `&first` |
| scalar 9 | `&second` |
| scalar 10 | `&third` |
| scalar 11 | `salt` |
| scalar 0 | 计算结果 |
| scalar 5 | dispatcher 次数 |
| scalar 6 | dispatcher 缺失标志 |

这些编号仅属于探针，不是生产 ABI。

### 4.3 七个语义变体

| entry | 运行参数 | 独立 context | 地址物化 | dispatcher |
| ----: | -------- | ------------ | -------: | ---------- |
| 0 | `weak-context-materialize-0` | 有 | 0 | weak |
| 1 | `weak-context-materialize-1` | 有 | 1 | weak |
| 2 | `weak-context-materialize-2` | 有 | 2 | weak |
| 3 | `weak-context-materialize-3` | 有 | 3 | weak |
| 4 | `weak-args-storage` | 无 | 3 | weak |
| 5 | `strong-context` | 有 | 0 | strong |
| 6 | `args-runtime-read` | 无 | 3 | 无回调 |

变体本身不再绑定固定 PASS/FAIL。结果必须同时写明构建 fixture。尤其 m0
在 all-inline fixture 中通过，在 only-weak-submit-noinline fixture 中失败。

### 4.4 三个持久构建 fixture

| target | AIC 输入对象 | submit FUNC 符号 | 默认运行 |
| ------ | -----------: | ---------------- | -------- |
| `nested_lambda_cross_tu` | 1 | 0 | args-runtime-read、m0 |
| `nested_lambda_inline_plus_empty_runtime` | 2 | 0 | m0 |
| `nested_lambda_only_weak_submit_noinline` | 1 | 仅 weak-context submit | strong、m1、m0 |

第三个 target 的 m0 放在最后；预期 runner 退出 1。默认 pytest 不运行该
故障组。

## 5. 源码和测试入口

| 文件 | 作用 |
| ---- | ---- |
| `nested_lambda_probe.h` | 三端共享的语言语义和 oracle |
| `cpu/nested_lambda.cpp` | 标准 C++17 语言对照 |
| `cpu/nested_lambda_args_runtime_read.cpp` | CPU inline runtime-read 对照 |
| `ccec/nested_lambda_cross_tu.cpp` | AIC caller、dispatcher 和七个入口 |
| `ccec/nested_lambda_cross_tu_api.h` | inline runtime 实现和 caller context |
| `ccec/nested_lambda_cross_tu_runtime.cpp` | `.text=0` 的第二对象控制 |
| `ccec/nested_lambda_inline_plus_empty_runtime.cpp` | 双对象数量控制 wrapper |
| `ccec/nested_lambda_only_weak_submit_noinline.cpp` | only-weak-submit-noinline wrapper |
| `ccec/nested_lambda_cross_tu_layout.h` | 变体、字段和精确 oracle |
| `ccec/nested_lambda_cross_tu_host.cpp` | raw ELF launcher 和结果校验 |
| `ccec/run_all.sh` | 三种 fixture 的构建、ELF 断言和运行 |
| `run_nested_lambda.sh` | CPU、AscendC、CCEC 统一入口 |
| `test_atomic_probe.py` | CPU 和默认 A5 正向 pytest |

文件名中的 `cross_tu` 是历史命名。当前默认 target 是单个有代码的 AIC
输入对象，不能再根据文件名推断构建形态。

## 6. 自动化测试

### 6.1 CPU pytest

```bash
export PYTHONPATH=python
.venv/bin/python -m pytest \
  tests/atomic_probe/test_atomic_probe.py::test_cpu_nested_lambda_compiler_probe \
  -q -s
```

### 6.2 A5 默认正向 pytest

```bash
export PYTHONPATH=python
.venv/bin/python -m pytest \
  tests/atomic_probe/test_atomic_probe.py::test_a5_ccec_nested_lambda_call_boundary_controls \
  --platform a5 --device 0 \
  -q -s
```

该 pytest 会：

1. build 单对象 all-inline target，并验证 submit FUNC 符号为零；
2. 先运行 args-runtime-read，要求 PASS；
3. 再运行 m0，要求 PASS。

它不会运行 only-weak-submit-noinline m0，避免默认 CI 故意制造 AICore
exception。双对象数量控制和 noinline 故障控制使用第 2.4、2.5 节的显式
命令。

### 6.3 完整语言入口

```bash
tests/atomic_probe/run_nested_lambda.sh all
```

`all` 依次运行 CPU、AscendC、纯 CCEC AIV 和默认 AIC caller-capture
正向组。

## 7. 本轮实测结果

软件环境：

```text
Repo HEAD: 3a3de54db0a900e489a5a5496cbee1dc5b76be7a
CCEC: clang 15.0.5, build 2026-07-07T20:35:46+08:00
GCC: 13.3.0
AIC arch: dav-c310-cube
Device: A5 device 0
```

持久 fixture 结果：

| fixture / variant | 次数 | 结果 |
| ----------------- | ---: | ---- |
| 单对象 all-inline / m0 | 5 | 5/5 PASS |
| 双对象、空 runtime / m0 | 5 | 5/5 PASS |
| only-weak-submit-noinline / m1 | 3 | 3/3 PASS |
| only-weak-submit-noinline / strong | 3 | 3/3 PASS |
| only-weak-submit-noinline / m0 | 5 | 5/5 507015 |
| 单对象 all-inline / args-runtime-read | 1 | PASS |
| CPU 语言语义 | 1 | PASS |
| CPU inline args-runtime-read | 1 | PASS |

此外，本轮从当前 HEAD 临时恢复原始 external runtime 源码，重新生成双
`.o` 产物；m0 为 5/5 507015。该临时产物只用于确认历史基线，不是当前
runner 的持久 target。

故障组在 `aclrtSynchronizeStream` 返回 507015，不能回读 rounds、
mismatch 或 checksum。表中不为故障组伪造设备结果。

## 8. 如何解释结果

| 观察 | 可以支持 | 不能推出 |
| ---- | -------- | -------- |
| 双对象空 runtime 通过 | 两个链接输入不足以触发 | `ld.lld` 已被全面证明无缺陷 |
| 单对象 noinline m0 失败 | 多对象链接不是必要条件 | 任意 noinline 调用都会失败 |
| all-inline m0 通过 | inline 可规避当前形态 | inline 修复了 CCEC 后端 |
| noinline m1 通过 | 地址物化会改变最终 codegen | 增加一次 store 是生产修复 |
| 同 fixture strong 通过 | strong 路径是有效控制 | weak 单独就是根因 |
| args-runtime-read 通过 | 数据驱动方案机制可行 | 真实 PA 业务已经完成修复 |

507015 在 CANN 中表示 AICore exception。业务经过 AICPU 外层时可能报告
507018；错误层级与调用路径不同，不能只凭错误码断言 fault PC 相同。

## 9. 生产落地约束

1. submit 返回前必须复制或物化 caller 栈数据；
2. 不得让异步阶段保存 caller 栈裸地址；
3. all-inline 必须检查最终 ELF 不含 `nested_probe_submit_*` FUNC 符号；
4. inline 会改变代码体积，真实业务必须检查指令空间和性能；
5. recipe slot 必须定义正式布局、容量、版本和边界检查；
6. 真实 PA 必须保留 eager、原 lazy 和候选方案三组 A/B；
7. 无业务地址写只能用于诊断，不能作为正式修复；
8. only-weak-submit-noinline m0 只属于显式诊断，不进入默认 CI。

## 10. 常见问题

### 10.1 build 目录中混入旧产物

每个 fixture 使用独立 kernel、caller object 和 host 文件名。不要拿
`nested_lambda_cross_tu_host` 启动 noinline kernel，也不要只看 build
目录中是否残留历史 runtime object。

以 runner 的构建输出为准：

```text
aic_input_objects=...
runtime_text=...
submit_symbols=...
```

### 10.2 noinline 完整运行退出 1

以下命令默认按 strong、m1、m0 顺序运行：

```bash
tests/atomic_probe/ccec/run_all.sh \
  nested_lambda_only_weak_submit_noinline run
```

如果前两组 PASS，最后 m0 返回 507015，则最终退出 1 是预期诊断结果。

### 10.3 m0 意外通过或失败

先确认：

- 使用了正确 fixture 的 kernel；
- all-inline target 没有 submit FUNC 符号；
- noinline target 只保留 weak-context submit；
- CCEC arch 为 `dav-c310-cube`；
- caller orchestration 和七个 metadata entry 仍存在；
- 没有复用另一 fixture 的 host/kernel 组合。

若 noinline m0 仍通过，应记录“当前工具链未命中”，不能添加无关代码强迫
失败。

### 10.4 `tensor.h` unused warning

CCEC 当前会报告 `buffer_elems` 未使用。它不属于本 probe 的 semantic
failure，不应为了该测试修改无关生产头文件。

## 11. 最终检查清单

- [ ] CPU 嵌套 lambda 语义 PASS；
- [ ] CPU inline args-runtime-read PASS；
- [ ] 默认 AIC target 只有一个输入对象；
- [ ] 默认 ELF 没有 `nested_probe_submit_*` FUNC 符号；
- [ ] 默认 all-inline m0 为 5/5 PASS；
- [ ] 双对象控制的第二个对象 `.text=0`；
- [ ] 双对象控制 m0 为 5/5 PASS；
- [ ] noinline target 只保留 weak-context submit；
- [ ] noinline m1 为 3/3 PASS；
- [ ] 同 fixture strong 为 3/3 PASS；
- [ ] noinline m0 为 5/5 507015；
- [ ] args-runtime-read checksum 精确匹配；
- [ ] `sizeof(L0TaskArgs) == 1024`；
- [ ] 没有把 `.o` 数量、`ld.lld`、weak 或某个 pass 写成已定位根因；
- [ ] 默认 pytest 不运行故障组。
