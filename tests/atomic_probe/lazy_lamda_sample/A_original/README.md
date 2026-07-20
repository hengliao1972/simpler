# A：original eager

本目录是一份可单独复制、构建和查看的原始版样例，不依赖 B/C 目录或上级目录中的源码、脚本与二进制。
完整 task 流为 `Alloc -> QK -> SF -> PV -> UP`，32 AIC + 64 AIV。

## 固定形态

- `VARIANT=original`；`ccec/build.sh` 会拒绝其他 shape；
- 不定义 `PA_LAZY_SAMPLE_SHAPE_ID`，不定义 `PA_LAZY_SAMPLE_SPLIT_FINISH`；
- `PA_BUILD_SWIMLANE=1`、`PA_BUILD_SUBMIT_PMU=0`，运行时显式 `--no-swimlane`；
- 每个 task 先在 Submit 外执行 eager `Build*Args`，再调用原始 `SubmitTask`；
- `SubmitTask` 内为 `EfDrain -> Materialize -> TensorMap retire -> Claim`，之后保持原始的
  winner fanin、全员 register、winner Build/Complete 流程。Alloc 的 register/Claim 次序见源码注释。

主路径源码在 `common/pa_scheduler_core.h` 的 `SubmitTask` 和默认 replay 分支；固定构建参数在
`build.sh` 与 `ccec/build.sh` 顶部。未删除的 inactive `#if` 分支只用于保留来源上下文，固定构建入口无法选择它。

## 独立交付内容

- `common/`、`ccec/`：本版完整源码；
- `build/original/`：清空历史目录后生成的完整本地编译结果和中间对象；
- `artifacts/measured/`：正式 24 样本性能测试实际使用的 host/final ELF；
- `artifacts/rebuilt/`：当前同一 clean-build 的发布副本；
- `artifacts/runtime_identity.json`：两套 ELF 的运行时节、NOBITS 布局、运行时符号和 host 身份校验；
- `disassembly/raw/`：final-linked 机器码权威解码；
- `disassembly/annotated/`：每个函数的完整源码注释版反汇编；
- `disassembly/key_flow/aic_original_submit.source.asm`：便于直接阅读的未压缩关键窗口；
- `output/functional.txt`：最新 device0 b1 功能门禁；
- `output/performance/`：24 次独立 b256 原始日志、运行 provenance、样本表和统计；
- `source.sha256`、`build/published.sha256`、`output/published.sha256`、
  `MANIFEST.sha256`：源码、干净构建、输出与整包校验入口。

Measured final `.text` 为 `780344 B`，SHA256
`018ae7dc29ce78249b2e3bff84b0faa2e671e2448c9316719fb48bc59139d2fc`。
去异常 Submit span 中位数为 `3956.298 us`（原始 `24` 个样本，保留 `22` 个）。

## 复跑

先 source CANN 9.1 环境，使 `ASCEND_HOME_PATH` 有效：

```bash
./build.sh                     # 先精确删除 build/original，再完整构建
./publish_artifacts.sh         # 只刷新 artifacts/rebuilt
./verify_runtime_identity.py   # 对比 measured/rebuilt 的运行时身份
./run_functional.sh            # device0、b1、PMU off、--no-swimlane
./generate_disassembly.sh      # 从 measured ELF 重建 raw + annotated 全量反汇编
./summarize_performance.py     # 重新校验本目录 24 个独立正式样本
```

`*.source.asm.gz` 用 `zless` 阅读。注释来自 DWARF 行号回查本目录源码；注释不是 ELF 中保存的文本，
文件头已明确标注这一证据边界。
