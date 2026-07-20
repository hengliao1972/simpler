# C：compete-first + lazy lambda

本目录是一份可单独复制、构建和查看的 lazy lambda 样例，不依赖 A/B 目录或上级目录中的源码、脚本与
二进制。完整 task 流为 `Alloc -> QK -> SF -> PV -> UP`，32 AIC + 64 AIV。

## 固定形态

- `VARIANT=compete-first-lazy`，`PA_LAZY_SAMPLE_SHAPE_ID=2`；`ccec/build.sh` 会拒绝其他 shape；
- `PA_LAZY_SAMPLE_SPLIT_FINISH=1`，finish 为 noinline cross-TU；
- `PA_BUILD_SWIMLANE=1`、`PA_BUILD_SUBMIT_PMU=0`，运行时显式 `--no-swimlane`；
- 控制流与 B 完全同族：`Begin -> EfDrain -> Claim -> outer callback<true> -> Materialize`
  `-> TensorMap retire -> winner Fanin -> Register -> winner Build/Complete`；
- `AddLocalInput`、`AddGmInput`、`AddScalar` 在 `Lazy=true && !won` 时不调用 thunk；
- output/inout 仍由所有 worker 求值，以保持 Tier-1 样例的私有 heap/TensorMap 状态一致；
- nested lambda 只在 `Add*` 内同步求值，没有 thunk/closure 越过 outer callback 或 split-TU 边界。

lazy 判定的正式源码在 `common/pa_frontend.h` 的 `LazySampleCallbackArgsBuilder<Lazy>`，五类 task 的
outer callback 在 `common/pa_scheduler_core.h::BuildLazySampleCallbackArgs`。固定构建入口无法选择 A/B；
未删除的 inactive `#if` 分支只保留来源上下文。

## 独立交付内容

- `common/`、`ccec/`：本版完整源码；
- `build/compete-first-lazy/`：清空历史目录后生成的完整本地编译结果和中间对象；
- `artifacts/measured/`：正式 24 样本性能测试实际使用的 host/final ELF；
- `artifacts/rebuilt/`：当前同一 clean-build 的发布副本；
- `artifacts/runtime_identity.json`：measured/rebuilt 运行时身份校验；
- `disassembly/raw/`：final-linked 机器码权威解码；
- `disassembly/annotated/`：每个函数的完整源码注释版反汇编；
- `disassembly/key_flow/aic_compete_first_lazy.source.asm`：compete-first 主路径窗口；
- `disassembly/key_flow/aic_lazy_input_policy.source.asm`：直接包含
  `if constexpr (Lazy) / if (!won_) return` 源码注释及对应机器码的窗口；
- `output/functional.txt`：最新 device0 b1 功能门禁；
- `output/performance/`：24 次独立 b256 原始日志、运行 provenance、样本表和统计；
- `source.sha256`、`build/published.sha256`、`output/published.sha256`、
  `MANIFEST.sha256`：源码、干净构建、输出与整包校验入口。

Measured final `.text` 为 `547640 B`，SHA256
`866ed1081ebb94038a8feca87ccedf95e67aebfdaef11651c109f86672be2bdd`。
去异常 Submit span 中位数为 `3818.781 us`（原始 `24` 个样本，保留 `22` 个）。b1 精确前端计数为
`views/tensor/scalar/resets = 97/1162/9/384`。

## 复跑

```bash
./build.sh                     # 先精确删除 build/compete-first-lazy
./publish_artifacts.sh         # 只刷新 artifacts/rebuilt
./verify_runtime_identity.py
./run_functional.sh            # device0、b1、PMU off、--no-swimlane
./generate_disassembly.sh      # measured ELF 的 raw + annotated 全量反汇编
./summarize_performance.py
```

`*.source.asm.gz` 用 `zless` 阅读。DWARF 只提供地址到文件/行映射；显示的源码与原注释由脚本从本目录
源码复制，文件头不会把它们冒充成 ELF 自带注释。
