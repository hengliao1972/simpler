# B：compete-first eager

本目录是一份可单独复制、构建和查看的 compete-first eager 样例，不依赖 A/C 目录或上级目录中的源码、
脚本与二进制。完整 task 流为 `Alloc -> QK -> SF -> PV -> UP`，32 AIC + 64 AIV。

## 固定形态

- `VARIANT=compete-first`，`PA_LAZY_SAMPLE_SHAPE_ID=1`；`ccec/build.sh` 会拒绝其他 shape；
- `PA_LAZY_SAMPLE_SPLIT_FINISH=1`，finish 为 noinline cross-TU；
- `PA_BUILD_SWIMLANE=1`、`PA_BUILD_SUBMIT_PMU=0`，运行时显式 `--no-swimlane`；
- 控制流是 `Begin -> EfDrain -> Claim -> outer callback<false> -> Materialize -> TensorMap retire`
  `-> winner Fanin -> Register -> winner Build/Complete`；
- callback 内的 nested lambda 由 `Add*` 同步求值，不保存 thunk/closure；`Lazy=false`，所以每个 worker
  都求值全部 input/output/inout/scalar thunk。

主路径源码在 `common/pa_scheduler_core.h` 的 `SubmitLazySampleCallback`、
`BuildLazySampleCallbackArgs`，builder 在 `common/pa_frontend.h`。固定构建入口无法选择 A/C；
未删除的 inactive `#if` 分支只保留来源上下文。

## 独立交付内容

- `common/`、`ccec/`：本版完整源码；
- `build/compete-first/`：清空历史目录后生成的完整本地编译结果和中间对象；
- `artifacts/measured/`：正式 24 样本性能测试实际使用的 host/final ELF；
- `artifacts/rebuilt/`：当前同一 clean-build 的发布副本；
- `artifacts/runtime_identity.json`：measured/rebuilt 运行时身份校验；
- `disassembly/raw/`：final-linked 机器码权威解码；
- `disassembly/annotated/`：每个函数的完整源码注释版反汇编；
- `disassembly/key_flow/aic_compete_first.source.asm`：未压缩的 Claim/callback 关键窗口；
- `output/functional.txt`：最新 device0 b1 功能门禁；
- `output/performance/`：24 次独立 b256 原始日志、运行 provenance、样本表和统计；
- `source.sha256`、`build/published.sha256`、`output/published.sha256`、
  `MANIFEST.sha256`：源码、干净构建、输出与整包校验入口。

Measured final `.text` 为 `547640 B`，SHA256
`8fd6209b2f0f9d1fb48d4d8a16cf27649e26071153b908e1e0e39ca107d63589`。
去异常 Submit span 中位数为 `3744.417 us`（24 个样本均保留）。b1 精确前端计数为
`views/tensor/scalar/resets = 192/2112/864/384`。

## 复跑

```bash
./build.sh                     # 先精确删除 build/compete-first
./publish_artifacts.sh         # 只刷新 artifacts/rebuilt
./verify_runtime_identity.py
./run_functional.sh            # device0、b1、PMU off、--no-swimlane
./generate_disassembly.sh      # measured ELF 的 raw + annotated 全量反汇编
./summarize_performance.py
```

`*.source.asm.gz` 用 `zless` 阅读。DWARF 只提供地址到文件/行映射；显示的源码与原注释由脚本从本目录
源码复制，文件头不会把它们冒充成 ELF 自带注释。
