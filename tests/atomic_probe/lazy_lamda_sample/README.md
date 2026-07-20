# standalone full-task lazy lambda 三版独立样例

本目录用于把三种 PA submit 形态作为可直接交给他人的独立样例进行静态与动态对照。三份包都覆盖
`Alloc -> QK -> SF -> PV -> UP`，使用 32 AIC + 64 AIV；不是只改 QK。

根目录只保存三份包之间的公共说明。`A_original/`、`B_compete_first/`、
`C_compete_first_lazy/` 各自包含完整源码、固定构建入口、干净构建结果、正式测试所用二进制、
功能/性能输出以及带源码注释的反汇编；任意一份都可脱离另外两份单独复制和查看。

## 三份代码

| 包 | 固定编译形态 | 参数求值和 Submit 顺序 |
| --- | --- | --- |
| [A_original](A_original/README.md) | `original`，不定义 `PA_LAZY_SAMPLE_SHAPE_ID` | 所有 worker 在 Submit 外 eager 构参，再走原始 Submit |
| [B_compete_first](B_compete_first/README.md) | `PA_LAZY_SAMPLE_SHAPE_ID=1` | 先 Claim，再由所有 worker 同步求值所有 callback 参数 |
| [C_compete_first_lazy](C_compete_first_lazy/README.md) | `PA_LAZY_SAMPLE_SHAPE_ID=2` | 与 B 同控制流；loser 不求值 input/scalar thunk |

A 的 eager `Build*Args` 在进入 Submit 前发生。原始 Submit 的共同前段是
`EfDrain -> Materialize -> TensorMap retire -> Claim`；非 Alloc 随后是 winner-only Fanin、全员
Register、winner-only Build，Alloc 则在 Claim 前 Register，winner 最后完成发布。

B/C 的固定顺序是：

```text
Begin -> EfDrain -> Claim -> 调用一次 outer callback
      -> Add* 内同步求值允许执行的 nested lambda
      -> callback 生命周期结束 -> split finish
      -> Materialize -> TensorMap retire -> winner-only Fanin
      -> 全员 Register -> winner-only Build/Complete
```

这里没有“先保存所有 thunk、Claim 后再回放”的机制。Claim 在 outer callback 之前已经完成，
`Add*` 在 callback 内立即调用传入的临时 lambda，lambda/closure 不跨越 callback 或 split-TU 边界。
C 只令 loser 跳过 input/scalar；output/inout 仍由所有 worker 求值，以保持本 Tier-1 样例的私有
heap/TensorMap 状态一致。

## 从哪里开始看

- [PERFORMANCE_COMPARISON.md](PERFORMANCE_COMPARISON.md)：固定测试口径、24 样本/版统计、功能计数、
  `.text` 与结论；
- `analyze_performance.py`：校验 72 个独立运行日志并复算异常规则、总体和配对统计；
- [DISASSEMBLY_GUIDE.md](DISASSEMBLY_GUIDE.md)：真实解码、DWARF 源码注释的生成方法和证据边界；
- 每份包的 `README.md`：该包的固定宏、控制流、内容和独立复跑命令；
- C 的 `disassembly/key_flow/aic_lazy_input_policy.source.asm`：最直接的 lazy 判定源码注释与机器码；
- A/B/C 的 `output/performance/raw/`：正式 device0 b256 原始输出，不只保留汇总数字。
- 根目录和每份包的 `MANIFEST.sha256`：公共文件与三份独立交付的完整哈希入口。

## 结论边界

正式主对照是 C-B：两者 `.text` 总尺寸相同、entry/finish 机器码相同，差别集中在
callback/orchestration。24 个独立样本/版按统一 Hampel 规则去异常后，C-B Submit span 总体
中位差为 `+74.364 us / +1.986%`；按相邻排列块配对后中位差只有 `+1.503 us / +0.040%`，
C/B 各自更快 11/22 块。数据存在明显时间漂移，配对结果不支持 lazy 有稳定收益或回退；差异
低于约定的 5% 门槛，因此没有采集 I-cache PMU，也不作 I-cache 因果结论。

A-B 同时改变了参数求值时点、Submit 顺序、split-TU/outlining 和 `.text` 布局，只能表示整套
compete-first 结构相对原始结构的结果，不能被称为 lazy 收益。

## 二进制身份说明

每份包同时保留：

- `artifacts/measured/`：正式 24 样本性能日志实际执行的 host/final ELF；
- `artifacts/rebuilt/`：从该包本地源码清空专属 build 目录后生成的同一 clean-build 发布副本；
- `artifacts/runtime_identity.json`：两份发布副本的完整 ELF、host 与运行时内容校验。

性能测试直接执行各包的 clean `build/<variant>`；`measured/rebuilt` 当前完整 final ELF 与 host
均相同。带源码注释的反汇编从这个 `measured` ELF 生成，并在文件头记录完整 ELF 哈希；性能
采集期间没有重新构建，因而执行二进制与反汇编直接对应。
