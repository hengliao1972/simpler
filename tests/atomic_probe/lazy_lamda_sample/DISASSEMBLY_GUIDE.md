# final-linked 源码注释反汇编说明

## 交付内容

每份 A/B/C 包都独立保存以下文件：

- `artifacts/measured/pa_scheduler_kernel.o`：正式性能测试实际使用的 final ELF；
- `disassembly/raw/*.asm.gz`：每个非空 `.text` `STT_FUNC` 的完整真实机器码解码；
- `disassembly/annotated/*.source.asm.gz`：对应函数的完整源码注释版；
- `disassembly/key_flow/*.source.asm`：未压缩的关键 Submit/lazy 阅读窗口；
- `disassembly/raw/manifest.tsv` 与 `gaps.tsv`：函数覆盖和 alignment/padding 缺口；
- `disassembly/published.sha256`：raw、annotated 和 key-flow 文件哈希。

最适合先打开的四个文件是：

```text
A_original/disassembly/key_flow/aic_original_submit.source.asm
B_compete_first/disassembly/key_flow/aic_compete_first.source.asm
C_compete_first_lazy/disassembly/key_flow/aic_compete_first_lazy.source.asm
C_compete_first_lazy/disassembly/key_flow/aic_lazy_input_policy.source.asm
```

最后一个文件直接包含 `if constexpr (Lazy) { if (!won_) return; }` 的本地源码、原始注释、DWARF
定位和相邻机器指令。

## 真实解码链路

`disassemble.py` 不把十六进制或中间 IR 冒充成反汇编。它执行以下可审计步骤：

1. 直接解析 final ELF64 little-endian section/symbol table；
2. 只选择 final `.text` 中 size 非零的 `STT_FUNC`，按最终 PC 和符号 size 截取真实函数体；
3. 使用 CANN 9.1 A5 `dav_3510` 随包解码器
   `$ASCEND_HOME_PATH/x86_64-linux/simulator/dav_3510/lib/libpem_davinci.so`；
4. 固定并校验解码器 SHA256：
   `29835d2439d6dd464d34a212ad4bbd5c29af6a38465da09a6c273401d9a96dcb`；
5. 按 4/8-byte 真实编码逐字节核对 decoder 输出与 ELF 原始字节；
6. 对 RVec `.vector.thread` 使用显式 full-body range；
7. 拒绝 `UNDEF/UNKNOWN/INVALID/ILLEGAL` 等无效 mnemonic；
8. 把函数符号之外的 alignment/padding 按地址、长度和 SHA256 记录到 `gaps.tsv`，不冒充指令。

当前完整解码覆盖如下：

| 形态 | 函数数 | 指令数 | 函数字节 / `.text` |
| --- | ---: | ---: | ---: |
| A | 11 | 194923 | 779700 / 780344 B（99.917%） |
| B | 17 | 136792 | 547176 / 547640 B（99.915%） |
| C | 17 | 136810 | 547248 / 547640 B（99.928%） |

## 源码和原注释是如何加入的

`annotate_disassembly.py` 把 raw 文件中的每个 final PC 批量交给 GNU `addr2line -e <final ELF> -C`，
取得 DWARF 文件/行映射，再把 DWARF 中的编译期绝对路径映射到该包自己的 `common/` 或 `ccec/`，
读取对应源码行。
因此输出中的普通代码和 `//` 注释确实来自随包 `.h/.cpp`，而不是人工重写的伪代码。

证据边界必须同时保留：

- DWARF 只证明地址对应到某个文件/行；
- `# [DWARF] file:line` 是定位结果；
- `# > line | source` 及其上下文是脚本从本地源码复制的文本；
- 源码注释不是 ELF 内保存的注释，也不能把每条上下文注释宣称为精确拥有下一条机器指令；
- raw 机器码/助记符才是静态 codegen 的权威内容。

每个 annotated 文件头都重复写明这一边界，避免脱离本说明后被误读。

## 阅读和复现

查看完整压缩文件：

```bash
zless disassembly/annotated/<function>.source.asm.gz
zless disassembly/raw/<function>.asm.gz
```

在任一变体目录内，source CANN 9.1 环境后可独立重建：

```bash
./generate_disassembly.sh
sha256sum -c disassembly/published.sha256
```

`generate_disassembly.sh` 只清空该变体自己的 `disassembly/raw` 和 `disassembly/annotated`，并始终
从 `artifacts/measured/pa_scheduler_kernel.o` 生成全量和 key-flow 结果，以保证反汇编与正式性能日志对应。反汇编可以
说明 codegen/布局差异，不能单独证明性能差异由 I-cache 引起；当前性能测试明确使用 PMU off，
没有 I-cache PMU 数据。

## 与性能复测的关系

当前带源码注释的反汇编是从三份包内的 `measured` final ELF 重新生成的，完整 ELF SHA256 分别为：

- A：`76c961f846c7efc89b40942c3a8113530a6da2c7bfbdc601615852f8c5b79cfc`；
- B：`82a27e206ee1b2411964c0530c73adf92a2b032ac202a7c65c6b5f7d76a4571b`；
- C：`8d293c52312429d13efe89592ed705c672ceef1f2875b5e3bd25582419d37893`。

24 样本/版的性能复测直接执行同一组 clean-build host/final ELF，没有修改源码或重新链接；因此
本次只更新性能日志和统计，不需要改反汇编。若任一源码或 ELF 哈希变化，必须重新运行对应包的
`./generate_disassembly.sh`。
