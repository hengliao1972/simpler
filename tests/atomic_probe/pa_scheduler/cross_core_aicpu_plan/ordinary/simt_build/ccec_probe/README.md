# canonical Plan v3 SIMT CCEC compile gate

该目录只回答一个问题：CCEC 是否能把公共
`RuntimeTaskPlanCell` ABI v3 的直接消费路径编译成真实 A5 SIMT VF，并在
同一 AIV entry 中完成 `async_invoke` 与 V→S 收口。

它锁定以下静态事实：

- 直接复用 `simt_plan_build_protocol.h` 的 128 thread、32 lane warp、4
  leader 合同；
- 仅 warp lane0 读取公共 Plan control 与 payload，不存在第二份 request
  ABI、Host task plan 或 PA task-kind 公式；
- control 使用返回型 atomic 观察，payload 逐 cache line 生成
  `asc_dcci_single`，随后再次观察 control，并校验 task id、ABI、payload
  lines 和通用 header 字段；
- ELF 仅导出一个 mixed AIV entry，并保留一个 LOCAL SIMT entry；metadata
  编码 `SIMD_SIMT_MIX_VF=4`。

## 明确不在本门槛内的结论

这是 compile gate，不是真机协议测试：

- `asc_dcci_single + asc_threadfence` 能成功 lowering，不代表已经证明它在
  A5 上等价于完整的 Plan 发布/获取内存合同；
- 本探针不领取 `build_next`，不调用动态 Build ticket 协议，也不执行完整
  Build、ordinary TensorMap 串行插入或 Execute；
- 因此不能用本探针宣称 ordinary/SIMT 全链路正确或达到任何性能目标。

构建命令：

```bash
tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/ordinary/simt_build/ccec_probe/build.sh
```
