# ordinary/SIMT production-shape CCEC runtime gate

这个目录只验证冻结架构能形成真实的双 TU A5 machine object，不接入
`scalar_build` 的正式 Host、kernel、manifest 或 `run.sh`。

- AIV0 mixed GLOBAL entry 在 VF launch 前记录端到端起点。
- 一个静态 128-thread SIMT VF 只让四个 warp lane0 动态 Attach closed
  Plan、领取 `build_next` ticket、绑定 task、执行完整窄 Build、唯一报到并
  由最后 leader 发布 `build_release`。Build owner 使用真实 Exec owner
  namespace 中可执行的 0..3；物理 SIMT leader 身份另存在 report 字段，
  不能借 96..99 这种越界 owner 值编码诊断身份。
- 四个 leader report 各自占 128B，使用 `stcg -> fence -> magic` 发布。
  AIV0 在 V->S join 后逐 cache line DCCI，再校验 `N+4`、四次 arrival、
  owner 0..3 与 release。
- success 和 fatal 都只能在 VF join 后调用另一份 Scalar-identity TU。
  continuation 会再次返回型读取 `build_release`、Plan fatal 和 scheduler
  fatal，并写入独立 GM report，因此不是空占位。

GLOBAL entry 的第三个 `reserved_flags` 参数首版必须为零；`UINT32_MAX`
分支只用于稳定编码 `MIX_VF=4` metadata，不属于可执行协议。

该 gate 仍不是上板正确性结论：它证明 CCEC 能生成并链接完整控制/Build
机器码，但 AICPU -> SIMT、四 leader -> AIV0、AIV0 -> Scalar 的 A5 可见性
仍需真机用例验证。其 RoutePolicy 只校验 canonical engine/function 自洽，
不包含 PA task kind、`task_id` 周期或固定 dispatch 答案。

运行：

```bash
./build.sh
```
