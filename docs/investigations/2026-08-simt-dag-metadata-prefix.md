# SIMT DAG metadata prefix did not reduce end-to-end time

**Date**: 2026-08-08
**Verdict**: dropped

## Question

Could mode4 replace returned atomic reads in its bounded DAG metadata scan
with ordinary L1-bypass loads if builders first established a contiguous
metadata-ready prefix?

The proposal was attractive because each INPUT or INOUT currently scans up to
64 prior tasks, and every candidate control word is read with a returned
`atomicAdd(0)`.

## What was tried

The prototype reused a reserved bit in each task's existing 64-bit metadata
control word:

1. A builder wrote the task's immutable writer payload.
2. Task N waited for task N-1 to publish its prefix-ready bit.
3. The builder atomically published task N with its own prefix-ready bit.
4. DAG lookup read candidate control words with CANN's `asc_ldcg`, which is
   L1 non-cacheable and retains normal L2 caching.

An earlier attempt to use `__builtin_cce_ld_dev` inside the SIMT vector
function made the dav-c310-vec compiler exit with code 139. It was discarded
before device measurement.

The final prototype passed the CPU prefix protocol test, four targeted DAG
identity/protocol/state tests, the mode4 CCEC build, the non-PA Bd24 INOUT
chain on A5, and PA B1/B256 numerical goldens.

## Result

PA B256 used shared TensorMap, K16/W4, and the startup-to-FinalDrain
`perf-clock` endpoint. Three independent processes measured:

```text
178.674 ms / 175.802 ms / 172.999 ms
median = 175.802 ms
```

The retained implementation's median is 173.206 ms. The prototype regressed
by 2.596 ms, or about 1.50%. Its first two samples were also slower than the
largest retained sample, 173.976 ms.

## Why not (now)

The replacement removed returned atomic reads from candidate lookup, but it
also serialized metadata publication across the complete task sequence. That
trade did not provide a stable end-to-end benefit, so the code and temporary
protocol test were fully reverted.

The prefix may also have removed useful natural staggering before builders
entered later shared publication paths. This is only a hypothesis because the
prototype did not add per-builder phase counters.

## When to reconsider

Reconsider returned-atomic removal only when it does not require a global
task-order prefix, for example after a local symbol/region index substantially
reduces the candidate set. Add phase counters first if publication staggering
is part of the new hypothesis.

## References

- `tests/atomic_probe/pa_scheduler/simpler四种调度模式迁移记录.md`
- `src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/`
  `simt_cross_core_builder.h`
