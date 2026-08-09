# SIMT DAG region shortcuts regressed the CCEC hot path

**Date**: 2026-08-08
**Verdict**: dropped

## Question

After merging per-Tensor DAG history scans into one task-level scan, could two
source-level shortcuts remove the remaining duplicate region work?

The candidate first compared the Tensor descriptor's buffer address before
computing its full region. It also skipped the caller's second region
validation for automatic INPUT/INOUT queries already validated by lookup.

## What was tried

Only the inlined mode4 SIMT builder changed. Metadata publication, returned
atomic count, K16/W4 topology, fanin order, shared state, DCCI, and execution
packet layout stayed identical to the retained batched-scan implementation.

The candidate passed CCEC compilation and PA B256 numerical correctness. It
was measured in independent processes with shared TensorMap and the
startup-to-FinalDrain `perf-clock` endpoint.

## Result

The two candidate runs measured 197.612 ms and 187.161 ms. The retained
batched-scan implementation had measured 172.157 / 173.895 / 172.656 ms.
After reverting only the shortcuts and rebuilding, the next run returned to
172.372 ms.

## Why not (now)

The regression is much larger than the nearby device variation, while the
shortcuts remove only duplicate scalar calculations and provide no protocol
benefit. They were fully reverted.

The exact cause is not assigned. In a large always-inlined SIMT vector
function, branch shape, register allocation, spills, and instruction placement
are plausible explanations, but no VF assembly or phase counter was collected
to distinguish them.

## When to reconsider

Revisit only with generated VF assembly or builder phase counters that can
show why the source shape changes code generation. Do not reintroduce either
shortcut based only on its lower source-level operation count.

## References

- `tests/atomic_probe/pa_scheduler/simpler四种调度模式迁移记录.md`
- `src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/`
  `simt_cross_core_builder.h`
