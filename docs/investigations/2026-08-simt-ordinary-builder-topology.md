# Intermediate SIMT ordinary builder counts were not retained

**Date**: 2026-08-08
**Verdict**: deferred-pending-stable-multi-workload-evidence

## Question

Could `simt_cross_core_ordinary` improve its PA B256 end-to-end time by using
an intermediate number of builder AIV0 cores instead of the retained K1 or the
previously rejected K32 endpoint?

## What was tried

The experiment kept shared TensorMap, W4, dynamic requests, execution tickets,
and the startup-to-FinalDrain `perf-clock` endpoint unchanged. It scanned
builder limits K1, K2, K4, K8, and K16 on the same 32-block PA B256 workload.

The retained K1 implementation assigns builder rank zero. For K greater than
one, the experiment first restored physical block id as builder rank; without
that corresponding change, multiple VFs correctly failed closed after trying
to consume the same task stream.

## Result

The single-pass topology scan measured:

```text
K1   94.732 ms
K2  136.467 ms
K4  118.651 ms
K8  106.304 ms
K16  93.843 ms
```

Because K16 was close to K1, both endpoints were repeated:

```text
K16: 93.843 / 100.282 / 97.869 ms, median 97.869 ms
K1 : 104.105 / 93.608 / 138.501 ms, median 104.105 ms
```

Including the initial K1 sample gives a four-sample K1 median of about
99.419 ms. K16 then leads by only about 1.56%, while the ranges overlap and K1
itself spans 93.608 to 138.501 ms.

All measured K2/K4/K8/K16 PA runs passed numerical correctness after fixing
the experimental rank. The code was restored to K1 after measurement.

## Why not (now)

The apparent K16 advantage is smaller than the observed run-to-run variation.
It also permanently removes 15 AIV replay/execute workers, a trade that can
change with operator Build weight. One PA workload is not enough to make that
the public ordinary-scheduler default.

## When to reconsider

Reconsider after builder limit is an explicit artifact identity and phase
counters separate Build supply from Execute supply. Require interleaved A/B
measurements on PA and at least two non-PA workloads before changing K1.

## References

- `tests/atomic_probe/pa_scheduler/simpler四种调度模式迁移记录.md`
- `src/a5/runtime/fully_distributed_within_core/runtime/`
  `fdwic_build_identity.h`
