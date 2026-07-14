# Fdwic: removing `RingSlot::scalars` from submit slot build

**Date**: 2026-07-14
**Verdict**: dropped

## Question

`RingSlot` stores scalar arguments twice: once in `RingSlot::scalars`, then
again in `RingSlot::args` for the kernel ABI. Removing the scalar array looked
like a low-risk way to reduce per-slot storage and the flushed `RingSlot`
region.

## What was tried

In `src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/`:

- removed `RingSlot::scalars[MAX_SCALAR_ARGS]`;
- changed both `build_ring_slot()` and `build_ring_slot_from_submit()` to
  write scalar values directly into `RingSlot::args`;
- kept `BuiltSubtask::scalars`, because joint tasks still need scalar payloads
  to cross lanes through `WonSlot`.

The change was tested on PA Case1 with fdwic swimlane:

- rebuilt `a5sim` and `a5` runtimes;
- ran PA Case1 `a5sim --enable-l2-swimlane`;
- ran PA Case1 `a5 --enable-l2-swimlane` three times through `task-submit`.

## Result

The combined prototype, measured after the output-layout no-zero baseline:

```text
span_us mean:          7751.766 -> 7782.044
SubmitExclusive avg_ns 1377.833 -> 1384.967
Materialize avg_ns     1336.767 -> 1330.567
```

The functional tests passed, but the performance signal was negative for both
overall swimlane span and submit exclusive time.

## Why not now

The proposed benefit was specifically to reduce submit slot build / flush
cost. The measured `SubmitExclusive` moved in the wrong direction, so the
change does not justify altering the slot layout.

The scalar array may also be helping the generated code keep the scalar-to-ABI
path simple despite the apparent duplicate storage; the current data does not
support spending more time on this path.

## When to reconsider

Re-open only if a future profile isolates scalar handling or `RingSlot` flush
size as a larger hotspot than the current PA Case1 traces show, or if the slot
ABI is being redesigned for a broader reason.

## References

- Plan: `fdwic-onboard-performance-optimization.md`
- Analyzer: `scripts/analyze_fdwic_swimlane_critical_path.py`
