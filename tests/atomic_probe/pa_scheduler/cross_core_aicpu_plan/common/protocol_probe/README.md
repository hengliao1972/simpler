# A5 AICPU → AIV Plan protocol probe

This directory is deliberately self-contained.  It copies and renames the
verified Path-A `MainAicpuLoader`/bootstrap-dispatcher mechanism and does not
link or include `cross_core_ordinary` or the production `scalar_build` tree.

The probe runs the PA `1 + 4G` sequence for G0 (1 task), G1 (5 tasks),
mixed `{0,1,8192,8193}` (20 tasks), and B256-G1 (1280 tasks).  Its local POD
copy is byte-for-byte aligned with the current public ABI: a 128-byte isolated
control, up to 4416 bytes/69 cache lines of ordinary payload, 64 bytes of tail
padding, and a 4608-byte cell stride.  Metadata tasks exercise the one-line
minimum; kind-1/kind-4 tasks exercise the full 69-line maximum.  A launch nonce
in adapter-owned words makes stale data from the reused GM allocation visible.

Two publication contracts are compared:

- `close-only`: publish every cell, then publish the final frontier and close;
- `per-item-frontier`: publish every cell and advance the continuous frontier
  after each item, then publish final count.

For every cell, the AICPU writes ordinary payload, cleans exactly the published
line range, executes `DSB`, then uses an ordinary control store plus an exact
control-line clean.  The AIV observes every control twice with return-ready
`atomicAdd(ptr, 0)`, invalidates exactly the encoded payload range, executes
`DSB`, and validates every published word and the aggregate checksum.  Atomic
control lines are never AIV DCCI targets.

All runs reuse one device allocation with a new nonce.  A final G1 fault run
writes a 69-line payload, cleans only its first 68 lines, and poisons the
deliberately uncommitted last line without cleaning it; the AIV must report
`PayloadMismatch` at word 544.  The poison makes this an oracle-negative that
does not depend on whether an unclean dirty line happens to be written back.
A complete publication then succeeds at the same address, proving post-fault
reuse/recovery.

```bash
./build.sh build
./build.sh run
```

Optional environment variables are `ATOMIC_PROBE_DEVICE`,
`PLAN_PROTOCOL_REPEATS`, `PLAN_PROTOCOL_DELAY_NOPS`, and
`PLAN_PROTOCOL_TIMEOUT`.  Timing output reports AICPU monotonic nanoseconds,
AIV system-counter ticks, and host overlapped wall time.
