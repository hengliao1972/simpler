<!-- markdownlint-disable MD060 -->

# Cache Coherency (GM ↔ AICore/AICPU)

This page is the authoritative reference for **when to insert a cache
operation** on Ascend onboard hardware. The coherency model is shared
across the chip generations supported in this repo (a2a3, a5); cycle
costs and a few code-path examples below are sampled on a2a3 unless
stated otherwise. Misapplying these rules either leaks stale data
(correctness bug) or burns a `dsb sy` per record on a hot path (perf
bug — see PR #863 lineage). Read it before touching any code that
writes `dc civac`, `dc cvac`, `cache_invalidate_range`, or any `dcci`
on AICore.

## Who shares GM, and with what consistency

GM (HBM) is read and written by four parties on Ascend onboard. Their
relationship to **AICPU's data cache** is **not symmetric** — that
asymmetry is the entire reason this page exists.

| Writer | AICPU sees write automatically? | AICPU must invalidate before read? |
| ------ | ------------------------------- | ---------------------------------- |
| AICPU itself | Yes (own cache) | No |
| **AICore** (AIC / AIV / MIX) | **Yes** | **No** |
| Host DMA (`rtMemcpy` from host RAM) | No | **Yes** |
| SDMA engine (device-side DMA) | No (assumed) | **Yes** (until hardware confirms otherwise) |

Likewise, **AICore's own cache is non-coherent with GM** in the other
direction: AICore-side writes stay in its data cache until explicitly
pushed out with `dcci`. On the current A5 main `aicpu_scheduler` Path-A,
AICPU stores reach the shared coherency point without an explicit AICPU
`dc cvac`; however, an AICore consumer that primed its local DCache must
still `dcci` that payload line before an ordinary load.

The rest of this doc fills in why each row of the table is what it is,
and what code lives on each side.

Do not collapse the model into a single "GM is coherent with AICPU" rule.
There are two distinct scenarios:

1. **Host/SDMA → AICPU** asks whether a DMA writer snoops AICPU cache.
   Host DMA does not; SDMA remains conservatively non-coherent. The AICPU
   consumer invalidates with `dc civac`.
2. **AICPU ↔ AICore Scalar** asks about the current A5 Path-A snoop domain
   and AICore's private Scalar DCache. AICPU publication needs release
   ordering but no `dc cvac`; a Scalar ordinary payload consumer still
   needs DCCI. In the reverse direction, Scalar performs DCCI before
   publication and AICPU does not invalidate again.

Evidence or cache operations from one scenario must not be used to infer the
other.

## The two cache primitives

| Primitive | Side | Purpose | Cost (rough, a2a3 / DAV_3510) |
| --------- | ---- | ------- | ----------------------------- |
| `dcci` (`__attribute__((aicore))` intrinsic) | AICore | Push a cache line out to GM (clean+invalidate). Required after AICore stores that AICPU or peer AICore must read. | 1 cache line per call; keep a following `dsb` at an unverified dependency boundary. The current A5 `DCCI lines -> atomicExch(control)` path has a direct no-DSB gate. |
| `cache_invalidate_range(addr, size)` (`src/{a2a3,a5}/platform/onboard/aicpu/cache_ops.cpp`) | AICPU | `dc civac` + `dsb sy` + `isb` over a byte range. Required before AICPU reads GM that **a non-coherent writer** (host DMA, SDMA) most recently published. | `dsb sy` dominates (tens to hundreds of cycles, fixed regardless of range). |

`cache_invalidate_range` is the protocol-correct primitive for the
**host-DMA → AICPU** case. It was introduced in PR #204 specifically
for `Runtime` struct hand-off where the host writes via `rtMemcpy` and
AICPU reads. **It is NOT the right primitive for AICore → AICPU.**

### `dcci` is whole-cache-line: no parallel sub-line writes

`dcci` cleans+invalidates **one whole cache line** (64 B on a2a3 = 16
`float`s); there is no sub-line granularity. A core's writeback emits its
entire line copy, including bytes it never wrote (stale in its copy).

**Consequence — two AICore cores must never write different elements of the
same cache line in parallel.** Each core flushes the whole line, so the last
flush clobbers the others' elements with stale values (classic false
sharing → last-writer-wins). This is invisible on `sim` (cache modeled as
no-op: `SINGLE_CACHE_LINE == 0` in `platform/sim/aicore/inner_kernel.h`) and
only fails on silicon. The kernel-author rule (each SPMD block writes its own
cache line) lives in
[../aicore-kernel-programming.md](../aicore-kernel-programming.md#each-block-must-write-to-its-own-cache-line).

## The "AICore → AICPU" path: AICPU does not invalidate, but DOES barrier

AICore and AICPU share a coherency domain on GM. When AICore writes a
slot, the correct handshake is:

```text
AICore                              AICPU
  store slot fields                   read COND (MMIO, Device-nGnRE)
  store task_id (last)                check FIN bit
  dcci slot, SINGLE_CACHE_LINE   →    rmb()                ← load-load barrier
  dsb (commit dcci before FIN)        read slot fields     ← Normal cacheable
  write FIN → COND                ←
```

The explicit `dsb` remains part of this FIN/COND handshake: the 2026-08-12
no-DSB result does not cover a Device-MMIO FIN write. That result is narrower:
on current A5, 32 dirty lines followed by 32 DCCI operations and an immediate
`atomicExch` control publication passed without an intervening explicit DSB.
See the direct-gate evidence below. Do not transfer that exception to this
MMIO sequence or to another successor primitive without a matching gate.

Two separate concerns, often conflated:

- **Cache coherency** (Do we need `dc civac`?): No. AICore's `dcci`
  pushes the line to GM and the AICPU's cache is in the same coherency
  domain, so a subsequent AICPU load fetches the fresh value. `dcci`
  is load-bearing on AICore's side — AICore's data cache is not
  coherent with GM in the other direction, so without `dcci` the line
  never reaches HBM and AICPU would observe an old value.

- **Load-load ordering** (Do we need `rmb()`?): **Yes.** The COND
  register is `Device-nGnRE` memory; the slot is Normal cacheable
  memory. ARM64 does not implicitly order Device reads against
  subsequent Normal reads — they can be reordered if there is no
  data/address dependency. In this path, the slot address is computed
  from a value the caller already holds (the dispatched `task_id`),
  not from the just-read COND value, so there is no architectural
  dependency. Without `rmb()` (`dsb ld`), the CPU can speculatively
  satisfy the slot load before the COND read indicates FIN, returning
  whatever was in the AICPU's cache at speculation time (likely a
  stale value from a previous round). The AICPU side must emit
  `rmb()` between the COND check and the slot reads.

Concretely, the L2 swimlane staging-slot read in
`src/{a2a3,a5}/platform/shared/aicpu/l2_swimlane_collector_aicpu.cpp` does
**not** call `cache_invalidate_range` on the slot, but it **does** call
`rmb()` before reading `slot->task_id` and the timing fields. All of
those fields are AICore writes covered by the AICore-side `dcci` in
`l2_swimlane_aicore_record_task`. The same pattern applies to the PMU
staging slot
(`src/{a2a3,a5}/platform/shared/aicpu/pmu_collector_aicpu.cpp`).

### Historical pitfall

PR #540 (2026-04-15) added `cache_invalidate_range(slot, 64)` on the
AICPU side of the L2 swimlane staging slot, mirroring the
host-DMA-protocol pattern from PR #204. The two situations are
**not** the same: host DMA bypasses the AICPU cache; AICore stores
plus `dcci` do not. The cache invalidate was redundant — but the
embedded `dsb sy` inside `cache_invalidate_range` was inadvertently
providing the COND→slot load-load ordering as a side effect. Replacing
the whole call with nothing would have left the ordering implicit (and
dependent on microarchitectural quirks); replacing it with an explicit
`rmb()` keeps the ordering as part of the documented protocol while
dropping the unnecessary cache op.

If you find yourself about to write `cache_invalidate_range` on an
AICPU-side read of an AICore-published value, **stop**. The right fix
is `rmb()` (for load-load ordering against a prior COND read) plus
making sure the AICore side does `dcci` before signaling. The cache
invalidate itself is not needed on this path.

## The "AICPU → AICore" path on A5 Path-A: release, not clean

The main `aicpu_scheduler` Path-A used by `cross_core_aicpu_plan` is in the
A5 AICore snoop domain. The validated publication sequence is:

```text
AICPU                              AICore Scalar
  ordinary store payload             return-value atomic poll control
  release atomic store control  →    dsb
    (HCC: stlr)                       dcci payload line
                                      dsb
                                      ordinary load payload
```

The AICPU producer does **not** need per-publication `dc cvac`, `dsb sy`, or
`isb` in this path. The release store is load-bearing: it preserves the
payload-before-control publication order. Do not replace it with an ordinary
store merely because the weaker variant happened to be fresh in the current
probe.

The AICore-side `dcci` remains load-bearing for ordinary payload reads. The
atomic control observation does not invalidate a separately cached payload
line. This rule is therefore not "delete all cache operations"; it moves the
protocol boundary to the actual non-coherent local cache.

This result is launch-path specific. A repository incident involving a cust
AICPU subprocess pinned to another cluster observed stale AICPU L1 data and
showed that the process was outside the AICore snoop domain. Re-run the same
direct-doorbell probe before applying this rule to a different AICPU loader,
cluster affinity, chip, or CANN version.

## The "host DMA → AICPU" path: AICPU MUST invalidate

When the host writes via `rtMemcpy`, the AICPU cache is not snooped.
A cached value from a previous round survives the DMA write, so the
next AICPU load returns stale data. The original PR #204 fix was:

```cpp
// src/a2a3/runtime/host_build_graph/aicpu/aicpu_executor.cpp
cache_invalidate_range(runtime, sizeof(Runtime));  // before reading host-written Runtime
```

This is **necessary** and must not be removed. It is the
load-bearing usage of `cache_invalidate_range` in this repo.

Same protocol applies to anything else the host DMAs into a GM region
that AICPU subsequently reads.

## The "SDMA → AICPU" path: treat as non-coherent until proven otherwise

SDMA is a separate device-side DMA engine. It writes GM but is not
known to snoop AICPU's data cache. SDMA backend code currently lives
only under the a2a3 runtime tree, but the principle applies to any
chip generation that exposes SDMA. Current runtime code (see
`src/a2a3/runtime/tensormap_and_ringbuffer/runtime/backend/sdma/sdma_completion_scheduler.h`
and the SDMA-engine async-wait completion path in
`runtime/pto_async_wait.h` / `runtime/scheduler/pto_scheduler.h`)
**does** invalidate before reading SDMA-written counters and records,
on the conservative assumption that SDMA is not in AICPU's coherency
domain.

If a hardware confirmation lifts that assumption — i.e. SDMA writes
become snooped — these invalidates can be removed using the same
reasoning as the AICore case. Until that confirmation exists, **leave
them in place**.

## A5 single-point evidence (2026-08-12)

The standalone
[AICPU/AICore cache and atomic probe](../../tests/atomic_probe/aicpu_aicore_cache/README.md)
tests both directions with a real HCC AICPU owner and CCEC AIV Scalar
kernels. It reuses the same addresses for 4096 rounds per case and ran five
independent host processes on device 0. Every classified sample was an exact
current generation or the exact primed stale generation; no torn, historical,
or otherwise invalid value was observed.

| Case | Five-run result | Protocol implication |
| ---- | --------------- | -------------------- |
| AICPU ordinary payload stores, then release atomic control; no `dc cvac`, explicit DMB/DSB, ISB, or separate done publication | 20480/20480 fresh | This is the current A5 Path-A production gate; per-publication AICPU clean/barrier is unnecessary. |
| Same release publication, but AICore omits payload `dcci` | 0 fresh / 20480 stale / 0 other; same-round `ld_dev` is 20480 fresh | A fresh atomic control does not invalidate a separate payload line. |
| AICPU ordinary payload/control, then a separate release atomic doorbell; no clean | 20480/20480 fresh | Release ordering works independently of control and doorbell sharing an address. |
| AICPU ordinary payload/control with no clean and no ordering primitive | 20480/20480 fresh | Observation only; do not turn a microarchitectural result into an ordinary-store publication contract. |
| AICore reuses a cached ordinary control load after a release doorbell | 0 fresh / 20480 stale / 0 other; same-round return-value atomic is 20480 fresh | Do not poll a cross-domain control with an ordinary cached load. |
| AICore stores 32 payload lines, executes default DCCI for every line + one trailing DSB, then `atomicExch` control | 20480/20480 fresh rounds; 655360/655360 lines | Positive DSB control; AICPU does not need `dc civac`. |
| Same 32-line path, but no explicit DSB between the final default DCCI and `atomicExch` | 20480/20480 fresh rounds; 655360/655360 lines; OUT DCCI gives the same result; 320 quiet post-publish ACK windows per selector pass | On this exact A5/CANN atomic publication path, a separate DSB is not required for correctness. |
| AICore stores the same 32 lines but executes no DCCI; DSB present or absent before `atomicExch` | 0 fresh / 20480 stale / 0 other in both variants, including the later AICPU `dc civac` reference | DSB is not a writeback primitive and cannot replace producer DCCI. |
| AICore omits producer `dcci` for payload or ordinary control | 0/20480/0 both before and after AICPU `dc civac` | Consumer invalidation cannot publish a dirty line still held by the producer. |

The direct release case excludes hidden clean assistance: the tested control
is the doorbell, and AICPU waits for a final AICore acknowledgement before it
may clean results or exit. Owner-ELF disassembly shows ordinary `str` payload
stores followed by `stlr`, with no `dc cvac`, DMB, DSB, or ISB. The weaker
ordinary-store case and a single AICore `st_dev` were also fresh in all 20480
observations, but remain observation-only and do not relax ordering rules or
the repository ban on `st_dev` as a general business write path.

For the no-DSB AICore cases, compiler-inserted Scalar DCCI and kernel-end DCCI
were disabled. Optimized device LLVM IR goes from the last
`llvm.hivm.DCCI.DST` to `llvm.hivm.atom.EXCH.G.s64` without an intervening
`llvm.hivm.DSB`; the with-DSB branch contains the DSB intrinsic. This is
runtime plus optimized-IR evidence, not final-machine-code disassembly. No
public CANN/ISA contract was found stating that DCCI itself completes before
every successor or that every atomic replaces DSB. Therefore the exception
is limited to the gated `DCCI lines -> atomicExch(control)` sequence; retain
DSB for reader-side DCCI, MMIO publication, other successors, and untested
chips/toolchains.

The AICPU consumer does not hide a compensating barrier. It reads payload in
reverse line order immediately after the acquire control observation, so the
last DCCI target is sampled first. AArch64 owner-ELF disassembly shows `ldar`
followed by the primary payload `ldr` loop; the first subsequent `dc civac`
and `dsb sy` belong to the later reference read and occur after primary values
have already been captured.

Nor can the next Scalar atomic hide completion in the quiet samples. Every
64th no-DSB round executes 1048576 pure NOPs after `atomicExch`, with no GM,
atomic, or DSB operation. AICPU release-publishes an ACK only after capturing
primary payload, and the producer's first later memory instruction checks that
ACK. Optimized device IR is `atomicExch -> NOP loop -> atomicAdd ACK`; all 320
quiet windows per selector passed across five processes.

## Quick decision table

When you are about to insert a cache operation, ask in order:

1. Who actually wrote the bytes I'm reading? Look at the producer
   code, not the address.
2. Is the producer in the AICPU coherency domain (AICPU itself or
   AICore)? If yes → no invalidate. If no (host, SDMA) → invalidate.
3. For AICore writes specifically, does the producer already `dcci`
   before signaling? If not, fix that instead of papering over it
   with an AICPU-side invalidate.
4. Did I just read a completion flag (COND / mailbox / counter) from
   a different memory type (Device-nGnRE MMIO) before this load? If
   yes, and there is no data/address dependency between that read and
   this one, insert `rmb()` between them — coherency does not imply
   load-load ordering on ARM64.

If the answer to (1) is "I'm not sure" — find out. The cost of one
wrong `cache_invalidate_range` is silent perf rot; the cost of a
missing `rmb()` is a stale-data bug that may only fire under specific
buffer-reuse patterns or aggressive OOO speculation. Both are paid
forever once they ship.

## Related code

- `src/{a2a3,a5}/platform/onboard/aicpu/cache_ops.cpp` — `cache_invalidate_range` implementation (`dc civac` / `dsb sy` / `isb`).
- `src/{a2a3,a5}/platform/sim/aicpu/cache_ops.cpp` — sim no-op.
- AICore-side `dcci` usage lives in the L2 swimlane / PMU AICore collectors and any kernel that publishes to a GM slot AICPU reads.

## Related docs

- [PMU staging-slot ordering](../dfx/pmu-profiling.md) —
  detailed AICore-side `dcci` + barrier order for staging-slot writes.
- [L2 swimlane profiling](../dfx/l2-swimlane-profiling.md) —
  the consumer of the rules above on the L2 swimlane path.
