// Probe: does dcci(ENTIRE_DATA_CACHE, CACHELINE_OUT) clobber a prior atomicMax?
//
// Pure CCEC intrinsic version. Replicates the execute_slot sequence:
//   1. atomicMax(&flag, 1)         — writes HBM directly (bypasses L1)
//   2. dcci(ptr, ENTIRE_DATA_CACHE, CACHELINE_OUT)  — next task's flush
//
// If the flush writes back a stale cached copy of flag, the atomicMax is lost.
// Multiple blocks run concurrently: each sets its own flag word, then all do
// ENTIRE flush, then re-read to check if their flag survived.
//
// Layout (gx[30]=BARRIER_SLOT reserved for ccec_barrier, never used as data):
//   gx[64 + bid]  = per-block flag word (one per block), init 0
//   gx[128]       = summary: count of blocks whose flag survived
//   gx[129 + bid] = per-block raw flag value read back after flush
#include "ccec_utils.h"

extern "C" __global__ __aicore__ void KERNEL_ENTRY(entire_flush_clobber)(
    __gm__ uint32_t *gx, uint32_t num_blocks)
{
    uint32_t bid = get_block_idx();

    ccec_barrier(gx, num_blocks, 1);

    // Step 1: each block sets its flag word via atomicMax (CCEC builtin)
    atomicMax(&gx[64 + bid], bid);
    __asm__ __volatile__("" ::: "memory");

    // Step 2: simulate the next task's dcci(ENTIRE_DATA_CACHE, CACHELINE_OUT)
    dcci(reinterpret_cast<__gm__ int32_t *>(&gx[64 + bid]), ENTIRE_DATA_CACHE, CACHELINE_OUT);
    __asm__ __volatile__("" ::: "memory");

    // Step 3: invalidate + re-read our flag to check if it survived
    dcci(reinterpret_cast<__gm__ int32_t *>(&gx[64 + bid]), SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
    uint32_t val = gx[64 + bid];

    ccec_barrier(gx, num_blocks, 2);

    if (val == bid) {
        atomicAdd(&gx[128], 1u);
    }

    // Per-block raw flag value for host-side reporting (replaces printf).
    gx[129 + bid] = val;

    ccec_barrier(gx, num_blocks, 3);
}
