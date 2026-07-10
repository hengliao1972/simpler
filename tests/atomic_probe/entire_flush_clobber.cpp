// Probe: does dcci(ENTIRE_DATA_CACHE, CACHELINE_OUT) clobber a prior atomicMax?
//
// This replicates the exact sequence in dist_engine.cpp execute_slot:
//   1. atomicMax(&flag, 1)         — dist_set_flag, writes HBM directly
//   2. dcci(ptr, ENTIRE_DATA_CACHE, CACHELINE_OUT)  — next task's pre-kernel flush
//
// If the flush writes back a stale cached copy of flag, the atomicMax is lost.
// Multiple blocks run concurrently: each sets its own flag word, then all do
// ENTIRE flush, then re-read to check if their flag survived.
//
// Layout: gx[0..63] = 64 flag words (one per block), init 0
//         gx[64]    = summary: count of blocks whose flag survived
#include "kernel_operator.h"
using namespace AscendC;

#ifdef __DAV_VEC__
#define KERNEL_ENTRY(x) x##_0_mix_aiv
#else
#define KERNEL_ENTRY(x) x##_0_mix_aic
#endif

extern "C" __global__ __aicore__ void KERNEL_ENTRY(entire_flush_clobber)(
    __gm__ uint32_t *gx, uint32_t num_blocks)
{
    (void)num_blocks;
    uint32_t bid = GetBlockIdx();

    SyncAll();

    // Step 1: each block sets its flag word via atomicMax (like dist_set_flag)
    AtomicMax(&gx[bid], bid);
    __asm__ __volatile__("" ::: "memory");

    // Step 2: simulate the next task's dcci(ENTIRE_DATA_CACHE, CACHELINE_OUT)
    // This is what execute_slot does before the NEXT kernel call.
    dcci(reinterpret_cast<__gm__ int32_t *>(&gx[bid]), ENTIRE_DATA_CACHE, CACHELINE_OUT);
    __asm__ __volatile__("" ::: "memory");

    // Step 3: invalidate + re-read our flag to check if it survived
    dcci(reinterpret_cast<__gm__ int32_t *>(&gx[bid]), SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
    uint32_t val = gx[bid];

    SyncAll();

    if (val == bid) {
        AtomicAdd(&gx[64], 1u);
    }

    // Also write result for host-side check
    gx[65 + bid] = val;

    SyncAll();

    if (bid == 0) {
        printf("entire_flush_clobber: survivors=%u/%u\n", gx[64], num_blocks);
    }
}
