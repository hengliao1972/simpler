// Probe: does dcci(ENTIRE_DATA_CACHE, CACHELINE_OUT) clobber a prior atomicMax?
//
// Pure CCEC intrinsic version. Replicates the execute_slot sequence:
//   1. atomicMax(&flag, 1)         — writes HBM directly (bypasses L1)
//   2. dcci(ptr, ENTIRE_DATA_CACHE, CACHELINE_OUT)  — next task's flush
//
// If the flush writes back a stale cached copy of flag, the atomicMax is lost.
// Multiple participants run concurrently: each sets its own flag word, does
// an ENTIRE flush, then re-reads to check if its flag survived.
//
// Layout:
//   gx[64 + participant]  = per-participant flag word, init 0
//   gx[256]               = summary: count of participants whose flag survived
//   gx[257 + participant] = per-participant raw flag value after flush
#include "ccec_utils.h"

CCEC_PROBE_KERNEL_META(entire_flush_clobber);

constexpr uint32_t FLAG_BASE = 64;
constexpr uint32_t SUMMARY_SLOT = 256;
constexpr uint32_t RAW_BASE = 257;

extern "C" __global__ __aicore__ void KERNEL_ENTRY(entire_flush_clobber)(
    __gm__ uint32_t *gx, uint32_t num_blocks)
{
    (void)num_blocks;
    uint32_t participant = ccec_participant_id((uint32_t)get_block_idx());

    // Step 1: each participant sets its flag word via atomicMax (CCEC builtin)
    atomicMax(&gx[FLAG_BASE + participant], participant + 1);
    __asm__ __volatile__("" ::: "memory");

    // Step 2: simulate the next task's dcci(ENTIRE_DATA_CACHE, CACHELINE_OUT)
    dcci(reinterpret_cast<__gm__ int32_t *>(&gx[FLAG_BASE + participant]),
         ENTIRE_DATA_CACHE, CACHELINE_OUT);
    __asm__ __volatile__("" ::: "memory");

    // Step 3: invalidate + re-read our flag to check if it survived
    dcci(reinterpret_cast<__gm__ int32_t *>(&gx[FLAG_BASE + participant]), SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
    uint32_t val = ld_dev_b32(&gx[FLAG_BASE + participant]);

    if (val == participant + 1) {
        atomicAdd(&gx[SUMMARY_SLOT], 1u);
    }

    st_dev_b32(&gx[RAW_BASE + participant], val);
    dsb(DSB_ALL);
}
