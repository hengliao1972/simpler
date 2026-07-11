// ccec probe: dcci publish/observe seam. Pure CCEC intrinsic version.
//
// CCEC version of mb8_dcci_seam.asc. Tests the producer-consumer seam:
// producer writes data via st_dev + sets flag via atomicAdd, consumer polls
// flag via ld_dev + dcci(inval) + normal read + verify.
//
// Uses lowercase CCEC builtins: atomicAdd, st_dev (via st_dev_b32 helper),
// ld_dev (via ld_dev_b32 helper), dcci.
//
// Layout (gx[30] = BARRIER_SLOT reserved for ccec_barrier):
//   gx[0..15] = 16-word data payload (64B = 1 cache line)
//   gx[16]    = flag (atomicAdd counter)
//   gx[17]    = error count (host checks)
//
// Build: see run_dcci_seam.sh / run_all.sh
#include "ccec_utils.h"

constexpr uint32_t DATA_ELEMS = 16;
constexpr uint32_t NUM_ROUNDS = 100;

extern "C" __global__ __aicore__ void KERNEL_ENTRY(dcci_seam)(
    __gm__ uint32_t *gx, uint32_t num_blocks)
{
    uint32_t bid = get_block_idx();
    __gm__ uint32_t *data = gx;          // gx[0..15] — payload (1 cache line)
    __gm__ uint32_t *flag = &gx[16];     // gx[16] — completion flag
    __gm__ uint32_t *errs = &gx[17];     // gx[17] — error count

    ccec_barrier(gx, num_blocks, 1);

    for (uint32_t round = 0; round < NUM_ROUNDS; round++) {
        if (bid == 0) {
            // Producer: write 16 words via st_dev (bypass DCache), then
            // set flag via atomicAdd.
            uint32_t pattern = 0xA5000000u | round;
            for (uint32_t i = 0; i < DATA_ELEMS; i++) {
                st_dev_b32(&data[i], pattern + i);
            }
            atomicAdd(flag, 1u);
        } else if (bid == 1) {
            // Consumer: poll flag via ld_dev until >= round+1, then
            // dcci(inval) + normal read 16 words + verify.
            uint32_t target = round + 1;
            while (ld_dev_b32(flag) < target) {
            }
            dcci(data, SINGLE_CACHE_LINE);  // invalidate stale L1 data line
            uint32_t pattern = 0xA5000000u | round;
            for (uint32_t i = 0; i < DATA_ELEMS; i++) {
                if (data[i] != pattern + i) {
                    atomicAdd(errs, 1u);
                    break;
                }
            }
        }
        ccec_barrier(gx, num_blocks, round + 2);
    }
}
