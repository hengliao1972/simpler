// ccec probe: tight-loop concurrent stress. Pure CCEC intrinsic version.
//
// CCEC version of cacheline_stress.asc. Tests tight-loop (no barrier between
// rounds) concurrent access to the same 64B cache line using raw CCEC
// intrinsics (st_dev, ld_dev, dcci). Demonstrates that store+dcci clobbers
// the entire 64B cache line across cores, while st_dev writes are isolated.
//
// Uses lowercase CCEC builtins: st_dev (via st_dev_b32), ld_dev (via
// ld_dev_b32), dcci. NOT AscendC templates.
//
// Layout:
//   gx[0..15]  = test cache line (64B)
//   gx[16..23] = result slots (w0, w1, ..., st_err, dcci_err, etc.)
//
// Modes (2 blocks, tight loop, no barrier between rounds):
//   0 = 2-block tight-loop store+dcci (block0 words 0-1, block1 words 2-3)
//   1 = 2-block tight-loop st_dev     (block0 words 0-3, block1 words 8-11)
//   2 = Mixed: block0 st_dev words 0-3 vs block1 store+dcci words 8-11
//
// Build: see run_concurrent_stress.sh / run_all.sh
#include "ccec_utils.h"

constexpr uint32_t ROUNDS = 1000;

CCEC_PROBE_KERNEL_META(concurrent_stress);

extern "C" __global__ __aicore__ void KERNEL_ENTRY(concurrent_stress)(
    __gm__ uint32_t *gx, uint32_t mode, uint32_t num_blocks)
{
    uint32_t bid = get_block_idx();

    // ================================================================
    // Mode 0: 2-block tight-loop store+dcci (no barrier between rounds)
    //   block0 writes words 0-1 + dcci, block1 writes words 2-3 + dcci.
    //   Each dcci writes back the ENTIRE 64B dirty L1 line -> clobbers the
    //   other block's words.
    // ================================================================
    if (mode == 0) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid < 2) {
            uint32_t my_base = (bid == 0) ? 0xA000A000u : 0xB000B000u;
            uint32_t my_w0 = bid * 2;
            for (uint32_t r = 0; r < ROUNDS; r++) {
                gx[my_w0]     = my_base + r;
                gx[my_w0 + 1] = my_base + r;
                dcci(gx, SINGLE_CACHE_LINE, CACHELINE_OUT);
            }
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 0) {
            uint32_t w0 = ld_dev_b32(&gx[0]);
            uint32_t w1 = ld_dev_b32(&gx[1]);
            uint32_t w2 = ld_dev_b32(&gx[2]);
            uint32_t w3 = ld_dev_b32(&gx[3]);
            uint32_t expected_a = 0xA000A000u + ROUNDS - 1;
            uint32_t expected_b = 0xB000B000u + ROUNDS - 1;
            st_dev_b32(&gx[16], w0);
            st_dev_b32(&gx[17], w1);
            st_dev_b32(&gx[18], w2);
            st_dev_b32(&gx[19], w3);
            st_dev_b32(&gx[20], expected_a);
            st_dev_b32(&gx[21], expected_b);
            uint32_t err = 0;
            if (w0 != expected_a) err++;
            if (w1 != expected_a) err++;
            if (w2 != expected_b) err++;
            if (w3 != expected_b) err++;
            st_dev_b32(&gx[22], err);
        }
        ccec_barrier(gx, num_blocks, 3);

    // ================================================================
    // Mode 1: 2-block tight-loop st_dev
    //   block0 st_dev words 0-3, block1 st_dev words 8-11.
    //   Per-round DSB completes each batch of st_dev writes before the next
    //   batch; the final values are an exact ordered control.
    // ================================================================
    } else if (mode == 1) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 0) {
            for (uint32_t r = 0; r < ROUNDS; r++) {
                st_dev_b32(&gx[0], 0xDEAD0000u + r);
                st_dev_b32(&gx[1], 0xDEAD0000u + r);
                st_dev_b32(&gx[2], 0xDEAD0000u + r);
                st_dev_b32(&gx[3], 0xDEAD0000u + r);
                // This final-round oracle requires round N to complete before
                // round N+1; serialize only this repeated-address test loop.
                dsb(DSB_ALL);
            }
        } else if (bid == 1) {
            for (uint32_t r = 0; r < ROUNDS; r++) {
                st_dev_b32(&gx[8],  0xBEEF0000u + r);
                st_dev_b32(&gx[9],  0xBEEF0000u + r);
                st_dev_b32(&gx[10], 0xBEEF0000u + r);
                st_dev_b32(&gx[11], 0xBEEF0000u + r);
                dsb(DSB_ALL);
            }
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 0) {
            uint32_t st_err = 0, b_err = 0;
            for (uint32_t w = 0; w < 4; w++) {
                if (ld_dev_b32(&gx[w]) != 0xDEAD0000u + ROUNDS - 1) st_err++;
            }
            for (uint32_t w = 8; w < 12; w++) {
                if (ld_dev_b32(&gx[w]) != 0xBEEF0000u + ROUNDS - 1) b_err++;
            }
            st_dev_b32(&gx[16], ld_dev_b32(&gx[0]));
            st_dev_b32(&gx[17], ld_dev_b32(&gx[8]));
            st_dev_b32(&gx[22], st_err);
            st_dev_b32(&gx[23], b_err);
        }
        ccec_barrier(gx, num_blocks, 3);

    // ================================================================
    // Mode 2: Mixed tight-loop — st_dev vs store+dcci
    //   block0 st_dev words 0-3, block1 store+dcci words 8-11.
    //   st_dev writes should survive; store+dcci words may be clobbered.
    // ================================================================
    } else if (mode == 2) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 0) {
            // Block 0: st_dev (bypass DCache) to words 0-3
            for (uint32_t r = 0; r < ROUNDS; r++) {
                st_dev_b32(&gx[0], 0xDEAD0000u + r);
                st_dev_b32(&gx[1], 0xDEAD0000u + r);
                st_dev_b32(&gx[2], 0xDEAD0000u + r);
                st_dev_b32(&gx[3], 0xDEAD0000u + r);
                dsb(DSB_ALL);
            }
        } else if (bid == 1) {
            // Block 1: store + dcci to words 8-11
            for (uint32_t r = 0; r < ROUNDS; r++) {
                gx[8]  = 0xBEEF0000u + r;
                gx[9]  = 0xBEEF0000u + r;
                gx[10] = 0xBEEF0000u + r;
                gx[11] = 0xBEEF0000u + r;
                dcci(gx, SINGLE_CACHE_LINE, CACHELINE_OUT);
            }
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 0) {
            uint32_t st_err = 0, dcci_err = 0;
            // st_dev words should survive
            for (uint32_t w = 0; w < 4; w++) {
                if (ld_dev_b32(&gx[w]) != 0xDEAD0000u + ROUNDS - 1) st_err++;
            }
            // store+dcci words may be clobbered
            for (uint32_t w = 8; w < 12; w++) {
                if (ld_dev_b32(&gx[w]) != 0xBEEF0000u + ROUNDS - 1) dcci_err++;
            }
            st_dev_b32(&gx[16], ld_dev_b32(&gx[0]));
            st_dev_b32(&gx[17], ld_dev_b32(&gx[8]));
            st_dev_b32(&gx[22], st_err);
            st_dev_b32(&gx[23], dcci_err);
        }
        ccec_barrier(gx, num_blocks, 3);
    }
}
