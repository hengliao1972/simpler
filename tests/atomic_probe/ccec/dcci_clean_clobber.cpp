// dcci clean-clobber probe. Pure CCEC intrinsic version.
//
// CRITICAL TEST: dcci(ptr, SINGLE_CACHE_LINE) = "clean + invalidate".
// The "clean" step writes back the ENTIRE dirty L1 cache line to HBM.
// If the core has a stale L1 line (from a prior load) that was made
// dirty by a scalar store to ANY word in the line, the clean step
// writes back ALL 64 bytes — including stale values for words the
// core never modified, CLOBBERING another core's st_dev writes to
// those words.
//
// This proves that dcci(inval) is NOT safe when:
//   1. The L1 line is dirty (core did a scalar store)
//   2. Another core wrote to a DIFFERENT word in the same line via st_dev
//
// Mode 0: dcci(inval) on DIRTY line  -> CLOBBERS st_dev write (HAZARD)
// Mode 1: dcci(inval) on CLEAN line  -> no clobber (safe)
// Mode 2: dcci(flush-only) on DIRTY line -> ALSO CLOBBERS
// Mode 3: Control — st_dev survives without any dcci
// Mode 4: dcci(inval) clobber — FULL 64B verification
//
// gx[30] = BARRIER_SLOT reserved for ccec_barrier.
// Build: see run_dcci_clean.sh / run_all.sh
#include "ccec_utils.h"

extern "C" __global__ __aicore__ void KERNEL_ENTRY(dcci_clean_probe)(
    __gm__ uint32_t *gx, uint32_t mode, uint32_t num_blocks)
{
    uint32_t bid = get_block_idx();

    // ================================================================
    // Mode 0: dcci(inval) on DIRTY line -> CLOBBERS st_dev write
    // ================================================================
    if (mode == 0) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 0) {
            gx[0] = 0xAA;  // L1 now dirty: word[0]=0xAA, word[1..15]=0(stale)
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 1) {
            st_dev_b32(&gx[3], 0xCAFEBABEu);
        }
        ccec_barrier(gx, num_blocks, 3);
        if (bid == 0) {
            dcci(gx, SINGLE_CACHE_LINE);  // clean + invalidate
        }
        ccec_barrier(gx, num_blocks, 4);
        if (bid == 1) {
            uint32_t w0 = ld_dev_b32(&gx[0]);  // expect 0xAA
            uint32_t w3 = ld_dev_b32(&gx[3]);  // expect 0 (CLOBBERED!)
            gx[16] = (w0 == 0xAA) ? 0 : 1;     // word[0] ok?
            gx[17] = (w3 == 0) ? 1 : 0;        // word[3] clobbered? 1=yes
            gx[18] = w0;
            gx[19] = w3;
            gx[20] = 0xCAFEBABEu;              // expected word[3]
        }
        ccec_barrier(gx, num_blocks, 5);

    // ================================================================
    // Mode 1: dcci(inval) on CLEAN line -> NO clobber (safe)
    // ================================================================
    } else if (mode == 1) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 0) {
            volatile __gm__ uint32_t *vgx = gx;
            uint32_t s = vgx[0] + vgx[3];  // read-only, L1 stays clean
            (void)s;
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 1) {
            st_dev_b32(&gx[3], 0xCAFEBABEu);
        }
        ccec_barrier(gx, num_blocks, 3);
        if (bid == 0) {
            dcci(gx, SINGLE_CACHE_LINE);
        }
        ccec_barrier(gx, num_blocks, 4);
        if (bid == 1) {
            uint32_t w3 = ld_dev_b32(&gx[3]);
            gx[16] = (w3 == 0xCAFEBABEu) ? 0 : 1;  // 0=ok
            gx[17] = 0;  // N/A for this mode
            gx[18] = 0;
            gx[19] = w3;
            gx[20] = 0xCAFEBABEu;
        }
        ccec_barrier(gx, num_blocks, 5);

    // ================================================================
    // Mode 2: dcci(flush-only) on DIRTY line -> ALSO CLOBBERS
    // ================================================================
    } else if (mode == 2) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 0) {
            gx[0] = 0xAA;  // dirty L1
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 1) {
            st_dev_b32(&gx[3], 0xCAFEBABEu);
        }
        ccec_barrier(gx, num_blocks, 3);
        if (bid == 0) {
            dcci(gx, SINGLE_CACHE_LINE, CACHELINE_OUT);  // clean only (flush)
        }
        ccec_barrier(gx, num_blocks, 4);
        if (bid == 1) {
            uint32_t w0 = ld_dev_b32(&gx[0]);
            uint32_t w3 = ld_dev_b32(&gx[3]);
            gx[16] = (w0 == 0xAA) ? 0 : 1;
            gx[17] = (w3 == 0) ? 1 : 0;  // clobbered?
            gx[18] = w0;
            gx[19] = w3;
            gx[20] = 0xCAFEBABEu;
        }
        ccec_barrier(gx, num_blocks, 5);

    // ================================================================
    // Mode 3: Control — st_dev survives, no dcci at all
    // ================================================================
    } else if (mode == 3) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 0) {
            gx[0] = 0xAA;  // dirty L1, no dcci
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 1) {
            st_dev_b32(&gx[3], 0xCAFEBABEu);
        }
        ccec_barrier(gx, num_blocks, 3);
        // Block 0 does NOT call dcci
        ccec_barrier(gx, num_blocks, 4);
        if (bid == 1) {
            uint32_t w0 = ld_dev_b32(&gx[0]);
            uint32_t w3 = ld_dev_b32(&gx[3]);
            gx[16] = (w0 == 0) ? 0 : 1;     // word[0] NOT visible (stuck in L1)
            gx[17] = (w3 == 0xCAFEBABEu) ? 0 : 1;  // word[3] survived
            gx[18] = w0;
            gx[19] = w3;
            gx[20] = 0xCAFEBABEu;
        }
        ccec_barrier(gx, num_blocks, 5);

    // ================================================================
    // Mode 4: dcci(inval) clobber — FULL 64B verification
    // ================================================================
    } else if (mode == 4) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 0) {
            gx[0] = 0xAA;  // dirty L1, rest stale zeros
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 1) {
            for (uint32_t w = 1; w < 16; w++)
                st_dev_b32(&gx[w], 0xB000u + w);
        }
        ccec_barrier(gx, num_blocks, 3);
        if (bid == 0) {
            dcci(gx, SINGLE_CACHE_LINE);  // clean+inval: writes back [0xAA, 0, 0, ..., 0]
        }
        ccec_barrier(gx, num_blocks, 4);
        if (bid == 1) {
            uint32_t clobbered = 0;
            for (uint32_t w = 1; w < 16; w++) {
                uint32_t v = ld_dev_b32(&gx[w]);
                if (v != 0xB000u + w) clobbered++;
            }
            gx[16] = clobbered;  // expect 15 (all clobbered)
            gx[17] = ld_dev_b32(&gx[0]);  // expect 0xAA (block 0's store)
        }
        ccec_barrier(gx, num_blocks, 5);
    }
}
