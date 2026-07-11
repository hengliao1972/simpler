// dcci clean-clobber probe.
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
// Mode 0: dcci(inval) on DIRTY line → CLOBBERS st_dev write (HAZARD)
// Mode 1: dcci(inval) on CLEAN line → no clobber (safe)
// Mode 2: dcci(flush-only) on DIRTY line → ALSO CLOBBERS
// Mode 3: Control — st_dev survives without any dcci
//
// Build: see run_dcci_clean.sh
#include "kernel_operator.h"
using namespace AscendC;

#ifndef SINGLE_CACHE_LINE
#define SINGLE_CACHE_LINE 0
#endif
#ifndef CACHELINE_OUT
#define CACHELINE_OUT 2
#endif

#ifdef __DAV_VEC__
#define KERNEL_ENTRY(x) x##_0_mix_aiv
#else
#define KERNEL_ENTRY(x) x##_0_mix_aic
#endif

__aicore__ inline void st_dev_b32(__gm__ uint32_t *addr, uint32_t val) {
    st_dev(val, addr, 0);
}
__aicore__ inline uint32_t ld_dev_b32(__gm__ uint32_t *addr) {
    return ld_dev(addr, 0);
}

extern "C" __global__ __aicore__ void KERNEL_ENTRY(dcci_clean_probe)(
    __gm__ uint32_t *gx, uint32_t mode, uint32_t num_blocks)
{
    (void)num_blocks;
    uint32_t bid = GetBlockIdx();

    // ================================================================
    // Mode 0: dcci(inval) on DIRTY line → CLOBBERS st_dev write
    //
    // Block 0: scalar store gx[0]=0xAA (L1 dirty, word[3] still stale 0)
    // Block 1: st_dev word[3] = 0xCAFEBABE (HBM word[3] fresh)
    // Block 0: dcci(ptr, 0) = clean+inval
    //   → clean writes back [0xAA, 0, 0, 0, ...] → HBM word[3] = 0 CLOBBERED!
    //   → inval discards L1 line
    // Verify: word[0]=0xAA(ok), word[3]=0(CLOBBERED)
    // ================================================================
    if (mode == 0) {
        SyncAll();
        // Block 0: dirty L1 with scalar store to word[0]
        if (bid == 0) {
            gx[0] = 0xAA;  // L1 now dirty: word[0]=0xAA, word[1..15]=0(stale)
        }
        SyncAll();
        // Block 1: st_dev write word[3] directly to HBM
        if (bid == 1) {
            st_dev_b32(&gx[3], 0xCAFEBABEu);
        }
        SyncAll();
        // Block 0: dcci(clean+inval) — clean step writes back dirty line!
        if (bid == 0) {
            dcci(gx, SINGLE_CACHE_LINE);  // clean + invalidate
        }
        SyncAll();
        // Verify via ld_dev (bypass read)
        if (bid == 1) {
            uint32_t w0 = ld_dev_b32(&gx[0]);  // expect 0xAA
            uint32_t w3 = ld_dev_b32(&gx[3]);  // expect 0 (CLOBBERED!)
            gx[16] = (w0 == 0xAA) ? 0 : 1;     // word[0] ok?
            gx[17] = (w3 == 0) ? 1 : 0;        // word[3] clobbered? 1=yes
            gx[18] = w0;
            gx[19] = w3;
            gx[20] = 0xCAFEBABEu;              // expected word[3]
        }
        SyncAll();

    // ================================================================
    // Mode 1: dcci(inval) on CLEAN line → NO clobber (safe)
    //
    // Block 0: only volatile reads (L1 clean, no stores)
    // Block 1: st_dev word[3] = 0xCAFEBABE
    // Block 0: dcci(ptr, 0) = clean+inval
    //   → clean is no-op (L1 not dirty) → NO writeback → NO clobber
    //   → inval discards L1 line
    // Verify: word[3] = 0xCAFEBABE (survived!)
    // ================================================================
    } else if (mode == 1) {
        SyncAll();
        // Block 0: load cache line via reads only (L1 clean)
        if (bid == 0) {
            volatile __gm__ uint32_t *vgx = gx;
            uint32_t s = vgx[0] + vgx[3];  // read-only, L1 stays clean
            (void)s;
        }
        SyncAll();
        // Block 1: st_dev write word[3]
        if (bid == 1) {
            st_dev_b32(&gx[3], 0xCAFEBABEu);
        }
        SyncAll();
        // Block 0: dcci(clean+inval) — clean is no-op (L1 clean)
        if (bid == 0) {
            dcci(gx, SINGLE_CACHE_LINE);
        }
        SyncAll();
        // Verify
        if (bid == 1) {
            uint32_t w3 = ld_dev_b32(&gx[3]);
            gx[16] = (w3 == 0xCAFEBABEu) ? 0 : 1;  // 0=ok
            gx[17] = 0;  // N/A for this mode
            gx[18] = 0;
            gx[19] = w3;
            gx[20] = 0xCAFEBABEu;
        }
        SyncAll();

    // ================================================================
    // Mode 2: dcci(flush-only) on DIRTY line → ALSO CLOBBERS
    //
    // Same as mode 0 but uses 3-arg dcci(ptr, 0, CACHELINE_OUT).
    // This is "clean only" (writeback, keep in L1) — same clobber risk.
    // ================================================================
    } else if (mode == 2) {
        SyncAll();
        if (bid == 0) {
            gx[0] = 0xAA;  // dirty L1
        }
        SyncAll();
        if (bid == 1) {
            st_dev_b32(&gx[3], 0xCAFEBABEu);
        }
        SyncAll();
        if (bid == 0) {
            dcci(gx, SINGLE_CACHE_LINE, CACHELINE_OUT);  // clean only (flush)
        }
        SyncAll();
        if (bid == 1) {
            uint32_t w0 = ld_dev_b32(&gx[0]);
            uint32_t w3 = ld_dev_b32(&gx[3]);
            gx[16] = (w0 == 0xAA) ? 0 : 1;
            gx[17] = (w3 == 0) ? 1 : 0;  // clobbered?
            gx[18] = w0;
            gx[19] = w3;
            gx[20] = 0xCAFEBABEu;
        }
        SyncAll();

    // ================================================================
    // Mode 3: Control — st_dev survives, no dcci at all
    //
    // Block 0: scalar store word[0] (dirty L1, but NO dcci)
    // Block 1: st_dev word[3]
    // Block 0: does NOT call dcci → no writeback → no clobber
    //   (but word[0] stays in L1, not visible to HBM)
    // Verify: word[3] survives, word[0] NOT visible (stuck in L1)
    // ================================================================
    } else if (mode == 3) {
        SyncAll();
        if (bid == 0) {
            gx[0] = 0xAA;  // dirty L1, no dcci
        }
        SyncAll();
        if (bid == 1) {
            st_dev_b32(&gx[3], 0xCAFEBABEu);
        }
        SyncAll();
        // Block 0 does NOT call dcci
        SyncAll();
        if (bid == 1) {
            uint32_t w0 = ld_dev_b32(&gx[0]);
            uint32_t w3 = ld_dev_b32(&gx[3]);
            gx[16] = (w0 == 0) ? 0 : 1;     // word[0] NOT visible (stuck in L1)
            gx[17] = (w3 == 0xCAFEBABEu) ? 0 : 1;  // word[3] survived
            gx[18] = w0;
            gx[19] = w3;
            gx[20] = 0xCAFEBABEu;
        }
        SyncAll();

    // ================================================================
    // Mode 4: dcci(inval) clobber — FULL 64B verification
    //
    // Block 0: scalar store word[0], then dcci(inval)
    // Block 1: st_dev to ALL 15 other words
    // Check which words survive the dcci clean writeback.
    // Expected: ALL 15 st_dev words CLOBBERED (clean writes entire line)
    // ================================================================
    } else if (mode == 4) {
        SyncAll();
        if (bid == 0) {
            gx[0] = 0xAA;  // dirty L1, rest stale zeros
        }
        SyncAll();
        if (bid == 1) {
            // Write words 1-15 via st_dev
            for (uint32_t w = 1; w < 16; w++)
                st_dev_b32(&gx[w], 0xB000u + w);
        }
        SyncAll();
        if (bid == 0) {
            dcci(gx, SINGLE_CACHE_LINE);  // clean+inval: writes back [0xAA, 0, 0, ..., 0]
        }
        SyncAll();
        if (bid == 1) {
            uint32_t clobbered = 0;
            for (uint32_t w = 1; w < 16; w++) {
                uint32_t v = ld_dev_b32(&gx[w]);
                if (v != 0xB000u + w) clobbered++;
            }
            gx[16] = clobbered;  // expect 15 (all clobbered)
            gx[17] = ld_dev_b32(&gx[0]);  // expect 0xAA (block 0's store)
        }
        SyncAll();
    }
}
