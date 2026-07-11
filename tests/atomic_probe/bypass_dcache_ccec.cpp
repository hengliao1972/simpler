// ccec probe: WriteGmByPassDCache / ReadGmByPassDCache via st_dev/ld_dev.
//
// Compiled with ccec -x cce (the low-level path used by fdwic runtime),
// NOT bisheng -xasc.  This allows precise control of compiler flags:
//   -mllvm -cce-aicore-dcci-insert-for-scalar=false
// which is critical — with auto-dcci enabled, scalar stores auto-flush,
// changing the cache behavior under test.
//
// Tests:
//   0 = 1B write blast radius (st_dev uint8)
//   1 = 4B write blast radius (st_dev uint32)
//   2 = ReadGmByPassDCache correctness (1B/2B/4B/8B)
//   3 = ReadGmByPassDCache vs stale L1 normal read
//   4 = Concurrent ByPass write + atomic (no dcci clobber)
//   5 = ReadGmByPassDCache doesn't pollute L1
//
// Build: see run_bypass_dcache.sh
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

// result layout (in separate cache lines from the test data):
//   gx[0..15]    = 64B test cache line
//   gx[16]       = error/corruption count
//   gx[17..18]   = actual / expected values
//   gx[19..34]   = snapshot of test line
//   gx[35]       = after_dcci value (mode 5)

extern "C" __global__ __aicore__ void KERNEL_ENTRY(bypass_dcache_probe)(
    __gm__ uint32_t *gx, uint32_t mode, uint32_t num_blocks)
{
    (void)num_blocks;
    uint32_t bid = GetBlockIdx();

    // ================================================================
    // Mode 0: 1-byte write blast radius
    // ================================================================
    if (mode == 0) {
        SyncAll();
        if (bid == 1) {
            __gm__ uint8_t *bytes = reinterpret_cast<__gm__ uint8_t*>(gx);
            for (uint32_t i = 0; i < 64; i++) bytes[i] = (uint8_t)(i + 1);
            dcci(gx, SINGLE_CACHE_LINE, CACHELINE_OUT);
        }
        SyncAll();
        if (bid == 0) {
            __gm__ uint8_t *bytes = reinterpret_cast<__gm__ uint8_t*>(gx);
            WriteGmByPassDCache<uint8_t>(&bytes[0], 0xFF);
        }
        SyncAll();
        if (bid == 1) {
            dcci(gx, SINGLE_CACHE_LINE);
            __gm__ uint8_t *bytes = reinterpret_cast<__gm__ uint8_t*>(gx);
            uint32_t corrupted = 0;
            for (uint32_t i = 1; i < 64; i++)
                if (bytes[i] != (uint8_t)(i + 1)) corrupted++;
            gx[16] = corrupted;
            for (uint32_t i = 0; i <= 15; i++) gx[17 + i] = gx[i];
        }
        SyncAll();

    // ================================================================
    // Mode 1: 4-byte write blast radius
    // ================================================================
    } else if (mode == 1) {
        SyncAll();
        if (bid == 1) {
            __gm__ uint8_t *bytes = reinterpret_cast<__gm__ uint8_t*>(gx);
            for (uint32_t i = 0; i < 64; i++) bytes[i] = (uint8_t)(i + 1);
            dcci(gx, SINGLE_CACHE_LINE, CACHELINE_OUT);
        }
        SyncAll();
        if (bid == 0) {
            WriteGmByPassDCache<uint32_t>(&gx[0], 0xFFFFFFFFu);
        }
        SyncAll();
        if (bid == 1) {
            dcci(gx, SINGLE_CACHE_LINE);
            __gm__ uint8_t *bytes = reinterpret_cast<__gm__ uint8_t*>(gx);
            uint32_t corrupted = 0;
            for (uint32_t i = 4; i < 64; i++)
                if (bytes[i] != (uint8_t)(i + 1)) corrupted++;
            gx[16] = corrupted;
            for (uint32_t i = 0; i <= 15; i++) gx[17 + i] = gx[i];
        }
        SyncAll();

    // ================================================================
    // Mode 2: ReadGmByPassDCache correctness (1B/2B/4B/8B)
    // ================================================================
    } else if (mode == 2) {
        SyncAll();
        if (bid == 1) {
            __gm__ uint8_t *bytes = reinterpret_cast<__gm__ uint8_t*>(gx);
            for (uint32_t i = 0; i < 64; i++)
                WriteGmByPassDCache<uint8_t>(&bytes[i], (uint8_t)(0x30 + i));
        }
        SyncAll();
        if (bid == 0) {
            uint32_t errors = 0;
            __gm__ uint8_t *bytes = reinterpret_cast<__gm__ uint8_t*>(gx);
            // 1B reads
            for (uint32_t i = 0; i < 64; i++) {
                uint8_t v = ReadGmByPassDCache<uint8_t>(&bytes[i]);
                if (v != (uint8_t)(0x30 + i)) errors++;
            }
            // 2B reads
            __gm__ uint16_t *h = reinterpret_cast<__gm__ uint16_t*>(gx);
            for (uint32_t i = 0; i < 32; i++) {
                uint16_t v = ReadGmByPassDCache<uint16_t>(&h[i]);
                uint16_t exp = (uint16_t)((0x30+2*i) | ((0x31+2*i) << 8));
                if (v != exp) errors++;
            }
            // 4B reads
            for (uint32_t i = 0; i < 16; i++) {
                uint32_t v = ReadGmByPassDCache<uint32_t>(&gx[i]);
                uint32_t b0 = 0x30+4*i, b1 = 0x31+4*i, b2 = 0x32+4*i, b3 = 0x33+4*i;
                uint32_t exp = b0 | (b1<<8) | (b2<<16) | (b3<<24);
                if (v != exp) errors++;
            }
            // 8B reads
            __gm__ uint64_t *q = reinterpret_cast<__gm__ uint64_t*>(gx);
            for (uint32_t i = 0; i < 8; i++) {
                uint64_t v = ReadGmByPassDCache<uint64_t>(&q[i]);
                uint32_t lo = ReadGmByPassDCache<uint32_t>(&gx[2*i]);
                uint32_t hi = ReadGmByPassDCache<uint32_t>(&gx[2*i+1]);
                uint64_t exp = (uint64_t)lo | ((uint64_t)hi << 32);
                if (v != exp) errors++;
            }
            gx[16] = errors;
            gx[17] = 64 + 32 + 16 + 8;
        }
        SyncAll();

    // ================================================================
    // Mode 3: ReadGmByPassDCache vs stale L1
    // ================================================================
    } else if (mode == 3) {
        SyncAll();
        if (bid == 0) {
            // Warm L1 with zeros
            volatile __gm__ uint32_t *vgx = gx;
            uint32_t sum = vgx[0] + vgx[1] + vgx[2];
            (void)sum;
        }
        SyncAll();
        if (bid == 1) {
            WriteGmByPassDCache<uint32_t>(&gx[0], 0xDEADBEEFu);
        }
        SyncAll();
        if (bid == 0) {
            volatile __gm__ uint32_t *vgx = gx;
            uint32_t normal_val = vgx[0];
            uint32_t bypass_val = ReadGmByPassDCache<uint32_t>(&gx[0]);
            gx[17] = normal_val;
            gx[18] = bypass_val;
            gx[19] = 0xDEADBEEFu;
            gx[16] = (bypass_val == 0xDEADBEEFu) ? 0 : 1;
        }
        SyncAll();

    // ================================================================
    // Mode 4: Concurrent ByPass write + atomic
    // ================================================================
    } else if (mode == 4) {
        SyncAll();
#ifndef __DAV_VEC__
        if (bid == 0) {
            for (uint32_t i = 0; i < 100; i++) AtomicAdd(&gx[0], 1u);
        }
#endif
        if (bid == 1) {
            for (uint32_t i = 1; i <= 15; i++)
                WriteGmByPassDCache<uint32_t>(&gx[i], 0xA000u + i);
        }
        SyncAll();
        if (bid == 1) {
            uint32_t g0 = ReadGmByPassDCache<uint32_t>(&gx[0]);
            uint32_t corrupted = 0;
            for (uint32_t i = 1; i <= 15; i++) {
                uint32_t v = ReadGmByPassDCache<uint32_t>(&gx[i]);
                if (v != 0xA000u + i) corrupted++;
            }
            gx[16] = corrupted;
            gx[17] = g0;
            gx[18] = 100;
        }
        SyncAll();

    // ================================================================
    // Mode 5: ReadGmByPassDCache doesn't pollute L1
    //   With ccec -mllvm -cce-aicore-dcci-insert-for-scalar=false,
    //   scalar store gx[0]=0xAA goes to L1 ONLY (no auto-flush).
    //   Then bypass read from HBM shouldn't touch L1.
    //   dcci(CACHELINE_OUT) should write L1's 0xAA back to HBM.
    // ================================================================
    } else if (mode == 5) {
        SyncAll();
        if (bid == 0) {
            gx[0] = 0xAA;  // dirty L1 (no auto-dcci with our flag)
        }
        SyncAll();
        if (bid == 1) {
            WriteGmByPassDCache<uint32_t>(&gx[0], 0xBEEFCAFEu);
        }
        SyncAll();
        if (bid == 0) {
            uint32_t bypass_val = ReadGmByPassDCache<uint32_t>(&gx[0]);
            volatile __gm__ uint32_t *vgx = gx;
            uint32_t normal_after = vgx[0];
            // Flush L1 to HBM
            dcci(gx, SINGLE_CACHE_LINE, CACHELINE_OUT);
            gx[17] = bypass_val;     // 0xBEEFCAFE (from HBM)
            gx[18] = normal_after;   // 0xAA (L1 not polluted)
            gx[19] = 0xAA;
            gx[20] = 0xBEEFCAFEu;
            gx[16] = (normal_after == 0xAA) ? 0 : 1;
        }
        SyncAll();
        if (bid == 1) {
            uint32_t after = ReadGmByPassDCache<uint32_t>(&gx[0]);
            gx[35] = after;  // 0xAA (dcci wrote L1's 0xAA back)
        }
        SyncAll();

    // ================================================================
    // Mode 6: Partial write + cross-core correct read (end-to-end)
    //
    // Core question: one core writes PART of a cache line (1B/2B/4B
    // at different offsets).  Can another core correctly read each
    // written fragment AND verify unwritten bytes are NOT corrupted?
    //
    // Writer (block 0): 3 partial writes at different widths/offsets
    //   word[0]      = 0xDEADBEEF  (4 bytes, offset 0)
    //   byte[8]      = 0xAB        (1 byte,  offset 8)
    //   half[5]      = 0xBABE      (2 bytes, offset 10)
    //
    // Reader (block 1): verifies via 3 read methods
    //   a) ReadGmByPassDCache — always fresh from HBM
    //   b) dcci(inval) + normal read — fresh after cache line invalidate
    //   c) normal read WITHOUT inval — may be stale (if L1 was warmed)
    //
    // Also verifies ALL 64 bytes: written = expected, unwritten = 0.
    // ================================================================
    } else if (mode == 6) {
        SyncAll();

        // Reader (block 1): warm L1 with all zeros BEFORE writer acts
        if (bid == 1) {
            volatile __gm__ uint8_t *vb = reinterpret_cast<__gm__ uint8_t*>(gx);
            uint32_t sum = 0;
            for (uint32_t i = 0; i < 64; i++) sum += vb[i];
            (void)sum;  // L1 now has all-zero cache line
        }

        SyncAll();

        // Writer (block 0): 3 partial ByPassDCache writes
        if (bid == 0) {
            // 4-byte write at word[0] (bytes 0-3)
            WriteGmByPassDCache<uint32_t>(&gx[0], 0xDEADBEEFu);
            // 1-byte write at byte[8] (word[2], low byte)
            __gm__ uint8_t *bytes = reinterpret_cast<__gm__ uint8_t*>(gx);
            WriteGmByPassDCache<uint8_t>(&bytes[8], 0xAB);
            // 2-byte write at half[5] (bytes 10-11)
            __gm__ uint16_t *half = reinterpret_cast<__gm__ uint16_t*>(gx);
            WriteGmByPassDCache<uint16_t>(&half[5], 0xBABEu);
        }

        SyncAll();

        // Reader (block 1): read back via 3 methods, verify everything
        if (bid == 1) {
            __gm__ uint8_t *bytes = reinterpret_cast<__gm__ uint8_t*>(gx);
            uint32_t errors_bypass = 0;
            uint32_t errors_inval  = 0;
            uint32_t errors_stale  = 0;

            // --- Method A: ReadGmByPassDCache (always fresh) ---
            // Check written bytes
            uint32_t w0 = ReadGmByPassDCache<uint32_t>(&gx[0]);
            if (w0 != 0xDEADBEEFu) errors_bypass++;
            uint8_t b8 = ReadGmByPassDCache<uint8_t>(&bytes[8]);
            if (b8 != 0xAB) errors_bypass++;
            uint16_t h5 = ReadGmByPassDCache<uint16_t>(
                reinterpret_cast<__gm__ uint16_t*>(&bytes[10]));
            if (h5 != 0xBABEu) errors_bypass++;
            // Check ALL 64 bytes: written = expected, unwritten = 0
            for (uint32_t i = 0; i < 64; i++) {
                uint8_t v = ReadGmByPassDCache<uint8_t>(&bytes[i]);
                uint8_t exp = 0;
                if (i < 4) exp = (uint8_t)(0xDEADBEEFu >> (i * 8));
                else if (i == 8) exp = 0xAB;
                else if (i == 10) exp = (uint8_t)0xBABEu;       // low byte
                else if (i == 11) exp = (uint8_t)(0xBABEu >> 8); // high byte
                if (v != exp) errors_bypass++;
            }

            // --- Method B: dcci inval + normal read (fresh) ---
            dcci(gx, SINGLE_CACHE_LINE);  // invalidate L1
            volatile __gm__ uint32_t *vgx = gx;
            uint32_t w0_inval = vgx[0];
            if (w0_inval != 0xDEADBEEFu) errors_inval++;
            volatile __gm__ uint8_t *vb = bytes;
            if (vb[8] != 0xAB) errors_inval++;
            volatile __gm__ uint16_t *vh = reinterpret_cast<volatile __gm__ uint16_t*>(gx);
            if (vh[5] != 0xBABEu) errors_inval++;
            // Check unwritten word[1] is still 0
            if (vgx[1] != 0) errors_inval++;

            // --- Method C: normal read WITHOUT inval ---
            // After method B's inval + reads, L1 is now warm with the
            // correct HBM data.  So a second round of normal reads
            // without inval should also be correct (L1 was populated
            // by method B's reads).
            // To test true stale behavior, we'd need a separate run.
            // For now, just verify current normal reads are consistent.
            uint32_t w0_normal = vgx[0];
            if (w0_normal != 0xDEADBEEFu) errors_stale++;

            // Pack results
            gx[16] = errors_bypass;  // expect 0
            gx[17] = errors_inval;   // expect 0
            gx[18] = errors_stale;   // expect 0
            gx[19] = w0;             // word[0] via bypass
            gx[20] = w0_inval;       // word[0] via inval+normal
            gx[21] = (uint32_t)b8;   // byte[8] via bypass
            gx[22] = (uint32_t)h5;   // half[5] via bypass
            // Snapshot all 64 bytes (via bypass for ground truth)
            for (uint32_t i = 0; i < 16; i++)
                gx[23 + i] = ReadGmByPassDCache<uint32_t>(&gx[i]);
        }
        SyncAll();
    }
}
