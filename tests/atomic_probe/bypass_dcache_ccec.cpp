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
    }
}
