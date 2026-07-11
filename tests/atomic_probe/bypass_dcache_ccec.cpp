// ccec probe: st_dev / ld_dev intrinsics (bypass DCache).
//
// st_dev and ld_dev are CCEC compiler builtins — the raw hardware
// instructions that AscendC's WriteGmByPassDCache / ReadGmByPassDCache
// wrap.  This probe uses them directly, without the AscendC template
// layer, to demonstrate the low-level CCEC intrinsic usage.
//
// st_dev signature:  st_dev(value, __gm__ ptr, flag=0)
// ld_dev signature:  ld_dev(__gm__ ptr, flag=0) -> value
//
// Tests:
//   0 = 1B st_dev write blast radius
//   1 = 4B st_dev write blast radius
//   2 = ld_dev read correctness (1B/2B/4B/8B)
//   3 = ld_dev vs stale L1 normal read
//   4 = Concurrent st_dev write + atomic (no clobber)
//   5 = st_dev write + scalar store+dcci to DIFFERENT words (competition)
//   6 = st_dev write then scalar store+dcci to SAME word (clobber hazard)
//   7 = Concurrent: block0 st_dev word[0] vs block1 store+dcci word[3]
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

// ---- Thin wrappers around st_dev / ld_dev for readability ----
// (These call the raw CCEC builtins, NOT AscendC's templates.)

__aicore__ inline void st_dev_b8(__gm__ uint8_t *addr, uint8_t val) {
    st_dev(val, addr, 0);
}
__aicore__ inline void st_dev_b16(__gm__ uint16_t *addr, uint16_t val) {
    st_dev(val, addr, 0);
}
__aicore__ inline void st_dev_b32(__gm__ uint32_t *addr, uint32_t val) {
    st_dev(val, addr, 0);
}
__aicore__ inline uint8_t ld_dev_b8(__gm__ uint8_t *addr) {
    return ld_dev(addr, 0);
}
__aicore__ inline uint16_t ld_dev_b16(__gm__ uint16_t *addr) {
    return ld_dev(addr, 0);
}
__aicore__ inline uint32_t ld_dev_b32(__gm__ uint32_t *addr) {
    return ld_dev(addr, 0);
}

extern "C" __global__ __aicore__ void KERNEL_ENTRY(bypass_dcache_probe)(
    __gm__ uint32_t *gx, uint32_t mode, uint32_t num_blocks)
{
    (void)num_blocks;
    uint32_t bid = GetBlockIdx();

    // ================================================================
    // Mode 0: 1-byte st_dev write blast radius
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
            st_dev_b8(&bytes[0], 0xFF);
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
    // Mode 1: 4-byte st_dev write blast radius
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
            st_dev_b32(&gx[0], 0xFFFFFFFFu);
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
    // Mode 2: ld_dev read correctness (1B/2B/4B)
    // ================================================================
    } else if (mode == 2) {
        SyncAll();
        if (bid == 1) {
            __gm__ uint8_t *bytes = reinterpret_cast<__gm__ uint8_t*>(gx);
            for (uint32_t i = 0; i < 64; i++)
                st_dev_b8(&bytes[i], (uint8_t)(0x30 + i));
        }
        SyncAll();
        if (bid == 0) {
            uint32_t errors = 0;
            __gm__ uint8_t *bytes = reinterpret_cast<__gm__ uint8_t*>(gx);
            __gm__ uint16_t *h = reinterpret_cast<__gm__ uint16_t*>(gx);
            // 1B
            for (uint32_t i = 0; i < 64; i++)
                if (ld_dev_b8(&bytes[i]) != (uint8_t)(0x30 + i)) errors++;
            // 2B
            for (uint32_t i = 0; i < 32; i++) {
                uint16_t exp = (uint16_t)((0x30+2*i) | ((0x31+2*i) << 8));
                if (ld_dev_b16(&h[i]) != exp) errors++;
            }
            // 4B
            for (uint32_t i = 0; i < 16; i++) {
                uint32_t b0=0x30+4*i, b1=0x31+4*i, b2=0x32+4*i, b3=0x33+4*i;
                uint32_t exp = b0|(b1<<8)|(b2<<16)|(b3<<24);
                if (ld_dev_b32(&gx[i]) != exp) errors++;
            }
            gx[16] = errors;
            gx[17] = 64 + 32 + 16;
        }
        SyncAll();

    // ================================================================
    // Mode 3: ld_dev vs stale L1
    // ================================================================
    } else if (mode == 3) {
        SyncAll();
        if (bid == 0) {
            volatile __gm__ uint32_t *vgx = gx;
            uint32_t sum = vgx[0] + vgx[1] + vgx[2];
            (void)sum;
        }
        SyncAll();
        if (bid == 1) {
            st_dev_b32(&gx[0], 0xDEADBEEFu);
        }
        SyncAll();
        if (bid == 0) {
            volatile __gm__ uint32_t *vgx = gx;
            uint32_t normal_val = vgx[0];
            uint32_t bypass_val = ld_dev_b32(&gx[0]);
            gx[17] = normal_val;
            gx[18] = bypass_val;
            gx[19] = 0xDEADBEEFu;
            gx[16] = (bypass_val == 0xDEADBEEFu) ? 0 : 1;
        }
        SyncAll();

    // ================================================================
    // Mode 4: Concurrent st_dev + atomic (no clobber)
    // ================================================================
    } else if (mode == 4) {
        SyncAll();
        if (bid == 0) {
            for (uint32_t i = 0; i < 100; i++) AtomicAdd(&gx[0], 1u);
        }
        if (bid == 1) {
            for (uint32_t i = 1; i <= 15; i++)
                st_dev_b32(&gx[i], 0xA000u + i);
        }
        SyncAll();
        if (bid == 1) {
            uint32_t g0 = ld_dev_b32(&gx[0]);
            uint32_t corrupted = 0;
            for (uint32_t i = 1; i <= 15; i++)
                if (ld_dev_b32(&gx[i]) != 0xA000u + i) corrupted++;
            gx[16] = corrupted;
            gx[17] = g0;
            gx[18] = 100;
        }
        SyncAll();

    // ================================================================
    // Mode 5: Concurrent competition — st_dev word[0] vs store+dcci word[3]
    // ================================================================
    } else if (mode == 5) {
        SyncAll();
        if (bid == 1) {
            volatile __gm__ uint32_t *vgx = gx;
            uint32_t s = vgx[0] + vgx[3];
            (void)s;
        }
        SyncAll();
        if (bid == 0) {
            st_dev_b32(&gx[0], 0xDEADBEEFu);
        }
        if (bid == 1) {
            gx[3] = 0xCAFEBABEu;
            dcci(gx, SINGLE_CACHE_LINE, CACHELINE_OUT);
        }
        SyncAll();
        if (bid == 0) {
            gx[17] = ld_dev_b32(&gx[0]);
            gx[18] = ld_dev_b32(&gx[3]);
            gx[19] = 0xDEADBEEFu;
            gx[16] = (gx[17] == 0xDEADBEEFu) ? 0 : 1;
        }
        SyncAll();

    // ================================================================
    // Mode 6: Deterministic dcci clobbers st_dev write
    // ================================================================
    } else if (mode == 6) {
        SyncAll();
        if (bid == 1) {
            volatile __gm__ uint32_t *vgx = gx;
            uint32_t s = vgx[0] + vgx[3];
            (void)s;
        }
        SyncAll();
        if (bid == 0) {
            st_dev_b32(&gx[0], 0xDEADBEEFu);
        }
        SyncAll();
        if (bid == 1) {
            gx[3] = 0xCAFEBABEu;
            dcci(gx, SINGLE_CACHE_LINE, CACHELINE_OUT);
        }
        SyncAll();
        if (bid == 0) {
            gx[17] = ld_dev_b32(&gx[0]);
            gx[18] = ld_dev_b32(&gx[3]);
            gx[19] = 0xDEADBEEFu;
            gx[16] = (gx[17] == 0) ? 1 : 0;
        }
        SyncAll();

    // ================================================================
    // Mode 7: dcci inval fix — both writes survive
    // ================================================================
    } else if (mode == 7) {
        SyncAll();
        if (bid == 1) {
            volatile __gm__ uint32_t *vgx = gx;
            uint32_t s = vgx[0] + vgx[3];
            (void)s;
        }
        SyncAll();
        if (bid == 0) {
            st_dev_b32(&gx[0], 0xDEADBEEFu);
        }
        SyncAll();
        if (bid == 1) {
            dcci(gx, SINGLE_CACHE_LINE);
            gx[3] = 0xCAFEBABEu;
            dcci(gx, SINGLE_CACHE_LINE, CACHELINE_OUT);
        }
        SyncAll();
        if (bid == 0) {
            gx[17] = ld_dev_b32(&gx[0]);
            gx[18] = ld_dev_b32(&gx[3]);
            gx[19] = 0xDEADBEEFu;
            gx[16] = (gx[17] == 0xDEADBEEFu) ? 0 : 1;
        }
        SyncAll();
    }
}
