// ccec probe: st_dev / ld_dev intrinsics (bypass DCache). Pure CCEC version.
//
// st_dev and ld_dev are CCEC compiler builtins — the raw hardware
// instructions that AscendC's WriteGmByPassDCache / ReadGmByPassDCache
// wrap. This probe uses them directly (via the ccec_utils.h helpers),
// without the AscendC template layer.
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
// Build: see run_bypass_dcache.sh / run_all.sh
#include "ccec_utils.h"

CCEC_PROBE_KERNEL_META(bypass_dcache_probe);

extern "C" __global__ __aicore__ void KERNEL_ENTRY(bypass_dcache_probe)(
    __gm__ uint32_t *gx, uint32_t mode, uint32_t num_blocks)
{
    uint32_t bid = get_block_idx();

    // ================================================================
    // Mode 0: 1-byte st_dev write blast radius
    // ================================================================
    if (mode == 0) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 1) {
            __gm__ uint8_t *bytes = reinterpret_cast<__gm__ uint8_t*>(gx);
            for (uint32_t i = 0; i < 64; i++) bytes[i] = (uint8_t)(i + 1);
            dcci(gx, SINGLE_CACHE_LINE, CACHELINE_OUT);
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 0) {
            __gm__ uint8_t *bytes = reinterpret_cast<__gm__ uint8_t*>(gx);
            st_dev_b8(&bytes[0], 0xFF);
        }
        ccec_barrier(gx, num_blocks, 3);
        if (bid == 1) {
            dcci(gx, SINGLE_CACHE_LINE);
            __gm__ uint8_t *bytes = reinterpret_cast<__gm__ uint8_t*>(gx);
            uint32_t corrupted = 0;
            for (uint32_t i = 1; i < 64; i++)
                if (bytes[i] != (uint8_t)(i + 1)) corrupted++;
            st_dev_b32(&gx[16], corrupted);
            for (uint32_t i = 0; i <= 15; i++) st_dev_b32(&gx[32 + i], ld_dev_b32(&gx[i]));
        }
        ccec_barrier(gx, num_blocks, 4);

    // ================================================================
    // Mode 1: 4-byte st_dev write blast radius
    // ================================================================
    } else if (mode == 1) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 1) {
            __gm__ uint8_t *bytes = reinterpret_cast<__gm__ uint8_t*>(gx);
            for (uint32_t i = 0; i < 64; i++) bytes[i] = (uint8_t)(i + 1);
            dcci(gx, SINGLE_CACHE_LINE, CACHELINE_OUT);
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 0) {
            st_dev_b32(&gx[0], 0xFFFFFFFFu);
        }
        ccec_barrier(gx, num_blocks, 3);
        if (bid == 1) {
            dcci(gx, SINGLE_CACHE_LINE);
            __gm__ uint8_t *bytes = reinterpret_cast<__gm__ uint8_t*>(gx);
            uint32_t corrupted = 0;
            for (uint32_t i = 4; i < 64; i++)
                if (bytes[i] != (uint8_t)(i + 1)) corrupted++;
            st_dev_b32(&gx[16], corrupted);
            for (uint32_t i = 0; i <= 15; i++) st_dev_b32(&gx[32 + i], ld_dev_b32(&gx[i]));
        }
        ccec_barrier(gx, num_blocks, 4);

    // ================================================================
    // Mode 2: ld_dev read correctness (1B/2B/4B/8B)
    // ================================================================
    } else if (mode == 2) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 1) {
            __gm__ uint8_t *bytes = reinterpret_cast<__gm__ uint8_t*>(gx);
            for (uint32_t i = 0; i < 64; i++)
                st_dev_b8(&bytes[i], (uint8_t)(0x30 + i));
        }
        ccec_barrier(gx, num_blocks, 2);
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
            // 8B
            __gm__ uint64_t *wide = reinterpret_cast<__gm__ uint64_t *>(gx);
            for (uint32_t i = 0; i < 8; i++) {
                uint64_t exp = 0;
                for (uint32_t b = 0; b < 8; b++) {
                    exp |= (uint64_t)(uint8_t)(0x30 + 8 * i + b) << (8 * b);
                }
                if (ld_dev_b64(&wide[i]) != exp) errors++;
            }
            st_dev_b32(&gx[16], errors);
            st_dev_b32(&gx[17], 64 + 32 + 16 + 8);
        }
        ccec_barrier(gx, num_blocks, 3);

    // ================================================================
    // Mode 3: ld_dev vs stale L1
    // ================================================================
    } else if (mode == 3) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 0) {
            volatile __gm__ uint32_t *vgx = gx;
            uint32_t sum = vgx[0] + vgx[1] + vgx[2];
            (void)sum;
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 1) {
            st_dev_b32(&gx[0], 0xDEADBEEFu);
        }
        ccec_barrier(gx, num_blocks, 3);
        if (bid == 0) {
            volatile __gm__ uint32_t *vgx = gx;
            uint32_t normal_val = vgx[0];
            uint32_t bypass_val = ld_dev_b32(&gx[0]);
            st_dev_b32(&gx[17], normal_val);
            st_dev_b32(&gx[18], bypass_val);
            st_dev_b32(&gx[19], 0xDEADBEEFu);
            st_dev_b32(&gx[16], (bypass_val == 0xDEADBEEFu) ? 0 : 1);
        }
        ccec_barrier(gx, num_blocks, 4);

    // ================================================================
    // Mode 4: Concurrent st_dev + atomic (no clobber)
    // ================================================================
    } else if (mode == 4) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 0) {
            for (uint32_t i = 0; i < 100; i++) atomicAdd(&gx[0], 1u);
            // Count the actually launched core-type participants without
            // assuming whether the linked AIC/AIV pair both execute.
            atomicAdd(&gx[18], 100u);
        }
        if (bid == 1) {
            for (uint32_t i = 1; i <= 15; i++)
                st_dev_b32(&gx[i], 0xA000u + i);
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 1) {
            uint32_t g0 = ld_dev_b32(&gx[0]);
            uint32_t corrupted = 0;
            for (uint32_t i = 1; i <= 15; i++)
                if (ld_dev_b32(&gx[i]) != 0xA000u + i) corrupted++;
            st_dev_b32(&gx[16], corrupted);
            st_dev_b32(&gx[17], g0);
        }
        ccec_barrier(gx, num_blocks, 3);

    // ================================================================
    // Mode 5: Concurrent competition — st_dev word[0] vs store+dcci word[3]
    // ================================================================
    } else if (mode == 5) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 1) {
            volatile __gm__ uint32_t *vgx = gx;
            uint32_t s = vgx[0] + vgx[3];
            (void)s;
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 0) {
            st_dev_b32(&gx[0], 0xDEADBEEFu);
        }
        if (bid == 1) {
            gx[3] = 0xCAFEBABEu;
            dcci(gx, SINGLE_CACHE_LINE, CACHELINE_OUT);
        }
        ccec_barrier(gx, num_blocks, 3);
        if (bid == 0) {
            uint32_t w0 = ld_dev_b32(&gx[0]);
            st_dev_b32(&gx[17], w0);
            st_dev_b32(&gx[18], ld_dev_b32(&gx[3]));
            st_dev_b32(&gx[19], 0xDEADBEEFu);
            st_dev_b32(&gx[16], (w0 == 0xDEADBEEFu) ? 0 : 1);
        }
        ccec_barrier(gx, num_blocks, 4);

    // ================================================================
    // Mode 6: Deterministic dcci clobbers st_dev write
    // ================================================================
    } else if (mode == 6) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 1) {
            volatile __gm__ uint32_t *vgx = gx;
            uint32_t s = vgx[0] + vgx[3];
            (void)s;
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 0) {
            st_dev_b32(&gx[0], 0xDEADBEEFu);
        }
        ccec_barrier(gx, num_blocks, 3);
        if (bid == 1) {
            gx[3] = 0xCAFEBABEu;
            dcci(gx, SINGLE_CACHE_LINE, CACHELINE_OUT);
        }
        ccec_barrier(gx, num_blocks, 4);
        if (bid == 0) {
            uint32_t w0 = ld_dev_b32(&gx[0]);
            st_dev_b32(&gx[17], w0);
            st_dev_b32(&gx[18], ld_dev_b32(&gx[3]));
            st_dev_b32(&gx[19], 0xDEADBEEFu);
            st_dev_b32(&gx[16], (w0 == 0) ? 1 : 0);
        }
        ccec_barrier(gx, num_blocks, 5);

    // ================================================================
    // Mode 7: dcci inval fix — both writes survive
    // ================================================================
    } else if (mode == 7) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 1) {
            volatile __gm__ uint32_t *vgx = gx;
            uint32_t s = vgx[0] + vgx[3];
            (void)s;
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 0) {
            st_dev_b32(&gx[0], 0xDEADBEEFu);
        }
        ccec_barrier(gx, num_blocks, 3);
        if (bid == 1) {
            dcci(gx, SINGLE_CACHE_LINE);
            gx[3] = 0xCAFEBABEu;
            dcci(gx, SINGLE_CACHE_LINE, CACHELINE_OUT);
        }
        ccec_barrier(gx, num_blocks, 4);
        if (bid == 0) {
            uint32_t w0 = ld_dev_b32(&gx[0]);
            st_dev_b32(&gx[17], w0);
            st_dev_b32(&gx[18], ld_dev_b32(&gx[3]));
            st_dev_b32(&gx[19], 0xDEADBEEFu);
            st_dev_b32(&gx[16], (w0 == 0xDEADBEEFu) ? 0 : 1);
        }
        ccec_barrier(gx, num_blocks, 5);
    }
}
