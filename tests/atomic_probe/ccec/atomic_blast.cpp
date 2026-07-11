// ccec probe: atomicMax blast radius. Pure CCEC intrinsic version.
//
// CCEC version of cacheline_blast.asc. Tests whether the lowercase CCEC
// builtin atomicMax (raw hardware atomic that bypasses L1) has a blast
// radius limited to the atomic width, or whether it clobbers neighbour
// bytes/words in the same 64B cache line.
//
// Uses atomicMax (lowercase CCEC builtin), NOT AscendC's AtomicMax.
//
// Layout (gx[30] = BARRIER_SLOT reserved for ccec_barrier):
//   gx[0..15]  = test cache line (64B = 16 x uint32)
//   gx[16]     = corruption count (defender reports clobbered neighbours)
//   gx[17]     = gx[0] actual value after test
//   gx[18]     = gx[0] expected value
//   gx[31..46] = snapshot of gx[0..15] after test (debugging)
//
// Modes:
//   0 = Sequential:   block1 fills 64B sentinels (bytes[i]=i+1) + flush,
//                     block0 atomicMax(&gx[0], 0xFFFFFFFF), verify bytes 4-63.
//   1 = Stale-L1:     block0 warms L1 (volatile reads), block1 fills word
//                     sentinels + flush, block0 atomicMax, verify words 1-15.
//   2 = 4B blast:     same as stale-L1 but byte-level check (bytes 4-63).
//
// Build: see run_atomic_blast.sh / run_all.sh
#include "ccec_utils.h"

constexpr uint32_t ATOMIC_TARGET = 0xFFFFFFFFu;  // atomicMax target for gx[0]
constexpr uint32_t SENTINEL_BASE = 0xA000u;      // gx[1]=0xA001, gx[2]=0xA002, ...

// Fill gx[16..18] summary + gx[31..46] snapshot for host consumption.
__aicore__ inline void report_word_result(__gm__ uint32_t *gx, uint32_t gx0_expected) {
    uint32_t corrupted = 0;
    for (uint32_t i = 1; i <= 15; i++) {
        if (gx[i] != SENTINEL_BASE + i) corrupted++;
    }
    gx[16] = corrupted;
    gx[17] = gx[0];
    gx[18] = gx0_expected;
    for (uint32_t i = 0; i <= 15; i++) gx[31 + i] = gx[i];
}

extern "C" __global__ __aicore__ void KERNEL_ENTRY(atomic_blast)(
    __gm__ uint32_t *gx, uint32_t mode, uint32_t num_blocks)
{
    uint32_t bid = get_block_idx();

    // ================================================================
    // Mode 0: Sequential — block1 fills 64B sentinels + flush, block0 atomicMax
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
            atomicMax(&gx[0], ATOMIC_TARGET);
        }
        ccec_barrier(gx, num_blocks, 3);
        if (bid == 1) {
            dcci(gx, SINGLE_CACHE_LINE);
            __gm__ uint8_t *bytes = reinterpret_cast<__gm__ uint8_t*>(gx);
            uint32_t corrupted = 0;
            for (uint32_t i = 4; i < 64; i++) {
                if (bytes[i] != (uint8_t)(i + 1)) corrupted++;
            }
            gx[16] = corrupted;
            gx[17] = gx[0];
            gx[18] = ATOMIC_TARGET;
            for (uint32_t i = 0; i <= 15; i++) gx[31 + i] = gx[i];
        }
        ccec_barrier(gx, num_blocks, 4);

    // ================================================================
    // Mode 1: Stale-L1 — block0 warms L1, then atomicMax (word-level check)
    // ================================================================
    } else if (mode == 1) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 0) {
            volatile __gm__ uint32_t *vgx = gx;
            uint32_t sum = 0;
            for (uint32_t i = 0; i <= 15; i++) sum += vgx[i];
            (void)sum;
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 1) {
            for (uint32_t i = 1; i <= 15; i++) gx[i] = SENTINEL_BASE + i;
            dcci(gx, SINGLE_CACHE_LINE, CACHELINE_OUT);
        }
        ccec_barrier(gx, num_blocks, 3);
        if (bid == 0) {
            atomicMax(&gx[0], 0xBEEFu);
        }
        ccec_barrier(gx, num_blocks, 4);
        if (bid == 1) {
            dcci(gx, SINGLE_CACHE_LINE);
            report_word_result(gx, 0xBEEFu);
        }
        ccec_barrier(gx, num_blocks, 5);

    // ================================================================
    // Mode 2: 4B blast — stale-L1 + byte-level check (bytes 4-63)
    // ================================================================
    } else if (mode == 2) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 0) {
            volatile __gm__ uint8_t *vb = reinterpret_cast<__gm__ uint8_t*>(gx);
            uint32_t sum = 0;
            for (uint32_t i = 0; i < 64; i++) sum += vb[i];
            (void)sum;
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 1) {
            __gm__ uint8_t *bytes = reinterpret_cast<__gm__ uint8_t*>(gx);
            for (uint32_t i = 0; i < 64; i++) bytes[i] = (uint8_t)(i + 1);
            dcci(gx, SINGLE_CACHE_LINE, CACHELINE_OUT);
        }
        ccec_barrier(gx, num_blocks, 3);
        if (bid == 0) {
            atomicMax(&gx[0], ATOMIC_TARGET);
        }
        ccec_barrier(gx, num_blocks, 4);
        if (bid == 1) {
            dcci(gx, SINGLE_CACHE_LINE);
            __gm__ uint8_t *bytes = reinterpret_cast<__gm__ uint8_t*>(gx);
            uint32_t corrupted = 0;
            for (uint32_t i = 4; i < 64; i++) {
                if (bytes[i] != (uint8_t)(i + 1)) corrupted++;
            }
            gx[16] = corrupted;
            gx[17] = gx[0];
            gx[18] = ATOMIC_TARGET;
            for (uint32_t i = 0; i <= 15; i++) gx[31 + i] = gx[i];
        }
        ccec_barrier(gx, num_blocks, 5);
    }
}
