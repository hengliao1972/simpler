#ifndef CCEC_UTILS_H
#define CCEC_UTILS_H

// Pure CCEC intrinsic utilities — no AscendC headers, no kernel_operator.h.
// All operations use raw __builtin_cce_* compiler builtins.

#ifdef __DAV_VEC__
#define KERNEL_ENTRY(x) x##_0_mix_aiv
#else
#define KERNEL_ENTRY(x) x##_0_mix_aic
#endif

#ifndef SINGLE_CACHE_LINE
#define SINGLE_CACHE_LINE 0
#endif
#ifndef CACHELINE_OUT
#define CACHELINE_OUT 2
#endif

// ---- Pipe constants ----
#define CCEC_PIPE_S    0
#define CCEC_PIPE_MTE3 5
#define CCEC_PIPE_ALL  6
#define CCEC_PIPE_FIX  10

// ---- Sync flag IDs (from AscendC dav_3510/kernel_operator_sync_impl.h) ----
#define SYNC_AIC_FLAG     11
#define SYNC_AIV_FLAG     12
#define SYNC_AIC_AIV_FLAG 13

// ---- Cross-block barrier (pure CCEC) ----
// A5 SyncAll is a software protocol requiring workspace memory (GM+UB),
// pipe management, and AscendC runtime — cannot be replicated with a few
// __builtin_cce_* calls. This atomic-based barrier is the pure-CCEC
// alternative: atomicAdd counter + ld_dev poll. No workspace needed.
//
// Usage: call ccec_barrier(gx, num_blocks, round) with incrementing round.
// gx[BARRIER_SLOT] must be zeroed by host before each kernel launch.
#define BARRIER_SLOT 30
#define ccec_barrier(gx, nb, round) do { \
    atomicAdd(&(gx)[BARRIER_SLOT], 1u); \
    uint32_t _tgt = (uint32_t)(nb) * (uint32_t)(round); \
    while ((uint32_t)__builtin_cce_ld_dev(&(gx)[BARRIER_SLOT], 0) < _tgt) {} \
} while (0)

// ---- st_dev / ld_dev ----
#define st_dev_b32(addr, val) __builtin_cce_st_dev((val), (addr), 0)
#define st_dev_b16(addr, val) __builtin_cce_st_dev((val), (addr), 0)
#define st_dev_b8(addr, val)  __builtin_cce_st_dev((val), (addr), 0)
#define ld_dev_b32(addr) ((uint32_t)__builtin_cce_ld_dev((addr), 0))
#define ld_dev_b16(addr) ((uint16_t)__builtin_cce_ld_dev((addr), 0))
#define ld_dev_b8(addr)  ((uint8_t)__builtin_cce_ld_dev((addr), 0))

// ---- dcci shorthand ----
#define dcci_flush(p) __builtin_cce_dcci((p), SINGLE_CACHE_LINE, CACHELINE_OUT)
#define dcci_inval(p) __builtin_cce_dcci((p), SINGLE_CACHE_LINE)

#endif
