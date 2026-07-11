#ifndef CCEC_UTILS_H
#define CCEC_UTILS_H

// Pure CCEC intrinsic utilities — no AscendC headers, no kernel_operator.h.
// All operations use raw __builtin_cce_* compiler builtins.
#include "cce_aicore_intrinsics.h"
#include <pto/common/kernel_meta.hpp>

#ifdef __DAV_VEC__
#define KERNEL_ENTRY(x) x##_0_mix_aiv
#else
#define KERNEL_ENTRY(x) x##_0_mix_aic
#endif

// Reuse the function-level metadata used by PTO-ISA's A5 SyncAll tests.
// The default probe runner emits AIV-only metadata; MIX metadata remains
// available only for explicit supplementary builds.
#define CCEC_PTO_MIX_1_1_META(name) PTO_SYNCALL_MIX_AIC_KERNEL_META(name, 1, 1)
#define CCEC_PTO_AIV_META(name) PTO_SYNCALL_AIV_KERNEL_META(name)
#ifdef __DAV_VEC__
#define CCEC_MIX_1_1_KERNEL_META(name) CCEC_PTO_MIX_1_1_META(name##_0_mix_aiv)
#define CCEC_AIV_ONLY_KERNEL_META(name) CCEC_PTO_AIV_META(name##_0_mix_aiv)
#else
#define CCEC_MIX_1_1_KERNEL_META(name) CCEC_PTO_MIX_1_1_META(name##_0_mix_aic)
#endif

#if defined(CCEC_SYNC_AIV_ONLY)
#define CCEC_PROBE_KERNEL_META(name) CCEC_AIV_ONLY_KERNEL_META(name)
#else
#define CCEC_PROBE_KERNEL_META(name) CCEC_MIX_1_1_KERNEL_META(name)
#endif

#ifndef SINGLE_CACHE_LINE
#define SINGLE_CACHE_LINE 0
#endif
#ifndef CACHELINE_OUT
#define CACHELINE_OUT 2
#endif

// ---- Sync flag IDs (from AscendC dav_3510/kernel_operator_sync_impl.h) ----
#define SYNC_AIC_FLAG     11
#define SYNC_AIV_FLAG     12
#define SYNC_AIC_AIV_FLAG 13

// ---- Cross-core SyncAll (pure CCEC) ----
// Mirrors local CANN 9.1 dav_3510/kernel_operator_sync_impl.h for the default
// AIV-only probes and the optional 1:1 MIX branch. This FFTS protocol does not
// consume a GM cache line or add atomic traffic to the workload.
#define CCEC_SYNC_AIV_ONLY_ALL 14
#define CCEC_MIX_CORES_PER_BLOCK 2

__aicore__ inline uint32_t ccec_core_kind()
{
#ifdef __DAV_VEC__
    return 1;
#else
    return 0;
#endif
}

__aicore__ inline uint32_t ccec_mix_participant_id(uint32_t block_idx)
{
    return block_idx * CCEC_MIX_CORES_PER_BLOCK + ccec_core_kind();
}

__aicore__ inline uint32_t ccec_participant_id(uint32_t block_idx)
{
#if defined(CCEC_SYNC_AIV_ONLY)
    return block_idx;
#else
    return ccec_mix_participant_id(block_idx);
#endif
}

__aicore__ inline uint16_t ccec_ffts_message(uint16_t mode, uint16_t flag_id)
{
    return 0x1u + ((mode & 0x3u) << 4) + ((flag_id & 0xfu) << 8);
}

__aicore__ inline void ccec_sync_all()
{
    __builtin_cce_pipe_barrier(PIPE_ALL);
#if defined(CCEC_SYNC_AIV_ONLY)
    __builtin_cce_ffts_cross_core_sync(
        PIPE_MTE3, ccec_ffts_message(0, CCEC_SYNC_AIV_ONLY_ALL));
    __builtin_cce_wait_flag_dev(PIPE_S, CCEC_SYNC_AIV_ONLY_ALL);
#elif defined(__DAV_VEC__)
    __builtin_cce_ffts_cross_core_sync(
        PIPE_MTE3, ccec_ffts_message(2, SYNC_AIV_FLAG));
    __builtin_cce_wait_flag_dev(PIPE_S, SYNC_AIC_AIV_FLAG);
#else
    __builtin_cce_wait_flag_dev(PIPE_S, SYNC_AIV_FLAG);
    __builtin_cce_ffts_cross_core_sync(
        PIPE_FIX, ccec_ffts_message(0, SYNC_AIC_FLAG));
    __builtin_cce_wait_flag_dev(PIPE_S, SYNC_AIC_FLAG);
    __builtin_cce_ffts_cross_core_sync(
        PIPE_MTE3, ccec_ffts_message(2, SYNC_AIC_AIV_FLAG));
#endif
}

// Keep the old call shape temporarily so existing probes are a mechanical
// protocol replacement; gx/nb/round are intentionally unused.
#define ccec_barrier(gx, nb, round) ccec_sync_all()

// ---- st_dev / ld_dev ----
#define st_dev_b32(addr, val) __builtin_cce_st_dev((val), (addr), 0)
#define st_dev_b64(addr, val) __builtin_cce_st_dev((val), (addr), 0)
#define st_dev_b16(addr, val) __builtin_cce_st_dev((val), (addr), 0)
#define st_dev_b8(addr, val)  __builtin_cce_st_dev((val), (addr), 0)
#define ld_dev_b32(addr) ((uint32_t)__builtin_cce_ld_dev((addr), 0))
#define ld_dev_b64(addr) ((uint64_t)__builtin_cce_ld_dev((addr), 0))
#define ld_dev_b16(addr) ((uint16_t)__builtin_cce_ld_dev((addr), 0))
#define ld_dev_b8(addr)  ((uint8_t)__builtin_cce_ld_dev((addr), 0))

// ---- dcci shorthand ----
#define dcci_flush(p) __builtin_cce_dcci((p), SINGLE_CACHE_LINE, CACHELINE_OUT)
#define dcci_inval(p) __builtin_cce_dcci((p), SINGLE_CACHE_LINE)

#endif
