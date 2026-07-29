/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the license.
 * -----------------------------------------------------------------------------------------------------------
 */

// Isolated physical models for the current PA shared path:
//   * writer-history 40 B destination footprint (not instruction-exact);
//   * one/three shared TensorDesc publications: 128 B / 384 B;
//   * shared TensorDesc consumption: invalidate, then ordinary GM-to-GM copy;
//   * current-PC ICache preload at a same-function target versus at a distant
//     caller.  The build script verifies the final linked layout.
//
// Preload is never used as a publication, coherency, or ordering primitive.
// Every DCCI/DSB required by the modeled path remains present.

#include "../shared_preload_model_shared.h"
#include "ccec_utils.h"

#if defined(SHARED_PRELOAD_MODEL_BUILD_AIC)
PTO_SYNCALL_MIX_AIC_KERNEL_META(shared_preload_model_0_mix_aic, 1, 2);
#elif defined(SHARED_PRELOAD_MODEL_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(shared_preload_model_0_mix_aiv, 1, 2);
#else
#error "Compile with SHARED_PRELOAD_MODEL_BUILD_AIC or SHARED_PRELOAD_MODEL_BUILD_AIV"
#endif

namespace {

__aicore__ __attribute__((always_inline)) inline uint64_t ReadOrderedSysCount() {
    uint64_t value = 0;
    asm volatile("MOV %0, SYS_CNT\n" : "=&l"(value) : : "memory");
    return value;
}

// Tied operands make the value-producing work stay between the two SYS_CNT
// boundaries.  A plain memory-clobber bracket is insufficient for a pure
// scalar helper because the compiler may legally hoist that helper.
__aicore__ __attribute__((always_inline)) inline uint64_t CycleBeforeValue(
    uint64_t &value
) {
    uint64_t cycle = 0;
    asm volatile(
        "MOV %1, SYS_CNT\n"
        "MOV %0, %0\n"
        : "+l"(value), "=&l"(cycle)
        :
        : "memory"
    );
    return cycle;
}

__aicore__ __attribute__((always_inline)) inline uint64_t CycleAfterValue(
    uint64_t &value
) {
    uint64_t cycle = 0;
    asm volatile(
        "MOV %0, %0\n"
        "MOV %1, SYS_CNT\n"
        : "+l"(value), "=&l"(cycle)
        :
        : "memory"
    );
    return cycle;
}

__aicore__ __attribute__((always_inline)) inline uint64_t OpaqueIdentity(
    uint64_t value
) {
    asm volatile("MOV %0, %0\n" : "+l"(value) : : "memory");
    return value;
}

__aicore__ __attribute__((always_inline)) inline void InvalidateLines(
    __gm__ uint8_t *base, uint32_t lines
) {
    for (uint32_t line = 0; line < lines; ++line) {
        dcci(
            base + static_cast<uint64_t>(line) *
                       shared_preload_model::kCacheLineBytes,
            SINGLE_CACHE_LINE
        );
    }
    dsb(DSB_ALL);
    asm volatile("" ::: "memory");
}

__aicore__ __attribute__((always_inline)) inline void FlushLines(
    __gm__ uint8_t *base, uint32_t lines
) {
    asm volatile("" ::: "memory");
    for (uint32_t line = 0; line < lines; ++line) {
        dcci(
            base + static_cast<uint64_t>(line) *
                       shared_preload_model::kCacheLineBytes,
            SINGLE_CACHE_LINE, CACHELINE_OUT
        );
    }
    dsb(DSB_ALL);
    asm volatile("" ::: "memory");
}

__aicore__ __attribute__((always_inline)) inline void PreloadLines(
    __gm__ uint8_t *base, uint32_t lines
) {
    __gm__ uint64_t *preload_base =
        reinterpret_cast<__gm__ uint64_t *>(base);
    for (uint32_t line = 0; line < lines; ++line) {
        dc_preload(
            preload_base,
            static_cast<int64_t>(
                static_cast<uint64_t>(line) *
                shared_preload_model::kCacheLineBytes
            )
        );
    }
}

__aicore__ __attribute__((noinline)) void CopyBytes(
    volatile __gm__ uint8_t *destination,
    const volatile __gm__ uint8_t *source, uint32_t bytes
) {
    for (uint32_t byte = 0; byte < bytes; ++byte) {
        destination[byte] = source[byte];
    }
}

__aicore__ __attribute__((noinline)) uint64_t ChecksumBytes(
    const volatile __gm__ uint8_t *source, uint32_t bytes
) {
    uint64_t checksum = 0xcbf29ce484222325ULL;
    for (uint32_t byte = 0; byte < bytes; ++byte) {
        checksum ^= static_cast<uint64_t>(source[byte]);
        checksum *= 0x100000001b3ULL;
    }
    return checksum;
}

__aicore__ __attribute__((noinline)) void WarmDestination(
    volatile __gm__ uint8_t *destination, uint32_t bytes
) {
    for (uint32_t byte = 0; byte < bytes; ++byte) {
        destination[byte] =
            static_cast<uint8_t>(0xa5U ^ byte ^ (byte >> 2U));
    }
    asm volatile("" ::: "memory");
}

__aicore__ __attribute__((noinline, used)) uint64_t
SharedPreloadModelGap(uint64_t seed, uint32_t gap_rounds) {
    return cache_preload::GapOracle(seed, gap_rounds);
}

__aicore__ __attribute__((always_inline)) inline void PublishResult(
    __gm__ shared_preload_model::ProbeResult *result,
    uint64_t setup_ticks, uint64_t issue_ticks, uint64_t gap_ticks,
    uint64_t access_ticks, uint64_t publish_ticks, uint64_t total_ticks,
    uint64_t preparation_checksum, uint64_t result_checksum,
    uint64_t icache_immediate_status, uint64_t icache_final_status,
    uint32_t worker, uint32_t experiment, uint32_t mode,
    uint32_t active_bytes, uint32_t gap_rounds, uint32_t sample_id
) {
    st_dev_b64(&result->setup_ticks, setup_ticks);
    st_dev_b64(&result->issue_ticks, issue_ticks);
    st_dev_b64(&result->gap_ticks, gap_ticks);
    st_dev_b64(&result->access_ticks, access_ticks);
    st_dev_b64(&result->publish_ticks, publish_ticks);
    st_dev_b64(&result->total_ticks, total_ticks);
    st_dev_b64(&result->preparation_checksum, preparation_checksum);
    st_dev_b64(&result->result_checksum, result_checksum);
    st_dev_b64(
        &result->icache_immediate_status, icache_immediate_status
    );
    st_dev_b64(&result->icache_final_status, icache_final_status);
    st_dev_b32(&result->worker_id, worker);
    st_dev_b32(&result->experiment, experiment);
    st_dev_b32(&result->mode, mode);
    st_dev_b32(&result->active_bytes, active_bytes);
    st_dev_b32(&result->gap_rounds, gap_rounds);
    st_dev_b32(&result->sample_id, sample_id);
    st_dev_b32(&result->status, shared_preload_model::kStatusComplete);
}

__aicore__ inline void RunDCache(
    __gm__ shared_preload_model::WorkerData *worker_data,
    __gm__ shared_preload_model::ProbeResult *result,
    const shared_preload_model::ProbeControl &control, uint32_t worker
) {
    const uint32_t bytes = control.active_bytes;
    const uint32_t lines =
        shared_preload_model::CacheLinesForBytes(bytes);
    volatile __gm__ uint8_t *source = worker_data->source;
    volatile __gm__ uint8_t *destination = worker_data->destination;
    const bool use_preload =
        control.mode ==
        static_cast<uint32_t>(
            shared_preload_model::Mode::DCachePreload
        );
    const bool publish =
        control.experiment ==
        static_cast<uint32_t>(
            shared_preload_model::Experiment::Publish
        );

    // The source TensorDesc/history payload is already materialized in the PA
    // path.  Warm it before the measured writer model.  The consume model then
    // performs its mandatory source invalidation inside the measured window.
    const uint64_t preparation_checksum =
        ChecksumBytes(source, bytes);
    if (publish) {
        InvalidateLines(
            reinterpret_cast<__gm__ uint8_t *>(worker_data->destination),
            lines
        );
    } else {
        WarmDestination(destination, bytes);
    }

    const uint64_t total_begin = ReadOrderedSysCount();
    uint64_t setup_ticks = 0;
    if (!publish) {
        const uint64_t setup_begin = ReadOrderedSysCount();
        InvalidateLines(
            reinterpret_cast<__gm__ uint8_t *>(worker_data->source),
            lines
        );
        const uint64_t setup_end = ReadOrderedSysCount();
        setup_ticks = setup_end - setup_begin;
    }

    const uint64_t issue_begin = ReadOrderedSysCount();
    if (use_preload) {
        PreloadLines(
            publish
                ? reinterpret_cast<__gm__ uint8_t *>(
                      worker_data->destination
                  )
                : reinterpret_cast<__gm__ uint8_t *>(
                      worker_data->source
                  ),
            lines
        );
    }
    const uint64_t issue_end = ReadOrderedSysCount();

    uint64_t gap_input =
        control.seed ^ static_cast<uint64_t>(worker);
    const uint64_t gap_begin = CycleBeforeValue(gap_input);
    uint64_t gap_checksum = SharedPreloadModelGap(
        gap_input, control.gap_rounds
    );
    const uint64_t gap_end = CycleAfterValue(gap_checksum);

    // The zero address delta makes the copy addresses data-dependent on the
    // noinline independent gap without changing the runtime address.
    const uint64_t opaque_gap = OpaqueIdentity(gap_checksum);
    const uint64_t address_delta = opaque_gap - gap_checksum;
    volatile __gm__ uint8_t *dependent_destination =
        reinterpret_cast<volatile __gm__ uint8_t *>(
            reinterpret_cast<uint64_t>(destination) + address_delta
        );
    const volatile __gm__ uint8_t *dependent_source =
        reinterpret_cast<const volatile __gm__ uint8_t *>(
            reinterpret_cast<uint64_t>(source) + address_delta
        );

    const uint64_t access_begin = ReadOrderedSysCount();
    CopyBytes(
        dependent_destination, dependent_source, bytes
    );
    const uint64_t access_end = ReadOrderedSysCount();

    uint64_t total_end = access_end;
    uint64_t publish_ticks = 0;
    if (publish) {
        const uint64_t publish_begin = ReadOrderedSysCount();
        FlushLines(
            reinterpret_cast<__gm__ uint8_t *>(
                worker_data->destination
            ),
            lines
        );
        total_end = ReadOrderedSysCount();
        publish_ticks = total_end - publish_begin;
    } else {
        // Result validation needs the copied payload in GM.  This cleanup is
        // deliberately after total_end and is not part of the consume model.
        FlushLines(
            reinterpret_cast<__gm__ uint8_t *>(
                worker_data->destination
            ),
            lines
        );
    }

    const uint64_t actual_checksum =
        ChecksumBytes(destination, bytes);
    const uint64_t rotated_actual_checksum =
        (actual_checksum << 1U) | (actual_checksum >> 63U);
    PublishResult(
        result, setup_ticks, issue_end - issue_begin,
        gap_end - gap_begin, access_end - access_begin, publish_ticks,
        total_end - total_begin, preparation_checksum,
        rotated_actual_checksum ^ gap_checksum,
        0, 0, worker, control.experiment, control.mode, bytes,
        control.gap_rounds, control.sample_id
    );
}

#if defined(SHARED_PRELOAD_MODEL_BUILD_AIV)

#define SHARED_PRELOAD_NOPS_1() asm volatile("nop");
#define SHARED_PRELOAD_NOPS_2() SHARED_PRELOAD_NOPS_1() SHARED_PRELOAD_NOPS_1()
#define SHARED_PRELOAD_NOPS_4() SHARED_PRELOAD_NOPS_2() SHARED_PRELOAD_NOPS_2()
#define SHARED_PRELOAD_NOPS_8() SHARED_PRELOAD_NOPS_4() SHARED_PRELOAD_NOPS_4()
#define SHARED_PRELOAD_NOPS_16() SHARED_PRELOAD_NOPS_8() SHARED_PRELOAD_NOPS_8()
#define SHARED_PRELOAD_NOPS_32() SHARED_PRELOAD_NOPS_16() SHARED_PRELOAD_NOPS_16()
#define SHARED_PRELOAD_NOPS_64() SHARED_PRELOAD_NOPS_32() SHARED_PRELOAD_NOPS_32()
#define SHARED_PRELOAD_NOPS_128() SHARED_PRELOAD_NOPS_64() SHARED_PRELOAD_NOPS_64()
#define SHARED_PRELOAD_NOPS_256() SHARED_PRELOAD_NOPS_128() SHARED_PRELOAD_NOPS_128()
#define SHARED_PRELOAD_NOPS_512() SHARED_PRELOAD_NOPS_256() SHARED_PRELOAD_NOPS_256()
#define SHARED_PRELOAD_NOPS_1024() SHARED_PRELOAD_NOPS_512() SHARED_PRELOAD_NOPS_512()
#define SHARED_PRELOAD_NOPS_2048() SHARED_PRELOAD_NOPS_1024() SHARED_PRELOAD_NOPS_1024()
#define SHARED_PRELOAD_NOPS_4096() SHARED_PRELOAD_NOPS_2048() SHARED_PRELOAD_NOPS_2048()
#define SHARED_PRELOAD_NOPS_8192() SHARED_PRELOAD_NOPS_4096() SHARED_PRELOAD_NOPS_4096()

extern "C" __aicore__ __attribute__((noinline, used, aligned(128)))
uint64_t shared_preload_model_icache_target(
    uint32_t mode, uint64_t seed, uint32_t gap_rounds,
    uint64_t wrapper_begin, uint64_t caller_issue_ticks,
    uint64_t caller_immediate_status, uint64_t preparation_checksum,
    uint32_t worker, uint32_t sample_id,
    __gm__ shared_preload_model::ProbeResult *result
) {
    uint64_t issue_ticks = caller_issue_ticks;
    uint64_t immediate_status = caller_immediate_status;
    const bool target_preload =
        mode ==
        static_cast<uint32_t>(
            shared_preload_model::Mode::ICacheTargetPreload
        );
    if (target_preload) {
        const uint64_t issue_begin = ReadOrderedSysCount();
        icache_preload(
            static_cast<int64_t>(
                shared_preload_model::kICachePreloadUnits
            )
        );
        const uint64_t issue_end = ReadOrderedSysCount();
        issue_ticks = issue_end - issue_begin;
        immediate_status =
            static_cast<uint64_t>(get_icache_prl_st());
    }

    uint64_t gap_input = seed ^ static_cast<uint64_t>(worker);
    const uint64_t gap_begin = CycleBeforeValue(gap_input);
    uint64_t gap_checksum =
        SharedPreloadModelGap(gap_input, gap_rounds);
    const uint64_t gap_end = CycleAfterValue(gap_checksum);

    uint64_t final_status = 0;
    if (mode != static_cast<uint32_t>(
                    shared_preload_model::Mode::ICacheBaseline
                )) {
        final_status = static_cast<uint64_t>(get_icache_prl_st());
    }

    uint64_t value = seed ^ gap_checksum;
    const uint64_t access_begin = ReadOrderedSysCount();
    SHARED_PRELOAD_NOPS_1024()
    value = cache_preload::ICacheTargetOracle(value);
    const uint64_t access_end = ReadOrderedSysCount();

    PublishResult(
        result, 0, issue_ticks, gap_end - gap_begin,
        access_end - access_begin, 0, access_end - wrapper_begin,
        preparation_checksum, value, immediate_status, final_status,
        worker,
        static_cast<uint32_t>(
            shared_preload_model::Experiment::ICachePlacement
        ),
        mode, 0, gap_rounds, sample_id
    );
    return value;
}

// The evictor exceeds the documented 16 KiB AIV ICache; the final ELF check
// enforces the 32 KiB minimum chosen by this cold-state model.
extern "C" __aicore__ __attribute__((noinline, used, aligned(128)))
uint64_t shared_preload_model_icache_evictor(uint64_t seed) {
    SHARED_PRELOAD_NOPS_8192()
    return cache_preload::ICacheEvictorOracle(seed);
}

// The linker check requires the target to lie outside the caller's forward
// current-PC preload window.  Thus caller-preload measures a real call-boundary
// mismatch, while target-preload uses the same physical target NOP region as
// baseline.
extern "C" __aicore__ __attribute__((noinline, used, aligned(128)))
void shared_preload_model_icache_caller(
    uint32_t mode, uint64_t seed, uint32_t gap_rounds,
    uint64_t preparation_checksum, uint32_t worker, uint32_t sample_id,
    __gm__ shared_preload_model::ProbeResult *result
) {
    const uint64_t wrapper_begin = ReadOrderedSysCount();
    uint64_t caller_issue_ticks = 0;
    uint64_t caller_immediate_status = 0;
    if (mode ==
        static_cast<uint32_t>(
            shared_preload_model::Mode::ICacheCallerPreload
        )) {
        const uint64_t issue_begin = ReadOrderedSysCount();
        icache_preload(
            static_cast<int64_t>(
                shared_preload_model::kICachePreloadUnits
            )
        );
        const uint64_t issue_end = ReadOrderedSysCount();
        caller_issue_ticks = issue_end - issue_begin;
        caller_immediate_status =
            static_cast<uint64_t>(get_icache_prl_st());
    }
    (void)shared_preload_model_icache_target(
        mode, seed, gap_rounds, wrapper_begin, caller_issue_ticks,
        caller_immediate_status, preparation_checksum, worker,
        sample_id, result
    );
}

__aicore__ inline void RunICache(
    __gm__ shared_preload_model::ProbeResult *result,
    const shared_preload_model::ProbeControl &control, uint32_t worker
) {
    const uint64_t preparation_checksum =
        shared_preload_model_icache_evictor(control.seed);
    shared_preload_model_icache_caller(
        control.mode, control.seed, control.gap_rounds,
        preparation_checksum, worker, control.sample_id, result
    );
}

#undef SHARED_PRELOAD_NOPS_8192
#undef SHARED_PRELOAD_NOPS_4096
#undef SHARED_PRELOAD_NOPS_2048
#undef SHARED_PRELOAD_NOPS_1024
#undef SHARED_PRELOAD_NOPS_512
#undef SHARED_PRELOAD_NOPS_256
#undef SHARED_PRELOAD_NOPS_128
#undef SHARED_PRELOAD_NOPS_64
#undef SHARED_PRELOAD_NOPS_32
#undef SHARED_PRELOAD_NOPS_16
#undef SHARED_PRELOAD_NOPS_8
#undef SHARED_PRELOAD_NOPS_4
#undef SHARED_PRELOAD_NOPS_2
#undef SHARED_PRELOAD_NOPS_1

#endif  // SHARED_PRELOAD_MODEL_BUILD_AIV

__aicore__ inline void RunParticipant(
    __gm__ shared_preload_model::ProbeControl *control_pointer,
    __gm__ shared_preload_model::ProbeResult *results,
    __gm__ shared_preload_model::WorkerData *data, uint32_t worker
) {
    dcci(control_pointer, SINGLE_CACHE_LINE);
    dsb(DSB_ALL);
    asm volatile("" ::: "memory");

    shared_preload_model::ProbeControl control{};
    control.magic = control_pointer->magic;
    control.experiment = control_pointer->experiment;
    control.mode = control_pointer->mode;
    control.active_bytes = control_pointer->active_bytes;
    control.gap_rounds = control_pointer->gap_rounds;
    control.sample_id = control_pointer->sample_id;
    control.first_worker = control_pointer->first_worker;
    control.worker_count = control_pointer->worker_count;
    control.seed = control_pointer->seed;

    if (control.magic != shared_preload_model::kControlMagic ||
        worker < control.first_worker ||
        worker >= control.first_worker + control.worker_count) {
        return;
    }
    __gm__ shared_preload_model::ProbeResult *result =
        &results[worker];

    if (control.experiment ==
        static_cast<uint32_t>(
            shared_preload_model::Experiment::ICachePlacement
        )) {
#if defined(SHARED_PRELOAD_MODEL_BUILD_AIV)
        if (control.active_bytes == 0U &&
            control.mode >= static_cast<uint32_t>(
                shared_preload_model::Mode::ICacheBaseline
            ) &&
            control.mode <= static_cast<uint32_t>(
                shared_preload_model::Mode::ICacheTargetPreload
            )) {
            RunICache(result, control, worker);
        }
#endif
        return;
    }

    const bool valid_bytes =
        control.active_bytes ==
            shared_preload_model::kWriterHistoryBytes ||
        control.active_bytes ==
            shared_preload_model::kOneDescriptorBytes ||
        control.active_bytes ==
            shared_preload_model::kThreeDescriptorBytes;
    const bool valid_experiment =
        control.experiment ==
            static_cast<uint32_t>(
                shared_preload_model::Experiment::Publish
            ) ||
        control.experiment ==
            static_cast<uint32_t>(
                shared_preload_model::Experiment::Consume
            );
    const bool valid_mode =
        control.mode ==
            static_cast<uint32_t>(
                shared_preload_model::Mode::DCacheBaseline
            ) ||
        control.mode ==
            static_cast<uint32_t>(
                shared_preload_model::Mode::DCachePreload
            );
    if (!valid_bytes || !valid_experiment || !valid_mode) {
        return;
    }
    RunDCache(&data[worker], result, control, worker);
}

}  // namespace

#if defined(SHARED_PRELOAD_MODEL_BUILD_AIC)
extern "C" __global__ __aicore__ void shared_preload_model_0_mix_aic(
    __gm__ shared_preload_model::ProbeControl *control,
    __gm__ shared_preload_model::ProbeResult *results,
    __gm__ shared_preload_model::WorkerData *data
) {
    RunParticipant(
        control, results, data,
        static_cast<uint32_t>(get_block_idx())
    );
}
#elif defined(SHARED_PRELOAD_MODEL_BUILD_AIV)
extern "C" __global__ __aicore__ void shared_preload_model_0_mix_aiv(
    __gm__ shared_preload_model::ProbeControl *control,
    __gm__ shared_preload_model::ProbeResult *results,
    __gm__ shared_preload_model::WorkerData *data
) {
    const uint32_t vector_id = static_cast<uint32_t>(
        get_block_idx() * get_subblockdim() + get_subblockid()
    );
    RunParticipant(
        control, results, data,
        shared_preload_model::kAicWorkers + vector_id
    );
}
#endif
