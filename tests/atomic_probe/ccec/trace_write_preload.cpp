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

// This probe deliberately models the physical PA swimlane write path rather
// than a single isolated store:
//   * one 32 B TraceRecord contains the same seven scalar fields;
//   * two consecutive records fill one 64 B DCache line;
//   * each worker owns an aligned, disjoint sequential output range;
//   * the publication window uses the same per-line CACHELINE_OUT loop and one
//     trailing DSB as CcecOps::FlushRegion.

#include "../trace_write_preload_shared.h"
#include "ccec_utils.h"

#if defined(TRACE_WRITE_PRELOAD_BUILD_AIC)
PTO_SYNCALL_MIX_AIC_KERNEL_META(trace_write_preload_0_mix_aic, 1, 2);
#elif defined(TRACE_WRITE_PRELOAD_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(trace_write_preload_0_mix_aiv, 1, 2);
#else
#error "Compile with TRACE_WRITE_PRELOAD_BUILD_AIC or TRACE_WRITE_PRELOAD_BUILD_AIV"
#endif

namespace {

__aicore__ __attribute__((always_inline)) inline uint64_t ReadOrderedSysCount() {
    uint64_t value = 0;
    asm volatile("MOV %0, SYS_CNT\n" : "=&l"(value) : : "memory");
    return value;
}

__aicore__ __attribute__((always_inline)) inline void InvalidateLines(
    __gm__ uint8_t *base, uint32_t lines
) {
    for (uint32_t line = 0; line < lines; ++line) {
        dcci(base + static_cast<uint64_t>(line) * trace_write_preload::kCacheLineBytes, SINGLE_CACHE_LINE);
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
            base + static_cast<uint64_t>(line) * trace_write_preload::kCacheLineBytes,
            SINGLE_CACHE_LINE, CACHELINE_OUT
        );
    }
    dsb(DSB_ALL);
    asm volatile("" ::: "memory");
}

__aicore__ __attribute__((always_inline)) inline void WriteRecord(
    volatile __gm__ trace_write_preload::TraceRecord *record,
    uint64_t seed, uint32_t worker, uint32_t record_id
) {
    const uint64_t stamp = trace_write_preload::RecordStamp(seed, worker, record_id);
    record->start_cycle = stamp;
    record->end_cycle = stamp + 1U;
    record->task_id = static_cast<int32_t>(record_id);
    record->function_id = static_cast<int32_t>(worker);
    record->flags = trace_write_preload::RecordFlags(worker, record_id);
    record->phase = static_cast<uint16_t>(record_id & 0xffffU);
    record->auxiliary = static_cast<uint16_t>(worker);
}

template <bool Preload>
__aicore__ __attribute__((noinline)) uint64_t WriteTraceRecords(
    __gm__ uint8_t *worker_base, uint32_t records, uint32_t distance_lines,
    uint32_t cadence_lines, uint64_t seed, uint32_t worker
) {
    const uint32_t lines = records / trace_write_preload::kRecordsPerLine;
    volatile __gm__ trace_write_preload::TraceRecord *output =
        reinterpret_cast<volatile __gm__ trace_write_preload::TraceRecord *>(worker_base);
    __gm__ uint64_t *preload_base = reinterpret_cast<__gm__ uint64_t *>(worker_base);

    for (uint32_t line = 0; line < lines; ++line) {
        if constexpr (Preload) {
            const uint32_t future_line = line + distance_lines;
            if ((line % cadence_lines) == 0U && future_line < lines) {
                dc_preload(
                    preload_base,
                    static_cast<int64_t>(
                        static_cast<uint64_t>(future_line) * trace_write_preload::kCacheLineBytes
                    )
                );
            }
        }
        const uint32_t first_record = line * trace_write_preload::kRecordsPerLine;
        WriteRecord(&output[first_record], seed, worker, first_record);
        WriteRecord(&output[first_record + 1U], seed, worker, first_record + 1U);
    }
    return trace_write_preload::RecordStamp(seed, worker, records - 1U);
}

__aicore__ __attribute__((always_inline)) inline void PublishResult(
    __gm__ trace_write_preload::ProbeResult *result,
    uint64_t begin, uint64_t split, uint64_t end, uint64_t terminal,
    uint32_t worker, uint32_t experiment, uint32_t records_or_lines,
    uint32_t distance, uint32_t cadence, uint32_t sample
) {
    st_dev_b64(&result->phase_begin, begin);
    st_dev_b64(&result->phase_split, split);
    st_dev_b64(&result->phase_end, end);
    st_dev_b64(&result->terminal_value, terminal);
    st_dev_b32(&result->worker_id, worker);
    st_dev_b32(&result->experiment, experiment);
    st_dev_b32(&result->records_or_lines, records_or_lines);
    st_dev_b32(&result->preload_distance_lines, distance);
    st_dev_b32(&result->preload_cadence_lines, cadence);
    st_dev_b32(&result->sample_id, sample);
    st_dev_b32(&result->status, trace_write_preload::kStatusComplete);
}

__aicore__ inline void RunTraceWrite(
    __gm__ uint8_t *worker_base, __gm__ trace_write_preload::ProbeResult *result,
    const trace_write_preload::ProbeControl &control, uint32_t worker
) {
    const uint32_t records = control.records_per_worker;
    const uint32_t lines = records / trace_write_preload::kRecordsPerLine;

    // Cold setup is deliberately outside both measured windows.
    InvalidateLines(worker_base, lines);

    const uint64_t begin = ReadOrderedSysCount();
    uint64_t terminal = 0;
    if (control.preload_distance_lines == 0U) {
        terminal = WriteTraceRecords<false>(
            worker_base, records, 0U, 1U, control.seed, worker
        );
    } else {
        terminal = WriteTraceRecords<true>(
            worker_base, records, control.preload_distance_lines,
            control.preload_cadence_lines, control.seed, worker
        );
    }
    const uint64_t split = ReadOrderedSysCount();
    FlushLines(worker_base, lines);
    const uint64_t end = ReadOrderedSysCount();

    PublishResult(
        result, begin, split, end, terminal, worker,
        static_cast<uint32_t>(trace_write_preload::Experiment::TraceWrite),
        records, control.preload_distance_lines, control.preload_cadence_lines,
        control.sample_id
    );
}

__aicore__ __attribute__((noinline)) uint32_t TraverseCapacityCycle(
    volatile __gm__ uint8_t *worker_base, uint32_t start_line,
    uint32_t lines, uint32_t passes
) {
    uint32_t current = start_line;
    const uint32_t accesses = lines * passes;
    for (uint32_t access = 0; access < accesses; ++access) {
        const volatile __gm__ uint32_t *next =
            reinterpret_cast<const volatile __gm__ uint32_t *>(
                worker_base + static_cast<uint64_t>(current) * trace_write_preload::kCacheLineBytes
            );
        current = *next;
    }
    return current;
}

__aicore__ inline void RunCapacitySweep(
    __gm__ uint8_t *worker_base, __gm__ trace_write_preload::ProbeResult *result,
    const trace_write_preload::ProbeControl &control, uint32_t worker
) {
    InvalidateLines(worker_base, control.capacity_lines);

    // One dependent cold traversal fills the working set. The immediately
    // following repeated traversals expose the effective resident-set knee
    // without allowing a sequential hardware prefetcher to hide misses.
    const uint64_t begin = ReadOrderedSysCount();
    uint32_t terminal = TraverseCapacityCycle(
        worker_base, control.capacity_start_line, control.capacity_lines, 1U
    );
    const uint64_t split = ReadOrderedSysCount();
    terminal ^= TraverseCapacityCycle(
        worker_base, control.capacity_start_line, control.capacity_lines,
        control.capacity_passes
    );
    const uint64_t end = ReadOrderedSysCount();

    PublishResult(
        result, begin, split, end, terminal, worker,
        static_cast<uint32_t>(trace_write_preload::Experiment::CapacitySweep),
        control.capacity_lines, 0U, 0U, control.sample_id
    );
}

__aicore__ inline void RunParticipant(
    __gm__ trace_write_preload::ProbeControl *control_pointer,
    __gm__ trace_write_preload::ProbeResult *results,
    __gm__ uint8_t *records, uint32_t worker
) {
    dcci(control_pointer, SINGLE_CACHE_LINE);
    dsb(DSB_ALL);
    asm volatile("" ::: "memory");
    trace_write_preload::ProbeControl control{};
    control.magic = control_pointer->magic;
    control.experiment = control_pointer->experiment;
    control.first_worker = control_pointer->first_worker;
    control.worker_count = control_pointer->worker_count;
    control.records_per_worker = control_pointer->records_per_worker;
    control.preload_distance_lines = control_pointer->preload_distance_lines;
    control.preload_cadence_lines = control_pointer->preload_cadence_lines;
    control.sample_id = control_pointer->sample_id;
    control.capacity_lines = control_pointer->capacity_lines;
    control.capacity_passes = control_pointer->capacity_passes;
    control.capacity_start_line = control_pointer->capacity_start_line;
    control.seed = control_pointer->seed;
    control.worker_stride_bytes = control_pointer->worker_stride_bytes;

    if (control.magic != trace_write_preload::kControlMagic ||
        control.worker_stride_bytes != trace_write_preload::kWorkerStrideBytes ||
        worker < control.first_worker ||
        worker >= control.first_worker + control.worker_count) {
        return;
    }

    __gm__ uint8_t *worker_base =
        records + static_cast<uint64_t>(worker) * control.worker_stride_bytes;
    __gm__ trace_write_preload::ProbeResult *result = &results[worker];
    if (control.experiment ==
        static_cast<uint32_t>(trace_write_preload::Experiment::TraceWrite)) {
        if (control.records_per_worker == 0U ||
            control.records_per_worker > trace_write_preload::kMaxRecordsPerWorker ||
            (control.records_per_worker % trace_write_preload::kRecordsPerLine) != 0U ||
            (control.preload_distance_lines != 0U &&
             control.preload_cadence_lines == 0U)) {
            return;
        }
        RunTraceWrite(worker_base, result, control, worker);
        return;
    }
    if (control.experiment ==
        static_cast<uint32_t>(trace_write_preload::Experiment::CapacitySweep)) {
        if (control.capacity_lines == 0U ||
            control.capacity_lines >
                trace_write_preload::kWorkerStrideBytes /
                    trace_write_preload::kCacheLineBytes ||
            control.capacity_passes == 0U ||
            control.capacity_start_line >= control.capacity_lines) {
            return;
        }
        RunCapacitySweep(worker_base, result, control, worker);
    }
}

}  // namespace

#if defined(TRACE_WRITE_PRELOAD_BUILD_AIC)
extern "C" __global__ __aicore__ void trace_write_preload_0_mix_aic(
    __gm__ trace_write_preload::ProbeControl *control,
    __gm__ trace_write_preload::ProbeResult *results,
    __gm__ uint8_t *records
) {
    RunParticipant(
        control, results, records, static_cast<uint32_t>(get_block_idx())
    );
}
#elif defined(TRACE_WRITE_PRELOAD_BUILD_AIV)
extern "C" __global__ __aicore__ void trace_write_preload_0_mix_aiv(
    __gm__ trace_write_preload::ProbeControl *control,
    __gm__ trace_write_preload::ProbeResult *results,
    __gm__ uint8_t *records
) {
    const uint32_t vector_id = static_cast<uint32_t>(
        get_block_idx() * get_subblockdim() + get_subblockid()
    );
    RunParticipant(
        control, results, records, trace_write_preload::kAicWorkers + vector_id
    );
}
#endif
