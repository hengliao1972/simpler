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

#ifndef TESTS_ATOMIC_PROBE_TRACE_WRITE_PRELOAD_SHARED_H_
#define TESTS_ATOMIC_PROBE_TRACE_WRITE_PRELOAD_SHARED_H_

#include <cstddef>
#include <cstdint>

namespace trace_write_preload {

#if defined(__CCE_AICORE__)
#define TRACE_WRITE_PRELOAD_FN __aicore__
#else
#define TRACE_WRITE_PRELOAD_FN
#endif

constexpr uint32_t kAicWorkers = 32;
constexpr uint32_t kAivWorkers = 64;
constexpr uint32_t kWorkers = kAicWorkers + kAivWorkers;
constexpr uint32_t kCacheLineBytes = 64;
constexpr uint32_t kRecordBytes = 32;
constexpr uint32_t kRecordsPerLine = kCacheLineBytes / kRecordBytes;
constexpr uint32_t kWorkerStrideBytes = 128 * 1024;
constexpr uint32_t kMaxRecordsPerWorker = kWorkerStrideBytes / kRecordBytes;
constexpr uint32_t kCapacityPasses = 8;
constexpr uint32_t kControlMagic = 0x5457504cU;  // "TWPL"
constexpr uint32_t kStatusComplete = 0x434f4d50U;  // "COMP"

enum class Experiment : uint32_t {
    TraceWrite = 0,
    CapacitySweep = 1,
};

struct alignas(32) TraceRecord {
    uint64_t start_cycle;
    uint64_t end_cycle;
    int32_t task_id;
    int32_t function_id;
    uint32_t flags;
    uint16_t phase;
    uint16_t auxiliary;
};

struct alignas(64) ProbeControl {
    uint32_t magic;
    uint32_t experiment;
    uint32_t first_worker;
    uint32_t worker_count;
    uint32_t records_per_worker;
    uint32_t preload_distance_lines;
    uint32_t preload_cadence_lines;
    uint32_t sample_id;
    uint32_t capacity_lines;
    uint32_t capacity_passes;
    uint32_t capacity_start_line;
    uint32_t reserved0;
    uint64_t seed;
    uint64_t worker_stride_bytes;
};

struct alignas(64) ProbeResult {
    uint64_t phase_begin;
    uint64_t phase_split;
    uint64_t phase_end;
    uint64_t terminal_value;
    uint32_t worker_id;
    uint32_t experiment;
    uint32_t records_or_lines;
    uint32_t preload_distance_lines;
    uint32_t preload_cadence_lines;
    uint32_t sample_id;
    uint32_t status;
    uint32_t reserved0;
};

TRACE_WRITE_PRELOAD_FN constexpr uint64_t RecordStamp(uint64_t seed, uint32_t worker, uint32_t record) {
    return seed + static_cast<uint64_t>(worker) * 0x100000001ULL + static_cast<uint64_t>(record) * 2ULL;
}

TRACE_WRITE_PRELOAD_FN constexpr uint32_t RecordFlags(uint32_t worker, uint32_t record) {
    return 0xa5000000U ^ (worker << 16U) ^ record;
}

static_assert(sizeof(TraceRecord) == kRecordBytes, "trace record must remain exactly 32 B");
static_assert(alignof(TraceRecord) == kRecordBytes, "trace record must remain 32 B aligned");
static_assert(sizeof(ProbeControl) == kCacheLineBytes, "control must occupy one cache line");
static_assert(sizeof(ProbeResult) == kCacheLineBytes, "each result must occupy one cache line");
static_assert(kMaxRecordsPerWorker % kRecordsPerLine == 0, "worker stride must contain whole cache lines");

#undef TRACE_WRITE_PRELOAD_FN

}  // namespace trace_write_preload

#endif  // TESTS_ATOMIC_PROBE_TRACE_WRITE_PRELOAD_SHARED_H_
