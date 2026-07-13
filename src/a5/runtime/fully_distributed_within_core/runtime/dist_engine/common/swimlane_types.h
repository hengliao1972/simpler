/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#pragma once

#include <cstdint>

constexpr uint32_t kFdwicSwimlaneMagic = 0x4653574Cu;  // FSWL
constexpr uint32_t kFdwicSwimlaneVersion = 1;
constexpr uint32_t kFdwicSwimlaneDefaultRecordsPerCore = 1u << 16;

enum class FdwicSwimlanePhase : int32_t {
    Kernel = 0,
    Alloc = 1,
    Build = 2,
    DrainWon = 3,
    Replay = 4,
    RingBp = 5,
    EfDrain = 6,
    Commit = 7,
};

struct FdwicSwimlaneCoreState {
    volatile uint32_t count;
    volatile uint32_t dropped;
    uint32_t pad[14];
} __attribute__((aligned(64)));

static_assert(sizeof(FdwicSwimlaneCoreState) == 64, "FdwicSwimlaneCoreState must occupy one cacheline");

struct FdwicSwimlaneHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_cores;
    uint32_t records_per_core;
    uint64_t freq_hz;
    FdwicSwimlaneCoreState cores[108];
} __attribute__((aligned(64)));

static_assert(sizeof(FdwicSwimlaneHeader) % 64 == 0, "FdwicSwimlaneHeader must be cacheline aligned");

struct FdwicSwimlaneRecord {
    uint64_t start_cycle;
    uint64_t end_cycle;
    int32_t task_id;
    int32_t func_id;
    int32_t phase;
    int32_t lane;
    int32_t block_id;
    int32_t core_idx;
    uint32_t flags;
    uint32_t aux;
} __attribute__((aligned(64)));

static_assert(sizeof(FdwicSwimlaneRecord) == 64, "FdwicSwimlaneRecord must occupy one cacheline");
