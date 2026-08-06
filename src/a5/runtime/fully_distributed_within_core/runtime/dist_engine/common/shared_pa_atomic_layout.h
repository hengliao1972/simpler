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

#include <cstddef>
#include <cstdint>

// A5 Core-to-GM atomic probes show that two 64-byte-aligned words in the same
// aligned 128-byte region still share one contention queue. This type models
// that measured conflict unit without changing the generic 64-byte cache-line
// ABI or introducing a runtime mode switch.
constexpr size_t kFdwicSharedAtomicConflictBytes = 128;

struct alignas(64) FdwicSharedAtomicConflictCell {
    volatile int64_t v;
    uint8_t active_line_padding[64 - sizeof(int64_t)];
    uint8_t isolation_line[64];
};

static_assert(
    sizeof(FdwicSharedAtomicConflictCell) == kFdwicSharedAtomicConflictBytes,
    "shared PA atomic conflict cell must occupy one measured A5 conflict unit"
);
static_assert(offsetof(FdwicSharedAtomicConflictCell, v) == 0, "shared PA atomic value must start its conflict unit");

template <size_t Count>
struct alignas(64) FdwicSharedAtomicConflictTable {
    FdwicSharedAtomicConflictCell cells[Count];
};
