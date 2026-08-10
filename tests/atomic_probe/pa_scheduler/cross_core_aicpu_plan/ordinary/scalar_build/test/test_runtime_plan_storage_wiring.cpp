/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "host_support.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <sys/mman.h>

namespace {

using namespace pa_scheduler;
using namespace pa_scheduler::aicpu_plan;
using namespace pa_scheduler::host;

[[noreturn]] void Fail(const char *message)
{
    std::fprintf(stderr, "[FAIL] %s\n", message);
    std::exit(1);
}

void Expect(bool condition, const char *message)
{
    if (!condition) {
        Fail(message);
    }
}

bool StorageRefIsZero(const RuntimePlanStorageRef &storage)
{
    const auto *bytes = reinterpret_cast<const uint8_t *>(&storage);
    for (size_t index = 0U; index < sizeof(storage); ++index) {
        if (bytes[index] != 0U) {
            return false;
        }
    }
    return true;
}

bool BufferHasPattern(const void *buffer, size_t bytes, uint8_t pattern)
{
    const auto *values = static_cast<const uint8_t *>(buffer);
    for (size_t index = 0U; index < bytes; ++index) {
        if (values[index] != pattern) {
            return false;
        }
    }
    return true;
}

void TestStorageRefValidation(const RuntimePlanStorageRef &valid)
{
    Expect(RuntimePlanStorageRefValid(valid), "legal storage ref was rejected");

    RuntimePlanStorageRef invalid = valid;
    invalid.cells_base += 1U;
    Expect(!RuntimePlanStorageRefValid(invalid), "misaligned base was accepted");

    invalid = valid;
    invalid.cells_bytes -= kAtomicIsolationBytes;
    Expect(!RuntimePlanStorageRefValid(invalid), "incorrect cells_bytes was accepted");

    invalid = valid;
    invalid.capacity = 0U;
    Expect(!RuntimePlanStorageRefValid(invalid), "zero capacity was accepted");

    invalid = valid;
    invalid.capacity = kMaxRuntimeTasks + 1U;
    Expect(!RuntimePlanStorageRefValid(invalid), "common ABI capacity overflow was accepted");

    invalid = valid;
    invalid.cell_bytes += kAtomicIsolationBytes;
    Expect(!RuntimePlanStorageRefValid(invalid), "incorrect cell stride was accepted");

    invalid = valid;
    invalid.abi_version += 1U;
    Expect(!RuntimePlanStorageRefValid(invalid), "incorrect ABI was accepted");

    invalid = valid;
    invalid.reserved0 = 1U;
    Expect(!RuntimePlanStorageRefValid(invalid), "nonzero reserved0 was accepted");

    for (uint32_t index = 0U; index < 12U; ++index) {
        invalid = valid;
        invalid.reserved[index] = UINT64_C(0xA5A5A5A5A5A5A5A5);
        Expect(!RuntimePlanStorageRefValid(invalid), "nonzero reserved word was accepted");
    }

    invalid = valid;
    invalid.cells_base = UINT64_MAX &
                         ~static_cast<uint64_t>(kAtomicIsolationBytes - 1U);
    Expect(!RuntimePlanStorageRefValid(invalid), "overflowing address range was accepted");
}

void TestHostWiringDoesNotTouchCells()
{
    constexpr uint8_t kCanary = 0xA5U;
    constexpr size_t kCellsBytes =
        static_cast<size_t>(kRuntimePlanConsumerCapacity) *
        sizeof(RuntimeTaskPlanCell);
    void *cells = ::operator new(
        kCellsBytes, std::align_val_t{kAtomicIsolationBytes}
    );
    std::memset(cells, kCanary, kCellsBytes);

    RuntimePlanStorageRef storage{};
    Expect(
        ConfigureRuntimePlanStorageRef(
            &storage, cells, kRuntimePlanConsumerCapacity
        ),
        "maximum consumer capacity was rejected"
    );
    Expect(storage.cells_base == reinterpret_cast<uintptr_t>(cells), "base was not recorded exactly");
    Expect(storage.cells_bytes == kCellsBytes, "byte count was not derived from capacity");
    Expect(storage.capacity == kRuntimePlanConsumerCapacity, "capacity was not recorded exactly");
    Expect(storage.cell_bytes == sizeof(RuntimeTaskPlanCell), "cell stride was not recorded exactly");
    Expect(storage.abi_version == kRuntimePlanAbiVersion, "ABI was not recorded exactly");
    Expect(RuntimePlanStorageRefValid(storage), "configured storage ref is not self-validating");
    Expect(
        BufferHasPattern(cells, kCellsBytes, kCanary),
        "Host storage wiring wrote PlanCell identity or payload"
    );
    TestStorageRefValidation(storage);

    RuntimePlanView view{};
    RuntimePlanControl control{};
    Expect(MakeRuntimePlanView(&control, storage, view), "configured storage cannot form a runtime view");
    Expect(view.control == &control, "runtime view changed the control address");
    Expect(view.cells == cells, "runtime view changed the cell address");
    Expect(view.capacity == kRuntimePlanConsumerCapacity, "runtime view changed capacity");

    // ConfigureRuntimePlanStorage 的 SchedulerState wrapper 也必须只写
    // storage-ref 尾字段。用匿名虚拟映射承载约 1 GiB 的真实类型，只会
    // 触碰 storage-ref 所在页，不会把庞大的 WorkerState 物理提交。
    void *state_mapping = mmap(
        nullptr, sizeof(SchedulerState), PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0
    );
    Expect(state_mapping != MAP_FAILED, "unable to reserve SchedulerState virtual storage");
    auto *state = ::new (state_mapping) SchedulerState;
    Expect(
        ConfigureRuntimePlanStorage(state, cells, 1U),
        "SchedulerState storage wrapper rejected a legal one-cell allocation"
    );
    Expect(state->runtime_plan_storage.capacity == 1U, "SchedulerState wrapper recorded wrong capacity");
    Expect(
        BufferHasPattern(cells, kCellsBytes, kCanary),
        "SchedulerState storage wrapper wrote PlanCell identity or payload"
    );
    state->~SchedulerState();
    Expect(munmap(state_mapping, sizeof(SchedulerState)) == 0, "unable to release SchedulerState mapping");

    // 所有失败路径都必须擦除旧 ref，留下 fail-closed 的全零描述。
    Expect(!ConfigureRuntimePlanStorageRef(&storage, nullptr, 1U), "null cell base was accepted");
    Expect(StorageRefIsZero(storage), "null-base failure retained a stale storage ref");

    Expect(!ConfigureRuntimePlanStorageRef(&storage, cells, 0U), "zero capacity was accepted by Host helper");
    Expect(StorageRefIsZero(storage), "zero-capacity failure retained a stale storage ref");

    Expect(
        !ConfigureRuntimePlanStorageRef(
            &storage, cells, kRuntimePlanConsumerCapacity + 1U
        ),
        "capacity beyond downstream task tables was accepted"
    );
    Expect(StorageRefIsZero(storage), "oversized-capacity failure retained a stale storage ref");

    auto *misaligned = static_cast<void *>(
        static_cast<uint8_t *>(cells) + 1U
    );
    Expect(!ConfigureRuntimePlanStorageRef(&storage, misaligned, 1U), "misaligned base was accepted by Host helper");
    Expect(StorageRefIsZero(storage), "misalignment failure retained a stale storage ref");

    void *overflow_base = reinterpret_cast<void *>(
        UINTPTR_MAX & ~static_cast<uintptr_t>(kAtomicIsolationBytes - 1U)
    );
    Expect(!ConfigureRuntimePlanStorageRef(&storage, overflow_base, 1U), "overflowing range was accepted by Host helper");
    Expect(StorageRefIsZero(storage), "overflow failure retained a stale storage ref");

    Expect(!ConfigureRuntimePlanStorageRef(nullptr, cells, 1U), "null output ref was accepted");
    Expect(!ConfigureRuntimePlanStorage(nullptr, cells, 1U), "null SchedulerState was accepted");

    ::operator delete(cells, std::align_val_t{kAtomicIsolationBytes});
}

}  // namespace

int main()
{
    static_assert(
        kRuntimePlanConsumerCapacity > 0U &&
            kRuntimePlanConsumerCapacity <= kMaxRuntimeTasks,
        "CPU gate must follow a valid task-indexed consumer capacity"
    );
    static_assert(
        alignof(SchedulerState) == kAtomicIsolationBytes,
        "CPU gate requires a 128B SchedulerState"
    );

    TestHostWiringDoesNotTouchCells();
    std::printf(
        "[PASS] runtime Plan storage wiring: capacity=%u cell_bytes=%zu total_bytes=%zu\n",
        kRuntimePlanConsumerCapacity, sizeof(RuntimeTaskPlanCell),
        static_cast<size_t>(kRuntimePlanConsumerCapacity) *
            sizeof(RuntimeTaskPlanCell)
    );
    return 0;
}
