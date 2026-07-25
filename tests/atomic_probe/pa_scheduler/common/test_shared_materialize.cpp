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

#include <cstdint>
#include <cstdio>
#include <memory>

#define PA_DEVICE inline
#define PA_GM
#include "pa_frontend.h"

namespace {

using namespace pa_scheduler;

int g_failures = 0;

void Check(bool condition, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(stderr, "[FAIL] %s\n", message);
    ++g_failures;
}

struct MaterializeTestOps {
    static int64_t Load(volatile int64_t *address) {
        return __atomic_load_n(address, __ATOMIC_ACQUIRE);
    }

    static int64_t FetchAdd(volatile int64_t *address, int64_t value) {
        return __atomic_fetch_add(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t Exchange(volatile int64_t *address, int64_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }
};

struct HeapSnapshot {
    int64_t cursor[kSharedHeapShards];
    int64_t vend;
    uint64_t worker_heap_next;
};

struct Fixture {
    std::unique_ptr<WorkerState> worker;
    std::unique_ptr<SharedTensorMapSidecar> map;
    std::unique_ptr<TaskPayload> payload;
    TaskArgs args;
    SubmitContext context;
    TensorCreateInfo create_info;

    Fixture()
        : worker(std::make_unique<WorkerState>()),
          map(std::make_unique<SharedTensorMapSidecar>()),
          payload(std::make_unique<TaskPayload>()),
          args{},
          context{},
          create_info{} {
        constexpr uint32_t kTaskId = 1;
        const uint32_t shape[kMaxTensorDims] = {
            kPaHeads, kPaBlocksPerRequest * kPaBlockSize, 0, 0, 0
        };
        InitCreateInfo(create_info, shape, 2, DataType::Bfloat16);
        args.tags[0] = static_cast<int32_t>(TensorArgType::Output);
        args.tensors[0].pointer.create_info = &create_info;
        args.tensors[0].kind = TensorRefKind::CreateInfo;
        args.tensor_count = 1;
        args.scalar_count = 0;
        args.has_error = false;

        context.self = worker.get();
        context.payload = payload.get();
        context.task_id = static_cast<int32_t>(kTaskId);
        context.tensor_count = 0;
        context.scalar_count = 0;
        context.result.task_id = kTaskId;
        context.result.count = 0;
        context.shared_result.Reset(static_cast<int32_t>(kTaskId));
        Check(
            PrepareSharedTaskOutputs(
                context.shared_result, static_cast<int32_t>(kTaskId),
                TaskKind::Qk
            ),
            "fixture prepares the one QK output symbol"
        );
    }
};

HeapSnapshot Snapshot(const Fixture &fixture) {
    HeapSnapshot snapshot{};
    for (uint32_t shard = 0; shard < kSharedHeapShards; ++shard) {
        snapshot.cursor[shard] =
            fixture.map->shared_heap_cursor[shard].value;
    }
    snapshot.vend = fixture.map->shared_heap_vend.value;
    snapshot.worker_heap_next = fixture.worker->heap_next;
    return snapshot;
}

bool SameSnapshot(
    const Fixture &fixture, const HeapSnapshot &expected
) {
    for (uint32_t shard = 0; shard < kSharedHeapShards; ++shard) {
        if (fixture.map->shared_heap_cursor[shard].value !=
            expected.cursor[shard]) {
            return false;
        }
    }
    return fixture.map->shared_heap_vend.value == expected.vend &&
           fixture.worker->heap_next == expected.worker_heap_next;
}

void ExpectRejectedWithoutHeapChange(
    Fixture &fixture, uint64_t heap_base, uint64_t heap_size,
    const char *reject_message, const char *state_message
) {
    const HeapSnapshot before = Snapshot(fixture);
    Check(
        !MaterializeTask<MaterializeTestOps>(
            *fixture.worker, 1, fixture.args, fixture.context, *fixture.map,
            heap_base, heap_size
        ),
        reject_message
    );
    Check(SameSnapshot(fixture, before), state_message);
}

void TestValidQkMaterialize() {
    Fixture fixture;
    Check(
        MaterializeTask<MaterializeTestOps>(
            *fixture.worker, 1, fixture.args, fixture.context, *fixture.map,
            kSyntheticHeapBase, kHeapBytes
        ),
        "valid QK output materializes"
    );

    const uint64_t bytes =
        static_cast<uint64_t>(kPaHeads) *
        kPaBlocksPerRequest * kPaBlockSize * 2;
    const uint64_t shard_span =
        SharedHeapAlignDown(kHeapBytes / kSharedHeapShards);
    Check(
        fixture.map->shared_heap_cursor[1].value ==
            static_cast<int64_t>(bytes),
        "valid QK advances only shard one"
    );
    Check(
        fixture.map->shared_heap_vend.value == static_cast<int64_t>(bytes) &&
            fixture.worker->heap_next == bytes,
        "valid QK publishes the aggregate vend"
    );
    Check(
        fixture.context.output_bytes == bytes &&
            fixture.context.result.count == 1,
        "valid QK returns one descriptor and exact bytes"
    );
    Check(
        fixture.payload->tensors[0].buffer_addr ==
            kSyntheticHeapBase + shard_span &&
            fixture.payload->tensors[0].buffer_size == bytes,
        "valid QK descriptor uses the physical shard address"
    );
}

void TestCheckedShapeAndStride() {
    {
        Fixture fixture;
        fixture.create_info.ndims = kMaxTensorDims;
        for (uint32_t dimension = 0; dimension < kMaxTensorDims;
             ++dimension) {
            fixture.create_info.shapes[dimension] = UINT32_MAX;
        }
        ExpectRejectedWithoutHeapChange(
            fixture, kSyntheticHeapBase, kHeapBytes,
            "shape product overflow is rejected",
            "shape product overflow changes no heap state"
        );
    }
    {
        Fixture fixture;
        fixture.create_info.ndims = 2;
        fixture.create_info.shapes[0] = 65536;
        fixture.create_info.shapes[1] = 65536;
        ExpectRejectedWithoutHeapChange(
            fixture, kSyntheticHeapBase, kHeapBytes,
            "uint32 descriptor stride overflow is rejected",
            "stride overflow changes no heap state"
        );
    }
    {
        Fixture fixture;
        fixture.create_info.start_offset = 1;
        ExpectRejectedWithoutHeapChange(
            fixture, kSyntheticHeapBase, kHeapBytes,
            "fresh Output start offset is rejected",
            "fresh Output start offset changes no heap state"
        );
    }
}

void TestMalformedInputPreflight() {
    {
        Fixture fixture;
        fixture.args.tensor_count =
            static_cast<int32_t>(kMaxTaskTensors) + 1;
        ExpectRejectedWithoutHeapChange(
            fixture, kSyntheticHeapBase, kHeapBytes,
            "oversized tensor count is rejected",
            "oversized tensor count changes no heap state"
        );
    }
    {
        Fixture fixture;
        fixture.args.has_error = true;
        ExpectRejectedWithoutHeapChange(
            fixture, kSyntheticHeapBase, kHeapBytes,
            "builder error state is rejected",
            "builder error state changes no heap state"
        );
    }
    {
        Fixture fixture;
        fixture.context.result.count = 1;
        ExpectRejectedWithoutHeapChange(
            fixture, kSyntheticHeapBase, kHeapBytes,
            "nonempty materialize result is rejected",
            "nonempty result changes no heap state"
        );
    }
    {
        Fixture fixture;
        fixture.args.tensors[0].kind = TensorRefKind::LocalTensor;
        ExpectRejectedWithoutHeapChange(
            fixture, kSyntheticHeapBase, kHeapBytes,
            "Output with a non-CreateInfo ref is rejected",
            "wrong Output ref kind changes no heap state"
        );
    }
    {
        Fixture fixture;
        fixture.args.tensors[0].pointer.create_info = nullptr;
        ExpectRejectedWithoutHeapChange(
            fixture, kSyntheticHeapBase, kHeapBytes,
            "null Output CreateInfo is rejected",
            "null Output CreateInfo changes no heap state"
        );
    }
    {
        Fixture fixture;
        fixture.args.tags[0] =
            static_cast<int32_t>(TensorArgType::NoDependency) + 1;
        ExpectRejectedWithoutHeapChange(
            fixture, kSyntheticHeapBase, kHeapBytes,
            "invalid tensor tag is rejected",
            "invalid tensor tag changes no heap state"
        );
    }
    {
        Fixture fixture;
        fixture.args.tensor_count = 2;
        fixture.args.tags[1] =
            static_cast<int32_t>(TensorArgType::Input);
        fixture.args.tensors[1].kind = TensorRefKind::LocalTensor;
        fixture.args.tensors[1].pointer.local_tensor = nullptr;
        ExpectRejectedWithoutHeapChange(
            fixture, kSyntheticHeapBase, kHeapBytes,
            "null non-Output descriptor is rejected before reserve",
            "null non-Output descriptor changes no heap state"
        );
    }
    {
        Fixture fixture;
        fixture.args.tensor_count = 2;
        fixture.args.tags[1] =
            static_cast<int32_t>(TensorArgType::Input);
        fixture.args.tensors[1].kind =
            TensorRefKind::SharedOutputRef;
        fixture.args.tensors[1].pointer.output_ref =
            FdwicOutputRef{1, 0, 0, 0, 0, 0};
        ExpectRejectedWithoutHeapChange(
            fixture, kSyntheticHeapBase, kHeapBytes,
            "future shared symbol is rejected before reserve",
            "future shared symbol changes no heap state"
        );
    }
}

void TestHeapAddressOverflowPreflight() {
    Fixture fixture;
    ExpectRejectedWithoutHeapChange(
        fixture, UINT64_MAX - 4095, kHeapBytes,
        "heap base plus heap size overflow is rejected",
        "heap address overflow changes no heap state"
    );
}

}  // namespace

int main() {
    TestValidQkMaterialize();
    TestCheckedShapeAndStride();
    TestMalformedInputPreflight();
    TestHeapAddressOverflowPreflight();
    if (g_failures != 0) {
        std::fprintf(
            stderr,
            "shared materialize self-test failed: %d assertion(s)\n",
            g_failures
        );
        return 1;
    }
    std::printf("shared materialize self-test passed\n");
    return 0;
}
