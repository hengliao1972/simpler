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

#include "pa_scheduler_core.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>

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

// 该 Ops 只验证公共 symbol helper 的原子状态机和 descriptor 搬运。
// fence 不模拟 A5 DCache；设备缓存可见性仍必须由 CCEC 上板门禁证明。
struct SymbolTestOps {
    static volatile int64_t *wait_address;
    static std::atomic<uint64_t> wait_loads;
    static std::atomic<uint64_t> now_calls;

    static int32_t Load(volatile int32_t *address) {
        return __atomic_fetch_add(address, int32_t{0}, __ATOMIC_ACQUIRE);
    }

    static int64_t Load(volatile int64_t *address) {
        if (address == wait_address) {
            wait_loads.fetch_add(1, std::memory_order_release);
        }
        return __atomic_fetch_add(address, int64_t{0}, __ATOMIC_ACQUIRE);
    }

    static int32_t Exchange(volatile int32_t *address, int32_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t Exchange(volatile int64_t *address, int64_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t FetchMax(
        volatile int64_t *address, int64_t value, uint64_t &retries
    ) {
        int64_t current = __atomic_load_n(address, __ATOMIC_ACQUIRE);
        retries = 0;
        while (current < value) {
            if (__atomic_compare_exchange_n(
                    address, &current, value, true, __ATOMIC_ACQ_REL,
                    __ATOMIC_ACQUIRE
                )) {
                break;
            }
            ++retries;
        }
        return current;
    }

    static void StoreBarrier() {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    static void FlushRegion(void *, uint64_t) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    static void InvalidateRegion(const void *, uint64_t) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    static uint64_t Now() {
        now_calls.fetch_add(1, std::memory_order_relaxed);
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            ).count()
        );
    }

    static void SpinHint() {
        std::this_thread::yield();
    }
};

volatile int64_t *SymbolTestOps::wait_address = nullptr;
std::atomic<uint64_t> SymbolTestOps::wait_loads{0};
std::atomic<uint64_t> SymbolTestOps::now_calls{0};

// 把第二次读钟直接推进到 watchdog 期限之后，避免用真实 2 秒等待测试
// timeout 终止语义。
struct ExpiredWaitOps : SymbolTestOps {
    static std::atomic<uint64_t> calls;

    static uint64_t Now() {
        const uint64_t call = calls.fetch_add(1, std::memory_order_relaxed);
        return call == 0 ? 0 : kWatchdogTicks + 1;
    }

    static void SpinHint() {}
};

std::atomic<uint64_t> ExpiredWaitOps::calls{0};

// 只在定向测试中模拟“预检后、atomic 执行前”出现的协议异常，覆盖正常
// exact-turn 不会命中的冷回滚分支。
struct PublicationFaultOps : SymbolTestOps {
    using SymbolTestOps::Exchange;

    static volatile int64_t *fetch_race_address;
    static volatile int64_t *exchange_race_address;

    static int64_t FetchMax(
        volatile int64_t *address, int64_t value, uint64_t &retries
    ) {
        if (address == fetch_race_address) {
            __atomic_store_n(address, int64_t{-2}, __ATOMIC_RELEASE);
            fetch_race_address = nullptr;
        }
        return SymbolTestOps::FetchMax(address, value, retries);
    }

    static int64_t Exchange(volatile int64_t *address, int64_t value) {
        if (address == exchange_race_address) {
            __atomic_store_n(address, int64_t{7}, __ATOMIC_RELEASE);
            exchange_race_address = nullptr;
        }
        return SymbolTestOps::Exchange(address, value);
    }
};

volatile int64_t *PublicationFaultOps::fetch_race_address = nullptr;
volatile int64_t *PublicationFaultOps::exchange_race_address = nullptr;

void ResetSharedState(SharedTensorMapSidecar &map) {
    std::memset(&map, 0, sizeof(map));
    map.reclaim_upto.value = -1;
    for (uint32_t index = 0; index < kMapCapacity; ++index) {
        map.slots[index].seq.value = -1;
    }
    for (uint32_t task = 0; task < kMaxTasks; ++task) {
        for (uint32_t slot = 0; slot < kSharedOutputMaxPerTask; ++slot) {
            map.shared_outputs[task].published[slot].value = -1;
            map.shared_outputs[task].last_writer[slot].value = -1;
        }
    }
}

TensorDesc MakeTensor(uint64_t address, uint32_t owner) {
    TensorDesc tensor{};
    tensor.buffer_addr = address;
    tensor.buffer_size = 4096;
    tensor.owner_task_id = owner;
    tensor.ndims = 1;
    tensor.dtype = DataType::Float32;
    tensor.is_contiguous = true;
    tensor.shapes[0] = 1024;
    tensor.strides[0] = 1;
    tensor.extent_elem_cache = 1024;
    return tensor;
}

bool SameTensor(const TensorDesc &left, const TensorDesc &right) {
    return std::memcmp(&left, &right, sizeof(TensorDesc)) == 0;
}

void TestPublishAndResolve() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    ResetSharedState(*map);
    SymbolTestOps::now_calls.store(0, std::memory_order_relaxed);

    TensorDesc first = MakeTensor(0x100000000ULL, 0);
    TensorDesc second = MakeTensor(0x100001000ULL, 0);
    SubmitContext producer{};
    producer.task_id = 0;
    producer.result.task_id = 0;
    producer.result.count = 2;
    producer.result.tensors[0] = &first;
    producer.result.tensors[1] = &second;
    producer.shared_result.Reset(0);
    Check(producer.shared_result.AddOutputRef(0, 0), "producer accepts output slot 0");
    Check(producer.shared_result.AddOutputRef(0, 1), "producer accepts output slot 1");

    Check(
        PublishSharedTaskOutputs<SymbolTestOps>(*map, producer, 0),
        "first publication succeeds"
    );
    Check(map->shared_outputs[0].published[0].value == 0, "slot 0 is published");
    Check(map->shared_outputs[0].published[1].value == 0, "slot 1 is published");
    Check(map->shared_outputs[0].last_writer[0].value == 0, "slot 0 writer starts at producer");
    Check(SameTensor(map->shared_outputs[0].tensors[0], first), "published descriptor is exact");
    TensorDesc replacement = MakeTensor(0x200000000ULL, 0);
    producer.result.tensors[0] = &replacement;
    Check(
        !PublishSharedTaskOutputs<SymbolTestOps>(*map, producer, 0),
        "duplicate publication fails closed"
    );
    Check(
        SameTensor(map->shared_outputs[0].tensors[0], first),
        "failed duplicate publication cannot replace descriptor"
    );
    Check(
        map->shared_outputs[0].published[0].value == 0 &&
            map->shared_outputs[0].last_writer[0].value == 0,
        "failed duplicate publication preserves control state"
    );
    producer.result.tensors[0] = &first;

    TaskArgs input_args;
    ConstructTaskArgs(input_args);
    AppendSharedOutputRef(
        input_args, producer.shared_result.OutputRef(0), TensorArgType::Input
    );
    auto input_payload = std::make_unique<TaskPayload>();
    SubmitContext input_context{};
    input_context.task_id = 1;
    input_context.payload = input_payload.get();
    LocalStats input_stats{};
    int32_t input_fanin[kMaxFanin] = {};
    bool protocol_ok = false;
    uint32_t ordinary_lookups = UINT32_MAX;
    const uint32_t input_count = CollectSharedFanin<SymbolTestOps>(
        *map, input_args, input_context, 1, kHeapWindow, input_stats,
        input_fanin, protocol_ok, ordinary_lookups
    );
    Check(protocol_ok, "plain symbolic INPUT resolves");
    Check(input_count == 1 && input_fanin[0] == 0, "INPUT depends on producer writer");
    Check(ordinary_lookups == 0, "symbol INPUT never enters region map");
    Check(
        input_stats.result.shared_symbol_input_loads == 1,
        "INPUT writer load is counted"
    );
    Check(
        SameTensor(input_context.payload->tensors[0], first),
        "resolver copies descriptor into task payload scratch"
    );

    TaskArgs inout_args;
    ConstructTaskArgs(inout_args);
    AppendSharedOutputRef(
        inout_args, producer.shared_result.OutputRef(0), TensorArgType::Inout
    );
    auto inout_payload = std::make_unique<TaskPayload>();
    SubmitContext inout_context{};
    inout_context.task_id = 2;
    inout_context.payload = inout_payload.get();
    LocalStats inout_stats{};
    int32_t inout_fanin[kMaxFanin] = {};
    protocol_ok = false;
    ordinary_lookups = UINT32_MAX;
    const uint32_t inout_count = CollectSharedFanin<SymbolTestOps>(
        *map, inout_args, inout_context, 2, kHeapWindow, inout_stats,
        inout_fanin, protocol_ok, ordinary_lookups
    );
    Check(protocol_ok, "plain symbolic INOUT resolves");
    Check(inout_count == 1 && inout_fanin[0] == 0, "INOUT consumes old writer");
    Check(map->shared_outputs[0].last_writer[0].value == 2, "INOUT publishes current writer");
    Check(
        inout_stats.result.shared_symbol_inout_exchanges == 1,
        "INOUT exchange is counted"
    );

    // A(0) -> B(2) -> C(3)：后继 INPUT 必须看到最近一次 INOUT writer，
    // 不能退回最初 producer。
    TaskArgs successor_args;
    ConstructTaskArgs(successor_args);
    AppendSharedOutputRef(
        successor_args, producer.shared_result.OutputRef(0),
        TensorArgType::Input
    );
    auto successor_payload = std::make_unique<TaskPayload>();
    SubmitContext successor_context{};
    successor_context.task_id = 3;
    successor_context.payload = successor_payload.get();
    LocalStats successor_stats{};
    int32_t successor_fanin[kMaxFanin] = {};
    protocol_ok = false;
    ordinary_lookups = UINT32_MAX;
    const uint32_t successor_count = CollectSharedFanin<SymbolTestOps>(
        *map, successor_args, successor_context, 3, kHeapWindow,
        successor_stats, successor_fanin, protocol_ok, ordinary_lookups
    );
    Check(protocol_ok, "successor INPUT resolves after INOUT");
    Check(
        successor_count == 1 && successor_fanin[0] == 2,
        "successor INPUT observes latest writer"
    );
    Check(
        SymbolTestOps::now_calls.load(std::memory_order_relaxed) == 0,
        "already-published symbol fast path never reads the watchdog clock"
    );
}

void TestPublicationPreflightIsAllOrNothing() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    ResetSharedState(*map);

    TensorDesc first = MakeTensor(0x300000000ULL, 0);
    TensorDesc second = MakeTensor(0x300001000ULL, 0);
    SubmitContext producer{};
    producer.task_id = 0;
    producer.result.task_id = 0;
    producer.result.count = 2;
    producer.result.tensors[0] = &first;
    producer.result.tensors[1] = &second;
    producer.shared_result.Reset(0);
    Check(producer.shared_result.AddOutputRef(0, 0), "preflight adds slot 0");
    Check(producer.shared_result.AddOutputRef(0, 1), "preflight adds slot 1");

    // 人为污染第二槽，验证失败发生在任何 descriptor/前槽控制字写入前。
    map->shared_outputs[0].last_writer[1].value = 7;
    const TensorDesc zero{};
    Check(
        !PublishSharedTaskOutputs<SymbolTestOps>(*map, producer, 0),
        "later occupied slot rejects whole publication"
    );
    Check(
        map->shared_outputs[0].published[0].value == -1 &&
            map->shared_outputs[0].last_writer[0].value == -1,
        "later-slot failure leaves earlier control state untouched"
    );
    Check(
        SameTensor(map->shared_outputs[0].tensors[0], zero) &&
            SameTensor(map->shared_outputs[0].tensors[1], zero),
        "later-slot failure leaves every descriptor untouched"
    );
}

void TestPublicationCommitFaultsRollback() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    TensorDesc first = MakeTensor(0x350000000ULL, 0);
    TensorDesc second = MakeTensor(0x350001000ULL, 0);
    SubmitContext producer{};
    producer.task_id = 0;
    producer.result.task_id = 0;
    producer.result.count = 2;
    producer.result.tensors[0] = &first;
    producer.result.tensors[1] = &second;
    producer.shared_result.Reset(0);
    Check(producer.shared_result.AddOutputRef(0, 0), "fault test adds slot 0");
    Check(producer.shared_result.AddOutputRef(0, 1), "fault test adds slot 1");
    const TensorDesc zero{};

    ResetSharedState(*map);
    PublicationFaultOps::fetch_race_address =
        &map->shared_outputs[0].last_writer[1].value;
    Check(
        !PublishSharedTaskOutputs<PublicationFaultOps>(*map, producer, 0),
        "FetchMax race rejects whole publication"
    );
    Check(
        map->shared_outputs[0].last_writer[0].value == -1 &&
            map->shared_outputs[0].last_writer[1].value == -2,
        "FetchMax failure restores reserved and raced writer values"
    );
    Check(
        map->shared_outputs[0].published[0].value == -1 &&
            map->shared_outputs[0].published[1].value == -1 &&
            SameTensor(map->shared_outputs[0].tensors[0], zero) &&
            SameTensor(map->shared_outputs[0].tensors[1], zero),
        "FetchMax failure publishes no descriptor"
    );

    ResetSharedState(*map);
    PublicationFaultOps::exchange_race_address =
        &map->shared_outputs[0].published[1].value;
    Check(
        !PublishSharedTaskOutputs<PublicationFaultOps>(*map, producer, 0),
        "published Exchange race rejects whole publication"
    );
    Check(
        map->shared_outputs[0].published[0].value == -1 &&
            map->shared_outputs[0].published[1].value == -1 &&
            map->shared_outputs[0].last_writer[0].value == -1 &&
            map->shared_outputs[0].last_writer[1].value == -1,
        "published Exchange failure rolls back every control word"
    );
    Check(
        SameTensor(map->shared_outputs[0].tensors[0], zero) &&
            SameTensor(map->shared_outputs[0].tensors[1], zero),
        "published Exchange failure clears flushed descriptors"
    );
}

void TestConsumerWaitsForDelayedPublication() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    ResetSharedState(*map);

    TensorDesc output = MakeTensor(0x380000000ULL, 0);
    SubmitContext producer{};
    producer.task_id = 0;
    producer.result.task_id = 0;
    producer.result.count = 1;
    producer.result.tensors[0] = &output;
    producer.shared_result.Reset(0);
    Check(
        producer.shared_result.AddOutputRef(0, 0),
        "delayed producer accepts output slot"
    );

    TaskArgs args;
    ConstructTaskArgs(args);
    AppendSharedOutputRef(
        args, producer.shared_result.OutputRef(0), TensorArgType::Input
    );
    auto payload = std::make_unique<TaskPayload>();
    SubmitContext consumer{};
    consumer.task_id = 1;
    consumer.payload = payload.get();
    LocalStats stats{};
    int32_t fanin[kMaxFanin] = {};
    bool protocol_ok = false;
    uint32_t ordinary_lookups = UINT32_MAX;
    volatile int32_t fatal = 0;
    std::atomic<bool> publish_ok{false};

    SymbolTestOps::wait_address =
        &map->shared_outputs[0].published[0].value;
    SymbolTestOps::wait_loads.store(0, std::memory_order_relaxed);
    SymbolTestOps::now_calls.store(0, std::memory_order_relaxed);
    std::thread publisher([&] {
        // 等 consumer 已经观察到未发布状态后再发布，避免把本测试退化成
        // “进入 helper 前已经 ready”的普通快路径。
        while (SymbolTestOps::wait_loads.load(std::memory_order_acquire) == 0) {
            std::this_thread::yield();
        }
        publish_ok.store(
            PublishSharedTaskOutputs<SymbolTestOps>(*map, producer, 0),
            std::memory_order_release
        );
    });

    const uint32_t count = CollectSharedFanin<SymbolTestOps>(
        *map, args, consumer, 1, kHeapWindow, stats, fanin,
        protocol_ok, ordinary_lookups, &fatal
    );
    publisher.join();
    SymbolTestOps::wait_address = nullptr;

    Check(publish_ok.load(std::memory_order_acquire), "delayed publication succeeds");
    Check(protocol_ok && fatal == 0, "consumer waits without protocol failure");
    Check(
        SymbolTestOps::wait_loads.load(std::memory_order_acquire) > 1,
        "consumer performs at least one unpublished retry"
    );
    Check(
        SymbolTestOps::now_calls.load(std::memory_order_relaxed) >= 1,
        "unpublished slow path establishes a watchdog window"
    );
    Check(count == 1 && fanin[0] == 0, "delayed INPUT closes producer fanin");
    Check(ordinary_lookups == 0, "delayed symbol never enters ordinary map");
    Check(
        SameTensor(payload->tensors[0], output),
        "descriptor is visible after publication becomes ready"
    );
}

void TestPublicationWaitFailuresFailClosed() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    const FdwicOutputRef output_ref{0, 0, 0, 0, 0, 0};

    ResetSharedState(*map);
    map->shared_outputs[0].published[0].value = 7;
    volatile int32_t fatal = 0;
    SymbolTestOps::now_calls.store(0, std::memory_order_relaxed);
    Check(
        !WaitForSharedOutputPublished<SymbolTestOps>(
            *map, output_ref, &fatal
        ),
        "unexpected publication value is rejected"
    );
    Check(fatal == 1, "unexpected publication value broadcasts fatal");
    Check(
        SymbolTestOps::now_calls.load(std::memory_order_relaxed) == 0,
        "unexpected ready value fails before opening a watchdog window"
    );

    ResetSharedState(*map);
    fatal = 1;
    SymbolTestOps::now_calls.store(0, std::memory_order_relaxed);
    Check(
        !WaitForSharedOutputPublished<SymbolTestOps>(
            *map, output_ref, &fatal
        ),
        "existing fatal terminates unpublished wait"
    );
    Check(
        SymbolTestOps::now_calls.load(std::memory_order_relaxed) == 1,
        "fatal wait exits before a watchdog recheck"
    );

    ResetSharedState(*map);
    fatal = 0;
    ExpiredWaitOps::calls.store(0, std::memory_order_relaxed);
    Check(
        !WaitForSharedOutputPublished<ExpiredWaitOps>(
            *map, output_ref, &fatal
        ),
        "watchdog terminates permanently unpublished symbol"
    );
    Check(fatal == 1, "publication watchdog broadcasts fatal");
    Check(
        ExpiredWaitOps::calls.load(std::memory_order_relaxed) == 2,
        "watchdog clock is read only at slow-path begin and periodic recheck"
    );
}

void TestInvalidReferencesFailClosed() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    ResetSharedState(*map);
    auto payload = std::make_unique<TaskPayload>();

    const FdwicOutputRef invalid_refs[] = {
        FdwicOutputRef{3, 0, 0, 0, 0, 0},
        FdwicOutputRef{0, 8, 0, 0, 0, 0},
        FdwicOutputRef{0, 0, 1, 1, 16, 0},
    };
    for (const FdwicOutputRef reference : invalid_refs) {
        TaskArgs args;
        ConstructTaskArgs(args);
        AppendSharedOutputRef(args, reference, TensorArgType::Input);
        SubmitContext context{};
        context.task_id = 2;
        context.payload = payload.get();
        LocalStats stats{};
        int32_t fanin[kMaxFanin] = {};
        bool protocol_ok = true;
        uint32_t ordinary_lookups = UINT32_MAX;
        (void)CollectSharedFanin<SymbolTestOps>(
            *map, args, context, 2, kHeapWindow, stats, fanin,
            protocol_ok, ordinary_lookups
        );
        Check(!protocol_ok, "invalid/future/view symbol fails closed");
        Check(ordinary_lookups == 0, "invalid symbol does not enter region map");
    }

    // 第一项是合法 INOUT，第二项才非法；两遍解析必须在第一遍就拒绝，
    // 不能提前改写 writer、payload、统计或输出 fanin。
    TensorDesc published = MakeTensor(0x400000000ULL, 0);
    SubmitContext producer{};
    producer.task_id = 0;
    producer.result.task_id = 0;
    producer.result.count = 1;
    producer.result.tensors[0] = &published;
    producer.shared_result.Reset(0);
    Check(producer.shared_result.AddOutputRef(0, 0), "late-failure producer output");
    Check(
        PublishSharedTaskOutputs<SymbolTestOps>(*map, producer, 0),
        "late-failure producer publishes"
    );

    TaskArgs mixed_args;
    ConstructTaskArgs(mixed_args);
    AppendSharedOutputRef(
        mixed_args, producer.shared_result.OutputRef(0), TensorArgType::Inout
    );
    AppendSharedOutputRef(
        mixed_args, FdwicOutputRef{3, 0, 0, 0, 0, 0},
        TensorArgType::Input
    );
    std::memset(payload.get(), 0xA5, sizeof(*payload));
    SubmitContext mixed_context{};
    mixed_context.task_id = 2;
    mixed_context.payload = payload.get();
    LocalStats mixed_stats{};
    int32_t mixed_fanin[kMaxFanin];
    for (uint32_t index = 0; index < kMaxFanin; ++index) {
        mixed_fanin[index] = -77;
    }
    bool mixed_protocol_ok = true;
    uint32_t mixed_ordinary_lookups = UINT32_MAX;
    (void)CollectSharedFanin<SymbolTestOps>(
        *map, mixed_args, mixed_context, 2, kHeapWindow, mixed_stats,
        mixed_fanin, mixed_protocol_ok, mixed_ordinary_lookups
    );
    Check(!mixed_protocol_ok, "late invalid symbol rejects whole resolve");
    Check(
        map->shared_outputs[0].last_writer[0].value == 0,
        "late invalid symbol does not publish earlier INOUT writer"
    );
    Check(
        mixed_stats.result.shared_symbol_input_loads == 0 &&
            mixed_stats.result.shared_symbol_inout_exchanges == 0,
        "late invalid symbol publishes no resolve statistics"
    );
    Check(
        mixed_fanin[0] == -77,
        "late invalid symbol publishes no fanin"
    );
    const unsigned char *payload_bytes =
        reinterpret_cast<const unsigned char *>(payload.get());
    bool payload_untouched = true;
    for (size_t index = 0; index < sizeof(*payload); ++index) {
        payload_untouched &= payload_bytes[index] == 0xA5;
    }
    Check(payload_untouched, "late invalid symbol does not touch payload scratch");
}

}  // namespace

int main() {
    TestPublishAndResolve();
    TestPublicationPreflightIsAllOrNothing();
    TestPublicationCommitFaultsRollback();
    TestConsumerWaitsForDelayedPublication();
    TestPublicationWaitFailuresFailClosed();
    TestInvalidReferencesFailClosed();
    if (g_failures != 0) {
        std::fprintf(stderr, "[FAIL] shared-output symbol tests: %d\n", g_failures);
        return 1;
    }
    std::puts("[PASS] shared-output symbol publish/resolve tests");
    return 0;
}
