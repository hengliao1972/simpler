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

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <new>
#include <sys/mman.h>

#ifndef PTO_FDWIC_SHARED_MAP
#define PTO_FDWIC_SHARED_MAP 1
#endif
#ifndef PA_BUILD_SWIMLANE
#define PA_BUILD_SWIMLANE 1
#endif
#define PA_DEVICE inline
#define PA_GM
#include "pa_exec_adapter.h"

namespace {

using namespace pa_scheduler;
using namespace pa_scheduler::cross_core;

constexpr uint32_t kBuildOwnerAic = 3;
constexpr uint32_t kBuildOwnerAiv = 41;

int g_failures = 0;

void Check(bool condition, TaskKind kind, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(
        stderr, "[FAIL] PA exec adapter kind=%u: %s\n",
        static_cast<uint32_t>(kind), message
    );
    ++g_failures;
}

void CheckMapping(bool condition, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(
        stderr, "[FAIL] PA execute candidate eligibility: %s\n",
        message
    );
    ++g_failures;
}

uint32_t WorkerLane(uint32_t owner) {
    return owner < kAicWorkers
        ? 0U : 1U + (owner - kAicWorkers) % 2U;
}

uint32_t WorkerBlock(uint32_t owner) {
    return owner < kAicWorkers
        ? owner : (owner - kAicWorkers) / 2U;
}

uint32_t WorkerSubBlock(uint32_t owner) {
    return WorkerLane(owner) == 2U ? 1U : 0U;
}

void ConfigureWorkerIdentity(WorkerState &worker, uint32_t owner) {
    worker.core_idx = static_cast<int32_t>(owner);
    worker.role = owner < kAicWorkers
        ? CoreRole::Aic : CoreRole::Aiv;
    worker.block_id = static_cast<int32_t>(WorkerBlock(owner));
    worker.lane = static_cast<int32_t>(WorkerLane(owner));
    worker.sub_block_id = static_cast<int32_t>(WorkerSubBlock(owner));
}

bool TestExecuteCandidateEligibility() {
    constexpr uint32_t kRoutingPeriod = kAivWorkers;
    std::array<bool, kWorkers> candidate_destinations{};
    bool exact = true;
    for (ExecEngineClass engine : {
             ExecEngineClass::Aic, ExecEngineClass::Aiv}) {
        const uint32_t role_begin = engine == ExecEngineClass::Aic
            ? 0U : kAicWorkers;
        const uint32_t role_end = engine == ExecEngineClass::Aic
            ? kAicWorkers : kWorkers;
        for (uint32_t task_id = 0;
             task_id < kRoutingPeriod; ++task_id) {
            uint32_t primary = kExecUnboundOwner;
            uint32_t secondary = kExecUnboundOwner;
            const bool candidates_ok = FixedPaExecuteCandidates(
                task_id, engine, primary, secondary
            );
            exact &= candidates_ok &&
                primary >= role_begin && primary < role_end &&
                secondary >= role_begin && secondary < role_end &&
                primary != secondary &&
                WorkerBlock(primary) != WorkerBlock(secondary);
            if (engine == ExecEngineClass::Aiv) {
                exact &= WorkerLane(primary) == WorkerLane(secondary) &&
                    WorkerSubBlock(primary) ==
                        WorkerSubBlock(secondary);
            }
            if (candidates_ok && primary < kWorkers &&
                secondary < kWorkers) {
                candidate_destinations[primary] = true;
                candidate_destinations[secondary] = true;
            }

            // Build owner 穷举全部 96 核，包含与 engine 不同类型
            // 的 worker；execute owner 同样穷举，确保资格判定只
            // 接受“同 engine 的双候选且非 builder”。
            for (uint32_t build_owner = 0;
                 build_owner < kWorkers; ++build_owner) {
                for (uint32_t execute_owner = 0;
                     execute_owner < kWorkers; ++execute_owner) {
                    const bool expected =
                        execute_owner != build_owner &&
                        (execute_owner == primary ||
                         execute_owner == secondary);
                    exact &= PaExecuteOwnerEligible(
                        task_id, build_owner, engine, execute_owner
                    ) == expected;
                }
            }
        }
    }
    for (uint32_t owner = 0; owner < kWorkers; ++owner) {
        exact &= candidate_destinations[owner];
        exact &= PaExecOwnerMatchesEngine(
                     owner,
                     owner < kAicWorkers
                         ? ExecEngineClass::Aic
                         : ExecEngineClass::Aiv
                 );
        exact &= !PaExecOwnerMatchesEngine(
                     owner,
                     owner < kAicWorkers
                         ? ExecEngineClass::Aiv
                         : ExecEngineClass::Aic
                 );
    }

    uint32_t primary = 0;
    uint32_t secondary = 0;
    exact &= !FixedPaExecuteCandidates(
                 7, ExecEngineClass::None, primary, secondary
             ) &&
             primary == kExecUnboundOwner &&
             secondary == kExecUnboundOwner;
    exact &= !FixedPaExecuteCandidates(
                 7, ExecEngineClass::Joint, primary, secondary
             ) &&
             primary == kExecUnboundOwner &&
             secondary == kExecUnboundOwner;
    exact &= !FixedPaExecuteCandidates(
                 kMaxTasks, ExecEngineClass::Aic,
                 primary, secondary
             ) &&
             primary == kExecUnboundOwner &&
             secondary == kExecUnboundOwner;

    uint32_t aic_primary = kExecUnboundOwner;
    uint32_t aic_secondary = kExecUnboundOwner;
    uint32_t aiv_primary = kExecUnboundOwner;
    uint32_t aiv_secondary = kExecUnboundOwner;
    exact &= FixedPaExecuteCandidates(
        7, ExecEngineClass::Aic, aic_primary, aic_secondary
    );
    exact &= FixedPaExecuteCandidates(
        7, ExecEngineClass::Aiv, aiv_primary, aiv_secondary
    );

    // Build owner 不要求匹配 engine：交叉核类型的 builder 仍可
    // 把 payload 交给合法的执行候选。
    exact &= PaExecuteOwnerEligible(
        7, kAicWorkers, ExecEngineClass::Aic, aic_primary
    );
    exact &= PaExecuteOwnerEligible(
        7, 0, ExecEngineClass::Aiv, aiv_primary
    );

    exact &= !PaExecOwnerMatchesEngine(
        kWorkers, ExecEngineClass::Aic
    );
    exact &= !PaExecOwnerMatchesEngine(
        kExecUnboundOwner, ExecEngineClass::Aiv
    );
    exact &= !PaExecOwnerMatchesEngine(
        0, ExecEngineClass::None
    );
    exact &= !PaExecOwnerMatchesEngine(
        0, ExecEngineClass::Joint
    );
    exact &= !PaExecuteOwnerEligible(
        kMaxTasks, 0, ExecEngineClass::Aic, aic_primary
    );
    exact &= !PaExecuteOwnerEligible(
        7, kWorkers, ExecEngineClass::Aic, aic_primary
    );
    exact &= !PaExecuteOwnerEligible(
        7, kExecUnboundOwner, ExecEngineClass::Aiv, aiv_primary
    );
    exact &= !PaExecuteOwnerEligible(
        7, 0, ExecEngineClass::Aic, kWorkers
    );
    exact &= !PaExecuteOwnerEligible(
        7, 0, ExecEngineClass::Aiv, kExecUnboundOwner
    );
    exact &= !PaExecuteOwnerEligible(
        7, 0, ExecEngineClass::None, aic_primary
    );
    exact &= !PaExecuteOwnerEligible(
        7, 0, ExecEngineClass::Joint, aic_primary
    );
    exact &= !PaExecuteOwnerEligible(
        7, aic_primary, ExecEngineClass::Aic, aic_primary
    );
    exact &= !PaExecuteOwnerEligible(
        7, aiv_primary, ExecEngineClass::Aiv, aiv_primary
    );
    CheckMapping(
        exact,
        "two candidates and generic eligibility cover all workers"
    );
    return exact;
}

// 端到端 adapter 用例需要一个具体 executor 才能验证 payload
// 搬运。这里仅在测试内从双候选中确定性挑选一个非 builder，
// 不定义生产环境的 owner 仲裁语义。
bool SelectTestExecuteOwner(
    uint32_t task_id, uint32_t build_owner,
    ExecEngineClass engine, uint32_t &execute_owner
) {
    execute_owner = kExecUnboundOwner;
    uint32_t primary = kExecUnboundOwner;
    uint32_t secondary = kExecUnboundOwner;
    if (!FixedPaExecuteCandidates(
            task_id, engine, primary, secondary
        )) {
        return false;
    }
    const uint32_t selected = primary != build_owner
        ? primary : secondary;
    if (!PaExecuteOwnerEligible(
            task_id, build_owner, engine, selected
        )) {
        return false;
    }
    execute_owner = selected;
    return true;
}

// 该 Ops 只承担 CPU 上的协议顺序与精确搬运门槛，不模拟 A5 DCache。
// 所有共享控制字仍使用 acquire/release 原子，payload 则沿用生产 helper
// 的普通 word 搬运接口，避免测试重新实现 pack/bind。
struct AdapterTestOps {
    static inline uint32_t flush_calls = 0;
    static inline uint32_t invalidate_calls = 0;

    static void ResetCounters() {
        flush_calls = 0;
        invalidate_calls = 0;
    }

    static int64_t Load(volatile int64_t *address) {
        return __atomic_fetch_add(
            address, int64_t{0}, __ATOMIC_ACQUIRE
        );
    }

    static int64_t CompareExchange(
        volatile int64_t *address, int64_t expected,
        int64_t desired
    ) {
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(
            address, &observed, desired, false,
            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        );
        return observed;
    }

    static void StorePayloadWord(
        volatile uint64_t *address, uint64_t value
    ) {
        *address = value;
    }

    static void StoreTokenPayloadWord(
        volatile uint64_t *address, uint64_t value
    ) {
        *address = value;
    }

    static uint64_t LoadPayloadWord(
        const volatile uint64_t *address
    ) {
        return *address;
    }

    static void PreloadBuildDestination(void *, uint64_t) {}

    static void PreloadPayloadSource(const void *, uint64_t) {}

    static void PreloadTokenDestination(void *, uint64_t) {}

    static void BeforeBuiltPublish(uint32_t) {}

    static void BeforePayloadAcquire(uint32_t) {}

    static void FlushRegion(void *, uint64_t) {
        ++flush_calls;
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
    }

    static void InvalidateRegion(const void *, uint64_t) {
        ++invalidate_calls;
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
    }
};

struct CaseShape {
    TaskKind kind;
    uint32_t task_id;
    uint32_t function_id;
    uint16_t tensor_count;
    uint16_t scalar_count;
    uint16_t fanin_count;
    uint32_t payload_bytes;
    uint32_t payload_lines;
    ExecEngineClass engine;
};

constexpr std::array<CaseShape, 4> kCases = {{
    {TaskKind::Qk, 1, 0, 4, 2, 0, 592, 10, ExecEngineClass::Aic},
    {TaskKind::Sf, 2, 1, 4, 3, 1, 604, 10, ExecEngineClass::Aiv},
    {TaskKind::Pv, 3, 2, 4, 2, 1, 596, 10, ExecEngineClass::Aic},
    {TaskKind::Up, 4, 3, 7, 2, 3, 988, 16, ExecEngineClass::Aiv},
}};

struct CaseFixture {
    TaskArgs args{};
    SubmitContext context{};
    TensorCreateInfo create_infos[kExecMaxTensors]{};
    TensorDesc local_sources[kExecMaxTensors]{};
    TensorDesc expected_tensors[kExecMaxTensors]{};
    uint64_t expected_scalars[kExecMaxScalars]{};
    int32_t expected_fanin[kExecMaxFanin]{};
    uint32_t shared_producers[kExecMaxTensors]{};
    uint32_t shared_slots[kExecMaxTensors]{};
    uint32_t shared_source_count = 0;
};

TensorDesc PatternTensor(uint32_t seed) {
    TensorDesc tensor{};
    auto *bytes = reinterpret_cast<uint8_t *>(&tensor);
    for (uint32_t index = 0; index < sizeof(TensorDesc); ++index) {
        bytes[index] = static_cast<uint8_t>(
            (seed * 37U + index * 13U + 11U) & 0xffU
        );
    }
    // 保留可读且彼此不同的核心字段；其余字节（包括 padding）继续使用
    // pattern，从而能发现只复制“常用字段”而不是完整 128B 的错误。
    tensor.buffer_addr = UINT64_C(0x100000000) + seed * 0x1000U;
    tensor.buffer_size = UINT64_C(0x2000) + seed * 64U;
    tensor.owner_task_id = seed;
    tensor.start_offset = seed * 16U;
    tensor.version = static_cast<int32_t>(seed + 7U);
    tensor.ndims = 2;
    tensor.dtype = seed % 2U == 0
        ? DataType::Float32 : DataType::Bfloat16;
    tensor.manual_dep = (seed & 1U) != 0;
    tensor.is_contiguous = true;
    tensor.child_memory = static_cast<uint8_t>(seed & 3U);
    tensor.shapes[0] = seed + 1U;
    tensor.shapes[1] = seed + 2U;
    tensor.extent_elem_cache =
        static_cast<uint64_t>(tensor.shapes[0]) * tensor.shapes[1];
    tensor.strides[0] = tensor.shapes[1];
    tensor.strides[1] = 1;
    return tensor;
}

void PolluteTensor(TensorDesc &tensor, uint32_t salt) {
    auto *bytes = reinterpret_cast<uint8_t *>(&tensor);
    for (uint32_t index = 0; index < sizeof(TensorDesc); ++index) {
        bytes[index] = static_cast<uint8_t>(
            (0xa5U + salt + index * 7U) & 0xffU
        );
    }
}

SchedulerState *MapSchedulerState() {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    void *memory = mmap(
        nullptr, sizeof(SchedulerState), PROT_READ | PROT_WRITE,
        flags, -1, 0
    );
    if (memory == MAP_FAILED) {
        std::perror("mmap SchedulerState");
        return nullptr;
    }
    // mmap 已提供全零 fresh pages；无花括号 placement-new 只建立
    // trivial SchedulerState 的对象生命周期，避免主动触碰近 1 GiB。
    return ::new (memory) SchedulerState;
}

FdwicOutputRef PlainOutputRef(
    uint32_t producer, uint32_t output_slot
) {
    return FdwicOutputRef{
        static_cast<int32_t>(producer),
        static_cast<int16_t>(output_slot), 0, 0, 0, 0,
    };
}

void AddLocalArg(
    CaseFixture &fixture, TensorArgType tag, uint32_t seed
) {
    const uint32_t index =
        static_cast<uint32_t>(fixture.args.tensor_count);
    fixture.local_sources[index] = PatternTensor(seed);
    AddLocalTensor(
        fixture.args, fixture.local_sources[index], tag
    );
    fixture.expected_tensors[index] =
        fixture.local_sources[index];
}

void AddSharedArg(
    CaseFixture &fixture, SchedulerState &state,
    uint32_t producer, uint32_t output_slot,
    TensorArgType tag, uint32_t seed
) {
    const uint32_t index =
        static_cast<uint32_t>(fixture.args.tensor_count);
    SharedOutputCell &cell =
        state.shared_map.shared_outputs[producer];
    cell.tensors[output_slot] = PatternTensor(seed);
    cell.published[output_slot].value =
        static_cast<int64_t>(producer);
    AddOutputHandleTensor(
        fixture.args, PlainOutputRef(producer, output_slot), tag
    );
    fixture.expected_tensors[index] = cell.tensors[output_slot];
    fixture.shared_producers[fixture.shared_source_count] = producer;
    fixture.shared_slots[fixture.shared_source_count] = output_slot;
    ++fixture.shared_source_count;
}

void AddOutputArg(
    CaseFixture &fixture, TaskPayload &payload, uint32_t seed
) {
    const uint32_t index =
        static_cast<uint32_t>(fixture.args.tensor_count);
    AddOutput(fixture.args, fixture.create_infos[index]);
    payload.tensors[index] = PatternTensor(seed);
    fixture.expected_tensors[index] = payload.tensors[index];
}

void AddScalarArg(CaseFixture &fixture, uint64_t value) {
    const uint32_t index =
        static_cast<uint32_t>(fixture.args.scalar_count);
    AddScalar(fixture.args, value);
    fixture.expected_scalars[index] = value;
}

void SetFanin(
    CaseFixture &fixture,
    std::initializer_list<int32_t> producers
) {
    fixture.context.fanin_count =
        static_cast<int32_t>(producers.size());
    uint32_t index = 0;
    for (int32_t producer : producers) {
        fixture.context.fanin[index] = producer;
        fixture.expected_fanin[index] = producer;
        ++index;
    }
}

void BuildCaseArgs(
    const CaseShape &shape, SchedulerState &state,
    WorkerState &worker, CaseFixture &fixture
) {
    ResetTaskArgs(fixture.args);
    TaskPayload &payload =
        worker.payloads[shape.task_id & kPayloadMask];
    fixture.context.self = &worker;
    fixture.context.payload = &payload;
    fixture.context.task_id = static_cast<int32_t>(shape.task_id);
    fixture.context.result.task_id = shape.task_id;
    fixture.context.joint = false;
    fixture.context.joint_init = false;
    fixture.context.joint_block = -1;
    fixture.context.joint_slot = -1;
    fixture.context.joint_count = 0;

    switch (shape.kind) {
        case TaskKind::Qk:
            AddLocalArg(fixture, TensorArgType::Input, 101);
            AddLocalArg(fixture, TensorArgType::Input, 102);
            AddLocalArg(fixture, TensorArgType::Input, 103);
            AddOutputArg(fixture, payload, 104);
            AddScalarArg(fixture, 64);
            AddScalarArg(fixture, UINT64_C(0x1122334455667788));
            SetFanin(fixture, {});
            break;
        case TaskKind::Sf:
            AddSharedArg(
                fixture, state, 1, 0,
                TensorArgType::Input, 201
            );
            AddOutputArg(fixture, payload, 202);
            AddOutputArg(fixture, payload, 203);
            AddOutputArg(fixture, payload, 204);
            AddScalarArg(fixture, UINT64_C(0x3f3504f3));
            AddScalarArg(fixture, 64);
            AddScalarArg(fixture, 128);
            SetFanin(fixture, {1});
            break;
        case TaskKind::Pv:
            AddSharedArg(
                fixture, state, 2, 0,
                TensorArgType::Input, 301
            );
            AddLocalArg(fixture, TensorArgType::Input, 302);
            AddLocalArg(fixture, TensorArgType::Input, 303);
            AddOutputArg(fixture, payload, 304);
            AddScalarArg(fixture, 64);
            AddScalarArg(fixture, UINT64_C(0x8877665544332211));
            SetFanin(fixture, {2});
            break;
        case TaskKind::Up:
            AddSharedArg(
                fixture, state, 2, 1,
                TensorArgType::Input, 401
            );
            AddSharedArg(
                fixture, state, 2, 2,
                TensorArgType::Input, 402
            );
            AddSharedArg(
                fixture, state, 3, 0,
                TensorArgType::Input, 403
            );
            AddSharedArg(
                fixture, state, 0, 2,
                TensorArgType::Inout, 404
            );
            AddSharedArg(
                fixture, state, 0, 1,
                TensorArgType::Inout, 405
            );
            AddSharedArg(
                fixture, state, 0, 0,
                TensorArgType::Inout, 406
            );
            AddLocalArg(fixture, TensorArgType::Inout, 407);
            AddScalarArg(fixture, 1);
            AddScalarArg(fixture, 1);
            SetFanin(fixture, {2, 3, 0});
            break;
        case TaskKind::Alloc:
        case TaskKind::Count:
            break;
    }

    fixture.context.tensor_count = fixture.args.tensor_count;
    fixture.context.scalar_count = fixture.args.scalar_count;
}

uint64_t TensorWord(const TensorDesc &tensor, uint32_t word) {
    uint64_t value = 0;
    std::memcpy(
        &value,
        reinterpret_cast<const uint8_t *>(&tensor) +
            word * sizeof(uint64_t),
        sizeof(value)
    );
    return value;
}

int32_t PayloadFanin(
    const ExecPayloadStorage &payload,
    const ExecPayloadLayout &layout, uint32_t edge
) {
    const uint64_t packed =
        payload.words[layout.fanin_word_offset + edge / 2U];
    return static_cast<int32_t>(
        edge % 2U == 0
            ? static_cast<uint32_t>(packed)
            : static_cast<uint32_t>(packed >> 32U)
    );
}

bool PayloadMatches(
    const ExecPayloadStorage &payload, const CaseShape &shape,
    const CaseFixture &fixture, uint64_t expected_vend,
    ExecPayloadLayout &layout
) {
    const ExecPayloadHeader header =
        DecodeExecPayloadHeader(payload);
    bool exact =
        ComputeExecPayloadLayout(
            shape.tensor_count, shape.scalar_count,
            shape.fanin_count, layout
        ) &&
        header.task_id == shape.task_id &&
        header.function_address == 0 &&
        header.completion_vend == expected_vend &&
        header.function_id == shape.function_id &&
        header.payload_bytes == shape.payload_bytes &&
        header.tensor_count == shape.tensor_count &&
        header.scalar_count == shape.scalar_count &&
        header.fanin_count == shape.fanin_count &&
        header.engine_class == shape.engine &&
        header.flags == 0 &&
        header.multicore_group_id == 0 &&
        header.multicore_rank == 0 &&
        header.multicore_size == 1 &&
        layout.payload_bytes == shape.payload_bytes &&
        layout.payload_lines == shape.payload_lines &&
        payload.words[6] == 0 && payload.words[7] == 0;

    for (uint32_t tensor = 0;
         tensor < shape.tensor_count; ++tensor) {
        for (uint32_t word = 0;
             word < kExecTensorDescWords; ++word) {
            exact &= payload.words[
                layout.tensor_word_offset +
                tensor * kExecTensorDescWords + word
            ] == TensorWord(fixture.expected_tensors[tensor], word);
        }
    }
    for (uint32_t scalar = 0;
         scalar < shape.scalar_count; ++scalar) {
        exact &= payload.words[
            layout.scalar_word_offset + scalar
        ] == fixture.expected_scalars[scalar];
    }
    for (uint32_t edge = 0;
         edge < shape.fanin_count; ++edge) {
        exact &= PayloadFanin(payload, layout, edge) ==
            fixture.expected_fanin[edge];
    }
    for (uint32_t word = layout.written_words;
         word < layout.payload_lines * kExecHeaderWords; ++word) {
        exact &= payload.words[word] == 0;
    }
    return exact;
}

void SnapshotPayload(
    const ExecPayloadStorage &payload,
    std::array<uint64_t, kExecMaxPayloadWords> &snapshot
) {
    for (uint32_t word = 0; word < snapshot.size(); ++word) {
        snapshot[word] = payload.words[word];
    }
}

bool PayloadEqualsSnapshot(
    const ExecPayloadStorage &payload,
    const std::array<uint64_t, kExecMaxPayloadWords> &snapshot
) {
    for (uint32_t word = 0; word < snapshot.size(); ++word) {
        if (payload.words[word] != snapshot[word]) {
            return false;
        }
    }
    return true;
}

void PolluteBuilderSources(
    SchedulerState &state, WorkerState &worker,
    const CaseShape &shape, CaseFixture &fixture,
    PaExecPayloadSource &source
) {
    for (uint32_t tensor = 0;
         tensor < shape.tensor_count; ++tensor) {
        PolluteTensor(fixture.local_sources[tensor], tensor + 1U);
        PolluteTensor(
            fixture.context.payload->tensors[tensor],
            tensor + 41U
        );
    }
    for (uint32_t index = 0;
         index < fixture.shared_source_count; ++index) {
        PolluteTensor(
            state.shared_map.shared_outputs[
                fixture.shared_producers[index]
            ].tensors[fixture.shared_slots[index]],
            index + 81U
        );
    }
    for (uint32_t scalar = 0;
         scalar < shape.scalar_count; ++scalar) {
        fixture.args.scalars[scalar] ^=
            UINT64_C(0xffff0000ffff0000);
    }
    for (uint32_t edge = 0;
         edge < shape.fanin_count; ++edge) {
        fixture.context.fanin[edge] = -1;
    }
    fixture.args.tensor_count = 0;
    fixture.args.scalar_count = 0;
    fixture.context.tensor_count = 0;
    fixture.context.scalar_count = 0;
    fixture.context.fanin_count = 0;
    worker.heap_next ^= UINT64_C(0x00ff00ff00ff0000);
    auto *source_bytes = reinterpret_cast<uint8_t *>(&source);
    for (uint32_t byte = 0; byte < sizeof(source); ++byte) {
        source_bytes[byte] = 0xa5;
    }
}

bool TokenDispatchMatches(
    const ExecutionToken &token, const CaseShape &shape,
    const CaseFixture &fixture, const WorkerState &executor,
    uint64_t expected_vend
) {
    const ExecPayloadHeader header = ExecutionTokenHeader(token);
    ExecPayloadLayout layout{};
    if (!ComputeExecPayloadLayout(
            header.tensor_count, header.scalar_count,
            header.fanin_count, layout
        )) {
        return false;
    }

    bool exact =
        token.control.phase == ExecTokenPhase::WaitingFanin &&
        token.control.task_id == shape.task_id &&
        token.control.engine_class == shape.engine &&
        token.control.payload_lines == shape.payload_lines &&
        token.control.payload_bytes == shape.payload_bytes &&
        header.completion_vend == expected_vend;

    const uint64_t *dispatch = ExecutionTokenDispatchArgs(token);
    for (uint32_t tensor = 0;
         tensor < shape.tensor_count; ++tensor) {
        const uint64_t expected_pointer = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(
                &token.payload.words[
                    layout.tensor_word_offset +
                    tensor * kExecTensorDescWords
                ]
            )
        );
        exact &= dispatch[tensor] == expected_pointer;
        for (uint32_t word = 0;
             word < kExecTensorDescWords; ++word) {
            uint64_t observed = 0;
            exact &= ExecutionTokenTensorWord(
                token, tensor, word, observed
            ) &&
                observed ==
                    TensorWord(fixture.expected_tensors[tensor], word);
        }
    }
    for (uint32_t scalar = 0;
         scalar < shape.scalar_count; ++scalar) {
        uint64_t observed = 0;
        exact &= ExecutionTokenScalar(token, scalar, observed) &&
            observed == fixture.expected_scalars[scalar] &&
            dispatch[shape.tensor_count + scalar] == observed;
    }
    for (uint32_t edge = 0;
         edge < shape.fanin_count; ++edge) {
        int32_t observed = -1;
        exact &= ExecutionTokenFanin(token, edge, observed) &&
            observed == fixture.expected_fanin[edge];
    }

    const uint64_t expected_local_pointer = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(
            &token.dispatch.local_context[0]
        )
    );
    const uint64_t expected_global_pointer = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(
            &token.dispatch.global_context[0]
        )
    );
    exact &= dispatch[kExecDispatchLocalContextIndex] ==
                 expected_local_pointer &&
             dispatch[kExecDispatchGlobalContextIndex] ==
                 expected_global_pointer;

    const auto &local = *reinterpret_cast<const PaLocalContext *>(
        &token.dispatch.local_context[0]
    );
    const auto &global = *reinterpret_cast<const PaGlobalContext *>(
        &token.dispatch.global_context[0]
    );
    exact &=
        local.block_index == 0 && local.block_count == 1 &&
        local.async.completion_count == 0 &&
        local.async.completion_error_code == 0 &&
        local.async.completion_entries == 0 &&
        local.async.completion_capacity == 0 &&
        local.async.alignment_padding == 0 &&
        local.async.task_token == kInvalidTaskId &&
        global.sub_block_id == executor.sub_block_id;
    return exact;
}

enum class ShapeField : uint32_t {
    Tensor,
    Scalar,
    Fanin,
};

const char *MakeRejectMessage(ShapeField field) {
    switch (field) {
        case ShapeField::Tensor:
            return "Make rejects malformed tensor count";
        case ShapeField::Scalar:
            return "Make rejects malformed scalar count";
        case ShapeField::Fanin:
            return "Make rejects malformed fanin count";
    }
    return "Make rejects unknown malformed shape";
}

const char *FinalRejectMessage(ShapeField field) {
    switch (field) {
        case ShapeField::Tensor:
            return "final validator rejects malformed tensor count";
        case ShapeField::Scalar:
            return "final validator rejects malformed scalar count";
        case ShapeField::Fanin:
            return "final validator rejects malformed fanin count";
    }
    return "final validator rejects unknown malformed shape";
}

const char *BindRejectMessage(ShapeField field) {
    switch (field) {
        case ShapeField::Tensor:
            return "Claim binding rejects malformed tensor count";
        case ShapeField::Scalar:
            return "Claim binding rejects malformed scalar count";
        case ShapeField::Fanin:
            return "Claim binding rejects malformed fanin count";
    }
    return "Claim binding rejects unknown malformed shape";
}

uint16_t MalformedCount(uint16_t expected) {
    return expected == 0 ? 1 : static_cast<uint16_t>(expected - 1U);
}

void SetMalformedSpecCount(
    ExecPayloadSpec &spec, ShapeField field
) {
    switch (field) {
        case ShapeField::Tensor:
            spec.tensor_count = MalformedCount(spec.tensor_count);
            break;
        case ShapeField::Scalar:
            spec.scalar_count = MalformedCount(spec.scalar_count);
            break;
        case ShapeField::Fanin:
            spec.fanin_count = MalformedCount(spec.fanin_count);
            break;
    }
}

bool MakeRejectsMalformedShape(
    const CaseShape &shape, const WorkerState &worker,
    const TaskArgs &valid_args, const SubmitContext &valid_context,
    ShapeField field
) {
    TaskArgs args = valid_args;
    SubmitContext context = valid_context;
    switch (field) {
        case ShapeField::Tensor:
            args.tensor_count = MalformedCount(shape.tensor_count);
            context.tensor_count = args.tensor_count;
            break;
        case ShapeField::Scalar:
            args.scalar_count = MalformedCount(shape.scalar_count);
            context.scalar_count = args.scalar_count;
            break;
        case ShapeField::Fanin:
            context.fanin_count = MalformedCount(shape.fanin_count);
            break;
    }
    ExecPayloadSpec rejected{};
    return !MakePaExecPayloadSpec(
        shape.task_id, shape.kind,
        static_cast<int32_t>(shape.function_id),
        /*function_address=*/0, worker, args, context, rejected
    );
}

bool ClaimBindingRejectsMalformedShape(
    const CaseShape &shape, WorkerState &worker,
    const ExecPayloadSpec &valid_spec,
    const PaExecPayloadSource &source, ShapeField field
) {
    ExecPayloadSpec malformed = valid_spec;
    SetMalformedSpecCount(malformed, field);
    SharedExecCell cell{};
    SharedExecFatalControl fatal{};
    ExecutionToken token{};
    ResetExecutionToken(token);
    const uint32_t owner = static_cast<uint32_t>(worker.core_idx);
    const ExecBuildResult build =
        BuildAndPublishExecPayload<AdapterTestOps>(
            cell, owner, malformed, source, fatal
        );
    if (build != ExecBuildResult::Published) {
        return false;
    }
    const ExecClaimResult claim =
        ClaimAndBindExecPayload<AdapterTestOps>(
            cell, shape.task_id, owner, shape.engine,
            token, fatal
        );
    return claim == ExecClaimResult::Claimed &&
           !BindPaExecutionTokenDispatchAfterClaim(token, worker);
}

void CorruptTokenShape(
    ExecutionToken &token, const CaseShape &shape, ShapeField field
) {
    constexpr uint64_t kFieldMask = UINT64_C(0xFFFF);
    uint32_t shift = 0;
    uint16_t expected = shape.tensor_count;
    if (field == ShapeField::Scalar) {
        shift = 16;
        expected = shape.scalar_count;
    } else if (field == ShapeField::Fanin) {
        shift = 32;
        expected = shape.fanin_count;
    }
    const uint64_t mask = kFieldMask << shift;
    token.payload.words[4] =
        (token.payload.words[4] & ~mask) |
        (static_cast<uint64_t>(MalformedCount(expected)) << shift);
}

bool FinalValidatorRejectsMalformedShape(
    const CaseShape &shape, WorkerState &worker,
    const ExecPayloadSpec &valid_spec,
    const PaExecPayloadSource &source, ShapeField field
) {
    SharedExecCell cell{};
    SharedExecFatalControl fatal{};
    ExecutionToken token{};
    ResetExecutionToken(token);
    const uint32_t owner = static_cast<uint32_t>(worker.core_idx);
    const ExecBuildResult build =
        BuildAndPublishExecPayload<AdapterTestOps>(
            cell, owner, valid_spec, source, fatal
        );
    if (build != ExecBuildResult::Published) {
        return false;
    }
    const ExecClaimResult claim =
        ClaimAndBindExecPayload<AdapterTestOps>(
            cell, shape.task_id, owner, shape.engine,
            token, fatal
        );
    if (claim != ExecClaimResult::Claimed ||
        !BindPaExecutionTokenDispatchAfterClaim(token, worker)) {
        return false;
    }
    CorruptTokenShape(token, shape, field);
    token.control.phase = ExecTokenPhase::EngineInflight;
    return !ValidatePaExecutionTokenDispatch(
        token, worker, shape.kind
    );
}

bool RunCase(SchedulerState &state, const CaseShape &shape) {
    const uint32_t build_owner = shape.engine == ExecEngineClass::Aic
        ? kBuildOwnerAic : kBuildOwnerAiv;
    uint32_t execute_owner = kExecUnboundOwner;
    const bool mapped = SelectTestExecuteOwner(
        shape.task_id, build_owner, shape.engine, execute_owner
    );
    Check(mapped, shape.kind, "test-only different-core owner selection");
    if (!mapped) {
        return false;
    }
    WorkerState &builder = state.workers[build_owner];
    WorkerState &executor = state.workers[execute_owner];
    ConfigureWorkerIdentity(builder, build_owner);
    ConfigureWorkerIdentity(executor, execute_owner);
    const uint64_t completion_vend =
        UINT64_C(0x100000) +
        static_cast<uint64_t>(shape.task_id) * kOutputAlignment;
    builder.heap_next = completion_vend;
    executor.heap_next = completion_vend + kOutputAlignment;

    CaseFixture fixture{};
    BuildCaseArgs(shape, state, builder, fixture);
    PaExecShape contract{};
    Check(
        ResolvePaExecShape(shape.kind, contract) &&
            contract.tensor_count == shape.tensor_count &&
            contract.scalar_count == shape.scalar_count &&
            contract.fanin_count == shape.fanin_count &&
            !fixture.args.has_error &&
            fixture.args.tensor_count == shape.tensor_count &&
            fixture.args.scalar_count == shape.scalar_count &&
            fixture.context.fanin_count == shape.fanin_count,
        shape.kind, "TaskArgs/SubmitContext shape"
    );

    SharedExecCell cell{};
    SharedExecFatalControl fatal{};
    ExecutionToken token{};
    ResetExecutionToken(token);
    AdapterTestOps::ResetCounters();

    ExecPayloadSpec spec{};
    PaExecPayloadSource source{};
    const bool prepared =
        PreparePaExecPublicationAfterFanin<AdapterTestOps>(
            state, builder, fixture.args, fixture.context,
            shape.task_id, shape.kind,
            static_cast<int32_t>(shape.function_id),
            /*function_address=*/0, spec, source
        );
    Check(prepared, shape.kind, "adapter source preparation");
    if (!prepared) {
        return false;
    }

    for (ShapeField field : {
             ShapeField::Tensor, ShapeField::Scalar,
             ShapeField::Fanin}) {
        Check(
            MakeRejectsMalformedShape(
                shape, builder, fixture.args, fixture.context, field
            ),
            shape.kind,
            MakeRejectMessage(field)
        );
        Check(
            ClaimBindingRejectsMalformedShape(
                shape, builder, spec, source, field
            ),
            shape.kind,
            BindRejectMessage(field)
        );
        Check(
            FinalValidatorRejectsMalformedShape(
                shape, builder, spec, source, field
            ),
            shape.kind,
            FinalRejectMessage(field)
        );
    }
    AdapterTestOps::ResetCounters();

    ExecPayloadLayout expected_layout{};
    Check(
        ValidateExecPayloadSpec(spec, expected_layout) &&
            spec.completion_vend == completion_vend &&
            expected_layout.payload_bytes == shape.payload_bytes &&
            expected_layout.payload_lines == shape.payload_lines,
        shape.kind, "payload spec and exact line shape"
    );

    const ExecBuildResult build =
        BuildAndPublishExecPayload<AdapterTestOps>(
            cell, build_owner, spec, source, fatal
        );
    Check(
        build == ExecBuildResult::Published &&
            AdapterTestOps::flush_calls == 1 &&
            !ExecFatalPublished<AdapterTestOps>(fatal),
        shape.kind, "Build publishes exactly one payload"
    );
    if (build != ExecBuildResult::Published) {
        return false;
    }

    const DecodedExecState built =
        DecodeExecState(cell.control.state);
    Check(
        built.valid && built.phase == ExecPhase::Built &&
            built.task_id == shape.task_id &&
            built.build_owner == build_owner &&
            built.execute_owner == kExecUnboundOwner &&
            built.engine_class == shape.engine &&
            built.payload_lines == shape.payload_lines,
        shape.kind, "BUILT control"
    );

    ExecPayloadLayout observed_layout{};
    Check(
        PayloadMatches(
            cell.payload, shape, fixture,
            completion_vend, observed_layout
        ),
        shape.kind, "cell carries the exact PA payload"
    );

    std::array<uint64_t, kExecMaxPayloadWords> snapshot{};
    SnapshotPayload(cell.payload, snapshot);
    PolluteBuilderSources(
        state, builder, shape, fixture, source
    );
    Check(
        PayloadEqualsSnapshot(cell.payload, snapshot),
        shape.kind,
        "published cell is independent of polluted builder sources"
    );

    const ExecClaimResult claim =
        ClaimAndBindExecPayload<AdapterTestOps>(
            cell, shape.task_id, execute_owner, shape.engine,
            token, fatal
        );
    const bool bound =
        claim == ExecClaimResult::Claimed &&
        BindPaExecutionTokenDispatchAfterClaim(token, executor);
    Check(bound, shape.kind, "Claim and PA dispatch binding");
    if (!bound) {
        return false;
    }

    const DecodedExecState claimed =
        DecodeExecState(cell.control.state);
    Check(
        claimed.valid && claimed.phase == ExecPhase::Claimed &&
            claimed.task_id == shape.task_id &&
            claimed.build_owner == build_owner &&
            claimed.execute_owner == execute_owner &&
            claimed.build_owner != claimed.execute_owner &&
            claimed.engine_class == shape.engine,
        shape.kind, "CLAIMED retains selected different-core owners"
    );
    Check(
        TokenDispatchMatches(
            token, shape, fixture, executor, completion_vend
        ),
        shape.kind,
        "token payload, self pointers and PA contexts"
    );
    token.control.phase = ExecTokenPhase::EngineInflight;
    Check(
        ValidatePaExecutionTokenDispatch(token, executor, shape.kind),
        shape.kind, "valid exact-shape dispatch passes final validator"
    );
    token.control.phase = ExecTokenPhase::WaitingFanin;
    Check(
        PayloadEqualsSnapshot(cell.payload, snapshot),
        shape.kind, "Claim does not rewrite shared payload"
    );

    std::printf(
        "[PA_EXEC_ADAPTER] kind=%u task=%u tensors=%u scalars=%u "
        "fanin=%u bytes=%u lines=%u PASS\n",
        static_cast<uint32_t>(shape.kind), shape.task_id,
        shape.tensor_count, shape.scalar_count, shape.fanin_count,
        shape.payload_bytes, shape.payload_lines
    );
    return true;
}

}  // namespace

int main() {
    SchedulerState *state = MapSchedulerState();
    if (state == nullptr) {
        return 1;
    }

    bool all_cases_ran = TestExecuteCandidateEligibility();
    for (const CaseShape &shape : kCases) {
        all_cases_ran &= RunCase(*state, shape);
    }

    (void)munmap(state, sizeof(SchedulerState));
    if (!all_cases_ran || g_failures != 0) {
        std::fprintf(
            stderr, "[FAIL] PA exec adapter: %d checks failed\n",
            g_failures
        );
        return 1;
    }
    std::printf("[PASS] PA execution payload adapter\n");
    return 0;
}
