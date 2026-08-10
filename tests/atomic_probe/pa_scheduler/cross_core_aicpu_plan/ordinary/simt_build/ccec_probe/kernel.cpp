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

// 该文件只证明“公共 Runtime Plan ABI v2 可以被真实 A5 SIMT VF 直接消费”。
// 它不生成 task，不复制第二份 request ABI，也不包含 PA task kind/固定公式。

#include <pto/common/kernel_meta.hpp>

#include "cce_aicore_intrinsics.h"
#include "simt_api/asc_simt.h"

#include "../common/simt_plan_build_protocol.h"

#include <cstddef>
#include <cstdint>

PTO_SYNCALL_MIX_AIC_KERNEL_META(aicpu_plan_simt_v2_probe_0_mix_aiv, 1, 2);

namespace {

namespace plan = pa_scheduler::aicpu_plan;
namespace simt_contract = pa_scheduler::aicpu_plan_simt;

constexpr uint32_t kReportWords = 12U;
constexpr uint64_t kReportMagic = 0x4135504C414E5632ULL;  // "A5PLANV2"

constexpr uint64_t kStatusCapacityValid = uint64_t{1} << 0U;
constexpr uint64_t kStatusPlanClosedPastTask = uint64_t{1} << 1U;
constexpr uint64_t kStatusFirstControlValid = uint64_t{1} << 2U;
constexpr uint64_t kStatusSecondControlStable = uint64_t{1} << 3U;
constexpr uint64_t kStatusHeaderTaskValid = uint64_t{1} << 4U;
constexpr uint64_t kStatusAbiV2 = uint64_t{1} << 5U;
constexpr uint64_t kStatusPayloadLinesValid = uint64_t{1} << 6U;
constexpr uint64_t kStatusHeaderFieldsValid = uint64_t{1} << 7U;

static_assert(simt_contract::kBuilderThreads == 128U, "SIMT Plan probe launch width changed");
static_assert(simt_contract::kWarpSize == 32U, "SIMT Plan probe warp width changed");
static_assert(simt_contract::kBuilderLeaders == 4U, "SIMT Plan probe must retain four warp leaders");
static_assert(sizeof(plan::RuntimeTaskPlanCell) == 4608U, "canonical PlanCell ABI changed");
static_assert(offsetof(plan::RuntimeTaskPlanCell, payload) == 128U, "canonical payload offset changed");
static_assert(sizeof(plan::RuntimeTaskPlanHeader) == 64U, "canonical Plan header changed");
static_assert(plan::kRuntimePlanAbiVersion == 2U, "this probe is for Plan ABI v2");

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtAtomicObserve(__gm__ volatile int64_t *address)
{
    return asc_atomic_add(
        reinterpret_cast<__gm__ uint64_t *>(
            const_cast<__gm__ int64_t *>(address)
        ),
        static_cast<uint64_t>(0U)
    );
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtLoadPayloadWord(__gm__ volatile uint64_t *address)
{
    return asc_ldcg(
        const_cast<__gm__ uint64_t *>(address)
    );
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline
plan::DecodedPlanCellControl SimtDecodePlanCellControl(uint64_t raw)
{
    plan::DecodedPlanCellControl decoded{
        static_cast<plan::PlanCellPhase>(raw & plan::kPlanPhaseMask),
        static_cast<uint32_t>(
            (raw >> plan::kPlanPayloadLinesShift) &
            plan::kPlanPayloadLinesMask
        ),
        static_cast<uint32_t>(
            (raw >> plan::kPlanTaskIdShift) & plan::kPlanTaskIdMask
        ),
        false,
    };
    if ((raw & ~plan::kPlanKnownMask) != 0U) {
        return decoded;
    }
    if (decoded.phase == plan::PlanCellPhase::Empty) {
        decoded.valid = raw == 0U;
    } else if (decoded.phase == plan::PlanCellPhase::Published) {
        decoded.valid = decoded.payload_lines >= 1U &&
                        decoded.payload_lines <= plan::kMaxPlanPayloadLines;
    }
    return decoded;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline void
PublishLeaderReport(__gm__ uint64_t *reports, uint32_t leader, uint32_t word, uint64_t value)
{
    asc_stcg(reports + leader * kReportWords + word, value);
}

static __simt_vf__ __aicore__ LAUNCH_BOUND(128) void ConsumeCanonicalPlanV2(
    __gm__ plan::RuntimePlanControl *control,
    __gm__ plan::RuntimeTaskPlanCell *cells,
    uint32_t capacity,
    uint32_t first_task_id,
    __gm__ uint64_t *leader_reports
)
{
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    const uint32_t warp = thread / simt_contract::kWarpSize;
    const uint32_t lane = thread % simt_contract::kWarpSize;
    if (thread >= simt_contract::kBuilderThreads || lane != 0U ||
        warp >= simt_contract::kBuilderLeaders) {
        return;
    }

    const uint32_t task_id = first_task_id + warp;
    uint64_t status = 0U;
    const uint64_t closed_task_count = SimtAtomicObserve(
        &control->closed_task_count.value
    );
    if (task_id < capacity) {
        status |= kStatusCapacityValid;
    }
    if (closed_task_count != static_cast<uint64_t>(plan::kPlanOpenTaskCount) &&
        static_cast<uint64_t>(task_id) < closed_task_count) {
        status |= kStatusPlanClosedPastTask;
    }

    uint64_t first_control = 0U;
    uint64_t second_control = 0U;
    uint64_t header_word0 = 0U;
    uint64_t header_word1 = 0U;
    uint64_t header_word2 = 0U;
    uint64_t header_word7 = 0U;
    uint32_t published_lines = 0U;
    uint32_t header_task_id = UINT32_MAX;
    uint32_t function_id = plan::kInvalidFunctionId;
    uint32_t abi_version = 0U;
    uint32_t expected_lines = 0U;

    if (task_id < capacity) {
        __gm__ plan::RuntimeTaskPlanCell *cell = cells + task_id;
        first_control = SimtAtomicObserve(&cell->control.value);
        const plan::DecodedPlanCellControl decoded =
            SimtDecodePlanCellControl(first_control);
        published_lines = decoded.payload_lines;
        if (decoded.valid && decoded.phase == plan::PlanCellPhase::Published &&
            decoded.task_id == task_id) {
            status |= kStatusFirstControlValid;

            __gm__ uint8_t *payload_bytes = reinterpret_cast<__gm__ uint8_t *>(
                const_cast<__gm__ plan::RuntimeTaskPlanStorage *>(&cell->payload)
            );
            for (uint32_t line = 0U; line < published_lines; ++line) {
                asc_dcci_single(static_cast<__gm__ void *>(
                    payload_bytes + line * plan::kPlanCacheLineBytes
                ));
            }
            // 这里仅锁定 CCEC 能生成逐行 DCCI 与 SIMT fence。它没有经过
            // A5 动态可见性验证，不能据此宣称该组合已经等价于完整 acquire
            // 合同；真机协议仍必须单独验证 control/payload 的跨核时序。
            asc_threadfence();

            second_control = SimtAtomicObserve(&cell->control.value);
            if (second_control == first_control) {
                status |= kStatusSecondControlStable;
            }

            __gm__ volatile uint64_t *payload_words = cell->payload.words;
            header_word0 = SimtLoadPayloadWord(payload_words + 0U);
            header_word1 = SimtLoadPayloadWord(payload_words + 1U);
            header_word2 = SimtLoadPayloadWord(payload_words + 2U);
            header_word7 = SimtLoadPayloadWord(payload_words + 7U);

            header_task_id = static_cast<uint32_t>(header_word0);
            function_id = static_cast<uint32_t>(header_word0 >> 32U);
            abi_version = static_cast<uint32_t>(header_word7 >> 32U);
            const uint32_t tensor_count = static_cast<uint32_t>(header_word1 & 0xFFFFU);
            const uint32_t scalar_count = static_cast<uint32_t>((header_word1 >> 16U) & 0xFFFFU);
            const uint32_t explicit_dep_count = static_cast<uint32_t>((header_word1 >> 32U) & 0xFFFFU);
            const uint32_t output_count = static_cast<uint32_t>(header_word1 >> 48U);
            const uint32_t payload_words_count =
                plan::kPlanHeaderWords +
                tensor_count * plan::kTensorCanonicalWords +
                scalar_count + explicit_dep_count;
            expected_lines =
                (payload_words_count * sizeof(uint64_t) +
                 plan::kPlanCacheLineBytes - 1U) /
                plan::kPlanCacheLineBytes;

            if (header_task_id == task_id) {
                status |= kStatusHeaderTaskValid;
            }
            if (abi_version == plan::kRuntimePlanAbiVersion) {
                status |= kStatusAbiV2;
            }
            if (expected_lines == published_lines && published_lines >= 1U &&
                published_lines <= plan::kMaxPlanPayloadLines) {
                status |= kStatusPayloadLinesValid;
            }

            const uint32_t engine_class = static_cast<uint32_t>(header_word2 & 0xFFU);
            const uint32_t require_sync_start = static_cast<uint32_t>((header_word2 >> 32U) & 0xFFU);
            const bool metadata_function_valid =
                engine_class == static_cast<uint32_t>(plan::EngineClass::MetadataOnly)
                    ? function_id == plan::kInvalidFunctionId
                    : function_id != plan::kInvalidFunctionId;
            if (tensor_count <= plan::kMaxTaskTensors &&
                scalar_count <= plan::kMaxTaskScalars &&
                explicit_dep_count <= plan::kMaxExplicitDependencies &&
                output_count <= tensor_count &&
                engine_class <= static_cast<uint32_t>(plan::EngineClass::Aiv) &&
                require_sync_start <= 1U && metadata_function_valid) {
                status |= kStatusHeaderFieldsValid;
            }
        }
    }

    PublishLeaderReport(leader_reports, warp, 1U, static_cast<uint64_t>(warp));
    PublishLeaderReport(leader_reports, warp, 2U, static_cast<uint64_t>(task_id));
    PublishLeaderReport(leader_reports, warp, 3U, closed_task_count);
    PublishLeaderReport(leader_reports, warp, 4U, first_control);
    PublishLeaderReport(leader_reports, warp, 5U, second_control);
    PublishLeaderReport(
        leader_reports, warp, 6U,
        static_cast<uint64_t>(published_lines) |
            (static_cast<uint64_t>(expected_lines) << 32U)
    );
    PublishLeaderReport(
        leader_reports, warp, 7U,
        static_cast<uint64_t>(header_task_id) |
            (static_cast<uint64_t>(function_id) << 32U)
    );
    PublishLeaderReport(
        leader_reports, warp, 8U,
        static_cast<uint64_t>(abi_version) |
            (static_cast<uint64_t>(plan::kRuntimePlanAbiVersion) << 32U)
    );
    PublishLeaderReport(leader_reports, warp, 9U, header_word1);
    PublishLeaderReport(leader_reports, warp, 10U, header_word2);
    PublishLeaderReport(leader_reports, warp, 11U, status);
    asc_threadfence();
    PublishLeaderReport(leader_reports, warp, 0U, kReportMagic);
}

// 只用于让 mixed AIV metadata 明确保留 SIMD_SIMT_MIX_VF=4。运行期正常
// 容量不可能为 UINT32_MAX，因此 probe 不会执行这个 metadata anchor。
static __simd_vf__ __aicore__ void SimdMetadataAnchor(__ubuf__ uint32_t *scratch)
{
    scratch[0] = scratch[0] + 1U;
}

}  // namespace

extern "C" __global__ __aicore__ void aicpu_plan_simt_v2_probe_0_mix_aiv(
    __gm__ pa_scheduler::aicpu_plan::RuntimePlanControl *control,
    __gm__ pa_scheduler::aicpu_plan::RuntimeTaskPlanCell *cells,
    uint32_t capacity,
    uint32_t first_task_id,
    __gm__ uint64_t *leader_reports
)
{
    if (capacity == UINT32_MAX) {
        SimdMetadataAnchor(reinterpret_cast<__ubuf__ uint32_t *>(0));
    }

    cce::async_invoke<ConsumeCanonicalPlanV2>(
        cce::dim3{simt_contract::kBuilderThreads, 1U, 1U},
        control, cells, capacity, first_task_id, leader_reports
    );
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
}
