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

#define PA_DEVICE inline
#define PA_GM
#include "../adapter/pa_simt_route_policy.h"

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

namespace plan = pa_scheduler::aicpu_plan;
namespace adapter = pa_scheduler::aicpu_plan_simt::adapter;
using pa_scheduler::TaskKind;

struct Case {
    TaskKind kind;
    plan::EngineClass engine;
    uint32_t function_id;
    uint32_t same_engine_wrong_function_id;
};

constexpr std::array<Case, 5> kCases{{
    {TaskKind::Alloc, plan::EngineClass::MetadataOnly,
     plan::kInvalidFunctionId, 0U},
    {TaskKind::Qk, plan::EngineClass::Aic, 0U, 2U},
    {TaskKind::Sf, plan::EngineClass::Aiv, 1U, 3U},
    {TaskKind::Pv, plan::EngineClass::Aic, 2U, 0U},
    {TaskKind::Up, plan::EngineClass::Aiv, 3U, 1U},
}};

uint8_t AdapterFlags(TaskKind kind)
{
    return static_cast<uint8_t>(
        pa_scheduler::kSharedPaTicketMetaPresent |
        static_cast<uint8_t>(kind)
    );
}

}  // namespace

int main()
{
    adapter::PaSimtRoutePolicy policy{};
    bool ok = true;
    for (const Case &test : kCases) {
        const uint8_t flags = AdapterFlags(test.kind);
        const uint32_t task_id = pa_scheduler::SharedPaTaskOffset(
            test.kind, 0U
        );
        ok &= adapter::PaSimtFunctionIdMatchesAdapterKind(
            flags, test.function_id
        );
        ok &= !adapter::PaSimtFunctionIdMatchesAdapterKind(
            flags, test.same_engine_wrong_function_id
        );
        ok &= policy.Validate(
            task_id, flags, 0U, test.function_id, test.engine
        );
        // 关键负向：engine 仍合法且不变，仅把 QK/PV 或 SF/UP 的
        // function 互换，也必须在 adapter 边界拒绝。
        ok &= !policy.Validate(
            task_id, flags, 0U,
            test.same_engine_wrong_function_id, test.engine
        );
    }

    const uint8_t missing_present =
        static_cast<uint8_t>(TaskKind::Qk);
    const uint8_t invalid_kind = static_cast<uint8_t>(
        pa_scheduler::kSharedPaTicketMetaPresent |
        static_cast<uint8_t>(TaskKind::Count)
    );
    ok &= !adapter::PaSimtFunctionIdMatchesAdapterKind(
        missing_present, 0U
    );
    ok &= !adapter::PaSimtFunctionIdMatchesAdapterKind(
        invalid_kind, 0U
    );

    std::printf(
        "[SIMT_ROUTE] exact kind/function and same-engine negative cases %s\n",
        ok ? "PASS" : "FAIL"
    );
    return ok ? 0 : 1;
}
