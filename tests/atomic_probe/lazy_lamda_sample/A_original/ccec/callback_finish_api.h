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
#ifndef TESTS_ATOMIC_PROBE_LAZY_LAMDA_SAMPLE_CCEC_CALLBACK_FINISH_API_H
#define TESTS_ATOMIC_PROBE_LAZY_LAMDA_SAMPLE_CCEC_CALLBACK_FINISH_API_H

#if !defined(PA_LAZY_SAMPLE_SPLIT_FINISH)
#error "callback_finish_api.h is only valid for split-finish artifacts"
#endif

// The runtime TU owns the role-specific block-local object.  The caller only
// imports it; no weak/generic state symbol is allowed in the mixed AIC/AIV ELF.
extern "C" {
#if defined(PA_BUILD_AIC)
[[block_local]] extern pa_scheduler::LazySampleSplitRuntimeState pa_scheduler_lazy_sample_callback_state_aic;
__aicore__ uint32_t
pa_scheduler_lazy_sample_callback_finish_aic(const pa_scheduler::LazySampleCallbackTicket *ticket, const pa_scheduler::TaskArgs *args);
#elif defined(PA_BUILD_AIV)
[[block_local]] extern pa_scheduler::LazySampleSplitRuntimeState pa_scheduler_lazy_sample_callback_state_aiv;
__aicore__ uint32_t
pa_scheduler_lazy_sample_callback_finish_aiv(const pa_scheduler::LazySampleCallbackTicket *ticket, const pa_scheduler::TaskArgs *args);
#else
#error "Compile split finish with PA_BUILD_AIC or PA_BUILD_AIV"
#endif
}

#endif  // TESTS_ATOMIC_PROBE_LAZY_LAMDA_SAMPLE_CCEC_CALLBACK_FINISH_API_H
