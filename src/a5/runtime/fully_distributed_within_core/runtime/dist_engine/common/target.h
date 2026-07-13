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

#if defined(__CPU_SIM)
#define DIST_SIM_HOST_CLOCK 1
#else
#define DIST_SIM_HOST_CLOCK 0
#endif

#ifndef PTO2_PROFILING
#define PTO2_PROFILING 1
#endif

#if defined(PTO2_PROFILING) && PTO2_PROFILING
#define DIST_TRACE_ENABLED 1
#else
#define DIST_TRACE_ENABLED 0
#endif

#if defined(__CCE_AICORE__)
#define DIST_API_ATTR __attribute__((weak))
#else
#define DIST_API_ATTR
#endif
