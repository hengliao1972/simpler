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

// Target identity for the transitional dist_engine translation unit.
//
// `__CPU_SIM` is defined for sim host, sim AICPU, and sim AICore targets, so it
// cannot identify the worker program by itself. The sim AICore CMake target
// defines `__SIM_AICORE__`; onboard AICore is identified by CCEC's
// `__CCE_AICORE__`.
#if defined(__CCE_AICORE__) || defined(__SIM_AICORE__)
#define DIST_AICORE_TARGET 1
#else
#define DIST_AICORE_TARGET 0
#endif

#if defined(__CCE_AICORE__)
#define DIST_ONBOARD_AICORE 1
#else
#define DIST_ONBOARD_AICORE 0
#endif
