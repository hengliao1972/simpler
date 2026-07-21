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

#include <cstdint>

// submit-pmu-none 只需要当前物理子核的 PMU MMIO 基址。独立接口避免把
// generic PMU ring、普通泳道和总 profiling flag 带入诊断 ELF。
extern __aicore__ void set_fdwic_submit_pmu_reg_base(uint64_t reg_base);
extern __aicore__ uint64_t get_fdwic_submit_pmu_reg_base();
