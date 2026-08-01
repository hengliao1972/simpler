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

#ifndef PA_SCHEDULER_CCEC_PMU_OWNER_MAIN_ABI_H_
#define PA_SCHEDULER_CCEC_PMU_OWNER_MAIN_ABI_H_

#include <cstddef>
#include <cstdint>

namespace pa_scheduler::pmu_owner {

// 主 aicpu_scheduler 的统一入口 simpler_aicpu_exec 根据该命令选择配置或
// 恢复。0 特意保留为 Invalid，避免零初始化参数意外改写 PMU 寄存器。
enum class PmuOwnerMainCommand : uint32_t {
    Invalid = 0U,
    Configure = 1U,
    Restore = 2U,
};

// 该结构逐字段复刻 A5 KernelArgs 的 152B ABI，但只使用固定宽度整数，因而
// 不依赖 Simpler 的 DeviceArgs/Runtime C++ 类型。runtime_args_device 指向
// PmuOwnerControl；command 位于原 enable_profiling_flag 的 offset 128。
// 其余字段保持为零，既满足主 aicpu_scheduler 固定布局，也不引入外部依赖。
struct PmuOwnerMainKernelArgs {
    uint64_t unused[5];                    // 0..39
    uint64_t device_args_device;           // 40，当前 owner 不使用
    uint64_t runtime_args_device;          // 48，PmuOwnerControl 的 GM 地址
    uint64_t register_bases_device;        // 56，当前 control 已内嵌基址，保持为零
    uint64_t dump_data_base;               // 64
    uint64_t l2_swimlane_data_base;        // 72
    uint64_t pmu_data_base;                // 80
    uint64_t dep_gen_data_base;            // 88
    uint64_t l2_swimlane_rotation_table;   // 96
    uint64_t aicore_pmu_ring_addrs;         // 104
    uint64_t scope_stats_data_base;         // 112
    uint32_t log_level;                     // 120
    uint32_t log_info_v;                    // 124
    uint32_t command;                       // 128，PmuOwnerMainCommand
    uint32_t reserved_alignment;            // 132
    uint64_t device_wall_data_base;         // 136
    uint32_t device_id;                     // 144
    uint32_t force_simt_anchor;              // 148
};

static_assert(sizeof(PmuOwnerMainCommand) == sizeof(uint32_t), "PMU owner command ABI changed");
static_assert(offsetof(PmuOwnerMainKernelArgs, device_args_device) == 40U, "device args offset changed");
static_assert(offsetof(PmuOwnerMainKernelArgs, runtime_args_device) == 48U, "control pointer offset changed");
static_assert(offsetof(PmuOwnerMainKernelArgs, register_bases_device) == 56U, "register pointer offset changed");
static_assert(offsetof(PmuOwnerMainKernelArgs, command) == 128U, "PMU owner command offset changed");
static_assert(offsetof(PmuOwnerMainKernelArgs, device_id) == 144U, "device id offset changed");
static_assert(sizeof(PmuOwnerMainKernelArgs) == 152U, "main aicpu_scheduler KernelArgs ABI changed");
static_assert(alignof(PmuOwnerMainKernelArgs) == 8U, "KernelArgs alignment changed");

inline PmuOwnerMainKernelArgs MakePmuOwnerMainKernelArgs(
    uint64_t control_device, PmuOwnerMainCommand command, uint32_t device_id
)
{
    PmuOwnerMainKernelArgs arguments{};
    arguments.runtime_args_device = control_device;
    arguments.command = static_cast<uint32_t>(command);
    arguments.device_id = device_id;
    return arguments;
}

}  // namespace pa_scheduler::pmu_owner

#endif  // PA_SCHEDULER_CCEC_PMU_OWNER_MAIN_ABI_H_
