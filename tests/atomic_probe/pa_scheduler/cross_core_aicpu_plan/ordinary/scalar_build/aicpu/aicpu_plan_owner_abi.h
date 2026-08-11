/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: CANN-2.0
 */

#ifndef PA_SCHEDULER_AICPU_PLAN_OWNER_ABI_H
#define PA_SCHEDULER_AICPU_PLAN_OWNER_ABI_H

#include <cstddef>
#include <cstdint>

#include "aicpu_clock_correlation_abi.h"
#include "aicpu_plan_backend_abi.h"

namespace pa_scheduler::aicpu_owner {

constexpr uint64_t kRequestMagic = UINT64_C(0x5041504c414e5251);
constexpr uint64_t kResultMagic = UINT64_C(0x5041504c414e5253);
constexpr uint32_t kRequestVersion = 3U;
constexpr uint32_t kOwnerCommandRun = aicpu_clock::kOwnerCommandRun;
constexpr uint32_t kOwnerCommandClockCorrelation =
    aicpu_clock::kOwnerCommandClockCorrelation;
constexpr uint32_t kCacheLineBytes = 64U;
constexpr uint32_t kAtomicIsolationBytes = 128U;
constexpr uint32_t kMaxTensorInputs = 32U;
constexpr uint32_t kMaxScalarInputs = 16U;
constexpr uint32_t kMaxTensorDims = 5U;

enum class OwnerStatus : int32_t {
    Ok = 0,
    BadKernelArgs = 1,
    BadRequest = 2,
    BadTensorMetadata = 3,
    BadOrchestrationConfig = 4,
    BackendBindFailed = 5,
    BackendCloseFailed = 6,
    BackendReportedFailure = 7,
    TraceFailed = 8,
    ClockReadFailed = 9,
};

// Host/AICPU 之间只传无指针的通用 Tensor 元数据。buffer_addr 是被描述的
// 设备数据地址，不是 Host C++ 对象地址；owner_task_raw 必须为 invalid。
struct alignas(kAtomicIsolationBytes) TensorMetadata {
    uint64_t buffer_addr;
    uint64_t buffer_bytes;
    uint64_t start_offset;
    uint64_t extent_elem_cache;
    uint64_t owner_task_raw;
    int32_t version;
    uint32_t ndims;
    uint8_t dtype;
    uint8_t manual_dep;
    uint8_t is_contiguous;
    uint8_t child_memory;
    uint32_t shapes[kMaxTensorDims];
    uint32_t strides[kMaxTensorDims];
    uint8_t reserved[36];
};

// Plan 地址、容量和 orchestration 的通用输入是 owner 的全部输入。
// 这里没有 task_count、task kind、function id 或 dispatch plan。
struct alignas(kAtomicIsolationBytes) OwnerRequestHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t request_bytes;
    uint64_t runtime_plan_control;
    uint64_t runtime_plan_cells;
    uint64_t context_lens;
    uint32_t capacity;
    uint32_t batches;
    uint32_t tensor_count;
    uint32_t scalar_count;
    uint32_t context_tensor_index;
    uint32_t reserved0;
    uint64_t task_trace_records;
    uint32_t task_trace_capacity;
    uint32_t task_trace_record_bytes;
    uint64_t operation_trace_records;
    uint32_t operation_trace_capacity;
    uint32_t operation_trace_record_bytes;
    uint32_t reserved[8];
};

struct alignas(kAtomicIsolationBytes) OwnerResult {
    uint64_t magic;
    uint32_t version;
    int32_t status;
    AicpuPlanBackendResult backend;
    uint64_t begin_ns;
    uint64_t end_ns;
    uint64_t input_ready_ns;
    uint64_t backend_bound_ns;
    uint64_t orchestration_end_ns;
    uint64_t backend_closed_ns;
};

struct alignas(kAtomicIsolationBytes) OwnerRequest {
    OwnerRequestHeader header;
    TensorMetadata tensors[kMaxTensorInputs];
    uint64_t scalars[kMaxScalarInputs];
    OwnerResult result;
};

// rtsLaunchCpuKernel 只搬运这个很小的 launch ABI；大 request 始终位于 GM。
struct OwnerKernelArgs {
    uint64_t request_device;
    uint32_t command;
    uint32_t version;
};

static_assert(sizeof(TensorMetadata) == 128U, "AICPU Tensor metadata ABI changed");
static_assert(alignof(TensorMetadata) == 128U, "AICPU Tensor metadata alignment changed");
static_assert(sizeof(OwnerRequestHeader) == 128U, "AICPU request header ABI changed");
static_assert(sizeof(OwnerResult) == 128U, "AICPU result ABI changed");
static_assert(sizeof(OwnerRequest) == 4480U, "AICPU request ABI changed");
static_assert(alignof(OwnerRequest) == 128U, "AICPU request alignment changed");
static_assert(offsetof(OwnerRequest, tensors) == 128U, "AICPU tensor offset changed");
static_assert(offsetof(OwnerRequest, scalars) == 4224U, "AICPU scalar offset changed");
static_assert(offsetof(OwnerRequest, result) == 4352U, "AICPU result offset changed");
static_assert(sizeof(OwnerKernelArgs) == 16U, "AICPU kernel args ABI changed");

}  // namespace pa_scheduler::aicpu_owner

#endif  // PA_SCHEDULER_AICPU_PLAN_OWNER_ABI_H
