/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: CANN-2.0
 */

#include "aicpu_plan_owner_abi.h"

#include <ctime>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "../../../common/aicpu_plan_protocol.h"
#include "pto_orchestration_api.h"

extern "C" PTO2OrchestrationConfig
aicpu_orchestration_config(const L2TaskArgs &args);
extern "C" void aicpu_orchestration_entry(const L2TaskArgs &args);

namespace {

using namespace pa_scheduler::aicpu_owner;

constexpr uint64_t kNanosecondsPerSecond = UINT64_C(1000000000);

uint64_t MonotonicNanoseconds()
{
    timespec timestamp{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp) != 0) return 0U;
    return static_cast<uint64_t>(timestamp.tv_sec) * kNanosecondsPerSecond +
           static_cast<uint64_t>(timestamp.tv_nsec);
}

void FullBarrier()
{
#if defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

void InstructionBarrier()
{
#if defined(__aarch64__)
    __asm__ volatile("isb" ::: "memory");
#endif
}

void InvalidateRegion(const void *address, size_t bytes)
{
#if defined(__aarch64__)
    if (address == nullptr || bytes == 0U) return;
    const uintptr_t begin =
        reinterpret_cast<uintptr_t>(address) & ~uintptr_t{63U};
    const uintptr_t end =
        (reinterpret_cast<uintptr_t>(address) + bytes + 63U) &
        ~uintptr_t{63U};
    for (uintptr_t line = begin; line < end; line += kCacheLineBytes) {
        __asm__ volatile("dc civac, %0" : : "r"(line) : "memory");
    }
    FullBarrier();
    InstructionBarrier();
#else
    (void)address;
    (void)bytes;
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
#endif
}

void CleanRegion(const void *address, size_t bytes)
{
#if defined(__aarch64__)
    if (address == nullptr || bytes == 0U) return;
    const uintptr_t begin =
        reinterpret_cast<uintptr_t>(address) & ~uintptr_t{63U};
    const uintptr_t end =
        (reinterpret_cast<uintptr_t>(address) + bytes + 63U) &
        ~uintptr_t{63U};
    for (uintptr_t line = begin; line < end; line += kCacheLineBytes) {
        __asm__ volatile("dc cvac, %0" : : "r"(line) : "memory");
    }
#else
    (void)address;
    (void)bytes;
    __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
}

bool IsZero(const void *address, size_t bytes)
{
    const auto *data = static_cast<const uint8_t *>(address);
    for (size_t index = 0U; index < bytes; ++index) {
        if (data[index] != 0U) return false;
    }
    return true;
}

bool HeaderValid(const OwnerRequestHeader &header)
{
    using namespace pa_scheduler::aicpu_plan;
    if (header.magic != kRequestMagic ||
        header.version != kRequestVersion ||
        header.request_bytes != sizeof(OwnerRequest) ||
        header.runtime_plan_control == 0U ||
        header.runtime_plan_cells == 0U ||
        header.context_lens == 0U || header.capacity == 0U ||
        header.capacity > kMaxRuntimeTasks || header.batches == 0U ||
        header.tensor_count == 0U ||
        header.tensor_count > kMaxTensorInputs ||
        header.scalar_count > kMaxScalarInputs ||
        header.context_tensor_index >= header.tensor_count ||
        (header.runtime_plan_control %
             pa_scheduler::aicpu_owner::kAtomicIsolationBytes) != 0U ||
        (header.runtime_plan_cells %
             pa_scheduler::aicpu_owner::kAtomicIsolationBytes) != 0U ||
        !IsZero(header.reserved, sizeof(header.reserved))) {
        return false;
    }
    return true;
}

bool TensorMetadataValid(const TensorMetadata &metadata)
{
    if (metadata.buffer_addr == 0U || metadata.buffer_bytes == 0U ||
        metadata.owner_task_raw != UINT64_MAX || metadata.version != 0 ||
        metadata.ndims == 0U || metadata.ndims > kMaxTensorDims ||
        metadata.dtype >= static_cast<uint8_t>(DataType::DATA_TYPE_NUM) ||
        metadata.manual_dep > 1U || metadata.is_contiguous > 1U ||
        metadata.child_memory > 1U ||
        !IsZero(metadata.reserved, sizeof(metadata.reserved))) {
        return false;
    }
    uint64_t extent = 1U;
    uint64_t contiguous_stride = 1U;
    for (uint32_t reverse = 0U; reverse < metadata.ndims; ++reverse) {
        const uint32_t dim = metadata.ndims - 1U - reverse;
        if (metadata.shapes[dim] == 0U || metadata.strides[dim] == 0U) {
            return false;
        }
        const uint64_t term =
            static_cast<uint64_t>(metadata.shapes[dim] - 1U) *
            metadata.strides[dim];
        if (term > UINT64_MAX - extent) return false;
        extent += term;
        if (metadata.is_contiguous != 0U) {
            if (metadata.strides[dim] != contiguous_stride ||
                metadata.shapes[dim] > UINT64_MAX / contiguous_stride) {
                return false;
            }
            contiguous_stride *= metadata.shapes[dim];
        }
    }
    for (uint32_t dim = metadata.ndims; dim < kMaxTensorDims; ++dim) {
        if (metadata.shapes[dim] != 0U || metadata.strides[dim] != 0U) {
            return false;
        }
    }
    if (metadata.extent_elem_cache != extent) return false;
    const uint64_t element_bytes =
        get_element_size(static_cast<DataType>(metadata.dtype));
    if (element_bytes == 0U ||
        metadata.start_offset > UINT64_MAX - extent ||
        metadata.start_offset + extent > UINT64_MAX / element_bytes ||
        (metadata.start_offset + extent) * element_bytes >
            metadata.buffer_bytes) {
        return false;
    }
    return true;
}

Tensor DecodeTensor(const TensorMetadata &metadata)
{
    Tensor tensor{};
    tensor.buffer.addr = metadata.buffer_addr;
    tensor.buffer.size = metadata.buffer_bytes;
    tensor.owner_task_id.raw = metadata.owner_task_raw;
    tensor.start_offset = metadata.start_offset;
    tensor.extent_elem_cache = metadata.extent_elem_cache;
    tensor.version = metadata.version;
    tensor.ndims = metadata.ndims;
    tensor.dtype = static_cast<DataType>(metadata.dtype);
    tensor.manual_dep = metadata.manual_dep != 0U;
    tensor.is_contiguous = metadata.is_contiguous != 0U;
    tensor.child_memory = metadata.child_memory;
    std::memcpy(tensor.shapes, metadata.shapes, sizeof(tensor.shapes));
    std::memcpy(tensor.strides, metadata.strides, sizeof(tensor.strides));
    std::memset(tensor._pad_cl2, 0, sizeof(tensor._pad_cl2));
    return tensor;
}

void PublishFatalIfAddressable(const OwnerRequestHeader &header)
{
    using namespace pa_scheduler::aicpu_plan;
    if (header.runtime_plan_control == 0U ||
        (header.runtime_plan_control %
             pa_scheduler::aicpu_owner::kAtomicIsolationBytes) != 0U) {
        return;
    }
    auto *control = reinterpret_cast<RuntimePlanControl *>(
        static_cast<uintptr_t>(header.runtime_plan_control));
    control->fatal.value = 1;
    CleanRegion(
        const_cast<const int64_t *>(&control->fatal.value),
        sizeof(control->fatal.value));
    FullBarrier();
    InstructionBarrier();
}

void PublishResult(
    OwnerRequest *request, OwnerStatus status,
    const AicpuPlanBackendResult &backend, uint64_t begin_ns
)
{
    OwnerResult result{};
    result.magic = kResultMagic;
    result.version = kRequestVersion;
    result.status = static_cast<int32_t>(status);
    result.backend = backend;
    result.begin_ns = begin_ns;
    result.end_ns = MonotonicNanoseconds();
    request->result = result;
    CleanRegion(&request->result, sizeof(request->result));
    FullBarrier();
    InstructionBarrier();
}

OwnerStatus RunOwner(OwnerRequest *request, AicpuPlanBackendResult &backend)
{
    const OwnerRequestHeader header = request->header;
    if (!HeaderValid(header)) return OwnerStatus::BadRequest;

    InvalidateRegion(
        request->tensors,
        static_cast<size_t>(header.tensor_count) * sizeof(TensorMetadata));
    InvalidateRegion(
        request->scalars,
        static_cast<size_t>(header.scalar_count) * sizeof(uint64_t));
    InvalidateRegion(
        reinterpret_cast<const void *>(
            static_cast<uintptr_t>(header.context_lens)),
        static_cast<size_t>(header.batches) * sizeof(int32_t));

    ChipStorageTaskArgs storage;
    storage.clear();
    for (uint32_t index = 0U; index < header.tensor_count; ++index) {
        const TensorMetadata &metadata = request->tensors[index];
        if (!TensorMetadataValid(metadata)) {
            return OwnerStatus::BadTensorMetadata;
        }
        storage.add_tensor(DecodeTensor(metadata));
    }
    for (uint32_t index = 0U; index < header.scalar_count; ++index) {
        storage.add_scalar(request->scalars[index]);
    }

    const TensorMetadata &context =
        request->tensors[header.context_tensor_index];
    if (context.buffer_addr != header.context_lens || context.ndims != 1U ||
        context.shapes[0] != header.batches ||
        context.dtype != static_cast<uint8_t>(DataType::INT32)) {
        return OwnerStatus::BadTensorMetadata;
    }

    L2TaskArgs args;
    args.create_from_chip_args(storage);
    const PTO2OrchestrationConfig config = aicpu_orchestration_config(args);
    if (config.expected_arg_count !=
            static_cast<int32_t>(header.tensor_count + header.scalar_count) ||
        config.expected_arg_count != 7) {
        return OwnerStatus::BadOrchestrationConfig;
    }

    const AicpuPlanBackendConfig backend_config{
        reinterpret_cast<void *>(
            static_cast<uintptr_t>(header.runtime_plan_control)),
        reinterpret_cast<void *>(
            static_cast<uintptr_t>(header.runtime_plan_cells)),
        header.capacity,
        0U,
    };
    if (aicpu_plan_backend_bind(&backend_config) != 0) {
        backend = aicpu_plan_backend_result();
        return OwnerStatus::BackendBindFailed;
    }
    aicpu_orchestration_entry(args);
    if (aicpu_plan_backend_close() != 0) {
        backend = aicpu_plan_backend_result();
        return OwnerStatus::BackendCloseFailed;
    }
    backend = aicpu_plan_backend_result();
    if (backend.status != 0) return OwnerStatus::BackendReportedFailure;
    return OwnerStatus::Ok;
}

}  // namespace

extern "C" __attribute__((visibility("default")))
int plan_protocol_aicpu_exec(void *argument)
{
    using namespace pa_scheduler::aicpu_owner;
    if (argument == nullptr) return 0;
    const auto *kernel_args = static_cast<const OwnerKernelArgs *>(argument);
    if (kernel_args->request_device == 0U ||
        kernel_args->command != kOwnerCommandRun ||
        kernel_args->version != kRequestVersion) {
        return 0;
    }

    auto *request = reinterpret_cast<OwnerRequest *>(
        static_cast<uintptr_t>(kernel_args->request_device));
    InvalidateRegion(&request->header, sizeof(request->header));
    const uint64_t begin_ns = MonotonicNanoseconds();
    AicpuPlanBackendResult backend{};
    const OwnerStatus status = RunOwner(request, backend);
    if (status != OwnerStatus::Ok) {
        PublishFatalIfAddressable(request->header);
    }
    PublishResult(request, status, backend, begin_ns);
    return 0;
}
