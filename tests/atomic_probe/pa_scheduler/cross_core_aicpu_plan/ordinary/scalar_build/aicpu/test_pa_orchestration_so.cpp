/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: CANN-2.0
 */

#include "aicpu_plan_backend_abi.h"

#include <dlfcn.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "../../../common/aicpu_plan_protocol.h"
#include "../../../common/runtime_plan_pipeline_policy.h"
#include "pto_orchestration_api.h"

namespace {

using namespace pa_scheduler::aicpu_plan;

template <typename Function>
Function Load(void *handle, const char *name)
{
    dlerror();
    auto function = reinterpret_cast<Function>(dlsym(handle, name));
    const char *error = dlerror();
    if (error != nullptr || function == nullptr) {
        std::cerr << "dlsym(" << name << ") failed: "
                  << (error == nullptr ? "null" : error) << '\n';
        std::exit(2);
    }
    return function;
}

bool Check(bool condition, const char *message)
{
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

Tensor External(
    void *address, const uint32_t *shape, uint32_t rank,
    DataType dtype
)
{
    Tensor tensor{};
    tensor.buffer.addr = reinterpret_cast<uintptr_t>(address);
    tensor.owner_task_id = PTO2TaskId::invalid();
    tensor.start_offset = 0U;
    tensor.version = 0;
    tensor.ndims = rank;
    tensor.dtype = dtype;
    tensor.manual_dep = false;
    tensor.child_memory = 0U;
    uint64_t elements = 1U;
    for (uint32_t dim = rank; dim > 0U; --dim) {
        const uint32_t index = dim - 1U;
        tensor.shapes[index] = shape[index];
        tensor.strides[index] = static_cast<uint32_t>(elements);
        elements *= shape[index];
    }
    tensor.is_contiguous = true;
    tensor.extent_elem_cache = elements;
    tensor.buffer.size = elements * get_element_size(dtype);
    return tensor;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " libpaged_attention_aicpu_plan.so\n";
        return 2;
    }
    void *handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        std::cerr << "dlopen failed: " << dlerror() << '\n';
        return 2;
    }

    using Bind = int32_t (*)(const AicpuPlanBackendConfig *);
    using Close = int32_t (*)();
    using Result = AicpuPlanBackendResult (*)();
    using Entry = void (*)(const L2TaskArgs &);
    using Config = PTO2OrchestrationConfig (*)(const L2TaskArgs &);
    using Initialize = int32_t (*)(void *, void *, uint32_t);
    using Stage = int32_t (*)(
        void *, void *, uint32_t,
        const void *, uint32_t, int32_t, uint8_t, uint8_t, uint32_t,
        void *, uint32_t *, uint16_t *
    );
    using PublishStaged = int32_t (*)(
        void *, void *, uint32_t, const void *, uint32_t, uint8_t
    );
    using AdapterClose = int32_t (*)(
        void *, void *, uint32_t, uint32_t
    );
    using PublishFatal = void (*)(
        void *, void *, uint32_t, int64_t
    );
    const Bind bind = Load<Bind>(handle, "aicpu_plan_backend_bind");
    const Close close = Load<Close>(handle, "aicpu_plan_backend_close");
    const Result result = Load<Result>(handle, "aicpu_plan_backend_result");
    const Entry entry = Load<Entry>(handle, "aicpu_orchestration_entry");
    const Config orch_config =
        Load<Config>(handle, "aicpu_orchestration_config");
    const Initialize initialize = Load<Initialize>(
        handle, "aicpu_plan_adapter_initialize"
    );
    const Stage stage = Load<Stage>(
        handle, "aicpu_plan_adapter_stage"
    );
    const PublishStaged publish_staged = Load<PublishStaged>(
        handle, "aicpu_plan_adapter_publish_staged"
    );
    const AdapterClose adapter_close = Load<AdapterClose>(
        handle, "aicpu_plan_adapter_close"
    );
    const PublishFatal publish_fatal = Load<PublishFatal>(
        handle, "aicpu_plan_adapter_publish_fatal"
    );

    // 两个 batch 分别产生 G1 和 G2，真实 callback 序列为 5+9=14 个 task。
    // 这样一次 smoke 同时覆盖 UP->Alloc 的 batch 尾和 UP->QK 的下一组。
    constexpr uint32_t kBatch = 2U;
    constexpr uint32_t kTotalBlocks = 128U;
    constexpr uint32_t kBlockSize = 128U;
    constexpr uint32_t kHeads = 16U;
    constexpr uint32_t kHeadDim = 128U;
    constexpr uint32_t kBlockTableWidth = 128U;
    constexpr uint32_t kExpectedTasks = 14U;
    alignas(64) uint64_t dummy_buffers[6]{};
    alignas(64) int32_t context_lens[kBatch] = {4096, 9000};

    const uint32_t query_shape[3] = {kBatch, kHeads, kHeadDim};
    const uint32_t cache_shape[4] = {
        kTotalBlocks, kBlockSize, 1U, kHeadDim,
    };
    const uint32_t block_table_shape[2] = {kBatch, kBlockTableWidth};
    const uint32_t context_shape[1] = {kBatch};
    const uint32_t output_shape[3] = {kBatch, kHeads, kHeadDim};
    Tensor query = External(
        &dummy_buffers[0], query_shape, 3U, DataType::BFLOAT16
    );
    Tensor key = External(
        &dummy_buffers[1], cache_shape, 4U, DataType::BFLOAT16
    );
    Tensor value = External(
        &dummy_buffers[2], cache_shape, 4U, DataType::BFLOAT16
    );
    Tensor block_table = External(
        &dummy_buffers[3], block_table_shape, 2U, DataType::INT32
    );
    Tensor context = External(
        context_lens, context_shape, 1U, DataType::INT32
    );
    Tensor output = External(
        &dummy_buffers[4], output_shape, 3U, DataType::FLOAT32
    );
    L2TaskArgs args;
    args.reset();
    args.add_input(query, key, value, block_table, context, output);
    args.add_scalar(uint64_t{0x3f800000U});

    bool ok = Check(!args.has_error, "failed to construct real L2TaskArgs");
    ok &= Check(
        orch_config(args).expected_arg_count == 7,
        "real PA orchestration config expected_arg_count changed"
    );

    // 构造一个能通过 PA source 前置校验，但会在公共
    // PackRuntimeTaskPlan 检查 view offset+shape 溢出时失败的
    // SharedOutputRef。Pack 此时已经写过 header，所以这是真正的
    // “中途失败”门槛：部分 payload 可以存在，control 必须
    // 仍为 Empty，staged metadata 不可 publish，fatal 单独可见。
    void *partial_control_memory = nullptr;
    void *partial_cells_memory = nullptr;
    constexpr uint32_t kPartialCapacity = 2U;
    if (posix_memalign(
            &partial_control_memory, kAtomicIsolationBytes,
            sizeof(RuntimePlanControl)
        ) != 0 ||
        posix_memalign(
            &partial_cells_memory, kAtomicIsolationBytes,
            kPartialCapacity * sizeof(RuntimeTaskPlanCell)
        ) != 0) {
        std::cerr << "partial-pack aligned allocation failed\n";
        return 2;
    }
    std::memset(partial_control_memory, 0, sizeof(RuntimePlanControl));
    std::memset(
        partial_cells_memory, 0,
        kPartialCapacity * sizeof(RuntimeTaskPlanCell)
    );
    ok &= Check(
        initialize(
            partial_control_memory, partial_cells_memory,
            kPartialCapacity
        ) == 0,
        "partial-pack initialize failed"
    );
    auto *partial_control =
        static_cast<RuntimePlanControl *>(partial_control_memory);
    auto *partial_cells =
        static_cast<RuntimeTaskPlanCell *>(partial_cells_memory);
    ok &= Check(
        partial_control->closed_task_count.value ==
            kPlanProducerNotReadyTaskCount,
        "initialize opened a capacity-below-threshold Plan"
    );
    partial_cells[0].control.value = static_cast<int64_t>(
        EncodePlanCellControl(PlanCellPhase::Published, 1U, 0U)
    );
    L0TaskArgs malformed_args;
    malformed_args.reset();
    FdwicOutputRef overflowing_view{
        0, 0, 1U, 1U, UINT32_MAX, 1U,
    };
    malformed_args.add_input(overflowing_view);
    alignas(kAtomicIsolationBytes)
        std::array<uint8_t, kAtomicIsolationBytes> staged_metadata{};
    uint32_t malformed_lines = 0U;
    uint16_t malformed_outputs = 0U;
    const int32_t stage_status = stage(
        partial_control_memory, partial_cells_memory, kPartialCapacity,
        &malformed_args, 1U, 0, 1U, 0x81U, 0U,
        staged_metadata.data(), &malformed_lines, &malformed_outputs
    );
    const RuntimeTaskPlanHeader partial_header =
        DecodeRuntimeTaskPlanHeader(partial_cells[1].payload);
    ok &= Check(
        stage_status != 0 && partial_header.task_id == 1U &&
        partial_cells[1].control.value == 0,
        "mid-Pack failure became visible or did not exercise partial write"
    );
    ok &= Check(
        publish_staged(
            partial_control_memory, partial_cells_memory,
            kPartialCapacity, staged_metadata.data(), 1U, 0x81U
        ) != 0 && partial_cells[1].control.value == 0,
        "invalid staged metadata published a partial cell"
    );
    publish_fatal(
        partial_control_memory, partial_cells_memory,
        kPartialCapacity, -77
    );
    ok &= Check(
        partial_control->fatal.value == -77 &&
        partial_control->closed_task_count.value ==
            kPlanProducerReadyFailedTaskCount &&
        partial_cells[1].control.value == 0,
        "mid-Pack failure did not preserve Empty cell plus fatal/-3"
    );
    std::free(partial_cells_memory);
    std::free(partial_control_memory);

    alignas(kAtomicIsolationBytes) RuntimePlanControl close_control{};
    alignas(kAtomicIsolationBytes) RuntimeTaskPlanCell close_cell{};
    const auto check_early_build_close = [&](bool set_build_next) {
        std::memset(&close_control, 0, sizeof(close_control));
        std::memset(&close_cell, 0, sizeof(close_cell));
        bool case_ok = Check(
            initialize(&close_control, &close_cell, 1U) == 0,
            "early-build close initialize failed"
        );
        // 从协议层可关闭的 Ready/Open 状态注入 consumer 进度：NotReady 下现有
        // PublishRuntimePlanReady 门槛本来就会让两条 policy 都拒绝非零
        // 计数，无法区分 PlanAheadClosed 与 StreamingFuture 的 Close 合同。
        close_control.closed_task_count.value = kPlanOpenTaskCount;
        if (set_build_next) {
            close_control.build_next.value = 1;
        } else {
            close_control.build_workers_done.value = 1;
        }
        const int32_t close_status =
            adapter_close(&close_control, &close_cell, 1U, 0U);
        if constexpr (pa_scheduler::kRuntimePlanPipelineIsPlanAheadClosed) {
            case_ok &= Check(
                close_status != 0 &&
                    close_control.closed_task_count.value != 0,
                set_build_next
                    ? "PlanAheadClosed accepted nonzero build_next"
                    : "PlanAheadClosed accepted nonzero build_workers_done"
            );
        } else {
            case_ok &= Check(
                close_status == 0 &&
                    close_control.closed_task_count.value == 0,
                set_build_next
                    ? "StreamingFuture rejected concurrent build_next"
                    : "StreamingFuture rejected concurrent build_workers_done"
            );
        }
        return case_ok;
    };
    ok &= check_early_build_close(true);
    ok &= check_early_build_close(false);

    void *control_memory = nullptr;
    void *cells_memory = nullptr;
    if (posix_memalign(
            &control_memory, kAtomicIsolationBytes,
            sizeof(RuntimePlanControl)
        ) != 0 ||
        posix_memalign(
            &cells_memory, kAtomicIsolationBytes,
            kExpectedTasks * sizeof(RuntimeTaskPlanCell)
        ) != 0) {
        std::cerr << "aligned allocation failed\n";
        return 2;
    }
    std::memset(control_memory, 0, sizeof(RuntimePlanControl));
    std::memset(
        cells_memory, 0,
        kExpectedTasks * sizeof(RuntimeTaskPlanCell)
    );
    const AicpuPlanBackendConfig backend_config{
        control_memory, cells_memory, kExpectedTasks, 0U,
    };
    ok &= Check(bind(&backend_config) == 0, "backend bind failed");
    auto *control = static_cast<RuntimePlanControl *>(control_memory);
    auto *cells = static_cast<RuntimeTaskPlanCell *>(cells_memory);
    ok &= Check(
        control->closed_task_count.value ==
            kPlanProducerNotReadyTaskCount,
        "short Plan bind did not remain NotReady"
    );
    if (ok) entry(args);
    constexpr int64_t kShortPrecloseFrontier = 5;
    constexpr int64_t kExpectedShortReadyState =
        pa_scheduler::kRuntimePlanPipelineIsStreamingFuture &&
            kRuntimePlanReadyPrefillTasks <=
                static_cast<uint32_t>(kShortPrecloseFrontier)
            ? kPlanOpenTaskCount
            : kPlanProducerNotReadyTaskCount;
    ok &= Check(
        control->planned_frontier.value == kShortPrecloseFrontier &&
        control->closed_task_count.value == kExpectedShortReadyState,
        "short Plan Ready state did not match its preclose frontier"
    );
    ok &= Check(close() == 0, "backend close failed");

    const AicpuPlanBackendResult backend_result = result();
    ok &= Check(backend_result.status == 0, "backend reported failure");
    ok &= Check(
        backend_result.task_count == kExpectedTasks &&
        backend_result.begin_count == kExpectedTasks &&
        backend_result.finish_count == kExpectedTasks &&
        backend_result.published_count == kExpectedTasks,
        "actual callback count did not close at 14"
    );
    ok &= Check(
        backend_result.alloc_count == 2U &&
        backend_result.aic_count == 6U &&
        backend_result.aiv_count == 6U,
        "actual Alloc/AIC/AIV callback distribution changed"
    );

    RuntimePlanStorageRef storage{};
    storage.cells_base = reinterpret_cast<uintptr_t>(cells);
    storage.cells_bytes =
        kExpectedTasks * sizeof(RuntimeTaskPlanCell);
    storage.capacity = kExpectedTasks;
    storage.cell_bytes = sizeof(RuntimeTaskPlanCell);
    storage.abi_version = kRuntimePlanAbiVersion;
    RuntimePlanView view{};
    ok &= Check(
        MakeRuntimePlanView(control, storage, view),
        "published storage reference is invalid"
    );
    ok &= Check(
        control->planned_frontier.value == kExpectedTasks &&
        control->closed_task_count.value == kExpectedTasks &&
        control->fatal.value == 0,
        "Plan close/frontier/fatal control mismatch"
    );

    // 期望来自真实 G1+G2 callback 顺序；backend 本身没有这张 task 表。
    const std::array<uint8_t, kExpectedTasks> expected_kind{
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4, 1, 2, 3, 4,
    };
    const std::array<uint8_t, kExpectedTasks> expected_group{
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 1, 1,
    };
    const std::array<uint16_t, 5> expected_outputs{3, 1, 3, 1, 0};
    for (uint32_t task = 0U; task < kExpectedTasks; ++task) {
        const DecodedPlanCellControl decoded =
            DecodePlanCellControl(cells[task].control.value);
        RuntimeTaskPlanHeader header{};
        RuntimeTaskPlanLayout layout{};
        ok &= Check(
            decoded.valid && decoded.phase == PlanCellPhase::Published &&
            decoded.task_id == task,
            "cell control is not a canonical published identity"
        );
        ok &= Check(
            ValidateRuntimeTaskPlanPayload(
                cells[task].payload, task,
                decoded.payload_lines, header, layout
            ),
            "published payload failed canonical validation"
        );
        const uint8_t kind = expected_kind[task];
        const uint8_t group = expected_group[task];
        const bool has_following = task == 9U;
        const bool last_in_batch = task == 4U || task == 13U;
        const uint16_t expected_batch_start = task < 5U ? 0U : 5U;
        const uint8_t expected_flags = static_cast<uint8_t>(
            0x80U | (last_in_batch ? 0x40U : 0U) |
            (has_following ? 0x20U : 0U) |
            (group << 3U) | kind
        );
        ok &= Check(
            header.task_id == task &&
            header.adapter_flags == expected_flags &&
            header.adapter_data == expected_batch_start &&
            header.output_count == expected_outputs[kind],
            "task identity/flags/batch_start/output arity mismatch"
        );
        const uint8_t expected_engine =
            kind == 0U ? 0U : ((kind == 1U || kind == 3U) ? 1U : 2U);
        const uint32_t expected_function =
            kind == 0U ? kInvalidFunctionId : kind - 1U;
        ok &= Check(
            header.engine_class == expected_engine &&
            header.function_id == expected_function,
            "engine/function registry mapping mismatch"
        );
    }

    const size_t g1_g2_cells_bytes =
        kExpectedTasks * sizeof(RuntimeTaskPlanCell);
    std::vector<uint8_t> first_run_cells(g1_g2_cells_bytes);
    std::memcpy(
        first_run_cells.data(), cells_memory, g1_g2_cells_bytes
    );

    // 在完全相同的 control/cells 地址上模拟正式 Host 的下一轮：先清零
    // 同一份 GM storage，再重新 bind -> callback -> close。x86 本身是
    // cache coherent 的，因此这里只负责锁死协议顺序和同地址复用语义；
    // AICPU 对 Host DMA stale clean line 的真机门槛由 CCEC --runs 2 覆盖。
    ok &= Check(
        bind(&backend_config) != 0,
        "same-address reuse without Host reset must reject Published cells"
    );
    ok &= Check(
        control->closed_task_count.value ==
            kPlanProducerReadyFailedTaskCount &&
        control->fatal.value != 0,
        "initialize failure did not wake NotReady consumer with -3"
    );
    std::memset(control_memory, 0, sizeof(RuntimePlanControl));
    std::memset(
        cells_memory, 0,
        kExpectedTasks * sizeof(RuntimeTaskPlanCell)
    );
    ok &= Check(bind(&backend_config) == 0, "same-address second bind failed");
    if (ok) entry(args);
    ok &= Check(close() == 0, "same-address second close failed");
    const AicpuPlanBackendResult reused_result = result();
    ok &= Check(
        reused_result.status == 0 &&
        reused_result.task_count == kExpectedTasks &&
        reused_result.begin_count == kExpectedTasks &&
        reused_result.finish_count == kExpectedTasks &&
        reused_result.published_count == kExpectedTasks &&
        reused_result.alloc_count == 2U &&
        reused_result.aic_count == 6U &&
        reused_result.aiv_count == 6U,
        "same-address second producer run did not close exactly"
    );
    ok &= Check(
        control->planned_frontier.value == kExpectedTasks &&
        control->closed_task_count.value == kExpectedTasks &&
        control->fatal.value == 0,
        "same-address second Plan close/frontier/fatal mismatch"
    );
    for (uint32_t task = 0U; task < kExpectedTasks; ++task) {
        const DecodedPlanCellControl decoded =
            DecodePlanCellControl(cells[task].control.value);
        RuntimeTaskPlanHeader header{};
        RuntimeTaskPlanLayout layout{};
        const uint16_t expected_batch_start = task < 5U ? 0U : 5U;
        ok &= Check(
            decoded.valid && decoded.phase == PlanCellPhase::Published &&
            decoded.task_id == task &&
            ValidateRuntimeTaskPlanPayload(
                cells[task].payload, task,
                decoded.payload_lines, header, layout
            ) &&
            header.adapter_data == expected_batch_start,
            "same-address second PlanCell is not canonical published payload"
        );
    }
    ok &= Check(
        std::memcmp(
            first_run_cells.data(), cells_memory, g1_g2_cells_bytes
        ) == 0,
        "G1+G2 producer bytes changed across same-address reuse"
    );

    // 不清 payload，只把 control 恢复为 Empty，然后把 callback
    // 序列从 G1+G2 换成 G2+G1。同一 task_id 上同时出现
    // short->long（task5: Alloc->QK）和 long->short（task9:
    // UP->Alloc），锁死 Pack 覆盖本轮全部 published lines 的语义。
    alignas(64) int32_t swapped_context_lens[kBatch] = {9000, 4096};
    Tensor swapped_context = External(
        swapped_context_lens, context_shape, 1U, DataType::INT32
    );
    L2TaskArgs swapped_args;
    swapped_args.reset();
    swapped_args.add_input(
        query, key, value, block_table, swapped_context, output
    );
    swapped_args.add_scalar(uint64_t{0x3f800000U});
    ok &= Check(!swapped_args.has_error, "failed to construct G2+G1 args");
    std::memset(control_memory, 0, sizeof(RuntimePlanControl));
    for (uint32_t task = 0U; task < kExpectedTasks; ++task) {
        cells[task].control.value = 0;
    }
    ok &= Check(bind(&backend_config) == 0, "G2+G1 reuse bind failed");
    if (ok) entry(swapped_args);
    ok &= Check(close() == 0, "G2+G1 reuse close failed");
    bool swapped_ok = true;
    for (uint32_t task = 0U; task < kExpectedTasks; ++task) {
        const DecodedPlanCellControl decoded =
            DecodePlanCellControl(cells[task].control.value);
        RuntimeTaskPlanHeader header{};
        RuntimeTaskPlanLayout layout{};
        uint8_t kind = 0U;
        uint8_t group = 0U;
        uint16_t batch_start = 0U;
        bool has_following = false;
        bool last = false;
        if (task < 5U) {
            kind = static_cast<uint8_t>(task);
            has_following = task == 4U;
        } else if (task < 9U) {
            kind = static_cast<uint8_t>(task - 4U);
            group = 1U;
            last = task == 8U;
        } else {
            kind = static_cast<uint8_t>(task - 9U);
            batch_start = 9U;
            last = task == 13U;
        }
        const uint8_t flags = static_cast<uint8_t>(
            0x80U | (last ? 0x40U : 0U) |
            (has_following ? 0x20U : 0U) |
            (group << 3U) | kind
        );
        if (!decoded.valid ||
            decoded.phase != PlanCellPhase::Published ||
            decoded.task_id != task ||
            !ValidateRuntimeTaskPlanPayload(
                cells[task].payload, task,
                decoded.payload_lines, header, layout
            ) ||
            header.adapter_flags != flags ||
            header.adapter_data != batch_start ||
            header.output_count != expected_outputs[kind]) {
            swapped_ok = false;
            break;
        }
    }
    ok &= Check(
        swapped_ok,
        "G2+G1 short/long same-address reuse is not canonical"
    );

    // 再把每个 payload 全部填成非零 poison，control 单独保持
    // Empty，然后重放 G1+G2。ValidateRuntimeTaskPlanPayload 会检查
    // 每个已发布 line 内的 tensor canonical-zero 和最后一条
    // tail-zero，所以任何短 layout 漏覆盖都会立即失败。
    std::memset(control_memory, 0, sizeof(RuntimePlanControl));
    for (uint32_t task = 0U; task < kExpectedTasks; ++task) {
        std::memset(
            &cells[task].payload,
            0xA5, sizeof(RuntimeTaskPlanStorage)
        );
        cells[task].control.value = 0;
    }
    ok &= Check(bind(&backend_config) == 0, "poison reuse bind failed");
    if (ok) entry(args);
    ok &= Check(close() == 0, "poison reuse close failed");
    bool poison_ok = true;
    for (uint32_t task = 0U; task < kExpectedTasks; ++task) {
        const DecodedPlanCellControl decoded =
            DecodePlanCellControl(cells[task].control.value);
        RuntimeTaskPlanHeader header{};
        RuntimeTaskPlanLayout layout{};
        if (!decoded.valid ||
            decoded.phase != PlanCellPhase::Published ||
            decoded.task_id != task ||
            !ValidateRuntimeTaskPlanPayload(
                cells[task].payload, task,
                decoded.payload_lines, header, layout
            )) {
            poison_ok = false;
            break;
        }
    }
    ok &= Check(
        poison_ok,
        "poisoned same-address payload was not fully overwritten"
    );

    // 容量在最后一个真实 callback 之前用尽，锁死 producer
    // 失败契约：发布 fatal、close 失败，且无法完成的 cell
    // control 必须仍为 Empty。即使之前已经对其 payload 写入
    // poison，consumer 也不得把它当成可 acquire 的 task。
    constexpr uint32_t kShortCapacity = kExpectedTasks - 1U;
    std::memset(control_memory, 0, sizeof(RuntimePlanControl));
    for (uint32_t task = 0U; task < kExpectedTasks; ++task) {
        std::memset(
            &cells[task].payload, 0x5A,
            sizeof(RuntimeTaskPlanStorage)
        );
        cells[task].control.value = 0;
    }
    const AicpuPlanBackendConfig short_config{
        control_memory, cells_memory, kShortCapacity, 0U,
    };
    ok &= Check(bind(&short_config) == 0, "short-capacity bind failed");
    if (ok) entry(args);
    ok &= Check(close() != 0, "short-capacity producer unexpectedly closed");
    const AicpuPlanBackendResult failed_result = result();
    ok &= Check(
        failed_result.status != 0 && control->fatal.value != 0 &&
        control->closed_task_count.value ==
            kPlanProducerReadyFailedTaskCount &&
        cells[kShortCapacity].control.value == 0,
        "producer failure did not preserve Empty control and publish fatal/-3"
    );

    // 用 context=0 的真实 PA batch 构造“每 batch 只有 Alloc”的
    // callback 序列，从而将任务数精确锁在编译期 Ready 门槛。
    // entry 返回时最后一个 Alloc 仍是 pending：前 threshold-1 个
    // cell 必须已在 NotReady 下成功 stage/publish，Close 再发布
    // 最后一个 cell，然后按两条 policy 共用的 Ready -> Closed(N) 收口。
    constexpr uint32_t kExactTasks = kRuntimePlanReadyPrefillTasks;
    constexpr uint32_t kExactCapacity =
        kExactTasks < kMaxRuntimeTasks ? kExactTasks + 1U : kExactTasks;
    std::vector<int32_t> exact_context_lens(kExactCapacity, 0);
    const uint32_t exact_query_shape[3] = {
        kExactTasks, kHeads, kHeadDim,
    };
    const uint32_t exact_block_table_shape[2] = {
        kExactTasks, kBlockTableWidth,
    };
    const uint32_t exact_context_shape[1] = {kExactTasks};
    const uint32_t exact_output_shape[3] = {
        kExactTasks, kHeads, kHeadDim,
    };
    Tensor exact_query = External(
        &dummy_buffers[0], exact_query_shape, 3U, DataType::BFLOAT16
    );
    Tensor exact_key = External(
        &dummy_buffers[1], cache_shape, 4U, DataType::BFLOAT16
    );
    Tensor exact_value = External(
        &dummy_buffers[2], cache_shape, 4U, DataType::BFLOAT16
    );
    Tensor exact_block_table = External(
        &dummy_buffers[3], exact_block_table_shape, 2U, DataType::INT32
    );
    Tensor exact_context = External(
        exact_context_lens.data(), exact_context_shape, 1U,
        DataType::INT32
    );
    Tensor exact_output = External(
        &dummy_buffers[4], exact_output_shape, 3U, DataType::FLOAT32
    );
    L2TaskArgs exact_args;
    exact_args.reset();
    exact_args.add_input(
        exact_query, exact_key, exact_value, exact_block_table,
        exact_context, exact_output
    );
    exact_args.add_scalar(uint64_t{0x3f800000U});
    ok &= Check(
        !exact_args.has_error,
        "failed to construct N==prefill-threshold L2TaskArgs"
    );

    void *exact_control_memory = nullptr;
    void *exact_cells_memory = nullptr;
    const size_t exact_cells_bytes =
        static_cast<size_t>(kExactCapacity) *
        sizeof(RuntimeTaskPlanCell);
    if (posix_memalign(
            &exact_control_memory, kAtomicIsolationBytes,
            sizeof(RuntimePlanControl)
        ) != 0 ||
        posix_memalign(
            &exact_cells_memory, kAtomicIsolationBytes,
            exact_cells_bytes
        ) != 0) {
        std::cerr << "exact-threshold aligned allocation failed\n";
        return 2;
    }
    std::memset(exact_control_memory, 0, sizeof(RuntimePlanControl));
    std::memset(exact_cells_memory, 0, exact_cells_bytes);
    const AicpuPlanBackendConfig exact_config{
        exact_control_memory, exact_cells_memory, kExactCapacity, 0U,
    };
    auto *exact_control =
        static_cast<RuntimePlanControl *>(exact_control_memory);
    auto *exact_cells =
        static_cast<RuntimeTaskPlanCell *>(exact_cells_memory);
    ok &= Check(bind(&exact_config) == 0, "exact-threshold bind failed");
    ok &= Check(
        exact_control->closed_task_count.value ==
            kPlanProducerNotReadyTaskCount,
        "exact-threshold bind did not remain NotReady"
    );
    if (ok) entry(exact_args);
    const AicpuPlanBackendResult exact_before_close = result();
    bool exact_prefill_cells_ok = true;
    for (uint32_t task = 0U; task + 1U < kExactTasks; ++task) {
        const DecodedPlanCellControl decoded =
            DecodePlanCellControl(exact_cells[task].control.value);
        if (!decoded.valid ||
            decoded.phase != PlanCellPhase::Published ||
            decoded.task_id != task) {
            exact_prefill_cells_ok = false;
            break;
        }
    }
    ok &= Check(
        exact_before_close.status == 0 &&
        exact_before_close.begin_count == kExactTasks &&
        exact_before_close.finish_count == kExactTasks &&
        exact_before_close.published_count == kExactTasks - 1U &&
        exact_before_close.alloc_count == kExactTasks &&
        exact_before_close.aic_count == 0U &&
        exact_before_close.aiv_count == 0U &&
        exact_control->planned_frontier.value ==
            static_cast<int64_t>(kExactTasks - 1U) &&
        exact_control->closed_task_count.value ==
            kPlanProducerNotReadyTaskCount &&
        exact_cells[kExactTasks - 1U].control.value == 0 &&
        exact_prefill_cells_ok,
        "NotReady exact-threshold prefill did not stage/publish its continuous prefix"
    );
    ok &= Check(close() == 0, "exact-threshold forced-ready close failed");
    const AicpuPlanBackendResult exact_result = result();
    const DecodedPlanCellControl exact_last = DecodePlanCellControl(
        exact_cells[kExactTasks - 1U].control.value
    );
    ok &= Check(
        exact_result.status == 0 &&
        exact_result.task_count == kExactTasks &&
        exact_result.published_count == kExactTasks &&
        exact_control->planned_frontier.value ==
            static_cast<int64_t>(kExactTasks) &&
        exact_control->closed_task_count.value ==
            static_cast<int64_t>(kExactTasks) &&
        exact_control->fatal.value == 0 && exact_last.valid &&
        exact_last.phase == PlanCellPhase::Published &&
        exact_last.task_id == kExactTasks - 1U,
        "N==threshold did not force Ready and close exactly"
    );

    if constexpr (kExactTasks < kMaxRuntimeTasks) {
        // 同一地址复用为 threshold+1 个 Alloc。第 threshold+1 个
        // Begin 会发布前一个 pending cell，使 frontier 精确达到
        // threshold。StreamingFuture 此时在 entry 内 Open；
        // PlanAheadClosed 必须继续保持 NotReady。随后的 fatal 在两条
        // policy 下都必须转为 -3。
        const uint32_t above_query_shape[3] = {
            kExactCapacity, kHeads, kHeadDim,
        };
        const uint32_t above_block_table_shape[2] = {
            kExactCapacity, kBlockTableWidth,
        };
        const uint32_t above_context_shape[1] = {kExactCapacity};
        const uint32_t above_output_shape[3] = {
            kExactCapacity, kHeads, kHeadDim,
        };
        Tensor above_query = External(
            &dummy_buffers[0], above_query_shape, 3U,
            DataType::BFLOAT16
        );
        Tensor above_block_table = External(
            &dummy_buffers[3], above_block_table_shape, 2U,
            DataType::INT32
        );
        Tensor above_context = External(
            exact_context_lens.data(), above_context_shape, 1U,
            DataType::INT32
        );
        Tensor above_output = External(
            &dummy_buffers[4], above_output_shape, 3U,
            DataType::FLOAT32
        );
        L2TaskArgs above_args;
        above_args.reset();
        above_args.add_input(
            above_query, exact_key, exact_value, above_block_table,
            above_context, above_output
        );
        above_args.add_scalar(uint64_t{0x3f800000U});
        std::memset(
            exact_control_memory, 0, sizeof(RuntimePlanControl)
        );
        std::memset(exact_cells_memory, 0, exact_cells_bytes);
        ok &= Check(
            !above_args.has_error && bind(&exact_config) == 0,
            "same-address N>threshold bind failed"
        );
        if (ok) entry(above_args);
        const AicpuPlanBackendResult above_result = result();
        ok &= Check(
            above_result.status == 0 &&
            above_result.begin_count == kExactCapacity &&
            above_result.finish_count == kExactCapacity &&
            above_result.published_count == kExactTasks &&
            exact_control->planned_frontier.value ==
                static_cast<int64_t>(kExactTasks) &&
            exact_control->closed_task_count.value ==
                (pa_scheduler::kRuntimePlanPipelineIsStreamingFuture
                    ? kPlanOpenTaskCount
                    : kPlanProducerNotReadyTaskCount) &&
            exact_cells[kExactTasks].control.value == 0,
            "N>threshold producer Ready timing violated the selected policy"
        );
        publish_fatal(
            exact_control_memory, exact_cells_memory,
            kExactCapacity, -88
        );
        ok &= Check(
            exact_control->fatal.value == -88 &&
            exact_control->closed_task_count.value ==
                kPlanProducerReadyFailedTaskCount &&
            exact_cells[kExactTasks].control.value == 0,
            "producer fatal did not publish ReadyFailed=-3"
        );
    }
    std::free(exact_cells_memory);
    std::free(exact_control_memory);

    // B256 的真实 callback 回放用所有 batch=G1，因此产生
    // 256 * (Alloc/QK/SF/PV/UP) = 1280 个 task。除了锁定计数和
    // canonical wire，还在同一 storage 上清零后再跑一轮并做
    // byte-for-byte 比较。这个门槛会直接捕到 single-pack 丢字、
    // 尾 cache-line 非 canonical zero 或 SharedOutputRef 不稳定。
    constexpr uint32_t kBulkBatch = 256U;
    constexpr uint32_t kBulkTasks = kBulkBatch * 5U;
    alignas(64) std::array<int32_t, kBulkBatch> bulk_context_lens{};
    bulk_context_lens.fill(4096);
    const uint32_t bulk_query_shape[3] = {
        kBulkBatch, kHeads, kHeadDim,
    };
    const uint32_t bulk_block_table_shape[2] = {
        kBulkBatch, kBlockTableWidth,
    };
    const uint32_t bulk_context_shape[1] = {kBulkBatch};
    const uint32_t bulk_output_shape[3] = {
        kBulkBatch, kHeads, kHeadDim,
    };
    Tensor bulk_query = External(
        &dummy_buffers[0], bulk_query_shape, 3U, DataType::BFLOAT16
    );
    Tensor bulk_key = External(
        &dummy_buffers[1], cache_shape, 4U, DataType::BFLOAT16
    );
    Tensor bulk_value = External(
        &dummy_buffers[2], cache_shape, 4U, DataType::BFLOAT16
    );
    Tensor bulk_block_table = External(
        &dummy_buffers[3], bulk_block_table_shape, 2U, DataType::INT32
    );
    Tensor bulk_context = External(
        bulk_context_lens.data(), bulk_context_shape, 1U,
        DataType::INT32
    );
    Tensor bulk_output = External(
        &dummy_buffers[4], bulk_output_shape, 3U, DataType::FLOAT32
    );
    L2TaskArgs bulk_args;
    bulk_args.reset();
    bulk_args.add_input(
        bulk_query, bulk_key, bulk_value, bulk_block_table,
        bulk_context, bulk_output
    );
    bulk_args.add_scalar(uint64_t{0x3f800000U});
    ok &= Check(
        !bulk_args.has_error,
        "failed to construct B256 real L2TaskArgs"
    );

    void *bulk_control_memory = nullptr;
    void *bulk_cells_memory = nullptr;
    const size_t bulk_cells_bytes =
        static_cast<size_t>(kBulkTasks) * sizeof(RuntimeTaskPlanCell);
    if (posix_memalign(
            &bulk_control_memory, kAtomicIsolationBytes,
            sizeof(RuntimePlanControl)
        ) != 0 ||
        posix_memalign(
            &bulk_cells_memory, kAtomicIsolationBytes,
            bulk_cells_bytes
        ) != 0) {
        std::cerr << "B256 aligned allocation failed\n";
        return 2;
    }
    const AicpuPlanBackendConfig bulk_config{
        bulk_control_memory, bulk_cells_memory, kBulkTasks, 0U,
    };
    std::vector<uint8_t> bulk_first_run(bulk_cells_bytes);
    std::array<double, 2> bulk_producer_us{};
    for (uint32_t run = 0U; run < 2U; ++run) {
        std::memset(
            bulk_control_memory, 0, sizeof(RuntimePlanControl)
        );
        std::memset(bulk_cells_memory, 0, bulk_cells_bytes);
        const auto begin = std::chrono::steady_clock::now();
        ok &= Check(bind(&bulk_config) == 0, "B256 backend bind failed");
        auto *bulk_control =
            static_cast<RuntimePlanControl *>(bulk_control_memory);
        ok &= Check(
            bulk_control->closed_task_count.value ==
                kPlanProducerNotReadyTaskCount,
            "B256 bind did not remain NotReady"
        );
        if (ok) entry(bulk_args);
        constexpr int64_t kBulkPrecloseFrontier =
            static_cast<int64_t>(kBulkTasks - 5U);
        constexpr int64_t kExpectedBulkReadyState =
            pa_scheduler::kRuntimePlanPipelineIsStreamingFuture &&
                kRuntimePlanReadyPrefillTasks <=
                    static_cast<uint32_t>(kBulkPrecloseFrontier)
                ? kPlanOpenTaskCount
                : kPlanProducerNotReadyTaskCount;
        ok &= Check(
            bulk_control->closed_task_count.value ==
                kExpectedBulkReadyState &&
            bulk_control->planned_frontier.value ==
                kBulkPrecloseFrontier,
            "B256 Ready state did not match its prefilled batch frontier"
        );
        ok &= Check(close() == 0, "B256 backend close failed");
        const auto end = std::chrono::steady_clock::now();
        bulk_producer_us[run] =
            std::chrono::duration<double, std::micro>(end - begin).count();

        const AicpuPlanBackendResult bulk_result = result();
        ok &= Check(
            bulk_result.status == 0 &&
            bulk_result.task_count == kBulkTasks &&
            bulk_result.begin_count == kBulkTasks &&
            bulk_result.finish_count == kBulkTasks &&
            bulk_result.published_count == kBulkTasks &&
            bulk_result.alloc_count == kBulkBatch &&
            bulk_result.aic_count == 2U * kBulkBatch &&
            bulk_result.aiv_count == 2U * kBulkBatch,
            "B256 callback/task distribution changed"
        );
        auto *bulk_cells =
            static_cast<RuntimeTaskPlanCell *>(bulk_cells_memory);
        ok &= Check(
            bulk_control->planned_frontier.value == kBulkTasks &&
            bulk_control->closed_task_count.value == kBulkTasks &&
            bulk_control->fatal.value == 0,
            "B256 Plan close/frontier/fatal mismatch"
        );

        bool cells_ok = true;
        for (uint32_t task = 0U; task < kBulkTasks; ++task) {
            const DecodedPlanCellControl decoded =
                DecodePlanCellControl(bulk_cells[task].control.value);
            RuntimeTaskPlanHeader header{};
            RuntimeTaskPlanLayout layout{};
            const uint8_t kind = static_cast<uint8_t>(task % 5U);
            const uint32_t batch_start = task - kind;
            const uint8_t flags = static_cast<uint8_t>(
                0x80U | (kind == 4U ? 0x40U : 0U) | kind
            );
            if (!decoded.valid ||
                decoded.phase != PlanCellPhase::Published ||
                decoded.task_id != task ||
                !ValidateRuntimeTaskPlanPayload(
                    bulk_cells[task].payload, task,
                    decoded.payload_lines, header, layout
                ) ||
                header.adapter_flags != flags ||
                header.adapter_data != batch_start ||
                header.output_count != expected_outputs[kind]) {
                cells_ok = false;
                break;
            }
        }
        ok &= Check(cells_ok, "B256 canonical Plan content changed");
        if (run == 0U) {
            std::memcpy(
                bulk_first_run.data(), bulk_cells_memory,
                bulk_cells_bytes
            );
        } else {
            ok &= Check(
                std::memcmp(
                    bulk_first_run.data(), bulk_cells_memory,
                    bulk_cells_bytes
                ) == 0,
                "B256 producer bytes changed across same-address reuse"
            );
        }
    }

    std::free(bulk_cells_memory);
    std::free(bulk_control_memory);
    std::free(cells_memory);
    std::free(control_memory);
    dlclose(handle);
    if (!ok) return 1;
    std::cout
        << "PASS real paged_attention orchestration SO: tasks="
        << backend_result.task_count
        << " alloc/aic/aiv=" << backend_result.alloc_count << '/'
        << backend_result.aic_count << '/' << backend_result.aiv_count
        << " producer_runs=2 same_address_reuse=yes"
        << " pipeline_policy="
        << (pa_scheduler::kRuntimePlanPipelineIsPlanAheadClosed
                ? "plan_ahead_closed"
                : "streaming_future")
        << " b256_producer_us=" << bulk_producer_us[0]
        << ',' << bulk_producer_us[1]
        << '\n';
    return 0;
}
