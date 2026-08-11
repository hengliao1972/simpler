/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: CANN-2.0
 */

#include "aicpu_plan_backend_abi.h"
#include "aicpu_plan_adapter_bridge.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "dist_engine/dist_engine_api.h"

namespace {

constexpr uint8_t kMetaPresent = uint8_t{1} << 7U;
constexpr uint8_t kLastInBatch = uint8_t{1} << 6U;
constexpr uint8_t kHasFollowingGroup = uint8_t{1} << 5U;
constexpr uint8_t kGroupShift = 3U;
constexpr uint32_t kMaxGroups = 4U;

enum class ObservedTaskKind : uint8_t {
    Alloc = 0,
    Qk = 1,
    Sf = 2,
    Pv = 3,
    Up = 4,
};

enum class ObservedEngine : uint8_t {
    MetadataOnly = 0,
    Aic = 1,
    Aiv = 2,
};

struct PendingPlan {
    // payload 已经在 Finish callback 存活期间唯一一次 Pack 到
    // control=Empty 的目标 GM cell。这里只保留不含指针的窄
    // metadata，供下一个 Begin/close patch 最终 flags 并发布。
    alignas(128) uint8_t staged_metadata[128];
    uint32_t payload_lines;
    uint32_t task_id;
    uint32_t batch_start;
    int32_t function_id;
    uint16_t output_count;
    ObservedTaskKind kind;
    ObservedEngine engine;
    uint8_t group;
    bool valid;
};

struct ActiveTicket {
    uint32_t task_id;
    uint32_t batch_start;
    int32_t function_id;
    ObservedTaskKind kind;
    ObservedEngine engine;
    uint8_t group;
    bool active;
};

struct BackendState {
    AicpuPlanBackendConfig config;
    AicpuPlanBackendResult result;
    PendingPlan pending;
    ActiveTicket active;
    ObservedTaskKind previous_kind;
    uint8_t previous_group;
    uint32_t current_batch_start;
    bool have_previous;
    bool bound;
    bool closed;
};

BackendState g_backend{};

void Fail(AicpuPlanBackendStatus status, int32_t fatal_code = 0)
{
    if (g_backend.result.status ==
        static_cast<int32_t>(AicpuPlanBackendStatus::Ok)) {
        g_backend.result.status = static_cast<int32_t>(status);
        g_backend.result.fatal_code = fatal_code;
    }
    if (g_backend.bound) {
        const int64_t published = fatal_code != 0
            ? static_cast<int64_t>(fatal_code)
            : static_cast<int64_t>(status);
        aicpu_plan_adapter_publish_fatal(
            g_backend.config.control, g_backend.config.cells,
            g_backend.config.capacity, published
        );
    }
}

uint8_t EncodeFlags(
    ObservedTaskKind kind, uint8_t group,
    bool has_following, bool last_in_batch
)
{
    if (group >= kMaxGroups ||
        (kind == ObservedTaskKind::Alloc &&
         (group != 0U || has_following)) ||
        (kind != ObservedTaskKind::Up && has_following) ||
        (last_in_batch && has_following)) {
        return 0U;
    }
    return static_cast<uint8_t>(
        kMetaPresent |
        (last_in_batch ? kLastInBatch : 0U) |
        (has_following ? kHasFollowingGroup : 0U) |
        (group << kGroupShift) |
        static_cast<uint8_t>(kind)
    );
}

bool ValidTransition(
    ObservedTaskKind previous, uint8_t previous_group,
    ObservedTaskKind next, uint8_t &next_group,
    bool &previous_has_following, bool &previous_last
)
{
    previous_has_following = false;
    previous_last = false;
    next_group = previous_group;
    switch (previous) {
        case ObservedTaskKind::Alloc:
            if (next == ObservedTaskKind::Qk) {
                next_group = 0U;
                return true;
            }
            if (next == ObservedTaskKind::Alloc) {
                previous_last = true;
                next_group = 0U;
                return true;
            }
            return false;
        case ObservedTaskKind::Qk:
            return next == ObservedTaskKind::Sf;
        case ObservedTaskKind::Sf:
            return next == ObservedTaskKind::Pv;
        case ObservedTaskKind::Pv:
            return next == ObservedTaskKind::Up;
        case ObservedTaskKind::Up:
            if (next == ObservedTaskKind::Qk &&
                previous_group + 1U < kMaxGroups) {
                previous_has_following = true;
                next_group = previous_group + 1U;
                return true;
            }
            if (next == ObservedTaskKind::Alloc) {
                previous_last = true;
                next_group = 0U;
                return true;
            }
            return false;
    }
    return false;
}

bool PublishPending(bool has_following, bool last_in_batch)
{
    if (!g_backend.pending.valid) return true;
    const uint8_t flags = EncodeFlags(
        g_backend.pending.kind, g_backend.pending.group,
        has_following, last_in_batch
    );
    if (flags == 0U ||
        aicpu_plan_adapter_publish_staged(
            g_backend.config.control, g_backend.config.cells,
            g_backend.config.capacity,
            g_backend.pending.staged_metadata,
            g_backend.pending.payload_lines, flags
        ) != 0) {
        Fail(AicpuPlanBackendStatus::PublishFailed);
        return false;
    }
    ++g_backend.result.published_count;
    g_backend.pending.valid = false;
    return true;
}

DistCompeteFirstTicket BeginTask(
    ObservedTaskKind kind, ObservedEngine engine, int32_t function_id
)
{
    DistCompeteFirstTicket ticket{};
    if (!g_backend.bound || g_backend.closed ||
        g_backend.active.active ||
        g_backend.result.status !=
            static_cast<int32_t>(AicpuPlanBackendStatus::Ok)) {
        Fail(g_backend.bound
                 ? AicpuPlanBackendStatus::BadSequence
                 : AicpuPlanBackendStatus::NotBound);
        return ticket;
    }

    uint8_t group = 0U;
    if (g_backend.have_previous) {
        bool has_following = false;
        bool last = false;
        if (!ValidTransition(
                g_backend.previous_kind, g_backend.previous_group,
                kind, group, has_following, last
            ) ||
            !PublishPending(has_following, last)) {
            Fail(AicpuPlanBackendStatus::BadSequence);
            return ticket;
        }
    } else if (kind != ObservedTaskKind::Alloc) {
        Fail(AicpuPlanBackendStatus::BadSequence);
        return ticket;
    }

    const uint32_t task_id = g_backend.result.begin_count;
    if (task_id >= g_backend.config.capacity ||
        function_id < INT16_MIN || function_id > INT16_MAX) {
        Fail(AicpuPlanBackendStatus::BadSequence);
        return ticket;
    }
    // batch_start 来自真实 Alloc callback 的位置，并沿当前 callback
    // continuation 传播；它不是由 task_id 的固定 PA 周期公式恢复。
    if (kind == ObservedTaskKind::Alloc) {
        g_backend.current_batch_start = task_id;
    } else if (!g_backend.have_previous && task_id != 0U) {
        Fail(AicpuPlanBackendStatus::BadSequence);
        return ticket;
    }
    g_backend.active = ActiveTicket{
        task_id, g_backend.current_batch_start,
        function_id, kind, engine, group, true,
    };
    g_backend.previous_kind = kind;
    g_backend.previous_group = group;
    g_backend.have_previous = true;
    ++g_backend.result.begin_count;
    if (kind == ObservedTaskKind::Alloc) {
        ++g_backend.result.alloc_count;
    } else if (engine == ObservedEngine::Aic) {
        ++g_backend.result.aic_count;
    } else {
        ++g_backend.result.aiv_count;
    }

    ticket.task_id = static_cast<int32_t>(task_id);
    ticket.kernel_id = static_cast<int16_t>(function_id);
    ticket.won = 1U;
    ticket.ready = 1U;
    return ticket;
}

bool FinishTask(
    const DistCompeteFirstTicket &ticket,
    const L0TaskArgs &args, uint32_t expected_output_count,
    ObservedEngine expected_engine, int32_t expected_function
)
{
    if (!g_backend.active.active || ticket.ready == 0U ||
        ticket.won == 0U || ticket.task_id < 0 ||
        static_cast<uint32_t>(ticket.task_id) !=
            g_backend.active.task_id ||
        ticket.kernel_id != g_backend.active.function_id ||
        g_backend.active.engine != expected_engine ||
        g_backend.active.function_id != expected_function ||
        g_backend.pending.valid) {
        Fail(AicpuPlanBackendStatus::BadTicket);
        return false;
    }
    const uint8_t provisional = EncodeFlags(
        g_backend.active.kind, g_backend.active.group, false, false
    );
    uint32_t payload_lines = 0U;
    uint16_t actual_output_count = 0U;
    if (provisional == 0U ||
        aicpu_plan_adapter_stage(
            g_backend.config.control, g_backend.config.cells,
            g_backend.config.capacity,
            &args, g_backend.active.task_id,
            g_backend.active.function_id,
            static_cast<uint8_t>(g_backend.active.engine), provisional,
            g_backend.active.batch_start,
            g_backend.pending.staged_metadata, &payload_lines,
            &actual_output_count
        ) != 0 ||
        actual_output_count != expected_output_count) {
        Fail(AicpuPlanBackendStatus::BadTaskArgs);
        g_backend.active.active = false;
        return false;
    }
    g_backend.pending.payload_lines = payload_lines;
    g_backend.pending.task_id = g_backend.active.task_id;
    g_backend.pending.batch_start = g_backend.active.batch_start;
    g_backend.pending.function_id = g_backend.active.function_id;
    g_backend.pending.output_count = actual_output_count;
    g_backend.pending.kind = g_backend.active.kind;
    g_backend.pending.engine = g_backend.active.engine;
    g_backend.pending.group = g_backend.active.group;
    g_backend.pending.valid = true;
    g_backend.active.active = false;
    ++g_backend.result.finish_count;
    return true;
}

bool ResolveKernel(
    const MixedKernels &mixed, ObservedTaskKind &kind,
    ObservedEngine &engine, int32_t &function_id
)
{
    // 这是 PA adapter 的函数注册表：身份来自真实 Submit API 传入的
    // MixedKernels，而不是 task_id、batch 或 Host 预制公式。
    if (mixed.aic_kernel_id == 0 &&
        mixed.aiv0_kernel_id == INVALID_KERNEL_ID &&
        mixed.aiv1_kernel_id == INVALID_KERNEL_ID) {
        kind = ObservedTaskKind::Qk;
        engine = ObservedEngine::Aic;
        function_id = 0;
        return true;
    }
    if (mixed.aic_kernel_id == 2 &&
        mixed.aiv0_kernel_id == INVALID_KERNEL_ID &&
        mixed.aiv1_kernel_id == INVALID_KERNEL_ID) {
        kind = ObservedTaskKind::Pv;
        engine = ObservedEngine::Aic;
        function_id = 2;
        return true;
    }
    if (mixed.aic_kernel_id == INVALID_KERNEL_ID &&
        mixed.aiv0_kernel_id == 1 &&
        mixed.aiv1_kernel_id == INVALID_KERNEL_ID) {
        kind = ObservedTaskKind::Sf;
        engine = ObservedEngine::Aiv;
        function_id = 1;
        return true;
    }
    if (mixed.aic_kernel_id == INVALID_KERNEL_ID &&
        mixed.aiv0_kernel_id == 3 &&
        mixed.aiv1_kernel_id == INVALID_KERNEL_ID) {
        kind = ObservedTaskKind::Up;
        engine = ObservedEngine::Aiv;
        function_id = 3;
        return true;
    }
    return false;
}

uint64_t TensorScalarAddress(
    const Tensor &tensor, uint32_t ndims, const uint32_t indices[]
)
{
    if (tensor.buffer.addr == 0U || indices == nullptr ||
        ndims != tensor.ndims || ndims == 0U ||
        ndims > MAX_TENSOR_DIMS) {
        return 0U;
    }
    uint64_t element = tensor.start_offset;
    for (uint32_t dim = 0U; dim < ndims; ++dim) {
        if (indices[dim] >= tensor.shapes[dim]) return 0U;
        element += static_cast<uint64_t>(indices[dim]) *
                   tensor.strides[dim];
    }
    return tensor.buffer.addr + element * get_element_size(tensor.dtype);
}

}  // namespace

extern "C" int32_t aicpu_plan_backend_bind(
    const AicpuPlanBackendConfig *config
)
{
    if (config == nullptr || config->control == nullptr ||
        config->cells == nullptr || config->capacity == 0U ||
        config->reserved != 0U) {
        return static_cast<int32_t>(AicpuPlanBackendStatus::BadConfig);
    }
    BackendState clean{};
    clean.config = *config;
    clean.result.status =
        static_cast<int32_t>(AicpuPlanBackendStatus::Ok);
    clean.result.fatal_code = 0;
    clean.bound = true;
    g_backend = clean;
    if (aicpu_plan_adapter_initialize(
            config->control, config->cells, config->capacity
        ) != 0) {
        Fail(AicpuPlanBackendStatus::BadConfig);
    }
    return g_backend.result.status;
}

extern "C" int32_t aicpu_plan_backend_close()
{
    if (!g_backend.bound || g_backend.closed ||
        g_backend.active.active ||
        g_backend.result.status !=
            static_cast<int32_t>(AicpuPlanBackendStatus::Ok) ||
        !g_backend.pending.valid ||
        (g_backend.pending.kind != ObservedTaskKind::Up &&
         g_backend.pending.kind != ObservedTaskKind::Alloc)) {
        Fail(g_backend.bound
                 ? AicpuPlanBackendStatus::CloseFailed
                 : AicpuPlanBackendStatus::NotBound);
        return g_backend.result.status;
    }
    if (!PublishPending(false, true) ||
        aicpu_plan_adapter_close(
            g_backend.config.control, g_backend.config.cells,
            g_backend.config.capacity, g_backend.result.finish_count
        ) != 0) {
        Fail(AicpuPlanBackendStatus::CloseFailed);
        return g_backend.result.status;
    }
    g_backend.result.task_count = g_backend.result.finish_count;
    g_backend.closed = true;
    return static_cast<int32_t>(AicpuPlanBackendStatus::Ok);
}

extern "C" AicpuPlanBackendResult aicpu_plan_backend_result()
{
    return g_backend.result;
}

DistCompeteFirstTicket dist_alloc_compete_first_begin(PTO2Runtime *)
{
    return BeginTask(
        ObservedTaskKind::Alloc,
        ObservedEngine::MetadataOnly, INVALID_KERNEL_ID
    );
}

bool dist_alloc_deferred_compete_first_finish(
    PTO2Runtime *, const DistCompeteFirstTicket &ticket,
    const L0TaskArgs &args, uint32_t expected_output_count
)
{
    return FinishTask(
        ticket, args, expected_output_count,
        ObservedEngine::MetadataOnly, INVALID_KERNEL_ID
    );
}

DistCompeteFirstTicket dist_submit_compete_first_begin(
    PTO2Runtime *, const MixedKernels &mixed
)
{
    ObservedTaskKind kind{};
    ObservedEngine engine{};
    int32_t function_id = INVALID_KERNEL_ID;
    if (!ResolveKernel(mixed, kind, engine, function_id)) {
        Fail(AicpuPlanBackendStatus::BadSequence);
        return DistCompeteFirstTicket{};
    }
    return BeginTask(kind, engine, function_id);
}

bool dist_submit_deferred_compete_first_finish(
    PTO2Runtime *, const MixedKernels &mixed,
    const DistCompeteFirstTicket &ticket, const L0TaskArgs &args,
    uint32_t expected_output_count
)
{
    ObservedTaskKind kind{};
    ObservedEngine engine{};
    int32_t function_id = INVALID_KERNEL_ID;
    if (!ResolveKernel(mixed, kind, engine, function_id) ||
        kind != g_backend.active.kind) {
        Fail(AicpuPlanBackendStatus::BadSequence);
        return false;
    }
    return FinishTask(
        ticket, args, expected_output_count, engine, function_id
    );
}

bool dist_is_fatal_query()
{
    return !g_backend.bound ||
           g_backend.result.status !=
               static_cast<int32_t>(AicpuPlanBackendStatus::Ok);
}

void dist_report_fatal_msg(
    int32_t code, const char *, const char *
)
{
    Fail(AicpuPlanBackendStatus::FatalReportedByOrchestration, code);
}

void dist_log_error_msg(const char *, const char *) {}
void dist_log_warn_msg(const char *, const char *) {}
void dist_log_debug_msg(const char *, const char *) {}
void dist_log_info_v_msg(const char *, int, const char *) {}

uint64_t dist_get_tensor_data_impl(
    PTO2Runtime *, const Tensor &tensor, uint32_t ndims,
    const uint32_t indices[]
)
{
    const uint64_t address = TensorScalarAddress(tensor, ndims, indices);
    if (address == 0U) {
        Fail(AicpuPlanBackendStatus::BadTaskArgs);
        return 0U;
    }
    const uint64_t bytes = get_element_size(tensor.dtype);
    if (bytes == 1U) return *reinterpret_cast<const uint8_t *>(address);
    if (bytes == 2U) return *reinterpret_cast<const uint16_t *>(address);
    if (bytes == 4U) return *reinterpret_cast<const uint32_t *>(address);
    return *reinterpret_cast<const uint64_t *>(address);
}

void dist_set_tensor_data_impl(
    PTO2Runtime *, const Tensor &tensor, uint32_t ndims,
    const uint32_t indices[], uint64_t value
)
{
    const uint64_t address = TensorScalarAddress(tensor, ndims, indices);
    if (address == 0U) {
        Fail(AicpuPlanBackendStatus::BadTaskArgs);
        return;
    }
    const uint64_t bytes = get_element_size(tensor.dtype);
    if (bytes == 1U) {
        *reinterpret_cast<uint8_t *>(address) = static_cast<uint8_t>(value);
    } else if (bytes == 2U) {
        *reinterpret_cast<uint16_t *>(address) = static_cast<uint16_t>(value);
    } else if (bytes == 4U) {
        *reinterpret_cast<uint32_t *>(address) = static_cast<uint32_t>(value);
    } else {
        *reinterpret_cast<uint64_t *>(address) = value;
    }
}

void dist_scope_begin_impl(PTO2Runtime *) {}
void dist_scope_end_impl(PTO2Runtime *) {}
void dist_orchestration_done_impl(PTO2Runtime *) {}
void dist_scope_set_site_impl(const char *, int) {}

void dist_perf_clock_expect_submits(uint32_t) {}
void dist_submit_pmu_expect_submits(uint32_t) {}

// fully_distributed orchestration header 的 always_assert 依赖这个真实符号。
// 正常 smoke 不应到达；若 PA 输入破坏其构造不变量，保留 fail-fast，而不是
// 为了让 SO 闭合而把断言伪装成成功。
[[noreturn]] void assert_impl(const char *, const char *, int)
{
    Fail(AicpuPlanBackendStatus::FatalReportedByOrchestration);
    std::abort();
}
