/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: CANN-2.0
 */
#ifndef PA_SCHEDULER_AICPU_PLAN_PROTOCOL_PROBE_SHARED_H_
#define PA_SCHEDULER_AICPU_PLAN_PROTOCOL_PROBE_SHARED_H_

#include <cstddef>
#include <cstdint>

namespace plan_protocol_probe {

#if defined(__CCE_AICORE__)
#define PLAN_PROTOCOL_HD __aicore__
#else
#define PLAN_PROTOCOL_HD
#endif

constexpr uint64_t kProbeMagic = UINT64_C(0x50354135504c414e);
constexpr uint64_t kProducerResultMagic = UINT64_C(0x50524f4455434552);
constexpr uint64_t kConsumerResultMagic = UINT64_C(0x434f4e53554d4552);
constexpr uint32_t kProbeVersion = 2U;
constexpr uint32_t kRuntimePlanAbiVersion = 2U;
constexpr uint32_t kCacheLineBytes = 64U;
constexpr uint32_t kAtomicIsolationBytes = 128U;
constexpr uint32_t kMaxTasks = 1280U;
constexpr uint32_t kMaxTaskTensors = 32U;
constexpr uint32_t kMaxTaskScalars = 16U;
constexpr uint32_t kMaxExplicitDependencies = 16U;
constexpr uint32_t kTensorDescWords = 16U;
constexpr uint32_t kPlanHeaderWords = 8U;
constexpr uint32_t kMaxPayloadWords =
    kPlanHeaderWords + kMaxTaskTensors * kTensorDescWords +
    kMaxTaskScalars + kMaxExplicitDependencies;
constexpr uint32_t kMaxPayloadBytes = kMaxPayloadWords * sizeof(uint64_t);
constexpr uint32_t kMaxPayloadLines = kMaxPayloadBytes / kCacheLineBytes;
constexpr uint32_t kCellBytes = 4608U;
constexpr int64_t kPlanOpen = -1;
constexpr uint32_t kInvalidFunctionId = UINT32_MAX;
constexpr uint32_t kNoFaultTask = UINT32_MAX;

static_assert(kMaxPayloadBytes == 4416U, "max Plan payload must be 4416 bytes");
static_assert(kMaxPayloadLines == 69U, "max Plan payload must be 69 lines");

enum class Workload : uint32_t {
    G0 = 0U,
    G1 = 1U,
    Mixed = 2U,
    B256 = 3U,
};

enum class Publication : uint32_t {
    CloseOnly = 0U,
    PerItemFrontier = 1U,
};

enum class FaultMode : uint32_t {
    None = 0U,
    SkipLastPayloadCleanAndPoison = 1U,
};

enum class EngineClass : uint8_t {
    MetadataOnly = 0U,
    Aic = 1U,
    Aiv = 2U,
};

enum RuntimeTaskFlags : uint8_t {
    HasFollowing = 1U << 0U,
    LastInBatch = 1U << 1U,
    LastInPlan = 1U << 2U,
};

enum class PlanCellPhase : uint8_t {
    Empty = 0U,
    Published = 1U,
};

enum class Status : uint32_t {
    Ok = 0U,
    BadArguments = 1U,
    BadConfig = 2U,
    BadTaskCount = 3U,
    ProducerFailure = 4U,
    Timeout = 5U,
    FrontierRegression = 6U,
    FrontierOvershoot = 7U,
    CloseMismatch = 8U,
    CellControlMismatch = 9U,
    PayloadMismatch = 10U,
    RemoteFatal = 11U,
};

// Byte-for-byte local copy of the public Plan header/storage/cell layout.
struct RuntimeTaskPlanHeader {
    uint32_t task_id;
    uint32_t function_id;
    uint16_t tensor_count;
    uint16_t scalar_count;
    uint16_t explicit_dep_count;
    uint16_t output_count;
    uint8_t engine_class;
    uint8_t flags;
    int16_t core_num;
    uint8_t require_sync_start;
    uint8_t reserved0;
    uint16_t adapter_data;
    uint8_t tensor_tags[kMaxTaskTensors];
    uint32_t tensor_reference_mask;
    uint32_t abi_version;
};

struct alignas(kCacheLineBytes) RuntimeTaskPlanStorage {
    volatile uint64_t words[kMaxPayloadWords];
};

struct alignas(kCacheLineBytes) PlanAtomicLine {
    volatile int64_t value;
    uint8_t isolation_padding[kAtomicIsolationBytes - sizeof(int64_t)];
};

struct alignas(kCacheLineBytes) RuntimeTaskPlanCell {
    PlanAtomicLine control;
    RuntimeTaskPlanStorage payload;
    uint8_t tail_padding[kCacheLineBytes];
};

struct alignas(kCacheLineBytes) ProbeConfig {
    uint64_t magic;
    uint32_t version;
    uint32_t workload;
    uint32_t publication;
    uint32_t expected_tasks;
    uint64_t nonce;
    uint64_t timeout_ticks;
    uint32_t producer_delay_nops;
    uint32_t fault_mode;
    uint32_t fault_task;
    uint32_t reserved0;
    uint64_t reserved1;
};

struct alignas(kCacheLineBytes) ProducerResult {
    uint64_t magic;
    uint32_t status;
    uint32_t task_count;
    uint64_t begin_ns;
    uint64_t end_ns;
    uint64_t payload_clean_lines;
    uint64_t payload_publish_barriers;
    uint64_t control_clean_lines;
    uint64_t omitted_clean_lines;
};

struct alignas(kCacheLineBytes) ConsumerResult {
    uint64_t magic;
    uint32_t status;
    uint32_t task_count;
    uint64_t begin_ticks;
    uint64_t end_ticks;
    uint64_t control_observations;
    uint64_t payload_invalidated_lines;
    uint64_t checksum;
    uint32_t first_bad_task;
    uint32_t first_bad_word;
};

struct alignas(kAtomicIsolationBytes) ProbeState {
    ProbeConfig config;
    ProducerResult producer;
    ConsumerResult consumer;
    uint8_t ordinary_padding[kCacheLineBytes];
    PlanAtomicLine planned_frontier;
    PlanAtomicLine closed_task_count;
    PlanAtomicLine fatal;
    RuntimeTaskPlanCell cells[kMaxTasks];
};

struct PlanAicpuKernelArgs {
    uint64_t unused[5];
    uint64_t device_args_device;
    uint64_t runtime_args_device;
    uint64_t register_bases_device;
    uint64_t dump_data_base;
    uint64_t l2_swimlane_data_base;
    uint64_t pmu_data_base;
    uint64_t dep_gen_data_base;
    uint64_t l2_swimlane_rotation_table;
    uint64_t aicore_pmu_ring_addrs;
    uint64_t scope_stats_data_base;
    uint32_t log_level;
    uint32_t log_info_v;
    uint32_t command;
    uint32_t reserved_alignment;
    uint64_t device_wall_data_base;
    uint32_t device_id;
    uint32_t force_simt_anchor;
};

struct AivKernelArgs {
    uint64_t state_device;
};

struct TaskShape {
    uint32_t task_id;
    uint32_t batch;
    uint32_t batch_start;
    uint32_t group;
    uint32_t context;
    uint32_t kind;
    uint32_t group_blocks;
    uint32_t final_count;
    bool last_in_batch;
    bool has_following;
};

struct PayloadLayout {
    uint32_t tensor_count;
    uint32_t scalar_count;
    uint32_t explicit_dep_count;
    uint32_t payload_words;
    uint32_t payload_lines;
};

struct DecodedCellControl {
    uint32_t phase;
    uint32_t payload_lines;
    uint32_t task_id;
    bool valid;
};

static_assert(sizeof(RuntimeTaskPlanHeader) == 64U, "Plan header layout changed");
static_assert(offsetof(RuntimeTaskPlanHeader, adapter_data) == 22U, "Plan adapter-data offset changed");
static_assert(offsetof(RuntimeTaskPlanHeader, tensor_tags) == 24U, "Plan tag offset changed");
static_assert(offsetof(RuntimeTaskPlanHeader, tensor_reference_mask) == 56U, "Plan mask offset changed");
static_assert(sizeof(RuntimeTaskPlanStorage) == 4416U, "Plan storage layout changed");
static_assert(sizeof(PlanAtomicLine) == 128U, "Plan control isolation changed");
static_assert(offsetof(RuntimeTaskPlanCell, payload) == 128U, "Plan payload offset changed");
static_assert(sizeof(RuntimeTaskPlanCell) == kCellBytes, "Plan cell stride changed");
static_assert(sizeof(ProbeConfig) == 64U, "config must be one line");
static_assert(sizeof(ProducerResult) == 64U, "producer result must be one line");
static_assert(sizeof(ConsumerResult) == 64U, "consumer result must be one line");
static_assert(offsetof(ProbeState, planned_frontier) == 256U, "frontier offset changed");
static_assert(offsetof(ProbeState, closed_task_count) == 384U, "close offset changed");
static_assert(offsetof(ProbeState, fatal) == 512U, "fatal offset changed");
static_assert(offsetof(ProbeState, cells) == 640U, "cell base changed");
static_assert((offsetof(ProbeState, cells) % kAtomicIsolationBytes) == 0U, "cell controls not isolated");
static_assert((kCellBytes % kAtomicIsolationBytes) == 0U, "successive controls not isolated");
static_assert(sizeof(PlanAicpuKernelArgs) == 152U, "AICPU KernelArgs ABI changed");
static_assert(offsetof(PlanAicpuKernelArgs, runtime_args_device) == 48U, "state offset changed");
static_assert(offsetof(PlanAicpuKernelArgs, command) == 128U, "command offset changed");
static_assert(sizeof(AivKernelArgs) == 8U, "AIV arguments changed");

constexpr uint64_t kPlanPhaseMask = UINT64_C(0x3);
constexpr uint32_t kPlanPayloadLinesShift = 2U;
constexpr uint64_t kPlanPayloadLinesMask = UINT64_C(0x7f);
constexpr uint32_t kPlanTaskIdShift = 9U;
constexpr uint64_t kPlanTaskIdMask = UINT64_C(0xffffffff);
constexpr uint64_t kPlanKnownMask =
    kPlanPhaseMask | (kPlanPayloadLinesMask << kPlanPayloadLinesShift) |
    (kPlanTaskIdMask << kPlanTaskIdShift);

PLAN_PROTOCOL_HD inline uint64_t EncodeCellControl(uint32_t payload_lines, uint32_t task_id)
{
    return static_cast<uint64_t>(PlanCellPhase::Published) |
        (static_cast<uint64_t>(payload_lines) << kPlanPayloadLinesShift) |
        (static_cast<uint64_t>(task_id) << kPlanTaskIdShift);
}

PLAN_PROTOCOL_HD inline DecodedCellControl DecodeCellControl(int64_t raw_state)
{
    const uint64_t raw = static_cast<uint64_t>(raw_state);
    DecodedCellControl decoded{
        static_cast<uint32_t>(raw & kPlanPhaseMask),
        static_cast<uint32_t>((raw >> kPlanPayloadLinesShift) & kPlanPayloadLinesMask),
        static_cast<uint32_t>((raw >> kPlanTaskIdShift) & kPlanTaskIdMask),
        false,
    };
    decoded.valid = (raw & ~kPlanKnownMask) == 0U &&
        decoded.phase == static_cast<uint32_t>(PlanCellPhase::Published) &&
        decoded.payload_lines >= 1U && decoded.payload_lines <= kMaxPayloadLines;
    return decoded;
}

PLAN_PROTOCOL_HD inline uint32_t BatchCount(Workload workload)
{
    if (workload == Workload::Mixed) return 4U;
    if (workload == Workload::B256) return 256U;
    return 1U;
}

PLAN_PROTOCOL_HD inline uint32_t ContextLength(Workload workload, uint32_t batch)
{
    if (workload == Workload::G0) return 0U;
    if (workload == Workload::G1 || workload == Workload::B256) return 8192U;
    constexpr uint32_t mixed[4] = {0U, 1U, 8192U, 8193U};
    return batch < 4U ? mixed[batch] : 0U;
}

PLAN_PROTOCOL_HD inline uint32_t GroupCount(uint32_t context)
{
    const uint32_t blocks = (context + 127U) / 128U;
    return (blocks + 63U) / 64U;
}

PLAN_PROTOCOL_HD inline uint32_t TaskCount(Workload workload)
{
    uint32_t count = 0U;
    for (uint32_t batch = 0U; batch < BatchCount(workload); ++batch) {
        count += 1U + 4U * GroupCount(ContextLength(workload, batch));
    }
    return count;
}

PLAN_PROTOCOL_HD inline bool ResolveTask(Workload workload, uint32_t wanted, TaskShape *shape)
{
    if (shape == nullptr) return false;
    const uint32_t final_count = TaskCount(workload);
    uint32_t task = 0U;
    for (uint32_t batch = 0U; batch < BatchCount(workload); ++batch) {
        const uint32_t context = ContextLength(workload, batch);
        const uint32_t blocks = (context + 127U) / 128U;
        const uint32_t groups = GroupCount(context);
        const uint32_t batch_start = task;
        const uint32_t batch_tasks = 1U + 4U * groups;
        for (uint32_t ordinal = 0U; ordinal < batch_tasks; ++ordinal, ++task) {
            if (task != wanted) continue;
            const uint32_t kind = ordinal == 0U ? 0U : 1U + ((ordinal - 1U) & 3U);
            const uint32_t group = ordinal == 0U ? 0U : (ordinal - 1U) / 4U;
            const uint32_t consumed = group * 64U;
            const uint32_t group_blocks = ordinal == 0U ? 0U :
                ((blocks - consumed) < 64U ? blocks - consumed : 64U);
            *shape = TaskShape{
                task, batch, batch_start, group, context, kind, group_blocks,
                final_count, ordinal + 1U == batch_tasks,
                kind == 4U && group + 1U < groups,
            };
            return true;
        }
    }
    return false;
}

// Kind 0 is the one-line metadata-only boundary.  Kind 1 deliberately uses
// the complete 32-tensor + 16-scalar + 16-dependency bound (69 lines).  The
// remaining executable kinds give intermediate exact-range sizes.
PLAN_PROTOCOL_HD inline PayloadLayout LayoutForKind(uint32_t kind)
{
    uint32_t tensors = 0U;
    uint32_t scalars = 0U;
    uint32_t dependencies = 0U;
    if (kind == 1U || kind == 4U) {
        tensors = 32U;
        scalars = 16U;
        dependencies = 16U;
    } else if (kind == 2U) {
        tensors = 1U;
    } else if (kind == 3U) {
        tensors = 2U;
        scalars = 4U;
    }
    const uint32_t words = kPlanHeaderWords + tensors * kTensorDescWords +
        scalars + dependencies;
    return PayloadLayout{tensors, scalars, dependencies, words, (words + 7U) / 8U};
}

// ABI2 的 adapter_data 是算子 adapter 自定义的 16-bit provenance。
// Probe 不复用 PA 语义，而是为每个 task 生成非零、可由独立 consumer
// oracle 重算的值，确保 wire word2 的高 16 bit 真正跨 AICPU/AIV 传递。
PLAN_PROTOCOL_HD inline uint16_t AdapterDataForShape(const TaskShape &shape)
{
    const uint32_t mixed = shape.task_id * 37U + shape.batch_start * 11U +
        shape.batch * 7U + shape.group * 3U + shape.kind;
    return static_cast<uint16_t>(1U + mixed % UINT16_MAX);
}

PLAN_PROTOCOL_HD inline uint64_t HeaderWord(
    const TaskShape &shape, const PayloadLayout &layout, uint32_t word)
{
    const uint32_t function_id = shape.kind == 0U ? kInvalidFunctionId : shape.kind;
    const uint8_t engine = static_cast<uint8_t>(
        shape.kind == 0U ? EngineClass::MetadataOnly :
        ((shape.kind == 1U || shape.kind == 3U) ? EngineClass::Aic : EngineClass::Aiv));
    const uint8_t flags = static_cast<uint8_t>(
        (shape.has_following ? static_cast<uint32_t>(HasFollowing) : 0U) |
        (shape.last_in_batch ? static_cast<uint32_t>(LastInBatch) : 0U) |
        (shape.task_id + 1U == shape.final_count
            ? static_cast<uint32_t>(LastInPlan) : 0U));
    if (word == 0U) return static_cast<uint64_t>(shape.task_id) |
        (static_cast<uint64_t>(function_id) << 32U);
    if (word == 1U) return static_cast<uint64_t>(layout.tensor_count) |
        (static_cast<uint64_t>(layout.scalar_count) << 16U) |
        (static_cast<uint64_t>(layout.explicit_dep_count) << 32U);
    if (word == 2U) return static_cast<uint64_t>(engine) |
        (static_cast<uint64_t>(flags) << 8U) | (UINT64_C(1) << 16U) |
        (static_cast<uint64_t>(AdapterDataForShape(shape)) << 48U);
    if (word >= 3U && word <= 6U) {
        uint64_t tags = 0U;
        const uint32_t base = (word - 3U) * 8U;
        for (uint32_t index = 0U; index < 8U; ++index) {
            if (base + index < layout.tensor_count) {
                tags |= UINT64_C(0) << (index * 8U);  // TensorTag::Input
            }
        }
        return tags;
    }
    if (word == 7U) return static_cast<uint64_t>(kRuntimePlanAbiVersion) << 32U;
    return 0U;
}

PLAN_PROTOCOL_HD inline uint64_t ExpectedPayloadWordForShape(
    const TaskShape &shape, const PayloadLayout &layout,
    uint64_t nonce, uint32_t word)
{
    if (word >= layout.payload_lines * 8U) return UINT64_MAX;
    if (word < kPlanHeaderWords) {
        if (word == 0U) {
            const uint32_t function_id = shape.kind == 0U ? kInvalidFunctionId : shape.kind;
            return static_cast<uint64_t>(shape.task_id) |
                (static_cast<uint64_t>(function_id) << 32U);
        }
        return HeaderWord(shape, layout, word);
    }
    if (word >= layout.payload_words) return 0U;
    const uint64_t domain = word < kPlanHeaderWords + layout.tensor_count * kTensorDescWords
        ? UINT64_C(0x1100000000000000)
        : (word < kPlanHeaderWords + layout.tensor_count * kTensorDescWords + layout.scalar_count
            ? UINT64_C(0x2200000000000000) : UINT64_C(0x3300000000000000));
    uint64_t value = domain ^ (nonce * UINT64_C(0x9e3779b97f4a7c15));
    value ^= static_cast<uint64_t>(shape.task_id) << 32U;
    value ^= static_cast<uint64_t>(shape.batch) << 24U;
    value ^= static_cast<uint64_t>(shape.group) << 16U;
    value ^= static_cast<uint64_t>(shape.context);
    value ^= static_cast<uint64_t>(shape.group_blocks) << 48U;
    value ^= static_cast<uint64_t>(word) * UINT64_C(0x100000001b3);
    return value;
}

// The canonical task identity is derived from workload/task id, never from
// the GM payload being checked.
PLAN_PROTOCOL_HD inline uint64_t ExpectedPayloadWord(
    Workload workload, uint32_t task_id, uint64_t nonce, uint32_t word)
{
    TaskShape shape{};
    if (!ResolveTask(workload, task_id, &shape)) return UINT64_MAX;
    return ExpectedPayloadWordForShape(shape, LayoutForKind(shape.kind), nonce, word);
}

PLAN_PROTOCOL_HD inline uint32_t PayloadLinesForTask(Workload workload, uint32_t task_id)
{
    TaskShape shape{};
    return ResolveTask(workload, task_id, &shape) ? LayoutForKind(shape.kind).payload_lines : 0U;
}

PLAN_PROTOCOL_HD inline uint64_t TotalPayloadLines(Workload workload)
{
    uint64_t lines = 0U;
    for (uint32_t task = 0U; task < TaskCount(workload); ++task) {
        lines += PayloadLinesForTask(workload, task);
    }
    return lines;
}

PLAN_PROTOCOL_HD inline uint64_t MixChecksum(uint64_t checksum, uint64_t value)
{
    checksum ^= value;
    checksum *= UINT64_C(1099511628211);
    checksum ^= checksum >> 29U;
    return checksum;
}

inline const char *WorkloadName(Workload workload)
{
    if (workload == Workload::G0) return "G0";
    if (workload == Workload::G1) return "G1";
    if (workload == Workload::Mixed) return "mixed";
    return "B256";
}

inline const char *PublicationName(Publication publication)
{
    return publication == Publication::CloseOnly ? "close-only" : "per-item-frontier";
}

#undef PLAN_PROTOCOL_HD

}  // namespace plan_protocol_probe

#endif  // PA_SCHEDULER_AICPU_PLAN_PROTOCOL_PROBE_SHARED_H_
