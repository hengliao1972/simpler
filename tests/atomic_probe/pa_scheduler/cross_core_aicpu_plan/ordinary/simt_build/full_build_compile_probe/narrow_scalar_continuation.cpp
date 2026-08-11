/* Scalar-side half of the dual-TU identity/link gate. */

#include "cce_aicore_intrinsics.h"

#include <cstdint>

#define PA_DEVICE __aicore__ inline
#define PA_DEVICE_NOINLINE static __aicore__ __attribute__((noinline))
#define PA_LOOP_NOUNROLL _Pragma("clang loop unroll(disable)")
#define PA_GM __gm__

// 先以真实 Scalar 身份实例化现有 scheduler core。这个 TU 不 include
// SIMT 窄头；反向地，AIV/VF TU 也不 include 本 Scalar core。
#include "../../scalar_build/common/pa_scheduler_core.h"

extern "C" __aicore__ __attribute__((noinline, visibility("hidden")))
void aicpu_plan_narrow_scalar_continuation(
    __gm__ uint64_t *reports, uint32_t task_count
)
{
    // compile gate 用现有 scheduler 的 completion 公式证明本符号确实由
    // Scalar core 实例化而来。production 会把这里替换为 execute-only
    // scheduler continuation，不改变 AIV/VF TU 的 include 身份。
    const int64_t initial =
        pa_scheduler::SharedInsertCompletionInitialValue(task_count);
    constexpr uint32_t kBuilderLeaderReports = 4U;
    reports[kBuilderLeaderReports] = static_cast<uint64_t>(initial);
}
