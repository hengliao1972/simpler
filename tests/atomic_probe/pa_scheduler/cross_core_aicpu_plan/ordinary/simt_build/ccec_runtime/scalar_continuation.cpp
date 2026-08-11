/* Independent Scalar-identity continuation for the SIMT Build runtime gate. */

#include "cce_aicore_intrinsics.h"

#include <cstdint>

#define PA_DEVICE __aicore__ inline
#define PA_GM __gm__

#include "../../scalar_build/common/pa_model.h"
#include "runtime_report.h"

namespace gate = pa_scheduler::aicpu_plan_simt::runtime_gate;

extern "C" __aicore__ __attribute__((noinline, visibility("hidden")))
void aicpu_plan_simt_scalar_continuation(
    __gm__ pa_scheduler::SchedulerState *state,
    __gm__ gate::RuntimeReport *report,
    uint32_t task_count, uint64_t validation,
    uint64_t join_clock
)
{
    int64_t release = pa_scheduler::aicpu_plan::kBuildReleaseFailed;
    int64_t plan_fatal = 1;
    int32_t scheduler_fatal = 1;
    if (state != nullptr) {
        release = atomicAdd(
            const_cast<__gm__ int64_t *>(
                &state->runtime_plan_control.build_release.value
            ), int64_t{0}
        );
        plan_fatal = atomicAdd(
            const_cast<__gm__ int64_t *>(
                &state->runtime_plan_control.fatal.value
            ), int64_t{0}
        );
        scheduler_fatal = atomicAdd(
            const_cast<__gm__ int32_t *>(&state->fatal.value),
            int32_t{0}
        );
    }

    if (report == nullptr) return;
    __gm__ uint64_t *words = report->continuation.words;
    for (uint32_t word = 0U; word < gate::kReportWords; ++word) {
        __builtin_cce_st_dev(uint64_t{0}, words + word, 0);
    }
    __builtin_cce_st_dev(
        static_cast<uint64_t>(get_sys_cnt()),
        words + gate::ContinuationClock, 0
    );
    __builtin_cce_st_dev(
        static_cast<uint64_t>(task_count),
        words + gate::ContinuationTaskCount, 0
    );
    __builtin_cce_st_dev(
        validation, words + gate::ContinuationValidation, 0
    );
    __builtin_cce_st_dev(
        static_cast<uint64_t>(release),
        words + gate::ContinuationReleaseObserved, 0
    );
    __builtin_cce_st_dev(
        static_cast<uint64_t>(plan_fatal),
        words + gate::ContinuationPlanFatalObserved, 0
    );
    __builtin_cce_st_dev(
        static_cast<uint64_t>(scheduler_fatal),
        words + gate::ContinuationSchedulerFatalObserved, 0
    );
    __builtin_cce_st_dev(
        join_clock, words + gate::ContinuationJoinClock, 0
    );
    dsb((mem_dsb_t)0);
    __builtin_cce_st_dev(
        gate::kContinuationMagic,
        words + gate::ContinuationMagic, 0
    );
    dsb((mem_dsb_t)0);
}
