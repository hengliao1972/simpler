// Multi-core AtomicCAS contention latency benchmark (pure CCEC intrinsic).
//
// Uses only CCEC compiler builtins (no kernel_operator.h, no namespace AscendC):
//   atomicCAS     — raw CCEC atomic CAS
//   get_sys_cnt() — raw CCEC cycle counter
//
// See ccec_utils.h for the dav-3510 AscendC-to-CCEC mapping.
//
// Build: see run.sh
#include "ccec_utils.h"

CCEC_PROBE_KERNEL_META(atomic_cas_bench);

constexpr uint32_t CYCLE_BASE = 1;
constexpr uint32_t SUCCESS_BASE = 129;

extern "C" __global__ __aicore__ void KERNEL_ENTRY(atomic_cas_bench)(
    __gm__ uint32_t *gx, uint32_t num_blocks)
{
    (void)num_blocks;
    uint32_t participant = ccec_participant_id((uint32_t)get_block_idx());

    int64_t cycle_before = get_sys_cnt();
    uint32_t cas_success = 0;

    for (int32_t i = 0; i < 100; i++) {
        uint32_t old = atomicCAS(&gx[0], (uint32_t)1000, (uint32_t)2000);
        if (old == 1000) cas_success++;
    }

    int64_t cycle_after = get_sys_cnt();
    int64_t num_cycle = cycle_after - cycle_before;

    st_dev_b32(&gx[CYCLE_BASE + participant], (uint32_t)num_cycle);
    st_dev_b32(&gx[SUCCESS_BASE + participant], cas_success);
    // The host consumes these exact result slots immediately after launch
    // completion, so this probe explicitly completes their publication.
    dsb(DSB_ALL);
}
