// Multi-core AtomicCAS contention latency benchmark (pure CCEC intrinsic).
//
// Uses only CCEC compiler builtins (no kernel_operator.h, no namespace AscendC):
//   atomicCAS     — raw CCEC atomic CAS
//   get_sys_cnt() — raw CCEC cycle counter
//   ccec_barrier  — atomicAdd + ld_dev based cross-block barrier
//
// Note: A5 SyncAll is a software protocol needing workspace memory (not a
// single hardware instruction), so this probe uses a pure-CCEC atomic
// barrier instead. See ccec_utils.h for details.
//
// Build: see run.sh
#include "cce_aicore_intrinsics.h"
#include "ccec_utils.h"

extern "C" __global__ __aicore__ void KERNEL_ENTRY(atomic_cas_bench)(
    __gm__ uint32_t *gx, uint32_t num_blocks)
{
    int32_t bid = get_block_idx();
    uint32_t nb = num_blocks;

    ccec_barrier(gx, nb, 1);

    int64_t cycle_before = get_sys_cnt();
    uint32_t cas_success = 0;

    for (int32_t i = 0; i < 100; i++) {
        uint32_t old = atomicCAS(&gx[0], (uint32_t)1000, (uint32_t)2000);
        if (old == 1000) cas_success++;
    }

    int64_t cycle_after = get_sys_cnt();
    int64_t num_cycle = cycle_after - cycle_before;

    st_dev_b32(&gx[1 + bid], (uint32_t)num_cycle);
    st_dev_b32(&gx[65 + bid], cas_success);

    ccec_barrier(gx, nb, 2);
}
