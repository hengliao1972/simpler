// AtomicCas latency probe — ccec -x cce version of tests/atomic.asc.
//
// All blocks race on the same GM word via AtomicCas, measuring per-block
// cycle count to characterize cross-core atomic contention scaling.
// Mirrors the .asc probe's logic (SyncAll → 100× AtomicCas loop → cycle
// delta) but compiled with ccec --cce-aicore-only and launched via ACL
// aclrtBinaryLoadFromData + aclrtLaunchKernelWithConfig.
//
// See run.sh for build & run instructions.
#include "kernel_operator.h"
using namespace AscendC;

#ifdef __DAV_VEC__
#define KERNEL_ENTRY(x) x##_0_mix_aiv
#else
#define KERNEL_ENTRY(x) x##_0_mix_aic
#endif

// result layout: [num_blocks] slots, each {cycles, cas_count}
//   result[bid*2]   = cycle delta for block bid
//   result[bid*2+1] = actual CAS iterations completed
extern "C" __global__ __aicore__ void KERNEL_ENTRY(atomic_cas_probe)(
    __gm__ uint32_t *gx, __gm__ int64_t *result, uint32_t num_blocks)
{
    (void)num_blocks;

    SyncAll();
    int64_t cycle_before = GetSystemCycle();

    uint32_t cas_count = 0;
    for (int i = 0; i < 100; i++) {
        auto t = AtomicCas(gx, (uint32_t)1000, (uint32_t)2000);
        if (t == 22222) break;
        cas_count++;
    }

    int64_t cycle_after = GetSystemCycle();
    int64_t num_cycle = cycle_after - cycle_before;

    int64_t bid = GetBlockIdx();
    result[bid * 2] = num_cycle;
    result[bid * 2 + 1] = (int64_t)cas_count;
}
