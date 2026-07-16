// Host runner for CCEC AtomicCAS contention benchmark.
//
// Loads the ccec-compiled kernel binary, launches with varying block
// counts (1~64), and reports per-block cycle statistics.
//
// Build: see run_atomic_cas.sh
#include "../probe_host.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <time.h>

constexpr uint32_t STORAGE_WORDS = 512;
constexpr uint32_t CYCLE_BASE = 1;
constexpr uint32_t SUCCESS_BASE = 129;
constexpr uint32_t CORES_PER_BLOCK = 1;

struct KernelArgs {
    uint64_t storage_pointer;
    uint32_t num_blocks;
    uint32_t padding;
};

static_assert(sizeof(KernelArgs) == 16, "unexpected CCEC kernel argument ABI");

static void check(aclError err, const char *msg) {
    if (err != ACL_SUCCESS) {
        fprintf(stderr, "ACL error %d: %s\n", err, msg);
        exit(1);
    }
}

static void run_sweep(aclrtFuncHandle fh, aclrtStream stream,
                      void *gx_dev, const uint32_t *blocks_arr,
                      int blocks_count, const char *label, atomic_probe::Result &result) {
    printf("\n--- %s ---\n", label);
    printf("%-12s %12s %12s %12s %12s %10s\n",
           "NUM_BLOCKS", "Host(us)", "Blk0_cycle", "Min_cycle", "Max_cycle", "CAS_ok");

    for (int i = 0; i < blocks_count; i++) {
        uint32_t nblk = blocks_arr[i];

        // Init: gx[0]=1000 (CAS target), gx[30]=0 (barrier), rest=0
        uint32_t init[STORAGE_WORDS] = {0};
        init[0] = 1000;
        // init[30] = 0 (BARRIER_SLOT, already zeroed by = {0})
        check(aclrtMemcpy(gx_dev, sizeof(init), init, sizeof(init),
                          ACL_MEMCPY_HOST_TO_DEVICE), "init");

        struct timespec ts0, ts1;
        clock_gettime(CLOCK_MONOTONIC, &ts0);

        KernelArgs args{(uint64_t)(uintptr_t)gx_dev, nblk, 0};
        check(aclrtLaunchKernelWithHostArgs(fh, nblk, stream, nullptr,
                                            &args, sizeof(args), nullptr, 0),
              "launch");
        check(aclrtSynchronizeStream(stream), "sync");

        clock_gettime(CLOCK_MONOTONIC, &ts1);
        double host_us = (ts1.tv_sec - ts0.tv_sec) * 1e6 +
                         (ts1.tv_nsec - ts0.tv_nsec) / 1e3;

        uint32_t r[STORAGE_WORDS] = {0};
        check(aclrtMemcpy(r, sizeof(r), gx_dev, sizeof(r),
                          ACL_MEMCPY_DEVICE_TO_HOST), "readResult");

        uint32_t min_cyc = 0xFFFFFFFF, max_cyc = 0;
        uint32_t total_cas = 0;
        uint32_t participants = nblk * CORES_PER_BLOCK;
        for (uint32_t participant = 0; participant < participants; participant++) {
            uint32_t cyc = r[CYCLE_BASE + participant];
            if (cyc < min_cyc) min_cyc = cyc;
            if (cyc > max_cyc) max_cyc = cyc;
            total_cas += r[SUCCESS_BASE + participant];
        }

        printf("%-12u %12.1f %12u %12u %12u %10u\n",
               nblk, host_us, r[CYCLE_BASE], min_cyc, max_cyc, total_cas);
        char assertion[80];
        std::snprintf(assertion, sizeof(assertion), "CCEC CAS exact, blocks=%u", nblk);
        result.Expect(r[0] == 2000 && total_cas == 1, assertion);
    }
}

int main(int argc, char *argv[]) {
    const char *kernel_path = "./atomic_cas_kernel.o";
    if (argc > 1) kernel_path = argv[1];

    int32_t deviceId = atomic_probe::DeviceId();
    if (deviceId < 0) return EXIT_FAILURE;
    check(aclInit(nullptr), "aclInit");
    check(aclrtSetDevice(deviceId), "aclrtSetDevice");

    aclrtStream stream;
    check(aclrtCreateStream(&stream), "aclrtCreateStream");

    std::ifstream f(kernel_path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "Cannot open %s\n", kernel_path); return 1; }
    size_t bin_size = f.tellg();
    f.seekg(0);
    std::vector<char> bin_data(bin_size);
    f.read(bin_data.data(), bin_size);
    printf("Kernel: %s (%zu bytes)\n", kernel_path, bin_size);

    aclrtBinHandle binHandle;
    check(atomic_probe::LoadAicoreBinaryFromData(bin_data.data(), bin_size, &binHandle),
          "LoadAicoreBinaryFromData");
    aclrtFuncHandle funcHandle;
    check(aclrtBinaryGetFunctionByEntry(binHandle, 0, &funcHandle),
          "getFunction");

    void *gx_dev = nullptr;
    check(aclrtMalloc(&gx_dev, STORAGE_WORDS * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
          "malloc");

    printf("=== AtomicCAS Contention Latency (CCEC) ===\n");
    printf("100 CAS per block, same GM word\n");
    atomic_probe::Result result;

    // Key calibration points
    uint32_t key_blocks[] = {1, 2, 4, 8, 16, 32, 64};
    run_sweep(funcHandle, stream, gx_dev,
              key_blocks, sizeof(key_blocks) / sizeof(key_blocks[0]),
              "Key calibration points", result);

    // Full 1~64 sweep for curve
    std::vector<uint32_t> full_blocks;
    for (uint32_t b = 1; b <= 64; b++) full_blocks.push_back(b);
    printf("\n--- Full sweep (block 0 cycles) ---\nblocks cycles\n");
    bool sweep_ok = true;
    for (uint32_t b = 1; b <= 64; b++) {
        uint32_t init[STORAGE_WORDS] = {0};
        init[0] = 1000;
        check(aclrtMemcpy(gx_dev, sizeof(init), init, sizeof(init),
                          ACL_MEMCPY_HOST_TO_DEVICE), "init");
        KernelArgs args{(uint64_t)(uintptr_t)gx_dev, b, 0};
        check(aclrtLaunchKernelWithHostArgs(funcHandle, b, stream, nullptr,
                                            &args, sizeof(args), nullptr, 0),
              "launch");
        check(aclrtSynchronizeStream(stream), "sync");
        uint32_t r[STORAGE_WORDS] = {0};
        check(aclrtMemcpy(r, sizeof(r), gx_dev, sizeof(r),
                          ACL_MEMCPY_DEVICE_TO_HOST), "copy");
        uint32_t total_cas = 0;
        for (uint32_t participant = 0; participant < b * CORES_PER_BLOCK; participant++) {
            total_cas += r[SUCCESS_BASE + participant];
        }
        sweep_ok &= r[0] == 2000 && total_cas == 1;
        printf("%u %u\n", b, r[CYCLE_BASE]);
    }
    result.Expect(sweep_ok, "CCEC CAS exact across 1..64 blocks");

    check(aclrtFree(gx_dev), "aclrtFree");
    check(aclrtBinaryUnLoad(binHandle), "aclrtBinaryUnLoad");
    check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    check(aclrtResetDevice(deviceId), "aclrtResetDevice");
    check(aclFinalize(), "aclFinalize");
    return result.ExitCode();
}
