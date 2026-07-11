// Host runner for CCEC AtomicCAS contention benchmark.
//
// Loads the ccec-compiled kernel binary, launches with varying block
// counts (1~64), and reports per-block cycle statistics.
//
// Build: see run_atomic_cas.sh
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <time.h>

static void check(aclError err, const char *msg) {
    if (err != ACL_SUCCESS) {
        fprintf(stderr, "ACL error %d: %s\n", err, msg);
        exit(1);
    }
}

static void run_sweep(aclrtFuncHandle fh, aclrtStream stream,
                      void *gx_dev, const uint32_t *blocks_arr,
                      int blocks_count, const char *label) {
    printf("\n--- %s ---\n", label);
    printf("%-12s %12s %12s %12s %12s %10s\n",
           "NUM_BLOCKS", "Host(us)", "Blk0_cycle", "Min_cycle", "Max_cycle", "CAS_ok");

    for (int i = 0; i < blocks_count; i++) {
        uint32_t nblk = blocks_arr[i];

        // Init: gx[0]=1000 (CAS target), gx[30]=0 (barrier), rest=0
        uint32_t init[256] = {0};
        init[0] = 1000;
        // init[30] = 0 (BARRIER_SLOT, already zeroed by = {0})
        check(aclrtMemcpy(gx_dev, sizeof(init), init, sizeof(init),
                          ACL_MEMCPY_HOST_TO_DEVICE), "init");

        struct timespec ts0, ts1;
        clock_gettime(CLOCK_MONOTONIC, &ts0);

        uint64_t gx_ptr = (uint64_t)(uintptr_t)gx_dev;
        uint64_t args[2] = {gx_ptr, (uint64_t)nblk};
        aclrtArgsHandle ah;
        check(aclrtKernelArgsInit(fh, &ah), "argsInit");
        aclrtParamHandle ph;
        check(aclrtKernelArgsAppend(ah, args, sizeof(args), &ph), "argsAppend");
        check(aclrtKernelArgsFinalize(ah), "argsFinalize");
        check(aclrtLaunchKernelWithConfig(fh, nblk, stream, nullptr, ah, nullptr),
              "launch");
        check(aclrtSynchronizeStream(stream), "sync");

        clock_gettime(CLOCK_MONOTONIC, &ts1);
        double host_us = (ts1.tv_sec - ts0.tv_sec) * 1e6 +
                         (ts1.tv_nsec - ts0.tv_nsec) / 1e3;

        uint32_t r[256] = {0};
        check(aclrtMemcpy(r, sizeof(r), gx_dev, sizeof(r),
                          ACL_MEMCPY_DEVICE_TO_HOST), "readResult");

        uint32_t min_cyc = 0xFFFFFFFF, max_cyc = 0;
        uint32_t total_cas = 0;
        for (uint32_t b = 0; b < nblk; b++) {
            uint32_t cyc = r[1 + b];
            if (cyc < min_cyc) min_cyc = cyc;
            if (cyc > max_cyc) max_cyc = cyc;
            total_cas += r[65 + b];
        }

        printf("%-12u %12.1f %12u %12u %12u %10u\n",
               nblk, host_us, r[1], min_cyc, max_cyc, total_cas);
    }
}

int main(int argc, char *argv[]) {
    const char *kernel_path = "./atomic_cas_kernel.o";
    if (argc > 1) kernel_path = argv[1];

    check(aclInit(nullptr), "aclInit");
    check(aclrtSetDevice(0), "aclrtSetDevice");

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
    check(aclrtBinaryLoadFromData(bin_data.data(), bin_size, nullptr, &binHandle),
          "loadBinary");
    aclrtFuncHandle funcHandle;
    check(aclrtBinaryGetFunctionByEntry(binHandle, 0, &funcHandle),
          "getFunction");

    void *gx_dev = nullptr;
    check(aclrtMalloc(&gx_dev, 256 * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
          "malloc");

    printf("=== AtomicCAS Contention Latency (CCEC) ===\n");
    printf("100 CAS per block, same GM word\n");

    // Key calibration points
    uint32_t key_blocks[] = {1, 2, 4, 8, 16, 32, 64};
    run_sweep(funcHandle, stream, gx_dev,
              key_blocks, sizeof(key_blocks) / sizeof(key_blocks[0]),
              "Key calibration points");

    // Full 1~64 sweep for curve
    std::vector<uint32_t> full_blocks;
    for (uint32_t b = 1; b <= 64; b++) full_blocks.push_back(b);
    printf("\n--- Full sweep (block 0 cycles) ---\nblocks cycles\n");
    for (uint32_t b = 1; b <= 64; b++) {
        uint32_t init[256] = {0};
        init[0] = 1000;
        check(aclrtMemcpy(gx_dev, sizeof(init), init, sizeof(init),
                          ACL_MEMCPY_HOST_TO_DEVICE), "init");
        uint64_t gx_ptr = (uint64_t)(uintptr_t)gx_dev;
        uint64_t args[2] = {gx_ptr, (uint64_t)b};
        aclrtArgsHandle ah;
        check(aclrtKernelArgsInit(funcHandle, &ah), "init");
        aclrtParamHandle ph;
        check(aclrtKernelArgsAppend(ah, args, sizeof(args), &ph), "append");
        check(aclrtKernelArgsFinalize(ah), "finalize");
        check(aclrtLaunchKernelWithConfig(funcHandle, b, stream, nullptr, ah, nullptr),
              "launch");
        check(aclrtSynchronizeStream(stream), "sync");
        uint32_t r[256] = {0};
        check(aclrtMemcpy(r, sizeof(r), gx_dev, sizeof(r),
                          ACL_MEMCPY_DEVICE_TO_HOST), "copy");
        printf("%u %u\n", b, r[1]);
    }

    aclrtFree(gx_dev);
    aclrtBinaryUnLoad(binHandle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
