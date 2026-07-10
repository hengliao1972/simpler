// Host-side runner for the ccec AtomicCas probe.
//
// Loads the linked AICore binary, launches it with varying block counts
// (1–64), and prints per-block cycle measurements alongside host-side
// wall-clock time. Equivalent to the main() in tests/atomic.asc.
//
// Build: see run.sh (g++ + libascendcl)
// Run:   ./atomic_cas_host <kernel.o>  (default: ./atomic_cas_kernel.o)
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

// Max blocks we support result storage for (must match kernel's result layout).
static constexpr int MAX_BLOCKS = 64;

static void run_sweep(aclrtFuncHandle funcHandle, aclrtStream stream,
                      void *gx_dev, void *result_dev,
                      const uint32_t *blocks_arr, int blocks_count,
                      const char *label) {
    printf("\n--- %s ---\n", label);
    printf("%-12s %12s %14s %14s %10s\n",
           "NUM_BLOCKS", "Host(us)", "DevCycles", "DevTime(us)", "CAS/block");
    for (int i = 0; i < blocks_count; i++) {
        uint32_t numBlocks = blocks_arr[i];

        uint32_t initVal = 1000;
        check(aclrtMemcpy(gx_dev, sizeof(uint32_t), &initVal, sizeof(uint32_t),
                          ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy init gx");
        int64_t zero_result[MAX_BLOCKS * 2] = {0};
        check(aclrtMemcpy(result_dev, sizeof(zero_result), zero_result, sizeof(zero_result),
                          ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy init result");

        struct timespec ts_start, ts_end;
        clock_gettime(CLOCK_MONOTONIC, &ts_start);

        uint64_t gx_ptr = (uint64_t)(uintptr_t)gx_dev;
        uint64_t result_ptr = (uint64_t)(uintptr_t)result_dev;
        uint64_t args[3] = {gx_ptr, result_ptr, (uint64_t)numBlocks};

        aclrtArgsHandle argsHandle;
        check(aclrtKernelArgsInit(funcHandle, &argsHandle), "aclrtKernelArgsInit");
        aclrtParamHandle paramHandle;
        check(aclrtKernelArgsAppend(argsHandle, args, sizeof(args), &paramHandle),
              "aclrtKernelArgsAppend");
        check(aclrtKernelArgsFinalize(argsHandle), "aclrtKernelArgsFinalize");

        check(aclrtLaunchKernelWithConfig(funcHandle, numBlocks, stream, nullptr,
                                          argsHandle, nullptr),
              "aclrtLaunchKernelWithConfig");
        check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");

        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        double host_us = (ts_end.tv_sec - ts_start.tv_sec) * 1e6 +
                         (ts_end.tv_nsec - ts_start.tv_nsec) / 1e3;

        int64_t result_host[MAX_BLOCKS * 2] = {0};
        check(aclrtMemcpy(result_host, sizeof(result_host), result_dev, sizeof(result_host),
                          ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy result");

        // Block 0 is representative (all blocks start simultaneously via SyncAll).
        int64_t num_cycle = result_host[0];
        int64_t cas_count = result_host[1];
        int64_t cycle_to_time_base = 1000;
        int64_t num_time = num_cycle / cycle_to_time_base;

        printf("%-12u %12.1f %14ld %14ld %10ld\n",
               numBlocks, host_us, num_cycle, num_time, cas_count);
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
    printf("Kernel binary: %s (%zu bytes)\n", kernel_path, bin_size);

    aclrtBinHandle binHandle;
    check(aclrtBinaryLoadFromData(bin_data.data(), bin_size, nullptr, &binHandle),
          "aclrtBinaryLoadFromData");

    aclrtFuncHandle funcHandle;
    check(aclrtBinaryGetFunctionByEntry(binHandle, 0, &funcHandle),
          "aclrtBinaryGetFunctionByEntry");

    void *gx_dev = nullptr;
    void *result_dev = nullptr;
    check(aclrtMalloc(&gx_dev, 32, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc gx");
    check(aclrtMalloc(&result_dev, MAX_BLOCKS * 2 * sizeof(int64_t),
                      ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc result");

    // Key calibration points (same as atomic.asc).
    uint32_t key_blocks[] = {1, 4, 8, 16, 32, 64};
    run_sweep(funcHandle, stream, gx_dev, result_dev,
              key_blocks, sizeof(key_blocks) / sizeof(key_blocks[0]),
              "Key calibration points");

    // Full 1~64 sweep for curve plotting (same as atomic.asc).
    std::vector<uint32_t> full_blocks;
    for (uint32_t b = 1; b <= 64; b++) full_blocks.push_back(b);
    run_sweep(funcHandle, stream, gx_dev, result_dev,
              full_blocks.data(), full_blocks.size(),
              "Full 1~64 sweep");

    aclrtFree(gx_dev);
    aclrtFree(result_dev);
    aclrtBinaryUnLoad(binHandle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
