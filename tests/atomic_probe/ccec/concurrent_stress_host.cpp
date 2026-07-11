// Host runner for concurrent_stress probe.
//
// Loads the linked AICore binary, launches each mode with 2 blocks,
// and prints results. Equivalent to the cacheline_stress.asc host but uses
// the ccec-compiled kernel binary.
//
// Build: see run_concurrent_stress.sh
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

static void run_mode(aclrtFuncHandle fh, aclrtStream stream,
                     void *gx_dev, uint32_t mode, const char *label) {
    // Zero whole region; gx[30] = BARRIER_SLOT must start at 0.
    uint32_t zeros[64] = {0};
    check(aclrtMemcpy(gx_dev, sizeof(zeros), zeros, sizeof(zeros),
                      ACL_MEMCPY_HOST_TO_DEVICE), "init gx");

    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    uint64_t gx_ptr = (uint64_t)(uintptr_t)gx_dev;
    uint64_t args[3] = {gx_ptr, (uint64_t)mode, 2};
    aclrtArgsHandle ah;
    check(aclrtKernelArgsInit(fh, &ah), "aclrtKernelArgsInit");
    aclrtParamHandle ph;
    check(aclrtKernelArgsAppend(ah, args, sizeof(args), &ph), "aclrtKernelArgsAppend");
    check(aclrtKernelArgsFinalize(ah), "aclrtKernelArgsFinalize");
    check(aclrtLaunchKernelWithConfig(fh, 2, stream, nullptr, ah, nullptr),
          "aclrtLaunchKernelWithConfig");
    check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double us = (ts1.tv_sec - ts0.tv_sec) * 1e6 + (ts1.tv_nsec - ts0.tv_nsec) / 1e3;

    uint32_t r[64] = {0};
    check(aclrtMemcpy(r, sizeof(r), gx_dev, sizeof(r), ACL_MEMCPY_DEVICE_TO_HOST),
          "read result");

    if (mode == 0) {
        printf("[%-18s] mode=%u  w0=0x%08x w1=0x%08x w2=0x%08x w3=0x%08x  err=%u/4  %.0fus\n",
               label, mode, r[16], r[17], r[18], r[19], r[22], us);
    } else if (mode == 1) {
        printf("[%-18s] mode=%u  st_err=%u/4  b_err=%u/4  st_w0=0x%08x b_w8=0x%08x  %.0fus\n",
               label, mode, r[22], r[23], r[16], r[17], us);
    } else if (mode == 2) {
        printf("[%-18s] mode=%u  st_err=%u/4  dcci_err=%u/4  st_w0=0x%08x dcci_w8=0x%08x  %.0fus\n",
               label, mode, r[22], r[23], r[16], r[17], us);
    }
}

int main(int argc, char *argv[]) {
    const char *kernel_path = "./concurrent_stress_kernel.o";
    if (argc > 1) kernel_path = argv[1];

    check(aclInit(nullptr), "aclInit");
    check(aclrtSetDevice(0), "aclrtSetDevice");

    aclrtStream stream;
    check(aclrtCreateStream(&stream), "aclrtCreateStream");

    std::ifstream f(kernel_path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "Cannot open %s\n", kernel_path); return 1; }
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<char> bin(sz);
    f.read(bin.data(), sz);
    printf("Kernel binary: %s (%zu bytes)\n\n", kernel_path, sz);

    aclrtBinHandle bh;
    check(aclrtBinaryLoadFromData(bin.data(), sz, nullptr, &bh), "aclrtBinaryLoadFromData");
    aclrtFuncHandle fh;
    check(aclrtBinaryGetFunctionByEntry(bh, 0, &fh), "aclrtBinaryGetFunctionByEntry");

    void *gx_dev = nullptr;
    check(aclrtMalloc(&gx_dev, 64 * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
          "aclrtMalloc gx");

    printf("=== ccec Concurrent Stress Probe (tight-loop, no SyncAll) ===\n");
    printf("Compiled with: ccec -x cce -mllvm -cce-aicore-dcci-insert-for-scalar=false\n");
    printf("ROUNDS=1000\n\n");

    run_mode(fh, stream, gx_dev, 0, "store+dcci 2blk");
    run_mode(fh, stream, gx_dev, 1, "st_dev 2blk");
    run_mode(fh, stream, gx_dev, 2, "Mixed st_dev+dcci");

    // Run mode 0 multiple times for clobber distribution
    printf("\n--- Mode 0 x5 runs (store+dcci stress) ---\n");
    for (int i = 0; i < 5; i++) {
        uint32_t zeros[64] = {0};
        check(aclrtMemcpy(gx_dev, sizeof(zeros), zeros, sizeof(zeros),
                          ACL_MEMCPY_HOST_TO_DEVICE), "init");
        uint64_t gx_ptr = (uint64_t)(uintptr_t)gx_dev;
        uint64_t args[3] = {gx_ptr, 0, 2};
        aclrtArgsHandle ah;
        check(aclrtKernelArgsInit(fh, &ah), "init");
        aclrtParamHandle ph;
        check(aclrtKernelArgsAppend(ah, args, sizeof(args), &ph), "append");
        check(aclrtKernelArgsFinalize(ah), "finalize");
        check(aclrtLaunchKernelWithConfig(fh, 2, stream, nullptr, ah, nullptr), "launch");
        check(aclrtSynchronizeStream(stream), "sync");
        uint32_t r[64] = {0};
        check(aclrtMemcpy(r, sizeof(r), gx_dev, sizeof(r), ACL_MEMCPY_DEVICE_TO_HOST), "copy");
        printf("  run %d: err=%u/4  w0=0x%08x w2=0x%08x\n", i, r[22], r[16], r[18]);
    }

    printf("\n=== Summary ===\n");
    printf("store+dcci: writes ENTIRE 64B dirty L1 line back → cross-core clobber\n");
    printf("st_dev:    isolated to write width, bypasses DCache → no clobber\n");

    aclrtFree(gx_dev);
    aclrtBinaryUnLoad(bh);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
