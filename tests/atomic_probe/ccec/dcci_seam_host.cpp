// Host runner for dcci_seam probe.
//
// Loads the linked AICore binary, launches it with 2 blocks, and prints
// the result. Equivalent to the mb8_dcci_seam.asc host but uses the
// ccec-compiled kernel binary.
//
// Build: see run_dcci_seam.sh
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

int main(int argc, char *argv[]) {
    const char *kernel_path = "./dcci_seam_kernel.o";
    if (argc > 1) kernel_path = argv[1];

    constexpr uint32_t NUM_ROUNDS = 100;

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

    // Zero flag + error counter; gx[30] = BARRIER_SLOT must start at 0.
    uint32_t zeros[64] = {0};
    check(aclrtMemcpy(gx_dev, sizeof(zeros), zeros, sizeof(zeros),
                      ACL_MEMCPY_HOST_TO_DEVICE), "init gx");

    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    uint64_t gx_ptr = (uint64_t)(uintptr_t)gx_dev;
    uint64_t args[2] = {gx_ptr, 2};
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

    uint32_t flag = r[16];
    uint32_t errors = r[17];
    bool pass = (errors == 0) && (flag == NUM_ROUNDS);
    printf("[dcci_seam] rounds=%u flag=%u(exp %u) errors=%u  %.0fus  %s\n",
           NUM_ROUNDS, flag, NUM_ROUNDS, errors, us, pass ? "PASS" : "FAIL");

    printf("\n=== Conclusion ===\n");
    printf("Producer st_dev + atomicAdd(flag), consumer ld_dev(poll) + dcci(inval) + read\n");
    printf("is a correct publish/observe seam: %s.\n",
           pass ? "consumer always reads fresh data" : "STALE DATA DETECTED");

    aclrtFree(gx_dev);
    aclrtBinaryUnLoad(bh);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
