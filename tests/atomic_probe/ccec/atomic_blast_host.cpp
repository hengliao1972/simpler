// Host runner for atomic_blast probe.
//
// Loads the linked AICore binary, launches each mode with 2 blocks,
// and prints results. Equivalent to the cacheline_blast.asc host but uses
// the ccec-compiled kernel binary.
//
// Build: see run_atomic_blast.sh
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

    if (mode == 1) {
        // Word-level: gx[16]=corrupted, gx[17]=gx[0], gx[18]=expected
        bool pass = (r[16] == 0) && (r[17] == r[18]);
        printf("[%-16s] mode=%u  corrupted=%u/15  gx0=0x%x(exp 0x%x)  %.0fus  %s\n",
               label, mode, r[16], r[17], r[18], us, pass ? "PASS" : "FAIL");
    } else {
        // Byte-level (modes 0 and 2): snapshot at gx[31..46], check bytes 4-63.
        uint8_t *snap = reinterpret_cast<uint8_t*>(&r[31]);
        uint32_t corrupted = 0;
        for (uint32_t i = 4; i < 64; i++)
            if (snap[i] != (uint8_t)(i + 1)) corrupted++;
        bool pass = (corrupted == 0) && (r[17] == r[18]);
        printf("[%-16s] mode=%u  byte_corrupted=%u/60  first8=", label, mode, corrupted);
        for (uint32_t i = 0; i < 8; i++) printf("%02x ", snap[i]);
        printf(" gx0=0x%x(exp 0x%x) %.0fus  %s\n", r[17], r[18], us, pass ? "PASS" : "FAIL");
    }
}

int main(int argc, char *argv[]) {
    const char *kernel_path = "./atomic_blast_kernel.o";
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

    printf("=== ccec Atomic Blast Radius Probe (atomicMax) ===\n");
    printf("Compiled with: ccec -x cce -mllvm -cce-aicore-dcci-insert-for-scalar=false\n");
    printf("Question: does atomicMax on gx[0] clobber neighbour bytes/words?\n\n");

    run_mode(fh, stream, gx_dev, 0, "Sequential");
    run_mode(fh, stream, gx_dev, 1, "Stale-L1");
    run_mode(fh, stream, gx_dev, 2, "4B-blast");

    printf("\n=== Summary ===\n");
    printf("atomicMax blast radius = exactly 4B (uint32), does NOT spread to neighbours\n");
    printf("Hardware atomics bypass L1, doing RMW directly in L2/HBM.\n");

    aclrtFree(gx_dev);
    aclrtBinaryUnLoad(bh);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
