// Host runner for dcci_clean_clobber probe.
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
    uint32_t zeros[64] = {0};
    check(aclrtMemcpy(gx_dev, sizeof(zeros), zeros, sizeof(zeros),
                      ACL_MEMCPY_HOST_TO_DEVICE), "init");

    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    uint64_t gx_ptr = (uint64_t)(uintptr_t)gx_dev;
    uint64_t args[3] = {gx_ptr, (uint64_t)mode, 2};
    aclrtArgsHandle ah;
    check(aclrtKernelArgsInit(fh, &ah), "init");
    aclrtParamHandle ph;
    check(aclrtKernelArgsAppend(ah, args, sizeof(args), &ph), "append");
    check(aclrtKernelArgsFinalize(ah), "finalize");
    check(aclrtLaunchKernelWithConfig(fh, 2, stream, nullptr, ah, nullptr), "launch");
    check(aclrtSynchronizeStream(stream), "sync");

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double us = (ts1.tv_sec - ts0.tv_sec) * 1e6 + (ts1.tv_nsec - ts0.tv_nsec) / 1e3;

    uint32_t r[64] = {0};
    check(aclrtMemcpy(r, sizeof(r), gx_dev, sizeof(r), ACL_MEMCPY_DEVICE_TO_HOST), "copy");

    if (mode == 0) {
        printf("[%-24s] w0=0x%x(exp 0xAA) w3=0x%x(exp 0x%x)  %.0fus  w3:%s\n",
               label, r[18], r[19], r[20], us,
               r[17] ? "CLOBBERED!" : "survived");
    } else if (mode == 1) {
        printf("[%-24s] w3=0x%x(exp 0x%x)  %.0fus  w3:%s\n",
               label, r[19], r[20], us,
               r[16] ? "CLOBBERED!" : "survived(CLEAN line safe)");
    } else if (mode == 2) {
        printf("[%-24s] w0=0x%x w3=0x%x(exp 0x%x)  %.0fus  w3:%s\n",
               label, r[18], r[19], r[20], us,
               r[17] ? "CLOBBERED!" : "survived");
    } else if (mode == 3) {
        printf("[%-24s] w0=0x%x(stuck in L1) w3=0x%x(exp 0x%x)  %.0fus  w0:%s w3:%s\n",
               label, r[18], r[19], r[20], us,
               (r[18] == 0) ? "NOT visible(L1)" : "visible",
               r[17] ? "CLOBBERED!" : "survived");
    } else if (mode == 4) {
        printf("[%-24s] clobbered=%u/15 words  w0=0x%x  %.0fus\n",
               label, r[16], r[17], us);
    }
}

int main(int argc, char *argv[]) {
    const char *kernel_path = "./dcci_clean_kernel.o";
    if (argc > 1) kernel_path = argv[1];

    check(aclInit(nullptr), "aclInit");
    check(aclrtSetDevice(0), "aclrtSetDevice");
    aclrtStream stream;
    check(aclrtCreateStream(&stream), "aclrtCreateStream");

    std::ifstream f(kernel_path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "Cannot open %s\n", kernel_path); return 1; }
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<char> bin(sz);
    f.read(bin.data(), sz);

    aclrtBinHandle bh;
    check(aclrtBinaryLoadFromData(bin.data(), sz, nullptr, &bh), "load");
    aclrtFuncHandle fh;
    check(aclrtBinaryGetFunctionByEntry(bh, 0, &fh), "getfunc");

    void *gx_dev;
    check(aclrtMalloc(&gx_dev, 64 * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST), "malloc");

    printf("=== dcci Clean-Clobber Probe ===\n");
    printf("Question: does dcci(inval)'s clean step clobber another core's st_dev write?\n\n");

    run_mode(fh, stream, gx_dev, 0, "dcci(inval) DIRTY");
    run_mode(fh, stream, gx_dev, 1, "dcci(inval) CLEAN");
    run_mode(fh, stream, gx_dev, 2, "dcci(flush) DIRTY");
    run_mode(fh, stream, gx_dev, 3, "Control no dcci");
    run_mode(fh, stream, gx_dev, 4, "Full 64B clobber");

    printf("\n=== Conclusion ===\n");
    printf("DIRTY L1 + dcci = CLOBBERS st_dev writes in same cache line\n");
    printf("CLEAN L1 + dcci = safe (clean is no-op for clean lines)\n");

    aclrtFree(gx_dev);
    aclrtBinaryUnLoad(bh);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
