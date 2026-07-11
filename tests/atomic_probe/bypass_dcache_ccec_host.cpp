// Host runner for bypass_dcache_ccec probe.
//
// Loads the linked AICore binary, launches each mode with 2 blocks,
// and prints results. Equivalent to the bisheng .asc host but uses
// the ccec-compiled kernel binary.
//
// Build: see run_bypass_dcache.sh
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

static void run_mode(aclrtFuncHandle funcHandle, aclrtStream stream,
                     void *gx_dev, uint32_t mode, const char *label) {
    uint32_t zeros[64] = {0};
    check(aclrtMemcpy(gx_dev, sizeof(zeros), zeros, sizeof(zeros),
                      ACL_MEMCPY_HOST_TO_DEVICE), "init gx");

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    uint64_t gx_ptr = (uint64_t)(uintptr_t)gx_dev;
    uint64_t args[3] = {gx_ptr, (uint64_t)mode, 2};

    aclrtArgsHandle argsHandle;
    check(aclrtKernelArgsInit(funcHandle, &argsHandle), "aclrtKernelArgsInit");
    aclrtParamHandle paramHandle;
    check(aclrtKernelArgsAppend(argsHandle, args, sizeof(args), &paramHandle),
          "aclrtKernelArgsAppend");
    check(aclrtKernelArgsFinalize(argsHandle), "aclrtKernelArgsFinalize");

    check(aclrtLaunchKernelWithConfig(funcHandle, 2, stream, nullptr,
                                      argsHandle, nullptr),
          "aclrtLaunchKernelWithConfig");
    check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double host_us = (ts_end.tv_sec - ts_start.tv_sec) * 1e6 +
                     (ts_end.tv_nsec - ts_start.tv_nsec) / 1e3;

    uint32_t r[64] = {0};
    check(aclrtMemcpy(r, sizeof(r), gx_dev, sizeof(r),
                      ACL_MEMCPY_DEVICE_TO_HOST), "read result");

    if (mode <= 1) {
        uint8_t *snap = reinterpret_cast<uint8_t*>(&r[17]);
        uint32_t from = (mode == 0) ? 1 : 4;
        uint32_t corrupted = 0;
        for (uint32_t i = from; i < 64; i++)
            if (snap[i] != (uint8_t)(i + 1)) corrupted++;
        bool pass = (corrupted == 0);
        printf("[%-16s] mode=%u  corrupted=%u/%u  first8=", label, mode, corrupted, 64 - from);
        for (uint32_t i = 0; i < 8; i++) printf("%02x ", snap[i]);
        printf(" %.0fus  %s\n", host_us, pass ? "PASS" : "FAIL");
    } else if (mode == 2) {
        bool pass = (r[16] == 0);
        printf("[%-16s] mode=%u  read_errors=%u/%u  %.0fus  %s\n",
               label, mode, r[16], r[17], host_us, pass ? "PASS" : "FAIL");
    } else if (mode == 3) {
        uint32_t normal = r[17], bypass = r[18], exp = r[19];
        printf("[%-16s] mode=%u  normal=0x%x bypass=0x%x exp=0x%x  %.0fus  bypass:%s  normal:%s\n",
               label, mode, normal, bypass, exp, host_us,
               (bypass == exp) ? "OK" : "FAIL",
               (normal == exp) ? "OK" : "STALE");
    } else if (mode == 4) {
        bool pass = (r[16] == 0) && (r[17] == 100);
        printf("[%-16s] mode=%u  corrupt=%u/15  gx0=%u(exp %u)  %.0fus  %s\n",
               label, mode, r[16], r[17], r[18], host_us, pass ? "PASS" : "FAIL");
    } else if (mode == 5) {
        uint32_t polluted = r[16];
        uint32_t bypass_val = r[17];
        uint32_t normal_after = r[18];
        uint32_t after_dcci = r[35];
        printf("[%-16s] mode=%u  bypass=0x%x normal_after=0x%x(exp 0xaa) after_dcci=0x%x  %.0fus  L1:%s\n",
               label, mode, bypass_val, normal_after, after_dcci, host_us,
               polluted ? "POLLUTED" : "CLEAN");
    }
}

int main(int argc, char *argv[]) {
    const char *kernel_path = "./bypass_dcache_kernel.o";
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
    printf("Kernel binary: %s (%zu bytes)\n\n", kernel_path, bin_size);

    aclrtBinHandle binHandle;
    check(aclrtBinaryLoadFromData(bin_data.data(), bin_size, nullptr, &binHandle),
          "aclrtBinaryLoadFromData");

    aclrtFuncHandle funcHandle;
    check(aclrtBinaryGetFunctionByEntry(binHandle, 0, &funcHandle),
          "aclrtBinaryGetFunctionByEntry");

    void *gx_dev = nullptr;
    check(aclrtMalloc(&gx_dev, 64 * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
          "aclrtMalloc gx");

    printf("=== ccec ByPassDCache Probe (st_dev/ld_dev) ===\n");
    printf("Compiled with: ccec -x cce -mllvm -cce-aicore-dcci-insert-for-scalar=false\n\n");

    run_mode(funcHandle, stream, gx_dev, 0, "ByPass1B(blast)");
    run_mode(funcHandle, stream, gx_dev, 1, "ByPass4B(blast)");
    run_mode(funcHandle, stream, gx_dev, 2, "ByPassReadCheck");
    run_mode(funcHandle, stream, gx_dev, 3, "ByPassReadVsL1");
    run_mode(funcHandle, stream, gx_dev, 4, "ByPass+Atomic");
    run_mode(funcHandle, stream, gx_dev, 5, "ByPassNoPollute");

    printf("\n=== Summary ===\n");
    printf("st_dev: write GM bypassing DCache (no dirty, no clobber)\n");
    printf("ld_dev: read GM bypassing DCache (no stale L1, always fresh)\n");

    aclrtFree(gx_dev);
    aclrtBinaryUnLoad(binHandle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
