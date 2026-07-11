// Host runner for bypass_dcache_ccec probe.
//
// Loads the linked AICore binary, launches each mode with 2 blocks,
// and prints results. Equivalent to the bisheng .asc host but uses
// the ccec-compiled kernel binary.
//
// Build: see run_bypass_dcache.sh
#include "../probe_host.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <time.h>

struct KernelArgs {
    uint64_t storage_pointer;
    uint32_t mode;
    uint32_t num_blocks;
};

static_assert(sizeof(KernelArgs) == 16, "unexpected CCEC kernel argument ABI");

static void check(aclError err, const char *msg) {
    if (err != ACL_SUCCESS) {
        fprintf(stderr, "ACL error %d: %s\n", err, msg);
        exit(1);
    }
}

static void run_mode(aclrtFuncHandle funcHandle, aclrtStream stream,
                     void *gx_dev, uint32_t mode, const char *label,
                     atomic_probe::Result &result) {
    uint32_t zeros[64] = {0};
    check(aclrtMemcpy(gx_dev, sizeof(zeros), zeros, sizeof(zeros),
                      ACL_MEMCPY_HOST_TO_DEVICE), "init gx");

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    KernelArgs args{(uint64_t)(uintptr_t)gx_dev, mode, 2};
    check(aclrtLaunchKernelWithHostArgs(funcHandle, 2, stream, nullptr,
                                        &args, sizeof(args), nullptr, 0),
          "aclrtLaunchKernelWithHostArgs");
    check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double host_us = (ts_end.tv_sec - ts_start.tv_sec) * 1e6 +
                     (ts_end.tv_nsec - ts_start.tv_nsec) / 1e3;

    uint32_t r[64] = {0};
    check(aclrtMemcpy(r, sizeof(r), gx_dev, sizeof(r),
                      ACL_MEMCPY_DEVICE_TO_HOST), "read result");

    if (mode <= 1) {
        uint8_t *snap = reinterpret_cast<uint8_t*>(&r[32]);
        uint32_t from = (mode == 0) ? 1 : 4;
        uint32_t corrupted = 0;
        for (uint32_t i = from; i < 64; i++)
            if (snap[i] != (uint8_t)(i + 1)) corrupted++;
        bool target_ok = mode == 0 ? snap[0] == 0xFF
                                   : snap[0] == 0xFF && snap[1] == 0xFF &&
                                         snap[2] == 0xFF && snap[3] == 0xFF;
        bool pass = (corrupted == 0) && target_ok;
        printf("[%-16s] mode=%u  corrupted=%u/%u  first8=", label, mode, corrupted, 64 - from);
        for (uint32_t i = 0; i < 8; i++) printf("%02x ", snap[i]);
        printf(" %.0fus  %s\n", host_us, pass ? "PASS" : "FAIL");
        char assertion[80];
        std::snprintf(assertion, sizeof(assertion), "CCEC st_dev mode %u exact width", mode);
        result.Expect(pass, assertion);
    } else if (mode == 2) {
        bool pass = (r[16] == 0) && (r[17] == 120);
        printf("[%-16s] mode=%u  read_errors=%u/%u  %.0fus  %s\n",
               label, mode, r[16], r[17], host_us, pass ? "PASS" : "FAIL");
        result.Expect(pass, "CCEC ld_dev 1B/2B/4B/8B exact reads");
    } else if (mode == 3) {
        uint32_t normal = r[17], bypass = r[18], exp = r[19];
        printf("[%-16s] mode=%u  normal=0x%x bypass=0x%x exp=0x%x  %.0fus  bypass:%s  normal:%s\n",
               label, mode, normal, bypass, exp, host_us,
               (bypass == exp) ? "OK" : "FAIL",
               (normal == exp) ? "OK" : "STALE");
        result.Expect(bypass == exp, "CCEC ld_dev bypasses stale L1");
    } else if (mode == 4) {
        bool pass = (r[16] == 0) && (r[18] >= 100) && (r[17] == r[18]);
        printf("[%-16s] mode=%u  corrupt=%u/15  gx0=%u(exp %u)  %.0fus  %s\n",
               label, mode, r[16], r[17], r[18], host_us, pass ? "PASS" : "FAIL");
        result.Expect(pass, "CCEC st_dev plus atomic exact result");
    } else {
        uint32_t status = r[16];
        uint32_t w0 = r[17];
        uint32_t w3 = r[18];
        uint32_t expected_w0 = r[19];
        if (mode == 5) {
            printf("[%-16s] mode=%u  st_dev gx[0]=0x%x(exp 0x%x) store+dcci gx[3]=0x%x  %.0fus  gx[0]:%s\n",
                   label, mode, w0, expected_w0, w3, host_us,
                   status ? "CLOBBERED" : "SURVIVED");
        } else if (mode == 6) {
            bool pass = w0 == 0 && w3 == 0xCAFEBABEu;
            printf("[%-16s] mode=%u  gx[0]=0x%x(exp 0 clobbered) gx[3]=0x%x(exp 0xCAFEBABE)  %.0fus  %s\n",
                   label, mode, w0, w3, host_us,
                   pass ? "CLOBBERED(Confirmed)" : "UNEXPECTED");
            result.Expect(pass, "CCEC deterministic dcci clobber");
        } else if (mode == 7) {
            bool pass = w0 == expected_w0 && w3 == 0xCAFEBABEu;
            printf("[%-16s] mode=%u  gx[0]=0x%x(exp 0x%x) gx[3]=0x%x(exp 0xCAFEBABE)  %.0fus  %s\n",
                   label, mode, w0, expected_w0, w3, host_us,
                   pass ? "PASS(inval-fix)" : "FAIL");
            result.Expect(pass, "CCEC invalidate-before-store exact result");
        }
    }
}

int main(int argc, char *argv[]) {
    const char *kernel_path = "./bypass_dcache_kernel.o";
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
    printf("Kernel binary: %s (%zu bytes)\n\n", kernel_path, bin_size);

    aclrtBinHandle binHandle;
    check(atomic_probe::LoadAicoreBinaryFromData(bin_data.data(), bin_size, &binHandle),
          "LoadAicoreBinaryFromData");

    aclrtFuncHandle funcHandle;
    check(aclrtBinaryGetFunctionByEntry(binHandle, 0, &funcHandle),
          "aclrtBinaryGetFunctionByEntry");

    void *gx_dev = nullptr;
    check(aclrtMalloc(&gx_dev, 64 * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
          "aclrtMalloc gx");

    printf("=== ccec ByPassDCache Probe (st_dev/ld_dev) ===\n");
    printf("Compiled with: scalar auto-dcci=false, kernel-end dcci=false\n\n");
    atomic_probe::Result result;

    uint32_t mode = 0;
    if (!atomic_probe::RequiredUintEnv("ATOMIC_PROBE_MODE", 8, &mode)) return EXIT_FAILURE;
    const char *labels[] = {
        "ByPass1B(blast)", "ByPass4B(blast)", "ByPassReadCheck", "ByPassReadVsL1",
        "stdev+Atomic", "ConcurrentMixed", "dcciClobbersStdev", "InvalFix"
    };
    run_mode(funcHandle, stream, gx_dev, mode, labels[mode], result);

    // Run concurrent mode 5 multiple times to show race distribution
    if (mode == 5) {
        printf("\n--- Mode 5 (ConcurrentMixed) x20 runs ---\n");
        uint32_t survived = 0, clobbered = 0;
        for (int i = 0; i < 20; i++) {
            uint32_t zeros[64] = {0};
            check(aclrtMemcpy(gx_dev, sizeof(zeros), zeros, sizeof(zeros),
                              ACL_MEMCPY_HOST_TO_DEVICE), "init");
            KernelArgs args{(uint64_t)(uintptr_t)gx_dev, 5, 2};
            check(aclrtLaunchKernelWithHostArgs(funcHandle, 2, stream, nullptr,
                                                &args, sizeof(args), nullptr, 0), "launch");
            check(aclrtSynchronizeStream(stream), "sync");
            uint32_t r2[64] = {0};
            check(aclrtMemcpy(r2, sizeof(r2), gx_dev, sizeof(r2), ACL_MEMCPY_DEVICE_TO_HOST), "copy");
            if (r2[16] == 0) survived++; else clobbered++;
        }
        printf("  st_dev survived=%u/20  clobbered_by_dcci=%u/20\n\n", survived, clobbered);
    }

    printf("\n=== Summary ===\n");
    printf("st_dev: single writes bypass DCache and preserve the tested write width\n");
    printf("ld_dev: reads bypass the local DCache in the tested publish/observe paths\n");
    printf("repeated cross-AIV same-line writes are tested separately by st_dev_same_line\n");

    check(aclrtFree(gx_dev), "aclrtFree");
    check(aclrtBinaryUnLoad(binHandle), "aclrtBinaryUnLoad");
    check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    check(aclrtResetDevice(deviceId), "aclrtResetDevice");
    check(aclFinalize(), "aclFinalize");
    return result.ExitCode();
}
