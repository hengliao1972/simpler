// Host runner for entire_flush_clobber probe.
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

static void check(aclError err, const char *msg) {
    if (err != ACL_SUCCESS) {
        fprintf(stderr, "ACL error %d: %s\n", err, msg);
        exit(1);
    }
}

int main(int argc, char *argv[]) {
    const char *kernel_path = "./entire_flush_clobber_kernel.o";
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

    aclrtBinHandle binHandle;
    check(aclrtBinaryLoadFromData(bin_data.data(), bin_size, nullptr, &binHandle),
          "aclrtBinaryLoadFromData");

    aclrtFuncHandle funcHandle;
    check(aclrtBinaryGetFunctionByEntry(binHandle, 0, &funcHandle),
          "aclrtBinaryGetFunctionByEntry");

    void *gx_dev = nullptr;
    check(aclrtMalloc(&gx_dev, 256 * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
          "aclrtMalloc gx");

    uint32_t test_blocks[] = {1, 2, 4, 8, 16, 32, 64};
    for (size_t i = 0; i < sizeof(test_blocks)/sizeof(test_blocks[0]); i++) {
        uint32_t numBlocks = test_blocks[i];
        // Zero whole region; gx[30] = BARRIER_SLOT must start at 0.
        uint32_t zero[256] = {0};
        check(aclrtMemcpy(gx_dev, sizeof(zero), zero, sizeof(zero),
                          ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy init");

        uint64_t gx_ptr = (uint64_t)(uintptr_t)gx_dev;
        uint64_t args[2] = {gx_ptr, (uint64_t)numBlocks};

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

        uint32_t result[256] = {0};
        check(aclrtMemcpy(result, sizeof(result), gx_dev, sizeof(result),
                          ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy result");

        // Flags live at gx[64+bid], summary at gx[128], raw at gx[129+bid].
        printf("blocks=%2u  survivors=%u/%u  flags: ", numBlocks, result[128], numBlocks);
        for (uint32_t b = 0; b < numBlocks && b < 10; b++)
            printf("[%u]=%u ", b, result[129+b]);
        printf("\n");
    }

    aclrtFree(gx_dev);
    aclrtBinaryUnLoad(binHandle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
