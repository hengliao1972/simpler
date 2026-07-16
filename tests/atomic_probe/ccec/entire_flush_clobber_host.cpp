// Host runner for entire_flush_clobber probe.
#include "../probe_host.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

constexpr uint32_t STORAGE_WORDS = 512;
constexpr uint32_t SUMMARY_SLOT = 256;
constexpr uint32_t RAW_BASE = 257;
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

int main(int argc, char *argv[]) {
    const char *kernel_path = "./entire_flush_clobber_kernel.o";
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

    aclrtBinHandle binHandle;
    check(atomic_probe::LoadAicoreBinaryFromData(bin_data.data(), bin_size, &binHandle),
          "LoadAicoreBinaryFromData");

    aclrtFuncHandle funcHandle;
    check(aclrtBinaryGetFunctionByEntry(binHandle, 0, &funcHandle),
          "aclrtBinaryGetFunctionByEntry");

    void *gx_dev = nullptr;
    check(aclrtMalloc(&gx_dev, STORAGE_WORDS * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
          "aclrtMalloc gx");

    atomic_probe::Result probe_result;
    uint32_t test_blocks[] = {1, 2, 4, 8, 16, 32, 64};
    for (size_t i = 0; i < sizeof(test_blocks)/sizeof(test_blocks[0]); i++) {
        uint32_t numBlocks = test_blocks[i];
        uint32_t zero[STORAGE_WORDS] = {0};
        check(aclrtMemcpy(gx_dev, sizeof(zero), zero, sizeof(zero),
                          ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy init");

        KernelArgs args{(uint64_t)(uintptr_t)gx_dev, numBlocks, 0};
        check(aclrtLaunchKernelWithHostArgs(funcHandle, numBlocks, stream, nullptr,
                                            &args, sizeof(args), nullptr, 0),
              "aclrtLaunchKernelWithHostArgs");
        check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");

        uint32_t result[STORAGE_WORDS] = {0};
        check(aclrtMemcpy(result, sizeof(result), gx_dev, sizeof(result),
                          ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy result");

        uint32_t participants = numBlocks * CORES_PER_BLOCK;
        printf("blocks=%2u  survivors=%u/%u  flags: ",
               numBlocks, result[SUMMARY_SLOT], participants);
        for (uint32_t participant = 0; participant < participants && participant < 10; participant++)
            printf("[%u]=%u ", participant, result[RAW_BASE + participant]);
        printf("\n");

        uint32_t calculated_survivors = 0;
        for (uint32_t participant = 0; participant < participants; participant++) {
            if (result[RAW_BASE + participant] == participant + 1) calculated_survivors++;
        }
        char assertion[96];
        std::snprintf(assertion, sizeof(assertion), "ENTIRE dcci report consistency, blocks=%u", numBlocks);
        probe_result.Expect(result[SUMMARY_SLOT] == calculated_survivors, assertion);
    }

    check(aclrtFree(gx_dev), "aclrtFree");
    check(aclrtBinaryUnLoad(binHandle), "aclrtBinaryUnLoad");
    check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    check(aclrtResetDevice(deviceId), "aclrtResetDevice");
    check(aclFinalize(), "aclFinalize");
    return probe_result.ExitCode();
}
