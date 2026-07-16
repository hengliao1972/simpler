// Host runner for the paired CCEC DCCI seam probe.
#include "../probe_host.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <time.h>
#include <vector>

constexpr uint32_t NUM_ROUNDS = 100;
constexpr uint32_t STORAGE_WORDS = 6 * 16;
constexpr uint32_t READY_OFFSET = 16;
constexpr uint32_t DONE_OFFSET = 32;
constexpr uint32_t WRITER_OFFSET = 48;
constexpr uint32_t RESULT_OFFSET = 64;

static uint32_t round_base(uint32_t round)
{
    return 0xA5000000u + round * 0x100u;
}

struct KernelArgs {
    uint64_t storage_pointer;
    uint32_t mode;
    uint32_t num_blocks;
};

static_assert(sizeof(KernelArgs) == 16, "unexpected CCEC kernel argument ABI");

static void check(aclError err, const char *msg)
{
    if (err != ACL_SUCCESS) {
        fprintf(stderr, "ACL error %d: %s\n", err, msg);
        exit(1);
    }
}

static void run_mode(aclrtFuncHandle function, aclrtStream stream, void *storage,
                     uint32_t mode, const char *label, atomic_probe::Result &result)
{
    uint32_t zeros[STORAGE_WORDS] = {};
    check(aclrtMemcpy(storage, sizeof(zeros), zeros, sizeof(zeros), ACL_MEMCPY_HOST_TO_DEVICE),
          "initialize probe storage");

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    KernelArgs args{static_cast<uint64_t>(reinterpret_cast<uintptr_t>(storage)), mode, 2};
    check(aclrtLaunchKernelWithHostArgs(function, 2, stream, nullptr, &args, sizeof(args), nullptr, 0),
          "launch dcci seam");
    check(aclrtSynchronizeStream(stream), "synchronize dcci seam");
    clock_gettime(CLOCK_MONOTONIC, &end);

    uint32_t values[STORAGE_WORDS] = {};
    check(aclrtMemcpy(values, sizeof(values), storage, sizeof(values), ACL_MEMCPY_DEVICE_TO_HOST),
          "read dcci seam results");
    double elapsed_us = (end.tv_sec - start.tv_sec) * 1e6 +
                        (end.tv_nsec - start.tv_nsec) / 1e3;

    const uint32_t *probe = &values[RESULT_OFFSET];
    printf("[%-11s] fresh=%u stale=%u normal_other=%u gm_bad=%u preload_current=%u "
           "ready=%u done=%u writer=%u reader=0x%x rounds=%u %.0fus\n",
           label, probe[0], probe[1], probe[2], probe[3], probe[10], values[READY_OFFSET],
           values[DONE_OFFSET], values[WRITER_OFFSET], probe[4], probe[5], elapsed_us);
    bool normal_distribution_unexpected =
        (mode <= 3 && probe[0] != NUM_ROUNDS) || (mode == 4 && probe[1] != NUM_ROUNDS);
    if (normal_distribution_unexpected) {
        printf("  首个不符合该 mode 预期的普通读取：round=%u value=0x%x\n", probe[6], probe[7]);
    }
    if (probe[3] != 0) {
        printf("  首个不等于写核本轮新值的 GM 读取：round=%u value=0x%x\n", probe[8], probe[9]);
    }

    bool data_exact = true;
    uint32_t final_base = round_base(NUM_ROUNDS - 1);
    for (uint32_t i = 0; i < 16; ++i) {
        data_exact = data_exact && values[i] == final_base + i;
    }
    result.Expect(data_exact, "GM 中 data 的 16 个 word 必须精确等于写核最后一轮完整值");

    bool ready_line_exact = values[READY_OFFSET] == NUM_ROUNDS;
    bool done_line_exact = values[DONE_OFFSET] == NUM_ROUNDS;
    bool writer_line_exact = values[WRITER_OFFSET] == 1;
    for (uint32_t i = 1; i < 16; ++i) {
        ready_line_exact = ready_line_exact && values[READY_OFFSET + i] == 0;
        done_line_exact = done_line_exact && values[DONE_OFFSET + i] == 0;
        writer_line_exact = writer_line_exact && values[WRITER_OFFSET + i] == 0;
    }
    result.Expect(ready_line_exact, "ready line 仅 word[0] 为 100，其余 15 个 word 必须为 0");
    result.Expect(done_line_exact, "done line 仅 word[0] 为 100，其余 15 个 word 必须为 0");
    result.Expect(writer_line_exact, "writer marker line 仅 word[0] 为 1，其余 15 个 word 必须为 0");
    result.Expect(probe[4] == 0xC001CAFEu && probe[5] == NUM_ROUNDS,
                  "reader marker 必须为 0xC001CAFE，完成轮数必须为 100");
    result.Expect(probe[10] == 0, "写核发布前，读核不得提前预读到本轮尚未发布的新值");
    bool result_tail_zero = true;
    for (uint32_t i = 11; i < 16; ++i) {
        result_tail_zero = result_tail_zero && probe[i] == 0;
    }
    result.Expect(result_tail_zero, "reader result 中未使用的 word[11..15] 必须保持为 0");
    bool unused_line_zero = true;
    for (uint32_t i = 5 * 16; i < STORAGE_WORDS; ++i) {
        unused_line_zero = unused_line_zero && values[i] == 0;
    }
    result.Expect(unused_line_zero, "未参与协议的最后一条 64B storage line 必须保持全 0");
    result.Expect(probe[0] + probe[1] + probe[2] == NUM_ROUNDS,
                  "100 轮普通读取必须逐轮且仅被归入新值、实际预读旧值或非法值三类之一");
    result.Expect(probe[3] == 0, "100 轮 ld_dev 都必须读到写核本轮发布的完整新 line");
    result.Expect(probe[8] == 0xffffffffu && probe[9] == 0,
                  "GM 无错误时，首错轮次/value 诊断槽必须保持 0xffffffff/0");
    result.Expect(probe[2] == 0, "普通读取不得出现既非本轮新 line、也非 DCCI 前实际预读 line 的值");
    if (mode == 0) {
        result.Expect(probe[0] == NUM_ROUNDS && probe[1] == 0,
                      "DEFAULT：100 轮普通读取都必须在 DCCI 后取得写核本轮新 line");
    } else if (mode == 1) {
        result.Expect(probe[0] == NUM_ROUNDS && probe[1] == 0,
                      "显式 ALL：100 轮结果必须与两参数 DEFAULT 相同，全部取得本轮新 line");
    } else if (mode == 2) {
        result.Expect(probe[0] == NUM_ROUNDS && probe[1] == 0,
                      "OUT：100 轮普通读取都必须在 DCCI 后取得写核本轮新 line");
    } else if (mode == 3) {
        result.Expect(probe[0] == NUM_ROUNDS && probe[1] == 0,
                      "ATOMIC：100 轮普通读取都必须在 DCCI 后取得写核本轮新 line");
    } else {
        result.Expect(probe[0] == 0 && probe[1] == NUM_ROUNDS,
                      "NO DCCI 对照：100 轮普通读取都必须精确保留本轮 DCCI 前预读的旧 line");
    }
}

int main(int argc, char *argv[])
{
    const char *kernel_path = "./dcci_seam_kernel.o";
    if (argc > 1) {
        kernel_path = argv[1];
    }

    int32_t device_id = atomic_probe::DeviceId();
    if (device_id < 0) {
        return EXIT_FAILURE;
    }
    check(aclInit(nullptr), "aclInit");
    check(aclrtSetDevice(device_id), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    check(aclrtCreateStream(&stream), "aclrtCreateStream");

    std::ifstream file(kernel_path, std::ios::binary | std::ios::ate);
    if (!file) {
        fprintf(stderr, "Cannot open %s\n", kernel_path);
        return EXIT_FAILURE;
    }
    size_t binary_size = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<char> binary(binary_size);
    file.read(binary.data(), static_cast<std::streamsize>(binary_size));

    aclrtBinHandle binary_handle;
    check(atomic_probe::LoadAicoreBinaryFromData(binary.data(), binary_size, &binary_handle),
          "LoadAicoreBinaryFromData");
    aclrtFuncHandle function;
    check(aclrtBinaryGetFunctionByEntry(binary_handle, 0, &function), "get kernel function");

    void *storage = nullptr;
    check(aclrtMalloc(&storage, STORAGE_WORDS * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
          "allocate probe storage");

    printf("=== CCEC clean-reader DCCI 第三参数对照 ===\n");
    printf("读核从不写 data line；每轮读写权限由独立 cache line 上的 atomic phase 串行交接。\n");
    atomic_probe::Result result;
    result.Expect((reinterpret_cast<uintptr_t>(storage) & 63u) == 0,
                  "storage 首地址必须 64B 对齐，使五个协议对象各自独占 cache line");
    uint32_t mode = 0;
    if (!atomic_probe::RequiredUintEnv("ATOMIC_PROBE_MODE", 5, &mode)) {
        return EXIT_FAILURE;
    }
    const char *labels[] = {"DEFAULT", "ALL", "OUT", "ATOMIC", "NO DCCI"};
    run_mode(function, stream, storage, mode, labels[mode], result);

    check(aclrtFree(storage), "free probe storage");
    check(aclrtBinaryUnLoad(binary_handle), "unload binary");
    check(aclrtDestroyStream(stream), "destroy stream");
    check(aclrtResetDevice(device_id), "reset device");
    check(aclFinalize(), "aclFinalize");
    return result.ExitCode();
}
