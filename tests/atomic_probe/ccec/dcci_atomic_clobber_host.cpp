// CCEC dcci_atomic_clobber 的 host 精确判定。
// 每个 Expect 都描述被检查的精确时序事实；selector mode 只接受三种完整状态，
// 不用“看起来正常”等模糊条件掩盖 torn line 或未解释状态。
#include "../probe_host.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <time.h>
#include <vector>

constexpr uint32_t CACHELINE_WORDS = 16;
constexpr uint32_t STORAGE_WORDS = 8 * CACHELINE_WORDS;
constexpr uint32_t DATA_OFFSET = 0 * CACHELINE_WORDS;
constexpr uint32_t SEPARATE_TARGET_OFFSET = 1 * CACHELINE_WORDS;
constexpr uint32_t READY_OFFSET = 2 * CACHELINE_WORDS;
constexpr uint32_t DONE_OFFSET = 3 * CACHELINE_WORDS;
constexpr uint32_t CORE1_RESULT_OFFSET = 4 * CACHELINE_WORDS;
constexpr uint32_t DATA_SNAPSHOT_OFFSET = 5 * CACHELINE_WORDS;
constexpr uint32_t TARGET_SNAPSHOT_OFFSET = 6 * CACHELINE_WORDS;
constexpr uint32_t CORE0_META_OFFSET = 7 * CACHELINE_WORDS;

constexpr uint32_t DIRTY_SLOT = 3;
constexpr uint32_t ATOMIC_SLOT = 0;
constexpr uint32_t DATA_INIT_BASE = 0x11000000u;
constexpr uint32_t TARGET_INIT_BASE = 0x22000000u;
constexpr uint32_t DIRTY_VALUE = 0xD17A0000u;
constexpr uint32_t ATOMIC_VALUE = 0xA70C0003u;
constexpr uint32_t CORE0_MARKER = 0xC0DEC000u;
constexpr uint32_t CORE1_MARKER = 0xC0DEC001u;

struct KernelArgs {
    uint64_t storage_pointer;
    uint32_t mode;
    uint32_t num_blocks;
};

static_assert(sizeof(KernelArgs) == 16, "unexpected CCEC kernel argument ABI");

static void Check(aclError error, const char *message)
{
    if (error != ACL_SUCCESS) {
        std::fprintf(stderr, "ACL error %d: %s\n", error, message);
        std::exit(EXIT_FAILURE);
    }
}

static bool EqualLines(const uint32_t *left, const uint32_t *right)
{
    for (uint32_t i = 0; i < CACHELINE_WORDS; ++i) {
        if (left[i] != right[i]) {
            return false;
        }
    }
    return true;
}

static bool ExactDataState(const uint32_t *actual, bool dirty_published,
                           bool atomic_new, bool same_line)
{
    for (uint32_t i = 0; i < CACHELINE_WORDS; ++i) {
        uint32_t expected = DATA_INIT_BASE + i;
        if (i == DIRTY_SLOT && dirty_published) {
            expected = DIRTY_VALUE;
        }
        if (same_line && i == ATOMIC_SLOT && atomic_new) {
            expected = ATOMIC_VALUE;
        }
        if (actual[i] != expected) {
            return false;
        }
    }
    return true;
}

static bool ExactTargetState(const uint32_t *actual, bool atomic_new)
{
    for (uint32_t i = 0; i < CACHELINE_WORDS; ++i) {
        uint32_t expected = TARGET_INIT_BASE + i;
        if (i == ATOMIC_SLOT && atomic_new) {
            expected = ATOMIC_VALUE;
        }
        if (actual[i] != expected) {
            return false;
        }
    }
    return true;
}

static bool ExactSingleWordLine(const uint32_t *line, uint32_t value)
{
    if (line[0] != value) {
        return false;
    }
    for (uint32_t i = 1; i < CACHELINE_WORDS; ++i) {
        if (line[i] != 0) {
            return false;
        }
    }
    return true;
}

static bool ExactCore1Result(const uint32_t *line, uint32_t old_value, uint32_t mode)
{
    if (line[0] != old_value || line[1] != CORE1_MARKER || line[2] != mode ||
        line[3] != ATOMIC_VALUE) {
        return false;
    }
    for (uint32_t i = 4; i < CACHELINE_WORDS; ++i) {
        if (line[i] != 0) {
            return false;
        }
    }
    return true;
}

static bool ExactCore0Meta(const uint32_t *line, uint32_t mode)
{
    if (line[0] != 0 || line[1] != 0 || line[2] != CORE0_MARKER || line[3] != mode) {
        return false;
    }
    for (uint32_t i = 4; i < CACHELINE_WORDS; ++i) {
        if (line[i] != 0) {
            return false;
        }
    }
    return true;
}

static void RunMode(aclrtFuncHandle function, aclrtStream stream, void *device_storage,
                    uint32_t mode, const char *label, atomic_probe::Result &result)
{
    uint32_t initial[STORAGE_WORDS] = {};
    for (uint32_t i = 0; i < CACHELINE_WORDS; ++i) {
        initial[DATA_OFFSET + i] = DATA_INIT_BASE + i;
        initial[SEPARATE_TARGET_OFFSET + i] = TARGET_INIT_BASE + i;
    }
    Check(aclrtMemcpy(device_storage, sizeof(initial), initial, sizeof(initial),
                      ACL_MEMCPY_HOST_TO_DEVICE),
          "初始化 dcci_atomic_clobber 的八条 cache line");

    struct timespec start = {}, end = {};
    clock_gettime(CLOCK_MONOTONIC, &start);
    KernelArgs args{static_cast<uint64_t>(reinterpret_cast<uintptr_t>(device_storage)), mode, 2};
    Check(aclrtLaunchKernelWithHostArgs(function, 2, stream, nullptr, &args, sizeof(args),
                                        nullptr, 0),
          "启动 dcci_atomic_clobber 双 AIV kernel");
    Check(aclrtSynchronizeStream(stream), "等待 dcci_atomic_clobber 双 AIV kernel 完成");
    clock_gettime(CLOCK_MONOTONIC, &end);

    uint32_t values[STORAGE_WORDS] = {};
    Check(aclrtMemcpy(values, sizeof(values), device_storage, sizeof(values),
                      ACL_MEMCPY_DEVICE_TO_HOST),
          "读取 dcci_atomic_clobber 的完整结果");
    const double elapsed_us = (end.tv_sec - start.tv_sec) * 1e6 +
                              (end.tv_nsec - start.tv_nsec) / 1e3;

    const bool same_line = mode < 4;
    const bool no_dcci = mode == 3 || mode == 7;
    const uint32_t expected_old = same_line ? DATA_INIT_BASE + ATOMIC_SLOT
                                            : TARGET_INIT_BASE + ATOMIC_SLOT;
    const uint32_t *data_snapshot = &values[DATA_SNAPSHOT_OFFSET];
    const uint32_t *target_snapshot = &values[TARGET_SNAPSHOT_OFFSET];

    const bool clobbered_writeback =
        same_line && ExactDataState(data_snapshot, true, false, true);
    const bool survived_writeback =
        same_line && ExactDataState(data_snapshot, true, true, true);
    const bool survived_without_publish =
        same_line && ExactDataState(data_snapshot, false, true, true);
    const bool separate_cleaned =
        !same_line && ExactDataState(data_snapshot, true, false, false);
    const bool separate_not_cleaned =
        !same_line && ExactDataState(data_snapshot, false, false, false);

    const char *classification = "未解释状态（精确判定将失败）";
    if (clobbered_writeback) {
        classification = "DCCI 发布普通 dirty line，并用旧快照覆盖 atomic 新值";
    } else if (survived_writeback) {
        classification = "DCCI 发布普通 dirty line，atomic 新值仍保留";
    } else if (survived_without_publish) {
        classification = "普通 dirty entry 未发布，atomic 新值仍保留";
    } else if (separate_cleaned) {
        classification = "DCCI 发布 data dirty line；独立 atomic line 保留新值";
    } else if (separate_not_cleaned) {
        classification = "data dirty entry 未发布；独立 atomic line 保留新值";
    }

    std::printf("[%-24s] 分类=%s data[%u](dirty)=0x%08x data[%u](atomic)=0x%08x "
                "separate[%u](atomic)=0x%08x atomic_old=0x%08x %.0fus\n",
                label, classification, DIRTY_SLOT, data_snapshot[DIRTY_SLOT],
                ATOMIC_SLOT, data_snapshot[ATOMIC_SLOT], ATOMIC_SLOT,
                target_snapshot[ATOMIC_SLOT],
                values[CORE1_RESULT_OFFSET], elapsed_us);

    result.Expect(ExactSingleWordLine(&values[READY_OFFSET], 1),
                  "ready 独占 64B：仅 word[0] 为 1，证明核0只发布一次写权限");
    result.Expect(ExactSingleWordLine(&values[DONE_OFFSET], 1),
                  "done 独占 64B：仅 word[0] 为 1，证明核1只发布一次 DCCI 权限");
    result.Expect(ExactCore1Result(&values[CORE1_RESULT_OFFSET], expected_old, mode),
                  "核1结果行精确：AtomicExch 返回目标初始化值，marker/mode/new-value 均匹配且尾部全零");
    result.Expect(ExactCore0Meta(&values[CORE0_META_OFFSET], mode),
                  "核0元数据行精确：data/target 各 16 个预读值全对，marker/mode 匹配且尾部全零");
    result.Expect(EqualLines(&values[DATA_OFFSET], data_snapshot),
                  "kernel 返回后的 data GM 与核0 DCCI 后保存的 16-word ld_dev 快照逐字一致");
    result.Expect(EqualLines(&values[SEPARATE_TARGET_OFFSET], target_snapshot),
                  "kernel 返回后的独立 target GM 与核0保存的 16-word ld_dev 快照逐字一致");

    if (same_line) {
        result.Expect(ExactTargetState(target_snapshot, false),
                      "同-line mode 未访问独立 target line：其 16 个 word 必须逐字保持初始化值");
        result.Expect(clobbered_writeback || survived_writeback || survived_without_publish,
                      "同-line data 快照必须精确属于：覆盖 atomic、回写且保留 atomic、未发布 dirty 且保留 atomic；拒绝 torn/第四种状态");
    } else {
        result.Expect(ExactTargetState(target_snapshot, true),
                      "分-line atomic 快照精确：仅目标 word 为新值，其余 15 个 word 保持初始化值");
        result.Expect(separate_cleaned || separate_not_cleaned,
                      "分-line data 快照必须精确属于：dirty line 已发布或未发布；拒绝任意部分写回/torn 状态");
    }

    if (no_dcci && same_line) {
        result.Expect(survived_without_publish,
                      "同-line 无 DCCI 对照精确：普通 dirty store 未进入 GM，AtomicExch 新值已进入 GM");
    } else if (no_dcci) {
        result.Expect(separate_not_cleaned,
                      "分-line 无 DCCI 对照精确：data 普通 dirty store 未进入 GM，独立 AtomicExch 新值已进入 GM");
    } else if (same_line) {
        result.Expect(survived_writeback,
                      "同-line正确性门禁：DCCI 发布 dirty 值后，AtomicExch 新值仍必须保留；被旧快照覆盖即失败");
    } else {
        result.Expect(separate_cleaned,
                      "分-line selector：DCCI 必须发布 data dirty 值，独立 atomic line 的新值必须保留");
    }
}

int main(int argc, char *argv[])
{
    const char *kernel_path = "./dcci_atomic_clobber_kernel.o";
    if (argc > 1) {
        kernel_path = argv[1];
    }

    int32_t device_id = atomic_probe::DeviceId();
    if (device_id < 0) {
        return EXIT_FAILURE;
    }
    Check(aclInit(nullptr), "初始化 ACL");
    Check(aclrtSetDevice(device_id), "设置 atomic probe 设备");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "创建 dcci_atomic_clobber stream");

    std::ifstream file(kernel_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "无法打开 kernel 文件：%s\n", kernel_path);
        return EXIT_FAILURE;
    }
    const size_t binary_size = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<char> binary(binary_size);
    file.read(binary.data(), static_cast<std::streamsize>(binary_size));

    aclrtBinHandle binary_handle;
    Check(atomic_probe::LoadAicoreBinaryFromData(binary.data(), binary_size, &binary_handle),
          "加载 dcci_atomic_clobber AICore binary");
    aclrtFuncHandle function;
    Check(aclrtBinaryGetFunctionByEntry(binary_handle, 0, &function),
          "取得 dcci_atomic_clobber kernel 入口");

    void *device_storage = nullptr;
    Check(aclrtMalloc(&device_storage, STORAGE_WORDS * sizeof(uint32_t),
                      ACL_MEM_MALLOC_HUGE_FIRST),
          "分配八条 64B 对齐 protocol cache line");

    if ((reinterpret_cast<uintptr_t>(device_storage) & 63u) != 0u) {
        std::fprintf(stderr,
                     "protocol storage 首地址未按 64B 对齐，不能执行 cacheline 隔离测试\n");
        Check(aclrtFree(device_storage), "释放未对齐 protocol storage");
        Check(aclrtBinaryUnLoad(binary_handle), "卸载 AICore binary");
        Check(aclrtDestroyStream(stream), "销毁 stream");
        Check(aclrtResetDevice(device_id), "重置 probe 设备");
        Check(aclFinalize(), "结束 ACL");
        return EXIT_FAILURE;
    }

    atomic_probe::Result result;

    uint32_t mode = 0;
    if (!atomic_probe::RequiredUintEnv("ATOMIC_PROBE_MODE", 8, &mode)) {
        return EXIT_FAILURE;
    }
    const char *labels[] = {
        "ALL 同 line",      "OUT 同 line",      "ATOMIC 同 line", "无 DCCI 同 line",
        "ALL 分 line",      "OUT 分 line",      "ATOMIC 分 line", "无 DCCI 分 line",
    };
    std::printf("=== CCEC DCCI selector × AtomicExch 精确状态矩阵 ===\n");
    RunMode(function, stream, device_storage, mode, labels[mode], result);

    Check(aclrtFree(device_storage), "释放 protocol storage");
    Check(aclrtBinaryUnLoad(binary_handle), "卸载 AICore binary");
    Check(aclrtDestroyStream(stream), "销毁 stream");
    Check(aclrtResetDevice(device_id), "重置 probe 设备");
    Check(aclFinalize(), "结束 ACL");
    return result.ExitCode();
}
