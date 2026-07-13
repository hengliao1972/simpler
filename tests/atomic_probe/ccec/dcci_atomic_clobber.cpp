// 测试目标：精确区分 DCCI 的第三参数 CACHELINE_ALL、CACHELINE_OUT、
// CACHELINE_ATOMIC 在“普通 dirty cache line 与 atomic 更新相遇”时的行为，并用
// atomic 目标独占 cache line 的对照组验证隔离规则。
//
// 参与者固定为两个 AIV。data、separate target、ready、done、核1结果、data 快照、
// target 快照、核0元数据各自独占一条 64B cache line；除 mode 0~3 故意把 atomic
// target 放进 data line 外，控制信息绝不与 DCCI 的 data line 共线。
//
// 每个 mode 的严格时序：
//   1. 核0用普通 scalar load 预读完整 data line，再普通 scalar store 邻接 slot，
//      从而留下包含 atomic 目标旧值的 dirty cache line；DSB 后通过独立 ready line
//      上的 atomicMax 把执行权交给核1。
//   2. 核1看到 ready 后，对目标 slot 执行一次 AtomicExch；保存交换返回的旧值，
//      DSB 后通过独立 done line 把执行权交回核0。
//   3. 核0看到 done 后先 DSB，再按 mode 对 data line 执行一次 DCCI（或不执行），
//      再 DSB，最后用 ld_dev 保存 data 与 separate target 的完整 16-word GM 快照。
//
// Mode：
//   0 ALL，同 line；      1 OUT，同 line；      2 ATOMIC，同 line；
//   3 NO DCCI，同 line；  4 ALL，分 line；      5 OUT，分 line；
//   6 ATOMIC，分 line；   7 NO DCCI，分 line。
//
// 精确判定：所有 mode 都检查两条 16-word 快照、ready/done、两个核的 marker、
// AtomicExch 返回的旧值和所有未使用 slot。mode 3/7 必须严格命中“普通 dirty store
// 未发布、atomic 新值已发布”的无 DCCI 对照结果。mode 0/1/2 是正确性看护：DCCI
// 发布 dirty line 后 atomic 新值仍必须保留；当前若复现整 line 旧快照覆盖则必须失败。
// mode 4/5/6 必须精确验证 dirty data 已发布且分-line atomic 新值保留。任何 torn 或
// 第四种状态都直接失败。
#include "ccec_utils.h"

constexpr uint32_t CACHELINE_WORDS = 16;
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

CCEC_PROBE_KERNEL_META(dcci_atomic_clobber);

__aicore__ inline bool IsSameLineMode(uint32_t mode)
{
    return mode < 4;
}

__aicore__ inline void ApplySelectedDcci(__gm__ uint32_t *data, uint32_t mode)
{
    if (mode == 0 || mode == 4) {
        dcci(data, SINGLE_CACHE_LINE, CACHELINE_ALL);
    } else if (mode == 1 || mode == 5) {
        dcci(data, SINGLE_CACHE_LINE, CACHELINE_OUT);
    } else if (mode == 2 || mode == 6) {
        dcci(data, SINGLE_CACHE_LINE, CACHELINE_ATOMIC);
    }
}

extern "C" __global__ __aicore__ void KERNEL_ENTRY(dcci_atomic_clobber)(
    __gm__ uint32_t *storage, uint32_t mode, uint32_t num_blocks)
{
    (void)num_blocks;
    const uint32_t bid = get_block_idx();
    __gm__ uint32_t *data = &storage[DATA_OFFSET];
    __gm__ uint32_t *separate_target = &storage[SEPARATE_TARGET_OFFSET];
    __gm__ uint32_t *ready = &storage[READY_OFFSET];
    __gm__ uint32_t *done = &storage[DONE_OFFSET];
    __gm__ uint32_t *core1_result = &storage[CORE1_RESULT_OFFSET];
    __gm__ uint32_t *data_snapshot = &storage[DATA_SNAPSHOT_OFFSET];
    __gm__ uint32_t *target_snapshot = &storage[TARGET_SNAPSHOT_OFFSET];
    __gm__ uint32_t *core0_meta = &storage[CORE0_META_OFFSET];

    if (bid == 0) {
        volatile __gm__ uint32_t *volatile_data = data;
        volatile __gm__ uint32_t *volatile_target = separate_target;
        uint32_t data_preload_bad = 0;
        uint32_t target_preload_bad = 0;
        for (uint32_t i = 0; i < CACHELINE_WORDS; ++i) {
            if (volatile_data[i] != DATA_INIT_BASE + i) {
                ++data_preload_bad;
            }
            if (volatile_target[i] != TARGET_INIT_BASE + i) {
                ++target_preload_bad;
            }
        }

        // 与 atomic 目标不同 slot；普通 store 故意让 data line 保持 dirty。
        volatile_data[DIRTY_SLOT] = DIRTY_VALUE;
        dsb(DSB_ALL);
        atomicMax(ready, 1u);

        while (atomicMax(done, 0u) < 1u) {
        }
        dsb(DSB_ALL);
        ApplySelectedDcci(data, mode);
        dsb(DSB_ALL);

        for (uint32_t i = 0; i < CACHELINE_WORDS; ++i) {
            st_dev_b32(&data_snapshot[i], ld_dev_b32(&data[i]));
            st_dev_b32(&target_snapshot[i], ld_dev_b32(&separate_target[i]));
        }
        st_dev_b32(&core0_meta[0], data_preload_bad);
        st_dev_b32(&core0_meta[1], target_preload_bad);
        st_dev_b32(&core0_meta[2], CORE0_MARKER);
        st_dev_b32(&core0_meta[3], mode);
        dsb(DSB_ALL);
    } else if (bid == 1) {
        while (atomicMax(ready, 0u) < 1u) {
        }
        dsb(DSB_ALL);

        __gm__ uint32_t *target = IsSameLineMode(mode) ? &data[ATOMIC_SLOT]
                                                       : &separate_target[ATOMIC_SLOT];
        const uint32_t old_value = atomicExch(target, ATOMIC_VALUE);
        st_dev_b32(&core1_result[0], old_value);
        st_dev_b32(&core1_result[1], CORE1_MARKER);
        st_dev_b32(&core1_result[2], mode);
        st_dev_b32(&core1_result[3], ATOMIC_VALUE);
        dsb(DSB_ALL);
        atomicMax(done, 1u);
    }
}
