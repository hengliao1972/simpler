// 测试目标：精确比较“读核只读、目标 cache line 始终 clean”时 DCCI 第三个参数的行为，
// 同时验证两参数调用的默认值是否与显式 CACHELINE_ALL 一致。
//
// 参与者固定为两个 AIV，data、ready、done、writer marker、reader result 各自独占
// 一条 64B cache line。两个核对 data 的访问由 ready/done atomic phase 严格串行，
// 不存在同时读写 data line；读核在整个用例中绝不写 data line。
//
// 每轮时序：
//   1. 读核用普通 scalar load 预读旧 data，使本核持有 clean cache line；DSB 后用
//      atomicMax(ready, round+1) 把写权限交给写核。
//   2. 写核看到 ready 后，用 st_dev 更新整条 data；DSB 后用
//      atomicMax(done, round+1) 发布新值。
//   3. 读核看到 done 后先执行 DSB，再执行本 mode 的 DCCI 和 DSB，随后分别用普通
//      load 与 ld_dev
//      读取。普通 load 检验本核 cache 是否更新，ld_dev 检验 GM 是否仍保留写核新值。
//   4. 下一轮 ready 只能在本轮检查完成后发布，因此写核不会提前覆盖下一轮数据。
//
// Mode 与预期：
//   0 DEFAULT：调用两参数 dcci(data, SINGLE_CACHE_LINE)。普通 load 每轮必须
//     读到新值，ld_dev 每轮也必须为新值，且不得出现 torn/impossible line。
//   1 ALL：显式调用三参数 dcci(..., CACHELINE_ALL)，结果必须与 mode 0 一致。
//   2 OUT：显式调用三参数 dcci(..., CACHELINE_OUT)。普通 load 每轮必须读到新值，
//     ld_dev 每轮也必须为新值；它只验证 OUT selector 对当前 ordinary clean entry 的
//     实际结果，不凭 selector 名称预设内部 clean/invalidate 动作。
//   3 ATOMIC：显式调用三参数 dcci(..., CACHELINE_ATOMIC)。本机 A5 重复板测后，
//     与前三种 DCCI 一样收紧为每轮普通 load 都必须读到本轮新值。
//   4 NO DCCI：只用于证明 stale-cache 前提；普通 load 每轮应精确保留本轮预读值，
//     而 ld_dev 仍应看到写核新值。它只做前提对照，不代表一种 DCCI 方案。
#include "ccec_utils.h"

constexpr uint32_t DATA_ELEMS = 16;
constexpr uint32_t NUM_ROUNDS = 100;
constexpr uint32_t CACHELINE_WORDS = 16;
constexpr uint32_t DATA_OFFSET = 0;
constexpr uint32_t READY_OFFSET = CACHELINE_WORDS;
constexpr uint32_t DONE_OFFSET = 2 * CACHELINE_WORDS;
constexpr uint32_t WRITER_OFFSET = 3 * CACHELINE_WORDS;
constexpr uint32_t RESULT_OFFSET = 4 * CACHELINE_WORDS;

enum ResultIndex : uint32_t {
    RESULT_FRESH = 0,
    RESULT_STALE = 1,
    RESULT_NORMAL_OTHER = 2,
    RESULT_GM_BAD = 3,
    RESULT_READER_MARKER = 4,
    RESULT_READER_ROUNDS = 5,
    RESULT_FIRST_NORMAL_BAD_ROUND = 6,
    RESULT_FIRST_NORMAL_BAD_VALUE = 7,
    RESULT_FIRST_GM_BAD_ROUND = 8,
    RESULT_FIRST_GM_BAD_VALUE = 9,
    RESULT_PRELOAD_CURRENT = 10,
};

CCEC_PROBE_KERNEL_META(dcci_seam);

__aicore__ inline uint32_t round_base(uint32_t round)
{
    return 0xA5000000u + round * 0x100u;
}

extern "C" __global__ __aicore__ void KERNEL_ENTRY(dcci_seam)(
    __gm__ uint32_t *gx, uint32_t mode, uint32_t num_blocks)
{
    uint32_t bid = get_block_idx();
    __gm__ uint32_t *data = &gx[DATA_OFFSET];
    __gm__ uint32_t *ready = &gx[READY_OFFSET];
    __gm__ uint32_t *done = &gx[DONE_OFFSET];
    __gm__ uint32_t *writer_marker = &gx[WRITER_OFFSET];
    __gm__ uint32_t *result = &gx[RESULT_OFFSET];

    ccec_barrier(gx, num_blocks, 0);

    if (bid == 0) {
        for (uint32_t round = 0; round < NUM_ROUNDS; ++round) {
            uint32_t target = round + 1;
            while (atomicMax(ready, 0u) < target) {
            }

            uint32_t base = round_base(round);
            for (uint32_t i = 0; i < DATA_ELEMS; ++i) {
                st_dev_b32(&data[i], base + i);
            }
            // The atomic publication is not used as an implicit st_dev fence.
            dsb(DSB_ALL);
            atomicMax(done, target);
        }
        atomicAdd(writer_marker, 1u);
    } else if (bid == 1) {
        uint32_t fresh = 0;
        uint32_t stale = 0;
        uint32_t normal_other = 0;
        uint32_t gm_bad = 0;
        uint32_t first_normal_bad_round = 0xffffffffu;
        uint32_t first_normal_bad_value = 0;
        uint32_t first_gm_bad_round = 0xffffffffu;
        uint32_t first_gm_bad_value = 0;
        uint32_t preload_current = 0;
        volatile __gm__ uint32_t *volatile_data = data;

        for (uint32_t round = 0; round < NUM_ROUNDS; ++round) {
            // Explicitly leave the writer's previous value resident and clean.
            uint32_t preload[DATA_ELEMS];
            bool preload_is_current = true;
            uint32_t base = round_base(round);
            for (uint32_t i = 0; i < DATA_ELEMS; ++i) {
                preload[i] = volatile_data[i];
                preload_is_current = preload_is_current && preload[i] == base + i;
            }
            preload_current += preload_is_current ? 1u : 0u;
            dsb(DSB_ALL);
            uint32_t target = round + 1;
            atomicMax(ready, target);

            while (atomicMax(done, 0u) < target) {
            }
            dsb(DSB_ALL);
            if (mode == 0) {
                dcci(data, SINGLE_CACHE_LINE);
            } else if (mode == 1) {
                dcci(data, SINGLE_CACHE_LINE, CACHELINE_ALL);
            } else if (mode == 2) {
                dcci(data, SINGLE_CACHE_LINE, CACHELINE_OUT);
            } else if (mode == 3) {
                dcci(data, SINGLE_CACHE_LINE, CACHELINE_ATOMIC);
            }
            dsb(DSB_ALL);

            uint32_t normal_first = volatile_data[0];
            bool normal_is_fresh = normal_first == base;
            bool normal_is_preload = normal_first == preload[0];
            for (uint32_t i = 1; i < DATA_ELEMS; ++i) {
                uint32_t value = volatile_data[i];
                normal_is_fresh = normal_is_fresh && value == base + i;
                normal_is_preload = normal_is_preload && value == preload[i];
            }

            if (normal_is_fresh) {
                ++fresh;
            } else if (normal_is_preload) {
                ++stale;
            } else {
                ++normal_other;
            }
            if (!normal_is_fresh && first_normal_bad_round == 0xffffffffu) {
                first_normal_bad_round = round;
                first_normal_bad_value = normal_first;
            }

            bool gm_is_fresh = true;
            uint32_t gm_first = ld_dev_b32(&data[0]);
            gm_is_fresh = gm_is_fresh && gm_first == base;
            for (uint32_t i = 1; i < DATA_ELEMS; ++i) {
                gm_is_fresh = gm_is_fresh && ld_dev_b32(&data[i]) == base + i;
            }
            if (!gm_is_fresh) {
                ++gm_bad;
                if (first_gm_bad_round == 0xffffffffu) {
                    first_gm_bad_round = round;
                    first_gm_bad_value = gm_first;
                }
            }
        }

        st_dev_b32(&result[RESULT_FRESH], fresh);
        st_dev_b32(&result[RESULT_STALE], stale);
        st_dev_b32(&result[RESULT_NORMAL_OTHER], normal_other);
        st_dev_b32(&result[RESULT_GM_BAD], gm_bad);
        st_dev_b32(&result[RESULT_READER_MARKER], 0xC001CAFEu);
        st_dev_b32(&result[RESULT_READER_ROUNDS], NUM_ROUNDS);
        st_dev_b32(&result[RESULT_FIRST_NORMAL_BAD_ROUND], first_normal_bad_round);
        st_dev_b32(&result[RESULT_FIRST_NORMAL_BAD_VALUE], first_normal_bad_value);
        st_dev_b32(&result[RESULT_FIRST_GM_BAD_ROUND], first_gm_bad_round);
        st_dev_b32(&result[RESULT_FIRST_GM_BAD_VALUE], first_gm_bad_value);
        st_dev_b32(&result[RESULT_PRELOAD_CURRENT], preload_current);
        dsb(DSB_ALL);
    }

    ccec_barrier(gx, num_blocks, NUM_ROUNDS + 1);
}
