// 测试目标：验证 dirty stale cache line 在后续 DCCI clean 时是否覆盖其他核的新值。
//
// 该用例不是并发竞态。每一步都由跨 AIV barrier 严格排序：核0先形成自己的缓存行，
// 核1随后用 st_dev 更新同一 64B line 的其他 slot，核0最后才执行 DCCI。问题若出现，
// 原因是核0把包含 stale slot 的整条 dirty line 写回，而不是两个核同时访问。
//
// Mode 与预期：
//   0 DIRTY + DEFAULT：核0 scalar store gx[0] 使 line dirty；核1 st_dev gx[3]；
//     核0默认 DCCI 后，预期 gx[3] 被旧值精确覆盖，用来暴露整 line writeback 风险。
//   1 CLEAN + DEFAULT：核0只 normal-read、不写 data line；核1 st_dev gx[3]；
//     核0默认 DCCI 后，预期 GM 中 gx[3] 保持核1新值，不发生 clean 回写覆盖。
//   2 DIRTY + OUT：与 mode 0 相同的 dirty stale 前提，但显式 CACHELINE_OUT；
//     预期同样发生整 line writeback 覆盖。
//   3 NO DCCI：核0保留 dirty line 但不执行 DCCI；预期核1的 st_dev 值继续留在 GM，
//     同时核0的普通 scalar store 尚未发布到 GM。
//   4 FULL 64B：核1更新 word[1..15]，核0随后默认 DCCI；预期 15 个新值全部被覆盖，
//     用整条 line 的精确结果确认覆盖粒度。
//
#include "ccec_utils.h"

CCEC_PROBE_KERNEL_META(dcci_clean_probe);

extern "C" __global__ __aicore__ void KERNEL_ENTRY(dcci_clean_probe)(
    __gm__ uint32_t *gx, uint32_t mode, uint32_t num_blocks)
{
    uint32_t bid = get_block_idx();

    // ================================================================
    // Mode 0：dirty line + 默认 DCCI，检查 st_dev 新值是否被旧快照覆盖
    // ================================================================
    if (mode == 0) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 0) {
            gx[0] = 0xAA;  // L1 now dirty: word[0]=0xAA, word[1..15]=0(stale)
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 1) {
            st_dev_b32(&gx[3], 0xCAFEBABEu);
        }
        ccec_barrier(gx, num_blocks, 3);
        if (bid == 0) {
            dcci(gx, SINGLE_CACHE_LINE);  // 默认第三参数（编码与 CACHELINE_ALL 相同）
        }
        ccec_barrier(gx, num_blocks, 4);
        if (bid == 1) {
            uint32_t w0 = ld_dev_b32(&gx[0]);  // expect 0xAA
            uint32_t w3 = ld_dev_b32(&gx[3]);  // expect 0 (CLOBBERED!)
            st_dev_b32(&gx[16], (w0 == 0xAA) ? 0 : 1);
            st_dev_b32(&gx[17], (w3 == 0) ? 1 : 0);
            st_dev_b32(&gx[18], w0);
            st_dev_b32(&gx[19], w3);
            st_dev_b32(&gx[20], 0xCAFEBABEu);
        }
        ccec_barrier(gx, num_blocks, 5);

    // ================================================================
    // Mode 1：clean line + 默认 DCCI，检查 st_dev 新值是否保留
    // ================================================================
    } else if (mode == 1) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 0) {
            volatile __gm__ uint32_t *vgx = gx;
            uint32_t s = vgx[0] + vgx[3];  // read-only, L1 stays clean
            (void)s;
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 1) {
            st_dev_b32(&gx[3], 0xCAFEBABEu);
        }
        ccec_barrier(gx, num_blocks, 3);
        if (bid == 0) {
            dcci(gx, SINGLE_CACHE_LINE);
        }
        ccec_barrier(gx, num_blocks, 4);
        if (bid == 1) {
            uint32_t w3 = ld_dev_b32(&gx[3]);
            st_dev_b32(&gx[16], (w3 == 0xCAFEBABEu) ? 0 : 1);
            st_dev_b32(&gx[17], 0);
            st_dev_b32(&gx[18], 0);
            st_dev_b32(&gx[19], w3);
            st_dev_b32(&gx[20], 0xCAFEBABEu);
        }
        ccec_barrier(gx, num_blocks, 5);

    // ================================================================
    // Mode 2：dirty line + CACHELINE_OUT，检查 st_dev 新值是否被旧快照覆盖
    // ================================================================
    } else if (mode == 2) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 0) {
            gx[0] = 0xAA;  // dirty L1
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 1) {
            st_dev_b32(&gx[3], 0xCAFEBABEu);
        }
        ccec_barrier(gx, num_blocks, 3);
        if (bid == 0) {
            dcci(gx, SINGLE_CACHE_LINE, CACHELINE_OUT);  // 显式选择 DcciDst::OUT
        }
        ccec_barrier(gx, num_blocks, 4);
        if (bid == 1) {
            uint32_t w0 = ld_dev_b32(&gx[0]);
            uint32_t w3 = ld_dev_b32(&gx[3]);
            st_dev_b32(&gx[16], (w0 == 0xAA) ? 0 : 1);
            st_dev_b32(&gx[17], (w3 == 0) ? 1 : 0);
            st_dev_b32(&gx[18], w0);
            st_dev_b32(&gx[19], w3);
            st_dev_b32(&gx[20], 0xCAFEBABEu);
        }
        ccec_barrier(gx, num_blocks, 5);

    // ================================================================
    // Mode 3：不执行 DCCI 的对照，检查 st_dev 新值是否保留
    // ================================================================
    } else if (mode == 3) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 0) {
            gx[0] = 0xAA;  // dirty L1, no dcci
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 1) {
            st_dev_b32(&gx[3], 0xCAFEBABEu);
        }
        ccec_barrier(gx, num_blocks, 3);
        // Block 0 does NOT call dcci
        ccec_barrier(gx, num_blocks, 4);
        if (bid == 1) {
            uint32_t w0 = ld_dev_b32(&gx[0]);
            uint32_t w3 = ld_dev_b32(&gx[3]);
            st_dev_b32(&gx[16], (w0 == 0) ? 0 : 1);
            st_dev_b32(&gx[17], (w3 == 0xCAFEBABEu) ? 0 : 1);
            st_dev_b32(&gx[18], w0);
            st_dev_b32(&gx[19], w3);
            st_dev_b32(&gx[20], 0xCAFEBABEu);
        }
        ccec_barrier(gx, num_blocks, 5);

    // ================================================================
    // Mode 4：dirty line + 默认 DCCI，逐 word 检查完整 64B
    // ================================================================
    } else if (mode == 4) {
        ccec_barrier(gx, num_blocks, 1);
        if (bid == 0) {
            gx[0] = 0xAA;  // dirty L1, rest stale zeros
        }
        ccec_barrier(gx, num_blocks, 2);
        if (bid == 1) {
            for (uint32_t w = 1; w < 16; w++)
                st_dev_b32(&gx[w], 0xB000u + w);
        }
        ccec_barrier(gx, num_blocks, 3);
        if (bid == 0) {
            dcci(gx, SINGLE_CACHE_LINE);  // 检查是否写回 [0xAA, 0, 0, ..., 0]
        }
        ccec_barrier(gx, num_blocks, 4);
        if (bid == 1) {
            uint32_t clobbered = 0;
            for (uint32_t w = 1; w < 16; w++) {
                uint32_t v = ld_dev_b32(&gx[w]);
                if (v != 0xB000u + w) clobbered++;
            }
            st_dev_b32(&gx[16], clobbered);
            st_dev_b32(&gx[17], ld_dev_b32(&gx[0]));
        }
        ccec_barrier(gx, num_blocks, 5);
    }
}
