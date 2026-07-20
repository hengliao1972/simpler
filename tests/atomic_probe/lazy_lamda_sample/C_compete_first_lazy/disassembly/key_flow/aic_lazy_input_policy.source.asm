# schema=pa_final_linked_disassembly/v1
# variant=compete-first-lazy
# final_elf=pa_scheduler_kernel.o
# final_elf_sha256=8d293c52312429d13efe89592ed705c672ceef1f2875b5e3bd25582419d37893
# final_text_address=0x0
# final_text_size=547640
# final_text_sha256=866ed1081ebb94038a8feca87ccedf95e67aebfdaef11651c109f86672be2bdd
# symbol=pa_scheduler_lazy_sample_callback_orchestration_aic
# binding=LOCAL
# final_pc=0x2c0
# size=161792
# instruction_count=40448
# encoded_word_count=40448
# body_sha256=35bcc11cdad126fa5d2f92ff1df421d8556843a412ea3497addfe820d84bd852
# decoder=$ASCEND_HOME_PATH/x86_64-linux/simulator/dav_3510/lib/libpem_davinci.so
# decoder_sha256=29835d2439d6dd464d34a212ad4bbd5c29af6a38465da09a6c273401d9a96dcb
# decoder_mode=scalar
# columns=final_pc function_relative_offset machine_word instruction
# annotation_schema=pa_source_annotated_disassembly/v1
# annotation_rule=DWARF supplies only file:line; SOURCE rows are copied from local source files
# annotation_warning=comments have source context only and do not own an exact machine address
# annotation_instruction_slice=15032:15833
#
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000eda0 (+0x0000eae0)  07690007  MOVK                            X20, #7, #1
0x000000000000eda4 (+0x0000eae4)  07a90000  MOVK                            X20, #0, #2
0x000000000000eda8 (+0x0000eae8)  07e90000  MOVK                            X20, #0, #3
0x000000000000edac (+0x0000eaec)  02020880  MOV_XD_SPR.S64                  X1, PC
0x000000000000edb0 (+0x0000eaf0)  00294081  ADD.S64                         X20, X20, X1
0x000000000000edb4 (+0x0000eaf4)  40000012  JUMP                            #18
0x000000000000edb8 (+0x0000eaf8)  0804e002  ADD_IMM.S64                     X2, X14, #2
0x000000000000edbc (+0x0000eafc)  00c2d78a  AND.B64                         X1, X13, X15
0x000000000000edc0 (+0x0000eb00)  02c4020e  SHL.B64                         X2, #14
0x000000000000edc4 (+0x0000eb04)  0006c381  ADD.S64                         X3, X12, X7
0x000000000000edc8 (+0x0000eb08)  00061104  MADD.S64                        X3, X1, X2
# [DWARF] common/pa_trace.h:280
#     274 |             call_count, static_cast<uint32_t>(site)
#     275 |         );
#     276 |         if (written) {
#     277 |             if (trace.poll_batch_records == UINT64_MAX) {
#     278 |                 trace.atomic_counter_overflow = true;
#     279 |             } else {
# >   280 |                 ++trace.poll_batch_records;
0x000000000000edcc (+0x0000eb0c)  08000001  ADD_IMM.S64                     X0, X0, #1
0x000000000000edd0 (+0x0000eb10)  00023101  ADD.S64                         X1, X3, X2
0x000000000000edd4 (+0x0000eb14)  08821680  SUB_IMM.S64                     X1, X1, #1664
0x000000000000edd8 (+0x0000eb18)  08021000  ADD_IMM.S64                     X1, X1, #0
0x000000000000eddc (+0x0000eb1c)  03c01578  ST_XD_XN_IMM.B64                X0, X1, #1400
# [DWARF] common/pa_trace.h:283
#     281 |             }
#     282 |         }
# >   283 |         trace.poll_burst.call_count[index] = 0;
0x000000000000ede0 (+0x0000eb20)  0f976700  STI_XN_IMM.B32                  X22, #1464
# [DWARF] common/pa_trace.h:262
#     256 | #else
#     257 |     if (!trace.atomics_enabled || trace.poll_burst.active_mask == 0) return;
#     258 |     const uint32_t active_mask = trace.poll_burst.active_mask;
#     259 |     // CCEC 默认会把固定 6-site 循环完整展开，再随几十个 phase 边界复制。
#     260 |     // 禁止展开只控制代码体积；循环次数、site 顺序和同 cycle 关闭语义不变。
#     261 |     PA_LOOP_NOUNROLL
# >   262 |     for (uint32_t index = 0; index < kAtomicPollBatchSiteCount; ++index) {
0x000000000000ede4 (+0x0000eb24)  08318001  ADD_IMM.S64                     X24, X24, #1
0x000000000000ede8 (+0x0000eb28)  08294004  ADD_IMM.S64                     X20, X20, #4
0x000000000000edec (+0x0000eb2c)  0001841e  CMP.S64.NE                      X24, X8
0x000000000000edf0 (+0x0000eb30)  08273001  ADD_IMM.S64                     X19, X19, #1
0x000000000000edf4 (+0x0000eb34)  40200002  JUMPC                           #2
0x000000000000edf8 (+0x0000eb38)  40000050  JUMP                            #80
# [DWARF] common/pa_trace.h:263
# >   263 |         const uint32_t bit = 1U << index;
0x000000000000edfc (+0x0000eb3c)  02813a00  ZEROEXT.U32                     X0, X19
# [DWARF] common/pa_trace.h:264
# >   264 |         if ((active_mask & bit) == 0) continue;
0x000000000000ee00 (+0x0000eb40)  02837a00  ZEROEXT.U32                     X1, X23
0x000000000000ee04 (+0x0000eb44)  024202c0  SHR.U64                         X1, X0, #0
0x000000000000ee08 (+0x0000eb48)  00c0188a  AND.B64                         X0, X1, X17
0x000000000000ee0c (+0x0000eb4c)  0000058e  CMP.S64.EQ                      X0, X11
0x000000000000ee10 (+0x0000eb50)  4020fff5  JUMPC                           #65525
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000ee14 (+0x0000eb54)  0804e002  ADD_IMM.S64                     X2, X14, #2
0x000000000000ee18 (+0x0000eb58)  00c0d78a  AND.B64                         X0, X13, X15
0x000000000000ee1c (+0x0000eb5c)  02c4020e  SHL.B64                         X2, #14
0x000000000000ee20 (+0x0000eb60)  0006c381  ADD.S64                         X3, X12, X7
0x000000000000ee24 (+0x0000eb64)  00060104  MADD.S64                        X3, X0, X2
# [DWARF] common/pa_trace.h:265
#     259 |     // CCEC 默认会把固定 6-site 循环完整展开，再随几十个 phase 边界复制。
#     260 |     // 禁止展开只控制代码体积；循环次数、site 顺序和同 cycle 关闭语义不变。
#     261 |     PA_LOOP_NOUNROLL
#     262 |     for (uint32_t index = 0; index < kAtomicPollBatchSiteCount; ++index) {
#     263 |         const uint32_t bit = 1U << index;
#     264 |         if ((active_mask & bit) == 0) continue;
# >   265 |         const uint32_t call_count = trace.poll_burst.call_count[index];
0x000000000000ee28 (+0x0000eb68)  02038800  MOV_XD_XN.S64                   X1, X24
0x000000000000ee2c (+0x0000eb6c)  00003101  ADD.S64                         X0, X3, X2
0x000000000000ee30 (+0x0000eb70)  08800680  SUB_IMM.S64                     X0, X0, #1664
0x000000000000ee34 (+0x0000eb74)  02c20202  SHL.B64                         X1, #2
0x000000000000ee38 (+0x0000eb78)  08000000  ADD_IMM.S64                     X0, X0, #0
0x000000000000ee3c (+0x0000eb7c)  002c0081  ADD.S64                         X22, X0, X1
0x000000000000ee40 (+0x0000eb80)  1c8b65b8  LD_XD_XN_IMM.B32                X5, X22, #1464
# [DWARF] common/pa_trace.h:266
# >   266 |         if (call_count == 0 || call_count > kAtomicPollCountMax) {
0x000000000000ee44 (+0x0000eb84)  07020000  MOV_XD_IMM                      X1, #0
0x000000000000ee48 (+0x0000eb88)  07430100  MOVK                            X1, #256, #1
0x000000000000ee4c (+0x0000eb8c)  07040000  MOV_XD_IMM                      X2, #0
0x000000000000ee50 (+0x0000eb90)  0745ff00  MOVK                            X2, #65280, #1
0x000000000000ee54 (+0x0000eb94)  00025082  SUB.S64                         X1, X5, X1
0x000000000000ee58 (+0x0000eb98)  02821a00  ZEROEXT.U32                     X1, X1
0x000000000000ee5c (+0x0000eb9c)  0040113e  CMP.U64.GT                      X1, X2
0x000000000000ee60 (+0x0000eba0)  40200003  JUMPC                           #3
# [DWARF] common/pa_trace.h:267
# >   267 |             trace.atomic_counter_overflow = true;
0x000000000000ee64 (+0x0000eba4)  0f160001  STI_XN_IMM.B8                   X0, #1408
0x000000000000ee68 (+0x0000eba8)  4000ffdf  JUMP                            #65503
# [DWARF] common/pa_trace.h:98
#      92 |         default:
#      93 |             return -1;
#      94 |     }
#      95 | }
#      96 | 
#      97 | PA_DEVICE AtomicSite TraceAtomicPollBatchSite(uint32_t index) {
# >    98 |     switch (index) {
0x000000000000ee6c (+0x0000ebac)  02813a00  ZEROEXT.U32                     X0, X19
0x000000000000ee70 (+0x0000ebb0)  07020005  MOV_XD_IMM                      X1, #5
0x000000000000ee74 (+0x0000ebb4)  004000be  CMP.U64.GT                      X0, X1
0x000000000000ee78 (+0x0000ebb8)  070c000f  MOV_XD_IMM                      X6, #15
0x000000000000ee7c (+0x0000ebbc)  40200002  JUMPC                           #2
0x000000000000ee80 (+0x0000ebc0)  1c8d4000  LD_XD_XN_IMM.B32                X6, X20, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000ee84 (+0x0000ebc4)  0804e002  ADD_IMM.S64                     X2, X14, #2
0x000000000000ee88 (+0x0000ebc8)  00c2d78a  AND.B64                         X1, X13, X15
0x000000000000ee8c (+0x0000ebcc)  02c4020e  SHL.B64                         X2, #14
0x000000000000ee90 (+0x0000ebd0)  0006c381  ADD.S64                         X3, X12, X7
0x000000000000ee94 (+0x0000ebd4)  00061104  MADD.S64                        X3, X1, X2
# [DWARF] common/pa_trace.h:273
#     267 |             trace.atomic_counter_overflow = true;
#     268 |             continue;
#     269 |         }
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
#     271 |         const bool written = WritePollBatchRecordRaw(
#     272 |             trace.core, trace.records, trace.capacity,
# >   273 |             trace.poll_burst.start_cycle[index], end_cycle,
0x000000000000ee98 (+0x0000ebd8)  02018800  MOV_XD_XN.S64                   X0, X24
0x000000000000ee9c (+0x0000ebdc)  00023101  ADD.S64                         X1, X3, X2
0x000000000000eea0 (+0x0000ebe0)  08821680  SUB_IMM.S64                     X1, X1, #1664
0x000000000000eea4 (+0x0000ebe4)  02c00203  SHL.B64                         X0, #3
0x000000000000eea8 (+0x0000ebe8)  08241000  ADD_IMM.S64                     X18, X1, #0
0x000000000000eeac (+0x0000ebec)  00012001  ADD.S64                         X0, X18, X0
0x000000000000eeb0 (+0x0000ebf0)  1cc60588  LD_XD_XN_IMM.B64                X3, X0, #1416
# [DWARF] common/pa_trace.h:271
#     265 |         const uint32_t call_count = trace.poll_burst.call_count[index];
#     266 |         if (call_count == 0 || call_count > kAtomicPollCountMax) {
#     267 |             trace.atomic_counter_overflow = true;
#     268 |             continue;
#     269 |         }
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
# >   271 |         const bool written = WritePollBatchRecordRaw(
0x000000000000eeb4 (+0x0000ebf4)  0201a800  MOV_XD_XN.S64                   X0, X26
0x000000000000eeb8 (+0x0000ebf8)  0203b800  MOV_XD_XN.S64                   X1, X27
0x000000000000eebc (+0x0000ebfc)  0205c800  MOV_XD_XN.S64                   X2, X28
0x000000000000eec0 (+0x0000ec00)  02095800  MOV_XD_XN.S64                   X4, X21
0x000000000000eec4 (+0x0000ec04)  070e62fc  MOV_XD_IMM                      X7, #25340
0x000000000000eec8 (+0x0000ec08)  074f0000  MOVK                            X7, #0, #1
0x000000000000eecc (+0x0000ec0c)  078f0000  MOVK                            X7, #0, #2
0x000000000000eed0 (+0x0000ec10)  40427000  CALL                            X7, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000eed4 (+0x0000ec14)  1ccfe990  LD_XD_XN_IMM.B64                X7, X30, #2448
0x000000000000eed8 (+0x0000ec18)  07160000  MOV_XD_IMM                      X11, #0
0x000000000000eedc (+0x0000ec1c)  1cdde9a0  LD_XD_XN_IMM.B64                X14, X30, #2464
0x000000000000eee0 (+0x0000ec20)  071e7fff  MOV_XD_IMM                      X15, #32767
0x000000000000eee4 (+0x0000ec24)  1cdbe9a8  LD_XD_XN_IMM.B64                X13, X30, #2472
# [DWARF] common/pa_trace.h:276
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
#     271 |         const bool written = WritePollBatchRecordRaw(
#     272 |             trace.core, trace.records, trace.capacity,
#     273 |             trace.poll_burst.start_cycle[index], end_cycle,
#     274 |             call_count, static_cast<uint32_t>(site)
#     275 |         );
# >   276 |         if (written) {
0x000000000000eee8 (+0x0000ec28)  0000058e  CMP.S64.EQ                      X0, X11
0x000000000000eeec (+0x0000ec2c)  1cd9e9b0  LD_XD_XN_IMM.B64                X12, X30, #2480
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000eef0 (+0x0000ec30)  07220001  MOV_XD_IMM                      X17, #1
0x000000000000eef4 (+0x0000ec34)  07100006  MOV_XD_IMM                      X8, #6
# [DWARF] common/pa_trace.h:276
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
#     271 |         const bool written = WritePollBatchRecordRaw(
#     272 |             trace.core, trace.records, trace.capacity,
#     273 |             trace.poll_burst.start_cycle[index], end_cycle,
#     274 |             call_count, static_cast<uint32_t>(site)
#     275 |         );
# >   276 |         if (written) {
0x000000000000eef8 (+0x0000ec38)  4020ffba  JUMPC                           #65466
# [DWARF] common/pa_trace.h:277
# >   277 |             if (trace.poll_batch_records == UINT64_MAX) {
0x000000000000eefc (+0x0000ec3c)  1cc12578  LD_XD_XN_IMM.B64                X0, X18, #1400
0x000000000000ef00 (+0x0000ec40)  07020001  MOV_XD_IMM                      X1, #1
0x000000000000ef04 (+0x0000ec44)  02021080  NEG.S64                         X1, X1
0x000000000000ef08 (+0x0000ec48)  0000009e  CMP.S64.NE                      X0, X1
0x000000000000ef0c (+0x0000ec4c)  4020ffab  JUMPC                           #65451
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000ef10 (+0x0000ec50)  0802e002  ADD_IMM.S64                     X1, X14, #2
0x000000000000ef14 (+0x0000ec54)  00c0d78a  AND.B64                         X0, X13, X15
0x000000000000ef18 (+0x0000ec58)  02c2020e  SHL.B64                         X1, #14
0x000000000000ef1c (+0x0000ec5c)  0004c381  ADD.S64                         X2, X12, X7
0x000000000000ef20 (+0x0000ec60)  00040084  MADD.S64                        X2, X0, X1
0x000000000000ef24 (+0x0000ec64)  00002081  ADD.S64                         X0, X2, X1
0x000000000000ef28 (+0x0000ec68)  08800680  SUB_IMM.S64                     X0, X0, #1664
# [DWARF] common/pa_trace.h:278
#     272 |             trace.core, trace.records, trace.capacity,
#     273 |             trace.poll_burst.start_cycle[index], end_cycle,
#     274 |             call_count, static_cast<uint32_t>(site)
#     275 |         );
#     276 |         if (written) {
#     277 |             if (trace.poll_batch_records == UINT64_MAX) {
# >   278 |                 trace.atomic_counter_overflow = true;
0x000000000000ef2c (+0x0000ec6c)  08000000  ADD_IMM.S64                     X0, X0, #0
0x000000000000ef30 (+0x0000ec70)  0f160001  STI_XN_IMM.B8                   X0, #1408
0x000000000000ef34 (+0x0000ec74)  4000ffab  JUMP                            #65451
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000ef38 (+0x0000ec78)  1ccbe968  LD_XD_XN_IMM.B64                X5, X30, #2408
0x000000000000ef3c (+0x0000ec7c)  0802e002  ADD_IMM.S64                     X1, X14, #2
0x000000000000ef40 (+0x0000ec80)  00c0d78a  AND.B64                         X0, X13, X15
0x000000000000ef44 (+0x0000ec84)  02c2020e  SHL.B64                         X1, #14
0x000000000000ef48 (+0x0000ec88)  0004c381  ADD.S64                         X2, X12, X7
0x000000000000ef4c (+0x0000ec8c)  00040084  MADD.S64                        X2, X0, X1
0x000000000000ef50 (+0x0000ec90)  07280001  MOV_XD_IMM                      X20, #1
0x000000000000ef54 (+0x0000ec94)  00002081  ADD.S64                         X0, X2, X1
0x000000000000ef58 (+0x0000ec98)  08800680  SUB_IMM.S64                     X0, X0, #1664
# [DWARF] common/pa_trace.h:285
#     279 |             } else {
#     280 |                 ++trace.poll_batch_records;
#     281 |             }
#     282 |         }
#     283 |         trace.poll_burst.call_count[index] = 0;
#     284 |     }
# >   285 |     trace.poll_burst.active_mask = 0;
0x000000000000ef5c (+0x0000ec9c)  08000000  ADD_IMM.S64                     X0, X0, #0
0x000000000000ef60 (+0x0000eca0)  02294080  NEG.S64                         X20, X20
0x000000000000ef64 (+0x0000eca4)  0f960a00  STI_XN_IMM.B32                  X0, #1488
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000ef68 (+0x0000eca8)  1ce3e920  LD_XD_XN_IMM.B64                X17, X30, #2336
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x000000000000ef6c (+0x0000ecac)  0001a58e  CMP.S64.EQ                      X26, X11
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000ef70 (+0x0000ecb0)  1ce5e988  LD_XD_XN_IMM.B64                X18, X30, #2440
0x000000000000ef74 (+0x0000ecb4)  1cefe978  LD_XD_XN_IMM.B64                X23, X30, #2424
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x000000000000ef78 (+0x0000ecb8)  40200027  JUMPC                           #39
0x000000000000ef7c (+0x0000ecbc)  0001b58e  CMP.S64.EQ                      X27, X11
0x000000000000ef80 (+0x0000ecc0)  40200025  JUMPC                           #37
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000ef84 (+0x0000ecc4)  0283ca00  ZEROEXT.U32                     X1, X28
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x000000000000ef88 (+0x0000ecc8)  0000158e  CMP.S64.EQ                      X1, X11
0x000000000000ef8c (+0x0000eccc)  40200022  JUMPC                           #34
# [DWARF] common/pa_trace.h:547
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
# >   547 |     const uint32_t slot = core.count;
0x000000000000ef90 (+0x0000ecd0)  1c81a000  LD_XD_XN_IMM.B32                X0, X26, #0
# [DWARF] common/pa_trace.h:548
# >   548 |     if (slot >= trace.capacity) {
0x000000000000ef94 (+0x0000ecd4)  004000ae  CMP.U64.LT                      X0, X1
0x000000000000ef98 (+0x0000ecd8)  40200002  JUMPC                           #2
0x000000000000ef9c (+0x0000ecdc)  4000001b  JUMP                            #27
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000efa0 (+0x0000ece0)  0806e002  ADD_IMM.S64                     X3, X14, #2
0x000000000000efa4 (+0x0000ece4)  00c4d78a  AND.B64                         X2, X13, X15
0x000000000000efa8 (+0x0000ece8)  02c6020e  SHL.B64                         X3, #14
0x000000000000efac (+0x0000ecec)  0008c381  ADD.S64                         X4, X12, X7
# [DWARF] common/pa_trace.h:552
#     546 |     PA_GM TraceCoreState &core = *trace.core;
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
# >   552 |     PA_GM TraceRecord &record = trace.records[slot];
0x000000000000efb0 (+0x0000ecf0)  02020800  MOV_XD_XN.S64                   X1, X0
0x000000000000efb4 (+0x0000ecf4)  00082184  MADD.S64                        X4, X2, X3
0x000000000000efb8 (+0x0000ecf8)  02c20206  SHL.B64                         X1, #6
0x000000000000efbc (+0x0000ecfc)  00044181  ADD.S64                         X2, X4, X3
0x000000000000efc0 (+0x0000ed00)  0003b081  ADD.S64                         X1, X27, X1
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000efc4 (+0x0000ed04)  08842680  SUB_IMM.S64                     X2, X2, #1664
# [DWARF] common/pa_trace.h:553
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
# >   553 |     record.start_cycle = start_cycle;
0x000000000000efc8 (+0x0000ed08)  09f21a81  STP_XI_XJ_XN.B64                X25, X21, X1, #0
# [DWARF] common/pa_trace.h:555
#     554 |     record.end_cycle = end_cycle;
# >   555 |     record.task_id = task_id;
0x000000000000efcc (+0x0000ed0c)  09a21ba1  STP_XI_XJ_XN.B32                X17, X23, X1, #16
# [DWARF] common/pa_trace.h:558
#     556 |     record.function_id = function_id;
#     557 |     record.phase = static_cast<int32_t>(trace_phase);
# >   558 |     record.lane = trace.lane;
0x000000000000efd0 (+0x0000ed10)  08042000  ADD_IMM.S64                     X2, X2, #0
0x000000000000efd4 (+0x0000ed14)  1c862560  LD_XD_XN_IMM.B32                X3, X2, #1376
# [DWARF] common/pa_trace.h:557
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
#     555 |     record.task_id = task_id;
#     556 |     record.function_id = function_id;
# >   557 |     record.phase = static_cast<int32_t>(trace_phase);
0x000000000000efd8 (+0x0000ed18)  0708000b  MOV_XD_IMM                      X4, #11
# [DWARF] common/pa_trace.h:565
#     558 |     record.lane = trace.lane;
#     559 |     record.block_id = trace.block_id;
#     560 |     record.core_idx = trace.core_idx;
#     561 |     record.flags = flags;
#     562 |     record.auxiliary = auxiliary;
#     563 |     // count 最后更新，使其始终指向下一空槽；单写者条件下无需 reserve/commit 两阶段。
#     564 |     // 最终 FlushTraceCore 会把记录体先于该计数一并导出。
# >   565 |     core.count = slot + 1;
0x000000000000efdc (+0x0000ed1c)  08000001  ADD_IMM.S64                     X0, X0, #1
# [DWARF] common/pa_trace.h:557
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
#     555 |     record.task_id = task_id;
#     556 |     record.function_id = function_id;
# >   557 |     record.phase = static_cast<int32_t>(trace_phase);
0x000000000000efe0 (+0x0000ed20)  098811b1  STP_XI_XJ_XN.B32                X4, X3, X1, #24
# [DWARF] common/pa_trace.h:559
#     558 |     record.lane = trace.lane;
# >   559 |     record.block_id = trace.block_id;
0x000000000000efe4 (+0x0000ed24)  08062564  ADD_IMM.S64                     X3, X2, #1380
0x000000000000efe8 (+0x0000ed28)  0c863100  LDP_XI_XJ_XN.B32                X3, X2, X3, #0
0x000000000000efec (+0x0000ed2c)  08081020  ADD_IMM.S64                     X4, X1, #32
# [DWARF] common/pa_trace.h:561
#     560 |     record.core_idx = trace.core_idx;
# >   561 |     record.flags = flags;
0x000000000000eff0 (+0x0000ed30)  08021028  ADD_IMM.S64                     X1, X1, #40
# [DWARF] common/pa_trace.h:559
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
#     555 |     record.task_id = task_id;
#     556 |     record.function_id = function_id;
#     557 |     record.phase = static_cast<int32_t>(trace_phase);
#     558 |     record.lane = trace.lane;
# >   559 |     record.block_id = trace.block_id;
0x000000000000eff4 (+0x0000ed34)  09864101  STP_XI_XJ_XN.B32                X3, X2, X4, #0
0x000000000000eff8 (+0x0000ed38)  00c6590b  OR.B64                          X3, X5, X18
# [DWARF] common/pa_trace.h:561
#     560 |     record.core_idx = trace.core_idx;
# >   561 |     record.flags = flags;
0x000000000000effc (+0x0000ed3c)  09861581  STP_XI_XJ_XN.B32                X3, X11, X1, #0
# [DWARF] common/pa_trace.h:565
#     562 |     record.auxiliary = auxiliary;
#     563 |     // count 最后更新，使其始终指向下一空槽；单写者条件下无需 reserve/commit 两阶段。
#     564 |     // 最终 FlushTraceCore 会把记录体先于该计数一并导出。
# >   565 |     core.count = slot + 1;
0x000000000000f000 (+0x0000ed40)  0381a000  ST_XD_XN_IMM.B32                X0, X26, #0
0x000000000000f004 (+0x0000ed44)  40000004  JUMP                            #4
# [DWARF] common/pa_trace.h:549
#     543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
# >   549 |         core.dropped = core.dropped + 1;
0x000000000000f008 (+0x0000ed48)  1c81a004  LD_XD_XN_IMM.B32                X0, X26, #4
0x000000000000f00c (+0x0000ed4c)  08000001  ADD_IMM.S64                     X0, X0, #1
0x000000000000f010 (+0x0000ed50)  0381a004  ST_XD_XN_IMM.B32                X0, X26, #4
# [DWARF] common/pa_frontend.h:278
#     272 |     args.explicit_dep_count = 0;
#     273 | }
#     274 | 
#     275 | PA_DEVICE void ResetTaskArgs(TaskArgs &args) {
#     276 |     // reset 的输出是不含 tensor/scalar/显式依赖的新逻辑参数表，但保留已分配对象及
#     277 |     // launch_spec；QK/SF/PV/UP 在同一个 1 KiB TaskArgs 上依次复用这一状态。
# >   278 |     args.tensor_count = 0;
0x000000000000f014 (+0x0000ed54)  0ff3e700  STI_XN_IMM.B64                  X30, #3256
# [DWARF] common/pa_frontend.h:0
# [SOURCE unavailable]
0x000000000000f018 (+0x0000ed58)  07000000  MOV_XD_IMM                      X0, #0
# [DWARF] common/pa_frontend.h:243
#     237 | 
#     238 | PA_DEVICE void ClearDumpArgSelection(PaDumpArgSelection &selection) {
#     239 |     // Volatile stores intentionally preserve the profiling-enabled PA reset
#     240 |     // traffic even though the standalone winner workload never consumes dump data.
#     241 |     // volatile 的目的不是同步，而是阻止编译器删掉这段生产基线中存在的写流量。
#     242 |     volatile uint64_t *masks = &selection.dump_arg_mask;
# >   243 |     masks[0] = 0;
0x000000000000f01c (+0x0000ed5c)  0ff3eb00  STI_XN_IMM.B64                  X30, #3288
# [DWARF] common/pa_frontend.h:703
#     697 |     PA_DEVICE void RecordView() { ++counts_.views_created; }
#     698 |     PA_DEVICE void RecordDynamicCreateInfo() { ++counts_.dynamic_create_infos; }
#     699 | 
#     700 |     template <typename Thunk>
#     701 |     PA_DEVICE void AddLocalInput(Thunk thunk) {
#     702 |         if constexpr (Lazy) {
# >   703 |             if (!won_) return;
0x000000000000f020 (+0x0000ed60)  0001200e  CMP.S64.EQ                      X18, X0
# [DWARF] common/pa_frontend.h:244
#     238 | PA_DEVICE void ClearDumpArgSelection(PaDumpArgSelection &selection) {
#     239 |     // Volatile stores intentionally preserve the profiling-enabled PA reset
#     240 |     // traffic even though the standalone winner workload never consumes dump data.
#     241 |     // volatile 的目的不是同步，而是阻止编译器删掉这段生产基线中存在的写流量。
#     242 |     volatile uint64_t *masks = &selection.dump_arg_mask;
#     243 |     masks[0] = 0;
# >   244 |     masks[1] = 0;
0x000000000000f024 (+0x0000ed64)  0ff3ec00  STI_XN_IMM.B64                  X30, #3296
# [DWARF] common/pa_frontend.h:0
# [SOURCE unavailable]
0x000000000000f028 (+0x0000ed68)  07060000  MOV_XD_IMM                      X3, #0
# [DWARF] common/pa_frontend.h:247
#     241 |     // volatile 的目的不是同步，而是阻止编译器删掉这段生产基线中存在的写流量。
#     242 |     volatile uint64_t *masks = &selection.dump_arg_mask;
#     243 |     masks[0] = 0;
#     244 |     masks[1] = 0;
#     245 |     volatile uint64_t *sources = &selection.scalar_source_ptrs[0];
#     246 |     for (uint32_t index = 0; index < kMaxTaskScalars; ++index) {
# >   247 |         sources[index] = 0;
0x000000000000f02c (+0x0000ed6c)  0ff3ed00  STI_XN_IMM.B64                  X30, #3304
# [DWARF] common/pa_frontend.h:0
# [SOURCE unavailable]
0x000000000000f030 (+0x0000ed70)  07020000  MOV_XD_IMM                      X1, #0
# [DWARF] common/pa_frontend.h:247
#     241 |     // volatile 的目的不是同步，而是阻止编译器删掉这段生产基线中存在的写流量。
#     242 |     volatile uint64_t *masks = &selection.dump_arg_mask;
#     243 |     masks[0] = 0;
#     244 |     masks[1] = 0;
#     245 |     volatile uint64_t *sources = &selection.scalar_source_ptrs[0];
#     246 |     for (uint32_t index = 0; index < kMaxTaskScalars; ++index) {
# >   247 |         sources[index] = 0;
0x000000000000f034 (+0x0000ed74)  0ff3ee00  STI_XN_IMM.B64                  X30, #3312
0x000000000000f038 (+0x0000ed78)  0ff3ef00  STI_XN_IMM.B64                  X30, #3320
0x000000000000f03c (+0x0000ed7c)  0ff5e000  STI_XN_IMM.B64                  X30, #3328
0x000000000000f040 (+0x0000ed80)  0ff5e100  STI_XN_IMM.B64                  X30, #3336
0x000000000000f044 (+0x0000ed84)  0ff5e200  STI_XN_IMM.B64                  X30, #3344
0x000000000000f048 (+0x0000ed88)  0ff5e300  STI_XN_IMM.B64                  X30, #3352
0x000000000000f04c (+0x0000ed8c)  0ff5e400  STI_XN_IMM.B64                  X30, #3360
0x000000000000f050 (+0x0000ed90)  0ff5e500  STI_XN_IMM.B64                  X30, #3368
0x000000000000f054 (+0x0000ed94)  0ff5e600  STI_XN_IMM.B64                  X30, #3376
0x000000000000f058 (+0x0000ed98)  0ff5e700  STI_XN_IMM.B64                  X30, #3384
0x000000000000f05c (+0x0000ed9c)  0ff5e800  STI_XN_IMM.B64                  X30, #3392
0x000000000000f060 (+0x0000eda0)  0ff5e900  STI_XN_IMM.B64                  X30, #3400
0x000000000000f064 (+0x0000eda4)  0ff5ea00  STI_XN_IMM.B64                  X30, #3408
0x000000000000f068 (+0x0000eda8)  0ff5eb00  STI_XN_IMM.B64                  X30, #3416
0x000000000000f06c (+0x0000edac)  0ff5ec00  STI_XN_IMM.B64                  X30, #3424
# [DWARF] common/pa_frontend.h:251
#     248 |     }
#     249 |     volatile uint8_t *dtypes = &selection.scalar_dtypes[0];
#     250 |     for (uint32_t index = 0; index < kMaxTaskScalars; ++index) {
# >   251 |         dtypes[index] = 0;
0x000000000000f070 (+0x0000edb0)  0f35ed00  STI_XN_IMM.B8                   X30, #3432
0x000000000000f074 (+0x0000edb4)  0f35ed20  STI_XN_IMM.B8                   X30, #3433
0x000000000000f078 (+0x0000edb8)  0f35ed40  STI_XN_IMM.B8                   X30, #3434
0x000000000000f07c (+0x0000edbc)  0f35ed60  STI_XN_IMM.B8                   X30, #3435
0x000000000000f080 (+0x0000edc0)  0f35ed80  STI_XN_IMM.B8                   X30, #3436
0x000000000000f084 (+0x0000edc4)  0f35eda0  STI_XN_IMM.B8                   X30, #3437
0x000000000000f088 (+0x0000edc8)  0f35edc0  STI_XN_IMM.B8                   X30, #3438
0x000000000000f08c (+0x0000edcc)  0f35ede0  STI_XN_IMM.B8                   X30, #3439
0x000000000000f090 (+0x0000edd0)  0f35ee00  STI_XN_IMM.B8                   X30, #3440
0x000000000000f094 (+0x0000edd4)  0f35ee20  STI_XN_IMM.B8                   X30, #3441
0x000000000000f098 (+0x0000edd8)  0f35ee40  STI_XN_IMM.B8                   X30, #3442
0x000000000000f09c (+0x0000eddc)  0f35ee60  STI_XN_IMM.B8                   X30, #3443
0x000000000000f0a0 (+0x0000ede0)  0f35ee80  STI_XN_IMM.B8                   X30, #3444
0x000000000000f0a4 (+0x0000ede4)  0f35eea0  STI_XN_IMM.B8                   X30, #3445
0x000000000000f0a8 (+0x0000ede8)  0f35eec0  STI_XN_IMM.B8                   X30, #3446
0x000000000000f0ac (+0x0000edec)  0f35eee0  STI_XN_IMM.B8                   X30, #3447
# [DWARF] common/pa_frontend.h:0
# [SOURCE unavailable]
0x000000000000f0b0 (+0x0000edf0)  1ce1e8f8  LD_XD_XN_IMM.B64                X16, X30, #2296
# [DWARF] common/pa_frontend.h:281
#     275 | PA_DEVICE void ResetTaskArgs(TaskArgs &args) {
#     276 |     // reset 的输出是不含 tensor/scalar/显式依赖的新逻辑参数表，但保留已分配对象及
#     277 |     // launch_spec；QK/SF/PV/UP 在同一个 1 KiB TaskArgs 上依次复用这一状态。
#     278 |     args.tensor_count = 0;
#     279 |     args.scalar_count = 0;
#     280 |     ClearDumpArgSelection(args.dump_arg_selection);
# >   281 |     args.explicit_deps = 0;
0x000000000000f0b4 (+0x0000edf4)  0ff5ef00  STI_XN_IMM.B64                  X30, #3448
# [DWARF] common/pa_frontend.h:282
# >   282 |     args.explicit_dep_count = 0;
0x000000000000f0b8 (+0x0000edf8)  0fb7e000  STI_XN_IMM.B32                  X30, #3456
# [DWARF] common/pa_frontend.h:283
# >   283 |     args.has_error = false;
0x000000000000f0bc (+0x0000edfc)  0f33e800  STI_XN_IMM.B8                   X30, #3264
# [DWARF] common/pa_frontend.h:284
# >   284 |     args.error_msg = 0;
0x000000000000f0c0 (+0x0000ee00)  0ff3e900  STI_XN_IMM.B64                  X30, #3272
0x000000000000f0c4 (+0x0000ee04)  02e0020b  SHL.B64                         X16, #11
# [DWARF] common/pa_frontend.h:703
#     697 |     PA_DEVICE void RecordView() { ++counts_.views_created; }
#     698 |     PA_DEVICE void RecordDynamicCreateInfo() { ++counts_.dynamic_create_infos; }
#     699 | 
#     700 |     template <typename Thunk>
#     701 |     PA_DEVICE void AddLocalInput(Thunk thunk) {
#     702 |         if constexpr (Lazy) {
# >   703 |             if (!won_) return;
0x000000000000f0c8 (+0x0000ee08)  4020003b  JUMPC                           #59
# [DWARF] common/pa_frontend.h:468
#     462 |     destination.buffer_addr = source.buffer_addr;
#     463 |     destination.buffer_size = source.buffer_size;
#     464 |     destination.owner_task_id = source.owner_task_id;
#     465 |     destination.start_offset = source.start_offset;
#     466 |     destination.version = source.version;
#     467 |     destination.ndims = source.ndims;
# >   468 |     destination.dtype = source.dtype;
0x000000000000f0cc (+0x0000ee0c)  0883e220  SUB_IMM.S64                     X1, X30, #544
# [DWARF] common/pa_frontend.h:464
#     458 | template <typename Source>
#     459 | PA_DEVICE void CopyTensorLine1(TensorDesc &destination, const Source &source) {
#     460 |     // view 只需复制 descriptor 第一条 cache line 的身份/shape 字段，随后由调用方
#     461 |     // 覆盖 offset、shape、stride 与 extent；不做整 128-byte 拷贝以匹配 PA 写流。
#     462 |     destination.buffer_addr = source.buffer_addr;
#     463 |     destination.buffer_size = source.buffer_size;
# >   464 |     destination.owner_task_id = source.owner_task_id;
0x000000000000f0d0 (+0x0000ee10)  1cd3edc8  LD_XD_XN_IMM.B64                X9, X30, #3528
# [DWARF] common/pa_frontend.h:301
#     295 |     return true;
#     296 | }
#     297 | 
#     298 | PA_DEVICE void AppendLocalTensor(TaskArgs &args, const TensorDesc &tensor, TensorArgType tag) {
#     299 |     const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
#     300 |     args.tensors[index].pointer.local_tensor = &tensor;
# >   301 |     args.tensors[index].kind = TensorRefKind::LocalTensor;
0x000000000000f0d4 (+0x0000ee14)  0f29e800  STI_XN_IMM.B8                   X30, #2624
# [DWARF] common/pa_frontend.h:468
#     462 |     destination.buffer_addr = source.buffer_addr;
#     463 |     destination.buffer_size = source.buffer_size;
#     464 |     destination.owner_task_id = source.owner_task_id;
#     465 |     destination.start_offset = source.start_offset;
#     466 |     destination.version = source.version;
#     467 |     destination.ndims = source.ndims;
# >   468 |     destination.dtype = source.dtype;
0x000000000000f0d8 (+0x0000ee18)  0c021100  LDP_XI_XJ_XN.B8                 X1, X2, X1, #0
# [DWARF] common/pa_frontend.h:470
#     469 |     destination.manual_dep = source.manual_dep;
# >   470 |     destination.is_contiguous = source.is_contiguous;
0x000000000000f0dc (+0x0000ee1c)  0887e21e  SUB_IMM.S64                     X3, X30, #542
# [DWARF] common/pa_frontend.h:473
#     471 |     destination.child_memory = source.child_memory;
#     472 |     for (uint32_t index = 0; index < kMaxTensorDims; ++index) {
# >   473 |         destination.shapes[index] = source.shapes[index];
0x000000000000f0e0 (+0x0000ee20)  088be210  SUB_IMM.S64                     X5, X30, #528
# [DWARF] common/pa_frontend.h:470
#     464 |     destination.owner_task_id = source.owner_task_id;
#     465 |     destination.start_offset = source.start_offset;
#     466 |     destination.version = source.version;
#     467 |     destination.ndims = source.ndims;
#     468 |     destination.dtype = source.dtype;
#     469 |     destination.manual_dep = source.manual_dep;
# >   470 |     destination.is_contiguous = source.is_contiguous;
0x000000000000f0e4 (+0x0000ee24)  0c063200  LDP_XI_XJ_XN.B8                 X3, X4, X3, #0
# [DWARF] common/pa_frontend.h:462
#     456 | }
#     457 | 
#     458 | template <typename Source>
#     459 | PA_DEVICE void CopyTensorLine1(TensorDesc &destination, const Source &source) {
#     460 |     // view 只需复制 descriptor 第一条 cache line 的身份/shape 字段，随后由调用方
#     461 |     // 覆盖 offset、shape、stride 与 extent；不做整 128-byte 拷贝以匹配 PA 写流。
# >   462 |     destination.buffer_addr = source.buffer_addr;
0x000000000000f0e8 (+0x0000ee28)  088fe248  SUB_IMM.S64                     X7, X30, #584
# [DWARF] common/pa_frontend.h:473
#     463 |     destination.buffer_size = source.buffer_size;
#     464 |     destination.owner_task_id = source.owner_task_id;
#     465 |     destination.start_offset = source.start_offset;
#     466 |     destination.version = source.version;
#     467 |     destination.ndims = source.ndims;
#     468 |     destination.dtype = source.dtype;
#     469 |     destination.manual_dep = source.manual_dep;
#     470 |     destination.is_contiguous = source.is_contiguous;
#     471 |     destination.child_memory = source.child_memory;
#     472 |     for (uint32_t index = 0; index < kMaxTensorDims; ++index) {
# >   473 |         destination.shapes[index] = source.shapes[index];
0x000000000000f0ec (+0x0000ee2c)  0c8a5300  LDP_XI_XJ_XN.B32                X5, X6, X5, #0
# [DWARF] common/pa_frontend.h:462
#     456 | }
#     457 | 
#     458 | template <typename Source>
#     459 | PA_DEVICE void CopyTensorLine1(TensorDesc &destination, const Source &source) {
#     460 |     // view 只需复制 descriptor 第一条 cache line 的身份/shape 字段，随后由调用方
#     461 |     // 覆盖 offset、shape、stride 与 extent；不做整 128-byte 拷贝以匹配 PA 写流。
# >   462 |     destination.buffer_addr = source.buffer_addr;
0x000000000000f0f0 (+0x0000ee30)  0cce7400  LDP_XI_XJ_XN.B64                X7, X8, X7, #0
# [DWARF] common/pa_frontend.h:466
#     463 |     destination.buffer_size = source.buffer_size;
#     464 |     destination.owner_task_id = source.owner_task_id;
#     465 |     destination.start_offset = source.start_offset;
# >   466 |     destination.version = source.version;
0x000000000000f0f4 (+0x0000ee34)  1c95edd8  LD_XD_XN_IMM.B32                X10, X30, #3544
# [DWARF] common/pa_frontend.h:473
#     467 |     destination.ndims = source.ndims;
#     468 |     destination.dtype = source.dtype;
#     469 |     destination.manual_dep = source.manual_dep;
#     470 |     destination.is_contiguous = source.is_contiguous;
#     471 |     destination.child_memory = source.child_memory;
#     472 |     for (uint32_t index = 0; index < kMaxTensorDims; ++index) {
# >   473 |         destination.shapes[index] = source.shapes[index];
0x000000000000f0f8 (+0x0000ee38)  1c97edec  LD_XD_XN_IMM.B32                X11, X30, #3564
# [DWARF] common/pa_frontend.h:468
#     462 |     destination.buffer_addr = source.buffer_addr;
#     463 |     destination.buffer_size = source.buffer_size;
#     464 |     destination.owner_task_id = source.owner_task_id;
#     465 |     destination.start_offset = source.start_offset;
#     466 |     destination.version = source.version;
#     467 |     destination.ndims = source.ndims;
# >   468 |     destination.dtype = source.dtype;
0x000000000000f0fc (+0x0000ee3c)  0303e0e0  ST_XD_XN_IMM.B8                 X1, X30, #224
# [DWARF] common/pa_frontend.h:469
# >   469 |     destination.manual_dep = source.manual_dep;
0x000000000000f100 (+0x0000ee40)  0803e0e1  ADD_IMM.S64                     X1, X30, #225
0x000000000000f104 (+0x0000ee44)  09041181  STP_XI_XJ_XN.B8                 X2, X3, X1, #0
# [DWARF] common/pa_frontend.h:473
#     470 |     destination.is_contiguous = source.is_contiguous;
#     471 |     destination.child_memory = source.child_memory;
#     472 |     for (uint32_t index = 0; index < kMaxTensorDims; ++index) {
# >   473 |         destination.shapes[index] = source.shapes[index];
0x000000000000f108 (+0x0000ee48)  0803e0f0  ADD_IMM.S64                     X1, X30, #240
# [DWARF] common/pa_frontend.h:471
#     465 |     destination.start_offset = source.start_offset;
#     466 |     destination.version = source.version;
#     467 |     destination.ndims = source.ndims;
#     468 |     destination.dtype = source.dtype;
#     469 |     destination.manual_dep = source.manual_dep;
#     470 |     destination.is_contiguous = source.is_contiguous;
# >   471 |     destination.child_memory = source.child_memory;
0x000000000000f10c (+0x0000ee4c)  0309e0e3  ST_XD_XN_IMM.B8                 X4, X30, #227
# [DWARF] common/pa_frontend.h:466
#     460 |     // view 只需复制 descriptor 第一条 cache line 的身份/shape 字段，随后由调用方
#     461 |     // 覆盖 offset、shape、stride 与 extent；不做整 128-byte 拷贝以匹配 PA 写流。
#     462 |     destination.buffer_addr = source.buffer_addr;
#     463 |     destination.buffer_size = source.buffer_size;
#     464 |     destination.owner_task_id = source.owner_task_id;
#     465 |     destination.start_offset = source.start_offset;
# >   466 |     destination.version = source.version;
0x000000000000f110 (+0x0000ee50)  0805e0d8  ADD_IMM.S64                     X2, X30, #216
# [DWARF] common/pa_frontend.h:473
#     467 |     destination.ndims = source.ndims;
#     468 |     destination.dtype = source.dtype;
#     469 |     destination.manual_dep = source.manual_dep;
#     470 |     destination.is_contiguous = source.is_contiguous;
#     471 |     destination.child_memory = source.child_memory;
#     472 |     for (uint32_t index = 0; index < kMaxTensorDims; ++index) {
# >   473 |         destination.shapes[index] = source.shapes[index];
0x000000000000f114 (+0x0000ee54)  098a1301  STP_XI_XJ_XN.B32                X5, X6, X1, #0
# [DWARF] common/pa_frontend.h:462
#     456 | }
#     457 | 
#     458 | template <typename Source>
#     459 | PA_DEVICE void CopyTensorLine1(TensorDesc &destination, const Source &source) {
#     460 |     // view 只需复制 descriptor 第一条 cache line 的身份/shape 字段，随后由调用方
#     461 |     // 覆盖 offset、shape、stride 与 extent；不做整 128-byte 拷贝以匹配 PA 写流。
# >   462 |     destination.buffer_addr = source.buffer_addr;
0x000000000000f118 (+0x0000ee58)  0803e0b8  ADD_IMM.S64                     X1, X30, #184
# [DWARF] common/pa_frontend.h:520
#     514 | }
#     515 | 
#     516 | PA_DEVICE void MakeLazySampleCallbackQueryView(PaOrchestrationState &orch, uint32_t batch) {
#     517 |     CopyTensorLine1(orch.query_view, orch.query);
#     518 |     orch.query_view.start_offset = static_cast<uint64_t>(batch) * kPaHeads * kPaHeadDim;
#     519 |     orch.query_view.ndims = 2;
# >   520 |     orch.query_view.shapes[0] = kPaHeads;
0x000000000000f11c (+0x0000ee5c)  0807e0e4  ADD_IMM.S64                     X3, X30, #228
# [DWARF] common/pa_frontend.h:462
#     456 | }
#     457 | 
#     458 | template <typename Source>
#     459 | PA_DEVICE void CopyTensorLine1(TensorDesc &destination, const Source &source) {
#     460 |     // view 只需复制 descriptor 第一条 cache line 的身份/shape 字段，随后由调用方
#     461 |     // 覆盖 offset、shape、stride 与 extent；不做整 128-byte 拷贝以匹配 PA 写流。
# >   462 |     destination.buffer_addr = source.buffer_addr;
0x000000000000f120 (+0x0000ee60)  09ce1401  STP_XI_XJ_XN.B64                X7, X8, X1, #0
# [DWARF] common/pa_frontend.h:464
#     463 |     destination.buffer_size = source.buffer_size;
# >   464 |     destination.owner_task_id = source.owner_task_id;
0x000000000000f124 (+0x0000ee64)  0803e0c8  ADD_IMM.S64                     X1, X30, #200
0x000000000000f128 (+0x0000ee68)  1ccfe990  LD_XD_XN_IMM.B64                X7, X30, #2448
0x000000000000f12c (+0x0000ee6c)  09d21801  STP_XI_XJ_XN.B64                X9, X16, X1, #0
# [DWARF] common/pa_frontend.h:519
#     513 |     orch.output_view.extent_elem_cache = kPaHeads * kPaHeadDim;
#     514 | }
#     515 | 
#     516 | PA_DEVICE void MakeLazySampleCallbackQueryView(PaOrchestrationState &orch, uint32_t batch) {
#     517 |     CopyTensorLine1(orch.query_view, orch.query);
#     518 |     orch.query_view.start_offset = static_cast<uint64_t>(batch) * kPaHeads * kPaHeadDim;
# >   519 |     orch.query_view.ndims = 2;
0x000000000000f130 (+0x0000ee70)  07020002  MOV_XD_IMM                      X1, #2
# [DWARF] common/pa_frontend.h:466
#     460 |     // view 只需复制 descriptor 第一条 cache line 的身份/shape 字段，随后由调用方
#     461 |     // 覆盖 offset、shape、stride 与 extent；不做整 128-byte 拷贝以匹配 PA 写流。
#     462 |     destination.buffer_addr = source.buffer_addr;
#     463 |     destination.buffer_size = source.buffer_size;
#     464 |     destination.owner_task_id = source.owner_task_id;
#     465 |     destination.start_offset = source.start_offset;
# >   466 |     destination.version = source.version;
0x000000000000f134 (+0x0000ee74)  09942081  STP_XI_XJ_XN.B32                X10, X1, X2, #0
# [DWARF] common/pa_frontend.h:520
#     514 | }
#     515 | 
#     516 | PA_DEVICE void MakeLazySampleCallbackQueryView(PaOrchestrationState &orch, uint32_t batch) {
#     517 |     CopyTensorLine1(orch.query_view, orch.query);
#     518 |     orch.query_view.start_offset = static_cast<uint64_t>(batch) * kPaHeads * kPaHeadDim;
#     519 |     orch.query_view.ndims = 2;
# >   520 |     orch.query_view.shapes[0] = kPaHeads;
0x000000000000f138 (+0x0000ee78)  07040010  MOV_XD_IMM                      X2, #16
# [DWARF] common/pa_frontend.h:521
# >   521 |     orch.query_view.shapes[1] = kPaHeadDim;
0x000000000000f13c (+0x0000ee7c)  07020080  MOV_XD_IMM                      X1, #128
# [DWARF] common/pa_frontend.h:473
#     467 |     destination.ndims = source.ndims;
#     468 |     destination.dtype = source.dtype;
#     469 |     destination.manual_dep = source.manual_dep;
#     470 |     destination.is_contiguous = source.is_contiguous;
#     471 |     destination.child_memory = source.child_memory;
#     472 |     for (uint32_t index = 0; index < kMaxTensorDims; ++index) {
# >   473 |         destination.shapes[index] = source.shapes[index];
0x000000000000f140 (+0x0000ee80)  0397e0ec  ST_XD_XN_IMM.B32                X11, X30, #236
# [DWARF] common/pa_frontend.h:520
#     514 | }
#     515 | 
#     516 | PA_DEVICE void MakeLazySampleCallbackQueryView(PaOrchestrationState &orch, uint32_t batch) {
#     517 |     CopyTensorLine1(orch.query_view, orch.query);
#     518 |     orch.query_view.start_offset = static_cast<uint64_t>(batch) * kPaHeads * kPaHeadDim;
#     519 |     orch.query_view.ndims = 2;
# >   520 |     orch.query_view.shapes[0] = kPaHeads;
0x000000000000f144 (+0x0000ee84)  09843081  STP_XI_XJ_XN.B32                X2, X1, X3, #0
# [DWARF] common/pa_frontend.h:522
#     521 |     orch.query_view.shapes[1] = kPaHeadDim;
# >   522 |     orch.query_view.strides[0] = kPaHeadDim;
0x000000000000f148 (+0x0000ee88)  07020080  MOV_XD_IMM                      X1, #128
0x000000000000f14c (+0x0000ee8c)  07830001  MOVK                            X1, #1, #2
0x000000000000f150 (+0x0000ee90)  03c3e100  ST_XD_XN_IMM.B64                X1, X30, #256
# [DWARF] common/pa_frontend.h:524
#     523 |     orch.query_view.strides[1] = 1;
# >   524 |     orch.query_view.extent_elem_cache = kPaHeads * kPaHeadDim;
0x000000000000f154 (+0x0000ee94)  07020800  MOV_XD_IMM                      X1, #2048
0x000000000000f158 (+0x0000ee98)  03c3e0f8  ST_XD_XN_IMM.B64                X1, X30, #248
# [DWARF] common/pa_frontend.h:300
#     294 |     }
#     295 |     return true;
#     296 | }
#     297 | 
#     298 | PA_DEVICE void AppendLocalTensor(TaskArgs &args, const TensorDesc &tensor, TensorArgType tag) {
#     299 |     const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
# >   300 |     args.tensors[index].pointer.local_tensor = &tensor;
0x000000000000f15c (+0x0000ee9c)  070200b8  MOV_XD_IMM                      X1, #184
0x000000000000f160 (+0x0000eea0)  0003e081  ADD.S64                         X1, X30, X1
0x000000000000f164 (+0x0000eea4)  03c3ea38  ST_XD_XN_IMM.B64                X1, X30, #2616
# [DWARF] common/pa_frontend.h:302
#     301 |     args.tensors[index].kind = TensorRefKind::LocalTensor;
# >   302 |     args.tags[index] = TagValue(tag);
0x000000000000f168 (+0x0000eea8)  07020648  MOV_XD_IMM                      X1, #1608
0x000000000000f16c (+0x0000eeac)  0003e082  SUB.S64                         X1, X30, X1
0x000000000000f170 (+0x0000eeb0)  0f801000  STI_XN_IMM.B32                  X1, #0
# [DWARF] common/pa_frontend.h:300
#     294 |     }
#     295 |     return true;
#     296 | }
#     297 | 
#     298 | PA_DEVICE void AppendLocalTensor(TaskArgs &args, const TensorDesc &tensor, TensorArgType tag) {
#     299 |     const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
# >   300 |     args.tensors[index].pointer.local_tensor = &tensor;
0x000000000000f174 (+0x0000eeb4)  070201c8  MOV_XD_IMM                      X1, #456
0x000000000000f178 (+0x0000eeb8)  0003e082  SUB.S64                         X1, X30, X1
# [DWARF] common/pa_frontend.h:301
# >   301 |     args.tensors[index].kind = TensorRefKind::LocalTensor;
0x000000000000f17c (+0x0000eebc)  0f29ea00  STI_XN_IMM.B8                   X30, #2640
# [DWARF] common/pa_frontend.h:300
#     294 |     }
#     295 |     return true;
#     296 | }
#     297 | 
#     298 | PA_DEVICE void AppendLocalTensor(TaskArgs &args, const TensorDesc &tensor, TensorArgType tag) {
#     299 |     const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
# >   300 |     args.tensors[index].pointer.local_tensor = &tensor;
0x000000000000f180 (+0x0000eec0)  03c3ea48  ST_XD_XN_IMM.B64                X1, X30, #2632
# [DWARF] common/pa_frontend.h:302
#     301 |     args.tensors[index].kind = TensorRefKind::LocalTensor;
# >   302 |     args.tags[index] = TagValue(tag);
0x000000000000f184 (+0x0000eec4)  1cc3e8e8  LD_XD_XN_IMM.B64                X1, X30, #2280
0x000000000000f188 (+0x0000eec8)  07060003  MOV_XD_IMM                      X3, #3
0x000000000000f18c (+0x0000eecc)  0f801000  STI_XN_IMM.B32                  X1, #0
# [DWARF] common/pa_frontend.h:300
#     294 |     }
#     295 |     return true;
#     296 | }
#     297 | 
#     298 | PA_DEVICE void AppendLocalTensor(TaskArgs &args, const TensorDesc &tensor, TensorArgType tag) {
#     299 |     const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
# >   300 |     args.tensors[index].pointer.local_tensor = &tensor;
0x000000000000f190 (+0x0000eed0)  070200c8  MOV_XD_IMM                      X1, #200
0x000000000000f194 (+0x0000eed4)  0003e082  SUB.S64                         X1, X30, X1
# [DWARF] common/pa_frontend.h:299
#     293 |         return false;
#     294 |     }
#     295 |     return true;
#     296 | }
#     297 | 
#     298 | PA_DEVICE void AppendLocalTensor(TaskArgs &args, const TensorDesc &tensor, TensorArgType tag) {
# >   299 |     const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
0x000000000000f198 (+0x0000eed8)  0387ecb8  ST_XD_XN_IMM.B32                X3, X30, #3256
# [DWARF] common/pa_frontend.h:300
# >   300 |     args.tensors[index].pointer.local_tensor = &tensor;
0x000000000000f19c (+0x0000eedc)  03c3ea58  ST_XD_XN_IMM.B64                X1, X30, #2648
# [DWARF] common/pa_frontend.h:302
#     301 |     args.tensors[index].kind = TensorRefKind::LocalTensor;
# >   302 |     args.tags[index] = TagValue(tag);
0x000000000000f1a0 (+0x0000eee0)  07020640  MOV_XD_IMM                      X1, #1600
0x000000000000f1a4 (+0x0000eee4)  0003e082  SUB.S64                         X1, X30, X1
# [DWARF] common/pa_frontend.h:301
#     295 |     return true;
#     296 | }
#     297 | 
#     298 | PA_DEVICE void AppendLocalTensor(TaskArgs &args, const TensorDesc &tensor, TensorArgType tag) {
#     299 |     const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
#     300 |     args.tensors[index].pointer.local_tensor = &tensor;
# >   301 |     args.tensors[index].kind = TensorRefKind::LocalTensor;
0x000000000000f1a8 (+0x0000eee8)  0f29ec00  STI_XN_IMM.B8                   X30, #2656
# [DWARF] common/pa_frontend.h:302
# >   302 |     args.tags[index] = TagValue(tag);
0x000000000000f1ac (+0x0000eeec)  0f801000  STI_XN_IMM.B32                  X1, #0
0x000000000000f1b0 (+0x0000eef0)  07020001  MOV_XD_IMM                      X1, #1
# [DWARF] common/pa_scheduler_core.h:847
#     841 |             builder.AddLocalInput([&]() PA_LAZY_LAMBDA_DEVICE -> const TensorDesc & {
#     842 |                 return orch.block_table;
#     843 |             });
#     844 |             builder.AddOutput([&]() PA_LAZY_LAMBDA_DEVICE -> const TensorCreateInfo & {
#     845 |                 const uint32_t score_shape[kMaxTensorDims] = {
#     846 |                     kPaHeads,
# >   847 |                     static_cast<uint32_t>(orch.current_nblocks * kPaBlockSize),
0x000000000000f1b4 (+0x0000eef4)  1cc9e2e0  LD_XD_XN_IMM.B64                X4, X30, #736
# [DWARF] common/pa_frontend.h:366
#     360 | 
#     361 | PA_DEVICE void InitCreateInfo(
#     362 |     TensorCreateInfo &info, const uint32_t shapes[kMaxTensorDims], uint32_t ndims, DataType dtype
#     363 | ) {
#     364 |     info.initial_value = 0;
#     365 |     info.has_initial_value = false;
# >   366 |     info.reserved0 = 0;
0x000000000000f1b8 (+0x0000eef8)  070a0000  MOV_XD_IMM                      X5, #0
# [DWARF] common/pa_frontend.h:364
#     358 |     AppendScalar(args, value2);
#     359 | }
#     360 | 
#     361 | PA_DEVICE void InitCreateInfo(
#     362 |     TensorCreateInfo &info, const uint32_t shapes[kMaxTensorDims], uint32_t ndims, DataType dtype
#     363 | ) {
# >   364 |     info.initial_value = 0;
0x000000000000f1bc (+0x0000eefc)  0fc9e700  STI_XN_IMM.B64                  X30, #568
0x000000000000f1c0 (+0x0000ef00)  1cf3e980  LD_XD_XN_IMM.B64                X25, X30, #2432
# [DWARF] common/pa_frontend.h:366
#     365 |     info.has_initial_value = false;
# >   366 |     info.reserved0 = 0;
0x000000000000f1c4 (+0x0000ef04)  080de248  ADD_IMM.S64                     X6, X30, #584
# [DWARF] common/pa_frontend.h:365
#     359 | }
#     360 | 
#     361 | PA_DEVICE void InitCreateInfo(
#     362 |     TensorCreateInfo &info, const uint32_t shapes[kMaxTensorDims], uint32_t ndims, DataType dtype
#     363 | ) {
#     364 |     info.initial_value = 0;
# >   365 |     info.has_initial_value = false;
0x000000000000f1c8 (+0x0000ef08)  0f09e800  STI_XN_IMM.B8                   X30, #576
# [DWARF] common/pa_frontend.h:366
# >   366 |     info.reserved0 = 0;
0x000000000000f1cc (+0x0000ef0c)  09ca6281  STP_XI_XJ_XN.B64                X5, X5, X6, #0
# [DWARF] common/pa_frontend.h:368
#     367 |     info.start_offset = 0;
# >   368 |     info.version = 0;
0x000000000000f1d0 (+0x0000ef10)  070c0000  MOV_XD_IMM                      X6, #0
0x000000000000f1d4 (+0x0000ef14)  078d0002  MOVK                            X6, #2, #2
0x000000000000f1d8 (+0x0000ef18)  03e1e8d8  ST_XD_XN_IMM.B64                X16, X30, #2264
0x000000000000f1dc (+0x0000ef1c)  03cde258  ST_XD_XN_IMM.B64                X6, X30, #600
# [DWARF] common/pa_frontend.h:370
#     369 |     info.ndims = ndims;
# >   370 |     info.dtype = dtype;
0x000000000000f1e0 (+0x0000ef20)  070c0000  MOV_XD_IMM                      X6, #0
0x000000000000f1e4 (+0x0000ef24)  074d0001  MOVK                            X6, #1, #1
0x000000000000f1e8 (+0x0000ef28)  078d0010  MOVK                            X6, #16, #2
# [DWARF] common/pa_frontend.h:314
#     308 |     args.tensors[index].kind = TensorRefKind::GmTensor;
#     309 |     args.tags[index] = TagValue(tag);
#     310 | }
#     311 | 
#     312 | PA_DEVICE void AppendOutput(TaskArgs &args, const TensorCreateInfo &create_info) {
#     313 |     const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
# >   314 |     args.tensors[index].pointer.create_info = &create_info;
0x000000000000f1ec (+0x0000ef2c)  020a3800  MOV_XD_XN.S64                   X5, X3
# [DWARF] common/pa_frontend.h:370
#     364 |     info.initial_value = 0;
#     365 |     info.has_initial_value = false;
#     366 |     info.reserved0 = 0;
#     367 |     info.start_offset = 0;
#     368 |     info.version = 0;
#     369 |     info.ndims = ndims;
# >   370 |     info.dtype = dtype;
0x000000000000f1f0 (+0x0000ef30)  03cde260  ST_XD_XN_IMM.B64                X6, X30, #608
# [DWARF] common/pa_frontend.h:314
#     308 |     args.tensors[index].kind = TensorRefKind::GmTensor;
#     309 |     args.tags[index] = TagValue(tag);
#     310 | }
#     311 | 
#     312 | PA_DEVICE void AppendOutput(TaskArgs &args, const TensorCreateInfo &create_info) {
#     313 |     const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
# >   314 |     args.tensors[index].pointer.create_info = &create_info;
0x000000000000f1f4 (+0x0000ef34)  070c05c8  MOV_XD_IMM                      X6, #1480
0x000000000000f1f8 (+0x0000ef38)  02ca0204  SHL.B64                         X5, #4
0x000000000000f1fc (+0x0000ef3c)  000de302  SUB.S64                         X6, X30, X6
0x000000000000f200 (+0x0000ef40)  000a6281  ADD.S64                         X5, X6, X5
0x000000000000f204 (+0x0000ef44)  070c0238  MOV_XD_IMM                      X6, #568
0x000000000000f208 (+0x0000ef48)  000de301  ADD.S64                         X6, X30, X6
0x000000000000f20c (+0x0000ef4c)  072a0004  MOV_XD_IMM                      X21, #4
# [DWARF] common/pa_frontend.h:0
# [SOURCE unavailable]
0x000000000000f210 (+0x0000ef50)  072c12c4  MOV_XD_IMM                      X22, #4804
0x000000000000f214 (+0x0000ef54)  07300020  MOV_XD_IMM                      X24, #32
# [DWARF] common/pa_scheduler_core.h:847
#     841 |             builder.AddLocalInput([&]() PA_LAZY_LAMBDA_DEVICE -> const TensorDesc & {
#     842 |                 return orch.block_table;
#     843 |             });
#     844 |             builder.AddOutput([&]() PA_LAZY_LAMBDA_DEVICE -> const TensorCreateInfo & {
#     845 |                 const uint32_t score_shape[kMaxTensorDims] = {
#     846 |                     kPaHeads,
# >   847 |                     static_cast<uint32_t>(orch.current_nblocks * kPaBlockSize),
0x000000000000f218 (+0x0000ef58)  02044800  MOV_XD_XN.S64                   X2, X4
0x000000000000f21c (+0x0000ef5c)  02c40207  SHL.B64                         X2, #7
# [DWARF] common/pa_frontend.h:377
#     371 |     info.manual_dep = false;
#     372 |     info.is_contiguous = true;
#     373 |     info.child_memory = 0;
#     374 |     // TensorCreateInfo's real constructor only writes active dimensions.
#     375 |     // 只写 ndims 个 shape，保留生产构造器的写入范围，不能为方便把五维全清零。
#     376 |     for (uint32_t index = 0; index < ndims; ++index) {
# >   377 |         info.shapes[index] = shapes[index];
0x000000000000f220 (+0x0000ef60)  0385e268  ST_XD_XN_IMM.B32                X2, X30, #616
# [DWARF] common/pa_frontend.h:313
#     307 |     args.tensors[index].pointer.gm_tensor = &tensor;
#     308 |     args.tensors[index].kind = TensorRefKind::GmTensor;
#     309 |     args.tags[index] = TagValue(tag);
#     310 | }
#     311 | 
#     312 | PA_DEVICE void AppendOutput(TaskArgs &args, const TensorCreateInfo &create_info) {
# >   313 |     const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
0x000000000000f224 (+0x0000ef64)  08043001  ADD_IMM.S64                     X2, X3, #1
0x000000000000f228 (+0x0000ef68)  0385ecb8  ST_XD_XN_IMM.B32                X2, X30, #3256
# [DWARF] common/pa_frontend.h:314
# >   314 |     args.tensors[index].pointer.create_info = &create_info;
0x000000000000f22c (+0x0000ef6c)  03cc5000  ST_XD_XN_IMM.B64                X6, X5, #0
# [DWARF] common/pa_frontend.h:315
# >   315 |     args.tensors[index].kind = TensorRefKind::CreateInfo;
0x000000000000f230 (+0x0000ef70)  070c0002  MOV_XD_IMM                      X6, #2
0x000000000000f234 (+0x0000ef74)  030c5008  ST_XD_XN_IMM.B8                 X6, X5, #8
# [DWARF] common/pa_frontend.h:316
# >   316 |     args.tags[index] = TagValue(TensorArgType::Output);
0x000000000000f238 (+0x0000ef78)  070a0648  MOV_XD_IMM                      X5, #1608
0x000000000000f23c (+0x0000ef7c)  000be282  SUB.S64                         X5, X30, X5
0x000000000000f240 (+0x0000ef80)  0e805181  STI_XN_XM.B32                   X5, X3
# [DWARF] common/pa_frontend.h:749
#     743 |         if (!args_.has_error) ++counts_.tensor_args_added;
#     744 |     }
#     745 | 
#     746 |     template <typename Thunk>
#     747 |     PA_DEVICE void AddScalar(Thunk thunk) {
#     748 |         if constexpr (Lazy) {
# >   749 |             if (!won_) return;
0x000000000000f244 (+0x0000ef84)  40200009  JUMPC                           #9
# [DWARF] common/pa_scheduler_core.h:858
#     852 |                 return orch.qk_create_info;
#     853 |             });
#     854 |             builder.AddScalar([&]() PA_LAZY_LAMBDA_DEVICE -> uint64_t {
#     855 |                 return orch.current_nblocks;
#     856 |             });
#     857 |             builder.AddScalar([&]() PA_LAZY_LAMBDA_DEVICE -> uint64_t {
# >   858 |                 return static_cast<uint64_t>(orch.current_batch) * kPaMaxBlocksPerRequest +
0x000000000000f248 (+0x0000ef88)  1c87e2f0  LD_XD_XN_IMM.B32                X3, X30, #752
0x000000000000f24c (+0x0000ef8c)  07000002  MOV_XD_IMM                      X0, #2
# [DWARF] common/pa_scheduler_core.h:859
# >   859 |                        orch.current_block_offset;
0x000000000000f250 (+0x0000ef90)  1ccbe2d8  LD_XD_XN_IMM.B64                X5, X30, #728
# [DWARF] common/pa_frontend.h:329
#     323 |         return false;
#     324 |     }
#     325 |     return true;
#     326 | }
#     327 | 
#     328 | PA_DEVICE void AppendScalar(TaskArgs &args, uint64_t value) {
# >   329 |     args.scalars[static_cast<uint32_t>(args.scalar_count++)] = value;
0x000000000000f254 (+0x0000ef94)  0381ecbc  ST_XD_XN_IMM.B32                X0, X30, #3260
# [DWARF] common/pa_scheduler_core.h:858
#     852 |                 return orch.qk_create_info;
#     853 |             });
#     854 |             builder.AddScalar([&]() PA_LAZY_LAMBDA_DEVICE -> uint64_t {
#     855 |                 return orch.current_nblocks;
#     856 |             });
#     857 |             builder.AddScalar([&]() PA_LAZY_LAMBDA_DEVICE -> uint64_t {
# >   858 |                 return static_cast<uint64_t>(orch.current_batch) * kPaMaxBlocksPerRequest +
0x000000000000f258 (+0x0000ef98)  02c60208  SHL.B64                         X3, #8
0x000000000000f25c (+0x0000ef9c)  00063281  ADD.S64                         X3, X3, X5
# [DWARF] common/pa_frontend.h:329
#     323 |         return false;
#     324 |     }
#     325 |     return true;
#     326 | }
#     327 | 
#     328 | PA_DEVICE void AppendScalar(TaskArgs &args, uint64_t value) {
# >   329 |     args.scalars[static_cast<uint32_t>(args.scalar_count++)] = value;
0x000000000000f260 (+0x0000efa0)  088be3c8  SUB_IMM.S64                     X5, X30, #968
0x000000000000f264 (+0x0000efa4)  09c85181  STP_XI_XJ_XN.B64                X4, X3, X5, #0
# [DWARF] common/pa_frontend.h:0
# [SOURCE unavailable]
0x000000000000f268 (+0x0000efa8)  0808e002  ADD_IMM.S64                     X4, X14, #2
# [DWARF] common/pa_scheduler_core.h:1161
#    1155 |     );
#    1156 | 
#    1157 |     if (!BuildLazySampleCallbackArgs<Kind, Lazy>(orch, args, batch, claim.won, stats)) {
#    1158 |         SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
#    1159 |         return false;
#    1160 |     }
# >  1161 |     const LazySampleCallbackTicket ticket{
0x000000000000f26c (+0x0000efac)  03a3e340  ST_XD_XN_IMM.B32                X17, X30, #832
0x000000000000f270 (+0x0000efb0)  00c6d78a  AND.B64                         X3, X13, X15
0x000000000000f274 (+0x0000efb4)  036fe344  ST_XD_XN_IMM.B16                X23, X30, #836
0x000000000000f278 (+0x0000efb8)  02c8020e  SHL.B64                         X4, #14
0x000000000000f27c (+0x0000efbc)  0325e346  ST_XD_XN_IMM.B8                 X18, X30, #838
0x000000000000f280 (+0x0000efc0)  000ac381  ADD.S64                         X5, X12, X7
0x000000000000f284 (+0x0000efc4)  0f0de8e0  STI_XN_IMM.B8                   X30, #839
# [DWARF] common/pa_scheduler_core.h:0
# [SOURCE unavailable]
0x000000000000f288 (+0x0000efc8)  000a3204  MADD.S64                        X5, X3, X4
0x000000000000f28c (+0x0000efcc)  00065201  ADD.S64                         X3, X5, X4
0x000000000000f290 (+0x0000efd0)  08863680  SUB_IMM.S64                     X3, X3, #1664
# [DWARF] common/pa_scheduler_core.h:945
#     939 |         }
#     940 |     };
#     941 | 
#     942 |     callback(callback_builder);
#     943 |     if (!callback_builder.Valid()) return false;
#     944 |     const LazySampleCallbackBuildCounts &counts = callback_builder.Counts();
# >   945 |     stats.result.arg_resets += counts.reset_calls;
0x000000000000f294 (+0x0000efd4)  08263000  ADD_IMM.S64                     X19, X3, #0
# [DWARF] common/pa_scheduler_core.h:946
# >   946 |     stats.result.views_created += counts.views_created;
0x000000000000f298 (+0x0000efd8)  080733c8  ADD_IMM.S64                     X3, X19, #968
# [DWARF] common/pa_scheduler_core.h:945
#     939 |         }
#     940 |     };
#     941 | 
#     942 |     callback(callback_builder);
#     943 |     if (!callback_builder.Valid()) return false;
#     944 |     const LazySampleCallbackBuildCounts &counts = callback_builder.Counts();
# >   945 |     stats.result.arg_resets += counts.reset_calls;
0x000000000000f29c (+0x0000efdc)  1ccf33d8  LD_XD_XN_IMM.B64                X7, X19, #984
# [DWARF] common/pa_scheduler_core.h:946
# >   946 |     stats.result.views_created += counts.views_created;
0x000000000000f2a0 (+0x0000efe0)  0cc63300  LDP_XI_XJ_XN.B64                X3, X6, X3, #0
# [DWARF] common/pa_scheduler_core.h:948
#     947 |     stats.result.dynamic_create_infos += counts.dynamic_create_infos;
# >   948 |     stats.result.tensor_args_added += counts.tensor_args_added;
0x000000000000f2a4 (+0x0000efe4)  080933e0  ADD_IMM.S64                     X4, X19, #992
0x000000000000f2a8 (+0x0000efe8)  0cc84280  LDP_XI_XJ_XN.B64                X4, X5, X4, #0
# [DWARF] common/pa_scheduler_core.h:946
#     940 |     };
#     941 | 
#     942 |     callback(callback_builder);
#     943 |     if (!callback_builder.Valid()) return false;
#     944 |     const LazySampleCallbackBuildCounts &counts = callback_builder.Counts();
#     945 |     stats.result.arg_resets += counts.reset_calls;
# >   946 |     stats.result.views_created += counts.views_created;
0x000000000000f2ac (+0x0000efec)  00023081  ADD.S64                         X1, X3, X1
# [DWARF] common/pa_scheduler_core.h:947
# >   947 |     stats.result.dynamic_create_infos += counts.dynamic_create_infos;
0x000000000000f2b0 (+0x0000eff0)  08066001  ADD_IMM.S64                     X3, X6, #1
# [DWARF] common/pa_scheduler_core.h:946
#     940 |     };
#     941 | 
#     942 |     callback(callback_builder);
#     943 |     if (!callback_builder.Valid()) return false;
#     944 |     const LazySampleCallbackBuildCounts &counts = callback_builder.Counts();
#     945 |     stats.result.arg_resets += counts.reset_calls;
# >   946 |     stats.result.views_created += counts.views_created;
0x000000000000f2b4 (+0x0000eff4)  080d33c8  ADD_IMM.S64                     X6, X19, #968
0x000000000000f2b8 (+0x0000eff8)  09c26181  STP_XI_XJ_XN.B64                X1, X3, X6, #0
# [DWARF] common/pa_scheduler_core.h:945
#     939 |         }
#     940 |     };
#     941 | 
#     942 |     callback(callback_builder);
#     943 |     if (!callback_builder.Valid()) return false;
#     944 |     const LazySampleCallbackBuildCounts &counts = callback_builder.Counts();
# >   945 |     stats.result.arg_resets += counts.reset_calls;
0x000000000000f2bc (+0x0000effc)  08027001  ADD_IMM.S64                     X1, X7, #1
# [DWARF] common/pa_scheduler_core.h:948
#     946 |     stats.result.views_created += counts.views_created;
#     947 |     stats.result.dynamic_create_infos += counts.dynamic_create_infos;
# >   948 |     stats.result.tensor_args_added += counts.tensor_args_added;
0x000000000000f2c0 (+0x0000f000)  00044101  ADD.S64                         X2, X4, X2
# [DWARF] common/pa_scheduler_core.h:945
#     939 |         }
#     940 |     };
#     941 | 
#     942 |     callback(callback_builder);
#     943 |     if (!callback_builder.Valid()) return false;
#     944 |     const LazySampleCallbackBuildCounts &counts = callback_builder.Counts();
# >   945 |     stats.result.arg_resets += counts.reset_calls;
0x000000000000f2c4 (+0x0000f004)  080733d8  ADD_IMM.S64                     X3, X19, #984
# [DWARF] common/pa_scheduler_core.h:949
#     946 |     stats.result.views_created += counts.views_created;
#     947 |     stats.result.dynamic_create_infos += counts.dynamic_create_infos;
#     948 |     stats.result.tensor_args_added += counts.tensor_args_added;
# >   949 |     stats.result.scalar_args_added += counts.scalar_args_added;
0x000000000000f2c8 (+0x0000f008)  00005001  ADD.S64                         X0, X5, X0
# [DWARF] common/pa_scheduler_core.h:945
#     939 |         }
#     940 |     };
#     941 | 
#     942 |     callback(callback_builder);
#     943 |     if (!callback_builder.Valid()) return false;
#     944 |     const LazySampleCallbackBuildCounts &counts = callback_builder.Counts();
# >   945 |     stats.result.arg_resets += counts.reset_calls;
0x000000000000f2cc (+0x0000f00c)  09c23101  STP_XI_XJ_XN.B64                X1, X2, X3, #0
# [DWARF] common/pa_scheduler_core.h:949
#     946 |     stats.result.views_created += counts.views_created;
#     947 |     stats.result.dynamic_create_infos += counts.dynamic_create_infos;
#     948 |     stats.result.tensor_args_added += counts.tensor_args_added;
# >   949 |     stats.result.scalar_args_added += counts.scalar_args_added;
0x000000000000f2d0 (+0x0000f010)  03c133e8  ST_XD_XN_IMM.B64                X0, X19, #1000
# [DWARF] common/pa_scheduler_core.h:1161
#    1155 |     );
#    1156 | 
#    1157 |     if (!BuildLazySampleCallbackArgs<Kind, Lazy>(orch, args, batch, claim.won, stats)) {
#    1158 |         SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
#    1159 |         return false;
#    1160 |     }
# >  1161 |     const LazySampleCallbackTicket ticket{
0x000000000000f2d4 (+0x0000f014)  1cc1e928  LD_XD_XN_IMM.B64                X0, X30, #2344
# [DWARF] ccec/ccec_ops.h:276
#     270 |     }
#     271 | 
#     272 |     __aicore__ static inline bool FinishLazySampleCallback(
#     273 |         const pa_scheduler::LazySampleCallbackTicket *ticket, const pa_scheduler::TaskArgs *args
#     274 |     ) {
#     275 | #if defined(PA_BUILD_AIC)
# >   276 |         return ::pa_scheduler_lazy_sample_callback_finish_aic(ticket, args) != 0;
0x000000000000f2d8 (+0x0000f018)  07020648  MOV_XD_IMM                      X1, #1608
0x000000000000f2dc (+0x0000f01c)  0003e082  SUB.S64                         X1, X30, X1
# [DWARF] common/pa_scheduler_core.h:1161
#    1155 |     );
#    1156 | 
#    1157 |     if (!BuildLazySampleCallbackArgs<Kind, Lazy>(orch, args, batch, claim.won, stats)) {
#    1158 |         SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
#    1159 |         return false;
#    1160 |     }
# >  1161 |     const LazySampleCallbackTicket ticket{
0x000000000000f2e0 (+0x0000f020)  03c1e338  ST_XD_XN_IMM.B64                X0, X30, #824
# [DWARF] ccec/ccec_ops.h:276
#     270 |     }
#     271 | 
#     272 |     __aicore__ static inline bool FinishLazySampleCallback(
#     273 |         const pa_scheduler::LazySampleCallbackTicket *ticket, const pa_scheduler::TaskArgs *args
#     274 |     ) {
#     275 | #if defined(PA_BUILD_AIC)
# >   276 |         return ::pa_scheduler_lazy_sample_callback_finish_aic(ticket, args) != 0;
0x000000000000f2e4 (+0x0000f024)  07000338  MOV_XD_IMM                      X0, #824
0x000000000000f2e8 (+0x0000f028)  0001e001  ADD.S64                         X0, X30, X0
0x000000000000f2ec (+0x0000f02c)  0704622a  MOV_XD_IMM                      X2, #25130
0x000000000000f2f0 (+0x0000f030)  07450000  MOVK                            X2, #0, #1
0x000000000000f2f4 (+0x0000f034)  07850000  MOVK                            X2, #0, #2
0x000000000000f2f8 (+0x0000f038)  40422000  CALL                            X2, #0
0x000000000000f2fc (+0x0000f03c)  02800a00  ZEROEXT.U32                     X0, X0
0x000000000000f300 (+0x0000f040)  071c0000  MOV_XD_IMM                      X14, #0
0x000000000000f304 (+0x0000f044)  0000070e  CMP.S64.EQ                      X0, X14
0x000000000000f308 (+0x0000f048)  07220001  MOV_XD_IMM                      X17, #1
0x000000000000f30c (+0x0000f04c)  07100006  MOV_XD_IMM                      X8, #6
# [DWARF] common/pa_scheduler_core.h:1393
#    1387 |                 )) {
#    1388 |                 break;
#    1389 |             }
#    1390 |             AcceptTaskOutputs(orchestration, TaskKind::Alloc, context.result);
#    1391 | 
#    1392 |             PreparePaBlockGroup(orchestration, 0);
# >  1393 |             if (!SubmitLazySampleCallback<
0x000000000000f310 (+0x0000f050)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x000000000000f314 (+0x0000f054)  07004302  MOV_XD_IMM                      X0, #17154
0x000000000000f318 (+0x0000f058)  07410000  MOVK                            X0, #0, #1
0x000000000000f31c (+0x0000f05c)  07810000  MOVK                            X0, #0, #2
0x000000000000f320 (+0x0000f060)  40220000  JUMPC                           X0, #0
0x000000000000f324 (+0x0000f064)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
# [DWARF] common/pa_frontend.h:835
#     829 |         case TaskKind::Alloc:
#     830 |             orch.accumulated_output = outputs.tensors[0];
#     831 |             orch.accumulated_sum = outputs.tensors[1];
#     832 |             orch.accumulated_max = outputs.tensors[2];
#     833 |             break;
#     834 |         case TaskKind::Qk:
# >   835 |             orch.qk_scores = outputs.tensors[0];
0x000000000000f328 (+0x0000f068)  1cc13050  LD_XD_XN_IMM.B64                X0, X19, #80
# [DWARF] common/pa_scheduler_core.h:790
#     784 | 
#     785 | PA_DEVICE void BeginLazySampleCallbackSubmit(PA_GM WorkerState &worker, SubmitContext &context) {
#     786 |     // This is BeginSubmit without an already-materialized TaskArgs.  The same
#     787 |     // fields are completed synchronously after the single callback builds args.
#     788 |     const uint32_t task_id = static_cast<uint32_t>(worker.local_index++);
#     789 |     context.self = &worker;
# >   790 |     context.payload = &worker.payloads[task_id & kPayloadMask];
0x000000000000f32c (+0x0000f06c)  070207ff  MOV_XD_IMM                      X1, #2047
0x000000000000f330 (+0x0000f070)  1cc5e960  LD_XD_XN_IMM.B64                X2, X30, #2400
# [DWARF] common/pa_scheduler_core.h:0
# [SOURCE unavailable]
0x000000000000f334 (+0x0000f074)  071e7fff  MOV_XD_IMM                      X15, #32767
# [DWARF] common/pa_frontend.h:835
#     829 |         case TaskKind::Alloc:
#     830 |             orch.accumulated_output = outputs.tensors[0];
#     831 |             orch.accumulated_sum = outputs.tensors[1];
#     832 |             orch.accumulated_max = outputs.tensors[2];
#     833 |             break;
#     834 |         case TaskKind::Qk:
# >   835 |             orch.qk_scores = outputs.tensors[0];
0x000000000000f338 (+0x0000f078)  03c1e310  ST_XD_XN_IMM.B64                X0, X30, #784
# [DWARF] common/pa_scheduler_core.h:788
#     782 | static_assert(offsetof(LazySampleCallbackTicket, function_id) == 12, "lazy sample callback ticket function offset mismatch");
#     783 | static_assert(offsetof(LazySampleCallbackTicket, won) == 14, "lazy sample callback ticket winner offset mismatch");
#     784 | 
#     785 | PA_DEVICE void BeginLazySampleCallbackSubmit(PA_GM WorkerState &worker, SubmitContext &context) {
#     786 |     // This is BeginSubmit without an already-materialized TaskArgs.  The same
#     787 |     // fields are completed synchronously after the single callback builds args.
# >   788 |     const uint32_t task_id = static_cast<uint32_t>(worker.local_index++);
0x000000000000f33c (+0x0000f07c)  1c862014  LD_XD_XN_IMM.B32                X3, X2, #20
0x000000000000f340 (+0x0000f080)  08003001  ADD_IMM.S64                     X0, X3, #1
0x000000000000f344 (+0x0000f084)  03c7e920  ST_XD_XN_IMM.B64                X3, X30, #2336
0x000000000000f348 (+0x0000f088)  03802014  ST_XD_XN_IMM.B32                X0, X2, #20
# [DWARF] common/pa_scheduler_core.h:790
#     789 |     context.self = &worker;
# >   790 |     context.payload = &worker.payloads[task_id & kPayloadMask];
0x000000000000f34c (+0x0000f08c)  1cc1e900  LD_XD_XN_IMM.B64                X0, X30, #2304
0x000000000000f350 (+0x0000f090)  00c2308a  AND.B64                         X1, X3, X1
0x000000000000f354 (+0x0000f094)  02c2020c  SHL.B64                         X1, #12
0x000000000000f358 (+0x0000f098)  00000081  ADD.S64                         X0, X0, X1
# [DWARF] common/pa_scheduler_core.h:789
#     783 | static_assert(offsetof(LazySampleCallbackTicket, won) == 14, "lazy sample callback ticket winner offset mismatch");
#     784 | 
#     785 | PA_DEVICE void BeginLazySampleCallbackSubmit(PA_GM WorkerState &worker, SubmitContext &context) {
#     786 |     // This is BeginSubmit without an already-materialized TaskArgs.  The same
#     787 |     // fields are completed synchronously after the single callback builds args.
#     788 |     const uint32_t task_id = static_cast<uint32_t>(worker.local_index++);
# >   789 |     context.self = &worker;
0x000000000000f35c (+0x0000f09c)  09c53031  STP_XI_XJ_XN.B64                X2, X0, X19, #24
# [DWARF] common/pa_scheduler_core.h:793
#     790 |     context.payload = &worker.payloads[task_id & kPayloadMask];
#     791 |     context.task_id = static_cast<int32_t>(task_id);
#     792 |     context.tensor_count = 0;
# >   793 |     context.scalar_count = 0;
0x000000000000f360 (+0x0000f0a0)  07000000  MOV_XD_IMM                      X0, #0
0x000000000000f364 (+0x0000f0a4)  08033030  ADD_IMM.S64                     X1, X19, #48
# [DWARF] common/pa_scheduler_core.h:791
#     785 | PA_DEVICE void BeginLazySampleCallbackSubmit(PA_GM WorkerState &worker, SubmitContext &context) {
#     786 |     // This is BeginSubmit without an already-materialized TaskArgs.  The same
#     787 |     // fields are completed synchronously after the single callback builds args.
#     788 |     const uint32_t task_id = static_cast<uint32_t>(worker.local_index++);
#     789 |     context.self = &worker;
#     790 |     context.payload = &worker.payloads[task_id & kPayloadMask];
# >   791 |     context.task_id = static_cast<int32_t>(task_id);
0x000000000000f368 (+0x0000f0a8)  03873028  ST_XD_XN_IMM.B32                X3, X19, #40
# [DWARF] common/pa_scheduler_core.h:792
# >   792 |     context.tensor_count = 0;
0x000000000000f36c (+0x0000f0ac)  0f813580  STI_XN_IMM.B32                  X19, #44
# [DWARF] common/pa_scheduler_core.h:794
#     793 |     context.scalar_count = 0;
# >   794 |     context.result.task_id = task_id;
0x000000000000f370 (+0x0000f0b0)  03c73040  ST_XD_XN_IMM.B64                X3, X19, #64
# [DWARF] common/pa_scheduler_core.h:795
# >   795 |     context.result.count = 0;
0x000000000000f374 (+0x0000f0b4)  0f813900  STI_XN_IMM.B32                  X19, #72
# [DWARF] common/pa_scheduler_core.h:793
#     787 |     // fields are completed synchronously after the single callback builds args.
#     788 |     const uint32_t task_id = static_cast<uint32_t>(worker.local_index++);
#     789 |     context.self = &worker;
#     790 |     context.payload = &worker.payloads[task_id & kPayloadMask];
#     791 |     context.task_id = static_cast<int32_t>(task_id);
#     792 |     context.tensor_count = 0;
# >   793 |     context.scalar_count = 0;
0x000000000000f378 (+0x0000f0b8)  09c01001  STP_XI_XJ_XN.B64                X0, X0, X1, #0
# [DWARF] common/pa_scheduler_core.h:798
#     794 |     context.result.task_id = task_id;
#     795 |     context.result.count = 0;
#     796 |     context.register_mask = 0;
#     797 |     context.output_bytes = 0;
# >   798 |     context.fanin_count = 0;
0x000000000000f37c (+0x0000f0bc)  07000000  MOV_XD_IMM                      X0, #0
0x000000000000f380 (+0x0000f0c0)  0781ffff  MOVK                            X0, #65535, #2
# [DWARF] common/pa_scheduler_core.h:800
#     799 |     context.kernel_id = -1;
# >   800 |     context.won = false;
0x000000000000f384 (+0x0000f0c4)  0f473300  STI_XN_IMM.B16                  X19, #408
# [DWARF] common/pa_scheduler_core.h:798
#     792 |     context.tensor_count = 0;
#     793 |     context.scalar_count = 0;
#     794 |     context.result.task_id = task_id;
#     795 |     context.result.count = 0;
#     796 |     context.register_mask = 0;
#     797 |     context.output_bytes = 0;
# >   798 |     context.fanin_count = 0;
0x000000000000f388 (+0x0000f0c8)  07c1ffff  MOVK                            X0, #65535, #3
# [DWARF] common/pa_scheduler_core.h:802
#     799 |     context.kernel_id = -1;
#     800 |     context.won = false;
#     801 |     context.joint = false;
# >   802 |     context.joint_init = false;
0x000000000000f38c (+0x0000f0cc)  0f073340  STI_XN_IMM.B8                   X19, #410
# [DWARF] common/pa_scheduler_core.h:798
#     792 |     context.tensor_count = 0;
#     793 |     context.scalar_count = 0;
#     794 |     context.result.task_id = task_id;
#     795 |     context.result.count = 0;
#     796 |     context.register_mask = 0;
#     797 |     context.output_bytes = 0;
# >   798 |     context.fanin_count = 0;
0x000000000000f390 (+0x0000f0d0)  03c13190  ST_XD_XN_IMM.B64                X0, X19, #400
# [DWARF] common/pa_scheduler_core.h:804
#     799 |     context.kernel_id = -1;
#     800 |     context.won = false;
#     801 |     context.joint = false;
#     802 |     context.joint_init = false;
#     803 |     context.joint_block = -1;
# >   804 |     context.joint_slot = -1;
0x000000000000f394 (+0x0000f0d4)  0700ffff  MOV_XD_IMM                      X0, #65535
0x000000000000f398 (+0x0000f0d8)  0741ffff  MOVK                            X0, #65535, #1
# [DWARF] common/pa_scheduler_core.h:803
#     797 |     context.output_bytes = 0;
#     798 |     context.fanin_count = 0;
#     799 |     context.kernel_id = -1;
#     800 |     context.won = false;
#     801 |     context.joint = false;
#     802 |     context.joint_init = false;
# >   803 |     context.joint_block = -1;
0x000000000000f39c (+0x0000f0dc)  0f873382  STI_XN_IMM.B32                  X19, #412
# [DWARF] common/pa_scheduler_core.h:804
# >   804 |     context.joint_slot = -1;
0x000000000000f3a0 (+0x0000f0e0)  03c131a0  ST_XD_XN_IMM.B64                X0, X19, #416
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x000000000000f3a4 (+0x0000f0e4)  02808880  MOV_XD_SPR.F32                  X0, SYS_CNT
# [DWARF] ccec/ccec_ops.h:0
# [SOURCE unavailable]
0x000000000000f3a8 (+0x0000f0e8)  1cd5e9b0  LD_XD_XN_IMM.B64                X10, X30, #2480
0x000000000000f3ac (+0x0000f0ec)  1cd7e9a8  LD_XD_XN_IMM.B64                X11, X30, #2472
0x000000000000f3b0 (+0x0000f0f0)  1cd9e9a0  LD_XD_XN_IMM.B64                X12, X30, #2464
0x000000000000f3b4 (+0x0000f0f4)  1cd3e990  LD_XD_XN_IMM.B64                X9, X30, #2448
# [DWARF] common/pa_trace.h:257
#     251 |     TraceContext &trace, uint64_t end_cycle
#     252 | ) {
#     253 | #if PA_BUILD_SUBMIT_PMU
#     254 |     (void)trace;
#     255 |     (void)end_cycle;
#     256 | #else
# >   257 |     if (!trace.atomics_enabled || trace.poll_burst.active_mask == 0) return;
0x000000000000f3b8 (+0x0000f0f8)  03c1e928  ST_XD_XN_IMM.B64                X0, X30, #2344
0x000000000000f3bc (+0x0000f0fc)  1c01355c  LD_XD_XN_IMM.B8                 X0, X19, #1372
0x000000000000f3c0 (+0x0000f100)  0000070e  CMP.S64.EQ                      X0, X14
0x000000000000f3c4 (+0x0000f104)  40200088  JUMPC                           #136
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000f3c8 (+0x0000f108)  1ca535d0  LD_XD_XN_IMM.B32                X18, X19, #1488
0x000000000000f3cc (+0x0000f10c)  02812a00  ZEROEXT.U32                     X0, X18
# [DWARF] common/pa_trace.h:257
#     251 |     TraceContext &trace, uint64_t end_cycle
#     252 | ) {
#     253 | #if PA_BUILD_SUBMIT_PMU
#     254 |     (void)trace;
#     255 |     (void)end_cycle;
#     256 | #else
# >   257 |     if (!trace.atomics_enabled || trace.poll_burst.active_mask == 0) return;
0x000000000000f3d0 (+0x0000f110)  0000070e  CMP.S64.EQ                      X0, X14
0x000000000000f3d4 (+0x0000f114)  40200084  JUMPC                           #132
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000f3d8 (+0x0000f118)  0802c002  ADD_IMM.S64                     X1, X12, #2
0x000000000000f3dc (+0x0000f11c)  00c0b78a  AND.B64                         X0, X11, X15
0x000000000000f3e0 (+0x0000f120)  02c2020e  SHL.B64                         X1, #14
0x000000000000f3e4 (+0x0000f124)  0004a481  ADD.S64                         X2, X10, X9
0x000000000000f3e8 (+0x0000f128)  00040084  MADD.S64                        X2, X0, X1
0x000000000000f3ec (+0x0000f12c)  07360001  MOV_XD_IMM                      X27, #1
0x000000000000f3f0 (+0x0000f130)  00002081  ADD.S64                         X0, X2, X1
0x000000000000f3f4 (+0x0000f134)  08800680  SUB_IMM.S64                     X0, X0, #1664
0x000000000000f3f8 (+0x0000f138)  08000000  ADD_IMM.S64                     X0, X0, #0
0x000000000000f3fc (+0x0000f13c)  082a0548  ADD_IMM.S64                     X21, X0, #1352
0x000000000000f400 (+0x0000f140)  1cb40558  LD_XD_XN_IMM.B32                X26, X0, #1368
0x000000000000f404 (+0x0000f144)  0ceb5c80  LDP_XI_XJ_XN.B64                X21, X25, X21, #0
0x000000000000f408 (+0x0000f148)  0237b080  NEG.S64                         X27, X27
0x000000000000f40c (+0x0000f14c)  07280000  MOV_XD_IMM                      X20, #0
0x000000000000f410 (+0x0000f150)  072e0000  MOV_XD_IMM                      X23, #0
0x000000000000f414 (+0x0000f154)  07266784  MOV_XD_IMM                      X19, #26500
0x000000000000f418 (+0x0000f158)  07670007  MOVK                            X19, #7, #1
0x000000000000f41c (+0x0000f15c)  07a70000  MOVK                            X19, #0, #2
0x000000000000f420 (+0x0000f160)  07e70000  MOVK                            X19, #0, #3
0x000000000000f424 (+0x0000f164)  02040880  MOV_XD_SPR.S64                  X2, PC
0x000000000000f428 (+0x0000f168)  00273101  ADD.S64                         X19, X19, X2
0x000000000000f42c (+0x0000f16c)  40000012  JUMP                            #18
0x000000000000f430 (+0x0000f170)  0804c002  ADD_IMM.S64                     X2, X12, #2
0x000000000000f434 (+0x0000f174)  00c2b78a  AND.B64                         X1, X11, X15
0x000000000000f438 (+0x0000f178)  02c4020e  SHL.B64                         X2, #14
0x000000000000f43c (+0x0000f17c)  0006a481  ADD.S64                         X3, X10, X9
0x000000000000f440 (+0x0000f180)  00061104  MADD.S64                        X3, X1, X2
# [DWARF] common/pa_trace.h:280
#     274 |             call_count, static_cast<uint32_t>(site)
#     275 |         );
#     276 |         if (written) {
#     277 |             if (trace.poll_batch_records == UINT64_MAX) {
#     278 |                 trace.atomic_counter_overflow = true;
#     279 |             } else {
# >   280 |                 ++trace.poll_batch_records;
0x000000000000f444 (+0x0000f184)  08000001  ADD_IMM.S64                     X0, X0, #1
0x000000000000f448 (+0x0000f188)  00023101  ADD.S64                         X1, X3, X2
0x000000000000f44c (+0x0000f18c)  08821680  SUB_IMM.S64                     X1, X1, #1664
0x000000000000f450 (+0x0000f190)  08021000  ADD_IMM.S64                     X1, X1, #0
0x000000000000f454 (+0x0000f194)  03c01578  ST_XD_XN_IMM.B64                X0, X1, #1400
# [DWARF] common/pa_trace.h:283
#     281 |             }
#     282 |         }
# >   283 |         trace.poll_burst.call_count[index] = 0;
0x000000000000f458 (+0x0000f198)  0f976700  STI_XN_IMM.B32                  X22, #1464
# [DWARF] common/pa_trace.h:262
#     256 | #else
#     257 |     if (!trace.atomics_enabled || trace.poll_burst.active_mask == 0) return;
#     258 |     const uint32_t active_mask = trace.poll_burst.active_mask;
#     259 |     // CCEC 默认会把固定 6-site 循环完整展开，再随几十个 phase 边界复制。
#     260 |     // 禁止展开只控制代码体积；循环次数、site 顺序和同 cycle 关闭语义不变。
#     261 |     PA_LOOP_NOUNROLL
# >   262 |     for (uint32_t index = 0; index < kAtomicPollBatchSiteCount; ++index) {
0x000000000000f45c (+0x0000f19c)  082f7001  ADD_IMM.S64                     X23, X23, #1
0x000000000000f460 (+0x0000f1a0)  08273004  ADD_IMM.S64                     X19, X19, #4
0x000000000000f464 (+0x0000f1a4)  0001741e  CMP.S64.NE                      X23, X8
0x000000000000f468 (+0x0000f1a8)  08294001  ADD_IMM.S64                     X20, X20, #1
0x000000000000f46c (+0x0000f1ac)  40200002  JUMPC                           #2
0x000000000000f470 (+0x0000f1b0)  4000004e  JUMP                            #78
# [DWARF] common/pa_trace.h:263
# >   263 |         const uint32_t bit = 1U << index;
0x000000000000f474 (+0x0000f1b4)  02814a00  ZEROEXT.U32                     X0, X20
# [DWARF] common/pa_trace.h:264
# >   264 |         if ((active_mask & bit) == 0) continue;
0x000000000000f478 (+0x0000f1b8)  02832a00  ZEROEXT.U32                     X1, X18
0x000000000000f47c (+0x0000f1bc)  024202c0  SHR.U64                         X1, X0, #0
0x000000000000f480 (+0x0000f1c0)  00c0188a  AND.B64                         X0, X1, X17
0x000000000000f484 (+0x0000f1c4)  0000070e  CMP.S64.EQ                      X0, X14
0x000000000000f488 (+0x0000f1c8)  4020fff5  JUMPC                           #65525
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000f48c (+0x0000f1cc)  0804c002  ADD_IMM.S64                     X2, X12, #2
0x000000000000f490 (+0x0000f1d0)  00c0b78a  AND.B64                         X0, X11, X15
0x000000000000f494 (+0x0000f1d4)  02c4020e  SHL.B64                         X2, #14
0x000000000000f498 (+0x0000f1d8)  0006a481  ADD.S64                         X3, X10, X9
0x000000000000f49c (+0x0000f1dc)  00060104  MADD.S64                        X3, X0, X2
# [DWARF] common/pa_trace.h:265
#     259 |     // CCEC 默认会把固定 6-site 循环完整展开，再随几十个 phase 边界复制。
#     260 |     // 禁止展开只控制代码体积；循环次数、site 顺序和同 cycle 关闭语义不变。
#     261 |     PA_LOOP_NOUNROLL
#     262 |     for (uint32_t index = 0; index < kAtomicPollBatchSiteCount; ++index) {
#     263 |         const uint32_t bit = 1U << index;
#     264 |         if ((active_mask & bit) == 0) continue;
# >   265 |         const uint32_t call_count = trace.poll_burst.call_count[index];
0x000000000000f4a0 (+0x0000f1e0)  02037800  MOV_XD_XN.S64                   X1, X23
0x000000000000f4a4 (+0x0000f1e4)  00003101  ADD.S64                         X0, X3, X2
0x000000000000f4a8 (+0x0000f1e8)  08800680  SUB_IMM.S64                     X0, X0, #1664
0x000000000000f4ac (+0x0000f1ec)  02c20202  SHL.B64                         X1, #2
0x000000000000f4b0 (+0x0000f1f0)  08000000  ADD_IMM.S64                     X0, X0, #0
0x000000000000f4b4 (+0x0000f1f4)  002c0081  ADD.S64                         X22, X0, X1
0x000000000000f4b8 (+0x0000f1f8)  1c8b65b8  LD_XD_XN_IMM.B32                X5, X22, #1464
# [DWARF] common/pa_trace.h:266
# >   266 |         if (call_count == 0 || call_count > kAtomicPollCountMax) {
0x000000000000f4bc (+0x0000f1fc)  07020000  MOV_XD_IMM                      X1, #0
0x000000000000f4c0 (+0x0000f200)  07430100  MOVK                            X1, #256, #1
0x000000000000f4c4 (+0x0000f204)  07040000  MOV_XD_IMM                      X2, #0
0x000000000000f4c8 (+0x0000f208)  0745ff00  MOVK                            X2, #65280, #1
0x000000000000f4cc (+0x0000f20c)  00025082  SUB.S64                         X1, X5, X1
0x000000000000f4d0 (+0x0000f210)  02821a00  ZEROEXT.U32                     X1, X1
0x000000000000f4d4 (+0x0000f214)  0040113e  CMP.U64.GT                      X1, X2
0x000000000000f4d8 (+0x0000f218)  40200003  JUMPC                           #3
# [DWARF] common/pa_trace.h:267
# >   267 |             trace.atomic_counter_overflow = true;
0x000000000000f4dc (+0x0000f21c)  0f160001  STI_XN_IMM.B8                   X0, #1408
0x000000000000f4e0 (+0x0000f220)  4000ffdf  JUMP                            #65503
# [DWARF] common/pa_trace.h:98
#      92 |         default:
#      93 |             return -1;
#      94 |     }
#      95 | }
#      96 | 
#      97 | PA_DEVICE AtomicSite TraceAtomicPollBatchSite(uint32_t index) {
# >    98 |     switch (index) {
0x000000000000f4e4 (+0x0000f224)  02814a00  ZEROEXT.U32                     X0, X20
0x000000000000f4e8 (+0x0000f228)  07020005  MOV_XD_IMM                      X1, #5
0x000000000000f4ec (+0x0000f22c)  004000be  CMP.U64.GT                      X0, X1
0x000000000000f4f0 (+0x0000f230)  070c000f  MOV_XD_IMM                      X6, #15
0x000000000000f4f4 (+0x0000f234)  40200002  JUMPC                           #2
0x000000000000f4f8 (+0x0000f238)  1c8d3000  LD_XD_XN_IMM.B32                X6, X19, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000f4fc (+0x0000f23c)  0804c002  ADD_IMM.S64                     X2, X12, #2
# [DWARF] common/pa_trace.h:271
#     265 |         const uint32_t call_count = trace.poll_burst.call_count[index];
#     266 |         if (call_count == 0 || call_count > kAtomicPollCountMax) {
#     267 |             trace.atomic_counter_overflow = true;
#     268 |             continue;
#     269 |         }
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
# >   271 |         const bool written = WritePollBatchRecordRaw(
0x000000000000f500 (+0x0000f240)  1cc9e928  LD_XD_XN_IMM.B64                X4, X30, #2344
0x000000000000f504 (+0x0000f244)  00c2b78a  AND.B64                         X1, X11, X15
0x000000000000f508 (+0x0000f248)  02c4020e  SHL.B64                         X2, #14
0x000000000000f50c (+0x0000f24c)  0006a481  ADD.S64                         X3, X10, X9
0x000000000000f510 (+0x0000f250)  00061104  MADD.S64                        X3, X1, X2
# [DWARF] common/pa_trace.h:273
#     272 |             trace.core, trace.records, trace.capacity,
# >   273 |             trace.poll_burst.start_cycle[index], end_cycle,
0x000000000000f514 (+0x0000f254)  02017800  MOV_XD_XN.S64                   X0, X23
0x000000000000f518 (+0x0000f258)  00023101  ADD.S64                         X1, X3, X2
0x000000000000f51c (+0x0000f25c)  08821680  SUB_IMM.S64                     X1, X1, #1664
0x000000000000f520 (+0x0000f260)  02c00203  SHL.B64                         X0, #3
0x000000000000f524 (+0x0000f264)  08301000  ADD_IMM.S64                     X24, X1, #0
0x000000000000f528 (+0x0000f268)  00018001  ADD.S64                         X0, X24, X0
0x000000000000f52c (+0x0000f26c)  1cc60588  LD_XD_XN_IMM.B64                X3, X0, #1416
# [DWARF] common/pa_trace.h:271
#     265 |         const uint32_t call_count = trace.poll_burst.call_count[index];
#     266 |         if (call_count == 0 || call_count > kAtomicPollCountMax) {
#     267 |             trace.atomic_counter_overflow = true;
#     268 |             continue;
#     269 |         }
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
# >   271 |         const bool written = WritePollBatchRecordRaw(
0x000000000000f530 (+0x0000f270)  02015800  MOV_XD_XN.S64                   X0, X21
0x000000000000f534 (+0x0000f274)  02039800  MOV_XD_XN.S64                   X1, X25
0x000000000000f538 (+0x0000f278)  0205a800  MOV_XD_XN.S64                   X2, X26
0x000000000000f53c (+0x0000f27c)  070e615e  MOV_XD_IMM                      X7, #24926
0x000000000000f540 (+0x0000f280)  074f0000  MOVK                            X7, #0, #1
0x000000000000f544 (+0x0000f284)  078f0000  MOVK                            X7, #0, #2
0x000000000000f548 (+0x0000f288)  40427000  CALL                            X7, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000f54c (+0x0000f28c)  1cd3e990  LD_XD_XN_IMM.B64                X9, X30, #2448
0x000000000000f550 (+0x0000f290)  071c0000  MOV_XD_IMM                      X14, #0
0x000000000000f554 (+0x0000f294)  1cd9e9a0  LD_XD_XN_IMM.B64                X12, X30, #2464
0x000000000000f558 (+0x0000f298)  071e7fff  MOV_XD_IMM                      X15, #32767
0x000000000000f55c (+0x0000f29c)  1cd7e9a8  LD_XD_XN_IMM.B64                X11, X30, #2472
# [DWARF] common/pa_trace.h:276
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
#     271 |         const bool written = WritePollBatchRecordRaw(
#     272 |             trace.core, trace.records, trace.capacity,
#     273 |             trace.poll_burst.start_cycle[index], end_cycle,
#     274 |             call_count, static_cast<uint32_t>(site)
#     275 |         );
# >   276 |         if (written) {
0x000000000000f560 (+0x0000f2a0)  0000070e  CMP.S64.EQ                      X0, X14
0x000000000000f564 (+0x0000f2a4)  1cd5e9b0  LD_XD_XN_IMM.B64                X10, X30, #2480
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000f568 (+0x0000f2a8)  07220001  MOV_XD_IMM                      X17, #1
0x000000000000f56c (+0x0000f2ac)  07100006  MOV_XD_IMM                      X8, #6
# [DWARF] common/pa_trace.h:276
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
#     271 |         const bool written = WritePollBatchRecordRaw(
#     272 |             trace.core, trace.records, trace.capacity,
#     273 |             trace.poll_burst.start_cycle[index], end_cycle,
#     274 |             call_count, static_cast<uint32_t>(site)
#     275 |         );
# >   276 |         if (written) {
0x000000000000f570 (+0x0000f2b0)  4020ffba  JUMPC                           #65466
# [DWARF] common/pa_trace.h:277
# >   277 |             if (trace.poll_batch_records == UINT64_MAX) {
0x000000000000f574 (+0x0000f2b4)  1cc18578  LD_XD_XN_IMM.B64                X0, X24, #1400
0x000000000000f578 (+0x0000f2b8)  00000d9e  CMP.S64.NE                      X0, X27
0x000000000000f57c (+0x0000f2bc)  4020ffad  JUMPC                           #65453
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000f580 (+0x0000f2c0)  0802c002  ADD_IMM.S64                     X1, X12, #2
0x000000000000f584 (+0x0000f2c4)  00c0b78a  AND.B64                         X0, X11, X15
0x000000000000f588 (+0x0000f2c8)  02c2020e  SHL.B64                         X1, #14
0x000000000000f58c (+0x0000f2cc)  0004a481  ADD.S64                         X2, X10, X9
0x000000000000f590 (+0x0000f2d0)  00040084  MADD.S64                        X2, X0, X1
0x000000000000f594 (+0x0000f2d4)  00002081  ADD.S64                         X0, X2, X1
0x000000000000f598 (+0x0000f2d8)  08800680  SUB_IMM.S64                     X0, X0, #1664
# [DWARF] common/pa_trace.h:278
#     272 |             trace.core, trace.records, trace.capacity,
#     273 |             trace.poll_burst.start_cycle[index], end_cycle,
#     274 |             call_count, static_cast<uint32_t>(site)
#     275 |         );
#     276 |         if (written) {
#     277 |             if (trace.poll_batch_records == UINT64_MAX) {
# >   278 |                 trace.atomic_counter_overflow = true;
0x000000000000f59c (+0x0000f2dc)  08000000  ADD_IMM.S64                     X0, X0, #0
0x000000000000f5a0 (+0x0000f2e0)  0f160001  STI_XN_IMM.B8                   X0, #1408
0x000000000000f5a4 (+0x0000f2e4)  4000ffad  JUMP                            #65453
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000f5a8 (+0x0000f2e8)  1cf3e980  LD_XD_XN_IMM.B64                X25, X30, #2432
0x000000000000f5ac (+0x0000f2ec)  0802c002  ADD_IMM.S64                     X1, X12, #2
0x000000000000f5b0 (+0x0000f2f0)  00c0b78a  AND.B64                         X0, X11, X15
0x000000000000f5b4 (+0x0000f2f4)  02c2020e  SHL.B64                         X1, #14
0x000000000000f5b8 (+0x0000f2f8)  0004a481  ADD.S64                         X2, X10, X9
0x000000000000f5bc (+0x0000f2fc)  00040084  MADD.S64                        X2, X0, X1
0x000000000000f5c0 (+0x0000f300)  07280001  MOV_XD_IMM                      X20, #1
0x000000000000f5c4 (+0x0000f304)  00002081  ADD.S64                         X0, X2, X1
0x000000000000f5c8 (+0x0000f308)  08800680  SUB_IMM.S64                     X0, X0, #1664
# [DWARF] common/pa_trace.h:285
#     279 |             } else {
#     280 |                 ++trace.poll_batch_records;
#     281 |             }
#     282 |         }
#     283 |         trace.poll_burst.call_count[index] = 0;
#     284 |     }
# >   285 |     trace.poll_burst.active_mask = 0;
0x000000000000f5cc (+0x0000f30c)  08000000  ADD_IMM.S64                     X0, X0, #0
0x000000000000f5d0 (+0x0000f310)  02294080  NEG.S64                         X20, X20
0x000000000000f5d4 (+0x0000f314)  0f960a00  STI_XN_IMM.B32                  X0, #1488
0x000000000000f5d8 (+0x0000f318)  072a0004  MOV_XD_IMM                      X21, #4
0x000000000000f5dc (+0x0000f31c)  072c12c4  MOV_XD_IMM                      X22, #4804
0x000000000000f5e0 (+0x0000f320)  07300020  MOV_XD_IMM                      X24, #32
# [DWARF] common/pa_scheduler_core.h:1128
#    1122 |     const uint32_t task_id = static_cast<uint32_t>(context.task_id);
#    1123 | #if PA_BUILD_SUBMIT_PMU
#    1124 |     const uint64_t submit_begin = task_id == 0 ? Ops::Now() : 0;
#    1125 | #else
#    1126 |     const uint64_t submit_begin = TraceTimestamp<Ops>(stats.trace, stats.result);
#    1127 | #endif
# >  1128 |     if (task_id == 0) stats.result.submit_begin = submit_begin;
0x000000000000f5e4 (+0x0000f324)  1cc1e920  LD_XD_XN_IMM.B64                X0, X30, #2336
0x000000000000f5e8 (+0x0000f328)  0000071e  CMP.S64.NE                      X0, X14
0x000000000000f5ec (+0x0000f32c)  4020000b  JUMPC                           #11
# [DWARF] common/pa_scheduler_core.h:0
# [SOURCE unavailable]
0x000000000000f5f0 (+0x0000f330)  0802c002  ADD_IMM.S64                     X1, X12, #2
0x000000000000f5f4 (+0x0000f334)  00c0b78a  AND.B64                         X0, X11, X15
0x000000000000f5f8 (+0x0000f338)  02c2020e  SHL.B64                         X1, #14
0x000000000000f5fc (+0x0000f33c)  0004a481  ADD.S64                         X2, X10, X9
0x000000000000f600 (+0x0000f340)  00040084  MADD.S64                        X2, X0, X1
0x000000000000f604 (+0x0000f344)  00002081  ADD.S64                         X0, X2, X1
# [DWARF] common/pa_scheduler_core.h:1128
#    1122 |     const uint32_t task_id = static_cast<uint32_t>(context.task_id);
#    1123 | #if PA_BUILD_SUBMIT_PMU
#    1124 |     const uint64_t submit_begin = task_id == 0 ? Ops::Now() : 0;
#    1125 | #else
#    1126 |     const uint64_t submit_begin = TraceTimestamp<Ops>(stats.trace, stats.result);
#    1127 | #endif
# >  1128 |     if (task_id == 0) stats.result.submit_begin = submit_begin;
0x000000000000f608 (+0x0000f348)  1cc3e928  LD_XD_XN_IMM.B64                X1, X30, #2344
0x000000000000f60c (+0x0000f34c)  08800680  SUB_IMM.S64                     X0, X0, #1664
0x000000000000f610 (+0x0000f350)  08000000  ADD_IMM.S64                     X0, X0, #0
0x000000000000f614 (+0x0000f354)  03c201c0  ST_XD_XN_IMM.B64                X1, X0, #448
# [DWARF] common/pa_scheduler_core.h:278
#     272 | template <typename Ops>
#     273 | PA_DEVICE uint32_t DrainReady(
#     274 |     PA_GM SchedulerState *state, PA_GM WorkerState &worker, DrainPlace place, LocalStats &stats
#     275 | ) {
#     276 |     // 同一套 drain 被三个位置复用：每次 Submit 开头的 EfDrain、ring 背压等待和所有 Submit 后的最终 drain。
#     277 |     // slot 属于当前 worker；只有其全部跨核 fanin 已 ready 时才执行所选 winner 负载、发布完成并释放 slot。
# >   278 |     if (worker.occupied_count == 0) {
0x000000000000f618 (+0x0000f358)  1cc1e960  LD_XD_XN_IMM.B64                X0, X30, #2400
0x000000000000f61c (+0x0000f35c)  0702dba0  MOV_XD_IMM                      X1, #56224
0x000000000000f620 (+0x0000f360)  0743000c  MOVK                            X1, #12, #1
0x000000000000f624 (+0x0000f364)  1cc5e940  LD_XD_XN_IMM.B64                X2, X30, #2368
0x000000000000f628 (+0x0000f368)  00000081  ADD.S64                         X0, X0, X1
0x000000000000f62c (+0x0000f36c)  1c800000  LD_XD_XN_IMM.B32                X0, X0, #0
0x000000000000f630 (+0x0000f370)  0000070e  CMP.S64.EQ                      X0, X14
0x000000000000f634 (+0x0000f374)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x000000000000f638 (+0x0000f378)  07001277  MOV_XD_IMM                      X0, #4727
0x000000000000f63c (+0x0000f37c)  07410000  MOVK                            X0, #0, #1
0x000000000000f640 (+0x0000f380)  07810000  MOVK                            X0, #0, #2
0x000000000000f644 (+0x0000f384)  40220000  JUMPC                           X0, #0
0x000000000000f648 (+0x0000f388)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
# [DWARF] common/pa_scheduler_core.h:0
# [SOURCE unavailable]
0x000000000000f64c (+0x0000f38c)  07020000  MOV_XD_IMM                      X1, #0
0x000000000000f650 (+0x0000f390)  4000003d  JUMP                            #61
0x000000000000f654 (+0x0000f394)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x000000000000f658 (+0x0000f398)  080ac002  ADD_IMM.S64                     X5, X12, #2
# [DWARF] common/pa_trace.h:552
#     546 |     PA_GM TraceCoreState &core = *trace.core;
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
# >   552 |     PA_GM TraceRecord &record = trace.records[slot];
0x000000000000f65c (+0x0000f39c)  02060800  MOV_XD_XN.S64                   X3, X0
0x000000000000f660 (+0x0000f3a0)  00c8b78a  AND.B64                         X4, X11, X15
0x000000000000f664 (+0x0000f3a4)  02ca020e  SHL.B64                         X5, #14
0x000000000000f668 (+0x0000f3a8)  000ca481  ADD.S64                         X6, X10, X9
0x000000000000f66c (+0x0000f3ac)  02c60206  SHL.B64                         X3, #6
0x000000000000f670 (+0x0000f3b0)  000c4284  MADD.S64                        X6, X4, X5
0x000000000000f674 (+0x0000f3b4)  0007b181  ADD.S64                         X3, X27, X3
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000f678 (+0x0000f3b8)  00086281  ADD.S64                         X4, X6, X5
# [DWARF] common/pa_trace.h:553
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
# >   553 |     record.start_cycle = start_cycle;
0x000000000000f67c (+0x0000f3bc)  09ea3a81  STP_XI_XJ_XN.B64                X21, X21, X3, #0
# [DWARF] common/pa_trace.h:555
#     554 |     record.end_cycle = end_cycle;
# >   555 |     record.task_id = task_id;
0x000000000000f680 (+0x0000f3c0)  098430a1  STP_XI_XJ_XN.B32                X2, X1, X3, #16
0x000000000000f684 (+0x0000f3c4)  08844680  SUB_IMM.S64                     X2, X4, #1664
# [DWARF] common/pa_trace.h:558
#     556 |     record.function_id = function_id;
#     557 |     record.phase = static_cast<int32_t>(trace_phase);
# >   558 |     record.lane = trace.lane;
0x000000000000f688 (+0x0000f3c8)  08022000  ADD_IMM.S64                     X1, X2, #0
0x000000000000f68c (+0x0000f3cc)  1c841560  LD_XD_XN_IMM.B32                X2, X1, #1376
# [DWARF] common/pa_trace.h:557
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
#     555 |     record.task_id = task_id;
#     556 |     record.function_id = function_id;
# >   557 |     record.phase = static_cast<int32_t>(trace_phase);
0x000000000000f690 (+0x0000f3d0)  07080007  MOV_XD_IMM                      X4, #7
# [DWARF] common/pa_trace.h:565
#     558 |     record.lane = trace.lane;
#     559 |     record.block_id = trace.block_id;
#     560 |     record.core_idx = trace.core_idx;
#     561 |     record.flags = flags;
#     562 |     record.auxiliary = auxiliary;
#     563 |     // count 最后更新，使其始终指向下一空槽；单写者条件下无需 reserve/commit 两阶段。
#     564 |     // 最终 FlushTraceCore 会把记录体先于该计数一并导出。
# >   565 |     core.count = slot + 1;
0x000000000000f694 (+0x0000f3d4)  08000001  ADD_IMM.S64                     X0, X0, #1
# [DWARF] common/pa_trace.h:557
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
#     555 |     record.task_id = task_id;
#     556 |     record.function_id = function_id;
# >   557 |     record.phase = static_cast<int32_t>(trace_phase);
0x000000000000f698 (+0x0000f3d8)  09883131  STP_XI_XJ_XN.B32                X4, X2, X3, #24
# [DWARF] common/pa_trace.h:559
#     558 |     record.lane = trace.lane;
# >   559 |     record.block_id = trace.block_id;
0x000000000000f69c (+0x0000f3dc)  08041564  ADD_IMM.S64                     X2, X1, #1380
0x000000000000f6a0 (+0x0000f3e0)  0c842080  LDP_XI_XJ_XN.B32                X2, X1, X2, #0
0x000000000000f6a4 (+0x0000f3e4)  08083020  ADD_IMM.S64                     X4, X3, #32
0x000000000000f6a8 (+0x0000f3e8)  09844081  STP_XI_XJ_XN.B32                X2, X1, X4, #0
# [DWARF] common/pa_trace.h:561
#     560 |     record.core_idx = trace.core_idx;
# >   561 |     record.flags = flags;
0x000000000000f6ac (+0x0000f3ec)  03dc3028  ST_XD_XN_IMM.B64                X14, X3, #40
# [DWARF] common/pa_trace.h:565
#     562 |     record.auxiliary = auxiliary;
#     563 |     // count 最后更新，使其始终指向下一空槽；单写者条件下无需 reserve/commit 两阶段。
#     564 |     // 最终 FlushTraceCore 会把记录体先于该计数一并导出。
# >   565 |     core.count = slot + 1;
0x000000000000f6b0 (+0x0000f3f0)  0381a000  ST_XD_XN_IMM.B32                X0, X26, #0
0x000000000000f6b4 (+0x0000f3f4)  40000002  JUMP                            #2
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000f6b8 (+0x0000f3f8)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
# [DWARF] common/pa_scheduler_core.h:305
#     299 |         WriteTrace<false>(
#     300 |             stats.trace, stats.result, static_cast<int32_t>(slot.task_id), static_cast<int32_t>(slot.kind),
#     301 |             TracePhase::Commit, ProfilePhase::ReplayTail, commit_cycle, commit_cycle
#     302 |         );
#     303 |         slot.built = false;
#     304 |         slot.occupied = false;
# >   305 |         --worker.occupied_count;
0x000000000000f6bc (+0x0000f3fc)  1cc1e960  LD_XD_XN_IMM.B64                X0, X30, #2400
0x000000000000f6c0 (+0x0000f400)  0702dba0  MOV_XD_IMM                      X1, #56224
# [DWARF] common/pa_scheduler_core.h:304
#     298 |         const uint64_t commit_cycle = TraceTimestamp<Ops>(stats.trace, stats.result);
#     299 |         WriteTrace<false>(
#     300 |             stats.trace, stats.result, static_cast<int32_t>(slot.task_id), static_cast<int32_t>(slot.kind),
#     301 |             TracePhase::Commit, ProfilePhase::ReplayTail, commit_cycle, commit_cycle
#     302 |         );
#     303 |         slot.built = false;
# >   304 |         slot.occupied = false;
0x000000000000f6c4 (+0x0000f404)  035c7000  ST_XD_XN_IMM.B16                X14, X7, #0
# [DWARF] common/pa_scheduler_core.h:305
# >   305 |         --worker.occupied_count;
0x000000000000f6c8 (+0x0000f408)  0743000c  MOVK                            X1, #12, #1
0x000000000000f6cc (+0x0000f40c)  0806c002  ADD_IMM.S64                     X3, X12, #2
0x000000000000f6d0 (+0x0000f410)  00c4b78a  AND.B64                         X2, X11, X15
0x000000000000f6d4 (+0x0000f414)  02c6020e  SHL.B64                         X3, #14
0x000000000000f6d8 (+0x0000f418)  0008a481  ADD.S64                         X4, X10, X9
0x000000000000f6dc (+0x0000f41c)  00082184  MADD.S64                        X4, X2, X3
0x000000000000f6e0 (+0x0000f420)  072a0004  MOV_XD_IMM                      X21, #4
# [DWARF] common/pa_scheduler_core.h:0
# [SOURCE unavailable]
0x000000000000f6e4 (+0x0000f424)  00044181  ADD.S64                         X2, X4, X3
0x000000000000f6e8 (+0x0000f428)  08842680  SUB_IMM.S64                     X2, X2, #1664
0x000000000000f6ec (+0x0000f42c)  072c12c4  MOV_XD_IMM                      X22, #4804
0x000000000000f6f0 (+0x0000f430)  07300020  MOV_XD_IMM                      X24, #32
# [DWARF] common/pa_scheduler_core.h:305
#     299 |         WriteTrace<false>(
#     300 |             stats.trace, stats.result, static_cast<int32_t>(slot.task_id), static_cast<int32_t>(slot.kind),
#     301 |             TracePhase::Commit, ProfilePhase::ReplayTail, commit_cycle, commit_cycle
#     302 |         );
#     303 |         slot.built = false;
#     304 |         slot.occupied = false;
# >   305 |         --worker.occupied_count;
0x000000000000f6f4 (+0x0000f434)  00000081  ADD.S64                         X0, X0, X1
0x000000000000f6f8 (+0x0000f438)  1c820000  LD_XD_XN_IMM.B32                X1, X0, #0
0x000000000000f6fc (+0x0000f43c)  08821001  SUB_IMM.S64                     X1, X1, #1
0x000000000000f700 (+0x0000f440)  03820000  ST_XD_XN_IMM.B32                X1, X0, #0
# [DWARF] common/pa_scheduler_core.h:306
# >   306 |         ++stats.result.placement[static_cast<uint32_t>(place)];
0x000000000000f704 (+0x0000f444)  08002000  ADD_IMM.S64                     X0, X2, #0
0x000000000000f708 (+0x0000f448)  1cc5e940  LD_XD_XN_IMM.B64                X2, X30, #2368
0x000000000000f70c (+0x0000f44c)  1cc202c8  LD_XD_XN_IMM.B64                X1, X0, #712
0x000000000000f710 (+0x0000f450)  08021001  ADD_IMM.S64                     X1, X1, #1
0x000000000000f714 (+0x0000f454)  03c202c8  ST_XD_XN_IMM.B64                X1, X0, #712
# [DWARF] common/pa_scheduler_core.h:0
# [SOURCE unavailable]
0x000000000000f718 (+0x0000f458)  40000002  JUMP                            #2
0x000000000000f71c (+0x0000f45c)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x000000000000f720 (+0x0000f460)  1cc3e978  LD_XD_XN_IMM.B64                X1, X30, #2424
# [DWARF] common/pa_scheduler_core.h:283
#     277 |     // slot 属于当前 worker；只有其全部跨核 fanin 已 ready 时才执行所选 winner 负载、发布完成并释放 slot。
#     278 |     if (worker.occupied_count == 0) {
#     279 |         return 0;
#     280 |     }
#     281 |     uint32_t freed = 0;
#     282 |     // 一次调用遍历完本核全部私有 slot；未就绪项保留 occupied/built，已完成项立即清槽并继续扫描。
# >   283 |     for (uint32_t index = 0; index < kPrivateSlots; ++index) {
0x000000000000f724 (+0x0000f464)  08021001  ADD_IMM.S64                     X1, X1, #1
0x000000000000f728 (+0x0000f468)  00001a9e  CMP.S64.NE                      X1, X21
0x000000000000f72c (+0x0000f46c)  40200006  JUMPC                           #6
0x000000000000f730 (+0x0000f470)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x000000000000f734 (+0x0000f474)  07001238  MOV_XD_IMM                      X0, #4664
0x000000000000f738 (+0x0000f478)  07410000  MOVK                            X0, #0, #1
0x000000000000f73c (+0x0000f47c)  07810000  MOVK                            X0, #0, #2
0x000000000000f740 (+0x0000f480)  40020000  JUMP                            X0, #0
# [DWARF] common/pa_scheduler_core.h:284
# >   284 |         PA_GM LocalSlot &slot = worker.slots[index];
0x000000000000f744 (+0x0000f484)  070012d8  MOV_XD_IMM                      X0, #4824
0x000000000000f748 (+0x0000f488)  03c3e978  ST_XD_XN_IMM.B64                X1, X30, #2424
0x000000000000f74c (+0x0000f48c)  00001003  MUL.S64                         X0, X1, X0
# [DWARF] common/pa_scheduler_core.h:285
# >   285 |         if (!slot.occupied || !slot.built || !SlotReady<Ops>(state, slot, stats)) {
0x000000000000f750 (+0x0000f490)  01022000  LD_XD_XN.B8                     X1, X2, X0
0x000000000000f754 (+0x0000f494)  0000170e  CMP.S64.EQ                      X1, X14
0x000000000000f758 (+0x0000f498)  4020fff2  JUMPC                           #65522
# [DWARF] common/pa_scheduler_core.h:0
# [SOURCE unavailable]
0x000000000000f75c (+0x0000f49c)  00242001  ADD.S64                         X18, X2, X0
# [DWARF] common/pa_scheduler_core.h:285
#     279 |         return 0;
#     280 |     }
#     281 |     uint32_t freed = 0;
#     282 |     // 一次调用遍历完本核全部私有 slot；未就绪项保留 occupied/built，已完成项立即清槽并继续扫描。
#     283 |     for (uint32_t index = 0; index < kPrivateSlots; ++index) {
#     284 |         PA_GM LocalSlot &slot = worker.slots[index];
# >   285 |         if (!slot.occupied || !slot.built || !SlotReady<Ops>(state, slot, stats)) {
0x000000000000f760 (+0x0000f4a0)  1c012001  LD_XD_XN_IMM.B8                 X0, X18, #1
0x000000000000f764 (+0x0000f4a4)  0000070e  CMP.S64.EQ                      X0, X14
0x000000000000f768 (+0x0000f4a8)  4020ffee  JUMPC                           #65518
# [DWARF] common/pa_scheduler_core.h:0
# [SOURCE unavailable]
0x000000000000f76c (+0x0000f4ac)  00012b01  ADD.S64                         X0, X18, X22
0x000000000000f770 (+0x0000f4b0)  03e5e968  ST_XD_XN_IMM.B64                X18, X30, #2408
# [DWARF] common/pa_scheduler_core.h:246
#     240 |     AdvanceFrontier<Ops>(state, stats);
#     241 | }
#     242 | 
#     243 | template <typename Ops>
#     244 | PA_DEVICE bool SlotReady(PA_GM SchedulerState *state, PA_GM LocalSlot &slot, LocalStats &stats) {
#     245 |     // 每个 fanin flag 都是跨核共享的完成条件；遇到第一个未就绪依赖即返回，后续 drain 会再次轮询。
# >   246 |     for (uint32_t index = 0; index < slot.fanin_count; ++index) {
0x000000000000f774 (+0x0000f4b4)  1c800000  LD_XD_XN_IMM.B32                X0, X0, #0
0x000000000000f778 (+0x0000f4b8)  0000070e  CMP.S64.EQ                      X0, X14
0x000000000000f77c (+0x0000f4bc)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x000000000000f780 (+0x0000f4c0)  0700048d  MOV_XD_IMM                      X0, #1165
0x000000000000f784 (+0x0000f4c4)  07410000  MOVK                            X0, #0, #1
0x000000000000f788 (+0x0000f4c8)  07810000  MOVK                            X0, #0, #2
0x000000000000f78c (+0x0000f4cc)  40220000  JUMPC                           X0, #0
0x000000000000f790 (+0x0000f4d0)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
# [DWARF] common/pa_scheduler_core.h:0
# [SOURCE unavailable]
0x000000000000f794 (+0x0000f4d4)  07001284  MOV_XD_IMM                      X0, #4740
0x000000000000f798 (+0x0000f4d8)  001b2001  ADD.S64                         X13, X18, X0
0x000000000000f79c (+0x0000f4dc)  072e0000  MOV_XD_IMM                      X23, #0
0x000000000000f7a0 (+0x0000f4e0)  03dbe930  ST_XD_XN_IMM.B64                X13, X30, #2352
0x000000000000f7a4 (+0x0000f4e4)  40000002  JUMP                            #2
0x000000000000f7a8 (+0x0000f4e8)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x000000000000f7ac (+0x0000f4ec)  0802c002  ADD_IMM.S64                     X1, X12, #2
0x000000000000f7b0 (+0x0000f4f0)  00c0b78a  AND.B64                         X0, X11, X15
0x000000000000f7b4 (+0x0000f4f4)  02c2020e  SHL.B64                         X1, #14
0x000000000000f7b8 (+0x0000f4f8)  0004a481  ADD.S64                         X2, X10, X9
0x000000000000f7bc (+0x0000f4fc)  00040084  MADD.S64                        X2, X0, X1
# [DWARF] common/pa_scheduler_core.h:247
#     241 | }
#     242 | 
#     243 | template <typename Ops>
#     244 | PA_DEVICE bool SlotReady(PA_GM SchedulerState *state, PA_GM LocalSlot &slot, LocalStats &stats) {
#     245 |     // 每个 fanin flag 都是跨核共享的完成条件；遇到第一个未就绪依赖即返回，后续 drain 会再次轮询。
#     246 |     for (uint32_t index = 0; index < slot.fanin_count; ++index) {
# >   247 |         const int32_t dependency = slot.fanin[index];
0x000000000000f7c0 (+0x0000f500)  0180db80  LD_XD_XN.B32                    X0, X13, X23
# [DWARF] common/pa_scheduler_core.h:0
# [SOURCE unavailable]
0x000000000000f7c4 (+0x0000f504)  00022081  ADD.S64                         X1, X2, X1
0x000000000000f7c8 (+0x0000f508)  08821680  SUB_IMM.S64                     X1, X1, #1664
# [DWARF] common/pa_trace.h:393
#     387 |     (void)result;
#     388 |     (void)task_id;
#     389 |     (void)site;
#     390 |     (void)result_used;
#     391 |     return Ops::Load(address);
#     392 | #else
# >   393 |     if (!trace.atomics_enabled) return Ops::Load(address);
0x000000000000f7cc (+0x0000f50c)  08021000  ADD_IMM.S64                     X1, X1, #0
0x000000000000f7d0 (+0x0000f510)  1c04155c  LD_XD_XN_IMM.B8                 X2, X1, #1372
# [DWARF] common/pa_scheduler_core.h:250
#     244 | PA_DEVICE bool SlotReady(PA_GM SchedulerState *state, PA_GM LocalSlot &slot, LocalStats &stats) {
#     245 |     // 每个 fanin flag 都是跨核共享的完成条件；遇到第一个未就绪依赖即返回，后续 drain 会再次轮询。
#     246 |     for (uint32_t index = 0; index < slot.fanin_count; ++index) {
#     247 |         const int32_t dependency = slot.fanin[index];
#     248 |         if (TraceAtomicLoad<Ops>(
#     249 |                 stats.trace, stats.result, dependency, AtomicSite::FaninFlagLoad,
# >   250 |                 &state->tasks[dependency].flag
0x000000000000f7d4 (+0x0000f514)  02860980  SIGNEXT.S32                     X3, X0
0x000000000000f7d8 (+0x0000f518)  02c60206  SHL.B64                         X3, #6
# [DWARF] common/pa_trace.h:393
#     387 |     (void)result;
#     388 |     (void)task_id;
#     389 |     (void)site;
#     390 |     (void)result_used;
#     391 |     return Ops::Load(address);
#     392 | #else
# >   393 |     if (!trace.atomics_enabled) return Ops::Load(address);
0x000000000000f7dc (+0x0000f51c)  0000271e  CMP.S64.NE                      X2, X14
# [DWARF] common/pa_scheduler_core.h:250
#     244 | PA_DEVICE bool SlotReady(PA_GM SchedulerState *state, PA_GM LocalSlot &slot, LocalStats &stats) {
#     245 |     // 每个 fanin flag 都是跨核共享的完成条件；遇到第一个未就绪依赖即返回，后续 drain 会再次轮询。
#     246 |     for (uint32_t index = 0; index < slot.fanin_count; ++index) {
#     247 |         const int32_t dependency = slot.fanin[index];
#     248 |         if (TraceAtomicLoad<Ops>(
#     249 |                 stats.trace, stats.result, dependency, AtomicSite::FaninFlagLoad,
# >   250 |                 &state->tasks[dependency].flag
0x000000000000f7e0 (+0x0000f520)  00059181  ADD.S64                         X2, X25, X3
# [DWARF] common/pa_trace.h:393
#     387 |     (void)result;
#     388 |     (void)task_id;
#     389 |     (void)site;
#     390 |     (void)result_used;
#     391 |     return Ops::Load(address);
#     392 | #else
# >   393 |     if (!trace.atomics_enabled) return Ops::Load(address);
0x000000000000f7e4 (+0x0000f524)  40200006  JUMPC                           #6
0x000000000000f7e8 (+0x0000f528)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x000000000000f7ec (+0x0000f52c)  0700001e  MOV_XD_IMM                      X0, #30
0x000000000000f7f0 (+0x0000f530)  07410000  MOVK                            X0, #0, #1
0x000000000000f7f4 (+0x0000f534)  07810000  MOVK                            X0, #0, #2
0x000000000000f7f8 (+0x0000f538)  40020000  JUMP                            X0, #0
# [DWARF] common/pa_trace.h:337
#     331 | 
#     332 | PA_DEVICE bool AtomicPollBatchEnabled(
#     333 |     TraceContext &trace, AtomicSite site, AtomicOp actual_op
#     334 | ) {
#     335 |     return trace.atomics_enabled && TraceAtomicSiteIsPollBatchable(site) &&
#     336 |            TraceAtomicSiteExpectedOp(site) == actual_op &&
# >   337 |            (trace.poll_burst.enabled_mask & TraceAtomicSiteMask(site)) != 0;
0x000000000000f7fc (+0x0000f53c)  1c0215d4  LD_XD_XN_IMM.B8                 X1, X1, #1492
0x000000000000f800 (+0x0000f540)  00c21c0a  AND.B64                         X1, X1, X24
0x000000000000f804 (+0x0000f544)  0000170e  CMP.S64.EQ                      X1, X14
# [DWARF] common/pa_trace.h:395
#     389 |     (void)site;
#     390 |     (void)result_used;
#     391 |     return Ops::Load(address);
#     392 | #else
#     393 |     if (!trace.atomics_enabled) return Ops::Load(address);
#     394 |     const bool poll_batch = result_used && AtomicPollBatchEnabled(trace, site, AtomicOp::Load);
# >   395 |     const int32_t poll_index = poll_batch ? TraceAtomicPollBatchIndex(site) : -1;
0x000000000000f808 (+0x0000f548)  40200006  JUMPC                           #6
0x000000000000f80c (+0x0000f54c)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x000000000000f810 (+0x0000f550)  07000018  MOV_XD_IMM                      X0, #24
0x000000000000f814 (+0x0000f554)  07410000  MOVK                            X0, #0, #1
0x000000000000f818 (+0x0000f558)  07810000  MOVK                            X0, #0, #2
0x000000000000f81c (+0x0000f55c)  40020000  JUMP                            X0, #0
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x000000000000f820 (+0x0000f560)  02828880  MOV_XD_SPR.F32                  X1, SYS_CNT
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:86
#      80 |     return __builtin_cce_atom_max_G_##SUFFIX(base, inc, (uint32_t)L2Cache);    \
#      81 |   }
#      82 | 
#      83 | __CCE__ATOM_G_BUILTIN(uint32_t, u32)
#      84 | __CCE__ATOM_G_BUILTIN(int32_t, s32)
#      85 | __CCE__ATOM_G_BUILTIN(uint64_t, u64)
# >    86 | __CCE__ATOM_G_BUILTIN(int64_t, s64)
0x000000000000f824 (+0x0000f564)  51a809c0  ATOM                            XN, XM, XD, ADD
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:0
# [SOURCE unavailable]
0x000000000000f828 (+0x0000f568)  0806c002  ADD_IMM.S64                     X3, X12, #2
0x000000000000f82c (+0x0000f56c)  00c4b78a  AND.B64                         X2, X11, X15
0x000000000000f830 (+0x0000f570)  02c6020e  SHL.B64                         X3, #14
0x000000000000f834 (+0x0000f574)  0008a481  ADD.S64                         X4, X10, X9
0x000000000000f838 (+0x0000f578)  00082184  MADD.S64                        X4, X2, X3
0x000000000000f83c (+0x0000f57c)  00044181  ADD.S64                         X2, X4, X3
0x000000000000f840 (+0x0000f580)  08842680  SUB_IMM.S64                     X2, X2, #1664
# [DWARF] common/pa_trace.h:236
#     230 |            (encoded_retries << kAtomicRetriesShift);
#     231 | }
#     232 | 
#     233 | PA_DEVICE void CountAtomicCall(
#     234 |     TraceContext &trace, WorkerResult &result, bool poll_batch
#     235 | ) {
# >   236 |     if (result.atomic_trace_calls == UINT64_MAX) {
0x000000000000f844 (+0x0000f584)  08042000  ADD_IMM.S64                     X2, X2, #0
# [DWARF] ccec/ccec_ops.h:239
#     233 |         static_assert(sizeof(T) == 4 || sizeof(T) == 8, "atomic dependency expects a scalar result");
#     234 |         uint64_t cycle = 0;
#     235 |         // 同一个 inline asm 块先真正消费 atomic 返回寄存器，再读取
#     236 |         // SYS_CNT；编译器不能把 t1 拆到依赖 MOV 之前。AIC/AIV 对该序列
#     237 |         // 生成相同指令字节，且不增加 DSB/ISB/GM 访存。该边界仍只表示
#     238 |         // 返回值已可被本核 scalar 消费，不表示跨核全局可见。
# >   239 |         asm volatile(
0x000000000000f848 (+0x0000f588)  020b0800  MOV_XD_XN.S64                   X5, X16
0x000000000000f84c (+0x0000f58c)  020a5800  MOV_XD_XN.S64                   X5, X5
0x000000000000f850 (+0x0000f590)  02868880  MOV_XD_SPR.F32                  X3, SYS_CNT
# [DWARF] common/pa_trace.h:236
#     230 |            (encoded_retries << kAtomicRetriesShift);
#     231 | }
#     232 | 
#     233 | PA_DEVICE void CountAtomicCall(
#     234 |     TraceContext &trace, WorkerResult &result, bool poll_batch
#     235 | ) {
# >   236 |     if (result.atomic_trace_calls == UINT64_MAX) {
0x000000000000f854 (+0x0000f594)  1cc824a0  LD_XD_XN_IMM.B64                X4, X2, #1184
0x000000000000f858 (+0x0000f598)  00004a1e  CMP.S64.NE                      X4, X20
0x000000000000f85c (+0x0000f59c)  40200002  JUMPC                           #2
0x000000000000f860 (+0x0000f5a0)  400000d5  JUMP                            #213
# [DWARF] common/pa_trace.h:240
#     237 |         trace.atomic_counter_overflow = true;
#     238 |         return;
#     239 |     }
# >   240 |     ++result.atomic_trace_calls;
0x000000000000f864 (+0x0000f5a4)  08084001  ADD_IMM.S64                     X4, X4, #1
0x000000000000f868 (+0x0000f5a8)  03c824a0  ST_XD_XN_IMM.B64                X4, X2, #1184
0x000000000000f86c (+0x0000f5ac)  400000d3  JUMP                            #211
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000f870 (+0x0000f5b0)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:86
#      80 |     return __builtin_cce_atom_max_G_##SUFFIX(base, inc, (uint32_t)L2Cache);    \
#      81 |   }
#      82 | 
#      83 | __CCE__ATOM_G_BUILTIN(uint32_t, u32)
#      84 | __CCE__ATOM_G_BUILTIN(int32_t, s32)
#      85 | __CCE__ATOM_G_BUILTIN(uint64_t, u64)
# >    86 | __CCE__ATOM_G_BUILTIN(int64_t, s64)
0x000000000000f874 (+0x0000f5b4)  51a809c0  ATOM                            XN, XM, XD, ADD
0x000000000000f878 (+0x0000f5b8)  40000436  JUMP                            #1078
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:0
# [SOURCE unavailable]
0x000000000000f87c (+0x0000f5bc)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x000000000000f880 (+0x0000f5c0)  0802c002  ADD_IMM.S64                     X1, X12, #2
0x000000000000f884 (+0x0000f5c4)  00c0b78a  AND.B64                         X0, X11, X15
0x000000000000f888 (+0x0000f5c8)  02c2020e  SHL.B64                         X1, #14
0x000000000000f88c (+0x0000f5cc)  0006a481  ADD.S64                         X3, X10, X9
0x000000000000f890 (+0x0000f5d0)  00060084  MADD.S64                        X3, X0, X1
0x000000000000f894 (+0x0000f5d4)  00003081  ADD.S64                         X0, X3, X1
0x000000000000f898 (+0x0000f5d8)  08800680  SUB_IMM.S64                     X0, X0, #1664
# [DWARF] common/pa_trace.h:396
#     390 |     (void)result_used;
#     391 |     return Ops::Load(address);
#     392 | #else
#     393 |     if (!trace.atomics_enabled) return Ops::Load(address);
#     394 |     const bool poll_batch = result_used && AtomicPollBatchEnabled(trace, site, AtomicOp::Load);
#     395 |     const int32_t poll_index = poll_batch ? TraceAtomicPollBatchIndex(site) : -1;
# >   396 |     const bool first_in_batch = poll_batch &&
0x000000000000f89c (+0x0000f5dc)  08060000  ADD_IMM.S64                     X3, X0, #0
0x000000000000f8a0 (+0x0000f5e0)  1ca635d0  LD_XD_XN_IMM.B32                X19, X3, #1488
0x000000000000f8a4 (+0x0000f5e4)  07000000  MOV_XD_IMM                      X0, #0
0x000000000000f8a8 (+0x0000f5e8)  00c33a8a  AND.B64                         X1, X19, X21
0x000000000000f8ac (+0x0000f5ec)  0000101e  CMP.S64.NE                      X1, X0
# [DWARF] common/pa_trace.h:398
#     397 |         (trace.poll_burst.active_mask & (1U << static_cast<uint32_t>(poll_index))) == 0;
# >   398 |     const uint64_t begin = !poll_batch || first_in_batch ? Ops::Now() : 0;
0x000000000000f8b0 (+0x0000f5f0)  40200002  JUMPC                           #2
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x000000000000f8b4 (+0x0000f5f4)  02808880  MOV_XD_SPR.F32                  X0, SYS_CNT
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:86
#      80 |     return __builtin_cce_atom_max_G_##SUFFIX(base, inc, (uint32_t)L2Cache);    \
#      81 |   }
#      82 | 
#      83 | __CCE__ATOM_G_BUILTIN(uint32_t, u32)
#      84 | __CCE__ATOM_G_BUILTIN(int32_t, s32)
#      85 | __CCE__ATOM_G_BUILTIN(uint64_t, u64)
# >    86 | __CCE__ATOM_G_BUILTIN(int64_t, s64)
0x000000000000f8b8 (+0x0000f5f8)  51a809c0  ATOM                            XN, XM, XD, ADD
# [DWARF] common/pa_trace.h:236
#     230 |            (encoded_retries << kAtomicRetriesShift);
#     231 | }
#     232 | 
#     233 | PA_DEVICE void CountAtomicCall(
#     234 |     TraceContext &trace, WorkerResult &result, bool poll_batch
#     235 | ) {
# >   236 |     if (result.atomic_trace_calls == UINT64_MAX) {
0x000000000000f8bc (+0x0000f5fc)  1cc434a0  LD_XD_XN_IMM.B64                X2, X3, #1184
0x000000000000f8c0 (+0x0000f600)  00002a1e  CMP.S64.NE                      X2, X20
0x000000000000f8c4 (+0x0000f604)  40200002  JUMPC                           #2
0x000000000000f8c8 (+0x0000f608)  40000413  JUMP                            #1043
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000f8cc (+0x0000f60c)  0808c002  ADD_IMM.S64                     X4, X12, #2
0x000000000000f8d0 (+0x0000f610)  00c6b78a  AND.B64                         X3, X11, X15
0x000000000000f8d4 (+0x0000f614)  02c8020e  SHL.B64                         X4, #14
0x000000000000f8d8 (+0x0000f618)  000aa481  ADD.S64                         X5, X10, X9
0x000000000000f8dc (+0x0000f61c)  000a3204  MADD.S64                        X5, X3, X4
# [DWARF] common/pa_trace.h:240
#     234 |     TraceContext &trace, WorkerResult &result, bool poll_batch
#     235 | ) {
#     236 |     if (result.atomic_trace_calls == UINT64_MAX) {
#     237 |         trace.atomic_counter_overflow = true;
#     238 |         return;
#     239 |     }
# >   240 |     ++result.atomic_trace_calls;
0x000000000000f8e0 (+0x0000f620)  08042001  ADD_IMM.S64                     X2, X2, #1
0x000000000000f8e4 (+0x0000f624)  00065201  ADD.S64                         X3, X5, X4
0x000000000000f8e8 (+0x0000f628)  08863680  SUB_IMM.S64                     X3, X3, #1664
0x000000000000f8ec (+0x0000f62c)  08063000  ADD_IMM.S64                     X3, X3, #0
# [DWARF] common/pa_trace.h:242
#     241 |     if (!poll_batch) return;
# >   242 |     if (trace.poll_calls == UINT64_MAX) {
0x000000000000f8f0 (+0x0000f630)  1cc83570  LD_XD_XN_IMM.B64                X4, X3, #1392
# [DWARF] common/pa_trace.h:240
#     234 |     TraceContext &trace, WorkerResult &result, bool poll_batch
#     235 | ) {
#     236 |     if (result.atomic_trace_calls == UINT64_MAX) {
#     237 |         trace.atomic_counter_overflow = true;
#     238 |         return;
#     239 |     }
# >   240 |     ++result.atomic_trace_calls;
0x000000000000f8f4 (+0x0000f634)  03c434a0  ST_XD_XN_IMM.B64                X2, X3, #1184
# [DWARF] common/pa_trace.h:242
#     241 |     if (!poll_batch) return;
# >   242 |     if (trace.poll_calls == UINT64_MAX) {
0x000000000000f8f8 (+0x0000f638)  00004a1e  CMP.S64.NE                      X4, X20
0x000000000000f8fc (+0x0000f63c)  40200002  JUMPC                           #2
0x000000000000f900 (+0x0000f640)  4000042c  JUMP                            #1068
# [DWARF] common/pa_trace.h:246
#     243 |         trace.atomic_counter_overflow = true;
#     244 |         return;
#     245 |     }
# >   246 |     ++trace.poll_calls;
0x000000000000f904 (+0x0000f644)  08044001  ADD_IMM.S64                     X2, X4, #1
0x000000000000f908 (+0x0000f648)  03c43570  ST_XD_XN_IMM.B64                X2, X3, #1392
# [DWARF] common/pa_trace.h:396
#     390 |     (void)result_used;
#     391 |     return Ops::Load(address);
#     392 | #else
#     393 |     if (!trace.atomics_enabled) return Ops::Load(address);
#     394 |     const bool poll_batch = result_used && AtomicPollBatchEnabled(trace, site, AtomicOp::Load);
#     395 |     const int32_t poll_index = poll_batch ? TraceAtomicPollBatchIndex(site) : -1;
# >   396 |     const bool first_in_batch = poll_batch &&
0x000000000000f90c (+0x0000f64c)  0000171e  CMP.S64.NE                      X1, X14
# [DWARF] common/pa_trace.h:351
#     345 |     if (signed_index < 0) {
#     346 |         trace.atomic_counter_overflow = true;
#     347 |         return;
#     348 |     }
#     349 |     const uint32_t index = static_cast<uint32_t>(signed_index);
#     350 |     const uint32_t bit = 1U << index;
# >   351 |     if ((trace.poll_burst.active_mask & bit) == 0) {
0x000000000000f910 (+0x0000f650)  4020000e  JUMPC                           #14
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000f914 (+0x0000f654)  0804c002  ADD_IMM.S64                     X2, X12, #2
0x000000000000f918 (+0x0000f658)  00c2b78a  AND.B64                         X1, X11, X15
0x000000000000f91c (+0x0000f65c)  02c4020e  SHL.B64                         X2, #14
0x000000000000f920 (+0x0000f660)  0006a481  ADD.S64                         X3, X10, X9
0x000000000000f924 (+0x0000f664)  00061104  MADD.S64                        X3, X1, X2
0x000000000000f928 (+0x0000f668)  00023101  ADD.S64                         X1, X3, X2
0x000000000000f92c (+0x0000f66c)  08821680  SUB_IMM.S64                     X1, X1, #1664
# [DWARF] common/pa_trace.h:352
#     346 |         trace.atomic_counter_overflow = true;
#     347 |         return;
#     348 |     }
#     349 |     const uint32_t index = static_cast<uint32_t>(signed_index);
#     350 |     const uint32_t bit = 1U << index;
#     351 |     if ((trace.poll_burst.active_mask & bit) == 0) {
# >   352 |         trace.poll_burst.start_cycle[index] = start_cycle;
0x000000000000f930 (+0x0000f670)  08021000  ADD_IMM.S64                     X1, X1, #0
0x000000000000f934 (+0x0000f674)  03c01598  ST_XD_XN_IMM.B64                X0, X1, #1432
# [DWARF] common/pa_trace.h:354
#     353 |         trace.poll_burst.call_count[index] = 0;
# >   354 |         trace.poll_burst.active_mask |= bit;
0x000000000000f938 (+0x0000f678)  07000002  MOV_XD_IMM                      X0, #2
0x000000000000f93c (+0x0000f67c)  02e60440  SBITSET.B64                     X19, X0
# [DWARF] common/pa_trace.h:353
#     347 |         return;
#     348 |     }
#     349 |     const uint32_t index = static_cast<uint32_t>(signed_index);
#     350 |     const uint32_t bit = 1U << index;
#     351 |     if ((trace.poll_burst.active_mask & bit) == 0) {
#     352 |         trace.poll_burst.start_cycle[index] = start_cycle;
# >   353 |         trace.poll_burst.call_count[index] = 0;
0x000000000000f940 (+0x0000f680)  0f961800  STI_XN_IMM.B32                  X1, #1472
# [DWARF] common/pa_trace.h:354
# >   354 |         trace.poll_burst.active_mask |= bit;
0x000000000000f944 (+0x0000f684)  03a615d0  ST_XD_XN_IMM.B32                X19, X1, #1488
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000f948 (+0x0000f688)  0802c002  ADD_IMM.S64                     X1, X12, #2
0x000000000000f94c (+0x0000f68c)  00c0b78a  AND.B64                         X0, X11, X15
0x000000000000f950 (+0x0000f690)  02c2020e  SHL.B64                         X1, #14
0x000000000000f954 (+0x0000f694)  0004a481  ADD.S64                         X2, X10, X9
0x000000000000f958 (+0x0000f698)  00040084  MADD.S64                        X2, X0, X1
# [DWARF] common/pa_trace.h:358
#     352 |         trace.poll_burst.start_cycle[index] = start_cycle;
#     353 |         trace.poll_burst.call_count[index] = 0;
#     354 |         trace.poll_burst.active_mask |= bit;
#     355 |     }
#     356 |     uint32_t &call_count = trace.poll_burst.call_count[index];
#     357 |     ++call_count;
# >   358 |     if (call_count == kAtomicPollCountMax) {
0x000000000000f95c (+0x0000f69c)  0706ffff  MOV_XD_IMM                      X3, #65535
0x000000000000f960 (+0x0000f6a0)  00002081  ADD.S64                         X0, X2, X1
0x000000000000f964 (+0x0000f6a4)  08800680  SUB_IMM.S64                     X0, X0, #1664
# [DWARF] common/pa_trace.h:357
#     351 |     if ((trace.poll_burst.active_mask & bit) == 0) {
#     352 |         trace.poll_burst.start_cycle[index] = start_cycle;
#     353 |         trace.poll_burst.call_count[index] = 0;
#     354 |         trace.poll_burst.active_mask |= bit;
#     355 |     }
#     356 |     uint32_t &call_count = trace.poll_burst.call_count[index];
# >   357 |     ++call_count;
0x000000000000f968 (+0x0000f6a8)  08000000  ADD_IMM.S64                     X0, X0, #0
0x000000000000f96c (+0x0000f6ac)  1c8205c0  LD_XD_XN_IMM.B32                X1, X0, #1472
# [DWARF] common/pa_trace.h:358
# >   358 |     if (call_count == kAtomicPollCountMax) {
0x000000000000f970 (+0x0000f6b0)  074700ff  MOVK                            X3, #255, #1
# [DWARF] common/pa_trace.h:357
#     351 |     if ((trace.poll_burst.active_mask & bit) == 0) {
#     352 |         trace.poll_burst.start_cycle[index] = start_cycle;
#     353 |         trace.poll_burst.call_count[index] = 0;
#     354 |         trace.poll_burst.active_mask |= bit;
#     355 |     }
#     356 |     uint32_t &call_count = trace.poll_burst.call_count[index];
# >   357 |     ++call_count;
0x000000000000f974 (+0x0000f6b4)  08021001  ADD_IMM.S64                     X1, X1, #1
0x000000000000f978 (+0x0000f6b8)  02841a00  ZEROEXT.U32                     X2, X1
0x000000000000f97c (+0x0000f6bc)  038205c0  ST_XD_XN_IMM.B32                X1, X0, #1472
# [DWARF] common/pa_trace.h:358
# >   358 |     if (call_count == kAtomicPollCountMax) {
0x000000000000f980 (+0x0000f6c0)  0000219e  CMP.S64.NE                      X2, X3
0x000000000000f984 (+0x0000f6c4)  402003f3  JUMPC                           #1011
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000f988 (+0x0000f6c8)  03e1e958  ST_XD_XN_IMM.B64                X16, X30, #2392
0x000000000000f98c (+0x0000f6cc)  08320548  ADD_IMM.S64                     X25, X0, #1352
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x000000000000f990 (+0x0000f6d0)  02828880  MOV_XD_SPR.F32                  X1, SYS_CNT
0x000000000000f994 (+0x0000f6d4)  07280000  MOV_XD_IMM                      X20, #0
0x000000000000f998 (+0x0000f6d8)  0cf39d00  LDP_XI_XJ_XN.B64                X25, X26, X25, #0
0x000000000000f99c (+0x0000f6dc)  07300000  MOV_XD_IMM                      X24, #0
0x000000000000f9a0 (+0x0000f6e0)  1cb60558  LD_XD_XN_IMM.B32                X27, X0, #1368
0x000000000000f9a4 (+0x0000f6e4)  03c3e988  ST_XD_XN_IMM.B64                X1, X30, #2440
0x000000000000f9a8 (+0x0000f6e8)  072461f0  MOV_XD_IMM                      X18, #25072
0x000000000000f9ac (+0x0000f6ec)  07650007  MOVK                            X18, #7, #1
0x000000000000f9b0 (+0x0000f6f0)  07a50000  MOVK                            X18, #0, #2
0x000000000000f9b4 (+0x0000f6f4)  07e50000  MOVK                            X18, #0, #3
0x000000000000f9b8 (+0x0000f6f8)  02040880  MOV_XD_SPR.S64                  X2, PC
0x000000000000f9bc (+0x0000f6fc)  00252101  ADD.S64                         X18, X18, X2
0x000000000000f9c0 (+0x0000f700)  4000001b  JUMP                            #27
# [DWARF] ccec/ccec_ops.h:0
# [SOURCE unavailable]
0x000000000000f9c4 (+0x0000f704)  1cc5e9a0  LD_XD_XN_IMM.B64                X2, X30, #2464
# [DWARF] common/pa_trace.h:280
#     274 |             call_count, static_cast<uint32_t>(site)
#     275 |         );
#     276 |         if (written) {
#     277 |             if (trace.poll_batch_records == UINT64_MAX) {
#     278 |                 trace.atomic_counter_overflow = true;
#     279 |             } else {
# >   280 |                 ++trace.poll_batch_records;
0x000000000000f9c8 (+0x0000f708)  08000001  ADD_IMM.S64                     X0, X0, #1
0x000000000000f9cc (+0x0000f70c)  1cc3e9a8  LD_XD_XN_IMM.B64                X1, X30, #2472
0x000000000000f9d0 (+0x0000f710)  1cc7e9b0  LD_XD_XN_IMM.B64                X3, X30, #2480
0x000000000000f9d4 (+0x0000f714)  1cc9e990  LD_XD_XN_IMM.B64                X4, X30, #2448
0x000000000000f9d8 (+0x0000f718)  08042002  ADD_IMM.S64                     X2, X2, #2
0x000000000000f9dc (+0x0000f71c)  00c21a8a  AND.B64                         X1, X1, X21
0x000000000000f9e0 (+0x0000f720)  02c4020e  SHL.B64                         X2, #14
0x000000000000f9e4 (+0x0000f724)  00063201  ADD.S64                         X3, X3, X4
0x000000000000f9e8 (+0x0000f728)  00061104  MADD.S64                        X3, X1, X2
0x000000000000f9ec (+0x0000f72c)  00023101  ADD.S64                         X1, X3, X2
0x000000000000f9f0 (+0x0000f730)  08821680  SUB_IMM.S64                     X1, X1, #1664
0x000000000000f9f4 (+0x0000f734)  08021000  ADD_IMM.S64                     X1, X1, #0
0x000000000000f9f8 (+0x0000f738)  03c01578  ST_XD_XN_IMM.B64                X0, X1, #1400
# [DWARF] common/pa_trace.h:283
#     281 |             }
#     282 |         }
# >   283 |         trace.poll_burst.call_count[index] = 0;
0x000000000000f9fc (+0x0000f73c)  0f976700  STI_XN_IMM.B32                  X22, #1464
0x000000000000fa00 (+0x0000f740)  071e7fff  MOV_XD_IMM                      X15, #32767
0x000000000000fa04 (+0x0000f744)  1cd5e9b0  LD_XD_XN_IMM.B64                X10, X30, #2480
0x000000000000fa08 (+0x0000f748)  1cd7e9a8  LD_XD_XN_IMM.B64                X11, X30, #2472
0x000000000000fa0c (+0x0000f74c)  1cd9e9a0  LD_XD_XN_IMM.B64                X12, X30, #2464
0x000000000000fa10 (+0x0000f750)  1cd3e990  LD_XD_XN_IMM.B64                X9, X30, #2448
# [DWARF] common/pa_trace.h:262
#     256 | #else
#     257 |     if (!trace.atomics_enabled || trace.poll_burst.active_mask == 0) return;
#     258 |     const uint32_t active_mask = trace.poll_burst.active_mask;
#     259 |     // CCEC 默认会把固定 6-site 循环完整展开，再随几十个 phase 边界复制。
#     260 |     // 禁止展开只控制代码体积；循环次数、site 顺序和同 cycle 关闭语义不变。
#     261 |     PA_LOOP_NOUNROLL
# >   262 |     for (uint32_t index = 0; index < kAtomicPollBatchSiteCount; ++index) {
0x000000000000fa14 (+0x0000f754)  08318001  ADD_IMM.S64                     X24, X24, #1
0x000000000000fa18 (+0x0000f758)  08252004  ADD_IMM.S64                     X18, X18, #4
0x000000000000fa1c (+0x0000f75c)  0001841e  CMP.S64.NE                      X24, X8
0x000000000000fa20 (+0x0000f760)  08294001  ADD_IMM.S64                     X20, X20, #1
