# schema=pa_final_linked_disassembly/v1
# variant=compete-first
# final_elf=pa_scheduler_kernel.o
# final_elf_sha256=82a27e206ee1b2411964c0530c73adf92a2b032ac202a7c65c6b5f7d76a4571b
# final_text_address=0x0
# final_text_size=547640
# final_text_sha256=8fd6209b2f0f9d1fb48d4d8a16cf27649e26071153b908e1e0e39ca107d63589
# symbol=pa_scheduler_lazy_sample_callback_orchestration_aic
# binding=LOCAL
# final_pc=0x2c0
# size=161788
# instruction_count=40447
# encoded_word_count=40447
# body_sha256=cab368a779b3870411a46bd169f6af9657c025967797ba7743b93fcfe27f45db
# decoder=$ASCEND_HOME_PATH/x86_64-linux/simulator/dav_3510/lib/libpem_davinci.so
# decoder_sha256=29835d2439d6dd464d34a212ad4bbd5c29af6a38465da09a6c273401d9a96dcb
# decoder_mode=scalar
# columns=final_pc function_relative_offset machine_word instruction
# annotation_schema=pa_source_annotated_disassembly/v1
# annotation_rule=DWARF supplies only file:line; SOURCE rows are copied from local source files
# annotation_warning=comments have source context only and do not own an exact machine address
# annotation_instruction_slice=9241:9962
#
# [DWARF] common/pa_trace.h:266
#     260 |     // 禁止展开只控制代码体积；循环次数、site 顺序和同 cycle 关闭语义不变。
#     261 |     PA_LOOP_NOUNROLL
#     262 |     for (uint32_t index = 0; index < kAtomicPollBatchSiteCount; ++index) {
#     263 |         const uint32_t bit = 1U << index;
#     264 |         if ((active_mask & bit) == 0) continue;
#     265 |         const uint32_t call_count = trace.poll_burst.call_count[index];
# >   266 |         if (call_count == 0 || call_count > kAtomicPollCountMax) {
0x0000000000009324 (+0x00009064)  0040113e  CMP.U64.GT                      X1, X2
0x0000000000009328 (+0x00009068)  40200003  JUMPC                           #3
# [DWARF] common/pa_trace.h:267
# >   267 |             trace.atomic_counter_overflow = true;
0x000000000000932c (+0x0000906c)  0f160001  STI_XN_IMM.B8                   X0, #1408
0x0000000000009330 (+0x00009070)  4000ffdf  JUMP                            #65503
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000009334 (+0x00009074)  07320001  MOV_XD_IMM                      X25, #1
# [DWARF] common/pa_trace.h:98
#      92 |         default:
#      93 |             return -1;
#      94 |     }
#      95 | }
#      96 | 
#      97 | PA_DEVICE AtomicSite TraceAtomicPollBatchSite(uint32_t index) {
# >    98 |     switch (index) {
0x0000000000009338 (+0x00009078)  0281ca00  ZEROEXT.U32                     X0, X28
0x000000000000933c (+0x0000907c)  07020005  MOV_XD_IMM                      X1, #5
0x0000000000009340 (+0x00009080)  02339080  NEG.S64                         X25, X25
0x0000000000009344 (+0x00009084)  004000be  CMP.U64.GT                      X0, X1
0x0000000000009348 (+0x00009088)  070c000f  MOV_XD_IMM                      X6, #15
0x000000000000934c (+0x0000908c)  40200002  JUMPC                           #2
0x0000000000009350 (+0x00009090)  1c8d7000  LD_XD_XN_IMM.B32                X6, X23, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000009354 (+0x00009094)  0804c002  ADD_IMM.S64                     X2, X12, #2
# [DWARF] common/pa_trace.h:271
#     265 |         const uint32_t call_count = trace.poll_burst.call_count[index];
#     266 |         if (call_count == 0 || call_count > kAtomicPollCountMax) {
#     267 |             trace.atomic_counter_overflow = true;
#     268 |             continue;
#     269 |         }
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
# >   271 |         const bool written = WritePollBatchRecordRaw(
0x0000000000009358 (+0x00009098)  1cc9e988  LD_XD_XN_IMM.B64                X4, X30, #2440
0x000000000000935c (+0x0000909c)  00c2b78a  AND.B64                         X1, X11, X15
0x0000000000009360 (+0x000090a0)  02c4020e  SHL.B64                         X2, #14
0x0000000000009364 (+0x000090a4)  0006a381  ADD.S64                         X3, X10, X7
0x0000000000009368 (+0x000090a8)  00061104  MADD.S64                        X3, X1, X2
# [DWARF] common/pa_trace.h:273
#     272 |             trace.core, trace.records, trace.capacity,
# >   273 |             trace.poll_burst.start_cycle[index], end_cycle,
0x000000000000936c (+0x000090ac)  02018800  MOV_XD_XN.S64                   X0, X24
0x0000000000009370 (+0x000090b0)  00023101  ADD.S64                         X1, X3, X2
0x0000000000009374 (+0x000090b4)  08821680  SUB_IMM.S64                     X1, X1, #1664
0x0000000000009378 (+0x000090b8)  02c00203  SHL.B64                         X0, #3
0x000000000000937c (+0x000090bc)  08241000  ADD_IMM.S64                     X18, X1, #0
0x0000000000009380 (+0x000090c0)  00012001  ADD.S64                         X0, X18, X0
0x0000000000009384 (+0x000090c4)  1cc60588  LD_XD_XN_IMM.B64                X3, X0, #1416
# [DWARF] common/pa_trace.h:271
#     265 |         const uint32_t call_count = trace.poll_burst.call_count[index];
#     266 |         if (call_count == 0 || call_count > kAtomicPollCountMax) {
#     267 |             trace.atomic_counter_overflow = true;
#     268 |             continue;
#     269 |         }
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
# >   271 |         const bool written = WritePollBatchRecordRaw(
0x0000000000009388 (+0x000090c8)  02015800  MOV_XD_XN.S64                   X0, X21
0x000000000000938c (+0x000090cc)  0203a800  MOV_XD_XN.S64                   X1, X26
0x0000000000009390 (+0x000090d0)  0205b800  MOV_XD_XN.S64                   X2, X27
0x0000000000009394 (+0x000090d4)  070e79c7  MOV_XD_IMM                      X7, #31175
0x0000000000009398 (+0x000090d8)  074f0000  MOVK                            X7, #0, #1
0x000000000000939c (+0x000090dc)  078f0000  MOVK                            X7, #0, #2
0x00000000000093a0 (+0x000090e0)  40427000  CALL                            X7, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000093a4 (+0x000090e4)  1ccfe990  LD_XD_XN_IMM.B64                X7, X30, #2448
0x00000000000093a8 (+0x000090e8)  071c0000  MOV_XD_IMM                      X14, #0
0x00000000000093ac (+0x000090ec)  1cd9e9a0  LD_XD_XN_IMM.B64                X12, X30, #2464
0x00000000000093b0 (+0x000090f0)  071e7fff  MOV_XD_IMM                      X15, #32767
0x00000000000093b4 (+0x000090f4)  1cd7e9a8  LD_XD_XN_IMM.B64                X11, X30, #2472
# [DWARF] common/pa_trace.h:276
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
#     271 |         const bool written = WritePollBatchRecordRaw(
#     272 |             trace.core, trace.records, trace.capacity,
#     273 |             trace.poll_burst.start_cycle[index], end_cycle,
#     274 |             call_count, static_cast<uint32_t>(site)
#     275 |         );
# >   276 |         if (written) {
0x00000000000093b8 (+0x000090f8)  0000070e  CMP.S64.EQ                      X0, X14
0x00000000000093bc (+0x000090fc)  1cd5e9b0  LD_XD_XN_IMM.B64                X10, X30, #2480
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000093c0 (+0x00009100)  07220001  MOV_XD_IMM                      X17, #1
0x00000000000093c4 (+0x00009104)  07120006  MOV_XD_IMM                      X9, #6
# [DWARF] common/pa_trace.h:276
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
#     271 |         const bool written = WritePollBatchRecordRaw(
#     272 |             trace.core, trace.records, trace.capacity,
#     273 |             trace.poll_burst.start_cycle[index], end_cycle,
#     274 |             call_count, static_cast<uint32_t>(site)
#     275 |         );
# >   276 |         if (written) {
0x00000000000093c8 (+0x00009108)  4020ffb6  JUMPC                           #65462
# [DWARF] common/pa_trace.h:277
# >   277 |             if (trace.poll_batch_records == UINT64_MAX) {
0x00000000000093cc (+0x0000910c)  1cc12578  LD_XD_XN_IMM.B64                X0, X18, #1400
0x00000000000093d0 (+0x00009110)  00000c9e  CMP.S64.NE                      X0, X25
0x00000000000093d4 (+0x00009114)  4020ffa9  JUMPC                           #65449
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000093d8 (+0x00009118)  0802c002  ADD_IMM.S64                     X1, X12, #2
0x00000000000093dc (+0x0000911c)  00c0b78a  AND.B64                         X0, X11, X15
0x00000000000093e0 (+0x00009120)  02c2020e  SHL.B64                         X1, #14
0x00000000000093e4 (+0x00009124)  0004a381  ADD.S64                         X2, X10, X7
0x00000000000093e8 (+0x00009128)  00040084  MADD.S64                        X2, X0, X1
0x00000000000093ec (+0x0000912c)  00002081  ADD.S64                         X0, X2, X1
0x00000000000093f0 (+0x00009130)  08800680  SUB_IMM.S64                     X0, X0, #1664
# [DWARF] common/pa_trace.h:278
#     272 |             trace.core, trace.records, trace.capacity,
#     273 |             trace.poll_burst.start_cycle[index], end_cycle,
#     274 |             call_count, static_cast<uint32_t>(site)
#     275 |         );
#     276 |         if (written) {
#     277 |             if (trace.poll_batch_records == UINT64_MAX) {
# >   278 |                 trace.atomic_counter_overflow = true;
0x00000000000093f4 (+0x00009134)  08000000  ADD_IMM.S64                     X0, X0, #0
0x00000000000093f8 (+0x00009138)  0f160001  STI_XN_IMM.B8                   X0, #1408
0x00000000000093fc (+0x0000913c)  4000ffa9  JUMP                            #65449
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000009400 (+0x00009140)  1cd1e978  LD_XD_XN_IMM.B64                X8, X30, #2424
0x0000000000009404 (+0x00009144)  0802c002  ADD_IMM.S64                     X1, X12, #2
0x0000000000009408 (+0x00009148)  00c0b78a  AND.B64                         X0, X11, X15
0x000000000000940c (+0x0000914c)  02c2020e  SHL.B64                         X1, #14
0x0000000000009410 (+0x00009150)  0004a381  ADD.S64                         X2, X10, X7
0x0000000000009414 (+0x00009154)  00040084  MADD.S64                        X2, X0, X1
0x0000000000009418 (+0x00009158)  00002081  ADD.S64                         X0, X2, X1
0x000000000000941c (+0x0000915c)  08800680  SUB_IMM.S64                     X0, X0, #1664
# [DWARF] common/pa_trace.h:285
#     279 |             } else {
#     280 |                 ++trace.poll_batch_records;
#     281 |             }
#     282 |         }
#     283 |         trace.poll_burst.call_count[index] = 0;
#     284 |     }
# >   285 |     trace.poll_burst.active_mask = 0;
0x0000000000009420 (+0x00009160)  08000000  ADD_IMM.S64                     X0, X0, #0
0x0000000000009424 (+0x00009164)  0f960a00  STI_XN_IMM.B32                  X0, #1488
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x0000000000009428 (+0x00009168)  08353548  ADD_IMM.S64                     X26, X19, #1352
0x000000000000942c (+0x0000916c)  1cb93558  LD_XD_XN_IMM.B32                X28, X19, #1368
0x0000000000009430 (+0x00009170)  0cf5ad80  LDP_XI_XJ_XN.B64                X26, X27, X26, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000009434 (+0x00009174)  070affff  MOV_XD_IMM                      X5, #65535
0x0000000000009438 (+0x00009178)  1cc5e8f8  LD_XD_XN_IMM.B64                X2, X30, #2296
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x000000000000943c (+0x0000917c)  0001a70e  CMP.S64.EQ                      X26, X14
0x0000000000009440 (+0x00009180)  40200026  JUMPC                           #38
0x0000000000009444 (+0x00009184)  0001b70e  CMP.S64.EQ                      X27, X14
0x0000000000009448 (+0x00009188)  40200024  JUMPC                           #36
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000944c (+0x0000918c)  0283ca00  ZEROEXT.U32                     X1, X28
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x0000000000009450 (+0x00009190)  0000170e  CMP.S64.EQ                      X1, X14
0x0000000000009454 (+0x00009194)  40200021  JUMPC                           #33
# [DWARF] common/pa_trace.h:547
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
# >   547 |     const uint32_t slot = core.count;
0x0000000000009458 (+0x00009198)  1c81a000  LD_XD_XN_IMM.B32                X0, X26, #0
# [DWARF] common/pa_trace.h:548
# >   548 |     if (slot >= trace.capacity) {
0x000000000000945c (+0x0000919c)  004000ae  CMP.U64.LT                      X0, X1
0x0000000000009460 (+0x000091a0)  40200002  JUMPC                           #2
0x0000000000009464 (+0x000091a4)  40000030  JUMP                            #48
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000009468 (+0x000091a8)  0806c002  ADD_IMM.S64                     X3, X12, #2
# [DWARF] common/pa_trace.h:553
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
# >   553 |     record.start_cycle = start_cycle;
0x000000000000946c (+0x000091ac)  1cdbe928  LD_XD_XN_IMM.B64                X13, X30, #2344
0x0000000000009470 (+0x000091b0)  00c4b78a  AND.B64                         X2, X11, X15
# [DWARF] common/pa_trace.h:554
# >   554 |     record.end_cycle = end_cycle;
0x0000000000009474 (+0x000091b4)  1ccde988  LD_XD_XN_IMM.B64                X6, X30, #2440
0x0000000000009478 (+0x000091b8)  02c6020e  SHL.B64                         X3, #14
0x000000000000947c (+0x000091bc)  0008a381  ADD.S64                         X4, X10, X7
0x0000000000009480 (+0x000091c0)  00082184  MADD.S64                        X4, X2, X3
# [DWARF] common/pa_trace.h:552
#     546 |     PA_GM TraceCoreState &core = *trace.core;
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
# >   552 |     PA_GM TraceRecord &record = trace.records[slot];
0x0000000000009484 (+0x000091c4)  02020800  MOV_XD_XN.S64                   X1, X0
0x0000000000009488 (+0x000091c8)  00044181  ADD.S64                         X2, X4, X3
# [DWARF] common/pa_trace.h:555
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
# >   555 |     record.task_id = task_id;
0x000000000000948c (+0x000091cc)  1cc7e8f8  LD_XD_XN_IMM.B64                X3, X30, #2296
# [DWARF] common/pa_trace.h:552
#     546 |     PA_GM TraceCoreState &core = *trace.core;
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
# >   552 |     PA_GM TraceRecord &record = trace.records[slot];
0x0000000000009490 (+0x000091d0)  02c20206  SHL.B64                         X1, #6
0x0000000000009494 (+0x000091d4)  0003b081  ADD.S64                         X1, X27, X1
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000009498 (+0x000091d8)  08842680  SUB_IMM.S64                     X2, X2, #1664
# [DWARF] common/pa_trace.h:558
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
#     555 |     record.task_id = task_id;
#     556 |     record.function_id = function_id;
#     557 |     record.phase = static_cast<int32_t>(trace_phase);
# >   558 |     record.lane = trace.lane;
0x000000000000949c (+0x000091dc)  08042000  ADD_IMM.S64                     X2, X2, #0
# [DWARF] common/pa_trace.h:565
#     559 |     record.block_id = trace.block_id;
#     560 |     record.core_idx = trace.core_idx;
#     561 |     record.flags = flags;
#     562 |     record.auxiliary = auxiliary;
#     563 |     // count 最后更新，使其始终指向下一空槽；单写者条件下无需 reserve/commit 两阶段。
#     564 |     // 最终 FlushTraceCore 会把记录体先于该计数一并导出。
# >   565 |     core.count = slot + 1;
0x00000000000094a0 (+0x000091e0)  08000001  ADD_IMM.S64                     X0, X0, #1
# [DWARF] common/pa_trace.h:553
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
# >   553 |     record.start_cycle = start_cycle;
0x00000000000094a4 (+0x000091e4)  09da1301  STP_XI_XJ_XN.B64                X13, X6, X1, #0
# [DWARF] common/pa_trace.h:555
#     554 |     record.end_cycle = end_cycle;
# >   555 |     record.task_id = task_id;
0x00000000000094a8 (+0x000091e8)  03861010  ST_XD_XN_IMM.B32                X3, X1, #16
# [DWARF] common/pa_trace.h:556
# >   556 |     record.function_id = function_id;
0x00000000000094ac (+0x000091ec)  0706ffff  MOV_XD_IMM                      X3, #65535
0x00000000000094b0 (+0x000091f0)  0747ffff  MOVK                            X3, #65535, #1
0x00000000000094b4 (+0x000091f4)  098614a9  STP_XI_XJ_XN.B32                X3, X9, X1, #20
# [DWARF] common/pa_trace.h:559
#     557 |     record.phase = static_cast<int32_t>(trace_phase);
#     558 |     record.lane = trace.lane;
# >   559 |     record.block_id = trace.block_id;
0x00000000000094b8 (+0x000091f8)  08062564  ADD_IMM.S64                     X3, X2, #1380
# [DWARF] common/pa_trace.h:558
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
#     555 |     record.task_id = task_id;
#     556 |     record.function_id = function_id;
#     557 |     record.phase = static_cast<int32_t>(trace_phase);
# >   558 |     record.lane = trace.lane;
0x00000000000094bc (+0x000091fc)  1c882560  LD_XD_XN_IMM.B32                X4, X2, #1376
# [DWARF] common/pa_trace.h:559
# >   559 |     record.block_id = trace.block_id;
0x00000000000094c0 (+0x00009200)  0c863100  LDP_XI_XJ_XN.B32                X3, X2, X3, #0
# [DWARF] common/pa_trace.h:558
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
#     555 |     record.task_id = task_id;
#     556 |     record.function_id = function_id;
#     557 |     record.phase = static_cast<int32_t>(trace_phase);
# >   558 |     record.lane = trace.lane;
0x00000000000094c4 (+0x00009204)  098811b9  STP_XI_XJ_XN.B32                X4, X3, X1, #28
# [DWARF] common/pa_trace.h:560
#     559 |     record.block_id = trace.block_id;
# >   560 |     record.core_idx = trace.core_idx;
0x00000000000094c8 (+0x00009208)  03841024  ST_XD_XN_IMM.B32                X2, X1, #36
0x00000000000094cc (+0x0000920c)  1cc5e8f8  LD_XD_XN_IMM.B64                X2, X30, #2296
# [DWARF] common/pa_trace.h:561
# >   561 |     record.flags = flags;
0x00000000000094d0 (+0x00009210)  03dc1028  ST_XD_XN_IMM.B64                X14, X1, #40
# [DWARF] common/pa_trace.h:565
#     562 |     record.auxiliary = auxiliary;
#     563 |     // count 最后更新，使其始终指向下一空槽；单写者条件下无需 reserve/commit 两阶段。
#     564 |     // 最终 FlushTraceCore 会把记录体先于该计数一并导出。
# >   565 |     core.count = slot + 1;
0x00000000000094d4 (+0x00009214)  0381a000  ST_XD_XN_IMM.B32                X0, X26, #0
# [DWARF] common/pa_scheduler_core.h:472
#     466 |     PA_GM SchedulerState *state, PA_GM WorkerState &worker, uint32_t task_id, TaskKind kind,
#     467 |     LocalStats &stats
#     468 | ) {
#     469 |     // Claim 在四个 shard 的单调 cursor 上执行 atomicMax：同一 task 只有观察到旧值更小的竞争者获胜。
#     470 |     // Alloc 由全部 96 个 worker 竞争；QK/PV 仅 32 个 AIC，SF/UP 仅 64 个 AIV 进入真正的 atomicMax。
#     471 |     ClaimOutcome outcome{false, false, 0, -1};
# >   472 |     if (task_id >= kTaskCellCapacity) {
0x00000000000094d8 (+0x00009218)  004022be  CMP.U64.GT                      X2, X5
0x00000000000094dc (+0x0000921c)  40200002  JUMPC                           #2
0x00000000000094e0 (+0x00009220)  40000016  JUMP                            #22
# [DWARF] common/pa_scheduler_core.h:0
# [SOURCE unavailable]
0x00000000000094e4 (+0x00009224)  0802c002  ADD_IMM.S64                     X1, X12, #2
0x00000000000094e8 (+0x00009228)  00c0b78a  AND.B64                         X0, X11, X15
0x00000000000094ec (+0x0000922c)  02c2020e  SHL.B64                         X1, #14
0x00000000000094f0 (+0x00009230)  0004a381  ADD.S64                         X2, X10, X7
0x00000000000094f4 (+0x00009234)  00040084  MADD.S64                        X2, X0, X1
0x00000000000094f8 (+0x00009238)  07320000  MOV_XD_IMM                      X25, #0
0x00000000000094fc (+0x0000923c)  00002081  ADD.S64                         X0, X2, X1
0x0000000000009500 (+0x00009240)  08800680  SUB_IMM.S64                     X0, X0, #1664
# [DWARF] common/pa_scheduler_core.h:1144
#    1138 |         efdrain_begin, efdrain_end
#    1139 |     );
#    1140 | 
#    1141 |     const uint64_t claim_begin = efdrain_end;
#    1142 |     BeginSubmitPmuPhase<SubmitPmuPhase::Claim, Ops>(pmu_context);
#    1143 |     const ClaimOutcome claim = Claim<Ops>(state, worker, task_id, Kind, stats);
# >  1144 |     context.won = claim.won;
0x0000000000009504 (+0x00009244)  08000000  ADD_IMM.S64                     X0, X0, #0
0x0000000000009508 (+0x00009248)  0f060300  STI_XN_IMM.B8                   X0, #408
# [DWARF] common/pa_scheduler_core.h:1145
# >  1145 |     context.kernel_id = claim.function_id;
0x000000000000950c (+0x0000924c)  0f860282  STI_XN_IMM.B32                  X0, #404
0x0000000000009510 (+0x00009250)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000009514 (+0x00009254)  0700007c  MOV_XD_IMM                      X0, #124
0x0000000000009518 (+0x00009258)  07410000  MOVK                            X0, #0, #1
0x000000000000951c (+0x0000925c)  07810000  MOVK                            X0, #0, #2
0x0000000000009520 (+0x00009260)  40020000  JUMP                            X0, #0
# [DWARF] common/pa_trace.h:549
#     543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
# >   549 |         core.dropped = core.dropped + 1;
0x0000000000009524 (+0x00009264)  1c81a004  LD_XD_XN_IMM.B32                X0, X26, #4
0x0000000000009528 (+0x00009268)  08000001  ADD_IMM.S64                     X0, X0, #1
0x000000000000952c (+0x0000926c)  0381a004  ST_XD_XN_IMM.B32                X0, X26, #4
# [DWARF] common/pa_scheduler_core.h:472
#     466 |     PA_GM SchedulerState *state, PA_GM WorkerState &worker, uint32_t task_id, TaskKind kind,
#     467 |     LocalStats &stats
#     468 | ) {
#     469 |     // Claim 在四个 shard 的单调 cursor 上执行 atomicMax：同一 task 只有观察到旧值更小的竞争者获胜。
#     470 |     // Alloc 由全部 96 个 worker 竞争；QK/PV 仅 32 个 AIC，SF/UP 仅 64 个 AIV 进入真正的 atomicMax。
#     471 |     ClaimOutcome outcome{false, false, 0, -1};
# >   472 |     if (task_id >= kTaskCellCapacity) {
0x0000000000009530 (+0x00009270)  004022be  CMP.U64.GT                      X2, X5
0x0000000000009534 (+0x00009274)  4020ffec  JUMPC                           #65516
# [DWARF] common/pa_scheduler_core.h:477
#     473 |         return outcome;
#     474 |     }
#     475 |     PA_GM AtomicLine *cursor = nullptr;
#     476 |     if (kind == TaskKind::Alloc) {
# >   477 |         cursor = &state->alloc_cursor[task_id % kCursorShards];
0x0000000000009538 (+0x00009278)  1cc3e880  LD_XD_XN_IMM.B64                X1, X30, #2176
0x000000000000953c (+0x0000927c)  07000003  MOV_XD_IMM                      X0, #3
0x0000000000009540 (+0x00009280)  00c0200a  AND.B64                         X0, X2, X0
0x0000000000009544 (+0x00009284)  02c00206  SHL.B64                         X0, #6
# [DWARF] common/pa_trace.h:257
#     251 |     TraceContext &trace, uint64_t end_cycle
#     252 | ) {
#     253 | #if PA_BUILD_SUBMIT_PMU
#     254 |     (void)trace;
#     255 |     (void)end_cycle;
#     256 | #else
# >   257 |     if (!trace.atomics_enabled || trace.poll_burst.active_mask == 0) return;
0x0000000000009548 (+0x00009288)  0000871e  CMP.S64.NE                      X8, X14
# [DWARF] common/pa_scheduler_core.h:477
#     471 |     ClaimOutcome outcome{false, false, 0, -1};
#     472 |     if (task_id >= kTaskCellCapacity) {
#     473 |         return outcome;
#     474 |     }
#     475 |     PA_GM AtomicLine *cursor = nullptr;
#     476 |     if (kind == TaskKind::Alloc) {
# >   477 |         cursor = &state->alloc_cursor[task_id % kCursorShards];
0x000000000000954c (+0x0000928c)  00001001  ADD.S64                         X0, X1, X0
# [DWARF] common/pa_trace.h:480
#     474 |     (void)result;
#     475 |     (void)task_id;
#     476 |     (void)site;
#     477 |     (void)result_used;
#     478 |     return Ops::FetchMax(address, value, retries);
#     479 | #else
# >   480 |     if (!trace.atomics_enabled) return Ops::FetchMax(address, value, retries);
0x0000000000009550 (+0x00009290)  40200006  JUMPC                           #6
0x0000000000009554 (+0x00009294)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000009558 (+0x00009298)  07000015  MOV_XD_IMM                      X0, #21
0x000000000000955c (+0x0000929c)  07410000  MOVK                            X0, #0, #1
0x0000000000009560 (+0x000092a0)  07810000  MOVK                            X0, #0, #2
0x0000000000009564 (+0x000092a4)  40020000  JUMP                            X0, #0
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x0000000000009568 (+0x000092a8)  02828880  MOV_XD_SPR.F32                  X1, SYS_CNT
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:86
#      80 |     return __builtin_cce_atom_max_G_##SUFFIX(base, inc, (uint32_t)L2Cache);    \
#      81 |   }
#      82 | 
#      83 | __CCE__ATOM_G_BUILTIN(uint32_t, u32)
#      84 | __CCE__ATOM_G_BUILTIN(int32_t, s32)
#      85 | __CCE__ATOM_G_BUILTIN(uint64_t, u64)
# >    86 | __CCE__ATOM_G_BUILTIN(int64_t, s64)
0x000000000000956c (+0x000092ac)  51c00040  ATOM                            XN, XM, XD, MAX
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:0
# [SOURCE unavailable]
0x0000000000009570 (+0x000092b0)  0806c002  ADD_IMM.S64                     X3, X12, #2
0x0000000000009574 (+0x000092b4)  00c4b78a  AND.B64                         X2, X11, X15
0x0000000000009578 (+0x000092b8)  02c6020e  SHL.B64                         X3, #14
0x000000000000957c (+0x000092bc)  0008a381  ADD.S64                         X4, X10, X7
0x0000000000009580 (+0x000092c0)  00082184  MADD.S64                        X4, X2, X3
0x0000000000009584 (+0x000092c4)  00044181  ADD.S64                         X2, X4, X3
0x0000000000009588 (+0x000092c8)  08842680  SUB_IMM.S64                     X2, X2, #1664
# [DWARF] common/pa_trace.h:236
#     230 |            (encoded_retries << kAtomicRetriesShift);
#     231 | }
#     232 | 
#     233 | PA_DEVICE void CountAtomicCall(
#     234 |     TraceContext &trace, WorkerResult &result, bool poll_batch
#     235 | ) {
# >   236 |     if (result.atomic_trace_calls == UINT64_MAX) {
0x000000000000958c (+0x000092cc)  08062000  ADD_IMM.S64                     X3, X2, #0
# [DWARF] ccec/ccec_ops.h:239
#     233 |         static_assert(sizeof(T) == 4 || sizeof(T) == 8, "atomic dependency expects a scalar result");
#     234 |         uint64_t cycle = 0;
#     235 |         // 同一个 inline asm 块先真正消费 atomic 返回寄存器，再读取
#     236 |         // SYS_CNT；编译器不能把 t1 拆到依赖 MOV 之前。AIC/AIV 对该序列
#     237 |         // 生成相同指令字节，且不增加 DSB/ISB/GM 访存。该边界仍只表示
#     238 |         // 返回值已可被本核 scalar 消费，不表示跨核全局可见。
# >   239 |         asm volatile(
0x0000000000009590 (+0x000092d0)  020a0800  MOV_XD_XN.S64                   X5, X0
0x0000000000009594 (+0x000092d4)  020a5800  MOV_XD_XN.S64                   X5, X5
0x0000000000009598 (+0x000092d8)  02848880  MOV_XD_SPR.F32                  X2, SYS_CNT
# [DWARF] common/pa_trace.h:236
#     230 |            (encoded_retries << kAtomicRetriesShift);
#     231 | }
#     232 | 
#     233 | PA_DEVICE void CountAtomicCall(
#     234 |     TraceContext &trace, WorkerResult &result, bool poll_batch
#     235 | ) {
# >   236 |     if (result.atomic_trace_calls == UINT64_MAX) {
0x000000000000959c (+0x000092dc)  1cc834a0  LD_XD_XN_IMM.B64                X4, X3, #1184
0x00000000000095a0 (+0x000092e0)  0000491e  CMP.S64.NE                      X4, X18
0x00000000000095a4 (+0x000092e4)  40200002  JUMPC                           #2
0x00000000000095a8 (+0x000092e8)  40000007  JUMP                            #7
# [DWARF] common/pa_trace.h:240
#     237 |         trace.atomic_counter_overflow = true;
#     238 |         return;
#     239 |     }
# >   240 |     ++result.atomic_trace_calls;
0x00000000000095ac (+0x000092ec)  08084001  ADD_IMM.S64                     X4, X4, #1
0x00000000000095b0 (+0x000092f0)  03c834a0  ST_XD_XN_IMM.B64                X4, X3, #1184
0x00000000000095b4 (+0x000092f4)  40000005  JUMP                            #5
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000095b8 (+0x000092f8)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:86
#      80 |     return __builtin_cce_atom_max_G_##SUFFIX(base, inc, (uint32_t)L2Cache);    \
#      81 |   }
#      82 | 
#      83 | __CCE__ATOM_G_BUILTIN(uint32_t, u32)
#      84 | __CCE__ATOM_G_BUILTIN(int32_t, s32)
#      85 | __CCE__ATOM_G_BUILTIN(uint64_t, u64)
# >    86 | __CCE__ATOM_G_BUILTIN(int64_t, s64)
0x00000000000095bc (+0x000092fc)  51c00040  ATOM                            XN, XM, XD, MAX
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:0
# [SOURCE unavailable]
0x00000000000095c0 (+0x00009300)  40000039  JUMP                            #57
# [DWARF] common/pa_trace.h:237
#     231 | }
#     232 | 
#     233 | PA_DEVICE void CountAtomicCall(
#     234 |     TraceContext &trace, WorkerResult &result, bool poll_batch
#     235 | ) {
#     236 |     if (result.atomic_trace_calls == UINT64_MAX) {
# >   237 |         trace.atomic_counter_overflow = true;
0x00000000000095c4 (+0x00009304)  0f163001  STI_XN_IMM.B8                   X3, #1408
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000095c8 (+0x00009308)  0808c002  ADD_IMM.S64                     X4, X12, #2
0x00000000000095cc (+0x0000930c)  00c6b78a  AND.B64                         X3, X11, X15
0x00000000000095d0 (+0x00009310)  02c8020e  SHL.B64                         X4, #14
0x00000000000095d4 (+0x00009314)  000aa381  ADD.S64                         X5, X10, X7
0x00000000000095d8 (+0x00009318)  000a3204  MADD.S64                        X5, X3, X4
0x00000000000095dc (+0x0000931c)  00065201  ADD.S64                         X3, X5, X4
0x00000000000095e0 (+0x00009320)  08863680  SUB_IMM.S64                     X3, X3, #1664
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x00000000000095e4 (+0x00009324)  08063000  ADD_IMM.S64                     X3, X3, #0
0x00000000000095e8 (+0x00009328)  08343548  ADD_IMM.S64                     X26, X3, #1352
0x00000000000095ec (+0x0000932c)  1cb83558  LD_XD_XN_IMM.B32                X28, X3, #1368
0x00000000000095f0 (+0x00009330)  0cf5ad80  LDP_XI_XJ_XN.B64                X26, X27, X26, #0
0x00000000000095f4 (+0x00009334)  0001a70e  CMP.S64.EQ                      X26, X14
0x00000000000095f8 (+0x00009338)  4020002b  JUMPC                           #43
0x00000000000095fc (+0x0000933c)  0001b70e  CMP.S64.EQ                      X27, X14
0x0000000000009600 (+0x00009340)  40200029  JUMPC                           #41
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000009604 (+0x00009344)  0289ca00  ZEROEXT.U32                     X4, X28
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x0000000000009608 (+0x00009348)  0000470e  CMP.S64.EQ                      X4, X14
0x000000000000960c (+0x0000934c)  40200026  JUMPC                           #38
# [DWARF] common/pa_trace.h:547
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
# >   547 |     const uint32_t slot = core.count;
0x0000000000009610 (+0x00009350)  1c87a000  LD_XD_XN_IMM.B32                X3, X26, #0
# [DWARF] common/pa_trace.h:548
# >   548 |     if (slot >= trace.capacity) {
0x0000000000009614 (+0x00009354)  0040322e  CMP.U64.LT                      X3, X4
0x0000000000009618 (+0x00009358)  40200002  JUMPC                           #2
0x000000000000961c (+0x0000935c)  4000001f  JUMP                            #31
# [DWARF] common/pa_trace.h:552
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
# >   552 |     PA_GM TraceRecord &record = trace.records[slot];
0x0000000000009620 (+0x00009360)  02083800  MOV_XD_XN.S64                   X4, X3
0x0000000000009624 (+0x00009364)  02c80206  SHL.B64                         X4, #6
0x0000000000009628 (+0x00009368)  0009b201  ADD.S64                         X4, X27, X4
# [DWARF] common/pa_trace.h:553
# >   553 |     record.start_cycle = start_cycle;
0x000000000000962c (+0x0000936c)  09c24101  STP_XI_XJ_XN.B64                X1, X2, X4, #0
# [DWARF] common/pa_trace.h:555
#     554 |     record.end_cycle = end_cycle;
# >   555 |     record.task_id = task_id;
0x0000000000009630 (+0x00009370)  1cc5e8f8  LD_XD_XN_IMM.B64                X2, X30, #2296
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000009634 (+0x00009374)  080cc002  ADD_IMM.S64                     X6, X12, #2
0x0000000000009638 (+0x00009378)  00cab78a  AND.B64                         X5, X11, X15
0x000000000000963c (+0x0000937c)  02cc020e  SHL.B64                         X6, #14
0x0000000000009640 (+0x00009380)  0002a381  ADD.S64                         X1, X10, X7
0x0000000000009644 (+0x00009384)  00025304  MADD.S64                        X1, X5, X6
# [DWARF] common/pa_trace.h:557
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
#     555 |     record.task_id = task_id;
#     556 |     record.function_id = function_id;
# >   557 |     record.phase = static_cast<int32_t>(trace_phase);
0x0000000000009648 (+0x00009388)  070a000e  MOV_XD_IMM                      X5, #14
0x000000000000964c (+0x0000938c)  00021301  ADD.S64                         X1, X1, X6
0x0000000000009650 (+0x00009390)  08821680  SUB_IMM.S64                     X1, X1, #1664
# [DWARF] common/pa_trace.h:558
# >   558 |     record.lane = trace.lane;
0x0000000000009654 (+0x00009394)  08021000  ADD_IMM.S64                     X1, X1, #0
# [DWARF] common/pa_trace.h:555
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
# >   555 |     record.task_id = task_id;
0x0000000000009658 (+0x00009398)  03844010  ST_XD_XN_IMM.B32                X2, X4, #16
# [DWARF] common/pa_trace.h:556
# >   556 |     record.function_id = function_id;
0x000000000000965c (+0x0000939c)  0704ffff  MOV_XD_IMM                      X2, #65535
0x0000000000009660 (+0x000093a0)  0745ffff  MOVK                            X2, #65535, #1
0x0000000000009664 (+0x000093a4)  03844014  ST_XD_XN_IMM.B32                X2, X4, #20
# [DWARF] common/pa_trace.h:558
#     557 |     record.phase = static_cast<int32_t>(trace_phase);
# >   558 |     record.lane = trace.lane;
0x0000000000009668 (+0x000093a8)  1c841560  LD_XD_XN_IMM.B32                X2, X1, #1376
# [DWARF] common/pa_trace.h:557
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
#     555 |     record.task_id = task_id;
#     556 |     record.function_id = function_id;
# >   557 |     record.phase = static_cast<int32_t>(trace_phase);
0x000000000000966c (+0x000093ac)  098a4131  STP_XI_XJ_XN.B32                X5, X2, X4, #24
# [DWARF] common/pa_trace.h:559
#     558 |     record.lane = trace.lane;
# >   559 |     record.block_id = trace.block_id;
0x0000000000009670 (+0x000093b0)  08041564  ADD_IMM.S64                     X2, X1, #1380
0x0000000000009674 (+0x000093b4)  0c842080  LDP_XI_XJ_XN.B32                X2, X1, X2, #0
0x0000000000009678 (+0x000093b8)  080a4020  ADD_IMM.S64                     X5, X4, #32
0x000000000000967c (+0x000093bc)  09845081  STP_XI_XJ_XN.B32                X2, X1, X5, #0
# [DWARF] common/pa_trace.h:561
#     560 |     record.core_idx = trace.core_idx;
# >   561 |     record.flags = flags;
0x0000000000009680 (+0x000093c0)  07020053  MOV_XD_IMM                      X1, #83
0x0000000000009684 (+0x000093c4)  07830004  MOVK                            X1, #4, #2
0x0000000000009688 (+0x000093c8)  03c24028  ST_XD_XN_IMM.B64                X1, X4, #40
# [DWARF] common/pa_trace.h:565
#     562 |     record.auxiliary = auxiliary;
#     563 |     // count 最后更新，使其始终指向下一空槽；单写者条件下无需 reserve/commit 两阶段。
#     564 |     // 最终 FlushTraceCore 会把记录体先于该计数一并导出。
# >   565 |     core.count = slot + 1;
0x000000000000968c (+0x000093cc)  08023001  ADD_IMM.S64                     X1, X3, #1
0x0000000000009690 (+0x000093d0)  0383a000  ST_XD_XN_IMM.B32                X1, X26, #0
0x0000000000009694 (+0x000093d4)  40000004  JUMP                            #4
# [DWARF] common/pa_trace.h:549
#     543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
# >   549 |         core.dropped = core.dropped + 1;
0x0000000000009698 (+0x000093d8)  1c83a004  LD_XD_XN_IMM.B32                X1, X26, #4
0x000000000000969c (+0x000093dc)  08021001  ADD_IMM.S64                     X1, X1, #1
0x00000000000096a0 (+0x000093e0)  0383a004  ST_XD_XN_IMM.B32                X1, X26, #4
# [DWARF] common/pa_scheduler_core.h:514
#     508 |     outcome.attempted = true;
#     509 |     // atomicMax 返回写入前的 cursor：old<task_id 表示本核完成首次推进并获胜，old>=task_id 则必须 Replay。
#     510 |     const int64_t old = TraceAtomicFetchMax<Ops>(
#     511 |         stats.trace, stats.result, static_cast<int32_t>(task_id), AtomicSite::ClaimMax,
#     512 |         &cursor->value, static_cast<int64_t>(task_id), outcome.retries
#     513 |     );
# >   514 |     outcome.won = old < static_cast<int64_t>(task_id);
0x00000000000096a4 (+0x000093e4)  1cc9e8f8  LD_XD_XN_IMM.B64                X4, X30, #2296
0x00000000000096a8 (+0x000093e8)  0804c002  ADD_IMM.S64                     X2, X12, #2
0x00000000000096ac (+0x000093ec)  00c2b78a  AND.B64                         X1, X11, X15
0x00000000000096b0 (+0x000093f0)  02c4020e  SHL.B64                         X2, #14
0x00000000000096b4 (+0x000093f4)  0006a381  ADD.S64                         X3, X10, X7
0x00000000000096b8 (+0x000093f8)  00061104  MADD.S64                        X3, X1, X2
0x00000000000096bc (+0x000093fc)  07320000  MOV_XD_IMM                      X25, #0
0x00000000000096c0 (+0x00009400)  0000022e  CMP.S64.LT                      X0, X4
# [DWARF] common/pa_scheduler_core.h:0
# [SOURCE unavailable]
0x00000000000096c4 (+0x00009404)  00003101  ADD.S64                         X0, X3, X2
0x00000000000096c8 (+0x00009408)  08800680  SUB_IMM.S64                     X0, X0, #1664
# [DWARF] common/pa_scheduler_core.h:1144
#    1138 |         efdrain_begin, efdrain_end
#    1139 |     );
#    1140 | 
#    1141 |     const uint64_t claim_begin = efdrain_end;
#    1142 |     BeginSubmitPmuPhase<SubmitPmuPhase::Claim, Ops>(pmu_context);
#    1143 |     const ClaimOutcome claim = Claim<Ops>(state, worker, task_id, Kind, stats);
# >  1144 |     context.won = claim.won;
0x00000000000096cc (+0x0000940c)  08000000  ADD_IMM.S64                     X0, X0, #0
# [DWARF] common/pa_scheduler_core.h:514
#     508 |     outcome.attempted = true;
#     509 |     // atomicMax 返回写入前的 cursor：old<task_id 表示本核完成首次推进并获胜，old>=task_id 则必须 Replay。
#     510 |     const int64_t old = TraceAtomicFetchMax<Ops>(
#     511 |         stats.trace, stats.result, static_cast<int32_t>(task_id), AtomicSite::ClaimMax,
#     512 |         &cursor->value, static_cast<int64_t>(task_id), outcome.retries
#     513 |     );
# >   514 |     outcome.won = old < static_cast<int64_t>(task_id);
0x00000000000096d0 (+0x00009410)  0202b880  MOV_XD_SPR.S64                  X1, CONDITION_FLAG
# [DWARF] common/pa_scheduler_core.h:1144
#    1138 |         efdrain_begin, efdrain_end
#    1139 |     );
#    1140 | 
#    1141 |     const uint64_t claim_begin = efdrain_end;
#    1142 |     BeginSubmitPmuPhase<SubmitPmuPhase::Claim, Ops>(pmu_context);
#    1143 |     const ClaimOutcome claim = Claim<Ops>(state, worker, task_id, Kind, stats);
# >  1144 |     context.won = claim.won;
0x00000000000096d4 (+0x00009414)  03020198  ST_XD_XN_IMM.B8                 X1, X0, #408
# [DWARF] common/pa_scheduler_core.h:1145
# >  1145 |     context.kernel_id = claim.function_id;
0x00000000000096d8 (+0x00009418)  0f860282  STI_XN_IMM.B32                  X0, #404
# [DWARF] common/pa_scheduler_core.h:520
#     514 |     outcome.won = old < static_cast<int64_t>(task_id);
#     515 |     if (!outcome.won) outcome.function_id = -1;
#     516 |     return outcome;
#     517 | }
#     518 | 
#     519 | PA_DEVICE void RecordClaimOutcome(LocalStats &stats, TaskKind kind, const ClaimOutcome &outcome) {
# >   520 |     if (outcome.attempted) ++stats.result.claim_attempts;
0x00000000000096dc (+0x0000941c)  1cc201e8  LD_XD_XN_IMM.B64                X1, X0, #488
0x00000000000096e0 (+0x00009420)  08021001  ADD_IMM.S64                     X1, X1, #1
0x00000000000096e4 (+0x00009424)  03c201e8  ST_XD_XN_IMM.B64                X1, X0, #488
# [DWARF] common/pa_scheduler_core.h:522
#     521 |     stats.result.cas_retries += outcome.retries;
# >   522 |     if (outcome.won) {
0x00000000000096e8 (+0x00009428)  40200002  JUMPC                           #2
0x00000000000096ec (+0x0000942c)  4000000a  JUMP                            #10
# [DWARF] common/pa_scheduler_core.h:523
# >   523 |         ++stats.result.claim_wins;
0x00000000000096f0 (+0x00009430)  1cc201f0  LD_XD_XN_IMM.B64                X1, X0, #496
0x00000000000096f4 (+0x00009434)  07320001  MOV_XD_IMM                      X25, #1
# [DWARF] common/pa_scheduler_core.h:524
# >   524 |         ++stats.result.wins[KindIndex(kind)];
0x00000000000096f8 (+0x00009438)  1cc40220  LD_XD_XN_IMM.B64                X2, X0, #544
# [DWARF] common/pa_scheduler_core.h:523
#     517 | }
#     518 | 
#     519 | PA_DEVICE void RecordClaimOutcome(LocalStats &stats, TaskKind kind, const ClaimOutcome &outcome) {
#     520 |     if (outcome.attempted) ++stats.result.claim_attempts;
#     521 |     stats.result.cas_retries += outcome.retries;
#     522 |     if (outcome.won) {
# >   523 |         ++stats.result.claim_wins;
0x00000000000096fc (+0x0000943c)  08021001  ADD_IMM.S64                     X1, X1, #1
0x0000000000009700 (+0x00009440)  03c201f0  ST_XD_XN_IMM.B64                X1, X0, #496
# [DWARF] common/pa_scheduler_core.h:524
# >   524 |         ++stats.result.wins[KindIndex(kind)];
0x0000000000009704 (+0x00009444)  08022001  ADD_IMM.S64                     X1, X2, #1
0x0000000000009708 (+0x00009448)  03c20220  ST_XD_XN_IMM.B64                X1, X0, #544
0x000000000000970c (+0x0000944c)  40000002  JUMP                            #2
# [DWARF] common/pa_scheduler_core.h:0
# [SOURCE unavailable]
0x0000000000009710 (+0x00009450)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000009714 (+0x00009454)  0802c002  ADD_IMM.S64                     X1, X12, #2
0x0000000000009718 (+0x00009458)  00c0b78a  AND.B64                         X0, X11, X15
0x000000000000971c (+0x0000945c)  02c2020e  SHL.B64                         X1, #14
0x0000000000009720 (+0x00009460)  0004a381  ADD.S64                         X2, X10, X7
0x0000000000009724 (+0x00009464)  00040084  MADD.S64                        X2, X0, X1
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x0000000000009728 (+0x00009468)  02aa8880  MOV_XD_SPR.F32                  X21, SYS_CNT
# [DWARF] ccec/ccec_ops.h:0
# [SOURCE unavailable]
0x000000000000972c (+0x0000946c)  00002081  ADD.S64                         X0, X2, X1
0x0000000000009730 (+0x00009470)  08800680  SUB_IMM.S64                     X0, X0, #1664
# [DWARF] common/pa_trace.h:257
#     251 |     TraceContext &trace, uint64_t end_cycle
#     252 | ) {
#     253 | #if PA_BUILD_SUBMIT_PMU
#     254 |     (void)trace;
#     255 |     (void)end_cycle;
#     256 | #else
# >   257 |     if (!trace.atomics_enabled || trace.poll_burst.active_mask == 0) return;
0x0000000000009734 (+0x00009474)  08000000  ADD_IMM.S64                     X0, X0, #0
0x0000000000009738 (+0x00009478)  1c02055c  LD_XD_XN_IMM.B8                 X1, X0, #1372
0x000000000000973c (+0x0000947c)  0000170e  CMP.S64.EQ                      X1, X14
0x0000000000009740 (+0x00009480)  40200077  JUMPC                           #119
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000009744 (+0x00009484)  1ca605d0  LD_XD_XN_IMM.B32                X19, X0, #1488
0x0000000000009748 (+0x00009488)  02813a00  ZEROEXT.U32                     X0, X19
# [DWARF] common/pa_trace.h:257
#     251 |     TraceContext &trace, uint64_t end_cycle
#     252 | ) {
#     253 | #if PA_BUILD_SUBMIT_PMU
#     254 |     (void)trace;
#     255 |     (void)end_cycle;
#     256 | #else
# >   257 |     if (!trace.atomics_enabled || trace.poll_burst.active_mask == 0) return;
0x000000000000974c (+0x0000948c)  0000070e  CMP.S64.EQ                      X0, X14
0x0000000000009750 (+0x00009490)  40200073  JUMPC                           #115
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000009754 (+0x00009494)  072e0000  MOV_XD_IMM                      X23, #0
0x0000000000009758 (+0x00009498)  07300000  MOV_XD_IMM                      X24, #0
0x000000000000975c (+0x0000949c)  0728c43c  MOV_XD_IMM                      X20, #50236
0x0000000000009760 (+0x000094a0)  07690007  MOVK                            X20, #7, #1
0x0000000000009764 (+0x000094a4)  07a90000  MOVK                            X20, #0, #2
0x0000000000009768 (+0x000094a8)  07e90000  MOVK                            X20, #0, #3
0x000000000000976c (+0x000094ac)  02020880  MOV_XD_SPR.S64                  X1, PC
0x0000000000009770 (+0x000094b0)  00294081  ADD.S64                         X20, X20, X1
0x0000000000009774 (+0x000094b4)  40000012  JUMP                            #18
0x0000000000009778 (+0x000094b8)  0804c002  ADD_IMM.S64                     X2, X12, #2
0x000000000000977c (+0x000094bc)  00c2b78a  AND.B64                         X1, X11, X15
0x0000000000009780 (+0x000094c0)  02c4020e  SHL.B64                         X2, #14
0x0000000000009784 (+0x000094c4)  0006a381  ADD.S64                         X3, X10, X7
0x0000000000009788 (+0x000094c8)  00061104  MADD.S64                        X3, X1, X2
# [DWARF] common/pa_trace.h:280
#     274 |             call_count, static_cast<uint32_t>(site)
#     275 |         );
#     276 |         if (written) {
#     277 |             if (trace.poll_batch_records == UINT64_MAX) {
#     278 |                 trace.atomic_counter_overflow = true;
#     279 |             } else {
# >   280 |                 ++trace.poll_batch_records;
0x000000000000978c (+0x000094cc)  08000001  ADD_IMM.S64                     X0, X0, #1
0x0000000000009790 (+0x000094d0)  00023101  ADD.S64                         X1, X3, X2
0x0000000000009794 (+0x000094d4)  08821680  SUB_IMM.S64                     X1, X1, #1664
0x0000000000009798 (+0x000094d8)  08021000  ADD_IMM.S64                     X1, X1, #0
0x000000000000979c (+0x000094dc)  03c01578  ST_XD_XN_IMM.B64                X0, X1, #1400
# [DWARF] common/pa_trace.h:283
#     281 |             }
#     282 |         }
# >   283 |         trace.poll_burst.call_count[index] = 0;
0x00000000000097a0 (+0x000094e0)  0f976700  STI_XN_IMM.B32                  X22, #1464
# [DWARF] common/pa_trace.h:262
#     256 | #else
#     257 |     if (!trace.atomics_enabled || trace.poll_burst.active_mask == 0) return;
#     258 |     const uint32_t active_mask = trace.poll_burst.active_mask;
#     259 |     // CCEC 默认会把固定 6-site 循环完整展开，再随几十个 phase 边界复制。
#     260 |     // 禁止展开只控制代码体积；循环次数、site 顺序和同 cycle 关闭语义不变。
#     261 |     PA_LOOP_NOUNROLL
# >   262 |     for (uint32_t index = 0; index < kAtomicPollBatchSiteCount; ++index) {
0x00000000000097a4 (+0x000094e4)  08318001  ADD_IMM.S64                     X24, X24, #1
0x00000000000097a8 (+0x000094e8)  08294004  ADD_IMM.S64                     X20, X20, #4
0x00000000000097ac (+0x000094ec)  0001849e  CMP.S64.NE                      X24, X9
0x00000000000097b0 (+0x000094f0)  082f7001  ADD_IMM.S64                     X23, X23, #1
0x00000000000097b4 (+0x000094f4)  40200002  JUMPC                           #2
0x00000000000097b8 (+0x000094f8)  40000050  JUMP                            #80
# [DWARF] common/pa_trace.h:263
# >   263 |         const uint32_t bit = 1U << index;
0x00000000000097bc (+0x000094fc)  02817a00  ZEROEXT.U32                     X0, X23
# [DWARF] common/pa_trace.h:264
# >   264 |         if ((active_mask & bit) == 0) continue;
0x00000000000097c0 (+0x00009500)  02833a00  ZEROEXT.U32                     X1, X19
0x00000000000097c4 (+0x00009504)  024202c0  SHR.U64                         X1, X0, #0
0x00000000000097c8 (+0x00009508)  00c0188a  AND.B64                         X0, X1, X17
0x00000000000097cc (+0x0000950c)  0000070e  CMP.S64.EQ                      X0, X14
0x00000000000097d0 (+0x00009510)  4020fff5  JUMPC                           #65525
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000097d4 (+0x00009514)  0804c002  ADD_IMM.S64                     X2, X12, #2
0x00000000000097d8 (+0x00009518)  00c0b78a  AND.B64                         X0, X11, X15
0x00000000000097dc (+0x0000951c)  02c4020e  SHL.B64                         X2, #14
0x00000000000097e0 (+0x00009520)  0006a381  ADD.S64                         X3, X10, X7
0x00000000000097e4 (+0x00009524)  00060104  MADD.S64                        X3, X0, X2
# [DWARF] common/pa_trace.h:265
#     259 |     // CCEC 默认会把固定 6-site 循环完整展开，再随几十个 phase 边界复制。
#     260 |     // 禁止展开只控制代码体积；循环次数、site 顺序和同 cycle 关闭语义不变。
#     261 |     PA_LOOP_NOUNROLL
#     262 |     for (uint32_t index = 0; index < kAtomicPollBatchSiteCount; ++index) {
#     263 |         const uint32_t bit = 1U << index;
#     264 |         if ((active_mask & bit) == 0) continue;
# >   265 |         const uint32_t call_count = trace.poll_burst.call_count[index];
0x00000000000097e8 (+0x00009528)  02038800  MOV_XD_XN.S64                   X1, X24
0x00000000000097ec (+0x0000952c)  00003101  ADD.S64                         X0, X3, X2
0x00000000000097f0 (+0x00009530)  08800680  SUB_IMM.S64                     X0, X0, #1664
0x00000000000097f4 (+0x00009534)  02c20202  SHL.B64                         X1, #2
0x00000000000097f8 (+0x00009538)  08000000  ADD_IMM.S64                     X0, X0, #0
0x00000000000097fc (+0x0000953c)  002c0081  ADD.S64                         X22, X0, X1
0x0000000000009800 (+0x00009540)  1c8b65b8  LD_XD_XN_IMM.B32                X5, X22, #1464
# [DWARF] common/pa_trace.h:266
# >   266 |         if (call_count == 0 || call_count > kAtomicPollCountMax) {
0x0000000000009804 (+0x00009544)  07020000  MOV_XD_IMM                      X1, #0
0x0000000000009808 (+0x00009548)  07430100  MOVK                            X1, #256, #1
0x000000000000980c (+0x0000954c)  07040000  MOV_XD_IMM                      X2, #0
0x0000000000009810 (+0x00009550)  0745ff00  MOVK                            X2, #65280, #1
0x0000000000009814 (+0x00009554)  00025082  SUB.S64                         X1, X5, X1
0x0000000000009818 (+0x00009558)  02821a00  ZEROEXT.U32                     X1, X1
0x000000000000981c (+0x0000955c)  0040113e  CMP.U64.GT                      X1, X2
0x0000000000009820 (+0x00009560)  40200003  JUMPC                           #3
# [DWARF] common/pa_trace.h:267
# >   267 |             trace.atomic_counter_overflow = true;
0x0000000000009824 (+0x00009564)  0f160001  STI_XN_IMM.B8                   X0, #1408
0x0000000000009828 (+0x00009568)  4000ffdf  JUMP                            #65503
# [DWARF] common/pa_trace.h:98
#      92 |         default:
#      93 |             return -1;
#      94 |     }
#      95 | }
#      96 | 
#      97 | PA_DEVICE AtomicSite TraceAtomicPollBatchSite(uint32_t index) {
# >    98 |     switch (index) {
0x000000000000982c (+0x0000956c)  02817a00  ZEROEXT.U32                     X0, X23
0x0000000000009830 (+0x00009570)  07020005  MOV_XD_IMM                      X1, #5
0x0000000000009834 (+0x00009574)  004000be  CMP.U64.GT                      X0, X1
0x0000000000009838 (+0x00009578)  070c000f  MOV_XD_IMM                      X6, #15
0x000000000000983c (+0x0000957c)  40200002  JUMPC                           #2
0x0000000000009840 (+0x00009580)  1c8d4000  LD_XD_XN_IMM.B32                X6, X20, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000009844 (+0x00009584)  0804c002  ADD_IMM.S64                     X2, X12, #2
0x0000000000009848 (+0x00009588)  00c2b78a  AND.B64                         X1, X11, X15
0x000000000000984c (+0x0000958c)  02c4020e  SHL.B64                         X2, #14
0x0000000000009850 (+0x00009590)  0006a381  ADD.S64                         X3, X10, X7
0x0000000000009854 (+0x00009594)  00061104  MADD.S64                        X3, X1, X2
# [DWARF] common/pa_trace.h:273
#     267 |             trace.atomic_counter_overflow = true;
#     268 |             continue;
#     269 |         }
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
#     271 |         const bool written = WritePollBatchRecordRaw(
#     272 |             trace.core, trace.records, trace.capacity,
# >   273 |             trace.poll_burst.start_cycle[index], end_cycle,
0x0000000000009858 (+0x00009598)  02018800  MOV_XD_XN.S64                   X0, X24
0x000000000000985c (+0x0000959c)  00023101  ADD.S64                         X1, X3, X2
0x0000000000009860 (+0x000095a0)  08821680  SUB_IMM.S64                     X1, X1, #1664
0x0000000000009864 (+0x000095a4)  02c00203  SHL.B64                         X0, #3
0x0000000000009868 (+0x000095a8)  08241000  ADD_IMM.S64                     X18, X1, #0
0x000000000000986c (+0x000095ac)  00012001  ADD.S64                         X0, X18, X0
0x0000000000009870 (+0x000095b0)  1cc60588  LD_XD_XN_IMM.B64                X3, X0, #1416
# [DWARF] common/pa_trace.h:271
#     265 |         const uint32_t call_count = trace.poll_burst.call_count[index];
#     266 |         if (call_count == 0 || call_count > kAtomicPollCountMax) {
#     267 |             trace.atomic_counter_overflow = true;
#     268 |             continue;
#     269 |         }
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
# >   271 |         const bool written = WritePollBatchRecordRaw(
0x0000000000009874 (+0x000095b4)  0201a800  MOV_XD_XN.S64                   X0, X26
0x0000000000009878 (+0x000095b8)  0203b800  MOV_XD_XN.S64                   X1, X27
0x000000000000987c (+0x000095bc)  0205c800  MOV_XD_XN.S64                   X2, X28
0x0000000000009880 (+0x000095c0)  02095800  MOV_XD_XN.S64                   X4, X21
0x0000000000009884 (+0x000095c4)  070e788b  MOV_XD_IMM                      X7, #30859
0x0000000000009888 (+0x000095c8)  074f0000  MOVK                            X7, #0, #1
0x000000000000988c (+0x000095cc)  078f0000  MOVK                            X7, #0, #2
0x0000000000009890 (+0x000095d0)  40427000  CALL                            X7, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000009894 (+0x000095d4)  1ccfe990  LD_XD_XN_IMM.B64                X7, X30, #2448
0x0000000000009898 (+0x000095d8)  071c0000  MOV_XD_IMM                      X14, #0
0x000000000000989c (+0x000095dc)  1cd9e9a0  LD_XD_XN_IMM.B64                X12, X30, #2464
0x00000000000098a0 (+0x000095e0)  071e7fff  MOV_XD_IMM                      X15, #32767
0x00000000000098a4 (+0x000095e4)  1cd7e9a8  LD_XD_XN_IMM.B64                X11, X30, #2472
# [DWARF] common/pa_trace.h:276
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
#     271 |         const bool written = WritePollBatchRecordRaw(
#     272 |             trace.core, trace.records, trace.capacity,
#     273 |             trace.poll_burst.start_cycle[index], end_cycle,
#     274 |             call_count, static_cast<uint32_t>(site)
#     275 |         );
# >   276 |         if (written) {
0x00000000000098a8 (+0x000095e8)  0000070e  CMP.S64.EQ                      X0, X14
0x00000000000098ac (+0x000095ec)  1cd5e9b0  LD_XD_XN_IMM.B64                X10, X30, #2480
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000098b0 (+0x000095f0)  07220001  MOV_XD_IMM                      X17, #1
0x00000000000098b4 (+0x000095f4)  07120006  MOV_XD_IMM                      X9, #6
# [DWARF] common/pa_trace.h:276
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
#     271 |         const bool written = WritePollBatchRecordRaw(
#     272 |             trace.core, trace.records, trace.capacity,
#     273 |             trace.poll_burst.start_cycle[index], end_cycle,
#     274 |             call_count, static_cast<uint32_t>(site)
#     275 |         );
# >   276 |         if (written) {
0x00000000000098b8 (+0x000095f8)  4020ffba  JUMPC                           #65466
# [DWARF] common/pa_trace.h:277
# >   277 |             if (trace.poll_batch_records == UINT64_MAX) {
0x00000000000098bc (+0x000095fc)  1cc12578  LD_XD_XN_IMM.B64                X0, X18, #1400
0x00000000000098c0 (+0x00009600)  07020001  MOV_XD_IMM                      X1, #1
0x00000000000098c4 (+0x00009604)  02021080  NEG.S64                         X1, X1
0x00000000000098c8 (+0x00009608)  0000009e  CMP.S64.NE                      X0, X1
0x00000000000098cc (+0x0000960c)  4020ffab  JUMPC                           #65451
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000098d0 (+0x00009610)  0802c002  ADD_IMM.S64                     X1, X12, #2
0x00000000000098d4 (+0x00009614)  00c0b78a  AND.B64                         X0, X11, X15
0x00000000000098d8 (+0x00009618)  02c2020e  SHL.B64                         X1, #14
0x00000000000098dc (+0x0000961c)  0004a381  ADD.S64                         X2, X10, X7
0x00000000000098e0 (+0x00009620)  00040084  MADD.S64                        X2, X0, X1
0x00000000000098e4 (+0x00009624)  00002081  ADD.S64                         X0, X2, X1
0x00000000000098e8 (+0x00009628)  08800680  SUB_IMM.S64                     X0, X0, #1664
# [DWARF] common/pa_trace.h:278
#     272 |             trace.core, trace.records, trace.capacity,
#     273 |             trace.poll_burst.start_cycle[index], end_cycle,
#     274 |             call_count, static_cast<uint32_t>(site)
#     275 |         );
#     276 |         if (written) {
#     277 |             if (trace.poll_batch_records == UINT64_MAX) {
# >   278 |                 trace.atomic_counter_overflow = true;
0x00000000000098ec (+0x0000962c)  08000000  ADD_IMM.S64                     X0, X0, #0
0x00000000000098f0 (+0x00009630)  0f160001  STI_XN_IMM.B8                   X0, #1408
0x00000000000098f4 (+0x00009634)  4000ffab  JUMP                            #65451
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000098f8 (+0x00009638)  0802c002  ADD_IMM.S64                     X1, X12, #2
0x00000000000098fc (+0x0000963c)  00c0b78a  AND.B64                         X0, X11, X15
0x0000000000009900 (+0x00009640)  02c2020e  SHL.B64                         X1, #14
0x0000000000009904 (+0x00009644)  0004a381  ADD.S64                         X2, X10, X7
0x0000000000009908 (+0x00009648)  00040084  MADD.S64                        X2, X0, X1
0x000000000000990c (+0x0000964c)  00002081  ADD.S64                         X0, X2, X1
0x0000000000009910 (+0x00009650)  08800680  SUB_IMM.S64                     X0, X0, #1664
# [DWARF] common/pa_trace.h:285
#     279 |             } else {
#     280 |                 ++trace.poll_batch_records;
#     281 |             }
#     282 |         }
#     283 |         trace.poll_burst.call_count[index] = 0;
#     284 |     }
# >   285 |     trace.poll_burst.active_mask = 0;
0x0000000000009914 (+0x00009654)  08000000  ADD_IMM.S64                     X0, X0, #0
0x0000000000009918 (+0x00009658)  0f960a00  STI_XN_IMM.B32                  X0, #1488
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x000000000000991c (+0x0000965c)  0001a70e  CMP.S64.EQ                      X26, X14
0x0000000000009920 (+0x00009660)  4020002f  JUMPC                           #47
0x0000000000009924 (+0x00009664)  0001b70e  CMP.S64.EQ                      X27, X14
0x0000000000009928 (+0x00009668)  4020002d  JUMPC                           #45
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000000992c (+0x0000966c)  0283ca00  ZEROEXT.U32                     X1, X28
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x0000000000009930 (+0x00009670)  0000170e  CMP.S64.EQ                      X1, X14
0x0000000000009934 (+0x00009674)  4020002a  JUMPC                           #42
# [DWARF] common/pa_trace.h:547
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
# >   547 |     const uint32_t slot = core.count;
0x0000000000009938 (+0x00009678)  1c81a000  LD_XD_XN_IMM.B32                X0, X26, #0
# [DWARF] common/pa_trace.h:548
# >   548 |     if (slot >= trace.capacity) {
0x000000000000993c (+0x0000967c)  004000ae  CMP.U64.LT                      X0, X1
0x0000000000009940 (+0x00009680)  40200002  JUMPC                           #2
0x0000000000009944 (+0x00009684)  40000023  JUMP                            #35
# [DWARF] common/pa_trace.h:553
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
# >   553 |     record.start_cycle = start_cycle;
0x0000000000009948 (+0x00009688)  1cc9e988  LD_XD_XN_IMM.B64                X4, X30, #2440
# [DWARF] common/pa_trace.h:552
#     546 |     PA_GM TraceCoreState &core = *trace.core;
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
# >   552 |     PA_GM TraceRecord &record = trace.records[slot];
0x000000000000994c (+0x0000968c)  02020800  MOV_XD_XN.S64                   X1, X0
0x0000000000009950 (+0x00009690)  02c20206  SHL.B64                         X1, #6
0x0000000000009954 (+0x00009694)  0003b081  ADD.S64                         X1, X27, X1
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000009958 (+0x00009698)  0806c002  ADD_IMM.S64                     X3, X12, #2
0x000000000000995c (+0x0000969c)  00c4b78a  AND.B64                         X2, X11, X15
0x0000000000009960 (+0x000096a0)  02c6020e  SHL.B64                         X3, #14
# [DWARF] common/pa_trace.h:557
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
#     555 |     record.task_id = task_id;
#     556 |     record.function_id = function_id;
# >   557 |     record.phase = static_cast<int32_t>(trace_phase);
0x0000000000009964 (+0x000096a4)  070a000b  MOV_XD_IMM                      X5, #11
# [DWARF] common/pa_trace.h:565
#     558 |     record.lane = trace.lane;
#     559 |     record.block_id = trace.block_id;
#     560 |     record.core_idx = trace.core_idx;
#     561 |     record.flags = flags;
#     562 |     record.auxiliary = auxiliary;
#     563 |     // count 最后更新，使其始终指向下一空槽；单写者条件下无需 reserve/commit 两阶段。
#     564 |     // 最终 FlushTraceCore 会把记录体先于该计数一并导出。
# >   565 |     core.count = slot + 1;
0x0000000000009968 (+0x000096a8)  08000001  ADD_IMM.S64                     X0, X0, #1
# [DWARF] common/pa_trace.h:553
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
# >   553 |     record.start_cycle = start_cycle;
0x000000000000996c (+0x000096ac)  09c81a81  STP_XI_XJ_XN.B64                X4, X21, X1, #0
0x0000000000009970 (+0x000096b0)  0008a381  ADD.S64                         X4, X10, X7
0x0000000000009974 (+0x000096b4)  00082184  MADD.S64                        X4, X2, X3
0x0000000000009978 (+0x000096b8)  00044181  ADD.S64                         X2, X4, X3
0x000000000000997c (+0x000096bc)  1cc9e8f8  LD_XD_XN_IMM.B64                X4, X30, #2296
# [DWARF] common/pa_trace.h:556
#     554 |     record.end_cycle = end_cycle;
#     555 |     record.task_id = task_id;
# >   556 |     record.function_id = function_id;
0x0000000000009980 (+0x000096c0)  0706ffff  MOV_XD_IMM                      X3, #65535
0x0000000000009984 (+0x000096c4)  08842680  SUB_IMM.S64                     X2, X2, #1664
0x0000000000009988 (+0x000096c8)  0747ffff  MOVK                            X3, #65535, #1
0x000000000000998c (+0x000096cc)  098811a1  STP_XI_XJ_XN.B32                X4, X3, X1, #16
# [DWARF] common/pa_trace.h:558
#     557 |     record.phase = static_cast<int32_t>(trace_phase);
# >   558 |     record.lane = trace.lane;
0x0000000000009990 (+0x000096d0)  08042000  ADD_IMM.S64                     X2, X2, #0
0x0000000000009994 (+0x000096d4)  1c862560  LD_XD_XN_IMM.B32                X3, X2, #1376
# [DWARF] common/pa_trace.h:557
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
#     555 |     record.task_id = task_id;
#     556 |     record.function_id = function_id;
# >   557 |     record.phase = static_cast<int32_t>(trace_phase);
0x0000000000009998 (+0x000096d8)  098a11b1  STP_XI_XJ_XN.B32                X5, X3, X1, #24
# [DWARF] common/pa_trace.h:559
#     558 |     record.lane = trace.lane;
# >   559 |     record.block_id = trace.block_id;
0x000000000000999c (+0x000096dc)  08062564  ADD_IMM.S64                     X3, X2, #1380
0x00000000000099a0 (+0x000096e0)  0c863100  LDP_XI_XJ_XN.B32                X3, X2, X3, #0
0x00000000000099a4 (+0x000096e4)  03861020  ST_XD_XN_IMM.B32                X3, X1, #32
0x00000000000099a8 (+0x000096e8)  07060000  MOV_XD_IMM                      X3, #0
0x00000000000099ac (+0x000096ec)  07470001  MOVK                            X3, #1, #1
0x00000000000099b0 (+0x000096f0)  004641af  CMPN.U64.LT                     X3, X4, X3
0x00000000000099b4 (+0x000096f4)  02c60201  SHL.B64                         X3, #1
0x00000000000099b8 (+0x000096f8)  00c63c8b  OR.B64                          X3, X3, X25
# [DWARF] common/pa_trace.h:560
# >   560 |     record.core_idx = trace.core_idx;
0x00000000000099bc (+0x000096fc)  08081024  ADD_IMM.S64                     X4, X1, #36
0x00000000000099c0 (+0x00009700)  09844181  STP_XI_XJ_XN.B32                X2, X3, X4, #0
# [DWARF] common/pa_trace.h:562
#     561 |     record.flags = flags;
# >   562 |     record.auxiliary = auxiliary;
0x00000000000099c4 (+0x00009704)  03a2102c  ST_XD_XN_IMM.B32                X17, X1, #44
# [DWARF] common/pa_trace.h:565
#     563 |     // count 最后更新，使其始终指向下一空槽；单写者条件下无需 reserve/commit 两阶段。
#     564 |     // 最终 FlushTraceCore 会把记录体先于该计数一并导出。
# >   565 |     core.count = slot + 1;
0x00000000000099c8 (+0x00009708)  0381a000  ST_XD_XN_IMM.B32                X0, X26, #0
0x00000000000099cc (+0x0000970c)  40000004  JUMP                            #4
# [DWARF] common/pa_trace.h:549
#     543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
# >   549 |         core.dropped = core.dropped + 1;
0x00000000000099d0 (+0x00009710)  1c81a004  LD_XD_XN_IMM.B32                X0, X26, #4
0x00000000000099d4 (+0x00009714)  08000001  ADD_IMM.S64                     X0, X0, #1
0x00000000000099d8 (+0x00009718)  0381a004  ST_XD_XN_IMM.B32                X0, X26, #4
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000099dc (+0x0000971c)  0802c002  ADD_IMM.S64                     X1, X12, #2
# [DWARF] common/pa_frontend.h:262
#     256 |     // TensorTagMixin<TensorArgType, 32>::tags_{} is value-initialized by the
#     257 |     // real L0TaskArgs constructor. TensorRef/scalar slots remain lazy.
#     258 |     // 构造只初始化真实构造函数会触碰的字段，未使用的 tensor/scalar
#     259 |     // 槽保持惰性；整对象 memset 会引入 PA 本身没有的额外前端开销。
#     260 |     volatile int32_t *tags = &args.tags[0];
#     261 |     for (uint32_t index = 0; index < kMaxTaskTensors; ++index) {
# >   262 |         tags[index] = 0;
0x00000000000099e0 (+0x00009720)  0fa7e700  STI_XN_IMM.B32                  X30, #2488
# [DWARF] common/pa_frontend.h:0
# [SOURCE unavailable]
0x00000000000099e4 (+0x00009724)  07080648  MOV_XD_IMM                      X4, #1608
0x00000000000099e8 (+0x00009728)  00c0b78a  AND.B64                         X0, X11, X15
0x00000000000099ec (+0x0000972c)  02c2020e  SHL.B64                         X1, #14
0x00000000000099f0 (+0x00009730)  0004a381  ADD.S64                         X2, X10, X7
0x00000000000099f4 (+0x00009734)  0009e202  SUB.S64                         X4, X30, X4
0x00000000000099f8 (+0x00009738)  00040084  MADD.S64                        X2, X0, X1
# [DWARF] common/pa_frontend.h:262
#     256 |     // TensorTagMixin<TensorArgType, 32>::tags_{} is value-initialized by the
#     257 |     // real L0TaskArgs constructor. TensorRef/scalar slots remain lazy.
#     258 |     // 构造只初始化真实构造函数会触碰的字段，未使用的 tensor/scalar
#     259 |     // 槽保持惰性；整对象 memset 会引入 PA 本身没有的额外前端开销。
#     260 |     volatile int32_t *tags = &args.tags[0];
#     261 |     for (uint32_t index = 0; index < kMaxTaskTensors; ++index) {
# >   262 |         tags[index] = 0;
0x00000000000099fc (+0x0000973c)  020a4800  MOV_XD_XN.S64                   X5, X4
0x0000000000009a00 (+0x00009740)  07060002  MOV_XD_IMM                      X3, #2
0x0000000000009a04 (+0x00009744)  00002081  ADD.S64                         X0, X2, X1
0x0000000000009a08 (+0x00009748)  02ca3440  SBITSET.B64                     X5, X3
0x0000000000009a0c (+0x0000974c)  08800680  SUB_IMM.S64                     X0, X0, #1664
0x0000000000009a10 (+0x00009750)  0f805000  STI_XN_IMM.B32                  X5, #0
0x0000000000009a14 (+0x00009754)  0fa7e800  STI_XN_IMM.B32                  X30, #2496
# [DWARF] common/pa_scheduler_core.h:948
#     942 |     callback(callback_builder);
#     943 |     if (!callback_builder.Valid()) return false;
#     944 |     const LazySampleCallbackBuildCounts &counts = callback_builder.Counts();
#     945 |     stats.result.arg_resets += counts.reset_calls;
#     946 |     stats.result.views_created += counts.views_created;
#     947 |     stats.result.dynamic_create_infos += counts.dynamic_create_infos;
# >   948 |     stats.result.tensor_args_added += counts.tensor_args_added;
0x0000000000009a18 (+0x00009758)  08260000  ADD_IMM.S64                     X19, X0, #0
# [DWARF] common/pa_frontend.h:262
#     256 |     // TensorTagMixin<TensorArgType, 32>::tags_{} is value-initialized by the
#     257 |     // real L0TaskArgs constructor. TensorRef/scalar slots remain lazy.
#     258 |     // 构造只初始化真实构造函数会触碰的字段，未使用的 tensor/scalar
#     259 |     // 槽保持惰性；整对象 memset 会引入 PA 本身没有的额外前端开销。
#     260 |     volatile int32_t *tags = &args.tags[0];
#     261 |     for (uint32_t index = 0; index < kMaxTaskTensors; ++index) {
# >   262 |         tags[index] = 0;
0x0000000000009a1c (+0x0000975c)  0fa7e880  STI_XN_IMM.B32                  X30, #2500
# [DWARF] common/pa_frontend.h:314
#     308 |     args.tensors[index].kind = TensorRefKind::GmTensor;
#     309 |     args.tags[index] = TagValue(tag);
#     310 | }
#     311 | 
#     312 | PA_DEVICE void AppendOutput(TaskArgs &args, const TensorCreateInfo &create_info) {
#     313 |     const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
# >   314 |     args.tensors[index].pointer.create_info = &create_info;
0x0000000000009a20 (+0x00009760)  070201b8  MOV_XD_IMM                      X1, #440
# [DWARF] common/pa_frontend.h:262
#     256 |     // TensorTagMixin<TensorArgType, 32>::tags_{} is value-initialized by the
#     257 |     // real L0TaskArgs constructor. TensorRef/scalar slots remain lazy.
#     258 |     // 构造只初始化真实构造函数会触碰的字段，未使用的 tensor/scalar
#     259 |     // 槽保持惰性；整对象 memset 会引入 PA 本身没有的额外前端开销。
#     260 |     volatile int32_t *tags = &args.tags[0];
#     261 |     for (uint32_t index = 0; index < kMaxTaskTensors; ++index) {
# >   262 |         tags[index] = 0;
0x0000000000009a24 (+0x00009764)  0fa7e900  STI_XN_IMM.B32                  X30, #2504
# [DWARF] common/pa_frontend.h:314
#     308 |     args.tensors[index].kind = TensorRefKind::GmTensor;
#     309 |     args.tags[index] = TagValue(tag);
#     310 | }
#     311 | 
#     312 | PA_DEVICE void AppendOutput(TaskArgs &args, const TensorCreateInfo &create_info) {
#     313 |     const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
# >   314 |     args.tensors[index].pointer.create_info = &create_info;
0x0000000000009a28 (+0x00009768)  0003e081  ADD.S64                         X1, X30, X1
# [DWARF] common/pa_frontend.h:262
#     256 |     // TensorTagMixin<TensorArgType, 32>::tags_{} is value-initialized by the
#     257 |     // real L0TaskArgs constructor. TensorRef/scalar slots remain lazy.
#     258 |     // 构造只初始化真实构造函数会触碰的字段，未使用的 tensor/scalar
#     259 |     // 槽保持惰性；整对象 memset 会引入 PA 本身没有的额外前端开销。
#     260 |     volatile int32_t *tags = &args.tags[0];
#     261 |     for (uint32_t index = 0; index < kMaxTaskTensors; ++index) {
# >   262 |         tags[index] = 0;
0x0000000000009a2c (+0x0000976c)  0fa7e980  STI_XN_IMM.B32                  X30, #2508
# [DWARF] common/pa_frontend.h:313
#     307 |     args.tensors[index].pointer.gm_tensor = &tensor;
#     308 |     args.tensors[index].kind = TensorRefKind::GmTensor;
#     309 |     args.tags[index] = TagValue(tag);
#     310 | }
#     311 | 
#     312 | PA_DEVICE void AppendOutput(TaskArgs &args, const TensorCreateInfo &create_info) {
# >   313 |     const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
0x0000000000009a30 (+0x00009770)  07040003  MOV_XD_IMM                      X2, #3
# [DWARF] common/pa_frontend.h:262
#     256 |     // TensorTagMixin<TensorArgType, 32>::tags_{} is value-initialized by the
#     257 |     // real L0TaskArgs constructor. TensorRef/scalar slots remain lazy.
#     258 |     // 构造只初始化真实构造函数会触碰的字段，未使用的 tensor/scalar
#     259 |     // 槽保持惰性；整对象 memset 会引入 PA 本身没有的额外前端开销。
#     260 |     volatile int32_t *tags = &args.tags[0];
#     261 |     for (uint32_t index = 0; index < kMaxTaskTensors; ++index) {
# >   262 |         tags[index] = 0;
0x0000000000009a34 (+0x00009774)  0fa7ea00  STI_XN_IMM.B32                  X30, #2512
0x0000000000009a38 (+0x00009778)  0fa7ea80  STI_XN_IMM.B32                  X30, #2516
0x0000000000009a3c (+0x0000977c)  0fa7eb00  STI_XN_IMM.B32                  X30, #2520
0x0000000000009a40 (+0x00009780)  0fa7eb80  STI_XN_IMM.B32                  X30, #2524
0x0000000000009a44 (+0x00009784)  0fa7ec00  STI_XN_IMM.B32                  X30, #2528
0x0000000000009a48 (+0x00009788)  0fa7ec80  STI_XN_IMM.B32                  X30, #2532
0x0000000000009a4c (+0x0000978c)  0fa7ed00  STI_XN_IMM.B32                  X30, #2536
0x0000000000009a50 (+0x00009790)  0fa7ed80  STI_XN_IMM.B32                  X30, #2540
0x0000000000009a54 (+0x00009794)  0fa7ee00  STI_XN_IMM.B32                  X30, #2544
0x0000000000009a58 (+0x00009798)  0fa7ee80  STI_XN_IMM.B32                  X30, #2548
0x0000000000009a5c (+0x0000979c)  0fa7ef00  STI_XN_IMM.B32                  X30, #2552
0x0000000000009a60 (+0x000097a0)  0fa7ef80  STI_XN_IMM.B32                  X30, #2556
0x0000000000009a64 (+0x000097a4)  0fa9e000  STI_XN_IMM.B32                  X30, #2560
0x0000000000009a68 (+0x000097a8)  0fa9e080  STI_XN_IMM.B32                  X30, #2564
0x0000000000009a6c (+0x000097ac)  0fa9e100  STI_XN_IMM.B32                  X30, #2568
0x0000000000009a70 (+0x000097b0)  0fa9e180  STI_XN_IMM.B32                  X30, #2572
0x0000000000009a74 (+0x000097b4)  0fa9e200  STI_XN_IMM.B32                  X30, #2576
0x0000000000009a78 (+0x000097b8)  0fa9e280  STI_XN_IMM.B32                  X30, #2580
0x0000000000009a7c (+0x000097bc)  0fa9e300  STI_XN_IMM.B32                  X30, #2584
0x0000000000009a80 (+0x000097c0)  0fa9e380  STI_XN_IMM.B32                  X30, #2588
0x0000000000009a84 (+0x000097c4)  0fa9e400  STI_XN_IMM.B32                  X30, #2592
0x0000000000009a88 (+0x000097c8)  0fa9e480  STI_XN_IMM.B32                  X30, #2596
0x0000000000009a8c (+0x000097cc)  0fa9e500  STI_XN_IMM.B32                  X30, #2600
0x0000000000009a90 (+0x000097d0)  0fa9e580  STI_XN_IMM.B32                  X30, #2604
0x0000000000009a94 (+0x000097d4)  0fa9e600  STI_XN_IMM.B32                  X30, #2608
0x0000000000009a98 (+0x000097d8)  0fa9e680  STI_XN_IMM.B32                  X30, #2612
# [DWARF] common/pa_frontend.h:265
#     263 |     }
#     264 |     args.tensor_count = 0;
# >   265 |     args.scalar_count = 0;
0x0000000000009a9c (+0x000097dc)  0fb3e780  STI_XN_IMM.B32                  X30, #3260
# [DWARF] common/pa_frontend.h:266
# >   266 |     args.has_error = false;
0x0000000000009aa0 (+0x000097e0)  0f33e800  STI_XN_IMM.B8                   X30, #3264
# [DWARF] common/pa_frontend.h:267
# >   267 |     args.error_msg = 0;
0x0000000000009aa4 (+0x000097e4)  0ff3e900  STI_XN_IMM.B64                  X30, #3272
# [DWARF] common/pa_frontend.h:268
# >   268 |     args.launch_spec.core_num = 1;
0x0000000000009aa8 (+0x000097e8)  0f73ea01  STI_XN_IMM.B16                  X30, #3280
# [DWARF] common/pa_frontend.h:269
# >   269 |     args.launch_spec.require_sync_start = false;
0x0000000000009aac (+0x000097ec)  0f33ea40  STI_XN_IMM.B8                   X30, #3282
# [DWARF] common/pa_frontend.h:243
#     237 | 
#     238 | PA_DEVICE void ClearDumpArgSelection(PaDumpArgSelection &selection) {
#     239 |     // Volatile stores intentionally preserve the profiling-enabled PA reset
#     240 |     // traffic even though the standalone winner workload never consumes dump data.
#     241 |     // volatile 的目的不是同步，而是阻止编译器删掉这段生产基线中存在的写流量。
#     242 |     volatile uint64_t *masks = &selection.dump_arg_mask;
# >   243 |     masks[0] = 0;
0x0000000000009ab0 (+0x000097f0)  0ff3eb00  STI_XN_IMM.B64                  X30, #3288
# [DWARF] common/pa_frontend.h:244
# >   244 |     masks[1] = 0;
0x0000000000009ab4 (+0x000097f4)  0ff3ec00  STI_XN_IMM.B64                  X30, #3296
# [DWARF] common/pa_frontend.h:247
#     245 |     volatile uint64_t *sources = &selection.scalar_source_ptrs[0];
#     246 |     for (uint32_t index = 0; index < kMaxTaskScalars; ++index) {
# >   247 |         sources[index] = 0;
0x0000000000009ab8 (+0x000097f8)  0ff3ed00  STI_XN_IMM.B64                  X30, #3304
0x0000000000009abc (+0x000097fc)  0ff3ee00  STI_XN_IMM.B64                  X30, #3312
0x0000000000009ac0 (+0x00009800)  0ff3ef00  STI_XN_IMM.B64                  X30, #3320
0x0000000000009ac4 (+0x00009804)  0ff5e000  STI_XN_IMM.B64                  X30, #3328
0x0000000000009ac8 (+0x00009808)  0ff5e100  STI_XN_IMM.B64                  X30, #3336
0x0000000000009acc (+0x0000980c)  0ff5e200  STI_XN_IMM.B64                  X30, #3344
0x0000000000009ad0 (+0x00009810)  0ff5e300  STI_XN_IMM.B64                  X30, #3352
0x0000000000009ad4 (+0x00009814)  0ff5e400  STI_XN_IMM.B64                  X30, #3360
0x0000000000009ad8 (+0x00009818)  0ff5e500  STI_XN_IMM.B64                  X30, #3368
0x0000000000009adc (+0x0000981c)  0ff5e600  STI_XN_IMM.B64                  X30, #3376
0x0000000000009ae0 (+0x00009820)  0ff5e700  STI_XN_IMM.B64                  X30, #3384
0x0000000000009ae4 (+0x00009824)  0ff5e800  STI_XN_IMM.B64                  X30, #3392
0x0000000000009ae8 (+0x00009828)  0ff5e900  STI_XN_IMM.B64                  X30, #3400
0x0000000000009aec (+0x0000982c)  0ff5ea00  STI_XN_IMM.B64                  X30, #3408
0x0000000000009af0 (+0x00009830)  0ff5eb00  STI_XN_IMM.B64                  X30, #3416
0x0000000000009af4 (+0x00009834)  0ff5ec00  STI_XN_IMM.B64                  X30, #3424
# [DWARF] common/pa_frontend.h:251
#     248 |     }
#     249 |     volatile uint8_t *dtypes = &selection.scalar_dtypes[0];
#     250 |     for (uint32_t index = 0; index < kMaxTaskScalars; ++index) {
# >   251 |         dtypes[index] = 0;
0x0000000000009af8 (+0x00009838)  0f35ed00  STI_XN_IMM.B8                   X30, #3432
0x0000000000009afc (+0x0000983c)  0f35ed20  STI_XN_IMM.B8                   X30, #3433
0x0000000000009b00 (+0x00009840)  0f35ed40  STI_XN_IMM.B8                   X30, #3434
0x0000000000009b04 (+0x00009844)  0f35ed60  STI_XN_IMM.B8                   X30, #3435
0x0000000000009b08 (+0x00009848)  0f35ed80  STI_XN_IMM.B8                   X30, #3436
0x0000000000009b0c (+0x0000984c)  0f35eda0  STI_XN_IMM.B8                   X30, #3437
0x0000000000009b10 (+0x00009850)  0f35edc0  STI_XN_IMM.B8                   X30, #3438
0x0000000000009b14 (+0x00009854)  0f35ede0  STI_XN_IMM.B8                   X30, #3439
0x0000000000009b18 (+0x00009858)  0f35ee00  STI_XN_IMM.B8                   X30, #3440
0x0000000000009b1c (+0x0000985c)  0f35ee20  STI_XN_IMM.B8                   X30, #3441
0x0000000000009b20 (+0x00009860)  0f35ee40  STI_XN_IMM.B8                   X30, #3442
0x0000000000009b24 (+0x00009864)  0f35ee60  STI_XN_IMM.B8                   X30, #3443
0x0000000000009b28 (+0x00009868)  0f35ee80  STI_XN_IMM.B8                   X30, #3444
0x0000000000009b2c (+0x0000986c)  0f35eea0  STI_XN_IMM.B8                   X30, #3445
0x0000000000009b30 (+0x00009870)  0f35eec0  STI_XN_IMM.B8                   X30, #3446
0x0000000000009b34 (+0x00009874)  0f35eee0  STI_XN_IMM.B8                   X30, #3447
# [DWARF] common/pa_scheduler_core.h:948
#     942 |     callback(callback_builder);
#     943 |     if (!callback_builder.Valid()) return false;
#     944 |     const LazySampleCallbackBuildCounts &counts = callback_builder.Counts();
#     945 |     stats.result.arg_resets += counts.reset_calls;
#     946 |     stats.result.views_created += counts.views_created;
#     947 |     stats.result.dynamic_create_infos += counts.dynamic_create_infos;
# >   948 |     stats.result.tensor_args_added += counts.tensor_args_added;
0x0000000000009b38 (+0x00009878)  1cc133e0  LD_XD_XN_IMM.B64                X0, X19, #992
# [DWARF] common/pa_frontend.h:314
#     308 |     args.tensors[index].kind = TensorRefKind::GmTensor;
#     309 |     args.tags[index] = TagValue(tag);
#     310 | }
#     311 | 
#     312 | PA_DEVICE void AppendOutput(TaskArgs &args, const TensorCreateInfo &create_info) {
#     313 |     const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
# >   314 |     args.tensors[index].pointer.create_info = &create_info;
0x0000000000009b3c (+0x0000987c)  03c3ea38  ST_XD_XN_IMM.B64                X1, X30, #2616
# [DWARF] common/pa_frontend.h:0
# [SOURCE unavailable]
0x0000000000009b40 (+0x00009880)  070201f8  MOV_XD_IMM                      X1, #504
0x0000000000009b44 (+0x00009884)  0003e081  ADD.S64                         X1, X30, X1
# [DWARF] common/pa_frontend.h:271
#     265 |     args.scalar_count = 0;
#     266 |     args.has_error = false;
#     267 |     args.error_msg = 0;
#     268 |     args.launch_spec.core_num = 1;
#     269 |     args.launch_spec.require_sync_start = false;
#     270 |     ClearDumpArgSelection(args.dump_arg_selection);
# >   271 |     args.explicit_deps = 0;
0x0000000000009b48 (+0x00009888)  0ff5ef00  STI_XN_IMM.B64                  X30, #3448
# [DWARF] common/pa_frontend.h:272
# >   272 |     args.explicit_dep_count = 0;
0x0000000000009b4c (+0x0000988c)  0fb7e000  STI_XN_IMM.B32                  X30, #3456
# [DWARF] common/pa_frontend.h:315
#     309 |     args.tags[index] = TagValue(tag);
#     310 | }
#     311 | 
#     312 | PA_DEVICE void AppendOutput(TaskArgs &args, const TensorCreateInfo &create_info) {
#     313 |     const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
#     314 |     args.tensors[index].pointer.create_info = &create_info;
# >   315 |     args.tensors[index].kind = TensorRefKind::CreateInfo;
0x0000000000009b50 (+0x00009890)  0307ea40  ST_XD_XN_IMM.B8                 X3, X30, #2624
# [DWARF] common/pa_frontend.h:316
# >   316 |     args.tags[index] = TagValue(TensorArgType::Output);
0x0000000000009b54 (+0x00009894)  0fa7e701  STI_XN_IMM.B32                  X30, #2488
# [DWARF] common/pa_frontend.h:314
#     308 |     args.tensors[index].kind = TensorRefKind::GmTensor;
#     309 |     args.tags[index] = TagValue(tag);
#     310 | }
#     311 | 
#     312 | PA_DEVICE void AppendOutput(TaskArgs &args, const TensorCreateInfo &create_info) {
#     313 |     const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
# >   314 |     args.tensors[index].pointer.create_info = &create_info;
0x0000000000009b58 (+0x00009898)  03c3ea48  ST_XD_XN_IMM.B64                X1, X30, #2632
# [DWARF] common/pa_frontend.h:315
# >   315 |     args.tensors[index].kind = TensorRefKind::CreateInfo;
0x0000000000009b5c (+0x0000989c)  0307ea50  ST_XD_XN_IMM.B8                 X3, X30, #2640
# [DWARF] common/pa_frontend.h:316
# >   316 |     args.tags[index] = TagValue(TensorArgType::Output);
0x0000000000009b60 (+0x000098a0)  0f805001  STI_XN_IMM.B32                  X5, #0
# [DWARF] common/pa_frontend.h:313
#     307 |     args.tensors[index].pointer.gm_tensor = &tensor;
#     308 |     args.tensors[index].kind = TensorRefKind::GmTensor;
#     309 |     args.tags[index] = TagValue(tag);
#     310 | }
#     311 | 
#     312 | PA_DEVICE void AppendOutput(TaskArgs &args, const TensorCreateInfo &create_info) {
# >   313 |     const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
0x0000000000009b64 (+0x000098a4)  0385ecb8  ST_XD_XN_IMM.B32                X2, X30, #3256
# [DWARF] common/pa_frontend.h:314
# >   314 |     args.tensors[index].pointer.create_info = &create_info;
0x0000000000009b68 (+0x000098a8)  03c3ea58  ST_XD_XN_IMM.B64                X1, X30, #2648
# [DWARF] ccec/ccec_ops.h:276
#     270 |     }
#     271 | 
#     272 |     __aicore__ static inline bool FinishLazySampleCallback(
#     273 |         const pa_scheduler::LazySampleCallbackTicket *ticket, const pa_scheduler::TaskArgs *args
#     274 |     ) {
#     275 | #if defined(PA_BUILD_AIC)
# >   276 |         return ::pa_scheduler_lazy_sample_callback_finish_aic(ticket, args) != 0;
0x0000000000009b6c (+0x000098ac)  02024800  MOV_XD_XN.S64                   X1, X4
0x0000000000009b70 (+0x000098b0)  03cbe8f0  ST_XD_XN_IMM.B64                X5, X30, #2288
# [DWARF] common/pa_frontend.h:315
#     309 |     args.tags[index] = TagValue(tag);
#     310 | }
#     311 | 
#     312 | PA_DEVICE void AppendOutput(TaskArgs &args, const TensorCreateInfo &create_info) {
#     313 |     const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
#     314 |     args.tensors[index].pointer.create_info = &create_info;
# >   315 |     args.tensors[index].kind = TensorRefKind::CreateInfo;
0x0000000000009b74 (+0x000098b4)  0307ea60  ST_XD_XN_IMM.B8                 X3, X30, #2656
# [DWARF] common/pa_frontend.h:316
# >   316 |     args.tags[index] = TagValue(TensorArgType::Output);
0x0000000000009b78 (+0x000098b8)  0fa7e801  STI_XN_IMM.B32                  X30, #2496
# [DWARF] common/pa_scheduler_core.h:948
#     942 |     callback(callback_builder);
#     943 |     if (!callback_builder.Valid()) return false;
#     944 |     const LazySampleCallbackBuildCounts &counts = callback_builder.Counts();
#     945 |     stats.result.arg_resets += counts.reset_calls;
#     946 |     stats.result.views_created += counts.views_created;
#     947 |     stats.result.dynamic_create_infos += counts.dynamic_create_infos;
# >   948 |     stats.result.tensor_args_added += counts.tensor_args_added;
0x0000000000009b7c (+0x000098bc)  08000003  ADD_IMM.S64                     X0, X0, #3
0x0000000000009b80 (+0x000098c0)  03c133e0  ST_XD_XN_IMM.B64                X0, X19, #992
# [DWARF] common/pa_scheduler_core.h:1161
#    1155 |     );
#    1156 | 
#    1157 |     if (!BuildLazySampleCallbackArgs<Kind, Lazy>(orch, args, batch, claim.won, stats)) {
#    1158 |         SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
#    1159 |         return false;
#    1160 |     }
# >  1161 |     const LazySampleCallbackTicket ticket{
0x0000000000009b84 (+0x000098c4)  1cc1e928  LD_XD_XN_IMM.B64                X0, X30, #2344
0x0000000000009b88 (+0x000098c8)  03c1e338  ST_XD_XN_IMM.B64                X0, X30, #824
0x0000000000009b8c (+0x000098cc)  1cc1e8f8  LD_XD_XN_IMM.B64                X0, X30, #2296
0x0000000000009b90 (+0x000098d0)  0381e340  ST_XD_XN_IMM.B32                X0, X30, #832
# [DWARF] ccec/ccec_ops.h:276
#     270 |     }
#     271 | 
#     272 |     __aicore__ static inline bool FinishLazySampleCallback(
#     273 |         const pa_scheduler::LazySampleCallbackTicket *ticket, const pa_scheduler::TaskArgs *args
#     274 |     ) {
#     275 | #if defined(PA_BUILD_AIC)
# >   276 |         return ::pa_scheduler_lazy_sample_callback_finish_aic(ticket, args) != 0;
0x0000000000009b94 (+0x000098d4)  07000338  MOV_XD_IMM                      X0, #824
0x0000000000009b98 (+0x000098d8)  0001e001  ADD.S64                         X0, X30, X0
# [DWARF] common/pa_scheduler_core.h:1161
#    1155 |     );
#    1156 | 
#    1157 |     if (!BuildLazySampleCallbackArgs<Kind, Lazy>(orch, args, batch, claim.won, stats)) {
#    1158 |         SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
#    1159 |         return false;
#    1160 |     }
# >  1161 |     const LazySampleCallbackTicket ticket{
0x0000000000009b9c (+0x000098dc)  0f4de882  STI_XN_IMM.B16                  X30, #836
0x0000000000009ba0 (+0x000098e0)  0333e346  ST_XD_XN_IMM.B8                 X25, X30, #838
0x0000000000009ba4 (+0x000098e4)  0f0de8e0  STI_XN_IMM.B8                   X30, #839
# [DWARF] ccec/ccec_ops.h:276
#     270 |     }
#     271 | 
#     272 |     __aicore__ static inline bool FinishLazySampleCallback(
#     273 |         const pa_scheduler::LazySampleCallbackTicket *ticket, const pa_scheduler::TaskArgs *args
#     274 |     ) {
#     275 | #if defined(PA_BUILD_AIC)
# >   276 |         return ::pa_scheduler_lazy_sample_callback_finish_aic(ticket, args) != 0;
0x0000000000009ba8 (+0x000098e8)  070477fb  MOV_XD_IMM                      X2, #30715
0x0000000000009bac (+0x000098ec)  07450000  MOVK                            X2, #0, #1
0x0000000000009bb0 (+0x000098f0)  07850000  MOVK                            X2, #0, #2
0x0000000000009bb4 (+0x000098f4)  40422000  CALL                            X2, #0
# [DWARF] ccec/ccec_ops.h:0
# [SOURCE unavailable]
0x0000000000009bb8 (+0x000098f8)  1cf3e980  LD_XD_XN_IMM.B64                X25, X30, #2432
0x0000000000009bbc (+0x000098fc)  071c0000  MOV_XD_IMM                      X14, #0
# [DWARF] ccec/ccec_ops.h:276
#     270 |     }
#     271 | 
#     272 |     __aicore__ static inline bool FinishLazySampleCallback(
#     273 |         const pa_scheduler::LazySampleCallbackTicket *ticket, const pa_scheduler::TaskArgs *args
#     274 |     ) {
#     275 | #if defined(PA_BUILD_AIC)
# >   276 |         return ::pa_scheduler_lazy_sample_callback_finish_aic(ticket, args) != 0;
0x0000000000009bc0 (+0x00009900)  02800a00  ZEROEXT.U32                     X0, X0
0x0000000000009bc4 (+0x00009904)  07220001  MOV_XD_IMM                      X17, #1
0x0000000000009bc8 (+0x00009908)  0000070e  CMP.S64.EQ                      X0, X14
0x0000000000009bcc (+0x0000990c)  07100004  MOV_XD_IMM                      X8, #4
0x0000000000009bd0 (+0x00009910)  07120006  MOV_XD_IMM                      X9, #6
0x0000000000009bd4 (+0x00009914)  07200020  MOV_XD_IMM                      X16, #32
# [DWARF] common/pa_scheduler_core.h:1382
#    1376 |         orchestration_begin = TraceTimestamp<Ops>(stats.trace, stats.result);
#    1377 |         InitPaOrchestration(orchestration, batches, &state->context_lens[0]);
#    1378 |         for (uint32_t batch = 0; batch < batches; ++batch) {
#    1379 | #if defined(PA_LAZY_SAMPLE_SHAPE_ID)
#    1380 |             BeginPaBatchForLazySampleCallback(orchestration, batch);
#    1381 |             ++stats.result.context_reads;
# >  1382 |             if (!SubmitLazySampleCallback<
0x0000000000009bd8 (+0x00009918)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000009bdc (+0x0000991c)  070058cf  MOV_XD_IMM                      X0, #22735
0x0000000000009be0 (+0x00009920)  07410000  MOVK                            X0, #0, #1
0x0000000000009be4 (+0x00009924)  07810000  MOVK                            X0, #0, #2
0x0000000000009be8 (+0x00009928)  40220000  JUMPC                           X0, #0
0x0000000000009bec (+0x0000992c)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
# [DWARF] common/pa_frontend.h:550
#     544 | }
#     545 | 
#     546 | PA_DEVICE void PreparePaBlockGroup(PaOrchestrationState &orch, uint64_t block_offset) {
#     547 |     // 输入 block_offset 位于 [0,current_blocks)；输出 nblocks 最多64，并计算最后
#     548 |     // 一个 block 的有效 token 数。Case1 只有一个 group，但仍执行通用边界算术。
#     549 |     orch.current_block_offset = block_offset;
# >   550 |     orch.current_nblocks = MinU64(kPaBlocksPerRequest, orch.current_blocks - block_offset);
0x0000000000009bf0 (+0x00009930)  0801e2c8  ADD_IMM.S64                     X0, X30, #712
# [DWARF] common/pa_frontend.h:832
#     826 |     // 每个 worker 都完整回放并接收自己 materialize 的 descriptor；只有 Claim winner
#     827 |     // 会执行 kernel，但 loser 的后续 orchestration 仍使用相同 task_id/owner 拓扑。
#     828 |     switch (kind) {
#     829 |         case TaskKind::Alloc:
#     830 |             orch.accumulated_output = outputs.tensors[0];
#     831 |             orch.accumulated_sum = outputs.tensors[1];
# >   832 |             orch.accumulated_max = outputs.tensors[2];
0x0000000000009bf4 (+0x00009934)  1cc93060  LD_XD_XN_IMM.B64                X4, X19, #96
# [DWARF] common/pa_frontend.h:550
#     544 | }
#     545 | 
#     546 | PA_DEVICE void PreparePaBlockGroup(PaOrchestrationState &orch, uint64_t block_offset) {
#     547 |     // 输入 block_offset 位于 [0,current_blocks)；输出 nblocks 最多64，并计算最后
#     548 |     // 一个 block 的有效 token 数。Case1 只有一个 group，但仍执行通用边界算术。
#     549 |     orch.current_block_offset = block_offset;
# >   550 |     orch.current_nblocks = MinU64(kPaBlocksPerRequest, orch.current_blocks - block_offset);
0x0000000000009bf8 (+0x00009938)  0cc20000  LDP_XI_XJ_XN.B64                X1, X0, X0, #0
# [DWARF] common/pa_frontend.h:830
#     824 | 
#     825 | PA_DEVICE void AcceptTaskOutputs(PaOrchestrationState &orch, TaskKind kind, const TaskOutputs &outputs) {
#     826 |     // 每个 worker 都完整回放并接收自己 materialize 的 descriptor；只有 Claim winner
#     827 |     // 会执行 kernel，但 loser 的后续 orchestration 仍使用相同 task_id/owner 拓扑。
#     828 |     switch (kind) {
#     829 |         case TaskKind::Alloc:
# >   830 |             orch.accumulated_output = outputs.tensors[0];
0x0000000000009bfc (+0x0000993c)  08053050  ADD_IMM.S64                     X2, X19, #80
0x0000000000009c00 (+0x00009940)  070a0040  MOV_XD_IMM                      X5, #64
0x0000000000009c04 (+0x00009944)  0cc42180  LDP_XI_XJ_XN.B64                X2, X3, X2, #0
# [DWARF] common/pa_frontend.h:0
# [SOURCE unavailable]
0x0000000000009c08 (+0x00009948)  071e7fff  MOV_XD_IMM                      X15, #32767
# [DWARF] common/pa_frontend.h:528
#     522 |     orch.query_view.strides[0] = kPaHeadDim;
#     523 |     orch.query_view.strides[1] = 1;
#     524 |     orch.query_view.extent_elem_cache = kPaHeads * kPaHeadDim;
#     525 | }
#     526 | #endif
#     527 | 
# >   528 | PA_DEVICE uint64_t MinU64(uint64_t lhs, uint64_t rhs) { return lhs < rhs ? lhs : rhs; }
0x0000000000009c0c (+0x0000994c)  004002ae  CMP.U64.LT                      X0, X5
0x0000000000009c10 (+0x00009950)  00c00289  SEL.B64                         X0, X0, X5
# [DWARF] common/pa_frontend.h:550
#     544 | }
#     545 | 
#     546 | PA_DEVICE void PreparePaBlockGroup(PaOrchestrationState &orch, uint64_t block_offset) {
#     547 |     // 输入 block_offset 位于 [0,current_blocks)；输出 nblocks 最多64，并计算最后
#     548 |     // 一个 block 的有效 token 数。Case1 只有一个 group，但仍执行通用边界算术。
#     549 |     orch.current_block_offset = block_offset;
# >   550 |     orch.current_nblocks = MinU64(kPaBlocksPerRequest, orch.current_blocks - block_offset);
0x0000000000009c14 (+0x00009954)  03c1e2e0  ST_XD_XN_IMM.B64                X0, X30, #736
# [DWARF] common/pa_frontend.h:552
#     551 |     const uint64_t last_block_sequence_start =
# >   552 |         (block_offset + orch.current_nblocks - 1) * kPaBlockSize;
0x0000000000009c18 (+0x00009958)  02c00207  SHL.B64                         X0, #7
0x0000000000009c1c (+0x0000995c)  00001002  SUB.S64                         X0, X1, X0
# [DWARF] common/pa_frontend.h:830
#     824 | 
#     825 | PA_DEVICE void AcceptTaskOutputs(PaOrchestrationState &orch, TaskKind kind, const TaskOutputs &outputs) {
#     826 |     // 每个 worker 都完整回放并接收自己 materialize 的 descriptor；只有 Claim winner
#     827 |     // 会执行 kernel，但 loser 的后续 orchestration 仍使用相同 task_id/owner 拓扑。
#     828 |     switch (kind) {
#     829 |         case TaskKind::Alloc:
# >   830 |             orch.accumulated_output = outputs.tensors[0];
0x0000000000009c20 (+0x00009960)  0803e2f8  ADD_IMM.S64                     X1, X30, #760
0x0000000000009c24 (+0x00009964)  09c41181  STP_XI_XJ_XN.B64                X2, X3, X1, #0
0x0000000000009c28 (+0x00009968)  1cc5e960  LD_XD_XN_IMM.B64                X2, X30, #2400
# [DWARF] common/pa_frontend.h:553
#     547 |     // 输入 block_offset 位于 [0,current_blocks)；输出 nblocks 最多64，并计算最后
#     548 |     // 一个 block 的有效 token 数。Case1 只有一个 group，但仍执行通用边界算术。
#     549 |     orch.current_block_offset = block_offset;
#     550 |     orch.current_nblocks = MinU64(kPaBlocksPerRequest, orch.current_blocks - block_offset);
#     551 |     const uint64_t last_block_sequence_start =
#     552 |         (block_offset + orch.current_nblocks - 1) * kPaBlockSize;
# >   553 |     orch.current_valid_len = MinU64(kPaBlockSize, orch.current_sequence - last_block_sequence_start);
0x0000000000009c2c (+0x0000996c)  08000080  ADD_IMM.S64                     X0, X0, #128
0x0000000000009c30 (+0x00009970)  07020080  MOV_XD_IMM                      X1, #128
# [DWARF] common/pa_frontend.h:832
#     826 |     // 每个 worker 都完整回放并接收自己 materialize 的 descriptor；只有 Claim winner
#     827 |     // 会执行 kernel，但 loser 的后续 orchestration 仍使用相同 task_id/owner 拓扑。
#     828 |     switch (kind) {
#     829 |         case TaskKind::Alloc:
#     830 |             orch.accumulated_output = outputs.tensors[0];
#     831 |             orch.accumulated_sum = outputs.tensors[1];
# >   832 |             orch.accumulated_max = outputs.tensors[2];
0x0000000000009c34 (+0x00009974)  03c9e308  ST_XD_XN_IMM.B64                X4, X30, #776
# [DWARF] common/pa_frontend.h:528
#     522 |     orch.query_view.strides[0] = kPaHeadDim;
#     523 |     orch.query_view.strides[1] = 1;
#     524 |     orch.query_view.extent_elem_cache = kPaHeads * kPaHeadDim;
#     525 | }
#     526 | #endif
#     527 | 
# >   528 | PA_DEVICE uint64_t MinU64(uint64_t lhs, uint64_t rhs) { return lhs < rhs ? lhs : rhs; }
0x0000000000009c38 (+0x00009978)  004000ae  CMP.U64.LT                      X0, X1
# [DWARF] common/pa_frontend.h:549
#     543 |     return static_cast<uint64_t>(*value);
#     544 | }
#     545 | 
#     546 | PA_DEVICE void PreparePaBlockGroup(PaOrchestrationState &orch, uint64_t block_offset) {
#     547 |     // 输入 block_offset 位于 [0,current_blocks)；输出 nblocks 最多64，并计算最后
#     548 |     // 一个 block 的有效 token 数。Case1 只有一个 group，但仍执行通用边界算术。
# >   549 |     orch.current_block_offset = block_offset;
0x0000000000009c3c (+0x0000997c)  0fcbeb00  STI_XN_IMM.B64                  X30, #728
# [DWARF] common/pa_frontend.h:528
#     522 |     orch.query_view.strides[0] = kPaHeadDim;
#     523 |     orch.query_view.strides[1] = 1;
#     524 |     orch.query_view.extent_elem_cache = kPaHeads * kPaHeadDim;
#     525 | }
#     526 | #endif
#     527 | 
# >   528 | PA_DEVICE uint64_t MinU64(uint64_t lhs, uint64_t rhs) { return lhs < rhs ? lhs : rhs; }
0x0000000000009c40 (+0x00009980)  00c00089  SEL.B64                         X0, X0, X1
# [DWARF] common/pa_frontend.h:553
#     547 |     // 输入 block_offset 位于 [0,current_blocks)；输出 nblocks 最多64，并计算最后
#     548 |     // 一个 block 的有效 token 数。Case1 只有一个 group，但仍执行通用边界算术。
#     549 |     orch.current_block_offset = block_offset;
#     550 |     orch.current_nblocks = MinU64(kPaBlocksPerRequest, orch.current_blocks - block_offset);
#     551 |     const uint64_t last_block_sequence_start =
#     552 |         (block_offset + orch.current_nblocks - 1) * kPaBlockSize;
# >   553 |     orch.current_valid_len = MinU64(kPaBlockSize, orch.current_sequence - last_block_sequence_start);
0x0000000000009c44 (+0x00009984)  03c1e2e8  ST_XD_XN_IMM.B64                X0, X30, #744
# [DWARF] common/pa_scheduler_core.h:790
#     784 | 
#     785 | PA_DEVICE void BeginLazySampleCallbackSubmit(PA_GM WorkerState &worker, SubmitContext &context) {
#     786 |     // This is BeginSubmit without an already-materialized TaskArgs.  The same
#     787 |     // fields are completed synchronously after the single callback builds args.
#     788 |     const uint32_t task_id = static_cast<uint32_t>(worker.local_index++);
#     789 |     context.self = &worker;
# >   790 |     context.payload = &worker.payloads[task_id & kPayloadMask];
0x0000000000009c48 (+0x00009988)  070207ff  MOV_XD_IMM                      X1, #2047
# [DWARF] common/pa_scheduler_core.h:788
#     782 | static_assert(offsetof(LazySampleCallbackTicket, function_id) == 12, "lazy sample callback ticket function offset mismatch");
#     783 | static_assert(offsetof(LazySampleCallbackTicket, won) == 14, "lazy sample callback ticket winner offset mismatch");
#     784 | 
#     785 | PA_DEVICE void BeginLazySampleCallbackSubmit(PA_GM WorkerState &worker, SubmitContext &context) {
#     786 |     // This is BeginSubmit without an already-materialized TaskArgs.  The same
#     787 |     // fields are completed synchronously after the single callback builds args.
# >   788 |     const uint32_t task_id = static_cast<uint32_t>(worker.local_index++);
0x0000000000009c4c (+0x0000998c)  1c862014  LD_XD_XN_IMM.B32                X3, X2, #20
0x0000000000009c50 (+0x00009990)  08003001  ADD_IMM.S64                     X0, X3, #1
0x0000000000009c54 (+0x00009994)  03802014  ST_XD_XN_IMM.B32                X0, X2, #20
# [DWARF] common/pa_scheduler_core.h:790
#     789 |     context.self = &worker;
# >   790 |     context.payload = &worker.payloads[task_id & kPayloadMask];
0x0000000000009c58 (+0x00009998)  1cc1e900  LD_XD_XN_IMM.B64                X0, X30, #2304
0x0000000000009c5c (+0x0000999c)  00c2308a  AND.B64                         X1, X3, X1
0x0000000000009c60 (+0x000099a0)  02c2020c  SHL.B64                         X1, #12
0x0000000000009c64 (+0x000099a4)  00000081  ADD.S64                         X0, X0, X1
0x0000000000009c68 (+0x000099a8)  02023800  MOV_XD_XN.S64                   X1, X3
# [DWARF] common/pa_scheduler_core.h:789
#     783 | static_assert(offsetof(LazySampleCallbackTicket, won) == 14, "lazy sample callback ticket winner offset mismatch");
#     784 | 
#     785 | PA_DEVICE void BeginLazySampleCallbackSubmit(PA_GM WorkerState &worker, SubmitContext &context) {
#     786 |     // This is BeginSubmit without an already-materialized TaskArgs.  The same
#     787 |     // fields are completed synchronously after the single callback builds args.
#     788 |     const uint32_t task_id = static_cast<uint32_t>(worker.local_index++);
# >   789 |     context.self = &worker;
0x0000000000009c6c (+0x000099ac)  09c53031  STP_XI_XJ_XN.B64                X2, X0, X19, #24
# [DWARF] common/pa_scheduler_core.h:793
#     790 |     context.payload = &worker.payloads[task_id & kPayloadMask];
#     791 |     context.task_id = static_cast<int32_t>(task_id);
#     792 |     context.tensor_count = 0;
# >   793 |     context.scalar_count = 0;
0x0000000000009c70 (+0x000099b0)  07000000  MOV_XD_IMM                      X0, #0
# [DWARF] common/pa_scheduler_core.h:791
#     785 | PA_DEVICE void BeginLazySampleCallbackSubmit(PA_GM WorkerState &worker, SubmitContext &context) {
#     786 |     // This is BeginSubmit without an already-materialized TaskArgs.  The same
#     787 |     // fields are completed synchronously after the single callback builds args.
#     788 |     const uint32_t task_id = static_cast<uint32_t>(worker.local_index++);
#     789 |     context.self = &worker;
#     790 |     context.payload = &worker.payloads[task_id & kPayloadMask];
# >   791 |     context.task_id = static_cast<int32_t>(task_id);
0x0000000000009c74 (+0x000099b4)  03833028  ST_XD_XN_IMM.B32                X1, X19, #40
# [DWARF] common/pa_scheduler_core.h:793
#     792 |     context.tensor_count = 0;
# >   793 |     context.scalar_count = 0;
0x0000000000009c78 (+0x000099b8)  08053030  ADD_IMM.S64                     X2, X19, #48
# [DWARF] common/pa_scheduler_core.h:792
#     786 |     // This is BeginSubmit without an already-materialized TaskArgs.  The same
#     787 |     // fields are completed synchronously after the single callback builds args.
#     788 |     const uint32_t task_id = static_cast<uint32_t>(worker.local_index++);
#     789 |     context.self = &worker;
#     790 |     context.payload = &worker.payloads[task_id & kPayloadMask];
#     791 |     context.task_id = static_cast<int32_t>(task_id);
# >   792 |     context.tensor_count = 0;
0x0000000000009c7c (+0x000099bc)  0f813580  STI_XN_IMM.B32                  X19, #44
# [DWARF] common/pa_scheduler_core.h:794
#     793 |     context.scalar_count = 0;
# >   794 |     context.result.task_id = task_id;
0x0000000000009c80 (+0x000099c0)  03c33040  ST_XD_XN_IMM.B64                X1, X19, #64
# [DWARF] common/pa_scheduler_core.h:795
# >   795 |     context.result.count = 0;
0x0000000000009c84 (+0x000099c4)  0f813900  STI_XN_IMM.B32                  X19, #72
# [DWARF] common/pa_scheduler_core.h:793
#     787 |     // fields are completed synchronously after the single callback builds args.
#     788 |     const uint32_t task_id = static_cast<uint32_t>(worker.local_index++);
#     789 |     context.self = &worker;
#     790 |     context.payload = &worker.payloads[task_id & kPayloadMask];
#     791 |     context.task_id = static_cast<int32_t>(task_id);
#     792 |     context.tensor_count = 0;
# >   793 |     context.scalar_count = 0;
0x0000000000009c88 (+0x000099c8)  09c02001  STP_XI_XJ_XN.B64                X0, X0, X2, #0
# [DWARF] common/pa_scheduler_core.h:798
#     794 |     context.result.task_id = task_id;
#     795 |     context.result.count = 0;
#     796 |     context.register_mask = 0;
#     797 |     context.output_bytes = 0;
# >   798 |     context.fanin_count = 0;
0x0000000000009c8c (+0x000099cc)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000009c90 (+0x000099d0)  0781ffff  MOVK                            X0, #65535, #2
# [DWARF] common/pa_scheduler_core.h:800
#     799 |     context.kernel_id = -1;
# >   800 |     context.won = false;
0x0000000000009c94 (+0x000099d4)  0f473300  STI_XN_IMM.B16                  X19, #408
# [DWARF] common/pa_scheduler_core.h:798
#     792 |     context.tensor_count = 0;
#     793 |     context.scalar_count = 0;
#     794 |     context.result.task_id = task_id;
#     795 |     context.result.count = 0;
#     796 |     context.register_mask = 0;
#     797 |     context.output_bytes = 0;
# >   798 |     context.fanin_count = 0;
0x0000000000009c98 (+0x000099d8)  07c1ffff  MOVK                            X0, #65535, #3
# [DWARF] common/pa_scheduler_core.h:802
#     799 |     context.kernel_id = -1;
#     800 |     context.won = false;
#     801 |     context.joint = false;
# >   802 |     context.joint_init = false;
0x0000000000009c9c (+0x000099dc)  0f073340  STI_XN_IMM.B8                   X19, #410
# [DWARF] common/pa_scheduler_core.h:798
#     792 |     context.tensor_count = 0;
#     793 |     context.scalar_count = 0;
#     794 |     context.result.task_id = task_id;
#     795 |     context.result.count = 0;
#     796 |     context.register_mask = 0;
#     797 |     context.output_bytes = 0;
# >   798 |     context.fanin_count = 0;
0x0000000000009ca0 (+0x000099e0)  03c13190  ST_XD_XN_IMM.B64                X0, X19, #400
# [DWARF] common/pa_scheduler_core.h:804
#     799 |     context.kernel_id = -1;
#     800 |     context.won = false;
#     801 |     context.joint = false;
#     802 |     context.joint_init = false;
#     803 |     context.joint_block = -1;
# >   804 |     context.joint_slot = -1;
0x0000000000009ca4 (+0x000099e4)  0700ffff  MOV_XD_IMM                      X0, #65535
0x0000000000009ca8 (+0x000099e8)  0741ffff  MOVK                            X0, #65535, #1
# [DWARF] common/pa_scheduler_core.h:803
#     797 |     context.output_bytes = 0;
#     798 |     context.fanin_count = 0;
#     799 |     context.kernel_id = -1;
#     800 |     context.won = false;
#     801 |     context.joint = false;
#     802 |     context.joint_init = false;
# >   803 |     context.joint_block = -1;
0x0000000000009cac (+0x000099ec)  0f873382  STI_XN_IMM.B32                  X19, #412
# [DWARF] common/pa_scheduler_core.h:804
# >   804 |     context.joint_slot = -1;
0x0000000000009cb0 (+0x000099f0)  03c131a0  ST_XD_XN_IMM.B64                X0, X19, #416
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x0000000000009cb4 (+0x000099f4)  02808880  MOV_XD_SPR.F32                  X0, SYS_CNT
# [DWARF] ccec/ccec_ops.h:0
# [SOURCE unavailable]
0x0000000000009cb8 (+0x000099f8)  03c3e920  ST_XD_XN_IMM.B64                X1, X30, #2336
0x0000000000009cbc (+0x000099fc)  1cd7e9b0  LD_XD_XN_IMM.B64                X11, X30, #2480
0x0000000000009cc0 (+0x00009a00)  1cd5e9a8  LD_XD_XN_IMM.B64                X10, X30, #2472
0x0000000000009cc4 (+0x00009a04)  1cd9e9a0  LD_XD_XN_IMM.B64                X12, X30, #2464
0x0000000000009cc8 (+0x00009a08)  1ccfe990  LD_XD_XN_IMM.B64                X7, X30, #2448
# [DWARF] common/pa_trace.h:257
#     251 |     TraceContext &trace, uint64_t end_cycle
#     252 | ) {
#     253 | #if PA_BUILD_SUBMIT_PMU
#     254 |     (void)trace;
#     255 |     (void)end_cycle;
#     256 | #else
# >   257 |     if (!trace.atomics_enabled || trace.poll_burst.active_mask == 0) return;
0x0000000000009ccc (+0x00009a0c)  03c1e928  ST_XD_XN_IMM.B64                X0, X30, #2344
0x0000000000009cd0 (+0x00009a10)  1c01355c  LD_XD_XN_IMM.B8                 X0, X19, #1372
0x0000000000009cd4 (+0x00009a14)  0000070e  CMP.S64.EQ                      X0, X14
0x0000000000009cd8 (+0x00009a18)  40200086  JUMPC                           #134
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000009cdc (+0x00009a1c)  1ca535d0  LD_XD_XN_IMM.B32                X18, X19, #1488
0x0000000000009ce0 (+0x00009a20)  02812a00  ZEROEXT.U32                     X0, X18
# [DWARF] common/pa_trace.h:257
#     251 |     TraceContext &trace, uint64_t end_cycle
#     252 | ) {
#     253 | #if PA_BUILD_SUBMIT_PMU
#     254 |     (void)trace;
#     255 |     (void)end_cycle;
#     256 | #else
# >   257 |     if (!trace.atomics_enabled || trace.poll_burst.active_mask == 0) return;
0x0000000000009ce4 (+0x00009a24)  0000070e  CMP.S64.EQ                      X0, X14
0x0000000000009ce8 (+0x00009a28)  40200082  JUMPC                           #130
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000009cec (+0x00009a2c)  0802c002  ADD_IMM.S64                     X1, X12, #2
0x0000000000009cf0 (+0x00009a30)  00c0a78a  AND.B64                         X0, X10, X15
0x0000000000009cf4 (+0x00009a34)  02c2020e  SHL.B64                         X1, #14
0x0000000000009cf8 (+0x00009a38)  0004b381  ADD.S64                         X2, X11, X7
0x0000000000009cfc (+0x00009a3c)  00040084  MADD.S64                        X2, X0, X1
0x0000000000009d00 (+0x00009a40)  07280000  MOV_XD_IMM                      X20, #0
0x0000000000009d04 (+0x00009a44)  00002081  ADD.S64                         X0, X2, X1
0x0000000000009d08 (+0x00009a48)  08800680  SUB_IMM.S64                     X0, X0, #1664
0x0000000000009d0c (+0x00009a4c)  08000000  ADD_IMM.S64                     X0, X0, #0
0x0000000000009d10 (+0x00009a50)  082a0548  ADD_IMM.S64                     X21, X0, #1352
0x0000000000009d14 (+0x00009a54)  1cb40558  LD_XD_XN_IMM.B32                X26, X0, #1368
0x0000000000009d18 (+0x00009a58)  0ceb5c80  LDP_XI_XJ_XN.B64                X21, X25, X21, #0
0x0000000000009d1c (+0x00009a5c)  072e0000  MOV_XD_IMM                      X23, #0
0x0000000000009d20 (+0x00009a60)  0726be78  MOV_XD_IMM                      X19, #48760
0x0000000000009d24 (+0x00009a64)  07670007  MOVK                            X19, #7, #1
0x0000000000009d28 (+0x00009a68)  07a70000  MOVK                            X19, #0, #2
0x0000000000009d2c (+0x00009a6c)  07e70000  MOVK                            X19, #0, #3
0x0000000000009d30 (+0x00009a70)  02040880  MOV_XD_SPR.S64                  X2, PC
0x0000000000009d34 (+0x00009a74)  00273101  ADD.S64                         X19, X19, X2
0x0000000000009d38 (+0x00009a78)  40000012  JUMP                            #18
0x0000000000009d3c (+0x00009a7c)  0804c002  ADD_IMM.S64                     X2, X12, #2
0x0000000000009d40 (+0x00009a80)  00c2a78a  AND.B64                         X1, X10, X15
0x0000000000009d44 (+0x00009a84)  02c4020e  SHL.B64                         X2, #14
0x0000000000009d48 (+0x00009a88)  0006b381  ADD.S64                         X3, X11, X7
0x0000000000009d4c (+0x00009a8c)  00061104  MADD.S64                        X3, X1, X2
# [DWARF] common/pa_trace.h:280
#     274 |             call_count, static_cast<uint32_t>(site)
#     275 |         );
#     276 |         if (written) {
#     277 |             if (trace.poll_batch_records == UINT64_MAX) {
#     278 |                 trace.atomic_counter_overflow = true;
#     279 |             } else {
# >   280 |                 ++trace.poll_batch_records;
0x0000000000009d50 (+0x00009a90)  08000001  ADD_IMM.S64                     X0, X0, #1
0x0000000000009d54 (+0x00009a94)  00023101  ADD.S64                         X1, X3, X2
0x0000000000009d58 (+0x00009a98)  08821680  SUB_IMM.S64                     X1, X1, #1664
0x0000000000009d5c (+0x00009a9c)  08021000  ADD_IMM.S64                     X1, X1, #0
0x0000000000009d60 (+0x00009aa0)  03c01578  ST_XD_XN_IMM.B64                X0, X1, #1400
# [DWARF] common/pa_trace.h:283
#     281 |             }
#     282 |         }
# >   283 |         trace.poll_burst.call_count[index] = 0;
0x0000000000009d64 (+0x00009aa4)  0f976700  STI_XN_IMM.B32                  X22, #1464
# [DWARF] common/pa_trace.h:262
#     256 | #else
#     257 |     if (!trace.atomics_enabled || trace.poll_burst.active_mask == 0) return;
#     258 |     const uint32_t active_mask = trace.poll_burst.active_mask;
#     259 |     // CCEC 默认会把固定 6-site 循环完整展开，再随几十个 phase 边界复制。
#     260 |     // 禁止展开只控制代码体积；循环次数、site 顺序和同 cycle 关闭语义不变。
#     261 |     PA_LOOP_NOUNROLL
# >   262 |     for (uint32_t index = 0; index < kAtomicPollBatchSiteCount; ++index) {
0x0000000000009d68 (+0x00009aa8)  082f7001  ADD_IMM.S64                     X23, X23, #1
0x0000000000009d6c (+0x00009aac)  08273004  ADD_IMM.S64                     X19, X19, #4
0x0000000000009d70 (+0x00009ab0)  0001749e  CMP.S64.NE                      X23, X9
0x0000000000009d74 (+0x00009ab4)  08294001  ADD_IMM.S64                     X20, X20, #1
0x0000000000009d78 (+0x00009ab8)  40200002  JUMPC                           #2
0x0000000000009d7c (+0x00009abc)  40000050  JUMP                            #80
# [DWARF] common/pa_trace.h:263
# >   263 |         const uint32_t bit = 1U << index;
0x0000000000009d80 (+0x00009ac0)  02814a00  ZEROEXT.U32                     X0, X20
# [DWARF] common/pa_trace.h:264
# >   264 |         if ((active_mask & bit) == 0) continue;
0x0000000000009d84 (+0x00009ac4)  02832a00  ZEROEXT.U32                     X1, X18
0x0000000000009d88 (+0x00009ac8)  024202c0  SHR.U64                         X1, X0, #0
0x0000000000009d8c (+0x00009acc)  00c0188a  AND.B64                         X0, X1, X17
0x0000000000009d90 (+0x00009ad0)  0000070e  CMP.S64.EQ                      X0, X14
0x0000000000009d94 (+0x00009ad4)  4020fff5  JUMPC                           #65525
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000009d98 (+0x00009ad8)  0804c002  ADD_IMM.S64                     X2, X12, #2
0x0000000000009d9c (+0x00009adc)  00c0a78a  AND.B64                         X0, X10, X15
0x0000000000009da0 (+0x00009ae0)  02c4020e  SHL.B64                         X2, #14
0x0000000000009da4 (+0x00009ae4)  0006b381  ADD.S64                         X3, X11, X7
0x0000000000009da8 (+0x00009ae8)  00060104  MADD.S64                        X3, X0, X2
# [DWARF] common/pa_trace.h:265
#     259 |     // CCEC 默认会把固定 6-site 循环完整展开，再随几十个 phase 边界复制。
#     260 |     // 禁止展开只控制代码体积；循环次数、site 顺序和同 cycle 关闭语义不变。
#     261 |     PA_LOOP_NOUNROLL
#     262 |     for (uint32_t index = 0; index < kAtomicPollBatchSiteCount; ++index) {
#     263 |         const uint32_t bit = 1U << index;
#     264 |         if ((active_mask & bit) == 0) continue;
# >   265 |         const uint32_t call_count = trace.poll_burst.call_count[index];
0x0000000000009dac (+0x00009aec)  02037800  MOV_XD_XN.S64                   X1, X23
0x0000000000009db0 (+0x00009af0)  00003101  ADD.S64                         X0, X3, X2
0x0000000000009db4 (+0x00009af4)  08800680  SUB_IMM.S64                     X0, X0, #1664
0x0000000000009db8 (+0x00009af8)  02c20202  SHL.B64                         X1, #2
0x0000000000009dbc (+0x00009afc)  08000000  ADD_IMM.S64                     X0, X0, #0
0x0000000000009dc0 (+0x00009b00)  002c0081  ADD.S64                         X22, X0, X1
0x0000000000009dc4 (+0x00009b04)  1c8b65b8  LD_XD_XN_IMM.B32                X5, X22, #1464
# [DWARF] common/pa_trace.h:266
# >   266 |         if (call_count == 0 || call_count > kAtomicPollCountMax) {
0x0000000000009dc8 (+0x00009b08)  07020000  MOV_XD_IMM                      X1, #0
0x0000000000009dcc (+0x00009b0c)  07430100  MOVK                            X1, #256, #1
0x0000000000009dd0 (+0x00009b10)  07040000  MOV_XD_IMM                      X2, #0
0x0000000000009dd4 (+0x00009b14)  0745ff00  MOVK                            X2, #65280, #1
0x0000000000009dd8 (+0x00009b18)  00025082  SUB.S64                         X1, X5, X1
0x0000000000009ddc (+0x00009b1c)  02821a00  ZEROEXT.U32                     X1, X1
0x0000000000009de0 (+0x00009b20)  0040113e  CMP.U64.GT                      X1, X2
0x0000000000009de4 (+0x00009b24)  40200003  JUMPC                           #3
# [DWARF] common/pa_trace.h:267
# >   267 |             trace.atomic_counter_overflow = true;
0x0000000000009de8 (+0x00009b28)  0f160001  STI_XN_IMM.B8                   X0, #1408
0x0000000000009dec (+0x00009b2c)  4000ffdf  JUMP                            #65503
# [DWARF] common/pa_trace.h:98
#      92 |         default:
#      93 |             return -1;
#      94 |     }
#      95 | }
#      96 | 
#      97 | PA_DEVICE AtomicSite TraceAtomicPollBatchSite(uint32_t index) {
# >    98 |     switch (index) {
0x0000000000009df0 (+0x00009b30)  02814a00  ZEROEXT.U32                     X0, X20
0x0000000000009df4 (+0x00009b34)  07020005  MOV_XD_IMM                      X1, #5
0x0000000000009df8 (+0x00009b38)  004000be  CMP.U64.GT                      X0, X1
0x0000000000009dfc (+0x00009b3c)  070c000f  MOV_XD_IMM                      X6, #15
0x0000000000009e00 (+0x00009b40)  40200002  JUMPC                           #2
0x0000000000009e04 (+0x00009b44)  1c8d3000  LD_XD_XN_IMM.B32                X6, X19, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000009e08 (+0x00009b48)  0804c002  ADD_IMM.S64                     X2, X12, #2
# [DWARF] common/pa_trace.h:271
#     265 |         const uint32_t call_count = trace.poll_burst.call_count[index];
#     266 |         if (call_count == 0 || call_count > kAtomicPollCountMax) {
#     267 |             trace.atomic_counter_overflow = true;
#     268 |             continue;
#     269 |         }
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
# >   271 |         const bool written = WritePollBatchRecordRaw(
0x0000000000009e0c (+0x00009b4c)  1cc9e928  LD_XD_XN_IMM.B64                X4, X30, #2344
0x0000000000009e10 (+0x00009b50)  00c2a78a  AND.B64                         X1, X10, X15
0x0000000000009e14 (+0x00009b54)  02c4020e  SHL.B64                         X2, #14
0x0000000000009e18 (+0x00009b58)  0006b381  ADD.S64                         X3, X11, X7
0x0000000000009e1c (+0x00009b5c)  00061104  MADD.S64                        X3, X1, X2
# [DWARF] common/pa_trace.h:273
#     272 |             trace.core, trace.records, trace.capacity,
# >   273 |             trace.poll_burst.start_cycle[index], end_cycle,
0x0000000000009e20 (+0x00009b60)  02017800  MOV_XD_XN.S64                   X0, X23
0x0000000000009e24 (+0x00009b64)  00023101  ADD.S64                         X1, X3, X2
0x0000000000009e28 (+0x00009b68)  08821680  SUB_IMM.S64                     X1, X1, #1664
0x0000000000009e2c (+0x00009b6c)  02c00203  SHL.B64                         X0, #3
0x0000000000009e30 (+0x00009b70)  08301000  ADD_IMM.S64                     X24, X1, #0
0x0000000000009e34 (+0x00009b74)  00018001  ADD.S64                         X0, X24, X0
0x0000000000009e38 (+0x00009b78)  1cc60588  LD_XD_XN_IMM.B64                X3, X0, #1416
# [DWARF] common/pa_trace.h:271
#     265 |         const uint32_t call_count = trace.poll_burst.call_count[index];
#     266 |         if (call_count == 0 || call_count > kAtomicPollCountMax) {
#     267 |             trace.atomic_counter_overflow = true;
#     268 |             continue;
#     269 |         }
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
# >   271 |         const bool written = WritePollBatchRecordRaw(
0x0000000000009e3c (+0x00009b7c)  02015800  MOV_XD_XN.S64                   X0, X21
0x0000000000009e40 (+0x00009b80)  02039800  MOV_XD_XN.S64                   X1, X25
0x0000000000009e44 (+0x00009b84)  0205a800  MOV_XD_XN.S64                   X2, X26
0x0000000000009e48 (+0x00009b88)  070e771a  MOV_XD_IMM                      X7, #30490
0x0000000000009e4c (+0x00009b8c)  074f0000  MOVK                            X7, #0, #1
0x0000000000009e50 (+0x00009b90)  078f0000  MOVK                            X7, #0, #2
0x0000000000009e54 (+0x00009b94)  40427000  CALL                            X7, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000009e58 (+0x00009b98)  1ccfe990  LD_XD_XN_IMM.B64                X7, X30, #2448
0x0000000000009e5c (+0x00009b9c)  071c0000  MOV_XD_IMM                      X14, #0
0x0000000000009e60 (+0x00009ba0)  1cd9e9a0  LD_XD_XN_IMM.B64                X12, X30, #2464
0x0000000000009e64 (+0x00009ba4)  071e7fff  MOV_XD_IMM                      X15, #32767
