# schema=pa_final_linked_disassembly/v1
# variant=original
# final_elf=pa_scheduler_kernel.o
# final_elf_sha256=76c961f846c7efc89b40942c3a8113530a6da2c7bfbdc601615852f8c5b79cfc
# final_text_address=0x0
# final_text_size=780344
# final_text_sha256=018ae7dc29ce78249b2e3bff84b0faa2e671e2448c9316719fb48bc59139d2fc
# symbol=pa_scheduler_0_mix_aic
# binding=GLOBAL
# final_pc=0x250
# size=387044
# instruction_count=96761
# encoded_word_count=96761
# body_sha256=d7337df3d1786f2236031aaa4d252ca82696e3a563e11ce56d9b98495bbf1aea
# decoder=$ASCEND_HOME_PATH/x86_64-linux/simulator/dav_3510/lib/libpem_davinci.so
# decoder_sha256=29835d2439d6dd464d34a212ad4bbd5c29af6a38465da09a6c273401d9a96dcb
# decoder_mode=scalar
# columns=final_pc function_relative_offset machine_word instruction
# annotation_schema=pa_source_annotated_disassembly/v1
# annotation_rule=DWARF supplies only file:line; SOURCE rows are copied from local source files
# annotation_warning=comments have source context only and do not own an exact machine address
# annotation_instruction_slice=87055:87776
#
# [DWARF] common/pa_trace.h:549
#     543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
# >   549 |         core.dropped = core.dropped + 1;
0x000000000005528c (+0x0005503c)  1c801004  LD_XD_XN_IMM.B32                X0, X1, #4
0x0000000000055290 (+0x00055040)  08000001  ADD_IMM.S64                     X0, X0, #1
0x0000000000055294 (+0x00055044)  03801004  ST_XD_XN_IMM.B32                X0, X1, #4
0x0000000000055298 (+0x00055048)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x000000000005529c (+0x0005504c)  0700ddbf  MOV_XD_IMM                      X0, #56767
0x00000000000552a0 (+0x00055050)  0741ffff  MOVK                            X0, #65535, #1
0x00000000000552a4 (+0x00055054)  0781ffff  MOVK                            X0, #65535, #2
0x00000000000552a8 (+0x00055058)  40020000  JUMP                            X0, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000552ac (+0x0005505c)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x00000000000552b0 (+0x00055060)  07000000  MOV_XD_IMM                      X0, #0
0x00000000000552b4 (+0x00055064)  03d3ea40  ST_XD_XN_IMM.B64                X9, X30, #2624
0x00000000000552b8 (+0x00055068)  1ce5eb48  LD_XD_XN_IMM.B64                X18, X30, #2888
0x00000000000552bc (+0x0005506c)  03c1e9e8  ST_XD_XN_IMM.B64                X0, X30, #2536
0x00000000000552c0 (+0x00055070)  07000000  MOV_XD_IMM                      X0, #0
0x00000000000552c4 (+0x00055074)  1cebeb40  LD_XD_XN_IMM.B64                X21, X30, #2880
0x00000000000552c8 (+0x00055078)  03c1ea90  ST_XD_XN_IMM.B64                X0, X30, #2704
0x00000000000552cc (+0x0005507c)  07000000  MOV_XD_IMM                      X0, #0
0x00000000000552d0 (+0x00055080)  1ce7eb00  LD_XD_XN_IMM.B64                X19, X30, #2816
0x00000000000552d4 (+0x00055084)  03c1e9c8  ST_XD_XN_IMM.B64                X0, X30, #2504
0x00000000000552d8 (+0x00055088)  07000000  MOV_XD_IMM                      X0, #0
0x00000000000552dc (+0x0005508c)  1cf9eb70  LD_XD_XN_IMM.B64                X28, X30, #2928
0x00000000000552e0 (+0x00055090)  03c1e988  ST_XD_XN_IMM.B64                X0, X30, #2440
0x00000000000552e4 (+0x00055094)  07000000  MOV_XD_IMM                      X0, #0
0x00000000000552e8 (+0x00055098)  03c1e980  ST_XD_XN_IMM.B64                X0, X30, #2432
0x00000000000552ec (+0x0005509c)  07000000  MOV_XD_IMM                      X0, #0
0x00000000000552f0 (+0x000550a0)  03c1e9c0  ST_XD_XN_IMM.B64                X0, X30, #2496
0x00000000000552f4 (+0x000550a4)  07000000  MOV_XD_IMM                      X0, #0
0x00000000000552f8 (+0x000550a8)  03c1e9b8  ST_XD_XN_IMM.B64                X0, X30, #2488
0x00000000000552fc (+0x000550ac)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000055300 (+0x000550b0)  03c1e9d0  ST_XD_XN_IMM.B64                X0, X30, #2512
0x0000000000055304 (+0x000550b4)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000055308 (+0x000550b8)  03c1e9d8  ST_XD_XN_IMM.B64                X0, X30, #2520
0x000000000005530c (+0x000550bc)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000055310 (+0x000550c0)  03c1e9e0  ST_XD_XN_IMM.B64                X0, X30, #2528
0x0000000000055314 (+0x000550c4)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000055318 (+0x000550c8)  03c1e9f0  ST_XD_XN_IMM.B64                X0, X30, #2544
0x000000000005531c (+0x000550cc)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000055320 (+0x000550d0)  03c1ea68  ST_XD_XN_IMM.B64                X0, X30, #2664
0x0000000000055324 (+0x000550d4)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000055328 (+0x000550d8)  03c1ea58  ST_XD_XN_IMM.B64                X0, X30, #2648
0x000000000005532c (+0x000550dc)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000055330 (+0x000550e0)  03c1ea20  ST_XD_XN_IMM.B64                X0, X30, #2592
0x0000000000055334 (+0x000550e4)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000055338 (+0x000550e8)  03c1e9f8  ST_XD_XN_IMM.B64                X0, X30, #2552
0x000000000005533c (+0x000550ec)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000055340 (+0x000550f0)  03c1e990  ST_XD_XN_IMM.B64                X0, X30, #2448
0x0000000000055344 (+0x000550f4)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000055348 (+0x000550f8)  03c1ea48  ST_XD_XN_IMM.B64                X0, X30, #2632
0x000000000005534c (+0x000550fc)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000055350 (+0x00055100)  03c1ea60  ST_XD_XN_IMM.B64                X0, X30, #2656
0x0000000000055354 (+0x00055104)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000055358 (+0x00055108)  03c1ea50  ST_XD_XN_IMM.B64                X0, X30, #2640
0x000000000005535c (+0x0005510c)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000055360 (+0x00055110)  03c1ea28  ST_XD_XN_IMM.B64                X0, X30, #2600
0x0000000000055364 (+0x00055114)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000055368 (+0x00055118)  03c1e998  ST_XD_XN_IMM.B64                X0, X30, #2456
0x000000000005536c (+0x0005511c)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000055370 (+0x00055120)  03c1e9a0  ST_XD_XN_IMM.B64                X0, X30, #2464
0x0000000000055374 (+0x00055124)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000055378 (+0x00055128)  03c1e9a8  ST_XD_XN_IMM.B64                X0, X30, #2472
0x000000000005537c (+0x0005512c)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000055380 (+0x00055130)  03c1eae8  ST_XD_XN_IMM.B64                X0, X30, #2792
0x0000000000055384 (+0x00055134)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000055388 (+0x00055138)  03c1eb28  ST_XD_XN_IMM.B64                X0, X30, #2856
0x000000000005538c (+0x0005513c)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000055390 (+0x00055140)  03c1ead0  ST_XD_XN_IMM.B64                X0, X30, #2768
0x0000000000055394 (+0x00055144)  07000000  MOV_XD_IMM                      X0, #0
0x0000000000055398 (+0x00055148)  07320000  MOV_XD_IMM                      X25, #0
0x000000000005539c (+0x0005514c)  03c1e9b0  ST_XD_XN_IMM.B64                X0, X30, #2480
0x00000000000553a0 (+0x00055150)  07340000  MOV_XD_IMM                      X26, #0
0x00000000000553a4 (+0x00055154)  070c0000  MOV_XD_IMM                      X6, #0
0x00000000000553a8 (+0x00055158)  0216f800  MOV_XD_XN.S64                   X11, X15
0x00000000000553ac (+0x0005515c)  021d7800  MOV_XD_XN.S64                   X14, X23
0x00000000000553b0 (+0x00055160)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x00000000000553b4 (+0x00055164)  070002ea  MOV_XD_IMM                      X0, #746
0x00000000000553b8 (+0x00055168)  07410000  MOVK                            X0, #0, #1
0x00000000000553bc (+0x0005516c)  07810000  MOVK                            X0, #0, #2
0x00000000000553c0 (+0x00055170)  40020000  JUMP                            X0, #0
0x00000000000553c4 (+0x00055174)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x00000000000553c8 (+0x00055178)  02808880  MOV_XD_SPR.F32                  X0, SYS_CNT
# [DWARF] ccec/ccec_ops.h:0
# [SOURCE unavailable]
0x00000000000553cc (+0x0005517c)  07040001  MOV_XD_IMM                      X2, #1
0x00000000000553d0 (+0x00055180)  07060001  MOV_XD_IMM                      X3, #1
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:45
#      39 |   CCE_INTRINSIC[aicore] FTYPE atomicSub(__gm__ FTYPE *base, FTYPE inc) {       \
#      40 |     return __builtin_cce_atom_add_G_##ADD_SUFFIX(base, -inc,                   \
#      41 |                                                   (uint32_t)L2Cache);          \
#      42 |   }
#      43 | 
#      44 | __CCE__ATOM_G_CAS_EXCH_INT32(uint32_t, u32, u32)
# >    45 | __CCE__ATOM_G_CAS_EXCH_INT32(int32_t, s32, s32)
0x00000000000553d4 (+0x00055184)  50109c40  ATOM                            XN, XM, XD, EXCH
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:0
# [SOURCE unavailable]
0x00000000000553d8 (+0x00055188)  02063080  NEG.S64                         X3, X3
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x00000000000553dc (+0x0005518c)  02828880  MOV_XD_SPR.F32                  X1, SYS_CNT
# [DWARF] common/pa_trace.h:236
#     230 |            (encoded_retries << kAtomicRetriesShift);
#     231 | }
#     232 | 
#     233 | PA_DEVICE void CountAtomicCall(
#     234 |     TraceContext &trace, WorkerResult &result, bool poll_batch
#     235 | ) {
# >   236 |     if (result.atomic_trace_calls == UINT64_MAX) {
0x00000000000553e0 (+0x00055190)  0001918e  CMP.S64.EQ                      X25, X3
0x00000000000553e4 (+0x00055194)  00da2689  SEL.B64                         X13, X2, X13
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x00000000000553e8 (+0x00055198)  1cc5eb40  LD_XD_XN_IMM.B64                X2, X30, #2880
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000553ec (+0x0005519c)  08099001  ADD_IMM.S64                     X4, X25, #1
# [DWARF] common/pa_trace.h:236
#     230 |            (encoded_retries << kAtomicRetriesShift);
#     231 | }
#     232 | 
#     233 | PA_DEVICE void CountAtomicCall(
#     234 |     TraceContext &trace, WorkerResult &result, bool poll_batch
#     235 | ) {
# >   236 |     if (result.atomic_trace_calls == UINT64_MAX) {
0x00000000000553f0 (+0x000551a0)  00e63209  SEL.B64                         X19, X3, X4
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x00000000000553f4 (+0x000551a4)  02162900  MOV_SPR_XN.S64                  CONDITION_FLAG, X2
0x00000000000553f8 (+0x000551a8)  40200151  JUMPC                           #337
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000553fc (+0x000551ac)  1cc9eb80  LD_XD_XN_IMM.B64                X4, X30, #2944
# [DWARF] common/pa_trace.h:548
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
#     543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
#     547 |     const uint32_t slot = core.count;
# >   548 |     if (slot >= trace.capacity) {
0x0000000000055400 (+0x000551b0)  1cc7eb98  LD_XD_XN_IMM.B64                X3, X30, #2968
# [DWARF] common/pa_trace.h:547
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
#     543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
# >   547 |     const uint32_t slot = core.count;
0x0000000000055404 (+0x000551b4)  1c844000  LD_XD_XN_IMM.B32                X2, X4, #0
# [DWARF] common/pa_trace.h:548
# >   548 |     if (slot >= trace.capacity) {
0x0000000000055408 (+0x000551b8)  004021ae  CMP.U64.LT                      X2, X3
0x000000000005540c (+0x000551bc)  40200002  JUMPC                           #2
0x0000000000055410 (+0x000551c0)  40000146  JUMP                            #326
# [DWARF] common/pa_trace.h:552
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
# >   552 |     PA_GM TraceRecord &record = trace.records[slot];
0x0000000000055414 (+0x000551c4)  1ccbeb90  LD_XD_XN_IMM.B64                X5, X30, #2960
0x0000000000055418 (+0x000551c8)  02062800  MOV_XD_XN.S64                   X3, X2
0x000000000005541c (+0x000551cc)  02c60206  SHL.B64                         X3, #6
0x0000000000055420 (+0x000551d0)  00065181  ADD.S64                         X3, X5, X3
# [DWARF] common/pa_trace.h:553
# >   553 |     record.start_cycle = start_cycle;
0x0000000000055424 (+0x000551d4)  09c03081  STP_XI_XJ_XN.B64                X0, X1, X3, #0
0x0000000000055428 (+0x000551d8)  07000001  MOV_XD_IMM                      X0, #1
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000005542c (+0x000551dc)  02000080  NEG.S64                         X0, X0
0x0000000000055430 (+0x000551e0)  0702000e  MOV_XD_IMM                      X1, #14
# [DWARF] common/pa_trace.h:555
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
# >   555 |     record.task_id = task_id;
0x0000000000055434 (+0x000551e4)  09c030a1  STP_XI_XJ_XN.B64                X0, X1, X3, #16
# [DWARF] common/pa_trace.h:559
#     556 |     record.function_id = function_id;
#     557 |     record.phase = static_cast<int32_t>(trace_phase);
#     558 |     record.lane = trace.lane;
# >   559 |     record.block_id = trace.block_id;
0x0000000000055438 (+0x000551e8)  08003020  ADD_IMM.S64                     X0, X3, #32
0x000000000005543c (+0x000551ec)  099e0781  STP_XI_XJ_XN.B32                X15, X15, X0, #0
0x0000000000055440 (+0x000551f0)  07000001  MOV_XD_IMM                      X0, #1
0x0000000000055444 (+0x000551f4)  07810003  MOVK                            X0, #3, #2
# [DWARF] common/pa_trace.h:561
#     560 |     record.core_idx = trace.core_idx;
# >   561 |     record.flags = flags;
0x0000000000055448 (+0x000551f8)  03c03028  ST_XD_XN_IMM.B64                X0, X3, #40
# [DWARF] common/pa_trace.h:565
#     562 |     record.auxiliary = auxiliary;
#     563 |     // count 最后更新，使其始终指向下一空槽；单写者条件下无需 reserve/commit 两阶段。
#     564 |     // 最终 FlushTraceCore 会把记录体先于该计数一并导出。
# >   565 |     core.count = slot + 1;
0x000000000005544c (+0x000551fc)  08002001  ADD_IMM.S64                     X0, X2, #1
0x0000000000055450 (+0x00055200)  03804000  ST_XD_XN_IMM.B32                X0, X4, #0
0x0000000000055454 (+0x00055204)  4000013a  JUMP                            #314
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055458 (+0x00055208)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x000000000005545c (+0x0005520c)  1ce5eb48  LD_XD_XN_IMM.B64                X18, X30, #2888
0x0000000000055460 (+0x00055210)  02179800  MOV_XD_XN.S64                   X11, X25
0x0000000000055464 (+0x00055214)  40000002  JUMP                            #2
0x0000000000055468 (+0x00055218)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
# [DWARF] common/pa_scheduler_core.h:1435
#    1429 |                 break;
#    1430 |             }
#    1431 | #else
#    1432 |             BuildAllocArgs(orchestration, args, batch);
#    1433 |             ++stats.result.context_reads;
#    1434 |             stats.result.views_created += 2;
# >  1435 |             stats.result.tensor_args_added += 3;
0x000000000005546c (+0x0005521c)  1cc1e930  LD_XD_XN_IMM.B64                X0, X30, #2352
# [DWARF] common/pa_trace.h:430
#     424 |     (void)result;
#     425 |     (void)task_id;
#     426 |     (void)site;
#     427 |     (void)result_used;
#     428 |     return Ops::Exchange(address, value);
#     429 | #else
# >   430 |     if (!trace.atomics_enabled) return Ops::Exchange(address, value);
0x0000000000055470 (+0x00055220)  02172900  MOV_SPR_XN.S64                  CONDITION_FLAG, X18
# [DWARF] common/pa_scheduler_core.h:1435
#    1429 |                 break;
#    1430 |             }
#    1431 | #else
#    1432 |             BuildAllocArgs(orchestration, args, batch);
#    1433 |             ++stats.result.context_reads;
#    1434 |             stats.result.views_created += 2;
# >  1435 |             stats.result.tensor_args_added += 3;
0x0000000000055474 (+0x00055224)  08000003  ADD_IMM.S64                     X0, X0, #3
# [DWARF] common/pa_trace.h:430
#     424 |     (void)result;
#     425 |     (void)task_id;
#     426 |     (void)site;
#     427 |     (void)result_used;
#     428 |     return Ops::Exchange(address, value);
#     429 | #else
# >   430 |     if (!trace.atomics_enabled) return Ops::Exchange(address, value);
0x0000000000055478 (+0x00055228)  03c1ea08  ST_XD_XN_IMM.B64                X0, X30, #2568
0x000000000005547c (+0x0005522c)  40200002  JUMPC                           #2
0x0000000000055480 (+0x00055230)  4000001d  JUMP                            #29
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055484 (+0x00055234)  03d7eb68  ST_XD_XN_IMM.B64                X11, X30, #2920
0x0000000000055488 (+0x00055238)  1cc5e928  LD_XD_XN_IMM.B64                X2, X30, #2344
0x000000000005548c (+0x0005523c)  40000057  JUMP                            #87
0x0000000000055490 (+0x00055240)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000055494 (+0x00055244)  1ce5eb48  LD_XD_XN_IMM.B64                X18, X30, #2888
0x0000000000055498 (+0x00055248)  1cd7eb68  LD_XD_XN_IMM.B64                X11, X30, #2920
0x000000000005549c (+0x0005524c)  40000002  JUMP                            #2
0x00000000000554a0 (+0x00055250)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
# [DWARF] common/pa_scheduler_core.h:753
#     747 |             // Register 后到 Submit.end 由离线 submit_tail_gap 补集展示；排他
#     748 |             // 报告汇总为 submit_tail_residual，避免零时长 marker 增加 raw
#     749 |             // 体积和 trace-buffer 写开销。
#     750 |         }
#     751 |     }
#     752 | 
# >   753 |     ++stats.result.submits;
0x00000000000554a4 (+0x00055254)  1cc1e940  LD_XD_XN_IMM.B64                X0, X30, #2368
# [DWARF] common/pa_trace.h:430
#     424 |     (void)result;
#     425 |     (void)task_id;
#     426 |     (void)site;
#     427 |     (void)result_used;
#     428 |     return Ops::Exchange(address, value);
#     429 | #else
# >   430 |     if (!trace.atomics_enabled) return Ops::Exchange(address, value);
0x00000000000554a8 (+0x00055258)  02172900  MOV_SPR_XN.S64                  CONDITION_FLAG, X18
0x00000000000554ac (+0x0005525c)  1ccbe948  LD_XD_XN_IMM.B64                X5, X30, #2376
# [DWARF] common/pa_scheduler_core.h:1444
#    1438 |                 )) {
#    1439 |                 break;
#    1440 |             }
#    1441 |             AcceptTaskOutputs(orchestration, TaskKind::Alloc, context.result);
#    1442 | 
#    1443 |             BuildQkArgs(orchestration, args, batch);
# >  1444 |             ++stats.result.dynamic_create_infos;
0x00000000000554b0 (+0x00055260)  1cc7ea40  LD_XD_XN_IMM.B64                X3, X30, #2624
# [DWARF] common/pa_scheduler_core.h:753
#     747 |             // Register 后到 Submit.end 由离线 submit_tail_gap 补集展示；排他
#     748 |             // 报告汇总为 submit_tail_residual，避免零时长 marker 增加 raw
#     749 |             // 体积和 trace-buffer 写开销。
#     750 |         }
#     751 |     }
#     752 | 
# >   753 |     ++stats.result.submits;
0x00000000000554b4 (+0x00055264)  08020001  ADD_IMM.S64                     X1, X0, #1
# [DWARF] common/pa_scheduler_core.h:1447
#    1441 |             AcceptTaskOutputs(orchestration, TaskKind::Alloc, context.result);
#    1442 | 
#    1443 |             BuildQkArgs(orchestration, args, batch);
#    1444 |             ++stats.result.dynamic_create_infos;
#    1445 |             ++stats.result.arg_resets;
#    1446 |             stats.result.tensor_args_added += 4;
# >  1447 |             stats.result.scalar_args_added += 2;
0x00000000000554b8 (+0x00055268)  1cc1e928  LD_XD_XN_IMM.B64                X0, X30, #2344
0x00000000000554bc (+0x0005526c)  08040002  ADD_IMM.S64                     X2, X0, #2
# [DWARF] common/pa_scheduler_core.h:1446
#    1440 |             }
#    1441 |             AcceptTaskOutputs(orchestration, TaskKind::Alloc, context.result);
#    1442 | 
#    1443 |             BuildQkArgs(orchestration, args, batch);
#    1444 |             ++stats.result.dynamic_create_infos;
#    1445 |             ++stats.result.arg_resets;
# >  1446 |             stats.result.tensor_args_added += 4;
0x00000000000554c0 (+0x00055270)  1cc1e930  LD_XD_XN_IMM.B64                X0, X30, #2352
0x00000000000554c4 (+0x00055274)  08000007  ADD_IMM.S64                     X0, X0, #7
0x00000000000554c8 (+0x00055278)  03c1ea08  ST_XD_XN_IMM.B64                X0, X30, #2568
0x00000000000554cc (+0x0005527c)  07000000  MOV_XD_IMM                      X0, #0
# [DWARF] common/pa_scheduler_core.h:1445
#    1439 |                 break;
#    1440 |             }
#    1441 |             AcceptTaskOutputs(orchestration, TaskKind::Alloc, context.result);
#    1442 | 
#    1443 |             BuildQkArgs(orchestration, args, batch);
#    1444 |             ++stats.result.dynamic_create_infos;
# >  1445 |             ++stats.result.arg_resets;
0x00000000000554d0 (+0x00055280)  02ca0440  SBITSET.B64                     X5, X0
# [DWARF] common/pa_scheduler_core.h:1444
#    1438 |                 )) {
#    1439 |                 break;
#    1440 |             }
#    1441 |             AcceptTaskOutputs(orchestration, TaskKind::Alloc, context.result);
#    1442 | 
#    1443 |             BuildQkArgs(orchestration, args, batch);
# >  1444 |             ++stats.result.dynamic_create_infos;
0x00000000000554d4 (+0x00055284)  02c60440  SBITSET.B64                     X3, X0
# [DWARF] common/pa_trace.h:430
#     424 |     (void)result;
#     425 |     (void)task_id;
#     426 |     (void)site;
#     427 |     (void)result_used;
#     428 |     return Ops::Exchange(address, value);
#     429 | #else
# >   430 |     if (!trace.atomics_enabled) return Ops::Exchange(address, value);
0x00000000000554d8 (+0x00055288)  03c7ea40  ST_XD_XN_IMM.B64                X3, X30, #2624
0x00000000000554dc (+0x0005528c)  40200002  JUMPC                           #2
0x00000000000554e0 (+0x00055290)  40000051  JUMP                            #81
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000554e4 (+0x00055294)  03cbe948  ST_XD_XN_IMM.B64                X5, X30, #2376
0x00000000000554e8 (+0x00055298)  03f3eb70  ST_XD_XN_IMM.B64                X25, X30, #2928
0x00000000000554ec (+0x0005529c)  03c3e940  ST_XD_XN_IMM.B64                X1, X30, #2368
0x00000000000554f0 (+0x000552a0)  40000086  JUMP                            #134
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x00000000000554f4 (+0x000552a4)  02808880  MOV_XD_SPR.F32                  X0, SYS_CNT
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:45
#      39 |   CCE_INTRINSIC[aicore] FTYPE atomicSub(__gm__ FTYPE *base, FTYPE inc) {       \
#      40 |     return __builtin_cce_atom_add_G_##ADD_SUFFIX(base, -inc,                   \
#      41 |                                                   (uint32_t)L2Cache);          \
#      42 |   }
#      43 | 
#      44 | __CCE__ATOM_G_CAS_EXCH_INT32(uint32_t, u32, u32)
# >    45 | __CCE__ATOM_G_CAS_EXCH_INT32(int32_t, s32, s32)
0x00000000000554f8 (+0x000552a8)  1cc3eac8  LD_XD_XN_IMM.B64                X1, X30, #2760
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:0
# [SOURCE unavailable]
0x00000000000554fc (+0x000552ac)  07040001  MOV_XD_IMM                      X2, #1
0x0000000000055500 (+0x000552b0)  07060001  MOV_XD_IMM                      X3, #1
0x0000000000055504 (+0x000552b4)  02063080  NEG.S64                         X3, X3
# [DWARF] common/pa_trace.h:236
#     230 |            (encoded_retries << kAtomicRetriesShift);
#     231 | }
#     232 | 
#     233 | PA_DEVICE void CountAtomicCall(
#     234 |     TraceContext &trace, WorkerResult &result, bool poll_batch
#     235 | ) {
# >   236 |     if (result.atomic_trace_calls == UINT64_MAX) {
0x0000000000055508 (+0x000552b8)  0001318e  CMP.S64.EQ                      X19, X3
0x000000000005550c (+0x000552bc)  08093001  ADD_IMM.S64                     X4, X19, #1
0x0000000000055510 (+0x000552c0)  00e63209  SEL.B64                         X19, X3, X4
0x0000000000055514 (+0x000552c4)  00da2689  SEL.B64                         X13, X2, X13
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x0000000000055518 (+0x000552c8)  02175900  MOV_SPR_XN.S64                  CONDITION_FLAG, X21
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:45
#      39 |   CCE_INTRINSIC[aicore] FTYPE atomicSub(__gm__ FTYPE *base, FTYPE inc) {       \
#      40 |     return __builtin_cce_atom_add_G_##ADD_SUFFIX(base, -inc,                   \
#      41 |                                                   (uint32_t)L2Cache);          \
#      42 |   }
#      43 | 
#      44 | __CCE__ATOM_G_CAS_EXCH_INT32(uint32_t, u32, u32)
# >    45 | __CCE__ATOM_G_CAS_EXCH_INT32(int32_t, s32, s32)
0x000000000005551c (+0x000552cc)  50108440  ATOM                            XN, XM, XD, EXCH
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x0000000000055520 (+0x000552d0)  02828880  MOV_XD_SPR.F32                  X1, SYS_CNT
# [DWARF] ccec/ccec_ops.h:0
# [SOURCE unavailable]
0x0000000000055524 (+0x000552d4)  1ce3eb10  LD_XD_XN_IMM.B64                X17, X30, #2832
0x0000000000055528 (+0x000552d8)  1ce9eb80  LD_XD_XN_IMM.B64                X20, X30, #2944
0x000000000005552c (+0x000552dc)  1cddeb98  LD_XD_XN_IMM.B64                X14, X30, #2968
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x0000000000055530 (+0x000552e0)  40200002  JUMPC                           #2
0x0000000000055534 (+0x000552e4)  400000b9  JUMP                            #185
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055538 (+0x000552e8)  1cc1e940  LD_XD_XN_IMM.B64                X0, X30, #2368
0x000000000005553c (+0x000552ec)  02398800  MOV_XD_XN.S64                   X28, X24
0x0000000000055540 (+0x000552f0)  1cd3ea30  LD_XD_XN_IMM.B64                X9, X30, #2608
0x0000000000055544 (+0x000552f4)  1cf3eb70  LD_XD_XN_IMM.B64                X25, X30, #2928
0x0000000000055548 (+0x000552f8)  03c1ea18  ST_XD_XN_IMM.B64                X0, X30, #2584
0x000000000005554c (+0x000552fc)  1cc1e948  LD_XD_XN_IMM.B64                X0, X30, #2376
0x0000000000055550 (+0x00055300)  03c1ea00  ST_XD_XN_IMM.B64                X0, X30, #2560
0x0000000000055554 (+0x00055304)  1cc1e928  LD_XD_XN_IMM.B64                X0, X30, #2344
0x0000000000055558 (+0x00055308)  03c1ea10  ST_XD_XN_IMM.B64                X0, X30, #2576
0x000000000005555c (+0x0005530c)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000055560 (+0x00055310)  070001bb  MOV_XD_IMM                      X0, #443
0x0000000000055564 (+0x00055314)  07410000  MOVK                            X0, #0, #1
0x0000000000055568 (+0x00055318)  07810000  MOVK                            X0, #0, #2
0x000000000005556c (+0x0005531c)  40020000  JUMP                            X0, #0
0x0000000000055570 (+0x00055320)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000055574 (+0x00055324)  1ce5eb48  LD_XD_XN_IMM.B64                X18, X30, #2888
0x0000000000055578 (+0x00055328)  1cd7eb68  LD_XD_XN_IMM.B64                X11, X30, #2920
0x000000000005557c (+0x0005532c)  40000002  JUMP                            #2
0x0000000000055580 (+0x00055330)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
# [DWARF] common/pa_scheduler_core.h:753
#     747 |             // Register 后到 Submit.end 由离线 submit_tail_gap 补集展示；排他
#     748 |             // 报告汇总为 submit_tail_residual，避免零时长 marker 增加 raw
#     749 |             // 体积和 trace-buffer 写开销。
#     750 |         }
#     751 |     }
#     752 | 
# >   753 |     ++stats.result.submits;
0x0000000000055584 (+0x00055334)  1cc1e940  LD_XD_XN_IMM.B64                X0, X30, #2368
0x0000000000055588 (+0x00055338)  07040001  MOV_XD_IMM                      X2, #1
0x000000000005558c (+0x0005533c)  1ccbe948  LD_XD_XN_IMM.B64                X5, X30, #2376
# [DWARF] common/pa_trace.h:430
#     424 |     (void)result;
#     425 |     (void)task_id;
#     426 |     (void)site;
#     427 |     (void)result_used;
#     428 |     return Ops::Exchange(address, value);
#     429 | #else
# >   430 |     if (!trace.atomics_enabled) return Ops::Exchange(address, value);
0x0000000000055590 (+0x00055340)  02172900  MOV_SPR_XN.S64                  CONDITION_FLAG, X18
# [DWARF] common/pa_scheduler_core.h:753
#     747 |             // Register 后到 Submit.end 由离线 submit_tail_gap 补集展示；排他
#     748 |             // 报告汇总为 submit_tail_residual，避免零时长 marker 增加 raw
#     749 |             // 体积和 trace-buffer 写开销。
#     750 |         }
#     751 |     }
#     752 | 
# >   753 |     ++stats.result.submits;
0x0000000000055594 (+0x00055344)  08020002  ADD_IMM.S64                     X1, X0, #2
# [DWARF] common/pa_scheduler_core.h:1459
#    1453 |             AcceptTaskOutputs(orchestration, TaskKind::Qk, context.result);
#    1454 | 
#    1455 |             BuildSfArgs(orchestration, args);
#    1456 |             ++stats.result.dynamic_create_infos;
#    1457 |             ++stats.result.arg_resets;
#    1458 |             stats.result.tensor_args_added += 4;
# >  1459 |             stats.result.scalar_args_added += 3;
0x0000000000055598 (+0x00055348)  1cc1e928  LD_XD_XN_IMM.B64                X0, X30, #2344
# [DWARF] common/pa_scheduler_core.h:1457
#    1451 |                 break;
#    1452 |             }
#    1453 |             AcceptTaskOutputs(orchestration, TaskKind::Qk, context.result);
#    1454 | 
#    1455 |             BuildSfArgs(orchestration, args);
#    1456 |             ++stats.result.dynamic_create_infos;
# >  1457 |             ++stats.result.arg_resets;
0x000000000005559c (+0x0005534c)  02ca2440  SBITSET.B64                     X5, X2
# [DWARF] common/pa_scheduler_core.h:1459
#    1458 |             stats.result.tensor_args_added += 4;
# >  1459 |             stats.result.scalar_args_added += 3;
0x00000000000555a0 (+0x00055350)  08060005  ADD_IMM.S64                     X3, X0, #5
# [DWARF] common/pa_scheduler_core.h:1458
#    1452 |             }
#    1453 |             AcceptTaskOutputs(orchestration, TaskKind::Qk, context.result);
#    1454 | 
#    1455 |             BuildSfArgs(orchestration, args);
#    1456 |             ++stats.result.dynamic_create_infos;
#    1457 |             ++stats.result.arg_resets;
# >  1458 |             stats.result.tensor_args_added += 4;
0x00000000000555a4 (+0x00055354)  1cc1e930  LD_XD_XN_IMM.B64                X0, X30, #2352
0x00000000000555a8 (+0x00055358)  0800000b  ADD_IMM.S64                     X0, X0, #11
0x00000000000555ac (+0x0005535c)  03c1ea08  ST_XD_XN_IMM.B64                X0, X30, #2568
# [DWARF] common/pa_trace.h:430
#     424 |     (void)result;
#     425 |     (void)task_id;
#     426 |     (void)site;
#     427 |     (void)result_used;
#     428 |     return Ops::Exchange(address, value);
#     429 | #else
# >   430 |     if (!trace.atomics_enabled) return Ops::Exchange(address, value);
0x00000000000555b0 (+0x00055360)  40200002  JUMPC                           #2
0x00000000000555b4 (+0x00055364)  40000079  JUMP                            #121
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000555b8 (+0x00055368)  1cc1ea30  LD_XD_XN_IMM.B64                X0, X30, #2608
0x00000000000555bc (+0x0005536c)  03cbe948  ST_XD_XN_IMM.B64                X5, X30, #2376
0x00000000000555c0 (+0x00055370)  02043800  MOV_XD_XN.S64                   X2, X3
0x00000000000555c4 (+0x00055374)  03f3eb70  ST_XD_XN_IMM.B64                X25, X30, #2928
0x00000000000555c8 (+0x00055378)  02198800  MOV_XD_XN.S64                   X12, X24
0x00000000000555cc (+0x0005537c)  03c3e940  ST_XD_XN_IMM.B64                X1, X30, #2368
0x00000000000555d0 (+0x00055380)  03c1ea40  ST_XD_XN_IMM.B64                X0, X30, #2624
0x00000000000555d4 (+0x00055384)  40000002  JUMP                            #2
0x00000000000555d8 (+0x00055388)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x00000000000555dc (+0x0005538c)  1cf5eb58  LD_XD_XN_IMM.B64                X26, X30, #2904
0x00000000000555e0 (+0x00055390)  40000002  JUMP                            #2
0x00000000000555e4 (+0x00055394)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x00000000000555e8 (+0x00055398)  1cc3eac8  LD_XD_XN_IMM.B64                X1, X30, #2760
0x00000000000555ec (+0x0005539c)  1cc7eab0  LD_XD_XN_IMM.B64                X3, X30, #2736
0x00000000000555f0 (+0x000553a0)  4000004e  JUMP                            #78
0x00000000000555f4 (+0x000553a4)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
# [DWARF] common/pa_trace.h:549
#     543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
# >   549 |         core.dropped = core.dropped + 1;
0x00000000000555f8 (+0x000553a8)  1c826004  LD_XD_XN_IMM.B32                X1, X6, #4
0x00000000000555fc (+0x000553ac)  08021001  ADD_IMM.S64                     X1, X1, #1
0x0000000000055600 (+0x000553b0)  03826004  ST_XD_XN_IMM.B32                X1, X6, #4
0x0000000000055604 (+0x000553b4)  40000002  JUMP                            #2
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055608 (+0x000553b8)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x000000000005560c (+0x000553bc)  07140000  MOV_XD_IMM                      X10, #0
0x0000000000055610 (+0x000553c0)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000055614 (+0x000553c4)  0700ac2f  MOV_XD_IMM                      X0, #44079
0x0000000000055618 (+0x000553c8)  0741fffe  MOVK                            X0, #65534, #1
0x000000000005561c (+0x000553cc)  0781ffff  MOVK                            X0, #65535, #2
0x0000000000055620 (+0x000553d0)  40020000  JUMP                            X0, #0
0x0000000000055624 (+0x000553d4)  03c5ea10  ST_XD_XN_IMM.B64                X2, X30, #2576
0x0000000000055628 (+0x000553d8)  07040001  MOV_XD_IMM                      X2, #1
0x000000000005562c (+0x000553dc)  03c3ea18  ST_XD_XN_IMM.B64                X1, X30, #2584
0x0000000000055630 (+0x000553e0)  07060001  MOV_XD_IMM                      X3, #1
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x0000000000055634 (+0x000553e4)  02808880  MOV_XD_SPR.F32                  X0, SYS_CNT
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:45
#      39 |   CCE_INTRINSIC[aicore] FTYPE atomicSub(__gm__ FTYPE *base, FTYPE inc) {       \
#      40 |     return __builtin_cce_atom_add_G_##ADD_SUFFIX(base, -inc,                   \
#      41 |                                                   (uint32_t)L2Cache);          \
#      42 |   }
#      43 | 
#      44 | __CCE__ATOM_G_CAS_EXCH_INT32(uint32_t, u32, u32)
# >    45 | __CCE__ATOM_G_CAS_EXCH_INT32(int32_t, s32, s32)
0x0000000000055638 (+0x000553e8)  1cc3eac8  LD_XD_XN_IMM.B64                X1, X30, #2760
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:0
# [SOURCE unavailable]
0x000000000005563c (+0x000553ec)  02063080  NEG.S64                         X3, X3
# [DWARF] common/pa_trace.h:236
#     230 |            (encoded_retries << kAtomicRetriesShift);
#     231 | }
#     232 | 
#     233 | PA_DEVICE void CountAtomicCall(
#     234 |     TraceContext &trace, WorkerResult &result, bool poll_batch
#     235 | ) {
# >   236 |     if (result.atomic_trace_calls == UINT64_MAX) {
0x0000000000055640 (+0x000553f0)  0001318e  CMP.S64.EQ                      X19, X3
0x0000000000055644 (+0x000553f4)  08093001  ADD_IMM.S64                     X4, X19, #1
0x0000000000055648 (+0x000553f8)  00e63209  SEL.B64                         X19, X3, X4
0x000000000005564c (+0x000553fc)  00da2689  SEL.B64                         X13, X2, X13
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x0000000000055650 (+0x00055400)  02175900  MOV_SPR_XN.S64                  CONDITION_FLAG, X21
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:45
#      39 |   CCE_INTRINSIC[aicore] FTYPE atomicSub(__gm__ FTYPE *base, FTYPE inc) {       \
#      40 |     return __builtin_cce_atom_add_G_##ADD_SUFFIX(base, -inc,                   \
#      41 |                                                   (uint32_t)L2Cache);          \
#      42 |   }
#      43 | 
#      44 | __CCE__ATOM_G_CAS_EXCH_INT32(uint32_t, u32, u32)
# >    45 | __CCE__ATOM_G_CAS_EXCH_INT32(int32_t, s32, s32)
0x0000000000055654 (+0x00055404)  50108440  ATOM                            XN, XM, XD, EXCH
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x0000000000055658 (+0x00055408)  02828880  MOV_XD_SPR.F32                  X1, SYS_CNT
# [DWARF] ccec/ccec_ops.h:0
# [SOURCE unavailable]
0x000000000005565c (+0x0005540c)  1ce9eb80  LD_XD_XN_IMM.B64                X20, X30, #2944
0x0000000000055660 (+0x00055410)  1cddeb98  LD_XD_XN_IMM.B64                X14, X30, #2968
0x0000000000055664 (+0x00055414)  1cf5eb58  LD_XD_XN_IMM.B64                X26, X30, #2904
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x0000000000055668 (+0x00055418)  40200006  JUMPC                           #6
0x000000000005566c (+0x0005541c)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000055670 (+0x00055420)  0700017a  MOV_XD_IMM                      X0, #378
0x0000000000055674 (+0x00055424)  07410000  MOVK                            X0, #0, #1
0x0000000000055678 (+0x00055428)  07810000  MOVK                            X0, #0, #2
0x000000000005567c (+0x0005542c)  40020000  JUMP                            X0, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055680 (+0x00055430)  1cd3ea30  LD_XD_XN_IMM.B64                X9, X30, #2608
0x0000000000055684 (+0x00055434)  03cbea00  ST_XD_XN_IMM.B64                X5, X30, #2560
0x0000000000055688 (+0x00055438)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x000000000005568c (+0x0005543c)  0700016b  MOV_XD_IMM                      X0, #363
0x0000000000055690 (+0x00055440)  07410000  MOVK                            X0, #0, #1
0x0000000000055694 (+0x00055444)  07810000  MOVK                            X0, #0, #2
0x0000000000055698 (+0x00055448)  40020000  JUMP                            X0, #0
0x000000000005569c (+0x0005544c)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x00000000000556a0 (+0x00055450)  1ce5eb48  LD_XD_XN_IMM.B64                X18, X30, #2888
0x00000000000556a4 (+0x00055454)  1cd7eb68  LD_XD_XN_IMM.B64                X11, X30, #2920
0x00000000000556a8 (+0x00055458)  40000002  JUMP                            #2
0x00000000000556ac (+0x0005545c)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
# [DWARF] common/pa_scheduler_core.h:753
#     747 |             // Register 后到 Submit.end 由离线 submit_tail_gap 补集展示；排他
#     748 |             // 报告汇总为 submit_tail_residual，避免零时长 marker 增加 raw
#     749 |             // 体积和 trace-buffer 写开销。
#     750 |         }
#     751 |     }
#     752 | 
# >   753 |     ++stats.result.submits;
0x00000000000556b0 (+0x00055460)  1cc1e940  LD_XD_XN_IMM.B64                X0, X30, #2368
# [DWARF] common/pa_trace.h:430
#     424 |     (void)result;
#     425 |     (void)task_id;
#     426 |     (void)site;
#     427 |     (void)result_used;
#     428 |     return Ops::Exchange(address, value);
#     429 | #else
# >   430 |     if (!trace.atomics_enabled) return Ops::Exchange(address, value);
0x00000000000556b4 (+0x00055464)  02172900  MOV_SPR_XN.S64                  CONDITION_FLAG, X18
# [DWARF] common/pa_scheduler_core.h:1468
#    1462 |                 )) {
#    1463 |                 break;
#    1464 |             }
#    1465 |             AcceptTaskOutputs(orchestration, TaskKind::Sf, context.result);
#    1466 | 
#    1467 |             BuildPvArgs(orchestration, args, batch);
# >  1468 |             ++stats.result.arg_resets;
0x00000000000556b8 (+0x00055468)  1cc7e948  LD_XD_XN_IMM.B64                X3, X30, #2376
# [DWARF] common/pa_scheduler_core.h:753
#     747 |             // Register 后到 Submit.end 由离线 submit_tail_gap 补集展示；排他
#     748 |             // 报告汇总为 submit_tail_residual，避免零时长 marker 增加 raw
#     749 |             // 体积和 trace-buffer 写开销。
#     750 |         }
#     751 |     }
#     752 | 
# >   753 |     ++stats.result.submits;
0x00000000000556bc (+0x0005546c)  08020003  ADD_IMM.S64                     X1, X0, #3
# [DWARF] common/pa_scheduler_core.h:1470
#    1464 |             }
#    1465 |             AcceptTaskOutputs(orchestration, TaskKind::Sf, context.result);
#    1466 | 
#    1467 |             BuildPvArgs(orchestration, args, batch);
#    1468 |             ++stats.result.arg_resets;
#    1469 |             stats.result.tensor_args_added += 4;
# >  1470 |             stats.result.scalar_args_added += 2;
0x00000000000556c0 (+0x00055470)  1cc1e928  LD_XD_XN_IMM.B64                X0, X30, #2344
0x00000000000556c4 (+0x00055474)  08040007  ADD_IMM.S64                     X2, X0, #7
# [DWARF] common/pa_scheduler_core.h:1469
#    1463 |                 break;
#    1464 |             }
#    1465 |             AcceptTaskOutputs(orchestration, TaskKind::Sf, context.result);
#    1466 | 
#    1467 |             BuildPvArgs(orchestration, args, batch);
#    1468 |             ++stats.result.arg_resets;
# >  1469 |             stats.result.tensor_args_added += 4;
0x00000000000556c8 (+0x00055478)  1cc1e930  LD_XD_XN_IMM.B64                X0, X30, #2352
0x00000000000556cc (+0x0005547c)  0800000f  ADD_IMM.S64                     X0, X0, #15
0x00000000000556d0 (+0x00055480)  03c1ea08  ST_XD_XN_IMM.B64                X0, X30, #2568
0x00000000000556d4 (+0x00055484)  07000003  MOV_XD_IMM                      X0, #3
# [DWARF] common/pa_scheduler_core.h:1468
#    1462 |                 )) {
#    1463 |                 break;
#    1464 |             }
#    1465 |             AcceptTaskOutputs(orchestration, TaskKind::Sf, context.result);
#    1466 | 
#    1467 |             BuildPvArgs(orchestration, args, batch);
# >  1468 |             ++stats.result.arg_resets;
0x00000000000556d8 (+0x00055488)  00c0300b  OR.B64                          X0, X3, X0
# [DWARF] common/pa_trace.h:430
#     424 |     (void)result;
#     425 |     (void)task_id;
#     426 |     (void)site;
#     427 |     (void)result_used;
#     428 |     return Ops::Exchange(address, value);
#     429 | #else
# >   430 |     if (!trace.atomics_enabled) return Ops::Exchange(address, value);
0x00000000000556dc (+0x0005548c)  40200006  JUMPC                           #6
0x00000000000556e0 (+0x00055490)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x00000000000556e4 (+0x00055494)  0700013c  MOV_XD_IMM                      X0, #316
0x00000000000556e8 (+0x00055498)  07410000  MOVK                            X0, #0, #1
0x00000000000556ec (+0x0005549c)  07810000  MOVK                            X0, #0, #2
0x00000000000556f0 (+0x000554a0)  40020000  JUMP                            X0, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000556f4 (+0x000554a4)  03c3e940  ST_XD_XN_IMM.B64                X1, X30, #2368
0x00000000000556f8 (+0x000554a8)  1cc3ea30  LD_XD_XN_IMM.B64                X1, X30, #2608
0x00000000000556fc (+0x000554ac)  03f3eb70  ST_XD_XN_IMM.B64                X25, X30, #2928
0x0000000000055700 (+0x000554b0)  03c1e948  ST_XD_XN_IMM.B64                X0, X30, #2376
0x0000000000055704 (+0x000554b4)  03c3ea40  ST_XD_XN_IMM.B64                X1, X30, #2624
0x0000000000055708 (+0x000554b8)  02198800  MOV_XD_XN.S64                   X12, X24
0x000000000005570c (+0x000554bc)  1cc7e950  LD_XD_XN_IMM.B64                X3, X30, #2384
0x0000000000055710 (+0x000554c0)  40000002  JUMP                            #2
0x0000000000055714 (+0x000554c4)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000055718 (+0x000554c8)  1cf5eb58  LD_XD_XN_IMM.B64                X26, X30, #2904
0x000000000005571c (+0x000554cc)  40000002  JUMP                            #2
0x0000000000055720 (+0x000554d0)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000055724 (+0x000554d4)  1cc3eac8  LD_XD_XN_IMM.B64                X1, X30, #2760
0x0000000000055728 (+0x000554d8)  07000001  MOV_XD_IMM                      X0, #1
0x000000000005572c (+0x000554dc)  50100400  ATOM                            XN, XM, XD, EXCH
0x0000000000055730 (+0x000554e0)  02023800  MOV_XD_XN.S64                   X1, X3
0x0000000000055734 (+0x000554e4)  40000002  JUMP                            #2
0x0000000000055738 (+0x000554e8)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x000000000005573c (+0x000554ec)  02808880  MOV_XD_SPR.F32                  X0, SYS_CNT
0x0000000000055740 (+0x000554f0)  03c5ea10  ST_XD_XN_IMM.B64                X2, X30, #2576
0x0000000000055744 (+0x000554f4)  03c3eab0  ST_XD_XN_IMM.B64                X1, X30, #2736
0x0000000000055748 (+0x000554f8)  1ce3eb10  LD_XD_XN_IMM.B64                X17, X30, #2832
0x000000000005574c (+0x000554fc)  07020000  MOV_XD_IMM                      X1, #0
0x0000000000055750 (+0x00055500)  1ce9eb80  LD_XD_XN_IMM.B64                X20, X30, #2944
0x0000000000055754 (+0x00055504)  0238c800  MOV_XD_XN.S64                   X28, X12
0x0000000000055758 (+0x00055508)  1cddeb98  LD_XD_XN_IMM.B64                X14, X30, #2968
0x000000000005575c (+0x0005550c)  1cd7eb68  LD_XD_XN_IMM.B64                X11, X30, #2920
0x0000000000055760 (+0x00055510)  1ce5eb48  LD_XD_XN_IMM.B64                X18, X30, #2888
0x0000000000055764 (+0x00055514)  1cebeb40  LD_XD_XN_IMM.B64                X21, X30, #2880
0x0000000000055768 (+0x00055518)  1cf3eb70  LD_XD_XN_IMM.B64                X25, X30, #2928
0x000000000005576c (+0x0005551c)  1ce7eb00  LD_XD_XN_IMM.B64                X19, X30, #2816
0x0000000000055770 (+0x00055520)  03c1ea70  ST_XD_XN_IMM.B64                X0, X30, #2672
0x0000000000055774 (+0x00055524)  1cc1e940  LD_XD_XN_IMM.B64                X0, X30, #2368
0x0000000000055778 (+0x00055528)  03c1ea18  ST_XD_XN_IMM.B64                X0, X30, #2584
0x000000000005577c (+0x0005552c)  1cc1e948  LD_XD_XN_IMM.B64                X0, X30, #2376
0x0000000000055780 (+0x00055530)  03c1ea00  ST_XD_XN_IMM.B64                X0, X30, #2560
0x0000000000055784 (+0x00055534)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000055788 (+0x00055538)  0700025a  MOV_XD_IMM                      X0, #602
0x000000000005578c (+0x0005553c)  07410000  MOVK                            X0, #0, #1
0x0000000000055790 (+0x00055540)  07810000  MOVK                            X0, #0, #2
0x0000000000055794 (+0x00055544)  40020000  JUMP                            X0, #0
# [DWARF] ccec/ccec_ops.h:0
# [SOURCE unavailable]
0x0000000000055798 (+0x00055548)  03c7ea10  ST_XD_XN_IMM.B64                X3, X30, #2576
0x000000000005579c (+0x0005554c)  07060001  MOV_XD_IMM                      X3, #1
0x00000000000557a0 (+0x00055550)  03c3ea18  ST_XD_XN_IMM.B64                X1, X30, #2584
0x00000000000557a4 (+0x00055554)  02063080  NEG.S64                         X3, X3
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x00000000000557a8 (+0x00055558)  02808880  MOV_XD_SPR.F32                  X0, SYS_CNT
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:45
#      39 |   CCE_INTRINSIC[aicore] FTYPE atomicSub(__gm__ FTYPE *base, FTYPE inc) {       \
#      40 |     return __builtin_cce_atom_add_G_##ADD_SUFFIX(base, -inc,                   \
#      41 |                                                   (uint32_t)L2Cache);          \
#      42 |   }
#      43 | 
#      44 | __CCE__ATOM_G_CAS_EXCH_INT32(uint32_t, u32, u32)
# >    45 | __CCE__ATOM_G_CAS_EXCH_INT32(int32_t, s32, s32)
0x00000000000557ac (+0x0005555c)  1cc3eac8  LD_XD_XN_IMM.B64                X1, X30, #2760
# [DWARF] common/pa_trace.h:236
#     230 |            (encoded_retries << kAtomicRetriesShift);
#     231 | }
#     232 | 
#     233 | PA_DEVICE void CountAtomicCall(
#     234 |     TraceContext &trace, WorkerResult &result, bool poll_batch
#     235 | ) {
# >   236 |     if (result.atomic_trace_calls == UINT64_MAX) {
0x00000000000557b0 (+0x00055560)  0001318e  CMP.S64.EQ                      X19, X3
0x00000000000557b4 (+0x00055564)  08093001  ADD_IMM.S64                     X4, X19, #1
0x00000000000557b8 (+0x00055568)  00e63209  SEL.B64                         X19, X3, X4
0x00000000000557bc (+0x0005556c)  00da2689  SEL.B64                         X13, X2, X13
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x00000000000557c0 (+0x00055570)  02175900  MOV_SPR_XN.S64                  CONDITION_FLAG, X21
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:45
#      39 |   CCE_INTRINSIC[aicore] FTYPE atomicSub(__gm__ FTYPE *base, FTYPE inc) {       \
#      40 |     return __builtin_cce_atom_add_G_##ADD_SUFFIX(base, -inc,                   \
#      41 |                                                   (uint32_t)L2Cache);          \
#      42 |   }
#      43 | 
#      44 | __CCE__ATOM_G_CAS_EXCH_INT32(uint32_t, u32, u32)
# >    45 | __CCE__ATOM_G_CAS_EXCH_INT32(int32_t, s32, s32)
0x00000000000557c4 (+0x00055574)  50108440  ATOM                            XN, XM, XD, EXCH
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x00000000000557c8 (+0x00055578)  02828880  MOV_XD_SPR.F32                  X1, SYS_CNT
# [DWARF] ccec/ccec_ops.h:0
# [SOURCE unavailable]
0x00000000000557cc (+0x0005557c)  1ce9eb80  LD_XD_XN_IMM.B64                X20, X30, #2944
0x00000000000557d0 (+0x00055580)  1cddeb98  LD_XD_XN_IMM.B64                X14, X30, #2968
0x00000000000557d4 (+0x00055584)  1cf5eb58  LD_XD_XN_IMM.B64                X26, X30, #2904
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x00000000000557d8 (+0x00055588)  40200006  JUMPC                           #6
0x00000000000557dc (+0x0005558c)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x00000000000557e0 (+0x00055590)  07000157  MOV_XD_IMM                      X0, #343
0x00000000000557e4 (+0x00055594)  07410000  MOVK                            X0, #0, #1
0x00000000000557e8 (+0x00055598)  07810000  MOVK                            X0, #0, #2
0x00000000000557ec (+0x0005559c)  40020000  JUMP                            X0, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000557f0 (+0x000555a0)  1cc1ea30  LD_XD_XN_IMM.B64                X0, X30, #2608
0x00000000000557f4 (+0x000555a4)  03cbea00  ST_XD_XN_IMM.B64                X5, X30, #2560
0x00000000000557f8 (+0x000555a8)  02398800  MOV_XD_XN.S64                   X28, X24
0x00000000000557fc (+0x000555ac)  02120800  MOV_XD_XN.S64                   X9, X0
0x0000000000055800 (+0x000555b0)  03c1ea40  ST_XD_XN_IMM.B64                X0, X30, #2624
0x0000000000055804 (+0x000555b4)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000055808 (+0x000555b8)  07000111  MOV_XD_IMM                      X0, #273
0x000000000005580c (+0x000555bc)  07410000  MOVK                            X0, #0, #1
0x0000000000055810 (+0x000555c0)  07810000  MOVK                            X0, #0, #2
0x0000000000055814 (+0x000555c4)  40020000  JUMP                            X0, #0
# [DWARF] common/pa_trace.h:547
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
#     543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
# >   547 |     const uint32_t slot = core.count;
0x0000000000055818 (+0x000555c8)  1c854000  LD_XD_XN_IMM.B32                X2, X20, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x000000000005581c (+0x000555cc)  070c0000  MOV_XD_IMM                      X6, #0
0x0000000000055820 (+0x000555d0)  1cf3eb70  LD_XD_XN_IMM.B64                X25, X30, #2928
0x0000000000055824 (+0x000555d4)  1ccbe948  LD_XD_XN_IMM.B64                X5, X30, #2376
0x0000000000055828 (+0x000555d8)  1ccfe940  LD_XD_XN_IMM.B64                X7, X30, #2368
# [DWARF] common/pa_trace.h:548
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
#     543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
#     547 |     const uint32_t slot = core.count;
# >   548 |     if (slot >= trace.capacity) {
0x000000000005582c (+0x000555dc)  0040272e  CMP.U64.LT                      X2, X14
0x0000000000055830 (+0x000555e0)  40200006  JUMPC                           #6
0x0000000000055834 (+0x000555e4)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000055838 (+0x000555e8)  07000160  MOV_XD_IMM                      X0, #352
0x000000000005583c (+0x000555ec)  07410000  MOVK                            X0, #0, #1
0x0000000000055840 (+0x000555f0)  07810000  MOVK                            X0, #0, #2
0x0000000000055844 (+0x000555f4)  40020000  JUMP                            X0, #0
# [DWARF] common/pa_trace.h:552
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
# >   552 |     PA_GM TraceRecord &record = trace.records[slot];
0x0000000000055848 (+0x000555f8)  1cc9eb90  LD_XD_XN_IMM.B64                X4, X30, #2960
0x000000000005584c (+0x000555fc)  02062800  MOV_XD_XN.S64                   X3, X2
0x0000000000055850 (+0x00055600)  03cfea18  ST_XD_XN_IMM.B64                X7, X30, #2584
# [DWARF] common/pa_trace.h:555
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
# >   555 |     record.task_id = task_id;
0x0000000000055854 (+0x00055604)  1cd1eaa8  LD_XD_XN_IMM.B64                X8, X30, #2728
# [DWARF] common/pa_trace.h:552
#     546 |     PA_GM TraceCoreState &core = *trace.core;
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
# >   552 |     PA_GM TraceRecord &record = trace.records[slot];
0x0000000000055858 (+0x00055608)  02c60206  SHL.B64                         X3, #6
0x000000000005585c (+0x0005560c)  03cbea00  ST_XD_XN_IMM.B64                X5, X30, #2560
0x0000000000055860 (+0x00055610)  1cd3ea30  LD_XD_XN_IMM.B64                X9, X30, #2608
0x0000000000055864 (+0x00055614)  02398800  MOV_XD_XN.S64                   X28, X24
0x0000000000055868 (+0x00055618)  00064181  ADD.S64                         X3, X4, X3
# [DWARF] common/pa_trace.h:553
# >   553 |     record.start_cycle = start_cycle;
0x000000000005586c (+0x0005561c)  09c03081  STP_XI_XJ_XN.B64                X0, X1, X3, #0
0x0000000000055870 (+0x00055620)  0700ffff  MOV_XD_IMM                      X0, #65535
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055874 (+0x00055624)  0741ffff  MOVK                            X0, #65535, #1
0x0000000000055878 (+0x00055628)  0702000e  MOV_XD_IMM                      X1, #14
# [DWARF] common/pa_trace.h:555
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
# >   555 |     record.task_id = task_id;
0x000000000005587c (+0x0005562c)  09903021  STP_XI_XJ_XN.B32                X8, X0, X3, #16
# [DWARF] common/pa_trace.h:557
#     556 |     record.function_id = function_id;
# >   557 |     record.phase = static_cast<int32_t>(trace_phase);
0x0000000000055880 (+0x00055630)  03c23018  ST_XD_XN_IMM.B64                X1, X3, #24
0x0000000000055884 (+0x00055634)  1cc3eb30  LD_XD_XN_IMM.B64                X1, X30, #2864
# [DWARF] common/pa_trace.h:565
#     558 |     record.lane = trace.lane;
#     559 |     record.block_id = trace.block_id;
#     560 |     record.core_idx = trace.core_idx;
#     561 |     record.flags = flags;
#     562 |     record.auxiliary = auxiliary;
#     563 |     // count 最后更新，使其始终指向下一空槽；单写者条件下无需 reserve/commit 两阶段。
#     564 |     // 最终 FlushTraceCore 会把记录体先于该计数一并导出。
# >   565 |     core.count = slot + 1;
0x0000000000055888 (+0x00055638)  08002001  ADD_IMM.S64                     X0, X2, #1
# [DWARF] common/pa_trace.h:559
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
#     555 |     record.task_id = task_id;
#     556 |     record.function_id = function_id;
#     557 |     record.phase = static_cast<int32_t>(trace_phase);
#     558 |     record.lane = trace.lane;
# >   559 |     record.block_id = trace.block_id;
0x000000000005588c (+0x0005563c)  08043020  ADD_IMM.S64                     X2, X3, #32
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055890 (+0x00055640)  07080001  MOV_XD_IMM                      X4, #1
0x0000000000055894 (+0x00055644)  07890003  MOVK                            X4, #3, #2
# [DWARF] common/pa_trace.h:559
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
#     555 |     record.task_id = task_id;
#     556 |     record.function_id = function_id;
#     557 |     record.phase = static_cast<int32_t>(trace_phase);
#     558 |     record.lane = trace.lane;
# >   559 |     record.block_id = trace.block_id;
0x0000000000055898 (+0x00055648)  09822081  STP_XI_XJ_XN.B32                X1, X1, X2, #0
0x000000000005589c (+0x0005564c)  1cc3e928  LD_XD_XN_IMM.B64                X1, X30, #2344
# [DWARF] common/pa_trace.h:561
#     560 |     record.core_idx = trace.core_idx;
# >   561 |     record.flags = flags;
0x00000000000558a0 (+0x00055650)  03c83028  ST_XD_XN_IMM.B64                X4, X3, #40
0x00000000000558a4 (+0x00055654)  03c3ea10  ST_XD_XN_IMM.B64                X1, X30, #2576
# [DWARF] common/pa_trace.h:565
#     562 |     record.auxiliary = auxiliary;
#     563 |     // count 最后更新，使其始终指向下一空槽；单写者条件下无需 reserve/commit 两阶段。
#     564 |     // 最终 FlushTraceCore 会把记录体先于该计数一并导出。
# >   565 |     core.count = slot + 1;
0x00000000000558a8 (+0x00055658)  03814000  ST_XD_XN_IMM.B32                X0, X20, #0
0x00000000000558ac (+0x0005565c)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x00000000000558b0 (+0x00055660)  070001ab  MOV_XD_IMM                      X0, #427
0x00000000000558b4 (+0x00055664)  07410000  MOVK                            X0, #0, #1
0x00000000000558b8 (+0x00055668)  07810000  MOVK                            X0, #0, #2
0x00000000000558bc (+0x0005566c)  40020000  JUMP                            X0, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000558c0 (+0x00055670)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x00000000000558c4 (+0x00055674)  1ce5eb48  LD_XD_XN_IMM.B64                X18, X30, #2888
0x00000000000558c8 (+0x00055678)  02177800  MOV_XD_XN.S64                   X11, X23
0x00000000000558cc (+0x0005567c)  1cc5e928  LD_XD_XN_IMM.B64                X2, X30, #2344
0x00000000000558d0 (+0x00055680)  40000002  JUMP                            #2
0x00000000000558d4 (+0x00055684)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
# [DWARF] common/pa_scheduler_core.h:753
#     747 |             // Register 后到 Submit.end 由离线 submit_tail_gap 补集展示；排他
#     748 |             // 报告汇总为 submit_tail_residual，避免零时长 marker 增加 raw
#     749 |             // 体积和 trace-buffer 写开销。
#     750 |         }
#     751 |     }
#     752 | 
# >   753 |     ++stats.result.submits;
0x00000000000558d8 (+0x00055688)  1cc1e940  LD_XD_XN_IMM.B64                X0, X30, #2368
# [DWARF] common/pa_trace.h:430
#     424 |     (void)result;
#     425 |     (void)task_id;
#     426 |     (void)site;
#     427 |     (void)result_used;
#     428 |     return Ops::Exchange(address, value);
#     429 | #else
# >   430 |     if (!trace.atomics_enabled) return Ops::Exchange(address, value);
0x00000000000558dc (+0x0005568c)  02172900  MOV_SPR_XN.S64                  CONDITION_FLAG, X18
# [DWARF] common/pa_scheduler_core.h:753
#     747 |             // Register 后到 Submit.end 由离线 submit_tail_gap 补集展示；排他
#     748 |             // 报告汇总为 submit_tail_residual，避免零时长 marker 增加 raw
#     749 |             // 体积和 trace-buffer 写开销。
#     750 |         }
#     751 |     }
#     752 | 
# >   753 |     ++stats.result.submits;
0x00000000000558e0 (+0x00055690)  08000004  ADD_IMM.S64                     X0, X0, #4
# [DWARF] common/pa_trace.h:430
#     424 |     (void)result;
#     425 |     (void)task_id;
#     426 |     (void)site;
#     427 |     (void)result_used;
#     428 |     return Ops::Exchange(address, value);
#     429 | #else
# >   430 |     if (!trace.atomics_enabled) return Ops::Exchange(address, value);
0x00000000000558e4 (+0x00055694)  40200006  JUMPC                           #6
0x00000000000558e8 (+0x00055698)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x00000000000558ec (+0x0005569c)  070000f5  MOV_XD_IMM                      X0, #245
0x00000000000558f0 (+0x000556a0)  07410000  MOVK                            X0, #0, #1
0x00000000000558f4 (+0x000556a4)  07810000  MOVK                            X0, #0, #2
0x00000000000558f8 (+0x000556a8)  40020000  JUMP                            X0, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000558fc (+0x000556ac)  03c1e940  ST_XD_XN_IMM.B64                X0, X30, #2368
0x0000000000055900 (+0x000556b0)  1cc1ea30  LD_XD_XN_IMM.B64                X0, X30, #2608
0x0000000000055904 (+0x000556b4)  03f3eb70  ST_XD_XN_IMM.B64                X25, X30, #2928
0x0000000000055908 (+0x000556b8)  1cd9eb38  LD_XD_XN_IMM.B64                X12, X30, #2872
0x000000000005590c (+0x000556bc)  03d7eb68  ST_XD_XN_IMM.B64                X11, X30, #2920
0x0000000000055910 (+0x000556c0)  1cc7e918  LD_XD_XN_IMM.B64                X3, X30, #2328
0x0000000000055914 (+0x000556c4)  1cf5eb58  LD_XD_XN_IMM.B64                X26, X30, #2904
0x0000000000055918 (+0x000556c8)  03c1ea40  ST_XD_XN_IMM.B64                X0, X30, #2624
0x000000000005591c (+0x000556cc)  1cc1e930  LD_XD_XN_IMM.B64                X0, X30, #2352
0x0000000000055920 (+0x000556d0)  03c1ea08  ST_XD_XN_IMM.B64                X0, X30, #2568
0x0000000000055924 (+0x000556d4)  4000ff80  JUMP                            #65408
# [DWARF] common/pa_trace.h:549
#     543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
# >   549 |         core.dropped = core.dropped + 1;
0x0000000000055928 (+0x000556d8)  1c804004  LD_XD_XN_IMM.B32                X0, X4, #4
0x000000000005592c (+0x000556dc)  08000001  ADD_IMM.S64                     X0, X0, #1
0x0000000000055930 (+0x000556e0)  03804004  ST_XD_XN_IMM.B32                X0, X4, #4
0x0000000000055934 (+0x000556e4)  40000002  JUMP                            #2
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055938 (+0x000556e8)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x000000000005593c (+0x000556ec)  02018800  MOV_XD_XN.S64                   X0, X24
0x0000000000055940 (+0x000556f0)  03cdeb60  ST_XD_XN_IMM.B64                X6, X30, #2912
# [DWARF] common/pa_trace.h:296
#     290 | PA_DEVICE void AtomicPollBoundary(TraceContext &trace, WorkerResult &result) {
#     291 | #if PA_BUILD_SUBMIT_PMU
#     292 |     (void)trace;
#     293 |     (void)result;
#     294 | #else
#     295 |     (void)result;
# >   296 |     if (trace.poll_burst.active_mask == 0) return;
0x0000000000055944 (+0x000556f4)  02800a00  ZEROEXT.U32                     X0, X0
0x0000000000055948 (+0x000556f8)  07240000  MOV_XD_IMM                      X18, #0
0x000000000005594c (+0x000556fc)  0000091e  CMP.S64.NE                      X0, X18
0x0000000000055950 (+0x00055700)  40200002  JUMPC                           #2
0x0000000000055954 (+0x00055704)  4000005f  JUMP                            #95
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055958 (+0x00055708)  03e7eb00  ST_XD_XN_IMM.B64                X19, X30, #2816
0x000000000005595c (+0x0005570c)  07040000  MOV_XD_IMM                      X2, #0
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x0000000000055960 (+0x00055710)  02ae8880  MOV_XD_SPR.F32                  X23, SYS_CNT
# [DWARF] ccec/ccec_ops.h:0
# [SOURCE unavailable]
0x0000000000055964 (+0x00055714)  07060001  MOV_XD_IMM                      X3, #1
0x0000000000055968 (+0x00055718)  07360001  MOV_XD_IMM                      X27, #1
0x000000000005596c (+0x0005571c)  07280001  MOV_XD_IMM                      X20, #1
0x0000000000055970 (+0x00055720)  072a0006  MOV_XD_IMM                      X21, #6
0x0000000000055974 (+0x00055724)  07450100  MOVK                            X2, #256, #1
0x0000000000055978 (+0x00055728)  0747ff00  MOVK                            X3, #65280, #1
0x000000000005597c (+0x0005572c)  0237b080  NEG.S64                         X27, X27
0x0000000000055980 (+0x00055730)  07320000  MOV_XD_IMM                      X25, #0
0x0000000000055984 (+0x00055734)  07380000  MOV_XD_IMM                      X28, #0
0x0000000000055988 (+0x00055738)  07268fa0  MOV_XD_IMM                      X19, #36768
0x000000000005598c (+0x0005573c)  07670006  MOVK                            X19, #6, #1
0x0000000000055990 (+0x00055740)  07a70000  MOVK                            X19, #0, #2
0x0000000000055994 (+0x00055744)  07e70000  MOVK                            X19, #0, #3
0x0000000000055998 (+0x00055748)  02020880  MOV_XD_SPR.S64                  X1, PC
0x000000000005599c (+0x0005574c)  00273081  ADD.S64                         X19, X19, X1
0x00000000000559a0 (+0x00055750)  40000020  JUMP                            #32
# [DWARF] common/pa_trace.h:273
#     267 |             trace.atomic_counter_overflow = true;
#     268 |             continue;
#     269 |         }
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
#     271 |         const bool written = WritePollBatchRecordRaw(
#     272 |             trace.core, trace.records, trace.capacity,
# >   273 |             trace.poll_burst.start_cycle[index], end_cycle,
0x00000000000559a4 (+0x00055754)  070006a8  MOV_XD_IMM                      X0, #1704
# [DWARF] common/pa_trace.h:271
#     265 |         const uint32_t call_count = trace.poll_burst.call_count[index];
#     266 |         if (call_count == 0 || call_count > kAtomicPollCountMax) {
#     267 |             trace.atomic_counter_overflow = true;
#     268 |             continue;
#     269 |         }
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
# >   271 |         const bool written = WritePollBatchRecordRaw(
0x00000000000559a8 (+0x00055758)  1cc3eb90  LD_XD_XN_IMM.B64                X1, X30, #2960
# [DWARF] common/pa_trace.h:273
#     272 |             trace.core, trace.records, trace.capacity,
# >   273 |             trace.poll_burst.start_cycle[index], end_cycle,
0x00000000000559ac (+0x0005575c)  0001e001  ADD.S64                         X0, X30, X0
# [DWARF] common/pa_trace.h:271
#     265 |         const uint32_t call_count = trace.poll_burst.call_count[index];
#     266 |         if (call_count == 0 || call_count > kAtomicPollCountMax) {
#     267 |             trace.atomic_counter_overflow = true;
#     268 |             continue;
#     269 |         }
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
# >   271 |         const bool written = WritePollBatchRecordRaw(
0x00000000000559b0 (+0x00055760)  1cc5eb98  LD_XD_XN_IMM.B64                X2, X30, #2968
# [DWARF] common/pa_trace.h:273
#     272 |             trace.core, trace.records, trace.capacity,
# >   273 |             trace.poll_burst.start_cycle[index], end_cycle,
0x00000000000559b4 (+0x00055764)  01c60e00  LD_XD_XN.B64                    X3, X0, X28
# [DWARF] common/pa_trace.h:271
#     265 |         const uint32_t call_count = trace.poll_burst.call_count[index];
#     266 |         if (call_count == 0 || call_count > kAtomicPollCountMax) {
#     267 |             trace.atomic_counter_overflow = true;
#     268 |             continue;
#     269 |         }
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
# >   271 |         const bool written = WritePollBatchRecordRaw(
0x00000000000559b8 (+0x00055768)  02097800  MOV_XD_XN.S64                   X4, X23
0x00000000000559bc (+0x0005576c)  1cc1eb80  LD_XD_XN_IMM.B64                X0, X30, #2944
0x00000000000559c0 (+0x00055770)  070e241a  MOV_XD_IMM                      X7, #9242
0x00000000000559c4 (+0x00055774)  074f0000  MOVK                            X7, #0, #1
0x00000000000559c8 (+0x00055778)  078f0000  MOVK                            X7, #0, #2
0x00000000000559cc (+0x0005577c)  40427000  CALL                            X7, #0
# [DWARF] common/pa_trace.h:276
#     272 |             trace.core, trace.records, trace.capacity,
#     273 |             trace.poll_burst.start_cycle[index], end_cycle,
#     274 |             call_count, static_cast<uint32_t>(site)
#     275 |         );
# >   276 |         if (written) {
0x00000000000559d0 (+0x00055780)  00016d8e  CMP.S64.EQ                      X22, X27
0x00000000000559d4 (+0x00055784)  08036001  ADD_IMM.S64                     X1, X22, #1
0x00000000000559d8 (+0x00055788)  021ba800  MOV_XD_XN.S64                   X13, X26
0x00000000000559dc (+0x0005578c)  00c3b089  SEL.B64                         X1, X27, X1
0x00000000000559e0 (+0x00055790)  00c54689  SEL.B64                         X2, X20, X13
# [DWARF] common/pa_trace.h:271
#     265 |         const uint32_t call_count = trace.poll_burst.call_count[index];
#     266 |         if (call_count == 0 || call_count > kAtomicPollCountMax) {
#     267 |             trace.atomic_counter_overflow = true;
#     268 |             continue;
#     269 |         }
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
# >   271 |         const bool written = WritePollBatchRecordRaw(
0x00000000000559e4 (+0x00055794)  02160900  MOV_SPR_XN.S64                  CONDITION_FLAG, X0
# [DWARF] common/pa_trace.h:276
#     272 |             trace.core, trace.records, trace.capacity,
#     273 |             trace.poll_burst.start_cycle[index], end_cycle,
#     274 |             call_count, static_cast<uint32_t>(site)
#     275 |         );
# >   276 |         if (written) {
0x00000000000559e8 (+0x00055798)  00da2689  SEL.B64                         X13, X2, X13
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000559ec (+0x0005579c)  07200690  MOV_XD_IMM                      X16, #1680
0x00000000000559f0 (+0x000557a0)  07040000  MOV_XD_IMM                      X2, #0
0x00000000000559f4 (+0x000557a4)  07060001  MOV_XD_IMM                      X3, #1
# [DWARF] common/pa_trace.h:276
#     270 |         const AtomicSite site = TraceAtomicPollBatchSite(index);
#     271 |         const bool written = WritePollBatchRecordRaw(
#     272 |             trace.core, trace.records, trace.capacity,
#     273 |             trace.poll_burst.start_cycle[index], end_cycle,
#     274 |             call_count, static_cast<uint32_t>(site)
#     275 |         );
# >   276 |         if (written) {
0x00000000000559f8 (+0x000557a8)  00ec1b09  SEL.B64                         X22, X1, X22
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x00000000000559fc (+0x000557ac)  0021e801  ADD.S64                         X16, X30, X16
0x0000000000055a00 (+0x000557b0)  07450100  MOVK                            X2, #256, #1
# [DWARF] common/pa_trace.h:283
#     277 |             if (trace.poll_batch_records == UINT64_MAX) {
#     278 |                 trace.atomic_counter_overflow = true;
#     279 |             } else {
#     280 |                 ++trace.poll_batch_records;
#     281 |             }
#     282 |         }
# >   283 |         trace.poll_burst.call_count[index] = 0;
0x0000000000055a04 (+0x000557b4)  0e810e00  STI_XN_XM.B32                   X16, X28
0x0000000000055a08 (+0x000557b8)  0747ff00  MOVK                            X3, #65280, #1
# [DWARF] common/pa_trace.h:262
#     256 | #else
#     257 |     if (!trace.atomics_enabled || trace.poll_burst.active_mask == 0) return;
#     258 |     const uint32_t active_mask = trace.poll_burst.active_mask;
#     259 |     // CCEC 默认会把固定 6-site 循环完整展开，再随几十个 phase 边界复制。
#     260 |     // 禁止展开只控制代码体积；循环次数、site 顺序和同 cycle 关闭语义不变。
#     261 |     PA_LOOP_NOUNROLL
# >   262 |     for (uint32_t index = 0; index < kAtomicPollBatchSiteCount; ++index) {
0x0000000000055a0c (+0x000557bc)  0839c001  ADD_IMM.S64                     X28, X28, #1
0x0000000000055a10 (+0x000557c0)  08273004  ADD_IMM.S64                     X19, X19, #4
0x0000000000055a14 (+0x000557c4)  0001ca8e  CMP.S64.EQ                      X28, X21
0x0000000000055a18 (+0x000557c8)  08339001  ADD_IMM.S64                     X25, X25, #1
0x0000000000055a1c (+0x000557cc)  40200017  JUMPC                           #23
# [DWARF] common/pa_trace.h:263
# >   263 |         const uint32_t bit = 1U << index;
0x0000000000055a20 (+0x000557d0)  02819a00  ZEROEXT.U32                     X0, X25
# [DWARF] common/pa_trace.h:264
# >   264 |         if ((active_mask & bit) == 0) continue;
0x0000000000055a24 (+0x000557d4)  02838a00  ZEROEXT.U32                     X1, X24
0x0000000000055a28 (+0x000557d8)  024202c0  SHR.U64                         X1, X0, #0
0x0000000000055a2c (+0x000557dc)  00c01a0a  AND.B64                         X0, X1, X20
0x0000000000055a30 (+0x000557e0)  0000090e  CMP.S64.EQ                      X0, X18
0x0000000000055a34 (+0x000557e4)  4020fff6  JUMPC                           #65526
# [DWARF] common/pa_trace.h:265
# >   265 |         const uint32_t call_count = trace.poll_burst.call_count[index];
0x0000000000055a38 (+0x000557e8)  018b0e00  LD_XD_XN.B32                    X5, X16, X28
# [DWARF] common/pa_trace.h:266
# >   266 |         if (call_count == 0 || call_count > kAtomicPollCountMax) {
0x0000000000055a3c (+0x000557ec)  00005102  SUB.S64                         X0, X5, X2
0x0000000000055a40 (+0x000557f0)  02800a00  ZEROEXT.U32                     X0, X0
0x0000000000055a44 (+0x000557f4)  004001ae  CMP.U64.LT                      X0, X3
0x0000000000055a48 (+0x000557f8)  40200002  JUMPC                           #2
0x0000000000055a4c (+0x000557fc)  40000003  JUMP                            #3
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055a50 (+0x00055800)  071a0001  MOV_XD_IMM                      X13, #1
0x0000000000055a54 (+0x00055804)  4000ffee  JUMP                            #65518
# [DWARF] common/pa_trace.h:98
#      92 |         default:
#      93 |             return -1;
#      94 |     }
#      95 | }
#      96 | 
#      97 | PA_DEVICE AtomicSite TraceAtomicPollBatchSite(uint32_t index) {
# >    98 |     switch (index) {
0x0000000000055a58 (+0x00055808)  02819a00  ZEROEXT.U32                     X0, X25
0x0000000000055a5c (+0x0005580c)  07020005  MOV_XD_IMM                      X1, #5
0x0000000000055a60 (+0x00055810)  0234d800  MOV_XD_XN.S64                   X26, X13
0x0000000000055a64 (+0x00055814)  004000be  CMP.U64.GT                      X0, X1
0x0000000000055a68 (+0x00055818)  070c000f  MOV_XD_IMM                      X6, #15
0x0000000000055a6c (+0x0005581c)  4020ffce  JUMPC                           #65486
0x0000000000055a70 (+0x00055820)  1c8d3000  LD_XD_XN_IMM.B32                X6, X19, #0
0x0000000000055a74 (+0x00055824)  4000ffcc  JUMP                            #65484
# [DWARF] common/pa_scheduler_core.h:1354
#    1348 |         if (WatchdogExpired<Ops>(state, stats, start_wait, start_polls)) {
#    1349 |             break;
#    1350 |         }
#    1351 |     }
#    1352 |     AtomicPollRegionEnd<Ops>(stats.trace, stats.result, startup_poll_region);
#    1353 | 
# >  1354 |     const uint32_t batches = state->config.batches;
0x0000000000055a78 (+0x00055828)  1cc1eaf0  LD_XD_XN_IMM.B64                X0, X30, #2800
0x0000000000055a7c (+0x0005582c)  07040000  MOV_XD_IMM                      X2, #0
# [DWARF] common/pa_scheduler_core.h:0
# [SOURCE unavailable]
0x0000000000055a80 (+0x00055830)  1ce3eb10  LD_XD_XN_IMM.B64                X17, X30, #2832
0x0000000000055a84 (+0x00055834)  07020000  MOV_XD_IMM                      X1, #0
0x0000000000055a88 (+0x00055838)  1cdfeb30  LD_XD_XN_IMM.B64                X15, X30, #2864
# [DWARF] common/pa_trace.h:393
#     387 |     (void)result;
#     388 |     (void)task_id;
#     389 |     (void)site;
#     390 |     (void)result_used;
#     391 |     return Ops::Load(address);
#     392 | #else
# >   393 |     if (!trace.atomics_enabled) return Ops::Load(address);
0x0000000000055a8c (+0x0005583c)  03c3eab0  ST_XD_XN_IMM.B64                X1, X30, #2736
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055a90 (+0x00055840)  1ce7eb00  LD_XD_XN_IMM.B64                X19, X30, #2816
0x0000000000055a94 (+0x00055844)  1ccfeac8  LD_XD_XN_IMM.B64                X7, X30, #2760
# [DWARF] common/pa_scheduler_core.h:1354
#    1348 |         if (WatchdogExpired<Ops>(state, stats, start_wait, start_polls)) {
#    1349 |             break;
#    1350 |         }
#    1351 |     }
#    1352 |     AtomicPollRegionEnd<Ops>(stats.trace, stats.result, startup_poll_region);
#    1353 | 
# >  1354 |     const uint32_t batches = state->config.batches;
0x0000000000055a98 (+0x00055848)  1ca40000  LD_XD_XN_IMM.B32                X18, X0, #0
# [DWARF] common/pa_trace.h:393
#     387 |     (void)result;
#     388 |     (void)task_id;
#     389 |     (void)site;
#     390 |     (void)result_used;
#     391 |     return Ops::Load(address);
#     392 | #else
# >   393 |     if (!trace.atomics_enabled) return Ops::Load(address);
0x0000000000055a9c (+0x0005584c)  1cc1eb48  LD_XD_XN_IMM.B64                X0, X30, #2888
0x0000000000055aa0 (+0x00055850)  0000010e  CMP.S64.EQ                      X0, X2
# [DWARF] common/pa_scheduler_core.h:1355
#    1349 |             break;
#    1350 |         }
#    1351 |     }
#    1352 |     AtomicPollRegionEnd<Ops>(stats.trace, stats.result, startup_poll_region);
#    1353 | 
#    1354 |     const uint32_t batches = state->config.batches;
# >  1355 |     const uint32_t task_count = batches * kTasksPerBatch;
0x0000000000055aa4 (+0x00055854)  08412005  MUL_IMM.S64                     X0, X18, #5
# [DWARF] common/pa_scheduler_core.h:0
# [SOURCE unavailable]
0x0000000000055aa8 (+0x00055858)  03c1ea70  ST_XD_XN_IMM.B64                X0, X30, #2672
# [DWARF] common/pa_trace.h:393
#     387 |     (void)result;
#     388 |     (void)task_id;
#     389 |     (void)site;
#     390 |     (void)result_used;
#     391 |     return Ops::Load(address);
#     392 | #else
# >   393 |     if (!trace.atomics_enabled) return Ops::Load(address);
0x0000000000055aac (+0x0005585c)  4020000d  JUMPC                           #13
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055ab0 (+0x00055860)  07380000  MOV_XD_IMM                      X28, #0
0x0000000000055ab4 (+0x00055864)  1ce9eb80  LD_XD_XN_IMM.B64                X20, X30, #2944
0x0000000000055ab8 (+0x00055868)  02176800  MOV_XD_XN.S64                   X11, X22
0x0000000000055abc (+0x0005586c)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000055ac0 (+0x00055870)  0700afdb  MOV_XD_IMM                      X0, #45019
0x0000000000055ac4 (+0x00055874)  0741fffe  MOVK                            X0, #65534, #1
0x0000000000055ac8 (+0x00055878)  0781ffff  MOVK                            X0, #65535, #2
0x0000000000055acc (+0x0005587c)  40020000  JUMP                            X0, #0
# [DWARF] common/pa_scheduler_core.h:1354
#    1348 |         if (WatchdogExpired<Ops>(state, stats, start_wait, start_polls)) {
#    1349 |             break;
#    1350 |         }
#    1351 |     }
#    1352 |     AtomicPollRegionEnd<Ops>(stats.trace, stats.result, startup_poll_region);
#    1353 | 
# >  1354 |     const uint32_t batches = state->config.batches;
0x0000000000055ad0 (+0x00055880)  1cc1eaf0  LD_XD_XN_IMM.B64                X0, X30, #2800
0x0000000000055ad4 (+0x00055884)  1ca40000  LD_XD_XN_IMM.B32                X18, X0, #0
# [DWARF] common/pa_scheduler_core.h:1355
# >  1355 |     const uint32_t task_count = batches * kTasksPerBatch;
0x0000000000055ad8 (+0x00055888)  08412005  MUL_IMM.S64                     X0, X18, #5
# [DWARF] common/pa_trace.h:393
#     387 |     (void)result;
#     388 |     (void)task_id;
#     389 |     (void)site;
#     390 |     (void)result_used;
#     391 |     return Ops::Load(address);
#     392 | #else
# >   393 |     if (!trace.atomics_enabled) return Ops::Load(address);
0x0000000000055adc (+0x0005588c)  03c1ea70  ST_XD_XN_IMM.B64                X0, X30, #2672
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055ae0 (+0x00055890)  07000000  MOV_XD_IMM                      X0, #0
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x0000000000055ae4 (+0x00055894)  02828880  MOV_XD_SPR.F32                  X1, SYS_CNT
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:84
#      78 |   template <ST_L2CacheType L2Cache = ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV>  \
#      79 |   CCE_INTRINSIC[aicore] FTYPE atomicMax(__gm__ FTYPE *base, FTYPE inc) {       \
#      80 |     return __builtin_cce_atom_max_G_##SUFFIX(base, inc, (uint32_t)L2Cache);    \
#      81 |   }
#      82 | 
#      83 | __CCE__ATOM_G_BUILTIN(uint32_t, u32)
# >    84 | __CCE__ATOM_G_BUILTIN(int32_t, s32)
0x0000000000055ae8 (+0x00055898)  03c1eab0  ST_XD_XN_IMM.B64                X0, X30, #2736
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:0
# [SOURCE unavailable]
0x0000000000055aec (+0x0005589c)  07040000  MOV_XD_IMM                      X2, #0
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:84
#      78 |   template <ST_L2CacheType L2Cache = ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV>  \
#      79 |   CCE_INTRINSIC[aicore] FTYPE atomicMax(__gm__ FTYPE *base, FTYPE inc) {       \
#      80 |     return __builtin_cce_atom_max_G_##SUFFIX(base, inc, (uint32_t)L2Cache);    \
#      81 |   }
#      82 | 
#      83 | __CCE__ATOM_G_BUILTIN(uint32_t, u32)
# >    84 | __CCE__ATOM_G_BUILTIN(int32_t, s32)
0x0000000000055af0 (+0x000558a0)  50a01c40  ATOM                            XN, XM, XD, ADD
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:0
# [SOURCE unavailable]
0x0000000000055af4 (+0x000558a4)  07060001  MOV_XD_IMM                      X3, #1
0x0000000000055af8 (+0x000558a8)  02063080  NEG.S64                         X3, X3
# [DWARF] common/pa_trace.h:236
#     230 |            (encoded_retries << kAtomicRetriesShift);
#     231 | }
#     232 | 
#     233 | PA_DEVICE void CountAtomicCall(
#     234 |     TraceContext &trace, WorkerResult &result, bool poll_batch
#     235 | ) {
# >   236 |     if (result.atomic_trace_calls == UINT64_MAX) {
0x0000000000055afc (+0x000558ac)  0001318e  CMP.S64.EQ                      X19, X3
0x0000000000055b00 (+0x000558b0)  08093001  ADD_IMM.S64                     X4, X19, #1
0x0000000000055b04 (+0x000558b4)  00e63209  SEL.B64                         X19, X3, X4
0x0000000000055b08 (+0x000558b8)  07060001  MOV_XD_IMM                      X3, #1
0x0000000000055b0c (+0x000558bc)  00da3689  SEL.B64                         X13, X3, X13
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055b10 (+0x000558c0)  02176800  MOV_XD_XN.S64                   X11, X22
# [DWARF] ccec/ccec_ops.h:239
#     233 |         static_assert(sizeof(T) == 4 || sizeof(T) == 8, "atomic dependency expects a scalar result");
#     234 |         uint64_t cycle = 0;
#     235 |         // 同一个 inline asm 块先真正消费 atomic 返回寄存器，再读取
#     236 |         // SYS_CNT；编译器不能把 t1 拆到依赖 MOV 之前。AIC/AIV 对该序列
#     237 |         // 生成相同指令字节，且不增加 DSB/ISB/GM 访存。该边界仍只表示
#     238 |         // 返回值已可被本核 scalar 消费，不表示跨核全局可见。
# >   239 |         asm volatile(
0x0000000000055b14 (+0x000558c4)  02040800  MOV_XD_XN.S64                   X2, X0
0x0000000000055b18 (+0x000558c8)  02042800  MOV_XD_XN.S64                   X2, X2
0x0000000000055b1c (+0x000558cc)  02868880  MOV_XD_SPR.F32                  X3, SYS_CNT
# [DWARF] ccec/ccec_ops.h:0
# [SOURCE unavailable]
0x0000000000055b20 (+0x000558d0)  1ce9eb80  LD_XD_XN_IMM.B64                X20, X30, #2944
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x0000000000055b24 (+0x000558d4)  1cc5eb40  LD_XD_XN_IMM.B64                X2, X30, #2880
0x0000000000055b28 (+0x000558d8)  02162900  MOV_SPR_XN.S64                  CONDITION_FLAG, X2
0x0000000000055b2c (+0x000558dc)  40200002  JUMPC                           #2
0x0000000000055b30 (+0x000558e0)  40000007  JUMP                            #7
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055b34 (+0x000558e4)  07380000  MOV_XD_IMM                      X28, #0
0x0000000000055b38 (+0x000558e8)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000055b3c (+0x000558ec)  0700afc0  MOV_XD_IMM                      X0, #44992
0x0000000000055b40 (+0x000558f0)  0741fffe  MOVK                            X0, #65534, #1
0x0000000000055b44 (+0x000558f4)  0781ffff  MOVK                            X0, #65535, #2
0x0000000000055b48 (+0x000558f8)  40020000  JUMP                            X0, #0
# [DWARF] common/pa_trace.h:547
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
#     543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
# >   547 |     const uint32_t slot = core.count;
0x0000000000055b4c (+0x000558fc)  1c854000  LD_XD_XN_IMM.B32                X2, X20, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055b50 (+0x00055900)  1cddeb98  LD_XD_XN_IMM.B64                X14, X30, #2968
# [DWARF] common/pa_trace.h:548
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
#     543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
#     547 |     const uint32_t slot = core.count;
# >   548 |     if (slot >= trace.capacity) {
0x0000000000055b54 (+0x00055904)  0040272e  CMP.U64.LT                      X2, X14
0x0000000000055b58 (+0x00055908)  40200002  JUMPC                           #2
0x0000000000055b5c (+0x0005590c)  40000018  JUMP                            #24
# [DWARF] common/pa_trace.h:552
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
# >   552 |     PA_GM TraceRecord &record = trace.records[slot];
0x0000000000055b60 (+0x00055910)  1ccdeb90  LD_XD_XN_IMM.B64                X6, X30, #2960
0x0000000000055b64 (+0x00055914)  020a2800  MOV_XD_XN.S64                   X5, X2
0x0000000000055b68 (+0x00055918)  02ca0206  SHL.B64                         X5, #6
0x0000000000055b6c (+0x0005591c)  02880a00  ZEROEXT.U32                     X4, X0
0x0000000000055b70 (+0x00055920)  000a6281  ADD.S64                         X5, X6, X5
# [DWARF] common/pa_trace.h:553
# >   553 |     record.start_cycle = start_cycle;
0x0000000000055b74 (+0x00055924)  09c25181  STP_XI_XJ_XN.B64                X1, X3, X5, #0
0x0000000000055b78 (+0x00055928)  07020001  MOV_XD_IMM                      X1, #1
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055b7c (+0x0005592c)  070c0000  MOV_XD_IMM                      X6, #0
0x0000000000055b80 (+0x00055930)  02021080  NEG.S64                         X1, X1
0x0000000000055b84 (+0x00055934)  0706000e  MOV_XD_IMM                      X3, #14
0x0000000000055b88 (+0x00055938)  0000430e  CMP.S64.EQ                      X4, X6
# [DWARF] common/pa_trace.h:555
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
# >   555 |     record.task_id = task_id;
0x0000000000055b8c (+0x0005593c)  09c251a1  STP_XI_XJ_XN.B64                X1, X3, X5, #16
0x0000000000055b90 (+0x00055940)  07020050  MOV_XD_IMM                      X1, #80
0x0000000000055b94 (+0x00055944)  07060070  MOV_XD_IMM                      X3, #112
# [DWARF] common/pa_trace.h:559
#     556 |     record.function_id = function_id;
#     557 |     record.phase = static_cast<int32_t>(trace_phase);
#     558 |     record.lane = trace.lane;
# >   559 |     record.block_id = trace.block_id;
0x0000000000055b98 (+0x00055948)  08085020  ADD_IMM.S64                     X4, X5, #32
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055b9c (+0x0005594c)  00c23089  SEL.B64                         X1, X3, X1
# [DWARF] common/pa_trace.h:559
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
#     555 |     record.task_id = task_id;
#     556 |     record.function_id = function_id;
#     557 |     record.phase = static_cast<int32_t>(trace_phase);
#     558 |     record.lane = trace.lane;
# >   559 |     record.block_id = trace.block_id;
0x0000000000055ba0 (+0x00055950)  099e4781  STP_XI_XJ_XN.B32                X15, X15, X4, #0
0x0000000000055ba4 (+0x00055954)  07060002  MOV_XD_IMM                      X3, #2
# [DWARF] common/pa_trace.h:561
#     560 |     record.core_idx = trace.core_idx;
# >   561 |     record.flags = flags;
0x0000000000055ba8 (+0x00055958)  080a5028  ADD_IMM.S64                     X5, X5, #40
0x0000000000055bac (+0x0005595c)  09825181  STP_XI_XJ_XN.B32                X1, X3, X5, #0
# [DWARF] common/pa_trace.h:565
#     562 |     record.auxiliary = auxiliary;
#     563 |     // count 最后更新，使其始终指向下一空槽；单写者条件下无需 reserve/commit 两阶段。
#     564 |     // 最终 FlushTraceCore 会把记录体先于该计数一并导出。
# >   565 |     core.count = slot + 1;
0x0000000000055bb0 (+0x00055960)  08022001  ADD_IMM.S64                     X1, X2, #1
0x0000000000055bb4 (+0x00055964)  03834000  ST_XD_XN_IMM.B32                X1, X20, #0
0x0000000000055bb8 (+0x00055968)  40000004  JUMP                            #4
# [DWARF] common/pa_trace.h:549
#     543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
# >   549 |         core.dropped = core.dropped + 1;
0x0000000000055bbc (+0x0005596c)  1c834004  LD_XD_XN_IMM.B32                X1, X20, #4
0x0000000000055bc0 (+0x00055970)  08021001  ADD_IMM.S64                     X1, X1, #1
0x0000000000055bc4 (+0x00055974)  03834004  ST_XD_XN_IMM.B32                X1, X20, #4
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055bc8 (+0x00055978)  07380000  MOV_XD_IMM                      X28, #0
0x0000000000055bcc (+0x0005597c)  03c1e870  ST_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000055bd0 (+0x00055980)  0700af9e  MOV_XD_IMM                      X0, #44958
0x0000000000055bd4 (+0x00055984)  0741fffe  MOVK                            X0, #65534, #1
0x0000000000055bd8 (+0x00055988)  0781ffff  MOVK                            X0, #65535, #2
0x0000000000055bdc (+0x0005598c)  40020000  JUMP                            X0, #0
0x0000000000055be0 (+0x00055990)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000055be4 (+0x00055994)  03c1ea00  ST_XD_XN_IMM.B64                X0, X30, #2560
0x0000000000055be8 (+0x00055998)  07060001  MOV_XD_IMM                      X3, #1
0x0000000000055bec (+0x0005599c)  03c5ea10  ST_XD_XN_IMM.B64                X2, X30, #2576
0x0000000000055bf0 (+0x000559a0)  07040001  MOV_XD_IMM                      X2, #1
0x0000000000055bf4 (+0x000559a4)  03c3ea18  ST_XD_XN_IMM.B64                X1, X30, #2584
0x0000000000055bf8 (+0x000559a8)  02063080  NEG.S64                         X3, X3
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x0000000000055bfc (+0x000559ac)  02808880  MOV_XD_SPR.F32                  X0, SYS_CNT
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:45
#      39 |   CCE_INTRINSIC[aicore] FTYPE atomicSub(__gm__ FTYPE *base, FTYPE inc) {       \
#      40 |     return __builtin_cce_atom_add_G_##ADD_SUFFIX(base, -inc,                   \
#      41 |                                                   (uint32_t)L2Cache);          \
#      42 |   }
#      43 | 
#      44 | __CCE__ATOM_G_CAS_EXCH_INT32(uint32_t, u32, u32)
# >    45 | __CCE__ATOM_G_CAS_EXCH_INT32(int32_t, s32, s32)
0x0000000000055c00 (+0x000559b0)  1cc3eac8  LD_XD_XN_IMM.B64                X1, X30, #2760
# [DWARF] common/pa_trace.h:236
#     230 |            (encoded_retries << kAtomicRetriesShift);
#     231 | }
#     232 | 
#     233 | PA_DEVICE void CountAtomicCall(
#     234 |     TraceContext &trace, WorkerResult &result, bool poll_batch
#     235 | ) {
# >   236 |     if (result.atomic_trace_calls == UINT64_MAX) {
0x0000000000055c04 (+0x000559b4)  0001318e  CMP.S64.EQ                      X19, X3
0x0000000000055c08 (+0x000559b8)  08093001  ADD_IMM.S64                     X4, X19, #1
0x0000000000055c0c (+0x000559bc)  00e63209  SEL.B64                         X19, X3, X4
0x0000000000055c10 (+0x000559c0)  00da2689  SEL.B64                         X13, X2, X13
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x0000000000055c14 (+0x000559c4)  02175900  MOV_SPR_XN.S64                  CONDITION_FLAG, X21
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:45
#      39 |   CCE_INTRINSIC[aicore] FTYPE atomicSub(__gm__ FTYPE *base, FTYPE inc) {       \
#      40 |     return __builtin_cce_atom_add_G_##ADD_SUFFIX(base, -inc,                   \
#      41 |                                                   (uint32_t)L2Cache);          \
#      42 |   }
#      43 | 
#      44 | __CCE__ATOM_G_CAS_EXCH_INT32(uint32_t, u32, u32)
# >    45 | __CCE__ATOM_G_CAS_EXCH_INT32(int32_t, s32, s32)
0x0000000000055c18 (+0x000559c8)  50108440  ATOM                            XN, XM, XD, EXCH
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x0000000000055c1c (+0x000559cc)  02828880  MOV_XD_SPR.F32                  X1, SYS_CNT
# [DWARF] ccec/ccec_ops.h:0
# [SOURCE unavailable]
0x0000000000055c20 (+0x000559d0)  1ce9eb80  LD_XD_XN_IMM.B64                X20, X30, #2944
0x0000000000055c24 (+0x000559d4)  1cddeb98  LD_XD_XN_IMM.B64                X14, X30, #2968
0x0000000000055c28 (+0x000559d8)  1cf5eb58  LD_XD_XN_IMM.B64                X26, X30, #2904
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x0000000000055c2c (+0x000559dc)  40200002  JUMPC                           #2
0x0000000000055c30 (+0x000559e0)  4000006e  JUMP                            #110
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055c34 (+0x000559e4)  1cc1ea30  LD_XD_XN_IMM.B64                X0, X30, #2608
0x0000000000055c38 (+0x000559e8)  02120800  MOV_XD_XN.S64                   X9, X0
0x0000000000055c3c (+0x000559ec)  03c1ea40  ST_XD_XN_IMM.B64                X0, X30, #2624
0x0000000000055c40 (+0x000559f0)  40000002  JUMP                            #2
0x0000000000055c44 (+0x000559f4)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000055c48 (+0x000559f8)  1cc1e950  LD_XD_XN_IMM.B64                X0, X30, #2384
0x0000000000055c4c (+0x000559fc)  02398800  MOV_XD_XN.S64                   X28, X24
0x0000000000055c50 (+0x00055a00)  03c1eab0  ST_XD_XN_IMM.B64                X0, X30, #2736
0x0000000000055c54 (+0x00055a04)  40000002  JUMP                            #2
0x0000000000055c58 (+0x00055a08)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000055c5c (+0x00055a0c)  070c0000  MOV_XD_IMM                      X6, #0
0x0000000000055c60 (+0x00055a10)  400000c3  JUMP                            #195
0x0000000000055c64 (+0x00055a14)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
# [DWARF] common/pa_trace.h:547
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
#     543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
# >   547 |     const uint32_t slot = core.count;
0x0000000000055c68 (+0x00055a18)  1c854000  LD_XD_XN_IMM.B32                X2, X20, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055c6c (+0x00055a1c)  070c0000  MOV_XD_IMM                      X6, #0
# [DWARF] common/pa_trace.h:548
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
#     543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
#     547 |     const uint32_t slot = core.count;
# >   548 |     if (slot >= trace.capacity) {
0x0000000000055c70 (+0x00055a20)  0040272e  CMP.U64.LT                      X2, X14
0x0000000000055c74 (+0x00055a24)  40200002  JUMPC                           #2
0x0000000000055c78 (+0x00055a28)  40000079  JUMP                            #121
# [DWARF] common/pa_trace.h:552
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
# >   552 |     PA_GM TraceRecord &record = trace.records[slot];
0x0000000000055c7c (+0x00055a2c)  1cc9eb90  LD_XD_XN_IMM.B64                X4, X30, #2960
0x0000000000055c80 (+0x00055a30)  02062800  MOV_XD_XN.S64                   X3, X2
0x0000000000055c84 (+0x00055a34)  03cbea00  ST_XD_XN_IMM.B64                X5, X30, #2560
0x0000000000055c88 (+0x00055a38)  02c60206  SHL.B64                         X3, #6
0x0000000000055c8c (+0x00055a3c)  1cd3ea30  LD_XD_XN_IMM.B64                X9, X30, #2608
0x0000000000055c90 (+0x00055a40)  00064181  ADD.S64                         X3, X4, X3
# [DWARF] common/pa_trace.h:555
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
# >   555 |     record.task_id = task_id;
0x0000000000055c94 (+0x00055a44)  1cc9eaa8  LD_XD_XN_IMM.B64                X4, X30, #2728
# [DWARF] common/pa_trace.h:553
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
# >   553 |     record.start_cycle = start_cycle;
0x0000000000055c98 (+0x00055a48)  09c03081  STP_XI_XJ_XN.B64                X0, X1, X3, #0
0x0000000000055c9c (+0x00055a4c)  0700ffff  MOV_XD_IMM                      X0, #65535
0x0000000000055ca0 (+0x00055a50)  0741ffff  MOVK                            X0, #65535, #1
0x0000000000055ca4 (+0x00055a54)  0702000e  MOV_XD_IMM                      X1, #14
# [DWARF] common/pa_trace.h:555
#     554 |     record.end_cycle = end_cycle;
# >   555 |     record.task_id = task_id;
0x0000000000055ca8 (+0x00055a58)  09883021  STP_XI_XJ_XN.B32                X4, X0, X3, #16
0x0000000000055cac (+0x00055a5c)  07000001  MOV_XD_IMM                      X0, #1
# [DWARF] common/pa_trace.h:557
#     556 |     record.function_id = function_id;
# >   557 |     record.phase = static_cast<int32_t>(trace_phase);
0x0000000000055cb0 (+0x00055a60)  03c23018  ST_XD_XN_IMM.B64                X1, X3, #24
# [DWARF] common/pa_trace.h:565
#     558 |     record.lane = trace.lane;
#     559 |     record.block_id = trace.block_id;
#     560 |     record.core_idx = trace.core_idx;
#     561 |     record.flags = flags;
#     562 |     record.auxiliary = auxiliary;
#     563 |     // count 最后更新，使其始终指向下一空槽；单写者条件下无需 reserve/commit 两阶段。
#     564 |     // 最终 FlushTraceCore 会把记录体先于该计数一并导出。
# >   565 |     core.count = slot + 1;
0x0000000000055cb4 (+0x00055a64)  08022001  ADD_IMM.S64                     X1, X2, #1
0x0000000000055cb8 (+0x00055a68)  1cc5eb30  LD_XD_XN_IMM.B64                X2, X30, #2864
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055cbc (+0x00055a6c)  07810003  MOVK                            X0, #3, #2
# [DWARF] common/pa_trace.h:559
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
#     555 |     record.task_id = task_id;
#     556 |     record.function_id = function_id;
#     557 |     record.phase = static_cast<int32_t>(trace_phase);
#     558 |     record.lane = trace.lane;
# >   559 |     record.block_id = trace.block_id;
0x0000000000055cc0 (+0x00055a70)  08083020  ADD_IMM.S64                     X4, X3, #32
0x0000000000055cc4 (+0x00055a74)  09844101  STP_XI_XJ_XN.B32                X2, X2, X4, #0
0x0000000000055cc8 (+0x00055a78)  40000061  JUMP                            #97
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055ccc (+0x00055a7c)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
0x0000000000055cd0 (+0x00055a80)  03c1ea18  ST_XD_XN_IMM.B64                X0, X30, #2584
0x0000000000055cd4 (+0x00055a84)  02102800  MOV_XD_XN.S64                   X8, X2
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x0000000000055cd8 (+0x00055a88)  02808880  MOV_XD_SPR.F32                  X0, SYS_CNT
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:45
#      39 |   CCE_INTRINSIC[aicore] FTYPE atomicSub(__gm__ FTYPE *base, FTYPE inc) {       \
#      40 |     return __builtin_cce_atom_add_G_##ADD_SUFFIX(base, -inc,                   \
#      41 |                                                   (uint32_t)L2Cache);          \
#      42 |   }
#      43 | 
#      44 | __CCE__ATOM_G_CAS_EXCH_INT32(uint32_t, u32, u32)
# >    45 | __CCE__ATOM_G_CAS_EXCH_INT32(int32_t, s32, s32)
0x0000000000055cdc (+0x00055a8c)  1cc3eac8  LD_XD_XN_IMM.B64                X1, X30, #2760
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:0
# [SOURCE unavailable]
0x0000000000055ce0 (+0x00055a90)  07040001  MOV_XD_IMM                      X2, #1
0x0000000000055ce4 (+0x00055a94)  07060001  MOV_XD_IMM                      X3, #1
0x0000000000055ce8 (+0x00055a98)  02063080  NEG.S64                         X3, X3
# [DWARF] common/pa_trace.h:236
#     230 |            (encoded_retries << kAtomicRetriesShift);
#     231 | }
#     232 | 
#     233 | PA_DEVICE void CountAtomicCall(
#     234 |     TraceContext &trace, WorkerResult &result, bool poll_batch
#     235 | ) {
# >   236 |     if (result.atomic_trace_calls == UINT64_MAX) {
0x0000000000055cec (+0x00055a9c)  0001318e  CMP.S64.EQ                      X19, X3
0x0000000000055cf0 (+0x00055aa0)  08093001  ADD_IMM.S64                     X4, X19, #1
0x0000000000055cf4 (+0x00055aa4)  00e63209  SEL.B64                         X19, X3, X4
0x0000000000055cf8 (+0x00055aa8)  00da2689  SEL.B64                         X13, X2, X13
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x0000000000055cfc (+0x00055aac)  02175900  MOV_SPR_XN.S64                  CONDITION_FLAG, X21
# [DWARF] $ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_atomic.h:45
#      39 |   CCE_INTRINSIC[aicore] FTYPE atomicSub(__gm__ FTYPE *base, FTYPE inc) {       \
#      40 |     return __builtin_cce_atom_add_G_##ADD_SUFFIX(base, -inc,                   \
#      41 |                                                   (uint32_t)L2Cache);          \
#      42 |   }
#      43 | 
#      44 | __CCE__ATOM_G_CAS_EXCH_INT32(uint32_t, u32, u32)
# >    45 | __CCE__ATOM_G_CAS_EXCH_INT32(int32_t, s32, s32)
0x0000000000055d00 (+0x00055ab0)  50108440  ATOM                            XN, XM, XD, EXCH
# [DWARF] ccec/ccec_ops.h:229
#     223 |     // coherency is handled by the runtime's DCCI protocol.
#     224 |     // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
#     225 |     // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
#     226 |     // 可见性则由各自既有的 DCCI 路径处理。
#     227 |     __aicore__ static inline void StoreBarrier() {}
#     228 | 
# >   229 |     __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }
0x0000000000055d04 (+0x00055ab4)  02828880  MOV_XD_SPR.F32                  X1, SYS_CNT
# [DWARF] ccec/ccec_ops.h:0
# [SOURCE unavailable]
0x0000000000055d08 (+0x00055ab8)  1ce9eb80  LD_XD_XN_IMM.B64                X20, X30, #2944
0x0000000000055d0c (+0x00055abc)  1cddeb98  LD_XD_XN_IMM.B64                X14, X30, #2968
0x0000000000055d10 (+0x00055ac0)  1cf5eb58  LD_XD_XN_IMM.B64                X26, X30, #2904
# [DWARF] common/pa_trace.h:543
#     537 |     (void)auxiliary;
#     538 |     return;
#     539 | #else
#     540 |     // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
# >   543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
0x0000000000055d14 (+0x00055ac4)  40200002  JUMPC                           #2
0x0000000000055d18 (+0x00055ac8)  40000056  JUMP                            #86
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055d1c (+0x00055acc)  1cc1ea30  LD_XD_XN_IMM.B64                X0, X30, #2608
0x0000000000055d20 (+0x00055ad0)  03d1ea10  ST_XD_XN_IMM.B64                X8, X30, #2576
0x0000000000055d24 (+0x00055ad4)  1cf9eb38  LD_XD_XN_IMM.B64                X28, X30, #2872
0x0000000000055d28 (+0x00055ad8)  02120800  MOV_XD_XN.S64                   X9, X0
0x0000000000055d2c (+0x00055adc)  03c1ea40  ST_XD_XN_IMM.B64                X0, X30, #2624
0x0000000000055d30 (+0x00055ae0)  1cc1e948  LD_XD_XN_IMM.B64                X0, X30, #2376
0x0000000000055d34 (+0x00055ae4)  03c1ea00  ST_XD_XN_IMM.B64                X0, X30, #2560
0x0000000000055d38 (+0x00055ae8)  1cc1e930  LD_XD_XN_IMM.B64                X0, X30, #2352
0x0000000000055d3c (+0x00055aec)  03c1ea08  ST_XD_XN_IMM.B64                X0, X30, #2568
0x0000000000055d40 (+0x00055af0)  1cc1e918  LD_XD_XN_IMM.B64                X0, X30, #2328
0x0000000000055d44 (+0x00055af4)  4000ffc3  JUMP                            #65475
0x0000000000055d48 (+0x00055af8)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
# [DWARF] common/pa_trace.h:547
#     541 |     // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
#     543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
# >   547 |     const uint32_t slot = core.count;
0x0000000000055d4c (+0x00055afc)  1c854000  LD_XD_XN_IMM.B32                X2, X20, #0
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055d50 (+0x00055b00)  070c0000  MOV_XD_IMM                      X6, #0
# [DWARF] common/pa_trace.h:548
#     542 |     AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
#     543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
#     547 |     const uint32_t slot = core.count;
# >   548 |     if (slot >= trace.capacity) {
0x0000000000055d54 (+0x00055b04)  0040272e  CMP.U64.LT                      X2, X14
0x0000000000055d58 (+0x00055b08)  40200002  JUMPC                           #2
0x0000000000055d5c (+0x00055b0c)  4000006e  JUMP                            #110
# [DWARF] common/pa_trace.h:552
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
# >   552 |     PA_GM TraceRecord &record = trace.records[slot];
0x0000000000055d60 (+0x00055b10)  1cc9eb90  LD_XD_XN_IMM.B64                X4, X30, #2960
0x0000000000055d64 (+0x00055b14)  02062800  MOV_XD_XN.S64                   X3, X2
0x0000000000055d68 (+0x00055b18)  03cbea00  ST_XD_XN_IMM.B64                X5, X30, #2560
0x0000000000055d6c (+0x00055b1c)  02c60206  SHL.B64                         X3, #6
0x0000000000055d70 (+0x00055b20)  02398800  MOV_XD_XN.S64                   X28, X24
0x0000000000055d74 (+0x00055b24)  00064181  ADD.S64                         X3, X4, X3
# [DWARF] common/pa_trace.h:555
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
# >   555 |     record.task_id = task_id;
0x0000000000055d78 (+0x00055b28)  1cc9eaa8  LD_XD_XN_IMM.B64                X4, X30, #2728
# [DWARF] common/pa_trace.h:553
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
# >   553 |     record.start_cycle = start_cycle;
0x0000000000055d7c (+0x00055b2c)  09c03081  STP_XI_XJ_XN.B64                X0, X1, X3, #0
0x0000000000055d80 (+0x00055b30)  0700ffff  MOV_XD_IMM                      X0, #65535
0x0000000000055d84 (+0x00055b34)  0741ffff  MOVK                            X0, #65535, #1
0x0000000000055d88 (+0x00055b38)  0702000e  MOV_XD_IMM                      X1, #14
# [DWARF] common/pa_trace.h:559
#     554 |     record.end_cycle = end_cycle;
#     555 |     record.task_id = task_id;
#     556 |     record.function_id = function_id;
#     557 |     record.phase = static_cast<int32_t>(trace_phase);
#     558 |     record.lane = trace.lane;
# >   559 |     record.block_id = trace.block_id;
0x0000000000055d8c (+0x00055b3c)  080e3020  ADD_IMM.S64                     X7, X3, #32
# [DWARF] common/pa_trace.h:555
#     549 |         core.dropped = core.dropped + 1;
#     550 |         return;
#     551 |     }
#     552 |     PA_GM TraceRecord &record = trace.records[slot];
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
# >   555 |     record.task_id = task_id;
0x0000000000055d90 (+0x00055b40)  09883021  STP_XI_XJ_XN.B32                X4, X0, X3, #16
0x0000000000055d94 (+0x00055b44)  1cc9eb30  LD_XD_XN_IMM.B64                X4, X30, #2864
0x0000000000055d98 (+0x00055b48)  07000001  MOV_XD_IMM                      X0, #1
# [DWARF] common/pa_trace.h:557
#     556 |     record.function_id = function_id;
# >   557 |     record.phase = static_cast<int32_t>(trace_phase);
0x0000000000055d9c (+0x00055b4c)  03c23018  ST_XD_XN_IMM.B64                X1, X3, #24
# [DWARF] common/pa_trace.h:565
#     558 |     record.lane = trace.lane;
#     559 |     record.block_id = trace.block_id;
#     560 |     record.core_idx = trace.core_idx;
#     561 |     record.flags = flags;
#     562 |     record.auxiliary = auxiliary;
#     563 |     // count 最后更新，使其始终指向下一空槽；单写者条件下无需 reserve/commit 两阶段。
#     564 |     // 最终 FlushTraceCore 会把记录体先于该计数一并导出。
# >   565 |     core.count = slot + 1;
0x0000000000055da0 (+0x00055b50)  08022001  ADD_IMM.S64                     X1, X2, #1
0x0000000000055da4 (+0x00055b54)  1cc5ea30  LD_XD_XN_IMM.B64                X2, X30, #2608
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055da8 (+0x00055b58)  07810003  MOVK                            X0, #3, #2
# [DWARF] common/pa_trace.h:559
#     553 |     record.start_cycle = start_cycle;
#     554 |     record.end_cycle = end_cycle;
#     555 |     record.task_id = task_id;
#     556 |     record.function_id = function_id;
#     557 |     record.phase = static_cast<int32_t>(trace_phase);
#     558 |     record.lane = trace.lane;
# >   559 |     record.block_id = trace.block_id;
0x0000000000055dac (+0x00055b5c)  09887201  STP_XI_XJ_XN.B32                X4, X4, X7, #0
0x0000000000055db0 (+0x00055b60)  02122800  MOV_XD_XN.S64                   X9, X2
0x0000000000055db4 (+0x00055b64)  03c5ea40  ST_XD_XN_IMM.B64                X2, X30, #2624
# [DWARF] common/pa_trace.h:561
#     560 |     record.core_idx = trace.core_idx;
# >   561 |     record.flags = flags;
0x0000000000055db8 (+0x00055b68)  03c03028  ST_XD_XN_IMM.B64                X0, X3, #40
# [DWARF] common/pa_trace.h:565
#     562 |     record.auxiliary = auxiliary;
#     563 |     // count 最后更新，使其始终指向下一空槽；单写者条件下无需 reserve/commit 两阶段。
#     564 |     // 最终 FlushTraceCore 会把记录体先于该计数一并导出。
# >   565 |     core.count = slot + 1;
0x0000000000055dbc (+0x00055b6c)  03834000  ST_XD_XN_IMM.B32                X1, X20, #0
0x0000000000055dc0 (+0x00055b70)  4000006b  JUMP                            #107
# [DWARF] common/pa_trace.h:0
# [SOURCE unavailable]
0x0000000000055dc4 (+0x00055b74)  1cc1e870  LD_XD_XN_IMM.B64                X0, X30, #2160
# [DWARF] common/pa_trace.h:549
#     543 |     if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
#     544 |         return;
#     545 |     }
#     546 |     PA_GM TraceCoreState &core = *trace.core;
#     547 |     const uint32_t slot = core.count;
#     548 |     if (slot >= trace.capacity) {
# >   549 |         core.dropped = core.dropped + 1;
0x0000000000055dc8 (+0x00055b78)  1c814004  LD_XD_XN_IMM.B32                X0, X20, #4
0x0000000000055dcc (+0x00055b7c)  03cfea18  ST_XD_XN_IMM.B64                X7, X30, #2584
