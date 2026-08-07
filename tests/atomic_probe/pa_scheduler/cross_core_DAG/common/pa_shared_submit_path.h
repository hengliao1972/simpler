/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef PA_SCHEDULER_COMMON_PA_SHARED_SUBMIT_PATH_H
#define PA_SCHEDULER_COMMON_PA_SHARED_SUBMIT_PATH_H

// 本文件由 pa_scheduler_core.h 在 pa_scheduler 命名空间内引入。shared
// TensorMap 的 Submit 控制流独立放在这里，避免继续把两套协议塞进同一
// 个宏分支密集的热函数；CPU 与 CCEC 仍复用相同的 Ops 和底层原语。

// 公共 DAG 层只消费 adapter 提供的 access、reference kind 和 packed
// SharedOutputRef key。PA TaskKind、固定五 task 形状以及参数构造细节都必须
// 留在 schema adapter 中，不能渗入下面的逻辑前驱算法。
struct SharedDagTensor {
    TensorArgType access;
    TensorRefKind reference_kind;
    uint32_t symbol_key;
    bool manual_dependency;
};

struct SharedMetadataDag {
    uint32_t writer_symbol_keys[kMaxTaskTensors];
    int32_t writer_previous[kMaxTaskTensors];
    int32_t dependencies[kMaxTaskTensors];
    int32_t prepared_task_id;
    uint32_t writer_count;
    uint32_t dependency_count;
    int32_t ordinary_previous_writer;
    bool ordinary_writer;
    bool ordinary_dependency_required;
};
static_assert(
    __is_trivially_constructible(SharedMetadataDag),
    "shared metadata DAG must remain trivial for CCEC local state"
);

PA_DEVICE bool SharedDagAccessReadsExistingValue(TensorArgType access) {
    return access == TensorArgType::Input ||
           access == TensorArgType::Inout ||
           access == TensorArgType::OutputExisting;
}

PA_DEVICE bool SharedDagAddDependency(
    SharedMetadataDag &dag, int32_t dependency
) {
    if (dependency < 0) {
        return true;
    }
    for (uint32_t index = 0; index < dag.dependency_count; ++index) {
        if (dag.dependencies[index] == dependency) {
            return true;
        }
    }
    if (dag.dependency_count >= kMaxTaskTensors) {
        return false;
    }
    dag.dependencies[dag.dependency_count++] = dependency;
    return true;
}

template <typename Schema>
PA_DEVICE bool SharedDagTaskHasSymbolWriter(
    const Schema &schema, uint32_t task_id, uint32_t expected_key,
    bool &has_writer
) {
    has_writer = false;
    uint32_t tensor_count = 0;
    if (!schema.TensorCount(task_id, tensor_count) ||
        tensor_count > kMaxTaskTensors) {
        return false;
    }
    for (uint32_t index = 0; index < tensor_count; ++index) {
        SharedDagTensor tensor{};
        if (!schema.TensorAt(task_id, index, tensor)) {
            return false;
        }
        if (!IsSharedWriterIntentTag(tensor.access) ||
            tensor.reference_kind != TensorRefKind::SharedOutputRef) {
            continue;
        }
        const FdwicOutputRef output_ref =
            SharedSymbolHistoryReference(tensor.symbol_key);
        if (!IsPlainSharedOutputRef(output_ref) ||
            output_ref.producer_task_id < 0 ||
            output_ref.producer_task_id >=
                static_cast<int32_t>(task_id)) {
            return false;
        }
        if (tensor.symbol_key != expected_key) {
            continue;
        }
        if (has_writer) {
            // 同一 task 对同一 symbol 发布两次会让第二次 CAS 把本 task
            // 自己当成 expected-old，必须在任何共享状态修改前拒绝。
            return false;
        }
        has_writer = true;
    }
    return true;
}

template <typename Schema>
PA_DEVICE bool SharedDagFindPreviousSymbolWriter(
    const Schema &schema, uint32_t task_id, uint32_t symbol_key,
    int32_t &previous_writer
) {
    const FdwicOutputRef output_ref =
        SharedSymbolHistoryReference(symbol_key);
    if (!IsPlainSharedOutputRef(output_ref) ||
        output_ref.producer_task_id < 0 ||
        output_ref.producer_task_id >=
            static_cast<int32_t>(task_id)) {
        return false;
    }
    previous_writer = output_ref.producer_task_id;
    int32_t candidate = static_cast<int32_t>(task_id) - 1;
    while (candidate > output_ref.producer_task_id) {
        bool has_writer = false;
        if (!SharedDagTaskHasSymbolWriter(
                schema, static_cast<uint32_t>(candidate),
                symbol_key, has_writer
            )) {
            return false;
        }
        if (has_writer) {
            previous_writer = candidate;
            return true;
        }
        --candidate;
    }
    return true;
}

template <typename Schema>
PA_DEVICE bool SharedDagTaskHasOrdinaryWriter(
    const Schema &schema, uint32_t task_id, bool &has_writer
) {
    has_writer = false;
    uint32_t tensor_count = 0;
    if (!schema.TensorCount(task_id, tensor_count) ||
        tensor_count > kMaxTaskTensors) {
        return false;
    }
    for (uint32_t index = 0; index < tensor_count; ++index) {
        SharedDagTensor tensor{};
        if (!schema.TensorAt(task_id, index, tensor)) {
            return false;
        }
        if (!IsSharedWriterIntentTag(tensor.access) ||
            tensor.manual_dependency) {
            continue;
        }
        if (tensor.reference_kind == TensorRefKind::GmTensor ||
            tensor.reference_kind == TensorRefKind::LocalTensor) {
            has_writer = true;
        }
    }
    return true;
}

template <typename Schema>
PA_DEVICE bool SharedDagFindPreviousOrdinaryWriter(
    const Schema &schema, uint32_t task_id,
    int32_t &previous_writer
) {
    previous_writer = -1;
    int32_t candidate = static_cast<int32_t>(task_id) - 1;
    while (candidate >= 0) {
        bool has_writer = false;
        if (!SharedDagTaskHasOrdinaryWriter(
                schema, static_cast<uint32_t>(candidate), has_writer
            )) {
            return false;
        }
        if (has_writer) {
            previous_writer = candidate;
            return true;
        }
        --candidate;
    }
    return true;
}

template <typename Schema>
PA_DEVICE bool BuildSharedMetadataDag(
    const Schema &schema, uint32_t task_id, SharedMetadataDag &dag
) {
    dag.prepared_task_id = -1;
    dag.writer_count = 0;
    dag.dependency_count = 0;
    dag.ordinary_previous_writer = -1;
    dag.ordinary_writer = false;
    dag.ordinary_dependency_required = false;
    const uint32_t task_count = schema.TaskCount();
    uint32_t tensor_count = 0;
    if (task_count == 0 || task_count > kMaxTasks ||
        task_id >= task_count ||
        !schema.TensorCount(task_id, tensor_count) ||
        tensor_count > kMaxTaskTensors) {
        return false;
    }

    for (uint32_t index = 0; index < tensor_count; ++index) {
        SharedDagTensor tensor{};
        if (!schema.TensorAt(task_id, index, tensor)) {
            return false;
        }
        if (tensor.reference_kind == TensorRefKind::SharedOutputRef) {
            if (!SharedDagAccessReadsExistingValue(tensor.access)) {
                return false;
            }
            int32_t previous_writer = -1;
            if (!SharedDagFindPreviousSymbolWriter(
                    schema, task_id, tensor.symbol_key,
                    previous_writer
                )) {
                return false;
            }
            const FdwicOutputRef output_ref =
                SharedSymbolHistoryReference(tensor.symbol_key);
            if (previous_writer != output_ref.producer_task_id &&
                !SharedDagAddDependency(dag, previous_writer)) {
                return false;
            }
            if (!IsSharedWriterIntentTag(tensor.access)) {
                continue;
            }
            for (uint32_t writer = 0;
                 writer < dag.writer_count; ++writer) {
                if (dag.writer_symbol_keys[writer] ==
                    tensor.symbol_key) {
                    return false;
                }
            }
            if (dag.writer_count >= kMaxTaskTensors) {
                return false;
            }
            dag.writer_symbol_keys[dag.writer_count] =
                tensor.symbol_key;
            dag.writer_previous[dag.writer_count] =
                previous_writer;
            ++dag.writer_count;
            continue;
        }

        if (tensor.reference_kind == TensorRefKind::GmTensor ||
            tensor.reference_kind == TensorRefKind::LocalTensor) {
            if (tensor.manual_dependency ||
                !SharedDagAccessReadsExistingValue(tensor.access)) {
                continue;
            }
            dag.ordinary_dependency_required = true;
            dag.ordinary_writer |=
                IsSharedWriterIntentTag(tensor.access);
            continue;
        }

        // CreateInfo 是输出构造描述，不读旧值也不登记 writer。若 adapter
        // 把它标成依赖访问，说明 schema 与实际 TaskArgs 已经漂移。
        if (tensor.reference_kind == TensorRefKind::CreateInfo &&
            !SharedDagAccessReadsExistingValue(tensor.access)) {
            continue;
        }
        return false;
    }

    if (dag.ordinary_dependency_required) {
        if (!SharedDagFindPreviousOrdinaryWriter(
                schema, task_id, dag.ordinary_previous_writer
            ) ||
            !SharedDagAddDependency(
                dag, dag.ordinary_previous_writer
            )) {
            return false;
        }
    }
    dag.prepared_task_id = static_cast<int32_t>(task_id);
    return true;
}

// PA schema adapter 只负责把当前 workload 的参数形状翻译为公共 DAG 输入。
// 动态前驱算法本身位于上方模板，不读取这里的 TaskKind 分支。
struct SharedPaDagSchema {
    PA_GM const SharedBuildDispatchState *dispatch;

    PA_DEVICE uint32_t TaskCount() const {
        return dispatch == nullptr ? 0U : dispatch->task_count;
    }

    PA_DEVICE bool TensorCount(
        uint32_t task_id, uint32_t &tensor_count
    ) const {
        SharedBuildDispatchTask task{};
        if (dispatch == nullptr ||
            !DecodeSharedBuildDispatchTask(
                *dispatch, task_id, task
            )) {
            return false;
        }
        switch (task.meta.kind) {
            case TaskKind::Alloc:
                tensor_count = 3;
                return true;
            case TaskKind::Qk:
            case TaskKind::Sf:
            case TaskKind::Pv:
                tensor_count = 4;
                return true;
            case TaskKind::Up:
                tensor_count = 7;
                return true;
            case TaskKind::Count:
                return false;
        }
        return false;
    }

    PA_DEVICE bool TensorAt(
        uint32_t task_id, uint32_t tensor_index,
        SharedDagTensor &tensor
    ) const {
        uint32_t tensor_count = 0;
        SharedBuildDispatchTask task{};
        if (dispatch == nullptr ||
            !DecodeSharedBuildDispatchTask(
                *dispatch, task_id, task
            ) ||
            !TensorCount(task_id, tensor_count) ||
            tensor_index >= tensor_count) {
            return false;
        }
        tensor.access = TensorArgType::Input;
        tensor.reference_kind = TensorRefKind::LocalTensor;
        tensor.symbol_key = 0;
        tensor.manual_dependency = false;

        const TaskKind kind = task.meta.kind;
        if (kind == TaskKind::Alloc) {
            tensor.access = TensorArgType::Output;
            tensor.reference_kind = TensorRefKind::CreateInfo;
            return true;
        }
        if (kind == TaskKind::Qk) {
            if (tensor_index == 3) {
                tensor.access = TensorArgType::Output;
                tensor.reference_kind = TensorRefKind::CreateInfo;
            }
            return true;
        }

        uint32_t producer = UINT32_MAX;
        uint32_t output_slot = 0;
        if (kind == TaskKind::Sf) {
            if (tensor_index == 0) {
                producer = task.meta.batch_start +
                    SharedPaTaskOffset(
                        TaskKind::Qk, task.meta.group_index
                    );
            } else {
                tensor.access = TensorArgType::Output;
                tensor.reference_kind = TensorRefKind::CreateInfo;
                return true;
            }
        } else if (kind == TaskKind::Pv) {
            if (tensor_index == 0) {
                producer = task.meta.batch_start +
                    SharedPaTaskOffset(
                        TaskKind::Sf, task.meta.group_index
                    );
            } else if (tensor_index == 3) {
                tensor.access = TensorArgType::Output;
                tensor.reference_kind = TensorRefKind::CreateInfo;
                return true;
            } else {
                return true;
            }
        } else if (kind == TaskKind::Up) {
            if (tensor_index == 0 || tensor_index == 1) {
                producer = task.meta.batch_start +
                    SharedPaTaskOffset(
                        TaskKind::Sf, task.meta.group_index
                    );
                output_slot = tensor_index + 1U;
            } else if (tensor_index == 2) {
                producer = task.meta.batch_start +
                    SharedPaTaskOffset(
                        TaskKind::Pv, task.meta.group_index
                    );
            } else if (tensor_index >= 3 && tensor_index <= 5) {
                producer = task.meta.batch_start;
                output_slot = 5U - tensor_index;
                tensor.access = TensorArgType::Inout;
            } else {
                tensor.access = TensorArgType::Inout;
                tensor.reference_kind = TensorRefKind::LocalTensor;
                tensor.manual_dependency = true;
                return true;
            }
        } else {
            return false;
        }

        if (producer >= task_id ||
            output_slot >= kSharedOutputMaxPerTask) {
            return false;
        }
        tensor.reference_kind = TensorRefKind::SharedOutputRef;
        tensor.symbol_key = producer * kSharedOutputMaxPerTask +
            output_slot + 1U;
        return true;
    }
};

template <typename Schema>
PA_DEVICE bool ValidateSharedDagTaskArgs(
    const Schema &schema, uint32_t task_id, const TaskArgs &args
) {
    uint32_t tensor_count = 0;
    if (args.has_error || args.tensor_count < 0 ||
        args.tensor_count > static_cast<int32_t>(kMaxTaskTensors) ||
        !schema.TensorCount(task_id, tensor_count) ||
        tensor_count != static_cast<uint32_t>(args.tensor_count)) {
        return false;
    }
    for (uint32_t index = 0; index < tensor_count; ++index) {
        SharedDagTensor expected{};
        if (!schema.TensorAt(task_id, index, expected)) {
            return false;
        }
        const TaskTensorRef &reference = args.tensors[index];
        if (expected.access != TaskTag(args, index) ||
            expected.reference_kind != reference.kind) {
            return false;
        }
        if (reference.kind == TensorRefKind::SharedOutputRef) {
            uint32_t symbol_key = 0;
            if (!SharedSymbolHistoryKey(
                    SharedOutputReference(reference), symbol_key
                ) ||
                symbol_key != expected.symbol_key) {
                return false;
            }
        } else if (reference.kind == TensorRefKind::GmTensor) {
            if (reference.pointer.gm_tensor == nullptr ||
                reference.pointer.gm_tensor->manual_dep !=
                    expected.manual_dependency) {
                return false;
            }
        } else if (reference.kind == TensorRefKind::LocalTensor) {
            if (reference.pointer.local_tensor == nullptr ||
                reference.pointer.local_tensor->manual_dep !=
                    expected.manual_dependency) {
                return false;
            }
        } else if (reference.kind != TensorRefKind::CreateInfo) {
            return false;
        }
    }
    return true;
}

PA_DEVICE bool SharedTensorArgNeedsMetadataPrefix(
    const TaskTensorRef &reference, TensorArgType tag,
    int32_t task_id, int32_t previous_metadata_writer,
    bool ordinary_metadata_writers_exist,
    bool &required
) {
    required = false;
    if (tag == TensorArgType::Output) {
        return true;
    }
    if (tag != TensorArgType::Input &&
        tag != TensorArgType::Inout &&
        tag != TensorArgType::OutputExisting) {
        return false;
    }
    if (reference.kind == TensorRefKind::SharedOutputRef) {
        const FdwicOutputRef output_ref =
            SharedOutputReference(reference);
        if (!IsPlainSharedOutputRef(output_ref) ||
            output_ref.producer_task_id < 0 ||
            output_ref.producer_task_id >= task_id) {
            return false;
        }
        required = previous_metadata_writer >
            output_ref.producer_task_id;
        return true;
    }
    if (reference.kind == TensorRefKind::GmTensor) {
        if (reference.pointer.gm_tensor == nullptr) {
            return false;
        }
        required = ordinary_metadata_writers_exist &&
            !reference.pointer.gm_tensor->manual_dep &&
            previous_metadata_writer >= 0;
        return true;
    }
    if (reference.kind == TensorRefKind::LocalTensor) {
        if (reference.pointer.local_tensor == nullptr) {
            return false;
        }
        required = ordinary_metadata_writers_exist &&
            !reference.pointer.local_tensor->manual_dep &&
            previous_metadata_writer >= 0;
        return true;
    }
    return false;
}

struct SharedTaskWriterDelta {
    // ordinary entry 先在 owner 私有上下文中完整准备；只有拿到 task 的
    // exact insert turn 后才批量预检和发布。bucket/同桶序号及 symbol
    // packed key 都在等待前计算；串行区只消费这份不可变提交计划。
    // prepared_task_id 只在全部结构检查通过后写入；Publish 阶段据此确认
    // 这份 owner-local delta 属于当前 task，不再在串行区重复扫描 args。
    SharedRegionValue ordinary_entries[kMaxTaskTensors];
    uint32_t symbol_keys[kMaxTaskTensors];
    uint16_t ordinary_buckets[kMaxTaskTensors];
    uint8_t ordinary_bucket_ordinals[kMaxTaskTensors];
    int32_t prepared_task_id;
    uint32_t ordinary_count;
    uint32_t symbol_count;
    bool writer_intent_required;
    bool metadata_prefix_required;
};
static_assert(
    __is_trivially_constructible(SharedTaskWriterDelta),
    "shared task writer delta must remain trivial for CCEC local state"
);
static_assert(
    kMapBuckets <= 65536 && kMaxTaskTensors <= 256,
    "prepared bucket and ordinal metadata no longer cover the build"
);

PA_DEVICE bool PrepareSharedTaskWriterDelta(
    const TaskArgs &args, const SubmitContext &context,
    SharedTaskWriterDelta &delta,
    int32_t previous_metadata_writer = -1,
    bool ordinary_metadata_writers_exist = false,
    bool classify_metadata_prefix = false
) {
    delta.prepared_task_id = -1;
    delta.ordinary_count = 0;
    delta.symbol_count = 0;
    delta.writer_intent_required = false;
    delta.metadata_prefix_required = false;
    const int32_t task_id = context.task_id;
    if (!context.won || task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks) ||
        args.has_error || args.tensor_count < 0 ||
        args.tensor_count > static_cast<int32_t>(kMaxTaskTensors) ||
        context.result.task_id != static_cast<uint64_t>(task_id) ||
        context.result.count > kSharedOutputMaxPerTask ||
        (classify_metadata_prefix &&
         (previous_metadata_writer < -1 ||
          previous_metadata_writer >= task_id))) {
        return false;
    }

    // 在同一轮通用参数扫描中完成 prefix 分类、writer 引用校验、唯一性
    // 校验、register mask 核对和不可变 delta 构造。此前先调用
    // InspectSharedWriterIntent/ValidateSharedWriterIntentSet，非 writer 会扫描
    // 两遍参数，真实 writer 最多扫描四遍；这些扫描读取的是同一份 const
    // TaskArgs，不提供额外同步证明。合并后仍逐项保留原有 fail-closed 条件，
    // 且只依赖 tag/reference/task-id，不识别任何算子的 TaskKind 或 DAG。
    uint32_t expected_register_mask = 0;
    uint32_t register_mask = context.register_mask;
    for (int32_t index = 0; index < args.tensor_count; ++index) {
        const uint32_t bit =
            1U << static_cast<uint32_t>(index);
        const TensorArgType tag =
            TaskTag(args, static_cast<uint32_t>(index));
        if (classify_metadata_prefix) {
            bool arg_requires_prefix = false;
            if (!SharedTensorArgNeedsMetadataPrefix(
                    args.tensors[index], tag, task_id,
                    previous_metadata_writer,
                    ordinary_metadata_writers_exist,
                    arg_requires_prefix
                )) {
                return false;
            }
            delta.metadata_prefix_required |=
                arg_requires_prefix;
        }
        if (IsSharedWriterIntentTag(tag)) {
            expected_register_mask |= bit;
        }
        if ((register_mask & bit) == 0) {
            continue;
        }
        if (!IsSharedWriterIntentTag(tag)) {
            return false;
        }
        const TaskTensorRef &reference = args.tensors[index];
        if (reference.kind == TensorRefKind::SharedOutputRef) {
            const FdwicOutputRef output_ref =
                SharedOutputReference(reference);
            uint32_t symbol_key = 0;
            if (output_ref.producer_task_id < 0 ||
                output_ref.producer_task_id >= task_id ||
                !SharedSymbolHistoryKey(output_ref, symbol_key) ||
                delta.symbol_count >= kMaxTaskTensors) {
                return false;
            }
            for (uint32_t previous = 0;
                 previous < delta.symbol_count; ++previous) {
                if (delta.symbol_keys[previous] == symbol_key) {
                    return false;
                }
            }
            delta.symbol_keys[delta.symbol_count] = symbol_key;
            ++delta.symbol_count;
        } else if (reference.kind == TensorRefKind::GmTensor) {
            if (reference.pointer.gm_tensor == nullptr) {
                return false;
            }
            PA_GM const TensorDesc &tensor =
                *reference.pointer.gm_tensor;
            if (!tensor.manual_dep) {
                if (delta.ordinary_count >= kMaxTaskTensors ||
                    !MakeValidatedSharedWriterRegion(
                        tensor, task_id,
                        delta.ordinary_entries[
                            delta.ordinary_count
                        ]
                    ) ||
                    !ValidateOrdinarySharedWriterOwner(
                        tensor, task_id
                    )) {
                    return false;
                }
                const uint32_t bucket = TensorMapHash(
                    delta.ordinary_entries[
                        delta.ordinary_count
                    ].buffer_addr
                );
                uint32_t ordinal = 0;
                for (uint32_t previous = 0;
                     previous < delta.ordinary_count;
                     ++previous) {
                    ordinal +=
                        delta.ordinary_buckets[previous] == bucket
                            ? 1U
                            : 0U;
                }
                delta.ordinary_buckets[
                    delta.ordinary_count
                ] = static_cast<uint16_t>(bucket);
                delta.ordinary_bucket_ordinals[
                    delta.ordinary_count
                ] = static_cast<uint8_t>(ordinal);
                ++delta.ordinary_count;
            }
        } else if (reference.kind ==
                   TensorRefKind::LocalTensor) {
            if (reference.pointer.local_tensor == nullptr) {
                return false;
            }
            const TensorDesc &tensor =
                *reference.pointer.local_tensor;
            if (!tensor.manual_dep) {
                if (delta.ordinary_count >= kMaxTaskTensors ||
                    !MakeValidatedSharedWriterRegion(
                        tensor, task_id,
                        delta.ordinary_entries[
                            delta.ordinary_count
                        ]
                    ) ||
                    !ValidateOrdinarySharedWriterOwner(
                        tensor, task_id
                    )) {
                    return false;
                }
                const uint32_t bucket = TensorMapHash(
                    delta.ordinary_entries[
                        delta.ordinary_count
                    ].buffer_addr
                );
                uint32_t ordinal = 0;
                for (uint32_t previous = 0;
                     previous < delta.ordinary_count;
                     ++previous) {
                    ordinal +=
                        delta.ordinary_buckets[previous] == bucket
                            ? 1U
                            : 0U;
                }
                delta.ordinary_buckets[
                    delta.ordinary_count
                ] = static_cast<uint16_t>(bucket);
                delta.ordinary_bucket_ordinals[
                    delta.ordinary_count
                ] = static_cast<uint8_t>(ordinal);
                ++delta.ordinary_count;
            }
        } else {
            return false;
        }
        register_mask &= ~bit;
    }
    if (register_mask != 0 ||
        context.register_mask != expected_register_mask) {
        return false;
    }
    const bool writer_required =
        delta.ordinary_count != 0 || delta.symbol_count != 0;
    if (delta.ordinary_count + delta.symbol_count >
            kMaxTaskTensors) {
        return false;
    }
    delta.writer_intent_required = writer_required;
    if (classify_metadata_prefix && writer_required) {
        delta.metadata_prefix_required = true;
    }
    delta.prepared_task_id = task_id;
    return true;
}

// 正式 PA 在进入全局有序 Register 之前校验 owner-local writer delta。
// 该对象此后只按 const 引用传递，predecessor wait 也不会修改本核栈，
// 因而 Register 可以复用这份证明，不必在串行区重复解码三个 key。
PA_DEVICE bool ValidatePreparedPaWriterShape(
    const SharedTaskWriterDelta &delta, TaskKind kind,
    int32_t task_id, int32_t expected_previous,
    int32_t expected_producer
) {
    if (task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks) ||
        delta.prepared_task_id != task_id ||
        delta.ordinary_count != 0) {
        return false;
    }
    if (kind != TaskKind::Up) {
        return expected_previous == -1 &&
            delta.symbol_count == 0 &&
            !delta.writer_intent_required;
    }
    if (expected_producer < 0 ||
        expected_producer >= task_id ||
        expected_producer >=
            static_cast<int32_t>(kMaxTasks) ||
        expected_previous < expected_producer ||
        expected_previous >= task_id ||
        delta.symbol_count != 3 ||
        !delta.writer_intent_required) {
        return false;
    }
    const uint32_t key_base =
        static_cast<uint32_t>(expected_producer) *
            kSharedOutputMaxPerTask +
        1U;
    return delta.symbol_keys[0] == key_base + 2U &&
        delta.symbol_keys[1] == key_base + 1U &&
        delta.symbol_keys[2] == key_base;
}

template <
    typename Ops, bool CheckFatal = true,
    bool CheckOutputPublished = true,
    bool UseExpectedPrevious = false,
    bool UsePaUpShape = false,
    bool TrustPreparedPaShape = false,
    bool UseDynamicDag = false
>
PA_DEVICE bool PublishSharedTaskWriterMetadata(
    PA_GM SchedulerState *state, const SubmitContext &context,
    const SharedTaskWriterDelta &delta, LocalStats &stats,
    int32_t expected_previous = -1,
    int32_t expected_producer = -1
#if !PA_BUILD_TRACE_FREE
    , DeferredSharedWriterMetadataTrace *deferred_trace = nullptr
#endif
    , const SharedMetadataDag *metadata_dag = nullptr
) {
    static_assert(
        !UsePaUpShape || UseExpectedPrevious,
        "PA UP metadata requires an expected previous writer"
    );
    static_assert(
        !TrustPreparedPaShape || UsePaUpShape,
        "only the PA path can trust a prepared writer shape"
    );
    static_assert(
        !UseDynamicDag ||
            (!UseExpectedPrevious && !UsePaUpShape &&
             !TrustPreparedPaShape),
        "dynamic DAG cannot be mixed with PA writer shortcuts"
    );
    const int32_t task_id = context.task_id;
    bool fatal_clear = true;
    if constexpr (CheckFatal) {
        fatal_clear =
            state != nullptr &&
            TraceAtomicLoad<Ops>(
                stats.trace, stats.result, task_id,
                AtomicSite::SharedMetadataFatalGuardLoad,
                &state->fatal.value
            ) == 0;
    }
    if (state == nullptr || !context.won || task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks) ||
        delta.prepared_task_id != task_id ||
        delta.ordinary_count > kMaxTaskTensors ||
        delta.symbol_count > kMaxTaskTensors ||
        delta.ordinary_count + delta.symbol_count >
            kMaxTaskTensors ||
        delta.writer_intent_required !=
            (delta.ordinary_count != 0 ||
             delta.symbol_count != 0) ||
        !fatal_clear) {
        if (state != nullptr) {
            SetFatal<Ops>(state, stats, task_id);
        }
        return false;
    }

    if constexpr (UseDynamicDag) {
        bool dag_valid = metadata_dag != nullptr &&
            metadata_dag->prepared_task_id == task_id &&
            metadata_dag->writer_count == delta.symbol_count &&
            metadata_dag->ordinary_writer ==
                (delta.ordinary_count != 0);
        for (uint32_t index = 0;
             dag_valid && index < delta.symbol_count; ++index) {
            dag_valid =
                metadata_dag->writer_symbol_keys[index] ==
                    delta.symbol_keys[index] &&
                metadata_dag->writer_previous[index] >= 0 &&
                metadata_dag->writer_previous[index] < task_id;
        }
        if (!dag_valid) {
            SetFatal<Ops>(state, stats, task_id);
            return false;
        }
    }

    if constexpr (UsePaUpShape) {
        // standalone PA 的 ordinary writer 集合恒为空；非 UP task 也没有
        // symbol。expected_previous 由 PA 计划在 Materialize 前推导，
        // 它非负即表示当前 task 是 UP；据此把“合法非 UP 空集合”和
        // “异常 UP 丢失三个 symbol”严格区分。错误形状不回退通用路径。
        if constexpr (!TrustPreparedPaShape) {
            const bool expects_pa_up = expected_previous >= 0;
            if (delta.ordinary_count != 0 ||
                delta.symbol_count !=
                    (expects_pa_up ? 3U : 0U)) {
                SetFatal<Ops>(state, stats, task_id);
                return false;
            }
        }
        if (delta.symbol_count == 0) {
            return true;
        }
    }

    // insert-before-lookup 不能用本 task 的 reader_done 回收自己仍可能
    // 消费的 N-H。首版只使用已经证明正确的 -1 前沿；容量不足明确
    // 终止，绝不覆盖 live producer 或错误推进 task turn。
    if constexpr (!UsePaUpShape) {
        if (SharedCheckPreparedTaskAppend<Ops, true>(
                state->shared_map, delta.ordinary_entries,
                delta.ordinary_buckets,
                delta.ordinary_bucket_ordinals,
                delta.ordinary_count, -1, task_id,
                &stats.trace, &stats.result
            ) != SharedAppendCheck::Ready) {
            SetFatal<Ops>(state, stats, task_id);
            return false;
        }
    }

    if (delta.symbol_count != 0 &&
        !CommitPreparedSymbolSharedWriterIntentSet<
            Ops, true, CheckOutputPublished,
            UseExpectedPrevious, UsePaUpShape,
            TrustPreparedPaShape, UseDynamicDag
        >(
            state->shared_map, delta.symbol_keys,
            delta.symbol_count, task_id, &state->fatal.value,
            &stats, expected_previous, expected_producer
#if !PA_BUILD_TRACE_FREE
            , deferred_trace
#endif
            , UseDynamicDag
                ? metadata_dag->writer_previous
                : nullptr
        )) {
        SetFatal<Ops>(state, stats, task_id);
        return false;
    }
    if constexpr (!UsePaUpShape) {
        if (!SharedAppendPreparedTask<Ops, true>(
                state->shared_map, delta.ordinary_entries,
                delta.ordinary_buckets, delta.ordinary_count,
                task_id, &stats.trace, &stats.result
            )) {
            SetFatal<Ops>(state, stats, task_id);
            return false;
        }
    }
    return true;
}

PA_DEVICE void RecordCommittedSharedTaskWriterStats(
    const SharedTaskWriterDelta &delta, LocalStats &stats
) {
    // 这两个字段只在 task-level completion FetchAdd 已发射后更新。
    // 发布端不等待返回值；严格完成与顺序由 N+1 的返回型 Load、最终 task
    // 的 host 精确终态共同闭合。
    stats.result.map_inserts += delta.ordinary_count;
    stats.result.shared_symbol_inout_commits += delta.symbol_count;
}

template <typename Ops>
PA_DEVICE bool HandoffSharedTaskInsertTurn(
    PA_GM SchedulerState *state, int32_t task_id, LocalStats &stats
) {
    if (state == nullptr || task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks)) {
        return false;
    }
    // ordinary payload 的 DCCI、symbol history/latest 与 fresh descriptor
    // 都必须先于本 task 的插入完成字对 N+1 owner 可见。每个 task 使用
    // 自己的 128B-isolated sidecar completion，不再与相邻 TaskCell atomic
    // 共用冲突单元，也不把一枚 baton 在 G 条线上轮换。
    Ops::StoreBarrier();
    // swimlane 构建走 CaptureAtomicFetchAddIssue；这里是无泳道构建和
    // 隔离测试共用的 trace-free helper。
    // PA_ATOMIC_DCCI_SOURCE_EXEMPT: trace-free - swimlane 使用统一捕获 helper
    (void)Ops::FetchAdd(
        &state->claim_tournament[static_cast<uint32_t>(task_id)]
             .root.insert_completion.value,
        static_cast<int64_t>(1)
    );
    (void)stats;
    return true;
}

#if !PA_BUILD_TRACE_FREE
template <typename Ops>
PA_DEVICE bool TraceHandoffSharedTaskInsertTurn(
    PA_GM SchedulerState *state, int32_t task_id, LocalStats &stats,
    uint64_t &publish_trace_begin, uint64_t &publish_trace_end
) {
    publish_trace_begin = 0;
    publish_trace_end = 0;
    if (state == nullptr || task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks)) {
        return false;
    }
    Ops::StoreBarrier();
    CaptureAtomicFetchAddIssue<Ops>(
        stats.trace,
        &state->claim_tournament[static_cast<uint32_t>(task_id)]
             .root.insert_completion.value,
        static_cast<int64_t>(1),
        publish_trace_begin, publish_trace_end
    );
    return true;
}
#endif

template <typename Ops>
PA_DEVICE bool WaitForSharedTaskInsertTurn(
    PA_GM SchedulerState *state, int32_t task_id,
    LocalStats &stats, int64_t &ready_observed
);

template <typename Ops>
PA_DEVICE bool PublishSharedTaskWriterDelta(
    PA_GM SchedulerState *state, const SubmitContext &context,
    const SharedTaskWriterDelta &delta, LocalStats &stats
) {
    // fresh output cell 由本 task 的唯一 Claim winner 独占，不参与
    // ordinary/symbol 的 task-ID 串行插入。先发布 descriptor，再等待
    // predecessor；最终 per-task insert-completion handoff 仍同时封口
    // 两类发布。
    if (state == nullptr || !context.won || context.task_id < 0 ||
        context.task_id >= static_cast<int32_t>(kMaxTasks) ||
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: test-only - generic 组合入口只供隔离测试；正式 dispatched 路径在领取新 Build ticket 前统一观察 scheduler fatal，已取得的合法工作单元允许闭合
        Ops::Load(&state->fatal.value) != 0 ||
        !PublishSharedTaskOutputs<Ops>(
            state->shared_map, context,
            static_cast<uint32_t>(context.task_id)
        )) {
        if (state != nullptr) {
            SetFatal<Ops>(state, stats, context.task_id);
        }
        return false;
    }
    // 隔离测试和复用调用必须走与正式 Submit 相同的资格门：task 0
    // 直接进入，其余 task 只等待 N-1。Metadata helper 本身不再重复
    // atomic load，避免正式热路径把一次前驱等待测成三次原子访问。
    int64_t ignored_ready_observed = -1;
    if (!WaitForSharedTaskInsertTurn<Ops>(
            state, context.task_id, stats,
            ignored_ready_observed
        )) {
        RollbackSharedTaskOutputs<Ops>(
            state->shared_map.shared_outputs[
                static_cast<uint32_t>(context.task_id)
            ],
            context.result.count, context.task_id, &stats
        );
        return false;
    }
    if (!PublishSharedTaskWriterMetadata<Ops>(
            state, context, delta, stats
        )) {
        RollbackSharedTaskOutputs<Ops>(
            state->shared_map.shared_outputs[
                static_cast<uint32_t>(context.task_id)
            ],
            context.result.count, context.task_id, &stats
        );
        return false;
    }
    const bool inserted = HandoffSharedTaskInsertTurn<Ops>(
        state, context.task_id, stats
    );
    if (!inserted) {
        RollbackSharedTaskOutputs<Ops>(
            state->shared_map.shared_outputs[
                static_cast<uint32_t>(context.task_id)
            ],
            context.result.count, context.task_id, &stats
        );
        return false;
    }
    RecordCommittedSharedTaskWriterStats(delta, stats);
    return true;
}

template <typename Ops>
PA_DEVICE bool WaitForSharedMetadataPredecessor(
    PA_GM SchedulerState *state, int32_t task_id,
    int32_t predecessor_task_id, LocalStats &stats,
    int64_t &ready_observed, uint64_t &load_count
) {
    ready_observed = -1;
    load_count = 0;
    if (state == nullptr || task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks)) {
        return false;
    }

    // -1 表示本 task 之前没有 metadata writer，无需为空
    // task 人为制造 baton。非负前驱必须严格早于当前 task。
    if (predecessor_task_id == -1) {
        ready_observed = -1;
        return true;
    }
    if (predecessor_task_id < 0 ||
        predecessor_task_id >= task_id) {
        return false;
    }

    PA_GM volatile int64_t *predecessor =
        &state->claim_tournament[
             static_cast<uint32_t>(predecessor_task_id)
         ].root.insert_completion.value;
    const uint64_t begin = Ops::Now();
    uint32_t polls = 0;
    while (true) {
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: aggregate - 循环结束后以 SharedInsertTurnPoll 单条 PollBatch 记录精确 load_count
        const int64_t observed = Ops::Load(predecessor);
        int64_t compare_observed = observed;
        int64_t dependency_observed = observed;
#if PA_BUILD_SWIMLANE && \
    (defined(PA_BUILD_AIC) || defined(PA_BUILD_AIV))
        // 分支判定和 SYS_CNT 依赖边界从同一个 atomic 返回寄存器派生，
        // 防止 CCEC 在 ready 分支把 ready_observed 常量折叠掉。
        compare_observed = Ops::ForkAtomicResultForBranch(
            observed, dependency_observed
        );
#endif
        if (compare_observed ==
            static_cast<int64_t>(predecessor_task_id)) {
            ready_observed = dependency_observed;
            load_count = static_cast<uint64_t>(polls) + 1;
            return true;
        }
        const int64_t pending =
            SharedInsertCompletionInitialValue(
                static_cast<uint32_t>(predecessor_task_id)
            );
        if (compare_observed != pending) {
            SetFatal<Ops>(state, stats, task_id);
            return false;
        }

        Ops::SpinHint();
        ++polls;
        if ((polls & 1023U) != 0) {
            continue;
        }
        if (IsFatal<Ops>(state, stats, task_id)) {
            return false;
        }
        if (Ops::Now() - begin > kSharedInsertWatchdogTicks) {
            SetFatal<Ops>(state, stats, task_id);
            return false;
        }
    }
}

template <typename Ops>
PA_DEVICE bool WaitForSharedTaskInsertTurn(
    PA_GM SchedulerState *state, int32_t task_id, LocalStats &stats,
    int64_t &ready_observed, uint64_t &load_count
) {
    return WaitForSharedMetadataPredecessor<Ops>(
        state, task_id, task_id == 0 ? -1 : task_id - 1,
        stats, ready_observed, load_count
    );
}

template <typename Ops>
PA_DEVICE bool WaitForSharedTaskInsertTurn(
    PA_GM SchedulerState *state, int32_t task_id, LocalStats &stats,
    int64_t &ready_observed
) {
    uint64_t ignored_load_count = 0;
    return WaitForSharedTaskInsertTurn<Ops>(
        state, task_id, stats, ready_observed,
        ignored_load_count
    );
}

template <typename Ops>
PA_DEVICE bool WaitForSharedTaskInsertTurn(
    PA_GM SchedulerState *state, int32_t task_id, LocalStats &stats
) {
    int64_t ignored_ready_observed = -1;
    return WaitForSharedTaskInsertTurn<Ops>(
        state, task_id, stats, ignored_ready_observed
    );
}

template <typename Ops>
PA_DEVICE bool WaitForSharedMetadataDagDependencies(
    PA_GM SchedulerState *state, int32_t task_id,
    const SharedMetadataDag &dag, LocalStats &stats,
    int64_t &ready_observed, uint64_t &load_count
) {
    ready_observed = -1;
    load_count = 0;
    if (state == nullptr || task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks) ||
        dag.prepared_task_id != task_id ||
        dag.dependency_count > kMaxTaskTensors) {
        return false;
    }
    for (uint32_t index = 0;
         index < dag.dependency_count; ++index) {
        const int32_t predecessor = dag.dependencies[index];
        int64_t dependency_observed = -1;
        uint64_t dependency_loads = 0;
        if (predecessor < 0 || predecessor >= task_id ||
            !WaitForSharedMetadataPredecessor<Ops>(
                state, task_id, predecessor, stats,
                dependency_observed, dependency_loads
            ) ||
            UINT64_MAX - load_count < dependency_loads) {
            return false;
        }
        load_count += dependency_loads;
        ready_observed = dependency_observed;
    }
    return true;
}

PA_DEVICE bool DecodeSharedMetadataWriterPlan(
    PA_GM const SharedBuildDispatchState &dispatch,
    uint32_t task_id, bool &publishes_metadata,
    int32_t &previous_metadata_writer
) {
    publishes_metadata = false;
    previous_metadata_writer = -1;
    if (dispatch.task_count == 0 ||
        dispatch.task_count > kMaxTasks ||
        task_id >= dispatch.task_count) {
        return false;
    }
    const uint32_t word_index = task_id / 64U;
    const uint32_t bit_index = task_id % 64U;
    const uint64_t word =
        dispatch.metadata_writer_bits[word_index];
    publishes_metadata =
        ((word >> bit_index) & static_cast<uint64_t>(1)) != 0;

    // 当前 word 只保留 task_id 之前的 writer。不依赖平台
    // CLZ intrinsic；最多 64 次寄存器位移，跨 word 才再读
    // 一个只读 GM word。
    uint64_t candidates = bit_index == 0
        ? static_cast<uint64_t>(0)
        : word &
              ((static_cast<uint64_t>(1) << bit_index) - 1U);
    int32_t scan_word = static_cast<int32_t>(word_index);
    while (true) {
        if (candidates != 0) {
            uint32_t highest = 0;
            while ((candidates >>= 1U) != 0) {
                ++highest;
            }
            previous_metadata_writer =
                scan_word * 64 + static_cast<int32_t>(highest);
            return previous_metadata_writer <
                static_cast<int32_t>(task_id);
        }
        --scan_word;
        if (scan_word < 0) {
            return true;
        }
        candidates = dispatch.metadata_writer_bits[
            static_cast<uint32_t>(scan_word)
        ];
    }
}

PA_DEVICE bool ValidateSharedMetadataWriterSummary(
    PA_GM const SharedBuildDispatchState &dispatch
) {
    if (dispatch.task_count == 0 ||
        dispatch.task_count > kMaxTasks ||
        dispatch.metadata_writer_count > dispatch.task_count ||
        dispatch.ordinary_metadata_writer_count >
            dispatch.task_count ||
        dispatch.symbol_metadata_writer_count >
            dispatch.task_count) {
        return false;
    }
    // 一个 task 可以同时发布 ordinary 与 symbol metadata，所以 union
    // writer 数必须落在 max(class_count) 与两类之和之间。
    const uint32_t class_max =
        dispatch.ordinary_metadata_writer_count >
                dispatch.symbol_metadata_writer_count
            ? dispatch.ordinary_metadata_writer_count
            : dispatch.symbol_metadata_writer_count;
    return dispatch.metadata_writer_count >= class_max &&
        dispatch.metadata_writer_count <=
            dispatch.ordinary_metadata_writer_count +
                dispatch.symbol_metadata_writer_count;
}

// 判断 task N 的 Fanin 查询是否必须等待全部 metadata writer 前缀。
//
// 严格顺序只约束真正发布 metadata 的 task；非 writer 不应仅仅因为计划中
// 存在更早 writer 就占住 Build worker 等待。对 SharedOutputRef，descriptor
// 由 producer 的独占 published 字直接证明；只有 (producer, N) 之间存在
// metadata writer 时，N 才必须等到 previous_metadata_writer 完成，确保
// latest/history 已包含所有 writer<N。由于 previous_metadata_writer 已是
// max(writer<N)，条件可等价写成 previous_metadata_writer>producer，无需
// 再扫描一次 GM bitset。
//
// ordinary TensorMap lookup 没有 per-key completion；只要全计划可能存在
// ordinary writer 且已有更早 metadata writer，就保守等待整个前缀。
// 全计划 ordinary writer 数为 0 时，symbol-only writer 不可能改变 ordinary
// lookup；这是算子无关的零写者快路径。manual_dep 不查询共享 TensorMap。
// 当前 task 自己是 writer 时始终要求该前缀，从而不放宽 writer-to-writer
// 的严格 task-id 顺序。该判定只读取算子通用 TaskArgs，不依赖 PA kind。
PA_DEVICE bool SharedTaskNeedsMetadataPrefix(
    const TaskArgs &args, int32_t task_id,
    bool publishes_metadata, int32_t previous_metadata_writer,
    bool ordinary_metadata_writers_exist,
    bool &required
) {
    required = false;
    if (task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks) ||
        previous_metadata_writer < -1 ||
        previous_metadata_writer >= task_id ||
        args.has_error || args.tensor_count < 0 ||
        args.tensor_count > static_cast<int32_t>(kMaxTaskTensors)) {
        return false;
    }
    if (publishes_metadata) {
        required = true;
        return true;
    }

    for (int32_t index = 0; index < args.tensor_count; ++index) {
        const TensorArgType tag =
            TaskTag(args, static_cast<uint32_t>(index));
        bool arg_requires_prefix = false;
        if (!SharedTensorArgNeedsMetadataPrefix(
                args.tensors[index], tag, task_id,
                previous_metadata_writer,
                ordinary_metadata_writers_exist,
                arg_requires_prefix
            )) {
            return false;
        }
        if (arg_requires_prefix) {
            required = true;
            return true;
        }
    }
    return true;
}

template <typename Ops>
PA_DEVICE bool WaitForPreparedSharedWriterOutputs(
    PA_GM SchedulerState *state,
    const SharedTaskWriterDelta &writer_delta,
    int32_t task_id, LocalStats &stats
) {
    if (state == nullptr || task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks) ||
        writer_delta.symbol_count >
            kSharedWriterHistoryMaxPerTask) {
        return false;
    }
    PA_LOOP_NOUNROLL
    for (uint32_t index = 0;
         index < writer_delta.symbol_count; ++index) {
        const FdwicOutputRef output_ref =
            SharedSymbolHistoryReference(
                writer_delta.symbol_keys[index]
            );
        if (!IsPlainSharedOutputRef(output_ref) ||
            output_ref.producer_task_id < 0 ||
            output_ref.producer_task_id >= task_id ||
            !WaitForSharedOutputPublished<Ops, true>(
                state->shared_map, output_ref,
                &state->fatal.value, &stats, task_id,
                AtomicSite::SharedMetadataOutputPublishedLoad
            )) {
            return false;
        }
    }
    return true;
}

template <typename Ops, bool Profile, bool Dispatched,
          typename PmuContext>
PA_DEVICE bool FinishSharedWinnerSubmitBody(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker,
    const TaskArgs &args, SubmitContext &context, LocalStats &stats,
    PmuContext &pmu_context, const CallbackSubmitTicket &ticket
) {
    const uint32_t task_id = ticket.task_id;
    SharedPaTaskMeta task_meta{};
    if (ticket.won == 0 ||
        !DecodeSharedPaTaskMeta(ticket.reserved, task_id, task_meta) ||
        !SharedPaFunctionIdMatches(
            task_meta.kind, true,
            static_cast<int32_t>(ticket.function_id)
        ) ||
        context.task_id != static_cast<int32_t>(task_id) ||
        !context.won) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }

    const TaskKind kind = task_meta.kind;
    const int32_t function_id =
        static_cast<int32_t>(ticket.function_id);
    const SharedPaDagSchema dag_schema{
        &state->build_dispatch
    };
    SharedMetadataDag metadata_dag{};
    if (!ValidateSharedDagTaskArgs(
            dag_schema, task_id, args
        ) ||
        !BuildSharedMetadataDag(
            dag_schema, task_id, metadata_dag
        )) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }

    // Claim owner 先构造 descriptor 和 writer delta；这一段不查询
    // TensorMap，也不占用有序插入通道。
    const uint64_t materialize_begin =
        TraceTimestamp<Ops>(stats.trace, stats.result);
    BeginSubmitPmuPhase<SubmitPmuPhase::Materialize, Ops>(
        pmu_context
    );
    // output_count 已由 callback 的 task-kind 形状闭合。先用原有返回型
    // FetchMax 预留 task-indexed writer 控制字，再允许 Materialize 覆盖
    // 最终 descriptor cell；这样非法旧轮状态仍在任何 payload 写入前失败。
    const uint32_t expected_output_count =
        context.shared_result.Size();
    const bool output_writers_reserved =
        ReserveSharedTaskOutputWriters<Ops, true>(
            state->shared_map, task_id, expected_output_count,
            &stats
        );
    const bool materialized =
        output_writers_reserved &&
        MaterializeTask<Ops, true, true>(
            worker, task_id, args, context, state->shared_map,
            state->heap_base, state->heap_size,
            kind, task_meta.batch_start, task_meta.group_index,
            &stats.trace, &stats.result
        );
    if (materialized) {
        stats.result.materialized_outputs += context.result.count;
    }
    if (!materialized) {
        if (output_writers_reserved) {
            RollbackSharedTaskOutputs<Ops>(
                state->shared_map.shared_outputs[task_id],
                expected_output_count,
                static_cast<int32_t>(task_id), &stats
            );
        }
        EndSubmitPmuPhase<SubmitPmuPhase::Materialize, Ops>(
            pmu_context
        );
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    SharedTaskWriterDelta writer_delta{};
    bool writer_delta_matches_dag =
        PrepareSharedTaskWriterDelta(
            args, context, writer_delta
        ) &&
        writer_delta.symbol_count == metadata_dag.writer_count &&
        (writer_delta.ordinary_count != 0) ==
            metadata_dag.ordinary_writer;
    for (uint32_t index = 0;
         writer_delta_matches_dag &&
         index < writer_delta.symbol_count; ++index) {
        writer_delta_matches_dag =
            writer_delta.symbol_keys[index] ==
                metadata_dag.writer_symbol_keys[index];
    }
    if (!writer_delta_matches_dag) {
        RollbackSharedTaskOutputs<Ops>(
            state->shared_map.shared_outputs[task_id],
            expected_output_count,
            static_cast<int32_t>(task_id), &stats
        );
        EndSubmitPmuPhase<SubmitPmuPhase::Materialize, Ops>(
            pmu_context
        );
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    const bool publishes_metadata =
        writer_delta.writer_intent_required;
#if PA_BUILD_TRACE_FREE
    const bool task_outputs_published =
        PublishSharedTaskOutputs<Ops, true, true, true>(
            state->shared_map, context, task_id, &stats
        );
#else
    uint64_t task_outputs_begin =
        TraceTimestamp<Ops>(stats.trace, stats.result);
    uint64_t task_outputs_copy_begin = task_outputs_begin;
    uint64_t task_outputs_copy_end = task_outputs_begin;
    uint64_t task_outputs_flush_begin = task_outputs_begin;
    uint64_t task_outputs_flush_end = task_outputs_begin;
    const bool task_outputs_published =
        PublishSharedTaskOutputs<Ops, true, true, true>(
            state->shared_map, context, task_id, &stats,
            &task_outputs_copy_begin, &task_outputs_copy_end,
            &task_outputs_flush_begin, &task_outputs_flush_end
        );
    const uint64_t task_outputs_end =
        TraceTimestamp<Ops>(stats.trace, stats.result);
#endif
    if (!task_outputs_published) {
        // published Exchange 的异常由 helper 原地回滚；若失败来自内部
        // descriptor/result 合同校验，则整轮立即进入 terminal fatal，残留
        // writer 控制字只作为错误证据保留，后续不会继续调度或复用本 cell。
        // 不在这个热函数复制一遍完整 rollback 冷路径，避免只为不可恢复
        // 错误放大 AIC/AIV 的正常指令工作集。
        EndSubmitPmuPhase<SubmitPmuPhase::Materialize, Ops>(
            pmu_context
        );
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    // 稀疏 writer 链只证明 metadata writer 之间的顺序，
    // 不再间接证明任意更早 task 的 fresh output 已发布。
    // symbol writer 在进入严格串行区前直接等待自己实际要
    // 改写的 output 目标；这样既不会抢占 metadata 顺序链，
    // 也不会让首个 writer 在 producer Materialize 前提交 CAS。
    if (!WaitForPreparedSharedWriterOutputs<Ops>(
            state, writer_delta,
            static_cast<int32_t>(task_id), stats
        )) {
        EndSubmitPmuPhase<SubmitPmuPhase::Materialize, Ops>(
            pmu_context
        );
        SetFatal<Ops>(
            state, stats, static_cast<int32_t>(task_id)
        );
        return false;
    }
#if !PA_BUILD_TRACE_FREE
    // 只初始化实际会写入的计数；端点在对应 DCCI/CAS 执行时覆盖，
    // 避免为空 writer 清零完整本地对象。
    DeferredSharedWriterMetadataTrace writer_metadata_trace;
    writer_metadata_trace.history_dcci_lines = 0;
    writer_metadata_trace.writer_cas_count = 0;
#endif
    EndSubmitPmuPhase<SubmitPmuPhase::Materialize, Ops>(
        pmu_context
    );
    const uint64_t materialize_end =
        TraceTimestamp<Ops>(stats.trace, stats.result);

    // 只等待当前 task 的精确 DAG 前驱：同 symbol writer 和 ordinary
    // 保守链分别推进，异 symbol 不再进入一条全局 writer baton。
    const uint64_t register_begin = materialize_end;
    BeginSubmitPmuPhase<SubmitPmuPhase::Register, Ops>(
        pmu_context
    );
    int64_t ready_observed = -1;
    uint64_t insert_turn_load_count = 0;
    const bool turn_ready =
        WaitForSharedMetadataDagDependencies<Ops>(
        state, static_cast<int32_t>(task_id),
        metadata_dag, stats,
        ready_observed, insert_turn_load_count
    );
    // wait_end 对最后一次返回 Ready 的 atomic Load 建立数据依赖。只在
    // swimlane 构建读取 SYS_CNT；trace-free 的 PMU/性能构建预处理后为 0。
    const uint64_t metadata_begin =
        turn_ready && insert_turn_load_count != 0
        ? TraceTimestampAfterAtomicResult<Ops>(
              stats.trace, stats.result, ready_observed
          )
        : TraceTimestamp<Ops>(stats.trace, stats.result);
    const bool metadata_published =
        turn_ready &&
        PublishSharedTaskWriterMetadata<
            Ops, true, false, false, false, false, true
        >(
            state, context, writer_delta, stats, -1, -1
#if !PA_BUILD_TRACE_FREE
            , &writer_metadata_trace
#endif
            , &metadata_dag
        );
    if (turn_ready && !metadata_published) {
        // 多 symbol CAS 不是事务；任何冲突都保留故障现场并终止整轮，
        // 且不得发布本 task 的 insert completion。
        SetFatal<Ops>(
            state, stats, static_cast<int32_t>(task_id)
        );
    }
    const uint64_t metadata_end = metadata_published
        ? TraceTimestamp<Ops>(stats.trace, stats.result)
        : metadata_begin;

#if PA_BUILD_TRACE_FREE
    const bool inserted =
        metadata_published &&
        (!publishes_metadata ||
         HandoffSharedTaskInsertTurn<Ops>(
             state, static_cast<int32_t>(task_id), stats
         ));
#else
    uint64_t publish_trace_begin = 0;
    uint64_t publish_trace_end = 0;
    const bool inserted =
        metadata_published &&
        (!publishes_metadata ||
         TraceHandoffSharedTaskInsertTurn<Ops>(
             state, static_cast<int32_t>(task_id), stats,
             publish_trace_begin, publish_trace_end
         ));
#endif
    // writer 发布端只记录 FetchAdd 的 source-issue 边界，不等待
    // 返回值；真正完成由下一个 metadata consumer/writer 的返回型
    // predecessor Load 建立。空 writer 不触碰自己的 completion word。
    const uint64_t register_end = metadata_published
        ? TraceTimestamp<Ops>(stats.trace, stats.result)
        : metadata_end;
    EndSubmitPmuPhase<SubmitPmuPhase::Register, Ops>(
        pmu_context
    );
#if !PA_BUILD_TRACE_FREE
    // task[N] 的 completion FetchAdd 已经发射；下一 owner 的返回型 Load
    // 会在新值真正可见后推进。现在才落 history DCCI 和 writer CAS 的
    // raw，既保留端点与事件数量，也不再让 16B 记录写入延长全局有序发布链。
    if (writer_metadata_trace.history_dcci_lines != 0) {
        (void)WriteDcciTrace(
            stats.trace, static_cast<int32_t>(task_id), -1,
            DcciSite::SharedWriterHistoryFlush,
            DcciOp::CleanOut, /*trailing_dsb=*/true,
            writer_metadata_trace.history_dcci_lines,
            writer_metadata_trace.history_dcci_begin,
            writer_metadata_trace.history_dcci_end
        );
    }
    if (AtomicSwimlaneEnabled(stats.trace)) {
        PA_LOOP_NOUNROLL
        for (uint32_t index = 0;
             index < writer_metadata_trace.writer_cas_count;
             ++index) {
            WriteAtomicTrace<Ops>(
                stats.trace, stats.result,
                static_cast<int32_t>(task_id),
                AtomicSite::SharedMetadataLastWriterCommit,
                AtomicOp::CompareExchange,
                writer_metadata_trace.writer_cas_begin[index],
                writer_metadata_trace.writer_cas_end[index],
                true, Ops::kAtomicReturnReadyObserved
            );
        }
    }
#endif
    // 所有端点完成后再按业务顺序写 raw，避免写 trace 本身落入
    // Materialize/Register 的测量区间。Materialize 的 output detail
    // 还原独占 cell 发布；Register 只闭合 wait、writer metadata 与
    // handoff，不逐 poll 扩张记录。
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id),
        function_id, TracePhase::Materialize,
        ProfilePhase::Materialize, materialize_begin,
        materialize_end, 0,
        kind == TaskKind::Alloc ? 1U : 0U
    );
#if !PA_BUILD_TRACE_FREE
    if (task_outputs_published) {
        WriteTrace<false>(
            stats.trace, stats.result,
            static_cast<int32_t>(task_id), function_id,
            TracePhase::SharedMaterializePublishTaskOutputs,
            ProfilePhase::Materialize, task_outputs_begin,
            task_outputs_end
        );
        WriteTrace<false>(
            stats.trace, stats.result,
            static_cast<int32_t>(task_id), function_id,
            TracePhase::SharedMaterializePublishTaskOutputsCopy,
            ProfilePhase::Materialize, task_outputs_copy_begin,
            task_outputs_copy_end
        );
        WriteTrace<false>(
            stats.trace, stats.result,
            static_cast<int32_t>(task_id), function_id,
            TracePhase::SharedMaterializePublishTaskOutputsFlush,
            ProfilePhase::Materialize, task_outputs_flush_begin,
            task_outputs_flush_end
        );
    }
#endif
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id),
        function_id, TracePhase::Register,
        ProfilePhase::Register, register_begin, register_end,
        0, writer_delta.ordinary_count
    );
    WriteTrace<false>(
        stats.trace, stats.result, static_cast<int32_t>(task_id),
        function_id, TracePhase::SharedRegisterPublishMetadata,
        ProfilePhase::Register, metadata_begin, metadata_end
    );
#if !PA_BUILD_TRACE_FREE
    // 记录动作延后到 Register 的全部时间端点之后，避免 32B raw 写入
    // 被误计入 metadata publish 或 handoff。循环内只累计本地 polls，
    // 每个成功 winner 固定至多增加这一条物理 PollBatch。
    if (turn_ready && insert_turn_load_count != 0) {
        (void)WriteAggregateAtomicPollBatch(
            stats.trace, stats.result,
            AtomicSite::SharedInsertTurnPoll,
            register_begin, metadata_begin,
            insert_turn_load_count,
            Ops::kAtomicReturnReadyObserved
        );
    }
    if (metadata_published && publishes_metadata &&
        AtomicSwimlaneEnabled(stats.trace)) {
        WriteAtomicTrace<Ops>(
            stats.trace, stats.result,
            static_cast<int32_t>(task_id),
            AtomicSite::SharedInsertTurnHandoff,
            AtomicOp::FetchAdd,
            publish_trace_begin, publish_trace_end,
            false, false
        );
    }
#endif
    if (!inserted) {
        // output cell 虽已在串行等待前短暂可见，但完成字尚未发布；失败路径
        // 恢复本 task 独占 cell，保留原有 fail-closed 终态。正常路径无额外
        // rollback 分支开销。
        RollbackSharedTaskOutputs<Ops>(
            state->shared_map.shared_outputs[task_id],
            context.result.count,
            static_cast<int32_t>(task_id), &stats
        );
        return false;
    }
    // Register 的业务终点已经取完，失败分支也已经退出；成功统计因此
    // 既不污染串行区泳道，也不会把发布前失败的 metadata 前缀误算成
    // 已提交事务。
    RecordCommittedSharedTaskWriterStats(
        writer_delta, stats
    );
#if defined(PA_TEST_SHARED_SUBMIT_HOOKS)
    // 仅供 CPU 定向并发门槛暂停某个 owner；正式 CPU/CCEC 构建预处理后
    // 不保留调用。测试借此证明 N+1 的 lookup/Build 不被 N 的 Build
    // 阶段串行化。
    Ops::AfterSharedTaskInsert(
        state, worker, task_id
    );
#endif

    // task[N] 的插入完成字已经发布，N+1 owner 可以进入有序插入段；
    // 当前 owner 的 fanin lookup、Build 和 slot 执行不再占住该链。
    uint64_t build_begin = register_end;
    if (kind != TaskKind::Alloc) {
        const uint64_t fanin_begin = register_end;
        bool lookup_protocol_ok = false;
        uint32_t ordinary_lookup_count = 0;
        // 本 task 的 writer 已经进入 history/latest；lookup 必须从
        // latest 沿不可变 history 回退到 max(writer < task_id)。这同时
        // 覆盖首组与后续组，不再依赖 PA 专用 chained-writer 特判。
        context.fanin_count = static_cast<int32_t>(
            CollectSharedFanin<Ops, false, true, true, true, false, true>(
                state->shared_map, args,
                static_cast<int32_t>(task_id),
                static_cast<int32_t>(state->heap_window),
                stats, context.fanin, lookup_protocol_ok,
                ordinary_lookup_count, &state->fatal.value,
                -1, -1,
                static_cast<int32_t>(task_meta.batch_start)
            )
        );
        if (!lookup_protocol_ok) {
            SetFatal<Ops>(
                state, stats, static_cast<int32_t>(task_id)
            );
            return false;
        }
        stats.result.map_lookups += ordinary_lookup_count;
        for (int32_t edge = 0;
             edge < context.fanin_count; ++edge) {
            stats.result.dependency_signature ^=
                DependencyEdgeSignature(
                    task_id,
                    static_cast<uint32_t>(context.fanin[edge])
                );
        }
        const uint64_t fanin_end =
            TraceTimestamp<Ops>(stats.trace, stats.result);
        WriteTrace<Profile>(
            stats.trace, stats.result,
            static_cast<int32_t>(task_id),
            function_id, TracePhase::Fanin,
            ProfilePhase::Fanin, fanin_begin, fanin_end, 0,
            static_cast<uint32_t>(context.fanin_count)
        );
        build_begin = fanin_end;
    }
    if (kind == TaskKind::Alloc) {
        CompleteTask<Ops>(state, worker, task_id, stats);
    } else if (!PublishCrossCoreExecTask<Ops>(
                   state, worker, task_id, kind, function_id,
                   args, context, stats
               )) {
        return false;
    }
#if defined(PA_TEST_SHARED_SUBMIT_HOOKS)
    Ops::AfterSharedTaskBuild(
        state, worker, task_id, kind
    );
#endif
    const uint64_t build_end =
        TraceTimestamp<Ops>(stats.trace, stats.result);
    WriteTrace<false>(
        stats.trace, stats.result, static_cast<int32_t>(task_id),
        function_id,
        kind == TaskKind::Alloc
            ? TracePhase::AllocComplete
            : TracePhase::WinnerBuild,
        ProfilePhase::ReplayTail, build_begin, build_end
    );

    if constexpr (Dispatched) {
        return CloseSharedDispatchedSubmit<Ops, Profile>(
            state, stats, ticket.task_id, kind,
            ticket.submit_begin
        );
    }
    return CloseSharedCallbackSubmit<Ops, Profile>(
        state, stats, ticket, task_meta, kind
    );
}

#endif  // PA_SCHEDULER_COMMON_PA_SHARED_SUBMIT_PATH_H
