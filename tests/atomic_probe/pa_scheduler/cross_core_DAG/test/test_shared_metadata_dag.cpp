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

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "pa_scheduler_core.h"

namespace {

using namespace pa_scheduler;

constexpr uint32_t kTestTasks = 14;
constexpr uint32_t kTestTensorsPerTask = 4;

struct TestTensor {
    TensorArgType access = TensorArgType::NoDependency;
    TensorRefKind reference_kind = TensorRefKind::GmTensor;
    uint32_t symbol_key = 0;
    bool manual_dependency = false;
};

struct TestTask {
    std::array<TestTensor, kTestTensorsPerTask> tensors{};
    uint32_t tensor_count = 0;
};

struct TestSchema {
    std::array<TestTask, kTestTasks> tasks{};

    uint32_t TaskCount() const { return kTestTasks; }

    bool TensorCount(uint32_t task_id, uint32_t &count) const {
        if (task_id >= TaskCount()) {
            return false;
        }
        count = tasks[task_id].tensor_count;
        return count <= kTestTensorsPerTask;
    }

    bool TensorAt(uint32_t task_id, uint32_t tensor_index, SharedDagTensor &tensor) const {
        if (task_id >= TaskCount() || tensor_index >= tasks[task_id].tensor_count) {
            return false;
        }
        const TestTensor &source = tasks[task_id].tensors[tensor_index];
        tensor.access = source.access;
        tensor.reference_kind = source.reference_kind;
        tensor.symbol_key = source.symbol_key;
        tensor.manual_dependency = source.manual_dependency;
        return true;
    }

    bool WriterIntentsAt(uint32_t task_id, SharedDagWriterIntents &intents) const {
        intents.symbol_count = 0;
        intents.ordinary_writer = false;
        if (task_id >= TaskCount()) {
            return false;
        }
        const TestTask &task = tasks[task_id];
        if (task.tensor_count > task.tensors.size()) {
            return false;
        }
        for (uint32_t index = 0; index < task.tensor_count; ++index) {
            const TestTensor &tensor = task.tensors[index];
            if (!IsSharedWriterIntentTag(tensor.access) || tensor.manual_dependency) {
                continue;
            }
            if (tensor.reference_kind == TensorRefKind::SharedOutputRef) {
                if (intents.symbol_count >= kMaxTaskTensors) {
                    return false;
                }
                intents.symbol_keys[intents.symbol_count++] = tensor.symbol_key;
            } else if (tensor.reference_kind == TensorRefKind::GmTensor ||
                       tensor.reference_kind == TensorRefKind::LocalTensor) {
                intents.ordinary_writer = true;
            }
        }
        return true;
    }
};

[[noreturn]] void Fail(const char *message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
}

void Require(bool condition, const char *message) {
    if (!condition) {
        Fail(message);
    }
}

constexpr uint32_t SymbolKey(uint32_t producer_task, uint32_t output_slot) {
    return producer_task * kSharedOutputMaxPerTask + output_slot + 1U;
}

void AddTensor(
    TestSchema &schema, uint32_t task_id, TensorArgType access,
    TensorRefKind reference_kind, uint32_t symbol_key = 0,
    bool manual_dependency = false
) {
    Require(task_id < schema.TaskCount(), "test task id must fit schema");
    TestTask &task = schema.tasks[task_id];
    Require(task.tensor_count < task.tensors.size(), "test tensor list overflow");
    task.tensors[task.tensor_count++] = TestTensor{
        access, reference_kind, symbol_key, manual_dependency
    };
}

bool HasDependency(const SharedMetadataDag &dag, int32_t task_id) {
    for (uint32_t index = 0; index < dag.dependency_count; ++index) {
        if (dag.dependencies[index] == task_id) {
            return true;
        }
    }
    return false;
}

TestSchema BuildSchema() {
    TestSchema schema{};
    constexpr uint32_t kSymbolA = SymbolKey(0, 0);
    constexpr uint32_t kSymbolB = SymbolKey(0, 1);

    // task 0 是两个 whole-object symbol 的 fresh descriptor producer。
    AddTensor(schema, 1, TensorArgType::Input, TensorRefKind::SharedOutputRef, kSymbolA);
    AddTensor(schema, 2, TensorArgType::Inout, TensorRefKind::SharedOutputRef, kSymbolA);
    AddTensor(schema, 3, TensorArgType::Inout, TensorRefKind::SharedOutputRef, kSymbolB);
    AddTensor(schema, 4, TensorArgType::Input, TensorRefKind::SharedOutputRef, kSymbolA);
    AddTensor(schema, 5, TensorArgType::Inout, TensorRefKind::SharedOutputRef, kSymbolA);
    AddTensor(schema, 5, TensorArgType::Input, TensorRefKind::SharedOutputRef, kSymbolB);
    AddTensor(schema, 6, TensorArgType::Inout, TensorRefKind::SharedOutputRef, kSymbolA);

    // ordinary region 使用保守的“全部 ordinary writer”逻辑链；中间的
    // symbol-only task 不能成为 ordinary predecessor。
    AddTensor(schema, 7, TensorArgType::Inout, TensorRefKind::GmTensor);
    AddTensor(schema, 8, TensorArgType::Inout, TensorRefKind::SharedOutputRef, kSymbolB);
    AddTensor(schema, 9, TensorArgType::Input, TensorRefKind::GmTensor);
    AddTensor(schema, 10, TensorArgType::Inout, TensorRefKind::GmTensor);

    // 两个非法 task 分别覆盖同 task 重复 writer 和 future producer。
    AddTensor(schema, 11, TensorArgType::Inout, TensorRefKind::SharedOutputRef, kSymbolA);
    AddTensor(schema, 11, TensorArgType::OutputExisting, TensorRefKind::SharedOutputRef, kSymbolA);
    AddTensor(schema, 12, TensorArgType::Input, TensorRefKind::SharedOutputRef, SymbolKey(13, 0));
    return schema;
}

void TestSameSymbolUsesExactLogicalPredecessor(const TestSchema &schema) {
    SharedMetadataDag dag{};
    Require(BuildSharedMetadataDag(schema, 5, dag), "task 5 DAG must build");
    Require(dag.prepared_task_id == 5, "task 5 DAG identity must be retained");
    Require(dag.writer_count == 1, "task 5 must publish exactly one symbol");
    Require(dag.writer_tensor_mask == 1,
            "task 5 writer tensor mask must retain its INOUT slot");
    Require(dag.writer_symbol_keys[0] == SymbolKey(0, 0), "task 5 writer symbol mismatch");
    Require(dag.writer_previous[0] == 2, "task 5 symbol A must follow task 2, not unrelated task 3");
    Require(dag.dependency_count == 2, "task 5 must wait for its two exact symbol predecessors");
    Require(HasDependency(dag, 2) && HasDependency(dag, 3), "task 5 dependency set must contain tasks 2 and 3");
}

void TestOutOfOrderBuildDoesNotReadPhysicalLatest(const TestSchema &schema) {
    SharedMetadataDag dag{};
    // 不先构建 task 2/5，直接推导 task 6。结果只能来自只读 schema，
    // 不能依赖 last_writer 当前物理值或 earlier task 是否已经发布。
    Require(BuildSharedMetadataDag(schema, 6, dag), "task 6 DAG must build out of order");
    Require(dag.writer_count == 1 && dag.writer_previous[0] == 5, "task 6 must find task 5 as logical predecessor");
    Require(dag.dependency_count == 1 && dag.dependencies[0] == 5, "task 6 must wait only for task 5");
}

void TestIndependentSymbolsDoNotCreateFalseOrder(const TestSchema &schema) {
    SharedMetadataDag dag{};
    Require(BuildSharedMetadataDag(schema, 3, dag), "task 3 DAG must build");
    Require(dag.writer_count == 1 && dag.writer_previous[0] == 0, "first symbol B writer must follow its producer");
    Require(dag.dependency_count == 0, "first symbol B writer must not wait for symbol A writer task 2");

    Require(BuildSharedMetadataDag(schema, 8, dag), "task 8 DAG must build");
    Require(dag.writer_previous[0] == 3, "task 8 symbol B must follow task 3");
    Require(dag.dependency_count == 1 && dag.dependencies[0] == 3, "ordinary task 7 must not enter symbol B chain");
}

void TestReadOnlyTaskWaitsForLatestEarlierWriter(const TestSchema &schema) {
    SharedMetadataDag dag{};
    Require(BuildSharedMetadataDag(schema, 4, dag), "read-only task 4 DAG must build");
    Require(dag.writer_count == 0, "read-only task must not publish metadata");
    Require(dag.writer_tensor_mask == 0,
            "read-only task must not retain writer tensor bits");
    Require(dag.dependency_count == 1 && dag.dependencies[0] == 2, "read-only task must wait for latest symbol A writer");
}

void TestOrdinaryFallbackUsesOnlyOrdinaryWriterChain(const TestSchema &schema) {
    SharedMetadataDag dag{};
    Require(BuildSharedMetadataDag(schema, 9, dag), "ordinary reader DAG must build");
    Require(!dag.ordinary_writer && dag.ordinary_dependency_required, "ordinary reader classification mismatch");
    Require(dag.ordinary_previous_writer == 7, "ordinary reader must ignore intervening symbol-only task");
    Require(dag.dependency_count == 1 && dag.dependencies[0] == 7, "ordinary reader dependency mismatch");

    Require(BuildSharedMetadataDag(schema, 10, dag), "ordinary writer DAG must build");
    Require(dag.ordinary_writer && dag.ordinary_previous_writer == 7, "ordinary writer must retain conservative predecessor");
    Require(dag.writer_tensor_mask == 1,
            "ordinary writer DAG must retain its materialized tensor bit");
    Require(dag.dependency_count == 1 && dag.dependencies[0] == 7, "ordinary writer dependency mismatch");
}

void TestMalformedSchemaFailsClosed(const TestSchema &schema) {
    SharedMetadataDag dag{};
    Require(!BuildSharedMetadataDag(schema, 11, dag), "duplicate writer symbol must fail");
    Require(dag.prepared_task_id == -1, "failed duplicate DAG must remain unpublished");
    Require(!BuildSharedMetadataDag(schema, 12, dag), "future producer must fail");
    Require(!BuildSharedMetadataDag(schema, kTestTasks, dag), "out-of-range task must fail");
}

}  // namespace

int main() {
    const TestSchema schema = BuildSchema();
    TestSameSymbolUsesExactLogicalPredecessor(schema);
    TestOutOfOrderBuildDoesNotReadPhysicalLatest(schema);
    TestIndependentSymbolsDoNotCreateFalseOrder(schema);
    TestReadOnlyTaskWaitsForLatestEarlierWriter(schema);
    TestOrdinaryFallbackUsesOnlyOrdinaryWriterChain(schema);
    TestMalformedSchemaFailsClosed(schema);
    std::puts("PASS: dynamic per-symbol metadata DAG");
    return 0;
}
