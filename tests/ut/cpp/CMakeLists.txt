# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# C++ unit tests for src/common/hierarchical using GoogleTest.
# Run with: cmake --build . --target run_ut_cpp
# Or directly: ctest --test-dir build/ut_cpp

cmake_minimum_required(VERSION 3.15)
project(hierarchical_ut CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Gate for tests that may need CANN headers/libs at build time or that only
# make sense on Ascend hardware runners.  Default OFF so no-hw runners
# (ubuntu-latest / macos-latest, no CANN installed) can build the tree
# without touching anything CANN-adjacent.  CI's hw jobs
# (ut-a2a3 / ut-a5) pass -DSIMPLER_ENABLE_HARDWARE_TESTS=ON.
#
# Rule of thumb: any new test that either
#   (a) links against CANN libs (libascendcl, libhcomm, libruntime), or
#   (b) needs to dlopen an onboard libhost_runtime.so to be meaningful,
# goes under `if(SIMPLER_ENABLE_HARDWARE_TESTS)` below.
option(SIMPLER_ENABLE_HARDWARE_TESTS
       "Build tests that require Ascend hardware toolchain to build or run"
       OFF)

# ---------------------------------------------------------------------------
# GoogleTest — fast path uses the system install (apt libgtest-dev / brew
# googletest / yum-distributed /usr/local). Fall back to FetchContent on
# runners that don't preinstall it (currently the a2a3 self-hosted runner).
# Either way the rest of the file sees GTEST_LIB / GTEST_MAIN_LIB /
# GTEST_INCLUDE_DIRS with a consistent meaning.
#
# Both the system install and FetchContent build gtest with the compiler
# default libstdc++ ABI (cxx11), matching the tests in this directory. If
# you hand-compile gtest with `-D_GLIBCXX_USE_CXX11_ABI=0` and drop it in
# `/usr/local/lib*`, find_library will pick it up and link will fail with
# `undefined reference to testing::internal::EqFailure(...)` — see
# docs/troubleshooting/ut-cpp-gtest-abi.md for the workaround.
# ---------------------------------------------------------------------------
if(APPLE)
    set(GTEST_SEARCH_PATHS /opt/homebrew/lib /usr/local/lib)
    set(GTEST_INCLUDE_DIRS /opt/homebrew/include /usr/local/include)
else()
    set(GTEST_SEARCH_PATHS /usr/local/lib64 /usr/local/lib)
    set(GTEST_INCLUDE_DIRS /usr/local/include)
endif()

find_library(GTEST_LIB      gtest      PATHS ${GTEST_SEARCH_PATHS})
find_library(GTEST_MAIN_LIB gtest_main PATHS ${GTEST_SEARCH_PATHS})

if(NOT GTEST_LIB OR NOT GTEST_MAIN_LIB)
    message(STATUS "System GoogleTest not found — fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL      https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
        URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    # FetchContent exposes the gtest / gtest_main targets directly; the
    # per-test functions below use these names uniformly with the system
    # path (where they resolve to the absolute .so/.a path via find_library).
    set(GTEST_LIB gtest)
    set(GTEST_MAIN_LIB gtest_main)
    set(GTEST_INCLUDE_DIRS "")  # include dirs are carried by the gtest target
endif()

# ---------------------------------------------------------------------------
# A2A3 runtime sources and stubs for ring-buffer / tensormap tests.
#
# Bundled into an OBJECT library so the runtime .cpp files compile once and
# the resulting .o files are reused across every test executable. PUBLIC
# include dirs / compile options propagate to consumers via
# target_link_libraries, so test executables get the same flags they would
# have had if the sources were listed directly.
# ---------------------------------------------------------------------------
set(A2A3_RUNTIME_DIR ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime)

add_library(a2a3_rt_objs OBJECT
    ${A2A3_RUNTIME_DIR}/pto_ring_buffer.cpp
    ${A2A3_RUNTIME_DIR}/shared/pto_shared_memory.cpp
    ${A2A3_RUNTIME_DIR}/scheduler/pto_scheduler.cpp
    ${A2A3_RUNTIME_DIR}/shared/pto_tensormap.cpp
    ${A2A3_RUNTIME_DIR}/shared/pto_runtime2_init.cpp
    ${CMAKE_SOURCE_DIR}/stubs/test_stubs.cpp
)
target_include_directories(a2a3_rt_objs PUBLIC
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/orchestration
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/common
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/sim/aicpu
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
    ${CMAKE_SOURCE_DIR}/../../../src/common
)

function(add_a2a3_runtime_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE
        a2a3_rt_objs
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

function(add_a2a3_orchestrator_runtime_test name src)
    add_executable(${name}
        ${src}
        ${A2A3_RUNTIME_DIR}/pto_orchestrator.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/aicpu/scope_stats_collector_aicpu.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/aicpu/tensor_dump_aicpu.cpp
    )
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime
    )
    target_link_libraries(${name} PRIVATE
        a2a3_rt_objs
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

# ---------------------------------------------------------------------------
# Distributed runtime sources under test, bundled into an OBJECT library so
# they compile once and link into every hierarchical test.
# ---------------------------------------------------------------------------
set(HIERARCHICAL_SRC_DIR ${CMAKE_SOURCE_DIR}/../../../src/common/hierarchical)
set(WORKER_SRC_DIR ${CMAKE_SOURCE_DIR}/../../../src/common/worker)

add_library(hierarchical_objs OBJECT
    ${HIERARCHICAL_SRC_DIR}/types.cpp
    ${HIERARCHICAL_SRC_DIR}/tensormap.cpp
    ${HIERARCHICAL_SRC_DIR}/ring.cpp
    ${HIERARCHICAL_SRC_DIR}/scope.cpp
    ${HIERARCHICAL_SRC_DIR}/remote_wire.cpp
    ${HIERARCHICAL_SRC_DIR}/remote_endpoint.cpp
    ${HIERARCHICAL_SRC_DIR}/orchestrator.cpp
    ${HIERARCHICAL_SRC_DIR}/worker_manager.cpp
    ${HIERARCHICAL_SRC_DIR}/scheduler.cpp
    ${HIERARCHICAL_SRC_DIR}/worker.cpp
    ${WORKER_SRC_DIR}/chip_worker.cpp
    # Host-side assert_impl / get_stacktrace for the unified Tensor's
    # always_assert (init_external, reached via make_tensor_external in
    # remote_wire.cpp / orchestrator.cpp). Mirrors the binding's CMake.
    # ODR invariant: assert_impl / get_stacktrace are ALSO defined in the runtime
    # orchestration/common.cpp. No target may link both that and this file, or
    # the linker sees duplicate symbols. This target links neither runtime
    # orchestration TU, so only this host-side definition is present.
    ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface/assert_compat.cpp
)
target_include_directories(hierarchical_objs PUBLIC
    ${HIERARCHICAL_SRC_DIR}
    ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
    ${WORKER_SRC_DIR}
)

function(add_hierarchical_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE
        hierarchical_objs
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

function(add_a2a3_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/orchestration
        ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime
        ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/common
        ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
        ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
        # src/common is needed so `#include "utils/device_arena.h"` resolves.
        ${CMAKE_SOURCE_DIR}/../../../src/common
    )
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

function(add_a5_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/orchestration
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/runtime
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/common
        ${CMAKE_SOURCE_DIR}/../../../src/a5/platform/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
        ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
        # src/common is needed so `#include "utils/device_arena.h"` resolves.
        ${CMAKE_SOURCE_DIR}/../../../src/common
    )
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# ---------------------------------------------------------------------------
# A5 runtime sources, mirroring a2a3_rt_objs. Bundled into an OBJECT library
# so the runtime .cpp files compile once and the resulting .o files are
# reused across every a5 runtime test executable.
# ---------------------------------------------------------------------------
set(A5_RUNTIME_DIR ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/runtime)

add_library(a5_rt_objs OBJECT
    ${A5_RUNTIME_DIR}/pto_ring_buffer.cpp
    ${A5_RUNTIME_DIR}/shared/pto_shared_memory.cpp
    ${A5_RUNTIME_DIR}/scheduler/pto_scheduler.cpp
    ${A5_RUNTIME_DIR}/shared/pto_tensormap.cpp
    ${A5_RUNTIME_DIR}/shared/pto_runtime2_init.cpp
    ${CMAKE_SOURCE_DIR}/stubs/test_stubs.cpp
)
target_include_directories(a5_rt_objs PUBLIC
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/orchestration
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/runtime
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/common
    ${CMAKE_SOURCE_DIR}/../../../src/a5/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/sim/aicpu
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
    ${CMAKE_SOURCE_DIR}/../../../src/common
)

function(add_a5_runtime_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE
        a5_rt_objs
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

function(add_a5_orchestrator_runtime_test name src)
    add_executable(${name}
        ${src}
        ${A5_RUNTIME_DIR}/pto_orchestrator.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/aicpu/scope_stats_collector_aicpu.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/aicpu/tensor_dump_aicpu.cpp
    )
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime
    )
    target_link_libraries(${name} PRIVATE
        a5_rt_objs
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

function(add_task_interface_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
    )
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

function(add_common_utils_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        # src/common is needed so utils/elf_build_id.h, utils/device_arena.h,
        # etc. resolve via the "utils/..." prefix.
        ${CMAKE_SOURCE_DIR}/../../../src/common
    )
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

enable_testing()

# ---------------------------------------------------------------------------
# Hierarchical runtime tests (src/common/hierarchical/)
# ---------------------------------------------------------------------------
add_hierarchical_test(test_tensormap  hierarchical/test_tensormap.cpp)
add_hierarchical_test(test_ring  hierarchical/test_ring.cpp)
add_hierarchical_test(test_scope      hierarchical/test_scope.cpp)
add_hierarchical_test(test_orchestrator hierarchical/test_orchestrator.cpp)
add_hierarchical_test(test_scheduler  hierarchical/test_scheduler.cpp)
add_hierarchical_test(test_remote_wire hierarchical/test_remote_wire.cpp)
add_hierarchical_test(test_remote_endpoint hierarchical/test_remote_endpoint.cpp)

# ---------------------------------------------------------------------------
# Types / task_interface tests (src/common/task_interface/)
# ---------------------------------------------------------------------------
add_task_interface_test(test_child_memory types/test_child_memory.cpp)
add_task_interface_test(test_chip_max_tensor_args types/test_chip_max_tensor_args.cpp)
add_task_interface_test(test_call_config types/test_call_config.cpp)

# Contract test for upload_chip_callable_buffer caller-buffer immutability.
# Pulls both task_interface (callable.h) and src/common (utils/fnv1a_64.h),
# so it can't use add_task_interface_test as-is.
add_executable(test_chip_callable_upload_immutable
    types/test_chip_callable_upload_immutable.cpp
)
target_include_directories(test_chip_callable_upload_immutable PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/common
    ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
)
target_link_libraries(test_chip_callable_upload_immutable PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_chip_callable_upload_immutable COMMAND test_chip_callable_upload_immutable)
set_tests_properties(test_chip_callable_upload_immutable PROPERTIES LABELS "no_hardware")

# ---------------------------------------------------------------------------
# Common utilities tests (src/common/utils/)
# ---------------------------------------------------------------------------
add_common_utils_test(test_elf_build_id common/test_elf_build_id.cpp)
add_common_utils_test(test_runtime_orch_so common/test_runtime_orch_so.cpp)
add_common_utils_test(test_device_arena common/test_device_arena.cpp)

add_executable(test_scope_stats_collector
    common/test_scope_stats_collector.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/host/scope_stats_collector.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/sim/host/profiling_copy.cpp
    ${CMAKE_SOURCE_DIR}/stubs/test_stubs.cpp
)
target_include_directories(test_scope_stats_collector PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
    ${CMAKE_SOURCE_DIR}/../../../src/common
)
target_link_libraries(test_scope_stats_collector PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_scope_stats_collector COMMAND test_scope_stats_collector)
set_tests_properties(test_scope_stats_collector PROPERTIES LABELS "no_hardware")

# Per-callable_id orch SO file naming regression (see rtStreamSynchronize
# 507018 root cause). Compiles the a2a3 onboard `create_orch_so_file`
# against the test source so it runs on no-hw runners too.
add_executable(test_orch_so_file
    common/test_orch_so_file.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/onboard/aicpu/orch_so_file.cpp
)
target_include_directories(test_orch_so_file PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
)
target_link_libraries(test_orch_so_file PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_orch_so_file COMMAND test_orch_so_file)
set_tests_properties(test_orch_so_file PROPERTIES LABELS "no_hardware")

# ---------------------------------------------------------------------------
# A2A3 tests (src/a2a3/runtime/tensormap_and_ringbuffer/)
# ---------------------------------------------------------------------------
add_a2a3_test(test_a2a3_fatal a2a3/test_a2a3_fatal.cpp)

# PTO2 runtime-linked tests
add_a2a3_runtime_test(test_task_allocator   a2a3/test_task_allocator.cpp)
add_a2a3_runtime_test(test_dep_list_pool    a2a3/test_dep_list_pool.cpp)
add_a2a3_runtime_test(test_scheduler_state  a2a3/test_scheduler_state.cpp)
add_a2a3_runtime_test(test_task_state       a2a3/test_task_state.cpp)
add_a2a3_runtime_test(test_ready_queue      a2a3/test_ready_queue.cpp)
add_a2a3_runtime_test(test_shared_memory    a2a3/test_shared_memory.cpp)
add_a2a3_runtime_test(test_a2a3_tensormap   a2a3/test_tensormap.cpp)
add_a2a3_runtime_test(test_fanin_pool       a2a3/test_fanin_pool.cpp)
add_a2a3_orchestrator_runtime_test(test_a2a3_orchestrator_fanin a2a3/test_orchestrator_fanin.cpp)
add_a2a3_runtime_test(test_spsc_queue       a2a3/test_spsc_queue.cpp)
add_a2a3_runtime_test(test_wiring           a2a3/test_wiring.cpp)
add_a2a3_runtime_test(test_aicore_completion_mailbox a2a3/test_aicore_completion_mailbox.cpp)
add_a2a3_runtime_test(test_a2a3_scope_stats_collector a2a3/test_scope_stats_collector.cpp)

# ---------------------------------------------------------------------------
# A5 tests (src/a5/runtime/tensormap_and_ringbuffer/)
# ---------------------------------------------------------------------------
add_a5_test(test_a5_fatal a5/test_a5_fatal.cpp)

# FDWIC's device-side PollBatch accumulator is header-only. Compile the exact
# production implementation through the A5Sim target macros so the 24-bit
# split boundary is exercised without introducing a second model.
add_a5_test(test_fdwic_swimlane_poll_batch a5/test_fdwic_swimlane_poll_batch.cpp)
target_compile_definitions(test_fdwic_swimlane_poll_batch PRIVATE __CPU_SIM=1)
target_include_directories(test_fdwic_swimlane_poll_batch BEFORE PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../src/a5/platform/sim/aicore
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/fully_distributed_within_core/runtime
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/fully_distributed_within_core/common
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime
)

# DistTensorMap ring 是 header-only production 实现。同一源码先锁默认
# CAP=128，再重编译四个结构边界，覆盖额外 bucket control、普通固定前缀
# 和单桶 hash 特例；reference 只比较逻辑语义，不复制被测 ring。
function(add_fdwic_tensor_map_ring_test name ring_cap)
    add_a5_test(${name} a5/test_fdwic_tensor_map_retire.cpp)
    target_compile_definitions(${name} PRIVATE
        __CPU_SIM=1
        PTO_FDWIC_TENSORMAP_RING_CAP=${ring_cap}
    )
    target_include_directories(${name} BEFORE PRIVATE
        ${CMAKE_SOURCE_DIR}/../../../src/a5/platform/sim/aicore
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/fully_distributed_within_core/runtime
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/fully_distributed_within_core/common
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime
    )
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

add_fdwic_tensor_map_ring_test(test_fdwic_tensor_map_retire 128)
add_fdwic_tensor_map_ring_test(test_fdwic_tensor_map_ring_cap32 32)
add_fdwic_tensor_map_ring_test(test_fdwic_tensor_map_ring_cap64 64)
add_fdwic_tensor_map_ring_test(test_fdwic_tensor_map_ring_cap256 256)
add_fdwic_tensor_map_ring_test(test_fdwic_tensor_map_ring_cap16384 16384)

# shared sidecar 先独立验证布局和 AICPU 一次性初始化，production 顶层仍
# fail-closed。五个 CAP 覆盖 bucket-control 长度变化，额外 private 目标
# 锁住“只在 shared DistGlobal 尾部追加”的 ABI 边界。
function(add_fdwic_shared_tensor_map_state_test name shared_map ring_cap)
    add_a5_test(${name} a5/test_fdwic_shared_tensor_map_state.cpp)
    target_compile_definitions(${name} PRIVATE
        __CPU_SIM=1
        PTO_FDWIC_SHARED_MAP=${shared_map}
        PTO_FDWIC_TENSORMAP_RING_CAP=${ring_cap}
    )
    target_include_directories(${name} BEFORE PRIVATE
        ${CMAKE_SOURCE_DIR}/../../../src/a5/platform/sim/aicore
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/fully_distributed_within_core/runtime
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/fully_distributed_within_core/common
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime
    )
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

add_fdwic_shared_tensor_map_state_test(test_fdwic_private_tensor_map_state_layout 0 128)
add_fdwic_shared_tensor_map_state_test(test_fdwic_shared_tensor_map_state 1 128)
add_fdwic_shared_tensor_map_state_test(test_fdwic_shared_tensor_map_state_cap32 1 32)
add_fdwic_shared_tensor_map_state_test(test_fdwic_shared_tensor_map_state_cap64 1 64)
add_fdwic_shared_tensor_map_state_test(test_fdwic_shared_tensor_map_state_cap256 1 256)
add_fdwic_shared_tensor_map_state_test(test_fdwic_shared_tensor_map_state_cap16384 1 16384)

# shared ring primitive 仍不接 Submit；五种 CAP 都直接实例化 production
# 状态机，验证 exact-turn、窗口 lookup、整 task preflight、seq 发布/复用与
# 独立 reference 差分。
function(add_fdwic_shared_tensor_map_ring_test name ring_cap)
    add_a5_test(${name} a5/test_fdwic_shared_tensor_map_ring.cpp)
    target_compile_definitions(${name} PRIVATE
        __CPU_SIM=1
        PTO_FDWIC_SHARED_MAP=1
        PTO_FDWIC_TENSORMAP_RING_CAP=${ring_cap}
    )
    target_include_directories(${name} BEFORE PRIVATE
        ${CMAKE_SOURCE_DIR}/../../../src/a5/platform/sim/aicore
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/fully_distributed_within_core/runtime
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/fully_distributed_within_core/common
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime
    )
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

add_fdwic_shared_tensor_map_ring_test(test_fdwic_shared_tensor_map_ring 128)
add_fdwic_shared_tensor_map_ring_test(test_fdwic_shared_tensor_map_ring_cap32 32)
add_fdwic_shared_tensor_map_ring_test(test_fdwic_shared_tensor_map_ring_cap64 64)
add_fdwic_shared_tensor_map_ring_test(test_fdwic_shared_tensor_map_ring_cap256 256)
add_fdwic_shared_tensor_map_ring_test(test_fdwic_shared_tensor_map_ring_cap16384 16384)

# 容量错误必须在真实 Register→Build 边界中止，并让后续 eager Begin 在
# Claim 前返回。该目标直接编译 production CPU-sim dist_engine TU，避免
# 用测试替身重写 Submit 状态机。
add_a5_test(test_fdwic_submit_capacity a5/test_fdwic_submit_capacity.cpp)
target_compile_definitions(test_fdwic_submit_capacity PRIVATE
    __CPU_SIM=1
    PTO_FDWIC_TENSORMAP_RING_CAP=128
)
target_include_directories(test_fdwic_submit_capacity BEFORE PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../src/a5/platform/sim/aicore
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/fully_distributed_within_core/runtime
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/fully_distributed_within_core/common
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime
)
set_tests_properties(test_fdwic_submit_capacity PROPERTIES LABELS "no_hardware")

# The shared Submit adapter owns the mapping from task-transaction outcomes to
# structured fatal codes. This target compiles the production CPU-sim TU under
# the shared build identity without opening the AICPU backend-ready gate.
add_a5_test(test_fdwic_shared_submit_contract a5/test_fdwic_shared_submit_contract.cpp)
target_compile_definitions(test_fdwic_shared_submit_contract PRIVATE
    __CPU_SIM=1
    PTO_FDWIC_SHARED_MAP=1
    PTO_FDWIC_TENSORMAP_RING_CAP=128
)
target_include_directories(test_fdwic_shared_submit_contract BEFORE PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../src/a5/platform/sim/aicore
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/fully_distributed_within_core/runtime
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/fully_distributed_within_core/common
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime
)
set_tests_properties(test_fdwic_shared_submit_contract PROPERTIES LABELS "no_hardware")

# This gate executes the production compete-first Submit flow under the shared
# build identity. It verifies transaction ordering at the Kernel/Alloc/joint
# boundaries instead of testing the ring primitive or result adapter alone.
add_a5_test(test_fdwic_shared_submit_wiring a5/test_fdwic_shared_submit_wiring.cpp)
target_compile_definitions(test_fdwic_shared_submit_wiring PRIVATE
    __CPU_SIM=1
    PTO_FDWIC_SHARED_MAP=1
    PTO_FDWIC_TENSORMAP_RING_CAP=128
)
target_include_directories(test_fdwic_shared_submit_wiring BEFORE PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../src/a5/platform/sim/aicore
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/fully_distributed_within_core/runtime
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/fully_distributed_within_core/common
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime
)
set_tests_properties(test_fdwic_shared_submit_wiring PROPERTIES LABELS "no_hardware")

# This multi-worker gate compiles the same production CPU-sim Submit TU and
# proves that a future-turn winner waits for committed_tasks before fanin,
# publication, and Build. SPIN_WAIT_HINT is observed only by the test.
add_a5_test(test_fdwic_shared_multiworker a5/test_fdwic_shared_multiworker.cpp)
target_compile_definitions(test_fdwic_shared_multiworker PRIVATE
    __CPU_SIM=1
    PTO_FDWIC_SHARED_MAP=1
    PTO_FDWIC_TENSORMAP_RING_CAP=128
    PTO_FDWIC_TRACE_ENABLED=0
)
target_include_directories(test_fdwic_shared_multiworker BEFORE PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../src/a5/platform/sim/aicore
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/fully_distributed_within_core/runtime
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/fully_distributed_within_core/common
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime
)
set_tests_properties(test_fdwic_shared_multiworker PROPERTIES LABELS "no_hardware" TIMEOUT 15)

# PTO_DIST_H 是 AICPU 控制面的纯配置输入。独立目标直接编译生产解析
# helper，锁定仅 ASCII 十进制数字串和闭区间边界，不引入设备或 runtime。
add_executable(test_fdwic_config_parse a5/test_fdwic_config_parse.cpp)
target_include_directories(test_fdwic_config_parse PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/fully_distributed_within_core/runtime
)
target_link_libraries(test_fdwic_config_parse PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_fdwic_config_parse COMMAND test_fdwic_config_parse)
set_tests_properties(test_fdwic_config_parse PROPERTIES LABELS "no_hardware")

# Stable Host/AICPU/AICore identity is compiled in both artifact modes.  The
# shared target intentionally verifies that the first-stage backend remains
# fail-closed while its ABI identity is still well-formed.
function(add_fdwic_build_identity_test name shared_map)
    add_executable(${name} a5/test_fdwic_build_identity.cpp)
    target_compile_definitions(${name} PRIVATE
        PTO_FDWIC_SHARED_MAP=${shared_map}
        PTO_FDWIC_TENSORMAP_RING_CAP=128
    )
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/fully_distributed_within_core/runtime
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/fully_distributed_within_core/common
        ${CMAKE_SOURCE_DIR}/../../../src/a5/platform/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
        ${CMAKE_SOURCE_DIR}/../../../src/common
    )
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

add_fdwic_build_identity_test(test_fdwic_build_identity_private 0)
add_fdwic_build_identity_test(test_fdwic_build_identity_shared 1)

# A5 trb runtime UTs — mirror of a2a3 trb runtime UTs, link against a5_rt_objs.
# Target names carry the a5_ prefix because hierarchical/test_tensormap (and
# the unprefixed a2a3 runtime targets test_scheduler_state / test_ready_queue
# / ...) already own those bare names.
add_a5_runtime_test(test_a5_task_allocator   a5/test_task_allocator.cpp)
add_a5_runtime_test(test_a5_dep_list_pool    a5/test_dep_list_pool.cpp)
add_a5_runtime_test(test_a5_scheduler_state  a5/test_scheduler_state.cpp)
add_a5_runtime_test(test_a5_task_state       a5/test_task_state.cpp)
add_a5_runtime_test(test_a5_ready_queue      a5/test_ready_queue.cpp)
add_a5_runtime_test(test_a5_shared_memory    a5/test_shared_memory.cpp)
add_a5_runtime_test(test_a5_tensormap        a5/test_tensormap.cpp)
add_a5_runtime_test(test_a5_fanin_pool       a5/test_fanin_pool.cpp)
add_a5_orchestrator_runtime_test(test_a5_orchestrator_fanin a5/test_orchestrator_fanin.cpp)
add_a5_runtime_test(test_a5_spsc_queue       a5/test_spsc_queue.cpp)
add_a5_runtime_test(test_a5_wiring           a5/test_wiring.cpp)
add_a5_runtime_test(test_a5_aicore_completion_mailbox a5/test_aicore_completion_mailbox.cpp)

# Host logger silent/off behavior — no runtime deps, just compile host_log.cpp
# alongside the test (faster than dlopen'ing libsimpler_log.so for a unit test).
set(SIMPLER_LOG_DIR ${CMAKE_SOURCE_DIR}/../../../src/common/log)
add_executable(test_host_log_off
    a5/test_host_log_off.cpp
    ${SIMPLER_LOG_DIR}/host_log.cpp
)
target_include_directories(test_host_log_off PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${SIMPLER_LOG_DIR}
    ${SIMPLER_LOG_DIR}/include
)
target_link_libraries(test_host_log_off PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_host_log_off COMMAND test_host_log_off)

# ---------------------------------------------------------------------------
# Standalone distributed-atomic minibench tests (MB-1/3/5/9)
# ---------------------------------------------------------------------------
# These tests re-implement the core data structures (claim cursor, frontier
# CAS, SharedTensorMap ring, private replica) from scratch using std::atomic
# and std::thread. No dist_engine.cpp dependency — pure host, pure logic.
function(add_standalone_dist_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/../../../src/common)
    target_link_libraries(${name} PRIVATE pthread)
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

add_standalone_dist_test(test_dist_tensormap_ring a2a3/test_dist_tensormap_ring.cpp)
add_standalone_dist_test(test_dist_atomic_mb1_claim a5/test_dist_atomic_mb1_claim.cpp)
add_standalone_dist_test(test_dist_atomic_mb3_frontier a5/test_dist_atomic_mb3_frontier.cpp)
add_standalone_dist_test(test_dist_atomic_mb5_shared_map a5/test_dist_atomic_mb5_shared_map.cpp)
add_standalone_dist_test(test_dist_atomic_mb9_private_det a5/test_dist_atomic_mb9_private_det.cpp)

# Hardware-gated tests.  Block is only entered when the project is configured
# with -DSIMPLER_ENABLE_HARDWARE_TESTS=ON.  CI's no-hw `ut` job does not pass
# this flag, so nothing here is compiled there — this is the structural
# invariant that lets future hw tests freely link against CANN libraries
# without breaking the no-hw build.
if(SIMPLER_ENABLE_HARDWARE_TESTS)
    # libhost_runtime.so is loaded via dlopen at runtime — it is the
    # subject under test and mirrors how ChipWorker selects a runtime
    # backend in production.  libascendcl.so, by contrast, is generic
    # CANN infrastructure: we link it directly so the test code uses
    # <acl/acl.h> types and values normally instead of going through
    # dlsym boilerplate.
    #
    # ctest label still gates execution via
    #   -L "^requires_hardware(_a2a3)?$"
    # so we can both (a) only build on hw runners, and (b) still filter
    # by runtime role when multiple hw tests coexist.
    set(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/../../..")
    set(HOST_RUNTIME_LIB
        "${PROJECT_ROOT}/build/lib/a2a3/onboard/tensormap_and_ringbuffer/libhost_runtime.so"
    )
    # libsimpler_log.so must be loaded RTLD_GLOBAL before host_runtime dlopen
    # so undefined HostLogger / unified_log_* symbols in host_runtime resolve.
    # Mirrors what ChipWorker::init does in production. The library is
    # process-global (one .so per host toolchain, not per arch/variant).
    set(SIMPLER_LOG_LIB
        "${PROJECT_ROOT}/build/lib/libsimpler_log.so"
    )

    if(NOT DEFINED ENV{ASCEND_HOME_PATH})
        message(FATAL_ERROR
            "SIMPLER_ENABLE_HARDWARE_TESTS=ON requires ASCEND_HOME_PATH to be set "
            "(usually via `source $ASCEND_HOME_PATH/bin/setenv.bash`)")
    endif()
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
    find_library(ASCENDCL_LIB ascendcl
        PATHS ${ASCEND_HOME_PATH}/lib64 NO_DEFAULT_PATH REQUIRED)

    # Black-box lifecycle UT for the comm_* C API on Path D. See the file
    # header for the contract it pins down.
    add_executable(test_comm_lifecycle hardware/test_comm_lifecycle.cpp)
    target_include_directories(test_comm_lifecycle PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${PROJECT_ROOT}/src/a2a3/platform/include
        ${PROJECT_ROOT}/src/common/platform/include
        # Shared platform_comm headers (comm.h / comm_context.h) live under
        # src/common so both a2a3 HCCL and a5/a2a3 sim can use them.
        ${PROJECT_ROOT}/src/common
        ${ASCEND_HOME_PATH}/include
    )
    target_compile_definitions(test_comm_lifecycle PRIVATE
        PTO_HOST_RUNTIME_LIB_PATH="${HOST_RUNTIME_LIB}"
        SIMPLER_LOG_LIB_PATH="${SIMPLER_LOG_LIB}"
    )
    target_link_libraries(test_comm_lifecycle PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        ${ASCENDCL_LIB}
        pthread
        dl
    )
    add_test(NAME test_comm_lifecycle COMMAND test_comm_lifecycle)
    set_tests_properties(test_comm_lifecycle PROPERTIES
        LABELS "requires_hardware_a2a3"
        RESOURCE_GROUPS "2,npus:1"
    )
endif()
