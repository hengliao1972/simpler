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
/**
 * assert_compat.h — shared assertion macros and diagnostics.
 *
 * Factored out of the per-arch runtime `common.h` so that the unified `Tensor`
 * (now in src/common/task_interface/tensor.h) can use `always_assert` /
 * `debug_assert` without pulling in a runtime-specific header. The runtime
 * `common.h` includes this header; `assert_impl` / `get_stacktrace` are
 * defined per linkage target (runtime: orchestration/common.cpp; the nanobind
 * binding: assert_compat.cpp).
 */

#pragma once

// Host / AICPU: <stdexcept> gives us std::runtime_error for AssertionError and
// throw semantics for the assert macros. CCEC does not carry <stdexcept> nor
// C++ exceptions, so the AICore build skips it and points assert_impl at a
// device-only stub that traps (see the __CCE_AICORE__ branch below).
#if !defined(__CCE_AICORE__)
#include <stdexcept>
#include <string>
#endif

#include "data_type.h"  // for PTO_DEVICE_FUNC

#if !defined(__CCE_AICORE__)

/**
 * Get the current stack trace, including file paths and line numbers.
 */
std::string get_stacktrace(int skip_frames = 1);

/**
 * Assertion failure exception with condition, file, line, and stack trace.
 */
class AssertionError : public std::runtime_error {
public:
    AssertionError(const char *condition, const char *file, int line);

    const char *condition() const { return condition_; }
    const char *file() const { return file_; }
    int line() const { return line_; }

private:
    const char *condition_;
    const char *file_;
    int line_;
};

/**
 * Assertion failure handler.
 */
[[noreturn]] void assert_impl(const char *condition, const char *file, int line);

#else  // __CCE_AICORE__

// AICore has no exceptions, no host-side stack-trace / stdlib, and no way to
// overload on __aicore__ attribute (CCEC treats same-signature overloads as
// redefinitions). Assertion failure has no meaningful runtime handler on the
// device either — the diagnostic value only exists on host. So under CCEC
// compile every debug_assert / always_assert to `((void)0)` and skip the
// assert_impl declarations entirely; the surrounding code loses only its
// failure-mode diagnostic, which is host-side by design.
//
// Note this override sits ABOVE the debug_assert / always_assert macro
// definitions below; the guarded #ifdef inside those blocks then compiles
// them out for CCEC (see the __CCE_AICORE__ short-circuit at each macro
// definition).

#endif  // __CCE_AICORE__

/**
 * debug_assert macro:
 * checks the condition in debug builds and throws with a stack trace on failure.
 * It is a no-op in release builds (NDEBUG). On CCEC it is a no-op unconditionally:
 * the AICore has no exception machinery and shared headers are parsed but their
 * host-tagged bodies never run on device.
 */
#if defined(__CCE_AICORE__) || defined(NDEBUG)
#define debug_assert(cond) ((void)0)
#else
#define debug_assert(cond)                          \
    do {                                            \
        if (!(cond)) {                              \
            assert_impl(#cond, __FILE__, __LINE__); \
        }                                           \
    } while (0)
#endif

/**
 * always_assert macro:
 * checks the condition in both debug and release builds. On CCEC it is a
 * no-op for the same reason as debug_assert above.
 */
#if defined(__CCE_AICORE__)
#define always_assert(cond) ((void)0)
#else
#define always_assert(cond)                         \
    do {                                            \
        if (!(cond)) {                              \
            assert_impl(#cond, __FILE__, __LINE__); \
        }                                           \
    } while (0)
#endif

#define PTO_PRAGMA(x) _Pragma(#x)

#if defined(__clang__)
#define MAYBE_UNINITIALIZED_BEGIN                          \
    PTO_PRAGMA(clang diagnostic push)                      \
    PTO_PRAGMA(clang diagnostic ignored "-Wuninitialized") \
    PTO_PRAGMA(clang diagnostic ignored "-Wsometimes-uninitialized")
#define MAYBE_UNINITIALIZED_END PTO_PRAGMA(clang diagnostic pop)
#elif defined(__GNUC__)
#define MAYBE_UNINITIALIZED_BEGIN                        \
    PTO_PRAGMA(GCC diagnostic push)                      \
    PTO_PRAGMA(GCC diagnostic ignored "-Wuninitialized") \
    PTO_PRAGMA(GCC diagnostic ignored "-Wmaybe-uninitialized")
#define MAYBE_UNINITIALIZED_END PTO_PRAGMA(GCC diagnostic pop)
#else
#define MAYBE_UNINITIALIZED_BEGIN
#define MAYBE_UNINITIALIZED_END
#endif
