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
#ifndef TESTS_ATOMIC_PROBE_PROBE_HOST_H
#define TESTS_ATOMIC_PROBE_PROBE_HOST_H

#include "acl/acl.h"

#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace atomic_probe {

inline int32_t DeviceId()
{
    const char *raw = std::getenv("ATOMIC_PROBE_DEVICE");
    if (raw == nullptr || raw[0] == '\0') {
        raw = std::getenv("TASK_DEVICE");
    }
    if (raw == nullptr || raw[0] == '\0') {
        return 0;
    }

    errno = 0;
    char *end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if (errno != 0 || end == raw || value < 0 || value > INT32_MAX) {
        std::fprintf(stderr, "Invalid probe device id: %s\n", raw);
        return -1;
    }
    return static_cast<int32_t>(value);
}

inline bool CheckAcl(aclError error, const char *expression, const char *file, int line)
{
    if (error == ACL_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "ACL error %d from %s at %s:%d\n", static_cast<int>(error), expression, file, line);
    return false;
}

inline aclError LoadAicoreBinaryFromData(const void *data, size_t length, aclrtBinHandle *handle)
{
    aclrtBinaryLoadOption option{};
    option.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
    option.value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
    aclrtBinaryLoadOptions options{&option, 1};
    return aclrtBinaryLoadFromData(data, length, &options, handle);
}

inline bool RequiredUintEnv(const char *name, uint32_t exclusive_max, uint32_t *value)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        std::fprintf(stderr, "%s=0..%u is required\n", name, exclusive_max - 1);
        return false;
    }
    errno = 0;
    char *end = nullptr;
    unsigned long parsed = std::strtoul(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || parsed >= exclusive_max) {
        std::fprintf(stderr, "Invalid %s: %s\n", name, raw);
        return false;
    }
    *value = static_cast<uint32_t>(parsed);
    return true;
}

class Result {
public:
    void Expect(bool condition, const char *label)
    {
        std::printf("[ASSERT] %-40s %s\n", label, condition ? "PASS" : "FAIL");
        if (!condition) {
            failures_++;
        }
    }

    void Observe(const char *label) const
    {
        std::printf("[OBSERVE] %s\n", label);
    }

    int ExitCode() const
    {
        std::printf("[SUMMARY] semantic_failures=%d\n", failures_);
        return failures_ == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

private:
    int failures_ = 0;
};

} // namespace atomic_probe

#define PROBE_ACL_CHECK(expression)                                                                      \
    do {                                                                                                 \
        if (!atomic_probe::CheckAcl((expression), #expression, __FILE__, __LINE__)) {                    \
            return EXIT_FAILURE;                                                                         \
        }                                                                                                \
    } while (false)

#endif // TESTS_ATOMIC_PROBE_PROBE_HOST_H
