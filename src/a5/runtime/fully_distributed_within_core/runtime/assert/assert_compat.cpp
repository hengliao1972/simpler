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
#include "common.h"

#if !defined(__CCE_AICORE__)

#ifdef __linux__
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>
#endif

// Unified-log error sink. Forward-declared here rather than pulled via
// common/unified_log.h: that header lives under common/log/include, which is
// not on every runtime target's include path. The symbol resolves at link time
// for runtime targets and at dlopen time for the orchestration .so.
extern "C" void unified_log_error(const char *func, const char *fmt, ...);

#ifdef __linux__
static std::string addr_to_line(const char *executable, void *addr, std::string *inline_chain = nullptr) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "addr2line -e %s -f -C -p -i %p 2>/dev/null", executable, addr);

    std::array<char, 256> buffer;
    std::string raw_output;

    FILE *pipe = popen(cmd, "r");  // NOLINT(bugprone-command-processor): diagnostic addr2line helper.
    if (pipe) {
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            raw_output += buffer.data();
        }
        pclose(pipe);
    }

    if (raw_output.empty() || raw_output.find("??") != std::string::npos) {
        return "";
    }

    std::vector<std::string> lines;
    size_t pos = 0;
    while (pos < raw_output.size()) {
        size_t nl = raw_output.find('\n', pos);
        if (nl == std::string::npos) nl = raw_output.size();
        std::string line = raw_output.substr(pos, nl - pos);
        while (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (!line.empty()) lines.push_back(line);
        pos = nl + 1;
    }

    if (lines.empty()) return "";

    if (inline_chain && lines.size() > 1) {
        *inline_chain = "";
        for (size_t j = 1; j < lines.size(); j++) {
            *inline_chain += "    [inlined by] " + lines[j] + "\n";
        }
    }

    return lines.front();
}
#endif

std::string get_stacktrace(int skip_frames) {
    (void)skip_frames;
    std::string result;
#ifdef __linux__
    const int max_frames = 64;
    void *buffer[max_frames];
    int nframes = backtrace(buffer, max_frames);
    char **symbols = backtrace_symbols(buffer, nframes);

    if (symbols) {
        result = "Stack trace:\n";
        for (int i = skip_frames; i < nframes; i++) {
            std::string frame_info;

            void *addr = static_cast<char *>(buffer[i]) - 1;

            Dl_info dl_info;
            std::string inline_chain;
            if (dladdr(addr, &dl_info) && dl_info.dli_fname) {
                const auto rel_addr_value =
                    reinterpret_cast<uintptr_t>(addr) - reinterpret_cast<uintptr_t>(dl_info.dli_fbase);
                void *rel_addr = reinterpret_cast<void *>(rel_addr_value);
                std::string addr2line_result = addr_to_line(dl_info.dli_fname, rel_addr, &inline_chain);

                if (addr2line_result.empty()) {
                    addr2line_result = addr_to_line(dl_info.dli_fname, addr, &inline_chain);
                }

                if (!addr2line_result.empty()) {
                    frame_info = std::string(dl_info.dli_fname) + ": " + addr2line_result;
                }
            }

            if (frame_info.empty()) {
                std::string frame(symbols[i]);

                size_t start = frame.find('(');
                size_t end = frame.find('+', start);
                if (start != std::string::npos && end != std::string::npos) {
                    std::string mangled = frame.substr(start + 1, end - start - 1);
                    int status;
                    char *demangled = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
                    if (status == 0 && demangled) {
                        frame = frame.substr(0, start + 1) + demangled + frame.substr(end);
                        free(demangled);
                    }
                }
                frame_info = frame;
            }

            char buf[16];
            snprintf(buf, sizeof(buf), "  #%d ", i - skip_frames);
            result += buf + frame_info + "\n";
            if (!inline_chain.empty()) {
                result += inline_chain;
            }
        }
        free(symbols);
    }
#else
    result = "(Stack trace is only available on Linux)\n";
#endif
    return result;
}

static std::string build_assert_message(const char *condition, const char *file, int line) {
    std::string msg = "Assertion failed: " + std::string(condition) + "\n";
    msg += "  Location: " + std::string(file) + ":" + std::to_string(line) + "\n";
    msg += get_stacktrace(3);
    return msg;
}

AssertionError::AssertionError(const char *condition, const char *file, int line) :
    std::runtime_error(build_assert_message(condition, file, line)),
    condition_(condition),
    file_(file),
    line_(line) {}

[[noreturn]] void assert_impl(const char *condition, const char *file, int line) {
    unified_log_error(__FUNCTION__, "\n========================================");
    unified_log_error(__FUNCTION__, "Assertion failed: %s", condition);
    unified_log_error(__FUNCTION__, "Location: %s:%d", file, line);
    unified_log_error(__FUNCTION__, "%s", get_stacktrace(2).c_str());
    unified_log_error(__FUNCTION__, "========================================\n");

    throw AssertionError(condition, file, line);
}

#endif  // !__CCE_AICORE__
