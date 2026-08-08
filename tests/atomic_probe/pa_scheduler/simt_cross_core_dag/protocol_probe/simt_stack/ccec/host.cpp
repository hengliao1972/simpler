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

#include "../common/simt_stack_probe.h"

#include "acl/acl.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace pa_scheduler::simt_cross_core::simt_stack;

struct Options {
    std::string kernel_path;
    std::string acl_config;
    int32_t device = 0;
    uint32_t runs = 10U;
};

bool ParseUnsigned(const char *raw, uint64_t maximum, uint64_t *value) {
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || parsed > maximum) {
        return false;
    }
    *value = static_cast<uint64_t>(parsed);
    return true;
}

bool ParseOptions(int argc, char **argv, Options *options) {
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--kernel") == 0 && index + 1 < argc) {
            options->kernel_path = argv[++index];
        } else if (std::strcmp(argv[index], "--acl-config") == 0 && index + 1 < argc) {
            options->acl_config = argv[++index];
        } else if (std::strcmp(argv[index], "--device") == 0 && index + 1 < argc) {
            uint64_t value = 0U;
            if (!ParseUnsigned(argv[++index], static_cast<uint64_t>(std::numeric_limits<int32_t>::max()), &value)) {
                std::fprintf(stderr, "invalid --device value: %s\n", argv[index]);
                return false;
            }
            options->device = static_cast<int32_t>(value);
        } else if (std::strcmp(argv[index], "--runs") == 0 && index + 1 < argc) {
            uint64_t value = 0U;
            if (!ParseUnsigned(argv[++index], 100000U, &value) || value == 0U) {
                std::fprintf(stderr, "invalid --runs value: %s\n", argv[index]);
                return false;
            }
            options->runs = static_cast<uint32_t>(value);
        } else {
            std::fprintf(
                stderr, "usage: %s --kernel FILE --acl-config FILE [--device N] [--runs N]\n", argv[0]
            );
            return false;
        }
    }
    if (options->kernel_path.empty() || options->acl_config.empty()) {
        std::fprintf(stderr, "--kernel FILE and --acl-config FILE are required\n");
        return false;
    }
    std::ifstream config(options->acl_config);
    if (!config.good()) {
        std::fprintf(stderr, "cannot open ACL config: %s\n", options->acl_config.c_str());
        return false;
    }
    return true;
}

bool CheckAcl(aclError error, const char *operation) {
    if (error == ACL_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "ACL error %d: %s\n", static_cast<int>(error), operation);
    return false;
}

bool ReadBinary(const std::string &path, std::vector<char> *data) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        std::fprintf(stderr, "cannot open kernel binary: %s\n", path.c_str());
        return false;
    }
    const std::streamsize size = input.tellg();
    if (size <= 0) {
        std::fprintf(stderr, "kernel binary is empty: %s\n", path.c_str());
        return false;
    }
    input.seekg(0, std::ios::beg);
    data->resize(static_cast<size_t>(size));
    if (!input.read(data->data(), size)) {
        std::fprintf(stderr, "cannot read kernel binary: %s\n", path.c_str());
        return false;
    }
    return true;
}

class AclSession {
public:
    ~AclSession() {
        if (device_state_ != nullptr) {
            (void)aclrtFree(device_state_);
        }
        if (binary_ != nullptr) {
            (void)aclrtBinaryUnLoad(binary_);
        }
        if (stream_ != nullptr) {
            (void)aclrtDestroyStream(stream_);
        }
        if (device_set_) {
            (void)aclrtResetDevice(device_);
        }
        if (acl_initialized_) {
            (void)aclFinalize();
        }
    }

    bool Initialize(
        int32_t device, const std::string &acl_config, const std::vector<char> &binary_data
    ) {
        device_ = device;
        if (!CheckAcl(aclInit(acl_config.c_str()), "aclInit(stack config)")) {
            return false;
        }
        acl_initialized_ = true;
        if (!CheckAcl(aclrtSetDevice(device_), "aclrtSetDevice")) {
            return false;
        }
        device_set_ = true;
        const char *soc_name = aclrtGetSocName();
        if (soc_name == nullptr || soc_name[0] == '\0') {
            std::fprintf(stderr, "aclrtGetSocName returned an empty SoC name\n");
            return false;
        }
        soc_name_ = soc_name;
        if (!CheckAcl(aclrtCreateStream(&stream_), "aclrtCreateStream")) {
            return false;
        }

        aclrtBinaryLoadOption option{};
        option.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
        option.value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
        aclrtBinaryLoadOptions options{&option, 1U};
        return CheckAcl(
                   aclrtBinaryLoadFromData(binary_data.data(), binary_data.size(), &options, &binary_),
                   "aclrtBinaryLoadFromData(SIMT stack ELF)"
               ) &&
               CheckAcl(aclrtBinaryGetFunctionByEntry(binary_, 0U, &function_), "aclrtBinaryGetFunctionByEntry") &&
               CheckAcl(
                   aclrtMalloc(&device_state_, sizeof(ProbeState), ACL_MEM_MALLOC_HUGE_FIRST),
                   "aclrtMalloc(SIMT stack ProbeState)"
               );
    }

    const std::string &SocName() const { return soc_name_; }

    bool Run(ProbeState *state) {
        if (!CheckAcl(
                aclrtMemcpy(device_state_, sizeof(*state), state, sizeof(*state), ACL_MEMCPY_HOST_TO_DEVICE),
                "aclrtMemcpy(H2D SIMT stack ProbeState)"
            )) {
            return false;
        }
        struct KernelArgs {
            uint64_t state_pointer;
        } args{reinterpret_cast<uint64_t>(device_state_)};
        static_assert(sizeof(KernelArgs) == sizeof(uint64_t), "unexpected CCEC kernel argument ABI");
        return CheckAcl(
                   aclrtLaunchKernelWithHostArgs(function_, 1U, stream_, nullptr, &args, sizeof(args), nullptr, 0U),
                   "aclrtLaunchKernelWithHostArgs(SIMT stack)"
               ) &&
               CheckAcl(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream(SIMT stack)") &&
               CheckAcl(
                   aclrtMemcpy(state, sizeof(*state), device_state_, sizeof(*state), ACL_MEMCPY_DEVICE_TO_HOST),
                   "aclrtMemcpy(D2H SIMT stack ProbeState)"
               );
    }

private:
    bool acl_initialized_ = false;
    bool device_set_ = false;
    int32_t device_ = 0;
    std::string soc_name_;
    aclrtStream stream_ = nullptr;
    aclrtBinHandle binary_ = nullptr;
    aclrtFuncHandle function_ = nullptr;
    void *device_state_ = nullptr;
};

template <typename T>
void FillWords(T *object, uint64_t value) {
    static_assert(sizeof(T) % sizeof(uint64_t) == 0U, "word-filled object must have a uint64-sized ABI");
    uint64_t *words = reinterpret_cast<uint64_t *>(object);
    std::fill(words, words + sizeof(T) / sizeof(uint64_t), value);
}

void InitializeState(ProbeState *state, uint64_t nonce) {
    FillWords(state, kPoison);
    state->control.magic = kProbeMagic;
    state->control.version = kProbeVersion;
    state->control.launch_nonce = nonce;
    state->control.launch_threads = kLaunchThreads;
    state->control.warp_count = kWarpCount;
    state->control.frame_words = kFrameWords;
    state->control.frame_depth = kFrameDepth;
    std::fill(std::begin(state->control.reserved), std::end(state->control.reserved), 0U);
}

bool ValidateState(const ProbeState &state, uint64_t nonce, std::string *reason) {
    if (state.control.magic != kProbeMagic || state.control.version != kProbeVersion ||
        state.control.launch_nonce != nonce || state.control.launch_threads != kLaunchThreads ||
        state.control.warp_count != kWarpCount || state.control.frame_words != kFrameWords ||
        state.control.frame_depth != kFrameDepth ||
        !std::all_of(std::begin(state.control.reserved), std::end(state.control.reserved), [](uint64_t value) {
            return value == 0U;
        })) {
        *reason = "control changed";
        return false;
    }
    for (uint32_t thread = 0U; thread < kLaunchThreads; ++thread) {
        if (state.thread_checksums[thread] != ExpectedThreadChecksum(nonce, thread)) {
            *reason = "thread checksum mismatch at tid=" + std::to_string(thread);
            return false;
        }
    }
    for (uint32_t warp = 0U; warp < kWarpCount; ++warp) {
        const WarpReport &report = state.reports[warp];
        if (report.marker != kReportMagic || report.launch_nonce != nonce || report.thread_id != warp * kWarpSize ||
            report.warp_id != warp || report.lane_id != 0U || report.frame_words != kFrameWords ||
            report.checksum != ExpectedChecksum(nonce, warp) || report.begin_clock == 0U ||
            report.end_clock < report.begin_clock || report.reserved != 0U) {
            *reason = "warp report mismatch at warp=" + std::to_string(warp);
            return false;
        }
    }
    const ProbeResult &result = state.result;
    if (result.magic != kResultMagic || result.launch_nonce != nonce || result.completed_warps != kWarpCount ||
        result.expected_warps != kWarpCount || result.checksum_xor != ExpectedChecksumXor(nonce) ||
        result.status != 1U ||
        !std::all_of(std::begin(result.reserved), std::end(result.reserved), [](uint64_t value) {
            return value == 0U;
        })) {
        *reason = "Scalar result mismatch";
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    Options options;
    if (!ParseOptions(argc, argv, &options)) {
        return 2;
    }

    std::vector<char> binary_data;
    if (!ReadBinary(options.kernel_path, &binary_data)) {
        return 1;
    }
    AclSession session;
    if (!session.Initialize(options.device, options.acl_config, binary_data)) {
        return 1;
    }
    if (session.SocName().rfind("Ascend950", 0U) != 0U) {
        std::fprintf(stderr, "SIMT stack probe requires Ascend950*, got: %s\n", session.SocName().c_str());
        return 1;
    }

    auto state = std::make_unique<ProbeState>();
    std::printf(
        "[DEVICE] id=%d soc=%s state_bytes=%zu threads=%u warps=%u acl_config=%s\n", options.device,
        session.SocName().c_str(), sizeof(ProbeState), kLaunchThreads, kWarpCount, options.acl_config.c_str()
    );
    for (uint32_t run = 0U; run < options.runs; ++run) {
        const uint64_t nonce = UINT64_C(0x535441434B000000) ^ (static_cast<uint64_t>(run) << 32U) ^ (run + 1U);
        InitializeState(state.get(), nonce);
        if (!session.Run(state.get())) {
            std::fprintf(stderr, "SIMT stack launch failed at run=%u\n", run + 1U);
            return 1;
        }
        std::string reason;
        if (!ValidateState(*state, nonce, &reason)) {
            std::fprintf(stderr, "SIMT stack validation failed at run=%u: %s\n", run + 1U, reason.c_str());
            return 1;
        }
    }
    std::printf(
        "[PASS] A5 SIMT stack ACL-init runs=%u same_address_reuse=yes frame=%u*%uB divergence_depth=%u\n",
        options.runs, kFrameDepth, kFrameBytes, kDivergenceDepth
    );
    return 0;
}
