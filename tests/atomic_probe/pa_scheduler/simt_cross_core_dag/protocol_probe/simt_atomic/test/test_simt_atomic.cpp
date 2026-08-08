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

#include "../common/atomic_probe.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

using namespace pa_scheduler::simt_cross_core::simt_atomic;

bool ValidateRound(
    uint64_t nonce, uint32_t thread_count, uint64_t cas_final, uint64_t add_final,
    const std::vector<uint64_t> &cas_returns, const std::vector<uint64_t> &add_returns,
    const std::vector<uint64_t> &markers
) {
    const uint64_t cas_initial = ExpectedCasInitial(nonce);
    const uint64_t add_initial = ExpectedAddInitial(nonce);
    uint32_t winner_count = 0U;
    uint32_t loser_count = 0U;
    bool final_matches_desired = false;
    bool markers_valid = true;
    for (uint32_t thread = 0U; thread < thread_count; ++thread) {
        winner_count += cas_returns[thread] == cas_initial ? 1U : 0U;
        loser_count += cas_returns[thread] == cas_final ? 1U : 0U;
        final_matches_desired = final_matches_desired || cas_final == ExpectedCasDesired(nonce, thread);
        markers_valid = markers_valid && markers[thread] == ExpectedThreadMarker(nonce, thread);
    }

    std::vector<uint64_t> sorted_tickets = add_returns;
    std::sort(sorted_tickets.begin(), sorted_tickets.end());
    bool tickets_valid = true;
    for (uint32_t ticket = 0U; ticket < thread_count; ++ticket) {
        tickets_valid = tickets_valid && sorted_tickets[ticket] == add_initial + ticket;
    }

    return winner_count == 1U && loser_count == thread_count - 1U && final_matches_desired &&
           add_final == add_initial + thread_count && tickets_valid && markers_valid && (cas_initial >> 32U) != 0U &&
           (cas_final >> 32U) != 0U && (add_initial >> 32U) != 0U && (add_returns[0] >> 32U) != 0U;
}

bool RunRound(uint32_t round, uint32_t thread_count) {
    const uint64_t nonce = 0xA500000000000000ULL ^ (static_cast<uint64_t>(round + 1U) << 9U);
    std::atomic<uint64_t> cas_cell{ExpectedCasInitial(nonce)};
    std::atomic<uint64_t> add_cell{ExpectedAddInitial(nonce)};
    std::atomic<uint32_t> ready{0U};
    std::atomic<bool> start{false};
    std::vector<uint64_t> cas_returns(thread_count);
    std::vector<uint64_t> add_returns(thread_count);
    std::vector<uint64_t> markers(thread_count);
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for (uint32_t thread = 0U; thread < thread_count; ++thread) {
        threads.emplace_back([&, thread] {
            ready.fetch_add(1U, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            uint64_t observed = ExpectedCasInitial(nonce);
            const uint64_t desired = ExpectedCasDesired(nonce, thread);
            (void
            )cas_cell.compare_exchange_strong(observed, desired, std::memory_order_seq_cst, std::memory_order_seq_cst);
            cas_returns[thread] = observed;
            add_returns[thread] = add_cell.fetch_add(1U, std::memory_order_seq_cst);
            markers[thread] = ExpectedThreadMarker(nonce, thread);
        });
    }
    while (ready.load(std::memory_order_acquire) != thread_count) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (std::thread &thread : threads) {
        thread.join();
    }

    return ValidateRound(
        nonce, thread_count, cas_cell.load(std::memory_order_seq_cst), add_cell.load(std::memory_order_seq_cst),
        cas_returns, add_returns, markers
    );
}

bool ParseRounds(int argc, char **argv, uint32_t *rounds) {
    *rounds = 128U;
    if (argc == 1) {
        return true;
    }
    if (argc != 3 || std::strcmp(argv[1], "--rounds") != 0) {
        std::fprintf(stderr, "usage: %s [--rounds N]\n", argv[0]);
        return false;
    }
    char *end = nullptr;
    const unsigned long value = std::strtoul(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || value == 0U || value > 100000U) {
        std::fprintf(stderr, "invalid --rounds value: %s\n", argv[2]);
        return false;
    }
    *rounds = static_cast<uint32_t>(value);
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    uint32_t rounds = 0U;
    if (!ParseRounds(argc, argv, &rounds)) {
        return EXIT_FAILURE;
    }
    constexpr uint32_t kCpuThreadConfigs[] = {32U, 64U, 128U};
    for (uint32_t thread_count : kCpuThreadConfigs) {
        for (uint32_t round = 0U; round < rounds; ++round) {
            if (!RunRound(round, thread_count)) {
                std::fprintf(
                    stderr, "[FAIL] SIMT atomic CPU semantic model threads=%u round=%u\n", thread_count, round
                );
                return EXIT_FAILURE;
            }
        }
    }
    std::printf(
        "[PASS] SIMT atomic CPU rounds=%u thread_configs=32/64/128: unique CAS winner, old values and "
        "add-ticket permutation\n",
        rounds
    );
    return EXIT_SUCCESS;
}
