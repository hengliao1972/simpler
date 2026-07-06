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
 * fully_distributed_within_core engine.
 *
 * SPMD orchestration + scheduling + execution on the AI cores. See
 * docs/fully_distributed_within_core.md for the authoritative design and
 * src/.../docs/RUNTIME_LOGIC.md for the local overview.
 *
 * Each AICore worker thread runs dist_core_main(), which:
 *   1. replays the full orchestration submit stream (every core builds an
 *      identical per-core TensorMap and computes identical deterministic GM
 *      output-heap addresses; only ownership differs);
 *   2. on each rt_submit_*, races to claim the task on one of two global
 *      cursors (cube for AIC-anchored, vector for AIV-only). The winner is
 *      owner = builder = executor and builds the task into its private ring;
 *   3. runs an EXECUTE-FIRST run-ahead loop: on every submit point it first
 *      drains ready owned tasks (and pulls follower deposits), THEN claims at
 *      most this one new task. Because claim+build is fast but execute is slow,
 *      interleaving execution with claiming stops a fast core from greedily
 *      claiming a full ring of consecutive tasks: while it executes a long task
 *      other cores advance the cursor and claim subsequent tasks (load balance,
 *      see docs §6/§6.1). The ring (small, kPrivateSlots) only back-pressures
 *      when genuinely full of not-yet-ready tasks. After orchestration returns,
 *      a final loop drains the ring to completion. A task is ready once all its
 *      fan-in producers have set their entry in the global completion-flag
 *      ring; on completion the owner sets its own flag (release).
 *
 * This file is compiled into the AICPU .so (build_config aicore source_dirs do
 * not include runtime/), but dist_core_main runs ON the AICore worker threads
 * (invoked through a function pointer), so kernels execute on AICore threads
 * with their sim TLS in place.
 *
 * M2 scope: single-core tasks (1C / 1V) only — sufficient for benchmark_bgemm.
 * Multi-core co-ownership (MIX / 2V, block.won) is M3; GM heap reclamation is
 * M4. A MIX task encountered in M2 raises a fatal error.
 *
 * CCEC-onboard readiness (C.3.c in the plan): most of the SPMD engine is now
 * gated correctly for CCEC — helpers wear PTO_DEVICE_FUNC, shared engine state
 * flows through a GM `DistGlobal` pointed at by g_dist_ptr, the per-shard
 * cursors / block.won / DistCore references carry `__gm__`, and the atom_*
 * wrappers ship both plain and __gm__ overloads. What still blocks a full
 * CCEC build is that DistTensorMap / DistCore / Tensor / TaskOutputTensors
 * define member functions but CCE rejects address-space qualifiers on the
 * `this` parameter ("function type may not be qualified with an address
 * space"), so calling `self->map.lookup(t)` on a __gm__ DistCore is illegal.
 * The fix — hoist those methods to free functions taking a __gm__ pointer,
 * or store engine state as plain POD accessed only through GM helpers — is
 * the remaining rework in this stage and touches every method call site on
 * these types.
 */

#include "dist_engine/dist_engine.h"

// Basic C-stdlib headers are always safe (CCEC ships a full libc / libstdc++
// subset; these show up transitively anyway). Everything else waits behind the
// DIST_HOST_ONLY gate below.
#include <cstdint>
#include <cstring>

// PTO2_PROFILING is defined by profiling_config.h (default 1). Pull it in
// explicitly so DIST_HOST_ONLY below can read it without relying on transitive
// includes from the project headers further down (which themselves depend on
// stdlib headers we haven't decided to bring in yet).
#include "profiling_config.h"

// Compile-time gate for host-only facilities.
//
// DIST_HOST_ONLY covers the swimlane tracer (per-task span capture + JSON
// dump), the sim-only trace-driven replay (use_example_exec_time busy-wait),
// the host wall-clock timer (now_ns / thread_cpu_ns), and every AICPU-side
// diagnostic (fprintf, getenv, signal handlers, watchdog dumps). These are all
// unavailable under CCEC — no <atomic>, <chrono>, <vector>, <csignal>, no
// posix APIs — so they collapse to a single gate: enabled only when
// PTO2_PROFILING is on AND we are NOT compiling for the AICore target. Sim /
// AICPU builds get the full facility; CCEC AICore compiles them all out.
//
// PTO2_PROFILING comes from profiling_config.h (default 1; explicit 0 for
// perf-only sim builds). __CCE_AICORE__ is defined by ccec under
// --cce-aicore-arch=*. No #ifndef fallback on purpose: undefined ⇒ off.
#if PTO2_PROFILING && !defined(__CCE_AICORE__)
#define DIST_HOST_ONLY 1
#else
#define DIST_HOST_ONLY 0
#endif

// Legacy aliases still referenced throughout this file. Kept as the single
// unified gate above so the code below reads exactly the same regardless of
// which alias a call site historically used.
#define DIST_TRACE_ENABLED DIST_HOST_ONLY
#define DIST_SIM_HOST_CLOCK DIST_HOST_ONLY

#if DIST_HOST_ONLY
// Host / sim / AICPU only: full stdlib. CCEC AICore skips these.
#include <atomic>
#include <chrono>
#include <csignal>
#include <ctime>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#endif

// Fatal-log call site macro. Host / sim / AICPU forward to fprintf(stderr, ...);
// AICore has no host stdio (nor variadic-fprintf support in CCEC's runtime), so
// the call collapses to a no-op — the fatal flag itself already tears the run
// down and the failure is observable via runtime state.
#if DIST_HOST_ONLY
#define DIST_ERRF(...) fprintf(stderr, __VA_ARGS__)
#else
#define DIST_ERRF(...) ((void)0)
#endif

#if defined(__CCE_AICORE__)
#define DIST_API_ATTR __attribute__((weak))
#else
#define DIST_API_ATTR
#endif

#include "callable.h"
#include "common/core_type.h"
#include "intrinsic.h"
#include "pto2_dispatch_payload.h"
#include "pto_constants.h"
#include "pto_runtime2.h"
#include "pto_submit_types.h"
#include "pto_types.h"
#include "runtime.h"

#if defined(__CPU_SIM)
extern "C" PTO_DEVICE_FUNC void aicpu_orchestration_entry(const L2TaskArgs &orch_args);
#elif defined(__CCE_AICORE__)
extern "C" PTO_DEVICE_FUNC void aicpu_orchestration_entry(const L2TaskArgs &orch_args) __attribute__((weak));
extern "C" PTO_DEVICE_FUNC int32_t pto_call_linked_kernel_aic(int32_t func_id, __gm__ int64_t *args)
    __attribute__((weak));
extern "C" PTO_DEVICE_FUNC int32_t pto_call_linked_kernel_aiv(int32_t func_id, __gm__ int64_t *args)
    __attribute__((weak));
#endif
#if defined(__CPU_SIM) && !defined(__CCE_AICORE__)
extern "C" void framework_bind_runtime(PTO2Runtime *rt);
#endif

#if defined(__CCE_AICORE__)
extern "C" __attribute__((weak)) PTO_DEVICE_FUNC void *memcpy(void *dst, const void *src, unsigned long n) {
    uint8_t *d = reinterpret_cast<uint8_t *>(dst);
    const uint8_t *s = reinterpret_cast<const uint8_t *>(src);
    for (unsigned long i = 0; i < n; i++) {
        d[i] = s[i];
    }
    return dst;
}
#endif

#if defined(__CCE_AICORE__)
// AICore has no scheduler to yield to; a spin hint here would be meaningless.
// Provide a local no-op so we don't have to depend on the AICPU spin_hint.h
// header being reachable from the AICore include path.
#include "inner_kernel.h"
#ifndef SPIN_WAIT_HINT
#define SPIN_WAIT_HINT() ((void)0)
#endif
#else
#include "spin_hint.h"
#endif
#include "tensor.h"
#include "tensor_create_info.h"

// -----------------------------------------------------------------------------
// Atomic wrappers.
//
// dist_engine.cpp is compiled into two targets: sim / AICPU (host toolchain,
// full <atomic>) and AICore (CCEC, no <atomic> and stricter language subset).
// Instead of #if'ing every load / store / CAS, we route every shared-memory
// atomic through a tiny wrapper layer that uses __atomic_* GCC intrinsics on
// both sides (host libstdc++ implements them on top of the same fences as
// std::atomic; trb's aicore_executor / aicpu_executor already use these under
// CCEC).
//
// Fields that used to be `std::atomic<T> x;` become plain `volatile T x;` so
// the object layout matches on both compilers (std::atomic<T> is not a POD
// under CCEC, and its members are host-tagged). Callers use atom_load(x, mo),
// atom_store(x, v, mo), atom_cas(x, exp, des, s_mo, f_mo), atom_fetch_add(x, d,
// mo), atom_thread_fence(mo) — all trivially inlined.
//
// One CCEC-backend gotcha (verified via aicore_executor.cpp comment): 32-bit
// atomic add on a GM (addrspace 1) address is rejected with "Cannot select:
// AtomicLoadAdd s32". Every field that participates in atom_fetch_add is
// therefore int64_t, not int32_t.
// -----------------------------------------------------------------------------
#if DIST_HOST_ONLY
void dist_dump_state(int);  // defined below; dumps full engine state for hangs
#endif

namespace {

template <typename T>
PTO_DEVICE_FUNC inline T atom_load(const volatile T &p, int mo) {
    return __atomic_load_n(&p, mo);
}
template <typename T>
PTO_DEVICE_FUNC inline T atom_load(const T &p, int mo) {
    return __atomic_load_n(&p, mo);
}
template <typename T, typename V>
PTO_DEVICE_FUNC inline void atom_store(volatile T &p, V v, int mo) {
    __atomic_store_n(&p, static_cast<T>(v), mo);
}
template <typename T, typename V>
PTO_DEVICE_FUNC inline void atom_store(T &p, V v, int mo) {
    __atomic_store_n(&p, static_cast<T>(v), mo);
}
template <typename T>
PTO_DEVICE_FUNC inline bool atom_cas_weak(volatile T &p, T &expected, T desired, int s_mo, int f_mo) {
    return __atomic_compare_exchange_n(&p, &expected, desired, /*weak=*/true, s_mo, f_mo);
}
template <typename T>
PTO_DEVICE_FUNC inline bool atom_cas_strong(volatile T &p, T &expected, T desired, int s_mo, int f_mo) {
    return __atomic_compare_exchange_n(&p, &expected, desired, /*weak=*/false, s_mo, f_mo);
}
template <typename T>
PTO_DEVICE_FUNC inline T atom_fetch_add(volatile T &p, T d, int mo) {
    return __atomic_fetch_add(&p, d, mo);
}
template <typename T>
PTO_DEVICE_FUNC inline T atom_fetch_sub(volatile T &p, T d, int mo) {
    return __atomic_fetch_sub(&p, d, mo);
}
PTO_DEVICE_FUNC inline void atom_thread_fence(int mo) { __atomic_thread_fence(mo); }

// __gm__ overloads: CCEC's shared DistGlobal lives in GM, so every field
// accessed through g_dist_ptr carries the __gm__ address space. Overload
// resolution otherwise fails ("cannot bind volatile T & to __gm__ volatile T").
// On sim / AICPU __gm__ expands to nothing (intrinsic.h) and these become
// duplicates of the plain overloads above — guard the block so we only emit
// them for the CCEC target.
#if defined(__CCE_AICORE__)
template <typename T>
PTO_DEVICE_FUNC inline T atom_load(__gm__ const volatile T &p, int mo) {
    return __atomic_load_n(&p, mo);
}
template <typename T>
PTO_DEVICE_FUNC inline T atom_load(__gm__ const T &p, int mo) {
    return __atomic_load_n(&p, mo);
}
template <typename T, typename V>
PTO_DEVICE_FUNC inline void atom_store(__gm__ volatile T &p, V v, int mo) {
    __atomic_store_n(&p, static_cast<T>(v), mo);
}
template <typename T, typename V>
PTO_DEVICE_FUNC inline void atom_store(__gm__ T &p, V v, int mo) {
    __atomic_store_n(&p, static_cast<T>(v), mo);
}
template <typename T>
PTO_DEVICE_FUNC inline bool atom_cas_weak(__gm__ volatile T &p, T &expected, T desired, int s_mo, int f_mo) {
    return __atomic_compare_exchange_n(&p, &expected, desired, /*weak=*/true, s_mo, f_mo);
}
template <typename T>
PTO_DEVICE_FUNC inline bool atom_cas_strong(__gm__ volatile T &p, T &expected, T desired, int s_mo, int f_mo) {
    return __atomic_compare_exchange_n(&p, &expected, desired, /*weak=*/false, s_mo, f_mo);
}
template <typename T>
PTO_DEVICE_FUNC inline T atom_fetch_add(__gm__ volatile T &p, T d, int mo) {
    return __atomic_fetch_add(&p, d, mo);
}
template <typename T>
PTO_DEVICE_FUNC inline T atom_fetch_sub(__gm__ volatile T &p, T d, int mo) {
    return __atomic_fetch_sub(&p, d, mo);
}
#endif  // __CCE_AICORE__

}  // namespace

namespace {

// -----------------------------------------------------------------------------
// Tunables. The completion-flag ring is sized to hold an entire run without
// wrap (>= total tasks); the GM output heap is a BOUNDED RING reclaimed by the
// completion frontier (M4, §9.5/§11.4) rather than a run-sized bump.
// -----------------------------------------------------------------------------
// Kept deliberately SMALL: the out-of-order window is num_cores * kPrivateSlots,
// and this also caps how far a single core can run ahead of "ready-to-execute".
// A large ring lets one fast core greedily claim a long run of consecutive tasks
// and serialize them while other cores starve (load imbalance, docs §6.1). OoO
// capacity should come from the core-count dimension, not a deep per-core ring.
constexpr int32_t kPrivateSlots = 4;  // PRIVATE_TASK_SLOT_NUM (back-pressure cap)
// Ring slots a core reserves for draining block.won deposits addressed to its
// lane. Self-claimed tasks (consumers / single-core / own anchor subtask) may
// only occupy kPrivateSlots - kWonReserve slots, so a follower can ALWAYS pull
// and run an (immediately-ready) deposit even when its ring is otherwise full of
// not-yet-ready consumers — breaking the consumer<->deposit priority inversion.
constexpr int32_t kWonReserve = 2;
constexpr int32_t kMaxFanin = 16;        // max distinct producers a task waits on
constexpr int32_t kOutPoolSlots = 1024;  // per-core ring of materialized output Tensors
constexpr int32_t kMapCap = 16384;       // per-core producer-map capacity (distinct regions)
constexpr int32_t kFlagCap = 1 << 16;    // global completion-flag ring (>= total tasks)

// M4 GM-heap reclamation (§9.5/§11.4).
//   kHeapRingDefault — bounded physical heap ring (env PTO_DIST_HEAP_MB overrides,
//     in MiB). The deterministic virtual bump is unbounded; physical address is
//     (virtual_offset mod ring). A region is reused only after its previous
//     occupant's task id <= R (the reclaim frontier), enforced by back-pressure.
//   kHDefault — dependency-span bound H (env PTO_DIST_H overrides): every consumer
//     of task N has id <= N + H. R = F - H. Must be >= the graph's true heap span
//     or a producer region could be recycled while a late consumer still reads it
//     (run-time-checked → fatal "heap span exceeded").
constexpr size_t kHeapRingDefault = 64ull << 20;
constexpr int32_t kHDefault = 64;

// -----------------------------------------------------------------------------
// Per-core producer map (the "full per-core duplicate TensorMap").
//
// A faithful, compact stand-in for PTO2TensorMap: keyed by GM byte range, it
// records the most recent producer task id of each written region. INPUT/INOUT
// fan-in resolves to the producer(s) whose region overlaps. Exact-region writes
// (e.g. an INOUT accumulation chain) replace in place; new regions append.
// Every core builds an identical map by replaying the same submit stream.
// -----------------------------------------------------------------------------
// Intrusive entry, modeled on PTO2TensorMapEntry (tensormap_and_ringbuffer) but
// compact: it keys overlap on a byte range [lo, hi) instead of mirroring a full
// Tensor cache line, since the distributed map only needs producer lookup.
//   - bucket chain (doubly linked) — O(1) unlink during cleanup
//   - task chain (singly linked)   — cleanup frees a retired task's entries by
//                                     walking ITS chain, never scanning the pool
struct MapEntry {
    uint64_t buf_addr;       // Tensor.buffer.addr (GM buffer base, bytes) — hash key
    uint64_t lo;             // byte offset of view origin within buffer
    uint64_t hi;             // byte offset one-past the view extent
    int32_t producer;        // task id that wrote this region
    int32_t bucket;          // owning bucket index, or -1 when free
    int32_t next_in_bucket;  // bucket-chain links (entry indices, -1 = none)
    int32_t prev_in_bucket;
    int32_t next_in_task;  // task-chain link (entry index, -1 = none)
};

// Hash buckets (power of 2). Hashing by buffer BASE address groups every
// sub-region of one buffer into one chain; overlap is then tested per entry.
constexpr int32_t kMapBuckets = 1 << 13;  // 8192
constexpr int32_t kMapBucketShift = 13;   // log2(kMapBuckets)
// Per-task entry-head window (power of 2). Task `id` parks its entries under
// slot id & (kTaskWindow-1); the slot is recycled by id + kTaskWindow. cleanup
// retires a task once it leaves the H span, so kTaskWindow MUST exceed H (with
// margin) or a slot could be reused before its prior task is cleaned. Validated
// against g_dist.H at register time.
constexpr int32_t kTaskWindow = 1 << 10;  // 1024  (>> kHDefault = 64)
constexpr int32_t kTaskWindowMask = kTaskWindow - 1;

// Per-core producer map ("full per-core duplicate TensorMap"), a direct compact
// port of tensormap_and_ringbuffer's PTO2TensorMap (hash table + bucket chains +
// per-task entry tracking + free list + lazy invalidation + cleanup_retired).
//
// WHY (vs. the original O(count) linear array, which made submit O(N^2)):
// bgemm writes hundreds of disjoint tiles of ONE flattened output buffer, so the
// old `entries[count]` grew with the whole run and every lookup/insert rescanned
// it. Following the proven runtime, we instead:
//   * hash by buffer base + chain — distinct buffers cost O(1);
//   * RETIRE by H window — an entry whose producer is older than `alive_floor`
//     (= N - H) can never be a fan-in of any future task (a consumer of producer
//     p has id <= p + H, §9.5/§11.4, the same bound under which p's GM heap region
//     is recycled), so cleanup frees it. This bounds each chain to ~the live
//     H-window instead of the entire run → O(N*H) ~ O(N).
// Like the reference, insert ALWAYS links a fresh entry under its producer's task
// chain (no in-place replace), so cleanup_retired can free a task's entries via
// that chain without scanning; lookup returns the MAX (newest) overlapping
// producer, which subsumes the old replace-in-place semantics.
//
// `alive_floor` is N-derived (deterministic, identical on every core), never
// frontier-based (timing-dependent), so every per-core map — including the free
// list and cleanup progress — evolves identically. Determinism is preserved.
struct DistTensorMap {
    MapEntry entries[kMapCap];
    int32_t buckets[kMapBuckets];     // bucket head entry idx, or -1
    int32_t task_heads[kTaskWindow];  // per-task entry-chain head idx, or -1
    int32_t free_head;                // recycled-slot free list head, or -1
    int32_t high_water;               // next never-used slot in `entries`
    int32_t alive_floor;              // producer < alive_floor == retired
    int32_t cleaned_upto;             // tasks < cleaned_upto already freed

    // Under CCEC, DistTensorMap lives in GM (nested inside DistGlobal.cores[]),
    // and CCE rejects address-space qualifiers on the `this` parameter of
    // non-static member functions ("function type may not be qualified with an
    // address space"). To keep a single implementation for both sim and CCEC,
    // every method that mutates or reads state takes an explicit `self` — the
    // qualifier travels on the reference instead of on `this`. `__gm__`
    // expands to empty on sim (see common/intrinsic.h), so sim call sites are
    // byte-for-byte equivalent to plain `self.field` access. Reference (not
    // pointer) because self is never null on any call path (dist_core_main
    // constructs it from &g_dist.cores[core_idx]) and the call sites read
    // cleaner without the explicit `&` / `->` dance.

    PTO_DEVICE_FUNC static void reset(__gm__ DistTensorMap &self) {
        self.free_head = -1;
        self.high_water = 0;
        self.alive_floor = 0;
        self.cleaned_upto = 0;
        for (int32_t i = 0; i < kMapBuckets; i++)
            self.buckets[i] = -1;
        for (int32_t i = 0; i < kTaskWindow; i++)
            self.task_heads[i] = -1;
    }

    PTO_DEVICE_FUNC static uint32_t hash(uint64_t addr) {
        addr *= 0x9E3779B97F4A7C15ULL;  // golden-ratio multiplicative mix
        return static_cast<uint32_t>(addr >> (64 - kMapBucketShift));
    }

    PTO_DEVICE_FUNC static void byte_range(__gm__ const Tensor &t, uint64_t &addr, uint64_t &lo, uint64_t &hi) {
        const uint64_t esz = get_element_size(t.dtype);
        addr = t.buffer.addr;
        lo = t.start_offset * esz;
        // Inline extent_elem() here — Tensor::extent_elem is a non-static member
        // and CCE cannot invoke it on a __gm__ receiver. `is_contiguous` /
        // `extent_elem_cache` / `shapes[]` are POD fields, so reading them off
        // the __gm__ reference is fine.
        uint64_t ext;
        if (t.is_contiguous) {
            ext = 1;
            for (uint32_t i = 0; i < t.ndims; i++)
                ext *= t.shapes[i];
        } else {
            ext = t.extent_elem_cache;
        }
        hi = (t.start_offset + ext) * esz;
    }

#if defined(__CCE_AICORE__)
    // Non-__gm__ overload for CCEC callers whose Tensor lives on the AICore
    // kernel stack (orch-side, address-space-agnostic type). Body is a byte
    // copy of the __gm__ version — POD field reads only.
    PTO_DEVICE_FUNC static void byte_range(const Tensor &t, uint64_t &addr, uint64_t &lo, uint64_t &hi) {
        const uint64_t esz = get_element_size(t.dtype);
        addr = t.buffer.addr;
        lo = t.start_offset * esz;
        uint64_t ext;
        if (t.is_contiguous) {
            ext = 1;
            for (uint32_t i = 0; i < t.ndims; i++)
                ext *= t.shapes[i];
        } else {
            ext = t.extent_elem_cache;
        }
        hi = (t.start_offset + ext) * esz;
    }
#endif

    PTO_DEVICE_FUNC static int32_t alloc_slot(__gm__ DistTensorMap &self) {
        if (self.free_head >= 0) {
            const int32_t s = self.free_head;
            self.free_head = self.entries[s].next_in_bucket;
            return s;
        }
        if (self.high_water < kMapCap) return self.high_water++;
        return -1;  // pool exhausted (live H-window exceeds kMapCap)
    }

    // Unlink `idx` from its bucket chain (O(1) via prev) and push to the free list.
    PTO_DEVICE_FUNC static void free_entry(__gm__ DistTensorMap &self, int32_t idx) {
        __gm__ MapEntry &e = self.entries[idx];
        if (e.prev_in_bucket < 0) self.buckets[e.bucket] = e.next_in_bucket;
        else self.entries[e.prev_in_bucket].next_in_bucket = e.next_in_bucket;
        if (e.next_in_bucket >= 0) self.entries[e.next_in_bucket].prev_in_bucket = e.prev_in_bucket;
        e.bucket = -1;
        e.next_in_bucket = self.free_head;
        self.free_head = idx;
    }

    // Free every entry produced by retired tasks [cleaned_upto, new_floor) by
    // walking each task's own chain (never the whole pool). Mirrors PTO2TensorMap
    // ::cleanup_retired. Advances alive_floor so lookups skip the freed window.
    PTO_DEVICE_FUNC static void advance_retire(__gm__ DistTensorMap &self, int32_t N, int32_t H) {
        const int32_t new_floor = N - H;
        if (new_floor <= self.cleaned_upto) {  // nothing newly retired
            if (new_floor > self.alive_floor) self.alive_floor = new_floor;
            return;
        }
        for (int32_t id = self.cleaned_upto; id < new_floor; id++) {
            int32_t cur = self.task_heads[id & kTaskWindowMask];
            while (cur >= 0) {
                const int32_t nxt = self.entries[cur].next_in_task;
                debug_assert(self.entries[cur].producer == id);
                free_entry(self, cur);
                cur = nxt;
            }
            self.task_heads[id & kTaskWindowMask] = -1;
        }
        self.cleaned_upto = new_floor;
        self.alive_floor = new_floor;
    }

    // Link a fresh entry for `producer`'s write of `t`'s region. Always a new
    // entry (no in-place replace) so it parks under producer's task chain.
    PTO_DEVICE_FUNC static void insert(__gm__ DistTensorMap &self, __gm__ const Tensor &t, int32_t producer) {
        uint64_t addr, lo, hi;
        byte_range(t, addr, lo, hi);
        const int32_t s = alloc_slot(self);
        if (s < 0) return;  // pool full within the live window (should not happen)
        const uint32_t b = hash(addr);
        __gm__ MapEntry &e = self.entries[s];
        e.buf_addr = addr;
        e.lo = lo;
        e.hi = hi;
        e.producer = producer;
        e.bucket = static_cast<int32_t>(b);
        // Insert at bucket head.
        e.prev_in_bucket = -1;
        e.next_in_bucket = self.buckets[b];
        if (self.buckets[b] >= 0) self.entries[self.buckets[b]].prev_in_bucket = s;
        self.buckets[b] = s;
        // Insert at task-chain head.
        const int32_t slot = producer & kTaskWindowMask;
        e.next_in_task = self.task_heads[slot];
        self.task_heads[slot] = s;
    }

#if defined(__CCE_AICORE__)
    // Non-__gm__ overload for the CCEC path. TensorRef::ref() now returns a
    // default-address-space reference (orch stack local), so args.tensor(i).ref()
    // and result.get_ref(...) — where result is a stack-local TaskOutputTensors
    // — bind here rather than the __gm__ version. Byte-range extraction reads
    // scalar fields only; the Tensor is physically on the AICore kernel stack
    // which is GM-backed at launch, so pointer-identity math over `addr` still
    // matches the on-core owner's earlier __gm__ insert.
    PTO_DEVICE_FUNC static void insert(__gm__ DistTensorMap &self, const Tensor &t, int32_t producer) {
        uint64_t addr, lo, hi;
        byte_range(t, addr, lo, hi);
        const int32_t s = alloc_slot(self);
        if (s < 0) return;
        const uint32_t b = hash(addr);
        __gm__ MapEntry &e = self.entries[s];
        e.buf_addr = addr;
        e.lo = lo;
        e.hi = hi;
        e.producer = producer;
        e.bucket = static_cast<int32_t>(b);
        e.prev_in_bucket = -1;
        e.next_in_bucket = self.buckets[b];
        if (self.buckets[b] >= 0) self.entries[self.buckets[b]].prev_in_bucket = s;
        self.buckets[b] = s;
        const int32_t slot = producer & kTaskWindowMask;
        e.next_in_task = self.task_heads[slot];
        self.task_heads[slot] = s;
    }
#endif

    // Most-recent producer whose region overlaps `t`, or -1 if none. Entries
    // below alive_floor are treated as already retired (skipped — defensive,
    // since cleanup has usually freed them already).
    // All Tensor lvalues that reach the dist engine live in GM: orch args
    // that reference host-uploaded Tensors, outpool[] entries the engine
    // itself materializes, block.won deposits. So byte_range / lookup /
    // insert all take a __gm__ const Tensor&. The __gm__ macro collapses
    // to empty on sim / AICPU, so the same signatures compile identically
    // in the host toolchain (host callers see plain `const Tensor&`).
    PTO_DEVICE_FUNC static int32_t lookup(__gm__ const DistTensorMap &self, __gm__ const Tensor &t) {
        uint64_t addr, lo, hi;
        byte_range(t, addr, lo, hi);
        int32_t best = -1;
        for (int32_t cur = self.buckets[hash(addr)]; cur >= 0; cur = self.entries[cur].next_in_bucket) {
            __gm__ const MapEntry &e = self.entries[cur];
            if (e.producer < self.alive_floor) continue;
            if (e.buf_addr == addr && lo < e.hi && e.lo < hi) {
                if (e.producer > best) best = e.producer;
            }
        }
        return best;
    }

#if defined(__CCE_AICORE__)
    // Non-__gm__ overload — same rationale as insert() above.
    PTO_DEVICE_FUNC static int32_t lookup(__gm__ const DistTensorMap &self, const Tensor &t) {
        uint64_t addr, lo, hi;
        byte_range(t, addr, lo, hi);
        int32_t best = -1;
        for (int32_t cur = self.buckets[hash(addr)]; cur >= 0; cur = self.entries[cur].next_in_bucket) {
            __gm__ const MapEntry &e = self.entries[cur];
            if (e.producer < self.alive_floor) continue;
            if (e.buf_addr == addr && lo < e.hi && e.lo < hi) {
                if (e.producer > best) best = e.producer;
            }
        }
        return best;
    }
#endif
};

// -----------------------------------------------------------------------------
// A private-ring slot: a fully materialized, self-contained task this core owns
// and will execute itself. Holds its own copy of the argument Tensors so it can
// be executed at any later point (deferred past further orchestration).
// -----------------------------------------------------------------------------
// One traced span on a core's timeline, recorded only when swimlane tracing is
// on. `phase` distinguishes the orchestration stage so the exported lane shows
// not just kernel execution but also the work between kernels (alloc, claim/
// build, deposit drains). Laid out in the Chrome trace by physical block (pid)
// and lane (tid).
#if DIST_TRACE_ENABLED
enum class TracePhase : int32_t {
    Kernel = 0,    // incore kernel execution (or busy-wait replay)
    Alloc = 1,     // dist_alloc_tensors body (materialize + reclaim back-pressure)
    Build = 2,     // winner-only: fan-in resolution + built[] assembly (up to back-pressure)
    DrainWon = 3,  // drain_block_won pulled+built a follower deposit
    Replay = 4,    // submit replayed but claim LOST (per-core map/heap bookkeeping only)
    RingBp = 5,    // winner spun on ring/heap back-pressure (waiting for a free slot / reclaim)
    EfDrain = 6,   // execute-first drain at submit entry (deposits + ready owned tasks)
    Commit = 7,    // winner-only: alloc ring/won slot + build_ring_slot (publish the task)
};

struct TraceEvent {
    int32_t task_id;
    int32_t func_id;  // kernel id (e.g. 0=GEMM, 1=ADD); -1 if unknown
    int32_t lane;     // AIC=0 / AIV0=1 / AIV1=2
    uint8_t multicore;
    TracePhase phase;
    // Raw nanosecond timestamps — NO unit conversion on the hot path. The dump
    // stage divides by 1000 to emit microseconds (the swimlane unit).
    uint64_t ts_ns;   // start, ns from g_trace_epoch (wall clock)
    uint64_t dur_ns;  // span duration, ns (wall clock)
    // CPU time this thread actually accrued during the span (CLOCK_THREAD_CPUTIME_ID).
    // On an oversubscribed host dur_ns inflates while the thread is descheduled;
    // cpu_ns does not, so a large dur_ns with small cpu_ns == "swapped out, not work".
    // Only meaningful for non-kernel overhead spans (kernel spans set it to dur_ns).
    uint64_t cpu_ns;
};
#endif  // DIST_TRACE_ENABLED

struct RingSlot {
    bool occupied;
    // A slot can be reserved (occupied=true) before it is fully populated: the
    // submit winner grabs a slot up front so concurrent drains do not reuse it,
    // then may spin in block.won back-pressure (which itself drains Phase B)
    // before calling build_ring_slot. `built` gates execution so Phase B never
    // (re)runs a reserved-but-unbuilt slot still holding a prior occupant's
    // task_id/fanin/won linkage. build_ring_slot sets it; execute_slot clears it.
    bool built;
    int32_t task_id;
    int32_t func_id;  // kernel id of this slot's lane (swimlane label); -1 if none
    uint64_t function_bin_addr;

    int32_t tensor_count;
    int32_t scalar_count;
    Tensor tensors[MAX_TENSOR_ARGS];
    uint64_t scalars[MAX_SCALAR_ARGS];

    uint64_t args[PTO2_DISPATCH_MAX_ARGS];
    LocalContext local_ctx;
    GlobalContext global_ctx;

    int32_t fanin[kMaxFanin];
    int32_t fanin_count;

    // Multi-core (MIX / 2V) linkage. When is_multicore, the completion flag for
    // task_id is owned jointly: each co-owner decrements block.won[won_slot].remaining
    // after executing its own subtask, and the one driving it to zero publishes
    // the single global task_completed_flag. Single-core tasks set the flag directly.
    bool is_multicore;
    int32_t won_block;
    int32_t won_slot;
};

// -----------------------------------------------------------------------------
// block.won — the id-keyed anchor→follower deposit table (block-shared, §3.1).
// One BlockWon per physical block (1 AIC + 2 AIV). The anchor that wins a
// multi-core task builds its OWN physical-lane subtask into its private ring and
// deposits the remaining active-lane subtasks here; followers asynchronously
// drain the entry addressed to their physical lane (no blocking, no per-walk
// wait). Keyed by task id via per-slot task_id so concurrent multi-core tasks of
// one block never alias. `remaining` = popcount(active_mask) drives the single
// completion flag (§3.1). Lane index uses PTO2SubtaskSlot (AIC=0/AIV0=1/AIV1=2).
// -----------------------------------------------------------------------------
struct BuiltSubtask {
    bool present;
    int32_t func_id;  // kernel id of this lane's subtask (swimlane label); -1 if none
    uint64_t function_bin_addr;
    int32_t tensor_count;
    int32_t scalar_count;
    Tensor tensors[MAX_TENSOR_ARGS];
    uint64_t scalars[MAX_SCALAR_ARGS];
    int32_t fanin[kMaxFanin];
    int32_t fanin_count;
    int32_t sub_block_id;
};

struct WonSlot {
    // volatile T + __atomic_*: std::atomic<T>'s members are host-tagged under
    // CCEC and its object is not a POD, so it can't sit in a struct the AICore
    // touches. We reproduce the same acq/rel/relaxed semantics through the
    // atom_* wrappers at the top of the file.
    volatile int32_t state;                        // 0=free, 1=published, 2=reserving
    int32_t task_id;
    // int64_t: CCEC backend refuses 32-bit atomic add on GM (see wrappers
    // preamble). fetch_sub happens on this field, so bump to 64-bit.
    volatile int64_t remaining;                    // co-owners (incl. anchor) left to finish
    volatile int32_t drained[PTO2_SUBTASK_SLOT_COUNT];  // 0/1 per follower lane
    BuiltSubtask lane[PTO2_SUBTASK_SLOT_COUNT];    // deposited follower subtasks
};

struct BlockWon {
    WonSlot slots[kPrivateSlots];
    // Monotone "has any anchor ever published a deposit into this block?" flag.
    // Lets follower drains short-circuit the per-slot scan for workloads with no
    // multi-core (e.g. 2V) tasks — the common case (bgemm is all single-core), so
    // every AIV core skips a 4-slot won-scan on every submit. Never reset within a
    // session; once true the scan path is taken (those workloads have real work).
    volatile int32_t any_pub;
};

enum LaneId : int32_t { LANE_AIC = 0, LANE_AIV0 = 1, LANE_AIV1 = 2, LANE_NONE = -1 };

#if DIST_TRACE_ENABLED
// Swimlane tracing globals. Defined here (before DistCore) so DistCore::reset can
// see g_trace_reserve; g_trace_on / g_trace_epoch_ns sit alongside for one place.
//   g_trace_on      — set from PTO_DIST_SWIMLANE at register time; gates capture.
//   g_trace_epoch_ns — run-start epoch so every core's span ts is relative to it.
//   g_trace_reserve — per-core span reserve: 0 when off (reset never reserves, so
//     a normal run pays nothing), else a generous upper bound on spans/core so
//     push_back never reallocs mid-run (stable heap layout).
bool g_trace_on = false;
uint64_t g_trace_epoch_ns = 0;
int32_t g_trace_reserve = 0;
#endif

struct CoreLayout {
    int32_t block_id;  // physical block index
    int32_t lane;      // LaneId of this core within its block
};

// -----------------------------------------------------------------------------
// Per-core engine state (the SPMD worker context).
// -----------------------------------------------------------------------------
struct DistCore {
    CoreType role;
    int32_t core_idx;  // index into g_dist.cores[] (for trace ownership)
    int32_t block_id;  // physical block this core belongs to
    int32_t lane;      // LaneId within the block (AIC / AIV0 / AIV1)
    int32_t sub_block_id;
    int32_t local_index;  // next task id this core will see (== tasks replayed)
    uint64_t heap_next;   // deterministic GM output-heap bump cursor (bytes)

    DistTensorMap map;

    RingSlot slots[kPrivateSlots];
    int32_t occupied_count;
    int32_t owned_total;  // tasks this core claimed+executed (debug)

    Tensor outpool[kOutPoolSlots];
    int32_t outpool_head;

#if DIST_TRACE_ENABLED
    // Per-core swimlane events (only populated when tracing is on). Owned solely
    // by this core's worker thread, so push_back is lock-free.
    std::vector<TraceEvent> trace;

    // Running-cursor timestamps for lap-style tracing (see trace_lap). Each span is
    // [trace_last_ns, now); after recording, the cursor advances to now, so the next
    // span abuts this one with zero gap — the whole submit flow (incl. the orch
    // round-trip between two submits) is covered by exactly one span each, no code
    // path left un-timed. Reset at replay entry; wall + this-thread CPU clocks.
    uint64_t trace_last_ns;
    uint64_t trace_last_cpu;

    // Per-core static dependency edges (tracing only): one per fan-in resolved at
    // build time — {consumer_task, producer_task}. Dumped as Chrome-trace flow
    // events (producer's span -> consumer's span) so the swimlane shows the full
    // dependency graph; following the arrows hop-by-hop walks the chain "what is
    // this task waiting on, and what is THAT waiting on". Recorded by whichever
    // core builds the task, so every executed task contributes its in-edges.
    struct DepEdge {
        int32_t consumer_task;
        int32_t producer_task;
    };
    std::vector<DepEdge> dep_edges;

    // Per-core SLOT-RELEASE edges (tracing only): why a ringbp actually stalls.
    // When task N's owner enters the ring back-pressure, it is waiting not on N's
    // data producers but on the tasks ALREADY occupying its private ring to
    // execute (free a slot). Snapshot those occupants ({waiter=N, occupant}).
    // Dumped as flow events occupant-kernel -> ringbp: the occupant's execution is
    // the release event that ends the wait. Chains with dep_edges: ringbp -> its
    // ring occupant (slot edge) -> that occupant's data producers (dep edges).
    std::vector<DepEdge> slot_edges;
#endif  // DIST_TRACE_ENABLED

    // Same rationale as DistTensorMap: CCEC cannot qualify the `this` of a
    // non-static member function with __gm__, so DistCore state mutation goes
    // through a static entry taking an explicit `self` reference. sim's
    // __gm__-empty expansion collapses this to a plain reference. Trace fields
    // (`trace`, `dep_edges`, `slot_edges`) are std::vectors used only under
    // DIST_HOST_ONLY (sim / AICPU), so the trace-reset block cannot run on
    // CCEC and does not need __gm__-aware member calls.
    PTO_DEVICE_FUNC static void reset(__gm__ DistCore &self, CoreType r, int32_t block, int32_t lane_id) {
        self.role = r;
        self.block_id = block;
        self.lane = lane_id;
        self.sub_block_id = (lane_id == LANE_AIV1) ? 1 : 0;
        self.local_index = 0;
        self.heap_next = 0;
        DistTensorMap::reset(self.map);
        self.occupied_count = 0;
        self.owned_total = 0;
        self.outpool_head = 0;
        for (int32_t i = 0; i < kPrivateSlots; i++) {
            self.slots[i].occupied = false;
            self.slots[i].built = false;
        }
#if DIST_TRACE_ENABLED
        self.trace_last_ns = 0;
        self.trace_last_cpu = 0;
        self.trace.clear();
        // Pre-size the trace vector only when tracing is on (see g_trace_on),
        // so push_back never reallocs mid-run (a realloc would perturb the heap
        // layout — exactly the kind of disturbance that historically interacted
        // badly with the sim; keep it stable). Costs nothing on a normal run.
        if (g_trace_reserve > 0) self.trace.reserve(g_trace_reserve);
        self.dep_edges.clear();
        if (g_trace_reserve > 0) self.dep_edges.reserve(g_trace_reserve);
        self.slot_edges.clear();
        if (g_trace_reserve > 0) self.slot_edges.reserve(g_trace_reserve);
#endif  // DIST_TRACE_ENABLED
    }
};

// -----------------------------------------------------------------------------
// Cursor sharding (docs §6.6). Each per-anchor-type claim cursor is split into
// kCursorShards independent sub-cursors; task id N claims on shard (N %
// kCursorShards). The shard is a pure function of N (identical on every core, no
// worker partitioning), so the claim semantics are byte-for-byte equivalent to a
// single cursor (exactly one owner per task, every core eligible) — sharding
// ONLY spreads the CAS traffic across kCursorShards cache lines, cutting the
// false-sharing / coherence contention that dominated us/task at high core
// counts (§6.5). Each sub-cursor is padded to its own cache line so adjacent
// shards never share a line; all entries init to -1 (no id claimed yet).
constexpr int32_t kCursorShards = 4;
constexpr size_t kCacheLine = 64;

struct alignas(kCacheLine) PaddedCursor {
    volatile int32_t v;
    uint8_t pad[kCacheLine - sizeof(int32_t)];
};

// -----------------------------------------------------------------------------
// Global engine state (shared by all worker threads in this process). Cursors +
// flags live here rather than in GM because in sim every core is a host thread
// in one address space; the GM output heap below is a real shared buffer.
// -----------------------------------------------------------------------------
struct DistGlobal {
    PaddedCursor cube_cursor[kCursorShards];    // highest claimed AIC-anchored id, per shard
    PaddedCursor vector_cursor[kCursorShards];  // highest claimed AIV-only id, per shard
    PaddedCursor alloc_cursor[kCursorShards];   // highest claimed kernel-less alloc id, per shard
    volatile uint8_t flags[kFlagCap];           // completion-flag ring (1 == task done)

    // M4 reclamation (§9.5/§11.4). `frontier` (F) is the global continuous
    // completion frontier — the largest prefix s.t. every task id <= F is done;
    // advanced cooperatively (CAS) by whichever core sets the flag that extends
    // the prefix. `R = frontier - H` is the reclaim frontier. `vend[N]` is the
    // cumulative virtual heap bytes through task N (deterministic & identical on
    // every core), so any core can compute the live byte window [vend[R], top).
    volatile int32_t frontier;
    int32_t H;
    volatile uint64_t vend[kFlagCap];

    uint8_t *heap_base;
    size_t heap_size;  // == bounded ring size

    const L2TaskArgs *orch_args;
    PTO2Runtime *rt;
    Runtime *runtime;  // outer Runtime (for kernel-address resolution + done_count)

    volatile int32_t fatal;

    // Physical-block topology (1 AIC + 2 AIV per block), derived once at register
    // time from Runtime::workers[].core_type, identical to the centralized
    // scheduler's cluster discovery (AIC core b pairs with the 2b-th / (2b+1)-th
    // AIV cores in worker-index order).
    int32_t num_workers;
    int32_t num_blocks;
    CoreLayout layout[RUNTIME_MAX_WORKER];
    BlockWon blocks[RUNTIME_MAX_WORKER];  // indexed by block_id (<= num AIC)

    // Global "all cores finished orchestration replay" counter. A follower must
    // not conclude "no more pushes are coming for my lane" until every core has
    // finished replaying the submit stream (§7 tail-idle). int64_t because CCEC
    // rejects 32-bit atomic add on GM addresses (see wrappers preamble).
    volatile int64_t replay_done;

    // Startup barrier: every worker thread bumps this on entry and spins until it
    // reaches num_workers before beginning replay. In sim each "core" is a host
    // pthread that the OS schedules in one at a time (hundreds of µs apart on a
    // busy box), so without this the first-claimed tasks start executing while
    // later cores have not even been scheduled — the swimlane shows a long
    // cold-start stagger that is host-scheduling noise, not engine behavior.
    // Aligning the start makes the trace reflect steady-state contention.
    // int64_t for the same GM-atomic-add reason as replay_done above.
    volatile int64_t started_count;

    DistCore cores[RUNTIME_MAX_WORKER];
};

// g_dist / g_self storage. The AICPU build owns the BSS DistGlobal and
// initializes it in dist_engine_register. AICore builds attach to the published
// Runtime::dist.shared_addr at dist_core_main entry: CCEC uses GM pointers,
// while CPU sim uses an ordinary process pointer into the AICPU DSO's BSS.
// The `g_dist` macro keeps hot-path code identical.
#if defined(__CCE_AICORE__)
[[block_local]] static __gm__ DistGlobal *g_dist_ptr;
[[block_local]] static __gm__ DistCore *g_self;
[[block_local]] static __gm__ Runtime *g_ccec_runtime;
[[block_local]] static int32_t g_ccec_core_idx;
[[block_local]] static int32_t g_ccec_core_type;
[[block_local]] static int32_t g_ccec_aic_count;
[[block_local]] static int32_t g_ccec_aiv_count;
[[block_local]] static int32_t g_ccec_ordinal;
[[block_local]] static bool g_ccec_valid_worker;
[[block_local]] static int32_t g_ccec_local_index;
[[block_local]] static uint64_t g_ccec_heap_next;
[[block_local]] static int32_t g_ccec_outpool_head;
[[block_local]] static int32_t g_ccec_map_count;
#define g_dist (*g_dist_ptr)
#elif defined(__CPU_SIM)
static DistGlobal *g_dist_ptr = nullptr;
thread_local DistCore *g_self = nullptr;
#define g_dist (*g_dist_ptr)
#else
DistGlobal g_dist;
thread_local DistCore *g_self = nullptr;
#endif

#if DIST_SIM_HOST_CLOCK
// Orchestration/scheduling overhead isolation (set PTO_DIST_SKIP_EXEC=1). When
// on, execute_slot skips the actual incore kernel call — every (sub)task is
// treated as 0-cost and "completes" instantly — while ALL ownership/completion
// bookkeeping runs unchanged, so the loop terminates identically. This lets a
// benchmark measure the pure cost of on-core orchestration + claim race +
// scheduling, independent of kernel work. Outputs are NOT computed (run with
// golden checks disabled). See examples/.../runtime_overhead_test.
bool g_skip_exec = false;

inline uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
}
#endif  // DIST_SIM_HOST_CLOCK

#if DIST_TRACE_ENABLED
// Per-thread CPU time (excludes time the thread spends descheduled). Used only by
// the swimlane to tell genuine work from host-oversubscription stalls, so it lives
// under DIST_TRACE_ENABLED (not the sim-clock gate — busy-wait never needs it).
inline uint64_t thread_cpu_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

// Snapshot the clock only when tracing is on (callers pass the result as a span
// start). Returns 0 otherwise so the matching trace_overhead() is a no-op.
inline uint64_t trace_now() { return g_trace_on ? now_ns() : 0; }
inline uint64_t trace_now_cpu() { return g_trace_on ? thread_cpu_ns() : 0; }

// Record a non-kernel overhead span [t0_ns, now) on this core's lane. Stores RAW
// nanoseconds (no unit conversion on the hot path — the dump stage divides by
// 1000). cpu_ns is this thread's CPU time over the span (small cpu with large dur
// == descheduled, not work). No-op unless tracing is on.
inline void trace_overhead_impl(
    DistCore *self, int32_t task_id, int32_t func_id, TracePhase phase, uint64_t t0_ns, uint64_t t0_cpu
) {
    if (!g_trace_on) return;
    const uint64_t t1 = now_ns();
    const uint64_t c1 = thread_cpu_ns();
    self->trace.push_back(
        TraceEvent{
            task_id, func_id, self->lane, /*multicore=*/0, phase, t0_ns - g_trace_epoch_ns, t1 - t0_ns, c1 - t0_cpu
        }
    );
}

// Reset the lap cursor to "now" — call once at replay entry so the first lap span
// measures from a well-defined origin (not from an uninitialized cursor).
inline void trace_lap_reset_impl(DistCore *self) {
    if (!g_trace_on) return;
    self->trace_last_ns = now_ns();
    self->trace_last_cpu = thread_cpu_ns();
}

// Lap-style span: record [trace_last_ns, now) then advance the cursor to now, so
// the next lap continues seamlessly from here (same idiom as pto_orchestrator's
// CYCLE_COUNT_LAP: acc += t1 - t0; t0 = t1). Every code path between two laps is
// attributed to exactly one span — no gaps, no double-counting. Stores raw ns.
inline void trace_lap_impl(DistCore *self, int32_t task_id, int32_t func_id, TracePhase phase) {
    if (!g_trace_on) return;
    const uint64_t t1 = now_ns();
    const uint64_t c1 = thread_cpu_ns();
    self->trace.push_back(
        TraceEvent{
            task_id, func_id, self->lane, /*multicore=*/0, phase, self->trace_last_ns - g_trace_epoch_ns,
            t1 - self->trace_last_ns, c1 - self->trace_last_cpu
        }
    );
    self->trace_last_ns = t1;
    self->trace_last_cpu = c1;
}

// Trace call-site macros forward to the _impl inlines above; the #else branch below
// expands them to nothing — so call sites need no #if, and the phase enum /
// TraceEvent need not even exist when off (the preprocessor eats the whole argument
// list, TracePhase::X included). Same idiom as pto_orchestrator's CYCLE_COUNT_LAP.
#define TRACE_LAP(self, task_id, func_id, phase) trace_lap_impl((self), (task_id), (func_id), (phase))
#define TRACE_LAP_RESET(self) trace_lap_reset_impl((self))
#define TRACE_OVERHEAD(self, task_id, func_id, phase, t0_ns, t0_cpu) \
    trace_overhead_impl((self), (task_id), (func_id), (phase), (t0_ns), (t0_cpu))
#else  // !DIST_TRACE_ENABLED — tracing compiled out; call sites become no-ops.
#define TRACE_LAP(self, task_id, func_id, phase) ((void)0)
#define TRACE_LAP_RESET(self) ((void)0)
#define TRACE_OVERHEAD(self, task_id, func_id, phase, t0_ns, t0_cpu) ((void)0)
#endif  // DIST_TRACE_ENABLED

// Opt-in per-core tracing (set PTO_DIST_TRACE=1). Off by default so a passing
// run is quiet; fatal/error/heap-exhaustion diagnostics are always emitted.
// Host-only: relies on getenv; every caller sits under DIST_HOST_ONLY.
#if DIST_HOST_ONLY
inline bool dist_trace() {
    static const bool on = (getenv("PTO_DIST_TRACE") != nullptr);
    return on;
}
#endif

// -----------------------------------------------------------------------------
// Fatal / claim / execution helpers
// -----------------------------------------------------------------------------
PTO_DEVICE_FUNC inline bool fatal_set() { return atom_load(g_dist.fatal, __ATOMIC_ACQUIRE) != 0; }
PTO_DEVICE_FUNC inline void set_fatal() { atom_store(g_dist.fatal, 1, __ATOMIC_RELEASE); }

// Env-gated stall watchdog (set PTO_DIST_WATCHDOG=<seconds>, default off). Called
// from inside the engine's spin loops on a worker thread (so fprintf is safe,
// unlike a signal handler). On the first call it records a start time; if a loop
// keeps spinning past the budget the engine is presumed deadlocked, so it dumps
// the full state once and sets fatal to unwind every core for a fast, diagnosed
// failure instead of an indefinite hang. CCEC/onboard has no getenv/chrono/
// fprintf, so the function collapses to a no-op there — call sites remain
// unchanged and pay a single unused-argument tag.
PTO_DEVICE_FUNC inline void watchdog([[maybe_unused]] uint64_t &start_ns) {
#if DIST_HOST_ONLY
    static const long budget_s = []() -> long {
        const char *e = getenv("PTO_DIST_WATCHDOG");
        return e ? atol(e) : 0;
    }();
    if (budget_s <= 0) return;
    const uint64_t now = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
    if (start_ns == 0) {
        start_ns = now;
        return;
    }
    if (now - start_ns > static_cast<uint64_t>(budget_s) * 1000000000ull) {
        static std::atomic<int32_t> dumped{0};
        int32_t exp = 0;
        if (dumped.compare_exchange_strong(exp, 1, std::memory_order_acq_rel)) {
            fprintf(stderr, "[dist_engine] WATCHDOG fired after %lds — presumed deadlock, dumping state\n", budget_s);
            dist_dump_state(0);
        }
        set_fatal();
    }
#endif
}

// CAS-loop fetch_max (§11.1): returns true (WON) iff this core advanced the
// cursor to N. No hardware fetch_max on the target, so this is the equivalent
// acq-rel CAS retry. Monotonic: each task id is claimed by exactly one core and
// no id is skipped within a cursor's subsequence.
PTO_DEVICE_FUNC bool claim(__gm__ volatile int32_t &cursor, int32_t N) {
    int32_t c = atom_load(cursor, __ATOMIC_ACQUIRE);
    while (true) {
        if (N <= c) return false;
        if (atom_cas_weak(cursor, c, N, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) return true;
    }
}

// Cooperatively advance the global completion frontier F (§11.4): after any core
// publishes flag(N), the contiguous-done prefix may have grown, so any core walks
// F forward while flag(F+1) is set. Lock-free; the CAS makes exactly one core win
// each step and the cost is amortized across all cores.
PTO_DEVICE_FUNC void advance_frontier() {
    int32_t f = atom_load(g_dist.frontier, __ATOMIC_ACQUIRE);
    while (true) {
        const int32_t next = f + 1;
        if (next >= kFlagCap) break;
        if (atom_load(g_dist.flags[next & (kFlagCap - 1)], __ATOMIC_ACQUIRE) == 0) break;
        if (atom_cas_weak(g_dist.frontier, f, next, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            f = next;
        }
        // On CAS failure f was reloaded with the current value; retry.
    }
}

// Resolve a kernel id to its executable address (CoreCallable::resolved_addr()).
// Reads Runtime::func_id_to_addr_ directly (public POD array) rather than
// calling the get_function_bin_addr() member so this compiles into libaicore
// too — the AICore .so does not link against libaicpu, so a member-function
// dispatch would fail with an unresolved symbol at dlopen.
PTO_DEVICE_FUNC uint64_t resolve_kernel_addr(Runtime *runtime, int32_t kernel_id) {
    if (kernel_id == INVALID_KERNEL_ID) return 0;
    if (kernel_id < 0 || kernel_id >= RUNTIME_MAX_FUNC_ID) return 0;
    uint64_t callable_addr = runtime->func_id_to_addr_[kernel_id];
    if (callable_addr == 0) return 0;
    const CoreCallable *callable = reinterpret_cast<const CoreCallable *>(callable_addr);
    return callable->resolved_addr();
}

// Execute one owned task, then publish its completion flag (release). In sim all
// cores share the address space, so the release/acquire pair is the visibility
// barrier between the kernel's output writes and a consumer's input reads.
PTO_DEVICE_FUNC void execute_slot([[maybe_unused]] __gm__ DistCore *self, __gm__ RingSlot &s) {
    // Kernel dispatch signature: on-device kernels take a __gm__ int64_t*
    // (their args live in GM alongside the RingSlot); sim host kernels ignore
    // the address space (empty __gm__ macro). One typedef covers both targets.
    typedef void (*KernelFn)(__gm__ int64_t *);
#if DIST_SIM_HOST_CLOCK
    // Sim-only trace-driven replay (CallConfig::use_example_exec_time): when the
    // host filled example_exec_time_ns_[func_id] > 0 for this func, "execute" it
    // by busy-waiting that many nanoseconds instead of calling the real kernel,
    // so a fast sim run reflects measured on-hardware kernel durations. 320 host
    // cores >> 72 workers, so the spin does not contend; funcs left at 0 fall
    // through to the real call below. See Runtime::example_exec_time_ns_.
    const Runtime *rt = g_dist.runtime;
    const int32_t sim_ns =
        (rt != nullptr && rt->use_example_exec_time_ && s.func_id >= 0 && s.func_id < RUNTIME_MAX_FUNC_ID) ?
            rt->example_exec_time_ns_[s.func_id] :
            0;
    if (sim_ns > 0) {
        const uint64_t t0 = now_ns();
        const uint64_t target = t0 + static_cast<uint64_t>(sim_ns);
        while (now_ns() < target) { /* spin: emulate kernel busy time */
        }
#if DIST_TRACE_ENABLED
        if (g_trace_on) {
            self->trace.push_back(
                TraceEvent{
                    s.task_id, s.func_id, self->lane, static_cast<uint8_t>(s.is_multicore ? 1 : 0), TracePhase::Kernel,
                    t0 - g_trace_epoch_ns, static_cast<uint64_t>(sim_ns), static_cast<uint64_t>(sim_ns)
                }
            );
        }
#endif
    } else if (s.function_bin_addr != 0 && !g_skip_exec) {
        // PTO_DIST_SKIP_EXEC: treat the incore task as 0-cost — skip the kernel call
        // but keep every flag/frontier/slot update below so termination is identical.
        KernelFn fn = reinterpret_cast<KernelFn>(s.function_bin_addr);
#if DIST_TRACE_ENABLED
        if (g_trace_on) {
            const uint64_t t0 = now_ns();
            fn(reinterpret_cast<__gm__ int64_t *>(s.args));
            const uint64_t t1 = now_ns();
            self->trace.push_back(
                TraceEvent{
                    s.task_id, s.func_id, self->lane, static_cast<uint8_t>(s.is_multicore ? 1 : 0), TracePhase::Kernel,
                    t0 - g_trace_epoch_ns, t1 - t0, t1 - t0
                }
            );
        } else {
            fn(reinterpret_cast<__gm__ int64_t *>(s.args));
        }
#else
        fn(reinterpret_cast<__gm__ int64_t *>(s.args));
#endif
    }
#else   // !DIST_SIM_HOST_CLOCK — AICore/CCEC: no host clock, no busy-wait emulation.
    if (s.function_bin_addr != 0) {
        KernelFn fn = reinterpret_cast<KernelFn>(s.function_bin_addr);
        fn(reinterpret_cast<__gm__ int64_t *>(s.args));
    }
#endif  // DIST_SIM_HOST_CLOCK
    if (s.is_multicore) {
        // Joint ownership: the co-owner that drives remaining to zero (the last
        // subtask to finish) publishes the single global completion flag (§3.1),
        // then frees the block.won entry for reuse.
        __gm__ WonSlot &w = g_dist.blocks[s.won_block].slots[s.won_slot];
        if (atom_fetch_sub<int64_t>(w.remaining, 1, __ATOMIC_ACQ_REL) == 1) {
            atom_store(g_dist.flags[s.task_id & (kFlagCap - 1)], 1, __ATOMIC_RELEASE);
            atom_store(w.state, 0, __ATOMIC_RELEASE);  // recycle the id-keyed slot
            advance_frontier();
        }
    } else {
        atom_store(g_dist.flags[s.task_id & (kFlagCap - 1)], 1, __ATOMIC_RELEASE);
        advance_frontier();
    }
    s.built = false;
    s.occupied = false;
}

// Phase B: execute every ready owned task in the private ring. A task is ready
// once all its fan-in producers have set their completion flag (acquire).
// Returns the number of slots freed this pass.
PTO_DEVICE_FUNC int32_t drain_phase_b(__gm__ DistCore *self) {
    // Fast path: an empty private ring has nothing to drain. Skips the per-slot
    // scan on every submit point (called twice per task, on every core) when the
    // ring is empty — the common case for fine-grained / skip-exec workloads.
    // Behavior-identical: the loop below is a no-op when occupied_count == 0.
    if (self->occupied_count == 0) return 0;
    int32_t freed = 0;
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        __gm__ RingSlot &s = self->slots[i];
        if (!s.occupied || !s.built) continue;  // skip reserved-but-unbuilt slots
        bool ready = true;
        for (int32_t f = 0; f < s.fanin_count; f++) {
            if (atom_load(g_dist.flags[s.fanin[f] & (kFlagCap - 1)], __ATOMIC_ACQUIRE) == 0) {
                ready = false;
                break;
            }
        }
        if (!ready) continue;
        execute_slot(self, s);
        self->occupied_count--;
        freed++;
    }
    return freed;
}

PTO_DEVICE_FUNC int32_t alloc_ring_slot(__gm__ DistCore *self) {
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        if (!self->slots[i].occupied) return i;
    }
    return -1;
}

// Kernel id for a physical lane (AIC/AIV0/AIV1) of a MixedKernels.
PTO_DEVICE_FUNC inline int32_t kernel_id_for_lane(const MixedKernels &mixed, int32_t lane) {
    switch (lane) {
    case LANE_AIC:
        return mixed.aic_kernel_id;
    case LANE_AIV0:
        return mixed.aiv0_kernel_id;
    case LANE_AIV1:
        return mixed.aiv1_kernel_id;
    default:
        return INVALID_KERNEL_ID;
    }
}

PTO_DEVICE_FUNC inline bool lane_active(const ActiveMask &M, int32_t lane) {
    return M.subtask_active(static_cast<PTO2SubtaskSlot>(lane));
}

// Materialize a private-ring slot from already-resolved components (shared by the
// owner build path and the follower drain path). `tensors`/`scalars` are copied
// in; args[] is (re)built to point at this slot's own copies so the slot is
// self-contained and executable at any later time.
// build_ring_slot has two callers with different address spaces on the
// input arrays: the winner branch of dist_submit_impl passes the LM-stack
// `built[]` / `scalars` / `fanin` local arrays, and drain_block_won passes
// `b.tensors[]` / `b.scalars[]` / `b.fanin[]` reached through a __gm__
// BuiltSubtask. Templating the three source-array pointer types lets both
// call sites bind naturally without a second overload — CCEC deduces the
// address-space qualifier on each pointer at instantiation time; sim just
// sees `const T *`.
template <typename TensorArrPtr, typename ScalarArrPtr, typename FaninArrPtr>
PTO_DEVICE_FUNC void build_ring_slot(
    __gm__ RingSlot &s, int32_t task_id, int32_t func_id, uint64_t fn_addr, TensorArrPtr tensors, int32_t tc,
    ScalarArrPtr scalars, int32_t sc, FaninArrPtr fanin, int32_t fc, int32_t sub_block_id, bool is_multicore,
    int32_t won_block, int32_t won_slot
) {
    s.occupied = true;
    s.task_id = task_id;
    s.func_id = func_id;
    s.function_bin_addr = fn_addr;
    s.built = true;  // fully populated below — now safe for Phase B to execute
    s.tensor_count = tc;
    s.scalar_count = sc;
    for (int32_t i = 0; i < tc; i++)
        Tensor::copy(s.tensors[i], tensors[i]);
    for (int32_t j = 0; j < sc; j++)
        s.scalars[j] = scalars[j];
    int32_t n = 0;
    for (int32_t i = 0; i < tc; i++)
        s.args[n++] = reinterpret_cast<uint64_t>(&s.tensors[i]);
    for (int32_t j = 0; j < sc; j++)
        s.args[n++] = s.scalars[j];
    s.local_ctx.s_block_idx = 0;
    s.local_ctx.s_block_num = 1;
    // Field-wise reset instead of `= AsyncCtx{}`: CCEC rejects overload
    // resolution across address spaces (assigning a host temporary into a
    // __gm__ struct member has no viable operator=). AsyncCtx is a POD, so
    // zeroing its fields directly gives the same semantics.
    s.local_ctx.async_ctx.completion_count = nullptr;
    s.local_ctx.async_ctx.completion_error_code = nullptr;
    s.local_ctx.async_ctx.completion_entries = nullptr;
    s.local_ctx.async_ctx.completion_capacity = 0;
    // Write the .raw uint64_t directly instead of `= PTO2TaskId::invalid()`:
    // CCEC has no viable operator= across address spaces for a __gm__ dest.
    s.local_ctx.async_ctx.task_token.raw = UINT64_MAX;
    s.global_ctx.sub_block_id = sub_block_id;
    s.args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&s.local_ctx);
    s.args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&s.global_ctx);
    s.fanin_count = fc;
    for (int32_t k = 0; k < fc; k++)
        s.fanin[k] = fanin[k];
    s.is_multicore = is_multicore;
    s.won_block = won_block;
    s.won_slot = won_slot;
}

// Reserve a free block.won slot in `block`. Returns slot index or -1 if full.
// 2V allows either AIV of the block to be an anchor, so allocation must be atomic.
PTO_DEVICE_FUNC int32_t alloc_won_slot(int32_t block) {
    __gm__ BlockWon &bw = g_dist.blocks[block];
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        int32_t exp = 0;
        if (atom_cas_strong(bw.slots[i].state, exp, 2, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
            return i;
        }
    }
    return -1;
}

// True if a published block.won deposit for this core's lane has not yet been
// taken — used by the termination check to avoid finishing before draining.
PTO_DEVICE_FUNC bool has_pending_won(__gm__ DistCore *self) {
    if (self->lane == LANE_AIC || self->lane == LANE_NONE) return false;
    __gm__ BlockWon &bw = g_dist.blocks[self->block_id];
    if (atom_load(bw.any_pub, __ATOMIC_ACQUIRE) == 0) return false;  // no deposit ever published
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        __gm__ WonSlot &w = bw.slots[i];
        if (atom_load(w.state, __ATOMIC_ACQUIRE) != 1) continue;
        if (w.lane[self->lane].present && atom_load(w.drained[self->lane], __ATOMIC_ACQUIRE) == 0) return true;
    }
    return false;
}

// Follower drain (§3.1, §6): pull every published block.won subtask addressed to
// this core's physical lane that we have not yet taken, building each into a free
// private-ring slot (back-pressure: stop when the ring is full). Non-blocking —
// if nothing is addressed to us we simply return.
PTO_DEVICE_FUNC void drain_block_won(__gm__ DistCore *self) {
    if (self->lane == LANE_AIC || self->lane == LANE_NONE) return;  // AIC is never a follower
    __gm__ BlockWon &bw = g_dist.blocks[self->block_id];
    // Fast path: if no anchor has ever published a deposit into this block, there
    // is nothing to drain — skip the per-slot scan on every submit (hot path).
    if (atom_load(bw.any_pub, __ATOMIC_ACQUIRE) == 0) return;
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        __gm__ WonSlot &w = bw.slots[i];
        if (atom_load(w.state, __ATOMIC_ACQUIRE) != 1) continue;
        if (!w.lane[self->lane].present) continue;
        int32_t exp = 0;
        if (!atom_cas_strong(w.drained[self->lane], exp, 1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
            continue;  // already taken by us on a prior pass
        int32_t si = alloc_ring_slot(self);
        if (si < 0) {
            // Ring full: hand the deposit back and let Phase B free a slot first.
            atom_store(w.drained[self->lane], 0, __ATOMIC_RELEASE);
            return;
        }
        __gm__ const BuiltSubtask &b = w.lane[self->lane];
#if DIST_TRACE_ENABLED
        const uint64_t t_won0 = trace_now();
        const uint64_t t_won0_cpu = trace_now_cpu();
#endif
        build_ring_slot(
            self->slots[si], w.task_id, b.func_id, b.function_bin_addr, b.tensors, b.tensor_count, b.scalars,
            b.scalar_count, b.fanin, b.fanin_count, b.sub_block_id, /*is_multicore=*/true, self->block_id, i
        );
        self->occupied_count++;
        self->owned_total++;
#if DIST_TRACE_ENABLED
        if (g_trace_on) {
            for (int32_t k = 0; k < b.fanin_count; k++)
                self->dep_edges.push_back({w.task_id, b.fanin[k]});
        }
        trace_overhead_impl(self, w.task_id, b.func_id, TracePhase::DrainWon, t_won0, t_won0_cpu);
#endif
    }
}

// -----------------------------------------------------------------------------
// Distributed submit op (replaces the centralized orchestrator submit).
//
// Every core runs this for every task (identical replay): materialize outputs
// at deterministic heap addresses, maintain the per-core producer map, then
// race to claim ownership. Only the winner builds the task into its private
// ring; losers return with map + outputs updated so downstream get_ref() and
// fan-in resolution stay consistent across cores.
// -----------------------------------------------------------------------------
}  // namespace

#if defined(__CCE_AICORE__)
namespace {
#endif
PTO_DEVICE_FUNC TaskOutputTensors dist_submit_impl(PTO2Runtime *, const MixedKernels &mixed, const L0TaskArgs &args) {
    __gm__ DistCore *self = g_self;
    if (self == nullptr) return TaskOutputTensors{};
    Runtime *runtime = g_dist.runtime;

    // EXECUTE-FIRST (docs §6 step 0+1, §6.1): before claiming this task, pull any
    // follower deposits and execute every ready owned task. This interleaves
    // execution with claiming so a fast core does not burst-claim a full ring of
    // consecutive tasks; while it executes a (long) task other cores advance the
    // cursor and claim subsequent ones. The deterministic replay below (id bump,
    // heap bump, map maintenance) is unaffected — draining only runs/flags tasks
    // this core already owns. Every core does this on every submit point.
    //
    // Reset the lap cursor at entry so the runtime's spans never absorb the orch
    // round-trip between two submits — that time is USER orchestration code, not
    // runtime work, and would bias EfDrain if counted here. It is left un-timed on
    // purpose (a deliberate gap between submits, not a runtime span).
    TRACE_LAP_RESET(self);
    if (!fatal_set()) {
        drain_block_won(self);
        drain_phase_b(self);
    }
    // Lap: the execute-first drain itself (deposits + ready owned kernels it ran).
    // Kernels show separately on the kernel sub-lane; this is the drain's own scan.
    TRACE_LAP(self, self->local_index, -1, TracePhase::EfDrain);

    const int32_t N = self->local_index++;
    const ActiveMask M = mixed.to_active_mask();
    const int32_t tc = args.tensor_count();
    if (N >= kFlagCap) {  // flag ring + vend[] are non-windowed; cap total tasks
        set_fatal();
        DIST_ERRF(
            "[dist_engine] task id %d exceeds kFlagCap %d (enlarge or window the flag/vend rings)\n", N,
            kFlagCap
        );
        return TaskOutputTensors{};
    }

    // (a) Deterministic GM output-heap allocation + materialization (§9.3, §11.4).
    // The virtual bump `heap_next` is unbounded and identical on every core; the
    // PHYSICAL address is (virtual mod ring). First sum this task's aligned output
    // bytes so we can keep the whole task within one ring lap: if it would straddle
    // the ring end, pad the virtual base up to the next ring boundary (deterministic
    // → every core agrees). A single task larger than the ring is unsatisfiable.
    const size_t ring = g_dist.heap_size;
    uint64_t total = 0;
    for (int32_t i = 0; i < tc; i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        total += PTO2_ALIGN_UP(TensorCreateInfo::buffer_size_bytes(args.tensor(i).create_info()), PTO2_PACKED_OUTPUT_ALIGN);
    }
    uint64_t task_base = PTO2_ALIGN_UP(self->heap_next, PTO2_PACKED_OUTPUT_ALIGN);
    if (total > 0 && g_dist.heap_base != nullptr) {
        if (total > ring) {
            set_fatal();
            DIST_ERRF(
                "[dist_engine] task %d outputs %llu B exceed heap ring %zu B (enlarge PTO_DIST_HEAP_MB)\n", N,
                (unsigned long long)total, ring
            );
            return TaskOutputTensors{};
        }
        if ((task_base % ring) + total > ring) {
            task_base = ((task_base / ring) + 1) * ring;  // skip the ring tail; start next lap
        }
    }
    uint64_t off = 0;
    TaskOutputTensors result;
    for (int32_t i = 0; i < tc; i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        // TensorRef::create_info() now returns a default-address-space ref
        // (see pto_types.h — orch is all stack-local under CCEC). The engine
        // still writes the derived Tensor into a GM outpool slot; the
        // cross-space transition happens inside init_tensor_from_create_info,
        // whose signature takes __gm__ dst + non-__gm__ src.
        const TensorCreateInfo &ci = args.tensor(i).create_info();
        const uint64_t logical = TensorCreateInfo::buffer_size_bytes(ci);
        const uint64_t sz = PTO2_ALIGN_UP(logical, PTO2_PACKED_OUTPUT_ALIGN);
        if (g_dist.heap_base == nullptr) {
            set_fatal();
            DIST_ERRF("[dist_engine] GM output heap not allocated at task %d\n", N);
            return result;
        }
        const uint64_t phys = (task_base + off) % ring;  // straddle-pad guarantees phys+logical <= ring
        __gm__ Tensor &slot_t = self->outpool[self->outpool_head];
        self->outpool_head = (self->outpool_head + 1) % kOutPoolSlots;
        init_tensor_from_create_info(slot_t, ci, g_dist.heap_base + phys, logical);
        result.materialize_output(slot_t);
        off += sz;
    }
    self->heap_next = task_base + off;
    // Publish cumulative virtual bytes through task N so any core can derive the
    // live window [vend[R], heap_next) for reclaim back-pressure. Deterministic, so
    // all cores store the same value (this core also reads its own writes for R<N).
    if (N >= 0 && N < kFlagCap) atom_store(g_dist.vend[N], self->heap_next, __ATOMIC_RELAXED);

    // Once fatal, stop claiming/executing but keep replaying the deterministic
    // allocation above so this task's `result` carries valid (materialized) output
    // refs — the orchestration may still call get_ref() on them. This degrades a
    // fatal (e.g. heap-too-small) into a clean wrong-answer failure + diagnostic
    // rather than an assertion crash mid-replay.
    if (fatal_set()) return result;

    // Retire producer-map entries that have left the H span (deterministic,
    // N-derived) before this task's lookups/inserts. Bounds chain length so
    // submit stays ~O(N) instead of O(N^2). See DistTensorMap.
    DistTensorMap::advance_retire(self->map, N, g_dist.H);

    // (b) Anchor type + claim race FIRST — resolved from the mask alone (no map
    // ops, no Tensor copies). Deciding the winner up front lets the ~2/3 of cores
    // that fail type_match / lose the race SKIP the fan-in lookup below; they only
    // still perform the unconditional output insert (so every core's duplicate
    // TensorMap stays identical — §4). Competition is by anchor TYPE (§2/§3.1):
    // cube tasks (any AIC subtask) contested by AIC cores; vector tasks (AIV-only,
    // incl. 2V) by AIV cores. The cursor CAS touches no map state, so doing it
    // before the insert below does not affect the deterministic map replay.
    const uint8_t cmask = M.core_mask();
    const int32_t pc = __builtin_popcount(cmask);
    const bool has_aic = (cmask & PTO2_SUBTASK_MASK_AIC) != 0;
    const bool anchor_is_cube = has_aic;
    const bool type_match = anchor_is_cube ? (self->role == CoreType::AIC) : (self->role == CoreType::AIV);
    bool is_winner = false;
    if (type_match) {
        // Pick the shard for this task (§6.6): shard = N % kCursorShards, a pure
        // function of the task id so every core targets the same sub-cursor for N.
        __gm__ PaddedCursor *cursors = anchor_is_cube ? g_dist.cube_cursor : g_dist.vector_cursor;
        __gm__ volatile int32_t &cursor = cursors[N % kCursorShards].v;
        is_winner = claim(cursor, N);
    }

    // (c) Fan-in resolution — WINNER ONLY. Look up producers of INPUT/INOUT regions
    // BEFORE this task registers its own writes (so an INOUT does not self-match).
    // Losers never consume fanin, so they skip these lookups entirely; correctness
    // is unaffected because the map state read here is identical on every core and
    // only the owner needs the result.
    int32_t fanin[kMaxFanin];
    int32_t fc = 0;
    if (is_winner) {
        for (int32_t i = 0; i < tc; i++) {
            const TensorArgType tag = args.tag(i);
            if (tag != TensorArgType::INPUT && tag != TensorArgType::INOUT) continue;
            // TensorRef::ref() now returns default-address-space; orch's
            // stack-local Tensor is the physical source. DistTensorMap::lookup
            // needs a Tensor to compare buffer.addr against — the address
            // itself is space-agnostic, so a plain reference is enough.
            const Tensor &t = args.tensor(i).ref();
            if (t.manual_dep) continue;
            const int32_t p = DistTensorMap::lookup(self->map, t);
            if (p < 0) continue;
            bool dup = false;
            for (int32_t k = 0; k < fc; k++)
                if (fanin[k] == p) {
                    dup = true;
                    break;
                }
            if (!dup && fc < kMaxFanin) fanin[fc++] = p;
        }
    }

    // (d) Register this task as the producer of its OUTPUT / INOUT / existing
    // outputs — UNCONDITIONAL (every core, so all duplicate maps stay identical).
    uint32_t out_idx = 0;
    for (int32_t i = 0; i < tc; i++) {
        const TensorArgType tag = args.tag(i);
        if (tag == TensorArgType::OUTPUT) {
            DistTensorMap::insert(self->map, result.get_ref(out_idx), N);
            out_idx++;
        } else if (tag == TensorArgType::INOUT || tag == TensorArgType::OUTPUT_EXISTING) {
            DistTensorMap::insert(self->map, args.tensor(i).ref(), N);
        }
    }

    if (!is_winner) {
        TRACE_LAP(self, N, -1, TracePhase::Replay);
        return result;  // wrong type or lost the race: map updated, nothing to build
    }

    // (e) Winner only: assemble the shared argument Tensors (identical for every
    // active lane of a multi-core task — they share the task tensors, each lane
    // writing its designated output per the kernels). Inputs are copied from the
    // args; outputs are the materialized heap-addressed descriptors. Done AFTER
    // the claim so the ~2/3 of cores that fail type_match / lose the race never
    // pay these tc x sizeof(Tensor) copies.
    const uint64_t *scalars = args.scalars();
    const int32_t sc = args.scalar_count();
    Tensor built[MAX_TENSOR_ARGS];
    {
        uint32_t bo = 0;
        for (int32_t i = 0; i < tc; i++) {
            if (args.tag(i) == TensorArgType::OUTPUT) {
                Tensor::copy(built[i], result.get_ref(bo));
                bo++;
            } else {
                Tensor::copy(built[i], args.tensor(i).ref());
            }
        }
    }

    // ---- Winner = owner (single-core) / anchor (multi-core). ----
    // The real per-task build work (claim + fan-in lookup + built[] assembly)
    // ends here; the two back-pressure spins below are WAITING, not work, so
    // close the Build span now and time the spins separately as RingBp. Without
    // this split the spin time was misattributed to "build" (it dominated build
    // under a small ring / few blocks — it is dependency/slot wait, not cost).
    TRACE_LAP(self, N, -1, TracePhase::Build);

    // Back-pressure for self-claimed work: wait until the ring has a non-reserved
    // slot free, draining block.won deposits + ready tasks meanwhile. The reserve
    // guarantees a follower can still pull its (ready) deposits when the rest of
    // the ring is full of not-yet-ready consumers (no priority inversion).
    uint64_t wd_self = 0;
#if DIST_TRACE_ENABLED
    // Swimlane (slot-release edges): if we are about to actually wait, snapshot the
    // tasks currently occupying our ring — those are what must execute to free a
    // slot, i.e. what this ringbp truly waits on. The ring only shrinks during the
    // wait, so the entry snapshot is the complete set.
    if (g_trace_on && self->occupied_count >= kPrivateSlots - kWonReserve) {
        for (int32_t i = 0; i < kPrivateSlots; i++) {
            __gm__ const RingSlot &rs = self->slots[i];
            if (rs.occupied && rs.built) self->slot_edges.push_back({N, rs.task_id});
        }
    }
#endif
    while (self->occupied_count >= kPrivateSlots - kWonReserve && !fatal_set()) {
        drain_block_won(self);
        if (drain_phase_b(self) == 0) {
            SPIN_WAIT_HINT();
            watchdog(wd_self);
        }
    }
    if (fatal_set()) return result;

    // Heap reclaim back-pressure (§9.5/§11.4): this owner is about to build (and
    // later write) task N's outputs at deterministic physical offsets. Recycling a
    // ring region is safe only once its previous occupant's task id <= R = F - H
    // (all that occupant's consumers, which have id <= occupant+H, are done). The
    // equivalent global-derivable test is: the live virtual window (heap_next minus
    // vend[R]) must fit in the ring. Spin (draining + advancing F) until it does.
    if (g_dist.heap_base != nullptr) {
        const size_t ring = g_dist.heap_size;
        uint64_t wd_heap = 0;
        while (!fatal_set()) {
            const int32_t f = atom_load(g_dist.frontier, __ATOMIC_ACQUIRE);
            const int32_t R = f - g_dist.H;
            const uint64_t vstart_live = (R < 0) ? 0 : atom_load(g_dist.vend[R], __ATOMIC_RELAXED);
            if (self->heap_next - vstart_live <= ring) break;  // window fits — region free
            if (f >= N - 1) {  // every predecessor done yet H-window still overflows the ring
                set_fatal();
                DIST_ERRF(
                    "[dist_engine] heap ring %zu B too small for H=%d window at task %d (live=%llu B); "
                    "enlarge PTO_DIST_HEAP_MB or reduce PTO_DIST_H\n",
                    ring, g_dist.H, N, (unsigned long long)(self->heap_next - vstart_live)
                );
                return result;
            }
            drain_block_won(self);
            if (drain_phase_b(self) == 0) {
                SPIN_WAIT_HINT();
                watchdog(wd_heap);
            }
        }
        if (fatal_set()) return result;
    }
    // Time spent in the two back-pressure spins above (ring-slot wait + heap
    // reclaim wait) — dependency/slot WAITING, kept separate from Build.
    TRACE_LAP(self, N, -1, TracePhase::RingBp);

    int32_t si = alloc_ring_slot(self);
    if (si < 0) {  // should not happen given the back-pressure gate above
        set_fatal();
        DIST_ERRF("[dist_engine] no free private-ring slot after back-pressure at task %d\n", N);
        return result;
    }
    // Reserve so concurrent drains (including the block.won back-pressure loop
    // below, which calls drain_phase_b) do not reuse this slot. Mark it unbuilt
    // so Phase B skips it until build_ring_slot populates it (avoids re-executing
    // the prior occupant's stale task_id/fanin/won linkage).
    self->slots[si].occupied = true;
    self->slots[si].built = false;

    int32_t own_lane;
    int32_t won_block = -1;
    int32_t won_slot = -1;
    bool is_multicore = (pc > 1);

    if (!is_multicore) {
        // Single core (1C / 1V): the one active lane is the only subtask. For 1V
        // the winner may be physically AIV0 or AIV1, but the active lane/kernel is
        // AIV0 (rt_submit_aiv fills aiv0). Find the single active lane.
        own_lane = has_aic ? LANE_AIC : LANE_AIV0;
    } else {
        // Multi-core (MIX / 2V): we are the anchor. Our own physical lane subtask
        // goes to our private ring; the remaining active lanes are deposited into
        // block.won for our same-block followers to drain (§3.1).
        own_lane = self->lane;
        won_block = self->block_id;
        won_slot = alloc_won_slot(won_block);
        uint64_t wd_won = 0;
        while (won_slot < 0 && !fatal_set()) {  // block.won full → back-pressure (drain, then retry)
            drain_block_won(self);
            if (drain_phase_b(self) == 0) {
                SPIN_WAIT_HINT();
                watchdog(wd_won);
            }
            won_slot = alloc_won_slot(won_block);
        }
        if (fatal_set()) return result;
        __gm__ WonSlot &w = g_dist.blocks[won_block].slots[won_slot];
        w.task_id = N;
        atom_store<int64_t>(w.remaining, pc, __ATOMIC_RELAXED);
        for (int32_t L = 0; L < PTO2_SUBTASK_SLOT_COUNT; L++) {
            atom_store(w.drained[L], 0, __ATOMIC_RELAXED);
            w.lane[L].present = false;
        }
        for (int32_t L = 0; L < PTO2_SUBTASK_SLOT_COUNT; L++) {
            if (L == own_lane || !lane_active(M, L)) continue;
            __gm__ BuiltSubtask &b = w.lane[L];
            b.present = true;
            b.func_id = kernel_id_for_lane(mixed, L);
            b.function_bin_addr = resolve_kernel_addr(runtime, kernel_id_for_lane(mixed, L));
            b.tensor_count = tc;
            b.scalar_count = sc;
            for (int32_t i = 0; i < tc; i++)
                Tensor::copy(b.tensors[i], built[i]);
            for (int32_t j = 0; j < sc; j++)
                b.scalars[j] = scalars[j];
            b.fanin_count = fc;
            for (int32_t k = 0; k < fc; k++)
                b.fanin[k] = fanin[k];
            b.sub_block_id = (L == LANE_AIV1) ? 1 : 0;
        }
        atom_thread_fence(__ATOMIC_RELEASE);
        atom_store(g_dist.blocks[won_block].any_pub, 1, __ATOMIC_RELEASE);  // enable follower drains
        atom_store(w.state, 1, __ATOMIC_RELEASE);                           // publish the deposits to followers
    }

    const int32_t own_sub_block = (own_lane == LANE_AIV1) ? 1 : 0;
    const int32_t own_func_id = kernel_id_for_lane(mixed, own_lane);
    build_ring_slot(
        self->slots[si], N, own_func_id, resolve_kernel_addr(runtime, own_func_id), built, tc, scalars, sc, fanin, fc,
        own_sub_block, is_multicore, won_block, won_slot
    );
    self->occupied_count++;
    self->owned_total++;

#if DIST_TRACE_ENABLED
    if (g_trace_on) {
        for (int32_t k = 0; k < fc; k++)
            self->dep_edges.push_back({N, fanin[k]});
    }
#endif
    TRACE_LAP(self, N, -1, TracePhase::Commit);
    return result;
}
#if defined(__CCE_AICORE__)
}  // namespace
#endif

// -----------------------------------------------------------------------------
// Device-callable API (dist_engine_api.h). CCEC orchestration wrappers in
// pto_orchestration_api.h call these directly instead of going through
// rt->ops. Host/sim definitions below are functional; CCEC currently carries a
// minimal onboard submit path until the full GM replay engine is enabled.
// -----------------------------------------------------------------------------

// Fatal-state query / report — no va_list version so this is safe to compile
// under CCEC when the gate lifts.
DIST_API_ATTR PTO_DEVICE_FUNC bool dist_is_fatal_query() {
#if defined(__CCE_AICORE__)
    return false;
#else
    return fatal_set();
#endif
}

DIST_API_ATTR PTO_DEVICE_FUNC void dist_report_fatal_msg(
    int32_t code, __gm__ const char *func, __gm__ const char *msg
) {
#if DIST_HOST_ONLY
    set_fatal();
    fprintf(stderr, "[dist_engine][FATAL][%s] code=%d: %s\n", func ? func : "?", code, msg ? msg : "");
#else
    (void)code;
    (void)func;
    (void)msg;
#endif
}

// Log sinks — const-string message API (no va_list).
DIST_API_ATTR PTO_DEVICE_FUNC void dist_log_error_msg(__gm__ const char *func, __gm__ const char *msg) {
#if DIST_HOST_ONLY
    fprintf(stderr, "[dist_engine][E][%s] %s\n", func ? func : "?", msg ? msg : "");
#else
    (void)func;
    (void)msg;
#endif
}
DIST_API_ATTR PTO_DEVICE_FUNC void dist_log_warn_msg(__gm__ const char *, __gm__ const char *) {}
DIST_API_ATTR PTO_DEVICE_FUNC void dist_log_debug_msg(__gm__ const char *, __gm__ const char *) {}
DIST_API_ATTR PTO_DEVICE_FUNC void dist_log_info_v_msg(__gm__ const char *, int, __gm__ const char *) {}

// Scope guard hooks — no-op in the distributed engine (per-core replay does not
// need scope batching).
DIST_API_ATTR PTO_DEVICE_FUNC void dist_scope_begin_impl(PTO2Runtime *) {}
DIST_API_ATTR PTO_DEVICE_FUNC void dist_scope_end_impl(PTO2Runtime *) {}
DIST_API_ATTR PTO_DEVICE_FUNC void dist_orchestration_done_impl(PTO2Runtime *) {}
DIST_API_ATTR PTO_DEVICE_FUNC void dist_scope_set_site_impl(const char *, int) {}

// Dependency-only task — currently unused by fdwic examples, kept as no-op.
DIST_API_ATTR PTO_DEVICE_FUNC TaskOutputTensors dist_submit_dummy_impl(PTO2Runtime *, const L0TaskArgs &) {
    return TaskOutputTensors{};
}

namespace {  // reopen internal namespace for wait_producer_ready only

// Orchestration-side tensor data access (get/set_tensor_data). Replay runs on the
// AICore worker and reads/writes real GM, so these are genuine memory accesses.
// The only subtlety is read-after-write across tasks: if the region has a producer
// in this core's map, wait until that producer's completion flag is set (draining
// this core's own ring meanwhile so an owned producer actually runs). External
// tensors (no producer) are accessed immediately. Consumer (WAR) tracking is not
// modeled, mirroring the centralized runtime's documented INPUT-reader limitation.
#if DIST_HOST_ONLY
// wait_producer_ready + the dist_get/set_tensor_data_impl path below are only
// exercised on sim / AICPU (host-side orchestration replay). CCEC-side orch
// runs through a different code path (C.4.b/c) that will re-introduce a
// __gm__-aware equivalent. Gate the whole cluster off under CCEC so we don't
// force DistCore/DistTensorMap through a non-__gm__ reference chain.
PTO_DEVICE_FUNC void wait_producer_ready(DistCore *self, const Tensor &t) {
    // Cold path (get/set_tensor_data); uses the map's current alive_floor.
    const int32_t p = DistTensorMap::lookup(self->map, t);
    if (p < 0) return;
    uint64_t wd = 0;
    while (!fatal_set()) {
        if (atom_load(g_dist.flags[p & (kFlagCap - 1)], __ATOMIC_ACQUIRE) != 0) break;
        drain_block_won(self);
        if (drain_phase_b(self) == 0) {
            SPIN_WAIT_HINT();
            watchdog(wd);
        }
    }
}
#endif  // DIST_HOST_ONLY

}  // namespace

DIST_API_ATTR PTO_DEVICE_FUNC uint64_t dist_get_tensor_data_impl(
    PTO2Runtime *, const Tensor &tensor, uint32_t ndims, const uint32_t indices[]
) {
    if (tensor.buffer.addr == 0) return 0;
#if DIST_HOST_ONLY
    // Sim/AICPU: g_self is a plain thread_local DistCore*. Reachable to
    // wait_producer_ready as a non-__gm__ pointer.
    DistCore *self = g_self;
    if (self != nullptr) wait_producer_ready(self, tensor);
#endif
    const uint64_t flat = tensor.compute_flat_offset(indices, ndims);
    const uint64_t esz = get_element_size(tensor.dtype);
    uint64_t result = 0;
#if DIST_HOST_ONLY
    __builtin_memcpy(&result, reinterpret_cast<const void *>(tensor.buffer.addr + flat * esz), esz);
#else
    const uint64_t addr = tensor.buffer.addr + flat * esz;
    if (esz == 1) {
        result = *reinterpret_cast<__gm__ const uint8_t *>(addr);
    } else if (esz == 2) {
        result = *reinterpret_cast<__gm__ const uint16_t *>(addr);
    } else if (esz == 4) {
        result = *reinterpret_cast<__gm__ const uint32_t *>(addr);
    } else {
        result = *reinterpret_cast<__gm__ const uint64_t *>(addr);
    }
#endif
    return result;
}

DIST_API_ATTR PTO_DEVICE_FUNC void dist_set_tensor_data_impl(
    PTO2Runtime *, const Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value
) {
    if (tensor.buffer.addr == 0) return;
#if DIST_HOST_ONLY
    DistCore *self = g_self;
    if (self != nullptr) wait_producer_ready(self, tensor);
#endif
    const uint64_t flat = tensor.compute_flat_offset(indices, ndims);
    const uint64_t esz = get_element_size(tensor.dtype);
#if DIST_HOST_ONLY
    __builtin_memcpy(reinterpret_cast<void *>(tensor.buffer.addr + flat * esz), &value, esz);
#else
    const uint64_t addr = tensor.buffer.addr + flat * esz;
    if (esz == 1) {
        *reinterpret_cast<__gm__ uint8_t *>(addr) = static_cast<uint8_t>(value);
    } else if (esz == 2) {
        *reinterpret_cast<__gm__ uint16_t *>(addr) = static_cast<uint16_t>(value);
    } else if (esz == 4) {
        *reinterpret_cast<__gm__ uint32_t *>(addr) = static_cast<uint32_t>(value);
    } else {
        *reinterpret_cast<__gm__ uint64_t *>(addr) = value;
    }
#endif
}

// -----------------------------------------------------------------------------
// Ops-table stubs used by CPU-sim AICore orchestration wrappers. CCEC wrappers
// call dist_engine_api.h directly and do not need this indirection.
// -----------------------------------------------------------------------------
#if DIST_HOST_ONLY
void dist_scope_begin(PTO2Runtime *rt) { dist_scope_begin_impl(rt); }
void dist_scope_end(PTO2Runtime *rt) { dist_scope_end_impl(rt); }
void dist_orchestration_done(PTO2Runtime *rt) { dist_orchestration_done_impl(rt); }
bool dist_is_fatal(PTO2Runtime *) { return fatal_set(); }

void dist_report_fatal(PTO2Runtime *, int32_t code, const char *func, const char *fmt, ...) {
    set_fatal();
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[dist_engine][FATAL][%s] code=%d: ", func ? func : "?", code);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

void dist_log_error(const char *func, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[dist_engine][E][%s] ", func ? func : "?");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}
void dist_log_warn(const char *, const char *, ...) {}
void dist_log_debug(const char *, const char *, ...) {}
void dist_log_info_v(const char *, int, const char *, ...) {}

// Adapters so the legacy ops table still has the ChipStorage/Tensor overloads
// it expects (device-side impls take pointer-array indices; ops table wants a
// (Tensor, ndims, uint32_t const *) function pointer that returns uint64_t).
uint64_t dist_get_tensor_data(PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t *indices) {
    return dist_get_tensor_data_impl(rt, tensor, ndims, indices);
}
void dist_set_tensor_data(
    PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t *indices, uint64_t value
) {
    dist_set_tensor_data_impl(rt, tensor, ndims, indices, value);
}
#endif  // DIST_HOST_ONLY

#if defined(__CCE_AICORE__)
namespace {

PTO_DEVICE_FUNC void ccec_invalidate_region(__gm__ const void *ptr, uint64_t bytes) {
    __gm__ const uint8_t *p = reinterpret_cast<__gm__ const uint8_t *>(ptr);
    for (uint64_t off = 0; off < bytes; off += 64) {
        dcci(const_cast<__gm__ uint8_t *>(p + off), SINGLE_CACHE_LINE);
    }
}

PTO_DEVICE_FUNC void ccec_flush_region(__gm__ void *ptr, uint64_t bytes) {
    __gm__ uint8_t *p = reinterpret_cast<__gm__ uint8_t *>(ptr);
    for (uint64_t off = 0; off < bytes; off += 64) {
        dcci(p + off, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
}

PTO_DEVICE_FUNC uint64_t ccec_tensor_bytes(const Tensor &t) {
    uint64_t elems = 1;
    for (uint32_t i = 0; i < t.ndims; i++) {
        elems *= t.shapes[i];
    }
    return elems * get_element_size(t.dtype);
}

PTO_DEVICE_FUNC int32_t ccec_lookup_producer(const Tensor &t) {
    if (g_ccec_runtime == nullptr || g_ccec_core_idx < 0 || g_ccec_core_idx >= RUNTIME_MAX_WORKER) return -1;
    const uint64_t addr = t.buffer.addr + t.start_offset * get_element_size(t.dtype);
    const uint64_t size = ccec_tensor_bytes(t);
    __gm__ Runtime::DistHandoff::CcecMapEntry *entries =
        g_ccec_runtime->dist.ccec_maps[g_ccec_core_idx].entries;
    for (int32_t i = g_ccec_map_count - 1; i >= 0; i--) {
        __gm__ const Runtime::DistHandoff::CcecMapEntry &entry = entries[i];
        if (entry.addr == addr && entry.size == size) return entry.task_id;
    }
    return -1;
}

PTO_DEVICE_FUNC void ccec_insert_producer(const Tensor &t, int32_t task_id) {
    if (g_ccec_runtime == nullptr || g_ccec_core_idx < 0 || g_ccec_core_idx >= RUNTIME_MAX_WORKER) return;
    const uint64_t addr = t.buffer.addr + t.start_offset * get_element_size(t.dtype);
    const uint64_t size = ccec_tensor_bytes(t);
    __gm__ Runtime::DistHandoff::CcecMapEntry *entries =
        g_ccec_runtime->dist.ccec_maps[g_ccec_core_idx].entries;
    for (int32_t i = g_ccec_map_count - 1; i >= 0; i--) {
        __gm__ Runtime::DistHandoff::CcecMapEntry &entry = entries[i];
        if (entry.addr == addr && entry.size == size) {
            entry.task_id = task_id;
            ccec_flush_region(&entry, sizeof(entry));
            return;
        }
    }
    if (g_ccec_map_count < 8) {
        __gm__ Runtime::DistHandoff::CcecMapEntry &entry = entries[g_ccec_map_count++];
        entry.addr = addr;
        entry.size = size;
        entry.task_id = task_id;
        entry.pad = 0;
        ccec_flush_region(&entry, sizeof(entry));
    }
}

PTO_DEVICE_FUNC void ccec_publish_flag(int32_t task_id) {
    if (g_ccec_runtime == nullptr || task_id < 0 || task_id >= 2048) return;
    g_ccec_runtime->dist.ccec_flags[task_id] = 1;
    ccec_flush_region(const_cast<__gm__ int64_t *>(&g_ccec_runtime->dist.ccec_flags[task_id]), 64);
}

PTO_DEVICE_FUNC void ccec_wait_flag(int32_t task_id) {
    if (g_ccec_runtime == nullptr || task_id < 0 || task_id >= 2048) return;
    while (true) {
        ccec_invalidate_region(const_cast<__gm__ int64_t *>(&g_ccec_runtime->dist.ccec_flags[task_id]), 64);
        if (g_ccec_runtime->dist.ccec_flags[task_id] != 0) return;
        SPIN_WAIT_HINT();
    }
}

PTO_DEVICE_FUNC void ccec_wait_fanin(const int32_t fanin[], int32_t count) {
    for (int32_t i = 0; i < count; i++) {
        ccec_wait_flag(fanin[i]);
    }
}

struct CcecSubmitCtx {
    int32_t task_id;
    TaskOutputTensors result;
    int32_t fanin[kMaxFanin];
    int32_t fanin_count;
    int32_t kernel_id;
    bool won;
};

enum class CcecSubmitKind : int32_t {
    Kernel = 0,
    Alloc = 1,
};

PTO_DEVICE_FUNC CcecSubmitCtx ccec_begin_submit() {
    CcecSubmitCtx ctx;
    ctx.task_id = g_ccec_local_index++;
    ctx.fanin_count = 0;
    ctx.kernel_id = INVALID_KERNEL_ID;
    ctx.won = false;
    return ctx;
}

PTO_DEVICE_FUNC void ccec_materialize_outputs(const L0TaskArgs &args, CcecSubmitCtx &ctx) {
    if (g_ccec_runtime == nullptr || g_ccec_core_idx < 0 || g_ccec_core_idx >= RUNTIME_MAX_WORKER) return;
    const uint64_t heap_base = g_ccec_runtime->dist.ccec_heap_base;
    const uint64_t heap_size = g_ccec_runtime->dist.ccec_heap_size;
    if (heap_base == 0 || heap_size == 0) return;

    uint64_t total = 0;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        total += PTO2_ALIGN_UP(TensorCreateInfo::buffer_size_bytes(args.tensor(i).create_info()), PTO2_PACKED_OUTPUT_ALIGN);
    }
    uint64_t task_base = PTO2_ALIGN_UP(g_ccec_heap_next, PTO2_PACKED_OUTPUT_ALIGN);
    if (total > 0 && (task_base % heap_size) + total > heap_size) {
        task_base = ((task_base / heap_size) + 1) * heap_size;
    }

    uint64_t off = 0;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        const TensorCreateInfo &ci = args.tensor(i).create_info();
        const uint64_t logical = TensorCreateInfo::buffer_size_bytes(ci);
        const uint64_t sz = PTO2_ALIGN_UP(logical, PTO2_PACKED_OUTPUT_ALIGN);
        const uint64_t phys = (task_base + off) % heap_size;
        __gm__ Tensor &slot_t =
            g_ccec_runtime->dist.ccec_output_tensors[g_ccec_core_idx].tensors[g_ccec_outpool_head];
        g_ccec_outpool_head = (g_ccec_outpool_head + 1) % 8;
        init_tensor_from_create_info(slot_t, ci, reinterpret_cast<void *>(heap_base + phys), logical);
        ccec_flush_region(&slot_t, sizeof(Tensor));
        ctx.result.materialize_output(slot_t);
        off += sz;
    }
    g_ccec_heap_next = task_base + off;
}

PTO_DEVICE_FUNC void ccec_register_outputs(const L0TaskArgs &args, CcecSubmitCtx &ctx) {
    uint32_t out_idx = 0;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        const TensorArgType tag = args.tag(i);
        if (tag == TensorArgType::OUTPUT) {
            Tensor t;
            Tensor::copy(t, ctx.result.get_ref(out_idx));
            ccec_insert_producer(t, ctx.task_id);
            out_idx++;
        } else if (tag == TensorArgType::INOUT || tag == TensorArgType::OUTPUT_EXISTING) {
            ccec_insert_producer(args.tensor(i).ref(), ctx.task_id);
        }
    }
}

PTO_DEVICE_FUNC bool ccec_claim_kernel_submit_deterministic(const MixedKernels &mixed, CcecSubmitCtx &ctx) {
    ctx.kernel_id = INVALID_KERNEL_ID;
    ctx.won = false;
    const bool has_aic = mixed.aic_kernel_id != INVALID_KERNEL_ID;
    const bool has_aiv = mixed.aiv0_kernel_id != INVALID_KERNEL_ID || mixed.aiv1_kernel_id != INVALID_KERNEL_ID;
    if (has_aic && !has_aiv) {
        if (g_ccec_core_type != static_cast<int32_t>(CoreType::AIC)) return false;
        if (g_ccec_aic_count <= 0 || g_ccec_ordinal != ctx.task_id % g_ccec_aic_count) return false;
        ctx.kernel_id = mixed.aic_kernel_id;
        ctx.won = true;
        return true;
    }
    if (!has_aic && has_aiv) {
        if (g_ccec_core_type != static_cast<int32_t>(CoreType::AIV)) return false;
        if (g_ccec_aiv_count <= 0 || g_ccec_ordinal != ctx.task_id % g_ccec_aiv_count) return false;
        ctx.kernel_id = mixed.aiv0_kernel_id != INVALID_KERNEL_ID ? mixed.aiv0_kernel_id : mixed.aiv1_kernel_id;
        ctx.won = true;
        return true;
    }
    return false;
}

PTO_DEVICE_FUNC bool ccec_claim_alloc_submit_single_owner(CcecSubmitCtx &ctx) {
    ctx.kernel_id = INVALID_KERNEL_ID;
    ctx.won = g_ccec_core_idx == 0;
    return ctx.won;
}

PTO_DEVICE_FUNC bool ccec_claim_submit(CcecSubmitKind kind, const MixedKernels *mixed, CcecSubmitCtx &ctx) {
    if (kind == CcecSubmitKind::Alloc) return ccec_claim_alloc_submit_single_owner(ctx);
    if (mixed == nullptr) return false;
    return ccec_claim_kernel_submit_deterministic(*mixed, ctx);
}

PTO_DEVICE_FUNC void ccec_collect_fanin(const L0TaskArgs &args, CcecSubmitCtx &ctx) {
    ctx.fanin_count = 0;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        const TensorArgType tag = args.tag(i);
        if (tag != TensorArgType::INPUT && tag != TensorArgType::INOUT) continue;
        const int32_t producer = ccec_lookup_producer(args.tensor(i).ref());
        if (producer < 0) continue;
        bool dup = false;
        for (int32_t k = 0; k < ctx.fanin_count; k++) {
            if (ctx.fanin[k] == producer) {
                dup = true;
                break;
            }
        }
        if (!dup && ctx.fanin_count < kMaxFanin) ctx.fanin[ctx.fanin_count++] = producer;
    }
}

PTO_DEVICE_FUNC bool ccec_build_kernel_payload(
    const L0TaskArgs &args, const TaskOutputTensors &result, __gm__ PTO2DispatchPayload *payload
) {
    if (payload == nullptr || g_ccec_runtime == nullptr) return false;
    int32_t argc = 0;
    uint32_t out_idx = 0;
    for (int32_t i = 0; i < args.tensor_count() && argc < PTO2_DISPATCH_MAX_ARGS; i++) {
        __gm__ Tensor &dst = g_ccec_runtime->dist.ccec_kernel_tensors[g_ccec_core_idx].tensors[i];
        if (args.tag(i) == TensorArgType::OUTPUT) {
            Tensor tensor;
            Tensor::copy(tensor, result.get_ref(out_idx));
            Tensor::copy(dst, tensor);
            out_idx++;
        } else {
            const Tensor &tensor = args.tensor(i).ref();
            Tensor::copy(dst, tensor);
        }
        ccec_flush_region(&dst, sizeof(Tensor));
        payload->args[argc++] = reinterpret_cast<uint64_t>(&dst);
    }
    for (int32_t i = 0; i < args.scalar_count() && argc < PTO2_DISPATCH_MAX_ARGS; i++) {
        payload->args[argc++] = args.scalar(i);
    }
    ccec_flush_region(payload, sizeof(PTO2DispatchPayload));
    return true;
}

PTO_DEVICE_FUNC void ccec_call_winner_kernel(int32_t kernel_id, __gm__ PTO2DispatchPayload *payload) {
    const bool is_aic = g_ccec_core_type == static_cast<int32_t>(CoreType::AIC);
    if (is_aic) {
        if (pto_call_linked_kernel_aic != nullptr) {
            pto_call_linked_kernel_aic(kernel_id, reinterpret_cast<__gm__ int64_t *>(payload->args));
        }
    } else if (pto_call_linked_kernel_aiv != nullptr) {
        pto_call_linked_kernel_aiv(kernel_id, reinterpret_cast<__gm__ int64_t *>(payload->args));
    }
}

PTO_DEVICE_FUNC void ccec_execute_won_submit(const L0TaskArgs &args, CcecSubmitCtx &ctx) {
    __gm__ PTO2DispatchPayload *payload =
        reinterpret_cast<__gm__ PTO2DispatchPayload *>(g_ccec_runtime->workers[g_ccec_core_idx].task);
    if (payload == nullptr) return;

    ccec_wait_fanin(ctx.fanin, ctx.fanin_count);
    if (!ccec_build_kernel_payload(args, ctx.result, payload)) return;
    ccec_call_winner_kernel(ctx.kernel_id, payload);
    OUT_OF_ORDER_STORE_BARRIER();
    ccec_publish_flag(ctx.task_id);
}

PTO_DEVICE_FUNC void ccec_complete_alloc_submit(CcecSubmitCtx &ctx) {
    if (ctx.won) {
        ccec_publish_flag(ctx.task_id);
    } else {
        ccec_wait_flag(ctx.task_id);
    }
}

PTO_DEVICE_FUNC void ccec_init_worker_layout() {
    if (g_ccec_runtime == nullptr) return;
    const int32_t num_workers = g_ccec_runtime->dist.num_workers;
    const int32_t block_dim = num_workers / PLATFORM_CORES_PER_BLOCKDIM;
    g_ccec_aic_count = block_dim;
    g_ccec_aiv_count = block_dim * PLATFORM_AIV_CORES_PER_BLOCKDIM;
    g_ccec_ordinal = INVALID_KERNEL_ID;
    g_ccec_valid_worker = false;
    const CoreType self_type = static_cast<CoreType>(g_ccec_core_type);
    if (self_type == CoreType::AIC && g_ccec_core_idx >= 0 && g_ccec_core_idx < block_dim) {
        g_ccec_ordinal = g_ccec_core_idx;
        g_ccec_valid_worker = true;
    } else if (self_type == CoreType::AIV && g_ccec_core_idx >= block_dim && g_ccec_core_idx < num_workers) {
        g_ccec_ordinal = g_ccec_core_idx - block_dim;
        g_ccec_valid_worker = true;
    }
}

PTO_DEVICE_FUNC bool ccec_is_valid_worker() {
    return g_ccec_valid_worker;
}

PTO_DEVICE_FUNC void ccec_publish_done() {
    OUT_OF_ORDER_STORE_BARRIER();
    write_reg(RegId::COND, MAKE_FIN_VALUE(0));
}

PTO_DEVICE_FUNC void ccec_replay_orch(__gm__ Runtime *runtime) {
    if (aicpu_orchestration_entry == nullptr || !ccec_is_valid_worker()) return;
    ccec_invalidate_region(runtime->dist.ccec_orch_tensors, sizeof(runtime->dist.ccec_orch_tensors));
    ccec_invalidate_region(runtime->dist.ccec_orch_scalars, sizeof(runtime->dist.ccec_orch_scalars));
    ccec_invalidate_region(
        const_cast<__gm__ const int32_t *>(&runtime->dist.ccec_orch_tensor_count), 64
    );
    L2TaskArgs local_args;
    Tensor local_tensors[CHIP_MAX_TENSOR_ARGS];
    const int32_t tensor_count = runtime->dist.ccec_orch_tensor_count;
    const int32_t scalar_count = runtime->dist.ccec_orch_scalar_count;
    for (int32_t i = 0; i < tensor_count && i < CHIP_MAX_TENSOR_ARGS; i++) {
        Tensor::copy(local_tensors[i], runtime->dist.ccec_orch_tensors[i]);
        local_args.add_input(local_tensors[i]);
    }
    for (int32_t i = 0; i < scalar_count && i < CHIP_MAX_SCALAR_ARGS; i++) {
        const uint64_t scalar = runtime->dist.ccec_orch_scalars[i];
        local_args.add_scalar(scalar);
    }
    aicpu_orchestration_entry(local_args);
}

}  // namespace

DIST_API_ATTR PTO_DEVICE_FUNC TaskOutputTensors dist_submit_impl(
    PTO2Runtime *, const MixedKernels &mixed, const L0TaskArgs &args
) {
    CcecSubmitCtx ctx = ccec_begin_submit();
    ccec_materialize_outputs(args, ctx);
    ccec_collect_fanin(args, ctx);
    ccec_register_outputs(args, ctx);
    if (ccec_claim_submit(CcecSubmitKind::Kernel, &mixed, ctx)) ccec_execute_won_submit(args, ctx);
    return ctx.result;
}
#else
// alloc_tensors — a kernel-less "hidden task" that only reserves GM output
// buffers (no compute). It consumes one task id, allocates its outputs on the
// deterministic heap exactly like dist_submit_impl step (a), registers itself as
// their producer, and completes INLINE (sets its own flag immediately) since no
// kernel runs. A later writer (INOUT / OUTPUT_EXISTING) becomes the new producer
// of the region, so real consumers depend on the writer, not on this alloc. Every
// core replays it identically, keeping heap addresses + maps consistent.
DIST_API_ATTR PTO_DEVICE_FUNC TaskOutputTensors dist_alloc_tensors(PTO2Runtime *, const L0TaskArgs &args) {
    __gm__ DistCore *self = g_self;
    if (self == nullptr) return TaskOutputTensors{};
    // EXECUTE-FIRST (docs §6 step 0+1, §6.1): every submit point first seeks an
    // execution opportunity before advancing the deterministic replay below.
    TRACE_LAP_RESET(self);  // exclude the inter-submit orch round-trip (user code) from runtime spans
    if (!fatal_set()) {
        drain_block_won(self);
        drain_phase_b(self);
    }
    TRACE_LAP(self, self->local_index, -1, TracePhase::EfDrain);
    const int32_t N = self->local_index++;
    const int32_t tc = args.tensor_count();
    if (N >= kFlagCap) {
        set_fatal();
        DIST_ERRF("[dist_engine] alloc task id %d exceeds kFlagCap %d\n", N, kFlagCap);
        return TaskOutputTensors{};
    }

    // Deterministic GM heap allocation + straddle-padding (identical to submit (a)).
    const size_t ring = g_dist.heap_size;
    uint64_t total = 0;
    for (int32_t i = 0; i < tc; i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        total += PTO2_ALIGN_UP(TensorCreateInfo::buffer_size_bytes(args.tensor(i).create_info()), PTO2_PACKED_OUTPUT_ALIGN);
    }
    uint64_t task_base = PTO2_ALIGN_UP(self->heap_next, PTO2_PACKED_OUTPUT_ALIGN);
    if (total > 0 && g_dist.heap_base != nullptr) {
        if (total > ring) {
            set_fatal();
            DIST_ERRF(
                "[dist_engine] alloc task %d outputs %llu B exceed heap ring %zu B\n", N,
                (unsigned long long)total, ring
            );
            return TaskOutputTensors{};
        }
        if ((task_base % ring) + total > ring) task_base = ((task_base / ring) + 1) * ring;
    }

    // (a) Materialize outputs + publish the deterministic heap layout — EVERY core
    // (like dist_submit_impl step (a)), so duplicate maps and vend[] stay identical.
    uint64_t off = 0;
    TaskOutputTensors result;
    for (int32_t i = 0; i < tc; i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        // TensorRef::create_info() now returns a default-address-space ref
        // (see pto_types.h — orch is all stack-local under CCEC). The engine
        // still writes the derived Tensor into a GM outpool slot; the
        // cross-space transition happens inside init_tensor_from_create_info,
        // whose signature takes __gm__ dst + non-__gm__ src.
        const TensorCreateInfo &ci = args.tensor(i).create_info();
        const uint64_t logical = TensorCreateInfo::buffer_size_bytes(ci);
        const uint64_t sz = PTO2_ALIGN_UP(logical, PTO2_PACKED_OUTPUT_ALIGN);
        if (g_dist.heap_base == nullptr) {
            set_fatal();
            DIST_ERRF("[dist_engine] GM output heap not allocated at alloc %d\n", N);
            return result;
        }
        const uint64_t phys = (task_base + off) % ring;
        __gm__ Tensor &slot_t = self->outpool[self->outpool_head];
        self->outpool_head = (self->outpool_head + 1) % kOutPoolSlots;
        init_tensor_from_create_info(slot_t, ci, g_dist.heap_base + phys, logical);
        result.materialize_output(slot_t);
        off += sz;
    }
    self->heap_next = task_base + off;
    if (N >= 0 && N < kFlagCap) atom_store(g_dist.vend[N], self->heap_next, __ATOMIC_RELAXED);
    if (fatal_set()) return result;

    // (b) Register this alloc as producer of each output — EVERY core (map parity).
    DistTensorMap::advance_retire(self->map, N, g_dist.H);
    uint32_t out_idx = 0;
    for (int32_t i = 0; i < tc; i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        DistTensorMap::insert(self->map, result.get_ref(out_idx), N);
        out_idx++;
    }

    // (c) Single-owner election (mirrors dist_submit_impl's claim). The first core
    // to reach this alloc id wins; that core is by construction at/ahead of the
    // completion frontier (N is not yet done, so F < N), hence the winner-only
    // back-pressure below can never see heap_next < vend[F-H] and never underflows.
    // Losers have finished the deterministic bookkeeping above and return — the
    // winner alone paces reclaim and publishes the completion flag (the leading
    // core was the one gating completion before this change too, so timing is
    // unchanged; this only drops the lagging cores' redundant pass).
    bool is_winner = claim(g_dist.alloc_cursor[N % kCursorShards].v, N);
    if (!is_winner) {
        TRACE_LAP(self, N, -1, TracePhase::Replay);
        return result;
    }

    // (d) Winner-only heap reclaim back-pressure: drain this core's ring while the
    // live virtual window [vend[F-H], heap_next) would overflow the physical ring.
    if (total > 0 && g_dist.heap_base != nullptr) {
        uint64_t wd_heap = 0;
        while (!fatal_set()) {
            const int32_t f = atom_load(g_dist.frontier, __ATOMIC_ACQUIRE);
            const int32_t R = f - g_dist.H;
            const uint64_t vstart_live = (R < 0) ? 0 : atom_load(g_dist.vend[R], __ATOMIC_RELAXED);
            if (self->heap_next - vstart_live <= ring) break;  // window fits — region free
            if (f >= N - 1) {
                set_fatal();
                DIST_ERRF(
                    "[dist_engine] heap ring %zu B too small for H=%d window at alloc %d (live=%llu B)\n", ring,
                    g_dist.H, N, (unsigned long long)(self->heap_next - vstart_live)
                );
                return result;
            }
            drain_block_won(self);
            if (drain_phase_b(self) == 0) {
                SPIN_WAIT_HINT();
                watchdog(wd_heap);
            }
        }
        if (fatal_set()) return result;
    }

    // (e) Winner completes inline (no kernel runs).
    atom_store(g_dist.flags[N & (kFlagCap - 1)], 1, __ATOMIC_RELEASE);
    advance_frontier();
    TRACE_LAP(self, N, -1, TracePhase::Alloc);
    return result;
}
#endif

#if defined(__CCE_AICORE__)
DIST_API_ATTR PTO_DEVICE_FUNC TaskOutputTensors dist_alloc_tensors(PTO2Runtime *, const L0TaskArgs &args) {
    CcecSubmitCtx ctx = ccec_begin_submit();
    ccec_materialize_outputs(args, ctx);
    ccec_register_outputs(args, ctx);
    ccec_claim_submit(CcecSubmitKind::Alloc, nullptr, ctx);
    ccec_complete_alloc_submit(ctx);
    return ctx.result;
}
#endif

#if DIST_HOST_ONLY
const PTO2RuntimeOps g_dist_ops = {
    dist_submit_impl,          dist_scope_begin,          dist_scope_end,     dist_orchestration_done, dist_is_fatal,
    dist_report_fatal,         dist_log_error,            dist_log_warn,      dist_log_debug,          dist_log_info_v,
    dist_get_tensor_data,      dist_set_tensor_data,      dist_alloc_tensors, dist_submit_dummy_impl,  dist_scope_set_site_impl,
};
#endif  // DIST_HOST_ONLY

// -----------------------------------------------------------------------------
// Deadlock diagnostics: dump the full engine state on SIGUSR1. Sim runs every
// core as a pthread in one process, so a single handler can walk g_dist. Used to
// debug hangs (kill -USR1 <pid>); compiled in but inert unless signalled.
// -----------------------------------------------------------------------------
#if DIST_HOST_ONLY
void dist_dump_state(int) {
    fprintf(stderr, "\n===== DIST STATE DUMP =====\n");
    fprintf(
        stderr, "frontier=%d H=%d ring=%zuB replay_done=%ld/%d num_blocks=%d fatal=%d\n",
        atom_load(g_dist.frontier, __ATOMIC_RELAXED), g_dist.H, g_dist.heap_size,
        static_cast<long>(atom_load(g_dist.replay_done, __ATOMIC_RELAXED)), g_dist.num_workers, g_dist.num_blocks,
        atom_load(g_dist.fatal, __ATOMIC_RELAXED)
    );
    fprintf(stderr, "cube_cursor[%d]=", kCursorShards);
    for (int32_t s = 0; s < kCursorShards; s++)
        fprintf(
            stderr, "%d%s", atom_load(g_dist.cube_cursor[s].v, __ATOMIC_RELAXED),
            s + 1 < kCursorShards ? "," : ""
        );
    fprintf(stderr, " vector_cursor[%d]=", kCursorShards);
    for (int32_t s = 0; s < kCursorShards; s++)
        fprintf(
            stderr, "%d%s", atom_load(g_dist.vector_cursor[s].v, __ATOMIC_RELAXED),
            s + 1 < kCursorShards ? "," : ""
        );
    fprintf(stderr, "\n");
    for (int32_t c = 0; c < g_dist.num_workers && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = g_dist.cores[c];
        fprintf(
            stderr, "core %d role=%d blk=%d lane=%d replayed=%d occ=%d owned=%d\n", c, static_cast<int>(co.role),
            co.block_id, co.lane, co.local_index, co.occupied_count, co.owned_total
        );
        for (int32_t i = 0; i < kPrivateSlots; i++) {
            RingSlot &s = co.slots[i];
            if (!s.occupied) continue;
            int32_t unmet = -1;
            for (int32_t f = 0; f < s.fanin_count; f++)
                if (atom_load(g_dist.flags[s.fanin[f] & (kFlagCap - 1)], __ATOMIC_RELAXED) == 0) {
                    unmet = s.fanin[f];
                    break;
                }
            fprintf(
                stderr, "    slot%d tid=%d built=%d mc=%d won=(%d,%d) fanin=%d unmet=%d\n", i, s.task_id, s.built,
                s.is_multicore, s.won_block, s.won_slot, s.fanin_count, unmet
            );
        }
    }
    for (int32_t b = 0; b < g_dist.num_blocks; b++) {
        for (int32_t i = 0; i < kPrivateSlots; i++) {
            WonSlot &w = g_dist.blocks[b].slots[i];
            int32_t st = atom_load(w.state, __ATOMIC_RELAXED);
            if (st == 0) continue;
            fprintf(
                stderr, "  won blk%d slot%d state=%d tid=%d remaining=%ld drained=[%d,%d,%d] present=[%d,%d,%d]\n", b,
                i, st, w.task_id, static_cast<long>(atom_load(w.remaining, __ATOMIC_RELAXED)),
                atom_load(w.drained[0], __ATOMIC_RELAXED), atom_load(w.drained[1], __ATOMIC_RELAXED),
                atom_load(w.drained[2], __ATOMIC_RELAXED), w.lane[0].present, w.lane[1].present, w.lane[2].present
            );
        }
    }
    fprintf(stderr, "===== END DUMP =====\n");
}
#endif  // DIST_HOST_ONLY

// -----------------------------------------------------------------------------
// Per-core entry point invoked by each AICore worker thread.
// -----------------------------------------------------------------------------
DIST_API_ATTR PTO_DEVICE_FUNC void dist_core_main(__gm__ Runtime *runtime, int core_idx, int core_type_int) {
#if defined(__CCE_AICORE__)
    if (runtime == nullptr || core_idx < 0 || core_idx >= RUNTIME_MAX_WORKER) return;

    g_ccec_runtime = runtime;
    g_ccec_core_idx = core_idx;
    g_ccec_core_type = core_type_int;
    g_ccec_local_index = 0;
    g_ccec_heap_next = 0;
    g_ccec_outpool_head = 0;
    g_ccec_map_count = 0;
    ccec_init_worker_layout();

    ccec_replay_orch(runtime);
    ccec_publish_done();

    g_ccec_runtime = nullptr;
    return;
#else
    if (core_idx < 0 || core_idx >= RUNTIME_MAX_WORKER) return;
#if defined(__CPU_SIM)
    g_dist_ptr = reinterpret_cast<DistGlobal *>(runtime->dist.shared_addr);
    if (g_dist_ptr == nullptr) return;
#endif
    __gm__ DistCore *self = &g_dist.cores[core_idx];
    const CoreType role = static_cast<CoreType>(core_type_int);

    // sub_block lane: only meaningful for AIV in MIX tasks (M3). bgemm's 1V add
    // ignores it, so 0 is correct for the M2 single-core scope.
    // Copy field-by-field from the __gm__ layout entry into a stack local so
    // downstream code reads plain int32_t (CCEC forbids copy-initializing a
    // non-__gm__ struct value from a __gm__ struct value, and CoreLayout has
    // no user-defined ctor / operator= to overload for the cross-space copy).
    __gm__ const CoreLayout &layout_gm = g_dist.layout[core_idx];
    const CoreLayout lay = {layout_gm.block_id, layout_gm.lane};
    DistCore::reset(*self, role, lay.block_id, lay.lane);
    self->core_idx = core_idx;
    g_self = self;
#if defined(__CPU_SIM) && !defined(__CCE_AICORE__)
    if (g_dist.rt != nullptr) {
        g_dist.rt->ops = &g_dist_ops;
    }
#endif
#if DIST_HOST_ONLY
    if (dist_trace())
        fprintf(
            stderr, "[dist] core %d role=%d block=%d lane=%d START\n", core_idx, core_type_int, lay.block_id, lay.lane
        );
#endif

    // Startup barrier: wait until every worker thread has been scheduled in and
    // reached this point before anyone begins replay. In sim the OS brings the
    // host threads up one at a time, so without this the cores that start early
    // race ahead and the swimlane's first-task stagger reflects thread-wakeup
    // skew rather than engine scheduling. Bare spin (no yield) per the AICPU
    // spin-wait convention. Skipped under fatal so a failed run still tears down.
    if (!fatal_set()) {
        atom_fetch_add<int64_t>(g_dist.started_count, 1, __ATOMIC_ACQ_REL);
        uint64_t wd_start = 0;
        while (atom_load(g_dist.started_count, __ATOMIC_ACQUIRE) < g_dist.num_workers && !fatal_set()) {
            SPIN_WAIT_HINT();
            watchdog(wd_start);
        }
    }

    // Replay the full orchestration submit stream: build the per-core map and
    // claim/build owned tasks into the private ring (back-pressure inline). MIX
    // anchors deposit follower subtasks into block.won during this replay.
    TRACE_LAP_RESET(self);  // origin for the first lap span (post-barrier, pre-replay)
    if (g_dist.orch_args != nullptr && !fatal_set()) {
#if defined(__CPU_SIM) && !defined(__CCE_AICORE__)
        framework_bind_runtime(g_dist.rt);
#endif
#if defined(__CPU_SIM)
        aicpu_orchestration_entry(*g_dist.orch_args);
#endif
    }

    // Publish "my replay is done" so followers can eventually conclude that no
    // further block.won deposits will arrive for them (§7 tail-idle).
    atom_fetch_add<int64_t>(g_dist.replay_done, 1, __ATOMIC_ACQ_REL);

    // Drain to completion: pull any follower deposits addressed to my lane, run
    // ready tasks, and only finish once every core has finished replay (no more
    // pushes), my private ring is empty, and there is no undrained deposit left
    // for my lane.
    uint64_t wd_drain = 0;
    while (!fatal_set()) {
        drain_block_won(self);
        int32_t freed = drain_phase_b(self);
        const bool all_replayed = atom_load(g_dist.replay_done, __ATOMIC_ACQUIRE) >= g_dist.num_workers;
        const bool ring_empty = (self->occupied_count == 0);
        const bool pending = has_pending_won(self);
        if (all_replayed && ring_empty && !pending) break;
        if (freed == 0) {
            SPIN_WAIT_HINT();
            watchdog(wd_drain);
        }
    }

#if DIST_HOST_ONLY
    if (dist_trace() || fatal_set()) {
        fprintf(
            stderr, "[dist] core %d role=%d DONE replayed=%d owned=%d fatal=%d\n", core_idx, core_type_int,
            self->local_index, self->owned_total, fatal_set() ? 1 : 0
        );
    }
#endif
    g_self = nullptr;
    __atomic_add_fetch(&runtime->dist.done_count, 1, __ATOMIC_ACQ_REL);
#endif
}

#if defined(__CPU_SIM)
extern "C" __attribute__((visibility("default"))) PTO_DEVICE_FUNC void
aicore_dist_core_main(__gm__ Runtime *runtime, int core_idx, int core_type_int) {
    dist_core_main(runtime, core_idx, core_type_int);
}
#endif

// (No trailing anonymous-namespace close: dist_alloc_tensors / g_dist_ops /
// dist_dump_state / dist_core_main were pulled out into external linkage above
// so orch wrappers can reach dist_submit_impl / dist_alloc_tensors directly.)

// dist_engine_register / dist_engine_dump_trace are host-side entry points
// called from libaicpu_kernel (sim orchestrator thread) and — in the current
// transitional layout — from the AICPU stub on onboard. They own configuration
// reads (env vars), signal-handler installation, host malloc, and the swimlane
// dumper, none of which are available under CCEC. Gate the entire host-only
// tail so CCEC compilations of dist_engine.cpp end at the namespace close and
// leave these symbols to libaicpu_kernel.
#if DIST_HOST_ONLY

void *dist_engine_register(PTO2Runtime *rt, const L2TaskArgs *orch_args, int num_workers, Runtime *runtime) {
    // GM output heap: a BOUNDED ring reclaimed by the completion frontier (M4).
    // Size from PTO_DIST_HEAP_MB (MiB) else kHeapRingDefault. Allocated once per
    // process; if a later run needs a different size, free + realloc.
    {
        size_t want = kHeapRingDefault;
        if (const char *e = getenv("PTO_DIST_HEAP_MB")) {
            const long mb = atol(e);
            if (mb > 0) want = static_cast<size_t>(mb) << 20;
        }
        if (g_dist.heap_base != nullptr && g_dist.heap_size != want) {
            free(g_dist.heap_base);
            g_dist.heap_base = nullptr;
        }
        if (g_dist.heap_base == nullptr) {
            g_dist.heap_base = static_cast<uint8_t *>(malloc(want));
            g_dist.heap_size = (g_dist.heap_base != nullptr) ? want : 0;
        }
        // Zero the heap each run so freshly-allocated output regions read as 0,
        // matching the centralized runtime's zero-initialized GM. Kernels that
        // read a padded tile (e.g. softmax/PV where valid_len < tile width) rely
        // on the unwritten remainder being zero; an uninitialized (malloc) or
        // recycled heap would otherwise yield nondeterministic results.
        if (g_dist.heap_base != nullptr) memset(g_dist.heap_base, 0, g_dist.heap_size);
    }
    // Dependency-span bound H (R = F - H). Env override for graphs with longer
    // heap spans; default kHDefault.
    g_dist.H = kHDefault;
    if (const char *e = getenv("PTO_DIST_H")) {
        const long h = atol(e);
        if (h >= 0) g_dist.H = static_cast<int32_t>(h);
    }
    // The producer map recycles a task's entry-head slot kTaskWindow tasks later;
    // cleanup retires a task once it leaves the H span, so H must stay below the
    // window (with margin) or a slot could be reused before its task is cleaned.
    always_assert(g_dist.H < kTaskWindow - 1);
#if DIST_TRACE_ENABLED
    // Swimlane tracing gate. Capture the epoch now so every core's event ts is
    // relative to the same run start.
    g_trace_on = (getenv("PTO_DIST_SWIMLANE") != nullptr);
    g_trace_epoch_ns = now_ns();
    // Per-core span reserve: 0 when off (reset never reserves → zero overhead on a
    // normal run); a generous bound when on so push_back never reallocs for the
    // sizes we actually analyze (a realloc would perturb heap layout + add timing
    // noise to the very gaps we measure). Best-effort: a huge trace may still grow.
    g_trace_reserve = g_trace_on ? (1 << 16) : 0;
#endif
#if DIST_SIM_HOST_CLOCK
    // Overhead-isolation gate (skip incore kernel calls, keep all bookkeeping).
    g_skip_exec = (getenv("PTO_DIST_SKIP_EXEC") != nullptr);
#endif

    for (int32_t s = 0; s < kCursorShards; s++) {
        atom_store(g_dist.cube_cursor[s].v, -1, __ATOMIC_RELAXED);
        atom_store(g_dist.vector_cursor[s].v, -1, __ATOMIC_RELAXED);
        atom_store(g_dist.alloc_cursor[s].v, -1, __ATOMIC_RELAXED);
    }
    atom_store(g_dist.frontier, -1, __ATOMIC_RELAXED);
    for (int32_t i = 0; i < kFlagCap; i++)
        atom_store(g_dist.flags[i], 0, __ATOMIC_RELAXED);
    atom_store(g_dist.fatal, 0, __ATOMIC_RELAXED);
    atom_store<int64_t>(g_dist.replay_done, 0, __ATOMIC_RELAXED);
    atom_store<int64_t>(g_dist.started_count, 0, __ATOMIC_RELAXED);
    g_dist.orch_args = orch_args;
    g_dist.rt = rt;
    g_dist.runtime = runtime;
    if (runtime != nullptr) {
        runtime->dist.ccec_heap_base = rt != nullptr ? reinterpret_cast<uint64_t>(rt->gm_heap) : 0;
        runtime->dist.ccec_heap_size = rt != nullptr ? rt->gm_heap_size : 0;
        for (int32_t i = 0; i < 2048; i++) {
            atom_store<int64_t>(runtime->dist.ccec_flags[i], 0, __ATOMIC_RELAXED);
        }
    }

    // Derive the physical-block topology (1 AIC + 2 AIV per block) the same way
    // the centralized scheduler discovers clusters: AIC/AIV cores in worker-index
    // order, AIC[b] paired with AIV[2b] (AIV0) and AIV[2b+1] (AIV1). Followers and
    // anchors use this to address block.won deposits. See §3.1.
    g_dist.num_workers = num_workers;
    int32_t aic_ids[RUNTIME_MAX_WORKER];
    int32_t aiv_ids[RUNTIME_MAX_WORKER];
    int32_t naic = 0, naiv = 0;
    for (int32_t i = 0; i < num_workers && i < RUNTIME_MAX_WORKER; i++) {
        g_dist.layout[i].block_id = -1;
        g_dist.layout[i].lane = LANE_NONE;
        if (runtime->workers[i].core_type == CoreType::AIC) {
            aic_ids[naic++] = i;
        } else {
            aiv_ids[naiv++] = i;
        }
    }
    g_dist.num_blocks = naic;
    for (int32_t b = 0; b < naic; b++) {
        g_dist.layout[aic_ids[b]] = CoreLayout{b, LANE_AIC};
        if (2 * b < naiv) g_dist.layout[aiv_ids[2 * b]] = CoreLayout{b, LANE_AIV0};
        if (2 * b + 1 < naiv) g_dist.layout[aiv_ids[2 * b + 1]] = CoreLayout{b, LANE_AIV1};
        atom_store(g_dist.blocks[b].any_pub, 0, __ATOMIC_RELAXED);
        for (int32_t s = 0; s < kPrivateSlots; s++) {
            atom_store(g_dist.blocks[b].slots[s].state, 0, __ATOMIC_RELAXED);
        }
    }

    if (dist_trace()) {
        fprintf(
            stderr, "[dist] register: num_workers=%d heap_base=%p heap_size=%zu\n", num_workers,
            (void *)g_dist.heap_base, g_dist.heap_size
        );
    }

    // Install the SIGUSR1 deadlock dumper once, but only when diagnostics are
    // opted in (PTO_DIST_WATCHDOG set) — default runs install no signal handler.
    static bool handler_installed = false;
    if (!handler_installed && getenv("PTO_DIST_WATCHDOG") != nullptr) {
        signal(SIGUSR1, dist_dump_state);
        handler_installed = true;
    }

    // Publish the DistGlobal struct address so device workers can wire up
    // their g_dist_ptr at dist_core_main entry. In sim the host BSS `g_dist`
    // is shared across every worker pthread, so its own address is what all
    // AICore workers see; onboard will replace this with a real GM allocation.
    runtime->dist.shared_addr = reinterpret_cast<uint64_t>(&g_dist);

    // Publish all of the above before AICPU wakes workers through their
    // per-core handshake flags.
    atom_thread_fence(__ATOMIC_RELEASE);
    return reinterpret_cast<void *>(&dist_core_main);
}

#if DIST_TRACE_ENABLED
void dist_engine_dump_trace() {
    if (!g_trace_on) return;
    const char *path = getenv("PTO_DIST_SWIMLANE");
    if (path == nullptr || path[0] == '\0') return;
    FILE *f = fopen(path, "w");
    if (f == nullptr) {
        fprintf(stderr, "[dist_engine] cannot open swimlane file %s for write\n", path);
        return;
    }

    auto lane_name = [](int32_t lane) -> const char * {
        switch (lane) {
        case LANE_AIC:
            return "AIC";
        case LANE_AIV0:
            return "AIV0";
        case LANE_AIV1:
            return "AIV1";
        default:
            return "?";
        }
    };

    // Chrome Trace Event Format (https://ui.perfetto.dev / chrome://tracing).
    // Two process groups: pid = block_id is the WALL-clock swimlane; pid =
    // block_id + kCpuPid is a parallel CPU-time swimlane (same spans, width =
    // cpu_us). process_sort_index forces all wall groups above all cpu groups.
    // Dependency arrows (flow events) are emitted only in the cpu group so they
    // stay within the cpu lanes instead of tangling across the wall lanes.
    constexpr int32_t kCpuPid = 1000;
    fprintf(f, "{\n  \"displayTimeUnit\": \"ns\",\n  \"traceEvents\": [\n");
    bool first = true;
    const int32_t nw = g_dist.num_workers;

    // Lane/process name + sort metadata first (so idle lanes still appear).
    for (int32_t c = 0; c < nw && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = g_dist.cores[c];
        if (co.block_id < 0 || co.lane < 0) continue;
        if (!first) fprintf(f, ",\n");
        first = false;
        fprintf(
            f, "    {\"ph\":\"M\",\"name\":\"process_name\",\"pid\":%d,\"args\":{\"name\":\"block%d (wall)\"}}",
            co.block_id, co.block_id
        );
        fprintf(
            f, ",\n    {\"ph\":\"M\",\"name\":\"process_sort_index\",\"pid\":%d,\"args\":{\"sort_index\":%d}}",
            co.block_id, co.block_id
        );
        fprintf(
            f,
            ",\n    {\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":%d,\"tid\":%d,"
            "\"args\":{\"name\":\"%s (core%d)\"}}",
            co.block_id, co.lane, lane_name(co.lane), c
        );
        fprintf(
            f, ",\n    {\"ph\":\"M\",\"name\":\"process_name\",\"pid\":%d,\"args\":{\"name\":\"block%d (cpu)\"}}",
            co.block_id + kCpuPid, co.block_id
        );
        fprintf(
            f, ",\n    {\"ph\":\"M\",\"name\":\"process_sort_index\",\"pid\":%d,\"args\":{\"sort_index\":%d}}",
            co.block_id + kCpuPid, co.block_id + kCpuPid
        );
        fprintf(
            f,
            ",\n    {\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":%d,\"tid\":%d,"
            "\"args\":{\"name\":\"%s (core%d)\"}}",
            co.block_id + kCpuPid, co.lane, lane_name(co.lane), c
        );
        // CPU-group kernel sub-lane (tid = lane + 3): kernel spans live here so a
        // ringbp bar that time-contains its releasing kernel does not nest+hide it.
        fprintf(
            f,
            ",\n    {\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":%d,\"tid\":%d,"
            "\"args\":{\"name\":\"%s·kernel (core%d)\"}}",
            co.block_id + kCpuPid, co.lane + 3, lane_name(co.lane), c
        );
    }

    auto phase_name = [](TracePhase p) -> const char * {
        switch (p) {
        case TracePhase::Kernel:
            return "kernel";
        case TracePhase::Alloc:
            return "alloc";
        case TracePhase::Build:
            return "build";
        case TracePhase::DrainWon:
            return "drain_won";
        case TracePhase::Replay:
            return "replay";
        case TracePhase::RingBp:
            return "ringbp";
        case TracePhase::EfDrain:
            return "efdrain";
        case TracePhase::Commit:
            return "commit";
        default:
            return "?";
        }
    };

    // Index: task_id -> its kernel span location in the CPU group, so a dep edge
    // can anchor an arrow at the producer's and consumer's actual spans.
    struct SpanLoc {
        int32_t pid;
        int32_t tid;
        double ts_us;
        double dur_us;
    };
    // In the CPU group, kernel spans go on a SEPARATE sub-lane (tid = lane +
    // kCpuKernelLane) from the build/ringbp/replay/alloc spans (tid = lane). A
    // ringbp span time-contains the kernel that ends its wait, so on one lane
    // perfetto would nest the kernel inside the ringbp bar and hide it; splitting
    // the kernel onto its own row keeps both visible.
    constexpr int32_t kCpuKernelLane = 3;
    std::vector<SpanLoc> kloc(static_cast<size_t>(kFlagCap), SpanLoc{-1, -1, 0.0, 0.0});
    for (int32_t c = 0; c < nw && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = g_dist.cores[c];
        if (co.block_id < 0 || co.lane < 0) continue;
        for (const TraceEvent &e : co.trace) {
            if (e.phase != TracePhase::Kernel || e.task_id < 0 || e.task_id >= kFlagCap) continue;
            kloc[static_cast<size_t>(e.task_id)] =
                SpanLoc{co.block_id + kCpuPid, co.lane + kCpuKernelLane, e.ts_ns / 1000.0, e.cpu_ns / 1000.0};
        }
    }
    // Index: task_id -> its ringbp span in the CPU group (the arrow head for a
    // slot-release edge anchors at the ringbp's END = when the wait was satisfied).
    std::vector<SpanLoc> rbloc(static_cast<size_t>(kFlagCap), SpanLoc{-1, -1, 0.0, 0.0});
    for (int32_t c = 0; c < nw && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = g_dist.cores[c];
        if (co.block_id < 0 || co.lane < 0) continue;
        for (const TraceEvent &e : co.trace) {
            if (e.phase != TracePhase::RingBp || e.task_id < 0 || e.task_id >= kFlagCap) continue;
            rbloc[static_cast<size_t>(e.task_id)] =
                SpanLoc{co.block_id + kCpuPid, co.lane, e.ts_ns / 1000.0, e.cpu_ns / 1000.0};
        }
    }

    // Duration events: kernel + non-kernel overhead spans, emitted once in the
    // wall group (pid=block) and once in the cpu group (pid=block+kCpuPid).
    for (int32_t c = 0; c < nw && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = g_dist.cores[c];
        if (co.block_id < 0 || co.lane < 0) continue;
        for (const TraceEvent &e : co.trace) {
            const char *ph = phase_name(e.phase);
            char name[64];
            if (e.phase != TracePhase::Kernel) {
                snprintf(name, sizeof(name), "%s#%d", ph, e.task_id);
            } else if (e.func_id >= 0) {
                snprintf(name, sizeof(name), "f%d#%d", e.func_id, e.task_id);
            } else {
                snprintf(name, sizeof(name), "task#%d", e.task_id);
            }
            if (!first) fprintf(f, ",\n");
            first = false;
            // Convert raw ns -> us (swimlane unit) here, at dump time — never on the
            // hot path (see TraceEvent).
            const double ts_us = e.ts_ns / 1000.0;
            const double dur_us = e.dur_ns / 1000.0;
            const double cpu_us = e.cpu_ns / 1000.0;
            fprintf(
                f,
                "    {\"ph\":\"X\",\"name\":\"%s\",\"pid\":%d,\"tid\":%d,\"ts\":%.3f,\"dur\":%.3f,"
                "\"args\":{\"phase\":\"%s\",\"task_id\":%d,\"func_id\":%d,\"core\":%d,\"mc\":%d,\"cpu_us\":%.3f}}",
                name, co.block_id, co.lane, ts_us, dur_us, ph, e.task_id, e.func_id, c, e.multicore, cpu_us
            );
            fprintf(
                f,
                ",\n    {\"ph\":\"X\",\"name\":\"%s\",\"pid\":%d,\"tid\":%d,\"ts\":%.3f,\"dur\":%.3f,"
                "\"args\":{\"phase\":\"%s\",\"task_id\":%d,\"func_id\":%d,\"wall_us\":%.3f}}",
                name, co.block_id + kCpuPid, e.phase == TracePhase::Kernel ? co.lane + kCpuKernelLane : co.lane, ts_us,
                cpu_us, ph, e.task_id, e.func_id, dur_us
            );
        }
    }

    // Flow events: the full static dependency graph. One arrow per dep edge, in
    // the cpu group, from the PRODUCER kernel span's end to the CONSUMER kernel
    // span's start (time always forward: a producer completes before its consumer
    // runs). Click any task and follow arrows backward hop-by-hop to walk the
    // chain "what was this waiting on, and what was THAT waiting on".
    int32_t flow_id = 0;
    for (int32_t c = 0; c < nw && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = g_dist.cores[c];
        if (co.block_id < 0 || co.lane < 0) continue;
        for (const DistCore::DepEdge &de : co.dep_edges) {
            if (de.producer_task < 0 || de.producer_task >= kFlagCap) continue;
            if (de.consumer_task < 0 || de.consumer_task >= kFlagCap) continue;
            const SpanLoc &pr = kloc[static_cast<size_t>(de.producer_task)];
            const SpanLoc &cs = kloc[static_cast<size_t>(de.consumer_task)];
            if (pr.pid < 0 || cs.pid < 0) continue;  // need both kernel spans
            fprintf(
                f, ",\n    {\"ph\":\"s\",\"name\":\"dep\",\"cat\":\"dep\",\"id\":%d,\"pid\":%d,\"tid\":%d,\"ts\":%.3f}",
                flow_id, pr.pid, pr.tid, pr.ts_us + pr.dur_us
            );
            fprintf(
                f,
                ",\n    {\"ph\":\"f\",\"name\":\"dep\",\"cat\":\"dep\",\"id\":%d,\"bp\":\"e\",\"pid\":%d,\"tid\":%d,"
                "\"ts\":%.3f}",
                flow_id, cs.pid, cs.tid, cs.ts_us
            );
            flow_id++;
        }
    }

    // Flow events (cat="slot"): slot-release edges that explain a ringbp's stall.
    // From the END of the occupant kernel's span (the moment it frees the slot) to
    // the END of the waiting ringbp span. Chains with the dep arrows: ringbp
    // --slot--> occupant kernel --dep--> the occupant's fan-in kernels.
    for (int32_t c = 0; c < nw && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = g_dist.cores[c];
        if (co.block_id < 0 || co.lane < 0) continue;
        for (const DistCore::DepEdge &se : co.slot_edges) {
            if (se.producer_task < 0 || se.producer_task >= kFlagCap) continue;  // occupant
            if (se.consumer_task < 0 || se.consumer_task >= kFlagCap) continue;  // ringbp waiter
            const SpanLoc &occ = kloc[static_cast<size_t>(se.producer_task)];
            const SpanLoc &rb = rbloc[static_cast<size_t>(se.consumer_task)];
            if (occ.pid < 0 || rb.pid < 0) continue;
            double tail = occ.ts_us + occ.dur_us;      // occupant kernel end (slot freed)
            const double head = rb.ts_us + rb.dur_us;  // ringbp end (wait satisfied)
            if (tail > head) tail = head;              // keep forward in time
            fprintf(
                f,
                ",\n    {\"ph\":\"s\",\"name\":\"slot\",\"cat\":\"slot\",\"id\":%d,\"pid\":%d,\"tid\":%d,\"ts\":%.3f}",
                flow_id, occ.pid, occ.tid, tail
            );
            fprintf(
                f,
                ",\n    {\"ph\":\"f\",\"name\":\"slot\",\"cat\":\"slot\",\"id\":%d,\"bp\":\"e\",\"pid\":%d,\"tid\":%d,"
                "\"ts\":%.3f}",
                flow_id, rb.pid, rb.tid, head
            );
            flow_id++;
        }
    }

    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    fprintf(stderr, "[dist_engine] swimlane trace written to %s\n", path);
}
#else   // !DIST_TRACE_ENABLED
// Tracing compiled out: keep the public symbol so aicpu_executor.cpp still links.
void dist_engine_dump_trace() {}
#endif  // DIST_TRACE_ENABLED

#endif  // DIST_HOST_ONLY
