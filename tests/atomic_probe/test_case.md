# A5 Cache-Line Cross-Core Probe — Test Case Summary

# goal：当前a5的多核架构里，每个核上的scalar操作的datacache没有多核间的cache coherence处理，而通用CPU是有的，
# 因此：需要探索在a5上基于scalar编程时的多核并发操作正确性问题，分别check ascendc/ccec两种方式，cpu模式作为参考对照，重点测试点需要覆盖多核并发读写同一cacheline的场景。

## 目录结构

```
tests/atomic_probe/
├── test_case.md               本文档
├── ascendc/                   AscendC API 探针 (bisheng -xasc)
│   ├── atomic.asc              AtomicCas 延迟
│   ├── atomic64_verify.asc     64bit 原子性验证 (atomicAdd/atomicMax u32/u64)
│   ├── mb2_flags_clobber.asc   flag cacheline clobber (AtomicMax vs store+dcci)
│   ├── mb8_dcci_seam.asc       dcci publish/observe seam
│   ├── cacheline_blast.asc     atomic blast radius (2B/4B/8B)
│   ├── bypass_dcache_probe.asc WriteGmByPassDCache / ReadGmByPassDCache
│   ├── concurrent_cacheline.asc N-block 并发读写
│   ├── cacheline_stress.asc    tight-loop store+dcci / torn read / mixed
│   ├── dcci_atomic_stress.asc  dcci type survey + st_dev/ld_dev 10k 压力
│   └── _run_asc_probe.sh
├── ccec/                      CCEC 底层 intrinsic 探针 (ccec -x cce)
│   ├── ccec_utils.h            纯 CCEC 工具头 (barrier/st_dev/ld_dev)
│   ├── atomic_cas_probe.cpp    atomicCAS (ccec 版)
│   ├── atomic_cas_host.cpp
│   ├── atomic_blast.cpp        atomicMax blast radius (ccec 版)
│   ├── atomic_blast_host.cpp
│   ├── entire_flush_clobber.cpp dcci(ENTIRE) clobber
│   ├── entire_flush_clobber_host.cpp
│   ├── bypass_dcache_ccec.cpp   st_dev / ld_dev 探针
│   ├── bypass_dcache_ccec_host.cpp
│   ├── dcci_clean_clobber.cpp   dcci clean 步骤 clobber 验证
│   ├── dcci_clean_clobber_host.cpp
│   ├── dcci_seam.cpp            dcci publish/observe (ccec 版)
│   ├── dcci_seam_host.cpp
│   ├── concurrent_stress.cpp    tight-loop 并发压力 (ccec 版)
│   ├── concurrent_stress_host.cpp
│   └── run_all.sh / run_*.sh
└── cpu/                       CPU 对照组 (x86-64, std::thread + std::atomic)
    └── cpu_atomicity.cpp       原子性/torn read/false sharing/spinlock
```
    ├── entire_flush_clobber.cpp dcci(ENTIRE) clobber
    ├── entire_flush_clobber_host.cpp
    ├── bypass_dcache_ccec.cpp   st_dev / ld_dev 探针
    ├── bypass_dcache_ccec_host.cpp
    ├── dcci_clean_clobber.cpp   dcci clean 步骤 clobber 验证
    ├── dcci_clean_clobber_host.cpp
    ├── run.sh / run_bypass_dcache.sh / run_dcci_clean.sh
    └── build/
```

## 两套指令体系

| | AscendC API (ascendc/) | CCEC intrinsic (ccec/) |
|---|---|---|
| 编译方式 | `bisheng -xasc --npu-arch=dav-3510` | `ccec -x cce --cce-aicore-only` |
| 绕过 DCache 写 | `WriteGmByPassDCache<T>(addr, val)` | `st_dev(val, addr, 0)` |
| 绕过 DCache 读 | `ReadGmByPassDCache<T>(addr)` | `ld_dev(addr, 0)` |
| 原子操作 | `AtomicMax` / `AtomicAdd` | `atomicMax` / `atomicAdd` |
| Cache 刷新 | `dcci(ptr, scope, type)` | 同 |
| 关系 | AscendC 是 CCEC intrinsic 的上层封装 | 底层硬件指令 |

两者映射到**同一套硬件指令**（st_dev / ld_dev / atomic），能力完全等价。

---

## 测试用例矩阵与结果

### 1. Atomic Blast Radius（原子操作影响范围）

**文件**: `ascendc/cacheline_blast.asc`
**问题**: AtomicMax 对 gx[0] 操作是否影响同 cache line 的 gx[1..15]？

| Mode | 场景 | 结果 |
|---|---|---|
| 0 Sequential | 先写 sentinel + flush，后 atomic | **PASS** — 邻居完好 |
| 1 Stale-L1 | L1 有脏 0，sentinel 在 HBM，后 atomic | **PASS** — atomic 绕过 L1 |
| 2 ConcurrentAtom | 双方都 atomic，无 sync | **PASS** |
| 3 Reverse(dcci) | atomic 先，后 store+dcci（有 SyncAll） | **PASS** |
| 4 ConcurrentMixed | atomic vs store+dcci 无 sync | 7/10 PASS, 3/10 dcci clobber |
| 5 Byte32(4B) | 字节级验证 uint32 atomic | blast radius = **恰好 4B** |
| 6 ByteHalf(2B) | 字节级验证 half atomicAdd | blast radius = **恰好 2B** |
| 7 Byte64(8B) | 字节级验证 int64 atomic | blast radius = **恰好 8B** |

**结论**: AtomicMax/atomicAdd 的 blast radius = 原子宽度，不扩散到同 cache line 的其他字节。

---

### 2. WriteGmByPassDCache / st_dev 写入 Blast Radius

**文件**: `ascendc/bypass_dcache_probe.asc`, `ccec/bypass_dcache_ccec.cpp`

| Mode | 场景 | 结果 |
|---|---|---|
| 1B write | WriteGmByPassDCache<uint8_t> | blast radius = **1 byte** |
| 4B write | WriteGmByPassDCache<uint32_t> | blast radius = **4 bytes** |
| Read correctness | ld_dev at 1B/2B/4B/8B | **0 errors / 120 reads** |
| Read vs stale L1 | normal=stale, bypass=fresh | ld_dev 始终读 HBM 最新值 |
| Concurrent+atomic | st_dev + AtomicAdd 并发 | **0 clobber** |
| L1 不污染 | bypass read 后 normal read 仍 stale | L1 **CLEAN**（未被污染） |

**结论**: st_dev 的 blast radius = 写入宽度（最小 1 byte），完全不碰 L1。

---

### 3. dcci 指令类型调查

**文件**: `ascendc/dcci_atomic_stress.asc`, `ccec/dcci_clean_clobber.cpp`

A5 上 dcci 的三种 type：

| Type | 名称 | 语义 | dirty L1 clobber 别核 st_dev？ |
|---|---|---|---|
| 0 | CACHELINE_ALL | clean + invalidate | **YES** (7/9 ~ 10/10) |
| 2 | CACHELINE_OUT | clean only (写回) | **YES** (9/9) |
| 3 | CACHELINE_ATOMIC | clean + invalidate | **YES** (10/10) |

**关键结论**:
- A5 上 dcci **没有** invalidate-only（不 clean）的指令。三种 type 都会先 clean（写回 dirty L1 data）再 invalidate
- dirty L1（有 scalar store）+ 任何 dcci → **必然 clobber** 同 cache line 别核的 st_dev 写入
- clean L1（只有读）+ dcci → **安全**（clean 是 no-op）

---

### 4. dcci Clean Clobber 验证

**文件**: `ccec/dcci_clean_clobber.cpp`

| Mode | L1 状态 | dcci 操作 | word[3] (st_dev 写入) | 结果 |
|---|---|---|---|---|
| 0 | DIRTY (scalar store) | dcci(ptr, 0) | 0x0 | **CLOBBERED** (7/9) |
| 1 | CLEAN (只有读) | dcci(ptr, 0) | 0xCAFEBABE | **SURVIVED** (9/9) |
| 2 | DIRTY | dcci(ptr, 0, 2) | 0x0 | **CLOBBERED** (9/9) |
| 3 | DIRTY | 不做 dcci | 0xCAFEBABE | SURVIVED (st_dev 不受影响) |
| 4 | DIRTY | dcci(ptr, 0) | — | **15/15 words 全 clobber** |

**结论**: clean 步骤把整条 64B dirty L1 line 写回 HBM，覆盖了别核通过 st_dev 写到同 line 不同 word 的值。只有 L1 clean 时才安全。

---

### 5. Tight-Loop 并发压力测试

**文件**: `ascendc/cacheline_stress.asc`

| Mode | 场景 | 结果 |
|---|---|---|
| 0 | 2 核 tight-loop store+dcci 不同 word (1000 轮) | **4/4 全 clobber** (w0-w3 = 0) |
| 1 | 3 核 tight-loop store+dcci | **6/6 全 clobber** |
| 2 | Torn read: dcci(flush) 写 + ld_dev 读 | **80/81 reads torn** (99%) |
| 3 | 混合: st_dev vs store+dcci | st_dev **0/4 err**, store+dcci **4/4 clobber** |
| 4 | 积累多 store 后一次 dcci | **8/8 全 clobber** |

---

### 6. st_dev / ld_dev 10K 压力验证

**文件**: `ascendc/dcci_atomic_stress.asc`

| Mode | 场景 | 结果 |
|---|---|---|
| 3 | 3 核 st_dev 各写不同 word, 10000 轮 | **0/16 err**（5 轮中 1 轮 1 word 偶发错） |
| 4 | st_dev 写 16 word + ld_dev 读一致性 | **1194/1194 全 torn** (100%) |
| 5 | st_dev + AtomicAdd 并发 | data **0/15 err**, atomic 正确 |
| 6 | ld_dev 读 word[0] vs word[15] | **4904/7936 mixed** (62%) |

---

## 核心结论

### 写入方式对比

| 方式 | blast radius | 并发安全 | 需要 dcci |
|---|---|---|---|
| scalar store + dcci(CACHELINE_OUT) | **64B（整 cache line）** | ❌ 100% clobber | ✅ 必须 |
| AtomicMax / AtomicAdd | 4B/8B（原子宽度） | ✅ | ❌ |
| st_dev (WriteGmByPassDCache) | **1B~8B（写入宽度）** | ✅ 值正确 | ❌ |

### 读取方式对比

| 方式 | 读到值 | 多 word 一致性 |
|---|---|---|
| normal scalar read (经过 L1) | 可能 stale | 同 line 内一致（同一 L1 line） |
| ld_dev (绕过 L1) | 始终 HBM 最新 | **不保证**（逐 word 读，期间 HBM 可能被别核修改） |
| normal read + dcci(inval) | HBM 最新（reload 后） | 同 line 内一致 |

### dcci 指令确认

A5 dcci 指令**没有** invalidate-only（discard without clean）的形式：

```
dcci(ptr, 0)     = clean + invalidate（写回 dirty + 丢弃 L1）
dcci(ptr, 0, 2)  = clean only（写回，保留 L1）
dcci(ptr, 0, 3)  = clean + invalidate（同 type=0）
dcci(ptr, 1)     = clean + invalidate ENTIRE_DATA_CACHE
```

所有 dcci 形式都会先 clean（写回 dirty 数据）。对于 dirty L1 line，clean 写回整条 64B line（含 stale 邻居 word），会 clobber 别核通过 st_dev 写到 HBM 的值。

### 安全规则

1. **同一 cache line 多核并发写不同 word**: 禁止用 store+dcci，必须用 st_dev 或 AtomicMax
2. **st_dev 写入单 word**: 值可靠，与 atomic 不冲突
3. **ld_dev 读取多 word 快照**: 需要 SyncAll 或 flag 同步后才能读，否则可能看到新旧值混合
4. **dcci(inval) 在 dirty L1 上**: 会 clobber 别核数据，不安全
5. **dcci(inval) 在 clean L1 上**: 安全（clean 是 no-op）
6. **producer-consumer 通信**: producer 用 st_dev 写 + AtomicMax 设置 flag，consumer 用 ld_dev 轮询 flag 后 ld_dev 读数据——全程不碰 dcci
