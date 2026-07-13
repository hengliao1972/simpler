# A5 DCCI、`st_dev` 与 atomic 使用手册

## 适用范围

本文面向当前仓库在 A5（`dav-3510`）、本机 CANN 9.1 上使用 scalar GM、DCCI、bypass
load/store 和 atomic 的代码。API 定义来自本机 CANN 头文件，行为结论来自
`tests/atomic_probe/` 的 AscendC/CCEC 同构用例与 device 0 实测。

本文刻意区分三类信息：

- **本机接口事实**：本机 CANN 头文件能直接确认的 API、参数和调用映射；
- **当前实测**：本仓精确用例已经在当前 A5 上复现或通过的行为；
- **工程规则**：为避免依赖未公开硬件细节而采用的保守约束。

除非另有说明，本文的“禁止并发”表示必须 100% 按不安全场景处理，不表示每次运行都有
100% 的 mismatch 发生率。

## 结论先行

1. A5 scalar data cache 不能按 CPU 多核 coherent cache 使用。跨核共享数据必须以 64B
   cacheline 为最小所有权和隔离单位，不能只按某个 4B/8B 变量推理。
2. 当前 DCCI 没有已验证的、按指定地址执行 **invalidate-only** 的模式。现有 DCCI 接口是
   `DataCacheCleanAndInvalid`/`dcci`；clean line 上能观察到失效效果，dirty line 上能观察到
   writeback 效果。A5 另有无参 `asc_dci()`/`dci()`，但本机头文件没有公开其作用域和同步语义，
   当前也没有精确用例，因此不能把它当作单 cacheline invalid-only 使用。
3. 即使独立 atomic phase 已让两个核严格串行，只要一个核持有包含 atomic 目标旧值的 ordinary
   stale dirty line，另一个核更新该目标后，前者再对这条 64B line 执行 DCCI，仍可能用旧快照覆盖
   已经成功完成的 atomic 更新。
4. 基于上述实测，本仓采用保守布局规则：atomic 控制字与被 DCCI 的 data 分 cacheline；关键 atomic
   默认一变量独占一条 64B line；其整条 line 按 atomic-only 管理，不混入普通 scalar store/DCCI。
5. `st_dev`/`WriteGmByPassDCache` 绕过 scalar DCache，但不因此获得跨核 coherence、原子性或
   repeated-store 精确终值保证。当前单 AIV 与 multi-AIV 场景都已复现终值回退；测试仍未确定
   根因就是编译器或硬件 store 重排。
6. AscendC/CCEC 单 AIV 独立压力已经排除跨核因素：line1 同址、257 次写后仅一次 DSB 分别复现
   `3/2000000` 与 `4/2000000` mismatch；同一个 AIV 每轮依次写 line1/line2 时分别为
   `83821/1000000` 与 `78130/1000000`。对应的逐写 DSB 控制均为 0。准确表述是“当前精确终值会
   回退，跨核不是必要条件”，不能把这一现象直接命名为已查明的底层乱序机制。
7. 工程调用默认显式选择 `SINGLE_CACHE_LINE`。在获得专项精确证据前，本仓并发协议默认禁用
   `ENTIRE_DATA_CACHE`；后者的 scope 是发指令核整个 data cache，可能把 SINGLE dirty-writeback
   的风险扩大到与当前目标地址无关的 entry，但该扩大风险目前是实现定义与实测机制的组合推论。
8. DSB 是当前核的 Data Synchronization Barrier，不是跨核 barrier，也不提供 cache coherence。
   跨核交权需要独立 atomic phase/lock；AIV 会合需要对应同步原语，并且写发布前仍要先 DSB。
9. 只要某条 data line 至少有一个核可能写，多个核对该 line 的 ordinary scalar 读写和 DCCI 就必须
   100% 按不安全处理，整个 64B line 放进同一个排他所有权区。仅当数据在全部并发读者生命周期内
   确定不可变时，才能把纯读共享作为另一种协议单独论证。

## 1. 三类访问的本质区别

| 方式 | AscendC API | CCEC/原始接口 | 使用的本核 scalar DCache | 主要能力 | 不能保证 |
|---|---|---|---|---|---|
| 普通 scalar load/store | `__gm__` 指针或 `GlobalTensor` 普通读写 | 普通 GM load/store | 是 | 本核缓存访问 | 跨核 coherence、自动看到新值 |
| DCCI | `DataCacheCleanAndInvalid` | `dcci` | 管理 DCache entry | clean/writeback 与 invalidation | 纯 invalid、跨核互斥、保护 atomic |
| bypass load/store | `ReadGmByPassDCache` / `WriteGmByPassDCache` | `ld_dev` / `st_dev` | 绕过 | 直接观察/修改 GM 路径 | atomic、跨核顺序、repeated store 终值 |
| atomic RMW | `AtomicAdd/Max/Min/Cas/Exch` | `atomicAdd/Max/Min/CAS/Exch` | 不使用普通 scalar load/store 封装 | 目标 word 的原子读改写 | 邻接 word、整条 line、DCCI writeback |

表中的“绕过 DCache”来自本机 `kernel_scalar.h`：AscendC 的 1/2/4/8B
`WriteGmByPassDCache`、`ReadGmByPassDCache` 分别直接调用 `st_dev`、`ld_dev`。这只说明访问路径，
不能外推为跨核内存模型保证。

## 2. DCCI API 与参数

### 2.1 AscendC

本机公共接口为：

```cpp
DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE,
                         DcciDst::CACHELINE_OUT>(globalTensor);
```

也可以选择 `CacheLine::ENTIRE_DATA_CACHE`，或将第三模板参数设为
`DcciDst::CACHELINE_ALL`/`CACHELINE_ATOMIC`。两模板参数版本最终调用两参数 `dcci`。

当前 AscendC selector 探针为了直接控制编码，调用的是 raw `dcci`，没有直接把公共 C++ wrapper
作为单独编译面回归；但本机 `kernel_reg.h` 显示三模板参数 wrapper 直接下沉为三参数 `dcci`，省略
`DcciDst` 的 wrapper 直接下沉为两参数 `dcci`。因此本文把公共 API 的调用映射列为本机实现事实，
把 selector 行为列为 raw 指令实测，不混为同一种证据。

### 2.2 CCEC

```cpp
dcci(address, SINGLE_CACHE_LINE);
dcci(address, SINGLE_CACHE_LINE, CACHELINE_ALL);
dcci(address, SINGLE_CACHE_LINE, CACHELINE_OUT);
dcci(address, SINGLE_CACHE_LINE, CACHELINE_ATOMIC);
```

当前 A5 可用编码如下：

| 参数 | 编码 | 本机头文件能确认的含义 |
|---|---:|---|
| `SINGLE_CACHE_LINE` | 0 | 处理地址对应的单个 data-cache cacheline entry |
| `ENTIRE_DATA_CACHE` | 1 | 处理整个 data cache；本机 A5 C API wrapper 传空地址 |
| `CACHELINE_ALL` | 0 | selector 编码 0；更细硬件分类未公开 |
| `CACHELINE_OUT` | 2 | selector 编码 2；更细硬件分类未公开 |
| `CACHELINE_ATOMIC` | 3 | selector 编码 3；更细硬件分类未公开 |

本机 A5 intrinsic 的第三参数默认值为 0；当前两参数用例的实测行为也与显式
`CACHELINE_ALL` 一致。因此不要把“两参数 DCCI”理解为另一种纯 invalid 指令。

### 2.3 selector 不等于动作开关

`ALL`、`OUT`、`ATOMIC` 的名称不能解释为 clean-only、invalidate-only，或“保护 atomic”。本机
公开代码没有给出三类 entry 的完整硬件分类规则，当前精确用例得到的是：

| 被测本核 entry | DEFAULT / 显式 ALL | 显式 OUT | 显式 ATOMIC | NO DCCI |
|---|---|---|---|---|
| ordinary clean stale line | 后续 normal load 读到 GM 新值 | 同左 | 同左 | 保持 stale |
| ordinary dirty stale line | 显式 ALL 发布；DEFAULT 仅由编码映射为 ALL，dirty matrix 未单列 | dirty line 被发布 | dirty line 被发布 | dirty 值未发布 |
| 本核 dirty 快照含目标旧值，GM 已由其他核 atomic 更新 | 显式 ALL 用旧快照覆盖；DEFAULT 仅由编码映射 | 旧快照覆盖 atomic 新值 | 旧快照覆盖 atomic 新值 | atomic 新值保留 |

clean-reader 用例中 DEFAULT 与三个显式 selector 均为
`100 fresh / 0 stale / 0 other / 0 gm_bad`，说明当前场景下 DCCI 后普通读取取得 GM 新值，而且
clean line 没有反向破坏 GM。dirty-line matrix 显式验证的是 ALL、OUT、ATOMIC 三种 selector，
三者都会发布当前 ordinary dirty line；DEFAULT 等于 ALL 是本机默认编码映射，不是 dirty matrix 的
额外 mode。由此只能得出：当前没有可依赖的 DCCI invalid-only selector。

这不等于 DCCI “没有 invalidation 效果”；准确含义是 **DCCI 没有已验证的按地址纯 invalid、
不回写 dirty line 的使用方式**。

### 2.4 A5 另有无参 DCI，但当前不能替代按地址 invalid

本机 A5 C API 声明了：

```cpp
asc_dci();  // 下沉为 dci() / __builtin_cce_dci
```

这是与 DCCI 分开的 A5 专用 primitive，不能省略不提。但它没有地址、scope 或 selector 参数，本机
头文件也没有给出其 data-cache entry 范围、是否隐含 barrier、以及多核交互语义；当前仓库尚无
针对性精确上板用例。因此只能确认“指令存在”，不能声称它等价于“失效指定地址对应的 64B line”，
也不能把本文的共享协议改为依赖 `dci()`。在完成专项查证和测试前，按地址刷新仍只能使用受所有权
保护的 SINGLE DCCI，并承担其可能 clean dirty line 的语义。

### 2.5 `SINGLE_CACHE_LINE` 与 `ENTIRE_DATA_CACHE`

`SINGLE_CACHE_LINE` 将影响限制在指定地址对应的本核 entry。它仍会按 64B 整条 line 处理，不能
只 clean/invalidate 其中一个 4B slot。

`ENTIRE_DATA_CACHE` 的 scope 是发指令核整个 data cache。本机 A5 C API 的
`asc_dcci_entire_all/out/atomic` wrapper 给 `dcci` 传空地址；C++
`DataCacheCleanAndInvalid<..., ENTIRE_DATA_CACHE, ...>` 仍把 `GlobalTensor` 地址向下传递，但 scope
参数仍是 ENTIRE，不能把该地址解释为 SINGLE 式作用边界。

ENTIRE 不会直接操作其他核的 cache，但它可能把本核持有的任意 stale dirty line 写回 GM。如果
该 line 的 GM 已被其他核用 atomic 或 `st_dev` 更新，本核旧快照就可能覆盖这些新值。

当前 `dcci_atomic_clobber` 精确证明了 SINGLE DCCI 的同-line覆盖；ENTIRE 的扩大风险来自本机
“整个 data cache”定义与该 writeback 机制的组合推论。现有 `entire_flush_clobber` 还不是足以把
每种 selector、entry 和时序全部定量化的精确用例，因此不能写成“只要执行 ENTIRE，所有 atomic
必然损坏”。本仓的保守工程政策是在多核并发协议中默认禁用 ENTIRE，直到相同业务时序有专项精确
用例。确认发指令核没有共享 dirty line、执行期间没有其他核更新相关 line，只能作为必要的审计前提，
当前证据尚未证明这些条件足以构成通用硬件安全保证。

## 3. `st_dev` / bypass load-store

### 3.1 API

AscendC：

```cpp
WriteGmByPassDCache<uint32_t>(address, value);
uint32_t value = ReadGmByPassDCache<uint32_t>(address);
```

本机 A5 支持 1/2/4/8B 整数类型。CCEC 探针使用 `st_dev_b8/b16/b32/b64`、
`ld_dev_b8/b16/b32/b64` 等便捷封装，底层对应 `st_dev`、`ld_dev` builtin。

### 3.2 必须避免的误解

- bypass DCache 不等于 atomic；两个核写同一个 word 仍是数据竞争。
- bypass DCache 不等于 coherent；它不会让其他核已经缓存的 ordinary clean line 自动更新。
- bypass DCache 不等于 store 顺序已经满足业务协议；发布前仍需要 DSB。
- DSB 只等待当前核此前的相应 memory access，不会让另一个核自动失效 cache，也不是 AIV 会合。
- 将两个 writer 拆到不同 64B line，仍不足以保证 repeated `st_dev` 的最后一轮值。

### 3.3 当前实测边界

当前固化的 AscendC/CCEC 单 AIV 独立压力每次 kernel 只启动 block0，写者和 bypass reader 相同，
不调用跨核同步。每个 launch 含 100 trial，每个 slot 每 trial 连续写 257 次：

| 单 AIV 路径 | CCEC | AscendC |
|---|---:|---:|
| line1 同址，仅 loop-end DSB | 4/2000000 | 3/2000000 |
| 每轮依次写 line1 / line2，仅 loop-end DSB | 78130/1000000 | 83821/1000000 |
| line1 同址，每次写后 DSB | 0/2000000 | 0/2000000 |
| 每轮依次写 line1 / line2，每个 slot 每次写后 DSB | 0/1000000 | 0/1000000 |

旧 `st_dev_same_line` 内嵌 CCEC 单 AIV mode 的 `0/2000` 是低压力历史样本，不能继续作为当前边界。
新用例另提供 line0/line2 单址和 line0/line1 双址 mode，用于显式扫描 allocation 内偏移。

两个 AIV 并发、各自只写自己的 64B line、每个地址只有一个 writer、循环末 DSB：

| data line | 活跃 block | CCEC | AscendC |
|---|---|---:|---:|
| line0 / line1 | block0 + block1 | 41484/100000 | 42165/100000 |
| line0 / line1 | block0 + block2 | 39974/100000 | 37320/100000 |
| line1 / line2 | block0 + block1 | 0/100000 | 46/100000 |
| line1 / line2 | block0 + block2 | 10/100000 | 5/100000 |

所以不能说“一个地址只有一个 writer 就安全”，也不能继续把问题限定为跨核。准确结论是：
**单个 AIV、同一写者与读者、同一 4B 地址的 repeated `st_dev` 已经出现精确终值回退；多 AIV 和
跨核会合不是必要条件。** 单/双地址和 allocation 内 line offset 的错误频率明显不同，但当前测试
没有证明编译器重排、硬件 store queue、cache bank/set 或其他机制中的任何一个是根因。

### 3.4 使用建议

1. `st_dev` 适合绕过本核 scalar DCache 的单次数据传输或诊断读取，不应直接当作多 AIV 共享状态的
   原子寄存器。
2. 如果业务要求“最后一次赋值必须精确保留”，`AtomicExch` 是优先候选，不要直接用 repeated
   `st_dev` 模拟 exchange。当前 AtomicExch 只有旧同构三路径各 `0/4000` 的证据；替换后仍必须使用
   atomic-only 布局，并按实际业务拓扑、次数和时序做同构精确回归。它也不能抵御同-line stale
   dirty DCCI writeback。
3. `st_dev` 写完、通过 atomic flag/phase 交权之前，至少执行一次 `DataSyncBarrier<MemDsbT::ALL>()`
   或 `dsb(DSB_ALL)`。
4. 逐写 DSB 在当前单 AIV 同址/双址压力中分别为两端 `0/2000000` 与 `0/1000000`，旧多 AIV
   同-line对照也为 `0/4000`。它在已测样本中抑制了错误，但仍只是控制路径，不是通用 ISA 正确性
   证明；性能代价也必须由业务 workload 单独评估。
5. 不要在同一 64B line 上混用 ordinary scalar store 和 bypass/atomic 更新；后续 DCCI 可能用
   ordinary dirty 快照覆盖 bypass/atomic 新值。

## 4. atomic API 与边界

### 4.1 AscendC API

```cpp
AtomicAdd(address, value);
AtomicMax(address, value);
AtomicMin(address, value);
AtomicCas(address, compareValue, newValue);
AtomicExch(address, newValue);
```

本机 A5 AscendC 实现支持：

- `AtomicAdd/Max/Min`：`uint32_t/int32_t/uint64_t/int64_t/float`；
- `AtomicCas/Exch`：`uint32_t/uint64_t`。

CCEC 对应使用 `atomicAdd`、`atomicMax`、`atomicMin`、`atomicCAS`、`atomicExch`。当前用例重点验证
了 32-bit CAS、Add、Max 与 Exch；未覆盖的类型和组合不能由此自动外推。

这些 AscendC/CCEC atomic API 没有 C++ 风格的 memory-order 参数。CCEC raw API 的模板默认项是
数值为 0 的 L2 cache hint `L2_CACHE_HINT_NORMAL_FV`，不是 `relaxed/acquire/release` 内存序。
因此不能从“默认值为 0”推导 relaxed 语义；本手册只使用精确测试已经覆盖的 DSB 与交权时序。

### 4.2 atomic 保证的是目标操作，不是整条 line 的所有权

atomic RMW 成功，只能说明该次目标 word 更新成功。它不能阻止另一个核随后把包含该 word 旧值的
dirty 64B 快照整体写回。

当前 `dcci_atomic_clobber` 的精确时序为：

1. 核0普通 load 完整 data line，再普通 store 邻接 slot，留下包含 atomic 目标旧值的 dirty line；
2. 核0通过独立 atomic phase 交权；
3. 核1对目标 slot 成功执行 `AtomicExch`，再交权回核0；
4. 核0对原 dirty line 执行 DCCI；
5. ALL、OUT、ATOMIC 三种 selector 都把旧 dirty 快照写回，AtomicExch 新值被覆盖。

这不是 atomic 指令执行失败，而是 atomic 完成后的新值又被整-line writeback 覆盖。本用例的
ready/done phase 位于独立 line；它证明即使独立 atomic phase 已严格串行两个核，放在 dirty data
line 内的 atomic target 仍会被旧快照覆盖。它没有直接测试“lock 自身与 data 共 line”的布局。

### 4.3 atomic cacheline 布局规则

本仓默认保守规则：

- atomic 所在整条 64B line 必须是 atomic-only line；
- 禁止任何核对其中任意 slot 执行普通 scalar store；
- 禁止对该 line 执行 SINGLE DCCI；
- atomic 控制 line 与受保护 data line 必须分离；
- 不能在存在 ENTIRE DCCI 的并发区域中仅依赖地址分-line隔离。

这些规则用于规避已经实测的 stale dirty writeback 风险，不表示当前用例已经证明它们对所有
atomic 类型和所有硬件时序都是必要且充分条件。clean atomic-only line 上执行 DCCI 也没有专项用例。

默认建议：

- 每个关键 lock、phase、ready、done、引用计数各自独占一条 64B line；
- host/device allocation 首地址和每个 offset 都实际检查 64B 对齐，不只依赖结构体视觉布局；
- 独立协议域的 atomic 不合并到同一 line，避免伪共享和生命周期耦合。

允许在充分审计后放宽：多个变量可以共享一条 **纯 atomic** line，但所有访问必须始终是 atomic，
整条 line 永不进入普通 scalar store/DCCI 路径。当前两个 AIV 对同-line不同 slot 重复
`AtomicExch` 的三组路径均为 `0/4000`，这是该特定模式的支持证据，不是所有 atomic 类型和时序的
通用证明。

## 5. 推荐的 cacheline 所有权协议

### 5.1 内存布局

概念布局如下；真实 device allocation 仍需运行时校验：

```cpp
struct alignas(64) AtomicLine {
    uint32_t value;
    uint8_t padding[60];
};

struct alignas(64) DataLine {
    uint32_t words[16];
};

struct SharedState {
    AtomicLine lock;       // atomic-only，禁止 DCCI/普通 store
    AtomicLine ready;      // atomic-only
    AtomicLine done;       // atomic-only
    DataLine data;         // 由 lock 保护，DCCI 只允许 SINGLE
};
```

如果将多个 atomic 合并到一条 line，必须把整条 line 视为一个协议对象，而不是多个互不相关的 4B
变量。

### 5.2 普通 scalar data + atomic lock

以下是基于现有边界给出的建议设计模板，当前探针尚未端到端验证多轮 owner 迁移、完整
acquire/release 语义、异常退出或多读者协议；落到业务前必须补同构测试。

写者推荐时序：

1. 使用独立 atomic line 获取 data line 的唯一所有权；
2. DSB，确认 acquire 之前的 atomic/访问完成；
3. 对 data 执行 SINGLE DCCI，再 DSB，使本核不继续使用旧 cache entry；
4. 普通 scalar load/store data；整个 64B line 期间只允许当前 owner 访问；
5. 对 data 执行 SINGLE DCCI，再 DSB，发布当前 dirty line；
6. 在独立 atomic line 上释放所有权。

读者推荐时序：

1. 获取同一个独立 atomic lock；
2. DSB；
3. 对 data 执行 SINGLE DCCI，再 DSB；
4. 普通 scalar load data；
5. 释放 lock。

atomic acquire 本身不会替当前核刷新 data cache，所以第 3 步不能省略。DCCI 可能 clean dirty line，
所以它必须在取得整条 data line 的唯一所有权之后执行。若需要并发多读者，必须另行设计 reader
生命周期和 writer 排他阶段；不能直接让多个读者各自 DCCI，同时允许 writer 修改同一 line。

### 5.3 bypass data + atomic phase

最小发布顺序为：

```text
writer: st_dev(data) -> DSB -> atomic publish phase
reader: observe atomic phase -> DSB -> ld_dev(data)
```

该顺序避免普通 scalar stale cache，并已在简单 publish/observe 用例中使用。但它不能修复本手册
记录的单 AIV 或多 AIV repeated `st_dev` 终值问题。若同一地址需要多次覆盖且最后值必须精确，可评估
`AtomicExch`，并为该业务建立独立精确压力用例。

## 6. 场景判定表

| 场景 | 当前结论 | 工程处理 |
|---|---|---|
| clean stale line，其他核更新 GM，随后 SINGLE DCCI | DEFAULT/ALL/OUT/ATOMIC 后读取 fresh，GM 未被破坏 | 可在严格交权后用于刷新 |
| dirty stale line，其他核 atomic 更新同 line，随后 SINGLE DCCI | atomic 新值被旧快照覆盖 | 禁止；atomic 与 data 分 line |
| dirty data 与 atomic 位于不同 line，随后 SINGLE DCCI(data) | 当前八模式对照中 atomic 新值保留 | 推荐基础隔离方式 |
| dirty data 与 atomic 分 line，但执行 ENTIRE DCCI | scope 定义可覆盖其他 entry；结合 SINGLE writeback 得到理论风险，尚无精确 clobber 证据 | 本仓并发区默认禁用，等待专项用例 |
| 单 AIV 独自 repeated `st_dev`，仅 loop-end DSB | 同址低频、双址高频复现终值错误；两种前端一致 | 不能用于必须保留最后值的状态 |
| 单 AIV repeated `st_dev`，每次写后 DSB | 当前同址/双址压力均为 0 mismatch | 仅作已测控制；不升级为通用保证，评估性能 |
| 两 AIV repeated `st_dev`，写同 line不同 slot | 稳定复现终值错误 | 禁止用于精确共享状态 |
| 两 AIV repeated `st_dev`，各写独占 line | 高频或低频复现终值错误 | 分 line 仍不能当修复方案 |
| 两 AIV `AtomicExch`，同/分 line | 当前三路径均 `0/4000` | 可作已测 atomic 对照，仍遵守 atomic-only line |
| 多个纯 atomic word 共 line | AtomicExch 特定压力未复现问题 | 可审计后使用；关键变量仍建议独占 line |

## 7. DSB 与跨核同步

本机调用链为：

```text
DataSyncBarrier<MemDsbT::ALL>()
  -> DataSyncBarrierImpl<MemDsbT::ALL>()
  -> dsb(DSB_ALL)
  -> __builtin_cce_dsb
```

本机 intrinsic 对 `DSB_ALL` 的注释是等待所有 memory access instructions。它只约束发指令核，不能：

- 让其他核的 scalar cache 自动失效；
- 替代 atomic lock/phase；
- 替代 `SyncAll<true>()` 或对应 FFTS 会合；
- 把 `st_dev` 变成 atomic；
- 阻止后续 DCCI stale dirty writeback。

推荐把“当前核完成”和“跨核交权”明确拆成两步：先 DSB，再写 atomic phase；接收方先观察 atomic
phase，再 DSB/DCCI 后访问数据。

## 8. CCEC 编译器自动 DCCI

本机 `ccec -mllvm -print-all-options` 显示：

```text
cce-aicore-dcci-insert-for-scalar = 1 (default: 1)
cce-aicore-dcci-before-kernel-end = 1 (default: 1)
```

CCEC runner 对全部 probe 显式关闭；AscendC runner 当前只对 `mb8_dcci_seam`、
`dcci_atomic_clobber`、`st_dev_separate_line_stress`、`st_dev_single_core_stress` 定向关闭：

```text
-mllvm -cce-aicore-dcci-insert-for-scalar=false
-mllvm -cce-aicore-dcci-before-kernel-end=false
```

否则编译器自动插入可能改变待测时序。业务代码如果依赖精确的 cacheline 所有权协议，也必须确认
实际编译选项和产物，不能只从源码中“没有显式 DCCI”推断运行时没有 DCCI。

## 9. 代码评审检查表

- [ ] 每个跨核共享对象都列出了 64B cacheline 归属，而不只是变量地址。
- [ ] atomic line 与普通 scalar/DCCI data line 完全分离。
- [ ] 关键 atomic 默认独占一条 64B line；合并时整条 line 保证 atomic-only。
- [ ] 没有任何 ordinary scalar store 写入 atomic line 的邻接 slot。
- [ ] 工程中的 DCCI scope 均显式选择，默认政策是 `SINGLE_CACHE_LINE`。
- [ ] 并发路径没有 `ENTIRE_DATA_CACHE`；若保留，已有专项精确测试，不把静态 quiescence 审计当成硬件证明。
- [ ] 没有把 OUT/ATOMIC selector 当成 invalid-only 或 atomic 保护开关。
- [ ] 没有在缺少 scope/order 证据时把无参 `dci()` 当成按地址 invalid-only。
- [ ] DCCI 只在取得被处理 data line 的唯一所有权后执行。
- [ ] `st_dev` 发布前有 DSB，跨核交权使用独立 atomic phase/lock。
- [ ] 没有把 `SyncAll`、DSB、DCCI、atomic 四者互相替代。
- [ ] repeated `st_dev` 的最终值使用精确值判定，不用范围判断或“出现错误即通过”。
- [ ] 压力测试覆盖多个 allocation 内 line offset、多个逻辑 block 映射和足够样本数。
- [ ] host 精确检查参与核、marker、guard、首错、每 slot 错误和最终 GM 快照。
- [ ] CCEC 是否自动插入 DCCI 已通过编译选项或产物确认。

## 10. 当前证据入口

- 总目标、判定边界与最新结论：[`test_case.md`](test_case.md)
- device 0 定量记录：[`ATOMIC_MINIBENCH_ONBOARD_LOG.md`](../ATOMIC_MINIBENCH_ONBOARD_LOG.md)
- clean line selector：`ascendc/mb8_dcci_seam.asc`、`ccec/dcci_seam.cpp`
- dirty line 覆盖 atomic：`ascendc/dcci_atomic_clobber.asc`、`ccec/dcci_atomic_clobber.cpp`
- repeated `st_dev` 同-line：`ascendc/st_dev_same_line.asc`、`ccec/st_dev_same_line.cpp`
- repeated `st_dev` 单 AIV 独立压力：`ascendc/st_dev_single_core_stress.asc`、
  `ccec/st_dev_single_core_stress.cpp`
- repeated `st_dev` 分-line独立压力：`ascendc/st_dev_separate_line_stress.asc`、
  `ccec/st_dev_separate_line_stress.cpp`
- AtomicExch 同构对照：`ascendc/atomic_exch_same_line.asc`、`ccec/atomic_exch_same_line.cpp`

本机 CANN 定义依据：

- `tools/bisheng_compiler/lib/clang/15.0.5/include/cce_aicore_intrinsics.h`
- `tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_simt_dav3510_intrinsic.h`
- `x86_64-linux/asc/impl/basic_api/kernel_scalar.h`
- `x86_64-linux/asc/impl/basic_api/kernel_reg.h`
- `x86_64-linux/asc/impl/basic_api/dav_3510/kernel_operator_atomic_impl.h`
- `x86_64-linux/asc/impl/c_api/instr_impl/npu_arch_3510/cache_ctrl_impl/asc_dcci_impl.h`
- `x86_64-linux/asc/include/c_api/cache_ctrl/cache_ctrl.h`
- `x86_64-linux/asc/impl/c_api/instr_impl/npu_arch_3510/cache_ctrl_impl/asc_dci_impl.h`

后续若新用例推翻当前行为，优先更新“当前实测”和“场景判定表”；不要把一次干净运行升级成硬件
保证，也不要把一次错误运行直接扩大为所有芯片、selector、数据宽度和时序的通用结论。
