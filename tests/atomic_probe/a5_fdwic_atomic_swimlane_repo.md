# A5 FDWIC Paged Attention 安装与复现指南

## 1. 目标、边界与已验证结论

本文记录在真实 A5 开发板上安装用户态依赖，并复现以下 Case1 的完整过程：

~~~text
examples/a5/fully_distributed_within_core/paged_attention_unroll/
test_paged_attention_unroll.py
~~~

范围严格限定为：

- 平台仅为 A5Sim 和 A5；
- runtime 仅为 fully_distributed_within_core；
- Case 仅为 Case1；
- Python 始终使用 $HOME/.venv；
- CANN 优先且固定使用 9.1 weekly 20260708；
- 性能口径是全局第一个 Submit 开始到最后一个 Submit 结束。

本文不覆盖其他测试目录、其他 runtime、A2/A3、L3 或整段 device wall time。
A5Sim 用于功能和调度流程验证；5.6 ms 基线只从真实 A5 生成的
l2_swimlane_records.json 中读取。直接 atomic 逐调用记录和等待区 poll 精确计数只在
`--enable-l2-swimlane 4` 开启；它们用于定位真实 PA 的 scalar atomic 分布，
不能替代 level 1 到 3 的 phase-only 性能基线。

### 已验证环境

| 项目 | 本次验证值 |
| --- | --- |
| 验证日期 | 2026-07-17（phase 基线）、2026-07-18（level-4 atomic） |
| 芯片 | Ascend950PR_958b |
| 设备 | /dev/davinci0 |
| Driver | 7.0.t9.0.B798，ascendhal 7.35.23 |
| CANN | 9.1.0 weekly 20260708 |
| CCEC | clang 15.0.5 |
| AICPU 交叉编译器 | Do-Compiler 7.3.0 |
| A5 计算核 | 32 CUBE + 64 VECTOR，共 96 条 swimlane |
| AICPU 用户池 | 5，OCCUPY 掩码 0x3e |
| Python | 3.12.3 |
| PyTorch | 2.6.0+cpu |
| pytest | 7.4.4 |
| GCC 15 | 15.0.1，Ubuntu 15-20250404-0ubuntu1 |
| PTO-ISA | ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8 |
| simpler 分支 | real-pa-atomic-swimlane |
| 历史 phase 实测 HEAD | 52ca4f5eba343c2f7b7a3a743e575cb9308d128f |
| schema-v3 迁移基线 HEAD | 5274945b（迁移改动基于此展开） |

系统的 /etc/os-release 标签为 Ubuntu 20.04.6，但实际
getconf GNU_LIBC_VERSION 输出 glibc 2.39。判断 GCC 15 二进制兼容性时，
应以实际 glibc 和 ldd 结果为准，不能只看发行版标签。

### 已验证结果

| 检查项 | 结果 |
| --- | --- |
| A5Sim Case1 | PASSED，约 71.62 s |
| A5 Case1 正确性 | PASSED |
| A5 level-1 phase swimlane Case1 | PASSED，pytest 约 85.48 s |
| A5 level-4 atomic swimlane Case1 | PASSED，pytest 81.90 s，96 核 schema v3 闭合 |
| 每核 Submit | 1280 个，task id 为 0 到 1279 |
| 全局首个至末个 Submit | 5.642245 ms |
| 排除 task 0 分配后的 kernel Submit | 5.635263 ms |
| 每核 Submit span 中位数 | 5.5725575 ms |
| 历史参考值 | 5.577570 ms，commit dbbf621ac2d1cf162d0807e170c042212d067e51 |

因此，用户关注的约 5.6 ms 基线已经复现。pytest wall time 和日志中的整段
device wall time 不属于这一性能口径。

### 必须包含的源码状态

复现使用的仓库版本必须同时包含以下三处源码调整：

1. Case1、Case2、Case3 不再硬编码 block_dim=36，只保留
   aicpu_thread_num=4，由 A5 平台自动解析实际 block 数；
2. 旧 Driver 的 HAL 和 DSMI 都不支持 CPU_TOPO、返回 65534 时，
   允许经过双重校验的 flat OCCUPY 回退；
3. flat 回退只有在 OCCUPY 的 popcount 与
   ACL_DEV_ATTR_AICPU_CORE_NUM 完全相等时才接受，否则保持失败关闭。

对应文件为：

~~~text
examples/a5/fully_distributed_within_core/paged_attention_unroll/
test_paged_attention_unroll.py
src/a5/platform/onboard/host/aicpu_topology_probe.cpp
src/a5/platform/onboard/host/aicpu_topology_probe.h
~~~

在仓库根目录执行以下检查。第一条应无输出，第二条应命中：

~~~bash
TEST_FILE=examples/a5/fully_distributed_within_core/paged_attention_unroll/test_paged_attention_unroll.py

if rg -n '"block_dim"[[:space:]]*:[[:space:]]*36' "$TEST_FILE"; then
    echo "ERROR: 当前 revision 仍硬编码 block_dim=36"
    exit 1
fi

rg -n 'ACL_DEV_ATTR_AICPU_CORE_NUM|flat OCCUPY fallback' \
    src/a5/platform/onboard/host/aicpu_topology_probe.cpp
~~~

## 2. 系统与设备前置检查

以下命令都以普通用户执行，不需要 sudo。Driver 和 firmware 是板端系统级
前置条件，本文只校验，不覆盖安装或升级。

先进入已经下载好的 simpler 仓库：

~~~bash
cd /path/to/simpler
export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

git rev-parse HEAD
git status --short
~~~

记录 HEAD 和工作区差异。若源码调整尚未提交，迁移环境时必须连同差异一起带走；
只有原始 HEAD 不能代表完整复现版本。

检查主机、Driver 和设备节点：

~~~bash
uname -m
getconf GNU_LIBC_VERSION
grep -E '^(Version|ascendhal_version|timestamp)=' \
    /usr/local/Ascend/driver/version.info

test -c /dev/davinci0
test -r /dev/davinci0
test -w /dev/davinci0
ls -l /dev/davinci0
~~~

本次预期为 x86_64、glibc 2.39、Driver 7.0.t9.0.B798，并且当前用户对
/dev/davinci0 可读写。任何一项失败时先修复系统权限或 Driver，不要用 Python
代码绕过。

检查安装过程会使用的基础工具：

~~~bash
for tool in bash git python3 rg sha256sum tar dpkg-deb; do
    command -v "$tool" || {
        echo "ERROR: missing tool: $tool"
        exit 1
    }
done
~~~

当前 A5 EVB 没有 npu-smi，也没有 task-submit。这不等于设备不可用；
本次通过 Driver 版本文件、设备节点和实际 ACL 调用完成了验证。如果另一个环境
提供设备预约工具，应先按该环境规则独占设备，再执行上板命令。

## 3. 安装 CANN 9.1 与用户级 GCC 15

### 安装 CANN 9.1

只使用以下两个 9.1 安装包，不要混入同目录下的 9.2 包：

| 安装包 | 字节数 | SHA-256 |
| --- | ---: | --- |
| Ascend-cann-toolkit_9.1.0~weekly.20260708.01_linux-x86_64.run | 1543071133 | 947165d939e83e4e73c14498e19b5ed69dd0de49bd9b5d71e04765bfa0c09313 |
| Ascend-cann-950-ops_9.1.0~weekly.20260708.01_linux-x86_64.run | 2669342311 | 9b5df71c1ca9a855f65027fb37c3fcd352ca607e997277aacff71508e36b8b91 |

先校验文件：

~~~bash
TOOLKIT="$HOME/cann/Ascend-cann-toolkit_9.1.0~weekly.20260708.01_linux-x86_64.run"
OPS="$HOME/cann/Ascend-cann-950-ops_9.1.0~weekly.20260708.01_linux-x86_64.run"

test -f "$TOOLKIT"
test -f "$OPS"

printf '%s  %s\n' \
    947165d939e83e4e73c14498e19b5ed69dd0de49bd9b5d71e04765bfa0c09313 \
    "$TOOLKIT" | sha256sum -c -
printf '%s  %s\n' \
    9b5df71c1ca9a855f65027fb37c3fcd352ca607e997277aacff71508e36b8b91 \
    "$OPS" | sha256sum -c -
~~~

安装包当前没有 executable bit，因此显式交给 bash。先按组织要求完成软件许可
确认，再使用 quiet 非交互安装：

~~~bash
export CANN_INSTALL_ROOT="$HOME/Ascend/cann-9.1.0-weekly-20260708"
mkdir -p "$CANN_INSTALL_ROOT"

bash "$TOOLKIT" \
    --full \
    --quiet \
    --install-path="$CANN_INSTALL_ROOT"

bash "$OPS" \
    --full \
    --quiet \
    --install-path="$CANN_INSTALL_ROOT"
~~~

本次成功安装未使用 --force。只有安装器明确报告兼容性问题，且已经核实
Driver/CANN 匹配关系时，才考虑该参数。

立即验证安装，不依赖 .bashrc：

~~~bash
test -f "$CANN_INSTALL_ROOT/cann/set_env.sh"
source "$CANN_INSTALL_ROOT/cann/set_env.sh"

test "$ASCEND_HOME_PATH" = \
    "$CANN_INSTALL_ROOT/cann-9.1.0"
test -x "$ASCEND_HOME_PATH/bin/ccec"
test -x \
    "$ASCEND_HOME_PATH/tools/hcc/bin/aarch64-target-linux-gnu-g++"

grep -E '^(Version|timestamp)=' \
    "$ASCEND_HOME_PATH/opp/version.info"
"$ASCEND_HOME_PATH/bin/ccec" --version | head
"$ASCEND_HOME_PATH/tools/hcc/bin/aarch64-target-linux-gnu-g++" \
    --version | head -n 1
~~~

预期 OPP Version 为 9.1.0，timestamp 为 20260708_000326093。

### 安装用户级 GCC 15

A5Sim 的 incore kernel 由仓库中的 Gxx15Toolchain 直接调用 g++-15，
所以仅有系统 g++ 不够。普通 host runtime 默认仍可使用系统 gcc/g++；
不要为了这一用例全局改写 CC 和 CXX。

本次验证使用从另一台已验证环境复制的 Ubuntu Plucky 解包目录：

~~~text
$HOME/.local/gcc-15/root
~~~

这是用户态解包，不是 dpkg -i。精确源码包版本可在 Ubuntu Launchpad 的
gcc-15 15-20250404-0ubuntu1 页面核对：

<https://launchpad.net/ubuntu/+source/gcc-15/15-20250404-0ubuntu1>

推荐直接从已验证环境打包并传输完整 root 目录：

~~~bash
# 在已验证的源环境执行
cd "$HOME/.local/gcc-15"
tar -czf "$HOME/gcc-15-plucky-20250404-root.tar.gz" root
cd "$HOME"
sha256sum gcc-15-plucky-20250404-root.tar.gz \
    > gcc-15-plucky-20250404-root.tar.gz.sha256

# 将归档及其 SHA-256 传到目标环境后执行
mkdir -p "$HOME/.local/gcc-15"
cp /path/to/gcc-15-plucky-20250404-root.tar.gz "$HOME/"
cp /path/to/gcc-15-plucky-20250404-root.tar.gz.sha256 "$HOME/"
cd "$HOME"
sha256sum -c gcc-15-plucky-20250404-root.tar.gz.sha256
tar -xzf gcc-15-plucky-20250404-root.tar.gz \
    -C "$HOME/.local/gcc-15"
~~~

如果使用原始 deb 重建目录，应准备同一版本的以下包，并逐个用
dpkg-deb -x 解到同一个 GCC15_ROOT：

~~~text
cpp-15
cpp-15-x86-64-linux-gnu
g++-15
g++-15-x86-64-linux-gnu
gcc-15
gcc-15-base
gcc-15-x86-64-linux-gnu
libasan8
libatomic1
libcc1-0
libgcc-15-dev
libgcc-s1
libgomp1
libhwasan0
libitm1
liblsan0
libquadmath0
libstdc++-15-dev
libstdc++6
libtsan2
libubsan1
~~~

~~~bash
export GCC15_ROOT="$HOME/.local/gcc-15/root"
mkdir -p "$GCC15_ROOT"

for deb in "$HOME/cann/gcc-15-plucky-debs"/*.deb; do
    dpkg-deb -f "$deb" Package Version
    test "$(dpkg-deb -f "$deb" Version)" = \
        "15-20250404-0ubuntu1"
    dpkg-deb -x "$deb" "$GCC15_ROOT"
done
~~~

不要把其他 Plucky 系统包或 libc6 一并放入该目录。当前编译器二进制要求
GLIBC_2.38，目标主机实际 glibc 必须满足要求。若 ldd 显示 not found，应先
补齐与主机兼容的 libisl、libmpc、libmpfr、libgmp、zlib、libzstd 或 binutils，
不要盲目混用另一发行版的 libc。

对复制结果做内容检查：

~~~bash
export GCC15_ROOT="$HOME/.local/gcc-15/root"

printf '%s  %s\n' \
    db5b698ddfbbefa3978b76c0f9dd7504bd82136db461a05320b74743a8933ec9 \
    "$GCC15_ROOT/usr/bin/x86_64-linux-gnu-g++-15" \
    | sha256sum -c -
printf '%s  %s\n' \
    34ffcc0db386d0d654c29464b84c57a8218b650dfd3b801720b778eacdca7a9e \
    "$GCC15_ROOT/usr/libexec/gcc/x86_64-linux-gnu/15/cc1plus" \
    | sha256sum -c -
printf '%s  %s\n' \
    9fb7d85e8aa687d1d8b27d5c189f3d938579a29e75af9d4651db4d33218fb401 \
    "$GCC15_ROOT/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.34" \
    | sha256sum -c -
~~~

### 写入用户 .bashrc

$HOME/.bashrc 是文件，不是目录。把以下内容追加到文件末尾；这里的 HOME
就是当前普通用户的 home，不是系统 /root：

~~~bash
# Ascend CANN user installation.
if [ -f "$HOME/Ascend/cann-9.1.0-weekly-20260708/cann/set_env.sh" ]; then
    source "$HOME/Ascend/cann-9.1.0-weekly-20260708/cann/set_env.sh"
fi

# User-local GCC 15 (Ubuntu 25.04 Plucky packages).
export GCC15_ROOT="$HOME/.local/gcc-15/root"
if [ -x "$GCC15_ROOT/usr/bin/g++-15" ]; then
    export PATH="$GCC15_ROOT/usr/bin:$PATH"
    if [ -n "$LD_LIBRARY_PATH" ]; then
        export LD_LIBRARY_PATH="$GCC15_ROOT/usr/lib/x86_64-linux-gnu:$GCC15_ROOT/usr/lib/gcc/x86_64-linux-gnu/15:$LD_LIBRARY_PATH"
    else
        export LD_LIBRARY_PATH="$GCC15_ROOT/usr/lib/x86_64-linux-gnu:$GCC15_ROOT/usr/lib/gcc/x86_64-linux-gnu/15"
    fi
fi

# User Python environment.
if [ -f "$HOME/.venv/bin/activate" ]; then
    source "$HOME/.venv/bin/activate"
fi
~~~

自动激活 venv 会影响所有新开的交互 shell，这是本次用户要求的行为。CI、
cron 或非交互脚本仍应显式 source 对应环境。

保存后打开新的交互 shell，或执行 exec bash，再验证：

~~~bash
command -v ccec
command -v g++-15
g++-15 --version | head -n 1
g++-15 -print-prog-name=cc1plus

ldd "$(g++-15 -print-prog-name=cc1plus)" | \
    grep 'not found' && exit 1 || true

printf '#include <iostream>\nint main(){std::cout << "gcc15-ok\\n";}\n' |
    g++-15 -x c++ -std=c++23 - -o /tmp/gcc15-smoke
/tmp/gcc15-smoke
~~~

预期版本首行为：

~~~text
g++-15 (Ubuntu 15-20250404-0ubuntu1) 15.0.1 20250404 (experimental)
~~~

## 4. Python、PTO-ISA 与精确构建

### 创建用户 Python 环境

本次使用 $HOME/.venv，而不是仓库内的 .venv。首次创建：

~~~bash
python3 -m venv --system-site-packages "$HOME/.venv"
source "$HOME/.venv/bin/activate"

python --version
python -m pip --version
~~~

--system-site-packages 与当前板端部署一致，使已安装的 torch 2.6.0+cpu
可见。目标用例直接 import torch 来生成输入和 golden，因此 PyTorch 必需；
它不直接 import torch_npu。

安装本次用到的 Python 和构建依赖：

~~~bash
python -m pip install \
    scikit-build-core==1.0.3 \
    nanobind==2.13.0 \
    cmake==4.4.0 \
    cloudpickle==3.1.2 \
    pytest==7.4.4 \
    pytest-xdist==3.8.0 \
    pytest-timeout==2.4.0 \
    ruff==0.14.8
~~~

若系统 site-packages 中没有 torch，再从当前环境认可的 wheel 源安装
torch 2.6.0+cpu；不要未经确认改成最新版本。能访问 PyTorch 官方 CPU
wheel 源时可执行：

~~~bash
python -c 'import torch; print(torch.__version__)' || \
    python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.6.0

python -c '
import pytest
import torch
print("python:", __import__("sys").version.split()[0])
print("pytest:", pytest.__version__)
print("torch:", torch.__version__)
'
~~~

### 固定 PTO-ISA

~~~bash
cd "$REPO_ROOT"
export PTO_ISA_COMMIT=ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8
export PTO_ISA_ROOT="$REPO_ROOT/build/pto-isa"

if [ ! -d "$PTO_ISA_ROOT/.git" ]; then
    git clone https://github.com/hw-native-sys/pto-isa.git \
        "$PTO_ISA_ROOT"
fi

git -C "$PTO_ISA_ROOT" fetch origin "$PTO_ISA_COMMIT"
git -C "$PTO_ISA_ROOT" checkout --detach "$PTO_ISA_COMMIT"
test "$(git -C "$PTO_ISA_ROOT" rev-parse HEAD)" = \
    "$PTO_ISA_COMMIT"
~~~

### 只构建 Python binding

直接执行 pip install -e . 会触发顶层 ALL target，并自动枚举当前可构建的
所有平台和 runtime。为保持本文边界，先只构建 _task_interface：

~~~bash
cd "$REPO_ROOT"
source "$HOME/.venv/bin/activate"
source "$HOME/Ascend/cann-9.1.0-weekly-20260708/cann/set_env.sh"

# 移除这个 venv 中可能残留的旧 simpler/editable import hook。
# 只删除 Python 安装记录，不删除当前源码树或 build 产物。
if python -m pip show simpler >/dev/null 2>&1; then
    python -m pip uninstall -y simpler
fi

cmake -S . -B build/python-bindings \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$(command -v python)" \
    -Dnanobind_DIR="$(python -c \
        'import nanobind; print(nanobind.cmake_dir())')"

cmake --build build/python-bindings \
    --target _task_interface \
    --parallel "$(nproc)"

if [ -n "$PYTHONPATH" ]; then
    export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/python:$PYTHONPATH"
else
    export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/python"
fi
python -c '
from pathlib import Path
import simpler
import _task_interface
print("simpler:", simpler.__file__)
print("_task_interface:", _task_interface.__file__)
assert Path(_task_interface.__file__).resolve().parent == \
    Path("python").resolve()
'
~~~

不要把项目专用 PYTHONPATH 永久写入全局 .bashrc。每次进入本仓工作时设置，
或在测试命令所在 shell 中保持以上 export 即可。若不移除旧 editable
安装，它注册的 import hook 可能优先加载 site-packages 中的旧 binding，
使刚构建的源码树产物没有真正被测试。

### 只构建目标 runtime

当前 build_runtimes.py 的 --platforms 只能限制平台，不能限制 runtime。
使用 RuntimeBuilder 的现有接口，精确构建 A5Sim/A5 的
fully_distributed_within_core 及它们必需的共享 helper：

~~~bash
cd "$REPO_ROOT"
source "$HOME/.venv/bin/activate"
source "$HOME/Ascend/cann-9.1.0-weekly-20260708/cann/set_env.sh"
if [ -n "$PYTHONPATH" ]; then
    export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/python:$PYTHONPATH"
else
    export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/python"
fi

python - <<'PY'
from simpler_setup.runtime_builder import RuntimeBuilder

runtime = "fully_distributed_within_core"
for platform in ("a5sim", "a5"):
    print(f"building {platform}/{runtime}")
    binaries = RuntimeBuilder(platform).get_binaries(runtime, build=True)
    print(binaries)
PY
~~~

验证目标产物：

~~~bash
test -f \
    build/lib/a5/sim/fully_distributed_within_core/libhost_runtime.so
test -f \
    build/lib/a5/sim/fully_distributed_within_core/libaicpu_kernel.so
test -f \
    build/lib/a5/sim/fully_distributed_within_core/libaicore_kernel.so

test -f \
    build/lib/a5/onboard/fully_distributed_within_core/libhost_runtime.so
test -f \
    build/lib/a5/onboard/fully_distributed_within_core/libaicpu_kernel.so
test -f \
    build/lib/a5/onboard/fully_distributed_within_core/aicore_kernel.o

test -f build/lib/a5/dispatcher/libsimpler_aicpu_dispatcher.so
test -f build/lib/libsimpler_log.so
test -f build/lib/libcpu_sim_context.so
~~~

编译器职责如下：

| 目标 | 编译器 |
| --- | --- |
| A5Sim incore kernel | 用户级 g++-15 |
| A5Sim host/runtime helper | 系统 gcc/g++ |
| A5 AICore kernel | CANN 9.1 ccec |
| A5 AICPU 目标 | CANN 9.1 AArch64 交叉编译器 |
| A5 host 目标 | 系统 gcc/g++ |

因此，CMake cache 中看到系统 g++ 不代表 GCC 15 被绕过；A5Sim incore
kernel 是后续由 KernelCompiler 直接调用 g++-15 编译的。

## 5. 执行 A5Sim 与真实 A5

每个新 shell 先执行统一准备：

~~~bash
cd "$REPO_ROOT"
source "$HOME/.venv/bin/activate"
source "$HOME/Ascend/cann-9.1.0-weekly-20260708/cann/set_env.sh"

if [ -n "$PYTHONPATH" ]; then
    export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/python:$PYTHONPATH"
else
    export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/python"
fi
export PTO_ISA_ROOT="$REPO_ROOT/build/pto-isa"
export PTO_ISA_COMMIT=ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8

TEST_FILE=examples/a5/fully_distributed_within_core/paged_attention_unroll/test_paged_attention_unroll.py

test "$(command -v python)" = "$HOME/.venv/bin/python"
test "$(git -C "$PTO_ISA_ROOT" rev-parse HEAD)" = \
    "$PTO_ISA_COMMIT"
~~~

### 运行 A5Sim Case1

~~~bash
python -m pytest "$TEST_FILE" \
    --platform a5sim \
    --case Case1 \
    --enable-l2-swimlane 4 \
    --use-example-exec-time \
    --clone-protocol https \
    --pto-isa-commit "$PTO_ISA_COMMIT" \
    --pto-session-timeout 1200 \
    --require-pto-isa \
    -s -v
~~~

--use-example-exec-time 仅适用于 fully_distributed_within_core 的 sim。
它不能用于真实 A5 命令。这里显式写出的 level 4 与裸参数
`--enable-l2-swimlane` 等价；A5Sim 用于检查 schema v3、记录结构、加权计数闭合和
转换结果，不提供真实 A5 atomic 完成时间。A5Sim 的直接 Atomic 只标记模拟执行的
源码包围边界，PollBatch 只标记模拟调度中的 poll 窗口；两者都不能当作真实硬件时延。

### 运行 A5 正确性 smoke

确认没有其他进程占用 device 0 后执行：

~~~bash
test -r /dev/davinci0
test -w /dev/davinci0

python -m pytest "$TEST_FILE" \
    --platform a5 \
    --device 0 \
    --case Case1 \
    --clone-protocol https \
    --pto-isa-commit "$PTO_ISA_COMMIT" \
    --pto-session-timeout 1200 \
    --require-pto-isa \
    -s -v
~~~

### 运行 A5 phase 泳道性能复现

~~~bash
python -m pytest "$TEST_FILE" \
    --platform a5 \
    --device 0 \
    --case Case1 \
    --enable-l2-swimlane 1 \
    --clone-protocol https \
    --pto-isa-commit "$PTO_ISA_COMMIT" \
    --pto-session-timeout 1200 \
    --require-pto-isa \
    -s -v
~~~

level 1 到 3 保留已有 FDWIC phase 记录，不增加 Atomic 和
`ClockBaseline`。上面的 level 1 与历史 trace 的有效 level 一致，适合继续复现本文
约 5.6 ms 的 Submit 性能口径。它们继续导出 schema 1 的 legacy Claim flags；
level 4 导出 schema v3，其中 Claim 使用 `attempted/won` 三态，Atomic 同时支持
逐调用直接记录和精确计数 PollBatch。源码 atomic 调用点现在统一经过 level 判断，
因此不能声称新旧二进制指令级完全相同；做前后性能比较时，应使用同一版二进制分别采
level 1 phase 基线和 level 4 诊断样本。

### 运行 A5 level-4 atomic 泳道

需要观察真实 PA 的直接 atomic 调用并统计等待区 poll 调用时，单独运行 level 4：

~~~bash
python -m pytest "$TEST_FILE" \
    --platform a5 \
    --device 0 \
    --case Case1 \
    --enable-l2-swimlane 4 \
    --clone-protocol https \
    --pto-isa-commit "$PTO_ISA_COMMIT" \
    --pto-session-timeout 1200 \
    --require-pto-isa \
    -s -v
~~~

level 4 在原有 phase 上增加真实 FDWIC PA 的 Atomic 记录：直接 atomic 保持一次
源码调用一条记录；显式等待区内允许合并的 observation load，以及唯一一种已确认的
幂等失败 exchange 重试，则用带精确调用次数的 PollBatch 表示。最终 drain 之后，每个
AIC/AIV 还会写两条 `ClockBaseline`。该模式会增加记录写、改变代码布局，也可能改变
多核到达和轮询次数，所以它是诊断样本，不是本文 5.6 ms 无 atomic 插桩基线。

### raw、merged 产物与重新转换

A5Sim 和 A5 的 Case1 都使用 SceneTest 的同一输出规则：

~~~text
outputs/TestPagedAttentionUnroll_Case1_<YYYYMMDD_HHMMSS>/
├── l2_swimlane_records.json
├── merged_swimlane.json
└── swimlane_exclusive_analysis.json
~~~

其中 `l2_swimlane_records.json` 是保留原始 cycle 的 raw 文件；pytest 在 case
结束时调用仓内共享转换器，生成可直接载入 Perfetto 的
`merged_swimlane.json`。当前真实 A5 FDWIC level-4 还必须生成严格父子/Kernel/
整数闭合报告 `swimlane_exclusive_analysis.json`；raw 缺失、转换失败或任一加工件
缺失/为空都会使成功的 pytest 用例失败。若设备执行本身已经失败，则仍保留设备侧
原始异常，不用离线转换错误覆盖根因。历史 phase 基线 raw 与本次真实 A5 level-4
raw 分别为：

~~~text
outputs/TestPagedAttentionUnroll_Case1_20260717_023809/l2_swimlane_records.json
outputs/TestPagedAttentionUnroll_Case1_20260718_161520/l2_swimlane_records.json
~~~

新的复现会生成不同时间戳目录。若需要重新转换已有 raw，执行：

~~~bash
python -m simpler_setup.tools.swimlane_converter \
    outputs/TestPagedAttentionUnroll_Case1_YYYYMMDD_HHMMSS/l2_swimlane_records.json \
    -v
~~~

未指定 `-o` 时，转换器仍写到 raw 同目录的 `merged_swimlane.json`。上面的历史
phase-only raw 约几十 MiB；level 4 的 JSON 大小取决于直接 Atomic、PollBatch 和
phase 的实际记录条数，不再随每一次连续 poll 线性增长。pytest 结束后再读取，
不要用 pytest wall time 代替 Submit 指标。

### level-4 atomic 数据契约

当前生产端导出的 level 4 raw，其 `metadata.trace_schema_version` 必须为 4；共享
converter 仍保留对历史 schema-v3 的读取兼容，但只有 schema-v4 生成当前排他闭合
报告。转换后的 `Atomic` 和
`ClockBaseline` 都画在对应 AIC、AIV0 或 AIV1 的原 scalar lane；它们不是与
scalar 并行的伪子轨。Kernel 仍画在独立的 `AIC/AIV·kernel` 轨。

直接 Atomic 的名称明确给出终点语义：

~~~text
atomic.return_ready.<site>.<op>#<task_id>
atomic.source_issue.<site>.<op>#<task_id>
~~~

真实 A5 上，返回值会被后续逻辑消费的直接调用使用 `return_ready`，表示返回旧值已
可被本核 scalar 使用；返回值被丢弃的发布型直接调用使用 `source_issue`，只表示源码
发射包围区间。两者都不表示跨核可见时刻，也不能直接称为 atomic retire 延迟。每条
直接 Atomic 的 `args.call_count` 固定为 1。结束 cycle 在 atomic 返回后立即采样，
本地调用计数更新、PollBatch 落盘和 direct 记录写入都发生在结束 cycle 之后，不能
混入该条 direct span。

以下九类 observation load 只有在对应的显式 scheduler 等待区内才允许聚合：

- 通用等待：`startup_poll`、`fatal_poll`、`fanin_flag_load`、
  `heap_frontier_load`、`heap_vend_load`、`replay_done_poll`；
- BlockWon 等待：`won_any_load`、`won_state_load`、`won_drained_load`。

BlockWon 三类 load 只在 slot-capacity、heap slow path、won-slot、sim producer-ready、
final-drain 这些已有外层等待区中聚合；普通 Submit 中的一次性或 opportunistic
BlockWon 扫描仍是直接 Atomic。

唯一允许聚合的 RMW 是 `won_lane_claim_exchange`，并且必须同时满足：位于上述显式
等待区、写入 `claimed(1)`、返回旧值也是 `claimed(1)`。这表示一次 1→1、没有改变
协议状态的失败 claim 重试。返回 `free(0)` 的成功 claim 始终逐条记录；
`won_slot_claim_max`、release、`fetch_sub`、state clear 以及其他 RMW 也全部逐条记录。

每个聚合记录转换为：

~~~text
atomic.poll_batch.<site>.<op>×<call_count>
~~~

其 `args.call_count` 是该等待区内实际执行的源码 atomic wrapper 调用次数，不是采样值；
`task_id=-1`、`func_id=-1` 表示它归属于 scheduler 等待区而非某个任务。一个等待区可
同时累积多个 site，因此不同 site 的 PollBatch 时间窗可以重叠；等待区内的直接
Atomic 也可能与该窗口交错。merged 用 `batch_semantics=observation_load_calls` 或
`idempotent_failed_exchange_retries` 区分两种计数，并显式写出
`may_contain_interleaved_direct_atomics=true`。

PollBatch 的 `duration`/`poll_window_cycles` 只是 logical poll episode 的包络：它既
不是纯 poll 时间或独占 scalar 时间，也不是其中任一次 atomic 的延迟，更不是
`call_count` 次 atomic 串行延迟之和。raw 中的物理相邻顺序也不等于严格时间顺序；
分析应以 cycle 字段为准，不能用 PollBatch duration 计算单次 atomic 的 median 或 p95。

schema v3 用 Atomic flags 的 bit 7 标识 PollBatch，bits 8..31 保存无符号 24 bit
`call_count`；低 4 bit 必须是该 site 的实际 op：九类 observation 是 `load(0)`，
`won_lane_claim_exchange` 是 `exchange(1)`。bit 4 表示返回值被消费，bit 5/6 在
PollBatch 中必须为 0。bit 7 为 0 的直接 Atomic 保留原有 flags 语义，不能把其高位
按 poll 次数解析。

site 0 到 14 与 standalone PA 的稳定编号完全一致；真实 FDWIC PA 只在末尾追加
15 到 27，覆盖生产实现的 BlockWon 路径和 `FetchSub`，没有重排已有编号：

| `site_id` | Perfetto `site` | `op` | 真实 PA 路径 |
| --------: | --------------- | ---- | ------------ |
| 0 | `startup_increment` | `fetch_add` | 启动屏障到达计数 |
| 1 | `startup_poll` | `load` | 启动屏障轮询 |
| 2 | `fatal_poll` | `load` | fatal 状态检查 |
| 3 | `fatal_set` | `exchange` | fatal 状态发布 |
| 4 | `claim_max` | `fetch_max` | Submit lane Claim |
| 5 | `fanin_flag_load` | `load` | fanin 依赖 flag |
| 6 | `completion_vend_exchange` | `exchange` | completion vend 发布 |
| 7 | `completion_flag_exchange` | `exchange` | completion flag 发布 |
| 8 | `frontier_initial_load` | `load` | completion frontier 首次读取 |
| 9 | `frontier_flag_load` | `load` | frontier 扫描 flag |
| 10 | `frontier_max` | `fetch_max` | frontier 推进 |
| 11 | `heap_frontier_load` | `load` | HeapGuard frontier |
| 12 | `heap_vend_load` | `load` | HeapGuard vend |
| 13 | `replay_done_increment` | `fetch_add` | final 树的 leaf/root arrival 和 release 发布 |
| 14 | `replay_done_poll` | `load` | 最终 drain 轮询 leaf/root arrival 和 release |
| 15 | `won_slot_claim_max` | `fetch_max` | BlockWon slot 认领 |
| 16 | `won_remaining_exchange` | `exchange` | BlockWon remaining 初始化 |
| 17 | `won_lane_reset_exchange` | `exchange` | BlockWon lane 重置 |
| 18 | `won_lane_deposit_exchange` | `exchange` | BlockWon lane 完成发布 |
| 19 | `won_state_publish_exchange` | `exchange` | BlockWon state 发布 |
| 20 | `won_any_publish_exchange` | `exchange` | BlockWon any 发布 |
| 21 | `won_any_load` | `load` | BlockWon any 读取 |
| 22 | `won_state_load` | `load` | BlockWon state 读取 |
| 23 | `won_lane_claim_exchange` | `exchange` | BlockWon lane claim |
| 24 | `won_lane_release_exchange` | `exchange` | BlockWon lane release |
| 25 | `won_remaining_fetch_sub` | `fetch_sub` | BlockWon remaining 递减并判断最后一个 lane |
| 26 | `won_state_clear_exchange` | `exchange` | BlockWon state 清理 |
| 27 | `won_drained_load` | `load` | BlockWon drained 检查 |

上表定义的是本文覆盖的真实 FDWIC PA / A5 hot path 可记录调用点集合，不包含
CPU sim watchdog/debug 诊断原子，也不表示 Case1 每轮一定出现全部 28 类事件。
Case1 的单 lane 图通常不进入 BlockWon 动态路径；某个 BlockWon site
计数为零不能单独判定为漏插桩。`fetch_sub` 的 op id 为 4，不能按 standalone
旧版只有 Load、Exchange、FetchAdd、FetchMax 四类 op 的假设解析。

frontier 扫描沿用 standalone 的任务归因：读取全局扫描起点的
`frontier_initial_load` 记为 `task_id=-1`；随后读取 `next` 完成 flag 的
`frontier_flag_load` 与推进同一个 `next` 的 `frontier_max` 都记为
`task_id=next`。因此两条扫描事件可以按 core、task_id 配对，不归到触发本轮
completion 的另一个任务上。

raw 的 `metadata.trace_schema_version=3` 中，Claim 的 flags 明确记录
`attempted` 和 `won`，merged 中对应三种互斥状态：

| Claim 状态 | `attempted` | `won` | Perfetto 名称 | Case1 预期数量（`N=metadata.num_cores`） |
| ---------- | ----------: | ----: | ------------- | -------------: |
| 未参与该 lane 的 Claim | 0 | 0 | `claim.not_attempted` | `N*512` |
| 已尝试但失败 | 1 | 0 | `claim.lost` | `N*768-1280` |
| 已尝试且获胜 | 1 | 1 | `claim.won` | `1280` |

三者合计 `N*1280` 条 Claim，其中实际执行 `claim_max.fetch_max` 的数量为
`N*768`。历史 96 核样本对应 49,152 / 72,448 / 1,280；108 核 A5Sim 对应
55,296 / 81,664 / 1,280。`attempted=0, won=1` 是非法组合，不应出现在有效导出中。

host 在发布 raw 前按核执行闭合校验：

- `count` 不得超过该核分区容量，禁止截断后继续导出；
- `dropped` 必须为 0；
- 每条直接 Atomic 计为一次调用；每条 PollBatch 按 flags 高 24 bit 编码的
  `call_count` 加权，且 `call_count` 必须大于 0；
- `atomic_calls = atomic_records - poll_batch_records + batched_poll_calls`；
- 所有 PollBatch 的 `call_count` 之和必须等于 `batched_poll_calls`，物理 PollBatch
  条数必须等于 `poll_batch_records`；
- level 4 每核必须恰有两条 `ClockBaseline`，总数必须为 `2*N`；
- level 1 到 3 不应出现 `Atomic` 或 `ClockBaseline`。

level 1 到 4 都使用每核 65,536 条（64K）分区。设备二进制 record version 3 不再
逐条重复 `core_idx/block_id/lane`，而是在每核 state 中保存并校验一次，导出十列 raw
JSON 时再补回；每条物理记录为 32 byte，JSON 列格式不变。仅计 record 分区时，96 个
worker 约占 192 MiB，108 个 worker 约占 216 MiB，另有很小的 header；raw metadata
中的 `records_per_core`、`record_size_bytes` 和 `device_trace_bytes` 给出本次运行的精确
配置。host 只初始化 header，导出时先读取 header/core 计数，再按核搬运实际
`count*32` byte 的有效记录，不常驻完整设备镜像。

PollBatch 让大量连续轮询按等待区和 site 合并，同时保留精确调用次数，因此不再需要
为每一次 poll 预留物理记录。若单个 batch 达到高 24 bit 可表示的最大次数，实现会
先落盘并开启下一条 batch，不会饱和后丢失计数。物理记录容量仍不是理论无界；任何
容量溢出都必须明确失败，不能截断后发布，也不能只放宽 `dropped` 校验。
设备侧边界 GTest 直接从 `0xFFFFFE` 累加到 `0xFFFFFF`，确认第一条立即落盘；随后
第 `0x1000000` 次调用以 `call_count=1` 重开第二条，并验证两条之和精确等于
`0x1000000`。

`batched_poll_calls` 包含上述九类 observation load 和幂等失败 exchange 重试，是
本次启用 schema v3 插桩后真实执行的精确调用数。插桩本身会改变代码布局、核间到达
时序和轮询节奏，因此不同插桩方案下的调用次数不能当作固定 workload 常量直接比较；
计数换算和前后对比应使用同一观察模式。

2026-07-18 最终 A5Sim 结构验证得到以下闭合结果；它们用于证明记录规模和计数契约，
不表示真实 A5 atomic 时延，也不参与后文按 160 ns 对真机计数所做的归因估算。
ReuseStress 只是十类规则与 BlockWon 路径的结构压力验收，不扩大本文只复现 PA Case1
性能的范围：

| 样本 | raw 大小 | 总记录 | 逻辑 `atomic_calls` | 物理 Atomic | PollBatch | 单核记录峰值 | dropped |
| ---- | -------: | -----: | ---------------------: | ----------: | --------: | ------------: | ------: |
| Case1 `20260718_152435` | 124,547,744 B（约 118.8 MiB） | 1,464,594 | 187,860,395 | 493,301 | 959 | 16,357 | 0 |
| BlockWon ReuseStress `20260718_154229` | 2,055,624 B（约 1.96 MiB） | 24,442 | 243,357,709 | 10,505 | 572 | 519 | 0 |

Case1 中 `batched_poll_calls=187,368,053`，满足
`187,860,395 - 187,368,053 + 959 = 493,301`；108 核共有 216 条
`ClockBaseline`。ReuseStress 中 `batched_poll_calls=243,347,776`，满足
`243,357,709 - 243,347,776 + 572 = 10,505`；其中 22 条 site 23 PollBatch
精确表示 625,394 次幂等失败 exchange 重试，另有 348 条 site 23 直接记录。这里的
直接记录包含成功 claim、等待区外调用或其他非聚合情形，不能全部等同为成功次数。

同一版 level-4 代码随后在真实 A5 device 0 上执行 Case1，得到以下实际 PA 记录：

| 样本 | raw 大小 | merged 大小 | 核拓扑 | 总记录 | 逻辑 `atomic_calls` | 物理 Atomic | PollBatch | 单核记录峰值 | dropped |
| ---- | -------: | ----------: | ------ | -----: | ---------------------: | ----------: | --------: | ------------: | ------: |
| A5 Case1 `20260718_161520` | 77,128,944 B（约 73.6 MiB） | 333,581,552 B（约 318.1 MiB） | 32 AIC + 64 AIV | 973,430 | 115,200 | 110,006 | 340 | 10,751 | 0 |

该真机样本中 `batched_poll_calls=5,534`，满足
`115,200 - 5,534 + 340 = 110,006`；96 核共有 192 条 `ClockBaseline`。340 条
PollBatch 分布在 `StartupPoll`、`FatalPoll`、`FaninFlagLoad` 和
`ReplayDonePoll`，精确表示 5,534 次等待区调用。Case1 没有动态进入 BlockWon，
因此没有出现 21/22/23/27 类 batch；这不影响十类 allowlist 的实现和独立
BlockWon ReuseStress 覆盖。直接事件中 106,914 条使用真机 `return_ready` 边界，
与 `result_used` 数量一致，可与 A5Sim 的 `source_issue` 边界明确区分。

若只为形成直观的 scalar 归因依据，暂统一使用 160 ns/次，则这份真实 A5 Case1 的
全核累计估计为 `115,200 × 160 ns = 18.432 ms core-work`。它是跨 96 核求和后的
工作量，不是 PA 墙钟耗时；不能把 18.432 ms 与约 5.6 ms Submit 包络直接相加。

任一条件不满足，export 返回非零；若 runtime 本身成功，该错误继续传播为本次运行
失败，不能把旧文件或不完整文件当作有效样本。共享 converter 会按 raw 行重新计算
`records`、`atomic_records`、`clock_baseline_records`、`atomic_calls`、
`batched_poll_calls` 和 `poll_batch_records`，逐项核对 `metadata.fdwic_summary`；
`dropped_records` 无法从已导出的有效行反推，因此 converter 要求 producer summary
明确给出 0。正式分析还应确认 raw 与 merged 的物理 Atomic 条数相等，并以
`atomic_calls` 而不是物理 Atomic 条数表示源码调用总数。

两条 `ClockBaseline` 分别是连续两次 `SYS_CNT` 读取，以及 atomic 返回值依赖 hook
的固定路径。它们用于观察计时分辨率和 hook 本身的分布，不是可以从每条 Atomic
机械相减的校正常数。

### atomic 性能解释边界

单条直接 Atomic 是某一 AIC/AIV scalar lane 上的本地 span。可以按
`core_type/site/op` 查看直接事件数、中位数、p95、最大值，也可以比较同一核上某个
直接 site 的累计分布；这些数据适合回答“哪类直接 atomic 常见、哪类本核返回等待
长”。PollBatch 只适合统计对应 site/op 的调用次数和等待 episode 分布；其窗口可能
包含交错的直接 atomic，duration 不能混入单次 atomic 延迟的 median/p95。

若已有独立 atomic probe 给出的标定开销，可把 schema v3 的精确调用数换算为直观的
scalar 工作量估计：

~~~text
estimated_atomic_core_work_ns = Σ(event.call_count × calibrated_atomic_cost_ns(site, op))
~~~

直接 Atomic 的 `call_count=1`，PollBatch 使用其精确计数。若暂时对所有 site/op 统一
采用约 160 ns 的单次标定值，公式简化为
`estimated_atomic_core_work_ns ≈ atomic_calls × 160 ns`；若只归因某个 scalar
阶段，则只对属于该阶段的事件求和。该数值是所有核累计的 scalar core-work 估计，
不是 Submit wall time，也不是从 PollBatch duration 反推的单次硬件时延。

不能把所有核的 Atomic span 简单求和后称为 Submit 墙钟开销：不同核并行执行，
大量 span 相互重叠；Atomic 还嵌套在 Claim、Replay、Submit 等外层 phase 中，外层
和内层也不能再次相加。全核求和只能解释为带观察的 aggregate core-work。若要判断
对墙钟时间的影响，应结合关键 scalar lane、全局最早 Submit 到最晚 Submit 的包络，
并对优化前后使用相同观察模式；最终端到端收益仍用关闭 atomic 诊断的独立运行确认。

### 提取首个到末个 Submit

先把 `TRACE` 指向 level 1 到 3 的 phase-only Case1 raw。以下脚本拒绝 level 4，
避免误把 Atomic 插桩后的诊断时间当成约 5.6 ms 基线；随后校验 96 个 core、
每核 1280 个 Submit 以及完整 task id，并输出用户关注的全局 span：

~~~bash
export TRACE=outputs/TestPagedAttentionUnroll_Case1_YYYYMMDD_HHMMSS/l2_swimlane_records.json

python - <<'PY'
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

trace = Path(os.environ["TRACE"])
if not trace.is_file():
    raise SystemExit(f"trace does not exist: {trace}")
with trace.open() as stream:
    data = json.load(stream)

level = int(data["l2_swimlane_level"])
assert level in (1, 2, 3), f"phase baseline requires level 1..3, got {level}"
hz = int(data["metadata"]["clock_freq_hz"])
submits = [row for row in data["fdwic_events"] if row[5] == "Submit"]
if not submits:
    raise SystemExit("trace contains no Submit events")

by_core = defaultdict(list)
for row in submits:
    by_core[int(row[0])].append(row)

assert int(data["metadata"]["num_cores"]) == 96
assert len(by_core) == 96
for core, rows in by_core.items():
    task_ids = sorted(int(row[3]) for row in rows)
    assert len(rows) == 1280, (core, len(rows))
    assert task_ids == list(range(1280)), core

first_cycle = min(int(row[6]) for row in submits)
last_cycle = max(int(row[7]) for row in submits)
first_to_last_ms = (last_cycle - first_cycle) * 1000 / hz

kernel_first_cycle = min(
    int(row[6]) for row in submits if int(row[3]) == 1
)
kernel_to_last_ms = (last_cycle - kernel_first_cycle) * 1000 / hz

per_core_ms = []
for rows in by_core.values():
    start = min(int(row[6]) for row in rows)
    end = max(int(row[7]) for row in rows)
    per_core_ms.append((end - start) * 1000 / hz)

first_row = min(submits, key=lambda row: int(row[6]))
last_row = max(submits, key=lambda row: int(row[7]))
assert int(first_row[3]) == 0
assert int(last_row[3]) == 1279

print("trace:", trace)
print("clock_freq_hz:", hz)
print("cores:", len(by_core))
print("submits_per_core:", len(next(iter(by_core.values()))))
print(f"first_to_last_submit_ms: {first_to_last_ms:.6f}")
print(f"task1_to_last_submit_ms: {kernel_to_last_ms:.6f}")
print(f"per_core_median_ms: {statistics.median(per_core_ms):.7f}")
print(f"per_core_max_ms: {max(per_core_ms):.6f}")
PY
~~~

本次预期输出的关键值：

~~~text
clock_freq_hz: 1000000000
cores: 96
submits_per_core: 1280
first_to_last_submit_ms: 5.642245
task1_to_last_submit_ms: 5.635263
per_core_median_ms: 5.5725575
per_core_max_ms: 5.641331
~~~

不同运行允许有小幅抖动。验收重点是正确性 PASSED、事件完整，并且
first_to_last_submit_ms 仍位于约 5.6 ms 的基线附近。

## 6. 验收清单与故障定位

### 最终验收

- CANN 安装包 SHA-256 与本文一致；
- ASCEND_HOME_PATH 指向用户目录下的 CANN 9.1；
- command -v python 为 $HOME/.venv/bin/python；
- command -v g++-15 指向 $HOME/.local/gcc-15/root；
- PTO-ISA HEAD 为固定 commit；
- 源码中没有 block_dim=36；
- 只构建 A5Sim/A5 的 fully_distributed_within_core；
- A5Sim Case1 PASSED；
- A5 Case1 PASSED；
- trace 为 96 core，每核 1280 个 Submit；
- 全局首末 Submit 约为 5.6 ms。

### 常见问题

**找不到 g++-15**

确认 GCC15_ROOT、PATH 和 LD_LIBRARY_PATH 已生效，并重新打开交互 shell。
A5Sim incore kernel 必须能直接执行 g++-15。

**找不到 ccec 或 AArch64 交叉编译器**

重新 source 用户 CANN 9.1 的 cann/set_env.sh，并检查
ASCEND_HOME_PATH。不要回退到同目录的 CANN 9.2。

**提示 pre-built runtime binaries not found**

重新执行“只构建目标 runtime”中的 RuntimeBuilder 片段。不要改用会自动
枚举所有 runtime 的顶层构建。

**CPU_TOPO 的 HAL/DSMI 返回 65534**

这是当前旧 Driver 的已知能力差异。只有日志同时表明 OCCUPY popcount 与
ACL AICPU count 一致，并出现 using flat OCCUPY fallback 时才可继续。
本次预期是 mask=0x3e、count=5。若出现 flat fallback rejected，停止运行，
不要删除校验或强行构造 CPU 列表。

**仍然使用 block_dim=36**

说明源码 revision 不完整。切换到同时包含本文三处源码调整的 revision，
再增量重建 A5 目标 runtime。

**PTO-ISA clone 超时或 commit 不一致**

先在 build/pto-isa 中独立完成 fetch 和 detached checkout，再运行 pytest。
--require-pto-isa 会让错误尽早暴露，不能去掉 pin 后继续跑未知版本。

**import torch 失败**

确认 venv 是用 --system-site-packages 创建，或从环境认可的 wheel 源安装
torch 2.6.0。该用例需要 torch，但不因这一点要求直接调用 torch_npu。

**出现 torch_npu library owner permission mismatch warning**

当前系统 site-packages 可能在 import 阶段报告某个 torch_npu 共享库 owner
不匹配。目标用例不直接使用 torch_npu；若 torch、simpler 均可导入且测试
PASSED，该 warning 不影响本次结论。不要以普通用户修改系统共享库的 owner；
若它升级为 import error，再交由系统环境维护者处理。

**/dev/davinci0 无权限或设备忙**

由系统管理员修复用户组/ACL，或等待当前任务释放设备。不要 sudo 运行 pytest，
否则会绕开用户 venv、HOME 和 CANN 安装路径。

**结果显示 70 到 80 ms**

这通常是整段 device wall time，不是本文指标。必须读取
l2_swimlane_records.json 的 fdwic_events，并按本文章节计算 Submit span。

### 建议保存的复现证据

每次正式复现至少保留：

~~~bash
git rev-parse HEAD
git status --short
git diff -- \
    examples/a5/fully_distributed_within_core/paged_attention_unroll/test_paged_attention_unroll.py \
    src/a5/platform/onboard/host/aicpu_topology_probe.cpp \
    src/a5/platform/onboard/host/aicpu_topology_probe.h

python --version
python -c 'import torch; print(torch.__version__)'
g++-15 --version | head -n 1
grep -E '^(Version|timestamp)=' \
    "$ASCEND_HOME_PATH/opp/version.info"
git -C "$PTO_ISA_ROOT" rev-parse HEAD
~~~

同时归档 pytest 完整日志和对应的 l2_swimlane_records.json。这样可以区分
代码变化、工具链变化、设备占用和真实性能回归。
