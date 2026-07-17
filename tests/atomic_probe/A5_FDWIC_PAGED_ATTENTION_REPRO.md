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
l2_swimlane_records.json 中读取。

### 已验证环境

| 项目 | 本次验证值 |
| --- | --- |
| 验证日期 | 2026-07-17 |
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
| simpler 分支 | fdwic-swimlane-deps |
| simpler 实测基准 HEAD | 52ca4f5eba343c2f7b7a3a743e575cb9308d128f |

系统的 /etc/os-release 标签为 Ubuntu 20.04.6，但实际
getconf GNU_LIBC_VERSION 输出 glibc 2.39。判断 GCC 15 二进制兼容性时，
应以实际 glibc 和 ldd 结果为准，不能只看发行版标签。

### 已验证结果

| 检查项 | 结果 |
| --- | --- |
| A5Sim Case1 | PASSED，约 71.62 s |
| A5 Case1 正确性 | PASSED |
| A5 swimlane Case1 | PASSED，pytest 约 85.48 s |
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
    --enable-l2-swimlane \
    --use-example-exec-time \
    --clone-protocol https \
    --pto-isa-commit "$PTO_ISA_COMMIT" \
    --pto-session-timeout 1200 \
    --require-pto-isa \
    -s -v
~~~

--use-example-exec-time 仅适用于 fully_distributed_within_core 的 sim。
它不能用于真实 A5 命令。

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

### 运行 A5 swimlane 性能复现

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

本次实测生成：

~~~text
outputs/TestPagedAttentionUnroll_Case1_20260717_023809/
l2_swimlane_records.json
~~~

新的复现会生成不同时间戳目录。trace 约几十 MiB，pytest 结束后再读取，
不要用 pytest wall time 代替 Submit 指标。

### 提取首个到末个 Submit

以下脚本自动选择最新 Case1 trace，校验 96 个 core、每核 1280 个 Submit
以及完整 task id，并输出用户关注的全局 span：

~~~bash
python - <<'PY'
import json
import statistics
from collections import defaultdict
from pathlib import Path

traces = list(
    Path("outputs").glob(
        "TestPagedAttentionUnroll_Case1_*/l2_swimlane_records.json"
    )
)
if not traces:
    raise SystemExit("no Case1 l2_swimlane_records.json found")

trace = max(traces, key=lambda path: path.stat().st_mtime)
with trace.open() as stream:
    data = json.load(stream)

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
