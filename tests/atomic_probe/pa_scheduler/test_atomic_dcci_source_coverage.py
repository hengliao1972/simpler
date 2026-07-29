#!/usr/bin/env python3
"""检查 standalone PA 生产头中的 atomic/DCCI 观察覆盖。

这个门槛刻意先于生产代码改造落地：只要 common 头文件仍直接调用受控
Ops 原语，测试就逐条报告文件、行号和源码。以后新增调用必须接入统一
观察封装；真正发生在 trace 建立前或仅供测试的例外，则在调用同一行或
紧邻上一行写出以下一种标记，并说明具体原因：

    // PA_ATOMIC_DCCI_SOURCE_EXEMPT: pretrace - <原因>
    // PA_ATOMIC_DCCI_SOURCE_EXEMPT: test-only - <原因>
    // PA_ATOMIC_DCCI_SOURCE_EXEMPT: aggregate - <原因>
    // PA_ATOMIC_DCCI_SOURCE_EXEMPT: trace-free - <原因>

不能用宽泛目录白名单掩盖新调用。唯一的结构性例外是 pa_trace.h 的中央
观察实现，以及 pa_shared_heap.h 中 ObserveAtomics=false 的两个中央
编译期 fallback。
"""

from __future__ import annotations

import ast
import re
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
COMMON = ROOT / "common"
MODEL = COMMON / "pa_model.h"
HOST_SUPPORT = COMMON / "host_support.h"
CONVERTER = ROOT / "swimlane_converter.py"

CONTROLLED_OPS: Sequence[str] = (
    "Load",
    "Exchange",
    "FetchAdd",
    "FetchMax",
    "CompareExchange",
    "InvalidateRegion",
    "FlushRegion",
)
CONTROLLED_CALL = re.compile(
    r"\b(?:Ops|[A-Za-z_]\w*Ops)\s*::\s*("
    + "|".join(CONTROLLED_OPS)
    + r")\s*\("
)
EXEMPT_PREFIX = "PA_ATOMIC_DCCI_SOURCE_EXEMPT:"
EXEMPT_MARKER = re.compile(
    r"//\s*PA_ATOMIC_DCCI_SOURCE_EXEMPT:\s*"
    r"(pretrace|test-only|aggregate|trace-free)\s*-\s*(\S.*)$"
)

# 这两个调用是 shared heap 的中央“无观察构建”编译期出口。精确匹配文本
# 并校验各自只出现一次，防止把整个文件变成不受检查的区域。
SHARED_HEAP_FALLBACKS: Mapping[str, str] = {
    "Load": "return Ops::Load(address);",
    "FetchAdd": "return Ops::FetchAdd(address, value);",
}

ATOMIC_OP_IDS: Mapping[str, int] = {
    "Load": 0,
    "Exchange": 1,
    "FetchAdd": 2,
    "FetchMax": 3,
    "CompareExchange": 4,
}
DCCI_OP_IDS: Mapping[str, int] = {
    "Invalidate": 0,
    "CleanOut": 1,
}


@dataclass(frozen=True)
class SourceCall:
    path: Path
    line: int
    operation: str
    text: str

    def diagnostic(self) -> str:
        return f"{self.path.relative_to(ROOT)}:{self.line}: {self.text}"


def _strip_cpp_comments_and_literals(source: str) -> str:
    """屏蔽注释与字面量，同时保留字符位置和换行，供正则计算准确行号。"""

    output = list(source)
    index = 0
    state = "code"
    quote = ""
    while index < len(source):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""

        if state == "code":
            if char == "/" and next_char == "/":
                output[index] = output[index + 1] = " "
                index += 2
                state = "line_comment"
                continue
            if char == "/" and next_char == "*":
                output[index] = output[index + 1] = " "
                index += 2
                state = "block_comment"
                continue
            if char in ('"', "'"):
                quote = char
                output[index] = " "
                index += 1
                state = "literal"
                continue
            index += 1
            continue

        if state == "line_comment":
            if char == "\n":
                state = "code"
            else:
                output[index] = " "
            index += 1
            continue

        if state == "block_comment":
            if char == "*" and next_char == "/":
                output[index] = output[index + 1] = " "
                index += 2
                state = "code"
            else:
                if char != "\n":
                    output[index] = " "
                index += 1
            continue

        # string/character literal
        if char == "\\":
            output[index] = " "
            if index + 1 < len(source):
                if source[index + 1] != "\n":
                    output[index + 1] = " "
                index += 2
            else:
                index += 1
            continue
        if char == quote:
            output[index] = " "
            index += 1
            state = "code"
            continue
        if char != "\n":
            output[index] = " "
        index += 1

    return "".join(output)


def _source_calls(path: Path) -> List[SourceCall]:
    source = path.read_text(encoding="utf-8")
    searchable = _strip_cpp_comments_and_literals(source)
    lines = source.splitlines()
    calls: List[SourceCall] = []
    for match in CONTROLLED_CALL.finditer(searchable):
        line = searchable.count("\n", 0, match.start()) + 1
        calls.append(
            SourceCall(
                path=path,
                line=line,
                operation=match.group(1),
                text=lines[line - 1].strip(),
            )
        )
    return calls


def _has_explicit_exemption(call: SourceCall) -> bool:
    lines = call.path.read_text(encoding="utf-8").splitlines()
    # 标记只绑定同一行或紧邻上一行，避免一个标记无意豁免整个代码块。
    for line_index in (call.line - 1, call.line - 2):
        if line_index < 0:
            continue
        if EXEMPT_MARKER.search(lines[line_index]):
            return True
    return False


def _is_shared_heap_fallback(call: SourceCall) -> bool:
    expected = SHARED_HEAP_FALLBACKS.get(call.operation)
    return (
        call.path.name == "pa_shared_heap.h"
        and expected is not None
        and call.text == expected
    )


def _parse_atomic_site_enum() -> tuple[Dict[str, int], int]:
    source = _strip_cpp_comments_and_literals(
        MODEL.read_text(encoding="utf-8")
    )
    match = re.search(
        r"enum\s+class\s+AtomicSite\s*:\s*uint32_t\s*\{(.*?)\};",
        source,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"{MODEL}: 找不到 AtomicSite 枚举")

    sites: Dict[str, int] = {}
    count: int | None = None
    for item in match.group(1).split(","):
        item = item.strip()
        if not item:
            continue
        entry = re.fullmatch(r"([A-Za-z_]\w*)\s*=\s*(\d+)", item)
        if entry is None:
            raise AssertionError(
                f"{MODEL}: 无法解析 AtomicSite 条目: {item!r}"
            )
        name, raw_value = entry.groups()
        value = int(raw_value)
        if name == "Count":
            count = value
        else:
            sites[name] = value
    if count is None:
        raise AssertionError(f"{MODEL}: AtomicSite 缺少 Count")
    return sites, count


def _parse_host_atomic_site_names() -> List[str]:
    source = HOST_SUPPORT.read_text(encoding="utf-8")
    function = re.search(
        r"inline\s+const\s+char\s*\*\s*AtomicSiteName\s*\("
        r".*?\)\s*\{(.*?)\n\}",
        source,
        re.DOTALL,
    )
    if function is None:
        raise AssertionError(f"{HOST_SUPPORT}: 找不到 AtomicSiteName")
    names = re.search(
        r"const\s+char\s*\*\s*names\s*\[\s*\]\s*=\s*\{(.*?)\};",
        function.group(1),
        re.DOTALL,
    )
    if names is None:
        raise AssertionError(f"{HOST_SUPPORT}: 找不到 AtomicSiteName::names")
    return re.findall(r'"([^"]+)"', names.group(1))


def _parse_python_literal_assignments(
    path: Path, requested: Iterable[str]
) -> Dict[str, object]:
    requested_set = set(requested)
    parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: Dict[str, object] = {}
    for node in parsed.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in requested_set:
            continue
        values[target.id] = ast.literal_eval(node.value)
    missing = requested_set - values.keys()
    if missing:
        raise AssertionError(
            f"{path}: 缺少 Python 映射: {', '.join(sorted(missing))}"
        )
    return values


def _parse_cpp_atomic_site_ops(
    sites: Mapping[str, int]
) -> Dict[int, int]:
    source = _strip_cpp_comments_and_literals(
        MODEL.read_text(encoding="utf-8")
    )
    function = re.search(
        r"AtomicSiteExpectedOp\s*\(\s*AtomicSite\s+site\s*\)\s*\{"
        r"(.*?)\n\}",
        source,
        re.DOTALL,
    )
    if function is None:
        raise AssertionError(f"{MODEL}: 找不到 AtomicSiteExpectedOp")

    # default 分支是 Load；显式 case 只覆盖其它操作。
    result = {value: ATOMIC_OP_IDS["Load"] for value in sites.values()}
    groups = re.finditer(
        r"((?:case\s+AtomicSite::[A-Za-z_]\w*\s*:\s*)+)"
        r"return\s+AtomicOp::([A-Za-z_]\w*)\s*;",
        function.group(1),
        re.DOTALL,
    )
    for group in groups:
        operation = group.group(2)
        if operation not in ATOMIC_OP_IDS:
            raise AssertionError(
                f"{MODEL}: 未知 AtomicOp::{operation}"
            )
        for name in re.findall(
            r"case\s+AtomicSite::([A-Za-z_]\w*)\s*:",
            group.group(1),
        ):
            if name == "Count":
                continue
            if name not in sites:
                raise AssertionError(
                    f"{MODEL}: AtomicSiteExpectedOp 引用了未知站点 {name}"
                )
            result[sites[name]] = ATOMIC_OP_IDS[operation]
    return result


def _parse_dcci_site_enum() -> tuple[Dict[str, int], int]:
    source = _strip_cpp_comments_and_literals(
        MODEL.read_text(encoding="utf-8")
    )
    match = re.search(
        r"enum\s+class\s+DcciSite\s*:\s*uint32_t\s*\{(.*?)\};",
        source,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"{MODEL}: 找不到 DcciSite 枚举")
    sites: Dict[str, int] = {}
    count: int | None = None
    for item in match.group(1).split(","):
        item = item.strip()
        if not item:
            continue
        entry = re.fullmatch(r"([A-Za-z_]\w*)\s*=\s*(\d+)", item)
        if entry is None:
            raise AssertionError(
                f"{MODEL}: 无法解析 DcciSite 条目: {item!r}"
            )
        name, raw_value = entry.groups()
        value = int(raw_value)
        if name == "Count":
            count = value
        else:
            sites[name] = value
    if count is None:
        raise AssertionError(f"{MODEL}: DcciSite 缺少 Count")
    return sites, count


def _parse_host_dcci_site_names() -> List[str]:
    source = HOST_SUPPORT.read_text(encoding="utf-8")
    function = re.search(
        r"inline\s+const\s+char\s*\*\s*DcciSiteName\s*\("
        r".*?\)\s*\{(.*?)\n\}",
        source,
        re.DOTALL,
    )
    if function is None:
        raise AssertionError(f"{HOST_SUPPORT}: 找不到 DcciSiteName")
    names = re.search(
        r"const\s+char\s*\*\s*names\s*\[\s*\]\s*=\s*\{(.*?)\};",
        function.group(1),
        re.DOTALL,
    )
    if names is None:
        raise AssertionError(f"{HOST_SUPPORT}: 找不到 DcciSiteName::names")
    return re.findall(r'"([^"]+)"', names.group(1))


def _parse_cpp_dcci_site_ops(
    sites: Mapping[str, int]
) -> Dict[int, int]:
    source = _strip_cpp_comments_and_literals(
        MODEL.read_text(encoding="utf-8")
    )
    function = re.search(
        r"DcciSiteExpectedOp\s*\(\s*DcciSite\s+site\s*\)\s*\{"
        r"(.*?)\n\}",
        source,
        re.DOTALL,
    )
    if function is None:
        raise AssertionError(f"{MODEL}: 找不到 DcciSiteExpectedOp")
    result = {
        value: DCCI_OP_IDS["Invalidate"] for value in sites.values()
    }
    groups = re.finditer(
        r"((?:case\s+DcciSite::[A-Za-z_]\w*\s*:\s*)+)"
        r"return\s+DcciOp::([A-Za-z_]\w*)\s*;",
        function.group(1),
        re.DOTALL,
    )
    for group in groups:
        operation = group.group(2)
        if operation not in DCCI_OP_IDS:
            raise AssertionError(f"{MODEL}: 未知 DcciOp::{operation}")
        for name in re.findall(
            r"case\s+DcciSite::([A-Za-z_]\w*)\s*:",
            group.group(1),
        ):
            if name not in sites:
                raise AssertionError(
                    f"{MODEL}: DcciSiteExpectedOp 引用了未知站点 {name}"
                )
            result[sites[name]] = DCCI_OP_IDS[operation]
    return result


class AtomicDcciSourceCoverageTest(unittest.TestCase):
    maxDiff = None

    def test_production_headers_have_no_bare_controlled_ops(self) -> None:
        violations: List[SourceCall] = []
        heap_fallback_counts = {
            operation: 0 for operation in SHARED_HEAP_FALLBACKS
        }

        for path in sorted(COMMON.glob("*.h")):
            # TraceAtomic/TraceDCCI 的中央实现必须最终落到真实 Ops。
            if path.name == "pa_trace.h":
                continue
            for call in _source_calls(path):
                if _is_shared_heap_fallback(call):
                    heap_fallback_counts[call.operation] += 1
                    continue
                if _has_explicit_exemption(call):
                    continue
                violations.append(call)

        self.assertEqual(
            heap_fallback_counts,
            {operation: 1 for operation in SHARED_HEAP_FALLBACKS},
            "pa_shared_heap.h 应且只应保留两个 ObserveAtomics=false "
            f"中央 fallback，实际为 {heap_fallback_counts}",
        )
        if violations:
            rendered = "\n".join(
                f"  {call.diagnostic()}" for call in violations
            )
            self.fail(
                "发现未接入统一观察封装的裸 atomic/DCCI 调用 "
                f"({len(violations)}处)：\n{rendered}"
            )

    def test_exemption_markers_are_well_formed(self) -> None:
        malformed: List[str] = []
        for path in sorted(COMMON.glob("*.h")):
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if EXEMPT_PREFIX in line and EXEMPT_MARKER.search(line) is None:
                    malformed.append(
                        f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}"
                    )
        if malformed:
            self.fail(
                "发现格式错误或缺少具体原因的源码豁免标记：\n  "
                + "\n  ".join(malformed)
            )

    def test_atomic_site_mapping_counts_and_ops_match(self) -> None:
        sites, count = _parse_atomic_site_enum()
        expected_ids = list(range(count))
        self.assertEqual(
            sorted(sites.values()),
            expected_ids,
            "AtomicSite 必须显式编号且在 [0, Count) 内连续",
        )

        host_names = _parse_host_atomic_site_names()
        self.assertEqual(
            len(host_names),
            count,
            "host AtomicSiteName 数量必须等于 AtomicSite::Count",
        )

        python_values = _parse_python_literal_assignments(
            CONVERTER, ("ATOMIC_SITE_NAMES", "ATOMIC_SITE_OP_IDS")
        )
        python_names = python_values["ATOMIC_SITE_NAMES"]
        python_ops = python_values["ATOMIC_SITE_OP_IDS"]
        self.assertIsInstance(python_names, dict)
        self.assertIsInstance(python_ops, dict)
        self.assertEqual(
            sorted(python_names),
            expected_ids,
            "Python ATOMIC_SITE_NAMES 键必须完整覆盖 [0, Count)",
        )
        self.assertEqual(
            sorted(python_ops),
            expected_ids,
            "Python ATOMIC_SITE_OP_IDS 键必须完整覆盖 [0, Count)",
        )
        self.assertEqual(
            python_ops,
            _parse_cpp_atomic_site_ops(sites),
            "Python site->op 映射必须与 C++ AtomicSiteExpectedOp 一致",
        )

    def test_dcci_site_mapping_counts_and_ops_match(self) -> None:
        sites, count = _parse_dcci_site_enum()
        expected_ids = list(range(count))
        self.assertEqual(
            sorted(sites.values()),
            expected_ids,
            "DcciSite 必须显式编号且在 [0, Count) 内连续",
        )
        self.assertEqual(
            len(_parse_host_dcci_site_names()),
            count,
            "host DcciSiteName 数量必须等于 DcciSite::Count",
        )
        python_values = _parse_python_literal_assignments(
            CONVERTER, ("DCCI_SITE_NAMES", "DCCI_SITE_OP_IDS")
        )
        python_names = python_values["DCCI_SITE_NAMES"]
        python_ops = python_values["DCCI_SITE_OP_IDS"]
        self.assertIsInstance(python_names, dict)
        self.assertIsInstance(python_ops, dict)
        self.assertEqual(sorted(python_names), expected_ids)
        self.assertEqual(sorted(python_ops), expected_ids)
        self.assertEqual(
            python_ops,
            _parse_cpp_dcci_site_ops(sites),
            "Python DCCI site->op 映射必须与 C++ DcciSiteExpectedOp 一致",
        )


if __name__ == "__main__":
    unittest.main()
