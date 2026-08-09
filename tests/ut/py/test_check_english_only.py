# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
from pathlib import Path


def _load_checker():
    checker_path = Path(__file__).resolve().parents[2] / "lint" / "check_english_only.py"
    spec = importlib.util.spec_from_file_location("check_english_only", checker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECKER = _load_checker()
CJK = "\u4e2d\u6587"


def test_c_family_comments_may_contain_chinese():
    source = f"int value = 1; // {CJK}\n/* {CJK} */\n"
    masked = CHECKER.mask_source_comments(source, ".cpp")
    assert not CHECKER.contains_non_english(masked)[0]


def test_c_family_strings_are_still_checked():
    source = f'const char *value = "// {CJK}";\nconst char *raw = R"(/* {CJK} */)";\n'
    masked = CHECKER.mask_source_comments(source, ".cpp")
    assert CHECKER.contains_non_english(masked)[0]


def test_python_comments_may_contain_chinese_but_strings_are_checked():
    comment = CHECKER.mask_source_comments(f"value = 1  # {CJK}\n", ".py")
    string = CHECKER.mask_source_comments(f'value = "{CJK}"\n', ".py")
    assert not CHECKER.contains_non_english(comment)[0]
    assert CHECKER.contains_non_english(string)[0]
