# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check documentation and non-comment source text for non-English scripts.

Source comments may be written in Chinese. String literals, identifiers, and
documentation remain subject to the English-only check.
"""

import argparse
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path
from typing import Optional

# Default excluded directories (can be overridden via --exclude)
DEFAULT_EXCLUDED_PATTERNS = ["3rdparty", "reference", "docs/zh-cn", "README.zh-CN.md"]

C_FAMILY_EXTENSIONS = {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx"}
PYTHON_EXTENSIONS = {".py", ".pyi"}


def _c_raw_string_end(text: str, index: int) -> Optional[int]:
    """Return the first index after a C++ raw string, or None."""
    if not text.startswith('R"', index):
        return None
    opening = text.find("(", index + 2, min(len(text), index + 19))
    if opening < 0:
        return None
    delimiter = text[index + 2 : opening]
    if len(delimiter) > 16 or any(char.isspace() or char in "\\()" for char in delimiter):
        return None
    terminator = ")" + delimiter + '"'
    closing = text.find(terminator, opening + 1)
    return None if closing < 0 else closing + len(terminator)


def _c_quoted_literal_end(text: str, index: int) -> Optional[int]:
    """Return the first index after a regular C/C++ string or character."""
    if text[index] not in {'"', "'"}:
        return None
    quote = text[index]
    index += 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        current = text[index]
        index += 1
        if current == quote:
            break
    return index


def _mask_c_family_comments(text: str) -> str:
    """Replace C/C++ comments with spaces while preserving strings and lines."""
    masked = list(text)
    index = 0
    size = len(text)
    while index < size:
        if text.startswith("//", index):
            while index < size and text[index] != "\n":
                masked[index] = " "
                index += 1
            continue
        if text.startswith("/*", index):
            while index < size:
                if text[index] != "\n":
                    masked[index] = " "
                if text.startswith("*/", index):
                    if index + 1 < size:
                        masked[index + 1] = " "
                    index += 2
                    break
                index += 1
            continue

        # Raw string bodies are code data, not comments. Recognize the R"tag(
        # opener even when it follows a u8/u/U/L prefix.
        raw_end = _c_raw_string_end(text, index)
        if raw_end is not None:
            index = raw_end
            continue
        quoted_end = _c_quoted_literal_end(text, index)
        if quoted_end is not None:
            index = quoted_end
            continue
        index += 1
    return "".join(masked)


def _mask_python_comments(text: str) -> str:
    """Replace Python COMMENT tokens with spaces while preserving line layout."""
    lines = [list(line) for line in text.splitlines(keepends=True)]
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            row, start = token.start
            _, end = token.end
            for column in range(start, end):
                lines[row - 1][column] = " "
    except (IndentationError, tokenize.TokenError):
        # Fail closed: malformed source is checked without masking.
        return text
    return "".join("".join(line) for line in lines)


def mask_source_comments(text: str, suffix: str) -> str:
    """Mask comments for supported source languages; leave docs untouched."""
    if suffix in C_FAMILY_EXTENSIONS:
        return _mask_c_family_comments(text)
    if suffix in PYTHON_EXTENSIONS:
        return _mask_python_comments(text)
    return text


def get_git_tracked_files(root_dir: Path) -> list[Path]:
    """Get list of files tracked by git."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        files = []
        for line in result.stdout.strip().split("\n"):
            if line:
                file_path = root_dir / line
                if file_path.is_file():
                    files.append(file_path)
        return files
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to get git tracked files: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: git command not found", file=sys.stderr)
        sys.exit(1)


def contains_non_english(text: str) -> tuple[bool, list[tuple[int, str]]]:
    """
    Check if text contains non-English characters.

    Returns:
        Tuple of (has_non_english, list of (line_number, non-english text) tuples)
    """
    # Unicode ranges for common non-English scripts:
    # - Chinese (CJK Unified Ideographs): \u4e00-\u9fff
    # - Japanese Hiragana: \u3040-\u309f
    # - Japanese Katakana: \u30a0-\u30ff
    # - Korean Hangul: \uac00-\ud7af
    # - Cyrillic: \u0400-\u04ff
    # - Arabic: \u0600-\u06ff
    # - Hebrew: \u0590-\u05ff
    # - Thai: \u0e00-\u0e7f
    # - CJK Symbols and Punctuation: \u3000-\u303f (ideographic period, comma, brackets)
    # - Full-width Latin and Punctuation: \uff01-\uff5e (full-width comma, colon, parens)
    non_english_pattern = re.compile(
        r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af"
        r"\u0400-\u04ff\u0600-\u06ff\u0590-\u05ff\u0e00-\u0e7f"
        r"\u3000-\u303f\uff01-\uff5e]+"
    )

    violations = []
    lines = text.split("\n")

    for i, line in enumerate(lines, 1):
        matches = non_english_pattern.findall(line)
        if matches:
            # Join all non-English text found in this line
            non_english_text = ", ".join(matches)
            violations.append((i, non_english_text))

    return bool(violations), violations


def check_file_english_only(file_path: Path) -> tuple[bool, list[tuple[int, str]]]:
    """
    Check if a file contains only English text.

    Returns:
        Tuple of (is_english_only, list of (line_number, non_english_text) violations)
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError) as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
        return True, []  # Skip files we can't read

    has_non_english, violations = contains_non_english(mask_source_comments(content, file_path.suffix.lower()))

    return not has_non_english, violations


SOURCE_EXTENSIONS = {".py", ".pyi", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx", ".md"}


def _collect_files(files: list[str], excluded_patterns: list[str]) -> Optional[list[Path]]:
    """Collect files to check, either from explicit list or full-repo scan."""
    if files:
        result = []
        for f in files:
            p = Path(f).resolve()
            if p.is_file() and p.suffix in SOURCE_EXTENSIONS:
                if not any(part in str(f) for part in excluded_patterns):
                    result.append(p)
        return result

    root_path = Path(".").resolve()
    if not (root_path / ".git").exists():
        print(f"Error: '{root_path}' is not a git repository", file=sys.stderr)
        return None

    all_files = get_git_tracked_files(root_path)
    filtered = [f for f in all_files if not any(str(f.relative_to(root_path)).startswith(p) for p in excluded_patterns)]
    return [f for f in filtered if f.suffix in SOURCE_EXTENSIONS]


def main() -> int:
    """Main function."""
    parser = argparse.ArgumentParser(description="Check docs and non-comment source text for non-English scripts")
    parser.add_argument(
        "files",
        nargs="*",
        help="Files to check (default: all git-tracked files)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Additional directory patterns to exclude (can be specified multiple times)",
    )

    args = parser.parse_args()

    excluded_patterns = DEFAULT_EXCLUDED_PATTERNS.copy()
    if args.exclude:
        excluded_patterns.extend(args.exclude)

    files_to_check = _collect_files(args.files, excluded_patterns)
    if files_to_check is None:
        return 1

    if not files_to_check:
        print("No source files found to check")
        return 0

    def _display_path(p: Path) -> str:
        try:
            return str(p.relative_to(Path(".").resolve()))
        except ValueError:
            return str(p)

    print(f"Checking {len(files_to_check)} file(s) for English-only content...")

    failed_files = []
    passed_files = []

    for file_path in files_to_check:
        is_english_only, violations = check_file_english_only(file_path)

        if is_english_only:
            passed_files.append(file_path)
            if args.verbose:
                print(f"✓ {_display_path(file_path)}")
        else:
            failed_files.append((file_path, violations))
            for line_num, non_english_text in violations[:5]:
                print(f"✗ {_display_path(file_path)}:{line_num} {non_english_text}")
            if len(violations) > 5:
                print(f"  ... and {len(violations) - 5} more line(s) in {_display_path(file_path)}")

    print()
    print(f"Results: {len(passed_files)} passed, {len(failed_files)} failed")

    if failed_files:
        print("\nFiles with non-English content:")
        for file_path, _ in failed_files:
            print(f"  - {_display_path(file_path)}")
        print("\n⚠ Please keep documentation and non-comment source text in English.")
        return 1

    print("\n✓ Documentation and non-comment source text are in English!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
