#!/usr/bin/env python3
"""Add DWARF-mapped source and local source comments to one raw disassembly."""

from __future__ import annotations

import argparse
import gzip
import os
import re
import subprocess
from pathlib import Path


INSTRUCTION_RE = re.compile(r"^(0x([0-9a-fA-F]+) \(\+0x[0-9a-fA-F]+\).*)$")
LOCATION_RE = re.compile(r"^(.*):(\d+)(?: \(discriminator \d+\))?$")
ANCHOR_RE = re.compile(r"^(.*):(\d+)$")


def read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as source:
            return source.read()
    return path.read_text(encoding="utf-8")


def normalize_source_path(path_text: str, source_root: Path) -> tuple[Path | None, str]:
    if path_text in {"??", ""}:
        return None, "??"
    # Some CCEC line-table rows are rendered as /path/file:?:0.  The final
    # :0 is parsed as the line number, leaving :? attached to the path.
    if path_text.endswith(":?"):
        path_text = path_text[:-2]
    candidate = Path(path_text)
    normalized = Path(candidate.as_posix().replace("/ccec/../common/", "/common/"))
    parts = normalized.parts
    if "lazy_lamda_sample" in parts:
        marker = len(parts) - 1 - list(reversed(parts)).index("lazy_lamda_sample")
        tail = parts[marker + 1 :]
        source_index = next(
            (index for index, part in enumerate(tail) if part in {"common", "ccec"}),
            None,
        )
        if source_index is not None:
            suffix = Path(*tail[source_index:])
            local = source_root / suffix
            return (local if local.is_file() else None), suffix.as_posix()
    if candidate.is_file():
        ascend_home = os.environ.get("ASCEND_HOME_PATH")
        if ascend_home:
            try:
                relative = candidate.resolve().relative_to(Path(ascend_home).resolve())
            except ValueError:
                pass
            else:
                return candidate, f"$ASCEND_HOME_PATH/{relative.as_posix()}"
        return candidate, candidate.as_posix()
    return None, candidate.as_posix()


def source_lines(path: Path | None, cache: dict[Path, list[str]]) -> list[str] | None:
    if path is None:
        return None
    if path not in cache:
        cache[path] = path.read_text(encoding="utf-8").splitlines()
    return cache[path]


def write_output(path: Path, text: str) -> None:
    if path.suffix == ".gz":
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                compressed.write(text.encode("utf-8"))
        return
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=script_dir)
    parser.add_argument("--anchor", help="First DWARF location matching SUFFIX:LINE")
    parser.add_argument("--before", type=int, default=40)
    parser.add_argument("--after", type=int, default=160)
    args = parser.parse_args()

    raw_lines = read_text(args.raw.resolve()).splitlines()
    instruction_lines = [line for line in raw_lines if INSTRUCTION_RE.match(line)]
    if not instruction_lines:
        raise SystemExit(f"No instruction rows in {args.raw}")
    pcs = [int(INSTRUCTION_RE.match(line).group(2), 16) for line in instruction_lines]
    process = subprocess.run(
        ["addr2line", "-e", str(args.elf.resolve()), "-C"],
        input="".join(f"0x{pc:x}\n" for pc in pcs),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    locations = process.stdout.splitlines()
    if len(locations) != len(instruction_lines):
        raise SystemExit(
            f"addr2line returned {len(locations)} locations for {len(instruction_lines)} instructions"
        )

    normalized_locations: list[tuple[Path | None, str, int]] = []
    source_root = args.source_root.resolve()
    for location in locations:
        if location.endswith(":?"):
            local_path, display = normalize_source_path(location[:-2], source_root)
            normalized_locations.append((local_path, display, 0))
            continue
        match = LOCATION_RE.match(location)
        if match is None:
            normalized_locations.append((None, location, 0))
            continue
        local_path, display = normalize_source_path(match.group(1), source_root)
        normalized_locations.append((local_path, display, int(match.group(2))))

    start = 0
    end = len(instruction_lines)
    if args.anchor:
        anchor_match = ANCHOR_RE.match(args.anchor)
        if anchor_match is None:
            raise SystemExit("--anchor must be SUFFIX:LINE")
        suffix = anchor_match.group(1)
        anchor_line = int(anchor_match.group(2))
        anchor_index = next(
            (
                index
                for index, (_, display, line) in enumerate(normalized_locations)
                if display.endswith(suffix) and line == anchor_line
            ),
            None,
        )
        if anchor_index is None:
            raise SystemExit(f"DWARF anchor not found: {args.anchor}")
        start = max(0, anchor_index - args.before)
        end = min(len(instruction_lines), anchor_index + args.after + 1)

    headers = [line for line in raw_lines if line.startswith("#")]
    output = headers + [
        "# annotation_schema=pa_source_annotated_disassembly/v1",
        "# annotation_rule=DWARF supplies only file:line; SOURCE rows are copied from local source files",
        "# annotation_warning=comments have source context only and do not own an exact machine address",
        f"# annotation_instruction_slice={start}:{end}",
        "#",
    ]
    cache: dict[Path, list[str]] = {}
    previous_display = ""
    previous_line = 0
    for index in range(start, end):
        local_path, display, line_number = normalized_locations[index]
        if display != previous_display or line_number != previous_line:
            lines = source_lines(local_path, cache)
            output.append(f"# [DWARF] {display}:{line_number}")
            if lines is None or line_number <= 0 or line_number > len(lines):
                output.append("# [SOURCE unavailable]")
            else:
                if display == previous_display and previous_line < line_number <= previous_line + 12:
                    context_start = previous_line + 1
                else:
                    context_start = max(1, line_number - 6)
                for source_line in range(context_start, line_number + 1):
                    marker = ">" if source_line == line_number else " "
                    output.append(f"# {marker} {source_line:5d} | {lines[source_line - 1]}")
            previous_display = display
            previous_line = line_number
        output.append(instruction_lines[index])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_output(args.output, "\n".join(output) + "\n")
    print(
        f"wrote {args.output}: instructions={end - start} "
        f"range=0x{pcs[start]:x}..0x{pcs[end - 1]:x}"
    )


if __name__ == "__main__":
    main()
