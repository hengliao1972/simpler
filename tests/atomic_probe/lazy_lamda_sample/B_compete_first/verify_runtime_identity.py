#!/usr/bin/env python3
"""Prove that measured and local-rebuild AICore ELFs have identical runtime content."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path


SHT_NOBITS = 8
SHT_SYMTAB = 2
STT_FUNC = 2
STT_OBJECT = 1


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def c_string(table: bytes, offset: int) -> str:
    end = table.find(b"\0", offset)
    if end < 0:
        raise ValueError("unterminated ELF string")
    return table[offset:end].decode("utf-8")


@dataclass(frozen=True)
class Section:
    index: int
    name: str
    section_type: int
    flags: int
    address: int
    offset: int
    size: int
    link: int
    alignment: int
    entry_size: int


class Elf64:
    def __init__(self, path: Path):
        self.path = path
        self.data = path.read_bytes()
        header = struct.unpack_from("<16sHHIQQQIHHHHHH", self.data, 0)
        if header[0][:6] != b"\x7fELF\x02\x01":
            raise ValueError(f"not ELF64 little-endian: {path}")
        section_offset, section_size, count, names_index = header[6], header[11], header[12], header[13]
        if section_size != 64:
            raise ValueError(f"unexpected section header size: {section_size}")
        raw = [
            struct.unpack_from("<IIQQQQIIQQ", self.data, section_offset + i * section_size)
            for i in range(count)
        ]
        names_header = raw[names_index]
        names = self.data[names_header[4] : names_header[4] + names_header[5]]
        self.sections = []
        for index, item in enumerate(raw):
            name_offset, section_type, flags, address, offset, size, link, _, alignment, entry_size = item
            self.sections.append(
                Section(
                    index, c_string(names, name_offset), section_type, flags, address,
                    offset, size, link, alignment, entry_size
                )
            )
        self.by_name = {section.name: section for section in self.sections}

    def content(self, section: Section) -> bytes:
        if section.section_type == SHT_NOBITS:
            return b""
        return self.data[section.offset : section.offset + section.size]

    def runtime_symbols(self) -> dict[str, tuple[int, int, int, int]]:
        symtab = next(section for section in self.sections if section.section_type == SHT_SYMTAB)
        strings = self.content(self.sections[symtab.link])
        symbols: dict[str, tuple[int, int, int, int]] = {}
        for offset in range(symtab.offset, symtab.offset + symtab.size, symtab.entry_size):
            name_offset, info, _, section_index, value, size = struct.unpack_from(
                "<IBBHQQ", self.data, offset
            )
            symbol_type = info & 0xF
            if symbol_type not in {STT_FUNC, STT_OBJECT} or section_index >= len(self.sections):
                continue
            section = self.sections[section_index]
            if section.name not in {".text", ".rodata", ".bl_uninit"}:
                continue
            name = c_string(strings, name_offset)
            if name:
                symbols[name] = (symbol_type, info >> 4, value, size)
        return symbols


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    measured_dir = script_dir / "artifacts" / "measured"
    rebuilt_dir = script_dir / "artifacts" / "rebuilt"
    measured = Elf64(measured_dir / "pa_scheduler_kernel.o")
    rebuilt = Elf64(rebuilt_dir / "pa_scheduler_kernel.o")

    content_sections = [
        ".text",
        ".rodata",
        ".ascend.meta.pa_scheduler_0_mix_aic",
        "__CCE_KernelArgSize",
        ".ascend.meta.pa_scheduler_0_mix_aiv",
    ]
    section_rows = []
    all_ok = True
    for name in content_sections:
        left = measured.by_name[name]
        right = rebuilt.by_name[name]
        metadata_equal = (
            left.section_type, left.flags, left.address, left.size, left.alignment
        ) == (
            right.section_type, right.flags, right.address, right.size, right.alignment
        )
        content_equal = measured.content(left) == rebuilt.content(right)
        all_ok &= metadata_equal and content_equal
        section_rows.append(
            {
                "name": name,
                "size": left.size,
                "sha256": digest(measured.content(left)),
                "metadata_equal": metadata_equal,
                "content_equal": content_equal,
            }
        )

    nobits_rows = []
    for name in [".bl_uninit"]:
        if name not in measured.by_name and name not in rebuilt.by_name:
            continue
        if name not in measured.by_name or name not in rebuilt.by_name:
            all_ok = False
            nobits_rows.append({"name": name, "layout_equal": False})
            continue
        left = measured.by_name[name]
        right = rebuilt.by_name[name]
        layout_equal = (
            left.section_type, left.flags, left.address, left.size, left.alignment
        ) == (
            right.section_type, right.flags, right.address, right.size, right.alignment
        )
        all_ok &= layout_equal and left.section_type == SHT_NOBITS
        nobits_rows.append(
            {
                "name": name,
                "type": "SHT_NOBITS",
                "size": left.size,
                "alignment": left.alignment,
                "layout_equal": layout_equal,
            }
        )

    measured_symbols = measured.runtime_symbols()
    rebuilt_symbols = rebuilt.runtime_symbols()
    symbols_equal = measured_symbols == rebuilt_symbols
    all_ok &= symbols_equal
    measured_host = (measured_dir / "pa_scheduler_host").read_bytes()
    rebuilt_host = (rebuilt_dir / "pa_scheduler_host").read_bytes()
    host_equal = measured_host == rebuilt_host
    all_ok &= host_equal

    document = {
        "schema": "pa_lazy_lambda_runtime_identity/v1",
        "variant": (script_dir / "VARIANT").read_text(encoding="utf-8").strip(),
        "status": "PASS" if all_ok else "FAIL",
        "measured_elf_sha256": digest(measured.data),
        "rebuilt_elf_sha256": digest(rebuilt.data),
        "full_elf_equal": measured.data == rebuilt.data,
        "runtime_content_sections": section_rows,
        "nobits_layout": nobits_rows,
        "runtime_symbols_equal": symbols_equal,
        "runtime_symbol_count": len(measured_symbols),
        "host_equal": host_equal,
        "host_sha256": digest(measured_host),
        "note": "Measured is the exact performance binary; full ELF and runtime identities are checked explicitly.",
    }
    output = script_dir / "artifacts" / "runtime_identity.json"
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not all_ok:
        raise SystemExit(f"runtime identity check failed; see {output}")
    print(
        f"PASS variant={document['variant']} runtime_sections={len(section_rows)} "
        f"runtime_symbols={len(measured_symbols)} host_sha256={document['host_sha256']}"
    )


if __name__ == "__main__":
    main()
