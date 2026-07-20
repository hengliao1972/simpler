#!/usr/bin/env python3
"""Disassemble every final-linked STT_FUNC with the verified A5 PEM decoder."""

from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import io
import re
import struct
from dataclasses import dataclass
from pathlib import Path


EXPECTED_DECODER_SHA256 = "29835d2439d6dd464d34a212ad4bbd5c29af6a38465da09a6c273401d9a96dcb"
INSTRUCTION_RE = re.compile(r"^\s*([0-9a-fA-F]+):\s+((?:[0-9a-fA-F]{8})+)\s+(.*?)\s*$")
BAD_MNEMONICS = {"UNDEF", "UNKNOWN", "INVALID", "ILLEGAL", "ERROR", "<NOT"}


def sha256(data: bytes) -> str:
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
    address: int
    offset: int
    size: int
    link: int
    entry_size: int


@dataclass(frozen=True)
class Function:
    name: str
    value: int
    size: int
    binding: str


class Elf64:
    def __init__(self, path: Path):
        self.path = path
        self.data = path.read_bytes()
        header_format = "<16sHHIQQQIHHHHHH"
        if len(self.data) < struct.calcsize(header_format):
            raise ValueError(f"truncated ELF: {path}")
        header = struct.unpack_from(header_format, self.data, 0)
        ident = header[0]
        if ident[:4] != b"\x7fELF" or ident[4] != 2 or ident[5] != 1:
            raise ValueError(f"expected ELF64 little-endian image: {path}")
        section_header_offset = header[6]
        section_header_size = header[11]
        section_count = header[12]
        section_name_index = header[13]
        if section_header_size != 64 or section_count == 0:
            raise ValueError(f"unsupported section table: {path}")

        raw_sections = []
        section_format = "<IIQQQQIIQQ"
        for index in range(section_count):
            offset = section_header_offset + index * section_header_size
            raw_sections.append(struct.unpack_from(section_format, self.data, offset))
        shstr = raw_sections[section_name_index]
        shstr_data = self.data[shstr[4] : shstr[4] + shstr[5]]
        self.sections = []
        for index, item in enumerate(raw_sections):
            name_offset, section_type, _, address, offset, size, link, _, _, entry_size = item
            self.sections.append(
                Section(
                    index=index,
                    name=c_string(shstr_data, name_offset),
                    section_type=section_type,
                    address=address,
                    offset=offset,
                    size=size,
                    link=link,
                    entry_size=entry_size,
                )
            )
        self.text = next(section for section in self.sections if section.name == ".text")

    def section_bytes(self, section: Section) -> bytes:
        return self.data[section.offset : section.offset + section.size]

    def functions(self) -> list[Function]:
        symtab = next(section for section in self.sections if section.section_type == 2)
        if symtab.entry_size != 24 or symtab.size % symtab.entry_size != 0:
            raise ValueError(f"unsupported symbol table: {self.path}")
        strings = self.section_bytes(self.sections[symtab.link])
        functions = []
        for offset in range(symtab.offset, symtab.offset + symtab.size, symtab.entry_size):
            name_offset, info, _, section_index, value, size = struct.unpack_from(
                "<IBBHQQ", self.data, offset
            )
            if info & 0xF != 2 or section_index != self.text.index or size == 0:
                continue
            binding = {0: "LOCAL", 1: "GLOBAL", 2: "WEAK"}.get(info >> 4, str(info >> 4))
            functions.append(Function(c_string(strings, name_offset), value, size, binding))
        functions.sort(key=lambda function: (function.value, function.name))
        if not functions:
            raise ValueError(f"no final-linked .text functions: {self.path}")
        return functions


class PemDecoder:
    def __init__(self, path: Path):
        self.path = path
        decoder_bytes = path.read_bytes()
        self.digest = sha256(decoder_bytes)
        if self.digest != EXPECTED_DECODER_SHA256:
            raise ValueError(
                f"unsupported PEM decoder SHA256 {self.digest}; expected {EXPECTED_DECODER_SHA256}"
            )
        library = ctypes.CDLL(str(path))
        self.dump = library.pem_turbo_objdump
        self.dump.argtypes = (
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        )
        self.dump.restype = ctypes.c_void_p
        self.libc = ctypes.CDLL(None)
        self.libc.free.argtypes = (ctypes.c_void_p,)

    def decode(self, body: bytes, rvec: bool) -> list[tuple[int, str, str]]:
        if len(body) % 4 != 0:
            raise ValueError("function body is not an integral number of instruction words")
        word_count = len(body) // 4
        words = (ctypes.c_uint32 * word_count).from_buffer_copy(body)
        range_pointer = None
        range_count = 0
        if rvec:
            ranges_type = (ctypes.c_uint64 * 2) * 1
            ranges = ranges_type()
            ranges[0][0] = 0
            ranges[0][1] = len(body)
            range_pointer = ctypes.cast(ranges, ctypes.c_void_p)
            range_count = 1
        result = self.dump(words, word_count, range_pointer, range_count)
        if not result:
            raise RuntimeError("pem_turbo_objdump returned null")
        try:
            decoded = ctypes.string_at(result).decode("utf-8")
        finally:
            self.libc.free(result)

        instructions = []
        for line in decoded.splitlines():
            match = INSTRUCTION_RE.fullmatch(line)
            if match is None:
                if line.strip():
                    raise ValueError(f"unexpected decoder output: {line!r}")
                continue
            relative = int(match.group(1), 16)
            machine_word = match.group(2).lower()
            mnemonic = match.group(3).strip()
            instructions.append((relative, machine_word, mnemonic))
        expected_relative = 0
        for index, (relative, machine_word, mnemonic) in enumerate(instructions):
            encoded_bytes = len(machine_word) // 2
            if encoded_bytes % 4 != 0:
                raise ValueError(
                    f"decoder returned a non-word-sized encoding at row {index}: {machine_word}"
                )
            expected_word = (
                f"{int.from_bytes(body[expected_relative:expected_relative + encoded_bytes], 'little'):0{len(machine_word)}x}"
            )
            if relative != expected_relative or machine_word != expected_word:
                raise ValueError(
                    f"decoder coverage mismatch at word {index}: {relative:x}/{machine_word} "
                    f"!= {expected_relative:x}/{expected_word}"
                )
            first_token = mnemonic.upper().split(maxsplit=1)[0] if mnemonic else ""
            if first_token in BAD_MNEMONICS or "NOT AVAILABLE" in mnemonic.upper():
                raise ValueError(f"invalid decoded instruction at +0x{relative:x}: {mnemonic}")
            expected_relative += encoded_bytes
        if expected_relative != len(body):
            raise ValueError(
                f"decoder covered {expected_relative} bytes of a {len(body)}-byte function"
            )
        return instructions


def safe_filename(index: int, name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "function"
    suffix = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return f"{index:02d}_{safe[:72]}_{suffix}.asm.gz"


def write_gzip(path: Path, text: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(text.encode("utf-8"))


def disassemble_variant(
    variant: str, elf_path: Path, output_dir: Path, decoder: PemDecoder
) -> tuple[list[str], list[str], dict[str, object]]:
    elf = Elf64(elf_path)
    text = elf.section_bytes(elf.text)
    functions = elf.functions()
    variant_dir = output_dir
    variant_dir.mkdir(parents=True)
    elf_digest = sha256(elf.data)
    text_digest = sha256(text)
    manifest_rows = []
    gap_rows = []
    cursor = elf.text.address
    total_function_bytes = 0

    for index, function in enumerate(functions):
        if function.value < cursor:
            raise ValueError(f"overlapping final function symbols in {elf_path}: {function.name}")
        if function.value > cursor:
            gap_start = cursor
            gap_end = function.value
            start = gap_start - elf.text.address
            gap = text[start : start + gap_end - gap_start]
            gap_rows.append(
                f"0x{gap_start:x}\t0x{gap_end:x}\t{len(gap)}\t{sha256(gap)}"
            )
        body_offset = function.value - elf.text.address
        body = text[body_offset : body_offset + function.size]
        if len(body) != function.size:
            raise ValueError(f"function outside .text: {function.name}")
        rvec = ".vector.thread" in function.name
        try:
            instructions = decoder.decode(body, rvec)
        except Exception as error:
            raise ValueError(f"{variant}:{function.name}: {error}") from error
        body_digest = sha256(body)
        filename = safe_filename(index, function.name)
        header = [
            "# schema=pa_final_linked_disassembly/v1",
            f"# variant={variant}",
            f"# final_elf={elf_path.name}",
            f"# final_elf_sha256={elf_digest}",
            f"# final_text_address=0x{elf.text.address:x}",
            f"# final_text_size={elf.text.size}",
            f"# final_text_sha256={text_digest}",
            f"# symbol={function.name}",
            f"# binding={function.binding}",
            f"# final_pc=0x{function.value:x}",
            f"# size={function.size}",
            f"# instruction_count={len(instructions)}",
            f"# encoded_word_count={function.size // 4}",
            f"# body_sha256={body_digest}",
            "# decoder=$ASCEND_HOME_PATH/x86_64-linux/simulator/dav_3510/lib/libpem_davinci.so",
            f"# decoder_sha256={decoder.digest}",
            f"# decoder_mode={'rvec' if rvec else 'scalar'}",
            "# columns=final_pc function_relative_offset machine_word instruction",
            "",
        ]
        body_lines = [
            f"0x{function.value + relative:016x} (+0x{relative:08x})  {machine_word}  {mnemonic}"
            for relative, machine_word, mnemonic in instructions
        ]
        write_gzip(variant_dir / filename, "\n".join(header + body_lines) + "\n")
        last_mnemonic = instructions[-1][2].split(maxsplit=1)[0]
        manifest_rows.append(
            "\t".join(
                (
                    variant,
                    filename,
                    function.binding,
                    function.name,
                    f"0x{function.value:x}",
                    str(function.size),
                    str(len(instructions)),
                    "rvec" if rvec else "scalar",
                    body_digest,
                    last_mnemonic,
                )
            )
        )
        cursor = function.value + function.size
        total_function_bytes += function.size

    text_end = elf.text.address + elf.text.size
    if cursor < text_end:
        start = cursor - elf.text.address
        gap = text[start:]
        gap_rows.append(f"0x{cursor:x}\t0x{text_end:x}\t{len(gap)}\t{sha256(gap)}")
    elif cursor > text_end:
        raise ValueError(f"function extends past .text: {elf_path}")

    metadata = {
        "variant": variant,
        "elf_sha256": elf_digest,
        "text_address": elf.text.address,
        "text_size": elf.text.size,
        "text_sha256": text_digest,
        "function_count": len(functions),
        "function_bytes": total_function_bytes,
        "function_coverage_percent": total_function_bytes / elf.text.size * 100.0,
        "gap_count": len(gap_rows),
    }
    return manifest_rows, gap_rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    fixed_variant = (script_dir / "VARIANT").read_text(encoding="utf-8").strip()
    parser.add_argument("--variant", default=fixed_variant)
    parser.add_argument(
        "--elf", type=Path,
        default=script_dir / "artifacts" / "measured" / "pa_scheduler_kernel.o"
    )
    parser.add_argument("--output", type=Path, default=script_dir / "disassembly" / "raw")
    default_decoder = None
    import os

    if os.environ.get("ASCEND_HOME_PATH"):
        default_decoder = (
            Path(os.environ["ASCEND_HOME_PATH"])
            / "x86_64-linux/simulator/dav_3510/lib/libpem_davinci.so"
        )
    parser.add_argument("--decoder", type=Path, default=default_decoder, required=default_decoder is None)
    args = parser.parse_args()
    elf_path = args.elf.resolve()
    output_dir = args.output.resolve()
    decoder_path = args.decoder.resolve()
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists; choose a new path: {output_dir}")
    decoder = PemDecoder(decoder_path)

    if not elf_path.is_file():
        raise SystemExit(f"Missing published final ELF: {elf_path}")
    manifest_rows, local_gaps, item = disassemble_variant(
        args.variant, elf_path, output_dir, decoder
    )
    gap_rows = [f"{args.variant}\t{row}" for row in local_gaps]

    manifest_header = (
        "variant\tfile\tbinding\tsymbol\tfinal_pc\tsize\tinstructions\tdecoder_mode\t"
        "body_sha256\tlast_mnemonic"
    )
    (output_dir / "manifest.tsv").write_text(
        manifest_header + "\n" + "\n".join(manifest_rows) + "\n", encoding="utf-8"
    )
    (output_dir / "gaps.tsv").write_text(
        "variant\tstart_pc\tend_pc\tsize\tsha256\n" + "\n".join(gap_rows) + "\n",
        encoding="utf-8",
    )
    readme = [
        "# Final-linked A5 disassembly",
        "",
        "Every non-empty final `.text` `STT_FUNC` symbol was extracted from the published final ELF and decoded with the verified `dav_3510` PEM decoder.",
        "RVec `.vector.thread` symbols are passed through an explicit full-body RVec range; all other symbols use scalar decoding.",
        "Each compressed file records final PC, function-relative offset, machine word, mnemonic, ELF/body hashes and decoder identity.",
        "Function gaps are alignment/padding outside symbol bodies and are hashed in `gaps.tsv`; they are not presented as instructions.",
        "",
        f"Decoder SHA256: `{decoder.digest}`",
        "",
        "| Variant | Final .text | Function symbols | Function bytes | Coverage | Final ELF SHA256 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    readme.append(
        f"| {item['variant']} | {item['text_size']} B | {item['function_count']} | "
        f"{item['function_bytes']} B | {item['function_coverage_percent']:.3f}% | "
        f"`{item['elf_sha256']}` |"
    )
    readme.extend(
        [
            "",
            "Inspect with `zless FILE.asm.gz`; the sibling `annotated/` directory adds DWARF-mapped local source and comments.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    published = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "published.sha256":
            continue
        published.append(f"{sha256(path.read_bytes())}  {path.relative_to(output_dir)}")
    (output_dir / "published.sha256").write_text("\n".join(published) + "\n", encoding="utf-8")
    print("\n".join(readme))


if __name__ == "__main__":
    main()
