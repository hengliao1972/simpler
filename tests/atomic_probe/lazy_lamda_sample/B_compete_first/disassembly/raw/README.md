# Final-linked A5 disassembly

Every non-empty final `.text` `STT_FUNC` symbol was extracted from the published final ELF and decoded with the verified `dav_3510` PEM decoder.
RVec `.vector.thread` symbols are passed through an explicit full-body RVec range; all other symbols use scalar decoding.
Each compressed file records final PC, function-relative offset, machine word, mnemonic, ELF/body hashes and decoder identity.
Function gaps are alignment/padding outside symbol bodies and are hashed in `gaps.tsv`; they are not presented as instructions.

Decoder SHA256: `29835d2439d6dd464d34a212ad4bbd5c29af6a38465da09a6c273401d9a96dcb`

| Variant | Final .text | Function symbols | Function bytes | Coverage | Final ELF SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| compete-first | 547640 B | 17 | 547176 B | 99.915% | `82a27e206ee1b2411964c0530c73adf92a2b032ac202a7c65c6b5f7d76a4571b` |

Inspect with `zless FILE.asm.gz`; the sibling `annotated/` directory adds DWARF-mapped local source and comments.
