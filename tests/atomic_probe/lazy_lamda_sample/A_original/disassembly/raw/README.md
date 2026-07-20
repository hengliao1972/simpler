# Final-linked A5 disassembly

Every non-empty final `.text` `STT_FUNC` symbol was extracted from the published final ELF and decoded with the verified `dav_3510` PEM decoder.
RVec `.vector.thread` symbols are passed through an explicit full-body RVec range; all other symbols use scalar decoding.
Each compressed file records final PC, function-relative offset, machine word, mnemonic, ELF/body hashes and decoder identity.
Function gaps are alignment/padding outside symbol bodies and are hashed in `gaps.tsv`; they are not presented as instructions.

Decoder SHA256: `29835d2439d6dd464d34a212ad4bbd5c29af6a38465da09a6c273401d9a96dcb`

| Variant | Final .text | Function symbols | Function bytes | Coverage | Final ELF SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| original | 780344 B | 11 | 779700 B | 99.917% | `76c961f846c7efc89b40942c3a8113530a6da2c7bfbdc601615852f8c5b79cfc` |

Inspect with `zless FILE.asm.gz`; the sibling `annotated/` directory adds DWARF-mapped local source and comments.
