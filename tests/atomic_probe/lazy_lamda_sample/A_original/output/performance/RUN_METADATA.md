# Final b256 run provenance

- Variant: `original`;
- collection interval: 2026-07-20 07:02:23–07:12:45 UTC;
- direct device0 execution, 96 workers, b256;
- 24 independent host launches and one device run per launch;
- all six A/B/C launch permutations repeated four times; launch positions 1/2/3 are 8/8/8;
- PMU explicitly off, runtime swimlane explicitly off, real-compute;
- executable source: this package's clean `build/original`;
- exact final ELF copy: `artifacts/measured/pa_scheduler_kernel.o`;
- final ELF SHA256: `76c961f846c7efc89b40942c3a8113530a6da2c7bfbdc601615852f8c5b79cfc`;
- host SHA256: `97cf327377857bcab61cbca10357a90bc49cd5c389c919873c2f3209d5b01188`.

`task-submit` and `npu-smi` were unavailable in the collection shell. The run used the user's explicit
authorization to access device0 directly and was not queue-isolated. All raw logs passed execution,
semantic and postprocess checks. Raw evidence is retained; the primary statistic applies the common
per-variant Hampel rule documented in the root performance comparison.
