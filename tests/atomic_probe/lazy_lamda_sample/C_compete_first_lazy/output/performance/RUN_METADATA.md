# Final b256 run provenance

- Variant: `compete-first-lazy`;
- collection interval: 2026-07-20 07:02:23–07:12:45 UTC;
- direct device0 execution, 96 workers, b256;
- 24 independent host launches and one device run per launch;
- all six A/B/C launch permutations repeated four times; launch positions 1/2/3 are 8/8/8;
- PMU explicitly off, runtime swimlane explicitly off, real-compute;
- executable source: this package's clean `build/compete-first-lazy`;
- exact final ELF copy: `artifacts/measured/pa_scheduler_kernel.o`;
- final ELF SHA256: `8d293c52312429d13efe89592ed705c672ceef1f2875b5e3bd25582419d37893`;
- host SHA256: `d6d9e9bb04e28c7682e47230e20722f1e7843937b50017ddfe8a515acbe0ccd9`.

`task-submit` and `npu-smi` were unavailable in the collection shell. The run used the user's explicit
authorization to access device0 directly and was not queue-isolated. All raw logs passed execution,
semantic and postprocess checks. Raw evidence is retained; the primary statistic applies the common
per-variant Hampel rule documented in the root performance comparison.
