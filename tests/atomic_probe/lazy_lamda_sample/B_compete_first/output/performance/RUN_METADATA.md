# Final b256 run provenance

- Variant: `compete-first`;
- collection interval: 2026-07-20 07:02:23–07:12:45 UTC;
- direct device0 execution, 96 workers, b256;
- 24 independent host launches and one device run per launch;
- all six A/B/C launch permutations repeated four times; launch positions 1/2/3 are 8/8/8;
- PMU explicitly off, runtime swimlane explicitly off, real-compute;
- executable source: this package's clean `build/compete-first`;
- exact final ELF copy: `artifacts/measured/pa_scheduler_kernel.o`;
- final ELF SHA256: `82a27e206ee1b2411964c0530c73adf92a2b032ac202a7c65c6b5f7d76a4571b`;
- host SHA256: `ee86b1e9457a190ba4594372853492997831f954953e1d037e34655fff212e70`.

`task-submit` and `npu-smi` were unavailable in the collection shell. The run used the user's explicit
authorization to access device0 directly and was not queue-isolated. All raw logs passed execution,
semantic and postprocess checks. Raw evidence is retained; the primary statistic applies the common
per-variant Hampel rule documented in the root performance comparison.
