# SWE-smith Debug-BED source V2 result

Date: 2026-08-13

Status: **source passed; zero-call mechanics only authorized**

## Result

The distinct V2 metadata-only source audit passes every frozen gate. It binds:

- 50,137 SWE-smith tasks across 128 repositories and 128 execution images;
- 5,865 same-file and 4,227 same-module composed defects;
- at least one fail-to-pass and one pass-to-pass test for every composed task;
- exact DebugGym code, tools, split config, and eleven content-addressed
  SWE-smith Parquet shards;
- native persistent PDB experiments, official test execution,
  regression-protected scoring, seedable state, and upstream-remote removal;
- released effective splits of 760 development and 121 confirmation tasks;
- a disjoint hash-ordered allocation of 8 mechanics, 64 opportunity, and 9,731
  reserve composed tasks.

V2 corrects only the source interpretation of DebugGym's published exclusions:
the raw named lists remain 789/125, while the released loader materializes them
as 760/121 after exclusions. V1 remains failed and is not retroactively rescued.

## Scientific meaning

This is a source admission, not evidence that non-myopic debugging works.
Composed bugs supply plausible multi-stage diagnostic structure, and the LLM
would irreducibly generate semantic bug hypotheses and observation likelihoods.
The required next gate must still show that native debugger observations create
an adaptive experiment dependency and that an exact depth-two diagnostic root
beats compute-matched receding myopic on at least five of eight mechanics tasks.

Only that zero-model-call execution mechanics gate is authorized. Opportunity,
development, confirmation, LLM serving, patch endpoints, and paper efficacy
claims remain sealed.

## Privacy and accounting

No individual instance ID, problem statement, patch, test name, test output,
repository source, gold fix, or endpoint was serialized or opened.

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none

## Bindings

- V1 protocol SHA-256: `576a044d94e7547eb1e7fa4233e17889c93c307b853c7d3c9388ea84cdfa8170`
- V1 result: `SWESMITH_DEBUG_BED_SOURCE_V1_RESULT_20260813.md`
- V2 protocol SHA-256: `f26b468deac21d730abd10569b1ce86c9147dd4b127fad48b77d1591d075902e`
- V2 canonical manifest digest:
  `f3cdc0006a77875760de9bfcd7d3676bfdb28032f2c957a1f303ee7b839985ce`
- V2 implementation: `scripts/swesmith_debug_bed_source_v2_audit.py`

Next: push this immutable boundary, then freeze and run the eight-task native
execution mechanics gate before any model call.
