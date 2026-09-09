# Herb runtime and interpreter bridge

The actual pinned Herb iterator now runs, and its converted candidates execute
in the existing isolated Hodel interpreter. This is engineering evidence, not
predictive-transfer qualification or a non-myopic result.

## Runtime

Julia1.10.10-bookworm image digest
2323a3445e7701cf8f7190293e360e4f67c7e43de24480183f178aa1062dc99b.
Herb04de909d and HerbSearchc7abd554 resolved successfully with their declared
dependencies. Project.toml and complete Manifest.toml are banked in
herb_bridge_runtime_20260909. Dependency installation used a new named Docker
volume bed-herb-runtime-20260909, no host repository or credentials mounted,
240second timeout, 2GiB memory/2CPU limit. Installation had network access;
candidate enumeration and execution did not.

Iterator runs used read-only filesystem and dependency volume, uid65534,
no network, no capabilities, no-new-privileges, 128process limit, 2GiB/2CPU,
180second external timeout. Compiled modules were disabled to avoid writing
precompile caches. Only the Julia smoke script was mounted from the project.
All containers exited; none remain running.

## Observations

- Cost-based enumeration with explicit program_to_outputs=nothing emitted
  128 candidates. A separately constructed second iterator emitted exactly the
  same ordered candidates. This tests fresh-state replay, not the unbuilt
  adaptive history-conditioned branch updater.
- Generic recursive expression lowering produces canonical graph references,
  including callable intermediates and shared repeated subexpressions. The
  existing strict graph validator accepts all128 records.
- There are127 distinct graphs: bare I and identity(I) both lower to identity(I),
  because the existing graph format requires a step output. A production support
  collector must deduplicate canonical graphs before assigning its declared prior.
- Seven selected fixtures executed with exact expected grids: identity, mirror,
  horizontal concatenation, row-wise identity, composed callable identity,
  repeating the first row, and object-based painting. Resizing is preserved.
- Identity and mirror both produce[[0,0]] on the demonstration and different
  outputs on[[1,2]]. Their syntactically distinct hypotheses are not merged.
- Execution receipts verify uid65534, read-only filesystem, network denial,
  and no OpenRouter key. Only handcrafted grids were used, no benchmark labels.

The first bridge emitted fn instead of the existing op field. It was corrected
before sandbox execution; original output is retained as v1_wrong_field.toml,
with a regression requiring rejection. An initial DAG test assumed a particular
expression occurred within128 candidates; it did not. The test now checks an
actually emitted repeated mirror subexpression, without changing search limits.
Six focused tests pass in.23s, including the earlier exact ambiguity regression.

## Remaining dependency

The smoke grammar is deliberately an engineering fixture, NOT a replacement
for the full160-operation Hodel language. This does not qualify semantic typing,
candidate coverage, guided-versus-uniform efficiency, or arbitrary higher-order
application. The next step is a generic full-export grammar/collector, checked
against the pinned DSL signatures and public-only executable proposals. Preserve
callable/row/object/resizing operations, canonical deduplication, equal work,
independent branch state, and search-guidance/prior separation. Do not select a
restricted first-order benchmark to turn this smoke into a positive claim.
Only independently qualified predictive transfer and branch fidelity can justify
fresh depth experiments. Closed Luna-medium cohorts remain closed.

Previous goal turn was a status restatement with no progress. This turn produces
new runtime evidence and a working bridge. No API calls/cost; authenticated
balance23.609293221, conservative London-day allowance4.0275717 unchanged.
Full goal remains active and unachieved. No cluster or automation changes.
