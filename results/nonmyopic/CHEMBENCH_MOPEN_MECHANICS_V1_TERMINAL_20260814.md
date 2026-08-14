# ChemBench M-open Mechanics V1 Terminal Result

Date: 2026-08-14 (Europe/London)

## Disposition

V1 is terminal `failed_closed` before scientific execution. The exact pushed
command at commit `704d13e2debdb3d333ad6e4a94a161aac1f93d7d` raised `KeyError:
'v3'` while looking up the first active domain,
`c0_michaelis_menten`.

The pinned source stores only `v0-v2` parameter states for the nine original
simple active mechanisms. The 48 generated compound and novel active
mechanisms store `v0-v5`. V1 incorrectly required a common `v3` key across all
57 models.

## Exposure Audit

- Source binding and active-domain metadata were read.
- The public proxy-query coordinates were generated from the frozen seed.
- The response loop stopped on the first parameter lookup, before obtaining a
  rate function or evaluating any assay or proxy endpoint.
- No `v3` rate, observation mean, categorical likelihood, target feature,
  policy action, truth loss, transition, aggregate, or gate value was
  constructed.
- No result file or transition bank exists.
- Model/API calls: `0`.
- Cost: `$0`.

Therefore the 48 outside-support `v3` parameter states remain scientifically
unopened. V1 itself and its exact all-57/common-version interface are closed.

## Successor Authority

A prospective V2 may retain the unopened outside-support `v3` cohort while
using the already opened `v2` representatives for the nine simple initial
candidate structures. It must evaluate only the 48 outside-support worlds as
truth cells, preflight every per-model version before generating query
coordinates, use new artifact paths and schema, add an end-to-end mixed-version
source test, and be committed and pushed before execution.

This source-schema null has no scientific interpretation and does not weaken or
rescue any non-myopic gate.
