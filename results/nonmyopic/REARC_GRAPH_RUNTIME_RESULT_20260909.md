# Isolated RE-ARC graph runtime mechanics

Implemented graph execution inside the already available pinned Python container
image989bb9480c98. Only three temporary read-only files are mounted: the hash-bound
upstream DSL, the graph validator, and the worker. The repository, task solvers,
example banks, user home and .env are not mounted. Candidate input is JSON only;
no submitted Python is exec'd or imported.

Controls: UID65534, network none, read-only root, dropped capabilities,
no-new-privileges,32process cap,256MiB memory+swap ceiling,1CPU quota,2CPU-second
worker limit and15second host wall timeout. The host always attempts named-container
removal. Input is capped at64KiB, graph128steps, input/output grids30x30/colors0..9,
and returned stdout16KiB. The output cap is checked after capture; the trusted
worker emits only a validated grid or a short error, not arbitrary candidate logs.

An initial test import failed because the container-only validator import was at
module scope; moving it inside main fixed host unit-test collection. Twelve grid
and graph tests then passed in .13s. A hand-built vmirror returned [[2,1],[4,3]].
The banked smoke verifies:

- Higher-order composition of two mirrors returns the original grid.
- Unknown eval operation is rejected with exit1.
- Deliberately excessive nested power work is terminated with exit137 under the
  configured resource limits. That code alone does not distinguish CPU from OOM;
  no specific kill cause is claimed.
- Normal completion reports nonroot UID, denied filesystem/network probes and no
  OpenRouter key. No bed-rearc containers remain after the checks.

This is a resource-bounded container boundary, not a proof against all container
escape vulnerabilities or all DSL bugs. The pinned image is the previously
available Python3.8 image; dependency age remains a limitation. Ordinary graph
type errors and resource termination remain explicit failures, not silent repair.

No selected benchmark task was executed, no generated examples opened, no LLM
called and no API cost incurred. The source-generator runtime is a SEPARATE next
dependency: generators require reviewed random helpers and bounded generation
attempts, and cannot be run as arbitrary code via this graph endpoint.

Next freeze source-generation seeds/counts and source-validity handling for all
four selected tasks before executing them. Then qualify proposals with same-runtime
symbolic and equal-call blind controls. Do not mistake runtime success for useful
hypotheses or non-myopic planning opportunity.

Previous turn was source/representation progress; current turn provides tested
isolated execution needed for an actual proposal study. Cost$0, balance23.693468061,
conservative daily remaining4.11174654 unchanged. Goal active/unachieved.
