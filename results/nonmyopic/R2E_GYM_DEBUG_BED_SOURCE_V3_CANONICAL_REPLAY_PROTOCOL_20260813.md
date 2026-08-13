# R2E-Gym Debug-BED source V3 canonical replay protocol

Date frozen: 2026-08-13 Europe/London

Status: **prospective verifier serialization correction; zero model calls**

V1 closed because its runtime checker used the wrong released class name. V2
correctly passed the producer runtime gate and reproduced V1's registered
canonical manifest SHA-256
`bc1c4eb4da8adf91f0fcedab3e4ab8c4c20e17dc0c67ee052e59ea72eacf2dbc`,
but its independent verifier compared Python dictionaries before
serialization. Integer repository-size histogram keys in the recomputation
became strings when the V1 JSON was read, making object equality false despite
byte-identical canonical hashes.

V2 remains terminally closed. V3 changes only
`v1_manifest_reproduced`: it requires the recomputed canonical manifest hash to
equal both V1's registered `manifest_sha256` and the fixed hash above. Every V1
source, projection, threshold, salt, split, privacy, mechanics, and accounting
condition remains exact. V3 must also reproduce all other V2 gates and prove
that V2's only failed gate was `v1_manifest_reproduced` while its recomputed
hash already equaled V1's registered hash.

A V3 pass authorizes only a separately frozen, predicate-pushed retrieval and
structural screen of the exact 16-row prefix. It authorizes no task execution,
model call, endpoint, or efficacy claim. Any further source-verifier failure
closes this route rather than opening another correction.

## Accounting

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none

