# ClariQ Multisample Likelihood Holdout Manifest Amendment

## Status

Frozen after the preregistered key-only holdout selector and before any model
call or selected-topic utility-value access.

## Manifest

- Path:
  `results/nonmyopic/clariq_multisample_likelihood_holdout_manifest/MANIFEST.json`
- SHA-256:
  `889a9a952945fea0c5cd910d559c1e12c9e40f5a9500f65810b0ce6a0eb82cfa`
- Selected topics: `12`.
- Structurally eligible holdout topics found: `16`.
- Evaluation keys inspected: yes.
- Evaluation utility values read: no.
- API calls: `0`.

The first 12 eligible topics and valid-root counts are:

| Topic | Facets | Roots | Initial request |
|---|---:|---:|---|
| `115` | 5 | 15 | Tell me about the pacific northwest laboratory. |
| `10` | 6 | 14 | Where can I find cheap internet |
| `104` | 3 | 12 | Child support in Indiana? |
| `14` | 4 | 12 | I'm interested in dinosaurs |
| `135` | 3 | 12 | Tell me about source of the nile |
| `122` | 3 | 13 | Tell me more about Culpeper National Cemetry |
| `105` | 3 | 10 | Tell me about sonoma county medical services. |
| `131` | 3 | 14 | What is equal opportunity employer? |
| `113` | 3 | 12 | Tell me more about HP mini 2140 |
| `137` | 3 | 15 | tell me about rock and gem shows |
| `119` | 3 | 12 | How to write a thank you letter after an interview? |
| `116` | 4 | 10 | What is California Franchise Tax Board |

The complete facet descriptions and sorted root question objects are
hash-locked in the manifest.

## Exact Requests

There are 151 roots. Five samples per root produce exactly 755 physical
requests and HTTP attempts. The runner rejects any manifest hash, topic order,
facet count, root count, or request count change.

All model, prompt, estimator, policy, control, stability, endpoint, inference,
threshold, budget, and no-OatML conditions remain unchanged.
