# MuSiQue Chain-Transition V2 Smoke Result

Date: 2026-07-24
Run: `musique-chain-transition-smoke-v2-20260724T195636Z`
Status: **serving smoke passed; frozen formal gate authorized**.

The structural six-root grammar completed exactly 14/14 requests: two initial chain
supports and six document-conditioned refreshes per row. All supports parsed to eight
unique ordered title pairs with six unique first titles. There were zero reasoning
tokens, retries, forced exits, or runtime failures. Cost was `$0.00433163`.

Smoke-only mechanism diagnostics were encouraging but are not efficacy evidence:

| Diagnostic | Row 1 | Row 2 |
|---|---:|---:|
| True root among six actions | yes | yes |
| Gold chain initially omitted | yes | yes |
| Some branch recovered gold chain | yes | no |
| Branch-dependent gold coverage | yes | no |
| Immediate-EIG realized regret | 0 | 0 |

For the first question, opening the true root document was the only one of six branches
that regenerated the correct ordered support pair. The second question did not recover
the complete pair after any first document. These are descriptive smoke observations
only. The 12 formal rows, branch outputs, and frozen conjunction remained unread before
authorization.

Artifacts:

- `results/nonmyopic/musique_chain_transition_smoke/musique-chain-transition-smoke-v2-20260724T195636Z/SERVING_SMOKE.json`
- private raw SHA-256 recorded in that artifact
- preregistration:
  `results/nonmyopic/MUSIQUE_CHAIN_TRANSITION_GATE_PREREGISTRATION.md`
