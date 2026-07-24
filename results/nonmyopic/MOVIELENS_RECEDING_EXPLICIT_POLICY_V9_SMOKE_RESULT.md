# MovieLens Receding Explicit Policy v9 Smoke Result

Date: 2026-07-24

Run: `movielens-receding-policy-v9-smoke-20260724T123941Z`

Status: passed; the preregistered formal policy gate is authorized.

The serving smoke completed exactly 22 physical requests with zero reasoning
tokens and cost `$0.14814151`. It produced one initial state, one shared
round-two policy state, and complete top-one-by-five transition trees at both
rounds. Every policy path selected two distinct queries.

All probability rows were valid without tolerance-based normalization. Candidate
ratings were read only after each round's transition tree was frozen, and
held-out ratings were read only after all policy paths were fixed. The smoke's
single-candidate design forces the receding explicit, immediate-EIG, and seeded
random policies to coincide, so its equal final NLL values are not an efficacy
result.

Raw model text remains private and untracked. Its SHA-256 is
`0b607d36a8266d1e74a8a24c489f3d8d3bf8871023e6e3ecc84b7a5a70516002`.
