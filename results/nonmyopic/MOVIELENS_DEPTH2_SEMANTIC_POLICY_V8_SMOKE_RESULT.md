# MovieLens Depth-2 Semantic Policy v8 Smoke Result

Date: 2026-07-24

Run: `movielens-depth2-policy-v8-smoke2-20260724T115940Z`

Status: passed; the preregistered formal policy gate is authorized.

The fresh retry completed exactly 62 requests with zero reasoning tokens and cost
`$0.28379369`. Private raw counts were exactly `1,1,5,5,25,25` for initial
profiles, initial likelihoods, first-level profiles, first-level likelihoods,
second-level profiles, and terminal likelihoods.

All five first-query outcomes and all 25 nested second-query outcomes completed.
Depth 2, depth 1, and immediate EIG each traversed a valid two-distinct-query path
through the same frozen semantic transition tree. Because the smoke deliberately
uses only one first-query candidate, the policies coincide; it tests interface and
accounting rather than efficacy.

Candidate outcomes were read only after the tree was frozen, and held-out ratings
only after all policy paths were fixed. Raw model text remains private and untracked.
Its SHA-256 is
`da00b41cae0309b9533eda82f24fcd86763b4c9c3047aa5203dfaa0b506bd532`.
