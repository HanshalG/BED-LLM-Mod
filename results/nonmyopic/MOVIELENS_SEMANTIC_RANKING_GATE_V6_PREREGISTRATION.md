# MovieLens Semantic Lookahead Ranking Gate v6

Date: 2026-07-24

Status: preregistered; no v6 response or endpoint viewed.

The final 15 untouched users from the established population are screened in frozen
seed-24307 order. The first four with best-of-16 EIG at least `.02` are enrolled before
outcomes; fewer than four stops after 30 calls. The scorer then receives generated
profiles, profile-conditioned candidate likelihoods, four candidate metadata rows, and
eight downstream movie metadata rows, but no ratings or outcomes. Non-thinking Gemma
returns four expected downstream semantic-belief-revision scores. Immediate EIG and a
seeded random choice are controls.

After scoring, all four recorded-rating branches are regenerated and evaluated exactly
as in v5. Full execution is 66 requests. Passage requires: semantic Spearman with
negative branch NLL at least `.25`; mean semantic top-1 regret at least `.02` lower
than immediate EIG; semantic choice beats immediate on at least 2/4 users; and semantic
regret no worse than seeded random. Exact requests, enrollment, and zero reasoning must
also pass. Failure closes this scorer; passage alone authorizes a fresh paired policy
test. No threshold changes or post-outcome exclusions.

The first formal attempt stopped after all 30 initial calls and before enrollment,
scoring, or outcomes because a mathematically exact `0.98` probability row summed to
`0.9799999999999999` in binary floating point. Before continuation, the parser
tolerance was amended by `1e-12` and a resume path frozen. It replays the exact 30 raw
responses in memory, includes their original usage in the 66-call accounting, and
makes only the remaining scorer/branch calls. No response, user, threshold, or endpoint
is resampled.
