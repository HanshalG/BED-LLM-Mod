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
