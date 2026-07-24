# MovieLens Prospective Uncertainty-Enriched Gate v5

Date: 2026-07-24

Status: preregistered. No v5 response or endpoint has been viewed.

V2 showed load-bearing but concentrated profile dynamics; v4 confirmed that only
about one third of fresh users have a natural semantic support with maximum legal-query
EIG above `0.02`. V5 makes that condition prospective rather than selecting a post-hoc
successful subset.

Seed `24306` freezes two smoke-only users and an ordered 48-user screening cohort,
all disjoint from v1-v4. Screening uses the v4 procedure: six broad Gemma profiles and
history-free GPT Mini likelihoods over 16 target-blind legal candidates plus eight
held-out movie metadata rows. No candidate or held-out rating is read during screening.

The first 12 users in frozen screening order with maximum EIG at least `0.02` are
enrolled. If fewer than 12 qualify, stop after exactly 96 calls and read no outcomes.
If 12 qualify, preserve those exact in-memory profiles and likelihoods, reveal all four
top-EIG recorded candidate ratings, regenerate six profiles per branch, and evaluate
held-out NLL. Total requests are exactly 192.

The conditional claim and numerical mechanism gates are frozen: mean oracle held-out
NLL improvement `>=0.05`; at least 6/12 improve by `0.05`; at least 6/12 have branch
spread `>=0.10`; mean immediate-EIG regret `>=0.03`; and at least 4/12 have regret
`>=0.05`. Profile sensitivity is guaranteed only by prospective eligibility and is
not an outcome. All other v2/v4 history-isolation, private-raw, non-copy support, and
zero-reasoning requirements remain.

The exact 10-call smoke uses two disjoint users and tests mechanics only. Passage of
the formal conjunction authorizes a fresh ranking-fidelity scorer; failure closes this
conditional apparatus. No threshold changes, reordered enrollment, replacement users,
or post-outcome exclusions.

OpenRouter ledger ceiling `$70.38480269545715`; run cap `$2.00`; projected formal
reservation `$1.30`; concurrency `64`. Check live and ledger balances before each paid
stage and use the lower remainder.
