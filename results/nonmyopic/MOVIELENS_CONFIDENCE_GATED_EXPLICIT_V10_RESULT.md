# MovieLens Confidence-Gated Explicit V10 Result

Date: 2026-07-24

Run: `movielens-confidence-v10-formal-20260724T224619Z`

Status: **failed the frozen pre-outcome enrollment gate; no confidence-policy
endpoint was produced.**

All 44 fresh screen users completed their initial profile and likelihood calls.
Only 14 had maximum immediate EIG at least `.02`, versus the preregistered 16
required for enrollment. Mean maximum EIG was `.01867783`, also below `.02`.

The run therefore stopped after exactly 88 screening requests. It did not
generate a hypothetical explicit-rollout tree, read a candidate rating, read a
held-out rating, select a confidence-gated query, or compute an efficacy
metric. Lowering the threshold, enrolling 14, replacing the history, or
choosing another cohort would repair the sample after observing the screen and
is not permitted.

The screen used zero reasoning tokens, retries, forced exits, or runtime
failures and cost `$0.63094681`. Private raw responses are checkpointed under
SHA-256
`81fc5b2bf9a110a2d0efe1741c176948fdebbb338fb516b2659c1bb7171d5bfc`.

This closes the exact confidence-gated one-round route without evidence for or
against its efficacy. The V7 four-user ranking result remains positive
mechanism evidence; V8 and V9 remain the relevant failed policy tests.
