# Number Game Qwen Pooled First-Link Confirmation-32 Preregistration

Date frozen: 2026-07-29, after the exact-10 pooled-support mechanics gate
passed and before any seed in `63100..63555` was sent to a model.

## Fixed Study

- 32 fresh trees, seeds `63100..63131`.
- Qwen 3.7 Plus nonreasoning planning with two independent draws at every one
  of 49 planning histories; second-draw seed offset `1,000,000`.
- Gemini 2.5 Flash single target draw, seeds `63200..63231`.
- Eight independent Gemini cross-fit validation supports per tree, seeds
  `63300..63555`.
- Retained rejuvenation, equal tree weighting, and the unchanged exact
  33-concept Tenenbaum--Griffiths endpoint.
- 20,000 tree-bootstrap samples, seed `63600`.

Every Qwen draw uses the strict schema plus preregistered complete-item
isolation. The two valid supports are deduplicated by executable extension
before posterior construction and lookahead. There is no semantic repair,
continuation, or endpoint-informed filtering.

## Primary Gates

On depth-three-versus-myopic changed roots:

- at least 28 changed roots;
- mean realized Brier advantage at least `0.008` and bootstrap lower above
  zero;
- simulated-to-realized Spearman at least `0.25` and bootstrap lower above
  zero;
- wins minus losses at least eight.

At policy level, depth three must reduce Brier versus myopic by at least 8%,
have a paired tree-bootstrap difference interval below zero, and win at least
20 trees.

## Mechanics Gates

- Exactly 3,424 accepted requests and exact attempt accounting.
- At most 24 retries and provider-error retries.
- Zero reasoning tokens and forced exits; cost at most `$5.25`.
- Exactly 1,568 pooled planning parse events and 288 single target/validation
  events.
- At most 16 provider draws use complete-item salvage.
- Every pooled initial support has at least 24 valid extensions.
- Every pooled generated first- and second-step support has at least eight.
- Every retained first support has at least 12 and every retained second
  support at least eight.
- All eight validation supports per tree have at least 16 valid extensions.

Starting provider-visible balance must be at least `$5.50`.

Depth two, fixed-support depth three, PTS, random roots, Hamming, coverage, and
ranking remain diagnostics. The result is reported once as `passed`,
`gated_null`, or `failed_closed`. There is no continuation, seed replacement,
third draw, threshold change, or response repair. This cohort cannot alter
the status of any prior run.
