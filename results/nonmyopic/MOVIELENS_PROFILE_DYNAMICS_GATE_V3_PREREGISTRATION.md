# MovieLens Candidate-Contrastive Profile-Dynamics Gate v3

Date: 2026-07-24

Status: preregistered mechanism gate. No v3 response or endpoint has been viewed.
This is not a policy or depth comparison.

## Motivation And Distinct Intervention

V2 made semantic regeneration load-bearing and passed mean oracle improvement, branch
spread, and immediate-EIG regret gates. It failed because only 5/12 users improved by
`0.05` and only 5/12 had maximum immediate EIG above `0.02`.

Private raw-output diagnosis found that low-EIG users had lexically varied but
semantically repetitive broad taste summaries. Their mean pairwise L1 distances among
candidate rating distributions were as low as `0.058` and `0.071`. V3 changes the
belief representation, not the thresholds: every hypothesis must explicitly contrast
its qualitative reaction to all four fixed design movies.

## Frozen Fresh Population

The data hashes, eligibility rule, four public-history movies, four candidate movies,
and eight-rating held-out endpoint are unchanged. Seed `24304` samples 14 of the 21
remaining eligible users after excluding every v1 and v2 user, using rating presence
only:

- smoke-only users: `26, 63`;
- formal users: `144, 178, 268, 293, 303, 345, 417, 425, 486, 487, 624, 663`.

Smoke and formal users are disjoint. Seed `24304000 + user_id` selects each held-out
set. Held-out ratings and user IDs never enter model prompts.

## Candidate-Contrastive Profiles

Gemma 4 26B A4B remains non-thinking. Each of six initial and refreshed hypotheses
contains:

- a complete semantic taste description;
- one `appeal`, `avoid`, or `uncertain` reaction and rationale for each fixed candidate
  movie;
- for refreshes, an explicit newest-evidence effect.

Every profile must use at least two reaction labels. For every candidate movie, the six
profiles must contain at least two distinct labels. Parsers fail closed otherwise.
Candidate movies enter only as public metadata without their ratings. Parsed reaction
text is included in the semantic profile passed to the likelihood model.

GPT-5.4 Mini remains non-reasoning and history-free: rating likelihood prompts receive
only the complete semantic profiles and movie metadata. Thus all prediction updates
remain load-bearing on generated hypotheses. Branch support remains six replacements
plus the two old profiles most compatible with the realized rating.

Raw responses remain ignored/private under `external/`; committed results contain
only counts, hashes, public movie metadata, and metrics.

## Serving Gate

The two smoke-only users run one Contact branch and one exact refresh replay each.
Passage requires exactly 10 requests, zero reasoning tokens, valid contrastive schemas,
six non-copy replacements, eight-profile branch supports, and no observed-history
payload in likelihood prompts. Manual review must find coherent, evidence-compatible
candidate contrasts rather than arbitrary label permutation.

## Formal Gate

The disjoint formal run remains exactly 120 requests over 12 users and four branches.
All v2 numerical conditions are unchanged:

1. exact completion and zero reasoning;
2. every refreshed support satisfies the contrastive replacement schema;
3. mean oracle held-out NLL improvement at least `0.05`;
4. at least 6/12 users improve by at least `0.05`;
5. at least 6/12 users have branch NLL spread at least `0.10`;
6. mean immediate-EIG held-out NLL regret at least `0.03`;
7. at least 4/12 users have immediate-EIG regret at least `0.05`;
8. mean maximum immediate EIG at least `0.02` and at least 8/12 users have maximum EIG
   at least `0.02`.

Passage authorizes only a fresh target-blind scorer/ranking-fidelity gate on users not
used by v1-v3. Failure closes this exact candidate-contrastive apparatus before policy
or depth evaluation. No threshold tuning, post-hoc subset, alternate split, history
bypass, or reaction-diversity relaxation is allowed after responses.

OpenRouter ledger ceiling: `$70.38480269545715`; per-run cap `$0.75`; projected formal
reservation `$0.50`; concurrency `64`. Check live and ledger balances before every
paid stage and use the lower remainder.
