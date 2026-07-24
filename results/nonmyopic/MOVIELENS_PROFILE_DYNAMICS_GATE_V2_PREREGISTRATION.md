# MovieLens Load-Bearing Profile-Dynamics Gate v2

Date: 2026-07-24

Status: preregistered mechanism gate. No v2 model response or endpoint has been
viewed. This is not a policy or depth comparison.

## Motivation And Single Repair

The v1 serving smoke completed mechanically, but both refreshed profile sets copied
their initial sets exactly. Its likelihood prompt also received the updated raw rating
history, so held-out predictions could change without semantic hypothesis regeneration.
The v1 formal gate was canceled before launch.

V2 removes that bypass. The likelihood model receives only semantic profile
descriptions and movie metadata. It never receives observed rating history. Initial
and branch predictions can therefore differ only through the profile support produced
by Gemma. The refresh schema also requires every generated profile to describe the
effect of the newest evidence and rejects exact copies of previous descriptions.

All outcomes and endpoints remain recorded MovieLens ratings. There is no LLM
answerer or judge.

## Frozen Population And Splits

The fixed four public-history movies, four candidate movies, data hashes, eligibility
rule, profile count, and eight-rating held-out endpoint are unchanged from v1. Seed
`24303` samples 14 users from the 35 eligible users not selected in v1, using rating
presence only:

- smoke-only users: `294, 327`;
- formal users: `378, 387, 416, 450, 470, 488, 533, 537, 580, 650, 676, 699`.

Smoke and formal users are disjoint. For each user, seed `24303000 + user_id` selects
eight other rated films as the held-out endpoint. Held-out titles and genres may enter
likelihood prompts, but held-out ratings never enter any model prompt. User IDs never
enter model prompts.

## Models And Belief Transition

- Non-thinking `google/gemma-4-26b-a4b-it` generates six initial semantic preference
  profiles from the four observed ratings.
- After a realized candidate rating, Gemma generates six replacement profiles from
  the full updated history. Each row contains a new description and a nonempty
  `new_evidence_effect`.
- A refresh fails closed if any description exactly copies an initial description,
  if descriptions repeat, or if an evidence-effect field is empty.
- The branch support is the six replacements plus the two old profiles assigning the
  highest probability to the realized rating, with exact-text deduplication.
- Non-reasoning `openai/gpt-5.4-mini` returns five-way rating likelihoods conditioned
  only on a semantic profile and movie metadata. It receives no raw observed history.
- Beliefs are uniform over each support, following BED-LLM's
  sample-filter-uniform construction.

The dataset remains local under its research terms and is not redistributed. Persisted
derived results omit source ratings and held-out item lists.

OpenRouter project-ledger ceiling: `$70.38480269545715`. Per-run cap: `$0.75`;
projected formal reservation: `$0.30`; concurrency: `64`. The live credit endpoint and
project ledger must be checked before each paid stage, and the lower remainder used.

## Serving Gate

The two smoke-only users run one candidate branch and one exact refresh replay each:

- two initial-profile requests;
- two profile-only initial-likelihood requests;
- two profile-refresh requests;
- two profile-only branch-likelihood requests;
- two exact refresh-prompt replay requests.

Passage requires exactly 10 physical requests, zero reasoning tokens, valid schemas,
two complete branches, two complete replay sets, and no copied initial description.
Manual audit must confirm the likelihood prompts contain no observed-history payload.
Failure closes this exact v2 interface before formal evaluation.

## Formal Mechanism Gate

The disjoint 12-user formal run has exactly 120 physical requests:

- 12 initial-profile requests;
- 12 profile-only initial-likelihood requests;
- 48 realized profile-refresh requests;
- 48 profile-only branch-likelihood requests.

For each user, initial and branch quality are mean negative log likelihood in nats on
the same eight held-out recorded ratings. One-step EIG is computed on the initial
profile support for each of the four candidate movies. Lower held-out NLL is better.

All conditions must pass:

1. all 12 users and 48 branches complete with exactly 120 requests and zero reasoning
   tokens;
2. every refreshed profile set satisfies the replacement schema;
3. mean oracle best-branch held-out NLL improvement is at least `0.05`;
4. at least 6/12 users have oracle improvement at least `0.05`;
5. at least 6/12 users have branch NLL spread at least `0.10`;
6. mean held-out NLL regret of the immediate-EIG choice is at least `0.03`;
7. at least 4/12 users have immediate-EIG regret at least `0.05`;
8. semantic profiles measurably affect rating likelihoods: mean maximum candidate EIG
   is at least `0.02` nats and at least 8/12 users have maximum candidate EIG at least
   `0.02`.

Passage establishes a path-dependent semantic-belief opportunity and authorizes only
a fresh target-blind scorer/ranking-fidelity gate. Failure closes this exact apparatus
before policy or depth evaluation. No threshold tuning, post-hoc user subset, alternate
held-out split, likelihood-history restoration, or profile-copy relaxation is allowed
after responses.
