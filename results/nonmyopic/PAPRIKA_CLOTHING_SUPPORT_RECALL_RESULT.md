# PAPRIKA Clothing Support-Recall Result

## Decision

The serving smoke passed, but the preregistered activity gate failed. The
20-case confirmation was not run. This exact clothing interface is closed.

## Serving smoke

The two-case smoke completed the exact 52 physical requests with zero reasoning
tokens, retries, forced exits, or malformed responses. It cost `$0.06346450`.
All supports and target-blind ranker scores parsed. One case had
candidate-dependent realized support coverage; the second had an omitted target
that no candidate recovered. The smoke therefore passed mechanics only, as
registered.

Artifact SHA-256:
`3fe46f97b93f1749bca130e773e3d17888c181d6ff6b832908f8dffe3136725e`.

## Activity gate

The eight fresh activity cases completed the exact 200 physical requests with
zero reasoning tokens, retries, forced exits, or malformed responses. It cost
`$0.24712375`. The frozen opportunity requirements all failed:

| Metric | Required | Observed |
|---|---:|---:|
| Current-support target omissions | >=3/8 | 2/8 |
| Omitted targets recovered by some realized branch | >=2 | 0 |
| Cases with candidate-dependent realized coverage | >=3/8 | 2/8 |

Six current supports already contained the target. More importantly, neither
omitted target, Bucket Hat or Harem Pants, appeared in any of the three
realized branch supports. Most cases therefore supplied no action-quality
variation for a target-blind ranker to exploit.

Artifact SHA-256:
`96fb5672c1d129a8630fd43ad2a7395860bab61939ec34452b691442a139f91b`.

## Environment audit

A zero-call inspection found that GPT-5.4 Mini's non-reasoning semantic
environment was not reliable enough for these fine-grained clothing questions.
Unambiguous examples include:

- T-shirt: full front opening with buttons or zipper -> `Yes`;
- Hoodie: typically has long sleeves -> `No`;
- Ankle Socks: typically covers the toes -> `No`; and
- Harem Pants: covers each leg separately -> `No`.

These incorrect answers send support regeneration down contradictory branches.
For example, the T-shirt `Yes` branch regenerated jackets and coats, while the
Hoodie `No` branch regenerated sleeveless upper-body items. This does not alter
the registered failure: opportunity was insufficient under the deployed
environment either way. It does mean the observed null cannot cleanly isolate
the target-blind ranker mechanism from environment-label error.

## Interpretation

The test sharpened the bottleneck. A useful LLM-native non-myopic policy needs
both:

1. action-conditioned regeneration that sometimes recovers omitted truth; and
2. a reliable semantic response function defining which branch is realized.

This interface had neither at adequate prevalence. K=3 multisampling made the
current support broad enough to contain common targets, while branch
regeneration often dropped even initially present targets. The weak Mini
environment compounded that instability. Running the sealed ranker confirmation
would spend money on an endpoint with almost no available signal, so it remains
untouched.

A distinct future test may use a fresh PAPRIKA category and GPT-5.4
non-reasoning as the semantic environment, with an answer-consistency serving
gate before any support-opportunity measurement. It must not reuse these 30
clothing targets as a repaired confirmation.

## Budget

Combined smoke and activity cost was `$0.31058825`. The project ledger is
`$70.32812466` spent with `$35.05667803` headroom under its frozen ceiling.
The live provider balance after the activity run was `$60.28002279`, leaving
`$35.28002279` above the protected `$25` Monday reserve. OatML was not used.
