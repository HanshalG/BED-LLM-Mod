# Tau2 MMS Gemma 26B Budgeted-Thinking Calibration Protocol

Date frozen: 2026-08-13

## Motivation And Closed Predecessor

The prior Gemma 26B thinking interface is terminally closed. It completed and
banked all twelve requests, but three requests consumed the full 8,704-token
output allowance in reasoning and returned empty final answers. Official
calibration observations were never loaded, so that result is a serving null,
not semantic evidence.

This successor changes only the prospective serving contract. It uses an
explicit reasoning-token budget smaller than the total completion budget so a
strict JSON answer has reserved headroom. It does not alter the documented
prompts, response schemas, likelihood construction, planner, semantic gates,
or any closed cohort.

## Fresh Cohort

Use four untouched `mms_abroad` reserve episodes at family positions 21--24
and the final two untouched `mms_home` reserve episodes at positions 21--22.
The 4:2 balance is fixed by source availability before any selected official
observation is opened. The six episodes are hash-bound in the public manifest
and disjoint from every previous Tau2 semantic cohort.

Task IDs, source fault names, official observations, repairs, rewards, and
task-success endpoints are not serialized publicly or shown to the model.

## Model And Interface

Use exact `google/gemma-4-26b-a4b-it` through OpenRouter. The model receives
the exact documented-split root and native prompts and strict JSON schemas:

- one eight-action root-equivalence request per episode;
- one native messaging-permission partition request per episode;
- the public `check_app_permissions("messaging")` granted-name contract;
- no selected official observation or endpoint label.

Each request sets `reasoning.max_tokens=4096`, `reasoning.exclude=false`, and
total `max_tokens=4608`. The 512-token difference is reserved for the final
structured answer. Forced-final continuation is disabled. A length exit,
empty answer, invalid schema, noncanonical partition, or actual reasoning
count above 4,096 fails closed and is never repaired.

OpenRouter's exact Gemma model page advertises configurable maximum reasoning
tokens, while its general reasoning documentation requires total output tokens
to exceed the reasoning budget. The run still treats provider compliance as a
measured serving gate rather than an assumption.

## Frozen Serving Envelope

- Root seeds `202608132100`--`2105`; native seeds
  `202608132200`--`2205`.
- Temperature zero; concurrency two.
- Exactly twelve accepted responses and exactly twelve HTTP attempts.
- Zero transport, provider, format, schema, or semantic retries.
- Positive total reasoning usage; every request reports at most 4,096
  reasoning tokens; all twelve finish normally with nonempty strict JSON.
- Zero forced exits and no forced-final continuation.
- `provider.require_parameters=false`; the producer banks every actual HTTP
  payload hash and an independent verifier reconstructs it byte-for-byte.
- Complete response bank before official observations.
- Per-request reservation `$0.006`; full stage cap `$0.08`.
- Hard account-wide Europe/London daily cap `$5.00`, including unrelated use.

No paid execution is authorized by this protocol alone. A fresh source audit,
producer, producer-independent verifier, adversarial tests, dated account-wide
budget wrapper, immutable execution binding, pushed commit, and authenticated
same-day preflight are required first.

## Unchanged Semantic Gates

Require at least `47/48` exact root decisions and root Brier at most `0.08`;
all six native partitions and all 36 pair relations exact with native
partition Brier at most `0.08`; all 24 native answers top-ranked with mean
truth posterior at least `0.65` and posterior Brier at most `0.18`;
equivalence mean/max TV at most `0.03`/`0.10`; greedy avoids and depth two
selects `installed_apps` in all six episodes; every horizon gain at least
`0.50` nats; semantic/source depth-two Spearman at least `0.90`.

A pass authorizes only a separately frozen paired development protocol with
compute-matched myopic and random controls and sealed common-random-number task
endpoints. Any serving or semantic failure closes this model/interface/cohort.
