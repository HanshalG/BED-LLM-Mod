# PAPRIKA Instrument Support-Recall Gate

## Purpose

Run a distinct, semantically cleaner test of the first LLM-native non-myopic
link: can a target-blind model identify a question whose truthful answer induces
a better next generated hypothesis support than immediate EIG? This is not a
repair or continuation of the failed clothing gate. It uses a fresh category,
fresh targets, a stronger response model, exact-name coverage, and a
pre-efficacy semantic consistency requirement.

## Frozen protocol

- Released PAPRIKA Twenty Questions eval data SHA-256:
  `d9b7616d1316886aa1aee84ddc4f126715be4983c50d5160c5c7d223b7e0f16a`.
- Selection seed `24332`, using `random.Random(seed).shuffle` over the 29
  instrument indices in file order.
- Smoke: `212, 204`.
- Activity: `198, 206, 213, 11, 211, 14, 203`.
- Untouched confirmation:
  `13, 196, 193, 210, 200, 15, 195, 205, 194, 207, 12, 192, 209, 17,
  215, 10, 202, 197, 16, 208`.
- GPT-5.4 non-reasoning is both open-world generator/ranker and semantic
  response model. Separate adapters and prompts are used.
- Three fixed prefix questions distinguish string, blown-air, and struck
  instruments.
- Current and hypothetical branch supports are K=3 normalized unions of
  independently generated 12-item supports at temperature `.6`.
- GPT-5.4 proposes three target-blind semantic candidate questions.
- Every prefix, current-support, and target candidate label is issued twice:
  once as Yes booleans in original order and once as No booleans with both axes
  reversed. After reversing and complementing, exact disagreement fails closed.
- Immediate EIG uses the checked current-support labels.
- Realized truth coverage is exact normalized target-name inclusion, not an LLM
  equivalence judgment. This conservatively excludes aliases.
- The ranker sees history, current support, answer probabilities, and branch
  supports, but never target identity, target coverage, or the evaluation list.
- Expected branch-support size is the non-LLM target-blind baseline.
- All calls have provider reasoning disabled.

## Stages

The exact 58-call two-target smoke tests complete serving, all complementary
label checks, target-blind ranker parsing, exact request count, and zero
reasoning. Any failure closes the route.

The exact 196-call seven-target activity stage does not call the ranker. It
passes only with:

- at least two current target omissions;
- at least two omitted targets recovered by some realized branch; and
- at least three cases with candidate-dependent realized coverage.

Failure closes the route without confirmation.

The exact 580-call 20-target confirmation is authorized only if both prior
stages pass. Its primary endpoint is paired realized next-support target
coverage for ranker selection minus immediate-EIG selection. Success requires:

- exact completion, request count, zero reasoning, and all label checks;
- different ranker/EIG choices on at least five cases;
- mean paired gain at least `.10`;
- a strictly positive 90% lower bound from 10,000 paired bootstrap resamples
  using seed `24333`;
- more ranker wins than losses;
- ranker mean coverage not below expected-support-size selection; and
- ranker-score candidate Spearman positive and greater than both baselines.

No malformed response, target, question, or case may be replaced. Smoke values
cannot alter the activity protocol; activity values can only authorize or stop
confirmation.

## Budget

Before implementation, the project ledger had `$35.05667803` headroom and the
live account had `$35.28002279` above the protected `$25` Monday reserve.
OatML remains paused. Projected stage costs are `$0.20/$0.75/$2.00`, with hard
caps `$0.75/$2/$4`. Check the live and ledger balances before every paid stage.
