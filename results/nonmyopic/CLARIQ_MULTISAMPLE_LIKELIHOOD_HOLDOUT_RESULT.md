# ClariQ Multisample Likelihood Holdout Result

## Status

Failed closed at the preregistered exact parsing gate. Scientific efficacy was
not measured because endpoint values were never loaded.

## Execution

- Run:
  `clariq-multisample-likelihood-holdout-20260725T164625Z`
- Model: `openai/gpt-5.4`, non-reasoning.
- Logical requests: `755`.
- Physical requests / HTTP attempts: `755 / 755`.
- Retries / reasoning tokens / forced exits: `0 / 0 / 0`.
- Prompt / completion tokens: `157225 / 4685`.
- Adapter cost: `$0.4633375`.
- Public failure SHA-256:
  `8b9eda95dfc6f21a5a506678cf3f668a5f99650c393727519b62f3cbfd769561`.
- Private raw SHA-256:
  `2fe911908d0b237a8e863f21eb787e46615a524003864ddb20ac6c708206dfb4`.

## Parsing Failure

Of the 755 returned strings, 743 satisfied the frozen exact Y/N/U grammar and
12 did not:

- 11 responses contained only valid Y/N/U characters but omitted one required
  facet label.
- Ten of those 11 occurred on six-facet topic `10`; repeated omissions
  occurred for several roots.
- One omission occurred on five-facet topic `115`.
- One four-facet response for topic `14` contained four labels separated by
  newlines rather than four contiguous characters.

The full response batch was checkpointed, but parsing stopped before maps,
likelihoods, policy scores, or roots were frozen. The raw checkpoint confirms
that holdout endpoints were not loaded.

## Decision

The exact fixed-support holdout interface is closed without response repair,
compaction, imputation, reissue, or rerun. Because no endpoint was accessed,
this is a serving/measurement failure rather than evidence for or against the
non-myopic estimator.

The positive three-topic V2 development result remains development evidence
only. It is not promoted to a confirmed result, and the preregistered
answer-conditioned support-regeneration stage is not authorized.

No OatML resources were used.
