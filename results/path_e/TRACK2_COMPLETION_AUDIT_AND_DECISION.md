# Track 2 Completion Audit and Decision

Date: 2026-07-14

Status: **STOP-and-discuss. No iCRAFT implementation, model call, or policy run is
authorized by this memo.**

## Decision In One Line

Choose exactly one:

- `AUTHORIZE ICRAFT GATES`: preregister and implement only the bounded iCRAFT-MD
  diagnosis-profile validation ladder. Stop at its first failed gate. A policy or depth
  run remains unauthorized until every upstream gate passes.
- `CLOSE EXTERNAL CLAIM`: make the current validation-first paper the final empirical
  package and make no further model calls for this project.

## Completion Audit Against The Project Goal

| Requirement | Authoritative evidence | Status | Consequence |
|---|---|---|---|
| External Paprika adapter and environment smoke | `STEP0A_OPENROUTER_SMOKE.md`; later endpoint audits | Partial then contradicted at scale | The small adapter gate passed, but the held-out simulator contradicted a private exact remedy. Headline outcomes are diagnostic only. |
| Exact non-myopic recursion test | Mastermind tests and constrained-location oracle package | Passed | The depth-two recursion can exploit a real delayed-information gap under an exact model. |
| External one-step BED transfer | Paprika and MediQ policy evidence | Not established | Paprika's endpoint is invalid. MediQ stopped before a calibrated policy comparison. |
| External depth-one versus depth-two claim | Paprika full2 and MediQ | Not established | Paprika full2 was worse and endpoint-invalid. No MediQ depth run was authorized. |
| Selective-lookahead efficiency claim | Paprika arbitration artifacts | Not established | Triggering and cost were measured, but no valid endpoint benefit exists to preserve. |
| MediQ environment interface | `mediq_step0/FINAL_REPORT.json` and manual review | Passed | Canonical atomic actions, grounding, relevance, deduplication, and mapping work. |
| MediQ probabilistic model | `FACTORED_RECORD_REPORT.json`; data-estimation bank report | Failed | Available mean truth-log gain was -0.118 nats; the replacement bank had only 8/30 answerable turns versus the frozen minimum of 15. |
| One-step semantic BED evidence | `ANIMALS_REANALYSIS.md` | Passed, with scope limits | Depth-one EIG beats both naive baselines over 40 paired trials. This is not an external non-myopic result. |
| Incremental animals horizon evidence | `ANIMALS_REANALYSIS.md` | Failed | The only paired d2-d1 block is -0.040 AUC and paired d3-d2 is -0.126. |
| Traceability and active-run cleanup | `EXPERIMENTS.md`; ledger validator | Passed | All 170 ledger rows are populated, complete-row artifacts exist, and no run is active. |
| 4-6 page honest paper | `paper/main.tex`; compiled PDF; paper validator | Passed for current evidence | The five-page paper makes a boundary/validation claim, not the originally desired positive external non-myopic claim. |

The original positive external non-myopic objective is therefore **not complete**. The
current package is complete only for the cross-environment boundary result. It must not
be relabeled as proof that non-myopic LLM BED improves an external benchmark.

## Why iMEDQA Is Closed

The final environment adapter is not the blocker. The blocker is the joint model:

1. The A-D target can denote a drug, treatment priority, mechanism, or diagnosis, so it
   is not generally a sufficient patient state for predicting findings.
2. Making missingness label-neutral repaired the algebra but available findings still
   reduced mean probability on the true option.
3. The policy-independent bank produced valid questions and faithful replies, but 22 of
   30 requested findings were absent from the static records.
4. The frozen preregistration says failure or an inadequate bank ends model changes and
   requires discussion. Reusing those cases, tuning thresholds, adding thinking, or
   increasing rollout count would violate that rule.

No further iMEDQA scorer, Claim-1, or depth run is admissible.

## What iCRAFT Fixes, And What It Does Not

The local hash-pinned official release already supports `mediq_dataset: icraft_md`.
A fresh read of commit `faa2ce62fef0423e35af4c31d7537aad973173eb` verifies:

- 140 raw and 140 usable cases;
- exactly four answer options in every case;
- the same question in all 140 cases: the most likely diagnosis;
- mean 14.82 atomic facts, median 14, minimum 9, maximum 28;
- no dataset-loader or hash-verification change is needed to select this split.

This removes the worst target/state mismatch: a diagnosis is a plausible causal label
for a patient. It does **not** make the diagnosis alone sufficient. Several patients
with the same diagnosis can answer the same history question differently. The required
model therefore remains

```text
p(z | initial evidence) p(response | z, query, history),
theta = diagnosis(z),
```

where `z` is a concrete patient profile and EIG targets the grouped diagnosis.

The profile model is not implemented. The current code has only direct A-D likelihood,
factored-record, and data-estimation modes. The iMEDQA analyzers also pin their expected
dataset and case shape. A valid iCRAFT path needs new profile generation, filtering,
grouped priors, profile-conditioned likelihoods, matched branch/deployment updates,
fresh analyzers, tests, and a preregistered case partition.

More importantly, iCRAFT has no obvious native delayed-information gate. FactSelect is
a static record channel and the questioner may ask any atomic predicate immediately.
The profile construction can improve model validity, but it cannot create the project's
required structural greedy gap. The first scientific risk after model validity is that
an exact profile-world depth-two oracle will not beat exact greedy EIG. If so, iCRAFT is
a one-step BED environment and the non-myopic claim closes without a policy run.

## Option A: Authorize A Gate-Only iCRAFT Preregistration

This is the only remaining path inside the allowed external environment. Authorization
should cover the following sequence and nothing beyond it:

1. Freeze a seed-shuffled, non-overlapping development, calibration, structural-gap,
   pilot, and untouched-headline case partition before any profile-model result is
   viewed.
2. Specify the profile support and grouped diagnosis prior. Profile count and repair
   rules must not make diagnosis mass depend accidentally on how verbose or easy an
   option is to sample.
3. Keep the scaffold non-thinking. The thinking model remains the naive baseline.
4. Validate profile consistency with initial evidence and diagnosis on held-out cases,
   including mandatory manual review.
5. Validate the grouped prior against uniform using held-out log loss and Brier score,
   while rejecting high-confidence wrong collapse.
6. Validate available-response likelihoods by positive mean true-diagnosis log gain;
   keep missingness exactly diagnosis-neutral unless the official interface makes it
   semantically informative.
7. Prove branch equivalence: the same canonical history must produce the same support,
   probabilities, and stopping state in simulation and deployment.
8. Run a no-policy structural-gap gate. Exact or high-fidelity depth two must beat exact
   greedy under the native question and budget rules. Failure closes the horizon claim.
9. Run ranking fidelity with shared candidates and common random numbers. Estimated
   incremental depth value must rank oracle incremental value and must not improve
   entropy by concentrating on the wrong diagnosis.
10. Only after all nine gates pass, return for a second authorization before a paired
    policy or depth pilot.

The first preregistration must freeze numeric thresholds, sample sizes, case partitions,
and cost caps. A cost micro-smoke should precede any paid calibration bank. The current
OpenRouter ledger is $16.81168851 of $40, leaving $23.18831149, but remaining budget is
not evidence that the path is valid. A recommended initial gate-only cap is $0.50; any
larger request should follow an observed micro-smoke projection rather than speculation.

### Expected Scientific Outcomes

- Profile/prior/likelihood failure: confirms that LLM-derived world models are still the
  limiting factor, with no policy run.
- Structural-gap failure: iCRAFT can remain a one-step BED transfer environment, but it
  cannot carry the non-myopic claim.
- Ranking-fidelity failure: exact lookahead may have value, but this LLM estimator cannot
  recover it reliably.
- All gates pass: a paired d1/d2 pilot becomes scientifically justified. It does not
  become automatically positive.

## Option B: Close The External Non-Myopic Claim

No further model calls are needed. The final package is the current five-page
validation-first paper plus its reproducible diagnostics. Its claims are:

- exact lookahead succeeds under a coherent simulator and an explicit delayed trap;
- one-step semantic EIG improves animals 20 Questions;
- natural-language planning depth is non-monotonic when rollout ranking is unreliable;
- Paprika exposes task-objective and endpoint failures;
- MediQ exposes latent-sufficiency, likelihood-calibration, and record-sparsity failures;
- deeper search is downstream of model validity and a measurable greedy gap.

This is honest and complete, but it is not the positive external-benchmark method paper
described in the original goal.

## Recommendation

If retaining a chance at the external non-myopic result is strategically important,
authorize **only** Option A's gate-only preregistration. It gives the idea one clean,
bounded attempt and stops before another expensive depth sweep. My confidence is higher
that profiles can repair likelihood semantics than that iCRAFT contains a material
native greedy gap.

If the priority is a finished defensible workshop submission, choose Option B. The
current paper is already validated and does not depend on another uncertain model.
