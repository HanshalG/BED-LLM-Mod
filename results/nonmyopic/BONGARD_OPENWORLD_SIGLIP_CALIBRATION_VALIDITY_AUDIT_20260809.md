# Bongard OpenWorld SigLIP Calibration-Validity Audit

Date: 2026-08-09 (Europe/London), after freezing the SigLIP plan bank and
before any Bongard mechanics, candidate, development, confirmation, or endpoint
outcome was opened.

Status: **complete zero-outcome diagnostic; zero model calls and zero cost**.

## Question

The final SigLIP prototype model selected a signed temperature of `-2` from
mechanics plus development initial-history leave-one-out labels. Is that
inversion a development artifact, or does it generalize to the untouched
confirmation tasks' four already-observed initial examples?

This audit cannot answer policy efficacy. It reads only `initial_history`,
which every policy observes before choosing a query. It does not access
candidate labels, endpoint labels, generated Luna beliefs, or realized policy
paths, and it cannot authorize or stop paid work.

The initial partition point estimates were inspected once before the audit
script and bootstrap summaries were written. Therefore all intervals below are
descriptive held-out calibration evidence, not prospectively gated hypothesis
tests. The frozen scale and plans were not changed after this audit.

## Method

For each task and each of its four initial examples, hold out that example,
construct positive and negative prototypes from the other three, and record
the signed cosine-difference score. Compare three fixed temperatures:

- registered signed scale `-2`;
- no-information scale `0`;
- banked positive-grid diagnostic scale `0.25`.

Report pooled initial-label log loss, Brier score, raw-score AUC, and the AUC
after applying the negative temperature. For scale `-2` minus scale `0`, report
a fixed 20,000-draw task bootstrap interval. Confirmation contains 96 tasks and
384 initial-label predictions and was not used to choose the temperature.

## Results

| Partition | Tasks | Raw score AUC | Frozen probability AUC | Log loss `-2` | Log loss `0` | Log loss `.25` |
|---|---:|---:|---:|---:|---:|---:|
| Mechanics | 4 | 0.12500 | 0.87500 | 0.641637 | 0.693147 | 0.700445 |
| Development | 64 | 0.44867 | 0.55133 | 0.688412 | 0.693147 | 0.694728 |
| Confirmation | 96 | 0.34465 | 0.65535 | 0.666651 | 0.693147 | 0.697576 |

On untouched confirmation initial histories:

- paired log-loss difference, frozen `-2` minus zero:
  `-0.026496`, 95% bootstrap CI `[-0.041460, -0.011673]`;
- paired Brier difference, frozen `-2` minus zero:
  `-0.013034`, 95% bootstrap CI `[-0.020240, -0.005640]`;
- mean raw score is `0.002416` for positive examples and `0.071280` for
  negative examples.

Development's task-paired intervals cross zero despite being part of the scale
selection set: log-loss difference `-0.004736`, CI
`[-0.024072, 0.014955]`, and Brier difference `-0.002334`, CI
`[-0.011609, 0.007313]`. The four-task mechanics values are directionally
strong but too small for efficacy interpretation.

## Interpretation

The score inversion replicates on confirmation initial histories and the
frozen negative temperature improves both held-out proper scores over an
uninformative predictor. The SigLIP comparator is therefore not merely a
positive-grid failure or constant-half predictor. Its unusual orientation is a
stable property of this four-example prototype construction on the frozen
task bank.

This strengthens only the credibility of the fixed semantic baseline. It does
not reveal how well SigLIP predicts candidate or endpoint labels, whether its
selected queries are useful, whether depth two beats myopic, or whether Luna
wins. All such outcomes remain sealed and the registered plans remain
unchanged.

## Bindings

- audit JSON SHA-256:
  `03d45b4b50ee681fa6b906d7e5d558f7aa8f8c72110d8a231ed97cfd74ca31ae`;
- audit implementation SHA-256:
  `4bfccf41349a7386273240d3e902d4a8983ade975b0827710f067c27594937b7`;
- audit tests SHA-256:
  `afdda94c4b0ec54f8761e2ffb819a4aa857425a0a8a15f45f0c2098bd9d41b54`;
- frozen SigLIP protocol SHA-256:
  `e4b126b9b95f18d62a8968959c67b1b5e2c9e8b01c7794b0dc23e7b8a80f5b86`;
- frozen final plans SHA-256:
  `a41cc3b01d18fa9008f67d6f60a3f113b8f3194b73e932b4e2c3cfc83215f587`.

Verification: the complete Bongard OpenWorld suite passes `205/205` tests.

No paid protocol, plan, calibration, action, gate, or reporting rule changes.
