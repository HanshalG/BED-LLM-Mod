# Cleveland Heart Workup Depth Qualification Preregistration

Registered 2026-07-23 after an exploratory zero-call Statlog Heart screen and before
any Cleveland policy endpoint was evaluated.

## Purpose

Qualify a compact non-spatial semantic workflow on an independent cohort. A
zero-information clinical-workup action unlocks eight laboratory, ECG, exercise, and
imaging features. The question is whether endpoint-aligned depth two orders that
workup earlier than one-step information gain and improves disease-class uncertainty
over the full sequential trajectory.

## Frozen Transfer Design

- Exploratory source: all 270 complete UCI Statlog Heart rows.
- Qualification cohort: the separate processed Cleveland cohort from UCI Heart
  Disease; remove its six rows with missing values and use all 297 complete rows.
- Target: absence (`0`) versus presence (`1`--`4`) of heart disease.
- Uniform empirical prior over the 297 qualification rows.
- Initially available: age, sex, chest-pain type, resting blood pressure, and fasting
  blood sugar.
- `order:clinical-workup`: consumes one round, produces no observation, and unlocks
  serum cholesterol, resting ECG, maximum heart rate, exercise angina, ST depression,
  ST slope, major-vessel count, and thal category.
- Feature observations are deterministic and cannot be repeated.
- Continuous values use three fixed bins with Statlog-derived cut points:
  age `(51, 59)`, resting BP `(120, 138)`, cholesterol `(226, 267)`, maximum heart
  rate `(142.6667, 162)`, and ST depression `(0.1, 1.4)`.
- Horizon: eight rounds. Exact d1 and d2 minimize expected cumulative post-action
  binary class entropy over their local horizon.

## Frozen Qualification

- Fresh row permutation and bootstrap seed `24133`.
- 10,000 paired bootstrap replicates over all 297 Cleveland rows.
- Primary: mean post-action class entropy over eight rounds.
- Corroborating: mean log posterior probability of the true binary class.
- Secondary: final entropy, final truth probability, MAP accuracy, action traces,
  workup timing, and exact scorer units.
- No LLM, OpenRouter, or GPU call is permitted.

The task qualifies only if all legality/pairing/no-repeat/zero-information mechanics
pass, d2 orders the workup earlier than d1 on average, and both paired 95% bootstrap
lower bounds for d2 minus d1 are strictly positive on entropy AUC and truth-log AUC.
A pass authorizes only a separately preregistered compact non-thinking 26B serving and
proposal gate.

## Exploratory Disclosure

The Statlog screen compared five initial-feature splits and 3/4-bin discretizations.
The frozen five-feature, three-bin design had the strongest entropy-AUC gap:
`+0.06754` with exploratory paired interval `[+0.05504,+0.07942]`; truth-log gain was
also `+0.06754` (`[+0.04551,+0.09064]`). D2 queried chest pain then ordered the
workup, while d1 delayed workup until round six. These Statlog endpoints select the
design but cannot qualify the independent Cleveland cohort.

Sources: UCI Statlog Heart, https://doi.org/10.24432/C57303; UCI Heart Disease,
https://doi.org/10.24432/C52P4X. Both repository pages report CC BY 4.0 licensing.
