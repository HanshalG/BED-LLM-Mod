# Detective CA-BED Aligned-Likelihood Diagnostic Preregistration

Date frozen: 2026-07-24

## Question

Does depth-two truth-ranking recover when its likelihood table is built from
the exact same hidden-role response function used by the environment?

The failed textual-likelihood gate showed a clean split: depth two ranked
realized entropy reduction well (`rho=.750`) but truth gain backwards
(`rho=-.208`). The direct probability estimator saw only public context, while
the answerer saw the targeted suspect's private story and murderer/innocent
role. This diagnostic removes only that simulator mismatch.

## Frozen Source

- Source gate:
  `detective-cabed-gemma-concise-ranking-20260724T192405Z/GATE.json`
- Source SHA-256:
  `561e80787e88bf19c8c340a62fe61436aeccea91199616ab958fef65bbb9aca9`
- Same 12 inspected development cases and all 252 generated root/follow-up
  question cells.
- Exact-string caching leaves 241 unique within-case questions.
- No new question generation, fresh case, textual likelihood, or old hidden
  answer is used.

## Aligned Likelihood

For each unique case/question pair, call the Gemma 4 26B A4B hidden-role
answerer at temperature zero exactly twice:

1. target suspect is the murderer;
2. target suspect is innocent.

Both prompts include the same public case context, target's private story, and
role instruction as deployment. For hypothesis `h`, use the murderer response
if `h` is the targeted suspect and the innocent response otherwise. Convert
`Yes` to `.85` and `No` to `.15`; this is exactly CA-BED confidence smoothing
`0.7 * hard_label + 0.3 * 0.5`.

The realized answer under the true murderer is the corresponding cached row
response, so likelihood construction and deployment are the same deterministic
semantic function. The LLM remains load-bearing: it maps arbitrary
question/story/role text to an answer, while external code performs only exact
Bayes and tree scoring.

Total: `241 * 2 = 482` requests. No content retry, parser repair, replacement,
or old-response reuse. Projected cost `$0.10`, hard cap `$0.50`.

## Development Gates

All gates are conjunctive:

1. Exactly 482 valid answer rows, zero reasoning tokens and forced exits.
2. At least 25% of unique questions produce different murderer and innocent
   responses; otherwise the aligned simulator is too uninformative.
3. At least 9/12 cases have rankable depth-two scores and truth gains.
4. Depth two changes the d1-selected root on at least 4/12 cases.
5. Mean d2-score versus truth-gain Spearman is at least `.20`.
6. Mean d2 Spearman exceeds d1 by at least `.10`.
7. Mean d2-selected truth gain exceeds d1 by more than zero with at least six
   strict wins.
8. Mean d2-selected truth gain exceeds the frozen seeded-random root by more
   than zero.

This inspected diagnostic is not a policy claim. Passing authorizes a newly
preregistered ranking gate on untouched cases with fresh questions and the
aligned likelihood. Failure closes Detective Cases: the released task then has
either no usable counterfactual response model or no depth opportunity after
alignment.
