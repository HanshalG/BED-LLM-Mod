# Tau2 Account Fixed-Support V2 Preregistration

Date frozen: 2026-07-24

## Question

Can a non-thinking LLM build a sufficiently coherent semantic forward model to
rank a real zero-information prerequisite above myopic diagnostics?

V1 tested open-world support generation and failed before rollouts. V2 is a
standard fixed-support BED problem: the six hypotheses and uniform prior are
visible to the policy. The official Tau2 tool outputs remain hidden until
realized scoring.

## Exact Opportunity

The support is the official same-issue 2-by-3 factorial:

- data allowance available or exhausted;
- crossed with account roaming enabled/device roaming off, account roaming
  disabled/device roaming on, or both roaming settings off.

Airplane mode is on in every world. Exact official values are:

- immediate `customer_lookup`: 0 nats;
- `customer_lookup` then `line_details`: 1.329661 nats;
- best legal direct two-step root: 0.636514 nats;
- setup advantage: 0.693147 nats.

## LLM Interface

Gemma 4 26B A4B runs non-thinking at temperature zero. For each of six real
Tau2 root diagnostics, it:

1. predicts an outcome category for every support hypothesis;
2. chooses one legal branch-specific follow-up; and
3. predicts the follow-up outcome partition.

The executable computes information from those partitions. Gemma receives no
entropy, EIG, utility card, official task state, initialization action, or
simulator response. Its semantic action model is therefore load-bearing.

The prompt variants are robustness replicates over one physical support family,
not independent environment samples.

## Stages

### Serving Smoke

- Two frozen prompt variants.
- Exactly 12 requests and zero reasoning tokens.
- Hard cap $0.50.
- Requires complete finite trees, fixed support coverage 6/6, and both
  `customer_lookup` trees choosing `line_details`.

### Formal

- Twelve untouched prompt variants.
- Exactly 72 requests.
- Hard cap $2.00.
- Run only if smoke passes unchanged.

Frozen conjunction:

1. exact completion/accounting and zero reasoning;
2. `customer_lookup` uses `line_details` on at least 10/12 variants;
3. predicted depth-two versus exact depth-two Spearman at least 0.25;
4. depth two selects `customer_lookup` on at least 8/12;
5. depth one never selects `customer_lookup`;
6. mean realized top-one regret improves by at least 0.50 nats; and
7. depth two beats depth one on at least 8/12.

Any substantive invalid tree closes V2 without repair or resampling. Passing
authorizes a fresh paired sequential trajectory confirmation.
