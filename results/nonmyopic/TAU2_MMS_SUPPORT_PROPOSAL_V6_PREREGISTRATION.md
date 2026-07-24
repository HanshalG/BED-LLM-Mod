# Tau2 MMS Semantic-Support V6 Preregistration

Date frozen: 2026-07-24

## Question

Can an LLM generate the remaining semantic fault support from unstructured tool
history, while the official Tau2 simulator supplies all likelihoods and an exact
planner supplies non-myopic lookahead?

This is a distinct complement to failed V5. The LLM no longer predicts tool
outputs or branch actions. Its only scientific role is BED-LLM-style targeted
hypothesis proposal. Deterministic code filters proposals against already
observed history, maps them to official worlds, and performs exact Bayesian
planning.

## Official Structural Opportunity

Tau2 contains 1,984 MMS worlds. Consider the official state after seven ordinary
diagnostics have been executed:

- status bar;
- network status;
- network mode;
- APN settings;
- Wi-Fi Calling;
- speed test; and
- an MMS probe.

For the common observed history used here, exact compatibility leaves three
official worlds:

1. messaging app SMS permission faulty;
2. messaging app storage permission faulty; and
3. both permissions faulty.

All remaining direct actions have zero immediate and zero two-step EIG.
`installed_apps` also has zero immediate EIG, but it unlocks
`messaging_permissions`, which identifies all three worlds for `ln(3)=1.098612`
nats. The complete direct history, exact values, and three-world support were
verified before any V6 response.

## Semantic Prior Interface

- Full `openai/gpt-5.4` through OpenRouter, no reasoning, temperature zero.
- One request per ticket variant.
- The model sees the literal seven-action observed history.
- It generates six distinct free-text hypotheses with structured status fields
  for network mode, Wi-Fi Calling, MMSC/APN, SMS permission, and storage
  permission.
- It receives no hidden world, future tool output, EIG, entropy, utility, or
  policy choice.
- Exact filtering removes hypotheses that contradict observed normal direct
  fields or fail to instantiate a concrete permission fault.
- Remaining rows expand only over official permission worlds compatible with
  their stated normal/faulty/unknown fields.
- The official simulator supplies every action likelihood and exact d1/d2 value.

This is LLM-Modulo support generation, not unaided LLM planning. The three-world
benchmark is enumerable for audit; the scientific role being tested is whether
the same semantic proposal interface can compress a 1,984-world catalog from
raw tool history.

## Fresh Splits

Seed `24322` freezes:

- two serving-smoke ticket variants;
- twelve disjoint formal variants; and
- twenty-four additional confirmation variants with a balanced eight-per-world
  hidden schedule.

No failed variant may be replaced. Raw responses remain private; parsed supports,
filter mappings, selected policies, official scores, endpoints, usage, and raw
hashes are public.

## Serving Smoke

Exactly two requests, projected `$0.05`, hard cap `$0.50`. Both must:

1. parse six distinct hypotheses;
2. retain at least three concrete permission hypotheses;
3. cover all three official permission worlds;
4. make exact depth one avoid `installed_apps`;
5. make exact depth two select `installed_apps`;
6. use `messaging_permissions` as its successor; and
7. use zero reasoning tokens with finite scores.

Failure closes V6 before formal use.

## Formal Gate

Exactly twelve requests, projected `$0.20`, hard cap `$1.00`. All gates are
conjunctive:

1. all variants parse, exact request count, zero reasoning, finite scores;
2. at least 10/12 retain three or more valid permission hypotheses;
3. at least 10/12 cover all three worlds;
4. depth one selects setup 0/12;
5. depth two selects setup at least 10/12; and
6. setup uses permissions at least 10/12.

Failure closes V6. Passage authorizes only the frozen confirmation.

## Conditional Paired Confirmation

Exactly twenty-four fresh requests, projected `$0.40`, hard cap `$1.50`.
Every model-generated support is shared by:

- exact depth-two planning;
- sequential greedy depth-one planning; and
- a seeded random root/follow-up control.

The selected two-action sequences are evaluated on the official balanced
three-world schedule. Endpoints are final entropy, trapezoidal entropy AUC, and
truth log posterior. Passage requires:

1. complete exact requests and zero reasoning;
2. full support coverage on at least 20/24 and truth coverage on at least 22/24;
3. depth two selects setup and beats greedy on at least 20/24;
4. depth two beats random on at least 16/24;
5. mean final-entropy gain over greedy at least `.80` nat with positive paired
   90% bootstrap lower bound;
6. mean gain over random at least `.40` nat with positive lower bound; and
7. truth-log-posterior lower bounds above zero versus both controls.

Passage would establish a positive external-tool LLM-Modulo result in which
semantic support proposal and non-myopic exact verification are both required.
It would not yet establish path-dependent support regeneration.
