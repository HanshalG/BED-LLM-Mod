# RegretBench DeepSeek Dynamic Depth-Two Confirmation Preregistration

Date frozen: 2026-08-07

## Purpose

This protocol prospectively freezes an independent confirmation of the
RegretBench LLM-native non-myopic policy result before any development model
response exists. It tests the same claim on the untouched 64-task confirmation
split: planning over answer-conditioned regeneration of the LLM's own semantic
belief state improves two-question identification of a hidden intent.

The confirmation can execute only after the exact development experiment in
`REGRETBENCH_DEEPSEEK_DYNAMIC_DEPTH2_POLICY_PREREGISTRATION.md` passes every
mechanics and scientific gate and an independent replay verifies that pass.
A development null, partial artifact, mechanics failure, hash mismatch, failed
replay, or unavailable predecessor authorizes zero confirmation calls.

## Frozen Data Boundary

- Source: RegretBench OpenDomainQA 1.0.0 at commit
  `b2978e1c2e31b7a7c4e1508ee3e1fa1cb98f4aa7`.
- Source protocol result SHA-256:
  `d7a10f15ecf6779520fbb20712c8d43059d8b03168fa904fe72e82ffe87578de`.
- Source protocol manifest SHA-256:
  `8a46b40395487aae0857d1a61d6f680beef579414c4580acf09d7e0b30b38e97`.
- Confirmation split: the 64 opaque task IDs already frozen in that manifest.
- Confirmation split SHA-256:
  `780a0e4e172251be2781729eeb4e591592b996dc9e1cd668b26240e3076660c9`.
- Development split SHA-256:
  `29d33b2fda0be7cc4eea6f4d9d4fe74fe200b5c580632dcb7ba844c3b825af69`.

The confirmation cohort is disjoint from mechanics and development. Its order
is exactly the order in the source protocol manifest. No task may be removed,
replaced, reordered, or selected using a model response or endpoint.

## Frozen Instrument

The confirmation uses the development instrument without scientific changes:

- model `deepseek/deepseek-v4-flash-0731` through OpenRouter;
- reasoning disabled and excluded for every call;
- temperature `0.7`, maximum output `2,200` tokens, concurrency `128`;
- exactly eight distinct semantic hypotheses and four distinct clarification
  questions per accepted support;
- two independent conditioned draws per root and simulated truth;
- paired same-seed history-blind draws, with each blind draw replayed under the
  same requested seed across all four roots;
- the conservative lexical truth matcher, exact generated-reply likelihood
  update, probability floor `1e-12`, and invalid-trajectory penalty;
- the official RegretBench semantic action mapper and exact environment reply;
- no retries, repair, coercion, continuation, replacement tasks, or scientific
  fallback.

The model sees only opaque task ID, ambiguous prompt, and the dialogue on the
particular simulated or realized path. Hidden intents, aliases, facets, slot
values, benchmark beliefs, mappings, policy labels, scores, and endpoints are
forbidden from every prompt.

## Policies And Endpoint

All policies share the exact generated tree and realized-history calls:

- `dynamic_depth2`: minimum conditioned expected terminal Brier;
- `history_blind_depth2`: identical scorer on prompt-only regenerated draws;
- `myopic_width`: maximum immediate EIG on the initial support;
- `fixed_depth2`: exact two-question lookahead on the initial fixed support;
- `random`: a root selected by the frozen local random seed.

The primary comparison remains `dynamic_depth2` versus the compute-matched
`myopic_width` control. The other controls retain their development roles.
There is no Luna or other optional baseline in confirmation.

Hidden truth is sampled locally only after every generated planning response
and policy selection is frozen. Each distinct selected root is executed once
and shared by all policies selecting it. The primary endpoint is the aligned
terminal truth mass on the support regenerated after question one and updated
by the generated likelihood partition for the selected question two. Brier is
`(1 - truth_mass)^2`; log loss uses floor `1e-12`. The separately regenerated
post-question-two support remains secondary and cannot replace or rescue the
primary endpoint.

An unsupported first action, unsupported second action, or second action on
the same official facet is an invalid trajectory and receives scored truth
mass zero, Brier one, and floor log loss. Raw masses remain descriptive only.

## Frozen Seeds And Counts

- initial support: `202608209000 + task_index`;
- matched/blind branch:
  `202608210000 + task_index*16 + hypothesis_index*2 + draw`, reused across
  all four roots;
- realized first-history support: `202608220000 + task_index`, reused across
  selected roots;
- realized final-history support: `202608230000 + task_index`, reused across
  selected roots;
- hidden truth: `202608240000 + task_index`;
- random root: `202608250000 + task_index`;
- paired bootstrap: `202608260000` with `20,000` replicates.

The planning block is exactly `8,256` accepted requests: 64 initial requests
plus 8,192 branch requests. Realized execution adds two requests per distinct
selected root and therefore at most `512` requests. Maximum confirmation size
is `8,768` requests. Every seed is disjoint from the development schedule.

## Mechanics Gates

All must pass:

1. Every source, split, preregistration, amendment, implementation, prompt,
   seed, privacy, predecessor, and result binding matches the frozen hashes.
2. Exact planning request and HTTP-attempt count `8,256`, primary total no
   greater than `8,768`, zero retries/provider retries/reasoning/forced exits.
3. Every initial, simulated-branch, and realized support satisfies the strict
   eight-hypothesis/four-question schema and uniqueness requirements.
4. Every task has at least two informative initial roots and at least 90% of
   simulated branch supports have an informative follow-up.
5. Every policy has at least 48 supported first actions, 40 supported second
   actions, 40 novel second actions, and 40 exact second replies represented
   by its generated likelihood partition.
6. Every one of the `64*8*2 = 1,024` history-blind CRN groups has four
   canonically identical parsed supports across roots.
7. Selections are frozen before hidden truth access, realized roots share the
   task-level first/final seeds, and all public artifacts pass the privacy
   boundary.
8. The independent result verifier exactly reconstructs selections, outcomes,
   endpoint metrics, uncertainty, mechanics, science, and status from private
   raw artifacts without importing the experiment producer.
9. Total confirmation spend is at most `$3.50` and the account-wide London-day
   spend is at most `$5.00`.

Any mechanics failure makes confirmation status `failed_closed` and forbids a
scientific claim, regardless of apparent endpoint values.

## Scientific Gates

The same nine conjunctive development gates are frozen for confirmation:

1. dynamic and myopic roots differ on at least `16/64` tasks;
2. dynamic and history-blind roots differ on at least `12/64` tasks;
3. dynamic and fixed roots differ on at least `12/64` tasks;
4. conditioned predicted Brier improves over the myopic-selected root by at
   least `0.01` on average;
5. dynamic minus myopic aligned Brier is at most `-0.02`, paired bootstrap
   probability of improvement is at least `0.90`, and wins exceed losses;
6. dynamic minus history-blind aligned Brier is at most `-0.015`, paired
   bootstrap probability of improvement is at least `0.80`, and wins exceed
   losses;
7. dynamic minus fixed aligned Brier is at most `-0.01`, paired bootstrap
   probability of improvement is at least `0.80`, and wins exceed losses;
8. dynamic mean aligned log loss is no worse than myopic, history blind, or
   fixed; and
9. among dynamic/myopic changed-root tasks, predicted Brier advantage has
   Spearman correlation at least `0.15` with realized Brier advantage and
   bootstrap probability of positive correlation at least `0.80`.

Only the conjunction supports a confirmed LLM-native non-myopic result. Pooled
development-plus-confirmation estimates, fresh-regeneration metrics, random
comparisons, and subgroup analyses are descriptive and cannot change status.
No threshold, support width, draw count, matcher, seed, model, policy, endpoint,
or favorable subset may change after development is observed.

## Budget And Execution Rule

Earliest execution is 2026-08-09 Europe/London. The run has a hard `$3.50` cap
inside the account-wide `$5.00` London-day cap. It does not repeat the optional
Luna baseline. Calls are not added merely to approach the cap, and unused daily
allowance does not roll over.

Once any confirmation model call is made, all mechanically valid scheduled
calls are mandatory; intermediate scientific endpoints remain sealed until
the complete result and independent replay exist. A provider failure may
produce only a partial, non-scientific artifact; it cannot authorize retrying
with changed settings or another model.

This freeze itself makes zero model calls and costs `$0`.
