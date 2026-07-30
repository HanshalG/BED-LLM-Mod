# Number Game Qwen Fully Fresh Source + Control-32 Preregistration

Date frozen: 2026-07-30

## Purpose

This is a fully fresh end-to-end replication of the Number Game's LLM-native
non-myopic policy and its path-dependent belief mechanism. Unlike the prior
matched controls, the dynamic planning trees, independent validation supports,
realized source outcomes, and history-blind controls are all unopened when
this protocol is frozen.

The run asks jointly whether:

1. non-myopic planning over answer-conditioned LLM support beats both myopic
   EIG and compute-matched fixed-support depth three; and
2. answer conditioning improves the second-stage belief state, with
   root-specific conditioning benefit calibrated to realized dynamic-selection
   advantage.

This is a new replication. It cannot rescue, relabel, or change any prior
source or control result.

## Models And Fresh Seeds

- planning and branch-support model: `qwen/qwen3.7-plus`, nonreasoning;
- independent target/validation model: `google/gemini-2.5-flash`,
  nonreasoning;
- source tree seeds: `100000..100031`;
- source target seeds: `100100..100131`;
- validation seeds: 16 per tree beginning at `100200`;
- source bootstrap seed: `100800`;
- history-blind control seeds begin at `10000000`;
- control bootstrap seed: `10100000`.

No prior tree, target, validation response, history-blind response, or
scientific endpoint is reused.

## Stage A: Fresh Source Study

Generate 32 retained-support depth-three planning trees using the exact
resilient Qwen/Gemini interface validated by the 96-tree source study:

- two Qwen draws per planning history;
- 49 planning histories per tree;
- one independent target draw per tree;
- 16 independent Gemini validation supports per tree;
- 115 accepted requests per tree, exactly 3,680 total.

Source run budget is `$5.25`. Mechanics require:

1. exactly 32 trees and 33 unique canonical targets;
2. exact accepted-request and HTTP/retry accounting;
3. no more than 96 retries, 96 provider-error retries, 16 item-salvaged
   draws, and 32 deterministic provider-seed fallback events;
4. zero reasoning tokens and forced exits;
5. every pooled initial support has at least 24 rules;
6. every deployed first branch has at least 12 retained rules and every
   deployed second branch at least 8;
7. all 16 validation supports per tree contain at least 16 rules;
8. source cost is at most `$5.25`.

Source scientific gates are:

### Non-Myopic Versus Myopic

1. at least 8% canonical Brier reduction;
2. paired tree-bootstrap 95% interval for the Brier difference entirely below
   zero;
3. at least 20 tree wins.

### Dynamic Versus Compute-Matched Fixed Support

1. dynamic and fixed roots differ on at least 20 trees;
2. at least 3% canonical Brier reduction;
3. paired tree-bootstrap 95% interval entirely below zero;
4. at least 16 tree wins.

## Stage B: Fresh Matched History-Blind Control

Stage B is authorized only when all Stage A mechanics gates pass and at least
20 dynamic/fixed roots differ. This is a structural opportunity gate, not a
policy-efficacy gate.

Crucially, Stage B proceeds regardless of the observed myopic or fixed-support
Brier comparisons. No human or endpoint-dependent decision occurs between
stages.

For each of the 32 fresh source trees:

- enumerate the same 48 first/second branch slots;
- make two fresh Qwen no-observation generations per slot;
- use exactly 3,072 accepted control requests;
- apply the same validity filtering, consistency filtering, and recursive
  parent retention to conditional and history-blind supports.

Control run budget is `$4.25`. Mechanics are the V3/confirmation mechanics:
exact accounting, at most 32 retries/provider retries, strict JSON, at least
16 valid rules per draw, at least 24 per pool, zero reasoning/forced exits,
and cost at most `$4.25`. Second-draw novelty is descriptive only.

Control scientific gates are:

1. at least 20 changed-root trees;
2. second-stage `conditional MSE - history-blind MSE` has a 95% interval
   entirely below zero;
3. second-stage `conditional coverage - history-blind coverage` has a 95%
   interval with lower endpoint at least zero;
4. changed-root conditioning-benefit contrast versus realized
   dynamic-selection advantage has a Spearman-bootstrap interval entirely
   above zero.

The selected-root mean conditioning-benefit contrast and interval remain
descriptive, not a success gate.

## Composite Decision

The composite passes only when:

- all Stage A mechanics gates pass;
- all Stage A non-myopic-versus-myopic gates pass;
- all Stage A dynamic-versus-fixed gates pass;
- all Stage B mechanics gates pass; and
- all Stage B scientific gates pass.

If Stage A mechanics fail, no Stage B calls are made and the run is
`mechanics_failed`. If fewer than 20 roots differ, no Stage B calls are made
and the run is `opportunity_failed`. If mechanics pass but any scientific
gate fails, the result is `gated_null`. There is no subset rescue, seed
substitution, response reuse, or endpoint-driven rerun.

## Budget And Authorization

- exact accepted requests on a complete run: `3,680 + 3,072 = 6,752`;
- Stage A cap: `$5.25`;
- Stage B cap: `$4.25`;
- composite cap: `$9.50`;
- minimum authenticated starting balance: `$10.25`;
- concurrency and transport behavior remain those of the validated component
  runners.

The Qwen and Gemini exact-10 serving artifacts and the completed 96-tree/V3
component runs are hash-bound as interface evidence. No new paid smoke is
needed unless the model identifier, prompt schema, parser, provider route, or
transport implementation changes.

At freeze time the authenticated balance is `$2.469132069`, so execution is
forbidden until at least `$7.780867931` is added and the live balance reaches
`$10.25`.

## Replay

Before banking any result:

- public source `RESULT.json`, `TREES.json`, and `TARGETS.json` hashes must be
  frozen;
- public control `RESULT.json` and `CONTROLS.json` hashes must be frozen;
- an independent zero-call replay must reproduce every control tree and the
  complete 20,000-sample control bootstrap;
- the claim manifest must bind all component mechanics and scientific gates.
