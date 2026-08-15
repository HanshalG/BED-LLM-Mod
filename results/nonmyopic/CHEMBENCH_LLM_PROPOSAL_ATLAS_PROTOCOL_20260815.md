# ChemBench LLM Proposal-Atlas Protocol

Date frozen: 2026-08-15 (Europe/London)

## Purpose

This is the first paid dependency after the factored M-open oracle gate. It
tests whether an LLM can perform the irreducible semantic transition required
by the planner:

```text
pool-wide residual history -> typed executable missing-mechanism edits.
```

It is not a policy-efficacy experiment. It uses source-only v3 synthetic
histories, opens no v4 LLM policy result, and leaves all v5 responses sealed.
Failure closes this exact proposer interface before any LLM policy endpoint.

## Immutable Dependencies

- Source commit/tree:
  `acf160eb6c96897748dd92b152703b59b74efc05` /
  `e042d418fc30c6c70f6d1c6b0636d43f0c1c0f7a`.
- Factored oracle protocol SHA256:
  `00d11d703098205122d3913ded06242f85d2c67f90fe7978130e483c3d39de66`.
- Factored oracle raw result content SHA256:
  `2f1cdb224a5628d9b9e342c8d29dccf064fd9165db4a0fe40358be4192c7eb81`.
- Factored oracle raw transition-bank content SHA256:
  `b9af378701f813c3a61cd5725e8f54b4b461826196c39b3386726fe7cc404db1`.
- Factored oracle independent verification SHA256:
  `d97daa7dacbbeb7ab0a08907fd4e3785f92a319e0aa224f5ebdfc5f9059594f3`.
- Required pushed result commit: `25ea5f11` or a descendant that hash-binds
  this protocol without changing its science.

The passed oracle ladder is d1/d2/d3 terminal MSE
`.05439110/.03781695/.03176550`, with successive reductions 30.47% and
16.00%. This authorizes only the first-link test below.

## Source-Only Task Construction

Build separate easy, medium, and hard banks with:

- primitive initial support at source version `v2`;
- all outside candidate/generator mechanisms at source version `v3`;
- the frozen 18 assays and categorical likelihood;
- no v4 or v5 value, parameter row, truth outcome, or policy result.

For each of the 48 outside generators and each difficulty:

1. start from the nine primitive support models;
2. among unused actions, choose the assay maximizing exact KL divergence from
   the generator's categorical outcome distribution to the current represented
   support predictive;
3. realize the generator's modal categorical outcome;
4. update the fixed primitive support without proposing a model;
5. stop at the earliest history length one, two, or three where the latest
   observation triggers the frozen surprise gate and the true missing model is
   in the typed registry oracle's top four proposals.

Rows not recoverable by depth three are inadmissible. This is a source
opportunity condition, not an LLM score.

Create nine strata `(difficulty, earliest_history_length)`. Select exactly four
tasks per stratum by deterministic greedy family coverage: repeatedly select a
row whose core family has appeared least often in that stratum, breaking ties
by SHA256 of

```text
2026083700 | difficulty | history_length | model_name
```

and then by model name. The resulting panel has exactly 36 unique hidden
generators. In each stratum, the first three selected rows are atlas-development
tasks and the fourth is held-out transition-fidelity. Thus there are 27 atlas
and nine held-out tasks. The task manifest is frozen and hash-bound before any
model request.

The source gate requires:

- exactly four admissible tasks in every stratum;
- 36 unique generators;
- at least ten distinct core families overall;
- every latest state triggers expansion;
- every true generator is in the source oracle top four;
- no truth/model name or oracle rank appears in a model prompt.

## LLM Interface

Use exact model `deepseek/deepseek-v4-flash-0731`, explicitly nonreasoning, with
OpenRouter parameter-constrained routing. Do not pin a provider that has
already failed transport. Require support for `seed`, `reasoning`, and strict
structured output from an active endpoint. Re-read all eligible endpoint prices
before each block and reserve the componentwise maximum under ceilings of
`$0.20/M` prompt tokens and `$0.50/M` completion tokens. A malformed, empty, or
higher-priced eligible set stops before requests.

Frozen generation settings:

- temperature `0.3`;
- maximum completion tokens `700`;
- zero retries;
- concurrency at most `64`;
- exact request seed shared by each residual-aware/history-blind pair;
- each serialized prompt at most 24,000 characters;
- no tools, browsing, images, or hidden chain-of-thought;
- reasoning explicitly disabled and reasoning content rejected.

Each of the 27 atlas tasks receives two seeds per arm. Each of the nine held-out
tasks receives one seed per arm. This is 63 residual-aware and 63 matched
history-blind requests, 126 total.

Seeds are assigned after sorting tasks by `(difficulty, history_length,
task_id)`: `202608370000 + request_position`, with the same seed used for the
two arms. The second atlas replicate uses the next disjoint block beginning at
`202608371000`.

## Prompt Information Boundary

Both arms receive:

- the enzyme-kinetics domain description and complete legal mechanism grammar;
- definitions of every allowed core family and modifier;
- the nine current executable primitive structures and canonical signatures;
- the frozen action-group semantics;
- remaining experiment budget and `explore` phase;
- the exact strict output schema.

The residual-aware arm additionally receives only public state information:

- action/outcome history using categorical low/mid/high outcomes;
- every represented model's evidence weight and categorical NLL;
- signed innovation by assay group;
- latest prequential probability and surprise;
- already tried structures.

The history-blind arm receives the history length and remaining budget but no
actions, outcomes, residuals, likelihoods, or surprise. It is independently
sampled at matched calls/seeds; responses are not reused across tasks.

Neither arm receives the hidden generator, correct edit, oracle proposals,
candidate registry combinations, endpoint loss, desired assay, policy value,
v4/v5 data, or another arm's response.

## Strict Typed Output

Each response must contain exactly four distinct proposals:

```json
{
  "proposals": [
    {
      "parent_model_id": "c0_michaelis_menten",
      "operation": "add_factor",
      "core_family": "michaelis_menten",
      "modifiers": ["arrhenius"],
      "residual_motif": "underprediction grows with temperature",
      "exposing_assay_group": "T",
      "falsifying_assay_group": "C_I"
    }
  ]
}
```

Allowed operations are `add_factor`, `remove_factor`, `replace_factor`, and
`replace_core`. Parent IDs must be currently represented. Core families and
modifiers must be exact members of the public grammar. The canonical
`(core_family, modifiers)` signature must map uniquely to one active outside
registry mechanism, and compiling that candidate relative to the declared
parent must produce the declared operation. Candidate signatures must be
distinct, untried, finite, and executable. The parser never repairs IDs,
aliases, operations, signatures, or proposal counts.

Rationales and exposing/falsifying groups must be nonempty but are not judged
for style. No LLM judge is used.

## Proposal Metrics

Score every accepted response independently.

- `schema_valid`: strict response shape and exactly four items.
- `item_compile_rate`: fraction of the four items that compile exactly.
- `response_executable`: all four items compile and are distinct.
- `truth_recall_at_4`: the hidden generator signature is proposed.
- `core_family_recall_at_4`: any proposal has the hidden core family.
- `modifier_f1`: best proposal-to-truth modifier-set F1.
- `semantic_score`: `2*truth_recall + core_family_recall + modifier_f1`, used
  only for paired dynamic-versus-blind wins/ties/losses.

Transport failures and invalid schemas score zero on all semantic metrics.

## Frozen Proposal Atlas

Represent each task state by a truth-free numerical vector:

- history length;
- latest action-group one-hot and outcome one-hot;
- latest surprise;
- nine primitive-model evidence weights and categorical NLLs;
- mean signed innovation for every `(primitive model, assay group)` pair.

Missing group-history entries are zero with a separate observed-mask bit.
Standardize dimensions using only the 27 atlas tasks.

For each held-out task, retrieve the three nearest atlas tasks in Euclidean
feature distance. Pool their residual-aware compiled proposals from both seeds,
weighting each occurrence by `exp(-distance)`, and select the four candidates of
highest total weight, breaking ties by registry ID. No hidden generator label,
oracle proposal, held-out LLM response, or endpoint value enters retrieval.

Report on the nine held-out tasks:

- atlas truth and core-family recall at four;
- mean Jaccard overlap between atlas and fresh residual-aware LLM core-family
  sets;
- proposal-induced one-step risk for atlas, fresh residual-aware, blind, random
  typed edits, and source oracle.

For a proposal set, proposal-induced risk is the minimum over unused assays of
the hidden generator's expected terminal held-out log-rate MSE after admitting
the proposals, observing one categorical outcome, updating numerically, and
making no further proposal. This evaluator is used only after responses and is
never exposed to the LLM or atlas retrieval.

Random typed edits use four distinct admissible outside signatures sampled from
the same public grammar with task-bound seed `202608372000 + task_position`.

## Conjunctive Gate

All conditions must pass:

1. Source/task manifest, exact model identity, prompt privacy, seed identity,
   nonreasoning, price, daily-budget, no-retry, and independent replay checks
   pass.
2. At least 90% of all 126 requests have a clean terminal response from the
   exact requested model, and at least 90% are strict-schema valid.
3. Residual-aware item compile rate is at least 80%, and at least 70% of its
   responses have all four distinct executable proposals.
4. Residual-aware truth recall at four is at least 25%, and core-family recall
   at four is at least 45%, over all 63 responses.
5. Residual-aware mean best modifier F1 is at least 0.45.
6. Residual-aware truth recall exceeds history-blind by at least ten percentage
   points, and core-family recall exceeds history-blind by at least ten points.
7. Residual-aware paired semantic-score wins exceed losses at exact tie
   tolerance `1e-12`.
8. On the nine held-out tasks, fresh residual-aware truth recall is at least
   `2/9` and core-family recall at least `4/9`.
9. On held-out tasks, atlas truth recall is at least `2/9`, atlas core-family
   recall at least `4/9`, and mean atlas/fresh core-family Jaccard is at least
   `0.20`.
10. Mean proposal-induced risk is lower for residual-aware than history-blind
    and random typed edits; atlas risk is lower than history-blind and random;
    and neither residual-aware nor atlas risk exceeds source-oracle risk by more
    than 25% relative.
11. An independent verifier reconstructs the task panel, prompts, parser,
    compiled candidates, metrics, atlas retrieval, one-step risks, costs, and
    every gate condition from immutable public records plus private raw-response
    hashes.

Failure is a semantic/transition first-link null. Do not tune task selection,
prompt content, grammar, model settings, thresholds, nearest-neighbor count,
distance features, seeds, or controls after responses. A transport-only failure
with zero clean responses may authorize a separately frozen transport repair;
a semantic failure may not.

A pass authorizes a new, prospectively frozen opened-v4 LLM policy-development
protocol comparing dynamic d1/d2/d3 against real-history-only MDA,
compute-matched myopic, history-blind, random-edit, fixed-support, and oracle
controls. It still does not directly authorize v5.

## Budget

The immutable Aug15 account-wide opening usage is `220.339269126`. Latest
authenticated cumulative credits/usage/balance before this freeze are
`245.000000000/220.348811737/24.651188263`, so prior Aug15 account spend is
`$0.009542611` and remaining hard daily allowance is `$4.990457389`.

Reserve the full 126-request worst-case exposure immediately before dispatch,
using each prompt's actual bounded length conservatively as one token per
character and 700 completion tokens at the maximum eligible endpoint prices.
The proposal-atlas stage cap is `$0.75`, within the account-wide `$5.00` cap.
Unrelated account usage counts. Any negative usage delta, malformed account or
catalog value, insufficient allowance, price-ceiling violation, or
post-preflight race opens no ledger or model request. Unused allowance is not a
reason to issue extra calls.
