# RevengeBench Source-Level Horizon Opportunity Protocol

Date: 2026-08-13

Status: **frozen after deterministic replay passed and before opening any
opportunity target-policy source, prior released trajectory, or outcome**.

## Purpose

Test whether RevengeBench's public intervention mechanics can support a genuine
non-myopic first-action advantage, rather than merely offering probes that
produce observations. This is a zero-model-call source audit. It cannot establish
LLM calibration or policy efficacy.

## Bindings

- RevengeBench commit:
  `351a5a7c2671150bae44c8bc46d7115ec996615f`;
- RevengeBench tree:
  `8991d42d09f3f8fb095580d87a68ba21b7dc6f5c`;
- source-admission protocol SHA-256:
  `90103e558a836c599442efd5e89e6e296d8c7c43d4c590f2ccb5053a1ad828a0`;
- deterministic-replay protocol SHA-256:
  `dea501bd7c17768cb847e7998c1ddd0bb765a57f775a70f27f2661c9398fc9e9`;
- deterministic-replay audit SHA-256 is recorded in the result before this
  audit runs.

The frozen opportunity cohort is exactly the three `opportunity` target IDs per
arena already committed by source admission. BattleSnake, Halite, HuskyBench,
and RoboCode are eligible because exact paired replay passed. RobotRumble is
excluded only for its banked local architecture incompatibility. No other
target or split may substitute after inspection.

## Privacy Boundary

Before this protocol was frozen, no opportunity target-policy source, target
provenance, released trace, result, action-distance endpoint, or model response
was opened.

The audit may now parse the 12 eligible opportunity target implementations only
into syntax/semantic dependency summaries. It must not execute them, inspect
their provenance, load released trajectories/outcomes, or serialize source
text, identifiers embedded inside comments, string literals unrelated to game
actions, or more than aggregate counts and salted hashes.

## Required Structural Contract

A target qualifies only if public source establishes all of:

1. **Semantic latent behavior:** its action policy contains at least two
   state-dependent strategic branches or thresholds; a constant/random policy
   does not qualify.
2. **Intervention-sensitive observation:** at least two executable public probe
   policies can induce distinguishable target-visible state/action histories
   under paired seeds without modifying target internals.
3. **Adaptive continuation:** after observing the first probe history, a second
   probe can be selected from at least two remaining policies and can produce a
   different likelihood partition over the target strategy class.
4. **Irreversible first action:** the first probe consumes one of the frozen
   finite probe opportunities; both candidate first probes cannot be recovered
   in the same two-probe budget.
5. **Non-additive horizon:** source-level branch partitions admit a witness in
   which depth two selects a different first probe than one-step information
   gain because the best second probe depends on the first observation.
6. **Compute-matched control:** the witness remains when receding-myopic is
   allowed the same number of candidate evaluations and chooses its second
   probe adaptively after the same first observation.
7. **Endpoint separation:** at least one held-out target-visible state lies
   outside the probe histories, and alternative surviving strategy branches
   prescribe different actions there. Thus held-out action distance need not
   saturate merely because probe histories differ.
8. **LLM-native opening:** the strategy class cannot be exhaustively represented
   by the audit's finite witness alone. The witness only proves opportunity; a
   later LLM must generate or score semantic strategy hypotheses from code and
   histories, and known-pool/BPI remain explicit classical controls.

## Audit Method

For each eligible target:

1. parse its language with a structured parser where available and otherwise a
   language-aware token/tree-sitter parser;
2. identify game-state inputs, conditional branches, thresholds, and emitted
   actions without retaining source text;
3. pair the target with public probe-policy abstract action signatures from the
   frozen mechanics/reserve pool, never with prior outcomes;
4. construct a target-blind symbolic response partition for each probe;
5. enumerate all two-step adaptive probe trees over that symbolic partition;
6. compare exact depth-two expected entropy to compute-matched receding-myopic,
   fixed, and random controls under a uniform source-level strategy-class prior;
7. verify held-out action disagreement symbolically;
8. serialize only gate booleans, counts, margins, salted target hashes, and
   parser/source bindings.

If static analysis cannot prove intervention response partitions without
executing policies, the audit returns `execution_opportunity_required`. That is
not a pass and authorizes only a separately frozen, endpoint-sealed execution
opportunity audit on the same cohort and paired replay machinery.

## Gates

The source-level opportunity gate passes only if:

- at least 8 of 12 targets are parseable and semantically nontrivial;
- at least 6 of 12, spanning at least three arenas, satisfy contracts 1--7;
- depth two changes the first probe versus compute-matched receding-myopic on at
  least 3 targets;
- depth two has strictly positive exact expected entropy advantage on every
  changed target;
- held-out action disagreement exists on every changed target;
- no target source/provenance/trajectory/outcome is serialized;
- OpenRouter calls and cost are zero.

## Decision Rule

- **Pass:** freeze a separate LLM semantic strategy-hypothesis and likelihood
  calibration interface before any model response.
- **Execution opportunity required:** freeze a zero-model-call, paired execution
  audit on these exact targets/probes. Do not open endpoints or model work.
- **Fail:** close RevengeBench if source proves fewer than six qualifying targets
  or no adaptive first-action witness.

No threshold may change after target source is opened. This protocol authorizes
no OpenRouter request, no endpoint evaluation, and no paper claim.
