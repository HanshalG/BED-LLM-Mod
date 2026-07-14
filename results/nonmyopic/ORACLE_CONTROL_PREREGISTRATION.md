# Phase 1 Preregistration: Restricted-Pool Oracle Control

Registered: 2026-07-14, before running `scripts/nonmyopic_oracle_control.py`.

## Purpose

This zero-LLM-call control tests the central mechanism before any new model
spend: a restricted question proposal pool can make a lower-immediate-EIG
question valuable because it leads to a posterior that is more splittable by the
next restricted pool. The task is an exact, in-contract 20 Questions variant:
the hidden target is an animal identity, every answer is a deterministic lookup
in a frozen trait matrix, and the endpoint is the MAP identity decode.

## Frozen Data and Protocol

- Matrix: `data/nonmyopic/uci_zoo.data`, UCI Zoo, 101 rows, SHA-256
  `cddc71c26ab9bc82795b8f4ff114cade41885d92720c6af29ffb69bcf73f0315`.
- Actions: 15 Boolean UCI traits plus the six predicates `legs == 0, 2, 4, 5,
  6, 8`; no direct identity-decode action is available.
- Prior: uniform over the 101 released rows. The two duplicated `frog` names are
  retained as distinct released instances; if the trait matrix cannot separate a
  target, exact MAP accuracy correctly reflects that ambiguity.
- Answerer and likelihood: exact deterministic matrix lookup and exact Bayesian
  filtering.
- Decode/endpoint: choose the deterministic first MAP row after each question;
  report exact-decode accuracy by round, AUC over rounds, final accuracy, and
  posterior entropy.
- Pairing/CRN: every policy condition shares target draws and a deterministic,
  history-keyed random ordering of all unasked traits. A restricted pool is the
  prefix of that ordering, so wider one-step pools contain the narrow-pool
  candidates exactly.
- Primary run: 2,000 paired target draws, 8 rounds, seed 1304, and 10,000
  deterministic percentile-bootstrap replicates.
- At any round with fewer than two actions remaining, the requested depth-two
  planner truncates to one-step EIG. It never plans a query past the fixed
  endpoint budget.

## Conditions and Predictions

1. **Exhaustive, noiseless:** all available traits are candidates. Exact depth
   two and one-step greedy should be close overall; any finite-horizon gap will
   be localized with the per-round trace.
2. **Restriction:** candidate widths `2, 3, 4, 6, 8`, all noiseless. Exact
   depth two versus one step uses the same current candidate pool. The predicted
   depth-two AUC advantage grows as the pool narrows, although strict monotonicity
   across every adjacent width is not required.
3. **Width control:** for every restricted width, one-step greedy also receives
   nested widths `2K` and `4K` (capped by the unasked action set). This measures
   how rapidly proposal coverage closes any restricted-pool gap.
4. **Noise frontier:** at primary width `K=3`, perturb each action value with
   independent deterministic zero-mean Gaussian score noise with standard
   deviations `0, 0.025, 0.05, 0.10, 0.20, 0.40` nats. Depth two has noise at
   both its root and future score comparisons. The predicted depth advantage
   decreases with noise; the first noise level whose paired AUC interval includes
   zero is the planning-viability boundary.

## Decision Rule

The script will report `proceed_to_llm_exploration` only if at least one
restricted noiseless condition has a positive paired depth-two minus one-step
AUC with a 95% bootstrap interval excluding zero, and the noise frontier shows
the pre-registered headwind direction. If no restricted condition opens a gap,
or injected noise does not act as a headwind, the mechanism is not established
in this control and no LLM experiment is authorized without discussion.

## Scope Fence

This is an oracle mechanism control, not an LLM result and not evidence that
the UCI trait matrix reproduces conversational 20 Questions. It precedes any
exploration pilot and makes no model calls or API spend.

## Execution Correction Log

The first complete local execution was quarantined before interpretation. It
allowed the requested depth-two selector to score a nonexistent ninth question
after the eighth and final action, violating the registered finite-horizon
endpoint. The anomaly was visible in an implausibly large exhaustive-pool
penalty. The implementation now truncates lookahead to the remaining action
budget, with a regression test; the control is rerun from the same seed and
conditions. This is a mechanics correction, not a change to the data, candidate
schedule, outcomes, widths, noise grid, or decision rule.
