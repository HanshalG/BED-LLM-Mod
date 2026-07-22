# Gated Sensor v4 Informative-Terminal Smoke Preregistration

Registered 2026-07-22 after inspecting the indexed-v2/v3 four-pair direct-vLLM
smokes and a zero-call v4 structural dry run, but before any v4 model response.

## Motivation

The 26B indexed-v3 smoke improved measurement-root continuation efficiency from
`0.3200` to `0.5040` and reduced terminal panel activations from `0.4904` to
`0.2404`, but StrategyEIG still lost `0.1579` entropy-AUC nats to matched random.
Trace inspection showed premature panel switching. Panel activation emits no
observation, so it cannot reduce terminal entropy when selected as the final action
of a depth-two strategy.

V4 changes only this horizon-invalid menu exposure. It retains v3's exact branch
belief summaries, machine-assigned roots, K4 width, parser, exact scorer, posterior,
controls, and action execution. At the final branch-policy step:

- an activation root offers the precise tests unlocked by that panel;
- a measurement root offers only screen or currently legal precise tests;
- no measurement-root branch menu contains a panel activation.

No EIG value, policy score, hidden truth, or oracle action ranking is shown to the
model. V2 and v3 remain unchanged and reproducible.

## Pre-Response Structural Check

A deterministic zero-call dry run at seed `24117` completed 4 paired trials by 8
rounds. Its choice audit reconstructed 164 branch choices and found zero terminal
activations and zero zero-information selections. Its policy endpoint was inspected,
so seed `24117` is quarantined and cannot be used for this smoke.

## Frozen Smoke

- Model: `google/gemma-4-26B-A4B-it`, direct vLLM, non-thinking, temperature zero.
- Scheduler: `msc,llm`, excluding `oat12`; one A100.
- Interface: `indexed_branch_v4`, K4, exact depth-two verifier.
- Evaluation: 4 paired trials, 8 rounds, fresh seed `24118`, 2,000 paired bootstrap
  replicates, trial concurrency one.
- Arms: StrategyEIG, shared-cell d1, exhaustive d1, matched-random v4 strategies,
  and exhaustive d2.
- Expected nonterminal proposal cells: 52 before any bounded retry.
- Exact rollout scoring, all controls, posterior updates, and metrics make zero LLM
  calls.

## Gate

The smoke passes only if:

1. all registered legality, pairing, root-coverage, and zero-rollout-LLM mechanics
   pass;
2. every accepted measurement-root terminal menu excludes activation actions;
3. the run completes with zero reasoning tokens and zero forced exits;
4. StrategyEIG's paired entropy-AUC 95% bootstrap lower bound is strictly positive
   against shared d1, exhaustive d1, and matched random.

Truth-log AUC, the gap to exhaustive d2, immediate-EIG choice efficiency, retries,
and action traces are mandatory diagnostics but cannot rescue a failed gate. A pass
would authorize only a separately preregistered fresh-seed confirmation. A failure
stops this Gated Sensor interface line; no threshold or same-seed repair is allowed.

## Registered Outcome

Job `106342` completed on `msc/oat14` and failed the frozen matched-random gate. All
mechanics passed; 52 cells were accepted after three corrected first-attempt index
errors, with zero reasoning tokens, forced exits, or rollout-scoring LLM calls.

V4 eliminated the targeted pathology: the independent choice audit found zero
terminal activations and zero zero-EIG selections across 164 branch choices. The
deployed policy used exactly two activations and six precise tests in every trial.
StrategyEIG beat shared and exhaustive d1 by `+1.1945` entropy-AUC nats (95% CI
`[+1.0965, +1.2924]`) and essentially matched exhaustive d2 (`+0.0039`,
`[0.0000, +0.0117]`). However, it was `-0.0352` versus matched random
(`[-0.1173, +0.0167]`; 2/0/2 wins/ties/losses), so no formal run is authorized.
