# RockSample[11,11] Exact Follow-Up Closure Screen

Registered 2026-07-21 before computing any eleven-rock follow-up-closure metric.

## Question

The preregistered Gemma K6 policy captures 73.9% of exhaustive d2 value over h2
decisions on RockSample[11,11], down from 85.8% in the original eight-rock run.
Does scale primarily degrade the generated second actions, or do the six proposed
roots themselves omit high-value parts of the exact action space?

## Frozen Screen

- Input: the banked seed-24079 Gemma root-slot confirmation,
  `results/nonmyopic/rocksample_11_11_gemma_slot_confirmation_20260721/L1.json`.
- Reconstruct every exact StrategyEIG and matched-random visited belief over the 330
  nonterminal-horizon decisions per arm.
- Hold each K6 candidate root multiset fixed. For scoring only, replace generated
  continuations with the exact best legal second action on every positive-mass branch.
- Compare original and closed values with exhaustive d2 at the same state. Do not
  execute a counterfactual policy or report a policy endpoint.
- Use the same implementation and thresholds as the preregistered eight-rock screen,
  generalized only for the frozen map, run ID, trial count, and round count.
- No LLM calls, random draws, likelihood approximation, or model changes.

## Gate

A separately preregistered closed-root policy is authorized only if all conditions
hold over StrategyEIG states:

1. mean closed exhaustive fraction is at least `0.97`;
2. mean closed fraction exceeds the original fraction by at least `0.08`;
3. at least `90%` of states contain an exhaustive-optimal d2 root; and
4. LLM-root closed fraction exceeds random-root closed fraction by at least `0.03`.

Passing is proposal-fidelity evidence only. Failure stops paid closure work on this
map. The result will be reported regardless of direction.

## Frozen Command

```bash
python scripts/analyze_nonmyopic_rock_followup_closure.py \
  results/nonmyopic/rocksample_11_11_gemma_slot_confirmation_20260721/L1.json \
  --output results/nonmyopic/rocksample_11_11_followup_closure_20260721/REPORT.json \
  --summary-output results/nonmyopic/rocksample_11_11_followup_closure_20260721/REPORT.md
```

## Registered Outcome

The gate failed. Exact continuation closure raised Gemma's mean exhaustive fraction
from `0.7390` to `0.9686` (`+0.2296`) and found an exhaustive-optimal root in
`317/330 = 0.9606` states. It never worsened a state. Matched random roots rose from
`0.2304` to `0.9669`, leaving an LLM-minus-random closed gap of only `+0.0017`.

Thus continuation quality explains most of both the original LLM/random difference
and Gemma's scale-dependent gap. The `0.97` closed-fraction condition narrowly failed,
and the required `+0.03` closed LLM/random separation failed decisively. The screen
made zero LLM calls and, as frozen, no paid closed-root policy follows.
