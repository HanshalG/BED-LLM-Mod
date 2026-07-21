# RockSample[7,8] Exact Follow-Up Closure Screen

Registered 2026-07-21 before computing any follow-up-closure metric.

## Question

The Gemma K6 StrategyEIG policy captures 85.8% of exhaustive d2 value over h2
decisions. Each LLM-proposed movement root currently carries only one proposed
follow-up check. Is the remaining proposal gap mostly due to those follow-up choices,
or are the root proposals themselves insufficient?

## Frozen Screen

- Input: the banked preregistered Gemma 4 26B A4B canonical RockSample[7,8] run,
  `results/nonmyopic/rocksample_7_8_scale_20260721/L1.json`.
- For every StrategyEIG h2 decision, reconstruct the exact visited belief and hold its
  K6 candidate root multiset fixed.
- For each proposed root, replace the single generated continuation only for scoring
  with the exact best legal second action on every nonzero outcome branch.
- Repeat the same closure for the matched random-strategy root sets on their own exact
  visited beliefs.
- Compare original and closed value to exhaustive d2 at that state. Do not execute a
  counterfactual policy, reuse a future LLM response, or report a policy endpoint.
- No LLM calls, random draws, likelihood approximation, or model changes.

## Gate

Engineering a paid closed-root policy is authorized only if all conditions hold over
the 270 nonterminal-horizon StrategyEIG decisions:

1. mean closed exhaustive fraction is at least `0.97`;
2. mean closed fraction exceeds mean original fraction by at least `0.08`;
3. at least `90%` of states contain an exhaustive-optimal d2 root after closure; and
4. the LLM-root closed fraction exceeds the random-root closed fraction by at least
   `0.03`.

Passing is proposal-fidelity evidence only. A subsequent policy run must be separately
preregistered and paired against original StrategyEIG, shared d1, random closed roots,
width, and exhaustive d2.

## Frozen Command

```bash
python scripts/analyze_nonmyopic_rock_followup_closure.py \
  results/nonmyopic/rocksample_7_8_scale_20260721/L1.json \
  --output results/nonmyopic/rocksample_7_8_followup_closure_20260721/REPORT.json \
  --summary-output results/nonmyopic/rocksample_7_8_followup_closure_20260721/REPORT.md
```
