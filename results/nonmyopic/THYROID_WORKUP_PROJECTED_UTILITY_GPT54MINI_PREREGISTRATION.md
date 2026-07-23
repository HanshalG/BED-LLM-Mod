# UCI Thyroid Projected-Utility GPT-5.4 Mini Preregistration

Status: frozen after the utility-grounded free-form serving failure and before any
projected-interface GPT-5.4 Mini response or scientific endpoint is observed.

## Method change

The utility-grounded model made the intended early choices but failed after both
responses repeated an already-used root in a late-state shrinking menu. This fresh
architecture retains the identical prompt, branch-local expected-entropy cards,
machine-fixed roots, and exact depth-two verifier. It changes only exhausted-invalid
handling:

1. The first invalid response receives the same bounded correction prompt.
2. If the second response remains invalid, every already-valid branch is preserved.
3. Each invalid or missing branch is projected to the legal continuation with minimum
   empirical posterior-predictive expected class entropy, with legal-order tie-break.
4. Every projected cell, branch, proposed value, replacement, and utility card is
   retained for independent replay.

Projection is a legality compiler, not an unreported retry. Scientific gates cap its
contribution so a result cannot pass if the machine silently authors much of the
policy.

## S0 late-state smoke

- Fresh seed `24161`; 12 exact-depth-two histories with lengths
  `0,1,2,3,4,5,6,6,5,4,3,2`.
- `openai/gpt-5.4-mini` via OpenRouter, non-thinking, temperature zero, 1,024-token
  output cap, one correction response.
- Required: all 12 cells complete and legal; no continuation repeats its root;
  history lengths 0--6 covered; **zero projected cells**; zero reasoning, forced
  exits, and scoring-time LLM calls.
- Projection behavior is separately unit-tested with deliberately invalid responses.
  Any live S0 projection or failure stops this line.

## Conditional S1 confirmation

S1 runs only if every S0 mechanic passes.

- Fresh seed `24162`; 50 patient rows without replacement; eight paired actions.
- Arms: projected-utility GPT depth-two continuations, matched-random continuations
  on identical machine-fixed roots, exact depth one, and exhaustive depth two. The
  last action uses the common exact depth-one rule.
- Primary endpoint: mean post-action target-entropy AUC. Corroboration:
  truth-log-posterior AUC. Ten thousand paired bootstrap replicates.
- Exactly 350 complete logical GPT cells. Physical requests may exceed 350 only by
  the one registered correction per invalid first response.

All gates are required:

1. Positive paired 95% lower bounds for entropy-AUC and truth-log-AUC gains over
   exact depth one.
2. Positive paired 95% lower bounds for both gains over matched random.
3. At least 60% recovery of exhaustive depth two's entropy gain over depth one.
4. Blood collection selected first on at least 75% of trajectories.
5. At most 5% of logical cells are projected (at most 17/350).
6. At most 1% of all returned branch choices are projected.
7. Fifty distinct paired truths, complete legal traces, exactly 350 logical cells,
   zero reasoning/forced/scoring calls, and independent replay of all decisions,
   controls, posteriors, aggregates, projection events, and fresh-bootstrap gates.

No alternate seed, prompt repair, threshold change, or replacement run follows a
failure. Expected OpenRouter cost is approximately `$2.2`, with the existing `$6`
run cap. Project spend is `$35.21476396 / $110`, leaving `$74.78523604` before S0.
