# Animals Multi-Sample Generator Screen

Status: **development interface screen on inspected histories**.

The fixed single-list generator recovered truth in only `3/20` branch unions
with Gemma 4 26B and `1/20` with GPT-5.4 Mini. This screen changes the
generation interface rather than the model:

- exact seed-24279 histories and 120 counterfactual branches;
- non-thinking Gemma 4 26B;
- four independent generation calls per branch at temperature `.7`;
- up to 16 names per call, merged before unchanged cleanup, validation,
  deduplication, and history filtering;
- no target-derived input or policy scoring.

Default `belief_generation_num_calls=1` preserves all existing behavior.

Pass requires all 20 states/120 branches, zero reasoning, cost below `$2`, and
truth in at least `8/20` branch unions. Pass authorizes scorer development on
the inspected multi-sample records. Failure rejects this Animals formulation
unless the task or hypothesis representation changes materially.
