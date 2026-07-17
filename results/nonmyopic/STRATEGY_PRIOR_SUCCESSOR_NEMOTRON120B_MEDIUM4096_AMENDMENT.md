# Nemotron 3 Super 120B Serving Configuration Amendment

Registered on 2026-07-18 before the replacement Nemotron serving-gate call. The prior
Nemotron smoke exhausted a 1,536-token completion allocation in reasoning and returned
no final JSON. It therefore did not test a viable total-output configuration.

This amendment changes only serving allocation: the model uses OpenRouter native
`reasoning.effort: medium` with `max_tokens: 4096`. Medium reasoning is retained, while
the larger output ceiling reserves room for the compact strategy JSON after reasoning.
The gate remains five strict L1 plus five strict L3 first-response cells, all parsed and
executed without repair and with zero forced exits.

The frozen formal endpoint is unchanged: 30 paired trials, 30 rounds, 64 particles plus
truth, K=4, horizon 4, 64 CRN rollouts, five arms, seed 31003, and the same paired
bootstrap pass rule. At 1,373 reference calls, 4,096 maximum completion tokens at
`$0.455/M` cost approximately `$2.56`; historical prompts at `$0.21/M` add about
`$0.10`. The formal configuration reserves `$2.75` against its independent `$3.00`
hard ledger cap.

OpenRouter reasoning allocation reference:
<https://openrouter.ai/docs/guides/best-practices/reasoning-tokens>.
