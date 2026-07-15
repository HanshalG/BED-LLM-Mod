# Dynamic Location LLM Candidate-Proposal Pilot: Preregistration

**Status:** preregistered before any calls from this pilot. This is an exploratory
screen, not confirmatory evidence.

## Rationale and Oracle Gate

The frozen UCI static-pool pilot was a non-promotion: exact matched-width one-step
selection is stronger there than lookahead. This new pilot uses the already validated
branch-decoy/local-bump locality-constrained location task. In the exact current-code
oracle, depth-two EIG beats an *exhaustive* one-step EIG planner that evaluates every
currently legal move; with five paired reproduction trials, final RMSE is `0.1550`
for depth two versus `0.6689` for exhaustive one step. Thus wider current-state
coverage cannot reach moves that become legal only after committing a local step.

## Frozen Contract

- Latent target: one two-dimensional source location sampled from the exact
  branch-decoy prior.
- Observations: log-normal local-bump signals with the fixed model and noise below.
- Legal action: one grid position within `0.5` Euclidean units of the prior position;
  the first query is fixed at the origin.
- Posterior: fixed exact particle support plus the latent truth, updated with the
  analytic log-normal likelihood.
- Decode: the posterior mean source location. Endpoint: its realized Euclidean RMSE
  to the latent source, with posterior entropy as a secondary read.
- The LLM only proposes exact legal `q_<grid-index>` action IDs. It cannot answer,
  score, update the posterior, or decode.

## Fixed Pilot

- Six paired trials, six rounds, one source, seed `1304`.
- Geometry: 11x11 grid over `[-2.5, 2.5]^2`; max step radius `0.5`; branch-decoy
  prior with radius `2.2`; local-bump signal with lengthscale `0.5`, amplitude `8.0`;
  log-normal noise SD `0.15`; 40 initial particles; the first origin query is fixed.
- Model: non-thinking `google/gemma-4-26b-a4b-it`, temperature `0.7`, output cap
  128 tokens. A candidate cell is a bare JSON object or one exact ` ```json ... ``` `
  wrapper around it, with exactly two distinct legal action IDs.
- One-step EIG uses the full fixed particle posterior. The depth-two continuation
  uses the top two posterior particles and three fixed log-noise quadrature values
  (`-1, 0, 1`) for its future-action branches.
- Arms:
  1. `d1_shared`: one-step EIG over a shared LLM `K=2` root cell.
  2. `d2`: two-step EIG rollout scoring over that same root cell plus LLM cells at
     each counterfactual future local state.
  3. `d1_matched_width`: one-step EIG over the union of the root cell plus the
     same number of current-state LLM proposal cells as the depth-two branch plan.
     Later width calls are shown earlier current-state IDs to encourage coverage;
     duplicates are legal but deduplicated and logged.
- Common random numbers: every arm receives the same hidden source, initial particle
  support, and per-round log-noise z values. Candidate cells are cached by trial,
  exact state fingerprint, and proposal label, sharing cells whenever states match.

## Mechanics and Read

Every selected action must be legal. A candidate cell is retried once when malformed;
a terminal cell failure aborts the run. Raw rejected attempts are logged but are not a
mechanics failure when the bounded retry returns a valid legal cell, because the
deployed candidate pool remains exact and the retry rule is fixed before this launch.

The pilot is promotable only when terminal mechanics pass and d2 has lower paired RMSE
AUC than both `d1_shared` and `d1_matched_width`. Results remain descriptive until one
powered, outcome-blind confirmatory run is separately preregistered.

## Budget

At five LLM-selected decisions per trial, d2 needs at most `1 + 2 * 2 * 3 = 13`
candidate cells per decision and the matched-width control receives the same allocation.
The worst case is `6 * 5 * (13 + 13 + 1) = 810` logical cells before cache reuse and
retries. Projected OpenRouter spend is `$0.15`; the hard run cap is `$0.50`, below the
protocol's `$1` exploratory ceiling.

## Command

```bash
set -a; source .env; set +a
PYTHONPATH=. python scripts/nonmyopic_dynamic_location_pilot.py \
  --config configs/config_nonmyopic_dynamic_location_pilot_openrouter.yaml \
  --run-id nonmyopic-dynamic-location-pilot-20260715
```
