# UCI Zoo LLM Candidate-Proposal Pilot: Preregistration

**Status:** preregistered before any calls from this pilot. This is an exploratory
pilot, not confirmatory evidence.

## Question

Does two-step exact EIG acquisition outperform one-step exact EIG when a
non-thinking `google/gemma-4-26b-a4b-it` LLM supplies only a small legal candidate
pool at each decision?

## Frozen Contract

- Data: the hash-pinned 101-entity UCI Zoo matrix in
  `data/nonmyopic/uci_zoo.data` (`cddc71c26ab9bc82795b8f4ff114cade41885d92720c6af29ffb69bcf73f0315`).
- Latent target: one sampled matrix row per paired task.
- Legal actions: the 21 frozen Boolean trait IDs defined in
  `scripts/nonmyopic_oracle_control.py`.
- Observations, posterior filtering, exact EIG, and deterministic MAP identity
  decoding are all programmatic. The LLM cannot answer an action, set a likelihood,
  or decode the target.
- The LLM's sole output is a JSON object containing exactly `K=3` distinct, unasked,
  legal trait IDs. Invalid output is retried once then fails closed; it is never
  padded or replaced by an analytic/deterministic candidate pool.

## Fixed Pilot

- Eight paired targets, six action rounds, seed `1304`.
- Model: `google/gemma-4-26b-a4b-it`, OpenRouter, non-thinking, temperature `0.7`,
  `max_tokens=128`.
- Arms:
  1. `d1_shared`: one-step exact EIG over the root `K=3` LLM candidate pool.
  2. `d2`: two-step exact EIG over that **same** root pool. For every feasible
     outcome of every root candidate it obtains a new `K=3` LLM candidate pool and
     evaluates the exact continuation EIG.
  3. `d1_matched_width`: one-step exact EIG over the union of one root pool plus
     the same number of further current-state candidate proposals as the feasible
     depth-two continuation branches. This is the predeclared matched-call width
     control. Each later width proposal is shown previously proposed IDs and instructed
     to avoid them whenever at least three fresh legal IDs remain; duplicates are still
     valid but deduplicated, and the realized union size is logged.
- Candidate requests are cached by exact history and proposal label. Thus arms share
  candidates whenever they reach the same state (in particular at all roots), while
  counterfactual depth-two branches remain genuine separately proposed pools.
- Common random numbers: all arms receive exactly the same ordered targets; all
  observations are deterministic functions of the target and selected frozen trait.

## Budget

The maximum nominal allocation is `8 * 6 * (1 + 2*3) = 336` candidate calls for
depth two, the same allocation for the width control, and 48 calls for base depth
one: 720 logical calls before cache reuse. A conservative projected spend is `$0.20`;
the OpenRouter run-level accounting cap is `$0.50`, below the protocol's `$1`
exploration ceiling. The shared ledger's project budget remains `$40`.

## Read Criterion

This pilot is promotable only when all mechanics checks pass (no parser failure,
all root candidate cells shared, and per-decision width allocation equals the
corresponding virtual depth-two allocation) **and** depth two has a positive paired
accuracy-AUC difference versus both `d1_shared` and `d1_matched_width`. Otherwise it
is logged as an exploratory non-promotion. Any conclusion remains descriptive until
a single, powered, outcome-blind confirmatory run is preregistered.

## Command

```bash
set -a; source .env; set +a
PYTHONPATH=. python scripts/nonmyopic_ucizoo_llm_pilot.py \
  --config configs/config_nonmyopic_ucizoo_llm_pilot_openrouter.yaml \
  --run-id nonmyopic-ucizoo-llm-pilot-20260714
```
