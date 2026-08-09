# Bongard OpenWorld August 10 Current-State Readiness

Checked: 2026-08-09 10:08 Europe/London.

Status: **ready_without_paid_calls** from pushed commit
`39ae76645b2e4c0e7a1b4625398d28bbc80b6dca`.

This is a read-only readiness refresh. It changes no model, prompt, policy,
task, seed, action, endpoint, gate, request count, or budget, and it does not
authorize execution before August 10.

## Exact Chain

- August 10 wrapper: `adf0cede0c14e1ac96206461371f2f53f434f5b748327f9cf93ae0e7f521f9a5`;
- Development64 V17: `7564ced7755f17be13f254067f013130b4beb277313fc51de16b43611a608676`;
- Confirmation96 V14: `0d9c6f52ea05aa93e40bf7aa61c8ebc6323f50a6454624d6f6c49ca946d3924a`;
- naive V8: `25db6fd3241d8ffaa4989ffaadfe6bb2bec111d7f1e3d6936dd9e39fd333d454`.

The branch and its upstream resolved to the checked commit. The complete
Bongard regression family passed `205/205` there.

Every independently opened endpoint result must still use the frozen
classical-suite outcome adapter (`8c2bc93b...`) and paper wrapper
(`c9d74f0e...`). Those bind DINO plans `58154f05...`, SigLIP plans
`a41cc3b0...`, and the held-out SigLIP calibration audit `f4f05e02...`.
Neither classical plan bank authorizes an endpoint.

## Read-Only Preflight

The exact production preflight returned interface
`bongard-openworld-luna-aug10-execute-4` and status
`ready_without_paid_calls`. Wrapper, serving, mechanics, and August 10 ledger
paths are absent. It verified four mechanics tasks, 56 images, ten serving
cases, the frozen image archive and belief schema, and Development V17.

OpenRouter still exposes exact `openai/gpt-5.6-luna` with image and text input,
strict structured output, 1.05M context, and 128K maximum completion. Live
prices remain `$0.10/M` input and `$0.60/M` output. The `$0.004` attempt reserve
covers the maximum output plus 20,800 prompt tokens.

Authenticated totals were byte-identical before and after:

- credits: `$245.000000000`;
- usage: `$220.121013787`;
- balance: `$24.878986213`.

The reported `$30` top-up remains unposted and is not counted. Model calls,
files written, candidate labels accessed, scientific endpoints accessed, and
cost were all zero.

## Execution Boundary

On August 10, run the fresh same-day preflight and execute exactly once only if
it again returns `ready_without_paid_calls`:

```bash
set -a
source .env
set +a
/opt/anaconda3/bin/python scripts/bongard_openworld_luna_aug10_execute.py --preflight
/opt/anaconda3/bin/python scripts/bongard_openworld_luna_aug10_execute.py
```

Any hash mismatch, non-pristine path, model-contract change, insufficient live
balance, or failed serving/mechanics gate stops the chain without rescue.
