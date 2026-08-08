# Bongard OpenWorld August 10 Final Readiness

Checked: 2026-08-08 19:24 Europe/London.

Status: **ready_without_paid_calls** on the exact pushed commit
`15f81de393b010ebd3583e62c7ccb31acaa20db1`.

This record supersedes the older current-commit readiness notes for execution
purposes. It changes no model, prompt, policy, task, seed, action, endpoint,
gate, request count, or budget.

## Exact Bound Chain

- August 10 wrapper:
  `adf0cede0c14e1ac96206461371f2f53f434f5b748327f9cf93ae0e7f521f9a5`;
- Development64 V17:
  `7564ced7755f17be13f254067f013130b4beb277313fc51de16b43611a608676`;
- Confirmation96 V14:
  `0d9c6f52ea05aa93e40bf7aa61c8ebc6323f50a6454624d6f6c49ca946d3924a`;
- naive first-link V8:
  `25db6fd3241d8ffaa4989ffaadfe6bb2bec111d7f1e3d6936dd9e39fd333d454`;
- endpoint-predictive amendment:
  `2fce4b66696d5b635f8d32c5f968a917aac20ab64305cab2728bc5f9973a55b8`;
- matched-updater integrity amendment:
  `1f0da098fbed4968a3594761194f661a0ebf49b12c477383d7c90bfa4989abf9`.

The local branch and its configured upstream both resolve to the exact commit
above. The complete Bongard regression family passed `166/166` at that commit,
and all independent development, confirmation, naive, execution, and paper
bindings passed.

## Read-Only Production Preflight

The exact production command was run with `--preflight`. It returned interface
`bongard-openworld-luna-aug10-execute-4` and status
`ready_without_paid_calls`. The wrapper, serving, mechanics, and daily-ledger
paths are all absent. It verified:

- official source protocol and the 5,124,375,111-byte image archive;
- all four mechanics tasks and 56 mechanics images;
- ten hidden-state-clean serving cases totalling 8,821,758 serialized bytes;
- the strict semantic-belief response format;
- the exact Development64 V17 manifest;
- Luna image/text input, structured output, 1.05M context, and live
  `$0.10/M` input plus `$0.60/M` output prices;
- the `$0.004` per-attempt precharge and hard account-wide `$5` daily cap.

Preflight JSON SHA-256:
`6cbf5b7d717717695b3b1e6493fac454eb037e5181a4cae82a8ef31907948fef`.

Authenticated OpenRouter totals were byte-identical before and after:

- credits: `$245.000000000`;
- usage: `$220.121013787`;
- balance: `$24.878986213`.

The newly reported `$30` remains unposted and is not counted. Model calls,
files written, candidate labels accessed, and scientific endpoints accessed
were all zero.

## Execution Boundary

On August 10, run a fresh same-day preflight and then execute exactly once only
if it again returns `ready_without_paid_calls`:

```bash
set -a
source .env
set +a
/opt/anaconda3/bin/python scripts/bongard_openworld_luna_aug10_execute.py --preflight
/opt/anaconda3/bin/python scripts/bongard_openworld_luna_aug10_execute.py
```

Any hash mismatch, non-pristine path, model-contract change, insufficient live
balance, or failed serving/mechanics gate stops the chain without rescue or
reissue.
