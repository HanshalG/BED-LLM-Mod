# Bongard Luna August 10 Execution

Frozen: 2026-08-06, before any Bongard model request.

This operational wrapper executes the interface-v2 exact10 serving gate and
interface-v6 full four-task mechanics tree in the only authorized order. It does not change
the scientific protocol, model, prompts, seeds, task set, request tree, gates,
or budgets.

## Command

Run no earlier or later than 2026-08-10 in Europe/London:

```bash
set -a
source .env
set +a
/opt/anaconda3/bin/python scripts/bongard_openworld_luna_aug10_execute.py
```

For an unopened sequence, this command automatically runs the complete
read-only preflight before creating the wrapper directory, daily ledger, or
model adapter. It proceeds only from `ready_without_paid_calls`, initializes
the ledger from that exact authenticated credit snapshot, and then lets the
serving gate reconcile subsequent account-wide usage. Date, source, image,
manifest, endpoint, path, or balance failures leave every execution path
untouched. Existing ledgers and banked components follow the no-repeat resume
path.

The preflight also binds `BONGARD_LUNA_PRECHARGE_AMENDMENT.md`. Every Luna
HTTP attempt reserves `$0.004` before dispatch, and live pricing must keep the
3,200-token output ceiling plus at least 8,000 prompt tokens inside that bound.
Ambiguous retries retain separate reservations, so concurrency cannot consume
the same uncommitted component allowance twice.

The fixed outputs are:

- exact10:
  `results/nonmyopic/bongard_openworld_luna_vlm_serving_smoke/bongard-openworld-luna-vlm-serving-smoke-20260810`;
- mechanics:
  `results/nonmyopic/bongard_openworld_luna_vlm_mechanics_tree/bongard-openworld-luna-vlm-mechanics-tree-20260810`;
- wrapper:
  `results/nonmyopic/bongard_openworld_luna_aug10_execution/bongard-openworld-luna-aug10-20260810`;
- account ledger:
  `results/nonmyopic/openrouter_daily_budget/2026-08-10.json`.

## Fail-Closed Sequence

1. Refuse any non-August-10 London date.
2. Require the full pristine preflight and freeze its live credit snapshot.
3. Run or independently replay exact10 interface-v2.
4. Reconcile and verify the exact `$5.00` account-wide ledger.
5. Stop permanently before mechanics if exact10 is `gated_null`.
6. Run or independently replay mechanics interface-v6 only after exact10
   passes.
7. Reconcile and verify both ledger records and total daily spend.
8. Authorize development only when mechanics status is `mechanics_pass`.

The wrapper refuses stale serving interface-v1 or mechanics interface-v1/v2/v3
artifacts, dual RESULT/FAILURE files,
partial directories without a banked artifact, changed raw hashes, changed
component hashes, unreconciled ledgers, and any unrecognized status.

## Resumption

A banked exact10 result is replayed and reused; it is never regenerated. A
banked mechanics result is replayed from its exact raw root, branch, and final
responses with zero model calls. A completed wrapper verifies component hashes
before returning its result. A partial or failed-closed paid component is not
resumed or rerun in place.

Independent mechanics replay reconstructs all root scores, six policy paths,
all eight realized first-action continuations, endpoint metrics, ranking
fidelity, paired conditioned/history-blind request seeds and prompt hashes,
same-batch adjacency, gates, usage, and raw-response hash. Equality is exact
canonical JSON.

The maximum authorized component caps remain `$0.25` for exact10 and `$1.75`
for mechanics under one account-wide `$5.00` day. No additional paid tail is
authorized by this wrapper.
