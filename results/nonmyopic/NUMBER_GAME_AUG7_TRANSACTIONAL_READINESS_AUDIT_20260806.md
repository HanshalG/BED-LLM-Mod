# Number Game Aug 7 Transactional Readiness Audit

Date: 2026-08-06 23:03 BST  
Code baseline: `23428ad8`  
Model calls: 0  
Cost: $0

## Decision

The frozen August 7 Number Game sequence is ready. This audit found no new
transactional or replay-validity defect, so the preregistered protocol and
scientific thresholds remain unchanged.

## Live preflight

The real read-only command

```bash
source .env
python scripts/number_game_budget_model_aug7_execute.py --preflight
```

returned `ready_without_paid_calls`, made zero model calls, and wrote zero
files. All paid output paths are pristine. The source/control handoff and both
predecessor hashes verify; the frozen inventories contain 128 reliability cases
per budget model and 3,584 stress cases.

Authenticated OpenRouter state was:

- total credits: `$245.000000000`;
- total usage: `$217.297890263`;
- balance: `$27.702109737`.

The reported `$30` top-up is still not visible and is not included in any
authorization. The frozen expected day cost is `$4.96` under the exact `$5.00`
London-day cap, leaving `$0.04` expected slack.

## Transaction audit

- Execution rechecks the London date, live account usage, current balance,
  model identities, prices, request bounds, source hashes, pristine paths, and
  the daily ledger. The earlier preflight is not a reusable authorization.
- Every HTTP attempt reserves a model-specific worst-case cost before dispatch:
  Qwen `$0.010`, Luna `$0.004`, and DeepSeek 0731 `$0.0015`.
- Concurrent attempts share a file-locked ledger. A retry must acquire separate
  headroom.
- Explicit nonbillable HTTP and zero-cost provider errors release reservations.
  Ambiguous transport or JSON failures retain them until account reconciliation.
- A response charged above its reserved maximum fails closed and leaves the
  reservation in place.
- Control, reliability, and stress stages reconcile local measured spend with
  cumulative provider usage using the larger value before authorizing a later
  block.
- Partial paid directories, failure artifacts, duplicate terminal artifacts,
  ledger disagreement, or replay mismatch prevent relaunch. Banked components
  are independently replayed rather than called again.
- Reliability gates always run in their frozen order when authorized. Stress is
  authorized only from terminal ledger identities and reconciled remaining
  allowance, without reading scientific efficacy values.

An external account-wide charge after this audit is therefore detected by the
fresh execution preflight or the next block's live budget check and consumes the
same `$5` allowance.

## Verification

```text
98 passed in 5.89s
```

The focused suite covered the OpenRouter adapter and reservations, Qwen control
daily execution, Aug 7 orchestration/resumption, both reliability gates, and the
selected-model stress run.

## Next action

Run the single frozen Aug 7 wrapper on August 7 London time. Do not invoke an
individual component directly and do not retry a partial or failed-closed path.
