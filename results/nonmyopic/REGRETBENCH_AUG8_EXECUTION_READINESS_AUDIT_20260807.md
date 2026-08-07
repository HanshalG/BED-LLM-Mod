# RegretBench Aug 8 Execution Readiness Audit

Date: 2026-08-07

## Scope

This is a read-only, zero-model-call audit of the frozen Aug 8 dependency
chain. It changes no prompt, model, endpoint, threshold, seed, cohort, budget,
or claim rule.

## Live Account Boundary

Authenticated OpenRouter cumulative values were:

- total credits: `$245.000000000`
- total usage: `$220.113606154`
- balance: `$24.886393846`

The reported top-up has not posted to `/credits` and is excluded. The Aug 7
ledger replays exactly with opening usage `$217.297890263`, spend
`$2.815715891`, and intentionally unspent allowance `$2.184284109`.

## Aug 8 Readiness

An Aug 8 clock was supplied only to the read-only Python preflight; live
credits and the OpenRouter model catalog were queried normally.

The Luna naive first-link smoke returned `ready_without_paid_calls` with:

- model `openai/gpt-5.6-luna`
- live pricing `$0.10/M` prompt and `$0.60/M` completion
- `$0.008` per-attempt reservation covering `30,848` prompt tokens after the
  complete frozen output allowance
- `$0.20` stage cap and `$4.80` remaining daily allowance
- zero model calls, zero files written, and every paid output path pristine

The RegretBench support preflight then stopped before catalog or dispatch
because the Luna smoke result is intentionally absent. This is the registered
dependency behavior, not a failure. It may be rerun only after the exact smoke
banks a verified pass.

## Exact Bindings

- Luna naive protocol manifest: `dfd55b7ed3577fc69431e1dd514d69d158cd7eb01376fbde447064bc0bfdcc7f`
- Bongard development manifest: `a0b70ff8bbe3e36eba56b357e563504f12e4792d92cedee237d4b261d15a7708`
- Luna naive core: `53692b0919a65c4442b0e47cd0ccd569eb5e0a922919b3802d52a9842dbac19a`
- Luna naive daily wrapper: `db8d13229444abbf3f549967af4e70271103b6f5abd917fcce88573ec44b95c9`
- RegretBench support core: `7e227e4d3a125b817dd45c31ce6b1fc94c24bae6ee982ce59f2a9082065752c2`
- RegretBench support daily wrapper: `eaf6274f6a048ca645bb3536296d46a87b4848e5c35efac998097747f525c9db`
- RegretBench policy core: `19970a2b1a488bc9c1b9b461b0de09d07e24c4a53dbd221d5c9809b0c8cbb390`
- RegretBench policy daily wrapper: `9134f6166c13be936d27a88a214d7ffa84122054f855196c1a54e673a993b13e`
- Independent result verifier: `ad5f516b473cddd4b4957f4921752c85e56936e28517fff1a10a9620500eaa5d`

The baseline/support/policy daily state-machine tests pass `25/25`. The active
daily executor remains bound to pushed commit `4ea21c54` and to the frozen
report and paper-fragment hashes.

## Execution Rule

On Aug 8, source `.env`, run every wrapper with `--preflight` immediately before
execution, and never bypass a waiting or null state. The order is:

1. Luna naive first-link exact smoke.
2. RegretBench support smoke and development, only after a verified Luna pass.
3. RegretBench policy smokes and development, only after a literal independently
   verified support-development pass.
4. Frozen report and deterministic paper fragment for every complete verified
   policy result, including null or mechanics failure.

The complete worst-case authorization remains `$4.80` inside the account-wide
`$5.00` Europe/London day. No call may be made merely to consume unused
allowance.
