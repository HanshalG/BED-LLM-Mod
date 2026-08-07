# RegretBench Aug 8 Post-Alignment Readiness Audit

Date: 2026-08-07

## Scope

This is a read-only, zero-model-call audit after the prospective first-reply
likelihood-alignment amendment. It supersedes the operational hashes in the
earlier Aug 8 readiness audit but does not rewrite that historical artifact.
It changes no prompt, model, endpoint, cohort, selection, science threshold,
seed, request count, or budget.

## Live Account Boundary

Authenticated OpenRouter cumulative values were:

- total credits: `$245.000000000`
- total usage: `$220.113606154`
- balance: `$24.886393846`

The reported top-up remains absent from `/credits` and is not spendable. The
Aug 7 ledger remains immutable at `$2.815715891` spent and `$2.184284109`
intentionally unspent under the account-wide `$5.00` Europe/London-day cap.

## Real Aug 8-Clock Preflight

An Aug 8 clock was supplied only to the read-only Python preflight. The live
OpenRouter model catalog and authenticated credit endpoint were queried
normally. The Luna naive first-link smoke returned
`ready_without_paid_calls` with:

- model `openai/gpt-5.6-luna`
- live pricing `$0.10/M` prompt and `$0.60/M` completion
- `$0.008` per-attempt reservation covering `30,848` prompt tokens after the
  frozen maximum completion allowance
- `$0.20` stage cap and `$4.80` remaining daily allowance
- zero model calls, zero files written, and exactly zero account-usage change

The smoke result directory and daily ledger were absent before and after the
preflight. RegretBench support recovery then stopped before catalog, credits,
or dispatch because the verified Luna predecessor is intentionally absent.
The policy stage likewise stopped before live access because the verified
support-development predecessor is absent. These are the registered one-way
dependency states.

## Post-Alignment Bindings

- execution commit: `82daa1de`
- Luna naive core:
  `53692b0919a65c4442b0e47cd0ccd569eb5e0a922919b3802d52a9842dbac19a`
- Luna naive daily wrapper:
  `db8d13229444abbf3f549967af4e70271103b6f5abd917fcce88573ec44b95c9`
- RegretBench support core:
  `7e227e4d3a125b817dd45c31ce6b1fc94c24bae6ee982ce59f2a9082065752c2`
- RegretBench support daily wrapper:
  `b05387325ea8e8cac3852c7e9393e96662aa1bee2cfdc76cf7be1f22146a29d6`
- RegretBench policy core:
  `903d3afab63a48aa133d81e7af9defcff10e065e1297930be2cea9096d4ebb33`
- RegretBench policy daily wrapper:
  `ef8874760b1183dbab33534b6c942474d2f0a5c570aac3266355d6d40d585f03`
- independent result verifier:
  `0cff99b50701930e608cafde47de465548777af1862426238bfaf7584727c14a`
- first-reply alignment amendment:
  `6102056866f7e0fc8b5cf6d95de6080ea699505a294f7a2a540d303499d78c44`

The amendment requires all three enriched-smoke realized first replies to
match truth-consistent initial hypotheses and at least `40/64` such matches
for every formal primary policy. Unsupported first actions do not count. The
independent verifier reconstructs the gate from raw initial support and the
official answer mapping; the gate can reject but cannot rescue or reclassify.

## Verification

- current RegretBench suite: `84/84` passing
- combined baseline/executor/mechanics subset: `59/59` passing
- confirmation supplemental protocol verification: all gates passing
- live preflight model calls: `0`
- live preflight usage change: `$0.000000000`
- paid artifacts opened: `0`

## Execution Rule

On Aug 8, source `.env` with variables exported, rerun every wrapper with
`--preflight` immediately before execution, and preserve this exact order:

1. Luna naive first-link exact smoke.
2. RegretBench support smoke and development only after a verified Luna pass.
3. RegretBench policy smokes and development only after a literal,
   independently verified support-development pass.
4. Independent result replay, frozen report, and deterministic paper fragment
   for every complete policy result, including a null or mechanics failure.

The complete worst-case authorization remains `$4.80` inside the account-wide
`$5.00` day. No missing predecessor, null, or failed gate may be bypassed, and
no call may be made merely to use the remaining allowance.
