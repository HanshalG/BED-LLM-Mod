# Number Game GPT-5.6 Luna Paired Efficacy32 Failure

Date completed: 2026-08-05

## Decision

**Close GPT-5.6 Luna under the frozen nonreasoning Number Game planner
interface.** The run failed closed before endpoint scoring, so it provides no
policy-efficacy estimate and cannot authorize replacing Qwen3.7 Plus.

## Failure

One planning response was not exact JSON:

```text
Expecting ',' delimiter: line 1 column 1653 (char 1652)
```

The outer executor banked one complete tree before receiving the failed
future. The accepted-request log contains 1,200 responses before termination:

- 1,185 with finish reason `stop` and 15 with `length`;
- 359,856 prompt tokens and 743,609 completion tokens;
- zero reasoning tokens;
- accepted-request cost `$0.4889055`.

The malformed response itself was not checkpointed because parsing failed
inside the tree worker before the worker returned its raw artifact. The 15
length-limited responses make truncation a plausible explanation, but the
saved public evidence does not identify the failed response's finish reason,
so this is not asserted as proven causality.

No `TREES.json` or `RESULT.json` was written, no canonical endpoint aggregate
was computed, and no subset score is permitted. The exact-10 success therefore
did not generalize to the 1,568-request scale under the unchanged strict
interface and token limit. There is no seed rerun, output-token amendment,
parser salvage, or partial-tree rescue under this protocol.

## Budget

The preceding two-model smoke cost `$0.006836192`; this failed efficacy run
cost `$0.4889055`. Conservative recorded spend for 2026-08-05 is therefore
`$0.495741692`, leaving `$4.504258308` of the `$5.00` daily allowance. The
daily guard now uses the larger of posted account usage and local accepted
cost, preventing delayed provider accounting from increasing the allowance.

## Artifacts

- public FAILURE SHA256:
  `246245f8209472a3bea4a22a24b8f9c7b202be6bd9940c4394db31c953dede40`;
- checkpointed raw-response SHA256:
  `cd86a94ed65db3cfbd2639c8b801234d31faa9fefcc1d450fe5ed06574ac13e8`;
- run directory:
  `results/nonmyopic/number_game_luna_paired_efficacy32/number-game-luna-paired-efficacy32-20260805T123000Z`.
