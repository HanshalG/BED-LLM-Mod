# ClinDiag Fixed-Slot Ordered-Pair Recovery Screen

Date: 2026-07-24

Status: **preregistered before any ordered-pair response.**

## Cases And Input

The input is the passing prevalence artifact with frozen SHA256
`a66bbef5f9c8619785e32cb12ad711be61ab97627ac42a52e20072fbd5375b74`.
Only its seven mechanically qualifying all-one-step-omission cases are eligible:

`31597024`, `24283228`, `rare287`, `rare79`, `rare66`, `rare207`, and
`rare243`.

No case substitution is allowed. Their initial, one-step supports, stored evidence,
and best one-step truth scores are reused exactly from the pinned artifact.

## Pair Generation

For each case, GPT-5.4 non-reasoning generates a terminal 12-diagnosis support for
all 56 ordered pairs of distinct actions. A pair `a -> b` receives:

- the frozen one-step support after `a`;
- the ordered stored observations for `a` and `b`;
- no truth, final diagnosis, title, or answer options.

GPT-5.4 Mini non-reasoning measures truth equivalence in fixed chunks of 14 supports.
The highest-scoring pair is selected with the frozen action order breaking ties.

## Winner Validation

For each case, the selected terminal prompt is replayed exactly through an independent
GPT-5.4 adapter. One fresh GPT-5.4 Mini call jointly remeasures truth equivalence for
the original and replay and measures semantic support overlap in both directions.

A case is a `validated_unlock` only when:

1. the worse original/replay truth score is at least `0.80`;
2. that worse score improves at least `0.30` over the frozen best one-step score;
3. the worse directional semantic overlap is at least `0.80`;
4. the original/replay truth-score gap is at most `0.05`.

This replay gate controls selection noise from taking the best of 56 generated paths.

## Frozen Gate

All conditions must pass:

- exactly 434 physical requests;
- zero reasoning tokens and zero retries;
- all 392 pair supports and seven replays parse to size 12;
- all selected replay prompts exactly equal their original prompts;
- no full target string occurs in source evidence;
- at least three of seven cases are validated unlocks;
- mean validated truth-score gain across all seven cases is at least `0.20`;
- no parser or runtime failure.

The run uses GPT-5.4 generation at temperature `0.5` and GPT-5.4 Mini measurement at
`0.0`. The OpenRouter run ceiling is `$3.00`; the projected ledger reservation is
`$2.20`; live provider credits and the stricter project ledger are checked before
launch.

Passing authorizes a separate semantic branch-likelihood gate. Failure closes the
coarse eight-slot line with no likelihood model, target-blind scorer, or policy.
