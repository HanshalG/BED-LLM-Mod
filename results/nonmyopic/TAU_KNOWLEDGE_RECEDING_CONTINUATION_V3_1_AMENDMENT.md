# tau-Knowledge Receding Continuation V3.1 Amendment

## Status

This is a transparent format-only amendment frozen before any response or
endpoint from the untouched 20-task confirmation split is generated.

It supersedes V3's statement that there would be no V4 and its requirement that
the strict development parser pass before confirmation. That deviation must be
reported with any result.

## Reason

V3 generated all four requested score fields on all 100 development roots.
Twenty-nine responses represented a valid single-digit band value with a
leading zero, such as `"02"` instead of the preregistered canonical `"2"`.
There were no missing fields, invalid count bands, retries, reasoning tokens, or
runtime failures.

The mapping is deterministic and does not alter any score or ranking:

- `"0"` and `"00"` map to integer 0;
- `"2"` and `"02"` map to integer 2;
- all values must still be one- or two-character digit strings in the frozen
  bands 0-9, 30-39, 60-69, or 90-99.

JSON numbers, signs, decimals, more than two digits, extra keys, and values
between bands remain invalid.

## Scientific Protocol

No semantic or experimental component changes:

- the V3 system and user messages are byte-for-byte identical;
- GPT-5.4, temperature zero, and disabled reasoning are unchanged;
- generated openings, five roots, four continuation branches, BM25 top-3
  retrieval, initial and refreshed path-dependent beliefs, and fallible-refresh
  instruction are unchanged;
- myopic and full-tree root scorers, joint selector, seeded random control,
  exact required-document endpoint, and original-order tie breaking are
  unchanged;
- no development request is rerun or replaced.

The existing exact 10-call public V3 smoke is replayed through the relaxed parser
without new calls. The paid stage is the original untouched 20-task confirmation
split selected with seed `24337`, for exactly 280 physical calls.

## Frozen Confirmation Gates

All conditions must hold:

- all 20 cases and 100 focused root scores complete;
- exactly 280 physical requests and zero reasoning tokens;
- at least 50 endpoint-distinct root pairs;
- non-myopic root pairwise accuracy at least 0.60 and at least 0.05 above
  myopic;
- at least 200 endpoint-distinct continuation pairs;
- focused continuation accuracy at least 0.60, optimal rate at least 0.70,
  mean regret at most 0.30, and selected-root loss at most 5;
- non-myopic receding versus myopic: at least 4 wins, at most 2 losses, and
  total gain at least 4 documents;
- versus seeded random: at least 6 wins, at most 4 losses, and total gain at
  least 5 documents;
- focused non-myopic gains at least 4 documents over the original joint
  continuation selector.

No confirmation response, task, branch, score, parser rule, threshold, or
endpoint may be repaired, replaced, or changed.

## Interpretation

Because the amendment and decision to release confirmation were informed by V3
development responses, the final study is a held-out test after iterative
development, not a pristine one-shot preregistration. The confirmation tasks and
endpoints remain untouched, so a clean paired result is still meaningful if the
frozen test gates pass.

## Budget

Live balance is `$54.420058`, leaving `$29.420058` above the protected `$25`
Monday reserve. Confirmation is projected at `$3.20` with a hard cap of `$8`.
OatML remains paused.
