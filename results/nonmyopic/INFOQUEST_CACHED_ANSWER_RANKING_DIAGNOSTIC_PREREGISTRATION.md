# InfoQuest Cached-Answer Ranking Diagnostic Preregistration

Frozen after viewing the V3 mechanics endpoints and before any diagnostic judge
response.

## Status and Purpose

This is an explicitly post hoc, disclosed-world development diagnostic. V3
showed no useful relationship between predicted semantic EIG and realized
checklist gain, but 18/30 dynamic/fixed cells chose the same follow-up while
independent Gemini calls still introduced nonzero paired endpoint differences.
The diagnostic removes that second-link simulator noise to isolate question
ranking.

It binds:

- V3 public SHA-256
  `0cfbe3d1590001af508d351d131d12d747f2fcadf0448e2e7868809f40d968f7`;
- V3 private raw SHA-256
  `134a9659dc5f86df3dba6e18fdc8c72a79f3524b166d2cd52d91a7fe88747e1e`;
- common-history public SHA-256
  `140f77447da9bd3d83e8a7be7fd45dc8fc792422d4e1cef398f6d8460d13ccc3`;
- common-history private SHA-256
  `c4dec013386083afb4279754f4ffb1b0a2f6559bb2a2cf1863d80a849fe6e93e`.

## Frozen Replay

V3's 30 dynamic and 30 fixed compact partitions are parsed unchanged and their
exact-EIG choices are not regenerated. Every chosen action is one of the four
remaining initial roots. Its answer is replaced by that root's already-cached
matched-world answer from the common-history bank.

Thus identical dynamic/fixed actions receive byte-identical questions and
answers. A fresh GPT-5.4 Mini non-reasoning checklist call scores all five cells
of each fixture jointly. The prompt requires identical paths to receive
identical bit strings, and any violation fails closed before aggregate metrics.
There are zero new support, partition, question-choice, or simulator calls.

## Gates and Budget

One synthetic checklist call must first pass exact accounting, parsing, and the
identical-path invariant under a `$0.02` cap. Only that pass authorizes exactly
six fixture checklist calls under a `$0.05` cap.

Mechanics requires exact six requests/HTTP attempts, zero
retry/reasoning/forced exits, exactly 18 identical paths, and every identical
path scored identically. The same twelve V3 scientific gates are then reported
without threshold changes.

This diagnostic cannot support a confirmatory or paper headline claim
regardless of outcome. It only determines whether fresh simulator variance
explains V3's adverse ranking result.

The pre-Monday operational allowance is `$2.08896445`; maximum diagnostic cost
is `$0.07`. OatML jobs: `0`.
