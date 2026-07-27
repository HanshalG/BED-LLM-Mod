# HotpotQA Shared Comparative V3 Result

## Decision

The exact HotpotQA shared-comparative V3 interface **fails closed at
serving**. The initial belief and all four root-conditioned refreshes parsed.
The next request, the unchanged one-line myopic control, violated its frozen
`ORDER|R...` grammar. Execution stopped before any complete-path ranker,
policy, or endpoint.

No response is coerced or repaired, this serving row is not rerun, and the
five-task development stage remains unopened.

## Failure

V3 replaced the aligned, fixed, and shuffled complete-path `ORDER` lines with
the new rank-per-root codec, but retained V2's standalone one-line `ORDER`
codec for the myopic control. The myopic response omitted the required
`ORDER|` shape and parsed as a four-field root sequence instead. The strict
parser raised:

```text
ValueError: order line has invalid shape
```

This is an implementation/interface oversight: V3 did not remove every use of
the delimiter-sensitive standalone order format. It is not evidence for or
against the non-myopic policy.

## Integrity

- Run: `hotpot-shared-comparative-v3-serving-20260727T220947Z`
- Serving row: `5ae67c685542996d980e7b84`
- Model: `openai/gpt-5.4`, temperature zero, no reasoning
- Accepted logical requests / HTTP attempts / retries: `6 / 6 / 0`
- Prompt / completion / reasoning tokens: `2,583 / 1,905 / 0`
- Forced exits: `0`
- Adapter-recorded cost: `$0.0350325`
- Private raw SHA-256:
  `eb2c54b2ba517455abdd8aa6e48eccaf800d15a35fe8e7d2d83bc70f9e799de4`

Aligned, fixed, shuffled, final-answer, support-coverage, and holdout stages
were never requested or computed.

## Outcome

Close exact V3 as preregistered. A scientifically unchanged successor is
admissible only if it freezes one rank-per-root codec for **all** four ranking
calls, including myopic, and uses another previously exposed confirmation row
for transport. Development and holdout remain sealed.

The authenticated OpenRouter endpoint reported `$29.549457094` remaining
afterward. There is no fixed reserve. OpenRouter only; no OatML or Slurm.
