# HotpotQA Shared Comparative V2 Result

## Decision

The exact HotpotQA shared-comparative V2 interface **fails closed at
serving**. The initial belief, four root-conditioned belief refreshes, and
myopic rank all parsed. The aligned complete-path response then violated the
frozen delimiter grammar, so execution stopped before a policy or endpoint
existed.

No response is coerced or repaired, the serving task is not rerun, and the
five-task development stage remains unopened.

## Failure

The aligned response contained the required five lines, but its first line
used spaces:

```text
ORDER R1 R2 R4 R3
```

instead of the required:

```text
ORDER|R1|R2|R4|R3
```

The parser therefore raised `ValueError: order line has invalid shape`.
Replacing spaces with delimiters or clarifying and rerunning after seeing the
response would violate the frozen strict-line gate. This is an interface
failure, not evidence for or against non-myopic first-link selection.

## Integrity

- Run: `hotpot-shared-comparative-v2-serving-20260727T220342Z`
- Model: `openai/gpt-5.4`, temperature zero, no reasoning
- Accepted logical requests / HTTP attempts / retries: `7 / 7 / 0`
- Prompt / completion / reasoning tokens: `6,898 / 2,292 / 0`
- Forced exits: `0`
- Adapter-recorded cost: `$0.051625`
- Private raw SHA-256:
  `a16de50dea45613ec0e2babf77836be084af70c818f10d3bfc9b8a33b110bf49`

The seven requests comprise one initial belief, four branch refreshes, one
myopic rank, and the malformed aligned plan. Fixed-belief, shuffled-belief,
final-answer, support-coverage, and holdout stages were never requested or
computed.

## Outcome

Close exact Hotpot shared-comparative V2 as preregistered. Development and
holdout stay sealed. The result does not weaken the underlying Hotpot
directional-unlock opportunity; it only shows that this exact natural-language
line interface was not serving-reliable on its first scientific use.

The authenticated OpenRouter endpoint reported `$29.591827094` remaining
afterward. There is no fixed reserve. OpenRouter only; no OatML or Slurm.
