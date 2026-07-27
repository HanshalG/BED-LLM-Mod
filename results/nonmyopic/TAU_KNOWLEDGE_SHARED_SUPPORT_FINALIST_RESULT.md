# tau-Knowledge Shared-Support Finalist Result

## Decision

The exact shared-support V1 interface **fails closed at serving**. All eight
logical calls completed, but two responses emitted four `BEST` lines rather
than the required single best-followup line. The strict parser therefore
stopped before any support score, finalist decision, or endpoint analysis.

No response is coerced and the 52-call development stage is not run.

## Failure

The ambiguous phrase

```text
first `BEST|F1` through `BEST|F4`
```

was intended to mean exactly one line whose value is in that range. Six
responses followed that interpretation. Two instead emitted:

```text
BEST|F1
BEST|F2
BEST|F3
BEST|F4
```

followed by a complete hypothesis mapping. This is an interface failure rather
than a scientific result. Accepting the first line, choosing a best line
post hoc, or rerunning with clarified wording would violate the frozen gate.

## Integrity

- Run: `tau-knowledge-shared-support-serving-20260728T014500Z`
- Model: `openai/gpt-5.4`, temperature zero, no reasoning
- Logical requests / HTTP attempts / retries: `8 / 8 / 0`
- Prompt / completion / reasoning tokens: `22,543 / 714 / 0`
- Forced exits: `0`
- Cost: `$0.0670675`
- Private raw SHA-256:
  `fd95e4347ac5d92b7c696d2b75f1e28151152363ac2ede505ec5dce580a15164`

There was no truncation and every response otherwise contained hypothesis
labels. The exact V1 method remains scientifically unmeasured.

## Decision

Close shared-support finalist scoring on tau as preregistered. Do not add a
parser exception, clarify and rerun the same tasks, or open the development
stage. The stronger surviving tau evidence remains the frozen balanced-duel
directional result (`+7`, zero losses), which did not pass its decision-coverage
gate and therefore does not authorize fresh tasks.

Authenticated balance after this run is approximately `$29.601082094`. There
is no fixed reserve. OpenRouter only; no OatML or Slurm.
