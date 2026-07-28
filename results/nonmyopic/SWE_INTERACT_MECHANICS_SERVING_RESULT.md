# SWE-Interact Mechanics Serving Result

Date: 2026-07-28

## Decision

**The frozen GPT-5.4 serving gate fails closed, and its zero-call diagnostic
also fails the key scientific gate. No first-link or development run is
authorized from V1.**

Development (21 tasks), confirmation (24), and retained (21) remain sealed.

## Formal Result

- Run:
  `swe-interact-mechanics-serving-v1-20260728T041500Z`
- Released-user model: `openai/gpt-5.4`, high reasoning
- Annotator: `openai/gpt-5.4-mini`, reasoning disabled
- Requests: exact `24 + 3 = 27`
- HTTP retries: `0`
- Forced exits: `0`
- User reasoning tokens: `2,681`
- Judge reasoning tokens: `0`
- Adapter-accounted cost: `$0.14773775`
- Private raw SHA256:
  `2a59fe5cda0517c23667f8f35e64210d9fbc32a52750b1b642ce2cb50b754a77`

The semantic annotator omitted the required `R` prefix on two tasks. Every row,
key, and numeric value was otherwise present, but the preregistered grammar
forbids prefix insertion. V1 therefore has no formal aggregate and cannot be
reparsed or rerun.

## Labeled Zero-Call Diagnostic

The completed raw artifact makes the underlying scientific outcome
unambiguous. A separate diagnostic accepted an optional missing `R` prefix
without making any model call. It is explicitly not a gate rescue.

| Family | Hidden atoms | New atoms after generic checklist request | Fraction |
| --- | ---: | ---: | ---: |
| DeepSWE | 14 | 13 | .929 |
| Refactoring | 7 | 6 | .857 |
| SWE-bench Pro | 7 | 7 | 1.000 |

The frozen requirement was zero generic progress on all three tasks. Observed:
`0/3`. The generic branch did not merely provide harmless framing: responses
gave task-specific behavior, API names, lifecycle rules, platform cases, and
exact version boundaries.

Other mechanics worked:

- initial reply contained at most one atom on `3/3` tasks;
- repeated targeted roots had exact requirement-set agreement on `6/6` pairs;
- distinct intended targeted roots passed on `2/3` tasks;
- distinct intended review corrections passed on `3/3` tasks.

This localizes the failure. The simulator is stable and path-conditioned, but
its disclosure policy is not robust to a direct request for the full task. An
unrestricted agent can collapse most or all hidden uncertainty in one turn, so
the released interaction does not supply the prerequisite structure needed for
a genuine non-myopic clarification result under this simulator model.

## Scope

This closes the exact GPT-5.4 simulator/model/interface route. It does not
retroactively invalidate the zero-call source audit: the code intends gradual
disclosure, but GPT-5.4 did not realize that contract under the decisive generic
probe.

The official release run configuration names `openai/gpt-5.5`, while V1 used
GPT-5.4 through OpenRouter. A native-model replication is admissible only as a
separately preregistered route if that exact model is available. It must keep
the generic anti-extraction gate and cannot reuse V1 as positive evidence.

## Artifacts

- Formal failure:
  `results/nonmyopic/swe_interact_mechanics_serving/swe-interact-mechanics-serving-v1-20260728T041500Z/FAILURE.json`
- Diagnostic:
  `results/nonmyopic/swe_interact_mechanics_serving/swe-interact-mechanics-serving-v1-20260728T041500Z/DIAGNOSTIC.json`
- Private raw content remains untracked; only its hash is public.
- Authenticated provider balance after the run: `$9.238956594`
- OatML, Slurm, SSH, and cluster use: `0`
