# SWE-Interact Native GPT-5.5 Mechanics Result

Date: 2026-07-28

## Decision

**The native-model gate fails. SWE-Interact is closed as the next non-myopic
LLM-BED environment, and no development or first-link policy run is
authorized.**

The exact released simulator model fixed V1's instruction-following problem on
the generic branch, but it exposed nearly the entire task in its initial
"short version" response on two of three source families.

## Run

- Run:
  `swe-interact-gpt55-mechanics-v2-20260728T043000Z`
- User simulator: `openai/gpt-5.5`, high reasoning
- Independent judge: `openai/gpt-5.4-mini`, reasoning disabled
- Requests: exact `24 + 3 = 27`
- HTTP retries: `0`
- Forced exits: `0`
- User reasoning tokens: `813`
- Judge reasoning tokens: `0`
- Adapter-accounted cost: `$0.21259750`
- Private raw SHA256:
  `fc574f854536856daa424e3831c1540453b498e6ea1f58fbc0377936f82fc3a3`

Every numeric keyed annotation parsed under the prospectively frozen V2
grammar. This is a formal scientific gate failure, not an interface failure.

## Results

| Family | Initial atoms disclosed | Generic new atoms | Distinct intended roots | Exact root repeats | Distinct intended reviews |
| --- | ---: | ---: | --- | ---: | --- |
| DeepSWE | 14/14 | 0 | pass | 2/2 | pass |
| Refactoring | 6/7 | 0 | pass | 1/2 | pass |
| SWE-bench Pro | 0/7 | 0 | pass | 2/2 | pass |

Passed:

- zero newly disclosed atoms after the generic checklist request on `3/3`;
- distinct intended targeted roots on `3/3`;
- distinct intended review corrections on `3/3`;
- all exact request, HTTP, parse, reasoning, forced-exit, and cost checks.

Failed:

- initial reply contains at most one atom on `1/3`, required `3/3`;
- exact repeated-root agreement on `5/6`, required `6/6`.

The generic result cannot rescue the environment. On DeepSWE and refactoring,
the initial reply had already disclosed 100% and 86% of the catalog,
respectively. The simulator was not gradually withholding requirements; there
was almost nothing left for a generic second request to reveal.

## Interpretation

SWE-Interact contains genuine semantic, workspace-conditioned feedback after
implementation. The targeted-root and review results demonstrate this clearly.
But its released planning interface does not reliably preserve uncertainty long
enough to create a non-myopic clarification problem:

- GPT-5.4 begins vaguely but dumps 86--100% of the task when asked directly;
- the exact GPT-5.5 release model dumps 86--100% on the first short request for
  two source families;
- an unrestricted agent can therefore obtain the operative task in one turn.

Restricting the initial question, hiding the user simulator until after a plan,
or forbidding broad requests could manufacture sequentiality, but those would
change the released environment. The current project needs an LLM-native result
whose enabling structure is native and load-bearing, not prompt-enforced after
observing this failure.

## Boundary And Accounting

- Development: 21 tasks, unopened
- Confirmation: 24 tasks, unopened
- Retained: 21 tasks, unopened
- Authenticated provider balance after the run: `$9.202752344`
- Reserve: none
- OatML, Slurm, SSH, or cluster use: `0`

Public result:
`results/nonmyopic/swe_interact_gpt55_mechanics/swe-interact-gpt55-mechanics-v2-20260728T043000Z/SERVING.json`.
Private prompts and replies remain untracked; only their hash is public.
