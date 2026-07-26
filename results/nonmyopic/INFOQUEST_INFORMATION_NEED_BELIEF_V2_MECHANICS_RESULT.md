# InfoQuest Information-Need Belief V2 Mechanics Result

## Status

The preregistered 30-cell mechanics run completed exactly, but the scientific
gates failed. The V2 linear expected-resolved-mass route is closed.

- run ID:
  `infoquest-information-need-v2-mechanics-20260726T042225Z`;
- public artifact SHA-256:
  `e38980e38254a3d1b1c19865cf3bde11c7cccfa43150c38973c2978f098b9312`;
- private raw SHA-256:
  `a12153fcdee1d532d843274b2eb1998a959b60706a6e1beb1293b378fe91d530`.

## Accounting

The run made exactly 30 physical requests and 30 HTTP attempts to
`openai/gpt-5.4`, all non-reasoning at temperature zero. Every strict response
parsed and all 30 cells had nonconstant scores. There were zero retries,
reasoning tokens, and forced exits. Cost was `$0.0744075`, below the frozen
`$0.35` cap.

There were zero simulator and checklist-judge calls. The compiler did not
receive hidden settings, checklist content, target labels, old scenario
hypotheses, prior scores, or prior choices. No response was repaired or
reissued.

## Scientific Result

Mean within-cell score/target-gain Spearman correlation was `-.1238`, below the
frozen `.20` gate. The information-need scorer selected `.4333` target bits per
cell, compared with `.4000` for frozen fixed support and `.3000` for V3 dynamic
support. Its gain over fixed was only `.0333`, below the required `.15`.

The scorer was target-optimal in 11/30 cells, below the required 12. It improved
over fixed in 3/6 fixtures, below the required four. Pairwise target gain versus
fixed was 5/21/4 wins/ties/losses, so that one directional gate passed, but the
conjunctive result failed.

## Interpretation

Generating target-relevant unresolved needs is directionally better than the
old scenario-support entropy choice, but linear averaging of predicted
resolution across all needs does not rank actions reliably. The exact V2
objective remains anti-aligned at the all-action level and does not support a
policy or non-myopic claim.

A subsequent zero-call scoring decomposition is post hoc development work only
and cannot rescue this result. Any alternative aggregation requires a
prospectively frozen confirmation on fresh records.
