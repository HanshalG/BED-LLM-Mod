# LHAW Structural Audit

Date: 2026-07-25

## Decision

**Do not spend OpenRouter credit on LHAW for the non-myopic BED claim.**

LHAW is a strong benchmark for deciding whether to clarify before executing a
long-horizon workflow, but its released clarification layer does not supply the
answer-contingent action structure needed for this project's sequential BED
question.

## Pinned Sources

- Dataset: `ScaleAI/lhaw`
- Hugging Face revision:
  `71560455d3df5bca03f2a14ee7a67f31c0ae138f`
- `test.json` SHA-256:
  `f67973650afd425cb0344c8b078a1c911cde0fcacb4005476c3cd520091a6ed5`
- Official code repository: `scaleapi/lhaw`
- Audited code commit:
  `cb433b6a1c5c660c96628eaa3edbd01465e86609`
- Paper: arXiv `2602.10525v2`

The audit downloaded and inspected public sources only. It made no model calls
and did not access OatML resources.

## Structural Findings

The release contains 285 variants:

| Property | Count |
|---|---:|
| One removed segment | 123 |
| Two removed segments | 162 |
| More than two removed segments | 0 |
| Outcome-critical variants | 121 |
| Divergent variants | 94 |
| Benign variants | 70 |

Of the 162 two-segment variants, 161 provide two distinct
`expected_questions` groups, one for each removed segment. The remaining
variant provides one group. Every question group is nonempty.

The official `ask_user` implementation gives the simulator the complete and
underspecified prompts plus every removed value. Its system prompt directs the
simulator to return the exact missing value requested. Therefore:

1. A one-segment task is exhausted by one direct clarification.
2. A two-segment task is exhausted by two direct clarifications.
3. With a one-question budget, selection is a one-step problem.
4. With a two-question budget, asking one segment and then the other exhausts
   the released hidden state, so question order has no intrinsic terminal
   advantage.
5. An unrestricted compound question can request both values in one call,
   creating a further action-granularity loophole.

Seventy-five outcome-critical variants remove two segments. Forty-nine of
those original prompts contain a broad conditional or dataflow marker such as
`if`, `otherwise`, `after`, or `based on`. Manual inspection confirms that some
underlying workflows are sequential. However, the released clarification
state still contains one fixed value per removed segment, and the simulator
can reveal either value directly. The long execution horizon does not induce a
long information-acquisition horizon.

The `terminal_states` field records empirical workflow checkpoint outcomes
used to classify underspecification. It is not a collection of alternative
hidden user worlds or an answer-contingent clarification graph.

## Why Not Repair It

Generating alternative values, forbidding compound questions, or hiding one
segment until another is answered could manufacture a depth-two environment,
but those transitions would be project-authored rather than supplied or
validated by LHAW. A positive result would then rest on the repair rather than
the benchmark.

The next environment must instead provide:

- at least three externally defined hidden facts or states, or a genuine
  answer-contingent clarification graph;
- an external truth and deterministic or benchmark-defined responses;
- a finite interaction budget under which first actions change the useful
  second-action set; and
- semantic hypothesis/action generation that remains irreducibly LLM-owned.

LHAW remains relevant literature and a possible supporting benchmark for
myopic clarification efficiency, but this exact non-myopic route is closed.
