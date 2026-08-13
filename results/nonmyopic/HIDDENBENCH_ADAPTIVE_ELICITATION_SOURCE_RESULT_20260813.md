# HiddenBench Adaptive-Elicitation Source Result

Date executed: 2026-08-13

Status: **source pass; a separately frozen four-task mechanics protocol is
authorized, with no model call or endpoint yet authorized.**

## Result

The exact official HiddenBench release at commit
`3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1` passes every frozen source gate.

| Quantity | Result |
|---|---:|
| tasks | 65 |
| tasks with 3 / 4 answer options | 59 / 6 |
| tasks with 3 / 4 private facts | 7 / 58 |
| unique task IDs / names | 65 / 65 |
| source gates passed | 7 / 7 |
| OpenRouter calls / cost | 0 / `$0` |
| OATML cluster use | 0 |

The released simulator creates one participant per private fact. In the hidden
profile, each participant receives every shared fact plus exactly one shuffled
private fact. Later participants receive earlier natural-language messages. This
is a native sequential semantic information channel, not a static answer table.

The hash-ordered split is complete and disjoint:

| Split | Tasks |
|---|---:|
| mechanics | 4 |
| opportunity | 12 |
| development | 16 |
| confirmation | 24 |
| reserve | 9 |

The public manifest contains only source bindings, aggregate histograms, and
split hashes. An independent value-level check confirms that it contains none of
the released task names, descriptions, facts, answers, or rationales.

## Scientific Boundary

This pass establishes the substrate, not the result. HiddenBench provides only
the realized private-evidence world for each task. It does not provide
option-conditioned counterfactual worlds or response likelihoods. A valid BED
descendant therefore requires the LLM to generate those semantic objects; that is
the load-bearing LLM-native component.

The next mechanics protocol must test whether this generated belief machinery
actually creates a non-myopic first-link opportunity. In particular, it must
reject the route before development unless depth two changes the first
elicitation action and improves a truth-anchored score on at least three of four
mechanics tasks while clearing semantic answer-obedience and compute-matched
myopic controls.

No HiddenBench task value, published multi-agent score, or registered correct
answer has been used as a policy endpoint. There is no efficacy claim.

## Integrity

- protocol SHA-256:
  `5340d17a11ae4654a82fff76a7e4244c39336f3b282240c05c6d0b2c88e5be28`;
- public manifest SHA-256:
  `b105c1f54e2b5ec56606b4eef2c7f464e2bdab996315f9a54a4d9ea817b6d56d`;
- source result SHA-256:
  `67b3f641bb8ac7956e33281c2bdd074402e775ac88c643f6db4c468788b73bf9`.
