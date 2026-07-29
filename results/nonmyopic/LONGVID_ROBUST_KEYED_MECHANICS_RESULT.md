# LongVid Robust Keyed First-Link Mechanics Result

Date executed: 2026-07-29

Status: **the frozen mechanics gate failed; the LongVid interface is closed and
the 22 reserve videos remain untouched.**

## Integrity

- Preregistration commit: `a270530`
- Passing serving-smoke commit: `94de6f9`
- Public result SHA-256:
  `5a4a6b90c04d7a6876e595bb662452ba2a0fa3ef9c2c0100bcfd3474b827a450`
- Private raw SHA-256:
  `a155969fc839c0eb4250c44afc649c8f8bcd533769df3863592d320cc9c265af`
- Model: `openai/gpt-5.4`, reasoning disabled, temperature `0`
- Accepted requests / HTTP attempts / retries: `22 / 22 / 0`
- Prompt / completion / reasoning tokens: `23,952 / 7,176 / 0`
- Cost: `$0.16752`

Every output parsed, all four paths retrieved four distinct clips, endpoint IDs
were loaded only after all immediate and final ranks were checkpointed, and no
semantic repair, reissue, extraction, or fallback was used.

## Result

| Row | Roots | Realized coverage | Immediate choice | Final choice | Final correct |
|---:|---|---|---|---|:---:|
| 319 | 0 / 9 | 1 / 2 | root 0 (74%) | root 9 (79%) | yes |
| 120 | 2 / 1 | 2 / 3 | root 2 (79%) | root 2 (80%) | no |

Aggregate mechanics:

- rankable pairs: `2/2`;
- immediate accuracy: `0/2`;
- final accuracy: `1/2`;
- correct policy changes: `1`;
- final-minus-immediate coverage: `+1` clip;
- final selected coverage: `4`, equal to the frozen random draw;
- changed support refreshes: `16/16`;
- refreshes with at least four valid anchors: `14/16`.

The full-history scorer therefore improved directionally over its matched
one-step scorer and made one correct first-link switch, but it missed the
frozen 2/2 correctness, two-correct-change, `+2` coverage, and 16/16 grounding
gates.

## Where The Simulator Diverged

On row 319, the final scorer recognized that root 9's trajectory maintained the
microphone, bedside audio gear, and instrument chain. It switched away from the
immediate root and recovered one additional necessary clip.

On row 120, root 1's generated searches recovered three necessary clips:
vintage globe, later political reference, and independence/South-Asia evidence.
Root 2 recovered only two. The final scorer nevertheless retained root 2 and
described the intermediate political reference as unresolved.

The scorer received only generated belief supports and queries, not the
simulated caption observations themselves. The row-120 histories show the
support compressor did not preserve enough of the evidence that made root 1
valuable. This is the same first-link bottleneck seen elsewhere in the project:
rollout generation can find a useful path, but the learned semantic value
representation does not reliably rank its realized endpoint value.

The two weak-grounding refreshes occurred on paths that were not selected
correctly, but removing them or weakening the grounding gate cannot rescue the
failed 2/2 endpoint.

## Decision

Per the preregistration, there is no rerun, model swap, prompt repair, row
replacement, parser change, or threshold relaxation. In particular, adding raw
captions to the scorer after observing this failure would be a post-hoc method
repair. LongVid is closed for this cycle.

The replicated zero-call structural opportunity and the single correct
LLM-native first-link switch remain descriptive. They do not authorize the
untouched 22-video reserve policy and are not a positive efficacy claim.
