# Bongard Luna Naive First-Link Baseline Implementation

Date: 2026-08-07

## Decision

Add a separate GPT-5.6 Luna medium-reasoning baseline that selects only the
first query. It receives the same four initially labelled images and eight
selectable images as the formal policy, but no endpoint image, candidate or
endpoint label, generated belief, EIG score, source metadata, or policy
choice. DeepSeek V4 Flash 0731 cannot serve this Bongard baseline because its
OpenRouter endpoint is text-only.

The baseline makes one reasoning call per task. Its selected action is scored
through the main development result's already-generated
`all_first_action_paths` entry. The realized first label, regenerated branch
belief, second-step EIG decision, terminal belief, and endpoint scorer are
therefore shared with the formal policy. This isolates first-link quality and
adds no terminal-generation calls.

## Validity Controls

- `seal_unqueried_labels` constructs each model-facing task with only the four
  initial labels materialized. Candidate and endpoint labels are absent from
  the object, not merely omitted from prompt serialization.
- Each request uses a seed-shuffled image order and strict
  `{"first_image_id": "..."}` output schema.
- Replay checks bind every request hash, response, seed, display order, and
  explicit privacy flag.
- Development blocks must run in A, B, C, D order, after the corresponding
  same-day main block. Prior baseline results, raw responses, supplemental
  ledgers, and execution records are replayed before a later block can start.
- Final analysis independently reconstructs the main combined development
  result from its four original blocks before joining baseline choices.
- The baseline cannot alter, rescue, or authorize the main development or
  confirmation result.

## Budget And Schedule

- Exact ten-request mechanics smoke: August 8, maximum `$0.20`.
- Development choices: eight requests after each main block on August 11--14,
  maximum `$0.20` per day.
- Main development cap `$4.75` plus baseline cap `$0.20` gives a complete
  worst-case daily exposure of `$4.95`, below the account-wide `$5.00` cap.
- Each request reserves `$0.008`, covering 30,848 prompt tokens plus the
  frozen 8,192-token completion at the live Luna price.

## Frozen Bindings

- Baseline manifest:
  `dfd55b7ed3577fc69431e1dd514d69d158cd7eb01376fbde447064bc0bfdcc7f`
- Main development V11 manifest:
  `a0b70ff8bbe3e36eba56b357e563504f12e4792d92cedee237d4b261d15a7708`
- Baseline core:
  `53692b0919a65c4442b0e47cd0ccd569eb5e0a922919b3802d52a9842dbac19a`
- Daily wrapper:
  `db8d13229444abbf3f549967af4e70271103b6f5abd917fcce88573ec44b95c9`

The manifest verifier passes. A real authenticated August 8 preflight
simulation reports `ready_without_paid_calls`, 30,848 covered prompt tokens,
model calls zero, and files written zero. The complete Bongard test family
passes 129 tests. No model response or scientific endpoint was opened during
implementation.
