# Battleship Executable Candidate-Bank Gate Result

Date: 2026-07-29

Status: **gate failed; branch-conditioned mechanics not authorized**.

## Frozen Run

- preregistration commit:
  `96acf51`
- run:
  `battleship-executable-bank-20260729T025133Z`
- public gate SHA-256:
  `c4fde4dd3e161d272f7b552ae1a64d664eff8d4f9ac190eb62068139a3fb7176`
- private raw SHA-256:
  `0ef35f540b111157263a9af8c91d993fc5f2f789f65129221e6e186ca718a7af`
- accepted requests / HTTP attempts: `10 / 10`
- retries / provider-error retries: `0 / 0`
- reasoning tokens / forced exits: `0 / 0`
- prompt / completion tokens: `5,990 / 3,283`
- cost: `$0.06422`

All ten strict schemas parsed. No response was repaired, reissued, normalized,
or substituted. No target board, failed response, policy endpoint, or branch
mechanics was accessed.

## Candidate Bank

GPT-5.4 produced 60 raw paired question/predicate candidates:

- 49 safely executed as Boolean and had Yes prevalence in `[.05,.95]` on
  both fresh 4,096-board blocks;
- every call contributed 4--6 valid candidates, above the frozen minimum of
  three;
- behavior deduplication retained 19 unique candidates, one below the frozen
  minimum of 20; and
- all 19 unique behaviors were novel relative to the released 39-program
  stage-zero bank, far above the frozen minimum of eight.

The unique questions include local occupancy, ship orientation, ship-region
counts, checkerboard parity, relative ship position, border contact, and
region-level ship-ID counts. The operational generator therefore clears the
safe-execution, balance, and behavioral-novelty problems of the failed
cross-translation interface. Its remaining diversity issue is convergence:
49 valid candidates collapse to 19 behaviors across ten independent calls.

## Planning Result

| Block | d1 three-question | d2 three-question | d3 three-question | Greedy-EIG three-question |
|---|---:|---:|---:|---:|
| seed `40010` | `.590886` | `.590886` | `.590886` | `.371999` |
| seed `40011` | `.579900` | `.579900` | `.579900` | `.363185` |

The d1 and d2 root behaviors differ on both blocks:

- d1: occupancy of the four cells `C3,C4,D3,D4`;
- d2: occupancy of the `B2:C3` 2x2 block.

The frozen d2-over-d1 endpoint gain nevertheless fails exactly: receding
horizon one replans after each answer and reaches the same three-question
utility as horizons two and three on both independent blocks. This is a true
non-myopic null for the generated bank, not sampling noise around the `.01`
threshold.

The task-utility policy strongly beats the immediate configuration-EIG
control by `.218887` and `.216715`. Greedy EIG selects checkerboard parity,
which is nearly perfectly balanced and has about `.531` bits of immediate
configuration information but is poorly aligned with the next-shot hit
endpoint.

## Interpretation

The fresh operational interface succeeds at the LLM-native measurement-model
link: it generates many safe, balanced, novel executable experiments without
hidden-target leakage or translation retries. But the richer bank removes the
depth advantage seen in the smaller released bank. Once one-step planning is
aligned to the actual task utility and replans after every observation,
lookahead does not improve the three-question policy.

This sharpens the project diagnosis:

1. Configuration EIG is badly misaligned with resolution utility.
2. LLM-generated executable measurements can be robust and genuinely novel.
3. Non-myopic gain still requires delayed complementarity that a rich static
   candidate bank does not provide.

Do not lower the unique-behavior threshold, switch from receding evaluation to
root-only values, change the bank order, or run branch-conditioned generation.
The exact interface is closed by the frozen conjunction. A future environment
must make action availability or semantic support genuinely path-dependent,
so one-step replanning cannot recover the same policy.
