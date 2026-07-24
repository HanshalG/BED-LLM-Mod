# ClinDiag Fixed-Slot Ordered-Pair Recovery Result

Date: 2026-07-24

Status: **failed; the exact coarse eight-slot support-refresh line is closed.**

## Frozen Gate Result

The run completed exactly 434 requests with zero reasoning/retries. All 392 ordered
pair supports and seven exact-prompt replays parsed to size 12. Replay prompts matched
their selected original prompts exactly, and no full target string appeared in source
evidence.

The scientific gates failed decisively:

- validated unlocks: `0/7` versus required `>=3`;
- mean validated gain: `-0.1214` versus required `>=0.20`.

| Case | Best one-step | Best pair, batch score | Replay-validated score | Gain |
|---|---:|---:|---:|---:|
| `31597024` | 0.00 | 0.92 | 0.00 | 0.00 |
| `24283228` | 0.00 | 0.00 | 0.00 | 0.00 |
| `rare287` | 0.00 | 1.00 | 0.00 | 0.00 |
| `rare79` | 0.00 | 0.22 | 0.00 | 0.00 |
| `rare66` | 0.00 | 0.00 | 0.00 | 0.00 |
| `rare207` | 0.15 | 0.25 | 0.00 | -0.15 |
| `rare243` | 0.70 | 0.78 | 0.00 | -0.70 |

## Mechanism Audit

The replay control caught two distinct winner failures:

- The selected adenovirus pair received a batch score of `0.92`, but its support did
  not contain adenovirus; the fresh joint validation correctly scored both original
  and replay at zero.
- Nine BPD pairs contained bronchopulmonary dysplasia in the first generation, but
  the selected exact-prompt replay dropped it. The original/replay score gap was
  `1.0`, so this was unstable generation rather than a validated unlock.

The remaining cases never reached a truth-equivalent pair. Hurler-Scheie reached the
broad MPS-I/Hurler parent but not the intermediate phenotype required by the frozen
equivalence rule.

Across many paths the generated differential changed little after new evidence. The
current refresh prompt explicitly asks the model to retain plausible earlier
diagnoses, which likely contributes to this support inertia. That is a useful
diagnostic, but it cannot repair or relabel this null.

## Decision

No semantic likelihood model, scorer, or policy follows from this run. The exact
single-list, prior-retaining coarse-slot interface is closed.

A distinct future line may preregister a genuinely de-anchored full-refresh prompt
that treats the previous support as context rather than an inclusion constraint. It
must start with a new serving/stability gate and fresh cases; the seven failed pair
cases remain diagnostic only.

## Cost

- 399 GPT-5.4 generation calls and 35 GPT-5.4 Mini measurements;
- 223,156 prompt and 69,573 completion tokens;
- zero reasoning tokens;
- `$1.30165225`;
- conservative project-ledger remainder: `$21.87834172`.
