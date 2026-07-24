# ClinDiag Fixed-Slot One-Step-Omission Prevalence Result

Date: 2026-07-24

Status: **passed; exhaustive ordered-pair generation is authorized only for the seven
qualifying cases.**

## Result

The run completed exactly 200 requests with zero reasoning/retries, all 180 supports
parsed to size 12, no full target string appeared in source evidence, and no parser or
runtime failure.

Seven of 20 cases retained truth omission after every one-step action, exceeding the
frozen requirement of four:

| Case | Subset | Initial score | Best one-step score |
|---|---|---:|---:|
| `31597024` | Challenging | 0.00 | 0.00 |
| `24283228` | Challenging | 0.00 | 0.00 |
| `rare287` | Rare | 0.00 | 0.00 |
| `rare79` | Rare | 0.00 | 0.00 |
| `rare66` | Rare | 0.00 | 0.00 |
| `rare207` | Rare | 0.15 | 0.15 |
| `rare243` | Rare | 0.00 | 0.70 |

The observed all-one-step-omission prevalence is `7/20 = 35%`, versus the
preregistered minimum of 20%. Six qualifying cases are well below the `0.80`
truth-equivalence threshold; this is not driven only by borderline semantic scores.

The remaining 13 cases were saturated either initially or by one informative stored
chunk, reproducing the four-case pilot's failure mode on part of the population.

## Interpretation

The coarse eight-slot task is heterogeneous rather than universally myopic. A
nontrivial natural subset requires evidence combinations before the GPT-5.4 support
contains the truth. This is a necessary opportunity, not yet a two-step result:
ordered pairs may still fail to recover the diagnosis, and a target-blind score may
fail to select any recovering path.

No case may be substituted. The next gate is restricted to these exact seven IDs and
must validate selected pair supports with exact-prompt replays to control winner's
curse and generation noise.

## Cost

- 180 GPT-5.4 support calls plus 20 GPT-5.4 Mini measurements;
- 78,932 prompt and 27,144 completion tokens;
- zero reasoning tokens;
- `$0.49397575`;
- conservative project-ledger remainder: `$23.17999397`.
