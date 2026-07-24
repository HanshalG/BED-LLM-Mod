# ClinDiag Fixed-Slot Structural Opportunity Screen Result

Date: 2026-07-24

Status: **stage 1 failed; no ordered two-step generation.**

## Frozen Gate

The necessary condition required at least two of four fresh cases to have both:

- initial truth-match score below `0.80`; and
- maximum truth-match score over all eight one-step supports below `0.80`.

The run completed exactly 40 requests with zero reasoning/retries, all 36 supports
parsed to size 12, no full target string appeared in source evidence, and no parser or
runtime failure. The scientific gate nevertheless failed: **zero of four cases**
retained two-step headroom.

## Case Results

| Case | Truth | Initial | Best one-step | Saturating action |
|---|---|---:|---:|---|
| `20220188` | POEMS syndrome | 1.00 | 1.00 | Already saturated |
| `11222813` | LHON | 0.12 | 0.95 | `family_social` |
| `rare140` | Alagille syndrome | 0.00 | 1.00 | `lab_1` |
| `rare122` | Buerger disease | 1.00 | 1.00 | Already saturated |

POEMS and Buerger disease were already named by the initial generator. The LHON
family-history chunk explicitly contained the acronym `LHON`, which the full-string
leak check did not catch. The Alagille liver panel was individually sufficient for
the generator to add the truth.

## Interpretation

The support generator is working, but these coarse retrospective chunks make truth
recovery myopically easy. This is not evidence against LLM path-dependent non-myopia
in general; it falsifies this four-case coarse-slot screen as a source of that
opportunity.

The result also sharpens the construction requirement. Full diagnosis-string checks
are insufficient because standard acronyms and highly characteristic findings can
saturate a single action. A next route must either:

1. estimate the prevalence of naturally occurring all-one-step omissions under a
   fixed, predeclared screening rule; or
2. atomize stored evidence into smaller target-independent actions and re-audit the
   common action interface.

No ordered pairs, semantic likelihoods, scorer, or policy are authorized from this
screen.

## Cost

- 36 GPT-5.4 support calls plus four GPT-5.4 Mini measurements;
- 16,106 prompt and 5,198 completion tokens;
- zero reasoning tokens;
- `$0.09775475`;
- conservative project-ledger remainder: `$23.67396972`.
