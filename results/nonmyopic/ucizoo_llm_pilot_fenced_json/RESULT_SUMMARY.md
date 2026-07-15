# UCI Zoo LLM Candidate-Proposal Pilot: Result

**Status: completed exploratory non-promotion.** This eight-target pilot is not
confirmatory evidence and does not authorize a powered run.

## Contract and Mechanics

- The latent target, answerer, posterior filtering, EIG calculation, and deterministic
  MAP decoder were exact functions of the frozen UCI Zoo matrix. The LLM only proposed
  legal candidate trait IDs.
- All initial candidate cells were shared across arms, and the width arm matched its
  own virtual two-step candidate-call allocation at every decision.
- The run made 495 OpenRouter requests for `$0.01758521` (147,402 prompt and 8,104
  completion tokens; no reasoning tokens or forced exits). It stayed below the `$0.50`
  per-run and `$1` exploratory caps.
- One raw response proposed the out-of-catalog ID `mammal` at a depth-two candidate
  cell. The configured retry returned a valid legal pool, so no illegal action was
  deployed. Nevertheless, the preregistered mechanics read requires no parser failure;
  the raw invalid-response count is one and this gate is false.

## Exploratory Read

| Arm | MAP accuracy AUC | Final MAP accuracy | Final entropy | Mean unique pool | Logical calls / decision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared-candidate d1 | 0.0417 | 0.1250 | 1.3170 | 3.00 | 1.00 |
| d2 | 0.0625 | 0.1250 | 1.2617 | 3.00 | 5.31 |
| Matched-call widened d1 | 0.0417 | 0.1250 | 1.2207 | 13.12 | 5.48 |

The paired d2 minus d1 MAP-AUC difference is `+0.0208` against both controls. It is
one win, seven ties, and zero losses; the only difference is an earlier MAP decode for
one target. The descriptive bootstrap interval is `[0.0000, +0.0625]` for both
comparisons. Final MAP accuracy has no difference. d2 improves final entropy over
shared d1 but is worse than the matched-width arm (`1.2617` versus `1.2207`).

## Verdict

The directional MAP-AUC pattern is interesting but inadequate: the mechanics gate is
false, the signal is one of eight targets, final MAP is tied, and the entropy read
favors matched width. The registered output correctly sets `promotable: false`; no
confirmatory run follows from this pilot. The full machine-readable traces and raw
candidate responses are in `PILOT.json`.
