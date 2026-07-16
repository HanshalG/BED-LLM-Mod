# Rock Diagnosis LLM Confirmation Analysis

## Verdict

**The preregistered 30-pair `3-6` Rock Diagnosis LLM confirmation passes.** This is
a positive result for two-step exact EIG under an LLM-restricted candidate interface,
not yet the final multi-environment claim.

| Paired comparison | Final entropy reduction (nats) | 95% bootstrap CI | W / T / L |
| --- | ---: | --- | --- |
| d2 - shared d1 | +0.2009 | [+0.0740, +0.3330] | 19 / 0 / 11 |
| d2 - call-matched width | +0.1903 | [+0.0666, +0.3167] | 19 / 0 / 11 |

Both lower bounds are strictly positive and all preregistered mechanics checks pass:
the d1 and d2 root cells were shared, every selected action was legal, and width used
the same logical candidate-call allocation as d2's virtual root tree.

## Mechanism

The root candidate cells contained both an enabling movement and immediate checks. On
all 30 paired trajectories, d2 chose `move-EAST`; shared d1 and call-matched width
each chose `check-1` on all 30. Thus the matched-width control had the same proposal
budget but did not buy the future sensing geometry that the two-step value function
recognized. D2 subsequently had lower entropy AUC by `0.0990` nats versus shared d1
and `0.0522` versus width, plus higher final true-vector log posterior by `0.1429` and
`0.0992` nats respectively.

The endpoint MAP accuracy does **not** improve: d2 and shared d1 both end at `0.200`,
while width ends at `0.267`. The claim is therefore deliberately about the exact
posterior-information objective and truth log probability, not task accuracy at this
short horizon.

## Interface And Cost

Six raw width-expansion completions included illegal `move-WEST` at the left edge. Each
was rejected and made legal by the preregistered single validation-feedback retry;
there were zero terminal candidate-cell failures and no programmatic substitution.
The run made 2,690 provider requests (2,684 valid physical cells; 2,774 logical cells),
used 1,004,155 prompt and 49,734 completion tokens, zero reasoning tokens, and cost
`$0.11281430`, below the `$0.18` projection and `$0.35` hard cap.

## Boundary

The exact transition, likelihood, posterior, EIG, actions, and decode are all
programmatic. The non-thinking LLM only chooses three legal candidates per cell. This
isolates the claimed depth-over-width effect from posterior or simulator noise, but it
is one external paper-defined map. The next pre-spend gate is an independently seeded
Figure 4 `5-7` Rock Diagnosis map replication before any final consolidation.
