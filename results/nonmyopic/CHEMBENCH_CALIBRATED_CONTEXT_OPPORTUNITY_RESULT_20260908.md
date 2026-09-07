# Calibrated-context opportunity: complete null

The prospectively frozen four-context audit from pushed commit `94e630d3`
completed all eight context/order evaluations in 325.744 seconds. No hidden
worlds were constructed, no model calls were made, and cost was $0.

Result: `chembench_calibrated_context_opportunity/20260908-v1/RESULT.json`
SHA256: `92777abeef3b9b07c2c935a29a5052178a6912345de88a16b0fac69a1ab9d837`.
Protocol SHA256: `3382362c5079baba0f6316373c1da7c061b44a65f6c23d7812c21916dec8fb2f`.

## Same-budget results

All policies receive the same calibration observation, then three new
measurements. Values are expected terminal target MSE under the public finite
prior, averaged equally over four frozen predictive-quartile contexts.

| Integration order | h1 | h2 | h3 | h1 to h2 gain | h2 to h3 gain |
|---|---:|---:|---:|---:|---:|
| 32 | 0.004069895 | 0.004032994 | 0.004023138 | 0.907% | 0.244% |
| 64 | 0.004084321 | 0.004056233 | 0.004043283 | 0.688% | 0.319% |

All four contexts at order 64, without outcome-based selection:

| Calibration quantile | h1 | h2 | h3 | First actions h1/h2/h3 |
|---|---:|---:|---:|---|
| 0.125 | 0.001829536 | 0.001828550 | 0.001822232 | 1/1/2 |
| 0.375 | 0.004662597 | 0.004604140 | 0.004604140 | 1/2/2 |
| 0.625 | 0.004935254 | 0.004906567 | 0.004898587 | 1/1/2 |
| 0.875 | 0.004909897 | 0.004885674 | 0.004848173 | 1/1/2 |

The maximum context-level refinement difference is 0.0000472841, below the
frozen absolute tolerance of 0.001. Completeness and refinement pass; the
requirement for at least 5% successive aggregate gains at both orders fails.
Root h3 actions change with integration order in contexts 0 and 2. Therefore
passing the absolute-risk tolerance is not proof of stable fine-grained action
ranking or a certified positive effect at this small scale.

## Decision and interpretation

Status is `conditional_opportunity_null`, not a runtime failure. Close this
exact calibrated-context proposal as prescribed. Do not sweep calibration
designs, quantiles, noise, targets, or seeds to obtain a more favorable curve.
This is not a population efficacy estimate, an equivalence result, or evidence
about LLM proposal quality. Four selected predictive quartiles only provide the
declared coarse public-prior diagnostic.

Together with the unconditional same-budget diagnostic (approximately 0.5%
h3-over-h1 expected gain), this does not support promoting the earlier eight-world
46% empirical improvement into a robust planning-benefit claim. That pilot and
its original engineering pass remain unchanged, with their sampling,
misspecification, and observation-order caveats.

The numerical instrument is now working; this particular finite-prior assay
formulation lacks the required planning headroom. No LLM gate, paid call, fresh
hidden panel, or powered confirmation is authorized by this result. The overall
LLM-native research goal remains incomplete. Any next formulation must have a
separate prospective scientific rationale and tests, not be a parameter rescue
of this closed audit. Useful real-history executable model proposals versus
history-blind and productive symbolic search remain an untested dependency.
