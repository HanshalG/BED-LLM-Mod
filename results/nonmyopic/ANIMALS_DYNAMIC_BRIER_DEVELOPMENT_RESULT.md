# Animals Dynamic-Brier Development Result

Status: **development gate failed; no fresh holdout**.

## Result

The 20-state seed-24278 development probe completed with 60 candidate rows.
Dynamic-Brier selection underperformed immediate EIG:

| Selector | Selected expected truth coverage |
| --- | ---: |
| Immediate EIG | .108117 |
| Dynamic Brier | .087533 |

The paired gain was `-.020583` with `0/19/1` wins/ties/losses. Candidate-level
Spearman association with expected truth coverage was `-.164591` for
dynamic-Brier gain versus `+.123267` for immediate EIG. The frozen development
requirements for positive association, positive mean gain, and more wins than
losses all failed, so no fresh holdout is authorized.

## Mechanism

Dynamic Brier changed EIG's selected candidate in 7 of 20 states. Six changes
had zero truth coverage under either choice. The sole consequential change was
the Echidna state:

- the hidden truth was absent from the current 12-hypothesis support;
- immediate EIG selected `Is it found in Australia?`, with expected truth
  coverage `.411667`;
- Dynamic Brier selected `Does it live in the water?`, with expected truth
  coverage `0`.

This is a structural failure, not merely estimator noise. The pseudo-truth
expectation ranges only over current hypotheses. It can reward calibrated
retention of that closed support, but cannot assign value to a question that
causes the LLM generator to recover an omitted truth. Only 2 of 20 targets were
present before the counterfactual branches, so open-world recovery is the
dominant problem.

The instrument remains useful as a diagnostic of approximate belief updates,
but it is not a suitable selector for the headline open-world claim.

## Serving And Cost

- 10,084 requests.
- 1,478,356 prompt + 142,455 completion tokens.
- Zero reasoning tokens.
- Cost: `$0.18178257`.
- Project spend after development: `$41.82505798` of `$110`.

Artifacts are in
`results/nonmyopic/animals_dynamic_brier_development/gemma26b_seed24278/`.
