# iCRAFT Stronger-Generator Retry: Final Report

Date: 2026-07-14

## Registered Scope

This was the one apparatus retry authorized after the banked animals re-analysis.
It used `openai/gpt-5.4` with reasoning effort `none` to author fixed counterfactual
profile narratives only. All profile validation, prior scoring, candidate generation,
likelihoods, FactSelect interaction, and policy roles remained non-thinking
`google/gemma-4-26b-a4b-it`. The fixed calibration partition, prompt text, support
size, bounded repairs, seed, likelihood construction, and pass thresholds were not
changed.

## Result

**Failed at the calibration availability gate.** GPT-5.4 successfully authored the
fixed support, so the run completed every calibration calculation. The grouped prior
was within both registered quality limits:

| Metric | Result | Threshold |
|---|---:|---:|
| Mean diagnosis log loss | 0.56755 | < 1.38629 |
| Mean diagnosis Brier | 0.28125 | < 0.75 |
| Available realized outcomes | 14 / 48 | >= 24 |

The final row fails, and is terminal. Available observations had positive mean
truth-log gain (`0.05214`) and 64.29% favored the true label, but the official static
record channel did not supply enough such observations to validate the likelihood.

The raw artifact also contains negative pre-fix EIG correlations with realized entropy
(`-0.04131`) and truth-log gain (`-0.06777`). These are not used as a second conclusion:
an offline audit identified an unrelated posterior-floor smoothing defect that made
low-mass profile particles move after a neutral unavailable outcome. A regression test
now protects exact neutrality. The availability result is raw FactSelect behavior and
independently closes the gate, so no rerun is justified or authorized.

## Accounting And Closure

The retry used 1,000 requests, 321,398 prompt tokens, 57,925 completion tokens, no
reasoning tokens, and no forced exits. It cost `$0.44032225` of the `$2` cap:
`$0.39681000` for 55 GPT-5.4 authoring calls and `$0.04351225` for 945 Gemma 26B calls.

No third attempt, prompt tuning, likelihood retry, structural test, ranking-fidelity
test, policy comparison, or depth experiment is authorized. The external non-myopic
claim is permanently closed under the project protocol.

Authoritative raw artifact:
`STRONGER_GENERATOR_RETRY_CALIBRATION.json`.
